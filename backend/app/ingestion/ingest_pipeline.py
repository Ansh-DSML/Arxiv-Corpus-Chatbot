"""
Main ingestion orchestrator.

Lazily iterates over papers in each category folder, runs the full
text + visual pipeline for each paper **sequentially**, and checkpoints
after every paper so the run can be resumed after a crash.

Usage:
    from app.ingestion.ingest_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    stats = pipeline.run(resume=True)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.ingestion.chunking import ParentChildChunker
from app.ingestion.dedup import TextDeduplicator
from app.ingestion.embedder import TextEmbedder, VisualEmbedder
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.pdf_parser import PDFTextParser
from app.ingestion.visual_pipeline import VisualExtractor
from app.models.schemas import IngestionProgress, PaperManifestEntry
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end pipeline: parse → chunk → embed → store → dedup."""

    def __init__(self) -> None:
        # --- Qdrant client ---
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=180,
        )

        # --- Pipeline components (all lazy-loaded internally) ---
        self._parser = PDFTextParser()
        self._chunker = ParentChildChunker(
            parent_size=settings.PARENT_CHUNK_SIZE,
            child_size=settings.CHILD_CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        self._metadata = MetadataExtractor()
        self._text_embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL)
        self._visual_embedder = VisualEmbedder(model_name=settings.COLPALI_MODEL)
        self._visual_extractor = VisualExtractor()

    # =================================================================
    # Collection management
    # =================================================================

    def ensure_collections(self, recreate: bool = False) -> None:
        """Create Qdrant collections if they don't exist.

        Parameters
        ----------
        recreate : bool
            If True, delete existing collections first and recreate.
        """
        self._ensure_text_collection(recreate)
        self._ensure_visual_collection(recreate)

    def _ensure_text_collection(self, recreate: bool) -> None:
        name = settings.QDRANT_TEXT_COLLECTION
        exists = self._client.collection_exists(name)

        if exists and recreate:
            logger.warning("Deleting existing text collection: {n}", n=name)
            self._client.delete_collection(name)
            exists = False

        if not exists:
            logger.info("Creating text collection: {n}", n=name)
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,                   # BGE-large-en-v1.5
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=Modifier.IDF,       # Qdrant applies IDF at query time
                    ),
                },
            )
            # Payload indices for filtering
            self._client.create_payload_index(name, "paper_id", PayloadSchemaType.KEYWORD)
            self._client.create_payload_index(name, "category", PayloadSchemaType.KEYWORD)
            self._client.create_payload_index(name, "chunk_type", PayloadSchemaType.KEYWORD)
            logger.info("Text collection '{n}' created with dense + sparse vectors", n=name)
        else:
            logger.info("Text collection '{n}' already exists", n=name)

    def _ensure_visual_collection(self, recreate: bool) -> None:
        name = settings.QDRANT_VISUAL_COLLECTION
        exists = self._client.collection_exists(name)

        if exists and recreate:
            logger.warning("Deleting existing visual collection: {n}", n=name)
            self._client.delete_collection(name)
            exists = False

        if not exists:
            logger.info("Creating visual collection: {n}", n=name)
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "colpali": VectorParams(
                        size=128,                    # ColPali patch dimension
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM,
                        ),
                    ),
                },
            )
            self._client.create_payload_index(name, "paper_id", PayloadSchemaType.KEYWORD)
            self._client.create_payload_index(name, "category", PayloadSchemaType.KEYWORD)
            self._client.create_payload_index(name, "visual_type", PayloadSchemaType.KEYWORD)
            logger.info("Visual collection '{n}' created with ColPali multi-vectors", n=name)
        else:
            logger.info("Visual collection '{n}' already exists", n=name)

    # =================================================================
    # Manifest loading
    # =================================================================

    def _load_manifest(self) -> dict[str, PaperManifestEntry]:
        """Load and filter manifest.json — only downloaded papers."""
        manifest_path = settings.MANIFEST_PATH
        logger.info("Loading manifest from {p}", p=str(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        entries: dict[str, PaperManifestEntry] = {}
        for arxiv_id, data in raw.items():
            if data.get("status") != "downloaded":
                continue
            if not data.get("pdf_path"):
                continue

            # Resolve relative pdf_path to absolute
            # manifest paths use: ..\data\raw\<category>\<file>.pdf
            rel_path = data["pdf_path"].replace("\\\\", "/").replace("\\", "/")
            abs_path = (manifest_path.parent / rel_path).resolve()

            entries[arxiv_id] = PaperManifestEntry(
                arxiv_id=data["arxiv_id"],
                title=data["title"],
                category=data["category"],
                status=data["status"],
                pdf_path=str(abs_path),
                matched_keyword=data["matched_keyword"],
                attempts=data["attempts"],
            )

        logger.info("Manifest loaded: {n} downloadable papers", n=len(entries))
        return entries

    # =================================================================
    # Checkpoint management
    # =================================================================

    def _load_checkpoint(self) -> IngestionProgress:
        """Load checkpoint from disk, or return a fresh one."""
        cp = settings.CHECKPOINT_PATH
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                progress = IngestionProgress(**data)
                logger.info(
                    "Resumed checkpoint: {n} papers already completed",
                    n=len(progress.completed_papers),
                )
                return progress
            except Exception as exc:
                logger.warning("Corrupt checkpoint, starting fresh: {e}", e=str(exc))
        return IngestionProgress()

    def _save_checkpoint(self, progress: IngestionProgress) -> None:
        """Persist checkpoint to disk."""
        from datetime import datetime, timezone
        progress.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(settings.CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(progress.model_dump(), f, indent=2)

    # =================================================================
    # Lazy paper iterator
    # =================================================================

    def _iter_papers(
        self,
        categories: list[str],
        manifest: dict[str, PaperManifestEntry],
        checkpoint: IngestionProgress,
        limit: int | None = None,
    ):
        """Yield ``(pdf_path, manifest_entry)`` lazily, skipping completed papers."""
        yielded = 0
        for category in categories:
            cat_dir = settings.RAW_DIR / category
            if not cat_dir.exists():
                logger.warning("Category directory not found: {d}", d=str(cat_dir))
                continue

            for pdf_file in sorted(cat_dir.glob("*.pdf")):
                if limit is not None and yielded >= limit:
                    return

                # Extract arxiv_id from filename (prefix before first underscore)
                arxiv_id = pdf_file.stem.split("_")[0]

                if arxiv_id in checkpoint.completed_papers:
                    continue
                if arxiv_id in checkpoint.failed_papers:
                    continue
                if arxiv_id not in manifest:
                    # Try matching by scanning manifest for this file
                    found = False
                    for mid, me in manifest.items():
                        if Path(me.pdf_path).name == pdf_file.name:
                            arxiv_id = mid
                            found = True
                            break
                    if not found:
                        continue

                yield pdf_file, manifest[arxiv_id]
                yielded += 1

    # =================================================================
    # Main run method
    # =================================================================

    def run(
        self,
        categories: list[str] | None = None,
        resume: bool = True,
        skip_visual: bool = False,
        skip_dedup: bool = False,
        limit: int | None = None,
    ) -> dict:
        """Run the ingestion pipeline.

        Parameters
        ----------
        categories : list[str] | None
            Category folders to process. Defaults to all 4.
        resume : bool
            If True, skip papers already in the checkpoint.
        skip_visual : bool
            If True, skip visual extraction + ColPali embedding.
        skip_dedup : bool
            If True, skip post-ingestion deduplication.
        limit : int | None
            Process at most *limit* papers (for testing).

        Returns
        -------
        dict
            Summary statistics.
        """
        t0 = time.time()
        cats = categories or settings.CATEGORIES

        # --- Setup ---
        self.ensure_collections(recreate=False)
        manifest = self._load_manifest()
        checkpoint = self._load_checkpoint() if resume else IngestionProgress()

        logger.info(
            "Starting ingestion: categories={cats}, resume={r}, skip_visual={sv}, limit={lim}",
            cats=cats,
            r=resume,
            sv=skip_visual,
            lim=limit,
        )

        papers_processed = 0
        papers_failed = 0

        for pdf_path, manifest_entry in self._iter_papers(cats, manifest, checkpoint, limit):
            arxiv_id = manifest_entry.arxiv_id
            paper_t0 = time.time()

            logger.info(
                "━━━ Processing [{idx}]: {title} ({pid}) ━━━",
                idx=papers_processed + len(checkpoint.completed_papers) + 1,
                title=manifest_entry.title[:80],
                pid=arxiv_id,
            )

            try:
                text_count, visual_count = self._process_single_paper(
                    pdf_path=pdf_path,
                    manifest_entry=manifest_entry,
                    skip_visual=skip_visual,
                )

                # Update checkpoint
                checkpoint.completed_papers.append(arxiv_id)
                checkpoint.total_text_chunks += text_count
                checkpoint.total_visual_elements += visual_count
                self._save_checkpoint(checkpoint)

                elapsed = time.time() - paper_t0
                papers_processed += 1
                logger.info(
                    "✓ {pid}: {tc} text chunks, {vc} visuals ({t:.1f}s)",
                    pid=arxiv_id,
                    tc=text_count,
                    vc=visual_count,
                    t=elapsed,
                )

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt — saving checkpoint and exiting")
                self._save_checkpoint(checkpoint)
                raise

            except Exception as exc:
                papers_failed += 1
                checkpoint.failed_papers[arxiv_id] = str(exc)[:500]
                self._save_checkpoint(checkpoint)
                logger.error(
                    "✗ Failed {pid}: {err}",
                    pid=arxiv_id,
                    err=str(exc),
                )

        # --- Post-ingestion dedup ---
        dedup_stats: dict = {}
        if not skip_dedup and papers_processed > 0:
            logger.info("Running post-ingestion deduplication...")
            dedup = TextDeduplicator(
                self._client,
                settings.QDRANT_TEXT_COLLECTION,
                settings.DEDUP_SIMILARITY_THRESHOLD,
            )
            dedup_stats = dedup.deduplicate()

        elapsed_total = time.time() - t0

        stats = {
            "papers_processed": papers_processed,
            "papers_failed": papers_failed,
            "papers_skipped_checkpoint": len(checkpoint.completed_papers) - papers_processed,
            "total_text_chunks": checkpoint.total_text_chunks,
            "total_visual_elements": checkpoint.total_visual_elements,
            "dedup_stats": dedup_stats,
            "elapsed_seconds": round(elapsed_total, 1),
        }

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("  Papers processed : {n}", n=papers_processed)
        logger.info("  Papers failed    : {n}", n=papers_failed)
        logger.info("  Text chunks      : {n}", n=checkpoint.total_text_chunks)
        logger.info("  Visual elements  : {n}", n=checkpoint.total_visual_elements)
        if dedup_stats:
            logger.info("  Duplicates removed: {n}", n=dedup_stats.get("duplicates_removed", 0))
        logger.info("  Total time       : {t:.0f}s ({m:.1f} min)", t=elapsed_total, m=elapsed_total / 60)
        logger.info("=" * 60)

        return stats

    # =================================================================
    # Per-paper processing
    # =================================================================

    def _process_single_paper(
        self,
        pdf_path: Path,
        manifest_entry: PaperManifestEntry,
        skip_visual: bool,
    ) -> tuple[int, int]:
        """Process one paper: text path then visual path.

        Returns ``(text_chunk_count, visual_element_count)``.
        """
        arxiv_id = manifest_entry.arxiv_id
        source_file = pdf_path.name
        abs_pdf = Path(manifest_entry.pdf_path)

        # Use the absolute path from manifest if it exists, else the raw path
        if abs_pdf.exists():
            actual_path = abs_pdf
        else:
            actual_path = pdf_path

        # ===================== TEXT PATH =====================

        # 1. Parse PDF
        pages = self._parser.parse(actual_path)
        if not pages:
            logger.warning("No text extracted from {pid}", pid=arxiv_id)
            return 0, 0

        # 2. Chunk
        parent_chunks, child_chunks = self._chunker.chunk(pages, arxiv_id)
        all_chunks = parent_chunks + child_chunks
        total_chunks = len(all_chunks)

        if total_chunks == 0:
            logger.warning("No chunks produced for {pid}", pid=arxiv_id)
            return 0, 0

        # 3. Build metadata payloads
        payloads = [
            self._metadata.build_text_payload(chunk, manifest_entry, total_chunks, source_file)
            for chunk in all_chunks
        ]

        # 4. Embed texts (dense + sparse in one forward pass via BGE-M3)
        texts = [p.text for p in payloads]
        dense_vectors, sparse_vectors = self._text_embedder.embed_batch_hybrid(
            texts, batch_size=settings.TEXT_EMBED_BATCH_SIZE
        )

        # 6. Build Qdrant points
        text_points: list[PointStruct] = []
        for i, payload in enumerate(payloads):
            point = PointStruct(
                id=payload.chunk_id,
                vector={
                    "dense": dense_vectors[i],
                    "bm25": sparse_vectors[i],
                },
                payload=payload.model_dump(),
            )
            text_points.append(point)

        # 7. Upsert in batches
        batch_size = settings.QDRANT_UPLOAD_BATCH_SIZE
        for i in range(0, len(text_points), batch_size):
            batch = text_points[i : i + batch_size]
            self._client.upsert(
                collection_name=settings.QDRANT_TEXT_COLLECTION,
                points=batch,
            )

        logger.debug("Upserted {n} text points for {pid}", n=len(text_points), pid=arxiv_id)

        # ===================== VISUAL PATH =====================

        visual_count = 0
        if not skip_visual:
            visuals = self._visual_extractor.extract(actual_path, arxiv_id)

            for vis in visuals:
                vis_payload = self._metadata.build_visual_payload(vis, manifest_entry, source_file)

                # Embed with ColPali
                multivecs = self._visual_embedder.embed_single(vis.image_base64)
                if not multivecs:
                    logger.warning(
                        "ColPali returned empty embeddings for visual {vid} in {pid}",
                        vid=vis.visual_id,
                        pid=arxiv_id,
                    )
                    continue

                point = PointStruct(
                    id=vis_payload.visual_id,
                    vector={"colpali": multivecs},
                    payload=vis_payload.model_dump(),
                )

                self._client.upsert(
                    collection_name=settings.QDRANT_VISUAL_COLLECTION,
                    points=[point],
                )
                visual_count += 1

            logger.debug("Upserted {n} visual points for {pid}", n=visual_count, pid=arxiv_id)

        return total_chunks, visual_count
