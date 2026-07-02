"""
Metadata builder — enriches text chunks and visual elements with
paper-level information from the manifest.

Every field in the returned payloads is **required** and populated
(no ``None`` values ever).

Usage:
    from app.ingestion.metadata_extractor import MetadataExtractor
    extractor = MetadataExtractor()
    payload = extractor.build_text_payload(chunk, manifest_entry, total_chunks, "paper.pdf")
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models.schemas import (
    ChunkPayload,
    PaperManifestEntry,
    TextChunk,
    VisualElement,
    VisualPayload,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MetadataExtractor:
    """Merges chunk / visual data with paper-level manifest metadata."""

    # -----------------------------------------------------------------
    # Text chunks → ChunkPayload
    # -----------------------------------------------------------------
    def build_text_payload(
        self,
        chunk: TextChunk,
        manifest: PaperManifestEntry,
        total_chunks: int,
        source_file: str,
    ) -> ChunkPayload:
        """Build a complete Qdrant payload for a text chunk.

        Parameters
        ----------
        chunk : TextChunk
            The parent or child chunk.
        manifest : PaperManifestEntry
            Paper-level metadata from ``manifest.json``.
        total_chunks : int
            Total number of text chunks (parents + children) for this paper.
        source_file : str
            Basename of the PDF file.
        """
        return ChunkPayload(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            title=manifest.title,
            category=manifest.category,
            matched_keyword=manifest.matched_keyword,
            chunk_type=chunk.chunk_type,
            parent_chunk_id=chunk.parent_chunk_id,
            child_chunk_ids=chunk.child_chunk_ids,
            page_number=chunk.page_number,
            section_header=chunk.section_header,
            chunk_index=chunk.chunk_index,
            total_chunks=total_chunks,
            char_count=chunk.char_count,
            token_count=chunk.token_count,
            text=chunk.text,
            source_file=source_file,
            ingested_at=_utc_now_iso(),
        )

    # -----------------------------------------------------------------
    # Visual elements → VisualPayload
    # -----------------------------------------------------------------
    def build_visual_payload(
        self,
        visual: VisualElement,
        manifest: PaperManifestEntry,
        source_file: str,
    ) -> VisualPayload:
        """Build a complete Qdrant payload for a visual element.

        Parameters
        ----------
        visual : VisualElement
            The extracted figure / table / full-page image.
        manifest : PaperManifestEntry
            Paper-level metadata from ``manifest.json``.
        source_file : str
            Basename of the PDF file.
        """
        return VisualPayload(
            visual_id=visual.visual_id,
            paper_id=visual.paper_id,
            title=manifest.title,
            category=manifest.category,
            matched_keyword=manifest.matched_keyword,
            visual_type=visual.visual_type,
            page_number=visual.page_number,
            bbox=visual.bbox,
            caption=visual.caption,
            ocr_text=visual.ocr_text,
            source_file=source_file,
            image_base64=visual.image_base64,
            ingested_at=_utc_now_iso(),
        )
