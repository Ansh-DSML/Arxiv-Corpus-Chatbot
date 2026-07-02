"""
Post-ingestion deduplication for text chunks.

Scrolls through the ``core_papers_text`` Qdrant collection, groups chunks
by ``paper_id``, and removes near-duplicates whose cosine similarity
exceeds a configurable threshold (default 0.90).

Only **text** chunks are deduped — visual elements are left untouched.

Usage:
    from app.ingestion.dedup import TextDeduplicator
    dedup = TextDeduplicator(qdrant_client, "core_papers_text", threshold=0.90)
    stats = dedup.deduplicate()
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextDeduplicator:
    """Removes near-duplicate text chunks within each paper."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        threshold: float = 0.90,
    ) -> None:
        self._client = qdrant_client
        self._collection = collection_name
        self._threshold = threshold

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------

    def deduplicate(self) -> dict:
        """Run deduplication. Returns stats dict.

        Returns
        -------
        dict
            ``total_checked``, ``duplicates_removed``, ``papers_processed``
        """
        logger.info(
            "Starting deduplication on '{col}' (threshold={t})",
            col=self._collection,
            t=self._threshold,
        )

        # --- Step 1: scroll ALL points with dense vectors ---------------
        all_points = self._scroll_all()
        if not all_points:
            logger.info("No points found in collection — nothing to deduplicate")
            return {"total_checked": 0, "duplicates_removed": 0, "papers_processed": 0}

        logger.info("Loaded {n} points for deduplication", n=len(all_points))

        # --- Step 2: group by paper_id ----------------------------------
        paper_groups: dict[str, list] = defaultdict(list)
        for point in all_points:
            paper_id = point.payload.get("paper_id", "unknown")
            paper_groups[paper_id].append(point)

        # --- Step 3: per-paper pairwise cosine comparison ---------------
        ids_to_delete: list[str] = []
        papers_processed = 0

        for paper_id, points in paper_groups.items():
            if len(points) < 2:
                papers_processed += 1
                continue

            # Extract dense vectors and chunk_index
            vectors = []
            point_info = []  # (point_id, chunk_index)
            for pt in points:
                vec = pt.vector
                if isinstance(vec, dict):
                    vec = vec.get("dense", [])
                if not vec:
                    continue
                vectors.append(vec)
                chunk_index = pt.payload.get("chunk_index", 0)
                point_info.append((pt.id, chunk_index))

            if len(vectors) < 2:
                papers_processed += 1
                continue

            mat = np.array(vectors, dtype=np.float32)

            # Normalise rows (should already be normalised, but be safe)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            mat = mat / norms

            # Pairwise cosine similarity
            sim_matrix = mat @ mat.T

            # Find pairs above threshold
            seen_deleted: set[int] = set()
            for i in range(len(vectors)):
                if i in seen_deleted:
                    continue
                for j in range(i + 1, len(vectors)):
                    if j in seen_deleted:
                        continue
                    if sim_matrix[i, j] >= self._threshold:
                        # Keep the one with lower chunk_index (earlier in paper)
                        idx_i = point_info[i][1]
                        idx_j = point_info[j][1]
                        if idx_i <= idx_j:
                            victim_idx = j
                        else:
                            victim_idx = i

                        ids_to_delete.append(point_info[victim_idx][0])
                        seen_deleted.add(victim_idx)

            papers_processed += 1
            if papers_processed % 100 == 0:
                logger.info(
                    "Dedup progress: {n}/{t} papers, {d} duplicates found so far",
                    n=papers_processed,
                    t=len(paper_groups),
                    d=len(ids_to_delete),
                )

        # --- Step 4: batch delete duplicates ----------------------------
        if ids_to_delete:
            # Delete in batches of 500
            batch_size = 500
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i : i + batch_size]
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=PointIdsList(points=batch),
                )
            logger.info(
                "Deleted {n} duplicate chunks from '{col}'",
                n=len(ids_to_delete),
                col=self._collection,
            )
        else:
            logger.info("No duplicates found above threshold {t}", t=self._threshold)

        stats = {
            "total_checked": len(all_points),
            "duplicates_removed": len(ids_to_delete),
            "papers_processed": papers_processed,
        }
        logger.info("Deduplication complete: {s}", s=stats)
        return stats

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _scroll_all(self) -> list:
        """Scroll through entire collection, fetching dense vectors."""
        all_points = []
        offset = None

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=["dense"],
            )
            all_points.extend(results)

            if next_offset is None:
                break
            offset = next_offset

        return all_points
