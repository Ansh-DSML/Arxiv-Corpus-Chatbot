"""
Build a local BM25 index from the Qdrant text collection.

This creates a ``rank_bm25.BM25Okapi`` index serialised to pickle for
use by the LlamaIndex BM25 retriever as a secondary / offline search
mechanism.

The **primary** hybrid search uses Qdrant's native sparse vectors
(populated during ingestion); this script builds a **backup** index.

Usage::

    python backend/scripts/build_bm25_index.py
    python backend/scripts/build_bm25_index.py --output custom_path.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("build_bm25_index")


def _simple_tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric. Remove short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]  # drop single chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from Qdrant text collection.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle path (default: config BM25_INDEX_PATH).",
    )
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else settings.BM25_INDEX_PATH

    t0 = time.time()

    # --- Connect to Qdrant ---
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=120,
    )

    collection = settings.QDRANT_TEXT_COLLECTION
    if not client.collection_exists(collection):
        logger.error("Collection '{c}' does not exist. Run ingestion first.", c=collection)
        sys.exit(1)

    # --- Scroll ALL points ---
    logger.info("Scrolling all points from '{c}'...", c=collection)
    all_points = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=["chunk_id", "text", "chunk_type"],
            with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset

        if len(all_points) % 5000 == 0:
            logger.info("  ... scrolled {n} points so far", n=len(all_points))

    logger.info("Total points loaded: {n}", n=len(all_points))

    if not all_points:
        logger.warning("No points found — aborting BM25 build.")
        sys.exit(0)

    # --- Build tokenized corpus ---
    chunk_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []

    for pt in all_points:
        text = pt.payload.get("text", "")
        cid = pt.payload.get("chunk_id", str(pt.id))
        tokens = _simple_tokenize(text)
        if tokens:
            chunk_ids.append(cid)
            tokenized_corpus.append(tokens)

    logger.info(
        "Tokenized {n} documents (avg {avg:.0f} tokens/doc)",
        n=len(tokenized_corpus),
        avg=sum(len(t) for t in tokenized_corpus) / max(len(tokenized_corpus), 1),
    )

    # --- Build BM25 index ---
    from rank_bm25 import BM25Okapi

    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Compute vocabulary size
    vocab: set[str] = set()
    for tokens in tokenized_corpus:
        vocab.update(tokens)

    logger.info("BM25 index built — vocab size: {v}", v=len(vocab))

    # --- Save to pickle ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "corpus": tokenized_corpus,
        "vocab_size": len(vocab),
        "num_documents": len(chunk_ids),
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        "BM25 index saved to {p} ({sz:.1f} MB, {t:.1f}s)",
        p=str(output_path),
        sz=file_size_mb,
        t=elapsed,
    )
    print(f"\n✓ BM25 index: {len(chunk_ids)} docs, {len(vocab)} vocab, {file_size_mb:.1f} MB → {output_path}")


if __name__ == "__main__":
    main()
