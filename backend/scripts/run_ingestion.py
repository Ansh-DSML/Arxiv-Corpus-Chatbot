"""
CLI entry point for the ingestion pipeline.

Run from the project root or backend directory::

    python backend/scripts/run_ingestion.py
    python backend/scripts/run_ingestion.py --categories ai dl --limit 5
    python backend/scripts/run_ingestion.py --resume --skip-visual
    python backend/scripts/run_ingestion.py --recreate-collections
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path so `from app.xxx import ...` works
# regardless of the working directory.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("run_ingestion")

ALL_CATEGORIES = ["ai", "dl", "ml", "neural_networks"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest arXiv research papers into Qdrant (text + visual).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full ingestion with resume
  python scripts/run_ingestion.py

  # Only process AI and DL categories, first 10 papers
  python scripts/run_ingestion.py --categories ai dl --limit 10

  # Skip visual pipeline (text only)
  python scripts/run_ingestion.py --skip-visual

  # Recreate collections from scratch (deletes existing data!)
  python scripts/run_ingestion.py --recreate-collections --no-resume
""",
    )

    p.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
        help="Category folders to process (default: all four).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        dest="resume",
        help="Resume from last checkpoint (default: True).",
    )
    p.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start from scratch, ignoring any existing checkpoint.",
    )
    p.add_argument(
        "--skip-visual",
        action="store_true",
        default=False,
        help="Skip visual extraction and ColPali embedding.",
    )
    p.add_argument(
        "--skip-dedup",
        action="store_true",
        default=False,
        help="Skip post-ingestion deduplication.",
    )
    p.add_argument(
        "--recreate-collections",
        action="store_true",
        default=False,
        help="Delete and recreate Qdrant collections before ingesting.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N papers (for testing).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override TEXT_EMBED_BATCH_SIZE from config.",
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Override batch size if specified
    if args.batch_size is not None:
        settings.TEXT_EMBED_BATCH_SIZE = args.batch_size

    # --- Print configuration summary ---
    print()
    print("=" * 62)
    print("  RAG Core Papers — Ingestion Pipeline")
    print("=" * 62)
    print(f"  Qdrant URL        : {settings.QDRANT_URL[:60]}...")
    print(f"  Text collection   : {settings.QDRANT_TEXT_COLLECTION}")
    print(f"  Visual collection : {settings.QDRANT_VISUAL_COLLECTION}")
    print(f"  Embedding model   : {settings.EMBEDDING_MODEL}")
    print(f"  ColPali model     : {settings.COLPALI_MODEL}")
    print(f"  Parent chunk size : {settings.PARENT_CHUNK_SIZE} tokens")
    print(f"  Child chunk size  : {settings.CHILD_CHUNK_SIZE} tokens")
    print(f"  Chunk overlap     : {settings.CHUNK_OVERLAP} tokens")
    print(f"  Dedup threshold   : {settings.DEDUP_SIMILARITY_THRESHOLD}")
    print(f"  Categories        : {args.categories}")
    print(f"  Resume            : {args.resume}")
    print(f"  Skip visual       : {args.skip_visual}")
    print(f"  Skip dedup        : {args.skip_dedup}")
    print(f"  Limit             : {args.limit or 'None (all papers)'}")
    print(f"  Batch size        : {settings.TEXT_EMBED_BATCH_SIZE}")
    print("=" * 62)
    print()

    # --- Run pipeline ---
    from app.ingestion.ingest_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()

    if args.recreate_collections:
        logger.warning("Recreating Qdrant collections (existing data will be deleted)...")
        pipeline.ensure_collections(recreate=True)

    try:
        stats = pipeline.run(
            categories=args.categories,
            resume=args.resume,
            skip_visual=args.skip_visual,
            skip_dedup=args.skip_dedup,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint has been saved.")
        print("Run again with --resume to continue from where you left off.")
        sys.exit(1)

    # --- Print final report ---
    print()
    print("=" * 62)
    print("  INGESTION REPORT")
    print("=" * 62)
    for key, val in stats.items():
        if key == "dedup_stats" and isinstance(val, dict):
            print(f"  Deduplication:")
            for dk, dv in val.items():
                print(f"    {dk:25s}: {dv}")
        else:
            print(f"  {key:25s}: {val}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
