"""
Structured logging using **loguru**.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing paper {paper_id}", paper_id="2311.12424")

Handlers
--------
* **stderr** — coloured, human-readable, level controlled by ``LOG_LEVEL``.
* **File**   — ``data/processed/ingestion.log``, rotated at 50 MB, 7-day retention.
"""

import sys
from pathlib import Path

from loguru import logger as _root_logger


# ---------------------------------------------------------------------------
# Remove the default loguru handler so we control formatting.
# ---------------------------------------------------------------------------
_root_logger.remove()

# ---------------------------------------------------------------------------
# Console handler
# ---------------------------------------------------------------------------
_root_logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=False,
)

# ---------------------------------------------------------------------------
# File handler — stored next to processed data
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_root_logger.add(
    str(_LOG_DIR / "ingestion.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]} | {message}",
    rotation="50 MB",
    retention="7 days",
    compression="gz",
    backtrace=True,
    diagnose=True,
    enqueue=True,           # thread-safe writes
)


def get_logger(name: str):
    """Return a loguru logger bound with *module=name* context.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    """
    return _root_logger.bind(module=name)
