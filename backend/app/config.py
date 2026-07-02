"""
Application configuration loaded from environment variables via pydantic-settings.

Reads from two .env locations (project-root takes precedence):
  1. backend/.env
  2. project-root/.env

Usage:
    from app.config import settings
    print(settings.QDRANT_URL)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Resolve .env paths relative to THIS file's location on disk.
# config.py lives at  backend/app/config.py
#   → parent.parent   = backend/
#   → parent³         = project-root/
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Centralised, type-safe application settings."""

    model_config = SettingsConfigDict(
        env_file=(
            str(_BACKEND_DIR / ".env"),
            str(_PROJECT_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,           # trims spaces around .env values
    )

    # ── Application ─────────────────────────────────────────
    ENVIRONMENT: str = "development"
    APP_NAME: str = "rag-core-papers"
    LOG_LEVEL: str = "INFO"
    API_KEY_SECRET: str = ""
    RATE_LIMIT_PER_MINUTE: int = 30

    # ── LLM: Groq ──────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # ── Reranking: Cohere ───────────────────────────────────
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # ── Vector Store: Qdrant Cloud ──────────────────────────
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_TEXT_COLLECTION: str = "core_papers_text"
    QDRANT_VISUAL_COLLECTION: str = "core_papers_visual"

    # ── Cache: Redis ────────────────────────────────────────
    REDIS_URL: str = ""
    SEMANTIC_CACHE_THRESHOLD: float = 0.95
    SEMANTIC_CACHE_TTL_SECONDS: int = 86400

    # ── Embeddings ──────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    COLPALI_MODEL: str = "vidore/colpali-v1.2"
    HF_TOKEN: str = ""

    # ── Evaluation & Tracing ────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "rag-core-papers"

    # ── Chunking ────────────────────────────────────────────
    CHILD_CHUNK_SIZE: int = 256       # tokens per child chunk
    PARENT_CHUNK_SIZE: int = 1024     # tokens per parent chunk
    CHUNK_OVERLAP: int = 50           # token overlap between consecutive child chunks
    DEDUP_SIMILARITY_THRESHOLD: float = 0.90

    # ── Paper Sourcing ──────────────────────────────────────
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    # ── AWS (optional S3 storage) ───────────────────────────
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str = ""

    # ── Ingestion Tuning (CPU-optimised defaults) ───────────
    TEXT_EMBED_BATCH_SIZE: int = 32    # texts per BGE encode call
    VISUAL_EMBED_BATCH_SIZE: int = 1  # images per ColPali call (CPU)
    QDRANT_UPLOAD_BATCH_SIZE: int = 64
    PDF_DPI: int = 300                # resolution for pdf2image rasterisation

    # ── Derived paths ───────────────────────────────────────
    @property
    def BACKEND_DIR(self) -> Path:
        """Root of the backend/ directory."""
        return _BACKEND_DIR

    @property
    def DATA_DIR(self) -> Path:
        return _BACKEND_DIR / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def CHECKPOINT_PATH(self) -> Path:
        return self.PROCESSED_DIR / "checkpoint.json"

    @property
    def MANIFEST_PATH(self) -> Path:
        return self.DATA_DIR / "manifest.json"

    @property
    def BM25_INDEX_PATH(self) -> Path:
        return self.PROCESSED_DIR / "bm25_index.pkl"

    @property
    def CATEGORIES(self) -> list[str]:
        """The four paper-category folder names inside data/raw/."""
        return ["ai", "dl", "ml", "neural_networks"]


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere.
# ---------------------------------------------------------------------------
settings = Settings()
