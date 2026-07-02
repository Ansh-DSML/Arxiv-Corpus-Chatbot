"""
Pydantic data models for the ingestion pipeline.

Design rules (per user requirement):
  • **Every** field is required — no ``Optional`` types.
  • Where a value may not exist, use ``""`` (empty string),
    ``[]`` (empty list), or ``0`` — **never** ``None``.
"""

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════════════════════════════════════

class PaperManifestEntry(BaseModel):
    """One row from ``manifest.json`` — paper-level metadata from arXiv."""

    arxiv_id: str
    title: str
    category: str              # e.g. "ai", "dl", "ml", "neural_networks"
    status: str                # "downloaded" | "skipped_applied"
    pdf_path: str              # relative path to PDF (always populated for downloaded papers)
    matched_keyword: str       # the keyword that selected this paper
    attempts: int              # download attempt count


# ═══════════════════════════════════════════════════════════════════════════
# PDF Parsing
# ═══════════════════════════════════════════════════════════════════════════

class ParsedPage(BaseModel):
    """Output of PyMuPDF text extraction for a single PDF page."""

    page_number: int           # 1-indexed
    text: str                  # full text content of the page
    section_headers: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Text Chunks
# ═══════════════════════════════════════════════════════════════════════════

class TextChunk(BaseModel):
    """A parent or child text chunk produced by the parent-child chunker."""

    chunk_id: str              # deterministic UUID5
    paper_id: str              # arXiv ID
    chunk_type: str            # "parent" | "child"
    text: str                  # chunk text content
    parent_chunk_id: str       # parent UUID; "" for parent-type chunks
    child_chunk_ids: list[str] = Field(default_factory=list)
    page_number: int           # 1-indexed source page
    section_header: str        # nearest heading; "" if none detected
    chunk_index: int           # sequential index within the paper
    token_count: int           # cl100k_base token count
    char_count: int            # len(text)


# ═══════════════════════════════════════════════════════════════════════════
# Visual Elements
# ═══════════════════════════════════════════════════════════════════════════

class VisualElement(BaseModel):
    """An extracted figure, table, or full-page image from a PDF."""

    visual_id: str             # deterministic UUID5
    paper_id: str              # arXiv ID
    visual_type: str           # "figure" | "table" | "full_page"
    page_number: int           # 1-indexed
    bbox: list[float]          # [x0, y0, x1, y1]; [0,0,0,0] for full_page
    caption: str               # detected caption text; "" if none
    ocr_text: str              # text extracted from the visual; "" if none
    image_base64: str          # base64-encoded PNG


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant Payloads  (what is stored as payload alongside the vectors)
# ═══════════════════════════════════════════════════════════════════════════

class ChunkPayload(BaseModel):
    """Complete Qdrant point payload for ``core_papers_text``."""

    chunk_id: str
    paper_id: str
    title: str
    category: str
    matched_keyword: str
    chunk_type: str            # "parent" | "child"
    parent_chunk_id: str
    child_chunk_ids: list[str] = Field(default_factory=list)
    page_number: int
    section_header: str
    chunk_index: int
    total_chunks: int          # total text chunks for this paper
    char_count: int
    token_count: int
    text: str                  # full chunk text — used for BM25 sparse search
    source_file: str           # PDF filename (basename)
    ingested_at: str           # ISO-8601 timestamp


class VisualPayload(BaseModel):
    """Complete Qdrant point payload for ``core_papers_visual``."""

    visual_id: str
    paper_id: str
    title: str
    category: str
    matched_keyword: str
    visual_type: str           # "figure" | "table" | "full_page"
    page_number: int
    bbox: list[float]
    caption: str
    ocr_text: str
    source_file: str
    image_base64: str          # kept in payload so UI can render it
    ingested_at: str           # ISO-8601 timestamp


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint (resume support)
# ═══════════════════════════════════════════════════════════════════════════

class IngestionProgress(BaseModel):
    """Checkpoint state persisted to ``data/processed/checkpoint.json``."""

    completed_papers: list[str] = Field(default_factory=list)
    failed_papers: dict[str, str] = Field(default_factory=dict)
    total_text_chunks: int = 0
    total_visual_elements: int = 0
    last_updated: str = ""     # ISO-8601 timestamp
