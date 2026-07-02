"""
PDF text extraction using **PyMuPDF** (``fitz``).

Extracts text page-by-page, detects section headers via font-size
heuristics, and returns a list of ``ParsedPage`` objects.

Usage:
    from app.ingestion.pdf_parser import PDFTextParser
    parser = PDFTextParser()
    pages = parser.parse(Path("paper.pdf"))
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from pathlib import Path

import fitz  # pymupdf

from app.models.schemas import ParsedPage
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Common section headings found in academic papers (case-insensitive match)
_KNOWN_HEADERS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "preliminaries",
    "problem statement",
    "problem formulation",
    "methods",
    "method",
    "methodology",
    "approach",
    "proposed method",
    "proposed approach",
    "model",
    "architecture",
    "framework",
    "system overview",
    "experiments",
    "experiment",
    "experimental setup",
    "experimental results",
    "setup",
    "evaluation",
    "results",
    "results and discussion",
    "analysis",
    "ablation study",
    "ablation studies",
    "discussion",
    "limitations",
    "future work",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "conclusions and future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
    "supplementary material",
    "supplementary materials",
}


def _normalise_text(text: str) -> str:
    """Normalise unicode, collapse excessive whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Collapse runs of whitespace (but preserve single newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header_by_pattern(text: str) -> bool:
    """Check if *text* matches a known academic section heading."""
    cleaned = text.strip().rstrip(".").lower()
    # Remove leading numbering: "1.", "1.1", "I.", "A.", "1 ", etc.
    cleaned = re.sub(r"^(?:\d+\.?\d*\.?\s*|[IVXLC]+\.?\s*|[A-Z]\.?\s+)", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned in _KNOWN_HEADERS


class PDFTextParser:
    """Extracts text and section headers from a PDF using PyMuPDF."""

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Parse a PDF and return one ``ParsedPage`` per page.

        Parameters
        ----------
        pdf_path : Path
            Absolute or relative path to the PDF file.

        Returns
        -------
        list[ParsedPage]
            One entry per page (1-indexed).  Returns an empty list
            if the PDF cannot be opened or is empty.
        """
        if not pdf_path.exists():
            logger.warning("PDF not found: {path}", path=str(pdf_path))
            return []

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error(
                "Failed to open PDF {path}: {err}",
                path=str(pdf_path),
                err=str(exc),
            )
            return []

        pages: list[ParsedPage] = []

        try:
            # ----------------------------------------------------------
            # First pass: collect all font sizes across the document
            # to compute a median for header detection.
            # ----------------------------------------------------------
            all_font_sizes: list[float] = []
            for page in doc:
                try:
                    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                    for block in blocks:
                        if block.get("type") != 0:  # 0 = text block
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                size = span.get("size", 0.0)
                                if size > 0:
                                    all_font_sizes.append(size)
                except Exception:
                    continue

            median_size = statistics.median(all_font_sizes) if all_font_sizes else 10.0
            header_threshold = median_size * 1.25  # 25 % larger → likely a heading

            # ----------------------------------------------------------
            # Second pass: extract text and detect headers per page
            # ----------------------------------------------------------
            for page_idx, page in enumerate(doc):
                page_number = page_idx + 1

                try:
                    page_text = page.get_text("text")
                except Exception:
                    page_text = ""

                page_text = _normalise_text(page_text)

                if not page_text:
                    continue  # skip blank pages

                # --- header detection --------------------------------
                section_headers: list[str] = []

                try:
                    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                    for block in blocks:
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            line_text_parts: list[str] = []
                            max_span_size = 0.0
                            is_bold = False

                            for span in line.get("spans", []):
                                line_text_parts.append(span.get("text", ""))
                                size = span.get("size", 0.0)
                                if size > max_span_size:
                                    max_span_size = size
                                flags = span.get("flags", 0)
                                if flags & (1 << 4):  # bit 4 = bold
                                    is_bold = True

                            line_text = " ".join(line_text_parts).strip()

                            if not line_text or len(line_text) > 120:
                                continue

                            # Criterion 1: font size significantly larger
                            size_is_header = max_span_size >= header_threshold

                            # Criterion 2: matches known heading pattern
                            pattern_is_header = _is_header_by_pattern(line_text)

                            # Criterion 3: bold + short text
                            bold_short = is_bold and len(line_text) < 80

                            if size_is_header or pattern_is_header or (bold_short and pattern_is_header):
                                header_clean = line_text.strip()
                                if header_clean and header_clean not in section_headers:
                                    section_headers.append(header_clean)
                except Exception:
                    # If dict-based extraction fails, fall back to pattern matching
                    for raw_line in page_text.split("\n"):
                        stripped = raw_line.strip()
                        if stripped and _is_header_by_pattern(stripped):
                            if stripped not in section_headers:
                                section_headers.append(stripped)

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=page_text,
                        section_headers=section_headers,
                    )
                )
        except Exception as exc:
            logger.error(
                "Error processing pages of {path}: {err}",
                path=str(pdf_path),
                err=str(exc),
            )
        finally:
            doc.close()

        logger.debug(
            "Parsed {n_pages} pages from {path}",
            n_pages=len(pages),
            path=pdf_path.name,
        )
        return pages
