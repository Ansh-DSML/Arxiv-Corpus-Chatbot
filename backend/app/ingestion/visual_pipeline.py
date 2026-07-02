"""
Visual content extraction pipeline.

Detects figures and tables in PDFs using ``unstructured`` (with fallback
to PyMuPDF), rasterises the relevant pages with ``pdf2image``, and
returns ``VisualElement`` objects ready for ColPali embedding.

Usage:
    from app.ingestion.visual_pipeline import VisualExtractor
    extractor = VisualExtractor()
    visuals = extractor.extract(Path("paper.pdf"), paper_id="2311.12424")
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

from app.config import settings
from app.models.schemas import VisualElement
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Deterministic namespace for UUID5 generation
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _make_visual_id(paper_id: str, page_number: int, visual_index: int) -> str:
    """Deterministic UUID5 for a visual element."""
    seed = f"{paper_id}:visual:{page_number}:{visual_index}"
    return str(uuid.uuid5(_UUID_NS, seed))


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rasterise_page(pdf_path: Path, page_number: int, dpi: int = 300) -> Image.Image | None:
    """Rasterise a single PDF page to a PIL image using pdf2image."""
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(pdf_path),
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
            fmt="png",
        )
        return images[0] if images else None
    except Exception as exc:
        logger.warning(
            "pdf2image rasterisation failed for page {p} of {f}: {e}",
            p=page_number,
            f=pdf_path.name,
            e=str(exc),
        )
        # Fallback: use PyMuPDF pixmap
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_number - 1]  # 0-indexed
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            return img
        except Exception as exc2:
            logger.error("PyMuPDF pixmap fallback also failed: {e}", e=str(exc2))
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Extraction strategies
# ═══════════════════════════════════════════════════════════════════════════

def _extract_with_unstructured(pdf_path: Path) -> list[dict]:
    """Try ``unstructured`` with hi_res → fast → give up.

    Returns a list of dicts with keys:
      type ("figure" | "table"), page_number, bbox, caption, ocr_text
    """
    elements_out: list[dict] = []

    for strategy in ("hi_res", "fast"):
        try:
            from unstructured.partition.pdf import partition_pdf

            logger.debug(
                "Trying unstructured strategy='{s}' on {f}",
                s=strategy,
                f=pdf_path.name,
            )

            elements = partition_pdf(
                str(pdf_path),
                strategy=strategy,
                infer_table_structure=True,
            )

            for idx, el in enumerate(elements):
                el_type = type(el).__name__

                if el_type in ("Table",):
                    page_num = getattr(el.metadata, "page_number", 1) or 1
                    coords = getattr(el.metadata, "coordinates", None)
                    bbox = _coords_to_bbox(coords)
                    ocr_text = ""
                    if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                        ocr_text = el.metadata.text_as_html
                    elif hasattr(el, "text") and el.text:
                        ocr_text = el.text

                    caption = _find_caption(elements, idx)

                    elements_out.append({
                        "type": "table",
                        "page_number": int(page_num),
                        "bbox": bbox,
                        "caption": caption,
                        "ocr_text": ocr_text,
                    })

                elif el_type in ("Image", "FigureCaption"):
                    if el_type == "FigureCaption":
                        continue  # handled as caption for adjacent Image
                    page_num = getattr(el.metadata, "page_number", 1) or 1
                    coords = getattr(el.metadata, "coordinates", None)
                    bbox = _coords_to_bbox(coords)
                    caption = _find_caption(elements, idx)

                    elements_out.append({
                        "type": "figure",
                        "page_number": int(page_num),
                        "bbox": bbox,
                        "caption": caption,
                        "ocr_text": "",
                    })

            if elements_out:
                logger.debug(
                    "unstructured (strategy={s}) found {n} visuals",
                    s=strategy,
                    n=len(elements_out),
                )
                return elements_out

        except ImportError:
            logger.warning("unstructured not available; skipping strategy={s}", s=strategy)
            break
        except Exception as exc:
            logger.warning(
                "unstructured strategy={s} failed: {e}",
                s=strategy,
                e=str(exc),
            )
            continue

    return elements_out


def _coords_to_bbox(coords) -> list[float]:
    """Convert unstructured coordinates to [x0, y0, x1, y1]."""
    if coords is None:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        points = coords.points
        if points and len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 0.0]


def _find_caption(elements, idx: int) -> str:
    """Look for a FigureCaption or Caption element adjacent to index *idx*."""
    for offset in (1, -1, 2, -2):
        neighbour_idx = idx + offset
        if 0 <= neighbour_idx < len(elements):
            el = elements[neighbour_idx]
            el_type = type(el).__name__
            if el_type in ("FigureCaption", "Caption", "NarrativeText"):
                text = getattr(el, "text", "")
                if text and len(text) < 500:
                    # Heuristic: captions often start with "Figure" or "Table"
                    lower = text.strip().lower()
                    if lower.startswith(("fig", "table", "tab.", "figure")):
                        return text.strip()
            # Only look at NarrativeText in first 2 offsets
            if el_type == "NarrativeText" and abs(offset) > 1:
                continue
    return ""


def _extract_images_with_pymupdf(pdf_path: Path) -> list[dict]:
    """Fallback: detect images embedded in PDF using PyMuPDF."""
    elements_out: list[dict] = []

    try:
        doc = fitz.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_number = page_idx + 1
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image or base_image.get("width", 0) < 100:
                        continue  # skip tiny images (icons, bullets)
                    if base_image.get("height", 0) < 100:
                        continue

                    elements_out.append({
                        "type": "figure",
                        "page_number": page_number,
                        "bbox": [0.0, 0.0, 0.0, 0.0],  # PyMuPDF doesn't give bbox easily
                        "caption": "",
                        "ocr_text": "",
                    })
                except Exception:
                    continue
        doc.close()
    except Exception as exc:
        logger.warning("PyMuPDF image extraction failed: {e}", e=str(exc))

    return elements_out


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

class VisualExtractor:
    """Extracts figures, tables, and representative page images from PDFs."""

    def extract(self, pdf_path: Path, paper_id: str) -> list[VisualElement]:
        """Extract visual elements from *pdf_path*.

        Processing order:
        1. Try ``unstructured`` (hi_res → fast) for layout-aware detection.
        2. Fall back to PyMuPDF embedded-image extraction.
        3. If nothing detected, select sample pages as ``full_page`` visuals.
        4. Rasterise relevant pages and encode as base64 PNG.

        Returns
        -------
        list[VisualElement]
            All fields populated; no ``None`` values.
        """
        if not pdf_path.exists():
            logger.warning("PDF not found for visual extraction: {p}", p=str(pdf_path))
            return []

        try:
            return self._extract_impl(pdf_path, paper_id)
        except Exception as exc:
            logger.error(
                "Visual extraction failed for {pid}: {err}",
                pid=paper_id,
                err=str(exc),
            )
            return []

    def _extract_impl(self, pdf_path: Path, paper_id: str) -> list[VisualElement]:
        """Core extraction logic."""
        # --- detect figures / tables ---
        detected = _extract_with_unstructured(pdf_path)

        if not detected:
            logger.debug("unstructured found nothing; trying PyMuPDF images for {pid}", pid=paper_id)
            detected = _extract_images_with_pymupdf(pdf_path)

        # Collect unique page numbers that have visual content
        visual_pages: set[int] = {d["page_number"] for d in detected}

        # --- if nothing detected, pick sample pages for full_page visuals ---
        total_pages = self._count_pages(pdf_path)

        if not visual_pages and total_pages > 0:
            # Select representative pages: first, second, middle, last
            sample = {1}
            if total_pages >= 2:
                sample.add(2)
            if total_pages >= 4:
                sample.add(total_pages // 2)
            if total_pages >= 3:
                sample.add(total_pages)

            for pn in sorted(sample):
                detected.append({
                    "type": "full_page",
                    "page_number": pn,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "caption": "",
                    "ocr_text": "",
                })
            visual_pages = sample

        # --- rasterise & build VisualElement objects ---
        visuals: list[VisualElement] = []
        visual_index = 0

        # Cache rasterised pages so we don't rasterise the same page twice
        page_cache: dict[int, Image.Image] = {}

        for det in detected:
            page_num = det["page_number"]

            # Rasterise page if not cached
            if page_num not in page_cache:
                img = _rasterise_page(pdf_path, page_num, dpi=settings.PDF_DPI)
                if img is None:
                    continue
                page_cache[page_num] = img

            page_img = page_cache[page_num]
            image_base64 = _pil_to_base64(page_img)

            visual = VisualElement(
                visual_id=_make_visual_id(paper_id, page_num, visual_index),
                paper_id=paper_id,
                visual_type=det["type"],
                page_number=page_num,
                bbox=det["bbox"],
                caption=det["caption"],
                ocr_text=det["ocr_text"],
                image_base64=image_base64,
            )
            visuals.append(visual)
            visual_index += 1

        logger.info(
            "Extracted {n} visual elements from {pid} ({pages} pages rasterised)",
            n=len(visuals),
            pid=paper_id,
            pages=len(page_cache),
        )
        return visuals

    @staticmethod
    def _count_pages(pdf_path: Path) -> int:
        """Return the number of pages in the PDF."""
        try:
            doc = fitz.open(str(pdf_path))
            n = len(doc)
            doc.close()
            return n
        except Exception:
            return 0
