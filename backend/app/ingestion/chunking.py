"""
Parent-child chunking strategy.

Creates large **parent** chunks (~1024 tokens) and subdivides each into
smaller **child** chunks (~256 tokens) with configurable overlap.
Children store a reference to their parent; parents store the list of
their children.  Both use deterministic UUID5 identifiers.

Usage:
    from app.ingestion.chunking import ParentChildChunker
    chunker = ParentChildChunker(parent_size=1024, child_size=256, overlap=50)
    parents, children = chunker.chunk(pages, paper_id="2311.12424")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.models.schemas import ParsedPage, TextChunk
from app.utils.logger import get_logger
from app.utils.token_counter import count_tokens

logger = get_logger(__name__)

# Deterministic namespace for UUID5 generation (DNS namespace)
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Sentence-boundary regex: split after .!? followed by whitespace + uppercase
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

@dataclass
class _Sentence:
    """Internal: a sentence with its metadata."""
    text: str
    page_number: int
    section_header: str  # most recent header seen before this sentence
    token_count: int


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using regex + newline heuristics."""
    # First split on double-newlines (paragraph breaks)
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split on sentence boundaries within the paragraph
        parts = _SENT_RE.split(para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _make_id(paper_id: str, chunk_type: str, index: int) -> str:
    """Deterministic UUID5 for a chunk."""
    seed = f"{paper_id}:{chunk_type}:{index}"
    return str(uuid.uuid5(_UUID_NS, seed))


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

class ParentChildChunker:
    """Splits parsed pages into parent and child text chunks."""

    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        overlap: int = 50,
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    # -------------------------------------------------------------------
    # chunk()
    # -------------------------------------------------------------------
    def chunk(
        self,
        pages: list[ParsedPage],
        paper_id: str,
    ) -> tuple[list[TextChunk], list[TextChunk]]:
        """Chunk *pages* into parent + child chunks.

        Returns
        -------
        (parent_chunks, child_chunks)
            Both lists are sorted by ``chunk_index``.
        """
        if not pages:
            return [], []

        # ----- Step 1: build a flat list of sentences with metadata -----
        sentences: list[_Sentence] = []
        current_header = ""

        for page in pages:
            # Update current header from this page's detected headers
            if page.section_headers:
                current_header = page.section_headers[-1]

            page_sents = _split_sentences(page.text)
            for sent_text in page_sents:
                tc = count_tokens(sent_text)
                if tc == 0:
                    continue
                sentences.append(
                    _Sentence(
                        text=sent_text,
                        page_number=page.page_number,
                        section_header=current_header,
                        token_count=tc,
                    )
                )

        if not sentences:
            return [], []

        # ----- Step 2: group sentences into parent chunks ---------------
        parent_groups: list[list[_Sentence]] = []
        current_group: list[_Sentence] = []
        current_tokens = 0

        for sent in sentences:
            # If adding this sentence would exceed parent size and we
            # already have content, start a new group.
            if current_tokens + sent.token_count > self.parent_size and current_group:
                parent_groups.append(current_group)
                current_group = []
                current_tokens = 0

            current_group.append(sent)
            current_tokens += sent.token_count

        if current_group:
            parent_groups.append(current_group)

        # ----- Step 3: build parent TextChunks --------------------------
        parent_chunks: list[TextChunk] = []
        child_chunks: list[TextChunk] = []
        global_chunk_idx = 0  # shared counter across parents + children

        for p_idx, group in enumerate(parent_groups):
            parent_text = " ".join(s.text for s in group)
            parent_page = group[0].page_number
            parent_header = group[0].section_header
            parent_token_count = count_tokens(parent_text)

            parent_id = _make_id(paper_id, "parent", p_idx)
            parent_chunk_idx = global_chunk_idx
            global_chunk_idx += 1

            # ----- Step 4: subdivide parent into child chunks -----------
            child_ids: list[str] = []
            children_for_parent = self._split_into_children(
                group, paper_id, parent_id, p_idx, global_chunk_idx,
            )

            for child in children_for_parent:
                child_ids.append(child.chunk_id)
                child_chunks.append(child)
                global_chunk_idx += 1

            parent_chunk = TextChunk(
                chunk_id=parent_id,
                paper_id=paper_id,
                chunk_type="parent",
                text=parent_text,
                parent_chunk_id="",       # parents have no parent
                child_chunk_ids=child_ids,
                page_number=parent_page,
                section_header=parent_header,
                chunk_index=parent_chunk_idx,
                token_count=parent_token_count,
                char_count=len(parent_text),
            )
            parent_chunks.append(parent_chunk)

        logger.debug(
            "Chunked paper {pid}: {np} parents, {nc} children",
            pid=paper_id,
            np=len(parent_chunks),
            nc=len(child_chunks),
        )
        return parent_chunks, child_chunks

    # -------------------------------------------------------------------
    # Internal: split a parent group into overlapping child chunks
    # -------------------------------------------------------------------
    def _split_into_children(
        self,
        sentences: list[_Sentence],
        paper_id: str,
        parent_id: str,
        parent_idx: int,
        start_global_idx: int,
    ) -> list[TextChunk]:
        """Subdivide a parent's sentences into overlapping child chunks."""
        children: list[TextChunk] = []
        child_local_idx = 0

        # Build sentence-level token prefix sums for window sliding
        sent_texts = [s.text for s in sentences]
        sent_tokens = [s.token_count for s in sentences]

        i = 0  # sentence pointer
        while i < len(sent_texts):
            # Collect sentences until we reach child_size tokens
            chunk_sents: list[int] = []  # indices into sent_texts
            chunk_tokens = 0

            j = i
            while j < len(sent_texts) and chunk_tokens + sent_tokens[j] <= self.child_size:
                chunk_sents.append(j)
                chunk_tokens += sent_tokens[j]
                j += 1

            # If no sentence fits (single sentence > child_size), take it anyway
            if not chunk_sents:
                chunk_sents.append(i)
                chunk_tokens = sent_tokens[i]
                j = i + 1

            child_text = " ".join(sent_texts[idx] for idx in chunk_sents)
            first_sent_idx = chunk_sents[0]

            child_id = _make_id(paper_id, "child", parent_idx * 1000 + child_local_idx)

            children.append(
                TextChunk(
                    chunk_id=child_id,
                    paper_id=paper_id,
                    chunk_type="child",
                    text=child_text,
                    parent_chunk_id=parent_id,
                    child_chunk_ids=[],  # children have no children
                    page_number=sentences[first_sent_idx].page_number,
                    section_header=sentences[first_sent_idx].section_header,
                    chunk_index=start_global_idx + child_local_idx,
                    token_count=count_tokens(child_text),
                    char_count=len(child_text),
                )
            )
            child_local_idx += 1

            # Advance pointer: move forward but overlap by `self.overlap` tokens
            overlap_tokens = 0
            advance_to = j
            if self.overlap > 0 and j < len(sent_texts):
                # Walk backwards from j to find how many sentences to re-include
                overlap_tokens = 0
                k = j - 1
                while k >= i and overlap_tokens + sent_tokens[k] <= self.overlap:
                    overlap_tokens += sent_tokens[k]
                    k -= 1
                advance_to = k + 1  # start next chunk from here

            # Ensure we always advance at least one sentence to avoid infinite loops
            i = max(advance_to, i + 1)

        return children
