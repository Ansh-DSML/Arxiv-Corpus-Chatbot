"""
Token counting utilities using **tiktoken** (``cl100k_base`` encoding).

The encoder is lazily initialised on first call and reused thereafter.

Usage:
    from app.utils.token_counter import count_tokens, truncate_to_tokens

    n = count_tokens("Hello, world!")       # → 4
    t = truncate_to_tokens(long_text, 256)  # → first 256 tokens decoded
"""

import tiktoken

# ---------------------------------------------------------------------------
# Lazy singleton — encoder is heavy; only allocate once.
# ---------------------------------------------------------------------------
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return (and cache) the ``cl100k_base`` encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*.

    Returns ``0`` for empty / whitespace-only strings.
    """
    if not text or not text.strip():
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* so that it contains at most *max_tokens* tokens.

    If the text is already within the limit it is returned unchanged.
    """
    enc = _get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def tokenize(text: str) -> list[int]:
    """Return the raw token-id list for *text*."""
    if not text:
        return []
    return _get_encoder().encode(text, disallowed_special=())
