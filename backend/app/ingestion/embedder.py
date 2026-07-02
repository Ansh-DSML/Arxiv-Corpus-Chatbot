"""
Dual-model embedder for text (BGE-large-en-v1.5) and images (ColPali).

Both models are **lazily loaded** — memory is allocated only on the first
call, not at import time.  Optimised for CPU execution.

Usage:
    from app.ingestion.embedder import TextEmbedder, VisualEmbedder

    te = TextEmbedder()
    vecs  = te.embed_batch(["hello world"])        # -> list[list[float]]  (1024-dim)
    sparse = te.compute_sparse_vectors(["hello"])  # -> list[SparseVector]

    ve = VisualEmbedder()
    mvecs = ve.embed_single(base64_str)            # -> list[list[float]]  (N×128-dim)
"""

from __future__ import annotations

import base64
import io
from collections import Counter

import torch
from PIL import Image
from qdrant_client.models import SparseVector

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Text Embedder (BGE-large-en-v1.5  →  1024-dim dense + sparse)
# ═══════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """Encodes text with BGE-large and produces dense + sparse vectors."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = None          # SentenceTransformer (lazy)
        self._tokenizer = None      # HF tokenizer (lazy, for sparse)

    # ---- lazy loaders --------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading text embedding model: {m}", m=self._model_name)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        self._model.eval()
        logger.info("Text embedding model loaded (dim={d})", d=self._model.get_sentence_embedding_dimension())

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return
        logger.info("Loading tokenizer for sparse vectors: {m}", m=self._model_name)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        logger.info("Tokenizer loaded (vocab_size={v})", v=self._tokenizer.vocab_size)

    # ---- dense embeddings ----------------------------------------------

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = settings.TEXT_EMBED_BATCH_SIZE,
    ) -> list[list[float]]:
        """Encode *texts* to 1024-dim normalised dense vectors.

        No instruction prefix is added — this is for **document** encoding.
        Query prefixes are handled at retrieval time.
        """
        if not texts:
            return []

        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Convenience: encode a single text."""
        result = self.embed_batch([text], batch_size=1)
        return result[0] if result else []

    # ---- sparse vectors (for Qdrant hybrid BM25) -----------------------

    def compute_sparse_vectors(self, texts: list[str]) -> list[SparseVector]:
        """Compute BM25-style sparse vectors using the BGE tokeniser.

        Each subword token ID becomes a sparse dimension; the value is the
        term frequency (TF).  Qdrant applies IDF weighting at query time via
        ``Modifier.IDF`` on the sparse vector config.
        """
        if not texts:
            return []

        self._load_tokenizer()

        sparse_vectors: list[SparseVector] = []
        for text in texts:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                sparse_vectors.append(SparseVector(indices=[], values=[]))
                continue

            counts = Counter(token_ids)
            indices = list(counts.keys())
            values = [float(v) for v in counts.values()]

            sparse_vectors.append(SparseVector(indices=indices, values=values))

        return sparse_vectors


# ═══════════════════════════════════════════════════════════════════════════
# Visual Embedder (ColPali  →  N × 128-dim multi-vectors)
# ═══════════════════════════════════════════════════════════════════════════

class VisualEmbedder:
    """Encodes page images with ColPali and returns multi-vector embeddings."""

    def __init__(self, model_name: str = settings.COLPALI_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    # ---- lazy loader ---------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading ColPali model: {m} (this may take a few minutes on CPU)", m=self._model_name)
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor

            self._model = ColPali.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32,    # float32 for CPU
            ).eval()

            self._processor = ColPaliProcessor.from_pretrained(self._model_name)
            logger.info("ColPali model loaded successfully")
        except Exception as exc:
            logger.error(
                "Failed to load ColPali model: {err}. Visual embeddings will be unavailable.",
                err=str(exc),
            )
            raise

    # ---- multi-vector embedding ----------------------------------------

    def embed_single(self, image_base64: str) -> list[list[float]]:
        """Decode a base64 image, run ColPali, return multi-vector embeddings.

        Returns
        -------
        list[list[float]]
            A list of N vectors, each 128-dimensional.
            Returns an empty list if embedding fails.
        """
        if not image_base64:
            return []

        try:
            self._load_model()
        except Exception:
            return []

        try:
            # Decode base64 → PIL Image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Process through ColPali
            batch = self._processor.process_images([image])

            # Move tensors to CPU explicitly
            inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Extract embeddings — shape: (1, num_patches, 128)
            # outputs is typically the model output; extract embeddings
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[0]
            elif isinstance(outputs, torch.Tensor):
                embeddings = outputs[0]
            else:
                # colpali_engine may return embeddings directly
                embeddings = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            if isinstance(embeddings, torch.Tensor):
                result = embeddings.cpu().float().numpy().tolist()
            else:
                result = embeddings

            logger.debug(
                "ColPali produced {n} patch vectors (128-dim each)",
                n=len(result),
            )
            return result

        except Exception as exc:
            logger.error("ColPali embedding failed: {err}", err=str(exc))
            return []

    def embed_batch(self, images_base64: list[str]) -> list[list[list[float]]]:
        """Embed multiple images. Returns a list of multi-vectors per image."""
        results: list[list[list[float]]] = []
        for img_b64 in images_base64:
            results.append(self.embed_single(img_b64))
        return results
