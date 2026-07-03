"""
Dual-model embedder for text (BGE-M3) and images (ColPali).

Both models are **lazily loaded** — memory is allocated only on the first
call, not at import time.  Optimised for CPU execution.

**BGE-M3** supports up to **8192 tokens** (vs 512 for BGE-large), so
parent chunks (1024 tokens) are never truncated.  It also produces
**native learned sparse vectors** — far superior to raw term-frequency
counting.

Usage:
    from app.ingestion.embedder import TextEmbedder, VisualEmbedder

    te = TextEmbedder()
    dense, sparse = te.embed_batch_hybrid(["hello world"])
    # dense:  list[list[float]]  (1024-dim)
    # sparse: list[SparseVector]

    ve = VisualEmbedder()
    mvecs = ve.embed_single(base64_str)   # -> list[list[float]]  (N×128-dim)
"""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image
from qdrant_client.models import SparseVector

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Text Embedder (BGE-M3  →  1024-dim dense + learned sparse)
# ═══════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """Encodes text with BGE-M3 and produces dense + sparse vectors.

    BGE-M3 key specs:
    * Max sequence length: **8192 tokens**
    * Dense dimension:     **1024**
    * Sparse output:       **learned lexical weights** (not raw TF)
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = None   # BGEM3FlagModel (lazy)

    # ---- lazy loader ---------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading BGE-M3 text embedding model: {m}", m=self._model_name)
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self._model_name,
            use_fp16=False,          # float32 for CPU
        )
        logger.info(
            "BGE-M3 loaded (dense=1024-dim, sparse=learned lexical, max_tokens=8192)"
        )

    # ---- hybrid encoding (dense + sparse in ONE forward pass) ----------

    def embed_batch_hybrid(
        self,
        texts: list[str],
        batch_size: int = settings.TEXT_EMBED_BATCH_SIZE,
    ) -> tuple[list[list[float]], list[SparseVector]]:
        """Encode *texts* and return **(dense_vectors, sparse_vectors)**.

        A single forward pass produces both dense (1024-dim, normalised)
        and sparse (learned lexical weights) representations.
        """
        if not texts:
            return [], []

        self._load_model()

        output = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,    # not needed for this pipeline
        )

        # Dense: numpy array (N, 1024) → list[list[float]]
        dense_vectors = output["dense_vecs"].tolist()

        # Sparse: list of dicts {token_id: learned_weight}
        sparse_vectors: list[SparseVector] = []
        for lexical_dict in output["lexical_weights"]:
            if not lexical_dict:
                sparse_vectors.append(SparseVector(indices=[], values=[]))
                continue
            indices = [int(k) for k in lexical_dict.keys()]
            values = [float(v) for v in lexical_dict.values()]
            sparse_vectors.append(SparseVector(indices=indices, values=values))

        return dense_vectors, sparse_vectors

    # ---- convenience methods (backward-compatible) ---------------------

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = settings.TEXT_EMBED_BATCH_SIZE,
    ) -> list[list[float]]:
        """Dense-only encoding. Prefer ``embed_batch_hybrid`` to avoid
        running the model twice."""
        dense, _ = self.embed_batch_hybrid(texts, batch_size)
        return dense

    def embed_single(self, text: str) -> list[float]:
        """Convenience: encode a single text (dense only)."""
        result = self.embed_batch([text], batch_size=1)
        return result[0] if result else []

    def compute_sparse_vectors(self, texts: list[str]) -> list[SparseVector]:
        """Sparse-only encoding. Prefer ``embed_batch_hybrid`` to avoid
        running the model twice."""
        _, sparse = self.embed_batch_hybrid(texts)
        return sparse


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
