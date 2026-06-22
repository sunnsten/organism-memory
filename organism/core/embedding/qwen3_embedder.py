from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default model for RAG embeddings
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIM = 1024


class Qwen3Embedder:
    """
    Embedding model using Qwen3-Embedding-0.6B.

    Produces 1024d L2-normalized vectors.
    Uses last_hidden_state mean pooling with attention mask.

    Example:
        >>> embedder = Qwen3Embedder()
        >>> vec = embedder.embed("How to learn Python?")
        >>> vec.shape
        (1024,)
        >>> np.linalg.norm(vec)  # ~1.0
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize the embedder by loading the model and tokenizer.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.).
                    If None, auto-detects CUDA availability.
            max_length: Maximum token length for input texts.
        """
        self._model_name = model_name
        self._max_length = max_length
        self._dim = DEFAULT_DIM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(
            "Loading embedding model %s on %s",
            self._model_name,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
        ).to(self._device)
        self._model.eval()

        # Detect actual dim from model config
        if hasattr(self._model.config, "hidden_size"):
            self._dim = self._model.config.hidden_size
            logger.info("Embedding dim detected: %d", self._dim)

    @property
    def dim(self) -> int:
        """Embedding dimensionality (1024 for Qwen3-Embedding-0.6B)."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text into a 1024d L2-normalized vector.

        Args:
            text: Input text string.

        Returns:
            numpy array of shape (1024,), dtype float32, L2-normalized.
        """
        self._ensure_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        device_type = self._device.split(":")[0] if ":" in self._device else self._device
        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=torch.float16)
            if device_type == "cuda"
            else torch.no_grad()
        )
        with torch.no_grad(), amp_ctx:
            outputs = self._model(**inputs)

        # Mean pooling over non-padding tokens
        embeddings = self._mean_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )

        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        vec = embeddings[0].cpu().float().numpy()
        return vec

    def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Embed multiple texts into L2-normalized vectors.

        Args:
            texts: List of input text strings.
            batch_size: Number of texts to process at once.

        Returns:
            List of numpy arrays, each of shape (1024,), dtype float32.
        """
        if not texts:
            return []

        self._ensure_loaded()
        results: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)

            device_type = self._device.split(":")[0] if ":" in self._device else self._device
            amp_ctx = (
                torch.autocast(device_type=device_type, dtype=torch.float16)
                if device_type == "cuda"
                else torch.no_grad()
            )
            with torch.no_grad(), amp_ctx:
                outputs = self._model(**inputs)

            embeddings = self._mean_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            for vec in embeddings.cpu().float().numpy():
                results.append(vec)

        return results

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling: average hidden states weighted by attention mask.

        Args:
            last_hidden_state: [B, L, D] hidden states.
            attention_mask: [B, L] binary mask.

        Returns:
            [B, D] pooled embeddings.
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def serialize_embedding(self, vec: np.ndarray) -> bytes:
        """
        Serialize a numpy embedding vector to bytes for SQLite BLOB storage.

        Args:
            vec: numpy array of shape (dim,), dtype float32.

        Returns:
            Raw bytes (float32 little-endian).
        """
        return vec.astype(np.float32).tobytes()

    def deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """
        Deserialize bytes from SQLite BLOB back to numpy vector.

        Args:
            blob: Raw bytes from database.

        Returns:
            numpy array of shape (dim,), dtype float32.
        """
        return np.frombuffer(blob, dtype=np.float32).copy()


__all__ = ["Qwen3Embedder"]
