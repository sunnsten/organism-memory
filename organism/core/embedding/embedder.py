from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for text embedding models used in RAG retrieval.

    Implementations must provide:
    - embed(): single text -> vector
    - embed_batch(): multiple texts -> list of vectors
    - dim: embedding dimensionality
    """

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g. 1024 for Qwen3-Embedding-0.6B)."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text into a vector.

        Args:
            text: Input text string.

        Returns:
            L2-normalized numpy array of shape (dim,), dtype float32.
        """
        ...

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed multiple texts into vectors.

        Args:
            texts: List of input text strings.
            batch_size: Processing batch size.

        Returns:
            List of L2-normalized numpy arrays, each of shape (dim,), dtype float32.
        """
        ...


__all__ = ["Embedder"]
