from __future__ import annotations

import logging
from typing import List

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class VLLMEmbedder:
    """
    Embedder that delegates to a vLLM (or any OpenAI-compatible) /v1/embeddings endpoint.

    The remote model must return L2-normalized float32 vectors.
    vLLM normalizes by default for embedding models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002/v1",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dim: int = 1024,
        timeout: float = 30.0,
        api_key: str = "not-needed",
    ):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dim = dim
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vecs = self._call([text])
        return vecs[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        if not texts:
            return []
        results: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._call(batch))
        return results

    def _call(self, texts: List[str]) -> List[np.ndarray]:
        resp = self._client.post(
            "/embeddings",
            json={"model": self._model_name, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = []
        for item in data:
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            vecs.append(vec)
        return vecs

    def close(self) -> None:
        self._client.close()


__all__ = ["VLLMEmbedder"]
