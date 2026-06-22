from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    Thin wrapper around FlagEmbedding's cross-encoder.
    Lazy-loads the model on first call; degrades to passthrough if unavailable.
    """

    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        self._model: Any = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.MODEL_NAME, use_fp16=True)
            self._available = True
            logger.info("Reranker loaded: %s", self.MODEL_NAME)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s) — passthrough mode", exc)

    def _score_pairs(self, pairs: List[tuple]) -> List[float]:
        result = self._model.compute_score(pairs, normalize=True)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return list(result)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Score (query, fact) pairs and return top_k by cross-encoder score."""
        if not self._available or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["content"]) for c in candidates]
        try:
            scores = self._score_pairs(pairs)
        except Exception as exc:
            logger.warning("Reranker scoring failed: %s", exc)
            return candidates[:top_k]

        for c, score in zip(candidates, scores):
            c["reranker_score"] = float(score)

        return sorted(
            candidates,
            key=lambda x: x.get("reranker_score", 0.0),
            reverse=True,
        )[:top_k]


__all__ = ["Reranker"]
