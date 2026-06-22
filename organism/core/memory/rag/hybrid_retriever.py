from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .fts_retriever import FTSRetriever, FTSResult
from .vector_retriever import VectorRetriever, VectorResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A single hybrid search result with RRF score."""
    id: int
    content: str
    rrf_score: float
    tier: str                         # "memory_item" or "rag_chunk"
    sources: List[str] = field(default_factory=list)  # ["fts", "vector"] or subset
    category: Optional[str] = None    # memory_items only
    source_type: Optional[str] = None  # rag_chunks only
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5           # from memory_items.importance column
    created_at: int = 0               # unix timestamp from memory_items.created_at
    valid_from: Optional[int] = None  # unix timestamp of the event itself (from summarizer date extraction)


class HybridRetriever:
    """
    Hybrid retriever combining FTS + Vector search via RRF.

    The Reciprocal Rank Fusion formula:
        rrf_score(item) = 1/(k + fts_rank) + 1/(k + vec_rank)

    Research tier results (memory_items) get a 1.2x boost to prioritize
    curated facts over raw chunks.

    Usage:
        hybrid = HybridRetriever(fts_retriever, vector_retriever)
        results = hybrid.search(
            tenant_id="t1",
            user_id="u1",
            query="Python programming",
            query_embedding=embedder.embed("Python programming"),
            top_k=10,
        )
    """

    def __init__(
        self,
        fts_retriever: FTSRetriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
        memory_items_boost: float = 1.2,
    ):
        """
        Args:
            fts_retriever: FTS retriever instance.
            vector_retriever: Vector retriever instance.
            rrf_k: RRF parameter k (default 60 per paper).
            memory_items_boost: Score multiplier for Research tier (memory_items).
        """
        self._fts = fts_retriever
        self._vector = vector_retriever
        self._rrf_k = rrf_k
        self._memory_items_boost = memory_items_boost

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        fts_top_k: int = 30,
        vec_top_k: int = 30,
        tiers: Optional[List[str]] = None,
    ) -> List[HybridResult]:
        """
        Hybrid search with RRF fusion.

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            query: Text query for FTS.
            query_embedding: Pre-computed query embedding for vector search.
            top_k: Final number of results to return.
            fts_top_k: Candidates from FTS per tier.
            vec_top_k: Candidates from vector per tier.
            tiers: Which tiers to search. Default: both.

        Returns:
            List of HybridResult, ranked by RRF score (best first).
        """
        # 1. FTS search
        fts_results = self._fts.search(
            tenant_id, user_id, query, top_k=fts_top_k, tiers=tiers,
        )

        # 2. Vector search
        vec_results = self._vector.search(
            tenant_id, user_id, query_embedding,
            top_k=vec_top_k, tiers=tiers,
        )

        # 3. RRF fusion
        fused = self._reciprocal_rank_fusion(fts_results, vec_results)

        return fused[:top_k]

    def search_text(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 10,
        fts_top_k: int = 30,
        vec_top_k: int = 30,
        tiers: Optional[List[str]] = None,
    ) -> List[HybridResult]:
        """
        Hybrid search using text query (auto-embeds).

        Requires the VectorRetriever to have an embedder.
        """
        if self._vector._embedder is None:
            raise RuntimeError("VectorRetriever has no embedder; use search() with pre-computed embedding")

        query_embedding = self._vector._embedder.embed(query)
        return self.search(
            tenant_id, user_id, query, query_embedding,
            top_k=top_k, fts_top_k=fts_top_k, vec_top_k=vec_top_k,
            tiers=tiers,
        )

    def _reciprocal_rank_fusion(
        self,
        fts_results: List[FTSResult],
        vec_results: List[VectorResult],
    ) -> List[HybridResult]:
        """
        Combine FTS and Vector results using Reciprocal Rank Fusion.

        RRF formula per item:
            score = 1/(k + fts_rank) + 1/(k + vec_rank)

        Items appearing in only one list get score from that list only.
        Research tier (memory_items) results get a 1.2x boost.

        Args:
            fts_results: FTS results (already sorted by BM25).
            vec_results: Vector results (already sorted by similarity).

        Returns:
            Fused results sorted by RRF score descending.
        """
        k = self._rrf_k

        # Build lookup: (tier, id) -> HybridResult candidate
        candidates: Dict[str, HybridResult] = {}

        # Process FTS results (rank is 1-based)
        for rank_0, fts_r in enumerate(fts_results):
            key = f"{fts_r.tier}:{fts_r.id}"
            rrf_contrib = 1.0 / (k + rank_0 + 1)

            if key not in candidates:
                candidates[key] = HybridResult(
                    id=fts_r.id,
                    content=fts_r.content,
                    rrf_score=0.0,
                    tier=fts_r.tier,
                    sources=[],
                    category=fts_r.category,
                    source_type=fts_r.source_type,
                    raw=fts_r.raw,
                    importance=float(fts_r.raw.get("importance", 0.5)),
                    created_at=int(fts_r.raw.get("created_at", 0)),
                    valid_from=fts_r.raw.get("valid_from") or None,
                )

            candidates[key].rrf_score += rrf_contrib
            candidates[key].fts_rank = rank_0 + 1
            candidates[key].sources.append("fts")

        # Process Vector results
        for rank_0, vec_r in enumerate(vec_results):
            key = f"{vec_r.tier}:{vec_r.id}"
            rrf_contrib = 1.0 / (k + rank_0 + 1)

            if key not in candidates:
                candidates[key] = HybridResult(
                    id=vec_r.id,
                    content=vec_r.content,
                    rrf_score=0.0,
                    tier=vec_r.tier,
                    sources=[],
                    category=vec_r.category,
                    source_type=vec_r.source_type,
                    raw=vec_r.raw,
                    importance=float(vec_r.raw.get("importance", 0.5)),
                    created_at=int(vec_r.raw.get("created_at", 0)),
                    valid_from=vec_r.raw.get("valid_from") or None,
                )

            candidates[key].rrf_score += rrf_contrib
            candidates[key].vector_rank = rank_0 + 1
            if "vector" not in candidates[key].sources:
                candidates[key].sources.append("vector")

        # Apply Research tier (memory_items) boost
        for cand in candidates.values():
            if cand.tier == "memory_item":
                cand.rrf_score *= self._memory_items_boost

        # Sort by RRF score descending
        result = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
        return result


__all__ = ["HybridRetriever", "HybridResult"]
