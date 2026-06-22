from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from organism.core.memory.rag.chunk_store import ChunkStore
from organism.core.stores.memory_item_store import MemoryItemStore

logger = logging.getLogger(__name__)


@dataclass
class FTSResult:
    """A single FTS search result."""
    id: int
    content: str
    bm25_score: float
    tier: str                      # "memory_item" or "rag_chunk"
    category: Optional[str] = None  # memory_items only
    source_type: Optional[str] = None  # rag_chunks only
    raw: Dict[str, Any] = field(default_factory=dict)


class FTSRetriever:
    """
    Full-Text Search retriever for both Research tier and Tier 1 (chunks).

    Searches memory_items_fts and rag_chunks_fts using BM25 ranking.
    Returns a unified list of results annotated by tier.
    """

    def __init__(
        self,
        memory_item_store: MemoryItemStore,
        chunk_store: ChunkStore,
    ):
        self._memory_items = memory_item_store
        self._chunks = chunk_store

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 20,
        tiers: Optional[List[str]] = None,
    ) -> List[FTSResult]:
        """
        Full-text search across tiers.

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            query: Search query text.
            top_k: Maximum results per tier.
            tiers: Which tiers to search. Default: both.
                   Options: ["memory_item"], ["rag_chunk"], or both.

        Returns:
            List of FTSResult, ranked by BM25 (best first).
        """
        if not query or not query.strip():
            return []

        search_tiers = tiers or ["memory_item", "rag_chunk"]
        results: List[FTSResult] = []

        # Research tier: memory_items (Tier 3)
        if "memory_item" in search_tiers:
            try:
                tier1 = self._memory_items.search_fts(
                    tenant_id, user_id, query, limit=top_k
                )
                for item_dict, score in tier1:
                    results.append(FTSResult(
                        id=item_dict["id"],
                        content=item_dict["content"],
                        bm25_score=score,
                        tier="memory_item",
                        category=item_dict.get("category"),
                        raw=item_dict,
                    ))
            except Exception as e:
                logger.warning("FTS memory_items search failed: %s", e)

        # Tier 1: rag_chunks
        if "rag_chunk" in search_tiers:
            try:
                tier2 = self._chunks.search_fts(
                    tenant_id, query, limit=top_k, user_id=user_id,
                )
                for chunk_dict, score in tier2:
                    results.append(FTSResult(
                        id=chunk_dict["id"],
                        content=chunk_dict["content"],
                        bm25_score=score,
                        tier="rag_chunk",
                        source_type=chunk_dict.get("source_type"),
                        raw=chunk_dict,
                    ))
            except Exception as e:
                logger.warning("FTS Tier 1 (chunks) search failed: %s", e)

        # Sort by BM25 (lower = better in FTS5)
        results.sort(key=lambda r: r.bm25_score)
        return results


__all__ = ["FTSRetriever", "FTSResult"]
