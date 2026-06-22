from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from organism.core.memory.rag.chunk_store import ChunkStore
from organism.core.stores.memory_item_store import MemoryItemStore

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """A single vector search result."""
    id: int
    content: str
    similarity: float
    tier: str                      # "memory_item" or "rag_chunk"
    category: Optional[str] = None  # memory_items only
    source_type: Optional[str] = None  # rag_chunks only
    raw: Dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    """
    Semantic vector search retriever for both Tier 1 and Tier 2.

    Searches memory_items and rag_chunks by cosine similarity against
    a query embedding. Results are unified and sorted by similarity.

    For production scale, the Python-loop cosine can be replaced with
    sqlite-vec ANN index by subclassing or patching the search methods
    in MemoryItemStore / ChunkStore.
    """

    def __init__(
        self,
        memory_item_store: MemoryItemStore,
        chunk_store: ChunkStore,
        embedder=None,
    ):
        """
        Args:
            memory_item_store: Store for Tier 1.
            chunk_store: Store for Tier 2.
            embedder: Optional Embedder for encoding text queries into vectors.
                      If provided, search_text() can be used instead of passing
                      a pre-computed embedding.
        """
        self._memory_items = memory_item_store
        self._chunks = chunk_store
        self._embedder = embedder

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        min_similarity: float = 0.0,
        tiers: Optional[List[str]] = None,
    ) -> List[VectorResult]:
        """
        Vector similarity search across tiers.

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            query_embedding: Query vector (1024d), L2-normalized.
            top_k: Maximum results per tier.
            min_similarity: Minimum cosine similarity threshold.
            tiers: Which tiers to search. Default: both.

        Returns:
            List of VectorResult, ranked by similarity (best first).
        """
        search_tiers = tiers or ["memory_item", "rag_chunk"]
        results: List[VectorResult] = []

        # Tier 1: memory_items
        if "memory_item" in search_tiers:
            try:
                tier1 = self._memory_items.search_vector(
                    tenant_id, user_id, query_embedding,
                    limit=top_k, min_similarity=min_similarity,
                )
                for item_dict, sim in tier1:
                    results.append(VectorResult(
                        id=item_dict["id"],
                        content=item_dict["content"],
                        similarity=sim,
                        tier="memory_item",
                        category=item_dict.get("category"),
                        raw=item_dict,
                    ))
            except Exception as e:
                logger.warning("Vector Tier 1 search failed: %s", e)

        # Tier 2: rag_chunks
        if "rag_chunk" in search_tiers:
            try:
                tier2 = self._chunks.search_vector(
                    tenant_id, query_embedding,
                    limit=top_k, min_similarity=min_similarity,
                    user_id=user_id,
                )
                for chunk_dict, sim in tier2:
                    results.append(VectorResult(
                        id=chunk_dict["id"],
                        content=chunk_dict["content"],
                        similarity=sim,
                        tier="rag_chunk",
                        source_type=chunk_dict.get("source_type"),
                        raw=chunk_dict,
                    ))
            except Exception as e:
                logger.warning("Vector Tier 2 search failed: %s", e)

        # Sort by similarity (higher = better)
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def search_text(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.0,
        tiers: Optional[List[str]] = None,
    ) -> List[VectorResult]:
        """
        Search by text query (auto-embeds using the embedder).

        Args:
            query: Text query to embed and search.
            (other args same as search())

        Returns:
            List of VectorResult.

        Raises:
            RuntimeError: If no embedder was provided.
        """
        if self._embedder is None:
            raise RuntimeError("No embedder provided; use search() with a pre-computed embedding")

        query_embedding = self._embedder.embed(query)
        return self.search(
            tenant_id, user_id, query_embedding,
            top_k=top_k, min_similarity=min_similarity, tiers=tiers,
        )


__all__ = ["VectorRetriever", "VectorResult"]
