from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from organism.core.memory.rag.chunk_store import ChunkStore
from organism.core.memory.rag.context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextAssemblerConfig,
)
from organism.core.memory.rag.fact_retriever import FactRetriever, _format_fact_content
from organism.core.memory.rag.fts_retriever import FTSRetriever
from organism.core.memory.rag.hybrid_retriever import HybridResult, HybridRetriever
from organism.core.memory.rag.vector_retriever import VectorRetriever
from organism.core.stores.memory_item_store import MemoryItemStore
from organism.core.stores.message_store import MessageStore

if TYPE_CHECKING:
    from organism.core.stores.fact_store import FactStore
    from organism.core.config.rag_config import RAGConfig
    pass  # SSMRetrievalSignal removed (research layer)

logger = logging.getLogger(__name__)

import math as _math


def _rerank(
    results: list,
    alpha: float = 0.85,
    beta: float = 0.15,
    gamma: float = 0.0,
) -> list:
    """
    Re-rank hybrid search results by blending normalised RRF with importance.

    final = alpha * norm_rrf + beta * importance
    recency penalty removed (gamma=0) — LongMemEval asks about any session, not just recent.
    """
    if not results:
        return results

    now = time.time()
    max_rrf = max((r.rrf_score for r in results), default=1.0) or 1.0

    def _score(r) -> float:
        norm_rrf = r.rrf_score / max_rrf
        importance = getattr(r, "importance", 0.5)
        created_at = getattr(r, "created_at", 0)
        age_days = (now - created_at) / 86400.0 if created_at else 0.0
        recency = _math.exp(-age_days / 180.0)
        return alpha * norm_rrf + beta * importance + gamma * recency

    return sorted(results, key=_score, reverse=True)


def _expand_rounds(
    hybrid_results: list,
    chunk_store: ChunkStore,
    tenant_id: str,
    user_id: str,
    max_expand: int = 2,
) -> list:
    """Pull neighboring subchunks for any multi-part round hits.

    When retrieval surfaces a subchunk of a long round (round_parts_total > 1),
    this fetches up to max_expand adjacent parts so the model never sees a
    fragment without its context.
    """
    import json as _json
    from organism.core.memory.rag.hybrid_retriever import HybridResult

    rounds_hit: dict[str, list[int]] = {}
    rounds_meta: dict[str, dict] = {}

    for r in hybrid_results:
        if r.tier != "rag_chunk":
            continue
        raw_tags = r.raw.get("tags")
        if not raw_tags:
            continue
        try:
            tags = _json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except Exception:
            continue
        rid = tags.get("round_id")
        total = tags.get("round_parts_total", 1)
        if rid and total and total > 1:
            rounds_hit.setdefault(rid, []).append(tags.get("round_part", 0))
            rounds_meta[rid] = {"total": total, "tenant_id": tenant_id, "user_id": user_id}

    if not rounds_hit:
        return hybrid_results

    existing_ids = {r.id for r in hybrid_results}
    additional: list = []

    for round_id, hit_parts in rounds_hit.items():
        meta = rounds_meta[round_id]
        total = meta["total"]
        wanted: set[int] = set()
        for p in hit_parts:
            for delta in range(-max_expand, max_expand + 1):
                if 0 <= p + delta < total:
                    wanted.add(p + delta)
        missing = wanted - set(hit_parts)
        if not missing:
            continue
        try:
            rows = chunk_store.fetch_round_parts(
                round_id, list(missing), tenant_id, user_id,
            )
        except Exception:
            continue
        for row in rows:
            if row["id"] not in existing_ids:
                existing_ids.add(row["id"])
                raw_tags = row.get("tags")
                try:
                    tags = _json.loads(raw_tags) if isinstance(raw_tags, str) else (raw_tags or {})
                except Exception:
                    tags = {}
                additional.append(HybridResult(
                    id=row["id"],
                    content=row["content"],
                    rrf_score=0.0,
                    tier="rag_chunk",
                    sources=["round_expand"],
                    raw=row,
                    created_at=row.get("created_at", 0),
                ))

    return hybrid_results + additional


def _facts_to_hybrid(fact_rows: List[Dict[str, Any]], base_score: float = 1.0) -> List["HybridResult"]:
    """Convert FactRetriever result dicts to HybridResult objects (tier='memory_item')."""
    results = []
    for i, row in enumerate(fact_rows):
        score = base_score * (1.0 / (i + 1))
        # Historical predecessors get a lower score so they don't crowd out current facts
        if row.get("_historical"):
            score *= 0.5
        results.append(HybridResult(
            id=row["id"],
            content=_format_fact_content(row),
            rrf_score=score,
            tier="memory_item",
            sources=["facts"],
            category=row.get("category", "fact"),
            importance=row.get("importance", 0.5),
            created_at=row.get("created_at", 0),
            valid_from=row.get("event_time"),  # propagate event_time so assembler can sort
        ))
    return results


def _profile_to_hybrid(profile_rows: List[Dict[str, Any]]) -> List["HybridResult"]:
    """Convert user_profile rows to top-priority HybridResult items."""
    results = []
    for row in profile_rows:
        results.append(HybridResult(
            id=0,
            content=f"{row['key']}: {row['value']}",
            rrf_score=2.0,
            tier="memory_item",
            sources=["profile"],
            category="profile",
            importance=row.get("confidence", 0.8),
            created_at=0,
        ))
    return results


class RetrievalService:
    """
    Main RAG retrieval service for the Core layer.

    Orchestrates the full retrieval pipeline:
    - Query embedding
    - Hybrid search (FTS + Vector + RRF) across Tier 1 (chunks) + Tier 2 (facts)
    - Working memory fetch (Tier 0)
    - Context assembly

    Usage:
        service = RetrievalService(
            message_store=store.messages,
            memory_item_store=store.memory_items,
            chunk_store=chunk_store,
            embedder=qwen3_embedder,
        )
        context = service.retrieve(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            query="How to learn Python?",
            system_prompt="You are a helpful tutor.",
        )
        # context.to_messages() -> ready for LM
    """

    def __init__(
        self,
        message_store: MessageStore,
        memory_item_store: MemoryItemStore,
        chunk_store: ChunkStore,
        embedder=None,
        rrf_k: int = 60,
        memory_items_boost: float = 1.2,
        assembler_config: Optional[ContextAssemblerConfig] = None,
        fact_store: Optional["FactStore"] = None,
        rag_config: Optional["RAGConfig"] = None,
    ):
        """
        Args:
            message_store: Store for messages (Tier 0 source).
            memory_item_store: Store for memory items (Tier 3 / Research).
            chunk_store: Store for RAG chunks (Tier 1).
            embedder: Qwen3Embedder (or any object with embed() method).
            rrf_k: RRF parameter.
            memory_items_boost: Score boost for Research tier (memory_items).
            assembler_config: Context assembly config.
            fact_store: Optional FactStore for Tier 2 (facts) retrieval.
                        If provided, FactRetriever replaces memory_items hybrid search.
        """
        self._messages = message_store
        self._memory_items = memory_item_store
        self._chunks = chunk_store
        self._embedder = embedder
        self._fact_store = fact_store
        reranker = None
        if fact_store is not None and getattr(rag_config, "reranker_enabled", False):
            from organism.core.memory.rag.reranker import Reranker
            reranker = Reranker()
        self._fact_retriever = FactRetriever(fact_store=fact_store, reranker=reranker) if fact_store is not None else None
        self._locomo_mode = getattr(rag_config, "locomo_mode", False) if rag_config else False

        # Build retriever chain
        self._fts = FTSRetriever(memory_item_store, chunk_store)
        self._vector = VectorRetriever(memory_item_store, chunk_store, embedder)
        self._hybrid = HybridRetriever(
            self._fts, self._vector,
            rrf_k=rrf_k, memory_items_boost=memory_items_boost,
        )
        self._assembler = ContextAssembler(assembler_config)

        # Last retrieval trace — read by EvalAdapter.get_last_trace() via MemoryFacade
        self._last_trace: Optional[Any] = None

    def retrieve(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: int = 20,
        working_memory_limit: Optional[int] = 5,
        ssm_signal: Optional[Any] = None,
        memory_core: Optional[Any] = None,
        neural_slots_top_k: int = 3,
    ) -> AssembledContext:
        """
        Full retrieval pipeline: embed -> search -> assemble.

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            session_id: Current session ID (for working memory).
            query: User's question.
            system_prompt: Custom system prompt.
            top_k: Max results from hybrid search.
            working_memory_limit: Max recent messages for Tier 0.
                None = unlimited (load all session messages — needed for overflow
                detection to see the full history).
            ssm_signal: Optional SSM retrieval signal (pre-computed offline).
                       Controls retrieval strategy (skip/memory_only/full).
            memory_core: Optional MemoryCore for neural slot retrieval.
            neural_slots_top_k: Number of slots to retrieve (if memory_core provided).

        Returns:
            AssembledContext ready for LM generation.
        """
        # Start timing for analytics
        start_time = time.time()
        # 1. Working memory (Tier 0: recent messages) — always loaded first
        # working_memory_limit=None loads ALL session messages (unlimited mode)
        recent = self._messages.get_by_session(
            session_id, tenant_id, limit=working_memory_limit,
        )
        working_memory = [
            {"role": r["role"], "content": r["content"]}
            for r in recent
        ]

        # Cap override: when unlimited mode, tell assembler not to re-cap the messages.
        # working_memory[-N:] where N >= len returns all, so passing len() is safe.
        assembler_wm_cap = len(working_memory) if working_memory_limit is None else None

        # 2. Check SSM signal: skip retrieval entirely?
        if ssm_signal and ssm_signal.mode == "skip":
            logger.debug(
                "RetrievalService: SSM signal mode=skip, using working memory only",
            )
            context = self._assembler.assemble(
                hybrid_results=[],
                working_memory=working_memory,
                user_question=query,
                system_prompt=system_prompt,
                max_working_memory=assembler_wm_cap,
            )
            return context

        # 3. Embed query
        query_embedding = self._embed_query(query)

        # 4. Determine tiers based on SSM signal (default: Tier 1 chunks + Research memory_items)
        tiers = None
        if ssm_signal and ssm_signal.mode == "memory_only":
            tiers = ["memory_item"]  # Research tier only
            logger.debug(
                "RetrievalService: SSM signal mode=memory_only, searching Research tier only",
            )

        # 5. Tier 2 retrieval: FactRetriever — always run, empty result is valid (cold start)
        if self._fact_retriever is not None and self._fact_store is not None:
            fact_rows = self._fact_retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                tenant_id=tenant_id,
                k=20 if self._locomo_mode else 15,
            )
            profile_rows = self._fact_store.get_profile(tenant_id, user_id)
            # Tier 1 chunks — always fetched independently of facts
            chunk_results = self._hybrid.search(
                tenant_id, user_id, query, query_embedding,
                top_k=top_k,
                tiers=["rag_chunk"],
            )
            hybrid_results = (
                _profile_to_hybrid(profile_rows)
                + _facts_to_hybrid(fact_rows)
                + chunk_results
            )
        else:
            # Legacy path (no fact store): respect SSM-determined tiers
            hybrid_results = self._hybrid.search(
                tenant_id, user_id, query, query_embedding,
                top_k=top_k,
                tiers=tiers,
            )
        hybrid_results = _rerank(hybrid_results)

        # Expand multi-part rounds: pull neighboring subchunks for any subchunk hit
        if self._chunks is not None:
            hybrid_results = _expand_rounds(
                hybrid_results, self._chunks, tenant_id, user_id,
            )

        # 6. Neural slot retrieval (if memory_core provided)
        slot_results = []
        if memory_core is not None:
            try:
                logger.debug(
                    "RetrievalService: neural slot retrieval enabled (top_k=%d)",
                    neural_slots_top_k,
                )
                # NOTE: Full integration requires hidden states from LM backend
                # For now, just use query embedding as proxy
                slot_results = memory_core.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=neural_slots_top_k,
                )
                logger.debug("RetrievalService: retrieved %d neural slots", len(slot_results))
            except Exception as e:
                logger.warning("Neural slot retrieval failed: %s", e, exc_info=True)

        # Save retrieval trace for eval tooling (EvalAdapter.get_last_trace)
        from organism.shared.domain import RetrievalTrace
        db_ids = [r.id for r in hybrid_results if r.tier == "memory_item"]
        db_previews = [r.content[:120] for r in hybrid_results if r.tier == "memory_item"]
        # Tier 1 chunk: source_id links chunk back to experience_block UUID
        chunk_source_ids = [
            str(r.raw.get("source_id", ""))
            for r in hybrid_results
            if r.tier == "rag_chunk" and r.raw.get("source_id")
        ]
        self._last_trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            db_results_count=len(db_ids),
            db_result_ids=db_ids,
            metadata={
                "db_result_text_previews": db_previews,
                "chunk_source_ids": chunk_source_ids,
                "slot_result_text_previews": [],
                "slot_scores": [],
            },
        )

        # 7. Assemble context (hybrid + slots)
        # NOTE: ContextAssembler doesn't support slots yet, so we ignore slot_results for now
        # TODO: Update ContextAssembler to merge slot results
        context = self._assembler.assemble(
            hybrid_results=hybrid_results,
            working_memory=working_memory,
            user_question=query,
            system_prompt=system_prompt,
            max_working_memory=assembler_wm_cap,
            # slot_results=slot_results,  # Future: add to assembler
        )

        # 8. Analytics: track retrieval latency
        latency_ms = (time.time() - start_time) * 1000
        from organism.shared.analytics import analytics
        analytics.metric_retrieval(
            tenant_id=tenant_id,
            latency_ms=latency_ms,
            tier="hybrid",  # Tier 1 (chunks) + Tier 2 (facts) combined
        )

        logger.debug(
            "RetrievalService: query=%r ssm_mode=%s facts=%d chunks=%d wm=%d slots=%d tokens~%d latency=%.2fms",
            query[:50],
            ssm_signal.mode if ssm_signal else "full",
            context.memory_item_count,
            context.rag_chunk_count,
            context.working_memory_count,
            len(slot_results),
            context.estimate_tokens(),
            latency_ms,
        )

        return context

    def retrieve_hybrid_only(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run only the hybrid search (no context assembly).

        Useful for debugging or when you need raw results.

        Returns:
            List of HybridResult as dicts.
        """
        query_embedding = self._embed_query(query)
        results = self._hybrid.search(
            tenant_id, user_id, query, query_embedding, top_k=top_k,
        )
        return [
            {
                "id": r.id,
                "content": r.content,
                "rrf_score": r.rrf_score,
                "tier": r.tier,
                "sources": r.sources,
                "category": r.category,
                "source_type": r.source_type,
                "source_session_id": r.raw.get("session_id"),
            }
            for r in results
        ]

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query text, or return zero vector if no embedder."""
        if self._embedder is not None:
            return self._embedder.embed(query)
        # Fallback: zero vector (FTS-only mode)
        logger.warning("No embedder configured; vector search will return no results")
        return np.zeros(1024, dtype=np.float32)


__all__ = ["RetrievalService"]
