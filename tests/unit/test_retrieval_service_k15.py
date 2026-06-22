from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from organism.core.memory.service.retrieval_service import RetrievalService
from organism.core.memory.rag.context_assembler import ContextAssemblerConfig
from organism.core.memory.rag.hybrid_retriever import HybridResult


def _make_retrieval_service(fact_rows=None):
    """Build a RetrievalService with all dependencies mocked."""
    msg_store = MagicMock()
    msg_store.get_by_session.return_value = []

    mem_store = MagicMock()
    chunk_store = MagicMock()

    embedder = MagicMock()
    embedder.embed.return_value = np.zeros(1024, dtype=np.float32)

    fact_store = MagicMock()
    fact_store.get_profile.return_value = []

    svc = RetrievalService(
        message_store=msg_store,
        memory_item_store=mem_store,
        chunk_store=chunk_store,
        embedder=embedder,
        fact_store=fact_store,
    )
    # Patch internal components
    svc._fact_retriever = MagicMock()
    svc._fact_retriever.retrieve.return_value = fact_rows if fact_rows is not None else []
    svc._hybrid = MagicMock()
    svc._hybrid.search.return_value = []
    return svc


def test_chunk_search_called_when_no_facts():
    """Chunk search must be called even when fact retriever returns no results (cold start)."""
    svc = _make_retrieval_service(fact_rows=[])  # no facts yet

    ctx = svc.retrieve(
        tenant_id="t1", user_id="new_user_no_facts",
        session_id="s1", query="hello world",
    )

    assert ctx is not None
    assert ctx.memory_item_count == 0
    svc._hybrid.search.assert_called_once()  # type: ignore[attr-defined]


def test_chunk_search_called_when_facts_present():
    """Chunk search is always called independently, even when facts exist."""
    svc = _make_retrieval_service(fact_rows=[
        {"id": 1, "content": "User likes Python", "category": "preference",
         "importance": 0.5, "created_at": 0, "event_time": None, "_historical": False},
    ])

    svc.retrieve(
        tenant_id="t1", user_id="u1",
        session_id="s1", query="hello",
    )

    svc._hybrid.search.assert_called_once()  # type: ignore[attr-defined]


def test_fact_retriever_called_with_k15():
    """FactRetriever must be called with k=15 (up from k=8)."""
    svc = _make_retrieval_service(fact_rows=[])

    svc.retrieve(
        tenant_id="t1", user_id="u1",
        session_id="s1", query="test query",
    )

    call_kwargs = svc._fact_retriever.retrieve.call_args  # type: ignore[union-attr, attr-defined]
    assert call_kwargs.kwargs.get("k") == 15 or (
        call_kwargs.args and len(call_kwargs.args) >= 5 and call_kwargs.args[4] == 15
    )
