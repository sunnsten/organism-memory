from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from organism.mcp_server import _make_handlers, ValidationError


# ---------------------------------------------------------------------------
# Minimal Organism stub
# ---------------------------------------------------------------------------

class _StubStore:
    """Minimal store stub for metrics."""
    class _Messages:
        def count(self, tenant_id: Any, user_id: Any = None) -> int:
            return 100
    class _Facts:
        def count(self, tenant_id: Any, user_id: Any = None) -> int:
            return 20
    class _Chunks:
        def count(self, tenant_id: Any) -> int:
            return 50
    messages = _Messages()
    facts = _Facts()
    chunks = _Chunks()


class _StubMemory:
    store = _StubStore()


class _StubOrchestrator:
    _memory = _StubMemory()


class _StubOrganism:
    """
    Minimal stub for testing _make_handlers without starting a server or loading models.
    Mirrors the public Organism API surface used by MCP handlers.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._tenant_id = "default"
        self._orchestrator = _StubOrchestrator()

    def chat(self, *, user_id: str, user_message: str, session_id: Any = None) -> Any:
        reply = MagicMock()
        reply.reply = f"echo: {user_message}"
        return reply

    def store_event(
        self,
        user_id: str,
        content: str,
        session_id: Any = None,
        source: str = "mcp",
        metadata: Any = None,
        tenant_id: Any = None,
    ) -> dict:
        ev_id = self._next_id
        self._next_id += 1
        return {"event_id": ev_id, "queued_for_extraction": False}

    def query_memory(
        self,
        user_id: str,
        query: str,
        max_facts: int = 8,
        max_chunks: int = 5,
        tenant_id: Any = None,
    ) -> dict:
        return {
            "facts": [{"id": 1, "content": f"fact for {query}", "score": 0.9, "source_type": "fact", "source_session_id": None}],
            "chunks": [],
            "total_facts": 1,
            "total_chunks": 0,
        }

    def remember(self, user_id: str, text: str) -> int:
        mem_id = self._next_id
        self._next_id += 1
        return mem_id

    def reset_user(self, user_id: str, tenant_id: Any = None) -> dict:
        return {"deleted": {"facts": 3, "messages": 10, "rag_chunks": 20, "memory_items": 1}}


@pytest.fixture
def org() -> _StubOrganism:
    return _StubOrganism()


@pytest.fixture
def handlers(org: _StubOrganism) -> dict:
    return _make_handlers(org)


# ---------------------------------------------------------------------------
# memory.store_event
# ---------------------------------------------------------------------------

def test_store_event_returns_event_id(handlers: dict) -> None:
    raw = handlers["memory.store_event"](user_id="alice", content="I live in Amsterdam")
    result = json.loads(raw)
    assert "event_id" in result
    assert isinstance(result["event_id"], int)
    assert "queued_for_extraction" in result


def test_store_event_missing_content_returns_validation_error(handlers: dict) -> None:
    raw = handlers["memory.store_event"](user_id="alice", content="")
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


def test_store_event_missing_user_id_returns_validation_error(handlers: dict) -> None:
    raw = handlers["memory.store_event"](user_id="", content="something")
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


# ---------------------------------------------------------------------------
# memory.query
# ---------------------------------------------------------------------------

def test_query_returns_facts_and_chunks(handlers: dict) -> None:
    raw = handlers["memory.query"](user_id="alice", query="Where do I live?")
    result = json.loads(raw)
    assert "facts" in result
    assert "chunks" in result
    assert "total_facts" in result
    assert "total_chunks" in result
    assert result["total_facts"] == 1
    fact = result["facts"][0]
    assert "id" in fact
    assert "content" in fact
    assert "score" in fact
    assert "source_type" in fact


def test_query_max_facts_clamped(handlers: dict) -> None:
    raw = handlers["memory.query"](user_id="alice", query="test", max_facts=0)
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


def test_query_max_facts_too_large(handlers: dict) -> None:
    raw = handlers["memory.query"](user_id="alice", query="test", max_facts=51)
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


def test_query_max_chunks_clamped(handlers: dict) -> None:
    raw = handlers["memory.query"](user_id="alice", query="test", max_chunks=25)
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


# ---------------------------------------------------------------------------
# memory.remember
# ---------------------------------------------------------------------------

def test_remember_returns_stored_and_id(handlers: dict) -> None:
    raw = handlers["memory.remember"](user_id="alice", text="Alice uses Python")
    result = json.loads(raw)
    assert result["stored"] is True
    assert isinstance(result["memory_id"], int)


def test_remember_empty_text_returns_error(handlers: dict) -> None:
    raw = handlers["memory.remember"](user_id="alice", text="")
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


# ---------------------------------------------------------------------------
# memory.reset
# ---------------------------------------------------------------------------

def test_reset_requires_confirm_true(handlers: dict) -> None:
    raw = handlers["memory.reset"](user_id="alice", confirm=False)
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


def test_reset_without_confirm_returns_error(handlers: dict) -> None:
    raw = handlers["memory.reset"](user_id="alice")
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


def test_reset_with_confirm_true_returns_deleted_counts(handlers: dict) -> None:
    raw = handlers["memory.reset"](user_id="alice", confirm=True)
    result = json.loads(raw)
    assert "deleted" in result
    d = result["deleted"]
    assert "facts" in d
    assert "messages" in d
    assert "rag_chunks" in d
    assert "memory_items" in d
    assert all(isinstance(v, int) for v in d.values())


# ---------------------------------------------------------------------------
# memory.metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_snapshot(handlers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from organism.shared.analytics import memory_metrics as mm_module
    from organism.shared.analytics.memory_metrics import MemoryMetricsSnapshot

    fake_snap = MemoryMetricsSnapshot(
        facts_extracted=30,
        facts_new=20,
        facts_confirmed=8,
        facts_invalidated=2,
        facts_errors=2,
        facts_latency_avg_s=0.5,
        retrieval_calls=10,
        retrieval_facts_avg=3.0,
        retrieval_latency_avg_s=0.012,
        profile_updates=5,
    )
    monkeypatch.setattr(mm_module, "take_snapshot", lambda: fake_snap)

    raw = handlers["memory.metrics"]()
    result = json.loads(raw)
    # DB counts from _StubStore (no user_id → total_chunks present)
    assert result["total_messages"] == 100
    assert result["total_facts"] == 20
    assert result["total_chunks"] == 50
    # pipeline counters
    assert result["facts_extracted"] == 30
    assert result["facts_extraction_errors"] == 2
    assert "avg_retrieval_latency_ms" in result


def test_metrics_with_user_id_returns_null_chunks(handlers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from organism.shared.analytics import memory_metrics as mm_module
    from organism.shared.analytics.memory_metrics import MemoryMetricsSnapshot

    fake_snap = MemoryMetricsSnapshot(
        facts_extracted=0, facts_new=0, facts_confirmed=0,
        facts_invalidated=0, facts_errors=0, facts_latency_avg_s=0.0,
        retrieval_calls=0, retrieval_facts_avg=0.0, retrieval_latency_avg_s=0.0,
        profile_updates=0,
    )
    monkeypatch.setattr(mm_module, "take_snapshot", lambda: fake_snap)

    raw = handlers["memory.metrics"](user_id="alice")
    result = json.loads(raw)
    assert "total_chunks" in result
    assert result["total_chunks"] is None


# ---------------------------------------------------------------------------
# organism_chat (backwards-compat)
# ---------------------------------------------------------------------------

def test_chat_returns_reply(handlers: dict) -> None:
    raw = handlers["organism_chat"](user_id="alice", message="Hello")
    assert "echo: Hello" in raw


def test_chat_empty_message_returns_error(handlers: dict) -> None:
    raw = handlers["organism_chat"](user_id="alice", message="")
    result = json.loads(raw)
    assert result["error"]["type"] == "ValidationError"


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_not_in_handlers(handlers: dict) -> None:
    assert "memory.nonexistent" not in handlers
