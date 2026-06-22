from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from organism.config import OrganismConfig
from tests.helpers import DummyLMBackend


# ------------------------------------------------------------------
# Fixture: FastAPI TestClient backed by Organism v2 + temp DB
# ------------------------------------------------------------------

@pytest.fixture
def api_env(tmp_path: Path):
    """
    Build test environment:
    - Reload server module with create_lm_backend patched (prevents HF download)
    - Inject Organism v2 with DummyLMBackend + temp UnifiedStore
    - Return (client, server_module)
    """
    from organism.core.organism import Organism
    from organism.core.stores import UnifiedStore

    # Always patch create_lm_backend before import/reload so the module-level
    # Organism.from_config() call in server.py uses DummyLMBackend, not a real model.
    import organism.backbone as bb
    _orig = bb.create_lm_backend
    bb.create_lm_backend = lambda cfg: DummyLMBackend(hidden_size=16, device="cpu")
    try:
        if "organism.api.server" in sys.modules:
            importlib.reload(sys.modules["organism.api.server"])
        server_module = importlib.import_module("organism.api.server")
    finally:
        bb.create_lm_backend = _orig

    # Replace the module-level organism with a fresh one backed by a temp DB
    store = UnifiedStore(tmp_path / "api_test.db")
    lm = DummyLMBackend(hidden_size=16, device="cpu")
    server_module.organism = Organism(store=store, lm_backend=lm, tenant_id="default")  # type: ignore[assignment]

    client = TestClient(server_module.app)
    return client, server_module


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.real_model
def test_health_ok(api_env):
    client, _ = api_env
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.integration
@pytest.mark.real_model
def test_chat_returns_reply(api_env):
    """POST /chat returns reply + session_id."""
    client, _ = api_env
    resp = client.post("/chat", json={"user_id": "alice", "message": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert data["reply"].startswith("echo:")
    assert "session_id" in data and isinstance(data["session_id"], str)


@pytest.mark.integration
@pytest.mark.real_model
def test_chat_saves_messages_to_store(api_env):
    """After /chat, both user and assistant messages are in UnifiedStore."""
    client, server = api_env
    resp = client.post("/chat", json={"user_id": "alice", "message": "hello from api"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    store = server.organism._orchestrator._memory.store
    messages = store.messages.get_by_session(session_id, "default", limit=10)
    roles = {m["role"] for m in messages}
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.integration
@pytest.mark.real_model
def test_remember_creates_memory_item(api_env):
    """POST /remember creates a MemoryItem in UnifiedStore."""
    client, server = api_env
    resp = client.post("/remember", json={"user_id": "bob", "text": "sky is blue"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["memory_id"], int)

    store = server.organism._orchestrator._memory.store
    items = store.memory_items.get_all(tenant_id="default", user_id="bob")
    assert any("sky is blue" in item["content"] for item in items)


@pytest.mark.integration
@pytest.mark.real_model
def test_session_start_and_end(api_env):
    """POST /session/start returns session_id; /session/end completes silently."""
    client, _ = api_env
    resp = client.post("/session/start", json={"user_id": "carol", "title": "test"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert isinstance(session_id, str) and len(session_id) > 0

    resp_end = client.post("/session/end", json={"user_id": "carol", "session_id": session_id})
    assert resp_end.status_code == 200
    assert resp_end.json()["status"] == "ok"


@pytest.mark.integration
@pytest.mark.real_model
def test_chat_rejects_bad_user_id(api_env):
    """user_id with spaces → 400 Invalid user_id."""
    client, _ = api_env
    resp = client.post("/chat", json={"user_id": "bad id with space", "message": "hi"})
    assert resp.status_code == 400
    assert "user_id" in resp.json().get("detail", "").lower()


@pytest.mark.integration
@pytest.mark.real_model
def test_chat_auto_creates_session(api_env):
    """When session_id is omitted, /chat auto-creates a session."""
    client, _ = api_env
    resp = client.post("/chat", json={"user_id": "dave", "message": "no session"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] is not None


@pytest.mark.integration
@pytest.mark.real_model
def test_chat_endpoint_latency_smoke(api_env):
    """Smoke latency test with DummyLMBackend (not a hard perf assertion)."""
    client, _ = api_env
    N = 10
    start = time.perf_counter()
    for i in range(N):
        resp = client.post("/chat", json={"user_id": "perf_user", "message": f"msg {i}"})
        assert resp.status_code == 200
    elapsed = time.perf_counter() - start
    print(f"\n[perf] /chat avg latency with DummyLMBackend: {elapsed / N:.4f}s")


@pytest.mark.integration
def test_debug_memory_empty_user(api_env):
    """Unknown user → 200 with all counts = 0 and empty lists."""
    client, _ = api_env
    resp = client.get("/api/debug/memory?user_id=nobody")
    assert resp.status_code == 200
    data = resp.json()
    assert data["memories_count"] == 0
    assert data["recent_memories"] == []


@pytest.mark.integration
def test_debug_memory_bad_user_id(api_env):
    """user_id with spaces → 400."""
    client, _ = api_env
    resp = client.get("/api/debug/memory?user_id=bad id")
    assert resp.status_code == 400
    assert "user_id" in resp.json().get("detail", "").lower()


@pytest.mark.integration
def test_debug_memory_default_user_id(api_env):
    """No user_id param → uses 'default', returns 200 with zero counts."""
    client, _ = api_env
    resp = client.get("/api/debug/memory")
    assert resp.status_code == 200
    data = resp.json()
    assert data["memories_count"] == 0
    assert data["recent_memories"] == []
