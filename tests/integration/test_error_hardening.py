import time
import pytest
from unittest.mock import MagicMock, patch  # patch used in test_chat_timeout_returns_504
from fastapi.testclient import TestClient


@pytest.fixture()
def fake_organism():
    reply = MagicMock()
    reply.reply = "hello back"
    org = MagicMock()
    org.chat.return_value = reply
    org.start_session.return_value = "sess-123"
    org.remember.return_value = 1
    return org


@pytest.fixture()
def test_client(fake_organism):
    """Build a TestClient with organism patched out at the module level."""
    import organism.api.server as srv

    # Swap organism singleton — no module reload needed (reload re-executes
    # Organism.from_config which loads the GPU model a second time → OOM).
    original_organism = srv.organism
    srv.organism = fake_organism

    # Reset in-memory rate-limit counters so each test starts from zero.
    # slowapi wraps the `limits` library; MemoryStorage.reset() clears all windows.
    _storage = getattr(srv.limiter, "_storage", None)
    if _storage is not None and hasattr(_storage, "reset"):
        _storage.reset()

    client = TestClient(srv.app, raise_server_exceptions=False)
    yield client, fake_organism

    # Restore
    srv.organism = original_organism
    if _storage is not None and hasattr(_storage, "reset"):
        _storage.reset()


def test_chat_returns_200(test_client):
    client, _ = test_client
    resp = client.post("/chat", json={"message": "hi", "user_id": "u1"})
    assert resp.status_code == 200


def test_chat_timeout_returns_504(test_client):
    """When LM call exceeds LM_TIMEOUT_SECONDS, server returns 504."""
    client, fake_organism = test_client

    def slow_chat(**kw):
        time.sleep(2)

    fake_organism.chat.side_effect = slow_chat

    with patch("organism.api.server.LM_TIMEOUT_SECONDS", 0.05):
        resp = client.post("/chat", json={"message": "hi", "user_id": "u1"})
    assert resp.status_code == 504


def test_chat_rate_limit_returns_429(test_client):
    """After 10 requests/min, the 11th must return 429."""
    client, fake_organism = test_client
    # Make 10 successful requests
    for _ in range(10):
        resp = client.post("/chat", json={"message": "hi", "user_id": "ratelimit_test"})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    # 11th must be rate limited
    resp = client.post("/chat", json={"message": "hi", "user_id": "ratelimit_test"})
    assert resp.status_code == 429, f"Expected 429, got {resp.status_code}"
