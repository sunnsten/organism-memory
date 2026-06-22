from __future__ import annotations

from unittest.mock import Mock
from pathlib import Path
import pytest


def _make_mock_orchestrator(reply: str = "Test reply") -> Mock:
    mock_orch = Mock()
    mock_orch.process_chat.return_value = {
        "reply": reply,
        "session_id": "s1",
        "user_message_id": 1,
        "assistant_message_id": 2,
        "experience_id": "exp-001",
        "experience_persisted": True,
    }
    return mock_orch


def _make_organism(mock_orchestrator=None):
    from organism.core.organism import Organism

    if mock_orchestrator is None:
        mock_orchestrator = _make_mock_orchestrator()
    return Organism(
        lm_backend=Mock(),
        chat_orchestrator=mock_orchestrator,
        tenant_id="default",
    )


def test_organism_chat_returns_reply():
    """Organism.chat() returns OrganismReply with .reply text."""
    organism = _make_organism()
    reply = organism.chat(user_id="u1", user_message="Hello?", session_id="s1")
    assert reply.reply == "Test reply"


def test_organism_chat_delegates_to_orchestrator():
    """Organism.chat() calls ChatOrchestrator.process_chat() with correct args."""
    mock_orch = _make_mock_orchestrator("OK")
    organism = _make_organism(mock_orchestrator=mock_orch)
    organism.chat("u1", "Hello?", session_id="s1", system_prompt="Be helpful")

    mock_orch.process_chat.assert_called_once_with(
        tenant_id="default",
        user_id="u1",
        user_message="Hello?",
        session_id="s1",
        system_prompt="Be helpful",
        max_new_tokens=None,
        model_override=None,
    )


def test_organism_raises_if_no_orchestrator_and_no_store():
    """Organism requires either a pre-built orchestrator or a store to build from."""
    from organism.core.organism import Organism
    with pytest.raises((ValueError, TypeError)):
        Organism()  # Nothing provided → must raise


def test_organism_raises_if_only_lm_no_store():
    """Without store, Organism cannot auto-build pipeline."""
    from organism.core.organism import Organism
    with pytest.raises((ValueError, TypeError)):
        Organism(lm_backend=Mock())  # No store → can't build orchestrator


def test_organism_mode2_from_store(tmp_path: Path):
    """Mode 2: Organism(store=..., lm_backend=...) builds its own orchestrator."""
    from organism.core.organism import Organism
    from organism.core.stores import UnifiedStore

    store = UnifiedStore(tmp_path / "test.db")
    lm = Mock()
    lm.generate.return_value = "hello"

    # Should construct successfully (no chat call since LM is mock without full retrieve setup)
    organism = Organism(store=store, lm_backend=lm)
    assert organism is not None


def test_organism_remember(tmp_path: Path):
    """remember() creates a MemoryItem in the store."""
    from organism.core.organism import Organism
    from organism.core.stores import UnifiedStore

    store = UnifiedStore(tmp_path / "test.db")
    lm = Mock()
    lm.generate.return_value = "hello"

    organism = Organism(store=store, lm_backend=lm, tenant_id="t1")
    item_id = organism.remember(user_id="u1", text="My favourite colour is blue")
    assert isinstance(item_id, int)
    assert item_id > 0

    # Verify it's in the store
    items = store.memory_items.get_all(tenant_id="t1", user_id="u1")
    assert any("blue" in item["content"] for item in items)


def test_organism_start_end_session(tmp_path: Path):
    """start_session() returns a session_id; end_session() closes it silently."""
    from organism.core.organism import Organism
    from organism.core.stores import UnifiedStore

    store = UnifiedStore(tmp_path / "test.db")
    lm = Mock()
    lm.generate.return_value = "hello"

    organism = Organism(store=store, lm_backend=lm, tenant_id="t1")
    session_id = organism.start_session(user_id="u1", title="Test session")
    assert isinstance(session_id, str)
    assert len(session_id) > 0

    # end_session should not raise
    organism.end_session(user_id="u1", session_id=session_id)
