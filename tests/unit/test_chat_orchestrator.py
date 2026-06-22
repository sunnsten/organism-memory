from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import pytest

from organism.core.chat.orchestrator import ChatOrchestrator
from organism.core.stores import UnifiedStore
from organism.core.memory.service import MemoryFacade


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "test_orchestrator.db")


@pytest.fixture
def memory_facade(unified_store: UnifiedStore) -> MemoryFacade:
    return MemoryFacade.from_store(unified_store, tenant_id="tenant1")


@pytest.fixture
def mock_lm():
    lm = Mock()
    lm.generate = Mock(return_value="Hello! How can I help you?")
    lm.count_tokens.side_effect = lambda text: len(text) // 4
    return lm


@pytest.fixture
def orchestrator(memory_facade: MemoryFacade, mock_lm) -> ChatOrchestrator:
    return ChatOrchestrator(memory_facade=memory_facade, lm_backend=mock_lm)


def test_orchestrator_accepts_memory_facade(memory_facade, mock_lm):
    orchestrator = ChatOrchestrator(memory_facade=memory_facade, lm_backend=mock_lm)
    assert orchestrator is not None


def test_orchestrator_generate_called_with_messages_list(memory_facade, mock_lm):
    """process_chat must call lm.generate(messages_list), NOT generate(prompt=string)."""
    orchestrator = ChatOrchestrator(memory_facade=memory_facade, lm_backend=mock_lm)
    result = orchestrator.process_chat(
        tenant_id="tenant1",
        user_id="u1",
        user_message="Hello?",
        session_id="s1",
    )

    mock_lm.generate.assert_called_once()
    call_args = mock_lm.generate.call_args
    messages_arg = call_args[0][0] if call_args[0] else call_args[1].get("messages")
    assert isinstance(messages_arg, list), "LM must receive messages list, not raw string"
    assert result["reply"] == "Hello! How can I help you?"


def test_orchestrator_basic_flow(orchestrator: ChatOrchestrator, unified_store: UnifiedStore):
    """Basic chat processing: user msg → generate → assistant msg saved."""
    result = orchestrator.process_chat(
        tenant_id="tenant1",
        user_id="user1",
        user_message="Hello!",
        session_id="session1",
    )

    assert result["reply"] == "Hello! How can I help you?"
    assert isinstance(result["user_message_id"], int)
    assert isinstance(result["assistant_message_id"], int)

    messages = unified_store.messages.get_by_session("session1", "tenant1", limit=10)
    assert len(messages) == 2
    roles = {m["role"] for m in messages}
    assert roles == {"user", "assistant"}


def test_orchestrator_does_not_call_consolidation():
    """process_chat must NOT call consolidation.trigger_later."""
    from organism.core.memory.rag.context_assembler import AssembledContext

    assembled = AssembledContext(
        system_prompt="You are helpful.",
        memory_block="",
        context_block="",
        working_memory=[],
        user_question="test question",
    )

    facade = Mock()
    facade.retrieval.retrieve.return_value = assembled
    facade.write.append_event.return_value = "exp-001"
    facade.store.messages.add.return_value = 1

    lm = Mock()
    lm.generate.return_value = "reply"
    lm.count_tokens.side_effect = lambda text: len(text) // 4

    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm)
    orchestrator.process_chat("t1", "u1", "Hello?", session_id="s1")

    # consolidation must never be touched
    assert not hasattr(facade, "_consolidation") or facade.consolidation.trigger_later.call_count == 0


def test_orchestrator_empty_message_raises(orchestrator: ChatOrchestrator):
    """Empty user message must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        orchestrator.process_chat(
            tenant_id="tenant1",
            user_id="user1",
            user_message="   ",
            session_id="session1",
        )


def test_orchestrator_result_keys(orchestrator: ChatOrchestrator):
    """process_chat result must contain expected keys."""
    result = orchestrator.process_chat(
        tenant_id="tenant1",
        user_id="user1",
        user_message="Hello!",
        session_id="session1",
    )
    assert "reply" in result
    assert "session_id" in result
    assert "user_message_id" in result
    assert "assistant_message_id" in result
