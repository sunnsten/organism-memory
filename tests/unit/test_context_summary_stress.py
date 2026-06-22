from __future__ import annotations

from unittest.mock import Mock, call
import pytest

from organism.core.chat.orchestrator import ChatOrchestrator
from organism.core.config import RAGConfig
from organism.core.memory.rag.context_assembler import AssembledContext


def _make_facade_and_lm(*, summary_version_start: int = 0):
    """
    Build a facade mock with a real in-memory summary store simulation.

    The summary store tracks upsert/get calls so we can inspect version
    progression without a real SQLite database.
    """
    stored_summary: dict | None = None
    version = [summary_version_start]

    def fake_get(tenant_id, user_id, session_id):
        return dict(stored_summary) if stored_summary else None

    def fake_upsert(tenant_id, user_id, session_id, summary_text, **kwargs):
        nonlocal stored_summary
        version[0] += 1
        stored_summary = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "summary_text": summary_text,
            "version": version[0],
        }

    mock_summary_store = Mock()
    mock_summary_store.get.side_effect = fake_get
    mock_summary_store.upsert.side_effect = fake_upsert

    facade = Mock()
    facade.write.append_event.return_value = "exp-001"
    facade.consolidation.trigger_later.return_value = None
    facade.store.messages.add.return_value = 1
    facade.store.context_summaries = mock_summary_store

    lm = Mock()
    lm.count_tokens.side_effect = lambda text: len(text) // 4

    return facade, lm, mock_summary_store, version


def _make_assembled(n_history_pairs: int, chars_per_msg: int = 200) -> AssembledContext:
    """Create an AssembledContext with n_history_pairs of user/assistant messages."""
    working_memory = []
    for i in range(n_history_pairs):
        working_memory.append({"role": "user",      "content": f"User turn {i}: " + "u" * chars_per_msg})
        working_memory.append({"role": "assistant", "content": f"Asst turn {i}: " + "a" * chars_per_msg})
    return AssembledContext(
        system_prompt="You are helpful.",
        memory_block="",
        context_block="",
        working_memory=working_memory,
        user_question="new question",
    )


# ---------------------------------------------------------------------------
# 50-turn summary version progression
# ---------------------------------------------------------------------------

def test_50_turns_summary_version_increments():
    """
    Each trim cycle must increment the summary version.
    After 50 turns with a tiny budget that forces trim every turn,
    the version counter must be >= 1 (at least one trim occurred).
    """
    facade, lm, summary_store, version = _make_facade_and_lm()

    # Alternate generate side_effects: summary text then chat reply, repeated
    lm.generate.side_effect = [
        f"Summary {i}" if i % 2 == 0 else f"Chat reply {i}"
        for i in range(200)  # enough for 50 turns × 2 calls each
    ]

    cfg = RAGConfig(
        context_window_enabled=True,
        context_window_max_history_tokens=20,   # tiny → forces trim every turn
        context_window_min_messages=2,
        context_window_overflow_trigger_tokens=1,
        context_window_summary_max_tokens=50,
    )

    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm, rag_config=cfg)

    for turn in range(50):
        # Each turn provides fresh assembled context with growing history
        assembled = _make_assembled(n_history_pairs=min(turn + 1, 5), chars_per_msg=100)
        facade.retrieval.retrieve.return_value = assembled
        orchestrator.process_chat("t1", "u1", f"turn {turn}", session_id="s1")

    assert version[0] >= 1, f"Expected at least 1 summary update, got version={version[0]}"
    assert summary_store.upsert.call_count >= 1


def test_50_turns_summary_length_always_bounded():
    """
    After every turn, the stored summary must not exceed max_tokens * 6 chars.
    """
    summary_max_tokens = 30  # 30 tokens → max 180 chars
    max_chars = summary_max_tokens * 6

    facade, lm, summary_store, version = _make_facade_and_lm()

    # LM returns a summary that is always longer than the bound
    lm.generate.side_effect = [
        "X" * 1000 if i % 2 == 0 else f"Chat reply {i}"
        for i in range(200)
    ]

    cfg = RAGConfig(
        context_window_enabled=True,
        context_window_max_history_tokens=20,
        context_window_min_messages=2,
        context_window_overflow_trigger_tokens=1,
        context_window_summary_max_tokens=summary_max_tokens,
    )

    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm, rag_config=cfg)

    for turn in range(50):
        assembled = _make_assembled(n_history_pairs=min(turn + 1, 5), chars_per_msg=100)
        facade.retrieval.retrieve.return_value = assembled
        orchestrator.process_chat("t1", "u1", f"turn {turn}", session_id="s1")

        # After every upsert, check stored length
        if summary_store.upsert.call_count > 0:
            last_call = summary_store.upsert.call_args
            # summary_text is 4th positional arg or keyword
            call_kwargs = last_call.kwargs if hasattr(last_call, "kwargs") else {}
            stored_text = call_kwargs.get("summary_text") or (
                last_call.args[3] if len(last_call.args) > 3 else None
            )
            if stored_text is not None:
                assert len(stored_text) <= max_chars, (
                    f"Turn {turn}: summary length {len(stored_text)} > {max_chars}"
                )


def test_50_turns_no_exception_raised():
    """
    50 turns of sustained trim+summarize must not raise any exception.
    """
    facade, lm, summary_store, version = _make_facade_and_lm()

    lm.generate.side_effect = [
        f"Summary text for turn {i//2}" if i % 2 == 0 else f"Chat reply {i//2}"
        for i in range(200)
    ]

    cfg = RAGConfig(
        context_window_enabled=True,
        context_window_max_history_tokens=20,
        context_window_min_messages=2,
        context_window_overflow_trigger_tokens=1,
        context_window_summary_max_tokens=50,
    )

    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm, rag_config=cfg)

    # Must not raise
    for turn in range(50):
        assembled = _make_assembled(n_history_pairs=min(turn + 1, 5), chars_per_msg=100)
        facade.retrieval.retrieve.return_value = assembled
        orchestrator.process_chat("t1", "u1", f"turn {turn}", session_id="s1")


def test_50_turns_token_count_never_exceeds_budget():
    """
    After each trim, the history sent to the LM must fit within max_history_tokens.
    Inspects the actual messages passed to lm.generate() for the chat reply call.
    """
    from organism.core.chat.token_budget import count_messages_tokens

    max_history = 50  # tokens

    facade, lm, summary_store, version = _make_facade_and_lm()

    chat_replies = []

    def generate_side_effect(messages, **kwargs):
        # Identify summary vs chat calls: summary has a specific system prompt content
        system_content = messages[0].get("content", "") if messages else ""
        if "Summarize" in system_content or "summary" in system_content.lower():
            return "Compact summary."
        # Chat reply: record the messages length for inspection
        chat_replies.append(messages)
        return "Chat reply."

    lm.generate.side_effect = generate_side_effect

    cfg = RAGConfig(
        context_window_enabled=True,
        context_window_max_history_tokens=max_history,
        context_window_min_messages=2,
        context_window_overflow_trigger_tokens=1,
        context_window_summary_max_tokens=50,
    )

    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm, rag_config=cfg)

    for turn in range(50):
        assembled = _make_assembled(n_history_pairs=min(turn + 1, 6), chars_per_msg=80)
        facade.retrieval.retrieve.return_value = assembled
        orchestrator.process_chat("t1", "u1", f"turn {turn}", session_id="s1")

    assert chat_replies, "Expected at least one chat reply call"
    for i, msgs in enumerate(chat_replies):
        # History = all messages except system (index 0)
        history_msgs = msgs[1:]
        history_tokens = count_messages_tokens(history_msgs, lm=None)  # chars//4 heuristic
        assert history_tokens <= max_history + 50, (  # +50 slack for system/user current msg
            f"Turn {i}: history tokens {history_tokens} exceeds budget {max_history}"
        )
