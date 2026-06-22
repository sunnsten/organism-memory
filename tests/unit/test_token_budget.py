from __future__ import annotations

import pytest

from organism.core.chat.token_budget import (
    count_messages_tokens,
    trim_messages_to_token_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(*contents: str, system: str = "system prompt") -> list:
    """Build a messages list: [system, *history, user_question].
    contents[-1] is the user question; everything before is history.
    """
    msgs = [{"role": "system", "content": system}]
    for i, c in enumerate(contents[:-1]):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": c})
    msgs.append({"role": "user", "content": contents[-1]})
    return msgs


def _token_len(text: str) -> int:
    return len(text) // 4


# ---------------------------------------------------------------------------
# count_messages_tokens
# ---------------------------------------------------------------------------

class TestCountMessagesTokens:
    def test_empty_list(self):
        assert count_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "system", "content": "abcd"}]  # 4 chars → 1 token
        assert count_messages_tokens(msgs) == 1

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "a" * 40},   # 10 tokens
            {"role": "user",   "content": "b" * 80},   # 20 tokens
        ]
        assert count_messages_tokens(msgs) == 30

    def test_missing_content_key(self):
        msgs = [{"role": "system"}]  # no 'content'
        assert count_messages_tokens(msgs) == 0


# ---------------------------------------------------------------------------
# trim_messages_to_token_budget — no trimming needed
# ---------------------------------------------------------------------------

class TestTrimNoTrimNeeded:
    def test_within_budget_returns_unchanged(self):
        msgs = _make_messages("Hi", "Hello there")  # short, well under any budget
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=10_000)
        assert result == msgs
        assert dropped == 0

    def test_empty_list_returns_empty(self):
        result, dropped = trim_messages_to_token_budget([], max_tokens=100)
        assert result == []
        assert dropped == 0

    def test_exactly_at_budget_not_trimmed(self):
        # Make messages that total exactly max_tokens
        total_chars = 400  # → 100 tokens
        msgs = [
            {"role": "system", "content": "x" * 200},
            {"role": "user",   "content": "y" * 200},
        ]
        assert count_messages_tokens(msgs) == 100
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=100)
        assert dropped == 0
        assert result == msgs


# ---------------------------------------------------------------------------
# trim_messages_to_token_budget — trimming occurs
# ---------------------------------------------------------------------------

class TestTrimOccurs:
    def test_drops_oldest_history(self):
        # system + 4 history + user_question = 6 messages
        msgs = [
            {"role": "system",    "content": "S" * 4},       # system (always kept)
            {"role": "user",      "content": "A" * 400},     # oldest history — should drop
            {"role": "assistant", "content": "B" * 400},     # older history — should drop
            {"role": "user",      "content": "C" * 400},     # newer history
            {"role": "assistant", "content": "D" * 400},     # newer history
            {"role": "user",      "content": "question" * 10},  # last user (always kept)
        ]
        # ~400 chars per middle msg → ~100 tokens each; total ~(4+400*4+70)//4 ≈ 420 tokens
        total = count_messages_tokens(msgs)
        assert total > 50  # sanity

        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=50, min_messages=2)
        # Dropped oldest history messages from index 1
        assert dropped > 0
        # System message always kept
        assert result[0]["role"] == "system"
        # Last message always kept
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "question" * 10
        # Total tokens within budget (or at min_messages limit)
        assert len(result) >= 2

    def test_result_within_budget(self):
        # Build a large context
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user",      "content": "user msg " * 20})
            msgs.append({"role": "assistant",  "content": "reply msg " * 20})
        msgs.append({"role": "user", "content": "final question"})

        total = count_messages_tokens(msgs)
        max_tokens = total // 2  # Force significant trim

        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=max_tokens, min_messages=2)
        assert dropped > 0
        assert count_messages_tokens(result) <= max_tokens or len(result) == 2

    def test_min_messages_respected(self):
        # Even if over budget, never go below min_messages
        msgs = [
            {"role": "system",    "content": "x" * 400},
            {"role": "user",      "content": "x" * 400},
            {"role": "assistant", "content": "x" * 400},
            {"role": "user",      "content": "x" * 400},
        ]
        # Budget of 1 token — would try to drop everything, but min_messages=3 stops it
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=1, min_messages=3)
        assert len(result) == 3  # system + 1 history + last user
        assert result[0]["role"] == "system"
        assert result[-1]["role"] == "user"

    def test_min_messages_clamped_to_2(self):
        # min_messages=0 or 1 is treated as 2 (system + user minimum)
        msgs = [
            {"role": "system",    "content": "x" * 400},
            {"role": "user",      "content": "x" * 400},
            {"role": "assistant", "content": "x" * 400},
            {"role": "user",      "content": "x" * 400},
        ]
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=1, min_messages=1)
        assert len(result) == 2  # Clamped to 2 minimum


# ---------------------------------------------------------------------------
# trim_messages_to_token_budget — edge cases
# ---------------------------------------------------------------------------

class TestTrimEdgeCases:
    def test_only_system_and_user_nothing_to_drop(self):
        # No history between system and user — can't drop anything
        msgs = [
            {"role": "system", "content": "x" * 2000},
            {"role": "user",   "content": "y" * 2000},
        ]
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=1, min_messages=2)
        assert dropped == 0  # Nothing to drop — already at min
        assert result == msgs

    def test_drop_count_accuracy(self):
        msgs = [
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "old1" * 100},
            {"role": "assistant", "content": "old2" * 100},
            {"role": "user",      "content": "final question"},
        ]
        # Force drop of both history messages
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=1, min_messages=2)
        # Should have dropped 2 history msgs (or stopped at min 2)
        assert len(result) == 2
        assert dropped == 2

    def test_order_preserved_after_trim(self):
        msgs = [
            {"role": "system",    "content": "system"},
            {"role": "user",      "content": "A" * 400},  # dropped
            {"role": "assistant", "content": "B" * 400},  # dropped
            {"role": "user",      "content": "C" * 400},  # kept
            {"role": "assistant", "content": "D" * 400},  # kept
            {"role": "user",      "content": "question"},
        ]
        result, dropped = trim_messages_to_token_budget(msgs, max_tokens=250, min_messages=4)
        # Must start with system, end with user question
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "question"
        # Chronological order must be preserved (no reordering)
        for i in range(1, len(result) - 1):
            assert result[i] in msgs


# ---------------------------------------------------------------------------
# Integration: ChatOrchestrator respects rag_config.context_window_enabled=False
# ---------------------------------------------------------------------------

class TestOrchestratorTokenBudget:
    def test_trim_disabled_when_context_window_disabled(self):
        """When context_window_enabled=False, generate receives all messages."""
        from unittest.mock import Mock
        from organism.core.chat.orchestrator import ChatOrchestrator
        from organism.core.config import RAGConfig
        from organism.core.memory.rag.context_assembler import AssembledContext

        # Build a large assembled context
        assembled = AssembledContext(
            system_prompt="You are helpful.",
            memory_block="",
            context_block="",
            working_memory=[
                {"role": "user",      "content": "u" * 1000},
                {"role": "assistant", "content": "a" * 1000},
            ] * 5,
            user_question="final?",
        )

        facade = Mock()
        facade.retrieval.retrieve.return_value = assembled
        facade.write.append_event.return_value = "exp-001"
        facade.consolidation.trigger_later.return_value = None
        facade.store.messages.add.return_value = 1

        lm = Mock()
        lm.generate.return_value = "reply"

        cfg = RAGConfig(context_window_enabled=False)
        orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm, rag_config=cfg)
        orchestrator.process_chat("t1", "u1", "final?", session_id="s1")

        # All messages passed through untrimmed
        call_args = lm.generate.call_args[0][0]
        assert len(call_args) == len(assembled.to_messages())


# ---------------------------------------------------------------------------
# count_messages_tokens — tokenizer-aware variant
# ---------------------------------------------------------------------------

class TestCountMessagesTokensWithLM:
    def test_uses_lm_count_tokens_when_provided(self):
        """When lm is provided, use lm.count_tokens() instead of chars // 4."""
        from unittest.mock import Mock
        lm = Mock()
        lm.count_tokens.side_effect = lambda text: len(text)  # 1 token per char

        messages = [{"role": "user", "content": "hello"}]
        result = count_messages_tokens(messages, lm=lm)

        assert result == 5  # len("hello") = 5, not 5 // 4 = 1
        lm.count_tokens.assert_called_once_with("hello")

    def test_falls_back_to_heuristic_without_lm(self):
        """Without lm, behaviour is identical to current implementation."""
        messages = [{"role": "user", "content": "hello"}]
        assert count_messages_tokens(messages, lm=None) == 1  # 5 // 4

    def test_missing_content_key_still_returns_zero_with_lm(self):
        from unittest.mock import Mock
        lm = Mock()
        lm.count_tokens.return_value = 0
        messages = [{"role": "system"}]
        assert count_messages_tokens(messages, lm=lm) == 0

    def test_trim_uses_lm_count_tokens(self):
        """trim_messages_to_token_budget passes lm through to count_messages_tokens."""
        from unittest.mock import Mock
        lm = Mock()
        # Simulate accurate tokenizer: 100 tokens per message
        lm.count_tokens.return_value = 100

        msgs = _make_messages("history1", "history2", "current question")
        # Total: 4 messages × 100 tokens = 400 > max_tokens=150
        result, n_dropped = trim_messages_to_token_budget(msgs, max_tokens=150, lm=lm)

        assert n_dropped > 0
        assert lm.count_tokens.called
