from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def count_messages_tokens(messages: List[Dict[str, str]], lm=None) -> int:
    """
    Estimate token count for a list of messages.

    If lm is provided, uses lm.count_tokens() for each message content.
    Falls back to chars // 4 heuristic (consistent with AssembledContext.estimate_tokens()).

    Args:
        messages: List of {role, content} dicts.
        lm: Optional LMBackend instance with count_tokens(text) -> int method.

    Returns:
        Approximate token count.
    """
    if lm is not None:
        return sum(lm.count_tokens(m.get("content", "")) for m in messages)
    return sum(len(m.get("content", "")) for m in messages) // 4


def trim_messages_to_token_budget(
    messages: List[Dict[str, str]],
    max_tokens: int,
    min_messages: int = 3,
    lm=None,
) -> Tuple[List[Dict[str, str]], int]:
    """
    Trim oldest conversation history to fit within a token budget.

    Strategy:
    - Always keep messages[0] (system prompt) and messages[-1] (latest user turn).
    - Remove messages starting from index 1 (oldest history first).
    - Stop trimming when within budget or min_messages limit is reached.
    - If messages has <= 2 entries (system + user), returns unchanged.

    Args:
        messages: List of {role, content} dicts as returned by AssembledContext.to_messages().
        max_tokens: Maximum allowed token count (chars // 4 estimate).
        min_messages: Minimum number of messages to keep (including system + last user).
                      Must be at least 2. Clamped to max(2, min_messages).

    Returns:
        Tuple of (trimmed_messages, n_dropped) where n_dropped is the number
        of history messages removed.
    """
    if not messages:
        return messages, 0

    if count_messages_tokens(messages, lm=lm) <= max_tokens:
        return messages, 0

    # Must keep at least: system + last user = 2 messages.
    actual_min = max(2, min_messages)

    result = list(messages)
    dropped = 0

    # Pop from index 1 (oldest history) until within budget or at minimum size.
    while len(result) > actual_min and count_messages_tokens(result, lm=lm) > max_tokens:
        result.pop(1)
        dropped += 1

    return result, dropped


__all__ = [
    "count_messages_tokens",
    "trim_messages_to_token_budget",
]
