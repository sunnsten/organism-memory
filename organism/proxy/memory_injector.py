from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("organism_proxy")

_MEMORY_HEADER = "## Memory about this user"


def inject_memory(
    messages: list[dict[str, Any]],
    facts: list[str],
    max_tokens: int = 500,
) -> list[dict[str, Any]]:
    """
    Inject memory facts into the system prompt of a messages list.

    - If no facts: messages returned unchanged.
    - Truncates facts to stay within max_tokens (chars // 4 heuristic).
    - Adds ## Memory block to the existing system message or prepends one.
    """
    if not facts:
        return messages

    block_lines: list[str] = []
    budget = max_tokens * 4  # chars budget
    for fact in facts:
        if len(fact) + 1 > budget:
            break
        block_lines.append(fact)
        budget -= len(fact) + 1

    if not block_lines:
        return messages

    memory_block = f"{_MEMORY_HEADER}\n" + "\n".join(block_lines)

    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        existing = messages[0]["content"]
        messages[0] = {
            **messages[0],
            "content": f"{existing}\n\n{memory_block}",
        }
    else:
        messages.insert(0, {"role": "system", "content": memory_block})

    return messages


def inject_memory_anthropic(
    body: dict[str, Any],
    facts: list[str],
    max_tokens: int = 500,
) -> dict[str, Any]:
    """
    Inject memory facts into the top-level `system` field of an Anthropic request body.
    Works in-place with the Anthropic Messages API format (no OpenAI conversion needed).
    """
    if not facts:
        return body

    block_lines: list[str] = []
    budget = max_tokens * 4
    for fact in facts:
        if len(fact) + 1 > budget:
            break
        block_lines.append(fact)
        budget -= len(fact) + 1

    if not block_lines:
        return body

    memory_block = f"{_MEMORY_HEADER}\n" + "\n".join(block_lines)

    system = body.get("system", "")
    if isinstance(system, str):
        new_system = f"{system}\n\n{memory_block}" if system else memory_block
    elif isinstance(system, list):
        new_system = system + [{"type": "text", "text": memory_block}]
    else:
        new_system = memory_block

    return {**body, "system": new_system}


__all__ = ["inject_memory", "inject_memory_anthropic"]
