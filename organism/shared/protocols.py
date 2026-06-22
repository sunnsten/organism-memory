from __future__ import annotations

from typing import Any, List, Protocol


class TextGenerator(Protocol):
    """Minimal LLM interface for text generation (used by summarizers)."""

    def generate(
        self,
        messages: List[Any],  # List[Message] or list[dict[str, str]]
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Generate text from a list of messages.

        Args:
            messages: List of messages (List[Message] or list[dict[str, str]])
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text string
        """
        ...


__all__ = ["TextGenerator"]