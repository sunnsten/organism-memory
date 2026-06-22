from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hybrid_retriever import HybridResult

logger = logging.getLogger(__name__)


@dataclass
class AssembledContext:
    """The final assembled context ready for LM generation."""
    system_prompt: str
    memory_block: str               # Tier 2 (facts) + Tier 3 (research) items formatted as text
    context_block: str              # Tier 1 chunks formatted as text
    working_memory: List[Dict[str, str]]  # Recent messages [{role, content}]
    user_question: str
    memory_item_count: int = 0
    rag_chunk_count: int = 0
    working_memory_count: int = 0

    def to_messages(self) -> List[Dict[str, str]]:
        """
        Convert to a messages list suitable for chat LM.

        Returns:
            List of {role, content} dicts in the correct order.
        """
        messages: List[Dict[str, str]] = []

        # 1. System prompt with memory and context blocks
        system_parts = [self.system_prompt]

        if self.memory_block:
            system_parts.append(
                f"\n\n## Known facts about the user\n{self.memory_block}"
            )

        if self.context_block:
            system_parts.append(
                f"\n\n## Timeline of relevant events (sorted by date)\n{self.context_block}"
            )

        messages.append({
            "role": "system",
            "content": "\n".join(system_parts),
        })

        # 2. Working memory (recent messages)
        for msg in self.working_memory:
            messages.append(msg)

        # 3. Current user question (if not already in working memory)
        if self.user_question:
            if not self.working_memory or self.working_memory[-1].get("content") != self.user_question:
                messages.append({
                    "role": "user",
                    "content": self.user_question,
                })

        return messages

    def estimate_tokens(self) -> int:
        """
        Rough token count estimate (chars / 4).

        Returns:
            Approximate token count.
        """
        total_chars = len(self.system_prompt) + len(self.memory_block) + len(self.context_block)
        total_chars += sum(len(m.get("content", "")) for m in self.working_memory)
        total_chars += len(self.user_question)
        return total_chars // 4


_LOCOMO_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal conversation history. "
    "Answer questions about events and facts from those conversations. "
    "When answering about dates: use natural language (8 May 2023, not 2023-05-08). "
    "Give only the essential answer. If not in history, say 'not mentioned'."
)


@dataclass
class ContextAssemblerConfig:
    """Configuration for context assembly."""
    max_memory_items: int = 15      # Max Tier 2/3 items (facts + memory_items) in prompt
    max_rag_chunks: int = 10        # Max Tier 1 chunks in prompt
    max_working_memory: int = 5     # Max recent messages (Tier 0)
    default_system_prompt: str = "You are a helpful AI assistant."
    locomo_mode: bool = False        # Use aggressive memory limits and LoCoMo system prompt


class ContextAssembler:
    """
    Assembles the final prompt from RAG results and working memory.

    Takes HybridRetriever results, separates by tier, formats
    into text blocks, and combines with system prompt and
    working memory into a complete prompt.
    """

    def __init__(self, config: Optional[ContextAssemblerConfig] = None):
        self._config = config or ContextAssemblerConfig()

    def assemble(
        self,
        hybrid_results: List[HybridResult],
        working_memory: List[Dict[str, str]],
        user_question: str,
        system_prompt: Optional[str] = None,
        max_working_memory: Optional[int] = None,
    ) -> AssembledContext:
        """
        Assemble the final context from all tiers.

        Args:
            hybrid_results: Results from HybridRetriever.search().
            working_memory: Recent messages [{role, content}, ...].
            user_question: Current user question.
            system_prompt: Custom system prompt (or uses default).
            max_working_memory: Override for the working memory cap.
                None = use self._config.max_working_memory (default=5).
                Pass len(working_memory) or a large int to include all messages.

        Returns:
            AssembledContext ready for LM generation.
        """
        prompt = system_prompt or (
            _LOCOMO_SYSTEM_PROMPT if self._config.locomo_mode
            else self._config.default_system_prompt
        )

        # Separate results by tier
        facts_items = [r for r in hybrid_results if r.tier == "memory_item"]
        chunk_items = [r for r in hybrid_results if r.tier == "rag_chunk"]

        # Truncate to configured limits (locomo_mode allows more facts)
        max_facts = 20 if self._config.locomo_mode else self._config.max_memory_items
        facts_items = facts_items[:max_facts]
        chunk_items = chunk_items[: self._config.max_rag_chunks]

        # Truncate working memory (override cap when unlimited mode is active)
        cap = max_working_memory if max_working_memory is not None else self._config.max_working_memory
        wm = working_memory[-cap:]

        # Format blocks
        memory_block = self._format_memory_items(facts_items)
        context_block = self._format_rag_chunks(chunk_items)

        return AssembledContext(
            system_prompt=prompt,
            memory_block=memory_block,
            context_block=context_block,
            working_memory=wm,
            user_question=user_question,
            memory_item_count=len(facts_items),
            rag_chunk_count=len(chunk_items),
            working_memory_count=len(wm),
        )

    def _format_memory_items(self, items: List[HybridResult]) -> str:
        """Format Tier 2/3 memory items sorted chronologically.

        Facts with known event_time are sorted oldest-first so the LLM
        can reason about temporal order. Facts without a date are appended
        after the chronological group, ordered by relevance (RRF rank).
        """
        if not items:
            return ""

        from datetime import datetime, timezone

        with_date = [item for item in items if item.valid_from]
        without_date = [item for item in items if not item.valid_from]
        with_date.sort(key=lambda x: x.valid_from or 0)
        ordered = with_date + without_date

        lines: List[str] = []
        for item in ordered:
            category = item.category or "fact"
            if item.valid_from:
                date_str = datetime.fromtimestamp(item.valid_from, tz=timezone.utc).strftime("%Y-%m-%d")
                lines.append(f"- [{category}, {date_str}] {item.content}")
            else:
                lines.append(f"- [{category}] {item.content}")

        return "\n".join(lines)

    def _format_rag_chunks(self, chunks: List[HybridResult]) -> str:
        """Format Tier 1 RAG Chunks sorted by date with [YYYY-MM-DD] prefix."""
        if not chunks:
            return ""

        from datetime import datetime, timezone

        sorted_chunks = sorted(chunks, key=lambda c: c.created_at or 0)
        lines: List[str] = []
        for chunk in sorted_chunks:
            if chunk.created_at:
                date_str = datetime.fromtimestamp(chunk.created_at, tz=timezone.utc).strftime("%Y-%m-%d")
                lines.append(f"- [{date_str}] {chunk.content}")
            else:
                lines.append(f"- {chunk.content}")

        return "\n".join(lines)


__all__ = [
    "ContextAssembler",
    "ContextAssemblerConfig",
    "AssembledContext",
]
