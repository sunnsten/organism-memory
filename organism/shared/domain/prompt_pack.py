from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .working_memory import WorkingMemoryPack
from .experience_block import SlotRetrieveResult
from .memory_item import MemoryItem
from .retrieval_trace import RetrievalTrace


@dataclass
class PromptMemoryPack:
    """
    Packed memory ready for prompt injection.

    Contains:
    - working: working memory (SSM / RAM) as a first-class entity
    - db: curated LTM (long-term memory from SQLite)
    - slots: search results from MemoryCore (slot-based memory)
    - rendered: pre-rendered text for prompt injection
    - trace: retrieval process information for debugging
    """
    working: WorkingMemoryPack          # working memory (always present, even if empty)
    db: Sequence[MemoryItem]            # curated LTM from SQLite
    slots: Sequence[SlotRetrieveResult] # MemoryCore retrieve results
    rendered: str                       # pre-rendered text for prompt injection
    trace: RetrievalTrace               # retrieval info for debugging and metrics


__all__ = [
    "PromptMemoryPack",
]
