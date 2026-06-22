from __future__ import annotations

import logging
from pathlib import Path

from .base_store import BaseStore
from .message_store import MessageStore
from .memory_item_store import MemoryItemStore
from .session_store import SessionStore
from .fact_store import FactStore

logger = logging.getLogger(__name__)


class UnifiedStore:
    """
    Composition-based unified store for the Core layer.

    Aggregates all sub-stores over a single BaseStore (one SQLite DB).
    Each sub-store handles a specific table/concern but shares
    the same connection pool.
    """

    def __init__(self, db_path: Path):
        from organism.core.memory.rag.chunk_store import ChunkStore  # lazy — avoids circular import

        self._base = BaseStore(db_path)

        self.messages = MessageStore(self._base)
        self.memory_items = MemoryItemStore(self._base)
        self.sessions = SessionStore(self._base)
        self.chunks = ChunkStore(self._base)
        self.facts = FactStore(self._base)

    @property
    def base(self) -> BaseStore:
        return self._base

    def close(self) -> None:
        self._base.close()


__all__ = ["UnifiedStore"]
