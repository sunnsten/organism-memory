from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from organism.core.config import CoreConfig
    from organism.core.memory.service.retrieval_service import RetrievalService
    from organism.core.memory.service.working_memory_service import WorkingMemoryService
    from organism.core.memory.service.write_service import WriteService
    from organism.core.stores import UnifiedStore
    from organism.shared.domain import EventRecord, PromptMemoryPack, WorkingMemoryPack

from organism.core.memory.service.reports import MemoryDebugView

logger = logging.getLogger(__name__)


class MemoryFacade:
    """
    Thin facade satisfying MemoryService protocol via sub-service delegation.

    Usage:
        facade = MemoryFacade(
            retrieval=retrieval_service,
            write=write_service,
            working=working_memory_service,
            store=unified_store,
            tenant_id="tenant1",
        )
        event_id = facade.append_event(event)
        wm = facade.get_working_memory(user_id="u1", session_id="s1")
    """

    def __init__(
        self,
        retrieval: "RetrievalService",
        write: "WriteService",
        working: "WorkingMemoryService",
        store: "UnifiedStore",
        tenant_id: str,
    ):
        self._retrieval = retrieval
        self._write = write
        self._working = working
        self._store = store
        self._tenant_id = tenant_id

    # --- Public properties ---

    @property
    def store(self) -> "UnifiedStore":
        return self._store

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def retrieval(self) -> "RetrievalService":
        return self._retrieval

    @property
    def _last_trace(self):
        """Proxy to RetrievalService._last_trace — used by EvalAdapter."""
        return self._retrieval._last_trace

    @property
    def write(self) -> "WriteService":
        return self._write

    @property
    def working(self) -> "WorkingMemoryService":
        return self._working

    @classmethod
    def from_store(
        cls,
        store: "UnifiedStore",
        tenant_id: str = "default",
        embedder=None,
        core_config: Optional["CoreConfig"] = None,
    ) -> "MemoryFacade":
        """
        Convenience factory: builds sub-services from a UnifiedStore.

        Args:
            store: UnifiedStore instance.
            tenant_id: Tenant identifier for multi-tenant setups.
            embedder: Optional embedder for vector search (e.g. Qwen3Embedder).
                      If None, vector search is disabled (FTS-only mode).
            core_config: Optional CoreConfig override. Defaults to CoreConfig().

        Returns:
            Fully wired MemoryFacade.
        """
        from organism.core.config import CoreConfig
        from organism.core.memory.service.write_service import WriteService
        from organism.core.memory.service.working_memory_service import WorkingMemoryService
        from organism.core.memory.service.retrieval_service import RetrievalService

        cfg = core_config if core_config is not None else CoreConfig()
        write = WriteService(store=store, config=cfg, embedder=embedder)
        working = WorkingMemoryService(store=store, config=cfg)
        chunk_store = store.chunks
        fact_store = store.facts
        retrieval = RetrievalService(
            message_store=store.messages,
            memory_item_store=store.memory_items,
            chunk_store=chunk_store,
            embedder=embedder,
            fact_store=fact_store,
            rag_config=getattr(cfg, "rag", None),
        )
        return cls(
            retrieval=retrieval,
            write=write,
            working=working,
            store=store,
            tenant_id=tenant_id,
        )

    # --- MemoryService protocol methods ---

    def append_event(self, event: "EventRecord") -> int:
        """
        Delegates to WriteService.append_event().

        Returns:
            int: Positive value if stored, 0 if filtered.
        """
        result = self._write.append_event(event, tenant_id=self._tenant_id)
        if result is None:
            return 0  # Filtered by importance threshold
        return 1  # Stored

    def get_working_memory(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> "WorkingMemoryPack":
        """Delegates to WorkingMemoryService.get_working_memory()."""
        return self._working.get_working_memory(
            tenant_id=self._tenant_id,
            user_id=user_id,
            session_id=session_id or "",
        )

    def get_debug_view(
        self,
        user_id: str,
        *,
        last_n: int = 50,
    ) -> MemoryDebugView:
        """
        Builds debug view by querying sub-stores directly.
        """
        # Memories (memory items)
        memories_count = 0
        recent_memories: list = []
        try:
            items = self._store.memory_items.search_fts(
                self._tenant_id, user_id, "", limit=last_n,
            )
            memories_count = len(items)
            recent_memories = items[:last_n]
        except Exception as e:
            logger.warning("get_debug_view: memory_items query failed: %s", e)

        return MemoryDebugView(
            memories_count=memories_count,
            recent_memories=recent_memories,
        )


__all__ = ["MemoryFacade"]
