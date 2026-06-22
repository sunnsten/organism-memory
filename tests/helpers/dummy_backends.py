from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from organism.shared.domain import (
    ChatMessage,
    MemoryItem,
)


class DummyPersonalStoreBackend:
    def __init__(self):
        from organism.config import OrganismConfig
        self.config = OrganismConfig()
        self._sessions: Dict[str, List[str]] = {}
        self._chat_messages: List[ChatMessage] = []
        self._memories: List[MemoryItem] = []
        self._next_id = 1

    def create_session(self, user_id: str, title: Optional[str] = None) -> str:
        session_id = f"session_{self._next_id}"
        self._next_id += 1
        self._sessions.setdefault(user_id, []).append(session_id)
        return session_id

    def end_session(self, user_id: str, session_id: str) -> None:
        if user_id in self._sessions and session_id in self._sessions[user_id]:
            self._sessions[user_id].remove(session_id)

    def add_chat_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> int:
        from datetime import datetime, timezone
        msg_id = self._next_id
        self._next_id += 1
        self._chat_messages.append(ChatMessage(
            id=msg_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            role=role,
            content=content,
        ))
        return msg_id

    def get_recent_messages(
        self,
        user_id: str,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        return self._chat_messages[-limit:]

    def clear_chat_log(self, user_id: Optional[str] = None) -> None:
        self._chat_messages.clear()

    def add_memory(
        self,
        content: str,
        mtype: str,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        namespace: str = "personal",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[Tensor] = None,
        *,
        commit: bool = True,
        conn: Optional[Any] = None,
        upsert_on_conflict: bool = False,
        **kwargs: Any,
    ) -> int:
        import time
        mem_id = self._next_id
        self._next_id += 1
        self._memories.append(MemoryItem(
            id=mem_id,
            created_at=time.time(),
            mtype=mtype,
            content=content,
            tags=tags or [],
            user_id=user_id,
        ))
        return mem_id

    def list_memories(
        self,
        limit: int = 50,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[MemoryItem]:
        return self._memories[-limit:]

    def find_memory_id_by_text(
        self,
        text: str,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        for mem in self._memories:
            if mem.content == text:
                return mem.id
        return None


__all__ = ["DummyPersonalStoreBackend"]
