from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List

# Lightweight structures for logs (no Tensor dependencies)
@dataclass
class RetrievedRef:
    """Reference to a memory search result (for logs, no Tensor)."""
    slot_index: int
    score: float
    text: str


@dataclass
class MemoryWrittenRef:
    """Reference to a written memory entry (for logs, no full metadata)."""
    slot_index: int
    importance: float
    text: str


@dataclass
class ChatMessage:
    """A message from the chat history."""
    id: int
    created_at: str
    role: str
    content: str


@dataclass
class InteractionLog:
    """
    Interaction log for a user–system exchange.

    Logs all interactions (chat, /remember, etc.) and can be saved to
    ExperienceStore for training.
    """
    user_id: str
    session_id: str
    turn_id: int
    role: str     # "user" / "assistant"
    text: str
    timestamp: float
    retrieved: Optional[List[RetrievedRef]] = None
    memory_written: Optional[List[MemoryWrittenRef]] = None
    meta: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict for JSONL / DB storage."""
        from datetime import datetime, timezone

        entry: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "text": self.text,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }

        if self.retrieved:
            entry["retrieved"] = [
                {"slot_index": r.slot_index, "score": r.score, "text": r.text}
                for r in self.retrieved
            ]

        if self.memory_written:
            entry["memory_written"] = [
                {"slot_index": r.slot_index, "importance": r.importance, "text": r.text}
                for r in self.memory_written
            ]

        if self.meta:
            entry["metadata"] = self.meta

        return entry

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionLog":
        """
        Deserialise from a dict.

        Note: retrieved and memory_written are restored partially (no Tensor vectors)
        since they are not persisted in JSONL.
        """
        import time

        retrieved = None
        if "retrieved" in data and data["retrieved"]:
            retrieved = [
                RetrievedRef(
                    slot_index=int(r_data["slot_index"]),
                    score=float(r_data.get("score", 0.0)),
                    text=str(r_data.get("text", "")),
                )
                for r_data in data["retrieved"]
            ]

        memory_written = None
        if "memory_written" in data and data["memory_written"]:
            memory_written = [
                MemoryWrittenRef(
                    slot_index=int(m_data["slot_index"]),
                    importance=float(m_data.get("importance", 0.0)),
                    text=str(m_data.get("text", "")),
                )
                for m_data in data["memory_written"]
            ]

        # Support both "meta" and "metadata" for backward compatibility
        meta = data.get("meta") or data.get("metadata")

        timestamp_value = data.get("timestamp", time.time())
        if isinstance(timestamp_value, str):
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except (ValueError, AttributeError):
                timestamp = time.time()
        else:
            timestamp = float(timestamp_value)

        return cls(
            user_id=str(data["user_id"]),
            session_id=str(data.get("session_id", "")),
            turn_id=int(data.get("turn_id", 0)),
            role=str(data.get("role", "user")),
            text=str(data.get("text", "")),
            timestamp=timestamp,
            retrieved=retrieved,
            memory_written=memory_written,
            meta=dict(meta) if meta else None,
        )


__all__ = [
    "RetrievedRef",
    "MemoryWrittenRef",
    "ChatMessage",
    "InteractionLog",
]
