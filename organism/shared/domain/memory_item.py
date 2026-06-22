from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import time


@dataclass
class MemoryRecord:
    """
    Slot memory metadata — the "card" for a memory entry.

    Stores when and how the record was created/updated, its importance,
    merge count, and optional metadata.
    """
    slot_index: int
    user_id: str | None
    created_at: float  # unix timestamp
    updated_at: float  # unix timestamp
    text: str
    importance: float
    num_merges: int
    tags: list[str] | None = None
    meta: dict[str, Any] | None = None

    @classmethod
    def create_new(
        cls,
        slot_index: int,
        text: str,
        importance: float,
        user_id: str | None = None,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "MemoryRecord":
        """Create a new MemoryRecord with the current timestamp."""
        now = time.time()
        return cls(
            slot_index=slot_index,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            text=text,
            importance=importance,
            num_merges=0,
            tags=tags,
            meta=meta,
        )

    def update_on_merge(
        self,
        new_importance: float,
        new_text: str,
    ) -> None:
        """Update the record when it is merged with new information."""
        self.updated_at = time.time()
        self.num_merges += 1
        self.importance = max(self.importance, new_importance)
        self.text = new_text

    def to_dict(self) -> dict[str, Any]:
        """Serialise MemoryRecord to a dict for storage in DB / file."""
        result = {
            "slot_index": self.slot_index,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "text": self.text,
            "importance": self.importance,
            "num_merges": self.num_merges,
        }
        result["tags"] = self.tags if self.tags is not None else []
        result["meta"] = self.meta if self.meta is not None else {}
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Deserialise MemoryRecord from a dict."""
        tags_value = data.get("tags")
        tags = None if (tags_value is None or (isinstance(tags_value, list) and len(tags_value) == 0)) else list(tags_value)

        meta_value = data.get("meta")
        meta = None if (meta_value is None or (isinstance(meta_value, dict) and len(meta_value) == 0)) else dict(meta_value)

        return cls(
            slot_index=int(data["slot_index"]),
            user_id=data.get("user_id"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            text=str(data.get("text", "")),
            importance=float(data.get("importance", 0.0)),
            num_merges=int(data.get("num_merges", 0)),
            tags=tags,
            meta=meta,
        )


# SlotRetrieveResult lives in domain/experience_block.py (used in events and interactions).
# Backward-compatibility aliases: MemoryResult -> SlotRetrieveResult -> RetrieveResult
from .experience_block import SlotRetrieveResult, RetrieveResult
MemoryResult = RetrieveResult


@dataclass
class MemoryItem:
    """
    Domain model for a text memory item.

    Represents business logic only — knows nothing about the storage format.
    Mapping to/from storage is handled by functions in the storage layer.
    """
    id: int
    created_at: float  # unix timestamp
    mtype: str         # memory type (fact, preference, etc.)
    content: str
    tags: list[str]    # tag list (domain format, not the storage "tag1,tag2" string)
    user_id: Optional[str] = None
    namespace: str = "personal"


__all__ = [
    "MemoryRecord",
    "MemoryResult",  # backward-compatibility alias for RetrieveResult
    "MemoryItem",
]
