from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MemoryDebugView:
    """Debug snapshot of memory state."""
    memories_count: int
    recent_memories: List[Dict[str, Any]]


__all__ = ["MemoryDebugView"]
