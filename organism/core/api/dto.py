from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OrganismReply:
    """Response from Organism.chat()."""
    reply: str
    used_memories: List[str] = field(default_factory=list)


__all__ = ["OrganismReply"]
