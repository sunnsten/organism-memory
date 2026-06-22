from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

# WorkingMemoryPack is a runtime structure used during retrieval.
# At runtime ssm_state is always a Tensor (from SimplePersistentSSM.get_state()).
# Serialisation (Tensor → bytes) happens in the storage layer (SSMStateStore), not here.
from torch import Tensor


@dataclass
class WorkingMemoryPack:
    """
    Working memory (SSM / RAM state) as a first-class entity.

    Holds the working memory state that becomes part of the MemoryModel
    and the retrieval result, allowing explicit tracking and use of working
    memory alongside long-term memory.
    """
    ssm_state: Optional[Tensor]              # SSM state [D_state] or [B, D_state]
    short_summary: Optional[str] = None      # brief summary of the current context (if any)
    recent_refs: list[str] = field(default_factory=list)   # chat span IDs (not full texts)
    trace: dict[str, Any] = field(default_factory=dict)    # debug info about working memory use

    @classmethod
    def empty(cls) -> "WorkingMemoryPack":
        """Create an empty working memory pack."""
        return cls(ssm_state=None)


__all__ = [
    "WorkingMemoryPack",
]
