from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AttentionTrace:
    """Attention metrics collected during generation."""
    attention_focus: float                              # entropy of the last-token attention vector
    mem_attention_scores: Dict[int, float] = field(default_factory=dict)  # {mem_id: score}
    attn_entropy_mean: Optional[float] = None           # mean entropy across all generated tokens


__all__ = [
    "AttentionTrace",
]
