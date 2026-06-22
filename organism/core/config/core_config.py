from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoreConfig:
    """Configuration for Core layer online operations (WriteService, WorkingMemoryService, etc.)."""

    # Single DB file shared across all backends
    db_path: str = "organism_data/organism.db"

    # Write-side importance gating
    block_min_importance: float = 0.1
    source_multiplier_remember: float = 0.5   # remember: lower threshold → higher priority
    source_multiplier_chat: float = 1.0        # chat: standard threshold
    source_multiplier_manual: float = 0.0      # manual: zero threshold → always write

    # Event recording toggle
    enable_write_events: bool = True

    # Working memory defaults
    working_memory_recent_k: int = 5


__all__ = ["CoreConfig"]
