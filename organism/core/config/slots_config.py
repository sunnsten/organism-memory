from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImportanceWeights:
    """Weights for computing episode importance."""
    attn: float = 0.4
    surprisal: float = 0.3
    length: float = 0.2
    mem: float = 0.1


@dataclass
class MemoryGatingConfig:
    """Configuration for the memory gating mechanism."""
    decay_lambda: float = 0.99
    read_bump: float = 0.1
    write_bump: float = 1.0
    default_surprisal: float = 0.5
    merge_old_coef: float = 0.7
    merge_new_coef: float = 0.3
    fill_low: float = 0.5
    fill_mid: float = 0.8
    mid_max_threshold: float = 0.85
    high_max_threshold: float = 0.80
    surprisal_factor_min: float = 0.3
    surprisal_factor_max: float = 1.0
    # Weights for combining query and SSM context in retrieve()
    query_weight: float = 0.7
    ssm_context_weight: float = 0.3
    # EMA for surprisal normalization
    ema_surprisal_init: float = 5.0
    ema_alpha: float = 0.01
    ema_floor: float = 1.0
    surprisal_norm_divisor: float = 2.0
    # Text length normalization factor for importance
    text_length_norm: float = 200.0
    # mem_factor normalization for importance
    mem_factor_max: float = 3.0


@dataclass
class SlotsConfig:
    """Configuration for neural memory slots (MemoryCore)."""

    # Dimensions
    d_state: int = 512             # SSM hidden state size
    d_compressed: int = 256        # Compressed representation size
    memory_size: int = 1000        # Max number of slots in RAM
    gate_hidden_size: int = 64     # Hidden layer size of gating MLP
    memory_threshold: float = 0.5  # Threshold for promotion to long-term memory

    # Gating
    gating: MemoryGatingConfig = field(default_factory=MemoryGatingConfig)

    # Importance
    importance_weights: ImportanceWeights = field(default_factory=ImportanceWeights)

    # Merge / deduplication
    merge_sim_threshold: float = 0.85   # Cosine similarity threshold for merge
    enable_value_merge: bool = True      # Enable merge on memory_vals
    max_slot_text_len: int = 200         # Max text length per slot
    merge_limit: int = 100               # Max slots to check during merge

    # Retrieval
    retrieve_top_k: int = 4              # Default top-k for slot retrieve
    enable_retrieve_slots: bool = True   # Enable slot retrieval

    # Persistence
    persist_template: str = "weights/memory_core_{user}.pt"
    autosave_every: int = 10             # Autosave every N interactions

    # Passive observer (logs signals, never affects responses)
    observer_enabled: bool = False


__all__ = [
    "SlotsConfig",
    "MemoryGatingConfig",
    "ImportanceWeights",
]
