from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MergePolicy:
    """
    Policy for merging similar memories.

    Used for deduplication: when two memories are very similar they may be
    merged into one.
    """
    # Cosine similarity threshold for merge
    sim_threshold: float = 0.85

    # Effective-importance parameters accounting for recency
    freshness_tau: float = 14.0    # τ: freshness half-life in days
    freshness_alpha: float = 0.15  # α: freshness weight in effective importance

    # Field-overwrite thresholds during merge
    overwrite_score_margin: float = 0.05        # minimum score margin to overwrite
    overwrite_importance_delta: float = 0.1     # overwrite importance only if newer and above delta


@dataclass
class PrunePolicy:
    """
    Policy for pruning old/unimportant memories.

    Determines when and which memories can be removed.
    """
    # TODO: add prune parameters once the feature is implemented
    pass


@dataclass
class MemoriesPolicy:
    """
    Top-level policy for long-term memory management.

    Contains all sub-policies (merge, prune) and shared rules.
    """
    merge: MergePolicy = field(default_factory=MergePolicy)
    prune: PrunePolicy = field(default_factory=PrunePolicy)

    # Metadata fields that can be overwritten when other is newer and sufficiently important
    merge_overwrite_fields: set[str] = field(default_factory=lambda: {"source", "original", "kind", "model_kind"})

    # Metadata fields to union-merge (lists/sets); tags are handled separately
    merge_merge_fields: set[str] = field(default_factory=lambda: {"source_event_ids"})


# Default policy (used when no config is provided)
DEFAULT_POLICY = MemoriesPolicy()
