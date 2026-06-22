from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class RetrievalTrace:
    """
    Trace of the retrieval process for debugging and metrics.

    Standardises the trace structure that was previously an untyped dict[str, Any].
    """
    query: str                # original query
    top_k: int                # requested result count
    slot_results_count: int = 0             # results from MemoryCore (slots)
    db_results_count: int = 0               # results from SQLite (FTS)
    db_result_ids: List[int] = field(default_factory=list)  # IDs of SQLite records found

    # Additional metadata for future extensions
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RetrievalTrace",
]
