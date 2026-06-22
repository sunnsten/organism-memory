from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

facts_extracted = Counter(
    "organism_facts_extracted_total",
    "Facts extracted from conversations (all candidates before dedup)",
)
facts_new = Counter(
    "organism_facts_new_total",
    "New facts inserted (not duplicates)",
)
facts_confirmed = Counter(
    "organism_facts_confirmed_total",
    "Existing facts confirmed (duplicates / cosine match)",
)
facts_invalidated = Counter(
    "organism_facts_invalidated_total",
    "Facts invalidated by supersession (knowledge update)",
)
facts_errors = Counter(
    "organism_facts_extraction_errors_total",
    "Fact extraction LLM call or parse failures",
)
facts_latency = Histogram(
    "organism_facts_extraction_latency_seconds",
    "Fact extraction LLM call duration per session",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

retrieval_facts_returned = Histogram(
    "organism_retrieval_facts_returned",
    "Facts injected into context per retrieval call",
    buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
)
retrieval_latency = Histogram(
    "organism_retrieval_latency_seconds",
    "End-to-end fact retrieval duration",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

profile_keys_updated = Counter(
    "organism_profile_keys_updated_total",
    "Profile keys upserted by ProfileUpdater",
    labelnames=["key"],
)

# ---------------------------------------------------------------------------
# Thread-safe in-process accumulators for eval snapshots
# ---------------------------------------------------------------------------

@dataclass
class _Accumulators:
    facts_extracted: int = 0
    facts_new: int = 0
    facts_confirmed: int = 0
    facts_invalidated: int = 0
    facts_errors: int = 0
    facts_latency_total_s: float = 0.0
    facts_latency_calls: int = 0

    retrieval_calls: int = 0
    retrieval_facts_total: int = 0
    retrieval_latency_total_s: float = 0.0

    profile_updates: int = 0


_acc = _Accumulators()
_acc_lock = threading.Lock()


def _inc(field: str, amount: float = 1.0) -> None:
    with _acc_lock:
        setattr(_acc, field, getattr(_acc, field) + amount)


# ---------------------------------------------------------------------------
# Public helpers called from instrumentation points
# ---------------------------------------------------------------------------

def record_facts_extracted(n: int) -> None:
    facts_extracted.inc(n)
    _inc("facts_extracted", n)


def record_fact_new() -> None:
    facts_new.inc()
    _inc("facts_new")


def record_fact_confirmed() -> None:
    facts_confirmed.inc()
    _inc("facts_confirmed")


def record_fact_invalidated() -> None:
    facts_invalidated.inc()
    _inc("facts_invalidated")


def record_extraction_error() -> None:
    facts_errors.inc()
    _inc("facts_errors")


def record_extraction_latency(seconds: float) -> None:
    facts_latency.observe(seconds)
    _inc("facts_latency_total_s", seconds)
    _inc("facts_latency_calls")


def record_retrieval(n_returned: int, latency_s: float) -> None:
    retrieval_facts_returned.observe(n_returned)
    retrieval_latency.observe(latency_s)
    _inc("retrieval_calls")
    _inc("retrieval_facts_total", n_returned)
    _inc("retrieval_latency_total_s", latency_s)


def record_profile_update(key: str) -> None:
    profile_keys_updated.labels(key=key).inc()
    _inc("profile_updates")


# ---------------------------------------------------------------------------
# Snapshot API for eval runner
# ---------------------------------------------------------------------------

@dataclass
class MemoryMetricsSnapshot:
    facts_extracted: int
    facts_new: int
    facts_confirmed: int
    facts_invalidated: int
    facts_errors: int
    facts_latency_avg_s: float   # 0 if no calls
    retrieval_calls: int
    retrieval_facts_avg: float   # avg facts returned per call; 0 if no calls
    retrieval_latency_avg_s: float
    profile_updates: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def delta(self, baseline: "MemoryMetricsSnapshot") -> "MemoryMetricsSnapshot":
        """Return the difference (self − baseline) as a new snapshot."""
        calls_ext = max(1, self.facts_extracted - baseline.facts_extracted)
        r_calls = max(1, self.retrieval_calls - baseline.retrieval_calls)
        return MemoryMetricsSnapshot(
            facts_extracted=self.facts_extracted - baseline.facts_extracted,
            facts_new=self.facts_new - baseline.facts_new,
            facts_confirmed=self.facts_confirmed - baseline.facts_confirmed,
            facts_invalidated=self.facts_invalidated - baseline.facts_invalidated,
            facts_errors=self.facts_errors - baseline.facts_errors,
            facts_latency_avg_s=(
                (self.facts_latency_avg_s * (self.facts_extracted or 1))
                - (baseline.facts_latency_avg_s * (baseline.facts_extracted or 1))
            ) / calls_ext,
            retrieval_calls=self.retrieval_calls - baseline.retrieval_calls,
            retrieval_facts_avg=(
                (self.retrieval_facts_avg * (self.retrieval_calls or 1))
                - (baseline.retrieval_facts_avg * (baseline.retrieval_calls or 1))
            ) / r_calls,
            retrieval_latency_avg_s=(
                (self.retrieval_latency_avg_s * (self.retrieval_calls or 1))
                - (baseline.retrieval_latency_avg_s * (baseline.retrieval_calls or 1))
            ) / r_calls,
            profile_updates=self.profile_updates - baseline.profile_updates,
        )


def take_snapshot() -> MemoryMetricsSnapshot:
    """Read current accumulator values as an immutable snapshot."""
    with _acc_lock:
        latency_avg = (
            _acc.facts_latency_total_s / _acc.facts_latency_calls
            if _acc.facts_latency_calls > 0 else 0.0
        )
        r_avg_facts = (
            _acc.retrieval_facts_total / _acc.retrieval_calls
            if _acc.retrieval_calls > 0 else 0.0
        )
        r_avg_lat = (
            _acc.retrieval_latency_total_s / _acc.retrieval_calls
            if _acc.retrieval_calls > 0 else 0.0
        )
        return MemoryMetricsSnapshot(
            facts_extracted=_acc.facts_extracted,
            facts_new=_acc.facts_new,
            facts_confirmed=_acc.facts_confirmed,
            facts_invalidated=_acc.facts_invalidated,
            facts_errors=_acc.facts_errors,
            facts_latency_avg_s=latency_avg,
            retrieval_calls=_acc.retrieval_calls,
            retrieval_facts_avg=r_avg_facts,
            retrieval_latency_avg_s=r_avg_lat,
            profile_updates=_acc.profile_updates,
        )


__all__ = [
    "record_facts_extracted",
    "record_fact_new",
    "record_fact_confirmed",
    "record_fact_invalidated",
    "record_extraction_error",
    "record_extraction_latency",
    "record_retrieval",
    "record_profile_update",
    "take_snapshot",
    "MemoryMetricsSnapshot",
]
