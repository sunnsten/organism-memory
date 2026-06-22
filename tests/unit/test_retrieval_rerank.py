import time

import pytest

from organism.core.memory.rag.hybrid_retriever import HybridResult
from organism.core.memory.service.retrieval_service import _rerank


def _h(rrf: float, importance: float, age_days: float = 0.0) -> HybridResult:
    created_at = int(time.time()) - int(age_days * 86400)
    return HybridResult(
        id=1,
        content="x",
        rrf_score=rrf,
        tier="memory_item",
        sources=["fts"],
        importance=importance,
        created_at=created_at,
    )


def test_rerank_prefers_high_importance():
    """Item with higher importance beats slightly higher RRF score."""
    low = _h(rrf=1.0, importance=0.1)
    high = _h(rrf=0.9, importance=0.9)
    ranked = _rerank([low, high])
    assert ranked[0] is high


def test_rerank_prefers_higher_rrf():
    """With recency removed (gamma=0), higher RRF wins regardless of age."""
    old = _h(rrf=1.0, importance=0.5, age_days=365)
    new = _h(rrf=0.9, importance=0.5, age_days=1)
    ranked = _rerank([old, new])
    assert ranked[0] is old


def test_rerank_empty_list():
    assert _rerank([]) == []


def test_rerank_single_item():
    r = _h(rrf=0.5, importance=0.5)
    assert _rerank([r]) == [r]


def test_rerank_preserves_all_items():
    items = [_h(rrf=float(i), importance=0.5) for i in range(5)]
    ranked = _rerank(items)
    assert len(ranked) == 5
    assert set(id(x) for x in ranked) == set(id(x) for x in items)
