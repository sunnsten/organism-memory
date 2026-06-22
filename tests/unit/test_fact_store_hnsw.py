from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from organism.core.stores import UnifiedStore


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "fact_hnsw_test.db")


def _emb(val: float, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(val)
    v[1] = 1.0 - float(val)
    return v / np.linalg.norm(v)


def test_search_vector_returns_nearest_fact(store):
    """search_vector returns the most similar active fact."""
    emb_target = _emb(0.2)
    emb_other = _emb(0.9)

    store.facts.add("t1", "u1", "Target fact", embedding=emb_target)
    store.facts.add("t1", "u1", "Unrelated fact", embedding=emb_other)

    results = store.facts.search_vector(emb_target, "t1", "u1", limit=1)
    assert results[0]["content"] == "Target fact"


def test_invalidated_fact_excluded_from_search(store):
    """Facts with valid_until set do not appear in search_vector."""
    emb = _emb(0.5)
    fid = store.facts.add("t1", "u1", "Old location", embedding=emb)
    store.facts.invalidate(fid)

    results = store.facts.search_vector(emb, "t1", "u1", limit=5)
    ids = [r["id"] for r in results]
    assert fid not in ids


def test_users_facts_are_isolated(store):
    """User A's facts do not appear in user B's vector search."""
    emb = _emb(0.5)
    store.facts.add("t1", "alice", "Alice's fact", embedding=emb)
    store.facts.add("t1", "bob", "Bob's fact", embedding=_emb(0.51))

    results = store.facts.search_vector(emb, "t1", "alice", limit=5)
    contents = [r["content"] for r in results]
    assert "Alice's fact" in contents
    assert "Bob's fact" not in contents
