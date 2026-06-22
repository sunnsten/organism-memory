from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from organism.core.stores.base_store import BaseStore
from organism.core.stores.schema import init_schema


@pytest.fixture
def base(tmp_path: Path) -> BaseStore:
    store = BaseStore(tmp_path / "hnsw_test.db")
    return store


def _emb(val: float, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(val)
    v[1] = 1.0 - float(val)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def test_table_name_is_stable_and_sql_safe(base):
    """Same tenant/user always produces the same table name with no special chars."""
    from organism.core.stores.per_user_hnsw import PerUserHNSW
    hnsw = PerUserHNSW(base, dim=8)
    name = hnsw.table_name("tenant-1", "user@domain.com")
    assert name.isidentifier() or all(c.isalnum() or c == '_' for c in name)
    assert hnsw.table_name("tenant-1", "user@domain.com") == name  # stable


def test_insert_and_search_returns_nearest(base):
    """Insert 5 vectors, search returns nearest by cosine."""
    from organism.core.stores.per_user_hnsw import PerUserHNSW
    hnsw = PerUserHNSW(base, dim=8)

    embs = [_emb(i * 0.2) for i in range(5)]
    for i, emb in enumerate(embs):
        hnsw.insert(rowid=i + 1, embedding=emb, tenant_id="t1", user_id="u1")

    results = hnsw.search(query=embs[2], tenant_id="t1", user_id="u1", limit=3)
    assert results, "Should return results"
    rowids = [r[0] for r in results]
    assert 3 in rowids, "Exact match (rowid=3) should be in top results"


def test_delete_removes_from_search(base):
    """Deleted rowid no longer appears in search results."""
    from organism.core.stores.per_user_hnsw import PerUserHNSW
    hnsw = PerUserHNSW(base, dim=8)

    emb = _emb(0.5)
    hnsw.insert(rowid=10, embedding=emb, tenant_id="t1", user_id="u1")
    hnsw.delete(rowid=10, tenant_id="t1", user_id="u1")

    results = hnsw.search(query=emb, tenant_id="t1", user_id="u1", limit=5)
    rowids = [r[0] for r in results]
    assert 10 not in rowids, "Deleted rowid must not appear in results"


def test_users_are_isolated(base):
    """Vectors from user A do not appear in user B's results."""
    from organism.core.stores.per_user_hnsw import PerUserHNSW
    hnsw = PerUserHNSW(base, dim=8)

    emb_a = _emb(0.1)
    emb_b = _emb(0.9)

    hnsw.insert(rowid=1, embedding=emb_a, tenant_id="t1", user_id="alice")
    hnsw.insert(rowid=2, embedding=emb_b, tenant_id="t1", user_id="bob")

    results_alice = hnsw.search(emb_a, "t1", "alice", limit=5)
    results_bob = hnsw.search(emb_b, "t1", "bob", limit=5)

    assert all(r[0] == 1 for r in results_alice), "Alice only sees her own vectors"
    assert all(r[0] == 2 for r in results_bob), "Bob only sees his own vectors"


def test_passthrough_when_vectorlite_unavailable(base, monkeypatch):
    """When vectorlite is not available, search() returns empty (caller falls back to Python)."""
    from organism.core.stores import per_user_hnsw as mod
    monkeypatch.setattr(mod, "VECTORLITE_AVAILABLE", False)

    from organism.core.stores.per_user_hnsw import PerUserHNSW
    hnsw = PerUserHNSW(base, dim=8)
    results = hnsw.search(_emb(0.5), "t1", "u1", limit=5)
    assert results == []
