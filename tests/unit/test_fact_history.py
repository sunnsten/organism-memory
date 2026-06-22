from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from organism.core.stores import UnifiedStore


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "history_test.db")


def _emb(angle: float, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle)
    v[1] = np.sin(angle)
    return v / np.linalg.norm(v)


def test_supersede_tombstones_and_links(store):
    """supersede(old_id, new_id) tombstones old and sets superseded_by_id."""
    old_id = store.facts.add("t1", "u1", "User lives in New York", embedding=_emb(0.0))
    new_id = store.facts.add("t1", "u1", "User moved to Seattle", embedding=_emb(0.5))

    store.facts.supersede(old_id, new_id)

    old = store.facts.get(old_id)
    assert old["valid_until"] is not None, "Old fact must be tombstoned"
    assert old["superseded_by_id"] == new_id, "Must link to replacement"


def test_history_chain_returns_full_sequence(store):
    """get_history_chain on latest fact returns [oldest, ..., current]."""
    id_a = store.facts.add("t1", "u1", "User lives in London", embedding=_emb(0.0))
    id_b = store.facts.add("t1", "u1", "User moved to Berlin", embedding=_emb(0.4))
    id_c = store.facts.add("t1", "u1", "User moved to Seattle", embedding=_emb(0.8))

    store.facts.supersede(id_a, id_b)
    store.facts.supersede(id_b, id_c)

    chain = store.facts.get_history_chain(id_c)
    assert len(chain) == 3
    assert chain[0]["id"] == id_a   # oldest first
    assert chain[-1]["id"] == id_c  # current last


def test_add_or_supersede_creates_history_link(store):
    """add_or_supersede sets superseded_by_id when cosine is in supersede range."""
    old_emb = _emb(0.0)
    new_emb = _emb(0.52)  # cosine ≈ 0.87 — supersede range

    old_id = store.facts.add("t1", "u1", "User lives in New York", embedding=old_emb)
    new_id, is_new = store.facts.add_or_supersede(
        "t1", "u1", "User moved to Seattle",
        old_embedding=new_emb, new_embedding=new_emb,
    )

    assert is_new
    old = store.facts.get(old_id)
    assert old["superseded_by_id"] == new_id


def test_tombstoned_excluded_from_normal_search(store):
    """Superseded facts do not appear in search_fts or search_vector."""
    old_id = store.facts.add("t1", "u1", "User lives in New York", embedding=_emb(0.0))
    new_id = store.facts.add("t1", "u1", "User moved to Seattle", embedding=_emb(0.5))
    store.facts.supersede(old_id, new_id)

    fts = store.facts.search_fts("New York", "t1", "u1")
    assert not any(r["id"] == old_id for r in fts)

    vec = store.facts.search_vector(_emb(0.0), "t1", "u1", limit=5)
    assert not any(r["id"] == old_id for r in vec)


def test_event_time_stored_and_retrieved(store):
    """event_time is persisted and returned by get()."""
    import time
    ts = int(time.time()) - 86400  # yesterday
    fact_id = store.facts.add(
        "t1", "u1", "User worked in Berlin",
        embedding=_emb(0.0),
        event_time=ts,
    )
    row = store.facts.get(fact_id)
    assert row["event_time"] == ts


def test_event_time_none_by_default(store):
    """event_time is None when not provided."""
    fact_id = store.facts.add("t1", "u1", "User likes coffee", embedding=_emb(0.2))
    row = store.facts.get(fact_id)
    assert row["event_time"] is None
