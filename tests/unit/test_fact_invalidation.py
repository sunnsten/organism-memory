from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from organism.core.stores import UnifiedStore


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "inv_test.db")


def _emb(angle: float, dim: int = 16) -> np.ndarray:
    """Return a normalised embedding where angle controls direction."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle)
    v[1] = np.sin(angle)
    return v / np.linalg.norm(v)


def test_high_similarity_confirms_existing(store):
    """cosine > 0.90 → confirm existing, no new row."""
    emb = _emb(0.0)
    fid = store.facts.add("t1", "u1", "User lives in Berlin", embedding=emb)
    before = store.facts.get(fid)["confirmed_count"]

    # Nearly identical embedding → should confirm, not add new row
    store.facts.add_or_supersede(
        "t1", "u1", "User lives in Berlin.",
        old_embedding=emb, new_embedding=emb,
    )
    after = store.facts.get(fid)["confirmed_count"]

    assert after == before + 1
    cur = store._base.execute(
        "SELECT COUNT(*) as n FROM facts WHERE tenant_id='t1' AND user_id='u1'"
    )
    assert cur.fetchone()["n"] == 1


def test_medium_similarity_invalidates_old_and_adds_new(store):
    """0.70 < cosine ≤ 0.90 → old fact gets valid_until set, new fact inserted."""
    old_emb = _emb(0.0)        # angle=0
    new_emb = _emb(0.52)       # cosine(0, 0.52) = cos(0.52) ≈ 0.868 — in (0.70, 0.90]

    old_id = store.facts.add("t1", "u1", "User lives in New York", embedding=old_emb)

    new_id, is_new = store.facts.add_or_supersede(
        "t1", "u1", "User moved to Seattle",
        old_embedding=new_emb,
        new_embedding=new_emb,
    )

    assert is_new, "Supersede must insert a new row"
    old_row = store.facts.get(old_id)
    assert old_row["valid_until"] is not None, "Old fact must be invalidated"
    assert new_id != old_id, "New fact must be a separate row"

    # New fact must appear in search; old must not
    results = store.facts.search_fts("Seattle", "t1", "u1")
    assert any(r["id"] == new_id for r in results)
    results_ny = store.facts.search_fts("New York", "t1", "u1")
    assert not any(r["id"] == old_id for r in results_ny), "Invalidated fact must not appear in FTS"


def test_low_similarity_adds_new_fact(store):
    """cosine ≤ 0.70 → unrelated facts, both kept, old not invalidated."""
    emb_a = _emb(0.0)
    emb_b = _emb(1.3)  # cosine(0, 1.3) = cos(1.3) ≈ 0.268 — different topic

    id_a = store.facts.add("t1", "u1", "User is a software engineer", embedding=emb_a)
    id_b, is_new = store.facts.add_or_supersede(
        "t1", "u1", "User enjoys hiking",
        old_embedding=emb_b,
        new_embedding=emb_b,
    )

    assert is_new
    assert id_a != id_b
    old = store.facts.get(id_a)
    assert old["valid_until"] is None, "Unrelated fact must not be invalidated"


def test_invalidated_fact_not_rematched(store):
    """An already-invalidated fact must not be re-matched in a subsequent add_or_supersede call."""
    emb_a = _emb(0.0)
    emb_b = _emb(0.52)  # cosine ≈ 0.868 — triggers supersede of emb_a
    emb_c = _emb(0.52)  # same direction as b — on second call, must NOT rematch invalidated emb_a

    # First call: inserts A, then B supersedes A → A is invalidated
    store.facts.add("t1", "u1", "User lives in New York", embedding=emb_a)
    id_b, _ = store.facts.add_or_supersede(
        "t1", "u1", "User moved to Seattle",
        old_embedding=emb_b, new_embedding=emb_b,
    )

    # Second call with same direction: must not re-invalidate B
    # (B is still valid, C is nearly identical to B → should confirm B, not invalidate it)
    id_c, is_new_c = store.facts.add_or_supersede(
        "t1", "u1", "User moved to Seattle",
        old_embedding=emb_c, new_embedding=emb_c,
    )
    assert id_c == id_b, "Second call with same content should confirm B, not insert C"
    assert not is_new_c, "Confirm must return is_new=False"
    row_b = store.facts.get(id_b)
    assert row_b["valid_until"] is None, "B must still be valid after second call"
