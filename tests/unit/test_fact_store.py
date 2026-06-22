import numpy as np
import pytest
import apsw
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema


@pytest.fixture
def store():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    base = BaseStore(conn)
    return FactStore(base)


def test_add_fact_returns_id(store):
    fid = store.add(
        tenant_id="t1", user_id="u1",
        content="User prefers Python over JavaScript",
        category="preference",
    )
    assert isinstance(fid, int) and fid > 0


def test_add_fact_idempotent_by_content(store):
    fid1 = store.add(tenant_id="t1", user_id="u1", content="User lives in Berlin")
    fid2 = store.add(tenant_id="t1", user_id="u1", content="User lives in Berlin")
    assert fid1 == fid2


def test_confirm_increments_count(store):
    fid = store.add(tenant_id="t1", user_id="u1", content="User runs every morning")
    store.confirm(fid)
    store.confirm(fid)
    row = store.get(fid)
    assert row["confirmed_count"] == 3


def test_find_similar_scored_returns_none_on_empty(store):
    emb = np.random.rand(1024).astype(np.float32)
    result = store.find_similar_scored(emb, "t1", "u1", min_score=0.9)
    assert result is None


def test_find_similar_scored_returns_id_for_identical_embedding(store):
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    fid = store.add(
        tenant_id="t1", user_id="u1",
        content="User prefers mornings",
        embedding=emb,
    )
    result = store.find_similar_scored(emb, "t1", "u1", min_score=0.99)
    assert result is not None
    assert result[0] == fid


def test_find_similar_scored_returns_none_below_threshold(store):
    emb_a = np.zeros(1024, dtype=np.float32)
    emb_a[0] = 1.0  # unit vector along dim 0
    emb_b = np.zeros(1024, dtype=np.float32)
    emb_b[1] = 1.0  # unit vector along dim 1 — orthogonal, cosine = 0
    store.add(tenant_id="t1", user_id="u1", content="Some fact", embedding=emb_a)
    result = store.find_similar_scored(emb_b, "t1", "u1", min_score=0.9)
    assert result is None


def test_upsert_profile(store):
    store.upsert_profile("t1", "u1", "name", "Alice", confidence=0.9)
    store.upsert_profile("t1", "u1", "name", "Alice Updated", confidence=0.95)
    rows = store.get_profile("t1", "u1")
    name_rows = [r for r in rows if r["key"] == "name"]
    assert len(name_rows) == 1
    assert name_rows[0]["value"] == "Alice Updated"


def test_get_profile_filters_low_confidence(store):
    store.upsert_profile("t1", "u1", "name", "Alice", confidence=0.9)
    store.upsert_profile("t1", "u1", "location", "Berlin", confidence=0.3)
    rows = store.get_profile("t1", "u1", min_confidence=0.6)
    assert len(rows) == 1
    assert rows[0]["key"] == "name"


def test_search_fts_returns_matching_facts(store):
    store.add(tenant_id="t1", user_id="u1", content="User prefers Python programming")
    store.add(tenant_id="t1", user_id="u1", content="User enjoys hiking on weekends")
    results = store.search_fts("Python programming", "t1", "u1", limit=5)
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)


def test_search_vector_returns_similar_facts(store):
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    store.add(tenant_id="t1", user_id="u1", content="User likes coffee", embedding=emb)
    store.add(tenant_id="t1", user_id="u1", content="User dislikes tea")  # no embedding
    results = store.search_vector(emb, "t1", "u1", limit=5)
    assert len(results) >= 1
    assert results[0]["content"] == "User likes coffee"


def test_add_concurrent_duplicate_does_not_create_two_rows(store):
    """INSERT OR IGNORE prevents duplicate rows even if called twice in sequence."""
    fid1 = store.add(tenant_id="t1", user_id="u1", content="User drinks coffee daily")
    fid2 = store.add(tenant_id="t1", user_id="u1", content="User drinks coffee daily")
    assert fid1 == fid2
    # Verify only one row exists
    row = store._base.execute(
        "SELECT COUNT(*) as cnt FROM facts WHERE tenant_id='t1' AND user_id='u1' AND content='User drinks coffee daily'"
    ).fetchone()
    assert row["cnt"] == 1


def test_fact_stores_event_date_raw(store):
    """event_date_raw column stores the raw 'when' string from LLM."""
    fid = store.add(
        tenant_id="t1", user_id="u1",
        content="User moved to Berlin",
        event_date_raw="March 2024",
        event_time=1709251200,
    )
    row = store.get(fid)
    assert row["event_date_raw"] == "March 2024"
