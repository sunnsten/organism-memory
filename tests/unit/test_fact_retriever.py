import numpy as np
import pytest
import apsw
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema
from organism.core.memory.rag.fact_retriever import FactRetriever, mmr_select


def make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.random(1024).astype(np.float32)
    return e / np.linalg.norm(e)


@pytest.fixture
def populated_store():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    base = BaseStore(conn)
    fs = FactStore(base)
    for i in range(10):
        fs.add(
            tenant_id="t1", user_id="u1",
            content=f"User fact number {i} about their preferences",
            category="fact",
            importance=0.5 + i * 0.02,
            embedding=make_embedding(i),
        )
    return fs


def test_mmr_select_returns_k_items():
    query_emb = make_embedding(99)
    candidates = [
        {"embedding": make_embedding(i).tobytes(), "id": i, "content": f"fact {i}"}
        for i in range(10)
    ]
    selected = mmr_select(query_emb, candidates, k=5)
    assert len(selected) == 5


def test_mmr_select_fewer_than_k():
    query_emb = make_embedding(99)
    candidates = [
        {"embedding": make_embedding(i).tobytes(), "id": i, "content": f"fact {i}"}
        for i in range(3)
    ]
    selected = mmr_select(query_emb, candidates, k=5)
    assert len(selected) == 3


def test_mmr_select_no_duplicates():
    query_emb = make_embedding(99)
    candidates = [
        {"embedding": make_embedding(i).tobytes(), "id": i, "content": f"fact {i}"}
        for i in range(8)
    ]
    selected = mmr_select(query_emb, candidates, k=5)
    ids = [c["id"] for c in selected]
    assert len(ids) == len(set(ids))


def test_fact_retriever_returns_results(populated_store):
    retriever = FactRetriever(fact_store=populated_store)
    query_emb = make_embedding(42)
    results = retriever.retrieve(
        query="user preferences",
        query_embedding=query_emb,
        user_id="u1",
        tenant_id="t1",
        k=5,
    )
    assert 1 <= len(results) <= 5
    assert all("content" in r for r in results)
    assert all("id" in r for r in results)


def test_fact_retriever_empty_store():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    fs = FactStore(BaseStore(conn))
    retriever = FactRetriever(fact_store=fs)
    results = retriever.retrieve(
        query="anything",
        query_embedding=make_embedding(0),
        user_id="u1",
        tenant_id="t1",
        k=5,
    )
    assert results == []


def test_fact_retriever_respects_k(populated_store):
    retriever = FactRetriever(fact_store=populated_store)
    results = retriever.retrieve(
        query="user fact preferences",
        query_embedding=make_embedding(0),
        user_id="u1",
        tenant_id="t1",
        k=3,
    )
    assert len(results) <= 3


def test_rrf_merge_combines_both_signals():
    """RRF should return items from both FTS and vector lists."""
    from organism.core.memory.rag.fact_retriever import _rrf_merge
    fts_rows = [{"id": 1, "content": "a", "importance": 0.5, "confirmed_count": 1, "embedding": None, "created_at": 0}]
    vec_rows = [{"id": 2, "content": "b", "importance": 0.5, "confirmed_count": 1, "embedding": None, "created_at": 0, "vector_score": 0.9}]
    merged = _rrf_merge(fts_rows, vec_rows)
    ids = {r["id"] for r in merged}
    assert 1 in ids
    assert 2 in ids
