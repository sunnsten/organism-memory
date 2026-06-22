import numpy as np
import apsw
import pytest
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema
from organism.core.memory.rag.fact_retriever import FactRetriever


def make_embedding(text_seed: str) -> np.ndarray:
    h = sum(ord(c) for c in text_seed)
    rng = np.random.default_rng(h % 10000)
    e = rng.random(1024).astype(np.float32)
    return e / np.linalg.norm(e)


@pytest.fixture
def store_and_retriever():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    base = BaseStore(conn)
    fs = FactStore(base)
    retriever = FactRetriever(fact_store=fs)
    return fs, retriever


def test_stored_fact_is_retrieved(store_and_retriever):
    fs, retriever = store_and_retriever
    content = "User is a senior Python developer"
    emb = make_embedding(content)
    fs.add(tenant_id="t1", user_id="u1", content=content, category="fact", embedding=emb)

    query_emb = make_embedding("python developer user")
    results = retriever.retrieve(
        query="python developer user",
        query_embedding=query_emb,
        user_id="u1", tenant_id="t1", k=5,
    )
    contents = [r["content"] for r in results]
    assert content in contents


def test_profile_is_returned_from_store(store_and_retriever):
    fs, _ = store_and_retriever
    fs.upsert_profile("t1", "u1", "name", "Alice", confidence=0.9)
    fs.upsert_profile("t1", "u1", "location", "Berlin", confidence=0.85)

    profile = fs.get_profile("t1", "u1")
    keys = [r["key"] for r in profile]
    assert "name" in keys and "location" in keys
