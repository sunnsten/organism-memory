from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from organism.core.stores import UnifiedStore


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "chunk_hnsw_test.db")


def _emb(val: float, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(val)
    v[1] = 1.0 - float(val)
    norm = np.linalg.norm(v)
    return v / norm


def test_vector_search_returns_user_own_chunks(store):
    """search_vector with user_id returns only that user's chunks, ordered by similarity."""
    emb_target = _emb(0.3)
    emb_other = _emb(0.9)

    # Alice's chunk (should be found)
    store.chunks.add(
        tenant_id="t1", source_type="msg", source_id="s1", chunk_index=0,
        content="Alice chunk", embedding=emb_target,
        session_id="sess1", user_id="alice",
    )
    # Bob's chunk (should NOT appear in Alice's results)
    store.chunks.add(
        tenant_id="t1", source_type="msg", source_id="s2", chunk_index=0,
        content="Bob chunk", embedding=emb_other,
        session_id="sess2", user_id="bob",
    )

    results = store.chunks.search_vector("t1", emb_target, limit=5, user_id="alice")
    contents = [r[0]["content"] for r in results]
    assert "Alice chunk" in contents
    assert "Bob chunk" not in contents


def test_delete_removes_from_hnsw(store):
    """After delete_by_source, vector search no longer finds the deleted chunk."""
    emb = _emb(0.5)
    store.chunks.add(
        tenant_id="t1", source_type="msg", source_id="s1", chunk_index=0,
        content="To be deleted", embedding=emb,
        session_id="sess1", user_id="alice",
    )

    store.chunks.delete_by_source("msg", "s1")

    results = store.chunks.search_vector("t1", emb, limit=5, user_id="alice")
    contents = [r[0]["content"] for r in results]
    assert "To be deleted" not in contents


def test_embed_pending_inserts_into_hnsw(store):
    """embed_pending() inserts embeddings into per-user HNSW."""

    class FakeEmbedder:
        def embed(self, text: str) -> np.ndarray:
            return _emb(0.4)

    # Add chunk without embedding
    store.chunks.add(
        tenant_id="t1", source_type="msg", source_id="s1", chunk_index=0,
        content="Pending chunk", embedding=None,
        session_id="sess1", user_id="alice",
    )

    count = store.chunks.embed_pending("sess1", "t1", "alice", FakeEmbedder())
    assert count == 1

    results = store.chunks.search_vector("t1", _emb(0.4), limit=5, user_id="alice")
    assert results, "Embedded chunk must appear in vector search"
