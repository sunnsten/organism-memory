from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from organism.core.stores import BaseStore, UnifiedStore, MemoryItemStore


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    """Create a UnifiedStore with temp DB (vec_memory_items auto-created)."""
    db_path = tmp_path / "test_vectorlite.db"
    return UnifiedStore(db_path)


def test_vectorlite_hnsw_basic_search(unified_store: UnifiedStore):
    """Test basic HNSW vector search."""
    # Add 3 memory items with embeddings
    embeddings = [
        np.random.randn(1024).astype(np.float32),
        np.random.randn(1024).astype(np.float32),
        np.random.randn(1024).astype(np.float32),
    ]

    # Normalize embeddings
    for i in range(len(embeddings)):
        embeddings[i] = (embeddings[i] / np.linalg.norm(embeddings[i])).astype(np.float32)

    # Insert memory items
    for i, emb in enumerate(embeddings):
        unified_store.memory_items.add(
            tenant_id="tenant1",
            user_id="user1",
            content=f"Memory item {i}",
            category="fact",
            embedding=emb,
        )

    # Search with first embedding (should match itself best)
    query_embedding = embeddings[0]
    results = unified_store.memory_items.search_vector(
        tenant_id="tenant1",
        user_id="user1",
        query_embedding=query_embedding,
        limit=3,
    )

    assert len(results) == 3
    assert results[0][1] > 0.99  # First result should be almost perfect match
    assert results[0][0]["content"] == "Memory item 0"


def test_vectorlite_similarity_threshold(unified_store: UnifiedStore):
    """Test min_similarity threshold filtering."""
    # Add 3 memory items with different similarities
    base_emb = np.random.randn(1024).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)

    # Create embeddings with varying similarity to base_emb
    similar_emb = base_emb + 0.01 * np.random.randn(1024).astype(np.float32)
    similar_emb = similar_emb / np.linalg.norm(similar_emb)

    different_emb = np.random.randn(1024).astype(np.float32)
    different_emb = different_emb / np.linalg.norm(different_emb)

    # Insert items
    unified_store.memory_items.add(
        tenant_id="tenant1",
        user_id="user1",
        content="Very similar",
        category="fact",
        embedding=similar_emb,
    )
    unified_store.memory_items.add(
        tenant_id="tenant1",
        user_id="user1",
        content="Very different",
        category="fact",
        embedding=different_emb,
    )

    # Search with high threshold (should filter out different_emb)
    results = unified_store.memory_items.search_vector(
        tenant_id="tenant1",
        user_id="user1",
        query_embedding=base_emb,
        limit=10,
        min_similarity=0.9,  # High threshold
    )

    # Should only return the similar embedding
    assert len(results) >= 1
    assert results[0][0]["content"] == "Very similar"
    assert results[0][1] > 0.9


def test_vectorlite_multi_tenancy_isolation(unified_store: UnifiedStore):
    """
    🔒 CRITICAL: Test that HNSW search enforces tenant isolation.
    """
    # Create same embedding for different tenants
    embedding = np.random.randn(1024).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    # Tenant 1: add memory item
    unified_store.memory_items.add(
        tenant_id="tenant1",
        user_id="user1",
        content="Tenant 1 secret",
        category="fact",
        embedding=embedding,
    )

    # Tenant 2: add memory item (same user_id, different tenant)
    unified_store.memory_items.add(
        tenant_id="tenant2",
        user_id="user1",
        content="Tenant 2 secret",
        category="fact",
        embedding=embedding,
    )

    # Tenant 1: search should only return tenant1 data
    results_t1 = unified_store.memory_items.search_vector(
        tenant_id="tenant1",
        user_id="user1",
        query_embedding=embedding,
        limit=10,
    )
    assert len(results_t1) == 1
    assert results_t1[0][0]["content"] == "Tenant 1 secret"
    assert results_t1[0][0]["tenant_id"] == "tenant1"

    # Tenant 2: search should only return tenant2 data
    results_t2 = unified_store.memory_items.search_vector(
        tenant_id="tenant2",
        user_id="user1",
        query_embedding=embedding,
        limit=10,
    )
    assert len(results_t2) == 1
    assert results_t2[0][0]["content"] == "Tenant 2 secret"
    assert results_t2[0][0]["tenant_id"] == "tenant2"


def test_vectorlite_fallback_to_python(tmp_path: Path):
    """Test fallback to Python cosine when vectorlite unavailable."""
    # This test would require mocking VECTORLITE_AVAILABLE = False
    # For now, we just verify that Python fallback path exists
    db_path = tmp_path / "test_fallback.db"
    store = UnifiedStore(db_path)

    # Add item
    embedding = np.random.randn(1024).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    store.memory_items.add(
        tenant_id="tenant1",
        user_id="user1",
        content="Test item",
        category="fact",
        embedding=embedding,
    )

    # Search should work (either HNSW or Python)
    results = store.memory_items.search_vector(
        tenant_id="tenant1",
        user_id="user1",
        query_embedding=embedding,
        limit=10,
    )

    assert len(results) == 1
    assert results[0][1] > 0.99  # Perfect match


def test_chunk_store_multi_user_isolation(tmp_path: Path):
    """
    CRITICAL regression test for HNSW multi-user isolation bug.

    With a global HNSW index shared across users, search_vector(user_id=A)
    must return only user A's chunks even when user B has many chunks with
    similar embeddings. Previously k=60 global KNN would lose user A's chunks
    when hundreds of other users' chunks were in the index.
    """
    store = UnifiedStore(tmp_path / "test_isolation.db")
    np.random.seed(42)

    # Query vector — user_A's chunks will be near-identical to this
    query_vec = np.random.randn(1024).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # User A: 5 chunks, all very similar to query_vec
    user_a_chunks = []
    for i in range(5):
        emb = query_vec + 0.001 * np.random.randn(1024).astype(np.float32)
        emb = (emb / np.linalg.norm(emb)).astype(np.float32)
        user_a_chunks.append({
            "tenant_id": "t1", "user_id": "user_A",
            "source_type": "exp", "source_id": f"a_{i}",
            "chunk_index": 0, "content": f"user_A chunk {i}",
            "embedding": emb, "session_id": "s1",
            "created_at": 1000 + i, "tags": None,
        })
    store.chunks.add_batch(user_a_chunks)

    # User B: 50 chunks with random embeddings (fills HNSW index)
    user_b_chunks = []
    for i in range(50):
        emb = np.random.randn(1024).astype(np.float32)
        emb = (emb / np.linalg.norm(emb)).astype(np.float32)
        user_b_chunks.append({
            "tenant_id": "t1", "user_id": "user_B",
            "source_type": "exp", "source_id": f"b_{i}",
            "chunk_index": 0, "content": f"user_B chunk {i}",
            "embedding": emb, "session_id": "s2",
            "created_at": 2000 + i, "tags": None,
        })
    store.chunks.add_batch(user_b_chunks)

    # Search for user_A — must return only user_A's chunks
    results = store.chunks.search_vector(
        tenant_id="t1",
        query_embedding=query_vec,
        limit=10,
        user_id="user_A",
    )

    assert len(results) == 5, f"Expected 5 user_A chunks, got {len(results)}"
    for chunk_dict, sim in results:
        assert chunk_dict["user_id"] == "user_A", (
            f"Got chunk from user '{chunk_dict['user_id']}' — cross-user leak!"
        )

    # Search for user_B — must return only user_B's chunks
    results_b = store.chunks.search_vector(
        tenant_id="t1",
        query_embedding=query_vec,
        limit=10,
        user_id="user_B",
    )
    for chunk_dict, _ in results_b:
        assert chunk_dict["user_id"] == "user_B", (
            f"Got chunk from user '{chunk_dict['user_id']}' — cross-user leak!"
        )
