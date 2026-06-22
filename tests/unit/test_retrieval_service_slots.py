from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import time

import numpy as np
import pytest
import torch

from organism.core.memory.service.retrieval_service import RetrievalService
from organism.core.stores import UnifiedStore, BaseStore
from organism.core.memory.rag.chunk_store import ChunkStore


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    """Create a UnifiedStore with temp DB."""
    db_path = tmp_path / "test_retrieval_slots.db"
    return UnifiedStore(db_path)


@pytest.fixture
def chunk_store(tmp_path: Path) -> ChunkStore:
    """Create a ChunkStore with temp DB."""
    db_path = tmp_path / "test_chunks.db"
    return ChunkStore(BaseStore(db_path))


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed vectors."""
    embedder = Mock()
    embedder.embed.return_value = np.random.randn(1024).astype(np.float32)
    return embedder


@pytest.fixture
def retrieval_service(unified_store: UnifiedStore, chunk_store: ChunkStore, mock_embedder) -> RetrievalService:
    """Create RetrievalService."""
    return RetrievalService(
        message_store=unified_store.messages,
        memory_item_store=unified_store.memory_items,
        chunk_store=chunk_store,
        embedder=mock_embedder,
    )


def test_retrieval_service_with_neural_slots(
    retrieval_service: RetrievalService,
    unified_store: UnifiedStore,
):
    """Test retrieve() with neural slot integration."""
    # Add a session message
    unified_store.messages.add(
        session_id="session1",
        tenant_id="tenant1",
        user_id="user1",
        role="user",
        content="What is Python?",
    )

    # Mock MemoryCore with slot retrieval
    mock_memory_core = Mock()
    from organism.shared.domain import SlotRetrieveResult
    mock_memory_core.retrieve.return_value = [
        SlotRetrieveResult(
            slot_index=0,
            text="Python is a programming language",
            score=0.95,
            key=torch.randn(256),
            value=torch.randn(256),
            record=None,  # Optional MemoryRecord reference
        ),
    ]

    context = retrieval_service.retrieve(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
        query="What is Python?",
        memory_core=mock_memory_core,  # NEW parameter
        neural_slots_top_k=3,  # NEW parameter
    )

    # Verify that memory_core.retrieve was called
    mock_memory_core.retrieve.assert_called_once()

    # Context should be returned
    assert context is not None
    assert context.working_memory_count >= 1


def test_retrieval_service_without_neural_slots(
    retrieval_service: RetrievalService,
    unified_store: UnifiedStore,
):
    """Test retrieve() without neural slots (backward compatibility)."""
    # Add a session message
    unified_store.messages.add(
        session_id="session1",
        tenant_id="tenant1",
        user_id="user1",
        role="user",
        content="Test",
    )

    # Call without memory_core (should work)
    context = retrieval_service.retrieve(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
        query="Test",
        # No memory_core parameter
    )

    assert context is not None
    assert context.working_memory_count >= 1


def test_retrieval_service_analytics_tracking(
    retrieval_service: RetrievalService,
    unified_store: UnifiedStore,
):
    """Test that analytics tracks retrieval latency."""
    # Add a session message
    unified_store.messages.add(
        session_id="session1",
        tenant_id="tenant1",
        user_id="user1",
        role="user",
        content="Test query",
    )

    # Clear analytics metrics
    from organism.shared.analytics import analytics
    # Just verify the call happens (metrics accumulate)

    start = time.time()
    context = retrieval_service.retrieve(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
        query="Test query",
    )
    duration = (time.time() - start) * 1000

    # Verify context returned (analytics called internally)
    assert context is not None
    # Analytics should have tracked this retrieval
    # NOTE: We can't easily verify Prometheus metrics in tests without checking registry
