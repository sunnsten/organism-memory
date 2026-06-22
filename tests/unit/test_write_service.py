from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from organism.core.memory.service.write_service import WriteService
from organism.core.stores import UnifiedStore
from organism.shared.domain import EventRecord


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    """Create a UnifiedStore with temp DB."""
    db_path = tmp_path / "test_write.db"
    return UnifiedStore(db_path)


@pytest.fixture
def write_service(unified_store: UnifiedStore) -> WriteService:
    """Create WriteService with default config."""
    from organism.core.config import CoreConfig
    config = CoreConfig()
    return WriteService(store=unified_store, config=config)


def test_write_service_append_event_basic(write_service: WriteService, unified_store: UnifiedStore):
    """Test basic event append - should write RAG chunks and return a block_id."""
    event = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="What is Python?",
        output_text="Python is a programming language.",
        kind="interaction",
        source="chat",
        importance=0.7,
        surprisal_norm=0.3,
        attention_focus=0.5,
    )

    block_id = write_service.append_event(event=event, tenant_id="tenant1")

    assert block_id is not None
    assert isinstance(block_id, str)  # UUID string

    # Verify RAG chunks were written
    rows = unified_store.chunks._base.execute(
        "SELECT * FROM rag_chunks WHERE session_id='session1'"
    ).fetchall()
    assert len(rows) >= 1


def test_write_service_importance_threshold_filtering(unified_store: UnifiedStore):
    """Test that events below importance threshold are filtered."""
    from organism.core.config import CoreConfig

    # Create config with custom threshold
    config = CoreConfig()
    config.block_min_importance = 0.1
    config.source_multiplier_chat = 1.0
    # Threshold = 0.1 * 1.0 = 0.1

    write_service = WriteService(store=unified_store, config=config)

    event_low = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="test",
        output_text="test",
        kind="interaction",
        source="chat",
        importance=0.05,  # Below threshold
    )

    result = write_service.append_event(event=event_low, tenant_id="tenant1")
    assert result is None  # Event was filtered


def test_write_service_remember_source_multiplier(unified_store: UnifiedStore):
    """Test that 'remember' source has lower threshold (higher priority)."""
    from organism.core.config import CoreConfig

    # Create config with custom threshold
    config = CoreConfig()
    config.block_min_importance = 0.1
    config.source_multiplier_remember = 0.5
    # Threshold for remember = 0.1 * 0.5 = 0.05

    write_service = WriteService(store=unified_store, config=config)

    event = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="Remember: I love Python",
        output_text="Remembered.",
        kind="explicit_fact",
        source="remember",
        importance=0.06,  # Would be filtered for chat, but passes for remember
    )

    block_id = write_service.append_event(event=event, tenant_id="tenant1")
    assert block_id is not None

    rows = unified_store.chunks._base.execute(
        "SELECT * FROM rag_chunks WHERE session_id='session1'"
    ).fetchall()
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# skip_chunk_embedding / embed_pending integration
# ---------------------------------------------------------------------------

def _make_event(user_id="u1", session_id="s1", importance=0.7) -> EventRecord:
    return EventRecord(
        id=None,
        user_id=user_id,
        session_id=session_id,
        timestamp=1_700_000_000.0,
        input_text="Hello world",
        output_text="Hi there",
        kind="interaction",
        source="chat",
        importance=importance,
    )


def test_skip_chunk_embedding_leaves_embeddings_null(unified_store: UnifiedStore):
    """skip_chunk_embedding=True must write chunks but leave embedding column NULL."""
    from unittest.mock import MagicMock
    from organism.core.config import CoreConfig

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 1024

    config = CoreConfig()
    svc = WriteService(store=unified_store, config=config, embedder=mock_embedder)

    event = _make_event()
    svc.append_event(event, tenant_id="t1", skip_chunk_embedding=True)

    # Embedder must not have been called
    mock_embedder.embed.assert_not_called()

    # Chunks must still be present (just with NULL embedding)
    rows = unified_store.chunks._base.execute(
        "SELECT embedding FROM rag_chunks WHERE session_id='s1'"
    ).fetchall()
    assert len(rows) >= 1
    assert all(r["embedding"] is None for r in rows)  # type: ignore[call-overload]


def test_skip_chunk_embedding_false_calls_embed(unified_store: UnifiedStore):
    """With skip_chunk_embedding=False (default), embedder.embed() must be called."""
    from unittest.mock import MagicMock
    import numpy as np
    from organism.core.config import CoreConfig

    emb = np.ones(1024, dtype=np.float32) / (1024 ** 0.5)
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = emb

    config = CoreConfig()
    svc = WriteService(store=unified_store, config=config, embedder=mock_embedder)

    event = _make_event(session_id="s2")
    svc.append_event(event, tenant_id="t1")

    mock_embedder.embed.assert_called()


def test_embed_pending_fills_null_embeddings(unified_store: UnifiedStore):
    """embed_pending() must set embedding for all NULL-embedding chunks of a session."""
    from unittest.mock import MagicMock
    import numpy as np
    from organism.core.config import CoreConfig

    emb = np.ones(1024, dtype=np.float32) / (1024 ** 0.5)
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [emb]

    # Write one chunk with no embedding
    config = CoreConfig()
    svc = WriteService(store=unified_store, config=config, embedder=None)
    event = _make_event(session_id="s3")
    svc.append_event(event, tenant_id="t1", skip_chunk_embedding=True)

    # Verify chunk has NULL embedding before embed_pending
    before = unified_store.chunks._base.execute(
        "SELECT embedding FROM rag_chunks WHERE session_id='s3'"
    ).fetchall()
    assert len(before) >= 1
    assert all(r["embedding"] is None for r in before)  # type: ignore[call-overload]

    # Now embed them
    n = unified_store.chunks.embed_pending(
        session_id="s3", tenant_id="t1", user_id="u1", embedder=mock_embedder
    )
    assert n == len(before)

    after = unified_store.chunks._base.execute(
        "SELECT embedding FROM rag_chunks WHERE session_id='s3'"
    ).fetchall()
    assert all(r["embedding"] is not None for r in after)  # type: ignore[call-overload]


def test_embed_pending_noop_when_all_embedded(unified_store: UnifiedStore):
    """embed_pending() returns 0 if all chunks already have embeddings."""
    from unittest.mock import MagicMock
    import numpy as np
    from organism.core.config import CoreConfig

    emb = np.ones(1024, dtype=np.float32) / (1024 ** 0.5)
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = emb
    mock_embedder.embed_batch.return_value = [emb]

    config = CoreConfig()
    svc = WriteService(store=unified_store, config=config, embedder=mock_embedder)
    event = _make_event(session_id="s4")
    svc.append_event(event, tenant_id="t1")  # default: embedding=True

    n = unified_store.chunks.embed_pending(
        session_id="s4", tenant_id="t1", user_id="u1", embedder=mock_embedder
    )
    assert n == 0
