from __future__ import annotations

import time
from pathlib import Path
import pytest

from organism.core.stores import UnifiedStore


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "test_flow.db")


def test_message_to_rag_chunk_lifecycle(store: UnifiedStore):
    """Full path: messages saved → WriteService writes RAG chunks → chunks retrievable."""
    from organism.core.memory.service.write_service import WriteService
    from organism.core.config import CoreConfig
    from organism.shared.domain import EventRecord

    tenant = "tenant1"
    user = "user1"
    session = "session1"

    user_msg_id = store.messages.add(
        session_id=session, tenant_id=tenant, user_id=user,
        role="user", content="What is Python?",
    )
    asst_msg_id = store.messages.add(
        session_id=session, tenant_id=tenant, user_id=user,
        role="assistant", content="Python is a programming language.",
    )

    assert isinstance(user_msg_id, int)
    assert isinstance(asst_msg_id, int)

    config = CoreConfig()
    write_service = WriteService(store=store, config=config)

    event = EventRecord(
        id=None,
        user_id=user,
        session_id=session,
        timestamp=time.time(),
        input_text="What is Python?",
        output_text="Python is a programming language.",
        kind="interaction",
        source="chat",
        importance=0.8,
    )

    block_id = write_service.append_event(event, tenant_id=tenant)
    assert block_id is not None
    assert isinstance(block_id, str)

    # Verify RAG chunks were stored
    rows = store.chunks._base.execute(
        "SELECT * FROM rag_chunks WHERE session_id=?", (session,)
    ).fetchall()
    assert len(rows) >= 1
    assert any("Python" in (r["content"] or "") for r in rows)  # type: ignore[call-overload]


def test_write_service_filters_low_importance(store: UnifiedStore):
    """Events below importance threshold return None and write no chunks."""
    from organism.core.memory.service.write_service import WriteService
    from organism.core.config import CoreConfig
    from organism.shared.domain import EventRecord

    config = CoreConfig()
    config.block_min_importance = 0.5
    write_service = WriteService(store=store, config=config)

    event = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="trivial",
        output_text="ok",
        kind="interaction",
        source="chat",
        importance=0.01,
    )

    result = write_service.append_event(event, tenant_id="tenant1")
    assert result is None

    rows = store.chunks._base.execute(
        "SELECT * FROM rag_chunks WHERE session_id='session1'"
    ).fetchall()
    assert len(rows) == 0
