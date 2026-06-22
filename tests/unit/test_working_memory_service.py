from __future__ import annotations

from pathlib import Path

import pytest
from organism.core.memory.service.working_memory_service import WorkingMemoryService
from organism.core.stores import UnifiedStore
from organism.shared.domain import WorkingMemoryPack


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    """Create a UnifiedStore with temp DB."""
    db_path = tmp_path / "test_working_memory.db"
    return UnifiedStore(db_path)


@pytest.fixture
def working_memory_service(unified_store: UnifiedStore) -> WorkingMemoryService:
    """Create WorkingMemoryService."""
    from organism.core.config import CoreConfig
    config = CoreConfig()
    return WorkingMemoryService(store=unified_store, config=config)


def test_working_memory_service_empty_state(working_memory_service: WorkingMemoryService):
    """Test get_working_memory with no SSM state and no messages."""
    pack = working_memory_service.get_working_memory(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
    )

    assert isinstance(pack, WorkingMemoryPack)
    assert pack.ssm_state is None
    assert pack.short_summary is None
    assert pack.recent_refs == []



def test_working_memory_service_with_recent_messages(
    working_memory_service: WorkingMemoryService,
    unified_store: UnifiedStore,
):
    """Test get_working_memory with recent messages."""
    # Add 5 messages to the session
    for i in range(5):
        unified_store.messages.add(
            session_id="session1",
            tenant_id="tenant1",
            user_id="user1",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i}",
        )

    pack = working_memory_service.get_working_memory(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
        recent_k=3,  # Fetch last 3 messages
    )

    assert len(pack.recent_refs) == 3
    # recent_refs contains message IDs (strings), not content
    # Verify all 3 refs are non-empty strings
    assert all(isinstance(ref, str) and len(ref) > 0 for ref in pack.recent_refs)


def test_working_memory_service_generates_summary(
    working_memory_service: WorkingMemoryService,
    unified_store: UnifiedStore,
):
    """Test that short_summary is generated from recent messages."""
    # Add messages
    unified_store.messages.add(
        session_id="session1",
        tenant_id="tenant1",
        user_id="user1",
        role="user",
        content="Hello",
    )
    unified_store.messages.add(
        session_id="session1",
        tenant_id="tenant1",
        user_id="user1",
        role="assistant",
        content="Hi there!",
    )

    pack = working_memory_service.get_working_memory(
        tenant_id="tenant1",
        user_id="user1",
        session_id="session1",
    )

    # short_summary should be present (even if simple placeholder)
    assert pack.short_summary is not None
    assert len(pack.short_summary) > 0
