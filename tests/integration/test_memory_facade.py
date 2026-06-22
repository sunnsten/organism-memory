from __future__ import annotations

import time
from pathlib import Path
import pytest

from organism.core.memory.service.memory_facade import MemoryFacade
from organism.core.stores import UnifiedStore
from organism.core.config import CoreConfig
from organism.shared.domain import EventRecord


@pytest.fixture
def unified_store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "test_facade.db")


@pytest.fixture
def facade(unified_store: UnifiedStore) -> MemoryFacade:
    return MemoryFacade.from_store(unified_store, tenant_id="tenant1")


def test_facade_delegates_append_event(facade: MemoryFacade, unified_store: UnifiedStore):
    event = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="Hello",
        output_text="Hi!",
        kind="interaction",
        source="chat",
        importance=0.8,
    )

    event_id = facade.append_event(event)
    assert event_id is not None
    assert event_id == 1


def test_facade_delegates_get_working_memory(facade: MemoryFacade, unified_store: UnifiedStore):
    unified_store.messages.add(
        session_id="session1", tenant_id="tenant1", user_id="user1",
        role="user", content="Test message",
    )

    wm = facade.get_working_memory(user_id="user1", session_id="session1")

    assert wm is not None
    assert len(wm.recent_refs) > 0


def test_facade_filters_low_importance(facade: MemoryFacade):
    event = EventRecord(
        id=None,
        user_id="user1",
        session_id="session1",
        timestamp=time.time(),
        input_text="Trivial",
        output_text="Ok",
        kind="interaction",
        source="chat",
        importance=0.01,
    )

    event_id = facade.append_event(event)
    assert event_id == 0
