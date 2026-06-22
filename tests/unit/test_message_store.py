import pytest
from pathlib import Path
from organism.core.stores.unified_store import UnifiedStore


@pytest.fixture
def store(tmp_path):
    s = UnifiedStore(tmp_path / "test.db")
    yield s
    s.close()


def test_get_by_span_returns_messages_in_range(store):
    tid = "t1"; uid = "u1"; sid = "s1"
    id1 = store.messages.add(session_id=sid, tenant_id=tid, role="user",    content="A", user_id=uid)
    id2 = store.messages.add(session_id=sid, tenant_id=tid, role="assistant", content="B", user_id=uid)
    id3 = store.messages.add(session_id=sid, tenant_id=tid, role="user",    content="C", user_id=uid)
    id4 = store.messages.add(session_id=sid, tenant_id=tid, role="assistant", content="D", user_id=uid)

    msgs = store.messages.get_by_span(tenant_id=tid, user_id=uid, start_id=id1, end_id=id4)
    assert len(msgs) == 4
    assert msgs[0]["content"] == "A"
    assert msgs[-1]["content"] == "D"


def test_get_by_span_empty_when_no_match(store):
    msgs = store.messages.get_by_span(tenant_id="t1", user_id="u1", start_id=999, end_id=1000)
    assert msgs == []