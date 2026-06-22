import pytest
import apsw
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema
from organism.core.memory.service.profile_updater import ProfileUpdater


@pytest.fixture
def store_with_facts():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    base = BaseStore(conn)
    fs = FactStore(base)
    facts = [
        "User's name is Alice",
        "User lives in Berlin",
        "User works as a software engineer",
        "User prefers Python",
        "User usually wakes up early",
    ]
    for f in facts:
        fs.add(tenant_id="t1", user_id="u1", content=f, category="fact")
    return fs


def test_profile_updated_from_name_fact(store_with_facts):
    updater = ProfileUpdater(fact_store=store_with_facts)
    updater.update_user("u1", "t1")
    profile = store_with_facts.get_profile("t1", "u1")
    keys = {r["key"] for r in profile}
    assert "name" in keys


def test_profile_updated_from_location_fact(store_with_facts):
    updater = ProfileUpdater(fact_store=store_with_facts)
    updater.update_user("u1", "t1")
    profile = store_with_facts.get_profile("t1", "u1")
    keys = {r["key"] for r in profile}
    assert "location" in keys


def test_update_idempotent(store_with_facts):
    updater = ProfileUpdater(fact_store=store_with_facts)
    updater.update_user("u1", "t1")
    updater.update_user("u1", "t1")
    profile = store_with_facts.get_profile("t1", "u1")
    name_rows = [r for r in profile if r["key"] == "name"]
    assert len(name_rows) == 1  # No duplicates
