import sqlite3
import pytest
from organism.core.stores.schema import init_schema, SCHEMA_VERSION


def test_schema_version_is_7():
    assert SCHEMA_VERSION == 7


def test_facts_table_exists():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.cursor()
    tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "facts" in tables
    assert "user_profile" in tables
    conn.close()


def test_facts_fts_triggers_exist():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.cursor()
    triggers = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert "facts_fts_insert" in triggers
    assert "facts_fts_delete" in triggers
    assert "facts_fts_update" in triggers
    conn.close()


def test_facts_columns():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.cursor()
    cols = {row[1] for row in cur.execute("PRAGMA table_info(facts)")}
    assert cols >= {"id", "tenant_id", "user_id", "content", "category",
                    "importance", "confirmed_count", "source_session_id",
                    "embedding", "created_at", "last_confirmed"}
    conn.close()


def test_user_profile_unique_constraint():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.cursor()
    cur.execute("INSERT INTO user_profile(tenant_id,user_id,key,value) VALUES('t1','u1','name','Alice')")
    conn.commit()
    with pytest.raises(Exception):
        cur.execute("INSERT INTO user_profile(tenant_id,user_id,key,value) VALUES('t1','u1','name','Bob')")
        conn.commit()
    conn.close()
