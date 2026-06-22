import sqlite3
import pytest
from organism.core.stores.schema import init_schema, SCHEMA_VERSION


def test_schema_version_is_7():
    assert SCHEMA_VERSION == 7


def test_migration_idempotent_on_existing_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    init_schema(conn)  # second call must not raise
    conn.close()


def test_core_tables_present():
    """Core tables are created after init_schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}
    assert "messages" in tables
    assert "memory_items" in tables
    assert "sessions" in tables
    assert "rag_chunks" in tables
    assert "facts" in tables
    conn.close()


def test_research_tables_absent():
    """Research tables (context_summary, experience_blocks, ssm_states, etc.) must not be in schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}
    assert "context_summary" not in tables
    assert "experience_blocks" not in tables
    assert "ssm_states" not in tables
    assert "sleep_queue" not in tables
    assert "memory_candidates" not in tables
    conn.close()
