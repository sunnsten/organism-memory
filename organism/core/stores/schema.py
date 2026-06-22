from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import apsw
    APSW_AVAILABLE = True
except ImportError:
    APSW_AVAILABLE = False
    import sqlite3
    apsw = None  # type: ignore

logger = logging.getLogger(__name__)

# Schema version for tracking migrations
SCHEMA_VERSION = 7


def get_schema_sql() -> str:
    """Return the complete schema SQL as a single string."""
    return _SCHEMA_SQL


def init_schema(conn: Any) -> None:
    """
    Initialize the database schema on an open connection.

    Creates all tables, FTS virtual tables, triggers, and indexes.
    If vectorlite is available, also creates HNSW index for vector search.

    Idempotent: safe to call on an already-initialized database.

    Args:
        conn: Open SQLite connection (apsw.Connection or sqlite3.Connection).
    """
    try:
        # Enable WAL mode for concurrent reads
        if APSW_AVAILABLE and isinstance(conn, apsw.Connection):  # type: ignore[union-attr]
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA foreign_keys=ON;")

            # apsw: execute multi-statement SQL by consuming iterator
            # This executes all statements in the SQL string
            # Note: apsw is autocommit by default, no need to call commit()
            for _ in cur.execute(_SCHEMA_SQL):
                pass  # Just consume the iterator to execute all statements

            # Also create vectorlite HNSW index if available
            try:
                import vectorlite_py
                # Vectorlite should already be loaded by BaseStore
                for _ in cur.execute(_VECTORLITE_SCHEMA):
                    pass
                logger.info("Created vectorlite HNSW index (426x faster vector search)")
            except ImportError:
                logger.warning("vectorlite not available - vector search will use slow Python cosine")
            except Exception as e:
                logger.warning(f"Failed to create vectorlite HNSW index: {e}")
        else:
            # sqlite3 path
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.executescript(_SCHEMA_SQL)
            conn.commit()

            # Note: vectorlite requires apsw, skip for sqlite3
            logger.warning("Using sqlite3 (not apsw) - vectorlite HNSW unavailable")

        _apply_migrations(conn)
        logger.debug("Schema initialized successfully (version=%d)", SCHEMA_VERSION)
    except Exception:
        # Only rollback for sqlite3 (apsw doesn't have rollback in autocommit mode)
        if hasattr(conn, "rollback") and not (APSW_AVAILABLE and isinstance(conn, apsw.Connection)):  # type: ignore[union-attr]
            conn.rollback()  # type: ignore[attr-defined]
        logger.error("Failed to initialize schema", exc_info=True)
        raise


def init_db(db_path: Path) -> None:
    """
    Create and initialize a new database file with the full schema.

    Args:
        db_path: Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if APSW_AVAILABLE:
        conn = apsw.Connection(str(db_path))  # type: ignore[union-attr]
    else:
        import sqlite3
        conn = sqlite3.connect(str(db_path))

    try:
        init_schema(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration steps (keyed by target schema version)
# ---------------------------------------------------------------------------

_MIGRATION_STEPS: dict[int, list[str]] = {
    3: [
        "ALTER TABLE rag_chunks ADD COLUMN user_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_user ON rag_chunks(tenant_id, user_id)",
        # idx_rag_chunks_tenant references user_id, so it belongs here not in v1 schema
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant ON rag_chunks(tenant_id, user_id, source_type)",
    ],
    4: [
        """CREATE TABLE IF NOT EXISTS facts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id       TEXT NOT NULL,
        user_id         TEXT NOT NULL,
        content         TEXT NOT NULL,
        category        TEXT NOT NULL DEFAULT 'fact',
        importance      REAL NOT NULL DEFAULT 0.5,
        confirmed_count INTEGER NOT NULL DEFAULT 1,
        source_session_id TEXT,
        source_message_ids TEXT,
        embedding       BLOB,
        valid_from      INTEGER,
        valid_until     INTEGER,
        created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        last_confirmed  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
    )""",
        "CREATE INDEX IF NOT EXISTS idx_facts_tenant_user ON facts(tenant_id, user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(tenant_id, user_id, category)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_content_unique ON facts(tenant_id, user_id, content)",
        """CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
        content,
        content='facts',
        content_rowid='id',
        tokenize='unicode61'
    )""",
        """CREATE TRIGGER IF NOT EXISTS facts_fts_insert AFTER INSERT ON facts BEGIN
        INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
    END""",
        """CREATE TRIGGER IF NOT EXISTS facts_fts_delete AFTER DELETE ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.id, old.content);
    END""",
        """CREATE TRIGGER IF NOT EXISTS facts_fts_update AFTER UPDATE ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.id, old.content);
        INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
    END""",
        """CREATE TABLE IF NOT EXISTS user_profile (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        confidence  REAL NOT NULL DEFAULT 0.8,
        source_fact_ids TEXT,
        created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        updated_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        UNIQUE(tenant_id, user_id, key)
    )""",
        "CREATE INDEX IF NOT EXISTS idx_profile_tenant_user ON user_profile(tenant_id, user_id)",
    ],
    5: [
        # --- Per-user HNSW registry ---
        """CREATE TABLE IF NOT EXISTS user_hnsw_registry (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        table_name  TEXT NOT NULL,
        created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        UNIQUE(tenant_id, user_id)
    )""",

        # --- Fact history chain ---
        "ALTER TABLE facts ADD COLUMN superseded_by_id INTEGER REFERENCES facts(id)",
        "CREATE INDEX IF NOT EXISTS idx_facts_superseded ON facts(superseded_by_id)",

        # --- Semantic fact graph ---
        """CREATE TABLE IF NOT EXISTS fact_edges (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id     TEXT NOT NULL,
        user_id       TEXT NOT NULL,
        fact_id_a     INTEGER NOT NULL REFERENCES facts(id),
        fact_id_b     INTEGER NOT NULL REFERENCES facts(id),
        relation_type TEXT NOT NULL DEFAULT 'related',
        weight        REAL NOT NULL DEFAULT 0.5,
        created_at    INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        UNIQUE(fact_id_a, fact_id_b)
    )""",
        "CREATE INDEX IF NOT EXISTS idx_fact_edges_a ON fact_edges(tenant_id, user_id, fact_id_a)",
        "CREATE INDEX IF NOT EXISTS idx_fact_edges_b ON fact_edges(tenant_id, user_id, fact_id_b)",
    ],
    6: [
        # event_time: unix timestamp of when the fact occurred (parsed from LLM "when" field)
        # NULL = time unknown. Used to prefix facts with [YYYY-MM] in context assembly.
        "ALTER TABLE facts ADD COLUMN event_time INTEGER",
        "CREATE INDEX IF NOT EXISTS idx_facts_event_time ON facts(tenant_id, user_id, event_time)",
    ],
    7: [
        # event_date_raw: original 'when' string from LLM before timestamp parsing.
        # Preserves "last year", "March 2024", "unknown" etc. for debugging.
        "ALTER TABLE facts ADD COLUMN event_date_raw TEXT",
    ],
}


def _apply_migrations(conn: Any) -> None:
    """Apply any pending schema migrations to an open connection."""
    cur = conn.cursor()
    try:
        row = cur.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = row[0] if row and row[0] else 1
    except Exception:
        current_version = 1

    for target_version in sorted(_MIGRATION_STEPS.keys()):
        if current_version >= target_version:
            continue
        for sql in _MIGRATION_STEPS[target_version]:
            try:
                cur.execute(sql)
            except Exception as exc:
                if "duplicate column" in str(exc).lower() or "already exists" in str(exc).lower():
                    pass
                else:
                    logger.warning(
                        "Migration step failed (target_version=%d): %s — %s",
                        target_version,
                        sql,
                        exc,
                    )
        try:
            cur.execute(
                "INSERT OR REPLACE INTO schema_version(version) VALUES (?)",
                (target_version,),
            )
        except Exception:
            pass
        if hasattr(conn, "commit"):
            try:
                conn.commit()
            except Exception:
                pass
        current_version = target_version
        logger.info("Schema migrated to version %d", current_version)


# ---------------------------------------------------------------------------
# Full schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- ==========================================================================
-- v1 baseline schema — DO NOT add new tables here.
-- New tables and columns belong in _MIGRATION_STEPS (see module docstring).
-- ==========================================================================
-- Schema version tracking
-- ==========================================================================
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);
INSERT OR IGNORE INTO schema_version(version) VALUES (1);

-- ==========================================================================
-- Messages (episodic chat log, source for Tier 0 Working Memory)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tenant_id  TEXT NOT NULL,
    user_id    TEXT,
    role       TEXT NOT NULL,           -- 'user' / 'assistant' / 'system'
    content    TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    tokens_used INTEGER,
    metadata   TEXT                     -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_tenant_user
    ON messages(tenant_id, user_id, created_at);

-- ==========================================================================
-- Sessions
-- ==========================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,         -- UUID
    tenant_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    started_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    ended_at   INTEGER,
    status     TEXT NOT NULL DEFAULT 'active',   -- active / closed
    title      TEXT,
    metadata   TEXT                     -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant_user
    ON sessions(tenant_id, user_id, status);

-- ==========================================================================
-- Memory Items (Tier 3 / Research: curated summaries from consolidation)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS memory_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'fact',
        -- fact / preference / pattern / instruction / summary / habit / plan
    confidence      REAL NOT NULL DEFAULT 1.0,
    importance      REAL NOT NULL DEFAULT 0.5,
    source_block_id TEXT,               -- FK-like ref to experience_blocks.id
    valid_from      INTEGER,            -- unix ts, NULL = always valid
    valid_until     INTEGER,            -- unix ts, NULL = no expiry
    embedding       BLOB,               -- Qwen3-Embedding-0.6B vector (1024d)
    embedding_dim   INTEGER,
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved_at INTEGER,
    content_hash    TEXT,               -- SHA-256 for idempotency
    namespace       TEXT NOT NULL DEFAULT 'personal',
    tags            TEXT,               -- JSON array
    metadata        TEXT                -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_memory_items_tenant_user
    ON memory_items(tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_category
    ON memory_items(tenant_id, user_id, category);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_items_hash
    ON memory_items(tenant_id, user_id, namespace, content_hash)
    WHERE content_hash IS NOT NULL;

-- FTS5 index for memory_items
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    content,
    category,
    content='memory_items',
    content_rowid='id',
    tokenize='unicode61'
);

-- Triggers to keep FTS in sync with memory_items
CREATE TRIGGER IF NOT EXISTS memory_items_fts_insert
AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(rowid, content, category)
    VALUES (new.id, new.content, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_fts_delete
AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, category)
    VALUES ('delete', old.id, old.content, old.category);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_fts_update
AFTER UPDATE OF content, category ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, category)
    VALUES ('delete', old.id, old.content, old.category);
    INSERT INTO memory_items_fts(rowid, content, category)
    VALUES (new.id, new.content, new.category);
END;

-- ==========================================================================
-- RAG Chunks (Tier 1: chunked raw experience)
-- user_id column and idx_rag_chunks_tenant/user added in migration v3
-- ==========================================================================
CREATE TABLE IF NOT EXISTS rag_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL,
    source_type   TEXT NOT NULL,         -- 'experience_block' / 'message' / 'doc'
    source_id     TEXT NOT NULL,         -- UUID of parent experience_block
    session_id    TEXT,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB,                  -- Qwen3-Embedding-0.6B vector (1024d)
    embedding_dim INTEGER,
    created_at    INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    tags          TEXT                   -- JSON: {"topic": ..., "channel": ...}
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
    ON rag_chunks(source_type, source_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_chunks_unique
    ON rag_chunks(source_type, source_id, chunk_index);

-- FTS5 index for rag_chunks
CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    content,
    content='rag_chunks',
    content_rowid='id',
    tokenize='unicode61'
);

-- Triggers to keep FTS in sync with rag_chunks
CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_insert
AFTER INSERT ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_delete
AFTER DELETE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_update
AFTER UPDATE OF content ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO rag_chunks_fts(rowid, content)
    VALUES (new.id, new.content);
END;

-- ==========================================================================
-- Vectorlite HNSW index for memory_items (Step K)
-- facts and user_profile tables are created by migration v4
-- ==========================================================================
-- NOTE: This table is created ONLY if vectorlite extension is loaded.
-- If vectorlite is not available, vector search falls back to Python cosine.
--
-- Performance: 426x faster than Python cosine similarity
-- - Python cosine: ~40ms per query (10k vectors)
-- - Vectorlite HNSW: ~0.09ms per query (10k vectors)
--
-- CRITICAL: rowid in vec_memory_items MUST match id in memory_items
-- This is maintained by MemoryItemStore.add(), update_embedding(), delete()
"""


# Vectorlite extension loading for HNSW index
# This is executed after the base schema if vectorlite is available
_VECTORLITE_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory_items USING vectorlite(
    embedding float32[1024],
    hnsw(max_elements=100000, ef_construction=200, M=32)
);
"""


__all__ = ["init_db", "init_schema", "get_schema_sql", "SCHEMA_VERSION", "_apply_migrations"]
