from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT    NOT NULL UNIQUE,
    user_id     TEXT    NOT NULL,
    tenant_id   TEXT    NOT NULL DEFAULT 'default',
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER,          -- NULL = never expires
    active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""

_PREFIX = "sk-organism-"


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_key() -> str:
    return _PREFIX + secrets.token_hex(32)


class ApiKeyStore:
    """Manages API keys for the organism proxy in organism.db."""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLE)

    def create_key(
        self,
        user_id: str,
        tenant_id: str = "default",
        expires_days: Optional[int] = None,
    ) -> str:
        raw_key = generate_key()
        expires_at = int(time.time() + expires_days * 86400) if expires_days else None
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_hash, user_id, tenant_id, created_at, expires_at, active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (_hash(raw_key), user_id, tenant_id, int(time.time()), expires_at),
            )
        return raw_key

    def resolve(self, raw_key: str) -> Optional[dict]:
        """Return {user_id, tenant_id} if key is valid, else None."""
        now = int(time.time())
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT user_id, tenant_id FROM api_keys
                WHERE key_hash = ?
                  AND active = 1
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (_hash(raw_key), now),
            ).fetchone()
        return dict(row) if row else None

    def list_keys(self, user_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, tenant_id, created_at, expires_at, active
                FROM api_keys WHERE user_id = ? ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_key(self, raw_key: str) -> bool:
        """Deactivate a key by its raw value. Returns True if found."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key_hash = ?",
                (_hash(raw_key),),
            )
        return cur.rowcount > 0


__all__ = ["ApiKeyStore", "generate_key"]
