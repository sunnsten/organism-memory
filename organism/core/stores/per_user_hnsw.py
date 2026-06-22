from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from .base_store import BaseStore

logger = logging.getLogger(__name__)

try:
    import vectorlite_py  # noqa: F401
    VECTORLITE_AVAILABLE = True
except ImportError:
    VECTORLITE_AVAILABLE = False

_HNSW_PARAMS = "hnsw(max_elements=50000, ef_construction=200, M=32)"


class PerUserHNSW:
    """
    Per-user vectorlite HNSW index.

    Each (tenant_id, user_id) gets its own virtual table named
    vec_u{sha1_16hex}. The SHA-1 hash is over "tenant_id:user_id"
    so the name is SQL-safe regardless of input characters.

    Usage:
        hnsw = PerUserHNSW(base_store, dim=1024)
        hnsw.insert(rowid=42, embedding=vec, tenant_id="t1", user_id="alice")
        results = hnsw.search(query_vec, "t1", "alice", limit=20)
        # → List[(rowid, distance)]
    """

    def __init__(self, base: BaseStore, dim: int = 1024) -> None:
        self._base = base
        self._dim = dim
        # Per-instance cache of already-created tables (avoids redundant CREATE IF NOT EXISTS)
        self._created_tables: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def table_name(self, tenant_id: str, user_id: str) -> str:
        """Return the deterministic, SQL-safe table name for this user."""
        key = f"{tenant_id}:{user_id}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return f"vec_u{digest}"

    def insert(
        self,
        rowid: int,
        embedding: np.ndarray,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Insert or replace a vector in the user's HNSW index."""
        if not VECTORLITE_AVAILABLE:
            return
        table = self._ensure_table(tenant_id, user_id)
        if table is None:
            return
        emb_blob = embedding.astype(np.float32).tobytes()
        try:
            # DELETE before INSERT — vectorlite does not support UPDATE in place
            self._base.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self._base.execute(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                (rowid, emb_blob),
                commit=True,
            )
        except Exception as exc:
            logger.warning("PerUserHNSW.insert failed for table=%s rowid=%d: %s", table, rowid, exc)

    def delete(
        self,
        rowid: int,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Remove a vector from the user's HNSW index."""
        if not VECTORLITE_AVAILABLE:
            return
        table = self._table_if_exists(tenant_id, user_id)
        if table is None:
            return
        try:
            self._base.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,), commit=True)
        except Exception as exc:
            logger.warning("PerUserHNSW.delete failed for table=%s rowid=%d: %s", table, rowid, exc)

    def reset_user(self, tenant_id: str, user_id: str) -> None:
        """Drop the HNSW virtual table for a user and remove from registry."""
        if not VECTORLITE_AVAILABLE:
            return
        table = self.table_name(tenant_id, user_id)
        try:
            self._base.execute(f"DROP TABLE IF EXISTS {table}", commit=False)
            self._base.execute(
                "DELETE FROM user_hnsw_registry WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
                commit=True,
            )
            self._created_tables.discard(table)
        except Exception as exc:
            logger.warning("PerUserHNSW.reset_user failed for %s/%s: %s", tenant_id, user_id, exc)

    def search(
        self,
        query: np.ndarray,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[Tuple[int, float]]:
        """Return [(rowid, distance)] sorted by distance ASC. Empty list if unavailable."""
        if not VECTORLITE_AVAILABLE:
            return []
        table = self._table_if_exists(tenant_id, user_id)
        if table is None:
            return []
        q_blob = (query / (np.linalg.norm(query) + 1e-9)).astype(np.float32).tobytes()
        try:
            cur = self._base.execute(
                f"SELECT rowid, distance FROM {table} "
                f"WHERE knn_search(embedding, knn_param(?, ?)) ORDER BY distance ASC",
                (q_blob, limit),
            )
            rows: list[Any] = cur.fetchall()
            return [(r["rowid"], r["distance"]) for r in rows]
        except Exception as exc:
            logger.warning("PerUserHNSW.search failed for table=%s: %s", table, exc)
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_table(self, tenant_id: str, user_id: str) -> Optional[str]:
        """Create the user's HNSW virtual table if it doesn't exist. Returns table name."""
        table = self.table_name(tenant_id, user_id)
        if table in self._created_tables:
            return table
        try:
            self._base.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vectorlite(
                    embedding float32[{self._dim}],
                    {_HNSW_PARAMS}
                )"""
            )
            self._base.execute(
                """INSERT OR IGNORE INTO user_hnsw_registry(tenant_id, user_id, table_name)
                   VALUES (?, ?, ?)""",
                (tenant_id, user_id, table),
                commit=True,
            )
            self._created_tables.add(table)
            return table
        except Exception as exc:
            logger.warning("PerUserHNSW: failed to create table %s: %s", table, exc)
            return None

    def _table_if_exists(self, tenant_id: str, user_id: str) -> Optional[str]:
        """Return table name only if it's already been created this session."""
        table = self.table_name(tenant_id, user_id)
        if table in self._created_tables:
            return table
        # Check registry (handles process restart — table exists but not in _created_tables)
        try:
            row = self._base.execute(
                "SELECT table_name FROM user_hnsw_registry WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            ).fetchone()
            if row:
                self._created_tables.add(table)
                return table
        except Exception:
            pass
        return None


__all__ = ["PerUserHNSW", "VECTORLITE_AVAILABLE"]
