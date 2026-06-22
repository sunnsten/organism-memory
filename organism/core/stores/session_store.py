from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .base_store import BaseStore

logger = logging.getLogger(__name__)


class SessionStore:
    """Store component for user sessions."""

    def __init__(self, base: BaseStore):
        self._base = base

    def create(
        self,
        tenant_id: str,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session.

        Returns:
            The session UUID.
        """
        session_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata) if metadata else None
        self._base.execute(
            """
            INSERT INTO sessions (id, tenant_id, user_id, started_at, title, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, tenant_id, user_id, int(time.time()), title, meta_json),
            commit=True,
        )
        return session_id

    def get(self, session_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        cur = self._base.execute(
            "SELECT * FROM sessions WHERE id = ? AND tenant_id = ?",
            (session_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def end(self, session_id: str, tenant_id: str) -> None:
        """Mark a session as closed."""
        self._base.execute(
            """
            UPDATE sessions SET ended_at = ?, status = 'closed'
            WHERE id = ? AND tenant_id = ?
            """,
            (int(time.time()), session_id, tenant_id),
            commit=True,
        )

    def get_active(
        self,
        tenant_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        cur = self._base.execute(
            """
            SELECT * FROM sessions
            WHERE tenant_id = ? AND user_id = ? AND status = 'active'
            ORDER BY started_at DESC
            """,
            (tenant_id, user_id),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]


__all__ = ["SessionStore"]
