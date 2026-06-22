from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .base_store import BaseStore

logger = logging.getLogger(__name__)


class MessageStore:
    """
    Store component for chat messages.

    Messages are episodic records of user-assistant dialogue.
    The last N messages form Tier 0 (Working Memory) in the RAG pipeline.
    """

    def __init__(self, base: BaseStore):
        self._base = base

    def add(
        self,
        session_id: str,
        tenant_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a new message.

        Args:
            session_id: Session UUID.
            tenant_id: Tenant identifier.
            role: Message role ('user', 'assistant', 'system').
            content: Message text.
            user_id: Optional user identifier.
            tokens_used: Optional token count.
            metadata: Optional JSON-serializable metadata.

        Returns:
            The new message ID.
        """
        meta_json = json.dumps(metadata) if metadata else None
        self._base.execute(
            """
            INSERT INTO messages (session_id, tenant_id, user_id, role, content, tokens_used, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, tenant_id, user_id, role, content, tokens_used, meta_json, int(time.time())),
            commit=True,
        )
        return self._base.last_insert_rowid()

    def get_recent(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent messages for a user (Tier 0 source).

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            limit: Maximum number of messages to return.
            session_id: Optional session filter.

        Returns:
            List of message dicts, ordered oldest-first (chronological).
        """
        if session_id:
            cur = self._base.execute(
                """
                SELECT * FROM messages
                WHERE tenant_id = ? AND user_id = ? AND session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, user_id, session_id, limit),
            )
        else:
            cur = self._base.execute(
                """
                SELECT * FROM messages
                WHERE tenant_id = ? AND user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, user_id, limit),
            )
        rows = cur.fetchall()
        # Reverse to get chronological order
        return [dict(r) for r in reversed(rows)]  # type: ignore[return-value]

    def get_by_session(
        self,
        session_id: str,
        tenant_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get messages in a session, ordered chronologically.

        Args:
            session_id: Session UUID.
            tenant_id: Tenant identifier.
            limit: If provided, return only the last N messages.

        Returns:
            List of message dicts in chronological order.
        """
        if limit is not None:
            # Get last N messages: subquery DESC then reverse
            cur = self._base.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE session_id = ? AND tenant_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                ) ORDER BY created_at ASC, id ASC
                """,
                (session_id, tenant_id, limit),
            )
        else:
            cur = self._base.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND tenant_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id, tenant_id),
            )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def get_by_span(
        self,
        tenant_id: str,
        user_id: str,
        start_id: int,
        end_id: int,
    ) -> List[Dict[str, Any]]:
        """Return messages with id BETWEEN start_id AND end_id (inclusive)."""
        cur = self._base.execute(
            """
            SELECT id, session_id, tenant_id, user_id, role, content, created_at, metadata
            FROM messages
            WHERE tenant_id = ? AND user_id = ? AND id BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (tenant_id, user_id, start_id, end_id),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def get_unindexed(
        self,
        tenant_id: str,
        after_id: int = 0,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get messages not yet indexed by the RAG Indexer.

        The RAG Indexer tracks the last processed message ID;
        this method returns messages with id > after_id.

        Args:
            tenant_id: Tenant identifier.
            after_id: Return messages with id greater than this.
            limit: Maximum number of messages.

        Returns:
            List of message dicts, ordered by id ASC.
        """
        cur = self._base.execute(
            """
            SELECT * FROM messages
            WHERE tenant_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (tenant_id, after_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def count(self, tenant_id: str, user_id: Optional[str] = None) -> int:
        """Count messages for a tenant (optionally filtered by user)."""
        if user_id:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE tenant_id = ? AND user_id = ?",
                (tenant_id, user_id),
            )
        else:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE tenant_id = ?",
                (tenant_id,),
            )
        return cur.fetchone()["cnt"]  # type: ignore[index]

    def delete_for_user(self, tenant_id: str, user_id: str) -> int:
        """Delete all messages for a user. Returns count of deleted rows."""
        cur = self._base.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id),
        )
        n: int = cur.fetchone()["cnt"]  # type: ignore[index]
        if n:
            self._base.execute(
                "DELETE FROM messages WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
                commit=True,
            )
        return n


__all__ = ["MessageStore"]
