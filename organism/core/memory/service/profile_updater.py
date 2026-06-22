from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from organism.shared.analytics.memory_metrics import record_profile_update

if TYPE_CHECKING:
    from organism.core.stores.fact_store import FactStore

logger = logging.getLogger(__name__)

_PROFILE_PATTERNS = {
    "name": [
        r"user'?s?\s+name\s+is\s+([\w\s]{2,25})",
        r"my\s+name\s+is\s+([\w\s]{2,25})",
        r"call\s+me\s+([\w\s]{2,15})",
    ],
    "location": [
        r"user\s+lives?\s+in\s+([\w\s,]{3,30})",
        r"user\s+is\s+from\s+([\w\s,]{3,30})",
        r"located\s+in\s+([\w\s,]{3,30})",
        r"based\s+in\s+([\w\s,]{3,30})",
    ],
    "profession": [
        r"user\s+works?\s+as\s+(?:a\s+)?([\w\s]{3,30})",
        r"user\s+is\s+a\s+([\w\s]{3,30}(?:engineer|developer|designer|manager|scientist|analyst|teacher|doctor|lawyer))",
        r"user'?s?\s+job\s+is\s+([\w\s]{3,30})",
    ],
    "language": [
        r"user\s+(?:speaks?|prefers?|uses?)\s+([\w\s]{2,20})\s+(?:language|as)",
        r"native\s+language\s+is\s+([\w]{2,20})",
    ],
}


class ProfileUpdater:
    """
    Batch job: scan accumulated facts → update user_profile key-value table.
    No LLM calls — pure regex matching on already-extracted facts.
    """

    def __init__(self, fact_store: "FactStore"):
        self._store = fact_store

    def update_user(self, user_id: str, tenant_id: str) -> int:
        """Scan facts for this user and upsert profile keys. Returns count of keys updated."""
        cur = self._store._base.execute(
            """SELECT id, content, confirmed_count FROM facts
               WHERE tenant_id=? AND user_id=?
                 AND (valid_until IS NULL OR valid_until > strftime('%s','now'))
               ORDER BY confirmed_count DESC, created_at DESC
               LIMIT 300""",
            (tenant_id, user_id),
        )
        facts: list[dict] = [dict(r) for r in cur.fetchall()]  # type: ignore[misc]

        updated = 0
        for key, patterns in _PROFILE_PATTERNS.items():
            for fact in facts:
                value = self._extract(fact["content"], patterns)
                if value:
                    confidence = min(1.0, 0.65 + fact["confirmed_count"] * 0.05)
                    self._store.upsert_profile(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        key=key,
                        value=value.strip().rstrip("."),
                        confidence=confidence,
                        source_fact_id=fact["id"],
                    )
                    record_profile_update(key)
                    updated += 1
                    break  # One match per key is enough

        logger.debug("profile_updater: user=%s updated=%d keys", user_id, updated)
        return updated

    @staticmethod
    def _extract(content: str, patterns: list) -> str | None:
        for pattern in patterns:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None


__all__ = ["ProfileUpdater"]
