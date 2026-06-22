from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from organism.shared.domain import WorkingMemoryPack
    from organism.core.stores import UnifiedStore
    from organism.core.config import CoreConfig

logger = logging.getLogger(__name__)


class WorkingMemoryService:
    """
    Manages working memory (Core layer — online path).

    Responsibilities:
    - get_working_memory(): retrieve recent messages + summary
    """

    def __init__(
        self,
        store: "UnifiedStore",
        config: "CoreConfig",
    ):
        """
        Args:
            store: UnifiedStore for accessing context summaries
            config: CoreConfig with settings for online operations
        """
        self._store = store
        self._config = config

    def get_working_memory(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        recent_k: int = 5,
    ) -> "WorkingMemoryPack":
        """
        Retrieve working memory for a user.

        Includes:
        - Recent messages from the session
        - Short summary generated from recent messages

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            session_id: Session identifier
            recent_k: Number of recent messages to load

        Returns:
            WorkingMemoryPack with the assembled working memory.
        """
        from organism.shared.domain import WorkingMemoryPack

        # 1. Fetch recent messages from session
        recent_refs = []
        recent_messages = []
        try:
            messages = self._store.messages.get_by_session(
                session_id, tenant_id, limit=recent_k,
            )
            recent_refs = [str(msg["id"]) for msg in messages]
            recent_messages = messages
            logger.debug("WorkingMemoryService: loaded %d recent messages", len(messages))
        except Exception as e:
            logger.warning("Failed to get recent messages: %s", e)

        # 3. Generate short_summary from recent messages
        # NOTE: simple placeholder for now; future: use LM for summarization.
        short_summary = self._generate_summary(recent_messages)

        return WorkingMemoryPack(
            ssm_state=None,
            short_summary=short_summary,
            recent_refs=recent_refs,
            trace={"loaded_messages": len(recent_messages)},
        )

    def _generate_summary(self, messages: list) -> Optional[str]:
        """
        Generate a short summary from recent messages.

        TODO: Use LM backend for proper summarization.
        """
        if not messages:
            return None

        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        return f"Recent context: {len(user_msgs)} user messages, {len(assistant_msgs)} assistant messages"


__all__ = ["WorkingMemoryService"]
