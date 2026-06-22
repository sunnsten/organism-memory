from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from organism.core.config import RAGConfig
    from organism.core.memory.service.memory_facade import MemoryFacade
    from organism.core.memory.service.fact_extractor import FactExtractor

from organism.core.chat.token_budget import count_messages_tokens, trim_messages_to_token_budget

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Chat pipeline orchestrator (composition-based).

    Accepts MemoryFacade + LMBackend.
    Does not know about SSM, importance, surprisal — those are Research Layer.
    """

    def __init__(
        self,
        memory_facade: "MemoryFacade",
        lm_backend: Any,
        rag_config: Optional["RAGConfig"] = None,
        fact_extractor: Optional["FactExtractor"] = None,
    ):
        self._memory = memory_facade
        self.lm = lm_backend
        if rag_config is None:
            from organism.core.config import RAGConfig as _RAGConfig
            rag_config = _RAGConfig()
        self._rag_config = rag_config
        # FactExtractor — None means disabled (default)
        self._fact_extractor = fact_extractor

    def process_chat(
        self,
        tenant_id: str,
        user_id: str,
        user_message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full chat pipeline.

        Returns:
            dict with: reply, session_id, user_message_id, assistant_message_id
        """
        if not user_message.strip():
            raise ValueError("user_message cannot be empty")

        session_id = session_id or f"{user_id}_default"

        logger.debug(
            "ChatOrchestrator: tenant=%s user=%s session=%s msg_len=%d",
            tenant_id, user_id, session_id, len(user_message),
        )

        # 1. Save user message
        user_message_id = self._memory.store.messages.add(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role="user",
            content=user_message,
        )

        # 2. Retrieve context: Tier 0 (working memory) + Tier 1 (RAG chunks) + Tier 2 (facts)
        working_memory_limit = self._rag_config.context_window_working_memory_limit or 5

        logger.debug(
            "ChatOrchestrator: working_memory_limit=%s",
            working_memory_limit,
        )

        assembled = self._memory.retrieval.retrieve(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            query=user_message,
            system_prompt=system_prompt or "",
            working_memory_limit=working_memory_limit,
        )

        # 3. Generate via LM (messages format)
        messages = assembled.to_messages()

        response_text = self.lm.generate(messages, max_new_tokens=max_new_tokens or 512, model_override=model_override)

        # 4. Save assistant message
        assistant_message_id = self._memory.store.messages.add(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role="assistant",
            content=response_text,
        )

        # 5. Persist event (RAG chunk write via WriteService)
        self._persist_experience(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_reply=response_text,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )

        # 6. Non-blocking fact extraction
        if self._fact_extractor is not None:
            messages_for_extraction = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
            ]
            self._fact_extractor.extract_and_store_later(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                messages=messages_for_extraction,
            )

        return {
            "reply": response_text,
            "session_id": session_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
        }

    def _persist_experience(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        user_message_id: int,
        assistant_message_id: int,
    ) -> None:
        """Create and persist EventRecord via WriteService (writes RAG chunks)."""
        from organism.shared.domain import EventRecord, ContextMeta

        context_meta = ContextMeta(
            system_hash="",
            memory_ids=[],
            chat_message_id_span=(user_message_id, assistant_message_id),
            memory_id_space=None,
        )

        text_preview = f"{user_message}\n{assistant_reply}"
        if len(text_preview) > 500:
            text_preview = text_preview[:500]

        event = EventRecord(
            id=None,
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            input_text=user_message,
            output_text=assistant_reply,
            kind="interaction",
            source="chat",
            importance=0.5,
            surprisal_norm=None,
            attention_focus=None,
            used_memories=[],
            used_memories_space=None,
            context_meta=context_meta,
            text_preview=text_preview,
            embedding=None,
            embedding_dim=None,
            embedding_dtype="float32",
            embedding_l2norm=False,
        )

        self._memory.write.append_event(event, tenant_id=tenant_id)


__all__ = ["ChatOrchestrator"]
