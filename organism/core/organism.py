from __future__ import annotations

import atexit
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from organism.core.api import OrganismReply

if TYPE_CHECKING:
    from organism.backbone.base import LMBackend
    from organism.config import OrganismConfig
    from organism.core.chat.orchestrator import ChatOrchestrator
    from organism.core.memory.service.memory_facade import MemoryFacade
    from organism.core.stores import UnifiedStore

logger = logging.getLogger(__name__)


class Organism:
    """
    Organism v2 — composition-based.

    Two construction modes:

    Mode 1 — Dependency Injection (tests and custom setups):
        organism = Organism(
            lm_backend=my_lm,
            chat_orchestrator=my_orchestrator,
            tenant_id="t1",
        )

    Mode 2 — From store + lm (auto-builds orchestrator):
        store = UnifiedStore(db_path)
        organism = Organism(store=store, lm_backend=lm)

    Mode 3 — From config file:
        config = OrganismConfig.from_yaml("organism_config.yaml")
        organism = Organism.from_config(config)
    """

    def __init__(
        self,
        *,
        lm_backend: Optional["LMBackend"] = None,
        chat_orchestrator: Optional["ChatOrchestrator"] = None,
        memory_facade: Optional["MemoryFacade"] = None,
        store: Optional["UnifiedStore"] = None,
        tenant_id: str = "default",
    ):
        # Note: FactExtractor (async fact extraction) is only wired when using
        # Organism.from_config(). Direct construction via this __init__ does not
        # initialise FactExtractor — use from_config() for the full memory pipeline.
        self._tenant_id = tenant_id

        if chat_orchestrator is not None:
            self._orchestrator = chat_orchestrator
        elif store is not None and lm_backend is not None:
            from organism.core.memory.service.memory_facade import MemoryFacade
            from organism.core.chat.orchestrator import ChatOrchestrator

            _facade = memory_facade or MemoryFacade.from_store(store, tenant_id=tenant_id)
            self._orchestrator = ChatOrchestrator(
                memory_facade=_facade,
                lm_backend=lm_backend,
            )
        else:
            raise ValueError(
                "Organism requires either:\n"
                "  (a) chat_orchestrator=...\n"
                "  (b) store=... + lm_backend=...\n"
                "For auto-init from OrganismConfig, use Organism.from_config()."
            )

        logger.info("Organism v2 initialized (tenant=%s)", tenant_id)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        user_id: str,
        user_message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> OrganismReply:
        """
        Process a chat message (Core Layer — online operation).

        Pipeline: save → RAG retrieve → assemble → generate → save → RAG chunk write
        """
        result = self._orchestrator.process_chat(
            tenant_id=self._tenant_id,
            user_id=user_id,
            user_message=user_message,
            session_id=session_id,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            model_override=model_override,
        )
        return OrganismReply(reply=result["reply"])

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        limit: int = 8,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        RAG retrieval without LLM — returns relevant facts for proxy memory injection.

        Combines Tier 1 (RAG chunks) and Tier 2 (facts) results.
        Does not call LLM, does not modify session state.
        """
        try:
            assembled = self._orchestrator._memory.retrieval.retrieve(
                tenant_id=self._tenant_id,
                user_id=user_id,
                session_id=session_id or f"{user_id}_proxy",
                query=query,
            )
        except Exception as exc:
            logger.warning("retrieve_context failed user=%s: %s", user_id, exc)
            return []

        facts: List[str] = []

        if assembled.memory_block:
            facts.extend(
                line.strip()
                for line in assembled.memory_block.splitlines()
                if line.strip()
            )

        if assembled.context_block:
            facts.extend(
                line.strip()
                for line in assembled.context_block.splitlines()
                if line.strip()
            )

        return facts[:limit]

    def extract_session_facts(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        session_ts: Optional[int] = None,
    ) -> int:
        """
        Synchronously extract facts from a session's messages and store them.

        Used by the benchmark after fast-replay sessions (which bypass org.chat()
        so FactExtractor's daemon thread never fires automatically).

        session_ts: unix timestamp of the session — used as event_time fallback so
        every fact carries a date even when the LLM omits the 'when' field.

        Returns count of new facts inserted (0 if FactExtractor not configured).
        """
        fact_extractor = getattr(self._orchestrator, "_fact_extractor", None)
        if fact_extractor is None:
            return 0
        try:
            return fact_extractor.extract_and_store(
                session_id=session_id,
                user_id=user_id,
                tenant_id=self._tenant_id,
                messages=messages,
                session_ts=session_ts,
            )
        except Exception as exc:
            logger.warning("extract_session_facts failed user=%s session=%s: %s", user_id, session_id, exc)
            return 0

    def embed_session_chunks(self, user_id: str, session_id: str) -> int:
        """
        Bulk-embed all un-embedded chunks for a session in a single forward pass.

        Use after fast-replay (where skip_chunk_embedding=True was set) to convert
        N sequential embed() calls into one embed_batch() call.

        Returns count of chunks embedded (0 if no embedder or no pending chunks).
        """
        write_svc = getattr(self._orchestrator._memory, "write", None)
        embedder = getattr(write_svc, "_embedder", None) if write_svc else None
        if embedder is None:
            return 0
        store = self._orchestrator._memory.store
        chunks = getattr(store, "chunks", None)
        if chunks is None:
            return 0
        try:
            return chunks.embed_pending(
                session_id=session_id,
                tenant_id=self._tenant_id,
                user_id=user_id,
                embedder=embedder,
            )
        except Exception as exc:
            logger.warning("embed_session_chunks failed user=%s session=%s: %s", user_id, session_id, exc)
            return 0

    def remember(self, user_id: str, text: str) -> int:
        """
        Explicit hot-path memory write (bypasses consolidation pipeline).

        Creates a MemoryItem directly with category='note', confidence=1.0.

        Returns:
            ID of the created MemoryItem.
        """
        store = self._orchestrator._memory.store
        return store.memory_items.add(
            tenant_id=self._tenant_id,
            user_id=user_id,
            content=text,
            category="note",
            confidence=1.0,
        )

    def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List stored MemoryItems for a user.

        Returns raw dicts from UnifiedStore (keys: id, content, category,
        confidence, importance, created_at, tags, namespace, ...).

        Args:
            user_id:  User identifier.
            limit:    Max number of items to return (0 = all).
            category: Filter by category (e.g. 'note', 'fact').
        """
        store = self._orchestrator._memory.store
        items = store.memory_items.get_all(
            tenant_id=self._tenant_id,
            user_id=user_id,
            category=category,
        )
        return items[:limit] if limit else items

    def store_event(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
        source: str = "mcp",
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store content into memory and queue async fact extraction.

        Writes to the `messages` table and writes RAG chunks (Tier 1).
        FactExtractor fires automatically if configured.

        Returns:
            {"event_id": int, "queued_for_extraction": bool}
        """
        _tenant = tenant_id or self._tenant_id
        _session = session_id or f"{user_id}_mcp"
        store = self._orchestrator._memory.store
        meta = {"source": source}
        if metadata:
            meta.update(metadata)
        msg_id = store.messages.add(
            session_id=_session,
            tenant_id=_tenant,
            role="user",
            content=content,
            user_id=user_id,
            metadata=meta,
        )

        # Write verbatim RAG chunk (Tier 1) — synchronous, no LLM needed.
        # This makes assert_query work without waiting for FactExtractor.
        if hasattr(store, "chunks"):
            write_svc = getattr(self._orchestrator._memory, "write", None)
            embedder = getattr(write_svc, "_embedder", None)
            embedding = None
            if embedder is not None:
                try:
                    embedding = embedder.embed(content)
                except Exception:
                    pass
            store.chunks.add(
                tenant_id=_tenant,
                source_type=source,
                source_id=str(msg_id),
                chunk_index=0,
                content=content,
                embedding=embedding,
                session_id=_session,
                user_id=user_id,
                tags={"source": source},
            )

        fact_extractor = getattr(self._orchestrator, "_fact_extractor", None)
        queued = fact_extractor is not None
        return {"event_id": msg_id, "queued_for_extraction": queued}

    def query_memory(
        self,
        user_id: str,
        query: str,
        max_facts: int = 8,
        max_chunks: int = 5,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant facts and chunks — no LLM, pure read.

        Returns structured dict with separate facts/chunks lists, each item
        including id, content, score, source_type, source_session_id.
        """
        _tenant = tenant_id or self._tenant_id
        try:
            raw = self._orchestrator._memory.retrieval.retrieve_hybrid_only(
                tenant_id=_tenant,
                user_id=user_id,
                query=query,
                top_k=max_facts + max_chunks,
            )
        except Exception as exc:
            logger.warning("query_memory failed user=%s: %s", user_id, exc)
            raw = []

        facts: List[Dict[str, Any]] = []
        chunks: List[Dict[str, Any]] = []
        for r in raw:
            item = {
                "id": r["id"],
                "content": r["content"],
                "score": round(r["rrf_score"], 4),
                "source_type": r.get("source_type") or r["tier"],
                "source_session_id": r.get("source_session_id"),
            }
            if r["tier"] == "rag_chunk":
                if len(chunks) < max_chunks:
                    chunks.append(item)
            else:
                if len(facts) < max_facts:
                    facts.append(item)

        return {
            "facts": facts,
            "chunks": chunks,
            "total_facts": len(facts),
            "total_chunks": len(chunks),
        }

    def reset_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete all memory for a user across all tiers.

        Removes facts, messages, RAG chunks, memory items, and HNSW index.

        Returns:
            {"deleted": {"facts": int, "messages": int, "rag_chunks": int, "memory_items": int}}
        """
        _tenant = tenant_id or self._tenant_id
        store = self._orchestrator._memory.store
        n_facts = store.facts.delete_for_user(_tenant, user_id)
        n_messages = store.messages.delete_for_user(_tenant, user_id)
        n_chunks = store.chunks.delete_for_user(_tenant, user_id)
        n_items = store.memory_items.delete_for_user(_tenant, user_id)
        return {
            "deleted": {
                "facts": n_facts,
                "messages": n_messages,
                "rag_chunks": n_chunks,
                "memory_items": n_items,
            }
        }

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(self, user_id: str, title: Optional[str] = None) -> str:
        """Start a new session. Returns session_id (UUID)."""
        store = self._orchestrator._memory.store
        return store.sessions.create(
            tenant_id=self._tenant_id,
            user_id=user_id,
            title=title,
        )

    def end_session(self, user_id: str, session_id: str) -> None:
        """Close an existing session."""
        store = self._orchestrator._memory.store
        try:
            store.sessions.end(session_id=session_id, tenant_id=self._tenant_id)
        except Exception as exc:
            logger.warning("end_session failed user=%s session=%s: %s", user_id, session_id, exc)

    # ------------------------------------------------------------------
    # Internal access (for eval/debug tooling)
    # ------------------------------------------------------------------

    @property
    def memory_service(self) -> "MemoryFacade":
        """MemoryFacade — used by EvalAdapter and debug tooling."""
        return self._orchestrator._memory

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Graceful shutdown — flush in-flight fact extractions before exit."""
        fact_extractor = getattr(self._orchestrator, "_fact_extractor", None)
        if fact_extractor is not None:
            fact_extractor.shutdown()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "OrganismConfig",
        db_path: Optional[str] = None,
        tenant_id: str = "default",
    ) -> "Organism":
        """
        Auto-initialize from OrganismConfig.

        Reads model type/name from config.base_model.
        DB path: explicit db_path arg > CoreConfig.db_path (canonical, model-independent).
        Embedder is initialized if config.rag.embedder_enabled is True.
        """
        from organism import backbone
        from organism.core.stores import UnifiedStore
        from organism.core.config import CoreConfig
        from organism.core.memory.service.memory_facade import MemoryFacade
        from organism.core.chat.orchestrator import ChatOrchestrator
        from pathlib import Path

        if db_path is not None:
            resolved_db_path = Path(db_path)
        else:
            core_cfg = CoreConfig()
            resolved_db_path = Path(core_cfg.db_path)
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path = resolved_db_path  # type: ignore[assignment]

        memory_mode = getattr(config, "memory_mode", "t2")
        logger.info("Organism memory_mode=%s", memory_mode)

        lm = backbone.create_lm_backend(config)
        store = UnifiedStore(resolved_db_path)

        core_cfg = CoreConfig()

        # Initialize embedder for vector search (Tier 1 chunks + Tier 2 facts)
        embedder = None
        if getattr(config.rag, "embedder_enabled", True):
            embedder_base_url = getattr(config.rag, "embedder_base_url", None)
            if embedder_base_url:
                try:
                    from organism.core.embedding.vllm_embedder import VLLMEmbedder
                    embedder = VLLMEmbedder(
                        base_url=embedder_base_url,
                        model_name=config.rag.embedder_model,
                        dim=getattr(config.rag, "embedder_dim", 1024),
                    )
                    logger.info("VLLMEmbedder initialized: %s @ %s", config.rag.embedder_model, embedder_base_url)
                except Exception as exc:
                    logger.warning("Failed to init VLLMEmbedder: %s — vector search disabled", exc)
            else:
                try:
                    from organism.core.embedding.qwen3_embedder import Qwen3Embedder
                    embedder = Qwen3Embedder(
                        model_name=config.rag.embedder_model,
                    )
                    logger.info("Embedder initialized: %s", config.rag.embedder_model)
                except Exception as exc:
                    logger.warning("Failed to initialize embedder: %s — vector search disabled", exc)

        facade = MemoryFacade.from_store(
            store, tenant_id=tenant_id, embedder=embedder, core_config=core_cfg,
        )

        # Initialize dedicated fact LLM backend if configured (separate from main LLM)
        fact_lm = lm
        fact_llm_cfg = getattr(config, "fact_llm", None)
        if fact_llm_cfg is not None:
            try:
                fact_lm = backbone.create_lm_backend_from_backbone(fact_llm_cfg)
                logger.info(
                    "FactExtractor LLM initialized: type=%s model=%s",
                    fact_llm_cfg.type,
                    getattr(fact_llm_cfg, "model_path", None) or getattr(fact_llm_cfg, "model_name", ""),
                )
            except Exception as exc:
                logger.warning("Failed to init fact_llm backend: %s — falling back to main LLM", exc)

        # Initialize FactExtractor — Tier 2+; skipped in "t1" mode
        fact_extractor = None
        if memory_mode != "t1" and embedder is not None and lm is not None:
            from organism.core.memory.service.fact_extractor import FactExtractor
            from organism.core.memory.service.profile_updater import ProfileUpdater
            profile_updater = ProfileUpdater(fact_store=store.facts)
            fact_extractor = FactExtractor(
                lm_backend=fact_lm,
                embedder=embedder,
                fact_store=store.facts,
                profile_updater=profile_updater,
            )
            dedicated = fact_lm is not lm
            logger.info("FactExtractor initialized (ProfileUpdater wired, dedicated_llm=%s)", dedicated)
        elif memory_mode == "t1":
            logger.info("FactExtractor skipped (memory_mode=t1)")

        orchestrator = ChatOrchestrator(
            memory_facade=facade,
            lm_backend=lm,
            fact_extractor=fact_extractor,
        )

        instance = cls(chat_orchestrator=orchestrator, tenant_id=tenant_id)
        atexit.register(instance.close)
        return instance


__all__ = ["Organism"]
