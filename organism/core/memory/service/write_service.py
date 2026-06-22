from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from organism.shared.domain import EventRecord
    from organism.core.stores import UnifiedStore
    from organism.core.config import CoreConfig

logger = logging.getLogger(__name__)

_MAX_ROUND_CHARS = 1200   # max chars per chunk; longer → split at sentence boundaries
_MIN_ROUND_CHARS = 50     # very short turns are not split
_SOFT_SPLIT_MARGIN = 150  # allowed overshoot for the last sentence


@dataclass(frozen=True)
class RoundChunk:
    content: str
    round_id: str
    round_boundary: bool    # True = full round (not subchunk)
    round_part: int         # 0, 1, 2, ...
    round_parts_total: int


def _sentences_split(text: str, max_chars: int, margin: int) -> list[str]:
    """Split text at sentence boundaries, respecting max_chars + margin."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts: list[str] = []
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= max_chars + margin:
            current = candidate
        else:
            if current:
                parts.append(current)
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    parts.append(sent[i:i + max_chars])
                current = ""
            else:
                current = sent
    if current:
        parts.append(current)
    return parts if parts else [text[:max_chars]]


def _split_by_rounds(
    input_text: str,
    output_text: str,
    date_prefix: str,
    round_id: str,
) -> list[RoundChunk]:
    """One User+Assistant turn → one or more RoundChunks.

    If the full round fits in _MAX_ROUND_CHARS → single chunk (round_boundary=True).
    Otherwise splits User and Assistant sections separately at sentence boundaries,
    keeping role markers intact. Each subchunk gets date_prefix prepended.
    """
    # Double date prefix: temporal signal appears before both User and Assistant
    full_text = f"{date_prefix}User: {input_text}\n{date_prefix}Assistant: {output_text}".strip()

    if len(full_text) <= _MAX_ROUND_CHARS:
        return [RoundChunk(
            content=full_text,
            round_id=round_id,
            round_boundary=True,
            round_part=0,
            round_parts_total=1,
        )]

    # Split User and Assistant parts separately to preserve role boundaries
    user_section = f"{date_prefix}User: {input_text}".strip()
    asst_section = f"{date_prefix}Assistant: {output_text}".strip()

    parts: list[str] = []
    for section in (user_section, asst_section):
        if len(section) <= _MAX_ROUND_CHARS:
            parts.append(section)
        else:
            parts.extend(_sentences_split(section, _MAX_ROUND_CHARS, _SOFT_SPLIT_MARGIN))

    # Ensure date_prefix on every subchunk for temporal reasoning
    final: list[str] = []
    for p in parts:
        if p.startswith("[") and "]" in p[:15]:
            final.append(p)
        else:
            final.append(f"{date_prefix}{p}")

    total = len(final)
    return [
        RoundChunk(
            content=text,
            round_id=round_id,
            round_boundary=False,
            round_part=i,
            round_parts_total=total,
        )
        for i, text in enumerate(final)
    ]


class WriteService:

    def __init__(
        self,
        store: "UnifiedStore",
        config: "CoreConfig",
        embedder=None,
    ):
        self._store = store
        self._config = config
        self._embedder = embedder

    def append_event(
        self,
        event: "EventRecord",
        tenant_id: str,
        skip_chunk_embedding: bool = False,
    ) -> Optional[str]:
        """
        Adds a new event to the memory system.

        Process:
        1. Check importance threshold (gating)
        2. Write verbatim RAG chunks (Tier 1)
        3. Return synthetic block_id or None if filtered
        """
        # Check importance threshold for source
        threshold = self._get_write_threshold_for_source(event.source)

        if event.importance < threshold:
            logger.debug(
                "Write skipped: importance %.3f < threshold %.3f (source=%s, kind=%s)",
                event.importance, threshold, event.source, event.kind,
            )
            from organism.shared.analytics import analytics
            analytics.metric_write(tenant_id, importance=event.importance, filtered=True)
            return None

        import uuid
        block_id = str(uuid.uuid4())
        logger.debug(
            "WriteService: processing event user=%s importance=%.3f source=%s kind=%s id=%s",
            event.user_id, event.importance, event.source, event.kind, block_id,
        )

        # Write verbatim chunks (Tier 1) — synchronous, no LLM
        if hasattr(self._store, "chunks"):
            self._write_chunks(
                tenant_id=tenant_id,
                source_id=block_id,
                session_id=event.session_id or "",
                input_text=event.input_text or "",
                output_text=event.output_text or "",
                timestamp=event.timestamp,
                user_id=event.user_id,
                skip_embedding=skip_chunk_embedding,
            )

        from organism.shared.analytics import analytics
        analytics.metric_write(tenant_id, importance=event.importance, filtered=False)

        return block_id

    def _write_chunks(
        self,
        tenant_id: str,
        source_id: str,
        session_id: str,
        input_text: str,
        output_text: str,
        timestamp: Optional[float] = None,
        user_id: Optional[str] = None,
        skip_embedding: bool = False,
    ) -> None:
        """Round-level chunking: one User+Assistant turn → RoundChunk(s) with round metadata in tags."""
        from datetime import datetime
        date_prefix = ""
        date_str = ""
        if timestamp is not None:
            try:
                date_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
                date_prefix = f"[{date_str}] "
            except Exception:
                pass

        if not input_text and not output_text:
            return

        round_id = f"{session_id}_{int(timestamp)}" if timestamp else f"{session_id}_0"
        round_chunks = _split_by_rounds(input_text, output_text, date_prefix, round_id)

        rows = []
        for rc in round_chunks:
            emb = None
            if not skip_embedding and self._embedder is not None:
                try:
                    emb = self._embedder.embed(rc.content)
                except Exception:
                    pass

            rows.append({
                "tenant_id": tenant_id,
                "user_id": user_id,
                "source_type": "experience_block",
                "source_id": source_id,
                "chunk_index": rc.round_part,
                "content": rc.content,
                "embedding": emb,
                "session_id": session_id,
                "created_at": int(timestamp) if timestamp else None,
                "tags": {
                    "round_id": rc.round_id,
                    "round_boundary": rc.round_boundary,
                    "round_part": rc.round_part,
                    "round_parts_total": rc.round_parts_total,
                    "event_date": date_str,
                    "session": session_id,
                },
            })

        self._store.chunks.add_batch(rows)
        logger.debug("WriteService: wrote %d round-chunks for block %s", len(rows), source_id)

    def _get_write_threshold_for_source(self, source: str) -> float:
        base = self._config.block_min_importance
        mult = {
            "remember": self._config.source_multiplier_remember,
            "chat": self._config.source_multiplier_chat,
            "manual": self._config.source_multiplier_manual,
        }.get(source, 1.0)

        return base * mult


__all__ = ["WriteService"]
