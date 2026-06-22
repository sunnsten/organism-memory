from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from organism.shared.analytics.memory_metrics import (
    record_extraction_error,
    record_extraction_latency,
    record_fact_confirmed,
    record_fact_new,
    record_facts_extracted,
)

if TYPE_CHECKING:
    from organism.core.stores.fact_store import FactStore

logger = logging.getLogger(__name__)

_FACT_PROMPT = """Extract {n_facts} atomic facts about the user from this conversation.

Rules:
- Each fact is ONE sentence, max 30 words
- Third person: "User prefers...", "User works at...", "User moved to..."
- Only facts explicitly stated or clearly implied by the user
- Skip trivial turns ("User said hello", "User asked about the weather today")
- Categories: fact | preference | habit | plan | instruction
- REQUIRED "when": time of the event/fact. Use exact date if known ("March 2024", "2023"),
  otherwise use relative ("recently", "last year", "childhood") or "unknown". NEVER omit.
- Optional "supersedes_topic": if this fact updates previous info, name the topic
  (e.g. "location", "job", "profession", "preference", "plan") so old facts can be retired
- Output JSON array ONLY:
  [{{"content": "...", "category": "...", "when": "recently", "supersedes_topic": "..."}}]
  (omit "supersedes_topic" if not applicable; "when" is always required)

Conversation:
{conversation}

Facts (JSON only):"""

_MAX_CHARS = 12000         # ↑ from 3000 — captures full LoCoMo sessions without truncating early facts

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _recover_partial_json(text: str) -> list:
    """Recover complete JSON objects from a truncated array string."""
    result = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(text[obj_start:i + 1])
                    if isinstance(obj, dict) and "content" in obj:
                        result.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return result


def _parse_when(when: str | None) -> int | None:
    """Parse LLM 'when' string → unix timestamp. Returns None if unparseable."""
    if not when:
        return None
    w = when.strip().lower()
    m = re.search(r'(\d{4})', w)
    year = int(m.group(1)) if m else None
    month = 1
    for name, num in _MONTH_NAMES.items():
        if name in w:
            month = num
            break
    m2 = re.search(r'-(\d{2})(?:\b|$)', w)
    if m2:
        month = int(m2.group(1))
    if year:
        try:
            return int(datetime(year, month, 1).timestamp())
        except Exception:
            return None
    return None


class FactExtractor:
    """
    Extracts atomic facts from a session and stores them in FactStore.
    Runs in a background thread after each chat turn.
    ~250 tokens per session vs 500-1000 for consolidation.
    """

    def __init__(self, lm_backend, embedder, fact_store: "FactStore",
                 profile_updater=None):
        self._lm = lm_backend
        self._embedder = embedder
        self._store = fact_store
        self._profile_updater = profile_updater
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fact-extractor")
        self._closed = False

    def extract_and_store(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str,
        messages: List[dict],
        session_ts: Optional[int] = None,
    ) -> int:
        """
        Synchronous extraction. Returns count of new facts inserted.
        Call extract_and_store_later() for non-blocking background execution.
        session_ts: unix timestamp of the session — used as event_time fallback when
        the LLM does not return an explicit 'when' field.
        """
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        if not user_turns:
            return 0

        conversation = "\n".join(user_turns)[-_MAX_CHARS:]
        # Scale fact count with conversation length: long sessions (10+ turns) get more facts
        n_turns = len(user_turns)
        n_facts = "3-7" if n_turns <= 5 else ("5-10" if n_turns <= 15 else "8-15")
        t0 = time.perf_counter()
        raw_facts = self._call_llm(conversation, n_facts=n_facts)
        record_extraction_latency(time.perf_counter() - t0)
        if not raw_facts:
            return 0

        # Build session date prefix for embedding (e.g. "[2023-03]") from session_ts
        session_date_prefix = (
            datetime.fromtimestamp(session_ts).strftime("%Y-%m")
            if session_ts else None
        )

        # Filter valid items first, then embed in one batch (single GPU forward pass)
        valid_items = [
            (
                (item.get("content") or "").strip(),
                item.get("category", "fact"),
                (item.get("when") or "").strip(),
                (item.get("supersedes_topic") or "").strip(),
                # Use LLM 'when' if available, otherwise fall back to session timestamp
                _parse_when(item.get("when")) or session_ts,
            )
            for item in raw_facts
            if isinstance(item, dict) and len((item.get("content") or "").strip()) >= 10
        ]
        if not valid_items:
            return 0

        record_facts_extracted(len(valid_items))

        # Prepend temporal markers before embedding so the vector captures time context.
        # Use LLM 'when' string if present, else session date derived from session_ts.
        contents = [
            f"[{when}] {content}" if when else (
                f"[{session_date_prefix}] {content}" if session_date_prefix else content
            )
            for content, _, when, _, _ in valid_items
        ]
        embeddings = self._embed_batch(contents)

        count = 0
        # Track IDs added in this batch so we don't deduplicate against them
        # within the same extraction call (two distinct facts may share embeddings
        # in tests, but are genuinely different user statements).
        newly_added_ids: set = set()
        for (_, category, when, supersedes_topic, event_time), content, embedding in zip(valid_items, contents, embeddings):
            # Invalidate old facts on the same named topic before inserting new one
            if supersedes_topic:
                self._store.invalidate_by_topic(supersedes_topic, tenant_id, user_id)

            # Preferences supersede aggressively (avoid accumulating duplicates);
            # other categories use a higher threshold so distinct counting facts
            # (e.g. separate purchase or trip events) are not incorrectly superseded.
            supersede_threshold = 0.70 if category in ("preference", "habit") else 0.82
            fact_id, is_new = self._store.add_or_supersede(
                tenant_id=tenant_id,
                user_id=user_id,
                content=content,
                old_embedding=embedding,
                new_embedding=embedding,
                category=category,
                source_session_id=session_id,
                exclude_ids=newly_added_ids,
                event_time=event_time,
                event_date_raw=when if when else None,
                supersede_threshold=supersede_threshold,
            )
            if is_new and fact_id not in newly_added_ids:
                count += 1
                newly_added_ids.add(fact_id)
                record_fact_new()
            else:
                record_fact_confirmed()

        logger.debug("fact_extractor: session=%s new=%d user=%s", session_id, count, user_id)
        if count > 0 and self._profile_updater is not None:
            try:
                self._profile_updater.update_user(user_id=user_id, tenant_id=tenant_id)
            except Exception:
                logger.warning("fact_extractor: profile update failed", exc_info=True)
        return count

    def extract_and_store_later(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str,
        messages: List[dict],
        session_ts: Optional[int] = None,
    ) -> None:
        """Submit extraction to background thread. No-op if already shut down."""
        if self._closed:
            return
        self._executor.submit(
            self._run_safe, session_id, user_id, tenant_id, messages, session_ts
        )

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """Drain in-flight extractions and shut down the thread pool."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("FactExtractor shut down cleanly")

    def _run_safe(self, session_id, user_id, tenant_id, messages, session_ts=None):
        try:
            self.extract_and_store(session_id, user_id, tenant_id, messages, session_ts=session_ts)
        except Exception:
            logger.warning("fact_extractor background thread failed", exc_info=True)

    def _call_llm(self, conversation: str, n_facts: str = "3-7") -> list:
        try:
            extra = {"thinking": False} if hasattr(self._lm, "enable_thinking") else {}
            # Use backend's own max_new_tokens so in-process thinking models
            # (e.g. Qwen3.5) have room for <think> tokens before the JSON array.
            fact_max_tokens = getattr(self._lm, "max_new_tokens", 300)
            response = self._lm.generate(
                messages=[{"role": "user", "content": _FACT_PROMPT.format(
                    conversation=conversation, n_facts=n_facts)}],
                max_new_tokens=fact_max_tokens,
                temperature=0.1,
                **extra,
            )
            text = response.strip()
            start = text.find("[")
            if start == -1:
                record_extraction_error()
                return []
            # raw_decode stops at the end of the first valid JSON value,
            # ignoring any trailing text the model appended after the array.
            try:
                data, _ = json.JSONDecoder().raw_decode(text, start)
                return data
            except json.JSONDecodeError:
                # Truncated JSON — recover all complete objects before the break
                return _recover_partial_json(text[start:])
        except Exception:
            record_extraction_error()
            logger.warning("fact_extractor: LLM call failed", exc_info=True)
            return []

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts in one model pass when the embedder supports it.

        Falls back to sequential _embed() if embed_batch is unavailable or fails.
        """
        if not texts:
            return []
        if hasattr(self._embedder, "embed_batch"):
            try:
                return [np.array(v, dtype=np.float32) for v in self._embedder.embed_batch(texts)]
            except Exception:
                logger.warning("fact_extractor: embed_batch failed, falling back to single embeds", exc_info=True)
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        try:
            emb = self._embedder.embed(text)
            if emb is None:
                return np.zeros(1024, dtype=np.float32)
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 1e-8 else arr
        except Exception:
            logger.warning("fact_extractor: embed failed", exc_info=True)
            return np.zeros(1024, dtype=np.float32)


__all__ = ["FactExtractor"]
