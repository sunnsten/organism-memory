from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_store import BaseStore
from .per_user_hnsw import PerUserHNSW, VECTORLITE_AVAILABLE
from organism.shared.analytics.memory_metrics import record_fact_invalidated

logger = logging.getLogger(__name__)


# Keyword sets used by invalidate_by_topic to match facts by semantic topic
# rather than literal topic-word presence in content.
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    # Only singular, clearly one-to-one facts — categories where a user has exactly ONE
    # current value and updating it means the old value is no longer true.
    # DO NOT include broad categories like "preference", "plan", "health" — a user can
    # have many simultaneous preferences/plans and invalidating all of them at once
    # destroys unrelated facts (e.g. both "likes Python" and "prefers dark mode").
    "location":   ["lives in", "moved to", "located in", "based in", "relocated", "residing"],
    "profession": ["works as", "employed as", "job is", "career is"],
    "job":        ["works as", "employed as", "job is"],
    "name":       ["name is", "called", "goes by"],
    "language":   ["native language", "first language", "speaks only"],
    "age":        ["years old", "born in", "age is"],
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


class FactStore:
    """Store for the facts table (Layer 2 of the fact-memory architecture)."""

    def __init__(self, base: BaseStore):
        self._base = base
        self._hnsw = PerUserHNSW(base, dim=1024)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: float = 0.5,
        source_session_id: Optional[str] = None,
        source_message_ids: Optional[List[int]] = None,
        embedding: Optional[np.ndarray] = None,
        event_time: Optional[int] = None,
        event_date_raw: Optional[str] = None,
    ) -> int:
        """Insert fact; idempotent — returns existing id if content already exists for user."""
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        msg_ids_json = json.dumps(source_message_ids) if source_message_ids else None

        # Atomic upsert: INSERT OR IGNORE avoids TOCTOU race between concurrent threads
        self._base.execute(
            """INSERT OR IGNORE INTO facts
               (tenant_id, user_id, content, category, importance,
                source_session_id, source_message_ids, embedding, event_time, event_date_raw)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (tenant_id, user_id, content, category, importance,
             source_session_id, msg_ids_json, emb_blob, event_time, event_date_raw),
            commit=True,
        )
        # If INSERT fired, last_insert_rowid() gives the new id (non-zero).
        # If INSERT was ignored (duplicate), last_insert_rowid() returns 0 for apsw,
        # so we fall back to a SELECT to retrieve the existing row's id.
        new_id = self._base.last_insert_rowid()
        if new_id:
            if embedding is not None:
                self._hnsw.insert(new_id, embedding, tenant_id, user_id)
            return new_id
        existing = self._base.execute(
            "SELECT id FROM facts WHERE tenant_id=? AND user_id=? AND content=?",
            (tenant_id, user_id, content),
        ).fetchone()
        return existing["id"] if existing else 0  # type: ignore[index]

    def confirm(self, fact_id: int) -> None:
        """Increment confirmed_count and refresh last_confirmed timestamp."""
        self._base.execute(
            """UPDATE facts
               SET confirmed_count = confirmed_count + 1,
                   last_confirmed = strftime('%s','now')
               WHERE id = ?""",
            (fact_id,),
            commit=True,
        )

    def invalidate(self, fact_id: int) -> None:
        """Set valid_until = now so the fact is excluded from future retrieval."""
        row = self._base.execute(
            "SELECT tenant_id, user_id FROM facts WHERE id=?", (fact_id,)
        ).fetchone()
        self._base.execute(
            "UPDATE facts SET valid_until = strftime('%s','now') WHERE id = ?",
            (fact_id,),
            commit=True,
        )
        record_fact_invalidated()
        if row:
            self._hnsw.delete(fact_id, row["tenant_id"], row["user_id"])

    def invalidate_by_topic(self, topic: str, tenant_id: str, user_id: str) -> int:
        """Invalidate all active facts that match a semantic topic. Returns count.

        Uses _TOPIC_KEYWORDS mapping so 'location' matches 'User lives in X' even
        though the word 'location' doesn't appear literally in the content.
        Falls back to literal topic-word match for unknown topics.
        """
        cur = self._base.execute(
            """SELECT id, content FROM facts
               WHERE tenant_id=? AND user_id=?
                 AND (valid_until IS NULL OR valid_until > strftime('%s','now'))""",
            (tenant_id, user_id),
        )
        rows: list[Any] = cur.fetchall()
        keywords = _TOPIC_KEYWORDS.get(topic.lower(), [topic.lower()])
        count = 0
        for row in rows:
            content_lower = row["content"].lower()
            if any(kw in content_lower for kw in keywords):
                self.invalidate(row["id"])
                count += 1
        return count

    def supersede(self, old_id: int, new_id: int) -> None:
        """Tombstone old_id and link it to its replacement new_id via superseded_by_id."""
        row = self._base.execute(
            "SELECT tenant_id, user_id FROM facts WHERE id=?", (old_id,)
        ).fetchone()
        self._base.execute(
            """UPDATE facts
               SET valid_until = strftime('%s','now'),
                   superseded_by_id = ?
               WHERE id = ?""",
            (new_id, old_id),
            commit=True,
        )
        record_fact_invalidated()
        # Remove from HNSW so ghost entries don't accumulate.
        # Historical lookup uses SQL (superseded_by_id chain), not HNSW.
        if row:
            self._hnsw.delete(old_id, row["tenant_id"], row["user_id"])

    def get_history_chain(self, fact_id: int) -> List[Dict[str, Any]]:
        """Return full chain [oldest, ..., current] by tracing superseded_by_id.

        Walks backwards to find the root (no predecessor), then forwards to
        build the ordered list. Safe against cycles via visited sets.
        """
        # Walk backwards to root
        root_id = fact_id
        visited: set = {fact_id}
        while True:
            prev = self._base.execute(
                "SELECT id FROM facts WHERE superseded_by_id = ?", (root_id,)
            ).fetchone()
            if not prev or prev["id"] in visited:
                break
            root_id = prev["id"]
            visited.add(root_id)

        # Walk forwards from root
        chain: List[Dict[str, Any]] = []
        cur_id: Optional[int] = root_id
        visited2: set = set()
        while cur_id and cur_id not in visited2:
            visited2.add(cur_id)
            row = self._base.execute(
                "SELECT * FROM facts WHERE id = ?", (cur_id,)
            ).fetchone()
            if not row:
                break
            chain.append(dict(row))
            cur_id = row["superseded_by_id"]
        return chain

    def find_similar_scored(
        self,
        embedding: np.ndarray,
        tenant_id: str,
        user_id: str,
        min_score: float = 0.70,
        exclude_ids: Optional[set] = None,
    ) -> Optional[tuple]:
        """Return (id, cosine_score) of the most similar fact at or above min_score, or None.

        exclude_ids: set of fact ids to skip (used to avoid within-batch self-matches).
        """
        cur = self._base.execute(
            """SELECT id, embedding FROM facts
               WHERE tenant_id=? AND user_id=? AND embedding IS NOT NULL
                 AND (valid_until IS NULL OR valid_until > strftime('%s','now'))""",
            (tenant_id, user_id),
        )
        rows: list[Any] = cur.fetchall()
        best_id, best_sim = None, -1.0
        for row in rows:
            if exclude_ids and row["id"] in exclude_ids:
                continue
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = _cosine(embedding, stored)
            if sim > best_sim:
                best_sim, best_id = sim, row["id"]
        if best_sim >= min_score:
            return (best_id, best_sim)
        return None

    def add_or_supersede(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        old_embedding: np.ndarray,
        new_embedding: np.ndarray,
        category: str = "fact",
        importance: float = 0.5,
        source_session_id: Optional[str] = None,
        source_message_ids: Optional[List[int]] = None,
        dedup_threshold: float = 0.90,
        supersede_threshold: float = 0.82,
        exclude_ids: Optional[set] = None,
        event_time: Optional[int] = None,
        event_date_raw: Optional[str] = None,
    ) -> tuple:
        """
        Add a fact with smart deduplication and invalidation.

        cosine > dedup_threshold                          → confirm existing (true duplicate), return existing id
        supersede_threshold < cosine ≤ dedup_threshold   → invalidate old + add new
        cosine ≤ supersede_threshold                     → unrelated topic, just add new

        old_embedding: used for similarity search against existing facts.
        new_embedding: stored on the new fact row.
        exclude_ids: fact ids added in the same batch — skipped during similarity search
                     to avoid treating sibling facts as duplicates of one another.
        Returns (fact_id, is_new) where is_new=False means an existing fact was confirmed.
        """
        hit = self.find_similar_scored(
            old_embedding, tenant_id, user_id,
            min_score=supersede_threshold,
            exclude_ids=exclude_ids,
        )
        if hit is not None:
            existing_id, score = hit
            if score > dedup_threshold:
                self.confirm(existing_id)
                return (existing_id, False)  # confirmed, not new
            # Same topic, updated value — insert new first (FK must exist), then supersede
            new_id = self.add(
                tenant_id=tenant_id,
                user_id=user_id,
                content=content,
                category=category,
                importance=importance,
                source_session_id=source_session_id,
                source_message_ids=source_message_ids,
                embedding=new_embedding,
                event_time=event_time,
                event_date_raw=event_date_raw,
            )
            self.supersede(existing_id, new_id)
            return (new_id, True)

        new_id = self.add(
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
            source_session_id=source_session_id,
            source_message_ids=source_message_ids,
            embedding=new_embedding,
            event_time=event_time,
            event_date_raw=event_date_raw,
        )
        return (new_id, True)  # new row inserted

    # ------------------------------------------------------------------
    # Similarity search (deduplication at write time)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Read (for retrieval)
    # ------------------------------------------------------------------

    def search_fts(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text search using the facts_fts FTS5 virtual table."""
        words = query.strip().split()
        if not words:
            return []
        fts_query = " ".join(f'"{w}"' for w in words)
        try:
            cur = self._base.execute(
                """SELECT f.id, f.content, f.category, f.importance,
                          f.confirmed_count, f.embedding, f.created_at,
                          fts.rank AS fts_rank
                   FROM facts_fts fts
                   JOIN facts f ON f.id = fts.rowid
                   WHERE facts_fts MATCH ?
                     AND f.tenant_id = ? AND f.user_id = ?
                     AND (f.valid_until IS NULL OR f.valid_until > strftime('%s','now'))
                   ORDER BY fts.rank
                   LIMIT ?""",
                (fts_query, tenant_id, user_id, limit),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]  # type: ignore[return-value]
        except Exception:
            logger.warning("fact_store: FTS search failed", exc_info=True)
            return []

    def search_vector(
        self,
        embedding: np.ndarray,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Vector search: HNSW O(log n) when vectorlite is available, Python cosine fallback."""
        if VECTORLITE_AVAILABLE:
            hnsw_results = self._hnsw.search(embedding, tenant_id, user_id, limit=limit * 2)
            if hnsw_results:
                return self._hydrate_hnsw(hnsw_results, tenant_id, user_id, limit)
        return self._search_vector_python(embedding, tenant_id, user_id, limit)

    def _hydrate_hnsw(
        self,
        hnsw_results: List,
        tenant_id: str,
        user_id: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fetch full fact rows for HNSW results, filtering tombstoned facts."""
        out = []
        for rowid, distance in hnsw_results:
            sim = max(-1.0, min(1.0, 1.0 - (distance * distance / 2.0)))
            row = self._base.execute(
                """SELECT id, content, category, importance, confirmed_count,
                          embedding, created_at FROM facts
                   WHERE id=? AND tenant_id=? AND user_id=?
                     AND (valid_until IS NULL OR valid_until > strftime('%s','now'))""",
                (rowid, tenant_id, user_id),
            ).fetchone()
            if row:
                out.append({**dict(row), "vector_score": sim})
            if len(out) >= limit:
                break
        return out

    def _search_vector_python(
        self,
        embedding: np.ndarray,
        tenant_id: str,
        user_id: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Python O(n) cosine scan fallback."""
        cur = self._base.execute(
            """SELECT id, content, category, importance, confirmed_count,
                      embedding, created_at
               FROM facts
               WHERE tenant_id=? AND user_id=?
                 AND embedding IS NOT NULL
                 AND (valid_until IS NULL OR valid_until > strftime('%s','now'))""",
            (tenant_id, user_id),
        )
        rows: list[Any] = cur.fetchall()
        scored = []
        for row in rows:
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = _cosine(embedding, stored)
            scored.append({**dict(row), "vector_score": sim})
        scored.sort(key=lambda x: x["vector_score"], reverse=True)
        return scored[:limit]

    def get(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single fact by id."""
        cur = self._base.execute("SELECT * FROM facts WHERE id=?", (fact_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def upsert_profile(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_fact_id: Optional[int] = None,
    ) -> None:
        """Insert or update a user-profile entry."""
        self._base.execute(
            """INSERT INTO user_profile(tenant_id, user_id, key, value, confidence,
                                        source_fact_ids, updated_at)
               VALUES (?,?,?,?,?,?,strftime('%s','now'))
               ON CONFLICT(tenant_id, user_id, key) DO UPDATE SET
                   value = excluded.value,
                   confidence = excluded.confidence,
                   source_fact_ids = excluded.source_fact_ids,
                   updated_at = strftime('%s','now')""",
            (
                tenant_id, user_id, key, value, confidence,
                json.dumps([source_fact_id]) if source_fact_id is not None else None,
            ),
            commit=True,
        )

    def get_profile(
        self,
        tenant_id: str,
        user_id: str,
        min_confidence: float = 0.6,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Return profile rows for a user filtered by minimum confidence."""
        cur = self._base.execute(
            """SELECT key, value, confidence FROM user_profile
               WHERE tenant_id=? AND user_id=? AND confidence >= ?
               ORDER BY confidence DESC LIMIT ?""",
            (tenant_id, user_id, min_confidence, limit),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def count(self, tenant_id: str, user_id: Optional[str] = None) -> int:
        """Count facts for a tenant (optionally filtered by user)."""
        if user_id:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM facts WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            )
        else:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM facts WHERE tenant_id=?",
                (tenant_id,),
            )
        return cur.fetchone()["cnt"]  # type: ignore[index]

    def delete_for_user(self, tenant_id: str, user_id: str) -> int:
        """Delete all facts and profile entries for a user. Returns count of deleted facts."""
        rows: list[Any] = self._base.execute(
            "SELECT id FROM facts WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id),
        ).fetchall()
        if rows:
            self._base.execute(
                "DELETE FROM facts WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
                commit=False,
            )
        self._base.execute(
            "DELETE FROM user_profile WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id),
            commit=True,
        )
        return len(rows)


__all__ = ["FactStore"]
