from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_store import BaseStore

logger = logging.getLogger(__name__)

# Vectorlite support (optional for backwards compatibility)
try:
    import vectorlite_py
    VECTORLITE_AVAILABLE = True
except ImportError:
    VECTORLITE_AVAILABLE = False
    logger.warning(
        "vectorlite_py not installed. Vector search will use slow Python cosine. "
        "Install with: pip install vectorlite-py apsw"
    )


def _sanitize_fts_query(query: str) -> str:
    """
    Sanitize a user query for FTS5 MATCH.

    Wraps each word in double quotes to treat special characters
    (@ - . etc.) as literals, not FTS5 syntax.
    """
    words = query.strip().split()
    if not words:
        return '""'
    return " ".join(f'"{w}"' for w in words)


class MemoryItemStore:
    """
    Store component for curated memory items (Tier 3 / Research).

    Memory items are facts, preferences, patterns, etc. created by
    the Consolidation pipeline or explicit "remember this" commands.
    """

    def __init__(self, base: BaseStore):
        self._base = base

    def add(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        category: str = "fact",
        confidence: float = 1.0,
        importance: float = 0.5,
        source_block_id: Optional[str] = None,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        namespace: str = "personal",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a new memory item.

        Idempotent via content_hash: if the same content already exists
        for this user/namespace, the existing ID is returned.

        Returns:
            The memory item ID.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for existing item with same hash (idempotency)
        cur = self._base.execute(
            """
            SELECT id FROM memory_items
            WHERE tenant_id = ? AND user_id = ? AND namespace = ? AND content_hash = ?
            """,
            (tenant_id, user_id, namespace, content_hash),
        )
        existing = cur.fetchone()
        if existing:
            return existing["id"]

        now = int(time.time())
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        emb_dim = embedding.shape[0] if embedding is not None else None
        tags_json = json.dumps(tags) if tags else None
        meta_json = json.dumps(metadata) if metadata else None

        self._base.execute(
            """
            INSERT INTO memory_items
                (tenant_id, user_id, content, category, confidence, importance,
                 source_block_id, valid_from, valid_until, embedding, embedding_dim,
                 created_at, updated_at, content_hash, namespace, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id, user_id, content, category, confidence, importance,
                source_block_id, valid_from, valid_until, emb_blob, emb_dim,
                now, now, content_hash, namespace, tags_json, meta_json,
            ),
            commit=True,
        )
        item_id = self._base.last_insert_rowid()

        # Also insert into vectorlite HNSW index if embedding provided
        if emb_blob is not None and VECTORLITE_AVAILABLE:
            try:
                self._base.execute(
                    "INSERT INTO vec_memory_items (rowid, embedding) VALUES (?, ?)",
                    (item_id, emb_blob),
                    commit=True,
                )
            except Exception as e:
                logger.warning(f"Failed to insert into vec_memory_items: {e}")

        return item_id

    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory item by ID."""
        cur = self._base.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all(
        self,
        tenant_id: str,
        user_id: str,
        namespace: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memory items for a user with optional filters."""
        sql = "SELECT * FROM memory_items WHERE tenant_id = ? AND user_id = ?"
        params: list = [tenant_id, user_id]

        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY importance DESC, created_at DESC"

        cur = self._base.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def search_fts(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 20,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Full-text search using FTS5 (BM25 ranking).

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            query: FTS5 search query string.
            limit: Maximum results.

        Returns:
            List of (item_dict, bm25_score) tuples, best match first.
            bm25_score is negative (more negative = better match).
        """
        cur = self._base.execute(
            """
            SELECT m.*, bm25(memory_items_fts) as bm25_score
            FROM memory_items m
            JOIN memory_items_fts fts ON fts.rowid = m.id
            WHERE memory_items_fts MATCH ?
              AND m.tenant_id = ? AND m.user_id = ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (_sanitize_fts_query(query), tenant_id, user_id, limit),
        )
        return [(dict(r), r["bm25_score"]) for r in cur.fetchall()]  # type: ignore[return-value]

    def search_vector(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: np.ndarray,
        limit: int = 20,
        min_similarity: float = -1.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector similarity search using vectorlite HNSW (426x faster).

        Falls back to Python cosine similarity if vectorlite not available.

        Performance:
        - vectorlite HNSW: ~0.09ms per query (10k vectors)
        - Python cosine: ~40ms per query (10k vectors)

        Args:
            tenant_id: Tenant identifier.
            user_id: User identifier.
            query_embedding: Query vector (dim,), L2-normalized.
            limit: Maximum results.
            min_similarity: Minimum cosine similarity threshold (-1.0 to 1.0).
                          Default -1.0 means no filtering (return all results).

        Returns:
            List of (item_dict, similarity_score) tuples, best first.
            similarity_score is cosine similarity (1.0 = identical, -1.0 = opposite).
        """
        if VECTORLITE_AVAILABLE:
            return self._search_vector_hnsw(
                tenant_id, user_id, query_embedding, limit, min_similarity
            )
        else:
            return self._search_vector_python(
                tenant_id, user_id, query_embedding, limit, min_similarity
            )

    def _search_vector_hnsw(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector search using vectorlite HNSW index (fast path).

        Note: vectorlite returns L2 distance, we convert to cosine similarity.
        For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
        """
        # Normalize query vector
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        q_blob = q_norm.astype(np.float32).tobytes()

        # Over-fetch to account for filtering by min_similarity and floating-point edge cases
        k = limit * 3

        try:
            cur = self._base.execute(
                """
                SELECT m.*, v.distance
                FROM vec_memory_items v
                INNER JOIN memory_items m ON v.rowid = m.id
                WHERE knn_search(v.embedding, knn_param(?, ?))
                  AND m.tenant_id = ?
                  AND m.user_id = ?
                ORDER BY v.distance ASC
                """,
                (q_blob, k, tenant_id, user_id),
            )

            results = []
            _hnsw_rows: list[Any] = cur.fetchall()  # type: ignore[union-attr]
            for row in _hnsw_rows:
                row_dict: dict[str, Any] = dict(row)
                distance = row_dict.pop("distance", None)

                if distance is None:
                    logger.warning(f"Row missing distance column: {row_dict.get('id')}")
                    continue

                # Convert L2 distance to cosine similarity (for normalized vectors)
                # cosine_sim = 1 - (distance^2 / 2); clamp to [-1, 1] to absorb fp noise
                similarity = max(-1.0, min(1.0, 1.0 - (distance * distance / 2.0)))

                if similarity >= min_similarity:
                    results.append((row_dict, similarity))

            # Already sorted by distance (best first)
            return results[:limit]

        except Exception as e:
            logger.warning(f"HNSW search failed, falling back to Python cosine: {e}")
            return self._search_vector_python(
                tenant_id, user_id, query_embedding, limit, min_similarity
            )

    def _search_vector_python(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float = -1.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector search using Python cosine similarity (slow fallback).

        Used when vectorlite is not available or HNSW query fails.
        """
        cur = self._base.execute(
            """
            SELECT * FROM memory_items
            WHERE tenant_id = ? AND user_id = ? AND embedding IS NOT NULL
            """,
            (tenant_id, user_id),
        )
        rows: list[Any] = cur.fetchall()

        if not rows:
            return []

        # Compute cosine similarity in Python
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored: List[Tuple[Dict[str, Any], float]] = []

        for row in rows:
            row_dict = dict(row)
            emb_blob = row_dict["embedding"]
            emb_dim = row_dict["embedding_dim"]
            if emb_blob is None or emb_dim is None:
                continue

            item_vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
            if item_vec.shape[0] != q_norm.shape[0]:
                continue

            sim = float(np.dot(q_norm, item_vec))
            if sim >= min_similarity:
                scored.append((row_dict, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def find_similar(
        self,
        tenant_id: str,
        user_id: str,
        embedding: "np.ndarray",
        threshold: float = 0.92,
        limit: int = 3,
    ) -> list:
        """Return memory items whose vector similarity to embedding >= threshold.

        Returns list of dicts with keys: id, content, similarity.
        Returns empty list if vectorlite is unavailable or no matches found.
        """
        try:
            results = self.search_vector(
                tenant_id=tenant_id,
                user_id=user_id,
                query_embedding=embedding,
                limit=limit,
                min_similarity=threshold,
            )
            return [
                {"id": r[0]["id"], "content": r[0]["content"], "similarity": r[1]}
                for r in results
            ]
        except Exception:
            return []

    def update_embedding(
        self,
        item_id: int,
        embedding: np.ndarray,
    ) -> None:
        """Update the embedding for an existing memory item."""
        emb_blob = embedding.astype(np.float32).tobytes()
        self._base.execute(
            """
            UPDATE memory_items
            SET embedding = ?, embedding_dim = ?, updated_at = ?
            WHERE id = ?
            """,
            (emb_blob, embedding.shape[0], int(time.time()), item_id),
            commit=True,
        )

        # Also update vectorlite HNSW index
        if VECTORLITE_AVAILABLE:
            try:
                # Delete old embedding
                self._base.execute(
                    "DELETE FROM vec_memory_items WHERE rowid = ?",
                    (item_id,),
                    commit=False,
                )
                # Insert new embedding
                self._base.execute(
                    "INSERT INTO vec_memory_items (rowid, embedding) VALUES (?, ?)",
                    (item_id, emb_blob),
                    commit=True,
                )
            except Exception as e:
                logger.warning(f"Failed to update vec_memory_items: {e}")

    def increment_retrieval(self, item_id: int) -> None:
        """Increment the retrieval counter for a memory item."""
        self._base.execute(
            """
            UPDATE memory_items
            SET retrieval_count = retrieval_count + 1, last_retrieved_at = ?
            WHERE id = ?
            """,
            (int(time.time()), item_id),
            commit=True,
        )

    def delete(self, item_id: int) -> None:
        """Delete a memory item."""
        # Delete from vectorlite HNSW index first
        if VECTORLITE_AVAILABLE:
            try:
                self._base.execute(
                    "DELETE FROM vec_memory_items WHERE rowid = ?",
                    (item_id,),
                    commit=False,
                )
            except Exception as e:
                logger.warning(f"Failed to delete from vec_memory_items: {e}")

        # Delete from main table
        self._base.execute(
            "DELETE FROM memory_items WHERE id = ?",
            (item_id,),
            commit=True,
        )

    def count(self, tenant_id: str, user_id: Optional[str] = None) -> int:
        """Count memory items for a tenant."""
        if user_id:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM memory_items WHERE tenant_id = ? AND user_id = ?",
                (tenant_id, user_id),
            )
        else:
            cur = self._base.execute(
                "SELECT COUNT(*) as cnt FROM memory_items WHERE tenant_id = ?",
                (tenant_id,),
            )
        return cur.fetchone()["cnt"]  # type: ignore[index]

    def delete_for_user(self, tenant_id: str, user_id: str) -> int:
        """Delete all memory items (and their vectors) for a user. Returns count deleted."""
        rows: list[Any] = self._base.execute(
            "SELECT id FROM memory_items WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id),
        ).fetchall()
        for r in rows:
            try:
                self._base.execute(
                    "DELETE FROM vec_memory_items WHERE rowid=?",
                    (r["id"],),  # type: ignore[index]
                    commit=False,
                )
            except Exception:
                pass
        if rows:
            self._base.execute(
                "DELETE FROM memory_items WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
                commit=True,
            )
        return len(rows)


__all__ = ["MemoryItemStore"]
