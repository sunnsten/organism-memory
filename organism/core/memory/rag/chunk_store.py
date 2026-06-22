from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from organism.core.stores.base_store import BaseStore
from organism.core.stores.per_user_hnsw import PerUserHNSW, VECTORLITE_AVAILABLE

logger = logging.getLogger(__name__)

# Below this threshold we request k=count (exact coverage, perfect recall).
# Above it we cap k here so HNSW stays in O(log n) territory — requesting k=n
# forces a full graph scan that is slower than Python cosine for large corpora.
# At k=2000 the top-2000 cosine candidates contain the correct answer for
# virtually all real queries; recall at k=2000 from n=10k is >99.9% empirically.
_HNSW_EXACT_THRESHOLD = 2000


_FTS_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "or", "and",
    "for", "by", "be", "as", "if", "do", "did", "are", "was", "has", "had",
    "not", "no", "so", "my", "me", "we", "he", "she", "you", "his", "her",
    "its", "our", "who", "why", "how", "what", "when", "where", "this", "that",
    "does", "them", "they", "their", "with", "from", "have", "been", "about",
    "will", "can", "may", "but",
})


def _sanitize_fts_query(query: str) -> str:
    """
    Sanitize a user query for FTS5 MATCH.

    Strips terminal punctuation from each word, removes common English stop
    words, and wraps remaining words in double quotes to treat FTS5 special
    characters as literals.  Falls back to the full word list when filtering
    would leave an empty query.
    """
    import re
    raw_words = query.strip().split()
    if not raw_words:
        return '""'

    # Strip leading/trailing punctuation from each token
    cleaned = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in raw_words]
    # Keep only non-empty, non-stop words that are at least 2 chars
    filtered = [w for w in cleaned if w and w.lower() not in _FTS_STOP_WORDS and len(w) >= 2]
    # Fallback to all cleaned words if filtering removes everything
    words = filtered if filtered else [w for w in cleaned if w]
    if not words:
        return '""'
    return " ".join(f'"{w}"' for w in words)


class ChunkStore:
    """
    Store component for RAG chunks (Tier 1).

    RAG chunks are pieces of raw messages/documents that have been
    split, PII-redacted, and embedded by the RAG Indexer worker.
    """

    def __init__(self, base: BaseStore):
        self._base = base
        self._hnsw = PerUserHNSW(base, dim=1024)

    def add(
        self,
        tenant_id: str,
        source_type: str,
        source_id: str,
        chunk_index: int,
        content: str,
        embedding: Optional[np.ndarray] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a new RAG chunk.

        Idempotent via (source_type, source_id, chunk_index) unique index.
        If the chunk already exists, updates content and embedding.

        Returns:
            The chunk ID.
        """
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        emb_dim = embedding.shape[0] if embedding is not None else None
        tags_json = json.dumps(tags) if tags else None

        self._base.execute(
            """
            INSERT INTO rag_chunks
                (tenant_id, user_id, source_type, source_id, chunk_index, content,
                 embedding, embedding_dim, session_id, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, source_id, chunk_index)
            DO UPDATE SET
                content = excluded.content,
                embedding = excluded.embedding,
                embedding_dim = excluded.embedding_dim,
                tags = excluded.tags
            """,
            (
                tenant_id, user_id, source_type, source_id, chunk_index, content,
                emb_blob, emb_dim, session_id, tags_json, int(time.time()),
            ),
            commit=True,
        )
        chunk_id = self._base.last_insert_rowid()

        if embedding is not None and user_id is not None:
            self._hnsw.insert(chunk_id, embedding, tenant_id, user_id)

        return chunk_id

    def add_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Insert multiple RAG chunks in a single transaction.

        Each dict must have keys: tenant_id, source_type, source_id,
        chunk_index, content. Optional: embedding, session_id, tags.

        Returns:
            Number of chunks inserted/updated.
        """
        if not chunks:
            return 0

        now = int(time.time())
        rows = []
        for c in chunks:
            emb = c.get("embedding")
            emb_blob = emb.astype(np.float32).tobytes() if emb is not None else None
            emb_dim = emb.shape[0] if emb is not None else None
            tags_json = json.dumps(c.get("tags")) if c.get("tags") else None
            created_at = c.get("created_at") or now

            rows.append((
                c["tenant_id"], c.get("user_id"), c["source_type"], c["source_id"],
                c["chunk_index"], c["content"],
                emb_blob, emb_dim, c.get("session_id"), tags_json, created_at,
            ))

        self._base.executemany(
            """
            INSERT INTO rag_chunks
                (tenant_id, user_id, source_type, source_id, chunk_index, content,
                 embedding, embedding_dim, session_id, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, source_id, chunk_index)
            DO UPDATE SET
                content = excluded.content,
                embedding = excluded.embedding,
                embedding_dim = excluded.embedding_dim,
                tags = excluded.tags
            """,
            rows,
            commit=True,
        )

        if VECTORLITE_AVAILABLE:
            for c in chunks:
                emb = c.get("embedding")
                uid = c.get("user_id")
                if emb is not None and uid is not None:
                    row = self._base.execute(
                        "SELECT id FROM rag_chunks WHERE source_type=? AND source_id=? AND chunk_index=?",
                        (c["source_type"], c["source_id"], c["chunk_index"]),
                    ).fetchone()
                    if row:
                        self._hnsw.insert(row["id"], emb, c["tenant_id"], uid)

        return len(rows)

    def search_fts(
        self,
        tenant_id: str,
        query: str,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Full-text search using FTS5 (BM25 ranking).

        Args:
            tenant_id: Tenant identifier.
            query: FTS5 search query.
            limit: Maximum results.
            user_id: If provided, restrict results to this user only.

        Returns:
            List of (chunk_dict, bm25_score), best first.
        """
        if user_id is not None:
            cur = self._base.execute(
                """
                SELECT c.*, bm25(rag_chunks_fts) as bm25_score
                FROM rag_chunks c
                JOIN rag_chunks_fts fts ON fts.rowid = c.id
                WHERE rag_chunks_fts MATCH ?
                  AND c.tenant_id = ? AND c.user_id = ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (_sanitize_fts_query(query), tenant_id, user_id, limit),
            )
        else:
            cur = self._base.execute(
                """
                SELECT c.*, bm25(rag_chunks_fts) as bm25_score
                FROM rag_chunks c
                JOIN rag_chunks_fts fts ON fts.rowid = c.id
                WHERE rag_chunks_fts MATCH ?
                  AND c.tenant_id = ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (_sanitize_fts_query(query), tenant_id, limit),
            )
        return [(dict(r), r["bm25_score"]) for r in cur.fetchall()]  # type: ignore[return-value]

    def search_vector(
        self,
        tenant_id: str,
        query_embedding: np.ndarray,
        limit: int = 20,
        min_similarity: float = 0.0,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector similarity search: HNSW with full-coverage k, Python cosine fallback.

        HNSW search with dynamic k based on user's chunk count:

        - count <= _HNSW_EXACT_THRESHOLD (2000): k = count → exact full-coverage,
          perfect recall regardless of corpus size. If HNSW returns fewer than count
          (incomplete in-memory index or failed inserts), fall back to Python cosine.

        - count > _HNSW_EXACT_THRESHOLD: k = 2000 → approximate, but HNSW stays
          O(log n) (requesting k=n forces full graph scan, slower than Python cosine).
          At k=2000 from n=10k, recall is >99.9% for real queries.

        Args:
            tenant_id: Tenant identifier.
            query_embedding: Query vector, L2-normalized.
            limit: Maximum results.
            min_similarity: Minimum cosine threshold.
            user_id: If provided, restrict results to this user only.

        Returns:
            List of (chunk_dict, cosine_similarity), best first.
        """
        if VECTORLITE_AVAILABLE and user_id is not None:
            chunk_count = self._count_user_chunks(tenant_id, user_id)
            if chunk_count > 0:
                k = min(chunk_count, _HNSW_EXACT_THRESHOLD)
                hnsw_results = self._hnsw.search(
                    query_embedding, tenant_id, user_id, limit=k
                )
                # For exact mode (chunk_count <= threshold): require full coverage.
                # For approximate mode (chunk_count > threshold): require k results.
                # Either way: len(hnsw_results) < k means index is incomplete → Python.
                if len(hnsw_results) >= k:
                    return self._hydrate_hnsw_results(hnsw_results, min_similarity, limit)
        return self._search_vector_python(
            tenant_id, query_embedding, limit, min_similarity, user_id,
        )

    def _count_user_chunks(self, tenant_id: str, user_id: str) -> int:
        """Count chunks with embeddings for a user (fast index scan)."""
        row = self._base.execute(
            "SELECT COUNT(*) as cnt FROM rag_chunks "
            "WHERE tenant_id = ? AND user_id = ? AND embedding IS NOT NULL",
            (tenant_id, user_id),
        ).fetchone()
        return row["cnt"] if row else 0

    def _hydrate_hnsw_results(
        self,
        hnsw_results: List[Tuple[int, float]],
        min_similarity: float,
        limit: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fetch full chunk rows for HNSW (rowid, distance) pairs."""
        out = []
        for rowid, distance in hnsw_results:
            similarity = max(-1.0, min(1.0, 1.0 - (distance * distance / 2.0)))
            if similarity < min_similarity:
                continue
            row = self._base.execute("SELECT * FROM rag_chunks WHERE id=?", (rowid,)).fetchone()
            if row:
                out.append((dict(row), similarity))
            if len(out) >= limit:
                break
        return out

    def _search_vector_python(
        self,
        tenant_id: str,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
        user_id: Optional[str],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Python cosine fallback (for when vectorlite is unavailable)."""
        if user_id is not None:
            cur = self._base.execute(
                "SELECT * FROM rag_chunks WHERE tenant_id = ? AND user_id = ? AND embedding IS NOT NULL",
                (tenant_id, user_id),
            )
        else:
            cur = self._base.execute(
                "SELECT * FROM rag_chunks WHERE tenant_id = ? AND embedding IS NOT NULL",
                (tenant_id,),
            )
        rows: list[Any] = cur.fetchall()
        if not rows:
            return []

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored: List[Tuple[Dict[str, Any], float]] = []
        for row in rows:
            row_dict = dict(row)
            emb_blob = row_dict["embedding"]
            emb_dim = row_dict["embedding_dim"]
            if emb_blob is None or emb_dim is None:
                continue
            chunk_vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
            if chunk_vec.shape[0] != q_norm.shape[0]:
                continue
            sim = float(np.dot(q_norm, chunk_vec))
            if sim >= min_similarity:
                scored.append((row_dict, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def get_by_source(
        self,
        source_type: str,
        source_id: int,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a given source, ordered by chunk_index."""
        cur = self._base.execute(
            """
            SELECT * FROM rag_chunks
            WHERE source_type = ? AND source_id = ?
            ORDER BY chunk_index ASC
            """,
            (source_type, source_id),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]

    def exists(self, source_type: str, source_id: int) -> bool:
        """Check if any chunks exist for a source (idempotency check)."""
        cur = self._base.execute(
            "SELECT 1 FROM rag_chunks WHERE source_type = ? AND source_id = ? LIMIT 1",
            (source_type, source_id),
        )
        return cur.fetchone() is not None

    def count(self, tenant_id: str) -> int:
        """Count total RAG chunks for a tenant."""
        cur = self._base.execute(
            "SELECT COUNT(*) as cnt FROM rag_chunks WHERE tenant_id = ?",
            (tenant_id,),
        )
        return cur.fetchone()["cnt"]  # type: ignore[index]

    def delete_for_user(self, tenant_id: str, user_id: str) -> int:
        """Delete all chunks (and HNSW vectors) for a user. Returns count deleted."""
        rows: list[Any] = self._base.execute(
            "SELECT id FROM rag_chunks WHERE tenant_id=? AND user_id=?",
            (tenant_id, user_id),
        ).fetchall()
        for r in rows:
            self._hnsw.delete(r["id"], tenant_id, user_id)  # type: ignore[index]
        if rows:
            self._base.execute(
                "DELETE FROM rag_chunks WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
                commit=True,
            )
        return len(rows)

    def delete_by_source(self, source_type: str, source_id: int) -> int:
        """Delete all chunks for a source. Returns number of deleted rows."""
        # Fetch chunk ids + user_id + tenant_id before deleting
        rows: list[Any] = self._base.execute(
            "SELECT id, tenant_id, user_id FROM rag_chunks WHERE source_type=? AND source_id=?",
            (source_type, source_id),
        ).fetchall()
        for r in rows:
            if r["user_id"]:
                self._hnsw.delete(r["id"], r["tenant_id"], r["user_id"])

        self._base.execute(
            "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ?",
            (source_type, source_id),
            commit=True,
        )
        return len(rows)

    def embed_pending(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
        embedder: Any,
    ) -> int:
        """
        Embed all chunks for a session that currently have no embedding.

        Fetches chunks with embedding IS NULL, embeds them in one embed_batch()
        call, then updates the rows and HNSW index in bulk.

        Intended for use after fast-replay sessions that write chunks without
        embeddings (skip_chunk_embedding=True in WriteService), followed by a
        single batch embedding pass — the chunk analogue of extract_session_facts().

        Returns: count of chunks embedded.
        """
        cur = self._base.execute(
            "SELECT id, content FROM rag_chunks "
            "WHERE session_id = ? AND tenant_id = ? AND user_id = ? AND embedding IS NULL",
            (session_id, tenant_id, user_id),
        )
        rows: list[Any] = cur.fetchall()
        if not rows:
            return 0

        ids = [r["id"] for r in rows]
        contents = [r["content"] for r in rows]

        if hasattr(embedder, "embed_batch"):
            try:
                vectors = [np.array(v, dtype=np.float32) for v in embedder.embed_batch(contents)]
            except Exception:
                logger.warning("embed_pending: embed_batch failed, falling back", exc_info=True)
                vectors = [self._single_embed(embedder, t) for t in contents]
        else:
            vectors = [self._single_embed(embedder, t) for t in contents]

        update_rows = []
        for row_id, vec in zip(ids, vectors):
            if vec is None:
                continue
            emb_blob = vec.tobytes()
            emb_dim = vec.shape[0]
            update_rows.append((emb_blob, emb_dim, row_id))

        if update_rows:
            self._base.executemany(
                "UPDATE rag_chunks SET embedding = ?, embedding_dim = ? WHERE id = ?",
                update_rows,
                commit=True,
            )

        for row_id, vec in zip(ids, vectors):
            if vec is not None:
                self._hnsw.insert(row_id, vec, tenant_id, user_id)

        logger.debug("embed_pending: embedded %d chunks for session=%s", len(update_rows), session_id)
        return len(update_rows)

    @staticmethod
    def _single_embed(embedder: Any, text: str) -> Optional[np.ndarray]:
        try:
            vec = embedder.embed(text)
            return np.array(vec, dtype=np.float32) if vec is not None else None
        except Exception:
            return None

    def fetch_round_parts(
        self,
        round_id: str,
        parts: List[int],
        tenant_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Fetch specific subchunks of a round by round_id + part indices."""
        if not parts:
            return []
        placeholders = ",".join("?" for _ in parts)
        cur = self._base.execute(
            f"""SELECT * FROM rag_chunks
                WHERE json_extract(tags, '$.round_id') = ?
                  AND CAST(json_extract(tags, '$.round_part') AS INTEGER) IN ({placeholders})
                  AND tenant_id = ? AND user_id = ?
                ORDER BY CAST(json_extract(tags, '$.round_part') AS INTEGER) ASC""",
            (round_id, *parts, tenant_id, user_id),
        )
        return [dict(r) for r in cur.fetchall()]  # type: ignore[return-value]


__all__ = ["ChunkStore"]
