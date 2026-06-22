from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from organism.shared.analytics.memory_metrics import record_retrieval

if TYPE_CHECKING:
    from organism.core.stores.fact_store import FactStore
    from organism.core.memory.rag.reranker import Reranker

logger = logging.getLogger(__name__)

_MMR_LAMBDA = 0.7        # relevance weight vs redundancy penalty
_CANDIDATES_MULT = 3     # fetch k*3 candidates before MMR

_TEMPORAL_KEYWORDS = frozenset([
    "before", "previously", "used to", "when did", "when was", "when were",
    "prior to", "ago", "last time", "first time", "originally", "earlier",
    "at the time", "back then", "moved from", "came from", "used to live",
    "used to work", "used to be", "what was", "what were",
])

_AGGREGATION_KEYWORDS = frozenset([
    "how many", "how much", "total", "in total", "altogether",
    "all of", "every", "each time", "average", "on average",
    "count", "tally", "sum", "cumulative", "combined",
    "across all", "over all", "add up", "overall",
])

# Stop-words to strip when extracting keywords for exhaustive FTS
_STOP_WORDS = frozenset([
    "how", "many", "much", "did", "do", "have", "i", "my", "the", "a", "an",
    "in", "on", "at", "to", "for", "of", "and", "or", "is", "was", "were",
    "what", "which", "when", "where", "who", "that", "this", "with",
    "total", "all", "every", "each", "time", "times",
])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def _rrf_merge(
    fts_rows: List[Dict],
    vec_rows: List[Dict],
    rrf_k: int = 60,
) -> List[Dict]:
    """Reciprocal Rank Fusion — merge FTS and vector ranked lists into one."""
    scores: Dict[int, float] = {}
    merged: Dict[int, Dict] = {}

    for rank, row in enumerate(fts_rows):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
        merged[rid] = dict(row)

    for rank, row in enumerate(vec_rows):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
        if rid not in merged:
            merged[rid] = dict(row)

    for rid, row in merged.items():
        row["rrf_score"] = scores[rid]

    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


def mmr_select(
    query_embedding: np.ndarray,
    candidates: List[Dict[str, Any]],
    k: int,
    lam: float = _MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection. O(k × |candidates|).
    Selects k items maximising relevance to query while minimising
    redundancy with already-selected items.
    """
    if len(candidates) <= k:
        return candidates

    # Pre-compute embeddings; skip candidates without embedding blob
    embeddings: List[np.ndarray | None] = []
    for c in candidates:
        raw = c.get("embedding")
        if raw and isinstance(raw, (bytes, bytearray)):
            embeddings.append(np.frombuffer(raw, dtype=np.float32))
        else:
            embeddings.append(None)

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(k):
        if not remaining:
            break
        best_idx, best_score = None, -np.inf

        for i in remaining:
            emb_i = embeddings[i]
            if emb_i is None:
                relevance = 0.0
            else:
                relevance = _cosine(query_embedding, emb_i)

            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    (_cosine(emb_i, embeddings[j]) if emb_i is not None and embeddings[j] is not None else 0.0)  # type: ignore[arg-type]
                    for j in selected
                )

            score = lam * relevance - (1 - lam) * redundancy
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def _is_aggregation_query(query: str) -> bool:
    """Return True if the query asks for a count, sum, or aggregate over multiple facts."""
    q = query.lower()
    return any(kw in q for kw in _AGGREGATION_KEYWORDS)


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful non-stop content words from a query for FTS matching."""
    words = re.findall(r"[a-zA-Z']+", query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 3]


def _is_temporal_query(query: str) -> bool:
    """Return True if the query asks about past state of a fact."""
    q = query.lower()
    return any(kw in q for kw in _TEMPORAL_KEYWORDS)


def _inject_predecessors(result: List[Dict], store: "FactStore") -> List[Dict]:
    """Append immediate predecessors for temporal context. Deduplicates by id."""
    seen_ids = {f["id"] for f in result}
    enriched = list(result)
    for fact in result:
        pred_row = store._base.execute(
            "SELECT * FROM facts WHERE superseded_by_id = ?", (fact["id"],)
        ).fetchone()
        if pred_row and pred_row["id"] not in seen_ids:
            d = dict(pred_row)
            d["_historical"] = True
            enriched.append(d)
            seen_ids.add(pred_row["id"])
    return enriched


class FactRetriever:
    """
    Tier 2 (facts) retrieval — replaces Research-tier MemoryItemStore for online path.
    Pipeline: FTS BM25 + vector cosine -> RRF merge -> importance blend -> MMR -> (optional) cross-encoder.
    """

    def __init__(
        self,
        fact_store: "FactStore",
        candidates_mult: int = _CANDIDATES_MULT,
        reranker: "Reranker | None" = None,
    ):
        self._store = fact_store
        self._candidates_mult = candidates_mult
        self._reranker = reranker

    def _retrieve_exhaustive(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """FTS scan across all active facts — used for aggregation queries.

        Returns all facts that match any content keyword from the query, up to
        `limit`. Deduplicates by id and preserves insertion order (most relevant
        FTS keyword first).
        """
        keywords = _extract_keywords(query)
        seen: Dict[int, Dict] = {}
        for kw in keywords[:4]:
            for row in self._store.search_fts(kw, tenant_id, user_id, limit=limit):
                if row["id"] not in seen:
                    seen[row["id"]] = row
            if len(seen) >= limit:
                break
        return list(seen.values())[:limit]

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_id: str,
        tenant_id: str,
        k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return up to k diverse facts ranked by relevance + importance + confirmation.

        For aggregation queries (how many / total / etc.) switches to exhaustive
        FTS scan so counting questions see all relevant facts, not just top-k.
        """
        t0 = time.perf_counter()

        if _is_aggregation_query(query):
            result = self._retrieve_exhaustive(query, user_id, tenant_id)
            record_retrieval(len(result), time.perf_counter() - t0)
            return result
        candidates_k = k * self._candidates_mult

        fts_rows = self._store.search_fts(query, tenant_id, user_id, limit=candidates_k)
        vec_rows = self._store.search_vector(query_embedding, tenant_id, user_id, limit=candidates_k)

        if not fts_rows and not vec_rows:
            record_retrieval(0, time.perf_counter() - t0)
            return []

        candidates = _rrf_merge(fts_rows, vec_rows)

        # Re-rank: blend RRF with importance and confirmation frequency
        max_rrf = max(c["rrf_score"] for c in candidates) or 1.0  # rrf_score always > 0; guard is defensive
        for c in candidates:
            c["combined_score"] = (
                0.7 * c["rrf_score"] / max_rrf
                + 0.2 * c.get("importance", 0.5)
                + 0.1 * min(1.0, c.get("confirmed_count", 1) / 5.0)
            )
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        # MMR on candidates that have embeddings; append remainder up to k
        with_emb = [c for c in candidates if c.get("embedding")]
        without_emb = [c for c in candidates if not c.get("embedding")]

        need = k
        selected = mmr_select(query_embedding, with_emb, k=min(need, len(with_emb)))
        remaining_slots = k - len(selected)
        if remaining_slots > 0:
            selected += without_emb[:remaining_slots]

        if self._reranker is not None:
            result = self._reranker.rerank(query, selected, top_k=k)
        else:
            result = selected[:k]

        # Targeted temporal injection: only for queries that ask about past state
        if _is_temporal_query(query):
            result = _inject_predecessors(result, self._store)

        record_retrieval(len(result), time.perf_counter() - t0)
        return result


def _format_fact_content(fact: Dict[str, Any]) -> str:
    """Prefix fact content with [YYYY-MM] temporal label or [HISTORICAL] marker."""
    if fact.get("event_time"):
        try:
            dt = datetime.fromtimestamp(fact["event_time"], tz=timezone.utc)
            prefix = f"[{dt.strftime('%Y-%m')}] "
        except Exception:
            prefix = ""
    elif fact.get("_historical"):
        prefix = "[HISTORICAL] "
    else:
        prefix = ""
    return f"{prefix}{fact['content']}"
