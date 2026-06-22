from __future__ import annotations
import json
import numpy as np
import pytest
import apsw
from unittest.mock import MagicMock
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema
from organism.core.memory.service.fact_extractor import FactExtractor


def _rng_emb(seed: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def fact_store():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    return FactStore(BaseStore(conn))


def make_extractor(fact_store, llm_response: str, embeddings: list | None = None):
    mock_lm = MagicMock()
    mock_lm.generate.return_value = llm_response

    mock_embedder = MagicMock()
    if embeddings is None:
        # Default: give every fact a unique embedding
        call_count = [0]
        def _embed_default(texts):
            result = [_rng_emb(call_count[0] + i) for i in range(len(texts))]
            call_count[0] += len(texts)
            return result
        embed_batch = _embed_default
    else:
        idx = [0]
        def _embed_indexed(texts):
            result = embeddings[idx[0]: idx[0] + len(texts)]
            idx[0] += len(texts)
            return result
        embed_batch = _embed_indexed

    mock_embedder.embed_batch.side_effect = embed_batch
    mock_embedder.embed.side_effect = lambda t: _rng_emb(42)
    return FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)


# ── core: event facts are never superseded ────────────────────────────────────

def test_event_facts_accumulate_across_sessions(fact_store):
    """Multiple 'event' facts from different sessions (different month prefix) all get stored.

    In real use, Fix B1 prepends [YYYY-MM] so identical event text from different
    monthly sessions becomes distinct content → all survive the UNIQUE constraint.
    """
    # Simulate 5 sessions spread across different months (as LongMemEval does)
    import time as _time
    base_ts = int(_time.mktime(_time.strptime("2024-01-15", "%Y-%m-%d")))
    month_secs = 30 * 24 * 3600  # ~1 month

    for i in range(5):
        session_ts = base_ts + i * month_secs  # 2024-01, 2024-02, ...
        facts_json = json.dumps([
            {"content": "User went on a camping trip", "category": "event"},
        ])
        extractor = make_extractor(fact_store, facts_json, [_rng_emb(i)])
        extractor.extract_and_store(
            session_id=f"s{i}", user_id="u1", tenant_id="t1",
            messages=[{"role": "user", "content": "I went camping"}],
            session_ts=session_ts,
        )

    rows = fact_store._base.execute(
        "SELECT id FROM facts WHERE user_id='u1' AND tenant_id='t1' AND valid_until IS NULL"
    ).fetchall()
    assert len(rows) == 5, f"Expected 5 event facts (one per month), got {len(rows)}"


def test_event_facts_not_deduplicated_within_session(fact_store):
    """Two 'event' facts with different content in the same session both stored."""
    emb = _rng_emb(1)
    facts_json = json.dumps([
        {"content": "User bought a mountain bike for $65", "category": "event"},
        {"content": "User bought cycling gloves for $20", "category": "event"},
    ])
    extractor = make_extractor(fact_store, facts_json, [emb, emb])
    count = extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "I bought a bike and gloves"}],
    )
    assert count == 2


def test_state_fact_still_supersedes(fact_store):
    """Non-event facts with high cosine still go through supersede logic."""
    # Two near-identical embeddings → will trigger dedup (cosine ≈ 1.0)
    emb = _rng_emb(5)
    for i in range(3):
        extractor = make_extractor(fact_store, json.dumps([
            {"content": "User lives in New York", "category": "fact"},
        ]), [emb])
        extractor.extract_and_store(
            session_id=f"s{i}", user_id="u1", tenant_id="t1",
            messages=[{"role": "user", "content": "I live in New York"}],
        )

    rows = fact_store._base.execute(
        "SELECT id FROM facts WHERE user_id='u1' AND tenant_id='t1' AND valid_until IS NULL"
    ).fetchall()
    # Should be 1 (confirmed, not 3 duplicates)
    assert len(rows) == 1


def test_event_category_label_stored(fact_store):
    """Stored event fact has category='event'."""
    extractor = make_extractor(fact_store, json.dumps([
        {"content": "User attended yoga class", "category": "event"},
    ]))
    extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "I went to yoga"}],
    )
    row = fact_store._base.execute(
        "SELECT category FROM facts WHERE user_id='u1'"
    ).fetchone()
    assert row["category"] == "event"
