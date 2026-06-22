from __future__ import annotations
import json
from unittest.mock import Mock
from pathlib import Path
import numpy as np
import pytest
from organism.core.stores import UnifiedStore
from organism.core.memory.service.fact_extractor import FactExtractor


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "5w_test.db")


def _extractor(store: UnifiedStore, lm_response: str) -> FactExtractor:
    lm = Mock()
    lm.generate.return_value = lm_response
    embedder = Mock()
    emb = np.ones(8, dtype=np.float32) / (8 ** 0.5)
    embedder.embed.return_value = emb
    embedder.embed_batch.return_value = [emb] * 10  # generous upper bound
    return FactExtractor(lm_backend=lm, embedder=embedder, fact_store=store.facts)


def test_supersedes_topic_invalidates_old_fact(store):
    """When new fact has supersedes_topic='location', old location facts are invalidated."""
    old_emb = np.zeros(8, dtype=np.float32)
    old_emb[0] = 1.0
    old_id = store.facts.add("t1", "u1", "User lives in New York",
                             category="fact", embedding=old_emb)

    extractor = _extractor(store, json.dumps([{
        "content": "User moved to Seattle",
        "category": "fact",
        "when": "April 2026",
        "supersedes_topic": "location",
    }]))
    extractor.extract_and_store("s1", "u1", "t1",
                                [{"role": "user", "content": "I moved to Seattle"}])

    old = store.facts.get(old_id)
    assert old["valid_until"] is not None, "Old location fact should be invalidated"

    active = store.facts.search_fts("user", "t1", "u1")
    contents = [r["content"] for r in active]
    assert any("Seattle" in c for c in contents)
    assert not any("New York" in c for c in contents), "Invalidated fact must not appear"


def test_when_field_prepended_to_content(store):
    """When 'when' field present, content gets temporal prefix '[when] ...'."""
    extractor = _extractor(store, json.dumps([{
        "content": "User is studying machine learning",
        "category": "fact",
        "when": "2025",
    }]))
    extractor.extract_and_store("s1", "u1", "t1",
                                [{"role": "user", "content": "In 2025 I started ML"}])

    cur = store.facts._base.execute(
        "SELECT content FROM facts WHERE user_id='u1' AND tenant_id='t1'"
    )
    rows = cur.fetchall()
    assert rows, "Fact should be inserted"
    assert any("2025" in r["content"] for r in rows), "Temporal marker must be in stored content"


def test_no_when_no_prefix(store):
    """Without 'when' field, content is stored as-is."""
    extractor = _extractor(store, json.dumps([{
        "content": "User prefers dark mode",
        "category": "preference",
    }]))
    extractor.extract_and_store("s1", "u1", "t1",
                                [{"role": "user", "content": "I prefer dark mode"}])

    cur = store.facts._base.execute(
        "SELECT content FROM facts WHERE user_id='u1' AND tenant_id='t1'"
    )
    row = cur.fetchone()
    assert row is not None
    assert not row["content"].startswith("["), "No temporal prefix without 'when'"


def test_supersedes_topic_without_when(store):
    """supersedes_topic works independently of when field."""
    old_emb = np.zeros(8, dtype=np.float32)
    old_emb[1] = 1.0
    old_id = store.facts.add("t1", "u1", "User works as a teacher",
                             category="fact", embedding=old_emb)

    extractor = _extractor(store, json.dumps([{
        "content": "User changed careers and now works as a software engineer",
        "category": "fact",
        "supersedes_topic": "profession",
    }]))
    extractor.extract_and_store("s1", "u1", "t1",
                                [{"role": "user", "content": "I became a software engineer"}])

    old = store.facts.get(old_id)
    assert old["valid_until"] is not None, "Old profession fact should be invalidated"
