from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from organism.core.stores import UnifiedStore
from organism.core.memory.rag.fact_retriever import (
    FactRetriever,
    _is_aggregation_query,
    _extract_keywords,
)


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "agg_test.db")


def _emb(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── detector tests ────────────────────────────────────────────────────────────

def test_aggregation_detected_how_many():
    assert _is_aggregation_query("how many camping trips did I take?")

def test_aggregation_detected_total():
    assert _is_aggregation_query("what is the total amount I spent on bikes?")

def test_aggregation_detected_average():
    assert _is_aggregation_query("what is the average age of my family?")

def test_aggregation_not_detected_factual():
    assert not _is_aggregation_query("where does the user live?")

def test_aggregation_not_detected_preference():
    assert not _is_aggregation_query("what kind of food does the user prefer?")


# ── keyword extractor tests ───────────────────────────────────────────────────

def test_extract_keywords_removes_stop_words():
    kws = _extract_keywords("how many camping trips did I take this year?")
    assert "how" not in kws
    assert "many" not in kws
    assert "camping" in kws
    assert "trips" in kws

def test_extract_keywords_min_length():
    kws = _extract_keywords("how many days did I go to gym?")
    assert "gym" in kws
    assert all(len(w) >= 3 for w in kws)


# ── exhaustive retrieval returns more than k ──────────────────────────────────

def test_exhaustive_returns_all_matching_facts(store):
    """Aggregation path returns > k=8 when more matching facts exist."""
    for i in range(15):
        store.facts.add("t1", "u1", f"User went on camping trip number {i}",
                        embedding=_emb(i))

    retriever = FactRetriever(store.facts)
    results = retriever.retrieve(
        "how many camping trips did I take?",
        _emb(99), "u1", "t1", k=8,
    )
    assert len(results) > 8


def test_normal_path_respects_k(store):
    """Normal (non-aggregation) path still returns at most k facts."""
    for i in range(20):
        store.facts.add("t1", "u1", f"User fact {i}", embedding=_emb(i))

    retriever = FactRetriever(store.facts)
    results = retriever.retrieve(
        "where does the user live?",
        _emb(99), "u1", "t1", k=8,
    )
    assert len(results) <= 8


def test_exhaustive_deduplicates(store):
    """Same fact matched by multiple keywords appears only once."""
    store.facts.add("t1", "u1", "User attended yoga class and pilates session",
                    embedding=_emb(0))

    retriever = FactRetriever(store.facts)
    results = retriever.retrieve(
        "how many yoga and pilates sessions did I attend?",
        _emb(99), "u1", "t1", k=8,
    )
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), "Duplicate fact ids returned"


def test_exhaustive_user_isolated(store):
    """Aggregation results respect user isolation."""
    for i in range(5):
        store.facts.add("t1", "alice", f"Alice went camping trip {i}", embedding=_emb(i))
    for i in range(5):
        store.facts.add("t1", "bob", f"Bob went camping trip {i}", embedding=_emb(i+10))

    retriever = FactRetriever(store.facts)
    results = retriever.retrieve(
        "how many camping trips did I take?",
        _emb(99), "alice", "t1", k=8,
    )
    assert all("alice" in r["content"].lower() for r in results)
