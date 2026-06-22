from __future__ import annotations
from unittest.mock import patch
import pytest


def _candidates():
    return [
        {"id": 1, "content": "User lives in Seattle",  "rrf_score": 0.9},
        {"id": 2, "content": "User likes pizza",        "rrf_score": 0.8},
        {"id": 3, "content": "User works as engineer",  "rrf_score": 0.7},
    ]


def test_reranker_sorts_by_score():
    from organism.core.memory.rag.reranker import Reranker
    reranker = Reranker.__new__(Reranker)
    reranker._model = None
    reranker._available = True

    with patch.object(reranker, "_score_pairs", return_value=[0.1, 0.9, 0.5]):
        results = reranker.rerank("where does user live?", _candidates(), top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == 2   # score 0.9
    assert results[1]["id"] == 3   # score 0.5


def test_reranker_passthrough_when_unavailable():
    from organism.core.memory.rag.reranker import Reranker
    reranker = Reranker.__new__(Reranker)
    reranker._model = None
    reranker._available = False

    results = reranker.rerank("query", _candidates(), top_k=10)
    assert [r["id"] for r in results] == [1, 2, 3]


def test_reranker_respects_top_k():
    from organism.core.memory.rag.reranker import Reranker
    reranker = Reranker.__new__(Reranker)
    reranker._model = None
    reranker._available = True

    with patch.object(reranker, "_score_pairs", return_value=[0.9, 0.8, 0.7]):
        results = reranker.rerank("q", _candidates(), top_k=1)

    assert len(results) == 1
