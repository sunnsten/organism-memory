import json
import pytest
from unittest.mock import MagicMock
import numpy as np
import apsw
from organism.core.stores.base_store import BaseStore
from organism.core.stores.fact_store import FactStore
from organism.core.stores.schema import init_schema
from organism.core.memory.service.fact_extractor import FactExtractor


@pytest.fixture
def fact_store():
    conn = apsw.Connection(":memory:")
    init_schema(conn)
    base = BaseStore(conn)
    return FactStore(base)


@pytest.fixture
def fact_extractor_fixture(fact_store):
    mock_lm = MagicMock()
    mock_lm.generate.return_value = '[]'
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = []
    return FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)


def make_extractor(fact_store, llm_response: str, embedding_seed: float = 0.1):
    mock_lm = MagicMock()
    mock_lm.generate.return_value = llm_response

    mock_embedder = MagicMock()
    rng = np.random.default_rng(int(embedding_seed * 1000))
    emb = rng.random(1024).astype(np.float32)
    emb /= np.linalg.norm(emb)
    mock_embedder.embed.return_value = emb
    mock_embedder.embed_batch.side_effect = lambda texts: [emb] * len(texts)

    return FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)


def test_extracts_facts_from_session(fact_store):
    llm_resp = json.dumps([
        {"content": "User prefers Python over JavaScript", "category": "preference"},
        {"content": "User works as a backend engineer", "category": "fact"},
    ])
    extractor = make_extractor(fact_store, llm_resp)
    count = extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "I code in Python, I'm a backend engineer"}],
    )
    assert count == 2


def test_deduplicates_on_second_call(fact_store):
    """Same content → second call confirms, doesn't insert duplicate."""
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    mock_lm = MagicMock()
    mock_lm.generate.return_value = json.dumps([
        {"content": "User runs every morning", "category": "habit"},
    ])
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = emb
    mock_embedder.embed_batch.side_effect = lambda texts: [emb] * len(texts)

    extractor = FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)
    extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I run every morning"}])
    extractor.extract_and_store("s2", "u1", "t1", [{"role": "user", "content": "I run every morning"}])

    rows = fact_store._base.execute(
        "SELECT confirmed_count FROM facts WHERE user_id='u1'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["confirmed_count"] == 2


def test_skips_on_invalid_json(fact_store):
    extractor = make_extractor(fact_store, "This is not JSON at all")
    count = extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "hello"}])
    assert count == 0


def test_skips_empty_messages(fact_store):
    extractor = make_extractor(fact_store, "[]")
    count = extractor.extract_and_store("s1", "u1", "t1", [])
    assert count == 0


def test_skips_assistant_only_messages(fact_store):
    """Only user messages are used — assistant messages contain no user facts."""
    extractor = make_extractor(fact_store, "[]")
    count = extractor.extract_and_store(
        "s1", "u1", "t1",
        [{"role": "assistant", "content": "How can I help you today?"}],
    )
    assert count == 0


def test_extract_and_store_later_does_not_block(fact_store):
    """Fire-and-forget must return immediately (< 1 second)."""
    import time
    slow_lm = MagicMock()
    slow_lm.generate.return_value = "[]"
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.zeros(1024, dtype=np.float32)
    mock_embedder.embed_batch.side_effect = lambda texts: [np.zeros(1024, dtype=np.float32)] * len(texts)

    extractor = FactExtractor(lm_backend=slow_lm, embedder=mock_embedder, fact_store=fact_store)
    start = time.time()
    extractor.extract_and_store_later("s1", "u1", "t1", [{"role": "user", "content": "hi"}])
    elapsed = time.time() - start
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# embed_batch acceleration
# ---------------------------------------------------------------------------

def test_embed_batch_called_once_for_multiple_facts(fact_store):
    """embed_batch must be called exactly once regardless of how many facts are extracted."""
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    mock_lm = MagicMock()
    mock_lm.generate.return_value = json.dumps([
        {"content": "User prefers Python over JavaScript", "category": "preference"},
        {"content": "User works as a backend engineer", "category": "fact"},
        {"content": "User has five years of experience coding", "category": "fact"},
    ])
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = lambda texts: [emb] * len(texts)

    extractor = FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)
    count = extractor.extract_and_store(
        "s1", "u1", "t1",
        [{"role": "user", "content": "I code Python, backend, 5 years"}],
    )

    assert count == 3
    mock_embedder.embed_batch.assert_called_once()
    mock_embedder.embed.assert_not_called()


def test_embed_batch_receives_all_fact_contents(fact_store):
    """embed_batch is called with exactly the list of valid fact strings."""
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    facts = [
        {"content": "User prefers dark mode in their editor", "category": "preference"},
        {"content": "User lives in Berlin and works remotely", "category": "fact"},
    ]
    mock_lm = MagicMock()
    mock_lm.generate.return_value = json.dumps(facts)
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = lambda texts: [emb] * len(texts)

    extractor = FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)
    extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I like dark mode, live in Berlin"}])

    called_texts = mock_embedder.embed_batch.call_args[0][0]
    assert called_texts == [f["content"] for f in facts]


def test_embed_batch_fallback_when_unavailable(fact_store):
    """If embedder has no embed_batch, falls back to single _embed() calls."""
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    mock_lm = MagicMock()
    mock_lm.generate.return_value = json.dumps([
        {"content": "User prefers Python over JavaScript", "category": "preference"},
        {"content": "User works as a backend engineer", "category": "fact"},
    ])
    # Embedder without embed_batch
    mock_embedder = MagicMock(spec=["embed"])
    mock_embedder.embed.return_value = emb

    extractor = FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)
    count = extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I code Python"}])

    assert count == 2
    assert mock_embedder.embed.call_count == 2


def test_call_llm_recovers_truncated_json(fact_extractor_fixture):
    """_call_llm returns partial facts when JSON array is truncated (last item cut off)."""
    truncated = '[{"content": "User lives in Berlin", "category": "fact"}, {"content": "User'
    extractor = fact_extractor_fixture
    extractor._lm.generate.return_value = truncated
    result = extractor._call_llm("some conversation")
    assert len(result) == 1
    assert result[0]["content"] == "User lives in Berlin"


def test_call_llm_returns_empty_on_no_json(fact_extractor_fixture):
    """_call_llm returns [] when LLM returns no JSON array."""
    extractor = fact_extractor_fixture
    extractor._lm.generate.return_value = "Sorry, I cannot extract facts."
    result = extractor._call_llm("some conversation")
    assert result == []


def test_embed_batch_fallback_on_exception(fact_store):
    """If embed_batch raises, falls back to sequential _embed() without losing facts."""
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    mock_lm = MagicMock()
    mock_lm.generate.return_value = json.dumps([
        {"content": "User prefers Python over JavaScript", "category": "preference"},
    ])
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = RuntimeError("GPU OOM")
    mock_embedder.embed.return_value = emb

    extractor = FactExtractor(lm_backend=mock_lm, embedder=mock_embedder, fact_store=fact_store)
    count = extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I code Python"}])

    assert count == 1
    mock_embedder.embed.assert_called_once()
