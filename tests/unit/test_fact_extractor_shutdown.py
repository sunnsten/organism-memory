from __future__ import annotations
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest


def _make_extractor(generate_fn=None):
    # Pre-import base_store to break the circular import that occurs when
    # organism.core.memory.service.__init__ loads retrieval_service -> chunk_store
    # -> base_store -> unified_store -> chunk_store (circular).
    import organism.core.stores.base_store  # noqa: F401
    from organism.core.memory.service.fact_extractor import FactExtractor

    lm = Mock()
    if generate_fn:
        lm.generate.side_effect = generate_fn
    else:
        lm.generate.return_value = "[]"

    embedder = Mock()
    embedder.embed.return_value = np.ones(8, dtype=np.float32)

    store = Mock()
    store.find_similar_scored.return_value = None

    return FactExtractor(lm_backend=lm, embedder=embedder, fact_store=store)


def test_shutdown_waits_for_inflight_task():
    """shutdown() blocks until the in-flight extraction completes."""
    completed = threading.Event()

    def slow_generate(*args, **kwargs):
        time.sleep(0.05)
        completed.set()
        return "[]"

    extractor = _make_extractor(generate_fn=slow_generate)
    extractor.extract_and_store_later(
        "s1", "u1", "t1", [{"role": "user", "content": "hello"}]
    )

    extractor.shutdown(timeout_s=2.0)
    assert completed.is_set(), "shutdown() must wait for in-flight extraction"


def test_shutdown_prevents_new_submissions():
    """After shutdown(), extract_and_store_later() is a silent no-op."""
    extractor = _make_extractor()
    extractor.shutdown()
    # Must not raise RuntimeError("cannot schedule new futures after shutdown")
    extractor.extract_and_store_later("s1", "u1", "t1", [])


def test_double_shutdown_is_idempotent():
    """Calling shutdown() twice does not raise."""
    extractor = _make_extractor()
    extractor.shutdown()
    extractor.shutdown()  # must not raise


def test_extract_and_store_later_submits_work():
    """extract_and_store_later() actually runs extract_and_store in background."""
    done = threading.Event()

    def fast_generate(*args, **kwargs):
        done.set()
        return "[]"

    extractor = _make_extractor(generate_fn=fast_generate)
    extractor.extract_and_store_later(
        "s1", "u1", "t1", [{"role": "user", "content": "ping"}]
    )
    extractor.shutdown(timeout_s=2.0)
    assert done.is_set(), "Background extraction must have run"
