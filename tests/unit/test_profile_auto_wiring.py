from __future__ import annotations
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from organism.core.stores import UnifiedStore
from organism.core.memory.service.fact_extractor import FactExtractor
from organism.core.memory.service.profile_updater import ProfileUpdater


@pytest.fixture
def store(tmp_path: Path) -> UnifiedStore:
    return UnifiedStore(tmp_path / "pw_test.db")


def test_profile_updater_called_after_new_facts(store):
    """When extract_and_store() inserts ≥1 new fact, ProfileUpdater.update_user is called."""
    lm = Mock()
    lm.generate.return_value = '[{"content": "User name is Alice Smith", "category": "fact"}]'

    embedder = Mock()
    embedder.embed.return_value = np.random.rand(8).astype(np.float32)

    profile_updater = Mock(spec=ProfileUpdater)

    extractor = FactExtractor(
        lm_backend=lm,
        embedder=embedder,
        fact_store=store.facts,
        profile_updater=profile_updater,
    )

    extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "My name is Alice Smith"}],
    )

    profile_updater.update_user.assert_called_once_with(user_id="u1", tenant_id="t1")


def test_profile_updater_not_called_when_no_new_facts(store):
    """When no new facts inserted (empty extraction), ProfileUpdater.update_user is NOT called."""
    lm = Mock()
    lm.generate.return_value = '[]'  # LLM returns empty list

    embedder = Mock()
    embedder.embed.return_value = np.random.rand(8).astype(np.float32)

    profile_updater = Mock(spec=ProfileUpdater)

    extractor = FactExtractor(
        lm_backend=lm,
        embedder=embedder,
        fact_store=store.facts,
        profile_updater=profile_updater,
    )

    extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "Hello"}],
    )

    profile_updater.update_user.assert_not_called()


def test_profile_updater_not_called_when_only_confirms(store):
    """When all facts are confirms (count=0), ProfileUpdater is NOT called."""
    import numpy as np

    # Pre-insert a fact with the same content
    emb = np.ones(8, dtype=np.float32) / (8 ** 0.5)
    store.facts.add("t1", "u1", "User name is Alice Smith", embedding=emb)

    lm = Mock()
    lm.generate.return_value = '[{"content": "User name is Alice Smith", "category": "fact"}]'

    embedder = Mock()
    embedder.embed.return_value = emb  # Same embedding → confirm, not new

    profile_updater = Mock(spec=ProfileUpdater)

    extractor = FactExtractor(
        lm_backend=lm,
        embedder=embedder,
        fact_store=store.facts,
        profile_updater=profile_updater,
    )

    extractor.extract_and_store(
        session_id="s2", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "My name is Alice Smith"}],
    )

    profile_updater.update_user.assert_not_called()


def test_extractor_works_without_profile_updater(store):
    """FactExtractor must work normally when profile_updater=None (default)."""
    lm = Mock()
    lm.generate.return_value = '[{"content": "User enjoys hiking outdoors", "category": "preference"}]'

    embedder = Mock()
    embedder.embed.return_value = np.random.rand(8).astype(np.float32)

    extractor = FactExtractor(
        lm_backend=lm,
        embedder=embedder,
        fact_store=store.facts,
        # no profile_updater — should default to None
    )

    count = extractor.extract_and_store(
        session_id="s1", user_id="u1", tenant_id="t1",
        messages=[{"role": "user", "content": "I enjoy hiking outdoors"}],
    )
    assert count == 1
