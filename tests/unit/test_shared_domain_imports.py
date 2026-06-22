from __future__ import annotations

import pytest


@pytest.mark.unit
def test_shared_domain_common():
    """Import from organism.shared.domain.common and from shared.domain."""
    from organism.shared.domain.common import SourceType, KindType, NamespaceType
    from organism.shared.domain import SourceType as ST2, KindType as KT2

    assert SourceType is not None
    assert KindType is not None
    assert ST2 is SourceType
    assert KT2 is KindType


@pytest.mark.unit
def test_shared_domain_core_types_available():
    """Core types are accessible from organism.shared.domain."""
    from organism.shared.domain import (
        SourceType,
        KindType,
        NamespaceType,
        EventRecord,
        ContextMeta,
        SlotRetrieveResult,
        RetrieveResult,
        MemoryRecord,
        MemoryItem,
        MemoryResult,
        WorkingMemoryPack,
        RetrievalTrace,
        PromptMemoryPack,
        MemoriesPolicy,
        MergePolicy,
        PrunePolicy,
        DEFAULT_POLICY,
        get_text_preview,
        ChatMessage,
        InteractionLog,
    )
    assert SourceType is not None
    assert EventRecord is not None
    assert RetrieveResult is SlotRetrieveResult
    assert RetrievalTrace is not None
    assert get_text_preview is not None


@pytest.mark.unit
def test_old_paths_still_work():
    """Core types are accessible directly from organism.shared.domain."""
    from organism.shared.domain import SourceType, KindType
    from organism.shared.domain import RetrievalTrace

    assert SourceType is not None
    assert KindType is not None
    assert RetrievalTrace is not None
