from __future__ import annotations

import sys
from unittest.mock import patch


def test_domain_imports_without_transformers():
    """
    Verify that memory/domain/* can be imported without transformers.

    This matters for:
    - Linters (mypy, pyright, ruff)
    - CI/CD environments without ML dependencies
    - Fast development iteration (no waiting for transformers to load)
    """
    with patch.dict(sys.modules, {"transformers": None}):
        from organism.shared.domain import (
            SourceType,
            KindType,
            NamespaceType,
            EventRecord,
            MemoryRecord,
            RetrieveResult,
        )

        assert SourceType is not None
        assert KindType is not None
        assert NamespaceType is not None
        assert EventRecord is not None
        assert MemoryRecord is not None
        assert RetrieveResult is not None


def test_backbone_base_imports_without_transformers():
    """Verify that base types from backbone can be imported without transformers."""
    with patch.dict(sys.modules, {"transformers": None}):
        from organism.backbone import LMBackend, EncodedText, EncodeAndUpdateSSM, BackboneConfig

        assert LMBackend is not None
        assert EncodedText is not None
        assert EncodeAndUpdateSSM is not None
        assert BackboneConfig is not None


def test_backbone_backend_classes_lazy_import():
    """Verify that backend classes are available via lazy import but are not in __all__."""
    from organism.backbone import LMBackend, EncodedText
    from organism.backbone import __all__

    assert "Llama31Backend" not in __all__
    assert "Qwen3Backend" not in __all__
    assert "Qwen3VLBackend" not in __all__

    import organism.backbone as backbone_module
    assert hasattr(backbone_module, "__getattr__")
