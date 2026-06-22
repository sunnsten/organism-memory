from __future__ import annotations

import os

import pytest

from tests.helpers import (
    DummyLMBackend,
    FakeLM,
    DummyTokenizer,
)

__all__ = [
    "FakeLM",
    "DummyTokenizer",
    "DummyLMBackend",
    "lm_backend_factory",
    "should_use_dummy_backend",
]

# Research helpers — present when organism/core/memory/slots/ is on disk
try:
    from tests.helpers import make_memory_core, FakeOrganism  # noqa: F401  # type: ignore[attr-defined]
    __all__ += ["make_memory_core", "FakeOrganism"]
except ImportError:
    pass


def _should_use_real_model() -> bool:
    """Return True if TEST_USE_REAL_MODEL=1 is set."""
    return os.getenv("TEST_USE_REAL_MODEL") == "1"


def should_use_dummy_backend() -> bool:
    """Return True when tests should use DummyLMBackend (the default)."""
    return not _should_use_real_model()


def _get_test_model_path() -> str | None:
    """Return the model path from TEST_MODEL_PATH, or None."""
    return os.getenv("TEST_MODEL_PATH")


def _get_test_device() -> str:
    """
    Determine the device for tests.

    Priority:
    1. TEST_DEVICE env var (explicit override)
    2. "cuda" if a GPU is available
    3. "cpu"
    """
    import torch

    explicit_device = os.getenv("TEST_DEVICE")
    if explicit_device:
        return explicit_device.lower()

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


@pytest.fixture(scope="function")
def lm_backend_factory(monkeypatch: pytest.MonkeyPatch):
    """
    Fixture that patches organism.backbone.create_lm_backend for tests.

    Modes:
    - DummyLMBackend (default) — fast, no model required
    - Real model — when TEST_USE_REAL_MODEL=1

    Environment variables:
        TEST_USE_REAL_MODEL=1       use a real model instead of DummyLMBackend
        TEST_MODEL_NAME=...         HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct)
        TEST_MODEL_PATH=/path/...   local path to a model (takes priority over TEST_MODEL_NAME)
        TEST_MODEL_TYPE=qwen25|...  backend type (default: qwen25)
        TEST_DEVICE=cuda|cpu|auto   device override (default: cuda if available, else cpu)
    """
    use_real = _should_use_real_model()
    model_path = _get_test_model_path()

    if use_real:
        from organism.backbone import create_lm_backend as real_create_lm_backend

        def create_backend(config):
            model_type = os.getenv("TEST_MODEL_TYPE", "qwen25")
            model_name = os.getenv("TEST_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

            config.base_model.type = model_type
            config.base_model.model_name = model_path or model_name

            test_device = _get_test_device()
            if test_device == "cuda":
                config.base_model.device_map = "cuda"
            elif test_device == "cpu":
                config.base_model.device_map = "cpu"

            return real_create_lm_backend(config)

        monkeypatch.setattr(
            "organism.backbone.create_lm_backend",
            create_backend,
            raising=True,
        )
    else:
        def create_dummy_backend(config):
            return DummyLMBackend(hidden_size=16, device="cpu")

        monkeypatch.setattr(
            "organism.backbone.create_lm_backend",
            create_dummy_backend,
            raising=True,
        )

    return {
        "use_real": use_real,
        "model_path": model_path,
    }
