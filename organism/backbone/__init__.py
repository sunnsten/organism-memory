from __future__ import annotations

import os
from typing import TYPE_CHECKING

# Base types import directly (no transformers dependency)
from .base import LMBackend, EncodedText, EncodeAndUpdateSSM
from .config import BackboneConfig

if TYPE_CHECKING:
    from organism.config import OrganismConfig
    from .llama31_backend import Llama31Backend
    from .qwen3_backend import Qwen3Backend
    from .qwen3_vl_backend import Qwen3VLBackend
    from .qwen25_backend import Qwen25Backend
    from .qwen35_backend import Qwen35Backend
    from .llama_cpp_backend import LlamaCppBackend

# __all__ contains only base types + factories (no backend classes) so that
# "from organism.backbone import *" does not trigger transformers loading.
__all__ = [
    "LMBackend",
    "EncodedText",
    "EncodeAndUpdateSSM",
    "BackboneConfig",
    "create_lm_backend",
    "create_lm_backend_from_backbone",
]


def __getattr__(name: str):
    """
    Lazy import for backend classes.

    Allows importing Llama31Backend, Qwen3Backend, etc. without loading
    transformers at module import time. Result is cached in globals() so
    repeated accesses skip the import.
    """
    if name == "Llama31Backend":
        from .llama31_backend import Llama31Backend
        globals()[name] = Llama31Backend
        return Llama31Backend
    if name == "Qwen3Backend":
        from .qwen3_backend import Qwen3Backend
        globals()[name] = Qwen3Backend
        return Qwen3Backend
    if name == "Qwen3VLBackend":
        from .qwen3_vl_backend import Qwen3VLBackend
        globals()[name] = Qwen3VLBackend
        return Qwen3VLBackend
    if name == "Qwen25Backend":
        from .qwen25_backend import Qwen25Backend
        globals()[name] = Qwen25Backend
        return Qwen25Backend
    if name == "Qwen35Backend":
        from .qwen35_backend import Qwen35Backend
        globals()[name] = Qwen35Backend
        return Qwen35Backend
    if name == "LlamaCppBackend":
        from .llama_cpp_backend import LlamaCppBackend
        globals()[name] = LlamaCppBackend
        return LlamaCppBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_lm_backend(config: "OrganismConfig") -> LMBackend:
    """Factory: returns the right LMBackend for config.base_model."""
    return create_lm_backend_from_backbone(config.base_model)


def create_lm_backend_from_backbone(base: "BackboneConfig") -> LMBackend:
    """
    Low-level factory that takes a BackboneConfig directly.
    Used for both base_model and fact_llm backends.
    """
    if base.type == "llama31":
        from .llama31_backend import Llama31Backend
        return Llama31Backend(
            model_name=base.model_name,
            device_map=base.device_map,
            dtype=base.dtype,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
        )

    if base.type == "qwen3":
        from .qwen3_backend import Qwen3Backend
        return Qwen3Backend(
            model_name=base.model_name,
            device_map=base.device_map,
            dtype=base.dtype,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            load_in_4bit=getattr(base, "load_in_4bit", False),
            load_in_8bit=getattr(base, "load_in_8bit", False),
        )

    if base.type == "qwen3_vl":
        from .qwen3_vl_backend import Qwen3VLBackend
        return Qwen3VLBackend(
            model_name=base.model_name,
            device_map=base.device_map,
            dtype=base.dtype,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
        )

    if base.type == "qwen25":
        from .qwen25_backend import Qwen25Backend
        return Qwen25Backend(
            model_name=base.model_name,
            device_map=base.device_map,
            dtype=base.dtype,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
        )

    if base.type == "qwen35":
        from .qwen35_backend import Qwen35Backend
        return Qwen35Backend(
            model_name=base.model_name,
            device_map=base.device_map,
            dtype=base.dtype,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            load_in_4bit=getattr(base, "load_in_4bit", False),
            load_in_8bit=getattr(base, "load_in_8bit", False),
        )

    if base.type == "openai":
        from .openai_backend import OpenAIBackend
        raw_key = getattr(base, "api_key", "not-needed")
        api_key = os.environ.get(raw_key.lstrip("$"), raw_key) if raw_key.startswith("$") else raw_key
        return OpenAIBackend(
            model_name=base.model_name,
            base_url=getattr(base, "base_url", "http://localhost:8080/v1"),
            api_key=api_key,
            temperature=base.temperature,
            top_p=base.top_p,
            max_new_tokens=base.max_new_tokens,
            strip_think=getattr(base, "strip_think", True),
            enable_thinking=getattr(base, "enable_thinking", False),
            thinking_budget=getattr(base, "thinking_budget", 0),
        )

    if base.type == "llama_cpp":
        from .llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend(
            model_path=base.model_path,
            n_gpu_layers=getattr(base, "n_gpu_layers", -1),
            n_ctx=getattr(base, "n_ctx", 8192),
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            strip_think=getattr(base, "strip_think", True),
        )

    raise ValueError(f"Unknown base_model.type: {base.type!r}")