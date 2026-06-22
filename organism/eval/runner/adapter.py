from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


class EvalAdapter:
    """Single access point for optional organism capabilities used in eval."""

    @staticmethod
    def get_last_trace(organism: Any) -> Any | None:
        ms = getattr(organism, "memory_service", None)
        if ms is None:
            return None
        return getattr(ms, "_last_trace", None)

    @staticmethod
    def get_last_write_skipped_stages(organism: Any) -> list[dict[str, Any]]:
        ms = getattr(organism, "memory_service", None)
        if ms is None:
            return []
        return getattr(ms, "_last_write_skipped_stages", [])

    @staticmethod
    def get_last_encoded_debug(organism: Any) -> dict[str, Any] | None:
        ms = getattr(organism, "memory_service", None)
        if ms is None:
            return None
        return getattr(ms, "_last_encoded_debug", None)

    @staticmethod
    def get_last_write_report(organism: Any) -> dict[str, Any] | None:
        ms = getattr(organism, "memory_service", None)
        if ms is None:
            return None
        return getattr(ms, "_last_write_report", None)

    @staticmethod
    def get_last_attention_trace(organism: Any) -> dict[str, Any] | None:
        ms = getattr(organism, "memory_service", None)
        if ms is None:
            return None
        return getattr(ms, "_last_attention_trace", None)

    @staticmethod
    def get_mcp_handlers(organism: Any) -> dict[str, Any]:
        from organism.mcp_server import _make_handlers
        return _make_handlers(organism)
