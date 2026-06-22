from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def _require_str(args: dict, key: str) -> str:
    v = args.get(key)
    if not v or not isinstance(v, str):
        raise ValidationError(f"'{key}' must be a non-empty string")
    return v


def _clamp_int(args: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = args.get(key, default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ValidationError(f"'{key}' must be an integer")
    if not (lo <= v <= hi):
        raise ValidationError(f"'{key}' must be between {lo} and {hi}")
    return v


# ---------------------------------------------------------------------------
# Structured error response
# ---------------------------------------------------------------------------

def _error(error_type: str, message: str) -> str:
    return json.dumps({"error": {"type": error_type, "message": message}}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool handler factory
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "organism_chat",
        "description": "Send a message to Organism and get a reply with long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id":    {"type": "string", "description": "Unique user identifier"},
                "message":    {"type": "string", "description": "The user's message"},
                "session_id": {"type": "string", "description": "Optional session ID"},
            },
            "required": ["user_id", "message"],
        },
    },
    {
        "name": "memory.store_event",
        "description": (
            "Store content into memory and queue async fact extraction. "
            "Writes to messages table; FactExtractor fires automatically if configured."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id":    {"type": "string"},
                "content":    {"type": "string", "description": "Content to store"},
                "session_id": {"type": "string", "description": "Optional session ID"},
                "source":     {"type": "string", "description": "Source identifier (default: mcp)"},
                "metadata":   {"type": "object", "description": "Optional metadata (file, tags, etc.)"},
            },
            "required": ["user_id", "content"],
        },
    },
    {
        "name": "memory.query",
        "description": (
            "Retrieve relevant facts and chunks — no LLM, pure read. "
            "Returns structured list of facts and RAG chunks with id, score, source."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id":    {"type": "string"},
                "query":      {"type": "string"},
                "max_facts":  {"type": "integer", "default": 8, "description": "Max facts (1–50)"},
                "max_chunks": {"type": "integer", "default": 5, "description": "Max chunks (1–20)"},
            },
            "required": ["user_id", "query"],
        },
    },
    {
        "name": "memory.remember",
        "description": "Explicitly store a fact in long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "text":    {"type": "string", "description": "Fact to remember"},
            },
            "required": ["user_id", "text"],
        },
    },
    {
        "name": "memory.reset",
        "description": (
            "Delete ALL memory for a user (facts, messages, chunks, memory items, HNSW index). "
            "Pass confirm=true to confirm the wipe."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "confirm": {"type": "boolean", "description": "Must be true to proceed"},
            },
            "required": ["user_id", "confirm"],
        },
    },
    {
        "name": "memory.metrics",
        "description": "Return a health snapshot of the memory store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Optional — filter by user"},
            },
            "required": [],
        },
    },
]


def _make_handlers(org: Any, log_format: str = "plain") -> dict[str, Any]:
    """
    Returns dict of tool_name → callable(**kwargs) → str.
    Pure functions — testable without starting the MCP server.
    """

    def _log(tool: str, user_id: str, t0: float, status: str, extra: str = "") -> None:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_format == "json":
            logger.info(json.dumps({
                "tool": tool, "user_id": user_id,
                "duration_ms": duration_ms, "status": status,
                **({} if not extra else {"detail": extra}),
            }))
        else:
            logger.info(
                "tool=%s user_id=%s duration_ms=%d status=%s%s",
                tool, user_id, duration_ms, status,
                f" {extra}" if extra else "",
            )

    def _call(tool: str, user_id: str, fn: Any) -> str:
        """Wrapper for tools that return a dict — serialises to JSON."""
        t0 = time.perf_counter()
        try:
            result = fn()
            extra = ""
            if isinstance(result, dict):
                if "facts" in result:
                    extra = f"facts={result['total_facts']} chunks={result['total_chunks']}"
                elif "deleted" in result:
                    d = result["deleted"]
                    extra = f"deleted={sum(d.values())}"
            _log(tool, user_id, t0, "ok", extra)
            return json.dumps(result, ensure_ascii=False)
        except ValidationError as exc:
            _log(tool, user_id, t0, "error", f"type=ValidationError message={exc}")
            return _error("ValidationError", str(exc))
        except Exception as exc:
            _log(tool, user_id, t0, "error", f"type=InternalError message={exc}")
            logger.exception("Unhandled error in tool %s", tool)
            return _error("InternalError", "An internal error occurred")

    def _call_str(tool: str, user_id: str, fn: Any) -> str:
        """Wrapper for tools that return plain text (not JSON)."""
        t0 = time.perf_counter()
        try:
            result = fn()
            _log(tool, user_id, t0, "ok")
            return result
        except ValidationError as exc:
            _log(tool, user_id, t0, "error", f"type=ValidationError message={exc}")
            return _error("ValidationError", str(exc))
        except Exception as exc:
            _log(tool, user_id, t0, "error", f"type=InternalError message={exc}")
            logger.exception("Unhandled error in tool %s", tool)
            return _error("InternalError", "An internal error occurred")

    # --- organism_chat (backwards-compat) ---
    def organism_chat(*, user_id: str, message: str, session_id: str = "", **_: Any) -> str:
        _uid = user_id.strip()
        def _run() -> str:
            if not _uid:
                raise ValidationError("'user_id' must be a non-empty string")
            if not message:
                raise ValidationError("'message' must be a non-empty string")
            return org.chat(user_id=_uid, user_message=message, session_id=session_id or None).reply
        return _call_str("organism_chat", _uid, _run)

    # --- memory.store_event ---
    def memory_store_event(**kwargs: Any) -> str:
        _uid = kwargs.get("user_id") or ""
        def _run() -> dict:
            uid = _require_str(kwargs, "user_id")
            content = _require_str(kwargs, "content")
            return org.store_event(
                user_id=uid,
                content=content,
                session_id=kwargs.get("session_id") or None,
                source=kwargs.get("source") or "mcp",
                metadata=kwargs.get("metadata"),
            )
        return _call("memory.store_event", _uid, _run)

    # --- memory.query ---
    def memory_query(**kwargs: Any) -> str:
        _uid = kwargs.get("user_id") or ""
        def _run() -> dict:
            uid = _require_str(kwargs, "user_id")
            query = _require_str(kwargs, "query")
            max_facts = _clamp_int(kwargs, "max_facts", 8, 1, 50)
            max_chunks = _clamp_int(kwargs, "max_chunks", 5, 1, 20)
            return org.query_memory(
                user_id=uid,
                query=query,
                max_facts=max_facts,
                max_chunks=max_chunks,
            )
        return _call("memory.query", _uid, _run)

    # --- memory.remember ---
    def memory_remember(**kwargs: Any) -> str:
        _uid = kwargs.get("user_id") or ""
        def _run() -> dict:
            uid = _require_str(kwargs, "user_id")
            text = _require_str(kwargs, "text")
            return {"stored": True, "memory_id": org.remember(user_id=uid, text=text)}
        return _call("memory.remember", _uid, _run)

    # --- memory.reset ---
    def memory_reset(**kwargs: Any) -> str:
        _uid = kwargs.get("user_id") or ""
        def _run() -> dict:
            uid = _require_str(kwargs, "user_id")
            if kwargs.get("confirm") is not True:
                raise ValidationError("'confirm' must be true to reset user memory")
            return org.reset_user(user_id=uid)
        return _call("memory.reset", _uid, _run)

    # --- memory.metrics ---
    def memory_metrics(**kwargs: Any) -> str:
        user_id = kwargs.get("user_id") or ""
        t0 = time.perf_counter()
        try:
            from organism.shared.analytics.memory_metrics import take_snapshot
            snap = take_snapshot()
            # DB row counts from the store
            store = org._orchestrator._memory.store  # type: ignore[attr-defined]
            tenant_id = org._tenant_id  # type: ignore[attr-defined]
            uid: Any = user_id or None
            total_messages = store.messages.count(tenant_id, uid)
            total_facts = store.facts.count(tenant_id, uid)
            # ChunkStore.count has no user_id filter — returns null when user_id is set
            total_chunks: Any = store.chunks.count(tenant_id) if not uid else None
            _log("memory.metrics", user_id, t0, "ok")
            return json.dumps({
                "total_messages":           total_messages,
                "total_facts":              total_facts,
                "total_chunks":             total_chunks,
                "facts_extracted":          snap.facts_extracted,
                "facts_new":                snap.facts_new,
                "facts_confirmed":          snap.facts_confirmed,
                "facts_extraction_errors":  snap.facts_errors,
                "avg_retrieval_latency_ms": round(snap.retrieval_latency_avg_s * 1000, 1),
            }, ensure_ascii=False)
        except Exception as exc:
            _log("memory.metrics", user_id, t0, "error", f"type=InternalError message={exc}")
            logger.exception("Unhandled error in memory.metrics")
            return _error("InternalError", "An internal error occurred")

    return {
        "organism_chat":      organism_chat,
        "memory.store_event": memory_store_event,
        "memory.query":       memory_query,
        "memory.remember":    memory_remember,
        "memory.reset":       memory_reset,
        "memory.metrics":     memory_metrics,
    }


# ---------------------------------------------------------------------------
# Organism bootstrap
# ---------------------------------------------------------------------------

def build_organism_from_env(config_path: str | None = None) -> Any:
    from organism.config import OrganismConfig
    from organism.core.organism import Organism

    path = config_path or os.environ.get("ORGANISM_CONFIG") or "organism_config.yaml"
    cfg = OrganismConfig.from_yaml(path)

    if _model_type := os.environ.get("ORGANISM_MODEL_TYPE"):
        cfg.base_model.type = _model_type
    if _model_name := os.environ.get("ORGANISM_MODEL_NAME"):
        cfg.base_model.model_name = _model_name

    return Organism.from_config(cfg)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def _build_server(org: Any, log_format: str = "plain") -> Any:
    from mcp.server import Server
    import mcp.types as types

    server = Server("organism")
    handlers = _make_handlers(org, log_format=log_format)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in _TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = handlers.get(name)
        if handler is None:
            text = _error("NotFoundError", f"Unknown tool: {name}")
        else:
            text = handler(**arguments)
        return [types.TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import asyncio
    from mcp.server.stdio import stdio_server

    parser = argparse.ArgumentParser(description="Organism MCP stdio server")
    parser.add_argument("--config", default=None, help="Path to organism_config.yaml")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--log-format", default="plain", choices=["plain", "json"],
        help="Log format for MCP tool calls (default: plain)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    org = build_organism_from_env(config_path=args.config)
    server = _build_server(org, log_format=args.log_format)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
