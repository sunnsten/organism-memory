from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


# ── Request conversion ────────────────────────────────────────────────────────

def anthropic_to_openai(body: dict, local_model: str) -> dict:
    """Convert Anthropic /v1/messages request body → OpenAI /v1/chat/completions."""
    messages: list[dict] = []

    # System prompt — top-level field in Anthropic (string or content-block array)
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            # Content-block array → plain string
            parts: list[str] = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                # images / tool_use / tool_result silently skipped for now
            content = "\n".join(parts)
        messages.append({"role": msg["role"], "content": content})

    oai: dict[str, Any] = {
        "model": local_model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 1024),
        "stream": body.get("stream", False),
    }
    if "temperature" in body:
        oai["temperature"] = body["temperature"]
    if "top_p" in body:
        oai["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        oai["stop"] = body["stop_sequences"]
    return oai


# ── Non-streaming response conversion ────────────────────────────────────────

def openai_to_anthropic(oai_resp: dict, model: str) -> dict:
    """Convert OpenAI chat completion response → Anthropic Messages API response."""
    choice = oai_resp["choices"][0]
    content_text = choice["message"].get("content") or ""
    usage = oai_resp.get("usage", {})
    finish = choice.get("finish_reason") or "stop"

    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": model,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── Streaming SSE conversion ──────────────────────────────────────────────────

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def openai_stream_to_anthropic(
    oai_chunks: AsyncIterator[bytes],
    model: str,
    msg_id: str,
) -> AsyncIterator[bytes]:
    """
    Wrap an OpenAI SSE byte-stream and yield Anthropic SSE events.

    Emits the full Anthropic streaming protocol:
      message_start → content_block_start → ping →
      N × content_block_delta → content_block_stop →
      message_delta → message_stop
    """
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    })
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield _sse("ping", {"type": "ping"})

    output_tokens = 0
    stop_reason = "end_turn"

    async for chunk in oai_chunks:
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta_content = choice.get("delta", {}).get("content")
            if delta_content:
                output_tokens += 1
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": delta_content},
                })
            finish = choice.get("finish_reason")
            if finish:
                stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


__all__ = ["anthropic_to_openai", "openai_to_anthropic", "openai_stream_to_anthropic"]
