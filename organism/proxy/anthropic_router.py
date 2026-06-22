from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic_converter import (
    anthropic_to_openai,
    openai_to_anthropic,
    openai_stream_to_anthropic,
)
from .forwarder import forward_simple, forward_streaming_raw
from .memory_injector import inject_memory, inject_memory_anthropic
from .router import _index_turn, _last_user_message, _extract_reply, _strip_think_from_response

logger = logging.getLogger("organism_proxy")

router = APIRouter()

# ── Pricing table (USD per 1 token). Updated May 2026. ────────────────────────
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7":   (15e-6, 75e-6),
    "claude-opus-4-5":   (15e-6, 75e-6),
    "claude-sonnet-4-6": (3e-6,  15e-6),
    "claude-sonnet-4-5": (3e-6,  15e-6),
    "claude-haiku-4-5":  (0.8e-6, 4e-6),
}
_DEFAULT_PRICING = (3e-6, 15e-6)  # Sonnet fallback for unknown models


def _calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD based on model and token counts."""
    for key, (in_price, out_price) in _ANTHROPIC_PRICING.items():
        if key in model:
            return input_tokens * in_price + output_tokens * out_price
    in_price, out_price = _DEFAULT_PRICING
    return input_tokens * in_price + output_tokens * out_price


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_local_model(cfg: Any, organism: Any) -> str:
    if getattr(cfg, "forward_model", ""):
        return cfg.forward_model
    try:
        return organism._orchestrator.lm.model_name
    except AttributeError:
        return "local"


def _last_user_message_anthropic(body: dict) -> str | None:
    """Extract last user message text from an Anthropic-format body."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [b["text"] for b in content if b.get("type") == "text"]
                return " ".join(parts) or None
    return None


def _extract_reply_anthropic(body: dict) -> str | None:
    """Extract assistant reply text from an Anthropic-format response body."""
    try:
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
    except Exception:
        pass
    return None


def _anthropic_passthrough_headers(original_headers: dict, cfg: Any = None) -> dict:
    """Build headers for forwarding to api.anthropic.com."""
    headers: dict[str, str] = {}

    # If proxy has its own Anthropic key, use it instead of passing client key through.
    # This allows clients to authenticate with proxy-specific keys (sk-organism-...)
    # rather than real Anthropic keys.
    if cfg is not None and getattr(cfg, "anthropic_api_key", ""):
        headers["x-api-key"] = cfg.anthropic_api_key
    else:
        for h in ("authorization", "x-api-key"):
            val = original_headers.get(h)
            if val:
                headers[h] = val

    # Anthropic-specific protocol headers
    for h in ("anthropic-version", "anthropic-beta"):
        val = original_headers.get(h)
        if val:
            headers[h] = val
    headers.setdefault("anthropic-version", "2023-06-01")
    headers["content-type"] = "application/json"
    return headers


# ── Anthropic-native forwarding ───────────────────────────────────────────────

async def _forward_anthropic_simple(
    body: dict,
    forward_url: str,
    headers: dict,
    connect_timeout: float,
    read_timeout: float,
) -> dict:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10, pool=5)
        ) as client:
            resp = await client.post(
                f"{forward_url}/v1/messages",
                json=body,
                headers=headers,
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API unreachable: {exc}")
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Anthropic API timed out")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    return resp.json()


async def _forward_anthropic_stream(
    body: dict,
    forward_url: str,
    headers: dict,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[AsyncIterator[bytes], asyncio.Future[str]]:
    """Stream from Anthropic API. Returns (raw_sse_iterator, reply_future)."""
    full_text_parts: list[str] = []
    reply_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    async def _gen() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10, pool=5)
            ) as client:
                async with client.stream(
                    "POST",
                    f"{forward_url}/v1/messages",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        if not reply_future.done():
                            reply_future.set_result("")
                        yield err
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        _extract_anthropic_text(chunk, full_text_parts)
            if not reply_future.done():
                reply_future.set_result("".join(full_text_parts))
        except Exception as exc:
            logger.error("Anthropic stream error: %s", exc)
            if not reply_future.done():
                reply_future.set_result("")

    return _gen(), reply_future


def _extract_anthropic_text(chunk: bytes, parts: list[str]) -> None:
    """Collect text from Anthropic SSE content_block_delta chunks."""
    try:
        for line in chunk.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("data: "):
                continue
            obj = json.loads(line[6:].strip())
            if obj.get("type") == "content_block_delta":
                delta = obj.get("delta", {})
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
    except Exception:
        pass


# ── Main route ────────────────────────────────────────────────────────────────

@router.post("/v1/messages")
async def messages(request: Request) -> Any:
    from .server import proxy_requests, proxy_memory_injected, proxy_memory_facts, proxy_forward_latency

    t_start = time.monotonic()
    body: dict = await request.json()
    cfg = request.app.state.proxy_config
    organism = request.app.state.organism

    user_id: str = getattr(request.state, "user_id", "default")
    tenant_id: str = getattr(request.state, "tenant_id", "default")
    forward_url = request.headers.get("x-forward-to", cfg.forward_url).rstrip("/")
    stream = body.get("stream", False)

    # ── Anthropic passthrough mode ────────────────────────────────────────────
    if cfg.forward_mode == "anthropic":
        query = _last_user_message_anthropic(body)
        facts_count = 0

        if query:
            facts = organism.retrieve_context(
                user_id=user_id,
                query=query,
                limit=cfg.memory_limit,
            )
            if facts:
                body = inject_memory_anthropic(body, facts, max_tokens=cfg.memory_max_tokens)
                facts_count = len(facts)
                proxy_memory_injected.labels(user_id=user_id).inc()
                logger.info("Injected %d memory facts for user=%s (anthropic passthrough)", facts_count, user_id)

        proxy_memory_facts.observe(facts_count)
        fwd_headers = _anthropic_passthrough_headers(dict(request.headers), cfg=cfg)

        if stream:
            gen, reply_future = await _forward_anthropic_stream(
                body, forward_url, fwd_headers, cfg.connect_timeout, cfg.read_timeout
            )
            if query:
                asyncio.create_task(
                    _index_after_stream(reply_future, organism, user_id, tenant_id, query)
                )
            return StreamingResponse(gen, media_type="text/event-stream",
                                     headers={"X-Accel-Buffering": "no"})
        else:
            result = await _forward_anthropic_simple(
                body, forward_url, fwd_headers, cfg.connect_timeout, cfg.read_timeout
            )
            if query:
                assistant_reply = _extract_reply_anthropic(result)
                if assistant_reply:
                    asyncio.create_task(
                        _index_turn(organism, user_id, tenant_id, query, assistant_reply)
                    )
            # Track token usage and cost
            usage = result.get("usage", {})
            input_tok = usage.get("input_tokens", 0)
            output_tok = usage.get("output_tokens", 0)
            model_used = result.get("model", body.get("model", "unknown"))
            cost = _calculate_cost_usd(model_used, input_tok, output_tok)
            from .server import (
                proxy_anthropic_input_tokens, proxy_anthropic_output_tokens,
                proxy_anthropic_cost_usd, proxy_memory_overhead_tokens,
            )
            proxy_anthropic_input_tokens.labels(user_id=user_id, model=model_used).inc(input_tok)
            proxy_anthropic_output_tokens.labels(user_id=user_id, model=model_used).inc(output_tok)
            proxy_anthropic_cost_usd.labels(user_id=user_id, model=model_used).inc(cost)
            proxy_memory_overhead_tokens.observe(facts_count * 50)  # ~50 tok per fact heuristic

            duration_ms = (time.monotonic() - t_start) * 1000
            proxy_requests.labels(user_id=user_id, status="200").inc()
            proxy_forward_latency.observe(duration_ms)
            return JSONResponse(result)

    # ── OpenAI / vLLM mode (default) ─────────────────────────────────────────
    local_model = _resolve_local_model(cfg, organism)
    oai_body = anthropic_to_openai(body, local_model)

    messages_list = oai_body.get("messages", [])
    query = _last_user_message(messages_list)
    facts_count = 0

    if query:
        facts = organism.retrieve_context(
            user_id=user_id,
            query=query,
            limit=cfg.memory_limit,
        )
        if facts:
            messages_list = inject_memory(messages_list, facts, max_tokens=cfg.memory_max_tokens)
            oai_body = {**oai_body, "messages": messages_list}
            facts_count = len(facts)
            proxy_memory_injected.labels(user_id=user_id).inc()
            logger.info("Injected %d memory facts for user=%s (openai mode)", facts_count, user_id)

    proxy_memory_facts.observe(facts_count)

    if cfg.strip_think and "chat_template_kwargs" not in oai_body:
        oai_body = {**oai_body, "chat_template_kwargs": {"enable_thinking": False}}

    if stream:
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        gen, reply_future = await forward_streaming_raw(
            payload=oai_body,
            forward_url=forward_url,
            original_headers=dict(request.headers),
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
        if query:
            asyncio.create_task(
                _index_after_stream(reply_future, organism, user_id, tenant_id, query)
            )
        return StreamingResponse(
            openai_stream_to_anthropic(gen, local_model, msg_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    else:
        result = await forward_simple(
            payload=oai_body,
            forward_url=forward_url,
            original_headers=dict(request.headers),
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
        if cfg.strip_think:
            result = _strip_think_from_response(result)
        if query:
            assistant_reply = _extract_reply(result)
            if assistant_reply:
                asyncio.create_task(
                    _index_turn(organism, user_id, tenant_id, query, assistant_reply)
                )
        duration_ms = (time.monotonic() - t_start) * 1000
        proxy_requests.labels(user_id=user_id, status="200").inc()
        proxy_forward_latency.observe(duration_ms)
        return JSONResponse(openai_to_anthropic(result, local_model))


async def _index_after_stream(
    reply_future: asyncio.Future,
    organism: Any,
    user_id: str,
    tenant_id: str,
    user_message: str,
) -> None:
    try:
        assistant_reply = await asyncio.wait_for(reply_future, timeout=180)
        if assistant_reply:
            await _index_turn(organism, user_id, tenant_id, user_message, assistant_reply)
    except Exception as exc:
        logger.warning("Index after stream failed: %s", exc)


__all__ = ["router"]
