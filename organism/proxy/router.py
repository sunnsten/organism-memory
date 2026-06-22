from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .forwarder import forward_simple, forward_streaming
from .memory_injector import inject_memory

logger = logging.getLogger("organism_proxy")

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    from .server import proxy_requests, proxy_memory_injected, proxy_memory_facts, proxy_forward_latency

    t_start = time.monotonic()
    body: dict = await request.json()
    cfg = request.app.state.proxy_config
    organism = request.app.state.organism

    # Auth result is attached by AuthMiddleware
    user_id: str = getattr(request.state, "user_id", "default")
    tenant_id: str = getattr(request.state, "tenant_id", "default")

    # Determine forward URL (header override or config default)
    forward_url = request.headers.get("x-forward-to", cfg.forward_url).rstrip("/")

    # Retrieve memory and inject into messages
    messages = body.get("messages", [])
    query = _last_user_message(messages)
    facts_count = 0

    if query:
        facts = organism.retrieve_context(
            user_id=user_id,
            query=query,
            limit=cfg.memory_limit,
        )
        if facts:
            messages = inject_memory(messages, facts, max_tokens=cfg.memory_max_tokens)
            body = {**body, "messages": messages}
            facts_count = len(facts)
            proxy_memory_injected.labels(user_id=user_id).inc()
            logger.info("Injected %d memory facts for user=%s", facts_count, user_id)

    proxy_memory_facts.observe(facts_count)

    # Disable thinking mode on the backend (vLLM/Qwen3.x accept this parameter)
    if cfg.strip_think and "chat_template_kwargs" not in body:
        body = {**body, "chat_template_kwargs": {"enable_thinking": False}}

    stream = body.get("stream", False)

    if stream:
        response, reply_future = await forward_streaming(
            payload=body,
            forward_url=forward_url,
            original_headers=dict(request.headers),
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
        # Index async after streaming completes
        if query:
            asyncio.create_task(
                _index_after_stream(
                    reply_future=reply_future,
                    organism=organism,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    user_message=query,
                )
            )
        return response
    else:
        result = await forward_simple(
            payload=body,
            forward_url=forward_url,
            original_headers=dict(request.headers),
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
        # Strip <think> blocks from response content before returning to client
        if cfg.strip_think:
            result = _strip_think_from_response(result)

        # Index async
        if query:
            assistant_reply = _extract_reply(result)
            if assistant_reply:
                asyncio.create_task(
                    _index_turn(
                        organism=organism,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        user_message=query,
                        assistant_reply=assistant_reply,
                    )
                )
        duration_ms = (time.monotonic() - t_start) * 1000
        proxy_requests.labels(user_id=user_id, status="200").inc()
        proxy_forward_latency.observe(duration_ms)
        return JSONResponse(result)


def _strip_think_from_response(result: dict) -> dict:
    """Remove <think>…</think> blocks from all choice message contents."""
    choices = result.get("choices")
    if not choices:
        return result
    new_choices = []
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and "<think>" in content:
            cleaned = _THINK_RE.sub("", content).strip()
            choice = {**choice, "message": {**msg, "content": cleaned}}
        new_choices.append(choice)
    return {**result, "choices": new_choices}


def _last_user_message(messages: list[dict]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return None


def _extract_reply(result: dict) -> Optional[str]:
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


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


async def _index_turn(
    organism: Any,
    user_id: str,
    tenant_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    """Save user+assistant pair as ExperienceBlock (Research tier raw)."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: _sync_index(organism, user_id, tenant_id, user_message, assistant_reply),
        )
    except Exception as exc:
        logger.warning("Index turn failed user=%s: %s", user_id, exc)


def _sync_index(
    organism: Any,
    user_id: str,
    tenant_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    import time
    from organism.shared.domain import EventRecord, ContextMeta

    store = organism._orchestrator._memory.store
    session_id = f"{user_id}_proxy"

    user_msg_id = store.messages.add(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        content=user_message,
    )
    asst_msg_id = store.messages.add(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role="assistant",
        content=assistant_reply,
    )

    event = EventRecord(
        id=None,
        user_id=user_id,
        session_id=session_id,
        timestamp=time.time(),
        input_text=user_message,
        output_text=assistant_reply,
        kind="interaction",
        source="proxy",
        importance=0.5,
        surprisal_norm=None,
        attention_focus=None,
        used_memories=[],
        used_memories_space=None,
        context_meta=ContextMeta(
            system_hash="",
            memory_ids=[],
            chat_message_id_span=(user_msg_id, asst_msg_id),
            memory_id_space=None,
        ),
        text_preview=f"{user_message}\n{assistant_reply}"[:500],
        embedding=None,
        embedding_dim=None,
        embedding_dtype="float32",
        embedding_l2norm=False,
    )
    organism._orchestrator._memory.write.append_event(event, tenant_id=tenant_id)
    logger.debug("Indexed proxy turn user=%s", user_id)

    # Tier 2: async fact extraction (non-blocking, background thread)
    fact_extractor = getattr(organism._orchestrator, "_fact_extractor", None)
    if fact_extractor is not None:
        fact_extractor.extract_and_store_later(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply},
            ],
        )


__all__ = ["router"]
