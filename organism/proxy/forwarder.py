from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("organism_proxy")


def _build_forward_headers(original_headers: dict) -> dict:
    headers = {
        k: v for k, v in original_headers.items()
        if k.lower() in ("authorization", "content-type", "x-forward-to")
    }
    headers["content-type"] = "application/json"
    return headers


async def forward_streaming_raw(
    payload: dict,
    forward_url: str,
    original_headers: dict,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[AsyncIterator[bytes], asyncio.Future[str]]:
    """
    Return (async_iterator_of_raw_bytes, reply_future).

    The iterator yields raw OpenAI SSE bytes from the backend.
    reply_future resolves to the full concatenated assistant text when done.
    Callers can wrap the iterator in StreamingResponse or convert to another format.
    """
    forward_headers = _build_forward_headers(original_headers)
    full_text_parts: list[str] = []
    reply_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    async def _gen() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10, pool=5)
            ) as client:
                async with client.stream(
                    "POST",
                    f"{forward_url}/chat/completions",
                    json=payload,
                    headers=forward_headers,
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        logger.error("Backend error %d: %s", resp.status_code, body[:200])
                        if not reply_future.done():
                            reply_future.set_result("")
                        yield b"data: [DONE]\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        _extract_delta(chunk, full_text_parts)
            if not reply_future.done():
                reply_future.set_result("".join(full_text_parts))
        except httpx.ConnectError as exc:
            logger.error("Backend unreachable: %s", exc)
            if not reply_future.done():
                reply_future.set_result("")
            yield b'data: {"error":"Backend LLM unreachable"}\n\ndata: [DONE]\n\n'
        except Exception as exc:
            logger.error("Forward error: %s", exc)
            if not reply_future.done():
                reply_future.set_result("")

    return _gen(), reply_future


async def forward_streaming(
    payload: dict,
    forward_url: str,
    original_headers: dict,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[StreamingResponse, asyncio.Future[str]]:
    """
    Stream response from backend LLM back to client (OpenAI SSE format).
    Returns (StreamingResponse, reply_future).
    """
    gen, reply_future = await forward_streaming_raw(
        payload, forward_url, original_headers, connect_timeout, read_timeout
    )
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    ), reply_future


def _extract_delta(chunk: bytes, parts: list[str]) -> None:
    """Parse SSE delta content from a raw chunk and append to parts."""
    import json
    try:
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                parts.append(content)
    except Exception:
        pass


async def forward_simple(
    payload: dict,
    forward_url: str,
    original_headers: dict,
    connect_timeout: float,
    read_timeout: float,
) -> dict:
    """Non-streaming forward — used when stream=False."""
    forward_headers = {
        k: v for k, v in original_headers.items()
        if k.lower() in ("authorization", "content-type", "x-forward-to")
    }
    forward_headers["content-type"] = "application/json"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10, pool=5)
        ) as client:
            resp = await client.post(
                f"{forward_url}/chat/completions",
                json=payload,
                headers=forward_headers,
            )
    except httpx.ConnectError as exc:
        logger.error("Backend unreachable: %s", exc)
        raise HTTPException(status_code=502, detail=f"Backend LLM unreachable: {exc}")
    except httpx.TimeoutException as exc:
        logger.error("Backend timeout: %s", exc)
        raise HTTPException(status_code=504, detail="Backend LLM timed out")

    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail=f"Backend error {resp.status_code}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    return resp.json()


__all__ = ["forward_streaming", "forward_simple"]
