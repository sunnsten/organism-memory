from __future__ import annotations

import logging

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("organism_proxy")


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Resolves user_id from request based on auth_mode.

    none:    X-User-Id header → user_id (default: "default")
    api_key: Bearer sk-organism-xxx → lookup in api_key_store
             X-User-Id silently ignored
             Missing/invalid bearer → 401
    """

    async def dispatch(self, request: Request, call_next):
        auth_mode: str = request.app.state.proxy_config.auth_mode

        if auth_mode == "api_key":
            # Accept both OpenAI-style Bearer and Anthropic-style x-api-key
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                raw_key = auth_header[7:].strip()
            else:
                raw_key = request.headers.get("x-api-key", "").strip()
            if not raw_key:
                return JSONResponse(
                    {"error": "Unauthorized", "detail": "Bearer token or x-api-key required"},
                    status_code=401,
                )
            key_store = request.app.state.api_key_store
            resolved = key_store.resolve(raw_key)
            if resolved is None:
                return JSONResponse(
                    {"error": "Unauthorized", "detail": "Invalid or expired API key"},
                    status_code=401,
                )
            request.state.user_id = resolved["user_id"]
            request.state.tenant_id = resolved.get("tenant_id", "default")
        else:
            # auth_mode == "none" — x-api-key accepted but ignored (any value passes)
            user_id = request.headers.get("x-user-id", "default").strip() or "default"
            request.state.user_id = user_id
            request.state.tenant_id = "default"

        return await call_next(request)


__all__ = ["AuthMiddleware"]
