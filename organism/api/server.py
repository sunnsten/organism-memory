import asyncio
import concurrent.futures
import os
import re
import time
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from organism.config import OrganismConfig
from organism.core.organism import Organism
from organism.shared.analytics import analytics

# =========================
# LOGGER
# =========================
logger = logging.getLogger("organism_api")
logging.basicConfig(level=logging.INFO)

LM_TIMEOUT_SECONDS: float = float(os.environ.get("ORGANISM_LM_TIMEOUT", "300"))
# Override with ORGANISM_CHAT_RATE_LIMIT=1000/minute for local benchmarks.
_CHAT_RATE_LIMIT: str = os.environ.get("ORGANISM_CHAT_RATE_LIMIT", "10/minute")
_chat_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="lm")
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# =========================
# INITIALIZATION
# =========================

ROOT_DIR = Path(__file__).resolve().parents[2]
_config_path_env = os.environ.get("ORGANISM_CONFIG_PATH")
CONFIG_PATH = Path(_config_path_env) if _config_path_env else ROOT_DIR / "organism_config_mcp.yaml"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT_DIR / "organism_config.yaml"

if CONFIG_PATH.exists():
    config = OrganismConfig.from_yaml(CONFIG_PATH)
    logger.info("Config loaded from %s", CONFIG_PATH)
else:
    config = OrganismConfig()

# ENV override: allows running multiple services from one organism_config.yaml
# by passing ORGANISM_MODEL_TYPE / ORGANISM_MODEL_NAME via docker compose environment.
if _model_type := os.environ.get("ORGANISM_MODEL_TYPE"):
    config.base_model.type = _model_type
if _model_name := os.environ.get("ORGANISM_MODEL_NAME"):
    config.base_model.model_name = _model_name

# LM backend is loaded lazily on the first request
organism = Organism.from_config(config)

# =========================
# FASTAPI APP
# =========================

@asynccontextmanager
async def _lifespan(app):
    yield
    organism.close()
    _chat_executor.shutdown(wait=False)
    logger.info("LM executor shut down")


app = FastAPI(title="Organism API", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Prometheus /metrics endpoint
app.mount("/metrics", make_asgi_app())


# =========================
# METRICS
# =========================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track per-endpoint latency, record to Prometheus and log."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    tenant_id = request.headers.get("x-tenant-id", "unknown")
    path = request.url.path
    # Skip /metrics itself to avoid self-referential noise
    if path != "/metrics":
        analytics.metric_http(
            method=request.method,
            endpoint=path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        logger.info(
            "API %s %s tenant=%s latency=%.1fms status=%d",
            request.method, path, tenant_id, duration_ms, response.status_code,
        )
    return response


# =========================
# REQUEST / RESPONSE MODELS
# =========================

class SessionStartRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


class SessionStartResponse(BaseModel):
    session_id: str


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    model: Optional[str] = None  # Per-request model override (OpenAI backend only)


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ReplayTurn(BaseModel):
    role: str
    content: str
    has_answer: bool = False


class SessionReplayRequest(BaseModel):
    user_id: str
    session_id: str
    turns: List[ReplayTurn]
    session_ts: Optional[float] = None


class SessionReplayResponse(BaseModel):
    session_id: str
    n_blocks: int
    facts_extracted: int
    chunks_embedded: int


class RememberRequest(BaseModel):
    user_id: str = "default"
    text: str


class RememberResponse(BaseModel):
    status: str
    memory_id: int


# =========================
# VALIDATION
# =========================

def _validate_user_id(user_id: str) -> str:
    """Validate and normalize user_id for API endpoints."""
    user_id = user_id.strip() or "default"
    if len(user_id) > 32:
        user_id = user_id[:32]
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    return user_id


# =========================
# ENDPOINTS
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest) -> SessionStartResponse:
    try:
        user_id = _validate_user_id(req.user_id)
        session_id = organism.start_session(user_id=user_id, title=req.title)
        return SessionStartResponse(session_id=session_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /session/start")
        raise HTTPException(status_code=500, detail="Failed to start session")


@app.post("/session/end")
def end_session(req: SessionEndRequest):
    try:
        user_id = _validate_user_id(req.user_id)
        organism.end_session(user_id=user_id, session_id=req.session_id)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /session/end")
        raise HTTPException(status_code=500, detail="Failed to end session")


@app.post("/session/replay", response_model=SessionReplayResponse)
def session_replay(req: SessionReplayRequest) -> SessionReplayResponse:
    """Fast-replay: write session turns directly to DB without LLM generation.

    Equivalent to _replay_session_fast() in direct mode. After writing all turns,
    synchronously runs fact extraction and chunk embedding so memory is ready before
    the eval question is asked via /chat.
    """
    try:
        user_id = _validate_user_id(req.user_id)
        from organism.shared.domain import EventRecord, ContextMeta

        store = organism._orchestrator._memory.store
        tenant_id = organism._tenant_id
        session_id = req.session_id
        session_ts = req.session_ts or time.time()

        n_blocks = 0
        pending_user: Optional[str] = None
        pending_user_msg_id: Optional[int] = None
        turn_offset = 0

        for turn in req.turns:
            if not turn.content:
                continue
            if turn.role == "user":
                msg_id = store.messages.add(
                    session_id=session_id, tenant_id=tenant_id,
                    user_id=user_id, role="user", content=turn.content,
                )
                pending_user = turn.content
                pending_user_msg_id = msg_id
                turn_offset += 1
            elif turn.role == "assistant" and pending_user is not None and pending_user_msg_id is not None:
                asst_msg_id = store.messages.add(
                    session_id=session_id, tenant_id=tenant_id,
                    user_id=user_id, role="assistant", content=turn.content,
                )
                event = EventRecord(
                    id=None, user_id=user_id, session_id=session_id,
                    timestamp=session_ts + turn_offset * 0.001,
                    input_text=pending_user, output_text=turn.content,
                    kind="interaction", source="fast_replay", importance=0.5,
                    surprisal_norm=None, attention_focus=None,
                    used_memories=[], used_memories_space=None,
                    context_meta=ContextMeta(
                        system_hash="", memory_ids=[],
                        chat_message_id_span=(pending_user_msg_id, asst_msg_id),
                        memory_id_space=None,
                    ),
                    text_preview=f"{pending_user}\n{turn.content}"[:500],
                    embedding=None, embedding_dim=None,
                    embedding_dtype="float32", embedding_l2norm=False,
                )
                organism._orchestrator._memory.write.append_event(event, tenant_id=tenant_id)
                n_blocks += 1
                pending_user = None
                pending_user_msg_id = None

        # Extract facts synchronously so FactRetriever has data at eval time
        facts_extracted = 0
        messages_for_extraction = [
            {"role": t.role, "content": t.content}
            for t in req.turns if t.role in ("user", "assistant")
        ]
        if messages_for_extraction and hasattr(organism, "extract_session_facts"):
            try:
                facts_extracted = organism.extract_session_facts(
                    user_id=user_id, session_id=session_id,
                    messages=messages_for_extraction,
                )
            except Exception as exc:
                logger.warning("Fact extraction in replay failed: %s", exc)

        # Bulk-embed chunks written during replay
        chunks_embedded = 0
        if hasattr(organism, "embed_session_chunks"):
            try:
                chunks_embedded = organism.embed_session_chunks(
                    user_id=user_id, session_id=session_id,
                )
            except Exception as exc:
                logger.warning("Chunk embedding in replay failed: %s", exc)

        return SessionReplayResponse(
            session_id=session_id, n_blocks=n_blocks,
            facts_extracted=facts_extracted, chunks_embedded=chunks_embedded,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /session/replay")
        raise HTTPException(status_code=500, detail="Failed to replay session")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(_CHAT_RATE_LIMIT)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    try:
        user_id = _validate_user_id(req.user_id)
        session_id = req.session_id
        if session_id is None:
            session_id = organism.start_session(user_id=user_id)

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _chat_executor,
                    lambda: organism.chat(
                        user_id=user_id,
                        user_message=req.message,
                        session_id=session_id,
                        system_prompt=req.system_prompt,
                        max_new_tokens=req.max_new_tokens,
                        model_override=req.model,
                    ),
                ),
                timeout=LM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("LM call timed out after %.0fs for user=%s", LM_TIMEOUT_SECONDS, user_id)
            logger.warning("LM thread is still running in background (not cancellable) — slot consumed until thread completes")
            raise HTTPException(status_code=504, detail="LM call timed out")

        return ChatResponse(reply=result.reply, session_id=session_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail="Chat failed")


@app.post("/remember", response_model=RememberResponse)
@limiter.limit("30/minute")
def remember(req: RememberRequest, request: Request) -> RememberResponse:
    try:
        user_id = _validate_user_id(req.user_id)
        mem_id = organism.remember(user_id=user_id, text=req.text)
        return RememberResponse(status="ok", memory_id=mem_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /remember")
        raise HTTPException(status_code=500, detail="Remember failed")


@app.get("/api/debug/memory")
@limiter.limit("30/minute")
def debug_memory(request: Request, user_id: str = "default"):
    try:
        user_id = _validate_user_id(user_id)
        view = organism.memory_service.get_debug_view(user_id, last_n=5)
        return asdict(view)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /api/debug/memory")
        raise HTTPException(status_code=500, detail="Debug view failed")


# =========================
# STATIC / HTML
# =========================

STATIC_DIR = Path(__file__).parent / "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)
