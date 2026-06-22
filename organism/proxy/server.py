from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app

from .anthropic_router import router as anthropic_router
from .auth import AuthMiddleware
from .config import ProxyConfig
from .router import router

# Proxy-specific Prometheus metrics
proxy_requests = Counter(
    "proxy_requests_total",
    "Total requests through the memory proxy",
    labelnames=["user_id", "status"],
)
proxy_memory_injected = Counter(
    "proxy_memory_injected_total",
    "Total requests where memory was injected",
    labelnames=["user_id"],
)
proxy_memory_facts = Histogram(
    "proxy_memory_facts_count",
    "Number of memory facts injected per request",
    buckets=[0, 1, 2, 3, 5, 8, 10],
)
proxy_forward_latency = Histogram(
    "proxy_forward_latency_ms",
    "End-to-end proxy request latency in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 5000, 15000, 30000],
)

# Anthropic token cost tracking
proxy_anthropic_input_tokens = Counter(
    "proxy_anthropic_input_tokens_total",
    "Total input tokens sent to Anthropic API",
    labelnames=["user_id", "model"],
)
proxy_anthropic_output_tokens = Counter(
    "proxy_anthropic_output_tokens_total",
    "Total output tokens received from Anthropic API",
    labelnames=["user_id", "model"],
)
proxy_anthropic_cost_usd = Counter(
    "proxy_anthropic_cost_usd_total",
    "Estimated total cost in USD for Anthropic API calls",
    labelnames=["user_id", "model"],
)
proxy_memory_overhead_tokens = Histogram(
    "proxy_memory_overhead_tokens",
    "Tokens added to prompt by Organism memory injection",
    buckets=[1, 50, 100, 200, 400, 600, 800, 1000],
)

logger = logging.getLogger("organism_proxy")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_config() -> ProxyConfig:
    cfg_path = os.environ.get("ORGANISM_PROXY_CONFIG")
    if cfg_path:
        return ProxyConfig.from_yaml(cfg_path)
    default = ROOT_DIR / "organism_proxy.yaml"
    if default.exists():
        return ProxyConfig.from_yaml(default)
    return ProxyConfig()


def _load_organism(cfg: ProxyConfig):
    from organism.config import OrganismConfig
    from organism.core.organism import Organism

    # Priority: ProxyConfig.organism_config_path (which already absorbed ORGANISM_CONFIG_PATH
    # env var during ProxyConfig.from_yaml) → auto-detection fallback
    org_cfg_path = cfg.organism_config_path or None
    if not org_cfg_path:
        for name in ("organism_config_mcp.yaml", "organism_config.yaml"):
            p = ROOT_DIR / name
            if p.exists():
                org_cfg_path = str(p)
                break

    if org_cfg_path:
        org_cfg = OrganismConfig.from_yaml(org_cfg_path)
        logger.info("Organism config: %s", org_cfg_path)
    else:
        org_cfg = OrganismConfig()

    return Organism.from_config(org_cfg)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Organism proxy starting on port %d", app.state.proxy_config.port)
    yield
    logger.info("Organism proxy stopped")


def create_app() -> FastAPI:
    cfg = _load_config()
    organism = _load_organism(cfg)

    from .api_key_store import ApiKeyStore
    from organism.core.config import CoreConfig
    db_path = Path(CoreConfig().db_path)
    api_key_store = ApiKeyStore(db_path)

    app = FastAPI(title="Organism Memory Proxy", lifespan=_lifespan)
    app.state.proxy_config = cfg
    app.state.organism = organism
    app.state.api_key_store = api_key_store

    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    app.include_router(anthropic_router)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    def health():
        return {"status": "ok", "auth_mode": cfg.auth_mode}

    return app


app = create_app()
