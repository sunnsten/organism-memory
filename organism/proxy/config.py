from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ProxyConfig:
    port: int = 9000
    auth_mode: str = "none"          # "none" | "api_key"
    forward_mode: str = "openai"     # "openai" (vLLM) | "anthropic" (api.anthropic.com)
    forward_url: str = "http://localhost:8080/v1"
    memory_limit: int = 8
    memory_max_tokens: int = 500
    connect_timeout: float = 30.0
    read_timeout: float = 120.0
    strip_think: bool = True         # strip <think>…</think> from responses
    anthropic_api_key: str = ""
    """Real Anthropic API key used when forwarding to api.anthropic.com.
    When set, substitutes the client's x-api-key so clients can authenticate
    with proxy-specific keys (sk-organism-...) instead of real Anthropic keys.
    Loaded from ORGANISM_ANTHROPIC_API_KEY env var if not set here."""
    organism_config_path: str = ""
    """Path to OrganismConfig YAML. Overrides ORGANISM_CONFIG_PATH env var and
    auto-detection fallback. Use to pin the proxy to a specific config file."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ProxyConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        proxy = data.get("proxy", {})
        return cls(
            port=proxy.get("port", 9000),
            auth_mode=proxy.get("auth_mode", "none"),
            forward_mode=proxy.get("forward_mode", "openai"),
            forward_url=proxy.get("forward_url", "http://localhost:8080/v1"),
            memory_limit=proxy.get("memory_limit", 8),
            memory_max_tokens=proxy.get("memory_max_tokens", 500),
            connect_timeout=float(proxy.get("connect_timeout", 30.0)),
            read_timeout=float(proxy.get("read_timeout", 120.0)),
            strip_think=proxy.get("strip_think", True),
            anthropic_api_key=os.environ.get("ORGANISM_ANTHROPIC_API_KEY") or str(proxy.get("anthropic_api_key") or ""),
            organism_config_path=os.environ.get("ORGANISM_CONFIG_PATH") or str(proxy.get("organism_config_path") or ""),
        )


__all__ = ["ProxyConfig"]
