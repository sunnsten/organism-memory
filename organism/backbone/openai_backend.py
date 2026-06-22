from __future__ import annotations

import logging
import re
from typing import List

import torch

from .base import LMBackend, Message, EncodedText

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAIBackend(LMBackend):
    """
    LMBackend that calls any OpenAI-compatible /v1/chat/completions endpoint.

    Works with vLLM, llama.cpp server, Ollama, LM Studio, OpenAI API.
    think-block stripping enabled by default for Qwen3/Qwen3.5 models.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        strip_think: bool = True,
        enable_thinking: bool = False,
        thinking_budget: int = 0,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for OpenAIBackend: pip install httpx"
            )

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.strip_think = strip_think
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300.0 if enable_thinking else 120.0,
        )
        logger.info("OpenAIBackend ready: %s → %s", model_name, base_url)

    # ------------------------------------------------------------------
    # LMBackend interface
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def hidden_size(self) -> int:
        return 0

    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
        thinking: bool | None = None,
    ) -> str:
        # thinking=None → use instance default; False/True → force override
        use_thinking = self.enable_thinking if thinking is None else thinking
        response_tokens = max_new_tokens or self.max_new_tokens
        if use_thinking and self.thinking_budget > 0:
            # vLLM counts thinking tokens against max_tokens, so we must expand
            # the budget to fit both the <think> block and the actual response.
            total_tokens = self.thinking_budget + response_tokens
        else:
            total_tokens = response_tokens
        body: dict = {
            "model": model_override or self.model_name,
            "messages": messages,
            "max_tokens": total_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
        }
        if use_thinking:
            # Enable reasoning mode; budget=0 means no limit.
            ktw: dict = {"enable_thinking": True}
            if self.thinking_budget > 0:
                ktw["thinking_budget"] = self.thinking_budget
            body["chat_template_kwargs"] = ktw
        elif self.strip_think or not use_thinking:
            # Explicitly disable thinking so the model skips reasoning entirely.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        text: str = resp.json()["choices"][0]["message"]["content"]
        if self.strip_think:
            # Strip <think> blocks before returning — keeps stored messages clean.
            text = _THINK_RE.sub("", text).strip()
        return text

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        # Server applies its own chat template — return last user message as fallback
        return messages[-1]["content"] if messages else ""

    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        raise NotImplementedError(
            "encode_text() requires access to model internals. "
            "Use a local backend for research/attention features."
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
