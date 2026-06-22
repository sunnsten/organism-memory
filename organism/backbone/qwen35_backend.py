from __future__ import annotations

import re
from typing import Dict, List, Optional

from .qwen3_backend import Qwen3Backend
from .base import Message
from .attention_trace import AttentionTrace

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()



class Qwen35Backend(Qwen3Backend):
    """
    Backend for Qwen/Qwen3.5-* models.

    Differences from Qwen3Backend:
    - render_chat does NOT pass enable_thinking= (not supported by Qwen3.5 template)
    - generate and generate_with_attention_trace strip <think>...</think> from output

    enable_thinking class attribute signals to callers (e.g. FactExtractor) that this
    backend accepts a thinking= kwarg on generate(). Step 2 will use it to suppress
    reasoning via /nothink; for now the kwarg is accepted and ignored.
    """

    enable_thinking: bool = True

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            # Qwen3.5 tokenizer may not support enable_thinking kwarg
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
        thinking: bool | None = None,
    ) -> str:
        text = super().generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        return _strip_think(text)

    def generate_with_attention_trace(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        mem_spans: Optional[Dict[int, tuple[int, int]]] = None,
        collect_every: int = 1,
        max_collect_steps: Optional[int] = None,
        heads_to_use: Optional[List[int]] = None,
        normalize_by_span_len: bool = True,
    ) -> tuple[str, Optional[AttentionTrace]]:
        text, trace = super().generate_with_attention_trace(
            messages,
            max_new_tokens=max_new_tokens,
            mem_spans=mem_spans,
            collect_every=collect_every,
            max_collect_steps=max_collect_steps,
            heads_to_use=heads_to_use,
            normalize_by_span_len=normalize_by_span_len,
        )
        return _strip_think(text), trace
