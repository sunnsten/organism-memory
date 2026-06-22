from __future__ import annotations

from typing import List

from .qwen3_backend import Qwen3Backend
from .base import Message


class Qwen25Backend(Qwen3Backend):
    """
    Backend for Qwen/Qwen2.5-* models (e.g. Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct).

    Qwen2.5 uses the same AutoModelForCausalLM architecture and HuggingFace API as
    Qwen3, but its tokenizer.apply_chat_template() does NOT accept the enable_thinking
    keyword argument that Qwen3 requires. This subclass overrides only render_chat()
    to use the standard call without enable_thinking.

    All other methods (generate, count_tokens, encode_text, generate_with_attention_trace)
    are inherited unchanged from Qwen3Backend.
    """

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        """Standard chat template rendering without enable_thinking (not supported in Qwen2.5)."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            # enable_thinking intentionally omitted — not supported in Qwen2.5
        )


__all__ = ["Qwen25Backend"]
