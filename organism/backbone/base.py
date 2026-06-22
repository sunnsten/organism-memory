from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from organism.backbone.attention_trace import AttentionTrace


Message = Dict[str, str]


@dataclass
class EncodedText:
    """Result of encoding text through LMBackend."""
    hidden_states: Tensor       # [1, L, D]
    attention_mask: Tensor      # [1, L]
    seq_len: int                # L
    d_model: int                # D
    attentions: Optional[Tensor] = None   # [1, H, L, L] — present when need_attn=True
    surprisal: Optional[Tensor] = None   # [1] — present when need_surprisal=True


EncodeAndUpdateSSM = Callable[[str], EncodedText]


class LMBackend(ABC):
    """Abstract interface over any language model backend."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Default: len(text) // 4. Override for accuracy."""
        return len(text) // 4

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
    ) -> str:
        """
        Generate a response from a list of chat messages.

        messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        temperature: overrides backend default when provided (None = use backend's self.temperature).
        model_override: per-request model name (OpenAIBackend only; local backends ignore it).
        Returns the assistant text.
        """
        raise NotImplementedError

    def generate_with_attention_trace(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        mem_spans: Optional[Dict[int, tuple[int, int]]] = None,
        collect_every: int = 1,
        max_collect_steps: Optional[int] = None,
        heads_to_use: Optional[List[int]] = None,
        normalize_by_span_len: bool = True,
    ) -> tuple[str, Optional["AttentionTrace"]]:
        """
        Generate a response and optionally collect an attention trace.

        Collects attention from the last layer and last token only (q_len=1),
        which is much cheaper on VRAM than a full L×L matrix.

        mem_spans: {mem_id: (start_token_idx, end_token_idx)} token spans for each <MEM id=...>.
            If None, mem_attention_scores will be empty.
        collect_every: collect attention every N steps (1 = every step).
        """
        text = self.generate(messages, max_new_tokens)
        return text, None

    @abstractmethod
    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        """Encode text into hidden states."""
        raise NotImplementedError

    @abstractmethod
    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        """
        Render messages to a plain string via the model's chat template.

        Returns a plain string (not tokenized). Tokenization is done separately in
        encode_text() or generate(). Both methods use this to guarantee consistent formatting.
        """
        raise NotImplementedError

    def encode(self, text: str) -> EncodedText:
        """Alias for encode_text with default parameters."""
        return self.encode_text(text, need_attn=True, need_surprisal=False)
