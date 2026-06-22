from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for Llama-style attention."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: [batch_size, seq_len]
        Returns:
            cos, sin: [batch_size, seq_len, head_dim]
        """
        device = position_ids.device
        dtype = position_ids.dtype
        batch_size, seq_len = position_ids.shape

        inv_freq_tensor = cast(torch.Tensor, self.inv_freq)
        freqs = torch.outer(position_ids[0].float(), inv_freq_tensor)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

        return cos.to(dtype).to(device), sin.to(dtype).to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_heads, seq_len, head_dim]
        cos, sin: [batch_size, seq_len, head_dim]
    Returns:
        q_rot, k_rot: [batch_size, num_heads, seq_len, head_dim]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CustomLlamaAttention(nn.Module):
    """
    Custom Llama attention with RoPE. Compatible with Llama architecture
    but without a dependency on transformers internals.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
        )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]
            position_ids: [batch_size, seq_len]
            position_embeddings: (cos, sin) — precomputed RoPE embeddings
            past_key_value: (past_key, past_value) for KV-cache
            output_attentions: whether to return attention weights
            use_cache: whether to return present key/value
            cache_position: cache positions (for newer transformers versions)
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] (if output_attentions)
            present_key_value: (key, value) for the next step (if use_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif position_ids is not None:
            cos, sin = self.rotary_emb(position_ids)
        else:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            cos, sin = self.rotary_emb(position_ids)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout_layer(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        output = self.o_proj(attn_output)

        if output_attentions:
            return output, attn_weights, present_key_value
        return output, None, present_key_value
