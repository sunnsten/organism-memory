from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class MemSpan:
    """Token span for a memory entry in the prompt."""
    mem_id: int  # memory ID
    start: int   # first token index (inclusive)
    end: int     # last token index (inclusive)


def aggregate_mem_attention(
    attn_over_prompt: Tensor,
    mem_spans: List[MemSpan],
    *,
    normalize_by_span_len: bool = False,
) -> Dict[int, float]:
    """
    Aggregate attention weights over memory spans.

    Args:
        attn_over_prompt: [prompt_len] — attention weights for the last token over prompt tokens
        mem_spans: memory spans within the prompt
        normalize_by_span_len: divide by span length (makes scores comparable across spans)

    Returns:
        {mem_id: score} — accumulated score per memory
    """
    scores: Dict[int, float] = {}
    prompt_len = int(attn_over_prompt.shape[0])

    for sp in mem_spans:
        s = max(0, min(int(sp.start), prompt_len - 1))
        e = max(0, min(int(sp.end), prompt_len - 1))
        if e < s:
            s, e = e, s

        seg = attn_over_prompt[s : e + 1]
        val = float(seg.sum().item())

        if normalize_by_span_len:
            denom = float((e - s + 1))
            if denom > 0:
                val = val / denom

        scores[sp.mem_id] = scores.get(sp.mem_id, 0.0) + val

    return scores


def pick_heads(
    attn: Tensor,
    *,
    heads_to_use: Optional[List[int]] = None,
) -> Tensor:
    """
    Select and average attention over chosen heads.

    Args:
        attn: [heads, prompt_len] or [layers, heads, prompt_len]
        heads_to_use: list of head indices to include (None = all heads)

    Returns:
        [prompt_len] — averaged attention over selected heads (and layers, if present)
    """
    if attn.dim() == 2:
        h = attn
        if heads_to_use is not None:
            h = h[torch.tensor(heads_to_use, dtype=torch.long, device=attn.device)]
        return h.mean(dim=0)

    if attn.dim() == 3:
        x = attn
        if heads_to_use is not None:
            x = x[:, torch.tensor(heads_to_use, dtype=torch.long, device=attn.device), :]
        return x.mean(dim=(0, 1))

    raise ValueError(f"Unexpected attn shape: {tuple(attn.shape)}")


def sample_steps(
    num_steps: int,
    *,
    collect_every: int = 1,
    max_collect_steps: Optional[int] = None,
) -> List[int]:
    """
    Return step indices at which to collect attention.

    Args:
        num_steps: total generation steps
        collect_every: collect every N-th step (1 = every step)
        max_collect_steps: maximum number of steps to collect (None = unlimited)

    Returns:
        List of step indices.
    """
    collect_every = max(int(collect_every), 1)
    steps = list(range(0, num_steps, collect_every))

    if max_collect_steps is not None:
        steps = steps[:int(max_collect_steps)]

    return steps


__all__ = [
    "MemSpan",
    "aggregate_mem_attention",
    "pick_heads",
    "sample_steps",
]
