from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from organism.backbone.base import LMBackend, Message
    from organism.core.exceptions import ChatGenerationError
    from organism.backbone.attention_trace import AttentionTrace

logger = logging.getLogger(__name__)


def extract_mem_spans_from_prompt(prompt: str, tokenizer) -> Dict[int, Tuple[int, int]]:
    """
    Extract token spans for each <MEM id=...> marker in the prompt.

    Uses the tokenizer's char_to_token mapping. Tokenizes with the same
    parameters as generate() to ensure index alignment.

    Returns:
        {mem_id: (start_token_idx, end_token_idx)}
    """
    mem_spans: Dict[int, Tuple[int, int]] = {}

    pattern = r'<MEM id=(\d+)>(.*?)</MEM>'
    matches = list(re.finditer(pattern, prompt, re.DOTALL))

    if not matches:
        return mem_spans

    try:
        encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors=None,
        )
    except (ValueError, TypeError, AttributeError):
        # Tokenizer does not support return_offsets_mapping (e.g. slow tokenizer)
        return {}

    offsets = encoding.get("offset_mapping", [])
    if not offsets:
        return {}

    # Normalize batch vs. single format
    if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2 and isinstance(offsets[0][0], int):
        pass  # already [(char_start, char_end), ...]
    else:
        offsets = offsets[0] if offsets else []

    for match in matches:
        mem_id = int(match.group(1))
        mem_start_char = match.start()
        mem_end_char = match.end()

        start_token = None
        end_token = None

        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_end > mem_start_char and char_start < mem_end_char:
                if start_token is None:
                    start_token = token_idx
                end_token = token_idx

        if start_token is not None and end_token is not None:
            mem_spans[mem_id] = (start_token, end_token)

    return mem_spans


def generate_reply(
    messages: List["Message"],
    lm: "LMBackend",
    max_new_tokens: int,
    user_id: str,
    collect_attention: bool = True,
    attention_trace_config: Optional[Any] = None,
) -> Tuple[str, Optional["AttentionTrace"]]:
    """
    Generate a reply from the LM with optional attention tracing.

    Returns:
        (reply_text, attention_trace) — attention_trace is None when
        collect_attention=False or the backend does not support tracing.

    Raises:
        ChatGenerationError: if generation fails.
    """
    from organism.core.exceptions import ChatGenerationError

    try:
        if collect_attention and hasattr(lm, 'generate_with_attention_trace'):
            prompt = lm.render_chat(messages, add_generation_prompt=True)
            tokenizer = getattr(lm, 'tokenizer', None)
            mem_spans = extract_mem_spans_from_prompt(prompt, tokenizer) if tokenizer is not None else None

            if attention_trace_config is not None:
                collect_every = attention_trace_config.collect_every
                max_collect_steps = attention_trace_config.max_collect_steps
                heads_to_use = attention_trace_config.heads_to_use
                normalize_by_span_len = attention_trace_config.normalize_by_span_len
            else:
                collect_every = 4
                max_collect_steps = 10
                heads_to_use = [0, 1]
                normalize_by_span_len = True

            reply_text, attention_trace = lm.generate_with_attention_trace(
                messages,
                max_new_tokens=max_new_tokens,
                mem_spans=mem_spans,
                collect_every=collect_every,
                max_collect_steps=max_collect_steps,
                heads_to_use=heads_to_use,
                normalize_by_span_len=normalize_by_span_len,
            )
            return reply_text, attention_trace
        else:
            reply_text = lm.generate(messages, max_new_tokens=max_new_tokens)
            return reply_text, None
    except Exception as e:
        logger.error("Chat generation failed for user %s: %s", user_id, e, exc_info=True)
        raise ChatGenerationError(f"Failed to generate reply: {e}") from e


__all__ = ["generate_reply"]
