from __future__ import annotations

from typing import Optional
from .experience_block import ExperienceBlock


def get_text_preview(block: ExperienceBlock) -> Optional[str]:
    """
    Extract a text preview from an ExperienceBlock.

    Priority order:
    1. metadata["text_preview"] — preview without the system prompt
    2. summary_preview — short preview for debug and FTS
    3. input_text — final fallback

    Returns None only if all three are absent/empty.
    """
    if block.metadata and "text_preview" in block.metadata:
        text_preview = block.metadata.get("text_preview")
        if text_preview:
            return text_preview

    if block.summary_preview:
        return block.summary_preview

    if block.input_text:
        return block.input_text

    return None


__all__ = [
    "get_text_preview",
]
