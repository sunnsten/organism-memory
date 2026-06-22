from __future__ import annotations


def safe_truncate_unicode(text: str, max_length: int) -> str:
    """
    Safely truncate a Unicode string.

    In Python 3, slicing by index works correctly for most Unicode including
    emoji. Complex composed emoji may be split mid-sequence, but that is
    acceptable for text previews.

    Args:
        text: source string
        max_length: maximum character length

    Returns:
        String truncated to at most max_length characters.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length]
