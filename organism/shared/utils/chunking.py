from __future__ import annotations

import re
from typing import List


# Default separators in priority order (most to least preferable)
DEFAULT_SEPARATORS = [
    "\n\n",   # Paragraph breaks
    "\n",     # Line breaks
    ". ",     # Sentence boundaries
    "! ",     # Exclamation
    "? ",     # Question
    "; ",     # Semicolon
    ", ",     # Comma
    " ",      # Word boundary
    "",       # Character-level (last resort)
]


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    The algorithm tries to split on the highest-priority separator that
    produces chunks within the size limit. If no separator works,
    falls back to character-level splitting.

    Args:
        text: Input text to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Custom list of separators (default: paragraph -> char).

    Returns:
        List of text chunks. Each chunk is <= chunk_size characters
        (except when a single unsplittable segment exceeds the limit).
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    if separators is None:
        separators = DEFAULT_SEPARATORS

    return _recursive_split(text, chunk_size, chunk_overlap, separators)


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """Recursively split text using separators in priority order."""
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Try each separator in priority order
    for i, sep in enumerate(separators):
        if sep == "":
            # Character-level: hard split
            return _hard_split(text, chunk_size, chunk_overlap)

        if sep not in text:
            continue

        # Split by this separator
        parts = text.split(sep)

        # Merge parts into chunks that fit within chunk_size
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # If this single part is too large, recurse with remaining separators
                if len(part) > chunk_size and i + 1 < len(separators):
                    sub_chunks = _recursive_split(
                        part, chunk_size, chunk_overlap, separators[i + 1:]
                    )
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current and current.strip():
            chunks.append(current.strip())

        # Add overlap between chunks
        if chunk_overlap > 0 and len(chunks) > 1:
            chunks = _add_overlap(chunks, chunk_overlap)

        # Filter empty chunks
        return [c for c in chunks if c.strip()]

    # No separators worked, hard split
    return _hard_split(text, chunk_size, chunk_overlap)


def _hard_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split text by character count when no separators work."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if end < len(text) else end
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Add overlapping text from previous chunk to the start of each chunk."""
    if not chunks or overlap <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take the last `overlap` characters from previous chunk
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        # Find a word boundary in the prefix to avoid mid-word overlap
        space_idx = prefix.find(" ")
        if space_idx > 0:
            prefix = prefix[space_idx + 1:]
        merged = prefix + " " + chunks[i] if prefix else chunks[i]
        result.append(merged)

    return result


__all__ = ["split_text"]
