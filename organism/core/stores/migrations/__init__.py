from __future__ import annotations

# Import migrations for easy access
try:
    from . import _002_add_vectorlite as add_vectorlite
except ImportError:
    add_vectorlite = None  # type: ignore

__all__ = ["add_vectorlite"]
