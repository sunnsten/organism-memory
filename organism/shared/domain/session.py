from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Session:
    """A user chat session."""
    id: str                         # UUID
    tenant_id: str
    user_id: str
    started_at: int                 # unix timestamp
    ended_at: Optional[int] = None
    status: str = "active"          # active / closed
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


__all__ = ["Session"]
