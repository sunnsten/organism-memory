from __future__ import annotations


class OrganismError(Exception):
    """Base exception for all Organism errors."""


class UserNotFoundError(OrganismError):
    """User not found."""


class MemoryLoadError(OrganismError):
    """Failed to load memory."""


class MemorySaveError(OrganismError):
    """Failed to save memory."""


class SSMStateError(OrganismError):
    """SSM state operation failed."""


class ChatGenerationError(OrganismError):
    """Text generation failed."""


class MemoryRetrievalError(OrganismError):
    """Memory retrieval failed."""


__all__ = [
    "OrganismError",
    "UserNotFoundError",
    "MemoryLoadError",
    "MemorySaveError",
    "SSMStateError",
    "ChatGenerationError",
    "MemoryRetrievalError",
]