from typing import Literal

SourceType = Literal["chat", "remember", "manual", "sleep", "fast_replay", "proxy"]

KindType = Literal[
    "interaction",
    "summary",
    "fact",
    "preference",
    "habit",
    "plan",
    "pattern",
    "explicit_fact",
    "general",
]

NamespaceType = Literal["personal", "task", "tool", "project"]

__all__ = [
    "SourceType",
    "KindType",
    "NamespaceType",
]
