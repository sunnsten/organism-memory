import json
import pytest
from unittest.mock import MagicMock

from organism.mcp_server import _make_handlers


class FakeMCPOrganism:
    def chat(self, user_id, user_message, session_id=None, **kw):
        return MagicMock(reply=f"echo:{user_message}")

    def remember(self, user_id, text):
        return 42

    def list_memories(self, user_id, limit=50, category=None):
        return [{"id": 1, "content": "test fact", "category": "note"}]


def test_chat_handler_returns_reply():
    handlers = _make_handlers(FakeMCPOrganism())
    result = handlers["organism_chat"](user_id="u1", message="hello")
    assert "echo:hello" in result


def test_remember_handler_returns_id():
    handlers = _make_handlers(FakeMCPOrganism())
    result = handlers["memory.remember"](user_id="u1", text="I like tea")
    data = json.loads(result)
    assert data["stored"] is True
    assert data["memory_id"] == 42
