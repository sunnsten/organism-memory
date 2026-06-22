from __future__ import annotations

import json
import pytest
import httpx

from organism.backbone.openai_backend import OpenAIBackend


def _make_response(content: str) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": content}}]
    }
    return httpx.Response(200, json=body)


def test_generate_returns_text(respx_mock):
    respx_mock.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=_make_response("Hello there")
    )
    backend = OpenAIBackend("test-model", base_url="http://localhost:8080/v1", strip_think=False)
    result = backend.generate([{"role": "user", "content": "Hi"}])
    assert result == "Hello there"


def test_generate_strips_think_blocks(respx_mock):
    respx_mock.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=_make_response("<think>reasoning here</think>Final answer")
    )
    backend = OpenAIBackend("test-model", base_url="http://localhost:8080/v1", strip_think=True)
    result = backend.generate([{"role": "user", "content": "Q"}])
    assert result == "Final answer"
    assert "<think>" not in result


def test_generate_passes_max_tokens(respx_mock):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _make_response("ok")

    respx_mock.post("http://localhost:8080/v1/chat/completions").mock(side_effect=handler)
    backend = OpenAIBackend("test-model", base_url="http://localhost:8080/v1", max_new_tokens=128)
    backend.generate([{"role": "user", "content": "Q"}], max_new_tokens=256)
    assert captured["body"]["max_tokens"] == 256


def test_encode_text_raises():
    backend = OpenAIBackend("test-model", base_url="http://localhost:8080/v1")
    with pytest.raises(NotImplementedError):
        backend.encode_text("hello")


def test_device_is_cpu():
    import torch
    backend = OpenAIBackend("test-model", base_url="http://localhost:8080/v1")
    assert backend.device == torch.device("cpu")


def test_create_lm_backend_openai():
    """Factory creates OpenAIBackend for type='openai'."""
    from organism.backbone.config import BackboneConfig
    from organism.backbone import create_lm_backend
    from organism.backbone.openai_backend import OpenAIBackend

    class FakeConfig:
        base_model = BackboneConfig(
            type="openai",
            model_name="test-model",
            base_url="http://localhost:9999/v1",
        )

    backend = create_lm_backend(FakeConfig())  # type: ignore[arg-type]
    assert isinstance(backend, OpenAIBackend)
    assert backend.base_url == "http://localhost:9999/v1"
