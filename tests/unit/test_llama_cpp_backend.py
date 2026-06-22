from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_backend(
    n_gpu_layers: int = -1,
    strip_think: bool = True,
    metadata: dict | None = None,
):
    """
    Create a LlamaCppBackend without spawning a real subprocess.

    _start_worker() is patched out; _conn and _metadata are set to mocks
    so tests can control what the 'worker' sends back.
    """
    from organism.backbone.llama_cpp_backend import LlamaCppBackend

    with patch.object(LlamaCppBackend, "_start_worker"):
        backend = LlamaCppBackend(
            model_path="/fake/model.gguf",
            n_gpu_layers=n_gpu_layers,
            strip_think=strip_think,
        )

    backend._metadata = metadata or {}
    backend._conn = MagicMock()
    backend._worker_proc = MagicMock()
    backend._worker_proc.is_alive.return_value = True
    return backend


def _wire_generate(backend, text: str) -> None:
    """Make the mocked connection return a successful generate response."""
    backend._conn.poll.return_value = True
    backend._conn.recv.return_value = {"status": "ok", "text": text}


# ------------------------------------------------------------------
# generate()
# ------------------------------------------------------------------

def test_generate_returns_text():
    backend = _make_backend(strip_think=False)
    _wire_generate(backend, "hello world")  # worker already strips whitespace
    result = backend.generate([{"role": "user", "content": "hi"}])
    assert result == "hello world"


def test_generate_strips_think_blocks():
    backend = _make_backend(strip_think=True)
    _wire_generate(backend, "<think>reasoning</think>Final answer")
    result = backend.generate([{"role": "user", "content": "Q"}])
    assert result == "Final answer"
    assert "<think>" not in result


def test_generate_no_strip_think_when_disabled():
    backend = _make_backend(strip_think=False)
    raw = "<think>reasoning</think>Answer"
    _wire_generate(backend, raw)
    result = backend.generate([{"role": "user", "content": "Q"}])
    assert "<think>" in result


def test_generate_passes_max_tokens_override():
    backend = _make_backend()
    _wire_generate(backend, "ok")
    backend.generate([{"role": "user", "content": "Q"}], max_new_tokens=999)
    sent = backend._conn.send.call_args[0][0]
    assert sent["max_tokens"] == 999


def test_generate_passes_temperature_override():
    backend = _make_backend()
    _wire_generate(backend, "ok")
    backend.generate([{"role": "user", "content": "Q"}], temperature=0.0)
    sent = backend._conn.send.call_args[0][0]
    assert sent["temperature"] == 0.0


def test_generate_returns_empty_on_worker_error():
    backend = _make_backend(strip_think=False)
    backend._conn.poll.return_value = True
    backend._conn.recv.return_value = {"status": "error", "error": "boom"}
    result = backend.generate([{"role": "user", "content": "Q"}])
    assert result == ""


# ------------------------------------------------------------------
# Worker restart on broken pipe
# ------------------------------------------------------------------

def test_generate_restarts_worker_on_eof():
    backend = _make_backend(strip_think=False)
    # First send raises EOFError (worker died); after restart succeeds.
    backend._conn.send.side_effect = [EOFError, None]
    backend._conn.poll.return_value = True
    backend._conn.recv.return_value = {"status": "ok", "text": "recovered"}

    with patch.object(backend, "_restart_worker") as mock_restart:
        # After restart, poll/recv on the refreshed conn should work.
        # Since _restart_worker is mocked, _conn stays the same mock.
        # Reset side_effect so second send succeeds.
        mock_restart.side_effect = lambda: setattr(
            backend._conn, "send", MagicMock()
        )
        result = backend.generate([{"role": "user", "content": "Q"}])

    mock_restart.assert_called_once()


# ------------------------------------------------------------------
# render_chat()
# ------------------------------------------------------------------

def test_render_chat_falls_back_to_chatml():
    backend = _make_backend(metadata={})
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    rendered = backend.render_chat(messages)
    assert "system" in rendered
    assert "You are helpful." in rendered
    assert "Hello" in rendered


def test_render_chat_uses_jinja_template():
    template = (
        "{% for m in messages %}"
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    backend = _make_backend(metadata={"tokenizer.chat_template": template})
    messages = [{"role": "user", "content": "Hi"}]
    rendered = backend.render_chat(messages, add_generation_prompt=True)
    assert "<|im_start|>user" in rendered
    assert "Hi" in rendered
    assert "<|im_start|>assistant" in rendered


# ------------------------------------------------------------------
# count_tokens()
# ------------------------------------------------------------------

def test_count_tokens():
    backend = _make_backend()
    backend._conn.poll.return_value = True
    backend._conn.recv.return_value = {"status": "ok", "count": 5}
    count = backend.count_tokens("hello world")
    assert count == 5
    sent = backend._conn.send.call_args[0][0]
    assert sent["type"] == "tokenize"
    assert sent["text"] == "hello world"


def test_count_tokens_fallback_on_error():
    backend = _make_backend()
    backend._conn.poll.return_value = True
    backend._conn.recv.return_value = {"status": "error", "error": "oops"}
    # Falls back to len(text)//4
    count = backend.count_tokens("a" * 40)
    assert count == 10


# ------------------------------------------------------------------
# device / hidden_size
# ------------------------------------------------------------------

def test_device_cuda_when_gpu_layers():
    import torch
    backend = _make_backend(n_gpu_layers=-1)
    assert backend.device == torch.device("cuda:0")


def test_device_cpu_when_zero_gpu_layers():
    import torch
    backend = _make_backend(n_gpu_layers=0)
    assert backend.device == torch.device("cpu")


def test_hidden_size_from_llama_metadata():
    backend = _make_backend(metadata={"llama.embedding_length": "4096"})
    assert backend.hidden_size == 4096


def test_hidden_size_from_qwen_metadata():
    backend = _make_backend(metadata={"qwen2.embedding_length": "5120"})
    assert backend.hidden_size == 5120


def test_hidden_size_fallback():
    backend = _make_backend(metadata={})
    assert backend.hidden_size == 4096


# ------------------------------------------------------------------
# /no_think injection
# ------------------------------------------------------------------

def test_no_think_injected_into_existing_system_message():
    backend = _make_backend(strip_think=True)
    _wire_generate(backend, "answer")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Q"},
    ]
    backend.generate(messages)
    sent = backend._conn.send.call_args[0][0]
    assert sent["messages"][0]["content"] == "You are helpful. /no_think"


def test_no_think_inserts_system_when_absent():
    backend = _make_backend(strip_think=True)
    _wire_generate(backend, "answer")
    messages = [{"role": "user", "content": "Q"}]
    backend.generate(messages)
    sent = backend._conn.send.call_args[0][0]
    assert sent["messages"][0] == {"role": "system", "content": "/no_think"}
    assert sent["messages"][1]["role"] == "user"


def test_no_think_not_injected_when_strip_think_false():
    backend = _make_backend(strip_think=False)
    _wire_generate(backend, "answer")
    messages = [{"role": "user", "content": "Q"}]
    backend.generate(messages)
    sent = backend._conn.send.call_args[0][0]
    assert sent["messages"][0]["role"] == "user"  # no system injected


# ------------------------------------------------------------------
# encode_text() — must raise NotImplementedError (Research Layer TODO)
# ------------------------------------------------------------------

def test_encode_text_raises_not_implemented():
    backend = _make_backend()
    with pytest.raises(NotImplementedError, match="Research Layer"):
        backend.encode_text("hello")


# ------------------------------------------------------------------
# Factory — create_lm_backend
# ------------------------------------------------------------------

def test_create_lm_backend_llama_cpp():
    from organism.backbone.config import BackboneConfig
    from organism.backbone import create_lm_backend
    from organism.backbone.llama_cpp_backend import LlamaCppBackend

    class FakeConfig:
        base_model = BackboneConfig(
            type="llama_cpp",
            model_path="/fake/model.gguf",
            n_gpu_layers=32,
            n_ctx=4096,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            strip_think=True,
        )

    with patch.object(LlamaCppBackend, "_start_worker"):
        backend = create_lm_backend(FakeConfig())  # type: ignore[arg-type]

    assert isinstance(backend, LlamaCppBackend)
    assert backend._n_gpu_layers == 32
    assert backend.temperature == 0.0
    assert backend.strip_think is True
