from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestCountTokensContract:
    def test_base_class_default_uses_chars_over_4(self):
        """Base class count_tokens falls back to len(text) // 4."""
        from organism.backbone.base import LMBackend

        # Use a concrete mock that calls the real base method
        import torch as _torch

        class MinimalBackend(LMBackend):
            @property
            def device(self) -> _torch.device: return _torch.device("cpu")
            @property
            def hidden_size(self): return 0
            def generate(self, messages, max_new_tokens=None, temperature=None, model_override=None): return ""
            def encode_text(self, text, *, need_attn=True, need_surprisal=False):
                raise NotImplementedError
            def render_chat(self, messages, add_generation_prompt=False): return ""

        backend = MinimalBackend()
        assert backend.count_tokens("hello world") == len("hello world") // 4
        assert backend.count_tokens("") == 0
        assert backend.count_tokens("a" * 40) == 10

    def test_backend_count_tokens_uses_tokenizer(self):
        """When tokenizer is present, count_tokens uses encode() not chars//4."""
        from organism.backbone.llama31_backend import Llama31Backend

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        backend = MagicMock(spec=Llama31Backend)
        backend.tokenizer = mock_tokenizer

        result = Llama31Backend.count_tokens(backend, "some text")

        assert result == 5
        mock_tokenizer.encode.assert_called_once_with("some text", add_special_tokens=False)

    def test_qwen_backend_count_tokens_uses_tokenizer(self):
        """Qwen3Backend also uses tokenizer for count_tokens."""
        from organism.backbone.qwen3_backend import Qwen3Backend

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [10, 20]  # 2 tokens

        backend = MagicMock(spec=Qwen3Backend)
        backend.tokenizer = mock_tokenizer

        result = Qwen3Backend.count_tokens(backend, "hi")

        assert result == 2
        mock_tokenizer.encode.assert_called_once_with("hi", add_special_tokens=False)
