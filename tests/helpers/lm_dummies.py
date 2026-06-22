from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from organism.backbone.base import LMBackend, Message, EncodedText


class FakeLM:
    """
    Alternative to DummyLMBackend with a different normalisation scheme.

    Uses simple division by 127.5 instead of mean/std normalisation.
    Returns [B=1, L=4, D] instead of [B=1, L=1, D].

    NOTE: FakeLM does not inherit from LMBackend; use DummyLMBackend for
    most tests — FakeLM is only for tests that specifically need this behaviour.
    """

    def __init__(self, hidden_size: int, device: torch.device) -> None:
        self.hidden_size = hidden_size
        self.device = device

    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        import hashlib
        import numpy as np

        hasher = hashlib.sha256(text.encode())
        hbytes = hasher.digest()
        L = 4
        D = self.hidden_size

        arr = np.frombuffer(hbytes, dtype=np.uint8)
        if arr.size < L * D:
            arr = arr.repeat((L * D + arr.size - 1) // arr.size)
        arr = arr[: L * D].astype("float32")
        arr = (arr / 127.5) - 1.0

        hidden = torch.from_numpy(arr.reshape(L, D)).unsqueeze(0).to(self.device)
        attn = None
        mask = torch.ones(1, L, dtype=torch.long, device=self.device)
        surprisal = None
        if need_surprisal:
            surprisal = torch.tensor([float(L)], device=self.device)

        seq_len = hidden.shape[1]
        d_model = hidden.shape[2]

        return EncodedText(
            hidden_states=hidden,
            attention_mask=mask,
            seq_len=seq_len,
            d_model=d_model,
            attentions=attn,
            surprisal=surprisal,
        )

    def encode(self, text: str) -> EncodedText:
        """Alias for encode_text (backward compatibility)."""
        return self.encode_text(text, need_attn=True, need_surprisal=False)


class BatchEncoding:
    """Minimal stand-in for transformers.BatchEncoding used in DummyTokenizer."""
    def __init__(self, data):
        self.data = data
        for key, value in data.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()


class DummyTokenizer:
    """
    Minimal tokenizer for SleepPipeline tests.

    Requirements:
    - encode() and decode() methods
    - callable (__call__) for transformers tokenizer API compatibility
    - pad() method
    """
    def __init__(self, pad_token_id: int = 0, padding_side: str = "right") -> None:
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def encode(self, text: str, **kwargs) -> list[int]:
        length = max(len(text) // 10, 1)
        return [1] * length

    def decode(self, ids: list[int], **kwargs) -> str:
        return "<dummy>"

    def __call__(self, texts, padding: str = "max_length", truncation: bool = True,
                 max_length: int = 512, return_tensors: str = "pt", **kwargs):
        """Callable interface for transformers tokenizer compatibility."""
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)
        token_lengths = [min(max(len(text) // 10, 1), max_length) for text in texts]
        max_seq_len = min(max(token_lengths), max_length)

        input_ids_list = []
        attention_mask_list = []

        for i, text in enumerate(texts):
            seq_len = token_lengths[i]
            ids = [1] * seq_len
            if padding == "max_length":
                ids = ids + [self.pad_token_id] * (max_seq_len - seq_len)
                mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
            else:
                mask = [1] * seq_len
            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        else:
            input_ids = input_ids_list
            attention_mask = attention_mask_list

        return BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

    def pad(
        self,
        encoded_inputs,
        padding: str = "max_length",
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        """pad() for transformers tokenizer API compatibility."""
        if isinstance(encoded_inputs, dict):
            encoded_inputs = [encoded_inputs]

        if not encoded_inputs:
            return BatchEncoding({
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
            })

        if max_length is None:
            if padding == "longest":
                max_length = max(len(enc.get("input_ids", [])) for enc in encoded_inputs)
            else:
                max_length = 512

        padded_input_ids = []
        padded_attention_mask = []

        for enc in encoded_inputs:
            input_ids = enc.get("input_ids", [])
            attention_mask = enc.get("attention_mask", [])

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            elif pad_len < 0:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

        if return_tensors == "pt" or return_tensors is None:
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)
        else:
            input_ids_tensor = padded_input_ids
            attention_mask_tensor = padded_attention_mask

        return BatchEncoding({
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        })


class DummyLMBackend(LMBackend):
    """
    Minimal LMBackend for tests:

    - generate() returns an echo of the last user message ("echo: <message>")
    - encode_text() returns deterministic hidden states derived from the input hash
    - hidden_size and device are available as properties
    """

    def __init__(self, hidden_size: int = 16, device: str | torch.device = "cpu") -> None:
        self._hidden_size = hidden_size
        self._device = torch.device(device)
        self.tokenizer = DummyTokenizer()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        """Render messages to plain text (dummy implementation for tests)."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role.capitalize()}: {content}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    def generate(self, messages: List[Message], max_new_tokens: int | None = None, temperature: float | None = None, model_override: str | None = None) -> str:
        """Return an echo of the last user message."""
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        """
        Return deterministic hidden states derived from a SHA-256 hash of the input.

        Returns:
            EncodedText with hidden_states [B=1, L=1, D], attention_mask [B=1, L=1],
            surprisal [1] (only when need_surprisal=True), attentions=None.
        """
        import hashlib
        import numpy as np

        hasher = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hasher.digest()

        arr = np.frombuffer(hash_bytes, dtype=np.uint8).astype("float32")
        if arr.size < self._hidden_size:
            reps = (self._hidden_size + arr.size - 1) // arr.size
            arr = np.tile(arr, reps)
        arr = arr[: self._hidden_size]
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        hidden = torch.from_numpy(arr).view(1, 1, self._hidden_size).to(self.device)
        mask = torch.ones(1, 1, dtype=torch.long, device=self.device)
        attn = None
        surprisal = None
        if need_surprisal:
            surprisal = torch.tensor([float(len(text))], device=self.device)

        seq_len = hidden.shape[1]
        d_model = hidden.shape[2]

        return EncodedText(
            hidden_states=hidden,
            attention_mask=mask,
            seq_len=seq_len,
            d_model=d_model,
            attentions=attn,
            surprisal=surprisal,
        )
