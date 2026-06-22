from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

from .base import LMBackend, Message, EncodedText

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Qwen3VLBackend(LMBackend):
    """
    Backend for Qwen3-VL-8B-Instruct.

    Exposes the standard text LMBackend interface. Multimodal input is available
    via generate_multimodal() but is not yet wired into the Organism pipeline.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if dtype == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype: torch.dtype = torch.float16 if device.type == "cuda" else torch.float32
        else:
            torch_dtype = SUPPORTED_DTYPES.get(dtype, torch.float16)

        logger.info("[Qwen3VLBackend] Loading model %s", model_name)
        print(f"[Qwen3VLBackend] Loading model: {model_name} (dtype={torch_dtype}, device_map={device_map})")

        # AutoProcessor encapsulates tokenizer + image pipeline for VL models
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.debug("[Qwen3VLBackend] Processor loaded")
        print("[Qwen3VLBackend] Processor loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device_map == "cuda":
            use_device_map: Any = None
        elif device_map == "auto" and device.type == "cuda":
            use_device_map = "auto"
        else:
            use_device_map = None

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=use_device_map,
            trust_remote_code=True,
        )
        self.model = cast(nn.Module, base_model)

        if use_device_map is None:
            self.model.to(device)  # type: ignore[arg-type]

        final_device = next(self.model.parameters()).device
        if hasattr(self.model, "hf_device_map"):
            logger.debug("Model device map: %s", self.model.hf_device_map)
        logger.info("[Qwen3VLBackend] Model loaded on device: %s", final_device)
        print(f"[Qwen3VLBackend] Model loaded on device: {final_device}")

        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device  # type: ignore[union-attr]

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size  # type: ignore[return-value]

    def _get_attn_impl(self) -> str | None:
        cfg = self.model.config
        if hasattr(cfg, "_attn_implementation"):
            return getattr(cfg, "_attn_implementation")
        if hasattr(cfg, "attn_implementation"):
            return getattr(cfg, "attn_implementation")
        return None

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
    ) -> str:
        """Text-only generation. Safe to use as a regular LMBackend."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        do_sample = self.temperature > 0

        prompt = self.render_chat(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(  # type: ignore[call-arg]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
        )

        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        if len(generated_ids) > max_new_tokens:
            generated_ids = generated_ids[:max_new_tokens]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_multimodal(
        self,
        messages: List[Message],
        images: Optional[List[Any]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Multimodal generation stub. Currently ignores images and delegates to generate()."""
        return self.generate(messages, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        """Encode raw text into hidden states (and optionally attention / surprisal)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.device)

        old_impl = None
        if need_attn:
            old_impl = self._get_attn_impl()
            try:
                self.model.set_attn_implementation("eager")  # type: ignore[attr-defined]
            except Exception:
                old_impl = None

        outputs = self.model(  # type: ignore[call-arg]
            **inputs,
            output_hidden_states=True,
            output_attentions=need_attn,
        )

        if old_impl is not None:
            try:
                self.model.set_attn_implementation(old_impl)  # type: ignore[attr-defined]
            except Exception:
                pass

        hidden = outputs.hidden_states[-1]  # type: ignore[index]
        attn = None
        if need_attn:
            attn = outputs.attentions[-1] if getattr(outputs, "attentions", None) is not None else None  # type: ignore[index]
        mask = inputs["attention_mask"]

        surprisal = None
        if need_surprisal:
            input_ids = inputs["input_ids"]
            logits_next = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = logits_next.log_softmax(dim=-1)
            token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            k = min(16, token_logp.size(1))
            surprisal = -token_logp[:, -k:].mean(dim=-1)

        return EncodedText(
            hidden_states=hidden,
            attention_mask=mask,
            seq_len=hidden.shape[1],
            d_model=hidden.shape[2],
            attentions=attn,
            surprisal=surprisal,
        )
