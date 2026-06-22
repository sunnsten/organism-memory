from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LMBackend, Message, EncodedText
from .attention_trace import AttentionTrace
from organism.shared.utils.attention_utils import MemSpan, aggregate_mem_attention, pick_heads, sample_steps

logger = logging.getLogger(__name__)


SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Llama31Backend(LMBackend):
    """Backend for meta-llama/Meta-Llama-3.1-8B-Instruct and compatible models."""

    def __init__(
        self,
        model_name: str,
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

        logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, torch_dtype, device_map)
        print(f"[Llama31Backend] Loading model: {model_name} (dtype={torch_dtype}, device_map={device_map})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.debug("Tokenizer loaded")
        print("[Llama31Backend] Tokenizer loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device_map == "cuda":
            use_device_map = None
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
        logger.info("Model loaded on device: %s", final_device)
        print(f"[Llama31Backend] Model loaded on device: {final_device}")

        # Llama 3.1 may not have pad_token — fall back to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def count_tokens(self, text: str) -> int:
        """Count tokens using the loaded HuggingFace tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device  # type: ignore[union-attr]

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size  # type: ignore[return-value]

    def _get_attn_impl(self) -> str | None:
        """Return the current attention implementation name, or None if not detectable."""
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
        max_new_tokens = max_new_tokens or self.max_new_tokens
        actual_temperature = temperature if temperature is not None else self.temperature

        prompt = self.render_chat(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = actual_temperature > 0
        output = self.model.generate(  # type: ignore[call-arg]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=actual_temperature if do_sample else None,
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
    def generate_with_attention_trace(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        mem_spans: Optional[Dict[int, tuple[int, int]]] = None,
        collect_every: int = 1,
        max_collect_steps: Optional[int] = None,
        heads_to_use: Optional[List[int]] = None,
        normalize_by_span_len: bool = True,
    ) -> tuple[str, Optional[AttentionTrace]]:
        """
        Generate a response and collect an attention trace.

        Uses a manual generation loop with q_len=1 at each step, which is much
        cheaper on VRAM than a full L×L attention matrix.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        prompt = self.render_chat(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        input_ids = inputs["input_ids"]          # [1, L]
        attention_mask = inputs["attention_mask"] # [1, L]
        prompt_len = input_ids.shape[1]

        old_impl = self._get_attn_impl()
        try:
            self.model.set_attn_implementation("eager")  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Failed to set eager attention implementation, attention trace will be None")
            text = self.generate(messages, max_new_tokens)
            return text, None

        mem_span_list: List[MemSpan] = []
        if mem_spans:
            mem_attention_accum = {mem_id: 0.0 for mem_id in mem_spans.keys()}
            for mem_id, (start, end) in mem_spans.items():
                mem_span_list.append(MemSpan(mem_id=mem_id, start=start, end=end))
        else:
            mem_attention_accum = {}

        steps_to_collect = sample_steps(
            max_new_tokens - 1,
            collect_every=collect_every,
            max_collect_steps=max_collect_steps,
        )
        steps_to_collect_set = set(steps_to_collect)

        generated_ids = []
        all_attn_entropies = []

        # Prefill without attention to avoid an L×L matrix on the first step
        prefill_outputs = self.model(  # type: ignore[call-arg]
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
        )

        logits = prefill_outputs.logits[:, -1, :]
        past_key_values = prefill_outputs.past_key_values

        if self.temperature > 0:
            logits = logits / self.temperature
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.item())
        input_ids = next_token  # [1, 1]
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)], dim=1
        )
        del prefill_outputs

        try:
            for step in range(max_new_tokens - 1):
                collect_attention = step in steps_to_collect_set

                outputs = self.model(  # type: ignore[call-arg]
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=collect_attention,
                )

                if collect_attention:
                    if outputs.attentions is None or len(outputs.attentions) == 0:
                        logger.warning("Attention is None at step %d, skipping", step)
                    else:
                        last_layer_attn = outputs.attentions[-1]         # [B, H, q_len, kv_len]
                        last_token_attn = last_layer_attn[0, :, -1, :]  # [H, kv_len]

                        attn_weights = pick_heads(last_token_attn, heads_to_use=heads_to_use)  # [L]

                        current_seq_len = attn_weights.shape[0]
                        attn_over_prompt = attn_weights[:prompt_len] if current_seq_len >= prompt_len else attn_weights

                        attn_cpu = attn_over_prompt.cpu()
                        attn_norm = attn_cpu / (attn_cpu.sum() + 1e-10)
                        entropy = -(attn_norm * (attn_norm + 1e-10).log()).sum().item()
                        all_attn_entropies.append(entropy)

                        if mem_span_list:
                            step_scores = aggregate_mem_attention(
                                attn_cpu, mem_span_list, normalize_by_span_len=normalize_by_span_len
                            )
                            for mem_id, score in step_scores.items():
                                mem_attention_accum[mem_id] += score

                        del last_layer_attn, last_token_attn, attn_weights, attn_over_prompt, attn_cpu, attn_norm

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if self.temperature > 0:
                    logits = logits / self.temperature
                    if self.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated_ids.append(next_token.item())
                input_ids = next_token  # [1, 1]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)], dim=1
                )
                del outputs

                if next_token.item() == self.eos_token_id:
                    break
        finally:
            try:
                restore = old_impl if old_impl is not None else "sdpa"
                self.model.set_attn_implementation(restore)  # type: ignore[attr-defined]
            except Exception:
                pass

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        attention_focus = float(sum(all_attn_entropies) / len(all_attn_entropies)) if all_attn_entropies else 0.0

        mem_attention_scores: Dict[int, float] = {}
        if mem_attention_accum and all_attn_entropies:
            n = len(all_attn_entropies)
            mem_attention_scores = {mid: total / n for mid, total in mem_attention_accum.items()}

        return text, AttentionTrace(
            attention_focus=attention_focus,
            mem_attention_scores=mem_attention_scores,
            attn_entropy_mean=attention_focus,
        )

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
                if old_impl is not None:
                    self.model.set_attn_implementation("eager")  # type: ignore[attr-defined]
            except Exception:
                old_impl = None

        try:
            outputs = self.model(  # type: ignore[call-arg]
                **inputs,
                output_hidden_states=True,
                output_attentions=need_attn,
            )
        finally:
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
