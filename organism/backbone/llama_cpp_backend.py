from __future__ import annotations

import logging
import multiprocessing
import re
from typing import TYPE_CHECKING, Any, List

from .base import EncodedText, LMBackend, Message

if TYPE_CHECKING:
    import torch
    from multiprocessing.process import BaseProcess

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_INIT_TIMEOUT = 180  # seconds to wait for worker to load model (large GGUF on cold disk)
_GEN_TIMEOUT = 120   # seconds to wait for a single generation call


class LlamaCppBackend(LMBackend):
    """
    LLM backend using llama-cpp-python (quantized inference).

    The Llama model lives in a subprocess for process-level isolation.
    n_gpu_layers controls CPU/GPU split:
        -1  → all layers on GPU
         0  → CPU only
         N  → first N layers on GPU, rest on CPU
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        strip_think: bool = True,
        verbose: bool = False,
    ):
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._verbose = verbose
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.strip_think = strip_think
        # Populated by _start_worker from the ready handshake
        self._metadata: dict = {}
        self._conn: Any = None
        self._worker_proc: "BaseProcess | None" = None
        self._start_worker()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Spawn the Granite subprocess and wait for the ready handshake."""
        from organism.backbone.llama_cpp_worker import run as _worker_run

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_worker_run,
            args=(child_conn, self._model_path, self._n_gpu_layers, self._n_ctx, self._verbose),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # parent only uses parent_conn

        if not parent_conn.poll(timeout=_INIT_TIMEOUT):
            proc.kill()
            proc.join(timeout=5)
            raise RuntimeError(
                f"LlamaCppBackend: worker did not start within {_INIT_TIMEOUT}s"
            )

        init_msg = parent_conn.recv()
        if init_msg.get("status") != "ready":
            proc.kill()
            proc.join(timeout=5)
            raise RuntimeError(
                f"LlamaCppBackend: worker init error — {init_msg.get('error', init_msg)}"
            )

        self._metadata = init_msg.get("metadata", {})
        self._conn = parent_conn
        self._worker_proc = proc
        logger.info(
            "LlamaCppBackend: worker ready (pid=%d, model=%s)",
            proc.pid,
            self._model_path,
        )

    def _restart_worker(self) -> None:
        """Kill the crashed worker and start a fresh one."""
        logger.warning("LlamaCppBackend: restarting worker subprocess")
        if self._worker_proc is not None:
            if self._worker_proc.is_alive():
                self._worker_proc.kill()
            self._worker_proc.join(timeout=10)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._worker_proc = None
        self._start_worker()

    def _send_request(self, request: dict) -> dict:
        """
        Send a request to the worker and return its response.
        Restarts and retries once if the worker has died.
        """
        for attempt in range(2):
            try:
                if self._conn is None:
                    raise OSError("no worker connection")
                self._conn.send(request)
                if not self._conn.poll(timeout=_GEN_TIMEOUT):
                    logger.error(
                        "LlamaCppBackend: worker timeout after %ds (attempt %d), restarting",
                        _GEN_TIMEOUT, attempt + 1,
                    )
                    self._restart_worker()
                    continue
                return self._conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                if attempt == 0:
                    logger.warning(
                        "LlamaCppBackend: worker connection lost, restarting (attempt 1)"
                    )
                    self._restart_worker()
                else:
                    logger.error("LlamaCppBackend: worker failed after restart — returning error")
                    return {"status": "error", "error": "worker unavailable after restart"}
        return {"status": "error", "error": "worker unavailable"}

    def __del__(self) -> None:
        try:
            if self._conn is not None:
                self._conn.send(None)  # graceful shutdown
        except Exception:
            pass
        try:
            if self._worker_proc is not None:
                self._worker_proc.join(timeout=3)
                if self._worker_proc.is_alive():
                    self._worker_proc.kill()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # LMBackend properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> "torch.device":
        import torch
        return torch.device("cuda:0" if self._n_gpu_layers != 0 else "cpu")

    @property
    def hidden_size(self) -> int:
        for key in (
            "llama.embedding_length",
            "qwen2.embedding_length",
            "mistral.embedding_length",
            "gemma.embedding_length",
            "phi2.embedding_length",
        ):
            if key in self._metadata:
                return int(self._metadata[key])
        return 4096

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _inject_no_think(self, messages: List[Message]) -> List[Message]:
        """Prepend /no_think to disable Qwen3-style thinking blocks."""
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + " /no_think"}
        else:
            msgs.insert(0, {"role": "system", "content": "/no_think"})
        return msgs

    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
    ) -> str:
        if self.strip_think:
            messages = self._inject_no_think(messages)

        response = self._send_request({
            "type": "generate",
            "messages": messages,
            "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
        })

        if response.get("status") != "ok":
            logger.warning("LlamaCppBackend: generate failed: %s", response.get("error"))
            return ""

        text = response.get("text", "")
        if self.strip_think:
            text = _THINK_RE.sub("", text).strip()
        return text

    def render_chat(self, messages: List[Message], add_generation_prompt: bool = False) -> str:
        """Render messages using the model's chat template (from cached metadata)."""
        chat_template = self._metadata.get("tokenizer.chat_template", "")
        if chat_template:
            try:
                from jinja2.sandbox import SandboxedEnvironment

                env = SandboxedEnvironment()
                tmpl = env.from_string(chat_template)
                return tmpl.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token="",
                    eos_token="",
                )
            except Exception:
                pass
        # Fallback: ChatML / Qwen style
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        response = self._send_request({"type": "tokenize", "text": text})
        if response.get("status") == "ok":
            return response.get("count", 0)
        return len(text) // 4  # rough fallback

    def encode_text(
        self,
        text: str,
        *,
        need_attn: bool = True,
        need_surprisal: bool = False,
    ) -> EncodedText:
        raise NotImplementedError(
            "encode_text() is not yet implemented for LlamaCppBackend. "
            "Required for Research Layer (ConsolidationWorker, SSM)."
        )


__all__ = ["LlamaCppBackend"]
