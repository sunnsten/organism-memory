from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """LLM backend configuration."""
    type: str = "llama31"               # "llama31", "qwen3", "openai", "llama_cpp", ...
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # device_map: "cuda" — single GPU, no offload (faster, more VRAM required)
    # device_map: "auto" — HuggingFace decides GPU/CPU placement (may offload)
    device_map: str = "auto"            # "auto", "cuda", "cpu", {"": 0, ...}
    dtype: str = "bfloat16"            # "float16", "bfloat16", "float32"

    # Quantization (requires bitsandbytes)
    load_in_4bit: bool = False          # 4-bit NF4 quantization
    load_in_8bit: bool = False          # 8-bit LLM.int8()

    # OpenAI-compatible API (for type="openai")
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    strip_think: bool = True
    enable_thinking: bool = False       # send enable_thinking=True to vLLM (Qwen3 reasoning mode)
    thinking_budget: int = 1024         # max thinking tokens; 0 = no limit

    # llama.cpp backend (for type="llama_cpp")
    model_path: str = ""                # absolute path to .gguf file
    n_gpu_layers: int = -1              # -1 = all on GPU; 0 = CPU only; N = split
    n_ctx: int = 8192                   # context window

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
