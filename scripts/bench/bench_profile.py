from __future__ import annotations

import os
import time
from statistics import mean

from organism.config import OrganismConfig
from organism.core import Organism


def make_timed(stats: dict, key: str, fn):
    """Wrap fn so that each call records its duration into stats[key]."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        stats.setdefault(key, []).append(time.perf_counter() - t0)
        return result
    return wrapper


def apply_env_overrides(cfg: OrganismConfig) -> None:
    model_type = os.getenv("TEST_MODEL_TYPE")
    if model_type:
        cfg.base_model.type = model_type
    model_name = os.getenv("TEST_MODEL_NAME")
    if model_name:
        cfg.base_model.model_name = model_name
    test_device = os.getenv("TEST_DEVICE")
    if test_device:
        cfg.base_model.device_map = test_device.lower()
    elif not cfg.base_model.device_map:
        import torch
        if torch.cuda.is_available():
            cfg.base_model.device_map = "cuda"


def main() -> None:
    t0_global = time.perf_counter()
    print("=== bench_profile: start ===")

    cfg = OrganismConfig()
    cfg.base_model.device_map = "cuda"
    cfg.base_model.dtype = "bfloat16"
    apply_env_overrides(cfg)

    t_cfg = time.perf_counter()
    print(f"[step] Config ready in {t_cfg - t0_global:.2f}s")
    print(f"Using model: {cfg.base_model.type} / {cfg.base_model.model_name}")
    print(f"Device: {cfg.base_model.device_map}\n")

    print("[step] Initializing Organism...")
    t_org0 = time.perf_counter()
    org = Organism.from_config(cfg)
    print(f"[step] Organism init done in {time.perf_counter() - t_org0:.2f}s\n")

    # Warmup
    print("[step] Warmup chat...")
    t_warm0 = time.perf_counter()
    org.chat("bench_warmup", "warmup", max_new_tokens=64)
    print(f"[step] Warmup done in {time.perf_counter() - t_warm0:.2f}s\n")

    # --- Hook profiling points (v2 API) ---
    orch = org._orchestrator
    stats: dict[str, list[float]] = {}

    orig_generate = orch.lm.generate
    orig_retrieve = orch._memory.retrieval.retrieve
    orig_append   = orch._memory.write.append_event

    orch.lm.generate                = make_timed(stats, "generate", orig_generate)  # type: ignore[assignment]
    orch._memory.retrieval.retrieve = make_timed(stats, "retrieve", orig_retrieve)  # type: ignore[assignment]
    orch._memory.write.append_event = make_timed(stats, "write",    orig_append)    # type: ignore[assignment]

    # --- Profiled runs ---
    runs = 3
    print(f"[step] Running {runs} profiled chats...\n")
    for i in range(runs):
        t0 = time.perf_counter()
        reply = org.chat(f"bench{i}", f"iteration {i}: how fast are you?", max_new_tokens=256)
        dt = time.perf_counter() - t0
        stats.setdefault("total", []).append(dt)
        print(f"[step] Chat {i}: total={dt:.3f}s  reply_len={len(reply.reply)}")

    # --- Restore originals ---
    orch.lm.generate                = orig_generate  # type: ignore[assignment]
    orch._memory.retrieval.retrieve = orig_retrieve  # type: ignore[assignment]
    orch._memory.write.append_event = orig_append    # type: ignore[assignment]

    # --- Summary ---
    def avg(key: str) -> float:
        vals = stats.get(key, [])
        return float(mean(vals)) if vals else 0.0

    print(f"\n=== Averages over {runs} runs ===")
    for key in ["retrieve", "generate", "write", "total"]:
        print(f"{key:>10}: {avg(key):.3f} s")

    print("\nRaw stats:", {k: [round(x, 3) for x in v] for k, v in stats.items()})
    print(f"\n=== bench_profile: finished in {time.perf_counter() - t0_global:.2f}s ===")


if __name__ == "__main__":
    main()