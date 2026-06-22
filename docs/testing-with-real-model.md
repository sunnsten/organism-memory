# Testing with a Real LM Backend

By default all tests run with `DummyLMBackend` (no model weights, instant). This doc explains how to run tests against a real model.

## Prerequisites

```bash
pip install transformers accelerate torch
```

A HuggingFace account with access to the model (or a local model path).

## Quick start

```bash
TEST_USE_REAL_MODEL=1 pytest tests/integration/test_real_model_backend.py -v -s
```

The fixture auto-detects CUDA. If a GPU is available it uses bfloat16; otherwise it falls back to CPU float32.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_USE_REAL_MODEL` | `0` | Set to `1` to load real weights |
| `TEST_MODEL_NAME` | `Qwen/Qwen3-8B` | HuggingFace model name |
| `TEST_MODEL_PATH` | — | Local path (overrides `TEST_MODEL_NAME`) |
| `TEST_MODEL_TYPE` | `qwen3` | Backend type (see table below) |
| `TEST_DEVICE` | auto (CUDA › CPU) | `cuda`, `cpu`, or `auto` |

## VRAM / RAM requirements

| Model | dtype | VRAM (GPU) | RAM (CPU) |
|-------|-------|-----------|---------|
| `Qwen/Qwen3.5-4B-Instruct` | bfloat16 | ~8 GB | ~8 GB |
| `Qwen/Qwen3-8B` | bfloat16 | ~16 GB | — |
| `meta-llama/Llama-3.1-8B-Instruct` | bfloat16 | ~16 GB | — |

## Examples

```bash
# Qwen3-8B on GPU (default)
TEST_USE_REAL_MODEL=1 \
pytest tests/integration/test_real_model_backend.py -v -s

# CPU only
TEST_USE_REAL_MODEL=1 \
TEST_DEVICE=cpu \
pytest tests/integration/test_real_model_backend.py -v -s

# Llama 3.1 8B
TEST_USE_REAL_MODEL=1 \
TEST_MODEL_TYPE=llama31 \
TEST_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
pytest tests/integration/test_real_model_backend.py -v -s

# Local model path
TEST_USE_REAL_MODEL=1 \
TEST_MODEL_PATH=/models/qwen3-8b \
TEST_MODEL_TYPE=qwen3 \
pytest tests/integration/test_real_model_backend.py -v -s

# llama.cpp backend (no GPU required)
TEST_USE_REAL_MODEL=1 \
TEST_MODEL_TYPE=llama_cpp \
TEST_MODEL_PATH=/models/qwen3-8b-q4.gguf \
pytest tests/integration/test_real_model_backend.py -v -s
```

## Running in Docker

```bash
# Build once
docker compose build organism-tests

# Drop into bash
docker compose run --rm organism-tests bash

# Inside the container — all TEST_* vars are pre-set
pytest tests/integration/test_real_model_backend.py -v -s

# Force CPU
docker compose run --rm -e TEST_DEVICE=cpu organism-tests \
  pytest tests/integration/test_real_model_backend.py -v -s
```

## CI

Set `TEST_USE_REAL_MODEL=0` (or omit it) for pipelines without GPU. Integration tests are automatically skipped when the flag is not set.

## Supported backends

| `TEST_MODEL_TYPE` | Class | Notes |
|-------------------|-------|-------|
| `qwen3` | `Qwen3Backend` | Qwen3-* family, thinking mode disabled |
| `qwen3_vl` | `Qwen3VLBackend` | Qwen3-VL multimodal models |
| `llama31` | `Llama31Backend` | Llama 3.1 family |
| `llama_cpp` | `LlamaCppBackend` | GGUF models via llama-cpp-python |
| `openai` | `OpenAIBackend` | Any OpenAI-compatible endpoint |
