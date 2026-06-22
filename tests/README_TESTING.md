# Testing

## Quick start

```bash
# Fast (no model weights, DummyLMBackend)
python -m pytest tests/unit/ tests/integration/ -x -q

# With a real model
TEST_USE_REAL_MODEL=1 python -m pytest tests/integration/ -v -s
```

## Test layout

```
tests/
  unit/         — pure unit tests (no DB, no LM)
  integration/  — integration tests (real SQLite, DummyLMBackend by default)
  helpers/      — shared fixtures and fakes
  conftest.py   — top-level pytest fixtures
```

## Running tests

### Unit tests only (fastest)

```bash
python -m pytest tests/unit/ -x -q
```

### Unit + integration (recommended before commit)

```bash
python -m pytest tests/unit/ tests/integration/ -x -q
```

### With verbose output

```bash
python -m pytest tests/unit/ tests/integration/ -v
```

### Stop on first failure

```bash
python -m pytest tests/unit/ tests/integration/ -x
```

## Using a real LM backend

By default all tests use `DummyLMBackend` — a lightweight stub that returns canned responses with no model weights.

To test against a real model:

```bash
TEST_USE_REAL_MODEL=1 python -m pytest tests/integration/ -v -s
```

See [docs/testing-with-real-model.md](../docs/testing-with-real-model.md) for environment variables, VRAM requirements, and Docker usage.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_USE_REAL_MODEL` | `0` | Set to `1` to load real model weights |
| `TEST_MODEL_NAME` | `Qwen/Qwen3-8B` | HuggingFace model name |
| `TEST_MODEL_PATH` | — | Local model path (overrides `TEST_MODEL_NAME`) |
| `TEST_MODEL_TYPE` | `qwen3` | Backend type (`qwen3`, `llama31`, `llama_cpp`, `openai`) |
| `TEST_DEVICE` | auto | `cuda`, `cpu`, or `auto` |

## Pytest markers

| Marker | Description |
|--------|-------------|
| `unit` | Pure unit test, no external dependencies |
| `integration` | Requires SQLite; uses DummyLMBackend by default |
| `real_model` | Can run with a real LM backend when `TEST_USE_REAL_MODEL=1` |
| `sleep` | Sleep/LoRA tests requiring real model and GPU |

```bash
# Run only unit-marked tests
python -m pytest -m unit

# Run integration tests that support real model
TEST_USE_REAL_MODEL=1 python -m pytest -m "integration and real_model" -v -s
```

## Test helpers

`tests/helpers/` provides shared infrastructure:

- `DummyLMBackend` — stub LM backend, returns fixed responses
- `FakeLM`, `DummyTokenizer` — lower-level stubs for unit tests
- `DummyPersonalStoreBackend` — stub for personal store backend

```python
from tests.helpers import DummyLMBackend, FakeLM
```

## Windows

```powershell
$env:TEST_USE_REAL_MODEL = "1"
$env:TEST_MODEL_TYPE = "qwen3"
python -m pytest tests/integration/ -v -s
```
