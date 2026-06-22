# Benchmarking

Scripts for measuring performance and evaluating memory quality on standard datasets.

## Available scripts

| Script | Purpose |
|--------|---------|
| `bench/locomo.py` | LoCoMo benchmark — multi-session memory quality |
| `bench/longmemeval.py` | LongMemEval benchmark — long-term memory retrieval |
| `bench/compare_runs.py` | Compare two benchmark result files |
| `bench/test_locomo_scoring.py` | Unit tests for LoCoMo scoring logic |
| `bench/bench_profile.py` | Component-level profiling (retrieve, generate, write) |
| `smoke/pipeline_health.py` | End-to-end pipeline health check (no GPU needed) |

---

## Smoke test (no GPU)

Verifies the full pipeline is wired correctly using a dummy LM backend.
Runs in seconds; useful after refactors or environment setup.

```bash
python -m scripts.smoke.pipeline_health
```

With a real model (checks cross-session retrieval and context injection):

```bash
python -m scripts.smoke.pipeline_health --real-model --config organism_config.yaml
```

Stages checked:
1. DB schema — messages, sessions, rag_chunks, facts, user_profile
2. `chat()` → messages + RAG chunk written
3. `store_event()` → RAG chunk written directly
4. FTS search on rag_chunks
5. Vector search on rag_chunks (skipped if no embedder)
6. Cross-session retrieval via `query_memory()` (real model only)
7. Memory in LM context — recalled in a new chat turn (real model only)

---

## LoCoMo benchmark

Measures memory quality on the [LoCoMo dataset](https://github.com/snap-research/locomo)
(multi-session conversations, 1986 QA pairs across 10 personas).

**Download data first:**
```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
     -o data/locomo10.json
```

**Run against a live API server (recommended):**
```bash
python -m scripts.bench.locomo \
  --api-url http://localhost:8000 \
  --out-dir runs/
```

**Run in-process (loads model directly, no server needed):**
```bash
python -m scripts.bench.locomo \
  --config organism_config.yaml \
  --out-dir runs/
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--api-url URL` | — | Server URL (fast-replay via `/session/replay`) |
| `--config PATH` | — | Config YAML for in-process mode |
| `--out-dir DIR` | `runs/` | Directory for result JSON + summary |
| `--samples N` | 10 | Number of personas to evaluate (1–10) |
| `--limit N` | all | Max QA pairs per persona (quick smoke: `--limit 5`) |
| `--workers N` | 4 | Parallel worker threads |
| `--trace` | off | Log retrieved facts + prediction per QA pair |
| `--persona-id ID` | — | Run a single persona only (e.g. `conv-26`) |

---

## LongMemEval benchmark

Measures long-term memory retrieval on the [LongMemEval dataset](https://github.com/xiaowu0162/LongMemEval)
(500 instances, 6 categories).

**Download data first:**
```bash
mkdir -p data/longmemeval
# Follow instructions at https://github.com/xiaowu0162/LongMemEval
```

**Run against a live API server:**
```bash
python -m scripts.bench.longmemeval \
  --api-url http://localhost:8000 \
  --out-dir runs/
```

**Run in-process:**
```bash
python -m scripts.bench.longmemeval \
  --config organism_config.yaml \
  --out-dir runs/
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--api-url URL` | — | Server URL |
| `--config PATH` | — | Config YAML for in-process mode |
| `--out-dir DIR` | `runs/` | Directory for result JSON + summary |
| `--limit N` | all | Max instances (quick smoke: `--limit 20`) |
| `--workers N` | 8 | Parallel worker threads |
| `--no-fast-replay` | off | Use slow LLM replay instead of direct DB writes |
| `--temperature T` | config | Override model temperature (e.g. `0.0` for stability) |

---

## Comparing runs

```bash
python -m scripts.bench.compare_runs runs/locomo_run_a.json runs/locomo_run_b.json
```

---

## bench_profile.py

Profiles the three main pipeline components over several chat turns.

```bash
# Default model from config
python -m scripts.bench.bench_profile

# Specific model via env vars
TEST_MODEL_TYPE=qwen3 TEST_MODEL_NAME=Qwen/Qwen3-8B \
python -m scripts.bench.bench_profile
```

Output: average latency (seconds) for `retrieve`, `generate`, `write`, and `total`.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODEL_TYPE` | from config | Backend type (`qwen3`, `llama31`, `llama_cpp`, `openai`) |
| `TEST_MODEL_NAME` | from config | HuggingFace model name |
| `TEST_DEVICE` | `cuda` | Device (`cuda`, `cpu`) |

---

## Running via Docker Compose

```bash
# Start API server
docker compose up -d organism

# Smoke test (no GPU)
docker compose run --rm organism python -m scripts.smoke.pipeline_health

# LoCoMo
docker compose run --rm organism \
  python -m scripts.bench.locomo --api-url http://organism:8000 --out-dir runs/

# LongMemEval
docker compose run --rm organism \
  python -m scripts.bench.longmemeval --api-url http://organism:8000 --out-dir runs/
```
