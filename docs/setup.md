# Setup Guide

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| CUDA 12.x + ≥ 2.5 GB VRAM | For local GPU mode. Not required for proxy/API mode. |
| Docker + NVIDIA Container Toolkit | For Docker deployment. [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

## Install

```bash
git clone https://github.com/yourorg/organism
cd organism
pip install -r requirements.txt
pip install -e .
```

---

## Mode A — Local GPU (HuggingFace Transformers)

Organism loads the LLM directly into GPU memory. No external services needed.

Copy the example config and start the server:

```bash
cp organism_config.example.yaml organism_config.yaml
# Edit organism_config.yaml: set model_name, device_map, load_in_4bit
organism-server --host 0.0.0.0 --port 8000
```

On first start, HuggingFace downloads model weights automatically (~8 GB for Qwen3.5-4B, ~1.2 GB for the embedder). Set `HF_HOME` to control where weights are cached:

```bash
export HF_HOME=/path/to/hf/cache
```

Config for 4-bit quantization (~2.5 GB VRAM):

```yaml
base_model:
  type: qwen35
  model_name: Qwen/Qwen3.5-4B
  device_map: cuda
  load_in_4bit: true
  max_new_tokens: 256
```

---

## Mode B — vLLM Backend

Run vLLM as the inference server; Organism handles memory only.

```bash
# Start vLLM (Linux / WSL2)
vllm serve Qwen/Qwen3.5-4B --port 8001 --dtype bfloat16

# Start Organism
ORGANISM_CONFIG_PATH=organism_config.yaml organism-server --host 0.0.0.0 --port 8000
```

Config (`organism_config.yaml`):

```yaml
base_model:
  type: openai
  model_name: Qwen/Qwen3.5-4B
  base_url: http://localhost:8001/v1
  api_key: not-needed
  max_new_tokens: 512

rag:
  embedder_enabled: true
  embedder_model: Qwen/Qwen3-Embedding-0.6B
```

---

## Mode C — Cloud API (OpenAI / Anthropic)

Organism handles memory locally; the LLM runs in the cloud.

```yaml
# OpenAI
base_model:
  type: openai
  model_name: gpt-4.1
  base_url: https://api.openai.com/v1
  api_key: sk-...
  max_new_tokens: 1024

# Anthropic
base_model:
  type: openai          # Anthropic supports the OpenAI-compatible API
  model_name: claude-sonnet-4-6
  base_url: https://api.anthropic.com/v1
  api_key: sk-ant-...
  max_new_tokens: 1024
```

To avoid sending fact-extraction calls to the cloud API, add a local `fact_llm`:

```yaml
fact_llm:
  type: openai
  model_name: Qwen/Qwen3-4B
  base_url: http://localhost:8001/v1   # local vLLM
  api_key: not-needed
  max_new_tokens: 512
  temperature: 0.1
```

---

## Docker

Build and start the Qwen3 service (port 8001):

```bash
mkdir -p cache/hf organism_data weights runs
docker compose up -d organism-qwen3
docker compose logs -f organism-qwen3
```

Service is ready when logs show `Application startup complete.`

```bash
# Health check
curl http://localhost:8001/health
# {"status":"ok"}
```

To override the model without editing the compose file:

```bash
ORGANISM_MODEL_TYPE=qwen35 ORGANISM_MODEL_NAME=Qwen/Qwen3.5-4B docker compose up -d organism-qwen3
```

Run tests inside Docker:

```bash
docker compose run --rm organism-tests
```

Run tests with a real model:

```bash
docker compose run --rm \
  -e TEST_USE_REAL_MODEL=1 \
  -e TEST_MODEL_TYPE=qwen35 \
  -e TEST_MODEL_NAME=Qwen/Qwen3.5-4B \
  organism-tests pytest -m real_model -v
```

---

## Proxy Mode (Claude Code / OpenAI client pass-through)

See [claude-code-proxy.md](claude-code-proxy.md) for the full guide on running Organism as an Anthropic-compatible proxy that injects memory into every request.

---

## Benchmarks

Organism ships with benchmark runners for [LongMemEval](https://github.com/xiaowu0162/long-mem-eval) and LoCoMo.

### LongMemEval

```bash
# Direct mode (loads Organism in-process)
python scripts/bench/longmemeval.py --config organism_config.yaml --limit 50 --workers 1

# HTTP mode (against a running organism-server)
ORGANISM_CHAT_RATE_LIMIT=1000/minute organism-server --port 8000 &
python scripts/bench/longmemeval.py --api-url http://localhost:8000 --limit 50
```

### LoCoMo

```bash
python scripts/bench/locomo.py --config organism_config.yaml
```

Results are saved to `runs/`.

| Benchmark | Score (Qwen3.5-4B, 4-bit) |
|---|---|
| LongMemEval overall | 53.4% |
| LoCoMo overall | 28.0% |

---

## Monitoring

Start Prometheus + Grafana:

```bash
docker compose -f monitoring/docker-compose.yml up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (default: admin / organism)
```

Key metrics exported at `/metrics`:

| Metric | Description |
|---|---|
| `organism_chat_latency_ms` | `/chat` response time |
| `organism_facts_extracted_total` | Facts written to the database |
| `organism_fact_extraction_latency_ms` | FactExtractor latency |
| `organism_rag_retrieval_latency_ms` | Retrieval latency |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ORGANISM_CONFIG_PATH` | `organism_config.yaml` | Path to config file |
| `ORGANISM_MODEL_TYPE` | from config | Override `base_model.type` |
| `ORGANISM_MODEL_NAME` | from config | Override `base_model.model_name` |
| `ORGANISM_CHAT_RATE_LIMIT` | `10/minute` | Rate limit for `/chat` endpoint |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache directory |
| `HF_TOKEN` | — | Required for gated models (e.g. Llama 3.1) |

---

## Troubleshooting

**`No module named 'organism'`**
```bash
pip install -e .
# or: export PYTHONPATH=.
```

**`CUDA out of memory`**
```yaml
base_model:
  load_in_4bit: true   # ~2.5 GB instead of ~8 GB
```

**`Connection refused` to vLLM**

vLLM is still loading. Wait for `Application startup complete.` in vLLM logs.

**`401 Unauthorized` when downloading Llama 3.1**

The model requires a HuggingFace token with accepted license:
```bash
export HF_TOKEN=hf_...
```

**Database not found / schema not initialized**

The database is created automatically on first start. If it fails:
```bash
mkdir -p organism_data
python -c "from organism import Organism; from organism.config import OrganismConfig; Organism.from_config(OrganismConfig()); print('OK')"
```

**Fact extraction is slow or not happening**

Facts are extracted asynchronously after each chat turn. After 2–3 turns, check the database:
```bash
python -c "
import sqlite3
conn = sqlite3.connect('organism_data/organism.db')
print('facts:', conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0])
print('chunks:', conn.execute('SELECT COUNT(*) FROM rag_chunks').fetchone()[0])
"
```
