# Organism

A local memory layer for AI agents. Organism gives any LLM persistent, searchable memory across sessions — without sending your data to external APIs.

## Architecture

Organism uses a four-tier memory system. Tiers 0–2 run on every request. Tier 3 is offline and disabled by default.

```
Tier 0  Working memory    Last-N messages from the database (no retrieval step)
Tier 1  RAG chunks        Verbatim turns → rag_chunks — FTS5 full-text + HNSW vector
Tier 2  Facts             LLM-extracted facts per session → facts + user_profile
Tier 3  Research          Consolidation + LoRA sleep  [experimental, off by default]
```

Online pipeline (every request):

```
User message
  ├─ Tier 0: recent messages (direct load)
  ├─ Tier 1: HybridRetriever on rag_chunks (FTS5 + HNSW)
  ├─ Tier 2: HybridRetriever on facts      (FTS5 + HNSW)
  └─ ContextAssembler → prompt → LLM → reply
       └─ async: FactExtractor daemon writes to facts table
```

Full architecture details: [docs/architecture.md](docs/architecture.md)

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA 12.x with ≥ 2.5 GB VRAM for the default 4-bit local model, ≥ 8 GB for full bfloat16
- Without a GPU: use [proxy mode](#proxy-mode) and point Organism at any OpenAI-compatible server

### Install

```bash
git clone https://github.com/sunnsten/organism-memory
cd organism-memory
pip install -r requirements.txt
cp organism_config.example.yaml organism_config.yaml
```

### Local GPU mode

```yaml
# organism_config.yaml
base_model:
  type: qwen35
  model_name: Qwen/Qwen3.5-4B
  device_map: cuda
  load_in_4bit: true      # ~2.5 GB VRAM
  max_new_tokens: 256

rag:
  embedder_enabled: true
  embedder_model: Qwen/Qwen3-Embedding-0.6B
```

```python
from organism import Organism
from organism.config import OrganismConfig

org = Organism.from_config(OrganismConfig.from_yaml("organism_config.yaml"))
reply = org.chat(user_id="alice", user_message="Hello!")
print(reply.reply)
```

### Proxy mode

Organism wraps an existing OpenAI-compatible endpoint and injects memory into every request — no local GPU needed for inference.

```bash
# Start your inference server (vLLM, Ollama, OpenAI, Anthropic, ...)
vllm serve Qwen/Qwen3.5-4B --port 8001

# Start Organism proxy
ORGANISM_CONFIG_PATH=organism_config.yaml uvicorn organism.proxy.server:app --port 9000
```

Point your client at `http://localhost:9000` instead of the original endpoint. Organism intercepts each request, retrieves relevant memory, injects it into the system prompt, and forwards to your backend. See [docs/claude-code-proxy.md](docs/claude-code-proxy.md) for the Claude Code / Anthropic variant.

### MCP server

```bash
python -m organism.mcp_server --config organism_config.yaml
```

Exposes six tools: `organism_chat`, `memory.store_event`, `memory.query`, `memory.remember`, `memory.reset`, `memory.metrics`. See [docs/mcp.md](docs/mcp.md) for full reference and configuration examples.

## Configuration

All options live in `organism_config.yaml`. The most common keys:

```yaml
base_model:
  type: qwen35              # qwen35 | vllm | llama_cpp | openai
  model_name: Qwen/Qwen3.5-4B
  device_map: cuda          # cuda | cpu | auto
  load_in_4bit: true        # 4-bit NF4 quantization via bitsandbytes
  max_new_tokens: 256

# Optional: dedicated small model for background fact extraction.
# When absent, fact extraction reuses base_model (shares GPU).
fact_llm:
  type: openai
  model_name: Qwen/Qwen3-4B
  base_url: http://localhost:8001/v1
  api_key: not-needed
  max_new_tokens: 512

rag:
  embedder_enabled: true
  embedder_model: Qwen/Qwen3-Embedding-0.6B
  embedder_base_url: ~      # leave blank for local; set for vLLM embedder endpoint

consolidation:
  enabled: false            # set true to activate Tier 3 (experimental)
```

Model selection via environment variables (useful in Docker):

| Variable | Effect |
|---|---|
| `ORGANISM_MODEL_TYPE` | Override `base_model.type` |
| `ORGANISM_MODEL_NAME` | Override `base_model.model_name` |

## Benchmarks

Evaluated on two long-term memory benchmarks using Qwen3.5-4B (4-bit) as both the chat and fact-extraction model.

### LongMemEval

Single-user, multi-session QA. 500 questions across 5 categories.

| Category | Score |
|---|---|
| Single-session | ~80% |
| Single-session preference | 6.7% |
| Multi-session | 41.4% |
| Temporal reasoning | 36.1% |
| **Overall** | **53.4%** |

### LoCoMo

Conversational long-term memory. Two-person dialogues across multiple sessions.

| Category | Score |
|---|---|
| Single-hop | ~42% |
| Multi-hop | 5.0% |
| Temporal | 9.8% |
| **Overall** | **28.0%** |

Both benchmarks run with `memory_mode: t2` (Tier 1 + Tier 2, no consolidation). See `scripts/bench/` for benchmark scripts and `runs/` for run artifacts.

## API Reference

### `Organism.chat`

```python
reply = org.chat(
    user_id="alice",
    user_message="What did I say about my job last week?",
    session_id=None,       # auto-generated if omitted
    system_prompt=None,    # prepended before memory context
    max_new_tokens=None,   # overrides config value
)
# reply.reply → str
```

### `Organism.retrieve_context`

Retrieval without LLM generation. Used by the proxy layer and for inspection.

```python
facts = org.retrieve_context(
    user_id="alice",
    query="job location",
    limit=8,
)
# → List[str]  (Tier 1 chunks + Tier 2 facts, ranked by relevance)
```

### `Organism.remember`

Explicit fact write, bypasses async extraction pipeline.

```python
count = org.remember(user_id="alice", text="Alice is a software engineer in Berlin.")
# → int  (number of new facts stored)
```

## Limitations

- **4B model ceiling**: LongMemEval ~54%, LoCoMo ~28%. Breaking these requires query expansion, session-aware retrieval, or a larger model (7B+).
- **Temporal reasoning**: The 4B model handles date arithmetic poorly. Temporal questions plateau at ~36% on LongMemEval.
- **Fact extraction is async**: Facts are written after the chat turn completes. A follow-up question in the same turn may not see newly extracted facts yet.
- **Single-session preference recall**: Currently 6.7% on LongMemEval — preference facts are over-deduplicated.
- **No streaming**: `Organism.chat` returns the full reply; streaming is not supported in direct mode (proxy mode inherits streaming from the backend).
- **SQLite only**: The store is a single SQLite file (`organism_data/organism.db`). Not designed for concurrent multi-process write access.

## Research Layer (Tier 3)

Tier 3 includes ConsolidationWorker, LoRA sleep fine-tuning, and SSM neural memory. The infrastructure is present in `organism/research/` but **not active by default**.

Enable consolidation:

```yaml
consolidation:
  enabled: true
  summary_temperature: 0.0
  summary_max_new_tokens: 1536
```

When enabled, `WriteService` writes `experience_blocks` after each turn, and `ConsolidationWorker` processes them into `memory_items` which are retrieved alongside Tier 2 facts. LoRA training is architecturally wired but requires a separate training run to activate.

## Database

Single SQLite file, auto-initialized on first run.

| Table | Tier | Description |
|---|---|---|
| `messages` | 0 | Raw conversation turns |
| `sessions` | — | Session metadata |
| `rag_chunks` | 1 | Verbatim chunks, FTS5 + HNSW |
| `facts` | 2 | LLM-extracted user facts |
| `user_profile` | 2 | Key→value profile (derived from facts) |
| `experience_blocks` | 3 | Raw experience feed (Tier 3, gated) |
| `memory_items` | 3 | Consolidated memories (Tier 3) |

## Tests

```bash
# Unit + integration (no GPU required)
python -m pytest tests/unit/ tests/integration/ -x -q

# With a real model (GPU required)
TEST_USE_REAL_MODEL=1 python -m pytest tests/sleep/ -m sleep -x -q
```

## Contributing

Open issues and PRs welcome. Please run the unit + integration suite before submitting:

```bash
python -m pytest tests/unit/ tests/integration/ -x -q
```
