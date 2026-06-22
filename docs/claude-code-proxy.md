# Claude Code + Organism Proxy

Connect **Claude Code** (or any Anthropic/OpenAI client) to Organism's memory layer via a local proxy. Fact extraction and embeddings run locally — Anthropic tokens are spent only on Claude responses.

## How it works

```
┌──────────────────────────────────────────────────────────────┐
│  VS Code + Claude Code                                       │
│  ANTHROPIC_BASE_URL=http://localhost:9000                    │
│  ANTHROPIC_API_KEY=<your-proxy-key>  (not your Anthropic key)│
└───────────────────┬──────────────────────────────────────────┘
                    │  POST /v1/messages  (Anthropic Messages API)
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Organism Proxy  :9000                                       │
│                                                              │
│  1. Auth: proxy key → user_id                                │
│  2. Memory injection:                                        │
│       • Tier 1: FTS5 + HNSW search over rag_chunks          │
│         (embeddings: local vLLM on :8002)                    │
│       • Tier 2: search over facts                            │
│       Retrieved facts prepended to system prompt.            │
│  3. Forward → api.anthropic.com                             │
│     (real key from ORGANISM_ANTHROPIC_API_KEY env var)       │
│  4. Async: FactExtractor → local vLLM on :8001              │
│     (new facts written to SQLite in background)              │
└───────────────────┬──────────────────────────────────────────┘
                    │  x-api-key: sk-ant-...
                    ▼
             api.anthropic.com
```

**Key property:** fact extraction (Qwen3-4B) and embeddings (Qwen3-Embedding-0.6B) run locally via vLLM. No Anthropic tokens spent on memory infrastructure.

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 8+ GB |
| RAM | 8 GB | 16 GB |
| Python | 3.10+ | 3.11 |
| Docker + NVIDIA Container Toolkit | required for vLLM | — |
| Anthropic API key (`sk-ant-...`) | required | — |

VRAM split when both vLLM services are running:
- `vllm-fact` (Qwen3-4B): ~3.5 GB
- `vllm-embed` (Qwen3-Embedding-0.6B): ~1.5 GB
- Total: ~5 GB

---

## Step 1 — Start vLLM services

```bash
docker compose -f docker-compose.vllm-local.yml up -d

# With monitoring (Prometheus + Grafana):
docker compose -f docker-compose.vllm-local.yml -f monitoring/docker-compose.yml up -d
```

Wait for both services to be ready (model download takes 5–15 min on first run):

```bash
curl http://localhost:8001/health   # fact LLM
curl http://localhost:8002/health   # embedder
```

---

## Step 2 — Create a proxy API key

Each proxy key maps to one user and one persistent memory store.

```bash
python -c "
from organism.proxy.api_key_store import ApiKeyStore
store = ApiKeyStore('organism_data/organism.db')
key = store.create_key(user_id='alice', tenant_id='default')
print('Proxy key:', key)
"
# Proxy key: sk-organism-<hex>
```

Save this key — you will need it in Step 4. To list existing keys:

```bash
python -c "
from organism.proxy.api_key_store import ApiKeyStore
store = ApiKeyStore('organism_data/organism.db')
for k in store.list_keys(): print(k)
"
```

---

## Step 3 — Start the proxy

```bash
export ORGANISM_ANTHROPIC_API_KEY=sk-ant-<your-real-anthropic-key>
export ORGANISM_PROXY_CONFIG=organism_proxy_claude.yaml
export ORGANISM_CONFIG_PATH=organism_config_claude_code.yaml

uvicorn organism.proxy.server:app --host 0.0.0.0 --port 9000
```

Expected output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000
```

### Proxy config (`organism_proxy_claude.yaml`)

```yaml
proxy:
  port: 9000
  auth_mode: api_key          # validates sk-organism-... keys
  forward_mode: anthropic     # forwards to api.anthropic.com
  forward_url: https://api.anthropic.com
  memory_limit: 10            # max facts/chunks injected per request
  memory_max_tokens: 600      # max tokens of memory context
  strip_think: false
```

### Organism config (`organism_config_claude_code.yaml`)

```yaml
fact_llm:
  type: openai
  model_name: Qwen/Qwen3-4B
  base_url: http://localhost:8001/v1
  api_key: not-needed
  max_new_tokens: 512
  temperature: 0.2
  strip_think: true

rag:
  embedder_enabled: true
  embedder_model: Qwen/Qwen3-Embedding-0.6B
  embedder_base_url: http://localhost:8002/v1
  embedder_dim: 1024

memory_mode: t2
```

---

## Step 4 — Configure Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:9000",
    "ANTHROPIC_API_KEY": "<your-proxy-key>"
  }
}
```

Replace `<your-proxy-key>` with the `sk-organism-...` key from Step 2.

> **Important:** `ANTHROPIC_API_KEY` here is the *proxy* key (`sk-organism-...`), not your real Anthropic key. The real key (`sk-ant-...`) lives only in `ORGANISM_ANTHROPIC_API_KEY` on the server side and is never sent to the client.

Restart Claude Code after editing `settings.json`.

---

## Step 5 — Verify

Send a message in Claude Code. Proxy logs should show:

```
INFO  [proxy] auth: user_id=alice
INFO  [proxy] memory: injected 3 facts, 1 chunk (220 tokens overhead)
INFO  [proxy] forward: POST https://api.anthropic.com/v1/messages → 200
INFO  [proxy] usage: input=1840 output=312 cost=$0.0062
```

Direct test (replace with your proxy key):

```bash
curl http://localhost:9000/v1/messages \
  -H "x-api-key: sk-organism-..." \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"Hello! What do you know about me?"}]}'
```

---

## Monitoring

Start Grafana at `http://localhost:3000` (default credentials: `admin` / `organism`).

Prometheus metrics at `http://localhost:9000/metrics`:

| Metric | Description |
|---|---|
| `proxy_anthropic_cost_usd_total` | Accumulated cost in USD |
| `proxy_anthropic_input_tokens_total` | Input tokens by user and model |
| `proxy_anthropic_output_tokens_total` | Output tokens by user and model |
| `proxy_memory_overhead_tokens_bucket` | Memory injection size histogram |
| `proxy_requests_total` | Request count by HTTP status |
| `proxy_forward_latency_ms_bucket` | End-to-end latency histogram |

---

## Token economics

```
Anthropic tokens per request = X (conversation) + 100–600 (memory overhead)
Fact extraction              = 0 Anthropic tokens  ← local Qwen3-4B
Embeddings                   = 0 Anthropic tokens  ← local Qwen3-Embedding
```

Memory overhead often replaces repeated context re-explanation, which can reduce net token spend.

---

## Troubleshooting

**`401 Unauthorized` in proxy logs**

Claude Code is sending the wrong key. Verify the key in `settings.json` starts with `sk-organism-` and matches what `list_keys()` returns.

**`502 Bad Gateway` when forwarding**

The proxy cannot reach `api.anthropic.com`. Check that `ORGANISM_ANTHROPIC_API_KEY` is set and starts with `sk-ant-`.

**vLLM services not responding**

```bash
docker compose -f docker-compose.vllm-local.yml ps
docker logs organism-vllm-fact --tail 20
docker logs organism-vllm-embed --tail 20
```

Status `starting` means the model is still loading — wait 1–3 minutes.

**Facts not being injected (memory = 0)**

Facts are extracted asynchronously. After 2–3 conversations, check:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('organism_data/organism.db')
print('facts:', conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0])
print('chunks:', conn.execute('SELECT COUNT(*) FROM rag_chunks').fetchone()[0])
"
```

**`memory_max_tokens` too large**

Reduce in `organism_proxy_claude.yaml`:
```yaml
proxy:
  memory_max_tokens: 300
```

---

## Related files

| File | Description |
|---|---|
| `organism_proxy_claude.yaml` | Proxy config (port, auth, limits) |
| `organism_config_claude_code.yaml` | Organism config (fact_llm, embedder) |
| `docker-compose.vllm-local.yml` | Docker Compose for vllm-fact + vllm-embed |
| `organism/proxy/server.py` | Proxy entry point, Prometheus metrics |
| `organism/proxy/anthropic_router.py` | Forwarding + token/cost accounting |
| `organism/proxy/api_key_store.py` | Proxy key management |
| `docs/setup.md` | Full setup guide |
| `docs/architecture.md` | Memory architecture |
