# Eval Module

Scenario-based evaluation for the Organism memory pipeline.

## Directory layout

```
organism/eval/
├── __init__.py
├── __main__.py          # python -m organism.eval entry point
├── cli.py               # CLI: --scenario / --scenario-dir, --config, --modes
├── scenarios/           # JSON scenario files
│   ├── T1_fact_smoke.json          # R-layer: basic store → retrieve (4 steps)
│   ├── T1_fact_long.json           # R-layer: 10 noise events after fact store (14 steps)
│   ├── T1_fact_chat.json           # C-layer: natural chat turns, must_include on LLM reply
│   ├── T2_cross_session.json       # R-layer: fact survives session_break (6 steps)
│   ├── T3_fact_update.json         # R-layer: newer event supersedes older in results (6 steps)
│   ├── T4_preference_recall.json   # R+C: 3 preference facts, assert_query × 3, 1 chat turn
│   ├── T5_negative_isolation.json  # R-layer: reset clears facts; cross-user isolation (6 steps)
│   └── T6_context_overflow.json    # R-layer: 20 noise events, early fact still retrievable (24 steps)
└── runner/
    ├── adapter.py       # EvalAdapter — thin wrapper exposing MCP handlers to the runner
    ├── artifact.py      # RunArtifact — JSON serialisation of a scenario run
    ├── context.py       # RunContext — per-run DB isolation
    ├── matchers.py      # check_step_expect — must_include / must_not_include
    └── run.py           # run_scenario — main loop

configs/
└── eval_r_layer.yaml    # Dummy-OpenAI config; runs R-layer without loading GPU weights

runs/                    # Output: JSON artifacts + SQLite DBs per run
```

---

## Two-layer architecture

| Layer | Requires LLM | How it works |
|-------|-------------|--------------|
| **R-layer** | No | `store_event` → `wait_indexing` → `assert_query` via MCP handlers; verifies Tier 1 (RAG chunks) + Tier 2 (facts) directly |
| **C-layer** | Yes | Natural chat turns (`role: user`) → `must_include` / `must_not_include` on the LLM reply |

R-layer scenarios run in seconds using `configs/eval_r_layer.yaml` (no GPU needed).
C-layer scenarios require a real model (local or API).

---

## Scenario schema

```json
{
  "test_id": "T1_fact_smoke",
  "description": "Human-readable description (optional)",
  "seed": 42,
  "steps": [...]
}
```

### Step types

#### Chat turn (C-layer)

```json
{
  "step_id": "S3",
  "role": "user",
  "content": "What is my favourite drink?",
  "expect": {
    "must_include": ["matcha"],
    "must_not_include": ["coffee"]
  }
}
```

#### Action step (R-layer)

```json
{ "step_id": "S0", "action": "reset_user" }
{ "step_id": "S0", "action": "reset_user", "user_id": "other_user" }

{ "step_id": "S1", "action": "store_event",
  "content": "My favourite drink is matcha latte with oat milk.",
  "source": "eval" }

{ "step_id": "S2", "action": "wait_indexing", "timeout_s": 5 }

{ "step_id": "S3", "action": "assert_query",
  "query": "favourite drink matcha oat",
  "expect": { "must_include": ["matcha", "oat"], "must_not_include": ["coffee"] } }

{ "step_id": "S4", "action": "session_break" }
```

### Action reference

| Action | What it does |
|--------|-------------|
| `reset_user` | Deletes all facts, messages, and RAG chunks for `user_id` (default: `eval_user`). Use `"user_id"` field to target another user. |
| `store_event` | Writes `content` to `messages` (Tier 0) and `rag_chunks` (Tier 1) synchronously; queues to FactExtractor (Tier 2) if available. |
| `wait_indexing` | Polls `facts.count` every 200 ms until stable or `timeout_s` elapsed; ensures async fact extraction finishes before `assert_query`. |
| `assert_query` | Calls `memory.query` MCP handler; checks combined text of returned facts + chunks against `expect`. |
| `session_break` | Starts a new session (`organism.start_session`); subsequent steps continue in the new session. |

### `expect` fields

| Key | Type | Description |
|-----|------|-------------|
| `must_include` | `string \| list[string]` | All items must appear in the response / query result text. |
| `must_not_include` | `string \| list[string]` | None of the items may appear. |

Keys starting with `_` (e.g. `_comment`) are ignored by the runner.

---

## Modes

| Mode | `enable_retrieve_db` | Description |
|------|---------------------|-------------|
| `B_memory_on` | `true` | Default. FTS5 + HNSW retrieval active. |
| `A_memory_off` | `false` | Retrieval disabled; tests LLM-only recall. |

Default when `--modes` is omitted: `B_memory_on`.

---

## Running evals

### All scenarios (R-layer, no GPU)

```bash
python -m organism.eval \
  --scenario-dir organism/eval/scenarios \
  --config configs/eval_r_layer.yaml \
  --modes B_memory_on
```

### Single scenario

```bash
python -m organism.eval \
  --scenario organism/eval/scenarios/T1_fact_smoke.json \
  --config configs/eval_r_layer.yaml \
  --modes B_memory_on
```

### With a real model (C-layer)

```bash
python -m organism.eval \
  --scenario organism/eval/scenarios/T1_fact_chat.json \
  --config organism_config.yaml \
  --modes B_memory_on
```

### Expected results without a real model

```
scenario               mode         pass  fail  total  success%
─────────────────────────────────────────────────────────────────
T1_fact_smoke          B_memory_on  4     0     4      100.0%
T1_fact_long           B_memory_on  14    0     14     100.0%
T1_fact_chat           B_memory_on  1     4     5      20.0%   ← chat turns need LLM
T2_cross_session       B_memory_on  6     0     6      100.0%
T3_fact_update         B_memory_on  6     0     6      100.0%
T4_preference_recall   B_memory_on  8     1     9      88.9%   ← 1 chat turn needs LLM
T5_negative_isolation  B_memory_on  6     0     6      100.0%
T6_context_overflow    B_memory_on  24    0     24     100.0%
```

---

## `configs/eval_r_layer.yaml`

```yaml
base_model:
  type: openai
  model_name: gpt-4o-mini
  base_url: http://localhost:9999/v1
  api_key: dummy
  max_new_tokens: 256
rag:
  embedder_enabled: false
  enable_retrieve_db: true
  top_k_fts: 5
consolidation:
  enabled: false
```

Uses the `openai` backend type so no GPU weights are loaded at startup.
The dummy URL is never contacted because R-layer scenarios contain no chat turns.
Embedder is disabled; retrieval falls back to FTS5-only.

---

## Run artifact

Each run produces `runs/{test_id}_{mode}_{timestamp}.json`:

```json
{
  "run_id": "2026-06-22T12:00:00Z",
  "test_id": "T1_fact_smoke",
  "mode": "B_memory_on",
  "turns": [
    {
      "step_index": 0,
      "step_id": "S0",
      "user": "reset_user:eval_user",
      "assistant": "{\"deleted\": {\"facts\": 0, \"messages\": 0, \"rag_chunks\": 0}}",
      "success": true,
      "expect_result": {}
    },
    {
      "step_index": 3,
      "step_id": "S3",
      "user": "assert_query:favourite drink matcha oat",
      "assistant": "{\"facts\": [...], \"chunks\": [...]}",
      "success": true,
      "expect_result": {
        "must_include": {"expected": ["matcha", "oat"], "found": true},
        "query_result": {"facts": 1, "chunks": 1}
      }
    }
  ],
  "metrics": {
    "total_turns": 4,
    "successful_turns": 4,
    "failed_turns": 0,
    "success_rate": 1.0
  }
}
```

The SQLite DB for each run is preserved at `runs/eval_{test_id}_{mode}_{timestamp}.db`
for post-run inspection.

---

## FTS5 query notes

`assert_query` uses FTS5 full-text search with implicit AND semantics: **every word in the
query must appear in the matched document.** A few rules to keep queries reliable:

- Use only words that literally appear in the stored `content`.
- Avoid stop words (`the`, `is`, `in`, `my`, etc.) — they are filtered automatically by
  `_sanitize_fts_query` in `chunk_store.py`.
- Prefer exact root forms; FTS5 does not stem (`remotely` ≠ `remote`).
- Two or three distinctive keywords are usually sufficient.
