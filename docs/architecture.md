# Organism Memory Architecture

## Overview

Organism uses a three-tier memory system. All tiers are online — active on every chat turn.

```
User message
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Tier 0 — Working Memory                                        │
│  Last-N messages from `messages` table (exact chat history)     │
└──────────────────────────────────────────────────────────────────┘
    │  retrieval (parallel)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Tier 1 — RAG Chunks                              [always-on]   │
│  Verbatim User+Assistant turns → `rag_chunks` table             │
│  Written synchronously on every turn (WriteService)             │
│  Retrieved via FTS5 + HNSW vector search (HybridRetriever)     │
└──────────────────────────────────────────────────────────────────┘
    │
┌──────────────────────────────────────────────────────────────────┐
│  Tier 2 — Facts                                   [always-on]   │
│  LLM-extracted facts per session → `facts` + `user_profile`    │
│  Written async by FactExtractor daemon (~250 tok/session)       │
│  Deduplication via cosine similarity (add_or_supersede)         │
│  ProfileUpdater extracts key→value pairs into user_profile      │
│  Retrieved via FTS5 + HNSW (same HybridRetriever pipeline)     │
└──────────────────────────────────────────────────────────────────┘
    │  assembled by ContextAssembler
    ▼
  LLM prompt  →  LLM response
```

---

## Tier 0 — Working Memory

**Table**: `messages`
**Written by**: `ChatOrchestrator.process_chat()`
**Read by**: `ChatOrchestrator` — the last-N turns are prepended directly to the LLM prompt.

No retrieval step; it is a direct slice of the message log.

---

## Tier 1 — RAG Chunks

**Table**: `rag_chunks`
**Written by**: `WriteService._write_chunks()` — synchronous, on every turn.
**Indexing**: FTS5 full-text search + HNSW dense vector index (via embedder).
**Retrieved by**: `FTSRetriever` + `HybridRetriever` → `RetrievalService`.
**Assembled by**: `ContextAssembler` → `context_block` in the prompt.

### Write path

```
ChatOrchestrator.process_chat()
  └─ WriteService.append_event()
       └─ _write_chunks()          # always runs
            └─ _split_by_rounds() → RoundChunk[]
                 └─ chunk_store.add_batch()
```

### Chunk format

Each chunk is one User+Assistant turn (or a sentence-boundary sub-chunk if the turn exceeds 1 200 chars). Every chunk carries:
- `round_id`, `round_boundary`, `round_part` / `round_parts_total`
- `event_date` (ISO date prefix `[YYYY-MM-DD]` prepended to content for temporal reasoning)

---

## Tier 2 — Facts

**Tables**: `facts`, `user_profile`
**Written by**: `FactExtractor` — async daemon thread, fires after each session.
**Token budget**: ~250 tokens per session (lightweight extraction prompt).
**Deduplication**: `add_or_supersede()` — cosine similarity threshold; superseded facts are soft-deleted (`valid_until` set).
**Profile extraction**: `ProfileUpdater` scans new facts for `key: value` patterns and upserts into `user_profile`.
**Retrieved by**: `FactRetriever` + `HybridRetriever` → `RetrievalService`.
**Assembled by**: `ContextAssembler` → `memory_block` in the prompt (highest priority, shown first).

### Write path

```
ChatOrchestrator.process_chat()
  └─ FactExtractor.submit(session_id, messages)   # non-blocking
       └─ [daemon thread] extract_and_store()
            ├─ LLM extraction prompt → fact strings
            ├─ fact_store.add_or_supersede()       # cosine dedup
            └─ ProfileUpdater.scan()               # key→value extraction
```

`FactExtractor` is wired automatically when using `Organism.from_config()` with an embedder and LM backend. Direct `Organism.__init__()` construction does not include it.

---

## Retrieval Pipeline

```
RetrievalService.retrieve(query)
  ├─ FactRetriever   → Tier 2 facts  (facts table, FTS5+HNSW)
  ├─ FTSRetriever    → Tier 1 chunks (rag_chunks, FTS5)
  └─ HybridRetriever → RRF fusion
       └─ ContextAssembler
            ├─ memory_block  = Tier 2 facts  (highest priority)
            └─ context_block = Tier 1 chunks
```

---

## Configuration

| Config key | Default | Effect |
|---|---|---|
| `memory_mode` | `"t2"` | `"t1"` = RAG chunks only; `"t2"` = RAG + Facts (FactExtractor enabled) |
| `rag.embedder_enabled` | `true` | Enables HNSW vector search for Tier 1 + Tier 2 |
| `rag.embedder_model` | `Qwen3-Embedding-0.6B` | Embedder model name |
| `fact_llm` | `null` | Dedicated LM backend for FactExtractor (falls back to `base_model`) |

---

## Proxy Layer

The proxy (`organism/proxy/`) provides an OpenAI-compatible HTTP API that wraps a local vLLM instance with memory injection:

```
Client (OpenAI or Anthropic format)
  └─ router.py
       ├─ Organism.retrieve_context(query)   # Tier 1+2 retrieval
       ├─ inject_memory() / inject_memory_anthropic()
       └─ Forward to vLLM backend
```

The proxy is stateless — it does not call `Organism.chat()` and does not write to any memory table.

---

## Database Tables

| Table | Tier | Written by | Read by |
|---|---|---|---|
| `messages` | 0 | ChatOrchestrator | ChatOrchestrator |
| `rag_chunks` | 1 | WriteService | FTSRetriever, HybridRetriever |
| `facts` | 2 | FactExtractor | FactRetriever, HybridRetriever |
| `user_profile` | 2 | ProfileUpdater | FactRetriever |
| `sessions` | — | ChatOrchestrator | ChatOrchestrator |
