# MCP Server

Organism exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server so that
MCP-compatible clients (Claude Desktop, Claude Code, etc.) can read and write memory directly.

## Quick start

```bash
# Start the MCP server (stdio transport)
python -m organism.mcp_server --config organism_config.yaml

# Debug mode with structured JSON logs
python -m organism.mcp_server --config organism_config.yaml --log-level DEBUG --log-format json
```

The server reads `organism_config.yaml` from the current directory (or the path passed via `--config`
or the `ORGANISM_CONFIG` env var).

## 5-minute smoke test

Three calls to verify the full store → retrieve → reset cycle:

```
# 1. Store a fact
memory.store_event
  user_id  = "demo"
  content  = "I live in Amsterdam"
  source   = "cli"
→ { "event_id": 1, "queued_for_extraction": true }

# 2. Query memory
memory.query
  user_id  = "demo"
  query    = "Where do I live?"
→ {
    "facts":  [{ "id": 1, "content": "lives in Amsterdam", "score": 0.93, "source_type": "fact", "source_session_id": null }],
    "chunks": [{ "id": 4, "content": "I live in Amsterdam", "score": 0.81, "source_type": "rag_chunk", "source_session_id": "demo_mcp" }],
    "total_facts": 1, "total_chunks": 1
  }

# 3. Wipe memory
memory.reset
  user_id  = "demo"
  confirm  = true
→ { "deleted": { "facts": 1, "messages": 1, "rag_chunks": 1, "memory_items": 0 } }
```

## Available tools

### organism_chat

Send a message and get a reply. Stores the turn in memory and queues fact extraction.

```json
{
  "user_id":    "alice",
  "message":    "What did we discuss last week?",
  "session_id": "session_2026-06"
}
```

Returns the assistant's reply as plain text.

---

### memory.store_event

Store content into memory without generating a reply. Writes to the `messages` table
and RAG chunks (Tier 1). FactExtractor fires automatically if configured.

**Input:**
```json
{
  "user_id":    "alice",
  "content":    "Alice prefers dark mode and uses Python 3.13.",
  "session_id": "ide_session_1",
  "source":     "ide",
  "metadata":   { "file": "settings.py", "tags": ["preference"] }
}
```

**Output:**
```json
{ "event_id": 42, "queued_for_extraction": true }
```

`event_id` is the primary key in the `messages` table.

---

### memory.query

Retrieve relevant facts and chunks — **no LLM called**, pure retrieval.

**Input:**
```json
{
  "user_id":    "alice",
  "query":      "What language does Alice use?",
  "max_facts":  8,
  "max_chunks": 5
}
```

`max_facts`: 1–50, default 8. `max_chunks`: 1–20, default 5.

**Output:**
```json
{
  "facts": [
    {
      "id": 17,
      "content": "Alice uses Python 3.13",
      "score": 0.91,
      "source_type": "fact",
      "source_session_id": null
    }
  ],
  "chunks": [
    {
      "id": 84,
      "content": "Alice prefers dark mode and uses Python 3.13.",
      "score": 0.76,
      "source_type": "rag_chunk",
      "source_session_id": "ide_session_1"
    }
  ],
  "total_facts": 1,
  "total_chunks": 1
}
```

`id` can be used for follow-up operations. `source_session_id` traces where a chunk came from.

---

### memory.remember

Explicitly store a fact in long-term memory (bypasses LLM extraction pipeline).

```json
{ "user_id": "alice", "text": "Alice prefers dark mode." }
```

```json
{ "stored": true, "memory_id": 42 }
```

---

### memory.reset

Delete **all** memory for a user — facts, messages, RAG chunks, memory items, and HNSW index.
Requires `confirm: true` to prevent accidental wipes.

```json
{ "user_id": "alice", "confirm": true }
```

```json
{
  "deleted": {
    "facts": 12,
    "messages": 48,
    "rag_chunks": 96,
    "memory_items": 5
  }
}
```

---

### memory.metrics

Health snapshot of the memory store for the current server instance.

```json
{}
```

```json
{
  "total_messages": 1024,
  "total_facts": 87,
  "total_chunks": 412,
  "facts_extracted": 142,
  "facts_new": 87,
  "facts_confirmed": 50,
  "facts_extraction_errors": 5,
  "avg_retrieval_latency_ms": 12.3
}
```

`total_messages`, `total_facts`, `total_chunks` — реальные строки в БД.
`facts_extracted`, `facts_new`, `facts_confirmed`, `facts_extraction_errors` — счётчики текущего процесса (с момента запуска сервера).
`avg_retrieval_latency_ms` — средняя задержка retrieval по всем вызовам.

---

## Error responses

All tools return structured errors — never raw stacktraces:

```json
{ "error": { "type": "ValidationError", "message": "max_facts must be between 1 and 50" } }
{ "error": { "type": "NotFoundError",   "message": "Unknown tool: memory.foo" } }
{ "error": { "type": "InternalError",   "message": "An internal error occurred" } }
```

---

## Configuration for Claude Code

Add to your `.mcp.json` (project-level) or `~/.claude/mcp.json` (global):

```json
{
  "mcpServers": {
    "organism": {
      "command": "python",
      "args": ["-m", "organism.mcp_server", "--config", "/path/to/organism_config.yaml"],
      "env": {}
    }
  }
}
```

## Configuration for Claude Desktop

In `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "organism": {
      "command": "python",
      "args": ["-m", "organism.mcp_server", "--config", "C:/path/to/organism_config.yaml"]
    }
  }
}
```

## Logging

Every tool call produces a structured log line (set `--log-level INFO` to see them):

```
tool=memory.query user_id=alice duration_ms=12 status=ok facts=3 chunks=2
tool=memory.reset user_id=alice duration_ms=5 status=error type=ValidationError message=confirm must be true
```

Use `--log-format json` for machine-readable output.
