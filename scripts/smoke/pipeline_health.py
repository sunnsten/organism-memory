from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

# Force UTF-8 output on Windows to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET} {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET} {msg}")

def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")


# ── Result tracking ───────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warned: List[str] = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        if passed:
            ok(f"{name}" + (f" — {detail}" if detail else ""))
            self.passed.append(name)
        else:
            fail(f"{name}" + (f" — {detail}" if detail else ""))
            self.failed.append(name)
        return passed

    def summary(self) -> None:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'─'*50}")
        print(f"{BOLD}Results: {len(self.passed)}/{total} checks passed{RESET}")
        if self.failed:
            print(f"{RED}Failed:{RESET} {', '.join(self.failed)}")
        if not self.failed:
            print(f"{GREEN}All checks passed — pipeline is healthy{RESET}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count(store, sql: str, params: tuple = ()) -> int:
    row = store.base.execute(sql, params).fetchone()
    if row is None:
        return 0
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


# ── Stage implementations ─────────────────────────────────────────────────────

def check_db_init(store, results: Results) -> bool:
    """Stage 1: DB schema created — all core tables exist."""
    header("Stage 1: DB initialisation")
    required_tables = [
        "messages", "sessions",
        "rag_chunks", "facts", "user_profile",
    ]
    ok_count = 0
    for tbl in required_tables:
        try:
            cnt = _count(store, f"SELECT COUNT(*) FROM {tbl}")
            ok(f"  table '{tbl}' exists ({cnt} rows)")
            ok_count += 1
        except Exception as e:
            fail(f"  table '{tbl}' error: {e}")

    return results.record("DB init", ok_count == len(required_tables),
                          f"{ok_count}/{len(required_tables)} tables")


def check_chat(org, store, user_id: str, session_id: str, results: Results) -> bool:
    """Stage 2: chat() saves messages and a RAG chunk."""
    header("Stage 2: chat() → messages + rag_chunks")

    msg_before  = _count(store, "SELECT COUNT(*) FROM messages   WHERE user_id=?", (user_id,))
    chunk_before = _count(store, "SELECT COUNT(*) FROM rag_chunks WHERE user_id=?", (user_id,))

    try:
        reply = org.chat(
            user_id=user_id,
            user_message="My favourite colour is deep blue and I love hiking in mountains.",
            session_id=session_id,
        )
        ok(f"  chat() returned: {reply.reply[:80]!r}")
    except Exception as e:
        fail(f"  chat() raised: {e}")
        traceback.print_exc()
        results.record("chat() message saved", False, str(e))
        results.record("chat() RAG chunk written", False)
        return False

    msg_after   = _count(store, "SELECT COUNT(*) FROM messages   WHERE user_id=?", (user_id,))
    chunk_after = _count(store, "SELECT COUNT(*) FROM rag_chunks WHERE user_id=?", (user_id,))

    r1 = results.record("chat() message saved", msg_after > msg_before,
                        f"{msg_before} → {msg_after} messages")
    r2 = results.record("chat() RAG chunk written", chunk_after > chunk_before,
                        f"{chunk_before} → {chunk_after} chunks")
    return r1 and r2


def check_store_event(org, store, user_id: str, results: Results) -> bool:
    """Stage 3: store_event() writes directly to rag_chunks (Tier 1)."""
    header("Stage 3: store_event() → rag_chunks")

    chunk_before = _count(store, "SELECT COUNT(*) FROM rag_chunks WHERE user_id=?", (user_id,))

    try:
        ev = org.store_event(
            user_id=user_id,
            content="The user prefers dark mode and uses Python 3.13.",
            source="smoke_test",
        )
        ok(f"  store_event() → event_id={ev.get('event_id')}")
    except Exception as e:
        fail(f"  store_event() raised: {e}")
        traceback.print_exc()
        return results.record("store_event → rag_chunks", False, str(e))

    chunk_after = _count(store, "SELECT COUNT(*) FROM rag_chunks WHERE user_id=?", (user_id,))
    return results.record("store_event → rag_chunks", chunk_after > chunk_before,
                          f"{chunk_before} → {chunk_after} chunks")


def check_fts_search(store, user_id: str, results: Results) -> bool:
    """Stage 4: FTS search finds stored chunk."""
    header("Stage 4: FTS search (rag_chunks)")
    try:
        hits = store.chunks.search_fts("default", "blue hiking colour", limit=5, user_id=user_id)
        passed = len(hits) > 0
        detail = f"found {len(hits)} results" if passed else "no results for 'blue hiking colour'"
        if passed:
            first = hits[0][0] if isinstance(hits[0], tuple) else hits[0]
            ok(f"  top result: {str(first.get('content', first))[:100]!r}")
        return results.record("FTS search", passed, detail)
    except Exception as e:
        fail(f"  FTS search raised: {e}")
        traceback.print_exc()
        return results.record("FTS search", False, str(e))


def check_vector_search(store, user_id: str, embedder, results: Results) -> bool:
    """Stage 5: Vector search via embedder (rag_chunks)."""
    header("Stage 5: Vector search (rag_chunks)")
    if embedder is None:
        warn("  embedder not configured — skipping vector search")
        results.warned.append("vector search (no embedder)")
        return True

    try:
        query_vec = embedder.embed("favourite colour hiking")
        ok(f"  embed() → shape={query_vec.shape} norm={float((query_vec**2).sum()**0.5):.3f}")
    except Exception as e:
        fail(f"  embed() raised: {e}")
        return results.record("vector search", False, str(e))

    try:
        hits = store.chunks.search_vector("default", user_id, query_vec, limit=5)
        passed = len(hits) > 0
        detail = f"found {len(hits)} results" if passed else "no results"
        if passed:
            first_item = hits[0][0] if isinstance(hits[0], tuple) else hits[0]
            score = hits[0][1] if isinstance(hits[0], tuple) else "?"
            ok(f"  top result score={score}: {str(first_item.get('content',''))[:80]!r}")
        return results.record("vector search", passed, detail)
    except Exception as e:
        fail(f"  vector search raised: {e}")
        traceback.print_exc()
        return results.record("vector search", False, str(e))


def check_cross_session_retrieval(org, user_id: str, results: Results) -> bool:
    """Stage 6: query_memory() retrieves facts/chunks from a previous session."""
    header("Stage 6: Cross-session retrieval")

    try:
        result = org.query_memory(user_id=user_id, query="blue hiking colour")
        chunks = result.get("chunks", [])
        facts  = result.get("facts", [])
        total  = len(chunks) + len(facts)
        passed = total > 0
        detail = f"{len(chunks)} chunks, {len(facts)} facts" if passed else "nothing retrieved"
        if passed:
            top = (chunks + facts)[0]
            ok(f"  top: {top['content'][:80]!r}")
        return results.record("cross-session retrieval", passed, detail)
    except Exception as e:
        fail(f"  query_memory() raised: {e}")
        traceback.print_exc()
        return results.record("cross-session retrieval", False, str(e))


def check_context_injection(org, user_id: str, results: Results) -> bool:
    """Stage 7: Retrieved memory appears in a new chat turn."""
    header("Stage 7: Memory in LM context")

    new_sid = org.start_session(user_id=user_id)
    try:
        reply = org.chat(
            user_id=user_id,
            user_message="What is my favourite colour?",
            session_id=new_sid,
        )
        answer = reply.reply.lower()
        ok(f"  reply: {reply.reply[:120]!r}")

        if "blue" in answer:
            return results.record("context injection", True,
                                  "model correctly recalled 'blue'")
        else:
            return results.record("context injection", False,
                                  "answer doesn't mention 'blue' — memory not injected")
    except Exception as e:
        fail(f"  chat() raised: {e}")
        traceback.print_exc()
        return results.record("context injection", False, str(e))
    finally:
        try:
            org.end_session(user_id=user_id, session_id=new_sid)
        except Exception:
            pass


# ── Build Organism ────────────────────────────────────────────────────────────

def build_organism(config_path: str, use_real_model: bool, db_path: Path):
    """Build Organism with either real or dummy LM backend."""
    from organism.config import OrganismConfig
    from organism.core.stores import UnifiedStore
    from organism.core.memory.service.memory_facade import MemoryFacade
    from organism.core.chat.orchestrator import ChatOrchestrator
    from organism.core.organism import Organism

    if config_path:
        cfg = OrganismConfig.from_yaml(config_path)
    else:
        cfg = OrganismConfig()

    store = UnifiedStore(db_path)
    embedder = None

    if use_real_model:
        print("  Loading real LM backend (this may take a moment)...")
        import organism.backbone as backbone
        lm = backbone.create_lm_backend(cfg)

        if getattr(cfg.rag, "embedder_enabled", True):
            try:
                from organism.core.embedding.qwen3_embedder import Qwen3Embedder
                embedder = Qwen3Embedder(model_name=cfg.rag.embedder_model)
                print(f"  Embedder: {cfg.rag.embedder_model}")
            except Exception as e:
                warn(f"  Embedder init failed: {e}")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tests.helpers.lm_dummies import DummyLMBackend
        lm = DummyLMBackend()
        print("  Using DummyLMBackend (fast, no GPU)")

    facade = MemoryFacade.from_store(store, tenant_id="default", embedder=embedder)
    orchestrator = ChatOrchestrator(memory_facade=facade, lm_backend=lm)
    org = Organism(chat_orchestrator=orchestrator, tenant_id="default")

    return org, store, embedder


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Organism pipeline health check")
    parser.add_argument("--config", default=None, help="Path to organism_config.yaml")
    parser.add_argument("--real-model", action="store_true",
                        help="Use real LM backend instead of DummyLMBackend")
    args = parser.parse_args()

    print(f"\n{BOLD}Organism Pipeline Health Check{RESET}")
    print("=" * 50)

    results = Results()
    tmpdir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmpdir = tmpdir_obj.name
    db_path = Path(tmpdir) / "health_check.db"
    user_id = "health_check_user"
    store = None

    try:
        header("Setup")
        try:
            org, store, embedder = build_organism(
                config_path=args.config,
                use_real_model=args.real_model,
                db_path=db_path,
            )
            ok("Organism built successfully")
        except Exception as e:
            fail(f"Failed to build Organism: {e}")
            traceback.print_exc()
            sys.exit(1)

        session_id = org.start_session(user_id=user_id)
        ok(f"Session started: {session_id}")

        check_db_init(store, results)
        chat_ok = check_chat(org, store, user_id, session_id, results)

        try:
            org.end_session(user_id=user_id, session_id=session_id)
        except Exception:
            pass

        if chat_ok:
            store_ok = check_store_event(org, store, user_id, results)
            if store_ok:
                check_fts_search(store, user_id, results)
                check_vector_search(store, user_id, embedder, results)
                if args.real_model:
                    check_cross_session_retrieval(org, user_id, results)
                    check_context_injection(org, user_id, results)
                else:
                    warn("  Skipping cross-session + context checks (use --real-model)")
                    results.warned.append("cross-session retrieval (no real model)")
                    results.warned.append("context injection (no real model)")
            else:
                warn("  Skipping retrieval checks (store_event failed)")
        else:
            warn("  Skipping retrieval checks (chat failed)")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        tmpdir_obj.cleanup()

    results.summary()
    sys.exit(0 if not results.failed else 1)


if __name__ == "__main__":
    main()
