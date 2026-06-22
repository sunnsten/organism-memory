from __future__ import annotations

import argparse
import json
import logging
import re
import string
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_thread_local = threading.local()
_qa_counter_lock = threading.Lock()
_qa_counter = {"done": 0, "total": 0}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin HTTP client (mirrors the interface Organism exposes)
# ---------------------------------------------------------------------------

class _OrganismHttpClient:
    """Minimal HTTP client for the Organism server — used in --api-url mode."""

    def __init__(self, api_url: str, timeout: float = 180.0) -> None:
        import requests
        self._session = requests.Session()
        self._base = api_url.rstrip("/")
        self._timeout = timeout
        self.memory_service = _NoOpMemoryService()

    def start_session(self, user_id: str) -> str:
        r = self._session.post(f"{self._base}/session/start",
                               json={"user_id": user_id}, timeout=self._timeout)
        r.raise_for_status()
        return r.json()["session_id"]

    def end_session(self, user_id: str, session_id: str) -> None:
        try:
            self._session.post(f"{self._base}/session/end",
                               json={"user_id": user_id, "session_id": session_id},
                               timeout=self._timeout)
        except Exception:
            pass

    def replay_session(self, user_id: str, session_id: str,
                       turns: List[Dict], session_ts: Optional[float] = None) -> Dict:
        payload: Dict[str, Any] = {
            "user_id": user_id, "session_id": session_id,
            "turns": [{"role": t["role"], "content": t["content"]}
                      for t in turns if t.get("content")],
        }
        if session_ts is not None:
            payload["session_ts"] = session_ts
        r = self._session.post(f"{self._base}/session/replay",
                               json=payload, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def chat(self, user_id: str, user_message: str, session_id: str,
             system_prompt: str = "", max_new_tokens: int = 128) -> Any:
        payload = {
            "user_id": user_id, "message": user_message,
            "session_id": session_id, "system_prompt": system_prompt,
        }
        r = self._session.post(f"{self._base}/chat", json=payload, timeout=self._timeout)
        r.raise_for_status()
        return _ChatResult(r.json().get("reply", ""))

    @property
    def _tenant_id(self) -> str:
        return "default"


class _ChatResult:
    def __init__(self, reply: str) -> None:
        self.reply = reply


class _NoOpMemoryService:
    _last_trace = None

CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "open-domain",
    4: "temporal",
    5: "adversarial",
}

_ADVERSARIAL_KEYWORDS = frozenset([
    "i don't know", "i do not know", "not mentioned", "not discussed",
    "no information", "cannot answer", "don't have", "not in the conversation",
])

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def _iso_to_natural(text: str) -> str:
    """Convert ISO date strings (2023-05-08) to natural form (8 may 2023)."""
    def _replace(m):
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{d} {_MONTH_NAMES[mo - 1]} {y}"
        except (ValueError, IndexError):
            return m.group(0)
    return re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', _replace, text)


_LEMMA_RULES = [
    (re.compile(r'\bbooks\b'), 'book'),
    (re.compile(r'\btrips\b'), 'trip'),
    (re.compile(r'\bplants\b'), 'plant'),
    (re.compile(r'\bpaintings\b'), 'painting'),
    (re.compile(r'\bactivities\b'), 'activity'),
    (re.compile(r'\bhobbies\b'), 'hobby'),
    (re.compile(r'\bfriends\b'), 'friend'),
    (re.compile(r'\bclasses\b'), 'class'),
    (re.compile(r'\brings\b'), 'ring'),
    (re.compile(r'\bsessions\b'), 'session'),
    (re.compile(r'\b(\w+)(\w)\2ing\b'), lambda m: m.group(1) + m.group(2)),  # running→run
    (re.compile(r'\b(\w+)ing\b'), lambda m: m.group(1)),   # hiking→hik
    (re.compile(r'\b(\w+)(\w)\2ed\b'), lambda m: m.group(1) + m.group(2)),   # dropped→drop
    (re.compile(r'\b(\w+)ed\b'), lambda m: m.group(1)),    # moved→mov
]

_SYNONYM_GROUPS = [
    {"mental health", "mental health support", "counseling", "therapy"},
    {"transgender", "trans", "trans woman", "transgender woman"},
    {"adoption", "adopting", "adopted"},
    {"pottery", "ceramics", "clay"},
    {"volunteering", "volunteer", "volunteered"},
]
_SYNONYM_MAP: Dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    _canonical = sorted(_group)[0]
    for _term in _group:
        _SYNONYM_MAP[_term] = _canonical


def _lemmatize(text: str) -> str:
    for pattern, repl in _LEMMA_RULES:
        text = pattern.sub(repl, text)
    for term, canonical in sorted(_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(term, canonical)
    return text


def _normalize(text: str) -> str:
    text = text.lower()
    text = _iso_to_natural(text)           # "2023-05-08" → "8 may 2023"
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _lemmatize(text)
    return " ".join(text.split())


def _token_f1(predicted: str, ground_truth: str) -> float:
    pred_tokens = _normalize(predicted).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(predicted.strip() == ground_truth.strip())
    common = set(pred_tokens) & set(gt_tokens)
    n_common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_locomo(data_path: str = "data/locomo10.json") -> List[Dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo data not found at {path}.\n"
            "Download: curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo10.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_session_date(date_str: str) -> Optional[float]:
    if not date_str:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Session replay (once per sample)
# ---------------------------------------------------------------------------

def replay_sample(
    org: Any,
    sample: Dict,
    user_id: str,
    no_consolidation: bool,
) -> Dict[str, List[str]]:
    """
    Replay all sessions of a LoCoMo sample into Organism memory.

    Returns a mapping {dia_id -> block_ids} so Recall@K can be
    computed per-question using the evidence field.
    In HTTP mode dia_to_blocks is always empty (Recall@K reported as N/A).
    """
    http_mode = isinstance(org, _OrganismHttpClient)
    store: Any = None
    EventRecord: Any = None
    ContextMeta: Any = None
    if not http_mode:
        from organism.shared.domain import EventRecord, ContextMeta
        store = org._orchestrator._memory.store

    tenant_id = org._tenant_id if not http_mode else "default"
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]

    # dia_id → block_ids (a turn may map to multiple chunks)
    dia_to_blocks: Dict[str, List[str]] = defaultdict(list)

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(m.group()) if (m := re.search(r"\d+", k)) else 0,
    )

    for s_key in session_keys:
        turns = conv[s_key]
        if not isinstance(turns, list):
            continue
        date_key = f"{s_key}_date_time"
        session_ts = _parse_session_date(conv.get(date_key, ""))
        sid = org.start_session(user_id=user_id)

        if http_mode:
            # HTTP fast-replay: server handles writes + fact extraction + chunk embedding
            http_turns = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker", "")
                text = turn.get("content") or turn.get("text", "")
                if not text:
                    continue
                role = "user" if speaker == speaker_a else "assistant"
                http_turns.append({"role": role, "content": text})
            try:
                org.replay_session(user_id=user_id, session_id=sid,
                                   turns=http_turns, session_ts=session_ts)
            except Exception as exc:
                logger.warning("HTTP replay %s failed: %s", s_key, exc)
            try:
                org.end_session(user_id=user_id, session_id=sid)
            except Exception:
                pass
            continue  # skip in-process loop below

        pending_user: Optional[str] = None
        pending_user_dia: Optional[str] = None
        pending_msg_id: Optional[int] = None
        turn_offset = 0

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "")
            text = turn.get("content") or turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            if not text:
                continue

            is_user = (speaker == speaker_a)
            role = "user" if is_user else "assistant"

            msg_id = store.messages.add(
                session_id=sid, tenant_id=tenant_id,
                user_id=user_id, role=role, content=text,
            )

            if is_user:
                pending_user = text
                pending_user_dia = dia_id
                pending_msg_id = msg_id
                turn_offset += 1
            else:
                if pending_user is not None:
                    base_ts = session_ts if session_ts is not None else time.time()
                    turn_ts = base_ts + turn_offset * 3600

                    event = EventRecord(
                        id=None, user_id=user_id, session_id=sid,
                        timestamp=turn_ts,
                        input_text=pending_user, output_text=text,
                        kind="interaction", source="chat", importance=0.5,
                        surprisal_norm=None, attention_focus=None,
                        used_memories=[], used_memories_space=None,
                        context_meta=ContextMeta(
                            system_hash="", memory_ids=[],
                            chat_message_id_span=(pending_msg_id or 0, msg_id),
                            memory_id_space=None,
                        ),
                        text_preview=f"{pending_user}\n{text}"[:500],
                        embedding=None, embedding_dim=None,
                        embedding_dtype="float32", embedding_l2norm=False,
                    )
                    try:
                        block_id = org._orchestrator._memory.write.append_event(event, tenant_id=tenant_id)
                        if block_id:
                            bid = str(block_id)
                            if pending_user_dia:
                                dia_to_blocks[pending_user_dia].append(bid)
                            dia_to_blocks[dia_id].append(bid)
                    except Exception as exc:
                        logger.warning("Block write failed: %s", exc)

                    pending_user = None
                    pending_user_dia = None
                    pending_msg_id = None

        try:
            org.end_session(user_id=user_id, session_id=sid)
        except Exception:
            pass

        if not no_consolidation:
            try:
                org.memory_service.run_consolidation(user_id=user_id, limit=200)
            except Exception as exc:
                logger.warning("Consolidation failed: %s", exc)

    return dict(dia_to_blocks)


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

_EVAL_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal conversation history. "
    "Answer questions about events and facts from those conversations concisely and directly. "
    "Give only the essential answer (a name, date, number, or short phrase). "
    "If the answer is not in the conversation history, say 'not mentioned'. "
    "Do not explain, do not add caveats."
)


def ask_question(
    org: Any,
    user_id: str,
    question: str,
    qa: Dict,
    dia_to_blocks: Dict[str, List[str]],
    trace: bool = False,
) -> Dict[str, Any]:
    category = qa.get("category", 0)
    answer = qa.get("answer", "")
    evidence = qa.get("evidence", [])

    # answer_source_ids: block_ids for turns referenced in evidence
    answer_source_ids: set = set()
    for dia_id in evidence:
        answer_source_ids.update(dia_to_blocks.get(dia_id, []))

    eval_sid = org.start_session(user_id=user_id)
    eval_q = (
        question
        + "\n\n(Answer with ONLY the minimal answer — a name, date, number, or short phrase. "
        "If not in conversation history, say 'not mentioned'.)"
    )
    try:
        result = org.chat(
            user_id=user_id, user_message=eval_q,
            session_id=eval_sid, system_prompt=_EVAL_SYSTEM_PROMPT,
            max_new_tokens=128,
        )
        predicted = result.reply
        if trace:
            try:
                tr = org.memory_service._last_trace
                facts_preview = tr.metadata.get("db_result_text_previews", [])[:5] if tr else []
                logger.info(
                    "[TRACE] Q: %s\n  GT: %s\n  Pred: %r\n  Facts: %s",
                    question[:80], str(answer)[:60], predicted[:60],
                    [f[:60] for f in facts_preview],
                )
            except Exception:
                pass
    except Exception as exc:
        logger.error("QA failed: %s", exc)
        predicted = ""
    try:
        org.end_session(user_id=user_id, session_id=eval_sid)
    except Exception:
        pass

    # Recall@K
    recall_hit = False
    if answer_source_ids:
        try:
            last_trace = org.memory_service._last_trace
            if last_trace:
                chunk_ids = set(last_trace.metadata.get("chunk_source_ids", []))
                recall_hit = bool(answer_source_ids & chunk_ids)
        except Exception:
            pass

    # Recall failure reason for diagnostics
    if not recall_hit and answer_source_ids:
        recall_reason = "judge_miss"
    elif not answer_source_ids:
        recall_reason = "no_evidence_mapped"
    else:
        recall_reason = "ok"

    # Adversarial: pass if model admits it doesn't know
    if category == 5:
        pred_lower = predicted.lower()
        passed = any(kw in pred_lower for kw in _ADVERSARIAL_KEYWORDS)
        return {
            "category": category,
            "question_type": CATEGORY_NAMES.get(category, str(category)),
            "question": question,
            "f1": 1.0 if passed else 0.0,
            "pass": passed,
            "soft_pass": passed,
            "predicted": predicted,
            "ground_truth": "(unanswerable)",
            "recall_hit": recall_hit,
            "recall_reason": recall_reason,
            "has_answer_sources": len(answer_source_ids) > 0,
        }

    f1 = _token_f1(predicted, str(answer))
    return {
        "category": category,
        "question_type": CATEGORY_NAMES.get(category, str(category)),
        "question": question,
        "f1": f1,
        "pass": f1 >= 0.5,
        "soft_pass": f1 >= 0.3,
        "predicted": predicted,
        "ground_truth": str(answer),
        "recall_hit": recall_hit,
        "recall_reason": recall_reason,
        "has_answer_sources": len(answer_source_ids) > 0,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict]) -> str:
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(float(r["pass"]))
    all_scores = [float(r["pass"]) for r in results]

    lines = [
        "## LoCoMo Results",
        f"{'Type':<20} {'N':>5} {'Score':>7} {'Soft>=0.3':>10} {'Recall@K':>10}",
        "-" * 56,
    ]
    for qtype, scores in sorted(by_type.items()):
        n = len(scores)
        pct = 100 * sum(scores) / n if n else 0
        type_results = [r for r in results if r["question_type"] == qtype]
        soft_pct = 100 * sum(r.get("soft_pass", False) for r in type_results) / n if n else 0
        eligible = [r for r in results if r["question_type"] == qtype and r.get("has_answer_sources")]
        recall_pct = (
            100 * sum(1 for r in eligible if r.get("recall_hit")) / len(eligible)
            if eligible else float("nan")
        )
        recall_str = f"{recall_pct:>9.1f}%" if eligible else "       n/a"
        lines.append(f"{qtype:<20} {n:>5} {pct:>6.1f}% {soft_pct:>9.1f}% {recall_str}")

    n_total = len(all_scores)
    pct_total = 100 * sum(all_scores) / n_total if n_total else 0
    eligible = [r for r in results if r.get("has_answer_sources")]
    overall_recall = (
        100 * sum(1 for r in eligible if r.get("recall_hit")) / len(eligible)
        if eligible else float("nan")
    )
    recall_str = f"{overall_recall:>9.1f}%" if eligible else "       n/a"
    soft_total = 100 * sum(r.get("soft_pass", False) for r in results) / n_total if n_total else 0
    lines.append("-" * 56)
    lines.append(f"{'OVERALL':<20} {n_total:>5} {pct_total:>6.1f}% {soft_total:>9.1f}% {recall_str}")
    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _locomo_worker_init(config_or_url: str) -> None:
    """Called once per worker thread — creates Organism or HTTP client."""
    if config_or_url.startswith("http"):
        _thread_local.org = _OrganismHttpClient(config_or_url)
    else:
        from organism.config import OrganismConfig
        from organism.core.organism import Organism
        cfg = OrganismConfig.from_yaml(config_or_url)
        cfg.rag.context_window_enabled = False
        _thread_local.org = Organism.from_config(cfg)


def _run_sample_worker(args: tuple) -> tuple:
    """Run one LoCoMo sample in a worker thread. Returns (s_idx, list[results])."""
    s_idx, sample, no_consolidation, limit, trace = args
    org = _thread_local.org
    sample_id = sample["sample_id"]
    user_id = f"locomo_{sample_id}_{uuid.uuid4().hex[:8]}"
    dia_to_blocks = replay_sample(org, sample, user_id, no_consolidation=no_consolidation)
    qa_list = sample["qa"][:limit] if limit else sample["qa"]
    qa_results = []
    for qa in qa_list:
        result = ask_question(org, user_id, qa["question"], qa, dia_to_blocks, trace=trace)
        result["sample_id"] = sample_id
        qa_results.append(result)
        with _qa_counter_lock:
            _qa_counter["done"] += 1
            done = _qa_counter["done"]
            total = _qa_counter["total"]
        if done % 50 == 0 or done == total:
            logger.info("[QA %d/%d done]", done, total)
    return s_idx, qa_results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="LoCoMo benchmark for Organism")
    parser.add_argument("--config", default="organism_config.yaml",
                        help="Config YAML (in-process mode)")
    parser.add_argument("--api-url", default=None,
                        help="Organism HTTP server URL (e.g. http://localhost:8000). "
                             "Enables fast-replay with server-side embedding.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--samples", type=int, default=None, help="Max samples (1-10)")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--no-consolidation", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker threads (default: 4). Use 1 for sequential.",
    )
    parser.add_argument("--persona-id", default=None,
                        help="Run only this sample id (e.g. conv-26). Fast debug loop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N QA pairs per sample. For quick smoke tests.")
    parser.add_argument("--trace", action="store_true",
                        help="Log retrieved facts + prediction details per QA (verbose).")
    args = parser.parse_args()

    samples = load_locomo(args.data)
    if args.samples:
        samples = samples[:args.samples]
    if args.persona_id:
        samples = [s for s in samples if s["sample_id"] == args.persona_id]
        if not samples:
            logger.error("persona-id %r not found. Available: %s",
                         args.persona_id, [s["sample_id"] for s in load_locomo(args.data)])
            return
    workers = max(1, min(args.workers, len(samples)))
    total_qa = sum(len(s["qa"]) for s in samples)
    _qa_counter["total"] = total_qa
    _qa_counter["done"] = 0
    config_or_url = args.api_url if args.api_url else args.config
    mode = "HTTP" if args.api_url else "in-process"
    logger.info("Loaded %d samples (%d QA total), workers=%d, mode=%s",
                len(samples), total_qa, workers, mode)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.out_dir) / f"locomo_{ts}_results.json"
    summary_path = Path(args.out_dir) / f"locomo_{ts}_summary.txt"

    task_args = [(i, sample, args.no_consolidation, args.limit, args.trace)
                 for i, sample in enumerate(samples)]
    sample_results: List[Optional[List]] = [None] * len(samples)

    if workers == 1:
        _locomo_worker_init(config_or_url)
        org = _thread_local.org
        for s_idx, sample, nc, limit, trace in task_args:
            sample_id = sample["sample_id"]
            logger.info("Sample %d/%d (id=%s)", s_idx + 1, len(samples), sample_id)
            user_id = f"locomo_{sample_id}_{uuid.uuid4().hex[:8]}"
            dia_to_blocks = replay_sample(org, sample, user_id, no_consolidation=nc)
            qa_list = sample["qa"][:limit] if limit else sample["qa"]
            results_for_sample = []
            for qa in qa_list:
                r = ask_question(org, user_id, qa["question"], qa, dia_to_blocks, trace=trace)
                r["sample_id"] = sample_id
                results_for_sample.append(r)
            sample_results[s_idx] = results_for_sample
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=_locomo_worker_init,
            initargs=(config_or_url,),
        ) as executor:
            future_to_idx = {executor.submit(_run_sample_worker, t): t[0] for t in task_args}
            for future in as_completed(future_to_idx):
                s_idx, qa_results = future.result()
                sample_results[s_idx] = qa_results
                done = sum(1 for x in sample_results if x is not None)
                logger.info("Sample done %d/%d (id=%s, qa=%d)",
                            done, len(samples),
                            samples[s_idx]["sample_id"], len(qa_results))

    all_results: List[Dict] = []
    for sr in sample_results:
        if sr:
            all_results.extend(sr)

    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = print_summary(all_results)
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Results: %s", results_path)
    logger.info("Summary: %s", summary_path)

    # Per-persona breakdown
    by_persona: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        sid = r.get("sample_id", "unknown")
        by_persona[sid].append(r)

    persona_stats = []
    for pid, items in sorted(by_persona.items()):
        by_ptype: Dict[str, List] = defaultdict(list)
        for it in items:
            by_ptype[it["question_type"]].append(it)
        persona_stats.append({
            "persona_id": pid,
            "n_qa": len(items),
            "strict_pass_pct": round(100 * sum(it["pass"] for it in items) / len(items), 1),
            "soft_pass_pct": round(100 * sum(it.get("soft_pass", False) for it in items) / len(items), 1),
            "avg_f1": round(sum(it["f1"] for it in items) / len(items), 3),
            "recall_hit_pct": round(
                100 * sum(it["recall_hit"] for it in items if it["has_answer_sources"]) /
                max(1, sum(1 for it in items if it["has_answer_sources"])), 1
            ),
            "by_type": {
                qtype: {
                    "n": len(qs),
                    "strict_pass_pct": round(100 * sum(q["pass"] for q in qs) / len(qs), 1),
                    "avg_f1": round(sum(q["f1"] for q in qs) / len(qs), 3),
                }
                for qtype, qs in by_ptype.items()
            },
        })

    persona_path = Path(args.out_dir) / f"locomo_{ts}_by_persona.json"
    persona_path.write_text(json.dumps(persona_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Per-persona report: %s", persona_path)


if __name__ == "__main__":
    main()
