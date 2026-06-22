from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-local storage: one Organism (or OrganismHttpClient) per worker thread
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# HTTP client — used when --api-url is given instead of --config
# ---------------------------------------------------------------------------

class _ChatReply:
    """Minimal stand-in for OrganismReply returned by org.chat()."""
    __slots__ = ("reply",)

    def __init__(self, reply: str) -> None:
        self.reply = reply


class _NoOpMemoryService:
    """Stub: consolidation and trace are handled server-side."""
    _last_trace = None

    def run_consolidation(self, user_id: str, limit: int = 200) -> None:
        pass


class OrganismHttpClient:
    """
    Thin HTTP wrapper that speaks the organism-server REST API.

    Exposes the same interface as Organism that run_instance() uses so the
    benchmark loop works without modification.  Fast-replay and Recall@K are
    not supported (those require direct DB access).

    Rate-limit handling: /chat returns HTTP 429 when the server-side limit is
    hit.  The client retries with exponential back-off (up to _MAX_RETRIES
    attempts).  Set ORGANISM_CHAT_RATE_LIMIT=1000/minute on the server to
    avoid throttling during benchmark runs.
    """

    _MAX_RETRIES = 6

    def __init__(self, api_url: str, timeout: float = 120.0) -> None:
        try:
            import requests as _requests
        except ImportError:
            raise ImportError("pip install requests") from None
        import requests  # noqa: PLC0415 — lazy import kept local to avoid top-level dep
        self._requests = requests
        self._session = requests.Session()
        self._base = api_url.rstrip("/")
        self._timeout = timeout
        self.memory_service = _NoOpMemoryService()

    # ------------------------------------------------------------------
    # Public interface (matches Organism)
    # ------------------------------------------------------------------

    def start_session(self, user_id: str) -> str:
        resp = self._session.post(
            f"{self._base}/session/start",
            json={"user_id": user_id},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["session_id"]

    def end_session(self, user_id: str, session_id: str) -> None:
        try:
            self._session.post(
                f"{self._base}/session/end",
                json={"user_id": user_id, "session_id": session_id},
                timeout=self._timeout,
            )
        except Exception as exc:
            logger.debug("end_session HTTP error (ignored): %s", exc)

    def replay_session(
        self,
        user_id: str,
        session_id: str,
        turns: List[Dict],
        session_ts: Optional[float] = None,
    ) -> Dict:
        """Fast-replay via /session/replay — writes turns to DB without LLM inference."""
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "turns": [
                {"role": t.get("role", ""), "content": t.get("content", ""),
                 "has_answer": bool(t.get("has_answer", False))}
                for t in turns if isinstance(t, dict)
            ],
        }
        if session_ts is not None:
            payload["session_ts"] = session_ts
        resp = self._session.post(
            f"{self._base}/session/replay",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def extract_session_facts(
        self, user_id: str, session_id: str, messages: List[Dict]
    ) -> int:
        # Facts are extracted server-side inside /session/replay — no-op here.
        return 0

    def embed_session_chunks(self, user_id: str, session_id: str) -> int:
        # Chunks are embedded server-side inside /session/replay — no-op here.
        return 0

    def chat(
        self,
        user_id: str,
        user_message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> _ChatReply:
        payload: Dict[str, Any] = {"user_id": user_id, "message": user_message}
        if session_id:
            payload["session_id"] = session_id
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if max_new_tokens:
            payload["max_new_tokens"] = max_new_tokens

        for attempt in range(self._MAX_RETRIES):
            resp = self._session.post(
                f"{self._base}/chat",
                json=payload,
                timeout=self._timeout,
            )
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)  # 5 10 20 40 80 160 s
                logger.warning(
                    "Rate limited by server (attempt %d/%d). Waiting %ds. "
                    "Set ORGANISM_CHAT_RATE_LIMIT=1000/minute on the server to avoid this.",
                    attempt + 1, self._MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return _ChatReply(resp.json().get("reply", ""))

        raise RuntimeError(
            f"HTTP /chat rate-limited after {self._MAX_RETRIES} retries. "
            "Start the server with ORGANISM_CHAT_RATE_LIMIT=1000/minute."
        )


# ---------------------------------------------------------------------------
# Scorer (pure functions — testable without LM)
# ---------------------------------------------------------------------------

_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, convert number words to digits."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [_WORD_TO_NUM.get(t, t) for t in tokens]
    return " ".join(tokens)


def token_f1(predicted: str, ground_truth: str) -> float:
    """
    Token-level F1 between predicted and ground_truth strings.
    Same metric used in SQuAD and the original LongMemEval paper.
    """
    pred_tokens = _normalize(predicted).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_gt_alternatives(ground_truth: str) -> List[str]:
    """
    LongMemEval sometimes packs multiple acceptable answers into one string:
      "7 days. 8 days (including the last day) is also acceptable."
      "'Data Analysis using Python' webinar"
    Extract the primary answer and any alternatives so each can be scored
    individually, and the best F1 is used.
    """
    alts = [ground_truth]  # always include full string as fallback

    # Split on ". " — each sentence may be a separate acceptable answer
    parts = [s.strip() for s in re.split(r"\.\s+", ground_truth) if s.strip()]
    for part in parts:
        # Strip trailing "is also acceptable" and parentheticals like "(including the last day)"
        clean = re.sub(r"\s*\([^)]*\)", "", part)           # remove (...) notes
        clean = re.sub(r"\s+is also acceptable$", "", clean, flags=re.IGNORECASE)
        clean = clean.strip(" .")
        if clean and clean not in alts:
            alts.append(clean)

    return alts


def score_response(predicted: str, ground_truth: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Score a predicted answer against ground truth. Returns dict with f1, pass, predicted, ground_truth."""
    alternatives = _extract_gt_alternatives(ground_truth)
    f1 = max(token_f1(predicted, alt) for alt in alternatives)
    return {
        "f1": f1,
        "pass": f1 >= threshold,
        "predicted": predicted,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_SPLIT_FILES = {
    "test":       "longmemeval_oracle",
    "oracle":     "longmemeval_oracle",
    "s":          "longmemeval_s",
    "m":          "longmemeval_m",
}


def load_dataset_split(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """
    Load LongMemEval from HuggingFace via hf_hub_download.

    Uses direct file download — avoids the deprecated trust_remote_code API.
    Cached after first download (~15 MB per split).

    Splits:
      - "test" / "oracle": longmemeval_oracle — 500 instances, oracle setting
      - "s": longmemeval_s — single-session subset
      - "m": longmemeval_m — multi-session subset
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    filename = _SPLIT_FILES.get(split)
    if filename is None:
        sys.exit(f"Unknown split {split!r}. Valid: {list(_SPLIT_FILES)}")

    logger.info("Downloading xiaowu0162/LongMemEval / %s ...", filename)
    path = hf_hub_download(
        repo_id="xiaowu0162/LongMemEval",
        filename=filename,
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    logger.info("Loaded %d instances from %s", len(items), filename)
    if limit is not None:
        items = _stratified_sample(items, limit)
        logger.info("Stratified sample: %d instances (%d requested)", len(items), limit)
    return items


def _stratified_sample(instances: List[Dict], limit: int) -> List[Dict]:
    """Sample proportionally from each question_type so --limit 60 gives ~10 per category."""
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for inst in instances:
        by_type[inst.get("question_type", "unknown")].append(inst)
    n_types = len(by_type)
    per_type = max(1, limit // n_types)
    result: List[Dict] = []
    for qtype, items in sorted(by_type.items()):
        result.extend(items[:per_type])
    # Fill remainder from largest types
    if len(result) < limit:
        for qtype, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            for inst in items[per_type:]:
                if len(result) >= limit:
                    break
                result.append(inst)
            if len(result) >= limit:
                break
    return result[:limit]


# ---------------------------------------------------------------------------
# Session date extraction for fast replay
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_PAT = "|".join(sorted(_MONTH_MAP, key=len, reverse=True))  # longest first


def _parse_session_timestamp(session_turns: List[Dict], session_idx: int, base_year: int = 2023) -> float:
    """
    Extract the earliest explicit date from session text and return as unix timestamp.

    Supports:
      - ISO: "2023-03-15"
      - "March 15th, 2024" / "March 15th" / "March 2024" / "March"
      - Ordinals: 1st/2nd/3rd/Nth

    Falls back to session_idx * 30 days from base_year-01-01 if no date found,
    so session order is always preserved even without explicit dates.
    """
    text = " ".join(
        t.get("content", "") for t in session_turns if isinstance(t, dict) and t.get("role") == "user"
    ).lower()

    # Pattern 1: ISO date YYYY-MM-DD
    m = re.search(r'\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b', text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).timestamp()
        except ValueError:
            pass

    # Pattern 2: "Month Day, Year" or "Month Day Year" — e.g. "February 10th, 2024"
    m = re.search(
        rf'\b({_MONTH_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b', text
    )
    if m:
        month = _MONTH_MAP[m.group(1)]
        day = min(int(m.group(2)), 28)
        year = int(m.group(3))
        try:
            return datetime(year, month, day).timestamp()
        except ValueError:
            pass

    # Pattern 3: "Month Day" (no year) — e.g. "March 15th"
    m = re.search(rf'\b({_MONTH_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', text)
    if m:
        month = _MONTH_MAP[m.group(1)]
        day = min(int(m.group(2)), 28)
        try:
            return datetime(base_year, month, day).timestamp()
        except ValueError:
            pass

    # Pattern 4: "Month Year" — e.g. "February 2024"
    m = re.search(rf'\b({_MONTH_PAT})\s+(20\d{{2}})\b', text)
    if m:
        month = _MONTH_MAP[m.group(1)]
        year = int(m.group(2))
        try:
            return datetime(year, month, 1).timestamp()
        except ValueError:
            pass

    # Fallback: preserve session order — 30-day spacing from base_year Jan 1
    base_ts = datetime(base_year, 1, 1).timestamp()
    return base_ts + session_idx * 30 * 86400


# ---------------------------------------------------------------------------
# Fast replay: write messages directly to DB, no LLM inference
# ---------------------------------------------------------------------------

def _replay_session_fast(
    org: Any,
    session_turns: List[Dict],
    user_id: str,
    sid: str,
    session_ts: Optional[float] = None,
) -> tuple:
    """
    Write session turns directly to the messages store and create ExperienceBlocks
    from user+assistant pairs — no LLM inference required.

    Returns (n_blocks, answer_block_ids) where answer_block_ids are experience
    block UUIDs for turns marked has_answer=True.

    session_ts: unix timestamp to use for ExperienceBlocks. If None, uses time.time().
                Pass a parsed date from the session text so the summarizer sees
                correct [Date: YYYY-MM-DD] tags and can preserve temporal order.

    Returns the number of ExperienceBlocks created.
    """
    from organism.shared.domain import EventRecord, ContextMeta

    store = org._orchestrator._memory.store
    tenant_id = org._tenant_id

    n_blocks = 0
    answer_block_ids: List[str] = []
    pending_user: Optional[str] = None
    pending_user_msg_id: Optional[int] = None
    pending_has_answer: bool = False
    turn_offset = 0  # unique timestamp per turn within session

    for turn in session_turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not content:
            continue

        if role == "user":
            # Store user message
            msg_id = store.messages.add(
                session_id=sid,
                tenant_id=tenant_id,
                user_id=user_id,
                role="user",
                content=content,
            )
            pending_user = content
            pending_user_msg_id = msg_id
            pending_has_answer = bool(turn.get("has_answer", False))
            turn_offset += 1

        elif role == "assistant":
            # Store assistant message
            asst_msg_id = store.messages.add(
                session_id=sid,
                tenant_id=tenant_id,
                user_id=user_id,
                role="assistant",
                content=content,
            )

            if pending_user is not None:
                # Each turn gets its own timestamp so temporal ordering is preserved
                base_ts = session_ts if session_ts is not None else time.time()
                turn_ts = base_ts + turn_offset * 3600

                # Create ExperienceBlock for this user+assistant pair
                text_preview = f"{pending_user}\n{content}"[:500]
                event = EventRecord(
                    id=None,
                    user_id=user_id,
                    session_id=sid,
                    timestamp=turn_ts,
                    input_text=pending_user,
                    output_text=content,
                    kind="interaction",
                    source="chat",
                    importance=0.5,
                    surprisal_norm=None,
                    attention_focus=None,
                    used_memories=[],
                    used_memories_space=None,
                    context_meta=ContextMeta(
                        system_hash="",
                        memory_ids=[],
                        chat_message_id_span=(
                            pending_user_msg_id or 0,
                            asst_msg_id,
                        ),
                        memory_id_space=None,
                    ),
                    text_preview=text_preview,
                    embedding=None,
                    embedding_dim=None,
                    embedding_dtype="float32",
                    embedding_l2norm=False,
                )
                try:
                    block_id = org._orchestrator._memory.write.append_event(
                        event, tenant_id=tenant_id, skip_chunk_embedding=True
                    )
                    if block_id:
                        n_blocks += 1
                        if pending_has_answer or bool(turn.get("has_answer", False)):
                            answer_block_ids.append(str(block_id))
                except Exception as exc:
                    logger.warning("Failed to write ExperienceBlock: %s", exc)

            pending_user = None
            pending_user_msg_id = None
            pending_has_answer = False

    return n_blocks, answer_block_ids


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_instance(org: Any, instance: Dict, user_id: str, fast_replay: bool = True, no_consolidation: bool = False) -> Dict[str, Any]:
    """
    Feed one LongMemEval instance through Organism and return scored result.

    Actual LongMemEval format:
      - haystack_sessions: list of sessions; each session is a list of turns
        {"role": "user"/"assistant", "content": str, "has_answer": bool}
      - question: str
      - answer: str
      - question_type: str (temporal-reasoning, single-session-user, etc.)
      - question_id: str

    Strategy: replay each session separately (new session per haystack session)
    so Organism builds independent memory per session, then ask the question
    in a final "eval" session. This matches the oracle setting where the model
    has seen all sessions.

    fast_replay=True (default): writes messages directly to DB without LLM
      inference. ~10-50x faster. Uses original assistant turns from dataset.
    fast_replay=False: calls org.chat() for each user turn (original slow mode).
    """
    haystack_sessions = instance.get("haystack_sessions", [])
    answer_source_ids: set = set()
    for idx, session_turns in enumerate(haystack_sessions):
        if not isinstance(session_turns, list):
            continue
        sid = org.start_session(user_id=user_id)

        session_ts: Optional[float] = None
        if fast_replay and hasattr(org, "replay_session"):
            # HTTP fast-replay: server writes turns directly to DB via /session/replay
            session_ts = _parse_session_timestamp(session_turns, session_idx=idx)
            try:
                result = org.replay_session(
                    user_id=user_id, session_id=sid,
                    turns=session_turns, session_ts=session_ts,
                )
                n_blocks = result.get("n_blocks", 0)
                logger.debug(
                    "Session %d: ts=%s wrote %d blocks via HTTP fast-replay",
                    idx, datetime.utcfromtimestamp(session_ts).strftime("%Y-%m-%d"), n_blocks,
                )
            except Exception as exc:
                logger.warning("HTTP fast-replay session %d failed: %s", idx, exc)
        elif fast_replay:
            # Direct fast-replay: write directly to DB in-process
            session_ts = _parse_session_timestamp(session_turns, session_idx=idx)
            n_blocks, answer_ids = _replay_session_fast(org, session_turns, user_id, sid, session_ts=session_ts)
            answer_source_ids.update(answer_ids)
            logger.debug(
                "Session %d: ts=%s wrote %d experience blocks (fast), %d answer blocks",
                idx,
                datetime.utcfromtimestamp(session_ts).strftime("%Y-%m-%d"),
                n_blocks,
                len(answer_ids),
            )
        else:
            # Slow mode: generate assistant responses via LLM for every turn
            for turn in session_turns:
                if not isinstance(turn, dict) or turn.get("role") != "user":
                    continue
                content = turn.get("content", "")
                if not content:
                    continue
                try:
                    org.chat(user_id=user_id, user_message=content, session_id=sid)
                except Exception as exc:
                    logger.warning("Replay session %d turn failed: %s", idx, exc)

        try:
            org.end_session(user_id=user_id, session_id=sid)
        except Exception:
            pass

        # Run synchronous consolidation after each session so experience_blocks
        # get promoted to MemoryItems before the eval question is asked.
        if not no_consolidation:
            try:
                org.memory_service.run_consolidation(user_id=user_id, limit=200)
            except Exception as exc:
                logger.warning("Consolidation after session %d failed: %s", idx, exc)

        # Extract facts from this session so FactRetriever has data at eval time.
        # Fast-replay bypasses org.chat(), so FactExtractor's daemon thread never fires;
        # we call it synchronously here instead.
        if fast_replay and hasattr(org, "extract_session_facts"):
            messages_for_extraction = [
                {"role": t["role"], "content": t.get("content", "")}
                for t in session_turns
                if isinstance(t, dict) and t.get("role") in ("user", "assistant")
            ]
            try:
                org.extract_session_facts(
                    user_id=user_id,
                    session_id=sid,
                    messages=messages_for_extraction,
                    session_ts=int(session_ts) if session_ts is not None else None,
                )
            except Exception as exc:
                logger.warning("Fact extraction after session %d failed: %s", idx, exc)

        # Bulk-embed all chunks written during fast-replay (skip_chunk_embedding=True
        # deferred embedding; one embed_batch() call here replaces N sequential embed()
        # calls that would have happened during replay).
        if fast_replay and hasattr(org, "embed_session_chunks"):
            try:
                org.embed_session_chunks(user_id=user_id, session_id=sid)
            except Exception as exc:
                logger.warning("Chunk embedding after session %d failed: %s", idx, exc)

    # Ask the eval question in a fresh session (all memory was built during replay above)
    # Type-specific prompts improve accuracy for temporal/multi-session/preference categories.
    _PROMPTS = {
        "single-session-preference": (
            "You are a memory assistant. Based on the user's conversation history, "
            "answer in exactly 2 sentences:\n"
            "Sentence 1: Start with 'The user would prefer responses that' then describe "
            "the specific type of response they want — name the exact tool, brand, platform, "
            "topic, or constraint mentioned in their history.\n"
            "Sentence 2: Start with 'They would not prefer' then describe what kind of "
            "response to avoid — generic advice, other brands, unrelated topics, etc.\n"
            "Use only facts from the conversation history. Do not invent details.",
            "",
        ),
        "multi-session": (
            "You are a helpful assistant with access to the user's personal memory. "
            "Answer by aggregating ALL relevant mentions across ALL sessions. "
            "Give only the final answer (number, name, or short phrase).",
            "\n\n(Check ALL sessions before answering. Give only the final answer.)",
        ),
        "temporal-reasoning": (
            "You are a helpful assistant with access to the user's personal memory. "
            "Use the timeline to identify relevant dates and reason about time order, duration, or sequence. "
            "Give a concise final answer — it may be a date, a name, a count, or a short phrase.",
            "\n\n(Give only the final answer — a date, name, number of days, or short phrase.)",
        ),
        "knowledge-update": (
            "You are a helpful assistant with access to the user's personal memory. "
            "Facts in the user's history may change over time. "
            "Always use the MOST RECENT value mentioned — if something was updated, corrected, "
            "or changed, give only the latest version. "
            "Spell out small numbers as words (one, two, three...). "
            "Give only the essential answer (a name, number, place, or short phrase).",
            "\n\n(Use the most recent value. Answer with ONLY the minimal answer.)",
        ),
    }
    _DEFAULT_PROMPT = (
        "You are a helpful assistant with access to the user's personal memory. "
        "Answer questions about the user's personal history concisely and directly. "
        "Give only the essential answer (a name, date, number, or short phrase). "
        "Do not explain, do not add caveats, do not ask follow-up questions.",
        "\n\n(Answer with ONLY the minimal answer — a single word, name, number, or short phrase. "
        "No verbs, no articles, no sentences, no explanation. "
        "Examples: 'Paris' / 'Monday' / '3 hours' / 'laptop'.)",
    )
    question_type = instance.get("question_type", "")
    sys_prompt, question_suffix = _PROMPTS.get(question_type, _DEFAULT_PROMPT)
    # Temporal needs more tokens for chain-of-thought date reasoning
    _MAX_TOKENS = {"temporal-reasoning": 256}
    max_new_tokens = _MAX_TOKENS.get(question_type, 128)

    eval_sid = org.start_session(user_id=user_id)
    eval_question = instance["question"] + question_suffix
    try:
        result = org.chat(
            user_id=user_id,
            user_message=eval_question,
            session_id=eval_sid,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
        )
        predicted = result.reply
    except Exception as exc:
        logger.error("Eval question failed: %s", exc)
        predicted = ""
    try:
        org.end_session(user_id=user_id, session_id=eval_sid)
    except Exception:
        pass

    # Recall@K: did retrieval surface any chunk from the answer-bearing turn?
    recall_hit = False
    if answer_source_ids:
        try:
            trace = org.memory_service._last_trace
            if trace:
                retrieved_source_ids = set(trace.metadata.get("chunk_source_ids", []))
                recall_hit = bool(answer_source_ids & retrieved_source_ids)
        except Exception:
            pass

    # Ground truth may be str or list[str]; take best F1 across alternatives
    ground_truths = instance.get("answer", "")
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        ground_truths = [str(ground_truths)]
    if not ground_truths:
        return {
            "question_type": instance.get("question_type", "unknown"),
            "question": instance.get("question", ""),
            "f1": 0.0,
            "pass": False,
            "predicted": predicted,
            "ground_truth": "",
        }

    best = max(
        (score_response(predicted, gt) for gt in ground_truths),
        key=lambda r: r["f1"],
    )

    return {
        "question_type": instance.get("question_type", "unknown"),
        "question": instance["question"],
        "recall_hit": recall_hit,
        "has_answer_sources": len(answer_source_ids) > 0,
        **best,
    }


def _worker_init(config_or_url: str, temperature_override: Optional[float]) -> None:
    """Called once per worker thread — creates a dedicated Organism or OrganismHttpClient."""
    if config_or_url.startswith("http://") or config_or_url.startswith("https://"):
        _thread_local.org = OrganismHttpClient(config_or_url)
    else:
        from organism.config import OrganismConfig
        from organism.core.organism import Organism
        cfg = OrganismConfig.from_yaml(config_or_url)
        if temperature_override is not None:
            cfg.base_model.temperature = temperature_override
        cfg.rag.context_window_enabled = False
        _thread_local.org = Organism.from_config(cfg)


def _run_instance_worker(
    args: tuple,
) -> tuple:
    """Wrapper for parallel execution: (index, result)."""
    idx, instance, user_id, fast_replay, no_consolidation = args
    result = run_instance(
        _thread_local.org, instance, user_id,
        fast_replay=fast_replay, no_consolidation=no_consolidation,
    )
    return idx, result


def print_summary(results: List[Dict], elapsed_seconds: float = 0.0) -> str:
    """Print per-type and overall scores with Recall@K and elapsed time. Returns summary string."""
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(float(r["pass"]))
    all_scores = [float(r["pass"]) for r in results]

    elapsed_str = f"{int(elapsed_seconds // 3600):02d}h{int((elapsed_seconds % 3600) // 60):02d}m{int(elapsed_seconds % 60):02d}s"
    lines = [
        "## LongMemEval Results",
        f"Elapsed: {elapsed_str}  ({elapsed_seconds:.0f}s)",
        f"{'Type':<40} {'N':>5} {'Score':>7} {'Recall@K':>10}",
        "-" * 67,
    ]
    for qtype, scores in sorted(by_type.items()):
        n = len(scores)
        pct = 100 * sum(scores) / n if n else 0
        type_results = [r for r in results if r["question_type"] == qtype]
        recall_eligible = [r for r in type_results if r.get("has_answer_sources")]
        recall_pct = (
            100 * sum(1 for r in recall_eligible if r.get("recall_hit")) / len(recall_eligible)
            if recall_eligible else float("nan")
        )
        recall_str = f"{recall_pct:>9.1f}%" if recall_eligible else "       n/a"
        lines.append(f"{qtype:<40} {n:>5} {pct:>6.1f}% {recall_str}")

    n_total = len(all_scores)
    pct_total = 100 * sum(all_scores) / n_total if n_total else 0
    eligible = [r for r in results if r.get("has_answer_sources")]
    overall_recall = (
        100 * sum(1 for r in eligible if r.get("recall_hit")) / len(eligible)
        if eligible else float("nan")
    )
    recall_overall_str = f"{overall_recall:>9.1f}%" if eligible else "       n/a"
    lines.append("-" * 67)
    lines.append(f"{'OVERALL':<40} {n_total:>5} {pct_total:>6.1f}% {recall_overall_str}")

    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from organism.shared.analytics.memory_metrics import take_snapshot
    import prometheus_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _bench_start = time.time()

    # Expose Prometheus metrics on port 9100 so Grafana can scrape during the run.
    try:
        prometheus_client.start_http_server(9100)
        logger.info("Prometheus metrics server started on :9100")
    except Exception as exc:
        logger.warning("Could not start Prometheus metrics server: %s", exc)

    parser = argparse.ArgumentParser(description="LongMemEval benchmark for Organism")
    parser.add_argument("--config",   default="organism_config.yaml")
    parser.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help=(
            "Send requests to a running organism-server instead of loading a model locally. "
            "Example: http://localhost:8000. "
            "Forces slow replay (fast-replay requires direct DB access). "
            "Set ORGANISM_CHAT_RATE_LIMIT=1000/minute on the server to avoid throttling."
        ),
    )
    parser.add_argument("--split",    default="test", help="Dataset split (default: test)")
    parser.add_argument("--limit",    type=int, default=None, help="Max instances (None = all)")
    parser.add_argument("--out-dir",  default="runs")
    parser.add_argument("--user-id",  default=None,
                        help="User ID prefix (default: auto-generated per run for isolation)")
    parser.add_argument(
        "--no-fast-replay",
        action="store_true",
        help="Disable fast replay (use slow LLM-based replay; original behaviour)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override model temperature for this run (e.g. 0.0 for deterministic/stable benchmark)",
    )
    parser.add_argument(
        "--no-consolidation",
        action="store_true",
        help="Skip LLM consolidation after each session (Tier 2 verbatim chunks only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker threads (default: 8). Each gets its own Organism. Use 1 for sequential.",
    )
    args = parser.parse_args()

    http_mode = bool(args.api_url)
    if http_mode and args.temperature is not None:
        logger.warning(
            "HTTP mode: --temperature is ignored (temperature is set in the server config)."
        )

    skip_set: set = set()

    # In HTTP mode, fast-replay is available via POST /session/replay on the server.
    # Only fall back to slow replay if --no-fast-replay is explicitly requested.
    fast_replay = not args.no_fast_replay
    if http_mode and fast_replay:
        logger.info("HTTP mode: fast replay via /session/replay on %s", args.api_url)
    elif http_mode:
        logger.info("HTTP mode: slow replay via /chat endpoint on %s", args.api_url)
    elif fast_replay:
        logger.info("Fast replay enabled: session history written directly to DB (no LLM inference during replay)")
    else:
        logger.info("Slow replay mode: LLM inference for every session turn")

    workers = max(1, args.workers)
    if http_mode and workers > 1:
        logger.warning(
            "HTTP mode with --workers %d: all workers share the server rate limit from the same IP. "
            "Consider --workers 1 for predictable pacing, or raise ORGANISM_CHAT_RATE_LIMIT on the server.",
            workers,
        )
    logger.info("Workers: %d", workers)

    instances = load_dataset_split(split=args.split, limit=args.limit)
    _mem_before = take_snapshot()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.out_dir) / f"longmemeval_{ts}_results.json"
    summary_path = Path(args.out_dir) / f"longmemeval_{ts}_summary.txt"

    user_prefix = args.user_id if args.user_id else f"lme_{ts}"
    task_args = [
        (i, instance, f"{user_prefix}_{i}", fast_replay, args.no_consolidation)
        for i, instance in enumerate(instances)
        if i not in skip_set
    ]
    if skip_set:
        logger.info("Skipping indices: %s", sorted(skip_set))

    # Results array pre-sized so order is preserved regardless of completion order
    results: List[Optional[Dict]] = [None] * len(instances)

    with open(results_path, "w", encoding="utf-8") as results_file:
        results_file.write("[\n")
        completed = 0

        if workers == 1:
            # Sequential — original behaviour, no extra Organisms created
            if http_mode:
                org: Any = OrganismHttpClient(args.api_url)
            else:
                from organism.config import OrganismConfig
                from organism.core.organism import Organism
                cfg = OrganismConfig.from_yaml(args.config)
                if args.temperature is not None:
                    cfg.base_model.temperature = args.temperature
                cfg.rag.context_window_enabled = False
                org = Organism.from_config(cfg)

            for idx, instance, uid, fr, nc in task_args:
                logger.info("[%d/%d] %s", idx + 1, len(instances), instance.get("question_type", "?"))
                r = run_instance(org, instance, user_id=uid, fast_replay=fr, no_consolidation=nc)
                results[idx] = r
                prefix = "" if idx == 0 else ","
                results_file.write(f"{prefix}\n  {json.dumps(r, ensure_ascii=False)}\n")
                results_file.flush()
                completed += 1
                if completed % 50 == 0:
                    snap = take_snapshot()
                    logger.info(
                        "Memory metrics @%d: facts_extracted=%d new=%d confirmed=%d errors=%d "
                        "retrieval_calls=%d facts_latency_avg=%.2fs",
                        completed, snap.facts_extracted, snap.facts_new, snap.facts_confirmed,
                        snap.facts_errors, snap.retrieval_calls, snap.facts_latency_avg_s,
                    )
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
        else:
            # Parallel — one Organism (or OrganismHttpClient) per worker thread via initializer
            config_or_url = args.api_url if http_mode else args.config
            with ThreadPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(config_or_url, args.temperature),
            ) as executor:
                future_to_idx = {
                    executor.submit(_run_instance_worker, t): t[0]
                    for t in task_args
                }
                pending_writes: Dict[int, Dict] = {}
                next_write = 0

                for future in as_completed(future_to_idx):
                    idx, r = future.result()
                    results[idx] = r
                    pending_writes[idx] = r
                    completed += 1
                    logger.info(
                        "[%d/%d done] %s → %s",
                        completed, len(instances),
                        r.get("question_type", "?"),
                        "PASS" if r.get("pass") else "fail",
                    )
                    # Flush in-order to keep JSON valid
                    while next_write in pending_writes:
                        r_w = pending_writes.pop(next_write)
                        prefix = "" if next_write == 0 else ","
                        results_file.write(f"{prefix}\n  {json.dumps(r_w, ensure_ascii=False)}\n")
                        results_file.flush()
                        next_write += 1

        results_file.write("]\n")

    results_clean = [r for r in results if r is not None]

    elapsed = time.time() - _bench_start
    summary = print_summary(results_clean, elapsed_seconds=elapsed)

    mem_delta = take_snapshot().delta(_mem_before)
    mem_section = "\n\n## Memory Metrics\n" + json.dumps(mem_delta.to_dict(), indent=2)
    summary_path.write_text(summary + mem_section, encoding="utf-8")

    # Save memory delta alongside results JSON
    mem_path = Path(args.out_dir) / f"longmemeval_{ts}_memory.json"
    mem_path.write_text(json.dumps(mem_delta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Results → %s", results_path)
    logger.info("Summary → %s", summary_path)
    logger.info("Memory  → %s", mem_path)


if __name__ == "__main__":
    main()
