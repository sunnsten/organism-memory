import json
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

from organism.config import OrganismConfig
from organism.shared.analytics.memory_metrics import take_snapshot
from organism.mcp_server import _make_handlers

from .context import RunContext
from .matchers import contains_all, contains_none
from .artifact import RunArtifact, TurnArtifact, create_run_id
from .adapter import EvalAdapter

logger = logging.getLogger(__name__)


def _get_cuda_memory_snapshot() -> Optional[Dict[str, float]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        return {
            "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
            "memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        }
    except (ImportError, Exception):
        return None


def _add_stage_with_cuda(
    stages: List[Dict[str, Any]],
    stage_name: str,
    t_ms: float,
    last_cuda_snapshot: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    cuda_snapshot = _get_cuda_memory_snapshot()
    peak_delta_mb = None
    if cuda_snapshot and last_cuda_snapshot:
        prev_peak = last_cuda_snapshot.get("max_memory_allocated_mb")
        curr_peak = cuda_snapshot.get("max_memory_allocated_mb")
        if prev_peak is not None and curr_peak is not None:
            peak_delta_mb = max(curr_peak - prev_peak, 0.0)
    stage_data: Dict[str, Any] = {"stage": stage_name, "t_ms": t_ms, "cuda": cuda_snapshot}
    if peak_delta_mb is not None:
        stage_data["peak_delta_mb"] = peak_delta_mb
    stages.append(stage_data)
    return cuda_snapshot


def _get_cuda_info() -> Optional[Dict[str, Any]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        current_device = torch.cuda.current_device()
        cuda_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
        }
        memory_snapshot = _get_cuda_memory_snapshot()
        if memory_snapshot:
            cuda_info.update(memory_snapshot)
        return cuda_info
    except ImportError:
        return None
    except Exception:
        return None


def apply_mode_config(config: OrganismConfig, mode: str) -> OrganismConfig:
    if mode == "A_memory_off":
        config.rag.enable_retrieve_db = False
        config.slots.enable_retrieve_slots = False
    elif mode == "B_memory_on":
        config.rag.enable_retrieve_db = True
        config.slots.enable_retrieve_slots = True
        config.slots.retrieve_top_k = 5
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return config


def _apply_rag_config_overrides(organism: Any, overrides: Dict[str, Any]) -> None:
    from organism.core.config import RAGConfig
    orchestrator = getattr(organism, "_orchestrator", None)
    if orchestrator is None:
        logger.warning("Cannot apply config_overrides: organism has no _orchestrator")
        return
    rag_config = getattr(orchestrator, "_rag_config", None)
    if rag_config is None:
        logger.warning("Cannot apply config_overrides: orchestrator has no _rag_config")
        return
    known_fields = set(RAGConfig.__dataclass_fields__.keys())
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if key in known_fields:
            setattr(rag_config, key, value)
            logger.debug("Applied config override: %s = %r", key, value)
        else:
            logger.warning("Unknown RAGConfig field in _config_overrides: %s", key)


def _write_event_trace(
    run_ctx: Dict[str, Any],
    turn: TurnArtifact,
    output_dir: Optional[str] = None,
    trace_jsonl_enabled: bool = False,
    event_type: str = "turn",
) -> None:
    if output_dir is None or not trace_jsonl_enabled:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id_safe = run_ctx["run_id"].replace(":", "-").replace(".", "-")
    trace_file = output_path / f"trace_{run_ctx['test_id']}_{run_ctx['mode']}_{run_id_safe}.jsonl"
    event = {
        "event_type": event_type,
        "context": run_ctx,
        "turn": {
            "step_index": turn.step_index,
            "step_id": turn.step_id,
            "user": turn.user,
            "assistant": turn.assistant,
            "retrieval": turn.retrieval,
            "write": turn.write,
            "success": turn.success,
            "expect_result": turn.expect_result,
            "timing_ms": turn.timing_ms,
            "stages": turn.stages,
            "errors": turn.errors,
            "cuda": turn.cuda,
        },
    }
    try:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write event trace: {e}")


def check_step_expect(
    answer: str,
    expect: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    """Check must_include / must_not_include assertions on an answer string."""
    details: Dict[str, Any] = {}
    checks: List[bool] = []

    if "must_include" in expect:
        must_include = expect["must_include"]
        if not isinstance(must_include, list):
            must_include = [must_include]
        result = contains_all(answer, must_include)
        details["must_include"] = {"expected": must_include, "found": result}
        checks.append(result)

    if "must_not_include" in expect:
        must_not_include = expect["must_not_include"]
        if not isinstance(must_not_include, list):
            must_not_include = [must_not_include]
        result = contains_none(answer, must_not_include)
        details["must_not_include"] = {"expected": must_not_include, "found": result}
        checks.append(result)

    success = all(checks) if checks else True
    return success, details


def run_scenario(
    scenario: Dict[str, Any],
    mode: str,
    base_config: Optional[OrganismConfig] = None,
    db_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RunArtifact:
    test_id = scenario.get("test_id", "unknown")
    seed = scenario.get("seed", 42)
    steps = scenario.get("steps", [])

    random.seed(seed)
    run_id = create_run_id()

    with RunContext(
        base_config=base_config,
        db_path=db_path,
        output_dir=output_dir,
        test_id=test_id,
        mode=mode,
        run_id=run_id,
    ) as ctx:
        apply_mode_config(ctx.config, mode)

        organism = ctx.create_organism()
        user_id = "eval_user"
        tenant_id = "eval"

        config_overrides = {
            k: v for k, v in scenario.get("_config_overrides", {}).items()
            if not k.startswith("_")
        }
        if config_overrides:
            _apply_rag_config_overrides(organism, config_overrides)
            logger.info("Applied %d _config_overrides from scenario", len(config_overrides))

        session_id = organism.start_session(user_id=user_id, title=f"eval_{test_id}_{mode}")

        # MCP handlers — shared between action steps and chat steps
        handlers = _make_handlers(organism)

        memory_snapshot_before = take_snapshot()

        artifact = RunArtifact(
            run_id=run_id,
            test_id=test_id,
            mode=mode,
            seed=seed,
            model=ctx.config.base_model.model_name,
            config={
                "temperature": getattr(ctx.config.base_model, "temperature", None),
                "retrieve_k": ctx.config.slots.retrieve_top_k,
                "max_new_tokens": scenario.get("max_new_tokens"),
                "retrieve_flags": {
                    "enable_retrieve_db": ctx.config.rag.enable_retrieve_db,
                    "enable_retrieve_slots": ctx.config.slots.enable_retrieve_slots,
                },
            },
            db_path=ctx.get_db_path(),
        )

        trace_jsonl_enabled = scenario.get("trace_jsonl", True)
        user_turn_index = -1

        for turn_index, step in enumerate(steps):
            step_id = step.get("step_id", f"S{turn_index + 1}")

            # --- action steps (no LLM call) ---
            action = step.get("action")
            if action is not None:
                t0 = time.perf_counter()
                action_errors: List[Dict[str, Any]] = []
                action_success: Optional[bool] = None
                action_expect_result: Optional[Dict[str, Any]] = None
                assistant_text = ""
                action_label = action

                try:
                    if action == "reset_user":
                        uid = step.get("user_id", user_id)
                        result_str = handlers["memory.reset"](user_id=uid, confirm=True)
                        result = json.loads(result_str)
                        action_label = f"reset_user:{uid}"
                        assistant_text = result_str
                        action_success = "error" not in result
                        action_expect_result = result

                    elif action == "store_event":
                        result_str = handlers["memory.store_event"](
                            user_id=user_id,
                            content=step.get("content", ""),
                            source=step.get("source", "eval"),
                            session_id=session_id,
                        )
                        result = json.loads(result_str)
                        action_label = step.get("content", "store_event")[:80]
                        assistant_text = result_str
                        action_success = "error" not in result
                        action_expect_result = result

                    elif action == "assert_query":
                        query = step.get("query", "")
                        result_str = handlers["memory.query"](
                            user_id=user_id,
                            query=query,
                            max_facts=step.get("max_facts", 8),
                            max_chunks=step.get("max_chunks", 5),
                        )
                        result = json.loads(result_str)
                        action_label = f"assert_query:{query}"
                        assistant_text = result_str
                        if "error" in result:
                            action_success = False
                            action_expect_result = {"query_error": result["error"]}
                        else:
                            all_text = " ".join(
                                item["content"]
                                for item in result.get("facts", []) + result.get("chunks", [])
                            )
                            expect = step.get("expect")
                            if expect:
                                action_success, action_expect_result = check_step_expect(
                                    all_text, expect
                                )
                                action_expect_result = action_expect_result or {}
                                action_expect_result["query_result"] = {
                                    "facts": result.get("total_facts", 0),
                                    "chunks": result.get("total_chunks", 0),
                                }
                            else:
                                action_success = True
                                action_expect_result = {
                                    "facts": result.get("total_facts", 0),
                                    "chunks": result.get("total_chunks", 0),
                                }

                    elif action == "wait_indexing":
                        timeout_s = float(step.get("timeout_s", 5))
                        store = organism._orchestrator._memory.store  # type: ignore[attr-defined]
                        deadline = time.perf_counter() + timeout_s
                        prev_count = store.facts.count(tenant_id, user_id)
                        while time.perf_counter() < deadline:
                            time.sleep(0.2)
                            curr_count = store.facts.count(tenant_id, user_id)
                            if curr_count == prev_count:
                                break
                            prev_count = curr_count
                        action_label = "wait_indexing"
                        assistant_text = json.dumps({"facts_count": prev_count})
                        action_success = True
                        action_expect_result = {"facts_count": prev_count}

                    elif action == "session_break":
                        session_id = organism.start_session(
                            user_id=user_id, title=f"eval_{test_id}_{mode}_s2"
                        )
                        action_label = "session_break"
                        assistant_text = json.dumps({"new_session_id": session_id})
                        action_success = True
                        action_expect_result = {"new_session_id": session_id}

                    else:
                        raise ValueError(f"Unknown action: {action!r}")

                except Exception as e:
                    logger.error(f"Error in action step {step_id} ({action}): {e}", exc_info=True)
                    assistant_text = f"ERROR: {e}"
                    action_success = False
                    action_expect_result = {"error": str(e)}
                    action_errors.append({
                        "stage": "action",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    })

                action_turn = TurnArtifact(
                    step_index=turn_index,
                    step_id=step_id,
                    user=action_label,
                    assistant=assistant_text,
                    retrieval={},
                    write=None,
                    success=action_success,
                    expect_result=action_expect_result,
                    context={
                        "run_id": run_id, "test_id": test_id, "mode": mode,
                        "user_id": user_id, "session_id": session_id,
                        "step_index": turn_index, "step_id": step_id, "action": action,
                    },
                    timing_ms={"action_total": (time.perf_counter() - t0) * 1000},
                    errors=action_errors,
                )
                artifact.turns.append(action_turn)
                continue

            # --- chat steps ---
            role = step.get("role", "user")
            content = step.get("content", "")
            expect = step.get("expect")

            retrieve_k_override = None
            if isinstance(expect, dict):
                retrieve_k_override = expect.get("retrieve_k")

            if role != "user":
                continue

            max_new_tokens_override = step.get("max_new_tokens") or scenario.get("max_new_tokens")

            if content == "_PHASE_BOUNDARY":
                logger.debug("Skipping phase boundary step: %s", step_id)
                continue

            user_turn_index += 1

            run_ctx = {
                "run_id": run_id,
                "test_id": test_id,
                "mode": mode,
                "user_id": user_id,
                "session_id": session_id,
                "step_index": turn_index,
                "step_id": step_id,
                "user_turn_index": user_turn_index,
            }

            original_retrieve_k = ctx.config.slots.retrieve_top_k
            if retrieve_k_override is not None:
                ctx.config.slots.retrieve_top_k = retrieve_k_override

            timing_ms: Dict[str, float] = {}
            stages: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            except (ImportError, Exception):
                pass

            last_cuda_snapshot: Optional[Dict[str, float]] = None

            t0 = time.perf_counter()
            turn_start = t0
            last_cuda_snapshot = _add_stage_with_cuda(stages, "TURN_START", 0.0, last_cuda_snapshot)

            chat_start: Optional[float] = None

            try:
                chat_start = time.perf_counter()
                last_cuda_snapshot = _add_stage_with_cuda(
                    stages, "CHAT_START", (chat_start - t0) * 1000, last_cuda_snapshot
                )

                reply = organism.chat(
                    user_id=user_id,
                    user_message=content,
                    system_prompt=scenario.get("system_prompt", None),
                    session_id=session_id,
                    max_new_tokens=max_new_tokens_override,
                )

                chat_end = time.perf_counter()
                timing_ms["chat_generate"] = (chat_end - chat_start) * 1000
                last_cuda_snapshot = _add_stage_with_cuda(
                    stages, "CHAT_END", (chat_end - t0) * 1000, last_cuda_snapshot
                )

                trace = EvalAdapter.get_last_trace(organism)
                last_cuda_snapshot = _add_stage_with_cuda(
                    stages, "RETRIEVE_END", (time.perf_counter() - t0) * 1000, last_cuda_snapshot
                )

                encoded_debug = EvalAdapter.get_last_encoded_debug(organism)
                if encoded_debug and stages and stages[0].get("stage") == "TURN_START":
                    if "meta" not in stages[0]:
                        stages[0]["meta"] = {}
                    stages[0]["meta"]["encoded_debug"] = encoded_debug

                write_skipped_stages = EvalAdapter.get_last_write_skipped_stages(organism)
                write_report = EvalAdapter.get_last_write_report(organism)
                attention_trace = EvalAdapter.get_last_attention_trace(organism)

                write_info: Dict[str, Any] = {}
                if write_report:
                    write_info["event_id"] = write_report.get("event_id")
                    write_info["chat_span"] = write_report.get("chat_span")
                    write_info["used_memories"] = write_report.get("used_memories", [])
                    write_info["embedding_dim"] = write_report.get("embedding_dim")
                    write_info["embedding_dtype"] = write_report.get("embedding_dtype")
                    write_info["embedding_l2norm"] = write_report.get("embedding_l2norm")
                    write_info["stored_tables"] = write_report.get("stored_tables", [])
                if attention_trace:
                    write_info["attention_trace"] = attention_trace
                if write_skipped_stages:
                    write_info["write_skipped_stages"] = write_skipped_stages
                if encoded_debug:
                    write_info["encoded_debug"] = encoded_debug

                success = None
                expect_result = None
                if expect:
                    if trace is None:
                        from organism.shared.domain import RetrievalTrace
                        trace = RetrievalTrace(query=content, top_k=0)
                    success, expect_result = check_step_expect(reply.reply, expect)

                retrieval_info: Dict[str, Any] = {}
                if trace:
                    metadata = getattr(trace, "metadata", {})
                    db_result_ids = getattr(trace, "db_result_ids", [])
                    db_previews = metadata.get("db_result_text_previews", [])
                    retrieval_info = {
                        "query": getattr(trace, "query", ""),
                        "top_k": getattr(trace, "top_k", 0),
                        "db_results_count": getattr(trace, "db_results_count", 0),
                        "slot_results_count": getattr(trace, "slot_results_count", 0),
                        "db_result_ids": db_result_ids,
                        "db_result_text_previews": db_previews,
                        "slot_result_text_previews": metadata.get("slot_result_text_previews", []),
                        "scores": metadata.get("slot_scores", []),
                        "reason": metadata.get("reason"),
                        "used_memories_ids": db_result_ids.copy(),
                        "used_memories_preview": db_previews.copy(),
                    }

                turn_end = time.perf_counter()
                timing_ms["turn_total"] = (turn_end - turn_start) * 1000
                last_cuda_snapshot = _add_stage_with_cuda(
                    stages, "TURN_END", (turn_end - t0) * 1000, last_cuda_snapshot
                )

                cuda_info = _get_cuda_info()
                if encoded_debug and cuda_info:
                    cuda_info["encoded_debug"] = encoded_debug  # type: ignore[assignment]

                turn = TurnArtifact(
                    step_index=turn_index,
                    step_id=step_id,
                    user=content,
                    assistant=reply.reply,
                    retrieval=retrieval_info,
                    write=write_info if write_info else None,
                    success=success,
                    expect_result=expect_result,
                    context=run_ctx,
                    timing_ms=timing_ms,
                    stages=stages,
                    errors=errors,
                    cuda=cuda_info,
                )
                artifact.turns.append(turn)
                _write_event_trace(run_ctx, turn, output_dir, trace_jsonl_enabled, event_type="turn")

            except Exception as e:
                logger.error(f"Error in step {step_id}: {e}", exc_info=True)

                error_end = time.perf_counter()
                if chat_start is not None:
                    timing_ms["chat_generate"] = (error_end - chat_start) * 1000

                cuda_snapshot = _get_cuda_memory_snapshot()
                peak_delta_mb = None
                if cuda_snapshot and last_cuda_snapshot:
                    prev_peak = last_cuda_snapshot.get("max_memory_allocated_mb")
                    curr_peak = cuda_snapshot.get("max_memory_allocated_mb")
                    if prev_peak is not None and curr_peak is not None:
                        peak_delta_mb = max(curr_peak - prev_peak, 0.0)

                error_info: Dict[str, Any] = {
                    "stage": "chat_generate",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "t_ms": (error_end - t0) * 1000,
                    "cuda": cuda_snapshot,
                }
                if peak_delta_mb is not None:
                    error_info["peak_delta_mb"] = peak_delta_mb
                errors.append(error_info)
                last_cuda_snapshot = cuda_snapshot

                turn_end = time.perf_counter()
                timing_ms["turn_total"] = (turn_end - turn_start) * 1000
                last_cuda_snapshot = _add_stage_with_cuda(
                    stages, "TURN_END", (turn_end - t0) * 1000, last_cuda_snapshot
                )

                cuda_info = _get_cuda_info()
                encoded_debug = EvalAdapter.get_last_encoded_debug(organism) if hasattr(organism, "memory_service") else None
                if encoded_debug and cuda_info:
                    cuda_info["encoded_debug"] = encoded_debug  # type: ignore[assignment]

                turn = TurnArtifact(
                    step_index=turn_index,
                    step_id=step_id,
                    user=content,
                    assistant=f"ERROR: {str(e)}",
                    retrieval={},
                    write=None,
                    success=False,
                    expect_result={"error": str(e)},
                    context=run_ctx,
                    timing_ms=timing_ms,
                    stages=stages,
                    errors=errors,
                    cuda=cuda_info,
                )
                artifact.turns.append(turn)
                _write_event_trace(run_ctx, turn, output_dir, trace_jsonl_enabled, event_type="turn_error")

            finally:
                if retrieve_k_override is not None:
                    ctx.config.slots.retrieve_top_k = original_retrieve_k

        total_turns = len(artifact.turns)
        successful_turns = sum(1 for turn in artifact.turns if turn.success is True)
        failed_turns = sum(1 for turn in artifact.turns if turn.success is False)
        turns_with_expect = sum(1 for turn in artifact.turns if turn.success is not None)

        memory_snapshot_after = take_snapshot()
        memory_delta = memory_snapshot_after.delta(memory_snapshot_before)

        artifact.metrics = {
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "failed_turns": failed_turns,
            "turns_with_expect": turns_with_expect,
            "success_rate": successful_turns / turns_with_expect if turns_with_expect > 0 else None,
            "memory": memory_delta.to_dict(),
        }

        return artifact
