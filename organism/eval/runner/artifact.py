import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class TurnArtifact:
    """Artifact for a single turn in an eval scenario."""
    step_index: int  # actual index in the steps array
    step_id: str
    user: str
    assistant: str
    retrieval: Dict[str, Any]  # retrieval trace
    write: Optional[Dict[str, Any]] = None  # write info (experience blocks, memories)
    success: Optional[bool] = None  # step success flag (present if the step has an expect)
    expect_result: Optional[Dict[str, Any]] = None  # details of the expect check
    context: Optional[Dict[str, Any]] = None  # unified tracing context: run_id, test_id, mode, user_id, session_id, step_index, step_id, user_turn_index
    timing_ms: Dict[str, float] = field(default_factory=dict)  # timings in milliseconds
    stages: List[Dict[str, Any]] = field(default_factory=list)  # stage events (TURN_START, RETRIEVE_END, ...)
    errors: List[Dict[str, Any]] = field(default_factory=list)  # errors: stage, error, traceback
    cuda: Optional[Dict[str, Any]] = None  # CUDA info (if available)
    # context window metrics
    prompt_tokens_total: Optional[int] = None  # total token count in the prompt
    history_tokens_used: Optional[int] = None  # history token count in the prompt
    summary_tokens_used: Optional[int] = None  # summary token count in the prompt
    retrieval_items_used_count: Optional[int] = None  # number of retrieval items used
    jobs_enqueued: List[str] = field(default_factory=list)  # IDs of enqueued jobs
    job_results_applied: List[str] = field(default_factory=list)  # IDs of applied job results


@dataclass
class RunArtifact:
    """Artifact for a full scenario run."""
    run_id: str  # UTC ISO timestamp
    test_id: str
    mode: str  # A_memory_off or B_memory_on
    seed: int
    model: str  # from cfg.base_model.model_name
    config: Dict[str, Any]  # temperature, retrieve_k, write_enabled, retrieve_flags
    db_path: Optional[str] = None  # path to the saved DB file
    turns: List[TurnArtifact] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)  # success, aggregates

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the artifact to a dict for JSON encoding."""
        return {
            "run_id": self.run_id,
            "test_id": self.test_id,
            "mode": self.mode,
            "seed": self.seed,
            "model": self.model,
            "config": self.config,
            "db_path": self.db_path,
            "turns": [asdict(turn) for turn in self.turns],
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the artifact to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str):
        """Save the artifact to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def create_run_id() -> str:
    """Create a unique run ID (UTC ISO timestamp)."""
    return datetime.utcnow().isoformat() + "Z"
