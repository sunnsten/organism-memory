from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, List, Tuple, Literal, TYPE_CHECKING
from torch import Tensor
import time
import json

from .common import SourceType, KindType

if TYPE_CHECKING:
    from .memory_item import MemoryRecord


@dataclass
class MemoryObservation:
    """
    Observation passed to MemoryCore.observe().

    hidden_states: last hidden layer of the backend [B, L, D]
    attention_mask: attention mask [B, L]
    attn_scores: averaged attention over the last token [B] or [B, H]
    surprisal: scalar signal — "how surprising is this"
    ssm_state: additional SSM state [B, D_state]
    text: human-readable text for log / experience buffer
    """
    hidden_states: Tensor
    attention_mask: Optional[Tensor] = None
    attn_scores: Optional[Tensor] = None
    surprisal: Optional[float] = None
    ssm_state: Optional[Tensor] = None
    text: str = ""


@dataclass
class ContextMeta:
    """
    Context metadata instead of the full context string.
    Allows the context to be reconstructed on demand without storing the entire prompt.
    """
    system_hash: str                              # hash of the system prompt
    memory_ids: List[int]                         # IDs from curated memories (SQLite) or slot indices
    chat_message_id_span: Tuple[int, int] | None  # (start_id, end_id) — message IDs from chat_log
    memory_id_space: Literal["curated", "slots"] | None = None  # "curated" = SQLite memory IDs, "slots" = MemoryCore slot indices
    prompt_tokens: int | None = None
    model_name: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    tool_calls: List[str] | None = None

    @property
    def history_span(self) -> Tuple[int, int] | None:
        """DEPRECATED: use chat_message_id_span instead. Kept for backward compatibility."""
        return self.chat_message_id_span


@dataclass
class EventRecord:
    """
    Raw interaction event — minimal record for replay / audit.
    Stores only metadata and a preview, not the full context.

    TODO: EventRecord should become canonical for the DB, without context: str.
    Currently it overlaps with ExperienceBlock (which stores context: str).
    In the future ExperienceBlock will either become a thin wrapper over EventRecord
    or be removed / marked legacy.
    """
    id: int | None
    user_id: str
    session_id: str | None
    timestamp: float  # unix timestamp

    input_text: str
    output_text: str

    kind: KindType
    source: SourceType

    # Metrics
    importance: float
    surprisal_norm: float | None = None
    attention_focus: float | None = None

    # Retrieved memories used
    used_memories: List[int] = field(default_factory=list)  # slot indices or memory ids
    used_memories_space: Literal["curated", "slots"] | None = None  # "curated" = SQLite memory IDs, "slots" = MemoryCore slot indices

    # Replaces full context string
    context_meta: Optional["ContextMeta"] = None  # context metadata (serialised as JSON in the store)
    text_preview: str | None = None               # first/last N chars for FTS / debug

    # Embedding (optional; serialised via embeddings.storage in the store)
    embedding: Tensor | None = None
    embedding_dim: int | None = None
    embedding_dtype: str = "float32"
    embedding_l2norm: bool = False  # whether the embedding is L2-normalised

    created_at: str | None = None  # ISO timestamp


@dataclass
class SlotRetrieveResult:
    """
    Result of a memory search — one relevant slot from MemoryCore.

    Distinct from DB retrieval (FTS search in SQLite): this is the result
    of slot-based retrieval from MemoryCore.
    """
    slot_index: int
    text: str
    score: float
    key: Tensor    # [d_compressed]
    value: Tensor  # [d_compressed]
    record: Optional["MemoryRecord"] = None


# Backward-compatibility alias
RetrieveResult = SlotRetrieveResult


@dataclass
class ExperienceBlock:
    """
    Intermediate experience block.

    Holds raw interaction data (input_text, output_text, context),
    metrics (importance, surprisal_norm, attention_focus), and optional
    fields for consolidation (summary, embedding).

    TODO: ExperienceBlock stores context: str (inflated prompt), which contradicts
    the goal of moving away from storing full context. Future plan:
    - EventRecord becomes canonical for the DB (no context: str)
    - ExperienceBlock becomes a thin wrapper over EventRecord, or is removed / marked legacy.
    """

    id: int | None
    user_id: str
    session_id: Optional[str]
    timestamp: float  # unix timestamp (float)
    created_at: str   # ISO timestamp (for compatibility)

    # Core interaction fields
    input_text: str          # raw user input
    output_text: str         # raw assistant output
    used_memories: list[int] # slot_indices from RetrieveResult

    # Metrics
    importance: float
    surprisal_norm: Optional[float] = None
    attention_focus: Optional[float] = None

    # Context for model training (teacher-forcing).
    # Contains the full prompt (system + memories + history) used during generation.
    context: Optional[str] = None

    # For sleep and consolidation (optional, generated later)
    summary_preview: Optional[str] = None  # text preview (raw slice of input/remember) for debug and FTS
    summary: Optional[str] = None          # final summary after consolidation (LLM-generated)
    embedding: Optional[Tensor] = None     # [D] — stored as bytes in SQLite
    embedding_dtype: str = "float32"

    # Metadata
    span_start_id: int | None = None
    span_end_id: int | None = None
    kind: KindType = "interaction"
    source: SourceType = "chat"
    stability: float = 0.0      # 0..1
    processed: bool = False     # whether the block has been promoted to long-term / weights
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """
        LEGACY: serialisation helper for backward compatibility.

        Used only by stores/sqlite/mappers.py for writing to the DB.
        In the future this should move to stores/sqlite/mappers.py or
        memory/stores/serialization.py — the domain should not know about the storage format.

        Note: embedding is serialised via embeddings.storage (bytes + dtype + dim),
        not via torch.save. At the store (SQLite) level, embedding is converted to BLOB.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "context": self.context,
            "used_memories": json.dumps(self.used_memories) if self.used_memories else None,
            "importance": self.importance,
            "surprisal_norm": self.surprisal_norm,
            "attention_focus": self.attention_focus,
            "summary_preview": self.summary_preview,
            "summary": self.summary,
            "embedding": self.embedding,  # Tensor in memory; serialised via embeddings.storage in the store
            "embedding_dtype": self.embedding_dtype,
            "span_start_id": self.span_start_id,
            "span_end_id": self.span_end_id,
            "kind": self.kind,
            "source": self.source,
            "stability": self.stability,
            "processed": self.processed,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embedding: Tensor | None = None) -> "ExperienceBlock":
        """
        LEGACY: deserialisation helper for backward compatibility.

        Used only by stores/sqlite/mappers.py for loading from the DB.

        Args:
            data: dict with ExperienceBlock fields
            embedding: Tensor embedding (if None, falls back to data["embedding"], may be None)
        """
        final_embedding: Tensor | None = None
        if embedding is not None:
            final_embedding = embedding
        elif "embedding" in data and data["embedding"] is not None:
            emb_value = data["embedding"]
            if isinstance(emb_value, Tensor):
                final_embedding = emb_value

        used_memories: list[int] = []
        if "used_memories" in data and data["used_memories"]:
            if isinstance(data["used_memories"], str):
                used_memories = json.loads(data["used_memories"])
            elif isinstance(data["used_memories"], list):
                used_memories = data["used_memories"]

        metadata: dict[str, Any] | None = None
        if "metadata" in data and data["metadata"]:
            if isinstance(data["metadata"], str):
                metadata = json.loads(data["metadata"])
            elif isinstance(data["metadata"], dict):
                metadata = data["metadata"]

        timestamp = data.get("timestamp")
        if timestamp is None:
            from datetime import datetime
            created_at_str = data.get("created_at", "")
            if created_at_str:
                try:
                    dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    timestamp = dt.timestamp()
                except (ValueError, AttributeError):
                    timestamp = 0.0
            else:
                timestamp = 0.0
        elif isinstance(timestamp, str):
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except (ValueError, AttributeError):
                timestamp = float(timestamp) if timestamp.replace(".", "").isdigit() else 0.0
        else:
            timestamp = float(timestamp)

        return cls(
            id=data.get("id"),
            user_id=str(data["user_id"]),
            session_id=data.get("session_id"),
            timestamp=timestamp,
            created_at=str(data.get("created_at", "")),
            input_text=str(data.get("input_text", "")),
            output_text=str(data.get("output_text", "")),
            context=str(data.get("context", "")),
            used_memories=used_memories,
            importance=float(data.get("importance", 0.0)),
            surprisal_norm=data.get("surprisal_norm"),
            attention_focus=data.get("attention_focus"),
            summary_preview=data.get("summary_preview"),
            summary=data.get("summary"),
            embedding=final_embedding,
            embedding_dtype=str(data.get("embedding_dtype", "float32")),
            span_start_id=data.get("span_start_id"),
            span_end_id=data.get("span_end_id"),
            kind=data.get("kind", "interaction"),  # type: ignore[assignment]
            source=data.get("source", "chat"),      # type: ignore[assignment]
            stability=float(data.get("stability", 0.0)),
            processed=bool(data.get("processed", False)),
            metadata=metadata,
        )


@dataclass
class ExperienceEvent:
    """
    Full episode record at runtime, before any transformations.

    Intermediate structure between:
    - chat() → ExperienceEvent → (ExperienceStore + experience_buffer)
    - ExperienceStore → SleepSample (via policy)
    """
    timestamp: float
    user_id: str
    input_text: str
    output_text: str
    session_id: Optional[str] = None

    context: Optional[str] = None  # assembled prompt (system + memories + history)

    retrieved: Optional[list["RetrieveResult"]] = None

    ssm_before: Optional[Tensor] = None
    ssm_after: Optional[Tensor] = None

    importance: Optional[float] = None
    attention_focus: Optional[float] = None
    surprisal_norm: Optional[float] = None

    meta: Optional[dict[str, Any]] = None


__all__ = [
    "MemoryObservation",
    "ContextMeta",
    "EventRecord",
    "SlotRetrieveResult",
    "RetrieveResult",  # backward-compatibility alias
    "ExperienceBlock",
    "ExperienceEvent",
]
