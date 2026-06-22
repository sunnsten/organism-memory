from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval and context assembly."""

    # Context Window Management
    context_window_enabled: bool = True
    context_window_max_history_tokens: int = 900
    context_window_min_messages: int = 3
    context_window_overflow_trigger_tokens: int = 1200
    context_window_summary_max_tokens: int = 400
    context_window_summary_temperature: float = 0.2
    context_window_summary_prompt: str = (
        "You receive a [Previous summary] (if present) and new conversation messages. "
        "Write an UPDATED summary that merges ALL key facts from both parts into 2-4 sentences. "
        "Never drop facts from the previous summary unless explicitly contradicted by newer information. "
        "Always preserve: names, locations, numbers, decisions, preferences, constraints."
    )
    context_window_max_total_tokens: int = 2048
    """Total model context window budget. Set to match your model's max_position_embeddings."""
    context_window_reserved_output_tokens: int = 256
    """Tokens reserved for model output generation. Subtracted from available history budget."""
    context_window_working_memory_limit: Optional[int] = None
    """Max messages loaded from session history as working memory (Tier 0).
    None = unlimited (load all session messages — required for overflow to fire on long sessions).
    Set to an integer (e.g. 5) to use a sliding window (original behaviour).
    """

    # Memory Extraction
    memory_extraction_enabled: bool = True
    memory_extraction_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    memory_extraction_max_input_tokens: int = 1200
    memory_extraction_max_output_tokens: int = 350
    memory_extraction_temperature: float = 0.2
    memory_extraction_run_mode: str = "periodic"  # "periodic" or "on_consolidation"

    # Embedder (vector search for Tier 1 chunks + Tier 2 facts)
    embedder_enabled: bool = True
    embedder_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedder_base_url: Optional[str] = None
    """When set, use VLLMEmbedder (HTTP) instead of Qwen3Embedder (local).
    Example: http://localhost:8002/v1"""
    embedder_dim: int = 1024
    """Vector dimension. Must match the serving model. Default 1024 for Qwen3-Embedding-0.6B."""

    # Retrieval toggles
    enable_retrieve_db: bool = True    # FTS retrieval from SQLite
    reranker_enabled: bool = False     # BGE-reranker-v2-m3; set True when FlagEmbedding installed
    locomo_mode: bool = False          # Aggressive memory settings for LoCoMo benchmark (k=20, more facts)

    # Tokenization (used by /remember)
    remember_max_length: int = 512
    answer_max_length: int = 256


__all__ = [
    "RAGConfig",
]
