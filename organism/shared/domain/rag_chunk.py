from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RAGChunk:
    """
    A chunk of text indexed for RAG retrieval (Tier 1).

    Created by the RAG Indexer worker from raw messages/documents.
    Each chunk has an embedding for vector search and is indexed
    in FTS5 for keyword search.
    """
    id: Optional[int] = None
    tenant_id: str = ""
    source_type: str = "message"    # 'message' / 'doc' / 'faq'
    source_id: int = 0              # FK to messages.id or doc id
    session_id: Optional[str] = None
    chunk_index: int = 0
    content: str = ""
    embedding: Optional[np.ndarray] = None
    created_at: Optional[int] = None
    tags: Optional[Dict[str, Any]] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self.embedding.shape[0] if self.embedding is not None else None


__all__ = ["RAGChunk"]
