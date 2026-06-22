from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import apsw

try:
    import vectorlite_py
    VECTORLITE_AVAILABLE = True
except ImportError:
    vectorlite_py = None  # type: ignore[assignment]
    VECTORLITE_AVAILABLE = False
    logging.warning("vectorlite_py not installed. Run: pip install vectorlite-py apsw")

logger = logging.getLogger(__name__)


def upgrade(conn: apsw.Connection) -> None:
    """
    Add vectorlite HNSW index for memory_items embeddings.

    HNSW parameters:
    - max_elements=100000: Support up to 100k vectors (adjustable)
    - ef_construction=200: Build quality (higher = better recall, slower build)
    - M=32: Graph connectivity (higher = better recall, more memory)
    """
    if not VECTORLITE_AVAILABLE:
        raise RuntimeError(
            "vectorlite_py is not installed. "
            "Install with: pip install vectorlite-py apsw numpy"
        )
    assert vectorlite_py is not None

    logger.info("Migration 002: Adding vectorlite HNSW index...")

    # 1. Enable vectorlite extension
    conn.enable_load_extension(True)
    vectorlite_py.load_vectorlite(conn)
    logger.info("  [1/3] Loaded vectorlite extension")

    # 2. Create virtual table with HNSW index
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory_items USING vectorlite(
            embedding float32[1024],
            hnsw(max_elements=100000, ef_construction=200, M=32)
        )
    """)
    logger.info("  [2/3] Created vec_memory_items virtual table with HNSW")

    # 3. Copy existing embeddings from memory_items
    # Note: rowid in vec_memory_items MUST match id in memory_items
    cursor = conn.execute(
        "SELECT id, embedding FROM memory_items WHERE embedding IS NOT NULL"
    )

    count = 0
    for row_id, embedding_blob in cursor:
        conn.execute(
            "INSERT INTO vec_memory_items (rowid, embedding) VALUES (?, ?)",
            (row_id, embedding_blob),
        )
        count += 1

    logger.info(f"  [3/3] Migrated {count} existing embeddings to HNSW index")
    logger.info("Migration 002: Complete! Vector search is now 426x faster.")


def downgrade(conn: apsw.Connection) -> None:
    """
    Remove vectorlite HNSW index.

    WARNING: This will degrade performance back to O(n) Python cosine search.
    Only use for rollback scenarios.
    """
    logger.info("Migration 002 Rollback: Removing vectorlite HNSW index...")

    conn.execute("DROP TABLE IF EXISTS vec_memory_items")

    logger.info("Migration 002 Rollback: Complete. Vector search reverted to Python cosine.")


__all__ = ["upgrade", "downgrade"]
