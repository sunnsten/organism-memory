from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Any

try:
    import apsw
    APSW_AVAILABLE = True
except ImportError:
    APSW_AVAILABLE = False
    # Fallback to stdlib sqlite3 if apsw not available
    import sqlite3
    apsw = None  # type: ignore

from .schema import init_schema

logger = logging.getLogger(__name__)

# Vectorlite support (optional, requires apsw)
try:
    import vectorlite_py
    VECTORLITE_AVAILABLE = True and APSW_AVAILABLE
except ImportError:
    VECTORLITE_AVAILABLE = False


class BaseStore:
    """
    Base class for all new Core stores.

    Manages a single SQLite database file with:
    - WAL mode for concurrent read access
    - Thread-local connections (apsw or sqlite3)
    - Schema auto-initialization on first use
    - vectorlite extension loading (if available)

    Usage:
        store = BaseStore(Path("data/organism.db"))
        cur = store.execute("SELECT * FROM memory_items")
        for row in cur:
            print(dict(row))
    """

    def __init__(self, db_path, *, auto_init: bool = True):
        """
        Args:
            db_path: Path to the SQLite database file, OR an already-open
                     apsw.Connection / sqlite3.Connection (useful in tests).
            auto_init: If True, initialize schema on first connection.
                       Ignored when db_path is an existing connection.
        """
        if not APSW_AVAILABLE:
            logger.warning(
                "apsw not installed. Falling back to sqlite3 (vectorlite unavailable). "
                "Install with: pip install apsw vectorlite-py"
            )

        # Support passing an already-open connection directly (e.g. in tests).
        _is_conn = (
            (APSW_AVAILABLE and isinstance(db_path, apsw.Connection))
            or (not APSW_AVAILABLE and hasattr(db_path, "execute") and not isinstance(db_path, Path))
        )
        if not _is_conn and not APSW_AVAILABLE:
            # Also catch sqlite3.Connection when apsw is unavailable
            try:
                import sqlite3 as _sqlite3
                _is_conn = isinstance(db_path, _sqlite3.Connection)
            except ImportError:
                pass

        if _is_conn:
            # Wrap the provided connection so _get_conn() always returns it.
            self._db_path = Path(":memory:")
            self._auto_init = False
            self._local = threading.local()
            self._local.conn = db_path
            self._initialized = True
            self._vectorlite_loaded = False
            self._init_lock = threading.Lock()
            # Apply row tracer for apsw connections that don't have one yet
            if APSW_AVAILABLE and isinstance(db_path, apsw.Connection):
                db_path.setrowtrace(self._row_tracer)
        else:
            self._db_path = db_path
            self._auto_init = auto_init
            self._local = threading.local()
            self._initialized = False
            self._vectorlite_loaded = False
            self._init_lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _get_conn(self):
        """
        Get or create a thread-local SQLite connection.

        Returns:
            apsw.Connection (preferred) or sqlite3.Connection (fallback).
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            if APSW_AVAILABLE:
                # Use apsw for vectorlite support
                conn = apsw.Connection(str(self._db_path))
                conn.setrowtrace(self._row_tracer)  # Dict-like row access
                cur = conn.cursor()
                cur.execute("PRAGMA busy_timeout=5000;")  # retry up to 5s before BusyError
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA foreign_keys=ON;")

                # Load vectorlite on every new connection (thread-local, must load per-conn)
                if VECTORLITE_AVAILABLE:
                    try:
                        conn.enable_load_extension(True)
                        vectorlite_py.load_vectorlite(conn)
                        if not self._vectorlite_loaded:
                            logger.info("Loaded vectorlite extension (HNSW enabled)")
                            self._vectorlite_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load vectorlite extension: {e}")
            else:
                # Fallback to sqlite3
                conn = sqlite3.connect(
                    str(self._db_path),
                    check_same_thread=False,
                    timeout=5.0,  # retry up to 5s before OperationalError: database is locked
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys=ON;")

            self._local.conn = conn

            # Initialize schema once per process
            if self._auto_init and not self._initialized:
                with self._init_lock:
                    if not self._initialized:
                        init_schema(conn)
                        self._initialized = True

        return conn

    @staticmethod
    def _row_tracer(cursor, row):
        """
        Row tracer for apsw to provide dict-like row access.

        Allows: row["column_name"] instead of row[0].
        """
        if row is None:
            return None
        # Get column names from cursor description
        names = [col[0] for col in cursor.getdescription()]
        return dict(zip(names, row))

    def connection(self):
        """
        Return the thread-local connection.

        Callers should NOT close this connection; it is reused across calls.
        """
        return self._get_conn()

    def execute(
        self,
        sql: str,
        params: tuple = (),
        *,
        commit: bool = False,
    ):
        """
        Execute a SQL statement on the thread-local connection.

        Args:
            sql: SQL statement.
            params: Bind parameters.
            commit: If True, commit after execution (no-op for apsw autocommit).

        Returns:
            apsw.Cursor or sqlite3.Cursor with results.
        """
        conn = self._get_conn()
        if APSW_AVAILABLE and isinstance(conn, apsw.Connection):
            cur = conn.cursor()
            cur.execute(sql, params)
            # apsw is autocommit by default, no need to commit
        else:
            cur = conn.execute(sql, params)
            if commit:
                conn.commit()
        return cur

    def executemany(
        self,
        sql: str,
        params_seq,
        *,
        commit: bool = False,
    ):
        """Execute a SQL statement for each set of parameters."""
        conn = self._get_conn()
        if APSW_AVAILABLE and isinstance(conn, apsw.Connection):
            cur = conn.cursor()
            cur.executemany(sql, params_seq)
            # apsw is autocommit by default, no need to commit
        else:
            cur = conn.executemany(sql, params_seq)
            if commit:
                conn.commit()
        return cur

    def last_insert_rowid(self) -> int:
        """
        Get the rowid of the last INSERT.

        Works for both apsw and sqlite3 connections.
        """
        conn = self._get_conn()
        if APSW_AVAILABLE and isinstance(conn, apsw.Connection):
            return conn.last_insert_rowid()
        else:
            # sqlite3: need to query manually
            cur = conn.execute("SELECT last_insert_rowid()")
            return cur.fetchone()[0]

    def commit(self) -> None:
        """
        Commit the current transaction.

        Note: apsw is autocommit by default, so this is a no-op for apsw.
        Only sqlite3 connections need explicit commit.
        """
        conn = self._get_conn()
        if hasattr(conn, "commit"):
            conn.commit()

    def close(self) -> None:
        """Close the thread-local connection (if open)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


__all__ = ["BaseStore"]
