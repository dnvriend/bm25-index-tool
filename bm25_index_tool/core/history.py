"""Search history with SQLite backend for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_config_dir

logger = get_logger(__name__)


class SearchHistory:
    """SQLite-backed search history manager.

    Stores search queries with metadata for analysis and replay.
    Database location: ~/.config/bm25-index-tool/history.db
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize search history.

        Args:
            db_path: Path to SQLite database (default: ~/.config/bm25-index-tool/history.db)
        """
        if db_path is None:
            config_dir = get_config_dir()
            config_dir.mkdir(parents=True, exist_ok=True)
            db_path = config_dir / "history.db"

        self.db_path = db_path
        self._initialize_db()

        logger.debug("SearchHistory initialized at %s", self.db_path)

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    indices TEXT NOT NULL,
                    query TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    path_filter TEXT,
                    exclude_path TEXT
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON search_history(timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query
                ON search_history(query)
            """)

            conn.commit()
            logger.debug("Database schema initialized")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection]:
        """Context manager for database connections.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def log_query(
        self,
        indices: list[str],
        query: str,
        top_k: int,
        result_count: int,
        elapsed_seconds: float,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
    ) -> None:
        """Log a search query to history.

        Args:
            indices: List of index names
            query: Search query
            top_k: Number of results requested
            result_count: Actual number of results returned
            elapsed_seconds: Query execution time in seconds
            path_filter: Include path patterns
            exclude_path: Exclude path patterns
        """
        timestamp = datetime.now().isoformat()

        # Serialize list fields to JSON
        indices_json = json.dumps(indices)
        path_filter_json = json.dumps(path_filter) if path_filter else None
        exclude_path_json = json.dumps(exclude_path) if exclude_path else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO search_history
                (timestamp, indices, query, top_k, result_count, elapsed_seconds,
                 path_filter, exclude_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    indices_json,
                    query,
                    top_k,
                    result_count,
                    elapsed_seconds,
                    path_filter_json,
                    exclude_path_json,
                ),
            )
            conn.commit()

        logger.debug(
            "Logged query: '%s' on %d indices (%d results, %.3fs)",
            query,
            len(indices),
            result_count,
            elapsed_seconds,
        )

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent search queries.

        Args:
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of query dictionaries ordered by timestamp (newest first)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, indices, query, top_k, result_count,
                       elapsed_seconds, path_filter, exclude_path
                FROM search_history
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            results = []
            for row in cursor.fetchall():
                result = {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "indices": json.loads(row["indices"]),
                    "query": row["query"],
                    "top_k": row["top_k"],
                    "result_count": row["result_count"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "path_filter": json.loads(row["path_filter"]) if row["path_filter"] else None,
                    "exclude_path": (
                        json.loads(row["exclude_path"]) if row["exclude_path"] else None
                    ),
                }
                results.append(result)

            logger.debug("Retrieved %d recent queries", len(results))
            return results

    def search(self, pattern: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search history by query pattern.

        Args:
            pattern: SQL LIKE pattern (e.g., '%kubernetes%')
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of matching query dictionaries ordered by timestamp (newest first)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, indices, query, top_k, result_count,
                       elapsed_seconds, path_filter, exclude_path
                FROM search_history
                WHERE query LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{pattern}%", limit),
            )

            results = []
            for row in cursor.fetchall():
                result = {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "indices": json.loads(row["indices"]),
                    "query": row["query"],
                    "top_k": row["top_k"],
                    "result_count": row["result_count"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "path_filter": json.loads(row["path_filter"]) if row["path_filter"] else None,
                    "exclude_path": (
                        json.loads(row["exclude_path"]) if row["exclude_path"] else None
                    ),
                }
                results.append(result)

            logger.debug("Search pattern '%s' matched %d queries", pattern, len(results))
            return results

    def clear(self) -> int:
        """Delete all search history.

        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM search_history")
            row = cursor.fetchone()
            count = int(row[0]) if row else 0

            cursor.execute("DELETE FROM search_history")
            conn.commit()

        logger.info("Cleared %d history entries", count)
        return count

    def count(self) -> int:
        """Get total number of history entries.

        Returns:
            Total entry count
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM search_history")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
