"""SQLite storage for BM25 index tool with FTS5 and sqlite-vec support.

Provides unified storage for:
- Documents with MD5 change detection
- FTS5 full-text search with BM25 ranking
- Vector embeddings via sqlite-vec
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import ensure_config_dir, get_index_dir, get_sqlite_db_path

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

# Vector dimensions for Nova multimodal embeddings
EMBEDDING_DIMENSIONS = 1024


def _serialize_float32(vector: Sequence[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_float32(data: bytes) -> list[float]:
    """Deserialize bytes to a float vector."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


@dataclass
class Document:
    """Represents an indexed document."""

    id: int
    path: str
    filename: str
    md5_hash: str
    content: str | None
    mime_type: str
    file_size: int
    indexed_at: str
    updated_at: str


@dataclass
class Chunk:
    """Represents a text or image chunk for vector embedding."""

    id: int
    document_id: int
    chunk_index: int
    chunk_type: str  # 'text' or 'image'
    text: str | None
    start_word: int | None
    end_word: int | None
    word_count: int | None


@dataclass
class BM25Result:
    """Result from BM25 search."""

    document_id: int
    path: str
    filename: str
    content: str | None
    score: float


@dataclass
class VectorResult:
    """Result from vector search."""

    chunk_id: int
    document_id: int
    path: str
    filename: str
    chunk_type: str
    text: str | None
    distance: float


class SQLiteStorage:
    """Unified SQLite storage for BM25 and vector search."""

    def __init__(self, index_name: str) -> None:
        """Initialize SQLite storage for an index.

        Args:
            index_name: Name of the index
        """
        self.index_name = index_name
        self.db_path = get_sqlite_db_path(index_name)
        self._conn: sqlite3.Connection | None = None
        self._vec_loaded = False

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            ensure_config_dir()
            index_dir = get_index_dir(self.index_name)
            index_dir.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Load sqlite-vec extension if available
            self._load_vec_extension()

        return self._conn

    def _load_vec_extension(self) -> None:
        """Load sqlite-vec extension if available."""
        if self._vec_loaded:
            return

        try:
            import sqlite_vec  # type: ignore[import-untyped]

            self._conn.enable_load_extension(True)  # type: ignore[union-attr]
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)  # type: ignore[union-attr]
            self._vec_loaded = True
            logger.debug("sqlite-vec extension loaded")
        except ImportError:
            logger.debug("sqlite-vec not available, vector search disabled")
        except Exception as e:
            logger.warning("Failed to load sqlite-vec: %s", e)

    def create_schema(self, with_vectors: bool = False) -> None:
        """Create database schema.

        Args:
            with_vectors: Whether to create vector tables (requires sqlite-vec)
        """
        cursor = self.conn.cursor()

        # Documents table with MD5 for change detection
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                md5_hash TEXT NOT NULL,
                content TEXT,
                mime_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # FTS5 virtual table for BM25 search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                path, filename, content,
                content='documents', content_rowid='id'
            )
        """)

        # Triggers to keep FTS5 in sync with documents table
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, path, filename, content)
                VALUES (new.id, new.path, new.filename, new.content);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, path, filename, content)
                VALUES('delete', old.id, old.path, old.filename, old.content);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, path, filename, content)
                VALUES('delete', old.id, old.path, old.filename, old.content);
                INSERT INTO documents_fts(rowid, path, filename, content)
                VALUES (new.id, new.path, new.filename, new.content);
            END
        """)

        # Chunks table for vector embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                text TEXT,
                start_word INTEGER,
                end_word INTEGER,
                word_count INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_md5 ON documents(md5_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")

        # Vector table using sqlite-vec (if requested and available)
        if with_vectors and self._vec_loaded:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    embedding FLOAT[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
                )
            """)
            logger.debug("Created chunks_vec table with %d dimensions", EMBEDDING_DIMENSIONS)

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        self.conn.commit()
        logger.info("Database schema created for index '%s'", self.index_name)

    def commit(self) -> None:
        """Commit current transaction."""
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteStorage:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    # Document operations

    def add_document(
        self,
        path: str,
        filename: str,
        md5_hash: str,
        content: str | None,
        mime_type: str,
        file_size: int,
    ) -> int:
        """Add a document to the index.

        If a document with the same path already exists, returns the existing ID.

        Args:
            path: Full path to the document
            filename: Filename only
            md5_hash: MD5 hash of file content
            content: Text content (None for images)
            mime_type: MIME type of the file
            file_size: File size in bytes

        Returns:
            Document ID
        """
        cursor = self.conn.cursor()

        # Check if document already exists
        cursor.execute("SELECT id FROM documents WHERE path = ?", (path,))
        existing = cursor.fetchone()
        if existing:
            existing_id: int = existing["id"]
            logger.debug("Document already exists, returning existing ID: %s", path)
            return existing_id

        now = datetime.now(UTC).isoformat()
        cursor.execute(
            """
            INSERT INTO documents
                (path, filename, md5_hash, content, mime_type, file_size, indexed_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, filename, md5_hash, content, mime_type, file_size, now, now),
        )
        self.conn.commit()
        doc_id = cursor.lastrowid
        logger.debug("Added document %d: %s", doc_id, path)
        return doc_id if doc_id is not None else 0

    def update_document(
        self,
        path: str,
        md5_hash: str,
        content: str | None,
        file_size: int,
    ) -> int | None:
        """Update an existing document.

        Args:
            path: Full path to the document
            md5_hash: New MD5 hash
            content: New text content
            file_size: New file size

        Returns:
            Document ID if found, None otherwise
        """
        now = datetime.now(UTC).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE documents
            SET md5_hash = ?, content = ?, file_size = ?, updated_at = ?
            WHERE path = ?
            """,
            (md5_hash, content, file_size, now, path),
        )
        self.conn.commit()

        if cursor.rowcount == 0:
            return None

        # Get the document ID
        cursor.execute("SELECT id FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        return row["id"] if row else None

    def delete_document(self, path: str) -> bool:
        """Delete a document and its chunks.

        Args:
            path: Full path to the document

        Returns:
            True if document was deleted
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE path = ?", (path,))
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted document: %s", path)
        return deleted

    def delete_document_by_id(self, doc_id: int) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_document(self, path: str) -> Document | None:
        """Get a document by path.

        Args:
            path: Full path to the document

        Returns:
            Document if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Document(**dict(row))

    def get_document_by_id(self, doc_id: int) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Document(**dict(row))

    def get_all_documents(self) -> list[Document]:
        """Get all documents in the index.

        Returns:
            List of all documents
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY path")
        return [Document(**dict(row)) for row in cursor.fetchall()]

    def get_document_count(self) -> int:
        """Get total number of documents.

        Returns:
            Document count
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_indexed_document_count(self) -> int:
        """Get number of documents that have vector embeddings.

        Returns:
            Count of distinct documents with chunks
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT document_id) as count FROM chunks")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_all_paths_with_hashes(self) -> dict[str, str]:
        """Get all document paths with their MD5 hashes.

        Returns:
            Dict mapping path to MD5 hash
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT path, md5_hash FROM documents")
        return {row["path"]: row["md5_hash"] for row in cursor.fetchall()}

    # Chunk operations

    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        chunk_type: str,
        text: str | None = None,
        start_word: int | None = None,
        end_word: int | None = None,
        word_count: int | None = None,
        embedding: Sequence[float] | None = None,
    ) -> int:
        """Add a chunk to the index.

        Args:
            document_id: ID of the parent document
            chunk_index: Index of the chunk within the document
            chunk_type: 'text' or 'image'
            text: Text content (for text chunks)
            start_word: Starting word index
            end_word: Ending word index
            word_count: Number of words in chunk
            embedding: Vector embedding (optional)

        Returns:
            Chunk ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks
                (document_id, chunk_index, chunk_type, text, start_word, end_word, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, chunk_index, chunk_type, text, start_word, end_word, word_count),
        )
        chunk_id = cursor.lastrowid
        if chunk_id is None:
            chunk_id = 0

        # Add embedding if provided and vec extension is loaded
        if embedding is not None and self._vec_loaded:
            cursor.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (chunk_id, _serialize_float32(embedding)),
            )

        self.conn.commit()
        return chunk_id

    def add_chunks_batch(
        self,
        chunks_data: list[dict[str, Any]],
        commit: bool = True,
    ) -> list[int]:
        """Add multiple chunks in a single transaction.

        Args:
            chunks_data: List of dicts with keys:
                - document_id: int
                - chunk_index: int
                - chunk_type: str ('text' or 'image')
                - text: str | None
                - start_word: int | None
                - end_word: int | None
                - word_count: int | None
                - embedding: list[float] | None
            commit: Whether to commit after batch (default True)

        Returns:
            List of chunk IDs
        """
        cursor = self.conn.cursor()
        chunk_ids: list[int] = []

        for chunk in chunks_data:
            cursor.execute(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, chunk_type, text, start_word, end_word, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["document_id"],
                    chunk["chunk_index"],
                    chunk["chunk_type"],
                    chunk.get("text"),
                    chunk.get("start_word"),
                    chunk.get("end_word"),
                    chunk.get("word_count"),
                ),
            )
            chunk_id = cursor.lastrowid or 0
            chunk_ids.append(chunk_id)

            # Add embedding if provided and vec extension is loaded
            embedding = chunk.get("embedding")
            if embedding is not None and self._vec_loaded:
                cursor.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_float32(embedding)),
                )

        if commit:
            self.conn.commit()

        return chunk_ids

    def delete_chunks_for_document(self, document_id: int) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        cursor = self.conn.cursor()

        # Get chunk IDs first (for vector table cleanup)
        cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
        chunk_ids = [row["id"] for row in cursor.fetchall()]

        # Delete from vector table if loaded
        if self._vec_loaded and chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            # Safe: placeholders is built from len(chunk_ids), not user input
            sql = f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})"  # nosec B608
            cursor.execute(sql, chunk_ids)

        # Delete chunks
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted

    def get_chunks_for_document(self, document_id: int) -> list[Chunk]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        )
        return [Chunk(**dict(row)) for row in cursor.fetchall()]

    def get_chunk_count(self) -> int:
        """Get total number of chunks.

        Returns:
            Chunk count
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        row = cursor.fetchone()
        return row["count"] if row else 0

    # BM25 Search

    def search_bm25(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Search documents using FTS5 BM25 ranking.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of search results sorted by relevance
        """
        cursor = self.conn.cursor()

        # FTS5 bm25() returns negative scores (more negative = better match)
        # We negate to get positive scores where higher = better
        cursor.execute(
            """
            SELECT d.id, d.path, d.filename, d.content, -bm25(documents_fts) as score
            FROM documents_fts f
            JOIN documents d ON f.rowid = d.id
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts)
            LIMIT ?
            """,
            (query, top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                BM25Result(
                    document_id=row["id"],
                    path=row["path"],
                    filename=row["filename"],
                    content=row["content"],
                    score=row["score"],
                )
            )

        logger.debug("BM25 search for '%s' returned %d results", query, len(results))
        return results

    # Vector Search

    def search_vector(self, embedding: Sequence[float], top_k: int = 10) -> list[VectorResult]:
        """Search chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results

        Returns:
            List of vector search results sorted by distance
        """
        # Access conn first to trigger lazy loading of sqlite-vec extension
        cursor = self.conn.cursor()

        if not self._vec_loaded:
            logger.warning("sqlite-vec not loaded, vector search unavailable")
            return []
        cursor.execute(
            """
            SELECT c.id, c.document_id, c.chunk_type, c.text, v.distance,
                   d.path, d.filename
            FROM chunks_vec v
            JOIN chunks c ON v.rowid = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_float32(embedding), top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                VectorResult(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    path=row["path"],
                    filename=row["filename"],
                    chunk_type=row["chunk_type"],
                    text=row["text"],
                    distance=row["distance"],
                )
            )

        logger.debug("Vector search returned %d results", len(results))
        return results

    def has_vector_index(self) -> bool:
        """Check if vector index exists.

        Returns:
            True if chunks_vec table exists
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'")
        return cursor.fetchone() is not None

    # Metadata operations

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value.

        Args:
            key: Metadata key

        Returns:
            Value if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value.

        Args:
            key: Metadata key
            value: Metadata value
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def get_all_metadata(self) -> dict[str, str]:
        """Get all metadata as a dictionary.

        Returns:
            Dict of all metadata key-value pairs
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        return {row["key"]: row["value"] for row in cursor.fetchall()}

    def get_indexed_chunk_keys(self) -> set[tuple[int, int]]:
        """Get set of (document_id, chunk_index) pairs for already indexed chunks.

        Returns:
            Set of (document_id, chunk_index) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT document_id, chunk_index FROM chunks")
        return {(row["document_id"], row["chunk_index"]) for row in cursor.fetchall()}

    def get_indexing_progress(self) -> dict[str, Any] | None:
        """Get indexing progress status.

        Returns:
            Dict with progress info or None if not found:
                - status: 'in_progress' or 'completed'
                - total_chunks: total chunks to process
                - indexed_chunks: chunks already indexed
                - current_batch: current batch number (if in_progress)
                - total_batches: total batches (if in_progress)
        """
        import json

        progress_json = self.get_metadata("indexing_progress")
        if progress_json:
            result: dict[str, Any] = json.loads(progress_json)
            return result
        return None


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal MD5 hash string
    """
    # MD5 is used for change detection, not security - safe to use
    hasher = hashlib.md5(usedforsecurity=False)  # nosec B324
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
