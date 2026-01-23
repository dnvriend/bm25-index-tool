"""Storage module for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from bm25_index_tool.storage.paths import ensure_config_dir, get_config_dir, get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import (
    BM25Result,
    Chunk,
    Document,
    SQLiteStorage,
    VectorResult,
    compute_file_hash,
    deserialize_float32,
)

__all__ = [
    "ensure_config_dir",
    "get_config_dir",
    "get_index_dir",
    "IndexRegistry",
    "SQLiteStorage",
    "Document",
    "Chunk",
    "BM25Result",
    "VectorResult",
    "compute_file_hash",
    "deserialize_float32",
]
