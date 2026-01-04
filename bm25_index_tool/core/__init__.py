"""Core modules for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from bm25_index_tool.core.file_discovery import discover_files
from bm25_index_tool.core.formatters import format_json, format_rich, format_simple
from bm25_index_tool.core.fusion import fuse_results
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.core.searcher import BM25Searcher

__all__ = [
    "discover_files",
    "BM25Indexer",
    "BM25Searcher",
    "fuse_results",
    "format_simple",
    "format_json",
    "format_rich",
]
