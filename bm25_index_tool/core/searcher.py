"""BM25 search functionality for BM25 index tool.

Uses SQLite FTS5 for full-text search with BM25 ranking.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from typing import Any

from bm25_index_tool.core.fragments import extract_fragments
from bm25_index_tool.core.merge_strategies import MergeStrategy, RRFMergeStrategy
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage

logger = get_logger(__name__)


class BM25Searcher:
    """Query BM25 indices using SQLite FTS5."""

    def __init__(self) -> None:
        """Initialize the BM25 searcher."""
        self.registry = IndexRegistry()

    def search_single(
        self,
        index_name: str,
        query: str,
        top_k: int = 10,
        extract_fragments_flag: bool = False,
        context_lines: int = 3,
    ) -> list[dict[str, Any]]:
        """Search a single index.

        Args:
            index_name: Index name
            query: Search query
            top_k: Number of results to return
            extract_fragments_flag: If True, extract text fragments with matches
            context_lines: Number of context lines before/after matches

        Returns:
            List of result dictionaries with 'path', 'name', 'score', 'content'
            If extract_fragments_flag=True, also includes 'fragments' key

        Raises:
            ValueError: If index doesn't exist
        """
        # Verify index exists in registry
        metadata_dict = self.registry.get_index(index_name)
        if not metadata_dict:
            logger.error("Index '%s' not found", index_name)
            raise ValueError(f"Index '{index_name}' not found")

        logger.debug("Searching index '%s' for query: %s", index_name, query)

        # Parse query into terms for fragment extraction
        query_terms = query.split()

        # Search using SQLiteStorage
        try:
            with SQLiteStorage(index_name) as storage:
                bm25_results = storage.search_bm25(query, top_k)
        except Exception as e:
            logger.error("Failed to search index '%s': %s", index_name, e)
            raise ValueError(f"Failed to search index '{index_name}': {e}")

        # Format results
        output = []
        for result in bm25_results:
            content = result.content or ""

            formatted = {
                "path": result.path,
                "name": result.filename,
                "score": result.score,
                "content": content,
            }

            # Extract fragments if requested
            if extract_fragments_flag and content:
                fragments = extract_fragments(
                    content=content,
                    query_terms=query_terms,
                    context_lines=context_lines,
                    max_fragments=3,
                )
                formatted["fragments"] = fragments

            output.append(formatted)

        logger.info("Found %d results for query in '%s'", len(output), index_name)
        return output

    def search_multi(
        self,
        index_names: list[str],
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        extract_fragments_flag: bool = False,
        context_lines: int = 3,
        merge_strategy: MergeStrategy | None = None,
    ) -> list[dict[str, Any]]:
        """Search multiple indices with configurable merge strategy.

        Args:
            index_names: List of index names
            query: Search query
            top_k: Number of results to return after fusion
            rrf_k: RRF k parameter (default: 60, used if merge_strategy is None)
            extract_fragments_flag: If True, extract text fragments with matches
            context_lines: Number of context lines before/after matches
            merge_strategy: Optional merge strategy (defaults to RRF if None)

        Returns:
            List of merged result dictionaries

        Raises:
            ValueError: If any index doesn't exist
        """
        # Default to RRF strategy if not provided (backward compatible)
        if merge_strategy is None:
            merge_strategy = RRFMergeStrategy(k=rrf_k)
            logger.info("Searching %d indices with RRF fusion (k=%d)", len(index_names), rrf_k)
        else:
            strategy_name = type(merge_strategy).__name__
            logger.info("Searching %d indices with %s merge", len(index_names), strategy_name)

        # Search each index (fetch more results for fusion)
        fetch_k = max(100, top_k * 3)
        all_results = []

        for idx_name in index_names:
            try:
                results = self.search_single(
                    idx_name,
                    query,
                    top_k=fetch_k,
                    extract_fragments_flag=extract_fragments_flag,
                    context_lines=context_lines,
                )
                all_results.append(results)
                logger.debug("Index '%s': %d results", idx_name, len(results))
            except Exception as e:
                logger.warning("Failed to search index '%s': %s", idx_name, e)
                continue

        if not all_results:
            logger.error("No indices returned results")
            return []

        # Merge results using strategy
        logger.debug("Merging results with %s", type(merge_strategy).__name__)
        output = merge_strategy.merge(all_results, top_k)

        logger.info("Merge complete: %d results", len(output))
        return output
