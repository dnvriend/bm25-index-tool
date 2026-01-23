"""Programmatic API for BM25 index tool.

This module provides a clean Python API for programmatic access to BM25 indexing
and search functionality. It wraps the core modules (indexer, searcher, etc.) and
provides a Facade pattern for easy integration into Python applications.

Example:
    Basic usage:

    >>> from bm25_index_tool import BM25Client
    >>> client = BM25Client()
    >>>
    >>> # Create an index
    >>> client.create_index("myindex", "/path/to/docs", "**/*.md")
    >>>
    >>> # Search
    >>> results = client.search("myindex", "kubernetes networking", top_k=10)
    >>> for result in results:
    ...     print(f"{result['score']:.2f} - {result['path']}")

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from pathlib import Path
from typing import Any

from bm25_index_tool.config.models import BM25Params, BM25Profile, TokenizationConfig
from bm25_index_tool.core.cache import SearchCache
from bm25_index_tool.core.file_discovery import discover_files
from bm25_index_tool.core.filters import PathFilter
from bm25_index_tool.core.history import SearchHistory
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.core.related import RelatedDocumentFinder
from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry

logger = get_logger(__name__)


class BM25Client:
    """Programmatic API client for BM25 indexing and search.

    This class provides a Facade pattern wrapping all CLI functionality for
    programmatic access. All methods return Python dicts/lists for easy integration.

    The client supports:
    - Index creation and management (create, update, delete)
    - Single and multi-index search with RRF fusion
    - Related document search (find similar documents)
    - Batch search (parallel or sequential)
    - Search result caching (LRU cache)
    - Search history tracking
    - Index statistics and metadata

    Example:
        >>> client = BM25Client()
        >>>
        >>> # Create index
        >>> client.create_index(
        ...     name="myindex",
        ...     path="/path/to/docs",
        ...     glob_pattern="**/*.md"
        ... )
        >>>
        >>> # Search with filters
        >>> results = client.search(
        ...     index="myindex",
        ...     query="kubernetes networking",
        ...     top_k=10,
        ...     path_filter=["reference/**"],
        ...     include_content=True
        ... )
        >>>
        >>> # Multi-index search
        >>> results = client.search_multi(
        ...     indices=["idx1", "idx2"],
        ...     query="docker",
        ...     merge_strategy="rrf"
        ... )
        >>>
        >>> # Related documents
        >>> related = client.search_related(
        ...     index="myindex",
        ...     document_path="docs/intro.md",
        ...     top_k=5
        ... )
        >>>
        >>> # Batch search
        >>> queries = ["query1", "query2", "query3"]
        >>> results = client.batch_search(
        ...     index="myindex",
        ...     queries=queries,
        ...     parallel=True
        ... )

    Attributes:
        indexer: BM25Indexer instance for index operations
        searcher: BM25Searcher instance for search operations
        registry: IndexRegistry for metadata management
        cache: Optional SearchCache for result caching (if enabled)
        history: Optional SearchHistory for query tracking (if enabled)
    """

    def __init__(
        self,
        enable_cache: bool = False,
        cache_max_size: int = 100,
        enable_history: bool = False,
    ) -> None:
        """Initialize the BM25 client.

        Args:
            enable_cache: Enable search result caching (default: False)
            cache_max_size: Maximum cache size for LRU eviction (default: 100)
            enable_history: Enable search history tracking (default: False)
        """
        self.indexer = BM25Indexer()
        self.searcher = BM25Searcher()
        self.registry = IndexRegistry()

        # Optional features
        self.cache: SearchCache | None = None
        self.history: SearchHistory | None = None

        if enable_cache:
            self.cache = SearchCache(max_size=cache_max_size)
            logger.debug("Cache enabled with max_size=%d", cache_max_size)

        if enable_history:
            self.history = SearchHistory()
            logger.debug("History tracking enabled")

    def create_index(
        self,
        name: str,
        path: str,
        glob_pattern: str,
        k1: float | None = None,
        b: float | None = None,
        method: str = "lucene",
        profile: str = "standard",
        stemmer: str = "",
        stopwords: str = "en",
        respect_gitignore: bool = True,
    ) -> dict[str, Any]:
        """Create a new BM25 index.

        Args:
            name: Index name (must be unique)
            path: Base directory path to index
            glob_pattern: Glob pattern for file discovery (e.g., "**/*.md")
            k1: BM25 k1 parameter (default: from profile, typically 1.5)
            b: BM25 b parameter (default: from profile, typically 0.75)
            method: BM25 method variant (default: "lucene")
            profile: BM25 parameter profile: "standard" or "code" (default: "standard")
            stemmer: Stemmer language code (e.g., "english"), empty string to disable
            stopwords: Stopwords language code (default: "en")
            respect_gitignore: Respect .gitignore files during discovery (default: True)

        Returns:
            Dictionary with index metadata:
                - name: Index name
                - created_at: Timestamp
                - file_count: Number of files indexed
                - glob_pattern: Original glob pattern
                - bm25_params: BM25 parameters (k1, b, method)
                - tokenization: Tokenization config (stemmer, stopwords)

        Raises:
            ValueError: If index already exists or no files found
            Exception: If indexing fails

        Example:
            >>> client.create_index(
            ...     name="docs",
            ...     path="/home/user/documents",
            ...     glob_pattern="**/*.txt",
            ...     k1=1.2,
            ...     b=0.5,
            ...     stemmer="english"
            ... )
        """
        logger.info("Creating index '%s' with pattern '%s'", name, glob_pattern)

        # Determine BM25 parameters
        if k1 is not None and b is not None:
            params = BM25Params(k1=k1, b=b, method=method)
            logger.debug("Using custom BM25 parameters: k1=%s, b=%s", k1, b)
        else:
            try:
                bm25_profile = BM25Profile(profile)
                params = BM25Params.from_profile(bm25_profile)
                params.method = method
                logger.debug("Using profile '%s': k1=%s, b=%s", profile, params.k1, params.b)
            except ValueError:
                logger.warning("Invalid profile '%s', using 'standard'", profile)
                params = BM25Params.from_profile(BM25Profile.STANDARD)
                params.method = method

        # Tokenization config
        tokenization = TokenizationConfig(stemmer=stemmer, stopwords=stopwords)

        # Discover files
        full_pattern = str(Path(path) / glob_pattern)
        logger.debug("Full pattern: %s", full_pattern)
        files = discover_files(full_pattern, respect_gitignore=respect_gitignore)
        logger.info("Discovered %d files to index", len(files))

        # Create index
        metadata = self.indexer.create_index(
            name=name,
            files=files,
            params=params,
            tokenization=tokenization,
            glob_pattern=full_pattern,
        )

        logger.info("Index '%s' created successfully with %d files", name, metadata.file_count)
        return metadata.model_dump(mode="json")

    def search(
        self,
        index: str,
        query: str,
        top_k: int = 10,
        fragments: bool = False,
        context: int = 3,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
        include_content: bool = False,
        content_max_length: int = 500,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Search a single BM25 index.

        Args:
            index: Index name
            query: Search query string
            top_k: Number of results to return (default: 10)
            fragments: Extract text fragments showing matched terms (default: False)
            context: Number of context lines before/after matches (default: 3)
            path_filter: Include only paths matching glob patterns (e.g., ["docs/**"])
            exclude_path: Exclude paths matching glob patterns (e.g., ["test/**"])
            include_content: Include full document content in results (default: False)
            content_max_length: Maximum content length in characters (default: 500)
            use_cache: Use cached results if available (default: True)

        Returns:
            List of result dictionaries, each containing:
                - path: Document path
                - name: Document filename
                - score: BM25 relevance score
                - content: Full document content (if include_content=True)
                - fragments: Text fragments with matches (if fragments=True)

        Raises:
            ValueError: If index doesn't exist
            Exception: If search fails

        Example:
            >>> results = client.search(
            ...     index="docs",
            ...     query="machine learning",
            ...     top_k=5,
            ...     path_filter=["tutorials/**"],
            ...     fragments=True
            ... )
            >>> for r in results:
            ...     print(f"{r['score']:.2f} - {r['path']}")
        """
        logger.info("Searching index '%s' for: '%s'", index, query)

        # Check cache
        if use_cache and self.cache and not fragments and not path_filter and not exclude_path:
            cached = self.cache.get(
                indices=[index],
                query=query,
                top_k=top_k,
                path_filter=None,
                exclude_path=None,
            )
            if cached is not None:
                logger.info("Cache hit for query: '%s'", query)
                results = cached
            else:
                # Execute search
                results = self.searcher.search_single(
                    index_name=index,
                    query=query,
                    top_k=top_k,
                    extract_fragments_flag=fragments,
                    context_lines=context,
                )
                # Store in cache
                self.cache.set(
                    indices=[index],
                    query=query,
                    top_k=top_k,
                    results=results,
                    path_filter=None,
                    exclude_path=None,
                )
        else:
            # Execute search
            results = self.searcher.search_single(
                index_name=index,
                query=query,
                top_k=top_k,
                extract_fragments_flag=fragments,
                context_lines=context,
            )

        # Apply path filtering
        if path_filter or exclude_path:
            filter_obj = PathFilter(
                include_patterns=path_filter if path_filter else None,
                exclude_patterns=exclude_path if exclude_path else None,
            )
            results = filter_obj.filter_results(results)

        # Truncate content if needed
        if not include_content:
            for result in results:
                if "content" in result:
                    content = result["content"]
                    if len(content) > content_max_length:
                        result["content"] = content[:content_max_length] + "..."
                    else:
                        # Keep full content if under limit
                        pass

        logger.info("Search complete: %d results", len(results))
        return results

    def search_multi(
        self,
        indices: list[str],
        query: str,
        top_k: int = 10,
        merge_strategy: str = "rrf",
        rrf_k: int = 60,
        fragments: bool = False,
        context: int = 3,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
        include_content: bool = False,
        content_max_length: int = 500,
    ) -> list[dict[str, Any]]:
        """Search multiple indices with result fusion.

        Args:
            indices: List of index names
            query: Search query string
            top_k: Number of results to return after fusion (default: 10)
            merge_strategy: Merge strategy: "rrf" (Reciprocal Rank Fusion)
            rrf_k: RRF k parameter for fusion (default: 60)
            fragments: Extract text fragments showing matched terms (default: False)
            context: Number of context lines before/after matches (default: 3)
            path_filter: Include only paths matching glob patterns
            exclude_path: Exclude paths matching glob patterns
            include_content: Include full document content in results (default: False)
            content_max_length: Maximum content length in characters (default: 500)

        Returns:
            List of fused result dictionaries with same structure as search()

        Raises:
            ValueError: If any index doesn't exist
            Exception: If search fails

        Example:
            >>> results = client.search_multi(
            ...     indices=["docs", "wiki"],
            ...     query="kubernetes",
            ...     top_k=10,
            ...     rrf_k=60
            ... )
        """
        logger.info("Multi-index search across %d indices: %s", len(indices), indices)

        if merge_strategy != "rrf":
            logger.warning("Only 'rrf' merge strategy is supported, using rrf")

        # Execute multi-index search with RRF
        results = self.searcher.search_multi(
            index_names=indices,
            query=query,
            top_k=top_k,
            rrf_k=rrf_k,
            extract_fragments_flag=fragments,
            context_lines=context,
        )

        # Apply path filtering
        if path_filter or exclude_path:
            filter_obj = PathFilter(
                include_patterns=path_filter if path_filter else None,
                exclude_patterns=exclude_path if exclude_path else None,
            )
            results = filter_obj.filter_results(results)

        # Truncate content if needed
        if not include_content:
            for result in results:
                if "content" in result:
                    content = result["content"]
                    if len(content) > content_max_length:
                        result["content"] = content[:content_max_length] + "..."

        logger.info("Multi-index search complete: %d results", len(results))
        return results

    def search_related(
        self,
        index: str,
        document_path: str,
        top_k: int = 10,
        fragments: bool = False,
        context: int = 3,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
        include_content: bool = False,
        content_max_length: int = 500,
    ) -> list[dict[str, Any]]:
        """Find documents related to a given document.

        Uses the document's content as the query to find similar documents.

        Args:
            index: Index name
            document_path: Path to document (relative to index)
            top_k: Number of results to return (default: 10)
            fragments: Extract text fragments showing matched terms (default: False)
            context: Number of context lines before/after matches (default: 3)
            path_filter: Include only paths matching glob patterns
            exclude_path: Exclude paths matching glob patterns
            include_content: Include full document content in results (default: False)
            content_max_length: Maximum content length in characters (default: 500)

        Returns:
            List of related document dictionaries with same structure as search()

        Raises:
            ValueError: If index or document doesn't exist
            Exception: If search fails

        Example:
            >>> related = client.search_related(
            ...     index="docs",
            ...     document_path="intro.md",
            ...     top_k=5
            ... )
        """
        logger.info("Finding documents related to '%s' in index '%s'", document_path, index)

        finder = RelatedDocumentFinder()
        results = finder.find_related(
            index_name=index,
            document_path=document_path,
            top_k=top_k,
            extract_fragments_flag=fragments,
            context_lines=context,
        )

        # Apply path filtering
        if path_filter or exclude_path:
            filter_obj = PathFilter(
                include_patterns=path_filter if path_filter else None,
                exclude_patterns=exclude_path if exclude_path else None,
            )
            results = filter_obj.filter_results(results)

        # Truncate content if needed
        if not include_content:
            for result in results:
                if "content" in result:
                    content = result["content"]
                    if len(content) > content_max_length:
                        result["content"] = content[:content_max_length] + "..."

        logger.info("Found %d related documents", len(results))
        return results

    def batch_search(
        self,
        index: str | list[str],
        queries: list[str],
        top_k: int = 10,
        parallel: bool = False,
        max_workers: int = 4,
        rrf_k: int = 60,
        fragments: bool = False,
        context: int = 3,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute batch queries.

        Args:
            index: Index name or list of index names
            queries: List of query strings
            top_k: Number of results to return per query (default: 10)
            parallel: Enable parallel execution (default: False)
            max_workers: Maximum number of parallel workers (default: 4)
            rrf_k: RRF k parameter for multi-index fusion (default: 60)
            fragments: Extract text fragments (default: False)
            context: Number of context lines (default: 3)
            path_filter: Include only paths matching glob patterns
            exclude_path: Exclude paths matching glob patterns

        Returns:
            List of result dictionaries, one per query:
                - query: Original query string
                - results: List of search results
                - count: Number of results
                - execution_time: Query execution time in seconds

        Raises:
            ValueError: If index doesn't exist
            Exception: If search fails

        Example:
            >>> queries = ["query1", "query2", "query3"]
            >>> results = client.batch_search(
            ...     index="docs",
            ...     queries=queries,
            ...     parallel=True,
            ...     max_workers=8
            ... )
            >>> for r in results:
            ...     print(f"{r['query']}: {r['count']} results in {r['execution_time']:.3f}s")
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Batch search: %d queries (parallel=%s)", len(queries), parallel)

        # Normalize index to list
        index_names = [index] if isinstance(index, str) else index

        results_list = []

        def _execute_query(query: str) -> dict[str, Any]:
            start = time.time()
            try:
                if len(index_names) == 1:
                    search_results = self.searcher.search_single(
                        index_names[0],
                        query,
                        top_k=top_k,
                        extract_fragments_flag=fragments,
                        context_lines=context,
                    )
                else:
                    search_results = self.searcher.search_multi(
                        index_names,
                        query,
                        top_k=top_k,
                        rrf_k=rrf_k,
                        extract_fragments_flag=fragments,
                        context_lines=context,
                    )

                # Apply path filtering
                if path_filter or exclude_path:
                    filter_obj = PathFilter(
                        include_patterns=path_filter if path_filter else None,
                        exclude_patterns=exclude_path if exclude_path else None,
                    )
                    search_results = filter_obj.filter_results(search_results)

            except Exception as e:
                logger.warning("Query failed: %s - %s", query, e)
                search_results = []

            elapsed = time.time() - start
            return {
                "query": query,
                "results": search_results,
                "count": len(search_results),
                "execution_time": elapsed,
            }

        if parallel:
            # Parallel execution
            logger.debug("Using %d parallel workers", max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_execute_query, q): q for q in queries}
                for future in as_completed(futures):
                    results_list.append(future.result())
        else:
            # Sequential execution
            for query in queries:
                results_list.append(_execute_query(query))

        logger.info("Batch search complete: %d queries processed", len(queries))
        return results_list

    def get_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about an index.

        Args:
            name: Index name

        Returns:
            Dictionary with index metadata:
                - name: Index name
                - created_at: Creation timestamp
                - file_count: Number of files indexed
                - glob_pattern: Original glob pattern
                - bm25_params: BM25 parameters
                - tokenization: Tokenization configuration

        Raises:
            ValueError: If index doesn't exist

        Example:
            >>> info = client.get_info("docs")
            >>> print(f"Index: {info['name']}")
            >>> print(f"Files: {info['file_count']}")
        """
        logger.debug("Getting info for index: %s", name)
        metadata = self.registry.get_index(name)
        if not metadata:
            raise ValueError(f"Index '{name}' not found")
        return metadata

    def get_stats(self, name: str, detailed: bool = False) -> dict[str, Any]:
        """Get statistics for an index.

        Args:
            name: Index name
            detailed: Compute detailed statistics (loads index, slower) (default: False)

        Returns:
            Dictionary with statistics:
                - name: Index name
                - created_at: Creation timestamp
                - file_count: Number of files
                - glob_pattern: Glob pattern
                - storage_size_bytes: Storage size in bytes
                - storage_size_formatted: Human-readable storage size
                - bm25_params: BM25 parameters
                - tokenization: Tokenization config

            If detailed=True, also includes:
                - document_count: Number of documents in database
                - chunk_count: Number of chunks (if vector index exists)
                - has_vector_index: Whether vector index exists

        Raises:
            ValueError: If index doesn't exist
            Exception: If statistics computation fails

        Example:
            >>> stats = client.get_stats("docs", detailed=True)
            >>> print(f"Documents: {stats['document_count']}")
        """
        from bm25_index_tool.storage.sqlite_storage import SQLiteStorage

        logger.debug("Computing statistics for index: %s (detailed=%s)", name, detailed)

        metadata = self.registry.get_index(name)
        if not metadata:
            raise ValueError(f"Index '{name}' not found")

        index_dir = get_index_dir(name)
        storage_bytes = self._get_directory_size(index_dir)

        stats: dict[str, Any] = {
            "name": name,
            "created_at": metadata["created_at"],
            "file_count": metadata["file_count"],
            "glob_pattern": metadata["glob_pattern"],
            "bm25_params": metadata["bm25_params"],
            "tokenization": metadata["tokenization"],
            "storage_size_bytes": storage_bytes,
            "storage_size_formatted": self._format_size(storage_bytes),
        }

        if detailed:
            # Load SQLite storage for detailed statistics
            with SQLiteStorage(name) as storage:
                stats["document_count"] = storage.get_document_count()
                stats["chunk_count"] = storage.get_chunk_count()
                stats["has_vector_index"] = storage.has_vector_index()

                # Get all metadata from storage
                storage_metadata = storage.get_all_metadata()
                if storage_metadata:
                    stats["storage_metadata"] = storage_metadata

        return stats

    def list_indices(self) -> list[dict[str, Any]]:
        """List all registered indices.

        Returns:
            List of index metadata dictionaries

        Example:
            >>> indices = client.list_indices()
            >>> for idx in indices:
            ...     print(f"{idx['name']}: {idx['file_count']} files")
        """
        logger.debug("Listing all indices")
        names = self.registry.list_indices()
        indices = []
        for name in names:
            metadata = self.registry.get_index(name)
            if metadata:
                indices.append(metadata)
        return indices

    def update_index(self, name: str) -> dict[str, Any]:
        """Update an existing index by re-indexing.

        Re-discovers files using the original glob pattern and re-indexes them
        with the same BM25 parameters.

        Args:
            name: Index name

        Returns:
            Dictionary with updated index metadata

        Raises:
            ValueError: If index doesn't exist or no files found
            Exception: If re-indexing fails

        Example:
            >>> metadata = client.update_index("docs")
            >>> print(f"Updated: {metadata['file_count']} files")
        """
        logger.info("Updating index: %s", name)

        # Get existing metadata
        metadata_dict = self.registry.get_index(name)
        if not metadata_dict:
            raise ValueError(f"Index '{name}' not found")

        glob_pattern = metadata_dict["glob_pattern"]
        logger.debug("Re-discovering files with pattern: %s", glob_pattern)

        # Discover files
        files = discover_files(glob_pattern, respect_gitignore=True)
        logger.info("Found %d files to re-index", len(files))

        # Update index
        updated_metadata = self.indexer.update_index(name, files)
        logger.info("Index '%s' updated successfully", name)

        return updated_metadata.model_dump(mode="json")

    def delete_index(self, name: str) -> None:
        """Delete an index.

        Removes the index directory and registry entry.

        Args:
            name: Index name

        Raises:
            ValueError: If index doesn't exist

        Example:
            >>> client.delete_index("docs")
        """
        import shutil

        logger.info("Deleting index: %s", name)

        if not self.registry.index_exists(name):
            raise ValueError(f"Index '{name}' not found")

        # Delete index directory
        index_dir = get_index_dir(name)
        if index_dir.exists():
            shutil.rmtree(index_dir)
            logger.debug("Removed index directory: %s", index_dir)

        # Remove from registry
        self.registry.remove_index(name)
        logger.info("Index '%s' deleted successfully", name)

    def enable_cache(self, max_size: int = 100) -> None:
        """Enable search result caching.

        Args:
            max_size: Maximum cache size for LRU eviction (default: 100)

        Example:
            >>> client.enable_cache(max_size=200)
        """
        self.cache = SearchCache(max_size=max_size)
        logger.info("Cache enabled with max_size=%d", max_size)

    def disable_cache(self) -> None:
        """Disable search result caching.

        Example:
            >>> client.disable_cache()
        """
        if self.cache:
            self.cache.clear()
        self.cache = None
        logger.info("Cache disabled")

    def clear_cache(self) -> None:
        """Clear all cached search results.

        Example:
            >>> client.clear_cache()
        """
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
        else:
            logger.warning("Cache is not enabled")

    def get_cache_stats(self) -> dict[str, int] | None:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats or None if cache disabled:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current cache size
                - max_size: Maximum cache size

        Example:
            >>> stats = client.get_cache_stats()
            >>> if stats:
            ...     print(f"Hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.2%}")
        """
        if self.cache:
            return self.cache.stats()
        return None

    def enable_history(self) -> None:
        """Enable search history tracking.

        Example:
            >>> client.enable_history()
        """
        self.history = SearchHistory()
        logger.info("History tracking enabled")

    def disable_history(self) -> None:
        """Disable search history tracking.

        Example:
            >>> client.disable_history()
        """
        self.history = None
        logger.info("History tracking disabled")

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent search history.

        Args:
            limit: Maximum number of entries to return (default: 20)

        Returns:
            List of history entry dictionaries:
                - id: Entry ID
                - timestamp: ISO timestamp
                - indices: List of index names
                - query: Query string
                - top_k: Number of results requested
                - result_count: Actual results returned
                - elapsed_seconds: Execution time
                - path_filter: Path filter patterns (if any)
                - exclude_path: Exclude patterns (if any)

        Example:
            >>> history = client.get_history(limit=10)
            >>> for entry in history:
            ...     print(f"{entry['timestamp']}: {entry['query']}")
        """
        if not self.history:
            logger.warning("History tracking is not enabled")
            return []

        return self.history.get_recent(limit=limit)

    def clear_history(self) -> int:
        """Clear all search history.

        Returns:
            Number of entries deleted

        Example:
            >>> deleted = client.clear_history()
            >>> print(f"Deleted {deleted} history entries")
        """
        if not self.history:
            logger.warning("History tracking is not enabled")
            return 0

        return self.history.clear()

    @staticmethod
    def _get_directory_size(path: Path) -> float:
        """Calculate total size of directory in bytes."""
        total_size: float = 0.0
        if path.exists() and path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    total_size += float(item.stat().st_size)
        return total_size

    @staticmethod
    def _format_size(size_bytes: float) -> str:
        """Format byte size as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
