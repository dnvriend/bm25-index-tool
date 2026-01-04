"""Search cache with LRU eviction for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


class SearchCache:
    """Thread-safe LRU cache for search results.

    Caches search results using a hash of (indices, query, top_k, filters) as key.
    Uses OrderedDict for O(1) get/set with LRU eviction.
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize search cache.

        Args:
            max_size: Maximum number of cache entries (default: 100)
        """
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        logger.debug("SearchCache initialized with max_size=%d", max_size)

    def _generate_key(
        self,
        indices: list[str],
        query: str,
        top_k: int,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
    ) -> str:
        """Generate cache key from search parameters.

        Args:
            indices: List of index names
            query: Search query
            top_k: Number of results
            path_filter: Include path patterns
            exclude_path: Exclude path patterns

        Returns:
            SHA256 hash of parameters as hex string
        """
        # Create deterministic JSON representation
        params = {
            "indices": sorted(indices),  # Sort for deterministic key
            "query": query,
            "top_k": top_k,
            "path_filter": sorted(path_filter) if path_filter else None,
            "exclude_path": sorted(exclude_path) if exclude_path else None,
        }
        params_json = json.dumps(params, sort_keys=True)

        # Generate SHA256 hash
        return hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    def get(
        self,
        indices: list[str],
        query: str,
        top_k: int,
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get cached search results.

        Args:
            indices: List of index names
            query: Search query
            top_k: Number of results
            path_filter: Include path patterns
            exclude_path: Exclude path patterns

        Returns:
            Cached results or None if not found
        """
        key = self._generate_key(indices, query, top_k, path_filter, exclude_path)

        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(
                    "Cache HIT: %s (hits=%d, misses=%d)", key[:16], self._hits, self._misses
                )
                return self._cache[key]
            else:
                self._misses += 1
                logger.debug(
                    "Cache MISS: %s (hits=%d, misses=%d)", key[:16], self._hits, self._misses
                )
                return None

    def set(
        self,
        indices: list[str],
        query: str,
        top_k: int,
        results: list[dict[str, Any]],
        path_filter: list[str] | None = None,
        exclude_path: list[str] | None = None,
    ) -> None:
        """Store search results in cache.

        Args:
            indices: List of index names
            query: Search query
            top_k: Number of results
            results: Search results to cache
            path_filter: Include path patterns
            exclude_path: Exclude path patterns
        """
        key = self._generate_key(indices, query, top_k, path_filter, exclude_path)

        with self._lock:
            # Add/update entry
            if key in self._cache:
                # Move to end if already exists
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = results

                # Evict oldest if over max size
                if len(self._cache) > self._max_size:
                    evicted_key = next(iter(self._cache))
                    del self._cache[evicted_key]
                    logger.debug("Cache eviction: %s (size=%d)", evicted_key[:16], len(self._cache))

            logger.debug("Cache SET: %s (size=%d)", key[:16], len(self._cache))

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared (%d entries removed)", size)

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, max_size
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
            }
