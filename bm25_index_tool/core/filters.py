"""Path filtering for search results.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import fnmatch
from pathlib import Path
from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


class PathFilter:
    """Filter search results by path patterns using glob matching.

    Applies include and exclude patterns to filter document paths.
    Uses fnmatch for glob-style pattern matching.
    """

    def __init__(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize path filter.

        Args:
            include_patterns: List of glob patterns to include (OR logic)
            exclude_patterns: List of glob patterns to exclude (AND NOT logic)
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        logger.debug(
            "PathFilter created with %d include patterns, %d exclude patterns",
            len(self.include_patterns),
            len(self.exclude_patterns),
        )

    def matches(self, path: str) -> bool:
        """Check if path matches filter criteria.

        Logic:
        1. If include_patterns exist, path must match at least one
        2. If exclude_patterns exist, path must not match any

        Args:
            path: Document path to check

        Returns:
            True if path passes filter, False otherwise
        """
        # Convert to Path for consistent separator handling
        path_obj = Path(path)
        path_str = str(path_obj)

        # Check include patterns (OR logic)
        if self.include_patterns:
            include_match = any(
                fnmatch.fnmatch(path_str, pattern) for pattern in self.include_patterns
            )
            if not include_match:
                logger.debug("Path %s excluded (no include pattern match)", path_str)
                return False

        # Check exclude patterns (AND NOT logic)
        if self.exclude_patterns:
            exclude_match = any(
                fnmatch.fnmatch(path_str, pattern) for pattern in self.exclude_patterns
            )
            if exclude_match:
                logger.debug("Path %s excluded (exclude pattern match)", path_str)
                return False

        return True

    def filter_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter list of search results by path.

        Args:
            results: List of result dictionaries with 'path' key

        Returns:
            Filtered list of results
        """
        if not self.include_patterns and not self.exclude_patterns:
            logger.debug("No filter patterns, returning all results")
            return results

        filtered = [result for result in results if self.matches(result["path"])]

        logger.info(
            "PathFilter: %d results → %d results (%d filtered out)",
            len(results),
            len(filtered),
            len(results) - len(filtered),
        )

        return filtered
