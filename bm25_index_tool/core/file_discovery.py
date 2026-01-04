"""File discovery utilities for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import os
import re
from pathlib import Path

import pathspec

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


def natural_sort_key(path: Path) -> tuple[float, str]:
    """Generate sort key for natural sorting.

    Extracts numeric portions from filename for natural ordering.

    Args:
        path: Path to file

    Returns:
        Tuple of (first_number_found, filename) for sorting
    """
    filename = path.name
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return (float(numbers[0]), filename)
    return (float("inf"), filename)


def load_gitignore_patterns(base_dir: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from a directory.

    Args:
        base_dir: Directory to search for .gitignore

    Returns:
        PathSpec instance or None if no .gitignore found
    """
    gitignore_path = base_dir / ".gitignore"
    if not gitignore_path.exists():
        return None

    try:
        with open(gitignore_path) as f:
            patterns = f.read().splitlines()
            spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            logger.debug("Loaded %d gitignore patterns from %s", len(patterns), gitignore_path)
            return spec
    except Exception as e:
        logger.warning("Failed to load .gitignore from %s: %s", gitignore_path, e)
        return None


def discover_files(pattern: str, respect_gitignore: bool = True) -> list[Path]:
    """Discover files matching glob pattern.

    Supports tilde expansion (~, ~user), environment variables ($VAR, ${VAR}),
    and standard glob patterns (*, **, ?).

    Args:
        pattern: Glob pattern supporting:
                 - Simple: "*.md"
                 - Recursive: "**/*.py"
                 - Absolute: "/path/to/**/*.md"
                 - Tilde: "~/docs/**/*.md"
                 - Env vars: "$HOME/docs/**/*.md", "${PROJECT_DIR}/**/*.py"
        respect_gitignore: If True, respect .gitignore files

    Returns:
        Sorted list of Path objects

    Raises:
        ValueError: If no files match pattern
    """
    logger.debug("Discovering files with pattern: %s", pattern)

    # Expand tilde (~), home directory, and environment variables
    # 1. Expand environment variables first (e.g., $HOME, ${VAR})
    pattern = os.path.expandvars(pattern)
    # 2. Expand user home directory (~, ~user)
    pattern = os.path.expanduser(pattern)
    logger.debug("Expanded pattern: %s", pattern)

    # Parse glob pattern
    # Split the pattern into base directory and glob pattern
    if pattern.startswith("/"):
        # Absolute path: /Users/dennis/vault/**/*.md
        # Find the first occurrence of ** or * to determine base directory
        parts = pattern.split("/")
        base_parts = []
        glob_parts = []
        found_wildcard = False

        for part in parts:
            if not found_wildcard and ("*" in part or "?" in part):
                found_wildcard = True
            if found_wildcard:
                glob_parts.append(part)
            else:
                base_parts.append(part)

        # Reconstruct base directory and glob pattern
        if base_parts:
            if base_parts[0]:
                base_dir = Path("/".join(base_parts))
            else:
                base_dir = Path("/" + "/".join(base_parts[1:]))
        else:
            base_dir = Path("/")

        glob_pattern = "/".join(glob_parts)
    elif "/" in pattern or "\\" in pattern:
        # Relative path with directory: docs/**/*.md
        base_dir = Path.cwd()
        glob_pattern = pattern
    else:
        # Just a pattern: *.md
        base_dir = Path.cwd()
        glob_pattern = pattern

    logger.debug("Base directory: %s", base_dir)
    logger.debug("Glob pattern: %s", glob_pattern)

    # Find matches
    if not base_dir.exists():
        logger.error("Base directory does not exist: %s", base_dir)
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # Use rglob for ** patterns, glob otherwise
    if "**" in glob_pattern:
        matches = list(base_dir.rglob(glob_pattern.replace("**/", "")))
    else:
        matches = list(base_dir.glob(glob_pattern))

    logger.debug("Found %d raw matches", len(matches))

    # Filter for regular files only
    paths = [p for p in matches if p.is_file()]
    logger.debug("Filtered to %d files", len(paths))

    if not paths:
        logger.error("No files found matching: %s", pattern)
        raise ValueError(f"No files found matching: {pattern}")

    # Apply .gitignore filtering
    if respect_gitignore:
        gitignore_spec = load_gitignore_patterns(base_dir)
        if gitignore_spec:
            original_count = len(paths)
            paths = [
                p for p in paths if not gitignore_spec.match_file(str(p.relative_to(base_dir)))
            ]
            filtered_count = original_count - len(paths)
            if filtered_count > 0:
                logger.debug("Filtered out %d files via .gitignore", filtered_count)

    if not paths:
        logger.error("No files remaining after filtering: %s", pattern)
        raise ValueError(f"No files remaining after filtering: {pattern}")

    # Sort naturally
    paths.sort(key=natural_sort_key)
    logger.info("Discovered %d files", len(paths))

    if logger.isEnabledFor(10):  # DEBUG level
        for idx, path in enumerate(paths[:5], 1):
            logger.debug("  %d. %s", idx, path.name)
        if len(paths) > 5:
            logger.debug("  ... and %d more", len(paths) - 5)

    return paths
