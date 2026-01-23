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


def expand_braces(pattern: str) -> list[str]:
    """Expand brace patterns like {md,jpg,png} into multiple patterns.

    Args:
        pattern: Pattern potentially containing brace expansions

    Returns:
        List of expanded patterns

    Examples:
        "**/*.{md,txt}" -> ["**/*.md", "**/*.txt"]
        "**/*.md" -> ["**/*.md"]
        "{src,lib}/**/*.py" -> ["src/**/*.py", "lib/**/*.py"]
    """
    # Find brace pattern
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    # Get the alternatives inside braces
    alternatives = match.group(1).split(",")
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]

    # Recursively expand any remaining braces
    expanded = []
    for alt in alternatives:
        sub_pattern = prefix + alt.strip() + suffix
        expanded.extend(expand_braces(sub_pattern))

    return expanded


def expand_pattern_to_absolute(pattern: str) -> str:
    """Expand a glob pattern to an absolute path.

    Expands environment variables, tilde, and relative paths to create
    a fully qualified absolute pattern suitable for index storage.

    Args:
        pattern: Glob pattern which may contain:
                 - Environment variables: $HOME, $OBSIDIAN_HOME, ${VAR}
                 - Tilde: ~, ~user
                 - Relative paths: ., .., ./path, ../path
                 - Relative globs: **/*.md, *.py

    Returns:
        Absolute pattern with all expansions applied

    Examples:
        "**/*.md" -> "/Users/dennis/projects/**/*.md"
        "./**/*.md" -> "/Users/dennis/projects/**/*.md"
        "$HOME/vault/**/*.md" -> "/Users/dennis/vault/**/*.md"
        "~/docs/*.txt" -> "/Users/dennis/docs/*.txt"
    """
    logger.debug("Expanding pattern to absolute: %s", pattern)

    # 1. Expand environment variables first (e.g., $HOME, $OBSIDIAN_HOME, ${VAR})
    expanded = os.path.expandvars(pattern)

    # 2. Expand user home directory (~, ~user)
    expanded = os.path.expanduser(expanded)

    # 3. Handle relative paths and patterns
    # Check if pattern starts with . or .. or is a relative glob
    if expanded.startswith("./"):
        # Remove leading ./ and prepend CWD
        expanded = str(Path.cwd() / expanded[2:])
    elif expanded.startswith("../"):
        # Resolve .. relative to CWD
        expanded = str((Path.cwd() / expanded).resolve())
    elif not expanded.startswith("/"):
        # Relative pattern without ./ prefix (e.g., "**/*.md", "src/**/*.py")
        expanded = str(Path.cwd() / expanded)

    logger.debug("Expanded pattern: %s", expanded)
    return expanded


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
    brace expansion ({md,txt}), and standard glob patterns (*, **, ?).

    Args:
        pattern: Glob pattern supporting:
                 - Simple: "*.md"
                 - Recursive: "**/*.py"
                 - Absolute: "/path/to/**/*.md"
                 - Tilde: "~/docs/**/*.md"
                 - Env vars: "$HOME/docs/**/*.md", "${PROJECT_DIR}/**/*.py"
                 - Brace expansion: "**/*.{md,txt,jpg}"
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

    # 3. Expand brace patterns (e.g., {md,txt,jpg})
    patterns = expand_braces(pattern)
    if len(patterns) > 1:
        logger.debug("Expanded to %d patterns: %s", len(patterns), patterns)

        # Collect files from all patterns
        all_paths: set[Path] = set()
        for sub_pattern in patterns:
            try:
                sub_paths = _discover_files_single(sub_pattern, respect_gitignore)
                all_paths.update(sub_paths)
            except ValueError:
                # Pattern found no files, continue with others
                pass

        if not all_paths:
            raise ValueError(f"No files found matching: {pattern}")

        paths = sorted(all_paths, key=natural_sort_key)
        logger.info("Discovered %d files from %d patterns", len(paths), len(patterns))
        return paths

    # Single pattern - use existing logic
    return _discover_files_single(pattern, respect_gitignore)


def _discover_files_single(pattern: str, respect_gitignore: bool = True) -> list[Path]:
    """Discover files matching a single glob pattern (no brace expansion).

    Args:
        pattern: Glob pattern (already expanded)
        respect_gitignore: If True, respect .gitignore files

    Returns:
        Sorted list of Path objects

    Raises:
        ValueError: If no files match pattern
    """

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
