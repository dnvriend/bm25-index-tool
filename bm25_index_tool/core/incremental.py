"""Incremental indexing with MD5 change detection.

Provides the IncrementalIndexer class for detecting file changes
(added, modified, deleted) by comparing MD5 hashes with stored values.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage, compute_file_hash

logger = get_logger(__name__)


@dataclass
class ChangeSet:
    """Represents detected file changes between filesystem and index.

    Attributes:
        added: List of file paths that are new (not in index).
        modified: List of file paths with changed content (MD5 hash differs).
        deleted: List of paths that are in index but no longer exist on filesystem.
    """

    added: list[Path]
    modified: list[Path]
    deleted: list[str]


class IncrementalIndexer:
    """Handles incremental indexing with MD5-based change detection.

    Compares current filesystem state with stored index to detect:
    - Added files (new paths not in index)
    - Modified files (existing paths with changed MD5 hash)
    - Deleted files (indexed paths no longer on filesystem)

    Example:
        >>> incremental = IncrementalIndexer()
        >>> with SQLiteStorage("myindex") as storage:
        ...     changes = incremental.detect_changes(files, storage)
        ...     print(f"Added: {len(changes.added)}")
        ...     print(f"Modified: {len(changes.modified)}")
        ...     print(f"Deleted: {len(changes.deleted)}")
    """

    def compute_file_hash(self, path: Path) -> str:
        """Compute MD5 hash of a file.

        Args:
            path: Path to the file to hash.

        Returns:
            Hexadecimal MD5 hash string.

        Raises:
            FileNotFoundError: If file does not exist.
            PermissionError: If file cannot be read.
        """
        return compute_file_hash(path)

    def detect_changes(self, current_files: list[Path], storage: SQLiteStorage) -> ChangeSet:
        """Compare current files with indexed files to detect changes.

        Args:
            current_files: List of file paths currently on filesystem.
            storage: SQLiteStorage instance to query for stored hashes.

        Returns:
            ChangeSet containing added, modified, and deleted file lists.
        """
        # Get stored paths and hashes from index
        stored_hashes = storage.get_all_paths_with_hashes()
        stored_paths = set(stored_hashes.keys())

        # Convert current files to set of absolute path strings
        current_paths: dict[str, Path] = {str(p.resolve()): p for p in current_files}
        current_path_set = set(current_paths.keys())

        # Detect deleted files (in index but not on filesystem)
        deleted_paths = list(stored_paths - current_path_set)

        # Categorize files as added or modified
        added: list[Path] = []
        modified: list[Path] = []

        for path_str, path in current_paths.items():
            if path_str not in stored_paths:
                # New file - not in index
                added.append(path)
                logger.debug("Detected added file: %s", path_str)
            else:
                # Existing file - check if content changed
                try:
                    current_hash = self.compute_file_hash(path)
                    stored_hash = stored_hashes[path_str]

                    if current_hash != stored_hash:
                        modified.append(path)
                        logger.debug(
                            "Detected modified file: %s (hash: %s -> %s)",
                            path_str,
                            stored_hash[:8],
                            current_hash[:8],
                        )
                except (OSError, PermissionError) as e:
                    logger.warning("Failed to hash file %s: %s", path_str, e)

        # Log deleted files
        for deleted_path in deleted_paths:
            logger.debug("Detected deleted file: %s", deleted_path)

        logger.info(
            "Change detection complete: %d added, %d modified, %d deleted",
            len(added),
            len(modified),
            len(deleted_paths),
        )

        return ChangeSet(added=added, modified=modified, deleted=deleted_paths)
