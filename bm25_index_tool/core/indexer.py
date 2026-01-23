"""BM25 indexing functionality for BM25 index tool.

Uses SQLite FTS5 for full-text search with BM25 ranking.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm  # type: ignore

from bm25_index_tool.config.models import BM25Params, IndexMetadata, TokenizationConfig
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage, compute_file_hash

logger = get_logger(__name__)


class BM25Indexer:
    """Creates BM25 indices using SQLite FTS5."""

    def __init__(self) -> None:
        """Initialize the BM25 indexer."""
        self.registry = IndexRegistry()

    def create_index(
        self,
        name: str,
        files: list[Path],
        params: BM25Params,
        tokenization: TokenizationConfig,
        glob_pattern: str,
    ) -> IndexMetadata:
        """Create a BM25 index.

        Args:
            name: Index name
            files: List of files to index
            params: BM25 parameters (stored for compatibility, FTS5 uses defaults)
            tokenization: Tokenization configuration (stored for compatibility)
            glob_pattern: Original glob pattern used

        Returns:
            IndexMetadata for the created index

        Raises:
            ValueError: If index already exists
        """
        # Check if index already exists
        if self.registry.index_exists(name):
            logger.error("Index '%s' already exists", name)
            raise ValueError(f"Index '{name}' already exists. Use update command instead.")

        logger.info("Creating index '%s' with %d files", name, len(files))

        # Create SQLite storage
        storage = SQLiteStorage(name)
        storage.create_schema(with_vectors=False)

        # Store glob pattern in metadata
        storage.set_metadata("glob_pattern", glob_pattern)

        # Index files with progress bar
        indexed_count = 0
        for file_path in tqdm(files, desc="Indexing files", unit="file"):
            try:
                # Read content
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Compute MD5 hash
                md5_hash = compute_file_hash(file_path)

                # Get file size
                file_size = file_path.stat().st_size

                # Add document to storage
                storage.add_document(
                    path=str(file_path),
                    filename=file_path.name,
                    md5_hash=md5_hash,
                    content=content,
                    mime_type="text/plain",
                    file_size=file_size,
                )
                indexed_count += 1

            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)
                continue

        logger.debug("Indexed %d files successfully", indexed_count)

        # Close storage connection
        storage.close()

        # Create metadata
        index_dir = get_index_dir(name)
        metadata = IndexMetadata(
            name=name,
            created_at=datetime.now(),
            file_count=indexed_count,
            glob_pattern=glob_pattern,
            bm25_params=params,
            tokenization=tokenization,
        )

        # Save metadata to JSON file
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)
        logger.debug("Saved metadata to %s", metadata_path)

        # Register index
        self.registry.add_index(name, metadata.model_dump(mode="json"))

        logger.info("Index '%s' created successfully", name)
        return metadata

    def update_metadata(self, name: str, metadata: IndexMetadata) -> None:
        """Update metadata for an existing index.

        Args:
            name: Index name
            metadata: Updated metadata

        Raises:
            ValueError: If index doesn't exist
        """
        if not self.registry.index_exists(name):
            raise ValueError(f"Index '{name}' not found")

        index_dir = get_index_dir(name)
        metadata_path = index_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

        self.registry.add_index(name, metadata.model_dump(mode="json"))
        logger.debug("Updated metadata for index '%s'", name)

    def update_index(
        self,
        name: str,
        files: list[Path],
    ) -> IndexMetadata:
        """Update an existing index by re-indexing.

        Args:
            name: Index name
            files: List of files to index

        Returns:
            Updated IndexMetadata

        Raises:
            ValueError: If index doesn't exist
        """
        # Load existing metadata
        existing_metadata_dict = self.registry.get_index(name)
        if not existing_metadata_dict:
            logger.error("Index '%s' not found", name)
            raise ValueError(f"Index '{name}' not found")

        existing_metadata = IndexMetadata(**existing_metadata_dict)
        logger.info("Updating index '%s' with %d files", name, len(files))

        # Delete old index directory
        index_dir = get_index_dir(name)
        if index_dir.exists():
            shutil.rmtree(index_dir)
            logger.debug("Removed old index directory: %s", index_dir)

        # Remove from registry
        self.registry.remove_index(name)

        # Create new index with same parameters
        try:
            metadata = self.create_index(
                name=name,
                files=files,
                params=existing_metadata.bm25_params,
                tokenization=existing_metadata.tokenization,
                glob_pattern=existing_metadata.glob_pattern,
            )
            logger.info("Index '%s' updated successfully", name)
            return metadata
        except Exception as e:
            # Re-add old metadata on failure
            self.registry.add_index(name, existing_metadata_dict)
            logger.error("Failed to update index: %s", e)
            raise
