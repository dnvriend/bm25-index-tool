"""BM25 indexing functionality for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from datetime import datetime
from pathlib import Path

import bm25s  # type: ignore
import Stemmer  # type: ignore
from tqdm import tqdm  # type: ignore

from bm25_index_tool.config.models import BM25Params, IndexMetadata, TokenizationConfig
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry

logger = get_logger(__name__)


class BM25Indexer:
    """Creates BM25 indices."""

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
            params: BM25 parameters
            tokenization: Tokenization configuration
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

        # Read file contents with progress bar
        corpus_text = []
        corpus_metadata = []

        for file_path in tqdm(files, desc="Reading files", unit="file"):
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    corpus_text.append(content)
                    corpus_metadata.append(
                        {"path": str(file_path), "name": file_path.name, "content": content}
                    )
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                continue

        logger.debug("Read %d files successfully", len(corpus_text))

        # Prepare stemmer
        stemmer = None
        if tokenization.stemmer_enabled:
            try:
                stemmer = Stemmer.Stemmer(tokenization.stemmer)
                logger.debug("Using stemmer: %s", tokenization.stemmer)
            except Exception as e:
                logger.warning("Failed to initialize stemmer: %s", e)

        # Tokenize corpus
        logger.info("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(
            corpus_text, stopwords=tokenization.stopwords, stemmer=stemmer
        )
        logger.debug("Tokenization complete")

        # Create BM25 retriever
        logger.info("Building BM25 index...")
        retriever = bm25s.BM25(method=params.method, k1=params.k1, b=params.b)
        retriever.index(corpus_tokens)
        logger.debug("BM25 index built")

        # Save index
        index_dir = get_index_dir(name)
        index_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving index to %s", index_dir)
        retriever.save(str(index_dir / "bm25s"), corpus=corpus_metadata)

        # Create metadata
        metadata = IndexMetadata(
            name=name,
            created_at=datetime.now(),
            file_count=len(files),
            glob_pattern=glob_pattern,
            bm25_params=params,
            tokenization=tokenization,
        )

        # Save metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)
        logger.debug("Saved metadata to %s", metadata_path)

        # Register index
        self.registry.add_index(name, metadata.model_dump(mode="json"))

        logger.info("Index '%s' created successfully", name)
        return metadata

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

        # Re-create index with same parameters
        # First, remove from registry
        self.registry.remove_index(name)

        # Create new index
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
