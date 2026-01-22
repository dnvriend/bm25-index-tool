"""Vector indexer using FAISS.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore

from bm25_index_tool.config.models import VectorConfig, VectorMetadata
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_chunks_path, get_faiss_index_path
from bm25_index_tool.vector.chunking import CharacterLimitChunker, TextChunker
from bm25_index_tool.vector.embeddings import BedrockEmbeddings, get_model_dimensions
from bm25_index_tool.vector.errors import VectorSearchError

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class VectorIndexer:
    """Creates FAISS vector indices from text files."""

    def __init__(self, config: VectorConfig | None = None) -> None:
        """Initialize the vector indexer.

        Args:
            config: Vector index configuration (uses defaults if None)
        """
        self.config = config or VectorConfig()

    def create_index(
        self,
        name: str,
        files: list[Path],
        show_progress: bool = True,
    ) -> VectorMetadata:
        """Create a FAISS vector index from files.

        Args:
            name: Index name
            files: List of files to index
            show_progress: Show progress bar

        Returns:
            VectorMetadata with index statistics

        Raises:
            VectorSearchError: If index creation fails
        """
        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np
        except ImportError as e:
            raise VectorSearchError(
                "FAISS or numpy not installed. Install with: uv sync --extra vector"
            ) from e

        logger.info("Creating vector index '%s' from %d files", name, len(files))

        # Create chunker pipeline with character limit for Nova model support
        text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        char_limit_chunker = CharacterLimitChunker(max_chars=self.config.max_chunk_chars)
        pipeline = text_chunker | char_limit_chunker

        logger.info("Chunking %d files...", len(files))
        chunks = pipeline.chunk_files(files)

        if not chunks:
            raise VectorSearchError("No chunks created from files")

        logger.info("Created %d chunks", len(chunks))

        # Generate embeddings
        embeddings_client = BedrockEmbeddings(
            model_id=self.config.model_id,
            dimensions=self.config.dimensions,
        )

        logger.info("Generating embeddings...")
        if show_progress:
            # Process in batches with progress bar
            batch_size = 50
            all_embeddings: list[list[float]] = []
            chunk_texts = [chunk.text for chunk in chunks]

            for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding"):
                batch = chunk_texts[i : i + batch_size]
                batch_embeddings = embeddings_client.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
        else:
            all_embeddings = embeddings_client.embed_chunks(chunks)

        if len(all_embeddings) != len(chunks):
            logger.warning(
                "Embedding count mismatch: %d embeddings for %d chunks",
                len(all_embeddings),
                len(chunks),
            )
            # Trim chunks to match embeddings
            chunks = chunks[: len(all_embeddings)]

        # Create FAISS index
        dimensions = get_model_dimensions(self.config.model_id, self.config.dimensions)
        logger.info("Creating FAISS index (dimensions=%d)", dimensions)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Create index with Inner Product (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimensions)
        index.add(embeddings_array)

        # Save index
        faiss_path = get_faiss_index_path(name)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(faiss_path))
        logger.info("Saved FAISS index to %s", faiss_path)

        # Save chunks metadata
        chunks_path = get_chunks_path(name)
        chunks_data = [chunk.to_dict() for chunk in chunks]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)
        logger.info("Saved %d chunks to %s", len(chunks), chunks_path)

        # Create metadata
        metadata = VectorMetadata(
            chunk_count=len(chunks),
            embedding_model=self.config.model_id,
            dimensions=dimensions,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            total_tokens=embeddings_client.total_tokens,
            estimated_cost_usd=embeddings_client.get_estimated_cost(),
        )

        logger.info(
            "Vector index created: %d chunks, %d dimensions, ~$%.4f cost",
            metadata.chunk_count,
            metadata.dimensions,
            metadata.estimated_cost_usd,
        )

        return metadata

    def delete_index(self, name: str) -> bool:
        """Delete a vector index.

        Args:
            name: Index name

        Returns:
            True if deleted, False if not found
        """
        faiss_path = get_faiss_index_path(name)
        chunks_path = get_chunks_path(name)

        deleted = False

        if faiss_path.exists():
            faiss_path.unlink()
            logger.info("Deleted FAISS index: %s", faiss_path)
            deleted = True

        if chunks_path.exists():
            chunks_path.unlink()
            logger.info("Deleted chunks file: %s", chunks_path)
            deleted = True

        return deleted

    def index_exists(self, name: str) -> bool:
        """Check if a vector index exists.

        Args:
            name: Index name

        Returns:
            True if index exists
        """
        return get_faiss_index_path(name).exists()
