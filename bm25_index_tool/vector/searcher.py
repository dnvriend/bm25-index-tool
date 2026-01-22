"""Vector search using FAISS.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

import json
from typing import Any

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_chunks_path, get_faiss_index_path
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.vector.chunking import Chunk
from bm25_index_tool.vector.embeddings import BedrockEmbeddings
from bm25_index_tool.vector.errors import (
    MissingModelMetadataError,
    VectorIndexNotFoundError,
    VectorSearchError,
)

logger = get_logger(__name__)


class VectorSearcher:
    """Performs semantic search using FAISS vector indices."""

    def __init__(
        self,
        model_id: str | None = None,
        strict_mode: bool = True,
    ) -> None:
        """Initialize the vector searcher.

        Args:
            model_id: Override embedding model ID. If None, uses model from index metadata.
            strict_mode: If True, raises error if index lacks model metadata.
                         If False, falls back to default model.
        """
        self._model_id_override = model_id
        self._strict_mode = strict_mode
        self._registry = IndexRegistry()
        self._embeddings_clients: dict[str, BedrockEmbeddings] = {}

    def _get_index_model_info(self, name: str) -> tuple[str, int]:
        """Get embedding model info from index metadata.

        Args:
            name: Index name

        Returns:
            Tuple of (model_id, dimensions)

        Raises:
            MissingModelMetadataError: If strict mode and metadata missing
        """
        metadata = self._registry.get_index(name)

        if metadata is None:
            if self._strict_mode:
                raise MissingModelMetadataError(
                    f"Index '{name}' not found in registry. "
                    "Cannot determine embedding model for search."
                )
            logger.warning("Index '%s' not in registry, using default Nova model", name)
            return ("amazon.nova-2-multimodal-embeddings-v1:0", 3072)

        vector_metadata = metadata.get("vector_metadata")
        if vector_metadata is None:
            if self._strict_mode:
                raise MissingModelMetadataError(
                    f"Index '{name}' has no vector metadata. "
                    "Was this index created without --no-vector? "
                    "Re-create the index or disable strict mode."
                )
            logger.warning("Index '%s' has no vector_metadata, using default Nova model", name)
            return ("amazon.nova-2-multimodal-embeddings-v1:0", 3072)

        model_id = vector_metadata.get("embedding_model")
        dimensions = vector_metadata.get("dimensions")

        if model_id is None:
            if self._strict_mode:
                raise MissingModelMetadataError(
                    f"Index '{name}' vector metadata lacks 'embedding_model'. "
                    "This index was created with an older version. "
                    "Re-create the index or disable strict mode."
                )
            logger.warning("Index '%s' lacks embedding_model, using default Nova model", name)
            return ("amazon.nova-2-multimodal-embeddings-v1:0", 3072)

        return (model_id, dimensions or 3072)

    def _get_embeddings_client(self, index_name: str) -> BedrockEmbeddings:
        """Get or create embeddings client for an index.

        Uses model from index metadata unless overridden in constructor.

        Args:
            index_name: Name of index being searched

        Returns:
            BedrockEmbeddings client configured for the index's model
        """
        # Use override if provided
        if self._model_id_override:
            cache_key = f"override:{self._model_id_override}"
            if cache_key not in self._embeddings_clients:
                self._embeddings_clients[cache_key] = BedrockEmbeddings(
                    model_id=self._model_id_override
                )
            return self._embeddings_clients[cache_key]

        # Get model from index metadata
        model_id, dimensions = self._get_index_model_info(index_name)
        cache_key = f"{model_id}:{dimensions}"

        if cache_key not in self._embeddings_clients:
            self._embeddings_clients[cache_key] = BedrockEmbeddings(
                model_id=model_id,
                dimensions=dimensions,
            )
            logger.info(
                "Using embedding model '%s' (dimensions=%d) from index '%s' metadata",
                model_id,
                dimensions,
                index_name,
            )

        return self._embeddings_clients[cache_key]

    def _load_index(self, name: str) -> tuple[Any, list[Chunk]]:
        """Load FAISS index and chunks.

        Args:
            name: Index name

        Returns:
            Tuple of (FAISS index, list of chunks)

        Raises:
            VectorIndexNotFoundError: If index doesn't exist
            VectorSearchError: If loading fails
        """
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as e:
            raise VectorSearchError(
                "FAISS not installed. Install with: uv sync --extra vector"
            ) from e

        faiss_path = get_faiss_index_path(name)
        chunks_path = get_chunks_path(name)

        if not faiss_path.exists():
            raise VectorIndexNotFoundError(f"Vector index not found for '{name}'")

        if not chunks_path.exists():
            raise VectorIndexNotFoundError(f"Chunks metadata not found for '{name}'")

        # Load FAISS index
        index = faiss.read_index(str(faiss_path))
        logger.debug("Loaded FAISS index with %d vectors", index.ntotal)

        # Load chunks
        with open(chunks_path) as f:
            chunks_data = json.load(f)
        chunks = [Chunk.from_dict(c) for c in chunks_data]
        logger.debug("Loaded %d chunks", len(chunks))

        return index, chunks

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search index using semantic similarity.

        Args:
            index_name: Name of index to search
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with path, score, and chunk info

        Raises:
            VectorIndexNotFoundError: If index doesn't exist
            VectorSearchError: If search fails
        """
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise VectorSearchError(
                "FAISS or numpy not installed. Install with: uv sync --extra vector"
            ) from e

        logger.info("Semantic search in '%s': %s", index_name, query)

        # Load index and chunks
        index, chunks = self._load_index(index_name)

        # Generate query embedding (uses model from index metadata)
        embeddings_client = self._get_embeddings_client(index_name)
        query_embedding = embeddings_client.embed_query(query)

        # Normalize for cosine similarity
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = index.search(query_vector, min(top_k, len(chunks)))

        # Build results
        results: list[dict[str, Any]] = []
        seen_paths: dict[str, float] = {}  # Track best score per path

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue

            chunk = chunks[idx]
            path = chunk.source_path

            # Keep best score per document
            if path in seen_paths:
                if score > seen_paths[path]:
                    # Update existing result with better score
                    for r in results:
                        if r["path"] == path:
                            r["score"] = float(score)
                            r["chunk_text"] = chunk.text
                            r["chunk_index"] = chunk.chunk_index
                            break
                    seen_paths[path] = float(score)
            else:
                seen_paths[path] = float(score)
                results.append(
                    {
                        "path": path,
                        "score": float(score),
                        "chunk_text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "word_count": chunk.word_count,
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k unique documents
        results = results[:top_k]

        logger.info("Found %d semantic matches", len(results))
        return results

    def search_multi(
        self,
        index_names: list[str],
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search multiple indices and combine results.

        Args:
            index_names: List of index names to search
            query: Search query
            top_k: Number of results per index

        Returns:
            Combined and sorted results from all indices
        """
        all_results: list[dict[str, Any]] = []

        for name in index_names:
            try:
                results = self.search(name, query, top_k=top_k)
                for r in results:
                    r["index_name"] = name
                all_results.extend(results)
            except VectorIndexNotFoundError:
                logger.warning("Vector index not found for '%s', skipping", name)
                continue

        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def index_exists(self, name: str) -> bool:
        """Check if a vector index exists.

        Args:
            name: Index name

        Returns:
            True if index exists
        """
        return get_faiss_index_path(name).exists()
