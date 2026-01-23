"""Vector search using SQLite with sqlite-vec.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

from typing import Any

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_sqlite_db_path
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage
from bm25_index_tool.vector.embeddings import BedrockEmbeddings
from bm25_index_tool.vector.errors import (
    VectorIndexNotFoundError,
    VectorSearchError,
)

logger = get_logger(__name__)


class VectorSearcher:
    """Performs semantic search using SQLite with sqlite-vec."""

    def __init__(self) -> None:
        """Initialize the vector searcher."""
        self._embeddings_client: BedrockEmbeddings | None = None

    def _get_embeddings_client(self) -> BedrockEmbeddings:
        """Get or create embeddings client.

        Returns:
            BedrockEmbeddings client
        """
        if self._embeddings_client is None:
            self._embeddings_client = BedrockEmbeddings()
        return self._embeddings_client

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
            List of search results with path, score, chunk_text, chunk_index, chunk_type

        Raises:
            VectorIndexNotFoundError: If index doesn't exist
            VectorSearchError: If search fails
        """
        logger.info("Semantic search in '%s': %s", index_name, query)

        if not self.index_exists(index_name):
            raise VectorIndexNotFoundError(f"Vector index not found for '{index_name}'")

        try:
            # Generate query embedding
            embeddings_client = self._get_embeddings_client()
            query_embedding = embeddings_client.embed_query(query)

            # Search using SQLiteStorage
            with SQLiteStorage(index_name) as storage:
                vector_results = storage.search_vector(query_embedding, top_k * 3)

            # Build results, grouping by document path and keeping best score
            results: list[dict[str, Any]] = []
            seen_paths: dict[str, float] = {}

            for vr in vector_results:
                path = vr.path
                # Convert distance to score (lower distance = higher score for cosine)
                score = 1.0 - vr.distance

                if path in seen_paths:
                    if score > seen_paths[path]:
                        # Update existing result with better score
                        for r in results:
                            if r["path"] == path:
                                r["score"] = score
                                r["chunk_text"] = vr.text
                                r["chunk_index"] = vr.chunk_id
                                r["chunk_type"] = vr.chunk_type
                                break
                        seen_paths[path] = score
                else:
                    seen_paths[path] = score
                    results.append(
                        {
                            "path": path,
                            "score": score,
                            "chunk_text": vr.text,
                            "chunk_index": vr.chunk_id,
                            "chunk_type": vr.chunk_type,
                        }
                    )

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limit to top_k unique documents
            results = results[:top_k]

            logger.info("Found %d semantic matches", len(results))
            return results

        except VectorIndexNotFoundError:
            raise
        except Exception as e:
            raise VectorSearchError(f"Vector search failed: {e}") from e

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
            True if index exists with vector support
        """
        db_path = get_sqlite_db_path(name)
        if not db_path.exists():
            return False

        # Check if chunks_vec table exists
        try:
            with SQLiteStorage(name) as storage:
                return bool(storage.has_vector_index())
        except Exception:
            return False
