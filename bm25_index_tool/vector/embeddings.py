"""AWS Bedrock embeddings client (Nova-only with image support).

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.vector.errors import AWSCredentialsError, EmbeddingError

if TYPE_CHECKING:
    from bm25_index_tool.vector.chunking import Chunk

logger = get_logger(__name__)

# Nova multimodal embeddings model (supports text and images)
MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
DIMENSIONS = 1024
PRICE_PER_1K_TOKENS = 0.00018


class BedrockEmbeddings:
    """AWS Bedrock embeddings client with parallel processing (Nova-only)."""

    def __init__(
        self,
        max_workers: int | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        """Initialize the Bedrock embeddings client.

        Args:
            max_workers: Maximum parallel embedding requests (defaults to CPU count)
            region_name: AWS region (defaults to us-east-1 for Nova model support)

        Raises:
            AWSCredentialsError: If boto3 import fails or credentials invalid
        """
        self.max_workers = max_workers or os.cpu_count() or 4
        self.total_tokens = 0

        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config  # type: ignore[import-untyped]

            # Configure connection pool to match max_workers
            config = Config(
                max_pool_connections=self.max_workers,
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                config=config,
            )
            logger.debug(
                "Initialized Bedrock client for model: %s (dimensions=%d)",
                MODEL_ID,
                DIMENSIONS,
            )
        except ImportError as e:
            raise AWSCredentialsError(
                "boto3 not installed. Install with: uv sync --extra vector"
            ) from e
        except Exception as e:
            raise AWSCredentialsError(f"Failed to initialize Bedrock client: {e}") from e

    def _embed_single(self, text: str, purpose: str = "GENERIC_INDEX") -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            purpose: Embedding purpose:
                     - GENERIC_INDEX: for indexing documents
                     - TEXT_RETRIEVAL: for search queries

        Returns:
            Embedding vector (1024 dimensions)

        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            body = json.dumps(
                {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": purpose,
                        "embeddingDimension": DIMENSIONS,
                        "text": {"truncationMode": "END", "value": text},
                    },
                }
            )

            response = self._client.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())

            # Nova returns embeddings array with embedding objects
            embedding: list[float] = list(result["embeddings"][0]["embedding"])
            # Estimate tokens (rough: ~4 chars per token)
            self.total_tokens += len(text) // 4

            return embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_image(self, image_bytes: bytes, format: str) -> list[float]:
        """Generate embedding for an image.

        Args:
            image_bytes: Raw image bytes
            format: Image format ('png' or 'jpeg')

        Returns:
            Embedding vector (1024 dimensions)

        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            body = json.dumps(
                {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX",
                        "embeddingDimension": DIMENSIONS,
                        "image": {
                            "format": format,
                            "source": {"bytes": base64.b64encode(image_bytes).decode()},
                        },
                    },
                }
            )

            response = self._client.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())

            # Nova returns embeddings array with embedding objects
            embedding: list[float] = list(result["embeddings"][0]["embedding"])

            return embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate image embedding: {e}") from e

    def embed_texts(self, texts: list[str], fail_fast: bool = True) -> list[list[float]]:
        """Generate embeddings for multiple texts in parallel.

        Args:
            texts: List of texts to embed
            fail_fast: If True, raise on first error. If False, skip failed texts.

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If any embedding fails and fail_fast=True
        """
        if not texts:
            return []

        embeddings: list[list[float] | None] = [None] * len(texts)
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._embed_single, text): idx for idx, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except EmbeddingError as e:
                    error_msg = f"Text {idx}: {e}"
                    errors.append(error_msg)
                    if fail_fast:
                        # Cancel remaining futures
                        for f in future_to_idx:
                            f.cancel()
                        raise EmbeddingError(
                            f"Embedding failed for text {idx}: {e}. Total errors: {len(errors)}"
                        ) from e

        if errors:
            logger.warning("Embedding errors (%d): %s", len(errors), errors[:5])

        # Filter out None values and return
        result = [e for e in embeddings if e is not None]

        logger.info(
            "Generated %d/%d embeddings (model=%s, tokens=%d)",
            len(result),
            len(texts),
            MODEL_ID,
            self.total_tokens,
        )

        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Uses TEXT_RETRIEVAL purpose, which is optimized for query
        embedding vs document embedding.

        Args:
            query: Search query text

        Returns:
            Embedding vector (1024 dimensions)
        """
        return self._embed_single(query, purpose="TEXT_RETRIEVAL")

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of embedding vectors (same order as chunks)
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def get_estimated_cost(self) -> float:
        """Get estimated cost for embeddings generated so far.

        Returns:
            Estimated cost in USD
        """
        return (self.total_tokens / 1000) * PRICE_PER_1K_TOKENS
