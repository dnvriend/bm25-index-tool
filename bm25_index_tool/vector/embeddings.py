"""AWS Bedrock embeddings client.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.vector.errors import AWSCredentialsError, EmbeddingError

if TYPE_CHECKING:
    from bm25_index_tool.vector.chunking import Chunk

logger = get_logger(__name__)

# Model dimensions mapping (default dimensions for each model)
MODEL_DIMENSIONS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
    # Nova models support configurable dimensions: 256, 512, 1024, 3072
    "amazon.nova-2-multimodal-embeddings-v1:0": 3072,
}

# Supported dimensions for models with configurable dimensions
CONFIGURABLE_DIMENSIONS: dict[str, list[int]] = {
    "amazon.nova-2-multimodal-embeddings-v1:0": [256, 512, 1024, 3072],
}

# Pricing per 1000 tokens (approximate, USD)
MODEL_PRICING: dict[str, float] = {
    "amazon.titan-embed-text-v2:0": 0.0002,
    "amazon.titan-embed-text-v1": 0.0001,
    "cohere.embed-english-v3": 0.0001,
    "cohere.embed-multilingual-v3": 0.0001,
    "amazon.nova-2-multimodal-embeddings-v1:0": 0.00018,  # estimate
}


def is_nova_model(model_id: str) -> bool:
    """Check if the model is a Nova embedding model.

    Args:
        model_id: Bedrock model ID

    Returns:
        True if Nova model
    """
    return "nova" in model_id.lower() and "embedding" in model_id.lower()


def get_model_dimensions(model_id: str, dimensions: int | None = None) -> int:
    """Get embedding dimensions for a model.

    For models with configurable dimensions (like Nova), validates the
    requested dimensions. For fixed-dimension models, ignores the
    dimensions parameter.

    Args:
        model_id: Bedrock model ID
        dimensions: Requested dimensions (for configurable models)

    Returns:
        Number of dimensions

    Raises:
        EmbeddingError: If requested dimensions not supported
    """
    # Check if model supports configurable dimensions
    if model_id in CONFIGURABLE_DIMENSIONS:
        supported = CONFIGURABLE_DIMENSIONS[model_id]
        if dimensions is not None:
            if dimensions not in supported:
                raise EmbeddingError(
                    f"Model {model_id} supports dimensions {supported}, "
                    f"but {dimensions} was requested"
                )
            return dimensions
        # Return default for this model
        return MODEL_DIMENSIONS.get(model_id, supported[-1])

    # Fixed dimension model
    return MODEL_DIMENSIONS.get(model_id, 1024)


def estimate_cost(model_id: str, total_tokens: int) -> float:
    """Estimate cost for embedding tokens.

    Args:
        model_id: Bedrock model ID
        total_tokens: Total number of tokens

    Returns:
        Estimated cost in USD
    """
    price_per_1k = MODEL_PRICING.get(model_id, 0.0002)
    return (total_tokens / 1000) * price_per_1k


class BedrockEmbeddings:
    """AWS Bedrock embeddings client with parallel processing."""

    def __init__(
        self,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        max_workers: int | None = None,
        region_name: str = "us-east-1",
        dimensions: int | None = None,
    ) -> None:
        """Initialize the Bedrock embeddings client.

        Args:
            model_id: Bedrock embedding model ID
            max_workers: Maximum parallel embedding requests (defaults to CPU count)
            region_name: AWS region (defaults to us-east-1 for Nova model support)
            dimensions: Embedding dimensions (for configurable models like Nova)

        Raises:
            AWSCredentialsError: If boto3 import fails or credentials invalid
        """
        self.model_id = model_id
        self.max_workers = max_workers or os.cpu_count() or 4
        self._requested_dimensions = dimensions
        self.dimensions = get_model_dimensions(model_id, dimensions)
        self.total_tokens = 0
        self._is_nova = is_nova_model(model_id)

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
                model_id,
                self.dimensions,
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
            purpose: Embedding purpose for Nova models:
                     - GENERIC_INDEX: for indexing documents
                     - TEXT_RETRIEVAL: for search queries

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            # Prepare request body based on model
            if self._is_nova:
                body = json.dumps(
                    {
                        "schemaVersion": "nova-multimodal-embed-v1",
                        "taskType": "SINGLE_EMBEDDING",
                        "singleEmbeddingParams": {
                            "embeddingPurpose": purpose,
                            "embeddingDimension": self.dimensions,
                            "text": {"truncationMode": "END", "value": text},
                        },
                    }
                )
            elif "titan" in self.model_id.lower():
                body = json.dumps({"inputText": text})
            elif "cohere" in self.model_id.lower():
                body = json.dumps({"texts": [text], "input_type": "search_document"})
            else:
                body = json.dumps({"inputText": text})

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())

            # Extract embedding based on model response format
            embedding: list[float]
            if self._is_nova:
                # Nova returns embeddings array with embedding objects
                embedding = list(result["embeddings"][0]["embedding"])
                # Estimate tokens (rough: ~4 chars per token)
                self.total_tokens += len(text) // 4
            elif "titan" in self.model_id.lower():
                embedding = list(result["embedding"])
                self.total_tokens += len(text) // 4
            elif "cohere" in self.model_id.lower():
                embedding = list(result["embeddings"][0])
                self.total_tokens += len(text) // 4
            else:
                embedding = list(result.get("embedding", result.get("embeddings", [[]])[0]))
                self.total_tokens += len(text) // 4

            return embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def _embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query (uses TEXT_RETRIEVAL purpose for Nova).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if self._is_nova:
            return self._embed_single(text, purpose="TEXT_RETRIEVAL")
        return self._embed_single(text)

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
            self.model_id,
            self.total_tokens,
        )

        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Uses SEARCH_QUERY purpose for Nova models, which is optimized
        for query embedding vs document embedding.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        return self._embed_query(query)

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
        return estimate_cost(self.model_id, self.total_tokens)
