"""Error classes for vector search functionality.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""


class VectorSearchError(Exception):
    """Base exception for vector search errors."""

    pass


class EmbeddingError(VectorSearchError):
    """Error during embedding generation."""

    pass


class AWSCredentialsError(VectorSearchError):
    """Error with AWS credentials or configuration."""

    pass


class VectorIndexNotFoundError(VectorSearchError):
    """Vector index not found for the specified index name."""

    pass


class ChunkingError(VectorSearchError):
    """Error during text chunking."""

    pass


class MissingModelMetadataError(VectorSearchError):
    """Error when index metadata lacks required model information."""

    pass
