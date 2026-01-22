"""Vector search module for semantic search using AWS Bedrock and FAISS.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from bm25_index_tool.vector.chunking import (
    CharacterLimitChunker,
    Chunk,
    Chunker,
    ChunkerPipeline,
    TextChunker,
)
from bm25_index_tool.vector.embeddings import (
    BedrockEmbeddings,
    get_model_dimensions,
    is_nova_model,
)
from bm25_index_tool.vector.errors import (
    AWSCredentialsError,
    ChunkingError,
    EmbeddingError,
    MissingModelMetadataError,
    VectorIndexNotFoundError,
    VectorSearchError,
)
from bm25_index_tool.vector.indexer import VectorIndexer
from bm25_index_tool.vector.searcher import VectorSearcher

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkerPipeline",
    "CharacterLimitChunker",
    "TextChunker",
    "BedrockEmbeddings",
    "get_model_dimensions",
    "is_nova_model",
    "VectorIndexer",
    "VectorSearcher",
    "VectorSearchError",
    "EmbeddingError",
    "AWSCredentialsError",
    "VectorIndexNotFoundError",
    "MissingModelMetadataError",
    "ChunkingError",
]
