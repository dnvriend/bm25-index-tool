"""Configuration models for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class BM25Profile(str, Enum):
    """BM25 parameter profiles."""

    STANDARD = "standard"
    CODE = "code"

    def get_params(self) -> tuple[float, float]:
        """Get (k1, b) parameters for the profile.

        Returns:
            Tuple of (k1, b) values
        """
        if self == BM25Profile.STANDARD:
            return (1.5, 0.75)
        elif self == BM25Profile.CODE:
            return (1.2, 0.5)
        else:
            return (1.5, 0.75)


class BM25Params(BaseModel):
    """BM25 algorithm parameters."""

    k1: float = Field(default=1.5, ge=0.0, le=10.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    method: str = Field(default="lucene")

    @classmethod
    def from_profile(cls, profile: BM25Profile) -> BM25Params:
        """Create BM25Params from a profile.

        Args:
            profile: BM25 profile enum

        Returns:
            BM25Params instance
        """
        k1, b = profile.get_params()
        return cls(k1=k1, b=b)


class TokenizationConfig(BaseModel):
    """Tokenization configuration."""

    stemmer: str = Field(default="")
    stopwords: str = Field(default="en")

    @property
    def stemmer_enabled(self) -> bool:
        """Check if stemming is enabled.

        Returns:
            True if stemmer is configured, False otherwise
        """
        return bool(self.stemmer)


class VectorConfig(BaseModel):
    """Vector index configuration."""

    model_id: str = Field(default="amazon.nova-2-multimodal-embeddings-v1:0")
    chunk_size: int = Field(default=300, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    dimensions: int = Field(default=3072, ge=256, le=4096)
    max_chunk_chars: int = Field(default=48000, ge=1000, le=100000)


class VectorMetadata(BaseModel):
    """Metadata for a vector index."""

    chunk_count: int = Field(ge=0)
    embedding_model: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class IndexMetadata(BaseModel):
    """Metadata for a BM25 index."""

    name: str
    created_at: datetime
    file_count: int = Field(ge=0)
    glob_pattern: str
    index_version: str = Field(default="1.0")
    bm25_params: BM25Params
    tokenization: TokenizationConfig
    vector_metadata: VectorMetadata | None = Field(default=None)

    model_config = {"json_schema_extra": {"examples": [{"name": "obsidian-vault"}]}}


class DefaultsConfig(BaseModel):
    """Default configuration values."""

    stemmer: str = Field(default="")
    profile: str = Field(default="standard")
    top_k: int = Field(default=10, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    format: str = Field(default="simple")


class GitignoreConfig(BaseModel):
    """Gitignore configuration."""

    enabled: bool = Field(default=True)


class GlobalConfig(BaseModel):
    """Global configuration for bm25-index-tool."""

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    gitignore: GitignoreConfig = Field(default_factory=GitignoreConfig)
