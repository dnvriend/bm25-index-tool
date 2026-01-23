"""Text chunking for vector embeddings.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.vector.errors import ChunkingError

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    source_path: str
    chunk_index: int
    start_word: int
    end_word: int
    word_count: int
    chunk_type: str = "text"

    def to_dict(self) -> dict[str, str | int]:
        """Convert chunk to dictionary for serialization."""
        return {
            "text": self.text,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "start_word": self.start_word,
            "end_word": self.end_word,
            "word_count": self.word_count,
            "chunk_type": self.chunk_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int]) -> Chunk:
        """Create Chunk from dictionary."""
        return cls(
            text=str(data["text"]),
            source_path=str(data["source_path"]),
            chunk_index=int(data["chunk_index"]),
            start_word=int(data["start_word"]),
            end_word=int(data["end_word"]),
            word_count=int(data["word_count"]),
            chunk_type=str(data.get("chunk_type", "text")),
        )


@dataclass
class ImageChunk:
    """A chunk representing an entire image for vector embedding."""

    source_path: str
    chunk_index: int = 0  # Always 0 for images (1 chunk per image)
    chunk_type: str = "image"  # Distinguish from text chunks

    def to_dict(self) -> dict[str, str | int]:
        """Convert chunk to dictionary for serialization."""
        return {
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int]) -> ImageChunk:
        """Create ImageChunk from dictionary."""
        return cls(
            source_path=str(data["source_path"]),
            chunk_index=int(data.get("chunk_index", 0)),
            chunk_type=str(data.get("chunk_type", "image")),
        )


@runtime_checkable
class Chunker(Protocol):
    """Protocol for text chunkers that support pipeline composition."""

    def chunk_text(self, text: str, source_path: str) -> list[Chunk]:
        """Split text into chunks."""
        ...

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Read and chunk a file."""
        ...

    def chunk_files(self, files: list[Path]) -> list[Chunk]:
        """Chunk multiple files."""
        ...


class ChunkerPipeline:
    """Pipeline for chaining multiple chunkers together."""

    def __init__(self, chunkers: list[Chunker]) -> None:
        """Initialize the chunker pipeline.

        Args:
            chunkers: List of chunkers to chain together
        """
        if not chunkers:
            raise ChunkingError("Pipeline requires at least one chunker")
        self.chunkers = chunkers

    def __or__(self, other: Chunker | ChunkerPipeline) -> ChunkerPipeline:
        """Chain another chunker or pipeline to this pipeline.

        Args:
            other: Chunker or pipeline to chain

        Returns:
            New ChunkerPipeline with all chunkers combined
        """
        if isinstance(other, ChunkerPipeline):
            return ChunkerPipeline(self.chunkers + other.chunkers)
        return ChunkerPipeline(self.chunkers + [other])

    def chunk_files(self, files: list[Path]) -> list[Chunk]:
        """Chunk files through the pipeline.

        The first chunker produces initial chunks, then each subsequent
        chunker processes those chunks.

        Args:
            files: List of file paths

        Returns:
            List of processed chunks
        """
        if not self.chunkers:
            return []

        # First chunker produces initial chunks from files
        chunks = self.chunkers[0].chunk_files(files)

        # Subsequent chunkers process the chunks
        for chunker in self.chunkers[1:]:
            if hasattr(chunker, "process_chunks"):
                chunks = chunker.process_chunks(chunks)
            else:
                # If chunker doesn't have process_chunks, skip it
                logger.warning(
                    "Chunker %s does not support process_chunks, skipping",
                    type(chunker).__name__,
                )

        return chunks


class TextChunker:
    """Splits text into chunks for embedding."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        """Initialize the text chunker.

        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __or__(self, other: Chunker | ChunkerPipeline) -> ChunkerPipeline:
        """Chain another chunker to create a pipeline.

        Args:
            other: Chunker or pipeline to chain

        Returns:
            ChunkerPipeline with this chunker and the other
        """
        if isinstance(other, ChunkerPipeline):
            return ChunkerPipeline([self] + other.chunkers)
        return ChunkerPipeline([self, other])

    def chunk_text(self, text: str, source_path: str) -> list[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            source_path: Path to source file for metadata

        Returns:
            List of Chunk objects
        """
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    start_word=start,
                    end_word=end,
                    word_count=len(chunk_words),
                )
            )

            start += step
            chunk_index += 1

            # Break if we've processed all words
            if end >= len(words):
                break

        logger.debug(
            "Chunked %s into %d chunks (size=%d, overlap=%d)",
            source_path,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        return chunks

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Read and chunk a file.

        Args:
            file_path: Path to file

        Returns:
            List of Chunk objects

        Raises:
            ChunkingError: If file cannot be read
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.chunk_text(content, str(file_path))
        except Exception as e:
            raise ChunkingError(f"Failed to chunk file {file_path}: {e}") from e

    def chunk_files(self, files: list[Path]) -> list[Chunk]:
        """Chunk multiple files.

        Args:
            files: List of file paths

        Returns:
            List of all chunks from all files
        """
        all_chunks: list[Chunk] = []
        for file_path in files:
            try:
                chunks = self.chunk_file(file_path)
                all_chunks.extend(chunks)
            except ChunkingError as e:
                logger.warning("Skipping file due to chunking error: %s", e)
                continue

        logger.info("Created %d chunks from %d files", len(all_chunks), len(files))
        return all_chunks


class CharacterLimitChunker:
    """Enforces character limits on chunks by truncation.

    This chunker is designed to be used in a pipeline after TextChunker
    to ensure chunks don't exceed embedding model limits. It handles
    edge cases like single words exceeding the limit by hard truncation.

    Example:
        pipeline = TextChunker(300, 50) | CharacterLimitChunker(48000)
        chunks = pipeline.chunk_files(files)
    """

    def __init__(self, max_chars: int = 48000) -> None:
        """Initialize the character limit chunker.

        Args:
            max_chars: Maximum characters per chunk (default: 48000 for Nova)
        """
        if max_chars <= 0:
            raise ChunkingError(f"max_chars must be positive, got {max_chars}")
        self.max_chars = max_chars

    def __or__(self, other: Chunker | ChunkerPipeline) -> ChunkerPipeline:
        """Chain another chunker to create a pipeline.

        Args:
            other: Chunker or pipeline to chain

        Returns:
            ChunkerPipeline with this chunker and the other
        """
        if isinstance(other, ChunkerPipeline):
            return ChunkerPipeline([self] + other.chunkers)
        return ChunkerPipeline([self, other])

    def process_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Process chunks to enforce character limits.

        Chunks exceeding the limit are truncated. If a single word exceeds
        the limit, it is hard-truncated at the limit.

        Args:
            chunks: List of chunks to process

        Returns:
            List of processed chunks with enforced limits
        """
        processed: list[Chunk] = []
        truncated_count = 0

        for chunk in chunks:
            if len(chunk.text) <= self.max_chars:
                processed.append(chunk)
            else:
                # Truncate the text
                truncated_text = self._truncate_text(chunk.text)
                truncated_count += 1

                # Create new chunk with truncated text
                new_chunk = Chunk(
                    text=truncated_text,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                    word_count=len(truncated_text.split()),
                )
                processed.append(new_chunk)

        if truncated_count > 0:
            logger.warning(
                "Truncated %d chunks exceeding %d character limit",
                truncated_count,
                self.max_chars,
            )

        return processed

    def _truncate_text(self, text: str) -> str:
        """Truncate text to the character limit.

        Tries to truncate at word boundaries, but will hard-truncate
        single words exceeding the limit.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        if len(text) <= self.max_chars:
            return text

        # Try to truncate at word boundary
        truncated = text[: self.max_chars]
        last_space = truncated.rfind(" ")

        if last_space > 0:
            # Found a space, truncate there
            return truncated[:last_space]
        else:
            # No space found (single giant word), hard truncate
            logger.warning(
                "Hard truncating text at %d chars (no word boundary found)",
                self.max_chars,
            )
            return truncated

    def chunk_text(self, text: str, source_path: str) -> list[Chunk]:
        """Not implemented for CharacterLimitChunker.

        This chunker is designed to process existing chunks, not create new ones.
        """
        raise NotImplementedError(
            "CharacterLimitChunker.chunk_text is not implemented. "
            "Use process_chunks() in a pipeline instead."
        )

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Not implemented for CharacterLimitChunker.

        This chunker is designed to process existing chunks, not create new ones.
        """
        raise NotImplementedError(
            "CharacterLimitChunker.chunk_file is not implemented. "
            "Use process_chunks() in a pipeline instead."
        )

    def chunk_files(self, files: list[Path]) -> list[Chunk]:
        """Not implemented for CharacterLimitChunker.

        This chunker is designed to process existing chunks, not create new ones.
        """
        raise NotImplementedError(
            "CharacterLimitChunker.chunk_files is not implemented. "
            "Use process_chunks() in a pipeline instead."
        )


class ImageChunker:
    """Creates one chunk per image file.

    Unlike TextChunker which splits text into overlapping word chunks,
    ImageChunker treats each image as a single unit for embedding.
    """

    def chunk_file(self, file_path: Path) -> list[ImageChunk]:
        """Create a single chunk for an image file.

        Args:
            file_path: Path to the image file

        Returns:
            List containing a single ImageChunk
        """
        return [ImageChunk(source_path=str(file_path))]

    def chunk_files(self, files: list[Path]) -> list[ImageChunk]:
        """Create chunks for multiple image files.

        Args:
            files: List of image file paths

        Returns:
            List of ImageChunks (one per file)
        """
        chunks = []
        for file_path in files:
            chunks.extend(self.chunk_file(file_path))
        logger.info("Created %d image chunks from %d files", len(chunks), len(files))
        return chunks
