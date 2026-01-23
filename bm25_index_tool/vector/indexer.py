"""Vector indexer using SQLite with sqlite-vec.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore

from bm25_index_tool.config.models import VectorConfig, VectorMetadata
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_sqlite_db_path
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage, compute_file_hash
from bm25_index_tool.vector.chunking import CharacterLimitChunker, TextChunker
from bm25_index_tool.vector.embeddings import DIMENSIONS, MODEL_ID, BedrockEmbeddings
from bm25_index_tool.vector.errors import VectorSearchError
from bm25_index_tool.vector.image_processor import ImageProcessor

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class VectorIndexer:
    """Creates SQLite vector indices from text and image files."""

    def __init__(self, config: VectorConfig | None = None) -> None:
        """Initialize the vector indexer.

        Args:
            config: Vector index configuration (uses defaults if None)
        """
        self.config = config or VectorConfig()
        self.image_processor = ImageProcessor()

    def _determine_mime_type(self, file_path: Path) -> str:
        """Determine MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string (e.g., 'text/plain', 'image/png')
        """
        # First check if it's an image using ImageProcessor
        if self.image_processor.is_image(file_path):
            try:
                fmt = self.image_processor.get_format(file_path)
                return f"image/{fmt}"
            except ValueError:
                pass

        # Fall back to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def create_index(
        self,
        name: str,
        files: list[Path],
        show_progress: bool = True,
    ) -> VectorMetadata:
        """Create a SQLite vector index from files.

        Args:
            name: Index name
            files: List of files to index
            show_progress: Show progress bar

        Returns:
            VectorMetadata with index statistics

        Raises:
            VectorSearchError: If index creation fails
        """
        logger.info("Creating vector index '%s' from %d files", name, len(files))

        # Open SQLite storage
        storage = SQLiteStorage(name)

        try:
            # Create schema with vector support
            storage.create_schema(with_vectors=True)

            # Separate text files and image files
            text_files: list[Path] = []
            image_files: list[Path] = []
            file_doc_ids: dict[str, int] = {}

            # First pass: get or add documents to storage
            logger.info("Processing documents...")
            file_iterator = tqdm(files, desc="Processing documents") if show_progress else files

            for file_path in file_iterator:
                try:
                    # Check if document already exists (e.g., added by BM25Indexer)
                    existing_doc = storage.get_document(str(file_path))
                    if existing_doc is not None:
                        # Use existing document ID
                        file_doc_ids[str(file_path)] = existing_doc.id
                        # Determine if image or text based on existing mime_type
                        if existing_doc.mime_type.startswith("image/"):
                            image_files.append(file_path)
                        else:
                            text_files.append(file_path)
                        continue

                    md5_hash = compute_file_hash(file_path)
                    mime_type = self._determine_mime_type(file_path)
                    file_size = file_path.stat().st_size

                    # Determine if image or text
                    is_image = self.image_processor.is_image(file_path)

                    if is_image:
                        # Images don't store content
                        content = None
                        image_files.append(file_path)
                    else:
                        # Text files store content
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="ignore")
                        except Exception as e:
                            logger.warning("Failed to read file %s: %s", file_path, e)
                            continue
                        text_files.append(file_path)

                    # Add document to storage
                    doc_id = storage.add_document(
                        path=str(file_path),
                        filename=file_path.name,
                        md5_hash=md5_hash,
                        content=content,
                        mime_type=mime_type,
                        file_size=file_size,
                    )
                    file_doc_ids[str(file_path)] = doc_id

                except Exception as e:
                    logger.warning("Failed to process file %s: %s", file_path, e)
                    continue

            logger.info(
                "Added %d documents (%d text, %d images)",
                len(file_doc_ids),
                len(text_files),
                len(image_files),
            )

            if not file_doc_ids:
                raise VectorSearchError("No files were successfully indexed")

            # Initialize embeddings client
            embeddings_client = BedrockEmbeddings()

            # Create chunker pipeline for text files
            text_chunker = TextChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            char_limit_chunker = CharacterLimitChunker(max_chars=self.config.max_chunk_chars)
            pipeline = text_chunker | char_limit_chunker

            # Process text files: chunk and generate embeddings
            total_chunks = 0

            if text_files:
                logger.info("Chunking %d text files...", len(text_files))
                chunks = pipeline.chunk_files(text_files)
                logger.info("Created %d chunks from text files", len(chunks))

                if chunks:
                    # Generate embeddings for text chunks
                    logger.info("Generating embeddings for text chunks...")
                    chunk_texts = [chunk.text for chunk in chunks]

                    if show_progress:
                        batch_size = 50
                        all_embeddings: list[list[float]] = []

                        for i in tqdm(
                            range(0, len(chunk_texts), batch_size), desc="Embedding text"
                        ):
                            batch = chunk_texts[i : i + batch_size]
                            batch_embeddings = embeddings_client.embed_texts(batch)
                            all_embeddings.extend(batch_embeddings)
                    else:
                        all_embeddings = embeddings_client.embed_texts(chunk_texts)

                    # Add chunks with embeddings to storage
                    logger.info("Storing %d text chunks...", len(chunks))
                    chunk_iterator = (
                        tqdm(enumerate(chunks), total=len(chunks), desc="Storing chunks")
                        if show_progress
                        else enumerate(chunks)
                    )

                    for idx, chunk in chunk_iterator:
                        if idx < len(all_embeddings):
                            maybe_doc_id = file_doc_ids.get(chunk.source_path)
                            if maybe_doc_id is not None:
                                doc_id = maybe_doc_id
                                storage.add_chunk(
                                    document_id=doc_id,
                                    chunk_index=chunk.chunk_index,
                                    chunk_type="text",
                                    text=chunk.text,
                                    start_word=chunk.start_word,
                                    end_word=chunk.end_word,
                                    word_count=chunk.word_count,
                                    embedding=all_embeddings[idx],
                                )
                                total_chunks += 1

            # Process image files: generate embeddings
            if image_files:
                logger.info("Processing %d image files...", len(image_files))
                image_iterator = (
                    tqdm(image_files, desc="Embedding images") if show_progress else image_files
                )

                for file_path in image_iterator:
                    try:
                        maybe_doc_id = file_doc_ids.get(str(file_path))
                        if maybe_doc_id is None:
                            logger.debug("Skipping image %s: no document ID", file_path)
                            continue
                        doc_id = maybe_doc_id

                        # Prepare image for embedding
                        logger.debug("Preparing image for embedding: %s", file_path)
                        image_bytes, image_format = self.image_processor.prepare_for_embedding(
                            file_path
                        )
                        logger.debug(
                            "Image prepared: %s (format=%s, size=%d bytes)",
                            file_path.name,
                            image_format,
                            len(image_bytes),
                        )

                        # Generate embedding
                        logger.debug("Generating embedding for image: %s", file_path.name)
                        embedding = embeddings_client.embed_image(image_bytes, image_format)
                        logger.debug(
                            "Embedding generated for %s: %d dimensions",
                            file_path.name,
                            len(embedding),
                        )

                        # Add single chunk for image
                        storage.add_chunk(
                            document_id=doc_id,
                            chunk_index=0,
                            chunk_type="image",
                            text=None,
                            start_word=None,
                            end_word=None,
                            word_count=None,
                            embedding=embedding,
                        )
                        total_chunks += 1
                        logger.debug(
                            "Stored image chunk for %s (doc_id=%d)", file_path.name, doc_id
                        )

                    except Exception as e:
                        logger.warning("Failed to embed image %s: %s", file_path, e)
                        continue

            # Store metadata
            metadata_dict = {
                "chunk_count": total_chunks,
                "embedding_model": MODEL_ID,
                "dimensions": DIMENSIONS,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "total_tokens": embeddings_client.total_tokens,
                "estimated_cost_usd": embeddings_client.get_estimated_cost(),
                "text_file_count": len(text_files),
                "image_file_count": len(image_files),
            }

            storage.set_metadata("vector_metadata", json.dumps(metadata_dict))

            # Create metadata object
            metadata = VectorMetadata(
                chunk_count=total_chunks,
                embedding_model=MODEL_ID,
                dimensions=DIMENSIONS,
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

        finally:
            storage.close()

    def delete_index(self, name: str) -> bool:
        """Delete a vector index.

        Args:
            name: Index name

        Returns:
            True if deleted, False if not found
        """
        db_path = get_sqlite_db_path(name)

        if db_path.exists():
            db_path.unlink()
            logger.info("Deleted SQLite database: %s", db_path)
            return True

        return False

    def index_exists(self, name: str) -> bool:
        """Check if a vector index exists.

        Args:
            name: Index name

        Returns:
            True if index exists
        """
        return get_sqlite_db_path(name).exists()
