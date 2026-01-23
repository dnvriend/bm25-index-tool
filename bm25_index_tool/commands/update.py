"""Update command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import time
from typing import Annotated

import typer

from bm25_index_tool.config.models import VectorConfig
from bm25_index_tool.core.file_discovery import discover_files
from bm25_index_tool.core.incremental import ChangeSet, IncrementalIndexer
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage, compute_file_hash

logger = get_logger(__name__)

app = typer.Typer()


@app.command(name="update")
def update_command(
    name: Annotated[str, typer.Argument(help="Index name to update")],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple, json"),
    ] = "simple",
    reindex: Annotated[
        bool,
        typer.Option("--reindex", help="Full rebuild instead of incremental update"),
    ] = False,
) -> None:
    """Update an existing BM25 index with current files.

    By default, performs incremental update - only processes files that have
    been added, modified, or deleted since the last index. This is much faster
    than a full rebuild for large indices.

    Use --reindex for a complete rebuild from scratch. This is useful when
    the index appears corrupted or you want to ensure a clean state.

    Examples:

    \b
        # Incremental update (default): Only process changed files
        bm25-index-tool update vault

    \b
        # Full rebuild: Delete and recreate entire index
        bm25-index-tool update vault --reindex

    \b
        # Verbose: See detailed progress
        bm25-index-tool update notes -vv

    \b
        # JSON output: Machine-readable update result
        bm25-index-tool update docs --format json

    \b
    Output Format:
        simple: Updated file count and time elapsed
        json: {"status": "success", "index": "vault", "file_count": 1234, ...}
    """
    setup_logging(verbose)
    logger.info("Updating index: %s", name)

    registry = IndexRegistry()

    # Check if index exists
    if not registry.index_exists(name):
        logger.error("Index not found: %s", name)
        error_msg = f"Error: Index '{name}' not found."
        if format == "json":
            typer.echo(json.dumps({"status": "error", "message": error_msg}), err=True)
        else:
            typer.echo(error_msg, err=True)
        raise typer.Exit(code=1)

    start_time = time.time()
    if format != "json":
        typer.echo(f"Updating index '{name}'...")

    # Get existing metadata
    metadata_dict = registry.get_index(name)
    if not metadata_dict:
        logger.error("Failed to load metadata for index: %s", name)
        error_msg = f"Error: Failed to load metadata for '{name}'."
        if format == "json":
            typer.echo(json.dumps({"status": "error", "message": error_msg}), err=True)
        else:
            typer.echo(error_msg, err=True)
        raise typer.Exit(code=1)

    glob_pattern = metadata_dict["glob_pattern"]
    logger.debug("Re-discovering files with pattern: %s", glob_pattern)
    if format != "json":
        typer.echo(f"Re-discovering files with pattern: {glob_pattern}")

    # Discover files
    try:
        files = discover_files(glob_pattern, respect_gitignore=True)
        logger.info("Found %d files to re-index", len(files))
        if format != "json":
            typer.echo(f"Found {len(files)} files")
    except ValueError as e:
        logger.error("File discovery failed: %s", e)
        if format == "json":
            typer.echo(json.dumps({"status": "error", "message": str(e)}), err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Handle incremental vs full reindex
    indexer = BM25Indexer()
    vector_metadata_dict = metadata_dict.get("vector_metadata")
    changes: ChangeSet | None = None
    added_count = 0
    modified_count = 0
    deleted_count = 0

    if not reindex:
        # Incremental update: detect changes
        logger.debug("Detecting changes for incremental update")
        incremental = IncrementalIndexer()

        with SQLiteStorage(name) as storage:
            changes = incremental.detect_changes(files, storage)

        # detect_changes always returns ChangeSet, never None
        assert changes is not None
        added_count = len(changes.added)
        modified_count = len(changes.modified)
        deleted_count = len(changes.deleted)

        if not changes.added and not changes.modified and not changes.deleted:
            elapsed = time.time() - start_time
            logger.info("Index is up to date, no changes detected")
            if format == "json":
                result: dict[str, str | int | float | bool] = {
                    "status": "success",
                    "index": name,
                    "up_to_date": True,
                    "added": 0,
                    "modified": 0,
                    "deleted": 0,
                    "elapsed_seconds": round(elapsed, 2),
                }
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo("Index is up to date, no changes detected.")
            return

        if format != "json":
            typer.echo(
                f"Changes detected: {added_count} added, "
                f"{modified_count} modified, {deleted_count} deleted"
            )
        logger.info(
            "Changes detected: %d added, %d modified, %d deleted",
            added_count,
            modified_count,
            deleted_count,
        )

        # Apply incremental changes to BM25 index
        try:
            with SQLiteStorage(name) as storage:
                # Delete removed and modified documents
                for deleted_path in changes.deleted:
                    storage.delete_document(deleted_path)
                    logger.debug("Deleted document: %s", deleted_path)

                for modified_path in changes.modified:
                    storage.delete_document(str(modified_path.resolve()))
                    logger.debug("Deleted modified document for re-add: %s", modified_path)

                # Add new and modified documents
                files_to_add = changes.added + changes.modified
                for file_path in files_to_add:
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        md5_hash = compute_file_hash(file_path)
                        file_size = file_path.stat().st_size

                        storage.add_document(
                            path=str(file_path.resolve()),
                            filename=file_path.name,
                            md5_hash=md5_hash,
                            content=content,
                            mime_type="text/plain",
                            file_size=file_size,
                        )
                        logger.debug("Added document: %s", file_path)
                    except Exception as e:
                        logger.warning("Failed to add document %s: %s", file_path, e)

            if format != "json":
                typer.echo("BM25 index updated!")

        except Exception as e:
            logger.exception("Failed to update BM25 index incrementally")
            if format == "json":
                typer.echo(json.dumps({"status": "error", "message": str(e)}), err=True)
            else:
                typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # For incremental, get updated metadata from registry
        updated_metadata_dict = registry.get_index(name)
        if updated_metadata_dict:
            from bm25_index_tool.config.models import IndexMetadata

            updated_metadata = IndexMetadata(**updated_metadata_dict)
            # Update file count based on current storage
            with SQLiteStorage(name) as storage:
                updated_metadata.file_count = storage.get_document_count()
            indexer.update_metadata(name, updated_metadata)
        else:
            # Fallback: create metadata
            from bm25_index_tool.config.models import IndexMetadata

            updated_metadata = IndexMetadata(**metadata_dict)
            with SQLiteStorage(name) as storage:
                updated_metadata.file_count = storage.get_document_count()

    else:
        # Full reindex: existing behavior
        logger.debug("Performing full reindex")
        if format != "json":
            typer.echo("Performing full reindex...")

        try:
            logger.info("Starting BM25 full reindex for %d files", len(files))
            updated_metadata = indexer.update_index(name, files)

            if format != "json":
                typer.echo("BM25 index rebuilt!")

        except Exception as e:
            logger.exception("Failed to update BM25 index")
            if format == "json":
                typer.echo(json.dumps({"status": "error", "message": str(e)}), err=True)
            else:
                typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    # Check if vector index existed and update it
    if vector_metadata_dict:
        if format != "json":
            typer.echo("\nUpdating vector index...")
        try:
            from bm25_index_tool.vector import VectorIndexer

            # Recreate config from existing metadata
            vector_config = VectorConfig(
                chunk_size=vector_metadata_dict["chunk_size"],
                chunk_overlap=vector_metadata_dict["chunk_overlap"],
            )

            vector_indexer = VectorIndexer(config=vector_config)

            if not reindex and changes is not None:
                # Incremental vector update
                # For simplicity, delete chunks for modified/deleted docs and re-embed changed files
                with SQLiteStorage(name) as storage:
                    from bm25_index_tool.vector.chunking import CharacterLimitChunker, TextChunker
                    from bm25_index_tool.vector.embeddings import BedrockEmbeddings

                    # Delete chunks for deleted/modified documents
                    for deleted_path in changes.deleted:
                        doc = storage.get_document(deleted_path)
                        if doc:
                            storage.delete_chunks_for_document(doc.id)

                    for modified_path in changes.modified:
                        doc = storage.get_document(str(modified_path.resolve()))
                        if doc:
                            storage.delete_chunks_for_document(doc.id)

                    # Re-embed added and modified files
                    files_to_embed = changes.added + changes.modified
                    if files_to_embed:
                        embeddings_client = BedrockEmbeddings()
                        text_chunker = TextChunker(
                            chunk_size=vector_config.chunk_size,
                            chunk_overlap=vector_config.chunk_overlap,
                        )
                        char_limit_chunker = CharacterLimitChunker(
                            max_chars=vector_config.max_chunk_chars
                        )
                        pipeline = text_chunker | char_limit_chunker

                        chunks = pipeline.chunk_files(files_to_embed)
                        if chunks:
                            chunk_texts = [chunk.text for chunk in chunks]
                            embeddings = embeddings_client.embed_texts(chunk_texts)

                            for idx, chunk in enumerate(chunks):
                                if idx < len(embeddings):
                                    doc = storage.get_document(chunk.source_path)
                                    if doc:
                                        storage.add_chunk(
                                            document_id=doc.id,
                                            chunk_index=chunk.chunk_index,
                                            chunk_type="text",
                                            text=chunk.text,
                                            start_word=chunk.start_word,
                                            end_word=chunk.end_word,
                                            word_count=chunk.word_count,
                                            embedding=embeddings[idx],
                                        )

                    # Update vector metadata
                    chunk_count = storage.get_chunk_count()
                    from bm25_index_tool.config.models import VectorMetadata
                    from bm25_index_tool.vector.embeddings import DIMENSIONS, MODEL_ID

                    updated_metadata.vector_metadata = VectorMetadata(
                        chunk_count=chunk_count,
                        embedding_model=MODEL_ID,
                        dimensions=DIMENSIONS,
                        chunk_size=vector_config.chunk_size,
                        chunk_overlap=vector_config.chunk_overlap,
                    )
                    indexer.update_metadata(name, updated_metadata)

                if format != "json":
                    typer.echo("Vector index updated!")
            else:
                # Full vector rebuild
                vector_indexer.delete_index(name)
                vector_metadata = vector_indexer.create_index(name=name, files=files)
                updated_metadata.vector_metadata = vector_metadata
                indexer.update_metadata(name, updated_metadata)

                if format != "json":
                    typer.echo("Vector index rebuilt!")

        except ImportError:
            logger.warning("Vector dependencies not installed, skipping vector index update")
            if format != "json":
                typer.echo(
                    "Warning: Vector dependencies not installed. Vector index not updated.",
                    err=True,
                )
        except Exception as e:
            logger.exception("Failed to update vector index")
            if format != "json":
                typer.echo(f"Warning: Vector index update failed: {e}", err=True)

    elapsed = time.time() - start_time
    logger.info("Index updated successfully in %.2fs", elapsed)

    # Format output
    if format == "json":
        result_dict: dict[str, str | int | float | bool] = {
            "status": "success",
            "index": name,
            "file_count": updated_metadata.file_count,
            "elapsed_seconds": round(elapsed, 2),
        }
        if not reindex and changes is not None:
            result_dict["incremental"] = True
            result_dict["added"] = added_count
            result_dict["modified"] = modified_count
            result_dict["deleted"] = deleted_count
        else:
            result_dict["incremental"] = False
        if updated_metadata.vector_metadata:
            result_dict["chunk_count"] = updated_metadata.vector_metadata.chunk_count
        typer.echo(json.dumps(result_dict, indent=2))
    else:
        typer.echo(f"\nIndex '{name}' updated successfully!")
        typer.echo(f"Files indexed: {updated_metadata.file_count}")
        if not reindex and changes is not None:
            typer.echo(
                f"Changes: {added_count} added, {modified_count} modified, {deleted_count} deleted"
            )
        if updated_metadata.vector_metadata:
            typer.echo(f"Chunks: {updated_metadata.vector_metadata.chunk_count}")
        typer.echo(f"Time: {elapsed:.2f}s")
