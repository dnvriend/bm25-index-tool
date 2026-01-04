"""Update command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import time
from typing import Annotated

import typer

from bm25_index_tool.core.file_discovery import discover_files
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.registry import IndexRegistry

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
) -> None:
    """Rebuild an existing BM25 index with current files.

    Performs a complete re-index using the original glob pattern. All files
    are re-discovered and re-indexed. Use this after adding, modifying, or
    deleting files from the source directory.

    Note: This is a full rebuild, not an incremental update. The entire
    index is recreated from scratch using the stored configuration.

    Examples:

    \b
        # Basic: Rebuild index with original settings
        bm25-index-tool update vault

    \b
        # Verbose: See detailed re-indexing progress
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

    # Update index
    logger.debug("Initializing BM25Indexer")
    indexer = BM25Indexer()
    try:
        logger.info("Starting index update for %d files", len(files))
        updated_metadata = indexer.update_index(name, files)

        elapsed = time.time() - start_time
        logger.info("Index updated successfully in %.2fs", elapsed)

        # Format output
        if format == "json":
            result = {
                "status": "success",
                "index": name,
                "file_count": updated_metadata.file_count,
                "elapsed_seconds": round(elapsed, 2),
            }
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(f"\nIndex '{name}' updated successfully!")
            typer.echo(f"Files indexed: {updated_metadata.file_count}")
            typer.echo(f"Time: {elapsed:.2f}s")

    except Exception as e:
        logger.exception("Failed to update index")
        if format == "json":
            typer.echo(json.dumps({"status": "error", "message": str(e)}), err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
