"""Delete command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import shutil
from typing import Annotated

import typer

from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry

logger = get_logger(__name__)

app = typer.Typer()


@app.command(name="delete")
def delete_command(
    name: Annotated[str, typer.Argument(help="Index name to delete")],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    force: Annotated[
        bool,
        typer.Option("--force", help="Skip confirmation prompt"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple, json"),
    ] = "simple",
) -> None:
    """Permanently delete a BM25 index and all associated files.

    Removes the index from the registry and deletes all stored data including
    the corpus, vocabulary, and BM25 matrices. This operation cannot be undone.

    Examples:

    \b
        # Interactive: Prompt for confirmation
        bm25-index-tool delete oldindex

    \b
        # Force: Skip confirmation prompt
        bm25-index-tool delete oldindex --force

    \b
        # JSON output: Machine-readable result
        bm25-index-tool delete oldindex --force --format json

    \b
    Output Format:
        simple: Confirmation message with deleted path
        json: {"status": "success", "index": "name", "deleted_path": "..."}
    """
    setup_logging(verbose)
    logger.info("Deleting index: %s", name)

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

    # Confirm deletion (skip in JSON mode)
    if not force and format != "json":
        confirm = typer.confirm(f"Delete index '{name}'?")
        if not confirm:
            if format == "json":
                typer.echo(json.dumps({"status": "cancelled", "index": name}))
            else:
                typer.echo("Cancelled.")
            return

    # Delete index directory
    index_dir = get_index_dir(name)
    deleted_path = None
    if index_dir.exists():
        logger.debug("Removing index directory: %s", index_dir)
        deleted_path = str(index_dir)
        shutil.rmtree(index_dir)
        if format != "json":
            typer.echo(f"Deleted index directory: {index_dir}")

    # Remove from registry
    logger.debug("Removing index from registry")
    registry.remove_index(name)

    logger.info("Index deleted successfully: %s", name)

    # Format output
    if format == "json":
        result = {
            "status": "success",
            "index": name,
            "deleted_path": deleted_path,
        }
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Index '{name}' deleted successfully.")
