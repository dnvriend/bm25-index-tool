"""List command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from typing import Annotated

import typer

from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.registry import IndexRegistry

logger = get_logger(__name__)

app = typer.Typer()


@app.command(name="list")
def list_command(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    detailed: Annotated[
        bool,
        typer.Option("--detailed", "-d", help="Show detailed index information"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple or json"),
    ] = "simple",
) -> None:
    """List all available BM25 indices with optional metadata.

    Shows all created indices in the system. Use --detailed to see
    additional metadata like file count, creation date, and glob pattern.

    Examples:

    \b
        # Basic: List all indices
        bm25-index-tool list

    \b
        # Detailed: Show metadata for each index
        bm25-index-tool list --detailed

    \b
        # JSON output: Machine-readable format
        bm25-index-tool list --format json

    \b
        # JSON with details: Full metadata in JSON
        bm25-index-tool list --detailed --format json

    \b
    Output Format:
        simple: Plain text list of index names
        json: {"count": 3, "indices": ["vault", "notes", "docs"]}
        json + detailed: Full metadata for each index
    """
    setup_logging(verbose)
    logger.info("Listing all indices")

    registry = IndexRegistry()
    index_names = registry.list_indices()

    if not index_names:
        typer.echo("No indices found.")
        return

    if format == "json":
        # JSON output
        if detailed:
            logger.debug("Retrieving detailed metadata for %d indices", len(index_names))
            indices = [registry.get_index(name) for name in index_names]
            typer.echo(json.dumps({"count": len(indices), "indices": indices}, indent=2))
        else:
            typer.echo(json.dumps({"count": len(index_names), "indices": index_names}, indent=2))
    else:
        # Simple output
        typer.echo(f"Found {len(index_names)} indices:\n")
        for name in index_names:
            if detailed:
                logger.debug("Retrieving metadata for index: %s", name)
                metadata = registry.get_index(name)
                if metadata:
                    typer.echo(f"  {name}:")
                    typer.echo(f"    Files: {metadata['file_count']}")
                    typer.echo(f"    Created: {metadata['created_at']}")
                    typer.echo(f"    Pattern: {metadata['glob_pattern']}")
                    typer.echo("")
            else:
                typer.echo(f"  - {name}")

    logger.info("Listed %d indices", len(index_names))
