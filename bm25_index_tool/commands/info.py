"""Info command for BM25 index tool.

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


@app.command(name="info")
def info_command(
    name: Annotated[str, typer.Argument(help="Index name")],
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
    """Display comprehensive metadata for a specific BM25 index.

    Shows creation date, file count, glob pattern, BM25 parameters,
    tokenization settings, and storage location.

    Examples:

    \b
        # Basic: Show index information
        bm25-index-tool info vault

    \b
        # JSON output: Machine-readable metadata
        bm25-index-tool info notes --format json

    \b
        # Pipe to jq: Extract specific fields
        bm25-index-tool info vault -f json | jq '.file_count'

    \b
    Output Format:
        simple: Human-readable formatted text
        json: Complete metadata as JSON object
    """
    setup_logging(verbose)
    logger.info("Getting info for index: %s", name)

    registry = IndexRegistry()

    metadata = registry.get_index(name)
    if not metadata:
        logger.error("Index not found: %s", name)
        typer.echo(f"Error: Index '{name}' not found.", err=True)
        raise typer.Exit(code=1)

    logger.debug("Retrieved metadata for index: %s", name)

    # Format output
    if format == "json":
        output = json.dumps(metadata, indent=2)
    else:
        # Simple format
        lines = []
        lines.append(f"\nIndex: {metadata['name']}")
        lines.append(f"Created: {metadata['created_at']}")
        lines.append(f"Files: {metadata['file_count']:,}")
        lines.append(f"Pattern: {metadata['glob_pattern']}")
        lines.append("\nBM25 Parameters:")
        lines.append(f"  k1: {metadata['bm25_params']['k1']}")
        lines.append(f"  b: {metadata['bm25_params']['b']}")
        lines.append(f"  method: {metadata['bm25_params']['method']}")
        lines.append("\nTokenization:")
        stemmer = metadata["tokenization"]["stemmer"]
        lines.append(f"  Stemmer: {stemmer if stemmer else 'disabled'}")
        lines.append(f"  Stopwords: {metadata['tokenization']['stopwords']}")

        # Vector search metadata
        vector_metadata = metadata.get("vector_metadata")
        if vector_metadata:
            lines.append("\nVector Search:")
            lines.append(f"  Model: {vector_metadata['embedding_model']}")
            lines.append(f"  Chunks: {vector_metadata['chunk_count']:,}")
            lines.append(f"  Dimensions: {vector_metadata['dimensions']}")
            lines.append(f"  Chunk Size: {vector_metadata['chunk_size']} words")
            lines.append(f"  Chunk Overlap: {vector_metadata['chunk_overlap']} words")
            lines.append(f"  Estimated Cost: ${vector_metadata['estimated_cost_usd']:.4f}")
        else:
            lines.append("\nVector Search: not available")

        output = "\n".join(lines)

    typer.echo(output)
