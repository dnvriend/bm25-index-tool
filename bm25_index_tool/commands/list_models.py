"""List models command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from typing import Annotated, Any

import typer

from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer()

# Model information - mirrors embeddings.py but avoids importing vector deps
MODELS: list[dict[str, Any]] = [
    {
        "id": "amazon.nova-2-multimodal-embeddings-v1:0",
        "name": "Amazon Nova Embeddings",
        "dimensions": 3072,
        "configurable_dimensions": [256, 512, 1024, 3072],
        "price_per_1k_tokens": 0.00018,
        "default": True,
        "notes": "Recommended. Query/document embedding purposes. Region: us-east-1.",
    },
    {
        "id": "amazon.titan-embed-text-v2:0",
        "name": "Amazon Titan Text Embeddings V2",
        "dimensions": 1024,
        "configurable_dimensions": None,
        "price_per_1k_tokens": 0.0002,
        "default": False,
        "notes": "Good general-purpose model. Widely available.",
    },
    {
        "id": "amazon.titan-embed-text-v1",
        "name": "Amazon Titan Text Embeddings V1",
        "dimensions": 1536,
        "configurable_dimensions": None,
        "price_per_1k_tokens": 0.0001,
        "default": False,
        "notes": "Legacy model.",
    },
    {
        "id": "cohere.embed-english-v3",
        "name": "Cohere Embed English V3",
        "dimensions": 1024,
        "configurable_dimensions": None,
        "price_per_1k_tokens": 0.0001,
        "default": False,
        "notes": "English-only, high quality.",
    },
    {
        "id": "cohere.embed-multilingual-v3",
        "name": "Cohere Embed Multilingual V3",
        "dimensions": 1024,
        "configurable_dimensions": None,
        "price_per_1k_tokens": 0.0001,
        "default": False,
        "notes": "Supports 100+ languages.",
    },
]


@app.command(name="list-models")
def list_models_command(
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
        typer.Option("--format", "-f", help="Output format: simple or json"),
    ] = "simple",
) -> None:
    """List available embedding models for vector search.

    Shows all supported AWS Bedrock embedding models that can be used
    with the --model option when creating indices.

    Examples:

    \b
        # List all models
        bm25-index-tool list-models

    \b
        # JSON output
        bm25-index-tool list-models --format json

    \b
        # Use a specific model when creating an index
        bm25-index-tool create myindex -p "**/*.md" \\
            --model amazon.titan-embed-text-v2:0
    """
    setup_logging(verbose)
    logger.info("Listing available embedding models")

    if format == "json":
        typer.echo(json.dumps({"models": MODELS}, indent=2))
    else:
        typer.echo("Available Embedding Models:\n")
        for model in MODELS:
            default_marker = " (default)" if model["default"] else ""
            typer.echo(f"  {model['id']}{default_marker}")
            typer.echo(f"    Name: {model['name']}")
            typer.echo(f"    Dimensions: {model['dimensions']}")
            config_dims = model["configurable_dimensions"]
            if config_dims is not None:
                dims = ", ".join(str(d) for d in config_dims)
                typer.echo(f"    Configurable: {dims}")
            typer.echo(f"    Price: ${model['price_per_1k_tokens']:.5f}/1K tokens")
            typer.echo(f"    Notes: {model['notes']}")
            typer.echo("")

    logger.info("Listed %d models", len(MODELS))
