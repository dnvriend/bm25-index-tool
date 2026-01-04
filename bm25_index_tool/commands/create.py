"""Create command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import time
from typing import Annotated

import typer

from bm25_index_tool.config.models import BM25Params, BM25Profile, TokenizationConfig
from bm25_index_tool.core.file_discovery import discover_files
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer()


@app.command(name="create")
def create_command(
    name: Annotated[str, typer.Argument(help="Index name")],
    pattern: Annotated[str, typer.Option("--pattern", "-p", help="Glob pattern for files")],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    profile: Annotated[
        str,
        typer.Option("--profile", help="BM25 profile: standard or code"),
    ] = "standard",
    k1: Annotated[
        float | None,
        typer.Option("--k1", help="Custom k1 parameter (overrides profile)"),
    ] = None,
    b: Annotated[
        float | None,
        typer.Option("--b", help="Custom b parameter (overrides profile)"),
    ] = None,
    stemmer: Annotated[
        str,
        typer.Option("--stemmer", help="Stemmer language (empty = disabled)"),
    ] = "",
    no_gitignore: Annotated[
        bool,
        typer.Option("--no-gitignore", help="Disable .gitignore respect"),
    ] = False,
) -> None:
    """Create a new BM25 index from files matching a glob pattern.

    Creates a full-text search index using the BM25 ranking algorithm. Supports
    recursive file discovery with glob patterns, .gitignore filtering, custom
    BM25 parameters, and stemming.

    Examples:

    \b
        # Basic: Index all markdown files in current directory
        bm25-index-tool create notes --pattern "*.md"

    \b
        # Recursive: Index all Python files in project
        bm25-index-tool create myproject --pattern "**/*.py"

    \b
        # Tilde expansion: Index Obsidian vault
        bm25-index-tool create vault --pattern "~/projects/obsidian/**/*.md"

    \b
        # Environment variables: Index from configured path
        bm25-index-tool create docs --pattern "$DOCS_DIR/**/*.txt"

    \b
        # Code profile: Better for source code (lower term frequency weight)
        bm25-index-tool create codebase --pattern "src/**/*.py" --profile code

    \b
        # Custom parameters: Fine-tune BM25 scoring
        bm25-index-tool create custom --pattern "docs/**/*.md" \
            --k1 1.5 --b 0.75

    \b
        # With stemming: Enable English word stemming
        bm25-index-tool create stemmed --pattern "**/*.txt" --stemmer english

    \b
        # Ignore .gitignore: Index everything including ignored files
        bm25-index-tool create all-files --pattern "**/*" --no-gitignore

    \b
        # Verbose: See detailed indexing progress
        bm25-index-tool create myindex --pattern "**/*.md" -vv
    """
    setup_logging(verbose)
    logger.info("Creating index: %s", name)

    start_time = time.time()

    typer.echo(f"Creating index '{name}'...")
    logger.debug("Pattern: %s", pattern)

    # Determine BM25 parameters
    if k1 is not None and b is not None:
        # Custom parameters
        params = BM25Params(k1=k1, b=b)
        logger.debug("Using custom parameters: k1=%s, b=%s", k1, b)
        typer.echo(f"Using custom parameters: k1={k1}, b={b}")
    else:
        # Use profile
        try:
            bm25_profile = BM25Profile(profile)
            params = BM25Params.from_profile(bm25_profile)
            logger.debug("Using profile: %s (k1=%s, b=%s)", profile, params.k1, params.b)
            typer.echo(f"Using profile: {profile} (k1={params.k1}, b={params.b})")
        except ValueError:
            logger.warning("Invalid profile '%s', using 'standard'", profile)
            typer.echo(f"Invalid profile '{profile}'. Using 'standard'.", err=True)
            params = BM25Params.from_profile(BM25Profile.STANDARD)

    # Tokenization config
    tokenization = TokenizationConfig(stemmer=stemmer)
    if stemmer:
        typer.echo(f"Stemmer: {stemmer}")
    else:
        typer.echo("Stemmer: disabled")

    # Discover files
    typer.echo(f"Discovering files with pattern: {pattern}")
    logger.debug("respect_gitignore=%s", not no_gitignore)
    try:
        files = discover_files(pattern, respect_gitignore=not no_gitignore)
        logger.info("Found %d files to index", len(files))
        typer.echo(f"Found {len(files)} files")
    except ValueError as e:
        logger.error("File discovery failed: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Create index
    logger.debug("Initializing BM25Indexer")
    indexer = BM25Indexer()
    try:
        logger.info("Starting indexing process for %d files", len(files))
        metadata = indexer.create_index(
            name=name,
            files=files,
            params=params,
            tokenization=tokenization,
            glob_pattern=pattern,
        )

        elapsed = time.time() - start_time
        logger.info("Index created successfully in %.2fs", elapsed)
        typer.echo(f"\nIndex '{name}' created successfully!")
        typer.echo(f"Files indexed: {metadata.file_count}")
        typer.echo(f"Time: {elapsed:.2f}s")

    except ValueError as e:
        logger.error("Index creation failed: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Failed to create index")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
