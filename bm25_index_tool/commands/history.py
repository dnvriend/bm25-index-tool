"""History management command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from typing import Annotated, Any

import typer

from bm25_index_tool.core.history import SearchHistory
from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

history_app = typer.Typer(help="Manage query history")


def _format_history_simple(entries: list[dict[str, Any]]) -> str:
    """Format history entries in simple text format.

    Args:
        entries: List of history entry dictionaries

    Returns:
        Formatted text output
    """
    if not entries:
        return "No history entries found."

    lines = []
    for entry in entries:
        timestamp = entry["timestamp"]
        query = entry["query"]
        indices = ", ".join(entry["indices"])
        result_count = entry["result_count"]
        elapsed = entry["elapsed_seconds"]

        lines.append(
            f"[{timestamp}] Query: '{query}' | Indices: {indices} | "
            f"Results: {result_count} | Time: {elapsed:.3f}s"
        )

    return "\n".join(lines)


def _format_history_json(entries: list[dict[str, Any]]) -> str:
    """Format history entries in JSON format.

    Args:
        entries: List of history entry dictionaries

    Returns:
        JSON formatted output
    """
    return json.dumps(entries, indent=2)


@history_app.command(name="show")
def show_history(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of entries to display"),
    ] = 20,
    search: Annotated[
        str | None,
        typer.Option("--search", "-s", help="Filter history by query pattern"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple or json"),
    ] = "simple",
) -> None:
    """Display recent search queries from SQLite history database.

    Shows the most recent queries with timestamps, indices searched,
    result counts, and execution times. Supports filtering by query pattern.

    Examples:

    \b
        # Show last 20 queries
        bm25-index-tool history show

    \b
        # Show last 50 queries
        bm25-index-tool history show --limit 50

    \b
        # Search history: Find kubernetes-related queries
        bm25-index-tool history show --search kubernetes

    \b
        # JSON output: Machine-readable format
        bm25-index-tool history show --format json

    \b
        # Filter and export: Pipe to jq
        bm25-index-tool history show -f json | jq '.[] | select(.index == "vault")'

    \b
    Output Format:
        simple: Human-readable table with timestamps
        json: Array of history entries
    """
    setup_logging(verbose)

    history = SearchHistory()

    try:
        if search:
            logger.info("Searching history for pattern: '%s'", search)
            entries = history.search(search, limit=limit)
        else:
            logger.info("Retrieving %d recent history entries", limit)
            entries = history.get_recent(limit=limit)

        # Format output
        if format == "json":
            output = _format_history_json(entries)
        else:
            output = _format_history_simple(entries)

        typer.echo(output)

        logger.info("Displayed %d history entries", len(entries))

    except Exception as e:
        logger.exception("Failed to retrieve history")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@history_app.command(name="clear")
def clear_history(
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
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Permanently delete all search history entries.

    Removes all query history from the SQLite database. This operation
    cannot be undone. Use with caution.

    Examples:

    \b
        # Interactive: Prompt for confirmation
        bm25-index-tool history clear

    \b
        # Force: Skip confirmation prompt
        bm25-index-tool history clear --force

    \b
        # Scripted: Non-interactive deletion
        bm25-index-tool history clear -f
    """
    setup_logging(verbose)

    history = SearchHistory()

    try:
        # Get count for confirmation
        count = history.count()

        if count == 0:
            typer.echo("History is already empty.")
            return

        # Confirm deletion unless --force
        if not force:
            confirm = typer.confirm(
                f"Delete all {count} history entries?",
                default=False,
            )
            if not confirm:
                typer.echo("Operation cancelled.")
                return

        # Clear history
        deleted = history.clear()
        typer.echo(f"Deleted {deleted} history entries.")
        logger.info("History cleared: %d entries deleted", deleted)

    except Exception as e:
        logger.exception("Failed to clear history")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@history_app.command(name="stats")
def show_stats(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
) -> None:
    """Display query history statistics.

    Shows the total number of recorded search queries in the history database.

    Examples:

    \b
        # Show total query count
        bm25-index-tool history stats

    \b
        # Verbose: See detailed logging
        bm25-index-tool history stats -vv
    """
    setup_logging(verbose)

    history = SearchHistory()

    try:
        count = history.count()
        typer.echo(f"Total history entries: {count}")
        logger.info("History statistics: %d entries", count)

    except Exception as e:
        logger.exception("Failed to retrieve history stats")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
