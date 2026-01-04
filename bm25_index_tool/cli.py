"""CLI entry point for bm25-index-tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import atexit
from typing import Annotated

import typer

from bm25_index_tool.commands.batch_query import batch_command
from bm25_index_tool.commands.create import create_command
from bm25_index_tool.commands.delete import delete_command
from bm25_index_tool.commands.history import history_app
from bm25_index_tool.commands.info import info_command
from bm25_index_tool.commands.list import list_command
from bm25_index_tool.commands.query import query_command
from bm25_index_tool.commands.stats import stats_command
from bm25_index_tool.commands.update import update_command
from bm25_index_tool.completion import completion_app
from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.telemetry import TelemetryConfig, TelemetryService, traced

logger = get_logger(__name__)

app = typer.Typer(invoke_without_command=True, help="BM25 Index Tool - Fast text search with BM25")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo("bm25-index-tool version 0.1.0")
        raise typer.Exit()


def _shutdown_telemetry() -> None:
    """Shutdown telemetry on exit."""
    TelemetryService.get_instance().shutdown()


@traced("main")
def _run_main_command(verbose: int, ctx: typer.Context) -> None:
    """Execute main command logic with tracing - shows help by default."""
    logger.info("bm25-index-tool started")
    logger.debug("Running with verbose level: %d", verbose)

    # Show help text
    typer.echo(ctx.get_help())

    logger.info("bm25-index-tool completed")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    telemetry: Annotated[
        bool,
        typer.Option(
            "--telemetry",
            envvar="OTEL_ENABLED",
            help="Enable OpenTelemetry observability.",
        ),
    ] = False,
) -> None:
    """A Python CLI tool"""
    setup_logging(verbose)

    # Initialize telemetry
    config = TelemetryConfig.from_env()
    config.enabled = telemetry or config.enabled
    TelemetryService.get_instance().initialize(config)
    atexit.register(_shutdown_telemetry)

    if ctx.invoked_subcommand is None:
        _run_main_command(verbose, ctx)


# Add subcommands
app.command(name="create", help="Create a new BM25 index")(create_command)
app.command(name="query", help="Query BM25 indices")(query_command)
app.command(name="batch", help="Batch query multiple queries")(batch_command)
app.command(name="list", help="List all indices")(list_command)
app.command(name="info", help="Show index information")(info_command)
app.command(name="stats", help="Show index statistics")(stats_command)
app.command(name="update", help="Update an index")(update_command)
app.command(name="delete", help="Delete an index")(delete_command)
app.add_typer(completion_app, name="completion", help="Shell completion")
app.add_typer(history_app, name="history", help="Manage query history")


if __name__ == "__main__":
    app()
