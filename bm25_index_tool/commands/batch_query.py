"""Batch query command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import typer

from bm25_index_tool.core.filters import PathFilter
from bm25_index_tool.core.merge_strategies import get_merge_strategy
from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer()


def _execute_single_query(
    searcher: BM25Searcher,
    index_names: list[str],
    query: str,
    top_k: int,
    rrf_k: int,
    extract_fragments_flag: bool,
    context_lines: int,
    path_filter: list[str] | None,
    exclude_path: list[str] | None,
    merge_strategy: str = "rrf",
    merge_params_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single query and return result with timing.

    Args:
        searcher: BM25Searcher instance
        index_names: List of index names
        query: Search query
        top_k: Number of results to return
        rrf_k: RRF k parameter for multi-index fusion
        extract_fragments_flag: If True, extract text fragments
        context_lines: Number of context lines
        path_filter: Include path patterns
        exclude_path: Exclude path patterns
        merge_strategy: Merge strategy name
        merge_params_dict: Merge strategy parameters

    Returns:
        Result dictionary with query, results, count, execution_time
    """
    start_time = time.time()

    try:
        # Search
        if len(index_names) == 1:
            results = searcher.search_single(
                index_names[0],
                query,
                top_k=top_k,
                extract_fragments_flag=extract_fragments_flag,
                context_lines=context_lines,
            )
        else:
            # Multi-index search with merge strategy
            strategy = get_merge_strategy(merge_strategy, **(merge_params_dict or {}))
            results = searcher.search_multi(
                index_names,
                query,
                top_k=top_k,
                rrf_k=rrf_k,
                extract_fragments_flag=extract_fragments_flag,
                context_lines=context_lines,
                merge_strategy=strategy,
            )

        # Apply path filtering if specified
        if path_filter or exclude_path:
            filter_obj = PathFilter(
                include_patterns=path_filter if path_filter else None,
                exclude_patterns=exclude_path if exclude_path else None,
            )
            results = filter_obj.filter_results(results)

    except Exception as e:
        logger.warning("Query failed: %s - %s", query, e)
        results = []

    execution_time = time.time() - start_time

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "execution_time": execution_time,
    }


@app.command(name="batch")
def batch_command(
    indices: Annotated[str, typer.Argument(help="Index name(s), comma-separated for multi-index")],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Input file with queries (one per line). If not specified, reads from stdin",
        ),
    ] = None,
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="Number of results to return per query"),
    ] = 10,
    rrf_k: Annotated[
        int,
        typer.Option(
            "--rrf-k",
            help="RRF k parameter for multi-index fusion (deprecated, use --merge-params)",
        ),
    ] = 60,
    merge_strategy: Annotated[
        str,
        typer.Option(
            "--merge-strategy",
            help="Merge strategy for multi-index: rrf, union, intersection, weighted",
        ),
    ] = "rrf",
    merge_params: Annotated[
        str | None,
        typer.Option(
            "--merge-params",
            help="JSON parameters for merge strategy (e.g., '{\"k\": 60}' for rrf)",
        ),
    ] = None,
    fragments: Annotated[
        bool,
        typer.Option(
            "--fragments", help="Extract text fragments showing matched terms with context"
        ),
    ] = False,
    context: Annotated[
        int,
        typer.Option("-C", "--context", help="Number of context lines before/after matches"),
    ] = 3,
    path_filter: Annotated[
        list[str],
        typer.Option(
            "--path-filter",
            "-pf",
            help="Include only paths matching glob pattern (can specify multiple)",
        ),
    ] = [],
    exclude_path: Annotated[
        list[str],
        typer.Option(
            "--exclude-path",
            "-ep",
            help="Exclude paths matching glob pattern (can specify multiple)",
        ),
    ] = [],
    parallel: Annotated[
        bool,
        typer.Option("--parallel", help="Enable parallel query execution"),
    ] = False,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Maximum number of parallel workers"),
    ] = 4,
) -> None:
    """Execute multiple queries efficiently with parallel processing.

    Reads queries from input file or stdin (one query per line) and outputs
    results in JSONL format (one JSON object per line). Supports parallel
    execution for improved throughput on multi-core systems.

    Examples:

    \b
        # From file: Process queries from file
        bm25-index-tool batch vault --input queries.txt

    \b
        # From stdin: Pipe queries from command
        cat queries.txt | bm25-index-tool batch vault

    \b
        # Echo queries: Inline query list
        echo -e "kubernetes\\ndocker\\naws" | bm25-index-tool batch vault

    \b
        # Parallel: Speed up with multiple workers
        bm25-index-tool batch vault -i queries.txt --parallel --max-workers 8

    \b
        # Multi-index: Search across multiple indices
        bm25-index-tool batch "vault,docs" -i queries.txt --parallel

    \b
        # With fragments: Include matched context
        bm25-index-tool batch vault -i queries.txt --fragments -C 5

    \b
        # Path filtering: Limit search scope
        bm25-index-tool batch vault -i queries.txt \
            --path-filter "reference/**"

    \b
        # Complex batch: All options combined
        bm25-index-tool batch "vault,docs" -i queries.txt \
            --parallel --max-workers 16 \
            --top 20 \
            --fragments -C 3 \
            --merge-strategy weighted \
            --merge-params '{"vault": 2.0, "docs": 1.0}'

    \b
    Output Format (JSONL):
        Each line is a JSON object:
        {
          "query": "kubernetes networking",
          "results": [{"path": "...", "score": 0.95, ...}],
          "count": 10,
          "execution_time": 0.023
        }

    \b
    Performance Tips:
        - Use --parallel for 50+ queries
        - Increase --max-workers for I/O-bound searches
        - Decrease --max-workers for CPU-bound searches
        - Default (4 workers) works well for most cases
    """
    setup_logging(verbose)

    # Parse index names
    index_names = [name.strip() for name in indices.split(",")]
    logger.info("Batch querying %d indices: %s", len(index_names), index_names)

    # Parse merge parameters
    merge_params_dict: dict[str, Any] = {}
    if merge_params:
        try:
            merge_params_dict = json.loads(merge_params)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid JSON in --merge-params: {e}", err=True)
            raise typer.Exit(code=1)
    elif merge_strategy == "rrf":
        # Use deprecated rrf_k if merge_params not provided
        merge_params_dict = {"k": rrf_k}

    # For weighted strategy, need to map index names to weights
    if merge_strategy == "weighted" and merge_params_dict:
        # Build weights dict with actual index names
        if not all(name in merge_params_dict for name in index_names):
            # If not all indices have weights, map by position
            if all(f"index_{i}" in merge_params_dict for i in range(len(index_names))):
                # Already in correct format
                pass
            else:
                # Try to map index names directly from params
                weights = {
                    f"index_{i}": merge_params_dict.get(name, 1.0)
                    for i, name in enumerate(index_names)
                }
                merge_params_dict = {"weights": weights}
        else:
            # Map index names to index_0, index_1, etc.
            weights = {f"index_{i}": merge_params_dict[name] for i, name in enumerate(index_names)}
            merge_params_dict = {"weights": weights}

    logger.debug("Merge strategy: %s, params: %s", merge_strategy, merge_params_dict)

    # Read queries
    queries: list[str] = []
    if input_file:
        logger.info("Reading queries from file: %s", input_file)
        try:
            with open(input_file) as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            typer.echo(f"Error reading input file: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        logger.info("Reading queries from stdin")
        queries = [line.strip() for line in sys.stdin if line.strip()]

    if not queries:
        typer.echo("Error: No queries provided", err=True)
        raise typer.Exit(code=1)

    logger.info("Processing %d queries (parallel=%s)", len(queries), parallel)

    # Initialize searcher
    searcher = BM25Searcher()

    # Execute queries
    if parallel:
        # Parallel execution with ThreadPoolExecutor
        logger.debug("Using %d parallel workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            futures = {
                executor.submit(
                    _execute_single_query,
                    searcher,
                    index_names,
                    query,
                    top,
                    rrf_k,
                    fragments,
                    context,
                    path_filter if path_filter else None,
                    exclude_path if exclude_path else None,
                    merge_strategy,
                    merge_params_dict,
                ): query
                for query in queries
            }

            # Process results as they complete
            for future in as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                    # Output JSONL
                    typer.echo(json.dumps(result, default=str))
                except Exception as e:
                    logger.error("Query '%s' failed: %s", query, e)
                    # Output error result
                    error_result = {
                        "query": query,
                        "results": [],
                        "count": 0,
                        "execution_time": 0.0,
                        "error": str(e),
                    }
                    typer.echo(json.dumps(error_result))
    else:
        # Sequential execution
        for query in queries:
            result = _execute_single_query(
                searcher,
                index_names,
                query,
                top,
                rrf_k,
                fragments,
                context,
                path_filter if path_filter else None,
                exclude_path if exclude_path else None,
                merge_strategy,
                merge_params_dict,
            )
            # Output JSONL
            typer.echo(json.dumps(result, default=str))

    logger.info("Batch query completed: %d queries processed", len(queries))
