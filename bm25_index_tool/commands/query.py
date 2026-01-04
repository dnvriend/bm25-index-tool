"""Query command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
import time
from typing import Annotated, Any

import typer

from bm25_index_tool.core.cache import SearchCache
from bm25_index_tool.core.filters import PathFilter
from bm25_index_tool.core.formatters import format_json, format_rich, format_simple
from bm25_index_tool.core.history import SearchHistory
from bm25_index_tool.core.merge_strategies import get_merge_strategy
from bm25_index_tool.core.related import RelatedDocumentFinder
from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer()

# Global cache and history instances (singleton pattern)
_cache: SearchCache | None = None
_history: SearchHistory | None = None


def get_cache() -> SearchCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = SearchCache()
    return _cache


def get_history() -> SearchHistory:
    """Get or create global history instance."""
    global _history
    if _history is None:
        _history = SearchHistory()
    return _history


@app.command(name="query")
def query_command(
    indices: Annotated[str, typer.Argument(help="Index name(s), comma-separated for multi-index")],
    query: Annotated[str | None, typer.Argument(help="Search query")] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output (use -v for INFO, -vv for DEBUG, -vvv for TRACE)",
        ),
    ] = 0,
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="Number of results to return"),
    ] = 10,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple, json, rich"),
    ] = "simple",
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
    include_content: Annotated[
        bool,
        typer.Option(
            "--include-content",
            "-c",
            help="Include full document content in results",
        ),
    ] = False,
    content_max_length: Annotated[
        int,
        typer.Option(
            "--content-max-length",
            help="Maximum content length in characters (default: 500)",
        ),
    ] = 500,
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
    related_to: Annotated[
        str | None,
        typer.Option(
            "--related-to",
            help="Find documents related to this document path (relative to index)",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable search result caching"),
    ] = False,
    no_history: Annotated[
        bool,
        typer.Option("--no-history", help="Disable logging query to history"),
    ] = False,
) -> None:
    """Search BM25 indices with powerful filtering and multi-index support.

    Performs full-text search across one or more indices using BM25 ranking.
    Supports path filtering, content extraction, related document discovery,
    caching, and multiple merge strategies for combining results.

    Examples:

    \b
        # Basic: Search single index
        bm25-index-tool query vault "kubernetes networking"

    \b
        # Top results: Get more matches
        bm25-index-tool query notes "python async" --top 20

    \b
        # JSON output: Machine-readable format
        bm25-index-tool query vault "docker" --format json

    \b
        # Rich output: Colored table with syntax highlighting
        bm25-index-tool query vault "aws lambda" --format rich

    \b
        # Include content: Show full file content
        bm25-index-tool query notes "meeting notes" --include-content

    \b
        # Truncate content: Limit content length
        bm25-index-tool query docs "api" -c --content-max-length 200

    \b
        # Path filter: Search only in specific subdirectories
        bm25-index-tool query vault "terraform" \
            --path-filter "reference/aws/**" \
            --path-filter "projects/**"

    \b
        # Exclude paths: Skip drafts and archives
        bm25-index-tool query notes "todo" \
            --exclude-path "**/*.draft.md" \
            --exclude-path "archive/**"

    \b
        # Related documents: Find similar files
        bm25-index-tool query vault \
            --related-to "reference/k8s/networking.md" --top 10

    \b
        # Multi-index: Search across multiple indices (RRF merge)
        bm25-index-tool query "vault,work,personal" "project ideas"

    \b
        # Union merge: Combine all results from all indices
        bm25-index-tool query "vault,docs" "api" --merge-strategy union

    \b
        # Intersection merge: Only docs in all indices
        bm25-index-tool query "vault,docs" "kubernetes" \
            --merge-strategy intersection

    \b
        # Weighted merge: Prioritize one index
        bm25-index-tool query "vault,work" "meeting" \
            --merge-strategy weighted \
            --merge-params '{"vault": 2.0, "work": 1.0}'

    \b
        # Fragments: Show matching context lines
        bm25-index-tool query vault "error handling" --fragments -C 5

    \b
        # Disable caching: Force fresh search
        bm25-index-tool query vault "latest updates" --no-cache

    \b
        # Disable history: Don't log this query
        bm25-index-tool query vault "private search" --no-history

    \b
        # Complex query: Combine multiple features
        bm25-index-tool query "vault,docs" "kubernetes ingress" \
            --top 15 \
            --format rich \
            --path-filter "**/*.md" \
            --exclude-path "archive/**" \
            --fragments \
            -vv

    \b
    Output Format:
        simple: Plain text with file paths and scores
        json: {"results": [{"path": "...", "score": 0.95, ...}]}
        rich: Colored table with syntax highlighting
    """
    setup_logging(verbose)

    # Handle related documents mode
    if related_to:
        # Related mode: only supports single index
        index_names = [name.strip() for name in indices.split(",")]
        if len(index_names) > 1:
            typer.echo("Error: --related-to only supports single index queries", err=True)
            raise typer.Exit(code=1)

        logger.info("Finding documents related to '%s' in index '%s'", related_to, index_names[0])

        finder = RelatedDocumentFinder()
        try:
            results = finder.find_related(
                index_name=index_names[0],
                document_path=related_to,
                top_k=top,
                extract_fragments_flag=fragments,
                context_lines=context,
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            logger.exception("Related document search failed")
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Apply path filtering
        if path_filter or exclude_path:
            filter_obj = PathFilter(
                include_patterns=path_filter if path_filter else None,
                exclude_patterns=exclude_path if exclude_path else None,
            )
            results = filter_obj.filter_results(results)

        # Format output
        if format == "json":
            output = format_json(results, include_content, content_max_length)
        elif format == "rich":
            output = format_rich(
                results,
                include_content=include_content,
                content_max_length=content_max_length,
            )
        else:
            output = format_simple(results, include_content, content_max_length)

        typer.echo(output)
        return

    # Normal query mode: query is required
    if query is None:
        typer.echo("Error: query argument is required when not using --related-to", err=True)
        raise typer.Exit(code=1)

    # Parse index names
    index_names = [name.strip() for name in indices.split(",")]
    logger.info("Querying %d indices: %s", len(index_names), index_names)
    logger.debug("Query: %s", query)
    logger.debug("Top results: %d", top)

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

    # Check cache first (if enabled and no filtering/fragments)
    cache_enabled = not no_cache
    results_cached: list[dict[str, Any]] | None = None
    cache_hit = False

    if cache_enabled and not fragments and not path_filter and not exclude_path:
        cache = get_cache()
        results_cached = cache.get(
            indices=index_names,
            query=query,
            top_k=top,
            path_filter=None,
            exclude_path=None,
        )
        if results_cached is not None:
            cache_hit = True
            logger.info("Cache hit for query: '%s'", query)

    # Execute search if not cached
    start_time = time.time()

    if results_cached is None:
        searcher = BM25Searcher()

        try:
            if len(index_names) == 1:
                # Single index search
                results_cached = searcher.search_single(
                    index_names[0],
                    query,
                    top_k=top,
                    extract_fragments_flag=fragments,
                    context_lines=context,
                )
            else:
                # Multi-index search with merge strategy
                typer.echo(f"Searching {len(index_names)} indices with {merge_strategy} merge...")

                # Create merge strategy instance
                strategy = get_merge_strategy(merge_strategy, **merge_params_dict)

                results_cached = searcher.search_multi(
                    index_names,
                    query,
                    top_k=top,
                    rrf_k=rrf_k,  # Still passed for backward compatibility
                    extract_fragments_flag=fragments,
                    context_lines=context,
                    merge_strategy=strategy,
                )

        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            logger.exception("Search failed")
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    elapsed_seconds = time.time() - start_time

    # Assign to results for further processing
    results = results_cached

    # Apply path filtering
    if path_filter or exclude_path:
        filter_obj = PathFilter(
            include_patterns=path_filter if path_filter else None,
            exclude_patterns=exclude_path if exclude_path else None,
        )
        results = filter_obj.filter_results(results)

    # Store in cache if enabled and was not from cache
    if cache_enabled and not cache_hit and not fragments and not path_filter and not exclude_path:
        cache = get_cache()
        cache.set(
            indices=index_names,
            query=query,
            top_k=top,
            results=results,
            path_filter=None,
            exclude_path=None,
        )

    # Log to history if enabled
    if not no_history:
        history = get_history()
        history.log_query(
            indices=index_names,
            query=query,
            top_k=top,
            result_count=len(results),
            elapsed_seconds=elapsed_seconds,
            path_filter=path_filter if path_filter else None,
            exclude_path=exclude_path if exclude_path else None,
        )

    # Format output
    if format == "json":
        output = format_json(results, include_content, content_max_length)
    elif format == "rich":
        output = format_rich(
            results,
            include_content=include_content,
            content_max_length=content_max_length,
        )
    else:
        output = format_simple(results, include_content, content_max_length)

    typer.echo(output)
