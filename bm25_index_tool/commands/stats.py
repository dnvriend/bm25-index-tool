"""Statistics command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage

logger = get_logger(__name__)

app = typer.Typer()


def _get_directory_size(path: Path) -> float:
    """Calculate total size of directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total_size: float = 0.0
    if path.exists() and path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                total_size += float(item.stat().st_size)
    return total_size


def _format_size(size_bytes: float) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def _compute_fast_stats(name: str, registry: IndexRegistry) -> dict[str, Any]:
    """Compute fast statistics from metadata only.

    Args:
        name: Index name
        registry: Index registry

    Returns:
        Statistics dictionary
    """
    metadata = registry.get_index(name)
    if not metadata:
        raise ValueError(f"Index '{name}' not found")

    index_dir = get_index_dir(name)
    storage_bytes = _get_directory_size(index_dir)

    stats: dict[str, Any] = {
        "name": name,
        "created_at": metadata["created_at"],
        "file_count": metadata["file_count"],
        "glob_pattern": metadata["glob_pattern"],
        "bm25_params": metadata["bm25_params"],
        "tokenization": metadata["tokenization"],
        "storage_size_bytes": storage_bytes,
        "storage_size_formatted": _format_size(storage_bytes),
    }

    return stats


def _compute_detailed_stats(name: str, registry: IndexRegistry) -> dict[str, Any]:
    """Compute detailed statistics by loading the index.

    Args:
        name: Index name
        registry: Index registry

    Returns:
        Statistics dictionary with detailed metrics
    """
    # Start with fast stats
    stats = _compute_fast_stats(name, registry)

    # Load SQLite storage for detailed analysis
    logger.debug("Loading SQLite database for detailed analysis")

    with SQLiteStorage(name) as storage:
        # Document statistics
        stats["document_count"] = storage.get_document_count()
        stats["chunk_count"] = storage.get_chunk_count()
        stats["has_vector_index"] = storage.has_vector_index()

        # Get all metadata from storage
        storage_metadata = storage.get_all_metadata()
        if storage_metadata:
            stats["storage_metadata"] = storage_metadata

        # Get document type breakdown
        cursor = storage.conn.cursor()
        cursor.execute("""
            SELECT mime_type, COUNT(*) as count
            FROM documents
            GROUP BY mime_type
            ORDER BY count DESC
        """)
        mime_counts = {row["mime_type"]: row["count"] for row in cursor.fetchall()}
        stats["document_types"] = mime_counts

        # Get chunk type breakdown
        cursor.execute("""
            SELECT chunk_type, COUNT(*) as count
            FROM chunks
            GROUP BY chunk_type
            ORDER BY count DESC
        """)
        chunk_type_counts = {row["chunk_type"]: row["count"] for row in cursor.fetchall()}
        stats["chunk_types"] = chunk_type_counts

        # Get total content size
        cursor.execute("SELECT SUM(file_size) as total FROM documents")
        row = cursor.fetchone()
        stats["total_content_bytes"] = row["total"] if row and row["total"] else 0
        stats["total_content_formatted"] = _format_size(stats["total_content_bytes"])

    return stats


@app.command(name="stats")
def stats_command(
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
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Compute detailed statistics (loads index, slower)",
        ),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: simple, json"),
    ] = "simple",
) -> None:
    """Display index statistics and health metrics.

    Fast mode (default) shows metadata-based statistics instantly.
    Detailed mode loads the index for document and chunk analysis.

    Examples:

    \b
        # Fast stats: Metadata only (instant)
        bm25-index-tool stats vault

    \b
        # Detailed stats: Include document and chunk breakdown
        bm25-index-tool stats vault --detailed

    \b
        # JSON output: Machine-readable statistics
        bm25-index-tool stats vault --format json

    \b
        # Detailed JSON: Full statistics with type breakdown
        bm25-index-tool stats vault -d -f json

    \b
    Output (Fast Mode):
        - File count, creation date, glob pattern
        - BM25 parameters (k1, b, method)
        - Tokenization settings (stemmer, stopwords)
        - Storage size (MB/GB)

    \b
    Output (Detailed Mode):
        - All fast mode stats
        - Document count and chunk count
        - Document type breakdown (by MIME type)
        - Chunk type breakdown (text vs image)
        - Vector index status
    """
    setup_logging(verbose)
    logger.info("Computing statistics for index: %s", name)

    registry = IndexRegistry()

    try:
        if detailed:
            stats = _compute_detailed_stats(name, registry)
        else:
            stats = _compute_fast_stats(name, registry)

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Failed to compute statistics")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Format output
    if format == "json":
        output = json.dumps(stats, indent=2)
    else:
        # Simple format
        lines = []
        lines.append(f"\nIndex: {stats['name']}")
        lines.append(f"Created: {stats['created_at']}")
        lines.append(f"Files: {stats['file_count']:,}")
        lines.append(f"Pattern: {stats['glob_pattern']}")
        lines.append(f"Storage: {stats['storage_size_formatted']}")
        lines.append("\nBM25 Parameters:")
        lines.append(f"  k1: {stats['bm25_params']['k1']}")
        lines.append(f"  b: {stats['bm25_params']['b']}")
        lines.append(f"  method: {stats['bm25_params']['method']}")
        lines.append("\nTokenization:")
        stemmer = stats["tokenization"]["stemmer"]
        lines.append(f"  Stemmer: {stemmer if stemmer else 'disabled'}")
        lines.append(f"  Stopwords: {stats['tokenization']['stopwords']}")

        if detailed:
            lines.append(f"\nDocuments: {stats['document_count']:,}")
            lines.append(f"Chunks: {stats['chunk_count']:,}")
            lines.append(f"Vector Index: {'Yes' if stats['has_vector_index'] else 'No'}")
            lines.append(f"Total Content: {stats['total_content_formatted']}")

            if stats.get("document_types"):
                lines.append("\nDocument Types:")
                for mime_type, count in stats["document_types"].items():
                    lines.append(f"  {mime_type}: {count}")

            if stats.get("chunk_types"):
                lines.append("\nChunk Types:")
                for chunk_type, count in stats["chunk_types"].items():
                    lines.append(f"  {chunk_type}: {count}")

        output = "\n".join(lines)

    typer.echo(output)
