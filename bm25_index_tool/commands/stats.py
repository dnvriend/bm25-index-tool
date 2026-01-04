"""Statistics command for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import bm25s  # type: ignore
import numpy as np
import typer

from bm25_index_tool.logging_config import get_logger, setup_logging
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry

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

    # Load index for analysis
    logger.debug("Loading index for detailed analysis")
    index_dir = get_index_dir(name)
    index_path = str(index_dir / "bm25s")
    bm25s_retriever = bm25s.BM25.load(index_path, load_corpus=False, mmap=True)

    # Extract BM25S internals
    vocab = bm25s_retriever.vocab
    scores_matrix = bm25s_retriever.scores

    # Vocabulary statistics
    stats["vocabulary_size"] = len(vocab)

    # Document length statistics
    if hasattr(scores_matrix, "data"):
        # Sparse matrix
        doc_lengths = np.array(scores_matrix.sum(axis=1)).flatten()
    else:
        # Dense matrix
        doc_lengths = scores_matrix.sum(axis=1)

    stats["document_lengths"] = {
        "min": float(doc_lengths.min()),
        "max": float(doc_lengths.max()),
        "mean": float(doc_lengths.mean()),
        "median": float(np.median(doc_lengths)),
    }

    # Term frequency statistics
    if hasattr(scores_matrix, "data"):
        # Sparse matrix - count non-zero entries per term
        term_doc_freq = np.array((scores_matrix > 0).sum(axis=0)).flatten()
    else:
        # Dense matrix
        term_doc_freq = (scores_matrix > 0).sum(axis=0)

    # Top 20 terms by document frequency
    top_indices = np.argsort(term_doc_freq)[::-1][:20]
    top_terms = []
    for idx in top_indices:
        term = vocab[int(idx)]
        freq = int(term_doc_freq[idx])
        top_terms.append({"term": term, "document_frequency": freq})

    stats["top_terms"] = top_terms

    # Matrix sparsity
    if hasattr(scores_matrix, "data"):
        sparsity = 1.0 - (scores_matrix.nnz / (scores_matrix.shape[0] * scores_matrix.shape[1]))
    else:
        sparsity = 1.0 - (np.count_nonzero(scores_matrix) / scores_matrix.size)

    stats["matrix_sparsity"] = float(sparsity)

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
    Detailed mode loads the index for term analysis and vocabulary stats.

    Examples:

    \b
        # Fast stats: Metadata only (instant)
        bm25-index-tool stats vault

    \b
        # Detailed stats: Include vocabulary and term frequencies
        bm25-index-tool stats vault --detailed

    \b
        # JSON output: Machine-readable statistics
        bm25-index-tool stats vault --format json

    \b
        # Detailed JSON: Full statistics with term analysis
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
        - Vocabulary size
        - Document length distribution (min/max/mean/median)
        - Matrix sparsity
        - Top 20 terms by document frequency
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
            lines.append(f"\nVocabulary Size: {stats['vocabulary_size']:,}")
            lines.append("\nDocument Lengths:")
            lines.append(f"  Min: {stats['document_lengths']['min']:.1f}")
            lines.append(f"  Max: {stats['document_lengths']['max']:.1f}")
            lines.append(f"  Mean: {stats['document_lengths']['mean']:.1f}")
            lines.append(f"  Median: {stats['document_lengths']['median']:.1f}")
            lines.append(f"\nMatrix Sparsity: {stats['matrix_sparsity']:.2%}")
            lines.append("\nTop 20 Terms by Document Frequency:")
            for term_info in stats["top_terms"]:
                lines.append(f"  {term_info['term']}: {term_info['document_frequency']} docs")

        output = "\n".join(lines)

    typer.echo(output)
