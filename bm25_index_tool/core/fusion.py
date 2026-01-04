"""RRF fusion utilities for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


def fuse_results(results: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Fuse multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF Formula: score(doc) = Σ(1 / (k + rank_i))

    Args:
        results: List of result lists from different indices
        k: RRF k parameter (default: 60)

    Returns:
        Fused and sorted list of results
    """
    logger.debug("Fusing %d result lists with RRF (k=%d)", len(results), k)

    # Build document score aggregation
    doc_scores: dict[str, float] = {}
    doc_data: dict[str, dict[str, Any]] = {}

    for result_list in results:
        for rank, doc in enumerate(result_list, start=1):
            doc_path = doc["path"]
            rrf_score = 1.0 / (k + rank)

            if doc_path in doc_scores:
                doc_scores[doc_path] += rrf_score
            else:
                doc_scores[doc_path] = rrf_score
                doc_data[doc_path] = doc

    # Sort by fused score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Build output
    output = []
    for doc_path, score in sorted_docs:
        doc = doc_data[doc_path].copy()
        doc["score"] = score
        output.append(doc)

    logger.debug("Fused %d unique documents", len(output))
    return output
