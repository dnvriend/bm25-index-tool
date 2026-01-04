"""Merge strategies for multi-index search results.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from abc import ABC, abstractmethod
from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


class MergeStrategy(ABC):
    """Abstract base class for merge strategies."""

    @abstractmethod
    def merge(
        self, results_per_index: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        """Merge results from multiple indices.

        Args:
            results_per_index: List of result lists, one per index
            top_k: Number of top results to return

        Returns:
            Merged and ranked list of results
        """
        pass


class RRFMergeStrategy(MergeStrategy):
    """Reciprocal Rank Fusion (RRF) merge strategy.

    RRF formula: score(d) = sum(1 / (k + rank(d, index_i)))
    where k is a constant (default 60) and rank starts from 1.
    """

    def __init__(self, k: int = 60) -> None:
        """Initialize RRF strategy.

        Args:
            k: RRF constant parameter (default: 60)
        """
        self.k = k
        logger.debug("RRF strategy initialized with k=%d", k)

    def merge(
        self, results_per_index: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        """Merge using Reciprocal Rank Fusion.

        Args:
            results_per_index: List of result lists, one per index
            top_k: Number of top results to return

        Returns:
            Merged results sorted by RRF score
        """
        # Build document map and calculate RRF scores
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, dict[str, Any]] = {}

        for results in results_per_index:
            for rank, doc in enumerate(results, start=1):
                doc_path = doc["path"]
                rrf_score = 1.0 / (self.k + rank)

                # Accumulate RRF scores
                if doc_path in doc_scores:
                    doc_scores[doc_path] += rrf_score
                else:
                    doc_scores[doc_path] = rrf_score
                    doc_data[doc_path] = doc

        # Sort by RRF score and take top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        output = []
        for doc_path, score in sorted_docs:
            doc = doc_data[doc_path].copy()
            doc["score"] = float(score)
            output.append(doc)

        logger.debug("RRF merged %d unique documents", len(output))
        return output


class UnionMergeStrategy(MergeStrategy):
    """Union merge strategy - combines all results, sorted by original score."""

    def merge(
        self, results_per_index: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        """Merge using union (combine all, sort by score).

        Args:
            results_per_index: List of result lists, one per index
            top_k: Number of top results to return

        Returns:
            Combined results sorted by original score (descending)
        """
        # Collect all unique documents, keeping highest score
        doc_map: dict[str, dict[str, Any]] = {}

        for results in results_per_index:
            for doc in results:
                doc_path = doc["path"]
                if doc_path not in doc_map:
                    doc_map[doc_path] = doc.copy()
                else:
                    # Keep document with higher score
                    if doc["score"] > doc_map[doc_path]["score"]:
                        doc_map[doc_path] = doc.copy()

        # Sort by score and take top-k
        sorted_docs = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        logger.debug("Union merged %d unique documents", len(sorted_docs))
        return sorted_docs


class IntersectionMergeStrategy(MergeStrategy):
    """Intersection merge strategy - only documents present in ALL indices."""

    def merge(
        self, results_per_index: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        """Merge using intersection (only docs in all indices).

        Args:
            results_per_index: List of result lists, one per index
            top_k: Number of top results to return

        Returns:
            Documents present in all indices, sorted by average score
        """
        if not results_per_index:
            return []

        # Build per-index doc maps
        index_docs: list[dict[str, dict[str, Any]]] = []
        for results in results_per_index:
            doc_map = {doc["path"]: doc for doc in results}
            index_docs.append(doc_map)

        # Find intersection (documents in all indices)
        all_paths = set(index_docs[0].keys())
        for doc_map in index_docs[1:]:
            all_paths &= set(doc_map.keys())

        if not all_paths:
            logger.debug("Intersection merge: no common documents")
            return []

        # Calculate average scores for common documents
        output = []
        for path in all_paths:
            # Average score across all indices
            scores = [doc_map[path]["score"] for doc_map in index_docs]
            avg_score = sum(scores) / len(scores)

            # Use first index's document data
            doc = index_docs[0][path].copy()
            doc["score"] = float(avg_score)
            output.append(doc)

        # Sort by average score and take top-k
        sorted_docs = sorted(output, key=lambda x: x["score"], reverse=True)[:top_k]

        logger.debug("Intersection merged %d common documents", len(sorted_docs))
        return sorted_docs


class WeightedMergeStrategy(MergeStrategy):
    """Weighted merge strategy - weighted score combination per index.

    Scores are normalized per-index (min-max normalization) before weighting.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """Initialize weighted strategy.

        Args:
            weights: Dict mapping index name to weight (default: 1.0 for all)
        """
        self.weights = weights
        logger.debug("Weighted strategy initialized with weights: %s", weights)

    def merge(
        self, results_per_index: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        """Merge using weighted score combination.

        Args:
            results_per_index: List of result lists, one per index
            top_k: Number of top results to return

        Returns:
            Merged results sorted by weighted score
        """
        # Build document map with weighted scores
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, dict[str, Any]] = {}

        for idx, results in enumerate(results_per_index):
            if not results:
                continue

            # Get weight for this index (default 1.0)
            index_name = f"index_{idx}"
            weight = self.weights.get(index_name, 1.0)

            # Normalize scores (min-max normalization)
            scores = [doc["score"] for doc in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for doc in results:
                doc_path = doc["path"]

                # Normalize score to [0, 1]
                if score_range > 0:
                    normalized_score = (doc["score"] - min_score) / score_range
                else:
                    normalized_score = 1.0

                # Apply weight
                weighted_score = normalized_score * weight

                # Accumulate weighted scores
                if doc_path in doc_scores:
                    doc_scores[doc_path] += weighted_score
                else:
                    doc_scores[doc_path] = weighted_score
                    doc_data[doc_path] = doc

        # Sort by weighted score and take top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        output = []
        for doc_path, score in sorted_docs:
            doc = doc_data[doc_path].copy()
            doc["score"] = float(score)
            output.append(doc)

        logger.debug("Weighted merged %d unique documents", len(output))
        return output


def get_merge_strategy(name: str, **params: Any) -> MergeStrategy:
    """Factory function to get merge strategy by name.

    Args:
        name: Strategy name ("rrf", "union", "intersection", "weighted")
        **params: Strategy-specific parameters
            - rrf: k (int, default 60)
            - union: (no parameters)
            - intersection: (no parameters)
            - weighted: weights (dict[str, float])

    Returns:
        Merge strategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    # Create strategy with parameters based on name
    if name == "rrf":
        k = params.get("k", 60)
        return RRFMergeStrategy(k=k)
    elif name == "union":
        return UnionMergeStrategy()
    elif name == "intersection":
        return IntersectionMergeStrategy()
    elif name == "weighted":
        weights = params.get("weights", {})
        return WeightedMergeStrategy(weights=weights)
    else:
        available = "rrf, union, intersection, weighted"
        raise ValueError(f"Unknown merge strategy '{name}'. Available: {available}")
