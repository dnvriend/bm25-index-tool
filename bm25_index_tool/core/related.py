"""Related document finder for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import re
from collections import Counter
from typing import Any

from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.registry import IndexRegistry
from bm25_index_tool.storage.sqlite_storage import SQLiteStorage

logger = get_logger(__name__)

# Common English stopwords to filter out
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
}


class RelatedDocumentFinder:
    """Find documents related to a given document using TF-IDF based similarity.

    This class extracts the most important terms from a source document
    and uses them as a query to find similar documents in the index.
    """

    def __init__(self) -> None:
        """Initialize the related document finder."""
        self.registry = IndexRegistry()
        self.searcher = BM25Searcher()

    def find_related(
        self,
        index_name: str,
        document_path: str,
        top_k: int = 10,
        num_terms: int = 10,
        extract_fragments_flag: bool = False,
        context_lines: int = 3,
    ) -> list[dict[str, Any]]:
        """Find documents related to the given document.

        Algorithm:
        1. Load the source document from the index
        2. Extract top N most important terms using TF-IDF
        3. Use these terms as a search query
        4. Exclude the source document from results

        Args:
            index_name: Index name
            document_path: Path to source document (relative to index)
            top_k: Number of related documents to return
            num_terms: Number of TF-IDF terms to extract for query
            extract_fragments_flag: If True, extract text fragments with matches
            context_lines: Number of context lines before/after matches

        Returns:
            List of related document dictionaries

        Raises:
            ValueError: If index or document doesn't exist
        """
        # Check if index exists
        if not self.registry.index_exists(index_name):
            logger.error("Index '%s' not found", index_name)
            raise ValueError(f"Index '{index_name}' not found")

        logger.info("Finding documents related to '%s' in index '%s'", document_path, index_name)

        # Find the source document
        with SQLiteStorage(index_name) as storage:
            source_doc = storage.get_document(document_path)

            if source_doc is None:
                logger.error("Document '%s' not found in index '%s'", document_path, index_name)
                raise ValueError(f"Document '{document_path}' not found in index '{index_name}'")

            # Get content
            content = source_doc.content
            if not content:
                logger.warning("Document '%s' has no content", document_path)
                return []

        # Extract top TF-IDF terms
        query_terms = self._extract_top_terms(
            content=content,
            num_terms=num_terms,
        )

        if not query_terms:
            logger.warning("No terms extracted from document '%s'", document_path)
            return []

        # Create query from top terms
        query = " ".join(query_terms)
        logger.debug("Generated query from %d terms: %s", len(query_terms), query)

        # Search using extracted terms (fetch extra to account for exclusion)
        results = self.searcher.search_single(
            index_name=index_name,
            query=query,
            top_k=top_k + 1,  # Get one extra to account for source doc
            extract_fragments_flag=extract_fragments_flag,
            context_lines=context_lines,
        )

        # Exclude source document from results
        filtered_results = [r for r in results if r["path"] != document_path]

        # Return requested number of results
        final_results = filtered_results[:top_k]
        logger.info("Found %d related documents", len(final_results))

        return final_results

    def _extract_top_terms(
        self,
        content: str,
        num_terms: int,
    ) -> list[str]:
        """Extract top N terms from document content using term frequency.

        This is a simplified TF implementation that:
        1. Tokenizes the content (lowercase, alphanumeric only)
        2. Filters out stopwords and short words
        3. Counts term frequencies
        4. Returns most frequent terms

        Args:
            content: Document content
            num_terms: Number of terms to extract

        Returns:
            List of top terms
        """
        # Simple tokenization: lowercase and extract words
        tokens = re.findall(r"\b[a-z][a-z0-9]+\b", content.lower())

        # Filter out stopwords and very short tokens
        filtered_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]

        if not filtered_tokens:
            logger.warning("No tokens extracted from content after filtering")
            return []

        # Count term frequencies
        term_counts = Counter(filtered_tokens)

        # Get top N terms by frequency
        top_terms = [term for term, _count in term_counts.most_common(num_terms)]

        logger.debug(
            "Extracted %d top terms from %d total tokens", len(top_terms), len(filtered_tokens)
        )

        return top_terms
