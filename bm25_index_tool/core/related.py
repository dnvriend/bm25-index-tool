"""Related document finder for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from collections import Counter
from typing import Any

import bm25s  # type: ignore
import Stemmer  # type: ignore

from bm25_index_tool.config.models import IndexMetadata
from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import get_index_dir
from bm25_index_tool.storage.registry import IndexRegistry

logger = get_logger(__name__)


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
        # Load index metadata
        metadata_dict = self.registry.get_index(index_name)
        if not metadata_dict:
            logger.error("Index '%s' not found", index_name)
            raise ValueError(f"Index '{index_name}' not found")

        metadata = IndexMetadata(**metadata_dict)
        logger.info("Finding documents related to '%s' in index '%s'", document_path, index_name)

        # Load index
        index_dir = get_index_dir(index_name)
        index_path = str(index_dir / "bm25s")

        try:
            retriever = bm25s.BM25.load(index_path, load_corpus=True, mmap=True)
        except Exception as e:
            logger.error("Failed to load index '%s': %s", index_name, e)
            raise ValueError(f"Failed to load index '{index_name}': {e}")

        # Find the source document in the corpus
        source_doc = None
        corpus = retriever.corpus

        for doc in corpus:
            if doc["path"] == document_path:
                source_doc = doc
                break

        if source_doc is None:
            logger.error("Document '%s' not found in index '%s'", document_path, index_name)
            raise ValueError(f"Document '{document_path}' not found in index '{index_name}'")

        # Extract content
        content = source_doc.get("content", "")
        if not content:
            logger.warning("Document '%s' has no content", document_path)
            return []

        # Extract top TF-IDF terms
        query_terms = self._extract_top_terms(
            content=content,
            num_terms=num_terms,
            metadata=metadata,
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
        metadata: IndexMetadata,
    ) -> list[str]:
        """Extract top N terms from document content using TF-IDF.

        This is a simplified TF-IDF implementation that:
        1. Tokenizes the content
        2. Applies stemming if configured
        3. Counts term frequencies
        4. Returns most frequent terms

        Args:
            content: Document content
            num_terms: Number of terms to extract
            metadata: Index metadata with tokenization config

        Returns:
            List of top terms (original form, not stemmed)
        """
        # Prepare stemmer if enabled
        stemmer = None
        if metadata.tokenization.stemmer_enabled:
            try:
                stemmer = Stemmer.Stemmer(metadata.tokenization.stemmer)
            except Exception as e:
                logger.warning("Failed to initialize stemmer: %s", e)

        # Tokenize content
        tokenized = bm25s.tokenize(
            [content],
            stopwords=metadata.tokenization.stopwords,
            stemmer=stemmer,
        )

        if not tokenized or len(tokenized) == 0 or len(tokenized[0]) == 0:
            logger.warning("No tokens extracted from content")
            return []

        # Get tokens (already processed with stopwords and stemming)
        tokens = tokenized[0]

        # Count term frequencies
        term_counts = Counter(tokens)

        # Get top N terms by frequency
        top_terms = [term for term, _count in term_counts.most_common(num_terms)]

        logger.debug("Extracted %d top terms from %d total tokens", len(top_terms), len(tokens))

        return top_terms
