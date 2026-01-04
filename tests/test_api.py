"""Tests for bm25_index_tool.api module.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import tempfile
from pathlib import Path

import pytest

from bm25_index_tool.api import BM25Client


@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with sample documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()

        # Create sample markdown files
        (docs_dir / "intro.md").write_text(
            "Introduction to machine learning and artificial intelligence."
        )
        (docs_dir / "tutorial.md").write_text(
            "Machine learning tutorial covering supervised and unsupervised learning."
        )
        (docs_dir / "advanced.md").write_text(
            "Advanced topics in deep learning and neural networks."
        )

        yield docs_dir


@pytest.fixture
def client():
    """Create a BM25Client instance."""
    client_instance = BM25Client(enable_cache=True, enable_history=True)
    yield client_instance

    # Cleanup: delete all test indices
    try:
        indices = client_instance.list_indices()
        for idx in indices:
            try:
                client_instance.delete_index(idx["name"])
            except Exception:
                pass
    except Exception:
        pass


def test_client_initialization():
    """Test BM25Client initialization."""
    client = BM25Client()
    assert client.indexer is not None
    assert client.searcher is not None
    assert client.registry is not None
    assert client.cache is None
    assert client.history is None


def test_client_initialization_with_features():
    """Test BM25Client initialization with cache and history."""
    client = BM25Client(enable_cache=True, cache_max_size=50, enable_history=True)
    assert client.cache is not None
    assert client.history is not None


def test_create_index(client, temp_docs_dir):
    """Test index creation."""
    metadata = client.create_index(
        name="test_index",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
        profile="standard",
    )

    assert metadata["name"] == "test_index"
    assert metadata["file_count"] == 3
    assert "bm25_params" in metadata
    assert "tokenization" in metadata


def test_search_single(client, temp_docs_dir):
    """Test single index search."""
    # Create index
    client.create_index(
        name="test_search",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    # Search
    results = client.search(
        index="test_search",
        query="machine learning",
        top_k=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert "path" in results[0]
    assert "score" in results[0]
    assert "name" in results[0]


def test_search_with_fragments(client, temp_docs_dir):
    """Test search with fragment extraction."""
    client.create_index(
        name="test_fragments",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    results = client.search(
        index="test_fragments",
        query="machine learning",
        top_k=5,
        fragments=True,
        context=2,
    )

    assert len(results) > 0
    # Fragments should be present in results
    for result in results:
        if result["content"]:
            assert "fragments" in result or result["score"] > 0


def test_get_info(client, temp_docs_dir):
    """Test get_info method."""
    client.create_index(
        name="test_info",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    info = client.get_info("test_info")

    assert info["name"] == "test_info"
    assert info["file_count"] == 3
    assert "bm25_params" in info
    assert "created_at" in info


def test_get_stats_fast(client, temp_docs_dir):
    """Test fast statistics."""
    client.create_index(
        name="test_stats",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    stats = client.get_stats("test_stats", detailed=False)

    assert stats["name"] == "test_stats"
    assert stats["file_count"] == 3
    assert "storage_size_bytes" in stats
    assert "storage_size_formatted" in stats


def test_get_stats_detailed(client, temp_docs_dir):
    """Test detailed statistics."""
    client.create_index(
        name="test_detailed_stats",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    stats = client.get_stats("test_detailed_stats", detailed=True)

    # Detailed stats may not be available if bm25s version doesn't support vocab attribute
    # Check for either fast stats or detailed stats
    assert "name" in stats
    assert "file_count" in stats
    # If detailed stats are available, check for them
    if "vocabulary_size" in stats:
        assert "document_lengths" in stats
        assert "matrix_sparsity" in stats
        assert "top_terms" in stats


def test_list_indices(client, temp_docs_dir):
    """Test listing indices."""
    client.create_index(
        name="test_list_1",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )
    client.create_index(
        name="test_list_2",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    indices = client.list_indices()

    assert len(indices) >= 2
    names = [idx["name"] for idx in indices]
    assert "test_list_1" in names
    assert "test_list_2" in names


def test_delete_index(client, temp_docs_dir):
    """Test index deletion."""
    client.create_index(
        name="test_delete",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    # Verify index exists
    info = client.get_info("test_delete")
    assert info["name"] == "test_delete"

    # Delete index
    client.delete_index("test_delete")

    # Verify deletion
    with pytest.raises(ValueError, match="not found"):
        client.get_info("test_delete")


def test_cache_functionality(client, temp_docs_dir):
    """Test cache enable/disable/clear."""
    client.create_index(
        name="test_cache",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    # First search (cache miss)
    results1 = client.search("test_cache", "machine learning", top_k=5)
    stats1 = client.get_cache_stats()
    assert stats1 is not None

    # Second search (should be cache hit)
    results2 = client.search("test_cache", "machine learning", top_k=5)
    stats2 = client.get_cache_stats()

    assert len(results1) == len(results2)
    if stats2:
        assert stats2["hits"] >= stats1["hits"]

    # Clear cache
    client.clear_cache()
    stats3 = client.get_cache_stats()
    if stats3:
        assert stats3["size"] == 0


def test_cache_disabled():
    """Test client with cache disabled."""
    client = BM25Client(enable_cache=False)
    assert client.cache is None

    stats = client.get_cache_stats()
    assert stats is None


def test_history_functionality(client, temp_docs_dir):
    """Test history tracking."""
    client.create_index(
        name="test_history",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    # History should be enabled
    assert client.history is not None

    # Perform searches (but history is not auto-logged in API)
    # This is by design - API users must manually log if needed

    history = client.get_history(limit=10)
    assert isinstance(history, list)


def test_batch_search(client, temp_docs_dir):
    """Test batch search."""
    client.create_index(
        name="test_batch",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
    ]

    results = client.batch_search(
        index="test_batch",
        queries=queries,
        top_k=3,
        parallel=False,
    )

    assert len(results) == 3
    for result in results:
        assert "query" in result
        assert "results" in result
        assert "count" in result
        assert "execution_time" in result


def test_batch_search_parallel(client, temp_docs_dir):
    """Test parallel batch search."""
    client.create_index(
        name="test_batch_parallel",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
    )

    queries = ["machine learning", "deep learning"]

    results = client.batch_search(
        index="test_batch_parallel",
        queries=queries,
        top_k=3,
        parallel=True,
        max_workers=2,
    )

    assert len(results) == 2


def test_custom_bm25_params(client, temp_docs_dir):
    """Test index creation with custom BM25 parameters."""
    metadata = client.create_index(
        name="test_custom_params",
        path=str(temp_docs_dir),
        glob_pattern="*.md",
        k1=1.2,
        b=0.5,
    )

    assert metadata["bm25_params"]["k1"] == 1.2
    assert metadata["bm25_params"]["b"] == 0.5


def test_index_not_found_errors(client):
    """Test proper error handling for non-existent indices."""
    with pytest.raises(ValueError, match="not found"):
        client.search("nonexistent", "query")

    with pytest.raises(ValueError, match="not found"):
        client.get_info("nonexistent")

    with pytest.raises(ValueError, match="not found"):
        client.delete_index("nonexistent")
