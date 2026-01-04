"""Example demonstrating the BM25Client programmatic API.

This script shows how to use the bm25-index-tool Python API for:
- Creating indices
- Searching with various options
- Multi-index search with RRF fusion
- Related document search
- Batch search
- Cache and history management

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from bm25_index_tool import BM25Client


def main() -> None:
    """Run API examples."""
    # Initialize client with cache and history enabled
    client = BM25Client(enable_cache=True, cache_max_size=100, enable_history=True)

    # Example 1: Create an index
    print("=== Creating Index ===")
    metadata = client.create_index(
        name="example_index",
        path="./docs",
        glob_pattern="**/*.md",
        profile="standard",  # or "code" for code files
        stemmer="english",  # or "" to disable
    )
    print(f"Created index: {metadata['name']}")
    print(f"Files indexed: {metadata['file_count']}")
    print()

    # Example 2: Simple search
    print("=== Simple Search ===")
    results = client.search(
        index="example_index",
        query="machine learning",
        top_k=5,
    )
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['score']:.2f} - {result['path']}")
    print()

    # Example 3: Search with fragments and filters
    print("=== Search with Fragments and Path Filters ===")
    results = client.search(
        index="example_index",
        query="neural networks",
        top_k=5,
        fragments=True,
        context=3,
        path_filter=["tutorials/**", "guides/**"],
        exclude_path=["drafts/**"],
        include_content=True,
        content_max_length=200,
    )
    for result in results:
        print(f"Path: {result['path']}")
        print(f"Score: {result['score']:.2f}")
        if "fragments" in result and result["fragments"]:
            print(f"Fragments: {result['fragments'][0][:100]}...")
        print()

    # Example 4: Multi-index search
    print("=== Multi-Index Search ===")
    # First, create another index
    client.create_index(
        name="example_index_2",
        path="./docs2",
        glob_pattern="**/*.txt",
    )

    results = client.search_multi(
        indices=["example_index", "example_index_2"],
        query="deep learning",
        top_k=10,
        merge_strategy="rrf",
        rrf_k=60,
    )
    print(f"Found {len(results)} results across 2 indices")
    for result in results[:3]:
        print(f"  {result['score']:.2f} - {result['path']}")
    print()

    # Example 5: Related document search
    print("=== Related Document Search ===")
    related = client.search_related(
        index="example_index",
        document_path="intro.md",
        top_k=5,
    )
    print("Documents related to 'intro.md':")
    for result in related:
        print(f"  {result['score']:.2f} - {result['path']}")
    print()

    # Example 6: Batch search
    print("=== Batch Search ===")
    queries = [
        "convolutional neural networks",
        "recurrent neural networks",
        "transformer architecture",
    ]
    batch_results = client.batch_search(
        index="example_index",
        queries=queries,
        top_k=3,
        parallel=True,
        max_workers=4,
    )
    for result in batch_results:
        print(f"Query: '{result['query']}'")
        print(f"  Results: {result['count']} in {result['execution_time']:.3f}s")
    print()

    # Example 7: Index info and statistics
    print("=== Index Info ===")
    info = client.get_info("example_index")
    print(f"Index: {info['name']}")
    print(f"Created: {info['created_at']}")
    print(f"Files: {info['file_count']}")
    print(f"BM25 params: k1={info['bm25_params']['k1']}, b={info['bm25_params']['b']}")
    print()

    print("=== Index Statistics ===")
    stats = client.get_stats("example_index", detailed=False)
    print(f"Storage: {stats['storage_size_formatted']}")

    if True:  # Set to True to compute detailed stats (slower)
        detailed_stats = client.get_stats("example_index", detailed=True)
        if "vocabulary_size" in detailed_stats:
            print(f"Vocabulary: {detailed_stats['vocabulary_size']:,} terms")
            print(f"Sparsity: {detailed_stats['matrix_sparsity']:.2%}")
            print("Top terms:")
            for term_info in detailed_stats["top_terms"][:5]:
                print(f"  {term_info['term']}: {term_info['document_frequency']} docs")
    print()

    # Example 8: List all indices
    print("=== List Indices ===")
    indices = client.list_indices()
    print(f"Total indices: {len(indices)}")
    for idx in indices:
        print(f"  {idx['name']}: {idx['file_count']} files")
    print()

    # Example 9: Cache statistics
    print("=== Cache Statistics ===")
    cache_stats = client.get_cache_stats()
    if cache_stats:
        hit_rate = (
            cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])
            if (cache_stats["hits"] + cache_stats["misses"]) > 0
            else 0
        )
        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache misses: {cache_stats['misses']}")
        print(f"Hit rate: {hit_rate:.1%}")
        print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print()

    # Example 10: Search history
    print("=== Search History ===")
    history = client.get_history(limit=5)
    print(f"Recent searches: {len(history)}")
    for entry in history:
        print(f"  [{entry['timestamp']}] '{entry['query']}'")
        print(f"    Indices: {', '.join(entry['indices'])}")
        print(f"    Results: {entry['result_count']} in {entry['elapsed_seconds']:.3f}s")
    print()

    # Example 11: Update index
    print("=== Update Index ===")
    updated_metadata = client.update_index("example_index")
    print(f"Updated index: {updated_metadata['name']}")
    print(f"Files indexed: {updated_metadata['file_count']}")
    print()

    # Example 12: Cleanup
    print("=== Cleanup ===")
    # Clear cache
    client.clear_cache()
    print("Cache cleared")

    # Delete indices
    client.delete_index("example_index")
    client.delete_index("example_index_2")
    print("Indices deleted")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Example failed: {e}")
        print("This example requires a 'docs' directory with markdown files.")
        print("Create sample files or adjust the paths in the example.")
