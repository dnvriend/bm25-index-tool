# BM25 Index Tool - Architecture Documentation

**Version**: 0.1.0
**Date**: 2026-01-04
**Author**: Dennis Vriend

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Data Storage](#data-storage)
6. [Command Architecture](#command-architecture)
7. [Design Patterns](#design-patterns)
8. [Data Flow](#data-flow)
9. [Configuration](#configuration)
10. [Dependencies](#dependencies)

---

## Overview

The BM25 Index Tool is a Python CLI application built with Typer that provides fast text search using the BM25 ranking algorithm. It's designed to be both human-friendly and AI-agent-friendly with comprehensive JSON output support, caching, history tracking, and a programmatic API.

### Key Characteristics

- **Language**: Python 3.14+
- **CLI Framework**: Typer (built on Click)
- **Search Algorithm**: BM25 (via bm25s library)
- **Storage**: File-based (JSON + SQLite + NumPy arrays)
- **Architecture Style**: Modular, SOLID principles, Strategy pattern
- **Deployment**: Single executable, installed via uv

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  CLI Commands (Typer)           │  Programmatic API (BM25Client)│
│  - create, query, batch         │  - Python class interface    │
│  - list, info, stats            │  - Wraps core modules        │
│  - update, delete, history      │  - Library usage            │
└─────────────────┬───────────────┴──────────────┬────────────────┘
                  │                              │
┌─────────────────▼──────────────────────────────▼────────────────┐
│                      APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Core Modules                                                    │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │  Indexer   │ │ Searcher │ │ Filters  │ │ RelatedDocs     │ │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────────┘ │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │   Cache    │ │ History  │ │  Fusion  │ │ MergeStrategies │ │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────────┘ │
│  ┌────────────┐ ┌──────────┐                                   │
│  │ Formatters │ │ FileDisc │                                   │
│  └────────────┘ └──────────┘                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      STORAGE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Storage Modules                                                 │
│  ┌────────────┐ ┌──────────┐ ┌──────────────────────────────┐  │
│  │  Registry  │ │  Paths   │ │  BM25S Native Format         │  │
│  │ (JSON)     │ │ (Helper) │ │  (NumPy + JSONL)             │  │
│  └────────────┘ └──────────┘ └──────────────────────────────┘  │
│  ┌────────────┐                                                 │
│  │  History   │                                                 │
│  │ (SQLite)   │                                                 │
│  └────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     FILE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────┤
│  ~/.config/bm25-index-tool/                                      │
│  ├── registry.json          (Index metadata registry)           │
│  ├── history.db             (Query history SQLite)              │
│  └── indices/                                                    │
│      └── <index-name>/                                           │
│          ├── metadata.json   (Per-index metadata)               │
│          └── bm25s/          (BM25S index files)                │
│              ├── corpus.jsonl                                    │
│              ├── vocab.index.json                                │
│              ├── *.npy       (NumPy sparse matrices)            │
│              └── params.index.json                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Project Structure

```
bm25-index-tool/
├── bm25_index_tool/              # Main package
│   ├── __init__.py               # Package exports (BM25Client)
│   ├── cli.py                    # CLI entry point + command registration
│   ├── api.py                    # Programmatic API (BM25Client class)
│   ├── logging_config.py         # Centralized logging setup
│   ├── utils.py                  # Utility functions
│   ├── completion.py             # Shell completion support
│   │
│   ├── commands/                 # CLI command modules (Typer)
│   │   ├── __init__.py
│   │   ├── create.py             # Index creation command
│   │   ├── query.py              # Search command (with filters, cache, history)
│   │   ├── batch_query.py        # Batch query command (parallel support)
│   │   ├── list.py               # List indices command
│   │   ├── info.py               # Show index info command
│   │   ├── stats.py              # Show index statistics command
│   │   ├── update.py             # Update/re-index command
│   │   ├── delete.py             # Delete index command
│   │   └── history.py            # Query history management command
│   │
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── indexer.py            # BM25Indexer - index creation/update
│   │   ├── searcher.py           # BM25Searcher - search operations
│   │   ├── file_discovery.py    # File discovery with glob patterns
│   │   ├── formatters.py         # Output formatters (simple, json, rich)
│   │   ├── filters.py            # PathFilter - glob-based filtering
│   │   ├── fragments.py          # Text fragment extraction
│   │   ├── fusion.py             # RRF fusion for multi-index
│   │   ├── merge_strategies.py  # Strategy pattern for merge algorithms
│   │   ├── related.py            # RelatedDocumentFinder - TF-IDF similarity
│   │   ├── cache.py              # SearchCache - LRU caching
│   │   └── history.py            # SearchHistory - SQLite history
│   │
│   ├── storage/                  # Storage abstraction layer
│   │   ├── __init__.py
│   │   ├── registry.py           # IndexRegistry - index metadata management
│   │   └── paths.py              # Path utilities for storage locations
│   │
│   ├── config/                   # Configuration management
│   │   └── __init__.py
│   │
│   └── telemetry/                # OpenTelemetry observability (optional)
│       ├── __init__.py
│       ├── config.py             # TelemetryConfig
│       ├── service.py            # TelemetryService singleton
│       ├── decorators.py         # @traced decorator
│       └── exporters.py          # Exporter factory
│
├── tests/                        # Test suite
│   ├── test_api.py               # API tests (17 tests)
│   └── test_utils.py             # Utility tests
│
├── examples/                     # Usage examples
│   └── api_example.py            # Python API examples
│
├── references/                   # Documentation
│   ├── architecture.md           # This file
│   ├── 10-feature-implementation-summary.md
│   └── search-effectiveness-report.md
│
├── pyproject.toml                # Project metadata + dependencies
├── README.md                     # User documentation
├── LICENSE                       # MIT License
├── Makefile                      # Development commands
└── .mise.toml                    # mise configuration
```

### Runtime Storage Structure

```
~/.config/bm25-index-tool/
├── registry.json                 # Master index registry
│   └── {
│       "index-name": {
│           "name": "...",
│           "created_at": "...",
│           "file_count": 1234,
│           "glob_pattern": "...",
│           "bm25_params": {...},
│           "tokenization": {...}
│       }
│   }
│
├── history.db                    # SQLite query history
│   └── search_history table:
│       - id, timestamp, indices, query, top_k,
│         result_count, elapsed_seconds, filters
│
└── indices/                      # Index storage
    └── <index-name>/
        ├── metadata.json         # Per-index metadata (duplicate of registry)
        └── bm25s/                # BM25S library format
            ├── corpus.jsonl      # Full document content
            │   └── {"path": "...", "name": "...", "content": "..."}
            ├── vocab.index.json  # Vocabulary (unique terms)
            ├── data.csc.index.npy         # BM25 scores (sparse matrix data)
            ├── indices.csc.index.npy      # Sparse matrix column indices
            ├── indptr.csc.index.npy       # Sparse matrix row pointers
            ├── corpus.mmindex.json        # Memory-map index
            └── params.index.json          # BM25 parameters
```

---

## Core Components

### 1. Indexer (`core/indexer.py`)

**Purpose**: Creates and updates BM25 indices from document collections.

**Key Class**: `BM25Indexer`

**Methods**:
- `create_index(name, files, k1, b, method, stemmer, stopwords)` → `IndexMetadata`
- `update_index(name, files)` → `IndexMetadata`

**Process**:
1. Read file contents
2. Tokenize with optional stemming and stopword removal
3. Build BM25 index using bm25s library
4. Save index to disk (NumPy sparse format + JSONL corpus)
5. Update registry with metadata

**Dependencies**:
- `bm25s` - BM25 implementation
- `Stemmer` - Snowball stemmer (optional)
- `storage.registry.IndexRegistry`

---

### 2. Searcher (`core/searcher.py`)

**Purpose**: Executes search queries against BM25 indices.

**Key Class**: `BM25Searcher`

**Methods**:
- `search_single(index_name, query, top_k, extract_fragments_flag, context_lines)` → `list[dict]`
- `search_multi(index_names, query, top_k, rrf_k, merge_strategy, ...)` → `list[dict]`

**Process**:
1. Load index from disk (memory-mapped)
2. Tokenize query with same settings as index
3. Execute BM25 search
4. Extract text fragments (optional)
5. Format results with scores and metadata

**Single-Index Search**:
```
Query → Tokenize → BM25.retrieve() → Extract Fragments → Results
```

**Multi-Index Search**:
```
Query → [Tokenize → BM25.retrieve()] × N indices
      → Merge Strategy (RRF/Union/Intersection/Weighted)
      → Deduplicate → Sort → Results
```

**Dependencies**:
- `bm25s` - BM25 index loading
- `core.merge_strategies` - Multi-index merging
- `core.fragments` - Fragment extraction

---

### 3. File Discovery (`core/file_discovery.py`)

**Purpose**: Discovers files matching glob patterns with path expansion.

**Key Function**: `discover_files(pattern, respect_gitignore=True)` → `list[Path]`

**Features**:
- Tilde expansion (`~`, `~user`)
- Environment variable expansion (`$VAR`, `${VAR}`)
- Recursive glob patterns (`**/*.md`)
- .gitignore respect
- Natural sorting (handles numeric filenames)

**Process**:
```
Pattern → Expand (~, $VAR) → Parse Base Dir + Glob
       → Discover Files → Filter .gitignore → Sort → Return Paths
```

---

### 4. Formatters (`core/formatters.py`)

**Purpose**: Format search results for output.

**Functions**:
- `format_simple(results, include_content, content_max_length)` → `str`
- `format_json(results, include_content, content_max_length)` → `str`
- `format_rich(results, include_content, content_max_length)` → `str`

**Features**:
- Simple: Plain text with scores
- JSON: Machine-readable output
- Rich: Formatted with fragments and context
- Content display with line numbers
- Content truncation support

---

### 5. Filters (`core/filters.py`)

**Purpose**: Post-search filtering by path patterns.

**Key Class**: `PathFilter`

**Methods**:
- `__init__(include_patterns, exclude_patterns)`
- `matches(path)` → `bool`
- `filter_results(results)` → `list[dict]`

**Logic**:
- Include patterns: OR logic (match any)
- Exclude patterns: AND NOT logic (match none)
- Uses `fnmatch` for glob matching

**Example**:
```python
filter = PathFilter(
    include_patterns=["reference/**", "docs/**"],
    exclude_patterns=["**/*.draft.md", "**/*.tmp.md"]
)
filtered = filter.filter_results(results)
```

---

### 6. Cache (`core/cache.py`)

**Purpose**: LRU cache for search results.

**Key Class**: `SearchCache`

**Methods**:
- `get(indices, query, top_k, path_filter, exclude_path)` → `list[dict] | None`
- `set(indices, query, top_k, results, path_filter, exclude_path)`
- `clear()`
- `stats()` → `dict`

**Features**:
- LRU eviction (OrderedDict)
- SHA256 cache keys
- Thread-safe (threading.Lock)
- Default max size: 100 entries
- In-memory only (not persisted)

**Cache Key Generation**:
```python
key_data = {
    "indices": sorted(indices),  # Normalized
    "query": query,
    "top_k": top_k,
    "path_filter": sorted(path_filter or []),
    "exclude_path": sorted(exclude_path or [])
}
cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

---

### 7. History (`core/history.py`)

**Purpose**: Query history tracking with SQLite.

**Key Class**: `SearchHistory`

**Methods**:
- `log_query(indices, query, top_k, result_count, elapsed_seconds, filters)`
- `get_recent(limit)` → `list[dict]`
- `search(pattern)` → `list[dict]`
- `clear()`
- `count()` → `int`

**Schema**:
```sql
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    indices TEXT NOT NULL,
    query TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    result_count INTEGER NOT NULL,
    elapsed_seconds REAL NOT NULL,
    path_filter TEXT,
    exclude_path TEXT
);
CREATE INDEX idx_timestamp ON search_history(timestamp);
CREATE INDEX idx_query ON search_history(query);
```

**Database Location**: `~/.config/bm25-index-tool/history.db`

---

### 8. Merge Strategies (`core/merge_strategies.py`)

**Purpose**: Strategy pattern for multi-index result merging.

**Abstract Base Class**: `MergeStrategy`
- `merge(results_per_index: dict[str, list[dict]], top_k: int)` → `list[dict]`

**Implementations**:

1. **RRFMergeStrategy** (Reciprocal Rank Fusion)
   ```python
   score(doc) = sum(1 / (k + rank_in_index)) for all indices
   ```

2. **UnionMergeStrategy**
   - Combines all results
   - Deduplicates (keeps highest score)
   - Sorts by original score

3. **IntersectionMergeStrategy**
   - Only documents in ALL indices
   - Averages scores across indices
   - Sorts by average score

4. **WeightedMergeStrategy**
   - Weighted score combination
   - Min-max normalization per index
   - User-specified weights per index

**Factory Function**:
```python
strategy = get_merge_strategy("weighted", weights={"idx1": 2.0, "idx2": 1.0})
merged_results = strategy.merge(results_per_index, top_k=10)
```

---

### 9. Related Documents (`core/related.py`)

**Purpose**: Find documents similar to a given document.

**Key Class**: `RelatedDocumentFinder`

**Method**:
- `find_related(index_name, document_path, top_k, extract_fragments_flag, context_lines)` → `list[dict]`

**Algorithm**:
1. Load source document from index corpus
2. Tokenize document (with index settings)
3. Calculate term frequencies
4. Extract top N terms (TF-IDF approximation)
5. Use terms as search query
6. Execute search and exclude source document
7. Return similar documents

---

### 10. Registry (`storage/registry.py`)

**Purpose**: Manage index metadata registry.

**Key Class**: `IndexRegistry`

**Methods**:
- `add_index(metadata: IndexMetadata)`
- `get_index(name: str)` → `dict | None`
- `list_indices()` → `list[dict]`
- `remove_index(name: str)`
- `index_exists(name: str)` → `bool`

**Storage**: `~/.config/bm25-index-tool/registry.json`

**Data Model**:
```python
@dataclass
class IndexMetadata:
    name: str
    created_at: str
    file_count: int
    glob_pattern: str
    index_version: str
    bm25_params: BM25Params
    tokenization: TokenizationConfig
```

---

### 11. BM25Client (`api.py`)

**Purpose**: Programmatic API facade for library usage.

**Key Class**: `BM25Client`

**Constructor**:
```python
client = BM25Client(
    enable_cache: bool = False,
    enable_history: bool = False,
    cache_max_size: int = 100
)
```

**Methods**:
- Index Management: `create_index()`, `update_index()`, `delete_index()`, `get_info()`, `get_stats()`, `list_indices()`
- Search: `search()`, `search_multi()`, `search_related()`, `batch_search()`
- Cache: `enable_cache()`, `disable_cache()`, `clear_cache()`, `get_cache_stats()`
- History: `enable_history()`, `disable_history()`, `get_history()`, `clear_history()`

**Design**: Facade pattern wrapping core modules for clean API.

---

## Data Storage

### Storage Locations

All data stored in: `~/.config/bm25-index-tool/`

**Why this location?**
- Standard XDG configuration directory on Unix
- User-specific (no root privileges needed)
- Easy to backup/delete
- Follows platform conventions

### Index Storage Format

**BM25S Native Format** (chosen for performance):
- Memory-mapped files for fast loading
- Sparse matrix format (CSC) for efficient storage
- JSONL corpus for document content
- NumPy arrays for numerical data

**Storage Efficiency**:
- 2,176 markdown files → 31 MB index
- ~14 KB per document (includes content + BM25 data)
- Sparse matrices save space vs. dense

**Trade-offs**:
- Fast: Memory-mapped loading (~50ms for 2K docs)
- Efficient: Sparse format for large vocabularies
- Portable: NumPy format is cross-platform
- Not human-readable: Binary format

---

## Command Architecture

### CLI Entry Point (`cli.py`)

**Framework**: Typer (built on Click)

**Structure**:
```python
app = typer.Typer(
    invoke_without_command=True,
    help="BM25 Index Tool - Fast text search with BM25"
)

@app.callback(invoke_without_command=True)
def main(ctx, verbose, version, telemetry):
    # Global setup: logging, telemetry
    if ctx.invoked_subcommand is None:
        # Show help by default
        typer.echo(ctx.get_help())

# Register commands
app.command(name="create")(create_command)
app.command(name="query")(query_command)
app.command(name="batch")(batch_command)
# ... etc
```

**Command Registration Pattern**:
Each command is a separate module in `commands/` with a Typer function.

### Command Flow

**Generic Command Flow**:
```
User Input → Typer Parsing → setup_logging(verbose)
          → Command Logic → Core Modules
          → Format Output → Display to User
```

**Query Command Flow (Detailed)**:
```
User: bm25-index-tool query my-index "search term" -n 10 --format json

1. Typer parses arguments
2. setup_logging(verbose)
3. Check cache (if enabled)
4. If not cached:
   - Load index from disk
   - Execute BM25 search
   - Extract fragments (if requested)
5. Apply path filters (if provided)
6. Store in cache (if enabled)
7. Log to history (if enabled)
8. Format output (json/simple/rich)
9. Display to stdout
```

---

## Design Patterns

### 1. Strategy Pattern

**Used in**: `core/merge_strategies.py`

**Purpose**: Select merge algorithm at runtime.

```python
# Strategy interface
class MergeStrategy(ABC):
    @abstractmethod
    def merge(self, results_per_index, top_k) -> list[dict]:
        pass

# Concrete strategies
class RRFMergeStrategy(MergeStrategy): ...
class UnionMergeStrategy(MergeStrategy): ...
class IntersectionMergeStrategy(MergeStrategy): ...
class WeightedMergeStrategy(MergeStrategy): ...

# Factory
strategy = get_merge_strategy(name="rrf", k=60)
results = strategy.merge(results_per_index, top_k=10)
```

---

### 2. Facade Pattern

**Used in**: `api.py` (BM25Client)

**Purpose**: Simplify interface for library users.

```python
# Complex subsystem (CLI + Core modules)
from bm25_index_tool.core.indexer import BM25Indexer
from bm25_index_tool.core.searcher import BM25Searcher
from bm25_index_tool.core.cache import SearchCache
# ... many more

# Simple facade
class BM25Client:
    def search(self, index, query, top_k=10, **kwargs):
        # Wraps: Searcher + Cache + History + Filters + Formatters
        return results
```

---

### 3. Singleton Pattern

**Used in**: `core/cache.py`, `core/history.py`, `telemetry/service.py`

**Purpose**: Single global instance for cache and history.

```python
# Global instances in query.py
_cache: SearchCache | None = None
_history: SearchHistory | None = None

def get_cache() -> SearchCache:
    global _cache
    if _cache is None:
        _cache = SearchCache()
    return _cache
```

---

### 4. Factory Pattern

**Used in**: `core/merge_strategies.py`

**Purpose**: Create strategy instances from string names.

```python
def get_merge_strategy(name: str, **params) -> MergeStrategy:
    if name == "rrf":
        return RRFMergeStrategy(k=params.get("k", 60))
    elif name == "union":
        return UnionMergeStrategy()
    # ... etc
```

---

### 5. Command Pattern

**Used in**: `commands/` directory structure

**Purpose**: Encapsulate CLI commands as objects.

Each command is a separate module with a single responsibility.

---

### 6. Template Method Pattern

**Used in**: Formatters

**Purpose**: Define algorithm structure, vary implementation.

```python
# Base structure (implicit)
def format_X(results, include_content, content_max_length):
    # 1. Check if empty
    # 2. Iterate results
    # 3. Format each result (varies by formatter)
    # 4. Optionally add content (common)
    # 5. Return formatted string
```

---

## Data Flow

### Index Creation Flow

```
User: bm25-index-tool create my-index -p "docs/**/*.md"

1. CLI: Parse arguments
2. FileDiscovery: discover_files("docs/**/*.md")
   ├─ Expand tilde/env vars
   ├─ Parse base dir + glob pattern
   ├─ Discover files (glob/rglob)
   ├─ Filter .gitignore
   └─ Sort naturally
   → [file1.md, file2.md, ...]

3. Indexer: create_index(name="my-index", files=[...])
   ├─ Read file contents
   ├─ Tokenize (with stemmer/stopwords)
   ├─ Build BM25 index (bm25s.BM25())
   ├─ Save index to disk
   │   ├─ corpus.jsonl (documents)
   │   ├─ vocab.index.json (vocabulary)
   │   ├─ *.npy (sparse matrices)
   │   └─ params.index.json (BM25 params)
   └─ Return metadata

4. Registry: add_index(metadata)
   └─ Save to registry.json

5. Display: "Index 'my-index' created! Files: 100, Time: 2.5s"
```

---

### Search Flow (Single Index)

```
User: bm25-index-tool query my-index "kubernetes" -n 10 --format json

1. CLI: Parse arguments
2. Query Command:
   ├─ Check cache (hash query params → cache key)
   │   ├─ Hit: Return cached results
   │   └─ Miss: Continue
   │
   ├─ Searcher: search_single(index_name, query, top_k)
   │   ├─ Load index (mmap from disk)
   │   ├─ Tokenize query (same settings as index)
   │   ├─ BM25.retrieve(query_tokens, k=10)
   │   └─ Format results with scores
   │   → [{"path": "...", "score": 0.95, ...}, ...]
   │
   ├─ Apply path filters (if provided)
   │   └─ PathFilter.filter_results(results)
   │
   ├─ Store in cache (if enabled)
   ├─ Log to history (if enabled)
   │
   └─ Format output (format_json)
       └─ Display JSON to stdout
```

---

### Search Flow (Multi-Index)

```
User: bm25-index-tool query idx1,idx2 "docker" --merge-strategy weighted

1. CLI: Parse arguments
2. Query Command:
   ├─ Parse index names: ["idx1", "idx2"]
   ├─ Parse merge strategy: "weighted"
   │
   ├─ Searcher: search_multi(index_names, query, merge_strategy)
   │   │
   │   ├─ For each index:
   │   │   ├─ Load index
   │   │   ├─ Tokenize query
   │   │   ├─ BM25.retrieve(query_tokens)
   │   │   └─ Store: results_per_index["idx1"] = [...]
   │   │
   │   ├─ Get merge strategy instance
   │   │   strategy = get_merge_strategy("weighted", weights={...})
   │   │
   │   ├─ Merge results
   │   │   strategy.merge(results_per_index, top_k=10)
   │   │   ├─ Normalize scores per index (min-max)
   │   │   ├─ Apply weights
   │   │   ├─ Deduplicate (by path)
   │   │   └─ Sort by weighted score
   │   │
   │   └─ Return merged results
   │
   └─ Format and display
```

---

### Batch Query Flow

```
User: bm25-index-tool batch my-index -i queries.txt --parallel --max-workers 4

1. CLI: Parse arguments
2. Batch Command:
   ├─ Read queries from file (one per line)
   │   queries = ["query1", "query2", "query3", ...]
   │
   ├─ If --parallel:
   │   ├─ Create ThreadPoolExecutor(max_workers=4)
   │   ├─ Submit tasks: [executor.submit(search, q) for q in queries]
   │   ├─ Collect results as they complete
   │   └─ Shutdown executor
   │ Else:
   │   └─ Execute sequentially
   │
   ├─ For each query result:
   │   ├─ Format as JSONL (one JSON per line)
   │   └─ Print to stdout
   │
   └─ Example output:
       {"query": "kubernetes", "results": [...], "count": 10, "time": 0.05}
       {"query": "docker", "results": [...], "count": 8, "time": 0.03}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry observability |
| `OTEL_SERVICE_NAME` | `bm25-index-tool` | Service name in traces |
| `OTEL_EXPORTER_TYPE` | `console` | Exporter type: `console` or `otlp` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |
| `LOG_FILE` | (none) | Path to log file (enables file logging) |
| `LOG_FORMAT` | (default) | Custom log format string |

### BM25 Parameters

**Configurable via CLI** (`create` command):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.5 | Term frequency saturation parameter |
| `b` | 0.75 | Length normalization parameter |
| `method` | `lucene` | BM25 variant: `lucene`, `atire`, `bm25l`, `bm25+` |
| `stemmer` | (disabled) | Stemmer: `english`, `german`, etc. |
| `stopwords` | `en` | Stopwords language: `en`, `es`, etc. |

**Profiles** (shortcuts):
- `--profile standard`: k1=1.5, b=0.75, method=lucene
- `--profile strict`: k1=1.2, b=0.8, method=lucene
- `--profile lenient`: k1=1.8, b=0.6, method=lucene

### Cache Configuration

**In-Memory Only** (not configurable via env vars, hardcoded):
- Max size: 100 entries
- Eviction: LRU (Least Recently Used)
- Thread-safe: Yes
- Persistence: No (cleared on restart)

**Future**: Could add env vars for cache size, TTL, persistence.

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `typer` | 0.21.0 | CLI framework |
| `bm25s` | 0.2.14 | BM25 search implementation |
| `numpy` | 2.3.5 | Numerical arrays (BM25S dependency) |
| `pystemmer` | 3.0.0 | Snowball stemmer |
| `pathspec` | 0.12.1 | .gitignore pattern matching |
| `rich` | 14.2.0 | Terminal formatting |

### Optional Dependencies

| Package | Purpose | Extra |
|---------|---------|-------|
| `opentelemetry-api` | Telemetry API | `telemetry` |
| `opentelemetry-sdk` | Telemetry SDK | `telemetry` |
| `opentelemetry-exporter-otlp` | OTLP exporter | `telemetry` |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `ruff` | Linting + formatting |
| `mypy` | Type checking |
| `bandit` | Security linting |
| `pip-audit` | Dependency vulnerability scanning |

---

## Performance Characteristics

### Index Creation

**Time Complexity**: O(N × M) where N = documents, M = avg terms per doc

**Measured Performance**:
- 2,176 markdown files → 1.05 seconds
- ~2,070 files/second

**Bottleneck**: File I/O (reading documents from disk)

**Storage**: ~14 KB per document (includes content + BM25 data)

---

### Search

**Time Complexity**: O(log N + K) where N = documents, K = top_k

**Measured Performance**:
- 2,176 documents → ~50ms (cold start, mmap load)
- 2,176 documents → ~5ms (warm, cached)

**Bottleneck**: Index loading (first query only)

---

### Multi-Index Search

**Time Complexity**: O(I × (log N + K) + K log K) where I = indices

**Process**:
- Search each index: I × O(log N + K)
- Merge results: O(K log K)

---

### Cache Performance

**Hit Rate**: Depends on query patterns (typically 20-40% for repeated searches)

**Memory Usage**: ~10 KB per cached query × 100 entries = ~1 MB

**Overhead**: <1ms for hash computation

---

## Security Considerations

### Input Validation

- **Glob patterns**: Path expansion (tilde, env vars) but no shell execution
- **Index names**: Validated (alphanumeric + hyphen only)
- **File paths**: Canonicalized to prevent directory traversal
- **Query strings**: No special validation (BM25 tokenizes safely)

### Storage Security

- **Permissions**: User-only (chmod 600 for sensitive files)
- **Location**: User home directory (no system-wide access)
- **Secrets**: No secrets stored (use env vars for sensitive config)

### Dependency Security

- **Scanning**: `bandit` (code), `pip-audit` (dependencies), `gitleaks` (secrets)
- **Updates**: Regular updates via `uv sync`

---

## Future Enhancements

### Potential Improvements

1. **Incremental Indexing**: Track file changes, update only modified docs
2. **Distributed Caching**: Redis backend for cache sharing
3. **Compression**: gzip/zstd for index storage
4. **Async API**: Async/await support for I/O-bound operations
5. **Query Suggestions**: Autocomplete based on history
6. **Export Formats**: CSV, Parquet for batch results
7. **Index Versioning**: Schema evolution support
8. **Monitoring**: Prometheus metrics exporter

---

## Appendix: Key Algorithms

### BM25 Scoring

```
BM25(d, q) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))

where:
- d = document
- q = query
- qi = query term i
- f(qi, d) = term frequency of qi in d
- |d| = length of document d
- avgdl = average document length in corpus
- k1 = term frequency saturation (default 1.5)
- b = length normalization (default 0.75)
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
- N = total documents
- n(qi) = documents containing qi
```

---

### Reciprocal Rank Fusion (RRF)

```
RRF(d) = Σ 1 / (k + rank_i(d))

where:
- d = document
- rank_i(d) = rank of d in result list i
- k = constant (default 60)
- Σ = sum over all result lists
```

**Example**:
```
Index 1 ranks: doc_a=1, doc_b=2, doc_c=3
Index 2 ranks: doc_b=1, doc_a=3, doc_d=2

RRF scores (k=60):
doc_a: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
doc_b: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325 (winner)
doc_c: 1/(60+3) = 0.0159
doc_d: 1/(60+2) = 0.0161
```

---

## Glossary

- **BM25**: Best Match 25, probabilistic ranking function
- **Corpus**: Collection of documents
- **IDF**: Inverse Document Frequency
- **LRU**: Least Recently Used (cache eviction policy)
- **RRF**: Reciprocal Rank Fusion
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Stemming**: Reducing words to root form (e.g., "running" → "run")
- **Stopwords**: Common words filtered out (e.g., "the", "a", "is")
- **CSC**: Compressed Sparse Column (matrix format)
- **JSONL**: JSON Lines (one JSON object per line)
- **mmap**: Memory-mapped file (virtual memory technique)

---

**End of Architecture Documentation**
