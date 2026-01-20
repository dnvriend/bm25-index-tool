# BM25 Implementation Memory

## Overview

This document provides a technical summary of the BM25 indexing and lookup implementation in bm25-index-tool.

## Indexing Pipeline

### Implementation: `indexer.py:30-127`

**1. Preprocessing**
- Files read sequentially with progress bar
- Each file stored in corpus with metadata structure:
  ```python
  {
    "path": str,      # Full file path
    "name": str,      # File name only
    "content": str    # Full text content
  }
  ```
- UTF-8 encoding with error tolerance (`errors="ignore"`)
- Failed reads logged as warnings, don't halt indexing

**2. Tokenization**
- Uses `bm25s.tokenize()` with configurable options:
  - **Stopwords**: Language-specific (default: "en")
  - **Stemming**: Optional Snowball stemmer (e.g., "english", "porter")
- Tokens processed per document before index building

**3. Index Build**
- BM25 scoring configured with parameters:
  - `k1` (default: 1.5): Controls term frequency saturation
  - `b` (default: 0.75): Controls document length normalization
  - `method`: "lucene" (Okapi BM25 with Lucene optimizations)
- Creates inverted index with term statistics

**4. Storage**
- Location: `~/.config/bm25-index-tool/indices/<name>/`
- Files:
  - `bm25s/`: Binary index files (inverted index + statistics)
  - `metadata.json`: Index metadata (see below)
- Corpus embedded in index for retrieval

## Lookup/Search Pipeline

### Implementation: `searcher.py:29-119`

**1. Index Loading**
- Memory-mapped loading for efficiency (`mmap=True`)
- Loads both index and corpus (`load_corpus=True`)
- Index path: `~/.config/bm25-index-tool/indices/<name>/bm25s/`

**2. Query Tokenization**
- Query processed with **same tokenization** as index:
  - Same stopwords setting
  - Same stemmer (if configured)
- Ensures query/document token compatibility

**3. BM25 Retrieval**
- Scoring formula (Lucene variant):
  ```
  score = IDF(term) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × (dl / avgdl)))
  ```
  - `tf`: term frequency in document
  - `dl`: document length
  - `avgdl`: average document length
  - `IDF`: inverse document frequency
- Returns top-k documents with scores

**4. Post-processing**
- Optional fragment extraction with context lines
- Results include path, name, score, content, and fragments

## Multi-Index Search

### Implementation: `searcher.py:121-184`

**Workflow**:
1. Search each index independently (fetches top 100 or 3×k results)
2. Merge results using configurable strategy:
   - **RRF (default)**: Reciprocal Rank Fusion with k=60
   - **Union**: Combine all unique results
   - **Intersection**: Only common results
   - **Weighted**: Custom score weighting per index
3. Return top-k merged results

**RRF Formula**:
```
RRF_score(doc) = Σ(1 / (k + rank_i))
```
where `rank_i` is document rank in index `i`, and `k` is constant (default: 60)

## Storage Format

### Directory Structure
```
~/.config/bm25-index-tool/
├── indices/
│   └── <name>/
│       ├── bm25s/           # Binary index files
│       └── metadata.json    # Index metadata
├── registry.json            # Index registry
└── history.db              # Query history (SQLite)
```

### Metadata Schema
```json
{
  "name": "index-name",
  "created_at": "2024-01-04T10:30:00",
  "file_count": 1500,
  "glob_pattern": "**/*.md",
  "index_version": "1.0",
  "bm25_params": {
    "k1": 1.5,
    "b": 0.75,
    "method": "lucene"
  },
  "tokenization": {
    "stemmer": "english",
    "stopwords": "en"
  }
}
```

## BM25 Parameters

### k1 (Term Frequency Saturation)
- Range: 0.0 - 10.0
- Default: 1.5
- Higher values = more emphasis on term frequency
- Profiles:
  - **standard**: 1.5 (general text)
  - **code**: 1.2 (code/technical docs)

### b (Document Length Normalization)
- Range: 0.0 - 1.0
- Default: 0.75
- Controls penalization of long documents
- Profiles:
  - **standard**: 0.75 (general text)
  - **code**: 0.5 (less length penalty for code)

### Method
- Default: "lucene"
- Okapi BM25 with Lucene optimizations
- Industry-standard variant

## Performance Characteristics

- **Indexing**: ~1 second for 2,000 files (SSD, typical markdown)
- **Search**: Sub-millisecond for single index queries
- **Multi-Index**: Parallel search with RRF merge overhead
- **Memory**: Memory-mapped index reduces RAM usage

## Dependencies

- **bm25s**: Core BM25 library with fast inverted index
- **PyStemmer**: Snowball stemmer for 15+ languages
- **Storage**: Local filesystem with JSON metadata

## References

- Files: `core/indexer.py`, `core/searcher.py`, `config/models.py`
- BM25s Library: https://github.com/xhluca/bm25s
- Okapi BM25: Robertson & Zaragoza (2009)
