# bm25-index-tool

<div align="center">
  <img src=".github/assets/logo_web.png" alt="BM25 Index Tool Logo" width="256"/>
</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)

</div>

Lightning-fast full-text search using BM25 ranking algorithm. Index thousands of files in seconds, query multiple indices simultaneously, filter by path, find related documents.

## Quick Start

```bash
# Install
git clone https://github.com/dnvriend/bm25-index-tool.git
cd bm25-index-tool
uv tool install .

# Create index
bm25-index-tool create notes --pattern "~/Documents/**/*.md"

# Search
bm25-index-tool query notes "kubernetes networking"

# Get help with examples
bm25-index-tool query --help
```

## Features

- **Fast Indexing**: Index 2,000+ files in ~1 second
- **Multi-Index Search**: Query across multiple indices with merge strategies (RRF, union, intersection, weighted)
- **Path Filtering**: Scope searches with glob patterns (`--path-filter "reference/**"`)
- **Related Documents**: Find similar content using TF-IDF
- **Batch Processing**: Process multiple queries with parallel execution
- **Query History**: Track all searches in SQLite database
- **JSON Output**: All commands support `--format json` for AI agents
- **Caching**: LRU cache for repeated queries

## Common Commands

```bash
# Index your Obsidian vault
bm25-index-tool create vault --pattern "~/vault/**/*.md"

# Search with filters
bm25-index-tool query vault "docker" --path-filter "reference/**"

# Multi-index search
bm25-index-tool query "vault,docs" "api design"

# Find related documents
bm25-index-tool query vault --related-to "notes/ai.md" --top 10

# Batch queries
echo -e "docker\nkubernetes\nterraform" | bm25-index-tool batch vault --parallel

# View history
bm25-index-tool history show --limit 50

# Index statistics
bm25-index-tool stats vault --detailed
```

## Available Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new BM25 index from files |
| `query` | Search one or more indices |
| `batch` | Execute multiple queries efficiently |
| `list` | List all available indices |
| `info` | Show detailed index metadata |
| `stats` | Display index statistics |
| `update` | Rebuild an existing index |
| `delete` | Permanently delete an index |
| `history` | Manage query history (show, clear, stats) |
| `completion` | Generate shell completion scripts |

Every command has comprehensive help with examples: `bm25-index-tool <command> --help`

## Query Features

- **Case-insensitive**: `kubernetes` matches `Kubernetes`, `KUBERNETES`
- **Multi-word queries**: `"kubernetes networking"` ranks by term frequency
- **Stopword filtering**: Common words (the, is, and) removed automatically
- **Stemming**: Optional word normalization (`--stemmer english`)
- **No boolean operators**: Use path filtering and multiple queries instead

## Python API

```python
from bm25_index_tool import BM25Client

client = BM25Client()
client.create_index("myindex", "/path/to/docs", "**/*.md")
results = client.search("myindex", "kubernetes", top_k=10)
```

## Documentation

- [Development Guide](./references/development.md) - Setup, testing, security, publishing
- [Architecture](./references/architecture.md) - System design, components, data flow
- [Telemetry](./references/telemetry.md) - OpenTelemetry setup with Grafana stack
- [Shell Completion](./references/shell-completion.md) - Bash, zsh, fish installation

## Development

```bash
# Setup
git clone https://github.com/dnvriend/bm25-index-tool.git
cd bm25-index-tool
make install

# Development workflow
make format      # Format code with ruff
make lint        # Run linting
make typecheck   # Type checking with mypy
make test        # Run tests
make security    # Security scanning (bandit, pip-audit, gitleaks)
make pipeline    # Full CI pipeline
```

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Dennis Vriend** - [@dnvriend](https://github.com/dnvriend)

---

Built with [Claude Code](https://www.anthropic.com/claude/code)
