# Development Guide

Complete guide for developing, testing, and publishing bm25-index-tool.

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- [mise](https://mise.jdx.dev/) (optional, for version management)

### Development Installation

```bash
# Clone repository
git clone https://github.com/dnvriend/bm25-index-tool.git
cd bm25-index-tool

# Install with mise (recommended)
mise trust
mise install
uv sync
uv tool install .

# Or install directly
uv sync
uv tool install .
```

## Development Workflow

### Available Make Commands

```bash
make install                 # Install dependencies
make format                  # Format code with ruff
make lint                    # Run linting with ruff
make typecheck               # Run type checking with mypy
make test                    # Run tests with pytest
make security-bandit         # Python security linter
make security-pip-audit      # Dependency vulnerability scanner
make security-gitleaks       # Secret/API key detection
make security                # Run all security checks
make check                   # Run all checks (lint, typecheck, test, security)
make pipeline                # Run full pipeline (format, lint, typecheck, test, security, build, install-global)
make build                   # Build package
make run ARGS="..."          # Run bm25-index-tool locally
make clean                   # Remove build artifacts
```

### Code Quality Standards

- **Type hints**: All functions must have type annotations
- **Docstrings**: All public functions require docstrings
- **PEP 8**: Follow via ruff
- **Line length**: 100 characters max
- **Mypy**: Strict mode enabled

## Testing

### Run Tests

```bash
# Run all tests
make test

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_utils.py

# Run with coverage
uv run pytest tests/ --cov=bm25_index_tool

# Run with coverage report
uv run pytest tests/ --cov=bm25_index_tool --cov-report=html
```

### Test Structure

```
tests/
├── __init__.py
├── test_api.py           # Programmatic API tests
├── test_cache.py         # Caching tests
├── test_filters.py       # Path filtering tests
├── test_history.py       # Query history tests
├── test_merge.py         # Merge strategies tests
└── test_utils.py         # Utility function tests
```

## Security

The project includes three lightweight security tools:

### 1. Bandit - Python Security Linter

Detects: SQL injection, hardcoded secrets, unsafe functions

```bash
make security-bandit
```

**Speed**: ~2-3 seconds

### 2. Pip-audit - Dependency Vulnerability Scanner

Detects: Known CVEs in dependencies

```bash
make security-pip-audit
```

**Speed**: ~2-3 seconds

### 3. Gitleaks - Secret Detection

Detects: AWS keys, GitHub tokens, API keys, private keys

```bash
# Install gitleaks (macOS)
brew install gitleaks

# Run scan
make security-gitleaks
```

**Speed**: ~1 second

### Run All Security Checks

```bash
make security  # Runs all three tools (~5-8 seconds)
```

Security checks run automatically in `make check` and `make pipeline`.

## Multi-Level Verbosity Logging

### Implementation

Centralized logging system in `bm25_index_tool/logging_config.py`:

- `setup_logging(verbose_count, log_file, log_format)` - Configure logging
- `get_logger(name)` - Get logger instance for module

### Integration Pattern

```python
from bm25_index_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

@app.command()
def command(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
) -> None:
    setup_logging(verbose)  # First thing in command
    logger.info("Operation started")
    logger.debug("Detailed info")
```

### Logging Levels

| Flag | Level | Output | Use Case |
|------|-------|--------|----------|
| (none) | WARNING | Errors and warnings only | Production, quiet mode |
| `-v` | INFO | + High-level operations | Normal debugging |
| `-vv` | DEBUG | + Detailed info, full tracebacks | Development, troubleshooting |
| `-vvv` | TRACE | + Library internals | Deep debugging |

### File Logging

Enable via environment variable or argument:

```bash
# Via environment
export LOG_FILE=/var/log/bm25-index-tool.log
bm25-index-tool -v

# Or programmatically
setup_logging(verbose_count=1, log_file="/var/log/app.log")
```

**Features**:
- Rotating file handler (10MB max, 5 backups)
- Creates parent directories automatically
- Includes timestamps in log format
- Custom format via `LOG_FORMAT` env var

## Shell Completion

### Supported Shells

| Shell | Version Requirement | Status |
|-------|-------------------|--------|
| **Bash** | ≥ 4.4 | ✅ Supported |
| **Zsh** | Any recent version | ✅ Supported |
| **Fish** | ≥ 3.0 | ✅ Supported |
| **PowerShell** | Any version | ❌ Not Supported |

### Installation Methods

#### Quick Setup (Temporary)

```bash
# Bash
eval "$(bm25-index-tool completion generate bash)"

# Zsh
eval "$(bm25-index-tool completion generate zsh)"

# Fish
bm25-index-tool completion generate fish | source
```

#### Permanent Setup

```bash
# Bash - add to ~/.bashrc
echo 'eval "$(bm25-index-tool completion generate bash)"' >> ~/.bashrc
source ~/.bashrc

# Zsh - add to ~/.zshrc
echo 'eval "$(bm25-index-tool completion generate zsh)"' >> ~/.zshrc
source ~/.zshrc

# Fish - save to completions directory
mkdir -p ~/.config/fish/completions
bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish
```

#### File-based Installation (Better Performance)

```bash
# Bash
bm25-index-tool completion generate bash > ~/.bm25-index-tool-complete.bash
echo 'source ~/.bm25-index-tool-complete.bash' >> ~/.bashrc

# Zsh
bm25-index-tool completion generate zsh > ~/.bm25-index-tool-complete.zsh
echo 'source ~/.bm25-index-tool-complete.zsh' >> ~/.zshrc

# Fish (automatic loading)
mkdir -p ~/.config/fish/completions
bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish
```

## Publishing to PyPI

### Setup PyPI Trusted Publishing

1. **Create PyPI Account** at https://pypi.org/account/register/
   - Enable 2FA (required)
   - Verify email

2. **Configure Trusted Publisher** at https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - **PyPI Project Name**: `bm25-index-tool`
   - **Owner**: `dnvriend`
   - **Repository name**: `bm25-index-tool`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

3. **(Optional) Configure TestPyPI** at https://test.pypi.org/manage/account/publishing/
   - Same settings but use environment: `testpypi`

### Publishing Workflow

The `.github/workflows/publish.yml` workflow:
- Builds on every push
- Publishes to TestPyPI and PyPI on git tags (v*)
- Uses trusted publishing (no secrets needed)

### Create a Release

```bash
# Commit your changes
git add .
git commit -m "Release v0.1.0"
git push

# Create and push tag
git tag v0.1.0
git push origin v0.1.0
```

The workflow automatically builds and publishes to PyPI.

### Build Locally

```bash
# Build package with force rebuild (avoids cache issues)
make build

# Output in dist/
ls dist/
```

## Project Structure

```
bm25-index-tool/
├── bm25_index_tool/           # Main package
│   ├── __init__.py
│   ├── api.py                 # Programmatic API
│   ├── cli.py                 # CLI entry point
│   ├── completion.py          # Shell completion
│   ├── logging_config.py      # Multi-level verbosity logging
│   ├── commands/              # CLI commands
│   │   ├── batch_query.py
│   │   ├── create.py
│   │   ├── delete.py
│   │   ├── history.py
│   │   ├── info.py
│   │   ├── list.py
│   │   ├── query.py
│   │   ├── stats.py
│   │   └── update.py
│   ├── config/                # Configuration models
│   │   └── models.py
│   ├── core/                  # Core functionality
│   │   ├── cache.py           # Search caching
│   │   ├── file_discovery.py # Glob pattern file discovery
│   │   ├── filters.py         # Path filtering
│   │   ├── formatters.py      # Output formatting
│   │   ├── fragments.py       # Fragment extraction
│   │   ├── history.py         # Query history
│   │   ├── indexer.py         # BM25 indexing
│   │   ├── merge_strategies.py # Multi-index merging
│   │   ├── related.py         # Related document finder
│   │   └── searcher.py        # BM25 searching
│   ├── storage/               # Data persistence
│   │   ├── paths.py
│   │   └── registry.py
│   └── telemetry/             # OpenTelemetry (optional)
│       ├── __init__.py
│       ├── config.py
│       ├── decorators.py
│       ├── exporters.py
│       └── service.py
├── tests/                     # Test suite
├── references/                # Documentation
│   ├── architecture.md
│   ├── development.md         # This file
│   ├── shell-completion.md
│   └── telemetry.md
├── pyproject.toml             # Project configuration
├── Makefile                   # Development commands
└── README.md                  # Main documentation
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the full pipeline (`make pipeline`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style Guidelines

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for public functions
- Format code with `ruff`
- Pass all linting and type checks
- Add tests for new features
