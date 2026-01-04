# bm25-index-tool - Project Specification

## Goal

A Python CLI tool for lightning-fast full-text search using BM25 ranking algorithm.

## What is bm25-index-tool?

`bm25-index-tool` provides production-ready full-text search capabilities with the BM25 ranking algorithm. Designed for AI agents and developers who need powerful search with minimal setup.

**Core Capabilities:**
- **Create Indices**: Index files using glob patterns with BM25 algorithm
- **Search**: Query single or multiple indices with BM25 ranking
- **Filter**: Path-based filtering with glob patterns
- **Related Docs**: Find similar documents using TF-IDF
- **Batch Query**: Process multiple queries efficiently with parallel execution
- **History**: Track queries in SQLite database
- **Statistics**: Analyze index health and term frequencies
- **Caching**: LRU cache for search results
- **Merge Strategies**: Combine multi-index results (RRF, union, intersection, weighted)

**Design Philosophy:**
- Agent-friendly with comprehensive help text and inline examples
- JSON output for all commands
- SOLID architecture with clean interfaces
- Type-safe with mypy strict mode
- Fast indexing (~1 second for 2,000 files)

**Storage:**
- Indices: `~/.config/bm25-index-tool/indices/<name>/`
- History: `~/.config/bm25-index-tool/history.db` (SQLite)
- Registry: `~/.config/bm25-index-tool/registry.json`

## Technical Requirements

### Runtime

- Python 3.14+
- Installable globally with mise
- Cross-platform (macOS, Linux, Windows)

### Dependencies

- `typer` - CLI framework (built on Click with type hints)

### Optional Dependencies

- `opentelemetry-api` - OpenTelemetry API
- `opentelemetry-sdk` - OpenTelemetry SDK
- `opentelemetry-exporter-otlp` - OTLP exporter for Grafana Alloy/OTEL Collector

### Development Dependencies

- `ruff` - Linting and formatting
- `mypy` - Type checking
- `pytest` - Testing framework
- `bandit` - Security linting
- `pip-audit` - Dependency vulnerability scanning
- `gitleaks` - Secret detection (requires separate installation)

## Available Commands

### Core Commands

- **create** - Create a new BM25 index from files
- **query** - Search one or more indices
- **list** - List all available indices
- **info** - Show detailed index metadata
- **stats** - Display index statistics and health metrics
- **update** - Rebuild an existing index
- **delete** - Permanently delete an index
- **batch** - Execute multiple queries efficiently
- **history** - Manage query history (show, clear, stats)
- **completion** - Generate shell completion scripts

### Global Options

- `-v, --verbose` - Enable verbose output (count flag: -v, -vv, -vvv)
  - `-v` (count=1): INFO level logging
  - `-vv` (count=2): DEBUG level logging
  - `-vvv` (count=3+): TRACE level (includes library internals)
- `--telemetry` - Enable OpenTelemetry observability (or set OTEL_ENABLED=true)
- `--help` / `-h` - Show help message with examples
- `--version` - Show version

### Command Examples

```bash
# Create index
bm25-index-tool create vault --pattern "~/vault/**/*.md"

# Search with filters
bm25-index-tool query vault "kubernetes" --path-filter "reference/**"

# Multi-index search
bm25-index-tool query "vault,docs" "api design"

# Related documents
bm25-index-tool query vault --related-to "notes/ai.md"

# Batch queries
echo -e "docker\\nkubernetes" | bm25-index-tool batch vault --parallel

# View history
bm25-index-tool history show --limit 50
```

## Project Structure

```
bm25-index-tool/
├── bm25_index_tool/
│   ├── __init__.py
│   ├── cli.py            # Typer CLI entry point (app with subcommands)
│   ├── completion.py     # Shell completion command
│   ├── logging_config.py # Multi-level verbosity logging + file logging
│   ├── telemetry/        # OpenTelemetry observability
│   │   ├── __init__.py
│   │   ├── config.py     # TelemetryConfig dataclass
│   │   ├── service.py    # TelemetryService singleton
│   │   ├── decorators.py # @traced decorator, trace_span context manager
│   │   └── exporters.py  # Exporter factory (console, OTLP) for traces/metrics/logs
│   └── utils.py          # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_utils.py
├── references/           # Observability stack reference configs
│   ├── docker-compose.yml
│   ├── alloy/            # Grafana Alloy config
│   ├── prometheus/       # Prometheus config
│   ├── loki/             # Loki config
│   ├── tempo/            # Tempo config
│   └── grafana/          # Grafana datasource provisioning
├── pyproject.toml        # Project configuration
├── README.md             # User documentation
├── CLAUDE.md             # This file
├── Makefile              # Development commands
├── LICENSE               # MIT License
├── .mise.toml            # mise configuration
├── .gitleaks.toml        # Gitleaks configuration
└── .gitignore
```

## Code Style

- Type hints for all functions
- Docstrings for all public functions
- Follow PEP 8 via ruff
- 100 character line length
- Strict mypy checking

## Development Workflow

```bash
# Install dependencies
make install

# Run linting
make lint

# Format code
make format

# Type check
make typecheck

# Run tests
make test

# Security scanning
make security-bandit       # Python security linting
make security-pip-audit    # Dependency CVE scanning
make security-gitleaks     # Secret detection
make security              # Run all security checks

# Run all checks (includes security)
make check

# Full pipeline (includes security)
make pipeline
```

## Security

The template includes three lightweight security tools:

1. **bandit** - Python code security linting
   - Detects: SQL injection, hardcoded secrets, unsafe functions
   - Speed: ~2-3 seconds

2. **pip-audit** - Dependency vulnerability scanning
   - Detects: Known CVEs in dependencies
   - Speed: ~2-3 seconds

3. **gitleaks** - Secret and API key detection
   - Detects: AWS keys, GitHub tokens, API keys, private keys
   - Speed: ~1 second
   - Requires: `brew install gitleaks` (macOS)

All security checks run automatically in `make check` and `make pipeline`.

## Multi-Level Verbosity Logging

The template includes a centralized logging system with progressive verbosity levels and optional file logging.

### Implementation Pattern

1. **logging_config.py** - Centralized logging configuration
   - `setup_logging(verbose_count, log_file, log_format)` - Configure logging
   - `get_logger(name)` - Get logger instance for module
   - Maps verbosity to Python logging levels (WARNING/INFO/DEBUG)
   - Supports file logging with rotation (10MB, 5 backups)

2. **CLI Integration** - Add to every CLI command
   ```python
   from typing import Annotated
   import typer
   from bm25_index_tool.logging_config import get_logger, setup_logging

   logger = get_logger(__name__)

   @app.command()
   def command(
       verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="...")] = 0,
   ) -> None:
       setup_logging(verbose)  # First thing in command
       logger.info("Operation started")
       logger.debug("Detailed info")
   ```

3. **Logging Levels**
   - **0 (no -v)**: WARNING only - production/quiet mode
   - **1 (-v)**: INFO - high-level operations
   - **2 (-vv)**: DEBUG - detailed debugging
   - **3+ (-vvv)**: TRACE - enable library internals

4. **File Logging**

   Enable file logging via environment variable or argument:
   ```bash
   # Via environment
   export LOG_FILE=/var/log/bm25-index-tool.log
   bm25-index-tool -v

   # Or programmatically
   setup_logging(verbose_count=1, log_file="/var/log/app.log")
   ```

   File logging features:
   - Rotating file handler (10MB max, 5 backups)
   - Creates parent directories automatically
   - Includes timestamps in log format
   - Custom format via `LOG_FORMAT` env var

5. **Environment Variables**

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `LOG_FILE` | (none) | Path to log file (enables file logging) |
   | `LOG_FORMAT` | (default) | Custom log format string |

6. **Best Practices**
   - Always log to stderr (keeps stdout clean for piping)
   - Use structured messages with placeholders: `logger.info("Found %d items", count)`
   - Call `setup_logging()` first in every command
   - Use `get_logger(__name__)` at module level
   - For TRACE level, enable third-party library loggers in `logging_config.py`

7. **Customizing Library Logging**
   Edit `logging_config.py` to add project-specific libraries:
   ```python
   if verbose_count >= 3:
       logging.getLogger("requests").setLevel(logging.DEBUG)
       logging.getLogger("urllib3").setLevel(logging.DEBUG)
   ```

## OpenTelemetry Observability

The template includes OpenTelemetry integration for traces, metrics, and logs. Designed for Grafana stack (Alloy, Tempo, Prometheus, Loki).

### Installation

```bash
# Install with telemetry support
pip install bm25-index-tool[telemetry]
# or with uv
uv sync --extra telemetry
```

### Enabling Telemetry

```bash
# Via CLI flag
bm25-index-tool --telemetry

# Via environment variable
export OTEL_ENABLED=true
bm25-index-tool
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable telemetry |
| `OTEL_SERVICE_NAME` | `bm25-index-tool` | Service name in traces |
| `OTEL_EXPORTER_TYPE` | `console` | `console` or `otlp` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Alloy/Collector endpoint |
| `OTEL_EXPORTER_OTLP_INSECURE` | `true` | Use insecure connection |

### Architecture (SOLID Design)

```
telemetry/
├── config.py      # TelemetryConfig - configuration from env vars (SRP)
├── service.py     # TelemetryService - singleton facade (ISP, DIP)
├── decorators.py  # @traced, trace_span - tracing utilities
└── exporters.py   # Exporter factory - extensible backends (OCP)
```

### Usage Patterns

**1. @traced Decorator**
```python
from bm25_index_tool.telemetry import traced

@traced("process_data")
def process_data(items: list) -> dict:
    return {"count": len(items)}

@traced(attributes={"operation.type": "batch"})
def batch_process():
    pass
```

**2. trace_span Context Manager**
```python
from bm25_index_tool.telemetry import trace_span

with trace_span("database_query", {"db.system": "postgres"}) as span:
    result = db.execute(query)
    if span:
        span.set_attribute("db.rows", len(result))
```

**3. Custom Metrics**
```python
from bm25_index_tool.telemetry import TelemetryService

meter = TelemetryService.get_instance().meter
counter = meter.create_counter("items_processed", description="Items processed")
counter.add(100, {"type": "batch"})

histogram = meter.create_histogram("processing_duration", unit="ms")
histogram.record(150.5, {"operation": "transform"})
```

### Grafana Stack Integration

**Push to Grafana Alloy (recommended)**
```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
bm25-index-tool
```

### Local Observability Stack

A complete local observability stack is included in `references/`:

```bash
# Start the stack
cd references
docker compose up -d

# Configure your CLI
export OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
bm25-index-tool -v
```

Access Grafana at http://localhost:3000 (admin/admin).

**Stack Components:**

| Service | Port | Description |
|---------|------|-------------|
| Grafana Alloy | 4317, 4318 | OTLP receiver |
| Tempo | 3200 | Distributed tracing |
| Prometheus | 9090 | Metrics storage |
| Loki | 3100 | Log aggregation |
| Grafana | 3000 | Visualization |

**Data Flow:**
```
bm25-index-tool --telemetry
      │ OTLP
      ▼
   Alloy ──► Tempo (traces)
      │
      ├────► Prometheus (metrics)
      │
      └────► Loki (logs)
              │
              ▼
           Grafana
```

See `references/README.md` for detailed configuration options.

### Development Mode

For development, use console exporter (default):
```bash
bm25-index-tool --telemetry -vv
```

This outputs traces, metrics, and logs to stderr.

## Shell Completion

The template includes shell completion for bash, zsh, and fish using Typer's completion system.

### Implementation

1. **completion.py** - Separate module for completion command
   - Uses Click's `BashComplete`, `ZshComplete`, `FishComplete` classes (Typer is built on Click)
   - Generates shell-specific completion scripts
   - Includes installation instructions in help text

2. **CLI Integration** - Added as sub-app
   ```python
   from bm25_index_tool.completion import completion_app

   app = typer.Typer(invoke_without_command=True)

   @app.callback(invoke_without_command=True)
   def main(ctx: typer.Context) -> None:
       # Default behavior when no subcommand
       if ctx.invoked_subcommand is None:
           # Main command logic here
           pass

   # Add completion sub-app
   app.add_typer(completion_app, name="completion")
   ```

3. **Usage Pattern** - User-friendly command
   ```bash
   # Generate completion script
   bm25-index-tool completion generate bash
   bm25-index-tool completion generate zsh
   bm25-index-tool completion generate fish

   # Install (eval or save to file)
   eval "$(bm25-index-tool completion generate bash)"
   ```

4. **Supported Shells**
   - **Bash** (≥ 4.4) - Uses bash-completion
   - **Zsh** (any recent) - Uses zsh completion system
   - **Fish** (≥ 3.0) - Uses fish completion system
   - **PowerShell** - Not supported

5. **Installation Methods**
   - **Temporary**: `eval "$(bm25-index-tool completion generate bash)"`
   - **Permanent**: Add eval to ~/.bashrc or ~/.zshrc
   - **File-based** (recommended): Save to dedicated completion file

### Adding More Commands

The CLI uses `typer.Typer()` for extensibility. To add new commands:

1. Create new command module in `bm25_index_tool/`
2. Import and add to CLI app:
   ```python
   from bm25_index_tool.new_command import new_command
   app.command()(new_command)
   # Or for sub-apps:
   app.add_typer(new_app, name="subcommand")
   ```

3. Completion will automatically work for new commands and their options

## Installation Methods

### Global installation with mise

```bash
cd /path/to/bm25-index-tool
mise use -g python@3.14
uv sync
uv tool install .
```

After installation, `bm25-index-tool` command is available globally.

### Local development

```bash
uv sync
uv run bm25-index-tool [args]
```

## Publishing to PyPI

The template includes GitHub Actions workflow for automated PyPI publishing with trusted publishing (no API tokens required).

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

### Install from PyPI

After publishing, users can install with:

```bash
pip install bm25-index-tool
```

### Build Locally

```bash
# Build package with force rebuild (avoids cache issues)
make build

# Output in dist/
ls dist/
```
