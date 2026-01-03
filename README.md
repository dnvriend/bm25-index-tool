# bm25-index-tool

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![AI Generated](https://img.shields.io/badge/AI-Generated-blueviolet.svg)](https://www.anthropic.com/claude)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-5A67D8.svg)](https://www.anthropic.com/claude/code)

A Python CLI tool

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Multi-Level Verbosity Logging](#multi-level-verbosity-logging)
- [OpenTelemetry Observability](#opentelemetry-observability)
- [Shell Completion](#shell-completion)
- [Development](#development)
- [Testing](#testing)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## About

`bm25-index-tool` is a Python CLI tool built with modern tooling and best practices.

## Features

- ✅ Type-safe with mypy strict mode
- ✅ Linted with ruff
- ✅ Tested with pytest
- 📊 Multi-level verbosity logging (-v/-vv/-vvv)
- 📡 OpenTelemetry observability (traces, metrics, logs)
- 🐚 Shell completion for bash, zsh, and fish
- 🔒 Security scanning with bandit, pip-audit, and gitleaks
- ✅ Modern Python tooling (uv, mise, typer)

## Installation

### Prerequisites

- Python 3.14 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/dnvriend/bm25-index-tool.git
cd bm25-index-tool

# Install globally with uv
uv tool install .
```

### Install with mise (recommended for development)

```bash
cd bm25-index-tool
mise trust
mise install
uv sync
uv tool install .
```

### Verify installation

```bash
bm25-index-tool --version
```

## Usage

### Basic Usage

```bash
# Show help
bm25-index-tool --help

# Run the tool
bm25-index-tool

# Run with verbose output
bm25-index-tool -v      # INFO level
bm25-index-tool -vv     # DEBUG level
bm25-index-tool -vvv    # TRACE level (includes library internals)
```

## Multi-Level Verbosity Logging

The CLI supports progressive verbosity levels for debugging and troubleshooting. All logs output to stderr, keeping stdout clean for data piping.

### Logging Levels

| Flag | Level | Output | Use Case |
|------|-------|--------|----------|
| (none) | WARNING | Errors and warnings only | Production, quiet mode |
| `-v` | INFO | + High-level operations | Normal debugging |
| `-vv` | DEBUG | + Detailed info, full tracebacks | Development, troubleshooting |
| `-vvv` | TRACE | + Library internals | Deep debugging |

### Examples

```bash
# Quiet mode - only errors and warnings
bm25-index-tool

# INFO - see operations and progress
bm25-index-tool -v
# Output:
# [INFO] bm25-index-tool started
# [INFO] bm25-index-tool completed

# DEBUG - see detailed information
bm25-index-tool -vv
# Output:
# [INFO] bm25-index-tool started
# [DEBUG] Running with verbose level: 2
# [INFO] bm25-index-tool completed

# TRACE - see library internals (configure in logging_config.py)
bm25-index-tool -vvv
```

### Customizing Library Logging

To enable DEBUG logging for third-party libraries at TRACE level (-vvv), edit `bm25_index_tool/logging_config.py`:

```python
# Configure dependent library loggers at TRACE level (-vvv)
if verbose_count >= 3:
    logging.getLogger("requests").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    # Add your project-specific library loggers here
```

## OpenTelemetry Observability

The CLI supports OpenTelemetry for distributed tracing, metrics, and logs. Designed for integration with Grafana stack (Alloy, Tempo, Prometheus).

### Installation

```bash
# Install with telemetry support
pip install bm25-index-tool[telemetry]
# or with uv
uv add "bm25-index-tool[telemetry]"
```

### Quick Start

```bash
# Enable telemetry via CLI flag
bm25-index-tool --telemetry

# Or via environment variable
export OTEL_ENABLED=true
bm25-index-tool
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable telemetry |
| `OTEL_SERVICE_NAME` | `bm25-index-tool` | Service name in traces |
| `OTEL_EXPORTER_TYPE` | `console` | `console` or `otlp` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |

### Production Setup (Grafana Alloy)

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4317
bm25-index-tool
```

### Development Mode

```bash
# Console output with verbose logging
bm25-index-tool --telemetry -vv
```

See [CLAUDE.md](CLAUDE.md) for detailed usage patterns and Grafana Alloy configuration.

## Shell Completion

The CLI provides native shell completion for bash, zsh, and fish shells.

### Supported Shells

| Shell | Version Requirement | Status |
|-------|-------------------|--------|
| **Bash** | ≥ 4.4 | ✅ Supported |
| **Zsh** | Any recent version | ✅ Supported |
| **Fish** | ≥ 3.0 | ✅ Supported |
| **PowerShell** | Any version | ❌ Not Supported |

### Installation

#### Quick Setup (Temporary)

```bash
# Bash - active for current session only
eval "$(bm25-index-tool completion generate bash)"

# Zsh - active for current session only
eval "$(bm25-index-tool completion generate zsh)"

# Fish - active for current session only
bm25-index-tool completion generate fish | source
```

#### Permanent Setup (Recommended)

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

For better shell startup performance, generate completion scripts to files:

```bash
# Bash
bm25-index-tool completion generate bash > ~/.bm25-index-tool-complete.bash
echo 'source ~/.bm25-index-tool-complete.bash' >> ~/.bashrc

# Zsh
bm25-index-tool completion generate zsh > ~/.bm25-index-tool-complete.zsh
echo 'source ~/.bm25-index-tool-complete.zsh' >> ~/.zshrc

# Fish (automatic loading from completions directory)
mkdir -p ~/.config/fish/completions
bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish
```

### Usage

Once installed, completion works automatically:

```bash
# Tab completion for commands
bm25-index-tool <TAB>
# Shows: completion

# Tab completion for options
bm25-index-tool --<TAB>
# Shows: --verbose --version --help

# Tab completion for shell types
bm25-index-tool completion generate <TAB>
# Shows: bash zsh fish
```

### Getting Help

```bash
# View completion installation instructions
bm25-index-tool completion --help
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/dnvriend/bm25-index-tool.git
cd bm25-index-tool

# Install dependencies
make install

# Show available commands
make help
```

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

### Project Structure

```
bm25-index-tool/
├── bm25_index_tool/    # Main package
│   ├── __init__.py
│   ├── cli.py          # CLI entry point
│   └── utils.py        # Utility functions
├── tests/              # Test suite
│   ├── __init__.py
│   └── test_utils.py
├── pyproject.toml      # Project configuration
├── Makefile            # Development commands
├── README.md           # This file
├── LICENSE             # MIT License
└── CLAUDE.md           # Development documentation
```

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run tests with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_utils.py

# Run with coverage
uv run pytest tests/ --cov=bm25_index_tool
```

## Security

The project includes lightweight security tools providing 80%+ coverage with fast scan times:

### Security Tools

| Tool | Purpose | Speed | Coverage |
|------|---------|-------|----------|
| **bandit** | Python code security linting | ⚡⚡ Fast | SQL injection, hardcoded secrets, unsafe functions |
| **pip-audit** | Dependency vulnerability scanning | ⚡⚡ Fast | Known CVEs in dependencies |
| **gitleaks** | Secret and API key detection | ⚡⚡⚡ Very Fast | Secrets in code and git history |

### Running Security Scans

```bash
# Run all security checks (~5-8 seconds)
make security

# Or run individually
make security-bandit       # Python security linting
make security-pip-audit    # Dependency CVE scanning
make security-gitleaks     # Secret detection
```

### Prerequisites

gitleaks must be installed separately:

```bash
# macOS
brew install gitleaks

# Linux
# See: https://github.com/gitleaks/gitleaks#installation
```

Security checks run automatically in `make check` and `make pipeline`.

### What's Protected

- ✅ AWS credentials (AKIA*, ASIA*, etc.)
- ✅ GitHub tokens (ghp_*, gho_*, etc.)
- ✅ API keys and secrets
- ✅ Private keys
- ✅ Slack tokens
- ✅ 100+ other secret types

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the full pipeline (`make pipeline`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for public functions
- Format code with `ruff`
- Pass all linting and type checks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Dennis Vriend**

- GitHub: [@dnvriend](https://github.com/dnvriend)

## Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) for CLI framework
- Developed with [uv](https://github.com/astral-sh/uv) for fast Python tooling

---

**Generated with AI**

This project was generated using [Claude Code](https://www.anthropic.com/claude/code), an AI-powered development tool by [Anthropic](https://www.anthropic.com/). Claude Code assisted in creating the project structure, implementation, tests, documentation, and development tooling.

Made with ❤️ using Python 3.14
