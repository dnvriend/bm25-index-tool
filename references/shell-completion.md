# Shell Completion Guide

Complete guide for installing and using shell completion for bm25-index-tool.

## Overview

The CLI provides native shell completion for bash, zsh, and fish shells using Typer's completion system.

## Supported Shells

| Shell | Version Requirement | Status |
|-------|-------------------|--------|
| **Bash** | ≥ 4.4 | ✅ Supported |
| **Zsh** | Any recent version | ✅ Supported |
| **Fish** | ≥ 3.0 | ✅ Supported |
| **PowerShell** | Any version | ❌ Not Supported |

## Quick Installation

### Temporary Setup

Completion active for current session only:

```bash
# Bash
eval "$(bm25-index-tool completion generate bash)"

# Zsh
eval "$(bm25-index-tool completion generate zsh)"

# Fish
bm25-index-tool completion generate fish | source
```

### Permanent Setup

Completion persists across sessions:

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

### File-based Installation (Recommended)

Better performance by avoiding command execution on every shell startup:

```bash
# Bash
bm25-index-tool completion generate bash > ~/.bm25-index-tool-complete.bash
echo 'source ~/.bm25-index-tool-complete.bash' >> ~/.bashrc
source ~/.bashrc

# Zsh
bm25-index-tool completion generate zsh > ~/.bm25-index-tool-complete.zsh
echo 'source ~/.bm25-index-tool-complete.zsh' >> ~/.zshrc
source ~/.zshrc

# Fish (automatic loading from completions directory)
mkdir -p ~/.config/fish/completions
bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish
```

## Usage

Once installed, completion works automatically:

### Command Completion

```bash
bm25-index-tool <TAB>
# Shows: create query batch list info stats update delete history completion
```

### Option Completion

```bash
bm25-index-tool --<TAB>
# Shows: --verbose --telemetry --version --help
```

### Subcommand Completion

```bash
bm25-index-tool query --<TAB>
# Shows: --top --format --rrf-k --merge-strategy --fragments --context
#        --include-content --content-max-length --path-filter --exclude-path
#        --related-to --no-cache --no-history --verbose --help
```

### Shell-specific Completion

```bash
# Bash
bm25-index-tool completion generate <TAB>
# Shows: bash zsh fish

# Zsh (more context-aware)
bm25-index-tool query notes <TAB>
# May suggest recent queries or index names

# Fish (most powerful)
bm25-index-tool query --format <TAB>
# Shows: simple json rich (with descriptions)
```

## Shell-Specific Details

### Bash Completion

**Requirements**:
- Bash ≥ 4.4
- `bash-completion` package (usually pre-installed)

**How It Works**:
- Uses Bash's programmable completion system
- Completion functions registered with `complete -F`
- Supports command, option, and argument completion

**Verify bash-completion**:
```bash
# Check if installed
type _init_completion

# If missing, install:
# macOS
brew install bash-completion@2

# Ubuntu/Debian
sudo apt-get install bash-completion

# Fedora/RHEL
sudo dnf install bash-completion
```

### Zsh Completion

**Requirements**:
- Zsh (any recent version)
- `compinit` loaded (usually in ~/.zshrc)

**How It Works**:
- Uses Zsh's completion system (compsys)
- Completion functions in `_bm25-index-tool` format
- Supports fuzzy matching and menu selection

**Enable compinit** (if not already):
```zsh
# Add to ~/.zshrc
autoload -Uz compinit
compinit
```

**Zsh Completion Features**:
- Fuzzy matching: `bm25<TAB>` matches `bm25-index-tool`
- Menu selection: Navigate with arrow keys
- Description display: Shows help text for options

### Fish Completion

**Requirements**:
- Fish ≥ 3.0

**How It Works**:
- Uses Fish's built-in completion system
- Completion files in `~/.config/fish/completions/`
- Automatic loading on shell startup

**Fish Completion Features**:
- Real-time suggestions as you type
- Rich descriptions for every option
- Command history integration
- Most user-friendly experience

## Troubleshooting

### Completion Not Working

**Bash**:
```bash
# Check if completion is registered
complete -p bm25-index-tool

# Reload bash completion
source ~/.bashrc

# Check bash-completion is installed
type _init_completion
```

**Zsh**:
```bash
# Check if compinit is loaded
which compinit

# Rebuild completion cache
rm -f ~/.zcompdump
compinit

# Reload zsh
source ~/.zshrc
```

**Fish**:
```bash
# Check completion file exists
ls ~/.config/fish/completions/bm25-index-tool.fish

# Reload fish config
source ~/.config/fish/config.fish

# Or restart fish
exec fish
```

### Slow Completion

If completion is slow, use file-based installation instead of `eval`:

```bash
# Instead of this (slow):
eval "$(bm25-index-tool completion generate bash)"

# Do this (fast):
bm25-index-tool completion generate bash > ~/.bm25-complete.bash
source ~/.bm25-complete.bash
```

### Completion Not Updating

After updating the CLI tool:

**Bash/Zsh**:
```bash
# Regenerate completion file
bm25-index-tool completion generate bash > ~/.bm25-complete.bash
source ~/.bm25-complete.bash
```

**Fish**:
```bash
# Regenerate completion
bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish
exec fish
```

## Advanced Configuration

### Custom Completion Directory

**Bash**:
```bash
# Save to custom location
bm25-index-tool completion generate bash > /usr/local/etc/bash_completion.d/bm25-index-tool

# Or system-wide (requires sudo)
sudo bm25-index-tool completion generate bash > /etc/bash_completion.d/bm25-index-tool
```

**Zsh**:
```bash
# Add to custom completion path
mkdir -p ~/.zsh/completions
bm25-index-tool completion generate zsh > ~/.zsh/completions/_bm25-index-tool

# Add to fpath in ~/.zshrc
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

**Fish**:
```bash
# System-wide installation (macOS)
sudo bm25-index-tool completion generate fish > /usr/local/share/fish/vendor_completions.d/bm25-index-tool.fish
```

### Zsh Menu Selection

Enable interactive menu for completion:

```zsh
# Add to ~/.zshrc
zstyle ':completion:*' menu select
zstyle ':completion:*' list-colors ''
```

### Fish Pager Customization

Customize Fish completion pager:

```fish
# Add to ~/.config/fish/config.fish
set -g fish_pager_color_completion normal
set -g fish_pager_color_description 555
set -g fish_pager_color_prefix cyan
```

## Getting Help

### Show Available Commands

```bash
bm25-index-tool completion --help
```

### Generate Completion Script

View the generated completion script:

```bash
# Bash
bm25-index-tool completion generate bash

# Zsh
bm25-index-tool completion generate zsh

# Fish
bm25-index-tool completion generate fish
```

### Test Completion

```bash
# Type this and press TAB to test
bm25-index-tool <TAB>
bm25-index-tool query --<TAB>
```

## Implementation Details

The completion system is implemented in `bm25_index_tool/completion.py`:

- Uses Click's `BashComplete`, `ZshComplete`, `FishComplete` classes
- Typer is built on Click, so Click completion classes work
- Generates shell-specific completion scripts
- Includes installation instructions in help text

### Adding Completion to New Commands

Completion automatically works for new commands added to the CLI:

```python
from bm25_index_tool.completion import completion_app

app = typer.Typer()

# New command automatically gets completion
@app.command()
def new_command(arg: str):
    pass

# Register completion sub-app
app.add_typer(completion_app, name="completion")
```

## Resources

- **Bash Completion**: https://github.com/scop/bash-completion
- **Zsh Completion**: https://zsh.sourceforge.io/Doc/Release/Completion-System.html
- **Fish Completion**: https://fishshell.com/docs/current/completions.html
- **Click Completion**: https://click.palletsprojects.com/en/8.1.x/shell-completion/
