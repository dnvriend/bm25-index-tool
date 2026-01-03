"""Completion command for bm25-index-tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from enum import Enum

import typer
from click.shell_completion import BashComplete, FishComplete, ShellComplete, ZshComplete

completion_app = typer.Typer(help="Generate shell completion scripts.")


class Shell(str, Enum):
    """Supported shell types for completion."""

    bash = "bash"
    zsh = "zsh"
    fish = "fish"


@completion_app.command(name="generate")
def generate_completion(
    shell: Shell = typer.Argument(..., help="Shell type (bash, zsh, fish)"),
) -> None:
    """Generate shell completion script.

    Install instructions:

    \b
    # Bash (add to ~/.bashrc):
    eval "$(bm25-index-tool completion generate bash)"

    \b
    # Zsh (add to ~/.zshrc):
    eval "$(bm25-index-tool completion generate zsh)"

    \b
    # Fish (add to ~/.config/fish/completions/bm25-index-tool.fish):
    bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish

    \b
    File-based Installation (Recommended for better performance):

    \b
    # Bash
    bm25-index-tool completion generate bash > ~/.bm25-index-tool-complete.bash
    echo 'source ~/.bm25-index-tool-complete.bash' >> ~/.bashrc

    \b
    # Zsh
    bm25-index-tool completion generate zsh > ~/.bm25-index-tool-complete.zsh
    echo 'source ~/.bm25-index-tool-complete.zsh' >> ~/.zshrc

    \b
    # Fish (automatic loading)
    mkdir -p ~/.config/fish/completions
    bm25-index-tool completion generate fish > ~/.config/fish/completions/bm25-index-tool.fish

    \b
    Supported Shells:
      - Bash (≥ 4.4)
      - Zsh (any recent version)
      - Fish (≥ 3.0)

    \b
    Note: PowerShell is not currently supported.
    """
    from bm25_index_tool.cli import app

    completion_classes: dict[str, type[ShellComplete]] = {
        "bash": BashComplete,
        "zsh": ZshComplete,
        "fish": FishComplete,
    }

    # Get the Click command from the Typer app
    click_command = typer.main.get_command(app)

    completion_class = completion_classes.get(shell.value)
    if completion_class:
        completer = completion_class(
            cli=click_command,
            ctx_args={},
            prog_name="bm25-index-tool",
            complete_var="_BM25_INDEX_TOOL_COMPLETE",
        )
        typer.echo(completer.source())
    else:
        raise typer.BadParameter(f"Unsupported shell: {shell}")
