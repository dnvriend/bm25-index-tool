"""Tests for bm25_index_tool.core.file_discovery module.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import os
from pathlib import Path
from unittest.mock import patch

from bm25_index_tool.core.file_discovery import expand_pattern_to_absolute


class TestExpandPatternToAbsolute:
    """Tests for expand_pattern_to_absolute function."""

    def test_relative_glob_pattern(self) -> None:
        """Relative glob patterns should be prefixed with CWD."""
        with patch.object(Path, "cwd", return_value=Path("/home/user/project")):
            result = expand_pattern_to_absolute("**/*.md")
            assert result == "/home/user/project/**/*.md"

    def test_dot_relative_pattern(self) -> None:
        """Patterns starting with ./ should expand to CWD."""
        with patch.object(Path, "cwd", return_value=Path("/home/user/project")):
            result = expand_pattern_to_absolute("./**/*.md")
            assert result == "/home/user/project/**/*.md"

    def test_double_dot_relative_pattern(self) -> None:
        """Patterns starting with ../ should resolve parent directory."""
        # Use actual CWD parent to avoid platform-specific path resolution issues
        cwd = Path.cwd()
        result = expand_pattern_to_absolute("../**/*.md")
        expected_base = str(cwd.parent.resolve())
        assert result == f"{expected_base}/**/*.md"

    def test_tilde_expansion(self) -> None:
        """Tilde should expand to user home directory."""
        home = os.path.expanduser("~")
        result = expand_pattern_to_absolute("~/docs/**/*.md")
        assert result == f"{home}/docs/**/*.md"

    def test_env_var_expansion(self) -> None:
        """Environment variables should be expanded."""
        with patch.dict(os.environ, {"OBSIDIAN_HOME": "/opt/obsidian"}):
            result = expand_pattern_to_absolute("$OBSIDIAN_HOME/**/*.md")
            assert result == "/opt/obsidian/**/*.md"

    def test_env_var_braces_expansion(self) -> None:
        """Environment variables with braces should be expanded."""
        with patch.dict(os.environ, {"MY_VAULT": "/data/vault"}):
            result = expand_pattern_to_absolute("${MY_VAULT}/**/*.md")
            assert result == "/data/vault/**/*.md"

    def test_absolute_pattern_unchanged(self) -> None:
        """Absolute patterns should remain unchanged."""
        result = expand_pattern_to_absolute("/absolute/path/**/*.md")
        assert result == "/absolute/path/**/*.md"

    def test_home_env_var(self) -> None:
        """$HOME should expand to home directory."""
        home = os.environ.get("HOME", os.path.expanduser("~"))
        result = expand_pattern_to_absolute("$HOME/vault/**/*.md")
        assert result == f"{home}/vault/**/*.md"

    def test_simple_glob_in_cwd(self) -> None:
        """Simple glob without path should use CWD."""
        with patch.object(Path, "cwd", return_value=Path("/workspace")):
            result = expand_pattern_to_absolute("*.py")
            assert result == "/workspace/*.py"

    def test_subdirectory_pattern(self) -> None:
        """Subdirectory patterns should be prefixed with CWD."""
        with patch.object(Path, "cwd", return_value=Path("/home/user")):
            result = expand_pattern_to_absolute("src/**/*.py")
            assert result == "/home/user/src/**/*.py"
