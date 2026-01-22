"""Pytest configuration and fixtures.

Provides test isolation by redirecting all storage to a temporary directory,
preventing tests from affecting production data in ~/.config/bm25-index-tool/.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def test_config_dir():
    """Create a session-scoped temporary config directory for all tests.

    This prevents tests from modifying production indices in ~/.config/.
    """
    with tempfile.TemporaryDirectory(prefix="bm25_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def isolate_config(test_config_dir: Path):
    """Automatically isolate all tests from production config.

    Monkeypatches get_config_dir() to return a temp directory,
    ensuring tests never touch ~/.config/bm25-index-tool/.

    Patches:
    - bm25_index_tool.storage.paths.get_config_dir: Source function, affects
      get_index_dir(), get_registry_path(), ensure_config_dir(), etc.
    - bm25_index_tool.core.history.get_config_dir: Direct import in history module
    """
    with patch(
        "bm25_index_tool.storage.paths.get_config_dir",
        return_value=test_config_dir,
    ):
        with patch(
            "bm25_index_tool.core.history.get_config_dir",
            return_value=test_config_dir,
        ):
            yield
