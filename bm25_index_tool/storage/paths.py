"""Path management for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from pathlib import Path


def get_config_dir() -> Path:
    """Get the configuration directory for bm25-index-tool.

    Returns:
        Path to ~/.config/bm25-index-tool/
    """
    return Path.home() / ".config" / "bm25-index-tool"


def get_index_dir(name: str) -> Path:
    """Get the directory for a specific index.

    Args:
        name: Name of the index

    Returns:
        Path to ~/.config/bm25-index-tool/indices/{name}/
    """
    return get_config_dir() / "indices" / name


def ensure_config_dir() -> None:
    """Create configuration directory if it doesn't exist.

    Creates:
        - ~/.config/bm25-index-tool/
        - ~/.config/bm25-index-tool/indices/
    """
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    indices_dir = config_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)


def get_registry_path() -> Path:
    """Get the path to the index registry file.

    Returns:
        Path to ~/.config/bm25-index-tool/registry.json
    """
    return get_config_dir() / "registry.json"


def get_config_file_path() -> Path:
    """Get the path to the global configuration file.

    Returns:
        Path to ~/.config/bm25-index-tool/config.toml
    """
    return get_config_dir() / "config.toml"


def get_faiss_index_path(name: str) -> Path:
    """Get the path to the FAISS vector index file.

    Args:
        name: Name of the index

    Returns:
        Path to ~/.config/bm25-index-tool/indices/{name}/vector.faiss
    """
    return get_index_dir(name) / "vector.faiss"


def get_chunks_path(name: str) -> Path:
    """Get the path to the chunks metadata file.

    Args:
        name: Name of the index

    Returns:
        Path to ~/.config/bm25-index-tool/indices/{name}/chunks.json
    """
    return get_index_dir(name) / "chunks.json"
