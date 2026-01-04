"""Index registry management for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from typing import Any

from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import ensure_config_dir, get_registry_path

logger = get_logger(__name__)


class IndexRegistry:
    """Manages the index registry (JSON file).

    The registry tracks all created indices and their metadata.
    """

    def __init__(self) -> None:
        """Initialize the index registry."""
        ensure_config_dir()
        self.registry_path = get_registry_path()
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """Create registry file if it doesn't exist."""
        if not self.registry_path.exists():
            logger.debug("Creating new registry file at %s", self.registry_path)
            self._save_registry({})

    def _load_registry(self) -> dict[str, Any]:
        """Load the registry from disk.

        Returns:
            Dictionary mapping index names to metadata
        """
        try:
            with open(self.registry_path) as f:
                data: dict[str, Any] = json.load(f)
                logger.debug("Loaded registry with %d indices", len(data))
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Failed to load registry: %s. Creating new registry.", e)
            return {}

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Save the registry to disk.

        Args:
            data: Registry data to save
        """
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved registry with %d indices", len(data))

    def list_indices(self) -> list[str]:
        """List all registered index names.

        Returns:
            List of index names
        """
        registry = self._load_registry()
        names = sorted(registry.keys())
        logger.debug("Found %d indices", len(names))
        return names

    def add_index(self, name: str, metadata: dict[str, Any]) -> None:
        """Register a new index.

        Args:
            name: Index name
            metadata: Index metadata dictionary
        """
        registry = self._load_registry()
        registry[name] = metadata
        self._save_registry(registry)
        logger.info("Added index '%s' to registry", name)

    def remove_index(self, name: str) -> None:
        """Remove an index from the registry.

        Args:
            name: Index name to remove
        """
        registry = self._load_registry()
        if name in registry:
            del registry[name]
            self._save_registry(registry)
            logger.info("Removed index '%s' from registry", name)
        else:
            logger.warning("Index '%s' not found in registry", name)

    def get_index(self, name: str) -> dict[str, Any] | None:
        """Get metadata for a specific index.

        Args:
            name: Index name

        Returns:
            Index metadata dictionary or None if not found
        """
        registry = self._load_registry()
        metadata = registry.get(name)
        if metadata:
            logger.debug("Found metadata for index '%s'", name)
        else:
            logger.debug("Index '%s' not found", name)
        return metadata

    def index_exists(self, name: str) -> bool:
        """Check if an index is registered.

        Args:
            name: Index name

        Returns:
            True if index exists, False otherwise
        """
        registry = self._load_registry()
        exists = name in registry
        logger.debug("Index '%s' exists: %s", name, exists)
        return exists

    def update_index(self, name: str, metadata: dict[str, Any]) -> None:
        """Update metadata for an existing index.

        Args:
            name: Index name
            metadata: Updated metadata dictionary
        """
        registry = self._load_registry()
        if name in registry:
            registry[name] = metadata
            self._save_registry(registry)
            logger.info("Updated index '%s' metadata", name)
        else:
            logger.warning("Cannot update: index '%s' not found", name)
