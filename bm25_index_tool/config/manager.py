"""Configuration manager for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import tomli

from bm25_index_tool.config.models import GlobalConfig
from bm25_index_tool.logging_config import get_logger
from bm25_index_tool.storage.paths import ensure_config_dir, get_config_file_path

logger = get_logger(__name__)


class ConfigManager:
    """Manages global configuration."""

    def __init__(self) -> None:
        """Initialize the config manager."""
        ensure_config_dir()
        self.config_path = get_config_file_path()

    def load_config(self) -> GlobalConfig:
        """Load global configuration from disk.

        Returns:
            GlobalConfig instance (defaults if file doesn't exist)
        """
        if not self.config_path.exists():
            logger.debug("Config file not found, using defaults")
            return GlobalConfig()

        try:
            with open(self.config_path, "rb") as f:
                data = tomli.load(f)
                config = GlobalConfig(**data)
                logger.debug("Loaded configuration from %s", self.config_path)
                return config
        except (tomli.TOMLDecodeError, ValueError) as e:
            logger.warning("Failed to load config: %s. Using defaults.", e)
            return GlobalConfig()

    def save_config(self, config: GlobalConfig) -> None:
        """Save global configuration to disk.

        Args:
            config: GlobalConfig instance to save
        """
        try:
            import tomli_w

            with open(self.config_path, "wb") as f:
                tomli_w.dump(config.model_dump(), f)
            logger.info("Saved configuration to %s", self.config_path)
        except ImportError:
            logger.warning("tomli_w not installed, cannot save config")
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def get_default_config(self) -> GlobalConfig:
        """Get default configuration.

        Returns:
            GlobalConfig with default values
        """
        return GlobalConfig()
