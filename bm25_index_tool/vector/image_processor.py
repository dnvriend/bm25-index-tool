"""Image detection and preparation for multimodal embedding.

Handles image format detection using python-magic and prepares images
for AWS Bedrock Nova multimodal embedding.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)

# Supported MIME types for Nova multimodal embedding
SUPPORTED_IMAGE_MIMES: set[str] = {"image/png", "image/jpeg"}

# MIME type to format name mapping (as required by Nova)
MIME_TO_FORMAT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
}


def _get_magic_module() -> Any:
    """Get the python-magic module if available.

    Returns:
        The magic module or None if not installed.
    """
    try:
        import magic

        return magic
    except ImportError:
        return None


class ImageProcessor:
    """Handles image detection and preparation for embedding.

    Uses python-magic for reliable MIME type detection based on file contents
    rather than file extensions. Only PNG and JPEG formats are supported
    as required by AWS Bedrock Nova multimodal embedding.

    Example:
        >>> processor = ImageProcessor()
        >>> if processor.is_image(Path("photo.jpg")):
        ...     image_bytes, format = processor.prepare_for_embedding(Path("photo.jpg"))
        ...     # format is 'png' or 'jpeg'
    """

    def __init__(self) -> None:
        """Initialize the ImageProcessor.

        Logs a warning if python-magic is not available.
        """
        if _get_magic_module() is None:
            logger.warning(
                "python-magic not installed. Image detection will be disabled. "
                "Install with: uv sync --extra vector"
            )

    def is_image(self, path: Path) -> bool:
        """Check if file is a supported image format.

        Uses python-magic to detect MIME type from file contents.
        Returns False if python-magic is not installed or if the file
        is not a supported image format (PNG or JPEG).

        Args:
            path: Path to the file to check.

        Returns:
            True if file is a supported image (PNG or JPEG), False otherwise.
        """
        magic = _get_magic_module()
        if magic is None:
            return False

        if not path.exists():
            logger.debug("File does not exist: %s", path)
            return False

        if not path.is_file():
            logger.debug("Path is not a file: %s", path)
            return False

        try:
            mime_type: str = magic.from_file(str(path), mime=True)
            is_supported = mime_type in SUPPORTED_IMAGE_MIMES
            logger.debug("File %s has MIME type %s, supported: %s", path, mime_type, is_supported)
            return is_supported
        except Exception as e:
            logger.warning("Failed to detect MIME type for %s: %s", path, e)
            return False

    def get_format(self, path: Path) -> str:
        """Get the image format suitable for Nova embedding.

        Args:
            path: Path to the image file.

        Returns:
            Format string ('png' or 'jpeg').

        Raises:
            ValueError: If file is not a supported image format or cannot be read.
        """
        magic = _get_magic_module()
        if magic is None:
            raise ValueError(
                "python-magic is not installed. Cannot determine image format. "
                "Install with: uv sync --extra vector"
            )

        if not path.exists():
            raise ValueError(f"File does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            mime_type: str = magic.from_file(str(path), mime=True)
        except Exception as e:
            raise ValueError(f"Failed to detect MIME type for {path}: {e}") from e

        if mime_type not in SUPPORTED_IMAGE_MIMES:
            raise ValueError(
                f"Unsupported image format: {mime_type}. "
                f"Supported formats: {', '.join(SUPPORTED_IMAGE_MIMES)}"
            )

        return MIME_TO_FORMAT[mime_type]

    def prepare_for_embedding(self, path: Path) -> tuple[bytes, str]:
        """Read image bytes and determine format for embedding.

        Reads the image file and returns the raw bytes along with the
        format string required by Nova multimodal embedding.

        Args:
            path: Path to the image file.

        Returns:
            Tuple of (image_bytes, format) where format is 'png' or 'jpeg'.

        Raises:
            ValueError: If file is not a supported image format or cannot be read.
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read due to permissions.
        """
        # Get format first (validates the file)
        image_format = self.get_format(path)

        # Read the image bytes
        try:
            image_bytes = path.read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {path}") from None
        except PermissionError:
            raise PermissionError(f"Permission denied reading image: {path}") from None
        except Exception as e:
            raise ValueError(f"Failed to read image file {path}: {e}") from e

        logger.debug(
            "Prepared image for embedding: %s, format=%s, size=%d bytes",
            path,
            image_format,
            len(image_bytes),
        )

        return image_bytes, image_format
