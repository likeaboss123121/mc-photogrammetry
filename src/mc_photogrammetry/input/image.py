from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import IMAGE_EXTENSIONS, SourceFrame


class ImageDecodeError(ValueError):
    """Raised when a still image cannot be decoded."""


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an owned RGB uint8 array with EXIF orientation applied."""
    source = Path(path)
    if source.suffix.casefold() not in IMAGE_EXTENSIONS:
        raise ImageDecodeError(f"unsupported image format: {source.suffix or '<none>'}")
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ImageDecodeError("image loading requires Pillow") from error

    try:
        with Image.open(source) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(oriented, dtype=np.uint8, copy=True)
    except (OSError, ValueError) as error:
        raise ImageDecodeError(f"could not decode image {source}: {error}") from error


def describe_image(frame: SourceFrame) -> SourceFrame:
    """Fill dimensions for an image manifest entry without changing its identity."""
    pixels = load_image(frame.source_path)
    height, width = pixels.shape[:2]
    return SourceFrame(
        frame_id=frame.frame_id,
        source_path=frame.source_path,
        kind=frame.kind,
        timestamp_seconds=frame.timestamp_seconds,
        content_hash=frame.content_hash,
        width=width,
        height=height,
    )
