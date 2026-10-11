from __future__ import annotations

from io import BytesIO
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
        from PIL import Image
    except ImportError as error:
        raise ImageDecodeError("image loading requires Pillow") from error

    try:
        with Image.open(source) as image:
            return _decode(image)
    except (OSError, ValueError) as error:
        raise ImageDecodeError(f"could not decode image {source}: {error}") from error


def load_image_bytes(data: bytes, *, suffix: str | None = None) -> np.ndarray:
    """Decode an already-read encoded buffer as an owned RGB uint8 array.

    This is the one-read ingress decode: the caller passes the exact bytes that
    ``discover_media`` hashed, so the discovery content hash and the decoded
    pixels are provably from the same physical read and no second
    ``Image.open(source_path)`` occurs.
    """
    if type(data) is not bytes or not data:
        raise ImageDecodeError("image ingress buffer must be non-empty bytes")
    if suffix is not None and suffix.casefold() not in IMAGE_EXTENSIONS:
        raise ImageDecodeError(f"unsupported image format: {suffix or '<none>'}")
    try:
        from PIL import Image
    except ImportError as error:
        raise ImageDecodeError("image loading requires Pillow") from error

    try:
        with Image.open(BytesIO(data)) as image:
            return _decode(image)
    except (OSError, ValueError) as error:
        raise ImageDecodeError(f"could not decode image buffer: {error}") from error


def _decode(image) -> np.ndarray:
    from PIL import ImageOps

    oriented = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(oriented, dtype=np.uint8, copy=True)


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
