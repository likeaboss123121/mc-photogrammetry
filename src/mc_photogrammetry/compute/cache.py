"""Bounded frame-pyramid caching shared by CPU and future GPU stages."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FramePyramid:
    """Immutable image levels ordered from full resolution to coarsest."""

    levels: tuple[np.ndarray, ...]

    @property
    def source_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.levels[0].shape)

    @property
    def nbytes(self) -> int:
        return sum(int(level.nbytes) for level in self.levels)


def build_frame_pyramid(
    image: np.ndarray,
    *,
    max_levels: int = 4,
    minimum_size: int = 64,
) -> FramePyramid:
    """Build read-only half-resolution levels for one frame."""
    import cv2

    image = np.asarray(image)
    if image.ndim not in (2, 3) or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image must have positive height/width and two or three dimensions")
    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    if minimum_size < 1:
        raise ValueError("minimum_size must be positive")
    # Own the caller's source before marking the pyramid read-only.  A view
    # created before this function must not be able to mutate cached levels.
    current = np.array(image, copy=True, order="C")
    levels = [current]
    for _ in range(max_levels - 1):
        if min(current.shape[:2]) <= minimum_size:
            break
        height = max(1, current.shape[0] // 2)
        width = max(1, current.shape[1] // 2)
        current = np.array(
            cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA),
            copy=True,
            order="C",
        )
        levels.append(current)
    for level in levels:
        level.setflags(write=False)
    return FramePyramid(tuple(levels))


class FramePyramidCache:
    """LRU cache bounded by total pyramid bytes rather than frame count."""

    def __init__(self, *, max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[str, FramePyramid] = OrderedDict()
        self._bytes = 0

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def get(self, frame_id: str) -> FramePyramid | None:
        pyramid = self._items.get(frame_id)
        if pyramid is not None:
            self._items.move_to_end(frame_id)
        return pyramid

    def put(self, frame_id: str, pyramid: FramePyramid) -> bool:
        """Insert a pyramid, evicting oldest entries; return whether it was kept."""
        if pyramid.nbytes > self.max_bytes:
            old = self._items.pop(frame_id, None)
            if old is not None:
                self._bytes -= old.nbytes
            return False
        old = self._items.pop(frame_id, None)
        if old is not None:
            self._bytes -= old.nbytes
        self._items[frame_id] = pyramid
        self._bytes += pyramid.nbytes
        while self._bytes > self.max_bytes and self._items:
            _key, evicted = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes
        return frame_id in self._items

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)


class BoundedArtifactCache(Generic[T]):
    """Byte-bounded LRU cache for feature, geometry, and accepted GPU artifacts.

    Values expose ``nbytes`` (NumPy arrays and ``PackedRenderBuffers`` do), or a
    caller may provide an explicit size.  The cache owns no GPU resources; it
    only bounds the CPU-side references that feed those resources.
    """

    def __init__(self, *, max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[str, tuple[T, int]] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _size(value: T, size: int | None) -> int:
        if size is not None:
            if size < 0:
                raise ValueError("size must not be negative")
            return int(size)
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None:
            raise ValueError("cached values must expose nbytes or provide size")
        return int(nbytes)

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def get(self, key: str) -> T | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        self._items.move_to_end(key)
        return entry[0]

    def put(self, key: str, value: T, *, size: int | None = None) -> bool:
        """Insert an artifact, evicting oldest entries; return whether retained."""
        item_size = self._size(value, size)
        if item_size > self.max_bytes:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            return False
        old = self._items.pop(key, None)
        if old is not None:
            self._bytes -= old[1]
        self._items[key] = (value, item_size)
        self._bytes += item_size
        while self._bytes > self.max_bytes and self._items:
            _key, (_value, evicted_size) = self._items.popitem(last=False)
            self._bytes -= evicted_size
        return key in self._items

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)
