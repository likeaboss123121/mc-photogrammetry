"""Bounded frame-pyramid caching shared by CPU and future GPU stages."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


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
    current = np.ascontiguousarray(image)
    levels = [current]
    for _ in range(max_levels - 1):
        if min(current.shape[:2]) <= minimum_size:
            break
        height = max(1, current.shape[0] // 2)
        width = max(1, current.shape[1] // 2)
        current = np.ascontiguousarray(
            cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
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
            self._items.pop(frame_id, None)
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
