from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    frame_id: str
    image: np.ndarray
    timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3 or self.image.dtype != np.uint8:
            raise ValueError("frame images must be RGB uint8 arrays")


def perceptual_hash(image: np.ndarray, size: int = 8) -> int:
    """Return a compact average hash suitable for near-duplicate filtering."""
    _validate_image(image)
    if size <= 0:
        raise ValueError("hash size must be positive")
    grayscale = image.astype(np.float32).mean(axis=2)
    y_indices = np.linspace(0, grayscale.shape[0] - 1, size).round().astype(int)
    x_indices = np.linspace(0, grayscale.shape[1] - 1, size).round().astype(int)
    reduced = grayscale[np.ix_(y_indices, x_indices)]
    average = float(reduced.mean())
    result = 0
    for value in reduced.flat:
        result = (result << 1) | int(value >= average)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def quality_score(image: np.ndarray) -> float:
    """Estimate usable detail and exposure, independent of model inference."""
    _validate_image(image)
    grayscale = image.astype(np.float32).mean(axis=2)
    if min(grayscale.shape) < 3:
        return 0.0
    horizontal = np.diff(grayscale, axis=1)
    vertical = np.diff(grayscale, axis=0)
    detail = min(1.0, float((horizontal.var() + vertical.var()) / 5000.0))
    exposure = float(np.clip(1.0 - abs(float(grayscale.mean()) - 127.5) / 127.5, 0.0, 1.0))
    return round(0.7 * detail + 0.3 * exposure, 6)


def select_keyframes(
    frames: list[FrameCandidate], *, max_frames: int | None = None, hash_distance: int = 4
) -> list[FrameCandidate]:
    """Keep quality-ranked, visually distinct frames while retaining source order."""
    if hash_distance < 0:
        raise ValueError("hash distance cannot be negative")
    ranked = sorted(
        enumerate(frames), key=lambda item: (-quality_score(item[1].image), item[0])
    )
    selected: list[FrameCandidate] = []
    hashes: list[int] = []
    for _, frame in ranked:
        current_hash = perceptual_hash(frame.image)
        if any(hamming_distance(current_hash, previous) <= hash_distance for previous in hashes):
            continue
        selected.append(frame)
        hashes.append(current_hash)
        if max_frames is not None and len(selected) >= max_frames:
            break
    selected_ids = {frame.frame_id for frame in selected}
    return [frame for frame in frames if frame.frame_id in selected_ids]


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("images must be RGB uint8 arrays")
