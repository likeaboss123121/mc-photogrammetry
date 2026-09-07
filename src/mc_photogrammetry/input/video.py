from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .frames import FrameCandidate


class VideoDecodeError(ValueError):
    """Raised when a video cannot be opened or decoded."""


def iter_video_frames(
    path: str | Path, *, sample_every: int = 1, max_frames: int | None = None
) -> Iterator[FrameCandidate]:
    """Yield sampled RGB frames without loading an entire video into memory."""
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    try:
        import cv2
    except ImportError as error:
        raise VideoDecodeError("video decoding requires opencv-python") from error

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"could not open video {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 0.0
    frame_number = 0
    yielded = 0
    try:
        while max_frames is None or yielded < max_frames:
            success, bgr = capture.read()
            if not success:
                break
            if frame_number % sample_every == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = frame_number / fps if fps else None
                yield FrameCandidate(
                    frame_id=f"video-{frame_number:06d}",
                    image=rgb,
                    timestamp_seconds=timestamp,
                )
                yielded += 1
            frame_number += 1
    finally:
        capture.release()
