from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class SourceFrame:
    frame_id: str
    source_path: str
    kind: MediaKind
    timestamp_seconds: float | None = None
    content_hash: str = ""
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class MediaManifest:
    frames: list[SourceFrame] = field(default_factory=list)

    @property
    def image_frames(self) -> list[SourceFrame]:
        return [frame for frame in self.frames if frame.kind is MediaKind.IMAGE]

    @property
    def video_sources(self) -> list[SourceFrame]:
        return [frame for frame in self.frames if frame.kind is MediaKind.VIDEO]


def discover_media(paths: Iterable[str | Path], *, recursive: bool = False) -> MediaManifest:
    """Discover supported files in stable path order without decoding media yet."""
    candidates: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates.extend(candidate for candidate in path.glob(pattern) if candidate.is_file())
        elif path.is_file():
            candidates.append(path)
        else:
            raise FileNotFoundError(path)

    frames: list[SourceFrame] = []
    seen_hashes: set[str] = set()
    for path in sorted(set(candidates), key=lambda item: str(item).casefold()):
        suffix = path.suffix.casefold()
        if suffix in IMAGE_EXTENSIONS:
            content_hash = _sha256(path)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            frame_id = f"image-{len(frames):04d}"
            frames.append(
                SourceFrame(
                    frame_id=frame_id,
                    source_path=str(path.resolve()),
                    kind=MediaKind.IMAGE,
                    content_hash=content_hash,
                )
            )
        elif suffix in VIDEO_EXTENSIONS:
            frames.append(
                SourceFrame(
                    frame_id=f"video-{len(frames):04d}",
                    source_path=str(path.resolve()),
                    kind=MediaKind.VIDEO,
                )
            )
    return MediaManifest(frames=frames)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
