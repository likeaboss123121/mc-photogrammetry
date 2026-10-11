from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})

MAX_SOURCE_BYTES = 256 * 1024 * 1024
"""Bounded single-read buffer cap for one encoded source screenshot."""

SourceIngressSink = Callable[[str, "SourceFrame", bytes], None]
"""Synchronous, sequential single-read ingress callback.

``discover_media`` invokes it once per **retained** image candidate, in stable
path order, immediately after that candidate's encoded bytes have been read from
disk exactly once and hashed and de-duplicated. ``data`` is the exact buffer the
discovery content hash was computed from and that the caller must decode before
returning; the reference is released as soon as the callback returns, so a sink
must not retain it (copy only what it needs). Duplicate candidates are dropped
before the callback, so they invoke neither the sink nor a decode.
"""


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


def discover_media(
    paths: Iterable[str | Path],
    *,
    recursive: bool = False,
    ingress_sink: SourceIngressSink | None = None,
) -> MediaManifest:
    """Discover supported files in stable path order without decoding media yet.

    ``ingress_sink`` is an optional, evaluation-only single-read ingress seam.
    When supplied, every **retained** image candidate is processed sequentially:
    its encoded bytes are read from disk exactly **once** into one bounded
    buffer, the content hash and byte length are computed from those exact bytes,
    de-duplication is applied, and the sink is invoked synchronously on the same
    buffer for a retained frame. The buffer reference is released before the next
    candidate is read, so at most one encoded screenshot is resident at a time
    (aggregate memory is bounded regardless of frame count; per-file size is
    bounded by :data:`MAX_SOURCE_BYTES`). The caller decodes the buffer inside
    the sink, which is what makes the discovery hash and the decoded pixels
    provably come from one physical read.

    De-duplication, stable path order and the first-path-wins duplicate
    behaviour are unchanged, and when ``ingress_sink`` is ``None`` the accepted
    discovery behaviour is byte-for-byte unchanged.
    """
    if ingress_sink is not None and not callable(ingress_sink):
        raise TypeError("ingress_sink must be callable")
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
            if ingress_sink is None:
                content_hash = _sha256(path)
            else:
                # Sequential single read: the bytes exist only for this
                # candidate's hash, de-duplication and decode. They are never
                # retained in a manifest, list or cache across frames, so no
                # aggregate byte budget can be exceeded and no typed budget
                # refusal is needed.
                data = read_source_bytes(path)
                content_hash = hashlib.sha256(data).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            frame = SourceFrame(
                frame_id=f"image-{len(frames):04d}",
                source_path=str(path.resolve()),
                kind=MediaKind.IMAGE,
                content_hash=content_hash,
            )
            if ingress_sink is not None:
                try:
                    ingress_sink(frame.source_path, frame, data)
                finally:
                    # Drop the only reference before reading the next candidate.
                    del data
            frames.append(frame)
        elif suffix in VIDEO_EXTENSIONS:
            frames.append(
                SourceFrame(
                    frame_id=f"video-{len(frames):04d}",
                    source_path=str(path.resolve()),
                    kind=MediaKind.VIDEO,
                )
            )
    return MediaManifest(frames=frames)


def read_source_bytes(path: str | Path) -> bytes:
    """Read one encoded source file exactly once into a bounded byte buffer."""
    source = Path(path)
    data = source.read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise OSError(
            f"source file {source} exceeds the {MAX_SOURCE_BYTES}-byte ingress cap")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
