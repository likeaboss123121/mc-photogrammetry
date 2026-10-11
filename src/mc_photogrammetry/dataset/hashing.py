"""Deterministic hashing helpers shared by the capture dataset factory.

The dataset factory has to answer two questions repeatedly and exactly: "are
these the bytes I recorded?" and "is this scene the same scene I planned?". Both
are answered with SHA-256 over canonical bytes, so no timestamp, dict ordering,
or filesystem enumeration order can change an answer.

:func:`staging_fingerprint` answers a third question - "is the world the client
opened the world I staged?" - over file *content*, and its row format is
duplicated by the Fabric mod's ``PlanAudit.stagingFingerprint``. The two
implementations must stay byte-identical because the mod recomputes the
fingerprint of the opened save and compares it with the one Python recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Worlds are user saves, not synthetic fixtures: bound the staging work so a
#: pathological directory tree cannot turn "copy the world" into an unbounded
#: read. Both Python and the mod's planner-side checks use the same defaults.
DEFAULT_MAX_STAGING_FILES = 200_000
DEFAULT_MAX_STAGING_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB

#: Save-directory entries that are never part of the staged world.
IGNORED_STAGING_ENTRIES = ("session.lock",)

_WINDOWS_REPARSE_POINT = 0x0400


class StagingLimitError(ValueError):
    """Raised when a save tree is unsafe or exceeds the staging caps."""


@dataclass(frozen=True, slots=True)
class StagedFile:
    """One regular file found under a save root, with everything needed to hash it."""

    relative: str
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class StagingLimits:
    """Assigned caps for one staged copy; both must be positive."""

    max_files: int = DEFAULT_MAX_STAGING_FILES
    max_bytes: int = DEFAULT_MAX_STAGING_BYTES

    def __post_init__(self) -> None:
        if int(self.max_files) <= 0 or int(self.max_bytes) <= 0:
            raise StagingLimitError("staging caps must be positive")


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so large screenshots stay cheap."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Stable JSON text: sorted keys, no incidental whitespace, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(payload: Any) -> str:
    """SHA-256 over the canonical form of a JSON-compatible payload."""
    return sha256_text(canonical_json(payload))


def is_link_like(path: Path) -> bool:
    """True for a symlink or any Windows reparse point (junction, mount point).

    ``shutil`` follows symlinks by default, which would let a crafted save drag
    arbitrary files from outside the save root into the staged copy. Everything
    that walks a save must check this *before* descending.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:  # pragma: no cover - racing/locked entries
        return True
    try:
        attributes = path.lstat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    return bool(attributes & _WINDOWS_REPARSE_POINT)


def scan_world_tree(
    root: Path | str,
    *,
    limits: StagingLimits | None = None,
    ignored_entries: Sequence[str] = IGNORED_STAGING_ENTRIES,
    reject_links: bool = True,
) -> tuple[StagedFile, ...]:
    """Walk a save tree without following links and enforce both caps.

    Returns every included regular file in sorted relative-path order, so the
    result is independent of enumeration order. Raises :class:`StagingLimitError`
    for a symlink/reparse point, too many files, or too many bytes.
    """
    caps = limits or StagingLimits()
    base = Path(root)
    if not base.is_dir():
        raise StagingLimitError(f"world copy source is not a directory: {base}")
    ignored = set(ignored_entries)
    found: list[StagedFile] = []
    total_bytes = 0
    for current, directory_names, file_names in os.walk(base, followlinks=False):
        current_path = Path(current)
        if is_link_like(current_path) and current_path != base:
            # os.walk with followlinks=False does not descend into a linked
            # directory, but it still yields it; refuse loudly instead of
            # silently skipping part of the world.
            raise StagingLimitError(
                f"save tree contains a symlink or reparse point: {current_path}")
        # A linked directory must be pruned before the walk enters it.
        kept_directories = []
        for name in sorted(directory_names):
            child = current_path / name
            if reject_links and is_link_like(child):
                raise StagingLimitError(
                    f"save tree contains a symlink or reparse point: {child}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            if name in ignored:
                continue
            path = current_path / name
            if reject_links and is_link_like(path):
                raise StagingLimitError(
                    f"save tree contains a symlink or reparse point: {path}")
            try:
                metadata = path.stat()
            except OSError as error:
                raise StagingLimitError(f"cannot stat {path}: {error}") from error
            total_bytes += int(metadata.st_size)
            if len(found) + 1 > caps.max_files:
                raise StagingLimitError(
                    f"world copy exceeds {caps.max_files} files")
            if total_bytes > caps.max_bytes:
                raise StagingLimitError(
                    f"world copy exceeds {caps.max_bytes} bytes")
            found.append(StagedFile(
                relative=path.relative_to(base).as_posix(),
                size=int(metadata.st_size),
                modified_ns=int(metadata.st_mtime_ns),
            ))
    found.sort(key=lambda item: item.relative)
    return tuple(found)


def staging_fingerprint_for(paths: Iterable[Path | str]) -> str:
    """Content fingerprint of already-scanned roots.

    Row format: ``sha256<TAB>size<TAB>relative-path`` joined by ``\\n``, each row
    terminated by ``\\n``. ``PlanAudit.stagingFingerprint`` produces exactly the
    same bytes on the Java side.
    """
    rows: list[tuple[str, int, str]] = []
    for root in paths:
        base = Path(root)
        for entry in scan_world_tree(base):
            rows.append((sha256_file(base / entry.relative), entry.size, entry.relative))
    rows.sort(key=lambda row: (row[2], row[0]))
    text = "".join(f"{digest}\t{size}\t{relative}\n" for digest, size, relative in rows)
    return sha256_text(text)


def staging_fingerprint(path: Path | str) -> str:
    """Fingerprint one save/staged-world root from its file contents."""
    return staging_fingerprint_for([path])
