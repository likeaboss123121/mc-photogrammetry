"""Read-only inventory of existing Minecraft saves, and staged world copies.

The factory never opens a source world for writing and never arms a plan against
one. It reads the save list, then copies the chosen world into a staging game
directory; the Fabric mod is only ever pointed at the copy. Everything that
joins a user-supplied name to a filesystem path goes through :func:`safe_join`,
so a crafted or accidental ``..`` cannot escape the intended root.

A staged copy is provably the copy that was requested, not merely a directory
that happens to have the right name:

* the source tree is preflighted *without following links* and any symlink,
  junction, or other reparse point refuses the whole copy;
* an assigned file-count cap and total-byte cap bound the work;
* each file's SHA-256 is recorded into a fingerprint that the plan carries as a
  staging token, so the Fabric mod can recompute it over the world it actually
  opened and refuse to mutate anything when it differs;
* the source inventory is taken again after the copy, so a save that changed
  underneath the copy is reported instead of silently staged half-updated.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.dataset.hashing import (
    IGNORED_STAGING_ENTRIES,
    StagedFile,
    StagingLimitError,
    StagingLimits,
    is_link_like,
    scan_world_tree,
    staging_fingerprint_for,
)
from mc_photogrammetry.dataset.schemas import DatasetSchemaError

_IGNORED_WORLD_ENTRIES = IGNORED_STAGING_ENTRIES

#: The authoritative staging marker, written into the *copied world's own root*
#: after the copy is verified. The Fabric mod resolves the real open save root
#: through the integrated server (the save folder name need not equal the level
#: name) and reads this file; the game-directory copy is provenance only. The
#: marker is created after copy verification, so it never perturbs the staged
#: fingerprint, and it is not re-created or rehashed while the game runs.
WORLD_MARKER_NAME = ".mc-photo-capture-staging.json"


def default_game_dir() -> Path:
    """The platform's usual ``.minecraft`` location (never created here)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / ".minecraft"
    return Path.home() / ".minecraft"


def _modified_utc(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class WorldSave:
    """One save directory found under ``saves/``. Metadata only; never opened for write."""

    name: str
    path: Path
    size_bytes: int
    modified_utc: str
    level_name: str | None
    last_played_ms: int | None

    @property
    def world_identity(self) -> str:
        """The identity the mod is asked to validate against.

        ``level.dat``'s ``LevelName`` is what the integrated server reports, so
        that is the primary identity; the directory name is the fallback for a
        world whose ``level.dat`` cannot be read.
        """
        return self.level_name or self.name

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "modified_utc": self.modified_utc,
            "level_name": self.level_name,
            "last_played_ms": self.last_played_ms,
            "world_identity": self.world_identity,
        }


def read_level_metadata(level_dat: Path) -> tuple[str | None, int | None]:
    """Best-effort ``(LevelName, LastPlayed)`` from ``level.dat``.

    ``nbtlib`` is a project dependency, but a save in an unusual state must not
    take the whole inventory down, so any failure degrades to ``(None, None)``
    and the caller falls back to the directory name.
    """
    try:
        import nbtlib

        root = nbtlib.load(level_dat)
        data = root["Data"]
        level_name = str(data["LevelName"]) if "LevelName" in data else None
        last_played = int(data["LastPlayed"]) if "LastPlayed" in data else None
        return level_name, last_played
    except Exception:  # pragma: no cover - damaged/absent level.dat
        return None, None


def inventory_saves(game_dir: Path | str) -> tuple[WorldSave, ...]:
    """List saves under ``<game_dir>/saves`` in stable name order. Read-only."""
    root = Path(game_dir)
    saves = root / "saves"
    if not root.is_dir():
        raise DatasetSchemaError(f"no such game directory: {root}")
    if not saves.is_dir():
        return ()
    found: list[WorldSave] = []
    for child in sorted(saves.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        level_name, last_played = read_level_metadata(child / "level.dat")
        found.append(WorldSave(
            name=child.name,
            path=child,
            size_bytes=_directory_size(child),
            modified_utc=_modified_utc(child),
            level_name=level_name,
            last_played_ms=last_played,
        ))
    return tuple(found)


def _directory_size(path: Path) -> int:
    """Read-only size sum used for inventory display; never follows links."""
    total = 0
    for current, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current)
        directory_names[:] = [
            name for name in sorted(directory_names)
            if not is_link_like(current_path / name)
        ]
        for name in file_names:
            candidate = current_path / name
            try:
                if candidate.is_file() and not is_link_like(candidate):
                    total += candidate.stat().st_size
            except OSError:  # pragma: no cover - racing/locked files
                continue
    return total


def safe_component(value: str) -> str:
    """Validate one user-supplied path component (world name, dataset id, frame id)."""
    if not value or value in (".", ".."):
        raise DatasetSchemaError(f"unsafe path component: {value!r}")
    if any(separator in value for separator in ("/", "\\", "\0")):
        raise DatasetSchemaError(f"path component must not contain separators: {value!r}")
    if value != value.strip() or value.endswith("."):
        raise DatasetSchemaError(f"unsafe path component: {value!r}")
    return value


def safe_join(root: Path | str, *components: str) -> Path:
    """Join components under ``root`` and refuse anything that escapes it."""
    base = Path(root).resolve()
    for component in components:
        safe_component(str(component))
    candidate = base.joinpath(*components)
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise DatasetSchemaError(f"path {resolved} escapes {base}")
    return candidate


@dataclass(frozen=True, slots=True)
class StagedWorld:
    """A completed staged copy plus the digests that prove what it is."""

    destination: Path
    source_fingerprint: str
    file_count: int
    total_bytes: int

    @property
    def staging_token(self) -> str:
        """The token the plan requires the staged world to carry."""
        return staging_token_for(self.source_fingerprint)

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": str(self.destination),
            "source_fingerprint": self.source_fingerprint,
            "staging_token": self.staging_token,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


def _inventory_rows(entries: Iterable[StagedFile]) -> dict[str, tuple[int, int]]:
    return {entry.relative: (entry.size, entry.modified_ns) for entry in entries}


def preflight_world_copy(
    source: Path | str,
    *,
    limits: StagingLimits | None = None,
) -> tuple[StagedFile, ...]:
    """Validate a save tree without following links and inside the assigned caps."""
    return scan_world_tree(source, limits=limits)


def copy_world_to_staging(
    save: WorldSave,
    staging_game_dir: Path | str,
    *,
    name: str | None = None,
    force: bool = False,
    limits: StagingLimits | None = None,
) -> Path:
    """Copy a save into ``<staging_game_dir>/saves/<name>`` and return the copy.

    The source is only ever read. ``session.lock`` is skipped: copying it would
    hand the staged world a lock file claiming a session that never opened, and
    Minecraft rewrites it anyway.

    Preflights the source (rejecting links, enforcing the caps), copies, verifies
    the copy's fingerprint equals the source's, and re-inventories the source to
    detect a save that changed while it was being copied.
    """
    staged = stage_world(save, staging_game_dir, name=name, force=force, limits=limits)
    return staged.destination


def stage_world(
    save: WorldSave,
    staging_game_dir: Path | str,
    *,
    name: str | None = None,
    force: bool = False,
    limits: StagingLimits | None = None,
) -> StagedWorld:
    """Like :func:`copy_world_to_staging` but returns the staged-copy digest record.

    The copy is verified against the source fingerprint before it is returned, so
    the digest cannot describe a copy that was silently truncated or swapped.
    """
    caps = limits or StagingLimits()
    target_name = safe_component(name or save.name)
    destination_root = Path(staging_game_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = safe_join(destination_root, "saves", target_name)
    try:
        before = preflight_world_copy(save.path, limits=caps)
    except StagingLimitError as error:
        raise DatasetSchemaError(f"refusing to stage {save.path}: {error}") from error
    before_rows = _inventory_rows(before)
    source_fingerprint = staging_fingerprint_for([save.path])

    if destination.exists():
        if not force:
            raise DatasetSchemaError(
                f"staged world already exists: {destination}; pass force to replace it")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(save.path, destination,
                    ignore=shutil.ignore_patterns(*_IGNORED_WORLD_ENTRIES))

    try:
        after = preflight_world_copy(save.path, limits=caps)
    except StagingLimitError as error:
        raise DatasetSchemaError(
            f"source save changed while it was being copied: {error}") from error
    if _inventory_rows(after) != before_rows:
        raise DatasetSchemaError(
            f"source save {save.path} changed while it was being copied; re-run once "
            "nothing is writing to it")
    if staging_fingerprint_for([destination]) != source_fingerprint:
        raise DatasetSchemaError(
            f"staged copy {destination} does not match the source fingerprint")
    return StagedWorld(
        destination=destination,
        source_fingerprint=source_fingerprint,
        file_count=len(before),
        total_bytes=sum(entry.size for entry in before),
    )


def staging_marker_payload(
    *,
    token: str,
    source_fingerprint: str,
    world_identity: str,
    dataset_id: str,
    scene_id: str,
    plan_id: str,
    schema_version: int = 1,
) -> dict[str, object]:
    """The staging marker the Fabric mod reads before it mutates anything."""
    return {
        "schema_version": int(schema_version),
        "token": str(token),
        "source_fingerprint": str(source_fingerprint),
        "world_identity": str(world_identity),
        "dataset_id": str(dataset_id),
        "scene_id": str(scene_id),
        "plan_id": str(plan_id),
    }


def staging_token_for(source_fingerprint: str) -> str:
    """Deterministic plan-required staging token for one staged world copy.

    The token *is* the content fingerprint of the source save: the mod recomputes
    the same fingerprint over the world it actually opened and refuses to mutate
    anything when the two differ. Re-using the fingerprint instead of hashing the
    digest again keeps the two sides identical by construction.
    """
    if not source_fingerprint:
        raise DatasetSchemaError("a staging token needs a source fingerprint")
    return str(source_fingerprint)


def world_marker_path(staged_world: Path | str) -> Path:
    """The marker inside a copied world's root that the Fabric mod reads.

    This is the only staging proof the mod accepts. It deliberately does not
    re-derive the path from the level name: the caller already has the exact
    destination directory returned by :func:`stage_world`.
    """
    return Path(staged_world) / WORLD_MARKER_NAME
