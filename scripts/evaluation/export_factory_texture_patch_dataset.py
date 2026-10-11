"""Evaluation-only E3 factory patch-dataset exporter.

This module is the dataset-side counterpart to the reviewed E1 patch-label
harness. It reads *finalized* factory datasets only - a dataset root with a
hash-pinned ``index.json`` and ``scenes/<id>/{capture,ground_truth,recipe,
reference}`` artifacts - derives exact ``(BlockState, coordinate, face slot)``
supervision from the reference geometry, the **recorded actual camera** and the
captured pixels, and writes one versioned, sealed, evaluation-only artifact.

Scope and refusals (all enforced here, none delegated to a caller):

* Only finalized dataset entries are admissible. Every scene-file and screenshot
  digest recorded in ``index.json`` is recomputed before a single screenshot is
  decoded; a staging (unindexed/incomplete) scene directory, a missing index, a
  hash mismatch or a path that escapes the dataset root is a refusal, never a
  warning.
* The camera is rebuilt from the frame's recorded ``actual`` pose
  (``eye_position``/``yaw_degrees``/``pitch_degrees``/``fov_degrees``) rather than
  the planned pose. A capture that does not record an actual pose is refused; the
  exporter never falls back to the planned cameras to make a frame admissible.
* A capture is admitted only when the client explicitly reported
  ``hud_hidden == true``, ``active_resource_packs == ["vanilla"]`` and no shader
  pack. Unknown pack/shader state is refused, never silently treated as vanilla.
  Capture admission is per frame: a refused capture is recorded with a typed
  reason and excluded while the scene's other captures may still export, because
  the audit requires the factory's HUD-visible frames to be dropped and its
  hidden ones kept (`docs/reports/texture-poc-data-audit-2026-10-03.md` §1.3).
* Labels come from the exact baked-model palette
  (``provider.palette_for_states(reference_states, include_partial=True)``) and
  the same alpha-tested ownership renderer the production sampler uses. A
  reference state without exact model polygons is counted as
  ``model_metadata_unavailable`` and never labelled through the cube fallback.
  Sampling runs at production ``shrink = raycast.FACE_SHRINK`` (1.0).
* No runtime coordinate is inferred from a label: the pose is the recorded one
  and geometry is the exported reference, so the projection and the label agree
  by construction.

The artifact stores the rectified 16x16x3 RGB patch, its boolean ``valid`` mask,
its fractional ``coverage``, the exact label state, frame/slot, cell coordinate,
``scene_group``, build provenance (``staging_token``/``world_identity``) and
per-array and per-example digests. ``scene_group`` is the authoritative leakage
group: the exported grouping keeps every capture of one ``scene_group`` in one
fold and records the reviewed leave-one-group-out manifest. The pinned T2/T6
fixtures are never an input (this module reads finalized factory roots only) and
the artifact asserts ``t2_t6_excluded``.

This task builds the exporter and its synthetic tests only. It does not run the
factory corpus, read a real image, reference, camera, scene or JAR, train a model
or touch production/runtime code.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # direct-script execution without PYTHONPATH=src
    sys.path.insert(0, str(_SRC))

from mc_photogrammetry.blocks.appearance import PaletteColor  # noqa: E402
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS  # noqa: E402
from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider  # noqa: E402
from mc_photogrammetry.dataset.blocks import parse_ground_truth  # noqa: E402
from mc_photogrammetry.dataset.hashing import canonical_json, sha256_file  # noqa: E402
from mc_photogrammetry.dataset.schemas import DatasetSchemaError  # noqa: E402
from mc_photogrammetry.domain.models import BlockState  # noqa: E402
from mc_photogrammetry.evaluation_pair_correspondence import (  # noqa: E402
    PairCorrespondenceEvaluationError,
    build_group_manifest,
    check_split_leakage,
    discover_finalized_entries,
)
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.input.image import load_image  # noqa: E402
from mc_photogrammetry.reconstruction import raycast  # noqa: E402
from mc_photogrammetry.reconstruction.camera import CameraModel  # noqa: E402
from mc_photogrammetry.reconstruction.face_sampling import (  # noqa: E402
    DEPTH_SCALE,
    MINIMUM_FACE_AREA,
    PATCH_SIZE,
    exact_model_states,
    sample_visible_faces,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform  # noqa: E402

SCHEMA = "mc-photogrammetry-factory-texture-patch-dataset-v1"
"""Schema identity of the sealed factory patch-dataset artifact."""

ENGINE_VERSION = "e3-1"
"""Exporter revision recorded in the artifact."""

MODE = "factory/recorded-camera exact-model label export (evaluation-only)"
"""The only mode this exporter reports: finalized factory captures, recorded poses."""

ARTIFACT_ID = "texture-poc-e3-factory-texture-patch-dataset"

MINIMUM_PATCH_VALID_FRACTION = 0.5
"""Below this share of owned patch pixels a face is refused, not labelled."""

MAX_EXAMPLES = 200_000
"""Hard cap on exported examples, so one dataset cannot produce an unbounded artifact."""

VANILLA_PACK = "vanilla"
"""The only explicitly-reported active pack this POC admits."""

MODEL_METADATA_UNAVAILABLE = "model_metadata_unavailable"
REASON_NO_VALID_PIXELS = "no_valid_pixels"
REASON_INSUFFICIENT_VALID_COVERAGE = "insufficient_valid_coverage"

# Capture admission refusals. Frame-level and recorded, never silent.
REASON_CAPTURE_FAILED = "capture_failed"
REASON_INVALID_FILENAME = "invalid_screenshot_filename"
REASON_MISSING_SCREENSHOT = "missing_screenshot_digest"
REASON_MISSING_ACTUAL = "missing_actual_metadata"
REASON_MISSING_POSE = "missing_actual_pose"
REASON_HUD_NOT_HIDDEN = "hud_not_hidden"
REASON_NON_VANILLA_PACK = "non_vanilla_pack_metadata"
REASON_UNKNOWN_PACK = "unknown_pack_metadata"

ADMISSION_REASONS = (
    REASON_CAPTURE_FAILED,
    REASON_INVALID_FILENAME,
    REASON_MISSING_SCREENSHOT,
    REASON_MISSING_ACTUAL,
    REASON_MISSING_POSE,
    REASON_HUD_NOT_HIDDEN,
    REASON_NON_VANILLA_PACK,
    REASON_UNKNOWN_PACK,
)
"""Every typed capture-admission refusal this exporter can record."""

_REQUIRED_SCENE_FILES = ("capture.json", "ground_truth.json", "reference.litematic")
_FILENAME_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.png$")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")

EXCLUDED_FIXTURE_DIRS = ("templates 2", "templates 6")
"""The pinned T2/T6 evaluation fixtures are never export or training data."""


class FactoryPatchExportError(ValueError):
    """Invalid, unverifiable, unpinned or unsafe E3 factory export input."""


def _default_jar_path() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / ".minecraft" / "versions" / "1.21.5" / "1.21.5.jar"


# --------------------------------------------------------------------------- #
# Primitive serialization and hashing
# --------------------------------------------------------------------------- #


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used by every digest in this module."""
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def state_to_record(state: BlockState) -> dict[str, Any]:
    return {"name": state.name, "properties": [[key, value] for key, value in state.properties]}


def state_from_record(record: object) -> BlockState:
    """Rehydrate a frozen state record, refusing anything malformed."""
    if not isinstance(record, Mapping):
        raise FactoryPatchExportError("state record must be an object")
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise FactoryPatchExportError("state record needs a non-empty name")
    raw_properties = record.get("properties", [])
    if not isinstance(raw_properties, list):
        raise FactoryPatchExportError("state record properties must be a list")
    pairs: list[tuple[str, str]] = []
    for item in raw_properties:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise FactoryPatchExportError("state record property must be a [key, value] pair")
        pairs.append((str(item[0]), str(item[1])))
    try:
        return BlockState(name, tuple(sorted(pairs)))
    except ValueError as error:
        raise FactoryPatchExportError(f"state record is not a valid BlockState: {error}") from error


def array_record(array: np.ndarray) -> dict[str, Any]:
    """Exact, self-describing bytes for one frozen array."""
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(size) for size in value.shape],
        "base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def array_from_record(record: object, *, label: str = "array") -> np.ndarray:
    if not isinstance(record, Mapping):
        raise FactoryPatchExportError(f"{label} record must be an object")
    try:
        dtype = np.dtype(str(record["dtype"]))
        shape = tuple(int(size) for size in record["shape"])
        raw = base64.b64decode(str(record["base64"]), validate=True)
        array = np.frombuffer(raw, dtype=dtype)
    except (KeyError, TypeError, ValueError) as error:
        raise FactoryPatchExportError(f"{label} record is malformed") from error
    expected = int(np.prod(shape)) if shape else 1
    if array.size != expected:
        raise FactoryPatchExportError(f"{label} byte length does not match its shape")
    return array.reshape(shape).copy()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FactoryPatchExportError(f"missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FactoryPatchExportError(f"unreadable JSON {path}: {error}") from error


# --------------------------------------------------------------------------- #
# Small numeric readers
# --------------------------------------------------------------------------- #


def _finite(value: Any) -> float | None:
    """A finite JSON number, or None; booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_triple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    parts = [_finite(item) for item in value]
    if any(part is None for part in parts):
        return None
    return (float(parts[0]), float(parts[1]), float(parts[2]))  # type: ignore[arg-type]


def _int_pair(value: Any) -> tuple[int, int] | None:
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)):
        return int(value[0]), int(value[1])
    return None


# --------------------------------------------------------------------------- #
# Dataset admission: finalized index, safe paths, provenance
# --------------------------------------------------------------------------- #


def _directory_parts(directory: object) -> tuple[str, ...]:
    """Split an index ``directory`` into safe relative parts, or refuse."""
    normalized = str(directory).replace("\\", "/")
    if not normalized or normalized.startswith("/") or _DRIVE_PREFIX.match(normalized):
        raise FactoryPatchExportError(f"refusing unsafe scene directory {directory!r}")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise FactoryPatchExportError(f"refusing unsafe scene directory {directory!r}")
    return parts


def _validate_scene_directory(dataset_root: Path, entry) -> None:
    parts = _directory_parts(entry.directory)
    if parts[0] != "scenes" or len(parts) != 2:
        raise FactoryPatchExportError(
            f"scene {entry.scene_id!r} directory {entry.directory!r} is not 'scenes/<id>'")
    scene_root = Path(entry.scene_root)
    root = dataset_root.resolve()
    resolved = scene_root.resolve()
    expected = (root / "scenes" / parts[1]).resolve()
    if resolved != expected:
        raise FactoryPatchExportError(
            f"scene {entry.scene_id!r} directory {entry.directory!r} does not resolve to its indexed scene")


def _preflight_index_paths(dataset_root: Path) -> None:
    """Validate every path in every index before the shared helper hashes files.

    ``discover_finalized_entries`` hashes index-referenced paths as it discovers
    them. Validate the untrusted directory and filename fields first so a
    malformed index cannot make that helper read outside the dataset root.
    """
    root = dataset_root.resolve()
    indexes = sorted(root.glob("**/index.json"))
    if not indexes:
        raise FactoryPatchExportError(f"refusing dataset root {root}: no index.json")
    for index_path in indexes:
        if index_path.resolve().parent != root:
            raise FactoryPatchExportError(
                f"refusing nested or escaping dataset index {index_path}")
        payload = _read_json(index_path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("scenes"), list):
            raise FactoryPatchExportError(f"{index_path}: malformed scene index")
        for scene in payload["scenes"]:
            if not isinstance(scene, Mapping):
                raise FactoryPatchExportError(f"{index_path}: malformed scene entry")
            parts = _directory_parts(scene.get("directory", ""))
            scene_id = str(scene.get("scene_id", ""))
            if (len(parts) != 2 or parts[0] != "scenes" or not scene_id
                    or parts[1] != scene_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scene_id)):
                raise FactoryPatchExportError(
                    f"{index_path}: unsafe or inconsistent scene directory {scene.get('directory')!r}")
            scene_root = root / "scenes" / scene_id
            if scene_root.resolve() != scene_root or root not in scene_root.resolve().parents:
                raise FactoryPatchExportError(
                    f"{index_path}: scene path is a symlink or escapes dataset root: {scene_root}")
            hashes = scene.get("hashes")
            if not isinstance(hashes, Mapping):
                raise FactoryPatchExportError(f"{index_path}: scene hashes must be an object")
            for name in hashes:
                relative = Path(str(name))
                if (relative.is_absolute() or any(part in ("..", "") for part in relative.parts)
                        or re.match(r"^[A-Za-z]:", str(name))):
                    raise FactoryPatchExportError(
                        f"{index_path}: unsafe scene hash path {name!r}")
                candidate = scene_root / relative
                if root not in candidate.resolve().parents:
                    raise FactoryPatchExportError(
                        f"{index_path}: scene hash path escapes dataset root: {name!r}")
            screenshots = scene.get("screenshots")
            if not isinstance(screenshots, list):
                raise FactoryPatchExportError(f"{index_path}: scene screenshots must be a list")
            for row in screenshots:
                if not isinstance(row, Mapping):
                    raise FactoryPatchExportError(f"{index_path}: malformed screenshot row")
                filename = str(row.get("filename", ""))
                if not _FILENAME_SAFE.fullmatch(filename):
                    raise FactoryPatchExportError(
                        f"{index_path}: unsafe screenshot filename {filename!r}")
                screenshot = scene_root / "captures" / filename
                if root not in screenshot.resolve().parents:
                    raise FactoryPatchExportError(
                        f"{index_path}: screenshot path escapes dataset root: {filename!r}")


def _assert_no_staging_scenes(dataset_root: Path, entries: Sequence[Any]) -> None:
    """Refuse any staging/incomplete scene directory, not just unindexed ones."""
    scenes_root = dataset_root / "scenes"
    if not scenes_root.is_dir():
        raise FactoryPatchExportError(f"no scenes directory under {dataset_root}")
    listed = {str(entry.directory).replace("\\", "/") for entry in entries}
    for child in sorted(scenes_root.iterdir()):
        if not child.is_dir():
            continue
        relative = child.relative_to(dataset_root).as_posix()
        if relative not in listed:
            raise FactoryPatchExportError(
                f"refusing staging/unindexed scene directory {child}: not a finalized index entry")
        missing = [name for name in _REQUIRED_SCENE_FILES if not (child / name).is_file()]
        if missing:
            raise FactoryPatchExportError(
                f"refusing incomplete scene {child}: missing {', '.join(missing)}")


def _index_scene_records(entries: Sequence[Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    processed: set[str] = set()
    for entry in entries:
        index_path = str(entry.index_path)
        if index_path in processed:
            continue
        processed.add(index_path)
        payload = _read_json(Path(index_path))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise FactoryPatchExportError(f"{index_path}: unsupported index schema")
        if payload.get("advanced_only") is not True:
            raise FactoryPatchExportError(
                f"{index_path}: index is not the advanced-only factory index")
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise FactoryPatchExportError(f"{index_path}: scenes must be a list")
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise FactoryPatchExportError(f"{index_path}: malformed scene entry")
            key = (str(scene.get("scene_id", "")), str(scene.get("directory", "")))
            if key in records:
                raise FactoryPatchExportError(f"{index_path}: duplicate scene entry {key}")
            records[key] = scene
    return records


def _scene_provenance(scene: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(staging_token, world_identity, split)`` for one finalized index entry."""
    provenance = scene.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FactoryPatchExportError("finalized scene entry needs a provenance object")
    staging_token = str(provenance.get("staging_token", ""))
    world_identity = str(provenance.get("world_identity", ""))
    if not is_sha256(staging_token):
        raise FactoryPatchExportError(
            "finalized scene provenance needs a 64-hex staging_token")
    if not world_identity:
        raise FactoryPatchExportError(
            "finalized scene provenance needs a non-empty world_identity")
    split = str(scene.get("split", ""))
    if split not in ("train", "dev", "test"):
        raise FactoryPatchExportError(f"scene has unknown index split {split!r}")
    return staging_token.lower(), world_identity, split


def _refuse_excluded_fixture_root(root: Path) -> None:
    """Refuse a T2/T6 fixture root by path component, case-insensitively."""
    parts = {part.casefold() for part in root.resolve().parts}
    for marker in EXCLUDED_FIXTURE_DIRS:
        if marker in parts:
            raise FactoryPatchExportError(
                f"refusing excluded T2/T6 fixture root {root}: the pinned evaluation "
                "fixtures are never export or training data")


def load_finalized_scenes(dataset_root: Path | str) -> tuple[tuple[Any, ...], dict[tuple[str, str], Mapping[str, Any]]]:
    """Hash-pin, safety-check and provenance-check every finalized scene entry.

    Wraps the reviewed ``discover_finalized_entries`` (which recomputes every
    pinned scene-file and screenshot digest) and adds the scene-directory,
    staging-scene and provenance checks this exporter needs.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FactoryPatchExportError(f"no dataset root at {root}")
    _refuse_excluded_fixture_root(root)
    _preflight_index_paths(root)
    try:
        entries = discover_finalized_entries(root)
    except PairCorrespondenceEvaluationError as error:
        raise FactoryPatchExportError(f"refusing dataset root {root}: {error}") from error
    for entry in entries:
        _validate_scene_directory(root, entry)
    _assert_no_staging_scenes(root, entries)
    records = _index_scene_records(entries)
    for entry in entries:
        key = (entry.scene_id, entry.directory)
        if key not in records:
            raise FactoryPatchExportError(
                f"finalized scene {entry.scene_id!r} is not present in its own index")
        _scene_provenance(records[key])
    return tuple(entries), records


# --------------------------------------------------------------------------- #
# Capture admission
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AdmittedCapture:
    """One capture that passed pose, HUD and vanilla-pack admission."""

    frame_id: str
    filename: str
    camera: CameraModel
    width: int
    height: int
    screenshot_sha256: str
    fov_degrees: float


def _pack_refusal(actual: Mapping[str, Any]) -> str | None:
    """``None`` only for an explicitly vanilla, shader-free capture.

    Unknown pack/shader state refuses rather than being treated as vanilla. A
    requested-but-unapplied resource pack name is provenance only and stays
    admissible because the client reports ``resource_pack_applied == false`` and
    lists only ``vanilla`` as active.
    """
    active = actual.get("active_resource_packs")
    if not isinstance(active, list) or any(isinstance(item, bool) for item in active):
        return REASON_UNKNOWN_PACK
    names = [str(item) for item in active]
    if any(name != VANILLA_PACK for name in names):
        return REASON_NON_VANILLA_PACK
    if names != [VANILLA_PACK]:
        return REASON_UNKNOWN_PACK
    applied = actual.get("resource_pack_applied")
    if applied is True:
        return REASON_NON_VANILLA_PACK
    if applied is not False:
        return REASON_UNKNOWN_PACK
    shader_applied = actual.get("shader_pack_applied")
    if shader_applied is True:
        return REASON_NON_VANILLA_PACK
    if shader_applied is not False:
        return REASON_UNKNOWN_PACK
    # Requested pack names are provenance only. The factory audit confirms
    # actual active/applied fields are authoritative; requiring the optional
    # requested.shader_pack field rejects valid vanilla captures.
    return None


def admit_capture(
    frame: Mapping[str, Any],
    *,
    expected_screenshot_sha256: str,
    fallback_resolution: Sequence[int] | None,
) -> tuple[AdmittedCapture | None, str | None]:
    """Admit one recorded capture, or return a typed refusal reason.

    The check order is fixed and documented so a frame with several problems
    always reports the same reason: capture success, filename safety, recorded
    screenshot digest, presence of the ``actual`` metadata, the recorded pose,
    HUD visibility and finally the pack/shader report.
    """
    if frame.get("captured") is not True:
        return None, REASON_CAPTURE_FAILED
    frame_id = str(frame.get("frame_id", ""))
    filename = str(frame.get("filename", ""))
    if not frame_id or not _FILENAME_SAFE.match(filename) or "/" in filename or "\\" in filename:
        return None, REASON_INVALID_FILENAME
    if not is_sha256(expected_screenshot_sha256):
        return None, REASON_MISSING_SCREENSHOT

    actual = frame.get("actual")
    if not isinstance(actual, Mapping):
        return None, REASON_MISSING_ACTUAL

    eye = _finite_triple(actual.get("eye_position"))
    feet = _finite_triple(actual.get("feet_position"))
    yaw = _finite(actual.get("yaw_degrees"))
    pitch = _finite(actual.get("pitch_degrees"))
    fov = _finite(actual.get("fov_degrees"))
    if feet is None or eye is None or yaw is None or pitch is None or fov is None:
        return None, REASON_MISSING_POSE
    try:
        camera = CameraModel(
            position=eye, yaw_degrees=yaw, pitch_degrees=pitch,
            fov_degrees=fov, eye_height=0.0,
        )
    except ValueError:
        return None, REASON_MISSING_POSE

    if actual.get("hud_hidden") is not True:
        return None, REASON_HUD_NOT_HIDDEN

    pack_problem = _pack_refusal(actual)
    if pack_problem is not None:
        return None, pack_problem

    resolution = _int_pair([actual.get("framebuffer_width"), actual.get("framebuffer_height")])
    if resolution is None or resolution[0] <= 0 or resolution[1] <= 0:
        fallback = _int_pair(list(fallback_resolution)) if fallback_resolution is not None else None
        resolution = fallback
    if resolution is None or resolution[0] <= 0 or resolution[1] <= 0:
        return None, REASON_MISSING_ACTUAL

    return AdmittedCapture(
        frame_id=frame_id,
        filename=filename,
        camera=camera,
        width=int(resolution[0]),
        height=int(resolution[1]),
        screenshot_sha256=expected_screenshot_sha256.lower(),
        fov_degrees=float(fov),
    ), None


# --------------------------------------------------------------------------- #
# Exact-state label derivation (mirror of the reviewed E1 derivation)
# --------------------------------------------------------------------------- #


def split_exact_states(
    blocks: Mapping[tuple[int, int, int], BlockState],
    palette: Sequence[PaletteColor],
) -> tuple[dict[tuple[int, int, int], BlockState], dict[tuple[int, int, int], BlockState], frozenset]:
    """Partition reference cells into exact-model and refused, nothing invented.

    Only states carrying exact baked model polygons in this palette are eligible
    for an exact label. A state without them is returned in the refused mapping
    so the caller counts ``model_metadata_unavailable``; it is never sampled
    through the cube fallback and silently labelled.
    """
    modelled = exact_model_states(palette)
    exact: dict[tuple[int, int, int], BlockState] = {}
    refused: dict[tuple[int, int, int], BlockState] = {}
    for coordinate, state in blocks.items():
        key = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
        (exact if state in modelled else refused)[key] = state
    return exact, refused, modelled


def _observed_patch(sample: object, valid: np.ndarray | None) -> np.ndarray:
    """The production observed patch: unowned pixels become the face's mean colour."""
    patch = np.asarray(getattr(sample, "patch"), dtype=np.float64)
    if valid is None or bool(valid.all()):
        return patch
    coverage = getattr(sample, "coverage", None)
    observed = patch.copy()
    pixels = patch[valid]
    if not len(pixels):
        observed[~valid] = 0.0
        return observed
    if coverage is None:
        observed[~valid] = pixels.mean(axis=0)
    else:
        weights = np.asarray(coverage, dtype=np.float64)[valid]
        observed[~valid] = (
            np.average(pixels, axis=0, weights=weights)
            if float(weights.sum()) > 1e-6
            else pixels.mean(axis=0)
        )
    return observed


def observation_weight(sample: object) -> float:
    """The exact production observation weight for one accepted face sample."""
    valid = getattr(sample, "valid", None)
    valid = None if valid is None else np.asarray(valid, dtype=bool)
    raw_coverage = getattr(sample, "coverage", None)
    coverage = None if raw_coverage is None else np.asarray(raw_coverage, dtype=np.float64)
    uniformity = float(
        raycast._face_uniformity(  # noqa: SLF001 - the frozen production divisor
            _observed_patch(sample, valid), coverage
        )
    )
    weight = float(sample.area) ** float(raycast.AREA_POWER) / (1.0 + float(sample.depth) / 16.0)
    if raycast.VISIBILITY_POWER:
        weight *= float(max(float(sample.visible), 0.0)) ** float(raycast.VISIBILITY_POWER)
    if raycast.OBLIQUITY_POWER:
        weight *= float(max(float(sample.facing), 0.0)) ** float(raycast.OBLIQUITY_POWER)
    weight /= 1.0 + uniformity / max(float(raycast.UNIFORMITY_SCALE), 1e-6)
    return float(weight)


def classify_sample(
    sample: object, *, minimum_valid_fraction: float = MINIMUM_PATCH_VALID_FRACTION
) -> tuple[bool, str | None, float, float]:
    """Accept a sampled face as a label, or return a typed refusal reason."""
    if minimum_valid_fraction <= 0.0 or minimum_valid_fraction > 1.0:
        raise FactoryPatchExportError("minimum_valid_fraction must be within (0, 1]")
    valid = getattr(sample, "valid", None)
    coverage = getattr(sample, "coverage", None)
    if valid is None:
        valid_fraction = 1.0
        coverage_mean = 1.0 if coverage is None else float(np.mean(coverage))
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.size == 0 or not valid.any():
            return False, REASON_NO_VALID_PIXELS, 0.0, 0.0
        valid_fraction = float(valid.mean())
        coverage_mean = (
            1.0 if coverage is None
            else float(np.mean(np.asarray(coverage, dtype=np.float64)[valid]))
        )
    if valid_fraction < minimum_valid_fraction:
        return False, REASON_INSUFFICIENT_VALID_COVERAGE, valid_fraction, coverage_mean
    return True, None, valid_fraction, coverage_mean


@dataclass(frozen=True, slots=True)
class PatchExample:
    """One frozen production-sampler patch and its exact reference-derived label."""

    scene_id: str
    dataset_id: str
    scene_group: str
    fold: str
    staging_token: str
    world_identity: str
    frame_id: str
    filename: str
    coordinate: tuple[int, int, int]
    world_coordinate: tuple[int, int, int]
    slot: str
    truth_state: BlockState
    weight: float
    area: float
    depth: float
    visible: float
    facing: float
    valid_fraction: float
    coverage_mean: float
    partial: bool
    screenshot_sha256: str
    patch: np.ndarray
    valid: np.ndarray
    coverage: np.ndarray

    def __post_init__(self) -> None:
        for name in ("scene_id", "dataset_id", "scene_group", "fold", "frame_id", "filename"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise FactoryPatchExportError(f"example {name} must be a non-empty string")
        if not is_sha256(self.staging_token):
            raise FactoryPatchExportError("example staging_token must be a sha256")
        if not self.world_identity:
            raise FactoryPatchExportError("example world_identity must not be empty")
        if self.slot not in FACE_SLOTS:
            raise FactoryPatchExportError(f"example slot must be one of {FACE_SLOTS}")
        for name in ("coordinate", "world_coordinate"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or len(value) != 3
                    or any(isinstance(part, bool) or not isinstance(part, int) for part in value)):
                raise FactoryPatchExportError(f"example {name} must be a 3-tuple of integers")
        if not isinstance(self.truth_state, BlockState):
            raise FactoryPatchExportError("example truth_state must be a BlockState")
        patch = np.asarray(self.patch, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        coverage = np.asarray(self.coverage, dtype=np.float32)
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise FactoryPatchExportError(
                f"example patch must be exactly ({PATCH_SIZE}, {PATCH_SIZE}, 3)")
        if valid.shape != (PATCH_SIZE, PATCH_SIZE) or coverage.shape != (PATCH_SIZE, PATCH_SIZE):
            raise FactoryPatchExportError("example valid/coverage masks must match the patch size")
        if not np.isfinite(patch).all() or not np.isfinite(coverage).all():
            raise FactoryPatchExportError("example patch and coverage must be finite")
        for name in ("weight", "area", "depth", "visible", "facing",
                     "valid_fraction", "coverage_mean"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise FactoryPatchExportError(f"example {name} must be finite and non-negative")
        if not 0.0 <= float(self.valid_fraction) <= 1.0:
            raise FactoryPatchExportError("example valid_fraction must lie in [0, 1]")
        if not 0.0 <= float(self.coverage_mean) <= 1.0:
            raise FactoryPatchExportError("example coverage_mean must lie in [0, 1]")
        object.__setattr__(self, "patch", np.ascontiguousarray(patch))
        object.__setattr__(self, "valid", np.ascontiguousarray(valid))
        object.__setattr__(self, "coverage", np.ascontiguousarray(coverage))

    @property
    def key(self) -> tuple[str, str, str, tuple[int, int, int], str]:
        return (self.scene_group, self.scene_id, self.frame_id, self.coordinate, self.slot)

    @property
    def frame_key(self) -> str:
        return f"{self.dataset_id}/{self.scene_id}/{self.frame_id}"

    def to_record(self) -> dict[str, Any]:
        record = {
            "scene_id": self.scene_id,
            "dataset_id": self.dataset_id,
            "scene_group": self.scene_group,
            "fold": self.fold,
            "frame_id": self.frame_id,
            "filename": self.filename,
            "coordinate": [int(value) for value in self.coordinate],
            "world_coordinate": [int(value) for value in self.world_coordinate],
            "slot": self.slot,
            "label": state_to_record(self.truth_state),
            "weight": float(self.weight),
            "area": float(self.area),
            "depth": float(self.depth),
            "visible": float(self.visible),
            "facing": float(self.facing),
            "valid_fraction": float(self.valid_fraction),
            "coverage_mean": float(self.coverage_mean),
            "partial": bool(self.partial),
            "patch": array_record(self.patch),
            "valid": array_record(self.valid),
            "coverage": array_record(self.coverage),
        }
        record["build"] = {
            "staging_token": self.staging_token,
            "world_identity": self.world_identity,
        }
        record["provenance"] = {
            "screenshot_sha256": self.screenshot_sha256,
            "frame_key": self.frame_key,
        }
        record["digests"] = {
            "patch_sha256": record["patch"]["sha256"],
            "valid_sha256": record["valid"]["sha256"],
            "coverage_sha256": record["coverage"]["sha256"],
        }
        record["digests"]["example_sha256"] = sha256_bytes(canonical_bytes(record))
        return record


@dataclass(frozen=True, slots=True)
class FrameDerivation:
    """One admitted frame's frozen patches plus every refusal it produced."""

    examples: tuple[PatchExample, ...]
    refusals: Mapping[str, int]
    sampler_events: Mapping[str, int]
    sample_rejections: Mapping[str, int]
    image_pixels: int
    exact_cells: int
    refused_cells: int
    refused_state_faces: int
    duplicate_faces: int


def _deduplicate(examples: Iterable[PatchExample]) -> tuple[tuple[PatchExample, ...], int]:
    grouped: dict[tuple[str, tuple[int, int, int], str], list[PatchExample]] = {}
    for example in examples:
        grouped.setdefault((example.frame_id, example.coordinate, example.slot), []).append(example)
    kept: list[PatchExample] = []
    duplicates = 0
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) > 1:
            duplicates += len(group) - 1
        kept.append(max(group, key=lambda item: (item.valid_fraction, item.weight, -item.depth)))
    return tuple(kept), duplicates


def derive_frame_examples(
    *,
    scene_id: str,
    dataset_id: str,
    scene_group: str,
    fold: str,
    staging_token: str,
    world_identity: str,
    capture: AdmittedCapture,
    image: np.ndarray,
    scene: SceneTransform,
    reference_blocks: Mapping[tuple[int, int, int], BlockState],
    palette: Sequence[PaletteColor],
    minimum_valid_fraction: float = MINIMUM_PATCH_VALID_FRACTION,
) -> FrameDerivation:
    """Derive one frame's exact labels from reference geometry and recorded camera.

    Only cells whose state carries exact baked polygons are labelled; the rest
    are counted as ``model_metadata_unavailable`` and never emitted through the
    cube fallback. Ownership comes from the same alpha-tested renderer the
    production sampler uses, and sampling runs at production ``raycast.FACE_SHRINK``.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise FactoryPatchExportError("frame image must be an RGB array")
    height, width = image.shape[:2]
    if (width, height) != (capture.width, capture.height):
        raise FactoryPatchExportError(
            f"{capture.frame_id}: decoded image {(width, height)} does not match the "
            f"admitted resolution {(capture.width, capture.height)}")
    normalised = {
        (int(key[0]), int(key[1]), int(key[2])): state for key, state in reference_blocks.items()
    }
    _exact, refused_blocks, modelled = split_exact_states(normalised, palette)
    refusals: dict[str, int] = {}
    if refused_blocks:
        refusals[MODEL_METADATA_UNAVAILABLE] = len(refused_blocks)

    sampler_events: dict[str, int] = {}
    faces = sample_visible_faces(
        set(normalised),
        capture.camera,
        scene,
        image,
        gui_mask=None,
        minimum_area=MINIMUM_FACE_AREA,
        minimum_visible_fraction=float(raycast.MINIMUM_VISIBLE_FRACTION),
        shrink=float(raycast.FACE_SHRINK),
        rejections=sampler_events,
        depth_scale=DEPTH_SCALE,
        blocks=normalised,
        palette=palette,
    )
    accepted: list[PatchExample] = []
    sample_rejections: dict[str, int] = {}
    refused_state_faces = 0
    for face in faces:
        coordinate = tuple(int(value) for value in face.coordinate)
        truth_state = normalised[coordinate]
        if truth_state not in modelled:
            refused_state_faces += 1
            continue
        ok, reason, valid_fraction, coverage_mean = classify_sample(
            face, minimum_valid_fraction=minimum_valid_fraction
        )
        if not ok:
            assert reason is not None
            sample_rejections[reason] = sample_rejections.get(reason, 0) + 1
            continue
        accepted.append(PatchExample(
            scene_id=scene_id,
            dataset_id=dataset_id,
            scene_group=scene_group,
            fold=fold,
            staging_token=staging_token,
            world_identity=world_identity,
            frame_id=capture.frame_id,
            filename=capture.filename,
            coordinate=(coordinate[0], coordinate[1], coordinate[2]),
            world_coordinate=tuple(int(value) for value in scene.to_world(coordinate)),
            slot=str(face.slot),
            truth_state=truth_state,
            weight=observation_weight(face),
            area=float(face.area),
            depth=float(face.depth),
            visible=float(min(1.0, max(0.0, face.visible))),
            facing=float(max(0.0, face.facing)),
            valid_fraction=valid_fraction,
            coverage_mean=coverage_mean,
            partial=valid_fraction < 1.0 - 1e-6,
            screenshot_sha256=capture.screenshot_sha256,
            patch=np.asarray(face.patch, dtype=np.float32),
            valid=np.asarray(face.valid, dtype=bool),
            coverage=np.asarray(face.coverage, dtype=np.float32),
        ))
    kept, duplicates = _deduplicate(accepted)
    return FrameDerivation(
        examples=tuple(kept),
        refusals=dict(refusals),
        sampler_events={str(reason): int(count) for reason, count in sampler_events.items()},
        sample_rejections=dict(sample_rejections),
        image_pixels=int(height * width),
        exact_cells=int(len(_exact)),
        refused_cells=int(len(refused_blocks)),
        refused_state_faces=int(refused_state_faces),
        duplicate_faces=int(duplicates),
    )


# --------------------------------------------------------------------------- #
# Sealing, writing and the artifact
# --------------------------------------------------------------------------- #


def patch_domain_problems(record: Mapping[str, Any]) -> list[str]:
    """Refuse any serialized patch domain that is not exactly 16x16x3 + masks."""
    problems: list[str] = []
    examples = record.get("examples")
    if not isinstance(examples, list):
        return ["artifact has no example list"]
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            problems.append(f"example {index} is not an object")
            continue
        for name, shape, dtype in (
            ("patch", [PATCH_SIZE, PATCH_SIZE, 3], np.dtype(np.float32).str),
            ("valid", [PATCH_SIZE, PATCH_SIZE], np.dtype(bool).str),
            ("coverage", [PATCH_SIZE, PATCH_SIZE], np.dtype(np.float32).str),
        ):
            entry = example.get(name)
            if not isinstance(entry, Mapping):
                problems.append(f"example {index} {name} is missing")
                continue
            if entry.get("shape") != shape or entry.get("dtype") != dtype:
                problems.append(
                    f"example {index} {name} is not the frozen {shape} {dtype} domain")
    return problems


def seal_record(record: Mapping[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key != "artifact_sha256"}
    return sha256_bytes(canonical_bytes(content))


def seal_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(record)
    sealed["artifact_sha256"] = seal_record(sealed)
    return sealed


def write_artifact(path: Path, record: Mapping[str, Any]) -> None:
    """Write the sealed artifact atomically, refusing to overwrite anything."""
    target = Path(path)
    if target.exists():
        raise FactoryPatchExportError(f"refusing to overwrite existing artifact: {target}")
    expected = seal_record(record)
    if record.get("artifact_sha256") != expected:
        raise FactoryPatchExportError("artifact digest does not match its own content")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-directory hard link is an atomic create-if-absent. A
            # check followed by os.replace() still has an overwrite race.
            os.link(temporary, target)
        except FileExistsError as error:
            raise FactoryPatchExportError(
                f"refusing to overwrite existing artifact: {target}") from error
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _safe_output_path(dataset_root: Path, output: Path | str) -> Path:
    """Refuse an output that is unsafe, is a directory, or lives in the dataset."""
    raw = Path(output)
    if any(part == ".." for part in raw.parts):
        raise FactoryPatchExportError(f"refusing unsafe output path {output!r}")
    if raw.exists() and raw.is_dir():
        raise FactoryPatchExportError(f"output path is an existing directory: {raw}")
    resolved = raw.resolve()
    root = dataset_root.resolve()
    if resolved == root or root in resolved.parents:
        raise FactoryPatchExportError(
            f"refusing to write the eval artifact inside the dataset root: {raw}")
    return raw


def _scene_reference(
    scene_root: Path,
    reference_loader,
    provider,
) -> tuple[Mapping[tuple[int, int, int], BlockState], Sequence[PaletteColor]]:
    """Load the hash-pinned reference geometry and build its exact-model palette."""
    reference_path = scene_root / "reference.litematic"
    try:
        snapshot = reference_loader(reference_path)
    except (OSError, ValueError) as error:
        raise FactoryPatchExportError(
            f"could not read reference geometry {reference_path}: {error}") from error
    blocks = {
        (int(coordinate[0]), int(coordinate[1]), int(coordinate[2])): state
        for coordinate, state in snapshot.blocks.items()
    }
    if not blocks:
        raise FactoryPatchExportError(f"reference geometry {reference_path} has no blocks")
    palette = tuple(provider.palette_for_states(tuple(snapshot.palette), include_partial=True))
    if not palette:
        raise FactoryPatchExportError(
            f"exact-model palette is empty for {reference_path}")
    return blocks, palette


def _reference_bounds_problem(
    blocks: Mapping[tuple[int, int, int], BlockState], size: Sequence[int]
) -> str | None:
    for coordinate in blocks:
        if any(not 0 <= coordinate[axis] < int(size[axis]) for axis in range(3)):
            return (
                f"reference block {coordinate} lies outside the region size "
                f"{tuple(int(value) for value in size)}"
            )
    return None


def export_dataset(
    dataset_root: Path | str,
    *,
    output: Path | str,
    jar: Path | str | None = None,
    provider=None,
    image_loader=load_image,
    reference_loader=load_litematic,
    ground_truth_parser=parse_ground_truth,
    max_examples: int = MAX_EXAMPLES,
    minimum_valid_fraction: float = MINIMUM_PATCH_VALID_FRACTION,
) -> dict[str, Any]:
    """Export one finalized factory dataset; never overwrite, never train.

    ``provider``/``jar`` are injection seams for synthetic tests: the default
    path builds a :class:`VanillaJarTextureProvider` and loads the local vanilla
    JAR. ``image_loader`` defaults to the production decoder and is only ever
    called after the screenshot digest has been re-verified.
    """
    root = Path(dataset_root)
    target = _safe_output_path(root, output)
    if target.exists():
        raise FactoryPatchExportError(f"refusing to overwrite existing artifact: {target}")

    entries, index_records = load_finalized_scenes(root)
    if not entries:
        raise FactoryPatchExportError(f"no finalized dataset entries under {root}")

    try:
        group_manifest = build_group_manifest(entries)
    except PairCorrespondenceEvaluationError as error:
        raise FactoryPatchExportError(f"refusing grouped export: {error}") from error
    leakage = tuple(check_split_leakage(entries))
    if leakage:
        raise FactoryPatchExportError("refusing split leakage: " + "; ".join(leakage))

    if provider is None and jar is None:
        raise FactoryPatchExportError(
            "a vanilla JAR (or an injected palette provider) is required for exact-model labels")
    if provider is None:
        provider = VanillaJarTextureProvider()
    jar_record: dict[str, Any] | None = None
    if jar is not None:
        jar_path = Path(jar)
        if not jar_path.is_file():
            raise FactoryPatchExportError(f"vanilla JAR is missing: {jar_path}")
        jar_record = {"path": str(jar_path), "sha256": sha256_file(jar_path)}
        loaded = int(provider.load_jar(jar_path))
        if not loaded:
            raise FactoryPatchExportError(f"vanilla JAR produced no texture samples: {jar_path}")

    ordered = sorted(entries, key=lambda item: (
        item.scene_group, item.scene_id, item.dataset_id, item.directory))

    examples: list[PatchExample] = []
    frame_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    admission_refusals: dict[str, int] = {}

    for entry in ordered:
        scene = index_records[(entry.scene_id, entry.directory)]
        staging_token, world_identity, fold = _scene_provenance(scene)
        scene_root = Path(entry.scene_root)
        capture_payload = _read_json(scene_root / "capture.json")
        ground_truth_payload = _read_json(scene_root / "ground_truth.json")
        try:
            export = ground_truth_parser(ground_truth_payload)
        except DatasetSchemaError as error:
            raise FactoryPatchExportError(
                f"{entry.scene_id}: invalid ground truth: {error}") from error
        region_min = tuple(int(value) for value in export.region.minimum)
        size = tuple(int(value) for value in export.region.size)
        region_max = tuple(int(value) for value in export.region.maximum)
        scene_transform = SceneTransform(region_min, size)

        reference_blocks, palette = _scene_reference(
            scene_root, reference_loader, provider)
        bounds_problem = _reference_bounds_problem(reference_blocks, size)
        if bounds_problem is not None:
            raise FactoryPatchExportError(f"{entry.scene_id}: {bounds_problem}")

        frames = capture_payload.get("frames")
        cameras = capture_payload.get("cameras")
        resolutions = capture_payload.get("resolutions")
        if not (isinstance(frames, list) and isinstance(cameras, list)
                and isinstance(resolutions, list)):
            raise FactoryPatchExportError(f"{entry.scene_id}: malformed capture arrays")
        if not (len(frames) == len(cameras) == len(resolutions)):
            raise FactoryPatchExportError(
                f"{entry.scene_id}: frame/camera/resolution counts differ")
        screenshot_rows = capture_payload.get("screenshots")
        if not isinstance(screenshot_rows, list):
            raise FactoryPatchExportError(f"{entry.scene_id}: screenshots must be a list")
        capture_digests: dict[str, str] = {}
        for row in screenshot_rows:
            if not isinstance(row, Mapping):
                raise FactoryPatchExportError(f"{entry.scene_id}: malformed screenshot row")
            name = str(row.get("filename", ""))
            if name in capture_digests:
                raise FactoryPatchExportError(
                    f"{entry.scene_id}: duplicate screenshot row {name!r}")
            capture_digests[name] = str(row.get("sha256", ""))
        index_digests = dict(entry.screenshots)

        scene_admitted = 0
        scene_refused: list[dict[str, str]] = []
        for position, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise FactoryPatchExportError(f"{entry.scene_id}: malformed frame record")
            filename = str(frame.get("filename", ""))
            index_digest = index_digests.get(filename, "")
            capture_digest = capture_digests.get(filename, "")
            if capture_digest and index_digest and capture_digest != index_digest:
                raise FactoryPatchExportError(
                    f"{entry.scene_id}/{filename}: capture.json digest and index digest differ")
            expected_digest = index_digest or capture_digest
            admitted, reason = admit_capture(
                frame,
                expected_screenshot_sha256=expected_digest,
                fallback_resolution=resolutions[position]
                if isinstance(resolutions[position], (list, tuple)) else None,
            )
            frame_id = str(frame.get("frame_id", ""))
            if admitted is None:
                assert reason is not None
                admission_refusals[reason] = admission_refusals.get(reason, 0) + 1
                scene_refused.append({"frame_id": frame_id, "reason": reason})
                frame_rows.append({
                    "scene_id": entry.scene_id,
                    "dataset_id": entry.dataset_id,
                    "scene_group": entry.scene_group,
                    "frame_id": frame_id,
                    "filename": filename,
                    "admitted": False,
                    "reason": reason,
                })
                continue

            # Re-verify the screenshot bytes immediately before the first decode.
            screenshot_path = scene_root / "captures" / admitted.filename
            if not screenshot_path.is_file():
                raise FactoryPatchExportError(
                    f"{entry.scene_id}/{admitted.filename}: screenshot is missing")
            observed_digest = sha256_file(screenshot_path)
            if observed_digest != admitted.screenshot_sha256:
                raise FactoryPatchExportError(
                    f"{entry.scene_id}/{admitted.filename}: screenshot hash "
                    f"{observed_digest} != pinned {admitted.screenshot_sha256}")
            try:
                image = image_loader(screenshot_path)
            except (OSError, ValueError) as error:
                raise FactoryPatchExportError(
                    f"{entry.scene_id}/{admitted.filename}: could not decode screenshot: {error}"
                ) from error

            derivation = derive_frame_examples(
                scene_id=entry.scene_id,
                dataset_id=entry.dataset_id,
                scene_group=entry.scene_group,
                fold=fold,
                staging_token=staging_token,
                world_identity=world_identity,
                capture=admitted,
                image=image,
                scene=scene_transform,
                reference_blocks=reference_blocks,
                palette=palette,
                minimum_valid_fraction=minimum_valid_fraction,
            )
            for example in derivation.examples:
                world = example.world_coordinate
                if any(not region_min[axis] <= world[axis] <= region_max[axis]
                       for axis in range(3)):
                    raise FactoryPatchExportError(
                        f"{entry.scene_id}/{admitted.filename}: label coordinate {world} "
                        "escapes the region bounds")
            examples.extend(derivation.examples)
            scene_admitted += 1
            frame_rows.append({
                "scene_id": entry.scene_id,
                "dataset_id": entry.dataset_id,
                "scene_group": entry.scene_group,
                "frame_id": admitted.frame_id,
                "filename": admitted.filename,
                "admitted": True,
                "reason": None,
                "camera": {
                    "position": [float(value) for value in admitted.camera.position],
                    "yaw_degrees": float(admitted.camera.yaw_degrees),
                    "pitch_degrees": float(admitted.camera.pitch_degrees),
                    "fov_degrees": float(admitted.camera.fov_degrees),
                    "eye_height": float(admitted.camera.eye_height),
                },
                "resolution": [admitted.width, admitted.height],
                "screenshot_sha256": admitted.screenshot_sha256,
                "examples": len(derivation.examples),
                "refusals": dict(derivation.refusals),
                "sampler_events": dict(derivation.sampler_events),
                "sample_rejections": dict(derivation.sample_rejections),
                "refused_state_faces": derivation.refused_state_faces,
                "duplicate_faces": derivation.duplicate_faces,
            })

        scene_rows.append({
            "scene_id": entry.scene_id,
            "dataset_id": entry.dataset_id,
            "scene_group": entry.scene_group,
            "fold": fold,
            "directory": entry.directory,
            "split": fold,
            "admitted": scene_admitted > 0,
            "admitted_frames": scene_admitted,
            "refused_frames": scene_refused,
            "region_min": [int(value) for value in region_min],
            "region_max": [int(value) for value in region_max],
            "exact_model_states": int(len(exact_model_states(palette))),
            "reference_cells": int(len(reference_blocks)),
        })

    if not examples:
        admitted_frames = sum(1 for row in frame_rows if row.get("admitted"))
        if admitted_frames:
            raise FactoryPatchExportError(
                "refusing an empty label export: captures passed metadata admission, "
                "but no exact-model face samples were visible and usable")
        raise FactoryPatchExportError(
            "refusing an empty export: no capture passed metadata admission")
    if len(examples) > max_examples:
        raise FactoryPatchExportError(
            f"refusing to export {len(examples)} examples above the cap {max_examples}")

    ordered_examples = sorted(examples, key=lambda item: item.key)
    grouping = _grouping_record(entries, index_records, group_manifest, ordered_examples)
    indexes = {
        str(entry.index_path): {
            "dataset_id": entry.dataset_id,
            "index_path": str(entry.index_path),
            "index_sha256": entry.index_sha256,
        }
        for entry in entries
    }
    inputs = {
        "dataset_root": str(root),
        "indexes": [indexes[path] for path in sorted(indexes)],
        "scenes": [
            {
                "scene_id": entry.scene_id,
                "dataset_id": entry.dataset_id,
                "directory": entry.directory,
                "scene_group": entry.scene_group,
                "hashes": dict(entry.hashes),
                "screenshots": [{"filename": name, "sha256": digest}
                                for name, digest in entry.screenshots],
            }
            for entry in ordered
        ],
        "minecraft_jar": jar_record,
        "reference_loader": getattr(reference_loader, "__name__", "injected"),
        "image_loader": getattr(image_loader, "__name__", "injected"),
    }
    record = build_artifact(
        inputs=inputs,
        admission={
            "policy": _admission_policy(),
            "scenes": scene_rows,
            "frames": frame_rows,
            "refusals": dict(sorted(admission_refusals.items())),
        },
        sampler=_sampler_record(minimum_valid_fraction),
        grouping=grouping,
        coverage=_coverage_record(scene_rows, frame_rows, ordered_examples),
        examples=ordered_examples,
    )
    write_artifact(target, record)
    return record


def _admission_policy() -> dict[str, Any]:
    return {
        "finalized_index_only": True,
        "hash_verified_before_decode": True,
        "actual_pose_required": True,
        "planned_pose_never_substituted": True,
        "hud_hidden_required": True,
        "active_resource_packs": [VANILLA_PACK],
        "shader_pack_required_absent": True,
        "unknown_pack_state": "refused",
        "capture_refusals_are_recorded_not_silent": True,
        "capture_refusal_reasons": list(ADMISSION_REASONS),
        "t2_t6_excluded": True,
    }


def _sampler_record(minimum_valid_fraction: float) -> dict[str, Any]:
    return {
        "patch_size": PATCH_SIZE,
        "shrink": float(raycast.FACE_SHRINK),
        "shrink_source": "raycast.FACE_SHRINK",
        "minimum_area": float(MINIMUM_FACE_AREA),
        "minimum_visible_fraction": float(raycast.MINIMUM_VISIBLE_FRACTION),
        "depth_scale": float(DEPTH_SCALE),
        "minimum_patch_valid_fraction": float(minimum_valid_fraction),
        "ownership": "exact_model_alpha_tested_renderer",
        "geometry": "exact_sampling_blocks(reference states in the geometry palette)",
        "geometry_palette": "provider.palette_for_states(reference_states, include_partial=True)",
        "gui_mask": "none (hud_hidden is an admission requirement)",
        "weight_formula": (
            "area**AREA_POWER / (1 + depth/16)"
            " * visible**VISIBILITY_POWER * facing**OBLIQUITY_POWER"
            " / (1 + _face_uniformity(observed_patch, coverage)/UNIFORMITY_SCALE)"
        ),
        "area_power": float(raycast.AREA_POWER),
        "visibility_power": float(raycast.VISIBILITY_POWER),
        "obliquity_power": float(raycast.OBLIQUITY_POWER),
        "uniformity_scale": float(raycast.UNIFORMITY_SCALE),
        "texture_weight": float(raycast.TEXTURE_WEIGHT),
        "label_source": "reference geometry state at the sampler (coordinate, slot)",
        "runtime_coordinates_inferred_from_labels": False,
    }


def _grouping_record(entries, index_records, group_manifest, examples) -> dict[str, Any]:
    split_for_group: dict[str, str] = {}
    for entry in entries:
        scene = index_records[(entry.scene_id, entry.directory)]
        _token, _world, split = _scene_provenance(scene)
        split_for_group.setdefault(entry.scene_group, split)
    counts: dict[str, int] = {}
    scenes_for_group: dict[str, set[str]] = {}
    for example in examples:
        counts[example.scene_group] = counts.get(example.scene_group, 0) + 1
        scenes_for_group.setdefault(example.scene_group, set()).add(example.scene_id)
    groups = []
    for group in sorted(split_for_group):
        groups.append({
            "scene_group": group,
            "fold": split_for_group[group],
            "scene_ids": sorted(scenes_for_group.get(group, set())),
            "example_count": counts.get(group, 0),
        })
    return {
        "grouping_key": "scene_group",
        "fold_key": "index_split",
        "fold_assignment": split_for_group,
        "groups": groups,
        "leakage_issues": list(check_split_leakage(entries)),
        "manifest": group_manifest,
    }


def _coverage_record(scene_rows, frame_rows, examples) -> dict[str, Any]:
    by_scene: dict[str, int] = {}
    by_group: dict[str, int] = {}
    by_slot: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_frame: dict[str, int] = {}
    labeled_cells: set[tuple[str, str, tuple[int, int, int]]] = set()
    for example in examples:
        by_scene[example.scene_id] = by_scene.get(example.scene_id, 0) + 1
        by_group[example.scene_group] = by_group.get(example.scene_group, 0) + 1
        by_slot[example.slot] = by_slot.get(example.slot, 0) + 1
        by_state[example.truth_state.name] = by_state.get(example.truth_state.name, 0) + 1
        by_frame[example.frame_key] = by_frame.get(example.frame_key, 0) + 1
        labeled_cells.add((example.dataset_id, example.scene_id, example.coordinate))
    abstentions: dict[str, int] = {}
    for row in frame_rows:
        for source in (row.get("refusals") or {}, row.get("sample_rejections") or {}):
            for reason, count in source.items():
                if reason == "accepted":
                    continue
                abstentions[reason] = abstentions.get(reason, 0) + int(count)
    return {
        "examples": len(examples),
        "scenes": len({example.scene_id for example in examples}),
        "groups": len({example.scene_group for example in examples}),
        "datasets": sorted({example.dataset_id for example in examples}),
        "labeled_cells": len(labeled_cells),
        "partial_observations": sum(1 for example in examples if example.partial),
        "examples_by_scene": dict(sorted(by_scene.items())),
        "examples_by_group": dict(sorted(by_group.items())),
        "examples_by_slot": dict(sorted(by_slot.items())),
        "examples_by_state": dict(sorted(by_state.items())),
        "examples_by_frame": dict(sorted(by_frame.items())),
        "abstention_reasons": dict(sorted(abstentions.items())),
        "admitted_frames": sum(1 for row in frame_rows if row.get("admitted")),
        "refused_frames": sum(1 for row in frame_rows if not row.get("admitted")),
        "admitted_scenes": sum(1 for row in scene_rows if row.get("admitted")),
        "refused_scenes": sum(1 for row in scene_rows if not row.get("admitted")),
    }


def build_artifact(
    *,
    inputs: Mapping[str, Any],
    admission: Mapping[str, Any],
    sampler: Mapping[str, Any],
    grouping: Mapping[str, Any],
    coverage: Mapping[str, Any],
    examples: Sequence[PatchExample],
) -> dict[str, Any]:
    """Assemble the deterministic, self-sealed factory patch-dataset artifact."""
    ordered = sorted(examples, key=lambda item: item.key)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "engine_version": ENGINE_VERSION,
        "artifact_id": ARTIFACT_ID,
        "mode": MODE,
        "evaluation_only": True,
        "training_permitted": False,
        "t2_t6_excluded": True,
        "inputs": dict(inputs),
        "admission": dict(admission),
        "sampler": dict(sampler),
        "grouping": dict(grouping),
        "coverage": dict(coverage),
        "examples": [example.to_record() for example in ordered],
    }
    problems = patch_domain_problems(record)
    if problems:
        raise FactoryPatchExportError(
            f"refusing to seal a non-frozen patch domain: {problems[0]}")
    return seal_artifact(record)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only E3 factory texture-patch dataset exporter. Reads a "
            "finalized factory dataset root, refuses staging/unverified scenes, "
            "non-vanilla captures, missing actual poses and HUD-visible frames, and "
            "writes one sealed, no-overwrite, evaluation-only artifact."
        )
    )
    parser.add_argument("dataset_root", type=Path,
                        help="finalized factory dataset root (holds index.json)")
    parser.add_argument("output", type=Path,
                        help="sealed artifact path (must not exist)")
    parser.add_argument("--minecraft-jar", type=Path, default=None,
                        help="vanilla 1.21.5 JAR for exact-model geometry")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record = export_dataset(
            args.dataset_root,
            output=args.output,
            jar=args.minecraft_jar or _default_jar_path(),
            max_examples=args.max_examples,
        )
    except FactoryPatchExportError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(f"mode: {record['mode']}")
    print(f"examples: {record['coverage']['examples']} "
          f"scenes: {record['coverage']['scenes']} "
          f"groups: {record['coverage']['groups']}")
    print(f"artifact: {args.output} sha256 {record['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
