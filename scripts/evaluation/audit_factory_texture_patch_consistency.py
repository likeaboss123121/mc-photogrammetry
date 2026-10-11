"""Evaluation-only E4 cross-view factory patch-label consistency audit.

This module is the auditor side of the E4 task
(``texture-poc-e4-cross-view-label-consistency-2026-10-03``). It reads only the
four accepted, sealed E3 factory patch-dataset artifacts under
``exports/texture-poc-e3/*.json`` and measures how consistently the *same* exact
``(BlockState, world coordinate, face slot)`` supervision looks across distinct
captures of the same build.

What it does, exactly:

* Every artifact seal, every per-example seal and every per-array seal is
  recomputed and verified **before** any patch/valid/coverage array is decoded
  into a typed array. A missing field, a drifted digest, a malformed domain or a
  wrong schema is a refusal, never a warning.
* Each example contributes one *observation* keyed by
  ``(scene_group, world_identity, world_coordinate, slot, exact BlockState)``.
  Only keys observed by at least two distinct captures are compared.
* Within a repeated key, identical frames and identical screenshot digests are
  deduplicated deterministically, and same-frame / duplicate-image pair
  equivalents are counted separately from the distinct-capture pairs that feed
  the metric. This is what stops one captured image repeated in two artifacts
  from being counted twice, and what keeps the "distinct capture" claim honest.
* A pair is measured only on the shared valid pixels (the intersection of both
  frozen ownership masks) and only when that overlap reaches the documented
  coverage floor. On that support the auditor reports the raw RGB mean absolute
  error and a scalar-brightness-normalized RGB mean absolute error, plus the
  pair/key/state/slot/group coverage denominators.
* Grouping is strict: ``(scene_group, world_identity)`` is the only pooling
  boundary, so no pair ever mixes two builds.

What it deliberately does not do:

* It makes **no** quality or correctness claim. Repeated-key agreement is a
  consistency proxy only: two captures of a consistently occluded face agree by
  construction even when both are wrong, and region-clipping / occluder
  contamination is invisible to this measurement. Nothing here is fitted to a
  threshold, and no pass/fail bar is defined.
* It never treats a capture as an independent sample: the pair is the unit, and
  duplicate images are suppressed before metrics, never split.
* It reads no raw screenshot, reference, camera manifest, scene definition, JAR,
  T2/T6 fixture or any input outside the named E3 artifacts. It writes no
  production/runtime code and no file outside the ignored report outputs.

This task builds the auditor, its tests and the analysis-only read of the four
named artifacts. It does not train a model, rerun the exporter, or touch
production/runtime code.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # direct-script execution without PYTHONPATH=src
    sys.path.insert(0, str(_SRC))

from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS  # noqa: E402
from mc_photogrammetry.dataset.hashing import canonical_json  # noqa: E402
from mc_photogrammetry.reconstruction.face_sampling import PATCH_SIZE  # noqa: E402

SCHEMA = "mc-photogrammetry-factory-texture-patch-consistency-audit-v1"
"""Schema identity of this audit payload."""

ENGINE_VERSION = "e4-1"
"""Auditor revision recorded in the payload."""

AUDIT_ID = "texture-poc-e4-cross-view-label-consistency"
MODE = "evaluation-only cross-view factory patch-label consistency audit"

E3_SCHEMA = "mc-photogrammetry-factory-texture-patch-dataset-v1"
"""The only E3 artifact schema this auditor admits."""

E3_ENGINE_VERSION = "e3-1"
"""The only E3 exporter revision this auditor admits."""

E3_ARTIFACT_ID = "texture-poc-e3-factory-texture-patch-dataset"

DEFAULT_ARTIFACT_PATHS: tuple[str, ...] = (
    "exports/texture-poc-e3/fresh-c4-canal-town-2026-09-12.json",
    "exports/texture-poc-e3/fresh-c4-canal-town-framed-2026-09-12.json",
    "exports/texture-poc-e3/pilot-batch-train-station-2026-09-12.json",
    "exports/texture-poc-e3/smoke-train-station-ds20c-2026-09-12.json",
)
"""The four accepted E3 artifacts this task names, and nothing else.

They are listed explicitly rather than globbed so a later file dropped into
``exports/texture-poc-e3/`` can never silently join the audited input set.
"""

MINIMUM_SHARED_VALID_PIXELS = 16
"""Coverage floor for a measured pair: at least this many jointly owned pixels.

This is a fixed coverage floor (one sixteenth of the 16x16 patch), not a tuned
or fitted threshold. A pair below it is counted as
``insufficient_shared_valid_pixels`` and contributes no MAE.
"""

BRIGHTNESS_EPSILON = 1e-6
"""At or below this mean the brightness-normalized metric is not defined."""

MAX_ANALYZED_PAIRS = 5_000_000
"""Hard cap so a malformed or adversarial artifact cannot blow up the audit."""

RAW_MAE_UNITS = "native 0-255 float32 RGB patch units"
BRIGHTNESS_NORMALIZATION = (
    "both patches rescaled so their shared-support mean RGB brightness equals "
    "the pair mean, then mean absolute RGB difference on the shared valid pixels"
)

_ARRAY_DOMAINS: dict[str, tuple[str, tuple[int, ...]]] = {
    "patch": (np.dtype(np.float32).str, (PATCH_SIZE, PATCH_SIZE, 3)),
    "valid": (np.dtype(bool).str, (PATCH_SIZE, PATCH_SIZE)),
    "coverage": (np.dtype(np.float32).str, (PATCH_SIZE, PATCH_SIZE)),
}
"""The only serialized array domains E3 can legally emit."""

_EXPECTED_EXAMPLE_DIGEST_KEYS = frozenset(
    {"patch_sha256", "valid_sha256", "coverage_sha256", "example_sha256"}
)

GROUPING_KEY = "scene_group"
FOLD_KEY = "index_split"


class TexturePatchConsistencyError(ValueError):
    """Invalid, unverifiable or unsafe E4 consistency-audit input."""


# --------------------------------------------------------------------------- #
# Primitive serialization and hashing (byte-compatible with the E3 exporter)
# --------------------------------------------------------------------------- #


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes, identical to the E3 exporter's sealing input."""
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def array_record(array: np.ndarray) -> dict[str, Any]:
    """Exact, self-describing bytes for one frozen array (E3-compatible)."""
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(size) for size in value.shape],
        "base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def seal_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Add the E3 per-example digest block to an example record."""
    record = dict(example)
    digests = {
        "patch_sha256": record["patch"]["sha256"],
        "valid_sha256": record["valid"]["sha256"],
        "coverage_sha256": record["coverage"]["sha256"],
    }
    record["digests"] = digests
    digests["example_sha256"] = sha256_bytes(canonical_bytes(record))
    return record


def seal_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add the E3 self-seal to an artifact record."""
    sealed = dict(record)
    sealed["artifact_sha256"] = sha256_bytes(canonical_bytes(sealed))
    return sealed


# --------------------------------------------------------------------------- #
# Seal verification: everything is checked before an array is decoded
# --------------------------------------------------------------------------- #


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TexturePatchConsistencyError(f"{label} must be an object")
    return value


def verify_artifact_seal(record: Mapping[str, Any], *, label: str) -> None:
    """Refuse anything that is not a self-consistent E3 v1 artifact record."""
    if record.get("schema") != E3_SCHEMA:
        raise TexturePatchConsistencyError(
            f"{label}: unsupported artifact schema {record.get('schema')!r}")
    if record.get("engine_version") != E3_ENGINE_VERSION:
        raise TexturePatchConsistencyError(
            f"{label}: unsupported engine_version {record.get('engine_version')!r}")
    if record.get("artifact_id") != E3_ARTIFACT_ID:
        raise TexturePatchConsistencyError(
            f"{label}: unexpected artifact_id {record.get('artifact_id')!r}")
    if record.get("evaluation_only") is not True:
        raise TexturePatchConsistencyError(f"{label}: artifact is not evaluation_only")
    if record.get("training_permitted") is not False:
        raise TexturePatchConsistencyError(f"{label}: artifact permits training")
    if record.get("t2_t6_excluded") is not True:
        raise TexturePatchConsistencyError(f"{label}: artifact does not exclude T2/T6")
    declared = record.get("artifact_sha256")
    if not is_sha256(declared):
        raise TexturePatchConsistencyError(f"{label}: artifact_sha256 is missing or malformed")
    content = {key: value for key, value in record.items() if key != "artifact_sha256"}
    observed = sha256_bytes(canonical_bytes(content))
    if observed != declared:
        raise TexturePatchConsistencyError(
            f"{label}: artifact seal mismatch ({observed} != {declared})")


def verify_array_seal(record: Mapping[str, Any], name: str, *, label: str) -> None:
    """Verify one frozen array's domain and digest without decoding it."""
    expected_dtype, expected_shape = _ARRAY_DOMAINS[name]
    entry = _require_mapping(record.get(name), f"{label} {name} record")
    if entry.get("dtype") != expected_dtype:
        raise TexturePatchConsistencyError(
            f"{label} {name}: dtype {entry.get('dtype')!r} is not the frozen {expected_dtype!r}")
    shape = entry.get("shape")
    if (not isinstance(shape, list) or len(shape) != len(expected_shape)
            or any(type(size) is not int for size in shape)
            or tuple(int(size) for size in shape) != expected_shape):
        raise TexturePatchConsistencyError(
            f"{label} {name}: shape {shape!r} is not the frozen {list(expected_shape)}")
    payload = entry.get("base64")
    if not isinstance(payload, str):
        raise TexturePatchConsistencyError(f"{label} {name}: base64 payload is missing")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise TexturePatchConsistencyError(f"{label} {name}: base64 payload is malformed") from error
    itemsize = np.dtype(expected_dtype).itemsize
    expected_bytes = int(np.prod(expected_shape)) * itemsize
    if len(raw) != expected_bytes:
        raise TexturePatchConsistencyError(
            f"{label} {name}: {len(raw)} bytes do not match the frozen domain")
    digest = sha256_bytes(raw)
    if entry.get("sha256") != digest:
        raise TexturePatchConsistencyError(f"{label} {name}: array digest mismatch")
    digests = record.get("digests")
    if not isinstance(digests, Mapping) or digests.get(f"{name}_sha256") != digest:
        raise TexturePatchConsistencyError(
            f"{label} {name}: per-example digest does not match the array")


def verify_example_seal(record: Mapping[str, Any], *, label: str) -> None:
    """Verify one example's own digest block over its exact serialized content."""
    digests = _require_mapping(record.get("digests"), f"{label} digests")
    if set(digests) != _EXPECTED_EXAMPLE_DIGEST_KEYS:
        raise TexturePatchConsistencyError(
            f"{label}: unexpected digest keys {sorted(digests)}")
    declared = digests.get("example_sha256")
    if not is_sha256(declared):
        raise TexturePatchConsistencyError(f"{label}: example_sha256 is missing or malformed")
    content = dict(record)
    content["digests"] = {
        key: value for key, value in digests.items() if key != "example_sha256"
    }
    observed = sha256_bytes(canonical_bytes(content))
    if observed != declared:
        raise TexturePatchConsistencyError(
            f"{label}: example seal mismatch ({observed} != {declared})")


def _decode_array(record: Mapping[str, Any], name: str) -> np.ndarray:
    """Decode an already-verified array record into a typed array."""
    entry = _require_mapping(record.get(name), f"{name} record")
    raw = base64.b64decode(str(entry["base64"]), validate=True)
    array = np.frombuffer(raw, dtype=np.dtype(str(entry["dtype"])))
    return array.reshape(tuple(int(size) for size in entry["shape"])).copy()


# --------------------------------------------------------------------------- #
# Observation extraction
# --------------------------------------------------------------------------- #


def normalize_state(record: object) -> tuple[str, str]:
    """``(display name, canonical key)`` for one exact BlockState record."""
    if not isinstance(record, Mapping):
        raise TexturePatchConsistencyError("label record must be an object")
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise TexturePatchConsistencyError("label record needs a non-empty name")
    raw_properties = record.get("properties", [])
    if not isinstance(raw_properties, list):
        raise TexturePatchConsistencyError("label record properties must be a list")
    pairs: list[list[str]] = []
    for item in raw_properties:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TexturePatchConsistencyError("label property must be a [key, value] pair")
        key, value = str(item[0]), str(item[1])
        if not key:
            raise TexturePatchConsistencyError("label property key must not be empty")
        pairs.append([key, value])
    pairs.sort()
    return name, canonical_json({"name": name, "properties": pairs})


def _require_non_empty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TexturePatchConsistencyError(f"{label} must be a non-empty string")
    return value


def _require_coordinate(value: object, label: str) -> tuple[int, int, int]:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(part) is not int for part in value)):
        raise TexturePatchConsistencyError(f"{label} must be a 3-list of integers")
    return (int(value[0]), int(value[1]), int(value[2]))


@dataclass(frozen=True, slots=True)
class Observation:
    """One frozen E3 patch, reduced to its audit identity and pixel content."""

    artifact: str
    artifact_sha256: str
    scene_group: str
    world_identity: str
    staging_token: str
    fold: str
    dataset_id: str
    scene_id: str
    frame_key: str
    frame_id: str
    screenshot_sha256: str
    slot: str
    coordinate: tuple[int, int, int]
    world_coordinate: tuple[int, int, int]
    state_name: str
    state_key: str
    valid_fraction: float
    patch: np.ndarray
    valid: np.ndarray

    @property
    def key(self) -> tuple[str, str, tuple[int, int, int], str, str]:
        return (
            self.scene_group,
            self.world_identity,
            self.world_coordinate,
            self.slot,
            self.state_key,
        )

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.scene_group, self.world_identity)


@dataclass(frozen=True, slots=True)
class ExampleFields:
    """Non-array example metadata, validated without decoding any array."""

    scene_group: str
    world_identity: str
    staging_token: str
    fold: str
    dataset_id: str
    scene_id: str
    frame_key: str
    frame_id: str
    screenshot_sha256: str
    slot: str
    coordinate: tuple[int, int, int]
    world_coordinate: tuple[int, int, int]
    state_name: str
    state_key: str


def validate_example_fields(example: Mapping[str, Any], *, label: str) -> ExampleFields:
    """Validate every non-array example field, decoding nothing.

    This is the admission-pass validator used by :func:`load_artifact`: it reads
    only record metadata, so a record that fails here cannot have caused a patch
    decode. Running it over every example before the decode pass is what makes
    the two-phase loading contract real.
    """
    build = _require_mapping(example.get("build"), f"{label} build")
    provenance = _require_mapping(example.get("provenance"), f"{label} provenance")
    staging_token = _require_non_empty_str(build.get("staging_token"), f"{label} staging_token")
    if not is_sha256(staging_token):
        raise TexturePatchConsistencyError(f"{label} staging_token must be a sha256")
    world_identity = _require_non_empty_str(build.get("world_identity"), f"{label} world_identity")
    screenshot = _require_non_empty_str(
        provenance.get("screenshot_sha256"), f"{label} screenshot_sha256")
    if not is_sha256(screenshot):
        raise TexturePatchConsistencyError(f"{label} screenshot_sha256 must be a sha256")
    frame_key = _require_non_empty_str(provenance.get("frame_key"), f"{label} frame_key")
    slot = _require_non_empty_str(example.get("slot"), f"{label} slot")
    if slot not in FACE_SLOTS:
        raise TexturePatchConsistencyError(f"{label} slot {slot!r} is not a known face slot")
    state_name, state_key = normalize_state(example.get("label"))
    return ExampleFields(
        scene_group=_require_non_empty_str(example.get("scene_group"), f"{label} scene_group"),
        world_identity=world_identity,
        staging_token=staging_token.lower(),
        fold=_require_non_empty_str(example.get("fold"), f"{label} fold"),
        dataset_id=_require_non_empty_str(example.get("dataset_id"), f"{label} dataset_id"),
        scene_id=_require_non_empty_str(example.get("scene_id"), f"{label} scene_id"),
        frame_key=frame_key,
        frame_id=_require_non_empty_str(example.get("frame_id"), f"{label} frame_id"),
        screenshot_sha256=screenshot.lower(),
        slot=slot,
        coordinate=_require_coordinate(example.get("coordinate"), f"{label} coordinate"),
        world_coordinate=_require_coordinate(
            example.get("world_coordinate"), f"{label} world_coordinate"),
        state_name=state_name,
        state_key=state_key,
    )


def _observation_from_example(
    example: Mapping[str, Any],
    fields: ExampleFields,
    *,
    artifact: str,
    artifact_sha256: str,
    label: str,
) -> Observation:
    """Decode one already-admitted example's arrays into an :class:`Observation`.

    Only the frozen arrays are touched here. Every metadata invariant was already
    established by :func:`validate_example_fields`, so this is a pure
    decode-and-copy step that never runs before the whole artifact has passed
    admission.
    """
    patch = _decode_array(example, "patch")
    valid = _decode_array(example, "valid").astype(bool, copy=False)
    coverage = _decode_array(example, "coverage")
    if not np.isfinite(patch).all():
        raise TexturePatchConsistencyError(f"{label} patch is not finite")
    if not np.isfinite(coverage).all():
        raise TexturePatchConsistencyError(f"{label} coverage is not finite")
    if not valid.any():
        raise TexturePatchConsistencyError(f"{label} valid mask owns no pixels")

    return Observation(
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        scene_group=fields.scene_group,
        world_identity=fields.world_identity,
        staging_token=fields.staging_token,
        fold=fields.fold,
        dataset_id=fields.dataset_id,
        scene_id=fields.scene_id,
        frame_key=fields.frame_key,
        frame_id=fields.frame_id,
        screenshot_sha256=fields.screenshot_sha256,
        slot=fields.slot,
        coordinate=fields.coordinate,
        world_coordinate=fields.world_coordinate,
        state_name=fields.state_name,
        state_key=fields.state_key,
        valid_fraction=float(valid.mean()),
        patch=np.ascontiguousarray(patch, dtype=np.float64),
        valid=np.ascontiguousarray(valid, dtype=bool),
    )


def _verify_grouping(record: Mapping[str, Any], examples: Sequence[Any], *, label: str) -> None:
    """Cross-check the artifact's declared grouping against its own examples."""
    grouping = _require_mapping(record.get("grouping"), f"{label} grouping")
    if grouping.get("grouping_key") != GROUPING_KEY:
        raise TexturePatchConsistencyError(
            f"{label}: grouping_key {grouping.get('grouping_key')!r} is not {GROUPING_KEY!r}")
    if grouping.get("fold_key") != FOLD_KEY:
        raise TexturePatchConsistencyError(
            f"{label}: fold_key {grouping.get('fold_key')!r} is not {FOLD_KEY!r}")
    fold_assignment = _require_mapping(
        grouping.get("fold_assignment"), f"{label} fold_assignment")
    groups = grouping.get("groups")
    if not isinstance(groups, list):
        raise TexturePatchConsistencyError(f"{label}: grouping.groups must be a list")
    declared_counts: dict[str, int] = {}
    for index, group in enumerate(groups):
        entry = _require_mapping(group, f"{label} grouping.groups[{index}]")
        scene_group = _require_non_empty_str(
            entry.get("scene_group"), f"{label} grouping.groups[{index}].scene_group")
        if type(entry.get("example_count")) is not int or entry["example_count"] < 0:
            raise TexturePatchConsistencyError(
                f"{label} grouping.groups[{index}].example_count is malformed")
        declared_counts[scene_group] = int(entry["example_count"])
    observed_counts: Counter[str] = Counter()
    for index, example in enumerate(examples):
        entry = _require_mapping(example, f"{label} example {index}")
        scene_group = _require_non_empty_str(
            entry.get("scene_group"), f"{label} example {index} scene_group")
        observed_counts[scene_group] += 1
        if scene_group not in fold_assignment:
            raise TexturePatchConsistencyError(
                f"{label}: scene_group {scene_group!r} has no declared fold")
        if entry.get("fold") != fold_assignment[scene_group]:
            raise TexturePatchConsistencyError(
                f"{label}: example fold {entry.get('fold')!r} disagrees with the "
                f"scene_group fold {fold_assignment[scene_group]!r}")
    for scene_group, count in observed_counts.items():
        if declared_counts.get(scene_group) != count:
            raise TexturePatchConsistencyError(
                f"{label}: scene_group {scene_group!r} declares "
                f"{declared_counts.get(scene_group)!r} examples but serializes {count}")
    for scene_group, count in declared_counts.items():
        if count and scene_group not in observed_counts:
            raise TexturePatchConsistencyError(
                f"{label}: scene_group {scene_group!r} declares {count} examples but serializes none")


def load_artifact(path: Path | str) -> tuple[dict[str, Any], list[Observation]]:
    """Verify every seal in one E3 artifact, then decode its observations.

    Loading is strictly two-phase. The artifact seal is checked first, then
    *every* example across the whole artifact has its record validated and its
    per-example and three array seals recomputed, and only after that entire
    admission pass succeeds is any patch decoded. A tampered late example
    therefore refuses before the first NumPy array is built.
    """
    source = Path(path)
    label = source.name or str(source)
    if not source.is_file():
        raise TexturePatchConsistencyError(f"{label}: artifact is missing")
    try:
        raw = source.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TexturePatchConsistencyError(f"{label}: unreadable JSON artifact") from error
    if not isinstance(record, Mapping):
        raise TexturePatchConsistencyError(f"{label}: artifact must be an object")
    verify_artifact_seal(record, label=label)
    examples = record.get("examples")
    if not isinstance(examples, list) or not examples:
        raise TexturePatchConsistencyError(f"{label}: artifact has no examples")
    _verify_grouping(record, examples, label=label)

    file_sha256 = sha256_bytes(raw)
    artifact_sha256 = str(record["artifact_sha256"])

    # Pass 1 - admission. Validate every example record and verify every
    # per-example and per-array seal across the whole artifact. No array is
    # decoded until this entire pass has succeeded, so a tampered later example
    # can never leave a partially decoded observation set behind.
    admitted: list[tuple[Mapping[str, Any], ExampleFields]] = []
    for index, example in enumerate(examples):
        example_label = f"{label} example {index}"
        entry = _require_mapping(example, example_label)
        verify_example_seal(entry, label=example_label)
        for name in _ARRAY_DOMAINS:
            verify_array_seal(entry, name, label=example_label)
        admitted.append((entry, validate_example_fields(entry, label=example_label)))

    # Pass 2 - decode. Only reached once the whole artifact passed admission.
    observations: list[Observation] = []
    for index, (entry, fields) in enumerate(admitted):
        observations.append(_observation_from_example(
            entry, fields, artifact=label, artifact_sha256=artifact_sha256,
            label=f"{label} example {index}"))
    meta = {
        "path": str(source),
        "file_sha256": file_sha256,
        "artifact_sha256": artifact_sha256,
        "schema": str(record["schema"]),
        "engine_version": str(record["engine_version"]),
        "examples": len(examples),
        "scene_groups": sorted({observation.scene_group for observation in observations}),
        "world_identities": sorted({observation.world_identity for observation in observations}),
        "folds": {
            str(scene_group): str(fold)
            for scene_group, fold in sorted(
                _require_mapping(record["grouping"]["fold_assignment"],
                                 f"{label} fold_assignment").items())
        },
    }
    return meta, observations


# --------------------------------------------------------------------------- #
# Pair metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PairMetrics:
    """The measured (or refused) comparison of one distinct-capture pair."""

    shared_valid_pixels: int
    overlap_fraction: float
    brightness_a: float
    brightness_b: float
    raw_mae: float | None
    normalized_mae: float | None
    excluded_reason: str | None


def pair_metrics(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    *,
    minimum_shared_valid_pixels: int = MINIMUM_SHARED_VALID_PIXELS,
) -> PairMetrics:
    """Raw and brightness-normalized RGB MAE on a pair's shared valid pixels.

    The support is the intersection of the two frozen ownership masks, so a
    pixel one capture does not own can never enter the comparison. A pair whose
    shared support is below the floor, or whose shared support is black in
    either image, is refused with a typed reason rather than assigned a
    misleading metric.
    """
    if minimum_shared_valid_pixels < 1:
        raise TexturePatchConsistencyError("minimum_shared_valid_pixels must be at least 1")
    a = np.asarray(patch_a, dtype=np.float64)
    b = np.asarray(patch_b, dtype=np.float64)
    mask_a = np.asarray(valid_a, dtype=bool)
    mask_b = np.asarray(valid_b, dtype=bool)
    for array, label in ((a, "patch_a"), (b, "patch_b")):
        if array.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise TexturePatchConsistencyError(f"{label} must be {PATCH_SIZE}x{PATCH_SIZE}x3")
    for mask, label in ((mask_a, "valid_a"), (mask_b, "valid_b")):
        if mask.shape != (PATCH_SIZE, PATCH_SIZE):
            raise TexturePatchConsistencyError(f"{label} must be {PATCH_SIZE}x{PATCH_SIZE}")
    shared = mask_a & mask_b
    shared_count = int(np.count_nonzero(shared))
    overlap_fraction = shared_count / float(PATCH_SIZE * PATCH_SIZE)
    if shared_count < minimum_shared_valid_pixels:
        return PairMetrics(
            shared_count, overlap_fraction, 0.0, 0.0, None, None,
            "insufficient_shared_valid_pixels")
    pixels_a = a[shared]
    pixels_b = b[shared]
    raw_mae = float(np.mean(np.abs(pixels_a - pixels_b)))
    brightness_a = float(np.mean(pixels_a))
    brightness_b = float(np.mean(pixels_b))
    if brightness_a <= BRIGHTNESS_EPSILON or brightness_b <= BRIGHTNESS_EPSILON:
        return PairMetrics(
            shared_count, overlap_fraction, brightness_a, brightness_b, raw_mae, None,
            "zero_brightness")
    target = 0.5 * (brightness_a + brightness_b)
    normalized = (
        np.abs(pixels_a * (target / brightness_a) - pixels_b * (target / brightness_b))
    )
    return PairMetrics(
        shared_count,
        overlap_fraction,
        brightness_a,
        brightness_b,
        raw_mae,
        float(np.mean(normalized)),
        None,
    )


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    """Deterministic count/min/median/mean/p90/max over a finite value list."""
    if not values:
        return {"count": 0}
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    return {
        "count": int(array.size),
        "min": float(array[0]),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90)),
        "max": float(array[-1]),
    }


def _shared_valid_histogram(counts: Sequence[int]) -> list[dict[str, Any]]:
    """Counts per 16-pixel shared-valid band, from 0 to the full 256 pixels."""
    edges = np.arange(0, PATCH_SIZE * PATCH_SIZE + 1, 16)
    if not counts:
        return [
            {"min": int(edges[index]), "max": int(edges[index + 1] - 1), "count": 0}
            for index in range(len(edges) - 1)
        ]
    observed, _ = np.histogram(np.asarray(list(counts), dtype=np.int64), bins=edges)
    return [
        {"min": int(edges[index]), "max": int(edges[index + 1] - 1),
         "count": int(observed[index])}
        for index in range(len(edges) - 1)
    ]


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def _empty_accounting() -> dict[str, int]:
    return {
        "raw_unordered_pairs": 0,
        "same_frame_pairs": 0,
        "duplicate_image_pairs": 0,
        "distinct_capture_pairs_raw": 0,
        "analyzed_distinct_capture_pairs": 0,
        "suppressed_duplicate_pair_equivalents": 0,
        "excluded_insufficient_shared_valid_pixels": 0,
        "excluded_zero_brightness": 0,
    }


def _pair_record(
    first: Observation,
    second: Observation,
    metrics: PairMetrics,
) -> dict[str, Any]:
    return {
        "scene_group": first.scene_group,
        "world_identity": first.world_identity,
        "world_coordinate": [int(value) for value in first.world_coordinate],
        "slot": first.slot,
        "state": first.state_name,
        "state_key": first.state_key,
        "frame_a": first.frame_key,
        "frame_b": second.frame_key,
        "screenshot_a": first.screenshot_sha256,
        "screenshot_b": second.screenshot_sha256,
        "artifact_a": first.artifact,
        "artifact_b": second.artifact,
        "shared_valid_pixels": int(metrics.shared_valid_pixels),
        "overlap_fraction": float(metrics.overlap_fraction),
        "valid_fraction_a": float(first.valid_fraction),
        "valid_fraction_b": float(second.valid_fraction),
        "brightness_a": float(metrics.brightness_a),
        "brightness_b": float(metrics.brightness_b),
        "raw_rgb_mae": None if metrics.raw_mae is None else float(metrics.raw_mae),
        "brightness_normalized_rgb_mae": (
            None if metrics.normalized_mae is None else float(metrics.normalized_mae)
        ),
        "excluded_reason": metrics.excluded_reason,
    }


def _summarize_strata(
    pairs: Sequence[Mapping[str, Any]],
    key_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair counts and MAE summaries bucketed by the named pair fields."""
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[tuple(pair[name] for name in key_names)].append(pair)
    rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = buckets[key]
        raw = [float(p["raw_rgb_mae"]) for p in group if p["raw_rgb_mae"] is not None]
        normalized = [
            float(p["brightness_normalized_rgb_mae"])
            for p in group
            if p["brightness_normalized_rgb_mae"] is not None
        ]
        row: dict[str, Any] = {name: key[index] for index, name in enumerate(key_names)}
        row.update({
            "attempted_pairs": len(group),
            "analyzed_pairs": len(raw),
            "excluded_insufficient_shared_valid_pixels": sum(
                1 for p in group if p["excluded_reason"] == "insufficient_shared_valid_pixels"),
            "excluded_zero_brightness": sum(
                1 for p in group if p["excluded_reason"] == "zero_brightness"),
            "raw_rgb_mae": _metric_summary(raw),
            "brightness_normalized_rgb_mae": _metric_summary(normalized),
        })
        rows.append(row)
    return rows


def _ordered(group: Sequence[Observation]) -> list[Observation]:
    """Deterministic observation order inside one repeated key."""
    return sorted(group, key=lambda item: (
        item.frame_key, item.screenshot_sha256, item.artifact,
        item.frame_id, item.coordinate,
    ))


def _deduplicate_captures(ordered: Sequence[Observation]) -> list[Observation]:
    """One representative per frame, then one per distinct screenshot digest.

    A repeated frame is a same-frame duplicate; a different frame carrying the
    same screenshot bytes is a duplicate image. Either way only the first
    deterministic representative survives, so neither can manufacture an extra
    "distinct capture" pair.
    """
    representatives: list[Observation] = []
    seen_frames: set[str] = set()
    seen_screenshots: set[str] = set()
    for observation in ordered:
        if observation.frame_key in seen_frames:
            continue
        seen_frames.add(observation.frame_key)
        if observation.screenshot_sha256 in seen_screenshots:
            continue
        seen_screenshots.add(observation.screenshot_sha256)
        representatives.append(observation)
    return representatives


def _group_summary(
    group_key: tuple[str, str],
    observations: Sequence[Observation],
    keys: Mapping[tuple[Any, ...], Sequence[Observation]],
    pairs: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, int],
) -> dict[str, Any]:
    scene_group, world_identity = group_key
    observation_counts = Counter(len(group) for group in keys.values())
    distinct_capture_counts = Counter(
        len(_deduplicate_captures(_ordered(group))) for group in keys.values())
    repeated_keys = sum(1 for group in keys.values() if len(group) >= 2)
    observations_in_repeated_keys = sum(
        len(group) for group in keys.values() if len(group) >= 2)
    keys_with_distinct_captures = sum(
        1 for group in keys.values() if len(_deduplicate_captures(_ordered(group))) >= 2)
    raw = [float(p["raw_rgb_mae"]) for p in pairs if p["raw_rgb_mae"] is not None]
    normalized = [
        float(p["brightness_normalized_rgb_mae"])
        for p in pairs
        if p["brightness_normalized_rgb_mae"] is not None
    ]
    return {
        "scene_group": scene_group,
        "world_identity": world_identity,
        "staging_tokens": sorted({observation.staging_token for observation in observations}),
        "scene_ids": sorted({observation.scene_id for observation in observations}),
        "dataset_ids": sorted({observation.dataset_id for observation in observations}),
        "folds": sorted({observation.fold for observation in observations}),
        "examples": len(observations),
        "unique_keys": len(keys),
        "repeated_keys": repeated_keys,
        "keys_with_distinct_captures": keys_with_distinct_captures,
        "observations_in_repeated_keys": observations_in_repeated_keys,
        "keys_by_observation_count": {
            str(count): observation_counts[count] for count in sorted(observation_counts)
        },
        "keys_by_distinct_capture_count": {
            str(count): distinct_capture_counts[count]
            for count in sorted(distinct_capture_counts)
        },
        "pairs": dict(accounting),
        "shared_valid_pixels": _metric_summary(
            [int(p["shared_valid_pixels"]) for p in pairs]),
        "shared_valid_pixels_histogram": _shared_valid_histogram(
            [int(p["shared_valid_pixels"]) for p in pairs]),
        "patch_valid_fraction": _metric_summary(
            [float(observation.valid_fraction) for observation in observations]),
        "raw_rgb_mae": _metric_summary(raw),
        "brightness_normalized_rgb_mae": _metric_summary(normalized),
        "by_state": _summarize_strata(pairs, ("state", "state_key")),
        "by_slot": _summarize_strata(pairs, ("slot",)),
        "by_state_slot": _summarize_strata(pairs, ("state", "state_key", "slot")),
    }


def audit_loaded(
    artifacts: Sequence[tuple[dict[str, Any], list[Observation]]],
    *,
    minimum_shared_valid_pixels: int = MINIMUM_SHARED_VALID_PIXELS,
) -> dict[str, Any]:
    """Aggregate already seal-verified artifacts into the audit payload."""
    if minimum_shared_valid_pixels < 1:
        raise TexturePatchConsistencyError("minimum_shared_valid_pixels must be at least 1")

    observations = [observation for _meta, loaded in artifacts for observation in loaded]
    if not observations:
        raise TexturePatchConsistencyError("no observations to audit")

    keys: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    for observation in observations:
        keys[observation.key].append(observation)

    accounting = _empty_accounting()
    pairs: list[dict[str, Any]] = []
    keys_with_distinct_captures = 0
    for key in sorted(keys):
        group = keys[key]
        if len(group) < 2:
            continue
        ordered = _ordered(group)
        # Stratify every raw unordered pair before any suppression.
        for first_index in range(len(ordered)):
            for second_index in range(first_index + 1, len(ordered)):
                first = ordered[first_index]
                second = ordered[second_index]
                accounting["raw_unordered_pairs"] += 1
                if first.frame_key == second.frame_key:
                    accounting["same_frame_pairs"] += 1
                elif first.screenshot_sha256 == second.screenshot_sha256:
                    accounting["duplicate_image_pairs"] += 1
                else:
                    accounting["distinct_capture_pairs_raw"] += 1
        representatives = _deduplicate_captures(ordered)
        if len(representatives) >= 2:
            keys_with_distinct_captures += 1
        for first_index in range(len(representatives)):
            for second_index in range(first_index + 1, len(representatives)):
                first = representatives[first_index]
                second = representatives[second_index]
                accounting["analyzed_distinct_capture_pairs"] += 1
                metrics = pair_metrics(
                    first.patch, second.patch, first.valid, second.valid,
                    minimum_shared_valid_pixels=minimum_shared_valid_pixels,
                )
                if metrics.excluded_reason == "insufficient_shared_valid_pixels":
                    accounting["excluded_insufficient_shared_valid_pixels"] += 1
                elif metrics.excluded_reason == "zero_brightness":
                    accounting["excluded_zero_brightness"] += 1
                pairs.append(_pair_record(first, second, metrics))
    repeated_key_count = sum(1 for group in keys.values() if len(group) >= 2)
    accounting["suppressed_duplicate_pair_equivalents"] = (
        accounting["distinct_capture_pairs_raw"]
        - accounting["analyzed_distinct_capture_pairs"]
    )

    if len(pairs) > MAX_ANALYZED_PAIRS:
        raise TexturePatchConsistencyError(
            f"refusing {len(pairs)} analyzed pairs above the cap {MAX_ANALYZED_PAIRS}")

    pairs.sort(key=lambda pair: (
        pair["scene_group"], pair["world_identity"], pair["world_coordinate"],
        pair["slot"], pair["state_key"], pair["frame_a"], pair["frame_b"],
    ))

    by_group_observations: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        by_group_observations[observation.group_key].append(observation)
    by_group_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_group_pairs[(pair["scene_group"], pair["world_identity"])].append(pair)

    groups: list[dict[str, Any]] = []
    for group_key in sorted(by_group_observations):
        group_observations = by_group_observations[group_key]
        group_keys = {observation.key: keys[observation.key] for observation in group_observations}
        group_pairs = by_group_pairs.get(group_key, [])
        group_accounting = _empty_accounting()
        for group in group_keys.values():
            if len(group) < 2:
                continue
            ordered = _ordered(group)
            for first_index in range(len(ordered)):
                for second_index in range(first_index + 1, len(ordered)):
                    first = ordered[first_index]
                    second = ordered[second_index]
                    group_accounting["raw_unordered_pairs"] += 1
                    if first.frame_key == second.frame_key:
                        group_accounting["same_frame_pairs"] += 1
                    elif first.screenshot_sha256 == second.screenshot_sha256:
                        group_accounting["duplicate_image_pairs"] += 1
                    else:
                        group_accounting["distinct_capture_pairs_raw"] += 1
            representatives = _deduplicate_captures(ordered)
            group_accounting["analyzed_distinct_capture_pairs"] += (
                len(representatives) * (len(representatives) - 1) // 2)
        group_accounting["excluded_insufficient_shared_valid_pixels"] = sum(
            1 for p in group_pairs if p["excluded_reason"] == "insufficient_shared_valid_pixels")
        group_accounting["excluded_zero_brightness"] = sum(
            1 for p in group_pairs if p["excluded_reason"] == "zero_brightness")
        group_accounting["suppressed_duplicate_pair_equivalents"] = (
            group_accounting["distinct_capture_pairs_raw"]
            - group_accounting["analyzed_distinct_capture_pairs"]
        )
        groups.append(_group_summary(
            group_key, group_observations, group_keys, group_pairs, group_accounting))

    key_counts = Counter(len(group) for group in keys.values())
    raw_values = [float(p["raw_rgb_mae"]) for p in pairs if p["raw_rgb_mae"] is not None]
    normalized_values = [
        float(p["brightness_normalized_rgb_mae"])
        for p in pairs
        if p["brightness_normalized_rgb_mae"] is not None
    ]
    coverage = {
        "artifacts": len(artifacts),
        "examples": len(observations),
        "unique_frames": len({observation.frame_key for observation in observations}),
        "unique_screenshots": len({observation.screenshot_sha256 for observation in observations}),
        "scene_groups": len({observation.scene_group for observation in observations}),
        "builds": len({observation.group_key for observation in observations}),
        "keys_total": len(keys),
        "single_observation_keys": key_counts.get(1, 0),
        "repeated_keys": repeated_key_count,
        "keys_with_distinct_captures": keys_with_distinct_captures,
        "observations_in_repeated_keys": sum(
            len(group) for group in keys.values() if len(group) >= 2),
        "keys_by_observation_count": {
            str(count): key_counts[count] for count in sorted(key_counts)
        },
        "patch_valid_fraction": _metric_summary(
            [observation.valid_fraction for observation in observations]),
        "shared_valid_pixels": _metric_summary(
            [int(p["shared_valid_pixels"]) for p in pairs]),
        "shared_valid_pixels_histogram": _shared_valid_histogram(
            [int(p["shared_valid_pixels"]) for p in pairs]),
    }

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "engine_version": ENGINE_VERSION,
        "audit_id": AUDIT_ID,
        "mode": MODE,
        "evaluation_only": True,
        "training_permitted": False,
        "quality_claim": "none",
        "consistency_is_not_correctness": True,
        "captures_split_independently": False,
        "parameters": {
            "patch_size": PATCH_SIZE,
            "minimum_shared_valid_pixels": int(minimum_shared_valid_pixels),
            "raw_rgb_mae_units": RAW_MAE_UNITS,
            "brightness_normalization": BRIGHTNESS_NORMALIZATION,
            "brightness_epsilon": BRIGHTNESS_EPSILON,
            "pair_unit": "one unordered pair of distinct deduplicated captures",
        },
        "inputs": [meta for meta, _loaded in artifacts],
        "seal_verification": {
            "artifacts": len(artifacts),
            "artifact_seals_verified": len(artifacts),
            "example_seals_verified": sum(int(meta["examples"]) for meta, _ in artifacts),
            "array_seals_verified": 3 * sum(int(meta["examples"]) for meta, _ in artifacts),
            "refused": 0,
        },
        "coverage": coverage,
        "pairs": dict(accounting),
        "raw_rgb_mae": _metric_summary(raw_values),
        "brightness_normalized_rgb_mae": _metric_summary(normalized_values),
        "by_state": _summarize_strata(pairs, ("state", "state_key")),
        "by_slot": _summarize_strata(pairs, ("slot",)),
        "by_state_slot": _summarize_strata(pairs, ("state", "state_key", "slot")),
        "groups": groups,
        "pair_records": pairs,
        "limitations": [
            "Repeated-key agreement is a consistency proxy, not correctness: two "
            "captures of a consistently occluded or mislabelled face agree by "
            "construction.",
            "Region-clipping and occluder contamination are invisible to this "
            "measurement; a shared wrong region still produces a low MAE.",
            "Only the frozen E3 patch domain is compared; no source pixels, "
            "cameras, scenes or references are read.",
            "No threshold was fitted and no pass/fail quality verdict is defined.",
        ],
    }
    return payload


def audit_paths(
    paths: Sequence[Path | str] = DEFAULT_ARTIFACT_PATHS,
    *,
    minimum_shared_valid_pixels: int = MINIMUM_SHARED_VALID_PIXELS,
) -> dict[str, Any]:
    """Seal-verify every named artifact, then run the consistency audit."""
    resolved = [Path(path) if Path(path).is_absolute() else _REPO_ROOT / path for path in paths]
    if not resolved:
        raise TexturePatchConsistencyError("no E3 artifacts were named")
    loaded = [load_artifact(path) for path in resolved]
    return audit_loaded(loaded, minimum_shared_valid_pixels=minimum_shared_valid_pixels)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only E4 audit of cross-view factory patch-label "
            "consistency. Verifies every E3 seal before decoding a patch, then "
            "reports raw and brightness-normalized RGB MAE over repeated exact "
            "faces. It makes no quality or correctness claim."
        )
    )
    parser.add_argument(
        "artifacts", nargs="*", type=Path, default=None,
        help="sealed E3 artifacts (default: the four named accepted artifacts)")
    parser.add_argument(
        "--minimum-shared-valid-pixels", type=int, default=MINIMUM_SHARED_VALID_PIXELS,
        help="pair coverage floor (default: %(default)s)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional ignored JSON report path (stdout summary is always printed)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = args.artifacts if args.artifacts else list(DEFAULT_ARTIFACT_PATHS)
    try:
        payload = audit_paths(
            paths, minimum_shared_valid_pixels=int(args.minimum_shared_valid_pixels))
    except TexturePatchConsistencyError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    coverage = payload["coverage"]
    print(f"mode: {payload['mode']}")
    print(f"artifacts: {coverage['artifacts']} examples: {coverage['examples']} "
          f"builds: {coverage['builds']}")
    print(f"keys: {coverage['keys_total']} repeated: {coverage['repeated_keys']} "
          f"analyzed pairs: {payload['pairs']['analyzed_distinct_capture_pairs']}")
    print(f"raw RGB MAE: {payload['raw_rgb_mae']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
