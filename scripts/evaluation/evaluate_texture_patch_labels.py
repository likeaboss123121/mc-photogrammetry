"""Evaluation-only E1 visible-face label and production-patch audit harness.

This module derives exact ``(BlockState, block coordinate, face slot)``
supervision from a pinned evaluation fixture (Template 2 or Template 6) and
freezes the exact patches the production sampler yields for those faces. It is
the label/patch half of the E1 experiment described in
``docs/reports/texture-poc-data-audit-2026-10-03.md`` section 6: before any
recognizer is trained, the patch set every later model must be compared against
has to exist, be byte-reproducible, and state its own denominators honestly.

Scope and refusals (all enforced here, none delegated to a caller):

* Only the two reviewed pinned roles in :data:`FIXTURE_ROLES` are admissible.
  The tracked expected-pin manifest is loaded and **every** reference digest,
  ordered frame digest and frozen usable-mask domain digest is verified before
  any image is decoded. A missing, malformed, unreviewed or drifting pin is a
  refusal, never a warning.
* Labels are exact only where the local vanilla JAR supplies exact baked model
  polygons. States without exact model metadata are counted and refused with
  ``model_metadata_unavailable``; they are never silently labelled with the
  cube fallback. The full reference still renders for ownership, so a refused
  cell occludes as its conservative cube instead of being dropped - dropping an
  occluder would silently label pixels behind it. Per-pixel ownership from the
  same alpha-tested renderer the production sampler uses is the only pixel
  authority, so cut-out and partially occluded pixels are excluded by the
  retained ``valid``/``coverage`` masks.
* The artifact stores only the rectified 16x16x3 patch, its valid mask and its
  fractional coverage. No source screenshot pixels, camera image, or unrelated
  frame content are serialized.
* The frozen sample is the reference-assisted exact-model sampler subset. Its
  denominators are reported separately from the historical production
  stage-trace rows (39.26% / 41.30% / 44.27% / 48.54%); this module records the
  denominator/interface gap and does **not** claim the two are identical.
* Geometry and scoring vocabularies are deliberately separate. Exact truth
  geometry and rendering use ``palette_for_states(reference_states,
  include_partial=True)``; scoring uses the canonical full-cube production
  vocabulary ``provider.canonical_palette()``. A truth state with exact geometry
  but no canonical-palette membership is geometry-labelable yet out-of-vocabulary
  for scoring, and is never counted as an oracle candidate.
* The visible-cell denominator is rendered through the *production* palette path
  ``provider.samples``, exactly as ``scripts/benchmark_templates.py`` does, not
  through the reference-restricted ``geometry_palette``. The geometry palette
  remains the authority for face-label ownership and exact-model labelability
  only: a palette built from the reference's own states can resolve a shared block
  name or shape differently and cull faces the production renderer keeps, which
  undercounts visible cells. Tightening the labels must never quietly move the
  denominator.
* The camera manifest, scene definition and local vanilla JAR are pinned by
  immutable SHA-256 in this module and verified, fail-closed, beside the reviewed
  reference/frame/mask pins and before the first screenshot decode.

This task creates the harness only. It does not run either fixture corpus and
does not generate the patch corpus; a lead-authorized E1 run does that.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
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
from mc_photogrammetry.domain.models import BlockState  # noqa: E402
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.export.visibility import (  # noqa: E402
    full_cube_keys,
    shape_table,
    visible_cells,
)
from mc_photogrammetry.input.image import load_image  # noqa: E402
from mc_photogrammetry.reconstruction import raycast  # noqa: E402
from mc_photogrammetry.reconstruction.appearance_scoring import AppearanceScorer  # noqa: E402
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    load_camera_manifest,
)
from mc_photogrammetry.reconstruction.face_observation import (  # noqa: E402
    face_observation_from_sample,
)
from mc_photogrammetry.reconstruction.face_sampling import (  # noqa: E402
    DEFAULT_SHRINK,
    DEPTH_SCALE,
    MINIMUM_FACE_AREA,
    PATCH_SIZE,
    exact_model_states,
    sample_visible_faces,
)
from mc_photogrammetry.reconstruction.scene import (  # noqa: E402
    SceneTransform,
    load_scene_transform,
)

SCHEMA = "mc-photogrammetry-texture-patch-labels-v1"
"""Schema identity of the sealed patch-label artifact."""

PIN_MANIFEST_SCHEMA = 1
"""Accepted ``manifest_schema`` of the tracked evaluator pin manifest."""

MODE = "assisted/supplied-camera evaluation"
"""The only mode this harness reports: supplied cameras, reference-assisted geometry."""

PIN_MANIFEST_PATH = _REPO_ROOT / "docs" / "reports" / "evaluator-fixture-pins.json"
MASK_ROOT = _REPO_ROOT / "docs" / "evaluation_masks"

FIXTURE_ROLES: dict[str, dict[str, str]] = {
    "template-2": {
        "fixture": "templates 2",
        "mask_dir": "templates-2",
        "reference": "Template 2.litematic",
    },
    "template-6": {
        "fixture": "templates 6",
        "mask_dir": "templates-6",
        "reference": "Template 6.litematic",
    },
}
"""The only reviewed fixture roles this harness will admit, and nothing else."""

LOCAL_JAR_PIN_KEY = "local-jar"
"""Expected-pin key of the local vanilla JAR (not a manifest role)."""

EXPECTED_ASSET_PINS: dict[str, dict[str, str]] = {
    "template-2": {
        "cameras": "4f743e4fc478430f608e0e879f73cd4a3dc9ce50c726841d2da897be78be6126",
        "scene": "9620608a2544e9460240c2efd0ff83da354d3ffcbee8c01e61e2d9a05b627881",
    },
    "template-6": {
        "cameras": "836ad62f99c5d960cd98c10cf8f850372c5b571be1edd14ff5935fd89027f70f",
        "scene": "0a5528187e2ed47d1756d1a93dd28feb68c8c695e72aa210f067d07c45704563",
    },
    LOCAL_JAR_PIN_KEY: {
        "jar": "522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11",
    },
}
"""Immutable SHA-256 pins for the E1 camera manifest, scene definition and JAR.

These are supplied-camera evaluation inputs, not reference truth: a mismatch on
any of them changes the projection or the exact-model palette this harness
freezes, so every mismatch is a fail-closed refusal that happens before a single
screenshot byte is decoded. They are deliberately held here rather than in the
tracked reviewed-pin manifest, which owns fixture reference/frame/mask identity
only and is never edited by this task.
"""

ASSET_PIN_LABELS: dict[str, str] = {
    "cameras": "camera manifest",
    "scene": "scene definition",
    "jar": "vanilla 1.21.5 JAR",
}

REQUIRED_PIN_FIELDS = ("reviewed", "fixture", "reference_sha256", "frames", "usable_masks")

TOP_K = (1, 8, 16, 32)
"""Reported ordered-recall depths; K=8 is the production candidate cap."""

MINIMUM_PATCH_VALID_FRACTION = 0.5
"""Below this share of owned patch pixels a face is refused, not labelled."""

MODEL_METADATA_UNAVAILABLE = "model_metadata_unavailable"
"""A reference state the JAR palette does not carry exact baked polygons for."""

REASON_NO_VALID_PIXELS = "no_valid_pixels"
REASON_INSUFFICIENT_VALID_COVERAGE = "insufficient_valid_coverage"

HISTORICAL_TOP_K_ROWS: tuple[dict[str, Any], ...] = (
    {
        "row": "T6 production top-8 set membership",
        "value": "788/2007 (39.26%)",
        "metric": "argpartition_set_membership",
        "denominator": "all-visible 2007",
        "source": "docs/reports/visible-state-stage-attribution-2026-10-03.md",
    },
    {
        "row": "T6 ordered K=8 recall",
        "value": "793/1920 (41.30%)",
        "metric": "ordered_recall",
        "denominator": "truth-in-canonical-palette 1920",
        "source": "docs/reports/visible-state-candidate-depth-2026-10-03.md",
    },
    {
        "row": "T6 ordered K=16 recall",
        "value": "850/1920 (44.27%)",
        "metric": "ordered_recall",
        "denominator": "truth-in-canonical-palette 1920",
        "source": "docs/reports/visible-state-candidate-depth-2026-10-03.md",
    },
    {
        "row": "T6 ordered K=32 recall",
        "value": "932/1920 (48.54%)",
        "metric": "ordered_recall",
        "denominator": "truth-in-canonical-palette 1920",
        "source": "docs/reports/visible-state-candidate-depth-2026-10-03.md",
    },
)

RECONCILIATION_GAP = (
    "The historical 39.26% / 41.30% / 44.27% / 48.54% rows are read post hoc "
    "from a reconstruct_scene_surfaces reference-assisted stage trace whose "
    "denominator is the all-visible production domain (2007 cells) and, for the "
    "ordered rows, the truth-in-canonical-palette domain (1920 cells). The frozen "
    "sample here is the reference-assisted exact-model sampler subset, whose "
    "denominators are computed independently (all_visible / scorer_usable / "
    "geometry_labelable / model_vocabulary). Sample identity and denominator "
    "equality with the historical trace are unproven from these exact patches, so "
    "this harness claims no reconciliation and stops before training."
)


class PatchLabelError(ValueError):
    """Invalid, unverifiable, unpinned or unsafe E1 label-harness input."""


def _default_jar_path() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / ".minecraft" / "versions" / "1.21.5" / "1.21.5.jar"


# --------------------------------------------------------------------------- #
# Primitive serialization and hashing
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes used by every digest in this module."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


# A pin table with a missing role/asset or a mistyped digest must fail at import,
# not at the first corpus run.
for _role_key in (*sorted(FIXTURE_ROLES), LOCAL_JAR_PIN_KEY):
    _declared_pins = EXPECTED_ASSET_PINS.get(_role_key)
    assert _declared_pins, f"asset pins are not declared for {_role_key!r}"
    for _asset_key, _asset_digest in _declared_pins.items():
        assert _asset_key in ASSET_PIN_LABELS, f"unknown asset pin {_role_key}.{_asset_key}"
        assert is_sha256(_asset_digest), f"asset pin {_role_key}.{_asset_key} is not a sha256"
del _role_key, _declared_pins, _asset_key, _asset_digest


def state_to_record(state: BlockState) -> dict[str, Any]:
    return {"name": state.name, "properties": [[key, value] for key, value in state.properties]}


def state_from_record(record: object) -> BlockState:
    """Rehydrate a frozen state record, refusing anything malformed."""
    if not isinstance(record, Mapping):
        raise PatchLabelError("state record must be an object")
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise PatchLabelError("state record needs a non-empty name")
    raw_properties = record.get("properties", [])
    if not isinstance(raw_properties, list):
        raise PatchLabelError("state record properties must be a list")
    pairs: list[tuple[str, str]] = []
    for item in raw_properties:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PatchLabelError("state record property must be a [key, value] pair")
        pairs.append((str(item[0]), str(item[1])))
    try:
        return BlockState(name, tuple(sorted(pairs)))
    except ValueError as error:
        raise PatchLabelError(f"state record is not a valid BlockState: {error}") from error


def array_record(array: np.ndarray) -> dict[str, Any]:
    """Exact, self-describing bytes for one frozen array."""
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(size) for size in value.shape],
        "base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
    }


def array_from_record(record: object, *, label: str = "array") -> np.ndarray:
    if not isinstance(record, Mapping):
        raise PatchLabelError(f"{label} record must be an object")
    try:
        dtype = np.dtype(str(record["dtype"]))
        shape = tuple(int(size) for size in record["shape"])
        raw = base64.b64decode(str(record["base64"]), validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise PatchLabelError(f"{label} record is malformed") from error
    array = np.frombuffer(raw, dtype=dtype)
    expected = int(np.prod(shape)) if shape else 1
    if array.size != expected:
        raise PatchLabelError(f"{label} byte length does not match its shape")
    return array.reshape(shape).copy()


# --------------------------------------------------------------------------- #
# Pins, masks and preflight
# --------------------------------------------------------------------------- #


def load_pin_manifest(path: str | Path = PIN_MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    """Load the tracked expected-pin manifest, trusting nothing malformed.

    A missing, unreadable, non-object or schema-mismatched manifest pins
    nothing. ``manifest_schema`` must be exactly an ``int``: JSON ``true`` and
    ``1.0`` both compare equal to ``1`` but are rejected as untrusted.
    """
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    schema = document.get("manifest_schema")
    if type(schema) is not int or schema != PIN_MANIFEST_SCHEMA:
        return {}
    roles = document.get("roles")
    if not isinstance(roles, dict):
        return {}
    return {str(role): dict(entry) for role, entry in roles.items() if isinstance(entry, dict)}


def expected_frame_entries(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    entries: list[dict[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None
        name, digest = item.get("name"), item.get("sha256")
        if not isinstance(name, str) or not name or name in names or not is_sha256(digest):
            return None
        names.add(name)
        entries.append({"name": name, "sha256": str(digest)})
    return entries


def expected_mask_entries(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    entries: list[dict[str, str]] = []
    frames: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None
        frame, domain = item.get("frame"), item.get("domain_sha256")
        if not isinstance(frame, str) or not frame or frame in frames or not is_sha256(domain):
            return None
        entry = {"frame": frame, "domain_sha256": str(domain)}
        file_digest = item.get("file_sha256")
        if file_digest is not None:
            if not is_sha256(file_digest):
                return None
            entry["file_sha256"] = str(file_digest)
        frames.add(frame)
        entries.append(entry)
    return entries


def expected_masks_match(
    expected: Sequence[Mapping[str, str]],
    observed_domains: Mapping[str, str],
    observed_files: Mapping[str, str],
) -> bool:
    if sorted(str(entry["frame"]) for entry in expected) != sorted(observed_domains):
        return False
    for entry in expected:
        frame = str(entry["frame"])
        if observed_domains.get(frame) != entry["domain_sha256"]:
            return False
        file_digest = entry.get("file_sha256")
        if file_digest is not None and observed_files.get(frame) != file_digest:
            return False
    return True


def verify_role_pins(
    specification: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    reference_sha256: str,
    observed_frames: Sequence[Mapping[str, str]],
    observed_mask_domains: Mapping[str, str],
    observed_mask_files: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """Missing and mismatched reviewed-pin fields for one role, fail-closed."""
    if expected is None:
        return list(REQUIRED_PIN_FIELDS), []
    missing: list[str] = []
    mismatched: list[str] = []
    if expected.get("reviewed") is not True:
        missing.append("reviewed")
    if str(expected.get("fixture") or "") != str(specification.get("fixture") or ""):
        missing.append("fixture")
    expected_reference = str(expected.get("reference_sha256") or "")
    if not is_sha256(expected_reference):
        missing.append("reference_sha256")
    elif expected_reference != reference_sha256:
        mismatched.append("reference_sha256")
    expected_frames = expected_frame_entries(expected.get("frames"))
    if expected_frames is None:
        missing.append("frames")
    elif expected_frames != [dict(entry) for entry in observed_frames]:
        mismatched.append("frames")
    expected_masks = expected_mask_entries(expected.get("usable_masks"))
    if expected_masks is None:
        missing.append("usable_masks")
    elif not expected_masks_match(expected_masks, observed_mask_domains, observed_mask_files):
        mismatched.append("usable_masks")
    return missing, mismatched


def mask_domain_sha256(mask: np.ndarray) -> str:
    """Digest the exact scored boolean domain, not just the mask file bytes."""
    array = np.ascontiguousarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mask_spec_name_for_frame(frame_name: str) -> str:
    """Tracked mask-spec file name for one frozen frame: same stem, ``.json``.

    Pinned frames are named ``<timestamp>.png`` while their tracked specs are
    ``<timestamp>.json``. The spec is therefore the frame's stem plus ``.json``,
    never the frame name plus ``.json`` (which would request
    ``<timestamp>.png.json`` and refuse every pinned frame). ``Path.stem`` is used
    so a timestamp's own dots are preserved rather than truncated at the first.
    """
    stem = Path(str(frame_name)).stem
    if not stem or stem == str(frame_name):
        raise PatchLabelError(f"frozen frame name has no .png stem: {frame_name!r}")
    return f"{stem}.json"


def load_mask_spec(path: str | Path) -> np.ndarray:
    """Load a tracked frozen JSON mask spec as a ``True == usable`` domain.

    The specification fixes an integer image size plus excluded rectangles and
    polygons, so the domain does not depend on the mutable production GUI-mask
    heuristic. Out-of-bounds or malformed geometry is a refusal.
    """
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise PatchLabelError(f"frozen mask spec must be tracked JSON: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PatchLabelError(f"frozen mask spec could not be read: {source}") from error
    if not isinstance(document, Mapping):
        raise PatchLabelError(f"frozen mask spec is not an object: {source}")
    size = document.get("image_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise PatchLabelError(f"frozen mask spec image_size is invalid: {source}")
    width, height = int(size[0]), int(size[1])
    mask = np.ones((height, width), dtype=bool)
    rectangles = document.get("excluded_rectangles", [])
    if not isinstance(rectangles, list):
        raise PatchLabelError(f"frozen mask spec excluded_rectangles is invalid: {source}")
    for rectangle in rectangles:
        if not isinstance(rectangle, list) or len(rectangle) != 4 or any(
            type(value) is not int for value in rectangle
        ):
            raise PatchLabelError(f"frozen mask spec has a malformed rectangle: {source}")
        x0, y0, x1, y1 = (int(value) for value in rectangle)
        if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
            raise PatchLabelError(f"frozen mask spec has an out-of-bounds rectangle: {source}")
        mask[y0:y1, x0:x1] = False
    polygons = document.get("excluded_polygons", [])
    if not isinstance(polygons, list):
        raise PatchLabelError(f"frozen mask spec excluded_polygons is invalid: {source}")
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise PatchLabelError(f"frozen mask spec has a malformed polygon: {source}")
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise PatchLabelError(f"frozen mask spec has a malformed polygon: {source}")
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise PatchLabelError(f"frozen mask spec has an out-of-bounds polygon: {source}")
        import cv2

        cv2.fillPoly(mask, [points], False)
    return mask


def verify_asset_pins(
    role: str,
    observed: Mapping[str, str],
    expected_pins: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Mismatched or missing immutable E1 asset pins for one role, fail-closed.

    The camera manifest, scene definition and local JAR are declared here, not
    read from the reviewed fixture manifest. A role or asset with no declared pin
    is itself a refusal: an unpinned input can never be silently accepted.
    """
    expected = expected_pins.get(role)
    if expected is None:
        return [f"role {role!r} asset pins"]
    problems: list[str] = []
    for asset, digest in expected.items():
        label = ASSET_PIN_LABELS.get(asset, asset)
        wanted = str(digest)
        if not is_sha256(wanted):
            problems.append(f"{role}.{asset} ({label}) is not a valid sha256")
        elif observed.get(asset) != wanted:
            problems.append(f"{role}.{asset} ({label})")
    jar_expected = expected_pins.get(LOCAL_JAR_PIN_KEY, {}).get("jar")
    if jar_expected is None:
        problems.append(f"{LOCAL_JAR_PIN_KEY}.jar (vanilla 1.21.5 JAR) is not pinned")
    elif not is_sha256(str(jar_expected)) or observed.get("jar") != str(jar_expected):
        problems.append(f"{LOCAL_JAR_PIN_KEY}.jar (vanilla 1.21.5 JAR)")
    return problems


@dataclass(frozen=True, slots=True)
class PinVerification:
    """The fail-closed reviewed-pin verdict bound into the sealed artifact."""

    role: str
    fixture: str
    reviewed: bool
    pinned: bool
    reference_sha256: str
    frames: tuple[tuple[str, str], ...]
    usable_masks: tuple[tuple[str, str, str | None], ...]
    missing_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fixture": self.fixture,
            "reviewed": self.reviewed,
            "pinned": self.pinned,
            "reference_sha256": self.reference_sha256,
            "frames": [{"name": name, "sha256": digest} for name, digest in self.frames],
            "usable_masks": [
                {"frame": frame, "domain_sha256": domain, "file_sha256": file_digest}
                for frame, domain, file_digest in self.usable_masks
            ],
            "missing_fields": list(self.missing_fields),
            "mismatched_fields": list(self.mismatched_fields),
            "pin_manifest": str(PIN_MANIFEST_PATH),
            "pin_manifest_schema": PIN_MANIFEST_SCHEMA,
        }


@dataclass(frozen=True, slots=True)
class Preflight:
    """Everything verified before a single screenshot byte is decoded."""

    role: str
    specification: Mapping[str, Any]
    fixture_dir: Path
    reference_path: Path
    camera_manifest: Path
    scene_definition: Path
    jar_path: Path
    frames: tuple[str, ...]
    reference_sha256: str
    frame_sha256: Mapping[str, str]
    mask_file_sha256: Mapping[str, str]
    mask_paths: Mapping[str, Path]
    masks: Mapping[str, np.ndarray]
    pin: PinVerification
    asset_sha256: Mapping[str, str]
    asset_pin_problems: tuple[str, ...]


def preflight(
    role: str,
    *,
    fixture_dir: str | Path | None = None,
    reference: str | Path | None = None,
    camera_manifest: str | Path | None = None,
    scene: str | Path | None = None,
    jar: str | Path | None = None,
    mask_root: str | Path | None = None,
    pin_manifest: Mapping[str, Any] | None = None,
    expected_asset_pins: Mapping[str, Mapping[str, str]] | None = None,
) -> Preflight:
    """Verify fixture admission, every pin and every mask domain, or refuse.

    Runs to completion before any image is decoded and before any output path is
    touched, so a pin or mask failure can never leave a partial artifact behind.
    The immutable camera/scene/JAR asset pins are verified here too, beside the
    reviewed reference/frame/mask pins and still before the first decode.
    """
    specification = FIXTURE_ROLES.get(role)
    if specification is None:
        raise PatchLabelError(
            f"role {role!r} is not a reviewed E1 fixture role; admitted roles are "
            f"{sorted(FIXTURE_ROLES)}"
        )
    fixture_dir = Path(fixture_dir) if fixture_dir is not None else _REPO_ROOT / specification["fixture"]
    reference = Path(reference) if reference is not None else fixture_dir / specification["reference"]
    camera_manifest = Path(camera_manifest) if camera_manifest is not None else fixture_dir / "cameras.json"
    scene = Path(scene) if scene is not None else fixture_dir / "scene.json"
    jar = Path(jar) if jar is not None else _default_jar_path()
    mask_root = Path(mask_root) if mask_root is not None else MASK_ROOT

    for path, label, kind in (
        (fixture_dir, "fixture directory", "dir"),
        (reference, "reference litematic", "file"),
        (camera_manifest, "camera manifest", "file"),
        (scene, "scene definition", "file"),
        (jar, "vanilla 1.21.5 JAR", "file"),
    ):
        present = path.is_dir() if kind == "dir" else path.is_file()
        if not present:
            raise PatchLabelError(f"{label} is missing: {path}")

    # The immutable camera/scene/JAR pins gate the projection and the exact-model
    # palette, so they are checked before any mask is parsed, before frame digests
    # are computed and long before the first screenshot decode.
    asset_sha256 = {
        "cameras": sha256_file(camera_manifest),
        "scene": sha256_file(scene),
        "jar": sha256_file(jar),
    }
    pins = expected_asset_pins if expected_asset_pins is not None else EXPECTED_ASSET_PINS
    asset_pin_problems = tuple(verify_asset_pins(role, asset_sha256, pins))
    if asset_pin_problems:
        raise PatchLabelError(
            f"role {role!r} fails immutable asset pins "
            f"({'; '.join(asset_pin_problems)}); refusing before any image processing"
        )

    frames = tuple(sorted(path.name for path in fixture_dir.glob("*.png")))
    if not frames:
        raise PatchLabelError(f"fixture directory has no PNG frames: {fixture_dir}")

    reference_sha256 = sha256_file(reference)
    frame_sha256 = {frame: sha256_file(fixture_dir / frame) for frame in frames}

    mask_dir = mask_root / specification["mask_dir"]
    masks: dict[str, np.ndarray] = {}
    mask_paths: dict[str, Path] = {}
    mask_file_sha256: dict[str, str] = {}
    for frame in frames:
        mask_path = mask_dir / mask_spec_name_for_frame(frame)
        if not mask_path.is_file():
            raise PatchLabelError(f"frozen mask spec is missing for {frame}: {mask_path}")
        masks[frame] = load_mask_spec(mask_path)
        mask_paths[frame] = mask_path
        mask_file_sha256[frame] = sha256_file(mask_path)

    observed_frames = [{"name": frame, "sha256": frame_sha256[frame]} for frame in frames]
    observed_domains = {frame: mask_domain_sha256(mask) for frame, mask in masks.items()}

    manifest = dict(pin_manifest) if pin_manifest is not None else load_pin_manifest()
    expected = manifest.get(role)
    missing, mismatched = verify_role_pins(
        specification, expected, reference_sha256, observed_frames, observed_domains,
        mask_file_sha256,
    )
    pin = PinVerification(
        role=role,
        fixture=str(specification["fixture"]),
        reviewed=bool(isinstance(expected, Mapping) and expected.get("reviewed") is True),
        pinned=not missing and not mismatched,
        reference_sha256=reference_sha256,
        frames=tuple((frame, frame_sha256[frame]) for frame in frames),
        usable_masks=tuple(
            (frame, observed_domains[frame], mask_file_sha256[frame]) for frame in frames
        ),
        missing_fields=tuple(missing),
        mismatched_fields=tuple(mismatched),
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing pin fields: {', '.join(missing)}")
        if mismatched:
            details.append(f"mismatched pin fields: {', '.join(mismatched)}")
        raise PatchLabelError(
            f"role {role!r} is not pinned by the reviewed manifest ({'; '.join(details)}); "
            "refusing before any image processing"
        )
    return Preflight(
        role=role,
        specification=specification,
        fixture_dir=fixture_dir,
        reference_path=reference,
        camera_manifest=camera_manifest,
        scene_definition=scene,
        jar_path=jar,
        frames=frames,
        reference_sha256=reference_sha256,
        frame_sha256=frame_sha256,
        mask_file_sha256=mask_file_sha256,
        mask_paths=mask_paths,
        masks=masks,
        pin=pin,
        asset_sha256=asset_sha256,
        asset_pin_problems=asset_pin_problems,
    )


# --------------------------------------------------------------------------- #
# Exact-state label derivation
# --------------------------------------------------------------------------- #


def palette_states(palette: Sequence[PaletteColor]) -> frozenset[BlockState]:
    """The exact states a palette can nominate, for vocabulary membership."""
    return frozenset(item.state for item in palette)


def split_exact_states(
    blocks: Mapping[tuple[int, int, int], BlockState],
    palette: Sequence[PaletteColor],
) -> tuple[dict[tuple[int, int, int], BlockState], dict[tuple[int, int, int], BlockState], frozenset]:
    """Partition reference cells into exact-model and refused, nothing invented.

    Only states carrying exact baked model polygons in this palette are eligible
    for an exact label. A state without them is returned in the refused mapping
    so the caller records ``model_metadata_unavailable``; it is never sampled
    through the cube fallback and silently labelled. This is the *geometry*
    palette (``palette_for_states(reference_states, include_partial=True)``) and
    says nothing about scoring vocabulary: exact geometry presence is not the
    same as membership in the canonical scoring palette.
    """
    modelled = exact_model_states(palette)
    exact: dict[tuple[int, int, int], BlockState] = {}
    refused: dict[tuple[int, int, int], BlockState] = {}
    for coordinate, state in blocks.items():
        key = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
        (exact if state in modelled else refused)[key] = state
    return exact, refused, modelled


def production_visibility_palette(provider: VanillaJarTextureProvider) -> Sequence[Any]:
    """The palette the production visible-cell domain is rendered through.

    ``scripts/benchmark_templates.py`` passes the raw extraction table
    (``provider.samples``) to ``visible_cells``. It is deliberately **not** a
    palette restricted to the reference's own states: ``full_cube_keys`` adds a
    bare block-name fallback only when *every* extracted state carrying that name
    is a full cube, and the ``face_colours``/``face_shapes``/``face_occluding_slots``
    tables the renderer consumes are resolved from the whole table. A
    reference-restricted palette can therefore classify a name the other way and
    cull faces the production renderer keeps, which undercounts visible cells.

    The exact reference-state ``geometry_palette`` stays the authority for
    face-label ownership and exact-model labelability; it must never set the
    denominator. An empty sample table is a refusal, never a silently empty
    domain.
    """
    samples = getattr(provider, "samples", None)
    if not samples:
        raise PatchLabelError(
            "vanilla JAR produced no texture samples for the production visibility domain"
        )
    return samples


def visibility_domain_arguments(palette: Sequence[Any]) -> dict[str, Any]:
    """The exact keyword payload the production benchmark passes to ``visible_cells``.

    Bundled so the E1 harness cannot drift from the benchmark call site: the
    ``full_cubes``, ``shapes`` and ``palette`` arguments must always be built from
    the *same* palette object.
    """
    return {
        "full_cubes": full_cube_keys(palette),
        "shapes": shape_table(palette),
        "palette": palette,
    }


def _sample_valid(sample: object) -> np.ndarray | None:
    valid = getattr(sample, "valid", None)
    return None if valid is None else np.asarray(valid, dtype=bool)


def _observed_patch(sample: object, valid: np.ndarray | None) -> np.ndarray:
    """The exact patch ``raycast`` hands to ``_face_uniformity``.

    ``FaceSample.observed_patch`` replaces every pixel the sample does not own
    with the face's own coverage-weighted mean colour, so a cut-out or occluded
    region cannot manufacture a quadrant gap. Reproducing that here (rather than
    passing the raw patch) is what keeps the weight identical to production.
    """
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
    """The exact production observation weight for one accepted face sample.

    Byte-for-byte the ``raycast.reconstruct_scene_surfaces`` formula:

    ``weight = area**AREA_POWER / (1 + depth/16)``
    ``weight *= visible**VISIBILITY_POWER`` when the power is non-zero
    ``weight *= facing**OBLIQUITY_POWER`` when the power is non-zero
    ``weight /= 1 + _face_uniformity(observed_patch, coverage) / UNIFORMITY_SCALE``

    The uniformity divisor is included exactly as production applies it. With the
    production ``UNIFORMITY_SCALE`` of 1e9 this harness still records the exact
    applied value: it is not assumed to be one.
    """
    valid = _sample_valid(sample)
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
    """Accept a sampled face as a label, or return a typed refusal reason.

    ``valid_fraction`` is the share of the rectified patch owned by this exact
    face; pixels excluded by alpha cut-out, occlusion or the HUD mask are not
    labelled. A face below ``minimum_valid_fraction`` is refused outright rather
    than frozen as a mostly-unobserved patch.
    """
    if minimum_valid_fraction <= 0.0 or minimum_valid_fraction > 1.0:
        raise PatchLabelError("minimum_valid_fraction must be within (0, 1]")
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
        if coverage is None:
            coverage_mean = 1.0
        else:
            coverage_mean = float(np.mean(np.asarray(coverage, dtype=np.float64)[valid]))
    if valid_fraction < minimum_valid_fraction:
        return False, REASON_INSUFFICIENT_VALID_COVERAGE, valid_fraction, coverage_mean
    return True, None, valid_fraction, coverage_mean


@dataclass(frozen=True, slots=True)
class PatchSample:
    """One frozen production-sampler patch and its exact reference-derived label."""

    frame_id: str
    coordinate: tuple[int, int, int]
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
    patch: np.ndarray
    valid: np.ndarray
    coverage: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise PatchLabelError("patch frame_id must be a non-empty string")
        coordinate = self.coordinate
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
        ):
            raise PatchLabelError("patch coordinate must be a 3-tuple of integers")
        if self.slot not in FACE_SLOTS:
            raise PatchLabelError(f"patch slot must be one of {FACE_SLOTS}")
        if not isinstance(self.truth_state, BlockState):
            raise PatchLabelError("patch truth_state must be a BlockState")
        patch = np.asarray(self.patch, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        coverage = np.asarray(self.coverage, dtype=np.float32)
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise PatchLabelError(
                f"patch must be exactly ({PATCH_SIZE}, {PATCH_SIZE}, 3); refusing unrelated image pixels"
            )
        if valid.shape != (PATCH_SIZE, PATCH_SIZE) or coverage.shape != (PATCH_SIZE, PATCH_SIZE):
            raise PatchLabelError("patch valid/coverage masks must match the patch size")
        if not np.isfinite(patch).all() or not np.isfinite(coverage).all():
            raise PatchLabelError("patch and coverage must be finite")
        for name in ("weight", "area", "depth", "visible", "facing", "valid_fraction", "coverage_mean"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise PatchLabelError(f"patch {name} must be finite and non-negative")
        if not 0.0 <= float(self.valid_fraction) <= 1.0 or not 0.0 <= float(self.coverage_mean) <= 1.0:
            raise PatchLabelError("patch valid/coverage fractions must lie in [0, 1]")
        object.__setattr__(self, "patch", np.ascontiguousarray(patch))
        object.__setattr__(self, "valid", np.ascontiguousarray(valid))
        object.__setattr__(self, "coverage", np.ascontiguousarray(coverage))

    @property
    def key(self) -> tuple[str, tuple[int, int, int], str]:
        return (self.frame_id, self.coordinate, self.slot)

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "coordinate": [int(value) for value in self.coordinate],
            "slot": self.slot,
            "truth_state": state_to_record(self.truth_state),
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

    @classmethod
    def from_record(cls, record: object) -> "PatchSample":
        if not isinstance(record, Mapping):
            raise PatchLabelError("patch record must be an object")
        coordinate = record.get("coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 3:
            raise PatchLabelError("patch record coordinate must be a 3-list")
        return cls(
            frame_id=str(record.get("frame_id", "")),
            coordinate=(int(coordinate[0]), int(coordinate[1]), int(coordinate[2])),
            slot=str(record.get("slot", "")),
            truth_state=state_from_record(record.get("truth_state")),
            weight=float(record.get("weight", 0.0)),
            area=float(record.get("area", 0.0)),
            depth=float(record.get("depth", 0.0)),
            visible=float(record.get("visible", 0.0)),
            facing=float(record.get("facing", 0.0)),
            valid_fraction=float(record.get("valid_fraction", 0.0)),
            coverage_mean=float(record.get("coverage_mean", 0.0)),
            partial=bool(record.get("partial", False)),
            patch=array_from_record(record.get("patch"), label="patch"),
            valid=array_from_record(record.get("valid"), label="valid").astype(bool),
            coverage=array_from_record(record.get("coverage"), label="coverage").astype(np.float32),
        )


def deduplicate_samples(
    samples: Iterable[PatchSample],
) -> tuple[tuple[PatchSample, ...], int]:
    """One deterministic patch per ``(frame, coordinate, slot)``, stably ordered.

    A multipart model can enumerate more than one polygon on the same cell face.
    Keeping the largest, most fully-observed one avoids an arbitrary last-writer
    label while recording how many duplicates were collapsed.
    """
    grouped: dict[tuple[str, tuple[int, int, int], str], list[PatchSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.key, []).append(sample)
    kept: list[PatchSample] = []
    duplicates = 0
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) > 1:
            duplicates += len(group) - 1
        kept.append(
            max(group, key=lambda item: (item.valid_fraction, item.weight, -item.depth))
        )
    return tuple(kept), duplicates


@dataclass(frozen=True, slots=True)
class FrameLabelResult:
    """One frame's frozen patches plus every refusal it produced."""

    frame_id: str
    samples: tuple[PatchSample, ...]
    refusals: Mapping[str, int]
    sampler_events: Mapping[str, int]
    sample_rejections: Mapping[str, int]
    image_pixels: int
    usable_pixels: int
    exact_cells: int
    refused_cells: int
    refused_state_faces: int
    duplicate_faces: int


def derive_frame_labels(
    frame_id: str,
    image: np.ndarray,
    camera: CameraModel,
    scene: SceneTransform,
    blocks: Mapping[tuple[int, int, int], BlockState],
    palette: Sequence[PaletteColor],
    usable_mask: np.ndarray | None,
    *,
    minimum_area: float = MINIMUM_FACE_AREA,
    minimum_visible_fraction: float = 0.6,
    shrink: float = DEFAULT_SHRINK,
    depth_scale: float = DEPTH_SCALE,
    minimum_valid_fraction: float = MINIMUM_PATCH_VALID_FRACTION,
) -> FrameLabelResult:
    """Derive this frame's exact labels from reference truth and camera ownership.

    Only cells whose state carries exact baked polygons are labelled; the rest
    are counted as ``model_metadata_unavailable`` and never emitted through the
    cube fallback. The full reference still renders for ownership, so a cell
    without exact metadata occludes as its conservative cube rather than being
    dropped - a dropped occluder would silently label pixels behind it. Ownership
    comes from the same alpha-tested renderer the production sampler uses, so
    partial models, cut-outs and occluded pixels are excluded by the retained
    masks rather than averaged in.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise PatchLabelError("frame image must be an RGB array")
    height, width = image.shape[:2]
    normalised = {
        (int(key[0]), int(key[1]), int(key[2])): state for key, state in blocks.items()
    }
    exact_blocks, refused_blocks, modelled = split_exact_states(normalised, palette)
    refusals: dict[str, int] = {}
    if refused_blocks:
        refusals[MODEL_METADATA_UNAVAILABLE] = len(refused_blocks)

    gui_mask = None
    if usable_mask is not None:
        mask = np.asarray(usable_mask, dtype=bool)
        if mask.shape != (height, width):
            raise PatchLabelError(
                f"usable mask for {frame_id} has shape {mask.shape}, expected {(height, width)}"
            )
        gui_mask = ~mask

    sampler_events: dict[str, int] = {}
    faces = sample_visible_faces(
        set(normalised),
        camera,
        scene,
        image,
        gui_mask=gui_mask,
        minimum_area=minimum_area,
        minimum_visible_fraction=minimum_visible_fraction,
        shrink=shrink,
        rejections=sampler_events,
        depth_scale=depth_scale,
        blocks=normalised,
        palette=palette,
    )

    accepted: list[PatchSample] = []
    sample_rejections: dict[str, int] = {}
    refused_state_faces = 0
    for face in faces:
        coordinate = tuple(int(value) for value in face.coordinate)
        truth_state = normalised[coordinate]
        if truth_state not in modelled:
            # Enumerated through the conservative cube fallback for occlusion
            # only; refusing here is what keeps it out of the label set.
            refused_state_faces += 1
            continue
        ok, reason, valid_fraction, coverage_mean = classify_sample(
            face, minimum_valid_fraction=minimum_valid_fraction
        )
        if not ok:
            assert reason is not None
            sample_rejections[reason] = sample_rejections.get(reason, 0) + 1
            continue
        accepted.append(
            PatchSample(
                frame_id=frame_id,
                coordinate=(coordinate[0], coordinate[1], coordinate[2]),
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
                patch=np.asarray(face.patch, dtype=np.float32),
                valid=np.asarray(face.valid, dtype=bool),
                coverage=np.asarray(face.coverage, dtype=np.float32),
            )
        )
    kept, duplicates = deduplicate_samples(accepted)
    usable_pixels = int(np.count_nonzero(usable_mask)) if usable_mask is not None else int(height * width)
    return FrameLabelResult(
        frame_id=frame_id,
        samples=kept,
        refusals=dict(refusals),
        sampler_events={str(reason): int(count) for reason, count in sampler_events.items()},
        sample_rejections=dict(sample_rejections),
        image_pixels=int(height * width),
        usable_pixels=usable_pixels,
        exact_cells=len(exact_blocks),
        refused_cells=len(refused_blocks),
        refused_state_faces=int(refused_state_faces),
        duplicate_faces=int(duplicates),
    )


# --------------------------------------------------------------------------- #
# Existing-scorer read over the frozen patches
# --------------------------------------------------------------------------- #


def score_patches(
    patches: Sequence[PatchSample],
    palette: Sequence[PaletteColor],
    *,
    texture_weight: float | None = None,
) -> dict[str, Any]:
    """Reference the existing appearance scorer on the frozen patches.

    ``palette`` here must be the *canonical scoring palette*
    (``provider.canonical_palette()``, the same full-cube production vocabulary
    the pinned benchmark scores against), never a palette built from reference
    truth. A truth state that is not a member of that scoring palette is counted
    as out-of-vocabulary and can never be an oracle candidate: the patch is not
    given a rank for it, so it cannot enter ``truth_in_vocabulary_patches`` or
    any top-K match. Membership is tested on the palette itself before the scorer
    is consulted, so a palette/scorer disagreement cannot silently promote an
    absent state into the denominator.

    Where the typed API permits it, every frozen patch is rehydrated as a
    :class:`FaceObservation` and scored with the production ``AppearanceScorer``.
    ``texture_weight`` defaults to the current production
    ``raycast.TEXTURE_WEIGHT``. The texture pattern is passed through
    ``use_pattern=True``: ``score_face_observation`` defaults that option to
    ``False``, which would build the pattern index only to ignore it and silently
    rank on colour alone, so the production pattern term must be requested
    explicitly. ``texture_weight=0`` still disables the term inside the scorer.
    Denominators stay separate: ``scored_patches`` is the abstention-free sample,
    and the ordered top-K rows divide only by ``truth_in_vocabulary_patches``.
    These are *ordered* recall rows, not the production ``argpartition``
    set-membership trace, and no equality is claimed.
    """
    if not palette:
        return {"available": False, "reason": "empty_palette"}
    weight = float(raycast.TEXTURE_WEIGHT if texture_weight is None else texture_weight)
    scorer = AppearanceScorer(tuple(palette), texture_weight=weight)
    vocabulary = palette_states(palette)
    scored = 0
    truth_in_vocabulary = 0
    out_of_vocabulary = 0
    abstentions: dict[str, int] = {}
    matches = {int(k): 0 for k in TOP_K}
    for patch in patches:
        observation = face_observation_from_sample(
            patch,
            frame_id=patch.frame_id,
            provenance=("e1-frozen-patch", f"frame:{patch.frame_id}"),
            weight=patch.weight,
        )
        score = scorer.score(observation, use_pattern=True)
        if score.abstained:
            reason = str(score.reason or "unknown")
            abstentions[reason] = abstentions.get(reason, 0) + 1
            continue
        scored += 1
        if patch.truth_state not in vocabulary:
            out_of_vocabulary += 1
            continue
        truth_in_vocabulary += 1
        ranked = [state for state, _cost in score.ranked()]
        position = ranked.index(patch.truth_state)
        for k in TOP_K:
            if position < k:
                matches[int(k)] += 1
    ordered = {
        str(k): {
            "matches": int(matches[int(k)]),
            "denominator": int(truth_in_vocabulary),
            "recall": (float(matches[int(k)]) / truth_in_vocabulary) if truth_in_vocabulary else None,
        }
        for k in TOP_K
    }
    return {
        "available": True,
        "scorer": "mc_photogrammetry.reconstruction.appearance_scoring.AppearanceScorer",
        "texture_weight": weight,
        "use_pattern": True,
        "patch_count": len(patches),
        "scored_patches": int(scored),
        "truth_in_vocabulary_patches": int(truth_in_vocabulary),
        "truth_out_of_vocabulary_patches": int(out_of_vocabulary),
        "vocabulary_size": len(vocabulary),
        "abstention_reasons": abstentions,
        "ordered_top_k": ordered,
        "production_set_membership_claimed": False,
        "note": (
            "ordered_top_k is this frozen sample's ordered recall; it is not the "
            "production argpartition set-membership trace and no equality is claimed"
        ),
    }


# --------------------------------------------------------------------------- #
# Sealing, writing and the artifact
# --------------------------------------------------------------------------- #


def patch_domain_problems(record: Mapping[str, Any]) -> list[str]:
    """Refuse any serialized patch domain that is not exactly 16x16x3 + masks."""
    problems: list[str] = []
    patches = record.get("patches")
    if not isinstance(patches, list):
        return ["artifact has no patch list"]
    for index, patch in enumerate(patches):
        if not isinstance(patch, Mapping):
            problems.append(f"patch {index} is not an object")
            continue
        for name, shape, dtype in (
            ("patch", [PATCH_SIZE, PATCH_SIZE, 3], np.dtype(np.float32).str),
            ("valid", [PATCH_SIZE, PATCH_SIZE], np.dtype(bool).str),
            ("coverage", [PATCH_SIZE, PATCH_SIZE], np.dtype(np.float32).str),
        ):
            entry = patch.get(name)
            if not isinstance(entry, Mapping):
                problems.append(f"patch {index} {name} is missing")
                continue
            if entry.get("shape") != shape or entry.get("dtype") != dtype:
                problems.append(
                    f"patch {index} {name} is not the frozen {shape} {dtype} domain"
                )
    return problems


def seal_record(record: Mapping[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key != "artifact_sha256"}
    return sha256_bytes(canonical_json(content))


def seal_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(record)
    sealed["artifact_sha256"] = seal_record(sealed)
    return sealed


def build_artifact(
    *,
    role: str,
    pin: PinVerification,
    inputs: Mapping[str, Any],
    sampler: Mapping[str, Any],
    denominators: Mapping[str, Any],
    coverage: Mapping[str, Any],
    patches: Sequence[PatchSample],
    scorer: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the deterministic, self-sealed patch-label artifact."""
    ordered = sorted(patches, key=lambda item: item.key)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_id": "texture-poc-e1-patch-labels",
        "role": role,
        "mode": MODE,
        "evaluation_only": True,
        "pin_verification": pin.to_record(),
        "inputs": dict(inputs),
        "sampler": dict(sampler),
        "denominators": dict(denominators),
        "coverage": dict(coverage),
        "patches": [patch.to_record() for patch in ordered],
        "scorer": dict(scorer),
        "reconciliation": dict(reconciliation),
    }
    problems = patch_domain_problems(record)
    if problems:
        raise PatchLabelError(f"refusing to seal a non-frozen patch domain: {problems[0]}")
    return seal_artifact(record)


def write_artifact(path: str | Path, record: Mapping[str, Any]) -> None:
    """Write the sealed artifact atomically, refusing to overwrite anything."""
    target = Path(path)
    if target.exists():
        raise PatchLabelError(f"refusing to overwrite existing artifact: {target}")
    expected = seal_record(record)
    if record.get("artifact_sha256") != expected:
        raise PatchLabelError("artifact digest does not match its own content")
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
        if target.exists():  # re-check immediately before replace
            raise PatchLabelError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _coverage_record(pre: Preflight, results: Sequence[FrameLabelResult], patches: Sequence[PatchSample]) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    abstentions: dict[str, int] = {}
    for result in results:
        events = {str(key): int(value) for key, value in result.sampler_events.items()}
        sources = (events, dict(result.refusals), dict(result.sample_rejections))
        reason_totals: dict[str, int] = {}
        for source in sources:
            for reason, count in source.items():
                if reason == "accepted":
                    continue
                reason_totals[reason] = reason_totals.get(reason, 0) + int(count)
                abstentions[reason] = abstentions.get(reason, 0) + int(count)
        frames[result.frame_id] = {
            "image_pixels": result.image_pixels,
            "usable_pixels": result.usable_pixels,
            "exact_cells": result.exact_cells,
            "refused_cells": result.refused_cells,
            "refused_state_faces": result.refused_state_faces,
            "frozen_patches": len(result.samples),
            "duplicate_faces": result.duplicate_faces,
            "sampler_events": events,
            "abstention_reasons": reason_totals,
        }
    by_slot: dict[str, int] = {}
    by_frame: dict[str, int] = {}
    by_state: dict[str, int] = {}
    for patch in patches:
        by_slot[patch.slot] = by_slot.get(patch.slot, 0) + 1
        by_frame[patch.frame_id] = by_frame.get(patch.frame_id, 0) + 1
        by_state[patch.truth_state.name] = by_state.get(patch.truth_state.name, 0) + 1
    return {
        "frames": frames,
        "frozen_patches": len(patches),
        "patches_by_slot": by_slot,
        "patches_by_frame": by_frame,
        "patches_by_state": by_state,
        "labeled_cells": len({patch.coordinate for patch in patches}),
        "partial_observations": sum(1 for patch in patches if patch.partial),
        "abstention_reasons": abstentions,
    }


def _reconciliation_record(denominators: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "denominator_interface_gap_recorded",
        "claimed_equal": False,
        "historical_rows": [dict(row) for row in HISTORICAL_TOP_K_ROWS],
        "frozen_denominators": {
            "all_visible_cells": int(denominators.get("all_visible_cells", 0)),
            "scorer_usable_cells": int(denominators.get("scorer_usable_cells", 0)),
            "geometry_labelable_cells": int(denominators.get("geometry_labelable_cells", 0)),
            "model_vocabulary_cells": int(denominators.get("model_vocabulary_cells", 0)),
        },
        "gap": RECONCILIATION_GAP,
        "stop_condition": (
            "Reconcile these denominators before any training; if the exact patches "
            "cannot be shown to reach the historical rows, record the gap and stop."
        ),
    }


def run(
    role: str,
    *,
    output: str | Path,
    fixture_dir: str | Path | None = None,
    reference: str | Path | None = None,
    camera_manifest: str | Path | None = None,
    scene: str | Path | None = None,
    jar: str | Path | None = None,
    mask_root: str | Path | None = None,
    pin_manifest: Mapping[str, Any] | None = None,
    expected_asset_pins: Mapping[str, Mapping[str, str]] | None = None,
    include_scorer: bool = True,
    minimum_area: float = MINIMUM_FACE_AREA,
    minimum_visible_fraction: float = 0.6,
    shrink: float = DEFAULT_SHRINK,
    minimum_valid_fraction: float = MINIMUM_PATCH_VALID_FRACTION,
) -> dict[str, Any]:
    """Derive, freeze and seal one pinned fixture's labels; never overwrite.

    Preflight (fixture admission, every pin, every mask domain) completes before
    any image is decoded. A refusal therefore leaves no output behind. This
    function is the only entry point that writes, and it is deliberately not
    invoked by this task.
    """
    target = Path(output)
    if target.exists():
        raise PatchLabelError(f"refusing to overwrite existing artifact: {target}")
    pre = preflight(
        role,
        fixture_dir=fixture_dir,
        reference=reference,
        camera_manifest=camera_manifest,
        scene=scene,
        jar=jar,
        mask_root=mask_root,
        pin_manifest=pin_manifest,
        expected_asset_pins=expected_asset_pins,
    )

    provider = VanillaJarTextureProvider()
    loaded = provider.load_jar(pre.jar_path)
    if not loaded:
        raise PatchLabelError(f"vanilla JAR produced no texture samples: {pre.jar_path}")
    snapshot = load_litematic(pre.reference_path)
    reference_states = tuple(snapshot.palette)
    # Two deliberately separate palettes. ``geometry_palette`` carries every
    # reference state (partial models included) so truth-owned exact geometry and
    # rendering are complete; it is never the scoring vocabulary. ``scoring_palette``
    # is the canonical full-cube production vocabulary the pinned benchmark scores
    # against, and is the only palette patches are scored on.
    geometry_palette = provider.palette_for_states(reference_states, include_partial=True)
    if not geometry_palette:
        raise PatchLabelError("JAR palette carries no exact model entries for the reference states")
    modelled = exact_model_states(geometry_palette)
    if not modelled:
        raise PatchLabelError("JAR palette carries no exact baked model polygons for the reference")
    scoring_palette = provider.canonical_palette()
    if not scoring_palette:
        raise PatchLabelError("JAR canonical scoring palette is empty")
    scoring_vocabulary = palette_states(scoring_palette)
    # Third, independent palette: the production visible-cell domain is rendered
    # through the raw extraction table, exactly as ``scripts/benchmark_templates.py``
    # does, so E1's ``all_visible_cells`` denominator is the same domain the pinned
    # benchmark reports. It is not the geometry palette and not the scoring palette.
    visibility_palette = production_visibility_palette(provider)

    cameras = load_camera_manifest(pre.camera_manifest)
    missing_cameras = sorted(set(pre.frames) - set(cameras))
    if missing_cameras:
        raise PatchLabelError(f"camera manifest lacks frames: {', '.join(missing_cameras)}")
    scene_transform = load_scene_transform(pre.scene_definition)

    images: dict[str, np.ndarray] = {}
    for frame in pre.frames:
        image = load_image(pre.fixture_dir / frame)
        if image.shape[:2] != pre.masks[frame].shape:
            raise PatchLabelError(
                f"decoded image dimensions differ from the frozen mask for {frame}"
            )
        images[frame] = image

    results = [
        derive_frame_labels(
            frame,
            images[frame],
            cameras[frame],
            scene_transform,
            snapshot.blocks,
            geometry_palette,
            pre.masks[frame],
            minimum_area=minimum_area,
            minimum_visible_fraction=minimum_visible_fraction,
            shrink=shrink,
            minimum_valid_fraction=minimum_valid_fraction,
        )
        for frame in pre.frames
    ]
    patches = tuple(
        sorted((sample for result in results for sample in result.samples), key=lambda item: item.key)
    )

    resolutions = {frame: (images[frame].shape[1], images[frame].shape[0]) for frame in pre.frames}
    # The production visibility path: provider.samples for the full-cube keys, the
    # shape table and the render palette, exactly like scripts/benchmark_templates.py.
    # geometry_palette is used only for ownership/label derivation above.
    visibility = visible_cells(
        snapshot.blocks,
        scene_transform,
        cameras,
        resolutions,
        usable_masks=pre.masks,
        **visibility_domain_arguments(visibility_palette),
    )
    all_visible_cells = sum(1 for count in visibility.values() if count >= 1)
    # Geometry-metadata denominator: visible cells whose truth state carries exact
    # baked polygons in the reference geometry palette. This is *not* the scoring
    # vocabulary; it is retained under a name that says what it actually counts.
    geometry_labelable_cells = sum(
        1
        for coordinate, count in visibility.items()
        if count >= 1 and snapshot.blocks[coordinate] in modelled
    )
    # Scoring-vocabulary denominator: visible cells whose truth state is a member
    # of the canonical full-cube scoring palette.
    model_vocabulary_cells = sum(
        1
        for coordinate, count in visibility.items()
        if count >= 1 and snapshot.blocks[coordinate] in scoring_vocabulary
    )
    scorer_usable_cells = len(
        {patch.coordinate for patch in patches if visibility.get(patch.coordinate, 0) >= 1}
    )
    denominators = {
        "all_visible_cells": int(all_visible_cells),
        "scorer_usable_cells": int(scorer_usable_cells),
        "geometry_labelable_cells": int(geometry_labelable_cells),
        "model_vocabulary_cells": int(model_vocabulary_cells),
        "reference_cells": int(len(snapshot.blocks)),
        "exact_model_reference_states": len(modelled),
        "canonical_scoring_states": len(scoring_vocabulary),
    }
    coverage = _coverage_record(pre, results, patches)
    scorer = (
        score_patches(patches, scoring_palette)
        if include_scorer
        else {"available": False, "reason": "scorer disabled by caller"}
    )
    sampler = {
        "patch_size": PATCH_SIZE,
        "minimum_area": float(minimum_area),
        "minimum_visible_fraction": float(minimum_visible_fraction),
        "shrink": float(shrink),
        "depth_scale": float(DEPTH_SCALE),
        "minimum_patch_valid_fraction": float(minimum_valid_fraction),
        "ownership": "exact_model_alpha_tested_renderer",
        "weight_formula": (
            "area**AREA_POWER / (1 + depth/16)"
            " * visible**VISIBILITY_POWER * facing**OBLIQUITY_POWER"
            " / (1 + _face_uniformity(observed_patch, coverage)/UNIFORMITY_SCALE)"
        ),
        "area_power": float(raycast.AREA_POWER),
        "visibility_power": float(raycast.VISIBILITY_POWER),
        "obliquity_power": float(raycast.OBLIQUITY_POWER),
        "uniformity_scale": float(raycast.UNIFORMITY_SCALE),
        "uniformity_divisor": (
            "included exactly: 1 + _face_uniformity(observed_patch, coverage)"
            "/ raycast.UNIFORMITY_SCALE"
        ),
        "texture_weight": float(raycast.TEXTURE_WEIGHT),
        "geometry_palette": "provider.palette_for_states(reference_states, include_partial=True)",
        "scoring_palette": "provider.canonical_palette()",
        "visibility_palette": "provider.samples",
    }
    inputs = {
        "fixture_dir": str(pre.fixture_dir),
        "reference": {"path": str(pre.reference_path), "sha256": pre.reference_sha256},
        "camera_manifest": {
            "path": str(pre.camera_manifest),
            "sha256": pre.asset_sha256["cameras"],
            "pinned_sha256": EXPECTED_ASSET_PINS[role]["cameras"],
        },
        "scene_definition": {
            "path": str(pre.scene_definition),
            "sha256": pre.asset_sha256["scene"],
            "pinned_sha256": EXPECTED_ASSET_PINS[role]["scene"],
        },
        "minecraft_jar": {
            "path": str(pre.jar_path),
            "sha256": pre.asset_sha256["jar"],
            "pinned_sha256": EXPECTED_ASSET_PINS[LOCAL_JAR_PIN_KEY]["jar"],
        },
        "asset_pins_verified": True,
        "frames": [
            {"name": frame, "sha256": pre.frame_sha256[frame]} for frame in pre.frames
        ],
        "masks": [
            {
                "frame": frame,
                "spec": str(pre.mask_paths[frame]),
                "domain_sha256": pin_domain,
                "file_sha256": pre.mask_file_sha256[frame],
            }
            for frame, pin_domain, _file_digest in pre.pin.usable_masks
        ],
    }
    record = build_artifact(
        role=role,
        pin=pre.pin,
        inputs=inputs,
        sampler=sampler,
        denominators=denominators,
        coverage=coverage,
        patches=patches,
        scorer=scorer,
        reconciliation=_reconciliation_record(denominators),
    )
    write_artifact(target, record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only E1 visible-face label and production-patch audit. "
            "Admits only the reviewed pinned Template 2 / Template 6 fixtures and "
            "refuses overwrite, malformed pins and unpinned roles."
        )
    )
    parser.add_argument("role", choices=sorted(FIXTURE_ROLES))
    parser.add_argument("output", type=Path, help="sealed patch-label artifact path (must not exist)")
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--camera-manifest", type=Path, default=None)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--minecraft-jar", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--no-scorer", action="store_true", help="skip the existing-scorer read")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record = run(
            args.role,
            output=args.output,
            fixture_dir=args.fixture_dir,
            reference=args.reference,
            camera_manifest=args.camera_manifest,
            scene=args.scene,
            jar=args.minecraft_jar,
            mask_root=args.mask_root,
            include_scorer=not args.no_scorer,
        )
    except PatchLabelError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(f"mode: {record['mode']}")
    print(f"role: {record['role']} (pinned={record['pin_verification']['pinned']})")
    print(f"frozen patches: {record['coverage']['frozen_patches']}")
    print(f"artifact: {args.output} sha256 {record['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
