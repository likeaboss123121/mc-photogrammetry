"""Post-freeze, ground-truth-assisted scoring for experimental plane proposals.

This module is deliberately outside reconstruction and the normal CLI. It first
verifies an immutable image-only freeze, then reads explicitly supplied capture
and reference artifacts. GT rendering is an evaluation proxy: the exported GT
volume omits exterior world geometry, so its ownership masks are not an
occluder-correct visibility oracle.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import zipfile

import cv2
import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.blocks.textures import (
    VanillaJarTextureProvider,
    _load_cached_samples,
    _palette_color,
)
from mc_photogrammetry.dataset.schemas import (
    MAX_REGION_VOLUME, BlockRecord, GroundTruthExport, RegionBounds,
)
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export import attribution
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.reconstruction.camera import CameraModel, intrinsics_for
from mc_photogrammetry.reconstruction.frozen_image_runner import (
    FrozenImageRunError,
    load_and_verify_frozen_image,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.vision.gui_mask import GuiMaskConfig, create_gui_mask


MAX_EVALUATION_PIXELS = 6_000_000
MAX_FROZEN_FRAMES = 32
MAX_TOTAL_EVALUATION_PIXELS = 48_000_000
MAX_GROUND_TRUTH_BLOCKS = 250_000
MAX_TRUTH_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_RENDERED_EVIDENCE_BYTES = 256 * 1024 * 1024
SCHEMA_VERSION = 1
_SLOT_AXIS_SIDE = {
    "west": (0, 0), "east": (0, 1),
    "down": (1, 0), "up": (1, 1),
    "north": (2, 0), "south": (2, 1),
}
_AIR_STATE_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})
_CONTEXT_ARTIFACT_TYPE = "mc-photogrammetry-context-ground-truth"
_CONTEXT_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "artifact_type", "scene_id", "world_identity",
    "capture_metadata_sha256", "target_ground_truth_sha256", "ordered_images",
    "target_region", "region", "blocks",
})


class FrozenPlaneEvaluationError(ValueError):
    """Invalid or unverifiable input to the post-freeze evaluator."""


@dataclass(frozen=True, slots=True)
class RenderedTruthSurface:
    direction: str
    coordinate: float
    mask: np.ndarray

    def __post_init__(self) -> None:
        if (self.direction not in _SLOT_AXIS_SIDE
                or not isinstance(self.mask, np.ndarray)
                or self.mask.dtype != np.bool_ or self.mask.ndim != 2
                or self.mask.flags.writeable):
            raise FrozenPlaneEvaluationError("rendered truth surface is mutable or malformed")


@dataclass(frozen=True, slots=True)
class RenderedTruthEvidence:
    schema: str
    camera: CameraModel
    truth_sha256: str
    vanilla_jar_sha256: str
    palette_sha256: str
    rgb: np.ndarray
    ownership: np.ndarray
    owner_index: tuple[tuple[tuple[int, int, int], str], ...]
    surfaces: tuple[RenderedTruthSurface, ...]
    usable: np.ndarray
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != "mc-photogrammetry-rendered-truth-evidence-v1":
            raise FrozenPlaneEvaluationError("rendered truth evidence schema is unsupported")
        arrays = (self.rgb, self.ownership, self.usable)
        if (any(not isinstance(value, np.ndarray) or value.flags.writeable for value in arrays)
                or self.rgb.dtype != np.uint8 or self.rgb.ndim != 3
                or self.rgb.shape[2] != 3 or self.ownership.dtype.kind not in "iu"
                or self.ownership.shape != self.rgb.shape[:2]
                or self.usable.dtype != np.bool_ or self.usable.shape != self.rgb.shape[:2]
                or any(surface.mask.shape != self.usable.shape for surface in self.surfaces)):
            raise FrozenPlaneEvaluationError("rendered truth evidence arrays are mutable or malformed")
        payload = self.to_record(include_digest=False)
        digest = _sha256(_canonical_json(payload))
        if self.evidence_sha256 not in ("", digest):
            raise FrozenPlaneEvaluationError("rendered truth evidence digest mismatch")
        object.__setattr__(self, "evidence_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        def array_row(value: np.ndarray) -> dict[str, Any]:
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": _sha256(value.tobytes(order="C"))}
        result = {
            "schema": self.schema, "camera": self.camera.to_dict(),
            "truth_sha256": self.truth_sha256,
            "vanilla_jar_sha256": self.vanilla_jar_sha256,
            "palette_sha256": self.palette_sha256,
            "rgb": array_row(self.rgb), "ownership": array_row(self.ownership),
            "usable": array_row(self.usable),
            "owner_index": [[list(position), direction]
                            for position, direction in self.owner_index],
            "surfaces": [{"direction": surface.direction,
                          "coordinate": surface.coordinate,
                          "mask": array_row(surface.mask)} for surface in self.surfaces],
        }
        if include_digest:
            result["evidence_sha256"] = self.evidence_sha256
        return result


@dataclass(frozen=True, slots=True)
class RenderedTruthOwnershipBindingV1:
    schema: str
    evidence_sha256: str
    ownership_sha256: str
    owner_index_sha256: str
    renderer_witness_sha256: str
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-rendered-truth-owner-binding-v1"
                or any(type(value) is not str or len(value) != 64
                       for value in (self.evidence_sha256, self.ownership_sha256,
                                     self.owner_index_sha256,
                                     self.renderer_witness_sha256))):
            raise FrozenPlaneEvaluationError("renderer ownership binding is malformed")
        digest = _sha256(_canonical_json(self.to_record(include_digest=False)))
        if self.binding_sha256 not in ("", digest):
            raise FrozenPlaneEvaluationError("renderer ownership binding digest mismatch")
        object.__setattr__(self, "binding_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {"schema": self.schema, "evidence_sha256": self.evidence_sha256,
                  "ownership_sha256": self.ownership_sha256,
                  "owner_index_sha256": self.owner_index_sha256,
                  "renderer_witness_sha256": self.renderer_witness_sha256}
        if include_digest:
            result["binding_sha256"] = self.binding_sha256
        return result


@dataclass(frozen=True, slots=True)
class RenderedTruthOwnershipWitnessPayloadV1:
    """Private, evaluation-only carrier for an exact renderer witness."""
    binding: RenderedTruthOwnershipBindingV1
    witness: attribution.RendererOwnershipWitnessV1

    def __post_init__(self) -> None:
        if (not isinstance(self.binding, RenderedTruthOwnershipBindingV1)
                or not isinstance(self.witness, attribution.RendererOwnershipWitnessV1)):
            raise FrozenPlaneEvaluationError("renderer ownership payload is malformed")
        binding = RenderedTruthOwnershipBindingV1(**{
            name: getattr(self.binding, name)
            for name in self.binding.__dataclass_fields__})
        replay = attribution.RendererOwnershipWitnessV1(**{
            name: getattr(self.witness, name)
            for name in self.witness.__dataclass_fields__})
        if replay.witness_sha256 != binding.renderer_witness_sha256:
            raise FrozenPlaneEvaluationError("renderer ownership payload digest mismatch")


def bind_renderer_ownership_witness(
    evidence: RenderedTruthEvidence,
    witness: attribution.RendererOwnershipWitnessV1,
) -> RenderedTruthOwnershipBindingV1:
    """Bind an evaluation renderer witness to the exact frozen evidence."""
    if (not isinstance(evidence, RenderedTruthEvidence)
            or not isinstance(witness, attribution.RendererOwnershipWitnessV1)
            or len(witness.owners) != len(evidence.owner_index)):
        raise FrozenPlaneEvaluationError("renderer owner index does not bind evidence")
    for row, (position, slot) in zip(witness.owners, evidence.owner_index):
        if row.owner_id < 0 or row.block_coordinate != position or row.slot != slot:
            raise FrozenPlaneEvaluationError("renderer owner index does not bind evidence")
    ownership_hash = _sha256(evidence.ownership.tobytes(order="C"))
    owner_index_hash = _sha256(_canonical_json([
        [list(position), slot] for position, slot in evidence.owner_index]))
    return RenderedTruthOwnershipBindingV1(
        "mc-photogrammetry-rendered-truth-owner-binding-v1",
        evidence.evidence_sha256, ownership_hash, owner_index_hash,
        witness.witness_sha256)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON value {value}")

    try:
        return json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FrozenPlaneEvaluationError(f"{label} is not valid strict JSON") from error


def _read_json(path: Path, label: str) -> tuple[bytes, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FrozenPlaneEvaluationError(f"{label} could not be read") from error
    return raw, _strict_json(raw, label)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_id(document: Mapping[str, Any]) -> str:
    content = dict(document)
    content.pop("artifact_id", None)
    return _sha256(_canonical_json(content))


def _code_hashes() -> dict[str, str]:
    from mc_photogrammetry.reconstruction import camera, frozen_image_runner
    from mc_photogrammetry.dataset import schemas
    from mc_photogrammetry.blocks import textures
    from mc_photogrammetry.vision import gui_mask

    modules = {
        "evaluator": Path(__file__),
        "attribution": Path(attribution.__file__),
        "camera": Path(camera.__file__),
        "calibration": Path(axis_directions.__code__.co_filename),
        "frozen_loader": Path(frozen_image_runner.__file__),
        "dataset_schema": Path(schemas.__file__),
        "texture_cache": Path(textures.__file__),
        "gui_mask": Path(gui_mask.__file__),
    }
    cli_path = Path(__file__).parents[2] / "scripts" / "evaluate_frozen_planes.py"
    if cli_path.is_file():
        modules["explicit_evaluator_script"] = cli_path
    return {name: _sha256(path.read_bytes()) for name, path in sorted(modules.items())}


def _decode_rgb(path: Path) -> tuple[bytes, np.ndarray]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise FrozenPlaneEvaluationError(f"capture image {path.name!r} could not be read") from error
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FrozenPlaneEvaluationError(f"capture image {path.name!r} is not a decodable PNG")
    if image.shape[0] * image.shape[1] > MAX_EVALUATION_PIXELS:
        raise FrozenPlaneEvaluationError(f"capture image {path.name!r} exceeds pixel cap")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return payload, rgb


def _record_basename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise FrozenPlaneEvaluationError(f"{label} must be a safe basename")
    return value


def _verify_capture_images(
    freeze: Mapping[str, Any], capture: Mapping[str, Any], capture_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    frames = freeze.get("frames")
    capture_frames = capture.get("frames")
    cameras = capture.get("cameras")
    resolutions = capture.get("resolutions")
    screenshots = capture.get("screenshots")
    if not all(isinstance(value, list) for value in
               (frames, capture_frames, cameras, resolutions, screenshots)):
        raise FrozenPlaneEvaluationError("capture frame, camera, resolution, and screenshot tables are required")
    count = len(frames)
    if not (1 <= count <= MAX_FROZEN_FRAMES and len(capture_frames) == len(cameras) == len(resolutions) == len(screenshots) == count):
        raise FrozenPlaneEvaluationError("capture tables must have exactly one entry per frozen frame")
    frame_by_name: dict[str, Mapping[str, Any]] = {}
    camera_by_name: dict[str, Mapping[str, Any]] = {}
    resolution_by_name: dict[str, Any] = {}
    for index, frame in enumerate(capture_frames):
        if not isinstance(frame, Mapping) or not isinstance(cameras[index], Mapping):
            raise FrozenPlaneEvaluationError("capture frame or camera record is malformed")
        name = _record_basename(frame.get("filename"), "capture frame filename")
        if not name.lower().endswith(".png"):
            raise FrozenPlaneEvaluationError("capture frame filenames must be PNG basenames")
        if frame.get("frame_id") is not None and frame.get("frame_id") != Path(name).stem:
            raise FrozenPlaneEvaluationError(f"capture frame ID differs for {name!r}")
        if name in frame_by_name:
            raise FrozenPlaneEvaluationError("capture frame filenames must be unique")
        frame_by_name[name] = frame
        camera_by_name[name] = cameras[index]
        resolution_by_name[name] = resolutions[index]
    screenshot_by_name: dict[str, Mapping[str, Any]] = {}
    for row in screenshots:
        if not isinstance(row, Mapping):
            raise FrozenPlaneEvaluationError("capture screenshot inventory is malformed")
        name = _record_basename(row.get("filename"), "screenshot filename")
        if name in screenshot_by_name:
            raise FrozenPlaneEvaluationError("screenshot filenames must be unique")
        screenshot_by_name[name] = row

    images: dict[str, np.ndarray] = {}
    camera_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    total_pixels = 0
    for frozen_frame in frames:
        if not isinstance(frozen_frame, Mapping) or not isinstance(frozen_frame.get("input"), Mapping):
            raise FrozenPlaneEvaluationError("frozen frame inventory is malformed")
        item = frozen_frame["input"]
        basename = _record_basename(item.get("basename"), "frozen input basename")
        if frozen_frame.get("frame_id") != basename:
            raise FrozenPlaneEvaluationError("frozen frame id differs from input basename")
        capture_frame = frame_by_name.get(basename)
        screenshot_row = screenshot_by_name.get(basename)
        camera_row = camera_by_name.get(basename)
        resolution = resolution_by_name.get(basename)
        if capture_frame is None or screenshot_row is None or camera_row is None or resolution is None:
            raise FrozenPlaneEvaluationError(f"capture inventory is missing frozen frame {basename!r}")
        if capture_frame.get("captured") is not True or capture_frame.get("filename") != basename:
            raise FrozenPlaneEvaluationError(f"capture frame {basename!r} is not marked captured")
        if screenshot_row.get("filename") != basename:
            raise FrozenPlaneEvaluationError(f"screenshot inventory disagrees for {basename!r}")
        width, height = item.get("width"), item.get("height")
        if (isinstance(width, bool) or not isinstance(width, int) or width <= 0
                or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
            raise FrozenPlaneEvaluationError(f"frozen dimensions are invalid for {basename!r}")
        if width * height > MAX_EVALUATION_PIXELS:
            raise FrozenPlaneEvaluationError(f"frozen dimensions exceed pixel cap for {basename!r}")
        if width * height > MAX_TOTAL_EVALUATION_PIXELS - total_pixels:
            raise FrozenPlaneEvaluationError("aggregate frozen dimensions exceed pixel cap")
        total_pixels += width * height
        if not isinstance(resolution, (list, tuple)) or tuple(resolution) != (width, height):
            raise FrozenPlaneEvaluationError(f"capture resolution differs for {basename!r}")
        actual = capture_frame.get("actual")
        if isinstance(actual, Mapping):
            if (actual.get("framebuffer_width") != width
                    or actual.get("framebuffer_height") != height):
                raise FrozenPlaneEvaluationError(f"capture framebuffer dimensions differ for {basename!r}")
        if camera_row.get("intrinsics") is not None:
            intrinsics = camera_row.get("intrinsics")
            if (not isinstance(intrinsics, Mapping)
                    or tuple(intrinsics.get("image_size", ())) != (width, height)):
                raise FrozenPlaneEvaluationError(f"capture camera dimensions differ for {basename!r}")
        effective_camera = dict(camera_row)
        if isinstance(actual, Mapping):
            # Prefer measured pose/optics recorded with this screenshot over the
            # planned camera row; retain its explicit calibrated intrinsics.
            feet = actual.get("feet_position")
            if isinstance(feet, (list, tuple)) and len(feet) == 3:
                effective_camera["position"] = list(feet)
            for key in ("yaw_degrees", "pitch_degrees", "fov_degrees"):
                if actual.get(key) is not None:
                    effective_camera[key] = actual[key]
        relative = Path("captures") / basename
        image_path = capture_path.parent / relative
        raw, image = _decode_rgb(image_path)
        digest = _sha256(raw)
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or digest != expected_hash:
            raise FrozenPlaneEvaluationError(f"capture image bytes differ from freeze for {basename!r}")
        if screenshot_row.get("sha256") != digest:
            raise FrozenPlaneEvaluationError(f"capture screenshot hash differs for {basename!r}")
        if screenshot_row.get("bytes") != len(raw):
            raise FrozenPlaneEvaluationError(f"capture screenshot size differs for {basename!r}")
        if isinstance(actual, Mapping):
            if (actual.get("screenshot_sha256") not in (None, digest)
                    or actual.get("screenshot_bytes") not in (None, len(raw))):
                raise FrozenPlaneEvaluationError(f"capture result hash or size differs for {basename!r}")
        if image.shape[:2] != (height, width):
            raise FrozenPlaneEvaluationError(f"decoded screenshot dimensions differ for {basename!r}")
        images[basename] = image
        camera_rows.append(effective_camera)
        resolution_rows.append({"basename": basename, "width": width, "height": height,
                                "sha256": digest})
    return images, camera_rows, resolution_rows


def _load_truth(path: Path) -> tuple[bytes, GroundTruthExport]:
    raw, document = _read_json(path, "ground-truth export")
    try:
        truth = GroundTruthExport.from_dict(document)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        raise FrozenPlaneEvaluationError("ground-truth export failed schema validation") from error
    if len(truth.blocks) > MAX_GROUND_TRUTH_BLOCKS:
        raise FrozenPlaneEvaluationError("ground-truth block count exceeds evaluation cap")
    return raw, truth


def _strict_context_region(value: Any, label: str) -> RegionBounds:
    if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
        raise FrozenPlaneEvaluationError(f"context {label} must contain exactly min/max")
    coordinates = []
    for field in ("min", "max"):
        raw = value[field]
        if (not isinstance(raw, list) or len(raw) != 3
                or any(type(component) is not int for component in raw)):
            raise FrozenPlaneEvaluationError(f"context {label} {field} must be three integers")
        coordinates.append(tuple(raw))
    try:
        return RegionBounds(coordinates[0], coordinates[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise FrozenPlaneEvaluationError(f"context {label} violates region limits") from error


def _validate_context_ground_truth(
    document: Any,
    *,
    capture_sha256: str,
    target_sha256: str,
    capture_doc: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    target_truth: GroundTruthExport,
) -> GroundTruthExport:
    """Strictly validate an evaluator-only context artifact against verified inputs."""
    if not isinstance(document, Mapping) or set(document) != _CONTEXT_TOP_LEVEL_KEYS:
        raise FrozenPlaneEvaluationError("context ground truth has unknown or missing top-level keys")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise FrozenPlaneEvaluationError("context ground truth schema version is unsupported")
    if document.get("artifact_type") != _CONTEXT_ARTIFACT_TYPE:
        raise FrozenPlaneEvaluationError("context ground truth artifact type is invalid")
    scene_id = document.get("scene_id")
    world_identity = document.get("world_identity")
    if (not isinstance(scene_id, str) or not scene_id
            or not isinstance(world_identity, str) or not world_identity):
        raise FrozenPlaneEvaluationError("context ground truth needs scene/world identities")
    if scene_id != capture_doc.get("scene_id") or scene_id != target_truth.scene_id:
        raise FrozenPlaneEvaluationError("context ground truth scene ID differs")
    if world_identity != capture_doc.get("world_identity") or world_identity != target_truth.world_identity:
        raise FrozenPlaneEvaluationError("context ground truth world identity differs")
    if document.get("capture_metadata_sha256") != capture_sha256:
        raise FrozenPlaneEvaluationError("context capture metadata hash differs")
    if document.get("target_ground_truth_sha256") != target_sha256:
        raise FrozenPlaneEvaluationError("context target ground-truth hash differs")

    ordered_images = document.get("ordered_images")
    if not isinstance(ordered_images, list) or len(ordered_images) != len(input_rows):
        raise FrozenPlaneEvaluationError("context ordered image inventory differs")
    for context_image, expected_image in zip(ordered_images, input_rows):
        if (not isinstance(context_image, Mapping)
                or set(context_image) != {"basename", "width", "height", "sha256"}
                or not isinstance(context_image.get("basename"), str)
                or type(context_image.get("width")) is not int
                or type(context_image.get("height")) is not int
                or not isinstance(context_image.get("sha256"), str)
                or dict(context_image) != dict(expected_image)):
            raise FrozenPlaneEvaluationError("context ordered image hash/dimensions differ")

    target_region = _strict_context_region(document.get("target_region"), "target region")
    if target_region != target_truth.region:
        raise FrozenPlaneEvaluationError("context target region differs from target ground truth")
    context_region = _strict_context_region(document.get("region"), "region")
    if not all(context_region.minimum[axis] <= target_region.minimum[axis]
               and context_region.maximum[axis] >= target_region.maximum[axis]
               for axis in range(3)):
        raise FrozenPlaneEvaluationError("context region does not contain the target region")
    if (context_region.size[0] * context_region.size[1] * context_region.size[2]
            > MAX_REGION_VOLUME):
        raise FrozenPlaneEvaluationError("context region volume exceeds evaluation cap")

    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) > MAX_GROUND_TRUTH_BLOCKS:
        raise FrozenPlaneEvaluationError("context block inventory exceeds evaluation cap or is malformed")
    blocks: list[BlockRecord] = []
    for row in raw_blocks:
        if not isinstance(row, Mapping) or set(row) != {"pos", "name", "properties"}:
            raise FrozenPlaneEvaluationError("context block record has unknown or missing keys")
        position = row.get("pos")
        properties = row.get("properties")
        if (not isinstance(position, list) or len(position) != 3
                or any(type(component) is not int for component in position)
                or not isinstance(row.get("name"), str)
                or not isinstance(properties, Mapping)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in properties.items())):
            raise FrozenPlaneEvaluationError("context block record has invalid field types")
        try:
            blocks.append(BlockRecord(tuple(position), row["name"], dict(properties)))
        except (TypeError, ValueError, OverflowError) as error:
            raise FrozenPlaneEvaluationError("context block record failed schema validation") from error
    try:
        context_truth = GroundTruthExport(scene_id, world_identity, context_region, tuple(blocks))
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        raise FrozenPlaneEvaluationError("context ground truth failed schema validation") from error
    if len(context_truth.blocks) > MAX_REGION_VOLUME:
        raise FrozenPlaneEvaluationError("context ground-truth block count exceeds volume cap")
    context_by_position = {
        record.position: (record.name, dict(record.properties)) for record in context_truth.blocks
    }
    for record in target_truth.blocks:
        if context_by_position.get(record.position) != (record.name, dict(record.properties)):
            raise FrozenPlaneEvaluationError(
                "context ground truth does not preserve every target block state")
    return context_truth


def _load_context_truth(
    path: Path,
    *,
    capture_sha256: str,
    target_sha256: str,
    capture_doc: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    target_truth: GroundTruthExport,
) -> tuple[bytes, GroundTruthExport]:
    raw, document = _read_json(path, "context ground-truth export")
    truth = _validate_context_ground_truth(
        document, capture_sha256=capture_sha256, target_sha256=target_sha256,
        capture_doc=capture_doc, input_rows=input_rows, target_truth=target_truth)
    return raw, truth


def _load_palette(jar_path: Path, palette_path: Path,
                  states: Sequence[BlockState]
                  ) -> tuple[tuple[Any, ...], str, str, list[dict[str, Any]]]:
    try:
        jar_bytes = jar_path.read_bytes()
        palette_bytes = palette_path.read_bytes()
        cached = _load_cached_samples(palette_path, jar_path)
    except OSError as error:
        raise FrozenPlaneEvaluationError("explicit vanilla jar or palette cache could not be read") from error
    if not jar_bytes or not palette_bytes or not cached:
        raise FrozenPlaneEvaluationError("vanilla palette cache is absent or does not match the supplied jar")
    provider = VanillaJarTextureProvider()
    provider.samples = cached
    provider.jar_path = jar_path.resolve()
    provider.source = "explicit_cached_palette"
    exact_palette = provider.palette_for_states(states, include_partial=True)
    exact_states = {entry.state for entry in exact_palette}
    jar_hash = _sha256(jar_bytes)
    blockstate_resources = _load_blockstate_resources(
        jar_path, (state for state in states if state not in exact_states))
    palette, projection = _project_palette_states(
        states, cached, exact_palette,
        blockstate_resources=blockstate_resources,
        vanilla_jar_sha256=jar_hash,
    )
    return palette, jar_hash, _sha256(palette_bytes), projection


def _load_blockstate_resources(
    jar_path: Path, states: Sequence[BlockState]
) -> dict[str, tuple[str, bytes, str]]:
    """Read only the required vanilla blockstate resources from the pinned JAR.

    Each value is ``(asset_path, raw_bytes, asset_sha256)``. A missing resource
    remains absent here and is a typed refusal only if projection later requires
    that state's blockstate selectors.
    """
    names = sorted({state.name for state in states})
    if not names:
        return {}
    try:
        with zipfile.ZipFile(jar_path, "r") as archive:
            resources: dict[str, tuple[str, bytes, str]] = {}
            for name in names:
                namespace, separator, path_name = name.partition(":")
                if not separator or namespace != "minecraft" or not path_name:
                    continue
                asset_path = f"assets/minecraft/blockstates/{path_name}.json"
                try:
                    payload = archive.read(asset_path)
                except KeyError:
                    continue
                resources[name] = (asset_path, payload, _sha256(payload))
            return resources
    except (OSError, zipfile.BadZipFile) as error:
        raise FrozenPlaneEvaluationError(
            "the explicit vanilla JAR could not supply blockstate selector evidence"
        ) from error


def _variant_selector(key: str) -> dict[str, str]:
    if not isinstance(key, str):
        raise FrozenPlaneEvaluationError("vanilla blockstate variant key is malformed")
    selector: dict[str, str] = {}
    if not key:
        return selector
    for assignment in key.split(","):
        property_name, separator, value = assignment.partition("=")
        property_name = property_name.strip()
        value = value.strip()
        if (not separator or not property_name or not value or "=" in value
                or property_name in selector):
            raise FrozenPlaneEvaluationError("vanilla blockstate variant key is malformed")
        selector[property_name] = value
    return selector


def _condition_property_names(condition: Any) -> set[str]:
    if condition is None:
        return set()
    if not isinstance(condition, Mapping):
        raise FrozenPlaneEvaluationError("vanilla multipart condition is malformed")
    names: set[str] = set()
    for key, value in condition.items():
        if key in {"OR", "AND"}:
            if not isinstance(value, list) or not value:
                raise FrozenPlaneEvaluationError("vanilla multipart boolean is malformed")
            for part in value:
                names.update(_condition_property_names(part))
        elif isinstance(key, str) and key and isinstance(value, str) and value:
            names.add(key)
        else:
            raise FrozenPlaneEvaluationError("vanilla multipart property condition is malformed")
    return names


def _condition_matches(condition: Any, properties: Mapping[str, str]) -> bool:
    if condition is None:
        return True
    if not isinstance(condition, Mapping):
        raise FrozenPlaneEvaluationError("vanilla multipart condition is malformed")
    for key, value in condition.items():
        if key == "OR":
            if not isinstance(value, list) or not value:
                raise FrozenPlaneEvaluationError("vanilla multipart OR condition is malformed")
            if not any(_condition_matches(part, properties) for part in value):
                return False
        elif key == "AND":
            if not isinstance(value, list) or not value:
                raise FrozenPlaneEvaluationError("vanilla multipart AND condition is malformed")
            if not all(_condition_matches(part, properties) for part in value):
                return False
        elif isinstance(key, str) and key and isinstance(value, str) and value:
            if key not in properties or properties[key] not in value.split("|"):
                return False
        else:
            raise FrozenPlaneEvaluationError("vanilla multipart property condition is malformed")
    return True


def _state_blockstate_signature(
    blockstate: Any, properties: Mapping[str, str]
) -> tuple[set[str], dict[str, Any] | None]:
    """Resolve a state to the exact variant/multipart selectors in one JAR JSON."""
    if not isinstance(blockstate, Mapping):
        raise FrozenPlaneEvaluationError("vanilla blockstate root must be an object")
    referenced: set[str] = set()
    signature: dict[str, Any] = {}
    has_selector_data = False

    if "variants" in blockstate:
        variants = blockstate["variants"]
        if not isinstance(variants, Mapping) or not variants:
            raise FrozenPlaneEvaluationError("vanilla blockstate variants are malformed")
        has_selector_data = True
        matches: list[tuple[str, Any]] = []
        for raw_key, apply in variants.items():
            selector = _variant_selector(raw_key)
            referenced.update(selector)
            if all(properties.get(name) == value for name, value in selector.items()):
                matches.append((raw_key, apply))
        if len(matches) > 1:
            raise FrozenPlaneEvaluationError(
                "vanilla blockstate variant selectors are ambiguous for the state"
            )
        if not matches:
            return referenced, None
        key, apply = matches[0]
        signature["variant"] = {
            "key": key,
            "apply_sha256": _sha256(_canonical_json(apply)),
        }

    if "multipart" in blockstate:
        multipart = blockstate["multipart"]
        if not isinstance(multipart, list) or not multipart:
            raise FrozenPlaneEvaluationError("vanilla blockstate multipart is malformed")
        has_selector_data = True
        active_parts = []
        for index, part in enumerate(multipart):
            if not isinstance(part, Mapping) or "apply" not in part:
                raise FrozenPlaneEvaluationError("vanilla blockstate multipart entry is malformed")
            condition = part.get("when")
            referenced.update(_condition_property_names(condition))
            if _condition_matches(condition, properties):
                active_parts.append({
                    "index": index,
                    "apply_sha256": _sha256(_canonical_json(part["apply"])),
                })
        signature["multipart"] = active_parts

    if not has_selector_data:
        return referenced, None
    return referenced, signature


def _project_palette_states(
    states: Sequence[BlockState], cached_samples: Sequence[Any],
    exact_palette: Sequence[Any],
    *,
    blockstate_resources: Mapping[str, tuple[str, bytes, str]] | None = None,
    vanilla_jar_sha256: str | None = None,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Resolve omitted GT properties only from unique, evidence-compatible models."""
    palette_by_state = {item.state: item for item in exact_palette}
    projection: list[dict[str, Any]] = []
    samples_by_name: dict[str, list[Any]] = {}
    for sample in cached_samples:
        samples_by_name.setdefault(sample.state.name, []).append(sample)
    for state in sorted(set(states), key=lambda item: (item.name, item.properties)):
        if state in palette_by_state:
            continue
        target_properties = dict(state.properties)
        compatible = [sample for sample in samples_by_name.get(state.name, ())
                      if all(target_properties.get(key) == value
                             for key, value in sample.state.properties)]
        if len(compatible) == 1:
            sample = compatible[0]
            palette_by_state[state] = _palette_color(sample, state=state)
            projection.append({
                "target_state": state.name,
                "target_properties": dict(state.properties),
                "cached_variant_properties": dict(sample.state.properties),
                "rule": "one unique same-name cache sample whose stored blockstate-variant properties are an exact subset of GT properties",
            })
            continue
        if len(compatible) > 1:
            raise FrozenPlaneEvaluationError(
                f"ambiguous cached variants for exact GT state {state.name}{target_properties}")

        resource = (blockstate_resources or {}).get(state.name)
        if resource is None:
            raise FrozenPlaneEvaluationError(
                f"no compatible cached variant or bound vanilla blockstate asset for exact GT state {state.name}{target_properties}")
        asset_path, raw_blockstate, asset_hash = resource
        blockstate = _strict_json(raw_blockstate, f"vanilla blockstate {asset_path}")
        referenced, target_signature = _state_blockstate_signature(
            blockstate, target_properties)
        if target_signature is None:
            raise FrozenPlaneEvaluationError(
                f"vanilla blockstate has no unique selected model for exact GT state {state.name}{target_properties}")

        selector_compatible: list[tuple[Any, set[str]]] = []
        for sample in samples_by_name.get(state.name, ()):
            cached_properties = dict(sample.state.properties)
            if any(name not in target_properties or name not in cached_properties
                   or target_properties[name] != cached_properties[name]
                   for name in referenced):
                continue
            different_properties = {
                name for name in set(target_properties) | set(cached_properties)
                if target_properties.get(name) != cached_properties.get(name)
            }
            if different_properties & referenced:
                continue
            candidate_referenced, candidate_signature = _state_blockstate_signature(
                blockstate, cached_properties)
            if candidate_referenced == referenced and candidate_signature == target_signature:
                selector_compatible.append((sample, different_properties))
        if len(selector_compatible) != 1:
            reason = ("no selector-compatible cached variant" if not selector_compatible
                      else "ambiguous selector-compatible cached variants")
            raise FrozenPlaneEvaluationError(
                f"{reason} for exact GT state {state.name}{target_properties}")

        sample, ignored = selector_compatible[0]
        cached_properties = dict(sample.state.properties)
        record = {
            "target_state": state.name,
            "target_properties": dict(state.properties),
            "cached_variant_properties": cached_properties,
            "rule": "unique same-name cache sample with identical selected vanilla blockstate variant/multipart signature; every differing GT property is absent from all selector conditions",
            "blockstate_asset_path": asset_path,
            "blockstate_asset_sha256": asset_hash,
            "vanilla_jar_sha256": vanilla_jar_sha256,
            "selector_properties": sorted(referenced),
            "selector_signature": target_signature,
            "ignored_properties": [
                {"name": name, "ground_truth": target_properties.get(name),
                 "cached_variant": cached_properties.get(name)}
                for name in sorted(ignored)
            ],
        }
        if "waterlogged" in ignored:
            record["rendering_limitation"] = (
                "the baked block model projection does not render a waterlogged fluid overlay")
        palette_by_state[state] = _palette_color(sample, state=state)
        projection.append(record)
    return (tuple(palette_by_state[state] for state in sorted(set(states),
                                                              key=lambda item: (item.name, item.properties))),
            projection)


def _plane_key(cell: Sequence[int], slot: str,
               plane_offsets: Sequence[float] | None = None,
               occurrence: int = 0) -> tuple[str, float]:
    if slot not in _SLOT_AXIS_SIDE or len(cell) != 3:
        raise FrozenPlaneEvaluationError("renderer returned an unknown owner face")
    axis, side = _SLOT_AXIS_SIDE[slot]
    if plane_offsets:
        offset = plane_offsets[min(occurrence, len(plane_offsets) - 1)]
        coordinate = round(float(cell[axis]) + float(offset), 6)
    else:
        coordinate = float(int(cell[axis]) + side)
    return slot, coordinate


def _state_plane_offsets(state: BlockState, slot: str, palette: Sequence[Any]) -> list[float]:
    """Resolve sub-cell plane coordinates where exact baked geometry is unique."""
    axis, side = _SLOT_AXIS_SIDE[slot]
    item = next((entry for entry in palette if entry.state == state), None)
    if item is None:
        return []
    offsets: list[float] = []
    for record in tuple(getattr(item, "model_faces", ()) or ()):
        if record[0] == slot:
            corners = np.asarray(record[1], dtype=np.float64)
            if corners.ndim == 2 and corners.shape[0] >= 3 and corners.shape[1] == 3:
                offsets.append(round(0.5 + float(np.mean(corners[:, axis])), 6))
    if not offsets:
        for low, high in tuple(getattr(item, "shape_boxes", ()) or ()):
            boundary = high[axis] if side else low[axis]
            offsets.append(round(float(boundary), 6))
    if not offsets:
        return [float(side)]
    return offsets


def group_visible_owners(
    ownership: np.ndarray,
    owner_index: Sequence[tuple[tuple[int, int, int], str]],
    usable: np.ndarray,
    *,
    block_states: Mapping[tuple[int, int, int], BlockState] | None = None,
    palette: Sequence[Any] | None = None,
) -> dict[tuple[str, float], np.ndarray]:
    """Group renderer owner pixels by outward slot and integer lattice plane."""
    if (not isinstance(ownership, np.ndarray) or ownership.ndim != 2
            or ownership.dtype.kind not in "iu"):
        raise FrozenPlaneEvaluationError("renderer ownership must be a 2D integer array")
    if (not isinstance(usable, np.ndarray) or usable.dtype != np.bool_
            or usable.shape != ownership.shape):
        raise FrozenPlaneEvaluationError("usable mask must match renderer ownership")
    grouped: dict[tuple[str, float], np.ndarray] = {}
    face_occurrences: dict[tuple[tuple[int, int, int], str], int] = {}
    ownership_flat = ownership.reshape(-1)
    usable_indices = np.flatnonzero(usable.reshape(-1) & (ownership_flat >= 0))
    if not len(usable_indices):
        return grouped
    pixel_owners = ownership_flat[usable_indices]
    if int(pixel_owners.max()) >= len(owner_index):
        raise FrozenPlaneEvaluationError("renderer ownership refers to an unknown face")
    order = np.argsort(pixel_owners, kind="stable")
    sorted_owners = pixel_owners[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_owners)) + 1]
    ends = np.r_[starts[1:], len(sorted_owners)]
    for start, end in zip(starts, ends):
        owner_id = int(sorted_owners[start])
        cell, slot = owner_index[owner_id]
        identity = (cell, slot)
        occurrence = face_occurrences.get(identity, 0)
        face_occurrences[identity] = occurrence + 1
        offsets = None
        if block_states is not None and palette is not None and cell in block_states:
            offsets = _state_plane_offsets(block_states[cell], slot, palette)
        key = _plane_key(cell, slot, offsets, occurrence)
        pixels = grouped.get(key)
        if pixels is None:
            pixels = np.zeros(ownership.shape, dtype=bool)
            grouped[key] = pixels
        linear_pixels = usable_indices[order[start:end]]
        pixels.reshape(-1)[linear_pixels] = True
    return grouped


def group_full_cube_owners(
    ownership: np.ndarray,
    owner_index: Sequence[tuple[tuple[int, int, int], str]],
    usable: np.ndarray,
    *,
    block_states: Mapping[tuple[int, int, int], BlockState],
    palette: Sequence[Any],
) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray, dict[str, Any]]:
    """Group only proven full-cube owners at cell boundaries.

    The returned validity mask excludes pixels owned by partial/non-cube states,
    but retains usable background pixels so predictions there still count false
    positive.  This deliberately avoids assigning pixels to individual baked
    sub-faces: attribution currently exposes only (cell, slot) ownership.
    """
    if (not isinstance(ownership, np.ndarray) or ownership.ndim != 2
            or ownership.dtype.kind not in "iu"):
        raise FrozenPlaneEvaluationError("renderer ownership must be a 2D integer array")
    if (not isinstance(usable, np.ndarray) or usable.dtype != np.bool_
            or usable.shape != ownership.shape):
        raise FrozenPlaneEvaluationError("usable mask must match renderer ownership")
    palette_by_state = {item.state: item for item in palette}
    valid = usable.copy()
    grouped: dict[tuple[str, int], np.ndarray] = {}
    non_full_cube_pixels: dict[str, int] = {}
    ambiguous_pixels: dict[str, int] = {}
    ownership_flat = ownership.reshape(-1)
    usable_indices = np.flatnonzero(usable.reshape(-1) & (ownership_flat >= 0))
    if not len(usable_indices):
        return grouped, valid, {
            "excluded_non_full_cube_owner_pixel_count": 0,
            "excluded_non_full_cube_states": [],
            "potentially_ambiguous_owner_pixel_count": 0,
            "potentially_ambiguous_owner_states": [],
        }
    pixel_owners = ownership_flat[usable_indices]
    if int(pixel_owners.max()) >= len(owner_index):
        raise FrozenPlaneEvaluationError("renderer ownership refers to an unknown face")
    order = np.argsort(pixel_owners, kind="stable")
    sorted_owners = pixel_owners[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_owners)) + 1]
    ends = np.r_[starts[1:], len(sorted_owners)]
    valid_flat = valid.reshape(-1)
    for start, end in zip(starts, ends):
        owner_id = int(sorted_owners[start])
        cell, slot = owner_index[owner_id]
        state = block_states.get(cell)
        item = None if state is None else palette_by_state.get(state)
        if state is None or item is None:
            raise FrozenPlaneEvaluationError("renderer owner has no exact state/palette entry")
        linear_pixels = usable_indices[order[start:end]]
        offsets = _state_plane_offsets(state, slot, palette)
        if len(offsets) > 1:
            ambiguous_pixels[state.name] = ambiguous_pixels.get(state.name, 0) + len(linear_pixels)
        if not bool(getattr(item, "full_cube", False)):
            valid_flat[linear_pixels] = False
            non_full_cube_pixels[state.name] = (
                non_full_cube_pixels.get(state.name, 0) + len(linear_pixels))
            continue
        if slot not in _SLOT_AXIS_SIDE:
            raise FrozenPlaneEvaluationError("renderer returned an unknown owner face")
        axis, side = _SLOT_AXIS_SIDE[slot]
        plane = int(cell[axis]) + side
        key = (slot, plane)
        mask = grouped.get(key)
        if mask is None:
            mask = np.zeros(ownership.shape, dtype=bool)
            grouped[key] = mask
        mask.reshape(-1)[linear_pixels] = True
    return grouped, valid, {
        "excluded_non_full_cube_owner_pixel_count": sum(non_full_cube_pixels.values()),
        "excluded_non_full_cube_states": [
            {"state": name, "pixel_count": count}
            for name, count in sorted(non_full_cube_pixels.items())],
        "potentially_ambiguous_owner_pixel_count": sum(ambiguous_pixels.values()),
        "potentially_ambiguous_owner_states": [
            {"state": name, "pixel_count": count}
            for name, count in sorted(ambiguous_pixels.items())],
    }


def _context_pixel_domain(
    ownership: np.ndarray,
    usable: np.ndarray,
    full_cube_masks: Mapping[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the symmetric known-owner domain and its pixel accounting."""
    if (not isinstance(ownership, np.ndarray) or ownership.ndim != 2
            or ownership.dtype.kind not in "iu"):
        raise FrozenPlaneEvaluationError("renderer ownership must be a 2D integer array")
    if (not isinstance(usable, np.ndarray) or usable.dtype != np.bool_
            or usable.shape != ownership.shape):
        raise FrozenPlaneEvaluationError("usable mask must match renderer ownership")
    known = usable & (ownership >= 0)
    unknown = usable & (ownership < 0)
    positive = np.zeros(ownership.shape, dtype=bool)
    for mask in full_cube_masks.values():
        if mask.shape != ownership.shape or mask.dtype != np.bool_:
            raise FrozenPlaneEvaluationError("context truth plane dimensions are invalid")
        positive |= mask
    if np.any(positive & ~known):
        raise FrozenPlaneEvaluationError("full-cube truth escaped context-owned pixels")
    known_count = int(np.count_nonzero(known))
    positive_count = int(np.count_nonzero(positive))
    return known, {
        "image_pixels": int(ownership.size),
        "usable_pixels": int(np.count_nonzero(usable)),
        "known_context_owned_pixels": known_count,
        "unknown_no_owner_usable_pixels": int(np.count_nonzero(unknown)),
        "excluded_unknown_pixels": int(np.count_nonzero(unknown)),
        "excluded_gui_masked_pixels": int(ownership.size - np.count_nonzero(usable)),
        "full_cube_positive_pixels": positive_count,
        "other_context_owned_known_negative_pixels": known_count - positive_count,
    }


def render_truth_planes(
    truth: GroundTruthExport,
    camera: CameraModel,
    image: np.ndarray,
    *,
    vanilla_jar_path: str | Path,
    palette_cache_path: str | Path,
    gui_mask_config: GuiMaskConfig | None = None,
) -> dict[tuple[str, float], np.ndarray]:
    """Render exact GT into screenshot coordinates and return usable truth planes."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8):
        raise FrozenPlaneEvaluationError("truth renderer expects an RGB uint8 image")
    height, width = image.shape[:2]
    if width * height > MAX_EVALUATION_PIXELS:
        raise FrozenPlaneEvaluationError("truth-render image exceeds pixel cap")
    visible_records = tuple(record for record in truth.blocks if record.name not in _AIR_STATE_NAMES)
    states = tuple(BlockState(record.name, tuple(sorted(record.properties.items())))
                   for record in visible_records)
    # Deduplicate while preserving stable block-state order.
    unique_states = tuple(dict.fromkeys(states))
    blocks = {record.position: BlockState(record.name, tuple(sorted(record.properties.items())))
              for record in visible_records}
    if not blocks:
        return {}
    palette, _jar_hash, _palette_hash, _projection = _load_palette(
        Path(vanilla_jar_path), Path(palette_cache_path), unique_states)
    return _render_block_map(blocks, unique_states, truth, camera, image,
                             gui_mask_config=gui_mask_config, palette=palette)


def render_truth_evidence(
    truth: GroundTruthExport,
    camera: CameraModel,
    *,
    width: int,
    height: int,
    truth_artifact_path: str | Path,
    vanilla_jar_path: str | Path,
    palette_cache_path: str | Path,
    gui_mask_config: GuiMaskConfig | None = None,
    max_pixels: int = MAX_EVALUATION_PIXELS,
    max_blocks: int = MAX_GROUND_TRUTH_BLOCKS,
    max_artifact_bytes: int = MAX_TRUTH_ARTIFACT_BYTES,
    max_evidence_bytes: int = MAX_RENDERED_EVIDENCE_BYTES,
    ownership_witness_sink=None,
    ownership_witness_payload_sink=None,
    witness_budget=None,
) -> RenderedTruthEvidence:
    """Render immutable evaluation evidence without changing legacy APIs."""
    if ownership_witness_payload_sink is not None and len(ownership_witness_payload_sink):
        raise FrozenPlaneEvaluationError("renderer witness payload sink must be empty")
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or width * height > max_pixels):
        raise FrozenPlaneEvaluationError("rendered truth evidence exceeds pixel cap")
    if len(truth.blocks) > max_blocks:
        raise FrozenPlaneEvaluationError("rendered truth evidence exceeds block cap")
    truth_path, jar_path, palette_path = map(
        Path, (truth_artifact_path, vanilla_jar_path, palette_cache_path))
    for path, label in ((truth_path, "truth"), (jar_path, "jar"),
                        (palette_path, "palette")):
        try:
            size = path.stat().st_size
        except OSError as error:
            raise FrozenPlaneEvaluationError(f"{label} artifact could not be read") from error
        if size <= 0 or size > max_artifact_bytes and label == "truth":
            raise FrozenPlaneEvaluationError(f"{label} artifact exceeds byte cap")
    truth_raw, truth_doc = _read_json(truth_path, "truth artifact")
    if isinstance(truth_doc, Mapping) and truth_doc.get("artifact_type") == _CONTEXT_ARTIFACT_TYPE:
        comparable = {"scene_id": truth_doc.get("scene_id"),
                      "world_identity": truth_doc.get("world_identity"),
                      "region": truth_doc.get("region"), "blocks": truth_doc.get("blocks")}
    else:
        comparable = {key: truth_doc.get(key) for key in
                      ("scene_id", "world_identity", "region", "blocks")}
    expected = truth.to_dict()
    if comparable != {key: expected[key] for key in comparable}:
        raise FrozenPlaneEvaluationError("truth object does not bind its artifact")
    visible = tuple(record for record in truth.blocks if record.name not in _AIR_STATE_NAMES)
    states = tuple(dict.fromkeys(
        BlockState(record.name, tuple(sorted(record.properties.items())))
        for record in visible))
    blocks = {record.position: BlockState(
        record.name, tuple(sorted(record.properties.items()))) for record in visible}
    palette, jar_hash, palette_hash, _projection = _load_palette(
        jar_path, palette_path, states)
    scene = SceneTransform((0, 0, 0), truth.region.size)
    try:
        render_kwargs = {}
        capture_payload = ownership_witness_payload_sink is not None
        private_witness_sink = [] if (ownership_witness_sink is not None or capture_payload) else None
        if private_witness_sink is not None:
            render_kwargs.update(witness_sink=private_witness_sink,
                                 witness_budget=witness_budget)
        rendered, owner_index = attribution.render_result(
            blocks, scene, camera, width, height,
            attribution.face_colours(palette),
            shapes=attribution.face_shapes(palette), scale=1.0, ambient=False,
            occluding_slots=attribution.face_occluding_slots(palette),
            model_faces=attribution.face_model_faces(palette), **render_kwargs)
    except (cv2.error, ValueError, TypeError, IndexError, MemoryError) as error:
        raise FrozenPlaneEvaluationError("exact-state GT evidence render failed") from error
    usable = ~create_gui_mask(height, width, gui_mask_config or GuiMaskConfig())
    grouped = group_visible_owners(
        rendered.ownership, owner_index, usable, block_states=blocks, palette=palette)

    def frozen(value: np.ndarray, dtype=None) -> np.ndarray:
        result = np.asarray(value, dtype=dtype).copy(order="C")
        result.flags.writeable = False
        return result

    rgb = frozen(rendered.colour, np.uint8)
    ownership = frozen(rendered.ownership)
    usable = frozen(usable, np.bool_)
    surfaces = tuple(RenderedTruthSurface(direction, float(coordinate), frozen(mask, np.bool_))
                     for (direction, coordinate), mask in sorted(grouped.items()))
    retained = rgb.nbytes + ownership.nbytes + usable.nbytes \
        + sum(surface.mask.nbytes for surface in surfaces)
    if retained > max_evidence_bytes:
        raise FrozenPlaneEvaluationError("rendered truth evidence exceeds retained-byte cap")
    canonical_index = tuple((tuple(int(v) for v in position), str(direction))
                            for position, direction in owner_index)
    evidence = RenderedTruthEvidence(
        "mc-photogrammetry-rendered-truth-evidence-v1", camera,
        _sha256(truth_raw), jar_hash, palette_hash, rgb, ownership,
        canonical_index, surfaces, usable)
    if ownership_witness_sink is not None:
        if len(private_witness_sink) != 1:
            raise FrozenPlaneEvaluationError("renderer witness publication is incomplete")
        captured = private_witness_sink[0]
        if isinstance(captured, attribution.RendererOwnershipWitnessV1):
            ownership_witness_sink.append(
                bind_renderer_ownership_witness(evidence, captured))
        elif isinstance(captured, lattice.WitnessRefusalV1):
            ownership_witness_sink.append(captured)
        else:
            raise FrozenPlaneEvaluationError("renderer witness publication is malformed")
    if ownership_witness_payload_sink is not None:
        if len(private_witness_sink) != 1:
            raise FrozenPlaneEvaluationError("renderer witness publication is incomplete")
        captured = private_witness_sink[0]
        if isinstance(captured, attribution.RendererOwnershipWitnessV1):
            binding = bind_renderer_ownership_witness(evidence, captured)
            ownership_witness_payload_sink.append(
                RenderedTruthOwnershipWitnessPayloadV1(binding, captured))
        elif isinstance(captured, lattice.WitnessRefusalV1):
            ownership_witness_payload_sink.append(captured)
        else:
            raise FrozenPlaneEvaluationError("renderer witness publication is malformed")
    return evidence


def _render_block_map(
    blocks: Mapping[tuple[int, int, int], BlockState],
    states: Sequence[BlockState],
    truth: GroundTruthExport,
    camera: CameraModel,
    image: np.ndarray,
    *,
    gui_mask_config: GuiMaskConfig | None,
    palette: Sequence[Any] | None = None,
    provider_assets: tuple[Path, Path] | None = None,
) -> dict[tuple[str, float], np.ndarray]:
    provisional, _primary, _valid, _owner_audit = _render_block_map_evidence(
        blocks, states, truth, camera, image,
        gui_mask_config=gui_mask_config, palette=palette,
        provider_assets=provider_assets)
    return provisional


def _render_block_map_evidence(
    blocks: Mapping[tuple[int, int, int], BlockState],
    states: Sequence[BlockState],
    truth: GroundTruthExport,
    camera: CameraModel,
    image: np.ndarray,
    *,
    gui_mask_config: GuiMaskConfig | None,
    palette: Sequence[Any] | None = None,
    provider_assets: tuple[Path, Path] | None = None,
    context_scoring: bool = False,
) -> tuple[dict[tuple[str, float], np.ndarray],
           dict[tuple[str, int], np.ndarray], np.ndarray, dict[str, Any]]:
    """Render once for legacy all-face and conservative full-cube evaluations."""
    height, width = image.shape[:2]
    if not blocks:
        config = gui_mask_config or GuiMaskConfig()
        usable = ~create_gui_mask(height, width, config)
        audit = {
            "excluded_non_full_cube_owner_pixel_count": 0,
            "excluded_non_full_cube_states": [],
            "potentially_ambiguous_owner_pixel_count": 0,
            "potentially_ambiguous_owner_states": [],
        }
        if context_scoring:
            ownership = np.full((height, width), -1, dtype=np.int32)
            scoring_valid, audit["context_pixel_domain"] = _context_pixel_domain(
                ownership, usable, {})
        else:
            scoring_valid = usable
        return {}, {}, scoring_valid, audit
    if palette is None:
        if provider_assets is None:
            raise FrozenPlaneEvaluationError("renderer palette assets are required")
        palette, _jar_hash, _palette_hash, _projection = _load_palette(
            provider_assets[0], provider_assets[1], states)
    if not palette:
        raise FrozenPlaneEvaluationError("renderer has no exact-state palette")
    scene = SceneTransform((0, 0, 0), truth.region.size)
    try:
        rendered, owner_index = attribution.render_result(
            blocks, scene, camera, width, height,
            attribution.face_colours(palette),
            shapes=attribution.face_shapes(palette),
            scale=1.0, ambient=False,
            occluding_slots=attribution.face_occluding_slots(palette),
            model_faces=attribution.face_model_faces(palette),
        )
    except (cv2.error, ValueError, TypeError, IndexError, MemoryError) as error:
        raise FrozenPlaneEvaluationError("exact-state GT render failed") from error
    config = gui_mask_config or GuiMaskConfig()
    usable = ~create_gui_mask(height, width, config)
    if rendered.ownership.shape != (height, width):
        raise FrozenPlaneEvaluationError("GT render did not retain screenshot coordinates")
    provisional = group_visible_owners(rendered.ownership, owner_index, usable,
                                       block_states=blocks, palette=palette)
    primary, scoring_valid, audit = group_full_cube_owners(
        rendered.ownership, owner_index, usable, block_states=blocks, palette=palette)
    if context_scoring:
        scoring_valid, audit["context_pixel_domain"] = _context_pixel_domain(
            rendered.ownership, usable, primary)
    return provisional, primary, scoring_valid, audit


def _decode_proposal_mask(
    proposal: Mapping[str, Any], sidecar: bytes, width: int, height: int,
    usable: np.ndarray,
) -> np.ndarray:
    descriptor = proposal.get("mask")
    if not isinstance(descriptor, Mapping):
        raise FrozenPlaneEvaluationError("frozen proposal has no mask descriptor")
    if descriptor.get("dtype") != "bool" or descriptor.get("encoding") != "packbits-little":
        raise FrozenPlaneEvaluationError("frozen proposal mask encoding is unsupported")
    shape = descriptor.get("shape")
    if shape != [height, width]:
        raise FrozenPlaneEvaluationError("frozen proposal mask dimensions differ from screenshot")
    offset, length = descriptor.get("offset"), descriptor.get("length")
    area = width * height
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(length, bool) or not isinstance(length, int)
            or length != (area + 7) // 8 or offset + length > len(sidecar)):
        raise FrozenPlaneEvaluationError("frozen proposal mask sidecar range is invalid")
    payload = sidecar[offset:offset + length]
    if descriptor.get("sha256") != _sha256(payload):
        raise FrozenPlaneEvaluationError("frozen proposal mask digest differs")
    mask = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:area]
    mask = mask.reshape((height, width)).astype(bool, copy=True)
    if usable.shape != mask.shape:
        raise FrozenPlaneEvaluationError("proposal usable mask dimensions differ")
    mask &= usable
    return mask


def _mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.dtype != np.bool_ or target.dtype != np.bool_:
        raise FrozenPlaneEvaluationError("plane masks must be same-sized boolean arrays")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction)) - tp
    fn = int(np.count_nonzero(target)) - tp
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "iou": tp / union if union else 0.0,
            "prediction_pixels": int(np.count_nonzero(prediction)),
            "truth_pixels": int(np.count_nonzero(target))}


def _maximum_overlap_assignment(overlaps: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight one-to-one assignment with deterministic Hungarian ties."""
    values = np.asarray(overlaps)
    if values.ndim != 2 or values.dtype.kind not in "iu":
        raise FrozenPlaneEvaluationError("overlap matrix must be a 2D integer array")
    rows, columns = values.shape
    if not rows or not columns:
        return []
    size = max(rows, columns)
    padded = np.zeros((size, size), dtype=np.int64)
    padded[:rows, :columns] = values.astype(np.int64, copy=False)
    maximum = int(padded.max(initial=0))
    cost = maximum - padded
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [10**30] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**30
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = int(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, size + 1):
        i = p[j]
        if 1 <= i <= rows and j <= columns and int(values[i - 1, j - 1]) > 0:
            pairs.append((i - 1, j - 1))
    return sorted(pairs)


def score_frame_planes(
    proposals: Sequence[tuple[str, Mapping[str, Any], np.ndarray]],
    truth_masks: Mapping[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    """Score proposal masks against visible GT planes with one-to-one matching."""
    truths = [(f"{slot}:{coordinate}", mask) for (slot, coordinate), mask
              in sorted(truth_masks.items()) if np.any(mask)]
    prediction_items = [(proposal_id, record, mask) for proposal_id, record, mask in proposals]
    if prediction_items:
        shape = prediction_items[0][2].shape
    elif truths:
        shape = truths[0][1].shape
    else:
        shape = (1, 1)
    for _, _, mask in prediction_items:
        if mask.shape != shape or mask.dtype != np.bool_:
            raise FrozenPlaneEvaluationError("proposal masks have inconsistent shape or dtype")
    for _, mask in truths:
        if mask.shape != shape or mask.dtype != np.bool_:
            raise FrozenPlaneEvaluationError("truth masks have inconsistent shape or dtype")
    overlaps = np.zeros((len(prediction_items), len(truths)), dtype=np.int64)
    for i, (_, _, prediction) in enumerate(prediction_items):
        for j, (_, target) in enumerate(truths):
            overlaps[i, j] = np.count_nonzero(prediction & target)
    assignment = _maximum_overlap_assignment(overlaps)
    by_prediction = {i: j for i, j in assignment}
    by_truth = {j: i for i, j in assignment}
    prediction_results = []
    truth_results = []
    pairs = []
    micro = {key: 0 for key in ("tp_pixels", "fp_pixels", "fn_pixels")}
    matched_positive_predictions = set()
    matched_positive_truths = set()
    for i, (proposal_id, record, prediction) in enumerate(prediction_items):
        j = by_prediction.get(i)
        if j is None:
            metrics = _mask_metrics(prediction, np.zeros(shape, dtype=bool))
            micro["fp_pixels"] += metrics["fp_pixels"]
            prediction_results.append({"proposal_id": proposal_id, "matched_truth": None,
                                       "metrics": metrics})
            continue
        target_id, target = truths[j]
        metrics = _mask_metrics(prediction, target)
        pairs.append({"proposal_id": proposal_id, "truth_plane_id": target_id,
                      "overlap_pixels": int(overlaps[i, j]), "metrics": metrics})
        prediction_results.append({"proposal_id": proposal_id, "matched_truth": target_id,
                                   "metrics": metrics})
        micro["tp_pixels"] += metrics["tp_pixels"]
        micro["fp_pixels"] += metrics["fp_pixels"]
        micro["fn_pixels"] += metrics["fn_pixels"]
        if int(overlaps[i, j]) > 0:
            matched_positive_predictions.add(i)
            matched_positive_truths.add(j)
    for j, (truth_id, target) in enumerate(truths):
        i = by_truth.get(j)
        if i is None:
            metrics = _mask_metrics(np.zeros(shape, dtype=bool), target)
            micro["fn_pixels"] += metrics["fn_pixels"]
            truth_results.append({"truth_plane_id": truth_id, "matched_proposal": None,
                                  "metrics": metrics})
        else:
            proposal_id = prediction_items[i][0]
            truth_results.append({"truth_plane_id": truth_id, "matched_proposal": proposal_id,
                                  "metrics": _mask_metrics(prediction_items[i][2], target)})
    micro_union = micro["tp_pixels"] + micro["fp_pixels"] + micro["fn_pixels"]
    tp, fp, fn = micro["tp_pixels"], micro["fp_pixels"], micro["fn_pixels"]
    micro_summary = {**micro,
                     "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                     "iou": tp / micro_union if micro_union else 0.0}
    return {
        "matches": pairs,
        "proposals": prediction_results,
        "truth_planes": truth_results,
        "micro": micro_summary,
        "unmatched_predictions": [prediction_items[i][0] for i in range(len(prediction_items))
                                  if i not in matched_positive_predictions],
        "unmatched_truth_planes": [truths[j][0] for j in range(len(truths))
                                   if j not in matched_positive_truths],
        "one_to_one_assignment_count": len(pairs),
    }


def _recall_at_k(
    proposals: Sequence[tuple[str, Mapping[str, Any], np.ndarray]],
    truth_masks: Mapping[tuple[str, int], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for requested_k in (1, 3, 5, 10, 32):
        partial = score_frame_planes(proposals[:requested_k], truth_masks)
        total_planes = len(partial["truth_planes"])
        rows.append({
            "k": min(requested_k, len(proposals)),
            "requested_k": requested_k,
            "matched_visible_gt_planes": len(partial["matches"]),
            "visible_gt_planes": total_planes,
            "value": len(partial["matches"]) / total_planes if total_planes else None,
        })
    return rows


def _axis_line_errors(estimated: Mapping[str, Any], camera: CameraModel) -> dict[str, Any]:
    try:
        estimated_axes = axis_directions(
            float(estimated["yaw_degrees"]), float(estimated["pitch_degrees"]),
            float(estimated.get("roll_degrees", 0.0)))
        capture_axes = axis_directions(camera.yaw_degrees, camera.pitch_degrees,
                                       camera.roll_degrees)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise FrozenPlaneEvaluationError("image calibration or capture camera is malformed") from error
    if estimated_axes.shape != (3, 3) or capture_axes.shape != (3, 3):
        raise FrozenPlaneEvaluationError("camera axis basis has invalid shape")
    best: tuple[float, tuple[int, int, int], list[float]] | None = None
    for permutation in itertools.permutations(range(3)):
        errors = []
        for source_axis, target_axis in enumerate(permutation):
            dot = abs(float(np.dot(estimated_axes[source_axis], capture_axes[target_axis])))
            errors.append(math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))
        candidate = (sum(errors), permutation, errors)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    errors = best[2]
    focal_estimate = float(estimated["focal_pixels"])
    expected_intrinsics = intrinsics_for(camera, int(estimated["width"]),
                                         int(estimated["height"]))
    fx, fy = expected_intrinsics.focal_length_pixels
    focal_percent = abs(focal_estimate - fy) / fy * 100.0
    yaw_delta = abs((float(estimated["yaw_degrees"]) - camera.yaw_degrees + 45.0) % 90.0 - 45.0)
    return {
        "focal_estimate_pixels": focal_estimate,
        "expected_focal_pixels_x": float(fx),
        "expected_focal_pixels_y": float(fy),
        "focal_percent_error_vs_vertical_focal": focal_percent,
        "expected_vertical_fov_degrees": float(camera.fov_degrees),
        "focal_convention": "vertical FOV: (height/2)/tan(fov_y/2); stored capture intrinsics scaled to screenshot",
        "axis_assignment_estimated_to_capture": list(best[1]),
        "axis_line_errors_degrees": errors,
        "max_axis_line_error_degrees": max(errors),
        "median_axis_line_error_degrees": float(np.median(errors)),
        "yaw_delta_modulo_90_degrees_diagnostic_only": yaw_delta,
        "gauge_note": "Manhattan axes are unoriented and permutable; acceptance uses best absolute-dot axis-line assignment, not raw yaw/pitch.",
    }


def evaluate_frozen_planes(
    freeze_path: str | Path,
    capture_path: str | Path,
    ground_truth_path: str | Path,
    vanilla_jar_path: str | Path,
    palette_cache_path: str | Path,
    output_path: str | Path | None = None,
    *,
    context_ground_truth_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a frozen image run, then score it against explicit evaluation data."""
    # Deliberately first: invalid R1 output must prevent all capture/label reads.
    freeze = load_and_verify_frozen_image(freeze_path)
    if not isinstance(freeze, Mapping) or freeze.get("labels_present") is not False:
        raise FrozenPlaneEvaluationError("input must be a strict label-free frozen image run")
    freeze_file = Path(freeze_path)
    capture_file = Path(capture_path)
    truth_file = Path(ground_truth_path)
    context_file = (None if context_ground_truth_path is None
                    else Path(context_ground_truth_path))
    jar_file = Path(vanilla_jar_path)
    palette_file = Path(palette_cache_path)
    if Path(freeze["mask_sidecar"]["file"]).name != freeze["mask_sidecar"]["file"]:
        raise FrozenPlaneEvaluationError("verified freeze has an unsafe sidecar filename")
    try:
        freeze_bytes = freeze_file.read_bytes()
        sidecar_bytes = freeze_file.with_name(freeze["mask_sidecar"]["file"]).read_bytes()
    except (OSError, KeyError, TypeError) as error:
        raise FrozenPlaneEvaluationError("verified freeze bytes could not be reread") from error
    if _strict_json(freeze_bytes, "frozen image run") != freeze:
        raise FrozenPlaneEvaluationError("frozen image run changed after strict verification")
    if (len(sidecar_bytes) != freeze["mask_sidecar"].get("size_bytes")
            or _sha256(sidecar_bytes) != freeze["mask_sidecar"].get("sha256")):
        raise FrozenPlaneEvaluationError("frozen mask sidecar changed after strict verification")
    capture_bytes, capture_doc = _read_json(capture_file, "capture metadata")
    if not isinstance(capture_doc, Mapping):
        raise FrozenPlaneEvaluationError("capture metadata root must be an object")
    images, camera_rows, input_rows = _verify_capture_images(freeze, capture_doc, capture_file)

    # Only after every PNG matches R1 do reference labels become readable.
    truth_bytes, truth = _load_truth(truth_file)
    if capture_doc.get("scene_id") != truth.scene_id:
        raise FrozenPlaneEvaluationError("capture and ground truth scene IDs differ")
    if capture_doc.get("region") != truth.region.to_dict():
        raise FrozenPlaneEvaluationError("capture and ground truth region bounds differ")
    if capture_doc.get("world_identity") != truth.world_identity:
        raise FrozenPlaneEvaluationError("capture and ground truth world identities differ")
    context_bytes: bytes | None = None
    context_truth: GroundTruthExport | None = None
    if context_file is not None:
        context_bytes, context_truth = _load_context_truth(
            context_file, capture_sha256=_sha256(capture_bytes),
            target_sha256=_sha256(truth_bytes), capture_doc=capture_doc,
            input_rows=input_rows, target_truth=truth)
    if not isinstance(freeze.get("configuration"), Mapping):
        raise FrozenPlaneEvaluationError("freeze configuration is malformed")
    raw_gui = freeze["configuration"].get("gui_mask")
    if not isinstance(raw_gui, Mapping):
        raise FrozenPlaneEvaluationError("freeze default GUI mask configuration is absent")
    try:
        gui_config = GuiMaskConfig(**dict(raw_gui))
    except (TypeError, ValueError) as error:
        raise FrozenPlaneEvaluationError("freeze GUI mask configuration is invalid") from error
    default_gui = {field: getattr(GuiMaskConfig(), field)
                   for field in GuiMaskConfig.__dataclass_fields__}
    if dict(raw_gui) != default_gui:
        raise FrozenPlaneEvaluationError("only the default derived GUI mask is accepted")

    render_truth = context_truth or truth
    visible_records = tuple(record for record in render_truth.blocks
                            if record.name not in _AIR_STATE_NAMES)
    filtered_air_count = len(render_truth.blocks) - len(visible_records)
    states = tuple(BlockState(record.name, tuple(sorted(record.properties.items())))
                   for record in visible_records)
    unique_states = tuple(dict.fromkeys(states))
    palette, jar_hash, palette_hash, state_projection = _load_palette(
        jar_file, palette_file, unique_states)
    calibration_by_frame = []
    frame_rows = []
    for frozen_frame, camera_row, name in zip(freeze["frames"], camera_rows,
                                               [row["basename"] for row in input_rows]):
        image = images[name]
        height, width = image.shape[:2]
        usable = ~create_gui_mask(height, width, gui_config)
        camera = CameraModel.from_dict(dict(camera_row))
        try:
            provisional_truth_masks, truth_masks, scoring_valid, owner_audit = _render_block_map_evidence(
                {record.position: BlockState(record.name, tuple(sorted(record.properties.items())))
                 for record in visible_records},
                unique_states, render_truth, camera, image, gui_mask_config=gui_config,
                palette=palette, context_scoring=context_truth is not None)
        except (TypeError, ValueError, IndexError, OverflowError) as error:
            raise FrozenPlaneEvaluationError(f"truth renderer failed for {name!r}") from error
        raw_proposals = frozen_frame.get("proposal", {}).get("proposals", [])
        if not isinstance(raw_proposals, list):
            raise FrozenPlaneEvaluationError(f"frozen proposal list malformed for {name!r}")
        proposal_masks = []
        for index, proposal in enumerate(raw_proposals):
            if not isinstance(proposal, Mapping):
                raise FrozenPlaneEvaluationError("frozen proposal record is malformed")
            mask = _decode_proposal_mask(proposal, sidecar_bytes, width, height, usable)
            proposal_id = f"{name}::proposal-{index:02d}"
            proposal_masks.append((proposal_id, proposal, mask))
        primary_proposal_masks = [
            (proposal_id, record, np.logical_and(mask, scoring_valid))
            for proposal_id, record, mask in proposal_masks]
        calibration = frozen_frame.get("calibration")
        calibration_metric = (None if not isinstance(calibration, Mapping)
                              else _axis_line_errors(calibration, camera))
        calibration_by_frame.append(calibration_metric)
        scored = score_frame_planes(primary_proposal_masks, truth_masks)
        recall_at_k = _recall_at_k(primary_proposal_masks, truth_masks)
        provisional_proposals = (primary_proposal_masks if context_truth is not None
                                 else proposal_masks)
        provisional_scored = score_frame_planes(provisional_proposals, provisional_truth_masks)
        provisional_recall_at_k = _recall_at_k(provisional_proposals, provisional_truth_masks)
        # An edge-touching GT render can indicate a cropped target, but never
        # proves anything about exterior occlusion absent from the GT volume.
        union_truth = np.zeros((height, width), dtype=bool)
        for mask in truth_masks.values():
            union_truth |= mask
        touches_edge = bool(union_truth[0, :].any() or union_truth[-1, :].any()
                            or union_truth[:, 0].any() or union_truth[:, -1].any())
        frame_rows.append({
            "frame_id": name,
            "image": {"width": width, "height": height,
                      "sha256": input_rows[len(frame_rows)]["sha256"]},
            "automatic_proposal_count": len(proposal_masks),
            "visible_gt_plane_count": len(truth_masks),
            "proposal_recall_at_k": recall_at_k,
            "scoring": scored,
            "context_pixel_domain": owner_audit.get("context_pixel_domain"),
            "provisional_owner_face_ambiguity": {
                "reason": "legacy all-model per-subface grouping can shift when an earlier same-slot polygon is culled or fully occluded; use full_cube_only as the primary score",
                "visible_gt_plane_count": len(provisional_truth_masks),
                "proposal_recall_at_k": provisional_recall_at_k,
                "scoring": provisional_scored,
                **owner_audit,
            },
            "calibration_comparison": calibration_metric,
            "rendered_gt_touches_frame_edge": touches_edge,
        })

    all_metrics = [frame["scoring"]["micro"] for frame in frame_rows]
    total_tp = sum(row["tp_pixels"] for row in all_metrics)
    total_fp = sum(row["fp_pixels"] for row in all_metrics)
    total_fn = sum(row["fn_pixels"] for row in all_metrics)
    total_union = total_tp + total_fp + total_fn
    provisional_metrics = [frame["provisional_owner_face_ambiguity"]["scoring"]["micro"]
                           for frame in frame_rows]
    provisional_tp = sum(row["tp_pixels"] for row in provisional_metrics)
    provisional_fp = sum(row["fp_pixels"] for row in provisional_metrics)
    provisional_fn = sum(row["fn_pixels"] for row in provisional_metrics)
    provisional_union = provisional_tp + provisional_fp + provisional_fn
    calibration_rows = [row for row in calibration_by_frame if row is not None]
    focal_errors = [row["focal_percent_error_vs_vertical_focal"] for row in calibration_rows]
    angular_errors = [angle for row in calibration_rows
                      for angle in row["axis_line_errors_degrees"]]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "",
        "inputs": {
            "freeze_artifact_id": freeze["artifact_id"],
            "freeze_json_sha256": _sha256(freeze_bytes),
            "freeze_mask_sidecar_sha256": freeze["mask_sidecar"]["sha256"],
            "capture_metadata_basename": capture_file.name,
            "capture_metadata_sha256": _sha256(capture_bytes),
            "ground_truth_basename": truth_file.name,
            "ground_truth_sha256": _sha256(truth_bytes),
            "context_ground_truth_basename": None if context_file is None else context_file.name,
            "context_ground_truth_sha256": None if context_bytes is None else _sha256(context_bytes),
            "ground_truth_provenance": {
                "target": {
                    "basename": truth_file.name,
                    "sha256": _sha256(truth_bytes),
                    "scene_id": truth.scene_id,
                    "world_identity": truth.world_identity,
                    "region": truth.region.to_dict(),
                    "block_record_count": len(truth.blocks),
                },
                "context": (None if context_truth is None or context_file is None
                            else {
                                "artifact_type": _CONTEXT_ARTIFACT_TYPE,
                                "basename": context_file.name,
                                "sha256": _sha256(context_bytes or b""),
                                "scene_id": context_truth.scene_id,
                                "world_identity": context_truth.world_identity,
                                "target_region": truth.region.to_dict(),
                                "region": context_truth.region.to_dict(),
                                "block_record_count": len(context_truth.blocks),
                                "bound_capture_metadata_sha256": _sha256(capture_bytes),
                                "bound_target_ground_truth_sha256": _sha256(truth_bytes),
                                "ordered_image_count": len(input_rows),
                            }),
            },
            "vanilla_jar_basename": jar_file.name,
            "vanilla_jar_sha256": jar_hash,
            "palette_cache_basename": palette_file.name,
            "palette_cache_sha256": palette_hash,
            "frames": input_rows,
        },
        "method": {
            "automatic_inputs": ["immutable image-only proposal masks"],
            "ground_truth_mode": ("context_region_known_owner_domain" if context_truth is not None
                                  else "target_bounded_legacy_provisional"),
            "evaluation_only_inputs": ["capture cameras", "exact-state ground truth",
                                       "vanilla model/texture cache"],
            "render": "export.attribution.render_result at scale 1 in original screenshot coordinates",
            "primary_truth_planes": "full_cube_only; identify exact owner state by cell, require palette full_cube=true, and group visible owners by slot at the cell boundary",
            "provisional_truth_planes": "legacy all-model grouping by face slot and baked sub-face coordinate; repeated offsets may be ambiguous when earlier same-slot polygons are culled or occluded",
            "matching": "maximum total overlap one-to-one assignment, independently per frame",
            "primary_scoring_validity": (
                "usable pixels with a rendered nearest owner from the bound context are known; full-cube owner face pixels are positive planes and all other context-owned pixels are known negatives; no-owner usable pixels are unknown and excluded from both prediction and truth"
                if context_truth is not None else
                "default derived usable pixels minus pixels owned by states whose palette entry does not prove full_cube; background remains valid for false-positive scoring"),
            "provisional_mask": ("frozen proposal mask intersected with the known context-owned usable-pixel domain"
                                 if context_truth is not None else
                                 "frozen proposal mask intersected with the default derived usable-pixel mask"),
            "proposal_masks_mutated_or_recomputed": False,
            "manual_masks_or_camera_inputs": False,
            "explicit_air_records_filtered": filtered_air_count,
            "ground_truth_records_in_export": len(render_truth.blocks),
            "target_ground_truth_records_in_export": len(truth.blocks),
            "context_ground_truth_records_in_export": (None if context_truth is None
                                                        else len(context_truth.blocks)),
            "unique_render_compatible_projection": state_projection,
        },
        "scoring_domain_summary": (None if context_truth is None else {
            key: sum(frame["context_pixel_domain"][key] for frame in frame_rows)
            for key in ("image_pixels", "usable_pixels", "known_context_owned_pixels",
                        "unknown_no_owner_usable_pixels", "excluded_unknown_pixels",
                        "excluded_gui_masked_pixels", "full_cube_positive_pixels",
                        "other_context_owned_known_negative_pixels")
        }),
        "calibration_summary": {
            "frames_compared": len(calibration_rows),
            "focal_error_median_percent": float(np.median(focal_errors)) if focal_errors else None,
            "focal_error_le_2_percent_count": sum(value <= 2.0 for value in focal_errors),
            "axis_line_error_median_degrees": float(np.median(angular_errors)) if angular_errors else None,
            "max_axis_line_error_le_0_5_degrees_count": sum(
                row["max_axis_line_error_degrees"] <= 0.5 for row in calibration_rows),
            "gauge": "best of all 3! axis permutations with absolute dot products; raw yaw is diagnostic only",
        },
        "frames": frame_rows,
        "micro_totals": {
            "tp_pixels": total_tp, "fp_pixels": total_fp, "fn_pixels": total_fn,
            "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
            "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
            "f1": 2 * total_tp / (2 * total_tp + total_fp + total_fn)
            if 2 * total_tp + total_fp + total_fn else 0.0,
            "iou": total_tp / total_union if total_union else 0.0,
        },
        "provisional_owner_face_ambiguity": {
            "reason": "legacy all-model sub-face grouping has no polygon identity in renderer ownership and can misassign repeated same-slot offsets",
            "micro_totals": {
                "tp_pixels": provisional_tp,
                "fp_pixels": provisional_fp,
                "fn_pixels": provisional_fn,
                "precision": provisional_tp / (provisional_tp + provisional_fp)
                if provisional_tp + provisional_fp else 0.0,
                "recall": provisional_tp / (provisional_tp + provisional_fn)
                if provisional_tp + provisional_fn else 0.0,
                "f1": 2 * provisional_tp / (2 * provisional_tp + provisional_fp + provisional_fn)
                if 2 * provisional_tp + provisional_fp + provisional_fn else 0.0,
                "iou": provisional_tp / provisional_union if provisional_union else 0.0,
            },
            "affected_non_full_cube_owner_pixel_count": sum(
                frame["provisional_owner_face_ambiguity"]["excluded_non_full_cube_owner_pixel_count"]
                for frame in frame_rows),
            "potentially_ambiguous_owner_pixel_count": sum(
                frame["provisional_owner_face_ambiguity"]["potentially_ambiguous_owner_pixel_count"]
                for frame in frame_rows),
            "promotion_eligible": False,
        },
        "promotion": {
            "promotion_eligible": False,
            "reason": ("the supplied context is finite and boundary/exterior-occluder completeness is not proven; rendered masks remain an evaluation proxy"
                       if context_truth is not None else
                       "the exported target GT is region-bounded and omits exterior world occluders; rendered masks are not occluder-correct visibility truth"),
            "cropped_occluder_warning": True,
            "context_boundary_completeness_proven": False,
            "rendered_target_touches_frame_edge_count": (None if context_truth is not None else sum(
                bool(row["rendered_gt_touches_frame_edge"]) for row in frame_rows)),
            "rendered_context_touches_frame_edge_count": (None if context_truth is None else sum(
                bool(row["rendered_gt_touches_frame_edge"]) for row in frame_rows)),
        },
        "code_hashes": _code_hashes(),
    }
    document["artifact_id"] = _artifact_id(document)
    encoded = _canonical_json(document) + b"\n"
    if output_path is not None:
        _atomic_write(Path(output_path), encoded)
    return document


__all__ = [
    "FrozenPlaneEvaluationError", "evaluate_frozen_planes", "group_full_cube_owners",
    "group_visible_owners",
    "render_truth_planes", "score_frame_planes",
]
