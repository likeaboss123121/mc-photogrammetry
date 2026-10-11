"""Evaluation-only, blinded raster landmark ingestion and analytic camera QA.

No image/world loader, renderer hook, camera fit or production camera utility is
used here. A caller must independently bind a *fresh* PNG and externally freeze
the landmark plan and manual observation bytes before exposing any projections.
The blinding/measurement declarations are auditable attestations, not something
this mathematical validator can prove from coordinates alone.

Camera PASS certifies only the fixed observed-landmark checks. No supported
outside-occluder completeness adapter exists yet: certified visibility always
refuses, including for a passing camera. Historical factory exports stay outside
this API and retain their historical conditional, unqualified semantics.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PLAN_SCHEMA = "camera-landmark-plan-v1"
PACKET_SCHEMA = "camera-landmark-observations-v1"
RECEIPT_SCHEMA = "camera-qualification-v1"
PROTOCOL = "blinded-native-png-landmarks-v1"
PIXEL_CONVENTION = "top-left-origin-integer-pixel-centers"
CAMERA_CONVENTION = "right-handed-x-right-y-up-z-back"
MAX_PACKET_BYTES = 64 * 1024
MAX_LANDMARKS = 32
MAX_JSON_DEPTH = 12
MAX_JSON_TOKENS = 4096
MAX_DIMENSION = 16384
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_COORDINATE = 30_000_000.0
MAX_CODE_BYTES = 256 * 1024

# Fixed before future acquisition; these are development QA criteria, not
# empirically validated native accuracy thresholds. They are not caller options.
POLICY = (
    ("version", "camera-landmark-policy-v1"),
    ("minimum_observed", 8), ("coverage_grid", 4), ("minimum_coverage_bins", 4),
    ("minimum_distinct_surfaces", 2), ("minimum_depth_separation_blocks", 0.1),
    ("maximum_uncertainty_pixels", 1.0), ("maximum_median_residual_pixels", 1.0),
    ("maximum_residual_pixels", 2.0), ("rigidity_tolerance", 1e-6),
    ("minimum_positive_depth", 1e-6), ("minimum_pair_baseline_blocks", 1e-6),
    ("noncoplanarity_relative_tolerance", 1e-6),
    ("pixel_convention", PIXEL_CONVENTION), ("camera_convention", CAMERA_CONVENTION),
    ("visibility_adapter", "UNAVAILABLE"),
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


POLICY_SHA256 = hashlib.sha256(_canonical(dict(POLICY))).hexdigest()


class QualificationError(ValueError):
    """A typed, fail-closed contract or certification refusal."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError(code)


def _sha(value: object) -> str:
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid_sha256")
    return value


def _text(value: object, limit: int = 128) -> str:
    _require(type(value) is str and 0 < len(value) <= limit
             and all(ord(c) >= 32 for c in value), "invalid_text")
    return value


def _number(value: object, limit: float = MAX_COORDINATE) -> float:
    _require(type(value) in (int, float), "invalid_number")
    _require(math.isfinite(value) and abs(value) <= limit, "invalid_number")
    return float(value)


def _integer(value: object, maximum: int) -> int:
    _require(type(value) is int and 0 < value <= maximum, "invalid_integer")
    return value


def _keys(value: object, names: tuple[str, ...]) -> dict:
    _require(type(value) is dict and set(value) == set(names), "invalid_fields")
    return value


def _vector(value: object, count: int, limit: float = MAX_COORDINATE) -> tuple[float, ...]:
    _require(type(value) in (list, tuple) and len(value) == count, "invalid_vector")
    return tuple(_number(x, limit) for x in value)


def _load(raw: bytes, expected_sha256: str) -> dict:
    _sha(expected_sha256)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_PACKET_BYTES, "packet_byte_cap")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "packet_digest_mismatch")
    # Bound depth/tokens BEFORE json allocates a recursive graph. The 64KiB
    # input bound also limits strings/scalars; strict schema limits retained data.
    depth = tokens = 0
    quoted = escaped = False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
            continue
        if char == 34:
            quoted = True
            tokens += 1
        elif char in (123, 91):
            depth += 1
            tokens += 1
        elif char in (125, 93):
            depth -= 1
        elif char in (44, 58):
            tokens += 1
        _require(0 <= depth <= MAX_JSON_DEPTH, "packet_depth_cap")
        _require(tokens <= MAX_JSON_TOKENS, "packet_token_cap")

    def pairs(rows):
        result = {}
        for key, item in rows:
            _require(key not in result, "duplicate_json_key")
            result[key] = item
        return result

    def integer(token):
        _require(len(token) <= 20, "integer_text_cap")
        return int(token)

    def floating(token):
        _require(len(token) <= 64, "float_text_cap")
        value = float(token)
        _require(math.isfinite(value), "invalid_number")
        return value

    def constant(_token):
        raise QualificationError("invalid_number")

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                              parse_int=integer, parse_float=floating,
                              parse_constant=constant)
    except QualificationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise QualificationError("invalid_json") from exc
    _require(type(document) is dict, "invalid_fields")
    return document


@dataclass(frozen=True)
class FrameBinding:
    frame_id: str
    png_sha256: str
    width: int
    height: int
    capture_id: str
    world_sha256: str
    model_sha256: str
    context_sha256: str


def _binding(value: FrameBinding) -> None:
    _require(type(value) is FrameBinding, "invalid_frame_binding")
    _text(value.frame_id)
    _text(value.capture_id)
    for digest in (value.png_sha256, value.world_sha256, value.model_sha256, value.context_sha256):
        _sha(digest)
    _integer(value.width, MAX_DIMENSION)
    _integer(value.height, MAX_DIMENSION)
    _require(value.width * value.height <= MAX_IMAGE_PIXELS, "image_dimension_cap")


@dataclass(frozen=True)
class Landmark:
    landmark_id: str
    world_xyz: tuple[float, ...]
    surface_id: str
    structural_identity: str
    identity_cue: str
    support_class: str


@dataclass(frozen=True)
class LandmarkPlan:
    sha256: str
    world_sha256: str
    model_sha256: str
    context_sha256: str
    landmarks: tuple[Landmark, ...]
    raw: bytes = field(repr=False)


def load_landmark_plan(raw: bytes, *, expected_sha256: str) -> LandmarkPlan:
    """Ingest a preregistered plan containing identities/XYZ, never predicted UV."""
    doc = _keys(_load(raw, expected_sha256), (
        "schema", "evaluation_only", "policy_sha256", "world_sha256", "model_sha256",
        "context_sha256", "landmarks"))
    _require(doc["schema"] == PLAN_SCHEMA and doc["evaluation_only"] is True, "plan_schema")
    _require(doc["policy_sha256"] == POLICY_SHA256, "policy_mismatch")
    for key in ("world_sha256", "model_sha256", "context_sha256"):
        _sha(doc[key])
    rows = doc["landmarks"]
    _require(type(rows) is list and 1 <= len(rows) <= MAX_LANDMARKS, "landmark_count_cap")
    result = []
    seen = set()
    positions = set()
    for row in rows:
        _keys(row, ("landmark_id", "world_xyz", "surface_id", "structural_identity",
                    "identity_cue", "support_class", "required"))
        identity = _text(row["landmark_id"])
        _require(identity not in seen, "duplicate_landmark")
        seen.add(identity)
        _require(row["required"] is True, "optional_landmark_forbidden")
        _require(row["support_class"] in ("natural_corner", "natural_edge_intersection"),
                 "unsupported_landmark")
        xyz = _vector(row["world_xyz"], 3)
        _require(xyz not in positions, "duplicate_world_landmark")
        positions.add(xyz)
        result.append(Landmark(identity, xyz,
                               _text(row["surface_id"]), _text(row["structural_identity"]),
                               _text(row["identity_cue"], 512), row["support_class"]))
    return LandmarkPlan(expected_sha256, doc["world_sha256"], doc["model_sha256"],
                        doc["context_sha256"], tuple(result), raw)


def _verify_plan(plan: LandmarkPlan, expected_sha256: str) -> None:
    _require(type(plan) is LandmarkPlan and plan.sha256 == _sha(expected_sha256),
             "landmark_plan_mismatch")
    _require(load_landmark_plan(plan.raw, expected_sha256=expected_sha256) == plan,
             "landmark_plan_tampered")


@dataclass(frozen=True)
class Observation:
    landmark_id: str
    disposition: str
    uv: tuple[float, ...] | None
    uncertainty_pixels: float | None
    support_segments: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    reason: str | None
    second_observer_id: str | None
    second_uv: tuple[float, ...] | None
    second_uncertainty_pixels: float | None


@dataclass(frozen=True)
class SealedObservations:
    sha256: str
    plan_sha256: str
    binding: FrameBinding
    observer_id: str
    tool_version: str
    observations: tuple[Observation, ...]
    raw: bytes = field(repr=False)


def _pixel(value: object, binding: FrameBinding) -> tuple[float, ...]:
    uv = _vector(value, 2, MAX_DIMENSION)
    _require(0 <= uv[0] <= binding.width - 1 and 0 <= uv[1] <= binding.height - 1,
             "pixel_out_of_bounds")
    return uv


def seal_observations(raw: bytes, plan: LandmarkPlan, *, expected_packet_sha256: str,
                      expected_binding: FrameBinding) -> SealedObservations:
    """Validate externally frozen manual observations; no camera is accepted.

    The caller independently anchors expected_packet_sha256 BEFORE exposing
    projections/residuals. A declaration/hash cannot prove an observer was blind.
    This function performs no image read and cannot manufacture an observation.
    """
    _binding(expected_binding)
    _verify_plan(plan, plan.sha256)
    doc = _keys(_load(raw, expected_packet_sha256), (
        "schema", "evaluation_only", "method", "observer_id", "tool_version", "protocol",
        "pixel_convention", "projected_coordinates_exposed", "frozen_before_projection",
        "landmark_plan_sha256", "policy_sha256", "frame_id", "png_sha256", "dimensions",
        "capture_id", "world_sha256", "model_sha256", "context_sha256", "observations"))
    _require(doc["schema"] == PACKET_SCHEMA and doc["evaluation_only"] is True,
             "observation_schema")
    _require(doc["method"] == "manual_png_observation", "observation_source_forbidden")
    _require(doc["protocol"] == PROTOCOL and doc["pixel_convention"] == PIXEL_CONVENTION,
             "observation_convention")
    _require(doc["projected_coordinates_exposed"] is False
             and doc["frozen_before_projection"] is True, "observation_not_blinded")
    _require(doc["landmark_plan_sha256"] == plan.sha256, "landmark_plan_mismatch")
    _require(doc["policy_sha256"] == POLICY_SHA256, "policy_mismatch")
    for key in ("frame_id", "png_sha256", "capture_id", "world_sha256", "model_sha256",
                "context_sha256"):
        _require(doc[key] == getattr(expected_binding, key), "frame_binding_mismatch")
    _require(type(doc["dimensions"]) is list
             and len(doc["dimensions"]) == 2
             and all(type(x) is int for x in doc["dimensions"])
             and doc["dimensions"] == [expected_binding.width, expected_binding.height],
             "frame_binding_mismatch")
    _require((plan.world_sha256, plan.model_sha256, plan.context_sha256)
             == (expected_binding.world_sha256, expected_binding.model_sha256,
                 expected_binding.context_sha256), "plan_context_mismatch")
    observer = _text(doc["observer_id"])
    tool = _text(doc["tool_version"])
    rows = doc["observations"]
    _require(type(rows) is list and len(rows) == len(plan.landmarks), "observation_membership")
    observations = []
    for landmark, row in zip(plan.landmarks, rows):
        _require(type(row) is dict and row.get("landmark_id") == landmark.landmark_id,
                 "observation_membership")
        disposition = row.get("disposition")
        _require(disposition in ("observed", "ambiguous", "occluded", "out_of_frame",
                                 "unresolvable"), "invalid_disposition")
        if disposition != "observed":
            _keys(row, ("landmark_id", "disposition", "reason"))
            observations.append(Observation(landmark.landmark_id, disposition, None, None,
                                            (), _text(row["reason"], 512), None, None, None))
            continue
        _keys(row, ("landmark_id", "disposition", "uv", "uncertainty_pixels",
                    "uncertainty_method", "support_segments", "second_annotation"))
        uv = _pixel(row["uv"], expected_binding)
        uncertainty = _number(row["uncertainty_pixels"], MAX_DIMENSION)
        _require(uncertainty >= 0, "invalid_uncertainty")
        _require(row["uncertainty_method"] == "manual_localization_bound",
                 "uncertainty_method")
        segments = row["support_segments"]
        _require(type(segments) is list and 1 <= len(segments) <= 4, "raster_support_cap")
        support = []
        for segment in segments:
            _require(type(segment) is list and len(segment) == 2, "invalid_raster_support")
            a, b = (_pixel(point, expected_binding) for point in segment)
            _require(a != b, "invalid_raster_support")
            support.append((a, b))
        second_id = second_uv = second_uncertainty = None
        if row["second_annotation"] is not None:
            second = _keys(row["second_annotation"], ("observer_id", "uv", "uncertainty_pixels"))
            second_id = _text(second["observer_id"])
            _require(second_id != observer, "second_observer_not_independent")
            second_uv = _pixel(second["uv"], expected_binding)
            second_uncertainty = _number(second["uncertainty_pixels"], MAX_DIMENSION)
            _require(second_uncertainty >= 0, "invalid_uncertainty")
        observations.append(Observation(landmark.landmark_id, disposition, uv, uncertainty,
                                        tuple(support), None, second_id, second_uv, second_uncertainty))
    return SealedObservations(expected_packet_sha256, plan.sha256, expected_binding,
                              observer, tool, tuple(observations), raw)


@dataclass(frozen=True)
class CameraHypothesis:
    """A hypothesis to test, NOT observed pixels or trusted camera ground truth.

    Rotation maps absolute world-minus-eye to proper RH camera axes x-right,
    y-up, z-back. Front has z<0; integer pixel centres use v=cy-fy*y/depth.
    Initial supported viewport is the whole native PNG, with no effects/crop.
    """
    binding: FrameBinding
    eye: tuple[float, ...]
    rotation: tuple[tuple[float, ...], ...]
    focal_pixels: tuple[float, ...]
    principal_point: tuple[float, ...]
    viewport: tuple[int, ...]
    convention: str = CAMERA_CONVENTION
    effects: tuple[str, ...] = ()
    source: str = "evaluation_camera_hypothesis"


def _det(rows: tuple[tuple[float, ...], ...]) -> float:
    a, b, c = rows
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _camera(candidate: CameraHypothesis, binding: FrameBinding) -> None:
    _require(type(candidate) is CameraHypothesis, "invalid_camera")
    _binding(candidate.binding)
    _require(candidate.binding == binding, "camera_frame_mismatch")
    _require(candidate.convention == CAMERA_CONVENTION, "unsupported_camera_convention")
    _require(candidate.source == "evaluation_camera_hypothesis", "unsupported_camera_source")
    _require(type(candidate.effects) is tuple and candidate.effects == (), "unsupported_camera_effects")
    _require(type(candidate.viewport) is tuple and len(candidate.viewport) == 4
             and all(type(x) is int for x in candidate.viewport)
             and candidate.viewport == (0, 0, binding.width, binding.height), "unsupported_viewport")
    _require(type(candidate.eye) is tuple, "invalid_camera")
    _vector(candidate.eye, 3)
    _require(type(candidate.rotation) is tuple and len(candidate.rotation) == 3,
             "invalid_rotation")
    for row in candidate.rotation:
        _require(type(row) is tuple, "invalid_rotation")
        _vector(row, 3, 1.0 + 1e-6)
    _require(abs(_det(candidate.rotation) - 1.0) <= 1e-6, "improper_rotation")
    for i in range(3):
        for j in range(3):
            dot = sum(candidate.rotation[i][k] * candidate.rotation[j][k] for k in range(3))
            _require(abs(dot - (1.0 if i == j else 0.0)) <= 1e-6, "nonrigid_rotation")
    _require(type(candidate.focal_pixels) is tuple and type(candidate.principal_point) is tuple,
             "invalid_intrinsics")
    fx, fy = _vector(candidate.focal_pixels, 2, 1e8)
    _require(fx > 0 and fy > 0, "invalid_intrinsics")
    cx, cy = _vector(candidate.principal_point, 2, MAX_DIMENSION)
    _require(0 <= cx <= binding.width - 1 and 0 <= cy <= binding.height - 1,
             "invalid_intrinsics")


def project_hypothesis(candidate: CameraHypothesis, xyz: tuple[float, ...]) -> tuple[float, float, float]:
    """Independent pinhole equation; no camera fitting, production imports or IO."""
    _camera(candidate, candidate.binding)
    point = _vector(xyz, 3)
    x, y, z = _camera_coordinates(candidate, point)
    depth = -z
    _require(depth > 1e-6, "nonpositive_depth")
    u = candidate.principal_point[0] + candidate.focal_pixels[0] * x / depth
    v = candidate.principal_point[1] - candidate.focal_pixels[1] * y / depth
    _require(math.isfinite(u) and math.isfinite(v), "invalid_projection")
    return u, v, depth


def _camera_coordinates(candidate: CameraHypothesis, xyz: tuple[float, ...]) -> tuple[float, ...]:
    relative = tuple(xyz[k] - candidate.eye[k] for k in range(3))
    return tuple(sum(row[k] * relative[k] for k in range(3)) for row in candidate.rotation)


@dataclass(frozen=True)
class LandmarkResidual:
    landmark_id: str
    disposition: str
    observed_uv: tuple[float, ...] | None
    projected_uv: tuple[float, ...] | None
    signed_depth: float | None
    residual_pixels: float | None
    uncertainty_pixels: float | None
    residual_lower_bound: float | None
    residual_upper_bound: float | None
    reason: str | None


@dataclass(frozen=True)
class CameraQAReceipt:
    schema: str
    status: str
    reasons: tuple[str, ...]
    binding: FrameBinding
    plan_sha256: str
    observation_sha256: str | None
    policy_sha256: str
    validator_sha256: str
    candidate_sha256: str
    residuals: tuple[LandmarkResidual, ...]
    observed_count: int
    coverage_bins: tuple[tuple[int, int], ...]
    median_residual_pixels: float | None
    maximum_residual_pixels: float | None
    visibility_status: str = "UNAVAILABLE"
    visibility_reason: str = "complete_outside_occluder_context_unverified"


def validator_code_sha256() -> str:
    """Bounded source identity only; this function reads no dataset payload."""
    with Path(__file__).open("rb") as stream:
        source = stream.read(MAX_CODE_BYTES + 1)
    _require(len(source) <= MAX_CODE_BYTES, "validator_code_cap")
    return hashlib.sha256(source).hexdigest()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _noncoplanar(landmarks: tuple[Landmark, ...]) -> bool:
    if len(landmarks) < 4:
        return False
    origin = landmarks[0].world_xyz
    vectors = tuple(tuple(item.world_xyz[k] - origin[k] for k in range(3))
                    for item in landmarks[1:])
    scale = max(math.sqrt(sum(x * x for x in vector)) for vector in vectors)
    if scale == 0:
        return False
    return any(abs(_det(rows)) > 1e-6 * scale ** 3
               for rows in itertools.combinations(vectors, 3))


def assess_camera(plan: LandmarkPlan, observations: SealedObservations | None,
                  candidate: CameraHypothesis, *, expected_binding: FrameBinding,
                  expected_validator_sha256: str, expected_plan_sha256: str,
                  expected_observation_sha256: str | None) -> CameraQAReceipt:
    """Test fixed hypothesis against frozen pixels; never fit or exclude an ID.

    Pixel PASS is separate from visibility certification, which stays unavailable.
    Caller supplies externally verified PNG/frame bindings and source-code pin.
    A synthetic packet can test math but is never evidence of a real acquisition.
    """
    _binding(expected_binding)
    code_pin = validator_code_sha256()
    _require(_sha(expected_validator_sha256) == code_pin, "validator_code_mismatch")
    _verify_plan(plan, expected_plan_sha256)
    _require((plan.world_sha256, plan.model_sha256, plan.context_sha256)
             == (expected_binding.world_sha256, expected_binding.model_sha256,
                 expected_binding.context_sha256), "plan_context_mismatch")
    _camera(candidate, expected_binding)
    candidate_hash = _candidate_sha256(candidate)
    residuals = []
    bins = set()
    failures = []
    unavailable = []
    depths = []
    observed_count = 0
    observed_pixels = set()
    if observations is None:
        _require(expected_observation_sha256 is None, "observation_binding_mismatch")
        unavailable.append("landmark_observations_missing")
    else:
        _require(type(observations) is SealedObservations
                 and observations.binding == expected_binding
                 and observations.plan_sha256 == plan.sha256, "observation_binding_mismatch")
        _require(observations.sha256 == _sha(expected_observation_sha256),
                 "observation_digest_mismatch")
        _require(seal_observations(observations.raw, plan,
                                  expected_packet_sha256=expected_observation_sha256,
                                  expected_binding=expected_binding) == observations,
                 "observations_tampered")
        _require(len(observations.observations) == len(plan.landmarks), "observation_membership")
        for landmark, observation in zip(plan.landmarks, observations.observations):
            _require(observation.landmark_id == landmark.landmark_id, "observation_membership")
            if observation.disposition != "observed":
                unavailable.append("landmark_" + observation.disposition)
                residuals.append(LandmarkResidual(landmark.landmark_id, observation.disposition,
                                                  None, None, None, None, None, None, None,
                                                  observation.reason))
                continue
            observed_count += 1
            u, v = observation.uv
            if observation.uv in observed_pixels:
                unavailable.append("indistinguishable_raster_landmarks")
            observed_pixels.add(observation.uv)
            bins.add((min(3, int(u * 4 / expected_binding.width)),
                      min(3, int(v * 4 / expected_binding.height))))
            uncertainty = observation.uncertainty_pixels
            if uncertainty > 1.0:
                unavailable.append("localization_uncertainty_excessive")
            if observation.second_uv is not None:
                disagreement = math.dist(observation.uv, observation.second_uv)
                if observation.second_uncertainty_pixels > 1.0:
                    unavailable.append("localization_uncertainty_excessive")
                if disagreement > uncertainty + observation.second_uncertainty_pixels:
                    unavailable.append("independent_annotations_disagree")
            try:
                projected_u, projected_v, depth = project_hypothesis(candidate, landmark.world_xyz)
            except QualificationError as error:
                failures.append(error.code)
                residuals.append(LandmarkResidual(landmark.landmark_id, "observed", observation.uv,
                                                  None, -_camera_coordinates(candidate, landmark.world_xyz)[2],
                                                  None, uncertainty, None, None,
                                                  error.code))
                continue
            depths.append(depth)
            if not (0 <= projected_u <= expected_binding.width - 1
                    and 0 <= projected_v <= expected_binding.height - 1):
                failures.append("projected_landmark_out_of_frame")
            error = math.dist((projected_u, projected_v), observation.uv)
            residuals.append(LandmarkResidual(landmark.landmark_id, "observed", observation.uv,
                                              (projected_u, projected_v), depth, error, uncertainty,
                                              max(0.0, error - uncertainty), error + uncertainty, None))
    measured = [row.residual_pixels for row in residuals if row.residual_pixels is not None]
    if observed_count < 8:
        unavailable.append("insufficient_landmarks")
    if len(bins) < 4:
        unavailable.append("insufficient_image_coverage")
    if len({item.surface_id for item in plan.landmarks}) < 2 or not _noncoplanar(plan.landmarks):
        unavailable.append("insufficient_noncoplanar_support")
    if len(depths) >= 2 and max(depths) - min(depths) < 0.1:
        unavailable.append("insufficient_depth_separation")
    if measured:
        lower = [row.residual_lower_bound for row in residuals if row.residual_pixels is not None]
        upper = [row.residual_upper_bound for row in residuals if row.residual_pixels is not None]
        if _median(lower) > 1.0 or max(lower) > 2.0:
            failures.append("residual_gate_failed")
        elif _median(upper) > 1.0 or max(upper) > 2.0:
            unavailable.append("residual_uncertainty_indeterminate")
    reasons = tuple(dict.fromkeys(failures + unavailable))
    status = "FAIL" if failures else "UNAVAILABLE" if unavailable else "PASS"
    return CameraQAReceipt(RECEIPT_SCHEMA, status, reasons, expected_binding, plan.sha256,
                           None if observations is None else observations.sha256,
                           POLICY_SHA256, code_pin, candidate_hash, tuple(residuals),
                           observed_count, tuple(sorted(bins)), _median(measured) if measured else None,
                           max(measured) if measured else None)


def _candidate_sha256(candidate: CameraHypothesis) -> str:
    binding = candidate.binding
    return hashlib.sha256(_canonical({
        "eye": candidate.eye, "rotation": candidate.rotation, "focal_pixels": candidate.focal_pixels,
        "principal_point": candidate.principal_point, "viewport": candidate.viewport,
        "convention": candidate.convention, "effects": candidate.effects, "source": candidate.source,
        "binding": (binding.frame_id, binding.png_sha256, binding.width, binding.height,
                    binding.capture_id, binding.world_sha256, binding.model_sha256, binding.context_sha256),
    })).hexdigest()


def require_certified_visibility(receipt: CameraQAReceipt) -> None:
    """There is currently no admissible complete outside-occluder proof source.

    A supplied bool/status/context hash cannot enable certification. This gate
    remains closed until a separately reviewed context verifier is implemented.
    """
    _require(type(receipt) is CameraQAReceipt and receipt.status == "PASS", "camera_not_qualified")
    raise QualificationError("complete_outside_occluder_context_unverified")


def sample_certified_visibility(receipt: CameraQAReceipt, sampler: Callable, *args, **kwargs):
    """Fail before calling any ownership sampler, even if the camera passed."""
    require_certified_visibility(receipt)
    return sampler(*args, **kwargs)


@dataclass(frozen=True)
class PairQAReceipt:
    status: str
    reasons: tuple[str, ...]
    baseline_blocks: float | None
    relative_rotation: tuple[tuple[float, ...], ...] | None
    relative_translation: tuple[float, ...] | None
    visibility_status: str = "UNAVAILABLE"


def assess_pair(first: CameraQAReceipt, first_camera: CameraHypothesis,
                second: CameraQAReceipt, second_camera: CameraHypothesis, *,
                first_plan_raw: bytes, second_plan_raw: bytes,
                first_packet_raw: bytes | None, second_packet_raw: bytes | None,
                expected_first_binding: FrameBinding, expected_second_binding: FrameBinding,
                expected_first_plan_sha256: str, expected_second_plan_sha256: str,
                expected_first_packet_sha256: str | None, expected_second_packet_sha256: str | None,
                expected_validator_sha256: str) -> PairQAReceipt:
    """Replay both externally anchored raster assessments before pair admission.

    Every input plan/observation is bounded and independently reloaded from raw
    bytes. None/None denotes missing evidence and can produce only UNAVAILABLE.
    Supplied receipts must equal COMPLETE fresh assessments, not just share a
    status or a self-computed digest. External expected anchors are mandatory;
    never derive them from the receipts/packets this boundary is qualifying.

    Relative pose uses absolute world gauge; no truth alignment/baseline reversal.

    Native correspondence support/spread/parallax/cheirality are separate gates
    and are not certified by this sparse landmark camera-only check.
    """
    _require(type(first) is CameraQAReceipt and type(second) is CameraQAReceipt, "invalid_receipt")
    _require(first.schema == second.schema == RECEIPT_SCHEMA, "receipt_schema_mismatch")
    _require(first.policy_sha256 == second.policy_sha256 == POLICY_SHA256, "policy_mismatch")
    _require(first.validator_sha256 == second.validator_sha256 == validator_code_sha256(),
             "validator_code_mismatch")
    _require(_sha(expected_validator_sha256) == validator_code_sha256(), "validator_code_mismatch")

    def replay(plan_raw, packet_raw, plan_pin, packet_pin, binding, candidate):
        plan = load_landmark_plan(plan_raw, expected_sha256=plan_pin)
        if packet_raw is None:
            _require(packet_pin is None, "observation_binding_mismatch")
            observations = None
        else:
            observations = seal_observations(packet_raw, plan, expected_packet_sha256=packet_pin,
                                            expected_binding=binding)
        return assess_camera(plan, observations, candidate, expected_binding=binding,
                             expected_validator_sha256=expected_validator_sha256,
                             expected_plan_sha256=plan_pin, expected_observation_sha256=packet_pin)

    replay_first = replay(first_plan_raw, first_packet_raw, expected_first_plan_sha256,
                          expected_first_packet_sha256, expected_first_binding, first_camera)
    replay_second = replay(second_plan_raw, second_packet_raw, expected_second_plan_sha256,
                           expected_second_packet_sha256, expected_second_binding, second_camera)
    _require(first == replay_first and second == replay_second, "pair_receipt_evidence_mismatch")
    _require(first.candidate_sha256 == _candidate_sha256(first_camera)
             and second.candidate_sha256 == _candidate_sha256(second_camera),
             "pair_camera_hypothesis_mismatch")
    if first.status != "PASS" or second.status != "PASS":
        return PairQAReceipt("UNAVAILABLE", ("pair_camera_not_qualified",), None, None, None)
    shared_first = (first.binding.capture_id, first.binding.world_sha256, first.binding.model_sha256,
                    first.binding.context_sha256, first.plan_sha256, first.policy_sha256,
                    first.validator_sha256)
    shared_second = (second.binding.capture_id, second.binding.world_sha256, second.binding.model_sha256,
                     second.binding.context_sha256, second.plan_sha256, second.policy_sha256,
                     second.validator_sha256)
    _require(shared_first == shared_second, "pair_context_mismatch")
    _require(first.binding.frame_id != second.binding.frame_id
             and first.binding.png_sha256 != second.binding.png_sha256, "pair_frame_reused")
    baseline = math.dist(first_camera.eye, second_camera.eye)
    if baseline <= 1e-6:
        return PairQAReceipt("FAIL", ("zero_baseline",), baseline, None, None)
    rotation = tuple(tuple(sum(second_camera.rotation[i][k] * first_camera.rotation[j][k]
                               for k in range(3)) for j in range(3)) for i in range(3))
    delta = tuple(first_camera.eye[k] - second_camera.eye[k] for k in range(3))
    translation = tuple(sum(row[k] * delta[k] for k in range(3)) for row in second_camera.rotation)
    return PairQAReceipt("PASS", (), baseline, rotation, translation)
