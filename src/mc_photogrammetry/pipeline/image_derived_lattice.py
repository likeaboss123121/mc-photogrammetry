"""DS-M9-C validating image-derived lattice/provenance adapter (in-memory only).

Replays caller-accepted session/pose-graph/calibration/proposal/reading/provenance
records, cross-binds the pose graph and each reading's rectification to the session
and calibration they came from, seals the readings through
``bind_geometry_provenance`` into one ``ImageDerivedLatticeResult``, and returns
``geometry_unavailable`` for an absent record or
``geometry_invalid``/``provenance_mismatch`` for a malformed or cross-bound one.
No measurement, filesystem, synthetic default, CLI/M6, or ``PlacementUniverse``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from mc_photogrammetry.pipeline.image_derived_pose_graph import ImageDerivedPoseGraph
from mc_photogrammetry.pipeline.image_derived_session import (
    SESSION_DIGEST_MISMATCH, ImageDerivedSession, ImageDerivedSessionError)
from mc_photogrammetry.reconstruction import boundary_candidates, lattice_planes
from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
from mc_photogrammetry.reconstruction.calibration import CalibrationResult, axis_directions
from mc_photogrammetry.reconstruction.plane_geometry_provenance import (
    GeometryProvenanceBridge, SplitSourceEvidence, StructuralWeightsEvidence)

__all__ = [
    "IMAGE_DERIVED_LATTICE_PRODUCER_VERSION", "IMAGE_DERIVED_LATTICE_REFUSAL_REASONS",
    "GEOMETRY_UNAVAILABLE", "ImageDerivedLatticeRefusal", "ImageDerivedLatticeResult",
    "assemble_image_derived_lattice",
]

IMAGE_DERIVED_LATTICE_PRODUCER_VERSION = "m9c-image-derived-lattice-adapter-1"
GEOMETRY_UNAVAILABLE = "geometry_unavailable"
IMAGE_DERIVED_LATTICE_REFUSAL_REASONS: tuple[str, ...] = (
    "geometry_unavailable", "geometry_invalid", "provenance_mismatch")

_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)

class _Refusal(Exception):
    """Internal control flow carrying one typed adapter refusal."""

    def __init__(self, reason: str, field: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason, self.field, self.detail = reason, field, detail

def _refuse(reason: str, field: str, detail: str = "") -> None:
    raise _Refusal(reason, field, detail)

def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))

def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _mask_sha256(mask: np.ndarray) -> str:
    array = np.asarray(mask)
    header = (b"m9c-proposal-mask-1:" + bytes([array.ndim]) + array.dtype.str.encode("ascii")
              + b"".join(int(d).to_bytes(8, "little") for d in array.shape))
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()

@dataclass(frozen=True, slots=True)
class ImageDerivedLatticeRefusal:
    """One typed, atomic adapter refusal; it carries no readings or bridge."""

    reason: str
    field: str
    detail: str = ""

    def __post_init__(self) -> None:
        if (self.reason not in IMAGE_DERIVED_LATTICE_REFUSAL_REASONS
                or type(self.field) is not str or not self.field
                or type(self.detail) is not str):
            raise ValueError("adapter refusal fields are malformed")

@dataclass(frozen=True, slots=True)
class ImageDerivedLatticeResult:
    """One sealed lattice/provenance bundle over an accepted session."""

    session_sha256: str
    frame_ids: tuple[str, ...]
    calibrations_sha256: str
    proposals_sha256: str
    pose_graph_sha256: str
    bridge: GeometryProvenanceBridge
    producer_version: str = IMAGE_DERIVED_LATTICE_PRODUCER_VERSION
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (not _hex64(self.session_sha256) or not _hex64(self.calibrations_sha256)
                or not _hex64(self.proposals_sha256) or not _hex64(self.pose_graph_sha256)
                or not isinstance(self.bridge, GeometryProvenanceBridge)
                or self.producer_version != IMAGE_DERIVED_LATTICE_PRODUCER_VERSION
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError("image-derived lattice result fields are malformed")
        digest = _sha256_json({
            "producer": self.producer_version, "session_sha256": self.session_sha256,
            "frame_ids": list(self.frame_ids),
            "calibrations_sha256": self.calibrations_sha256,
            "proposals_sha256": self.proposals_sha256,
            "pose_graph_sha256": self.pose_graph_sha256,
            "bridge_sha256": self.bridge.bridge_sha256})
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived lattice result digest mismatch")
        object.__setattr__(self, "result_sha256", digest)

def _replay_session(session: object) -> ImageDerivedSession:
    if not isinstance(session, ImageDerivedSession):
        _refuse("geometry_unavailable" if session is None else "geometry_invalid", "session")
    try:
        replayed = replace(session)
        for frame in replayed.frames:
            replace(frame)
    except ImageDerivedSessionError as error:
        _refuse("provenance_mismatch" if error.reason == SESSION_DIGEST_MISMATCH
                else "geometry_invalid", "session", error.reason)
    except _ERRORS as error:
        _refuse("geometry_invalid", "session", type(error).__name__)
    return replayed

def _replay_pose_graph(pose_graph: object, session: ImageDerivedSession) -> ImageDerivedPoseGraph:
    """Replay the pose graph (graph, budget, poses), then bind it to the session."""
    if not isinstance(pose_graph, ImageDerivedPoseGraph):
        _refuse("geometry_unavailable" if pose_graph is None else "geometry_invalid",
                "pose_graph")
    try:
        replayed = replace(pose_graph)
        replace(replayed.budgets)
        for pose in replayed.poses:
            replace(pose)
    except _ERRORS as error:
        _refuse("provenance_mismatch", "pose_graph", type(error).__name__)
    if replayed.status != "ok":
        _refuse("geometry_unavailable", "pose_graph", replayed.reason)
    if replayed.session_sha256 != session.session_sha256:
        _refuse("provenance_mismatch", "pose_graph.session_sha256",
                "pose graph belongs to a different session")
    if replayed.frame_ids != tuple(session.cameras) or replayed.unresolved != session.unresolved:
        _refuse("provenance_mismatch", "pose_graph.frame_ids", "foreign session frames")
    return replayed

def _coverage(frame_ids: set, records: Mapping, name: str) -> None:
    missing = sorted(frame_ids - set(records))
    if missing:
        _refuse("geometry_unavailable", f"{name}[{missing[0]}]", "no record for frame")

def _calibrations(frame_ids: set, shapes: dict, readings: tuple, calibrations: object) -> list:
    if calibrations is None or (isinstance(calibrations, Mapping) and not calibrations):
        _refuse("geometry_unavailable", "calibrations")
    if not isinstance(calibrations, Mapping):
        _refuse("geometry_invalid", "calibrations", "expected a mapping keyed by frame id")
    _coverage(frame_ids, calibrations, "calibrations")
    validated = {}
    payload = []
    for frame in sorted(calibrations):
        field = f"calibrations[{frame}]"
        if frame not in frame_ids:
            _refuse("geometry_invalid", field, "frame is not in the session")
        calibration = calibrations[frame]
        if not isinstance(calibration, CalibrationResult):
            _refuse("geometry_invalid", field, "expected CalibrationResult")
        try:
            numbers = [float(calibration.focal_pixels), float(calibration.yaw_degrees),
                       float(calibration.pitch_degrees), float(calibration.roll_degrees),
                       float(calibration.median_residual)]
            height, width = int(calibration.height), int(calibration.width)
            principal = calibration.principal_point
            if principal is not None and (not isinstance(principal, (tuple, list))
                                          or len(principal) != 2):
                raise ValueError("principal point must be a pair")
            pair = None if principal is None else (float(principal[0]), float(principal[1]))
            notes = list(calibration.notes)
            axes = list(calibration.per_axis_inliers)
        except _ERRORS as error:
            _refuse("geometry_invalid", field, type(error).__name__)
        if (not all(np.isfinite(value) for value in numbers) or numbers[0] <= 0.0
                or width <= 0 or height <= 0
                or not all(type(note) is str for note in notes)
                or not all(type(value) is int for value in axes)
                or (pair is not None and any(not np.isfinite(value) for value in pair))):
            _refuse("geometry_invalid", field, "calibration fields are malformed")
        if (height, width) != tuple(shapes[frame][:2]):
            _refuse("provenance_mismatch", field + ".dimensions", "frame mismatch")
        validated[frame] = calibration
        payload.append([field, *numbers, width, height, int(calibration.inliers),
                        int(calibration.segments), axes,
                        None if pair is None else list(pair), notes])
    _bind_rectifications(readings, validated)
    return payload

def _bind_rectifications(readings: tuple, calibrations: dict) -> None:
    """Bind each reading's rectification to its own frame's calibration optics."""
    for reading in readings:
        calibration = calibrations[reading.frame]
        field = f"calibrations[{reading.frame}].rectification"
        rectification = reading.rectification
        if not isinstance(rectification, lattice_planes.PlaneRectification):
            _refuse("geometry_invalid", field, "expected PlaneRectification")
        try:
            expected_axes = tuple(
                tuple(float(value) for value in row)
                for row in axis_directions(calibration.yaw_degrees,
                                           calibration.pitch_degrees,
                                           calibration.roll_degrees))
            expected_focal = float(calibration.focal_pixels)
            principal = calibration.principal_point or (
                calibration.width / 2.0, calibration.height / 2.0)
            expected_principal = (float(principal[0]), float(principal[1]))
            actual_axes = tuple(tuple(float(value) for value in row)
                                for row in rectification.axis_matrix)
            actual_principal = tuple(float(value)
                                     for value in rectification.principal_point)
            actual_focal = float(rectification.focal)
        except _ERRORS as error:
            _refuse("geometry_invalid", field, type(error).__name__)
        if (actual_focal != expected_focal or actual_principal != expected_principal
                or actual_axes != expected_axes):
            _refuse("provenance_mismatch", field, "foreign calibration for this frame")

def _proposals(frame_ids: set, shapes: dict, readings: tuple, proposals: object) -> list:
    if proposals is None or (isinstance(proposals, Mapping) and not proposals):
        _refuse("geometry_unavailable", "proposals")
    if not isinstance(proposals, Mapping):
        _refuse("geometry_invalid", "proposals", "expected a mapping keyed by frame id")
    _coverage(frame_ids, proposals, "proposals")
    payload = []
    for frame in sorted(proposals):
        field = f"proposals[{frame}]"
        if frame not in frame_ids:
            _refuse("geometry_invalid", field, "frame is not in the session")
        result = proposals[frame]
        if not isinstance(result, boundary_candidates.PlaneProposalResult):
            _refuse("geometry_invalid", field, "expected PlaneProposalResult")
        if result.status not in ("ok", "abstain") or type(result.reason) is not str:
            _refuse("geometry_invalid", field, "proposal status/reason is malformed")
        if not isinstance(result.proposals, tuple):
            _refuse("geometry_invalid", field, "proposals must be a tuple")
        entries = []
        for proposal in result.proposals:
            if not isinstance(proposal, boundary_candidates.PlaneProposal):
                _refuse("geometry_invalid", field, "proposal entry has an invalid type")
            try:
                axis = int(proposal.axis)
                points = tuple((float(point[0]), float(point[1])) for point in proposal.polygon)
                sources = tuple(proposal.sources)
                strength, coverage = float(proposal.strength), float(proposal.coverage)
                mask = np.asarray(proposal.mask)
            except _ERRORS as error:
                _refuse("geometry_invalid", field, type(error).__name__)
            if (axis not in (0, 1, 2) or len(points) < 3
                    or any(not np.isfinite(value) for point in points for value in point)
                    or not all(type(source) is str for source in sources)
                    or not np.isfinite(strength) or not np.isfinite(coverage)):
                _refuse("geometry_invalid", field, "proposal fields are malformed")
            if (mask.ndim != 2 or tuple(mask.shape) != tuple(shapes[frame][:2])
                    or not (np.issubdtype(mask.dtype, np.number)
                            or np.issubdtype(mask.dtype, np.bool_))
                    or (np.issubdtype(mask.dtype, np.number)
                        and not np.isfinite(mask).all())):
                _refuse("geometry_invalid", field + ".mask", "mask does not match the frame")
            entries.append([axis, int(proposal.side), [[x, y] for x, y in points],
                            strength, list(sources), coverage, _mask_sha256(mask)])
        payload.append([frame, result.status, result.reason, entries])
    for reading in readings:
        result = proposals.get(reading.frame)
        if result is None:
            _refuse("geometry_unavailable", f"proposals[{reading.frame}]")
        if result.status != "ok" or not result.proposals:
            _refuse("geometry_unavailable", f"proposals[{reading.frame}]", "no proposals")
        if not any(_proposal_supports(item, reading) for item in result.proposals):
            _refuse("provenance_mismatch", "proposals.binding", reading.frame)
    return payload

def _proposal_supports(proposal: object, reading: lattice_planes.PlaneReading) -> bool:
    try:
        return bool(
            isinstance(proposal, boundary_candidates.PlaneProposal)
            and isinstance(reading.region_evidence, lattice_planes.PlaneRegionEvidence)
            and int(proposal.axis) == int(reading.axis)
            and tuple(sorted(proposal.sources)) == reading.region_evidence.sources
            and lattice_planes._canonical_region_polygon(proposal.polygon)
            == lattice_planes._canonical_region_polygon(reading.polygon))
    except _ERRORS:
        return False

def _bridge_refusal(refusal: object) -> ImageDerivedLatticeRefusal:
    reason = getattr(refusal, "reason", "invalid_input")
    detail = getattr(refusal, "detail", {})
    field = detail.get("field", "provenance") if isinstance(detail, Mapping) else "provenance"
    field = field if type(field) is str and field else "provenance"
    typed = ("geometry_unavailable" if reason == "missing_provenance"
             else "provenance_mismatch" if reason in ("provenance_mismatch", "replay_mismatch")
             else "geometry_invalid")
    return ImageDerivedLatticeRefusal(typed, field, reason)

def _assemble(*, session: object, pose_graph: object, calibrations: object,
              proposals: object, readings: object, structural_weights: object,
              split_sources: object) -> ImageDerivedLatticeResult | ImageDerivedLatticeRefusal:
    replayed = _replay_session(session)
    replayed_pose = _replay_pose_graph(pose_graph, replayed)
    shapes = {frame.frame_id: tuple(frame.image_shape) for frame in replayed.frames}
    if readings is None or (isinstance(readings, tuple) and not readings):
        _refuse("geometry_unavailable", "readings")
    if not isinstance(readings, tuple) or any(
            not isinstance(reading, lattice_planes.PlaneReading) for reading in readings):
        _refuse("geometry_invalid", "readings", "expected PlaneReading values")
    for reading in readings:
        if reading.frame not in shapes:
            _refuse("provenance_mismatch", "readings.frame", f"{reading.frame!r} unknown")
        evidence = reading.region_evidence
        if not isinstance(evidence, lattice_planes.PlaneRegionEvidence):
            _refuse("geometry_unavailable", "readings.region_evidence", reading.frame)
        if tuple(evidence.image_shape) != shapes[reading.frame][:2]:
            _refuse("provenance_mismatch", "readings.region_evidence.image_shape", reading.frame)
    for name, value, kind in (
            ("structural_weights", structural_weights, StructuralWeightsEvidence),
            ("split_sources", split_sources, SplitSourceEvidence)):
        if value is None or (isinstance(value, tuple) and not value):
            _refuse("geometry_unavailable", name)
        if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
            _refuse("geometry_invalid", name, f"expected {kind.__name__} values")
    calibration_payload = _calibrations(set(shapes), shapes, readings, calibrations)
    proposal_payload = _proposals(set(shapes), shapes, readings, proposals)
    bridge = provenance.bind_geometry_provenance(
        readings=readings, structural_weights=structural_weights,
        split_sources=split_sources)
    if not isinstance(bridge, GeometryProvenanceBridge):
        return _bridge_refusal(bridge)
    return ImageDerivedLatticeResult(
        session_sha256=replayed.session_sha256,
        frame_ids=tuple(frame.frame_id for frame in replayed.frames),
        calibrations_sha256=_sha256_json(calibration_payload),
        proposals_sha256=_sha256_json(proposal_payload),
        pose_graph_sha256=replayed_pose.pose_graph_sha256, bridge=bridge)

def assemble_image_derived_lattice(
    *, session: object, pose_graph: object, calibrations: object, proposals: object,
    readings: object, structural_weights: object, split_sources: object,
) -> ImageDerivedLatticeResult | ImageDerivedLatticeRefusal:
    """Replay, cross-bind and seal accepted in-memory records; never measure.

    ``session``/``pose_graph`` are the accepted session and pose result;
    ``calibrations`` and ``proposals`` map every session frame id to its record.
    """
    try:
        return _assemble(
            session=session, pose_graph=pose_graph, calibrations=calibrations,
            proposals=proposals, readings=readings,
            structural_weights=structural_weights, split_sources=split_sources)
    except _Refusal as refusal:
        return ImageDerivedLatticeRefusal(refusal.reason, refusal.field, refusal.detail)
    except _ERRORS as error:
        return ImageDerivedLatticeRefusal("geometry_invalid", "adapter", type(error).__name__)
