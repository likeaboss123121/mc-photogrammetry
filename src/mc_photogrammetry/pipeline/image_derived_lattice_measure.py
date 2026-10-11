"""Bounded M9 image-derived lattice measurement and provenance seam.

This is the DS-M9-C *measurement* seam between the accepted M7 raw-image entry
and the accepted M9-A placement assembler.  It consumes the ordinary,
image-derived in-memory records M7 already built - an
:class:`~mc_photogrammetry.pipeline.image_derived_session.ImageDerivedSession`
whose frames retain their immutable pixels and whose camera map retains the
accepted per-frame :class:`~mc_photogrammetry.pipeline.contracts.CameraEstimate`,
plus the accepted
:class:`~mc_photogrammetry.pipeline.image_derived_pose_graph.ImageDerivedPoseGraph`
for those frames - and walks the *image* chain one honest step further:

    retained pixels + accepted camera records
        -> the accepted image-only proposal source
           (:func:`~mc_photogrammetry.reconstruction.region_growth.propose_image_planes_v2`)
        -> the accepted lattice plane reader
           (:func:`~mc_photogrammetry.reconstruction.lattice_planes.frame_plane_readings_from_proposals`)
        -> the exact rectified structural-weight patch each reading measured and
           the reader discards
        -> the accepted P0 split-phase producer over that exact patch
        -> one sealed
           :class:`~mc_photogrammetry.reconstruction.plane_geometry_provenance.GeometryProvenanceBridge`
        -> a typed :data:`GEOMETRY_UNAVAILABLE` terminal

It is deliberately not a re-binding helper: no caller supplies a reading, a
structural-weight patch or a split source.  Every one of those is *measured* from
the retained pixels inside this call, so a caller cannot present evidence it did
not derive.

What it owns
------------

#. **Camera recovery from the pixels, bound to the accepted camera record.**
   The accepted session keeps only the calibration's focal length, yaw, pitch,
   roll, image size, principal point and median residual; the calibration's
   inlier count that the proposal source gates on was dropped.  This seam
   re-derives that count, the segment count and the per-axis counts from the
   retained pixels and the session's own camera with the accepted residual rule
   (:func:`~mc_photogrammetry.reconstruction.calibration.segment_residuals` over
   :func:`~mc_photogrammetry.reconstruction.calibration.vanishing_points`), and
   refuses a session camera whose retained fields do not describe the retained
   pixels.  No calibration is re-searched and no camera is invented.
#. **The discarded structural-weight patch.**  The reader rectifies a
   per-segment weight image through each reading's own recorded rectification and
   keeps only the derived share and phase evidence.  This seam rebuilds that
   exact rectified patch from the retained pixels with the accepted helpers,
   requires it to reproduce the reading's retained ``structural_share`` and
   ``phase_evidence`` exactly, and retains it as
   :class:`~mc_photogrammetry.reconstruction.plane_geometry_provenance.StructuralWeightsEvidence`.
#. **The P0 split source.**  It runs the accepted frozen producer over that same
   exact patch with an explicit bounded budget derived from the patch geometry,
   and retains the accepted result, budget and held-out patch as
   :class:`~mc_photogrammetry.reconstruction.plane_geometry_provenance.SplitSourceEvidence`.
#. **One all-or-nothing provenance bind.**  Every reading, its structural
   evidence, its split source, the accepted session digest and the accepted pose
   graph digest are sealed through
   :func:`~mc_photogrammetry.reconstruction.plane_geometry_provenance.bind_geometry_provenance`;
   a partial bridge is never returned.

Typed honesty
-------------

The terminal is always :data:`GEOMETRY_UNAVAILABLE`: this seam does **not**
construct a ``PlacementUniverse``, an aligned region support, a resolved face
placement, a plane volume, an M5 observation, a proposal or a promotion, and it
never calls the M6 orchestration facade.  When an honest prerequisite is absent
the call returns an :class:`ImageDerivedLatticeResult` with ``status
"unavailable"`` and one typed :data:`LATTICE_UNAVAILABLE_REASONS` reason
(pixel-free session, refused pose graph, no lattice reading, or incomplete
structural/split provenance).  When a carried record contradicts the accepted
contract - a forged frame, a stale session digest, a foreign pose-graph binding,
or a patch that does not reproduce its reading - the call returns one atomic
:class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementRefusal`.
Neither path fabricates V1/V2/V3 geometry or a metric scale.

Input discipline
----------------

The public call shape accepts exactly two already-accepted in-memory records and
nothing else.  There is no parameter for a manifest, an F3/OCR value, a
coordinate, a direction, a bound, a reference, a scale, a translation, an
annotation, a material palette, a budget, a mask or a rectified side, and no
``**kwargs`` through which one could arrive.  The usable mask is derived per
frame with the accepted
:func:`~mc_photogrammetry.vision.gui_mask.create_gui_mask`; the rectified-patch
ceiling is a module constant.  The module performs no IO, has no
filesystem/path/network/process/environment access, defines no CLI, does not
touch the persistence schema, and does not import the M7 entry module.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline.image_derived_pose_graph import (
    GaugeCameraPose,
    ImageDerivedPoseGraph,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    ImageDerivedFrame,
    ImageDerivedSession,
)
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import region_growth
from mc_photogrammetry.reconstruction.calibration import (
    INLIER_PIXELS,
    CalibrationResult,
    detect_segments,
    enhance,
    segment_residuals,
    structural_weights,
    vanishing_points,
)
from mc_photogrammetry.vision.gui_mask import create_gui_mask

__all__ = [
    "IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA",
    "IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER",
    "GEOMETRY_UNAVAILABLE",
    "LATTICE_UNAVAILABLE_REASONS",
    "LATTICE_MAX_RECTIFIED_SIDE",
    "LATTICE_MAX_PROPOSALS",
    "LatticeAbstention",
    "ImageDerivedLatticeResult",
    "measure_image_derived_lattice",
]

IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA = "mc-photogrammetry-image-derived-lattice-measure"
"""Explicit schema name for the lattice/provenance terminal result."""

IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER = "m9c-image-derived-lattice-measure-3"
"""Producer identity carried by every terminal result."""

GEOMETRY_UNAVAILABLE = "geometry_unavailable"
"""The typed M7 terminal boundary, preserved verbatim.

The accepted M7 entry owns this token; this module defines the same value rather
than importing the entry (the entry is a terminal leaf, and a future M6 wiring
must not create an import cycle).  A focused control asserts the two tokens are
equal, so the boundary cannot silently drift.
"""

LATTICE_UNAVAILABLE_REASONS: tuple[str, ...] = (
    "image_unavailable",
    "pose_unavailable",
    "no_lattice_reading",
    "provenance_incomplete",
)
"""Every typed ``reason`` an honest ``status "unavailable"`` result may carry."""

LATTICE_MAX_RECTIFIED_SIDE = 512
"""Explicit ceiling on every reading's rectified patch side.

The reader's own ceiling is 2048; this seam lowers it so the retained
structural-weight and held-out split patches stay bounded in memory.  The value
is a module constant, never a caller parameter, so the measurement cannot be
widened from the call site.
"""

LATTICE_MAX_PROPOSALS = region_growth.MAX_REGION_GROWTH_PROPOSALS
"""Explicit bound on the accepted image-only proposal source per frame."""

_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
    FloatingPointError,
)
_MEASUREMENT_ERRORS = _ERRORS + (cv2.error, MemoryError, RuntimeError)


class _InputRefusal(Exception):
    """Internal control flow: one typed refusal and its JSON detail."""

    def __init__(self, reason: str, detail: object = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = {} if detail is None else detail


def _refuse(reason: str, field: str) -> None:
    raise _InputRefusal(reason, {"field": field})


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _safe_hex(value: object) -> str:
    """A carried digest, or ``""`` where an accepted record legitimately omits it."""
    return value if _hex64(value) else ""  # type: ignore[return-value]


def _declared_fields(record: object) -> dict:
    return {field.name: getattr(record, field.name)
            for field in dataclasses.fields(record)}


def _finite_token(value: object) -> object:
    """A deterministic JSON token for a float, including non-finite sentinels."""
    number = float(value)  # type: ignore[arg-type]
    if number != number:
        return {"nonfinite": "nan"}
    if number == float("inf"):
        return {"nonfinite": "inf"}
    if number == float("-inf"):
        return {"nonfinite": "-inf"}
    return number


# ---------------------------------------------------------------------------
# Accepted-record replay
# ---------------------------------------------------------------------------

def _replay_frame(frame: object, record_frame: object) -> None:
    """Rebuild one retained frame through the accepted constructor.

    The canonical session record is pixel-free, so it cannot see a retained
    ``image`` replaced by a writable, non-``uint8`` or otherwise
    constructor-bypassed array of identical bytes.  The accepted frame
    constructor re-runs the immutable-``uint8`` checks, the exact-shape check and
    the pixel digest; the rebuilt identity/shape/digest must then equal the frame
    the canonical record rebuilt.

    The retained ingress pin and frame payload must match the canonical record,
    including for pixel-free records. The constructor re-checks a carried pin
    against this frame's id, shape and digest.
    """
    if not isinstance(frame, ImageDerivedFrame) or not isinstance(
            record_frame, ImageDerivedFrame):
        _refuse("invalid_input", "session.frame")
    if session_mod._frame_payload(frame) != session_mod._frame_payload(record_frame):
        field = ("session.frame.pixels" if frame.ingress != record_frame.ingress
                 else "session.frame")
        _refuse("provenance_mismatch", field)
    if frame.image is None:
        if record_frame.image is not None:
            _refuse("provenance_mismatch", "session.frame.pixels")
        return
    try:
        replayed = ImageDerivedFrame(
            frame_id=frame.frame_id, source_label=frame.source_label,
            image=frame.image, image_shape=frame.image_shape,
            image_sha256=frame.image_sha256,
            ingress=frame.ingress)
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "session.frame.pixels",
             "detail": type(error).__name__}) from error
    if (replayed.frame_id != record_frame.frame_id
            or session_mod._frame_payload(replayed)
            != session_mod._frame_payload(record_frame)):
        _refuse("provenance_mismatch", "session.frame")


def _replay_session(session: object) -> ImageDerivedSession:
    """Round-trip the session record and re-digest every retained frame."""
    if not isinstance(session, ImageDerivedSession):
        _refuse("invalid_input", "session")
    try:
        record = session_mod.image_derived_session_to_record(session)
        rebuilt = session_mod.image_derived_session_from_record(record)
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "session", "detail": type(error).__name__}) from error
    if rebuilt.session_sha256 != session.session_sha256:
        _refuse("provenance_mismatch", "session.session_sha256")
    try:
        frames = tuple(session.input_frames)
        active_frames = tuple(session.frames)
        frame_ids = tuple(frame.frame_id for frame in frames)
        active_ids = tuple(frame.frame_id for frame in active_frames)
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "session.frames", "detail": type(error).__name__}) from error
    if tuple(frame.frame_id for frame in rebuilt.input_frames) != frame_ids:
        _refuse("provenance_mismatch", "session.frames")
    for frame, record_frame in zip(frames, rebuilt.input_frames):
        _replay_frame(frame, record_frame)
    if active_ids != tuple(frame.frame_id for frame in rebuilt.frames):
        _refuse("provenance_mismatch", "session.frames")
    inventory = {frame.frame_id: frame for frame in frames}
    for frame, record_frame in zip(active_frames, rebuilt.frames):
        _replay_frame(frame, record_frame)
        if frame is not inventory[frame.frame_id]:
            _refuse("provenance_mismatch", "session.frame")
    return session


def _replay_pose_budget(budget: object) -> None:
    if not isinstance(budget, pose_graph_mod.PoseGraphBudgets):
        _refuse("invalid_input", "pose_graph.budgets")
    try:
        rebuilt = pose_graph_mod.PoseGraphBudgets(**_declared_fields(budget))
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "pose_graph.budgets",
             "detail": type(error).__name__}) from error
    if rebuilt.to_payload() != budget.to_payload():
        _refuse("provenance_mismatch", "pose_graph.budgets")


def _replay_pose_records(poses: object) -> None:
    if not isinstance(poses, tuple):
        _refuse("invalid_input", "pose_graph.poses")
    for pose in poses:
        if not isinstance(pose, GaugeCameraPose):
            _refuse("invalid_input", "pose_graph.poses")
        try:
            rebuilt = GaugeCameraPose(**_declared_fields(pose))
        except _ERRORS as error:
            raise _InputRefusal(
                "provenance_mismatch",
                {"field": "pose_graph.poses",
                 "detail": type(error).__name__}) from error
        if rebuilt.to_payload() != pose.to_payload():
            _refuse("provenance_mismatch", "pose_graph.poses")


def _replay_pose_graph(
    pose_graph: object, session: ImageDerivedSession,
) -> ImageDerivedPoseGraph:
    """Recompute the pose-graph digest, rebuild it and bind it to the session."""
    if not isinstance(pose_graph, ImageDerivedPoseGraph):
        _refuse("invalid_input", "pose_graph")
    try:
        payload = pose_graph_mod._pose_graph_payload(pose_graph)
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "pose_graph", "detail": type(error).__name__}) from error
    if pose_graph.pose_graph_sha256 != pose_graph_mod._digest(payload):
        _refuse("provenance_mismatch", "pose_graph.pose_graph_sha256")
    try:
        rebuilt = ImageDerivedPoseGraph(**_declared_fields(pose_graph))
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "pose_graph", "detail": type(error).__name__}) from error
    if rebuilt.pose_graph_sha256 != pose_graph.pose_graph_sha256:
        _refuse("provenance_mismatch", "pose_graph.pose_graph_sha256")
    if pose_graph.status != "ok":
        return pose_graph
    _replay_pose_budget(pose_graph.budgets)
    _replay_pose_records(pose_graph.poses)
    if pose_graph.session_sha256 != session.session_sha256:
        _refuse("provenance_mismatch", "pose_graph.session_sha256")
    if tuple(pose_graph.frame_ids) != tuple(
            frame.frame_id for frame in session.frames):
        _refuse("provenance_mismatch", "pose_graph.frame_ids")
    return pose_graph


# ---------------------------------------------------------------------------
# Camera reconstruction from the retained pixels
# ---------------------------------------------------------------------------

def _calibration_grey(image: np.ndarray) -> np.ndarray:
    """The accepted calibrator's single-channel input for one retained frame."""
    frame = np.asarray(image)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _refuse("invalid_input", "session.frame.image")


def _reader_rgb(image: np.ndarray) -> np.ndarray:
    """The accepted reader's three-channel input; a grey frame replicates."""
    frame = np.asarray(image)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    _refuse("invalid_input", "session.frame.image")


def _reconstruct_calibration(
    frame: ImageDerivedFrame, estimate: object,
) -> CalibrationResult:
    """Rebuild the accepted camera record from the retained pixels.

    The session keeps every measured calibration field except the inlier,
    segment and per-axis counts.  This re-derives those three from the frame's
    own detected segments and the session camera's vanishing points, using the
    accepted residual rule, and refuses a camera record whose retained fields do
    not describe the retained pixels.  No search is re-run and no value is
    invented: the inlier count is measured, not assumed.
    """
    if frame.image is None:
        _refuse("invalid_input", "session.frame.image")
    intrinsics = getattr(estimate, "intrinsics", None)
    pose = getattr(estimate, "pose", None)
    if intrinsics is None or pose is None:
        _refuse("provenance_mismatch", "camera.record")
    try:
        focal_x = float(intrinsics.focal_length_pixels[0])
        focal_y = float(intrinsics.focal_length_pixels[1])
        yaw = float(pose.yaw_degrees)
        pitch = float(pose.pitch_degrees)
        roll = float(pose.roll_degrees)
        width = int(intrinsics.image_size[0])
        height = int(intrinsics.image_size[1])
        centre = (float(intrinsics.principal_point_pixels[0]),
                  float(intrinsics.principal_point_pixels[1]))
        raw_residual = getattr(estimate, "residual_pixels", None)
        residual = float("nan") if raw_residual is None else float(raw_residual)
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "camera.record", "detail": type(error).__name__}) from error
    image_height, image_width = int(frame.image.shape[0]), int(frame.image.shape[1])
    if (width, height) != (image_width, image_height) or focal_x != focal_y \
            or focal_x <= 0.0 or width <= 0 or height <= 0 \
            or not all(np.isfinite(value) for value in (focal_x, yaw, pitch, roll)) \
            or not all(np.isfinite(value) for value in centre):
        _refuse("provenance_mismatch", "camera.record")

    grey = _calibration_grey(frame.image)
    segments = detect_segments(enhance(grey))
    principal = None if centre == (width / 2.0, height / 2.0) else centre
    try:
        residuals = segment_residuals(segments, vanishing_points(
            focal_x, yaw, pitch, width, height, roll, principal))
        owner = residuals.argmin(axis=1)
        closest = residuals.min(axis=1)
        inside = closest <= INLIER_PIXELS
        inliers = int(inside.sum())
        counts = tuple(int(((owner == axis) & inside).sum()) for axis in range(3))
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "camera.residuals", "detail": type(error).__name__}) from error
    return CalibrationResult(
        focal_pixels=focal_x, yaw_degrees=yaw, pitch_degrees=pitch,
        roll_degrees=roll, width=width, height=height, inliers=inliers,
        segments=int(len(segments)), median_residual=residual,
        per_axis_inliers=counts, principal_point=principal, notes=())


# ---------------------------------------------------------------------------
# Reader measurement and the discarded patch
# ---------------------------------------------------------------------------

def _measure_frame(
    frame: ImageDerivedFrame, calibration: CalibrationResult,
    *, _atom_snapshot_observer=None, _session_sha256=None,
) -> tuple[tuple, tuple["LatticeAbstention", ...]]:
    """Run the accepted proposal source and lattice reader for one frame."""
    frame_id = frame.frame_id
    image = _reader_rgb(frame.image)
    height, width = int(image.shape[0]), int(image.shape[1])
    usable = ~create_gui_mask(height, width)
    observer = region_growth._attrition_state.get()
    with region_growth._scalar_attrition_context(frame=frame_id):
        try:
            diagnostic_kwargs = ({} if _atom_snapshot_observer is None else {
                "_atom_snapshot_observer": _atom_snapshot_observer,
                "_diagnostic_frame_binding": (frame_id, _session_sha256, frame.image_sha256),
                "_diagnostic_source_image": frame.image})
            proposals = region_growth.propose_image_planes_v2(
                image, usable, calibration, max_candidates=LATTICE_MAX_PROPOSALS,
                **diagnostic_kwargs)
        except _MEASUREMENT_ERRORS:
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    observer.expect("frame", (frame_id, "proposal"))
                    observer.finish("frame", (frame_id, "proposal"), "proposal_stage_failed")
            return (), (LatticeAbstention(frame_id, "proposal_stage_failed"),)
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            observer.expect("frame", (frame_id, "proposal"))
            observer.finish("frame", (frame_id, "proposal"), str(proposals.reason or proposals.status),
                            returned_count=len(proposals.proposals))
            for axis in range(3):
                for side in (-1, 1):
                    identity = (frame_id, axis, side)
                    if observer._encode(identity).decode() not in observer._stage("seed")["expected"]:
                        observer.expect("seed", identity)
                        observer.finish("seed", identity, "not_reached", axis=axis, side=side,
                                        eligible_seed_count=None, reason=str(proposals.reason))
    if proposals.status != "ok" or not proposals.proposals:
        return (), (LatticeAbstention(
            frame_id, str(proposals.reason or "proposal_abstained"),
            str(proposals.status)),)
    diagnostics: dict = {"abstentions": [], "segments": 0, "low_structural_share": 0}
    try:
        readings = lattice_planes.frame_plane_readings_from_proposals(
            image, usable, calibration, proposals, frame=frame_id,
            max_readings=lattice_planes.MAX_READINGS,
            max_rectified_side=LATTICE_MAX_RECTIFIED_SIDE, diagnostics=diagnostics)
    except _MEASUREMENT_ERRORS:
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                for identity in observer.pending("reading", frame_id):
                    observer.finish("reading", identity, "reading_stage_failed")
        return (), (LatticeAbstention(frame_id, "reading_stage_failed"),)
    abstentions = tuple(
        LatticeAbstention(frame_id, str(item.reason), str(item.detail))
        for item in diagnostics.get("abstentions", ()))
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            observer.check("proposal_to_reading:" + frame_id,
                           {tuple(value) for value in observer._stage("reading")["expected"].values()
                            if value[0] == frame_id}
                           == {tuple(value) for key, value in observer.bindings.items()
                               if key[0] == "proposal_return" and key[1] == frame_id}
                           == {(frame_id, ordinal) for ordinal in range(len(proposals.proposals))})
            for ordinal, item in enumerate(abstentions):
                observer.expect("frame", (frame_id, "reading_abstention", ordinal))
                observer.finish("frame", (frame_id, "reading_abstention", ordinal), item.reason)
    return tuple(readings), abstentions


def _frame_weight_image(image: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """The exact rasterised per-segment weight image the reader rectifies from."""
    grey = lattice_planes._masked_grey(image, usable)
    segments = detect_segments(enhance(grey))
    weights = structural_weights(grey, segments)
    height, width = int(usable.shape[0]), int(usable.shape[1])
    return lattice_planes._weight_image(segments, weights, height, width)


def _structural_evidence(
    reading: lattice_planes.PlaneReading, weight_image: np.ndarray,
    canonical_index: int,
) -> tuple[provenance.StructuralWeightsEvidence, np.ndarray]:
    """Rebuild and verify the exact rectified structural-weight patch.

    The reader computes this patch from the frame's weight image through the
    reading's own recorded rectification, consumes its share and phase, and
    discards it.  Reproducing it here is a *measurement*, not a copy: the rebuilt
    patch must reproduce the reading's retained ``structural_share`` and
    ``phase_evidence`` exactly, or the call refuses rather than retain a patch the
    reading was not measured from.
    """
    region = reading.region_evidence
    if region is None:
        _refuse("missing_provenance", "readings.region_evidence")
    try:
        patch = lattice.rectify_plane(
            weight_image, int(reading.axis),
            np.asarray(reading.rectification.axis_matrix, dtype=float),
            float(reading.rectification.focal), extent=float(reading.extent),
            samples=int(reading.samples),
            centre=tuple(float(value) for value in reading.rectification.centre),
            principal_point=tuple(
                float(value) for value in reading.rectification.principal_point),
            side=float(reading.side))
    except _ERRORS as error:
        raise _InputRefusal(
            "provenance_mismatch",
            {"field": "structural_weights.patch",
             "detail": type(error).__name__}) from error
    if patch is None:
        _refuse("provenance_mismatch", "structural_weights.patch")
    rows, columns = int(patch.shape[0]), int(patch.shape[1])
    if (rows != columns or rows != int(reading.samples)
            or not np.all(np.isfinite(patch)) or np.any(patch < 0.0)):
        _refuse("provenance_mismatch", "structural_weights.patch")
    share = float(lattice_planes._structural_share(patch, reading.period_pixels))
    phase = lattice_planes._periodic_energy_phase_evidence(
        patch, reading.period_pixels,
        centre=tuple(float(value) for value in reading.rectification.centre),
        extent=float(reading.extent), samples=int(reading.samples),
        source_region_id=region.source_region_id)
    if share != float(reading.structural_share) or phase != reading.phase_evidence:
        _refuse("provenance_mismatch", "structural_weights.reading_binding")
    stored = np.ascontiguousarray(patch, dtype="<f4")
    evidence = provenance.StructuralWeightsEvidence(
        frame=reading.frame, canonical_index=int(canonical_index),
        reading_axis=int(reading.axis), source_region_id=region.source_region_id,
        patch_rows=rows, patch_columns=columns,
        patch_float32_le=stored.tobytes(order="C"), structural_share=share,
        period_pixels=float(reading.period_pixels), extent=float(reading.extent),
        samples=int(reading.samples),
        centre_uv=tuple(float(value) for value in reading.rectification.centre),
        phase_evidence=phase)
    return evidence, patch


def _split_budget(rows: int, columns: int) -> lattice_planes.SplitPhaseProducerBudget:
    """A bounded P0 budget derived from the exact patch geometry.

    Every cap is a deterministic function of the patch shape, so the producer's
    planned-work gate is sized to this patch and no caller can widen it.  The
    caps follow the accepted reconstruction
    :func:`~mc_photogrammetry.reconstruction.plane_phase_anchor._p0_budget_for`
    uses over an already-produced result.
    """
    samples = int(rows) * int(columns)
    span = int(rows) + int(columns)
    return lattice_planes._make_budget({
        "max_lanes_per_axis": max(1, int(rows), int(columns)),
        "max_patch_samples": max(1, samples),
        "max_inline_patch_samples": max(1, samples),
        "max_g_samples_per_axis": max(1, samples),
        "max_d_samples_per_axis": max(1, samples),
        "max_guard_samples_per_axis": max(1, samples),
        "max_profile_terms": max(1, span),
        "max_d_profile_terms": max(1, 3 * span),
        "max_payload_bytes": (2 * len(
            lattice_planes.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION) + 48 + 24 * span),
        "max_source_records": 9,
        "max_replay_checks": 10,
    })


def _split_source(
    reading: lattice_planes.PlaneReading, patch: np.ndarray, canonical_index: int,
) -> provenance.SplitSourceEvidence | None:
    """Run the accepted P0 producer over the exact retained patch."""
    region = reading.region_evidence
    if region is None:
        return None
    rows, columns = int(patch.shape[0]), int(patch.shape[1])
    budget = _split_budget(rows, columns)
    try:
        geometry = lattice_planes.SplitPhaseGeometry(
            centre=tuple(float(value) for value in reading.rectification.centre),
            extent=float(reading.extent), samples=int(reading.samples),
            rectified_pixel_step=lattice.rectified_pixel_step(
                float(reading.extent), int(reading.samples)),
            period_pixels=float(reading.period_pixels))
        result = lattice_planes.produce_split_phase_evidence(
            np.asarray(patch), geometry=geometry, frame=reading.frame,
            plane_index=int(canonical_index), reading_axis=int(reading.axis),
            source_region_id=region.source_region_id,
            partition_config=lattice_planes.SPLIT_PHASE_PARTITION_CONFIG,
            budget=budget)
    except _ERRORS:
        return None
    if result.status != "complete":
        return None
    held_out = np.ascontiguousarray(np.asarray(patch).astype("<f8", copy=False))
    try:
        return provenance.SplitSourceEvidence(
            frame=reading.frame, canonical_index=int(canonical_index),
            reading_axis=int(reading.axis), result=result, budget=budget,
            patch_float64_le=held_out.tobytes(order="C"), patch_rows=rows,
            patch_columns=columns)
    except _ERRORS:
        return None


# ---------------------------------------------------------------------------
# Deterministic reading order and result records
# ---------------------------------------------------------------------------

def _ordered_readings(readings: tuple, frame_order: dict) -> tuple:
    """Order readings by frame order then the accepted per-frame content key."""
    try:
        return tuple(sorted(
            readings,
            key=lambda reading: (frame_order[reading.frame],
                                 lattice_planes.reading_order_key(reading))))
    except (AttributeError, KeyError, TypeError) as error:
        raise _InputRefusal(
            "invalid_input", {"field": "readings",
                              "detail": type(error).__name__}) from error


def _canonical_indices(readings: tuple) -> dict:
    """Map each reading object to its accepted canonical per-frame index."""
    try:
        index = placement._readings_index(readings)
    except placement._InputRefusal as refusal:
        raise _InputRefusal(refusal.reason, refusal.detail) from refusal
    except _ERRORS as error:
        raise _InputRefusal(
            "invalid_input", {"field": "readings",
                              "detail": type(error).__name__}) from error
    canonical = {id(reading): position
                 for (_frame, position, _axis), reading in index.items()}
    if len(canonical) != len(readings):
        _refuse("invalid_input", "readings.duplicate_identity")
    return canonical


@dataclasses.dataclass(frozen=True, slots=True)
class LatticeAbstention:
    """One frame or reading that did not contribute, and the typed reason."""

    frame: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("lattice abstention needs a non-empty frame id")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("lattice abstention needs a typed reason")
        if not isinstance(self.detail, str):
            raise ValueError("lattice abstention detail must be a string")


def _calibration_payload(calibration: CalibrationResult) -> list:
    principal = calibration.principal_point
    return [
        _finite_token(calibration.focal_pixels),
        _finite_token(calibration.yaw_degrees),
        _finite_token(calibration.pitch_degrees),
        _finite_token(calibration.roll_degrees),
        int(calibration.width), int(calibration.height),
        int(calibration.inliers), int(calibration.segments),
        _finite_token(calibration.median_residual),
        [int(value) for value in calibration.per_axis_inliers],
        None if principal is None else [_finite_token(principal[0]),
                                        _finite_token(principal[1])],
        [str(note) for note in calibration.notes],
    ]


def _result_payload(result: "ImageDerivedLatticeResult") -> dict:
    bridge = result.bridge
    return {
        "schema": result.schema,
        "producer": result.producer,
        "terminal": result.terminal,
        "status": result.status,
        "reason": result.reason,
        "detail": result.detail,
        "session_sha256": result.session_sha256,
        "pose_graph_sha256": result.pose_graph_sha256,
        "graph_sha256": result.graph_sha256,
        "frame_ids": list(result.frame_ids),
        "camera_calibrations": [
            _calibration_payload(item) for item in result.camera_calibrations],
        "reading_universe_sha256": "" if bridge is None
        else bridge.reading_universe_sha256,
        "bridge_sha256": "" if bridge is None else bridge.bridge_sha256,
        "abstentions": [[item.frame, item.reason, item.detail]
                        for item in result.abstentions],
    }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageDerivedLatticeResult:
    """The typed terminal of one image-derived lattice/provenance measurement.

    ``terminal`` is always :data:`GEOMETRY_UNAVAILABLE`.  ``status`` is
    ``"assembled"`` exactly when a complete provenance bridge was sealed over an
    image-derived reading universe; it is ``"unavailable"`` with a typed
    :data:`LATTICE_UNAVAILABLE_REASONS` ``reason`` when an honest prerequisite was
    absent, in which case ``readings == ()`` and ``bridge is None``.  No
    ``PlacementUniverse``, volume, observation, proposal or promotion is present,
    by construction.

    Construction replays the whole carried chain rather than trusting it: the
    frame ids, every reconstructed calibration record and the sealed provenance
    bridge (through the accepted all-or-nothing binder) are rebuilt, and the
    canonical reading-universe digest is recomputed from the carried readings and
    required to equal the bridge's.  ``session_sha256`` binds the accepted camera
    records and retained pixels, ``pose_graph_sha256``/``graph_sha256`` bind the
    accepted pose records, and ``result_sha256`` replays from the canonical
    payload.
    """

    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    pose_graph_sha256: str
    graph_sha256: str
    frame_ids: tuple[str, ...]
    camera_calibrations: tuple[CalibrationResult, ...]
    readings: tuple
    bridge: provenance.GeometryProvenanceBridge | None
    abstentions: tuple[LatticeAbstention, ...]
    result_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA:
            raise ValueError("unsupported image-derived lattice schema")
        if self.producer != IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER:
            raise ValueError("unsupported image-derived lattice producer")
        if self.terminal != GEOMETRY_UNAVAILABLE:
            raise ValueError("an image-derived lattice cannot leave geometry_unavailable")
        if self.status not in ("assembled", "unavailable"):
            raise ValueError("image-derived lattice status must be assembled or unavailable")
        if not isinstance(self.reason, str) or not isinstance(self.detail, str):
            raise ValueError("image-derived lattice reason/detail must be strings")
        if not _hex64(self.session_sha256) or not _hex64(self.pose_graph_sha256):
            raise ValueError("image-derived lattice digests must be lowercase SHA-256")
        if self.graph_sha256 != "" and not _hex64(self.graph_sha256):
            raise ValueError("image-derived lattice digests must be lowercase SHA-256")
        if self.status == "assembled" and not _hex64(self.graph_sha256):
            raise ValueError("an assembled lattice needs the source graph digest")
        if (not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or any(not isinstance(name, str) or not name for name in self.frame_ids)
                or len(set(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError("image-derived lattice frame ids are malformed")
        if not isinstance(self.abstentions, tuple) or any(
                not isinstance(item, LatticeAbstention) for item in self.abstentions):
            raise ValueError("image-derived lattice abstentions are malformed")
        self._replay_calibrations()
        if self.status == "assembled":
            self._replay_assembled()
        else:
            if self.reason not in LATTICE_UNAVAILABLE_REASONS:
                raise ValueError("image-derived lattice unavailable needs a typed reason")
            if self.readings != () or self.bridge is not None:
                raise ValueError("an unavailable lattice cannot carry readings or a bridge")
        if self.result_sha256 is None:
            object.__setattr__(self, "result_sha256", placement._sha256(
                _result_payload(self)))
        elif (not _hex64(self.result_sha256)
              or self.result_sha256 != placement._sha256(_result_payload(self))):
            raise ValueError("image-derived lattice result digest mismatch")

    def _replay_calibrations(self) -> None:
        if not self.camera_calibrations:
            return
        if (not isinstance(self.camera_calibrations, tuple)
                or len(self.camera_calibrations) != len(self.frame_ids)):
            raise ValueError("image-derived lattice calibrations do not cover the frames")
        for calibration in self.camera_calibrations:
            if not isinstance(calibration, CalibrationResult):
                raise ValueError("image-derived lattice needs accepted calibrations")
            try:
                rebuilt = CalibrationResult(**_declared_fields(calibration))
            except _ERRORS as error:
                raise ValueError(
                    "image-derived lattice calibration does not replay") from error
            if _calibration_payload(rebuilt) != _calibration_payload(calibration):
                raise ValueError("image-derived lattice calibration does not replay")

    def _replay_assembled(self) -> None:
        if self.reason != "" or self.abstentions != ():
            raise ValueError("an assembled lattice cannot carry a refusal reason")
        if (not isinstance(self.readings, tuple) or not self.readings
                or not isinstance(self.bridge, provenance.GeometryProvenanceBridge)):
            raise ValueError("an assembled lattice needs readings and a provenance bridge")
        try:
            rebuilt = provenance.bind_geometry_provenance(
                readings=self.readings,
                structural_weights=self.bridge.structural_weights,
                split_sources=self.bridge.split_sources)
        except _ERRORS as error:
            raise ValueError(
                "image-derived lattice provenance bridge does not replay") from error
        if not isinstance(rebuilt, provenance.GeometryProvenanceBridge):
            raise ValueError("image-derived lattice provenance bridge does not replay")
        if (rebuilt.bridge_sha256 != self.bridge.bridge_sha256
                or rebuilt.reading_universe_sha256 != self.bridge.reading_universe_sha256
                or len(rebuilt.readings) != len(self.readings)
                or any(first is not second for first, second in zip(
                    self.readings, rebuilt.readings))):
            raise ValueError("image-derived lattice provenance bridge does not replay")

    def to_record(self) -> dict:
        """Canonical terminal record; carries accepted records, never geometry."""
        record = _result_payload(self)
        record["result_sha256"] = self.result_sha256
        return record


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def measure_image_derived_lattice(
    *,
    session: ImageDerivedSession,
    pose_graph: ImageDerivedPoseGraph,
    _atom_snapshot_observer=None,
) -> ImageDerivedLatticeResult | placement.PlacementRefusal:
    """Measure the image-derived lattice and seal its retained provenance.

    ``session`` is the accepted image-derived session - the only source of frames
    and camera records - and ``pose_graph`` is the accepted image-derived pose
    graph for that exact session.  The call re-runs the accepted image-only
    proposal source and lattice reader over the retained pixels, reconstructs and
    verifies each reading's discarded structural-weight patch, runs the accepted
    P0 split producer over that exact patch, and returns one sealed
    :class:`ImageDerivedLatticeResult` whose terminal is
    :data:`GEOMETRY_UNAVAILABLE`.

    An honest missing prerequisite - a pixel-free session, a refused pose graph,
    no lattice reading, or an incomplete structural/split provenance set - is a
    typed ``status "unavailable"`` result, never a fabricated reading.  A carried
    record that contradicts the accepted contract is one atomic
    :class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementRefusal`.
    """
    try:
        diagnostic_kwargs = ({} if _atom_snapshot_observer is None else
                             {"_atom_snapshot_observer": _atom_snapshot_observer})
        return _assemble(session=session, pose_graph=pose_graph, **diagnostic_kwargs)
    except _InputRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except placement._InputRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except _ERRORS as error:
        return placement._make_refusal(
            "invalid_input", {"detail": type(error).__name__})


def _assemble(
    *, session: object, pose_graph: object,
    _atom_snapshot_observer=None,
) -> ImageDerivedLatticeResult | placement.PlacementRefusal:
    accepted_session = _replay_session(session)
    accepted_pose_graph = _replay_pose_graph(pose_graph, accepted_session)
    frame_ids = tuple(frame.frame_id for frame in accepted_session.frames)
    base = dict(
        session_sha256=accepted_session.session_sha256,
        pose_graph_sha256=accepted_pose_graph.pose_graph_sha256,
        graph_sha256=_safe_hex(accepted_pose_graph.graph_sha256),
        frame_ids=frame_ids)

    if accepted_pose_graph.status != "ok":
        return _unavailable(
            **base, reason="pose_unavailable",
            detail=(accepted_pose_graph.reason or "pose graph refused"))
    if not accepted_session.has_pixels:
        return _unavailable(
            **base, reason="image_unavailable",
            detail="the accepted session retains no frame pixels")

    frames_by_id = {frame.frame_id: frame for frame in accepted_session.frames}
    calibrations: list[CalibrationResult] = []
    found: dict[str, tuple] = {}
    abstentions: list[LatticeAbstention] = []
    for frame in accepted_session.frames:
        calibration = _reconstruct_calibration(
            frame, accepted_session.cameras[frame.frame_id])
        calibrations.append(calibration)
        diagnostic_kwargs = ({} if _atom_snapshot_observer is None else {
            "_atom_snapshot_observer": _atom_snapshot_observer,
            "_session_sha256": accepted_session.session_sha256})
        readings, frame_abstentions = _measure_frame(frame, calibration, **diagnostic_kwargs)
        found[frame.frame_id] = readings
        abstentions.extend(frame_abstentions)

    collected = tuple(
        reading for frame in accepted_session.frames
        for reading in found[frame.frame_id])
    if not collected:
        return _unavailable(
            **base, reason="no_lattice_reading",
            detail="no frame produced an accepted lattice plane reading",
            abstentions=tuple(abstentions),
            calibrations=tuple(calibrations))

    frame_order = {name: position for position, name in enumerate(frame_ids)}
    ordered = _ordered_readings(collected, frame_order)
    canonical = _canonical_indices(ordered)

    weight_images: dict[str, np.ndarray] = {}
    structural: list[provenance.StructuralWeightsEvidence] = []
    split_sources: list[provenance.SplitSourceEvidence] = []
    incomplete = False
    for reading in ordered:
        if reading.frame not in weight_images:
            image = _reader_rgb(frames_by_id[reading.frame].image)
            usable = ~create_gui_mask(int(image.shape[0]), int(image.shape[1]))
            weight_images[reading.frame] = _frame_weight_image(image, usable)
        evidence, patch = _structural_evidence(
            reading, weight_images[reading.frame], canonical[id(reading)])
        source = _split_source(reading, patch, canonical[id(reading)])
        if source is None:
            abstentions.append(LatticeAbstention(
                reading.frame, "split_source_unavailable",
                f"axis {int(reading.axis)}"))
            incomplete = True
            continue
        structural.append(evidence)
        split_sources.append(source)

    if incomplete:
        return _unavailable(
            **base, reason="provenance_incomplete",
            detail="not every reading retained a complete split-source provenance",
            abstentions=tuple(abstentions),
            calibrations=tuple(calibrations))

    bridge = provenance.bind_geometry_provenance(
        readings=ordered, structural_weights=tuple(structural),
        split_sources=tuple(split_sources))
    if isinstance(bridge, placement.PlacementRefusal):
        return bridge
    return ImageDerivedLatticeResult(
        **base,
        schema=IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA,
        producer=IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER,
        terminal=GEOMETRY_UNAVAILABLE, status="assembled", reason="",
        detail=(
            "image-derived lattice readings and retained structural/split "
            "provenance are sealed; V1/V2/V3/M5 geometry remains unavailable"),
        camera_calibrations=tuple(calibrations), readings=ordered,
        bridge=bridge, abstentions=())


def _unavailable(
    *, session_sha256: str, pose_graph_sha256: str, graph_sha256: str,
    frame_ids: tuple[str, ...], reason: str, detail: str,
    abstentions: tuple[LatticeAbstention, ...] = (),
    calibrations: tuple[CalibrationResult, ...] = (),
) -> ImageDerivedLatticeResult:
    return ImageDerivedLatticeResult(
        schema=IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA,
        producer=IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER,
        terminal=GEOMETRY_UNAVAILABLE, status="unavailable", reason=reason,
        detail=detail, session_sha256=session_sha256,
        pose_graph_sha256=pose_graph_sha256, graph_sha256=graph_sha256,
        frame_ids=frame_ids, camera_calibrations=calibrations, readings=(),
        bridge=None, abstentions=abstentions)
