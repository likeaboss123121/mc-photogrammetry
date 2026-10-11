"""Typed image-derived session producer and deserializer.

This is the bounded M4 production seam that turns an ordered set of in-memory
frames into a reviewable, digest-bound record of what image-only recovery
actually established - and nothing more. It owns exactly four things:

* an immutable capture of each frame (a private ``bytes``-backed ``uint8`` array
  the caller cannot mutate or reopen, plus its shape and SHA-256 digest);
* the delegation of camera recovery to :func:`image_only.solve_image_cameras`,
  which is the existing M3 calibrator path and is not reimplemented here;
* the adaptation of each solved estimate to a local-frame
  :class:`CameraModel` via :func:`image_only.camera_model_from_estimate` with
  ``position=LOCAL_FRAME_ORIGIN``;
* a canonical, deterministic serialization of that session and a deserializer
  that rebuilds it while refusing anything the record cannot honestly support.

The input surface is deliberately two arguments: ``frames`` (an ordered mapping
of frame id to a ``uint8`` grayscale or RGB :class:`numpy.ndarray`) and an
optional ``source_paths`` labeling map. There is no parameter for a camera
manifest, an F3/manual coordinate, a direction, a bound, a scene transform, an
annotation, or any other metadata, and there is no ``**kwargs`` through which one
could arrive.

What one view cannot establish stays explicitly unresolved and is never
invented. The producer preserves the :class:`CameraSolution` facts - a
``None`` pose position, ``scale_known is False``, and the per-frame position /
scale / yaw-quadrant notes - and the session constructor refuses a record that
claims metric scale, claims a camera translation, puts a local camera anywhere
but :data:`image_only.LOCAL_FRAME_ORIGIN`, or drops one of those facts.

Serialization carries only frame ids, source labels, image shapes and digests,
camera solution fields, unresolved flags, and local-frame camera data.
Deserialization cannot recompute an image digest (the pixels are intentionally
not serialized) and therefore never fabricates a pixel array; it verifies the
record's structure, its forbidden-claim invariants, and its deterministic
session digest, and it leaves each frame's ``image`` as ``None``. Because the
session digest is self-describing, a fully resealed record that recomputes a
matching digest is indistinguishable from an original without an externally
retained digest; the constructor's semantic invariants (no scale, no
translation, local-frame adaptation coherence, retained unresolved facts) hold
independently of the digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import hashlib
import json
import math

import numpy as np

from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.pipeline.frame_ingress import (
    FrameIngressProvenanceV1,
    IngressProvenanceError,
)
from mc_photogrammetry.reconstruction.camera import CameraModel


IMAGE_DERIVED_SESSION_SCHEMA = "mc-photogrammetry-image-derived-session"
"""Explicit serialization schema name."""

IMAGE_DERIVED_SESSION_VERSION = 1
"""Explicit serialization version; an unknown value is refused, never guessed."""
IMAGE_DERIVED_SESSION_PARTIAL_VERSION = 2
CAMERA_ESTIMATE_UNAVAILABLE = "camera_estimate_unavailable"
CAMERA_POOLING_CAVEAT = "Excluded inputs may have influenced pooled focal estimation; no reduced-subset rerun or independence claim."
CAMERA_POOLING_CAVEAT_LEGACY = CAMERA_POOLING_CAVEAT
CAMERA_INDEPENDENT_CAVEAT = "Camera estimates were fitted independently per input frame; no focal pooling or reduced-subset rerun was used."

SESSION_INPUT_INVALID = "session_input_invalid"
SESSION_FRAME_INVALID = "session_frame_invalid"
SESSION_CAMERA_UNRESOLVED = "session_camera_unresolved"
SESSION_RECORD_INVALID = "session_record_invalid"
SESSION_SCHEMA_UNSUPPORTED = "session_schema_unsupported"
SESSION_DIGEST_MISMATCH = "session_digest_mismatch"
SESSION_UNRESOLVED_FACT_MISSING = "session_unresolved_fact_missing"
SESSION_FORBIDDEN_CLAIM = "session_forbidden_claim"
"""Machine-readable refusal reasons carried on :class:`ImageDerivedSessionError`."""

_IMAGE_DIGEST_DOMAIN = b"mc-photogrammetry-image-derived-session-frame-v1"

_SESSION_KEYS = frozenset({
    "schema", "version", "frames", "cameras", "components", "unresolved",
    "scale_known", "local_cameras", "session_sha256",
})
_FRAME_KEYS = frozenset({"frame_id", "source_label", "image_shape", "image_sha256"})
_SESSION_V2_KEYS = _SESSION_KEYS | {"active_frame_ids", "admissions", "camera_pooling_caveat"}
_FRAME_KEYS_INGRESS = _FRAME_KEYS | {"ingress"}
_CAMERA_KEYS = frozenset({
    "frame_id", "intrinsics", "pose", "residual_pixels", "confidence"})
_INTRINSICS_KEYS = frozenset({
    "image_size", "focal_length_pixels", "principal_point_pixels"})
_POSE_KEYS = frozenset({
    "position", "yaw_degrees", "pitch_degrees", "roll_degrees"})
_LOCAL_CAMERA_KEYS = frozenset({
    "frame_id", "position", "yaw_degrees", "pitch_degrees", "fov_degrees",
    "eye_height", "roll_degrees", "intrinsics"})

# Each fact is named verbatim on its own per-frame line by
# `calibration.to_camera_solution` (reconstruction/calibration.py lines 312-314).
# Exact-string matching is required: a resealed note such as "frame.png: position"
# preserves the frame-id prefix and fact noun but drops the unresolved wording, so
# substring matching would let it masquerade as the recorded fact.
_UNRESOLVED_FACT_NOTES = (
    ("position", "{frame_id}: position (vanishing points fix no translation)"),
    ("scale", "{frame_id}: scale (needs the lattice period)"),
    ("yaw quadrant", "{frame_id}: yaw quadrant (recovered modulo 90 degrees)"),
)


class ImageDerivedSessionError(ValueError):
    """Invalid session input, an unverifiable record, or a forbidden claim.

    ``reason`` is one of the ``SESSION_*`` constants; ``detail`` is a human
    diagnostic that never replaces the typed reason.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        self.review_evidence = None
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class ImageDerivedFrame:
    """One captured frame: identity, optional label, immutable pixels, digest.

    A produced frame carries a private read-only ``uint8`` array in ``image``
    whose buffer chain terminates at an immutable ``bytes`` object, so it cannot
    be reopened with ``setflags(write=True)``. A deserialized frame carries
    ``image=None``: the record never contained pixels, so none are fabricated.
    """

    frame_id: str
    source_label: str | None
    image: np.ndarray | None
    image_shape: tuple[int, ...]
    image_sha256: str
    ingress: FrameIngressProvenanceV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "frame id must be a non-empty string")
        if self.source_label is not None and not isinstance(self.source_label, str):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "source label must be a string or null")
        shape = _validated_shape(self.image_shape, reason=SESSION_RECORD_INVALID)
        if not _is_sha256(self.image_sha256):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "image digest must be a lowercase SHA-256 hex string")
        if self.ingress is not None:
            if not isinstance(self.ingress, FrameIngressProvenanceV1):
                raise ImageDerivedSessionError(
                    SESSION_RECORD_INVALID, "frame ingress pin has an invalid type")
            if self.ingress.frame_id != self.frame_id:
                raise ImageDerivedSessionError(
                    SESSION_RECORD_INVALID, "frame ingress pin names a different frame")
            if tuple(self.ingress.decoded_shape) != shape:
                raise ImageDerivedSessionError(
                    SESSION_RECORD_INVALID, "frame ingress shape disagrees with the frame")
            if self.ingress.decoded_pixel_sha256 != self.image_sha256:
                raise ImageDerivedSessionError(
                    SESSION_DIGEST_MISMATCH,
                    "frame ingress pixel digest disagrees with the frame")
        if self.image is None:
            return
        actual_shape = _validated_image(self.image, require_immutable=True)
        if actual_shape != shape:
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "image shape does not match the declared shape")
        if image_sha256(self.image) != self.image_sha256:
            raise ImageDerivedSessionError(
                SESSION_DIGEST_MISMATCH, "image digest does not match the pixel data")

    @property
    def has_pixels(self) -> bool:
        """True only for a produced frame; a deserialized frame never has pixels."""
        return self.image is not None


@dataclass(frozen=True, slots=True)
class CameraAdmission:
    frame_id: str
    status: str
    stage: str = "camera_recovery"
    reason: str = ""

    def __post_init__(self):
        if (not isinstance(self.frame_id, str) or not self.frame_id
                or self.status not in ("admitted", "excluded")
                or self.stage != "camera_recovery"
                or self.reason != (CAMERA_ESTIMATE_UNAVAILABLE if self.status == "excluded" else "")):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "invalid camera admission outcome")

    def to_record(self):
        return {"frame_id": self.frame_id, "status": self.status, "stage": self.stage, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ImageDerivedSession:
    """A digest-bound image-derived camera session.

    ``cameras`` holds the unmodified :class:`CameraEstimate` values from
    :func:`image_only.solve_image_cameras`; ``local_cameras`` holds each estimate
    adapted by :func:`image_only.camera_model_from_estimate` at
    :data:`image_only.LOCAL_FRAME_ORIGIN`. ``session_sha256`` is computed from the
    canonical record when omitted and verified when supplied.
    """

    schema: str
    version: int
    frames: tuple[ImageDerivedFrame, ...]
    cameras: Mapping[str, CameraEstimate]
    components: tuple[tuple[str, ...], ...]
    unresolved: tuple[str, ...]
    scale_known: bool
    local_cameras: Mapping[str, CameraModel]
    session_sha256: str | None = None
    input_frames: tuple[ImageDerivedFrame, ...] | None = None
    admissions: tuple[CameraAdmission, ...] = ()
    camera_pooling_caveat: str = CAMERA_POOLING_CAVEAT_LEGACY

    def __post_init__(self) -> None:
        if self.schema != IMAGE_DERIVED_SESSION_SCHEMA:
            raise ImageDerivedSessionError(
                SESSION_SCHEMA_UNSUPPORTED, f"unsupported session schema {self.schema!r}")
        if type(self.version) is not int or self.version not in (1, 2):
            raise ImageDerivedSessionError(
                SESSION_SCHEMA_UNSUPPORTED, f"unsupported session version {self.version!r}")
        if (type(self.camera_pooling_caveat) is not str
                or self.camera_pooling_caveat not in (
                    CAMERA_POOLING_CAVEAT_LEGACY, CAMERA_INDEPENDENT_CAVEAT)):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "invalid camera pooling caveat")
        if (not isinstance(self.frames, tuple) or (not self.frames and self.version == 1)
                or any(not isinstance(frame, ImageDerivedFrame) for frame in self.frames)):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "frames must be a non-empty tuple of frames")
        inventory = self.frames if self.input_frames is None else self.input_frames
        if (not isinstance(inventory, tuple) or not inventory
                or any(not isinstance(frame, ImageDerivedFrame) for frame in inventory)):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "full input inventory required")
        object.__setattr__(self, "input_frames", inventory)
        input_ids = tuple(frame.frame_id for frame in inventory)
        if len(set(input_ids)) != len(input_ids):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "input frame ids must be unique")
        if self.version == 1:
            if (len(inventory) != len(self.frames)
                    or any(frame is not expected for frame, expected in zip(inventory, self.frames))
                    or self.admissions):
                raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "v1 admits every recorded frame")
        else:
            if (not isinstance(self.admissions, tuple)
                    or any(not isinstance(outcome, CameraAdmission) for outcome in self.admissions)
                    or tuple(outcome.frame_id for outcome in self.admissions) != input_ids):
                raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "outcomes must partition inputs in original order")
            for outcome in self.admissions:
                CameraAdmission(outcome.frame_id, outcome.status, outcome.stage, outcome.reason)
            admitted = tuple(frame for frame, outcome in zip(inventory, self.admissions)
                             if outcome.status == "admitted")
            if (tuple(frame.frame_id for frame in self.frames) != tuple(frame.frame_id for frame in admitted)
                    or any(frame is not expected for frame, expected in zip(self.frames, admitted))):
                raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "active frames are not the admitted input subsequence")
        frame_ids = tuple(frame.frame_id for frame in self.frames)
        if len(set(frame_ids)) != len(frame_ids):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "frame ids must be unique")
        if not isinstance(self.cameras, Mapping) or not isinstance(self.local_cameras, Mapping):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "camera maps must be mappings keyed by frame id")
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))
        object.__setattr__(self, "local_cameras", MappingProxyType(dict(self.local_cameras)))
        if tuple(self.cameras) != frame_ids or tuple(self.local_cameras) != frame_ids:
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID,
                "camera maps must key every frame exactly once, in frame order")
        if (not isinstance(self.components, tuple)
                or any(not isinstance(component, tuple)
                       or any(not isinstance(name, str) for name in component)
                       for component in self.components)):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "components must be a tuple of frame-id tuples")
        if self.version == 2:
            component_ids = tuple(name for component in self.components for name in component)
            if set(component_ids) - set(frame_ids) or len(set(component_ids)) != len(component_ids):
                raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "components may name admitted IDs only, without overlap")
        if (not isinstance(self.unresolved, tuple)
                or any(not isinstance(note, str) for note in self.unresolved)):
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, "unresolved notes must be a tuple of strings")
        if self.scale_known is not False:
            raise ImageDerivedSessionError(
                SESSION_FORBIDDEN_CLAIM, "an image-derived session cannot claim metric scale")
        for frame_id in frame_ids:
            estimate = self.cameras[frame_id]
            if (not isinstance(estimate, CameraEstimate)
                    or estimate.intrinsics is None or estimate.pose is None):
                raise ImageDerivedSessionError(
                    SESSION_CAMERA_UNRESOLVED, f"{frame_id}: camera estimate is incomplete")
            if estimate.pose.position is not None:
                raise ImageDerivedSessionError(
                    SESSION_FORBIDDEN_CLAIM,
                    f"{frame_id}: an image-derived estimate cannot claim a position")
            local_camera = self.local_cameras[frame_id]
            if (not isinstance(local_camera, CameraModel)
                    or tuple(float(value) for value in local_camera.position)
                    != _local_origin()):
                raise ImageDerivedSessionError(
                    SESSION_FORBIDDEN_CLAIM,
                    f"{frame_id}: the local camera must sit at LOCAL_FRAME_ORIGIN")
        _require_unresolved_facts(self.unresolved, frame_ids)
        for frame_id in frame_ids:
            if self.local_cameras[frame_id] != _adapt(self.cameras[frame_id], frame_id):
                raise ImageDerivedSessionError(
                    SESSION_RECORD_INVALID,
                    f"{frame_id}: local camera is not the adapted estimate")
        expected = _digest(_session_payload(self))
        if self.session_sha256 is None:
            object.__setattr__(self, "session_sha256", expected)
        elif not _is_sha256(self.session_sha256):
            raise ImageDerivedSessionError(
                SESSION_DIGEST_MISMATCH, "session digest is not a lowercase SHA-256 hex string")
        elif self.session_sha256 != expected:
            raise ImageDerivedSessionError(
                SESSION_DIGEST_MISMATCH, "session digest does not match the record")

    @property
    def has_pixels(self) -> bool:
        """True only for a produced session; a deserialized session has no pixels."""
        return all(frame.image is not None for frame in self.input_frames)

    @property
    def admission_summary(self) -> dict[str, Any]:
        return {"original": len(self.input_frames), "admitted": len(self.frames),
                "excluded": len(self.input_frames) - len(self.frames),
                "exclusions": [outcome.to_record() for outcome in self.admissions
                               if outcome.status == "excluded"]}


def image_sha256(image: np.ndarray) -> str:
    """Lowercase SHA-256 over a frame's exact shape-framed C-order pixel bytes.

    The frame's rank, every dimension, and the dtype are bound ahead of the
    bytes, so two frames that merely concatenate to the same byte stream cannot
    collide on one digest.
    """
    array = np.asarray(image)
    framed = [_IMAGE_DIGEST_DOMAIN, bytes([array.ndim])]
    for dimension in array.shape:
        framed.append(int(dimension).to_bytes(8, "little", signed=False))
    framed.append(array.dtype.str.encode("ascii"))
    framed.append(np.ascontiguousarray(array).tobytes(order="C"))
    return hashlib.sha256(b"".join(framed)).hexdigest()


def build_image_derived_session(
    frames: Mapping[str, np.ndarray],
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    ingress: Mapping[str, FrameIngressProvenanceV1] | None = None,
) -> ImageDerivedSession:
    """Produce a typed session from ordered frames and optional source labels.

    ``frames`` maps a non-empty ``frame_id`` to a ``uint8`` grayscale
    ``(height, width)`` or RGB ``(height, width, 3)`` array, in the order the
    session must preserve. Each frame's exact C-order bytes are captured into a
    private ``bytes``-backed read-only array before anything else sees it, so
    neither the caller nor a delegated solver can mutate or reopen it, and
    neither the session nor its digests can change. ``source_paths`` only labels
    frames; it never supplies geometry.

    Camera recovery is delegated, unchanged, to
    :func:`image_only.solve_image_cameras`, and every solved estimate is adapted
    by :func:`image_only.camera_model_from_estimate` at
    :data:`image_only.LOCAL_FRAME_ORIGIN`. A frame with no solved estimate, an
    estimate that cannot become a camera, or a recovered solution that claims
    scale remains a hard error. Only an absent estimate becomes an exclusion;
    partial sessions bind every captured input and outcome in schema v2.
    """
    ordered = _ordered_frames(frames)
    labels = _source_labels(source_paths, {frame_id for frame_id, _ in ordered})
    ingress_by_frame = _ingress_records(ingress, {frame_id for frame_id, _ in ordered})

    captured: list[ImageDerivedFrame] = []
    for frame_id, image in ordered:
        shape = _validated_image(image, require_immutable=False)
        pixels = _immutable_pixels(image)
        captured.append(ImageDerivedFrame(
            frame_id=frame_id,
            source_label=labels.get(frame_id),
            image=pixels,
            image_shape=shape,
            image_sha256=image_sha256(pixels),
            ingress=ingress_by_frame.get(frame_id),
        ))

    frame_ids = [frame.frame_id for frame in captured]
    try:
        solution = image_only.solve_image_cameras(
            {frame.frame_id: frame.image for frame in captured})
    except image_only.ImageOnlyCameraError as error:
        raise ImageDerivedSessionError(SESSION_CAMERA_UNRESOLVED, str(error)) from error
    except Exception as error:
        raise ImageDerivedSessionError(SESSION_CAMERA_UNRESOLVED,
                                       f"camera recovery runtime failure: {type(error).__name__}: {error}") from error
    if not isinstance(solution, CameraSolution):
        raise ImageDerivedSessionError(
            SESSION_CAMERA_UNRESOLVED, "camera recovery did not return a CameraSolution")

    if not isinstance(solution.cameras, Mapping) or set(solution.cameras) - set(frame_ids):
        raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "camera recovery returned foreign or malformed IDs")
    if solution.scale_known is not False:
        raise ImageDerivedSessionError(SESSION_FORBIDDEN_CLAIM, "camera recovery claimed scale")
    if (not isinstance(solution.components, tuple)
            or any(not isinstance(component, tuple) or any(not isinstance(name, str) for name in component)
                   for component in solution.components)
            or not isinstance(solution.unresolved, tuple)
            or any(not isinstance(note, str) for note in solution.unresolved)):
        raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "malformed camera uncertainty/components")
    cameras: dict[str, CameraEstimate] = {}
    outcomes = []
    for frame_id in frame_ids:
        if frame_id not in solution.cameras:
            outcomes.append(CameraAdmission(frame_id, "excluded", reason=CAMERA_ESTIMATE_UNAVAILABLE))
            continue
        estimate = solution.cameras.get(frame_id)
        if not isinstance(estimate, CameraEstimate):
            raise ImageDerivedSessionError(
                SESSION_CAMERA_UNRESOLVED,
                f"{frame_id}: camera recovery returned no estimate for this frame")
        try:
            _camera_payload(frame_id, estimate)  # invalid/forbidden estimates are never abstentions
        except ImageDerivedSessionError as error:
            error.review_evidence = (frame_id, estimate)  # retained in-memory, never admitted
            raise
        cameras[frame_id] = estimate
        outcomes.append(CameraAdmission(frame_id, "admitted"))

    partial = len(cameras) != len(captured)
    active = tuple(frame for frame in captured if frame.frame_id in cameras)
    if partial and any(name not in cameras for component in solution.components for name in component):
        raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "camera component names an excluded or foreign input")

    return ImageDerivedSession(
        schema=IMAGE_DERIVED_SESSION_SCHEMA,
        version=2 if partial else IMAGE_DERIVED_SESSION_VERSION,
        frames=active,
        cameras=cameras,
        components=tuple(tuple(component) for component in solution.components),
        unresolved=tuple(solution.unresolved) + tuple(
            f"{outcome.frame_id}: camera estimate unavailable; position, scale and orientation unknown"
            for outcome in outcomes if outcome.status == "excluded"),
        scale_known=solution.scale_known,
        local_cameras={frame_id: _adapt(cameras[frame_id], frame_id)
                       for frame_id in cameras},
        input_frames=tuple(captured) if partial else None,
        admissions=tuple(outcomes) if partial else (),
        camera_pooling_caveat=CAMERA_INDEPENDENT_CAVEAT,
    )


def image_derived_session_to_record(session: ImageDerivedSession) -> dict[str, Any]:
    """Canonical record carrying only session identity, digests, and cameras."""
    if not isinstance(session, ImageDerivedSession):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "expected an ImageDerivedSession")
    record = _session_payload(session)
    record["session_sha256"] = session.session_sha256
    return record


def image_derived_session_from_record(record: Mapping[str, Any]) -> ImageDerivedSession:
    """Rebuild and verify a session from :func:`image_derived_session_to_record`.

    The record is checked for its exact key set, schema and version, frame and
    camera shapes, digests, forbidden translation/scale claims, retained
    unresolved facts, and its deterministic session digest. Pixels are not
    present in a record and are never fabricated: every rebuilt frame keeps
    ``image=None``.
    """
    if not isinstance(record, Mapping):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "a session record must be a mapping")
    version = record.get("version")
    if type(version) is not int or version not in (1, 2):
        raise ImageDerivedSessionError(SESSION_SCHEMA_UNSUPPORTED, f"unsupported session version {version!r}")
    if set(record) != (_SESSION_V2_KEYS if version == 2 else _SESSION_KEYS):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "session record has unexpected or missing keys")
    if record["schema"] != IMAGE_DERIVED_SESSION_SCHEMA:
        raise ImageDerivedSessionError(
            SESSION_SCHEMA_UNSUPPORTED, f"unsupported session schema {record['schema']!r}")
    if type(record["version"]) is not int \
            or record["version"] not in (1, 2):
        raise ImageDerivedSessionError(
            SESSION_SCHEMA_UNSUPPORTED, f"unsupported session version {record['version']!r}")

    frames_payload = record["frames"]
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "a session record needs a non-empty frame list")
    frames = tuple(_frame_from_payload(item) for item in frames_payload)
    frame_ids = tuple(frame.frame_id for frame in frames)
    if len(set(frame_ids)) != len(frame_ids):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "session record has duplicate frame ids")

    input_frames = frames
    admissions = ()
    camera_pooling_caveat = CAMERA_POOLING_CAVEAT_LEGACY
    if version == 2:
        active_ids = record["active_frame_ids"]
        if (not isinstance(active_ids, list) or any(not isinstance(name, str) for name in active_ids)
                or len(set(active_ids)) != len(active_ids) or set(active_ids) - set(frame_ids)):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "invalid active frame IDs")
        outcomes = record["admissions"]
        if (not isinstance(outcomes, list) or any(not isinstance(item, Mapping)
                or set(item) != {"frame_id", "status", "stage", "reason"} for item in outcomes)
                or type(record["camera_pooling_caveat"]) is not str
                or record["camera_pooling_caveat"] not in (
                    CAMERA_POOLING_CAVEAT_LEGACY, CAMERA_INDEPENDENT_CAVEAT)):
            raise ImageDerivedSessionError(SESSION_RECORD_INVALID, "invalid admissions/camera pooling caveat")
        camera_pooling_caveat = record["camera_pooling_caveat"]
        admissions = tuple(CameraAdmission(**item) for item in outcomes)
        by_id = {frame.frame_id: frame for frame in frames}
        frames = tuple(by_id[name] for name in active_ids)
        frame_ids = tuple(active_ids)

    cameras = _keyed_cameras(record["cameras"], len(frame_ids), "cameras",
                             _camera_from_payload)
    local_cameras = _keyed_cameras(record["local_cameras"], len(frame_ids),
                                   "local_cameras", _local_camera_from_payload)

    components_payload = record["components"]
    if (not isinstance(components_payload, list)
            or any(not isinstance(component, list)
                   or any(not isinstance(name, str) for name in component)
                   for component in components_payload)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "components must be a list of string lists")
    unresolved_payload = record["unresolved"]
    if (not isinstance(unresolved_payload, list)
            or any(not isinstance(note, str) for note in unresolved_payload)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "unresolved must be a list of strings")
    scale_known = record["scale_known"]
    if not isinstance(scale_known, bool):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "scale_known must be a boolean")

    return ImageDerivedSession(
        schema=record["schema"],
        version=record["version"],
        frames=frames,
        cameras=cameras,
        components=tuple(tuple(component) for component in components_payload),
        unresolved=tuple(unresolved_payload),
        scale_known=scale_known,
        local_cameras=local_cameras,
        session_sha256=record["session_sha256"],
        input_frames=input_frames if version == 2 else None,
        admissions=admissions,
        camera_pooling_caveat=camera_pooling_caveat,
    )


# --- Producer-side validation -------------------------------------------------

def _ordered_frames(frames: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    if not isinstance(frames, Mapping):
        raise ImageDerivedSessionError(
            SESSION_INPUT_INVALID,
            "frames must be an ordered mapping of frame id to image")
    if not frames:
        raise ImageDerivedSessionError(
            SESSION_INPUT_INVALID, "an image-derived session needs at least one frame")
    ordered: list[tuple[str, np.ndarray]] = []
    for frame_id, image in frames.items():
        if not isinstance(frame_id, str) or not frame_id:
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID, "frame ids must be non-empty strings")
        ordered.append((frame_id, image))
    return ordered


def _source_labels(
    source_paths: Mapping[str, str | Path] | None, frame_ids: set[str],
) -> dict[str, str]:
    if source_paths is None:
        return {}
    if not isinstance(source_paths, Mapping):
        raise ImageDerivedSessionError(
            SESSION_INPUT_INVALID,
            "source_paths must be a mapping of frame id to source label")
    labels: dict[str, str] = {}
    for frame_id, label in source_paths.items():
        if frame_id not in frame_ids:
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID, f"source label for unknown frame {frame_id!r}")
        if isinstance(label, Path):
            label = str(label)
        if not isinstance(label, str) or not label:
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID,
                f"{frame_id}: source label must be a non-empty string or path")
        labels[frame_id] = label
    return labels


def _ingress_records(
    ingress: Mapping[str, FrameIngressProvenanceV1] | None, frame_ids: set[str],
) -> dict[str, FrameIngressProvenanceV1]:
    if ingress is None:
        return {}
    if not isinstance(ingress, Mapping):
        raise ImageDerivedSessionError(
            SESSION_INPUT_INVALID,
            "ingress must be a mapping of frame id to ingress provenance")
    records: dict[str, FrameIngressProvenanceV1] = {}
    for frame_id, record in ingress.items():
        if frame_id not in frame_ids:
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID, f"ingress pin for unknown frame {frame_id!r}")
        if not isinstance(record, FrameIngressProvenanceV1):
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID, f"{frame_id}: ingress pin has an invalid type")
        if record.frame_id != frame_id:
            raise ImageDerivedSessionError(
                SESSION_INPUT_INVALID, f"{frame_id}: ingress pin names another frame")
        records[frame_id] = record
    return records


def _validated_image(image: Any, *, require_immutable: bool) -> tuple[int, ...]:
    if not isinstance(image, np.ndarray):
        raise ImageDerivedSessionError(
            SESSION_FRAME_INVALID, "frame image must be a numpy array")
    if image.dtype != np.uint8:
        raise ImageDerivedSessionError(
            SESSION_FRAME_INVALID, f"frame image must be uint8, got {image.dtype}")
    shape = _validated_shape(tuple(image.shape), reason=SESSION_FRAME_INVALID)
    if require_immutable:
        if image.flags.writeable:
            raise ImageDerivedSessionError(
                SESSION_FRAME_INVALID,
                "session frame images must be immutable copies (writeable=False)")
        if not _has_immutable_bytes_base(image):
            raise ImageDerivedSessionError(
                SESSION_FRAME_INVALID,
                "session frame images must be backed by immutable bytes, not "
                "reopenable numpy-owned memory")
    return shape


def _immutable_pixels(image: np.ndarray) -> np.ndarray:
    """Capture a frame's exact bytes in a read-only, non-reopenable array.

    ``np.array(image, copy=True)`` allocates writeable memory owned by numpy, so
    ``setflags(write=True)`` can reopen it and a delegated solver could silently
    change captured pixels. Buffering the exact C-order bytes makes an immutable
    ``bytes`` object the array's base, which numpy cannot reopen; the shape and
    every digest are preserved exactly.
    """
    return np.frombuffer(image.tobytes(order="C"), dtype=np.uint8).reshape(image.shape)


def _has_immutable_bytes_base(image: np.ndarray) -> bool:
    """True when the array's ``base`` chain terminates at an immutable ``bytes``.

    A numpy-owned array (``owndata is True``) or a view into another numpy array
    is reopenable with ``setflags(write=True)``; only a buffer whose ultimate
    base is ``bytes`` is genuinely non-reopenable.
    """
    base: Any = image
    while base is not None:
        if isinstance(base, bytes):
            return True
        if not isinstance(base, np.ndarray):
            return False
        base = base.base
    return False


def _validated_shape(shape: Any, *, reason: str) -> tuple[int, ...]:
    if not isinstance(shape, (tuple, list)) or len(shape) not in (2, 3):
        raise ImageDerivedSessionError(
            reason, "image shape must be (height, width) or (height, width, 3)")
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ImageDerivedSessionError(
            reason, "image dimensions must be positive exact integers")
    if len(shape) == 3 and shape[2] != 3:
        raise ImageDerivedSessionError(
            reason, "an RGB image must have exactly three channels")
    return tuple(int(dimension) for dimension in shape)


# --- Record parsing -----------------------------------------------------------

def _frame_from_payload(payload: Any) -> ImageDerivedFrame:
    if not isinstance(payload, Mapping) or set(payload) not in (
            _FRAME_KEYS, _FRAME_KEYS_INGRESS):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "frame record has unexpected or missing keys")
    frame_id = payload["frame_id"]
    if not isinstance(frame_id, str) or not frame_id:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "frame id must be a non-empty string")
    label = payload["source_label"]
    if label is not None and not isinstance(label, str):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "source label must be a string or null")
    shape = _validated_shape(payload["image_shape"], reason=SESSION_RECORD_INVALID)
    digest = payload["image_sha256"]
    if not _is_sha256(digest):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "image digest must be a lowercase SHA-256 hex string")
    ingress = None
    if "ingress" in payload:
        try:
            ingress = FrameIngressProvenanceV1.from_record(payload["ingress"])
        except (IngressProvenanceError, KeyError, TypeError, ValueError) as error:
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID,
                f"frame ingress record is not usable: {error}") from error
    return ImageDerivedFrame(
        frame_id=frame_id, source_label=label, image=None,
        image_shape=shape, image_sha256=digest, ingress=ingress)


def _keyed_cameras(
    payload: Any, expected: int, name: str, parser,
) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{name} must be a list")
    if len(payload) != expected:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID,
            f"{name} must carry exactly one entry per frame")
    keyed: dict[str, Any] = {}
    for entry in payload:
        frame_id, value = parser(entry)
        if frame_id in keyed:
            raise ImageDerivedSessionError(
                SESSION_RECORD_INVALID, f"{name} repeats frame {frame_id!r}")
        keyed[frame_id] = value
    return keyed


def _camera_from_payload(payload: Any) -> tuple[str, CameraEstimate]:
    if not isinstance(payload, Mapping) or set(payload) != _CAMERA_KEYS:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "camera record has unexpected or missing keys")
    frame_id = payload["frame_id"]
    if not isinstance(frame_id, str) or not frame_id:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "camera record needs a non-empty frame id")
    intrinsics = _intrinsics_from_payload(payload["intrinsics"])
    pose_payload = payload["pose"]
    if not isinstance(pose_payload, Mapping) or set(pose_payload) != _POSE_KEYS:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "camera pose record has unexpected or missing keys")
    if pose_payload["position"] is not None:
        raise ImageDerivedSessionError(
            SESSION_FORBIDDEN_CLAIM, f"{frame_id}: serialized camera claims a position")
    pose = CameraPose(
        position=None,
        yaw_degrees=_finite_float(pose_payload["yaw_degrees"], detail="yaw"),
        pitch_degrees=_finite_float(pose_payload["pitch_degrees"], detail="pitch"),
        roll_degrees=_finite_float(pose_payload["roll_degrees"], detail="roll"),
    )
    return frame_id, CameraEstimate(
        intrinsics=intrinsics,
        pose=pose,
        residual_pixels=_optional_finite(payload["residual_pixels"]),
        confidence=_finite_float(payload["confidence"], detail="confidence"),
    )


def _local_camera_from_payload(payload: Any) -> tuple[str, CameraModel]:
    if not isinstance(payload, Mapping) or set(payload) != _LOCAL_CAMERA_KEYS:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "local camera record has unexpected or missing keys")
    frame_id = payload["frame_id"]
    if not isinstance(frame_id, str) or not frame_id:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "local camera record needs a non-empty frame id")
    position = payload["position"]
    if not isinstance(position, list) or len(position) != 3:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "local camera position must be three numbers")
    point = tuple(_finite_float(value, detail="position") for value in position)
    if point != _local_origin():
        raise ImageDerivedSessionError(
            SESSION_FORBIDDEN_CLAIM,
            f"{frame_id}: local camera must sit at LOCAL_FRAME_ORIGIN")
    intrinsics_payload = payload["intrinsics"]
    intrinsics = (None if intrinsics_payload is None
                  else _intrinsics_from_payload(intrinsics_payload))
    try:
        camera = CameraModel(
            position=point,
            yaw_degrees=_finite_float(payload["yaw_degrees"], detail="yaw"),
            pitch_degrees=_finite_float(payload["pitch_degrees"], detail="pitch"),
            fov_degrees=_finite_float(payload["fov_degrees"], detail="fov"),
            eye_height=_finite_float(payload["eye_height"], detail="eye height"),
            intrinsics=intrinsics,
            roll_degrees=_finite_float(payload["roll_degrees"], detail="roll"),
        )
    except ValueError as error:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{frame_id}: local camera is not usable: {error}") from error
    return frame_id, camera


def _intrinsics_from_payload(payload: Any) -> CameraIntrinsics:
    if not isinstance(payload, Mapping) or set(payload) != _INTRINSICS_KEYS:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "intrinsics record has unexpected or missing keys")
    size = payload["image_size"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in size)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "image size must be two positive exact integers")
    focal = _pair(payload["focal_length_pixels"], "focal length")
    principal = _pair(payload["principal_point_pixels"], "principal point")
    return CameraIntrinsics(
        image_size=(size[0], size[1]),
        focal_length_pixels=(_positive_float(focal[0], detail="focal length"),
                             _positive_float(focal[1], detail="focal length")),
        principal_point_pixels=(_finite_float(principal[0], detail="principal point"),
                                _finite_float(principal[1], detail="principal point")),
    )


# --- Canonical payload and digests -------------------------------------------

def _session_payload(session: ImageDerivedSession) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": session.schema,
        "version": session.version,
        "frames": [_frame_payload(frame) for frame in (session.input_frames if session.version == 2 else session.frames)],
        "cameras": [],
        "components": [list(component) for component in session.components],
        "unresolved": list(session.unresolved),
        "scale_known": bool(session.scale_known),
        "local_cameras": [],
    }
    if session.version == 2:
        payload.update(active_frame_ids=[frame.frame_id for frame in session.frames],
                       admissions=[outcome.to_record() for outcome in session.admissions],
                       camera_pooling_caveat=session.camera_pooling_caveat)
    for frame in session.frames:
        frame_id = frame.frame_id
        payload["cameras"].append(_camera_payload(frame_id, session.cameras[frame_id]))
        payload["local_cameras"].append(
            _local_camera_payload(frame_id, session.local_cameras[frame_id]))
    return payload


def _frame_payload(frame: ImageDerivedFrame) -> dict[str, Any]:
    payload = {
        "frame_id": frame.frame_id,
        "source_label": frame.source_label,
        "image_shape": [int(dimension) for dimension in frame.image_shape],
        "image_sha256": frame.image_sha256,
    }
    if frame.ingress is not None:
        payload["ingress"] = frame.ingress.to_record()
    return payload


def _camera_payload(frame_id: str, estimate: CameraEstimate) -> dict[str, Any]:
    if not isinstance(estimate, CameraEstimate):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{frame_id}: camera is not a CameraEstimate")
    if estimate.intrinsics is None or estimate.pose is None:
        raise ImageDerivedSessionError(
            SESSION_CAMERA_UNRESOLVED, f"{frame_id}: camera estimate is incomplete")
    return {
        "frame_id": frame_id,
        "intrinsics": _intrinsics_payload(estimate.intrinsics),
        "pose": _pose_payload(estimate.pose, frame_id),
        "residual_pixels": _optional_finite(estimate.residual_pixels),
        "confidence": _finite_float(estimate.confidence, detail="confidence"),
    }


def _pose_payload(pose: CameraPose, frame_id: str) -> dict[str, Any]:
    if not isinstance(pose, CameraPose):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{frame_id}: pose is not a CameraPose")
    if pose.position is not None:
        raise ImageDerivedSessionError(
            SESSION_FORBIDDEN_CLAIM, f"{frame_id}: estimate claims a position")
    return {
        "position": None,
        "yaw_degrees": _finite_float(pose.yaw_degrees, detail="yaw"),
        "pitch_degrees": _finite_float(pose.pitch_degrees, detail="pitch"),
        "roll_degrees": _finite_float(pose.roll_degrees, detail="roll"),
    }


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "intrinsics are not a CameraIntrinsics")
    size = intrinsics.image_size
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in size)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "image size must be two positive exact integers")
    focal = _pair(intrinsics.focal_length_pixels, "focal length")
    principal = _pair(intrinsics.principal_point_pixels, "principal point")
    return {
        "image_size": [int(size[0]), int(size[1])],
        "focal_length_pixels": [_positive_float(focal[0], detail="focal length"),
                                _positive_float(focal[1], detail="focal length")],
        "principal_point_pixels": [_finite_float(principal[0], detail="principal point"),
                                   _finite_float(principal[1], detail="principal point")],
    }


def _local_camera_payload(frame_id: str, camera: CameraModel) -> dict[str, Any]:
    if not isinstance(camera, CameraModel):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{frame_id}: local camera is not a CameraModel")
    point = tuple(_finite_float(value, detail="position") for value in camera.position)
    if point != _local_origin():
        raise ImageDerivedSessionError(
            SESSION_FORBIDDEN_CLAIM,
            f"{frame_id}: local camera must sit at LOCAL_FRAME_ORIGIN")
    return {
        "frame_id": frame_id,
        "position": list(point),
        "yaw_degrees": _finite_float(camera.yaw_degrees, detail="yaw"),
        "pitch_degrees": _finite_float(camera.pitch_degrees, detail="pitch"),
        "fov_degrees": _finite_float(camera.fov_degrees, detail="fov"),
        "eye_height": _finite_float(camera.eye_height, detail="eye height"),
        "roll_degrees": _finite_float(camera.roll_degrees, detail="roll"),
        "intrinsics": (None if camera.intrinsics is None
                       else _intrinsics_payload(camera.intrinsics)),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


# --- Shared helpers -----------------------------------------------------------

def _adapt(estimate: CameraEstimate, frame_id: str) -> CameraModel:
    try:
        return image_only.camera_model_from_estimate(
            estimate, position=image_only.LOCAL_FRAME_ORIGIN)
    except image_only.ImageOnlyCameraError as error:
        raise ImageDerivedSessionError(
            SESSION_CAMERA_UNRESOLVED, f"{frame_id}: {error}") from error


def _require_unresolved_facts(
    unresolved: tuple[str, ...], frame_ids: tuple[str, ...],
) -> None:
    """Require the exact canonical ``to_camera_solution`` note for every fact.

    Each frame must carry its three unresolved facts as the verbatim strings
    emitted by ``calibration.to_camera_solution`` (reconstruction/calibration.py
    lines 312-314), exactly once each, with no other note prefixed by the frame
    id. Substring or prefix matching would accept a resealed ``"<id>: position"``
    that merely names the fact while dropping the unresolved wording, so it is
    deliberately not used.
    """
    for frame_id in frame_ids:
        prefix = f"{frame_id}: "
        expected = [template.format(frame_id=frame_id)
                    for _, template in _UNRESOLVED_FACT_NOTES]
        for note in unresolved:
            if note.startswith(prefix) and note not in expected:
                raise ImageDerivedSessionError(
                    SESSION_UNRESOLVED_FACT_MISSING,
                    f"{frame_id}: unresolved note is not a canonical fact: {note!r}")
        for fact, template in _UNRESOLVED_FACT_NOTES:
            if unresolved.count(template.format(frame_id=frame_id)) != 1:
                raise ImageDerivedSessionError(
                    SESSION_UNRESOLVED_FACT_MISSING,
                    f"{frame_id}: {fact} must stay explicitly unresolved")


def _local_origin() -> tuple[float, float, float]:
    return tuple(float(value) for value in image_only.LOCAL_FRAME_ORIGIN)  # type: ignore[return-value]


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _pair(value: Any, detail: str) -> Any:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{detail} must be a two-element sequence")
    return value


def _finite_float(value: Any, *, detail: str) -> float:
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{detail} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{detail} must be finite")
    return result


def _positive_float(value: Any, *, detail: str) -> float:
    result = _finite_float(value, detail=detail)
    if result <= 0.0:
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, f"{detail} must be positive")
    return result


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ImageDerivedSessionError(
            SESSION_RECORD_INVALID, "residual pixels must be a finite number or null")
    result = float(value)
    return result if math.isfinite(result) else None


__all__ = [
    "IMAGE_DERIVED_SESSION_SCHEMA",
    "IMAGE_DERIVED_SESSION_VERSION",
    "IMAGE_DERIVED_SESSION_PARTIAL_VERSION",
    "CameraAdmission",
    "SESSION_INPUT_INVALID",
    "SESSION_FRAME_INVALID",
    "SESSION_CAMERA_UNRESOLVED",
    "SESSION_RECORD_INVALID",
    "SESSION_SCHEMA_UNSUPPORTED",
    "SESSION_DIGEST_MISMATCH",
    "SESSION_UNRESOLVED_FACT_MISSING",
    "SESSION_FORBIDDEN_CLAIM",
    "ImageDerivedFrame",
    "ImageDerivedSession",
    "ImageDerivedSessionError",
    "build_image_derived_session",
    "image_derived_session_to_record",
    "image_derived_session_from_record",
    "image_sha256",
]
