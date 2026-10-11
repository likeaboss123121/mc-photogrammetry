"""Bounded M7 normal-mode raw-image entry seam.

This is the one production entry that starts from *ordinary images* and walks the
accepted image-derived chain as far as it honestly can:

    ordinary image paths / directories
        -> :class:`~mc_photogrammetry.pipeline.image_derived_session.ImageDerivedSession`
        -> :class:`~mc_photogrammetry.pipeline.image_derived_connections.ImageDerivedConnectionGraph`
        -> :class:`~mc_photogrammetry.pipeline.image_derived_pose_graph.ImageDerivedPoseGraph`
        -> a persisted schema-2 ``.mcscan`` session record
        -> a typed camera- or geometry-unavailable terminal result

It owns exactly the raw-image boundary the accepted M4/M6 seams left open: it
discovers and decodes still images, delegates every camera/graph/pose decision to
the accepted adapters unchanged, declares the connection and pose-graph budgets
explicitly, persists the accepted session record through the existing project
store, and stops at a typed terminal. It does **not** construct a
``PlacementUniverse``, aligned region support, a resolved face placement, a plane
volume, a face observation, a proposal, or a promotion, and it never calls
:func:`~mc_photogrammetry.pipeline.image_derived_orchestration.compose_image_derived_pipeline`.
Those V1/V2/V3/M5 records require geometry (a lattice, supports, a placement) that
this entry does not construct, so it reports an unavailable terminal instead of
fabricating geometry.

The terminal result
-------------------

:class:`ImageDerivedEntryResult` carries the full input inventory and its camera
admission outcomes. Missing estimates become explicit exclusions; admitted frames
retain their original IDs and alone reach the connection and pose adapters. Zero
admitted cameras yields ``camera_unavailable`` without graph or pose invocation.
One admitted camera retains local review only, with no pair matching or relative
geometry. Otherwise the terminal is :data:`GEOMETRY_UNAVAILABLE`, and the graphs
may themselves carry typed refusals. The result also carries the canonical
persisted session record, declared budgets, full source paths and deterministic
entry digest. This entry does not promote geometry.

Input discipline
----------------

The public call shape accepts only ordinary still-image paths (files or
directories) plus explicit budget caps and the accepted injectable test seams
(``matcher``). There is no parameter for a camera manifest, a scene/bounds file,
an F3/OCR value, a coordinate, a direction, a scale, a translation, or an
annotation, and no ``**kwargs`` through which one could arrive. Inputs are
resolved by the accepted :func:`input.media.discover_media` (stable path order,
exact-sha256 de-duplication) and decoded by the accepted
:func:`input.image.load_image`. Video sources are refused: this seam is for
ordinary images only.

Hard non-claims
---------------

No metric scale, camera translation, world coordinate, bound, scene transform or
block volume is produced, inferred or persisted. The session's ``scale_known``,
the graph's ``scale_known`` and the pose graph's ``metric_scale_known`` all stay
``False``; the persisted volume is empty, and the persisted payload carries only
the canonical digest-bound session record. Every refusal reason is a typed token
from this module or from the accepted connection/pose-graph seams.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from mc_photogrammetry.domain.models import (
    SCAN_PROJECT_SCHEMA_VERSION,
    ScanProject,
)
from mc_photogrammetry.input.image import (
    ImageDecodeError,
    load_image_bytes,
)
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.pipeline import image_derived_connections as connections_mod
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.pipeline.frame_ingress import (
    IngressProvenanceError,
    build_frame_ingress_provenance,
)
from mc_photogrammetry.pipeline.image_derived_connections import (
    CONNECTIONS_REFUSAL_REASONS,
    ConnectionBudgets,
    ImageDerivedConnectionError,
    ImageDerivedConnectionGraph,
    RelativeCameraEvidence,
    YawClassDiagnostic,
    make_connection_budgets,
)
from mc_photogrammetry.pipeline.image_derived_pose_graph import (
    POSE_GRAPH_REFUSAL_REASONS,
    GaugeCameraPose,
    ImageDerivedPoseGraph,
    ImageDerivedPoseGraphError,
    PoseGraphBudgets,
    make_pose_graph_budgets,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    ImageDerivedFrame,
    ImageDerivedSession,
    ImageDerivedSessionError,
    build_image_derived_session,
)
from mc_photogrammetry.pipeline.joint_matching import (
    JointPairResult,
    match_and_solve_pair,
)
from mc_photogrammetry.project.store import (
    ProjectStoreError,
    load_project,
    save_project,
)

__all__ = [
    "IMAGE_DERIVED_ENTRY_SCHEMA",
    "IMAGE_DERIVED_ENTRY_PRODUCER",
    "GEOMETRY_UNAVAILABLE",
    "ENTRY_INPUT_INVALID",
    "ENTRY_VIDEO_UNSUPPORTED",
    "ENTRY_NO_IMAGES",
    "ENTRY_FRAME_LIMIT",
    "ENTRY_IMAGE_DECODE_FAILED",
    "ENTRY_SESSION_FAILED",
    "ENTRY_CONNECTION_FAILED",
    "ENTRY_POSE_GRAPH_FAILED",
    "ENTRY_BUDGET_INVALID",
    "ENTRY_PERSIST_FAILED",
    "ENTRY_ERROR_REASONS",
    "ENTRY_TERMINAL_REASONS",
    "DEFAULT_MAX_FRAMES",
    "MAX_FRAMES_CAP",
    "DEFAULT_MAX_PAIRS",
    "MAX_PAIRS_CAP",
    "DEFAULT_MAX_TOTAL_SUPPORT",
    "MAX_TOTAL_SUPPORT_CAP",
    "DEFAULT_MAX_EDGES",
    "MAX_EDGES_CAP",
    "DEFAULT_MAX_CYCLE_CHECKS",
    "MAX_CYCLE_CHECKS_CAP",
    "DEFAULT_ROTATION_TOLERANCE_DEGREES",
    "DEFAULT_DIRECTION_TOLERANCE",
    "ImageDerivedEntryError",
    "ImageDerivedEntryResult",
    "build_image_derived_entry",
]

IMAGE_DERIVED_ENTRY_SCHEMA = "mc-photogrammetry-image-derived-entry"
"""Explicit schema name for the terminal entry result."""

IMAGE_DERIVED_ENTRY_PRODUCER = "m7-image-derived-entry-1"
"""Producer identity carried by every terminal entry result."""

GEOMETRY_UNAVAILABLE = "geometry_unavailable"
"""The only terminal a normal-mode raw-image entry can reach.

Images plus the accepted image-derived chain establish cameras, relative
connections and relative gauge poses. They do not establish the V1/V2/V3/M5
geometry the accepted orchestration facade consumes, and this seam never invents
it, so the honest terminal is exactly this token.
"""

ENTRY_INPUT_INVALID = "entry_input_invalid"
"""Raised reason: the image input collection, or a budget, is malformed."""

ENTRY_VIDEO_UNSUPPORTED = "entry_video_unsupported"
"""Raised reason: a video source was supplied; this seam is still images only."""

ENTRY_NO_IMAGES = "entry_no_images"
"""Raised reason: discovery found no ordinary still image."""

ENTRY_FRAME_LIMIT = "entry_frame_limit"
"""Raised reason: more images were discovered than the declared frame cap."""

ENTRY_IMAGE_DECODE_FAILED = "entry_image_decode_failed"
"""Raised reason: a discovered still image could not be decoded."""

ENTRY_SESSION_FAILED = "entry_session_failed"
"""Raised reason: the accepted session producer refused the image evidence."""

ENTRY_CONNECTION_FAILED = "entry_connection_failed"
"""Raised reason: the accepted connection adapter reported a contract violation."""

ENTRY_POSE_GRAPH_FAILED = "entry_pose_graph_failed"
"""Raised reason: the accepted pose-graph composer reported a contract violation."""

ENTRY_BUDGET_INVALID = "entry_budget_invalid"
"""Raised reason: a declared budget is negative, mistyped, or above its hard cap."""

ENTRY_PERSIST_FAILED = "entry_persist_failed"
"""Raised reason: the schema-2 session could not be persisted or did not replay."""

ENTRY_ERROR_REASONS: tuple[str, ...] = (
    ENTRY_INPUT_INVALID,
    ENTRY_VIDEO_UNSUPPORTED,
    ENTRY_NO_IMAGES,
    ENTRY_FRAME_LIMIT,
    ENTRY_IMAGE_DECODE_FAILED,
    ENTRY_SESSION_FAILED,
    ENTRY_CONNECTION_FAILED,
    ENTRY_POSE_GRAPH_FAILED,
    ENTRY_BUDGET_INVALID,
    ENTRY_PERSIST_FAILED,
)
"""Every typed ``reason`` an :class:`ImageDerivedEntryError` may carry."""

ENTRY_TERMINAL_REASONS: tuple[str, ...] = (
    CONNECTIONS_REFUSAL_REASONS + POSE_GRAPH_REFUSAL_REASONS + ("camera_unavailable", "single_camera_local_review")
)
"""Every typed refusal a terminal result's ``reason`` may carry."""

DEFAULT_MAX_FRAMES = 12
"""Default still-image cap for one normal-mode entry."""

MAX_FRAMES_CAP = 32
"""Hard upper bound on the declared frame cap."""

MAX_PAIRS_CAP = MAX_FRAMES_CAP * (MAX_FRAMES_CAP - 1) // 2
"""Hard upper bound on the declared pair budget: ``C(MAX_FRAMES_CAP, 2)``."""

MAX_EDGES_CAP = MAX_PAIRS_CAP
"""Hard upper bound on the declared solved-edge budget."""

DEFAULT_MAX_PAIRS = DEFAULT_MAX_FRAMES * (DEFAULT_MAX_FRAMES - 1) // 2
"""Default pair budget: exactly the pairs of :data:`DEFAULT_MAX_FRAMES` frames."""

DEFAULT_MAX_TOTAL_SUPPORT = 4096
"""Default sum-of-pair-support budget for one entry."""

MAX_TOTAL_SUPPORT_CAP = 1_000_000
"""Hard upper bound on the declared total-support budget."""

DEFAULT_MAX_EDGES = DEFAULT_MAX_PAIRS
"""Default solved-edge budget: exactly the default pair budget."""

DEFAULT_MAX_CYCLE_CHECKS = 200_000
"""Default elementary cycle-check budget for the pose graph."""

MAX_CYCLE_CHECKS_CAP = 1_000_000
"""Hard upper bound on the declared cycle-check budget."""

DEFAULT_ROTATION_TOLERANCE_DEGREES = 2.0
"""Default largest accepted cycle rotation error, in degrees."""

DEFAULT_DIRECTION_TOLERANCE = 0.05
"""Default largest accepted cycle direction-hull distance."""


class ImageDerivedEntryError(ValueError):
    """A malformed call, an unusable input, or a delegated contract violation.

    ``reason`` is one of :data:`ENTRY_ERROR_REASONS`; ``detail`` is a human
    diagnostic that never replaces the typed reason. Data conditions a caller can
    receive as a typed terminal result are *not* raised: a refused connection
    graph or pose graph is carried on :class:`ImageDerivedEntryResult`.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class ImageDerivedEntryResult:
    """The typed terminal of one normal-mode raw-image entry.

    ``terminal`` is always :data:`GEOMETRY_UNAVAILABLE`. ``session`` is the
    accepted digest-bound session, ``graph`` and ``pose_graph`` are the accepted
    records built from it (each may itself be a typed refusal, carried unchanged).
    ``reason`` is ``""`` only when both the graph and the pose graph are ``ok``;
    otherwise it is the strongest typed refusal in
    :data:`ENTRY_TERMINAL_REASONS`. ``project_path`` names the schema-2 ``.mcscan``
    that carries the canonical session record, and ``source_paths`` are the
    resolved ordinary image paths the session was built from.

    Construction replays the whole cross-binding chain rather than trusting the
    carried objects. The session is rebuilt through
    :func:`image_derived_session_from_record`, every retained frame is then
    rebuilt through the accepted :class:`ImageDerivedFrame` constructor (so a
    writable, non-``uint8``, wrongly shaped or otherwise constructor-bypassed
    ``image`` cannot ride the pixel-free record digest), the graph and pose graph
    are rebuilt through their own accepted constructors, their canonical payload
    digests are recomputed, every nested budget/evidence/pose record is rebuilt
    through its own constructor, and the nested source bindings are re-checked.
    The graph must bind the session digest, the pose graph must bind both the
    graph and the session digests, ``connection_budgets`` must be exactly the
    carried graph's budgets, ``pose_budgets`` must be exactly the pose graph's
    budgets, ``source_paths`` must be exactly the session frame source labels in
    canonical frame order, and ``reason``/``detail`` must equal the exact
    terminal derived from the carried graph and pose graph. No stage claims
    metric scale, and ``entry_sha256`` replays from the canonical payload, which
    hashes the full carried record of every stage (not selected scalar
    summaries). No V1/V2/V3/M5 record is present, by construction.
    """

    schema: str
    producer: str
    terminal: str
    reason: str
    detail: str
    project_path: str
    source_paths: tuple[str, ...]
    connection_budgets: ConnectionBudgets
    pose_budgets: PoseGraphBudgets
    session: ImageDerivedSession
    graph: ImageDerivedConnectionGraph | None
    pose_graph: ImageDerivedPoseGraph | None
    entry_sha256: str | None = None
    project_lease: Any = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.project_lease is not None:
            from mc_photogrammetry.project.store import ProjectWriteLease
            if not isinstance(self.project_lease, ProjectWriteLease):
                raise ValueError("entry project lease must be typed")
            self.project_lease.validate_live(self.project_path)
        if self.schema != IMAGE_DERIVED_ENTRY_SCHEMA:
            raise ValueError("unsupported image-derived entry schema")
        if self.producer != IMAGE_DERIVED_ENTRY_PRODUCER:
            raise ValueError("unsupported image-derived entry producer")
        if self.terminal not in (GEOMETRY_UNAVAILABLE, "camera_unavailable"):
            raise ValueError("an image-derived entry can only terminate geometry_unavailable")
        if not isinstance(self.reason, str):
            raise ValueError("image-derived entry reason must be a string")
        if not isinstance(self.detail, str):
            raise ValueError("image-derived entry detail must be a string")
        if not isinstance(self.project_path, str) or not self.project_path:
            raise ValueError("image-derived entry project path must be a non-empty string")
        if (not isinstance(self.source_paths, tuple) or not self.source_paths
                or any(not isinstance(path, str) or not path for path in self.source_paths)):
            raise ValueError("image-derived entry source paths must be non-empty strings")
        if not isinstance(self.connection_budgets, ConnectionBudgets):
            raise ValueError("image-derived entry needs accepted connection budgets")
        if not isinstance(self.pose_budgets, PoseGraphBudgets):
            raise ValueError("image-derived entry needs accepted pose-graph budgets")
        if not isinstance(self.session, ImageDerivedSession):
            raise ValueError("image-derived entry needs an accepted session")
        if not self.session.frames:
            _replay_session(self.session)
            _replay_connection_budget(self.connection_budgets)
            _replay_pose_budget(self.pose_budgets)
            if (self.graph is not None or self.pose_graph is not None
                    or self.terminal != "camera_unavailable"
                    or (self.reason, self.detail) != _camera_unavailable_terminal()
                    or self.source_paths != tuple(frame.source_label for frame in self.session.input_frames)):
                raise ValueError("zero-camera terminal cannot carry downstream stages or altered intake")
            expected = _digest(_entry_payload(self))
            if self.entry_sha256 is not None and self.entry_sha256 != expected:
                raise ValueError("image-derived entry digest mismatch")
            object.__setattr__(self, "entry_sha256", expected)
            return
        if self.terminal != GEOMETRY_UNAVAILABLE:
            raise ValueError("camera_unavailable requires zero admitted cameras")
        if not isinstance(self.graph, ImageDerivedConnectionGraph):
            raise ValueError("image-derived entry needs an accepted connection graph")
        if not isinstance(self.pose_graph, ImageDerivedPoseGraph):
            raise ValueError("image-derived entry needs an accepted pose graph")

        # Deep replay: every carried record is rebuilt through its own accepted
        # validator and canonical payload digest, so a resealed enclosing digest
        # cannot hide an invariant violation, a stale nested record or a stale
        # source binding.
        _replay_session(self.session)
        _replay_connection_graph(self.graph, self.session)
        _replay_pose_graph(self.pose_graph, self.graph, self.session)

        if (self.graph.session_sha256 != self.session.session_sha256
                or self.pose_graph.session_sha256 != self.session.session_sha256):
            raise ValueError("image-derived entry records do not share the session digest")
        if self.pose_graph.graph_sha256 != self.graph.graph_sha256:
            raise ValueError("image-derived entry pose graph does not bind its graph")
        if self.connection_budgets != self.graph.budgets:
            raise ValueError(
                "image-derived entry connection budgets are not the graph budgets")
        if self.pose_budgets != self.pose_graph.budgets:
            raise ValueError(
                "image-derived entry pose budgets are not the pose-graph budgets")

        labels = tuple(frame.source_label for frame in self.session.input_frames)
        if (any(not isinstance(label, str) or not label for label in labels)
                or self.source_paths != labels):
            raise ValueError(
                "image-derived entry source paths are not the session frame labels")

        if (self.session.scale_known is not False
                or self.graph.scale_known is not False
                or self.pose_graph.metric_scale_known is not False):
            raise ValueError("an image-derived entry cannot claim metric scale")

        expected_reason, expected_detail = _terminal_reason(self.graph, self.pose_graph)
        if (self.reason, self.detail) != (expected_reason, expected_detail):
            raise ValueError(
                "image-derived entry reason/detail is not the derived terminal state")

        if self.entry_sha256 is None:
            object.__setattr__(self, "entry_sha256", _digest(_entry_payload(self)))
        elif (not _is_sha256(self.entry_sha256)
              or self.entry_sha256 != _digest(_entry_payload(self))):
            raise ValueError("image-derived entry digest mismatch")

    @property
    def session_record(self) -> dict[str, Any]:
        """The canonical, digest-bound schema-2 session record the store persists."""
        return session_mod.image_derived_session_to_record(self.session)

    def to_record(self) -> dict[str, Any]:
        """Canonical terminal record; carries accepted records, never geometry."""
        record = _entry_payload(self)
        record["entry_sha256"] = self.entry_sha256
        return record


def build_image_derived_entry(
    image_inputs: Iterable[str | Path], *, project_path: str | Path,
    matcher: Callable[..., JointPairResult] | None = None,
    recursive: bool = False, max_frames: int = DEFAULT_MAX_FRAMES,
    max_pairs: int = DEFAULT_MAX_PAIRS, max_total_support: int = DEFAULT_MAX_TOTAL_SUPPORT,
    max_edges: int = DEFAULT_MAX_EDGES, max_cycle_checks: int = DEFAULT_MAX_CYCLE_CHECKS,
    rotation_tolerance_degrees: float = DEFAULT_ROTATION_TOLERANCE_DEGREES,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
    capture_ingress: bool = False, project_max_bytes: int | None = None,
    retain_project_lease: bool = False,
) -> ImageDerivedEntryResult:
    """Public image-only entry; its accepted API has no diagnostic controls."""
    return _build_image_derived_entry(
        image_inputs, project_path=project_path, matcher=matcher, recursive=recursive,
        max_frames=max_frames, max_pairs=max_pairs, max_total_support=max_total_support,
        max_edges=max_edges, max_cycle_checks=max_cycle_checks,
        rotation_tolerance_degrees=rotation_tolerance_degrees,
        direction_tolerance=direction_tolerance, capture_ingress=capture_ingress,
        project_max_bytes=project_max_bytes, retain_project_lease=retain_project_lease)


def _build_image_derived_entry(
    image_inputs: Iterable[str | Path],
    *,
    project_path: str | Path,
    matcher: Callable[..., JointPairResult] | None = None,
    recursive: bool = False,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_total_support: int = DEFAULT_MAX_TOTAL_SUPPORT,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_cycle_checks: int = DEFAULT_MAX_CYCLE_CHECKS,
    rotation_tolerance_degrees: float = DEFAULT_ROTATION_TOLERANCE_DEGREES,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
    capture_ingress: bool = False,
    project_max_bytes: int | None = None,
    retain_project_lease: bool = False,
    _correspondence_observer=None,
) -> ImageDerivedEntryResult:
    """Load ordinary images, compose the accepted image-derived chain, persist it.

    ``image_inputs`` is an ordered collection of still-image file or directory
    paths. Directories are expanded with :func:`input.media.discover_media` (pass
    ``recursive=True`` for a deep walk); exact duplicate bytes are de-duplicated
    by that accepted discovery. Every discovered video source is refused - this
    seam is still images only.

    The session, connection graph and pose graph are built by the accepted
    adapters with the explicitly declared caps, and the accepted session record is
    persisted as a schema-2 ``.mcscan`` through the existing project store and
    reloaded to verify the digest binds. The returned
    :class:`ImageDerivedEntryResult` always terminates at
    :data:`GEOMETRY_UNAVAILABLE`; a refused graph or pose graph is carried on it
    unchanged rather than fabricated into geometry. ``matcher`` defaults to the
    accepted production matcher and exists only as the injectable test seam the
    connection adapter itself exposes.
    """
    caps = _validated_caps(
        max_frames=max_frames,
        max_pairs=max_pairs,
        max_total_support=max_total_support,
        max_edges=max_edges,
        max_cycle_checks=max_cycle_checks,
        rotation_tolerance_degrees=rotation_tolerance_degrees,
        direction_tolerance=direction_tolerance,
    )
    if type(retain_project_lease) is not bool or (retain_project_lease and (
            project_max_bytes is None or os.name != "nt")):
        raise ImageDerivedEntryError(ENTRY_BUDGET_INVALID, "retained project lease needs capped Windows capture")
    if project_max_bytes is not None and (
            type(project_max_bytes) is not int or project_max_bytes < 1):
        raise ImageDerivedEntryError(ENTRY_PERSIST_FAILED, "invalid project byte limit")
    paths = _normalized_paths(image_inputs)
    # DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-IMPL-R4 (A): one encoded-byte read per
    # input screenshot. Discovery reads each retained candidate's bytes exactly
    # once, hashes and de-duplicates them, then synchronously hands that same
    # buffer to this sink, which decodes it before discovery moves to the next
    # path. The buffer reference is released before the next file, so no
    # aggregate byte cache is retained (the lead's 17:42/17:45 memory bound);
    # no second ``_sha256(source_path)`` or ``Image.open(source_path)`` occurs.
    # ``capture_ingress`` only controls whether the typed ingress pin is attached
    # to the session; the default session/result digests stay byte-for-byte
    # unchanged.
    frames: dict[str, Any] = {}
    labels: dict[str, str] = {}
    ingress_records: dict[str, Any] = {}

    def _ingress_sink(resolved_path: str, frame: Any, data: bytes) -> None:
        try:
            image = load_image_bytes(data, suffix=Path(resolved_path).suffix)
        except ImageDecodeError as error:
            raise ImageDerivedEntryError(
                ENTRY_IMAGE_DECODE_FAILED, f"{frame.frame_id}: {error}") from error
        frames[frame.frame_id] = image
        labels[frame.frame_id] = resolved_path
        if capture_ingress:
            try:
                pin = build_frame_ingress_provenance(
                    frame_id=frame.frame_id, source_label=resolved_path,
                    encoded_bytes=data, decoded_pixels=image,
                    transform=FrameTransform(
                        source_size=(int(image.shape[1]), int(image.shape[0]))),
                    decoded_pixel_sha256=session_mod.image_sha256(image))
            except IngressProvenanceError as error:
                raise ImageDerivedEntryError(
                    ENTRY_SESSION_FAILED,
                    f"{error.reason}: {error.field}") from error
            ingress_records[frame.frame_id] = pin

    manifest = _discover(paths, recursive=recursive, ingress_sink=_ingress_sink)
    if manifest.video_sources:
        raise ImageDerivedEntryError(
            ENTRY_VIDEO_UNSUPPORTED,
            f"{len(manifest.video_sources)} video source(s) were supplied; "
            "the normal image-derived entry accepts ordinary still images only")
    image_frames = manifest.image_frames
    if not image_frames:
        raise ImageDerivedEntryError(
            ENTRY_NO_IMAGES, "no ordinary still image was discovered in the given inputs")
    if len(image_frames) > caps["max_frames"]:
        raise ImageDerivedEntryError(
            ENTRY_FRAME_LIMIT,
            f"discovered {len(image_frames)} images, above max_frames "
            f"{caps['max_frames']}")
    for frame in image_frames:
        if frame.frame_id not in frames:
            raise ImageDerivedEntryError(
                ENTRY_IMAGE_DECODE_FAILED,
                f"{frame.frame_id}: the single-read ingress buffer is unavailable")

    try:
        session = build_image_derived_session(
            frames, source_paths=labels,
            ingress=(ingress_records if capture_ingress else None))
    except ImageDerivedSessionError as error:
        refusal = ImageDerivedEntryError(ENTRY_SESSION_FAILED, f"{error.reason}: {error.detail}")
        refusal.review_evidence = error.review_evidence
        raise refusal from error

    try:
        connection_budgets = make_connection_budgets(
            max_pairs=caps["max_pairs"],
            max_total_support=caps["max_total_support"])
        pose_budgets = make_pose_graph_budgets(
            max_edges=caps["max_edges"],
            max_cycle_checks=caps["max_cycle_checks"],
            rotation_tolerance_degrees=caps["rotation_tolerance_degrees"],
            direction_tolerance=caps["direction_tolerance"])
    except ValueError as error:
        raise ImageDerivedEntryError(ENTRY_BUDGET_INVALID, str(error)) from error

    graph = pose_graph = None
    if session.frames:
        resolved_matcher = match_and_solve_pair if matcher is None else matcher
        try:
            diagnostic_kwargs = ({} if _correspondence_observer is None else
                                 {"_correspondence_observer": _correspondence_observer})
            graph = connections_mod.build_image_derived_connections(
                session, budgets=connection_budgets, matcher=resolved_matcher,
                **diagnostic_kwargs)
        except ImageDerivedConnectionError as error:
            raise ImageDerivedEntryError(
                ENTRY_CONNECTION_FAILED, f"{error.reason}: {error.detail}") from error

        try:
            pose_graph = pose_graph_mod.compose_image_derived_pose_graph(
                graph, budgets=pose_budgets)
        except ImageDerivedPoseGraphError as error:
            raise ImageDerivedEntryError(
                ENTRY_POSE_GRAPH_FAILED, f"{error.reason}: {error.detail}") from error

    destination = Path(project_path)
    reason, detail = _terminal_reason(graph, pose_graph)
    project_lease = _persist_session(
        destination=destination,
        session=session,
        source_paths=tuple(frame.source_path for frame in image_frames),
        name=_project_name(destination, image_frames),
        project_max_bytes=project_max_bytes,
        retain_project_lease=retain_project_lease,
    )

    try:
        return ImageDerivedEntryResult(
            schema=IMAGE_DERIVED_ENTRY_SCHEMA,
            producer=IMAGE_DERIVED_ENTRY_PRODUCER,
            terminal=GEOMETRY_UNAVAILABLE if session.frames else "camera_unavailable",
            reason=reason,
            detail=detail,
            project_path=str(destination),
            source_paths=tuple(frame.source_path for frame in image_frames),
            connection_budgets=connection_budgets,
            pose_budgets=pose_budgets,
            session=session,
            graph=graph,
            pose_graph=pose_graph,
            project_lease=project_lease,
        )
    except BaseException:
        if project_lease is not None:
            project_lease.discard()
        raise


# --- Input handling -----------------------------------------------------------

def _normalized_paths(image_inputs: Iterable[str | Path]) -> list[Path]:
    """Validate that the entry received an ordered collection of path objects."""
    if image_inputs is None or isinstance(image_inputs, (str, bytes, Path)):
        raise ImageDerivedEntryError(
            ENTRY_INPUT_INVALID,
            "image_inputs must be an ordered collection of file or directory paths")
    try:
        items = list(image_inputs)
    except TypeError as error:
        raise ImageDerivedEntryError(
            ENTRY_INPUT_INVALID,
            "image_inputs must be an ordered collection of file or directory paths") from error
    if not items:
        raise ImageDerivedEntryError(
            ENTRY_INPUT_INVALID, "image_inputs must name at least one image or directory")
    paths: list[Path] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (str, Path)):
            raise ImageDerivedEntryError(
                ENTRY_INPUT_INVALID, "every image input must be a string or Path")
        paths.append(Path(item))
    return paths


def _discover(paths: list[Path], *, recursive: bool, ingress_sink=None):
    try:
        return discover_media(paths, recursive=recursive, ingress_sink=ingress_sink)
    except ImageDerivedEntryError:
        # The synchronous single-read ingress sink raises a typed entry error
        # (decode/session failure) naming the frame; never re-wrap it as an
        # input-discovery failure.
        raise
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise ImageDerivedEntryError(
            ENTRY_INPUT_INVALID, f"image discovery failed: {error}") from error


def _validated_caps(
    *,
    max_frames: Any,
    max_pairs: Any,
    max_total_support: Any,
    max_edges: Any,
    max_cycle_checks: Any,
    rotation_tolerance_degrees: Any,
    direction_tolerance: Any,
) -> dict[str, Any]:
    """Bound every declared cap; a caller may lower a default, never exceed it."""
    return {
        "max_frames": _bounded_int(
            max_frames, name="max_frames", minimum=1, cap=MAX_FRAMES_CAP),
        "max_pairs": _bounded_int(
            max_pairs, name="max_pairs", minimum=0, cap=MAX_PAIRS_CAP),
        "max_total_support": _bounded_int(
            max_total_support, name="max_total_support", minimum=0,
            cap=MAX_TOTAL_SUPPORT_CAP),
        "max_edges": _bounded_int(
            max_edges, name="max_edges", minimum=0, cap=MAX_EDGES_CAP),
        "max_cycle_checks": _bounded_int(
            max_cycle_checks, name="max_cycle_checks", minimum=0,
            cap=MAX_CYCLE_CHECKS_CAP),
        "rotation_tolerance_degrees": _bounded_float(
            rotation_tolerance_degrees, name="rotation_tolerance_degrees",
            minimum=0.0, maximum=180.0),
        "direction_tolerance": _bounded_float(
            direction_tolerance, name="direction_tolerance",
            minimum=0.0, maximum=1.0),
    }


def _bounded_int(value: Any, *, name: str, minimum: int, cap: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImageDerivedEntryError(
            ENTRY_BUDGET_INVALID, f"{name} must be an integer")
    if value < minimum or value > cap:
        raise ImageDerivedEntryError(
            ENTRY_BUDGET_INVALID,
            f"{name} must be between {minimum} and {cap}, got {value}")
    return int(value)


def _bounded_float(
    value: Any, *, name: str, minimum: float, maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImageDerivedEntryError(
            ENTRY_BUDGET_INVALID, f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ImageDerivedEntryError(
            ENTRY_BUDGET_INVALID, f"{name} must be a finite number")
    if result < minimum or result > maximum:
        raise ImageDerivedEntryError(
            ENTRY_BUDGET_INVALID,
            f"{name} must be between {minimum} and {maximum}, got {result}")
    return result


def _project_name(destination: Path, image_frames: list[Any]) -> str:
    stem = destination.stem
    if stem:
        return stem
    return Path(image_frames[0].source_path).stem


# --- Persistence --------------------------------------------------------------

def _persist_session(
    *,
    destination: Path,
    session: ImageDerivedSession,
    source_paths: tuple[str, ...],
    name: str,
    project_max_bytes: int | None = None,
    retain_project_lease: bool = False,
):
    """Persist the schema-2 session record through the existing store, then replay it."""
    project = ScanProject(
        name=name,
        source_paths=list(source_paths),
        schema_version=SCAN_PROJECT_SCHEMA_VERSION,
        image_derived_session=session_mod.image_derived_session_to_record(session),
    )
    lease = None
    try:
        if project_max_bytes is None:
            save_project(project, destination)
        else:
            if retain_project_lease:
                lease = save_project(project, destination, max_bytes=project_max_bytes, retain_lease=True)
                from mc_photogrammetry.project.store import ProjectWriteLease
                if not isinstance(lease, ProjectWriteLease) or lease.max_bytes != project_max_bytes:
                    raise ProjectStoreError("retained project lease missing or cap mismatched")
                lease.validate_live(destination)
            else:
                save_project(project, destination, max_bytes=project_max_bytes)
    except ProjectStoreError as error:
        if lease is not None:
            lease.close()
        raise ImageDerivedEntryError(
            ENTRY_PERSIST_FAILED, f"could not save the session project: {error}") from error
    try:
        reloaded = load_project(destination) if lease is None else load_project(destination, snapshot_bytes=lease.read_bytes())
    except ProjectStoreError as error:
        if lease is not None:
            lease.discard()
        raise ImageDerivedEntryError(
            ENTRY_PERSIST_FAILED, f"could not reload the saved session project: {error}") from error
    record = reloaded.image_derived_session
    if (not isinstance(record, dict)
            or record.get("session_sha256") != session.session_sha256):
        if lease is not None:
            lease.discard()
        raise ImageDerivedEntryError(
            ENTRY_PERSIST_FAILED,
            "the saved session record does not replay the built session digest")
    return lease


# --- Record replay ------------------------------------------------------------

_REPLAY_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError)


def _declared_fields(record: Any) -> dict[str, Any]:
    """Every declared dataclass field of ``record`` as a plain mapping."""
    return {field.name: getattr(record, field.name)
            for field in dataclasses.fields(record)}


def _replay_session(session: ImageDerivedSession) -> None:
    """Rebuild the carried session and every retained frame through its validators.

    The canonical record round-trip is deliberately pixel-free - a record carries
    only frame identity, shape and digest - so on its own it cannot see that a
    retained ``image`` was replaced with a writable, reopenable, non-``uint8``,
    wrongly shaped or otherwise constructor-bypassed array of identical bytes.
    Each original frame is therefore rebuilt through the accepted
    :class:`ImageDerivedFrame` constructor, which re-runs the immutable-``uint8``
    image checks, the exact-shape check, the digest-format check and the pixel
    digest, and its declared identity/shape/digest are compared to the frame the
    canonical record rebuilt. A produced entry session must carry pixels: a
    deserialized (pixel-free) session is not an accepted entry session.
    """
    if not isinstance(session, ImageDerivedSession):
        raise ValueError("image-derived entry needs an accepted session")
    try:
        record = session_mod.image_derived_session_to_record(session)
        rebuilt = session_mod.image_derived_session_from_record(record)
    except _REPLAY_ERRORS as error:
        raise ValueError(
            "image-derived entry session does not replay its canonical record") from error
    if rebuilt.session_sha256 != session.session_sha256:
        raise ValueError("image-derived entry session digest mismatch")
    try:
        frames = tuple(session.input_frames)
        frame_ids = tuple(frame.frame_id for frame in frames)
        active_frames = tuple(session.frames)
        active_ids = tuple(frame.frame_id for frame in active_frames)
    except _REPLAY_ERRORS as error:
        raise ValueError(
            "image-derived entry session frames do not replay") from error
    if tuple(session.cameras) != active_ids or tuple(session.local_cameras) != active_ids:
        raise ValueError("image-derived entry session camera maps are not canonical")
    if (len(rebuilt.input_frames) != len(frames)
            or tuple(frame.frame_id for frame in rebuilt.input_frames) != frame_ids):
        raise ValueError("image-derived entry session frames do not replay")
    for frame, record_frame in zip(frames, rebuilt.input_frames):
        _replay_session_frame(frame, record_frame)
    if active_ids != tuple(frame.frame_id for frame in rebuilt.frames):
        raise ValueError("image-derived entry session active frames do not replay")
    inventory = {frame.frame_id: frame for frame in frames}
    for frame, record_frame in zip(active_frames, rebuilt.frames):
        _replay_session_frame(frame, record_frame)
        if frame is not inventory[frame.frame_id]:
            raise ValueError("image-derived entry session frame does not replay its input identity")


def _replay_session_frame(
    frame: ImageDerivedFrame, record_frame: ImageDerivedFrame,
) -> None:
    """Rebuild one retained frame through the accepted constructor and its record.

    This is the only place a retained frame's actual array is re-validated: the
    constructor refuses a writable/reopenable buffer, a non-``uint8`` dtype, an
    invalid or mismatched shape and an unmatching pixel digest, and it refuses a
    pixel-free frame outright. The frame's declared identity, shape and digest
    must then equal the frame the canonical record rebuilt, so a retained frame
    cannot drift from the record it is bound to.
    """
    if (not isinstance(frame, ImageDerivedFrame)
            or not isinstance(record_frame, ImageDerivedFrame)):
        raise ValueError("image-derived entry session frame does not replay")
    if frame.image is None:
        raise ValueError("image-derived entry session frame has no retained pixels")
    try:
        replayed = ImageDerivedFrame(
            frame_id=frame.frame_id,
            source_label=frame.source_label,
            image=frame.image,
            image_shape=frame.image_shape,
            image_sha256=frame.image_sha256,
            # Validate the carried pin as well as the retained pixels; a
            # canonical pin must not hide a divergent active-frame identity.
            ingress=frame.ingress,
        )
    except _REPLAY_ERRORS as error:
        raise ValueError(
            "image-derived entry session frame does not replay") from error
    if (replayed.frame_id != record_frame.frame_id
            or replayed.source_label != record_frame.source_label
            or session_mod._frame_payload(replayed)
            != session_mod._frame_payload(record_frame)):
        raise ValueError(
            "image-derived entry session frame does not replay its record")
    try:
        digest = session_mod.image_sha256(replayed.image)
    except _REPLAY_ERRORS as error:
        raise ValueError(
            "image-derived entry session frame pixels do not replay") from error
    if digest != record_frame.image_sha256:
        raise ValueError("image-derived entry session frame pixels do not replay")


def _replay_connection_budget(budget: ConnectionBudgets) -> None:
    """Re-run the accepted ``ConnectionBudgets`` constructor over a carried budget."""
    if not isinstance(budget, ConnectionBudgets):
        raise ValueError("image-derived entry needs accepted connection budgets")
    try:
        rebuilt = ConnectionBudgets(
            max_pairs=budget.max_pairs,
            max_total_support=budget.max_total_support)
    except _REPLAY_ERRORS as error:
        raise ValueError(
            "image-derived entry connection budget does not replay") from error
    if rebuilt.to_payload() != budget.to_payload():
        raise ValueError("image-derived entry connection budget does not replay")


def _replay_connection_hypothesis(hypothesis: Any) -> YawClassDiagnostic:
    """Rebuild one nested yaw diagnostic through its accepted constructor."""
    if not isinstance(hypothesis, YawClassDiagnostic):
        raise ValueError("image-derived entry graph evidence does not replay")
    try:
        rebuilt = YawClassDiagnostic(**_declared_fields(hypothesis))
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry graph evidence does not replay") from error
    if (connections_mod._hypothesis_payload(rebuilt)
            != connections_mod._hypothesis_payload(hypothesis)):
        raise ValueError("image-derived entry graph evidence does not replay")
    return rebuilt


def _replay_connection_evidence(
        records: tuple[RelativeCameraEvidence, ...],
        session: ImageDerivedSession,
) -> None:
    """Rebuild every nested evidence record and its session binding."""
    if not isinstance(records, tuple):
        raise ValueError("image-derived entry graph evidence does not replay")
    for record in records:
        if not isinstance(record, RelativeCameraEvidence):
            raise ValueError("image-derived entry graph evidence does not replay")
        if not isinstance(record.hypotheses, tuple):
            raise ValueError("image-derived entry graph evidence does not replay")
        rebuilt_hypotheses = tuple(
            _replay_connection_hypothesis(hypothesis)
            for hypothesis in record.hypotheses)
        try:
            rebuilt = RelativeCameraEvidence(
                **{**_declared_fields(record), "hypotheses": rebuilt_hypotheses})
        except _REPLAY_ERRORS as error:
            raise ValueError(
                "image-derived entry graph evidence does not replay") from error
        if rebuilt.to_payload() != record.to_payload():
            raise ValueError("image-derived entry graph evidence does not replay")
        if record.session_sha256 != session.session_sha256:
            raise ValueError(
                "image-derived entry graph evidence does not bind the session")


def _replay_connection_graph(
        graph: ImageDerivedConnectionGraph, session: ImageDerivedSession,
) -> None:
    """Recompute the graph digest, rebuild it and replay every nested record."""
    if not isinstance(graph, ImageDerivedConnectionGraph):
        raise ValueError("image-derived entry needs an accepted connection graph")
    try:
        payload = connections_mod._graph_payload(graph)
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry graph does not replay") from error
    if graph.graph_sha256 != connections_mod._digest(payload):
        raise ValueError("image-derived entry graph digest mismatch")
    try:
        connections_mod.ImageDerivedConnectionGraph(**_declared_fields(graph))
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry graph does not replay") from error
    _replay_connection_budget(graph.budgets)
    _replay_connection_evidence(graph.evidence, session)
    if graph.session_sha256 != session.session_sha256:
        raise ValueError("image-derived entry graph does not bind the session")
    frame_ids = tuple(frame.frame_id for frame in session.frames)
    if tuple(graph.frame_ids) != frame_ids:
        raise ValueError("image-derived entry graph frames do not match the session")
    if graph.status == "ok":
        if (graph.pair_count > graph.budgets.max_pairs
                or sum(record.support for record in graph.evidence)
                > graph.budgets.max_total_support):
            raise ValueError(
                "image-derived entry graph exceeds its declared budget")


def _replay_pose_budget(budget: PoseGraphBudgets) -> None:
    """Re-run the accepted ``PoseGraphBudgets`` constructor over a carried budget."""
    if not isinstance(budget, PoseGraphBudgets):
        raise ValueError("image-derived entry needs accepted pose-graph budgets")
    try:
        rebuilt = PoseGraphBudgets(
            max_edges=budget.max_edges,
            max_cycle_checks=budget.max_cycle_checks,
            rotation_tolerance_degrees=budget.rotation_tolerance_degrees,
            direction_tolerance=budget.direction_tolerance)
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry pose budget does not replay") from error
    if rebuilt.to_payload() != budget.to_payload():
        raise ValueError("image-derived entry pose budget does not replay")


def _replay_pose_records(
        poses: tuple[GaugeCameraPose, ...],
        graph: ImageDerivedConnectionGraph,
) -> None:
    """Rebuild every nested gauge pose and its source-graph binding."""
    if not isinstance(poses, tuple):
        raise ValueError("image-derived entry pose-graph poses do not replay")
    for pose in poses:
        if not isinstance(pose, GaugeCameraPose):
            raise ValueError("image-derived entry pose-graph poses do not replay")
        try:
            rebuilt = GaugeCameraPose(**_declared_fields(pose))
        except _REPLAY_ERRORS as error:
            raise ValueError(
                "image-derived entry pose-graph poses do not replay") from error
        if rebuilt.to_payload() != pose.to_payload():
            raise ValueError("image-derived entry pose-graph poses do not replay")
        if pose.graph_sha256 != graph.graph_sha256:
            raise ValueError(
                "image-derived entry pose-graph poses do not bind the graph")


def _replay_pose_graph(
        pose_graph: ImageDerivedPoseGraph,
        graph: ImageDerivedConnectionGraph,
        session: ImageDerivedSession,
) -> None:
    """Recompute the pose-graph digest, rebuild it and replay its nested records."""
    if not isinstance(pose_graph, ImageDerivedPoseGraph):
        raise ValueError("image-derived entry needs an accepted pose graph")
    try:
        payload = pose_graph_mod._pose_graph_payload(pose_graph)
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry pose graph does not replay") from error
    if pose_graph.pose_graph_sha256 != pose_graph_mod._digest(payload):
        raise ValueError("image-derived entry pose graph digest mismatch")
    try:
        pose_graph_mod.ImageDerivedPoseGraph(**_declared_fields(pose_graph))
    except _REPLAY_ERRORS as error:
        raise ValueError("image-derived entry pose graph does not replay") from error
    _replay_pose_budget(pose_graph.budgets)
    _replay_pose_records(pose_graph.poses, graph)
    if pose_graph.graph_sha256 != graph.graph_sha256:
        raise ValueError("image-derived entry pose graph does not bind its graph")
    if pose_graph.session_sha256 != session.session_sha256:
        raise ValueError("image-derived entry pose graph does not bind the session")
    if tuple(pose_graph.frame_ids) != tuple(graph.frame_ids):
        raise ValueError(
            "image-derived entry pose-graph frames do not match the graph")
    if pose_graph.status == "ok":
        if graph.status != "ok":
            raise ValueError(
                "image-derived entry pose graph cannot be composed from a refused graph")
        if tuple(pose_graph.components) != tuple(graph.components):
            raise ValueError(
                "image-derived entry pose-graph components do not match the graph")
        solved_edges = sum(1 for record in graph.evidence if record.status == "solved")
        fundamental_cycles = solved_edges - (
            len(graph.frame_ids) - len(graph.components))
        if (solved_edges > pose_graph.budgets.max_edges
                or fundamental_cycles > pose_graph.budgets.max_cycle_checks):
            raise ValueError(
                "image-derived entry pose graph exceeds its declared budget")


# --- Terminal and digest ------------------------------------------------------

def _terminal_reason(
    graph: ImageDerivedConnectionGraph, pose_graph: ImageDerivedPoseGraph,
) -> tuple[str, str]:
    if graph is None and pose_graph is None:
        return _camera_unavailable_terminal()
    if graph.status != "ok":
        return graph.reason, graph.detail
    if pose_graph.status != "ok":
        return pose_graph.reason, pose_graph.detail
    if len(graph.frame_ids) == 1:
        return "single_camera_local_review", "One admitted camera retains local calibration only; no relative translation or metric geometry."
    return "", (
        "image-derived session and relative camera evidence are established; "
        "V1/V2/V3/M5 geometry is unavailable from images alone in normal mode")


def _camera_unavailable_terminal() -> tuple[str, str]:
    return "camera_unavailable", "No input camera admitted; full intake and camera exclusions retained."


def _entry_payload(result: ImageDerivedEntryResult) -> dict[str, Any]:
    """Canonical entry payload: full carried records plus every bound field.

    ``entry_sha256`` binds each stage's own canonical record (the session record,
    the graph record and the pose-graph record), not a selected scalar summary, so
    a resealed or swapped nested record changes the entry digest even when its
    scalar digest field is copied.
    """
    return {
        "schema": result.schema,
        "producer": result.producer,
        "terminal": result.terminal,
        "reason": result.reason,
        "detail": result.detail,
        "project_path": result.project_path,
        "source_paths": list(result.source_paths),
        "connection_budgets": result.connection_budgets.to_payload(),
        "pose_budgets": result.pose_budgets.to_payload(),
        "session_sha256": result.session.session_sha256,
        "graph_sha256": result.graph.graph_sha256 if result.graph is not None else None,
        "pose_graph_sha256": result.pose_graph.pose_graph_sha256 if result.pose_graph is not None else None,
        "graph_status": result.graph.status if result.graph is not None else "not_run",
        "pose_graph_status": result.pose_graph.status if result.pose_graph is not None else "not_run",
        "unresolved": list(result.session.unresolved),
        "scale_known": bool(result.session.scale_known),
        "session_record": session_mod.image_derived_session_to_record(result.session),
        "graph_record": result.graph.to_record() if result.graph is not None else None,
        "pose_graph_record": result.pose_graph.to_record() if result.pose_graph is not None else None,
    }


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
