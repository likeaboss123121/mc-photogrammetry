"""Find the block grid on a plane, once the camera's orientation is known.

`calibration` recovers where the camera is *pointing* and how wide its lens is.
It cannot say where the camera stands, because vanishing points are unchanged by
moving along a ray: a wall twice as far away with blocks twice as big looks
identical. Distance has to come from something with a known size, and in
Minecraft everything has one - a block is a metre.

So this measures the block grid directly. Given an orientation and a candidate
plane normal, there is a homography that maps the image to a fronto-parallel view
of any plane with that normal; only the plane's distance is unknown, and distance
acts on the rectified image as a pure scale. Seams that are one block apart in
the world are therefore one *period* apart in the rectified image, and reading
that period off reads the distance off with it.

The catch, and the reason this returns hypotheses rather than an answer: a
Minecraft texture is 16x16 texels, so a strongly periodic signal at spacing `p`
is either block seams a block apart or texel edges a sixteenth of a block apart.
Those differ in distance by a factor of sixteen and both are entirely consistent
with one image. `PeriodEstimate` keeps both, ranked, and resolving between them
needs a second view, a user-supplied length, or a recognisable feature.

What this does not attempt is deciding *which* planes are there. Given a region
and a normal it measures the grid on it; finding the regions is surface fitting,
which is M4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np


# Internal evaluation-only sidecar. No public camera/coordinate input or legacy
# witness-sink behavior is changed by this scope.
from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses as _capture_dataclasses
import threading as _capture_threading

_PERIOD_SUPPORT_CONTEXT = ContextVar("_plane_period_support_context", default=None)
_PERIOD_SUPPORT_OWNER = _capture_threading.Lock()
_PERIOD_SUPPORT_ARRAY_CAP = 32 * 1024**2
_PERIOD_SUPPORT_MEMORY_CAP = 256 * 1024**2


class _PeriodSupportMemory:
    """Monotone allocation ledger: freed objects remain charged for the run.

    This bounds outstanding capture memory even when a callback retains every
    snapshot. Reserving before allocation can refuse earlier than live-memory
    accounting, but cannot silently exceed the diagnostic cap.
    """
    def __init__(self, limit=None):
        self.limit = _PERIOD_SUPPORT_MEMORY_CAP if limit is None else limit
        self.used = self.peak = 0

    def reserve(self, amount, detail):
        if type(amount) is not int or amount < 0 or self.used + amount > self.limit:
            raise _PeriodSupportIncomplete("cap_exhausted", "aggregate_memory:" + detail)
        self.used += amount
        self.peak = max(self.peak, self.used)


def _period_support_size_plan(value, depth=0, _nodes=None):
    """Pre-copy upper bound, including scalar payloads and container overhead."""
    if _nodes is None:
        _nodes = [0]
    _nodes[0] += 1
    if depth > 24 or _nodes[0] > 100000:
        raise _PeriodSupportIncomplete("cap_exhausted", "snapshot_depth")
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 1024 + 64 * value.ndim
    if _capture_dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = _capture_dataclasses.fields(value)
        return 2048 + sum(512 + _period_support_size_plan(getattr(value, f.name), depth+1, _nodes)
                          for f in items)
    if isinstance(value, dict):
        if len(value) > 100000:
            raise _PeriodSupportIncomplete("cap_exhausted", "snapshot_nodes")
        return 1024 + sum(512 + _period_support_size_plan(k, depth+1, _nodes)
                          + _period_support_size_plan(v, depth+1, _nodes) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        if len(value) > 100000:
            raise _PeriodSupportIncomplete("cap_exhausted", "snapshot_nodes")
        return 1024 + sum(128 + _period_support_size_plan(v, depth+1, _nodes) for v in value)
    if isinstance(value, (str, bytes)):
        return 256 + (4 * len(value) if isinstance(value, str) else len(value))
    if isinstance(value, (int, np.integer)):
        return 256 + (int(value).bit_length()+7)//8
    return 256


def _period_support_reserve(amount, detail):
    state = _period_support_state()
    if state is not None:
        state.memory.reserve(amount, detail)


class _PeriodSupportIncomplete(RuntimeError):
    def __init__(self, reason, detail=""):
        self.reason, self.detail = reason, str(detail)[:256]
        super().__init__(self.detail or reason)


class _PeriodSupportState:
    def __init__(self, observer, run_id, signature_targets, match_targets):
        self.observer, self.run_id = observer, run_id
        self.thread = _capture_threading.get_ident()
        self.signature_targets = frozenset(signature_targets)
        self.match_targets = frozenset(match_targets)
        if (not isinstance(run_id, str) or not run_id
                or len(self.signature_targets) > 4 or len(self.match_targets) > 3):
            raise _PeriodSupportIncomplete("identity_drift", "scope identity/cap")
        self.context = {}
        self.incomplete = None
        self.signature_seen, self.match_seen = set(), set()
        self.rectifications = []
        self.warp = None
        self.memory = _PeriodSupportMemory()

    def fail(self, reason, detail=""):
        if self.incomplete is None:
            self.incomplete = {"status": "incomplete", "reason": reason,
                               "detail": str(detail)[:256], "run_id": self.run_id}

    def active(self):
        if self.thread != _capture_threading.get_ident():
            self.fail("identity_drift", "foreign_thread")
            return False
        return self.incomplete is None


@contextmanager
def _period_support_scope(observer, *, run_id, signature_targets, match_targets):
    """Explicit single-owner evaluation setup; never used by default inference."""
    if _PERIOD_SUPPORT_CONTEXT.get() is not None or not _PERIOD_SUPPORT_OWNER.acquire(False):
        raise _PeriodSupportIncomplete("identity_drift", "nested_or_concurrent_capture")
    token = None
    try:
        state = _PeriodSupportState(observer, run_id, signature_targets, match_targets)
        token = _PERIOD_SUPPORT_CONTEXT.set(state)
        yield state
    finally:
        if token is not None:
            _PERIOD_SUPPORT_CONTEXT.reset(token)
        _PERIOD_SUPPORT_OWNER.release()


def _period_support_state():
    state = _PERIOD_SUPPORT_CONTEXT.get()
    return state if state is not None and state.active() else None


@contextmanager
def _period_support_context(**fields):
    state = _period_support_state()
    if state is None:
        yield
        return
    previous = state.context
    try:
        state.memory.reserve(_period_support_size_plan(previous) +
                             _period_support_size_plan(fields), "context")
        state.context = dict(previous, **fields)
    except Exception as error:
        state.fail(getattr(error, "reason", "capture_failure"), type(error).__name__)
        yield
        return
    try:
        yield
    finally:
        state.context = previous


def _period_support_snapshot(value, counters=None, depth=0):
    """No callback ever sees an original object or writable array buffer."""
    counters = [0, 0] if counters is None else counters
    if depth == 0:
        state = _period_support_state()
        budget = state.memory if state is not None else _PeriodSupportMemory()
        budget.reserve(_period_support_size_plan(value), "snapshot")
    counters[1] += 1
    if depth > 24 or counters[1] > 100_000:
        raise _PeriodSupportIncomplete("cap_exhausted", "snapshot_structure")
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.dtype.itemsize not in (1, 2, 4, 8):
            raise _PeriodSupportIncomplete("unsupported_dtype", value.dtype.str)
        if value.nbytes > _PERIOD_SUPPORT_ARRAY_CAP:
            raise _PeriodSupportIncomplete("cap_exhausted", "native_array")
        counters[0] += int(value.nbytes)
        raw = value.tobytes(order="C")
        return np.frombuffer(raw, dtype=value.dtype).reshape(value.shape)
    if _capture_dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"record_type": type(value).__name__, "fields": {
            field.name: _period_support_snapshot(getattr(value, field.name), counters, depth + 1)
            for field in _capture_dataclasses.fields(value)}}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            return {"mapping_entries": tuple(
                (_period_support_snapshot(key, counters, depth + 1),
                 _period_support_snapshot(item, counters, depth + 1))
                for key, item in value.items())}
        return {key: _period_support_snapshot(item, counters, depth + 1)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_period_support_snapshot(item, counters, depth + 1) for item in value)
    if isinstance(value, np.generic):
        return _period_support_snapshot(value.item(), counters, depth + 1)
    if value is None or type(value) in (bool, int, float, str, bytes):
        if isinstance(value, (str, bytes)) and len(value) > _PERIOD_SUPPORT_ARRAY_CAP:
            raise _PeriodSupportIncomplete("cap_exhausted", "native_scalar_payload")
        return value
    raise _PeriodSupportIncomplete("missing_dependency", type(value).__name__)


def _period_support_event(kind, **values):
    state = _period_support_state()
    if state is None:
        return
    if (state.context.get("candidate_axis") not in (None, 1)
            and kind not in ("candidate_begin", "candidate_window", "candidate_reading")):
        return
    try:
        context = _period_support_snapshot(dict(state.context, run_id=state.run_id))
        if state.context.get("candidate_axis") not in (None, 1):
            values = {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
        snapshot = _period_support_snapshot(values)
        state.observer(kind, context, snapshot)
    except Exception as error:
        state.fail("cap_exhausted" if isinstance(error, _WitnessBudgetExceeded) else getattr(error, "reason", "capture_failure"), type(error).__name__)


def _period_support_private(factory):
    """Catch only diagnostic failures; interrupts still unwind scope finally."""
    state = _period_support_state()
    if state is None:
        return None
    try:
        return factory(state)
    except Exception as error:
        state.fail("cap_exhausted" if isinstance(error, _WitnessBudgetExceeded) else getattr(error, "reason", "capture_failure"), type(error).__name__)
        return None


def _period_support_signature(identity):
    state = _period_support_state()
    if state is None or tuple(identity[:2]) not in state.signature_targets:
        return False
    identity = tuple(identity)
    if identity in state.signature_seen:
        state.fail("ambiguous_join", "duplicate_signature")
        return False
    if len(state.signature_seen) >= 64:
        state.fail("cap_exhausted", "signature_witnesses")
        return False
    state.signature_seen.add(identity)
    return True


def _period_support_match(identity):
    state = _period_support_state()
    if state is None or tuple(identity) not in state.match_targets:
        return False
    identity = tuple(identity)
    if identity in state.match_seen:
        state.fail("ambiguous_join", "duplicate_match")
        return False
    if len(state.match_seen) >= 3:
        state.fail("cap_exhausted", "matcher_witnesses")
        return False
    state.match_seen.add(identity)
    return True


class _WitnessBudgetExceeded(Exception):
    def __init__(self, stage: str, limit: int, planned: int, actual: int = 0):
        self.stage, self.limit, self.planned, self.actual = stage, limit, planned, actual


@dataclass(frozen=True, slots=True)
class WitnessBudgetV1:
    max_rectified_samples: int = 10_000_000
    max_bilinear_taps: int = 40_000_000
    max_cell_memberships: int = 40_000_000
    max_payload_bytes: int = 600_000_000
    max_pair_entries: int = 10_000_000
    max_owner_faces: int = 1_000_000
    max_owner_vertices: int = 8_000_000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (
                self.max_rectified_samples, self.max_bilinear_taps,
                self.max_cell_memberships, self.max_payload_bytes,
                self.max_pair_entries, self.max_owner_faces,
                self.max_owner_vertices)):
            raise ValueError("witness budget caps must be non-negative integers")


@dataclass(frozen=True, slots=True)
class WitnessRefusalV1:
    status: str
    reason: str
    stage: str
    limit: int
    planned: int
    actual: int

    def __post_init__(self) -> None:
        if (self.status != "refused" or self.reason != "budget_exhausted"
                or not self.stage or any(type(value) is not int or value < 0
                                         for value in (self.limit, self.planned,
                                                       self.actual))):
            raise ValueError("witness refusal is malformed")


def _budget_refusal(error: _WitnessBudgetExceeded) -> WitnessRefusalV1:
    return WitnessRefusalV1("refused", "budget_exhausted", error.stage,
                            error.limit, error.planned, error.actual)

TEXELS_PER_BLOCK = 16
"""Minecraft's texture resolution, and the size of the period ambiguity."""

MINIMUM_PERIOD = 4.0
"""Shortest period worth believing, in rectified pixels.

Below about four pixels a period is indistinguishable from pixel noise and from
the resampling the rectification itself introduces.
"""

OVERSAMPLE = 2.0
"""Rectified samples per source pixel across the patch.

Sampling the patch too coarsely is not a loss of precision, it is a wrong answer.
A texel on a wall eight blocks away lands on about 1.9 rectified pixels at 512
samples - under `MINIMUM_PERIOD` - so the estimator skips it and locks onto the
first harmonic instead, reporting a period twice too long and a distance half
too near. Measured on Template 2's floor, an isolated plane read 4.13 blocks
against a true 8.38 at 512 samples and 8.39 at 2048.

Sampling at twice the source resolution keeps the finest real detail - texels -
comfortably above the floor without inventing information that is not in the
pixels.
"""

MINIMUM_CONTRAST = 1e-6
"""Signal variance under which a profile carries no periodic information."""

COARSE_SUPPORT = 0.5
"""Correlation at sixteen times the period, relative to the period's own, above
which the measured period is read as texels rather than block seams."""

HARMONIC_SHARE = 0.75
"""How tall a shorter peak must be, against the tallest, to be preferred.

Autocorrelation cannot tell a spacing from twice that spacing, and the double
wins whenever alternate seams differ - two alternating materials, a repeating
two-block pattern, a light falling across the wall. Since a harmonic implies a
distance wrong by exactly that integer factor, the shortest peak of comparable
height is the safer reading.
"""


@dataclass(frozen=True)
class PeriodHypothesis:
    """One reading of a measured period, and what it implies about distance."""

    interpretation: str
    blocks_per_period: float
    plane_distance: float
    score: float


@dataclass(frozen=True)
class PeriodEstimate:
    """A measured spatial period, its phase, and the readings consistent with it."""

    period_pixels: float
    phase_pixels: float
    strength: float
    block_supports: tuple[tuple[int, float], ...] = ()
    """Correlation at each candidate block-seam multiple of the period.

    A period is not guaranteed to be one texel. Many Minecraft textures have
    their strongest detail at two texels, and then block seams sit eight periods
    away rather than sixteen. Reading the multiple off the correlation, instead
    of assuming sixteen, is what separates a wall fourteen blocks off from the
    same wall reported at seven.
    """

    coarse_support: float = 0.0
    """Correlation at sixteen times the period, relative to the period's own.

    This is what tells texels from block seams without a second view. If the
    measured spacing is texels then block seams sit sixteen of them apart and
    leave their own peak; if the measured spacing is already block seams there is
    nothing at sixteen times it but the wall's own repetition. Strong support
    identifies the fine period as texels, which is the more precise reading of
    the two because its scale is sixteen times finer.
    """
    hypotheses: tuple[PeriodHypothesis, ...] = field(default_factory=tuple)

    @property
    def best(self) -> PeriodHypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None


def sampling_for(extent: float, focal: float,
                 oversample: float = OVERSAMPLE) -> int:
    """Rectified patch size for a given extent and focal length.

    Both `rectify_plane` and `period_hypotheses` need the same number, because
    one produces the pixels and the other converts their spacing into a distance.
    Disagreeing about it scales every distance by the ratio.
    """
    return int(max(64, min(8192, round(2.0 * extent * focal * oversample))))


def rectified_pixel_step(extent: float, samples: int) -> float:
    """Return the world-space spacing of an endpoint-inclusive rectified grid.

    :func:`rectify_plane` places ``samples`` coordinates from ``-extent`` to
    ``+extent`` with ``numpy.linspace``. The exact interval is therefore
    ``2 * extent / (samples - 1)``. At least two samples are required to define
    a spacing. Invalid dimensions raise ``ValueError`` so callers cannot
    silently fall back to the former ``2*extent/samples`` approximation.
    """
    if (not isinstance(samples, (int, np.integer))
            or isinstance(samples, (bool, np.bool_)) or samples < 2):
        raise ValueError("endpoint-inclusive rectification needs at least two integer samples")
    if isinstance(extent, (bool, np.bool_)):
        raise ValueError("rectified extent must be finite and positive")
    try:
        extent_value = float(extent)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("rectified extent must be finite and positive") from error
    if not np.isfinite(extent_value) or extent_value <= 0.0:
        raise ValueError("rectified extent must be finite and positive")
    step = (2.0 * extent_value) / (int(samples) - 1)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("rectified pixel step must be finite and positive")
    return float(step)


GRAZING_INCIDENCE = 0.05
"""Smallest normalized ray-to-plane incidence worth intersecting.

Below this a ray runs nearly parallel to the plane, so the intersection races off
toward infinity and a pixel of pointing error moves it by an unbounded amount.
The constraint is reported ill-conditioned rather than answered.
"""


@dataclass(frozen=True)
class PlaneRayMap:
    """Stable rays mapped to one unit-distance plane on one camera side."""

    side: float
    coordinates: np.ndarray
    stable: np.ndarray
    incidence: np.ndarray


@dataclass(frozen=True)
class PlaneRayAbstention:
    """Typed failure to intersect image rays with a unit-distance plane."""

    reason: str
    detail: str


def _map_rays_to_unit_plane(
    points,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    image_size: tuple[int, int],
    principal_point: tuple[float, float] | None,
    *,
    require_all: bool,
    side_epsilon: float = 0.0,
) -> PlaneRayMap | PlaneRayAbstention:
    """Map stable image rays with a single normalized-incidence contract.

    Window extents may discard pixels on the other side of the vanishing line;
    point-to-point constraints cannot, and set ``require_all`` to fail closed.
    """
    if (not isinstance(axis, (int, np.integer)) or isinstance(axis, (bool, np.bool_))
            or int(axis) not in (0, 1, 2)):
        return PlaneRayAbstention("unmeasurable_plane", "axis must be 0, 1, or 2")
    try:
        pixels = np.atleast_2d(np.asarray(points, dtype=float))
        matrix = np.asarray(axis_matrix, dtype=float)
        width, height = int(image_size[0]), int(image_size[1])
        focal_value = float(focal)
        side_epsilon_value = float(side_epsilon)
        centre = (principal_point if principal_point is not None
                  else (float(width) / 2.0, float(height) / 2.0))
        centre_x, centre_y = float(centre[0]), float(centre[1])
    except (TypeError, ValueError, IndexError, OverflowError):
        return PlaneRayAbstention("unmeasurable_plane", "plane mapping inputs are malformed")
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0
            or not np.isfinite(pixels).all() or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or not np.isfinite(focal_value)
            or focal_value <= 0.0 or not np.isfinite(side_epsilon_value)
            or side_epsilon_value < 0.0 or int(width) <= 0 or int(height) <= 0
            or not np.isfinite((centre_x, centre_y)).all()):
        return PlaneRayAbstention("unmeasurable_plane", "plane mapping inputs are invalid")
    normal, first, second = plane_basis(int(axis), matrix)
    normal_length = float(np.linalg.norm(normal))
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if (not np.isfinite((normal_length, first_length, second_length)).all()
            or min(normal_length, first_length, second_length) <= 0.0):
        return PlaneRayAbstention("unmeasurable_plane", "plane basis is degenerate")
    rays = np.stack(((pixels[:, 0] - centre_x) / focal_value,
                     -(pixels[:, 1] - centre_y) / focal_value,
                     np.ones(pixels.shape[0], dtype=float)), axis=1)
    lengths = np.linalg.norm(rays, axis=1)
    if not np.isfinite(rays).all() or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        return PlaneRayAbstention("unmeasurable_plane", "image rays are non-finite or degenerate")
    unit_rays = rays / lengths[:, None]
    unit_normal = normal / normal_length
    incidence = unit_rays @ unit_normal
    raw_projection = rays @ normal
    positive = int(np.count_nonzero(raw_projection > side_epsilon_value))
    negative = int(np.count_nonzero(raw_projection < -side_epsilon_value))
    if positive == 0 and negative == 0:
        return PlaneRayAbstention("unmeasurable_plane", "all rays lie on the plane vanishing line")
    side = 1.0 if positive >= negative else -1.0
    signed_incidence = incidence * side
    stable = signed_incidence > GRAZING_INCIDENCE
    if not np.any(stable):
        return PlaneRayAbstention("unmeasurable_plane", "no ray exceeds the grazing-incidence limit")
    if require_all and not bool(np.all(stable)):
        return PlaneRayAbstention(
            "unmeasurable_plane",
            "point rays cross the vanishing line or include grazing incidence")
    # The normalized dot product gates conditioning. Retain the original
    # homogeneous ray/axis-basis algebra for accepted coordinates.
    denominator = raw_projection * side
    points_on_plane = rays[stable] / denominator[stable, None]
    coordinates = np.stack((points_on_plane @ first, points_on_plane @ second), axis=1)
    if not np.isfinite(coordinates).all():
        return PlaneRayAbstention("unmeasurable_plane", "plane coordinates are non-finite")
    return PlaneRayMap(side, coordinates, stable, incidence)

MINIMUM_SEPARATION = 1e-4
"""Separation on the unit-distance plane below which two points state nothing."""


@dataclass(frozen=True)
class LengthConstraint:
    """What a user's stated length did or did not establish.

    A distance is returned only when the constraint is consistent. Anything else
    carries a status and no number, because a plausible-looking distance derived
    from an impossible constraint is worse than no answer: nothing downstream can
    tell it from a measured one.
    """

    status: str
    distance: float | None = None
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.distance is not None


def plane_coordinates(
    points,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    image_size: tuple[int, int],
    *,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """Where image points land on the unit-distance plane, as ``(u, v)`` rows.

    The same intersection and normalized incidence gate as `plane_window`,
    exposed because a user pointing at two places on a wall needs it too. Every
    point must lie on the selected camera side and exceed
    :data:`GRAZING_INCIDENCE`; otherwise the whole constraint abstains.
    """
    mapped = _map_rays_to_unit_plane(points, axis, axis_matrix, focal, image_size,
                                     principal_point, require_all=True)
    if isinstance(mapped, PlaneRayAbstention):
        return None
    return mapped.coordinates


def scale_from_known_length(
    point_a,
    point_b,
    blocks_apart: float,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    image_size: tuple[int, int],
    *,
    principal_point: tuple[float, float] | None = None,
) -> "LengthConstraint":
    """Distance to a plane, from a length the user states across it.

    One image fixes a plane's *shape* and never its size: a wall twice as far
    away with blocks twice as big is the same picture. Everything else in this
    module works around that by reading the block grid, which leaves the
    block-versus-texel ambiguity because both readings are consistent with the
    same picture.

    A stated length settles it outright and needs no correspondence, no second
    view and no feature matching - which is what makes it the cheapest thing to
    ask of a user, and the one constraint that survives Minecraft's repetition.
    The user names two points on one surface and says how many blocks apart they
    are; the plane's distance follows, because separation on the unit-distance
    plane scales exactly with distance.

    Both points must lie on the *same* plane. That is the user's assertion and
    this cannot check it.
    """
    if not np.isfinite(blocks_apart) or blocks_apart <= 0.0:
        return LengthConstraint("invalid_length",
                                detail=f"stated length {blocks_apart!r} is not a positive number")
    pixels = np.asarray([point_a, point_b], dtype=float)
    if not np.all(np.isfinite(pixels)):
        return LengthConstraint("non_finite", detail="a stated point is not finite")

    coordinates = plane_coordinates(pixels, axis, axis_matrix, focal,
                                    image_size, principal_point=principal_point)
    if coordinates is None:
        return LengthConstraint(
            "inconsistent",
            detail=("the two points do not meet one plane in front of the camera: "
                    "their rays fall on opposite sides of its vanishing line, or "
                    "one grazes it too nearly to intersect stably"))
    separation = float(np.linalg.norm(coordinates[1] - coordinates[0]))
    if not np.isfinite(separation) or separation <= MINIMUM_SEPARATION:
        return LengthConstraint(
            "ill_conditioned",
            detail=f"the points are {separation:.2e} apart on the plane, which "
                   f"states nothing about its scale")
    return LengthConstraint("resolved", distance=blocks_apart / separation)


def resolve_hypotheses(
    estimate: PeriodEstimate,
    known_distance: float,
    *,
    tolerance: float = 0.25,
) -> PeriodEstimate:
    """Reorder the readings by which one a known distance actually supports.

    `period_hypotheses` keeps both the block and texel readings because a single
    image cannot separate them. A stated length can: whichever reading lands
    nearer the distance it implies is the true one, and the loser is retained
    rather than deleted so the ambiguity stays visible in the record.
    """
    if not estimate.hypotheses or known_distance <= 0.0:
        return estimate
    ranked = sorted(estimate.hypotheses,
                    key=lambda h: abs(h.plane_distance - known_distance))
    best = ranked[0]
    agrees = abs(best.plane_distance - known_distance) <= tolerance * known_distance
    return PeriodEstimate(
        period_pixels=estimate.period_pixels,
        phase_pixels=estimate.phase_pixels,
        strength=estimate.strength,
        block_supports=estimate.block_supports,
        coarse_support=estimate.coarse_support,
        hypotheses=tuple(ranked) if agrees else estimate.hypotheses,
    )


def plane_window(
    mask: np.ndarray,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    *,
    principal_point: tuple[float, float] | None = None,
    margin: float = 1.05,
) -> tuple[float, float, float, float] | None:
    """Where a masked region lands on the unit-distance plane: centre and half-span.

    Rectifying a fixed window about the plane's centre works only when the
    surface of interest happens to sit there. A proposed region generally does
    not, and then the window either misses it or gives it a handful of pixels in
    a corner - which is why every region measured on Template 2 returned no
    period at all. The window has to follow the region.

    Returned square, because a rectangular one would put a different scale on
    each axis and the period-to-distance conversion assumes a single scale.

    The fourth value is which **side** of the camera the surface is on. The
    dominant side is retained and only same-side rays whose normalized
    incidence exceeds :data:`GRAZING_INCIDENCE` contribute coordinates. A plane
    normal names an axis, not a direction along it, so a floor below the eye and
    a ceiling above it share one. Placing the reference plane only at `+normal`
    silently excludes half of every scene - the first version did, and returned
    nothing at all for four regions out of four on Template 2, having worked
    earlier only because the surface it was tested on happened to be a ceiling.
    """
    image_mask = np.asarray(mask)
    if image_mask.ndim != 2:
        return None
    ys, xs = np.nonzero(image_mask > 0.5)
    if ys.size == 0:
        return None
    height, width = image_mask.shape[:2]
    try:
        margin_value = float(margin)
    except (TypeError, ValueError, OverflowError):
        return None
    points = np.column_stack((xs, ys))
    mapped = _map_rays_to_unit_plane(
        points, axis, axis_matrix, focal, (width, height), principal_point,
        require_all=False, side_epsilon=1e-6)
    if isinstance(mapped, PlaneRayAbstention):
        return None
    us, vs = mapped.coordinates[:, 0], mapped.coordinates[:, 1]
    centre_u = float((us.min() + us.max()) / 2.0)
    centre_v = float((vs.min() + vs.max()) / 2.0)
    half = float(max(us.max() - us.min(), vs.max() - vs.min()) / 2.0) * margin_value
    if (not np.isfinite((centre_u, centre_v, half, margin_value)).all()
            or half <= 0.0 or margin_value <= 0.0):
        return None
    return centre_u, centre_v, half, mapped.side


def plane_basis(axis: int, axis_matrix: np.ndarray):
    """Normal and two in-plane directions for a world-axis-aligned plane.

    ``axis_matrix`` is `calibration.axis_directions`: row *i* is world axis *i*
    expressed in camera coordinates.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    others = [index for index in range(3) if index != axis]
    return axis_matrix[axis], axis_matrix[others[0]], axis_matrix[others[1]]


def rectify_plane(
    image: np.ndarray,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    *,
    extent: float = 2.0,
    samples: int | None = None,
    principal_point: tuple[float, float] | None = None,
    oversample: float = OVERSAMPLE,
    centre: tuple[float, float] = (0.0, 0.0),
    side: float = 1.0,
    witness_sink=None,
    witness_budget: WitnessBudgetV1 | None = None,
) -> np.ndarray | None:
    """Warp the image to a fronto-parallel view of a plane at unit distance.

    The output spans ``[-extent, extent]`` in plane coordinates on both axes, so
    the exact endpoint-inclusive pixel step is
    ``rectified_pixel_step(extent, samples)`` world units *at unit distance*.
    A real plane at distance ``d`` shows its blocks ``d`` times larger than that,
    which is exactly the relation `period_hypotheses` inverts.

    ``samples`` defaults to the source resolution across the patch times
    `OVERSAMPLE`, because choosing it too small silently halves the answer; see
    that constant. Pass a number only to override the rule deliberately, and pass
    the same number to `period_hypotheses` so the two agree about scale.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - warping needs OpenCV
        raise RuntimeError("plane rectification needs OpenCV")

    height, width = image.shape[:2]
    centre_x, centre_y = principal_point or (width / 2.0, height / 2.0)
    normal, first, second = plane_basis(axis, axis_matrix)
    if samples is None:
        samples = sampling_for(extent, focal, oversample)
    rectified_pixel_step(extent, samples)

    grid_u, grid_v = np.meshgrid(
        np.linspace(centre[0] - extent, centre[0] + extent, samples),
        np.linspace(centre[1] - extent, centre[1] + extent, samples))
    # A point on the unit-distance plane at in-plane coordinates (u, v).
    points = (side * normal[None, None, :]
              + grid_u[..., None] * first[None, None, :]
              + grid_v[..., None] * second[None, None, :])
    depth = points[..., 2]
    valid = depth > 1e-6
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = centre_x + points[..., 0] * focal / depth
        ys = centre_y - points[..., 1] * focal / depth
    xs = np.where(valid, xs, -1.0).astype(np.float32)
    ys = np.where(valid, ys, -1.0).astype(np.float32)
    result = cv2.remap(image, xs, ys, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if witness_sink is not None:
        budget = witness_budget or WitnessBudgetV1()
        try:
            witness_sink.append(rectification_witness_v1(
                xs, ys, valid, source_shape=(height, width),
                max_samples=budget.max_rectified_samples,
                max_taps=budget.max_bilinear_taps,
                max_payload_bytes=budget.max_payload_bytes))
        except _WitnessBudgetExceeded as error:
            witness_sink.append(_budget_refusal(error))
    state = _period_support_state()
    if state is not None and state.context.get("signature_target"):
        def observe_rectification(capture):
            if len(capture.rectifications) >= 3:
                raise _PeriodSupportIncomplete("ambiguous_join", "rectification_count")
            capture.memory.reserve(512 * int(xs.size) + 1024**2,
                                   "rectification_mapping_lists_and_buffers")
            witness = rectification_witness_v1(
                xs, ys, valid, source_shape=(height, width),
                max_samples=262144, max_taps=1048576, max_payload_bytes=32 * 1024**2)
            capture.rectifications.append(witness)
            _period_support_event("signature_rectification",
                rectification_index=len(capture.rectifications)-1,
                geometry=dict(axis=axis, axis_matrix=axis_matrix, focal=focal,
                    principal_point=(centre_x, centre_y), centre=centre,
                    extent=extent, samples=samples, side=side),
                source_sha256=hashlib.sha256(memoryview(image)).hexdigest(),
                source_dtype=image.dtype.str, result=result,
                metadata=witness.to_record(), arrays={
                    name: getattr(witness, name) for name in (
                        "map_x", "map_y", "depth_valid", "tap_ptr",
                        "tap_source_indices", "tap_weight_numerators")})
        _period_support_private(observe_rectification)
    elif state is not None and state.context.get("candidate_axis") == 1:
        _period_support_private(lambda _capture: _period_support_event("measurement_rectification", axis=axis, axis_matrix=axis_matrix,
            focal=focal, principal_point=(centre_x, centre_y), centre=centre,
            extent=extent, samples=samples, side=side, map_x=xs, map_y=ys,
            depth_valid=valid, normal=normal, first=first, second=second,
            source_sha256=hashlib.sha256(memoryview(image)).hexdigest(),
            source_dtype=image.dtype.str, result=result,
            interpolation="INTER_LINEAR", border="BORDER_CONSTANT_ZERO"))
    return result


def _witness_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_record(value: np.ndarray) -> dict:
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


@dataclass(frozen=True, slots=True)
class RectificationWitnessV1:
    schema: str
    opencv_version: str
    opencv_build_sha256: str
    interpolation: str
    border: str
    inter_bits: int
    weight_denominator: int
    source_shape: tuple[int, int]
    map_x: np.ndarray
    map_y: np.ndarray
    depth_valid: np.ndarray
    tap_ptr: np.ndarray
    tap_source_indices: np.ndarray
    tap_weight_numerators: np.ndarray
    witness_sha256: str = ""

    def __post_init__(self) -> None:
        import cv2
        arrays = (self.map_x, self.map_y, self.depth_valid, self.tap_ptr,
                  self.tap_source_indices, self.tap_weight_numerators)
        if (self.schema != "mc-photogrammetry-rectification-witness-v1"
                or self.opencv_version != cv2.__version__
                or self.opencv_build_sha256 != hashlib.sha256(
                    cv2.getBuildInformation().encode()).hexdigest()
                or self.interpolation != "INTER_LINEAR"
                or self.border != "BORDER_CONSTANT_ZERO"
                or self.inter_bits != 5 or self.weight_denominator != 1024
                or any(not isinstance(value, np.ndarray) or value.flags.writeable
                       for value in arrays)
                or self.map_x.dtype != np.float32 or self.map_y.dtype != np.float32
                or self.depth_valid.dtype != np.bool_
                or self.map_x.shape != self.map_y.shape != self.depth_valid.shape
                or self.tap_ptr.dtype != np.int64
                or self.tap_source_indices.dtype != np.int64
                or self.tap_weight_numerators.dtype != np.uint16
                or self.tap_ptr.shape != (self.map_x.size + 1,)
                or self.tap_ptr[-1] != len(self.tap_source_indices)
                or len(self.tap_source_indices) != len(self.tap_weight_numerators)):
            raise ValueError("rectification witness is malformed or mutable")
        digest = _witness_sha(self.to_record(include_digest=False))
        if self.witness_sha256 not in ("", digest):
            raise ValueError("rectification witness digest mismatch")
        object.__setattr__(self, "witness_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict:
        result = {
            "schema": self.schema, "opencv_version": self.opencv_version,
            "opencv_build_sha256": self.opencv_build_sha256,
            "interpolation": self.interpolation, "border": self.border,
            "inter_bits": self.inter_bits,
            "weight_denominator": self.weight_denominator,
            "source_shape": list(self.source_shape),
            "map_x": _array_record(self.map_x), "map_y": _array_record(self.map_y),
            "depth_valid": _array_record(self.depth_valid),
            "tap_ptr": _array_record(self.tap_ptr),
            "tap_source_indices": _array_record(self.tap_source_indices),
            "tap_weight_numerators": _array_record(self.tap_weight_numerators)}
        if include_digest:
            result["witness_sha256"] = self.witness_sha256
        return result


def _readonly(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value)
    value.flags.writeable = False
    return value


def rectification_witness_v1(map_x: np.ndarray, map_y: np.ndarray,
                             depth_valid: np.ndarray, *, source_shape,
                             max_samples: int = 10_000_000,
                             max_taps: int = 40_000_000,
                             max_payload_bytes: int = 600_000_000) -> RectificationWitnessV1:
    """Decode the exact OpenCV INTER_LINEAR fixed-point tap table."""
    import cv2
    xs = np.asarray(map_x, dtype=np.float32)
    ys = np.asarray(map_y, dtype=np.float32)
    valid = np.asarray(depth_valid, dtype=bool)
    if xs.shape != ys.shape or xs.shape != valid.shape:
        raise ValueError("rectification witness maps differ")
    samples, taps = int(xs.size), int(xs.size * 4)
    payload = samples * (4 + 4 + 1 + 8) + taps * (8 + 2) + 8
    for stage, limit, planned in (
            ("rectified_samples", max_samples, samples),
            ("bilinear_taps", max_taps, taps),
            ("payload_bytes", max_payload_bytes, payload)):
        if planned > limit:
            raise _WitnessBudgetExceeded(stage, limit, planned)
    fixed_xy, fractions = cv2.convertMaps(xs, ys, cv2.CV_16SC2)
    base_x = fixed_xy[..., 0].astype(np.int64)
    base_y = fixed_xy[..., 1].astype(np.int64)
    table = fractions.astype(np.int64)
    fx, fy = table & 31, table >> 5
    numerators = ((32 - fx) * (32 - fy), fx * (32 - fy),
                  (32 - fx) * fy, fx * fy)
    offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    height, width = source_shape
    pointers = [0]
    indices = []
    weights = []
    for position in range(xs.size):
        row, column = np.unravel_index(position, xs.shape)
        for weight_map, (dx, dy) in zip(numerators, offsets):
            weight = int(weight_map[row, column])
            if weight == 0:
                continue
            x, y = int(base_x[row, column] + dx), int(base_y[row, column] + dy)
            indices.append(y * width + x if 0 <= x < width and 0 <= y < height else -1)
            weights.append(weight)
        pointers.append(len(indices))
    return RectificationWitnessV1(
        "mc-photogrammetry-rectification-witness-v1", cv2.__version__,
        hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "INTER_LINEAR", "BORDER_CONSTANT_ZERO", 5, 1024,
        (int(height), int(width)), _readonly(xs.copy()), _readonly(ys.copy()),
        _readonly(valid.copy()), _readonly(np.asarray(pointers, dtype=np.int64)),
        _readonly(np.asarray(indices, dtype=np.int64)),
        _readonly(np.asarray(weights, dtype=np.uint16)))


def rectified_pixel_to_plane_coordinates(
    pixel: tuple[float, float],
    *,
    extent: float,
    samples: int,
    centre: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Invert the ``linspace`` in `rectify_plane`: an output pixel back to (u, v).

    `rectify_plane` builds its sampling grid as
    ``linspace(centre - extent, centre + extent, samples)`` along each axis, so
    a pixel at index *i* sits at ``centre - extent + i * (2*extent)/(samples-1)``.
    This is that map run backwards - the one piece every caller that wants to
    know *where in the world* a scored cell actually is needs, and had been
    reimplemented ad hoc (with the coarser ``2*extent/samples`` approximation)
    in an evaluation script rather than built once and tested here.
    """
    column, row = pixel
    step = rectified_pixel_step(extent, samples)
    u = centre[0] - extent + column * step
    v = centre[1] - extent + row * step
    return (u, v)


def plane_coordinates_to_camera_point(
    axis: int,
    axis_matrix: np.ndarray,
    side: float,
    distance: float,
    u: float,
    v: float,
) -> np.ndarray:
    """A rectified plane's (u, v) coordinate, at a real distance, in camera space.

    The forward half of what `rectify_plane` computes per output pixel -
    ``side*normal + u*first + v*second`` - scaled from the unit-distance plane
    `rectify_plane` works on out to the plane's real distance. Camera space,
    not world space: turning this into a world point still needs the camera's
    own rotation and eye position, which this module does not have access to.
    """
    normal, first, second = plane_basis(axis, axis_matrix)
    unit_point = side * normal + u * first + v * second
    return distance * unit_point


def camera_point_to_pixel(
    point: np.ndarray,
    focal: float,
    principal_point: tuple[float, float],
) -> tuple[float, float] | None:
    """The forward pinhole projection `rectify_plane` uses, for one bare point.

    Returns `None` for a point behind the camera (non-positive depth), the same
    case `rectify_plane` marks invalid.
    """
    depth = float(point[2])
    if depth <= 1e-6:
        return None
    centre_x, centre_y = principal_point
    return (centre_x + point[0] * focal / depth,
            centre_y - point[1] * focal / depth)


def _period_scalar_tools():
    # Local import keeps the optional scalar observer out of the public API.
    from mc_photogrammetry.reconstruction import region_growth
    return region_growth


def _profile(patch: np.ndarray, along: int,
             weights: np.ndarray | None = None) -> np.ndarray:
    """Collapse a rectified patch to a 1-D signal across one in-plane axis.

    Seams run straight in a rectified view, so averaging along them keeps the
    periodic edge signal and averages the block interiors away.

    ``weights`` restricts the average to part of the patch, which is how a single
    planar region is measured without the surfaces around it. Masking the image
    instead would work no better than not masking at all: a hard mask edge is
    itself a strong gradient, and it would be averaged in alongside the seams.
    """
    tools = _period_scalar_tools()
    state = tools._period_scalar_state.get() if tools._attrition_state.get() is not None else None
    positive_bins = None
    result = None
    try:
        grey = patch if patch.ndim == 2 else patch.mean(axis=2)
        if patch.ndim == 2 and np.issubdtype(grey.dtype, np.integer):
            grey = grey.astype(np.float64)
        gradient = np.abs(np.diff(grey, axis=1 - along))
        if weights is None:
            result = gradient.mean(axis=along)
            positive_bins = int(result.size)
            _period_support_event("profile", along=along, profile=result)
            return result
        mask = weights if weights.ndim == 2 else weights.mean(axis=2)
        mask = mask[:, :-1] if along == 0 else mask[:-1, :]
        total = mask.sum(axis=along)
        covered = (gradient * mask).sum(axis=along)
        positive = total > 0
        result = np.divide(covered, total, out=np.zeros_like(covered), where=positive)
        if state is not None:
            with tools._scalar_observation_guard(tools._attrition_state.get()):
                positive_bins = tools._scalar_period_bins(positive, bool)

        _period_support_event("profile", along=along, profile=result)
        return result
    finally:
        if state is not None:
            with tools._scalar_observation_guard(tools._attrition_state.get()):
                state["profile"] = dict(profile_length=int(result.size) if result is not None else None,
                    profile_length_state="measured" if result is not None else "not_evaluated",
                    positive_weight_bins=positive_bins,
                    positive_weight_bins_state="measured" if positive_bins is not None else "not_evaluated",
                    profile_status="returned" if result is not None else "exception")


def estimate_period(
    patch: np.ndarray,
    along: int = 0,
    *,
    minimum_period: float = MINIMUM_PERIOD,
    maximum_period: float | None = None,
    weights: np.ndarray | None = None,
) -> PeriodEstimate | None:
    """The dominant spatial period and phase of a rectified patch.

    Autocorrelation rather than a Fourier peak, because the signal is a train of
    narrow seams rather than a sinusoid: its energy is spread across harmonics,
    and the autocorrelation's first strong peak is the spacing, whereas the
    spectrum's tallest line may be any harmonic of it.
    """
    tools = _period_scalar_tools()
    state = tools._period_scalar_state.get() if tools._attrition_state.get() is not None else None
    tools._scalar_period_expect("period_profile")
    reason = "exception"
    variance = zero_lag = low = high = tallest = None
    peak_below = peak_inside = peak_above = None
    captured_estimate = None
    with _period_support_context(period_direction=along, profile_use="directional"):
        try:
            signal = _profile(patch, along, weights)
            signal = signal - signal.mean()
            if signal.size < 8 or (variance := float(signal.var())) < MINIMUM_CONTRAST:
                reason = "short_profile" if signal.size < 8 else "flat_profile"
                return None
            limit = maximum_period or signal.size / 3.0

            correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
            zero_lag = float(correlation[0])
            if correlation[0] <= 0:
                reason = "nonpositive_zero_lag"
                return None
            correlation = correlation / correlation[0]

            low = max(int(np.floor(minimum_period)), 1)
            high = min(int(np.ceil(limit)), correlation.size - 1)
            if high <= low:
                reason = "empty_lag_interval"
                return None

            # Peaks are found across the whole correlation and only then restricted to
            # the allowed range. Finding them inside the window instead hides any peak
            # sitting on its edge, so a period landing exactly on `minimum_period` was
            # invisible and the estimator fell through to its first harmonic - twice the
            # spacing, and so half the distance.
            interior = ((correlation[1:-1] >= correlation[:-2])
                        & (correlation[1:-1] >= correlation[2:]))
            peaks = np.flatnonzero(interior) + 1
            if state is not None:
                with tools._scalar_observation_guard(tools._attrition_state.get()):
                    peak_below = tools._scalar_period_bins(peaks, lambda p: p < low)
                    peak_above = tools._scalar_period_bins(peaks, lambda p: p >= high)
            peaks = peaks[(peaks >= low) & (peaks < high)]
            peak_inside = int(peaks.size)
            if peaks.size == 0:
                reason = "no_eligible_peak"
                return None
            tallest = float(correlation[peaks].max())
            if tallest <= 0.0:
                reason = "nonpositive_peaks"
                return None

            # Prefer the fundamental over its harmonics. Autocorrelation peaks just as
            # hard at twice the spacing as at the spacing, and if alternate seams differ
            # at all - alternating materials, a two-block pattern, a light falling across
            # the wall - the double can win outright. Every harmonic implies a distance
            # wrong by that integer factor, so among peaks of comparable height the
            # shortest is taken.
            candidates = peaks[correlation[peaks] >= HARMONIC_SHARE * tallest]
            peak = int(candidates[0]) if candidates.size else int(peaks[np.argmax(correlation[peaks])])
            strength = float(correlation[peak])

            # Parabolic interpolation around the integer peak: the period is rarely a
            # whole number of pixels and rounding it biases every distance derived here.
            if 1 <= peak < correlation.size - 1:
                before, centre, after = (correlation[peak - 1], correlation[peak],
                                         correlation[peak + 1])
                denominator = before - 2.0 * centre + after
                if abs(denominator) > 1e-12:
                    peak = peak + 0.5 * (before - after) / denominator
            # The peak index was already constrained to the allowed range; sub-pixel
            # interpolation can nudge it a fraction below, which is not a reason to
            # discard a real measurement.
            period = float(peak)
            if period < minimum_period - 1.0:
                reason = "interpolated_below_minimum"
                return None

            # Phase from the strongest seam: project the signal onto one cycle.
            angles = 2.0 * np.pi * np.arange(signal.size) / period
            phase = float(np.angle(np.sum(signal * np.exp(-1j * angles))))
            phase_pixels = float((phase / (2.0 * np.pi)) * period % period)

            # Is there a second, coarser period exactly sixteen times this one? That is
            # the signature of texels: their block seams are sixteen texels apart and
            # leave their own correlation peak. A period that is already block seams has
            # nothing there.
            # Where does the block seam sit, in multiples of this period? If the measured
            # period is one texel it is sixteen periods away; if the texture's dominant
            # detail is two texels wide - and many are - it is eight. Checking a range of
            # multiples is what lets the reading be right in both cases.
            supports: dict[int, float] = {}
            for multiple in BLOCK_MULTIPLES:
                index = int(round(period * multiple))
                if 0 < index < correlation.size - 1:
                    nearby = correlation[max(index - 2, 0):index + 3]
                    supports[multiple] = float(nearby.max() / strength) if strength > 0 else 0.0
            coarse = supports.get(TEXELS_PER_BLOCK, 0.0)

            captured_estimate = PeriodEstimate(period_pixels=period, phase_pixels=phase_pixels,
                                  strength=strength, coarse_support=coarse,
                                  block_supports=tuple(sorted(supports.items())))

            reason = "accepted"
            return captured_estimate
        finally:
            if state is not None:
                with tools._scalar_observation_guard(tools._attrition_state.get()):
                    if state["profile"] is not None:
                        tools._scalar_period_finish("period_profile", state["profile"]["profile_status"],
                            **state["profile"], variance=variance,
                            variance_state="measured" if variance is not None else "not_evaluated")
                    fields = dict(direction=int(along), profile_length=int(signal.size) if "signal" in locals() else None,
                        variance=variance, zero_lag=zero_lag, low=low, high=high,
                        peaks_below=peak_below, peaks_inside=peak_inside, peaks_above=peak_above, tallest=tallest,
                        selected_period=float(captured_estimate.period_pixels) if captured_estimate is not None else None,
                        selected_strength=float(captured_estimate.strength) if captured_estimate is not None else None)
                    for name in ("profile_length", "variance", "zero_lag", "low", "high", "peaks_below",
                                 "peaks_inside", "peaks_above", "tallest", "selected_period", "selected_strength"):
                        fields[name + "_state"] = "measured" if fields[name] is not None else "not_evaluated"
                    tools._scalar_period_finish("period_estimate", reason, **fields)
            _period_support_event("period_estimate", along=along, estimate=captured_estimate,
                minimum_period=minimum_period, maximum_period=maximum_period,
                profile=locals().get("signal"), correlation=locals().get("correlation"),
                peaks=locals().get("peaks"), harmonic_candidates=locals().get("candidates"))


def _plane_period_with_direction(
    patch: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    minimum_period: float = MINIMUM_PERIOD,
    tolerance: float = 0.12,
) -> tuple[PeriodEstimate, int] | None:
    """Reconcile the two in-plane readings of one plane into a single spacing.

    A plane has one lattice, so its two in-plane directions must report the same
    spacing. Measured on Template 2 they frequently do not: one direction returns
    the true period and the other exactly twice it, and *which* one flips between
    views of the same floor. That rules out a material pattern - it is the more
    foreshortened direction, carrying fewer source pixels, whose texels fall below
    the resolution and whose estimate lands on the first harmonic instead.

    Since a harmonic is always a whole multiple of the fundamental and never a
    fraction of it, the shorter of two readings related by an integer factor is
    the right one. Readings that are not integer multiples are left alone and the
    stronger is returned, because then they are measuring genuinely different
    things and picking the shorter would be guessing.
    """
    try:
        tools = _period_scalar_tools()
        state = tools._period_scalar_state.get() if tools._attrition_state.get() is not None else None
        attempted = []
        reason = "exception"
        if state is not None:
            for along in (0, 1):
                tools._scalar_period_expect("period_estimate", (*state["identity"], along))
        readings = []
        for along in (0, 1):
            attempted.append(along)
            with tools._scalar_period_context(identity=(*state["identity"], along) if state is not None else None):
                readings.append(estimate_period(patch, along, weights=weights,
                                               minimum_period=minimum_period))
        found = [(along, reading) for along, reading in enumerate(readings)
                 if reading is not None]
        if not found:
            reason = "both_directions_failed"
            _period_support_event("period_reconciliation", readings=readings, selected=None, tolerance=tolerance)
            return None
        if len(found) == 1:
            selected = found[0]
            reason = "single_direction"
            _period_support_event("period_reconciliation", readings=readings, selected=found[0], tolerance=tolerance)
            return found[0][1], found[0][0]

        short, long_ = sorted(found, key=lambda item: item[1].period_pixels)
        factor = long_[1].period_pixels / short[1].period_pixels
        nearest = round(factor)
        if nearest >= 2 and abs(factor - nearest) <= tolerance:
            selected = short
            reason = "harmonic_reconciliation"
        else:
            reason = "strongest_direction"
            selected = max(found, key=lambda item: item[1].strength)
        _period_support_event("period_reconciliation", readings=readings, selected=selected, tolerance=tolerance)
        return selected[1], selected[0]
    finally:
        if "state" in locals() and state is not None:
            with tools._scalar_observation_guard(tools._attrition_state.get()):
                tools._scalar_period_finish("period_reconciliation", reason,
                    attempted_directions=tuple(attempted),
                    selected_direction=selected[0] if "selected" in locals() else None,
                    selected_period=float(selected[1].period_pixels) if "selected" in locals() else None,
                    selected_strength=float(selected[1].strength) if "selected" in locals() else None,
                    selected_direction_state="measured" if "selected" in locals() else "not_evaluated",
                    selected_period_state="measured" if "selected" in locals() else "not_evaluated",
                    selected_strength_state="measured" if "selected" in locals() else "not_evaluated")


def plane_period(
    patch: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    minimum_period: float = MINIMUM_PERIOD,
    tolerance: float = 0.12,
) -> PeriodEstimate | None:
    """Reconcile two profiles without changing the public estimate contract."""
    result = _plane_period_with_direction(
        patch, weights=weights, minimum_period=minimum_period, tolerance=tolerance)
    return None if result is None else result[0]


BLOCK_MULTIPLES = (2, 4, 8, 16)
"""Candidate spacings, in periods, at which a block seam may sit.

Equivalently the measured period is 16 / multiple texels wide. Only powers of two
are offered because Minecraft textures are 16 texels square, so their internal
repeats divide it.
"""

TARGET_PERIOD = 10.0
"""Rectified pixels the finest real detail should span, after refinement.

`MINIMUM_PERIOD` is the floor below which a period is rejected; this is the
comfortable margin above it that the sampling is adjusted to reach. A texel
sitting at four or five pixels is measurable in principle and in practice lands
on its first harmonic, which is how a wall fourteen blocks away came back at
seven.
"""


def measure_plane(
    image: np.ndarray,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    *,
    mask: np.ndarray | None = None,
    extent: float = 1.0,
    rounds: int = 3,
    target_period: float = TARGET_PERIOD,
    oversample: float = OVERSAMPLE,
    side: float = 1.0,
) -> tuple["PeriodEstimate | None", int, float]:
    """Measure a plane's lattice, adjusting the sampling to what it turns out to be.

    How finely the patch must be sampled depends on how far away the plane is,
    and the distance is the thing being measured - so a single fixed sampling
    cannot serve both a floor two blocks under the camera and a wall fourteen
    blocks off. The near one is over-sampled harmlessly; the far one has its
    texels squeezed under the floor and returns a harmonic, reading half the true
    distance.

    So measure, use the distance that implies to choose a sampling that would put
    the finest detail at ``target_period`` pixels, and measure again. Two or three
    rounds settle it. Returns the estimate and the sampling it was made at, which
    the caller needs to convert the period into a distance.

    When a ``mask`` is given the window follows it, via `plane_window`, rather
    than sitting at the plane's centre - the caller's region is rarely there, and
    a window that misses it measures nothing. The returned ``extent`` is then the
    region's own half-span, so pass it, not the argument, to
    `period_hypotheses`.

    Without a mask, ``side`` says which of the two planes perpendicular to the
    axis is meant - the floor under the camera or the ceiling over it. They are
    different surfaces and measuring one while claiming the other is how a wall's
    period ends up attached to a ceiling's distance. A mask answers this by
    itself, so the argument is ignored when one is given.
    """
    centre = (0.0, 0.0)
    if mask is not None:
        window = plane_window(mask, axis, axis_matrix, focal)
        if window is None:
            return None, sampling_for(extent, focal, oversample), extent
        centre = (window[0], window[1])
        extent, side = window[2], window[3]

    samples = sampling_for(extent, focal, oversample)
    found = None
    for _round in range(max(1, rounds)):
        patch = rectify_plane(image, axis, axis_matrix, focal,
                              extent=extent, samples=samples, centre=centre,
                              side=side)
        if patch is None:
            return None, samples, extent
        weights = None
        if mask is not None:
            weights = rectify_plane(mask, axis, axis_matrix, focal,
                                    extent=extent, samples=samples, centre=centre,
                                    side=side)
            if weights is None or float(weights.sum()) <= 0.0:
                return None, samples, extent
        found = plane_period(patch, weights=weights)
        if found is None:
            return None, samples, extent
        if found.period_pixels >= target_period:
            break
        wanted = int(round(samples * target_period / max(found.period_pixels, 1e-6)))
        wanted = int(max(64, min(8192, wanted)))
        if wanted <= samples:
            break
        samples = wanted
    return found, samples, extent


def period_hypotheses(
    estimate: PeriodEstimate,
    *,
    extent: float = 2.0,
    samples: int = 512,
) -> PeriodEstimate:
    """Attach the block and texel readings of a measured period.

    One output pixel spans ``rectified_pixel_step(extent, samples)`` world units
    at unit distance,
    so a period of ``p`` pixels is that many world units at unit distance. If
    those units are one block the plane sits that many blocks away; if they are
    one texel it sits sixteen times closer.

    Both are always kept, because a single image can leave them genuinely
    undecided. They are *ordered* by evidence rather than by a prior: a peak at
    sixteen times the measured period says the measured one is texels, and the
    texel reading is then ranked first. Measured on Template 2 at about two
    blocks' range the block-first ordering put walls thirty to fifty blocks away
    when they were under three, so the prior was not merely unproven but wrong.

    **A known limitation, measured and not solved.** This assumes a measured
    period is one texel. Where a texture's strongest detail is two texels wide -
    which is common - the block seam sits eight periods away rather than sixteen
    and the distance comes out exactly half. Template 2's floors read correctly
    and its Z walls read half. Choosing the multiple from the correlation was
    tried and does not work: a periodic signal peaks at *every* multiple of
    itself, so a block seam cannot be told from a texture harmonic by the
    presence of a peak. `block_supports` records the candidates for whatever
    eventually resolves them - a second view, or a user-given length.
    """
    try:
        per_pixel = rectified_pixel_step(extent, samples)
    except ValueError:
        return estimate
    world = estimate.period_pixels * per_pixel
    if world <= 0.0 or not np.isfinite(world):
        return estimate
    block = PeriodHypothesis("one block per period", 1.0, 1.0 / world,
                             estimate.strength)
    texel = PeriodHypothesis("one texel per period", 1.0 / TEXELS_PER_BLOCK,
                             1.0 / (world * TEXELS_PER_BLOCK),
                             estimate.strength * estimate.coarse_support)
    texel_first = estimate.coarse_support >= COARSE_SUPPORT
    return PeriodEstimate(period_pixels=estimate.period_pixels,
                          phase_pixels=estimate.phase_pixels,
                          strength=estimate.strength,
                          block_supports=estimate.block_supports,
                          coarse_support=estimate.coarse_support,
                          hypotheses=(texel, block) if texel_first else (block, texel))
