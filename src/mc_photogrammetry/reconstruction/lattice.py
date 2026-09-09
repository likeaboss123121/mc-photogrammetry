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

import numpy as np

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


GRAZING_INCIDENCE = 0.05
"""Smallest ray-to-plane incidence that is worth intersecting.

Below this a ray runs nearly parallel to the plane, so the intersection races off
toward infinity and a pixel of pointing error moves it by an unbounded amount.
The constraint is reported ill-conditioned rather than answered.
"""

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

    The same mapping `plane_window` uses, exposed because a user pointing at two
    places on a wall needs it too.
    """
    width, height = image_size
    centre_x, centre_y = principal_point or (width / 2.0, height / 2.0)
    normal, first, second = plane_basis(axis, axis_matrix)
    pixels = np.atleast_2d(np.asarray(points, dtype=float))
    rays = np.stack([(pixels[:, 0] - centre_x) / focal,
                     -(pixels[:, 1] - centre_y) / focal,
                     np.ones(pixels.shape[0], dtype=float)], axis=1)
    projected = rays @ normal
    side = 1.0 if float(np.sum(projected > 0)) >= float(np.sum(projected < 0)) else -1.0
    denominator = projected * side
    # Every ray must meet the plane on the same side and in front of the eye.
    # Testing only the magnitude accepted points whose intersection lies behind
    # the camera, so two pixels on opposite sides of the plane's vanishing line
    # returned a plausible distance for a constraint that cannot exist.
    if not np.all(denominator > GRAZING_INCIDENCE):
        return None
    on_plane = rays / denominator[:, None]
    return np.stack([on_plane @ first, on_plane @ second], axis=1)


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
) -> tuple[float, float, float] | None:
    """Where a masked region lands on the unit-distance plane: centre and half-span.

    Rectifying a fixed window about the plane's centre works only when the
    surface of interest happens to sit there. A proposed region generally does
    not, and then the window either misses it or gives it a handful of pixels in
    a corner - which is why every region measured on Template 2 returned no
    period at all. The window has to follow the region.

    Returned square, because a rectangular one would put a different scale on
    each axis and the period-to-distance conversion assumes a single scale.

    The fourth value is which **side** of the camera the surface is on. A plane
    normal names an axis, not a direction along it, so a floor below the eye and
    a ceiling above it share one. Placing the reference plane only at `+normal`
    silently excludes half of every scene - the first version did, and returned
    nothing at all for four regions out of four on Template 2, having worked
    earlier only because the surface it was tested on happened to be a ceiling.
    """
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if ys.size == 0:
        return None
    height, width = mask.shape[:2]
    centre_x, centre_y = principal_point or (width / 2.0, height / 2.0)
    normal, first, second = plane_basis(axis, axis_matrix)

    rays = np.stack([(xs - centre_x) / focal,
                     -(ys - centre_y) / focal,
                     np.ones(xs.shape, dtype=float)], axis=1)
    projected = rays @ normal
    forward = int((projected > 1e-6).sum())
    backward = int((projected < -1e-6).sum())
    if forward == 0 and backward == 0:
        return None
    side = 1.0 if forward >= backward else -1.0
    denominator = projected * side
    usable = denominator > 1e-6
    if not usable.any():
        return None
    points = rays[usable] / denominator[usable][:, None]
    us, vs = points @ first, points @ second
    centre_u = float((us.min() + us.max()) / 2.0)
    centre_v = float((vs.min() + vs.max()) / 2.0)
    half = float(max(us.max() - us.min(), vs.max() - vs.min()) / 2.0) * margin
    if not np.isfinite(half) or half <= 0.0:
        return None
    return centre_u, centre_v, half, side


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
) -> np.ndarray | None:
    """Warp the image to a fronto-parallel view of a plane at unit distance.

    The output spans ``[-extent, extent]`` in plane coordinates on both axes, so
    one output pixel is ``2 * extent / samples`` world units *at unit distance*.
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
    return cv2.remap(image, xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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
    step = (2.0 * extent) / max(samples - 1, 1)
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
    grey = patch if patch.ndim == 2 else patch.mean(axis=2)
    gradient = np.abs(np.diff(grey, axis=1 - along))
    if weights is None:
        return gradient.mean(axis=along)
    mask = weights if weights.ndim == 2 else weights.mean(axis=2)
    mask = mask[:, :-1] if along == 0 else mask[:-1, :]
    total = mask.sum(axis=along)
    covered = (gradient * mask).sum(axis=along)
    return np.divide(covered, total, out=np.zeros_like(covered),
                     where=total > 0)


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
    signal = _profile(patch, along, weights)
    signal = signal - signal.mean()
    if signal.size < 8 or float(signal.var()) < MINIMUM_CONTRAST:
        return None
    limit = maximum_period or signal.size / 3.0

    correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    low = max(int(np.floor(minimum_period)), 1)
    high = min(int(np.ceil(limit)), correlation.size - 1)
    if high <= low:
        return None

    # Peaks are found across the whole correlation and only then restricted to
    # the allowed range. Finding them inside the window instead hides any peak
    # sitting on its edge, so a period landing exactly on `minimum_period` was
    # invisible and the estimator fell through to its first harmonic - twice the
    # spacing, and so half the distance.
    interior = ((correlation[1:-1] >= correlation[:-2])
                & (correlation[1:-1] >= correlation[2:]))
    peaks = np.flatnonzero(interior) + 1
    peaks = peaks[(peaks >= low) & (peaks < high)]
    if peaks.size == 0:
        return None
    tallest = float(correlation[peaks].max())
    if tallest <= 0.0:
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

    return PeriodEstimate(period_pixels=period, phase_pixels=phase_pixels,
                          strength=strength, coarse_support=coarse,
                          block_supports=tuple(sorted(supports.items())))


def plane_period(
    patch: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    minimum_period: float = MINIMUM_PERIOD,
    tolerance: float = 0.12,
) -> PeriodEstimate | None:
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
    readings = [estimate_period(patch, along, weights=weights,
                                minimum_period=minimum_period)
                for along in (0, 1)]
    found = [r for r in readings if r is not None]
    if not found:
        return None
    if len(found) == 1:
        return found[0]

    short, long_ = sorted(found, key=lambda r: r.period_pixels)
    factor = long_.period_pixels / short.period_pixels
    nearest = round(factor)
    if nearest >= 2 and abs(factor - nearest) <= tolerance:
        return short
    return max(found, key=lambda r: r.strength)


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

    One output pixel spans ``2 * extent / samples`` world units at unit distance,
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
    per_pixel = 2.0 * extent / samples
    world = estimate.period_pixels * per_pixel
    if world <= 0.0:
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
