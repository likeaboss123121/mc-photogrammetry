"""Which cells of a plane the picture actually supports.

`lattice` measures where a plane is. That says nothing about how much of it is
*there*: a wall with a doorway is one plane and not one rectangle, and a
reconstruction that fills the doorway because the wall's extent looked
rectangular has invented geometry from an assumption rather than from evidence.

So the plane is divided into its own lattice cells and each is asked separately
whether the image supports a face at it. The evidence is phase agreement. Once
the distance and phase are known the block seams are *predicted*, at specific
positions, and a cell that genuinely lies on the plane shows its edges there. A
cell looking through a doorway at something farther away shows edges too, but at
the wrong spacing and the wrong offset, because that surface is at a different
depth and its seams project differently.

Five outcomes, kept distinct, because collapsing any pair of them is how holes
get filled or real evidence gets thrown away:

* **supported** - the image shows the plane's own lattice at this cell.
* **contradicted** - the image shows real structure here, at odds with the
  plane: most often a farther surface seen past an opening, whose own texture
  reads finer than the plane's blocks once rectified as if it were at the
  plane's distance.
* **weak_evidence** - the cell has too little edge energy, of any kind, to
  judge either way. A cell this reports is not "not a block" - a plain wool or
  concrete face is genuinely flat and would score exactly the same. Collapsing
  this into `contradicted` was a real bug: a uniform 64x64 test plane with
  *no* seam evidence anywhere scored every one of its 16 cells contradicted,
  which is not what "the image shows something that disagrees" is supposed to
  mean.
* **foreground_occlusion** - the cell shows real structure whose own scale
  reads coarser than the plane's blocks once rectified at the plane's
  distance, which is what a nearer object looks like under a rectification
  that assumes it is farther away than it is.
* **outside_image** - the cell falls outside the source photograph. Not
  evidence of absence, and not the same claim as `weak_evidence`: one is a
  hole in the picture, the other is a hole in the picture's information about
  a location it did capture.

Bounds come from the supported cells, never from a reference extent. A local
coordinate frame is enough for one plane in one view; relating two views needs
evidence this module does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
WEAK_EVIDENCE = "weak_evidence"
FOREGROUND_OCCLUSION = "foreground_occlusion"
OUTSIDE_IMAGE = "outside_image"
UNOBSERVED = OUTSIDE_IMAGE
"""Old name, kept as an alias: every caller written against it still reads
cells this module now calls `outside_image` (a plain coverage judgement, not a
merger of `outside_image` and `weak_evidence` - those stay genuinely distinct)."""

MINIMUM_COVERAGE = 0.6
"""Fraction of a cell that must fall inside the frame before it is judged."""

PHASE_TOLERANCE = 0.18
"""How near a predicted seam an edge must fall, as a fraction of the period.

Wide enough to absorb the rectification's own resampling, narrow enough that a
surface at another depth does not drift into agreement by chance.
"""

SUPPORT_THRESHOLD = 0.55
"""Share of a cell's edge energy that must sit on predicted seams.

At `PHASE_TOLERANCE` of 0.18 the predicted bands cover about 36% of each cell, so
edges scattered with no regard to the grid score **about 0.36 by chance** -
measured at 0.34 to 0.36 on random noise. The threshold has to clear that with
room to spare, not sit near it; an earlier value of 0.34 was below chance, so
pure noise was classified as a supported wall.

A patch genuinely on its own lattice puts nearly all its edge energy on the
predicted lines and scores close to 1. 0.55 sits between the two and well clear
of chance.
"""

WEAK_EVIDENCE_RATIO = 0.15
"""A cell's own edge energy density, as a share of the patch's typical density,
below which there is too little signal to judge the cell at all.

The reference is the *median* per-pixel edge-energy density over every cell
with enough coverage to judge, computed once per plane - not a fixed absolute
brightness-gradient value, because screenshots vary too much in contrast and
JPEG-like artefacting for an absolute number to mean the same thing twice.
0.15 was chosen so a genuinely flat block face (wool, concrete, quartz - real,
common Minecraft materials) reads as weak rather than contradicted, while
still leaving room for the faint, real edges a photographed face legitimately
carries under compression.
"""

WEAK_EVIDENCE_FLOOR = 1e-6
"""An absolute edge-energy-density floor, for when the *whole* plane is flat.

`WEAK_EVIDENCE_RATIO` compares a cell against the plane's own median density,
which is meaningless when every cell is uniform and that median is itself
zero - a plane entirely covered in one flat material, which is a real case
(a wool or concrete wall), not a degenerate one. This floor catches it: a
cell below it is weak regardless of what the rest of the plane looks like.
"""

FOREGROUND_PERIOD_RATIO = 1.6
"""A cell's own dominant local period, as a multiple of the plane's predicted
block period, above which the cell reads as a nearer surface rather than a
farther one.

A rectification assumes every point sits at the plane's distance. A surface
that is actually *nearer* subtends a larger angle per real-world unit than
that assumption, so its own texture appears magnified - coarser, with a larger
period - once rectified. A surface that is actually *farther*, seen past an
opening, subtends a smaller angle and appears demagnified - finer, with a
smaller period. The ratio is symmetric around 1.0 in log-space by construction
(`1 / FOREGROUND_PERIOD_RATIO` is the matching farther-surface threshold), and
1.6 was chosen to sit clear of the noise in period estimation from a single
cell's own limited pixel count, which is small.
"""


@dataclass(frozen=True)
class CellSupport:
    """One lattice cell of a plane, and what the image said about it."""

    cell: tuple[int, int]
    verdict: str
    score: float
    coverage: float


@dataclass(frozen=True)
class SupportedPlane:
    """A plane, the cells the image supports on it, and the ones it does not."""

    axis: int
    distance: float
    phase: tuple[float, float]
    side: float = 1.0
    cells: tuple[CellSupport, ...] = field(default_factory=tuple)
    assisted: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def of(self, verdict: str) -> tuple[CellSupport, ...]:
        return tuple(cell for cell in self.cells if cell.verdict == verdict)

    @property
    def supported(self) -> tuple[CellSupport, ...]:
        return self.of(SUPPORTED)

    def bounds(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """The extent of the *supported* cells, which is the only honest extent.

        Deriving it from anything else - a reference region, the frame, the
        rectified window - states a size the evidence does not.
        """
        found = self.supported
        if not found:
            return None
        rows = [cell.cell[0] for cell in found]
        columns = [cell.cell[1] for cell in found]
        return (min(rows), min(columns)), (max(rows), max(columns))

    def holes(self) -> tuple[CellSupport, ...]:
        """Contradicted cells inside the supported extent: doorways and windows."""
        extent = self.bounds()
        if extent is None:
            return ()
        (low_row, low_column), (high_row, high_column) = extent
        return tuple(cell for cell in self.of(CONTRADICTED)
                     if low_row <= cell.cell[0] <= high_row
                     and low_column <= cell.cell[1] <= high_column)


def _edge_profiles(patch: np.ndarray):
    """Absolute gradient along each in-plane direction of a rectified patch."""
    grey = patch if patch.ndim == 2 else patch.mean(axis=2)
    across = np.abs(np.diff(grey, axis=1))
    down = np.abs(np.diff(grey, axis=0))
    return across, down


def phase_at_period(profile: np.ndarray, offsets: np.ndarray,
                    period: float) -> float:
    """Where the seams of a lattice of this period actually sit in the image.

    The period comes from the distance; the phase does not follow from it and has
    to be read. It is measured **once for the whole plane**, deliberately: a phase
    fitted per cell would let every cell find an offset that suits it, which is a
    licence to agree with anything and exactly the invention this module exists to
    avoid. One plane has one lattice.

    Fitting the phase does raise what noise scores, since a fitted offset beats a
    fixed one. `SUPPORT_THRESHOLD` is set against the fitted-phase chance level
    for that reason, not the fixed-phase one.

    The estimate is the circular mean of the edge energy at this period - the
    angle of its first Fourier coefficient. Scanning offsets for the one with the
    most energy inside the tolerance band looks equivalent and is not: wherever
    the band can shift without losing energy the scan returns the *edge* of that
    plateau rather than its centre, putting the predicted seams half a tolerance
    off the real ones. The circular mean has no such bias.
    """
    total = float(profile.sum())
    if total <= 1e-9 or period < 2.0:
        return 0.0
    angle = 2.0 * np.pi * np.asarray(offsets, dtype=float) / period
    response = np.sum(profile * np.exp(-1j * angle))
    if abs(response) <= 1e-12:
        return 0.0
    return float((-np.angle(response) / (2.0 * np.pi) * period) % period)


def _phase_agreement(profile: np.ndarray, offsets: np.ndarray, period: float,
                     phase: float, tolerance: float) -> float:
    """Share of edge energy sitting within tolerance of a predicted seam."""
    total = float(profile.sum())
    if total <= 1e-9:
        return 0.0
    distance = np.abs(((offsets - phase + period / 2.0) % period) - period / 2.0)
    near = distance <= tolerance * period
    return float(profile[near].sum() / total)


def _dominant_period(profile: np.ndarray, low: float, high: float) -> float | None:
    """The strongest autocorrelation lag of a 1-D edge profile within a range.

    Used only once a cell has real edge energy that does not match the plane's
    own predicted seams, to ask what scale that energy *is* at - not to predict
    seam positions, so no phase is involved. `None` means no clear peak: the
    energy present does not read as periodic at any scale in range, which is
    itself useful (it rules out a confident foreground/farther-surface call and
    leaves the cell `contradicted` by default rather than guessing which).
    """
    centred = profile - profile.mean()
    if not np.any(centred):
        return None
    full = np.correlate(centred, centred, mode="full")
    auto = full[full.size // 2:]
    denom = auto[0]
    if denom <= 1e-9:
        return None
    auto = auto / denom
    lo, hi = max(1, int(low)), min(auto.size - 1, int(high))
    if hi <= lo:
        return None
    peak = lo + int(np.argmax(auto[lo:hi]))
    if auto[peak] <= 0.1:
        return None
    return float(peak)


def support_cells(
    patch: np.ndarray,
    coverage: np.ndarray,
    period_pixels: float,
    phase_pixels: tuple[float, float],
    *,
    minimum_coverage: float = MINIMUM_COVERAGE,
    tolerance: float = PHASE_TOLERANCE,
    threshold: float = SUPPORT_THRESHOLD,
    local_window_periods: float | None = None,
) -> list[CellSupport]:
    """Judge every lattice cell of a rectified plane against its predicted seams.

    ``coverage`` marks which patch pixels came from inside the source frame, so a
    cell that simply fell outside the picture is reported unobserved rather than
    contradicted. The distinction matters: one is a hole in the wall and the
    other is a hole in the photograph.

    ``phase_pixels`` is a single, plane-wide phase by default. That assumes the
    predicted seams stay in step with the real ones everywhere on the plane, which
    measured false on a real screenshot: autocorrelation at the block period was
    0.015 (noise) over an 18-block-wide patch and 0.12-0.29 (real signal) inside
    2.5-block windows of the same patch. A plane's calibration is accurate enough
    locally that its own small residual error still compounds linearly with
    distance from wherever the phase was anchored, and over a wide-enough patch
    that drift exceeds the tolerance band and desyncs the global prediction from
    the real lattice - the true grid is there, the *global* prediction of where it
    sits is not.

    ``local_window_periods`` re-reads the phase for each cell from a window this
    many periods wider than the cell on every side, rather than trusting the one
    passed in for the whole plane. The window has to stay wider than the cell
    itself: fitting a cell's phase from only its own pixels agrees with anything,
    which is the invention this module exists to prevent. Passing `None` keeps the
    original single global phase, which the tests still exercise directly.

    A cell that scores below ``threshold`` is not automatically `contradicted`.
    Three things can produce a low score, and they mean different things: too
    little edge energy anywhere to judge (`weak_evidence`), real energy at a
    scale that reads farther than the plane (`contradicted` - an opening), and
    real energy at a scale that reads nearer than the plane
    (`foreground_occlusion`). Collapsing these was a real bug: a uniform,
    textureless plane - a plain wool or concrete wall, not a hypothetical -
    scored every cell `contradicted` before this distinction existed, which
    reports a flat material as active evidence against itself.
    """
    if period_pixels < 2.0:
        return []
    across, down = _edge_profiles(patch)
    rows, columns = patch.shape[:2]
    found: list[CellSupport] = []
    margin = (int(local_window_periods * period_pixels)
              if local_window_periods else 0)

    cells_down = int(rows // period_pixels)
    cells_across = int(columns // period_pixels)

    # The reference for "how much edge energy is typical here" - computed once,
    # from every cell with enough coverage to judge, so a cell is compared
    # against this plane's own contrast rather than an absolute number that
    # means something different on every screenshot.
    densities = []
    for row in range(cells_down):
        for column in range(cells_across):
            v0, v1 = int(row * period_pixels), int((row + 1) * period_pixels)
            u0, u1 = int(column * period_pixels), int((column + 1) * period_pixels)
            if v1 <= v0 + 2 or u1 <= u0 + 2:
                continue
            if coverage[v0:v1, u0:u1].mean() < minimum_coverage:
                continue
            area = (v1 - v0) * (u1 - u0)
            energy = (across[v0:v1, u0:min(u1, across.shape[1])].sum()
                      + down[v0:min(v1, down.shape[0]), u0:u1].sum())
            densities.append(energy / max(area, 1))
    reference_density = float(np.median(densities)) if densities else 0.0

    for row in range(cells_down):
        for column in range(cells_across):
            v0, v1 = int(row * period_pixels), int((row + 1) * period_pixels)
            u0, u1 = int(column * period_pixels), int((column + 1) * period_pixels)
            if v1 <= v0 + 2 or u1 <= u0 + 2:
                continue
            seen = float(coverage[v0:v1, u0:u1].mean())
            if seen < minimum_coverage:
                found.append(CellSupport((row, column), OUTSIDE_IMAGE, 0.0, seen))
                continue

            horizontal = across[v0:v1, u0:min(u1, across.shape[1])].mean(axis=0)
            vertical = down[v0:min(v1, down.shape[0]), u0:u1].mean(axis=1)
            if horizontal.size < 3 or vertical.size < 3:
                found.append(CellSupport((row, column), OUTSIDE_IMAGE, 0.0, seen))
                continue

            area = (v1 - v0) * (u1 - u0)
            energy = (across[v0:v1, u0:min(u1, across.shape[1])].sum()
                      + down[v0:min(v1, down.shape[0]), u0:u1].sum())
            density = energy / max(area, 1)
            if (density < WEAK_EVIDENCE_FLOOR
                    or (reference_density > 0.0
                        and density < WEAK_EVIDENCE_RATIO * reference_density)):
                found.append(CellSupport((row, column), WEAK_EVIDENCE, 0.0, seen))
                continue

            phase = phase_pixels
            if local_window_periods:
                wu0, wu1 = max(0, u0 - margin), min(across.shape[1], u1 + margin)
                wv0, wv1 = max(0, v0 - margin), min(down.shape[0], v1 + margin)
                window_h = across[v0:v1, wu0:wu1].mean(axis=0)
                window_v = down[wv0:wv1, u0:u1].mean(axis=1)
                phase = (phase_at_period(window_h, np.arange(wu0, wu0 + window_h.size),
                                         period_pixels)
                         if window_h.size >= 3 else phase_pixels[0],
                         phase_at_period(window_v, np.arange(wv0, wv0 + window_v.size),
                                         period_pixels)
                         if window_v.size >= 3 else phase_pixels[1])

            score = 0.5 * (
                _phase_agreement(horizontal, np.arange(u0, u0 + horizontal.size),
                                 period_pixels, phase[0], tolerance)
                + _phase_agreement(vertical, np.arange(v0, v0 + vertical.size),
                                   period_pixels, phase[1], tolerance))

            if score >= threshold:
                found.append(CellSupport((row, column), SUPPORTED, score, seen))
                continue

            # Real energy, wrong seams: ask what scale it actually is, not
            # whether it matches the prediction, to tell an opening (farther,
            # finer once rectified at this distance) from an occluder (nearer,
            # coarser). A cell's own profile is only `period_pixels` samples
            # long, which cannot represent a lag anywhere near
            # FOREGROUND_PERIOD_RATIO times that - autocorrelation has nothing
            # to find a coarser period *in*. So this reads from a window wide
            # enough to contain one, centred on the cell, independent of
            # whether local phase windowing is on.
            occlusion_margin = int(period_pixels * FOREGROUND_PERIOD_RATIO * 1.5)
            ou0, ou1 = max(0, u0 - occlusion_margin), min(across.shape[1], u1 + occlusion_margin)
            ov0, ov1 = max(0, v0 - occlusion_margin), min(down.shape[0], v1 + occlusion_margin)
            wide_h = across[v0:v1, ou0:ou1].mean(axis=0)
            wide_v = down[ov0:ov1, u0:u1].mean(axis=1)
            local_low, local_high = period_pixels * 0.15, period_pixels * 3.0
            local_h = _dominant_period(wide_h, local_low, local_high)
            local_v = _dominant_period(wide_v, local_low, local_high)
            candidates = [p for p in (local_h, local_v) if p is not None]
            verdict = CONTRADICTED
            if candidates:
                local_period = float(np.mean(candidates))
                ratio = local_period / period_pixels
                if ratio >= FOREGROUND_PERIOD_RATIO:
                    verdict = FOREGROUND_OCCLUSION
            found.append(CellSupport((row, column), verdict, score, seen))
    return found


def plane_from_evidence(
    image: np.ndarray,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    image_size: tuple[int, int],
    *,
    distance: float | None = None,
    mask: np.ndarray | None = None,
    extent: float = 1.0,
    principal_point: tuple[float, float] | None = None,
    side: float = 1.0,
    local_window_periods: float | None = 1.5,
) -> SupportedPlane | None:
    """Measure a plane and judge its cells, optionally with a stated distance.

    With ``distance`` supplied - from a user's stated length - the scale is known
    and only the phase has to be read. Without it the distance comes from the
    lattice period, which carries the block-versus-texel ambiguity and can be
    wrong by a factor of sixteen while looking entirely reasonable. The two paths
    are reported separately for that reason and should not be averaged.

    ``side`` picks which of the two planes perpendicular to the axis is meant -
    the floor below the camera or the ceiling above it. Choosing a surface is a
    framing decision and not a measurement, so when the two paths are compared
    they must be given the same one; a mask, where there is one, settles it
    instead.

    ``local_window_periods`` reads each cell's phase from its own neighbourhood
    rather than one phase for the whole plane; see `support_cells`. 1.5 default
    matches the window size (~2.5 blocks) that showed real block-period
    autocorrelation (0.12-0.29) where the plane-wide phase was at chance (0.015)
    on real data. Pass `None` for the old plane-wide behaviour.
    """
    from mc_photogrammetry.reconstruction.lattice import (
        measure_plane,
        period_hypotheses,
        rectify_plane,
        plane_window,
    )

    window = plane_window(mask, axis, axis_matrix, focal,
                          principal_point=principal_point) if mask is not None else None
    centre = (window[0], window[1]) if window else (0.0, 0.0)
    span = window[2] if window else extent
    side = window[3] if window else side

    found, samples, span = measure_plane(image, axis, axis_matrix, focal,
                                         mask=mask, extent=span, side=side)
    assisted = distance is not None
    if found is None:
        # Automatic mode has nothing to read a distance from at all if the
        # lattice period was never found - genuinely unresolvable. Assisted
        # mode already has its distance from the caller; period detection
        # failing (a low-texture surface, a bad sampling round) should not
        # throw away evidence the caller already supplied. `samples`/`span`
        # are still the sampling this round used, even without a found period,
        # so rectification below proceeds at that resolution rather than none.
        if not assisted:
            return None
        note = f"distance {distance:.3f} supplied by a stated length"
    else:
        ranked = period_hypotheses(found, extent=span, samples=samples)
        if distance is None:
            best = ranked.best
            if best is None:
                return None
            distance = best.plane_distance
            note = (f"distance {distance:.3f} read from the lattice period as "
                    f"'{best.interpretation}'; the alternative reading is retained "
                    f"and differs by a factor of sixteen")
        else:
            note = f"distance {distance:.3f} supplied by a stated length"

    patch = rectify_plane(image, axis, axis_matrix, focal, extent=span,
                          samples=samples, centre=centre, side=side,
                          principal_point=principal_point)
    if patch is None:
        return None
    # Which output pixels came from inside the photograph. Warping a solid frame
    # through the same transform answers it exactly. Assuming full coverage
    # instead - the obvious shortcut when no mask is given - makes every cell the
    # picture never saw score zero energy and read as *contradicted*, turning a
    # hole in the photograph into a hole in the wall.
    inside = rectify_plane(np.ones(image.shape[:2], dtype=np.float32), axis,
                           axis_matrix, focal, extent=span, samples=samples,
                           centre=centre, side=side,
                           principal_point=principal_point)
    coverage = (np.ones(patch.shape[:2], dtype=float) if inside is None
                else np.asarray(inside, dtype=float))
    if coverage.ndim == 3:
        coverage = coverage.mean(axis=2)
    if mask is not None:
        seen = rectify_plane(mask, axis, axis_matrix, focal, extent=span,
                             samples=samples, centre=centre, side=side,
                             principal_point=principal_point)
        if seen is not None:
            seen = np.asarray(seen, dtype=float)
            if seen.ndim == 3:
                seen = seen.mean(axis=2)
            coverage = coverage * (seen > 0.5)

    # One block, in rectified pixels, at the distance now believed.
    per_pixel = 2.0 * span / samples
    block_pixels = (1.0 / distance) / per_pixel
    if block_pixels < 2.0:
        return None

    # The measured lattice's phase belongs to the measured period, which is
    # usually the texel period and a different modulus entirely. Reading it at
    # the block period instead is the difference between testing the grid that is
    # there and testing one sixteenth of it. This global reading is still used as
    # the fallback phase for cells whose local window (below) has too little
    # data of its own.
    across, down = _edge_profiles(patch)
    phase = (phase_at_period(across.mean(axis=0),
                             np.arange(across.shape[1]), block_pixels),
             phase_at_period(down.mean(axis=1),
                             np.arange(down.shape[0]), block_pixels))

    # A single global phase assumes the predicted seams stay in step with the
    # real ones across the whole plane. Measured false: on an 18-block-wide real
    # patch, block-period autocorrelation was 0.015 (noise) globally and
    # 0.12-0.29 (real signal) inside 2.5-block windows of the same patch, because
    # calibration's own small residual error compounds linearly with distance
    # from wherever the phase was anchored. Reading the phase locally keeps each
    # cell's test anchored to nearby evidence instead of a plane-wide average.
    cells = support_cells(patch, coverage, block_pixels, phase,
                          local_window_periods=local_window_periods)
    period_note = (f"lattice period {found.period_pixels:.2f}px" if found is not None
                   else "no lattice period detected; distance came from the "
                        "caller, not from measurement")
    return SupportedPlane(axis=axis, distance=float(distance), side=float(side),
                          phase=phase, cells=tuple(cells), assisted=assisted,
                          notes=(note, f"block period {block_pixels:.2f}px, "
                                       f"{period_note}"))
