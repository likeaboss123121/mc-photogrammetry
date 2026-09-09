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

Three outcomes, kept distinct, because collapsing them is how holes get filled:

* **supported** - the image shows the plane's own lattice at this cell.
* **contradicted** - the image shows something, and it does not agree.
* **unobserved** - the cell falls outside the frame, or too little of it was
  seen to judge. Not evidence of absence.

Bounds come from the supported cells, never from a reference extent. A local
coordinate frame is enough for one plane in one view; relating two views needs
evidence this module does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNOBSERVED = "unobserved"

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


def support_cells(
    patch: np.ndarray,
    coverage: np.ndarray,
    period_pixels: float,
    phase_pixels: tuple[float, float],
    *,
    minimum_coverage: float = MINIMUM_COVERAGE,
    tolerance: float = PHASE_TOLERANCE,
    threshold: float = SUPPORT_THRESHOLD,
) -> list[CellSupport]:
    """Judge every lattice cell of a rectified plane against its predicted seams.

    ``coverage`` marks which patch pixels came from inside the source frame, so a
    cell that simply fell outside the picture is reported unobserved rather than
    contradicted. The distinction matters: one is a hole in the wall and the
    other is a hole in the photograph.
    """
    if period_pixels < 2.0:
        return []
    across, down = _edge_profiles(patch)
    rows, columns = patch.shape[:2]
    found: list[CellSupport] = []

    cells_down = int(rows // period_pixels)
    cells_across = int(columns // period_pixels)
    for row in range(cells_down):
        for column in range(cells_across):
            v0, v1 = int(row * period_pixels), int((row + 1) * period_pixels)
            u0, u1 = int(column * period_pixels), int((column + 1) * period_pixels)
            if v1 <= v0 + 2 or u1 <= u0 + 2:
                continue
            seen = float(coverage[v0:v1, u0:u1].mean())
            if seen < minimum_coverage:
                found.append(CellSupport((row, column), UNOBSERVED, 0.0, seen))
                continue

            horizontal = across[v0:v1, u0:min(u1, across.shape[1])].mean(axis=0)
            vertical = down[v0:min(v1, down.shape[0]), u0:u1].mean(axis=1)
            if horizontal.size < 3 or vertical.size < 3:
                found.append(CellSupport((row, column), UNOBSERVED, 0.0, seen))
                continue

            score = 0.5 * (
                _phase_agreement(horizontal, np.arange(u0, u0 + horizontal.size),
                                 period_pixels, phase_pixels[0], tolerance)
                + _phase_agreement(vertical, np.arange(v0, v0 + vertical.size),
                                   period_pixels, phase_pixels[1], tolerance))
            verdict = SUPPORTED if score >= threshold else CONTRADICTED
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
    if found is None:
        return None
    ranked = period_hypotheses(found, extent=span, samples=samples)
    assisted = distance is not None
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
    # there and testing one sixteenth of it.
    across, down = _edge_profiles(patch)
    phase = (phase_at_period(across.mean(axis=0),
                             np.arange(across.shape[1]), block_pixels),
             phase_at_period(down.mean(axis=1),
                             np.arange(down.shape[0]), block_pixels))

    cells = support_cells(patch, coverage, block_pixels, phase)
    return SupportedPlane(axis=axis, distance=float(distance), side=float(side),
                          phase=phase, cells=tuple(cells), assisted=assisted,
                          notes=(note, f"block period {block_pixels:.2f}px, "
                                       f"lattice period {found.period_pixels:.2f}px"))
