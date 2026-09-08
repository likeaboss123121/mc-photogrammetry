"""Propose planar regions from one image, using only what the camera tells us.

`lattice` can read a plane's distance to a fraction of a percent, and could not
be used for anything, because it needed to be told which pixels belong to one
plane. Borrowing that from the reference litematic makes a demonstration, not a
pipeline stage. This finds the regions from the picture.

The cue is a negative one, and it is the useful part. A face perpendicular to a
world axis is *bounded and crossed* by edges running along the other two axes,
and contains no edge running along its own normal - an edge along the normal
points away from the face, so it can only be a silhouette or a depth boundary.
`calibration` already sorts every detected segment into one of three vanishing
directions. So the normal of a small patch of image is the direction that is
**missing** from it, which is a far sharper signal than asking which directions
are present: an edge running along axis *b* lies in the plane whose normal is
*a* and equally in the plane whose normal is *c*, so presence cannot separate
those two, and absence can.

The output is deliberately a set of *candidates* with their evidence, not a
segmentation. A region that is cleanly bounded but reads the wrong depth is
worse than no region at all, so the caller is expected to measure each one and
discard those that disagree with themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TILE = 48
"""Side of the square patch a normal is voted on, in source pixels.

Small enough that a tile rarely straddles two surfaces, large enough to catch
several block seams at the ranges these fixtures are shot at.
"""

MINIMUM_WEIGHT = 60.0
"""Total segment length a tile needs before its vote counts, in pixels.

A tile crossed by one short segment has an opinion about which direction is
missing, and it is worthless: two of the three directions are absent from it
simply because nothing was detected.
"""

MINIMUM_MARGIN = 0.35
"""How clearly the absent direction must be absent.

Measured as the shortfall of the least-represented direction against the median
of the other two, as a fraction of the tile's total weight. Without a margin
every tile votes, including the ones whose three counts are all noise.
"""

MINIMUM_TILES = 6
"""Tiles a region needs before it is worth measuring.

Below this the region is smaller than the lattice it is supposed to reveal.
"""


@dataclass(frozen=True)
class PlanarRegion:
    """A candidate plane: which way it faces, and which pixels voted for it."""

    axis: int
    tiles: tuple[tuple[int, int], ...]
    confidence: float
    tile_size: int = TILE

    @property
    def area_tiles(self) -> int:
        return len(self.tiles)

    def mask(self, height: int, width: int) -> np.ndarray:
        """The region as a pixel mask, for handing to `lattice.measure_plane`."""
        out = np.zeros((height, width), dtype=np.float32)
        for row, column in self.tiles:
            y0, x0 = row * self.tile_size, column * self.tile_size
            out[y0:y0 + self.tile_size, x0:x0 + self.tile_size] = 1.0
        return out

    def centroid(self) -> tuple[float, float]:
        rows = np.asarray([r for r, _c in self.tiles], dtype=float)
        columns = np.asarray([c for _r, c in self.tiles], dtype=float)
        half = self.tile_size / 2.0
        return (float(columns.mean() * self.tile_size + half),
                float(rows.mean() * self.tile_size + half))


def direction_weights(
    segments: np.ndarray,
    focal: float,
    yaw_degrees: float,
    pitch_degrees: float,
    width: int,
    height: int,
    *,
    tile: int = TILE,
    threshold: float = 3.0,
) -> np.ndarray:
    """Length of segment assigned to each lattice direction, per tile.

    Returns a ``(rows, columns, 3)`` array. A segment is credited to **every**
    tile it crosses, sharing its length between them, and not to the tile holding
    its midpoint: a seam running the width of a wall is evidence about that whole
    wall, and crediting it to one tile leaves the rest of a plainly legible
    surface with no evidence at all.
    """
    from mc_photogrammetry.reconstruction.calibration import (
        segment_residuals,
        vanishing_points,
    )

    rows = int(np.ceil(height / tile))
    columns = int(np.ceil(width / tile))
    out = np.zeros((rows, columns, 3), dtype=np.float64)
    if not len(segments):
        return out

    points = vanishing_points(focal, yaw_degrees, pitch_degrees, width, height)
    residuals = segment_residuals(segments, points)
    owner = residuals.argmin(axis=1)
    closest = residuals.min(axis=1)
    lengths = np.hypot(segments[:, 2] - segments[:, 0],
                       segments[:, 3] - segments[:, 1])
    step = max(tile / 2.0, 1.0)
    for index in np.flatnonzero(closest <= threshold):
        length = float(lengths[index])
        count = max(2, int(length / step) + 1)
        fractions = np.linspace(0.0, 1.0, count)
        xs = segments[index, 0] + fractions * (segments[index, 2] - segments[index, 0])
        ys = segments[index, 1] + fractions * (segments[index, 3] - segments[index, 1])
        tile_row = np.clip((ys // tile).astype(int), 0, rows - 1)
        tile_column = np.clip((xs // tile).astype(int), 0, columns - 1)
        np.add.at(out, (tile_row, tile_column, int(owner[index])), length / count)
    return out


def normal_votes(
    weights: np.ndarray,
    *,
    minimum_weight: float = MINIMUM_WEIGHT,
    minimum_margin: float = MINIMUM_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Each tile's likely plane normal, and how strongly it is held.

    The normal is the direction that is *missing*. Presence cannot distinguish a
    wall from a floor, because an edge running east lies in both; absence can,
    because only a wall facing east lacks eastward edges.

    Tiles that fail either test return ``-1``, which is not a defect to be tuned
    away: most of an image is not a legible plane, and saying so is the point.
    """
    total = weights.sum(axis=2)
    ordered = np.sort(weights, axis=2)
    smallest, middle = ordered[..., 0], ordered[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(total > 0, (middle - smallest) / np.maximum(total, 1e-9), 0.0)
    axis = weights.argmin(axis=2)
    ok = (total >= minimum_weight) & (margin >= minimum_margin)
    return np.where(ok, axis, -1), np.where(ok, margin, 0.0)


def _grow(axis_map: np.ndarray, start: tuple[int, int], seen: np.ndarray):
    """Four-connected flood over tiles sharing a normal."""
    rows, columns = axis_map.shape
    wanted = axis_map[start]
    stack, found = [start], []
    seen[start] = True
    while stack:
        row, column = stack.pop()
        found.append((row, column))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, column + dc
            if 0 <= r < rows and 0 <= c < columns and not seen[r, c] \
                    and axis_map[r, c] == wanted:
                seen[r, c] = True
                stack.append((r, c))
    return found


def planar_regions(
    segments: np.ndarray,
    focal: float,
    yaw_degrees: float,
    pitch_degrees: float,
    width: int,
    height: int,
    *,
    tile: int = TILE,
    minimum_tiles: int = MINIMUM_TILES,
    minimum_weight: float = MINIMUM_WEIGHT,
    minimum_margin: float = MINIMUM_MARGIN,
) -> list[PlanarRegion]:
    """Connected groups of tiles that agree on which way their surface faces.

    Grouping is four-connected over tiles alone, so two coplanar walls separated
    by a doorway come back as two regions. That is the right way round: merging
    them would need a depth they have not been measured at yet, and a region that
    silently spans a discontinuity reads a period that belongs to neither half.
    """
    weights = direction_weights(segments, focal, yaw_degrees, pitch_degrees,
                                width, height, tile=tile)
    axis_map, margin = normal_votes(weights, minimum_weight=minimum_weight,
                                    minimum_margin=minimum_margin)
    seen = np.zeros(axis_map.shape, dtype=bool)
    regions: list[PlanarRegion] = []
    for row in range(axis_map.shape[0]):
        for column in range(axis_map.shape[1]):
            if seen[row, column] or axis_map[row, column] < 0:
                continue
            found = _grow(axis_map, (row, column), seen)
            if len(found) < minimum_tiles:
                continue
            confidence = float(np.mean([margin[r, c] for r, c in found]))
            regions.append(PlanarRegion(axis=int(axis_map[row, column]),
                                        tiles=tuple(sorted(found)),
                                        confidence=confidence, tile_size=tile))
    regions.sort(key=lambda region: -region.area_tiles)
    return regions
