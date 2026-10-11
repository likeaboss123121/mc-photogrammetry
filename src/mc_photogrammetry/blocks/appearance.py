"""Neutral appearance records and face-scoring primitives.

This module owns the palette/state records and the low-level appearance
comparisons that used to live in the legacy reconstruction facade and raycast
modules. It deliberately depends only on the domain models and the block model
atlas, so an image-supported reconstruction path (M4/M5) can score exact block
states against face observations without importing the legacy facade or the
raycast implementation.

Nothing here decides geometry. A caller supplies an observed face colour,
descriptor and (optionally) a rectified texture pattern, and receives a
relative, capped cost per palette entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mc_photogrammetry.domain.models import BlockState
from .model_atlas import FACE_SHADE, FACE_SLOTS

CHROMA_WEIGHT = 0.75
"""Share of the appearance distance carried by hue rather than brightness.

Block lighting scales a face's brightness far more than it moves its hue, and
the texture term already carries the pattern, so hue is what colour is left to
contribute. Anything from 0.8 to 1.0 measures within a point of the best on
Template 2; the value is kept below 1.0 deliberately, because a scene built from
greys - stone, concrete, wool - has no hue to separate and needs the brightness
term that a weight of 1.0 throws away.
"""

MINIMUM_LIGHT = 1.0
"""Darkest a face may be assumed lit, as a fraction of its full-bright colour."""

AXIS_OF_SLOT = {"east": 0, "west": 0, "up": 1, "down": 1, "south": 2, "north": 2}


@runtime_checkable
class AppearanceCalibrationLike(Protocol):
    """Per-channel affine correction from texture colours to screenshot colours.

    Structural so this module never imports the reconstruction-side calibration
    record (which itself imports the palette type from here).
    """

    scale: tuple[float, float, float]
    offset: tuple[float, float, float]

    def apply(self, rgb: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class PaletteColor:
    state: BlockState
    rgb: tuple[int, int, int]
    face_rgbs: tuple[tuple[int, int, int], ...] = ()
    family: str = "other"
    descriptors: tuple[tuple[float, ...], ...] = ()
    face_map: tuple[tuple[str, tuple[int, int, int]], ...] = ()
    face_thumbnails: tuple[tuple[str, str], ...] = ()
    face_corners: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = ()
    model_faces: tuple[tuple, ...] = ()
    """Exact baked polygons: slot, corners, normal, cullface, optional alpha PNG."""
    full_cube: bool = True
    """Whether the block fills its cell. False for stairs, slabs, carpets, fences."""
    shape_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    """The model's boxes in cell coordinates, so a partial block keeps its shape."""
    covered_slots: tuple[str, ...] = ()
    """Which faces the shape actually fills, and so can hide what is behind them."""

    @property
    def occupies(self) -> float:
        """Share of the cell the shape fills; one for a full cube."""
        if self.full_cube or not self.shape_boxes:
            return 1.0
        return sum(
            max(0.0, high[0] - low[0]) * max(0.0, high[1] - low[1]) * max(0.0, high[2] - low[2])
            for low, high in self.shape_boxes
        )


DEFAULT_PALETTE = (
    PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),
    PaletteColor(BlockState("minecraft:dirt"), (134, 96, 67)),
    PaletteColor(BlockState("minecraft:oak_planks"), (162, 130, 79)),
    PaletteColor(BlockState("minecraft:grass_block"), (105, 150, 56)),
    PaletteColor(BlockState("minecraft:bricks"), (150, 75, 65)),
    PaletteColor(BlockState("minecraft:glass"), (170, 210, 220)),
    PaletteColor(BlockState("minecraft:white_concrete"), (207, 213, 214)),
    PaletteColor(BlockState("minecraft:black_concrete"), (8, 10, 15)),
)


def slot_coverage(item: PaletteColor, slot: str, tolerance: float = 1e-3) -> float:
    """How much of a cell face this block's shape hides, seen through that face.

    The quantity is the shape's **projected footprint** along the face's axis - the
    union of its boxes' cross sections - not whether it touches the cell wall.
    Wall contact was the first attempt and it is wrong in the case that matters: a
    carpet's top surface sits at y=0.0625, so it never touches the ceiling of its
    cell and scored zero on ``up``, the very face a camera sees it through. With the
    maximum shape penalty on its only visible face, a carpet could never win a cell,
    and residual proposals that should have become carpets came back as
    ``birch_planks``.

    By footprint a carpet seen from above covers all of it (1.0) and from the side a
    sixteenth; a bottom slab covers all of it from above and half from the side; a
    fence post a sixteenth from above and a quarter from the side. Direction does
    not matter, only the axis: a shape occludes the same cross section whichever
    way you look along it.

    This deliberately no longer separates a bottom slab from a full block seen from
    directly overhead - they really do hide the same area - so that call is left to
    the side faces and to depth, which is where the evidence actually is.
    """
    if item.full_cube or not item.shape_boxes:
        return 1.0
    axis = AXIS_OF_SLOT.get(slot)
    if axis is None:
        return 1.0
    others = [index for index in range(3) if index != axis]
    rectangles = [
        ((low[others[0]], low[others[1]]), (high[others[0]], high[others[1]]))
        for low, high in item.shape_boxes
    ]
    rectangles = [r for r in rectangles
                  if r[1][0] - r[0][0] > tolerance and r[1][1] - r[0][1] > tolerance]
    if not rectangles:
        return 0.0
    # Small boxes per model, so the union is cheapest over the grid their edges
    # induce rather than by a sweep.
    edges_u = sorted({v for r in rectangles for v in (r[0][0], r[1][0])})
    edges_v = sorted({v for r in rectangles for v in (r[0][1], r[1][1])})
    area = 0.0
    for u0, u1 in zip(edges_u, edges_u[1:]):
        for v0, v1 in zip(edges_v, edges_v[1:]):
            midpoint = ((u0 + u1) / 2.0, (v0 + v1) / 2.0)
            if any(r[0][0] <= midpoint[0] <= r[1][0] and r[0][1] <= midpoint[1] <= r[1][1]
                   for r in rectangles):
                area += (u1 - u0) * (v1 - v0)
    return min(1.0, max(0.0, area))


def palette_face_colors(
    palette: tuple[PaletteColor, ...],
) -> dict[BlockState, dict[str, np.ndarray]]:
    """Expected screen colour of each world face slot, per palette entry."""
    colors: dict[BlockState, dict[str, np.ndarray]] = {}
    for item in palette:
        mapping = {slot: np.asarray(color, dtype=np.float32) for slot, color in item.face_map}
        if not mapping:
            fallback = np.asarray(item.face_rgbs[0] if item.face_rgbs else item.rgb, dtype=np.float32)
            mapping = {slot: fallback for slot in FACE_SLOTS}
        default = next(iter(mapping.values()))
        for slot in FACE_SLOTS:
            mapping.setdefault(slot, default)
        # Vanilla darkens a face by its direction before any block light applies,
        # so an unshaded texture colour never matches what the screenshot shows.
        colors[item.state] = {
            slot: color * FACE_SHADE.get(slot, 1.0) for slot, color in mapping.items()
        }
    return colors


def normalised_pattern(patch: np.ndarray, coverage: np.ndarray | None = None) -> np.ndarray | None:
    """Zero-mean, unit-norm luminance of a face patch, or ``None`` if it is flat."""
    luminance = patch.mean(axis=2).reshape(-1)
    weights = None if coverage is None else coverage.reshape(-1)
    mean = float(np.average(luminance, weights=weights)) if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.mean())
    centred = luminance - mean
    if weights is not None:
        centred = centred * np.sqrt(np.clip(weights, 0.0, 1.0))
    norm = float(np.linalg.norm(centred))
    if norm < 1e-3:
        return None
    return (centred / norm).astype(np.float32)


def appearance_distance(
    observed: np.ndarray, expected: np.ndarray, *, minimum_light: float | None = None
) -> float:
    """Compare colour allowing for how dark Minecraft's block lighting can make a face.

    A texture is only ever darkened by lighting, never brightened past its own
    colour, so the expected colour may be scaled by a gain in
    ``[minimum_light, 1]`` before the comparison. Without that, bright blocks are
    punished for standing in shadow far more than dark ones: a birch floor in a
    dim corner is further from birch, in plain RGB, than dark oak is. Chroma is
    already gain free and carries the hue.
    """
    if minimum_light is None:
        minimum_light = MINIMUM_LIGHT
    expected_energy = float(expected @ expected)
    if expected_energy > 1e-6 and minimum_light < 1.0:
        gain = float(np.clip(float(observed @ expected) / expected_energy, minimum_light, 1.0))
    else:
        gain = 1.0
    rgb_distance = float(np.linalg.norm(observed - gain * expected))
    observed_total = float(observed.sum())
    expected_total = float(expected.sum())
    if observed_total <= 1e-6 or expected_total <= 1e-6:
        return rgb_distance
    observed_chroma = observed / observed_total
    expected_chroma = expected / expected_total
    chroma_distance = float(np.linalg.norm(observed_chroma - expected_chroma)) * 441.67
    return (1.0 - CHROMA_WEIGHT) * rgb_distance + CHROMA_WEIGHT * chroma_distance


def texture_pattern_distance(observed: np.ndarray, encoded: str) -> float:
    try:
        import base64
        from io import BytesIO
        from PIL import Image

        expected = np.asarray(Image.open(BytesIO(base64.b64decode(encoded))), dtype=np.float32).reshape(-1, 3)
    except (ValueError, OSError, TypeError):
        return float("inf")
    actual = observed.astype(np.float32).reshape(-1, 3)
    if actual.shape != expected.shape:
        return float("inf")
    actual_centered = actual - actual.mean(axis=0)
    expected_centered = expected - expected.mean(axis=0)
    actual_scale = np.maximum(actual_centered.std(axis=0), 1.0)
    expected_scale = np.maximum(expected_centered.std(axis=0), 1.0)
    return float(np.linalg.norm(actual_centered / actual_scale - expected_centered / expected_scale) / np.sqrt(actual.size))


def patch_descriptor(patch: np.ndarray, coverage: np.ndarray | None = None) -> tuple[float, ...]:
    pixels = patch.astype(np.float32).reshape(-1, 3)
    weights = None if coverage is None else coverage.reshape(-1)
    normalized = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.0)
    histogram = np.histogram(normalized[:, 0], bins=4, range=(0.0, 1.0), weights=weights, density=True)[0]
    luminance = pixels.mean(axis=1)
    average = float(np.average(luminance, weights=weights)) if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.mean())
    variance = (float(np.average((luminance - average) ** 2, weights=weights))
                if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.var()))
    if patch.ndim == 2:
        gradients = np.abs(np.diff(luminance)).mean() if len(luminance) > 1 else 0.0
    else:
        gradients = np.abs(np.diff(patch.astype(np.float32).mean(axis=2), axis=0)).mean()
    return tuple(float(value) for value in (*histogram, average / 255.0, np.sqrt(variance) / 255.0, gradients / 255.0))


def candidate_face_distance(
    rgb: np.ndarray,
    item: PaletteColor,
    appearance_calibration: AppearanceCalibrationLike | None = None,
    state_appearance_profiles: dict[BlockState, tuple[tuple[int, int, int], ...]] | None = None,
    observed_texture: np.ndarray | None = None,
) -> float:
    profiles = state_appearance_profiles.get(item.state, ()) if state_appearance_profiles else ()
    colors = profiles or tuple(color for _, color in item.face_map) or item.face_rgbs or (item.rgb,)
    if not colors:
        return float("inf")
    color_distance = min(
        appearance_distance(
            rgb,
            appearance_calibration.apply(np.asarray(color, dtype=np.float32))
            if appearance_calibration and not profiles
            else np.asarray(color, dtype=np.float32),
        )
        for color in colors
    )
    if observed_texture is None or not item.face_thumbnails:
        return color_distance
    texture_distance = min(
        texture_pattern_distance(observed_texture, encoded)
        for _, encoded in item.face_thumbnails
    )
    return color_distance + texture_distance * 120.0


def palette_candidates(
    rgb: np.ndarray,
    palette: tuple[PaletteColor, ...],
    appearance_calibration: AppearanceCalibrationLike | None = None,
    state_appearance_profiles: dict[BlockState, tuple[tuple[int, int, int], ...]] | None = None,
    family_priors: dict[str, float] | None = None,
    descriptor: tuple[float, ...] | None = None,
    *,
    limit: int,
    observed_texture: np.ndarray | None = None,
) -> list[tuple[BlockState, float, float]]:
    distances = []
    for item in palette:
        distance = candidate_face_distance(
            rgb,
            item,
            appearance_calibration=appearance_calibration,
            state_appearance_profiles=state_appearance_profiles,
            observed_texture=observed_texture,
        )
        if descriptor and item.descriptors:
            descriptor_distance = min(
                float(np.linalg.norm(np.asarray(descriptor) - np.asarray(candidate)))
                for candidate in item.descriptors
            )
            distance += descriptor_distance * 80.0
        if family_priors:
            distance /= max(0.1, family_priors.get(item.family, 1.0))
        distances.append(distance)
    order = sorted(range(len(distances)), key=distances.__getitem__)[:limit]
    return [
        (
            palette[index].state,
            distances[index],
            distances[order[1]] - distances[index] if len(order) > 1 else distances[index],
        )
        for index in order
    ]


__all__ = [
    "AXIS_OF_SLOT",
    "AppearanceCalibrationLike",
    "CHROMA_WEIGHT",
    "DEFAULT_PALETTE",
    "MINIMUM_LIGHT",
    "PaletteColor",
    "appearance_distance",
    "candidate_face_distance",
    "normalised_pattern",
    "palette_candidates",
    "palette_face_colors",
    "patch_descriptor",
    "slot_coverage",
    "texture_pattern_distance",
]
