"""Simulate Minecraft's block lighting over a reconstructed volume.

A block face is not rendered at its texture colour: it is scaled by how much
light reaches the air in front of it. That scaling is large - a face two blocks
further from a torch is visibly darker - and it moves pale materials much further
than dark ones, which is why an unlit comparison mistakes a shaded birch floor for
a darker wood. Given the blocks the reconstruction has already decided on, the
light that produced the screenshot can be recomputed the way the game does it:
flood fill outward from each light source, losing one level per block, stopping at
anything opaque.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState

MAXIMUM_LIGHT = 15

# Vanilla emission levels, keyed by the part of the block name that identifies it.
# Longer keys are matched first, so "soul_lantern" beats "lantern".
BLOCK_LIGHT: dict[str, int] = {
    "beacon": 15,
    "campfire": 15,
    "conduit": 15,
    "end_rod": 14,
    "fire": 15,
    "froglight": 15,
    "glowstone": 15,
    "jack_o_lantern": 15,
    "lava": 15,
    "lantern": 15,
    "sea_lantern": 15,
    "shroomlight": 15,
    "torch": 14,
    "soul_campfire": 10,
    "soul_fire": 10,
    "soul_lantern": 10,
    "soul_torch": 10,
    "crying_obsidian": 10,
    "redstone_torch": 7,
    "ender_chest": 7,
    "glow_lichen": 7,
    "sculk_catalyst": 6,
    "amethyst_cluster": 5,
    "large_amethyst_bud": 4,
    "medium_amethyst_bud": 2,
    "small_amethyst_bud": 1,
    "magma_block": 3,
    "brewing_stand": 1,
    "brown_mushroom": 1,
    "trial_spawner": 4,
    "vault": 6,
    "cave_vines": 14,
    "candle": 3,
    "light": 15,
}

# Copper bulbs dim as they oxidise, and only while lit. Ordered most oxidised
# first, because every one of these names also contains "copper_bulb".
COPPER_BULB_LIGHT = (("oxidized", 4), ("weathered", 8), ("exposed", 12))
COPPER_BULB_UNOXIDISED = 15

# Blocks that hold their place in the grid without stopping light.
TRANSPARENT_TOKENS = (
    "air", "torch", "glass", "lantern", "candle", "rod", "fire", "vines", "lichen",
    "cluster", "bud", "chain", "ladder", "rail", "button", "lever", "pressure_plate",
    "sign", "banner", "flower", "sapling", "mushroom", "web", "carpet", "snow",
    "barrier", "light", "water", "bars", "pane", "trapdoor", "door", "fence", "wall",
)


def emission_of(state: BlockState) -> int:
    """Light level a block state emits, 0 for anything that does not glow."""
    name = state.name.rsplit(":", 1)[-1]
    properties = dict(state.properties)
    if "copper_bulb" in name:
        if properties.get("lit", "false") != "true":
            return 0
        for token, level in COPPER_BULB_LIGHT:
            if token in name:
                return level
        return COPPER_BULB_UNOXIDISED
    if name.endswith("redstone_lamp") or name == "redstone_lamp":
        return 15 if properties.get("lit", "false") == "true" else 0
    if "candle" in name:
        count = int(properties.get("candles", "1") or 1)
        return 0 if properties.get("lit", "true") != "true" else 3 * max(1, count)
    if "respawn_anchor" in name:
        charges = int(properties.get("charges", "0") or 0)
        return max(0, charges * 4 - 2)
    for token in sorted(BLOCK_LIGHT, key=len, reverse=True):
        if token in name:
            return BLOCK_LIGHT[token]
    return 0


def blocks_light(state: BlockState) -> bool:
    """Whether a block stops light travelling through its cell."""
    name = state.name.rsplit(":", 1)[-1]
    if emission_of(state) > 0 and any(token in name for token in ("torch", "lantern", "candle")):
        return False
    return not any(token in name for token in TRANSPARENT_TOKENS)


def brightness(level: int, *, ambient: float = 0.0) -> float:
    """Minecraft's light curve: the factor a face's colour is multiplied by.

    Matches ``LightTexture``: ``f / (4 - 3f)`` for ``f = level / 15``, which falls
    away steeply - level 14 is already down to 0.78, level 10 to 0.33.
    """
    fraction = max(0.0, min(1.0, level / MAXIMUM_LIGHT))
    curve = fraction / (4.0 - 3.0 * fraction)
    return ambient + (1.0 - ambient) * curve


@dataclass(frozen=True, slots=True)
class LightResponse:
    """How a simulated light level maps onto observed brightness.

    The default is deliberately flat at 1.0: a light field nobody has fitted a
    response for must not dim anything, or a scene with no detected light source
    would have every expected colour driven to black.

    The client's brightness slider and gamma sit between the game's light level
    and the pixels in a screenshot, so the curve is fitted rather than assumed.
    ``floor`` is what an unlit face still shows; ``scale`` is what a fully lit one
    reaches.
    """

    floor: float = 1.0
    scale: float = 1.0
    confidence: float = 0.0
    """How well simulated level actually predicted the observed brightness."""

    def factor(self, level: int) -> float:
        return self.floor + (self.scale - self.floor) * brightness(level)


@dataclass(frozen=True, slots=True)
class LightField:
    """Per-cell block light over a scene-local grid."""

    levels: np.ndarray
    response: LightResponse = LightResponse()

    @property
    def size(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.levels.shape)

    def level_at(self, coordinate: BlockCoordinate) -> int:
        if not all(0 <= coordinate[axis] < self.levels.shape[axis] for axis in range(3)):
            return 0
        return int(self.levels[coordinate])

    def face_level(self, coordinate: BlockCoordinate, slot: str) -> int:
        """Light reaching a face, which is the light in the cell it faces into."""
        normal = FACE_BASIS[slot][0]
        return self.level_at(
            (
                coordinate[0] + int(normal[0]),
                coordinate[1] + int(normal[1]),
                coordinate[2] + int(normal[2]),
            )
        )

    def face_factor(self, coordinate: BlockCoordinate, slot: str) -> float:
        return self.response.factor(self.face_level(coordinate, slot))

    def with_response(self, response: LightResponse) -> LightField:
        return LightField(self.levels, response)


def ambient_occlusion(
    coordinate: BlockCoordinate, slot: str, occupied: set[BlockCoordinate]
) -> float:
    """How much the surrounding blocks darken a face, as vanilla smooth lighting does.

    Each of the face's four corners is dimmed by the blocks meeting it just
    outside the face: two edge neighbours and the diagonal between them. Two
    edges together close the corner off completely. This is what makes a ceiling
    or an inside corner read darker than an open wall of the same material, and
    without it a pale block tucked into a corner looks like a darker material.
    """
    normal = tuple(int(value) for value in FACE_BASIS[slot][0])
    axis = next(index for index, value in enumerate(normal) if value)
    plane = [index for index in range(3) if index != axis]
    total = 0.0
    for first in (-1, 1):
        for second in (-1, 1):
            offsets = []
            for weights in ((first, 0), (0, second), (first, second)):
                cell = list(coordinate)
                for index, value in zip(plane, weights):
                    cell[index] += value
                for index in range(3):
                    cell[index] += normal[index]
                offsets.append(tuple(cell) in occupied)
            side_one, side_two, corner = offsets
            level = 0 if (side_one and side_two) else 3 - (side_one + side_two + corner)
            total += (level + 1) / 4.0
    return total / 4.0


def propagate(
    size: tuple[int, int, int],
    emitters: dict[BlockCoordinate, int],
    opaque: set[BlockCoordinate],
) -> np.ndarray:
    """Flood fill light outward from each source, losing a level per block.

    The same breadth-first fill the game uses: a cell takes the brightest value
    any neighbour can hand it, and opaque cells pass nothing on.
    """
    levels = np.zeros(size, dtype=np.int16)
    queue: deque[tuple[BlockCoordinate, int]] = deque()
    for coordinate, level in sorted(emitters.items(), key=lambda item: -item[1]):
        if not all(0 <= coordinate[axis] < size[axis] for axis in range(3)):
            continue
        if level > levels[coordinate]:
            levels[coordinate] = level
            queue.append((coordinate, level))
    directions = tuple(tuple(int(v) for v in FACE_BASIS[slot][0]) for slot in FACE_BASIS)
    while queue:
        coordinate, level = queue.popleft()
        if level <= 1 or level < levels[coordinate]:
            continue
        for dx, dy, dz in directions:
            neighbour = (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz)
            if not all(0 <= neighbour[axis] < size[axis] for axis in range(3)):
                continue
            if neighbour in opaque:
                continue
            if levels[neighbour] >= level - 1:
                continue
            levels[neighbour] = level - 1
            queue.append((neighbour, level - 1))
    return levels


def light_field(
    size: tuple[int, int, int],
    blocks: dict[BlockCoordinate, BlockState],
    *,
    extra_emitters: dict[BlockCoordinate, int] | None = None,
    response: LightResponse = LightResponse(),
) -> LightField:
    """Build the light field implied by a set of decided blocks."""
    emitters: dict[BlockCoordinate, int] = {}
    opaque: set[BlockCoordinate] = set()
    for coordinate, state in blocks.items():
        level = emission_of(state)
        if level > 0:
            emitters[coordinate] = max(emitters.get(coordinate, 0), level)
        if blocks_light(state):
            opaque.add(coordinate)
    for coordinate, level in (extra_emitters or {}).items():
        if level > 0:
            emitters[coordinate] = max(emitters.get(coordinate, 0), level)
            opaque.discard(coordinate)
    return LightField(propagate(size, emitters, opaque), response)


def fit_response(
    measurements: list[tuple[int, float]], *, minimum: int = 12
) -> LightResponse:
    """Fit brightness against simulated level by least squares.

    ``measurements`` are ``(level, observed / expected)`` pairs taken from faces
    the classifier was confident about. With too few of them, or with no spread
    in level to fit against, the result is a flat response so that an
    uninformative simulation leaves the classifier where it was.
    """
    if len(measurements) < minimum:
        return LightResponse(floor=1.0, scale=1.0)
    curve = np.asarray([brightness(level) for level, _ratio in measurements], dtype=np.float64)
    observed = np.asarray([ratio for _level, ratio in measurements], dtype=np.float64)
    if float(curve.std()) < 1e-3:
        # Every face came out at the same simulated level, so the simulation says
        # nothing about their relative brightness. Fall back to one flat gain
        # rather than to the curve, which at level zero would darken every
        # expected colour to black.
        flat = float(np.clip(np.median(observed), 0.05, 1.5))
        return LightResponse(floor=flat, scale=flat, confidence=0.0)
    design = np.column_stack((curve, np.ones_like(curve)))
    slope, intercept = np.linalg.lstsq(design, observed, rcond=None)[0]
    floor = float(np.clip(intercept, 0.0, 1.0))
    scale = float(np.clip(intercept + slope, 0.05, 1.5))
    if scale <= floor:
        return LightResponse()
    spread = float(observed.std())
    correlation = (
        float(np.corrcoef(curve, observed)[0, 1]) if spread > 1e-6 else 0.0
    )
    return LightResponse(
        floor=floor, scale=scale, confidence=float(max(0.0, np.nan_to_num(correlation)))
    )


LEVEL_CURVE = np.array([brightness(level) for level in range(16)])
"""Rendered brightness of each light level, for inverting an observation."""

SOURCE_MINIMUM_LEVEL = 13
"""How brightly lit a cell must be before it is used to locate a source.

A brightly lit cell is close to whatever lights it, so its distance measurement is
unlikely to be dominated by a different source further away.
"""

SOURCE_CANDIDATES = 6
"""How many source positions to keep from the vote."""


def level_of(ratio: float) -> int:
    """The light level whose rendered brightness best matches an observed ratio."""
    return int(np.argmin(np.abs(LEVEL_CURVE - ratio)))


def estimate_light_levels(
    observations: list[tuple[BlockCoordinate, str, np.ndarray, np.ndarray]],
    occupied: set[BlockCoordinate],
    size: tuple[int, int, int],
) -> dict[BlockCoordinate, int]:
    """Light level of the air beside each observed face, from how dark it looks.

    A face is lit by the cell it faces into, not by its own cell, so the level is
    recorded against that air cell. Ambient occlusion is divided out first: smooth
    lighting darkens a face by up to half on its own, and reading that darkness as
    distance puts any source inferred from it several blocks too far away.
    """
    per_cell: dict[BlockCoordinate, list[int]] = {}
    for coordinate, slot, observed, expected in observations:
        normal = FACE_BASIS[slot][0]
        air = (
            coordinate[0] + int(normal[0]),
            coordinate[1] + int(normal[1]),
            coordinate[2] + int(normal[2]),
        )
        if not all(0 <= air[axis] < size[axis] for axis in range(3)):
            continue
        if air in occupied:
            continue
        total = float(np.mean(np.maximum(expected, 1e-6)))
        if total <= 1e-6:
            continue
        ratio = float(np.mean(observed)) / total
        shade = ambient_occlusion(coordinate, slot, occupied)
        if shade > 1e-3:
            ratio /= shade
        per_cell.setdefault(air, []).append(level_of(ratio))
    return {cell: int(np.median(values)) for cell, values in per_cell.items()}


def _propagation_distances(
    start: BlockCoordinate, size: tuple[int, int, int], opaque: set[BlockCoordinate]
) -> np.ndarray:
    """Steps light needs to reach each cell, going only through cells that pass it."""
    reach = np.full(size, 127, dtype=np.int16)
    reach[start] = 0
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        step = reach[cell] + 1
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nearby = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if not all(0 <= nearby[axis] < size[axis] for axis in range(3)):
                continue
            if reach[nearby] <= step:
                continue
            reach[nearby] = step
            if nearby not in opaque:
                queue.append(nearby)
    return reach


def locate_sources(
    levels: dict[BlockCoordinate, int],
    size: tuple[int, int, int],
    opaque: set[BlockCoordinate],
    *,
    minimum_level: int = SOURCE_MINIMUM_LEVEL,
    keep: int = SOURCE_CANDIDATES,
    sample_limit: int = 400,
) -> dict[BlockCoordinate, int]:
    """Where a light source has to be, given how bright the scene is around it.

    Minecraft light falls by exactly one level per block, so a cell observed at
    level ``L`` sits exactly ``15 - L`` propagation steps from a level-15 source -
    and the air beside that source is level 14, so the source itself is one step
    further again. That is a distance measurement, and enough of them intersect at
    a position. The source never has to be visible, which is the point: builders
    hide light sources under carpets and slabs, and a simulation that only knows
    about emitters it can see reports darkness in a room that is plainly lit.
    """
    # A cell reading full brightness carries no distance at all - the source could
    # be inside it or anywhere nearer than one step - so the informative cells are
    # the bright ones that are still short of the maximum. Sorting by level and
    # taking the top of the list selects precisely the useless ones.
    bright = [
        (cell, level)
        for cell, level in levels.items()
        if minimum_level <= level <= 14
    ]
    if not bright:
        return {}
    bright.sort(key=lambda item: (-item[1], item[0]))
    bright = bright[:sample_limit]

    # Find one source, drop the cells it accounts for, then look again. Taking
    # several peaks from a single vote finds the brightest corner of the scene
    # over and over: every cell near one lamp votes for it, and the runners-up are
    # its own neighbours rather than the next lamp along.
    reach_of = {cell: _propagation_distances(cell, size, opaque) for cell, _level in bright}
    remaining = list(bright)
    found: dict[BlockCoordinate, int] = {}
    while remaining and len(found) < keep:
        votes = np.zeros(size, dtype=np.float32)
        for cell, level in remaining:
            want = 15 - level - 1  # the air beside a level-15 source reads 14
            if want < 0:
                continue
            reach = reach_of[cell]
            votes[reach == want] += 1.0
            votes[np.abs(reach.astype(np.int32) - want) == 1] += 0.5
        if not votes.any():
            break
        best = tuple(int(value) for value in np.unravel_index(int(np.argmax(votes)), size))
        if best in found:
            break
        found[best] = 15
        explained = _propagation_distances(best, size, opaque)
        remaining = [
            (cell, level)
            for cell, level in remaining
            if abs(int(explained[cell]) - (15 - level - 1)) > 1
        ]
    return found
