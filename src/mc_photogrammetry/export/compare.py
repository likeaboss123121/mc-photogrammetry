from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path

from mc_photogrammetry.domain.models import BlockState


@dataclass(frozen=True, slots=True)
class LitematicSnapshot:
    path: Path
    version: int
    minecraft_data_version: int
    position: tuple[int, int, int]
    size: tuple[int, int, int]
    palette: tuple[BlockState, ...]
    blocks: dict[tuple[int, int, int], BlockState]

    @property
    def non_air_count(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True, slots=True)
class LitematicComparison:
    reference: LitematicSnapshot
    candidate: LitematicSnapshot
    exact_coordinate_matches: int
    exact_state_matches: int
    candidate_palette_matches: int
    best_translation: tuple[int, int, int] = (0, 0, 0)
    direct_coordinate_matches: int = 0
    direct_state_matches: int = 0
    confusion_pairs: tuple[tuple[str, str, int], ...] = ()
    per_state_accuracy: dict[str, float] = field(default_factory=dict)
    per_state_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    reference_exposed: int = 0
    surface_matches: int = 0

    @property
    def coordinate_recall(self) -> float:
        return _ratio(self.exact_coordinate_matches, self.reference.non_air_count)

    @property
    def state_accuracy(self) -> float:
        return _ratio(self.exact_state_matches, self.exact_coordinate_matches)

    @property
    def surface_precision(self) -> float:
        return _ratio(self.surface_matches, self.candidate.non_air_count)

    @property
    def surface_recall(self) -> float:
        return _ratio(self.surface_matches, self.reference_exposed)

    @property
    def surface_f1(self) -> float:
        """How well the placed cells match the surface a camera could actually see.

        ``coordinate_f1`` scores against every reference block, including ones
        buried where nothing can see them, so a reconstruction that emits the
        region's bounding-box shell scores well on a build that happens to fill its
        own box - which is how a geometry stage that never worked went unnoticed
        behind a 94.70% figure. This scores against the blocks genuinely exposed to
        air inside the region, which is the set a reconstruction can actually be
        asked to recover.
        """
        precision = self.surface_precision
        recall = self.surface_recall
        return _ratio(2 * precision * recall, precision + recall)

    @property
    def whole_reference_state_accuracy(self) -> float:
        """Correct blocks as a share of every reference block, placed or not.

        ``state_accuracy`` is conditional on having placed a block at all, so it
        rises when the reconstruction declines to guess. Reporting both keeps a
        drop in recall from reading as a gain in accuracy.
        """
        return _ratio(self.exact_state_matches, self.reference.non_air_count)

    @property
    def direct_coordinate_recall(self) -> float:
        return _ratio(self.direct_coordinate_matches, self.reference.non_air_count)

    @property
    def coordinate_precision(self) -> float:
        return _ratio(self.exact_coordinate_matches, self.candidate.non_air_count)

    @property
    def coordinate_f1(self) -> float:
        recall = self.coordinate_recall
        precision = self.coordinate_precision
        return _ratio(2 * precision * recall, precision + recall)

    @property
    def aligned_candidate_blocks(self) -> int:
        return self.candidate.non_air_count


def load_litematic(path: str | Path) -> LitematicSnapshot:
    try:
        import nbtlib
    except ImportError as error:
        raise RuntimeError("litematic comparison requires nbtlib") from error

    source = Path(path)
    document = nbtlib.load(source, gzipped=True)
    version = int(document["Version"])
    data_version = int(document["MinecraftDataVersion"])
    if len(document["Regions"]) != 1:
        raise ValueError("comparison currently supports exactly one region")
    region = next(iter(document["Regions"].values()))
    position = tuple(int(region["Position"][axis]) for axis in "xyz")
    size = tuple(int(region["Size"][axis]) for axis in "xyz")
    palette = tuple(_read_state(state) for state in region["BlockStatePalette"])
    bits = max(2, (len(palette) - 1).bit_length())
    indices = _unpack_indices(region["BlockStates"], bits, size[0] * size[1] * size[2])
    blocks: dict[tuple[int, int, int], BlockState] = {}
    index = 0
    for y in range(size[1]):
        for z in range(size[2]):
            for x in range(size[0]):
                palette_index = indices[index]
                index += 1
                if palette_index != 0:
                    blocks[(position[0] + x, position[1] + y, position[2] + z)] = palette[palette_index]
    return LitematicSnapshot(source, version, data_version, position, size, palette, blocks)


NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def camera_reachable_air(
    reference: LitematicSnapshot,
    camera_cells: Iterable[tuple[int, int, int]] | None = None,
) -> set[tuple[int, int, int]]:
    """The air a photographer could have stood in, flooded from where they did.

    A block is only visible if there is air beside it connected to a camera. On an
    open build that is nearly all the air there is, so the distinction costs
    nothing; on a build photographed from inside a sealed space it is the whole
    question, and counting the air around the outside of the mass credits a
    reconstruction for surfaces no photograph contains.

    With no camera cells to flood from, this returns the air inside the region,
    which is the older and more conservative reading.
    """
    solid = set(reference.blocks)
    if not solid:
        return set()
    lowest = tuple(min(c[axis] for c in solid) - 1 for axis in range(3))
    highest = tuple(max(c[axis] for c in solid) + 1 for axis in range(3))

    def within(cell: tuple[int, int, int]) -> bool:
        return all(lowest[axis] <= cell[axis] <= highest[axis] for axis in range(3))

    seeds = [
        cell for cell in (camera_cells or ()) if within(cell) and cell not in solid
    ]
    if not seeds:
        width, height, depth = reference.size
        return {
            (x, y, z)
            for x in range(width)
            for y in range(height)
            for z in range(depth)
            if (x, y, z) not in solid
        }
    reached = set(seeds)
    queue = deque(seeds)
    while queue:
        cell = queue.popleft()
        for dx, dy, dz in NEIGHBOURS:
            nearby = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if nearby in reached or nearby in solid or not within(nearby):
                continue
            reached.add(nearby)
            queue.append(nearby)
    return reached


def compare_litematics(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    camera_cells: Iterable[tuple[int, int, int]] | None = None,
) -> LitematicComparison:
    reference = load_litematic(reference_path)
    candidate = load_litematic(candidate_path)
    translation, common_coordinates = _best_translation(reference.blocks, candidate.blocks)
    state_matches = sum(
        reference.blocks[coordinate]
        == candidate.blocks[_subtract(coordinate, translation)]
        for coordinate in common_coordinates
    )
    reference_names = {state.name for state in reference.palette}
    candidate_palette_matches = sum(state.name in reference_names for state in candidate.palette)
    direct_coordinates = set(reference.blocks) & set(candidate.blocks)
    direct_states = sum(reference.blocks[coordinate] == candidate.blocks[coordinate] for coordinate in direct_coordinates)

    pair_counts: Counter[tuple[str, str]] = Counter()
    per_state_totals: Counter[str] = Counter()
    per_state_correct: Counter[str] = Counter()
    for coordinate in common_coordinates:
        reference_state = reference.blocks[coordinate]
        candidate_state = candidate.blocks[_subtract(coordinate, translation)]
        pair_counts[(reference_state.name, candidate_state.name)] += 1
        per_state_totals[reference_state.name] += 1
        if reference_state == candidate_state:
            per_state_correct[reference_state.name] += 1

    confusion_pairs = tuple(
        (reference_state, candidate_state, count)
        for (reference_state, candidate_state), count in sorted(
            pair_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[:25]
    )
    per_state_accuracy = {
        state_name: _ratio(per_state_correct[state_name], per_state_totals[state_name])
        for state_name in sorted(per_state_totals)
    }
    # Blocks with air beside them that a camera could reach: what was seeable.
    #
    # Neither "air inside the region" nor "air anywhere" is the right test, and the
    # two disagree wildly on an enclosed build. Template 4 is 88% solid with its
    # cameras sealed in an interior room: air-inside-the-region counts 703 of its
    # blocks as visible, air-anywhere counts 2629 - and the truth is 576, because
    # the outside of the mass is air no camera ever stood in.
    #
    # Given the camera cells, the answer is exact: flood the air from where the
    # photographer stood and take the blocks that touch it. Without them, fall back
    # to air inside the region.
    air = camera_reachable_air(reference, camera_cells)
    exposed = {
        coordinate
        for coordinate in reference.blocks
        if any(
            (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz) in air
            for dx, dy, dz in NEIGHBOURS
        )
    }
    placed = {_add(coordinate, translation) for coordinate in candidate.blocks}
    surface_matches = len(placed & exposed)

    reference_totals: Counter[str] = Counter(state.name for state in reference.blocks.values())
    per_state_counts = {
        state_name: {
            "in_reference": reference_totals[state_name],
            "seen": per_state_totals[state_name],
            "correct": per_state_correct[state_name],
            "missed": reference_totals[state_name] - per_state_totals[state_name],
        }
        for state_name in sorted(reference_totals)
    }

    return LitematicComparison(
        reference,
        candidate,
        len(common_coordinates),
        state_matches,
        candidate_palette_matches,
        translation,
        len(direct_coordinates),
        direct_states,
        confusion_pairs,
        per_state_accuracy,
        per_state_counts,
        len(exposed),
        surface_matches,
    )


def _best_translation(
    reference: dict[tuple[int, int, int], BlockState],
    candidate: dict[tuple[int, int, int], BlockState],
) -> tuple[tuple[int, int, int], set[tuple[int, int, int]]]:
    """Align two local schematics by maximizing occupied-coordinate overlap."""
    if not reference or not candidate:
        return (0, 0, 0), set()
    reference_coordinates = set(reference)
    candidate_coordinates = set(candidate)
    best_translation = (0, 0, 0)
    best_overlap: set[tuple[int, int, int]] = set()
    candidate_sample = sorted(candidate_coordinates)[: min(2000, len(candidate_coordinates))]
    reference_sample = sorted(reference_coordinates)[: min(20000, len(reference_coordinates))]
    translation_votes = Counter(
        (
            reference_coordinate[0] - candidate_coordinate[0],
            reference_coordinate[1] - candidate_coordinate[1],
            reference_coordinate[2] - candidate_coordinate[2],
        )
        for reference_coordinate in reference_sample
        for candidate_coordinate in candidate_sample
    )
    for translation, _ in translation_votes.most_common(32):
        overlap = {
            coordinate
            for coordinate in candidate_coordinates
            if _add(coordinate, translation) in reference_coordinates
        }
        if len(overlap) > len(best_overlap):
            best_translation = translation
            best_overlap = {_add(coordinate, translation) for coordinate in overlap}
    return best_translation, best_overlap


def _add(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(left[index] + right[index] for index in range(3))


def _subtract(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(left[index] - right[index] for index in range(3))


def _read_state(value: object) -> BlockState:
    name = str(value["Name"])  # type: ignore[index]
    properties_value = value.get("Properties", {})  # type: ignore[union-attr]
    properties = tuple(sorted((str(key), str(item)) for key, item in properties_value.items()))
    return BlockState(name, properties)


def _unpack_indices(values: object, bits: int, count: int) -> list[int]:
    raw = [int(value) & ((1 << 64) - 1) for value in values]  # type: ignore[union-attr]
    mask = (1 << bits) - 1
    indices: list[int] = []
    for index in range(count):
        bit_index = index * bits
        long_index, offset = divmod(bit_index, 64)
        value = raw[long_index] >> offset
        if offset + bits > 64:
            value |= raw[long_index + 1] << (64 - offset)
        indices.append(value & mask)
    return indices


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
