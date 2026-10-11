"""Reference-only reconstruction metrics.

This module deliberately accepts already loaded blocks and visibility masks.  It
does not load fixture files, so production reconstruction code cannot acquire a
reference simply by importing an evaluator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from typing import Iterable, Mapping

from mc_photogrammetry.domain.models import BlockCoordinate, BlockState, BlockVolume, ObservationKind


@dataclass(frozen=True, slots=True)
class VisibleMetrics:
    """Metrics defined against the reference cells visible in usable pixels.

    Hidden reference cells (reference minus visible) carry their own inference
    credit: a candidate cell that lands on one is counted as
    ``hidden_inferred_blocks`` rather than as a phantom, and it never inflates
    visible recall or visible exact-state recovery.
    """

    visible_reference_blocks: int
    visible_coordinate_matches: int
    visible_exact_state_matches: int
    aligned_candidate_blocks: int
    phantom_blocks: int
    hidden_reference_blocks: int
    hidden_inferred_blocks: int
    hidden_exact_state_blocks: int

    @property
    def exact_state_recovery(self) -> float:
        return _ratio(self.visible_exact_state_matches, self.visible_reference_blocks)

    @property
    def geometry_recall(self) -> float:
        return _ratio(self.visible_coordinate_matches, self.visible_reference_blocks)

    @property
    def geometry_precision(self) -> float:
        return _ratio(self.visible_coordinate_matches, self.visible_coordinate_matches + self.phantom_blocks)

    @property
    def geometry_f1(self) -> float:
        return _harmonic_mean(self.geometry_precision, self.geometry_recall)

    @property
    def hidden_inference_recovery(self) -> float:
        """Recovered hidden cells as a share of the whole hidden reference.

        Separate credit only: hidden blocks are never counted as visible
        evidence and a hidden recovery is never a phantom.
        """
        return _ratio(self.hidden_inferred_blocks, self.hidden_reference_blocks)

    @property
    def hidden_exact_state_recovery(self) -> float:
        return _ratio(self.hidden_exact_state_blocks, self.hidden_reference_blocks)

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = asdict(self)
        result.update(
            exact_state_recovery=self.exact_state_recovery,
            geometry_recall=self.geometry_recall,
            geometry_precision=self.geometry_precision,
            geometry_f1=self.geometry_f1,
            hidden_inference_recovery=self.hidden_inference_recovery,
            hidden_exact_state_recovery=self.hidden_exact_state_recovery,
        )
        return result


def visible_metrics(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    visible_reference: Iterable[BlockCoordinate],
    *,
    translation: BlockCoordinate = (0, 0, 0),
    rotation_quarter_turns: int = 0,
) -> VisibleMetrics:
    """Score an aligned candidate using only reference cells visible in images.

    ``translation`` maps candidate-local coordinates into the reference frame.
    A candidate cell outside the full reference is a phantom even when it is not
    itself visible; this follows the promotion metric's ``C minus R`` term.
    """
    visible = set(visible_reference) & set(reference)
    aligned = _aligned_candidate(candidate, translation, rotation_quarter_turns)
    matching_coordinates = visible & set(aligned)
    exact_states = sum(reference[coordinate] == aligned[coordinate] for coordinate in matching_coordinates)
    phantoms = len(set(aligned) - set(reference))
    hidden = set(reference) - visible
    hidden_inferred = hidden & set(aligned)
    hidden_exact_states = sum(
        reference[coordinate] == aligned[coordinate] for coordinate in hidden_inferred
    )
    return VisibleMetrics(
        visible_reference_blocks=len(visible),
        visible_coordinate_matches=len(matching_coordinates),
        visible_exact_state_matches=exact_states,
        aligned_candidate_blocks=len(aligned),
        phantom_blocks=phantoms,
        hidden_reference_blocks=len(hidden),
        hidden_inferred_blocks=len(hidden_inferred),
        hidden_exact_state_blocks=hidden_exact_states,
    )


def best_visible_alignment(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    visible_reference: Iterable[BlockCoordinate],
) -> tuple[int, BlockCoordinate]:
    """Find the permitted Y-axis lattice rotation and integer translation.

    The rotation is applied to both coordinates and directional state properties,
    before an integer translation maps candidate coordinates into the reference
    frame.  It deliberately excludes reflections and scale fitting.
    """
    reference_coordinates = set(reference)
    visible = set(visible_reference) & reference_coordinates
    if not reference_coordinates or not candidate:
        return 0, (0, 0, 0)
    reference_sample = sorted(reference_coordinates)[:20_000]
    best = (0, (0, 0, 0), -1, -1)
    for turns in range(4):
        rotated = {_rotate_coordinate(coordinate, turns): state for coordinate, state in candidate.items()}
        candidate_sample = sorted(rotated)[:2_000]
        votes = Counter(
            (reference_coordinate[0] - candidate_coordinate[0],
             reference_coordinate[1] - candidate_coordinate[1],
             reference_coordinate[2] - candidate_coordinate[2])
            for reference_coordinate in reference_sample
            for candidate_coordinate in candidate_sample
        )
        for translation, _count in votes.most_common(32):
            aligned = {
                _translate(coordinate, translation): _rotate_state(state, turns)
                for coordinate, state in rotated.items()
            }
            coordinate_matches = len(visible & set(aligned))
            state_matches = sum(
                reference[coordinate] == aligned[coordinate]
                for coordinate in visible & set(aligned)
            )
            rank = (coordinate_matches, state_matches)
            if rank > best[2:]:
                best = (turns, translation, *rank)
    return best[0], best[1]


def visible_breakdowns(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    visible_pixels: Mapping[BlockCoordinate, int],
    *,
    translation: BlockCoordinate = (0, 0, 0),
    rotation_quarter_turns: int = 0,
) -> dict[str, dict[str, dict[str, int]]]:
    """Break visible exact/geometry evidence down by state and pixel-area bin."""
    aligned = _aligned_candidate(candidate, translation, rotation_quarter_turns)
    by_state: dict[str, Counter[str]] = {}
    by_area: dict[str, Counter[str]] = {}
    for coordinate, pixels in visible_pixels.items():
        state = reference.get(coordinate)
        if state is None:
            continue
        for group, key in ((by_state, _state_key(state)), (by_area, _pixel_area_bin(pixels))):
            counts = group.setdefault(key, Counter())
            counts["visible_reference_blocks"] += 1
            counts["visible_pixels"] += int(pixels)
            if coordinate in aligned:
                counts["coordinate_matches"] += 1
                if state == aligned[coordinate]:
                    counts["exact_state_matches"] += 1
    return {
        "by_state": {key: dict(counts) for key, counts in sorted(by_state.items())},
        "by_pixel_area": {key: dict(counts) for key, counts in sorted(by_area.items())},
    }


def observation_kind_counts(volume: BlockVolume) -> dict[str, int]:
    """Count observed and inferred output separately for benchmark reports."""
    counts = {kind.value: 0 for kind in ObservationKind}
    for observation in volume.blocks.values():
        counts[observation.kind.value] += 1
    return counts


def _aligned_candidate(
    candidate: Mapping[BlockCoordinate, BlockState],
    translation: BlockCoordinate,
    turns: int,
) -> dict[BlockCoordinate, BlockState]:
    return {
        _translate(_rotate_coordinate(coordinate, turns), translation): _rotate_state(state, turns)
        for coordinate, state in candidate.items()
    }


def _translate(coordinate: BlockCoordinate, translation: BlockCoordinate) -> BlockCoordinate:
    return tuple(coordinate[index] + translation[index] for index in range(3))  # type: ignore[return-value]


def _rotate_coordinate(coordinate: BlockCoordinate, turns: int) -> BlockCoordinate:
    x, y, z = coordinate
    for _ in range(turns % 4):
        x, z = -z, x
    return x, y, z


def _rotate_state(state: BlockState, turns: int) -> BlockState:
    properties = dict(state.properties)
    order = ("north", "east", "south", "west")
    turns %= 4

    def rotate(direction: str) -> str:
        return order[(order.index(direction) + turns) % 4] if direction in order else direction

    # Fence, wall and pane connections encode their direction in the *key*, not
    # only in a property value. Preserve their state (true/low/tall/false) while
    # moving it to the rotated key.
    directional = {key: properties.pop(key) for key in order if key in properties}
    for key, value in directional.items():
        properties[rotate(key)] = value
    facing = properties.get("facing")
    if facing in order:
        properties["facing"] = rotate(facing)
    if turns % 2 and properties.get("axis") in {"x", "z"}:
        properties["axis"] = "z" if properties["axis"] == "x" else "x"
    if properties.get("rotation", "").isdigit():
        properties["rotation"] = str((int(properties["rotation"]) + 4 * turns) % 16)
    # Rail shape values are orientation-bearing compound directions. Stair
    # shapes such as `inner_left` contain no cardinal token and remain intact.
    shape = properties.get("shape")
    if shape:
        properties["shape"] = "_".join(rotate(part) for part in shape.split("_"))
    return BlockState(state.name, tuple(sorted(properties.items())))


def _state_key(state: BlockState) -> str:
    if not state.properties:
        return state.name
    values = ",".join(f"{key}={value}" for key, value in state.properties)
    return f"{state.name}[{values}]"


def _pixel_area_bin(pixels: int) -> str:
    if pixels < 10:
        return "1-9"
    if pixels < 100:
        return "10-99"
    if pixels < 1000:
        return "100-999"
    return "1000+"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0
