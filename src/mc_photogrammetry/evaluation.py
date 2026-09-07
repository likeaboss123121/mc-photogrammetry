"""Reference-only reconstruction metrics.

This module deliberately accepts already loaded blocks and visibility masks.  It
does not load fixture files, so production reconstruction code cannot acquire a
reference simply by importing an evaluator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from mc_photogrammetry.domain.models import BlockCoordinate, BlockState, BlockVolume, ObservationKind


@dataclass(frozen=True, slots=True)
class VisibleMetrics:
    """Metrics defined against the reference cells visible in usable pixels."""

    visible_reference_blocks: int
    visible_coordinate_matches: int
    visible_exact_state_matches: int
    aligned_candidate_blocks: int
    phantom_blocks: int

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

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = asdict(self)
        result.update(
            exact_state_recovery=self.exact_state_recovery,
            geometry_recall=self.geometry_recall,
            geometry_precision=self.geometry_precision,
            geometry_f1=self.geometry_f1,
        )
        return result


def visible_metrics(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    visible_reference: Iterable[BlockCoordinate],
    *,
    translation: BlockCoordinate = (0, 0, 0),
) -> VisibleMetrics:
    """Score an aligned candidate using only reference cells visible in images.

    ``translation`` maps candidate-local coordinates into the reference frame.
    A candidate cell outside the full reference is a phantom even when it is not
    itself visible; this follows the promotion metric's ``C minus R`` term.
    """
    visible = set(visible_reference) & set(reference)
    aligned = {
        (coordinate[0] + translation[0], coordinate[1] + translation[1], coordinate[2] + translation[2]): state
        for coordinate, state in candidate.items()
    }
    matching_coordinates = visible & set(aligned)
    exact_states = sum(reference[coordinate] == aligned[coordinate] for coordinate in matching_coordinates)
    phantoms = len(set(aligned) - set(reference))
    return VisibleMetrics(
        visible_reference_blocks=len(visible),
        visible_coordinate_matches=len(matching_coordinates),
        visible_exact_state_matches=exact_states,
        aligned_candidate_blocks=len(aligned),
        phantom_blocks=phantoms,
    )


def observation_kind_counts(volume: BlockVolume) -> dict[str, int]:
    """Count observed and inferred output separately for benchmark reports."""
    counts = {kind.value: 0 for kind in ObservationKind}
    for observation in volume.blocks.values():
        counts[observation.kind.value] += 1
    return counts


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0
