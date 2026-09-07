from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
    BlockCoordinate,
)


class FillPolicy(str, Enum):
    SURFACE_ONLY = "surface_only"
    CONSERVATIVE_CLOSE = "conservative_close"
    SOLID_FILL = "solid_fill"


@dataclass(slots=True)
class EvidenceGrid:
    """Sparse occupancy evidence accumulated from one or more views."""

    scores: dict[BlockCoordinate, float] = field(default_factory=dict)
    frame_ids: dict[BlockCoordinate, list[str]] = field(default_factory=dict)

    def add(self, coordinate: BlockCoordinate, score: float, frame_id: str | None = None) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError("evidence scores must be between 0 and 1")
        self.scores[coordinate] = max(score, self.scores.get(coordinate, 0.0))
        if frame_id and frame_id not in self.frame_ids.setdefault(coordinate, []):
            self.frame_ids[coordinate].append(frame_id)


def evidence_to_volume(
    evidence: EvidenceGrid,
    *,
    block_state: BlockState = BlockState("minecraft:stone"),
    threshold: float = 0.5,
    policy: FillPolicy = FillPolicy.SURFACE_ONLY,
) -> BlockVolume:
    """Convert positive evidence to a sparse volume without claiming hidden blocks."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    volume = BlockVolume()
    for coordinate, score in evidence.scores.items():
        if score < threshold:
            continue
        volume.set(
            coordinate,
            BlockObservation(
                state=block_state,
                confidence=score,
                kind=ObservationKind.OBSERVED,
                frame_ids=list(evidence.frame_ids.get(coordinate, [])),
            ),
        )

    if policy is FillPolicy.CONSERVATIVE_CLOSE:
        _add_single_voxel_holes(volume, block_state)
    elif policy is FillPolicy.SOLID_FILL:
        _fill_bounding_box(volume, block_state)
    return volume


def _add_single_voxel_holes(volume: BlockVolume, block_state: BlockState) -> None:
    existing = set(volume.blocks)
    if not existing:
        return
    minimum = tuple(min(axis[index] for axis in existing) for index in range(3))
    maximum = tuple(max(axis[index] for axis in existing) for index in range(3))
    for x in range(minimum[0] + 1, maximum[0]):
        for y in range(minimum[1] + 1, maximum[1]):
            for z in range(minimum[2] + 1, maximum[2]):
                coordinate = (x, y, z)
                if coordinate in existing:
                    continue
                neighbors = ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1))
                if all(neighbor in existing for neighbor in neighbors):
                    volume.set(coordinate, BlockObservation(block_state, confidence=0.5, kind=ObservationKind.INFERRED))


def _fill_bounding_box(volume: BlockVolume, block_state: BlockState) -> None:
    existing = set(volume.blocks)
    if not existing:
        return
    minimum = tuple(min(axis[index] for axis in existing) for index in range(3))
    maximum = tuple(max(axis[index] for axis in existing) for index in range(3))
    for x in range(minimum[0], maximum[0] + 1):
        for y in range(minimum[1], maximum[1] + 1):
            for z in range(minimum[2], maximum[2] + 1):
                coordinate = (x, y, z)
                if coordinate not in existing:
                    volume.set(coordinate, BlockObservation(block_state, confidence=0.25, kind=ObservationKind.INFERRED))
