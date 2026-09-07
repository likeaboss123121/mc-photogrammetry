from mc_photogrammetry.domain.models import BlockObservation, BlockState
from mc_photogrammetry.reconstruction.raycast import _smooth_adjacent_states


def test_adjacency_smoothing_uses_majority_without_flooding_edges():
    stone = BlockState("minecraft:stone")
    dirt = BlockState("minecraft:dirt")
    merged = {
        (0, 0, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, 0, 0): (dirt, 1.0, set(), [dirt], {dirt: 1.0, stone: 0.5}),
        (2, 0, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, 1, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, -1, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
    }

    _smooth_adjacent_states(merged, 0.5, 3)

    assert merged[(1, 0, 0)][4][stone] >= merged[(1, 0, 0)][4][dirt]
    assert merged[(0, 0, 0)][0] is stone


def test_bounded_adjacency_does_not_invent_neighbor_candidates():
    stone = BlockState("minecraft:stone")
    dirt = BlockState("minecraft:dirt")
    merged = {
        (0, 0, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, 0, 0): (dirt, 1.0, set(), [dirt], {dirt: 1.0}),
        (2, 0, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, 1, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
        (1, -1, 0): (stone, 1.0, set(), [stone], {stone: 1.0}),
    }

    _smooth_adjacent_states(merged, 0.25, 3, bounded=True)

    assert stone not in merged[(1, 0, 0)][4]