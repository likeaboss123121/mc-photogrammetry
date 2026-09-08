from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume, ObservationKind
from mc_photogrammetry.evaluation import (
    best_visible_alignment,
    observation_kind_counts,
    visible_breakdowns,
    visible_metrics,
)


STONE = BlockState("minecraft:stone")
DIRT = BlockState("minecraft:dirt")


def test_visible_metrics_use_visible_reference_and_all_phantoms():
    metrics = visible_metrics(
        {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): STONE},
        {(9, 0, 0): STONE, (10, 0, 0): STONE, (99, 0, 0): STONE},
        [(0, 0, 0), (1, 0, 0)],
        translation=(-9, 0, 0),
    )

    assert metrics.visible_reference_blocks == 2
    assert metrics.visible_coordinate_matches == 2
    assert metrics.visible_exact_state_matches == 1
    assert metrics.phantom_blocks == 1
    assert metrics.exact_state_recovery == 0.5
    assert metrics.geometry_recall == 1.0
    assert metrics.geometry_precision == 2 / 3


def test_observation_kind_counts_preserve_observed_and_inferred_outputs():
    volume = BlockVolume()
    volume.set((0, 0, 0), BlockObservation(STONE))
    volume.set((1, 0, 0), BlockObservation(DIRT, kind=ObservationKind.INFERRED))

    assert observation_kind_counts(volume) == {"observed": 1, "inferred": 1}


def test_visible_alignment_rotates_coordinates_and_directional_states():
    north_stairs = BlockState("minecraft:oak_stairs", (("facing", "north"),))
    east_stairs = BlockState("minecraft:oak_stairs", (("facing", "east"),))
    reference = {(0, 0, 0): east_stairs, (0, 0, 1): STONE}
    candidate = {(0, 0, 0): north_stairs, (1, 0, 0): STONE}

    turns, translation = best_visible_alignment(reference, candidate, reference)
    metrics = visible_metrics(reference, candidate, reference,
                              translation=translation, rotation_quarter_turns=turns)

    assert (turns, translation) == (1, (0, 0, 0))
    assert metrics.visible_exact_state_matches == 2


def test_visible_breakdowns_preserve_state_and_pixel_area_evidence():
    breakdowns = visible_breakdowns(
        {(0, 0, 0): STONE, (1, 0, 0): DIRT},
        {(0, 0, 0): STONE, (1, 0, 0): STONE},
        {(0, 0, 0): 3, (1, 0, 0): 120},
    )

    assert breakdowns["by_state"]["minecraft:stone"]["exact_state_matches"] == 1
    assert breakdowns["by_state"]["minecraft:dirt"]["coordinate_matches"] == 1
    assert breakdowns["by_pixel_area"]["1-9"]["visible_pixels"] == 3
    assert breakdowns["by_pixel_area"]["100-999"]["visible_reference_blocks"] == 1
