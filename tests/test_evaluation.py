from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume, ObservationKind
from mc_photogrammetry.evaluation import observation_kind_counts, visible_metrics


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
