from mc_photogrammetry.domain.models import BlockState, ObservationKind
from mc_photogrammetry.reconstruction.grid import EvidenceGrid, FillPolicy, evidence_to_volume


def test_evidence_preserves_confidence_and_frame_provenance():
    evidence = EvidenceGrid()
    evidence.add((0, 64, 0), 0.8, "frame-1")
    evidence.add((0, 64, 0), 0.6, "frame-2")

    volume = evidence_to_volume(evidence, block_state=BlockState("minecraft:oak_planks"))
    observation = volume.get((0, 64, 0))

    assert observation is not None
    assert observation.confidence == 0.8
    assert observation.frame_ids == ["frame-1", "frame-2"]
    assert observation.kind is ObservationKind.OBSERVED


def test_solid_fill_marks_new_blocks_as_inferred():
    evidence = EvidenceGrid()
    evidence.add((0, 0, 0), 1.0)
    evidence.add((2, 0, 0), 1.0)

    volume = evidence_to_volume(evidence, policy=FillPolicy.SOLID_FILL)
    inferred = volume.get((1, 0, 0))

    assert inferred is not None
    assert inferred.kind is ObservationKind.INFERRED
    assert inferred.confidence == 0.25
