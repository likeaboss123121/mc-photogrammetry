from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
    ScanProject,
)
from mc_photogrammetry.project.store import load_project, save_project


def test_project_round_trip_preserves_sparse_volume(tmp_path):
    project = ScanProject(name="Test build", source_paths=["capture.png"])
    project.volume = BlockVolume(
        blocks={
            (0, 64, 0): BlockObservation(BlockState("minecraft:stone")),
            (1, 64, 0): BlockObservation(
                BlockState("minecraft:oak_log", (("axis", "y"),)),
                confidence=0.42,
                kind=ObservationKind.INFERRED,
                frame_ids=["frame-001"],
            ),
        }
    )

    path = tmp_path / "scan.mcscan"
    save_project(project, path)
    restored = load_project(path)

    assert restored.to_dict() == project.to_dict()
