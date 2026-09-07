import numpy as np

from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.multiview import reconstruct_visible_surfaces
from mc_photogrammetry.reconstruction.scene import SceneTransform


def test_visible_surface_fusion_merges_frame_provenance():
    image = np.full((128, 128, 3), 125, dtype=np.uint8)
    frames = {"one.png": image, "two.png": image.copy()}
    cameras = {
        "one.png": CameraModel((4.0, 4.0, -10.0), 0.0, 0.0, eye_height=0.0),
        "two.png": CameraModel((4.0, 4.0, -10.0), 0.0, 0.0, eye_height=0.0),
    }

    volume = reconstruct_visible_surfaces(
        frames,
        cameras,
        scene=SceneTransform((0, 0, 0), (8, 8, 8)),
        columns=8,
        rows=8,
        depth_layers=2,
    )

    assert volume.blocks
    assert any(observation.frame_ids == ["one.png", "two.png"] for observation in volume.blocks.values())