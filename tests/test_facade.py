import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState, ObservationKind
from mc_photogrammetry.reconstruction.facade import PaletteColor, estimate_horizon, reconstruct_facade
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.scene import SceneTransform


def test_facade_reconstruction_ignores_masked_cells():
    image = np.full((4, 4, 3), (162, 130, 79), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    palette = (PaletteColor(BlockState("minecraft:oak_planks"), (162, 130, 79)),)

    volume = reconstruct_facade(image, mask, columns=2, rows=2, palette=palette)

    assert volume.get((0, 1, 0)) is None
    assert volume.get((1, 0, 0)) is not None
    assert len(volume.blocks) == 3


def test_facade_reconstruction_validates_mask_shape():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="match the image"):
        reconstruct_facade(image, np.zeros((2, 2), dtype=bool))


def test_facade_depth_estimate_creates_multiple_z_layers():
    image = np.full((16, 16, 3), (125, 125, 125), dtype=np.uint8)

    volume = reconstruct_facade(image, columns=4, rows=4, depth_layers=4)

    assert max(coordinate[2] for coordinate in volume.blocks) > 0
    assert any(coordinate[2] > 0 for coordinate in volume.blocks)
    assert all(observation.kind is ObservationKind.OBSERVED for observation in volume.blocks.values())


def test_facade_perspective_maps_horizon_to_far_depth():
    image = np.full((20, 20, 3), (125, 125, 125), dtype=np.uint8)

    volume = reconstruct_facade(image, columns=5, rows=5, depth_layers=5, horizon=0.4)

    top_depths = {coordinate[2] for coordinate in volume.blocks if coordinate[1] >= 4}
    bottom_depths = {coordinate[2] for coordinate in volume.blocks if coordinate[1] <= 1}
    assert top_depths == {0}
    assert max(bottom_depths) > 0


def test_depth_estimate_does_not_shear_image_columns():
    image = np.full((32, 32, 3), (125, 125, 125), dtype=np.uint8)

    volume = reconstruct_facade(image, columns=8, rows=8, depth_layers=4, horizon=0.4)

    assert {coordinate[0] for coordinate in volume.blocks} == set(range(8))


def test_default_mode_emits_visible_surface_blocks_at_estimated_depth():
    image = np.full((32, 32, 3), (125, 125, 125), dtype=np.uint8)

    volume = reconstruct_facade(image, columns=8, rows=8, depth_layers=4, horizon=0.4)

    assert max(coordinate[2] for coordinate in volume.blocks) > 0
    assert all(observation.kind is ObservationKind.OBSERVED for observation in volume.blocks.values())
    assert len(volume.blocks) <= 64


def test_visible_surface_coordinates_are_image_aligned():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, :16] = (162, 130, 79)
    image[:, 16:] = (125, 125, 125)
    palette = (
        PaletteColor(BlockState("minecraft:oak_planks"), (162, 130, 79)),
        PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),
    )

    volume = reconstruct_facade(image, columns=8, rows=4, palette=palette)

    assert {coordinate[2] for coordinate in volume.blocks} == {0}
    assert all(0 <= coordinate[0] < 8 for coordinate in volume.blocks)
    assert all(0 <= coordinate[1] < 4 for coordinate in volume.blocks)


def test_camera_aware_surface_uses_world_projection():
    image = np.full((32, 32, 3), (125, 125, 125), dtype=np.uint8)
    camera = CameraModel((0.0, 0.0, 0.0), yaw_degrees=-90.0, pitch_degrees=0.0, eye_height=0.0)

    volume = reconstruct_facade(image, columns=4, rows=4, depth_layers=4, camera=camera)

    assert all(coordinate[0] >= 0 for coordinate in volume.blocks)
    assert any(coordinate[2] < 0 for coordinate in volume.blocks)


def test_camera_aware_scene_bounds_keep_surface_inside_volume():
    image = np.full((32, 32, 3), (125, 125, 125), dtype=np.uint8)
    camera = CameraModel((4.0, 4.0, -10.0), yaw_degrees=0.0, pitch_degrees=0.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (8, 8, 8))

    volume = reconstruct_facade(image, columns=4, rows=4, depth_layers=4, camera=camera, scene_transform=scene)

    assert volume.blocks
    assert all(0 <= coordinate[index] < scene.size[index] for coordinate in volume.blocks for index in range(3))


def test_horizon_estimation_ignores_masked_edges():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[10:] = 200
    mask = np.zeros((20, 20), dtype=bool)
    mask[9:12] = True

    horizon = estimate_horizon(image, mask)

    assert 0.2 <= horizon <= 0.7
