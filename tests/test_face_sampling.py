import numpy as np
import pytest

from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.face_sampling import (
    PATCH_SIZE,
    DEPTH_SCALE,
    default_face_corners,
    quad_area,
    rectify_face,
    sample_visible_faces,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.vision.gui_mask import create_gui_mask


def _striped_image(height, width, period=8):
    """Horizontal stripes, so averaging over them is visibly different from sampling one."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    rows = (np.arange(height) // period) % 2
    image[rows == 0] = 240
    image[rows == 1] = 20
    return image


def test_rectify_face_area_averages_instead_of_point_sampling():
    # A large striped face must average to mid grey. Point sampling would land on
    # individual stripes and return near-black or near-white noise instead.
    image = _striped_image(600, 600, period=8)
    quad = np.asarray([[50, 50], [550, 50], [550, 550], [50, 550]], dtype=np.float32)

    patch = rectify_face(image, quad)

    assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert 100.0 < float(patch.mean()) < 160.0
    assert float(patch.std()) < 40.0


def test_face_sampling_defaults_to_source_pixel_depth_ownership():
    assert DEPTH_SCALE == 1.0


def test_rectify_face_preserves_a_pattern_it_can_resolve():
    # Eight stripes across sixteen output rows stays resolvable; a finer period
    # would alias to flat grey no matter how faithful the resampling is.
    image = _striped_image(320, 320, period=40)
    quad = np.asarray([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32)

    patch = rectify_face(image, quad).mean(axis=2)

    assert patch.max() - patch.min() > 150.0


def test_rectify_face_rejects_malformed_input():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        rectify_face(image, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        rectify_face(image, np.zeros((4, 2), dtype=np.float32), size=1)


def test_default_face_corners_lie_on_their_own_face():
    for slot, axis, sign in (("up", 1, 0.5), ("down", 1, -0.5), ("east", 0, 0.5), ("west", 0, -0.5)):
        corners = default_face_corners(slot)
        assert corners.shape == (4, 3)
        assert np.allclose(corners[:, axis], sign)


def test_sample_visible_faces_returns_only_unoccluded_front_faces():
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    camera = CameraModel((2.0, 2.0, -6.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (4, 4, 4))
    # Two blocks in a line away from the camera; the far one hides behind the near one.
    coordinates = {(2, 2, 1), (2, 2, 2)}

    samples = sample_visible_faces(coordinates, camera, scene, image, minimum_area=1.0)

    slots = {(sample.coordinate, sample.slot) for sample in samples}
    assert ((2, 2, 1), "north") in slots
    # The shared boundary between the two blocks is never sampled from either side.
    assert ((2, 2, 2), "north") not in slots
    assert ((2, 2, 1), "south") not in slots
    assert all(sample.patch.shape == (PATCH_SIZE, PATCH_SIZE, 3) for sample in samples)


def test_sample_visible_faces_skips_faces_under_the_hud():
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    camera = CameraModel((2.0, 2.0, -6.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (4, 4, 4))
    coordinates = {(2, 2, 1)}

    without_mask = sample_visible_faces(coordinates, camera, scene, image, minimum_area=1.0)
    covered = np.ones((400, 400), dtype=bool)
    with_mask = sample_visible_faces(
        coordinates, camera, scene, image, gui_mask=covered, minimum_area=1.0
    )

    assert without_mask
    assert with_mask == []


def test_sample_visible_faces_keeps_the_unmasked_part_of_a_hud_overlapped_face():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    mask = np.zeros((100, 100), dtype=bool)
    mask[42:44, 42:59] = True
    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, scene, image, gui_mask=mask, minimum_area=1.0, shrink=1.0
    )

    assert samples
    north = next(sample for sample in samples if sample.slot == "north")
    assert north.valid is not None and not north.valid.all()
    assert north.pixels.shape[0] < north.patch.shape[0] * north.patch.shape[1]


def test_sample_visible_faces_rectifies_the_front_part_of_a_near_plane_face():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -0.1), 60.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    rejections = {}
    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, SceneTransform((0, 0, 0), (2, 2, 2)), image,
        minimum_area=1.0, shrink=1.0, rejections=rejections,
    )

    assert any(sample.slot == "north" and sample.pixels.size for sample in samples)
    assert rejections.get("behind_camera", 0) == 0


def test_quad_area_matches_a_known_rectangle():
    quad = np.asarray([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float32)
    assert quad_area(quad) == pytest.approx(40.0)


def test_gui_mask_covers_the_held_item_corner():
    mask = create_gui_mask(1000, 2000)
    assert bool(mask[930, 1900])
    # Well clear of the HUD, the crosshair and the held item.
    assert not bool(mask[300, 1100])
