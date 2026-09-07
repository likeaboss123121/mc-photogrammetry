"""Stereo along the ray, and the two tests that make it work.

Cross-view appearance agreement alone produces interior proposals 8-23% correct,
because a phantom cell in front of a wall reprojects onto a different part of the
same wall and correlates just as well. Uniqueness and cross-camera consistency lift
that to 75-98%, so both are pinned here along with the runner-up bookkeeping they
depend on, which a plain depth buffer discards.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.reconstruction.stereo import (
    UNIQUENESS_MARGIN,
    face_scores,
    stereo_surface,
)

SIZE = (6, 6, 6)


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=SIZE)


def _camera(position, yaw=0.0, pitch=0.0):
    return CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                       fov_degrees=110.0)


def _frames(count=2, value=128):
    return {
        f"frame{index}.png": np.full((64, 64, 3), value, dtype=np.uint8)
        for index in range(count)
    }


def _cameras(count=2):
    return {
        f"frame{index}.png": _camera((-4.0 - index, 3.0, 3.0 + index * 2.0), yaw=-90.0)
        for index in range(count)
    }


def test_no_scores_means_no_surface():
    assert stereo_surface(_scene(), _cameras(), _frames(), scores={}) == set()


def test_a_cell_needs_two_cameras_to_be_believed():
    # One camera cannot place anything: a phantom that wins one line of sight is
    # exactly what cross-camera consistency exists to reject.
    scene = _scene()
    scores = {((2, 2, 2), "west"): 0.9}
    single = stereo_surface(scene, dict(list(_cameras(1).items())),
                            _frames(1), scores=scores, minimum_cameras=2)
    assert single == set()


def test_the_margin_rejects_a_tie_and_keeps_a_clear_winner():
    # Two faces competing for the same pixels: with equal scores neither clears the
    # margin and the ray stays unknown; with a gap the winner is taken.
    scene = _scene()
    tied = {((2, 2, 2), "west"): 0.80, ((3, 2, 2), "west"): 0.80}
    clear = {((2, 2, 2), "west"): 0.95, ((3, 2, 2), "west"): 0.30}
    cameras, frames = _cameras(2), _frames(2)
    assert stereo_surface(scene, cameras, frames, scores=tied, margin=0.25) == set()
    assert stereo_surface(scene, cameras, frames, scores=clear, margin=0.25)


def test_a_lower_margin_admits_what_a_higher_one_refuses():
    scene = _scene()
    scores = {((2, 2, 2), "west"): 0.60, ((3, 2, 2), "west"): 0.50}
    cameras, frames = _cameras(2), _frames(2)
    assert stereo_surface(scene, cameras, frames, scores=scores, margin=0.05)
    assert stereo_surface(scene, cameras, frames, scores=scores, margin=0.40) == set()


def test_the_default_margin_is_the_measured_one():
    # 0.25 is where interior proposals measured 75-80% correct on the fixtures.
    # Changing it silently changes the precision the pipeline ships with.
    assert UNIQUENESS_MARGIN == 0.25


def test_a_bad_scale_is_refused():
    with pytest.raises(ValueError):
        stereo_surface(_scene(), _cameras(), _frames(), scores={((1, 1, 1), "west"): 1.0},
                       scale=0)


def test_a_bad_camera_count_is_refused():
    with pytest.raises(ValueError):
        stereo_surface(_scene(), _cameras(), _frames(), scores={((1, 1, 1), "west"): 1.0},
                       minimum_cameras=0)


def test_a_face_no_camera_faces_is_never_placed():
    # Every camera here looks along -x, so an east face points away from all of
    # them and must not win anything.
    scene = _scene()
    scores = {((2, 2, 2), "east"): 0.99}
    assert stereo_surface(scene, _cameras(2), _frames(2), scores=scores) == set()


def test_a_flat_image_yields_no_scores():
    # A patch with no texture has no pattern, so nothing to agree about. This is
    # also why the measure cannot be trusted on a repeated wall.
    scene = _scene()
    assert face_scores(scene, _cameras(2), _frames(2)) == {}


def test_scores_need_two_views_of_the_same_face():
    scene = _scene()
    rng = np.random.default_rng(0)
    frames = {"frame0.png": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)}
    cameras = {"frame0.png": _camera((-4.0, 3.0, 3.0), yaw=-90.0)}
    assert face_scores(scene, cameras, frames) == {}
