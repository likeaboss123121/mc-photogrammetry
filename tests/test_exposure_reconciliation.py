import numpy as np

from mc_photogrammetry.reconstruction.raycast import _reconcile_exposure


def _observation(frame, coordinate, brightness, slot="north", area=100.0):
    return (
        frame,
        coordinate,
        np.asarray([brightness, brightness, brightness], dtype=float),
        (),
        slot,
        1.0,
        None,
        area,
    )


def _shared(brightness_a, brightness_b, count=12, area_a=100.0, area_b=100.0):
    """Both frames seeing the same faces, one recording them brighter."""
    raw = []
    for index in range(count):
        coordinate = (index, 0, 0)
        raw.append(_observation("a", coordinate, brightness_a, area=area_a))
        raw.append(_observation("b", coordinate, brightness_b, area=area_b))
    return raw


def test_a_brighter_frame_is_scaled_down_relative_to_a_darker_one():
    gains = _reconcile_exposure(_shared(120.0, 60.0))
    assert gains["a"] < gains["b"]
    # The pair differ by a factor of two, and only that difference is recoverable.
    assert np.isclose(gains["b"] / gains["a"], 2.0, rtol=0.02)


def test_the_overall_brightness_of_the_scene_is_left_alone():
    # How bright the scene really was cannot be recovered from the images, so the
    # fit must correct only the disagreement between frames and not push the
    # whole scene up or down.
    gains = _reconcile_exposure(_shared(120.0, 60.0))
    assert np.isclose(float(np.exp(np.mean(np.log(list(gains.values()))))), 1.0, rtol=1e-6)


def test_frames_that_already_agree_are_left_at_unity():
    gains = _reconcile_exposure(_shared(80.0, 80.0))
    assert all(np.isclose(gain, 1.0, atol=1e-6) for gain in gains.values())


def test_too_few_shared_faces_leaves_every_frame_alone():
    # With almost nothing seen twice there is no measurement to fit, and inventing
    # one would move every block for no reason.
    gains = _reconcile_exposure(_shared(120.0, 60.0, count=1))
    assert all(gain == 1.0 for gain in gains.values())


def test_frames_with_no_shared_faces_are_left_alone():
    raw = [_observation("a", (0, 0, 0), 120.0), _observation("b", (9, 9, 9), 60.0)]
    gains = _reconcile_exposure(raw)
    assert all(gain == 1.0 for gain in gains.values())


def test_centre_patch_samples_are_ignored():
    # A centre patch is not a measurement of one face, so it cannot anchor a
    # ratio between two views of that face.
    gains = _reconcile_exposure(_shared(120.0, 60.0, area_a=0.0, area_b=0.0))
    assert all(gain == 1.0 for gain in gains.values())


def test_a_single_frame_needs_no_reconciling():
    raw = [_observation("a", (index, 0, 0), 100.0) for index in range(12)]
    assert _reconcile_exposure(raw) == {"a": 1.0}


def test_an_extreme_ratio_is_clamped():
    gains = _reconcile_exposure(_shared(250.0, 5.0), limit=2.0)
    assert all(0.5 - 1e-6 <= gain <= 2.0 + 1e-6 for gain in gains.values())


def test_three_frames_are_solved_together():
    raw = []
    for index in range(12):
        coordinate = (index, 0, 0)
        raw.append(_observation("a", coordinate, 100.0))
        raw.append(_observation("b", coordinate, 110.0))
        raw.append(_observation("c", coordinate, 90.0))
    gains = _reconcile_exposure(raw)
    # Correcting each frame by its gain should bring the three onto one scale.
    corrected = [100.0 * gains["a"], 110.0 * gains["b"], 90.0 * gains["c"]]
    assert max(corrected) - min(corrected) < 1e-6
