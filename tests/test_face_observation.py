"""Immutable FaceObservation contract: identity, validation, abstention, retention."""

from __future__ import annotations

import numpy as np
import pytest

from mc_photogrammetry.reconstruction.face_observation import (
    NORMAL_OF_SLOT,
    FaceObservation,
    face_observation_from_sample,
)

FRAME = "frame-0000"


def _patch(value: float = 100.0, size: int = 2) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def _observation(**overrides: object) -> FaceObservation:
    fields: dict[str, object] = {
        "cell": (1, 2, 3),
        "slot": "north",
        "frame_id": FRAME,
        "normal": NORMAL_OF_SLOT["north"],
        "patch": _patch(),
        "usable_coverage": 1.0,
        "support_frames": (FRAME,),
        "provenance": ("camera-solution-0", "region-0"),
    }
    fields.update(overrides)
    return FaceObservation(**fields)  # type: ignore[arg-type]


def test_observation_identity_is_cell_slot_and_frame():
    observation = _observation()
    assert observation.identity == ((1, 2, 3), "north", FRAME)
    assert hash(observation) == hash(_observation())
    assert observation == _observation()


def test_different_frame_is_a_different_observation():
    other = _observation(frame_id="frame-0001", support_frames=("frame-0001",))
    assert other != _observation()
    assert hash(other) != hash(_observation())


def test_observation_is_frozen():
    observation = _observation()
    with pytest.raises(Exception):
        observation.slot = "south"  # type: ignore[misc]


def test_retained_arrays_are_copies_and_read_only():
    source = np.full((2, 2, 3), 10.0, dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    coverage = np.full((2, 2), 0.5, dtype=np.float64)
    observation = _observation(patch=source, valid=valid, coverage=coverage)

    assert observation.patch is not source
    source[:] = 200.0
    valid[:] = False
    coverage[:] = 0.0
    assert float(observation.patch.max()) == 10.0
    assert observation.valid is not None and observation.valid.all()
    assert float(observation.coverage.max()) == 0.5

    for array in (observation.patch, observation.valid, observation.coverage):
        assert array is not None and not array.flags.writeable


def test_retained_arrays_cannot_be_reopened_for_writing():
    observation = _observation(
        patch=np.full((2, 2, 3), 10.0, dtype=np.float32),
        valid=np.ones((2, 2), dtype=bool),
        coverage=np.full((2, 2), 0.5, dtype=np.float64),
    )
    for array in (observation.patch, observation.valid, observation.coverage):
        assert array is not None
        with pytest.raises(ValueError):
            array.setflags(write=True)
        assert array.base is not None
        with pytest.raises(ValueError):
            array.base.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_every_world_face_slot_accepts_its_canonical_normal():
    for slot, normal in NORMAL_OF_SLOT.items():
        observation = _observation(slot=slot, normal=normal)
        assert observation.slot == slot
        assert observation.normal == normal


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cell": (1, 2)}, "cell"),
        ({"cell": (1, 2, 3.0)}, "cell"),
        ({"cell": (1, 2, True)}, "cell"),
        ({"slot": "sideways"}, "slot"),
        ({"frame_id": ""}, "frame_id"),
        ({"normal": (0.0, 0.0, -2.0)}, "unit vector"),
        ({"normal": (float("nan"), 0.0, 0.0)}, "finite"),
        ({"normal": (0.0, 0.0, 1.0)}, "inconsistent"),
        ({"patch": np.zeros((2, 2), dtype=np.float32)}, "patch"),
        ({"patch": np.full((2, 2, 3), np.nan, dtype=np.float32)}, "finite"),
        ({"usable_coverage": 1.5}, "usable_coverage"),
        ({"usable_coverage": float("inf")}, "usable_coverage"),
        ({"usable_coverage": True}, "usable_coverage"),
        ({"support_frames": ()}, "support_frames"),
        ({"support_frames": ("other",)}, "include frame_id"),
        ({"support_frames": (FRAME, FRAME)}, "unique"),
        ({"provenance": ("",)}, "provenance"),
        ({"provenance": ("a", "a")}, "unique"),
        ({"weight": -1.0}, "weight"),
        ({"weight": float("nan")}, "weight"),
    ],
)
def test_malformed_observations_are_rejected(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        _observation(**overrides)


def test_valid_and_coverage_shapes_must_match_the_patch():
    with pytest.raises(ValueError, match="valid"):
        _observation(valid=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="coverage"):
        _observation(coverage=np.ones((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="coverage"):
        _observation(coverage=np.full((2, 2), 2.0, dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        _observation(coverage=np.full((2, 2), np.nan, dtype=np.float64))


def test_zero_coverage_abstains():
    observation = _observation(usable_coverage=0.0)
    assert observation.abstains
    assert observation.abstention_reason == "zero_coverage"


def test_no_valid_pixels_abstains():
    observation = _observation(valid=np.zeros((2, 2), dtype=bool))
    assert observation.abstains
    assert observation.abstention_reason == "no_valid_pixels"


def test_all_zero_coverage_abstains_despite_positive_usable_coverage():
    observation = _observation(coverage=np.zeros((2, 2), dtype=np.float64))
    assert observation.usable_coverage == 1.0
    assert not observation.has_effective_coverage
    assert observation.abstains
    assert observation.abstention_reason == "zero_coverage"


def test_all_zero_valid_abstains_despite_positive_usable_coverage():
    observation = _observation(valid=np.zeros((2, 2), dtype=bool))
    assert observation.usable_coverage == 1.0
    assert observation.abstains
    assert observation.abstention_reason == "no_valid_pixels"


def test_masks_that_never_overlap_leave_no_effective_coverage():
    valid = np.asarray([[True, True], [False, False]])
    coverage = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    observation = _observation(valid=valid, coverage=coverage)
    assert not observation.has_effective_coverage
    assert observation.abstention_reason == "zero_coverage"


def test_usable_observation_does_not_abstain():
    observation = _observation()
    assert observation.is_usable
    assert observation.abstention_reason is None


def test_excluded_pixels_never_reach_rgb_pattern_or_descriptor():
    valid = np.asarray([[True, True], [True, False]])
    coverage = np.asarray([[1.0, 0.5], [0.5, 0.0]])
    clean = np.zeros((2, 2, 3), dtype=np.float32)
    clean[0, 0] = (30.0, 60.0, 90.0)
    clean[0, 1] = (40.0, 70.0, 100.0)
    clean[1, 0] = (50.0, 80.0, 110.0)
    clean[1, 1] = (60.0, 90.0, 120.0)
    dirty = clean.copy()
    dirty[~valid] = (255.0, 0.0, 255.0)  # excluded pixel only

    a = _observation(patch=clean, valid=valid, coverage=coverage)
    b = _observation(patch=dirty, valid=valid, coverage=coverage)

    assert np.allclose(a.rgb, b.rgb)
    assert a.pattern is not None and b.pattern is not None
    assert np.array_equal(a.pattern, b.pattern)
    assert a.descriptor == b.descriptor


def test_coverage_zero_pixels_do_not_change_the_descriptor():
    coverage = np.asarray([[1.0, 1.0], [1.0, 0.0]])
    clean = np.full((2, 2, 3), 125.0, dtype=np.float32)
    dirty = clean.copy()
    dirty[1, 1] = (255.0, 0.0, 255.0)
    a = _observation(patch=clean, coverage=coverage)
    b = _observation(patch=dirty, coverage=coverage)
    assert a.descriptor == b.descriptor


def test_invalid_pixels_do_not_change_the_descriptor():
    valid = np.asarray([[True, True], [True, False]])
    clean = np.full((2, 2, 3), 125.0, dtype=np.float32)
    dirty = clean.copy()
    dirty[~valid] = (255.0, 0.0, 255.0)
    a = _observation(patch=clean, valid=valid)
    b = _observation(patch=dirty, valid=valid)
    assert a.descriptor == b.descriptor


def test_coverage_weights_the_mean_colour():
    patch = np.zeros((2, 1, 3), dtype=np.float32)
    patch[0, 0] = (100.0, 100.0, 100.0)
    patch[1, 0] = (0.0, 0.0, 0.0)
    full = _observation(patch=patch, coverage=np.asarray([[1.0], [1.0]]))
    weighted = _observation(patch=patch, coverage=np.asarray([[1.0], [0.0]]))
    assert np.allclose(full.rgb, (50.0, 50.0, 50.0))
    assert np.allclose(weighted.rgb, (100.0, 100.0, 100.0))


def test_face_observation_from_sample_adapts_a_rectified_sample():
    class Sample:
        coordinate = (4, 5, 6)
        slot = "east"
        patch = _patch(120.0)
        visible = 0.75
        valid = np.ones((2, 2), dtype=bool)
        coverage = np.full((2, 2), 0.75, dtype=np.float64)

    observation = face_observation_from_sample(
        Sample(), frame_id=FRAME, provenance=("region-9",)
    )
    assert observation.cell == (4, 5, 6)
    assert observation.slot == "east"
    assert observation.normal == NORMAL_OF_SLOT["east"]
    assert observation.support_frames == (FRAME,)
    assert observation.usable_coverage == pytest.approx(0.75)
    assert observation.provenance == ("region-9",)


def test_face_observation_from_sample_accepts_integral_float_coordinates():
    class Sample:
        coordinate = (4.0, 5.0, 6.0)
        slot = "east"
        patch = _patch()
        visible = 0.5

    observation = face_observation_from_sample(Sample(), frame_id=FRAME)
    assert observation.cell == (4, 5, 6)


@pytest.mark.parametrize(
    "coordinate, message",
    [
        ((1.9, 2.0, 3.0), "integral"),
        ((1, 2.0, 3.0 + 1e-9), "integral"),
        ((float("nan"), 2, 3), "integral"),
        ((1, 2, float("inf")), "integral"),
        ((True, 2, 3), "booleans"),
        ((1, 2), "3-tuple"),
        ((1, 2, 3, 4), "3-tuple"),
        (("1", "2", "3"), "integral numbers"),
        (None, "3-tuple"),
    ],
)
def test_face_observation_from_sample_rejects_lossy_coordinates(coordinate, message):
    class Sample:
        slot = "north"
        patch = _patch()
        visible = 1.0

    Sample.coordinate = coordinate  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=message):
        face_observation_from_sample(Sample(), frame_id=FRAME)


def test_face_observation_from_sample_rejects_invalid_or_missing_fields():
    class MissingCoordinate:
        slot = "east"
        patch = _patch()
        visible = 1.0

    with pytest.raises(ValueError, match="coordinate"):
        face_observation_from_sample(MissingCoordinate(), frame_id=FRAME)

    class BadSlot:
        coordinate = (1, 2, 3)
        slot = "sideways"
        patch = _patch()
        visible = 1.0

    with pytest.raises(ValueError, match="slot"):
        face_observation_from_sample(BadSlot(), frame_id=FRAME)

    class BadVisible:
        coordinate = (1, 2, 3)
        slot = "north"
        patch = _patch()
        visible = float("nan")

    with pytest.raises(ValueError, match="usable_coverage"):
        face_observation_from_sample(BadVisible(), frame_id=FRAME)

    class BooleanVisible:
        coordinate = (1, 2, 3)
        slot = "north"
        patch = _patch()
        visible = True

    with pytest.raises(ValueError, match="usable_coverage"):
        face_observation_from_sample(BooleanVisible(), frame_id=FRAME)

    class BadPatch:
        coordinate = (1, 2, 3)
        slot = "north"
        patch = np.zeros((2, 2), dtype=np.float32)
        visible = 1.0

    with pytest.raises(ValueError, match="patch"):
        face_observation_from_sample(BadPatch(), frame_id=FRAME)

