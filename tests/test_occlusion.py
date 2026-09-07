import numpy as np

from mc_photogrammetry.reconstruction.face_sampling import (
    OPPOSITE_SLOT,
    _hidden_by,
    _nearest_surface,
)

SQUARE = np.asarray([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]], dtype=np.float32)


def _quad(depths, corners=SQUARE, coordinate=(0, 0, 0), slot="north", facing=1.0):
    depths = np.asarray(depths, dtype=np.float64)
    return (
        float(depths.mean()),
        np.asarray(corners, dtype=np.float32),
        depths,
        coordinate,
        slot,
        facing,
    )


def test_opposite_slots_pair_up():
    for slot, other in OPPOSITE_SLOT.items():
        assert OPPOSITE_SLOT[other] == slot


def test_without_shape_any_neighbour_is_assumed_to_occlude():
    occupied = {(0, 1, 0)}
    assert _hidden_by((0, 1, 0), "up", occupied, None)
    assert not _hidden_by((0, -1, 0), "down", occupied, None)


def test_a_carpet_above_hides_the_face_beneath_it():
    # A carpet fills a sixteenth of its cell but closes that cell's underside,
    # so the floor below is completely hidden even though the cell is not full.
    shapes = {(0, 1, 0): frozenset({"down"})}
    assert _hidden_by((0, 1, 0), "up", {(0, 1, 0)}, shapes)


def test_a_top_slab_above_does_not_hide_the_face_beneath_it():
    # A top slab occupies the upper half of its cell, so there is air between it
    # and the block below and that block's upward face is still visible. Treating
    # occupancy alone as occlusion gets this exactly backwards.
    shapes = {(0, 1, 0): frozenset({"up"})}
    assert not _hidden_by((0, 1, 0), "up", {(0, 1, 0)}, shapes)


def test_a_neighbour_with_no_recorded_shape_does_not_occlude():
    assert not _hidden_by((0, 1, 0), "up", {(0, 1, 0)}, {})


def test_the_nearer_face_owns_the_pixels():
    far = _quad([10.0, 10.0, 10.0, 10.0])
    near = _quad([2.0, 2.0, 2.0, 2.0])
    owner = _nearest_surface([far, near], 100, 100, 1.0)
    assert owner[50, 50] == 1
    assert (owner == 0).sum() == 0
    assert (owner == -1)[0, 0]


def test_interpenetrating_faces_are_ordered_per_pixel():
    # Both faces span the same square and share a mean depth, but one is nearer
    # on the left and the other on the right. Sorting whole faces by mean depth
    # would hand every pixel to whichever was painted last; comparing depth per
    # pixel splits them where they actually cross.
    left_near = _quad([1.0, 10.0, 10.0, 1.0])
    right_near = _quad([10.0, 1.0, 1.0, 10.0])
    assert left_near[0] == right_near[0]
    owner = _nearest_surface([left_near, right_near], 100, 100, 1.0)
    assert owner[50, 20] == 0
    assert owner[50, 80] == 1
    assert (owner == 0).sum() > 0 and (owner == 1).sum() > 0


def test_a_face_running_off_the_frame_is_clipped_not_crashed():
    off = np.asarray([[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]], dtype=np.float32)
    owner = _nearest_surface([_quad([3.0] * 4, corners=off)], 100, 100, 1.0)
    assert owner[10, 10] == 0
    assert owner.shape == (100, 100)
    # Nothing outside the drawn corner is claimed.
    assert owner[90, 90] == -1


def test_visible_share_of_an_unobstructed_face_is_its_whole_area():
    quad = _quad([5.0] * 4)
    owner = _nearest_surface([quad], 100, 100, 1.0)
    drawn = float((owner == 0).sum())
    area = 80.0 * 80.0
    assert 0.9 < drawn / area <= 1.05


def test_a_half_covered_face_reports_about_half_its_area():
    behind = _quad([9.0] * 4)
    cover = np.asarray([[10.0, 10.0], [50.0, 10.0], [50.0, 90.0], [10.0, 90.0]], dtype=np.float32)
    front = _quad([2.0] * 4, corners=cover)
    owner = _nearest_surface([behind, front], 100, 100, 1.0)
    remaining = float((owner == 0).sum()) / (80.0 * 80.0)
    assert 0.4 < remaining < 0.6
