"""Proposing planar regions from segment directions alone.

The load-bearing claim is that a plane's normal is the lattice direction
*missing* from it, not one of the directions present. These check that claim
holds on segments built from the project's own projection, and that the
proposal declines to answer where the evidence is thin.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points
from mc_photogrammetry.reconstruction.surfaces import (
    PlanarRegion,
    direction_weights,
    normal_votes,
    planar_regions,
)

WIDTH, HEIGHT = 640, 480
FOCAL = (HEIGHT / 2.0) / np.tan(np.radians(70.0) / 2.0)


def _camera(yaw=0.0, pitch=0.0):
    return CameraModel(position=(0.5, 0.5, 0.5), yaw_degrees=yaw,
                       pitch_degrees=pitch, fov_degrees=70.0)


def _face_segments(camera, axis, offset, span=6):
    """Edges of a lattice face perpendicular to ``axis`` at world ``offset``.

    Draws only the two in-plane directions, which is what a real face shows: an
    edge along the normal points away from the surface.
    """
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    others = [i for i in range(3) if i != axis]
    rows = []
    for along, across in ((others[0], others[1]), (others[1], others[0])):
        for step in range(-span, span + 1):
            start = np.array(eye, dtype=float)
            start[axis] += offset
            start[across] += step
            start[along] -= span
            finish = start.copy()
            finish[along] += 2 * span
            pair = project_points([tuple(start), tuple(finish)], camera,
                                  width=WIDTH, height=HEIGHT)
            if any(p is None for p in pair):
                continue
            (x0, y0, _a), (x1, y1, _b) = pair
            rows.append((x0, y0, x1, y1))
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 4))


def test_a_face_normal_is_the_direction_missing_from_it():
    """The claim the whole module rests on.

    A face perpendicular to Z shows edges along X and Y and none along Z, so Z
    is identifiable by its absence. Presence could not tell it from a face
    perpendicular to X, which also shows edges along Y.
    """
    camera = _camera()
    segments = _face_segments(camera, axis=2, offset=6.0)
    assert len(segments) > 8
    weights = direction_weights(segments, FOCAL, 0.0, 0.0, WIDTH, HEIGHT)
    total = weights.sum(axis=(0, 1))
    assert total[2] < 0.05 * total.sum(), "edges along the normal should be absent"
    assert total[0] > 0 and total[1] > 0


def test_the_voted_normal_matches_the_face_that_was_drawn():
    camera = _camera()
    for axis in (0, 1, 2):
        segments = _face_segments(camera, axis=axis, offset=6.0)
        if len(segments) < 8:
            continue
        weights = direction_weights(segments, FOCAL, 0.0, 0.0, WIDTH, HEIGHT)
        voted, _margin = normal_votes(weights)
        chosen = voted[voted >= 0]
        assert chosen.size > 0, f"no tile voted for axis {axis}"
        winner = np.bincount(chosen, minlength=3).argmax()
        assert winner == axis


def test_a_tile_with_almost_nothing_in_it_declines_to_vote():
    # Two of three directions are absent from an empty tile for want of any
    # evidence at all, which would otherwise read as a confident normal.
    weights = np.zeros((3, 3, 3))
    weights[1, 1, 0] = 5.0
    voted, _margin = normal_votes(weights, minimum_weight=60.0)
    assert (voted == -1).all()


def test_an_evenly_mixed_tile_declines_to_vote():
    weights = np.full((2, 2, 3), 100.0)
    voted, margin = normal_votes(weights, minimum_weight=10.0)
    assert (voted == -1).all(), "no direction is missing, so no normal is implied"
    assert (margin == 0.0).all()


def test_a_clear_absence_does_vote():
    weights = np.zeros((1, 1, 3))
    weights[0, 0] = (200.0, 180.0, 2.0)
    voted, margin = normal_votes(weights, minimum_weight=60.0)
    assert voted[0, 0] == 2
    assert margin[0, 0] > 0.35


def test_regions_group_tiles_that_agree_and_are_ranked_by_size():
    camera = _camera()
    segments = _face_segments(camera, axis=2, offset=6.0)
    regions = planar_regions(segments, FOCAL, 0.0, 0.0, WIDTH, HEIGHT,
                             minimum_tiles=2)
    assert regions, "a full-frame wall produced no region"
    assert regions[0].axis == 2
    assert regions[0].area_tiles >= regions[-1].area_tiles
    assert 0.0 < regions[0].confidence <= 1.0


def test_a_region_renders_a_mask_over_the_tiles_it_owns():
    region = PlanarRegion(axis=1, tiles=((0, 0), (0, 1)), confidence=0.5,
                          tile_size=10)
    mask = region.mask(30, 30)
    assert mask[:10, :20].all()
    assert not mask[10:, :].any()
    assert mask.sum() == pytest.approx(200.0)


def test_a_region_reports_where_it_sits():
    region = PlanarRegion(axis=0, tiles=((1, 1),), confidence=0.5, tile_size=10)
    x, y = region.centroid()
    assert (x, y) == (15.0, 15.0)


def test_no_segments_means_no_regions_rather_than_an_error():
    assert planar_regions(np.zeros((0, 4)), FOCAL, 0.0, 0.0, WIDTH, HEIGHT) == []


def test_regions_smaller_than_the_lattice_are_not_offered():
    camera = _camera()
    segments = _face_segments(camera, axis=2, offset=6.0)
    huge = planar_regions(segments, FOCAL, 0.0, 0.0, WIDTH, HEIGHT,
                          minimum_tiles=10_000)
    assert huge == [], "a region below the size floor should not be proposed"
