"""Tiny support-array falsifiers; no images, detector, production or packet calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import region_growth as rg
from mc_photogrammetry.reconstruction.plane_graph import axis_transform


def one_atom(axis, side, missing_tangent=False):
    tangents = [other for other in range(3) if other != axis]
    support = np.zeros((1, 3), dtype=np.float32)
    support[0, tangents] = 60.0
    if missing_tangent:
        support[0, tangents[1]] = 0.0
    return SimpleNamespace(node_area=np.array([100]), supports=support,
        plane_scores=rg._normal_scores(support), sides=np.full((1, 3), side, dtype=np.int8),
        node_lab=np.zeros((1, 3), dtype=np.float32), adjacency=((),),
        axis_line_dirs=np.zeros((1, 3, 2), dtype=np.float32))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
def test_default_seeding_is_axis_symmetric_with_two_tangents(axis, side):
    graph = one_atom(axis, side)
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    assert graph.plane_scores[0, axis] > config.min_seed_score
    assert graph.supports[0, axis] == 0
    assert rg._seed_and_grow(graph, axis, side, config) == [[0]]
    assert rg._seed_and_grow(graph, axis, -side, config) == []
    assert sum(len(rg._seed_and_grow(graph, other, side, config))
               for other in range(3) if other != axis) == 0


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
def test_one_tangent_cannot_invent_a_plane_seed_under_default_gates(axis, side):
    graph = one_atom(axis, side, missing_tangent=True)
    assert graph.plane_scores[0, axis] == 0
    assert graph.supports.sum() == 60  # ample one-direction mass is insufficient
    assert rg._seed_and_grow(graph, axis, side, rg.DEFAULT_REGION_GROWTH_CONFIG) == []


@pytest.mark.parametrize("quadrant", [0, 1, 2, 3])
def test_yaw_quadrants_do_not_turn_horizontal_normals_into_vertical_normal(quadrant):
    assert axis_transform(1, quadrant)[0] == 1
    assert {axis_transform(axis, quadrant)[0] for axis in (0, 2)} == {0, 2}
