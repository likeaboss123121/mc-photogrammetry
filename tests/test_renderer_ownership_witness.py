from dataclasses import replace
import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mc_photogrammetry.export import attribution
from mc_photogrammetry.reconstruction import lattice
from test_attribution import WHITE, _camera, _palette, _scene, face_colours


def test_renderer_witness_preserves_legacy_outputs_and_distinct_same_slot_faces():
    corners = tuple(map(tuple, attribution.box_face_corners(
        (0, 0, 0), (1, 1, 1), "north")))
    models = {WHITE: (
        ("north", corners, (0.0, 0.0, -1.0), None, None),
        ("north", corners, (0.0, 0.0, -1.0), None, None))}
    args = ({(3, 2, 4): WHITE}, _scene(), _camera(), 120, 120,
            face_colours(_palette()))
    plain = attribution.render_result(*args, scale=1.0, model_faces=models)
    sink = []
    observed = attribution.render_result(
        *args, scale=1.0, model_faces=models, witness_sink=sink)
    np.testing.assert_array_equal(observed[0].ownership, plain[0].ownership)
    assert observed[1] == plain[1]
    assert len(sink) == 1
    rows = sink[0].owners
    assert [row.source_ordinal for row in rows] == [0, 1]
    assert all(row.source_kind == "model_face" for row in rows)
    assert rows[0].physical_plane_key == rows[1].physical_plane_key


def test_culling_nonaxis_and_depth_tie_semantics_are_recorded_without_selection_change():
    corners = tuple(map(tuple, attribution.box_face_corners(
        (0, 0, 0), (1, 1, 1), "north")))
    diagonal = tuple((x, y, z + x * 0.25) for x, y, z in corners)
    models = {WHITE: (("north", diagonal, (0.25, 0.0, -1.0), None, None),)}
    sink = []
    attribution.render_result(
        {(3, 2, 4): WHITE}, _scene(), _camera(), 120, 120,
        face_colours(_palette()), scale=1.0, model_faces=models,
        witness_sink=sink)
    assert sink[0].owners and sink[0].owners[0].physical_plane_key is None

    # Duplicate boxes are a depth tie. Both owner rows remain distinct while
    # strict-nearer drawing leaves ownership with the first face.
    tie_sink = []
    result, index = attribution.render_result(
        {(3, 2, 4): WHITE}, _scene(), _camera(), 120, 120,
        face_colours(_palette()), scale=1.0,
        shapes={WHITE: (attribution.CUBE[0], attribution.CUBE[0])},
        witness_sink=tie_sink)
    assert len(tie_sink[0].owners) == len(index)
    assert any(row.source_ordinal == 1 for row in tie_sink[0].owners)
    assert np.all(result.ownership[result.ownership >= 0] < len(index))


def test_owner_witness_is_deterministic_tamper_bound_and_budget_atomic():
    args = ({(3, 2, 4): WHITE}, _scene(), _camera(), 120, 120,
            face_colours(_palette()))
    first, second = [], []
    plain = attribution.render_result(*args, witness_sink=first)
    observed = attribution.render_result(*args, witness_sink=second)
    np.testing.assert_array_equal(plain[0].ownership, observed[0].ownership)
    assert first[0].witness_sha256 == second[0].witness_sha256
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(first[0], scale=0.5)
    refused = []
    capped = attribution.render_result(
        *args, witness_sink=refused,
        witness_budget=lattice.WitnessBudgetV1(max_owner_faces=0))
    np.testing.assert_array_equal(capped[0].ownership, plain[0].ownership)
    assert len(refused) == 1 and isinstance(refused[0], lattice.WitnessRefusalV1)
    assert refused[0].stage == "owner_faces" and refused[0].actual == 0


def test_culled_and_fully_transparent_faces_publish_no_owner_row():
    corners = tuple(map(tuple, attribution.box_face_corners(
        (0, 0, 0), (1, 1, 1), "north")))
    foreground, neighbour = (3, 2, 4), (3, 2, 3)
    culled = {WHITE: (("north", corners, (0.0, 0.0, -1.0), "north", None),)}
    sink = []
    attribution.render_result(
        {foreground: WHITE, neighbour: WHITE}, _scene(), _camera(), 120, 120,
        face_colours(_palette()), scale=1.0, model_faces=culled,
        occluding_slots={WHITE: frozenset(attribution.FACE_SLOTS)},
        witness_sink=sink)
    assert all(not (row.block_coordinate == foreground and row.slot == "north")
               for row in sink[0].owners)

    alpha = np.zeros((16, 16), dtype=np.uint8)
    buffer = BytesIO(); Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    transparent = {WHITE: (("north", corners, (0.0, 0.0, -1.0), None, encoded),)}
    transparent_sink = []
    _result, index = attribution.render_result(
        {foreground: WHITE}, _scene(), _camera(), 120, 120,
        face_colours(_palette()), scale=1.0, model_faces=transparent,
        witness_sink=transparent_sink)
    assert index == [] and transparent_sink[0].owners == ()
