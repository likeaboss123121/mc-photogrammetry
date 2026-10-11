"""Independent controls for the isolated M4 V3 geometry-only volume bridge.

The genuine end-to-end input is the accepted mixed-normal P0 recipe from
``tests/test_plane_placement.py``: its solver-produced nine-face
``ResolvedFacePlacement``, its original ``PlacementUniverse`` and its nine
accepted V1 supports are reused here through those public/test APIs only.  No
certificate, support or budget is fabricated and no gate is weakened.  Every
expected coordinate, count and bound below is declared from the recipe's own
literal construction (``_literal_recipe_normal`` / ``_literal_recipe_tangents`` /
``_literal_recipe_cells`` and the declared supported/unknown grids), never read
back from ``plane_volume``.

The literal recipe emits 66 supported face cells over 58 distinct coordinates,
omits the 15 declared boundary-unknown cells, and records **no** conflict: the
seven merged coordinates are compatible perpendicular faces of one cube, and the
two opposite faces of a cube are compatible too.  All three world axes, both
outward signs, six reflected tangents, signed/fractional translations and
negative coordinates are present.

The source-bound API is exercised with the original universe, supports,
placement budget and volume budget, and the tamper controls rebuild a
self-consistent resealed placement (``+100`` normal translation), a stale
placement digest, a resealed support cell and resealed/modified volume results.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_plane_placement as recipe_tests  # noqa: E402
from test_plane_placement import (  # noqa: E402
    RECIPE_AXES,
    RECIPE_FRAMES,
    RECIPE_SUPPORTED_GRID,
    RECIPE_UNKNOWN_GRID,
    _literal_recipe_cells,
    _literal_recipe_normal,
    _literal_recipe_tangents,
)

from mc_photogrammetry.domain.models import (  # noqa: E402
    BlockState,
    BlockVolume,
    ObservationKind,
)
from mc_photogrammetry.export.litematic import LitematicExporter  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_region_support as v1  # noqa: E402
from mc_photogrammetry.reconstruction import plane_volume as pv  # noqa: E402


SOURCE = Path(pv.__file__)
PREVIEW_STATE = BlockState("minecraft:stone")

# Literal V3 expectations declared from the recipe construction.  The seven
# merged coordinates are compatible faces of one cube, so the valid conflict
# tuple is empty.
RECIPE_VOLUME_WORK = {
    "faces": 9,
    "support_cells": 81,
    "contributions": 66,
    "coordinates": 58,
    "conflicts": 0,
    "omitted_unknown": 15,
    "dense_bounds_cells": 224,
}
RECIPE_VOLUME_BOUNDS = ((-4, 0, -4), (2, 3, 3))
RECIPE_MERGED_COORDINATES = {
    (0, 1, 1): 2, (1, 0, 1): 2, (1, 1, 0): 2, (1, 1, 1): 3, (1, 1, 2): 2,
    (1, 2, 1): 2, (2, 1, 1): 2,
}


def _literal_volume_contributions():
    """The declared coordinate of every literal supported recipe cell."""
    contributions = []
    for frame in RECIPE_FRAMES:
        for axis in RECIPE_AXES:
            normal = _literal_recipe_normal(frame, axis)
            tangents = _literal_recipe_tangents(frame, axis)
            for row, column, lateral_u, lateral_v in _literal_recipe_cells(
                    frame, axis):
                coordinate = [0, 0, 0]
                coordinate[normal["world_axis"]] = normal["normal_block"]
                coordinate[tangents[0][2]] = lateral_u
                coordinate[tangents[1][2]] = lateral_v
                contributions.append((tuple(coordinate),
                                      (frame, axis, row, column)))
    return tuple(contributions)


def _budget(**overrides):
    caps = dict(
        max_faces=9, max_support_cells=81, max_contributions=66,
        max_coordinates=58, max_conflicts=1, max_omitted_unknown=15,
        max_dense_bounds_cells=224)
    caps.update(overrides)
    return pv.make_volume_budget(**caps)


def _clone(record, **changes):
    """Copy a frozen record without re-running its validating constructor."""
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _reseal_result(result, **changes):
    """Copy a result and recompute its digest like a fresh reseal would."""
    clone = _clone(result, **changes)
    object.__setattr__(clone, "volume_sha256",
                       pv._sha256(pv._volume_payload(clone)))
    return clone


def _valid_phantom_result(recipe_result):
    """A fully self-consistent resealed result carrying one phantom coordinate.

    Every derived field the result validates (work counters, cell bounds, dense
    size, normalization offset and the binding counts) is recomputed, so the
    record passes its own ``PlaneVolumeResult`` constructor.  Only the
    source-bound replay can reject it, which is exactly the case a tiny volume
    budget must not be allowed to mask with ``budget_exhausted``.
    """
    claim = _synthetic_claim(coordinate=(9, 9, 9), face=9, normal_block=9,
                             lateral_u=9, lateral_v=9)
    phantom = _observation((claim,))
    cells = tuple(recipe_result.cells) + (phantom,)
    bindings = [dict(binding) for binding in recipe_result.support_bindings]
    bindings[0]["supported_cells"] += 1
    bindings[0]["support_cells"] += 1
    low = tuple(min(cell.coordinate[axis] for cell in cells) for axis in range(3))
    high = tuple(max(cell.coordinate[axis] for cell in cells) for axis in range(3))
    dense = 1
    for axis in range(3):
        dense *= high[axis] - low[axis] + 1
    work = pv._make_work({
        "faces": recipe_result.work.faces,
        "support_cells": recipe_result.work.support_cells + 1,
        "contributions": recipe_result.work.contributions + 1,
        "coordinates": recipe_result.work.coordinates + 1,
        "conflicts": recipe_result.work.conflicts,
        "omitted_unknown": recipe_result.work.omitted_unknown,
        "dense_bounds_cells": dense,
    })
    return pv.PlaneVolumeResult.build(
        schema_version=recipe_result.schema_version,
        producer_version=recipe_result.producer_version,
        gauge_state=recipe_result.gauge_state,
        gauge_axes=recipe_result.gauge_axes,
        alpha=recipe_result.alpha,
        placement_sha256=recipe_result.placement_sha256,
        support_bindings=tuple(sorted(bindings, key=pv._binding_sort_key)),
        cells=cells,
        conflicts=recipe_result.conflicts,
        omitted=recipe_result.omitted,
        cell_bounds=(low, high),
        dense_bounds_cells=dense,
        normalization_offset=low,
        export_bounds=recipe_result.export_bounds,
        complete=recipe_result.complete,
        work=work)


def _source_kwargs(context):
    universe, supports, placement, placement_budget, budget = context
    return dict(universe=universe, supports=supports, placement=placement,
                placement_budget=placement_budget, budget=budget)


def _preview_kwargs(context):
    kwargs = _source_kwargs(context)
    kwargs.update(preview_state=PREVIEW_STATE, geometry_only=True)
    return kwargs


def _reseal_face_translation(placement, position, delta=100):
    """Freshly reseal one face's normal translation by ``delta`` blocks.

    The face's own gauge, integer face and normal block are shifted with the
    translation, so the resealed face and placement replay every one of their
    own public fields and carry a fresh canonical digest.  This is exactly the
    self-consistent ``+100`` reseal the source-bound replay must still refuse.
    """
    faces = list(placement.faces)
    target = faces[position]
    fields = {field.name: getattr(target, field.name)
              for field in dataclasses.fields(target)
              if field.name != "face_sha256"}
    axis = int(target.world_normal_axis)
    translation = list(fields["translation"])
    numerator, denominator = translation[axis]
    translation[axis] = (numerator + delta * denominator, denominator)
    fields["translation"] = tuple(translation)
    gauge = list(fields["face_gauge"])
    gauge[0] = gauge[0] + delta * gauge[1]
    fields["face_gauge"] = tuple(gauge)
    fields["face"] = fields["face"] + delta
    fields["normal_block"] = fields["normal_block"] + delta
    faces[position] = plane_placement.ResolvedFace.build(**fields)
    placement_fields = {field.name: getattr(placement, field.name)
                        for field in dataclasses.fields(placement)
                        if field.name != "placement_sha256"}
    placement_fields["faces"] = tuple(
        sorted(faces, key=plane_placement.face_sort_key))
    return plane_placement.ResolvedFacePlacement.build(**placement_fields)


def _synthetic_claim(**overrides):
    """One valid record-level claim used for compatibility controls."""
    fields = dict(
        frame="RX", plane_index=0, reading_axis=0, absolute_quadrant=0,
        source_region_id="a" * 64, reading_identity_sha256="b" * 64,
        support_sha256="c" * 64, face_sha256="d" * 64,
        row=0, column=0,
        world_normal_axis=0, world_normal_sign=1, outward_sign=-1,
        face=0, normal_block=0,
        tangent_u_axis=1, lateral_u=0, tangent_v_axis=2, lateral_v=0,
        coordinate=(0, 0, 0))
    fields.update(overrides)
    return pv.CellClaim.build(**fields)


def _observation(claims):
    ordered = tuple(sorted(claims, key=pv._claim_sort_key))
    frame_ids = tuple(sorted({claim.frame for claim in ordered}))
    return pv.SupportedCellObservation.build(
        coordinate=ordered[0].coordinate, claims=ordered,
        frame_ids=frame_ids, conflict_kind=None)


@pytest.fixture(scope="module")
def recipe_context():
    fixture = recipe_tests._build_recipe_fixture()
    universe = recipe_tests._recipe_universe(fixture)
    supports = recipe_tests._recipe_supports(fixture)
    placement_budget = recipe_tests._budget()
    placement = plane_placement.resolve_face_placement(
        universe=universe, supports=supports, budget=placement_budget)
    assert isinstance(placement, plane_placement.ResolvedFacePlacement), placement
    return universe, supports, placement, placement_budget, _budget()


@pytest.fixture(scope="module")
def recipe_result(recipe_context):
    result = pv.build_plane_volume(**_source_kwargs(recipe_context))
    assert isinstance(result, pv.PlaneVolumeResult), result
    return result


@pytest.fixture(scope="module")
def recipe_preview(recipe_context, recipe_result):
    preview = pv.to_geometry_only_preview(
        recipe_result, **_preview_kwargs(recipe_context))
    assert isinstance(preview, pv.GeometryOnlyPreview), preview
    return preview


# ---------------------------------------------------------------------------
# Exact geometry: one adjacent block per supported cell, no shell, no fill
# ---------------------------------------------------------------------------

def test_recipe_volume_emits_exactly_the_literal_supported_cells(recipe_result):
    """Every emitted coordinate is a declared supported cell and nothing else."""
    contributions = _literal_volume_contributions()
    expected_coordinates = {coordinate for coordinate, _ in contributions}
    assert {cell.coordinate for cell in recipe_result.cells} == expected_coordinates
    observed = sorted((claim.coordinate, (claim.frame, claim.reading_axis,
                                          claim.row, claim.column))
                      for cell in recipe_result.cells for claim in cell.claims)
    assert observed == sorted(contributions)
    assert len(recipe_result.cells) == len(expected_coordinates) == 58
    assert sum(len(cell.claims) for cell in recipe_result.cells) == 66


def test_recipe_volume_work_and_counts_are_exact(recipe_result):
    assert {name: getattr(recipe_result.work, name)
            for name in RECIPE_VOLUME_WORK} == RECIPE_VOLUME_WORK
    assert len(recipe_result.cells) == RECIPE_VOLUME_WORK["coordinates"]
    assert len(recipe_result.conflicts) == 0
    assert len(recipe_result.omitted) == RECIPE_VOLUME_WORK["omitted_unknown"]
    assert recipe_result.complete is False
    assert recipe_result.gauge_state == plane_placement.V2_PLACEMENT_GAUGE_STATE
    assert recipe_result.gauge_axes == (0, 1, 2)


def test_recipe_volume_never_fills_the_bounding_box(recipe_result):
    """58 emitted blocks over a 224-cell dense box: no shell or gap filling."""
    assert recipe_result.cell_bounds == RECIPE_VOLUME_BOUNDS
    assert recipe_result.dense_bounds_cells == 224
    assert len(recipe_result.cells) < recipe_result.dense_bounds_cells
    assert recipe_result.export_bounds is None
    assert recipe_result.normalization_offset == RECIPE_VOLUME_BOUNDS[0]


def test_recipe_volume_bounds_come_from_emitted_cells_only(recipe_result):
    low = tuple(min(cell.coordinate[axis] for cell in recipe_result.cells)
                for axis in range(3))
    high = tuple(max(cell.coordinate[axis] for cell in recipe_result.cells)
                 for axis in range(3))
    assert (low, high) == recipe_result.cell_bounds
    assert any(value < 0 for value in low)
    assert high == (2, 3, 3)
    assert recipe_result.normalization_offset == low
    dense = 1
    for axis in range(3):
        dense *= high[axis] - low[axis] + 1
    assert dense == recipe_result.dense_bounds_cells


# ---------------------------------------------------------------------------
# Provenance, signs, reflections and negative coordinates
# ---------------------------------------------------------------------------

def test_recipe_volume_preserves_real_provenance(recipe_context, recipe_result):
    universe, supports, placement, _, _ = recipe_context
    faces = {face.face_sha256: face for face in placement.faces}
    by_support = {support.support_sha256: support for support in supports}
    assert len(faces) == 9 and len(by_support) == 9
    for cell in recipe_result.cells:
        for claim in cell.claims:
            face = faces[claim.face_sha256]
            support = by_support[claim.support_sha256]
            assert support.reading_identity.identity_sha256 \
                == claim.reading_identity_sha256
            assert claim.frame == face.frame
            assert claim.plane_index == face.plane_index
            assert claim.reading_axis == face.reading_axis
            assert claim.absolute_quadrant == face.absolute_quadrant
            assert claim.source_region_id == face.source_region_id
            assert claim.reading_identity_sha256 == face.reading_identity_sha256
            assert claim.support_sha256 == support.support_sha256
            assert claim.coordinate[claim.world_normal_axis] == face.normal_block
    for binding in recipe_result.support_bindings:
        assert binding["reading_identity_sha256"] in {
            support.reading_identity_sha256 for support in supports}
        assert binding["support_sha256"] in by_support
        assert binding["face_sha256"] in faces


def test_recipe_volume_preserves_both_outward_signs_and_reflections(recipe_context,
                                                                    recipe_result):
    _, _, placement, _, _ = recipe_context
    faces = {face.face_sha256: face for face in placement.faces}
    assert {claim.outward_sign for cell in recipe_result.cells
            for claim in cell.claims} == {1, -1}
    reflected = {face.face_sha256 for face in placement.faces
                 if any(mapping.world_sign < 0 for mapping in face.tangent_mappings)}
    assert reflected
    for cell in recipe_result.cells:
        for claim in cell.claims:
            face = faces[claim.face_sha256]
            mapping_u, mapping_v = face.tangent_mappings
            for cell_record in face.cells:
                if (cell_record.row, cell_record.column) != (claim.row,
                                                             claim.column):
                    continue
                expected_u = pv._cell_lower_endpoint(
                    mapping_u.anchor, mapping_u.world_sign, cell_record.column)
                expected_v = pv._cell_lower_endpoint(
                    mapping_v.anchor, mapping_v.world_sign, cell_record.row)
                assert (claim.lateral_u, claim.lateral_v) == (expected_u, expected_v)
                break
            else:  # pragma: no cover - defensive
                raise AssertionError("claim cell is not on its face")


def test_recipe_volume_keeps_signed_translations_and_negative_coordinates(
        recipe_context, recipe_result):
    _, _, placement, _, _ = recipe_context
    translations = {tuple(Fraction_pair) for Fraction_pair in
                    (tuple(face.translation) for face in placement.faces)}
    assert len(translations) >= 1
    assert any(Fraction_pair[0] == (3, 2) for Fraction_pair in translations)
    assert any(Fraction_pair[0] == (-1, 2) for Fraction_pair in translations)
    assert any(any(value < 0 for value in cell.coordinate)
               for cell in recipe_result.cells)


# ---------------------------------------------------------------------------
# Compatible merges and the reserved conflict type
# ---------------------------------------------------------------------------

def test_recipe_compatible_perpendicular_corners_merge_without_conflict(
        recipe_result):
    """Perpendicular corner and opposite-face claims merge with no conflict."""
    assert recipe_result.conflicts == ()
    assert all(cell.conflict_kind is None for cell in recipe_result.cells)
    merged = {cell.coordinate: cell for cell in recipe_result.cells}
    assert {coordinate: len(cell.claims)
            for coordinate, cell in merged.items()
            if len(cell.claims) > 1} == RECIPE_MERGED_COORDINATES
    for coordinate in RECIPE_MERGED_COORDINATES:
        cell = merged[coordinate]
        assert len({claim.world_normal_axis for claim in cell.claims}) > 1
        assert all(pv._claim_bounds_cell(claim, coordinate)
                   for claim in cell.claims)
    corner = merged[(1, 1, 1)]
    assert len(corner.claims) == 3
    assert corner.frame_ids == ("R0",)
    assert {claim.world_normal_axis for claim in corner.claims} == {0, 1, 2}


def test_perpendicular_corner_claims_are_compatible_evidence():
    claims = (
        _synthetic_claim(world_normal_axis=0, tangent_u_axis=1, tangent_v_axis=2),
        _synthetic_claim(world_normal_axis=1, tangent_u_axis=0, tangent_v_axis=2),
        _synthetic_claim(world_normal_axis=2, tangent_u_axis=0, tangent_v_axis=1),
    )
    assert pv._incompatible_evidence_kind((0, 0, 0), claims) is None
    observation = _observation(claims)
    assert observation.conflict_kind is None
    with pytest.raises(ValueError):
        pv.CellConflict.build(coordinate=(0, 0, 0), kind="perpendicular_faces",
                              claims=observation.claims,
                              frame_ids=observation.frame_ids)


def test_opposite_faces_of_one_cube_are_compatible_evidence():
    """The two opposite faces of one cube bound that cube and merge cleanly."""
    claims = (
        _synthetic_claim(outward_sign=-1, face=0, normal_block=0),
        _synthetic_claim(outward_sign=1, face=1, normal_block=0),
    )
    assert pv._incompatible_evidence_kind((0, 0, 0), claims) is None
    observation = _observation(claims)
    assert observation.conflict_kind is None
    with pytest.raises(ValueError):
        pv.CellConflict.build(coordinate=(0, 0, 0), kind="opposed_sides",
                              claims=observation.claims,
                              frame_ids=observation.frame_ids)


def test_cell_conflict_type_is_preserved_and_reserved():
    assert "CellConflict" in pv.__all__
    assert pv.V3_VOLUME_CONFLICT_KINDS == ("opposed_sides", "perpendicular_faces")
    # A conflict record is a real, validated type, but it is reserved for
    # explicitly incompatible evidence: compatible claims cannot fabricate one.
    claims = (
        _synthetic_claim(outward_sign=-1, face=0, normal_block=0),
        _synthetic_claim(outward_sign=1, face=1, normal_block=0),
    )
    observation = _observation(claims)
    for kind in pv.V3_VOLUME_CONFLICT_KINDS:
        with pytest.raises(ValueError):
            pv.CellConflict.build(coordinate=(0, 0, 0), kind=kind,
                                  claims=observation.claims,
                                  frame_ids=observation.frame_ids)


def test_recipe_volume_omits_unknown_cells_explicitly(recipe_result):
    omitted = {(record.frame, record.reading_axis, record.row, record.column): record
               for record in recipe_result.omitted}
    assert len(omitted) == 15
    for frame in RECIPE_FRAMES:
        for row, column in RECIPE_UNKNOWN_GRID[frame]:
            # Each frame carries three normal-axis readings with the same
            # declared unknown/supported split.
            for axis in RECIPE_AXES:
                record = omitted[(frame, axis, row, column)]
                assert record.state == "unknown"
                assert record.evidence in v1.V1_SUPPORT_CELL_EVIDENCE
                assert record.evidence != "full"
                assert record.face_sha256
        for row, column in RECIPE_SUPPORTED_GRID[frame]:
            assert not any(key[0] == frame and key[2] == row and key[3] == column
                           for key in omitted)
    # An omitted cell contributes no coordinate and no air block: only the 66
    # genuinely supported cells became 58 merged blocks.
    assert sum(len(cell.claims) for cell in recipe_result.cells) == 66
    assert len(recipe_result.cells) == 58


# ---------------------------------------------------------------------------
# Determinism, canonical order and JSON safety
# ---------------------------------------------------------------------------

def test_recipe_volume_is_permutation_invariant(recipe_context, recipe_result):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    reordered = pv.build_plane_volume(
        universe=universe, supports=tuple(reversed(supports)),
        placement=placement, placement_budget=placement_budget, budget=budget)
    assert isinstance(reordered, pv.PlaneVolumeResult), reordered
    assert reordered.volume_sha256 == recipe_result.volume_sha256
    assert reordered.to_record() == recipe_result.to_record()


def test_canonical_volume_order_is_deterministic(recipe_result):
    refusal = pv._make_refusal("invalid_input", {"field": "x"})
    forward = pv.canonical_volume_order([recipe_result, refusal])
    backward = pv.canonical_volume_order([refusal, recipe_result])
    assert forward == backward
    assert pv.volume_sort_key(recipe_result)[0] == "volume"
    assert pv.volume_sort_key(refusal)[0] == "refusal"


def test_result_record_is_json_safe(recipe_result):
    record = recipe_result.to_record()
    encoded = json.dumps(record, sort_keys=True)
    assert json.loads(encoded)["volume_sha256"] == recipe_result.volume_sha256
    assert record["complete"] is False
    assert record["export_bounds"] is None


# ---------------------------------------------------------------------------
# Source-bound placement replay: the Astra +100 reseal must refuse
# ---------------------------------------------------------------------------

def test_freshly_resealed_plus100_translation_is_refused(recipe_context):
    """A self-consistent +100 normal reseal with original bindings refuses."""
    context = recipe_context
    _, _, placement, _, _ = context
    resealed = _reseal_face_translation(placement, 0, delta=100)
    assert resealed.placement_sha256 != placement.placement_sha256
    outcome = pv.build_plane_volume(
        **(_source_kwargs(context) | {"placement": resealed}))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"
    assert outcome.work is None
    assert "cells" not in outcome.to_record()


def test_stale_placement_digest_is_refused(recipe_context):
    """Even a resealed face cannot ride along under the original digest."""
    context = recipe_context
    _, _, placement, _, _ = context
    resealed = _reseal_face_translation(placement, 0, delta=100)
    stale = _clone(resealed, placement_sha256=placement.placement_sha256)
    outcome = pv.build_plane_volume(
        **(_source_kwargs(context) | {"placement": stale}))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"
    assert outcome.work is None


def test_resealed_support_cell_is_rejected(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    broken = list(supports)
    target = broken[0]
    cells = list(target.cells)
    cells[-1] = _clone(cells[-1], row=999)
    broken[0] = _clone(target, cells=tuple(cells))
    outcome = pv.build_plane_volume(
        universe=universe, supports=tuple(broken), placement=placement,
        placement_budget=placement_budget, budget=budget)
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "support_replay_mismatch"
    assert outcome.work is None
    assert "cells" not in outcome.to_record()


def test_resealed_face_cell_is_rejected(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    faces = list(placement.faces)
    target = faces[0]
    cells = list(target.cells)
    cells[0] = _clone(cells[0], lateral_u=int(cells[0].lateral_u) + 1)
    faces[0] = _clone(target, cells=tuple(cells))
    broken = _clone(placement, faces=tuple(faces))
    outcome = pv.build_plane_volume(
        universe=universe, supports=supports, placement=broken,
        placement_budget=placement_budget, budget=budget)
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"
    assert outcome.work is None


def test_resealed_placement_record_is_rejected(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    broken = _clone(placement, component_bindings=({"foreign": 1},))
    outcome = pv.build_plane_volume(
        universe=universe, supports=supports, placement=broken,
        placement_budget=placement_budget, budget=budget)
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"


def test_missing_and_duplicate_supports_are_refused(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    missing = pv.build_plane_volume(
        universe=universe, supports=supports[:-1], placement=placement,
        placement_budget=placement_budget, budget=budget)
    assert isinstance(missing, pv.VolumeRefusal)
    assert missing.reason == "missing_provenance"
    duplicate = pv.build_plane_volume(
        universe=universe, supports=tuple(supports) + (supports[0],),
        placement=placement, placement_budget=placement_budget, budget=budget)
    assert isinstance(duplicate, pv.VolumeRefusal)
    assert duplicate.reason == "provenance_mismatch"


def test_invalid_input_precedes_budget(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    target = supports[0]
    cells = list(target.cells)
    cells[-1] = _clone(cells[-1], row=999)
    broken = list(supports)
    broken[0] = _clone(target, cells=tuple(cells))
    outcome = pv.build_plane_volume(
        universe=universe, supports=tuple(broken), placement=placement,
        placement_budget=placement_budget,
        budget=_budget(max_faces=1))
    assert isinstance(outcome, pv.VolumeRefusal)
    assert outcome.reason == "support_replay_mismatch"
    assert outcome.work is None


def test_placement_budget_refusal_precedes_volume_budget(recipe_context):
    """A placement rerun that exhausts its own budget refuses without cells."""
    context = recipe_context
    universe, supports, placement, _, budget = context
    outcome = pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=recipe_tests._budget(max_faces=1),
        budget=_budget(max_faces=1))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "placement_replay_refused"
    assert outcome.work is None


def test_result_digest_rejects_a_resealed_cell(recipe_result):
    cells = list(recipe_result.cells)
    cells[0] = _clone(cells[0], frame_ids=("foreign",))
    fields = {field.name: getattr(recipe_result, field.name)
              for field in dataclasses.fields(recipe_result)}
    fields["cells"] = tuple(cells)
    fields.pop("volume_sha256")
    with pytest.raises(ValueError):
        pv.PlaneVolumeResult.build(**fields)


# ---------------------------------------------------------------------------
# Atomic budgets and the dense-bounds preflight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cap,observed", [
    ("max_faces", 9),
    ("max_support_cells", 81),
    ("max_contributions", 66),
    ("max_coordinates", 58),
    ("max_omitted_unknown", 15),
])
def test_each_work_cap_is_atomic_and_exact(recipe_context, cap, observed):
    context = recipe_context
    exact = pv.build_plane_volume(**(_source_kwargs(context) | {
        "budget": _budget(**{cap: observed})}))
    assert isinstance(exact, pv.PlaneVolumeResult), exact
    refusal = pv.build_plane_volume(**(_source_kwargs(context) | {
        "budget": _budget(**{cap: observed - 1})}))
    assert isinstance(refusal, pv.VolumeRefusal), refusal
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail == {"cap": cap, "limit": observed - 1,
                              "observed": observed}
    assert refusal.work is not None
    assert getattr(refusal.work, pv._CAP_TO_WORK[cap]) == observed
    assert "cells" not in refusal.to_record()
    assert isinstance(refusal.to_record()["work"], dict)


def test_conflict_cap_minimum_is_valid_with_empty_conflicts(recipe_context,
                                                            recipe_result):
    """A valid empty conflict tuple is accepted under the minimal cap."""
    assert recipe_result.conflicts == ()
    assert recipe_result.work.conflicts == 0
    exact = pv.build_plane_volume(
        **(_source_kwargs(recipe_context) | {"budget": _budget(max_conflicts=1)}))
    assert isinstance(exact, pv.PlaneVolumeResult)
    assert exact.work.conflicts == 0
    assert exact.to_record() == recipe_result.to_record()


def test_dense_bounds_preflight_is_typed_and_atomic(recipe_context):
    context = recipe_context
    exact = pv.build_plane_volume(**(_source_kwargs(context) | {
        "budget": _budget(max_dense_bounds_cells=224)}))
    assert isinstance(exact, pv.PlaneVolumeResult)
    refusal = pv.build_plane_volume(**(_source_kwargs(context) | {
        "budget": _budget(max_dense_bounds_cells=223)}))
    assert isinstance(refusal, pv.VolumeRefusal), refusal
    assert refusal.reason == "dense_bounds_exceeded"
    assert refusal.detail == {"cap": "max_dense_bounds_cells", "limit": 223,
                              "observed": 224}
    assert refusal.work is not None
    assert refusal.work.dense_bounds_cells == 224
    assert "cells" not in refusal.to_record()


def test_volume_budget_rejects_non_exact_caps():
    with pytest.raises(ValueError):
        pv.make_volume_budget(max_faces=9)
    with pytest.raises(ValueError):
        pv.make_volume_budget(
            max_faces=0, max_support_cells=1, max_contributions=1,
            max_coordinates=1, max_conflicts=1, max_omitted_unknown=1,
            max_dense_bounds_cells=1)
    with pytest.raises(ValueError):
        pv.make_volume_budget(
            max_faces=9.0, max_support_cells=1, max_contributions=1,
            max_coordinates=1, max_conflicts=1, max_omitted_unknown=1,
            max_dense_bounds_cells=1)


def test_invalid_inputs_are_refused(recipe_context):
    context = recipe_context
    universe, supports, placement, placement_budget, budget = context
    assert pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=None).reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=None, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=budget).reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=None, budget=budget).reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=universe, supports=supports, placement=None,
        placement_budget=placement_budget, budget=budget).reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=universe, supports=(), placement=placement,
        placement_budget=placement_budget, budget=budget).reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=budget,
        producer_version="nope").reason == "invalid_input"
    assert pv.build_plane_volume(
        universe=universe, supports=("not-a-support",), placement=placement,
        placement_budget=placement_budget, budget=budget).reason == "invalid_input"


# ---------------------------------------------------------------------------
# Explicit geometry-only immutable preview adapter
# ---------------------------------------------------------------------------

def test_preview_requires_the_explicit_geometry_only_flag(recipe_context,
                                                          recipe_result):
    for value in (False, None, 1, "yes"):
        outcome = pv.to_geometry_only_preview(
            recipe_result, **_source_kwargs(recipe_context),
            preview_state=PREVIEW_STATE, geometry_only=value)
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "geometry_only_required"
    assert pv.to_geometry_only_preview(
        recipe_result, **_source_kwargs(recipe_context),
        preview_state="stone", geometry_only=True).reason == "preview_state_invalid"


def test_preview_forbids_air_states(recipe_context, recipe_result):
    for name in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air"):
        outcome = pv.to_geometry_only_preview(
            recipe_result, **_source_kwargs(recipe_context),
            preview_state=BlockState(name), geometry_only=True)
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "air_state_forbidden"


def test_preview_marks_inferred_zero_confidence_and_ineligible(recipe_result,
                                                               recipe_preview):
    assert recipe_preview.exact_state_eligible is False
    assert recipe_preview.geometry_only is True
    assert recipe_preview.export_bounds is None
    assert recipe_preview.preview_state == PREVIEW_STATE
    assert len(recipe_preview.blocks) == 58
    cells = {cell.coordinate: cell for cell in recipe_result.cells}
    assert {block.coordinate for block in recipe_preview.blocks} == set(cells)
    volume = recipe_preview.to_block_volume()
    assert isinstance(volume, BlockVolume)
    assert set(volume.blocks) == set(cells)
    for coordinate, observation in volume.items():
        assert observation.state == PREVIEW_STATE
        assert observation.kind is ObservationKind.INFERRED
        assert observation.confidence == 0.0
        assert tuple(observation.frame_ids) == cells[coordinate].frame_ids
    assert recipe_preview.normalization_offset == RECIPE_VOLUME_BOUNDS[0]


def test_preview_is_immutable_and_exposes_no_mutable_blockvolume(recipe_preview):
    assert "volume" not in {field.name
                            for field in dataclasses.fields(recipe_preview)}
    assert not hasattr(recipe_preview, "volume")
    assert "volume" not in recipe_preview.to_record()
    before = recipe_preview.to_record()
    first = recipe_preview.to_block_volume()
    first.set((99, 99, 99), next(iter(first.blocks.values())))
    second = recipe_preview.to_block_volume()
    assert (99, 99, 99) not in second.blocks
    assert recipe_preview.to_record() == before


def test_preview_replay_context_refuses_a_stale_result(recipe_context,
                                                       recipe_result):
    context = recipe_context
    stale = _clone(recipe_result, volume_sha256="0" * 64)
    outcome = pv.to_geometry_only_preview(stale, **_preview_kwargs(context))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"
    assert outcome.work is None


def test_preview_replay_context_refuses_a_phantom_coordinate(recipe_context,
                                                             recipe_result):
    context = recipe_context
    phantom = _clone(recipe_result.cells[0], coordinate=(9, 9, 9))
    tampered = _reseal_result(
        recipe_result, cells=tuple(recipe_result.cells) + (phantom,))
    assert tampered.volume_sha256 != recipe_result.volume_sha256
    outcome = pv.to_geometry_only_preview(tampered, **_preview_kwargs(context))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"


def test_preview_replay_context_refuses_a_resealed_placement(recipe_context,
                                                             recipe_result):
    context = recipe_context
    _, _, placement, _, _ = context
    resealed = _reseal_face_translation(placement, 0, delta=100)
    outcome = pv.to_geometry_only_preview(
        recipe_result, **(_preview_kwargs(context) | {"placement": resealed}))
    assert isinstance(outcome, pv.VolumeRefusal), outcome
    assert outcome.reason == "replay_mismatch"


def test_preview_does_not_mutate_the_result(recipe_context, recipe_result):
    before = recipe_result.to_record()
    preview = pv.to_geometry_only_preview(
        recipe_result, **_preview_kwargs(recipe_context))
    assert isinstance(preview, pv.GeometryOnlyPreview)
    assert recipe_result.to_record() == before
    preview.to_block_volume().set((99, 99, 99),
                                  next(iter(preview.to_block_volume().blocks.values())))
    assert recipe_result.to_record() == before


# ---------------------------------------------------------------------------
# Final adapter hardening: immutable state snapshot, nested validation, order
# ---------------------------------------------------------------------------

def test_preview_snapshots_list_backed_state_immutably(recipe_context,
                                                       recipe_result):
    """A list-backed preview state is copied, so later mutation cannot leak in."""
    state = BlockState("minecraft:stone", [["facing", "north"]])
    assert type(state.properties) is list
    preview = pv.to_geometry_only_preview(
        recipe_result,
        **(_preview_kwargs(recipe_context) | {"preview_state": state}))
    assert isinstance(preview, pv.GeometryOnlyPreview), preview
    # The sealed record stores an immutable name+tuple-properties snapshot.
    assert type(preview.preview_state_name) is str
    assert type(preview.preview_state_properties) is tuple
    assert preview.preview_state_properties == (("facing", "north"),)
    assert preview.preview_state == BlockState(
        "minecraft:stone", (("facing", "north"),))
    before = preview.to_record()
    # A caller mutation of the original list-backed state must not reach the
    # sealed record or its exported observations.
    state.properties[0][1] = "south"
    state.properties.append(["waterlogged", "true"])
    assert preview.to_record() == before
    assert preview.preview_sha256 == before["preview_sha256"]
    exported = preview.to_block_volume()
    for observation in exported.blocks.values():
        assert observation.state.name == "minecraft:stone"
        assert observation.state.properties == (("facing", "north"),)


def test_preview_rejects_malformed_nested_state(recipe_context, recipe_result):
    """Genuinely malformed nested states are typed refusals, never exceptions."""
    malformed = (
        _clone(PREVIEW_STATE, properties=None),
        _clone(PREVIEW_STATE, properties="facing"),
        _clone(PREVIEW_STATE, properties=[["facing"]]),
        _clone(PREVIEW_STATE, properties=(("facing",),)),
        _clone(PREVIEW_STATE, properties=(("facing", ""),)),
        _clone(PREVIEW_STATE, properties=((1, "north"),)),
        _clone(PREVIEW_STATE, properties=(("facing", 2),)),
        _clone(PREVIEW_STATE, name=None),
        _clone(PREVIEW_STATE, name="stone"),
    )
    for state in malformed:
        outcome = pv.to_geometry_only_preview(
            recipe_result, **_source_kwargs(recipe_context),
            preview_state=state, geometry_only=True)
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "preview_state_invalid"
        assert outcome.work is None


def test_preview_record_rejects_list_backed_properties():
    with pytest.raises(ValueError):
        pv.GeometryOnlyPreview.build(
            schema_version=pv.V3_VOLUME_SCHEMA,
            producer_version=pv.V3_VOLUME_PRODUCER_VERSION,
            preview_state_name="minecraft:stone",
            preview_state_properties=[["facing", "north"]],
            blocks=(), normalization_offset=None, export_bounds=None,
            geometry_only=True, exact_state_eligible=False,
            volume_sha256="a" * 64)


def test_preview_phantom_with_tiny_budget_refuses_replay_before_budget(
        recipe_context, recipe_result):
    """A fully valid phantom result + tiny budget is a typed replay refusal."""
    context = recipe_context
    phantom = _valid_phantom_result(recipe_result)
    # The phantom record is self-consistent, so only the source-bound replay
    # can reject it; a tiny budget must not pre-empt that with budget_exhausted.
    assert phantom.volume_sha256 != recipe_result.volume_sha256
    for cap in ("max_faces", "max_contributions", "max_dense_bounds_cells"):
        outcome = pv.to_geometry_only_preview(
            phantom, **(_source_kwargs(context) | {"budget": _budget(**{cap: 1})}),
            preview_state=PREVIEW_STATE, geometry_only=True)
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "replay_mismatch"
        assert outcome.work is None
        assert "cells" not in outcome.to_record()


def test_preview_malformed_result_data_is_a_typed_refusal(recipe_context,
                                                          recipe_result):
    """Malformed nested result data cannot raise AttributeError out of the API."""
    context = recipe_context
    for broken in (
        _clone(recipe_result, cells=(None,)),
        _clone(recipe_result, cells=("not-a-cell",)),
        _clone(recipe_result, cells=tuple(recipe_result.cells) + (None,)),
    ):
        outcome = pv.to_geometry_only_preview(
            broken, **_preview_kwargs(context))
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "replay_mismatch"
        assert outcome.work is None
        assert "cells" not in outcome.to_record()


def test_preview_valid_result_still_applies_the_volume_budget(recipe_context,
                                                              recipe_result):
    """After result validation the preview budget matches the builder exactly."""
    context = recipe_context
    for cap in ("max_faces", "max_contributions", "max_dense_bounds_cells"):
        budget = _budget(**{cap: 1})
        preview = pv.to_geometry_only_preview(
            recipe_result, **(_source_kwargs(context) | {"budget": budget}),
            preview_state=PREVIEW_STATE, geometry_only=True)
        builder = pv.build_plane_volume(
            **(_source_kwargs(context) | {"budget": budget}))
        assert isinstance(preview, pv.VolumeRefusal), preview
        assert isinstance(builder, pv.VolumeRefusal), builder
        assert preview.reason == builder.reason
        assert preview.detail == builder.detail
        assert preview.to_record() == builder.to_record()
        assert preview.work == builder.work
    exact = pv.to_geometry_only_preview(
        recipe_result, **_preview_kwargs(context))
    assert isinstance(exact, pv.GeometryOnlyPreview), exact


def test_preview_state_cannot_be_mutated_into_an_air_claim(recipe_context,
                                                          recipe_result):
    """The air refusal reads the sealed snapshot, not a live state reference."""
    context = recipe_context
    for name in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air"):
        outcome = pv.to_geometry_only_preview(
            recipe_result, **_source_kwargs(context),
            preview_state=BlockState(name, [["foo", "bar"]]), geometry_only=True)
        assert isinstance(outcome, pv.VolumeRefusal), outcome
        assert outcome.reason == "air_state_forbidden"


def test_exporter_smoke_uses_bounds_none_and_derived_normalization(tmp_path,
                                                                   recipe_preview):
    """Exporter smoke belongs to the tests; the bridge imports no exporter."""
    assert recipe_preview.normalization_offset == (-4, 0, -4)
    destination = tmp_path / "v3-preview.litematic"
    LitematicExporter().export(recipe_preview.to_block_volume(), destination,
                               bounds=None)
    assert destination.exists()
    import nbtlib

    document = nbtlib.load(str(destination))
    region = document["Regions"]["scan"]
    size = tuple(int(region["Size"][key]) for key in ("x", "y", "z"))
    assert size == (7, 4, 8)
    assert int(document["Metadata"]["TotalBlocks"]) == 58
    names = {str(entry["Name"])
             for entry in region["BlockStatePalette"]}
    assert PREVIEW_STATE.name in names
    assert "minecraft:air" in names


# ---------------------------------------------------------------------------
# Isolation boundary
# ---------------------------------------------------------------------------

def test_module_imports_stay_inside_the_boundary():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "dataclasses", "fractions", "hashlib",
                       "json", "math", "typing", "mc_photogrammetry"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "export" not in node.module
            assert "cli" not in node.module
            assert "ui" not in node.module
            assert "project" not in node.module
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not (calls & {"open", "exec", "eval", "compile", "__import__"})


def test_module_exposes_no_export_cli_or_manual_symbols():
    for name in dir(pv):
        lowered = name.lower()
        assert "litematic" not in lowered
        assert "cli" not in lowered
        assert "manifest" not in lowered
    assert not hasattr(pv, "LitematicExporter")
    assert not hasattr(pv, "Path")
    assert hasattr(pv, "build_plane_volume")
    assert hasattr(pv, "to_geometry_only_preview")
    assert hasattr(pv, "CellConflict")
    assert hasattr(pv, "PreviewBlock")
    assert pv.V3_VOLUME_COMPLETE is False
