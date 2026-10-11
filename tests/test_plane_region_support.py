"""Independent controls for the isolated M4 V1 phase-aligned region support.

Every fixture is declared by hand: masks, polygons, the rectification calibration
and the geometry that makes the projected cell grid exactly one image pixel per
cell.  Expected cell states, exact excluded-area evidence and work counters are
computed from the fixture, never read back from the module under test.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_region_support as support


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "mc_photogrammetry" / "reconstruction"
          / "plane_region_support.py")
UNIVERSE = hashlib.sha256(b"support-universe").hexdigest()
ORDER = hashlib.sha256(b"support-order").hexdigest()
IMAGE_SHA = hashlib.sha256(b"support-image").hexdigest()
OTHER_SHA = hashlib.sha256(b"support-other").hexdigest()
PHASE_SHA = hashlib.sha256(b"support-phase").hexdigest()
ABSENT_SHA = "a" * 64

ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "hashlib", "json", "math", "dataclasses", "fractions",
    "typing", "numpy", "mc_photogrammetry",
})
ALLOWED_RECONSTRUCTION_MODULES = frozenset({
    "lattice_planes", "plane_endpoint_provenance",
})
DENIED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess",
    "socket", "pathlib", "shutil", "io", "tempfile", "ctypes",
    "plane_phase_anchor", "plane_volume", "plane_placement", "BlockVolume",
    "ReconstructionResult",
})


# ---------------------------------------------------------------------------
# Fixtures: one image pixel per projected unit cell
# ---------------------------------------------------------------------------

def _budget(**overrides):
    limits = {
        "max_cells": 4096,
        "max_region_pixel_tests": 1_000_000,
        "max_usable_pixel_tests": 1_000_000,
        "max_polygon_vertices": 1_000_000,
        "max_supported_cells": 4096,
    }
    limits.update(overrides)
    return support.make_support_budget(**limits)


def _mask(shape, true_cells):
    result = np.zeros(shape, dtype=bool)
    for row, column in true_cells:
        result[row, column] = True
    return result


def _polygon(x0, y0, x1, y1):
    return ((float(x0), float(y0)), (float(x1), float(y0)),
            (float(x1), float(y1)), (float(x0), float(y1)))


def _unit_axis_matrix(axis, side=1.0):
    axes = np.zeros((3, 3), dtype=np.float64)
    axes[axis, 2] = float(side)
    for camera_axis, local_label in enumerate(i for i in range(3) if i != axis):
        axes[local_label, camera_axis] = 1.0
    return axes


def _reader(mask, polygon, *, axis=2, side=1.0, distance=1.0, focal=1.0,
            principal=None, b=1.0, frame="frame-a",
            sources=("synthetic-region",), alternatives=None, axis_matrix=None):
    height = int(mask.shape[0])
    if principal is None:
        principal = (0.0, float(height))
    axes = (_unit_axis_matrix(axis, side) if axis_matrix is None
            else np.asarray(axis_matrix, dtype=np.float64))
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=tuple(tuple(float(value) for value in row) for row in axes),
        focal=float(focal),
        principal_point=(float(principal[0]), float(principal[1])),
        centre=(0.0, height / 2.0))
    evidence = lattice_planes._make_region_evidence(
        mask, axis, polygon, tuple(sources))
    if alternatives is None:
        alternatives = (lattice_planes.DistanceAlternative(
            float(b), float(distance), 1.0),)
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=float(side),
        alternatives=tuple(alternatives), period_pixels=1.0,
        phase_pixels=(0.0, 0.0), extent=1.0, samples=8, polygon=polygon,
        coverage_cells=int(np.count_nonzero(mask)), structural_share=1.0,
        confidence=1.0, rectification=rectification, region_evidence=evidence)


def _bare_reader(frame="frame-a", axis=2):
    axes = np.eye(3, dtype=np.float64)
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=tuple(tuple(float(value) for value in row) for row in axes),
        focal=1.0, principal_point=(0.0, 4.0), centre=(0.0, 2.0))
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=1.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=8,
        polygon=None, coverage_cells=0, structural_share=0.0, confidence=0.0,
        rectification=rectification, region_evidence=None)


def _canonical(reading):
    """The reading with alternatives in the V1 canonical content order."""
    ordered = support.canonical_alternative_order(reading.alternatives)
    if ordered == tuple(reading.alternatives):
        return reading
    return replace(reading, alternatives=ordered)


def _universe(reading):
    return (_canonical(reading),)


def _identity(reading, *, universe=None, frame=None, source_region_id=None,
              axis=None, index=None, key_sha=None, universe_sha=None):
    """A genuinely content-derived V2-A identity for ``reading``.

    Every digest is recomputed from the (canonicalized) reading and universe by
    the accepted V2-A helpers; no dummy identity hash is used anywhere.
    """
    canonical = _canonical(reading)
    if universe is None:
        universe = (canonical,)
    universe = tuple(_canonical(item) for item in universe)
    ordered = v2a._ordered_readings(universe)
    if index is None:
        index = v2a._canonical_position(
            canonical, v2a._canonical_index_lookup(ordered))
    if key_sha is None:
        key_sha = v2a._reading_order_key_sha256(canonical)
    if universe_sha is None:
        universe_sha = v2a._canonical_universe_sha256(universe)
    if frame is None:
        frame = canonical.frame
    if axis is None:
        axis = canonical.axis
    if source_region_id is None:
        evidence = canonical.region_evidence
        source_region_id = (None if evidence is None
                            else evidence.source_region_id)
    return v2a.ReadingIdentity.build(
        frame=frame, canonical_index=index,
        reading_order_key_sha256=key_sha, reading_axis=axis,
        source_region_id=source_region_id,
        canonical_universe_sha256=universe_sha)


def _build(reading, identity, usable, *, universe=None, origin=(0.0, 0.0),
           grid_shape=(1, 1), b=1.0, budget=None, image_sha256=IMAGE_SHA,
           phase_anchor=None,
           producer_version=support.V1_SUPPORT_PRODUCER_VERSION):
    if universe is None:
        universe = _universe(reading)
    return support.build_aligned_region_support(
        reading=reading, reading_identity=identity, reading_universe=universe,
        usable_mask=usable, image_sha256=image_sha256, blocks_per_period=b,
        origin_uv=origin, grid_shape=grid_shape,
        budget=_budget() if budget is None else budget,
        phase_anchor_sha256=phase_anchor, producer_version=producer_version)


def _reseal_refusal(reason, detail, work=None):
    """Independently reseal a refusal so only semantic invariants can reject it."""
    encoded = json.dumps(detail, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    detail_sha = hashlib.sha256(encoded).hexdigest()
    payload = {
        "reason": reason, "detail_sha256": detail_sha,
        "work": None if work is None else work.to_record(),
    }
    refusal_sha = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()
    return support.AlignedSupportRefusal(reason, detail, detail_sha, work,
                                         refusal_sha)


def _rect_support():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    polygon = _polygon(0, 2, 2, 4)
    reading = _reader(mask, polygon)
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    grid_shape=(2, 2))
    assert isinstance(result, support.AlignedRegionSupport)
    return reading, result


def _support_fields(result):
    """Every constructor field of ``result`` except the two recomputed digests."""
    return {
        "schema_version": result.schema_version,
        "producer_version": result.producer_version,
        "coverage_policy": result.coverage_policy,
        "source_region_id": result.source_region_id,
        "region_mask_sha256": result.region_mask_sha256,
        "image_sha256": result.image_sha256,
        "image_shape": result.image_shape,
        "usable_mask_sha256": result.usable_mask_sha256,
        "reading_identity": result.reading_identity,
        "reading_identity_sha256": result.reading_identity_sha256,
        "calibration": result.calibration,
        "alternative": result.alternative,
        "origin_uv": result.origin_uv,
        "grid_shape": result.grid_shape,
        "row_label": result.row_label,
        "column_label": result.column_label,
        "row_convention": result.row_convention,
        "column_convention": result.column_convention,
        "phase_anchor_sha256": result.phase_anchor_sha256,
        "cells": result.cells,
        "work": result.work,
    }


# ---------------------------------------------------------------------------
# Basic supported map and exact counters
# ---------------------------------------------------------------------------

def test_rectangular_mask_supports_only_fully_covered_cells_with_exact_work():
    reading, result = _rect_support()
    assert [cell.state for cell in result.cells] == ["supported"] * 4
    assert [cell.evidence for cell in result.cells] == ["full"] * 4
    assert [cell.cell_area for cell in result.cells] == [(1, 1)] * 4
    assert [cell.region_excluded for cell in result.cells] == [(0, 1)] * 4
    assert [cell.usable_excluded for cell in result.cells] == [(0, 1)] * 4
    assert [cell.frame_excluded for cell in result.cells] == [(0, 1)] * 4
    work = result.work
    assert work.cells_examined == 4
    assert work.frame_clips == 4
    assert work.region_pixel_tests == 4
    assert work.usable_pixel_tests == 4
    assert work.polygon_vertices == 32
    assert work.supported_cells == 4
    assert work.unknown_cells == 0
    assert result.supported_count == 4 and result.unknown_count == 0


def test_support_record_binds_every_declared_provenance_field():
    reading, result = _rect_support()
    evidence = reading.region_evidence
    assert result.schema_version == support.V1_SUPPORT_SCHEMA
    assert result.producer_version == support.V1_SUPPORT_PRODUCER_VERSION
    assert result.coverage_policy == support.V1_SUPPORT_COVERAGE_POLICY
    assert result.source_region_id == evidence.source_region_id
    assert result.region_mask_sha256 == evidence.mask_sha256
    assert result.image_sha256 == IMAGE_SHA
    assert result.image_shape == (4, 4)
    assert result.usable_mask_sha256 == support.usable_mask_sha256(
        np.ones((4, 4), dtype=bool))
    assert result.reading_identity_sha256 == result.reading_identity.identity_sha256
    assert result.reading_identity.frame == "frame-a"
    assert len(result.reading_binding_sha256) == 64
    assert result.calibration.calibration_sha256 == \
        result.calibration.to_record()["calibration_sha256"]
    assert result.alternative.alternative_sha256 == \
        result.alternative.to_record()["alternative_sha256"]
    assert result.origin_uv == ((0, 1), (0, 1))
    assert result.grid_shape == (2, 2)
    assert (result.column_label, result.column_convention) == (0, "u")
    assert (result.row_label, result.row_convention) == (1, "v")
    assert result.phase_anchor_sha256 is None
    record = result.to_record()
    assert json.dumps(record, sort_keys=True) == json.dumps(
        result.to_record(), sort_keys=True)
    assert record["support_sha256"] == result.support_sha256
    assert record["reading_binding_sha256"] == result.reading_binding_sha256


def test_phase_anchor_binding_changes_the_canonical_digest():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    reading = _reader(mask, _polygon(0, 2, 2, 4))
    identity = _identity(reading)
    usable = np.ones((4, 4), dtype=bool)
    without = _build(reading, identity, usable, grid_shape=(2, 2))
    with_anchor = _build(reading, identity, usable, grid_shape=(2, 2),
                         phase_anchor=PHASE_SHA)
    assert with_anchor.phase_anchor_sha256 == PHASE_SHA
    assert with_anchor.support_sha256 != without.support_sha256
    assert with_anchor.cells == without.cells


def test_tangent_axis_labels_follow_the_shared_sorted_convention():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    expected = {2: (0, 1), 0: (1, 2), 1: (0, 2)}
    for axis, (column_label, row_label) in expected.items():
        reading = _reader(mask, _polygon(0, 2, 2, 4), axis=axis)
        result = _build(reading, _identity(reading),
                        np.ones((4, 4), dtype=bool), grid_shape=(2, 2))
        assert isinstance(result, support.AlignedRegionSupport)
        assert (result.column_label, result.row_label) == (column_label, row_label)


# ---------------------------------------------------------------------------
# Conservative policy: boundary, partial, holes, exclusions
# ---------------------------------------------------------------------------

def test_only_interior_fully_covered_cells_are_supported():
    interior = _mask((4, 4), [(1, 1), (1, 2), (2, 1), (2, 2)])
    reading = _reader(interior, _polygon(1, 1, 3, 3))
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    grid_shape=(4, 4))
    supported = {(cell.row, cell.column) for cell in result.cells
                 if cell.state == "supported"}
    assert supported == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert result.unknown_count == 12
    for cell in result.cells:
        if (cell.row, cell.column) in supported:
            continue
        assert cell.state == "unknown"
        assert cell.region_excluded > (0, 1)
        assert cell.region_excluded == cell.cell_area


def test_fractional_origin_retains_partial_area_as_unknown_evidence():
    mask = _mask((4, 4), [(3, 0)])
    reading = _reader(mask, _polygon(0, 3, 1, 4))
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    origin=(0.5, 0.0), grid_shape=(1, 1))
    assert result.origin_uv == ((1, 2), (0, 1))
    cell = result.cells[0]
    assert cell.state == "unknown" and cell.evidence == "outside_region"
    assert cell.cell_area == (1, 1)
    assert cell.region_excluded == (1, 2)
    assert cell.frame_excluded == (0, 1)
    assert cell.usable_excluded == (0, 1)


def test_hole_in_the_retained_mask_is_never_filled():
    holed = np.ones((4, 4), dtype=bool)
    holed[2, 2] = False
    reading = _reader(holed, _polygon(0, 0, 4, 4))
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    grid_shape=(4, 4))
    states = {(cell.row, cell.column): cell for cell in result.cells}
    hole = states[(1, 2)]
    assert hole.state == "unknown" and hole.evidence == "outside_region"
    assert hole.region_excluded == (1, 1)
    assert result.supported_count == 15


def test_l_shaped_mask_marks_the_missing_corner_unknown():
    l_shape = _mask((4, 4), [(1, 1), (2, 1), (2, 2)])
    reading = _reader(l_shape, _polygon(1, 1, 3, 3))
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    grid_shape=(4, 4))
    states = {(cell.row, cell.column): cell.state for cell in result.cells}
    assert states[(2, 1)] == "supported"
    assert states[(1, 1)] == "supported"
    assert states[(1, 2)] == "supported"
    assert states[(2, 2)] == "unknown"
    assert result.supported_count == 3


def test_disconnected_mask_keeps_the_gap_unknown():
    disconnected = _mask((6, 6), [(1, 1), (1, 4)])
    reading = _reader(disconnected, _polygon(1, 1, 5, 2))
    result = _build(reading, _identity(reading), np.ones((6, 6), dtype=bool),
                    grid_shape=(5, 5))
    states = {(cell.row, cell.column): cell.state for cell in result.cells}
    assert states[(4, 1)] == "supported"
    assert states[(4, 4)] == "supported"
    assert states[(4, 2)] == "unknown"
    assert states[(4, 3)] == "unknown"
    assert result.supported_count == 2


def test_usable_exclusions_mark_cells_unknown_without_fabricating_air():
    region = np.ones((4, 4), dtype=bool)
    reading = _reader(region, _polygon(0, 0, 4, 4))
    usable = np.ones((4, 4), dtype=bool)
    usable[3, 1] = False
    result = _build(reading, _identity(reading), usable, grid_shape=(4, 4))
    states = {(cell.row, cell.column): cell for cell in result.cells}
    excluded = states[(0, 1)]
    assert excluded.state == "unknown" and excluded.evidence == "outside_usable"
    assert excluded.region_excluded == (0, 1)
    assert excluded.usable_excluded == (1, 1)
    assert excluded.frame_excluded == (0, 1)
    assert result.supported_count == 15


def test_non_usable_pixels_are_never_treated_as_region_gap():
    region = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    reading = _reader(region, _polygon(0, 2, 2, 4))
    usable = np.ones((4, 4), dtype=bool)
    usable[0, 0] = False
    result = _build(reading, _identity(reading), usable, grid_shape=(2, 2))
    assert [cell.state for cell in result.cells] == ["supported"] * 4


def test_proposal_bounding_polygon_is_never_rasterized():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    tight = _reader(mask, _polygon(0, 2, 2, 4), frame="frame-a")
    loose = _reader(mask, _polygon(-9, -9, 40, 40), frame="frame-a")
    tight_result = _build(tight, _identity(tight), np.ones((4, 4), dtype=bool),
                          grid_shape=(2, 2))
    loose_result = _build(loose, _identity(loose), np.ones((4, 4), dtype=bool),
                          grid_shape=(2, 2))
    assert tight.region_evidence.source_region_id != \
        loose.region_evidence.source_region_id
    assert [cell.state for cell in tight_result.cells] == \
        [cell.state for cell in loose_result.cells]
    assert tight_result.cells == loose_result.cells


# ---------------------------------------------------------------------------
# Alignment and scale
# ---------------------------------------------------------------------------

def test_integer_window_shift_translates_the_canonical_support_result():
    region = _mask((4, 6), [(2, 0), (2, 1), (3, 0), (3, 1)])
    reading = _reader(region, _polygon(0, 2, 2, 4))
    identity = _identity(reading)
    usable = np.ones((4, 6), dtype=bool)
    base = _build(reading, identity, usable, origin=(0.0, 0.0), grid_shape=(2, 3))
    shifted = _build(reading, identity, usable, origin=(1.0, 0.0), grid_shape=(2, 3))
    base_states = {(cell.row, cell.column): cell.state for cell in base.cells}
    shifted_states = {(cell.row, cell.column): cell.state
                      for cell in shifted.cells}
    assert base_states == {(0, 0): "supported", (0, 1): "supported",
                           (0, 2): "unknown", (1, 0): "supported",
                           (1, 1): "supported", (1, 2): "unknown"}
    for row in range(2):
        for column in range(2):
            assert shifted_states[(row, column)] == base_states[(row, column + 1)]
    assert shifted.origin_uv == ((1, 1), (0, 1))
    assert shifted.support_sha256 != base.support_sha256


def test_alternative_specific_distance_scales_projected_area_exactly():
    mask = _mask((4, 4), [(3, 0)])
    polygon = _polygon(0, 3, 1, 4)
    near = _reader(mask, polygon, distance=1.0)
    far = _reader(mask, polygon, distance=2.0)
    usable = np.ones((4, 4), dtype=bool)
    near_result = _build(near, _identity(near), usable, grid_shape=(1, 1))
    far_result = _build(far, _identity(far), usable, grid_shape=(1, 1))
    assert near_result.cells[0].cell_area == (1, 1)
    assert far_result.cells[0].cell_area == (1, 4)
    assert Fraction(*near_result.cells[0].cell_area) == 4 * Fraction(
        *far_result.cells[0].cell_area)
    assert near_result.alternative.distance_blocks == 1.0
    assert far_result.alternative.distance_blocks == 2.0
    assert near_result.support_sha256 != far_result.support_sha256


# ---------------------------------------------------------------------------
# Determinism and canonicalization
# ---------------------------------------------------------------------------

def test_reordered_reading_alternatives_preserve_the_support_digest():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    polygon = _polygon(0, 2, 2, 4)
    alternatives = (
        lattice_planes.DistanceAlternative(1.0, 1.0, 0.9),
        lattice_planes.DistanceAlternative(2.0, 3.0, 0.4),
        lattice_planes.DistanceAlternative(4.0, 7.0, 0.2),
    )
    forward = _reader(mask, polygon, alternatives=alternatives)
    reversed_reading = replace(forward, alternatives=tuple(reversed(alternatives)))
    usable = np.ones((4, 4), dtype=bool)
    first = _build(forward, _identity(forward), usable, grid_shape=(2, 2), b=1.0)
    second = _build(reversed_reading, _identity(reversed_reading), usable,
                    grid_shape=(2, 2), b=1.0)
    assert first.support_sha256 == second.support_sha256
    assert first.to_record() == second.to_record()
    ordered = support.canonical_alternative_order(alternatives)
    assert [item.blocks_per_period for item in ordered] == [1.0, 2.0, 4.0]


def test_two_equivalent_readings_produce_identical_records():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    polygon = _polygon(0, 2, 2, 4)
    first_reading = _reader(mask, polygon)
    second_reading = _reader(mask.copy(), polygon)
    usable = np.ones((4, 4), dtype=bool)
    first = _build(first_reading, _identity(first_reading), usable, grid_shape=(2, 2))
    second = _build(second_reading, _identity(second_reading), usable,
                    grid_shape=(2, 2))
    assert first_reading is not second_reading
    assert first.support_sha256 == second.support_sha256
    assert first.to_record() == second.to_record()


def test_canonical_support_order_is_independent_of_insertion_order():
    mask = _mask((4, 6), [(2, 0), (2, 1), (3, 0), (3, 1)])
    reading = _reader(mask, _polygon(0, 2, 2, 4))
    identity = _identity(reading)
    usable = np.ones((4, 6), dtype=bool)
    records = [
        _build(reading, identity, usable, origin=(0.0, 0.0), grid_shape=(1, 1)),
        _build(reading, identity, usable, origin=(1.0, 0.0), grid_shape=(1, 1)),
        _build(reading, identity, usable, origin=(2.0, 0.0), grid_shape=(1, 1)),
        support.build_aligned_region_support(
            reading=reading, reading_identity=identity,
            reading_universe=_universe(reading), usable_mask=usable,
            image_sha256="nope", blocks_per_period=1.0, origin_uv=(0.0, 0.0),
            grid_shape=(1, 1), budget=_budget()),
    ]
    canonical = support.canonical_support_order(records)
    permutations = [records, list(reversed(records)),
                    [records[2], records[0], records[3], records[1]]]
    for permutation in permutations:
        assert support.canonical_support_order(permutation) == canonical
    assert isinstance(canonical[0], support.AlignedSupportRefusal)
    assert all(isinstance(record, support.AlignedRegionSupport)
               for record in canonical[1:])


# ---------------------------------------------------------------------------
# Fail-closed validation, precedence and budget
# ---------------------------------------------------------------------------

def test_missing_provenance_precedes_mask_and_shape_validation():
    bare = _bare_reader()
    identity = v2a.ReadingIdentity.build(
        frame=bare.frame, canonical_index=0, reading_order_key_sha256=ORDER,
        reading_axis=bare.axis, source_region_id=ABSENT_SHA,
        canonical_universe_sha256=UNIVERSE)
    refusal = _build(bare, identity, np.ones((4, 4), dtype=int),
                     grid_shape=(1, 1))
    assert isinstance(refusal, support.AlignedSupportRefusal)
    assert refusal.reason == "missing_provenance"
    assert refusal.work is None


def test_provenance_mismatch_precedes_usable_mask_shape():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    reading = _reader(mask, _polygon(0, 2, 2, 4))
    wrong_frame = _identity(reading, frame="other-frame")
    refusal = _build(reading, wrong_frame, np.ones((9, 9), dtype=bool),
                     grid_shape=(1, 1))
    assert refusal.reason == "provenance_mismatch"
    wrong_axis = _identity(reading, axis=1)
    assert _build(reading, wrong_axis, np.ones((4, 4), dtype=bool),
                  grid_shape=(1, 1)).reason == "provenance_mismatch"
    wrong_source = _identity(reading, source_region_id=ABSENT_SHA)
    assert _build(reading, wrong_source, np.ones((4, 4), dtype=bool),
                  grid_shape=(1, 1)).reason == "provenance_mismatch"


@pytest.mark.parametrize("kwargs, reason", [
    ({"image_sha256": "zz"}, "invalid_input"),
    ({"producer_version": "other"}, "invalid_input"),
    ({"phase_anchor": "zz"}, "invalid_input"),
])
def test_malformed_scalar_inputs_refuse_before_any_work(kwargs, reason):
    reading, _result = _rect_support()
    identity = _identity(reading)
    refusal = _build(reading, identity, np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), **kwargs)
    assert refusal.reason == reason
    assert refusal.work is None


@pytest.mark.parametrize("usable, reason", [
    (np.ones((4, 4), dtype=int), "invalid_usable_mask"),
    (np.zeros((4, 4), dtype=bool), "invalid_usable_mask"),
    (np.ones((4, 4), dtype=bool).reshape(2, 8), "image_shape_mismatch"),
    (np.ones((3, 3), dtype=bool), "image_shape_mismatch"),
    ("not-an-array", "invalid_usable_mask"),
])
def test_invalid_usable_masks_fail_closed(usable, reason):
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), usable, grid_shape=(2, 2))
    assert isinstance(refusal, support.AlignedSupportRefusal)
    assert refusal.reason == reason


def test_invalid_region_mask_is_mapped_to_a_typed_refusal(monkeypatch):
    reading, _result = _rect_support()

    def _boom(*args, **kwargs):
        raise ValueError("tampered region mask")

    monkeypatch.setattr(support.lattice_planes, "_decode_region_evidence", _boom)
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2))
    assert refusal.reason == "invalid_region_mask"


@pytest.mark.parametrize("kwargs, reason", [
    ({"b": 7.0}, "alternative_not_retained"),
    ({"origin": (float("nan"), 0.0)}, "invalid_origin"),
    ({"origin": (0.0, 0.0, 0.0)}, "invalid_origin"),
    ({"origin": "not-a-pair"}, "invalid_origin"),
    ({"grid_shape": (0, 1)}, "invalid_grid"),
    ({"grid_shape": (1, 2, 3)}, "invalid_grid"),
    ({"grid_shape": (1.5, 2)}, "invalid_grid"),
])
def test_alternative_origin_and_grid_fail_closed(kwargs, reason):
    reading, _result = _rect_support()
    parameters = {"origin": (0.0, 0.0), "grid_shape": (2, 2), "b": 1.0}
    parameters.update(kwargs)
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     **parameters)
    assert refusal.reason == reason


def test_planned_budget_exhaustion_is_atomic():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(4, 4), budget=_budget(max_cells=4))
    assert isinstance(refusal, support.AlignedSupportRefusal)
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail["cap"] == "max_cells"
    assert refusal.detail["limit"] == 4
    assert refusal.detail["observed"] == 16
    assert refusal.work.cells_examined == 0
    assert refusal.work.supported_cells == 0


def test_runtime_budget_exhaustion_discards_partial_cells():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2),
                     budget=_budget(max_region_pixel_tests=1))
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail["cap"] == "max_region_pixel_tests"
    assert refusal.work.cells_examined == 2
    assert refusal.work.region_pixel_tests == 2
    assert refusal.work.supported_cells == 1
    assert refusal.work.unknown_cells == 0
    assert not hasattr(refusal, "cells")


def test_supported_cell_cap_exhaustion_is_exact():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), budget=_budget(max_supported_cells=1))
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail["cap"] == "max_supported_cells"
    assert refusal.work.supported_cells == 2
    assert refusal.work.cells_examined == 2


# ---------------------------------------------------------------------------
# Canonical digest mutation rejection
# ---------------------------------------------------------------------------

def test_support_and_cell_digests_reject_mutation():
    _reading_value, result = _rect_support()
    with pytest.raises(ValueError):
        replace(result, image_sha256=OTHER_SHA)
    with pytest.raises(ValueError):
        replace(result, support_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(result, grid_shape=(1, 4))
    with pytest.raises(ValueError):
        replace(result.cells[0], state="unknown")
    with pytest.raises(ValueError):
        replace(result.cells[0], region_excluded=(2, 1))
    with pytest.raises(ValueError):
        replace(result.cells[0], cell_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(result.work, supported_cells=0)
    with pytest.raises(ValueError):
        replace(result.calibration, focal=2.0)
    with pytest.raises(ValueError):
        replace(result.alternative, distance_blocks=2.0)
    with pytest.raises(ValueError):
        replace(result, cells=tuple(reversed(result.cells)))


def test_refusal_digests_reject_mutation():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), b=7.0)
    with pytest.raises(ValueError):
        replace(refusal, reason="budget_exhausted")
    with pytest.raises(ValueError):
        replace(refusal, detail={"cap": "max_cells"})
    with pytest.raises(ValueError):
        replace(refusal, refusal_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Usable-mask digest
# ---------------------------------------------------------------------------

def test_usable_mask_digest_is_content_addressed():
    first = np.ones((4, 4), dtype=bool)
    first[1, 2] = False
    copy = first.copy()
    other = first.copy()
    other[3, 3] = False
    assert support.usable_mask_sha256(first) == support.usable_mask_sha256(copy)
    assert support.usable_mask_sha256(first) != support.usable_mask_sha256(other)
    assert support.usable_mask_sha256(np.ones((4, 4), dtype=bool)) != \
        support.usable_mask_sha256(np.ones((5, 4), dtype=bool))
    with pytest.raises(ValueError):
        support.usable_mask_sha256(np.ones((4, 4), dtype=int))
    with pytest.raises(ValueError):
        support.usable_mask_sha256(np.ones(4, dtype=bool))


def test_support_budget_and_work_records_are_self_validating():
    budget = _budget()
    with pytest.raises(ValueError):
        support.make_support_budget(max_cells=1)
    with pytest.raises(ValueError):
        support.make_support_budget(
            max_cells=0, max_region_pixel_tests=1, max_usable_pixel_tests=1,
            max_polygon_vertices=1, max_supported_cells=1)
    with pytest.raises(ValueError):
        support.AlignedSupportBudget(
            **{name: getattr(budget, name) for name in (
                "max_cells", "max_region_pixel_tests", "max_usable_pixel_tests",
                "max_polygon_vertices", "max_supported_cells")},
            budget_sha256="0" * 64)
    _reading_value, result = _rect_support()
    with pytest.raises(ValueError):
        support.AlignedSupportWork(
            cells_examined=4, frame_clips=4, region_pixel_tests=4,
            usable_pixel_tests=4, polygon_vertices=32, supported_cells=4,
            unknown_cells=0, work_sha256="0" * 64)
    assert result.work.to_record()["work_sha256"] == result.work.work_sha256


# ---------------------------------------------------------------------------
# Exact projection repair: rational coefficient composition
# ---------------------------------------------------------------------------

# Astra's binary64 counterexample and oblique/nontrivial controls.  ``distance``
# and the principal point are fractional and the grid origin is negative, so the
# float64-composed closed form differs from the exact rational composition at the
# projected boundary; only exact composition can stay conservative there.
_OBLIQUE_AXIS_MATRIX = (
    (0.9, 0.0, 0.43589),
    (0.0, 1.0, 0.0),
    (-0.43589, 0.0, 0.9),
)
_NONTRIVIAL_AXIS_MATRIX = (
    (0.5735, 0.6599, 0.4851),
    (-0.6821, 0.7276, -0.0736),
    (-0.4015, -0.2886, 0.8695),
)


def _exact_plane_basis_reference(axis, axis_matrix):
    """Independent plane basis in exact rationals (retained tangent convention)."""
    rows = tuple(tuple(Fraction(float(value)) for value in row)
                 for row in axis_matrix)
    others = [index for index in range(3) if index != axis]
    return rows[axis], rows[others[0]], rows[others[1]]


def _exact_homography_reference(axis, axis_matrix, focal, principal,
                                side, distance):
    """Independently recompose ``projection @ basis`` in exact rationals.

    This never evaluates a floating-point product: every declared coefficient is
    converted to :class:`Fraction` first, then only rational multiplication and
    addition build the homography.
    """
    normal, first, second = _exact_plane_basis_reference(axis, axis_matrix)
    focal_r = Fraction(float(focal))
    centre_x = Fraction(float(principal[0]))
    centre_y = Fraction(float(principal[1]))
    scale = Fraction(float(side)) * Fraction(float(distance))
    basis = (
        (first[0], second[0], scale * normal[0]),
        (first[1], second[1], scale * normal[1]),
        (first[2], second[2], scale * normal[2]),
    )
    projection = (
        (focal_r, Fraction(0), centre_x),
        (Fraction(0), -focal_r, centre_y),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    return tuple(
        tuple(sum((projection[i][k] * basis[k][j] for k in range(3)),
                  Fraction(0)) for j in range(3))
        for i in range(3))


def _float_composed_homography(reading):
    """The accepted closed form's rounded binary64 coefficients as rationals.

    This is the pre-repair coefficient source: the float64 product is formed
    first and only then converted, so the composition carries its rounding.
    """
    accepted = lattice_planes.rectification_homography(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    return tuple(tuple(Fraction(float(value)) for value in row)
                 for row in accepted)


def _classify_with_homography(matrix, origin, grid_shape, mask, usable):
    """Classify cells from an arbitrary coefficient matrix under V1 policy."""
    counter = support._SupportCounter(_budget())
    rows, columns = grid_shape
    origin_u, origin_v = origin
    states = []
    for row in range(rows):
        for column in range(columns):
            corners = ((origin_u + column, origin_v + row),
                       (origin_u + column + 1, origin_v + row),
                       (origin_u + column + 1, origin_v + row + 1),
                       (origin_u + column, origin_v + row + 1))
            projected = []
            degenerate = False
            for u, v in corners:
                denominator = (matrix[2][0] * u + matrix[2][1] * v
                               + matrix[2][2])
                if denominator <= 0:
                    degenerate = True
                    break
                projected.append(
                    ((matrix[0][0] * u + matrix[0][1] * v + matrix[0][2])
                     / denominator,
                     (matrix[1][0] * u + matrix[1][1] * v + matrix[1][2])
                     / denominator))
            if degenerate:
                states.append("degenerate")
                continue
            cell_area = support._polygon_area(tuple(projected))
            frame_polygon = support._clip_rect(
                tuple(projected), Fraction(0), Fraction(mask.shape[1]),
                Fraction(0), Fraction(mask.shape[0]))
            frame_excluded = cell_area - support._polygon_area(frame_polygon)
            region_excluded = frame_excluded + support._excluded_area(
                frame_polygon, mask, counter, "region")
            usable_excluded = frame_excluded + support._excluded_area(
                frame_polygon, usable, counter, "usable")
            states.append(support._classify_cell(
                cell_area, frame_excluded, region_excluded, usable_excluded))
    return states


def _astra_counterexample_reading(*, axis_matrix=None, origin=None):
    """Distance 0.3, principal 0.3 and grid origin -0.09 on a 4x4 mask."""
    mask = _mask((4, 4), [(3, 0)])
    reading = _reader(mask, _polygon(0, 3, 1, 4), distance=0.3,
                      focal=1.0, side=1.0, principal=(0.3, 4.0),
                      axis_matrix=axis_matrix)
    return mask, reading


def _counterexample_build(reading, origin, *, grid_shape=(3, 3)):
    return _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                  origin=origin, grid_shape=grid_shape)

def test_exact_homography_composes_retained_coefficients_in_rationals():
    """The consumed homography is exactly the independently composed rational."""
    mask, reading = _astra_counterexample_reading()
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    observed = support._exact_homography(reading.axis, calibration, alternative)
    expected = _exact_homography_reference(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    assert observed == expected
    assert all(type(value) is Fraction for row in observed for value in row)


def test_exact_homography_is_not_the_rounded_float_composition():
    """On the counterexample the float64 composition is provably different.

    The projected boundary sits where ``0.3 * 0.3`` rounds in binary64, so the
    rounded coefficient path and the exact rational composition disagree.  The
    module must follow the exact one.
    """
    mask, reading = _astra_counterexample_reading()
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    exact = support._exact_homography(reading.axis, calibration, alternative)
    rounded = _float_composed_homography(reading)
    assert rounded != exact
    # The translation column carries the exact rational product of the retained
    # `distance_blocks * side` (here 0.3 * 0.3 = 9/100), which the binary64
    # product rounds to a different dyadic even though both print as `0.09`.
    for row in range(3):
        for column in range(3):
            assert float(exact[row][column]) == float(rounded[row][column])
    assert Fraction(0.3 * 0.3) != Fraction(0.3) * Fraction(0.3)
    # The identity-calibration translation column is exactly principal *
    # (distance * side) * normal in rationals, with no floating-point product.
    assert exact[0][2] == (
        Fraction(reading.rectification.principal_point[0])
        * (Fraction(alternative.distance_blocks) * Fraction(reading.side))
        * Fraction(reading.rectification.axis_matrix[reading.axis][2]))
    assert rounded[0][2] != exact[0][2]


def test_counterexample_boundary_cell_stays_conservative_and_exact():
    """Distance 0.3, principal 0.3, origin -0.09: the single cell's left edge is
    exactly on the image border.  The cell may not be ``supported``/``full``, and
    its projected boundary must be the exact rational composition rather than the
    rounded ``0.3 * 0.3`` binary64 product.
    """
    mask, reading = _astra_counterexample_reading()
    origin = (Fraction(-9, 100), Fraction(0))
    result = _counterexample_build(reading, origin, grid_shape=(1, 1))
    assert isinstance(result, support.AlignedRegionSupport)
    assert all(cell.state == "unknown" for cell in result.cells)
    assert all(cell.evidence == "outside_frame" for cell in result.cells)
    assert all(Fraction(*cell.frame_excluded) > 0 for cell in result.cells)
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    exact_matrix = support._exact_homography(reading.axis, calibration,
                                             alternative)
    # The consumed translation coefficient is exactly principal * (distance *
    # side) * normal in rationals; the binary64 product rounds it differently.
    expected_translation = (
        Fraction(reading.rectification.principal_point[0])
        * (Fraction(alternative.distance_blocks) * Fraction(reading.side))
        * Fraction(reading.rectification.axis_matrix[2][2]))
    assert exact_matrix[0][2] == expected_translation
    # The rounded float64 product is provably a different rational here, so the
    # exact composition is observable, not cosmetic.
    assert Fraction(0.3 * 0.3) != Fraction(0.3) * Fraction(0.3)
    exact_states = _classify_with_homography(exact_matrix, origin, (1, 1), mask,
                                             np.ones((4, 4), dtype=bool))
    assert exact_states == ["outside_frame"]
    assert [cell.evidence for cell in result.cells] == exact_states


def test_astra_counterexample_exact_boundary_exposes_the_float_defect():
    """Astra's binary64 boundary counterexample, retained as one explicit diagnostic.

    Identity axes, ``focal=1``, ``distance=0.3``, principal point ``(0.3, 4)`` and
    the exact grid origin ``-Fraction(0.09)``: the rounded float64 composition of
    the retained calibration/alternative coefficients places the single cell's
    left edge at exactly ``0``, so that composition reports the all-true cell as
    ``full`` with every exclusion zero -- the defect Astra used to reject the
    rounded path.  The exact rational composition places the same edge at
    ``-Fraction(1, 90071992547409920)``, outside the frame, so the module under
    test must report ``unknown``/``outside_frame`` for that cell.
    """
    region = np.ones((4, 4), dtype=bool)
    usable = np.ones((4, 4), dtype=bool)
    reading = _reader(region, _polygon(0, 0, 4, 4), distance=0.3, focal=1.0,
                      principal=(0.3, 4.0))
    origin_u = -Fraction(0.09)
    origin = (origin_u, Fraction(0))

    def _left_edge(matrix):
        zero = Fraction(0)
        denominator = (matrix[2][0] * origin_u + matrix[2][1] * zero
                       + matrix[2][2])
        return ((matrix[0][0] * origin_u + matrix[0][1] * zero + matrix[0][2])
                / denominator)

    exact = _exact_homography_reference(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    assert _left_edge(exact) == Fraction(-1, 90071992547409920)

    rounded = _float_composed_homography(reading)
    assert rounded != exact
    assert _left_edge(rounded) == 0
    assert _classify_with_homography(
        rounded, origin, (1, 1), region, usable) == ["full"]

    result = _build(reading, _identity(reading), usable, origin=origin,
                    grid_shape=(1, 1))
    assert isinstance(result, support.AlignedRegionSupport)
    cell = result.cells[0]
    assert (cell.state, cell.evidence) == ("unknown", "outside_frame")
    assert Fraction(*cell.frame_excluded) > 0
    assert cell.region_excluded == cell.frame_excluded
    assert cell.usable_excluded == cell.frame_excluded
    assert result.supported_count == 0


def test_oblique_counterexample_control_remains_conservative():
    """A genuinely oblique calibration with a fractional origin stays exact and
    conservative: every cell edge on the image border is ``unknown``."""
    mask, reading = _astra_counterexample_reading(
        axis_matrix=_OBLIQUE_AXIS_MATRIX)
    result = _counterexample_build(reading, origin=(-0.09, 0.0))
    assert isinstance(result, support.AlignedRegionSupport)
    assert all(cell.state == "unknown" for cell in result.cells)
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    assert support._exact_homography(reading.axis, calibration, alternative) == \
        _exact_homography_reference(
            reading.axis, reading.rectification.axis_matrix,
            reading.rectification.focal, reading.rectification.principal_point,
            reading.side, reading.alternatives[0].distance_blocks)


def test_nontrivial_oblique_calibration_matches_closed_form_and_stays_exact():
    """A non-axis-aligned axis matrix at a non-unit distance composes exactly."""
    mask = _mask((4, 4), [(3, 0)])
    reading = _reader(mask, _polygon(0, 3, 1, 4), distance=3.3, focal=720.5,
                      side=0.7, principal=(320.2, 240.7),
                      axis_matrix=_NONTRIVIAL_AXIS_MATRIX)
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    exact = support._exact_homography(reading.axis, calibration, alternative)
    assert exact == _exact_homography_reference(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    accepted = lattice_planes.rectification_homography(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    for row in range(3):
        for column in range(3):
            reference = float(accepted[row][column])
            observed = float(exact[row][column])
            assert abs(observed - reference) <= 1e-9 * max(
                1.0, abs(observed), abs(reference))
    assert support._determinant3(exact) != 0


def test_exact_projection_keeps_invalid_input_precedence_over_budget():
    """The exact composition path still validates before any budget decision."""
    mask, reading = _astra_counterexample_reading(
        axis_matrix=_OBLIQUE_AXIS_MATRIX)
    identity = _identity(reading)
    universe = _universe(reading)
    refusal = _build(reading, identity, np.ones((4, 4), dtype=bool),
                     origin=(-0.09, 0.0), grid_shape=(3, 3),
                     budget=_budget(max_cells=1), universe=universe)
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail["cap"] == "max_cells"
    assert refusal.work.cells_examined == 0


# ---------------------------------------------------------------------------
# Finding 1: conservative projection at exact floating-point boundaries
# ---------------------------------------------------------------------------

def test_projection_uses_exact_rational_coefficients_not_rounded_float(
        monkeypatch):
    """The projection must not consume the rounded float64 homography.

    ``rectification_homography`` is perturbed by a finite amount that would move
    a supported cell across a mask edge if its returned coefficients were used
    directly.  The exact rational projection must be unaffected.
    """
    reading, base = _rect_support()
    identity = _identity(reading)
    universe = _universe(reading)
    accepted = lattice_planes.rectification_homography

    def _rounded(*args, **kwargs):
        matrix = np.array(accepted(*args, **kwargs), dtype=float).copy()
        matrix[0, 0] += 0.25
        matrix[1, 1] += 0.25
        return matrix

    monkeypatch.setattr(support.lattice_planes, "rectification_homography",
                        _rounded)
    patched = _build(reading, identity, np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), universe=universe)
    assert isinstance(patched, support.AlignedRegionSupport)
    assert patched.cells == base.cells
    assert patched.support_sha256 == base.support_sha256


def test_exact_negative_projected_edge_is_never_supported():
    """A cell whose exact projected edge lies outside the image is unknown."""
    mask = _mask((4, 4), [(3, 0)])
    reading = _reader(mask, _polygon(0, 3, 1, 4))
    result = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                    origin=((-1, 1 << 20), (0, 1)), grid_shape=(1, 1))
    cell = result.cells[0]
    assert cell.state == "unknown"
    assert cell.evidence == "outside_frame"
    assert cell.cell_area == (1, 1)
    assert cell.frame_excluded == (1, 1 << 20)
    assert cell.region_excluded == (1, 1 << 20)
    assert cell.usable_excluded == (1, 1 << 20)
    assert result.supported_count == 0


def test_exact_homography_matches_the_accepted_closed_form():
    reading = _reader(_mask((4, 4), [(3, 0)]), _polygon(0, 3, 1, 4),
                      axis=0, focal=720.5, principal=(320.2, 240.7),
                      side=0.7, distance=3.3)
    calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix,
        focal=reading.rectification.focal,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    alternative = support.RetainedAlternative.build(reading.alternatives[0])
    exact = support._exact_homography(reading.axis, calibration, alternative)
    accepted = lattice_planes.rectification_homography(
        reading.axis, reading.rectification.axis_matrix,
        reading.rectification.focal, reading.rectification.principal_point,
        reading.side, reading.alternatives[0].distance_blocks)
    for row in range(3):
        for column in range(3):
            reference = float(accepted[row][column])
            observed = float(exact[row][column])
            assert abs(observed - reference) <= 1e-9 * max(
                1.0, abs(observed), abs(reference))
    assert support._determinant3(exact) != 0


# ---------------------------------------------------------------------------
# Finding 2: content-bound ReadingIdentity
# ---------------------------------------------------------------------------

def _content_mutation_cases(reading):
    """One (mutated reading, supplied identity, universe) attack per binding."""
    def _resealed_identity(**overrides):
        return _identity(reading, universe=_universe(reading), **overrides)

    other = _reader(_mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 2)]),
                    _polygon(0, 2, 2, 4))
    universe = _universe(reading)
    identity = _identity(reading, universe=universe)
    return {
        "alternative_period": (
            replace(reading, alternatives=(
                lattice_planes.DistanceAlternative(2.0, 1.0, 1.0),)),
            identity, universe),
        "alternative_distance": (
            replace(reading, alternatives=(
                lattice_planes.DistanceAlternative(1.0, 5.0, 1.0),)),
            identity, universe),
        "reading_order": (
            replace(reading, period_pixels=3.0), identity, universe),
        "region_evidence": (
            replace(reading, region_evidence=other.region_evidence),
            identity, universe),
        "rectification": (
            replace(reading, rectification=replace(
                reading.rectification, focal=2.0)), identity, universe),
        "frame": (reading, _resealed_identity(frame="frame-b"), universe),
        "axis": (reading, _resealed_identity(axis=1), universe),
        "source_region": (
            reading, _resealed_identity(source_region_id=ABSENT_SHA),
            universe),
        "dummy_order_key": (
            reading, _resealed_identity(key_sha=ORDER), universe),
        "dummy_universe": (
            reading, _resealed_identity(universe_sha=UNIVERSE), universe),
    }


@pytest.mark.parametrize("case", [
    "alternative_period", "alternative_distance", "reading_order",
    "region_evidence", "rectification", "frame", "axis", "source_region",
    "dummy_order_key", "dummy_universe"])
def test_changed_reading_content_cannot_reuse_the_old_identity(case):
    reading, _result = _rect_support()
    mutated, identity, universe = _content_mutation_cases(reading)[case]
    refusal = _build(mutated, identity, np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), universe=universe)
    assert isinstance(refusal, support.AlignedSupportRefusal)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.work is None


def test_support_record_rejects_resealed_identity_content_mutations():
    reading, result = _rect_support()
    other_calibration = support.RegionSupportCalibration.build(
        axis_matrix=reading.rectification.axis_matrix, focal=2.0,
        principal_point=reading.rectification.principal_point,
        centre=reading.rectification.centre, side=reading.side)
    other_alternative = support.RetainedAlternative.build(
        lattice_planes.DistanceAlternative(1.0, 2.0, 1.0))
    for change in (
        {"calibration": other_calibration},
        {"alternative": other_alternative},
        {"region_mask_sha256": "0" * 64},
        {"image_sha256": OTHER_SHA},
        {"reading_identity_sha256": "0" * 64},
        {"reading_binding_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError):
            replace(result, **change)
    rebased = support.AlignedRegionSupport.build(**_support_fields(result))
    assert rebased.support_sha256 == result.support_sha256


def test_reordered_alternatives_share_the_canonical_identity():
    mask = _mask((4, 4), [(2, 0), (2, 1), (3, 0), (3, 1)])
    polygon = _polygon(0, 2, 2, 4)
    alternatives = (
        lattice_planes.DistanceAlternative(1.0, 1.0, 0.9),
        lattice_planes.DistanceAlternative(2.0, 3.0, 0.4),
        lattice_planes.DistanceAlternative(4.0, 7.0, 0.2),
    )
    forward = _reader(mask, polygon, alternatives=alternatives)
    reversed_reading = replace(forward,
                               alternatives=tuple(reversed(alternatives)))
    assert _identity(forward) == _identity(reversed_reading)
    refusal = _build(forward, _identity(forward, key_sha=ORDER),
                     np.ones((4, 4), dtype=bool), grid_shape=(2, 2))
    assert refusal.reason == "provenance_mismatch"


# ---------------------------------------------------------------------------
# Finding 3: sealed records reject freshly resealed contradictions
# ---------------------------------------------------------------------------

def _runtime_budget_refusal():
    reading, _result = _rect_support()
    return _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                  grid_shape=(2, 2), budget=_budget(max_region_pixel_tests=1))


def _planned_budget_refusal():
    reading, _result = _rect_support()
    return _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                  grid_shape=(4, 4), budget=_budget(max_cells=4))


def _work_sha(payload):
    """Independently recompute the work-record canonical digest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _work_payload(**overrides):
    payload = {name: 0 for name in support._WORK_FIELDS}
    payload.update(overrides)
    return payload


def _forge_work(**overrides):
    """A digest-correct work record carrying impossible counters.

    It is built through ``object.__new__`` so ``AlignedSupportWork`` construction
    cannot reject it: the receiving refusal/support record must reject the
    freshly resealed contradiction on its own semantic invariants.
    """
    payload = _work_payload(**overrides)
    work = object.__new__(support.AlignedSupportWork)
    for name, value in payload.items():
        object.__setattr__(work, name, value)
    object.__setattr__(work, "work_sha256", _work_sha(payload))
    return work


def test_support_record_rejects_normal_axis_as_tangent_label():
    reading, result = _rect_support()
    fields = _support_fields(result)
    for label in ("row_label", "column_label"):
        with pytest.raises(ValueError):
            support.AlignedRegionSupport.build(**{**fields, label: reading.axis})
        with pytest.raises(ValueError):
            replace(result, **{label: reading.axis})
    assert support.AlignedRegionSupport.build(**fields).support_sha256 \
        == result.support_sha256


def test_resealed_budget_refusal_rejects_a_nonexistent_cap():
    refusal = _runtime_budget_refusal()
    assert refusal.detail["cap"] == "max_region_pixel_tests"
    with pytest.raises(ValueError):
        _reseal_refusal("budget_exhausted",
                        {"cap": "max_widgets", "limit": 1, "observed": 2},
                        refusal.work)


def test_resealed_budget_refusal_rejects_nonpositive_observed_limits():
    refusal = _runtime_budget_refusal()
    cap = refusal.detail["cap"]
    for limit, observed in ((0, 2), (1, 0), (-1, 2), (1, -2), (2, 2)):
        with pytest.raises(ValueError):
            _reseal_refusal(
                "budget_exhausted",
                {"cap": cap, "limit": limit, "observed": observed},
                refusal.work)


def test_resealed_budget_refusal_requires_work_evidence():
    refusal = _runtime_budget_refusal()
    with pytest.raises(ValueError):
        _reseal_refusal("budget_exhausted", dict(refusal.detail), None)


def test_resealed_budget_refusal_rejects_embedded_cells():
    refusal = _runtime_budget_refusal()
    with pytest.raises(ValueError):
        _reseal_refusal("budget_exhausted", dict(refusal.detail, cells=[]),
                        refusal.work)
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            dict(refusal.detail, cells=[{"state": "supported"}]),
            refusal.work)


def test_resealed_non_budget_refusal_rejects_work_evidence():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), b=7.0)
    assert refusal.reason == "alternative_not_retained"
    assert refusal.work is None
    with pytest.raises(ValueError):
        _reseal_refusal("alternative_not_retained", refusal.detail,
                        _runtime_budget_refusal().work)


def test_resealed_budget_refusal_observed_must_match_work_counter():
    refusal = _runtime_budget_refusal()
    assert refusal.work.cells_examined > 0
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_region_pixel_tests", "limit": 1, "observed": 3},
            refusal.work)
    planned = _planned_budget_refusal()
    assert planned.work.cells_examined == 0
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_supported_cells", "limit": 1, "observed": 2},
            planned.work)


@pytest.mark.parametrize("counter", [
    "frame_clips", "region_pixel_tests", "usable_pixel_tests",
    "polygon_vertices", "supported_cells", "unknown_cells",
])
def test_resealed_planned_budget_refusal_rejects_nonzero_work_counters(counter):
    """A planned refusal recomputes all digests yet carries nonzero counters."""
    planned = _planned_budget_refusal()
    assert planned.work.cells_examined == 0
    assert all(getattr(planned.work, name) == 0
               for name in support._WORK_FIELDS)
    forged = _forge_work(**{counter: 1})
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_cells", "limit": 4, "observed": 16},
            forged)


def test_resealed_max_cells_refusal_rejects_examined_work():
    """``max_cells`` is ex-ante, so any examined work is a contradiction."""
    forged = _forge_work(cells_examined=1, frame_clips=1,
                         region_pixel_tests=1, unknown_cells=1)
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_cells", "limit": 1, "observed": 2},
            forged)


def test_resealed_runtime_budget_refusal_rejects_state_count_overflow():
    """``supported_cells + unknown_cells`` may never exceed examined cells."""
    forged = _forge_work(cells_examined=2, frame_clips=2,
                         region_pixel_tests=2, supported_cells=2,
                         unknown_cells=1)
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_region_pixel_tests", "limit": 1, "observed": 2},
            forged)


def test_resealed_runtime_budget_refusal_rejects_frame_clip_overflow():
    """``frame_clips`` may never exceed examined cells."""
    forged = _forge_work(cells_examined=2, frame_clips=3,
                         region_pixel_tests=2, supported_cells=2)
    with pytest.raises(ValueError):
        _reseal_refusal(
            "budget_exhausted",
            {"cap": "max_region_pixel_tests", "limit": 1, "observed": 2},
            forged)


def test_work_record_rejects_impossible_counter_relations():
    for overrides in (
        {"cells_examined": 2, "frame_clips": 2, "region_pixel_tests": 2,
         "supported_cells": 2, "unknown_cells": 1},
        {"cells_examined": 1, "frame_clips": 2, "region_pixel_tests": 1,
         "supported_cells": 1},
    ):
        payload = _work_payload(**overrides)
        with pytest.raises(ValueError):
            support.AlignedSupportWork(
                **payload, work_sha256=_work_sha(payload))


def test_resealed_support_record_rejects_forged_work_relation():
    _reading_value, result = _rect_support()
    fields = _support_fields(result)
    forged = _forge_work(cells_examined=4, frame_clips=5,
                         region_pixel_tests=4, usable_pixel_tests=4,
                         polygon_vertices=32, supported_cells=4,
                         unknown_cells=0)
    with pytest.raises(ValueError):
        support.AlignedRegionSupport.build(**{**fields, "work": forged})


def test_resealed_genuine_planned_and_runtime_budget_refusals_still_construct():
    planned = _planned_budget_refusal()
    resealed_planned = _reseal_refusal(
        "budget_exhausted", dict(planned.detail), planned.work)
    assert resealed_planned.detail == planned.detail
    runtime = _runtime_budget_refusal()
    resealed_runtime = _reseal_refusal(
        "budget_exhausted", dict(runtime.detail), runtime.work)
    assert resealed_runtime.detail == runtime.detail
    assert resealed_runtime.work == runtime.work


# ---------------------------------------------------------------------------
# Finding 4: invalid input and projection validation precede any budget
# ---------------------------------------------------------------------------

def test_non_finite_homography_precedes_budget(monkeypatch):
    reading, _result = _rect_support()
    identity = _identity(reading)
    universe = _universe(reading)

    def _non_finite(*args, **kwargs):
        return np.full((3, 3), np.nan)

    monkeypatch.setattr(support.lattice_planes, "rectification_homography",
                        _non_finite)
    small = _build(reading, identity, np.ones((4, 4), dtype=bool),
                   grid_shape=(2, 2), budget=_budget(max_cells=1),
                   universe=universe)
    large = _build(reading, identity, np.ones((4, 4), dtype=bool),
                   grid_shape=(2, 2), universe=universe)
    assert small.reason == "calibration_mismatch"
    assert large.reason == "calibration_mismatch"
    assert small.to_record() == large.to_record()


def test_degenerate_homography_precedes_budget():
    reading, _result = _rect_support()
    degenerate = replace(reading, rectification=replace(
        reading.rectification,
        axis_matrix=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))))
    universe = _universe(degenerate)
    identity = _identity(degenerate, universe=universe)
    small = _build(degenerate, identity, np.ones((4, 4), dtype=bool),
                   grid_shape=(2, 2), budget=_budget(max_cells=1),
                   universe=universe)
    large = _build(degenerate, identity, np.ones((4, 4), dtype=bool),
                   grid_shape=(2, 2), universe=universe)
    assert small.reason == "calibration_mismatch"
    assert large.reason == "calibration_mismatch"
    assert small.to_record() == large.to_record()


def test_valid_homography_still_exhausts_a_small_budget():
    reading, _result = _rect_support()
    refusal = _build(reading, _identity(reading), np.ones((4, 4), dtype=bool),
                     grid_shape=(2, 2), budget=_budget(max_cells=1))
    assert refusal.reason == "budget_exhausted"
    assert refusal.detail["cap"] == "max_cells"
    assert refusal.detail["observed"] == 4
    assert refusal.work.cells_examined == 0


# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------

def test_module_imports_only_its_declared_dependencies():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported_roots = set()
    reconstruction_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_roots.add(module.split(".")[0])
            if module.startswith("mc_photogrammetry.reconstruction."):
                reconstruction_modules.add(module.rsplit(".", 1)[-1])
            if module == "mc_photogrammetry.reconstruction":
                reconstruction_modules.update(
                    alias.name for alias in node.names)
    assert imported_roots <= ALLOWED_IMPORT_ROOTS
    assert reconstruction_modules == ALLOWED_RECONSTRUCTION_MODULES


def test_module_source_has_no_io_or_wiring_calls():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in DENIED_NAMES, node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in DENIED_NAMES, node.attr
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(token in alias.name for token in (
                    "litematic", "argparse", "subprocess", "socket"))
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not any(token in module for token in (
                "litematic", "argparse", "subprocess", "socket"))
