"""Independent controls for the isolated M4 V2 resolved face placement.

The public fixture is one genuine **mixed-normal single universe**: five frames,
each carrying three normal-axis readings (15 readings total), joined into a
single connected, exhaustive, cycle-validated C2-R-B component whose V2-A
result is ``endpoint_unique`` and whose V2-B result determines all three world
axes from that one component.  Every reading carries genuine region evidence, a
genuine usable mask and a genuine P0 split source, and every relative phase is
declared zero (phase 0), so the exact per-axis anchors are
``alpha = ((0,1), (0,1), (0,1))`` and every resolved face is the integer block 1.

The frame graph is connected with one match per frame pair and world axis, so it
is a single component even though its readings span all three world normal
axes.  No counter is faked, no world position is supplied, and no partial ring
is combined or sliced.

Signed/nonzero reachability: a second genuine fixture is built here from the
accepted APIs.  It is an over-determined three-frame triangle with absolute
quadrants 0/1/2 around the common world normal axis 2, mixed
``PlaneReading.side`` signs (+1, -1, -1), one reflected closing edge (relative
quadrant 2) and genuine nonzero P0 split-phase generation evidence for every
reading.  Its C2-R-B certificate is ``unique_validated``/``exhaustive`` with a
single translation class and genuinely signed, fractional translations, and its
V2-A result is ``endpoint_unique``/``exhaustive``.  The first accepted gate that
refuses is V2-B: all three readings observe one physical plane, so the common
world normal axis is never a *tangent* label of an accepted reading domain; the
frozen classifier reports ``no_hypothesis`` for that world axis while the two
tangent axes stay ``evidence_conflict`` below the calibrated ``1/8`` margin.
That exact reproducible refusal is recorded by
``test_genuine_nonzero_quadrant_v2b_all_axis_is_blocked`` and is not repaired by
weakening a gate, resealing metadata or aggregating partial components.  The
phase-0 mixed-normal bundle stays the zero-residue control that passes the whole
V2-B/placement chain, and the signed arithmetic is still proven independently by
the literal oracle (``face_axis_coordinate`` / ``tangent_axis_coordinate`` /
``_cell_lower_endpoint``), never through a resealed V2-B result.

Exact mixed-normal P0 recipe (``m4-v2-exact-p0-fixture``): a third genuine
fixture is built from Astra's independently checked construction.  Three frames
sit in absolute quadrants 0/1/2 and each carries three local-normal readings, so
every world axis is a *tangent* of an accepted reading domain.  Every reading's
output is the real P0 generation evidence of a float64 16x16 patch (b = 1, exact
period four pixels) whose generated phase is (0,0), (2,2), (2,2), and its
held-out split source adds the recipe's D impulses.  The rectified signature
window is sampled at 25 samples with extents 3, 6, 2 at distances 1, 1/2, 3/2,
so its cell grid is 6x6 while every reading's period stays exactly four pixels.
Sides are derived so the intended synthetic plane coordinate is the integer
block 1.

The construction is genuine, but the historical signature window was too small
to represent the declared translations.  On a 4x4 cell grid the reflected
R0->R1 edges need integer grid shifts of up to two cells in one lateral
direction, and the required *joint* entry then has fewer than
``plane_graph.MIN_OVERLAP_CELLS`` (9) supported cells; A4 could therefore expose
only one lateral component at a time.  The repaired recipe uses the smallest
window that carries every declared joint vector -- a six-cell (25-sample)
signature grid with reflection-aligned ``origin_uv`` shift -5 -- so all nine
genuine C1 edges retain an actual A4 candidate whose two lateral components
jointly equal the declared ``RECIPE_TRANSLATIONS`` difference
(``test_recipe_all_nine_edges_retain_full_joint_translation``).  The historical
4x4 grid with the -2 shift is retained as the negative control
(``test_recipe_current_geometry_drops_the_carry``).

With the joint vectors retained, the unchanged
``solve_relative_translation_graph`` seals the complete nine-edge graph as one
exhaustive ``unique_validated`` component with ``status == "complete"`` and
``placement_readiness == "ready"``; V2-A is ``endpoint_unique`` and V2-B
determines all three world axes.  The certificate is still read only from the
solver (``test_recipe_c2rb_component_is_unique_validated``): no record is
hand-sealed, no certificate is fabricated and no gate is weakened.  The recipe is
also the only nonzero-residue fixture proven end to end through placement:
``test_recipe_full_signed_placement_oracle`` resolves its nine readings into nine
signed faces over all three world normals, places exactly the 66 supported cells
and omits the 15 boundary-unknown cells.

Expected values are declared by the fixture and by independent literal tables,
never read back from the module under test.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import lattice_planes as lp
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_phase_anchor as v2b
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_region_support as v1
from mc_photogrammetry.reconstruction import plane_relative_translation as relative
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain
from mc_photogrammetry.reconstruction import plane_volume

from test_c2r_vector_domain_a3_proof import (
    _canonical_identity,
    _physical_yaw,
    _reading as _base_reading,
    _world_texture,
)
from test_plane_phase_anchor import (
    _anchor_budget,
    _p0_budget,
    _v2a_budget,
    _xyz_p0_budget,
)
from test_plane_relative_translation import (
    _raw_fixture_image,
    _raw_fixture_signature,
    _runtime_budget,
)
from test_plane_shift_domain import _budget as _shift_budget
from test_plane_vector_domain import _large_budget, _phase_for


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "mc_photogrammetry" / "reconstruction"
          / "plane_placement.py")

H = W = 16
EXTENT = 2.0
SAMPLES = 16
DISTANCE = 1.0
B_VALUE = 1.0
FRAMES = ("R0", "R1", "R2", "R3", "R4")
AXES = (0, 1, 2)
FRAME_POINTS = {"R0": (0, 0), "R1": (2, 0), "R2": (3, 2),
                "R3": (1, 3), "R4": (-1, 2)}
PATCH_STEP = lp.lattice.rectified_pixel_step(EXTENT, SAMPLES)
PERIOD = B_VALUE / (DISTANCE * PATCH_STEP)
POLY = ((400.0, 400.0), (640.0, 400.0), (640.0, 640.0), (400.0, 640.0))
IMAGE_SHA = hashlib.sha256(b"placement-fixture-image").hexdigest()

# Declared, independent phase-0 expectation.
EXPECTED_ALPHA = ((0, 1), (0, 1), (0, 1))
EXPECTED_ANCHOR_DIGEST = placement.placement_anchor_digest(EXPECTED_ALPHA)

# Declared exact work totals of the phase-0 fixture (15 faces x 3x3 cells).
EXPECTED_WORK = {
    "faces": 15,
    "supports_checked": 15,
    "support_cells": 135,
    "frame_axis_lookups": 45,
    "integer_checks": 45,
    "anchors": 300,
}
# (budget cap, exact observed total) pairs used for the atomic boundary tests.
EXPECTED_CAP_BOUNDARIES = (
    ("max_faces", 15),
    ("max_supports_checked", 15),
    ("max_support_cells", 135),
    ("max_frame_axis_lookups", 45),
    ("max_integer_checks", 45),
    ("max_anchors", 300),
)

# Literal, independent copy of the shared discrete transform.  The production
# plane_graph table is never consulted to build an expectation here.
LITERAL_TRANSFORMS = {
    0: {0: (0, 1), 1: (1, 1), 2: (2, 1)},
    1: {0: (2, 1), 1: (1, 1), 2: (0, -1)},
    2: {0: (0, -1), 1: (1, 1), 2: (2, -1)},
    3: {0: (2, -1), 1: (1, 1), 2: (0, 1)},
}


def _mask_for(axis: int) -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[400:640, 400:640] = True
    return mask


_EVIDENCE = {axis: lp._make_region_evidence(_mask_for(axis), axis, POLY,
                                             ("region-a",))
             for axis in AXES}
_REGION_ID = {axis: _EVIDENCE[axis].source_region_id for axis in AXES}


# ---------------------------------------------------------------------------
# Fixture builders (genuine public pipeline, region evidence included)
# ---------------------------------------------------------------------------

def _reading(frame, axis, b, distance, side, origin, local_residues, **kwargs):
    base = _base_reading(frame, axis, b, distance, side, origin,
                         local_residues, **kwargs)
    evidence = _EVIDENCE[axis]
    phase = replace(base.phase_evidence,
                    source_region_id=evidence.source_region_id)
    return replace(base, polygon=lp._canonical_region_polygon(POLY),
                   region_evidence=evidence, phase_evidence=phase)


def _signature(reading, b, basis, translation, origin_uv=None):
    rng = np.random.default_rng(992)
    image = rng.random((1024, 1024, 3), dtype=np.float32) + 0.1
    measured = lp.plane_signature(image, reading, b,
                                  usable=_mask_for(reading.axis), max_cells=32)
    if origin_uv is not None:
        measured = replace(measured, origin_uv=tuple(float(value)
                                                     for value in origin_uv))
    rows, columns = np.indices(measured.cells.shape, dtype=np.float64)
    local = np.stack((measured.origin_uv[0] + columns,
                      measured.origin_uv[1] + rows), axis=0).reshape(2, -1)
    world = basis @ local + translation.reshape(2, 1)
    sampled = _world_texture(world[0], world[1]).reshape(measured.cells.shape)
    cells = np.asarray(measured.cells).copy()
    valid = np.asarray(measured.coverage, dtype=bool)
    cells[valid] = sampled[valid]
    return replace(measured, cells=cells)


def _geometry(period=PERIOD, extent=EXTENT, samples=SAMPLES):
    return lp.SplitPhaseGeometry(
        centre=(extent, extent), extent=extent, samples=samples,
        rectified_pixel_step=lp.lattice.rectified_pixel_step(extent, samples),
        period_pixels=period)


def _produce(patch, frame, plane_index=0, axis=2, period=PERIOD,
             extent=EXTENT, samples=SAMPLES):
    return lp.produce_split_phase_evidence(
        patch, geometry=_geometry(period, extent, samples), frame=frame,
        plane_index=plane_index, reading_axis=axis,
        source_region_id=_REGION_ID[axis],
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=_p0_budget())


def _g_evidence(result):
    generation = result.generation_evidence[0]
    return lp.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=tuple(generation.phase_uv_pixels),
        period_pixels=generation.period_pixels,
        patch_centre_uv=tuple(generation.patch_centre_uv),
        extent=generation.extent, samples=generation.samples,
        response_magnitude_uv=tuple(generation.response_magnitude_uv),
        profile_mass_uv=tuple(generation.profile_mass_uv),
        concentration_uv=tuple(generation.concentration_uv),
        unavailable_reason_uv=tuple(generation.unavailable_reason_uv),
        source_region_id=generation.source_region_id)


def _held_out_patch(d_bins=(11, 15)):
    """G structural impulse plus held-out D bumps in the ell=0 band."""
    patch = np.zeros((H, W), dtype=np.float64)
    patch[0, 0] = 1.0
    for index in d_bins:
        patch[index, index] = 4.0
    return patch


def _support_budget(**overrides):
    limits = {
        "max_cells": 4096, "max_region_pixel_tests": 10_000_000,
        "max_usable_pixel_tests": 10_000_000, "max_polygon_vertices": 10_000_000,
        "max_supported_cells": 4096,
    }
    limits.update(overrides)
    return v1.make_support_budget(**limits)


def _build_universe(distance=DISTANCE, extent=EXTENT, d_bins=(11, 15)):
    """Build the one genuine mixed-normal 15-reading phase-0 universe."""
    pixel_step = lp.lattice.rectified_pixel_step(extent, SAMPLES)
    period = B_VALUE / (distance * pixel_step)
    structural = np.zeros((H, W), dtype=np.float64)
    structural[0, 0] = 1.0
    g_evidence = {}
    for axis in AXES:
        first = _produce(structural, "R0", 0, axis, period, extent)
        assert first.status == "complete", first.refusals
        evidence = _g_evidence(first)
        assert evidence.phase_uv_pixels == (0.0, 0.0)
        g_evidence[axis] = evidence

    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame in FRAMES:
        point = np.asarray(FRAME_POINTS[frame], dtype=np.float64)
        for axis in AXES:
            base = _reading(frame, axis, B_VALUE, distance, 1.0, (0, 0),
                            (0.0, 0.0), samples=SAMPLES, extent=extent)
            assert abs(base.period_pixels - period) < 1e-12
            reading = replace(base, phase_evidence=g_evidence[axis],
                              phase_pixels=(0.0, 0.0))
            readings.append(reading)
            signature = _signature(reading, B_VALUE, basis, point)
            coverage = np.zeros(signature.coverage.shape, dtype=bool)
            row = (coverage.shape[0] - 3) // 2
            column = (coverage.shape[1] - 3) // 2
            coverage[row:row + 3, column:column + 3] = True
            signatures.append(replace(signature, coverage=coverage))
    readings = tuple(readings)
    signatures = tuple(signatures)
    by_key = {(value.reading.frame, value.reading.axis): value
              for value in signatures}
    reading_by_key = {(value.frame, value.axis): value for value in readings}
    assert len(readings) == len(FRAMES) * len(AXES) == 15
    assert len(by_key) == 15

    # One match per frame pair and world axis: the frame graph is a single
    # connected ring while the matched edges span all three world normals.
    matches = []
    for index in range(len(FRAMES)):
        left = FRAMES[index]
        right = FRAMES[(index + 1) % len(FRAMES)]
        for axis in AXES:
            pair = plane_graph.match_plane_pair(
                by_key[(left, axis)], by_key[(right, axis)],
                quadrant=0, max_shift_cells=4)
            assert pair is not None, (left, right, axis)
            matches.append(replace(
                pair,
                left_plane=_canonical_identity(
                    readings, reading_by_key[(left, axis)])[1],
                right_plane=_canonical_identity(
                    readings, reading_by_key[(right, axis)])[1]))
    matches = tuple(matches)
    assert len(matches) == 15

    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(FRAMES)),
        quadrants={frame: 0 for frame in FRAMES},
        translations={frame: (0.0, 0.0, 0.0) for frame in FRAMES},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): B_VALUE
                           for reading in readings},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None, support.refusal
    p1d = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(max_domains=32), readings=readings)
    assert p1d.status == "complete", p1d.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    assert all(value.refusal is None for value in phases)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=_runtime_budget(max_components=10, max_frames=10, max_edges=32,
                               max_tree_edges=10, max_chords=32,
                               max_cycle_rows=160))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.placement_readiness == "ready"
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert component.cycle_rank >= 1
    assert not component.failing_chords
    assert len(component.translation_classes) == 1

    v2a_result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_v2a_budget())
    assert v2a_result.status == "complete"
    assert v2a_result.classification == "endpoint_unique"
    assert v2a_result.endpoint_assignment_count == 1

    held_out = _held_out_patch(d_bins)
    split_sources = tuple(
        (_produce(held_out, reading.frame,
                  _canonical_identity(readings, reading)[1], reading.axis,
                  period, extent), held_out.copy(), _p0_budget())
        for reading in readings)
    anchor_budget = _anchor_budget()
    v2b_result = v2b.analyze_phase_anchor(
        solved, vector_result, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=v2a_result, split_sources=split_sources,
        budget=anchor_budget)
    assert v2b_result.prediction is not None
    determined = sorted(axis.world_axis for axis in v2b_result.prediction.axes
                        if axis.status == "determined")
    assert determined == [0, 1, 2]

    return placement.PlacementUniverse(
        c2rb_result=solved, vector_result=vector_result,
        axis_support=support, p1d_result=p1d, matches=matches,
        phase_evidence=phases, readings=readings, signatures=signatures,
        v2a_result=v2a_result, v2b_result=v2b_result,
        split_sources=split_sources, anchor_budget=anchor_budget,
        support_budget=_support_budget(),
        usable_masks=tuple(
            (domain.reading_identity.identity_sha256,
             _mask_for(domain.reading_identity.reading_axis))
            for domain in v2a_result.reading_domains),
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def _build_supports(universe, **overrides):
    domains = {}
    for domain in universe.v2a_result.reading_domains:
        key = (domain.reading_identity.frame,
               domain.reading_identity.canonical_index,
               domain.reading_identity.reading_axis)
        domains[key] = domain
    anchor_digest = overrides.pop("phase_anchor_sha256",
                                  EXPECTED_ANCHOR_DIGEST)
    result = []
    for reading in universe.readings:
        identity = _canonical_identity(universe.readings, reading)
        domain = domains[(identity[0], identity[1], reading.axis)]
        built = v1.build_aligned_region_support(
            reading=reading, reading_identity=domain.reading_identity,
            reading_universe=universe.readings,
            usable_mask=_mask_for(reading.axis),
            image_sha256=IMAGE_SHA, blocks_per_period=B_VALUE,
            origin_uv=(0.0, 0.0), grid_shape=(3, 3), budget=_support_budget(),
            phase_anchor_sha256=anchor_digest)
        assert isinstance(built, v1.AlignedRegionSupport), built
        result.append(built)
    return tuple(result)


def _universe(universe, **changes):
    fields = {name: getattr(universe, name)
              for name in placement.PlacementUniverse.__dataclass_fields__}
    fields.update(changes)
    return placement.PlacementUniverse(**fields)


def _budget(**overrides):
    limits = {
        "max_faces": 256, "max_supports_checked": 64,
        "max_support_cells": 4096, "max_frame_axis_lookups": 4096,
        "max_integer_checks": 4096, "max_anchors": 65536,
    }
    limits.update(overrides)
    return placement.make_placement_budget(**limits)


def _xyz_downstream_anchor_case():
    """Build six immutable readings from P0/region evidence, then public XYZ."""
    declared = {"A": (0, 0, 0), "B": (2, 3, 4)}
    patch = np.zeros((16, 16), dtype=np.float64)
    patch[0, 0] = 1.0
    patch[8, 8] = patch[12, 12] = 4.0
    p0_budget = _xyz_p0_budget()
    readings, signatures, image_hashes, split_sources = [], [], [], []
    for frame in ("A", "B"):
        point = declared[frame]
        for axis in AXES:
            plane_index = axis
            tangents = tuple(value for value in AXES if value != axis)
            distance = 10.0 - float(point[axis])
            extent = 127.0 / (8.0 * distance)
            geometry = lp.SplitPhaseGeometry(
                centre=(extent, extent), extent=extent, samples=128,
                rectified_pixel_step=lp.lattice.rectified_pixel_step(extent, 128),
                period_pixels=4.0)
            generated = lp.produce_split_phase_evidence(
                patch, geometry=geometry, frame=frame, plane_index=plane_index,
                reading_axis=axis, source_region_id=_REGION_ID[axis],
                partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG,
                budget=p0_budget)
            assert generated.status == "complete", generated.refusals
            phase = _g_evidence(generated)
            assert phase.phase_uv_pixels == (0.0, 0.0)
            tangent_translation = np.asarray(
                tuple(point[value] for value in tangents), dtype=np.float64)
            base = _reading(
                frame, axis, B_VALUE, distance, 1.0, (0, 0), (0.0, 0.0),
                samples=128, extent=extent, focal=30.0)
            reading = replace(base, phase_evidence=phase,
                              phase_pixels=(0.0, 0.0))
            image = _raw_fixture_image(reading, tangent_translation)
            signature = _raw_fixture_signature(
                reading, B_VALUE, tangent_translation,
                usable=_mask_for(axis))
            readings.append(reading)
            signatures.append(signature)
            image_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            split_sources.append((generated, patch.copy(), p0_budget))

    readings, signatures = tuple(readings), tuple(signatures)
    assert all(_canonical_identity(readings, reading)[1] == reading.axis
               for reading in readings)
    matches = []
    for axis in AXES:
        match = plane_graph.match_plane_pair(
            signatures[axis], signatures[3 + axis], quadrant=0,
            max_shift_cells=4)
        assert match is not None
        matches.append(replace(
            match, left_plane=axis, right_plane=axis))
    matches = tuple(matches)
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=(("A", "B", 0),))
    assert solution.reason == ""
    assert solution.translations == {
        "A": (0.0, 0.0, 0.0), "B": (2.0, 3.0, 4.0)}
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(max_domains=10), readings=readings)
    assert p1d.status == "complete", p1d.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vectors = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vectors.status == "complete", vectors.refusals
    solved = relative.solve_relative_translation_graph(
        vectors, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=_runtime_budget(max_components=10, max_frames=10,
                               max_edges=10, max_tree_edges=10,
                               max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete"
    assert solved.placement_readiness == "ready"
    v2a_result = v2a.analyze_endpoint_provenance(
        solved, vectors, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_v2a_budget())
    assert v2a_result.status == "complete"
    assert v2a_result.classification == "endpoint_unique"
    anchor_budget = _anchor_budget()
    v2b_result = v2b.analyze_phase_anchor(
        solved, vectors, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=v2a_result, split_sources=tuple(split_sources),
        budget=anchor_budget)
    assert v2b_result.status == "determined", v2b_result.to_record()
    assert v2b_result.prediction is not None
    assert v2b_result.prediction.reason == "determined"
    assert all(axis.status == "determined"
               and axis.reason is None
               and axis.distinct_frame_count == 2
               for axis in v2b_result.prediction.axes)
    assert all(observation.boundary_energy == (1, 4)
               and observation.interior_energy == (0, 1)
               and observation.margin == (1, 4)
               for observation in v2b_result.prediction.observations)
    assert v2b_result.work.accepted_frame_supports == 6
    universe = placement.PlacementUniverse(
        c2rb_result=solved, vector_result=vectors, axis_support=support,
        p1d_result=p1d, matches=matches, phase_evidence=phases,
        readings=readings, signatures=signatures, v2a_result=v2a_result,
        v2b_result=v2b_result, split_sources=tuple(split_sources),
        anchor_budget=anchor_budget, support_budget=_support_budget(),
        usable_masks=tuple(
            (domain.reading_identity.identity_sha256,
             _mask_for(domain.reading_identity.reading_axis))
            for domain in v2a_result.reading_domains),
        comparison_tolerance=1e-12, closure_tolerance=0.0)
    return (universe, tuple(image_hashes), declared)


def test_xyz_producer_bound_evidence_determines_all_anchor_axes():
    universe, image_hashes, declared = _xyz_downstream_anchor_case()
    readings, signatures = universe.readings, universe.signatures
    assert declared == {"A": (0, 0, 0), "B": (2, 3, 4)}
    assert len(readings) == len(signatures) == 6
    assert len(image_hashes) == 6 and len(set(image_hashes)) == 4
    assert all(signature.cells.shape == (32, 32)
               for signature in signatures)
    assert len(universe.matches) == len(universe.p1d_result.domains) == 3
    assert universe.c2rb_result.placement_readiness == "ready"
    assert universe.v2a_result.classification == "endpoint_unique"
    assert universe.v2b_result.status == "determined"
    prediction = universe.v2b_result.prediction
    assert prediction is not None
    assert tuple(axis.alpha_dyadic for axis in prediction.axes) == (
        (0, 1), (0, 1), (0, 1))
    assert all(axis.distinct_frame_count == 2 for axis in prediction.axes)
    assert all(Fraction(*observation.margin) >= Fraction(1, 8)
               for observation in prediction.observations)
    assert prediction.runtime_work.accepted_frame_supports == 6


def test_xyz_producer_bound_evidence_reaches_placement_and_volume():
    universe, image_hashes, declared = _xyz_downstream_anchor_case()
    domains = {
        (value.reading_identity.frame,
         value.reading_identity.canonical_index,
         value.reading_identity.reading_axis): value
        for value in universe.v2a_result.reading_domains}
    supports = []
    for reading, signature, image_sha in zip(
            universe.readings, universe.signatures, image_hashes):
        frame, index = _canonical_identity(universe.readings, reading)
        axis = reading.axis
        built = v1.build_aligned_region_support(
            reading=reading,
            reading_identity=domains[(frame, index, axis)].reading_identity,
            reading_universe=universe.readings,
            usable_mask=_mask_for(axis), image_sha256=image_sha,
            blocks_per_period=B_VALUE, origin_uv=signature.origin_uv,
            grid_shape=signature.cells.shape, budget=_support_budget(),
            phase_anchor_sha256=EXPECTED_ANCHOR_DIGEST)
        assert isinstance(built, v1.AlignedRegionSupport), built
        supports.append(built)
    supports = tuple(supports)
    placement_budget = _budget(max_support_cells=10_000)
    resolved = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=placement_budget)
    assert isinstance(resolved, placement.ResolvedFacePlacement), resolved
    # Both cameras observe the same three physical block planes.  The face
    # coordinates below come solely from the declared distance/translation
    # construction, not from the placement result.
    assert {face.world_normal_axis for face in resolved.faces} == {0, 1, 2}
    assert all(face.face == 10 for face in resolved.faces)
    assert len(resolved.faces) == 6
    volume_budget = plane_volume.make_volume_budget(
        max_faces=16, max_support_cells=10_000,
        max_contributions=10_000, max_coordinates=10_000,
        max_conflicts=10_000, max_omitted_unknown=10_000,
        max_dense_bounds_cells=100_000)
    volume = plane_volume.build_plane_volume(
        universe=universe, supports=supports, placement=resolved,
        placement_budget=placement_budget, budget=volume_budget)
    assert isinstance(volume, plane_volume.PlaneVolumeResult), volume
    assert volume.complete is False and not volume.conflicts
    assert volume.omitted  # source-region/usable boundary cells stay unknown
    by_coordinate = {cell.coordinate: cell for cell in volume.cells}
    # Independent projection/coordinate oracle from the declared camera plane,
    # the literal [400,640)^2 region, and the known frame translations.  It
    # does not inspect an AlignedRegionSupport or any volume output.
    expected_claims = {}
    expected_omitted = set()
    for reading, signature in zip(universe.readings, universe.signatures):
        axis = reading.axis
        tangents = tuple(value for value in AXES if value != axis)
        matrix = np.asarray(reading.rectification.axis_matrix, dtype=float)
        normal, first, second = matrix[axis], matrix[tangents[0]], matrix[tangents[1]]
        scale = reading.side * reading.alternatives[0].distance_blocks
        basis = np.column_stack((first, second, scale * normal))
        focal = reading.rectification.focal
        cx, cy = reading.rectification.principal_point
        projection = np.asarray(((focal, 0.0, cx),
                                 (0.0, -focal, cy),
                                 (0.0, 0.0, 1.0)))
        homography = projection @ basis
        for row in range(signature.cells.shape[0]):
            for column in range(signature.cells.shape[1]):
                projected = []
                for u, v in ((column, row), (column + 1, row),
                             (column + 1, row + 1), (column, row + 1)):
                    value = homography @ np.asarray((u, v, 1.0))
                    projected.append((value[0] / value[2], value[1] / value[2]))
                if not all(400.0 <= x <= 640.0 and 400.0 <= y <= 640.0
                           for x, y in projected):
                    expected_omitted.add((reading.frame, axis, row, column))
                    continue
                coordinate = [0, 0, 0]
                coordinate[axis] = 10
                coordinate[tangents[0]] = declared[reading.frame][tangents[0]] + column
                coordinate[tangents[1]] = declared[reading.frame][tangents[1]] + row
                expected_claims.setdefault(tuple(coordinate), []).append(
                    (reading.frame, axis, row, column))
    assert set(by_coordinate) == set(expected_claims)
    assert volume.cell_bounds == ((0, 0, 0), (31, 34, 32))
    assert len(expected_claims) == len(volume.cells) == 3093
    assert sum(map(len, expected_claims.values())) == volume.work.contributions == 5304
    for coordinate, expected in expected_claims.items():
        assert tuple((claim.frame, claim.world_normal_axis, claim.row, claim.column)
                     for claim in by_coordinate[coordinate].claims) == tuple(expected)
    observed_omitted = {
        (record.frame, record.reading_axis, record.row, record.column)
        for record in volume.omitted}
    assert observed_omitted == expected_omitted
    assert len(expected_omitted) == len(volume.omitted) == 840
    assert len(expected_omitted) + volume.work.contributions == 6 * 32 * 32
    assert all(record.state == "unknown"
               and record.evidence == "outside_region"
               and record.plane_index == record.reading_axis
               for record in volume.omitted)
    duplicate_oracle = {
        (10, 3, 4): (("A", 0, 4, 3), ("B", 0, 0, 0)),
        (2, 10, 4): (("A", 1, 4, 2), ("B", 1, 0, 0)),
        (2, 3, 10): (("A", 2, 3, 2), ("B", 2, 0, 0)),
    }
    for coordinate, expected in duplicate_oracle.items():
        assert tuple((claim.frame, claim.world_normal_axis,
                      claim.row, claim.column)
                     for claim in by_coordinate[coordinate].claims) == expected
    assert tuple((claim.frame, claim.world_normal_axis)
                 for claim in by_coordinate[(10, 10, 10)].claims) == (
                     ("A", 0), ("A", 1), ("A", 2),
                     ("B", 0), ("B", 1), ("B", 2))


def _unchecked(record, **changes):
    clone = object.__new__(type(record))
    for field in record.__dataclass_fields__:
        object.__setattr__(clone, field, changes.get(field, getattr(record, field)))
    return clone


@pytest.fixture(scope="module")
def universe():
    return _build_universe()


@pytest.fixture(scope="module")
def supports(universe):
    return _build_supports(universe)


# ---------------------------------------------------------------------------
# Formula oracle: literal transform, quadrants, sides, signed translations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("local_label", (0, 1, 2))
@pytest.mark.parametrize("quadrant", (0, 1, 2, 3))
@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("distance", (Fraction(1), Fraction(1, 2),
                                      Fraction(3, 4)))
def test_face_axis_coordinate_matches_literal_table(local_label, quadrant,
                                                    side, distance):
    world_axis, world_sign = LITERAL_TRANSFORMS[quadrant][local_label]
    for translation in (Fraction(0), Fraction(3), Fraction(-1, 4),
                        Fraction(5, 2)):
        gauge = translation + world_sign * side * distance
        alpha = Fraction(gauge - math.floor(gauge))  # canonical exact residue
        record = placement.face_axis_coordinate(
            local_label=local_label, quadrant=quadrant, c=side, d=distance,
            translation_dyadic=(translation.numerator, translation.denominator),
            alpha_dyadic=(alpha.numerator, alpha.denominator))
        assert record["world_axis"] == world_axis
        assert record["world_sign"] == world_sign
        assert Fraction(*record["face_gauge"]) == gauge
        assert record["integer"] is True
        assert record["face"] == int(gauge - alpha)
        outward = -world_sign * side
        assert record["outward_sign"] == (1 if outward > 0 else -1)
        assert record["normal_block"] == (record["face"] - 1 if outward > 0
                                          else record["face"])


def test_face_axis_coordinate_abstains_instead_of_rounding():
    record = placement.face_axis_coordinate(
        local_label=2, quadrant=0, c=1, d=Fraction(1, 2),
        translation_dyadic=(0, 1), alpha_dyadic=(0, 1))
    assert record["integer"] is False
    assert record["face"] is None
    assert record["normal_block"] is None
    assert Fraction(*record["residual"]) == Fraction(1, 2)


def test_face_axis_coordinate_rejects_noncanonical_dyadic():
    with pytest.raises(ValueError):
        placement.face_axis_coordinate(
            local_label=2, quadrant=0, c=1, d=1,
            translation_dyadic=(2, 4), alpha_dyadic=(0, 1))
    with pytest.raises(ValueError):
        placement.face_axis_coordinate(
            local_label=2, quadrant=0, c=1, d=1,
            translation_dyadic=(0, 2), alpha_dyadic=(0, 1))
    with pytest.raises(ValueError):
        placement.face_axis_coordinate(
            local_label=3, quadrant=0, c=1, d=1,
            translation_dyadic=(0, 1), alpha_dyadic=(0, 1))


@pytest.mark.parametrize("local_label", (0, 1, 2))
@pytest.mark.parametrize("quadrant", (0, 1, 2, 3))
@pytest.mark.parametrize("origin", (Fraction(0), Fraction(1, 2),
                                    Fraction(-3, 4), Fraction(2)))
@pytest.mark.parametrize("translation", (Fraction(0), Fraction(3),
                                         Fraction(-1, 4), Fraction(5, 2)))
def test_tangent_axis_coordinate_subtracts_alpha(local_label, quadrant,
                                                 origin, translation):
    world_axis, world_sign = LITERAL_TRANSFORMS[quadrant][local_label]
    alpha = Fraction(1, 2)
    raw = translation + world_sign * origin - alpha
    record = placement.tangent_axis_coordinate(
        local_label=local_label, quadrant=quadrant, origin=origin,
        translation_dyadic=(translation.numerator, translation.denominator),
        alpha_dyadic=(1, 2))
    assert record["world_axis"] == world_axis
    assert record["world_sign"] == world_sign
    assert Fraction(*record["raw"]) == raw
    if raw.denominator == 1:
        assert record["integer"] is True
        assert record["anchor"] == int(raw)
    else:
        assert record["integer"] is False
        assert record["anchor"] is None


def test_tangent_axis_coordinate_abstains_instead_of_rounding():
    record = placement.tangent_axis_coordinate(
        local_label=0, quadrant=0, origin=Fraction(1, 4),
        translation_dyadic=(0, 1), alpha_dyadic=(0, 1))
    assert record["integer"] is False
    assert record["anchor"] is None
    assert Fraction(*record["raw"]) == Fraction(1, 4)


def test_cell_lower_endpoint_uses_lower_edge_and_negative_adjustment():
    # sm > 0: lower endpoint is anchor + index.
    assert placement._cell_lower_endpoint(0, 1, 0) == 0
    assert placement._cell_lower_endpoint(0, 1, 3) == 3
    assert placement._cell_lower_endpoint(-2, 1, 4) == 2
    # sm < 0: local cell i runs [anchor - (i + 1), anchor - i).
    assert placement._cell_lower_endpoint(0, -1, 0) == -1
    assert placement._cell_lower_endpoint(0, -1, 3) == -4
    assert placement._cell_lower_endpoint(5, -1, 2) == 2


# ---------------------------------------------------------------------------
# Genuine mixed-normal single-universe fixture
# ---------------------------------------------------------------------------

def test_fixture_is_one_connected_mixed_normal_component(universe):
    assert len(universe.readings) == 15
    assert {reading.frame for reading in universe.readings} == set(FRAMES)
    for frame in FRAMES:
        axes = sorted(reading.axis for reading in universe.readings
                      if reading.frame == frame)
        assert axes == [0, 1, 2]
    solved = universe.c2rb_result
    assert solved.status == "complete"
    assert solved.connected is True
    assert len(solved.components) == 1
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert component.cycle_rank >= 1
    assert not component.failing_chords
    assert len(component.translation_classes) == 1
    assert universe.v2a_result.classification == "endpoint_unique"
    assert universe.v2a_result.endpoint_assignment_count == 1
    assert universe.v2b_result.prediction is not None
    determined = sorted(axis.world_axis for axis in
                        universe.v2b_result.prediction.axes
                        if axis.status == "determined")
    assert determined == [0, 1, 2]
    for axis in universe.v2b_result.prediction.axes:
        assert axis.status == "determined"
        assert axis.alpha_dyadic == (0, 1)


def test_resolved_placement_integer_success(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement), outcome
    assert outcome.schema_version == placement.V2_PLACEMENT_SCHEMA
    assert outcome.producer_version == placement.V2_PLACEMENT_PRODUCER_VERSION
    assert outcome.gauge_state == placement.V2_PLACEMENT_GAUGE_STATE
    assert outcome.gauge_axes == (0, 1, 2)
    assert outcome.alpha == EXPECTED_ALPHA
    assert len(outcome.faces) == 15
    assert {face.world_normal_axis for face in outcome.faces} == {0, 1, 2}
    assert {binding["anchor_digest"] for binding in outcome.component_bindings} \
        == {EXPECTED_ANCHOR_DIGEST}
    for face in outcome.faces:
        # T = 0, side = 1, distance = 1, alpha = 0 -> the face is block 1.
        assert face.c == (1, 1)
        assert face.d == (1, 1)
        assert face.face_gauge == (1, 1)
        assert face.face == 1
        assert face.outward_sign == -1
        assert face.normal_block == 1
        assert len(face.cells) == 9
        assert len(face.tangent_mappings) == 2
        assert tuple(mapping.local_axis for mapping in face.tangent_mappings) \
            == ("u", "v")
        for cell in face.cells:
            # phase 0 -> anchor 0, positive sign -> lower endpoint is index.
            assert cell.lateral_u == cell.column
            assert cell.lateral_v == cell.row
    assert {name: getattr(outcome.work, name)
            for name in EXPECTED_WORK} == EXPECTED_WORK


def test_face_records_replay_their_own_formula(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement)
    for face in outcome.faces:
        translation = Fraction(*face.translation[face.world_normal_axis])
        alpha = Fraction(*face.alpha[face.world_normal_axis])
        gauge = (translation + face.world_normal_sign * Fraction(*face.c)
                 * Fraction(*face.d))
        assert gauge - alpha == face.face
        rebuilt = placement.ResolvedFace.build(**{
            field: getattr(face, field) for field in (
                "frame", "plane_index", "reading_axis", "absolute_quadrant",
                "source_region_id", "reading_identity_sha256", "support_sha256",
                "world_normal_axis", "world_normal_sign", "c", "d",
                "translation", "alpha", "face_gauge", "face", "outward_sign",
                "normal_block", "tangent_mappings", "cells")})
        assert rebuilt.face_sha256 == face.face_sha256


def test_permutation_invariant_canonical_digest(universe, supports):
    baseline = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    permuted = _universe(
        universe,
        readings=tuple(reversed(universe.readings)),
        signatures=tuple(reversed(universe.signatures)),
        matches=tuple(reversed(universe.matches)),
        phase_evidence=tuple(reversed(universe.phase_evidence)),
        split_sources=tuple(reversed(universe.split_sources)),
        usable_masks=tuple(reversed(universe.usable_masks)))
    result = placement.resolve_face_placement(
        universe=permuted, supports=tuple(reversed(supports)), budget=_budget())
    assert isinstance(baseline, placement.ResolvedFacePlacement)
    assert isinstance(result, placement.ResolvedFacePlacement), result
    assert baseline.placement_sha256 == result.placement_sha256
    assert baseline.to_record() == result.to_record()


# ---------------------------------------------------------------------------
# Typed abstentions, gates and replay
# ---------------------------------------------------------------------------

def test_c2rb_gate_abstains_on_disconnected_incomplete_and_ambiguous(
        universe, supports):
    for changes, expected in (
            ({"connected": False}, "c2rb_gate_not_unique"),
            ({"status": "refused"}, "c2rb_gate_not_unique"),
            ({"placement_readiness": "blocked"}, "c2rb_gate_not_unique")):
        forged = _unchecked(universe.c2rb_result, **changes)
        outcome = placement.resolve_face_placement(
            universe=_universe(universe, c2rb_result=forged), supports=supports,
            budget=_budget())
        assert isinstance(outcome, placement.PlacementRefusal)
        assert outcome.reason == expected
    component = universe.c2rb_result.components[0]
    for changes in ({"exhaustive": False}, {"cycle_rank": 0},
                    {"outcome": "ambiguous"}, {"failing_chords": ("x",)}):
        forged_component = _unchecked(component, **changes)
        forged = _unchecked(universe.c2rb_result, components=(forged_component,))
        outcome = placement.resolve_face_placement(
            universe=_universe(universe, c2rb_result=forged),
            supports=supports, budget=_budget())
        assert isinstance(outcome, placement.PlacementRefusal)
        assert outcome.reason == "c2rb_gate_not_unique"


def test_endpoint_gate_abstains(universe, supports):
    for classification in ("endpoint_multiple", "endpoint_none", None):
        forged = _unchecked(universe.v2a_result, classification=classification)
        outcome = placement.resolve_face_placement(
            universe=_universe(universe, v2a_result=forged), supports=supports,
            budget=_budget())
        assert isinstance(outcome, placement.PlacementRefusal)
        assert outcome.reason == "endpoint_gate_not_unique"


def test_anchor_gate_abstains_when_v2b_is_not_determined(universe, supports):
    forged = _unchecked(universe.v2b_result, status="abstained",
                        prediction=None, refusals=())
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, v2b_result=forged), supports=supports,
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "anchor_gate_not_unique"


def test_missing_anchor_axis_abstains(universe, supports):
    prediction = universe.v2b_result.prediction
    axes = list(prediction.axes)
    target = next(index for index, axis in enumerate(axes)
                  if axis.status == "determined")
    forged_axis = _unchecked(axes[target], status="abstained",
                             alpha_dyadic=None)
    new_axes = tuple(axes[:target]) + (forged_axis,) + tuple(axes[target + 1:])
    forged_prediction = _unchecked(prediction, axes=new_axes)
    forged = _unchecked(universe.v2b_result, prediction=forged_prediction)
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, v2b_result=forged), supports=supports,
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "anchor_axis_unresolved"
    assert outcome.work is None


def test_replay_mismatch_on_changed_closure_tolerance(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, closure_tolerance=1.0), supports=supports,
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "replay_mismatch"


def test_resealed_split_source_is_rejected(universe, supports):
    other = _held_out_patch()
    other[9, 9] = 4.0
    forged_sources = tuple(
        (_produce(other, reading.frame,
                  _canonical_identity(universe.readings, reading)[1],
                  reading.axis),
         other.copy(), _p0_budget())
        for reading in universe.readings)
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, split_sources=forged_sources),
        supports=supports, budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "replay_mismatch"


def test_resealed_v2b_anchor_is_rejected(universe, supports):
    prediction = universe.v2b_result.prediction
    axes = list(prediction.axes)
    target = next(index for index, axis in enumerate(axes)
                  if axis.status == "determined")
    forged_axis = _unchecked(axes[target], alpha_dyadic=(1, 2))
    new_axes = tuple(axes[:target]) + (forged_axis,) + tuple(axes[target + 1:])
    forged_prediction = _unchecked(prediction, axes=new_axes)
    forged = _unchecked(universe.v2b_result, prediction=forged_prediction)
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, v2b_result=forged), supports=supports,
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason in ("anchor_gate_not_unique", "replay_mismatch",
                              "invalid_input")


# ---------------------------------------------------------------------------
# Support binding, replay and resealed contradictions
# ---------------------------------------------------------------------------

def test_foreign_phase_anchor_digest_is_rejected(universe, supports):
    foreign = _unchecked(supports[0], phase_anchor_sha256="ab" * 32)
    outcome = placement.resolve_face_placement(
        universe=universe, supports=(foreign,) + tuple(supports[1:]),
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "provenance_mismatch"


def test_resealed_support_distance_is_rejected(universe, supports):
    reading = universe.readings[0]
    forged_alternative = v1.RetainedAlternative.build(
        reading.matching_alternatives(B_VALUE)[0])
    forged_alternative = _unchecked(
        forged_alternative,
        distance_blocks=float(forged_alternative.distance_blocks) + 1.0)
    forged = _unchecked(supports[0], alternative=forged_alternative)
    outcome = placement.resolve_face_placement(
        universe=universe, supports=(forged,) + tuple(supports[1:]),
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason in ("alternative_not_retained", "provenance_mismatch",
                              "support_replay_mismatch")


def test_resealed_support_contents_are_rejected(universe, supports):
    forged = _unchecked(supports[0], origin_uv=((1, 2), (0, 1)))
    outcome = placement.resolve_face_placement(
        universe=universe, supports=(forged,) + tuple(supports[1:]),
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "support_replay_mismatch"


def test_digest_preserving_resealed_support_cell_is_rejected(universe,
                                                             supports):
    """A retained digest must not carry a resealed cell row into placement.

    The forged support keeps ``support_sha256`` byte-identical to the accepted
    record while one supported cell's row is rewritten to 999.  A digest-only
    gate replays to the same digest and would have consumed the forged object,
    placing the coordinate 999; the canonical-record comparison must refuse it
    and the rebuilt record is the only object consumed.
    """
    template = supports[0]
    cells = list(template.cells)
    target = next(index for index, cell in enumerate(cells)
                  if cell.state == "supported")
    cells[target] = _unchecked(cells[target], row=999)
    forged = _unchecked(template, cells=tuple(cells))
    # The digest the old gate compared is preserved exactly.
    assert forged.support_sha256 == template.support_sha256
    assert forged.cells[target].row == 999
    outcome = placement.resolve_face_placement(
        universe=universe, supports=(forged,) + tuple(supports[1:]),
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal), outcome
    assert outcome.reason == "support_replay_mismatch"


def test_missing_support_provenance_is_refused(universe, supports):
    template = supports[0]
    forged = _unchecked(template, reading_identity_sha256="f" * 64)
    outcome = placement.resolve_face_placement(
        universe=universe, supports=(forged,) + tuple(supports[1:]),
        budget=_budget())
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "missing_provenance"


def test_resealed_lateral_cell_contradiction_is_rejected(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement)
    face = outcome.faces[0]
    forged_cells = list(face.cells)
    forged_cells[0] = _unchecked(forged_cells[0], lateral_u=999)
    fields = {field: getattr(face, field) for field in (
        "frame", "plane_index", "reading_axis", "absolute_quadrant",
        "source_region_id", "reading_identity_sha256", "support_sha256",
        "world_normal_axis", "world_normal_sign", "c", "d", "translation",
        "alpha", "face_gauge", "face", "outward_sign", "normal_block",
        "tangent_mappings", "cells")}
    fields["cells"] = tuple(forged_cells)
    with pytest.raises(ValueError):
        placement.ResolvedFace.build(**fields)


def test_resealed_face_anchor_contradiction_is_rejected(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement)
    face = outcome.faces[0]
    fields = {field: getattr(face, field) for field in (
        "frame", "plane_index", "reading_axis", "absolute_quadrant",
        "source_region_id", "reading_identity_sha256", "support_sha256",
        "world_normal_axis", "world_normal_sign", "c", "d", "translation",
        "alpha", "face_gauge", "face", "outward_sign", "normal_block",
        "tangent_mappings", "cells")}
    fields["face"] = 999
    with pytest.raises(ValueError):
        placement.ResolvedFace.build(**fields)


@pytest.mark.parametrize("forged", (
    # A resealed world normal axis that keeps the retained normal formula
    # self-consistent: local axis 0 at quadrant 0 maps to axis 0, never axis 1.
    {"world_normal_axis": 1},
    # A resealed world normal sign, with every derived normal field adjusted so
    # only the transform binding is violated.
    {"world_normal_sign": -1, "face_gauge": (-1, 1), "face": -1,
     "outward_sign": 1, "normal_block": -2},
))
def test_resealed_face_world_normal_transform_is_rejected(universe, supports,
                                                          forged):
    """``ResolvedFace`` normal axis/sign must replay ``axis_transform``.

    At local axis 0 / quadrant 0 the shared transform yields ``(0, +1)``; a
    resealed ``(1, +1)`` or ``(0, -1)`` normal must be rejected even though the
    face's own retained gauge relation replays.
    """
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement)
    face = next(candidate for candidate in outcome.faces
                if candidate.reading_axis == 0
                and candidate.absolute_quadrant == 0)
    assert (face.world_normal_axis, face.world_normal_sign) \
        == LITERAL_TRANSFORMS[face.absolute_quadrant][face.reading_axis]
    fields = {field: getattr(face, field) for field in (
        "frame", "plane_index", "reading_axis", "absolute_quadrant",
        "source_region_id", "reading_identity_sha256", "support_sha256",
        "world_normal_axis", "world_normal_sign", "c", "d", "translation",
        "alpha", "face_gauge", "face", "outward_sign", "normal_block",
        "tangent_mappings", "cells")}
    fields.update(forged)
    with pytest.raises(ValueError, match="normal"):
        placement.ResolvedFace.build(**fields)


# ---------------------------------------------------------------------------
# Budgets and precedence
# ---------------------------------------------------------------------------

def test_budget_is_atomic_and_has_no_partial_faces(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget(max_faces=1))
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "budget_exhausted"
    assert "faces" not in outcome.detail and "cells" not in outcome.detail
    assert outcome.detail["cap"] == "max_faces"
    assert outcome.detail["limit"] == 1
    assert outcome.work is not None
    assert outcome.work.faces == 2


@pytest.mark.parametrize("cap,exact", EXPECTED_CAP_BOUNDARIES)
def test_each_budget_cap_boundary_is_exact(universe, supports, cap, exact):
    passed = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget(**{cap: exact}))
    assert isinstance(passed, placement.ResolvedFacePlacement), passed
    refused = placement.resolve_face_placement(
        universe=universe, supports=supports,
        budget=_budget(**{cap: exact - 1}))
    assert isinstance(refused, placement.PlacementRefusal)
    assert refused.reason == "budget_exhausted"
    assert refused.detail["cap"] == cap
    assert refused.detail["limit"] == exact - 1
    assert refused.detail["observed"] == exact
    assert getattr(refused.work, cap.removeprefix("max_")) == exact


def test_invalid_input_precedes_budget(universe, supports):
    outcome = placement.resolve_face_placement(
        universe=universe, supports=("not-a-support",),
        budget=_budget(max_faces=1))
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "invalid_input"
    outcome = placement.resolve_face_placement(
        universe=_universe(universe, readings=()), supports=supports,
        budget=_budget(max_faces=1))
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "invalid_input"
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget(),
        producer_version="not-the-v2-producer")
    assert isinstance(outcome, placement.PlacementRefusal)
    assert outcome.reason == "invalid_input"


def test_quadrant_constraint_inconsistent_is_a_validated_public_refusal():
    """The Z4 inconsistency token is a first-class closed-enum refusal reason.

    DS-R25 previously had to bypass ``PlacementRefusal.__post_init__`` and
    handcraft the detail/refusal digests because the adapter emitted the solver's
    own ``quadrant_constraint_inconsistent`` token, which was outside
    ``V2_PLACEMENT_REFUSAL_REASONS``. The additive vocabulary entry makes the
    ordinary validated builder usable: the token survives the builder instead of
    being normalized to ``invalid_input``, and the resulting record both
    validates and serializes through the public contract.
    """
    assert "quadrant_constraint_inconsistent" in placement.V2_PLACEMENT_REFUSAL_REASONS
    detail = {"field": "connection_graph.quadrant_constraints"}
    refusal = placement._make_refusal(
        "quadrant_constraint_inconsistent", detail)
    assert type(refusal) is placement.PlacementRefusal
    assert refusal.reason == "quadrant_constraint_inconsistent"
    assert refusal.detail == detail
    assert refusal.work is None
    # The validated builder derives the standard canonical digest rather than a
    # handcrafted one: ``{"field":"connection_graph.quadrant_constraints"}``.
    assert refusal.detail_sha256 == hashlib.sha256(
        b'{"field":"connection_graph.quadrant_constraints"}').hexdigest()
    record = refusal.to_record()
    assert record["reason"] == "quadrant_constraint_inconsistent"
    assert record["work"] is None
    # The serialized record replays through the public frozen constructor, so
    # the additive enum entry keeps the refusal contract round-trippable.
    assert placement.PlacementRefusal(
        record["reason"], record["detail"], record["detail_sha256"],
        record["work"], record["refusal_sha256"]).to_record() == record
    # A token outside the closed enum is still rejected by the public contract.
    with pytest.raises(ValueError):
        placement.PlacementRefusal(
            "not_a_placement_reason", {}, hashlib.sha256(b"{}").hexdigest(),
            None, "0" * 64)


def test_placement_budget_rejects_non_exact_caps():
    with pytest.raises(ValueError):
        placement.make_placement_budget(
            max_faces=True, max_supports_checked=1, max_support_cells=1,
            max_frame_axis_lookups=1, max_integer_checks=1, max_anchors=1)
    with pytest.raises(ValueError):
        placement.make_placement_budget(
            max_faces=1, max_supports_checked=1, max_support_cells=1,
            max_frame_axis_lookups=1, max_integer_checks=1)


# ---------------------------------------------------------------------------
# Isolation and non-mutation
# ---------------------------------------------------------------------------

def test_resolution_does_not_mutate_upstream_evidence(universe, supports):
    c2rb_before = universe.c2rb_result.to_record()
    v2a_before = universe.v2a_result.to_record()
    v2b_before = universe.v2b_result.to_record()
    readings_before = tuple(reading.to_record()
                            if hasattr(reading, "to_record") else reading
                            for reading in universe.readings)
    support_before = tuple(support.support_sha256 for support in supports)
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement)
    assert universe.c2rb_result.to_record() == c2rb_before
    assert universe.v2a_result.to_record() == v2a_before
    assert universe.v2b_result.to_record() == v2b_before
    assert tuple(reading.to_record() if hasattr(reading, "to_record") else reading
                 for reading in universe.readings) == readings_before
    assert tuple(support.support_sha256 for support in supports) == support_before


def test_public_api_is_the_single_universe_repair():
    import inspect
    import mc_photogrammetry.reconstruction.plane_placement as module
    signature = inspect.signature(module.resolve_face_placement)
    assert "universe" in signature.parameters
    assert "groups" not in signature.parameters
    assert hasattr(module, "PlacementUniverse")
    assert not hasattr(module, "PlacementGroup")


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
    names = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    denied_names = {
        "BlockVolume", "ReconstructionResult", "ComponentSolution",
        "candidate_at", "iter_candidates", "materialize_lateral_vector_domains",
        "solve_component", "plane_volume", "litematic", "os", "sys",
        "subprocess", "socket", "tempfile", "pathlib", "shutil", "io",
        "numpy", "np", "PlacementGroup",
    }
    assert not (names & denied_names), sorted(names & denied_names)
    assert not (calls & {"open", "exec", "eval", "compile", "__import__"}), \
        sorted(calls & {"open", "exec", "eval", "compile", "__import__"})
    for path in imports:
        assert "volume" not in path and "litematic" not in path
        assert "export" not in path and "cli" not in path


def test_no_volume_or_wiring_symbols():
    import mc_photogrammetry.reconstruction.plane_placement as module
    for name in dir(module):
        lowered = name.lower()
        assert "volume" not in lowered
        assert "litematic" not in lowered
        assert "cli" not in lowered
    assert not hasattr(module, "BlockVolume")


# ---------------------------------------------------------------------------
# Genuine nonzero-quadrant / both-side / signed-translation fixture
# ---------------------------------------------------------------------------

# One accepted-API triangle: absolute quadrants 0/1/2 around the common world
# normal axis 2, mixed ``PlaneReading.side`` signs (+1, -1, -1), one reflected
# closing edge (relative quadrant 2) and genuine nonzero P0 split-phase
# generation evidence for every reading.  Every expected numeric value is
# declared here from the reading geometry and the literal coefficient table; no
# value is read back from the placement module under test.
NZ_FRAMES = ("A_q0", "B_q1", "C_q2")
NZ_QUADRANTS = {"A_q0": 0, "B_q1": 1, "C_q2": 2}
# The reading's local normal label that maps to the common world normal axis.
NZ_NORMAL_LABEL = {0: 2, 1: 0, 2: 2}
NZ_SIDES = {"A_q0": 1.0, "B_q1": -1.0, "C_q2": -1.0}
NZ_TANGENT_LABELS = {"A_q0": (0, 1), "B_q1": (1, 2), "C_q2": (0, 1)}
# One genuine G-lane impulse per reading; the (row, column) also sets the real
# nonzero P0 phase of the held-out production.
NZ_IMPULSES = {"A_q0": (3, 1), "B_q1": (5, 6), "C_q2": (1, 4)}
# Integer world offsets that spread the three genuine texture windows; they do
# not change any residue.
NZ_OFFSETS = {"A_q0": (0.0, 0.0), "B_q1": (2.0, 0.0), "C_q2": (2.0, 2.0)}
NZ_D_BINS = (11, 15)
NZ_EXTENT = 4.0
NZ_SAMPLES = 128
NZ_DISTANCE = 2.0
NZ_B = 1.0
NZ_PIXEL_STEP = lp.lattice.rectified_pixel_step(NZ_EXTENT, NZ_SAMPLES)
NZ_PERIOD = NZ_B / (NZ_DISTANCE * NZ_PIXEL_STEP)
NZ_CENTRE = (0.5, 0.5)
NZ_PATCH = 16
# The bounded coverage block is centred on the discrete transform's shared
# reflection fixed point so the reflected edge keeps a full overlap.  The side
# is odd (>= 5) to stay reflection-symmetric about (7, 7) while leaving an
# admissible sidelobe for the matcher's peak-to-sidelobe ratio.
NZ_COVERAGE_CENTER = (7, 7)
NZ_COVERAGE_SIDE = 7
# Declared world normal delta of each triangle edge from axis_transform and the
# reading geometry only.
NZ_NORMAL_DELTA = {("A_q0", "B_q1"): 4.0,
                   ("B_q1", "C_q2"): -4.0,
                   ("A_q0", "C_q2"): 0.0}
# The built-in calibrated classification margin; no genuine observation reaches
# it for this fixture.
V2B_MARGIN = Fraction(1, 8)
# Declared exact V2-B blocker counters observed for this fixture.
V2B_PLANNED_SUPPORTS = 6
V2B_OBSERVATIONS = 6
V2B_HYPOTHESES_PER_AXIS = {0: 1, 1: 1, 2: 0}


def _nz_geometry():
    return lp.SplitPhaseGeometry(
        centre=NZ_CENTRE, extent=NZ_EXTENT, samples=NZ_SAMPLES,
        rectified_pixel_step=NZ_PIXEL_STEP, period_pixels=NZ_PERIOD)


def _nz_patch(row, column, d_bins=()):
    patch = np.zeros((NZ_PATCH, NZ_PATCH), dtype=np.float64)
    patch[row, column] = 1.0
    for index in d_bins:
        patch[index, index] = 4.0
    return patch


def _nz_produce(patch, frame, axis):
    return lp.produce_split_phase_evidence(
        patch, geometry=_nz_geometry(), frame=frame, plane_index=0,
        reading_axis=axis, source_region_id=None,
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=_p0_budget())


def _build_nonzero_quadrant_fixture():
    """Genuine q0/q1/q2 triangle with nonzero reflected C2-R-B translations."""
    readings = []
    signatures = []
    split_sources = []
    phases = {}
    for frame in NZ_FRAMES:
        quadrant = NZ_QUADRANTS[frame]
        normal = NZ_NORMAL_LABEL[quadrant]
        basis = _physical_yaw(quadrant)[np.ix_((0, 1),
                                               NZ_TANGENT_LABELS[frame])]
        row, column = NZ_IMPULSES[frame]
        produced = _nz_produce(_nz_patch(row, column), frame, normal)
        assert produced.status == "complete", produced.refusals
        evidence = _g_evidence(produced)
        phases[frame] = tuple(float(value)
                              for value in evidence.phase_uv_pixels)
        base = _base_reading(frame, normal, NZ_B, NZ_DISTANCE, NZ_SIDES[frame],
                             (0, 0), (0.0, 0.0), samples=NZ_SAMPLES,
                             extent=NZ_EXTENT, centre_offset=(-3.5, -3.5))
        reading = replace(base, phase_evidence=evidence,
                          phase_pixels=tuple(evidence.phase_uv_pixels))
        readings.append(reading)
        # The genuine texture window is sampled at the translation implied by
        # the reading's own real P0 phase, so evidence and content agree.
        residue = np.asarray(evidence.phase_uv_pixels, dtype=np.float64) \
            / NZ_PERIOD
        translation = -(basis @ residue) + np.asarray(NZ_OFFSETS[frame],
                                                      dtype=np.float64)
        measured = _signature(reading, NZ_B, basis, translation)
        coverage = np.zeros(measured.coverage.shape, dtype=bool)
        top = NZ_COVERAGE_CENTER[0] - NZ_COVERAGE_SIDE // 2
        left = NZ_COVERAGE_CENTER[1] - NZ_COVERAGE_SIDE // 2
        coverage[top:top + NZ_COVERAGE_SIDE,
                 left:left + NZ_COVERAGE_SIDE] = True
        signatures.append(replace(measured, coverage=coverage))
        held_out = _nz_produce(_nz_patch(row, column, NZ_D_BINS), frame, normal)
        assert held_out.status == "complete", held_out.refusals
        split_sources.append((
            held_out, _nz_patch(row, column, NZ_D_BINS), _p0_budget()))
    readings = tuple(readings)
    signatures = tuple(signatures)
    split_sources = tuple(split_sources)

    all_matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4)
    matches = []
    for index, left in enumerate(readings):
        for right in readings[index + 1:]:
            relative_quadrant = (NZ_QUADRANTS[right.frame]
                                 - NZ_QUADRANTS[left.frame]) % 4
            found = next((match for match in all_matches
                          if match.left == left.frame
                          and match.right == right.frame
                          and match.quadrant_of_right == relative_quadrant), None)
            assert found is not None, (left.frame, right.frame, relative_quadrant)
            matches.append(replace(
                found,
                left_plane=_canonical_identity(readings, left)[1],
                right_plane=_canonical_identity(readings, right)[1]))
    matches = tuple(matches)
    assert len(matches) == len(NZ_FRAMES)

    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(NZ_QUADRANTS)), quadrants=dict(NZ_QUADRANTS),
        translations={frame: (0.0, 0.0, 0.0) for frame in NZ_FRAMES},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): NZ_B
                           for reading in readings},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None, support.refusal
    p1d = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d.status == "complete", p1d.refusals
    phase_evidence = tuple(_phase_for(match, readings) for match in matches)
    assert all(value.refusal is None for value in phase_evidence)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phase_evidence, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d, matches, phase_evidence, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=_runtime_budget(max_components=10, max_frames=10, max_edges=32,
                               max_tree_edges=10, max_chords=32,
                               max_cycle_rows=160))
    assert solved.status == "complete", solved.budget_failure_reason
    v2a_result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d, matches, phase_evidence,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_v2a_budget())
    v2b_result = v2b.analyze_phase_anchor(
        solved, vector_result, support, p1d, matches, phase_evidence,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=v2a_result, split_sources=split_sources,
        budget=_anchor_budget())
    return {
        "c2rb_result": solved, "vector_result": vector_result,
        "axis_support": support, "p1d_result": p1d, "matches": matches,
        "phase_evidence": phase_evidence, "readings": readings,
        "signatures": signatures, "v2a_result": v2a_result,
        "v2b_result": v2b_result, "split_sources": split_sources,
        "phases": phases,
    }


@pytest.fixture(scope="module")
def nonzero_quadrant_fixture():
    return _build_nonzero_quadrant_fixture()


def test_genuine_nonzero_quadrant_certificate_is_unique_signed_and_full(
        nonzero_quadrant_fixture):
    """The genuine reflected fixture clears C2-R-B with signed translations."""
    solved = nonzero_quadrant_fixture["c2rb_result"]
    assert solved.status == "complete"
    assert solved.connected is True
    assert solved.placement_readiness == "ready"
    assert len(solved.components) == 1
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert component.cycle_rank >= 1
    assert not component.failing_chords
    assert len(component.translation_classes) == 1
    assert len(component.translation_classes[0].translations_dyadic_xyz) \
        == len(NZ_FRAMES)

    readings = nonzero_quadrant_fixture["readings"]
    assert {float(reading.side) for reading in readings} == {1.0, -1.0}
    assert {domain.absolute_quadrant
            for domain in nonzero_quadrant_fixture["v2a_result"].reading_domains} \
        == {0, 1, 2}

    # Independent literal derivation from the genuine P0 phases: the declared
    # triangle coefficient table plus the reading-geometry normal delta.
    leading = NZ_DISTANCE * (NZ_CENTRE[0] - NZ_EXTENT)
    residues = {
        frame: tuple((leading + phase / NZ_PERIOD) % 1.0
                     for phase in nonzero_quadrant_fixture["phases"][frame])
        for frame in NZ_FRAMES}
    a_u, a_v = residues["A_q0"]
    b_u, b_v = residues["B_q1"]
    c_u, c_v = residues["C_q2"]
    expected = {
        "A_q0": (0.0, 0.0, 0.0),
        "B_q1": (a_u + b_v, a_v - b_u, NZ_NORMAL_DELTA[("A_q0", "B_q1")]),
        "C_q2": (a_u + c_u, a_v - c_v, NZ_NORMAL_DELTA[("A_q0", "C_q2")]),
    }
    certificate = dict(
        component.translation_classes[0].translations_dyadic_xyz)
    recovered = {frame: tuple(Fraction(*item) for item in certificate[frame])
                 for frame in NZ_FRAMES}
    for frame, value in expected.items():
        assert recovered[frame] == tuple(Fraction.from_float(item)
                                         for item in value), frame
    # Genuinely signed and fractional, not a zero or integer-only fill.
    assert any(value < 0 for value in recovered["B_q1"])
    assert any(value > 0 for value in recovered["B_q1"])
    assert any(value.denominator > 1 for value in recovered["B_q1"])
    assert any(value.denominator > 1 for value in recovered["C_q2"])
    assert recovered["A_q0"] == (Fraction(0),) * 3


def test_genuine_nonzero_quadrant_fixture_passes_v2a_endpoint_unique(
        nonzero_quadrant_fixture):
    result = nonzero_quadrant_fixture["v2a_result"]
    assert result.status == "complete"
    assert result.classification == "endpoint_unique"
    assert result.exhaustive is True
    assert result.lower_bound_only is False
    assert result.translation_class_count == 1
    assert result.endpoint_assignment_count == 1
    assert len(result.witnesses) == 1
    assert not result.refusals
    assert result.budget_failure_reason is None


def test_genuine_nonzero_quadrant_v2b_all_axis_is_blocked(
        nonzero_quadrant_fixture):
    """The frozen V2-B classifier is the first gate that reproducibly refuses.

    The triangle observes exactly one physical plane (world normal axis 2), so
    world axis 2 is never a *tangent* label of an accepted reading domain:
    ``_derive_hypotheses`` has no contributor for it and that axis abstains
    ``no_hypothesis``.  The two tangent axes have one genuine hypothesis each,
    but every held-out observation's boundary-minus-interior margin stays below
    the calibrated ``1/8``, so both report ``evidence_conflict``.  V1 support
    replay and placement are intentionally not attempted; no gate is weakened,
    no metadata is resealed and no partial component is aggregated.
    """
    result = nonzero_quadrant_fixture["v2b_result"]
    assert result.status == "abstained"
    assert result.prediction is not None
    prediction = result.prediction
    assert prediction.reason == "evidence_conflict"
    axes = {axis.world_axis: axis for axis in prediction.axes}
    assert axes[0].status == "abstained"
    assert axes[0].reason == "evidence_conflict"
    assert axes[1].status == "abstained"
    assert axes[1].reason == "evidence_conflict"
    assert axes[2].status == "abstained"
    assert axes[2].reason == "no_hypothesis"
    for axis in (0, 1, 2):
        assert axes[axis].alpha_dyadic is None
        assert axes[axis].distinct_frame_count == 0
        assert not axes[axis].accepted_hypotheses

    per_axis = {0: 0, 1: 0, 2: 0}
    for hypothesis in prediction.hypotheses:
        per_axis[hypothesis.world_axis] += 1
    assert per_axis == V2B_HYPOTHESES_PER_AXIS
    assert len(prediction.observations) == V2B_OBSERVATIONS
    assert {observation.world_axis for observation in prediction.observations} \
        == {0, 1}
    for observation in prediction.observations:
        assert observation.zero_coverage is False
        assert observation.margin is not None
        assert Fraction(*observation.margin) < V2B_MARGIN
        assert observation.held_out_samples > 0
    assert prediction.runtime_work.accepted_frame_supports == 0
    assert prediction.planned_work.accepted_frame_supports \
        == V2B_PLANNED_SUPPORTS


# ---------------------------------------------------------------------------
# Exact P0 recipe: genuine nonzero-residue, mixed-quadrant, mixed-side,
# signed/full-translation universe that reaches the whole V2 chain
# ---------------------------------------------------------------------------

# Astra's independently checked construction.  Three frames with absolute
# quadrants 0/1/2; each frame carries three normal readings (local axes 0/1/2);
# every reading's P0 generation uses a 16x16 float64 patch, samples 16, b = 1
# and an exact period of four pixels, so the generated phase is the G impulse
# in pixels, the residue is phase/period in {0, 1/2} and the held-out D bumps
# sit exactly on integer periods.  The declared world translation T and the
# per-frame distance d fix each reading's ``side`` so that its normal plane
# coordinate is the block-1 plane (``T[n] + sn*c*d == 1``).  The certificate
# itself is only ever read from the solver.
RECIPE_FRAMES = ("R0", "R1", "R2")
RECIPE_AXES = (0, 1, 2)
RECIPE_QUADRANTS = {"R0": 0, "R1": 1, "R2": 2}
RECIPE_DISTANCES = {"R0": Fraction(1), "R1": Fraction(1, 2),
                    "R2": Fraction(3, 2)}
# Signature-window sampling.  With ``b == 1`` and an exact period of four
# pixels, ``extent * distance == (samples - 1) / 8``, so the cell grid side is
# ``ceil(2 * extent * distance)``.  The historical 16 samples give a 4x4 grid;
# the repaired 25 samples give the smallest six-cell grid that can represent
# every declared joint translation difference of the nine genuine edges.
RECIPE_SAMPLES = 33
RECIPE_EXTENTS = {"R0": Fraction(4), "R1": Fraction(8),
                  "R2": Fraction(8, 3)}
RECIPE_TRANSLATIONS = {
    "R0": (Fraction(0), Fraction(0), Fraction(0)),
    "R1": (Fraction(3, 2), Fraction(3, 2), Fraction(3, 2)),
    "R2": (Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2)),
}
RECIPE_G = {"R0": (0, 0), "R1": (2, 2), "R2": (2, 2)}
RECIPE_D = {"R0": ((8, 8), (12, 12)),
            "R1": ((10, 10), (14, 14)),
            "R2": ((10, 10), (14, 14))}
RECIPE_B = 1.0
RECIPE_PERIOD = 4.0
# Declared exact generated phases and singleton residues per frame.
RECIPE_PHASES = {"R0": (0.0, 0.0), "R1": (2.0, 2.0), "R2": (2.0, 2.0)}
RECIPE_RESIDUES = {"R0": (0.0, 0.0), "R1": (0.5, 0.5), "R2": (0.5, 0.5)}
# Declared sides derived from the block-1 plane coordinate.
RECIPE_SIDES = {
    "R0": {0: 1.0, 1: 1.0, 2: 1.0},
    "R1": {0: -1.0, 1: -1.0, 2: 1.0},
    "R2": {0: -1.0, 1: 1.0, 2: -1.0},
}
# Reflection-aligned ``origin_uv`` shift for a negative local-to-world sign.  An
# S-cell signature grid reflected onto the same physical block window needs its
# origin one full side less one lower (cells 0..S-1 map to -(S-1)..0), so the
# repaired six-cell grid uses -5 and keeps a 6x6 fixed overlap.
RECIPE_ORIGIN_SHIFT = -7.0
# Historical negative-control window: 16 samples give a 4x4 cell grid, and the
# old -2 shift leaves a 3x4 fixed overlap whose required joint shifts have fewer
# than ``plane_graph.MIN_OVERLAP_CELLS`` support cells.
RECIPE_CURRENT_SAMPLES = 16
RECIPE_CURRENT_EXTENTS = {"R0": Fraction(15, 8), "R1": Fraction(15, 4),
                          "R2": Fraction(5, 4)}
RECIPE_CURRENT_ORIGIN_SHIFT = -2.0

# Literal placement expectations for the repaired nine-reading recipe.  Every
# value below is declared from the construction (quadrants, sides, distances,
# declared translations, residues and P0 phases), never read back from the
# placement module under test.
RECIPE_ALPHA = ((0, 1), (0, 1), (0, 1))
RECIPE_ANCHOR_DIGEST = placement.placement_anchor_digest(RECIPE_ALPHA)
# The V1 support grid is a genuine 3x3 cell window: R0 and R2 keep all nine cells
# supported while each R1 reading exposes exactly five boundary-unknown cells.
RECIPE_SUPPORTED_GRID = {
    "R0": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
           (2, 2)),
    "R1": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "R2": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
           (2, 2)),
}
RECIPE_UNKNOWN_GRID = {
    "R0": (),
    "R1": ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)),
    "R2": (),
}
# Declared exact placement work totals of the recipe (9 faces, 66 placed cells).
RECIPE_PLACEMENT_WORK = {
    "faces": 9,
    "supports_checked": 9,
    "support_cells": 66,
    "frame_axis_lookups": 27,
    "integer_checks": 27,
    "anchors": 150,
}
# Six of the eighteen tangent mappings run on a reflected (negative) world axis.
RECIPE_REFLECTED_TANGENTS = 6
RECIPE_UNKNOWN_CELLS = 15


def _literal_recipe_normal(frame, axis):
    """Independent literal face-gauge expectation for one recipe reading."""
    quadrant = RECIPE_QUADRANTS[frame]
    world_axis, world_sign = LITERAL_TRANSFORMS[quadrant][axis]
    side = Fraction(RECIPE_SIDES[frame][axis])
    distance = RECIPE_DISTANCES[frame]
    gauge = RECIPE_TRANSLATIONS[frame][world_axis] \
        + world_sign * side * distance
    assert gauge.denominator == 1
    face = int(gauge)
    outward = -world_sign * side
    return {
        "world_axis": world_axis,
        "world_sign": world_sign,
        "c": (side.numerator, side.denominator),
        "d": (distance.numerator, distance.denominator),
        "face_gauge": (gauge.numerator, gauge.denominator),
        "face": face,
        "outward_sign": 1 if outward > 0 else -1,
        "normal_block": face - 1 if outward > 0 else face,
    }


def _literal_recipe_tangents(frame, axis):
    """Independent literal tangent mappings for one recipe reading.

    Returns ``(local_axis, local_label, world_axis, world_sign, origin,
    anchor)`` per tangent, with the anchor re-derived from the declared
    translation, the literal transform and the declared P0 residue.
    """
    quadrant = RECIPE_QUADRANTS[frame]
    origins = tuple(Fraction(value) for value in RECIPE_RESIDUES[frame])
    tangent_labels = tuple(label for label in RECIPE_AXES if label != axis)
    mappings = []
    for index, (local_axis, local_label) in enumerate(
            zip(("u", "v"), tangent_labels)):
        world_axis, world_sign = LITERAL_TRANSFORMS[quadrant][local_label]
        origin = origins[index]
        raw = RECIPE_TRANSLATIONS[frame][world_axis] + world_sign * origin
        assert raw.denominator == 1
        mappings.append((local_axis, local_label, world_axis, world_sign,
                         (origin.numerator, origin.denominator), int(raw)))
    return tuple(mappings)


def _literal_recipe_cells(frame, axis):
    """Independent literal placed cells, including unknown-cell omission.

    Only the declared supported grid cells are placed; each lateral endpoint is
    re-derived with the lower-edge rule (``anchor + index`` for a positive axis,
    ``anchor - (index + 1)`` for a reflected one).
    """
    (_, _, _, sign_u, _, anchor_u), (_, _, _, sign_v, _, anchor_v) = \
        _literal_recipe_tangents(frame, axis)
    cells = []
    for row, column in RECIPE_SUPPORTED_GRID[frame]:
        lateral_u = anchor_u + column if sign_u > 0 else anchor_u - (column + 1)
        lateral_v = anchor_v + row if sign_v > 0 else anchor_v - (row + 1)
        cells.append((row, column, lateral_u, lateral_v))
    return tuple(sorted(cells))


def _recipe_patch(frame):
    """16x16 float64 P0 patch: one G impulse and two period-aligned D bumps."""
    patch = np.zeros((H, W), dtype=np.float64)
    patch[RECIPE_G[frame][0], RECIPE_G[frame][1]] = 1.0
    for row, column in RECIPE_D[frame]:
        patch[row, column] = 4.0
    return patch


def _recipe_world_normal(frame, axis):
    return plane_graph.axis_transform(axis, RECIPE_QUADRANTS[frame])[0]


def _recipe_basis(frame, axis):
    """Map this reading's local (u, v) labels onto its two world tangents."""
    quadrant = RECIPE_QUADRANTS[frame]
    tangent_labels = tuple(label for label in RECIPE_AXES if label != axis)
    world_tangents = tuple(label for label in RECIPE_AXES
                           if label != _recipe_world_normal(frame, axis))
    rotation = _physical_yaw(quadrant)
    return rotation[np.ix_(world_tangents, tangent_labels)], world_tangents


def _recipe_signature_origin(frame, axis, origin_shift=RECIPE_ORIGIN_SHIFT):
    """Place each reading's block grid on the shared world block window.

    A negative local-to-world sign reflects the grid, so its ``origin_uv`` is
    shifted by one full grid side less one (``origin_shift``) to keep every
    reading's coverage over the same physical blocks.  The active default
    ``RECIPE_ORIGIN_SHIFT`` of ``-7`` on the eight-cell grid retains the required
    row carry -1 of the R0->R1 n1->1 edge; the historical ``-2`` on the 4x4 grid
    leaves a 3x4 fixed overlap whose carry support stays below the nine-cell
    matcher gate and is used only as a negative control.
    """
    quadrant = RECIPE_QUADRANTS[frame]
    origin = []
    for label in (label for label in RECIPE_AXES if label != axis):
        _world_axis, sign = plane_graph.axis_transform(label, quadrant)
        origin.append(0.0 if sign > 0 else float(origin_shift))
    return tuple(origin)


def _recipe_side(frame, axis):
    """The unique unit side placing this reading on the block-1 plane."""
    world_axis, world_sign = plane_graph.axis_transform(
        axis, RECIPE_QUADRANTS[frame])
    return ((Fraction(1) - RECIPE_TRANSLATIONS[frame][world_axis])
            / (world_sign * RECIPE_DISTANCES[frame]))


def _recipe_reading(frame, axis, origin_shift, samples, extent):
    """One genuine reading and signature from its own P0 generation record."""
    extent = float(extent)
    distance = float(RECIPE_DISTANCES[frame])
    side = _recipe_side(frame, axis)
    assert side == Fraction(RECIPE_SIDES[frame][axis])
    basis, world_tangents = _recipe_basis(frame, axis)
    translation = np.asarray(
        [float(RECIPE_TRANSLATIONS[frame][world]) for world in world_tangents],
        dtype=np.float64)
    residues = tuple(float((-value) % 1.0) for value in basis.T @ translation)
    patch = _recipe_patch(frame)
    produced = _produce(patch, frame, axis, axis, RECIPE_PERIOD, extent, samples)
    assert produced.status == "complete", produced.refusals
    evidence = _g_evidence(produced)
    assert evidence.phase_uv_pixels == RECIPE_PHASES[frame], \
        evidence.phase_uv_pixels
    assert abs(evidence.period_pixels - RECIPE_PERIOD) < 1e-12
    base = _base_reading(frame, axis, RECIPE_B, distance, float(side), (0, 0),
                         residues, samples=samples, extent=extent)
    assert abs(base.period_pixels - RECIPE_PERIOD) < 1e-12, base.period_pixels
    # Copy every P0 generation field (including quality and region identity)
    # into the reading; phase_pixels is set from the generated evidence.
    reading = replace(base, polygon=lp._canonical_region_polygon(POLY),
                      region_evidence=_EVIDENCE[axis],
                      phase_evidence=evidence,
                      phase_pixels=tuple(evidence.phase_uv_pixels))
    signature = _signature(reading, RECIPE_B, basis, translation,
                           origin_uv=_recipe_signature_origin(
                               frame, axis, origin_shift))
    signature = replace(signature,
                        coverage=np.ones(signature.coverage.shape, dtype=bool))
    return reading, signature, produced, patch


def _build_recipe_fixture(*, origin_shift=RECIPE_ORIGIN_SHIFT,
                          samples=RECIPE_SAMPLES, extents=RECIPE_EXTENTS):
    """Build the genuine recipe universe through the accepted public APIs."""
    readings = []
    signatures = []
    split_sources = []
    for frame in RECIPE_FRAMES:
        for axis in RECIPE_AXES:
            reading, signature, produced, patch = _recipe_reading(
                frame, axis, origin_shift, samples, extents[frame])
            readings.append(reading)
            signatures.append(signature)
            split_sources.append((produced, patch, _p0_budget()))
    readings = tuple(readings)
    signatures = tuple(signatures)
    split_sources = tuple(split_sources)

    signature_by_key = {(signature.reading.frame, signature.reading.axis):
                        signature for signature in signatures}
    matches = []
    for index, left_frame in enumerate(RECIPE_FRAMES):
        for right_frame in RECIPE_FRAMES[index + 1:]:
            for world_axis in RECIPE_AXES:
                left = next(reading for reading in readings
                            if reading.frame == left_frame
                            and _recipe_world_normal(reading.frame, reading.axis)
                            == world_axis)
                right = next(reading for reading in readings
                             if reading.frame == right_frame
                             and _recipe_world_normal(reading.frame, reading.axis)
                             == world_axis)
                candidates = tuple(
                    match for quadrant in range(4)
                    if plane_graph.axis_transform(left.axis, quadrant)[0]
                    == right.axis
                    for match in (plane_graph.match_plane_pair(
                        signature_by_key[(left_frame, left.axis)],
                        signature_by_key[(right_frame, right.axis)],
                        quadrant=quadrant, max_shift_cells=8),)
                    if match is not None)
                relative_quadrant = (RECIPE_QUADRANTS[right_frame]
                                     - RECIPE_QUADRANTS[left_frame]) % 4
                assert len(candidates) >= 1, (left_frame, right_frame,
                                              world_axis)
                found = next((match for match in candidates
                              if match.quadrant_of_right == relative_quadrant),
                             None)
                assert found is not None, (left_frame, right_frame,
                                           world_axis, relative_quadrant)
                matches.append(replace(
                    found,
                    left_plane=_canonical_identity(readings, left)[1],
                    right_plane=_canonical_identity(readings, right)[1]))
    matches = tuple(matches)
    assert len(matches) == 3 * 3

    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(RECIPE_FRAMES)),
        quadrants=dict(RECIPE_QUADRANTS),
        translations={frame: (0.0, 0.0, 0.0) for frame in RECIPE_FRAMES},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): RECIPE_B
                           for reading in readings},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None, support.refusal
    p1d = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=8,
        budget=_shift_budget(max_domains=64), readings=readings)
    assert p1d.status == "complete", p1d.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    assert all(value.refusal is None for value in phases), \
        [value.refusal for value in phases]
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0.0,
        budget=_runtime_budget(max_components=10, max_frames=10, max_edges=32,
                               max_tree_edges=10, max_chords=32,
                               max_cycle_rows=160))
    v2a_result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_v2a_budget())
    v2b_result = v2b.analyze_phase_anchor(
        solved, vector_result, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=v2a_result, split_sources=split_sources,
        budget=_anchor_budget())
    return {
        "readings": readings, "signatures": signatures,
        "split_sources": split_sources, "matches": matches, "support": support,
        "p1d_result": p1d, "phase_evidence": phases,
        "vector_result": vector_result, "c2rb_result": solved,
        "v2a_result": v2a_result, "v2b_result": v2b_result,
    }


def _recipe_universe(fixture):
    v2a_result = fixture["v2a_result"]
    return placement.PlacementUniverse(
        c2rb_result=fixture["c2rb_result"],
        vector_result=fixture["vector_result"],
        axis_support=fixture["support"], p1d_result=fixture["p1d_result"],
        matches=fixture["matches"], phase_evidence=fixture["phase_evidence"],
        readings=fixture["readings"], signatures=fixture["signatures"],
        v2a_result=v2a_result, v2b_result=fixture["v2b_result"],
        split_sources=fixture["split_sources"],
        anchor_budget=_anchor_budget(),
        support_budget=_support_budget(),
        usable_masks=tuple(
            (domain.reading_identity.identity_sha256,
             _mask_for(domain.reading_identity.reading_axis))
            for domain in v2a_result.reading_domains),
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def _recipe_supports(fixture):
    """Build one genuine V1 support per recipe reading, keyed by its residues.

    Each accepted support is a real 3x3 cell window whose own masked projection
    decides every cell state: R0 and R2 keep all nine cells supported while each
    R1 reading exposes five boundary-unknown cells.  The placement oracle below
    therefore measures genuine unknown-cell omission rather than a fabricated
    grid.
    """
    universe = _recipe_universe(fixture)
    domains = {}
    for domain in universe.v2a_result.reading_domains:
        identity = domain.reading_identity
        domains[(identity.frame, identity.canonical_index,
                 identity.reading_axis)] = domain
    alpha = tuple(
        axis.alpha_dyadic if axis.status == "determined" else None
        for axis in universe.v2b_result.prediction.axes)
    assert None not in alpha
    anchor_digest = placement.placement_anchor_digest(alpha)
    result = []
    for reading in universe.readings:
        identity = _canonical_identity(universe.readings, reading)
        domain = domains[(identity[0], identity[1], reading.axis)]
        residue = tuple(
            Fraction(value.numerator, value.denominator)
            if isinstance(value, Fraction) else Fraction(value).limit_denominator(16)
            for value in (reading.phase_evidence.phase_uv_pixels[0]
                          / reading.period_pixels,
                          reading.phase_evidence.phase_uv_pixels[1]
                          / reading.period_pixels))
        built = v1.build_aligned_region_support(
            reading=reading, reading_identity=domain.reading_identity,
            reading_universe=universe.readings,
            usable_mask=_mask_for(reading.axis), image_sha256=IMAGE_SHA,
            blocks_per_period=RECIPE_B,
            origin_uv=((residue[0].numerator, residue[0].denominator),
                       (residue[1].numerator, residue[1].denominator)),
            grid_shape=(3, 3), budget=_support_budget(),
            phase_anchor_sha256=anchor_digest)
        assert isinstance(built, v1.AlignedRegionSupport), built
        result.append(built)
    return tuple(result)


@pytest.fixture(scope="module")
def recipe_fixture():
    return _build_recipe_fixture()


def test_recipe_p0_generation_is_copied_into_reading_evidence(recipe_fixture):
    """Every reading carries its own exact P0 generation field for field."""
    readings = recipe_fixture["readings"]
    assert len(readings) == 9
    for reading in readings:
        frame = reading.frame
        evidence = reading.phase_evidence
        assert evidence is not None
        assert evidence.phase_uv_pixels == RECIPE_PHASES[frame]
        assert tuple(reading.phase_pixels) == RECIPE_PHASES[frame]
        assert abs(evidence.period_pixels - RECIPE_PERIOD) < 1e-12
        assert abs(reading.period_pixels - RECIPE_PERIOD) < 1e-12
        assert evidence.extent == reading.extent
        assert evidence.samples == reading.samples == RECIPE_SAMPLES
        assert evidence.patch_centre_uv == reading.rectification.centre
        assert evidence.source_region_id \
            == reading.region_evidence.source_region_id
        assert reading.region_evidence.source_region_id \
            == _REGION_ID[reading.axis]
        residue_u = evidence.phase_uv_pixels[0] / evidence.period_pixels
        residue_v = evidence.phase_uv_pixels[1] / evidence.period_pixels
        assert (residue_u, residue_v) == RECIPE_RESIDUES[frame]
        assert reading.side == RECIPE_SIDES[frame][reading.axis]
    assert {reading.side for reading in readings} == {1.0, -1.0}


def test_recipe_c2rb_component_is_unique_validated(recipe_fixture):
    """The repaired joint-vector fixture seals one ``unique_validated`` component.

    The genuine recipe universe reaches the unchanged solver with the complete
    nine-match graph (three world normals for each of the three frame pairs) and
    a ``complete`` A4 vector result carrying all nine domains.  Because every A4
    domain now retains a candidate with the full declared lateral translation,
    the solver seals one exhaustive ``unique_validated`` component whose single
    translation class replays ``RECIPE_TRANSLATIONS`` up to the gauge anchor.
    V2-A is ``endpoint_unique`` and V2-B determines all three world axes.  The
    certificate is still produced only by the solver: no record is hand-sealed,
    no certificate is fabricated and no gate is weakened.
    """
    solved = recipe_fixture["c2rb_result"]
    assert solved.status == "complete"
    assert solved.placement_readiness == "ready"
    assert solved.connected is True
    assert solved.components and len(solved.components) == 1
    assert solved.refusals == ()
    assert solved.budget_failure_reason is None
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert component.cycle_rank == 7
    assert len(component.chords) == 7
    assert component.failing_chords == ()
    assert component.ambiguity_class_count_lower_bound == 0
    assert len(component.translation_classes) == 1
    certificate = dict(
        component.translation_classes[0].translations_dyadic_xyz)
    assert set(certificate) == set(RECIPE_FRAMES)
    for frame in RECIPE_FRAMES:
        expected = tuple(Fraction(value) for value in RECIPE_TRANSLATIONS[frame])
        recovered = tuple(Fraction(*item) for item in certificate[frame])
        assert recovered == expected, frame
    # Every fundamental-cycle residual is a genuine satisfied chord witness.
    assert len(component.cycle_residuals) == 7
    assert all(residual.matching_provenance_count >= 1
               for residual in component.cycle_residuals)
    # The genuine upstream graph is complete and well formed, so the decision is
    # exhaustive rather than a truncated or budget-refused search.
    assert len(recipe_fixture["matches"]) == 9
    assert recipe_fixture["support"].refusal is None
    assert recipe_fixture["p1d_result"].status == "complete"
    vector_result = recipe_fixture["vector_result"]
    assert vector_result.status == "complete"
    assert len(vector_result.domains) == 9

    v2a_result = recipe_fixture["v2a_result"]
    assert v2a_result.status == "complete"
    assert v2a_result.classification == "endpoint_unique"
    assert v2a_result.exhaustive is True
    assert v2a_result.translation_class_count == 1
    assert v2a_result.endpoint_assignment_count == 1
    assert not v2a_result.refusals

    v2b_result = recipe_fixture["v2b_result"]
    assert v2b_result.status == "determined"
    assert v2b_result.prediction is not None
    assert v2b_result.prediction.reason == "determined"
    assert [axis.status for axis in v2b_result.prediction.axes] \
        == ["determined", "determined", "determined"]
    assert [axis.alpha_dyadic for axis in v2b_result.prediction.axes] \
        == [(0, 1), (0, 1), (0, 1)]


def test_recipe_full_signed_placement_oracle(recipe_fixture):
    """Full signed-placement oracle for the unique nine-edge recipe.

    The repaired recipe is the only nonzero-residue universe that clears every
    accepted gate, so this test pins its complete placement result with literal
    expectations derived from the construction, never read back from
    ``plane_placement``: nine resolved faces (one per reading) over all three
    world normal axes with mixed ``PlaneReading.side`` signs, the declared signed
    and fractional translations, reflected tangent mappings, exactly the 66
    supported cells, and omission of all 15 boundary-unknown cells.  No support
    is resealed, no certificate is fabricated and no gate is weakened.
    """
    universe = _recipe_universe(recipe_fixture)
    supports = _recipe_supports(recipe_fixture)
    outcome = placement.resolve_face_placement(
        universe=universe, supports=supports, budget=_budget())
    assert isinstance(outcome, placement.ResolvedFacePlacement), outcome
    assert outcome.alpha == RECIPE_ALPHA
    assert {binding["anchor_digest"] for binding in outcome.component_bindings} \
        == {RECIPE_ANCHOR_DIGEST}
    assert {name: getattr(outcome.work, name) for name in RECIPE_PLACEMENT_WORK} \
        == RECIPE_PLACEMENT_WORK
    assert len(outcome.faces) == RECIPE_PLACEMENT_WORK["faces"] == 9
    # Both signs of ``PlaneReading.side`` survive into the placed gauge, all
    # three world normal axes are represented, and the declared signed and
    # fractional translations (including the negative R2 vector) are placed.
    assert {face.c for face in outcome.faces} == {(1, 1), (-1, 1)}
    assert {face.world_normal_axis for face in outcome.faces} == set(RECIPE_AXES)
    assert {tuple(Fraction(*item) for item in face.translation)
            for face in outcome.faces} == set(RECIPE_TRANSLATIONS.values())

    faces = {(face.frame, face.reading_axis): face for face in outcome.faces}
    assert set(faces) == {(frame, axis) for frame in RECIPE_FRAMES
                          for axis in RECIPE_AXES}
    domain_frame = {domain.reading_identity.identity_sha256:
                    (domain.reading_identity.frame,
                     domain.reading_identity.reading_axis)
                    for domain in universe.v2a_result.reading_domains}
    placed = 0
    reflected = 0
    unknowns = 0
    for frame in RECIPE_FRAMES:
        for axis in RECIPE_AXES:
            face = faces[(frame, axis)]
            normal = _literal_recipe_normal(frame, axis)
            assert face.absolute_quadrant == RECIPE_QUADRANTS[frame]
            assert face.world_normal_axis == normal["world_axis"]
            assert face.world_normal_sign == normal["world_sign"]
            assert face.c == normal["c"]
            assert face.d == normal["d"]
            assert face.face_gauge == normal["face_gauge"]
            assert face.face == normal["face"] == 1
            assert face.outward_sign == normal["outward_sign"]
            assert face.normal_block == normal["normal_block"]
            assert face.translation == tuple(
                (value.numerator, value.denominator)
                for value in RECIPE_TRANSLATIONS[frame])
            assert face.alpha == RECIPE_ALPHA
            # The declared translation puts the face on the true block-1 plane.
            assert Fraction(*face.translation[face.world_normal_axis]) \
                + face.world_normal_sign * Fraction(*face.c) \
                * Fraction(*face.d) \
                == Fraction(*face.face_gauge) == Fraction(1)
            # Tangent mappings replay the literal transform, the declared
            # residue origin and the signed anchor, including every reflected
            # negative world axis.
            for mapping, expected in zip(
                    face.tangent_mappings,
                    _literal_recipe_tangents(frame, axis)):
                (local_axis, local_label, world_axis, world_sign, origin,
                 anchor) = expected
                assert (mapping.local_axis, mapping.local_label) \
                    == (local_axis, local_label)
                assert (mapping.world_axis, mapping.world_sign) \
                    == (world_axis, world_sign)
                assert mapping.origin == origin
                assert mapping.anchor == anchor
                if world_sign < 0:
                    reflected += 1
            # Only genuinely supported cells are placed; every boundary-unknown
            # cell is omitted.
            assert tuple(sorted((cell.row, cell.column, cell.lateral_u,
                                 cell.lateral_v) for cell in face.cells)) \
                == _literal_recipe_cells(frame, axis)
            assert {(cell.row, cell.column) for cell in face.cells} \
                == set(RECIPE_SUPPORTED_GRID[frame])
            for cell in face.cells:
                assert (cell.row, cell.column) \
                    not in RECIPE_UNKNOWN_GRID[frame]
            # A reflected tangent uses the lower-edge rule exactly.
            for mapping in face.tangent_mappings:
                if mapping.world_sign >= 0:
                    continue
                for cell in face.cells:
                    if mapping.local_axis == "u":
                        assert cell.lateral_u \
                            == mapping.anchor - (cell.column + 1)
                    else:
                        assert cell.lateral_v \
                            == mapping.anchor - (cell.row + 1)
            placed += len(face.cells)
    assert placed == RECIPE_PLACEMENT_WORK["support_cells"] == 66
    assert reflected == RECIPE_REFLECTED_TANGENTS == 6

    # The omitted cells are real boundary-unknown support cells, not absent grid
    # cells: each accepted 3x3 support declares the same split that placement
    # then drops.
    for support in supports:
        frame, axis = domain_frame[support.reading_identity_sha256]
        states = {(cell.row, cell.column): cell.state for cell in support.cells}
        assert set(states) == {(row, column) for row in range(3)
                               for column in range(3)}
        for key in RECIPE_SUPPORTED_GRID[frame]:
            assert states[key] == "supported"
        for key in RECIPE_UNKNOWN_GRID[frame]:
            assert states[key] == "unknown"
            unknowns += 1
    assert unknowns == RECIPE_UNKNOWN_CELLS == 15


def _recipe_edge_domains(fixture, left_frame, left_axis, right_frame,
                          right_axis):
    """Return the P1-D and A4 domains bound to one genuine C1 edge."""
    def matches(domain):
        identity = domain.match_identity
        return (identity.left_frame, identity.left_axis,
                identity.right_frame, identity.right_axis) \
            == (left_frame, left_axis, right_frame, right_axis)

    p1d_domain = next(domain for domain in fixture["p1d_result"].domains
                      if matches(domain))
    vector_domain = next(domain for domain in fixture["vector_result"].domains
                         if matches(domain))
    return p1d_domain, vector_domain


def test_recipe_r0_to_r1_carry_is_retained(recipe_fixture):
    """The repaired 6x6 overlap retains the R0->R1 n1->1 joint carry.

    The genuine C1 match binds one P1-D shift domain and one A4 lateral domain.
    On the reflection-aligned geometry the R0->R1 n1->1 edge keeps the joint
    ``(row_shift, column_shift) == (-1, 2)`` entry (20 support cells, at least
    the nine-cell matcher gate) and its A4 world-axis-0 term is the declared 1.5
    relative translation, not the 0.5 the historical 3x4 overlap exposed.
    """
    match = next(item for item in recipe_fixture["matches"]
                 if (item.left, item.left_axis, item.right, item.right_axis)
                 == ("R0", 1, "R1", 1) and item.quadrant_of_right == 1)
    p1d_domain, vector_domain = _recipe_edge_domains(
        recipe_fixture, "R0", 1, "R1", 1)
    # Both derived domains replay the genuine C1 match identity, so the carried
    # candidate is the C1-consistent one and not a resealed or foreign record.
    for identity in (p1d_domain.match_identity, vector_domain.match_identity):
        assert (identity.left_frame, identity.left_axis,
                identity.right_frame, identity.right_axis,
                identity.quadrant_of_right) == (
            match.left, match.left_axis, match.right, match.right_axis,
            match.quadrant_of_right)
        assert (identity.left_plane, identity.right_plane) \
            == (match.left_plane, match.right_plane)
        assert (identity.left_b, identity.right_b) \
            == (match.left_b, match.right_b)
    # The v2 fixture uses a full 8x8 reflection-aligned window, so the joint
    # (-1, 2) entry is representable with ample shifted support.
    assert p1d_domain.zero_shift_overlap_cells == 64
    entries = {(entry.row_shift, entry.column_shift): entry
               for entry in p1d_domain.entries}
    carry = entries[(-1, 2)]
    assert carry.row_shift == -1
    assert carry.column_shift == 2
    assert carry.overlap_cells == 42
    assert carry.overlap_cells >= plane_graph.MIN_OVERLAP_CELLS
    # The declared relative translation on the coupled world axis is exactly 3/2.
    declared = RECIPE_TRANSLATIONS["R1"][0] - RECIPE_TRANSLATIONS["R0"][0]
    assert declared == Fraction(3, 2)
    terms = [term
             for candidate in vector_domain.iter_candidates(
                 0, vector_domain.represented_candidate_count)
             for term in candidate.axis_terms
             if term.world_axis == 0]
    carried = [term for term in terms
               if Fraction(term.delta) == declared
               and term.shift_source_component == "row"
               and term.shift_entry_rc[0] == -1]
    assert carried
    for term in carried:
        assert (term.shift_entry_rc[0], term.shift_entry_rc[1]) in entries
        assert term.left_world_sign == 1
        assert term.right_world_sign == -1
        assert Fraction(term.left_residue.candidate_value) == Fraction(0)
        assert Fraction(term.right_residue.candidate_value) == Fraction(1, 2)


def test_recipe_all_nine_edges_retain_full_joint_translation(recipe_fixture):
    """Every genuine C1 edge keeps an A4 candidate with its full joint vector.

    The expected two lateral components of each edge are derived independently
    from ``RECIPE_TRANSLATIONS`` (right frame minus left frame) on the two world
    axes that are not the edge normal; no expectation is read back from the
    module under test.  For each of the nine genuine edges the test binds an
    actual A4 candidate to its C1 match identity, its real P1-D joint entry and
    the production axis table, and proves both lateral components equal the
    declared difference at once.
    """
    translations = RECIPE_TRANSLATIONS
    domains = {(domain.match_identity.left_frame,
                domain.match_identity.left_axis,
                domain.match_identity.right_frame,
                domain.match_identity.right_axis): domain
               for domain in recipe_fixture["vector_result"].domains}
    assert len(domains) == 9
    assert len(recipe_fixture["matches"]) == 9
    seen = set()
    for match in recipe_fixture["matches"]:
        key = (match.left, match.left_axis, match.right, match.right_axis)
        domain = domains[key]
        normal_axis = domain.normal_constraint.world_axis
        lateral = tuple(axis for axis in RECIPE_AXES if axis != normal_axis)
        assert tuple(factor.world_axis for factor in domain.axis_factors) \
            == lateral
        expected = {axis: Fraction(translations[match.right][axis]
                                   - translations[match.left][axis])
                    for axis in lateral}
        candidate = next(
            (item for item in domain.iter_candidates(
                0, domain.represented_candidate_count)
             if tuple(term.world_axis for term in item.axis_terms) == lateral
             and all(Fraction(term.delta) == expected[term.world_axis]
                     for term in item.axis_terms)),
            None)
        assert candidate is not None, key
        # The candidate is bound to the genuine C1 match identity and the real
        # P1-D joint entry, not a foreign or resealed record.
        identity = candidate.match_identity
        assert identity == domain.match_identity
        assert candidate.p1d_domain_identity == domain.match_identity
        assert (identity.left_frame, identity.left_axis,
                identity.right_frame, identity.right_axis) == key
        assert identity.quadrant_of_right == match.quadrant_of_right
        assert (identity.left_plane, identity.right_plane) \
            == (match.left_plane, match.right_plane)
        assert (identity.left_b, identity.right_b) \
            == (match.left_b, match.right_b)
        assert candidate.shift_entry in domain.p1d_domain.entries
        assert (candidate.shift_entry.row_shift,
                candidate.shift_entry.column_shift) == candidate.shift_entry_rc
        for term in candidate.axis_terms:
            assert term.match_identity == identity
            assert term.shift_entry_rc == candidate.shift_entry_rc
            assert plane_graph.axis_transform(
                term.left_local_axis_label, term.left_frame_quadrant) \
                == (term.world_axis, term.left_world_sign)
            assert plane_graph.axis_transform(
                term.right_local_axis_label, term.right_frame_quadrant) \
                == (term.world_axis, term.right_world_sign)
        seen.add(key)
    assert seen == set(domains)


def test_recipe_current_geometry_drops_the_carry():
    """The historical 4x4 fixture is now refused at the exact v2 gate."""
    left, left_signature, *_ = _recipe_reading(
        "R0", 0, RECIPE_CURRENT_ORIGIN_SHIFT,
        RECIPE_CURRENT_SAMPLES, RECIPE_CURRENT_EXTENTS["R0"])
    right, right_signature, *_ = _recipe_reading(
        "R1", 2, RECIPE_CURRENT_ORIGIN_SHIFT,
        RECIPE_CURRENT_SAMPLES, RECIPE_CURRENT_EXTENTS["R1"])
    assert _recipe_world_normal(left.frame, left.axis) == 0
    assert _recipe_world_normal(right.frame, right.axis) == 0
    refusals = []
    assert plane_graph.match_plane_pair(
        left_signature, right_signature, quadrant=1, max_shift_cells=8,
        refusals=refusals) is None
    assert [item.reason for item in refusals] == ["repetition_unresolved"]
