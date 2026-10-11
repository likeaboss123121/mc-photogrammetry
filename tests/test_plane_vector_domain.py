"""Indexed, bounded, all-or-nothing C2-R-A4 vector-domain tests."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tracemalloc

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph, plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d
from mc_photogrammetry.reconstruction import plane_vector_domain as c2r

from test_plane_shift_domain import (
    _budget as _p1d_budget,
    _build_case,
    _reading_universe,
    _two_identity_case,
)


def _canonical_readings(signatures):
    return _reading_universe(tuple(signatures))


def _phase_for(match, readings):
    ordered = {}
    for reading in readings:
        ordered.setdefault(reading.frame, []).append(reading)
    by_index = {}
    for frame, values in ordered.items():
        values.sort(key=lattice_planes.reading_order_key)
        by_index.update({(frame, index): reading
                         for index, reading in enumerate(values)})
    return plane_residue.hypothesis_phase_shift_evidence(match, by_index)


def _large_budget():
    return c2r.C2RVectorBudget(
        max_requested_match_identities=100,
        max_joint_entries=1_000_000,
        max_axis_residue_pairs=10_000_000,
        max_vector_candidates=10_000_000,
        max_provenance_references=100_000_000,
    )


def _build_vector_case():
    p1d_result, signatures, match, support = _build_case(radius=8)
    readings = _canonical_readings(signatures)
    phase = _phase_for(match, readings)
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, (match,), (phase,), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    return result, p1d_result, signatures, (match,), support, (phase,), readings


def test_v2_result_and_sources_bind_algorithm_radius_and_infinite_psr():
    result, p1d_result, _signatures, matches, _support, phases, _readings = \
        _build_vector_case()
    assert result.schema == c2r.C2R_VECTOR_DOMAIN_SCHEMA
    assert result.version == c2r.C2R_VECTOR_DOMAIN_VERSION == 2
    assert result.correlation_algorithm_id == plane_graph.PLANE_CORRELATION_ALGORITHM
    domain_source = c2r._p1d_domain_source(p1d_result.domains[0])
    assert domain_source["schema"] == p1d.SHIFT_DOMAIN_SCHEMA
    assert domain_source["version"] == p1d.SHIFT_DOMAIN_VERSION
    assert domain_source["correlation_algorithm_id"] == \
        plane_graph.PLANE_CORRELATION_ALGORITHM
    assert domain_source["max_shift_cells"] == p1d_result.domains[0].max_shift_cells
    infinite = replace(matches[0], psr=float("inf"))
    assert c2r._match_record(infinite)["psr"] == {
        "value": None, "positive_infinity": True}
    assert phases[0].relative_shift.schema == \
        plane_residue.RELATIVE_SHIFT_SUPPORT_SCHEMA


@pytest.mark.parametrize("field,value", (
    ("schema", "mc-photogrammetry-c2r-vector-domain"),
    ("version", 1),
    ("correlation_algorithm_id", "fixed-mask-v1"),
))
def test_v1_and_mixed_result_provenance_is_rejected(field, value):
    result, *_rest = _build_vector_case()
    with pytest.raises(ValueError, match="provenance"):
        replace(result, **{field: value})


def _multi_residue_case():
    """A small physical fixture with two residues in each endpoint table."""
    from test_c2r_vector_domain_a3_proof import (
        _canonical_identity, _reading, _signature,
    )

    left_translation = np.asarray((0.2, -0.35), dtype=np.float64)
    right_translation = np.asarray((1.6, 0.45), dtype=np.float64)
    basis = np.eye(2, dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0)
                          for value in basis.T @ left_translation)
    right_residues = tuple(float((-value) % 1.0)
                           for value in basis.T @ right_translation)
    left = _reading("a_left", 2, 0.5, 2.0, 1.0, (3, -2), left_residues)
    right = _reading("b_right", 2, 0.5, 3.0, 1.0, (-4, 5), right_residues)
    signatures = (
        _signature(left, 0.5, basis, left_translation),
        _signature(right, 0.5, basis, right_translation),
    )
    readings = (left, right)
    matches = plane_graph.match_plane_graph(readings, signatures, max_shift_cells=4)
    match = next(value for value in matches if value.quadrant_of_right == 0)
    left_identity = _canonical_identity(readings, left)
    right_identity = _canonical_identity(readings, right)
    solution = plane_graph.ComponentSolution(
        frames=("a_left", "b_right"),
        quadrants={"a_left": 0, "b_right": 0},
        translations={"a_left": (0.0, 0.0, 0.0),
                      "b_right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={left_identity: 0.5, right_identity: 0.5},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, (match,), signatures)
    assert support.refusal is None
    p1d_result = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=4,
        budget=_p1d_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phase = _phase_for(match, readings)
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, (match,), (phase,), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert result.status == "complete", result.refusals
    return result, p1d_result, signatures, (match,), support, (phase,), readings


def _million_candidate_case():
    """Factorized rational-b case with over one million represented vectors."""
    from test_c2r_vector_domain_a3_proof import (
        _canonical_identity, _forward_physical_carry, _reading, _signature,
    )

    blocks = 1 / 8
    left_translation = np.asarray((0.2, -0.35), dtype=np.float64)
    right_translation = np.asarray((1.6, 0.45), dtype=np.float64)
    basis = np.eye(2, dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0)
                          for value in basis.T @ left_translation)
    right_residues = tuple(float((-value) % 1.0)
                           for value in basis.T @ right_translation)
    left_origin, right_origin = (3, -2), (-4, 5)
    left = _reading("a_left", 2, blocks, 2.0, 1.0, left_origin,
                    left_residues, samples=2048)
    right = _reading("b_right", 2, blocks, 3.0, 1.0, right_origin,
                     right_residues, samples=2048)
    signatures = (
        _signature(left, blocks, basis, left_translation),
        _signature(right, blocks, basis, right_translation),
    )
    readings = (left, right)
    matches = plane_graph.match_plane_graph(readings, signatures, max_shift_cells=8)
    match = next(value for value in matches if value.quadrant_of_right == 0)
    left_identity = _canonical_identity(readings, left)
    right_identity = _canonical_identity(readings, right)
    solution = plane_graph.ComponentSolution(
        frames=("a_left", "b_right"),
        quadrants={"a_left": 0, "b_right": 0},
        translations={"a_left": (0.0, 0.0, 0.0),
                      "b_right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={left_identity: blocks, right_identity: blocks},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, (match,), signatures)
    assert support.refusal is None
    p1d_result = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=8,
        budget=_p1d_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phase = _phase_for(match, readings)
    forward = _forward_physical_carry(
        basis, basis, left_translation, right_translation,
        left_origin, right_origin, left_residues, right_residues)
    return (support, p1d_result, signatures, match, readings, phase,
            left_residues, right_residues, forward)


def _a3_scale_case():
    """Two complete rational-b domains representing more than the A3 census."""
    from test_c2r_vector_domain_a3_proof import (
        _canonical_identity, _forward_physical_carry, _reading, _signature,
    )

    blocks = 1 / 8
    left_translation = np.asarray((0.2, -0.35), dtype=np.float64)
    right_translation = np.asarray((1.6, 0.45), dtype=np.float64)
    basis = np.eye(2, dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0)
                          for value in basis.T @ left_translation)
    right_residues = tuple(float((-value) % 1.0)
                           for value in basis.T @ right_translation)
    left_origin, right_origin = (3, -2), (-4, 5)
    readings, signatures, endpoint_pairs = [], [], []
    for extent in (4.0, 5.0):
        left = _reading("a_left", 2, blocks, 2.0, 1.0, left_origin,
                        left_residues, samples=2048, extent=extent)
        right = _reading("b_right", 2, blocks, 3.0, 1.0, right_origin,
                         right_residues, samples=2048, extent=extent)
        readings.extend((left, right))
        signatures.extend((
            _signature(left, blocks, basis, left_translation),
            _signature(right, blocks, basis, right_translation),
        ))
        endpoint_pairs.append((left, right))
    readings, signatures = tuple(readings), tuple(signatures)
    graph_matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=8)
    selected, blocks_by_identity, forward = [], {}, []
    for left, right in endpoint_pairs:
        left_identity = _canonical_identity(readings, left)
        right_identity = _canonical_identity(readings, right)
        match = next(value for value in graph_matches
                     if (value.left_plane, value.right_plane,
                         value.quadrant_of_right)
                     == (left_identity[1], right_identity[1], 0))
        selected.append(match)
        blocks_by_identity[left_identity] = blocks
        blocks_by_identity[right_identity] = blocks
        forward.append(_forward_physical_carry(
            basis, basis, left_translation, right_translation,
            left_origin, right_origin, left_residues, right_residues))
    selected = tuple(selected)
    solution = plane_graph.ComponentSolution(
        frames=("a_left", "b_right"),
        quadrants={"a_left": 0, "b_right": 0},
        translations={"a_left": (0.0, 0.0, 0.0),
                      "b_right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True, blocks_per_period=blocks_by_identity,
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, selected, signatures)
    assert support.refusal is None
    p1d_result = p1d.build_shift_domains(
        support, selected, signatures, max_shift_cells=8,
        budget=_p1d_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in selected)
    return (support, p1d_result, signatures, selected, readings, phases,
            left_residues, right_residues, tuple(forward))


def _a3_census_parameters():
    from test_c2r_vector_domain_a3_proof import _physical_yaw

    for normal in range(3):
        for left_quadrant in range(4):
            for relative_quadrant in range(4):
                case_number = normal * 16 + left_quadrant * 4 + relative_quadrant
                left_side = 1.0 if case_number % 2 == 0 else -1.0
                right_side = left_side if case_number % 3 == 0 else -left_side
                right_quadrant = (left_quadrant + relative_quadrant) % 4
                left_rotation = _physical_yaw(left_quadrant)
                right_rotation = _physical_yaw(right_quadrant)
                right_normal = int(np.flatnonzero(
                    np.abs(_physical_yaw(relative_quadrant)[:, normal]) > 0.5)[0])
                common_normal = int(np.flatnonzero(
                    np.abs(left_rotation[:, normal]) > 0.5)[0])
                world_tangents = tuple(axis for axis in range(3)
                                       if axis != common_normal)
                left_labels = tuple(axis for axis in range(3) if axis != normal)
                right_labels = tuple(axis for axis in range(3)
                                     if axis != right_normal)
                basis_l = left_rotation[np.ix_(world_tangents, left_labels)]
                basis_r = right_rotation[np.ix_(world_tangents, right_labels)]
                left_origin = np.asarray((7, -5), dtype=np.float64)
                right_origin = tuple(int(value) for value in np.rint(
                    basis_r.T @ (basis_l @ (left_origin + 7.5)) - 7.5))
                yield {
                    "normal": normal, "left_quadrant": left_quadrant,
                    "relative_quadrant": relative_quadrant,
                    "left_b": 1.0, "right_b": 1.0,
                    "left_side": left_side, "right_side": right_side,
                    "left_distance": 2.0, "right_distance": 2.0,
                    "left_origin": tuple(int(value) for value in left_origin),
                    "right_origin": right_origin,
                    "left_samples": 128, "right_samples": 128, "radius": 4,
                }
    from test_c2r_vector_domain_a3_proof import _SUPPORTED_BLOCK_PERIODS

    for blocks in _SUPPORTED_BLOCK_PERIODS:
        yield {
            "normal": 2, "left_quadrant": 0, "relative_quadrant": 0,
            "left_b": blocks, "right_b": blocks,
            "left_side": 1.0, "right_side": -1.0,
            "left_distance": 2.0, "right_distance": 2.0,
            "left_origin": (7, -5), "right_origin": (7, -5),
            "left_samples": 2048, "right_samples": 2048, "radius": 2,
        }
    for pair_index, left_b in enumerate(_SUPPORTED_BLOCK_PERIODS):
        right_b = _SUPPORTED_BLOCK_PERIODS[(pair_index + 1) % len(_SUPPORTED_BLOCK_PERIODS)]
        yield {
            "normal": 2, "left_quadrant": 0, "relative_quadrant": 0,
            "left_b": left_b, "right_b": right_b,
            "left_side": 1.0 if pair_index % 2 == 0 else -1.0,
            "right_side": -1.0 if pair_index % 2 == 0 else 1.0,
            "left_distance": 2.0, "right_distance": 2.0,
            "left_origin": (7, -5), "right_origin": (7, -5),
            "left_samples": 2048, "right_samples": 2048, "radius": 2,
        }


def _build_a3_source_case(parameters):
    from test_c2r_vector_domain_a3_proof import (
        _canonical_identity, _forward_physical_carry, _physical_yaw,
        _reading, _signature,
    )

    normal = parameters["normal"]
    left_quadrant = parameters["left_quadrant"]
    relative_quadrant = parameters["relative_quadrant"]
    right_quadrant = (left_quadrant + relative_quadrant) % 4
    left_rotation, right_rotation = (_physical_yaw(left_quadrant),
                                     _physical_yaw(right_quadrant))
    right_normal = int(np.flatnonzero(
        np.abs(_physical_yaw(relative_quadrant)[:, normal]) > 0.5)[0])
    world_normal = int(np.flatnonzero(np.abs(left_rotation[:, normal]) > 0.5)[0])
    world_tangents = tuple(axis for axis in range(3) if axis != world_normal)
    left_labels = tuple(axis for axis in range(3) if axis != normal)
    right_labels = tuple(axis for axis in range(3) if axis != right_normal)
    basis_l = left_rotation[np.ix_(world_tangents, left_labels)]
    basis_r = right_rotation[np.ix_(world_tangents, right_labels)]
    left_translation = np.asarray((0.2, -0.35), dtype=np.float64)
    right_translation = np.asarray((1.6, 0.45), dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0)
                          for value in basis_l.T @ left_translation)
    right_residues = tuple(float((-value) % 1.0)
                           for value in basis_r.T @ right_translation)
    left_frame, right_frame = "a_left", "b_right"
    left_config = {"samples": parameters["left_samples"]}
    right_config = {"samples": parameters["right_samples"]}
    left = _reading(left_frame, normal, parameters["left_b"],
                    parameters["left_distance"], parameters["left_side"],
                    parameters["left_origin"], left_residues, **left_config)
    right = _reading(right_frame, right_normal, parameters["right_b"],
                     parameters["right_distance"], parameters["right_side"],
                     parameters["right_origin"], right_residues, **right_config)
    dummy = _reading(
        left_frame, normal, parameters["left_b"], parameters["left_distance"] * 2.0,
        parameters["left_side"], (0, 0), (0.21, 0.37), **left_config)
    signatures = (
        _signature(left, parameters["left_b"], basis_l, left_translation),
        _signature(right, parameters["right_b"], basis_r, right_translation),
    )
    readings = (right, dummy, left)
    match = next(value for value in plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=parameters["radius"])
        if value.quadrant_of_right == relative_quadrant)
    left_identity = _canonical_identity(readings, left)
    right_identity = _canonical_identity(readings, right)
    assert (match.left_plane, match.right_plane) == (left_identity[1], right_identity[1])
    solution = plane_graph.ComponentSolution(
        frames=(left_frame, right_frame),
        quadrants={left_frame: left_quadrant, right_frame: right_quadrant},
        translations={left_frame: (0.0, 0.0, 0.0),
                      right_frame: (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={left_identity: parameters["left_b"],
                           right_identity: parameters["right_b"]},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, (match,), signatures)
    assert support.refusal is None
    p1d_result = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=parameters["radius"],
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000, max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phase = _phase_for(match, readings)
    forward = _forward_physical_carry(
        basis_l, basis_r, left_translation, right_translation,
        parameters["left_origin"], parameters["right_origin"],
        left_residues, right_residues)
    return (support, p1d_result, signatures, match, readings, phase,
            basis_l, basis_r, left_residues, right_residues,
            left_translation, right_translation, forward)


def _two_domain_vector_case():
    support, matches, signatures = _two_identity_case()
    readings = _canonical_readings(signatures)
    p1d_result = p1d.build_shift_domains(
        support, matches, signatures, max_shift_cells=8,
        budget=_p1d_budget(), readings=readings)
    assert p1d_result.status == "complete"
    phases = tuple(_phase_for(match, readings) for match in matches)
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert result.status == "complete", result.refusals
    return result, p1d_result, signatures, matches, support, phases, readings


def _materialization_budget(result, *, candidate_delta=0, domain_delta=0):
    return c2r.C2RMaterializationBudget(
        max_materialized_candidates=(result.planned_work.vector_candidates
                                     + candidate_delta),
        max_materialized_domains=(result.planned_work.requested_match_identities
                                  + domain_delta),
    )


def _oracle_digest(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eager_candidate_record_oracle(domain, entry, indices):
    """Test-local eager record and digest builder, independent of candidate_at."""
    term_records = []
    vector = []
    provenance_axes = []
    for factor, (left_index, right_index) in zip(domain.axis_factors, indices):
        component = (entry.column_shift
                     if factor.right_local_axis == "u" else entry.row_shift)
        left_value = factor.left_side.candidate_values[left_index]
        right_value = factor.right_side.candidate_values[right_index]
        delta = (factor.right_world_sign * component
                 + factor.left_world_sign * left_value
                 - factor.right_world_sign * right_value)
        shift_rc = [entry.row_shift, entry.column_shift]
        left_residue = {
            "frame": factor.left_side.frame,
            "plane_index": factor.left_side.plane_index,
            "reading_axis": factor.left_side.reading_axis,
            "local_axis": factor.left_side.local_axis,
            "blocks_per_period": float(factor.left_side.blocks_per_period),
            "source_region_id": factor.left_side.source_region_id,
            "record_sha256": factor.left_side.record_sha256,
            "candidate_index": left_index,
            "candidate_value": float(left_value),
        }
        right_residue = {
            "frame": factor.right_side.frame,
            "plane_index": factor.right_side.plane_index,
            "reading_axis": factor.right_side.reading_axis,
            "local_axis": factor.right_side.local_axis,
            "blocks_per_period": float(factor.right_side.blocks_per_period),
            "source_region_id": factor.right_side.source_region_id,
            "record_sha256": factor.right_side.record_sha256,
            "candidate_index": right_index,
            "candidate_value": float(right_value),
        }
        term = {
            "formula_version": "c2r-lateral-vector-v1",
            "match_identity": domain.match_identity.to_record(),
            "world_axis": factor.world_axis,
            "left_world_sign": factor.left_world_sign,
            "right_world_sign": factor.right_world_sign,
            "left_frame_quadrant": factor.left_frame_quadrant,
            "right_frame_quadrant": factor.right_frame_quadrant,
            "left_local_axis": factor.left_local_axis,
            "left_local_axis_label": factor.left_local_axis_label,
            "right_local_axis": factor.right_local_axis,
            "right_local_axis_label": factor.right_local_axis_label,
            "shift_entry_rc": shift_rc,
            "shift_source_component": "column" if factor.right_local_axis == "u" else "row",
            "right_grid_integer_component": component,
            "signed_shift_component": factor.right_world_sign * component,
            "left_residue": left_residue,
            "right_residue": right_residue,
            "delta": float(delta),
        }
        term["formula_replay_sha256"] = _oracle_digest(term)
        term_records.append(term)
        vector.append(float(delta))
        provenance_axes.append([
            factor.world_axis, factor.left_side.record_sha256, left_index,
            factor.right_side.record_sha256, right_index,
        ])
    candidate = {
        "match_identity": domain.match_identity.to_record(),
        "p1d_domain_identity": domain.p1d_domain.match_identity.to_record(),
        "p1d_replay_sha256": domain.p1d_domain.replay_sha256,
        "p1d_max_shift_cells": domain.p1d_domain.max_shift_cells,
        "p1d_left_signature_content_sha256":
            domain.p1d_domain.left_signature_content_sha256,
        "p1d_right_signature_content_sha256":
            domain.p1d_domain.right_signature_content_sha256,
        "shift_entry": entry.to_record(),
        "c1_axis_source_sha256": [factor.c1_axis_source_sha256
                                  for factor in domain.axis_factors],
        "left_signature_geometry_sha256": domain.left_signature_geometry_sha256,
        "right_signature_geometry_sha256": domain.right_signature_geometry_sha256,
        "cell_homography_sha256": domain.cell_homography_sha256,
        "axis_terms": term_records,
    }
    candidate["candidate_sha256"] = _oracle_digest(candidate)
    candidate["shift_entry_rc"] = [entry.row_shift, entry.column_shift]
    candidate["vector"] = vector
    candidate["provenance_key"] = [[entry.row_shift, entry.column_shift],
                                   provenance_axes]
    return candidate


def _current_working_set_bytes():
    """Current process RSS on Windows; None where the OS API is unavailable."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def test_index_only_build_serialization_and_validation_construct_no_candidates(monkeypatch):
    _baseline, p1d_result, signatures, matches, support, phases, readings = _build_vector_case()

    def forbidden_candidate(*_args, **_kwargs):
        raise AssertionError("normal indexed paths constructed a candidate")

    monkeypatch.setattr(c2r, "_candidate_at_index", forbidden_candidate)
    tracemalloc.start()
    try:
        result = c2r.build_lateral_vector_domains(
            support, p1d_result, matches, phases, readings=readings,
            signatures=signatures, comparison_tolerance=1e-12,
            budget=_large_budget())
        record_bytes = result.to_json_bytes()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.status == "complete", result.refusals
    assert result.planned_work.vector_candidates == 289
    assert result.represented_work == result.planned_work
    assert len(record_bytes) <= 256 * 1024
    assert peak <= 4 * 1024 * 1024
    storage = result.to_record()["storage"]
    assert storage == {
        "stored_p1d_entries": 289,
        "stored_residue_values": 4,
        "stored_factor_records": 6,
        "candidates_materialized_during_construction": 0,
    }
    assert c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12)


def test_million_vector_index_build_stays_factorized_and_truth_is_directly_addressable():
    (support, p1d_result, signatures, match, readings, phase,
     left_residues, right_residues, forward) = _million_candidate_case()
    tracemalloc.start()
    try:
        result = c2r.build_lateral_vector_domains(
            support, p1d_result, (match,), (phase,), readings=readings,
            signatures=signatures, comparison_tolerance=1e-12,
            budget=_large_budget())
        payload = result.to_json_bytes()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.status == "complete", result.refusals
    assert result.planned_work.vector_candidates == 1_183_744
    assert len(payload) <= 8 * 1024 * 1024
    assert peak <= 32 * 1024 * 1024
    assert result.to_record()["storage"][
        "candidates_materialized_during_construction"] == 0

    domain = result.domains[0]
    entry_rc = forward["entry_rc"]
    entry_index = next(index for index, entry in enumerate(domain.p1d_domain.entries)
                       if (entry.row_shift, entry.column_shift) == entry_rc)
    local_values = {"u": (left_residues[0], right_residues[0]),
                    "v": (left_residues[1], right_residues[1])}
    selected_indices = []
    for factor in domain.axis_factors:
        left_target, right_target = local_values[factor.left_local_axis][0], \
            local_values[factor.right_local_axis][1]
        left_index = min(range(factor.left_side.candidate_count), key=lambda index:
                         abs(factor.left_side.candidate_values[index] - left_target))
        right_index = min(range(factor.right_side.candidate_count), key=lambda index:
                          abs(factor.right_side.candidate_values[index] - right_target))
        assert abs(factor.left_side.candidate_values[left_index] - left_target) < 1e-12
        assert abs(factor.right_side.candidate_values[right_index] - right_target) < 1e-12
        selected_indices.append((left_index, right_index))
    factor0, factor1 = domain.axis_factors
    q0 = selected_indices[0][0] * factor0.right_side.candidate_count + selected_indices[0][1]
    q1 = selected_indices[1][0] * factor1.right_side.candidate_count + selected_indices[1][1]
    p1_count = factor1.left_side.candidate_count * factor1.right_side.candidate_count
    vectors_per_entry = (factor0.left_side.candidate_count
                         * factor0.right_side.candidate_count * p1_count)
    index = entry_index * vectors_per_entry + q0 * p1_count + q1
    candidate = domain.candidate_at(index)
    expected_truth = (1.4, 0.8)
    assert candidate.shift_entry_rc == entry_rc
    assert candidate.vector == pytest.approx(expected_truth, abs=1e-12)


def test_a3_scale_census_over_two_million_vectors_remains_index_only():
    (support, p1d_result, signatures, matches, readings, phases,
     left_residues, right_residues, forwards) = _a3_scale_case()
    tracemalloc.start()
    try:
        result = c2r.build_lateral_vector_domains(
            support, p1d_result, matches, phases, readings=readings,
            signatures=signatures, comparison_tolerance=1e-12,
            budget=_large_budget())
        encoded = result.to_json_bytes()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.status == "complete", result.refusals
    assert result.planned_work.vector_candidates == 2_367_488
    assert result.planned_work.vector_candidates > 2_195_063
    assert len(encoded) <= 8 * 1024 * 1024
    assert peak <= 32 * 1024 * 1024
    assert result.to_record()["storage"] == {
            "stored_p1d_entries": 578,
        "stored_residue_values": 64,
        "stored_factor_records": 12,
        "candidates_materialized_during_construction": 0,
    }

    for domain, forward in zip(result.domains, forwards):
        entry_rc = forward["entry_rc"]
        entry_index = next(index for index, entry in enumerate(domain.p1d_domain.entries)
                           if (entry.row_shift, entry.column_shift) == entry_rc)
        local_values = {"u": (left_residues[0], right_residues[0]),
                        "v": (left_residues[1], right_residues[1])}
        selected_indices = []
        for factor in domain.axis_factors:
            left_target = local_values[factor.left_local_axis][0]
            right_target = local_values[factor.right_local_axis][1]
            left_index = min(range(factor.left_side.candidate_count), key=lambda index:
                             abs(factor.left_side.candidate_values[index] - left_target))
            right_index = min(range(factor.right_side.candidate_count), key=lambda index:
                              abs(factor.right_side.candidate_values[index] - right_target))
            assert abs(factor.left_side.candidate_values[left_index] - left_target) < 1e-12
            assert abs(factor.right_side.candidate_values[right_index] - right_target) < 1e-12
            selected_indices.append((left_index, right_index))
        factor0, factor1 = domain.axis_factors
        q0 = selected_indices[0][0] * factor0.right_side.candidate_count + selected_indices[0][1]
        q1 = selected_indices[1][0] * factor1.right_side.candidate_count + selected_indices[1][1]
        p1_count = factor1.left_side.candidate_count * factor1.right_side.candidate_count
        vectors_per_entry = (factor0.left_side.candidate_count
                             * factor0.right_side.candidate_count * p1_count)
        index = entry_index * vectors_per_entry + q0 * p1_count + q1
        candidate = domain.candidate_at(index)
        assert candidate.shift_entry_rc == entry_rc
        assert candidate.vector == pytest.approx((1.4, 0.8), abs=1e-12)


def test_exact_68_case_a3_census_is_indexed_and_every_truth_is_addressable():
    total_vectors = 0
    strata = {"orientation": 0, "equal_b": 0, "unequal_b": 0}
    serialized_bytes = 0
    max_traced_peak = 0
    rss_deltas = []

    for case_index, parameters in enumerate(_a3_census_parameters()):
        sources = _build_a3_source_case(parameters)
        (support, p1d_result, signatures, match, readings, phase,
         _basis_l, _basis_r, left_residues, right_residues,
         left_translation, right_translation, forward) = sources
        if case_index < 48:
            stratum = "orientation"
        elif case_index < 58:
            stratum = "equal_b"
        else:
            stratum = "unequal_b"
        gc.collect()
        rss_before = _current_working_set_bytes()
        tracemalloc.start()
        try:
            result = c2r.build_lateral_vector_domains(
                support, p1d_result, (match,), (phase,), readings=readings,
                signatures=signatures, comparison_tolerance=1e-12,
                budget=_large_budget())
            payload = result.to_json_bytes()
            max_traced_peak = max(max_traced_peak,
                                  tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        rss_after = _current_working_set_bytes()
        if rss_before is not None and rss_after is not None:
            rss_deltas.append(max(0, rss_after - rss_before))

        assert result.status == "complete", (case_index, result.refusals)
        domain = result.domains[0]
        total_vectors += result.planned_work.vector_candidates
        strata[stratum] += result.planned_work.vector_candidates
        serialized_bytes += len(payload)
        assert result.represented_work == result.planned_work
        assert result.to_record()["storage"][
            "candidates_materialized_during_construction"] == 0

        # Independently generated common translation and lattice-point carry
        # address a known truth element without scanning the Cartesian product.
        entry_rc = forward["entry_rc"]
        entry_index = next(index for index, entry in enumerate(domain.p1d_domain.entries)
                           if (entry.row_shift, entry.column_shift) == entry_rc)
        local_values = {"u": (left_residues[0], right_residues[0]),
                        "v": (left_residues[1], right_residues[1])}
        residue_indices = []
        for factor in domain.axis_factors:
            left_target = local_values[factor.left_local_axis][0]
            right_target = local_values[factor.right_local_axis][1]
            left_index = min(range(factor.left_side.candidate_count), key=lambda index:
                             abs(factor.left_side.candidate_values[index] - left_target))
            right_index = min(range(factor.right_side.candidate_count), key=lambda index:
                              abs(factor.right_side.candidate_values[index] - right_target))
            assert abs(factor.left_side.candidate_values[left_index] - left_target) < 1e-12
            assert abs(factor.right_side.candidate_values[right_index] - right_target) < 1e-12
            residue_indices.append((left_index, right_index))
        factor0, factor1 = domain.axis_factors
        pairs0 = factor0.left_side.candidate_count * factor0.right_side.candidate_count
        pairs1 = factor1.left_side.candidate_count * factor1.right_side.candidate_count
        vectors_per_entry = pairs0 * pairs1
        q0 = (residue_indices[0][0] * factor0.right_side.candidate_count
              + residue_indices[0][1])
        q1 = (residue_indices[1][0] * factor1.right_side.candidate_count
              + residue_indices[1][1])
        global_index = entry_index * vectors_per_entry + q0 * pairs1 + q1
        candidate = domain.candidate_at(global_index)
        assert candidate.shift_entry_rc == entry_rc
        truth = tuple(float(value) for value in right_translation - left_translation)
        assert candidate.vector == pytest.approx(truth, abs=1e-12), case_index

        del candidate, result, sources
        gc.collect()

    assert total_vectors == 2_195_063
    assert strata == {"orientation": 3_888,
                      "equal_b": 1_747_750,
                      "unequal_b": 443_425}
    assert serialized_bytes <= 8 * 1024 * 1024
    assert max_traced_peak <= 32 * 1024 * 1024
    if rss_deltas:
        assert max(rss_deltas) <= 32 * 1024 * 1024


def test_mixed_radix_is_exhaustive_and_uses_exact_large_integer_arithmetic():
    result, _p1d, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    domain = result.domains[0]
    factor0, factor1 = domain.axis_factors
    a0, b0 = factor0.left_side.candidate_count, factor0.right_side.candidate_count
    a1, b1 = factor1.left_side.candidate_count, factor1.right_side.candidate_count
    p0, p1 = a0 * b0, a1 * b1
    vectors_per_entry = p0 * p1
    assert domain.represented_candidate_count == len(domain.p1d_domain.entries) * vectors_per_entry
    seen = set()
    for entry_index, entry in enumerate(domain.p1d_domain.entries):
        for i_left0 in range(a0):
            for i_right0 in range(b0):
                for i_left1 in range(a1):
                    for i_right1 in range(b1):
                        q0 = i_left0 * b0 + i_right0
                        q1 = i_left1 * b1 + i_right1
                        index = entry_index * vectors_per_entry + q0 * p1 + q1
                        assert c2r._decode_mixed_radix(domain, index) == (
                            entry_index, (i_left0, i_right0, i_left1, i_right1))
                        assert index not in seen
                        seen.add(index)
    assert seen == set(range(domain.represented_candidate_count))

    dimensions = (10**40 + 7, 10**35 + 9, 10**24 + 3, 10**18 + 5)
    factor_a = SimpleNamespace(left_side=SimpleNamespace(candidate_count=dimensions[0]),
                               right_side=SimpleNamespace(candidate_count=dimensions[1]))
    factor_b = SimpleNamespace(left_side=SimpleNamespace(candidate_count=dimensions[2]),
                               right_side=SimpleNamespace(candidate_count=dimensions[3]))
    vectors = math.prod(dimensions)
    fake = SimpleNamespace(axis_factors=(factor_a, factor_b),
                           represented_candidate_count=vectors * 11)
    index = fake.represented_candidate_count - 1
    entry_index, indices = c2r._decode_mixed_radix(fake, index)
    assert entry_index == 10
    assert indices == tuple(value - 1 for value in dimensions)


def test_independent_eager_order_formula_and_every_provenance_index_replay():
    result, p1d_result, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    domain = result.domains[0]
    factor0, factor1 = domain.axis_factors
    a0, b0 = factor0.left_side.candidate_count, factor0.right_side.candidate_count
    a1, b1 = factor1.left_side.candidate_count, factor1.right_side.candidate_count
    p0, p1 = a0 * b0, a1 * b1
    count = 0
    materialized = c2r.materialize_lateral_vector_domains(
        result, _materialization_budget(result))[0]
    assert len(materialized) == domain.represented_candidate_count
    for entry_index, entry in enumerate(domain.p1d_domain.entries):
        for i_left0 in range(a0):
            for i_right0 in range(b0):
                for i_left1 in range(a1):
                    for i_right1 in range(b1):
                        index = entry_index * p0 * p1 + (i_left0 * b0 + i_right0) * p1 \
                            + i_left1 * b1 + i_right1
                        candidate = domain.candidate_at(index)
                        assert candidate.to_record() == materialized[index].to_record()
                        assert candidate.to_record() == _eager_candidate_record_oracle(
                            domain, entry, ((i_left0, i_right0),
                                            (i_left1, i_right1)))
                        assert candidate.shift_entry is entry
                        assert candidate.shift_entry is p1d_result.domains[0].entries[entry_index]
                        residue_indices = ((i_left0, i_right0), (i_left1, i_right1))
                        expected_provenance_axes = []
                        expected_vector = []
                        for factor, (left_index, right_index) in zip(
                                (factor0, factor1), residue_indices):
                            component = (entry.column_shift
                                         if factor.right_shift_source_component == "column"
                                         else entry.row_shift)
                            left_value = factor.left_side.candidate_values[left_index]
                            right_value = factor.right_side.candidate_values[right_index]
                            delta = (factor.right_world_sign * component
                                     + factor.left_world_sign * left_value
                                     - factor.right_world_sign * right_value)
                            expected_vector.append((factor.world_axis, float(delta)))
                            expected_provenance_axes.append((
                                factor.world_axis, factor.left_side.record_sha256,
                                left_index, factor.right_side.record_sha256, right_index))
                            term = candidate.axis_terms[len(expected_vector) - 1]
                            assert term.left_residue.candidate_index == left_index
                            assert term.right_residue.candidate_index == right_index
                            assert term.delta == delta
                        assert candidate.vector == pytest.approx(tuple(
                            value for _axis, value in expected_vector))
                        assert candidate.provenance_key == (
                            (entry.row_shift, entry.column_shift),
                            tuple(expected_provenance_axes))
                        view = domain.iter_views(index, index + 1).__next__()
                        assert view.vector == tuple(expected_vector)
                        assert view.provenance_key == candidate.provenance_key
                        assert domain.provenance_key_at(index) == candidate.provenance_key
                        count += 1
    assert count == domain.represented_candidate_count
    assert domain.index_schema == "c2r-indexed-vector-domain-v1"
    assert domain.reconstruction_version == "c2r-candidate-reconstruct-v1"
    assert domain.ordering_identifier == \
        "entry_row_major_axis0_left_right_axis1_left_right"


def test_entry_groups_duplicate_numeric_vectors_and_residue_multiplicity_are_retained():
    result, _p1d, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    domain = result.domains[0]
    groups = tuple(domain.iter_entry_groups())
    vectors_per_entry = domain.represented_candidate_count // domain.joint_entry_count
    assert len(groups) == domain.joint_entry_count
    for entry_index, (entry, start, stop) in enumerate(groups):
        assert entry is domain.p1d_domain.entries[entry_index]
        assert (start, stop) == (entry_index * vectors_per_entry,
                                 (entry_index + 1) * vectors_per_entry)
        candidates = tuple(domain.iter_candidates(start, stop))
        assert len(candidates) == vectors_per_entry
        assert all(candidate.shift_entry is entry for candidate in candidates)
    vectors = [domain.vector_at(index) for index in range(domain.represented_candidate_count)]
    multiplicity = Counter(vectors)
    assert any(count > 1 for count in multiplicity.values())
    duplicate_values = next(value for value, count in multiplicity.items() if count > 1)
    matching_indices = [index for index, value in enumerate(vectors) if value == duplicate_values]
    assert len({domain.provenance_key_at(index) for index in matching_indices}) == len(matching_indices)
    start, stop = 3, 19
    independent_range_digest = hashlib.sha256()
    for view in domain.iter_views(start, stop):
        independent_range_digest.update(json.dumps(
            view.to_record(), sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8"))
        independent_range_digest.update(b"\n")
    assert domain.range_sha256(start, stop) == independent_range_digest.hexdigest()
    with pytest.raises(ValueError):
        tuple(domain.iter_views(-1, 0))
    with pytest.raises(ValueError):
        tuple(domain.iter_candidates(0, domain.represented_candidate_count + 1))


def test_exact_and_one_below_materialization_budgets_are_atomic(monkeypatch):
    result, *_ = _two_domain_vector_case()
    before = result.to_json_bytes()
    exact_budget = _materialization_budget(result)
    exact = c2r.materialize_lateral_vector_domains(result, exact_budget)
    assert len(exact) == 2
    assert sum(map(len, exact)) == result.planned_work.vector_candidates

    calls = 0
    original = c2r._candidate_at_index

    def counted(domain, index):
        nonlocal calls
        calls += 1
        return original(domain, index)

    monkeypatch.setattr(c2r, "_candidate_at_index", counted)
    with pytest.raises(c2r.C2RMaterializationBudgetExceeded):
        c2r.materialize_lateral_vector_domains(
            result, _materialization_budget(result, candidate_delta=-1))
    assert calls == 0
    with pytest.raises(c2r.C2RMaterializationBudgetExceeded):
        c2r.materialize_lateral_vector_domains(
            result, _materialization_budget(result, domain_delta=-1))
    assert calls == 0
    assert result.to_json_bytes() == before
    assert result.represented_work == result.planned_work
    with pytest.raises(ValueError):
        c2r.C2RMaterializationBudget(0, 2)
    with pytest.raises(ValueError):
        c2r.C2RMaterializationBudget(True, 2)


def test_exact_and_one_below_five_logical_work_budgets():
    support, matches, signatures = _two_identity_case()
    readings = _canonical_readings(signatures)
    p1d_result = p1d.build_shift_domains(
        support, matches, signatures, max_shift_cells=8,
        budget=_p1d_budget(), readings=readings)
    phases = tuple(_phase_for(match, readings) for match in matches)
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert result.status == "complete"
    plan = result.planned_work
    exact_budget = c2r.C2RVectorBudget(**{
        "max_" + key: value for key, value in plan.to_record().items()})
    exact = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=exact_budget)
    assert exact.status == "complete" and exact.represented_work == plan
    for field, count in plan.to_record().items():
        assert count > 1
        limits = exact_budget.to_record()
        limits["max_" + field] = count - 1
        exhausted = c2r.build_lateral_vector_domains(
            support, p1d_result, matches, phases, readings=readings,
            signatures=signatures, comparison_tolerance=1e-12,
            budget=c2r.C2RVectorBudget(**limits))
        assert exhausted.status == "budget_exhausted"
        assert exhausted.budget_failure_reason == "planned_budget_exceeded"
        assert exhausted.planned_work == plan
        assert exhausted.represented_work == c2r.C2RVectorWork(0, 0, 0, 0, 0)
        assert not exhausted.domains and not exhausted.refusals
    with pytest.raises(ValueError):
        c2r.C2RVectorBudget(True, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        c2r.C2RVectorBudget(1, 0, 1, 1, 1)
    with pytest.raises(TypeError):
        c2r.materialize_lateral_vector_domains(exact)
    assert inspect.signature(c2r.build_lateral_vector_domains).parameters[
        "budget"].default is inspect.Parameter.empty
    assert inspect.signature(c2r.materialize_lateral_vector_domains).parameters[
        "budget"].default is inspect.Parameter.empty


def test_input_permutations_identity_refusals_and_recomputed_source_changes():
    result, p1d_result, signatures, matches, support, phases, readings = _build_vector_case()
    permuted = c2r.build_lateral_vector_domains(
        support, p1d_result, tuple(reversed(matches)), tuple(reversed(phases)),
        readings=tuple(reversed(readings)), signatures=tuple(reversed(signatures)),
        comparison_tolerance=1e-12, budget=_large_budget())
    assert permuted.to_json_bytes() == result.to_json_bytes()

    missing_phase = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, (), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert missing_phase.status == "refused" and not missing_phase.domains
    assert missing_phase.refusals[0].reason == "residue_evidence_missing"
    duplicate_phase = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, (phases[0], phases[0]),
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, budget=_large_budget())
    assert duplicate_phase.status == "refused" and not duplicate_phase.domains
    assert duplicate_phase.refusals[0].reason == "residue_evidence_ambiguous"
    missing_match = c2r.build_lateral_vector_domains(
        support, p1d_result, (), phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert missing_match.status == "refused"
    assert missing_match.refusals[0].reason == "requested_identity_missing"

    changed_cells = signatures[0].cells.copy()
    changed_cells[0, 0] += 1.0
    changed_signatures = (replace(signatures[0], cells=changed_cells), *signatures[1:])
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, matches, phases, readings=readings,
        signatures=changed_signatures, comparison_tolerance=1e-12)
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=float("nan"))
    changed_support_record = replace(phases[0].relative_shift,
                                     score=phases[0].relative_shift.score + 0.25)
    changed_phase = replace(phases[0], relative_shift=changed_support_record)
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, matches, (changed_phase,), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12)
    changed_match = replace(matches[0], score=matches[0].score + 0.25)
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, (changed_match,), phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12)
    extra_reading = replace(readings[0], frame="unselected_frame")
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, p1d_result, matches, phases,
        readings=(*readings, extra_reading), signatures=signatures,
        comparison_tolerance=1e-12)
    changed_p1d = p1d_result
    object.__setattr__(changed_p1d, "input_sha256", "0" * 64)
    assert not c2r.validate_lateral_vector_domain_inputs(
        result, support, changed_p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12)


def test_multi_identity_input_permutation_is_nonvacuously_byte_identical():
    result, p1d_result, signatures, matches, support, phases, readings = (
        _two_domain_vector_case())
    assert len(matches) == len(phases) == len(result.domains) == 2
    permuted = c2r.build_lateral_vector_domains(
        support, p1d_result, tuple(reversed(matches)), tuple(reversed(phases)),
        readings=tuple(reversed(readings)), signatures=tuple(reversed(signatures)),
        comparison_tolerance=1e-12, budget=_large_budget())
    assert permuted.status == "complete"
    assert permuted.to_json_bytes() == result.to_json_bytes()


def test_mixed_valid_and_refused_identities_publish_only_real_refusal_records():
    _baseline, p1d_result, signatures, matches, support, phases, readings = (
        _two_domain_vector_case())
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, (phases[0],), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert result.status == "refused"
    assert not result.domains
    assert result.represented_work == c2r.C2RVectorWork(0, 0, 0, 0, 0)
    assert len(result.refusals) == 1
    refused_identity = result.refusals[0].match_identity
    assert refused_identity in result.requested_match_identities
    assert result.refusals[0].reason == "residue_evidence_missing"
    plans = {plan.match_identity: plan for plan in result._preflight_plans}
    assert plans[refused_identity].refusal_reason == "residue_evidence_missing"
    ready_identity = next(identity for identity in result.requested_match_identities
                          if identity != refused_identity)
    assert plans[ready_identity].refusal_reason is None
    result.__post_init__()


def test_constructor_and_replace_attacks_fail_closed():
    result, _p1d, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    domain = result.domains[0]
    candidate = domain.candidate_at(0)
    with pytest.raises(ValueError):
        replace(candidate, candidate_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(candidate, p1d_replay_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(candidate, shift_entry=replace(
            candidate.shift_entry,
            row_shift=candidate.shift_entry.row_shift + 1))
    with pytest.raises(ValueError):
        replace(candidate, axis_terms=tuple(reversed(candidate.axis_terms)))
    term = candidate.axis_terms[0]
    with pytest.raises(ValueError):
        replace(term, delta=term.delta + 0.125)
    with pytest.raises(ValueError):
        replace(term, right_grid_integer_component=term.right_grid_integer_component + 1)
    with pytest.raises(ValueError):
        replace(term, signed_shift_component=term.signed_shift_component + 1)
    with pytest.raises(ValueError):
        replace(term, formula_replay_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(term, left_residue=replace(
            term.left_residue, candidate_value=(term.left_residue.candidate_value + 0.25) % 1.0))
    with pytest.raises(ValueError):
        replace(term, left_residue=replace(
            term.left_residue, candidate_index=term.left_residue.candidate_index + 1))
    with pytest.raises(ValueError):
        replace(domain, domain_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(domain, axis_factors=tuple(reversed(domain.axis_factors)))
    with pytest.raises(ValueError):
        replace(domain, index_schema="c2r-eager-v0")
    with pytest.raises(ValueError):
        replace(domain, reconstruction_version="mutated")
    with pytest.raises(ValueError):
        replace(domain, ordering_identifier="axis1-first")
    with pytest.raises(ValueError):
        replace(domain, materialization_scope="eager")
    with pytest.raises(ValueError):
        replace(domain, match_identity=replace(
            domain.match_identity,
            quadrant_of_right=(domain.match_identity.quadrant_of_right + 1) % 4))
    with pytest.raises(ValueError):
        replace(domain, represented_candidate_count=domain.represented_candidate_count + 1)
    factor = domain.axis_factors[0]
    with pytest.raises(ValueError):
        replace(factor, left_world_sign=-factor.left_world_sign)
    with pytest.raises(ValueError):
        replace(factor.left_side,
                candidate_values=(factor.left_side.candidate_values[0] + 0.01,
                                 *factor.left_side.candidate_values[1:]))
    with pytest.raises(ValueError):
        replace(factor, world_axis=domain.normal_constraint.world_axis)
    with pytest.raises(ValueError):
        replace(result, planned_work=replace(
            result.planned_work,
            vector_candidates=result.planned_work.vector_candidates + 1))
    with pytest.raises(ValueError):
        replace(result, represented_work=c2r.C2RVectorWork(0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        replace(result, result_sha256="0" * 64)


def test_finite_compatibility_positive_projective_version_and_digest_controls():
    result, *_ = _build_vector_case()
    domain = result.domains[0]
    homography = np.asarray(domain.homography, dtype=np.float64).reshape(3, 3)
    permutation = np.asarray(domain.physical_permutation, dtype=np.float64).reshape(2, 2)
    binding = c2r._build_binding(
        domain.match_identity, domain.c1_mapped_axes, domain.p1d_domain,
        domain.left_signature_geometry_sha256,
        domain.right_signature_geometry_sha256,
        domain.left_signature_content_sha256,
        domain.right_signature_content_sha256,
        domain.cell_homography_sha256)
    args = (homography, permutation, domain.left_origin_uv,
            domain.right_origin_uv, domain.comparison_tolerance)
    assert c2r._finite_compatible_map(*args, binding) is None

    projective = homography.copy()
    projective[2, 0] = 0.125
    assert c2r._finite_compatible_map(
        projective, permutation, domain.left_origin_uv,
        domain.right_origin_uv, domain.comparison_tolerance, binding
    ) == "homography_projective"
    unstable = homography.copy()
    unstable[2, 2] = 0.0
    assert c2r._finite_compatible_map(
        unstable, permutation, domain.left_origin_uv,
        domain.right_origin_uv, domain.comparison_tolerance, binding
    ) == "homography_denominator_invalid"
    unstable_scale = homography.copy()
    unstable_scale[2, 2] = 1e-320
    assert c2r._finite_compatible_map(
        unstable_scale, permutation, domain.left_origin_uv,
        domain.right_origin_uv, domain.comparison_tolerance, binding
    ) == "homography_denominator_unstable"
    incompatible_permutation = permutation.copy()
    incompatible_permutation[0, 0] *= -1
    assert c2r._finite_compatible_map(
        homography, incompatible_permutation, domain.left_origin_uv,
        domain.right_origin_uv, domain.comparison_tolerance, binding
    ) == "grid_basis_incompatible"
    assert c2r._finite_compatible_map(
        homography, permutation, domain.left_origin_uv,
        domain.right_origin_uv, domain.comparison_tolerance, {}
    ) == "compatibility_binding_missing"
    old_version = dict(binding, mapping_version="block-cell-grid-v1")
    assert c2r._finite_compatible_map(*args, old_version) == "homography_version_unsupported"
    changed_digest = dict(binding, c1_geometry_digests=("0" * 64,
                                                         binding["c1_geometry_digests"][1]))
    assert c2r._finite_compatible_map(*args, changed_digest) == "c1_geometry_digest_mismatch"
    changed_content = dict(
        binding, p1d_content_digests=("0" * 64, binding["p1d_content_digests"][1]))
    assert c2r._finite_compatible_map(*args, changed_content) == "p1d_content_digest_mismatch"
    changed_replay = dict(binding, replayed_homography_sha256="bad")
    assert c2r._finite_compatible_map(*args, changed_replay) == "p1f_homography_digest_invalid"
    budget = _materialization_budget(result)
    with pytest.raises(ValueError):
        replace(budget, max_materialized_domains=0)


def test_runtime_factor_count_mismatch_is_atomic(monkeypatch):
    _baseline, p1d_result, signatures, matches, support, phases, readings = _build_vector_case()

    def broken_factor(*_args, **_kwargs):
        raise ValueError("injected factor mismatch")

    monkeypatch.setattr(c2r, "_make_axis_factor", broken_factor)
    result = c2r.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert result.status == "budget_exhausted"
    assert result.budget_failure_reason == "runtime_count_mismatch"
    assert result.domains == result.refusals == ()
    assert result.represented_work == c2r.C2RVectorWork(0, 0, 0, 0, 0)


def test_full_refusal_taxonomy_is_typed_and_all_or_nothing(monkeypatch):
    result, p1d_result, signatures, matches, support, phases, readings = _build_vector_case()

    def build(*, support_value=support, p1d_value=p1d_result, match_values=matches,
              phase_values=phases, tolerance=1e-12):
        return c2r.build_lateral_vector_domains(
            support_value, p1d_value, match_values, phase_values,
            readings=readings, signatures=signatures,
            comparison_tolerance=tolerance, budget=_large_budget())

    invalid_c1 = plane_axis_support.C1AxisSupport(
        "no_constraints", (), None, (), (), (), (), ())
    assert build(support_value=invalid_c1).refusals[0].reason == "c1_status_invalid"
    monkeypatch.setattr(p1d, "validate_shift_domain_inputs", lambda *_a, **_k: False)
    assert build().refusals[0].reason == "p1d_status_invalid"
    monkeypatch.undo()

    assert build(match_values=(matches[0], matches[0])).refusals[0].reason == \
        "requested_identity_ambiguous"
    assert build(match_values=()).refusals[0].reason == "requested_identity_missing"
    assert build(tolerance=float("inf")).refusals[0].reason == "formula_replay_mismatch"

    wrong_shift = replace(
        phases[0].relative_shift,
        cell_shift_rc=(phases[0].relative_shift.cell_shift_rc[0] + 1,
                       phases[0].relative_shift.cell_shift_rc[1]))
    wrong_phase = replace(phases[0], relative_shift=wrong_shift)
    assert build(phase_values=(wrong_phase,)).refusals[0].reason == "provenance_mismatch"

    refused_phase = replace(phases[0], phase_candidates=(),
                            refusal="phase_evidence_absent")
    assert build(phase_values=(refused_phase,)).refusals[0].reason == \
        "residue_candidate_invalid"
    monkeypatch.setattr(c2r, "_mapped_axes_for",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad mapping")))
    assert build().refusals[0].reason == "axis_mapping_invalid"
    monkeypatch.undo()
    monkeypatch.setattr(c2r, "_finite_compatible_map",
                        lambda *_a, **_k: "grid_basis_incompatible")
    assert build().refusals[0].reason == "homography_incompatible"
    monkeypatch.undo()

    object.__setattr__(p1d_result.domains[0], "entries", ())
    monkeypatch.setattr(p1d, "validate_shift_domain_inputs", lambda *_a, **_k: True)
    assert build().refusals[0].reason == "joint_shift_invalid"
    assert result.status == "complete" and result.domains


def test_a3_physical_truth_provenance_is_reconstructible_by_exact_index():
    from test_c2r_vector_domain_a3_proof import _evaluate_physical_case

    physical = _evaluate_physical_case(0, 0, 0)
    match = physical["selected_match"]
    support = physical["support"]
    p1d_result = physical["domains"]
    readings = physical["readings"]
    signatures = physical["signatures"]
    phase = _phase_for(match, readings)
    indexed = c2r.build_lateral_vector_domains(
        support, p1d_result, (match,), (phase,), readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert indexed.status == "complete", indexed.refusals
    domain = indexed.domains[0]
    entry_indices = {(entry.row_shift, entry.column_shift): index
                     for index, entry in enumerate(domain.p1d_domain.entries)}
    factor0, factor1 = domain.axis_factors
    p1 = factor1.left_side.candidate_count * factor1.right_side.candidate_count
    vectors_per_entry = (factor0.left_side.candidate_count
                         * factor0.right_side.candidate_count * p1)
    for hit in physical["truth_supporting_provenance"]:
        row, column = hit["entry_rc"]
        e = entry_indices[(row, column)]
        (i_l0, i_r0), (i_l1, i_r1) = hit["residue_indices"]
        q0 = i_l0 * factor0.right_side.candidate_count + i_r0
        q1 = i_l1 * factor1.right_side.candidate_count + i_r1
        index = e * vectors_per_entry + q0 * p1 + q1
        candidate = domain.candidate_at(index)
        assert candidate.shift_entry_rc == (row, column)
        assert candidate.vector == pytest.approx(physical["truth"], abs=1e-12)


def test_production_module_import_boundary_is_closed():
    path = (Path(__file__).parents[1] / "src" / "mc_photogrammetry" /
            "reconstruction" / "plane_vector_domain.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_external = {
        "__future__", "dataclasses", "hashlib", "json", "math", "typing", "numpy",
    }
    allowed_internal = {
        "mc_photogrammetry.reconstruction.lattice_planes",
        "mc_photogrammetry.reconstruction.plane_axis_support",
        "mc_photogrammetry.reconstruction.plane_graph",
        "mc_photogrammetry.reconstruction.plane_residue",
        "mc_photogrammetry.reconstruction.plane_shift_domain",
    }
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "mc_photogrammetry.reconstruction":
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                modules.add(node.module)
    assert all((module in allowed_external or module in allowed_internal)
               for module in modules)
    dangerous_calls = {"open", "exec", "eval", "compile", "__import__"}
    assert not any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id in dangerous_calls
                   for node in ast.walk(tree))
