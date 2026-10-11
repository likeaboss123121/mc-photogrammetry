"""Independent exact-graph controls for the isolated C2-R-B solver kernel."""
from __future__ import annotations

from fractions import Fraction
import ast
import ctypes
from dataclasses import replace
import gc
import inspect
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from ctypes import wintypes

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_relative_translation as relative
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plane_vector_domain import (
    _a3_scale_case, _build_vector_case, _large_budget, _multi_residue_case,
    _phase_for,
)
from test_plane_shift_domain import _budget as _shift_budget
from test_c2r_vector_domain_a3_proof import (
    _canonical_identity, _physical_yaw, _reading, _signature,
)


def _raw_fixture_image(reading, translation: np.ndarray):
    """Render the coherent physical texture used by the public XYZ controls."""
    size = 1024
    rows, columns = np.indices((size, size), dtype=np.float64)
    rectification = reading.rectification
    principal_x, principal_y = rectification.principal_point
    local_u = ((columns - principal_x) / rectification.focal
               * reading.alternatives[0].distance_blocks)
    local_v = (-(rows - principal_y) / rectification.focal
               * reading.alternatives[0].distance_blocks)
    world_u = local_u + float(translation[0])
    world_v = local_v + float(translation[1])
    integer_u = np.floor(world_u).astype(np.int64)
    integer_v = np.floor(world_v).astype(np.int64)
    fraction_u = world_u - integer_u
    fraction_v = world_v - integer_v
    hashed = (integer_u * 73_856_093 + integer_v * 19_349_663 + integer_u * integer_v * 83_492_791) % 104_729
    amplitude = 0.3 + 2.0 * hashed.astype(np.float64) / 104_728.0
    grey = (3.0 + amplitude * np.sin(2.0 * np.pi * fraction_u)
            * np.sin(2.0 * np.pi * fraction_v)).astype(np.float32)
    return np.repeat(grey[..., None], 3, axis=2)


def _raw_fixture_signature(reading, blocks_per_period: float,
                           translation: np.ndarray, *, usable=None):
    """Render a physical cell texture before invoking ``plane_signature``."""
    image = _raw_fixture_image(reading, translation)
    return lattice_planes.plane_signature(
        image, reading, blocks_per_period, usable=usable, max_cells=32)


def _identity(left: str, right: str, ordinal: int = 0):
    return plane_axis_support.C1MatchIdentity(
        left, right, ordinal, ordinal, 0, 0, 1.0, 1.0, 0)


def _edge(left: str, right: str, delta: tuple[float, float, float], ordinal: int = 0):
    return relative._SyntheticEdge(
        _identity(left, right, ordinal),
        tuple(relative._dyadic_from_float(value) for value in delta))


def _fractions(record):
    return tuple(Fraction(numerator, denominator) for numerator, denominator in record)


class _GeneralSearchDomain:
    """Minimal A4-like domain for the private general-search unit controls.

    It exposes exactly the surface ``_solve_general_component`` reads: a mixed
    radix of one row-shift per joint entry, two zero-sign lateral factors and a
    fixed zero normal constraint.  No certificates are materialized for a
    contradiction, so the synthetic surface never reaches ``_make_certificate``.
    """

    def __init__(self, identity, values):
        self.match_identity = identity
        self.represented_candidate_count = len(values)
        self.joint_entry_count = len(values)
        self.p1d_domain = SimpleNamespace(entries=tuple(
            SimpleNamespace(row_shift=value, column_shift=0)
            for value in values))
        zero_side = SimpleNamespace(candidate_values=(0.0,), candidate_count=1)
        self.axis_factors = (
            SimpleNamespace(world_axis=0, right_shift_source_component="row",
                            left_world_sign=1, right_world_sign=1,
                            left_side=zero_side, right_side=zero_side),
            SimpleNamespace(world_axis=1, right_shift_source_component="column",
                            left_world_sign=1, right_world_sign=1,
                            left_side=zero_side, right_side=zero_side),
        )
        self.normal_constraint = SimpleNamespace(world_axis=2,
                                                 delta_translation=0.0)

    def iter_entry_groups(self):
        for index, entry in enumerate(self.p1d_domain.entries):
            yield entry, index, index + 1

    def vector_at(self, index):
        return ((0, float(self.p1d_domain.entries[index].row_shift)), (1, 0.0))

    def iter_views(self, start, stop):
        for index in range(start, stop):
            entry = self.p1d_domain.entries[index]
            yield plane_vector_domain.LateralVectorView(
                index, (entry.row_shift, 0), (),
                ((0, float(entry.row_shift)), (1, 0.0)))


def _general_search_case(row_values):
    """Six-edge A/B/C/D input whose three chords are three distinct cycles.

    ``row_values`` maps each identity to its single-axis shift options.  The
    caller decides which chords close, so the same helper builds both a unique
    graph and a genuine partial-failure contradiction.
    """
    identities = (
        _identity("A", "B"), _identity("A", "C"), _identity("A", "D"),
        _identity("B", "C"), _identity("B", "D"), _identity("C", "D"),
    )
    plan = relative._make_graph_plan(identities)
    component = plan.components[0]
    domains = {identity: _GeneralSearchDomain(identity, row_values[identity])
               for identity in identities}
    return component, domains


def test_directed_triangle_and_reverse_traversal_match_independent_sign_oracle():
    ab = (2.0, -1.0, 4.0)
    bc = (3.0, 5.0, -2.0)
    ac = (5.0, 4.0, 2.0)
    _plan, forward = relative._singleton_graph_kernel((
        _edge("A", "B", ab), _edge("B", "C", bc), _edge("A", "C", ac)),
        Fraction(0))
    assert len(forward) == 1
    assert forward[0].anchor == "A"
    assert dict(forward[0].translations) == {
        "A": ((0, 1), (0, 1), (0, 1)),
        "B": ((2, 1), (-1, 1), (4, 1)),
        "C": ((5, 1), (4, 1), (2, 1)),
    }
    assert forward[0].outcome == "unique_validated"
    assert _fractions(forward[0].residuals[0]) == (Fraction(0),) * 3

    _reverse_plan, reverse = relative._singleton_graph_kernel((
        _edge("B", "A", (-ab[0], -ab[1], -ab[2])),
        _edge("C", "B", (-bc[0], -bc[1], -bc[2])),
        _edge("C", "A", (-ac[0], -ac[1], -ac[2])),
    ), Fraction(0))
    assert reverse[0].outcome == "unique_validated"
    assert dict(reverse[0].translations) == dict(forward[0].translations)


def test_tree_only_component_is_explicitly_unvalidated():
    plan, result = relative._singleton_graph_kernel((
        _edge("A", "B", (0.5, -2.0, 7.0)),
        _edge("B", "C", (-1.25, 3.0, 0.0)),
    ), Fraction(0))
    assert len(plan.components) == 1
    assert plan.components[0].chords == ()
    assert plan.components[0].anchor == "A"
    assert result[0].outcome == "unique_unvalidated_tree"
    assert dict(result[0].translations)["C"] == (
        (-3, 4), (1, 1), (7, 1))


def test_disconnected_closing_triangles_have_independent_canonical_gauges():
    edges = (
        _edge("A", "B", (1.0, 2.0, 0.0)),
        _edge("B", "C", (3.0, -1.0, 4.0)),
        _edge("A", "C", (4.0, 1.0, 4.0)),
        _edge("X", "Y", (-2.0, 0.0, 1.0)),
        _edge("Y", "Z", (5.0, 3.0, -2.0)),
        _edge("X", "Z", (3.0, 3.0, -1.0)),
    )
    plan, result = relative._singleton_graph_kernel(edges, Fraction(0))
    assert tuple(component.frames for component in plan.components) == (
        ("A", "B", "C"), ("X", "Y", "Z"))
    assert tuple(component.anchor for component in plan.components) == ("A", "X")
    assert tuple(component.outcome for component in result) == (
        "unique_validated", "unique_validated")


def test_frame_renaming_preserves_pairwise_differences_after_regauging():
    original = (
        _edge("A", "B", (2.0, -1.0, 4.0)),
        _edge("B", "C", (3.0, 5.0, -2.0)),
        _edge("A", "C", (5.0, 4.0, 2.0)),
    )
    renamed = tuple(
        _edge({"A": "Z", "B": "X", "C": "Y"}[edge.identity.left_frame],
              {"A": "Z", "B": "X", "C": "Y"}[edge.identity.right_frame],
              tuple(float(Fraction(n, d)) for n, d in edge.delta),
              edge.identity.left_plane)
        for edge in original)
    _original_plan, original_results = relative._singleton_graph_kernel(
        original, Fraction(0))
    _renamed_plan, renamed_results = relative._singleton_graph_kernel(
        renamed, Fraction(0))
    original_translations = {
        frame: tuple(Fraction(n, d) for n, d in xyz)
        for frame, xyz in original_results[0].translations}
    renamed_translations = {
        frame: tuple(Fraction(n, d) for n, d in xyz)
        for frame, xyz in renamed_results[0].translations}
    frame_map = {"A": "Z", "B": "X", "C": "Y"}
    for left in original_translations:
        for right in original_translations:
            old = tuple(original_translations[right][axis]
                        - original_translations[left][axis] for axis in range(3))
            new = tuple(renamed_translations[frame_map[right]][axis]
                        - renamed_translations[frame_map[left]][axis]
                        for axis in range(3))
            assert new == old


def test_parallel_edges_and_multiple_chords_close_under_all_input_orders():
    ab = _edge("A", "B", (2.0, -1.0, 4.0), 0)
    ab_parallel = _edge("A", "B", (2.0, -1.0, 4.0), 1)
    bc = _edge("B", "C", (3.0, 5.0, -2.0), 0)
    ac = _edge("A", "C", (5.0, 4.0, 2.0), 0)
    expected_plan, expected = relative._singleton_graph_kernel(
        (ab, ab_parallel, bc, ac), Fraction(0))
    assert len(expected_plan.components[0].tree_edges) == 2
    assert len(expected_plan.components[0].chords) == 2
    assert expected[0].outcome == "unique_validated"
    assert all(_fractions(residual) == (Fraction(0),) * 3
               for residual in expected[0].residuals)
    for permutation in ((ac, bc, ab_parallel, ab), (bc, ab, ac, ab_parallel)):
        plan, result = relative._singleton_graph_kernel(permutation, Fraction(0))
        assert plan == expected_plan
        assert result == expected


def test_parallel_edge_with_incompatible_cycle_delta_is_a_contradiction():
    plan, result = relative._singleton_graph_kernel((
        _edge("A", "B", (2.0, -1.0, 4.0), 0),
        _edge("A", "B", (2.0, -1.0, 5.0), 1),
        _edge("B", "C", (3.0, 5.0, -2.0)),
        _edge("A", "C", (5.0, 4.0, 2.0)),
    ), Fraction(0))
    assert len(plan.components[0].chords) == 2
    assert result[0].outcome == "contradiction"
    assert any(component != (0, 1) for residual in result[0].residuals
               for component in residual)


def test_general_search_uses_one_shared_assignment_for_multiple_cycles(monkeypatch):
    identities = (
        _identity("A", "B"), _identity("A", "C"), _identity("A", "D"),
        _identity("B", "C"), _identity("B", "D"), _identity("C", "D"),
    )
    component, domains = _general_search_case({
        identities[0]: (0, 1),  # A -> B
        identities[1]: (0, 1),  # A -> C
        identities[2]: (0, 1),  # A -> D
        identities[3]: (0,),    # B -> C: xC-xB=0
        identities[4]: (0,),    # B -> D: xD-xB=0
        identities[5]: (1,),    # C -> D: xD-xC=1
    })
    assert component.tree_edges == identities[:3]
    assert component.chords == identities[3:]
    # Each cycle is independently satisfiable, but no single global assignment
    # to A->B, A->C, A->D can satisfy all three simultaneously.
    tree_options = {identity: (0, 1) for identity in component.tree_edges}
    chord_targets = {identities[3]: 0, identities[4]: 0, identities[5]: 1}
    for chord, target in chord_targets.items():
        assert any(
            values[chord.right_frame] - values[chord.left_frame] == target
            for values in (
                {"A": 0, "B": ab, "C": ac, "D": ad}
                for ab in tree_options[identities[0]]
                for ac in tree_options[identities[1]]
                for ad in tree_options[identities[2]]))
    assert not any(
        all(values[chord.right_frame] - values[chord.left_frame] == target
            for chord, target in chord_targets.items())
        for values in (
            {"A": 0, "B": ab, "C": ac, "D": ad}
            for ab in tree_options[identities[0]]
            for ac in tree_options[identities[1]]
            for ad in tree_options[identities[2]]))

    monkeypatch.setattr(
        relative, "_make_component_output",
        lambda _component, _domains, outcome, _assignments, _failures,
               exhaustive, _runtime: SimpleNamespace(
                   outcome=outcome, exhaustive=exhaustive))
    result = relative._solve_general_component(
        component, domains, Fraction(0), relative._RuntimeCounter(_runtime_budget()))
    assert result.outcome == "contradiction"
    assert result.exhaustive


def test_partial_failure_contradiction_seals_a_valid_component_record():
    """A cycle_rank-3 contradiction with one open chord must seal.

    The component record is outcome-specific: a contradiction retains exactly
    its failing chords, so a proper-subset failure (fewer residuals than the
    cycle rank) is a genuine, well-formed record.  Before the repair the removed
    universal ``len(cycle_residuals) == cycle_rank`` guard rejected it and the
    public solver downgraded the whole graph to
    ``budget_exhausted`` / ``runtime_count_mismatch``.
    """
    identities = (
        _identity("A", "B"), _identity("A", "C"), _identity("A", "D"),
        _identity("B", "C"), _identity("B", "D"), _identity("C", "D"),
    )
    component, domains = _general_search_case({
        identities[0]: (0,),    # A -> B
        identities[1]: (0,),    # A -> C
        identities[2]: (0,),    # A -> D
        identities[3]: (0,),    # B -> C: xC-xB=0 closes
        identities[4]: (1,),    # B -> D: xD-xB=1 cannot close
        identities[5]: (0,),    # C -> D: xD-xC=0 closes
    })
    result = relative._solve_general_component(
        component, domains, Fraction(0),
        relative._RuntimeCounter(_runtime_budget()))
    assert isinstance(result, relative.C2RBComponentResult)
    assert result.outcome == "contradiction"
    assert result.exhaustive is True
    assert result.cycle_rank == 3
    assert len(result.failing_chords) == 1
    assert result.cycle_rank > len(result.failing_chords)
    assert result.translation_classes == ()
    assert tuple(residual.chord_identity
                 for residual in result.cycle_residuals) == result.failing_chords
    assert all(residual.matching_provenance_count == 0
               for residual in result.cycle_residuals)
    # The digest is resealed over the retained (proper-subset) residual tuple.
    assert relative._digest_json(relative._component_payload(result)) \
        == result.component_sha256


def test_partial_failure_contradiction_still_binds_its_failing_chords():
    """Removing the universal length guard must not drop the subset binding."""
    identities = (
        _identity("A", "B"), _identity("A", "C"), _identity("A", "D"),
        _identity("B", "C"), _identity("B", "D"), _identity("C", "D"),
    )
    component, domains = _general_search_case({
        identities[0]: (0,), identities[1]: (0,), identities[2]: (0,),
        identities[3]: (0,), identities[4]: (1,), identities[5]: (0,),
    })
    result = relative._solve_general_component(
        component, domains, Fraction(0),
        relative._RuntimeCounter(_runtime_budget()))
    assert len(result.cycle_residuals) == 1
    # A contradiction with no named failing chord, or with residuals that do
    # not match its failing chords, is still rejected.
    with pytest.raises(ValueError):
        type(result).__post_init__(_unchecked_record_copy(
            result, cycle_residuals=()))
    with pytest.raises(ValueError):
        type(result).__post_init__(_unchecked_record_copy(
            result, failing_chords=(), cycle_residuals=()))
    with pytest.raises(ValueError):
        type(result).__post_init__(_unchecked_record_copy(
            result, cycle_residuals=result.cycle_residuals * 2))



def _working_set_bytes():
    class ProcessMemoryCounters(ctypes.Structure):
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
        ]
    if sys.platform != "win32":
        pytest.skip("working-set RSS gate uses the Windows process API")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    process = kernel32.GetCurrentProcess()
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
    get_memory_info.restype = wintypes.BOOL
    if not get_memory_info(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.WorkingSetSize


def test_1000_edge_singleton_kernel_only_meets_time_and_rss_gate():
    # Kernel-only coverage: these are private synthetic edges, not replay-valid
    # A4 domains. Public matcher evidence is intentionally exercised by the
    # ambiguity integration below; singleton behavior belongs to this private
    # kernel boundary only.
    edges = tuple(
        _edge(f"F{index:04d}", f"F{index + 1:04d}",
              (0.125, -0.25, 0.5), index)
        for index in range(1000))
    gc.collect()
    before = _working_set_bytes()
    started = time.perf_counter()
    plan, result = relative._singleton_graph_kernel(edges, Fraction(0))
    elapsed = time.perf_counter() - started
    after = _working_set_bytes()
    assert elapsed <= 2.0
    assert max(0, after - before) <= 32 * 1024 * 1024
    assert len(plan.components) == 1
    assert len(plan.components[0].frames) == 1001
    assert result[0].outcome == "unique_unvalidated_tree"


def test_public_a4_ring_retains_supported_shifts_and_abstains_ambiguously():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _ambiguous_a4_ring_case())
    assert tuple(domain.represented_candidate_count
                 for domain in vector_result.domains) == (81,) * 5
    assert tuple(match.cell_shift_rc for match in matches) == (
        (0, 2), (2, 1), (1, -2), (-1, -2), (-2, 1))
    assert all(match.overlap_cells >= plane_graph.MIN_OVERLAP_CELLS
               and np.isfinite(match.psr) for match in matches)
    assert tuple(sum(match.cell_shift_rc[index] for match in matches)
                 for index in range(2)) == (0, 0)
    declared_points = {
        "R0": (0, 0), "R1": (2, 0), "R2": (3, 2),
        "R3": (1, 3), "R4": (-1, 2),
    }
    for domain in vector_result.domains:
        identity = domain.match_identity
        left = declared_points[identity.left_frame]
        right = declared_points[identity.right_frame]
        expected = ((0, float(right[0] - left[0])),
                    (1, float(right[1] - left[1])))
        assert any(view.vector == expected for view in domain.iter_views(
            0, domain.represented_candidate_count))
    gc.collect()
    before = _working_set_bytes()
    started = time.perf_counter()
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10,
            max_tree_edges=10, max_chords=10, max_cycle_rows=30))
    elapsed = time.perf_counter() - started
    after = _working_set_bytes()
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.connected and len(solved.components) == 1
    assert solved.components[0].outcome == "ambiguous"
    assert solved.placement_readiness == "review_required"
    assert solved.deterministic_work.edges == 5
    assert solved.deterministic_work.chords == 1
    assert solved.runtime_work.candidate_views_scanned == 4 * 81
    assert solved.runtime_work.candidate_at_calls == 0
    assert solved.runtime_work.membership_probes > 4 * 81
    assert elapsed <= 5.0
    assert max(0, after - before) <= 32 * 1024 * 1024
    assert relative.validate_relative_translation_graph(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)
    endpoint = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=v2a.V2ABudget(
            max_readings=100, max_edges=100, max_endpoint_labels=1000,
            max_residues_inspected=10000,
            max_edge_candidates_inspected=10000,
            max_partial_label_assignments=10000,
            max_consistency_checks=10000,
            max_result_validation_cross_links=10000,
            max_distinct_assignments=2, max_retained_witnesses=2))
    assert endpoint.status == "refused"
    assert endpoint.refusals[0].reason == "gate_not_ready"
    assert endpoint.classification is None and endpoint.witnesses == ()


def test_over_two_million_a4_input_preflight_is_factorized_and_atomic():
    (support, p1d_result, signatures, matches, readings, phases,
     _left_residues, _right_residues, _forwards) = _a3_scale_case()
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    assert vector_result.planned_work.vector_candidates > 2_195_063
    assert sum(domain.represented_candidate_count
               for domain in vector_result.domains) == 2_367_488

    gc.collect()
    before = _working_set_bytes()
    started = time.perf_counter()
    result = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(max_p1d_entries=1))
    elapsed = time.perf_counter() - started
    after = _working_set_bytes()
    assert result.status == "budget_exhausted"
    assert result.budget_failure_reason == "planned_budget_exceeded"
    assert result.deterministic_work.represented_candidate_count == 2_367_488
    assert result.runtime_work == relative.C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)
    assert result.components == () and result.refusals == ()
    assert len(result.to_json_bytes()) <= 8 * 1024 * 1024
    assert elapsed <= 5.0
    assert max(0, after - before) <= 32 * 1024 * 1024

    gc.collect()
    before = _working_set_bytes()
    started = time.perf_counter()
    bounded = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(max_candidate_views_scanned=1))
    elapsed = time.perf_counter() - started
    after = _working_set_bytes()
    assert bounded.status == "budget_exhausted"
    assert bounded.budget_failure_reason == "runtime_budget_exceeded"
    assert bounded.runtime_work.candidate_views_scanned == 2
    assert bounded.runtime_work.membership_probes > 0
    assert bounded.runtime_work.candidate_at_calls == 0
    assert bounded.components == () and bounded.refusals == ()
    assert len(bounded.to_json_bytes()) <= 8 * 1024 * 1024
    assert elapsed <= 5.0
    assert max(0, after - before) <= 32 * 1024 * 1024


def test_cycle_with_wrong_directed_edge_is_contradiction_without_nearest_choice():
    plan, result = relative._singleton_graph_kernel((
        _edge("A", "B", (1.0, 0.0, 0.0)),
        _edge("B", "C", (1.0, 0.0, 0.0)),
        _edge("A", "C", (3.0, 0.0, 0.0)),
    ), Fraction(0))
    assert len(plan.components[0].chords) == 1
    assert result[0].outcome == "contradiction"
    assert result[0].residuals[0][0] == (-1, 1)


def test_budget_records_reject_boolean_and_inconsistent_counts():
    fields = {name: 4 for name in relative.C2RBudget.__dataclass_fields__}
    fields["max_provenance_witnesses"] = 2
    fields["max_certificates"] = 2
    assert relative.C2RBudget(**fields).max_edges == 4
    fields["max_edges"] = True
    with pytest.raises(ValueError):
        relative.C2RBudget(**fields)
    with pytest.raises(ValueError):
        relative.C2RBDeterministicWork(1, 2, 3, 1, 1, 3, 0, 0, 0)
    fields = {name: 4 for name in relative.C2RBudget.__dataclass_fields__}
    fields["max_certificates"] = 1
    with pytest.raises(ValueError):
        relative.C2RBudget(**fields)
    fields["max_certificates"] = 2
    fields["max_p1d_entries"] = 0
    with pytest.raises(ValueError):
        relative.C2RBudget(**fields)


def _runtime_budget(**overrides):
    limits = {name: 100_000 for name in relative.C2RBudget.__dataclass_fields__}
    limits.update({
        "max_components": 100, "max_frames": 100, "max_edges": 100,
        "max_tree_edges": 100, "max_chords": 100, "max_cycle_rows": 300,
        "max_p1d_entries": 100_000, "max_residue_values": 100_000,
        "max_provenance_witnesses": 100_000, "max_certificates": 100_000,
    })
    limits.update(overrides)
    return relative.C2RBudget(**limits)


def _unchecked_record_copy(record, **changes):
    """Make deliberately malformed frozen records to exercise public guards."""
    clone = object.__new__(type(record))
    for field in record.__dataclass_fields__:
        object.__setattr__(clone, field, changes.get(field, getattr(record, field)))
    return clone


def _a4_triangle_case(radius=4):
    """Three forward-sampled planes with one canonical directed cycle."""
    translations = {
        "A_left": np.asarray((0.0, 0.0), dtype=np.float64),
        "B_left": np.asarray((2.0, 0.0), dtype=np.float64),
        "C_left": np.asarray((0.0, 3.0), dtype=np.float64),
    }
    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame, translation in translations.items():
        residues = tuple(float((-value) % 1.0) for value in translation)
        reading = _reading(
            frame, 2, 0.5, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signatures.append(_signature(reading, 0.5, basis, translation))
    readings, signatures = tuple(readings), tuple(signatures)
    matches = tuple(match for match in plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=radius)
        if match.quadrant_of_right == 0)
    assert len(matches) == 3
    blocks = {_canonical_identity(readings, reading): 0.5 for reading in readings}
    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(translations)),
        quadrants={frame: 0 for frame in translations},
        translations={frame: (0.0, 0.0, 0.0) for frame in translations},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True, blocks_per_period=blocks, reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=radius,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches,
            support, phases, readings)


def _unique_a4_triangle_case():
    """Public A4 cycle with one supported vector on each edge."""
    translations = {
        "A_left": np.asarray((0.0, 0.0), dtype=np.float64),
        "B_left": np.asarray((2.0, 0.0), dtype=np.float64),
        "C_left": np.asarray((0.0, 3.0), dtype=np.float64),
    }
    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame, translation in translations.items():
        residues = tuple(float((-value) % 1.0) for value in translation)
        reading = _reading(
            frame, 2, 1.0, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signature = _signature(reading, 1.0, basis, translation)
        coverage = np.zeros(signature.coverage.shape, dtype=bool)
        row = (coverage.shape[0] - 3) // 2
        column = (coverage.shape[1] - 3) // 2
        coverage[row:row + 3, column:column + 3] = True
        signatures.append(replace(signature, coverage=coverage))

    readings, signatures = tuple(readings), tuple(signatures)
    matches = tuple(match for match in plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4)
        if match.quadrant_of_right == 0)
    assert len(matches) == 3
    blocks = {_canonical_identity(readings, reading): 1.0 for reading in readings}
    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(translations)),
        quadrants={frame: 0 for frame in translations},
        translations={frame: (0.0, 0.0, 0.0) for frame in translations},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True, blocks_per_period=blocks, reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches,
            support, phases, readings)


def _ambiguous_a4_ring_case(size: int = 5):
    """Public A4 ring with measured shifts and complete supported domains.

    All frames share the normal axis, side, distance and b, and their integer
    translations put every residue on the period origin.  The selected shifts
    close exactly, while P1-D truthfully retains every support-admissible shift;
    the downstream solver must therefore preserve ambiguity rather than choose
    one of the closing assignments.
    """
    lattice = ((0, 0), (2, 0), (3, 2), (1, 3), (-1, 2))
    assert size == len(lattice)
    translations = {
        f"R{index}": np.asarray(point, dtype=np.float64)
        for index, point in enumerate(lattice)}
    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame, translation in translations.items():
        residues = tuple(float((-value) % 1.0) for value in translation)
        reading = _reading(
            frame, 2, 1.0, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signatures.append(_raw_fixture_signature(reading, 1.0, translation))

    readings, signatures = tuple(readings), tuple(signatures)
    signature_by_frame = {signature.reading.frame: signature
                          for signature in signatures}
    matches = []
    for index in range(size):
        left = f"R{index}"
        right = f"R{(index + 1) % size}"
        pair = plane_graph.match_plane_pair(
            signature_by_frame[left], signature_by_frame[right],
            quadrant=0, max_shift_cells=4)
        assert pair is not None
        matches.append(replace(
            pair, left_plane=_canonical_identity(readings, signature_by_frame[left].reading)[1],
            right_plane=_canonical_identity(readings, signature_by_frame[right].reading)[1]))
    matches = tuple(matches)
    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(translations)),
        quadrants={frame: 0 for frame in translations},
        translations={frame: (0.0, 0.0, 0.0) for frame in translations},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): 1.0
                           for reading in readings},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches,
            support, phases, readings)


def _xyz_ready_case():
    """Two frames whose three physical normals jointly fix one translation."""
    declared = {"A": (0, 0, 0), "B": (2, 3, 4)}
    readings, signatures = [], []
    for frame in ("A", "B"):
        point = declared[frame]
        for axis in range(3):
            tangents = tuple(value for value in range(3) if value != axis)
            distance = 10.0 - float(point[axis])
            tangent_translation = np.asarray(
                tuple(point[value] for value in tangents), dtype=np.float64)
            reading = _reading(
                frame, axis, 1.0, distance, 1.0, (0, 0),
                tuple(float((-value) % 1.0) for value in tangent_translation),
                samples=128, extent=16.0 / distance, focal=30.0)
            readings.append(reading)
            signatures.append(_raw_fixture_signature(
                reading, 1.0, tangent_translation))
    readings, signatures = tuple(readings), tuple(signatures)
    matches = []
    for axis in range(3):
        left, right = signatures[axis], signatures[3 + axis]
        match = plane_graph.match_plane_pair(
            left, right, quadrant=0, max_shift_cells=4)
        assert match is not None
        matches.append(replace(
            match,
            left_plane=_canonical_identity(readings, left.reading)[1],
            right_plane=_canonical_identity(readings, right.reading)[1]))
    matches = tuple(matches)
    graph_solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=(("A", "B", 0),))
    assert graph_solution.reason == ""
    assert graph_solution.translations == {
        "A": (0.0, 0.0, 0.0), "B": (2.0, 3.0, 4.0)}
    support = plane_axis_support.build_axis_support(
        graph_solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(max_domains=10), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches, support, phases,
            readings, declared)


def test_public_xyz_parallel_edges_select_one_declared_translation_class():
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings, declared) = _xyz_ready_case()
    assert tuple(domain.represented_candidate_count
                 for domain in vector_result.domains) == (81, 81, 81)
    assert all(signature.cells.shape == (32, 32)
               and signature.origin_uv == (0.0, 0.0)
               for signature in signatures)
    for axis, forward in enumerate(matches):
        tangents = tuple(value for value in range(3) if value != axis)
        tangent_delta = tuple(
            declared["B"][value] - declared["A"][value]
            for value in tangents)
        expected_shift = (tangent_delta[1], tangent_delta[0])
        assert forward.cell_shift_rc == expected_shift
        assert forward.overlap_cells >= plane_graph.MIN_OVERLAP_CELLS
        assert np.isfinite(forward.psr)
        reverse = plane_graph.match_plane_pair(
            signatures[3 + axis], signatures[axis],
            quadrant=0, max_shift_cells=4)
        assert reverse is not None
        assert reverse.cell_shift_rc == tuple(-value for value in expected_shift)
        assert reverse.overlap_cells == forward.overlap_cells
        assert reverse.psr == pytest.approx(forward.psr, rel=0.0, abs=1e-12)
    for domain in vector_result.domains:
        identity = domain.match_identity
        axis = identity.left_axis
        tangents = tuple(value for value in range(3) if value != axis)
        expected = tuple(
            (world_axis, float(declared[identity.right_frame][world_axis]
                               - declared[identity.left_frame][world_axis]))
            for world_axis in tangents)
        assert any(view.vector == expected for view in domain.iter_views(
            0, domain.represented_candidate_count))
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10,
            max_tree_edges=10, max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.placement_readiness == "ready"
    component = solved.components[0]
    assert component.outcome == "unique_validated" and component.exhaustive
    assert len(component.translation_classes) == 1
    assert dict(component.translation_classes[0].translations_dyadic_xyz) == {
        "A": ((0, 1), (0, 1), (0, 1)),
        "B": ((2, 1), (3, 1), (4, 1)),
    }


def _public_normal_sign_mutant_cycle_case():
    """Rebuild source evidence where one selected plane's normal side is flipped."""
    # Each frame carries two distinct same-axis readings.  The chosen edges
    # intentionally use different readings at shared frame vertices, so the
    # signed normal differences do not telescope as one per-frame potential.
    reading_specs = (
        ("A", 1.0, 0), ("A", -1.0, 1),
        ("B", 1.0, 0), ("B", -1.0, 1),
        ("C", 1.0, 0), ("C", -1.0, 1),
    )
    basis = np.eye(2, dtype=np.float64)
    translation = np.asarray((0.0, 0.0), dtype=np.float64)
    readings = []
    signatures = []
    by_role = {}
    for frame, side, role in reading_specs:
        reading = _reading(
            frame, 2, 1.0, 2.0, side, (0, 0), (0.0, 0.0),
            samples=128, extent=4.0, centre_offset=(role * 0.01, 0.0))
        readings.append(reading)
        by_role[(frame, role)] = reading
        signatures.append(_raw_fixture_signature(reading, 1.0, translation))

    readings, signatures = tuple(readings), tuple(signatures)

    chosen_pairs = (
        (by_role[("A", 0)], by_role[("B", 0)]),
        (by_role[("B", 1)], by_role[("C", 0)]),
        (by_role[("A", 1)], by_role[("C", 1)]),
    )
    matches = []
    for left, right in chosen_pairs:
        left_signature = next(value for value in signatures if value.reading is left)
        right_signature = next(value for value in signatures if value.reading is right)
        pair = plane_graph.match_plane_pair(
            left_signature, right_signature, quadrant=0, max_shift_cells=4)
        assert pair is not None
        matches.append(replace(
            pair, left_plane=_canonical_identity(readings, left)[1],
            right_plane=_canonical_identity(readings, right)[1]))
    matches = tuple(matches)
    solution = plane_graph.ComponentSolution(
        frames=("A", "B", "C"),
        quadrants={frame: 0 for frame in ("A", "B", "C")},
        translations={frame: (0.0, 0.0, 0.0) for frame in ("A", "B", "C")},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): 1.0
                           for reading in readings},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches,
            support, phases, readings)


def _public_lateral_sign_mutant_cycle_case():
    """Rebuild source evidence where one lateral residue phase is sign-flipped.

    Every reading keeps the same normal axis, side and distance, so all three
    normal deltas are zero.  The two readings of each frame carry opposite
    lateral residue phases; the closing cycle reuses different readings at the
    same frame vertices, so the sign-flipped lateral phase cannot telescope and
    the cycle must be rejected on a lateral axis rather than the normal one.
    """
    reading_specs = (
        ("A", (0.0, 0.0), 0), ("A", (0.5, 0.0), 1),
        ("B", (0.0, 0.0), 0), ("B", (0.5, 0.0), 1),
        ("C", (0.0, 0.0), 0), ("C", (0.5, 0.0), 1),
    )
    basis = np.eye(2, dtype=np.float64)
    translation = np.asarray((0.0, 0.0), dtype=np.float64)
    readings = []
    signatures = []
    by_role = {}
    for frame, residues, role in reading_specs:
        reading = _reading(
            frame, 2, 1.0, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0, centre_offset=(role * 0.01, 0.0))
        readings.append(reading)
        by_role[(frame, role)] = reading
        signatures.append(_raw_fixture_signature(reading, 1.0, translation))

    readings, signatures = tuple(readings), tuple(signatures)

    chosen_pairs = (
        (by_role[("A", 0)], by_role[("B", 0)]),
        (by_role[("B", 1)], by_role[("C", 0)]),
        (by_role[("A", 1)], by_role[("C", 1)]),
    )
    matches = []
    for left, right in chosen_pairs:
        left_signature = next(value for value in signatures if value.reading is left)
        right_signature = next(value for value in signatures if value.reading is right)
        pair = plane_graph.match_plane_pair(
            left_signature, right_signature, quadrant=0, max_shift_cells=4)
        assert pair is not None
        matches.append(replace(
            pair, left_plane=_canonical_identity(readings, left)[1],
            right_plane=_canonical_identity(readings, right)[1]))
    matches = tuple(matches)
    solution = plane_graph.ComponentSolution(
        frames=("A", "B", "C"),
        quadrants={frame: 0 for frame in ("A", "B", "C")},
        translations={frame: (0.0, 0.0, 0.0) for frame in ("A", "B", "C")},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): 1.0
                           for reading in readings},
        reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches,
            support, phases, readings)


def _a3_reflected_triangle_case():
    """Physical q0/q1/q2 triangle with one repaired tangent reflection."""
    quadrants = {"A_q0": 0, "B_q1": 1, "C_q2": 2}
    normal_labels = {0: 2, 1: 0, 2: 2}
    sides = {"A_q0": 1.0, "B_q1": -1.0, "C_q2": -1.0}
    translations = {
        "A_q0": np.asarray((0.2, -0.35), dtype=np.float64),
        "B_q1": np.asarray((1.6, 0.45), dtype=np.float64),
        "C_q2": np.asarray((1.2, 0.45), dtype=np.float64),
    }
    readings = []
    signatures = []
    for frame, quadrant in quadrants.items():
        normal = normal_labels[quadrant]
        tangent_labels = tuple(label for label in range(3) if label != normal)
        basis = _physical_yaw(quadrant)[np.ix_((0, 1), tangent_labels)]
        residues = tuple(float((-value) % 1.0)
                         for value in basis.T @ translations[frame])
        reading = _reading(
            frame, normal, 0.5, 2.0, sides[frame], (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signatures.append(_signature(reading, 0.5, basis, translations[frame]))
    readings, signatures = tuple(readings), tuple(signatures)
    all_matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4)
    matches = tuple(
        next(match for match in all_matches
             if match.left == left.frame and match.right == right.frame
             and match.quadrant_of_right
                 == (quadrants[right.frame] - quadrants[left.frame]) % 4)
        for index, left in enumerate(readings)
        for right in readings[index + 1:])
    blocks = {_canonical_identity(readings, reading): 0.5 for reading in readings}
    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(quadrants)), quadrants=quadrants,
        translations={frame: (0.0, 0.0, 0.0) for frame in quadrants},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period=blocks, reason="", alternatives=(),
    )
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d_result = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
    assert p1d_result.status == "complete", p1d_result.refusals
    phases = tuple(_phase_for(match, readings) for match in matches)
    vector_result = plane_vector_domain.build_lateral_vector_domains(
        support, p1d_result, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=_large_budget())
    assert vector_result.status == "complete", vector_result.refusals
    return (vector_result, p1d_result, signatures, matches, support,
            phases, readings, quadrants, translations)


def test_public_solver_replays_a4_and_preserves_tree_ambiguity(monkeypatch):
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    def forbidden_candidate_at(_domain, _index):
        raise AssertionError("normal solving must not materialize an A4 candidate")
    monkeypatch.setattr(
        plane_vector_domain.MatchLateralVectorDomain,
        "candidate_at", forbidden_candidate_at)
    gc.collect()
    before = _working_set_bytes()
    started = time.perf_counter()
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    elapsed = time.perf_counter() - started
    after = _working_set_bytes()
    assert solved.status == "complete"
    assert solved.placement_readiness == "review_required"
    assert solved.connected
    assert len(solved.components) == 1
    assert solved.components[0].outcome == "ambiguous"
    assert solved.components[0].ambiguity_class_count_lower_bound == 2
    assert len(solved.components[0].translation_classes) == 2
    assert sum(domain.represented_candidate_count
               for domain in vector_result.domains) == 289
    assert solved.runtime_work == relative.C2RBRuntimeWork(
        candidate_views_scanned=289, partial_assignments=2,
        closure_component_checks=0, membership_probes=584,
        memoized_terms=289, provenance_witnesses=2,
        candidate_at_calls=0, certificates_retained=2)
    assert len(solved.to_json_bytes()) == 5194
    assert elapsed <= 5.0
    assert max(0, after - before) <= 32 * 1024 * 1024
    assert solved.runtime_work.candidate_at_calls == 0
    assert solved.runtime_work.candidate_views_scanned == (
        vector_result.domains[0].represented_candidate_count)
    assert relative.validate_relative_translation_graph(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def test_public_xyz_cycle_is_unique_and_ready_with_complete_domains():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _xyz_ready_case()[:7])
    assert tuple(domain.represented_candidate_count
                 for domain in vector_result.domains) == (81, 81, 81)
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert solved.status == "complete"
    assert solved.components[0].outcome == "unique_validated"
    assert solved.placement_readiness == "ready"
    assert solved.components[0].cycle_rank == 2
    assert solved.runtime_work.candidate_views_scanned > 0
    assert solved.runtime_work.candidate_at_calls == 0
    assert solved.runtime_work.membership_probes > 0
    certificate = solved.components[0].translation_classes[0]
    translations = dict(certificate.translations_dyadic_xyz)
    assert translations == {
        "A": ((0, 1), (0, 1), (0, 1)),
        "B": ((2, 1), (3, 1), (4, 1)),
    }
    assert certificate.assignment_provenance_multiplicity == 1
    for reference in certificate.edge_references:
        if reference.match_identity in solved.components[0].tree_edges:
            assert reference.tree_delta_class_multiplicity == 1
            assert reference.chord_satisfying_provenance_count is None
        else:
            assert reference.chord_satisfying_provenance_count == 1
            assert reference.tree_delta_class_multiplicity is None
    assert len(solved.to_json_bytes()) <= 8 * 1024 * 1024
    assert relative.validate_relative_translation_graph(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def test_public_a4_signed_normal_mutant_is_a_cycle_contradiction():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _public_normal_sign_mutant_cycle_case())
    normal_deltas = tuple(domain.normal_constraint.delta_translation
                          for domain in vector_result.domains)
    assert sorted(normal_deltas) == [-4.0, 0.0, 0.0]
    # Lateral A4 choices include zero on both tangent axes for every edge.  The
    # cycle fails only because the rebuilt, signed normal evidence has a -4
    # chord delta.
    for domain in vector_result.domains:
        lateral_axes = {factor.world_axis for factor in domain.axis_factors}
        assert any(all(relative._delta_xyz(domain, index)[axis] == 0.0
                       for axis in lateral_axes)
                   for index in range(domain.represented_candidate_count))
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert solved.status == "complete"
    assert solved.components[0].outcome == "contradiction"
    assert solved.components[0].cycle_rank == 1
    assert any(
        not (low <= (0, 1) <= high)
        for cycle in solved.components[0].cycle_residuals
        for low, high in (cycle.residual_bounds_dyadic_xyz[2],))
    assert solved.runtime_work.candidate_at_calls == 0
    assert relative.validate_relative_translation_graph(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def test_public_a4_lateral_sign_mutant_is_a_cycle_contradiction():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _public_lateral_sign_mutant_cycle_case())
    assert all(domain.normal_constraint.delta_translation == 0.0
               for domain in vector_result.domains)
    lateral_values = {value for domain in vector_result.domains
                      for factor in domain.axis_factors
                      for value in (*factor.left_side.candidate_values,
                                    *factor.right_side.candidate_values)}
    assert 0.0 in lateral_values
    assert any(abs(value - 0.5) <= 1e-12 for value in lateral_values)
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_membership_probes=1_000_000,
            max_partial_assignments=1_000_000,
            max_closure_component_checks=1_000_000))
    assert solved.status == "complete"
    component = solved.components[0]
    assert component.outcome == "contradiction"
    assert component.cycle_rank == 1
    assert all(cycle.matching_provenance_count == 0
               for cycle in component.cycle_residuals)
    assert solved.runtime_work.candidate_at_calls == 0
    # The normal component telescopes exactly; the sign-flipped lateral
    # half-period leaves no A4 candidate within tolerance.
    by_identity = {domain.match_identity: domain for domain in vector_result.domains}
    chord = component.failing_chords[0]
    required = tuple(Fraction(*axis) for axis in
                     component.cycle_residuals[0].required_chord_delta_dyadic_xyz)
    oracle = _independent_a4_delta_oracle(by_identity[chord])
    closest = min(oracle, key=lambda delta: max(
        abs(Fraction.from_float(delta[axis]) - required[axis])
        for axis in range(3)))
    mismatch = [abs(Fraction.from_float(closest[axis]) - required[axis])
                for axis in range(3)]
    assert mismatch[2] == 0
    assert any(mismatch[axis] > Fraction(1, 4) for axis in (0, 1))
    assert relative.validate_relative_translation_graph(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def _independent_a4_delta_oracle(domain):
    """Materialize only in tests, from factor tables and the written formula."""
    vectors = []
    per_entry = domain.represented_candidate_count // domain.joint_entry_count
    for index in range(domain.represented_candidate_count):
        entry_index = index // per_entry
        entry = domain.p1d_domain.entries[entry_index]
        provenance = domain.provenance_key_at(index)
        values = [None, None, None]
        for factor, item in zip(domain.axis_factors, provenance[1]):
            left_index, right_index = item[2], item[4]
            shift = (entry.column_shift
                     if factor.right_shift_source_component == "column"
                     else entry.row_shift)
            # This oracle deliberately spells out the contract's float order.
            lateral_delta = (factor.right_world_sign * shift
                             + factor.left_world_sign
                             * factor.left_side.candidate_values[left_index]
                             - factor.right_world_sign
                             * factor.right_side.candidate_values[right_index])
            values[factor.world_axis] = float(lateral_delta)
        normal = domain.normal_constraint
        values[normal.world_axis] = float(normal.delta_translation)
        assert all(type(value) is float and math.isfinite(value) for value in values)
        vectors.append(tuple(values))
    return tuple(vectors)


def test_factorized_chord_membership_matches_eager_cross_entry_oracle():
    vector_result, _p1d, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    domain = vector_result.domains[0]
    oracle = _independent_a4_delta_oracle(domain)
    assert len(oracle) == domain.represented_candidate_count
    exact_values = {}
    for index, delta in enumerate(oracle):
        exact = tuple(relative._dyadic_from_float(value) for value in delta)
        exact_values.setdefault(exact, []).append(index)
    grouped_runtime = relative._RuntimeCounter(_runtime_budget())
    grouped = relative._delta_classes(domain, grouped_runtime)
    assert len(grouped) == len(exact_values)
    for delta_class in grouped:
        provenance_indices = exact_values[delta_class.delta_dyadic_xyz]
        assert delta_class.multiplicity == len(provenance_indices)
        assert delta_class.witness_index == min(provenance_indices)
    assert grouped_runtime.snapshot().candidate_at_calls == 0
    assert grouped_runtime.snapshot().candidate_views_scanned == (
        domain.represented_candidate_count)
    for target, expected_indices in tuple(exact_values.items())[::max(1, len(exact_values) // 7)]:
        runtime = relative._RuntimeCounter(_runtime_budget())
        membership = relative._factorized_membership(
            domain, tuple(Fraction(*axis) for axis in target), Fraction(0), runtime)
        assert membership.count == len(expected_indices)
        assert membership.witness_index == min(expected_indices)
        assert domain.provenance_key_at(membership.witness_index)[0] == (
            domain.p1d_domain.entries[membership.witness_index
                                      // (len(oracle) // domain.joint_entry_count)].row_shift,
            domain.p1d_domain.entries[membership.witness_index
                                      // (len(oracle) // domain.joint_entry_count)].column_shift)

    # A two-entry diagonal table makes mixed row/column provenance absent.
    # This stays test-local; the production result remains fully replayed.
    entries = (domain.p1d_domain.entries[0], domain.p1d_domain.entries[-1])
    pair_counts = tuple(factor.left_side.candidate_count
                        * factor.right_side.candidate_count
                        for factor in domain.axis_factors)
    per_entry = pair_counts[0] * pair_counts[1]
    fake_domain = SimpleNamespace(
        normal_constraint=domain.normal_constraint,
        axis_factors=domain.axis_factors,
        p1d_domain=SimpleNamespace(entries=entries),
        joint_entry_count=2,
        represented_candidate_count=2 * per_entry,
    )
    fake_domain.iter_entry_groups = lambda: (
        (entries[0], 0, per_entry), (entries[1], per_entry, 2 * per_entry))
    fake_vectors = []
    vectors_by_entry = []
    for entry in entries:
        current = []
        factor_values = []
        for factor in domain.axis_factors:
            shift = (entry.column_shift
                     if factor.right_shift_source_component == "column"
                     else entry.row_shift)
            pair_values = []
            for left_value in factor.left_side.candidate_values:
                for right_value in factor.right_side.candidate_values:
                    pair_values.append(float(
                        factor.right_world_sign * shift
                        + factor.left_world_sign * left_value
                        - factor.right_world_sign * right_value))
            factor_values.append(pair_values)
        for first in factor_values[0]:
            for second in factor_values[1]:
                values = [None, None, None]
                values[domain.axis_factors[0].world_axis] = first
                values[domain.axis_factors[1].world_axis] = second
                values[domain.normal_constraint.world_axis] = float(
                    domain.normal_constraint.delta_translation)
                current.append(tuple(values))
        vectors_by_entry.append(tuple(current))
        fake_vectors.extend(current)
    cross_required = (vectors_by_entry[0][0][0],
                      vectors_by_entry[1][0][1],
                      float(domain.normal_constraint.delta_translation))
    assert cross_required not in set(fake_vectors)
    runtime = relative._RuntimeCounter(_runtime_budget())
    membership = relative._factorized_membership(
        fake_domain, tuple(Fraction.from_float(value) for value in cross_required),
        Fraction(0), runtime)
    assert membership.count == 0 and membership.witness_index is None
    assert runtime.snapshot().membership_probes > 0

    normal_mismatch = list(Fraction.from_float(value) for value in oracle[0])
    normal_mismatch[domain.normal_constraint.world_axis] += Fraction(1)
    runtime = relative._RuntimeCounter(_runtime_budget())
    membership = relative._factorized_membership(
        domain, tuple(normal_mismatch), Fraction(0), runtime)
    assert membership.count == 0 and membership.witness_index is None
    assert runtime.snapshot().membership_probes > 0


def test_membership_probe_counter_includes_factorized_presearch_bounds():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    domain = vector_result.domains[0]
    eligibility_runtime = relative._RuntimeCounter(_runtime_budget())
    assert relative._factorized_single_delta_class(domain, eligibility_runtime) is None
    eligibility_probes = eligibility_runtime.snapshot().membership_probes
    assert eligibility_probes > 0
    assert eligibility_runtime.snapshot().candidate_views_scanned == 0

    expected_bound_probes = 0
    for entry, _start, _stop in domain.iter_entry_groups():
        for factor in domain.axis_factors:
            left = tuple(dict.fromkeys((
                factor.left_side.candidate_values[0],
                factor.left_side.candidate_values[-1])))
            right = tuple(dict.fromkeys((
                factor.right_side.candidate_values[0],
                factor.right_side.candidate_values[-1])))
            expected_bound_probes += len(left) * len(right)
    bounds_runtime = relative._RuntimeCounter(_runtime_budget())
    relative._domain_bounds(domain, bounds_runtime)
    assert bounds_runtime.snapshot().membership_probes == expected_bound_probes

    # The same endpoint-expression count is charged during the public solve
    # before any A4 candidate view enters the general search.
    presearch_probes = eligibility_probes + expected_bound_probes
    assert presearch_probes > 1
    exhausted = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(max_membership_probes=presearch_probes - 1))
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_failure_reason == "runtime_budget_exceeded"
    assert exhausted.runtime_work.membership_probes == presearch_probes
    assert exhausted.runtime_work.candidate_views_scanned == 0
    assert exhausted.runtime_work.candidate_at_calls == 0
    assert exhausted.components == () and exhausted.refusals == ()


def test_three_edge_a4_cycle_is_canonical_permutation_invariant_and_atomic():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _a4_triangle_case())
    budget = _runtime_budget()
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0, budget=budget)
    assert solved.status == "complete"
    assert solved.connected
    assert solved.deterministic_work.edges == 3
    assert solved.deterministic_work.tree_edges == 2
    assert solved.deterministic_work.chords == 1
    assert solved.components[0].outcome in {"unique_validated", "ambiguous"}
    assert solved.components[0].cycle_rank == 1
    matches_identity_set = {domain.match_identity for domain in vector_result.domains}
    assert all(reference.match_identity in matches_identity_set
               for certificate in solved.components[0].translation_classes
               for reference in certificate.edge_references)
    by_identity = {domain.match_identity: domain for domain in vector_result.domains}
    tree_ids = set(solved.components[0].tree_edges)
    for certificate in solved.components[0].translation_classes:
        translations = {
            frame: tuple(Fraction(*value) for value in xyz)
            for frame, xyz in certificate.translations_dyadic_xyz
        }
        for reference in certificate.edge_references:
            domain = by_identity[reference.match_identity]
            oracle = _independent_a4_delta_oracle(domain)
            assert tuple(relative._dyadic_from_float(value)
                         for value in oracle[reference.global_index]) == (
                             reference.delta_dyadic_xyz)
            if reference.match_identity in tree_ids:
                count = sum(
                    tuple(relative._dyadic_from_float(value) for value in delta)
                    == reference.delta_dyadic_xyz for delta in oracle)
                assert reference.tree_delta_class_multiplicity == count
            else:
                required = tuple(
                    translations[reference.match_identity.right_frame][axis]
                    - translations[reference.match_identity.left_frame][axis]
                    for axis in range(3))
                count = sum(
                    all(abs(Fraction.from_float(delta[axis]) - required[axis]) == 0
                        for axis in range(3))
                    for delta in oracle)
                assert reference.chord_satisfying_provenance_count == count

    # The physical translations were declared before matching and residue
    # enumeration.  Replay each known edge delta through the complete factor
    # domain, then close that exact assignment around the third edge.
    physical = {
        "A_left": (0.0, 0.0, 0.0),
        "B_left": (2.0, 0.0, 0.0),
        "C_left": (0.0, 3.0, 0.0),
    }
    runtime = relative._RuntimeCounter(_runtime_budget())
    domain_by_identity = {domain.match_identity: domain
                          for domain in vector_result.domains}
    graph_plan = relative._make_graph_plan(tuple(domain_by_identity))
    component_plan = graph_plan.components[0]
    delta_classes = {}
    bounds = {}
    for identity, domain in domain_by_identity.items():
        target = tuple(physical[identity.right_frame][axis]
                       - physical[identity.left_frame][axis]
                       for axis in range(3))
        target_key = tuple(relative._dyadic_from_float(value) for value in target)
        classes = relative._delta_classes(domain, runtime)
        target_class = next(
            (value for value in classes if value.delta_dyadic_xyz == target_key), None)
        assert target_class is not None
        delta_classes[identity] = target_class
        bounds[identity] = relative._domain_bounds(domain, runtime)
    tree_assignment = {identity: delta_classes[identity]
                       for identity in component_plan.tree_edges}
    feasible, failed = relative._evaluate_tree_assignment(
        component_plan, domain_by_identity, tree_assignment, bounds,
        Fraction(0), runtime)
    assert not failed and feasible is not None
    assert feasible.translations == (
        ("A_left", ((0, 1), (0, 1), (0, 1))),
        ("B_left", ((2, 1), (0, 1), (0, 1))),
        ("C_left", ((0, 1), (3, 1), (0, 1))),
    )

    replayed = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, tuple(reversed(matches)),
        tuple(reversed(phases)), readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0, budget=budget)
    assert replayed.to_json_bytes() == solved.to_json_bytes()
    assert solved.runtime_work.candidate_at_calls == 0

    exhausted = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(max_partial_assignments=1))
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_failure_reason == "runtime_budget_exceeded"
    assert exhausted.components == () and exhausted.refusals == ()
    assert exhausted.runtime_work.partial_assignments == 2


def test_q0_and_repaired_reflection_truths_close_in_one_a4_cycle():
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings, quadrants, translations) = _a3_reflected_triangle_case()
    reading_by_frame = {reading.frame: reading for reading in readings}
    domain_by_identity = {domain.match_identity: domain
                          for domain in vector_result.domains}
    assert {identity.quadrant_of_right for identity in domain_by_identity} == {1, 2}
    truth_by_identity = {}
    tolerance = Fraction.from_float(1e-12)
    runtime = relative._RuntimeCounter(_runtime_budget(
        max_candidate_views_scanned=100_000,
        max_partial_assignments=100_000,
        max_closure_component_checks=100_000,
        max_membership_probes=1_000_000,
        max_memoized_terms=100_000,
    ))
    classes_by_identity = {}
    bounds = {}
    for identity, domain in domain_by_identity.items():
        left = reading_by_frame[identity.left_frame]
        right = reading_by_frame[identity.right_frame]
        left_normal_axis, left_normal_sign = plane_graph.axis_transform(
            left.axis, quadrants[identity.left_frame])
        right_normal_axis, right_normal_sign = plane_graph.axis_transform(
            right.axis, quadrants[identity.right_frame])
        assert left_normal_axis == right_normal_axis
        left_distance = left.matching_alternatives(identity.left_b)[0].distance_blocks
        right_distance = right.matching_alternatives(identity.right_b)[0].distance_blocks
        target = [None, None, None]
        target[left_normal_axis] = (
            left_normal_sign * left.side * left_distance
            - right_normal_sign * right.side * right_distance)
        target[0] = float(translations[identity.right_frame][0]
                          - translations[identity.left_frame][0])
        target[1] = float(translations[identity.right_frame][1]
                          - translations[identity.left_frame][1])
        target_tuple = tuple(float(value) for value in target)
        truth_by_identity[identity] = target_tuple
        oracle = _independent_a4_delta_oracle(domain)
        assert any(all(abs(candidate[axis] - target_tuple[axis]) <= 1e-12
                       for axis in range(3)) for candidate in oracle)
        classes = relative._delta_classes(domain, runtime)
        matching_class = next(
            (delta_class for delta_class in classes
             if all(abs(Fraction(*delta_class.delta_dyadic_xyz[axis])
                        - Fraction.from_float(target_tuple[axis])) <= tolerance
                    for axis in range(3))), None)
        assert matching_class is not None
        classes_by_identity[identity] = matching_class
        bounds[identity] = relative._domain_bounds(domain, runtime)

    plan = relative._make_graph_plan(tuple(domain_by_identity))
    component = plan.components[0]
    truth_tree_assignment = {
        identity: classes_by_identity[identity] for identity in component.tree_edges}
    feasible, failures = relative._evaluate_tree_assignment(
        component, domain_by_identity, truth_tree_assignment, bounds,
        tolerance, runtime)
    assert failures == {} and feasible is not None

    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=1e-12,
        budget=_runtime_budget(
            max_candidate_views_scanned=100_000,
            max_partial_assignments=100_000,
            max_closure_component_checks=100_000,
            max_membership_probes=1_000_000,
            max_memoized_terms=100_000,
        ))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.components[0].cycle_rank == 1
    assert solved.components[0].outcome in {"unique_validated", "ambiguous"}


def _budget_from_observed(result):
    deterministic = result.deterministic_work
    runtime = result.runtime_work
    fields = {
        "max_components": max(1, deterministic.components),
        "max_frames": max(1, deterministic.frames),
        "max_edges": max(1, deterministic.edges),
        "max_tree_edges": max(1, deterministic.tree_edges),
        "max_chords": max(1, deterministic.chords),
        "max_cycle_rows": max(1, deterministic.cycle_rows),
        "max_p1d_entries": max(1, deterministic.p1d_entries),
        "max_residue_values": max(1, deterministic.residue_values),
        "max_candidate_views_scanned": max(1, runtime.candidate_views_scanned),
        "max_partial_assignments": max(1, runtime.partial_assignments),
        "max_closure_component_checks": max(1, runtime.closure_component_checks),
        "max_membership_probes": max(1, runtime.membership_probes),
        "max_memoized_terms": max(1, runtime.memoized_terms),
        "max_provenance_witnesses": max(2, runtime.provenance_witnesses),
        "max_certificates": max(2, runtime.certificates_retained),
    }
    return relative.C2RBudget(**fields)


def test_exact_observed_budget_succeeds_and_one_below_is_atomic():
    sources = _a4_triangle_case()
    vector_result, p1d_result, signatures, matches, support, phases, readings = sources
    baseline = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert baseline.status == "complete"
    exact_budget = _budget_from_observed(baseline)
    exact = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=exact_budget)
    assert exact.status == "complete"
    assert exact.deterministic_work == baseline.deterministic_work
    assert exact.runtime_work == baseline.runtime_work
    assert "max_candidate_at_calls" not in exact_budget.to_record()
    with pytest.raises(ValueError, match="never materialize"):
        replace(baseline.runtime_work, candidate_at_calls=1)

    deterministic_map = {
        "components": "max_components", "frames": "max_frames",
        "edges": "max_edges", "tree_edges": "max_tree_edges",
        "chords": "max_chords", "cycle_rows": "max_cycle_rows",
        "p1d_entries": "max_p1d_entries",
        "residue_values": "max_residue_values",
    }
    for counter, cap in deterministic_map.items():
        value = getattr(baseline.deterministic_work, counter)
        if value > 1:
            limited = replace(exact_budget, **{cap: value - 1})
            response = relative.solve_relative_translation_graph(
                vector_result, support, p1d_result, matches, phases,
                readings=readings, signatures=signatures,
                comparison_tolerance=1e-12, closure_tolerance=0.0,
                budget=limited)
            assert response.status == "budget_exhausted", counter
            assert response.budget_failure_reason == "planned_budget_exceeded"
            assert response.runtime_work == relative.C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)
            assert response.components == () and response.refusals == ()
        else:
            with pytest.raises(ValueError):
                replace(exact_budget, **{cap: 0})

    runtime_map = {
        "candidate_views_scanned": "max_candidate_views_scanned",
        "partial_assignments": "max_partial_assignments",
        "closure_component_checks": "max_closure_component_checks",
        "membership_probes": "max_membership_probes",
        "memoized_terms": "max_memoized_terms",
        "provenance_witnesses": "max_provenance_witnesses",
        "certificates_retained": "max_certificates",
    }
    for counter, cap in runtime_map.items():
        value = getattr(baseline.runtime_work, counter)
        minimum = 2 if cap in {"max_provenance_witnesses", "max_certificates"} else 1
        if value > minimum:
            limited = replace(exact_budget, **{cap: value - 1})
            response = relative.solve_relative_translation_graph(
                vector_result, support, p1d_result, matches, phases,
                readings=readings, signatures=signatures,
                comparison_tolerance=1e-12, closure_tolerance=0.0,
                budget=limited)
            assert response.status == "budget_exhausted", counter
            assert response.budget_failure_reason == "runtime_budget_exceeded"
            assert response.components == () and response.refusals == ()
        elif cap in {"max_provenance_witnesses", "max_certificates"}:
            with pytest.raises(ValueError):
                replace(exact_budget, **{cap: minimum - 1})
        else:
            assert value == 0, counter
            assert getattr(exact_budget, cap) == 1


def test_nonnegative_tolerance_refusal_is_typed_and_canonical():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    refusal = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=-1.0, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert refusal.status == "refused"
    assert refusal.refusals[0].reason == "comparison_tolerance_mismatch"
    assert refusal.comparison_tolerance is None


@pytest.mark.parametrize(
    "comparison,closure,reason",
    [
        (float("nan"), 0.0, "comparison_tolerance_mismatch"),
        (1e-12, float("inf"), "closure_tolerance_invalid"),
        (1e-12, -0.25, "closure_tolerance_invalid"),
        (1e-12, True, "closure_tolerance_invalid"),
    ],
)
def test_malformed_tolerances_return_typed_atomic_refusals(
        comparison, closure, reason):
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    result = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=comparison, closure_tolerance=closure,
        budget=_runtime_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason == reason
    assert result.components == ()
    assert result.deterministic_work == relative.C2RBDeterministicWork(
        0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert result.runtime_work == relative.C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)


def test_result_and_nested_provenance_replacement_attacks_fail_closed():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _a4_triangle_case())
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    certificate = solved.components[0].translation_classes[0]
    reference = certificate.edge_references[0]
    with pytest.raises(ValueError):
        replace(solved, closure_tolerance=1.0)
    with pytest.raises(ValueError):
        replace(certificate, assignment_provenance_multiplicity=(
            certificate.assignment_provenance_multiplicity + 1))
    with pytest.raises(ValueError):
        replace(reference, global_index=reference.global_index + 1)
    forged = list(reference.delta_dyadic_xyz)
    forged[0] = (forged[0][0] * 2, forged[0][1] * 2)
    with pytest.raises(ValueError):
        replace(reference, delta_dyadic_xyz=tuple(forged))
    with pytest.raises(ValueError):
        replace(reference, shift_entry_rc=(
            reference.shift_entry_rc[0] + 1, reference.shift_entry_rc[1]))
    with pytest.raises(ValueError):
        replace(reference, provenance_key=((0, 0), ()))
    forged_factor_keys = list(reference.provenance_key[1])
    first_factor_key = list(forged_factor_keys[0])
    first_factor_key[1] = "0" * 64
    forged_factor_keys[0] = tuple(first_factor_key)
    with pytest.raises(ValueError):
        replace(reference, provenance_key=(reference.shift_entry_rc,
                                           tuple(forged_factor_keys)))
    with pytest.raises(ValueError):
        replace(reference, axis_worlds=(reference.axis_worlds[0],
                                        reference.axis_worlds[0]))
    with pytest.raises(ValueError):
        replace(reference, residue_candidate_counts=(0, *reference.residue_candidate_counts[1:]))
    forged_indices = list(reference.residue_indices)
    forged_indices[0] += 1
    with pytest.raises(ValueError):
        replace(reference, residue_indices=tuple(forged_indices))
    with pytest.raises(ValueError):
        replace(reference, p1d_entry_index=reference.p1d_entry_count)
    with pytest.raises(ValueError):
        replace(reference, p1d_max_shift_cells=0)
    with pytest.raises(ValueError):
        replace(reference, ref_sha256="0" * 64)
    cycle = solved.components[0].cycle_residuals[0]
    with pytest.raises(ValueError):
        replace(cycle, matching_provenance_count=cycle.matching_provenance_count + 1)
    with pytest.raises(ValueError):
        replace(solved.components[0], outcome="unique_validated")
    with pytest.raises(ValueError):
        replace(solved.components[0], component_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(solved, result_sha256="0" * 64)
    with pytest.raises(ValueError):
        relative._fraction((1, 3))


def test_public_solver_wrong_result_type_is_a_typed_refusal():
    result = relative.solve_relative_translation_graph(
        None, None, None, (), (), readings=(), signatures=(),
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason == "vector_result_invalid"
    assert result.components == ()


def test_public_solver_refusal_taxonomy_is_typed_and_atomic(monkeypatch):
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    empty = _unchecked_record_copy(vector_result, domains=())
    graph_empty = relative.solve_relative_translation_graph(
        empty, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert graph_empty.status == "refused"
    assert graph_empty.refusals[0].reason == "graph_empty"
    assert graph_empty.components == ()

    comparison_mismatch = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=2e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert comparison_mismatch.status == "refused"
    assert comparison_mismatch.refusals[0].reason == "comparison_tolerance_mismatch"

    monkeypatch.setattr(
        plane_vector_domain, "validate_lateral_vector_domain_inputs",
        lambda *args, **kwargs: True)
    domain = vector_result.domains[0]
    duplicate = _unchecked_record_copy(vector_result, domains=(domain, domain))
    duplicate_result = relative.solve_relative_translation_graph(
        duplicate, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert duplicate_result.status == "refused"
    assert duplicate_result.refusals[0].reason == "duplicate_match_identity"
    assert duplicate_result.components == ()

    invalid_identity = _unchecked_record_copy(
        domain.match_identity, left_frame="")
    invalid_domain = _unchecked_record_copy(domain, match_identity=invalid_identity)
    invalid_endpoints = _unchecked_record_copy(vector_result, domains=(invalid_domain,))
    endpoint_result = relative.solve_relative_translation_graph(
        invalid_endpoints, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert endpoint_result.status == "refused"
    assert endpoint_result.refusals[0].reason == "identity_endpoint_invalid"

    def malformed_domain(identity, vector):
        return SimpleNamespace(
            match_identity=identity,
            normal_constraint=SimpleNamespace(world_axis=2, delta_translation=0.0),
            vector_at=lambda _index: vector)

    for vector, reason in (
            (((0, float("nan")), (1, 0.0)), "nonfinite_delta"),
            (((0, 0.0), (0, 1.0)), "factor_contract_invalid")):
        malformed = _unchecked_record_copy(
            vector_result,
            domains=(malformed_domain(domain.match_identity, vector),))
        refused = relative.solve_relative_translation_graph(
            malformed, support, p1d_result, matches, phases,
            readings=readings, signatures=signatures,
            comparison_tolerance=1e-12, closure_tolerance=0.0,
            budget=_runtime_budget())
        assert refused.status == "refused"
        assert refused.refusals[0].reason == reason
        assert refused.components == ()


def test_changed_replay_sources_and_short_a4_iterator_fail_closed(monkeypatch):
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _build_vector_case())
    changed_sources = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases[:-1],
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert changed_sources.status == "refused"
    assert changed_sources.refusals[0].reason == "vector_result_replay_mismatch"
    assert changed_sources.components == ()

    original_iter_views = plane_vector_domain.MatchLateralVectorDomain.iter_views

    def short_iterator(domain, start, stop):
        yield from original_iter_views(domain, start, max(start, stop - 1))

    monkeypatch.setattr(
        plane_vector_domain.MatchLateralVectorDomain, "iter_views", short_iterator)
    mismatch = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert mismatch.status == "budget_exhausted"
    assert mismatch.budget_failure_reason == "runtime_count_mismatch"
    assert mismatch.components == () and mismatch.refusals == ()


def test_represented_multi_numeric_singleton_is_unreachable_under_a4_invariants():
    vector_result, _p1d, _signatures, _matches, _support, _phases, _readings = (
        _multi_residue_case())
    multi = vector_result.domains[0]
    single_pair = _build_vector_case()[0].domains[0]
    assert multi.represented_candidate_count > 1
    assert single_pair.represented_candidate_count > 1
    # Strictly increasing residue tables and unique row-major P1-D entries are
    # the A4 invariants that make a multi-represented numeric singleton
    # unreachable.  No impossible singleton fixture is fabricated.
    for domain in (multi, single_pair):
        for factor in domain.axis_factors:
            for values in (factor.left_side.candidate_values,
                           factor.right_side.candidate_values):
                assert all(left < right for left, right in zip(values, values[1:]))
        entries = [(entry.row_shift, entry.column_shift)
                   for entry in domain.p1d_domain.entries]
        assert len(set(entries)) == len(entries)
    side = multi.axis_factors[0].left_side
    with pytest.raises(ValueError):
        replace(side, candidate_values=(side.candidate_values[0],) * 2,
                candidate_count=2)
    for domain in (multi, single_pair):
        runtime = relative._RuntimeCounter(_runtime_budget())
        assert relative._factorized_single_delta_class(domain, runtime) is None
        assert runtime.snapshot().candidate_views_scanned == 0
        assert runtime.snapshot().membership_probes > 0
        classes = relative._delta_classes(
            domain, relative._RuntimeCounter(_runtime_budget()))
        assert len(classes) > 1


def test_zero_materialization_is_falsifiable_through_public_paths(monkeypatch):
    ambiguous_sources = _build_vector_case()
    unique_sources = _xyz_ready_case()[:7]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the solver must not materialize or select A4 candidates")

    monkeypatch.setattr(
        plane_vector_domain.MatchLateralVectorDomain, "candidate_at", forbidden)
    monkeypatch.setattr(
        plane_vector_domain.MatchLateralVectorDomain, "iter_candidates", forbidden)
    monkeypatch.setattr(
        plane_vector_domain, "materialize_lateral_vector_domains", forbidden)
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)

    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        ambiguous_sources)
    ambiguous = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert ambiguous.status == "complete"
    assert ambiguous.components[0].outcome == "ambiguous"
    assert ambiguous.runtime_work.candidate_at_calls == 0

    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        unique_sources)
    unique = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert unique.status == "complete"
    assert unique.components[0].outcome == "unique_validated"
    assert unique.runtime_work.membership_probes > 0
    assert unique.runtime_work.candidate_at_calls == 0


def test_component_and_result_validation_binds_translation_and_closure_proofs():
    ambiguous_sources = _build_vector_case()
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        ambiguous_sources)
    ambiguous = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    ambiguous_component = ambiguous.components[0]
    assert ambiguous_component.outcome == "ambiguous"
    first, second = ambiguous_component.translation_classes
    assert first.translation_key != second.translation_key
    duplicate = _unchecked_record_copy(
        ambiguous_component, translation_classes=(first, first))
    with pytest.raises(ValueError):
        relative.C2RBComponentResult.__post_init__(duplicate)

    unique_sources = _xyz_ready_case()[:7]
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        unique_sources)
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    cycle = component.cycle_residuals[0]
    assert cycle.matching_provenance_count > 0
    zero_provenance = _unchecked_record_copy(cycle, matching_provenance_count=0)
    with pytest.raises(ValueError):
        relative.C2RBComponentResult.__post_init__(_unchecked_record_copy(
            component, cycle_residuals=(zero_provenance,)))

    widened = ((1, 1), (2, 1))
    out_of_band = tuple(_unchecked_record_copy(
        value, residual_bounds_dyadic_xyz=(widened, widened, widened))
        for value in component.cycle_residuals)
    resealed = relative._make_component_result({
        "frames": component.frames,
        "gauge_anchor": component.gauge_anchor,
        "tree_edges": component.tree_edges,
        "chords": component.chords,
        "cycle_rank": component.cycle_rank,
        "outcome": component.outcome,
        "translation_classes": component.translation_classes,
        "cycle_residuals": out_of_band,
        "failing_chords": component.failing_chords,
        "ambiguity_class_count_lower_bound":
            component.ambiguity_class_count_lower_bound,
        "exhaustive": component.exhaustive,
    })
    result_fields = {
        "status": "complete", "budget": solved.budget,
        "deterministic_work": solved.deterministic_work,
        "runtime_work": solved.runtime_work,
        "vector_result_sha256": solved.vector_result_sha256,
        "vector_input_sha256": solved.vector_input_sha256,
        "comparison_tolerance": solved.comparison_tolerance,
        "closure_tolerance": solved.closure_tolerance,
    }
    with pytest.raises(ValueError):
        relative._make_result(components=(resealed,), **result_fields)
    control = relative._make_result(components=(component,), **result_fields)
    assert control.status == "complete"
    assert control.to_json_bytes() == solved.to_json_bytes()


def test_public_xyz_budget_boundary_is_exact_and_atomic():
    vector_result, p1d_result, signatures, matches, support, phases, readings = (
        _xyz_ready_case()[:7])
    baseline = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget())
    assert baseline.status == "complete"
    assert baseline.runtime_work.candidate_views_scanned > 0
    assert baseline.runtime_work.membership_probes > 0
    exact_budget = _budget_from_observed(baseline)
    assert exact_budget.max_candidate_views_scanned == \
        baseline.runtime_work.candidate_views_scanned
    exact = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=exact_budget)
    assert exact.status == "complete"
    assert exact.runtime_work == baseline.runtime_work
    one_below = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=replace(
            exact_budget,
            max_membership_probes=baseline.runtime_work.membership_probes - 1))
    assert one_below.status == "budget_exhausted"
    assert one_below.budget_failure_reason == "runtime_budget_exceeded"
    assert one_below.runtime_work.membership_probes == \
        baseline.runtime_work.membership_probes
    assert one_below.runtime_work.candidate_at_calls == 0
    assert one_below.components == () and one_below.refusals == ()


def test_refused_results_reject_any_deterministic_work():
    refusal = relative.C2RBRefusal("vector_result_invalid", "forged refusal")
    zero_work = relative.C2RBDeterministicWork(0, 0, 0, 0, 0, 0, 0, 0, 0)
    zero_runtime = relative.C2RBRuntimeWork(0, 0, 0, 0, 0, 0, 0, 0)
    fields = {
        "status": "refused", "budget": _runtime_budget(),
        "runtime_work": zero_runtime,
        "vector_result_sha256": "0" * 64, "vector_input_sha256": "0" * 64,
        "comparison_tolerance": 1e-12, "closure_tolerance": 0.0,
        "refusals": (refusal,),
    }
    accepted = relative._make_result(deterministic_work=zero_work, **fields)
    assert accepted.status == "refused"
    assert accepted.deterministic_work == zero_work
    with pytest.raises(ValueError):
        relative._make_result(
            deterministic_work=relative.C2RBDeterministicWork(
                1, 2, 1, 1, 0, 0, 0, 0, 0),
            **fields)


def test_solver_module_imports_and_calls_stay_inside_the_isolation_boundary():
    source = Path(inspect.getsourcefile(relative)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_imports = {
        "__future__", "dataclasses", "collections", "fractions", "hashlib",
        "json", "math", "typing",
        "mc_photogrammetry.reconstruction",
    }
    forbidden_module_prefixes = (
        "mc_photogrammetry.camera", "mc_photogrammetry.dataset",
        "mc_photogrammetry.evaluation", "mc_photogrammetry.renderer",
        "mc_photogrammetry.selector", "mc_photogrammetry.solver",
        "mc_photogrammetry.placement", "mc_photogrammetry.export",
        "mc_photogrammetry.cli", "numpy", "PIL", "cv2", "pathlib",
        "os", "socket", "subprocess", "urllib", "requests",
    )
    forbidden_attributes = {
        "candidate_at", "iter_candidates", "materialize",
        "materialize_lateral_vector_domains", "solve_component",
        "place_cells", "write_world", "export_world",
    }
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
            assert all(alias.name in allowed_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
            assert node.module in allowed_imports
            if node.module == "mc_photogrammetry.reconstruction":
                assert {alias.name for alias in node.names} <= {
                    "plane_axis_support", "plane_graph", "plane_residue",
                    "plane_shift_domain", "plane_vector_domain",
                }
        elif isinstance(node, ast.Call):
            call_name = (node.func.id if isinstance(node.func, ast.Name)
                         else node.func.attr if isinstance(node.func, ast.Attribute)
                         else "")
            assert call_name not in {
                "open", "exec", "eval", "compile", "__import__",
                "import_module", "system", "popen", "run", "connect",
                "urlopen", "request", "unlink", "write_text",
                "candidate_at", "iter_candidates", "materialize",
                "materialize_lateral_vector_domains", "solve_component",
            }
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
            if isinstance(node.value, ast.Name):
                if node.value.id == "plane_graph":
                    assert node.attr in {"PlaneMatch", "axis_transform"}
                elif node.value.id in {
                    "plane_axis_support", "plane_residue",
                    "plane_shift_domain", "plane_vector_domain",
                }:
                    assert not node.attr.startswith("_")
    assert all(not any(module == prefix or module.startswith(prefix + ".")
                       for prefix in forbidden_module_prefixes)
               for module in imported_modules)
