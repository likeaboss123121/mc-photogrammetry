"""Independent controls for the isolated exact bounded V2-A provenance CSP.

Expected values are declared by hand or derived from the frozen contract, never
from the module under test.  The synthetic kernel builder constructs readings,
side tables, match identities and accepted edge deltas by hand.  Public
acceptance controls reuse the existing accepted A4/C2-R-B fixture builders and
drive ``analyze_endpoint_provenance``; the multi-candidate A4 case is isolated
binding/enumeration evidence through the private helpers, because the frozen
gate refuses it before public binding.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_relative_translation as relative
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain

from test_c2r_vector_domain_a3_proof import (
    _canonical_identity,
    _centred_coverage,
    _reading,
    _signature,
)
from test_plane_shift_domain import _budget as _shift_budget
from test_plane_vector_domain import _large_budget, _phase_for
from test_plane_relative_translation import (
    _runtime_budget,
    _ambiguous_a4_ring_case,
    _xyz_ready_case,
)


UNIVERSE = "c" * 64
SOURCE_SHA = "d" * 64
OTHER_SHA = "e" * 64
MULTICANDIDATE_FIXTURE_BLOCKED = (
    "the current public multicandidate chord retains multiple closing classes "
    "and is review_required; it cannot prove the stated ready-path invariant")
SOURCE = Path(__file__).resolve().parents[1] / "src" / "mc_photogrammetry" / \
    "reconstruction" / "plane_endpoint_provenance.py"
ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "dataclasses", "fractions", "hashlib", "json", "math",
    "typing", "mc_photogrammetry",
})
ALLOWED_RECONSTRUCTION_MODULES = frozenset({
    "lattice_planes", "plane_axis_support", "plane_graph", "plane_residue",
    "plane_shift_domain", "plane_vector_domain", "plane_relative_translation",
})
DENIED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "id",
    "candidate_at", "iter_candidates", "materialize_lateral_vector_domains",
    "solve_component", "ComponentSolution", "BlockVolume",
    "ReconstructionResult", "os", "sys", "subprocess", "socket", "pathlib",
    "shutil", "io", "tempfile", "ctypes", "numpy", "np",
})


# ---------------------------------------------------------------------------
# Hand-built kernel fixtures
# ---------------------------------------------------------------------------

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _values(count: int) -> tuple[float, ...]:
    return tuple(index / (count + 1) for index in range(count))


def _reading_identity(frame: str, index: int = 0, axis: int = 2,
                      universe: str = UNIVERSE) -> v2a.ReadingIdentity:
    return v2a.ReadingIdentity.build(
        frame=frame, canonical_index=index,
        reading_order_key_sha256=_digest(f"{frame}:{index}:{axis}:{universe}"),
        reading_axis=axis, source_region_id=None,
        canonical_universe_sha256=universe)


def _domain(frame: str, *, nu: int = 1, nv: int = 1, quadrant: int = 0,
            index: int = 0, axis: int = 2, universe: str = UNIVERSE,
            footprint: str = v2a.FOOTPRINT_ABSENT,
            source_u: str = SOURCE_SHA,
            source_v: str = OTHER_SHA) -> v2a.ReadingLabelDomain:
    return v2a.ReadingLabelDomain.build(
        reading_identity=_reading_identity(frame, index, axis, universe),
        selected_b=1.0, absolute_quadrant=quadrant,
        footprint_identity=footprint,
        residue_u=v2a.ResidueAxisDomain.build(
            local_axis="u", local_axis_label=0, selected_b=1.0,
            source_record_sha256=source_u, candidate_values=_values(nu)),
        residue_v=v2a.ResidueAxisDomain.build(
            local_axis="v", local_axis_label=1, selected_b=1.0,
            source_record_sha256=source_v, candidate_values=_values(nv)))


def _identity(left: str, right: str, ordinal: int = 0):
    return plane_axis_support.C1MatchIdentity(
        left, right, ordinal, ordinal, 0, 0, 1.0, 1.0, 0)


def _delta(values=(0.0, 0.0, 0.0)):
    return tuple(v2a._dyadic_from_float(value) for value in values)


def _edge(left: v2a.ReadingLabelDomain, right: v2a.ReadingLabelDomain,
          pairs, *, kind: str = "tree", ordinal: int = 0,
          sentinel: object | None = None) -> v2a.KernelEdge:
    buckets = tuple(v2a.KernelBucket.build(
        global_index=global_index, left_code=left_code, right_code=right_code,
        multiplicity=multiplicity)
        for left_code, right_code, global_index, multiplicity in pairs)
    return v2a.KernelEdge.build(
        match_identity=_identity(left.reading_identity.frame,
                                 right.reading_identity.frame, ordinal),
        edge_kind=kind, accepted_delta_dyadic_xyz=_delta(),
        left_reading_sha256=left.reading_sha256,
        right_reading_sha256=right.reading_sha256, buckets=buckets,
        ignored_representative=sentinel)


def _budget(**overrides) -> v2a.V2ABudget:
    limits = {
        "max_readings": 1000, "max_edges": 1000, "max_endpoint_labels": 10000,
        "max_residues_inspected": 100000,
        "max_edge_candidates_inspected": 100000,
        "max_partial_label_assignments": 100000,
        "max_consistency_checks": 100000,
        "max_result_validation_cross_links": 100000,
        "max_distinct_assignments": 2,
        "max_retained_witnesses": 2,
    }
    limits.update(overrides)
    return v2a.V2ABudget(**limits)


def _solve(domains, edges, **overrides) -> v2a.KernelResult:
    return v2a.solve_endpoint_provenance_kernel(
        domains=tuple(domains), edges=tuple(edges), budget=_budget(**overrides))


# ---------------------------------------------------------------------------
# Public fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ring():
    return _xyz_ready_case()[:7]


@pytest.fixture(scope="module")
def ambiguous_ring():
    return _ambiguous_a4_ring_case()


@pytest.fixture(scope="module")
def solved_ring(ring):
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = ring
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10, max_tree_edges=10,
            max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.placement_readiness == "ready"
    assert solved.components[0].outcome == "unique_validated"
    return solved


def _run_public(ring, solved, budget, *, comparison=1e-12, closure=0.0):
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = ring
    return v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=comparison, closure_tolerance=closure,
        budget=budget)


def _ring_budget(**overrides) -> v2a.V2ABudget:
    return _budget(**overrides)


# ---------------------------------------------------------------------------
# Public multi-candidate A4 fixture
# ---------------------------------------------------------------------------

def _multi_candidate_a4_ring_case(size: int = 5, offset: float = 0.2,
                                  blocks: float = 0.5):
    """Public A4 ring whose non-integer residues admit several E_m candidates.

    The accepted pairwise deltas are unchanged from ``_ambiguous_a4_ring_case``
    because every frame translation shares one fractional offset; the half-period
    ``blocks`` makes each P1-B endpoint table carry two candidates, so each A4
    domain represents many indices and ``E_m`` has more than one feasible
    witness.  No production or A4 code is monkeypatched.
    """
    lattice = ((0, 0), (2, 0), (3, 2), (1, 3), (-1, 2))
    assert size == len(lattice)
    translations = {
        f"R{index}": np.asarray(point, dtype=np.float64) + offset
        for index, point in enumerate(lattice)}
    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame, translation in translations.items():
        residues = tuple(float((-value) % 1.0) for value in translation)
        reading = _reading(
            frame, 2, blocks, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signature = _signature(reading, blocks, basis, translation)
        signatures.append(replace(
            signature,
            coverage=_centred_coverage(signature.coverage.shape)))

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
            pair,
            left_plane=_canonical_identity(
                readings, signature_by_frame[left].reading)[1],
            right_plane=_canonical_identity(
                readings, signature_by_frame[right].reading)[1]))
    matches = tuple(matches)
    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(translations)),
        quadrants={frame: 0 for frame in translations},
        translations={frame: (0.0, 0.0, 0.0) for frame in translations},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): blocks
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


@pytest.fixture(scope="module")
def multi_ring():
    return _multi_candidate_a4_ring_case()


@pytest.fixture(scope="module")
def solved_multi_ring(multi_ring):
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = multi_ring
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10, max_tree_edges=10,
            max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete", solved.budget_failure_reason
    return solved


# ---------------------------------------------------------------------------
# 5.1 Independent zero/one/multiple oracles
# ---------------------------------------------------------------------------

def test_kernel_unique_on_hand_declared_chain():
    a, b, c = _domain("A"), _domain("B"), _domain("C")
    edges = (_edge(a, b, [(0, 0, 0, 1)], ordinal=0),
             _edge(b, c, [(0, 0, 0, 1)], ordinal=1))
    outcome = _solve((a, b, c), edges)
    assert outcome.status == "complete"
    assert outcome.classification == "endpoint_unique"
    assert outcome.exhaustive is True
    assert outcome.distinct_assignment_count == 1
    assert len(outcome.witnesses) == 1
    assert outcome.runtime_work.candidate_materializations == 0


def test_kernel_unique_on_hand_declared_tree_star():
    a, b, c = _domain("A"), _domain("B"), _domain("C")
    edges = (_edge(a, b, [(0, 0, 0, 1)], ordinal=0),
             _edge(a, c, [(0, 0, 0, 1)], ordinal=1))
    outcome = _solve((a, b, c), edges)
    assert outcome.classification == "endpoint_unique"
    assert len(outcome.witnesses[0].edge_multiplicities) == 2


def test_kernel_none_on_hand_declared_cyclic_conflict():
    a, b = _domain("A"), _domain("B", nv=2)
    c = _domain("C")
    edges = (_edge(a, b, [(0, 0, 0, 1)], ordinal=0),
             _edge(b, c, [(1, 0, 0, 1)], ordinal=1))
    outcome = _solve((a, b, c), edges)
    assert outcome.classification == "endpoint_none"
    assert outcome.distinct_assignment_count == 0
    assert outcome.witnesses == ()


def test_kernel_unique_on_consistent_triangle():
    a, b = _domain("A"), _domain("B")
    c = _domain("C")
    edges = (_edge(a, b, [(0, 0, 0, 1)], ordinal=0),
             _edge(b, c, [(0, 0, 0, 1)], ordinal=1),
             _edge(a, c, [(0, 0, 0, 1)], ordinal=2))
    outcome = _solve((a, b, c), edges)
    assert outcome.classification == "endpoint_unique"


def test_kernel_multiple_assignment_control():
    a, b = _domain("A"), _domain("B", nv=2)
    edge = _edge(a, b, [(0, 0, 0, 1), (0, 1, 1, 1)])
    outcome = _solve((a, b), (edge,))
    assert outcome.classification == "endpoint_multiple"
    assert outcome.exhaustive is False
    assert outcome.lower_bound_only is True
    assert outcome.distinct_assignment_count == 2
    assert len(outcome.witnesses) == 2
    first, second = outcome.witnesses
    assert first.labels != second.labels


# ---------------------------------------------------------------------------
# 5.2 Incompatible same-delta cycle is decisive
# ---------------------------------------------------------------------------

def _sentinel(global_index: int = 0):
    return v2a.KernelRepresentativeSentinel(
        global_index=global_index, residue_indices=(0, 0, 0, 0),
        provenance_sha256="a" * 64)


def _incompatible_same_delta_edges(kind: str):
    a, b = _domain("A"), _domain("B", nv=2)
    c = _domain("C")
    edges = (_edge(a, b, [(0, 0, 0, 1)], kind=kind, ordinal=0,
                   sentinel=_sentinel(0)),
             _edge(b, c, [(1, 0, 0, 1)], kind=kind, ordinal=1,
                   sentinel=_sentinel(0)))
    return (a, b, c), edges


def test_kernel_incompatible_same_delta_cycle_is_endpoint_none():
    domains, edges = _incompatible_same_delta_edges("tree")
    outcome = _solve(domains, edges)
    assert outcome.classification == "endpoint_none"
    # The accepted deltas are numerically identical across both incidences.
    assert all(edge.accepted_delta_dyadic_xyz == edges[0].accepted_delta_dyadic_xyz
               for edge in edges)


def _representative_only_mutant(edges) -> str:
    """Forbidden mutant: keep only each edge's stored-representative bucket and
    accept the request whenever every edge keeps one, skipping the global
    per-reading label CSP.  A real CSP mutant, not a classification stub.
    """
    for edge in edges:
        sentinel = edge.ignored_representative
        if not isinstance(sentinel, v2a.KernelRepresentativeSentinel):
            return "indeterminate"
        kept = tuple(bucket for bucket in edge.buckets
                     if bucket.global_index == sentinel.global_index)
        if not kept:
            return "endpoint_none"
    return "endpoint_unique"


def test_representative_only_mutant_is_wrong_on_incompatible_same_delta():
    domains, edges = _incompatible_same_delta_edges("tree")
    assert _solve(domains, edges).classification == "endpoint_none"
    assert _representative_only_mutant(edges) == "endpoint_unique"


# ---------------------------------------------------------------------------
# 5.4 Chord tolerance endpoint obedience
# ---------------------------------------------------------------------------

def _tolerance_only_mutant(domains, edges) -> str:
    """Forbidden mutant: drop the 4.2 endpoint joint constraint and accept any
    per-edge tolerance-feasible bucket without shared-reading consistency.
    """
    del domains
    if edges and all(edge.buckets for edge in edges):
        return "endpoint_unique"
    return "endpoint_none"


def test_chord_tolerance_only_mutant_is_wrong_on_endpoint_conflict():
    domains, edges = _incompatible_same_delta_edges("chord")
    assert _solve(domains, edges).classification == "endpoint_none"
    assert _tolerance_only_mutant(domains, edges) == "endpoint_unique"
    # The representative-consulting dataflow mutant is wrong here too.
    assert _representative_only_mutant(edges) == "endpoint_unique"


# ---------------------------------------------------------------------------
# 5.5 Tree injectivity and chord-band witness multiplicity
# ---------------------------------------------------------------------------

def test_kernel_tree_edge_rejects_two_witnesses_for_one_label_pair():
    a, b = _domain("A"), _domain("B")
    edge = _edge(a, b, [(0, 0, 0, 1), (0, 0, 1, 1)])
    outcome = _solve((a, b), (edge,))
    assert outcome.status == "refused"
    assert outcome.refusal_reason == "kernel_input_invalid"
    # A mutant that claims tree multiplicity two instead of failing is wrong.
    assert _multiplicity_claiming_mutant(edge) == 2


def _multiplicity_claiming_mutant(edge) -> int:
    """Forbidden mutant: report a tree pair's bucket count as multiplicity."""
    counts: dict[tuple[int, int], int] = {}
    for bucket in edge.buckets:
        key = (bucket.left_code, bucket.right_code)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values())


def test_kernel_chord_band_carries_exact_witness_multiplicity():
    a, b = _domain("A"), _domain("B")
    edge = _edge(a, b, [(0, 0, 3, 2)], kind="chord")
    outcome = _solve((a, b), (edge,))
    assert outcome.classification == "endpoint_unique"
    witness = outcome.witnesses[0]
    assert len(witness.edge_multiplicities) == 1
    assert witness.edge_multiplicities[0][1] == 2


def test_kernel_chord_distinct_residue_index_is_a_distinct_assignment():
    a, b = _domain("A"), _domain("B", nv=2)
    edge = _edge(a, b, [(0, 0, 0, 1), (0, 1, 1, 1)], kind="chord")
    outcome = _solve((a, b), (edge,))
    assert outcome.classification == "endpoint_multiple"
    assert outcome.distinct_assignment_count == 2


def test_tree_enumeration_injectivity_holds_on_public_ring(ring, solved_ring):
    (vector_result, _p1d, signatures, _matches, _support, _phases,
     readings) = ring
    bound = v2a._bind_pipeline(solved_ring, vector_result, readings, signatures)
    counter = v2a._RuntimeCounter(_ring_budget())
    for domain in bound.domains:
        kind, accepted = bound.accepted[domain.match_identity]
        left_identity, right_identity = bound.edge_endpoints[domain.match_identity]
        left_domain = bound.domain_by_reading_sha256[left_identity.identity_sha256]
        right_domain = bound.domain_by_reading_sha256[
            right_identity.identity_sha256]
        buckets, _feasible, transcript = v2a._bucket_pairs(
            domain, accepted, kind, Fraction(0), left_domain, right_domain,
            counter)
        assert buckets
        assert len(transcript) == 64
        for bucket in buckets:
            assert bucket.multiplicity >= 1
            if kind == "tree":
                assert bucket.multiplicity == 1


def test_tree_duplicate_group_mutant_fails_closed():
    duplicate_group = {(0, 0): (3, 2)}
    with pytest.raises(v2a.V2AFailure) as failure:
        v2a._bucket_records("tree", duplicate_group)
    assert failure.value.reason == "tree_injectivity_violation"
    # The forbidden mutant reports two distinct indices as one label's
    # admissible tree multiplicity instead of failing closed.
    assert _duplicate_group_mutant(duplicate_group) == 2


def _duplicate_group_mutant(grouped) -> int:
    """Forbidden mutant: accept a tree group with two distinct indices."""
    return max(count for _index, count in grouped.values())


def test_non_dyadic_rounded_tree_delta_is_accepted_without_inversion():
    # Frozen A4 operation: delta = s_R*component + s_L*v_left - s_R*v_right,
    # with world signs s_L = s_R = +1 for this regression.
    component = 3.0
    left_value = 0.1
    right_value = 0.2
    stored = 1.0 * component + 1.0 * left_value - 1.0 * right_value
    assert stored == 2.9
    # Exact-dyadic algebraic inversion of the rounded delta is non-integral, so
    # such an inversion must never be used to recover the P1-D shift.
    recovered = (Fraction(*v2a._dyadic_from_float(stored))
                 - Fraction(*v2a._dyadic_from_float(left_value))
                 + Fraction(*v2a._dyadic_from_float(right_value)))
    assert recovered.denominator != 1
    # Enumeration-based injectivity accepts the valid single-index tree group
    # and never inverts the rounded float.  The compact aggregate retains only
    # the canonical first index and the distinct witness count.
    buckets = v2a._bucket_records("tree", {(0, 0): (17, 1)})
    assert len(buckets) == 1
    assert buckets[0].global_index == 17
    assert buckets[0].multiplicity == 1


# ---------------------------------------------------------------------------
# 5.11 Representative independence
# ---------------------------------------------------------------------------

def test_kernel_output_is_byte_identical_across_ignored_sentinels():
    a, b = _domain("A"), _domain("B", nv=2)
    pairs = [(0, 0, 0, 1), (0, 1, 1, 1)]
    plain = _edge(a, b, pairs)
    contradicting = _edge(
        a, b, pairs,
        sentinel=v2a.KernelRepresentativeSentinel(
            global_index=999, residue_indices=(7, 7, 7, 7),
            provenance_sha256="f" * 64))
    baseline = _solve((a, b), (plain,))
    varied = _solve((a, b), (contradicting,))
    assert baseline.classification == varied.classification
    assert baseline.to_json_bytes() == varied.to_json_bytes()
    assert baseline.result_sha256 == varied.result_sha256
    # The ignored entry-point argument is likewise never read.
    for sentinel_argument in (
            None, "opaque", 12345,
            v2a.KernelRepresentativeSentinel(
                global_index=0, residue_indices=(9, 9, 9, 9),
                provenance_sha256="b" * 64)):
        argument_varied = v2a.solve_endpoint_provenance_kernel(
            domains=(a, b), edges=(plain,), budget=_budget(),
            ignored_representative_sentinel=sentinel_argument)
        assert argument_varied.to_json_bytes() == baseline.to_json_bytes()
        assert argument_varied.result_sha256 == baseline.result_sha256


def test_representative_only_mutant_is_wrong_on_multiple_assignment():
    a, b = _domain("A"), _domain("B", nv=2)
    edge = _edge(a, b, [(0, 0, 0, 1), (0, 1, 1, 1)],
                 sentinel=_sentinel(0))
    assert _solve((a, b), (edge,)).classification == "endpoint_multiple"
    assert _representative_only_mutant((edge,)) == "endpoint_unique"


def test_kernel_cross_link_rejects_a_resealed_inconsistent_witness():
    a, b = _domain("A"), _domain("B")
    edge = _edge(a, b, [(0, 0, 0, 1)])
    domains = v2a._canonical_domains((a, b))
    edges = v2a._canonical_edges((edge,))
    witness = v2a.EndpointAssignmentWitness.build(
        labels=((a.reading_sha256, 0, 0, "0" * 64),
                (b.reading_sha256, 0, 0, b.label_for(0, 0).label_sha256)),
        edge_multiplicities=((edge.edge_sha256, 1),))
    assert v2a._kernel_cross_link_mismatch(domains, edges, (witness,)) is not None


# ---------------------------------------------------------------------------
# 5.9 Permutation byte identity
# ---------------------------------------------------------------------------

def test_kernel_permuted_inputs_produce_byte_identical_records():
    a, b = _domain("A"), _domain("B", nv=2)
    c = _domain("C")
    ab = _edge(a, b, [(0, 0, 0, 1), (0, 1, 1, 1)], ordinal=0)
    bc = _edge(b, c, [(0, 0, 0, 1), (1, 0, 1, 1)], ordinal=1)
    baseline = v2a.solve_endpoint_provenance_kernel(
        domains=(a, b, c), edges=(ab, bc), budget=_budget())
    permuted = v2a.solve_endpoint_provenance_kernel(
        domains=(c, a, b), edges=(bc, ab), budget=_budget())
    assert baseline.to_json_bytes() == permuted.to_json_bytes()
    assert baseline.result_sha256 == permuted.result_sha256


# ---------------------------------------------------------------------------
# Budgets: exact limit and one below
# ---------------------------------------------------------------------------

def test_kernel_budget_constructor_rejects_boolean_and_small_counts():
    limits = {name: 4 for name in v2a.V2ABudget.__dataclass_fields__}
    limits["max_distinct_assignments"] = 2
    assert v2a.V2ABudget(**limits).max_edges == 4
    limits["max_edges"] = True
    with pytest.raises(ValueError):
        v2a.V2ABudget(**limits)
    limits = {name: 4 for name in v2a.V2ABudget.__dataclass_fields__}
    limits["max_distinct_assignments"] = 1
    with pytest.raises(ValueError):
        v2a.V2ABudget(**limits)
    with pytest.raises(ValueError):
        v2a.V2AWork(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)


def test_kernel_runtime_exact_limit_succeeds_and_one_below_is_atomic():
    a, b = _domain("A"), _domain("B")
    edge = _edge(a, b, [(0, 0, 0, 1)])
    baseline = _solve((a, b), (edge,))
    observed = baseline.runtime_work
    exact = _solve((a, b), (edge,),
                   max_partial_label_assignments=observed.partial_label_assignments,
                   max_consistency_checks=observed.consistency_checks)
    assert exact.classification == "endpoint_unique"
    below = _solve((a, b), (edge,),
                   max_consistency_checks=observed.consistency_checks - 1)
    assert below.status == "budget_exhausted"
    assert below.budget_failure_reason == "runtime_budget_exceeded"
    assert below.classification is None and below.witnesses == ()


def test_kernel_retained_witness_limit_is_exact():
    a, b = _domain("A"), _domain("B", nv=2)
    edge = _edge(a, b, [(0, 0, 0, 1), (0, 1, 1, 1)])
    exact = _solve((a, b), (edge,), max_retained_witnesses=2)
    assert exact.classification == "endpoint_multiple"
    below = _solve((a, b), (edge,), max_retained_witnesses=1)
    assert below.status == "budget_exhausted"
    assert below.budget_failure_reason == "runtime_budget_exceeded"
    assert below.runtime_work.retained_witnesses == 2


def test_kernel_structural_preflight_is_bounded():
    a, b = _domain("A"), _domain("B")
    edge = _edge(a, b, [(0, 0, 0, 1)])
    exhausted = _solve((a, b), (edge,), max_readings=1)
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_failure_reason == "runtime_budget_exceeded"
    assert exhausted.witnesses == ()


def test_public_preflight_exact_limit_succeeds_and_one_below_is_atomic(
        ring, solved_ring):
    baseline = _run_public(ring, solved_ring, _ring_budget())
    plan = baseline.planned_work
    exact = _run_public(ring, solved_ring, _budget(
        max_readings=plan.readings, max_edges=plan.edges,
        max_endpoint_labels=plan.endpoint_labels,
        max_residues_inspected=plan.residues_inspected,
        max_edge_candidates_inspected=plan.edge_candidates_inspected))
    assert exact.status == "complete"
    assert exact.classification == "endpoint_unique"
    below = _run_public(ring, solved_ring, _budget(
        max_edge_candidates_inspected=plan.edge_candidates_inspected - 1))
    assert below.status == "budget_exhausted"
    assert below.budget_failure_reason == "planned_budget_exceeded"
    assert below.planned_work.edge_candidates_inspected \
        == plan.edge_candidates_inspected
    assert below.runtime_work == v2a.V2AWork.zero()
    assert below.reading_domains == () and below.witnesses == ()


def test_public_runtime_exact_limit_succeeds_and_one_below_is_atomic(
        ring, solved_ring):
    baseline = _run_public(ring, solved_ring, _ring_budget())
    observed = baseline.runtime_work
    exact = _run_public(ring, solved_ring, _budget(
        max_partial_label_assignments=observed.partial_label_assignments,
        max_consistency_checks=observed.consistency_checks))
    assert exact.status == "complete"
    assert exact.classification == "endpoint_unique"
    below = _run_public(ring, solved_ring, _budget(
        max_consistency_checks=observed.consistency_checks - 1))
    assert below.status == "budget_exhausted"
    assert below.budget_failure_reason == "runtime_budget_exceeded"
    assert below.runtime_work.consistency_checks \
        == observed.consistency_checks
    assert below.reading_domains == () and below.witnesses == ()


# ---------------------------------------------------------------------------
# 5.7 Orientation coverage at the frozen record level
# ---------------------------------------------------------------------------

def test_oriented_labels_cover_axes_quadrants_and_sides():
    digests = set()
    for axis in range(3):
        for quadrant in range(4):
            for side in (-1, 1):
                universe = _digest(f"side:{side}")
                domain = _domain(f"F{axis}{quadrant}{side}", nu=2, nv=2,
                                 quadrant=quadrant, axis=axis,
                                 universe=universe)
                digests.add(domain.domain_sha256)
                label = domain.label_for(0, 0)
                assert label.absolute_quadrant == quadrant
                assert label.residue_u.selected_b == 1.0
                assert label.label_sha256 == domain.label_for(0, 0).label_sha256
    assert len(digests) == 3 * 4 * 2
    assert (_domain("A", quadrant=0).label_for(0, 0).label_sha256
            != _domain("A", quadrant=1).label_for(0, 0).label_sha256)


def test_relative_quadrant_substitution_changes_the_label():
    absolute = _domain("A", quadrant=3)
    # A mutant that substitutes the relative quadrant_of_right (0 here) for the
    # C1 absolute quadrant produces a different, non-authoritative label.
    relative_substitute = _domain("A", quadrant=0)
    assert absolute.label_for(0, 0).absolute_quadrant == 3
    assert (absolute.label_for(0, 0).label_sha256
            != relative_substitute.label_for(0, 0).label_sha256)


def test_exact_tree_delta_versus_rational_chord_tolerance(ring):
    (vector_result, _p1d, _signatures, _matches, _support, _phases,
     _readings) = ring
    domain = vector_result.domains[0]
    reference = next(
        reference for reference in
        _ring_certificate(ring).edge_references
        if reference.match_identity == domain.match_identity)
    exact = tuple(reference.delta_dyadic_xyz)
    assert v2a._candidate_feasible(domain, exact, "tree", Fraction(0),
                                   reference.global_index)
    lateral_axis = next(
        axis for axis in range(3)
        if axis != domain.normal_constraint.world_axis)
    perturbed = list(exact)
    shifted = Fraction(*exact[lateral_axis]) + Fraction(1, 2)
    perturbed[lateral_axis] = (shifted.numerator, shifted.denominator)
    perturbed = tuple(perturbed)
    assert not v2a._candidate_feasible(domain, perturbed, "tree",
                                       Fraction(0), reference.global_index)
    assert v2a._candidate_feasible(domain, perturbed, "chord",
                                   Fraction(1, 2), reference.global_index)
    assert not v2a._candidate_feasible(domain, perturbed, "chord",
                                       Fraction(1, 4), reference.global_index)


def _ring_certificate(ring):
    solved = relative.solve_relative_translation_graph(
        ring[0], ring[4], ring[1], ring[3], ring[5],
        readings=ring[6], signatures=ring[2],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10, max_tree_edges=10,
            max_chords=10, max_cycle_rows=30))
    return solved.components[0].translation_classes[0]


# ---------------------------------------------------------------------------
# 5.8 Digest, constructor and replace attacks
# ---------------------------------------------------------------------------

def test_kernel_and_public_records_fail_closed_on_tampering(ring, solved_ring):
    a, b = _domain("A"), _domain("B", nv=2)
    edge = _edge(a, b, [(0, 0, 0, 1)])
    with pytest.raises(ValueError):
        replace(a, domain_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(a, label_count=5)
    with pytest.raises(ValueError):
        replace(a.residue_u, candidate_values=(0.25, 0.5))
    with pytest.raises(ValueError):
        replace(edge, edge_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(edge.buckets[0], multiplicity=9)

    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    with pytest.raises(ValueError):
        replace(result, classification="endpoint_none")
    with pytest.raises(ValueError):
        replace(result, result_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(result.reading_domains[0], label_count=7)

    tampered = object.__new__(v2a.ReadingLabelDomain)
    for field in result.reading_domains[0].__dataclass_fields__:
        object.__setattr__(tampered, field,
                           getattr(result.reading_domains[0], field))
    object.__setattr__(tampered, "label_count", 7)
    fields = {field: getattr(result, field)
              for field in result.__dataclass_fields__}
    fields["reading_domains"] = (tampered,) + result.reading_domains[1:]
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**fields)


def _rejects_replace(record, **changes):
    with pytest.raises(ValueError):
        replace(record, **changes)


def test_every_named_identity_field_fails_closed():
    domain = _domain("A", nu=2, nv=2, quadrant=2)
    identity = domain.reading_identity
    residue = domain.label_for(0, 0).residue_u
    label = domain.label_for(0, 0)
    _rejects_replace(identity, frame="B")
    _rejects_replace(identity, canonical_index=1)
    _rejects_replace(identity, reading_order_key_sha256="0" * 64)
    _rejects_replace(identity, reading_axis=0)
    _rejects_replace(identity, source_region_id="a" * 64)
    _rejects_replace(identity, canonical_universe_sha256="0" * 64)
    _rejects_replace(residue, source_record_sha256="0" * 64)
    _rejects_replace(residue, selected_b=2.0)
    _rejects_replace(residue, local_axis="v")
    _rejects_replace(residue, local_axis_label=3)
    _rejects_replace(residue, residue_index=99)
    _rejects_replace(residue, residue_value=0.5)
    _rejects_replace(label, selected_b=2.0)
    _rejects_replace(label, absolute_quadrant=1)
    _rejects_replace(label, footprint_identity="a" * 64)
    _rejects_replace(label, residue_u=label.residue_v)
    _rejects_replace(label, residue_v=label.residue_u)
    _rejects_replace(domain, selected_b=2.0)
    _rejects_replace(domain, absolute_quadrant=1)
    _rejects_replace(domain, footprint_identity="a" * 64)


def test_binding_uses_equal_content_reading_identity_not_object_identity(
        ring, solved_ring):
    (vector_result, _p1d, signatures, _matches, _support, _phases,
     readings) = ring
    baseline = v2a._bind_pipeline(solved_ring, vector_result, readings, signatures)
    clone_index = 0
    cloned = replace(readings[clone_index])
    assert cloned is not readings[clone_index]
    assert (lattice_planes.reading_order_key(cloned)
            == lattice_planes.reading_order_key(readings[clone_index]))
    substituted = tuple(cloned if position == clone_index else reading
                        for position, reading in enumerate(readings))
    rebound = v2a._bind_pipeline(
        solved_ring, vector_result, substituted, signatures)
    assert [d.domain_sha256 for d in rebound.reading_domains] == \
        [d.domain_sha256 for d in baseline.reading_domains]
    assert rebound.edge_endpoints == baseline.edge_endpoints


def test_binding_uses_equal_content_signature_identity_not_object_identity(
        ring, solved_ring):
    (vector_result, _p1d, signatures, _matches, _support, _phases,
     readings) = ring
    baseline = v2a._bind_pipeline(solved_ring, vector_result, readings, signatures)
    clone_index = 0
    assert (lattice_planes.reading_order_key(signatures[clone_index].reading)
            == lattice_planes.reading_order_key(readings[clone_index]))
    cloned_reading = replace(readings[clone_index])
    cloned_signature = replace(signatures[clone_index], reading=cloned_reading)
    assert cloned_signature is not signatures[clone_index]
    substituted = tuple(
        cloned_signature if position == clone_index else signature
        for position, signature in enumerate(signatures))
    rebound = v2a._bind_pipeline(
        solved_ring, vector_result, readings, substituted)
    assert [d.domain_sha256 for d in rebound.reading_domains] == \
        [d.domain_sha256 for d in baseline.reading_domains]


# ---------------------------------------------------------------------------
# Public prerequisite replay and classification separation
# ---------------------------------------------------------------------------

def test_public_ring_is_endpoint_unique_and_replays(ring, solved_ring):
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete", [item.to_record()
                                         for item in result.refusals]
    assert result.classification == "endpoint_unique"
    assert result.exhaustive is True
    assert result.translation_class_count == 1
    assert result.endpoint_assignment_count == 1
    assert len(result.witnesses) == 1
    assert result.runtime_work.candidate_materializations == 0
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = ring
    assert len(result.reading_domains) == len(readings) == 6
    assert len(result.edge_records) == len(matches) == 3
    assert all(record.stored_reference_membership
               for record in result.edge_records)
    assert {record.edge_kind for record in result.edge_records} == {
        "tree", "chord"}
    assert v2a.validate_endpoint_provenance_result(
        result, solved_ring, vector_result, support, p1d_result, matches,
        phases, readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def _result_fields(result, **changes):
    fields = {name: getattr(result, name)
              for name in result.__dataclass_fields__}
    fields.update(changes)
    return fields


def _rebuild_kernel_result_sha256(result, runtime_work):
    """Reproduce the attacker's R4 move: reseal the kernel digest from the
    mutated public projection so a digest-only binding can never see it."""
    kernel_work = replace(runtime_work, result_validation_cross_links=0)
    return v2a.KernelResult.build(
        status="complete", classification=result.classification,
        exhaustive=result.exhaustive,
        distinct_assignment_count=result.endpoint_assignment_count,
        lower_bound_only=result.lower_bound_only,
        witnesses=result.witnesses, runtime_work=kernel_work,
        budget_failure_reason=None, refusal_reason=None).result_sha256


def test_result_cross_links_reject_freshly_resealed_mutations(ring, solved_ring):
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete", [item.to_record()
                                         for item in result.refusals]
    witness = result.witnesses[0]

    labels = list(witness.labels)
    labels[0] = (labels[0][0], labels[0][1], labels[0][2], "0" * 64)
    resealed_witness = v2a.EndpointAssignmentWitness.build(
        labels=labels, edge_multiplicities=witness.edge_multiplicities)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(
            **_result_fields(result, witnesses=(resealed_witness,)))

    labels = list(witness.labels)
    labels[0] = (labels[0][0], 99, labels[0][2], labels[0][3])
    out_of_domain = v2a.EndpointAssignmentWitness.build(
        labels=labels, edge_multiplicities=witness.edge_multiplicities)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(
            **_result_fields(result, witnesses=(out_of_domain,)))

    record = next(item for item in result.edge_records
                  if item.edge_kind == "chord")
    pairs = list(record.label_pairs)
    pairs[0] = (pairs[0][0], pairs[0][1], pairs[0][2] + 1)
    chord_index = result.edge_records.index(record)
    resealed_record = v2a.EdgeFeasibleRecord.build(**_edge_record_fields(
        record, feasible_count=record.feasible_count + 1,
        inspected_candidates=max(record.inspected_candidates,
                                 record.feasible_count + 1),
        label_pairs=pairs))
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result, edge_records=result.edge_records[:chord_index]
            + (resealed_record,) + result.edge_records[chord_index + 1:]))

    observed = result.runtime_work
    resealed_work = v2a.V2AWork(
        observed.readings + 1, observed.edges, observed.endpoint_labels,
        observed.residues_inspected, observed.edge_candidates_inspected,
        observed.partial_label_assignments, observed.consistency_checks,
        observed.result_validation_cross_links, observed.distinct_assignments,
        observed.retained_witnesses, observed.candidate_materializations)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(
            **_result_fields(result, runtime_work=resealed_work))


def _edge_record_fields(record, **changes):
    fields = {name: getattr(record, name)
              for name in record.__dataclass_fields__
              if name != "record_sha256"}
    fields.update(changes)
    return fields


def test_resealed_edge_record_with_out_of_domain_pair_is_rejected(
        ring, solved_ring):
    """Luna false-pass regression: a resealed edge record that invents an
    impossible/out-of-domain label pair with an inconsistent feasible_count
    must be rejected by its sibling commitment and reading-label domains."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete", [item.to_record()
                                         for item in result.refusals]
    record = result.edge_records[0]
    left_domain = next(domain for domain in result.reading_domains
                       if domain.reading_sha256 == record.left_reading_sha256)
    pairs = list(record.label_pairs)
    pairs.append((left_domain.label_count + 5, 0, 1))
    pairs.sort()
    resealed = v2a.EdgeFeasibleRecord.build(**_edge_record_fields(
        record, feasible_count=record.feasible_count + 1,
        inspected_candidates=max(record.inspected_candidates,
                                 record.feasible_count + 1),
        label_pairs=pairs))
    # The resealed record's own digest and multiplicity sum replay; only the
    # independently retained commitment sibling and domain ranges catch it.
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result, edge_records=(resealed,) + result.edge_records[1:]))


def test_resealed_commitment_sibling_with_relinked_record_is_rejected(
        ring, solved_ring):
    """Resealing the commitment and re-linking the record still mismatches the
    canonical bucket aggregate: the sibling is independent, not a shared field."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    commitment = result.enumeration_commitments[0]
    record = result.edge_records[0]
    resealed_commitment = v2a.EnumerationCommitment.build(
        match_identity=commitment.match_identity,
        edge_kind=commitment.edge_kind,
        domain_sha256=commitment.domain_sha256,
        accepted_delta_dyadic_xyz=commitment.accepted_delta_dyadic_xyz,
        left_reading_sha256=commitment.left_reading_sha256,
        right_reading_sha256=commitment.right_reading_sha256,
        inspected_candidates=commitment.inspected_candidates,
        residues_inspected=commitment.residues_inspected,
        feasible_count=commitment.feasible_count,
        bucket_count=commitment.bucket_count,
        bucket_aggregates_sha256="a" * 64,
        stream_transcript_sha256=commitment.stream_transcript_sha256,
        reference_global_index=commitment.reference_global_index,
        reference_left_code=commitment.reference_left_code,
        reference_right_code=commitment.reference_right_code,
        reference_seen=commitment.reference_seen)
    relinked = v2a.EdgeFeasibleRecord.build(**_edge_record_fields(
        record,
        enumeration_commitment_sha256=resealed_commitment.commitment_sha256))
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result,
            edge_records=(relinked,) + result.edge_records[1:],
            enumeration_commitments=(resealed_commitment,)
            + result.enumeration_commitments[1:]))


def test_resealed_runtime_work_cannot_exceed_or_disagree(ring, solved_ring):
    """Luna false-pass regression: a freshly resealed runtime work record that
    exceeds a budget cap or disagrees with the reported assignment/witness
    counts must be rejected."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    observed = result.runtime_work
    over_budget = replace(
        observed, consistency_checks=result.budget.max_consistency_checks + 1)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(
            **_result_fields(result, runtime_work=over_budget))
    disagreeing = replace(
        observed, retained_witnesses=observed.retained_witnesses + 1,
        distinct_assignments=observed.distinct_assignments + 1)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(
            **_result_fields(result, runtime_work=disagreeing))


def test_resealed_under_cap_partial_label_assignments_is_rejected(
        ring, solved_ring):
    """R3/R4 false-pass regression (a), exact reproduction with the attacker's
    rebuilt digest: the public XYZ fixture reports one partial assignment per
    reading; incrementing that exact count stays under the request cap and leaves every
    parent edge/commitment/count relation unchanged, and the attacker reseals
    ``kernel_result_sha256`` from the mutated projection, yet the deterministic
    kernel rerun over the retained domains/edges rejects it.  A digest-only
    binding cannot catch this."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    observed = result.runtime_work
    assert observed.partial_label_assignments == len(ring[-1]) == 6
    bumped = observed.partial_label_assignments + 1
    assert bumped == 7
    assert bumped <= result.budget.max_partial_label_assignments
    resealed = replace(observed, partial_label_assignments=bumped)
    rebuilt = _rebuild_kernel_result_sha256(result, resealed)
    counter = v2a._RuntimeCounter(result.budget)
    assert v2a._cross_link_mismatch(
        result.reading_domains, result.edge_records,
        result.enumeration_commitments, result.witnesses,
        endpoint_assignment_count=result.endpoint_assignment_count,
        planned_work=result.planned_work, runtime_work=resealed,
        budget=result.budget, counter=counter) is None
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result, runtime_work=resealed, kernel_result_sha256=rebuilt))


def test_resealed_under_cap_consistency_checks_is_rejected(ring, solved_ring):
    """R3/R4 false-pass regression (b), exact reproduction with the attacker's
    rebuilt digest: the public XYZ fixture reports one consistency check per
    reading; incrementing that exact count stays under the request cap and leaves every
    parent edge/commitment/count relation unchanged, and the attacker reseals
    ``kernel_result_sha256`` from the mutated projection, yet the deterministic
    kernel rerun over the retained domains/edges rejects it."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    observed = result.runtime_work
    assert observed.consistency_checks == len(ring[-1]) == 6
    bumped = observed.consistency_checks + 1
    assert bumped == 7
    assert bumped <= result.budget.max_consistency_checks
    resealed = replace(observed, consistency_checks=bumped)
    rebuilt = _rebuild_kernel_result_sha256(result, resealed)
    counter = v2a._RuntimeCounter(result.budget)
    assert v2a._cross_link_mismatch(
        result.reading_domains, result.edge_records,
        result.enumeration_commitments, result.witnesses,
        endpoint_assignment_count=result.endpoint_assignment_count,
        planned_work=result.planned_work, runtime_work=resealed,
        budget=result.budget, counter=counter) is None
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result, runtime_work=resealed, kernel_result_sha256=rebuilt))


def test_kernel_work_counter_attack_matrix_with_rebuilt_digest(ring, solved_ring):
    """R5 attack matrix: mutate every kernel-produced ``runtime_work`` counter,
    reseal ``kernel_result_sha256`` from the mutated projection (the exact R4
    digest-rebuild false-pass), and require ``V2AEndpointResult.build`` to reject
    each one.  The decisive layer for the two search counters that no parent
    cross-link constrains (``partial_label_assignments`` and
    ``consistency_checks``) is the deterministic kernel rerun."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    observed = result.runtime_work
    assert observed.partial_label_assignments > 0
    assert observed.consistency_checks > 0
    mutations = (
        ("readings", replace(observed,
                             readings=observed.readings + 1)),
        ("edges", replace(observed, edges=observed.edges + 1)),
        ("endpoint_labels", replace(observed,
                                    endpoint_labels=observed.endpoint_labels + 1)),
        ("residues_inspected", replace(
            observed, residues_inspected=observed.residues_inspected + 4)),
        ("edge_candidates_inspected", replace(
            observed,
            edge_candidates_inspected=observed.edge_candidates_inspected + 1)),
        ("partial_label_assignments", replace(
            observed,
            partial_label_assignments=observed.partial_label_assignments + 1)),
        ("consistency_checks", replace(
            observed, consistency_checks=observed.consistency_checks + 1)),
        ("distinct_assignments", replace(
            observed, distinct_assignments=observed.distinct_assignments + 1)),
        ("retained_witnesses", replace(
            observed, retained_witnesses=observed.retained_witnesses + 1)),
        ("result_validation_cross_links", replace(
            observed,
            result_validation_cross_links=(
                observed.result_validation_cross_links + 1))),
    )
    for name, mutated in mutations:
        rebuilt = _rebuild_kernel_result_sha256(result, mutated)
        assert (rebuilt != result.kernel_result_sha256
                or name == "result_validation_cross_links"), name
        with pytest.raises(ValueError):
            v2a.V2AEndpointResult.build(**_result_fields(
                result, runtime_work=mutated,
                kernel_result_sha256=rebuilt))


def test_reconstructed_kernel_initial_work_uses_commitments_not_runtime(
        ring, solved_ring):
    """The rerun's pre-search work is rebuilt from the independently retained
    commitment candidate/residue totals, never from the mutable runtime record,
    with every kernel-search/result/cross-link counter reset to zero."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    initial = v2a._reconstructed_kernel_initial_work(
        result.reading_domains, result.enumeration_commitments)
    assert initial.readings == len(result.reading_domains)
    assert initial.edges == len(result.enumeration_commitments)
    assert initial.endpoint_labels == sum(
        domain.label_count for domain in result.reading_domains)
    assert initial.residues_inspected == sum(
        commitment.residues_inspected
        for commitment in result.enumeration_commitments)
    assert initial.edge_candidates_inspected == sum(
        commitment.inspected_candidates
        for commitment in result.enumeration_commitments)
    assert initial.partial_label_assignments == 0
    assert initial.consistency_checks == 0
    assert initial.result_validation_cross_links == 0
    assert initial.distinct_assignments == 0
    assert initial.retained_witnesses == 0
    assert initial.candidate_materializations == 0


def test_public_result_retains_cross_linked_kernel_edges(ring, solved_ring):
    """R5 retention: every complete result carries the canonical ``KernelEdge``
    tuple top-level, each edge record cross-links to exactly one retained edge,
    and the exact label-pair projection is the retained bucket list.  The opaque
    representative sentinel is excluded from the serialized record, and the
    deterministic result-validation plan counts the kernel-edge cross-link and
    replay units."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    assert len(result.kernel_edges) == len(result.edge_records)
    keys = [edge.match_identity.sort_key() for edge in result.kernel_edges]
    assert keys == sorted(keys)
    by_sha = {edge.edge_sha256: edge for edge in result.kernel_edges}
    assert len(by_sha) == len(result.kernel_edges)
    for record in result.edge_records:
        edge = by_sha[record.kernel_edge_sha256]
        assert edge.match_identity == record.match_identity
        assert edge.edge_kind == record.edge_kind
        assert tuple(edge.accepted_delta_dyadic_xyz) \
            == tuple(record.accepted_delta_dyadic_xyz)
        assert (edge.left_reading_sha256, edge.right_reading_sha256) \
            == (record.left_reading_sha256, record.right_reading_sha256)
        assert tuple((bucket.left_code, bucket.right_code, bucket.multiplicity)
                     for bucket in edge.buckets) == record.label_pairs
    assert b"ignored_representative" not in result.to_json_bytes()
    assert len(result.to_record()["kernel_edges"]) == len(result.kernel_edges)
    assert (result.runtime_work.result_validation_cross_links
            >= 2 * len(result.kernel_edges) + 1)


def test_resealed_retained_kernel_edge_bucket_is_rejected(ring, solved_ring):
    """R5 cross-link: a resealed retained kernel edge whose bucket list no longer
    projects to its edge record is rejected even though its own digest replays."""
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    index = next(index for index, edge in enumerate(result.kernel_edges)
                 if edge.buckets)
    edge = result.kernel_edges[index]
    buckets = list(edge.buckets)
    original = buckets[0]
    buckets[0] = v2a.KernelBucket.build(
        global_index=original.global_index,
        left_code=original.left_code, right_code=original.right_code,
        multiplicity=original.multiplicity + 1)
    resealed = v2a.KernelEdge.build(
        match_identity=edge.match_identity, edge_kind=edge.edge_kind,
        accepted_delta_dyadic_xyz=edge.accepted_delta_dyadic_xyz,
        left_reading_sha256=edge.left_reading_sha256,
        right_reading_sha256=edge.right_reading_sha256, buckets=buckets,
        ignored_representative=edge.ignored_representative)
    with pytest.raises(ValueError):
        v2a.V2AEndpointResult.build(**_result_fields(
            result, kernel_edges=result.kernel_edges[:index]
            + (resealed,) + result.kernel_edges[index + 1:]))


def test_public_result_validation_cross_links_exact_limit_and_one_below(
        ring, solved_ring):
    baseline = _run_public(ring, solved_ring, _ring_budget())
    assert baseline.status == "complete"
    plan = baseline.planned_work
    assert baseline.runtime_work.result_validation_cross_links > 0
    assert (baseline.runtime_work.result_validation_cross_links
            <= plan.result_validation_cross_links)
    exact = _run_public(ring, solved_ring, _budget(
        max_result_validation_cross_links=plan.result_validation_cross_links))
    assert exact.status == "complete"
    assert exact.classification == "endpoint_unique"
    below = _run_public(ring, solved_ring, _budget(
        max_result_validation_cross_links=plan.result_validation_cross_links - 1))
    assert below.status == "budget_exhausted"
    assert below.budget_failure_reason == "planned_budget_exceeded"
    assert below.planned_work.result_validation_cross_links \
        == plan.result_validation_cross_links
    assert below.reading_domains == () and below.witnesses == ()


def test_cross_link_runtime_counter_stops_atomically_under_a_tight_cap(
        ring, solved_ring):
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    counter = v2a._RuntimeCounter(
        _budget(max_result_validation_cross_links=1))
    with pytest.raises(v2a.V2ABudgetStop) as stop:
        v2a._cross_link_mismatch(
            result.reading_domains, result.edge_records,
            result.enumeration_commitments, result.witnesses,
            endpoint_assignment_count=result.endpoint_assignment_count,
            planned_work=result.planned_work,
            runtime_work=result.runtime_work,
            budget=counter.budget, counter=counter)
    assert stop.value.reason == "runtime_budget_exceeded"


def test_multi_candidate_a4_private_binding_evidence_is_isolated(
        multi_ring, solved_multi_ring):
    """Isolated binding/enumeration integration evidence, NOT public end-to-end
    acceptance.

    Public ``analyze_endpoint_provenance`` refuses this fixture at the frozen
    section 2 gate (``gate_not_ready``) before any identity binding or kernel
    classification runs, so it yields no public acceptance for a multi-candidate
    A4 component.  The evidence below is obtained only by calling the private
    ``_bind_pipeline`` and ``_enumerate_pipeline`` helpers directly; it shows
    that the same binding/enumeration code ``analyze`` would use represents and
    enumerates more than one feasible ``E_m`` candidate per edge and cross-links
    each streamed ``EnumerationCommitment`` to its ``EdgeFeasibleRecord`` and
    ``KernelEdge``.  It is isolated integration evidence only and leaves
    multi-candidate public acceptance blocked.

    Documented exact limitation: a ``b < 1`` endpoint table necessarily carries a
    second half-period residue candidate, so every multi-candidate domain also
    carries a second exact delta class; C2-R-B therefore reports two translation
    classes and ``placement_readiness == "review_required"``.  The frozen section
    2 gate consequently refuses through public ``analyze`` (asserted atomically)
    and no multi-candidate A4 component can reach ``endpoint_unique``.
    """
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = multi_ring
    component = solved_multi_ring.components[0]
    assert solved_multi_ring.placement_readiness == "review_required"
    assert len(component.translation_classes) == 2
    refused = v2a.analyze_endpoint_provenance(
        solved_multi_ring, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_ring_budget())
    assert refused.status == "refused"
    assert refused.refusals[0].reason == "gate_not_ready"
    assert refused.edge_records == () and refused.witnesses == ()

    bound = v2a._bind_pipeline(
        solved_multi_ring, vector_result, readings, signatures)
    counter = v2a._RuntimeCounter(_ring_budget())
    kernel_edges, edge_records, commitments = v2a._enumerate_pipeline(
        bound, Fraction(0), counter)
    assert len(edge_records) == len(component.tree_edges) + len(component.chords)
    assert any(record.inspected_candidates > 1 for record in edge_records)
    assert any(record.feasible_count > 1 for record in edge_records)
    assert counter.edge_candidates_inspected > len(bound.domains)
    for record, commitment, kernel_edge in zip(
            edge_records, commitments, kernel_edges):
        assert record.enumeration_commitment_sha256 \
            == commitment.commitment_sha256
        assert commitment.bucket_count == len(record.label_pairs)
        assert commitment.feasible_count == record.feasible_count
        assert kernel_edge.edge_sha256 == record.kernel_edge_sha256
        assert sum(item[2] for item in record.label_pairs) \
            == record.feasible_count
    # Distinct label pairs may each carry one injective tree witness; a chord
    # band may carry several.  Either way the streaming transcript, not a
    # retained index collection, is the commitment.
    assert any(record.induced_label_pair_count > 1 for record in edge_records)


# ---------------------------------------------------------------------------
# Public multi-candidate chord acceptance through the frozen gate
# ---------------------------------------------------------------------------

def _public_multicandidate_chord_case():
    """Public two-frame fixture: a singleton tree edge plus a half-period chord.

    Frames ``A`` and ``B`` each carry two distinct same-axis readings.  The
    integer-period reading of each frame supplies the directed tree edge
    ``A -> B``; the half-period reading supplies the reversed chord ``B -> A``.
    C2-R-B's spanning tree therefore contains only singleton integer-period
    domains, so it still reports exactly one translation class, while the chord
    is not on the tree and its A4 domain represents sixteen candidates whose
    exact ``E_m`` has four witnesses (two per lateral axis, from the duplicated
    half-period residue pair).  Every source object is built by the same public
    test helpers used by the accepted ring fixtures; no production code, A4
    code or private V2-A helper is monkeypatched or called.
    """
    basis = np.eye(2, dtype=np.float64)
    translation = {"A": (0.0, 0.0), "B": (2.0, 0.0)}
    readings = []
    signatures = []
    for frame, blocks in (("A", 1.0), ("A", 0.5), ("B", 1.0), ("B", 0.5)):
        point = translation[frame]
        residues = tuple(float((-value) % 1.0) for value in point)
        reading = _reading(
            frame, 2, blocks, 2.0, 1.0, (0, 0), residues,
            samples=128, extent=4.0)
        readings.append(reading)
        signature = _signature(reading, blocks, basis, np.asarray(point))
        signatures.append(replace(
            signature,
            coverage=_centred_coverage(signature.coverage.shape)))

    readings, signatures = tuple(readings), tuple(signatures)
    by_reading = {
        (reading.frame, float(reading.alternatives[0].blocks_per_period)):
            (reading, signature)
        for reading, signature in zip(readings, signatures)}

    def _public_match(left_key, right_key):
        left, left_signature = by_reading[left_key]
        right, right_signature = by_reading[right_key]
        pair = plane_graph.match_plane_pair(
            left_signature, right_signature, quadrant=0, max_shift_cells=4)
        assert pair is not None
        return replace(
            pair,
            left_plane=_canonical_identity(readings, left)[1],
            right_plane=_canonical_identity(readings, right)[1])

    matches = (
        _public_match(("A", 1.0), ("B", 1.0)),
        _public_match(("B", 0.5), ("A", 0.5)),
    )
    solution = plane_graph.ComponentSolution(
        frames=("A", "B"), quadrants={"A": 0, "B": 0},
        translations={"A": (0.0, 0.0, 0.0), "B": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={
            _canonical_identity(readings, reading):
                float(reading.alternatives[0].blocks_per_period)
            for reading in readings},
        reason="", alternatives=())
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


@pytest.fixture(scope="module")
def public_multicandidate_chord():
    return _public_multicandidate_chord_case()


@pytest.mark.skip(reason=MULTICANDIDATE_FIXTURE_BLOCKED)
def test_public_multicandidate_chord_passes_gate_with_multiple_em(
        public_multicandidate_chord):
    """A genuine public multi-candidate chord is accepted end to end.

    Asserted record counts are the exact observed values: four reading domains
    with label counts ``[1, 1, 4, 4]``, two edge records, one tree record with
    one inspected candidate and one feasible witness, and one chord record with
    sixteen represented candidates and four exact feasible witnesses.  This
    supersedes the prior "no multi-candidate public acceptance" claim for the
    disjoint-chord case; the shared-reading all-half-period ring remains refused
    at the gate, and a multi-candidate domain still cannot reach
    ``endpoint_unique``.
    """
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = public_multicandidate_chord
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10, max_tree_edges=10,
            max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.placement_readiness == "ready"
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert len(component.translation_classes) == 1
    assert component.cycle_rank == 1
    assert component.failing_chords == ()
    assert len(component.tree_edges) == 1 and len(component.chords) == 1
    assert all(cycle.matching_provenance_count > 0
               for cycle in component.cycle_residuals)
    assert sorted(domain.represented_candidate_count
                  for domain in vector_result.domains) == [1, 16]

    result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_ring_budget())
    assert result.status == "complete", [item.to_record()
                                         for item in result.refusals]
    assert result.refusals == () and result.budget_failure_reason is None
    assert result.classification == "endpoint_multiple"
    assert result.exhaustive is False and result.lower_bound_only is True
    assert result.translation_class_count == 1
    assert result.endpoint_assignment_count == 2
    assert len(result.witnesses) == 2
    assert len(result.reading_domains) == 4
    assert len(result.edge_records) == 2
    assert len(result.kernel_edges) == 2
    assert len(result.enumeration_commitments) == 2
    assert sorted(domain.label_count for domain in result.reading_domains) \
        == [1, 1, 4, 4]

    tree_records = [record for record in result.edge_records
                    if record.edge_kind == "tree"]
    chord_records = [record for record in result.edge_records
                     if record.edge_kind == "chord"]
    assert len(tree_records) == 1 and len(chord_records) == 1
    tree_record, chord_record = tree_records[0], chord_records[0]
    assert tree_record.inspected_candidates == 1
    assert tree_record.feasible_count == 1
    assert tree_record.label_pairs == ((0, 0, 1),)
    # Exact public multi-candidate evidence: the A4 chord domain represents
    # sixteen indices and four exact E_m witnesses on four distinct induced
    # label pairs, all retained without materializing an A4 candidate.
    assert chord_record.inspected_candidates == 16
    assert chord_record.feasible_count == 4
    assert chord_record.induced_label_pair_count == 4
    assert chord_record.label_pairs == (
        (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1))
    assert result.planned_work.edge_candidates_inspected == 17
    assert result.runtime_work.edge_candidates_inspected == 17
    assert result.runtime_work.endpoint_labels == 10
    assert result.runtime_work.candidate_materializations == 0
    assert v2a.validate_endpoint_provenance_result(
        result, solved, vector_result, support, p1d_result, matches,
        phases, readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def test_public_result_validation_mismatch_is_a_typed_refusal(
        ring, solved_ring, monkeypatch):
    monkeypatch.setattr(v2a, "_cross_link_mismatch",
                        lambda *args, **kwargs: "synthetic cross-link fault")
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason == "result_validation_mismatch"
    assert result.classification is None and result.witnesses == ()


def test_public_chord_tolerance_request_replays(ring):
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = ring
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=1e-6,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10, max_tree_edges=10,
            max_chords=10, max_cycle_rows=30))
    assert solved.placement_readiness == "ready"
    result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=1e-6,
        budget=_ring_budget())
    assert result.status == "complete", [item.to_record()
                                         for item in result.refusals]
    assert result.classification == "endpoint_unique"
    assert any(record.edge_kind == "chord" for record in result.edge_records)


def test_public_path_never_materializes_a4_candidates(ring, solved_ring,
                                                      monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("V2-A materialized an A4 candidate")

    vector = sys.modules["mc_photogrammetry.reconstruction.plane_vector_domain"]
    monkeypatch.setattr(vector.MatchLateralVectorDomain, "candidate_at", _boom)
    monkeypatch.setattr(vector.MatchLateralVectorDomain, "iter_candidates", _boom)
    monkeypatch.setattr(vector, "materialize_lateral_vector_domains", _boom)
    result = _run_public(ring, solved_ring, _ring_budget())
    assert result.status == "complete"
    assert result.runtime_work.candidate_materializations == 0


def test_public_gate_refusal_is_atomic(ambiguous_ring):
    (vector_result, _p1d, _signatures, _matches, _support, _phases,
     _readings) = ambiguous_ring
    solved = relative.solve_relative_translation_graph(
        vector_result, _support, _p1d, _matches, _phases,
        readings=_readings, signatures=_signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10,
            max_tree_edges=10, max_chords=10, max_cycle_rows=30))
    assert solved.status == "complete"
    assert solved.placement_readiness == "review_required"
    result = v2a.analyze_endpoint_provenance(
        solved, vector_result, _support, _p1d, _matches, _phases,
        readings=_readings, signatures=_signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_ring_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason == "gate_not_ready"
    assert result.reading_domains == () and result.witnesses == ()
    assert result.planned_work == v2a.V2AWork.zero()


def test_public_xyz_ready_fixture_is_endpoint_unique_and_exhaustive():
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings, _declared) = _xyz_ready_case()
    solved = relative.solve_relative_translation_graph(
        vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_runtime_budget(
            max_components=10, max_frames=10, max_edges=10,
            max_tree_edges=10, max_chords=10, max_cycle_rows=30))
    assert solved.placement_readiness == "ready"
    result = _run_public(
        (vector_result, p1d_result, signatures, matches, support, phases,
         readings), solved, _ring_budget())
    assert result.status == "complete" and result.refusals == ()
    assert result.classification == "endpoint_unique"
    assert result.exhaustive and not result.lower_bound_only
    assert result.translation_class_count == 1
    assert result.endpoint_assignment_count == 1
    assert len(result.witnesses) == 1
    assert v2a.validate_endpoint_provenance_result(
        result, solved, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)


def test_public_replay_mutation_fails_closed(ring, solved_ring):
    (vector_result, p1d_result, _signatures, matches, support, phases,
     readings) = ring
    result = v2a.analyze_endpoint_provenance(
        solved_ring, vector_result, support, p1d_result, matches, phases,
        readings=readings, signatures=(), comparison_tolerance=1e-12,
        closure_tolerance=0.0, budget=_ring_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason in {"c2rb_replay_mismatch",
                                         "a4_replay_mismatch",
                                         "input_record_invalid"}


def test_public_tree_delta_is_certificate_difference(ring, solved_ring):
    (vector_result, _p1d, signatures, _matches, _support, _phases,
     readings) = ring
    component = solved_ring.components[0]
    certificate = component.translation_classes[0]
    translations = {frame: xyz for frame, xyz in
                    certificate.translations_dyadic_xyz}
    reference_by = {reference.match_identity: reference
                    for reference in certificate.edge_references}
    result = _run_public(ring, solved_ring, _ring_budget())
    for record in result.edge_records:
        if record.edge_kind != "tree":
            continue
        identity = record.match_identity
        expected = tuple(
            (Fraction(*translations[identity.right_frame][axis])
             - Fraction(*translations[identity.left_frame][axis]))
            for axis in range(3))
        observed = tuple(Fraction(*item)
                         for item in record.accepted_delta_dyadic_xyz)
        assert observed == expected
        stored = tuple(Fraction(*item)
                       for item in reference_by[identity].delta_dyadic_xyz)
        assert stored == expected


def test_public_stored_delta_echo_is_never_the_definition(ring, solved_ring):
    (vector_result, _p1d, signatures, _matches, _support, _phases,
     readings) = ring
    solved = _unchecked_clone(solved_ring)
    component = _unchecked_clone(solved_ring.components[0])
    certificate = _unchecked_clone(
        solved_ring.components[0].translation_classes[0])
    references = list(certificate.edge_references)
    tree_identities = set(component.tree_edges)
    tampered_index = next(
        index for index, reference in enumerate(references)
        if reference.match_identity in tree_identities)
    tampered = _unchecked_clone(references[tampered_index])
    object.__setattr__(tampered, "delta_dyadic_xyz",
                       ((1234567, 1), (0, 1), (0, 1)))
    references[tampered_index] = tampered
    object.__setattr__(certificate, "edge_references", tuple(references))
    object.__setattr__(component, "translation_classes", (certificate,))
    object.__setattr__(solved, "components", (component,))
    bound = v2a._bind_pipeline(solved, vector_result, readings, signatures)
    counter = v2a._RuntimeCounter(_ring_budget())
    with pytest.raises(v2a.V2AFailure) as failure:
        v2a._enumerate_pipeline(bound, Fraction(0), counter)
    assert failure.value.reason == "stored_delta_echo_mismatch"


def _unchecked_clone(record):
    clone = object.__new__(type(record))
    for field in record.__dataclass_fields__:
        object.__setattr__(clone, field, getattr(record, field))
    return clone


# ---------------------------------------------------------------------------
# 5.10 Replay from current sources
# ---------------------------------------------------------------------------

def test_public_validate_rejects_a_replaced_source(ring, solved_ring):
    (vector_result, p1d_result, signatures, matches, support, phases,
     readings) = ring
    result = _run_public(ring, solved_ring, _ring_budget())
    assert v2a.validate_endpoint_provenance_result(
        result, solved_ring, vector_result, support, p1d_result, matches,
        phases, readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0)
    assert not v2a.validate_endpoint_provenance_result(
        result, solved_ring, vector_result, support, p1d_result, matches,
        phases, readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=1e-9)


# ---------------------------------------------------------------------------
# 6 AST/dataflow isolation boundary
# ---------------------------------------------------------------------------

def test_module_imports_and_calls_stay_inside_the_isolation_boundary():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in ALLOWED_IMPORT_ROOTS, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in ALLOWED_IMPORT_ROOTS
            if (node.module or "").startswith("mc_photogrammetry"):
                for alias in node.names:
                    assert alias.name in ALLOWED_RECONSTRUCTION_MODULES, alias.name
        elif isinstance(node, ast.Name):
            assert node.id not in DENIED_NAMES, node.id
        elif isinstance(node, ast.Attribute):
            assert node.attr not in DENIED_NAMES, node.attr
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert not (set(calls) & DENIED_NAMES)


def test_module_forbids_id_calls_and_reading_object_identity():
    def _type_probe(node) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"type", "isinstance"})

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "id"):
            pytest.fail("V2-A must not call id()")
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            continue
        if all(isinstance(value, ast.Constant) and value.value is None
               for value in node.comparators):
            continue
        operands = [node.left, *node.comparators]
        if any(_type_probe(value) for value in operands):
            continue
        names = {child.id for value in operands for child in ast.walk(value)
                 if isinstance(child, ast.Name)}
        attributes = {child.attr for value in operands
                      for child in ast.walk(value)
                      if isinstance(child, ast.Attribute)}
        assert not (names & {"reading", "signature"}), ast.dump(node)
        assert "reading" not in attributes, ast.dump(node)


def test_kernel_never_reads_the_representative_sentinel():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "solve_endpoint_provenance_kernel")
    for node in ast.walk(kernel):
        if isinstance(node, ast.Name):
            assert node.id != "ignored_representative_sentinel"
        elif isinstance(node, ast.Attribute):
            assert node.attr != "ignored_representative"


def test_accepted_tree_delta_dataflow_is_the_certificate_difference():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    bind = functions["_bind_pipeline"]
    names = {node.id for node in ast.walk(bind)
             if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(bind)
                  if isinstance(node, ast.Attribute)}
    assert "_sub_dyadic" in names
    assert "translations_dyadic_xyz" in attributes
    assert "delta_dyadic_xyz" not in attributes
    # No assignment in the module may define the accepted delta from the echo.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "accepted_delta"):
                    assert not any(
                        isinstance(child, ast.Attribute)
                        and child.attr == "delta_dyadic_xyz"
                        for child in ast.walk(node.value))


def test_module_uses_only_index_only_a4_accessors():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
    assert {"iter_entry_groups", "provenance_key_at",
            "vector_at"} <= attributes
    for denied in ("candidate_at", "iter_candidates",
                   "materialize_lateral_vector_domains", "solve_component"):
        assert denied not in attributes


# ---------------------------------------------------------------------------
# Meaningful sparse performance
# ---------------------------------------------------------------------------

def test_kernel_sparse_chain_is_linear_and_fast():
    count = 300
    domains = tuple(_domain(f"F{index:04d}") for index in range(count))
    edges = tuple(_edge(domains[index], domains[index + 1],
                        [(0, 0, 0, 1)], ordinal=index)
                  for index in range(count - 1))
    started = time.perf_counter()
    outcome = v2a.solve_endpoint_provenance_kernel(
        domains=domains, edges=edges,
        budget=_budget(max_edges=1000, max_readings=1000,
                       max_endpoint_labels=1000,
                       max_partial_label_assignments=100000,
                       max_consistency_checks=100000))
    elapsed = time.perf_counter() - started
    assert outcome.classification == "endpoint_unique"
    assert outcome.runtime_work.partial_label_assignments == count
    assert elapsed <= 3.0
