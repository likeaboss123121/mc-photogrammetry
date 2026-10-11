"""Focused hostile controls for the validating production universe assembler.

The genuine input recipe is the accepted ``test_plane_placement`` fixture, used
here **only as test input**: :func:`test_plane_placement._build_universe` builds
one real mixed-normal 15-reading phase-0 universe through the accepted public
producers.  Nothing from that recipe is copied into the assembler module.

Every hostile case takes that genuine universe and either tampers one nested
record, swaps a record across a second genuine universe, removes/partialises the
anchor axes, replaces the anchor budget or trims the usable-mask coverage.  The
assembler must return one typed, atomic
:class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementRefusal` and
never a partial universe.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_phase_anchor as v2b
from mc_photogrammetry.reconstruction import plane_placement_assembly as assembly

from test_plane_placement import (
    EXPECTED_ANCHOR_DIGEST,
    _budget as _placement_budget,
    _build_supports,
    _build_universe,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "mc_photogrammetry" / "reconstruction"
          / "plane_placement_assembly.py")


def _assemble(universe, **overrides):
    """Call the assembler with every explicit record from ``universe``."""
    fields = {
        "axis_support": universe.axis_support,
        "p1d_result": universe.p1d_result,
        "vector_result": universe.vector_result,
        "c2rb_result": universe.c2rb_result,
        "v2a_result": universe.v2a_result,
        "v2b_result": universe.v2b_result,
        "readings": universe.readings,
        "signatures": universe.signatures,
        "matches": universe.matches,
        "phase_evidence": universe.phase_evidence,
        "usable_masks": universe.usable_masks,
        "split_sources": universe.split_sources,
        "anchor_budget": universe.anchor_budget,
        "support_budget": universe.support_budget,
        "comparison_tolerance": universe.comparison_tolerance,
        "closure_tolerance": universe.closure_tolerance,
    }
    fields.update(overrides)
    return assembly.assemble_placement_universe(**fields)


def _unchecked(record, **changes):
    """Constructor-bypassed clone used to simulate a tampered nested record."""
    clone = object.__new__(type(record))
    for field in record.__dataclass_fields__:
        object.__setattr__(
            clone, field, changes.get(field, getattr(record, field)))
    return clone


def _assert_refusal(outcome, reason=None, field=None):
    assert isinstance(outcome, placement.PlacementRefusal), type(outcome).__name__
    assert outcome.reason in assembly.PLACEMENT_ASSEMBLY_REFUSAL_REASONS
    assert outcome.work is None
    record = outcome.to_record()
    assert record["refusal_sha256"] == outcome.refusal_sha256
    if reason is not None:
        assert outcome.reason == reason, (outcome.reason, outcome.detail)
    if field is not None:
        assert outcome.detail.get("field") == field, outcome.detail
    return outcome


@pytest.fixture(scope="module")
def genuine():
    return _build_universe()


# ---------------------------------------------------------------------------
# Valid replay through the accepted downstream seam
# ---------------------------------------------------------------------------

def test_valid_universe_assembles_and_places(genuine):
    outcome = _assemble(genuine)
    assert isinstance(outcome, placement.PlacementUniverse)
    # The assembler carries the caller's exact immutable objects, never copies.
    for name in placement.PlacementUniverse.__dataclass_fields__:
        actual = getattr(outcome, name)
        expected = getattr(genuine, name)
        if name in ("comparison_tolerance", "closure_tolerance"):
            assert actual == expected
        else:
            assert actual is expected
    assert outcome.readings is genuine.readings
    assert outcome.usable_masks is genuine.usable_masks
    assert outcome.split_sources is genuine.split_sources
    assert outcome.anchor_budget is genuine.anchor_budget

    # The assembled universe is accepted by the same placement seam as the
    # genuine fixture, and resolves the same frozen placement record.
    supports = _build_supports(outcome)
    assembled = placement.resolve_face_placement(
        universe=outcome, supports=supports, budget=_placement_budget())
    direct = placement.resolve_face_placement(
        universe=genuine, supports=_build_supports(genuine),
        budget=_placement_budget())
    assert isinstance(assembled, placement.ResolvedFacePlacement)
    assert isinstance(direct, placement.ResolvedFacePlacement)
    assert assembled.placement_sha256 == direct.placement_sha256
    assert len(assembled.faces) == 15


def test_valid_replay_does_not_mutate_upstream_records(genuine):
    before = {
        "c2rb": genuine.c2rb_result.to_json_bytes(),
        "v2a": genuine.v2a_result.to_json_bytes(),
        "v2b": genuine.v2b_result.to_json_bytes(),
        "signatures": tuple(signature.cells.tobytes()
                            for signature in genuine.signatures),
    }
    outcome = _assemble(genuine)
    assert isinstance(outcome, placement.PlacementUniverse)
    assert genuine.c2rb_result.to_json_bytes() == before["c2rb"]
    assert genuine.v2a_result.to_json_bytes() == before["v2a"]
    assert genuine.v2b_result.to_json_bytes() == before["v2b"]
    assert tuple(signature.cells.tobytes()
                 for signature in genuine.signatures) == before["signatures"]


def test_anchor_is_the_declared_all_axis_phase_zero_anchor(genuine):
    outcome = _assemble(genuine)
    assert isinstance(outcome, placement.PlacementUniverse)
    alpha = tuple(placement._canonical_dyadic(
        outcome.v2b_result.prediction.axes[axis].alpha_dyadic)
        for axis in (0, 1, 2))
    assert placement.placement_anchor_digest(
        tuple(placement._reduce(value) for value in alpha)) \
        == EXPECTED_ANCHOR_DIGEST


# ---------------------------------------------------------------------------
# Tampered nested records
# ---------------------------------------------------------------------------

def test_tampered_nested_signature_record_refused(genuine):
    signature = genuine.signatures[0]
    tampered = replace(signature,
                       coverage=np.zeros_like(signature.coverage))
    outcome = _assemble(genuine, signatures=(tampered,) + genuine.signatures[1:])
    _assert_refusal(outcome, "replay_mismatch", "p1d_result")


def test_tampered_nested_shift_domain_record_refused(genuine):
    domain = genuine.p1d_result.domains[0]
    tampered_domain = _unchecked(domain, entries=domain.entries[:-1])
    tampered = _unchecked(
        genuine.p1d_result,
        domains=(tampered_domain,) + genuine.p1d_result.domains[1:])
    outcome = _assemble(genuine, p1d_result=tampered)
    _assert_refusal(outcome, "replay_mismatch", "vector_result")


def test_tampered_nested_match_record_refused(genuine):
    match = genuine.matches[0]
    tampered = replace(
        match, quadrant_of_right=(match.quadrant_of_right + 1) % 4)
    outcome = _assemble(genuine, matches=(tampered,) + genuine.matches[1:])
    _assert_refusal(outcome, "replay_mismatch")


def test_tampered_nested_v2b_axis_record_refused(genuine):
    prediction = genuine.v2b_result.prediction
    axis = prediction.axes[0]
    tampered_axis = _unchecked(axis, alpha_dyadic=None)
    tampered_prediction = _unchecked(
        prediction, axes=(tampered_axis,) + prediction.axes[1:])
    tampered = _unchecked(genuine.v2b_result, prediction=tampered_prediction)
    outcome = _assemble(genuine, v2b_result=tampered)
    _assert_refusal(outcome)


# ---------------------------------------------------------------------------
# Cross-bound digests and swapped sources
# ---------------------------------------------------------------------------

def test_cross_bound_v2a_digest_refused(genuine):
    tampered = _unchecked(genuine.v2a_result, c2rb_result_sha256="0" * 64)
    outcome = _assemble(genuine, v2a_result=tampered)
    _assert_refusal(outcome, "provenance_mismatch", "v2a_result.c2rb_result")


def test_cross_bound_vector_digest_refused(genuine):
    tampered = _unchecked(genuine.c2rb_result, vector_result_sha256="1" * 64)
    outcome = _assemble(genuine, c2rb_result=tampered)
    _assert_refusal(outcome, "provenance_mismatch", "c2rb_result.vector_result")


def test_cross_bound_reading_domains_refused(genuine):
    tampered = _unchecked(genuine.v2a_result, canonical_universe_sha256="2" * 64)
    outcome = _assemble(genuine, v2a_result=tampered)
    _assert_refusal(outcome, "provenance_mismatch",
                    "v2a_result.canonical_universe")


def test_second_genuine_universe_split_sources_refused(genuine):
    other = _build_universe(d_bins=(11, 14))
    assert other.v2b_result.prediction is not None
    assert other.split_sources != genuine.split_sources
    outcome = _assemble(genuine, split_sources=other.split_sources)
    _assert_refusal(outcome, "replay_mismatch")


# ---------------------------------------------------------------------------
# Anchor readiness
# ---------------------------------------------------------------------------

def test_missing_anchor_prediction_refused(genuine):
    outcome = _assemble(
        genuine, v2b_result=_unchecked(genuine.v2b_result, prediction=None))
    _assert_refusal(outcome, "anchor_gate_not_unique")


def test_partial_anchor_axis_refused(genuine):
    prediction = genuine.v2b_result.prediction
    axis = prediction.axes[0]
    tampered_axis = _unchecked(axis, status="abstained", alpha_dyadic=None)
    tampered_prediction = _unchecked(
        prediction, axes=(tampered_axis,) + prediction.axes[1:])
    tampered = _unchecked(genuine.v2b_result, prediction=tampered_prediction)
    outcome = _assemble(genuine, v2b_result=tampered)
    refusal = _assert_refusal(outcome, "anchor_axis_unresolved")
    assert refusal.detail.get("axes") == [0]


def test_unsupported_required_frame_axis_refused(genuine, monkeypatch):
    """A replay-valid universe whose required axis is unconstrained is refused.

    The accepted records bind ``axis_support`` by digest, so an inconsistent
    support is normally refused earlier as a replay mismatch.  This control
    exercises the assembler's own final placement-readiness guard directly: it
    returns the accepted replay tuple with one required ``(frame, axis)``
    translation marked unsupported.
    """
    real = placement._replay_universe

    def with_unsupported_axis(universe):
        component, certificate, translations, supported, alpha, binding = real(
            universe)
        frame = certificate.translations_dyadic_xyz[0][0]
        degraded = dict(supported)
        degraded[(frame, 0)] = False
        return component, certificate, translations, degraded, alpha, binding

    monkeypatch.setattr(placement, "_replay_universe", with_unsupported_axis)
    outcome = _assemble(genuine)
    _assert_refusal(outcome, "translation_unsupported")


# ---------------------------------------------------------------------------
# Budget overrun and foreign budget binding
# ---------------------------------------------------------------------------

def test_overrun_anchor_budget_refused(genuine):
    caps = genuine.anchor_budget.to_record()
    caps.pop("budget_sha256")
    caps["max_hypotheses_per_axis"] = 1
    overrun = v2b.make_anchor_budget(**caps)
    assert overrun.budget_sha256 != genuine.anchor_budget.budget_sha256
    outcome = _assemble(genuine, anchor_budget=overrun)
    _assert_refusal(outcome, "provenance_mismatch", "v2b_result.anchor_budget")


def test_constructor_bypassed_anchor_budget_refused(genuine):
    """A frozen ``max_readings`` mutation under a stale digest never binds.

    ``V2BAnchorBudget`` is frozen, so the tamper uses ``object.__setattr__`` and
    leaves ``budget_sha256`` stale.  The prediction still carries that same stale
    digest, so the cross-record binding alone cannot see the mutation; only the
    exact constructor replay refuses it before the digest binding is reached.
    """
    tampered = _unchecked(
        genuine.anchor_budget,
        max_readings=genuine.anchor_budget.max_readings + 1)
    assert tampered.max_readings != genuine.anchor_budget.max_readings
    assert tampered.budget_sha256 == genuine.anchor_budget.budget_sha256
    assert (genuine.v2b_result.prediction.budget_sha256
            == tampered.budget_sha256)
    outcome = _assemble(genuine, anchor_budget=tampered)
    _assert_refusal(outcome, "provenance_mismatch", "anchor_budget")


def test_constructor_bypassed_support_budget_refused(genuine):
    """A frozen ``max_cells`` mutation under a stale digest never reaches use.

    ``support_budget`` has no cross-record digest carried by any accepted record,
    so without the constructor replay a bypassed ``max_cells`` would be carried
    into the candidate universe unchanged.
    """
    tampered = _unchecked(
        genuine.support_budget,
        max_cells=genuine.support_budget.max_cells + 1)
    assert tampered.max_cells != genuine.support_budget.max_cells
    assert tampered.budget_sha256 == genuine.support_budget.budget_sha256
    outcome = _assemble(genuine, support_budget=tampered)
    _assert_refusal(outcome, "provenance_mismatch", "support_budget")


def test_malformed_budgets_refused(genuine):
    _assert_refusal(_assemble(genuine, anchor_budget=None), "invalid_input",
                    "anchor_budget")
    _assert_refusal(_assemble(genuine, support_budget=None), "invalid_input",
                    "support_budget")


# ---------------------------------------------------------------------------
# Read/signature/match/mask identity bindings
# ---------------------------------------------------------------------------

def test_unbound_usable_mask_identity_refused(genuine):
    outcome = _assemble(
        genuine,
        usable_masks=(("0" * 64, genuine.usable_masks[0][1]),)
        + genuine.usable_masks[1:])
    _assert_refusal(outcome, "provenance_mismatch", "usable_masks.identity")


def test_wrong_shape_usable_mask_refused(genuine):
    outcome = _assemble(
        genuine,
        usable_masks=((genuine.usable_masks[0][0],
                       np.zeros((3, 3), dtype=bool)),)
        + genuine.usable_masks[1:])
    _assert_refusal(outcome, "provenance_mismatch", "usable_masks.shape")


def test_missing_usable_mask_coverage_refused(genuine):
    outcome = _assemble(genuine, usable_masks=genuine.usable_masks[:-1])
    _assert_refusal(outcome, "missing_provenance", "usable_masks.coverage")


def test_unbound_signature_reading_refused(genuine):
    foreign = _build_universe(d_bins=(11, 14))
    outcome = _assemble(
        genuine, signatures=(foreign.signatures[0],) + genuine.signatures[1:])
    _assert_refusal(outcome, "provenance_mismatch", "signatures.reading")


def test_unbound_match_endpoint_refused(genuine):
    match = genuine.matches[0]
    # Three readings per frame, so +3 is always outside the frame's index range.
    tampered = replace(match, left_plane=match.left_plane + 3)
    outcome = _assemble(genuine, matches=(tampered,) + genuine.matches[1:])
    _assert_refusal(outcome, "missing_provenance", "matches.endpoint")


def test_malformed_split_source_refused(genuine):
    outcome = _assemble(
        genuine, split_sources=(("not-a-source",),) + genuine.split_sources[1:])
    _assert_refusal(outcome, "invalid_input", "split_sources.entry")


def test_nonfinite_tolerances_refused(genuine):
    _assert_refusal(_assemble(genuine, comparison_tolerance=float("nan")),
                    "invalid_input", "comparison_tolerance")
    _assert_refusal(_assemble(genuine, closure_tolerance="0.0"),
                    "invalid_input", "closure_tolerance")
    _assert_refusal(_assemble(genuine, comparison_tolerance=-1.0),
                    "invalid_input", "comparison_tolerance")


def test_malformed_containers_refused(genuine):
    _assert_refusal(_assemble(genuine, axis_support=None), "invalid_input",
                    "axis_support")
    _assert_refusal(_assemble(genuine, readings=()), "invalid_input", "readings")
    _assert_refusal(_assemble(genuine, matches=()), "invalid_input", "matches")
    _assert_refusal(_assemble(genuine, phase_evidence=()), "invalid_input",
                    "phase_evidence")


# ---------------------------------------------------------------------------
# Boundary and isolation
# ---------------------------------------------------------------------------

def test_refusals_never_carry_a_partial_universe(genuine):
    """A hostile outcome is a typed refusal, never a half-built universe."""
    hostile = [
        _assemble(genuine, readings=()),
        _assemble(genuine, usable_masks=genuine.usable_masks[:-1]),
        _assemble(
            genuine, v2b_result=_unchecked(genuine.v2b_result, prediction=None)),
        _assemble(
            genuine, v2a_result=_unchecked(genuine.v2a_result,
                                           canonical_universe_sha256="f" * 64)),
    ]
    for outcome in hostile:
        assert isinstance(outcome, placement.PlacementRefusal)
        assert not isinstance(outcome, placement.PlacementUniverse)
        assert set(outcome.detail) & {"faces", "cells", "blocks"} == set()


def test_module_boundary_is_in_memory_only():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "math", "numpy", "mc_photogrammetry"}
    denied_names = {
        "open", "Path", "os", "sys", "subprocess", "socket", "tempfile",
        "pathlib", "shutil", "io", "BlockVolume", "PlaneVolumeResult",
        "resolve_face_placement", "manifest", "f3", "bounds", "coordinates",
        "direction", "directions", "cli", "store",
    }
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert not (names & denied_names), sorted(names & denied_names)


def test_assembler_declares_its_closed_refusal_vocabulary():
    assert assembly.PLACEMENT_ASSEMBLY_REFUSAL_REASONS
    assert set(assembly.PLACEMENT_ASSEMBLY_REFUSAL_REASONS) <= set(
        placement.V2_PLACEMENT_REFUSAL_REASONS)
    assert assembly.PLACEMENT_ASSEMBLY_PRODUCER_VERSION
