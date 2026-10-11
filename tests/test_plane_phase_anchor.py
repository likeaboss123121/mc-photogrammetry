"""Independent controls for the isolated V2-B physical phase anchor.

The public fixture builds a genuine gate-passing bundle through the accepted
P0 producer, the V2-A endpoint-provenance analysis and the C2-R-B relative
translation solver.  Expected values are declared by the fixture, never read
back from the module under test.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import lattice_planes as lp
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_phase_anchor as anchor
from mc_photogrammetry.reconstruction import plane_relative_translation as relative
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain

from test_c2r_vector_domain_a3_proof import (
    _canonical_identity,
    _reading,
    _signature,
)
from test_plane_relative_translation import _runtime_budget
from test_plane_shift_domain import _budget as _shift_budget
from test_plane_vector_domain import _large_budget, _phase_for


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mc_photogrammetry" / "reconstruction" / \
    "plane_phase_anchor.py"

H = W = 16
EXTENT = 4.0
SAMPLES = 16
DISTANCE = 0.5
B_VALUE = 1.0
PATCH_STEP = lp.lattice.rectified_pixel_step(EXTENT, SAMPLES)
PERIOD = B_VALUE / (DISTANCE * PATCH_STEP)
XYZ_PERIOD = 127.0 / 32.0


def _v2a_budget(**overrides):
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


def _p0_budget():
    samples = H * W
    return lp._make_budget({
        "max_lanes_per_axis": max(H, W),
        "max_patch_samples": samples,
        "max_inline_patch_samples": samples,
        "max_g_samples_per_axis": samples,
        "max_d_samples_per_axis": samples,
        "max_guard_samples_per_axis": samples,
        "max_profile_terms": H + W,
        "max_d_profile_terms": 3 * (H + W),
        "max_payload_bytes": (2 * len(
            lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION) + 48 + 24 * (H + W)),
        "max_source_records": 1000,
        "max_replay_checks": 1000,
    })


def _xyz_p0_budget():
    samples = 128
    return lp._make_budget({
        "max_lanes_per_axis": samples,
        "max_patch_samples": samples * samples,
        "max_inline_patch_samples": samples * samples,
        "max_g_samples_per_axis": samples * samples,
        "max_d_samples_per_axis": samples * samples,
        "max_guard_samples_per_axis": samples * samples,
        "max_profile_terms": 2 * samples,
        "max_d_profile_terms": 6 * samples,
        "max_payload_bytes": 2 * len(lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION)
                             + 48 + 24 * (2 * samples),
        "max_source_records": 100,
        "max_replay_checks": 100,
    })


def _xyz_split_patch():
    patch = np.zeros((128, 128), dtype=np.float64)
    # The frozen selector assigns lanes [64,128) to G on both axes.  The final
    # endpoint has exact residue zero at period 127/32.  D-only energy stays in
    # the disjoint [0,64) x [0,64) held-out block.
    patch[127, 127] = 1.0
    patch[0, 0] = 4.0
    patch[4, 4] = 4.0
    return patch


def test_xyz_period_p0_produces_exact_g_phase_and_held_out_d_evidence():
    extent = 1.6
    geometry = lp.SplitPhaseGeometry(
        centre=(extent, extent), extent=extent, samples=128,
        rectified_pixel_step=lp.lattice.rectified_pixel_step(extent, 128),
        period_pixels=XYZ_PERIOD)
    patch = _xyz_split_patch()
    budget = _xyz_p0_budget()
    result = lp.produce_split_phase_evidence(
        patch, geometry=geometry, frame="A", plane_index=0,
        reading_axis=0, source_region_id="a" * 64,
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
    assert result.status == "complete" and result.refusals == ()
    generation = result.generation_evidence[0]
    assert generation.phase_uv_pixels == (0.0, 0.0)
    assert generation.unavailable_reason_uv == (None, None)
    assert generation.period_pixels == XYZ_PERIOD
    assert {(lane.local_axis, lane.role, lane.lane_index_start,
             lane.lane_index_stop) for lane in result.lane_sets} == {
        ("u", "G", 64, 128), ("u", "D", 0, 64),
        ("v", "G", 64, 128), ("v", "D", 0, 64),
    }
    for seam in result.seam_profile_evidence:
        samples = np.frombuffer(base64.b64decode(
            seam.profile_samples_float64_le_base64), dtype="<f8")
        assert samples.shape == (128,) and float(samples.max()) > 0.0
        assert seam.coverage["valid"] == seam.coverage["total"]
    assert lp.validate_split_phase_evidence(
        result, patch, geometry=geometry,
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)


def _anchor_budget(**overrides):
    limits = {
        "max_readings": 100, "max_observations": 10000, "max_frames": 100,
        "max_hypotheses_per_axis": 100, "max_hypotheses_total": 1000,
        "max_contributor_refs_per_hypothesis": 1000,
        "max_contributor_refs_total": 10000, "max_source_records": 100,
        "max_boundary_interior_terms": 100000, "max_d_profile_terms": 100000,
        "max_payload_bytes": 100000, "max_replay_crosslinks": 100000,
        "max_pixel_samples": 100000,
        "max_held_out_samples_per_reading": 100000,
        "max_split_reconstructions": 100,
        "max_accepted_frame_supports": 100000,
    }
    limits.update(overrides)
    return anchor.make_anchor_budget(**limits)


def _delta_patch():
    """Structural patch: a single G-lane impulse gives exact phase zero."""
    patch = np.zeros((H, W), dtype=np.float64)
    patch[0, 0] = 1.0
    return patch


def _geometry():
    return lp.SplitPhaseGeometry(
        centre=(EXTENT, EXTENT), extent=EXTENT, samples=SAMPLES,
        rectified_pixel_step=PATCH_STEP, period_pixels=PERIOD)


def _d_shaped_patch():
    """Patch with a held-out D bump at local residue zero on both axes.

    Only the ``D x D`` block is touched, so neither ``G`` profile changes and
    the structural phase evidence is preserved exactly.
    """
    patch = _delta_patch().copy()
    # Bin 11 and bin 15 both fall inside the inclusive ell=0 band and outside
    # the ell=1/2 band for this period/index step.
    patch[11, 11] = 4.0
    patch[15, 15] = 4.0
    return patch


def _produce(patch, frame, plane_index=0, axis=2):
    return lp.produce_split_phase_evidence(
        patch, geometry=_geometry(), frame=frame, plane_index=plane_index,
        reading_axis=axis, source_region_id=None,
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


@pytest.fixture(scope="module")
def bundle():
    return _build_bundle()


def _build_bundle(g_only=True, axis=2):
    translations = {"R0": (0, 0), "R1": (2, 0), "R2": (3, 2),
                    "R3": (1, 3), "R4": (-1, 2)}
    geometry = _geometry()
    structural = _delta_patch()
    first = _produce(structural, "R0", 0, axis)
    assert first.status == "complete", first.refusals
    evidence = _g_evidence(first)
    assert evidence.phase_uv_pixels == (0.0, 0.0)

    basis = np.eye(2, dtype=np.float64)
    readings = []
    signatures = []
    for frame, point in translations.items():
        base = _reading(frame, axis, B_VALUE, DISTANCE, 1.0, (0, 0),
                        (0.0, 0.0), samples=SAMPLES, extent=EXTENT)
        assert abs(base.period_pixels - PERIOD) < 1e-12
        reading = replace(base, phase_evidence=evidence,
                          phase_pixels=(0.0, 0.0)) if g_only else base
        readings.append(reading)
        signature = _signature(
            reading, B_VALUE, basis, np.asarray(point, dtype=np.float64))
        coverage = np.zeros(signature.coverage.shape, dtype=bool)
        row = (coverage.shape[0] - 3) // 2
        column = (coverage.shape[1] - 3) // 2
        coverage[row:row + 3, column:column + 3] = True
        signatures.append(replace(signature, coverage=coverage))
    readings = tuple(readings)
    signatures = tuple(signatures)
    by_frame = {value.reading.frame: value for value in signatures}

    matches = []
    for index in range(len(translations)):
        left, right = f"R{index}", f"R{(index + 1) % len(translations)}"
        pair = plane_graph.match_plane_pair(
            by_frame[left], by_frame[right], quadrant=0, max_shift_cells=4)
        assert pair is not None
        matches.append(replace(
            pair,
            left_plane=_canonical_identity(readings, by_frame[left].reading)[1],
            right_plane=_canonical_identity(readings, by_frame[right].reading)[1]))
    matches = tuple(matches)

    solution = plane_graph.ComponentSolution(
        frames=tuple(sorted(translations)),
        quadrants={frame: 0 for frame in translations},
        translations={frame: (0.0, 0.0, 0.0) for frame in translations},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={_canonical_identity(readings, reading): B_VALUE
                           for reading in readings},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None
    p1d = plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=_shift_budget(), readings=readings)
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
        budget=_runtime_budget(max_components=10, max_frames=10, max_edges=10,
                               max_tree_edges=10, max_chords=10,
                               max_cycle_rows=30))
    assert solved.status == "complete", solved.budget_failure_reason
    assert solved.placement_readiness == "ready"
    component = solved.components[0]
    assert component.outcome == "unique_validated"
    assert component.exhaustive is True
    assert len(component.translation_classes) == 1

    v2a_result = v2a.analyze_endpoint_provenance(
        solved, vector_result, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        budget=_v2a_budget())
    assert v2a_result.status == "complete", [
        value.to_record() for value in v2a_result.refusals]
    assert v2a_result.classification == "endpoint_unique", v2a_result.classification
    assert v2a_result.endpoint_assignment_count == 1
    assert len(v2a_result.witnesses) == 1

    held_out = _d_shaped_patch()
    split_sources = tuple(
        (_produce(held_out, reading.frame, 0, reading.axis), held_out.copy(),
         _p0_budget())
        for reading in readings)
    return {
        "c2rb": solved, "vector_result": vector_result, "support": support,
        "p1d": p1d, "matches": matches, "phases": phases,
        "readings": readings, "signatures": signatures,
        "v2a_result": v2a_result, "split_sources": split_sources,
        "structural": structural, "held_out": held_out,
        "translations": translations, "axis": axis,
    }


@pytest.fixture(scope="module", params=(0, 1, 2))
def axis_bundle(request):
    """The public fixture for each reading normal axis (all three normals)."""
    return _build_bundle(axis=request.param)


def _run_anchor(bundle, budget=None, **overrides):
    budget = budget or _anchor_budget(**overrides)
    return anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=bundle["split_sources"], budget=budget)


# ---------------------------------------------------------------------------
# Accepted calibration binding
# ---------------------------------------------------------------------------

def test_builtin_calibration_binds_accepted_record():
    payload = json.loads((ROOT / "exports" / "calibration" / "v2b-p0" /
                          "calibration.json").read_text(encoding="utf-8"))
    record = lp.SplitPhaseCalibration(
        calibration_version=payload["calibration_version"],
        partition_config_sha256=payload["partition_config_sha256"],
        tau_band=tuple(payload["tau_band"]),
        tau_margin=tuple(payload["tau_margin"]),
        tau_energy=tuple(payload["tau_energy"]),
        min_held_out_samples=payload["min_held_out_samples"],
        min_distinct_frames_per_axis=payload["min_distinct_frames_per_axis"],
        tie_break=payload["tie_break"],
        target_power=tuple(payload["target_power"]),
        calibration_corpus_sha256=payload["calibration_corpus_sha256"],
        evaluation_corpus_sha256=payload["evaluation_corpus_sha256"],
        calibration_sample_sha256s=tuple(payload["calibration_sample_sha256s"]),
        evaluation_sample_sha256s=tuple(payload["evaluation_sample_sha256s"]),
        null_control_ids=tuple(payload["null_control_ids"]),
        seam_control_ids=tuple(payload["seam_control_ids"]),
        calibration_grid_sha256=payload["calibration_grid_sha256"],
        calibration_sha256=payload["calibration_sha256"])
    bound = anchor.V2BAnchorCalibration.from_split_phase_calibration(record)
    assert bound is anchor.V2B_BUILTIN_CALIBRATION
    assert bound.tau_band == (1, 8)
    assert bound.tau_margin == (1, 8)
    assert bound.tau_energy == (0, 1)
    assert bound.min_held_out_samples == 1
    assert bound.min_distinct_frames_per_axis == 2
    assert bound.source_calibration_sha256 == \
        "dc4a3ac698c80c915cd4e61933044c37766e5be394e56d220e5bda80d2fbbc06"


def test_calibration_tamper_is_refused():
    payload = json.loads((ROOT / "exports" / "calibration" / "v2b-p0" /
                          "calibration.json").read_text(encoding="utf-8"))
    payload["tau_margin"] = [1, 4]
    with pytest.raises(ValueError):
        anchor.V2BAnchorCalibration.from_split_phase_calibration(
            lp.SplitPhaseCalibration(
                calibration_version=payload["calibration_version"],
                partition_config_sha256=payload["partition_config_sha256"],
                tau_band=tuple(payload["tau_band"]),
                tau_margin=tuple(payload["tau_margin"]),
                tau_energy=tuple(payload["tau_energy"]),
                min_held_out_samples=payload["min_held_out_samples"],
                min_distinct_frames_per_axis=payload[
                    "min_distinct_frames_per_axis"],
                tie_break=payload["tie_break"],
                target_power=tuple(payload["target_power"]),
                calibration_corpus_sha256=payload["calibration_corpus_sha256"],
                evaluation_corpus_sha256=payload["evaluation_corpus_sha256"],
                calibration_sample_sha256s=tuple(
                    payload["calibration_sample_sha256s"]),
                evaluation_sample_sha256s=tuple(
                    payload["evaluation_sample_sha256s"]),
                null_control_ids=tuple(payload["null_control_ids"]),
                seam_control_ids=tuple(payload["seam_control_ids"]),
                calibration_grid_sha256=payload["calibration_grid_sha256"],
                calibration_sha256=payload["calibration_sha256"]))


# ---------------------------------------------------------------------------
# Path B determination
# ---------------------------------------------------------------------------

def test_public_bundle_determines_zero_anchor_on_mapped_axes(bundle):
    result = _run_anchor(bundle)
    assert result.status in ("partial", "determined"), [
        value.to_record() for value in result.refusals]
    prediction = result.prediction
    assert prediction is not None
    assert prediction.v2a_replay_ok is True
    by_axis = {value.world_axis: value for value in prediction.axes}
    # Local ``u`` maps to world axis 0 and ``v`` to world axis 1 at quadrant 0.
    assert by_axis[0].status == "determined"
    assert by_axis[1].status == "determined"
    assert by_axis[0].alpha_dyadic == (0, 1)
    assert by_axis[1].alpha_dyadic == (0, 1)
    # World axis 2 has no mapped local axis in this fixture.
    assert by_axis[2].status == "abstained"
    assert by_axis[2].reason == "no_hypothesis"
    assert prediction.status == "partial"
    assert prediction.alpha_x == (0, 1)
    assert prediction.alpha_y == (0, 1)
    assert prediction.alpha_z is None
    assert prediction.observations
    assert all(value.zero_coverage is False
               for value in prediction.observations)
    assert prediction.planned_work.hypotheses_total >= 1
    assert prediction.runtime_work.observations == len(prediction.observations)


def test_bundle_replays_byte_identically(bundle):
    first = _run_anchor(bundle)
    second = _run_anchor(bundle)
    assert first.to_json_bytes() == second.to_json_bytes()


def test_gate_endpoint_multiple_abstains(bundle):
    original = bundle["v2a_result"]

    def _forced(classification, **changes):
        forced = object.__new__(v2a.V2AEndpointResult)
        for field in original.__dataclass_fields__:
            object.__setattr__(forced, field, getattr(original, field))
        object.__setattr__(forced, "classification", classification)
        for name, value in changes.items():
            object.__setattr__(forced, name, value)
        return forced

    multiple = _forced("endpoint_multiple", exhaustive=False,
                       lower_bound_only=True, endpoint_assignment_count=2,
                       witnesses=original.witnesses * 2)
    result = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=multiple, split_sources=bundle["split_sources"],
        budget=_anchor_budget())
    assert result.status == "abstained"
    assert result.prediction.reason == "endpoint_multiple"
    assert all(value.alpha_dyadic is None for value in result.prediction.axes)

    none = _forced("endpoint_none", exhaustive=True, lower_bound_only=False,
                   endpoint_assignment_count=0, witnesses=())
    result = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=none, split_sources=bundle["split_sources"],
        budget=_anchor_budget())
    assert result.status == "abstained"
    assert result.prediction.reason == "endpoint_none"


def test_p0_patch_mismatch_is_refused(bundle):
    sources = list(bundle["split_sources"])
    result, patch, budget = sources[0]
    wrong = patch.copy()
    wrong[0, 0] = 2.0
    sources[0] = (result, wrong, budget)
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=tuple(sources),
        budget=_anchor_budget())
    assert outcome.status == "refused"
    assert outcome.refusals[0].reason in (
        "p0_patch_mismatch", "split_replay_mismatch",
        "split_not_reconstructible")


def test_missing_p0_patch_is_refused(bundle):
    sources = list(bundle["split_sources"])
    result, _patch, budget = sources[0]
    sources[0] = (result, None, budget)
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=tuple(sources),
        budget=_anchor_budget())
    assert outcome.status == "refused"
    assert outcome.refusals[0].reason == "p0_patch_unavailable"


def test_planned_budget_exhaustion_is_atomic(bundle):
    outcome = _run_anchor(bundle, max_hypotheses_per_axis=1,
                          max_observations=1_000_000)
    # The fixture has one hypothesis per mapped axis, so a cap of one does not
    # exhaust; a cap of zero is rejected by the budget constructor instead.
    assert outcome.status in ("partial", "determined")
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=bundle["split_sources"],
        budget=_anchor_budget(max_payload_bytes=1))
    assert outcome.status == "budget_exhausted"
    assert outcome.budget_stage == "planned"
    assert outcome.prediction is None
    assert outcome.refusals[0].reason == "budget_exhausted"
    assert outcome.refusals[0].detail["budget_stage"] == "planned"


def test_observation_acceptor_rejects_without_seam_energy(bundle):
    """A uniform held-out D profile must never determine an axis."""
    flat = np.zeros((H, W), dtype=np.float64)
    flat[0, 0] = 1.0
    sources = tuple(
        (_produce(flat, reading.frame, 0, reading.axis), flat.copy(),
         _p0_budget())
        for reading in bundle["readings"])
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=sources,
        budget=_anchor_budget())
    assert outcome.status == "abstained"
    assert all(value.alpha_dyadic is None for value in outcome.prediction.axes)
    assert {value.reason for value in outcome.prediction.axes} <= {
        "no_seam_evidence", "evidence_conflict", "no_hypothesis"}


# ---------------------------------------------------------------------------
# Exact math, records, budgets
# ---------------------------------------------------------------------------

def test_dyadic_canonicalization():
    assert anchor._reduce_dyadic(Fraction(0)) == (0, 1)
    assert anchor._reduce_dyadic(Fraction(1, 2)) == (1, 2)
    assert anchor._validate_dyadic((1, 2)) is True
    assert anchor._validate_dyadic((0, 2)) is False
    assert anchor._validate_dyadic((2, 4)) is False
    with pytest.raises(ValueError):
        anchor._fraction((0, 2))
    with pytest.raises(ValueError):
        anchor._fraction((2, 4))  # even numerator is not reduced
    assert anchor._mod1(Fraction(5, 4)) == Fraction(1, 4)
    assert anchor._d1(Fraction(1, 8), Fraction(7, 8)) == Fraction(1, 4)


def test_budget_and_work_are_exact_and_boolean_free():
    budget = _anchor_budget()
    with pytest.raises(ValueError):
        anchor.V2BAnchorBudget(
            **{**{name: getattr(budget, name) for name in anchor._BUDGET_FIELDS},
               "max_readings": True},
            budget_sha256=budget.budget_sha256)
    with pytest.raises(ValueError):
        anchor._make_work({"readings": -1})
    work = anchor._make_work({"readings": 3})
    assert work.readings == 3
    assert anchor._work_within_cap(work, budget)


def test_refusal_detail_is_bounded():
    with pytest.raises(ValueError):
        anchor.V2BAnchorRefusal(
            "v2b_input_invalid", {"x": 1}, None, "0" * 64)
    with pytest.raises(ValueError):
        anchor._make_refusal("not_a_reason")
    refusal = anchor._make_refusal("p0_patch_unavailable")
    assert refusal.reason == "p0_patch_unavailable"
    assert anchor.V2BAnchorRefusal(**{
        "reason": refusal.reason, "detail": refusal.detail,
        "detail_sha256": refusal.detail_sha256,
        "refusal_sha256": refusal.refusal_sha256}).to_record() == refusal.to_record()


# ---------------------------------------------------------------------------
# Path A consistency-only diagnostic
# ---------------------------------------------------------------------------

def test_path_a_is_consistency_only(bundle):
    prediction = _run_anchor(bundle).prediction
    hypotheses = {
        axis: [value for value in prediction.hypotheses
               if value.world_axis == axis]
        for axis in (0, 1, 2)}
    diagnostic = anchor.diagnose_full_patch_consistency(
        bundle["readings"], hypotheses)
    assert diagnostic.status in ("consistent", "inconsistent", "unavailable",
                                 "insufficient_data")
    assert all("determined" not in json.dumps(value)
               for value in diagnostic.axes)
    record = diagnostic.to_record()
    assert "alpha" not in json.dumps(record)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_module_imports_stay_inside_the_boundary():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__", "base64", "dataclasses", "fractions", "hashlib", "json",
        "math", "typing", "numpy", "mc_photogrammetry",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module


def test_module_forbids_io_and_dynamic_execution():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    denied = {"open", "exec", "eval", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in denied, node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in denied, node.attr
    text = SOURCE.read_text(encoding="utf-8")
    for token in ("os.", "sys.", "subprocess", "pathlib", "socket",
                  "write_text", "read_text", "Path("):
        assert token not in text, token


# ---------------------------------------------------------------------------
# Independent forward oracle, provenance and budget boundaries
# ---------------------------------------------------------------------------

_ORACLE_AXIS_TRANSFORMS = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)


def _oracle_axis_transform(local_label, quadrant):
    return _ORACLE_AXIS_TRANSFORMS[quadrant][local_label]


def _oracle_d1(first, second):
    difference = abs(first - second) % 1
    return min(difference, 1 - difference)


def _oracle_band(energies, weights, leading, selected_b, period, anchor,
                 tau_band):
    numerator = Fraction(0)
    denominator = 0
    for index, (energy, weight) in enumerate(zip(energies, weights)):
        x_value = (leading + selected_b * Fraction(index)
                   / Fraction(*period.as_integer_ratio())) % 1
        if _oracle_d1(x_value, anchor) <= tau_band:
            numerator += Fraction(*energy.as_integer_ratio())
            denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def test_independent_forward_oracle_matches_module_energies(bundle):
    """Declared seams are recovered independently of the code under test."""
    prediction = _run_anchor(bundle).prediction
    assert prediction.status in ("partial", "determined")
    certificate = bundle["c2rb"].components[0].translation_classes[0]
    translations = {frame: tuple(Fraction(*value)
                                 for value in xyz)
                    for frame, xyz in certificate.translations_dyadic_xyz}
    sources = {source[0].reading_identity["frame"]: source
               for source in bundle["split_sources"]}
    tau_band = Fraction(1, 8)
    checked = 0
    for observation in prediction.observations:
        if observation.zero_coverage:
            continue
        frame = observation.reading_identity["frame"]
        axis = observation.world_axis
        source = sources[frame]
        seam = next(value for value in source[0].seam_profile_evidence
                    if value.local_axis == observation.local_axis)
        import base64
        energies = [float(value) for value in np.frombuffer(
            base64.b64decode(seam.profile_samples_float64_le_base64), "<f8")]
        weights = [int(value) for value in np.frombuffer(
            base64.b64decode(seam.profile_weights_float64_le_base64), "<f8")]
        # Declared true anchor is zero on every mapped axis.
        anchor = (observation.world_sign * (Fraction(0) - translations[frame][axis])) % 1
        expected_boundary = _oracle_band(
            energies, weights, Fraction(0), Fraction(1), seam.period_pixels,
            anchor, tau_band)
        expected_interior = _oracle_band(
            energies, weights, Fraction(0), Fraction(1), seam.period_pixels,
            (anchor + Fraction(1, 2)) % 1, tau_band)
        assert observation.boundary_energy is not None
        assert Fraction(*observation.boundary_energy) == expected_boundary
        assert Fraction(*observation.interior_energy) == expected_interior
        checked += 1
    assert checked >= 2


def test_hypotheses_retain_complete_ordered_contributors(bundle):
    prediction = _run_anchor(bundle).prediction
    values = set()
    for axis in (0, 1, 2):
        for hypothesis in [value for value in prediction.hypotheses
                           if value.world_axis == axis]:
            assert hypothesis.contributor_count == len(hypothesis.contributors)
            assert tuple(value.contributor_order
                         for value in hypothesis.contributors) \
                == tuple(range(hypothesis.contributor_count))
            assert (axis, hypothesis.numerator, hypothesis.denominator) \
                not in values
            values.add((axis, hypothesis.numerator, hypothesis.denominator))
            for contributor in hypothesis.contributors:
                assert contributor.world_sign in (-1, 1)
                assert anchor._hex64(contributor.contributor_sha256)
                assert contributor.contributor_sha256 == \
                    anchor._sha256(contributor._payload())


def test_budget_boundaries_are_exact(bundle):
    passing = _run_anchor(bundle, max_source_records=len(bundle["split_sources"]))
    assert passing.status in ("partial", "determined")
    exhausted = _run_anchor(bundle, max_source_records=len(bundle["split_sources"]) - 1)
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_stage == "planned"
    assert exhausted.refusals[0].detail["cap"] == "max_source_records"
    assert exhausted.prediction is None
    # The fixture plans ten observation terms; nine must exhaust and ten pass.
    assert _run_anchor(bundle, max_observations=10).status in (
        "partial", "determined")
    assert _run_anchor(bundle, max_observations=9).status == "budget_exhausted"


def test_record_digests_replay_and_tampering_fails_closed(bundle):
    prediction = _run_anchor(bundle).prediction
    record = prediction.to_record()
    assert record["result_sha256"] == anchor._sha256(
        {key: value for key, value in record.items()
         if key != "result_sha256"})
    tampered = {name: getattr(prediction, name)
                for name in prediction.__dataclass_fields__}
    tampered["status"] = "determined"
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction(**tampered)
    hypothesis = prediction.hypotheses[0].to_record()
    broken = dict(hypothesis)
    broken["contributor_count"] = hypothesis["contributor_count"] + 1
    with pytest.raises(ValueError):
        anchor.V2BAnchorHypothesis(**broken)


def test_wrong_calibration_is_refused():
    with pytest.raises(ValueError):
        anchor.V2BAnchorCalibration(
            **{**{name: getattr(anchor.V2B_BUILTIN_CALIBRATION, name)
                  for name in (
                      "calibration_version", "source_calibration_sha256",
                      "partition_config_sha256", "tau_band", "tau_margin",
                      "tau_energy", "min_held_out_samples",
                      "min_distinct_frames_per_axis", "tie_break",
                      "target_power", "calibration_grid_sha256",
                      "calibration_corpus_sha256", "evaluation_corpus_sha256",
                      "null_control_ids_sha256", "seam_control_ids_sha256",
                      "calibration_sample_root_sha256",
                      "evaluation_sample_root_sha256",
                      "replay_comparison_tolerance",
                      "replay_closure_tolerance")},
               "calibration_sha256": "0" * 64})
    with pytest.raises(ValueError):
        anchor.V2BAnchorCalibration.from_split_phase_calibration(object())


def test_full_patch_g_evidence_mismatch_is_p1b_source_mismatch():
    """An accepted full-patch bundle whose G-only replay differs must refuse."""
    full = _build_bundle(g_only=False)
    assert full["readings"][0].phase_evidence.response_magnitude_uv == (8.0, 7.0)
    result = anchor.analyze_phase_anchor(
        full["c2rb"], full["vector_result"], full["support"], full["p1d"],
        full["matches"], full["phases"], readings=full["readings"],
        signatures=full["signatures"], comparison_tolerance=1e-12,
        closure_tolerance=0.0, v2a_result=full["v2a_result"],
        split_sources=full["split_sources"], budget=_anchor_budget())
    assert result.status == "refused"
    assert result.refusals[0].reason == "p1b_source_mismatch"
    assert result.prediction is None


def test_permuted_bundle_replays_byte_identically(bundle):
    baseline = _run_anchor(bundle)
    permuted = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], tuple(reversed(bundle["matches"])),
        tuple(reversed(bundle["phases"])),
        readings=tuple(reversed(bundle["readings"])),
        signatures=tuple(reversed(bundle["signatures"])),
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=tuple(reversed(bundle["split_sources"])),
        budget=_anchor_budget())
    assert permuted.to_json_bytes() == baseline.to_json_bytes()


# ---------------------------------------------------------------------------
# Amended V2-B runtime accounting: planned upper bound U, runtime supports K,
# distinct ordered D-coordinate held-out union
# ---------------------------------------------------------------------------

# Independent expectations for the public five-frame ring fixture.  These are
# declared here, never read back from the module under test.
_EXPECTED_PLANNED = {
    "readings": 5,
    "observations": 10,
    "frames": 5,
    "hypotheses_per_axis": 1,
    "hypotheses_total": 2,
    "contributor_refs_per_hypothesis": 20,
    "contributor_refs_total": 40,
    "source_records": 5,
    "boundary_interior_terms": 20,
    "d_profile_terms": 480,
    "payload_bytes": 4290,
    "replay_crosslinks": 10,
    "pixel_samples": 1280,
    "held_out_samples_per_reading": 192,
    "split_reconstructions": 5,
    "accepted_frame_supports": 10,
}
_EXPECTED_HELD_OUT_UNION = 192


def _structural_sources(bundle):
    """The D-only impulse-removed control: identical geometry, no D energy."""
    patch = bundle["structural"]
    return tuple(
        (_produce(patch, reading.frame, 0, reading.axis), patch.copy(),
         _p0_budget())
        for reading in bundle["readings"])


def _accepted_support_triples(prediction):
    """Independently recompute K from the retained observation records."""
    triples = set()
    for observation in prediction.observations:
        if observation.zero_coverage:
            continue
        if (Fraction(*observation.margin) >= Fraction(1, 8)
                and Fraction(*observation.boundary_energy) >= Fraction(0, 1)
                and observation.held_out_samples >= 1):
            triples.add((observation.world_axis, observation.hypothesis_sha256,
                         observation.reading_identity["frame"]))
    return triples


def _observation_support_triples(prediction):
    """Independently recompute U's (axis, hypothesis, frame) key set."""
    return {(observation.world_axis, observation.hypothesis_sha256,
             observation.reading_identity["frame"])
            for observation in prediction.observations}


def _produce_at(samples, frame="R0", plane_index=0, axis=2):
    """One accepted P0 result at a different patch size (unequal readings)."""
    extent = 4.0
    step = lp.lattice.rectified_pixel_step(extent, samples)
    period = B_VALUE / (DISTANCE * step)
    geometry = lp.SplitPhaseGeometry(
        centre=(extent, extent), extent=extent, samples=samples,
        rectified_pixel_step=step, period_pixels=period)
    patch = np.zeros((samples, samples), dtype=np.float64)
    patch[0, 0] = 1.0
    total = samples * samples
    budget = lp._make_budget({
        "max_lanes_per_axis": samples,
        "max_patch_samples": total,
        "max_inline_patch_samples": total,
        "max_g_samples_per_axis": total,
        "max_d_samples_per_axis": total,
        "max_guard_samples_per_axis": total,
        "max_profile_terms": 2 * samples,
        "max_d_profile_terms": 3 * 2 * samples,
        "max_payload_bytes": (2 * len(lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION)
                              + 48 + 24 * 2 * samples),
        "max_source_records": 1000,
        "max_replay_checks": 1000,
    })
    result = lp.produce_split_phase_evidence(
        patch, geometry=geometry, frame=frame, plane_index=plane_index,
        reading_axis=axis, source_region_id=None,
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
    assert result.status == "complete", result.refusals
    return result, patch, budget


def test_public_fixture_records_exact_planned_and_runtime_accounting(bundle):
    result = _run_anchor(bundle)
    assert result.status == "partial"
    prediction = result.prediction
    planned = prediction.planned_work
    runtime = prediction.runtime_work
    for name, expected in _EXPECTED_PLANNED.items():
        assert getattr(planned, name) == expected, name
        assert getattr(runtime, name) == expected, name
    # U is the distinct (axis, hypothesis, frame) key set, and every frame that
    # qualified for a hypothesis also passed the frozen test, so K == U == 10.
    upper = _observation_support_triples(prediction)
    accepted = _accepted_support_triples(prediction)
    assert len(upper) == 2 * 5
    assert len(accepted) == len(upper) == 10
    assert planned.accepted_frame_supports == len(upper)
    assert runtime.accepted_frame_supports == len(accepted)
    # The amended plan records U without comparing it to the runtime cap.
    assert result.work is runtime


def test_d_coordinate_union_is_a_distinct_ordered_set_union(bundle):
    """Per-reading held-out cardinality is |D_u union D_v|, not a sum."""
    result = _run_anchor(bundle).prediction
    assert result.planned_work.held_out_samples_per_reading \
        == _EXPECTED_HELD_OUT_UNION
    for source in bundle["split_sources"]:
        producer = source[0]
        by_axis = {}
        for lane in producer.lane_sets:
            if lane.role != "D":
                continue
            rows = int(producer.patch_ref.rows)
            columns = int(producer.patch_ref.columns)
            if lane.local_axis == "u":
                coordinates = {(row, column)
                               for row in range(int(lane.lane_index_start),
                                                int(lane.lane_index_stop))
                               for column in range(columns)}
            else:
                coordinates = {(row, column)
                               for column in range(int(lane.lane_index_start),
                                                   int(lane.lane_index_stop))
                               for row in range(rows)}
            by_axis[lane.local_axis] = coordinates
        expected = len(by_axis["u"] | by_axis["v"])
        assert len(by_axis["u"]) == 128
        assert len(by_axis["v"]) == 128
        assert len(by_axis["u"] & by_axis["v"]) == 64
        assert expected == _EXPECTED_HELD_OUT_UNION
        assert anchor._d_coordinate_union_size(producer) == expected
    # Five identical readings must not multiply the per-reading count.
    assert _EXPECTED_HELD_OUT_UNION != 5 * 128


def test_public_null_fixture_keeps_upper_bound_with_zero_supports(bundle):
    """The D-only impulse-removed control keeps U=10 but never inserts."""
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=_structural_sources(bundle), budget=_anchor_budget())
    assert outcome.status == "abstained"
    prediction = outcome.prediction
    assert prediction.planned_work.accepted_frame_supports == 10
    assert prediction.runtime_work.accepted_frame_supports == 0
    assert prediction.planned_work.held_out_samples_per_reading \
        == _EXPECTED_HELD_OUT_UNION
    assert prediction.runtime_work.held_out_samples_per_reading \
        == _EXPECTED_HELD_OUT_UNION
    assert len(_accepted_support_triples(prediction)) == 0
    assert len(prediction.observations) == 10


def test_accepted_frame_support_cap_boundary_9_10_11(bundle):
    exhausted = _run_anchor(bundle, max_accepted_frame_supports=9)
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_stage == "runtime"
    assert exhausted.prediction is None
    detail = exhausted.refusals[0].detail
    assert detail["cap"] == "max_accepted_frame_supports"
    assert detail["limit"] == 9
    # Exact runtime counter K = cap + 1 <= U; never planned + 1.
    assert detail["observed"] == 10
    assert exhausted.work.accepted_frame_supports == 10
    assert detail["planned_work"]["accepted_frame_supports"] == 10
    assert detail["runtime_work"]["accepted_frame_supports"] == 10
    for cap in (10, 11):
        outcome = _run_anchor(bundle, max_accepted_frame_supports=cap)
        assert outcome.status == "partial"
        assert outcome.prediction.runtime_work.accepted_frame_supports == 10


def test_runtime_exhaustion_is_atomic_at_the_insertion_boundary(bundle):
    """K stops at the first insertion above the cap with truthful counters."""
    exhausted = _run_anchor(bundle, max_accepted_frame_supports=5)
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_stage == "runtime"
    runtime = exhausted.work
    detail = exhausted.refusals[0].detail
    assert detail["observed"] == 6
    assert runtime.accepted_frame_supports == 6
    assert runtime.observations == 6
    assert runtime.boundary_interior_terms == 12
    assert runtime.held_out_samples_per_reading == _EXPECTED_HELD_OUT_UNION
    # The triggering observation is inside the exact runtime snapshot.
    assert detail["runtime_work"] == runtime.to_record()
    assert detail["planned_work"]["accepted_frame_supports"] == 10
    assert detail["planned_work"]["observations"] == 10
    assert exhausted.work.work_sha256 == anchor._sha256(
        {key: value for key, value in runtime.to_record().items()
         if key != "work_sha256"})


def test_held_out_samples_cap_boundary_191_192_193(bundle):
    exhausted = _run_anchor(bundle, max_held_out_samples_per_reading=191)
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_stage == "planned"
    detail = exhausted.refusals[0].detail
    assert detail["cap"] == "max_held_out_samples_per_reading"
    assert detail["limit"] == 191
    assert detail["observed"] == _EXPECTED_HELD_OUT_UNION
    assert exhausted.work.held_out_samples_per_reading \
        == _EXPECTED_HELD_OUT_UNION
    assert "runtime_work" not in detail
    for cap in (192, 193):
        assert _run_anchor(
            bundle, max_held_out_samples_per_reading=cap).status == "partial"


def test_every_static_cap_boundary_w_minus_one_w_w_plus_one(bundle):
    for work_name in anchor._STATIC_WORK_FIELDS:
        cap_name = anchor._WORK_TO_CAP[work_name]
        observed = _EXPECTED_PLANNED[work_name]
        assert _run_anchor(bundle, **{cap_name: observed}).status == "partial"
        assert _run_anchor(
            bundle, **{cap_name: observed + 1}).status == "partial"
        if observed < 2:
            with pytest.raises(ValueError):
                _anchor_budget(**{cap_name: 0})
            continue
        outcome = _run_anchor(bundle, **{cap_name: observed - 1})
        assert outcome.status == "budget_exhausted", work_name
        assert outcome.budget_stage == "planned", work_name
        assert outcome.prediction is None, work_name
        detail = outcome.refusals[0].detail
        assert detail["cap"] == cap_name, work_name
        assert detail["limit"] == observed - 1, work_name
        assert detail["observed"] == observed, work_name
        assert getattr(outcome.work, work_name) == observed, work_name
        assert "runtime_work" not in detail, work_name
    assert "accepted_frame_supports" not in anchor._STATIC_WORK_FIELDS


def test_invalid_input_refuses_before_any_budget_cap(bundle):
    """An invalid final source with tiny caps refuses, never exhausts."""
    sources = list(bundle["split_sources"])
    result, patch, budget = sources[-1]
    wrong = patch.copy()
    wrong[0, 0] = 9.0
    sources[-1] = (result, wrong, budget)
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=tuple(sources),
        budget=_anchor_budget(max_readings=1, max_observations=1,
                              max_source_records=1, max_payload_bytes=1,
                              max_accepted_frame_supports=1))
    assert outcome.status == "refused"
    assert outcome.refusals[0].reason in (
        "p0_patch_mismatch", "split_replay_mismatch",
        "split_not_reconstructible")
    assert outcome.budget_stage is None
    # A duplicate reading identity is also invalid input, not a budget stop.
    duplicate = list(bundle["split_sources"])
    duplicate.append(bundle["split_sources"][0])
    outcome = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=tuple(duplicate),
        budget=_anchor_budget(max_source_records=1,
                              max_accepted_frame_supports=1))
    assert outcome.status == "refused"
    assert outcome.refusals[0].reason == "split_not_reconstructible"


def test_unequal_reading_sizes_take_the_maximum_union():
    """Different-size readings aggregate by maximum, not sum."""
    small = _produce_at(H)
    large = _produce_at(20)
    mapping = {
        ("R0", 0, 2): small,
        ("R1", 0, 2): large,
    }
    assert anchor._d_coordinate_union_size(small[0]) == 192
    assert anchor._d_coordinate_union_size(large[0]) == 300
    assert anchor._max_held_out_union(mapping, set(mapping)) == 300
    assert anchor._max_held_out_union(mapping, {("R0", 0, 2)}) == 192
    assert anchor._max_held_out_union(mapping, set()) == 0


def test_simultaneous_planned_breaches_report_the_deterministic_first(bundle):
    outcome = _run_anchor(bundle, max_readings=1, max_observations=1,
                          max_payload_bytes=1, max_d_profile_terms=1)
    assert outcome.status == "budget_exhausted"
    assert outcome.budget_stage == "planned"
    detail = outcome.refusals[0].detail
    # ``readings`` is the first static counter, so it is the single trigger.
    assert detail["cap"] == "max_readings"
    assert detail["observed"] == 5
    assert detail["limit"] == 1
    assert outcome.work.accepted_frame_supports == 10


def test_permuted_bundle_preserves_runtime_accounting(bundle):
    baseline = _run_anchor(bundle)
    permuted = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], tuple(reversed(bundle["matches"])),
        tuple(reversed(bundle["phases"])),
        readings=tuple(reversed(bundle["readings"])),
        signatures=tuple(reversed(bundle["signatures"])),
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=tuple(reversed(bundle["split_sources"])),
        budget=_anchor_budget())
    assert permuted.prediction.planned_work.to_record() \
        == baseline.prediction.planned_work.to_record()
    assert permuted.prediction.runtime_work.to_record() \
        == baseline.prediction.runtime_work.to_record()


def _resealed_work(work, **changes):
    """Return a freshly resealed work record with recomputed digest."""
    payload = {name: getattr(work, name) for name in anchor._WORK_FIELDS}
    payload.update(changes)
    return anchor._make_work(payload)


def _prediction_fields(prediction, **changes):
    fields = {name: getattr(prediction, name)
              for name in prediction.__dataclass_fields__}
    fields.update(changes)
    return fields


def _result_fields(result, **changes):
    fields = {name: getattr(result, name)
              for name in result.__dataclass_fields__}
    fields.update(changes)
    return fields


def test_resealed_work_alias_relations_fail_closed(bundle):
    """A resealed work record with contradictory alias counters is refused."""
    runtime = _run_anchor(bundle).prediction.runtime_work
    payload = {name: getattr(runtime, name) for name in anchor._WORK_FIELDS}
    payload["accepted_frame_supports"] = 11
    # A stale digest can never hide the mutation.
    with pytest.raises(ValueError):
        anchor.V2BAnchorWork(**payload, work_sha256=runtime.work_sha256)
    # Resealing the digest is not enough: the alias relations are re-derived.
    resealed = anchor._make_work(payload)
    assert resealed.accepted_frame_supports == 11
    assert resealed.work_sha256 != runtime.work_sha256
    assert anchor._work_within_cap(resealed, _anchor_budget())
    with pytest.raises(ValueError):
        anchor._make_work({**payload, "observations": 9})
    with pytest.raises(ValueError):
        anchor._make_work({**payload, "replay_crosslinks": 9})
    with pytest.raises(ValueError):
        anchor._make_work({"accepted_frame_supports": -1})
    for bad in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            anchor._make_work({"accepted_frame_supports": bad})


def test_make_anchor_budget_rejects_non_exact_integer_caps():
    """Public caps reject bool/str/float/None before any coercion."""
    budget = _anchor_budget()
    caps = {name: getattr(budget, name) for name in anchor._BUDGET_FIELDS}
    for bad in (True, False, "9", 9.0, 9.5, None, (9,)):
        for name in ("max_readings", "max_accepted_frame_supports"):
            with pytest.raises(ValueError):
                anchor.make_anchor_budget(**{**caps, name: bad})
    rebuilt = anchor.make_anchor_budget(**caps)
    assert rebuilt.to_record() == budget.to_record()


def test_resealed_prediction_k_above_upper_bound_fails_closed(bundle):
    """``K > U`` and a contradicted hypothesis/contributor sum are refused."""
    prediction = _run_anchor(bundle).prediction
    upper = prediction.planned_work.accepted_frame_supports
    bumped = _resealed_work(prediction.runtime_work,
                            accepted_frame_supports=upper + 1)
    assert bumped.accepted_frame_supports == upper + 1
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, runtime_work=bumped))
    wrong_total = _resealed_work(
        prediction.runtime_work,
        hypotheses_total=prediction.runtime_work.hypotheses_total + 1)
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, runtime_work=wrong_total))
    wrong_contributors = _resealed_work(
        prediction.planned_work,
        contributor_refs_total=prediction.planned_work.contributor_refs_total + 1)
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, planned_work=wrong_contributors))


def test_resealed_prediction_held_out_and_observation_mismatch(bundle):
    """Held-out maximum and observation count are bound to the records."""
    prediction = _run_anchor(bundle).prediction
    count = len(prediction.observations)
    mismatched = _resealed_work(
        prediction.runtime_work, held_out_samples_per_reading=191)
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, runtime_work=mismatched))
    lower, upper = anchor._derived_held_out_bounds(prediction.observations)
    assert lower == 128 and upper == 256
    fabricated = upper + 1
    both = _resealed_work(
        prediction.runtime_work, held_out_samples_per_reading=fabricated)
    planned = _resealed_work(
        prediction.planned_work, held_out_samples_per_reading=fabricated)
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, runtime_work=both,
                                 planned_work=planned))
    counted = _resealed_work(
        prediction.runtime_work, observations=count + 1,
        boundary_interior_terms=2 * (count + 1))
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction.build(
            **_prediction_fields(prediction, runtime_work=counted))


def test_resealed_outer_work_and_completed_prediction_binding(bundle):
    """The outer snapshot must be the prediction's exact runtime snapshot."""
    result = _run_anchor(bundle)
    assert result.status in ("partial", "determined")
    changed = _resealed_work(result.work, readings=result.work.readings + 1)
    with pytest.raises(ValueError):
        anchor._make_result(**_result_fields(result, work=changed))
    with pytest.raises(ValueError):
        anchor._make_result(**_result_fields(result, work=None))


def test_resealed_budget_branch_consistency_fails_closed(bundle):
    """Every exhaustion branch alias/snapshot mutation is refused."""
    exhausted = _run_anchor(bundle, max_accepted_frame_supports=9)
    assert exhausted.status == "budget_exhausted"
    detail = exhausted.refusals[0].detail
    runtime_work = exhausted.work

    def resealed_result(detail_value, *, stage="runtime", work=runtime_work):
        refusal = anchor._make_refusal("budget_exhausted", detail_value,
                                       cap=detail_value.get("cap"))
        return anchor._make_result(
            status="budget_exhausted", prediction=None, refusals=(refusal,),
            budget_stage=stage, work=work)

    # Control: the exact accepted branch still reseals and validates.
    assert resealed_result(dict(detail)).budget_stage == "runtime"

    malformed = anchor._make_refusal("budget_exhausted",
                                     {"budget_stage": "runtime"})
    with pytest.raises(ValueError):
        anchor._make_result(status="budget_exhausted", prediction=None,
                            refusals=(malformed,), budget_stage=None,
                            work=runtime_work)

    wrong_observed = dict(detail, observed=detail["observed"] + 1)
    with pytest.raises(ValueError):
        resealed_result(wrong_observed)
    # A forged cap that skipped the exact first-over insertion stop is refused.
    with pytest.raises(ValueError):
        resealed_result(dict(detail, limit=detail["limit"] - 1))
    with pytest.raises(ValueError):
        resealed_result(dict(detail), stage="planned")
    with pytest.raises(ValueError):
        resealed_result(dict(detail), work=None)

    mismatched_runtime = dict(detail)
    mismatched_runtime["runtime_work"] = _resealed_work(
        runtime_work, observations=4, boundary_interior_terms=8).to_record()
    with pytest.raises(ValueError):
        resealed_result(mismatched_runtime)

    planned_work = anchor.V2BAnchorWork(**detail["planned_work"])
    below_upper = _resealed_work(planned_work, accepted_frame_supports=0)
    k_above_upper = dict(detail, planned_work=below_upper.to_record())
    with pytest.raises(ValueError):
        resealed_result(k_above_upper)

    with_evidence = dict(detail, observations=[])
    with pytest.raises(ValueError):
        resealed_result(with_evidence)

    planned = _run_anchor(bundle, max_held_out_samples_per_reading=191)
    assert planned.budget_stage == "planned"
    planned_detail = dict(planned.refusals[0].detail)
    planned_detail["runtime_work"] = runtime_work.to_record()
    refusal = anchor._make_refusal("budget_exhausted", planned_detail,
                                   cap=planned_detail["cap"])
    with pytest.raises(ValueError):
        anchor._make_result(status="budget_exhausted", prediction=None,
                            refusals=(refusal,), budget_stage="planned",
                            work=planned.work)

    budget = _anchor_budget()
    caps = {name: getattr(budget, name) for name in anchor._BUDGET_FIELDS}
    with pytest.raises(ValueError):
        anchor.make_anchor_budget(**{**caps,
                                     "max_accepted_frame_supports": 0})
    for bad in (True, "9", 1.0):
        with pytest.raises(ValueError):
            anchor.V2BAnchorBudget(
                **{**caps, "max_accepted_frame_supports": bad},
                budget_sha256=budget.budget_sha256)
    prediction = _run_anchor(bundle).prediction
    with pytest.raises(ValueError):
        anchor.V2BAnchorPrediction(
            **_prediction_fields(prediction, runtime_work=10))


# ---------------------------------------------------------------------------
# Source-bound replay validator and branch-specific cap eligibility
# ---------------------------------------------------------------------------

def _validate_replay(bundle, result, budget, sources=None):
    """Replay the claimed record from the original bundle sources and budget."""
    return anchor.validate_phase_anchor_replay(
        result, bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=(bundle["split_sources"] if sources is None else sources),
        budget=budget)


def test_source_bound_replay_accepts_unchanged_completed_result(bundle):
    """An unchanged completed record replays byte-identically."""
    result = _run_anchor(bundle)
    assert result.status == "partial"
    assert _validate_replay(bundle, result, _anchor_budget()) is True
    # A non-record claim is rejected, never raised.
    assert anchor.validate_phase_anchor_replay(
        object(), bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"],
        split_sources=bundle["split_sources"], budget=_anchor_budget()) is False


def test_source_bound_replay_accepts_genuine_planned_and_runtime_exhaustion(bundle):
    """Genuine planned and runtime stops replay byte-identically."""
    planned = _run_anchor(bundle, max_held_out_samples_per_reading=191)
    assert planned.status == "budget_exhausted"
    assert planned.budget_stage == "planned"
    assert planned.refusals[0].detail["cap"] == \
        "max_held_out_samples_per_reading"
    assert _validate_replay(
        bundle, planned,
        _anchor_budget(max_held_out_samples_per_reading=191)) is True

    runtime = _run_anchor(bundle, max_accepted_frame_supports=9)
    assert runtime.status == "budget_exhausted"
    assert runtime.budget_stage == "runtime"
    assert runtime.refusals[0].detail["cap"] == "max_accepted_frame_supports"
    assert runtime.refusals[0].detail["observed"] == 10
    assert _validate_replay(
        bundle, runtime, _anchor_budget(max_accepted_frame_supports=9)) is True


def test_source_bound_replay_rejects_resealed_held_out_snapshots(bundle):
    """192 -> 191 in both snapshots survives the constructor, not the replay."""
    result = _run_anchor(bundle)
    prediction = result.prediction
    assert prediction.planned_work.held_out_samples_per_reading \
        == _EXPECTED_HELD_OUT_UNION
    planned = _resealed_work(
        prediction.planned_work,
        held_out_samples_per_reading=_EXPECTED_HELD_OUT_UNION - 1)
    runtime = _resealed_work(
        prediction.runtime_work,
        held_out_samples_per_reading=_EXPECTED_HELD_OUT_UNION - 1)
    # The retained-record bounds [128, 256] and the planned == runtime rule both
    # still hold, so the semantic constructor accepts the resealed prediction.
    forged_prediction = anchor.V2BAnchorPrediction.build(
        **_prediction_fields(prediction, planned_work=planned,
                             runtime_work=runtime))
    assert forged_prediction.planned_work.held_out_samples_per_reading == 191
    assert forged_prediction.runtime_work.held_out_samples_per_reading == 191
    forged = anchor._make_result(**_result_fields(
        result, prediction=forged_prediction, work=runtime))
    assert forged.to_json_bytes() != result.to_json_bytes()
    assert forged.prediction.runtime_work.to_record() \
        == forged.work.to_record()
    # Only the source-bound replay can recover the real union (192).
    assert _validate_replay(bundle, forged, _anchor_budget()) is False
    assert _validate_replay(bundle, result, _anchor_budget()) is True


def test_source_bound_replay_rejects_forged_runtime_support_stop(bundle):
    """The real cap-9 stop (K=10) rejects a rewritten limit=8/K=9 forgery."""
    genuine = _run_anchor(bundle, max_accepted_frame_supports=9)
    assert genuine.budget_stage == "runtime"
    detail = genuine.refusals[0].detail
    assert detail["limit"] == 9 and detail["observed"] == 10
    assert genuine.work.accepted_frame_supports == 10
    assert genuine.work.observations == 10
    # Rewrite the stop to limit=8 with K=9 over the same ten observations; every
    # constructor alias (K <= observations, K == limit + 1 <= U) still holds.
    forged_runtime = _resealed_work(
        genuine.work, accepted_frame_supports=9, observations=10)
    forged_detail = dict(detail, limit=8, observed=9,
                         runtime_work=forged_runtime.to_record())
    refusal = anchor._make_refusal("budget_exhausted", forged_detail,
                                   cap="max_accepted_frame_supports")
    forged = anchor._make_result(
        status="budget_exhausted", prediction=None, refusals=(refusal,),
        budget_stage="runtime", work=forged_runtime)
    assert forged.refusals[0].detail["limit"] == 8
    assert forged.work.accepted_frame_supports == 9
    budget = _anchor_budget(max_accepted_frame_supports=9)
    assert _validate_replay(bundle, forged, budget) is False
    assert _validate_replay(bundle, genuine, budget) is True


def test_planned_branch_cannot_claim_runtime_only_support_cap(bundle):
    """``max_accepted_frame_supports`` can never be a planned exhaustion."""
    genuine = _run_anchor(bundle, max_held_out_samples_per_reading=191)
    assert genuine.budget_stage == "planned"
    detail = dict(genuine.refusals[0].detail)
    forged_detail = dict(
        detail, cap="max_accepted_frame_supports", limit=9,
        observed=genuine.work.accepted_frame_supports)
    refusal = anchor._make_refusal("budget_exhausted", forged_detail,
                                   cap="max_accepted_frame_supports")
    with pytest.raises(ValueError):
        anchor._make_result(
            status="budget_exhausted", prediction=None, refusals=(refusal,),
            budget_stage="planned", work=genuine.work)
    # Every static preflight cap remains a legitimate planned branch.
    assert genuine.refusals[0].detail["cap"] \
        in anchor._PLANNED_EXHAUSTIBLE_CAPS
    assert "max_accepted_frame_supports" \
        not in anchor._PLANNED_EXHAUSTIBLE_CAPS
    assert _validate_replay(
        bundle, genuine,
        _anchor_budget(max_held_out_samples_per_reading=191)) is True


def test_budget_refusal_limit_below_one_is_rejected(bundle):
    """A refusal limit of zero or below is refused on both branches."""
    planned = _run_anchor(bundle, max_payload_bytes=1)
    assert planned.budget_stage == "planned"
    planned_detail = dict(planned.refusals[0].detail)
    assert planned_detail["limit"] == 1
    for bad_limit in (0, -1):
        forged_detail = dict(planned_detail, limit=bad_limit)
        refusal = anchor._make_refusal(
            "budget_exhausted", forged_detail, cap=forged_detail["cap"])
        with pytest.raises(ValueError):
            anchor._make_result(
                status="budget_exhausted", prediction=None, refusals=(refusal,),
                budget_stage="planned", work=planned.work)

    runtime = _run_anchor(bundle, max_accepted_frame_supports=9)
    assert runtime.budget_stage == "runtime"
    runtime_detail = dict(runtime.refusals[0].detail)
    assert runtime_detail["limit"] == 9
    for bad_limit in (0, -1):
        forged_detail = dict(runtime_detail, limit=bad_limit)
        refusal = anchor._make_refusal(
            "budget_exhausted", forged_detail, cap=forged_detail["cap"])
        with pytest.raises(ValueError):
            anchor._make_result(
                status="budget_exhausted", prediction=None, refusals=(refusal,),
                budget_stage="runtime", work=runtime.work)
    assert _validate_replay(
        bundle, planned, _anchor_budget(max_payload_bytes=1)) is True


def test_source_bound_replay_preserves_invalid_input_before_budget(bundle):
    """Invalid input refuses before any cap, and only the real refusal replays."""
    sources = list(bundle["split_sources"])
    result, patch, p0_budget = sources[-1]
    wrong = patch.copy()
    wrong[0, 0] = 9.0
    sources[-1] = (result, wrong, p0_budget)
    sources = tuple(sources)
    tiny = _anchor_budget(max_readings=1, max_observations=1,
                          max_source_records=1, max_payload_bytes=1,
                          max_accepted_frame_supports=1)
    refused = anchor.analyze_phase_anchor(
        bundle["c2rb"], bundle["vector_result"], bundle["support"],
        bundle["p1d"], bundle["matches"], bundle["phases"],
        readings=bundle["readings"], signatures=bundle["signatures"],
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=bundle["v2a_result"], split_sources=sources, budget=tiny)
    assert refused.status == "refused"
    assert refused.budget_stage is None
    assert _validate_replay(bundle, refused, tiny, sources=sources) is True
    # A genuine budget branch over valid sources never validates against the
    # invalid source set: the replay refuses first and the bytes differ.
    forged = _run_anchor(bundle, max_payload_bytes=1)
    assert forged.status == "budget_exhausted"
    assert _validate_replay(bundle, forged, tiny, sources=sources) is False


# ---------------------------------------------------------------------------
# Compatibility repair: full signed dyadic translations (C2-R-B domain)
# ---------------------------------------------------------------------------

# Declared by the fixture, never read back from the module under test.  The
# first frame is the gauge anchor and must stay exactly zero; every other entry
# covers an integer or a negative fraction on each of the three world axes.
_SIGNED_TRANSLATIONS = (
    ("A", ((0, 1), (0, 1), (0, 1))),
    ("B", ((3, 1), (-1, 4), (5, 8))),
    ("C", ((-3, 1), (7, 2), (-1, 2))),
    ("D", ((1, 1), (-5, 4), (9, 8))),
)
# The exact modulo-one residues of the same signed values.
_MOD_ONE_TRANSLATIONS = (
    ("A", ((0, 1), (0, 1), (0, 1))),
    ("B", ((0, 1), (3, 4), (5, 8))),
    ("C", ((0, 1), (1, 2), (1, 2))),
    ("D", ((0, 1), (3, 4), (1, 8))),
)
# The public five-frame ring fixture's frame names, with signed values on every
# world axis and non-zero residues so the equivalence check is not degenerate.
_BUNDLE_SIGNED = {
    "R0": ((0, 1), (0, 1), (0, 1)),
    "R1": ((3, 1), (-1, 4), (5, 8)),
    "R2": ((-3, 1), (7, 2), (-1, 2)),
    "R3": ((1, 1), (-5, 4), (9, 8)),
    "R4": ((-1, 1), (3, 4), (-7, 8)),
}
_BUNDLE_MOD_ONE = {
    "R0": ((0, 1), (0, 1), (0, 1)),
    "R1": ((0, 1), (3, 4), (5, 8)),
    "R2": ((0, 1), (1, 2), (1, 2)),
    "R3": ((0, 1), (3, 4), (1, 8)),
    "R4": ((0, 1), (3, 4), (1, 8)),
}


def _certificate(translations, edge_references=()):
    """Build a structurally valid certificate through its own C2-R-B sealer."""
    fields = {
        "gauge_anchor": translations[0][0],
        "translations_dyadic_xyz": tuple(translations),
        "edge_references": tuple(edge_references),
        "assignment_provenance_multiplicity": 1,
    }
    return relative._seal_record(
        relative.C2RBTranslationClassCertificate, fields,
        "certificate_sha256", relative._certificate_payload)


def test_translation_parser_accepts_full_signed_dyadic_domain():
    certificate = _certificate(_SIGNED_TRANSLATIONS)
    parsed = anchor._translations_by_frame(certificate)
    for frame, xyz in _SIGNED_TRANSLATIONS:
        assert parsed[frame] == tuple(Fraction(*value) for value in xyz)
    # Plane values that used to be rejected are preserved exactly: `3` stays the
    # integer three, not a wrapped zero, and `-1/4` stays negative.
    assert parsed["B"] == (Fraction(3), Fraction(-1, 4), Fraction(5, 8))
    assert parsed["C"] == (Fraction(-3), Fraction(7, 2), Fraction(-1, 2))


def test_translation_parser_admits_exactly_the_c2rb_validated_domain():
    """The parser accepts exactly what C2-R-B's own contract admits."""
    for numerator in range(-8, 9):
        for denominator in (1, 2, 4, 8, 16):
            value = (numerator, denominator)
            if relative._validate_dyadic(value):
                assert anchor._signed_dyadic(value) \
                    == relative._fraction(value)
            else:
                with pytest.raises(ValueError):
                    anchor._signed_dyadic(value)


@pytest.mark.parametrize("value", [
    (1, 3), (2, 4), (0, 4), (0, 2), (1, 0), (-1, 0), (3, -2),
    (1,), (1, 4, 2), (1.0, 4.0), (True, 1), "1/4", None, 3,
])
def test_translation_parser_rejects_noncanonical_values(value):
    """Non-dyadic, non-reduced and malformed tuples keep the exact refusal."""
    assert relative._validate_dyadic(value) is False
    with pytest.raises(ValueError) as error:
        anchor._signed_dyadic(value)
    assert str(error.value) == "noncanonical_dyadic"


def _derive_with_certificate(bundle, certificate, evidence=None):
    """Run hypothesis derivation against one stand-in C2-R-B certificate."""
    holder = SimpleNamespace(
        components=[SimpleNamespace(translation_classes=[certificate])])
    hypotheses, failures = anchor._derive_hypotheses(
        holder, bundle["v2a_result"],
        bundle["phases"] if evidence is None else evidence,
        anchor._ordered_readings(bundle["readings"]), _anchor_budget())
    return ({(value.world_axis): value.to_record()
             for axis in (0, 1, 2) for value in hypotheses[axis]}, failures)


def test_signed_translation_certificate_replays_identically_to_mod_one(bundle):
    """Full signed parsing reaches the same exact anchors as explicit mod-one."""
    certificate = bundle["c2rb"].components[0].translation_classes[0]
    frames = tuple(frame for frame, _xyz in certificate.translations_dyadic_xyz)
    references = certificate.edge_references
    signed = _certificate(
        tuple((frame, _BUNDLE_SIGNED[frame]) for frame in frames), references)
    canonical = _certificate(
        tuple((frame, _BUNDLE_MOD_ONE[frame]) for frame in frames), references)
    signed_hypotheses, signed_failures = _derive_with_certificate(bundle, signed)
    canonical_hypotheses, canonical_failures = \
        _derive_with_certificate(bundle, canonical)
    baseline_hypotheses, _ = _derive_with_certificate(bundle, certificate)
    assert signed_failures == {}
    assert canonical_failures == {}
    # The declared residues are non-zero, so this is a real replay and not an
    # all-zero certificate that would pass under any parser.
    assert signed_hypotheses != baseline_hypotheses
    assert signed_hypotheses == canonical_hypotheses
    # Contributor records keep the accepted canonical modulo-one representation.
    for hypothesis in signed_hypotheses.values():
        for contributor in hypothesis["contributors"]:
            assert anchor._validate_dyadic(
                tuple(contributor["frame_translation_dyadic"]))
    # Canonical ordering is by content, so evidence permutation cannot reorder
    # contributors or change any digest.
    permuted, permuted_failures = _derive_with_certificate(
        bundle, signed, tuple(reversed(bundle["phases"])))
    assert permuted_failures == {}
    assert permuted == signed_hypotheses


# ---------------------------------------------------------------------------
# Compatibility repair: P1-B tangent labels across every normal axis
# ---------------------------------------------------------------------------

_TANGENT_LABELS = {
    axis: tuple(label for label in range(3) if label != axis)
    for axis in range(3)}


def _tangent_label(normal_axis, local_axis):
    tangents = _TANGENT_LABELS[normal_axis]
    return tangents[0 if local_axis == "u" else 1]


def test_tangent_labels_follow_the_shared_sorted_convention():
    """The label helper selects the shared labels, never a world axis itself."""
    for normal in range(3):
        labels = tuple(anchor._reading_tangent_label(normal, name)
                       for name in ("u", "v"))
        assert labels == _TANGENT_LABELS[normal]
        assert labels[0] != labels[1]
        assert normal not in labels
        for quadrant in range(4):
            mapped = tuple(plane_graph.axis_transform(label, quadrant)
                           for label in labels)
            mapped_normal, _sign = plane_graph.axis_transform(normal, quadrant)
            # The shared transform is a signed permutation: the two tangent
            # labels must land on exactly the two non-normal world axes.
            assert {axis for axis, _sign in mapped} \
                == set(range(3)) - {mapped_normal}
            assert all(sign in (-1, 1) for _axis, sign in mapped)


@pytest.mark.parametrize("reading_axis,local_axis", [
    (3, "u"), (-1, "v"), (True, "u"), (0, "w"), (0, None), (0.0, "u"),
])
def test_tangent_label_helper_rejects_invalid_inputs(reading_axis, local_axis):
    with pytest.raises(ValueError):
        anchor._reading_tangent_label(reading_axis, local_axis)


def test_all_normal_axes_derive_actual_p1b_tangent_labels(axis_bundle):
    """Every normal derives its real u/v labels; the normal axis stays empty."""
    normal = axis_bundle["axis"]
    tangents = _TANGENT_LABELS[normal]
    prediction = _run_anchor(axis_bundle).prediction
    assert prediction is not None
    by_axis = {axis.world_axis: axis for axis in prediction.axes}
    assert by_axis[normal].status == "abstained"
    assert by_axis[normal].reason == "no_hypothesis"
    assert {value.world_axis for value in prediction.hypotheses} == set(tangents)
    for hypothesis in prediction.hypotheses:
        kinds = {}
        for contributor in hypothesis.contributors:
            kinds[contributor.source_kind] = \
                kinds.get(contributor.source_kind, 0) + 1
            # The P1-B branch (and every other branch) must use the reading's
            # real tangent label, not a linear index.
            assert contributor.local_axis_label == _tangent_label(
                normal, contributor.local_axis)
            assert plane_graph.axis_transform(
                contributor.local_axis_label,
                contributor.absolute_quadrant) == (
                    hypothesis.world_axis, contributor.world_sign)
        assert kinds == {"reading_domain": 5, "phase_candidates": 10,
                         "edge_reference": 5}
        assert hypothesis.contributor_count == 20
    for axis in tangents:
        assert by_axis[axis].status == "determined"
        assert by_axis[axis].alpha_dyadic == (0, 1)


def test_all_normal_axes_permutation_and_replay(axis_bundle):
    """All-axis bundles replay byte-identically under input permutation."""
    baseline = _run_anchor(axis_bundle)
    permuted = anchor.analyze_phase_anchor(
        axis_bundle["c2rb"], axis_bundle["vector_result"],
        axis_bundle["support"], axis_bundle["p1d"],
        tuple(reversed(axis_bundle["matches"])),
        tuple(reversed(axis_bundle["phases"])),
        readings=tuple(reversed(axis_bundle["readings"])),
        signatures=tuple(reversed(axis_bundle["signatures"])),
        comparison_tolerance=1e-12, closure_tolerance=0.0,
        v2a_result=axis_bundle["v2a_result"],
        split_sources=tuple(reversed(axis_bundle["split_sources"])),
        budget=_anchor_budget())
    assert permuted.to_json_bytes() == baseline.to_json_bytes()
    assert _validate_replay(axis_bundle, baseline, _anchor_budget()) is True






