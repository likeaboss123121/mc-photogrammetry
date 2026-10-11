"""P1-B controls for per-reading periodic-energy residue candidates.

Synthetic evidence only: these tests preserve the complete rational ambiguity
and the match-relative cell shift independently. No image, capture, manifest,
reference, or evaluation inputs participate.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice, lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import plane_residue


PERIOD = 16.0
CENTRE = (13.5, -7.0)
EXTENT = 4.0
SAMPLES = 32
PLANE = 2


def _phase_evidence(
    phase=(0.0, 7.5), *, period=PERIOD, centre=CENTRE, extent=EXTENT,
    samples=SAMPLES, source_region_id=None,
):
    return lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=tuple(phase), period_pixels=float(period),
        patch_centre_uv=tuple(centre), extent=float(extent), samples=int(samples),
        response_magnitude_uv=(8.0, 6.0), profile_mass_uv=(16.0, 12.0),
        concentration_uv=(0.5, 0.5), unavailable_reason_uv=(None, None),
        source_region_id=source_region_id,
    )


def _reading(frame: str, *, axis: int, distances=None, phase=(0.0, 7.5),
             evidence=True, period=PERIOD, centre=CENTRE, extent=EXTENT,
             samples=SAMPLES, region_evidence=None, polygon=None):
    if distances is None:
        distances = tuple((value, 1.0, 1.0 - 0.01 * index)
                          for index, value in enumerate(
                              lattice_planes.DISTANCE_ALTERNATIVES))
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=100.0, principal_point=(32.0, 24.0), centre=tuple(centre),
    )
    phase_value = (_phase_evidence(
        phase, period=period, centre=centre, extent=extent, samples=samples,
        source_region_id=(region_evidence.source_region_id
                          if region_evidence is not None else None))
        if evidence else None)
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=1.0,
        alternatives=tuple(lattice_planes.DistanceAlternative(*item)
                           for item in distances),
        period_pixels=float(period), phase_pixels=tuple(phase),
        extent=float(extent), samples=int(samples), polygon=polygon,
        coverage_cells=8, structural_share=0.6, confidence=0.8,
        rectification=rectification, region_evidence=region_evidence,
        phase_evidence=phase_value,
    )


def _match(*, left_b=1.0, right_b=1.0, shift=(0, 0), psr=4.5):
    return plane_graph.PlaneMatch(
        left="left", right="right", left_axis=0, right_axis=2,
        left_b=left_b, right_b=right_b, quadrant_of_right=3,
        psr=psr, peak=0.75, second_peak=0.15, overlap_cells=37,
        integer_residual=None, score=0.625, left_plane=PLANE,
        right_plane=PLANE, cell_shift_rc=shift,
    )


def _run(left, right, match=None):
    return plane_residue.hypothesis_phase_shift_evidence(
        _match() if match is None else match,
        {(left.frame, PLANE): left, (right.frame, PLANE): right},
    )


def _candidate_for(record, frame, axis):
    return next(value for value in record.phase_candidates
                if value.frame == frame and value.local_axis == axis)


def _assert_circular_set_equal(actual, expected, *, tolerance=2e-10):
    assert len(actual) == len(expected)
    unmatched = list(actual)
    for value in expected:
        index = next((i for i, found in enumerate(unmatched)
                      if min(abs(found - value), 1.0 - abs(found - value))
                      <= tolerance), None)
        assert index is not None, (actual, expected, value)
        unmatched.pop(index)
    assert not unmatched


@pytest.mark.parametrize("b_value", lattice_planes.DISTANCE_ALTERNATIVES)
def test_all_supported_b_values_retain_complete_rational_candidate_sets(b_value):
    left = _reading("left", axis=0)
    right = _reading("right", axis=2)
    result = _run(left, right, _match(left_b=b_value, right_b=b_value))
    assert result.refusal is None
    candidate = _candidate_for(result, "left", "u")
    rational = Fraction(str(float(b_value)))
    assert (candidate.rational_numerator, candidate.rational_denominator) == (
        rational.numerator, rational.denominator)
    assert candidate.blocks_per_period == b_value
    assert candidate.phase_error_amplification == abs(b_value)
    assert len(candidate.residue_candidates) == rational.denominator
    assert tuple(sorted(set(candidate.residue_candidates))) == candidate.residue_candidates
    assert all(0.0 <= value < 1.0 and math.isfinite(value)
               for value in candidate.residue_candidates)

    phase = left.phase_evidence.phase_uv_pixels[0]
    fraction = phase / PERIOD
    distance = left.matching_alternatives(b_value)[0].distance_blocks
    leading = distance * (CENTRE[0] - EXTENT)
    enumerated = {
        (leading + (fraction + n) * b_value) % 1.0
        for n in range(-4 * rational.denominator,
                       4 * rational.denominator + 1)
    }
    _assert_circular_set_equal(candidate.residue_candidates, enumerated)
    assert len(result.to_json_bytes()) > 0


@pytest.mark.parametrize("b_value,expected_count", [(1.0, 1), (0.5, 2), (1.0 / 16.0, 16)])
def test_integer_and_fractional_periods_preserve_expected_ambiguity(b_value, expected_count):
    result = _run(_reading("left", axis=0), _reading("right", axis=2),
                  _match(left_b=b_value, right_b=b_value))
    candidate = _candidate_for(result, "left", "v")
    assert len(candidate.residue_candidates) == expected_count
    assert candidate.refusal is None
    if expected_count > 1:
        assert len(set(candidate.residue_candidates)) == expected_count


def test_phase_change_and_one_period_window_translation_have_expected_effect():
    b_value = 0.5
    world_period = PERIOD * lattice.rectified_pixel_step(EXTENT, SAMPLES)
    distance = b_value / world_period
    assert distance * world_period == pytest.approx(b_value)
    distances = tuple((value, distance if value == b_value else 1.0,
                       1.0 - 0.01 * index)
                      for index, value in enumerate(lattice_planes.DISTANCE_ALTERNATIVES))
    baseline = _run(_reading("left", axis=0, phase=(3.0, 7.5),
                             distances=distances),
                    _reading("right", axis=2),
                    _match(left_b=b_value, right_b=b_value))
    moved = _run(_reading("left", axis=0, phase=(3.0, 7.5), distances=distances,
                          centre=(CENTRE[0] + world_period, CENTRE[1])),
                 _reading("right", axis=2),
                 _match(left_b=b_value, right_b=b_value))
    assert distance * (world_period) == pytest.approx(b_value)
    base_set = _candidate_for(baseline, "left", "u").residue_candidates
    moved_set = _candidate_for(moved, "left", "u").residue_candidates
    _assert_circular_set_equal(base_set, moved_set)

    changed_phase = _run(_reading("left", axis=0, phase=(4.0, 7.5),
                                 distances=distances),
                         _reading("right", axis=2),
                         _match(left_b=b_value, right_b=b_value))
    changed = _candidate_for(changed_phase, "left", "u").residue_candidates
    predicted = tuple(sorted((value + b_value / PERIOD) % 1.0
                             for value in base_set))
    _assert_circular_set_equal(changed, predicted)


def test_residue_leading_edge_uses_untrimmed_measurement_window(monkeypatch):
    reading = _reading("left", axis=0, phase=(0.0, 7.5), centre=(13.5, -7.0),
                       extent=20.5)
    right = _reading("right", axis=2)
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    def rectifier(_image, _axis, _matrix, _focal, *, samples, **_kwargs):
        return np.ones((samples, samples), dtype=np.float32)

    monkeypatch.setattr(lattice, "rectify_plane", rectifier)
    untrimmed = lattice_planes.plane_signature(
        image, reading, 1.0, max_cells=64)
    trimmed = lattice_planes.plane_signature(
        image, reading, 1.0, max_cells=8)
    leading = 1.0 * (CENTRE[0] - 20.5)
    assert untrimmed.origin_uv[0] == math.floor(leading)
    assert trimmed.origin_uv[0] != untrimmed.origin_uv[0]

    record = _run(reading, right)
    candidate = _candidate_for(record, "left", "u")
    assert candidate.leading_cell_coordinate == pytest.approx(leading)
    assert candidate.leading_cell_coordinate != trimmed.origin_uv[0]
    assert candidate.residue_candidates == _candidate_for(
        _run(reading, right), "left", "u").residue_candidates


def test_near_equal_match_b_is_canonicalized_to_the_unique_retained_alternative():
    for delta in (-0.9e-9, 0.9e-9):
        result = _run(_reading("left", axis=0), _reading("right", axis=2),
                      _match(left_b=1.0 + delta, right_b=1.0 - delta))
        assert result.refusal is None
        left = _candidate_for(result, "left", "u")
        right = _candidate_for(result, "right", "u")
        assert left.blocks_per_period == right.blocks_per_period == 1.0
        assert left.rational_denominator == right.rational_denominator == 1
        assert result.left_b == 1.0 + delta  # identity preserves the admitted match


def test_measured_zero_phase_is_valid_but_missing_phase_evidence_is_not():
    measured = _run(_reading("left", axis=0, phase=(0.0, 7.5)),
                    _reading("right", axis=2))
    zero = _candidate_for(measured, "left", "u")
    assert zero.refusal is None
    assert zero.phase_fraction == 0.0
    assert zero.quality.response_magnitude > 1e-12

    absent = _run(_reading("left", axis=0, evidence=False),
                  _reading("right", axis=2))
    for value in absent.phase_candidates[:2]:
        assert value.refusal == "phase_evidence_absent"
        assert value.residue_candidates == ()


def test_periodic_modulo_endpoint_is_canonical_zero():
    reading = _reading("left", axis=0, phase=(0.0, 7.5), centre=(5.0, 1.0),
                       extent=4.0)
    right = _reading("right", axis=2)
    result = _run(reading, right)
    candidate = _candidate_for(result, "left", "u")
    assert candidate.leading_cell_coordinate == 1.0
    assert candidate.residue_candidates == (0.0,)

    close_to_one = 1.0 - 5e-13
    reading = _reading("left", axis=0,
                       centre=(EXTENT + close_to_one, CENTRE[1]))
    near_endpoint = _candidate_for(_run(reading, _reading("right", axis=2)),
                                   "left", "u")
    assert near_endpoint.residue_candidates[0] == pytest.approx(close_to_one, abs=1e-15)
    assert near_endpoint.residue_candidates[0] != 0.0


def _unavailable_reading(frame: str, reason: str):
    period = 1.0 if reason == "insufficient_period" else PERIOD
    if reason == "empty_profile":
        response, mass, concentration = 0.0, 0.0, None
    elif reason == "zero_resultant":
        response, mass, concentration = 0.0, 12.0, 0.0
    else:
        response, mass, concentration = None, None, None
    evidence = lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=(None, None), period_pixels=period,
        patch_centre_uv=CENTRE, extent=EXTENT, samples=SAMPLES,
        response_magnitude_uv=(response, response),
        profile_mass_uv=(mass, mass),
        concentration_uv=(concentration, concentration),
        unavailable_reason_uv=(reason, reason),
    )
    return _reading(frame, axis=(0 if frame == "left" else 2), phase=(0.0, 0.0),
                    period=period, evidence=False), evidence


@pytest.mark.parametrize("reason", [
    "insufficient_period", "empty_profile", "nonfinite_profile",
    "zero_resultant", "invalid_profile", "invalid_period",
])
def test_each_p1a_axis_unavailability_reason_is_preserved(reason):
    left, evidence = _unavailable_reading("left", reason)
    left = replace(left, phase_evidence=evidence)
    result = _run(left, _reading("right", axis=2))
    for value in result.phase_candidates[:2]:
        assert value.refusal == "phase_axis_unavailable"
        assert value.refusal_detail == reason
        assert value.quality.unavailable_reason == reason
        assert value.residue_candidates == ()


def test_relative_shift_and_existing_support_are_copied_without_phase_fusion():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    for shift in ((0, 0), (-3, 4), (91, -127)):
        match = _match(left_b=0.5, right_b=1.0, shift=shift)
        result = _run(left, right, match)
        copied = result.relative_shift
        assert copied.cell_shift_rc == shift
        assert (copied.left_frame, copied.right_frame) == ("left", "right")
        assert (copied.left_axis, copied.right_axis) == (0, 2)
        assert (copied.left_b, copied.right_b, copied.quadrant_of_right) == (0.5, 1.0, 3)
        assert (copied.psr, copied.peak, copied.second_peak, copied.overlap_cells,
                copied.score) == (match.psr, match.peak, match.second_peak,
                                  match.overlap_cells, match.score)
        assert result.phase_candidates[0].blocks_per_period == 0.5
        assert result.phase_candidates[2].blocks_per_period == 1.0

    absent_shift = _run(left, right, _match(shift=None))
    assert absent_shift.relative_shift.refusal == "cell_shift_absent"
    assert absent_shift.relative_shift.cell_shift_rc is None


@pytest.mark.parametrize("psr", (4.5, float("inf")))
def test_relative_shift_v2_psr_round_trip_is_canonical_and_immutable(psr):
    match = _match(psr=psr)
    before = replace(match)
    support = _run(_reading("left", axis=0), _reading("right", axis=2),
                   match).relative_shift
    assert support.schema == plane_residue.RELATIVE_SHIFT_SUPPORT_SCHEMA
    assert support.version == plane_residue.RELATIVE_SHIFT_SUPPORT_VERSION == 2
    assert support.correlation_algorithm_id == \
        plane_graph.PLANE_CORRELATION_ALGORITHM
    tag = support.to_record()["psr"]
    assert plane_residue._psr_from_record(tag) == psr
    assert tag == ({"value": None, "positive_infinity": True}
                   if psr == float("inf") else
                   {"value": psr, "positive_infinity": False})
    assert match == before


@pytest.mark.parametrize("tag", (
    {"value": 1.0, "positive_infinity": True},
    {"value": None, "positive_infinity": False, "extra": 1},
    {"value": float("nan"), "positive_infinity": False},
    {"value": float("-inf"), "positive_infinity": False},
))
def test_relative_shift_psr_rejects_inconsistent_or_nonfinite_tags(tag):
    with pytest.raises(ValueError):
        plane_residue._psr_from_record(tag)


@pytest.mark.parametrize("field,value", (
    ("schema", "mc-photogrammetry-relative-shift-support"),
    ("version", 1),
    ("correlation_algorithm_id", "fixed-mask-v1"),
))
def test_relative_shift_v1_and_mixed_provenance_are_rejected(field, value):
    support = _run(_reading("left", axis=0),
                   _reading("right", axis=2)).relative_shift
    with pytest.raises(ValueError, match="provenance"):
        replace(support, **{field: value})


def test_source_region_identity_is_retained_and_b_bound_is_reading_local():
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 3:11] = True
    polygon = ((3.0, 2.0), (11.0, 2.0), (11.0, 8.0), (3.0, 8.0))
    source = lattice_planes._make_region_evidence(mask, 0, polygon, ("test",))
    reading = _reading("left", axis=0, region_evidence=source, polygon=polygon)
    result = _run(reading, _reading("right", axis=2))
    candidates = [value for value in result.phase_candidates[:2]]
    assert {value.source_region_id for value in candidates} == {source.source_region_id}
    assert all(value.source_region_id != "" for value in candidates)

    mismatched = replace(reading.phase_evidence, source_region_id="f" * 64)
    with pytest.raises(ValueError, match="not bound"):
        replace(reading, phase_evidence=mismatched)


def test_nonfinite_b_or_derived_geometry_abstains_and_remains_json_safe():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    invalid_match = _match(left_b=float("nan"))
    invalid = _run(left, right, invalid_match)
    assert invalid.refusal == "invalid_period_fraction"
    assert invalid.left_b is None
    json.loads(invalid.to_json_bytes())

    object.__setattr__(left, "extent", float("nan"))
    invalid_geometry = _run(left, right)
    assert all(value.refusal == "nonfinite_derived_value"
               for value in invalid_geometry.phase_candidates[:2])
    json.loads(invalid_geometry.to_json_bytes())


def test_missing_mismatched_or_ambiguous_readings_are_typed_refusals():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    missing = plane_residue.hypothesis_phase_shift_evidence(
        _match(), {("left", PLANE): left})
    assert missing.refusal == "reading_missing"
    assert missing.phase_candidates == ()

    wrong_axis = _reading("left", axis=1)
    mismatched = _run(wrong_axis, right)
    assert mismatched.refusal == "reading_identity_mismatch"
    assert mismatched.phase_candidates == ()

    duplicate = replace(left, alternatives=left.alternatives + (left.alternatives[4],))
    ambiguous = _run(duplicate, right)
    assert ambiguous.refusal == "alternative_not_retained"
    assert ambiguous.phase_candidates == ()

    absent = _run(left, right, _match(left_b=0.75, right_b=0.75))
    assert absent.refusal == "alternative_not_retained"


def test_records_are_deterministic_and_keep_only_relative_evidence_fields():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    match = _match(left_b=1.0 / 16.0, right_b=0.5, shift=(-4, 7))
    forward = plane_residue.hypothesis_phase_shift_evidence(
        match, {("left", PLANE): left, ("right", PLANE): right})
    reverse = plane_residue.hypothesis_phase_shift_evidence(
        match, {("right", PLANE): right, ("left", PLANE): left})
    assert forward.to_json_bytes() == reverse.to_json_bytes()
    record = forward.to_record()
    keys = set(record)
    assert keys == {"identity", "phase_candidates", "relative_shift", "refusal"}
    encoded = forward.to_json_bytes().decode("utf-8")
    for forbidden in ("world_axis", "world_coordinate", "lateral_origin",
                      "placement", "seam_id", "chosen_period_index"):
        assert forbidden not in encoded
    assert json.loads(encoded) == record
    assert record["identity"] == {
        "left_frame": "left", "right_frame": "right",
        "left_plane": PLANE, "right_plane": PLANE,
        "left_axis": 0, "right_axis": 2,
        "left_b": 1.0 / 16.0, "right_b": 0.5, "quadrant_of_right": 3,
    }


def test_frozen_candidate_records_reject_directly_mutated_invariants():
    result = _run(_reading("left", axis=0), _reading("right", axis=2),
                  _match(left_b=0.5, right_b=0.5))
    candidate = result.phase_candidates[0]

    with pytest.raises(ValueError, match="rational"):
        replace(candidate, rational_denominator=4)
    with pytest.raises(ValueError, match="amplification"):
        replace(candidate, phase_error_amplification=0.25)
    with pytest.raises(ValueError, match="canonical formula"):
        replace(candidate, residue_candidates=(candidate.residue_candidates[0] + 0.1,
                                                candidate.residue_candidates[1]))
    with pytest.raises(ValueError, match="incomplete"):
        replace(candidate, residue_candidates=candidate.residue_candidates[:1])
    with pytest.raises(ValueError, match="incomplete"):
        replace(candidate, quality=None)

    quality = candidate.quality
    with pytest.raises(ValueError, match="finite"):
        replace(quality, response_magnitude=float("nan"))
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(quality, response_magnitude=-1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(quality, profile_mass=-1.0)
    with pytest.raises(ValueError, match="concentration"):
        replace(quality, concentration=1.01)
    with pytest.raises(ValueError, match="unknown unavailable reason"):
        replace(quality, unavailable_reason="made_up_reason")


def test_frozen_unavailable_candidate_quality_and_reason_must_agree():
    left, evidence = _unavailable_reading("left", "zero_resultant")
    left = replace(left, phase_evidence=evidence)
    record = _run(left, _reading("right", axis=2))
    candidate = record.phase_candidates[0]
    with pytest.raises(ValueError, match="exact P1-A reason"):
        replace(candidate, refusal_detail="empty_profile")
    with pytest.raises(ValueError, match="exact P1-A reason"):
        replace(candidate, quality=replace(candidate.quality,
                                           unavailable_reason="insufficient_period"))


def test_tampered_missing_phase_without_reason_returns_typed_refusal():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    evidence = left.phase_evidence
    # Bypass the upstream frozen P1-A constructor to exercise this module's
    # defensive boundary against an impossible phase/reason combination.
    object.__setattr__(evidence, "phase_uv_pixels", (None, evidence.phase_uv_pixels[1]))

    result = _run(left, right)
    assert result.refusal is None
    candidate = _candidate_for(result, "left", "u")
    assert candidate.refusal == "candidate_set_invalid"
    assert candidate.refusal_detail is None
    assert candidate.quality.unavailable_reason is None
    assert candidate.residue_candidates == ()
    assert result.to_json_bytes()


@pytest.mark.parametrize(("field", "value"), [
    ("left_frame", "foreign-frame"),
    ("right_plane", PLANE + 1),
    ("left_axis", 1),
    ("right_axis", 1),
    ("left_b", 0.5),
    ("right_b", 1.0),
    ("quadrant_of_right", 2),
])
def test_hypothesis_rejects_foreign_nested_shift_identity(field, value):
    result = _run(_reading("left", axis=0), _reading("right", axis=2),
                  _match(left_b=1.0 / 16.0, right_b=0.5))
    foreign_shift = replace(result.relative_shift, **{field: value})
    with pytest.raises(ValueError, match="relative shift identity"):
        replace(result, relative_shift=foreign_shift)


def test_hypothesis_binds_phase_reading_axis_and_refusal_completeness():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    result = _run(left, right, _match(left_b=1.0 / 16.0, right_b=0.5))
    with pytest.raises(ValueError, match="hypothesis identity and b pair"):
        replace(result, phase_candidates=(
            replace(result.phase_candidates[0], reading_axis=1),
            *result.phase_candidates[1:],
        ))

    other_b = _run(left, right, _match(left_b=0.5, right_b=0.5))
    with pytest.raises(ValueError, match="hypothesis identity and b pair"):
        replace(result, phase_candidates=(
            other_b.phase_candidates[0], *result.phase_candidates[1:]))

    with pytest.raises(ValueError, match="cannot carry phase candidates"):
        replace(result, refusal="reading_missing")


def test_relative_shift_frozen_record_enforces_refusal_and_nonfinite_fields():
    record = _run(_reading("left", axis=0), _reading("right", axis=2)).relative_shift
    with pytest.raises(ValueError, match="refusal disagrees"):
        replace(record, cell_shift_rc=None)
    with pytest.raises(ValueError, match="nonfinite field list"):
        replace(record, nonfinite_fields=("imaginary",),
                refusal="nonfinite_derived_value")
    with pytest.raises(ValueError, match="nonfinite support fields"):
        replace(record, nonfinite_fields=("psr",),
                refusal="nonfinite_derived_value")

    absent_psr = replace(record, psr=None, nonfinite_fields=("psr",),
                         refusal="nonfinite_derived_value")
    assert absent_psr.to_record()["refusal"] == "nonfinite_derived_value"

    absent_overlap = replace(record, overlap_cells=None,
                              nonfinite_fields=("overlap_cells",),
                              refusal="nonfinite_derived_value")
    assert absent_overlap.to_record()["overlap_cells"] is None


def test_negative_match_overlap_is_normalized_to_typed_nonfinite_refusal():
    left, right = _reading("left", axis=0), _reading("right", axis=2)
    match = _match()
    # PlaneMatch permits this field through construction; force the corrupted
    # value to exercise the residue boundary regardless of upstream behavior.
    object.__setattr__(match, "overlap_cells", -3)

    result = _run(left, right, match)
    support = result.relative_shift
    assert support.overlap_cells is None
    assert support.nonfinite_fields == ("overlap_cells",)
    assert support.refusal == "nonfinite_derived_value"
    assert support.to_record()["overlap_cells"] is None


def test_module_imports_no_capture_evaluator_or_reference_metadata():
    source = Path(plane_residue.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert set(imported) == {
        "__future__", "dataclasses", "fractions", "json", "math", "typing",
        "mc_photogrammetry.reconstruction",
    }
    for forbidden in ("dataset", "evaluation", "capture", "manifest", "reference"):
        assert all(forbidden not in value.lower() for value in imported)
