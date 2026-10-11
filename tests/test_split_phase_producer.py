"""Isolated V2-B P0 split-phase producer controls (contract 2026-09-14).

These tests exercise only the frozen P0 producer surface added to
``lattice_planes``: the pure selector/lane plan, the typed records, the
planned-only budget stage, the embedded D-profile transport, and the two
independent D-leak controls (per-axis ``D_a`` and the simultaneous
``D_u ∩ D_v`` intersection).  No image, manifest, label, evaluator, graph,
runner, or ``plane_residue`` symbol is used; every fixture is synthetic.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import inspect

from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes as lp

HEIGHT, WIDTH = 12, 10
PERIOD_PIXELS = 4.0
SAMPLES = 8
EXTENT = 1.0


def _geometry(period: float = PERIOD_PIXELS) -> lp.SplitPhaseGeometry:
    return lp.SplitPhaseGeometry(
        centre=(0.0, 0.0), extent=EXTENT, samples=SAMPLES,
        rectified_pixel_step=lattice.rectified_pixel_step(EXTENT, SAMPLES),
        period_pixels=period,
    )


def _caps(**overrides: int) -> dict:
    caps = {name: 1_000_000 for name in lp._SPLIT_BUDGET_FIELDS}
    caps.update(overrides)
    return caps


def _budget(**overrides: int) -> lp.SplitPhaseProducerBudget:
    return lp._make_budget(_caps(**overrides))


def _patch(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(size=(HEIGHT, WIDTH))).astype("<f8")


def _produce(patch, *, budget=None, geometry=None, config=None, **overrides):
    return lp.produce_split_phase_evidence(
        patch,
        geometry=geometry if geometry is not None else _geometry(),
        frame="frame-0", plane_index=0, reading_axis=2,
        source_region_id=None,
        partition_config=config if config is not None else lp.SPLIT_PHASE_PARTITION_CONFIG,
        budget=budget if budget is not None else _budget(),
        **overrides,
    )


def _plans(height=HEIGHT, width=WIDTH, config=None):
    config = config if config is not None else lp.SPLIT_PHASE_PARTITION_CONFIG
    return (lp.split_phase_lane_plan("u", height, width, config),
            lp.split_phase_lane_plan("v", width, height, config))


def _tamper_copy(record: lp.SplitPhaseProducerResult, **changes):
    return replace(record, **changes)


# --- 1. Complete branch, exact aggregate counters, canonical record ---------

def test_complete_result_branch_and_aggregate_counters():
    patch = _patch()
    result = _produce(patch)
    assert result.status == "complete"
    assert result.budget_stage is None
    assert result.refusals == ()
    assert result.runtime_work == result.planned_work
    assert result.planned_work is not None
    assert [(item.local_axis, item.role) for item in result.lane_sets] == [
        ("u", "G"), ("u", "D"), ("v", "G"), ("v", "D")]
    assert len(result.generation_evidence) == 1
    assert [item.local_axis for item in result.seam_profile_evidence] == ["u", "v"]

    work = result.planned_work
    assert work.lanes_per_axis == max(HEIGHT, WIDTH)
    assert work.patch_samples == HEIGHT * WIDTH
    assert work.inline_patch_samples == HEIGHT * WIDTH
    assert work.profile_terms == HEIGHT + WIDTH
    assert work.d_profile_terms == 3 * (HEIGHT + WIDTH)
    assert work.payload_bytes == (2 * len(lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION)
                                  + 48 + 24 * (HEIGHT + WIDTH))
    # L is the present lane-set count (four for the built-in g = 0 config).
    assert work.source_records == 4 + 3
    assert work.replay_checks == 4 + 4

    # The frozen D payload framing carries the version bytes before the three
    # length-prefixed canonical little-endian arrays.
    for seam in result.seam_profile_evidence:
        samples_raw = base64.b64decode(seam.profile_samples_float64_le_base64)
        weights_raw = base64.b64decode(seam.profile_weights_float64_le_base64)
        indices_raw = base64.b64decode(seam.axis_bin_indices_int64_le_base64)
        blob = b"".join(
            len(raw).to_bytes(8, "little") + raw
            for raw in (samples_raw, weights_raw, indices_raw))
        expected = hashlib.sha256(
            lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION + blob).hexdigest()
        assert seam.payload_sha256 == expected
        assert len(samples_raw) == len(weights_raw) == len(indices_raw) == 8 * seam.sample_count
        assert seam.weight_count == seam.index_count == seam.sample_count
        indices = np.frombuffer(indices_raw, dtype="<i8")
        assert np.array_equal(indices, np.arange(seam.sample_count, dtype="<i8"))


def test_result_to_record_field_order_and_non_self_referential_digest():
    result = _produce(_patch())
    record = result.to_record()
    assert list(record) == [
        "status", "producer_version_hex", "producer_version_sha256", "geometry",
        "reading_identity", "patch_ref", "partition_config_sha256", "lane_sets",
        "generation_evidence", "seam_profile_evidence", "planned_work",
        "runtime_work", "budget_stage", "refusals", "selector_salt_sha256",
        "partition_sha256", "G_u_sha256", "G_v_sha256", "D_u_sha256",
        "D_v_sha256", "X_u_sha256", "X_v_sha256", "producer_input_sha256",
        "result_sha256"]
    material = {key: value for key, value in record.items() if key != "result_sha256"}
    assert record["result_sha256"] == hashlib.sha256(
        lp.SPLIT_PHASE_CANONICAL_JSON(material)).hexdigest()
    assert lp.SPLIT_PHASE_CANONICAL_JSON(record) == result.to_json_bytes()


def test_validator_accepts_complete_replay_and_rejects_mismatches():
    patch = _patch()
    geometry = _geometry()
    config = lp.SPLIT_PHASE_PARTITION_CONFIG
    budget = _budget()
    result = _produce(patch, geometry=geometry, config=config, budget=budget)
    assert lp.validate_split_phase_evidence(
        result, patch, geometry=geometry, partition_config=config, budget=budget)
    # A different in-memory P (shape or bytes) is a patch mismatch.
    assert not lp.validate_split_phase_evidence(
        result, patch * 2.0, geometry=geometry, partition_config=config,
        budget=budget)
    assert not lp.validate_split_phase_evidence(
        result, patch[:-1, :], geometry=geometry, partition_config=config,
        budget=budget)
    assert not lp.validate_split_phase_evidence(
        result, None, geometry=geometry, partition_config=config, budget=budget)
    # A different frozen geometry is not the retained context.
    assert not lp.validate_split_phase_evidence(
        result, patch, geometry=_geometry(period=5.0), partition_config=config,
        budget=budget)
    # Non-complete branches are never accepted.
    exhausted = _produce(patch, budget=_budget(max_replay_checks=1))
    assert exhausted.status == "budget_exhausted"
    assert not lp.validate_split_phase_evidence(
        exhausted, patch, geometry=geometry, partition_config=config,
        budget=budget)
    refused = lp.produce_split_phase_evidence(
        None, geometry=geometry, frame="frame-0", plane_index=0, reading_axis=2,
        source_region_id=None, partition_config=config, budget=budget)
    assert refused.status == "refused"
    assert not lp.validate_split_phase_evidence(
        refused, patch, geometry=geometry, partition_config=config, budget=budget)


# --- 2. Planned-only budget staging, no runtime stage ----------------------

def test_planned_only_exhaustion_and_no_runtime_helpers():
    assert not hasattr(lp, "_split_observed_counters")
    assert not hasattr(lp, "_split_runtime_trigger")

    patch = _patch()
    baseline = _produce(patch)
    planned = baseline.planned_work
    geometry = _geometry()
    config = lp.SPLIT_PHASE_PARTITION_CONFIG
    for cap in lp._SPLIT_BUDGET_FIELDS:
        if cap == "max_inline_patch_samples":
            # This cap selects the patch transport mode rather than a work
            # amount; the counter is the number of samples actually inlined.
            continue
        work_field = lp._SPLIT_CAP_TO_WORK[cap]
        observed = getattr(planned, work_field)
        if observed == 0:
            # Caps are positive, so a zero work counter has no representable
            # "one-below" cap; the smallest legal cap still succeeds.
            assert _produce(patch, budget=_budget(**{cap: 1})).status == "complete"
            continue
        # Exact limit and one-over both succeed.
        assert _produce(patch, budget=_budget(**{cap: observed})).status == "complete"
        assert _produce(patch, budget=_budget(**{cap: observed + 1})).status == "complete"
        # One-below the planned work exhausts atomically before any evidence.
        below = _produce(patch, budget=_budget(**{cap: observed - 1}))
        assert below.status == "budget_exhausted"
        assert below.budget_stage == "planned"
        assert below.runtime_work is None
        assert getattr(below.planned_work, work_field) == observed
        assert below.planned_work == planned
        assert below.geometry is None and below.patch_ref is None
        assert below.reading_identity is None
        assert below.lane_sets is None and below.generation_evidence is None
        assert below.seam_profile_evidence is None
        assert below.selector_salt_sha256 is None and below.partition_sha256 is None
        assert all(getattr(below, name) is None for name in lp._SPLIT_COORDINATE_FIELDS)
        assert len(below.refusals) == 1
        assert below.refusals[0].reason == "budget_exhausted"
        assert below.refusals[0].detail == {
            "budget_stage": "planned", "cap": cap, "limit": observed - 1,
            "observed": observed}


def test_inline_transport_cap_switches_mode_without_exhaustion():
    patch = _patch()
    inline = _produce(patch, budget=_budget(max_inline_patch_samples=HEIGHT * WIDTH))
    assert inline.status == "complete"
    assert inline.patch_ref.mode == "inline"
    assert inline.planned_work.inline_patch_samples == HEIGHT * WIDTH
    digest = _produce(patch, budget=_budget(max_inline_patch_samples=HEIGHT * WIDTH - 1))
    assert digest.status == "complete"
    assert digest.patch_ref.mode == "digest"
    assert digest.patch_ref.inline_samples_float64_le_base64 is None
    assert digest.planned_work.inline_patch_samples == 0


def test_budget_stage_runtime_is_invalid_and_refused_work_is_consistent():
    refused = lp.produce_split_phase_evidence(
        None, geometry=_geometry(), frame="f", plane_index=0, reading_axis=2,
        source_region_id=None, partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG,
        budget=_budget())
    assert refused.status == "refused"
    with pytest.raises(ValueError, match="budget_stage_invalid"):
        _tamper_copy(refused, budget_stage="runtime")

    planned = _produce(_patch()).planned_work
    version_hex = lp.SPLIT_PHASE_PRODUCER_VERSION.hex()
    version_sha = hashlib.sha256(lp.SPLIT_PHASE_PRODUCER_VERSION).hexdigest()
    ok = lp._make_result({
        "status": "refused", "producer_version_hex": version_hex,
        "producer_version_sha256": version_sha,
        "planned_work": planned, "runtime_work": planned,
        "refusals": (lp._make_refusal("split_not_reconstructible"),)})
    assert ok.planned_work == ok.runtime_work
    with pytest.raises(ValueError, match="work_over_cap"):
        lp._make_result({
            "status": "refused", "producer_version_hex": version_hex,
            "producer_version_sha256": version_sha,
            "planned_work": planned, "runtime_work": None,
            "refusals": (lp._make_refusal("split_not_reconstructible"),)})


def test_missing_patch_is_p0_patch_unavailable():
    result = lp.produce_split_phase_evidence(
        None, geometry=_geometry(), frame="f", plane_index=0, reading_axis=2,
        source_region_id=None, partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG,
        budget=_budget())
    assert result.status == "refused"
    assert [item.reason for item in result.refusals] == ["p0_patch_unavailable"]


def test_non_builtin_config_is_refused_by_producer_and_validator():
    alternative = lp.SplitPhasePartitionConfig(
        partition_version=lp.SPLIT_PHASE_PARTITION_VERSION,
        selector_version=lp.SPLIT_SELECTOR_VERSION,
        selector_salt="v2b-p0-split-selector-salt-1",
        min_role_lanes=2, profile_stencil_radius_samples=0,
        seam_stencil_radius_samples=0, guard_lanes=0)
    patch = _patch()
    geometry = _geometry()
    budget = _budget()
    result = _produce(patch, geometry=geometry, config=alternative, budget=budget)
    assert result.status == "refused"
    assert [item.reason for item in result.refusals] == ["partition_config_not_builtin"]
    assert not lp.validate_split_phase_evidence(
        _produce(patch), patch, geometry=geometry, partition_config=alternative,
        budget=budget)


# --- 3. Coordinate-only role digests and lane-plan freeze ------------------

def test_role_digests_are_coordinate_only():
    patch = _patch()
    result = _produce(patch)
    pu, pv = _plans()
    mutated = patch.copy()
    mutated[pu.d_start:pu.d_stop, :] += 3.0
    changed = _produce(mutated)
    # Coordinate-set digests never read a patch value; value content is bound
    # only by patch_sha256 / *_profile_sha256 / producer_input_sha256 / result_sha256.
    assert changed.G_u_sha256 == result.G_u_sha256
    assert changed.G_v_sha256 == result.G_v_sha256
    assert changed.D_u_sha256 == result.D_u_sha256
    assert changed.D_v_sha256 == result.D_v_sha256
    assert changed.X_u_sha256 == result.X_u_sha256
    assert changed.X_v_sha256 == result.X_v_sha256
    assert changed.partition_sha256 == result.partition_sha256
    assert changed.selector_salt_sha256 == result.selector_salt_sha256
    assert changed.patch_ref.patch_sha256 != result.patch_ref.patch_sha256
    assert changed.producer_input_sha256 != result.producer_input_sha256
    assert changed.result_sha256 != result.result_sha256
    # The digest is sensitive to a lane coordinate, not to a value.
    assert (lp._split_coordinate_sha256("u", pu.d_start, pu.d_stop, HEIGHT, WIDTH)
            != lp._split_coordinate_sha256("u", pu.d_start + 1, pu.d_stop + 1,
                                           HEIGHT, WIDTH))


def test_lane_plan_orientation_one_constructs_and_validates():
    pu, pv = _plans()
    assert isinstance(pu, lp.SplitPhaseLanePlan)
    assert isinstance(pv, lp.SplitPhaseLanePlan)
    assert pu.orientation in (0, 1)
    assert list(pu.to_record()) == [
        "local_axis", "orientation", "lane_count", "orthogonal_extent",
        "guard_lanes", "nG", "nD", "g_start", "g_stop", "d_start", "d_stop",
        "guard_start", "guard_stop", "lane_plan_sha256"]
    if pu.orientation == 1:
        assert (pu.d_start, pu.d_stop) == (0, pu.nG)
        assert (pu.g_start, pu.g_stop) == (pu.nG + pu.guard_lanes, pu.lane_count)
    else:
        assert (pu.g_start, pu.g_stop) == (0, pu.nG)
        assert (pu.d_start, pu.d_stop) == (pu.nG + pu.guard_lanes, pu.lane_count)
    assert pu.lane_plan_sha256 == lp._split_digest(pu.to_record(), "lane_plan_sha256")


def test_lane_sets_six_records_when_guard_present():
    config = lp.SplitPhasePartitionConfig(
        partition_version=lp.SPLIT_PHASE_PARTITION_VERSION,
        selector_version=lp.SPLIT_SELECTOR_VERSION,
        selector_salt="v2b-p0-split-selector-salt-1",
        min_role_lanes=1, profile_stencil_radius_samples=1,
        seam_stencil_radius_samples=1, guard_lanes=2)
    pu, pv = _plans(config=config)
    assert isinstance(pu, lp.SplitPhaseLanePlan)
    assert isinstance(pv, lp.SplitPhaseLanePlan)
    records, digests = lp._split_lane_sets(pu, pv, HEIGHT, WIDTH)
    assert [(item.local_axis, item.role) for item in records] == [
        ("u", "G"), ("u", "guard"), ("u", "D"),
        ("v", "G"), ("v", "guard"), ("v", "D")]
    for item in records:
        key = ({"G": "G", "D": "D", "guard": "X"}[item.role]
               + "_" + item.local_axis + "_sha256")
        assert digests[key] == item.coordinate_sha256
    assert set(digests) == set(lp._SPLIT_COORDINATE_FIELDS)


def test_lane_plan_covers_both_orientations():
    config = lp.SplitPhasePartitionConfig(
        partition_version=lp.SPLIT_PHASE_PARTITION_VERSION,
        selector_version=lp.SPLIT_SELECTOR_VERSION,
        selector_salt="v2b-p0-split-selector-salt-1",
        min_role_lanes=1, profile_stencil_radius_samples=0,
        seam_stencil_radius_samples=0, guard_lanes=0)
    seen = set()
    for extent in range(2, 24):
        plan = lp.split_phase_lane_plan("u", 6, extent, config)
        assert isinstance(plan, lp.SplitPhaseLanePlan)
        seen.add(plan.orientation)
    assert seen == {0, 1}


def test_selector_domain_errors_and_min_lane_refusal_shape():
    config = lp.SplitPhasePartitionConfig(
        partition_version=lp.SPLIT_PHASE_PARTITION_VERSION,
        selector_version=lp.SPLIT_SELECTOR_VERSION,
        selector_salt="v2b-p0-split-selector-salt-1",
        min_role_lanes=2, profile_stencil_radius_samples=1,
        seam_stencil_radius_samples=1, guard_lanes=2)
    # N = 2m + g pivots; below it the pure selector returns shape (ii).
    blocked = lp.lane_role("u", 0, 5, 9, config)
    assert blocked.role is None and blocked.orientation is None
    assert blocked.refusal == "selector_min_lanes_unmet"
    plan = lp.split_phase_lane_plan("v", 6, 9, config)
    assert isinstance(plan, lp.SplitPhaseLanePlan)
    assert plan.orientation in (0, 1)
    assert plan.nG >= 2 and plan.nD >= 2
    assert plan.guard_start is not None and plan.guard_stop == plan.guard_start + 2
    assert lp.split_phase_lane_plan("v", 5, 9, config).reason == "selector_min_lanes_unmet"
    assert lp.split_phase_lane_plan("u", True, 9, config).reason == "selector_domain_invalid"
    for bad_index in (True, -1, 6, 1.5, "0"):
        with pytest.raises(ValueError, match="selector_domain_invalid"):
            lp.lane_role("u", bad_index, 6, 9, config)


# --- 4. Independent D-leak controls ----------------------------------------

def _generation(result):
    return result.generation_evidence[0]


def test_per_axis_d_u_mutation_leaves_g_u_unchanged():
    patch = _patch()
    pu, _pv = _plans()
    baseline = _produce(patch)
    mutated = patch.copy()
    mutated[pu.d_start:pu.d_stop, :] += 4.0
    changed = _produce(mutated)
    base_gen, new_gen = _generation(baseline), _generation(changed)
    # Axis-u G phase/profile component is untouched by D_u values.
    assert new_gen.phase_uv_pixels[0] == base_gen.phase_uv_pixels[0]
    assert new_gen.response_magnitude_uv[0] == base_gen.response_magnitude_uv[0]
    assert new_gen.profile_mass_uv[0] == base_gen.profile_mass_uv[0]
    assert new_gen.concentration_uv[0] == base_gen.concentration_uv[0]
    assert new_gen.G_u_sha256 == base_gen.G_u_sha256
    assert new_gen.G_u_sample_count == base_gen.G_u_sample_count
    # The other axis may change: cross-axis overlap is outside the claim.
    assert new_gen.phase_uv_pixels[1] != base_gen.phase_uv_pixels[1]
    assert baseline.lane_sets == changed.lane_sets
    assert baseline.partition_sha256 == changed.partition_sha256


def test_per_axis_d_v_mutation_leaves_g_v_unchanged():
    patch = _patch()
    _pu, pv = _plans()
    baseline = _produce(patch)
    mutated = patch.copy()
    mutated[:, pv.d_start:pv.d_stop] += 4.0
    changed = _produce(mutated)
    base_gen, new_gen = _generation(baseline), _generation(changed)
    assert new_gen.phase_uv_pixels[1] == base_gen.phase_uv_pixels[1]
    assert new_gen.response_magnitude_uv[1] == base_gen.response_magnitude_uv[1]
    assert new_gen.profile_mass_uv[1] == base_gen.profile_mass_uv[1]
    assert new_gen.concentration_uv[1] == base_gen.concentration_uv[1]
    assert new_gen.G_v_sha256 == base_gen.G_v_sha256
    assert new_gen.G_v_sample_count == base_gen.G_v_sample_count
    assert new_gen.phase_uv_pixels[0] != base_gen.phase_uv_pixels[0]
    assert baseline.lane_sets == changed.lane_sets
    assert baseline.partition_sha256 == changed.partition_sha256


def test_simultaneous_d_u_intersection_d_v_control():
    patch = _patch()
    pu, pv = _plans()
    baseline = _produce(patch)
    mutated = patch.copy()
    # Only samples in D_u ∩ D_v are retuned: they touch neither G_u nor G_v.
    mutated[pu.d_start:pu.d_stop, pv.d_start:pv.d_stop] += 4.0
    changed = _produce(mutated)
    base_gen, new_gen = _generation(baseline), _generation(changed)
    assert base_gen.g_profile_sha256 == new_gen.g_profile_sha256
    assert base_gen.generation_evidence_sha256 == new_gen.generation_evidence_sha256
    assert base_gen.phase_uv_pixels == new_gen.phase_uv_pixels
    assert base_gen.response_magnitude_uv == new_gen.response_magnitude_uv
    # Both held-out D profiles (and every value-content binding) change.
    for base_seam, new_seam in zip(baseline.seam_profile_evidence,
                                   changed.seam_profile_evidence):
        assert base_seam.d_profile_sha256 != new_seam.d_profile_sha256
        assert base_seam.payload_sha256 != new_seam.payload_sha256
        assert base_seam.D_a_sha256 == new_seam.D_a_sha256
    assert baseline.patch_ref.patch_sha256 != changed.patch_ref.patch_sha256
    assert baseline.producer_input_sha256 != changed.producer_input_sha256
    assert baseline.result_sha256 != changed.result_sha256
    # Every coordinate digest and the partition stay fixed.
    assert baseline.lane_sets == changed.lane_sets
    assert baseline.partition_sha256 == changed.partition_sha256
    for name in lp._SPLIT_COORDINATE_FIELDS:
        assert getattr(baseline, name) == getattr(changed, name)


# --- 5. Dyadic zero, textureless transport, isolation ----------------------

def test_canonical_dyadic_zero_and_noncanonical_rejection():
    assert lp._split_reduced_dyadic((0, 1), allow_zero=True, allow_one=True) == (0, 1)
    assert lp._split_reduced_dyadic((1, 2), allow_zero=True, allow_one=True) == (1, 2)
    for bad in ((0, 2), (2, 4), (1, 3), (3, 2), (-1, 2)):
        with pytest.raises(ValueError):
            lp._split_reduced_dyadic(bad, allow_zero=True, allow_one=True)
    with pytest.raises(ValueError):
        lp._split_reduced_dyadic((0, 1), allow_zero=False, allow_one=True)


def test_textureless_patch_yields_typed_abstention_not_a_phase():
    result = _produce(np.zeros((HEIGHT, WIDTH), dtype="<f8"))
    assert result.status == "complete"
    generation = _generation(result)
    assert generation.phase_uv_pixels == (None, None)
    assert generation.unavailable_reason_uv == ("empty_profile", "empty_profile")
    # The empty profile carries its measured zero mass/response, never a phase.
    assert generation.response_magnitude_uv == (0.0, 0.0)
    assert generation.profile_mass_uv == (0.0, 0.0)
    # Zero held-out energy stays a valid sample rather than a fabricated phase.
    for seam in result.seam_profile_evidence:
        samples = np.frombuffer(
            base64.b64decode(seam.profile_samples_float64_le_base64), dtype="<f8")
        assert np.all(samples == 0.0)
        assert seam.coverage["valid"] == seam.coverage["total"]


def test_module_does_not_import_or_reference_plane_residue():
    source = inspect.getsource(lp)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("plane_residue" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "plane_residue" not in (node.module or "")
        elif isinstance(node, ast.Name):
            assert node.id != "plane_residue"
        elif isinstance(node, ast.Attribute):
            assert node.attr != "plane_residue"


def test_calibration_grid_schema_round_trip():
    # The digest is over the record's JSON-safe (list) serialization.
    serialized = {
        "grid_version": "v2b-p0-calibration-grid-1",
        "evaluation_only": True,
        "target_power": [1, 2],
        "tuples": [
            [[0, 1], [1, 2], [0, 1], 4, 2],
            [[1, 4], [1, 2], [1, 4], 8, 3],
        ],
        "null_control_ids": ["a" * 64],
        "seam_control_ids": ["b" * 64],
    }
    grid_sha = lp._split_digest(serialized, "grid_sha256")
    grid = lp.SplitPhaseCalibrationGrid(
        grid_version="v2b-p0-calibration-grid-1", evaluation_only=True,
        target_power=(1, 2),
        tuples=(((0, 1), (1, 2), (0, 1), 4, 2),
                ((1, 4), (1, 2), (1, 4), 8, 3)),
        null_control_ids=("a" * 64,), seam_control_ids=("b" * 64,),
        grid_sha256=grid_sha)
    assert grid.to_record() == {**serialized, "grid_sha256": grid_sha}
    assert grid.grid_sha256 == lp._split_digest(grid.to_record(), "grid_sha256")
    assert grid.target_power == (1, 2)
    with pytest.raises(ValueError):
        replace(grid, tuples=(((1, 2), (1, 2), (0, 1), 4, 2),
                              ((0, 1), (1, 2), (0, 1), 8, 3)))
    with pytest.raises(ValueError):
        replace(grid, target_power=(0, 2))


# --- 6. Resealed budget_exhausted constructor invariants (Luna reject) ------

def _resealed_exhausted(planned, detail, *, runtime_work=None,
                        budget_stage="planned", producer_input_sha256="a" * 64):
    """Reseal a budget_exhausted envelope so only branch validation can fail.

    ``_make_result`` recomputes ``result_sha256`` over the exact record, so a
    rejection proves the frozen branch/detail invariant rather than a stale
    digest.  This mirrors the Luna counterexample, which resealed an arbitrary
    ``{"nonsense": true}`` detail on an under-cap planned plan and was accepted.
    """
    return lp._make_result({
        "status": "budget_exhausted",
        "producer_version_hex": lp.SPLIT_PHASE_PRODUCER_VERSION.hex(),
        "producer_version_sha256": hashlib.sha256(
            lp.SPLIT_PHASE_PRODUCER_VERSION).hexdigest(),
        "partition_config_sha256": lp.SPLIT_PHASE_PARTITION_CONFIG_SHA256,
        "planned_work": planned,
        "runtime_work": runtime_work,
        "budget_stage": budget_stage,
        "refusals": (lp._make_refusal("budget_exhausted", detail),),
        "producer_input_sha256": producer_input_sha256,
    })


def test_forged_budget_exhausted_detail_shapes_are_rejected():
    planned = _produce(_patch()).planned_work
    observed = planned.patch_samples
    valid = {"budget_stage": "planned", "cap": "max_patch_samples",
             "limit": observed - 1, "observed": observed}
    # Control: the genuine four-key over-cap detail reseals and is accepted.
    assert _resealed_exhausted(planned, valid).refusals[0].detail == valid
    for detail in (
        {"nonsense": True},                                    # arbitrary
        {"budget_stage": "planned", "cap": "max_patch_samples",
         "limit": 1},                                          # missing observed
        {**valid, "extra": 0},                                 # extra key
        {**valid, "budget_stage": "runtime"},                  # wrong stage
        {**valid, "cap": "max_not_a_cap"},                     # unknown cap
        {**valid, "cap": ["max_patch_samples"]},               # unhashable cap
        {**valid, "cap": 7},                                   # non-str cap
        {**valid, "limit": "1"},                               # wrong type
        {**valid, "limit": True},                              # bool is not int
        {**valid, "limit": 0},                                 # illegal domain
        {**valid, "observed": observed + 1},                   # wrong observed
        {**valid, "observed": True},                           # bool observed
        {**valid, "observed": observed, "limit": observed},    # observed<=limit
        {**valid, "observed": observed, "limit": observed + 1},
        {**valid, "cap": "max_guard_samples_per_axis", "limit": 1,
         "observed": planned.guard_samples_per_axis},          # zero counter
    ):
        with pytest.raises(ValueError, match="split_not_reconstructible"):
            _resealed_exhausted(planned, detail)


def test_luna_under_cap_forgery_is_rejected():
    # The exact Luna counterexample: under-cap planned_work plus an arbitrary
    # refusal detail.  Both the detail shape and the over-cap requirement fail.
    patch = _patch()
    planned = _produce(patch).planned_work
    assert lp._split_work_within_cap(planned, _budget())
    with pytest.raises(ValueError, match="split_not_reconstructible"):
        _resealed_exhausted(planned, {"nonsense": True})
    valid = {"budget_stage": "planned", "cap": "max_patch_samples",
             "limit": 1, "observed": planned.patch_samples}
    for kwargs in ({"runtime_work": planned}, {"budget_stage": None}):
        with pytest.raises(ValueError, match="split_not_reconstructible"):
            _resealed_exhausted(planned, valid, **kwargs)


def test_all_genuine_producer_exhaustions_reseal_and_are_accepted():
    patch = _patch()
    planned = _produce(patch).planned_work
    seen = 0
    for cap in lp._SPLIT_BUDGET_FIELDS:
        if cap == "max_inline_patch_samples":
            continue
        observed = getattr(planned, lp._SPLIT_CAP_TO_WORK[cap])
        if observed == 0:
            continue
        below = _produce(patch, budget=_budget(**{cap: observed - 1}))
        assert below.status == "budget_exhausted"
        detail = below.refusals[0].detail
        assert detail == {"budget_stage": "planned", "cap": cap,
                          "limit": observed - 1, "observed": observed}
        resealed = _resealed_exhausted(
            below.planned_work, detail,
            producer_input_sha256=below.producer_input_sha256)
        assert resealed.to_record() == below.to_record()
        seen += 1
    # Every representable cap except the transport selector and the zero guard.
    assert seen == len(lp._SPLIT_BUDGET_FIELDS) - 2
