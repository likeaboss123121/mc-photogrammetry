"""Isolated V2-B split-phase calibration controls (prereg 2026-09-14).

These tests exercise only the evaluation-only calibration harness: the frozen
1000-tuple grid, the deterministic RNG-free generator and its exact ground
truth, the literal ``x_i``/``B``/``I``/``accept`` classifier, the label-blind
prediction freeze, the corpus/sample/control/provenance/patch disjointness
re-checks, the frozen budgets, and the score/lexicographic-selection path.

The forward oracle below is a second, independent implementation of the frozen
section 7 formulas.  It deliberately does **not** import
``split_phase_calibration`` (the code under test): it declares each control's
class and ``alpha`` before synthesis and recomputes the patch and the
open-form ``E``/``C`` reductions directly from the frozen ``lattice_planes``
partition config.  Expectations are never derived from the harness.
"""
from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes as lp
from mc_photogrammetry.reconstruction import split_phase_calibration as sp

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / "src" / "mc_photogrammetry" / "reconstruction"
               / "split_phase_calibration.py")
SCRIPT_PATH = ROOT / "scripts" / "fit_split_phase_calibration.py"

PATCH_SHAPE = (64, 64)
TAU_BANDS = (Fraction(0), Fraction(1, 8), Fraction(1, 4),
             Fraction(3, 8), Fraction(1, 2))
B_SET = (0.5, 1.0, 2.0)
P_SET = (8.0, 16.0, 32.0)


# --------------------------------------------------------------------------
# Independent oracle (never imports split_phase_calibration)
# --------------------------------------------------------------------------

def o_mod_one(value: Fraction) -> Fraction:
    return value - (value.numerator // value.denominator)


def o_d1(first: Fraction, second: Fraction) -> Fraction:
    delta = abs(first - second)
    return min(delta, 1 - delta)


def o_class(k: int) -> str:
    if k < 16:
        return "textureless"
    if k < 32:
        return "same_block"
    if k < 48:
        return "seam_null"
    return "positive"


def o_axis(k: int, w: int, j: int) -> str:
    return "u" if (k + w + j) % 2 == 0 else "v"


def o_sign(k: int, w: int) -> int:
    return 1 if (k + w) % 2 == 0 else -1


def o_b(k: int, w: int, sigma: int) -> float:
    return B_SET[(k + w + sigma) % 3]


def o_period(k: int, w: int, sigma: int) -> float:
    return P_SET[(k + w + sigma) % 3]


def o_alpha(k: int, w: int, sigma: int) -> Fraction:
    return Fraction((2 * k + 3 * w + sigma) % 16, 16)


def o_L(k: int, w: int, j: int, sigma: int) -> Fraction:
    return Fraction((2 * k + 5 * w + 3 * j + sigma) % 16, 16)


def o_delta(k: int, w: int, j: int, sigma: int) -> Fraction:
    if o_class(k) != "positive":
        return Fraction(0)
    if sigma == 0 and (k - 48) % 3 == 0:
        return Fraction(0)
    return Fraction(1, 32) if (k + w + j + sigma) % 2 == 0 else Fraction(-1, 32)


def o_ell(k: int, w: int, j: int, sigma: int) -> Fraction:
    return o_mod_one(o_L(k, w, j, sigma) + o_delta(k, w, j, sigma))


def o_T(k: int, w: int, j: int, sigma: int) -> Fraction:
    return o_mod_one(o_alpha(k, w, sigma)
                     - o_sign(k, w) * (o_L(k, w, j, sigma)
                                       + o_delta(k, w, j, sigma)))


def o_H(k: int, w: int, sigma: int) -> tuple:
    return tuple(o_mod_one(o_alpha(k, w, sigma) + Fraction(d, 16))
                 for d in (0, 3, 7, 13))


def o_contrast(k: int, w: int) -> Fraction:
    return Fraction(1 + Fraction((k + w) % 4, 4))


def o_nuisance(k: int, w: int) -> Fraction:
    return Fraction(1, 2) + Fraction((k + 2 * w) % 4, 4)


def o_patch(k: int, w: int, j: int, sigma: int) -> np.ndarray:
    name = o_class(k)
    axis = o_axis(k, w, j)
    leading = o_L(k, w, j, sigma)
    ell = o_ell(k, w, j, sigma)
    step = (Fraction(*o_b(k, w, sigma).as_integer_ratio())
            / Fraction(*o_period(k, w, sigma).as_integer_ratio()))
    base = o_nuisance(k, w) / 8 + 2 * sigma
    contrast = o_contrast(k, w)
    nuisance = o_nuisance(k, w)
    values = np.zeros(64, dtype="<f8")
    if name != "textureless":
        for t in range(64):
            x_t = o_mod_one(leading + step * t)
            if name == "same_block":
                quantity = base
            elif name == "seam_null":
                quarter = 4 * x_t
                even = (quarter.numerator // quarter.denominator) % 2 == 0
                quantity = base + (nuisance if even else 0)
            else:
                quantity = (base + contrast
                            if o_d1(x_t, ell) <= Fraction(1, 16) else base)
            values[t] = float(quantity)
    patch = np.empty(PATCH_SHAPE, dtype="<f8")
    if axis == "u":
        patch[:, :] = values[np.newaxis, :]
    else:
        patch[:, :] = values[:, np.newaxis]
    return patch


def o_lane_bounds(axis: str) -> tuple:
    plan = lp.split_phase_lane_plan(axis, 64, 64, lp.SPLIT_PHASE_PARTITION_CONFIG)
    assert isinstance(plan, lp.SplitPhaseLanePlan)
    return plan.d_start, plan.d_stop


def o_profile(k: int, w: int, j: int, sigma: int) -> tuple:
    patch = o_patch(k, w, j, sigma)
    axis = o_axis(k, w, j)
    start, stop = o_lane_bounds(axis)
    lane_count = stop - start
    if axis == "u":
        energies = patch[start:stop, :].sum(axis=0)
    else:
        energies = patch[:, start:stop].sum(axis=1)
    weights = np.full(64, lane_count, dtype="<f8")
    return energies, weights


def o_classify_from(leading, translation, sign, b_value, period_value,
                    hypothesis, tau_band, energies, weights) -> tuple | None:
    step = (Fraction(*b_value.as_integer_ratio())
            / Fraction(*period_value.as_integer_ratio()))
    ell = o_mod_one(Fraction(sign) * (Fraction(hypothesis) - translation))
    interior = o_mod_one(ell + Fraction(1, 2))
    boundary_energy = Fraction(0)
    boundary_coverage = Fraction(0)
    interior_energy = Fraction(0)
    interior_coverage = Fraction(0)
    for index in range(len(energies)):
        x_i = o_mod_one(leading + step * index)
        energy = Fraction(*float(energies[index]).as_integer_ratio())
        coverage = Fraction(*float(weights[index]).as_integer_ratio())
        if o_d1(x_i, ell) <= tau_band:
            boundary_energy += energy
            boundary_coverage += coverage
        if o_d1(x_i, interior) <= tau_band:
            interior_energy += energy
            interior_coverage += coverage
    if boundary_coverage == 0 or interior_coverage == 0:
        return None
    return (boundary_energy / boundary_coverage,
            interior_energy / interior_coverage)


def o_classify(k: int, w: int, j: int, sigma: int, hypothesis,
               tau_band) -> tuple | None:
    energies, weights = o_profile(k, w, j, sigma)
    return o_classify_from(o_L(k, w, j, sigma), o_T(k, w, j, sigma),
                           o_sign(k, w), Fraction(*o_b(k, w, sigma).as_integer_ratio()),
                           Fraction(*o_period(k, w, sigma).as_integer_ratio()),
                           hypothesis, tau_band, energies, weights)


def o_accepts(k: int, w: int, j: int, sigma: int, hypothesis, tau_band,
              tau_margin, tau_energy) -> bool:
    outcome = o_classify(k, w, j, sigma, hypothesis, tau_band)
    if outcome is None:
        return False
    return bool(outcome[0] - outcome[1] >= tau_margin
                and outcome[0] >= tau_energy)


def o_declared_class(k: int) -> str:
    return "positive" if k >= 48 else o_class(k)


# --------------------------------------------------------------------------
# Shared reduced fixtures
# --------------------------------------------------------------------------

REDUCED_ORDINALS = (0, 49)


@pytest.fixture(scope="module")
def reduced_splits():
    calibration = sp.synthesize_split(0, ordinals=REDUCED_ORDINALS)
    evaluation = sp.synthesize_split(1, ordinals=REDUCED_ORDINALS)
    return calibration, evaluation


@pytest.fixture(scope="module")
def reduced_freeze(reduced_splits):
    calibration, _ = reduced_splits
    outcome = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)
    assert outcome["status"] == "complete"
    frozen = sp.freeze_prediction_payload(
        outcome["payload"], frozen_at="2026-09-14T00:00:00Z")
    return calibration, frozen


# --------------------------------------------------------------------------
# 1. Frozen grid literal
# --------------------------------------------------------------------------

def test_frozen_grid_is_exactly_1000_strictly_increasing_tuples():
    tuples = sp.frozen_grid_tuples()
    assert len(tuples) == 1000
    assert tuples == sp.FROZEN_GRID.tuples
    assert sp.FROZEN_GRID.grid_sha256 == sp.FROZEN_GRID_SHA256
    assert all(sp.calibration_tuple_lt(first, second)
               for first, second in zip(tuples, tuples[1:]))
    assert tuples[0] == ((0, 1), (1, 8), (0, 1), 1, 2)
    assert tuples[199] == ((0, 1), (1, 1), (1, 2), 8, 3)
    assert tuples[200] == ((1, 8), (1, 8), (0, 1), 1, 2)
    assert tuples[-1] == ((1, 2), (1, 1), (1, 2), 8, 3)


def test_frozen_grid_component_sets_and_target_power():
    bands = sorted({Fraction(*value[0]) for value in sp.FROZEN_GRID.tuples})
    margins = sorted({Fraction(*value[1]) for value in sp.FROZEN_GRID.tuples})
    energies = sorted({Fraction(*value[2]) for value in sp.FROZEN_GRID.tuples})
    samples = sorted({value[3] for value in sp.FROZEN_GRID.tuples})
    frames = sorted({value[4] for value in sp.FROZEN_GRID.tuples})
    assert bands == sorted(sp.SPLIT_PHASE_TAU_BAND_SET)
    assert margins == sorted(sp.SPLIT_PHASE_TAU_MARGIN_SET)
    assert energies == sorted(sp.SPLIT_PHASE_TAU_ENERGY_SET)
    assert samples == sorted(sp.SPLIT_PHASE_MIN_HELD_OUT_SET)
    assert frames == sorted(sp.SPLIT_PHASE_MIN_DISTINCT_FRAMES_SET)
    assert tuple(sp.FROZEN_GRID.target_power) == (1, 2)
    assert sp.FROZEN_GRID.null_control_ids == tuple(
        sp.control_id(0, k) for k in range(48))
    assert sp.FROZEN_GRID.seam_control_ids == tuple(
        sp.control_id(0, k) for k in range(48, 96))


def test_frozen_grid_structure_parts_are_all_true():
    grid = sp.FROZEN_GRID
    assert grid.grid_version == sp.SPLIT_PHASE_GRID_VERSION
    assert grid.evaluation_only is True
    assert tuple(grid.target_power) == (1, 2)
    assert grid.tuples and len(grid.tuples) <= sp.MAX_GRID_TUPLES
    validated = tuple(sp.validate_grid_tuple(value) for value in grid.tuples)
    assert validated == tuple(grid.tuples)
    assert all(sp.calibration_tuple_lt(first, second)
               for first, second in zip(grid.tuples, grid.tuples[1:]))
    assert tuple(grid.null_control_ids) == tuple(
        sp.control_id(0, k) for k in range(48))
    assert tuple(grid.seam_control_ids) == tuple(
        sp.control_id(0, k) for k in range(48, 96))
    assert grid.grid_sha256 == sp.digest_record(grid.to_record(), "grid_sha256")


def test_control_id_hashes_the_exact_class_keyed_preimage():
    split_names = ("calibration", "evaluation")
    for sigma in (0, 1):
        for k in (0, 15, 17, 47, 48, 95):
            preimage = ('{{"class":"{}","ordinal":{},"sigma":{},"split":"{}"}}'
                        .format(o_class(k), k, sigma, split_names[sigma]))
            assert sp.canonical_json({
                "sigma": sigma, "split": split_names[sigma],
                "class": o_class(k), "ordinal": k}).decode("utf-8") == preimage
            assert sp.control_id(sigma, k) \
                == hashlib.sha256(preimage.encode("utf-8")).hexdigest()
            assert "control_class" not in preimage


def test_frozen_grid_validation_is_exact_and_reduced_grid_is_not_frozen():
    reduced = sp.build_grid(tuples=(((1, 8), (1, 8), (0, 1), 1, 2),))
    assert sp.validate_grid_structure(reduced) is True
    assert sp.validate_frozen_grid(reduced) is False
    assert sp.validate_frozen_grid(sp.FROZEN_GRID) is True
    assert len(sp._lp.SplitPhaseCalibrationGrid(  # noqa: SLF001
        grid_version="v2b-p0-calibration-grid-1", evaluation_only=True,
        target_power=(1, 2), tuples=sp.FROZEN_GRID.tuples,
        null_control_ids=sp.FROZEN_GRID.null_control_ids,
        seam_control_ids=sp.FROZEN_GRID.seam_control_ids,
        grid_sha256=sp.FROZEN_GRID.grid_sha256).tuples) == 1000


def test_grid_and_dyadic_refusals():
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((0, 2), (1, 8), (0, 1), 1, 2),))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 2), (2, 4), (0, 1), 1, 2),))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 8), (1, 8), (0, 1), 1, 2),
                              ((1, 8), (1, 8), (0, 1), 1, 2)))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 2), (1, 8), (0, 1), 1, 2),
                              ((1, 8), (1, 8), (0, 1), 1, 2)))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 2), (0, 1), (0, 1), 1, 2),))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 8), (1, 8), (0, 1), 0, 2),))
    with pytest.raises(ValueError):
        sp.build_grid(tuples=(((1, 8), (1, 8), (0, 1), 1, 1),))
    assert sp.is_reduced_dyadic([0, 2]) is False
    assert sp.is_reduced_dyadic([2, 4]) is False
    assert sp.is_reduced_dyadic([1, 3]) is False
    assert sp.is_reduced_dyadic([3, 4]) is True


# --------------------------------------------------------------------------
# 2. Generator formulas and independent patch/profile agreement
# --------------------------------------------------------------------------

def test_generator_row_materialization_matches_declared_formulas():
    for k in (0, 15, 16, 31, 32, 47, 48, 49, 50, 95):
        for w in (0, 1, 2):
            for j in (0, 1, 2):
                for sigma in (0, 1):
                    assert sp.class_of_ordinal(k) == o_declared_class(k)
                    assert sp.local_axis(k, w, j) == o_axis(k, w, j)
                    assert sp.world_sign(k, w) == o_sign(k, w)
                    assert sp.selected_b(k, w, sigma) == o_b(k, w, sigma)
                    assert sp.period_pixels(k, w, sigma) == o_period(k, w, sigma)
                    assert sp.alpha_true(k, w, sigma) == o_alpha(k, w, sigma)
                    assert sp.leading_cell(k, w, j, sigma) == o_L(k, w, j, sigma)
                    assert sp.ell_true_of(k, w, j, sigma, o_declared_class(k)) \
                        == o_ell(k, w, j, sigma)
                    assert sp.frame_translation(k, w, j, sigma, o_declared_class(k)) \
                        == o_T(k, w, j, sigma)
                    assert sp.hypothesis_set(k, w, sigma) == o_H(k, w, sigma)
                    assert (Fraction(*sp.boundary_contrast(k, w).as_integer_ratio())
                            == o_contrast(k, w))
                    assert (Fraction(*sp.nuisance_amplitude(k, w).as_integer_ratio())
                            == o_nuisance(k, w))


def test_oracle_patch_bytes_are_identical_for_every_class_and_split():
    for k, w, j, sigma in (
            (0, 0, 0, 0), (3, 2, 1, 1), (16, 1, 2, 0), (17, 0, 0, 1),
            (32, 2, 0, 0), (40, 1, 1, 1), (48, 0, 1, 0), (49, 1, 2, 0),
            (50, 2, 0, 1), (95, 0, 2, 1)):
        assert sp.patch_for(k, w, j, sigma, o_declared_class(k)).tobytes() \
            == o_patch(k, w, j, sigma).tobytes()


def test_producer_profiles_match_the_open_form_oracle():
    for k, w, j, sigma in ((0, 0, 0, 0), (16, 2, 1, 0), (48, 1, 0, 0),
                           (49, 2, 2, 0), (95, 0, 1, 1)):
        observation = sp._make_observation(k, w, j, sigma)  # noqa: SLF001
        assert sp.observation_matches_frozen(observation) is True
        result = sp._produce_observation(observation)  # noqa: SLF001
        for evidence in result.seam_profile_evidence:
            start, stop = o_lane_bounds(evidence.local_axis)
            patch = o_patch(k, w, j, sigma)
            if evidence.local_axis == "u":
                expected = patch[start:stop, :].sum(axis=0)
            else:
                expected = patch[:, start:stop].sum(axis=1)
            samples, weights = sp.profile_arrays(evidence)
            assert np.array_equal(samples, expected)
            assert np.array_equal(weights, np.full(64, stop - start, dtype="<f8"))


def test_module_classifier_matches_the_oracle_exactly():
    observation = sp._make_observation(49, 1, 2, 0)  # noqa: SLF001
    result = sp._produce_observation(observation)  # noqa: SLF001
    evidence = {item.local_axis: item for item in result.seam_profile_evidence}
    owning = evidence[observation.local_axis]
    samples, weights = sp.profile_arrays(owning)
    for hypothesis in observation.H_w:
        for tau_band in TAU_BANDS:
            expected = o_classify(49, 1, 2, 0, sp.dyadic_fraction(hypothesis),
                                  tau_band)
            outcome = sp.classify_observation(
                observation, sp.dyadic_fraction(hypothesis), tau_band,
                samples, weights)
            if expected is None:
                assert outcome["status"] == "zero_coverage"
            else:
                assert outcome["status"] == "complete"
                assert outcome["boundary_energy"] == expected[0]
                assert outcome["interior_energy"] == expected[1]


def test_coverage_normalization_is_by_sum_of_weights_not_bin_count():
    observation = sp._make_observation(49, 0, 0, 0)  # noqa: SLF001
    samples = np.arange(1, 65, dtype="<f8")
    weights = np.full(64, 3.0, dtype="<f8")
    weights[0] = 100.0
    for hypothesis in observation.H_w:
        for tau_band in (Fraction(1, 8), Fraction(1, 4)):
            expected = o_classify_from(
                sp.dyadic_fraction(observation.L_dyadic),
                sp.dyadic_fraction(observation.T_dyadic),
                observation.world_sign,
                Fraction(*observation.selected_b.as_integer_ratio()),
                Fraction(*observation.period_pixels.as_integer_ratio()),
                sp.dyadic_fraction(hypothesis), tau_band, samples, weights)
            outcome = sp.classify_observation(
                observation, sp.dyadic_fraction(hypothesis), tau_band,
                samples, weights)
            if expected is None:
                assert outcome["status"] == "zero_coverage"
            else:
                assert outcome["boundary_energy"] == expected[0]
                assert outcome["interior_energy"] == expected[1]


def test_positive_truth_is_declared_and_recovers_exactly():
    for k in (48, 49, 50, 51, 95):
        for w in (0, 1, 2):
            for j in (0, 1, 2):
                sigma = 0 if (k - 48) % 3 == 0 else 1
                alpha = o_alpha(k, w, sigma)
                low = [hypothesis for hypothesis in o_H(k, w, sigma)
                       if o_accepts(k, w, j, sigma, hypothesis,
                                    Fraction(1, 8), Fraction(1, 8),
                                    Fraction(0))]
                strict = [hypothesis for hypothesis in o_H(k, w, sigma)
                          if o_accepts(k, w, j, sigma, hypothesis,
                                       Fraction(1, 8), Fraction(3, 8),
                                       Fraction(0))]
                assert alpha in low
                assert strict == [alpha]
                observation = sp._make_observation(k, w, j, sigma)  # noqa: SLF001
                assert sp.observation_matches_frozen(observation) is True
                assert sp.dyadic_fraction(observation.alpha_dyadic_hidden) == alpha
                evidence = {item.local_axis: item for item in
                            sp._produce_observation(  # noqa: SLF001
                                observation).seam_profile_evidence}
                samples, weights = sp.profile_arrays(
                    evidence[observation.local_axis])
                for hypothesis in observation.H_w:
                    value = sp.dyadic_fraction(hypothesis)
                    assert sp.hypothesis_accepts(
                        observation, value, Fraction(1, 8), Fraction(1, 8),
                        Fraction(0), samples, weights) \
                        == o_accepts(k, w, j, sigma, value, Fraction(1, 8),
                                     Fraction(1, 8), Fraction(0))


def test_seam_null_has_the_exact_half_turn_identity():
    for k in (32, 33, 47):
        for w in (0, 1, 2):
            for j in (0, 1, 2):
                for sigma in (0, 1):
                    for hypothesis in o_H(k, w, sigma):
                        for tau_band in TAU_BANDS:
                            outcome = o_classify(k, w, j, sigma, hypothesis,
                                                 tau_band)
                            assert outcome is not None
                            assert outcome[0] == outcome[1]


def test_textureless_and_same_block_never_accept():
    for k in (0, 5, 16, 31):
        for w in (0, 1, 2):
            for j in (0, 1, 2):
                sigma = k % 2
                for hypothesis in o_H(k, w, sigma):
                    for tau_band in TAU_BANDS:
                        outcome = o_classify(k, w, j, sigma, hypothesis,
                                             tau_band)
                        if outcome is not None:
                            assert outcome[0] == outcome[1]


def test_module_null_controls_do_not_accept_any_hypothesis():
    for k in (0, 16, 32):
        observation = sp._make_observation(k, 1, 0, 0)  # noqa: SLF001
        result = sp._produce_observation(observation)  # noqa: SLF001
        evidence = {item.local_axis: item for item in result.seam_profile_evidence}
        samples, weights = sp.profile_arrays(
            evidence[observation.local_axis])
        for hypothesis in observation.H_w:
            assert sp.hypothesis_accepts(
                observation, sp.dyadic_fraction(hypothesis), Fraction(1, 8),
                Fraction(1, 8), Fraction(0), samples, weights) is False


# --------------------------------------------------------------------------
# 3. Frozen tau_band=0 infeasibility
# --------------------------------------------------------------------------

def test_tau_band_zero_is_empty_for_every_nonzero_delta_observation():
    assert sum(
        1 for k in range(48, 96)
        if all(o_delta(k, w, j, 0) == 0
               for w in (0, 1, 2) for j in (0, 1, 2))) == 16
    for k in (49, 50, 52, 95):
        for w in (0, 1, 2):
            for j in (0, 1, 2):
                assert o_delta(k, w, j, 0) != 0
                ell = o_ell(k, w, j, 0)
                nearest = min(o_d1(o_mod_one(o_L(k, w, j, 0) + Fraction(i, 16)),
                                   ell) for i in range(64))
                assert nearest == Fraction(1, 32)
                observation = sp._make_observation(k, w, j, 0)  # noqa: SLF001
                result = sp._produce_observation(observation)  # noqa: SLF001
                evidence = {item.local_axis: item
                            for item in result.seam_profile_evidence}
                samples, weights = sp.profile_arrays(
                    evidence[observation.local_axis])
                for hypothesis in observation.H_w:
                    outcome = sp.classify_observation(
                        observation, sp.dyadic_fraction(hypothesis),
                        Fraction(0), samples, weights)
                    assert outcome["status"] == "zero_coverage"


def test_tau_band_zero_tuples_are_never_feasible(reduced_freeze):
    calibration, frozen = reduced_freeze
    for index, value in enumerate(sp.FROZEN_GRID.tuples):
        if Fraction(*value[0]) == 0:
            feasible, _ = sp.tuple_feasible(
                frozen, sp.FROZEN_GRID, index, {})
            assert feasible is False
    selection = sp.select_first_feasible(frozen, sp.FROZEN_GRID, truth={})
    assert selection["status"] == "refused"
    assert selection["reason"] == "no_feasible_calibration"


# --------------------------------------------------------------------------
# 4. Reduced end-to-end selection, score, and audit
# --------------------------------------------------------------------------

def test_reduced_end_to_end_selection_and_score(reduced_splits):
    calibration, evaluation = reduced_splits
    assert sp.validate_grid_structure(sp.FROZEN_GRID) is True
    assert sp.FROZEN_GRID.tuples[200] == ((1, 8), (1, 8), (0, 1), 1, 2)
    outcome = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)
    assert outcome["status"] == "complete"
    frozen = sp.freeze_prediction_payload(
        outcome["payload"], frozen_at="2026-09-14T00:00:00Z")
    assert sp.prediction_refusal(frozen, calibration["corpus"],
                                 sp.FROZEN_GRID) is None
    direct = sp.select_first_feasible(
        frozen, sp.FROZEN_GRID, truth=calibration["labels"].truth())
    assert direct["status"] == "selected", direct
    assert direct["tuple_index"] == 200
    record = sp.build_calibration_record(
        sp.FROZEN_GRID, sp.FROZEN_GRID.tuples[200],
        calibration_corpus=calibration["corpus"],
        evaluation_corpus=evaluation["corpus"])
    assert sp.validate_calibration_record_refusal(
        calibration["corpus"], evaluation["corpus"], sp.FROZEN_GRID,
        record) is None
    scored = sp.score_freeze(
        frozen, calibration["labels"], sp.FROZEN_GRID,
        calibration_corpus=calibration["corpus"],
        evaluation_corpus=evaluation["corpus"])
    assert scored["status"] == "selected", scored
    assert scored["tuple_index"] == 200
    assert scored["tuple"] == ((1, 8), (1, 8), (0, 1), 1, 2)
    assert scored["audit"] == {"false_accepts": 0, "successes": 1,
                               "positives": 1, "target_successes": 1}
    record = scored["calibration"]
    assert sp.validate_calibration_record_refusal(
        calibration["corpus"], evaluation["corpus"], sp.FROZEN_GRID,
        record) is None
    assert tuple(record.calibration_sample_sha256s) \
        == calibration["corpus"].sample_sha256s
    assert tuple(record.evaluation_sample_sha256s) \
        == evaluation["corpus"].sample_sha256s
    audit = sp.score_audit_payload(
        sigma=0, split="calibration",
        corpus_sha256=calibration["corpus"].corpus_sha256,
        labels_sha256=calibration["labels"].labels_sha256,
        predictions_sha256=frozen["predictions_sha256"],
        frozen_at=frozen["frozen_at"], scored_at="2026-09-14T00:00:01Z",
        outcome=scored, grid_sha256=sp.FROZEN_GRID.grid_sha256,
        caps=sp.default_caps())
    assert audit["scored_after_frozen"] is True
    assert audit["selected_tuple"] == [[1, 8], [1, 8], [0, 1], 1, 2]
    assert audit["outcome"] == "selected"


def test_full_frozen_schedule_selection_and_power():
    calibration = sp.synthesize_split(0)
    evaluation = sp.synthesize_split(1)
    assert len(calibration["corpus"].controls) == 96
    assert len(calibration["corpus"].observations) == 864
    assert len(evaluation["corpus"].controls) == 96
    assert len(evaluation["corpus"].observations) == 864
    assert len(calibration["labels"].entries) == 96
    assert len(calibration["corpus"].null_control_ids) == 48
    assert len(calibration["corpus"].seam_control_ids) == 48
    # Every evaluation positive is scored on a nonzero sub-bin offset (§7).
    assert all(sp.delta_true(k, w, j, 1, "positive") != 0
               for k in range(48, 96) for w in (0, 1, 2) for j in (0, 1, 2))
    # Exactly the 16 calibration positives with (k - 48) % 3 == 0 have delta 0.
    assert sum(
        1 for k in range(48, 96)
        if all(sp.delta_true(k, w, j, 0, "positive") == 0
               for w in (0, 1, 2) for j in (0, 1, 2))) == 16
    assert sp.disjointness_refusal(calibration["corpus"],
                                   evaluation["corpus"]) is None
    outcome = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)
    assert outcome["status"] == "complete"
    frozen = sp.freeze_prediction_payload(
        outcome["payload"], frozen_at="2026-09-14T00:00:00Z")
    assert sp.prediction_refusal(frozen, calibration["corpus"],
                                 sp.FROZEN_GRID) is None
    selection = sp.select_first_feasible(
        frozen, sp.FROZEN_GRID, truth=calibration["labels"].truth())
    assert selection["status"] == "selected"
    assert selection["tuple_index"] == 200
    # tau_band=0 is infeasible (16 < 24); the first feasible tuple gives
    # 32/48 = 2/3 >= target_power with zero null false accepts.
    assert selection["audit"] == {"false_accepts": 0, "successes": 32,
                                  "positives": 48, "target_successes": 24}
    scored = sp.score_freeze(
        frozen, calibration["labels"], sp.FROZEN_GRID,
        calibration_corpus=calibration["corpus"],
        evaluation_corpus=evaluation["corpus"])
    assert scored["status"] == "selected"
    assert scored["tuple"] == ((1, 8), (1, 8), (0, 1), 1, 2)
    assert sp.validate_calibration_record_refusal(
        calibration["corpus"], evaluation["corpus"], sp.FROZEN_GRID,
        scored["calibration"]) is None


def test_d4_any_determined_null_and_wrong_alpha_fail_closed(reduced_freeze):
    calibration, frozen = reduced_freeze
    payload = {key: value for key, value in frozen.items()
               if key not in ("predictions_sha256", "frozen_at")}
    controls = payload["per_control"]
    positive_index = next(index for index, item in enumerate(controls)
                          if item["control_class"] == "positive")
    null_index = next(index for index, item in enumerate(controls)
                      if item["control_class"] != "positive")
    # A single determined null axis is a false accept and an infeasible tuple.
    forged = json.loads(json.dumps(payload))
    forged["per_control"][null_index]["tuple_results"][200][0]["status"] = \
        "determined"
    forged = sp.freeze_prediction_payload(
        forged, frozen_at="2026-09-14T00:00:00Z")
    feasible, audit = sp.tuple_feasible(
        forged, sp.FROZEN_GRID, 200, calibration["labels"].truth())
    assert feasible is False
    assert audit["false_accepts"] == 1
    # A determined positive whose emitted alpha is not the declared vector is
    # not a success, no matter how many axes report determined.
    forged = json.loads(json.dumps(payload))
    for axis in forged["per_control"][positive_index]["tuple_results"][200]:
        axis["status"] = "determined"
        axis["emitted_alpha"] = [0, 1]
    forged = sp.freeze_prediction_payload(
        forged, frozen_at="2026-09-14T00:00:00Z")
    feasible, audit = sp.tuple_feasible(
        forged, sp.FROZEN_GRID, 200, calibration["labels"].truth())
    assert audit["successes"] == 0
    assert feasible is False


def test_audit_refuses_a_scored_at_not_after_frozen(reduced_freeze):
    calibration, frozen = reduced_freeze
    outcome = {"status": "refused", "reason": "no_feasible_calibration"}
    audit = sp.score_audit_payload(
        sigma=0, split="calibration",
        corpus_sha256=calibration["corpus"].corpus_sha256,
        labels_sha256=calibration["labels"].labels_sha256,
        predictions_sha256=frozen["predictions_sha256"],
        frozen_at="2026-09-14T00:00:05Z", scored_at="2026-09-14T00:00:05Z",
        outcome=outcome, grid_sha256=sp.FROZEN_GRID.grid_sha256,
        caps=sp.default_caps())
    assert audit["scored_after_frozen"] is False


def test_score_refuses_before_prediction_bytes_exist(reduced_splits):
    calibration, evaluation = reduced_splits
    unfrozen = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)["payload"]
    outcome = sp.score_freeze(
        unfrozen, calibration["labels"], sp.FROZEN_GRID,
        calibration_corpus=calibration["corpus"],
        evaluation_corpus=evaluation["corpus"])
    assert outcome["status"] == "refused"
    assert outcome["reason"] == "split_not_reconstructible"


def test_score_refuses_a_prediction_with_the_wrong_label_digest(reduced_splits):
    calibration, evaluation = reduced_splits
    payload = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)["payload"]
    payload["labels_sha256"] = "0" * 64
    frozen = sp.freeze_prediction_payload(
        payload, frozen_at="2026-09-14T00:00:00Z")
    outcome = sp.score_freeze(
        frozen, calibration["labels"], sp.FROZEN_GRID,
        calibration_corpus=calibration["corpus"],
        evaluation_corpus=evaluation["corpus"])
    assert outcome["status"] == "refused"


# --------------------------------------------------------------------------
# 5. Prediction digest, integrity, and non-owning profile validation
# --------------------------------------------------------------------------

def test_prediction_digest_covers_the_body_before_frozen_at(reduced_freeze):
    _, frozen = reduced_freeze
    assert sp.verify_prediction_digest(frozen) is True
    retimed = dict(frozen)
    retimed["frozen_at"] = "2030-01-01T00:00:00Z"
    assert sp.verify_prediction_digest(retimed) is True
    tampered = dict(frozen)
    tampered["sigma"] = 1
    assert sp.verify_prediction_digest(tampered) is False
    assert sp.prediction_refusal(tampered, reduced_freeze[0]["corpus"],
                                 sp.FROZEN_GRID) is not None


def test_prediction_refusal_detects_a_dropped_or_mutated_non_owning_profile(
        reduced_freeze):
    calibration, frozen = reduced_freeze
    corpus = calibration["corpus"]
    assert sp.prediction_refusal(frozen, corpus, sp.FROZEN_GRID) is None
    altered = json.loads(json.dumps(frozen))
    entry = altered["per_observation"][0]
    raw = bytearray(base64.b64decode(
        entry["non_owning_profile"]["profile_samples_float64_le_base64"]))
    raw[0] ^= 0x01
    entry["non_owning_profile"]["profile_samples_float64_le_base64"] = \
        base64.b64encode(bytes(raw)).decode("ascii")
    altered = sp.freeze_prediction_payload(
        {key: value for key, value in altered.items()
         if key not in ("predictions_sha256", "frozen_at")},
        frozen_at="2026-09-14T00:00:00Z")
    assert sp.prediction_refusal(altered, corpus, sp.FROZEN_GRID) is not None
    dropped = json.loads(json.dumps(frozen))
    del dropped["per_observation"][0]["non_owning_profile"]
    dropped = sp.freeze_prediction_payload(
        {key: value for key, value in dropped.items()
         if key not in ("predictions_sha256", "frozen_at")},
        frozen_at="2026-09-14T00:00:00Z")
    assert sp.prediction_refusal(dropped, corpus, sp.FROZEN_GRID) is not None


def test_prediction_refusal_refuses_a_resealed_consumed_profile_digest(
        reduced_freeze):
    calibration, frozen = reduced_freeze
    corpus = calibration["corpus"]
    assert sp.prediction_refusal(frozen, corpus, sp.FROZEN_GRID) is None
    altered = json.loads(json.dumps(frozen))
    entry = altered["per_observation"][0]
    entry["consumed_profile_sha256"] = "f" * 64
    assert entry["consumed_profile_sha256"] \
        != entry["non_owning_profile_sha256"]
    resealed = sp.freeze_prediction_payload(
        {key: value for key, value in altered.items()
         if key not in ("predictions_sha256", "frozen_at")},
        frozen_at="2026-09-14T00:00:00Z")
    # The body digest is recomputed, so only the bound-axis replay can refuse.
    assert sp.verify_prediction_digest(resealed) is True
    assert sp.prediction_refusal(resealed, corpus, sp.FROZEN_GRID) is not None


def test_prediction_records_the_consumed_and_retained_profile_digests(
        reduced_freeze):
    calibration, frozen = reduced_freeze
    observations = calibration["corpus"].observations
    assert [item["observation_id"] for item in frozen["per_observation"]] \
        == [item.observation_id for item in observations]
    for entry, observation in zip(frozen["per_observation"], observations):
        assert entry["local_axis"] == observation.local_axis
        assert entry["non_owning_profile_sha256"] \
            == observation.non_owning_profile_sha256
        assert entry["consumed_profile_sha256"] \
            != entry["non_owning_profile_sha256"]
        expected_other = "v" if observation.local_axis == "u" else "u"
        assert entry["non_owning_profile"]["local_axis"] == expected_other


def test_payload_decode_and_rehash_refuses_tampering():
    observation = sp._make_observation(48, 0, 0, 0)  # noqa: SLF001
    result = sp._produce_observation(observation)  # noqa: SLF001
    evidence = result.seam_profile_evidence[0]
    record = evidence.to_record()
    record["coverage"] = {**record["coverage"],
                          "per_bin": tuple(record["coverage"]["per_bin"])}
    rebuilt = lp.SplitSeamProfileEvidence(**record)
    assert rebuilt.payload_sha256 == evidence.payload_sha256
    record["profile_samples_float64_le_base64"] = base64.b64encode(
        b"\x00" * 8 * evidence.sample_count).decode("ascii")
    with pytest.raises(ValueError):
        lp.SplitSeamProfileEvidence(**record)
    assert sp.profile_sha256(
        *[np.frombuffer(base64.b64decode(getattr(evidence, name)), dtype=dtype)
          for name, dtype in (
              ("profile_samples_float64_le_base64", "<f8"),
              ("profile_weights_float64_le_base64", "<f8"),
              ("axis_bin_indices_int64_le_base64", "<i8"))]) \
        == evidence.payload_sha256


# --------------------------------------------------------------------------
# 6. Disjointness and the split-name-only reseal mutant
# --------------------------------------------------------------------------

def test_reduced_splits_are_digest_and_set_disjoint(reduced_splits):
    calibration, evaluation = reduced_splits
    assert sp.disjointness_refusal(calibration["corpus"],
                                   evaluation["corpus"]) is None
    first = sp.corpus_summary(calibration["corpus"])
    second = sp.corpus_summary(evaluation["corpus"])
    assert not set(first["observation_ids"]) & set(second["observation_ids"])
    assert not set(first["sample_sha256s"]) & set(second["sample_sha256s"])
    assert not set(first["provenance_sha256s"]) & set(second["provenance_sha256s"])
    assert not (set(first["nontextureless_patch_sha256s"])
                & set(second["nontextureless_patch_sha256s"]))
    assert first["corpus_sha256"] != second["corpus_sha256"]
    # Textureless patches are all-zero and may collide across splits, but every
    # corresponding (k, w) key keeps a distinct geometry and provenance/sample.
    keys = {(item["ordinal"], item["world_axis"]) for item in first["textureless"]}
    assert keys
    second_by_key = {(item["ordinal"], item["world_axis"]): item
                     for item in second["textureless"]}
    for item in first["textureless"]:
        other = second_by_key[(item["ordinal"], item["world_axis"])]
        assert item["patch_sha256"] == other["patch_sha256"]
        assert item["geometry_sha256"] != other["geometry_sha256"]
        assert item["provenance_sha256"] != other["provenance_sha256"]
        assert item["sample_sha256"] != other["sample_sha256"]


def test_split_name_only_reseal_is_refused(reduced_splits):
    calibration, _ = reduced_splits
    record = calibration["corpus"].to_record()
    record["split"] = "evaluation"
    with pytest.raises(ValueError):
        sp.corpus_from_record(record)
    resealed = json.loads(json.dumps(calibration["corpus"].to_record()))
    resealed["sigma"] = 1
    with pytest.raises(ValueError):
        sp.corpus_from_record(resealed)


def test_corpus_digest_tamper_is_refused(reduced_splits):
    calibration, _ = reduced_splits
    record = calibration["corpus"].to_record()
    record["sample_sha256s"][0] = "f" * 64
    with pytest.raises(ValueError):
        sp.corpus_from_record(record)


def test_split_calibration_constructor_refuses_sample_hash_overlap(reduced_splits):
    calibration, evaluation = reduced_splits
    overlapping = tuple(evaluation["corpus"].sample_sha256s)
    with pytest.raises(ValueError):
        sp._lp.SplitPhaseCalibration(  # noqa: SLF001
            calibration_version="v2b-p0-calibration-1",
            partition_config_sha256=sp._lp.SPLIT_PHASE_PARTITION_CONFIG_SHA256,
            tau_band=(1, 8), tau_margin=(1, 8), tau_energy=(0, 1),
            min_held_out_samples=1, min_distinct_frames_per_axis=2,
            tie_break="no_tie_break_abstain", target_power=(1, 2),
            calibration_corpus_sha256="a" * 64,
            evaluation_corpus_sha256="b" * 64,
            calibration_sample_sha256s=overlapping,
            evaluation_sample_sha256s=overlapping,
            null_control_ids=("c" * 64,), seam_control_ids=("d" * 64,),
            calibration_grid_sha256="e" * 64, calibration_sha256="f" * 64)


# --------------------------------------------------------------------------
# 7. Budgets and refusals
# --------------------------------------------------------------------------

def _refusal(corpus, grid, **lowers):
    caps = sp.default_caps().lower(**lowers)
    return sp.predict_corpus(corpus, grid, caps=caps)


def test_every_frozen_ceiling_is_enforced(reduced_splits):
    calibration, _ = reduced_splits
    corpus = calibration["corpus"]
    grid = sp.FROZEN_GRID
    cases = (
        ("max_controls_per_split", 1),
        ("max_observations_per_split", 17),
        ("max_producer_calls_per_split", 17),
        ("max_p0_replay_checks_per_split", 17),
        ("max_patch_bytes_per_split", 17),
        ("max_produced_profile_bins_per_result", 127),
        ("max_profile_bins_per_observation", 63),
        ("max_grid_tuples", 999),
        ("max_hypotheses_per_axis", 3),
        ("max_hypotheses_total", 11),
        ("max_axes_per_control", 2),
        ("max_classifier_terms_per_split", 1),
        ("max_payload_bytes_per_split", 1),
    )
    for name, value in cases:
        outcome = _refusal(corpus, grid, **{name: value})
        assert outcome["status"] == "refused", name
        assert outcome["reason"] == "work_over_cap", name
        assert outcome["cap"] == name
    frames = _refusal(corpus, grid, max_frames_per_axis=2)
    assert frames["status"] == "refused"
    assert frames["reason"] == "work_over_cap"


def test_exact_limit_succeeds(reduced_splits):
    calibration, _ = reduced_splits
    corpus = calibration["corpus"]
    observations = len(corpus.observations)
    caps = sp.default_caps().lower(
        max_controls_per_split=len(corpus.controls),
        max_observations_per_split=observations,
        max_producer_calls_per_split=observations,
        max_p0_replay_checks_per_split=observations * 8,
        max_patch_bytes_per_split=observations * 64 * 64 * 8,
        max_grid_tuples=len(sp.FROZEN_GRID.tuples),
        max_hypotheses_per_axis=4,
        max_hypotheses_total=12,
        max_axes_per_control=3,
        max_frames_per_axis=3,
        max_produced_profile_bins_per_result=128,
        max_profile_bins_per_observation=64,
        max_classifier_terms_per_split=(
            len(sp.FROZEN_GRID.tuples) * observations * 4 * 64 * 2),
    )
    outcome = sp.predict_corpus(corpus, sp.FROZEN_GRID, caps=caps)
    assert outcome["status"] == "complete"


def test_synthesis_caps_refuse_an_over_large_schedule():
    assert sp.synthesize_split(0, ordinals=(0,))["corpus"] \
        .corpus_version == sp.SPLIT_PHASE_CALIBRATION_CORPUS_VERSION
    with pytest.raises(ValueError):
        sp.synthesize_split(0, ordinals=(0, 0))
    with pytest.raises(ValueError):
        sp.synthesize_split(0, ordinals=(96,))
    with pytest.raises(ValueError):
        sp.synthesize_split(0, ordinals=(0,), caps=sp.default_caps().lower(
            max_controls_per_split=0))


# --------------------------------------------------------------------------
# 8. Observation schema and dyadic refusals
# --------------------------------------------------------------------------

def test_observation_record_schema_is_exact():
    observation = sp._make_observation(48, 1, 0, 0)  # noqa: SLF001
    keys = set(observation.to_record())
    assert keys == {
        "observation_id", "sigma", "split", "control_id", "control_class",
        "ordinal", "patch_sha256", "L_dyadic", "T_dyadic",
        "alpha_dyadic_hidden", "H_w", "selected_b", "selected_b_ratio",
        "period_pixels", "local_axis", "world_axis", "world_sign",
        "boundary_contrast", "nuisance_amplitude", "geometry", "geometry_sha256",
        "patch_float64_le_base64", "non_owning_profile_sha256",
        "provenance_sha256", "label_payload_sha256", "observation_sha256"}
    assert set(observation.redacted_record()) == keys - {
        "alpha_dyadic_hidden", "label_payload_sha256"}
    assert "bound_profile_float64_le_base64" not in keys
    assert observation.patch().shape == PATCH_SHAPE
    assert sp.patch_sha256(observation.patch()) == observation.patch_sha256
    assert observation.selected_b in B_SET
    assert Fraction(*observation.selected_b_ratio) \
        == Fraction(*observation.selected_b.as_integer_ratio())
    assert observation.geometry["period_pixels"] == observation.period_pixels
    assert observation.geometry["rectified_pixel_step"] == 1.0


def test_observation_refuses_invalid_selected_b_and_ratio():
    observation = sp._make_observation(48, 0, 0, 0)  # noqa: SLF001
    payload = {key: getattr(observation, key)
               for key in observation.to_record()}
    payload["selected_b"] = 0.25
    with pytest.raises(ValueError):
        sp.CalibrationObservation(**payload)
    payload = {key: getattr(observation, key)
               for key in observation.to_record()}
    payload["selected_b_ratio"] = (1, 3)
    with pytest.raises(ValueError):
        sp.CalibrationObservation(**payload)


def test_control_and_label_refusals():
    with pytest.raises(ValueError):
        sp.class_of_ordinal(96)
    with pytest.raises(ValueError):
        sp.class_of_ordinal(-1)
    with pytest.raises(ValueError):
        sp.CalibrationLabels(entries=({"control_id": "a" * 64, "class": "positive",
                                       "components": [[1, 8]], "absence_reason": None},),
                             labels_sha256="b" * 64)
    with pytest.raises(ValueError):
        sp.CalibrationLabels(entries=({"control_id": "a" * 64,
                                       "class": "textureless",
                                       "components": None,
                                       "absence_reason": None},),
                             labels_sha256="b" * 64)


# --------------------------------------------------------------------------
# 9. Evaluator checks and label-blind gates
# --------------------------------------------------------------------------

def test_observation_matches_frozen_detects_a_wrong_bound_axis(monkeypatch):
    observation = sp._make_observation(48, 0, 1, 0)  # noqa: SLF001
    assert observation.local_axis == "v"
    assert sp.observation_matches_frozen(observation) is True
    monkeypatch.setattr(sp, "local_axis", lambda k, w, j: "u")
    assert sp.observation_matches_frozen(observation) is False


def _resealed_corpus_with_mutated_patch(corpus):
    """Alter one observation's patch bytes and fully reseal its digests.

    Returns ``(corpus, observation)``.  The mutated carrier is internally
    consistent (patch digest, derived non-owning digest, observation, control,
    and corpus all match the altered bytes), so only the frozen regeneration
    check may reject it.
    """
    original = corpus.observations[0]
    mutated_patch = np.array(original.patch(), dtype="<f8", copy=True)
    mutated_patch[0, 0] = mutated_patch[0, 0] + 1.0
    geometry = sp.observation_geometry(original)
    non_owning_axis = "v" if original.local_axis == "u" else "u"
    non_owning_digest, _ = sp._profile_digest(  # noqa: SLF001
        mutated_patch, geometry, non_owning_axis)
    payload = {name: getattr(original, name)
               for name in original.to_record()}
    payload["patch_float64_le_base64"] = base64.b64encode(
        np.ascontiguousarray(mutated_patch).tobytes(order="C")).decode("ascii")
    payload["patch_sha256"] = sp.patch_sha256(mutated_patch)
    payload["non_owning_profile_sha256"] = non_owning_digest
    payload.pop("observation_sha256", None)
    payload["observation_sha256"] = sp.digest_record(
        payload, "observation_sha256")
    mutated = sp.CalibrationObservation(**payload)
    controls = []
    for control in corpus.controls:
        observations = tuple(
            mutated if item.observation_id == mutated.observation_id else item
            for item in control.observations)
        controls.append(sp.make_control(control.ordinal, corpus.sigma,
                                        observations))
    return sp.make_corpus(tuple(controls), corpus.sigma,
                          corpus.labels_sha256), mutated


def test_observation_matches_frozen_refuses_a_fully_resealed_patch_mutant(
        reduced_splits):
    calibration, _ = reduced_splits
    corpus = calibration["corpus"]
    mutated_corpus, mutated = _resealed_corpus_with_mutated_patch(corpus)
    original = corpus.observations[0]
    # The carrier really is resealed: the altered bytes hash consistently and
    # every enclosing digest changed.
    assert mutated.patch_sha256 != original.patch_sha256
    assert sp.patch_sha256(mutated.patch()) == mutated.patch_sha256
    assert mutated.observation_sha256 != original.observation_sha256
    assert mutated_corpus.corpus_sha256 != corpus.corpus_sha256
    # Only the frozen regeneration refuses it.
    assert sp.observation_matches_frozen(mutated) is False
    outcome = sp.predict_corpus(mutated_corpus, sp.FROZEN_GRID)
    assert outcome["status"] == "refused"
    assert outcome["reason"] == "split_not_reconstructible"


def test_predict_never_opens_a_file_or_label(monkeypatch, reduced_splits):
    calibration, _ = reduced_splits
    original = builtins.open

    def refuse_open(*args, **kwargs):
        raise AssertionError("predict must not open any file or label")

    monkeypatch.setattr(builtins, "open", refuse_open)
    try:
        outcome = sp.predict_corpus(calibration["corpus"], sp.FROZEN_GRID)
    finally:
        monkeypatch.setattr(builtins, "open", original)
    assert outcome["status"] == "complete"


def test_module_has_no_io_or_forbidden_imports():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = ("dataset", "camera", "ground_truth", "f3", "evaluation",
                         "plane_residue", "supported", "cli", "renderer", "export")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(part in alias.name for part in forbidden_modules)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not any(part in module for part in forbidden_modules)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in ("open", "exec", "eval", "compile",
                                            "__import__")
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("read_text", "read_bytes",
                                              "write_text", "write_bytes")
    for name in ("predict_corpus", "_build_prediction_payload",
                 "_produce_observation", "_validate_predict_workload"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == name)
        text = ast.get_source_segment(source, function)
        assert "alpha_dyadic_hidden" not in text
        assert "SplitPhaseCalibration" not in text
        assert "label_payload" not in text
        assert "open(" not in text


def test_script_predict_path_never_opens_labels():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "command_predict")
    text = ast.get_source_segment(source, function)
    assert "open(" not in text
    assert "labels" not in text.lower().replace("--labels-sha256", "") \
        .replace("labels_sha256", "")
    assert "read_text" not in text


def test_module_predicts_only_from_evidence_not_the_generator_schedule():
    observation = sp._make_observation(49, 0, 0, 0)  # noqa: SLF001
    zeros = np.zeros(64, dtype="<f8")
    weights = np.full(64, 32.0, dtype="<f8")
    for hypothesis in observation.H_w:
        outcome = sp.classify_observation(
            observation, sp.dyadic_fraction(hypothesis), Fraction(1, 8),
            zeros, weights)
        assert outcome["margin"] == 0
    substituted = np.square(np.arange(1, 65, dtype="<f8"))
    changed = sp.classify_observation(
        observation, sp.dyadic_fraction(observation.H_w[0]), Fraction(1, 8),
        substituted, weights)
    assert changed["margin"] != 0, changed
    assert changed["margin"] != sp.classify_observation(
        observation, sp.dyadic_fraction(observation.H_w[0]), Fraction(1, 8),
        zeros, weights)["margin"]


def test_labels_are_only_consumed_by_score(reduced_splits):
    calibration, evaluation = reduced_splits
    labels = calibration["labels"]
    assert labels.truth()[sp.control_id(0, 49)] == (
        (1, 8), (5, 16), (1, 2))
    # The sealed file exposes truth only through the label entries, and the
    # corpus carries no alpha truth at all.
    record = labels.to_record()
    assert set(record) == {"entries", "labels_sha256"}
    assert record["labels_sha256"] == labels.labels_sha256
    for observation in calibration["corpus"].observations:
        assert "alpha_dyadic_hidden" not in observation.redacted_record()
        assert "label_payload_sha256" not in observation.redacted_record()
    assert sp.disjointness_refusal(calibration["corpus"],
                                   evaluation["corpus"]) is None


# --------------------------------------------------------------------------
# 10. P0 call boundary: synthesis never calls P0, predict calls it once
# --------------------------------------------------------------------------

def _refuse_p0(*args, **kwargs):
    raise AssertionError("synthesis must not call the P0 producer or validator")


def test_synthesis_never_calls_the_p0_producer_or_validator(monkeypatch):
    # Prereg D2/§5: P0 is called once per observation, only in label-blind
    # predict.  Synthesis derives the retained non-owning digest itself.
    monkeypatch.setattr(lp, "produce_split_phase_evidence", _refuse_p0)
    monkeypatch.setattr(lp, "validate_split_phase_evidence", _refuse_p0)
    for sigma in (0, 1):
        split = sp.synthesize_split(sigma, ordinals=(0, 16, 32, 48, 49, 95))
        assert len(split["corpus"].controls) == 6
        assert len(split["corpus"].observations) == 6 * 9
    # With the real P0 restored, the derived digest is the exact contract
    # digest of the independently produced non-owning evidence.
    monkeypatch.undo()
    for k, w, j, sigma in ((0, 0, 0, 0), (16, 2, 1, 0), (48, 0, 1, 0),
                           (49, 1, 2, 0), (95, 0, 2, 1)):
        observation = sp._make_observation(k, w, j, sigma)  # noqa: SLF001
        evidence = {item.local_axis: item for item in
                    sp._produce_observation(  # noqa: SLF001
                        observation).seam_profile_evidence}
        other = "v" if observation.local_axis == "u" else "u"
        assert evidence[other].d_profile_sha256 \
            == observation.non_owning_profile_sha256
        assert evidence[observation.local_axis].d_profile_sha256 \
            != observation.non_owning_profile_sha256


@pytest.mark.parametrize("ordinals", [REDUCED_ORDINALS, None],
                         ids=["reduced", "full"])
def test_predict_calls_the_producer_exactly_once_per_observation(
        monkeypatch, ordinals):
    calibration = sp.synthesize_split(0, ordinals=ordinals)
    corpus = calibration["corpus"]
    calls = {"producer": 0, "validator": 0}
    original_produce = lp.produce_split_phase_evidence
    original_validate = lp.validate_split_phase_evidence

    def counting_produce(*args, **kwargs):
        calls["producer"] += 1
        return original_produce(*args, **kwargs)

    def counting_validate(*args, **kwargs):
        calls["validator"] += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(lp, "produce_split_phase_evidence", counting_produce)
    monkeypatch.setattr(lp, "validate_split_phase_evidence", counting_validate)
    outcome = sp.predict_corpus(corpus, sp.FROZEN_GRID)
    assert outcome["status"] == "complete"
    assert calls["producer"] == len(corpus.observations)
    assert calls["validator"] >= len(corpus.observations)
    assert len(corpus.observations) <= sp.MAX_PRODUCER_CALLS_PER_SPLIT
    # The retained profile is replay-checked against the synthesis-declared
    # digest and the bound axis is the only consumed one.
    for entry, observation in zip(outcome["payload"]["per_observation"],
                                  corpus.observations):
        assert entry["non_owning_profile_sha256"] \
            == observation.non_owning_profile_sha256
        assert entry["local_axis"] == observation.local_axis
        assert entry["consumed_profile_sha256"] \
            != entry["non_owning_profile_sha256"]


def test_predict_binds_a_distinct_frame_identity_per_observation(monkeypatch):
    # The P0 frame identity must carry the derived j, so the three frames of
    # each control/world axis (and all nine frames of each control) stay
    # distinct instead of reusing one identity.
    calibration = sp.synthesize_split(0, ordinals=REDUCED_ORDINALS)
    corpus = calibration["corpus"]
    frames = []
    original_produce = lp.produce_split_phase_evidence

    def capturing_produce(*args, **kwargs):
        frames.append(kwargs["frame"])
        return original_produce(*args, **kwargs)

    monkeypatch.setattr(lp, "produce_split_phase_evidence", capturing_produce)
    outcome = sp.predict_corpus(corpus, sp.FROZEN_GRID)
    assert outcome["status"] == "complete"
    assert len(frames) == len(corpus.observations)
    frames_by_control_axis = {}
    frames_by_control = {}
    for frame, observation in zip(frames, corpus.observations):
        frames_by_control_axis.setdefault(
            (observation.control_id, observation.world_axis), []).append(frame)
        frames_by_control.setdefault(observation.control_id, []).append(frame)
    assert set(frames_by_control_axis) == {
        (control.control_id, world_axis)
        for control in corpus.controls for world_axis in (0, 1, 2)}
    for identities in frames_by_control_axis.values():
        assert len(identities) == sp.MAX_FRAMES_PER_AXIS
        assert len(set(identities)) == sp.MAX_FRAMES_PER_AXIS
    for identities in frames_by_control.values():
        assert len(identities) \
            == sp.MAX_AXES_PER_CONTROL * sp.MAX_FRAMES_PER_AXIS
        assert len(set(identities)) \
            == sp.MAX_AXES_PER_CONTROL * sp.MAX_FRAMES_PER_AXIS

