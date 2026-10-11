"""Authored-pixel synthetic controls for the spatial-context A/B descriptor contract.

Every fixture below is authored numpy pixel data. No screenshot, mask file,
raw-mask decode, inference, OpenCV/SIFT, GPU, or production import is used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "evaluation" / "spatial_context_descriptor.py"
_SPEC = importlib.util.spec_from_file_location("spatial_context_descriptor", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
scd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scd  # dataclasses requires the module to be registered
_SPEC.loader.exec_module(scd)

LEFT_CENTERS = ((40, 40), (136, 40), (40, 136), (136, 136))
TRANSLATION = (16, 32)
PATTERNS = ("H", "V", "D", "E")
SHAPE = (256, 256)


def _tile_value(pattern: str, u: int, v: int) -> int:
    if pattern == "H":
        return 64 + 4 * u
    if pattern == "V":
        return 64 + 4 * v
    if pattern == "D":
        return 32 + 4 * (u + v)
    if pattern == "E":
        return 96 + 4 * (u - v)
    raise ValueError(pattern)


def _fill_patch(image, mask, center_x, center_y, neighbor_pattern, blank_central=False):
    x0, y0 = center_x - 24, center_y - 24
    for tile_y in range(3):
        for tile_x in range(3):
            index = tile_y * 3 + tile_x
            pattern = "H" if index == 4 else neighbor_pattern
            blank = blank_central and index == 4
            for v in range(16):
                for u in range(16):
                    value = 100 if blank else _tile_value(pattern, u, v)
                    py, px = y0 + tile_y * 16 + v, x0 + tile_x * 16 + u
                    image[py, px, :] = value
                    mask[py, px] = True


def build_fixture(neighbor_patterns=PATTERNS, blank_central=False):
    left = np.zeros((*SHAPE, 3), np.uint8)
    left_mask = np.zeros(SHAPE, bool)
    right = np.zeros((*SHAPE, 3), np.uint8)
    right_mask = np.zeros(SHAPE, bool)
    for patch, (x, y) in enumerate(LEFT_CENTERS):
        _fill_patch(left, left_mask, x, y, neighbor_patterns[patch], blank_central=blank_central)
        _fill_patch(
            right,
            right_mask,
            x + TRANSLATION[0],
            y + TRANSLATION[1],
            neighbor_patterns[patch],
            blank_central=blank_central,
        )
    return left, left_mask, right, right_mask


def usable(shape=SHAPE):
    return np.ones(shape, bool)


def perturb_outside_mask(image, mask):
    """Replace only out-of-mask pixels; in-mask pixels stay byte-identical."""
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width]
    result = image.copy()
    for channel in range(3):
        values = ((17 * xs + 31 * ys + 67 * channel + 13) % 256).astype(np.uint8)
        result[:, :, channel] = np.where(mask, image[:, :, channel], values)
    return result


def analyze(arm, left, left_mask, right, right_mask, **limits):
    return scd.analyze_arm(arm, left, left_mask, usable(), right, right_mask, usable(), **limits)


# ---------------------------------------------------------------------------
# Site planning and identity
# ---------------------------------------------------------------------------


def test_plan_sites_selects_exactly_four_contained_centers():
    left, left_mask, _, _ = build_fixture()
    plan = scd.plan_sites(left, left_mask, usable())
    assert plan.grid_centers == 16 * 16
    assert plan.eligible_count == 4
    assert [site.xy for site in plan.sites] == list(LEFT_CENTERS)
    assert len({site.site_id for site in plan.sites}) == 4
    assert plan.source_mask_sha256 == scd.sha(left_mask.tobytes(order="C"))


def test_site_identity_binds_mask_digest_but_not_center_only():
    left, left_mask, _, _ = build_fixture()
    baseline = scd.plan_sites(left, left_mask, usable())
    altered_mask = left_mask.copy()
    altered_mask[0, 0] = True
    altered = scd.plan_sites(left, altered_mask, usable())
    assert [site.xy for site in altered.sites] == [site.xy for site in baseline.sites]
    assert [site.site_id for site in altered.sites] != [site.site_id for site in baseline.sites]
    assert altered.source_mask_sha256 != baseline.source_mask_sha256


# ---------------------------------------------------------------------------
# Positive and periodic controls
# ---------------------------------------------------------------------------


def test_positive_context_distinct_a_ties_b_recovers_exact_quadruple():
    left, left_mask, right, right_mask = build_fixture()

    analysis_a = analyze("A", left, left_mask, right, right_mask)
    match_a = analysis_a.match
    assert match_a.forward_counts == {
        "origins": 4,
        "absent_second_neighbor": 0,
        "nearest_distance_tie": 4,
        "ratio_rejected": 0,
        "ratio_accepted": 0,
    }
    assert match_a.reverse_counts["nearest_distance_tie"] == 4
    assert match_a.pairs == ()
    assert match_a.witness["status"] == "exhaustive_no_witness"
    assert match_a.witness["work_nodes"] == 0
    assert match_a.status == "no_admitted_pairs"
    central = analysis_a.query_descriptors.descriptors
    assert central.shape == (4, scd.TILE_DIM)
    for row in range(4):
        assert np.array_equal(central[0], central[row])
        assert np.any(central[row] != 0)

    analysis_b = analyze("B", left, left_mask, right, right_mask)
    match_b = analysis_b.match
    assert analysis_b.query_descriptors.descriptors.shape == (4, scd.B_DIM)
    assert len(match_b.pairs) == 4
    expected = {(x, y): (x + TRANSLATION[0], y + TRANSLATION[1]) for x, y in LEFT_CENTERS}
    observed = set()
    for pair in match_b.pairs:
        left_xy = tuple(pair["left_xy"])
        right_xy = tuple(pair["right_xy"])
        assert expected[left_xy] == right_xy
        assert pair["distance_squared"] == 0.0
        observed.add((left_xy, right_xy))
    assert observed == set(expected.items())
    assert match_b.forward_counts["ratio_accepted"] == 4
    assert match_b.forward_counts["ratio_rejected"] == 0
    assert match_b.forward_counts["nearest_distance_tie"] == 0
    assert match_b.reverse_counts["ratio_accepted"] == 4
    assert match_b.witness["status"] == "found"
    assert len(match_b.witness["pair_ids"]) == 4
    assert match_b.witness["work_nodes"] == 1
    assert match_b.witness["determinants"] == 8
    assert match_b.status == "candidate_discrimination_witness"
    work = match_b.comparison_work
    assert work["requested"] == work["charged"] == work["processed"] == 16
    assert work["uncharged"] == 0 and work["unprocessed"] == 0

    # Identical shared domains for both arms.
    assert match_a.query_domain_ids == match_b.query_domain_ids
    assert match_a.reference_domain_ids == match_b.reference_domain_ids


def test_fully_periodic_negative_a_and_b_tie_and_abstain():
    left, left_mask, right, right_mask = build_fixture(neighbor_patterns=("H", "H", "H", "H"))
    for arm in ("A", "B"):
        analysis = analyze(arm, left, left_mask, right, right_mask)
        match = analysis.match
        assert match.forward_counts["nearest_distance_tie"] == 4
        assert match.forward_counts["ratio_accepted"] == 0
        assert match.reverse_counts["nearest_distance_tie"] == 4
        assert match.pairs == ()
        assert match.witness["status"] == "exhaustive_no_witness"
        assert match.status == "no_admitted_pairs"
        descriptors = analysis.query_descriptors.descriptors
        assert np.any(descriptors[0] != 0)
        for row in range(4):
            assert np.array_equal(descriptors[0], descriptors[row])


def test_witness_requires_nondegenerate_quartet():
    collinear = [
        scd.make_pair_record(f"L{index}", f"R{index}", (index, 0), (index, 0), 0.0)
        for index in range(4)
    ]
    determinants = scd.Budget(scd.WITNESS_DETERMINANT_LIMIT)
    result = scd.common_four(collinear, scd.Budget(10), determinants)
    assert result["status"] == "exhaustive_no_witness"
    assert result["determinants"] == 8


# ---------------------------------------------------------------------------
# Containment / domain controls
# ---------------------------------------------------------------------------


def test_outside_mask_perturbation_preserves_numerical_payloads():
    left, left_mask, right, right_mask = build_fixture()
    baseline = analyze("B", left, left_mask, right, right_mask)
    perturbed = analyze(
        "B",
        perturb_outside_mask(left, left_mask),
        left_mask,
        perturb_outside_mask(right, right_mask),
        right_mask,
    )
    assert baseline.query_descriptors.descriptors.tobytes() == perturbed.query_descriptors.descriptors.tobytes()
    assert baseline.reference_descriptors.descriptors.tobytes() == perturbed.reference_descriptors.descriptors.tobytes()
    assert baseline.query_descriptors.tile_tables.tobytes() == perturbed.query_descriptors.tile_tables.tobytes()
    assert baseline.reference_descriptors.tile_tables.tobytes() == perturbed.reference_descriptors.tile_tables.tobytes()
    assert baseline.match.distance_matrix.tobytes() == perturbed.match.distance_matrix.tobytes()
    assert [site.xy for site in baseline.query_descriptors.sites] == [
        site.xy for site in perturbed.query_descriptors.sites
    ]
    assert baseline.match.forward_counts == perturbed.match.forward_counts
    assert baseline.match.reverse_counts == perturbed.match.reverse_counts
    baseline_pairs = {
        (tuple(pair["left_xy"]), tuple(pair["right_xy"]), pair["distance_squared"])
        for pair in baseline.match.pairs
    }
    perturbed_pairs = {
        (tuple(pair["left_xy"]), tuple(pair["right_xy"]), pair["distance_squared"])
        for pair in perturbed.match.pairs
    }
    assert baseline_pairs == perturbed_pairs
    assert baseline.match.witness["status"] == perturbed.match.witness["status"]
    assert baseline.match.witness["work_nodes"] == perturbed.match.witness["work_nodes"]
    # Provenance digests must change; source-bound IDs are not asserted unchanged.
    assert baseline.query_plan.source_mask_sha256 == perturbed.query_plan.source_mask_sha256
    assert baseline.query_plan.frame_sha256 != perturbed.query_plan.frame_sha256
    assert baseline.match.source_binding_sha256 != perturbed.match.source_binding_sha256


def test_blank_central_tile_gives_empty_shared_domain():
    left, left_mask, right, right_mask = build_fixture(blank_central=True)
    for arm in ("A", "B"):
        analysis = analyze(arm, left, left_mask, right, right_mask)
        assert analysis.query_descriptors.shared_indices.size == 0
        assert analysis.reference_descriptors.shared_indices.size == 0
        assert analysis.query_descriptors.central_zero.all()
        assert (analysis.query_descriptors.neighbor_zero_counts == 0).all()
        match = analysis.match
        assert match.status == "insufficient_support"
        assert match.pairs == ()
        assert match.witness["status"] == "exhaustive_no_witness"
        assert match.comparison_work["requested"] == 0
        assert match.comparison_work["charged"] == 0
        assert match.comparison_work["processed"] == 0


def test_single_false_pixel_excludes_site_in_both_arms():
    left, left_mask, right, right_mask = build_fixture()
    left_mask[40, 40] = False
    for arm in ("A", "B"):
        analysis = analyze(arm, left, left_mask, right, right_mask)
        coordinates = [site.xy for site in analysis.query_descriptors.sites]
        assert (40, 40) not in coordinates
        assert coordinates == [(136, 40), (40, 136), (136, 136)]
    assert len(analyze("B", left, left_mask, right, right_mask).match.pairs) == 3


# ---------------------------------------------------------------------------
# Signed arithmetic, role swap, enumeration permutation
# ---------------------------------------------------------------------------


def test_descending_255_to_0_edge_uses_signed_float_arithmetic():
    tile = np.zeros((16, 16, 3), np.uint8)
    for u in range(16):
        tile[:, u, :] = 255 - 17 * u
    luma = scd.luma_from_rgb(tile)
    gx, gy = scd.tile_gradients(luma)
    assert gx.shape == (14, 14)
    assert np.all(gx < 0)
    assert np.allclose(gx, -34.0 / 255.0, rtol=0.0, atol=1e-12)
    assert np.all(gy == 0.0)


def test_role_swap_recovers_inverse_translation():
    left, left_mask, right, right_mask = build_fixture()
    forward = analyze("B", left, left_mask, right, right_mask).match
    reverse = analyze("B", right, right_mask, left, left_mask).match
    assert len(forward.pairs) == 4
    assert len(reverse.pairs) == 4
    forward_map = {(tuple(pair["left_xy"]), tuple(pair["right_xy"])) for pair in forward.pairs}
    reverse_map = {(tuple(pair["left_xy"]), tuple(pair["right_xy"])) for pair in reverse.pairs}
    assert reverse_map == {(right_xy, left_xy) for left_xy, right_xy in forward_map}
    for pair in reverse.pairs:
        left_xy = tuple(pair["left_xy"])
        assert left_xy == (pair["right_xy"][0] + TRANSLATION[0], pair["right_xy"][1] + TRANSLATION[1])


def test_shuffled_enumeration_preserves_canonical_result():
    left, left_mask, right, right_mask = build_fixture()
    query_plan = scd.plan_sites(left, left_mask, usable())
    reference_plan = scd.plan_sites(right, right_mask, usable())
    canonical = scd.match_arms(
        scd.describe_arm(left, query_plan.sites, "B"),
        scd.describe_arm(right, reference_plan.sites, "B"),
    )
    shuffled = scd.match_arms(
        scd.describe_arm(left, list(reversed(query_plan.sites)), "B"),
        scd.describe_arm(right, list(reversed(reference_plan.sites)), "B"),
    )
    assert canonical.query_domain_ids == shuffled.query_domain_ids
    assert canonical.reference_domain_ids == shuffled.reference_domain_ids
    assert canonical.distance_matrix.tobytes() == shuffled.distance_matrix.tobytes()
    assert [pair["pair_id"] for pair in canonical.pairs] == [pair["pair_id"] for pair in shuffled.pairs]
    assert canonical.witness["pair_ids"] == shuffled.witness["pair_ids"]


# ---------------------------------------------------------------------------
# Budgets, caps, validation
# ---------------------------------------------------------------------------


def test_comparison_budget_exact_cap_and_cap_plus_one():
    left, left_mask, right, right_mask = build_fixture()
    exact = analyze("B", left, left_mask, right, right_mask, comparison_limit=16)
    work = exact.match.comparison_work
    assert work["limit"] == 16
    assert work["requested"] == 16
    assert work["charged"] == 16
    assert work["processed"] == 16
    with pytest.raises(scd.Refusal):
        analyze("B", left, left_mask, right, right_mask, comparison_limit=15)


def test_witness_budget_exact_cap_and_cap_plus_one():
    pairs = [
        scd.make_pair_record("L0", f"R{index}", (0, 0), (index, 0), float(index))
        for index in range(5)
    ]
    determinants = scd.Budget(scd.WITNESS_DETERMINANT_LIMIT)
    exact = scd.common_four(pairs, scd.Budget(5), determinants)
    assert exact["status"] == "exhaustive_no_witness"
    assert exact["work_nodes"] == 5
    assert determinants.requested == determinants.charged == determinants.processed == 0

    capped = scd.Budget(4)
    over = scd.common_four(pairs, capped, scd.Budget(scd.WITNESS_DETERMINANT_LIMIT))
    assert over["status"] == "budget_inconclusive"
    assert over["work_nodes"] == 4
    assert capped.requested == 5
    assert capped.charged == capped.processed == 4
    assert over["node_work"] == capped.close()
    assert over["node_work"]["uncharged"] == 1
    assert over["node_work"]["unprocessed"] == 0


def test_site_caps_exact_and_cap_plus_one():
    assert scd.check_site_caps(scd.LEFT_SITE_LIMIT, scd.RIGHT_SITE_LIMIT) is None
    for left_count, right_count in (
        (scd.LEFT_SITE_LIMIT + 1, 0),
        (0, scd.RIGHT_SITE_LIMIT + 1),
        (scd.LEFT_SITE_LIMIT, scd.RIGHT_SITE_LIMIT + 1),
        (scd.LEFT_SITE_LIMIT + 1, scd.RIGHT_SITE_LIMIT),
        (-1, 0),
        (0, -1),
    ):
        with pytest.raises(scd.Refusal):
            scd.check_site_caps(left_count, right_count)


def test_exact_luma_conversion_and_dimensions():
    pixels = np.array([[[0, 0, 0], [255, 255, 255], [1, 2, 3]]], np.uint8)
    luma = scd.luma_from_rgb(pixels)
    assert luma.shape == (1, 3)
    assert luma[0, 0] == 0.0
    assert luma[0, 1] == 1.0
    assert luma[0, 2] == ((77 * 1 + 150 * 2 + 29 * 3 + 128) // 256) / 255.0


def test_arm_descriptor_dimensions_and_dtype():
    left, left_mask, _, _ = build_fixture()
    sites = scd.plan_sites(left, left_mask, usable()).sites
    arm_a = scd.describe_arm(left, sites, "A")
    arm_b = scd.describe_arm(left, sites, "B")
    assert arm_a.descriptors.shape == (4, scd.TILE_DIM)
    assert arm_b.descriptors.shape == (4, scd.B_DIM)
    assert arm_a.descriptors.dtype == scd.FLOAT32_LE
    assert arm_b.descriptors.dtype == scd.FLOAT32_LE
    assert np.any(arm_a.descriptors[0] != 0)
    assert np.any(arm_b.descriptors[0] != 0)
    assert len(set(arm_a.descriptor_identities())) == 4


def test_malformed_or_nonfinite_inputs_fail_closed():
    good = np.zeros((64, 64, 3), np.uint8)
    good_mask = np.ones((64, 64), bool)
    with pytest.raises(scd.Refusal):
        scd.plan_sites([[0]], good_mask, good_mask)
    with pytest.raises(scd.Refusal):
        scd.plan_sites(good.astype(np.float32), good_mask, good_mask)
    with pytest.raises(scd.Refusal):
        scd.plan_sites(np.zeros((64, 64, 4), np.uint8), good_mask, good_mask)
    with pytest.raises(scd.Refusal):
        scd.plan_sites(good, np.ones((64, 64), np.uint8), good_mask)
    with pytest.raises(scd.Refusal):
        scd.plan_sites(good, np.ones((63, 64), bool), good_mask)
    with pytest.raises(scd.Refusal):
        scd.describe_arm(good, [], "C")

    nonfinite = np.zeros((1, scd.TILE_DIM), scd.FLOAT32_LE)
    nonfinite[0, 0] = np.nan
    with pytest.raises(scd.Refusal):
        scd.validate_descriptor_array(nonfinite, scd.TILE_DIM)
    with pytest.raises(scd.Refusal):
        scd.validate_descriptor_array(np.zeros((1, scd.TILE_DIM), np.float64), scd.TILE_DIM)


def _empty_arm(sites=()):
    """No pixels/descriptors allocated for public-entry guard controls."""
    return scd.ArmDescriptors(
        "A", tuple(sites), np.empty((0, scd.TILE_DIM), scd.FLOAT32_LE),
        np.zeros(0, bool), np.zeros(0, np.int64),
        np.empty((0, 9, scd.TILE_DIM), scd.FLOAT32_LE), "synthetic", SHAPE,
    )


@pytest.mark.parametrize("name,ceiling", [
    ("comparison_limit", scd.COMPARISON_LIMIT),
    ("witness_node_limit", scd.WITNESS_NODE_LIMIT),
    ("witness_determinant_limit", scd.WITNESS_DETERMINANT_LIMIT),
])
def test_direct_match_frozen_limits_accept_exact_and_refuse_cap_plus_one(monkeypatch, name, ceiling):
    query, reference = _empty_arm(), _empty_arm()
    exact = scd.match_arms(query, reference, **{name: ceiling})
    assert exact.distance_matrix.shape == (0, 0)
    assert exact.witness["status"] == "exhaustive_no_witness"

    def unexpected(*args, **kwargs):
        raise AssertionError("validation/allocation before frozen-limit refusal")

    monkeypatch.setattr(scd, "validate_descriptor_array", unexpected)
    monkeypatch.setattr(scd, "pairwise_squared", unexpected)
    with pytest.raises(scd.Refusal, match=name + "_above_frozen_cap"):
        scd.match_arms(query, reference, **{name: ceiling + 1})
    for invalid in (-1, 1.5, True, None):
        with pytest.raises(scd.Refusal, match="invalid_" + name):
            scd.match_arms(query, reference, **{name: invalid})


@pytest.mark.parametrize("side,cap", [("query", scd.LEFT_SITE_LIMIT), ("reference", scd.RIGHT_SITE_LIMIT)])
def test_direct_match_site_cap_refuses_before_descriptor_validation_or_distance(monkeypatch, side, cap):
    oversized = _empty_arm((scd.Site("unused", 40, 40),) * (cap + 1))

    def unexpected(*args, **kwargs):
        raise AssertionError("descriptor work reached before direct site-cap refusal")

    monkeypatch.setattr(scd, "validate_descriptor_array", unexpected)
    monkeypatch.setattr(scd, "pairwise_squared", unexpected)
    query, reference = (oversized, _empty_arm()) if side == "query" else (_empty_arm(), oversized)
    with pytest.raises(scd.Refusal, match="left_site_limit" if side == "query" else "right_site_limit"):
        scd.match_arms(query, reference)


def _nondegenerate_pairs():
    return [scd.make_pair_record(f"L{i}", f"R{i}", xy, (xy[0] + 16, xy[1] + 32), 0.0)
            for i, xy in enumerate(((0, 0), (2, 0), (0, 2), (2, 2)))]


@pytest.mark.parametrize("determinant_limit,expected", [(8, "found"), (7, "budget_inconclusive")])
def test_common_four_determinant_batch_exact_cap_and_one_required_over_cap(monkeypatch, determinant_limit, expected):
    nodes, determinants = scd.Budget(1), scd.Budget(determinant_limit)
    original = scd.good_four
    evaluations = []

    def precharged(points):
        assert nodes.charged == 1 and determinants.charged == 8
        evaluations.append(tuple(points))
        return original(points)

    monkeypatch.setattr(scd, "good_four", precharged)
    result = scd.common_four(_nondegenerate_pairs(), nodes, determinants)
    assert result["status"] == expected
    assert result["node_work"] == nodes.close()
    assert result["determinant_work"] == determinants.close()
    assert nodes.requested == nodes.charged == 1
    assert determinants.requested == 8
    if determinant_limit == 8:
        assert len(evaluations) == 2
        assert nodes.processed == 1
        assert determinants.charged == determinants.processed == 8
        assert result["determinant_work"]["uncharged"] == 0
        assert result["node_work"]["unprocessed"] == 0
    else:
        assert evaluations == []
        assert nodes.processed == 0
        assert determinants.charged == determinants.processed == 0
        assert result["determinant_work"]["uncharged"] == 8
        assert result["node_work"]["unprocessed"] == 1
        assert result["pair_ids"] == result["coordinates"] == []


def test_common_four_empty_exhaustion_and_refused_first_node_counter_closure(monkeypatch):
    empty = scd.common_four([], scd.Budget(0), scd.Budget(0))
    assert empty["status"] == "exhaustive_no_witness"
    for name in ("node_work", "determinant_work"):
        assert all(empty[name][key] == 0 for key in ("requested", "charged", "processed", "uncharged", "unprocessed"))

    def unexpected(points):
        raise AssertionError("determinants evaluated with unchargeable first tuple")

    monkeypatch.setattr(scd, "good_four", unexpected)
    result = scd.common_four(_nondegenerate_pairs(), scd.Budget(0), scd.Budget(8))
    assert result["status"] == "budget_inconclusive"
    assert result["node_work"] == {"limit": 0, "requested": 1, "charged": 0, "processed": 0, "uncharged": 1, "unprocessed": 0}
    assert result["determinant_work"]["requested"] == 0


def test_common_four_determinant_exact_final_completion_is_exhaustive_not_inconclusive():
    pairs = [scd.make_pair_record(f"L{i}", f"R{i}", (i, 0), (i + 16, 32), 0.0) for i in range(4)]
    result = scd.common_four(pairs, scd.Budget(1), scd.Budget(8))
    assert result["status"] == "exhaustive_no_witness"
    assert result["node_work"]["requested"] == result["node_work"]["charged"] == result["node_work"]["processed"] == 1
    assert result["determinant_work"]["requested"] == result["determinant_work"]["charged"] == result["determinant_work"]["processed"] == 8


def test_common_four_closures_keep_previous_budget_work_and_rejected_batch():
    nodes, determinants = scd.Budget(2), scd.Budget(15)
    nodes.reserve(1)
    nodes.processed += 1
    determinants.reserve(8)
    determinants.processed += 8
    result = scd.common_four(_nondegenerate_pairs(), nodes, determinants)
    assert result["status"] == "budget_inconclusive"
    assert result["work_nodes"] == 1 and result["determinants"] == 0
    assert result["node_work"] == {"limit": 2, "requested": 2, "charged": 2, "processed": 1, "uncharged": 0, "unprocessed": 1}
    assert result["determinant_work"] == {"limit": 15, "requested": 16, "charged": 8, "processed": 8, "uncharged": 8, "unprocessed": 0}


@pytest.mark.parametrize("limits", [{"witness_node_limit": 0}, {"witness_determinant_limit": 7}])
def test_public_matching_retains_pairs_but_abstains_when_common_witness_budget_refused(limits):
    left, left_mask, right, right_mask = build_fixture()
    match = analyze("B", left, left_mask, right, right_mask, **limits).match
    assert len(match.pairs) == 4
    assert match.status == "witness_budget_inconclusive"
    assert match.witness["status"] == "budget_inconclusive"
    for name in ("node_work", "determinant_work"):
        work = match.witness[name]
        assert work["requested"] >= work["charged"] >= work["processed"] >= 0
        assert work["uncharged"] == work["requested"] - work["charged"]
        assert work["unprocessed"] == work["charged"] - work["processed"]
    assert match.witness["pair_ids"] == match.witness["coordinates"] == []


def test_support_first_four_same_canonical_ids_coordinates_readiness_over_four_sites():
    # Authored native pixels; coordinate order's first three are collinear.
    ys, xs = np.mgrid[:256, :256]
    image = np.repeat(((xs + 2 * ys) % 256).astype(np.uint8)[:, :, None], 3, axis=2)
    sites = (scd.Site("z", 40, 40), scd.Site("y", 56, 40), scd.Site("a", 72, 40),
             scd.Site("x", 40, 56), scd.Site("w", 56, 56))
    arm = scd.describe_arm(image, sites, "A")
    assert not scd.good_four([site.xy for site in arm.sites[:4]])
    by_id = {site.site_id: site.xy for site in arm.sites}
    report = scd.support_report(arm, arm)
    for block in report.values():
        ids = block["all_domain"]["first_four_ids"]
        assert ids == ["w", "x", "y", "z"]
        assert block["first_four"]["first_four_ids"] == ids
        assert block["all_domain"]["first_four_ready"] == block["first_four"]["first_four_ready"] == scd.good_four([by_id[i] for i in ids])
        assert block["first_four"]["distinct_sites"] == 4
        assert block["first_four"]["all_domain_general_position"] is True
        assert [i for group in block["first_four"]["bins"] for i in group] == ids
    # Permuting descriptors and aligned metadata cannot change canonical support.
    permutation = np.arange(len(sites))[::-1]
    shuffled = scd.ArmDescriptors(arm.arm, tuple(arm.sites[i] for i in permutation),
        arm.descriptors[permutation], arm.central_zero[permutation], arm.neighbor_zero_counts[permutation],
        arm.tile_tables[permutation], arm.frame_sha256, arm.shape)
    assert scd.support_report(shuffled, shuffled) == report


def _aligned_arm(count=2, arm="A"):
    """Small descriptor objects for validation, never image-derived evidence."""
    return scd.ArmDescriptors(
        arm, tuple(scd.Site(f"synthetic-{i}", 40 + i * 16, 40) for i in range(count)),
        np.zeros((count, scd.dimension_for_arm(arm)), scd.FLOAT32_LE),
        np.zeros(count, bool), np.zeros(count, np.int64),
        np.zeros((count, 9, scd.TILE_DIM), scd.FLOAT32_LE), "synthetic", SHAPE,
    )


@pytest.mark.parametrize("field", ["descriptors", "central_zero", "neighbor_zero_counts", "tile_tables"])
@pytest.mark.parametrize("damage", ["short", "long", "dtype", "not_ndarray", "rank"])
@pytest.mark.parametrize("side", ["query", "reference"])
def test_all_arm_layouts_refuse_before_any_finite_scan_gather_or_distance(monkeypatch, field, damage, side):
    query, reference = _aligned_arm(), _aligned_arm()
    bad = query if side == "query" else reference
    array = getattr(bad, field)
    if damage == "short":
        malformed = array[:1]
    elif damage == "long":
        malformed = np.zeros((3, *array.shape[1:]), array.dtype)
    elif damage == "dtype":
        wrong_dtype = np.uint8 if field == "central_zero" else (np.int32 if field == "neighbor_zero_counts" else np.float64)
        malformed = array.astype(wrong_dtype)
    elif damage == "not_ndarray":
        malformed = array.tolist()
    else:
        malformed = array[:, None] if array.ndim == 1 else array.reshape(-1)
    setattr(bad, field, malformed)

    def unexpected(*args, **kwargs):
        raise AssertionError("numerical scan/gather/distance reached before both layout guards")

    monkeypatch.setattr(scd.np, "isfinite", unexpected)
    monkeypatch.setattr(scd, "pairwise_squared", unexpected)
    expected = field + ("_dtype" if damage == "dtype" else "_not_ndarray" if damage == "not_ndarray" else "_shape")
    with pytest.raises(scd.Refusal, match=expected):
        scd.match_arms(query, reference)


@pytest.mark.parametrize("side", ["query", "reference"])
def test_zero_site_huge_zero_copy_descriptor_view_refuses_before_finite_scan(monkeypatch, side):
    query, reference = _aligned_arm(0), _aligned_arm(0)
    base = np.zeros((1, scd.TILE_DIM), scd.FLOAT32_LE)
    huge = np.broadcast_to(base, (2**30, scd.TILE_DIM))
    assert np.shares_memory(base, huge) and base.nbytes == 128
    assert huge.nbytes == 137438953472 and not huge.flags.owndata
    (query if side == "query" else reference).descriptors = huge

    def unexpected(*args, **kwargs):
        raise AssertionError("oversized virtual descriptor rows scanned or used for distance work")

    monkeypatch.setattr(scd.np, "isfinite", unexpected)
    monkeypatch.setattr(scd, "pairwise_squared", unexpected)
    with pytest.raises(scd.Refusal, match="descriptors_shape"):
        scd.match_arms(query, reference)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_correctly_aligned_empty_arm_remains_accepted_without_descriptor_copy(monkeypatch, arm):
    query, reference = _aligned_arm(0, arm), _aligned_arm(0, arm)
    identities = [id(getattr(a, field)) for a in (query, reference)
                  for field in ("descriptors", "central_zero", "neighbor_zero_counts", "tile_tables")]
    scans = []
    finite = scd.np.isfinite

    def bounded_scan(array):
        assert array.shape == (0, scd.dimension_for_arm(arm))
        scans.append(id(array))
        return finite(array)

    monkeypatch.setattr(scd.np, "isfinite", bounded_scan)
    result = scd.match_arms(query, reference)
    assert scans == [id(query.descriptors), id(reference.descriptors)]
    assert identities == [id(getattr(a, field)) for a in (query, reference)
                          for field in ("descriptors", "central_zero", "neighbor_zero_counts", "tile_tables")]
    assert result.status == "insufficient_support"
    assert result.distance_matrix.shape == (0, 0)
    assert result.witness["status"] == "exhaustive_no_witness"


@pytest.mark.parametrize("determinant_limit", [32768, 32767])
def test_actual_frozen_witness_caps_twenty_collinear_pairs(determinant_limit):
    assert scd.WITNESS_NODE_LIMIT == 4096 and scd.WITNESS_DETERMINANT_LIMIT == 32768
    pairs = [scd.make_pair_record(f"L{i}", f"R{i}", (i, 0), (i + 16, 32), 0.0) for i in range(20)]
    nodes, determinants = scd.Budget(4096), scd.Budget(determinant_limit)
    result = scd.common_four(pairs, nodes, determinants)
    assert result["status"] == "budget_inconclusive"
    assert result["pair_ids"] == result["coordinates"] == []
    assert result["node_work"] == nodes.close()
    assert result["determinant_work"] == determinants.close()
    if determinant_limit == 32768:
        assert result["work_nodes"] == 4096 and result["determinants"] == 32768
        assert result["node_work"] == {"limit": 4096, "requested": 4097, "charged": 4096, "processed": 4096, "uncharged": 1, "unprocessed": 0}
        assert result["determinant_work"] == {"limit": 32768, "requested": 32768, "charged": 32768, "processed": 32768, "uncharged": 0, "unprocessed": 0}
    else:
        assert result["work_nodes"] == 4096 and result["determinants"] == 32760
        assert result["node_work"] == {"limit": 4096, "requested": 4096, "charged": 4096, "processed": 4095, "uncharged": 0, "unprocessed": 1}
        assert result["determinant_work"] == {"limit": 32767, "requested": 32768, "charged": 32760, "processed": 32760, "uncharged": 8, "unprocessed": 0}
