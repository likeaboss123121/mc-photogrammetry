"""Generated RGB only; the v2 held-out panel is never evaluated by these tests."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import diagnose_multiview_matcher_ablation as ablation
from scripts.evaluation import diagnose_multiview_mnn_proposals_v2 as module


@pytest.fixture(scope="module")
def dev_case():
    return next(module.synthetic_cases())


@pytest.fixture(scope="module")
def dev_report(dev_case):
    return module.diagnose(dev_case["frames"])


def reseal(report):
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = module._sha(module._canonical(payload))


def reseal_geometry(report):
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = module.geometry._digest_bytes(module.geometry._canonical(payload))


def test_low_wide_geometry_is_byte_identical_to_accepted_policy(dev_case, dev_report):
    prior = ablation._geometry_report(module.geometry._frame_input(dev_case["frames"])[0],
                                      ablation.VARIANTS[3], ablation.MAX_TOTAL_DESCRIPTOR_COMPARISONS)[0]
    assert module.geometry.canonical_report_bytes(dev_report["policies"][0]["geometry"]) == (
        module.geometry.canonical_report_bytes(prior))
    assert module.POLICIES == ("reciprocal-lowe-085", "strict-unique-mutual-nearest")
    assert dev_report["policies"][1]["geometry_compatibility_adapter"]["meaning"] == (
        "strict-unique-first-neighbor-directional-survivors-no-lowe-margin")


def test_shared_features_witness_and_partition_are_deterministic(dev_case, dev_report):
    assert dev_report == module.diagnose([(name, image.copy()) for name, image in dev_case["frames"]])
    assert module.canonical_report_bytes(dev_report)
    module.verify_frame_binding(dev_report, dev_case["frames"])
    assert dev_report["shared_descriptor_comparisons"] == 2 * sum(
        len(dev_report["feature_points"][a]) * len(dev_report["feature_points"][b])
        for a, b in module.geometry._pair_schedule(dev_report["frame_ids"]))
    for policy in dev_report["policies"]:
        for row in policy["geometry"]["pair_telemetry"]:
            proposals = row["reciprocal_pairs"]
            first = row["attempts"][0]
            expected = [i for i in range(len(proposals))
                        if module.geometry._seed(row["frame_ids"][0], proposals[i][0]) % 5 != 0]
            assert first["fit_indices"] == expected
            assert first["held_indices"] == [i for i in range(len(proposals)) if i not in expected]


def test_strict_mnn_ties_abstain_and_ratio_policy_is_distinct():
    forward = [[0, 1., 1, 1.], [1, 1., 0, 2.]]
    reverse = [[0, 1., 1, 2.], [1, 1., 0, 1.]]
    a = module._proposal_rows(forward, reverse, module.POLICIES[0])
    b = module._proposal_rows(forward, reverse, module.POLICIES[1])
    assert a == ([[1, 1]], [[0, 0]], [])
    assert b == ([[1, 1]], [[0, 0]], [])
    forward[0] = [0, 1., 1, 1.1]
    reverse[0] = [0, 1., 1, 1.1]
    assert module._proposal_rows(forward, reverse, module.POLICIES[0])[2] == []
    assert module._proposal_rows(forward, reverse, module.POLICIES[1])[2] == [[0, 0]]


@pytest.mark.parametrize("mutation", ["policy_id", "tie_rule", "adapter", "tie_count", "nearest_identity",
                                      "survivor_count", "reciprocal_identity", "geometry_identity",
                                      "frame_dimension", "shared_work"])
def test_resealed_mutation_refuses_before_publication(dev_case, dev_report, mutation, tmp_path, monkeypatch):
    report = copy.deepcopy(dev_report)
    b = report["policies"][1]
    witness = report["nearest_witness"][0]
    if mutation == "policy_id":
        b["id"] = "ratio-survivors"
    elif mutation == "tie_rule":
        b["tie_rule"] = "ties-allowed"
    elif mutation == "adapter":
        b["geometry_compatibility_adapter"]["meaning"] = "Lowe-ratio-pass"
    elif mutation == "tie_count":
        witness["forward_exact_ties"] += 1
    elif mutation == "nearest_identity":
        witness["forward"][0][0] = (witness["forward"][0][0] + 2) % len(report["feature_points"]["view-1"])
    elif mutation == "survivor_count":
        b["proposal_telemetry"][0]["forward_survivor_count"] += 1
    elif mutation == "reciprocal_identity":
        b["proposal_telemetry"][0]["reciprocal_pairs"][0][0] += 1
    elif mutation == "geometry_identity":
        b["geometry"]["pair_telemetry"][0]["reciprocal_pairs"][0][0] += 1
        reseal_geometry(b["geometry"])
    elif mutation == "frame_dimension":
        b["geometry"]["frame_dimensions"]["view-0"] = [192, 193]
        reseal_geometry(b["geometry"])
    elif mutation == "shared_work":
        report["shared_descriptor_comparisons"] += 1
    reseal(report)
    monkeypatch.setattr(module.tempfile, "mkstemp", lambda **_: pytest.fail("temporary file created"))
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "invalid.json", report, dev_case["frames"])
    assert not list(tmp_path.iterdir())


def test_source_bound_publish_no_clobber_and_fully_resealed_mismatch(dev_case, dev_report, tmp_path, monkeypatch):
    target = tmp_path / "honest.json"
    with pytest.raises(TypeError):
        module.persist_report(target, dev_report)
    assert not list(tmp_path.iterdir())
    module.persist_report(target, dev_report, dev_case["frames"])
    original = target.read_bytes()
    assert original == module.canonical_report_bytes(dev_report)
    with pytest.raises(FileExistsError):
        module.persist_report(target, dev_report, dev_case["frames"])
    assert target.read_bytes() == original
    rectangular = [(f"view-{i}", np.zeros((128, 192, 3), np.uint8)) for i in range(3)]
    mutated = module.diagnose(rectangular)
    mutated["frame_dimensions"]["view-0"] = [192, 128]
    for policy in mutated["policies"]:
        policy["geometry"]["frame_dimensions"]["view-0"] = [192, 128]
        reseal_geometry(policy["geometry"])
    reseal(mutated)
    assert module.canonical_report_bytes(mutated)
    monkeypatch.setattr(module.tempfile, "mkstemp", lambda **_: pytest.fail("temporary file created"))
    with pytest.raises(ValueError, match="frame source drift"):
        module.persist_report(tmp_path / "invalid.json", mutated, rectangular)
    assert list(tmp_path.iterdir()) == [target]
    assert target.read_bytes() == original


def test_truth_is_evaluator_only_and_pins_precede_extraction(dev_case, monkeypatch):
    assert list(inspect.signature(module.diagnose).parameters) == ["frames"]
    monkeypatch.setattr(module, "_label", lambda *_: pytest.fail("evaluator truth entered diagnosis"))
    assert module.diagnose(dev_case["frames"])["schema"] == module.SCHEMA
    with pytest.raises(TypeError):
        module.diagnose(dev_case["frames"], oracle=dev_case["oracle"])
    monkeypatch.undo()
    monkeypatch.setattr(module, "ABLATION_SHA256", "f" * 64)
    monkeypatch.setattr(module.ablation, "_extract", lambda *_: pytest.fail("extraction after pin drift"))
    with pytest.raises(ValueError, match="pin drift"):
        module.diagnose(dev_case["frames"])
    monkeypatch.undo()
    monkeypatch.setattr(module, "GEOMETRY_SHA256", "e" * 64)
    monkeypatch.setattr(module, "_extract", lambda *_: pytest.fail("extraction after geometry pin drift"))
    with pytest.raises(ValueError, match="pin drift"):
        module.diagnose(dev_case["frames"])


def test_geometry_pin_is_current_and_v1_basis_is_frozen():
    source = Path(module.geometry.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == module.GEOMETRY_SHA256
    assert module.GEOMETRY_SHA256 == "f69be2ef78e8ae90a2bad09f973ed1ce3ec37d1a3a99f77101ecb68f9b2291b0"
    assert module.GEOMETRY_SHA256 != "97eaac2bdcc09649a2a7e364ff7589f9cd09916ff62922fc8fe52f24bdff8db7"
    assert module.configuration()["v1_source_sha256"] == module.V1_SOURCE_SHA256
    assert module.configuration()["v1_test_sha256"] == module.V1_TEST_SHA256
    assert module.configuration()["geometry_source_sha256"] == module.GEOMETRY_SHA256


def test_resource_budgets_refuse_before_costly_stage(dev_case, monkeypatch):
    monkeypatch.setattr(module.geometry, "MAX_DESCRIPTOR_COMPARISONS", 1)
    monkeypatch.setattr(module, "_nearest", lambda *_: pytest.fail("nearest after work refusal"))
    with pytest.raises(ValueError, match="budget"):
        module.diagnose(dev_case["frames"])


def test_output_cap_refuses_and_publication_failure_cleans_up(dev_case, dev_report, tmp_path, monkeypatch):
    target = tmp_path / "failed.json"
    monkeypatch.setattr(module.os, "link", lambda *_: (_ for _ in ()).throw(OSError("injected link failure")))
    with pytest.raises(OSError, match="injected link failure"):
        module.persist_report(target, dev_report, dev_case["frames"])
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 1)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    with pytest.raises(ValueError, match="output budget"):
        module.diagnose(dev_case["frames"])
    assert not list(tmp_path.iterdir())


def test_task_shared_work_refusal_precedes_partial_score(monkeypatch):
    monkeypatch.setattr(module, "MAX_SHARED_COMPARISONS", 0)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    monkeypatch.setattr(module, "_nearest", lambda *_: pytest.fail("nearest query after shared work refusal"))
    monkeypatch.setattr(module, "_case_metrics", lambda *_: pytest.fail("partial score after work refusal"))
    with pytest.raises(ValueError, match="task shared descriptor work budget"):
        module.calibrate_synthetic_dev()


def test_generated_group_schedule_is_frozen_and_heldout_is_not_authorized():
    assert len(module.SEEDS) == 8 and len(set(seed for _, _, seed in module.SEEDS)) == 8
    assert [split for split, _, _ in module.SEEDS] == ["dev"] * 4 + ["heldout"] * 4
    dev_groups = {(family, seed) for split, family, seed in module.SEEDS if split == "dev"}
    heldout_groups = {(family, seed) for split, family, seed in module.SEEDS if split == "heldout"}
    assert not dev_groups & heldout_groups
    assert set(family for _, family, _ in module.SEEDS) == {
        "weak-contrast", "repeated-brick-tile", "mixed-scale-pixel", "sparse-corners"}
    assert set(module.AUTHORIZED_CALIBRATION_SPLITS) == {"dev"}
    with pytest.raises(ValueError, match="not authorized"):
        next(module.synthetic_cases("heldout"))
    assert len(module.CASE_KINDS) == 8 and module.MAX_CASES == 64 and module.MAX_DEV_CASES == 32
    assert module.MAX_SHARED_COMPARISONS == 188_160_000
    assert module.MAX_ARTIFACT_BYTES == 96_000_000
    assert module.FEATURE_CAP == 700 and module.FEATURE_CAP == module.geometry.MAX_FEATURES


def _capped_frames(case):
    return [(name, image.copy()) for name, image in case["frames"]]


def _tie_witness(left, right):
    """Exactly-capped feature set where both policies abstain on exact ties."""
    return ([[i, 1., (i + 1) % len(left), 1.] for i in range(len(left))],
            [[i, 1., (i + 1) % len(right), 1.] for i in range(len(right))])


def test_exact_feature_cap_is_allowed_and_701_refuses_before_nearest(dev_case, monkeypatch):
    size = module.FEATURE_CAP
    xy = np.array([[float(3 + (i % 25) * 7), float(3 + ((i // 25) % 25) * 7)] for i in range(size)], np.float64)
    descriptors = np.zeros((size, 128), np.float32)

    def capped_extract(image, variant):
        return xy.copy(), descriptors.copy(), size

    monkeypatch.setattr(module.ablation, "_extract", capped_extract)
    monkeypatch.setattr(module, "_nearest", _tie_witness)
    report = module.diagnose(_capped_frames(dev_case))
    assert report["detected_feature_counts"] == {"view-0": size, "view-1": size, "view-2": size}
    assert all(len(report["feature_points"][name]) == size for name in report["frame_ids"])
    assert report["shared_descriptor_comparisons"] == 2 * 3 * size * size
    assert module.canonical_report_bytes(report)
    assert all(row["reciprocal_count"] == 0 for policy in report["policies"]
               for row in policy["proposal_telemetry"])

    def over_extract(image, variant):
        return np.vstack([xy, xy[:1]]), np.vstack([descriptors, descriptors[:1]]), size + 1

    monkeypatch.setattr(module.ablation, "_extract", over_extract)
    monkeypatch.setattr(module, "_nearest", lambda *_: pytest.fail("nearest after feature-cap refusal"))
    with pytest.raises(ValueError, match="feature budget"):
        module.diagnose(_capped_frames(dev_case))
    with pytest.raises(ValueError, match="feature budget"):
        module._enforce_feature_cap(size + 1, size + 1)
    module._enforce_feature_cap(size, size)


def test_development_case_above_cap_fails_before_metrics(dev_case, monkeypatch):
    names = ["view-0", "view-1", "view-2"]
    counts = {name: module.FEATURE_CAP + 1 for name in names}
    fake_report = dict(frame_ids=names, detected_feature_counts=counts,
                       feature_points={name: [[0., 0.]] * module.FEATURE_CAP for name in names})
    monkeypatch.setattr(module, "_diagnose", lambda *_: fake_report)
    monkeypatch.setattr(module, "_case_metrics", lambda *_: pytest.fail("match-policy metrics after refusal"))
    monkeypatch.setattr(module, "synthetic_cases", lambda split="dev": iter([dev_case]))
    with pytest.raises(ValueError, match="feature budget"):
        module.calibrate_synthetic_dev()


def test_dev_only_calibration_records_counts_and_publishes_atomically(dev_case, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_DEV_CASES", 1)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    monkeypatch.setattr(module, "synthetic_cases", lambda split="dev": iter([dev_case]))
    result = module.calibrate_synthetic_dev()
    assert result["schema"] == module.CALIBRATION_SCHEMA
    assert result["evaluation_split"] == "dev" and result["heldout_run"] is False
    assert len(result["cases"]) == 1
    recorded = result["cases"][0]["frame_feature_counts"]
    assert set(recorded) == {"view-0", "view-1", "view-2"}
    assert all(0 < entry["retained"] <= entry["detected"] <= module.FEATURE_CAP for entry in recorded.values())
    assert set(result["gates"]) == {
        "family_positive_and_precision", "pooled_precision", "pure_negative_zero",
        "structural_invalid_zero", "no_a_positive_loss", "strict_recall_gain",
        "strict_partition_gain", "correct_geometry_pair", "a_not_saturated"}
    raw = module.canonical_calibration_bytes(result)

    target = tmp_path / "development.json"
    module.persist_calibration(target, result)
    assert target.read_bytes() == raw
    with pytest.raises(FileExistsError):
        module.persist_calibration(target, result)
    assert target.read_bytes() == raw

    over = copy.deepcopy(result)
    over["cases"][0]["frame_feature_counts"]["view-0"]["detected"] = module.FEATURE_CAP + 1
    reseal(over)
    with pytest.raises(ValueError, match="feature budget"):
        module.canonical_calibration_bytes(over)

    heldout = copy.deepcopy(result)
    heldout["heldout_run"] = True
    reseal(heldout)
    with pytest.raises(ValueError, match="split drift"):
        module.canonical_calibration_bytes(heldout)
    assert not (tmp_path / "extra.json").exists()
    monkeypatch.setattr(module.os, "link", lambda *_: (_ for _ in ()).throw(OSError("injected link failure")))
    with pytest.raises(OSError, match="injected link failure"):
        module.persist_calibration(tmp_path / "extra.json", result)
    assert list(tmp_path.iterdir()) == [target]
