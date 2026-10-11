"""Generated RGB only; the held-out panel is not evaluated by these tests."""
from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest

from scripts.evaluation import diagnose_multiview_matcher_ablation as ablation
from scripts.evaluation import diagnose_multiview_mnn_proposals as module


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
    monkeypatch.setattr(module, "ABLATION_SHA256", "f" * 64)
    monkeypatch.setattr(ablation, "_extract", lambda *_: pytest.fail("extraction after pin drift"))
    with pytest.raises(ValueError, match="pin drift"):
        module.diagnose(dev_case["frames"])


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
        module.calibrate_synthetic()


def test_generated_group_schedule_is_frozen_without_evaluating_heldout():
    assert len(module.SEEDS) == 8 and len(set(seed for _, _, seed in module.SEEDS)) == 8
    assert [split for split, _, _ in module.SEEDS] == ["dev"] * 4 + ["heldout"] * 4
    assert len(module.CASE_KINDS) == 8 and module.MAX_CASES == 64
    assert module.MAX_SHARED_COMPARISONS == 188_160_000
    assert module.MAX_ARTIFACT_BYTES == 96_000_000
