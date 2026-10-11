"""Generated RGB and evaluator-only physical labels; no held-out/image inputs."""
from __future__ import annotations

import copy
import hashlib
import inspect

import numpy as np
import pytest

from scripts.evaluation import diagnose_multiview_mnn_quota_v2 as module
from scripts.evaluation import diagnose_multiview_mnn_quota_v1 as v1
from scripts.evaluation import diagnose_multiview_mnn_proposals as prior_v1
from scripts.evaluation import diagnose_multiview_mnn_proposals_v2 as prior_v2


@pytest.fixture(scope="module")
def dev_case():
    return next(module.synthetic_cases("dev"))


@pytest.fixture(scope="module")
def dev_report(dev_case):
    return module.diagnose(dev_case["frames"])


def _reseal(report, engine=module):
    payload = dict(report)
    payload.pop("canonical_sha256", None)
    report["canonical_sha256"] = engine._sha(engine._canonical(payload)) if engine is module else (
        engine._digest_bytes(engine._canonical(payload)))


def test_paired_arms_share_exact_features_proposals_and_inputs(dev_case, dev_report):
    assert list(inspect.signature(module.diagnose).parameters) == ["frames"]
    assert dev_report == module.diagnose([(n, image.copy()) for n, image in dev_case["frames"]])
    assert module.canonical_report_bytes(dev_report)
    assert dev_report["schema"] == module.SCHEMA == "opencv-mnn-quota-proposals-v2"
    assert module.CALIBRATION_SCHEMA == "opencv-mnn-quota-synthetic-calibration-v2"
    assert module.CONFIG_SHA256 != v1.CONFIG_SHA256
    module.verify_frame_binding(dev_report, dev_case["frames"])
    assert module.GEOMETRY_ARMS == ("legacy-v4", "identity-sha256-held-quota-v1")
    for policy in dev_report["policies"]:
        legacy = policy["geometry_arms"][module.GEOMETRY_ARMS[0]]
        quota = policy["geometry_arms"][module.GEOMETRY_ARMS[1]]
        assert module.base_geometry.canonical_report_bytes(legacy)
        assert module.geometry.canonical_report_bytes(quota)
        assert legacy["frame_sha256"] == quota["frame_sha256"] == dev_report["frame_sha256"]
        assert legacy["work"]["feature_counts"] == quota["work"]["feature_counts"]
        assert [row["reciprocal_pairs"] for row in legacy["pair_telemetry"]] == (
            [row["reciprocal_pairs"] for row in quota["pair_telemetry"]])
        assert quota["partition_policy"]["config_sha256"] == module.geometry.QUOTA_CONFIG_SHA256


def test_physical_corner_has_parallax_visibility_and_equal_motion_control():
    corner = next(c for c in module.synthetic_cases("dev") if c["kind"] == "physical-corner")
    oracle = corner["oracle"]
    assert oracle["kind"] == "structural" and oracle["mode"] == "physical-corner"
    assert len({tuple(c) for c in oracle["camera_centers"]}) == 3
    assert all((mask == 1).sum() > 5000 and (mask == 2).sum() > 5000
               for mask in oracle["visibility"])
    def relative(surface):
        hs = oracle["transforms"][surface]
        result = hs[1] @ np.linalg.inv(hs[0])
        return result / result[2, 2]
    assert np.linalg.norm(relative("face-a") - relative("face-b")) > 1.0
    for i, mask in enumerate(oracle["visibility"]):
        for surface, code in (("face-a", 1), ("face-b", 2)):
            y, x = np.argwhere(mask == code)[len(np.argwhere(mask == code)) // 2]
            label = module._label(oracle, f"view-{i}", [float(x), float(y)])
            assert label is not None and label[0] == surface
    equal = next(c for c in module.synthetic_cases("dev") if c["kind"] == "equal-motion-unidentifiable")
    assert equal["oracle"]["kind"] == "unidentifiable"
    h = equal["oracle"]["transforms"]
    a = h["face-a"][1] @ np.linalg.inv(h["face-a"][0])
    b = h["face-b"][1] @ np.linalg.inv(h["face-b"][0])
    assert np.allclose(a / a[2, 2], b / b[2, 2])


@pytest.mark.parametrize("kind", ["occlusion", "incompatible-tiles", "physical-corner",
                                   "disconnected-fragments", "equal-motion-unidentifiable"])
def test_synthetic_falsifier_panel_has_no_false_accepted_support(kind):
    case = next(c for c in module.synthetic_cases("dev")
                if c["family"] == "weak-contrast" and c["kind"] == kind)
    metrics = module._case_metrics(case, module.diagnose(case["frames"]))
    for policy in metrics["policies"]:
        for arm in module.GEOMETRY_ARMS:
            result = policy["geometry_arms"][arm]
            assert result["invalid_tracks"] == 0
            assert not result["invalid_structural_support"]
            if case["oracle"]["kind"] == "pure-negative":
                assert result["accepted_pair_count"] == 0
    if kind == "equal-motion-unidentifiable":
        assert metrics["label_kind"] == "unidentifiable"


@pytest.mark.parametrize("mutation", ["proposal", "quota_partition", "quota_policy", "frame_dimension"])
def test_resealed_cross_arm_mutation_refused(dev_report, mutation):
    report = copy.deepcopy(dev_report)
    quota = report["policies"][1]["geometry_arms"][module.GEOMETRY_ARMS[1]]
    if mutation == "proposal":
        quota["pair_telemetry"][0]["reciprocal_pairs"][0][0] += 1
    elif mutation == "quota_partition":
        first = quota["pair_telemetry"][0]["attempts"][0]
        first["fit_indices"], first["held_indices"] = first["held_indices"], first["fit_indices"]
        first["fit_count"], first["held_count"] = first["held_count"], first["fit_count"]
    elif mutation == "quota_policy":
        quota["partition_policy"]["config_sha256"] = "0" * 64
    else:
        quota["frame_dimensions"]["view-0"] = [192, 193]
    _reseal(quota, module.geometry)
    _reseal(report)
    with pytest.raises(ValueError):
        module.canonical_report_bytes(report)


def test_hard_caps_source_pins_and_dev_only(monkeypatch, dev_case):
    assert module.FEATURE_CAP == module.geometry.MAX_FEATURES == module.base_geometry.MAX_FEATURES == 700
    assert module.MAX_ARTIFACT_BYTES == 96_000_000
    assert len(module.SEEDS) == 4 and all(split == "dev" for split, _, _ in module.SEEDS)
    assert tuple(seed for _, _, seed in module.SEEDS) == (96111, 96121, 96131, 96141)
    assert not set(seed for _, _, seed in module.SEEDS) & set(seed for _, _, seed in v1.SEEDS)
    assert not set(seed for _, _, seed in module.SEEDS) & {
        seed for prior in (prior_v1, prior_v2) for _, _, seed in prior.SEEDS}
    assert len(module.CASE_KINDS) == 9 and module.MAX_DEV_CASES == 36
    with pytest.raises(ValueError, match="not authorized"):
        next(module.synthetic_cases("heldout"))
    assert hashlib.sha256(open(module.geometry.__file__, "rb").read()).hexdigest() == module.GEOMETRY_SHA256
    assert hashlib.sha256(open(module.base_geometry.__file__, "rb").read()).hexdigest() == module.BASE_GEOMETRY_SHA256
    assert hashlib.sha256(open(v1.__file__, "rb").read()).hexdigest() == module.V1_EVALUATOR_SHA256
    monkeypatch.setattr(module, "GEOMETRY_SHA256", "0" * 64)
    monkeypatch.setattr(module, "_extract", lambda *_: pytest.fail("extraction after pin drift"))
    with pytest.raises(ValueError, match="pin drift"):
        module.diagnose(dev_case["frames"])


def test_source_pin_and_historical_v1_files_are_immutable():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    expected = {
        "scripts/evaluation/diagnose_multiview_mnn_quota_v1.py": module.V1_EVALUATOR_SHA256,
        "scripts/evaluation/diagnose_multiview_plane_tracks_quota_v1.py": module.GEOMETRY_SHA256,
        "tests/test_diagnose_multiview_mnn_quota_v1.py":
            "a4ac6ac2bcb952879651111c5b866b958eb3b82fa039be040710156615ad8195",
        "docs/reports/SOL-MULTIVIEW-QUOTA-PARTITION-V1-2026-10-08/RUN.json":
            "7a337efffb5b65fcf6c7e463b5317a924120e0484b2ac513a0a1b3ce56d8fcc8",
        "docs/reports/SOL-MULTIVIEW-QUOTA-PARTITION-V1-2026-10-08/RESULT.json":
            "5baf3570b579e1a0a29a5ef66e8236edd60a1ec84068ac955e2ca2a585c06296",
    }
    assert {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in expected} == expected


def test_feature_work_and_output_caps_refuse_before_downstream_work(dev_case, monkeypatch):
    monkeypatch.setattr(module.ablation, "_extract",
                        lambda *_: (np.zeros((701, 2)), np.zeros((701, 128), np.float32), 701))
    monkeypatch.setattr(module, "_nearest", lambda *_: pytest.fail("nearest after feature refusal"))
    with pytest.raises(ValueError, match="feature budget"):
        module.diagnose(dev_case["frames"])
    monkeypatch.undo()
    monkeypatch.setattr(module.geometry, "MAX_DESCRIPTOR_COMPARISONS", 1)
    monkeypatch.setattr(module, "_nearest", lambda *_: pytest.fail("nearest after work refusal"))
    with pytest.raises(ValueError, match="descriptor work budget"):
        module.diagnose(dev_case["frames"])
    monkeypatch.undo()
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 1)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    with pytest.raises(ValueError, match="output budget"):
        module.diagnose(dev_case["frames"])


def test_atomic_source_bound_publication(dev_case, dev_report, tmp_path):
    target = tmp_path / "paired.json"
    module.persist_report(target, dev_report, dev_case["frames"])
    original = target.read_bytes()
    assert original == module.canonical_report_bytes(dev_report)
    with pytest.raises(FileExistsError):
        module.persist_report(target, dev_report, dev_case["frames"])
    assert target.read_bytes() == original
    altered = [(name, image.copy()) for name, image in dev_case["frames"]]
    altered[0][1][0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="source drift"):
        module.persist_report(tmp_path / "invalid.json", dev_report, altered)
    assert list(tmp_path.iterdir()) == [target]


def test_resealed_duplicate_missing_dev_case_roster_refused(monkeypatch, dev_case, dev_report):
    first = module._case_metrics(dev_case, dev_report)
    monkeypatch.setattr(module, "SEEDS", (("dev", "weak-contrast", 96111),))
    monkeypatch.setattr(module, "CASE_KINDS", ("viewpoint", "brightness"))
    monkeypatch.setattr(module, "MAX_DEV_CASES", 2)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    second = copy.deepcopy(first)
    second["kind"] = "brightness"
    cases = [first, second]
    pooled = {policy: {arm: module._aggregate(cases, i, arm) for arm in module.GEOMETRY_ARMS}
              for i, policy in enumerate(module.POLICIES)}
    aggregates = dict(dev=dict(pooled=pooled, families={"weak-contrast": pooled}))
    gates = module._dev_gates(cases, aggregates)
    result = dict(schema=module.CALIBRATION_SCHEMA, configuration=module.configuration(),
                  config_sha256=module.CONFIG_SHA256, evaluation_split="dev", heldout_run=False,
                  cases=cases, aggregates=aggregates,
                  actual_shared_descriptor_comparisons=sum(c["shared_descriptor_comparisons"] for c in cases),
                  gates=gates, decision="go" if all(gates.values()) else "no-go",
                  selection_basis="synthetic validator control")
    _reseal(result)
    assert module.canonical_calibration_bytes(result)
    result["cases"][1] = copy.deepcopy(result["cases"][0])
    _reseal(result)
    with pytest.raises(ValueError, match="roster drift"):
        module.canonical_calibration_bytes(result)
