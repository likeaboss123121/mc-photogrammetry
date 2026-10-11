"""Generated pixels only; no ordinary screenshots or scene metadata."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.evaluation import diagnose_multiview_matcher_ablation as module


@pytest.fixture(scope="module")
def positive_frames():
    return module.make_synthetic_suite()[0]["frames"]


@pytest.fixture(scope="module")
def supported_report(positive_frames):
    return module.diagnose(positive_frames)


def reseal(report):
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = module._sha(module._canonical(payload))


def reseal_geometry(report):
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = module.geometry._digest_bytes(module.geometry._canonical(payload))


def test_baseline_arm_is_byte_identical_to_unchanged_geometric_diagnostic(positive_frames, supported_report):
    baseline = module.geometry.diagnose(positive_frames)
    assert module.geometry.canonical_report_bytes(baseline) == module.geometry.canonical_report_bytes(
        supported_report["variants"][0]["geometry"])
    assert baseline["status"] == "supported"
    assert module.configuration()["variants"][0]["ratio"] == module.geometry.RATIO
    assert module.geometry.TRANSFER_PX == 3
    assert module.geometry.MIN_HOLDOUT == 8


def test_determinism_copy_and_image_binding(positive_frames, supported_report):
    frames = [(name, image.copy()) for name, image in positive_frames]
    assert supported_report == module.diagnose(frames)
    module.verify_frame_binding(supported_report, frames)
    frames[0][1][:] = 0
    assert module.canonical_report_bytes(supported_report)
    with pytest.raises(ValueError, match="drift"):
        module.verify_frame_binding(supported_report, frames)
    with pytest.raises(TypeError):
        module.diagnose(positive_frames, camera_manifest={})


@pytest.fixture(scope="module")
def rectangular_blank_frames():
    return [(f"view-{i}", np.zeros((64, 96, 3), np.uint8)) for i in range(3)]


@pytest.fixture(scope="module")
def rectangular_blank_report(rectangular_blank_frames):
    return module.diagnose(rectangular_blank_frames)


@pytest.mark.parametrize("scope", ["one_arm", "all_arms"])
def test_resealed_same_area_arm_dimensions_refuse_before_publication(
        rectangular_blank_frames, rectangular_blank_report, scope, tmp_path, monkeypatch):
    report = copy.deepcopy(rectangular_blank_report)
    assert report["schema"] == "opencv-matcher-ablation-v2"
    assert report["frame_dimensions"] == {name: [64, 96] for name, _ in rectangular_blank_frames}
    assert module.canonical_report_bytes(report)
    module.verify_frame_binding(report, rectangular_blank_frames)
    unavailable = [row for row in report["variants"] if row["availability"] == "unavailable"]
    assert all(row["geometry"] is None for row in unavailable)
    available = [row for row in report["variants"] if row["availability"] == "available"]
    assert available
    for row in available[:1] if scope == "one_arm" else available:
        row["geometry"]["frame_dimensions"]["view-0"] = [96, 64]
        reseal_geometry(row["geometry"])
    reseal(report)
    monkeypatch.setattr(module.tempfile, "mkstemp", lambda **_: pytest.fail("temporary publication began"))
    with pytest.raises(ValueError, match="variant frame binding drift"):
        module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="variant frame binding drift"):
        module.persist_report(tmp_path / "invalid.json", report, rectangular_blank_frames)
    assert not list(tmp_path.iterdir())


def test_resealed_outer_and_all_arm_dimensions_refuse_against_source(
        rectangular_blank_frames, rectangular_blank_report, tmp_path, monkeypatch):
    report = copy.deepcopy(rectangular_blank_report)
    report["frame_dimensions"]["view-0"] = [96, 64]
    for row in report["variants"]:
        if row["availability"] == "available":
            row["geometry"]["frame_dimensions"]["view-0"] = [96, 64]
            reseal_geometry(row["geometry"])
    reseal(report)
    # A digest-only check sees consistent declarations; supplied pixels disprove them.
    assert module.canonical_report_bytes(report)
    monkeypatch.setattr(module.tempfile, "mkstemp", lambda **_: pytest.fail("temporary publication began"))
    with pytest.raises(ValueError, match="frame source drift"):
        module.verify_frame_binding(report, rectangular_blank_frames)
    with pytest.raises(ValueError, match="frame source drift"):
        module.persist_report(tmp_path / "invalid.json", report, rectangular_blank_frames)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("count", [3, 8, 9, 32])
def test_all_supported_frame_counts_and_empty_features(count):
    frames = [(f"frame-{i:02}", np.zeros((64, 64, 3), np.uint8)) for i in range(count)]
    report = module.diagnose(frames)
    assert len(report["variants"]) == module.MAX_VARIANTS
    for row in report["variants"]:
        if row["availability"] == "available":
            assert row["stage_summary"]["proposal_count"] == 0
            assert row["geometry"]["status"] == "abstained"
            assert row["geometry"]["work"]["pair_count"] == (count * (count - 1) // 2 if count <= 8 else 2 * count)
    module.verify_frame_binding(report, frames)


def test_missing_akaze_is_explicit_and_bound(monkeypatch, positive_frames):
    monkeypatch.delattr(module.cv2, "AKAZE_create", raising=False)
    report = module.diagnose(positive_frames)
    row = report["variants"][4]
    assert row == dict(id="akaze", availability="unavailable", reason="OpenCV API AKAZE_create absent",
                      feature_points=None, detected_feature_counts=None, stage_summary=None, geometry=None)
    row["reason"] = "silently substituted"
    reseal(report)
    with pytest.raises(ValueError, match="unavailable"):
        module.canonical_report_bytes(report)


@pytest.mark.parametrize("variant_index,norm,dtype", [(0, cv2.NORM_L2, np.float32),
                                                       (4, cv2.NORM_HAMMING, np.uint8),
                                                       (5, cv2.NORM_HAMMING, np.uint8)])
def test_matcher_uses_descriptor_appropriate_norm(monkeypatch, variant_index, norm, dtype):
    seen = []
    original = module.cv2.BFMatcher
    def spy(value):
        seen.append(value)
        return original(value)
    monkeypatch.setattr(module.cv2, "BFMatcher", spy)
    rng = np.random.default_rng(142)
    data = rng.integers(0, 255, (32, 32)).astype(dtype)
    record = {}
    result = module._proposals(data, data.copy(), module.VARIANTS[variant_index], record)
    assert seen == [norm]
    assert len(result) == 32
    assert record["reciprocal_count"] == 32


def test_wider_ratio_only_changes_proposals_and_cannot_skip_holdout(monkeypatch):
    # 20 perfect reciprocal proposals still fail the unchanged16-fit/8-held split.
    grid = np.array([(x, y) for x in range(30, 300, 60) for y in range(30, 300, 60)], float)[:20]
    matches = tuple((i, i) for i in range(20))
    record = {"attempts": []}
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("RANSAC called with starved fixed partition")
    monkeypatch.setattr(module.geometry.cv2, "findHomography", forbidden_fit)
    models = module.geometry._pair_models(("view-0", "view-1"), matches, grid, grid,
                                           ((320, 320), (320, 320)), record)
    assert not models
    assert record["attempts"][0]["first_rejection"] in ("fit_starved", "held_starved")


@pytest.mark.parametrize("mutation", ["config", "config_pin", "config_type", "summary_type", "variant_id", "variants_missing",
                                       "variants_reordered", "summary", "total_work", "frame_hash",
                                       "point_coordinates", "feature_count", "geometry_binding",
                                       "geometry_residual", "geometry_track_identity"])
def test_consistently_resealed_mutations_refuse_before_publication(supported_report, positive_frames, mutation, tmp_path, monkeypatch):
    report = copy.deepcopy(supported_report)
    row = report["variants"][0]
    geo = row["geometry"]
    if mutation == "config":
        report["configuration"]["variants"][0]["ratio"] = .99
        report["config_sha256"] = module._sha(module._canonical(report["configuration"]))
    elif mutation == "config_type":
        report["configuration"]["synthetic_gates"]["negative_cases_with_accepted_pair_max"] = False
    elif mutation == "summary_type":
        row["stage_summary"]["track_count"] = True
    elif mutation == "config_pin":
        report["config_sha256"] = "f" * 64
    elif mutation == "variant_id":
        row["id"] = "orb"
    elif mutation == "variants_missing":
        report["variants"].pop()
    elif mutation == "variants_reordered":
        report["variants"][0], report["variants"][1] = report["variants"][1], report["variants"][0]
    elif mutation == "summary":
        row["stage_summary"]["proposal_count"] += 1
    elif mutation == "total_work":
        report["total_descriptor_comparisons"] -= 1
    elif mutation == "frame_hash":
        report["frame_sha256"]["view-0"] = "f" * 64
    elif mutation == "point_coordinates":
        feature_index = geo["tracks"][0]["feature_points"]["view-0"][0][0]
        row["feature_points"]["view-0"][feature_index][0] += .25
    elif mutation == "feature_count":
        row["feature_points"]["view-0"].pop()
    elif mutation == "geometry_binding":
        geo["frame_sha256"]["view-0"] = "f" * 64
    elif mutation == "geometry_residual":
        attempt = geo["pair_telemetry"][0]["attempts"][0]
        attempt["held_rows"][0][1] += .5
        attempt["held_forward"] = module.geometry._residual_summary([r[1] for r in attempt["held_rows"]])
    elif mutation == "geometry_track_identity":
        geo["tracks"][0]["triangles"][0][0] += 1
    if mutation.startswith("geometry"):
        reseal_geometry(geo)
    reseal(report)
    def forbidden_temp(*args, **kwargs):
        raise AssertionError("publication began before validation")
    monkeypatch.setattr(module.tempfile, "mkstemp", forbidden_temp)
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "invalid.json", report, positive_frames)
    assert not list(tmp_path.iterdir())


def test_atomic_no_clobber_and_publication_failure_cleanup(supported_report, positive_frames, tmp_path, monkeypatch):
    target = tmp_path / "sealed.json"
    with pytest.raises(TypeError):
        module.persist_report(target, supported_report)
    assert not list(tmp_path.iterdir())
    module.persist_report(target, supported_report, positive_frames)
    original = target.read_bytes()
    assert original == module.canonical_report_bytes(supported_report)
    with pytest.raises(FileExistsError):
        module.persist_report(target, supported_report, positive_frames)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".matcher-ablation-*"))
    def failure(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(module.os, "link", failure)
    with pytest.raises(OSError, match="injected"):
        module.persist_report(tmp_path / "new.json", supported_report, positive_frames)
    assert not (tmp_path / "new.json").exists()
    assert not list(tmp_path.glob(".matcher-ablation-*"))


def test_source_pin_failure_precedes_feature_extraction(positive_frames, monkeypatch):
    monkeypatch.setattr(module, "BASELINE_SHA256", "f" * 64)
    def forbidden(*args):
        raise AssertionError("feature extraction after source drift")
    monkeypatch.setattr(module, "_extract", forbidden)
    with pytest.raises(ValueError, match="source pin drift"):
        module.diagnose(positive_frames)


@pytest.mark.parametrize("budget", ["rgb", "pair", "descriptor", "total_descriptor", "cycle", "output"])
def test_resource_gates_refuse_before_relevant_stage(positive_frames, budget, monkeypatch):
    def forbidden(*args):
        raise AssertionError("expensive stage ran after exceeded budget")
    if budget == "rgb":
        monkeypatch.setattr(module.geometry, "MAX_TOTAL_RGB_BYTES", 1)
        monkeypatch.setattr(module, "_extract", forbidden)
    elif budget == "pair":
        monkeypatch.setattr(module.geometry, "MAX_PAIRS", 1)
        monkeypatch.setattr(module, "_extract", forbidden)
    elif budget in ("descriptor", "total_descriptor"):
        if budget == "descriptor":
            monkeypatch.setattr(module.geometry, "MAX_DESCRIPTOR_COMPARISONS", 1)
        else:
            monkeypatch.setattr(module, "MAX_TOTAL_DESCRIPTOR_COMPARISONS", 1)
        monkeypatch.setattr(module, "_proposals", forbidden)
    elif budget == "cycle":
        monkeypatch.setattr(module.geometry, "MAX_CYCLE_COMBINATIONS", 0)
        monkeypatch.setattr(module.geometry, "_track", forbidden)
    else:
        monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        module.diagnose(positive_frames)


def test_invalid_input_order_count_and_array_fail_closed():
    image = np.zeros((64, 64, 3), np.uint8)
    for frames in ([('a', image), ('b', image)], [('c', image), ('b', image), ('a', image)],
                   [('a', image), ('a', image), ('b', image)], [(f"v-{i:02}", image) for i in range(33)],
                   [('a', image), ('b', image), ('c', image.astype(float))]):
        with pytest.raises(ValueError):
            module.diagnose(frames)


def test_synthetic_suite_and_frozen_promotion_metrics():
    first = module.calibrate_synthetic()
    second = module.calibrate_synthetic()
    assert first == second
    assert first["configuration"] == module.configuration()
    assert first["config_sha256"] == module.CONFIG_SHA256
    assert len(first["cases"]) == 6
    assert [c["kind"] for c in first["cases"]].count("positive") == 4
    assert [c["kind"] for c in first["cases"]].count("negative") == 2
    for row in first["rankings"]:
        assert 0 <= row["proposal_recall"] <= 1
        assert 0 <= row["accepted_precision"] <= 1
        if row["passing"]:
            assert row["available"] and row["positive_supported"] >= 3
            assert row["accepted_precision"] >= .95 and row["negative_supported"] == 0
    chosen = first["chosen_variant"]
    assert chosen is None or next(row for row in first["rankings"] if row["id"] == chosen)["passing"]
    payload = dict(first)
    digest = payload.pop("canonical_sha256")
    assert module._sha(module._canonical(payload)) == digest


def test_oracle_recall_denominator_and_wrong_repeat_identity_are_explicit():
    points = {"view-0": [[10., 10.], [42., 10.]], "view-1": [[10., 10.], [42., 10.]]}
    transforms = [np.eye(3), np.eye(3)]
    assert module._oracle_pair(points, ["view-0", "view-1"], transforms, [[0, 0], [1, 1]]) == (2, 2)
    assert module._oracle_pair(points, ["view-0", "view-1"], transforms, [[0, 1], [1, 0]]) == (0, 2)


@pytest.mark.parametrize("kind", ["sift", "orb"])
def test_feature_caps_are_explicit_and_selection_deterministic(kind, monkeypatch):
    keys = [cv2.KeyPoint(float(i % 60 + 1), float(i // 60 + 1), 1., response=float(i)) for i in range(701)]
    dtype = np.float32 if kind == "sift" else np.uint8
    descriptors = np.zeros((701, 128 if kind == "sift" else 32), dtype=dtype)
    class Detector:
        def detectAndCompute(self, *args):
            return keys, descriptors
    if kind == "sift":
        monkeypatch.setattr(module.cv2, "SIFT_create", lambda **kwargs: Detector())
        with pytest.raises(ValueError, match="feature budget"):
            module._extract(np.zeros((64, 64, 3), np.uint8), module.VARIANTS[1])
    else:
        monkeypatch.setattr(module.cv2, "ORB_create", lambda **kwargs: Detector())
        xy, desc, detected = module._extract(np.zeros((64, 64, 3), np.uint8), module.VARIANTS[5])
        assert detected == 701 and len(xy) == len(desc) == 700
        assert np.array_equal(xy[0], keys[700].pt)
        assert np.array_equal(xy[-1], keys[1].pt)


def test_in_memory_preregistration_drift_fails_before_extraction(positive_frames, monkeypatch):
    monkeypatch.setitem(module.GATES, "accepted_precision_min", .1)
    with pytest.raises(ValueError, match="configuration pin drift"):
        module.diagnose(positive_frames)
