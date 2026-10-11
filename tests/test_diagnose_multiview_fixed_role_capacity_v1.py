"""Record-level synthetic evidence only; inference and calibration are forbidden."""
from __future__ import annotations

import copy

import pytest

from scripts.evaluation import diagnose_multiview_fixed_role_capacity_v1 as module


@pytest.fixture(autouse=True)
def forbid_inference(monkeypatch):
    monkeypatch.setattr(module.evaluator, "diagnose",
                        lambda *_: pytest.fail("inference during source-prep tests"))


def _fixture(n=32):
    pair = ["view-0", "view-1"]
    proposals = [[i, i] for i in range(n)]
    fit, held, mode = module.geometry._quota_partition(tuple(pair), tuple(map(tuple, proposals)))
    points = [[20. + 20. * (i % 8), 25. + 25. * (i // 8)] for i in range(n)]
    xy = {name: points for name in pair}
    dimensions = {name: [192, 192] for name in pair}
    first = dict(index=0, remaining_indices=list(range(n)), partition_mode=mode,
                 fit_indices=list(fit), fit_count=len(fit), held_indices=list(held),
                 held_count=len(held), fit_inlier_indices=list(fit[:16]),
                 held_rows=[[i, 4., 4.] for i in held],
                 first_rejection="joint_holdout_starved")
    geo = dict(frame_ids=pair, attempts=[first], accepted_model_count=0)
    return geo, proposals, xy, dimensions, list(fit), list(held)


def _truth(proposals, true_ids, surface="plane"):
    return [[*proposal, surface if i in true_ids else None]
            for i, proposal in enumerate(proposals)]


def _record(geo, proposals, truth, surfaces, points, dimensions):
    analysis = module.analyze_pair(geo, proposals, truth, surfaces, points, dimensions)
    row = dict(schema=module.SCHEMA, config_sha256=module.CONFIG_SHA256,
               geo_row=geo, proposals=proposals, truth_rows=truth, surfaces=surfaces,
               points=points, dimensions=dimensions, analysis=analysis)
    row["canonical_sha256"] = module._sha(module._canonical(row))
    return row


def _reseal(row):
    payload = dict(row)
    payload.pop("canonical_sha256", None)
    row["canonical_sha256"] = module._sha(module._canonical(payload))


def test_identical_raw_totals_distinguish_true_fixed_role_capacity():
    geo, proposals, points, dims, fit, held = _fixture()
    sufficient = module.analyze_pair(geo, proposals,
                                     _truth(proposals, set(fit[:16] + held)),
                                     ["plane"], points, dims)
    limited = module.analyze_pair(geo, proposals,
                                  _truth(proposals, set(fit[:20] + held[:4])),
                                  ["plane"], points, dims)
    assert sufficient["proposal_count"] == limited["proposal_count"] == 32
    assert sufficient["surface_capacity"][0]["true_fit_count"] == 16
    assert sufficient["surface_capacity"][0]["true_held_count"] == 8
    assert sufficient["surface_capacity"][0]["capacity_sufficient"]
    assert sufficient["classification"] == "model_or_selection_limited"
    assert limited["surface_capacity"][0]["true_fit_count"] == 20
    assert limited["surface_capacity"][0]["true_held_count"] == 4
    assert not limited["surface_capacity"][0]["capacity_sufficient"]
    assert limited["classification"] == "supply_limited"


def test_mixed_accepted_model_and_two_separate_homogeneous_models():
    geo, proposals, points, dims, fit, held = _fixture()
    geo["attempts"][0]["fit_inlier_indices"] = fit
    geo["attempts"][0]["held_rows"] = [[i, 0., 0.] for i in held]
    geo["attempts"][0]["first_rejection"] = "accepted"
    geo["accepted_model_count"] = 1
    labels = [[*p, "a" if i in set(fit[:12] + held[:4]) else "b"]
              for i, p in enumerate(proposals)]
    mixed = module.analyze_pair(geo, proposals, labels, ["a", "b"], points, dims)
    assert len(mixed["accepted_models"]) == 1
    assert mixed["accepted_models"][0]["mixed_surface"]
    assert mixed["accepted_models"][0]["invalid_match_count"] == 0
    assert mixed["classification"] == "unresolved"
    assert mixed["findings"] == ["supply_limited", "model_or_selection_limited"]

    geo, proposals, points, dims, fit, held = _fixture(80)
    first_fit, second_fit = fit[:16], fit[16:32]
    first_held, second_held = held[:8], held[8:16]
    assert len(second_held) == 8
    first = geo["attempts"][0]
    first.update(fit_inlier_indices=first_fit,
                 held_rows=[[i, 0., 0.] if i in first_held else [i, 4., 4.] for i in held],
                 first_rejection="accepted")
    remaining = [i for i in range(80) if i not in set(first_fit + first_held)]
    second = dict(index=1, remaining_indices=remaining, partition_mode=first["partition_mode"],
                  fit_indices=[i for i in fit if i in remaining],
                  held_indices=[i for i in held if i in remaining],
                  fit_inlier_indices=second_fit,
                  held_rows=[[i, 0., 0.] if i in second_held else [i, 4., 4.]
                             for i in held if i in remaining],
                  first_rejection="accepted")
    second["fit_count"] = len(second["fit_indices"])
    second["held_count"] = len(second["held_indices"])
    geo["attempts"] = [first, second]
    geo["accepted_model_count"] = 2
    truth = [[*proposal, "a" if i in set(first_fit + first_held) else
              "b" if i in set(second_fit + second_held) else None]
             for i, proposal in enumerate(proposals)]
    separate = module.analyze_pair(geo, proposals, truth, ["a", "b"], points, dims)
    assert len(separate["accepted_models"]) == 2
    assert all(not model["mixed_surface"] and model["invalid_match_count"] == 0
               for model in separate["accepted_models"])
    assert separate["accepted_models"][0]["surface_counts"]["a"] == 24
    assert separate["accepted_models"][1]["surface_counts"]["b"] == 24


@pytest.mark.parametrize("mutation", ["unknown_identity", "duplicate_identity", "unknown_surface", "role"])
def test_unknown_duplicate_and_mutated_roles_refuse(mutation):
    geo, proposals, points, dims, fit, held = _fixture()
    truth = _truth(proposals, set(fit[:16] + held))
    if mutation == "unknown_identity":
        truth[0][:2] = [999, 999]
    elif mutation == "duplicate_identity":
        truth[1] = truth[0].copy()
    elif mutation == "unknown_surface":
        truth[0][2] = "other"
    else:
        geo["attempts"][0]["fit_indices"][0] = held[0]
    with pytest.raises(ValueError):
        module.analyze_pair(geo, proposals, truth, ["plane"], points, dims)


def test_resealed_pair_analysis_tamper_refuses():
    geo, proposals, points, dims, fit, held = _fixture()
    row = _record(geo, proposals, _truth(proposals, set(fit[:16] + held)),
                  ["plane"], points, dims)
    assert module.canonical_pair_record_bytes(row)
    mutated = copy.deepcopy(row)
    mutated["analysis"]["surface_capacity"][0]["true_held_count"] += 1
    _reseal(mutated)
    with pytest.raises(ValueError, match="resealed pair capacity drift"):
        module.canonical_pair_record_bytes(mutated)


def test_label_access_only_after_seal_and_annotation_on_off_parity(monkeypatch):
    report = {"opaque": [1, 2, 3]}
    raw = module._canonical(report)
    seen = []
    def canonical(value):
        seen.append("seal")
        assert value is report
        return raw
    def labels():
        seen.append("labels")
        return ["plane"]
    monkeypatch.setattr(module.evaluator, "canonical_report_bytes", canonical)
    without, _ = module._postseal_truth(report, module._sha(raw), lambda: None)
    with_labels, value = module._postseal_truth(report, module._sha(raw), labels)
    assert without == with_labels == raw and value == ["plane"]
    assert seen == ["seal", "seal", "seal", "labels", "seal"]
    monkeypatch.setattr(module.evaluator, "canonical_report_bytes", module._canonical)
    with pytest.raises(ValueError, match="annotation changed sealed"):
        module._postseal_truth(report, module._sha(module._canonical(report)),
                               lambda: report["opaque"].append(4))
    monkeypatch.setattr(module.evaluator, "canonical_report_bytes",
                        lambda *_: (_ for _ in ()).throw(ValueError("bad seal")))
    with pytest.raises(ValueError, match="bad seal"):
        module._postseal_truth(report, module._sha(raw),
                               lambda: pytest.fail("label access before seal"))


def test_exact_frozen_roster_and_existing_positive_corner_controls():
    baseline = module._sealed_baseline()
    assert set(baseline) == set(module.ROSTER) and len(baseline) == 4
    sparse = baseline[("dev-sparse-corners-96141", "viewpoint")]
    corner = baseline[("dev-mixed-scale-pixel-96131", "physical-corner")]
    def arm(row):
        return next(p for p in row["policies"] if p["id"] == module.POLICY)["geometry_arms"][module.ARM]
    assert [p["accepted_model_count"] for p in arm(sparse)["pairs"]] == [1, 1, 1]
    assert arm(sparse)["valid_tracks"] == 1 and arm(sparse)["invalid_tracks"] == 0
    assert arm(corner)["invalid_tracks"] == 1 and arm(corner)["invalid_structural_support"]
    wrong = dict(schema=module.SCHEMA, configuration=module.configuration(),
                 config_sha256=module.CONFIG_SHA256, cases=[], work={})
    _reseal(wrong)
    with pytest.raises(ValueError, match="roster drift"):
        module.canonical_annotation_bytes(wrong)


def test_work_and_output_caps_refuse_and_atomic_no_clobber(monkeypatch, tmp_path):
    pair = {"attempts": [{"index": 0}]}
    geo = {"pair_telemetry": [pair, pair, pair]}
    policy = {"id": module.POLICY, "geometry_arms": {module.ARM: geo}}
    rows = [{"inference_report": {"shared_descriptor_comparisons": 4,
                                   "policies": [policy]}} for _ in range(4)]
    assert module._work_counts(rows)["designated_pair_slots"] == 12
    monkeypatch.setattr(module, "MAX_SHARED_COMPARISONS", 1)
    with pytest.raises(ValueError, match="work cap"):
        module._work_counts(rows)
    monkeypatch.undo()
    geo, proposals, points, dims, fit, held = _fixture()
    record = _record(geo, proposals, _truth(proposals, set(fit[:16] + held)),
                     ["plane"], points, dims)
    monkeypatch.setattr(module, "MAX_RETAINED_ANNOTATION_BYTES", 1)
    with pytest.raises(ValueError, match="byte cap"):
        module.canonical_pair_record_bytes(record)
    monkeypatch.undo()
    monkeypatch.setattr(module, "MAX_FINAL_ANNOTATION_BYTES", 1)
    with pytest.raises(ValueError, match="final annotation byte cap"):
        module._enforce_final_cap(b"12")
    monkeypatch.undo()
    monkeypatch.setattr(module, "canonical_annotation_bytes", lambda _: b"sealed-synthetic-record")
    path = tmp_path / "annotation.json"
    module.persist_annotation(path, {})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.persist_annotation(path, {})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def _synthetic_four_case_annotation(monkeypatch, *, accepted=False):
    """A source-free complete replay envelope with independently sealed pair totals."""
    geo, proposals, points, dims, fit, held = _fixture()
    if accepted:
        geo["attempts"][0]["held_rows"] = [[i, 0., 0.] for i in held]
        geo["attempts"][0]["first_rejection"] = "accepted"
        geo["accepted_model_count"] = 1
    truth = _truth(proposals, set(fit[:16] + held))
    analysis = module.analyze_pair(geo, proposals, truth, ["plane"], points, dims)
    sealed_pair = dict(true_proposals=analysis["true_proposal_count"],
                       false_proposals=analysis["false_proposal_count"],
                       true_accepted_support=analysis["true_accepted_support_count"],
                       accepted_support=analysis["accepted_support_count"],
                       reciprocal_count=analysis["proposal_count"],
                       first_rejections=[analysis["first_rejection"]],
                       fit_count=analysis["fit_count"], held_count=analysis["held_count"])
    policy = dict(id=module.POLICY,
                  proposal_telemetry=[dict(frame_ids=geo["frame_ids"],
                                           reciprocal_pairs=proposals) for _ in range(3)],
                  geometry_arms={module.ARM: dict(pair_telemetry=[geo] * 3)})
    report = dict(policies=[policy], frame_sha256={"view-0": "a", "view-1": "b"},
                  frame_dimensions=dims, feature_points=points,
                  shared_descriptor_comparisons=0)
    report_sha = module._sha(module._canonical(report))
    sealed = {}
    rows = []
    for group, kind in module.ROSTER:
        metric = dict(input_sha256=report["frame_sha256"], report_sha256=report_sha,
                      policies=[dict(id=module.POLICY,
                                     geometry_arms={module.ARM: dict(pairs=[sealed_pair] * 3)})])
        sealed[(group, kind)] = metric
        rows.append(dict(group=group, kind=kind,
                         input_sha256=report["frame_sha256"],
                         inference_report_sha256=report_sha,
                         case_metrics_sha256=module._sha(module._canonical(metric)),
                         inference_report=report, surfaces=["plane"],
                         truth_rows=[dict(frame_ids=geo["frame_ids"], truth_rows=copy.deepcopy(truth))
                                     for _ in range(3)],
                         pair_annotations=[copy.deepcopy(analysis) for _ in range(3)]))
    monkeypatch.setattr(module, "_sealed_baseline", lambda: sealed)
    monkeypatch.setattr(module.evaluator, "canonical_report_bytes", module._canonical)
    result = dict(schema=module.SCHEMA, configuration=module.configuration(),
                  config_sha256=module.CONFIG_SHA256, cases=rows,
                  work=module._work_counts(rows))
    _reseal(result)
    return result


def test_cumulative_retained_cap_in_build_and_complete_replay(monkeypatch):
    result = _synthetic_four_case_annotation(monkeypatch)
    rows = result["cases"]
    per_row = max(len(module._canonical(row)) for row in rows)
    monkeypatch.setattr(module, "MAX_RETAINED_ANNOTATION_BYTES", per_row)
    monkeypatch.setattr(module, "CONFIG_SHA256", module._sha(module._canonical(module.configuration())))
    result["configuration"] = module.configuration()
    result["config_sha256"] = module.CONFIG_SHA256
    _reseal(result)
    assert module._enforce_retained_cap([rows[0]]) <= per_row
    with pytest.raises(ValueError, match="cumulative retained annotation byte cap"):
        module._enforce_retained_cap(rows)
    with pytest.raises(ValueError, match="cumulative retained annotation byte cap"):
        module.canonical_annotation_bytes(result)


def test_disconnected_true_outliers_do_not_prove_supply_shortage():
    geo, proposals, points, dims, fit, held = _fixture(40)
    passing = fit[:16] + held
    outliers = fit[16:20]
    for ordinal, i in enumerate(passing):
        xy = [25. + 20. * (ordinal % 6), 25. + 20. * (ordinal // 6)]
        for frame in geo["frame_ids"]:
            points[frame][i] = xy
    for i in outliers:
        for frame in geo["frame_ids"]:
            points[frame][i] = [185., 185.]
    assert module._coverage([points["view-0"][i] for i in passing], dims["view-0"])["passed"]
    analysis = module.analyze_pair(geo, proposals,
                                   _truth(proposals, set(passing + outliers)),
                                   ["plane"], points, dims)
    surface = analysis["surface_capacity"][0]
    assert surface["true_fit_count"] == 20 and surface["true_held_count"] == 8
    assert surface["role_count_sufficient"]
    assert not surface["capacity_sufficient"]
    assert analysis["classification"] == "unresolved"
    assert analysis["findings"] == ["coverage_subset_unassessed"]


def test_resealed_truth_and_analysis_still_refuse_sealed_pair_aggregate(monkeypatch):
    result = _synthetic_four_case_annotation(monkeypatch)
    assert module.canonical_annotation_bytes(result)
    mutated = copy.deepcopy(result)
    row = mutated["cases"][0]
    geo = row["inference_report"]["policies"][0]["geometry_arms"][module.ARM]["pair_telemetry"][0]
    proposals = row["inference_report"]["policies"][0]["proposal_telemetry"][0]["reciprocal_pairs"]
    row["truth_rows"][0]["truth_rows"][0][2] = None
    row["pair_annotations"][0] = module.analyze_pair(
        geo, proposals, row["truth_rows"][0]["truth_rows"], row["surfaces"],
        row["inference_report"]["feature_points"],
        row["inference_report"]["frame_dimensions"])
    _reseal(mutated)
    with pytest.raises(ValueError, match="truth ledger contradicts sealed per-pair aggregates"):
        module.canonical_annotation_bytes(mutated)


def test_same_total_truth_substitution_changes_sealed_accepted_support(monkeypatch):
    result = _synthetic_four_case_annotation(monkeypatch, accepted=True)
    assert module.canonical_annotation_bytes(result)
    mutated = copy.deepcopy(result)
    row = mutated["cases"][0]
    policy = row["inference_report"]["policies"][0]
    geo = policy["geometry_arms"][module.ARM]["pair_telemetry"][0]
    proposals = policy["proposal_telemetry"][0]["reciprocal_pairs"]
    fit = geo["attempts"][0]["fit_indices"]
    assert row["truth_rows"][0]["truth_rows"][fit[0]][2] == "plane"
    assert row["truth_rows"][0]["truth_rows"][fit[16]][2] is None
    row["truth_rows"][0]["truth_rows"][fit[0]][2] = None
    row["truth_rows"][0]["truth_rows"][fit[16]][2] = "plane"
    old = row["pair_annotations"][0]
    row["pair_annotations"][0] = module.analyze_pair(
        geo, proposals, row["truth_rows"][0]["truth_rows"], row["surfaces"],
        row["inference_report"]["feature_points"],
        row["inference_report"]["frame_dimensions"])
    assert row["pair_annotations"][0]["true_proposal_count"] == old["true_proposal_count"]
    assert (row["pair_annotations"][0]["true_accepted_support_count"] ==
            old["true_accepted_support_count"] - 1)
    _reseal(mutated)
    with pytest.raises(ValueError, match="truth ledger contradicts sealed per-pair aggregates"):
        module.canonical_annotation_bytes(mutated)
