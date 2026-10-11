"""Post-seal synthetic fit/held capacity attribution; never changes inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from scripts.evaluation import diagnose_multiview_mnn_quota_v2 as evaluator
from scripts.evaluation import diagnose_multiview_plane_tracks_quota_v1 as geometry


SCHEMA = "crossview-fixed-role-capacity-v1"
EVALUATOR_SHA256 = "8d0136b27b2fc0603d5efdeab31c2b3562165502d74cce3c65577f10892efda6"
GEOMETRY_SHA256 = "f1b6b28d7df7bb8c4122fbe5d7cc1749bb0cba33cf2f7230c6ce11f7fd26c836"
BASELINE_SHA256 = "aa410761b433a3026261d10ae302184613a1b3152d3e5b8ccb892f2c30892f6b"
BASELINE_PATH = (Path(__file__).resolve().parents[2] / "docs/reports" /
                 "SOL-MULTIVIEW-QUOTA-PARTITION-V2-CALIBRATION-2026-10-08/development.json")
POLICY = "strict-unique-mutual-nearest"
ARM = "identity-sha256-held-quota-v1"
ROSTER = (("dev-weak-contrast-96111", "viewpoint"),
          ("dev-mixed-scale-pixel-96131", "viewpoint"),
          ("dev-sparse-corners-96141", "viewpoint"),
          ("dev-mixed-scale-pixel-96131", "physical-corner"))
MAX_CASES = 4
MAX_PAIR_SLOTS = 12
MAX_ARM_ATTEMPTS = 36
MAX_ALL_ARM_ATTEMPTS = 144
MAX_ARM_RANSAC_ITERATIONS = 18_000
MAX_ALL_ARM_RANSAC_ITERATIONS = 72_000
MAX_SHARED_COMPARISONS = 11_760_000
MAX_RETAINED_ANNOTATION_BYTES = 8 * 1024 * 1024
MAX_FINAL_ANNOTATION_BYTES = 4 * 1024 * 1024
RANSAC_ITERATIONS_PER_ATTEMPT = 500


def _canonical(value: object) -> bytes:
    return evaluator._canonical(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def configuration() -> dict:
    return dict(schema=SCHEMA, evaluator_sha256=EVALUATOR_SHA256,
                evaluator_config_sha256=evaluator.CONFIG_SHA256,
                geometry_sha256=GEOMETRY_SHA256, baseline_sha256=BASELINE_SHA256,
                policy=POLICY, arm=ARM, roster=[list(row) for row in ROSTER],
                capacity_rule="same-surface true fit>=16, held>=8, true-held/all-held>=0.38, both role-capacity coverages",
                classification_rule="count/fraction shortage is supply_limited; count-sufficient union coverage failure is unresolved coverage_subset_unassessed",
                aggregate_parity="true proposal and accepted-support totals must equal sealed per-pair metrics",
                retained_rule="sum canonical bytes of all four retained case rows <=8 MiB",
                transfer_px=geometry.TRANSFER_PX, fit_min=16, held_min=geometry.MIN_HOLDOUT,
                max_cases=MAX_CASES, max_pair_slots=MAX_PAIR_SLOTS,
                max_arm_attempts=MAX_ARM_ATTEMPTS, max_all_arm_attempts=MAX_ALL_ARM_ATTEMPTS,
                max_arm_ransac_iterations=MAX_ARM_RANSAC_ITERATIONS,
                max_all_arm_ransac_iterations=MAX_ALL_ARM_RANSAC_ITERATIONS,
                max_shared_comparisons=MAX_SHARED_COMPARISONS,
                max_retained_annotation_bytes=MAX_RETAINED_ANNOTATION_BYTES,
                max_final_annotation_bytes=MAX_FINAL_ANNOTATION_BYTES)


CONFIG_SHA256 = _sha(_canonical(configuration()))


def _sealed_baseline() -> dict[tuple[str, str], dict]:
    evaluator._check_pins()
    if (_sha(Path(evaluator.__file__).read_bytes()) != EVALUATOR_SHA256 or
            _sha(Path(geometry.__file__).read_bytes()) != GEOMETRY_SHA256 or
            _sha(_canonical(configuration())) != CONFIG_SHA256):
        raise ValueError("capacity source or configuration pin drift")
    raw = BASELINE_PATH.read_bytes()
    if _sha(raw) != BASELINE_SHA256:
        raise ValueError("sealed development report pin drift")
    report = json.loads(raw)
    if evaluator.canonical_calibration_bytes(report) != raw:
        raise ValueError("sealed development report replay drift")
    selected = {(row["group"], row["kind"]): row for row in report["cases"]
                if (row["group"], row["kind"]) in ROSTER}
    if len(selected) != MAX_CASES or any(key not in selected for key in ROSTER):
        raise ValueError("frozen four-case roster drift")
    return selected


def _coverage(points: list[list[float]], shape: list[int]) -> dict:
    trace: dict = {}
    geometry._coverage(np.asarray(points, dtype=np.float64).reshape((-1, 2)), tuple(shape), trace)
    return trace


def _enforce_retained_cap(rows: list[dict]) -> int:
    total = sum(len(_canonical(row)) for row in rows)
    if total > MAX_RETAINED_ANNOTATION_BYTES:
        raise ValueError("cumulative retained annotation byte cap exceeded")
    return total


def _validate_truth_rows(proposals: list, truth_rows: list, surfaces: list[str],
                         points: dict, pair: tuple[str, str]) -> list[str | None]:
    if (not isinstance(proposals, list) or
            not isinstance(surfaces, list) or not surfaces or
            any(not isinstance(s, str) or not s or not s.isprintable() for s in surfaces) or
            surfaces != sorted(set(surfaces)) or
            not isinstance(truth_rows, list) or len(truth_rows) != len(proposals)):
        raise ValueError("malformed surface roster or truth ledger")
    left, right = pair
    if (not isinstance(proposals, list) or
            any(not isinstance(p, list) or len(p) != 2 or
                any(type(v) is not int for v in p) or
                not 0 <= p[0] < len(points[left]) or not 0 <= p[1] < len(points[right])
                for p in proposals) or
            len({tuple(p) for p in proposals}) != len(proposals) or
            len({p[0] for p in proposals}) != len(proposals) or
            len({p[1] for p in proposals}) != len(proposals)):
        raise ValueError("malformed proposal identities")
    labels = []
    for proposal, row in zip(proposals, truth_rows):
        if (not isinstance(row, list) or len(row) != 3 or row[:2] != proposal or
                row[2] is not None and row[2] not in surfaces):
            raise ValueError("unknown, duplicate, or omitted truth identity")
        labels.append(row[2])
    return labels


def analyze_pair(geo_row: dict, proposals: list, truth_rows: list,
                 surfaces: list[str], points: dict, dimensions: dict) -> dict:
    """Join truth to fixed proposal roles and observed attempts, without refitting."""
    pair = tuple(geo_row["frame_ids"])
    if (len(pair) != 2 or pair[0] not in points or pair[1] not in points or
            pair[0] not in dimensions or pair[1] not in dimensions):
        raise ValueError("pair frame binding drift")
    labels = _validate_truth_rows(proposals, truth_rows, surfaces, points, pair)
    true_proposals = sum(label is not None for label in labels)
    fixed_fit, fixed_held, mode = geometry._quota_partition(pair, tuple(map(tuple, proposals)))
    attempts = geo_row["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= geometry.MAX_MODELS_PER_PAIR:
        raise ValueError("malformed attempt roster")
    remaining = list(range(len(proposals)))
    accepted_models = []
    for index, attempt in enumerate(attempts):
        active = set(remaining)
        fit = [i for i in fixed_fit if i in active]
        held = [i for i in fixed_held if i in active]
        if (attempt.get("index") != index or attempt.get("remaining_indices") != remaining or
                attempt.get("partition_mode") != mode or
                attempt.get("fit_indices") != fit or attempt.get("held_indices") != held or
                attempt.get("fit_count") != len(fit) or attempt.get("held_count") != len(held)):
            raise ValueError("mutated fixed-role identity")
        inliers = attempt.get("fit_inlier_indices") or []
        if len(set(inliers)) != len(inliers) or not set(inliers) <= set(fit):
            raise ValueError("malformed fit-inlier identity")
        held_rows = attempt.get("held_rows") or []
        if held_rows and ([row[0] for row in held_rows] != held or
                          any(len(row) != 3 for row in held_rows)):
            raise ValueError("malformed held identity or residual")
        good_held = []
        for row in held_rows:
            values = row[1:]
            if any(v != "inf" and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
                   for v in values):
                raise ValueError("malformed held residual")
            if all(v != "inf" and v <= geometry.TRANSFER_PX for v in values):
                good_held.append(row[0])
        if attempt.get("first_rejection") == "accepted":
            support = inliers + good_held
            surface_counts = {surface: sum(labels[i] == surface for i in support)
                              for surface in surfaces}
            invalid = sum(labels[i] is None for i in support)
            mixed = sum(value > 0 for value in surface_counts.values()) > 1
            accepted_models.append(dict(attempt_index=index, support_count=len(support),
                                        surface_counts=surface_counts,
                                        invalid_match_count=invalid, mixed_surface=mixed))
            used = set(support)
            remaining = [i for i in remaining if i not in used]
        elif index != len(attempts) - 1:
            raise ValueError("attempt after rejection")
    if geo_row.get("accepted_model_count") != len(accepted_models):
        raise ValueError("accepted model count drift")
    first = attempts[0]
    first_fit = first["fit_indices"]
    first_held = first["held_indices"]
    first_inliers = first.get("fit_inlier_indices") or []
    first_joint = [row[0] for row in (first.get("held_rows") or [])
                   if all(v != "inf" and v <= geometry.TRANSFER_PX for v in row[1:])]
    surface_rows = []
    for surface in surfaces:
        true_fit = [i for i in first_fit if labels[i] == surface]
        true_held = [i for i in first_held if labels[i] == surface]
        true_inliers = [i for i in first_inliers if labels[i] == surface]
        true_joint = [i for i in first_joint if labels[i] == surface]
        capacity_ids = true_fit + true_held
        selected_ids = true_inliers + true_joint
        def coverage(indices: list[int], frame: str, side: int) -> dict:
            return _coverage([points[frame][proposals[i][side]] for i in indices], dimensions[frame])
        source_capacity = coverage(capacity_ids, pair[0], 0)
        dest_capacity = coverage(capacity_ids, pair[1], 1)
        held_fraction = len(true_held) / len(first_held) if first_held else 0.0
        role_count_sufficient = (len(true_fit) >= 16 and
                                 len(true_held) >= geometry.MIN_HOLDOUT and
                                 held_fraction >= .38)
        surface_rows.append(dict(surface=surface, true_fit_count=len(true_fit),
                                 true_held_count=len(true_held),
                                 true_held_fraction=held_fraction,
                                 true_fit_inlier_count=len(true_inliers),
                                 true_joint_held_pass_count=len(true_joint),
                                 role_capacity_source_coverage=source_capacity,
                                 role_capacity_destination_coverage=dest_capacity,
                                 selected_true_source_coverage=coverage(selected_ids, pair[0], 0),
                                 selected_true_destination_coverage=coverage(selected_ids, pair[1], 1),
                                 role_count_sufficient=role_count_sufficient,
                                 capacity_sufficient=(role_count_sufficient and
                                                      source_capacity["passed"] and dest_capacity["passed"])))
    capacity = any(row["capacity_sufficient"] for row in surface_rows)
    count_sufficient = any(row["role_count_sufficient"] for row in surface_rows)
    coverage_unassessed = count_sufficient and not capacity
    model_fault = ((capacity and first["first_rejection"] != "accepted") or
                   any(model["invalid_match_count"] or model["mixed_surface"] for model in accepted_models))
    supply_fault = not count_sufficient
    findings = (["supply_limited"] if supply_fault else []) + (
        ["coverage_subset_unassessed"] if coverage_unassessed else []) + (
        ["model_or_selection_limited"] if model_fault else [])
    classification = findings[0] if len(findings) == 1 else "unresolved"
    if classification == "coverage_subset_unassessed":
        classification = "unresolved"
    return dict(frame_ids=list(pair), proposal_count=len(proposals),
                true_proposal_count=true_proposals,
                false_proposal_count=len(proposals) - true_proposals,
                true_accepted_support_count=sum(sum(model["surface_counts"].values())
                                                for model in accepted_models),
                accepted_support_count=sum(model["support_count"] for model in accepted_models),
                partition_mode=mode, fit_count=len(first_fit), held_count=len(first_held),
                first_rejection=first["first_rejection"], surface_capacity=surface_rows,
                accepted_models=accepted_models, classification=classification,
                findings=findings)


def _require_sealed_pair_aggregate(analysis: dict, sealed_pair: dict) -> None:
    if (analysis["true_proposal_count"] != sealed_pair["true_proposals"] or
            analysis["false_proposal_count"] != sealed_pair["false_proposals"] or
            analysis["true_accepted_support_count"] != sealed_pair["true_accepted_support"] or
            analysis["accepted_support_count"] != sealed_pair["accepted_support"] or
            analysis["proposal_count"] != sealed_pair["reciprocal_count"]):
        raise ValueError("truth ledger contradicts sealed per-pair aggregates")


def canonical_pair_record_bytes(record: dict) -> bytes:
    """Independently replay a small record, including consistently resealed rows."""
    fields = {"schema", "config_sha256", "geo_row", "proposals", "truth_rows",
              "surfaces", "points", "dimensions", "analysis", "canonical_sha256"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("malformed pair record")
    payload = dict(record)
    digest = payload.pop("canonical_sha256")
    if (_sha(_canonical(payload)) != digest or record["schema"] != SCHEMA or
            record["config_sha256"] != CONFIG_SHA256 or
            record["analysis"] != analyze_pair(record["geo_row"], record["proposals"],
                                               record["truth_rows"], record["surfaces"],
                                               record["points"], record["dimensions"])):
        raise ValueError("resealed pair capacity drift")
    raw = _canonical(record)
    if len(raw) > MAX_RETAINED_ANNOTATION_BYTES:
        raise ValueError("retained annotation byte cap exceeded")
    return raw


def _enforce_final_cap(raw: bytes) -> bytes:
    if len(raw) > MAX_FINAL_ANNOTATION_BYTES:
        raise ValueError("final annotation byte cap exceeded")
    return raw


def _postseal_truth(report: dict, expected_sha256: str, truth_supplier: Callable[[], object]
                    ) -> tuple[bytes, object]:
    raw = evaluator.canonical_report_bytes(report)
    if _sha(raw) != expected_sha256:
        raise ValueError("inference report differs from sealed development case")
    truth = truth_supplier()
    if evaluator.canonical_report_bytes(report) != raw:
        raise ValueError("annotation changed sealed inference report")
    return raw, truth


def _truth_rows(oracle: dict, report: dict) -> tuple[list[str], list[dict]]:
    policy = next(row for row in report["policies"] if row["id"] == POLICY)
    surfaces = sorted(oracle["transforms"])
    points = report["feature_points"]
    rows = []
    for proposal_row in policy["proposal_telemetry"]:
        left, right = proposal_row["frame_ids"]
        truth = []
        for i, j in proposal_row["reciprocal_pairs"]:
            a = evaluator._label(oracle, left, points[left][i])
            b = evaluator._label(oracle, right, points[right][j])
            surface = a[0] if (a is not None and b is not None and a[0] == b[0] and
                               np.linalg.norm(a[1] - b[1]) <= 3) else None
            truth.append([i, j, surface])
        rows.append(dict(frame_ids=[left, right], truth_rows=truth))
    return surfaces, rows


def _case_annotation(case: dict, baseline: dict, report: dict) -> dict:
    evaluator.verify_frame_binding(report, case["frames"])
    raw, truth = _postseal_truth(report, baseline["report_sha256"],
                                 lambda: _truth_rows(case["oracle"], report))
    metrics = evaluator._case_metrics(case, report)
    if metrics != baseline:
        raise ValueError("existing case metrics parity drift")
    surfaces, truth_rows = truth
    policy = next(row for row in report["policies"] if row["id"] == POLICY)
    geo = policy["geometry_arms"][ARM]
    pair_annotations = [analyze_pair(geo_row, proposal_row["reciprocal_pairs"],
                                     truth_row["truth_rows"], surfaces,
                                     report["feature_points"], report["frame_dimensions"])
                        for geo_row, proposal_row, truth_row in zip(
                            geo["pair_telemetry"], policy["proposal_telemetry"], truth_rows)]
    if len(pair_annotations) != 3:
        raise ValueError("case pair roster drift")
    sealed_pairs = next(entry for entry in baseline["policies"]
                        if entry["id"] == POLICY)["geometry_arms"][ARM]["pairs"]
    for analysis, sealed_pair in zip(pair_annotations, sealed_pairs):
        _require_sealed_pair_aggregate(analysis, sealed_pair)
    row = dict(group=case["group"], kind=case["kind"],
               input_sha256=baseline["input_sha256"], inference_report_sha256=_sha(raw),
               case_metrics_sha256=_sha(_canonical(metrics)), inference_report=report,
               surfaces=surfaces, truth_rows=truth_rows, pair_annotations=pair_annotations)
    _enforce_retained_cap([row])
    return row


def _work_counts(rows: list[dict], *, complete: bool = True) -> dict:
    shared = arm_attempts = all_attempts = pair_slots = 0
    for row in rows:
        report = row["inference_report"]
        shared += report["shared_descriptor_comparisons"]
        for policy in report["policies"]:
            for arm, geo in policy["geometry_arms"].items():
                attempts = sum(len(pair["attempts"]) for pair in geo["pair_telemetry"])
                all_attempts += attempts
                if policy["id"] == POLICY and arm == ARM:
                    arm_attempts += attempts
                    pair_slots += len(geo["pair_telemetry"])
    work = dict(shared_descriptor_comparisons=shared, designated_pair_slots=pair_slots,
                designated_attempts=arm_attempts, all_arm_attempts=all_attempts,
                designated_ransac_iteration_upper_bound=arm_attempts * RANSAC_ITERATIONS_PER_ATTEMPT,
                all_arm_ransac_iteration_upper_bound=all_attempts * RANSAC_ITERATIONS_PER_ATTEMPT)
    if (shared > MAX_SHARED_COMPARISONS or
            (pair_slots != MAX_PAIR_SLOTS if complete else pair_slots > MAX_PAIR_SLOTS) or
            arm_attempts > MAX_ARM_ATTEMPTS or all_attempts > MAX_ALL_ARM_ATTEMPTS or
            work["designated_ransac_iteration_upper_bound"] > MAX_ARM_RANSAC_ITERATIONS or
            work["all_arm_ransac_iteration_upper_bound"] > MAX_ALL_ARM_RANSAC_ITERATIONS):
        raise ValueError("fixed-role attribution work cap exceeded")
    return work


def build_four_case_annotation() -> dict:
    """Future separately authorized one-shot entry point; never called in prep."""
    baseline = _sealed_baseline()
    found = {}
    for case in evaluator.synthetic_cases("dev"):
        key = (case["group"], case["kind"])
        if key not in ROSTER:
            del case  # Generated by the pinned stream; never inferred or retained.
            continue
        if key in found:
            raise ValueError("duplicate generated case")
        data, _ = geometry._frame_input(case["frames"])
        if {name: digest for name, _, digest in data} != baseline[key]["input_sha256"]:
            raise ValueError("generated RGB input hash drift")
        report = evaluator.diagnose(case["frames"])
        found[key] = _case_annotation(case, baseline[key], report)
        _enforce_retained_cap(list(found.values()))
        _work_counts(list(found.values()), complete=False)
        del case, report
        if len(found) == MAX_CASES:
            break
    if set(found) != set(ROSTER):
        raise ValueError("generated four-case roster drift")
    rows = [found[key] for key in ROSTER]
    result = dict(schema=SCHEMA, configuration=configuration(),
                  config_sha256=CONFIG_SHA256, cases=rows, work=_work_counts(rows))
    result["canonical_sha256"] = _sha(_canonical(result))
    canonical_annotation_bytes(result)
    return result


def canonical_annotation_bytes(result: dict) -> bytes:
    baseline = _sealed_baseline()
    if (not isinstance(result, dict) or set(result) !=
            {"schema", "configuration", "config_sha256", "cases", "work", "canonical_sha256"}):
        raise ValueError("malformed fixed-role annotation")
    payload = dict(result)
    digest = payload.pop("canonical_sha256")
    if (_sha(_canonical(payload)) != digest or result["schema"] != SCHEMA or
            result["configuration"] != configuration() or result["config_sha256"] != CONFIG_SHA256):
        raise ValueError("annotation digest or configuration drift")
    rows = result["cases"]
    if (not isinstance(rows, list) or len(rows) != MAX_CASES or
            [(row.get("group"), row.get("kind")) for row in rows] != list(ROSTER)):
        raise ValueError("four-case annotation roster drift")
    _enforce_retained_cap(rows)
    for row in rows:
        if set(row) != {"group", "kind", "input_sha256", "inference_report_sha256",
                        "case_metrics_sha256", "inference_report", "surfaces", "truth_rows",
                        "pair_annotations"}:
            raise ValueError("malformed case annotation")
        key = row["group"], row["kind"]
        sealed = baseline[key]
        report = row["inference_report"]
        raw = evaluator.canonical_report_bytes(report)
        if (_sha(raw) != sealed["report_sha256"] or
                row["inference_report_sha256"] != sealed["report_sha256"] or
                row["input_sha256"] != sealed["input_sha256"] or
                report["frame_sha256"] != sealed["input_sha256"] or
                row["case_metrics_sha256"] != _sha(_canonical(sealed))):
            raise ValueError("case source, metric parity, or retained cap drift")
        policy = next(entry for entry in report["policies"] if entry["id"] == POLICY)
        geo = policy["geometry_arms"][ARM]
        if (not isinstance(row["truth_rows"], list) or len(row["truth_rows"]) != 3 or
                not isinstance(row["pair_annotations"], list) or len(row["pair_annotations"]) != 3):
            raise ValueError("malformed pair annotation roster")
        for geo_row, proposal_row, truth_row, analysis in zip(
                geo["pair_telemetry"], policy["proposal_telemetry"],
                row["truth_rows"], row["pair_annotations"]):
            if truth_row.get("frame_ids") != geo_row["frame_ids"] or (
                    analyze_pair(geo_row, proposal_row["reciprocal_pairs"],
                                 truth_row["truth_rows"], row["surfaces"],
                                 report["feature_points"], report["frame_dimensions"]) != analysis):
                raise ValueError("resealed truth/role/capacity annotation drift")
        original = next(entry for entry in sealed["policies"] if entry["id"] == POLICY)["geometry_arms"][ARM]
        for analysis, sealed_pair in zip(row["pair_annotations"], original["pairs"]):
            _require_sealed_pair_aggregate(analysis, sealed_pair)
        if ([x["first_rejection"] for x in row["pair_annotations"]] !=
                [x["first_rejections"][0] for x in original["pairs"]] or
                [(x["fit_count"], x["held_count"]) for x in row["pair_annotations"]] !=
                [(x["fit_count"], x["held_count"]) for x in original["pairs"]]):
            raise ValueError("sealed first-exit denominator drift")
    if result["work"] != _work_counts(rows):
        raise ValueError("annotation work ledger drift")
    return _enforce_final_cap(_canonical(result))


def persist_annotation(path: str | Path, result: dict) -> None:
    raw = canonical_annotation_bytes(result)
    target = Path(path)
    if not target.parent.is_dir():
        raise ValueError("output parent missing")
    fd, temporary = tempfile.mkstemp(prefix=".fixed-role-capacity-v1-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            if output.write(raw) != len(raw):
                raise OSError("partial annotation write")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
