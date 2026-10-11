"""Frozen real-pair joint-matcher diagnostic (DS-14), two phases.

This repeats DS-12's exact six frozen pairs (same rule, same calibration
artifacts, same derived ``None`` usable masks) through the corrected joint
matcher, with the same hard separation between estimating and knowing:

1. **Blind joint extraction/estimation.** Load the frames and each frame's
   *image-derived* M3 calibration artifact, run
   :func:`~mc_photogrammetry.pipeline.joint_matching.match_and_solve_pair`, and
   write ``joint_estimates.json`` + ``joint_estimates.csv``. No ``cameras.json``,
   reference litematic, lattice or asserted metadata is opened in this phase, and
   ``load_camera_manifest`` is imported *inside* :func:`run_evaluation` so the
   blind phase cannot see it.

2. **Reference evaluation (separate).** Load the already-written
   ``joint_estimates.json``, record its SHA-256, and only then open the fixture
   manifests to compare the solved relative yaw class and baseline direction.
   Write ``joint_evaluation.json`` + ``joint_evaluation.csv``. No matcher or
   solver parameter is changed after this point.

The frozen pair list is imported verbatim from DS-12's script, so the six pairs
are the same bytes-of-rule, not a re-selection. The matcher's settings are the
module defaults, declared before the first run; ``joint_estimates.json`` also
carries a ``canonical_pairs_digest`` over the outcome payload with wall-clock
fields stripped, so two phase-1 runs can be compared for determinism.

Run with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_joint_pairs.py --extract-only
    .venv\\Scripts\\python.exe scripts/diagnose_joint_pairs.py --evaluate-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mc_photogrammetry.pipeline.connected_camera import PairGates  # noqa: E402
from mc_photogrammetry.pipeline.joint_matching import (  # noqa: E402
    JointMatchSettings,
    JointPairResult,
    match_and_solve_pair,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    camera_rotation_matrix,
)

# DS-12's frozen pair rule and helpers, reused verbatim so the pair list cannot
# drift: first two adjacent pairs in sorted-filename order per fixture.
from diagnose_connected_pairs import (  # noqa: E402
    PAIR_SPECS,
    _Estimate,
    _GAUGE_ROTATIONS,
    _reference_class,
    angle_between,
    digest,
    load_estimates,
    load_rgb,
    plain,
    source_revision,
    to_camera_estimate,
)

OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds14"
REPORTS = ROOT / "docs" / "reports"


def _hypothesis_record(score) -> dict:
    return {
        "yaw_class": score.yaw_class,
        "mode_index": score.mode_index,
        "support": score.support,
        "spatial_bins_left": score.spatial_bins_left,
        "spatial_bins_right": score.spatial_bins_right,
        "spatial_bins": score.spatial_bins,
        "median_selected_residual_degrees": score.median_selected_residual_degrees,
        "descriptor_cost": score.descriptor_cost,
        "median_all_edge_residual_degrees": score.median_all_edge_residual_degrees,
        "parallax_degrees": score.parallax_degrees,
        "cheirality_fraction": score.cheirality_fraction,
        "singular_ratio": score.singular_ratio,
        "baseline_direction": list(score.baseline_direction),
        "passed": bool(score.passed),
        "refusal": score.refusal,
    }


def _pose_record(pose) -> dict:
    return {
        "solved": bool(pose.solved),
        "reason": pose.reason,
        "right_yaw_class": pose.right_yaw_class,
        "relative_rotation": (
            np.asarray(pose.relative_rotation).tolist()
            if pose.relative_rotation is not None
            else None
        ),
        "baseline_direction": (
            list(pose.baseline_direction) if pose.baseline_direction is not None else None
        ),
        "inliers": pose.inliers,
        "inlier_fraction": pose.inlier_fraction,
        "median_rho_degrees": pose.median_rho_degrees,
        "parallax_degrees": pose.parallax_degrees,
        "cheirality_fraction": pose.cheirality_fraction,
        "classes": [
            {
                "yaw_class": score.yaw_class,
                "passed": score.passed,
                "refusal": score.refusal,
                "inliers": score.inliers,
                "informative": score.informative,
                "median_rho_degrees": score.median_rho_degrees,
                "parallax_degrees": score.parallax_degrees,
                "cheirality_fraction": score.cheirality_fraction,
            }
            for score in pose.classes
        ],
    }


def _result_record(result: JointPairResult) -> dict:
    return {
        "status": result.status,
        "reason": result.reason,
        "solved": bool(result.solved),
        "right_yaw_class": result.right_yaw_class,
        "raw_keypoints": {"left": result.raw_keypoints[0], "right": result.raw_keypoints[1]},
        "boundary_dropped": {
            "left": result.boundary_dropped[0],
            "right": result.boundary_dropped[1],
        },
        "candidate_edges": result.candidate_edges,
        "total_graph_edges": result.total_graph_edges,
        "truncated": result.truncated,
        "ratio_survivors": {
            "left": result.ratio_survivors[0],
            "right": result.ratio_survivors[1],
        },
        "mutual_survivors": result.mutual_survivors,
        "support": result.support,
        "support_floor": result.support_floor,
        "spatial_bins": {
            "left": result.spatial_bins_left,
            "right": result.spatial_bins_right,
            "total": result.spatial_bins_total,
        },
        "median_selected_residual_degrees": result.median_selected_residual_degrees,
        "descriptor_cost": result.descriptor_cost,
        "parallax_degrees": result.parallax_degrees,
        "cheirality_fraction": result.cheirality_fraction,
        "singular_ratio": result.singular_ratio,
        "ambiguity_margin": result.ambiguity_margin,
        "hypotheses": [_hypothesis_record(score) for score in result.hypotheses],
        "matches": result.matches.tolist(),
        "ds11_pose": None if result.pose is None else _pose_record(result.pose),
    }


def canonical_digest(pairs: list[dict]) -> str:
    """Digest of the outcome payload with wall-clock fields stripped."""
    stripped = []
    for record in pairs:
        copy = json.loads(json.dumps(record, default=plain))
        copy.pop("extraction_seconds", None)
        copy.pop("solve_seconds", None)
        stripped.append(copy)
    encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_extraction(out_dir: Path) -> dict:
    """Phase 1: joint match and solve with no manifest access."""
    matcher = JointMatchSettings()
    gates = PairGates()
    out_dir.mkdir(parents=True, exist_ok=True)

    estimates_cache: dict[Path, dict[str, _Estimate]] = {}
    input_records: dict[str, dict] = {}
    pairs: list[dict] = []

    for fixture, calibration_name, left_name, right_name in PAIR_SPECS:
        calibration_path = REPORTS / calibration_name
        if calibration_path not in estimates_cache:
            estimates_cache[calibration_path] = load_estimates(calibration_path)
        estimates = estimates_cache[calibration_path]

        folder = ROOT / fixture
        left_path = folder / left_name
        right_path = folder / right_name
        left_rgb = load_rgb(left_path)
        right_rgb = load_rgb(right_path)
        for path, image in ((left_path, left_rgb), (right_path, right_rgb)):
            input_records[str(path.relative_to(ROOT))] = {
                "sha256": digest(path),
                "size": [int(image.shape[1]), int(image.shape[0])],
            }

        started = time.perf_counter()
        result = match_and_solve_pair(
            left_name,
            left_rgb,
            right_name,
            right_rgb,
            left_camera=to_camera_estimate(estimates[left_name]),
            right_camera=to_camera_estimate(estimates[right_name]),
            usable_masks=None,
            matcher=matcher,
            gates=gates,
        )
        elapsed = time.perf_counter() - started

        record = {
            "fixture": fixture,
            "calibration_artifact": str(calibration_path.relative_to(ROOT)),
            "left": left_name,
            "right": right_name,
            "left_estimate": asdict(estimates[left_name]),
            "right_estimate": asdict(estimates[right_name]),
            "joint": _result_record(result),
            "extraction_seconds": elapsed,
            "solve_seconds": 0.0,
        }
        pairs.append(record)

    payload = {
        "schema": "ds14-1",
        "phase": "extraction_and_estimation_blind",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "settings": {
            "joint_matcher": asdict(matcher),
            "pair_gates": asdict(gates),
            "usable_masks": "None (derived ~create_gui_mask per frame)",
        },
        "canonical_pairs_digest": canonical_digest(pairs),
        "calibration_artifacts": {
            str(path.relative_to(ROOT)): digest(path) for path in sorted(estimates_cache)
        },
        "inputs": input_records,
        "pairs": pairs,
        "recipe": " ".join(sys.argv),
        "note": (
            "No cameras.json manifest, reference litematic, lattice, coordinate or "
            "asserted geometry was opened while producing this file. The reference "
            "evaluation runs only after it is written; joint_evaluation.json records "
            "the SHA-256 of these exact bytes."
        ),
    }
    estimates_path = out_dir / "joint_estimates.json"
    estimates_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_estimates_csv(out_dir / "joint_estimates.csv", pairs)
    payload["_digest"] = digest(estimates_path)
    payload["_path"] = str(estimates_path.relative_to(ROOT))
    return payload


def _write_estimates_csv(path: Path, pairs: list[dict]) -> None:
    columns = [
        "fixture",
        "left",
        "right",
        "status",
        "reason",
        "raw_left",
        "raw_right",
        "boundary_dropped_left",
        "boundary_dropped_right",
        "candidate_edges",
        "total_graph_edges",
        "truncated",
        "ratio_left",
        "ratio_right",
        "mutual",
        "support",
        "support_floor",
        "bins_left",
        "bins_right",
        "median_selected_residual_degrees",
        "parallax_degrees",
        "cheirality_fraction",
        "ambiguity_margin",
        "right_yaw_class",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in pairs:
            joint = record["joint"]
            writer.writerow(
                {
                    "fixture": record["fixture"],
                    "left": record["left"],
                    "right": record["right"],
                    "status": joint["status"],
                    "reason": joint["reason"],
                    "raw_left": joint["raw_keypoints"]["left"],
                    "raw_right": joint["raw_keypoints"]["right"],
                    "boundary_dropped_left": joint["boundary_dropped"]["left"],
                    "boundary_dropped_right": joint["boundary_dropped"]["right"],
                    "candidate_edges": joint["candidate_edges"],
                    "total_graph_edges": joint["total_graph_edges"],
                    "truncated": joint["truncated"],
                    "ratio_left": joint["ratio_survivors"]["left"],
                    "ratio_right": joint["ratio_survivors"]["right"],
                    "mutual": joint["mutual_survivors"],
                    "support": joint["support"],
                    "support_floor": joint["support_floor"],
                    "bins_left": joint["spatial_bins"]["left"],
                    "bins_right": joint["spatial_bins"]["right"],
                    "median_selected_residual_degrees": joint[
                        "median_selected_residual_degrees"
                    ],
                    "parallax_degrees": joint["parallax_degrees"],
                    "cheirality_fraction": joint["cheirality_fraction"],
                    "ambiguity_margin": joint["ambiguity_margin"],
                    "right_yaw_class": joint["right_yaw_class"],
                    "seconds": round(record["extraction_seconds"], 3),
                }
            )


def _estimate_from_record(record: dict) -> _Estimate:
    return _Estimate(**record)


def run_evaluation(out_dir: Path, frozen: dict) -> dict:
    """Phase 2: the only place a manifest is opened."""
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest

    estimates_path = out_dir / "joint_estimates.json"
    frozen_digest = digest(estimates_path)
    manifests: dict[str, dict] = {}
    rows: list[dict] = []
    for record in frozen["pairs"]:
        fixture = record["fixture"]
        manifest_path = ROOT / fixture / "cameras.json"
        if fixture not in manifests:
            manifests[fixture] = {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": digest(manifest_path),
                "cameras": load_camera_manifest(manifest_path),
            }
        cameras = manifests[fixture]["cameras"]
        left_name, right_name = record["left"], record["right"]
        truth_left, truth_right = cameras[left_name], cameras[right_name]
        left = _estimate_from_record(record["left_estimate"])
        right = _estimate_from_record(record["right_estimate"])

        class_direct, class_floors = _reference_class(left, right, truth_left, truth_right)
        joint = record["joint"]
        solved = bool(joint["solved"])
        row = {
            "fixture": fixture,
            "left": left_name,
            "right": right_name,
            "joint_status": joint["status"],
            "joint_reason": joint["reason"],
            "support": joint["support"],
            "support_floor": joint["support_floor"],
            "candidate_edges": joint["candidate_edges"],
            "solved": solved,
            "solved_yaw_class": joint["right_yaw_class"],
            "reference_yaw_class": class_direct,
            "reference_yaw_class_floors": class_floors,
            "class_correct": bool(solved and joint["right_yaw_class"] == class_direct),
            "relative_rotation_error_deg": None,
            "baseline_error_deg": None,
            "baseline_error_signless_deg": None,
            "baseline_error_without_gauge_deg": None,
            "gauge_index": None,
            "baseline_length_blocks": float(
                np.linalg.norm(
                    np.asarray(truth_right.position, dtype=float)
                    - np.asarray(truth_left.position, dtype=float)
                )
            ),
            "manifest_left_yaw_degrees": float(truth_left.yaw_degrees),
            "manifest_right_yaw_degrees": float(truth_right.yaw_degrees),
        }
        pose = joint["ds11_pose"]
        if solved and pose is not None and pose["relative_rotation"] is not None:
            true_rotation = (
                camera_rotation_matrix(truth_right) @ camera_rotation_matrix(truth_left).T
            )
            got_rotation = np.asarray(pose["relative_rotation"], dtype=float)
            row["relative_rotation_error_deg"] = angle_between(got_rotation, true_rotation)

            baseline_true = (
                np.asarray(truth_right.position, dtype=float)
                - np.asarray(truth_left.position, dtype=float)
            )
            length = float(np.linalg.norm(baseline_true))
            if length > 1e-9 and pose["baseline_direction"] is not None:
                baseline_true = baseline_true / length
                baseline_got = np.asarray(pose["baseline_direction"], dtype=float)
                baseline_got = baseline_got / max(float(np.linalg.norm(baseline_got)), 1e-12)
                errors = {}
                for step in _GAUGE_ROTATIONS:
                    angle = np.radians(float(step))
                    rotation = np.asarray(
                        (
                            (np.cos(angle), 0.0, np.sin(angle)),
                            (0.0, 1.0, 0.0),
                            (-np.sin(angle), 0.0, np.cos(angle)),
                        )
                    )
                    candidate = rotation @ baseline_true
                    errors[step] = float(
                        np.degrees(
                            np.arccos(
                                float(np.clip(np.dot(candidate, baseline_got), -1.0, 1.0))
                            )
                        )
                    )
                best = min(errors, key=lambda step: (errors[step], step))
                row["gauge_index"] = best // 90
                row["baseline_error_deg"] = errors[best]
                row["baseline_error_without_gauge_deg"] = errors[0]
                cosine = float(np.clip(np.dot(baseline_true, baseline_got), -1.0, 1.0))
                row["baseline_error_signless_deg"] = float(np.degrees(np.arccos(abs(cosine))))
        rows.append(row)

    solved_rows = [row for row in rows if row["solved"]]
    class_rows = [row for row in solved_rows if row["solved_yaw_class"] is not None]
    rotation_errors = [
        row["relative_rotation_error_deg"]
        for row in solved_rows
        if row["relative_rotation_error_deg"] is not None
    ]
    baseline_errors = [
        row["baseline_error_deg"] for row in solved_rows if row["baseline_error_deg"] is not None
    ]
    payload = {
        "schema": "ds14-1",
        "phase": "reference_evaluation",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_estimates": {
            "path": str(estimates_path.relative_to(ROOT)),
            "sha256": frozen_digest,
        },
        "manifests": {
            fixture: {"path": entry["path"], "sha256": entry["sha256"]}
            for fixture, entry in manifests.items()
        },
        "summary": {
            "pairs": len(rows),
            "solved": len(solved_rows),
            "refused": len(rows) - len(solved_rows),
            "class_correct": sum(1 for row in class_rows if row["class_correct"]),
            "median_relative_rotation_error_deg": (
                float(np.median(rotation_errors)) if rotation_errors else None
            ),
            "median_baseline_error_deg_gauge_aligned": (
                float(np.median(baseline_errors)) if baseline_errors else None
            ),
            "within_5deg_rotation": int(sum(1 for value in rotation_errors if value <= 5.0)),
        },
        "note": (
            "Evaluation only. The manifest is scaffolding and was opened only after "
            "joint_estimates.json was written; no matcher or solver parameter was "
            "changed after this comparison. The baseline direction is compared after "
            "removing the four-fold yaw gauge the single-view calibrator leaves free."
        ),
        "pairs": rows,
    }
    evaluation_path = out_dir / "joint_evaluation.json"
    evaluation_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_evaluation_csv(out_dir / "joint_evaluation.csv", rows)
    payload["_digest"] = digest(evaluation_path)
    payload["_path"] = str(evaluation_path.relative_to(ROOT))
    return payload


def _write_evaluation_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="stop after the blind phase; do not open any manifest",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate the existing frozen joint_estimates.json without re-extracting",
    )
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    if args.extract_only and args.evaluate_only:
        parser.error("--extract-only and --evaluate-only are mutually exclusive")

    if args.evaluate_only:
        estimates_path = args.out_dir / "joint_estimates.json"
        if not estimates_path.exists():
            parser.error(f"no frozen estimates to evaluate: {estimates_path}")
        frozen = json.loads(estimates_path.read_text(encoding="utf-8"))
    else:
        frozen = run_extraction(args.out_dir)
        solved = sum(1 for p in frozen["pairs"] if p["joint"]["solved"])
        print(
            f"[blind] {len(frozen['pairs'])} pairs; solved {solved}; "
            f"canonical digest {frozen['canonical_pairs_digest']}; "
            f"frozen {frozen['_path']} sha256 {frozen['_digest']}"
        )
        if args.extract_only:
            return 0

    evaluation = run_evaluation(args.out_dir, frozen)
    summary = evaluation["summary"]
    print(
        f"[evaluate] solved {summary['solved']}/{summary['pairs']}; "
        f"class correct {summary['class_correct']}; "
        f"median rotation error {summary['median_relative_rotation_error_deg']}; "
        f"median gauge-aligned baseline error "
        f"{summary['median_baseline_error_deg_gauge_aligned']}"
    )
    print(f"[evaluate] wrote {evaluation['_path']} sha256 {evaluation['_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
