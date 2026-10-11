"""DS-18 directed-candidate research diagnostic over the frozen DS-15 pairs.

DeepSeek CLI DS-18, 2026-09-12. This script is diagnostic only and repeats the
exact 37 unordered pairs of the three frozen calibrated DS-15 frame sets (T2,
T6, ``t3-sparse-8``) through the new **directed** matcher
(:mod:`mc_photogrammetry.pipeline.directed_matching`), with the same hard
separation between estimating and knowing that DS-12/DS-14/DS-15 use:

1. **Blind directed extraction/estimation** (``--extract-only``). The frame
   inventory and pair enumeration come from the frozen image-derived M3
   calibration artifacts through DS-15's :func:`frozen_frame_sets`; no manifest,
   reference litematic, lattice or outcome selects a pair. Each pair goes through
   :func:`directed_matching.match_and_solve_pair`, and ``directed_estimates.json``
   + ``directed_estimates.csv`` are written and frozen with a canonical digest.
   ``load_camera_manifest`` is imported *inside* the phase-2 function only.

2. **Reference evaluation** (``--evaluate-only``). Only after the frozen estimate
   bytes are digested does this open each fixture's ``cameras.json`` and compare
   solved yaw class / relative rotation / baseline direction exactly as DS-14/15
   do, then apply the **preregistered** DS-18 promotion criterion and report the
   DS-14 matcher comparator over the same 37 pairs.

Nothing here changes a DS-14/DS-15 threshold, source file, test or artifact. The
DS-14 matcher's all-pairs result (8 solves / 4 correct / 4 false) is read from
``exports/diagnostics/ds15/joint_evaluation.json`` in phase 2 only, so no
reference-derived number can reach phase 1.

Run with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_directed_pairs.py
    .venv\\Scripts\\python.exe scripts/diagnose_directed_pairs.py --extract-only
    .venv\\Scripts\\python.exe scripts/diagnose_directed_pairs.py --evaluate-only
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mc_photogrammetry.pipeline import directed_matching  # noqa: E402
from mc_photogrammetry.pipeline.connected_camera import PairGates  # noqa: E402
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings  # noqa: E402
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    camera_rotation_matrix,
)

# DS-12/DS-14/DS-15 helpers reused verbatim so the pair rule, provenance digests,
# reference class definition and comparator cannot drift from the accepted runs.
from diagnose_connected_pairs import (  # noqa: E402
    _GAUGE_ROTATIONS,
    _reference_class,
    angle_between,
    digest,
    load_rgb,
    plain,
    source_revision,
    to_camera_estimate,
)
from diagnose_joint_all_pairs import frozen_frame_sets  # noqa: E402
from diagnose_joint_pairs import (  # noqa: E402
    _estimate_from_record,
    _result_record,
    _write_evaluation_csv,
    canonical_digest,
)

OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds18"
REPORTS = ROOT / "docs" / "reports"

DS14_COMPARATOR_PATH = ROOT / "exports" / "diagnostics" / "ds15" / "joint_evaluation.json"
DS14_COMPARATOR_DECLARED = {
    "matcher": "DS-14 reciprocal joint matcher",
    "source": "exports/diagnostics/ds15/joint_evaluation.json",
    "pairs": 37,
    "solved": 8,
    "class_correct": 4,
    "false_solves": 4,
    "precision_among_solves": 0.5,
    "per_set_correct": {"T2": 2, "T6": 2, "t3-sparse-8": 0},
    "per_set_false": {"T2": 0, "T6": 1, "t3-sparse-8": 3},
}

# Preregistered before the first DS-18 run. Every predicate must hold.
PROMOTION_CRITERION = {
    "min_class_correct": 6,
    "min_precision_among_solves": 0.75,
    "t2_t6_min_class_correct": 5,
    "t2_t6_max_false_solves": 1,
    "t3_sparse_max_false_solves": 0,
}


# --------------------------------------------------------------------------- #
# Phase 1: blind directed extraction and estimation.
# --------------------------------------------------------------------------- #
def _directed_record(result) -> dict:
    """DS-14's result record plus the directed support statistics."""
    record = _result_record(result)
    record["left_support"] = int(result.left_support)
    record["right_support"] = int(result.right_support)
    for hypothesis, score in zip(record["hypotheses"], result.hypotheses):
        hypothesis["left_support"] = int(score.left_support)
        hypothesis["right_support"] = int(score.right_support)
    return record


def run_extraction(out_dir: Path, *, verbose: bool = True) -> dict:
    """Phase 1: directed-match and solve every frozen DS-15 pair, blind."""
    matcher = JointMatchSettings()
    gates = PairGates()
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = frozen_frame_sets()
    input_records: dict[str, dict] = {}
    image_cache: dict[Path, np.ndarray] = {}
    pairs: list[dict] = []
    frame_set_records: list[dict] = []
    started_all = time.perf_counter()

    for entry in sets:
        calibration_path = ROOT / entry["calibration_artifact"]
        estimates = entry["estimates"]
        frames = entry["frames"]
        frame_set_records.append(
            {
                "set": entry["set"],
                "fixture": entry["fixture"],
                "calibration_artifact": entry["calibration_artifact"],
                "frames": list(frames),
                "pairs": len(frames) * (len(frames) - 1) // 2,
            }
        )
        for left_name, right_name in itertools.combinations(frames, 2):
            left_path = ROOT / entry["fixture"] / left_name
            right_path = ROOT / entry["fixture"] / right_name
            for path in (left_path, right_path):
                if path not in image_cache:
                    image_cache[path] = load_rgb(path)
                    input_records[str(path.relative_to(ROOT))] = {
                        "sha256": digest(path),
                        "size": [
                            int(image_cache[path].shape[1]),
                            int(image_cache[path].shape[0]),
                        ],
                    }

            pair_started = time.perf_counter()
            result = directed_matching.match_and_solve_pair(
                left_name,
                image_cache[left_path],
                right_name,
                image_cache[right_path],
                left_camera=to_camera_estimate(estimates[left_name]),
                right_camera=to_camera_estimate(estimates[right_name]),
                usable_masks=None,
                matcher=matcher,
                gates=gates,
            )
            elapsed = time.perf_counter() - pair_started
            pairs.append(
                {
                    "set": entry["set"],
                    "fixture": entry["fixture"],
                    "calibration_artifact": entry["calibration_artifact"],
                    "left": left_name,
                    "right": right_name,
                    "left_estimate": asdict(estimates[left_name]),
                    "right_estimate": asdict(estimates[right_name]),
                    "directed": _directed_record(result),
                    "extraction_seconds": elapsed,
                    "solve_seconds": 0.0,
                }
            )
            if verbose:
                print(
                    f"[blind] {entry['set']:12s} {left_name[11:16]}->{right_name[11:16]} "
                    f"{result.status:8s} {result.reason:32s} "
                    f"edges={result.candidate_edges:5d} support={result.support:4d}/"
                    f"{result.support_floor:4d} L/R={result.left_support}/{result.right_support} "
                    f"class={result.right_yaw_class} {elapsed:6.2f}s",
                    flush=True,
                )

    payload = {
        "schema": "ds18-1",
        "phase": "directed_extraction_and_estimation_blind",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "settings": {
            "directed_matcher": asdict(matcher),
            "pair_gates": asdict(gates),
            "usable_masks": "None (derived ~create_gui_mask per frame)",
            "candidate_graph": (
                "directed left -> right top-k nearest descriptors; no reciprocity "
                "requirement; sorted by (distance, left, right); capped and reported"
            ),
            "selection": (
                "per-left best residual-qualified edge first, then deterministic "
                "right-collision resolution; ranked on the resolved one-to-one "
                "support, then the pre-collision unique-left/unique-right supports"
            ),
            "pair_rule": (
                "every unordered pair C(n,2) of the frozen DS-15 calibrated frame "
                "set, sorted-filename order; no outcome selection"
            ),
        },
        "refusal_tokens": list(directed_matching.DIRECTED_REFUSAL_REASONS),
        "frame_sets": frame_set_records,
        "canonical_pairs_digest": canonical_digest(pairs),
        "calibration_artifacts": {
            entry["calibration_artifact"]: digest(ROOT / entry["calibration_artifact"])
            for entry in sets
        },
        "inputs": input_records,
        "pairs": pairs,
        "extraction_seconds_total": time.perf_counter() - started_all,
        "recipe": " ".join(sys.argv),
        "note": (
            "No cameras.json manifest, reference litematic, lattice, coordinate or "
            "asserted geometry was opened while producing this file. The reference "
            "evaluation runs only after these bytes are written and digested."
        ),
    }
    estimates_path = out_dir / "directed_estimates.json"
    estimates_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_estimates_csv(out_dir / "directed_estimates.csv", pairs)
    payload["_digest"] = digest(estimates_path)
    payload["_path"] = str(estimates_path.relative_to(ROOT))
    return payload


def _write_estimates_csv(path: Path, pairs: list[dict]) -> None:
    columns = [
        "set",
        "fixture",
        "left",
        "right",
        "status",
        "reason",
        "raw_left",
        "raw_right",
        "boundary_dropped_left",
        "boundary_dropped_right",
        "directed_edges",
        "total_directed_edges",
        "truncated",
        "ratio_left",
        "ratio_right",
        "mutual",
        "support",
        "left_support",
        "right_support",
        "support_floor",
        "bins_left",
        "bins_right",
        "median_selected_residual_degrees",
        "descriptor_cost",
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
            directed = record["directed"]
            writer.writerow(
                {
                    "set": record["set"],
                    "fixture": record["fixture"],
                    "left": record["left"],
                    "right": record["right"],
                    "status": directed["status"],
                    "reason": directed["reason"],
                    "raw_left": directed["raw_keypoints"]["left"],
                    "raw_right": directed["raw_keypoints"]["right"],
                    "boundary_dropped_left": directed["boundary_dropped"]["left"],
                    "boundary_dropped_right": directed["boundary_dropped"]["right"],
                    "directed_edges": directed["candidate_edges"],
                    "total_directed_edges": directed["total_graph_edges"],
                    "truncated": directed["truncated"],
                    "ratio_left": directed["ratio_survivors"]["left"],
                    "ratio_right": directed["ratio_survivors"]["right"],
                    "mutual": directed["mutual_survivors"],
                    "support": directed["support"],
                    "left_support": directed["left_support"],
                    "right_support": directed["right_support"],
                    "support_floor": directed["support_floor"],
                    "bins_left": directed["spatial_bins"]["left"],
                    "bins_right": directed["spatial_bins"]["right"],
                    "median_selected_residual_degrees": directed[
                        "median_selected_residual_degrees"
                    ],
                    "descriptor_cost": directed["descriptor_cost"],
                    "parallax_degrees": directed["parallax_degrees"],
                    "cheirality_fraction": directed["cheirality_fraction"],
                    "ambiguity_margin": directed["ambiguity_margin"],
                    "right_yaw_class": directed["right_yaw_class"],
                    "seconds": round(record["extraction_seconds"], 3),
                }
            )


# --------------------------------------------------------------------------- #
# Phase 2: reference evaluation (the only place a manifest is opened).
# --------------------------------------------------------------------------- #
def _comparator() -> dict:
    """The DS-14 all-pairs comparator, read from its frozen artifact when present."""
    block = dict(DS14_COMPARATOR_DECLARED)
    if DS14_COMPARATOR_PATH.exists():
        frozen = json.loads(DS14_COMPARATOR_PATH.read_text(encoding="utf-8"))
        summary = frozen.get("summary", {})
        block.update(
            {
                "artifact_sha256": digest(DS14_COMPARATOR_PATH),
                "artifact_solved": summary.get("solved"),
                "artifact_class_correct": summary.get("class_correct"),
                "artifact_false_solves": summary.get("false_solves"),
                "artifact_precision_among_solves": summary.get("precision_among_solves"),
                "artifact_refusal_reasons": summary.get("refusal_reasons"),
            }
        )
    else:  # pragma: no cover - the DS-15 run is part of the frozen evidence
        block["artifact_sha256"] = None
    return block


def run_reference_evaluation(out_dir: Path, frozen: dict) -> dict:
    """Phase 2: open manifests and score the already-frozen directed outputs."""
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest

    estimates_path = out_dir / "directed_estimates.json"
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
        directed = record["directed"]
        solved = bool(directed["solved"])
        row = {
            "set": record["set"],
            "fixture": fixture,
            "left": left_name,
            "right": right_name,
            "directed_status": directed["status"],
            "directed_reason": directed["reason"],
            "support": directed["support"],
            "left_support": directed["left_support"],
            "right_support": directed["right_support"],
            "support_floor": directed["support_floor"],
            "candidate_edges": directed["candidate_edges"],
            "median_selected_residual_degrees": directed["median_selected_residual_degrees"],
            "parallax_degrees": directed["parallax_degrees"],
            "cheirality_fraction": directed["cheirality_fraction"],
            "solved": solved,
            "solved_yaw_class": directed["right_yaw_class"],
            "reference_yaw_class": class_direct,
            "reference_yaw_class_floors": class_floors,
            "class_correct": bool(solved and directed["right_yaw_class"] == class_direct),
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
            "pair_seconds": round(float(record["extraction_seconds"]), 4),
        }
        pose = directed["ds11_pose"]
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
    class_correct = sum(1 for row in solved_rows if row["class_correct"])
    false_solves = len(solved_rows) - class_correct
    rotation_errors = [
        row["relative_rotation_error_deg"]
        for row in solved_rows
        if row["relative_rotation_error_deg"] is not None
    ]
    baseline_errors = [
        row["baseline_error_deg"] for row in solved_rows if row["baseline_error_deg"] is not None
    ]
    baseline_signless = [
        row["baseline_error_signless_deg"]
        for row in solved_rows
        if row["baseline_error_signless_deg"] is not None
    ]
    precision = class_correct / len(solved_rows) if solved_rows else None

    per_set: dict[str, dict] = {}
    for label in [entry["set"] for entry in frozen["frame_sets"]]:
        set_rows = [row for row in rows if row["set"] == label]
        set_solved = [row for row in set_rows if row["solved"]]
        set_correct = sum(1 for row in set_solved if row["class_correct"])
        per_set[label] = {
            "pairs": len(set_rows),
            "solved": len(set_solved),
            "refused": len(set_rows) - len(set_solved),
            "class_correct": set_correct,
            "false_solves": len(set_solved) - set_correct,
            "precision_among_solves": (
                set_correct / len(set_solved) if set_solved else None
            ),
            "refusal_reasons": dict(
                Counter(row["directed_reason"] for row in set_rows if not row["solved"])
            ),
            "per_class_solved": dict(
                Counter(
                    row["solved_yaw_class"]
                    for row in set_solved
                    if row["solved_yaw_class"] is not None
                )
            ),
            "answer_counts": {
                "correct": set_correct,
                "false": len(set_solved) - set_correct,
                "refused": len(set_rows) - len(set_solved),
                "reference_class_distribution": dict(
                    Counter(row["reference_yaw_class"] for row in set_rows)
                ),
            },
        }

    t2_t6_rows = [row for row in rows if row["set"] in ("T2", "T6")]
    t2_t6_solved = [row for row in t2_t6_rows if row["solved"]]
    t2_t6_correct = sum(1 for row in t2_t6_solved if row["class_correct"])
    t2_t6_false = len(t2_t6_solved) - t2_t6_correct
    t3_solved = [row for row in rows if row["set"] == "t3-sparse-8" and row["solved"]]
    t3_correct = sum(1 for row in t3_solved if row["class_correct"])
    t3_false = len(t3_solved) - t3_correct

    predicates = {
        "class_correct_at_least_6_of_37": class_correct
        >= PROMOTION_CRITERION["min_class_correct"],
        "precision_among_solves_at_least_0_75": precision is not None
        and precision >= PROMOTION_CRITERION["min_precision_among_solves"],
        "t2_t6_class_correct_at_least_5_of_9": t2_t6_correct
        >= PROMOTION_CRITERION["t2_t6_min_class_correct"],
        "t2_t6_false_solves_at_most_1": t2_t6_false
        <= PROMOTION_CRITERION["t2_t6_max_false_solves"],
        "t3_sparse_zero_false_solves": t3_false
        <= PROMOTION_CRITERION["t3_sparse_max_false_solves"],
    }

    seconds = [float(record["extraction_seconds"]) for record in frozen["pairs"]]
    payload = {
        "schema": "ds18-1",
        "phase": "reference_evaluation",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "frozen_estimates": {
            "path": str(estimates_path.relative_to(ROOT)),
            "sha256": frozen_digest,
            "canonical_pairs_digest": frozen.get("canonical_pairs_digest"),
        },
        "manifests": {
            fixture: {"path": entry["path"], "sha256": entry["sha256"]}
            for fixture, entry in manifests.items()
        },
        "ds14_comparator": _comparator(),
        "promotion_criterion": {
            "declared": PROMOTION_CRITERION,
            "predicates": predicates,
            "all_predicates_hold": bool(all(predicates.values())),
            "verdict": (
                "promote_candidate" if all(predicates.values()) else "retain_research_only"
            ),
        },
        "summary": {
            "pairs": len(rows),
            "solved": len(solved_rows),
            "refused": len(rows) - len(solved_rows),
            "class_correct": class_correct,
            "false_solves": false_solves,
            "precision_among_solves": precision,
            "within_5deg_rotation": int(sum(1 for value in rotation_errors if value <= 5.0)),
            "geometrically_wrong_solves_5deg": int(
                sum(1 for value in rotation_errors if value > 5.0)
            ),
            "median_relative_rotation_error_deg": (
                float(np.median(rotation_errors)) if rotation_errors else None
            ),
            "median_baseline_error_deg_gauge_aligned": (
                float(np.median(baseline_errors)) if baseline_errors else None
            ),
            "median_baseline_error_deg_signless": (
                float(np.median(baseline_signless)) if baseline_signless else None
            ),
            "t2_t6": {
                "pairs": len(t2_t6_rows),
                "solved": len(t2_t6_solved),
                "class_correct": t2_t6_correct,
                "false_solves": t2_t6_false,
            },
            "t3_sparse": {
                "pairs": sum(1 for row in rows if row["set"] == "t3-sparse-8"),
                "solved": len(t3_solved),
                "class_correct": t3_correct,
                "false_solves": t3_false,
            },
            "refusal_reasons": dict(
                Counter(row["directed_reason"] for row in rows if not row["solved"])
            ),
            "runtime": {
                "pairs": len(seconds),
                "total_seconds": float(np.sum(seconds)),
                "mean_seconds_per_pair": float(np.mean(seconds)) if seconds else None,
                "max_seconds_per_pair": float(np.max(seconds)) if seconds else None,
                "min_seconds_per_pair": float(np.min(seconds)) if seconds else None,
                "per_set_seconds": {
                    label: float(
                        np.sum(
                            [
                                float(record["extraction_seconds"])
                                for record in frozen["pairs"]
                                if record["set"] == label
                            ]
                        )
                    )
                    for label in per_set
                },
            },
            "per_set": per_set,
        },
        "note": (
            "Evaluation only. The manifest and the DS-14 comparator artifact were "
            "opened only after directed_estimates.json was written and digested; no "
            "matcher or solver parameter was changed after this comparison. The "
            "baseline direction is compared after removing the four-fold yaw gauge "
            "the single-view calibrator leaves free. The promotion criterion was "
            "preregistered before this run."
        ),
        "pairs": rows,
    }
    evaluation_path = out_dir / "directed_evaluation.json"
    evaluation_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_evaluation_csv(out_dir / "directed_evaluation.csv", rows)
    payload["_digest"] = digest(evaluation_path)
    payload["_path"] = str(evaluation_path.relative_to(ROOT))
    return payload


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="stop after the blind directed phase; do not open any manifest",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate the existing frozen directed_estimates.json without re-extracting",
    )
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    if args.extract_only and args.evaluate_only:
        parser.error("--extract-only and --evaluate-only are mutually exclusive")

    estimates_path = args.out_dir / "directed_estimates.json"
    if args.evaluate_only:
        if not estimates_path.exists():
            parser.error(f"no frozen directed estimates to evaluate: {estimates_path}")
        frozen = json.loads(estimates_path.read_text(encoding="utf-8"))
    else:
        frozen = run_extraction(args.out_dir)
        solved = sum(1 for record in frozen["pairs"] if record["directed"]["solved"])
        print(
            f"[blind] {len(frozen['pairs'])} unordered pairs over "
            f"{len(frozen['frame_sets'])} sets; solved {solved}; "
            f"canonical digest {frozen['canonical_pairs_digest']}; "
            f"frozen {frozen['_path']} sha256 {frozen['_digest']}"
        )
        if args.extract_only:
            return 0

    evaluation = run_reference_evaluation(args.out_dir, frozen)
    summary = evaluation["summary"]
    comparator = evaluation["ds14_comparator"]
    print(
        f"[evaluate] directed solved {summary['solved']}/{summary['pairs']}; "
        f"class correct {summary['class_correct']}; false solves {summary['false_solves']}; "
        f"precision {summary['precision_among_solves']}; "
        f"T2+T6 {summary['t2_t6']['class_correct']}/{summary['t2_t6']['pairs']} "
        f"({summary['t2_t6']['false_solves']} false); "
        f"t3 {summary['t3_sparse']['class_correct']}/{summary['t3_sparse']['pairs']} "
        f"({summary['t3_sparse']['false_solves']} false)"
    )
    print(
        f"[evaluate] DS-14 comparator solved {comparator['solved']}/{comparator['pairs']}; "
        f"class correct {comparator['class_correct']}; false solves {comparator['false_solves']}"
    )
    print(
        f"[evaluate] promotion verdict {evaluation['promotion_criterion']['verdict']}; "
        f"predicates {evaluation['promotion_criterion']['predicates']}"
    )
    print(f"[evaluate] wrote {evaluation['_path']} sha256 {evaluation['_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
