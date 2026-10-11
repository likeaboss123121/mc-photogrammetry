"""DS-15 all-pairs blind graph diagnostic over three frozen calibrated frame sets.

DeepSeek CLI DS-15, 2026-09-12. This script is diagnostic only. It repeats the
DS-14 joint matcher **unchanged** (module defaults, same ``PairGates()``, same
``usable_masks=None``) over *every unordered pair* of each frozen calibrated
frame set -- T2 (3 frames), T6 (4 frames) and ``t3-sparse-8`` (8 frames), 37
pairs total -- with the same hard separation between estimating and knowing that
DS-12/DS-14 use:

1. **Blind all-pairs extraction/estimation** (``--extract-only``). The frame
   inventory is read from each set's *image-derived* M3 calibration artifact and
   the pairs are the full ``C(n, 2)`` enumeration in sorted-filename order --
   no manifest, reference litematic, lattice or outcome is consulted to pick a
   pair. Each pair goes through
   :func:`~mc_photogrammetry.pipeline.joint_matching.match_and_solve_pair`, and
   ``joint_estimates.json`` + ``joint_estimates.csv`` are written and frozen.
   ``load_camera_manifest`` is imported *inside* the phase-2 function only, so it
   is unreachable here.

2a. **Manifest-free graph diagnostics** (``--graph-only``, or the first half of
   ``--evaluate-only``). Computed **only** from the already-frozen pair outputs
   and written to ``joint_graph_diagnostics.json`` + ``joint_triangles.csv``
   *before* any manifest is opened. Three diagnostics:

   * **Connectivity** of the solved-edge graph per set (union-find over the
     frames; refusals are missing edges, never a re-selection).
   * **Relative-class cycle consistency**: for each triangle whose three edges
     all solved, the yaw classes must close exactly modulo 4,
     ``c(i,j) + c(j,k) - c(i,k) == 0 (mod 4)``; the reported relative rotations
     must satisfy ``R(j,k) R(i,j) == R(i,k)`` as a continuous companion. When a
     triangle has exactly two solved edges the cycle still *implies* the class
     on the missing edge; that derived requirement (with the missing edge's
     refusal reason) is recorded as supplementary evidence, never as a solve.
   * **Triangle direction-closure feasibility**: transport the three unit
     baseline directions into the anchor frame ``i`` using the estimated
     relative rotation ``R(i,j)`` and solve ``min ||M x||`` over ``x >= 0`` with
     the single scale gauge ``||x||_2 = 1`` (exact active-set enumeration of the
     three-column problem). The normalized residual ``sigma_min / sigma_max`` and
     the nonnegativity penalty are reported; nothing is fed back into a pair.

2b. **Reference evaluation** (``--evaluate-only``). Only after the frozen
   estimate bytes and the graph artifact are digested does this open each
   fixture's ``cameras.json`` and compare solved yaw class / relative rotation /
   baseline direction exactly as DS-14 does, then report precision among solves,
   false-solve count, refusal reasons and runtime.

Nothing here changes a DS-14 threshold, source file or test. The two declared
tolerances below are *reporting* constants fixed before the first run
(``class`` closure is exact arithmetic; the closure tolerance is on the
normalized residual and is also reported raw so a reader can re-threshold).

Run with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_joint_all_pairs.py --extract-only
    .venv\\Scripts\\python.exe scripts/diagnose_joint_all_pairs.py --graph-only
    .venv\\Scripts\\python.exe scripts/diagnose_joint_all_pairs.py --evaluate-only
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

from mc_photogrammetry.pipeline.connected_camera import PairGates  # noqa: E402
from mc_photogrammetry.pipeline.joint_matching import (  # noqa: E402
    JointMatchSettings,
    match_and_solve_pair,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    camera_rotation_matrix,
)

# DS-12/DS-14 helpers reused verbatim so the pair rule, provenance digests and
# reference definition cannot drift from the accepted baseline.
from diagnose_connected_pairs import (  # noqa: E402
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
from diagnose_joint_pairs import (  # noqa: E402
    _estimate_from_record,
    _result_record,
    _write_estimates_csv,
    canonical_digest,
)

OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds15"
REPORTS = ROOT / "docs" / "reports"

# Frozen calibrated frame sets: (set label, screenshot folder, image-derived M3
# calibration artifact). The frame inventory is the artifact's estimate list.
FRAME_SETS: tuple[tuple[str, str, str], ...] = (
    ("T2", "templates 2", "M3-calibration-templates-2.json"),
    ("T6", "templates 6", "M3-calibration-templates-6.json"),
    ("t3-sparse-8", "templates 3", "M3-calibration-templates-3-t3-sparse-8.json"),
)

# Declared diagnostic-only constants, fixed before the first run. They only
# label the report; the raw residuals are always emitted.
CLASS_CYCLE_MODULUS = 4
CLOSURE_RESIDUAL_TOLERANCE = 0.05
_TINY = 1e-12


# --------------------------------------------------------------------------- #
# Frozen frame inventory (image-derived calibration artifacts only).
# --------------------------------------------------------------------------- #
def frozen_frame_sets() -> list[dict]:
    """Every set's frames and all unordered pairs, deterministic and blind."""
    sets: list[dict] = []
    for label, fixture, calibration_name in FRAME_SETS:
        calibration_path = REPORTS / calibration_name
        estimates = load_estimates(calibration_path)
        frames = sorted(estimates)
        sets.append(
            {
                "set": label,
                "fixture": fixture,
                "calibration_artifact": str(calibration_path.relative_to(ROOT)),
                "frames": frames,
                "estimates": estimates,
            }
        )
    return sets


# --------------------------------------------------------------------------- #
# Phase 1: blind all-pairs extraction and estimation.
# --------------------------------------------------------------------------- #
def run_extraction(out_dir: Path) -> dict:
    """Phase 1: match and solve every unordered pair with no manifest access."""
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
            result = match_and_solve_pair(
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
                    "joint": _result_record(result),
                    "extraction_seconds": elapsed,
                    "solve_seconds": 0.0,
                }
            )

    payload = {
        "schema": "ds15-1",
        "phase": "all_pairs_extraction_and_estimation_blind",
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
            "pair_rule": (
                "every unordered pair C(n,2) of the frozen calibrated frame set, "
                "sorted-filename order; no outcome selection"
            ),
        },
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
            "evaluation runs only after this file and the manifest-free graph "
            "artifact are written; joint_evaluation.json records the SHA-256 of "
            "these exact bytes."
        ),
    }
    estimates_path = out_dir / "joint_estimates.json"
    estimates_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_estimates_csv(out_dir / "joint_estimates.csv", pairs)
    payload["_digest"] = digest(estimates_path)
    payload["_path"] = str(estimates_path.relative_to(ROOT))
    return payload


# --------------------------------------------------------------------------- #
# Phase 2a: manifest-free graph diagnostics from the frozen pair outputs.
# --------------------------------------------------------------------------- #
def _unit(vector) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= _TINY:
        return None
    return value / norm


def nonnegative_closure(columns: np.ndarray) -> dict:
    """Exact ``min ||M x||`` over ``x >= 0``, ``||x||_2 = 1`` for 3 columns.

    The feasible set is the union of the nonnegative orthants of the seven
    nonempty coordinate subsets. On the relative interior of one face the
    stationary points are the right singular vectors of the restricted matrix,
    so the global minimum is the smallest singular value over subsets whose
    smallest right singular vector can be sign-fixed nonnegative (a mixed-sign
    vector cannot be the interior optimum; that face's optimum lies on a
    sub-face, which is also enumerated). This is exact and deterministic.
    """
    matrix = np.asarray(columns, dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular[0])
    best: dict | None = None
    for size in (3, 2, 1):
        for subset in itertools.combinations(range(3), size):
            restricted = matrix[:, list(subset)]
            _, values, right = np.linalg.svd(restricted)
            vector = right[-1]
            if np.any(vector > _TINY) and np.any(vector < -_TINY):
                continue
            residual = float(values[-1])
            if best is None or residual < best["residual"] - 1e-12:
                lengths = np.abs(vector)
                best = {
                    "residual": residual,
                    "subset": subset,
                    "lengths": lengths,
                }
    lengths = best["lengths"]
    total = float(np.sum(lengths))
    normalised_lengths = lengths / total if total > _TINY else lengths
    residual_vector = matrix @ lengths
    return {
        "sigma_max": sigma_max,
        "sigma_min_unconstrained": float(singular[-1]),
        "closure_residual_absolute": float(np.linalg.norm(residual_vector)),
        "closure_residual_normalized": (
            float(np.linalg.norm(residual_vector) / sigma_max) if sigma_max > _TINY else None
        ),
        "nonnegativity_penalty": float(np.linalg.norm(residual_vector) - singular[-1]),
        "closure_support": list(best["subset"]),
        "closure_lengths_unit_norm": lengths.tolist(),
        "closure_lengths_sum_one": normalised_lengths.tolist(),
        "closure_residual_vector": residual_vector.tolist(),
    }


def _pair_index(frozen: dict) -> dict[tuple[str, str, str], dict]:
    return {
        (record["fixture"], record["left"], record["right"]): record
        for record in frozen["pairs"]
    }


def _components(frames: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {frame: frame for frame in frames}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    groups: dict[str, list[str]] = {}
    for frame in frames:
        groups.setdefault(find(frame), []).append(frame)
    return sorted((sorted(group) for group in groups.values()), key=lambda g: g[0])


def _triangle_diagnostics(
    left: str, middle: str, right: str, records: dict[str, dict]
) -> dict:
    """Class/rotation/closure diagnostics for one ordered triple ``i<j<k``."""
    record_ij, record_jk, record_ik = records["ij"], records["jk"], records["ik"]
    ordered = {"ij": record_ij, "jk": record_jk, "ik": record_ik}
    solved_flags = {key: bool(record["joint"]["solved"]) for key, record in ordered.items()}
    solved = all(solved_flags.values())
    row = {
        "i": left,
        "j": middle,
        "k": right,
        "solved_edges": sum(1 for value in solved_flags.values() if value),
        "all_solved": solved,
        "classes": None,
        "class_cycle_residual_mod4": None,
        "class_cycle_closed": None,
        "missing_edge": None,
        "missing_edge_reason": None,
        "partial_classes": None,
        "implied_missing_class": None,
        "rotation_cycle_error_deg": None,
        "closure_residual_normalized": None,
        "closure_residual_absolute": None,
        "sigma_min_unconstrained": None,
        "sigma_max": None,
        "nonnegativity_penalty": None,
        "closure_support": None,
        "closure_lengths_unit_norm": None,
        "closure_residual_vector": None,
        "closure_within_tolerance": None,
    }
    if not solved:
        # With exactly two solved edges the class cycle still *requires* a class
        # on the missing edge; report that requirement. It is a derived
        # requirement on a refused pair, never a solve and never an input to one.
        if sum(1 for value in solved_flags.values() if value) == 2:
            classes = {
                key: int(record["joint"]["ds11_pose"]["right_yaw_class"])
                for key, record in ordered.items()
                if solved_flags[key] and record["joint"]["ds11_pose"] is not None
            }
            missing = next(key for key, value in solved_flags.items() if not value)
            if missing == "ij" and {"ik", "jk"} <= set(classes):
                implied = classes["ik"] - classes["jk"]
            elif missing == "jk" and {"ik", "ij"} <= set(classes):
                implied = classes["ik"] - classes["ij"]
            elif missing == "ik" and {"ij", "jk"} <= set(classes):
                implied = classes["ij"] + classes["jk"]
            else:
                implied = None
            row["missing_edge"] = missing
            row["missing_edge_reason"] = ordered[missing]["joint"]["reason"]
            row["partial_classes"] = classes
            row["implied_missing_class"] = (
                None if implied is None else int(implied % CLASS_CYCLE_MODULUS)
            )
        return row

    pose_ij = record_ij["joint"]["ds11_pose"]
    pose_jk = record_jk["joint"]["ds11_pose"]
    pose_ik = record_ik["joint"]["ds11_pose"]
    if any(pose["relative_rotation"] is None for pose in (pose_ij, pose_jk, pose_ik)):
        return row

    class_ij = int(pose_ij["right_yaw_class"])
    class_jk = int(pose_jk["right_yaw_class"])
    class_ik = int(pose_ik["right_yaw_class"])
    cycle = (class_ij + class_jk - class_ik) % CLASS_CYCLE_MODULUS
    rotation_ij = np.asarray(pose_ij["relative_rotation"], dtype=float)
    rotation_jk = np.asarray(pose_jk["relative_rotation"], dtype=float)
    rotation_ik = np.asarray(pose_ik["relative_rotation"], dtype=float)

    direction_ij = _unit(pose_ij["baseline_direction"])
    direction_jk = _unit(pose_jk["baseline_direction"])
    direction_ik = _unit(pose_ik["baseline_direction"])
    if any(value is None for value in (direction_ij, direction_jk, direction_ik)):
        return row

    # Loop i -> j -> k -> i, all expressed in anchor frame i. ``R(i,j)`` maps
    # frame-i coordinates to frame-j coordinates, so ``R(i,j)^T b(j,k)``
    # transports the j-frame direction back to frame i; the third side is the
    # negative of the i-anchored i->k direction.
    columns = np.column_stack(
        (direction_ij, rotation_ij.T @ direction_jk, -direction_ik)
    )
    closure = nonnegative_closure(columns)
    row.update(
        {
            "classes": [class_ij, class_jk, class_ik],
            "class_cycle_residual_mod4": int(cycle),
            "class_cycle_closed": bool(cycle == 0),
            "rotation_cycle_error_deg": angle_between(rotation_jk @ rotation_ij, rotation_ik),
            "closure_residual_normalized": closure["closure_residual_normalized"],
            "closure_residual_absolute": closure["closure_residual_absolute"],
            "sigma_min_unconstrained": closure["sigma_min_unconstrained"],
            "sigma_max": closure["sigma_max"],
            "nonnegativity_penalty": closure["nonnegativity_penalty"],
            "closure_support": closure["closure_support"],
            "closure_lengths_unit_norm": closure["closure_lengths_unit_norm"],
            "closure_residual_vector": closure["closure_residual_vector"],
            "closure_within_tolerance": bool(
                closure["closure_residual_normalized"] is not None
                and closure["closure_residual_normalized"] <= CLOSURE_RESIDUAL_TOLERANCE
            ),
        }
    )
    return row


def run_graph_diagnostics(out_dir: Path, frozen: dict) -> dict:
    """Phase 2a: connectivity/cycles/closure from frozen pair outputs only."""
    records = _pair_index(frozen)
    per_set: dict[str, dict] = {}
    triangles: list[dict] = []
    totals = {
        "sets": len(frozen["frame_sets"]),
        "frames": 0,
        "pairs": 0,
        "solved_edges": 0,
        "all_solved_triangles": 0,
        "class_cycle_closed": 0,
        "closure_within_tolerance": 0,
        "class_and_closure_closed": 0,
        "triangle_solved_edge_counts": {"0": 0, "1": 0, "2": 0, "3": 0},
        "two_solved_triangles": 0,
        "implied_missing_class_counts": {},
        "two_solved_missing_edge_reasons": {},
        "cycle_rank": 0,
    }

    for entry in frozen["frame_sets"]:
        label = entry["set"]
        fixture = entry["fixture"]
        frames = list(entry["frames"])
        set_triangles: list[dict] = []
        edges: list[tuple[str, str]] = []
        for left, right in itertools.combinations(frames, 2):
            statement = records[(fixture, left, right)]
            if statement["joint"]["solved"]:
                edges.append((left, right))
        components = _components(frames, edges)

        for left, middle, right in itertools.combinations(frames, 3):
            members = {
                "ij": records[(fixture, left, middle)],
                "jk": records[(fixture, middle, right)],
                "ik": records[(fixture, left, right)],
            }
            row = _triangle_diagnostics(left, middle, right, members)
            row["set"] = label
            set_triangles.append(row)
            triangles.append(row)

        all_solved = [row for row in set_triangles if row["all_solved"]]
        cycle_closed = [row for row in all_solved if row["class_cycle_closed"]]
        closure_ok = [row for row in all_solved if row["closure_within_tolerance"]]
        both = [
            row
            for row in all_solved
            if row["class_cycle_closed"] and row["closure_within_tolerance"]
        ]
        solved_edges = len(edges)
        complete_pairs = len(frames) * (len(frames) - 1) // 2
        solved_edge_counts = Counter(row["solved_edges"] for row in set_triangles)
        two_solved = [row for row in set_triangles if row["solved_edges"] == 2]
        implied_counts = Counter(
            row["implied_missing_class"]
            for row in two_solved
            if row["implied_missing_class"] is not None
        )
        missing_reasons = Counter(
            row["missing_edge_reason"] for row in two_solved if row["missing_edge_reason"]
        )
        cycle_rank = solved_edges - len(frames) + len(components)
        per_set[label] = {
            "fixture": fixture,
            "calibration_artifact": entry["calibration_artifact"],
            "frames": len(frames),
            "complete_pairs": complete_pairs,
            "solved_edges": solved_edges,
            "refused_edges": complete_pairs - solved_edges,
            "solving_fraction": solved_edges / complete_pairs if complete_pairs else None,
            "components": components,
            "component_sizes": [len(group) for group in components],
            "isolated_frames": [group[0] for group in components if len(group) == 1],
            "connected": len(components) == 1,
            "cycle_rank": cycle_rank,
            "is_forest": cycle_rank == 0,
            "triangles": len(set_triangles),
            "triangle_solved_edge_counts": {
                str(count): int(solved_edge_counts.get(count, 0)) for count in range(4)
            },
            "two_solved_triangles": len(two_solved),
            "implied_missing_class_counts": {
                str(key): int(value) for key, value in sorted(implied_counts.items())
            },
            "two_solved_missing_edge_reasons": dict(missing_reasons),
            "all_solved_triangles": len(all_solved),
            "class_cycle_closed": len(cycle_closed),
            "class_cycle_closed_fraction": (
                len(cycle_closed) / len(all_solved) if all_solved else None
            ),
            "closure_within_tolerance": len(closure_ok),
            "class_and_closure_closed": len(both),
        }
        totals["frames"] += len(frames)
        totals["pairs"] += complete_pairs
        totals["solved_edges"] += solved_edges
        totals["all_solved_triangles"] += len(all_solved)
        totals["class_cycle_closed"] += len(cycle_closed)
        totals["closure_within_tolerance"] += len(closure_ok)
        totals["class_and_closure_closed"] += len(both)
        totals["cycle_rank"] += cycle_rank
        totals["two_solved_triangles"] += len(two_solved)
        for key, value in implied_counts.items():
            key = str(key)
            totals["implied_missing_class_counts"][key] = (
                totals["implied_missing_class_counts"].get(key, 0) + int(value)
            )
        for key, value in missing_reasons.items():
            totals["two_solved_missing_edge_reasons"][key] = (
                totals["two_solved_missing_edge_reasons"].get(key, 0) + int(value)
            )
        for count in range(4):
            totals["triangle_solved_edge_counts"][str(count)] += int(
                solved_edge_counts.get(count, 0)
            )

    payload = {
        "schema": "ds15-graph-1",
        "phase": "graph_diagnostics_manifest_free",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "frozen_estimates": {
            "path": str((out_dir / "joint_estimates.json").relative_to(ROOT)),
            "sha256": digest(out_dir / "joint_estimates.json"),
            "canonical_pairs_digest": frozen.get("canonical_pairs_digest"),
        },
        "constants": {
            "class_cycle_modulus": CLASS_CYCLE_MODULUS,
            "closure_residual_tolerance_normalized": CLOSURE_RESIDUAL_TOLERANCE,
            "closure_scale_gauge": "||lengths||_2 = 1",
            "closure_direction_convention": (
                "loop i->j->k->i in anchor frame i; R(i,j)^T transports the "
                "j-anchored baseline into frame i; third side = -b(i,k)"
            ),
        },
        "totals": totals,
        "sets": per_set,
        "triangles": triangles,
        "note": (
            "Manifest-free. Every number here is derived only from the frozen "
            "joint_estimates.json pair outputs; no cameras.json, litematic, "
            "lattice or asserted geometry was opened. Pair outputs are never "
            "modified by these diagnostics."
        ),
    }
    graph_path = out_dir / "joint_graph_diagnostics.json"
    graph_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_triangles_csv(out_dir / "joint_triangles.csv", triangles)
    payload["_digest"] = digest(graph_path)
    payload["_path"] = str(graph_path.relative_to(ROOT))
    return payload


def _write_triangles_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "set",
        "i",
        "j",
        "k",
        "solved_edges",
        "all_solved",
        "classes",
        "class_cycle_residual_mod4",
        "class_cycle_closed",
        "missing_edge",
        "missing_edge_reason",
        "partial_classes",
        "implied_missing_class",
        "rotation_cycle_error_deg",
        "closure_residual_normalized",
        "closure_residual_absolute",
        "sigma_min_unconstrained",
        "sigma_max",
        "nonnegativity_penalty",
        "closure_support",
        "closure_within_tolerance",
        "closure_lengths_unit_norm",
        "closure_residual_vector",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


# --------------------------------------------------------------------------- #
# Phase 2b: reference evaluation (the only place a manifest is opened).
# --------------------------------------------------------------------------- #
def run_reference_evaluation(out_dir: Path, frozen: dict, graph: dict) -> dict:
    """Phase 2b: open manifests and score the already-frozen pair outputs."""
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
            "set": record["set"],
            "fixture": fixture,
            "left": left_name,
            "right": right_name,
            "joint_status": joint["status"],
            "joint_reason": joint["reason"],
            "support": joint["support"],
            "support_floor": joint["support_floor"],
            "candidate_edges": joint["candidate_edges"],
            "median_selected_residual_degrees": joint["median_selected_residual_degrees"],
            "parallax_degrees": joint["parallax_degrees"],
            "cheirality_fraction": joint["cheirality_fraction"],
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
            "pair_seconds": round(float(record["extraction_seconds"]), 4),
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
    class_correct = sum(1 for row in class_rows if row["class_correct"])
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
    refusal_reasons = Counter(row["joint_reason"] for row in rows if not row["solved"])

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
                Counter(row["joint_reason"] for row in set_rows if not row["solved"])
            ),
        }

    seconds = [float(record["extraction_seconds"]) for record in frozen["pairs"]]
    payload = {
        "schema": "ds15-1",
        "phase": "reference_evaluation",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_estimates": {
            "path": str(estimates_path.relative_to(ROOT)),
            "sha256": frozen_digest,
            "canonical_pairs_digest": frozen.get("canonical_pairs_digest"),
        },
        "graph_diagnostics": {
            "path": graph["_path"],
            "sha256": graph["_digest"],
            "canonical_pairs_digest": graph["frozen_estimates"]["canonical_pairs_digest"],
            "summary": graph["totals"],
            "sets": graph["sets"],
            "note": (
                "Written before any manifest was opened; see "
                "joint_graph_diagnostics.json and joint_triangles.csv."
            ),
        },
        "manifests": {
            fixture: {"path": entry["path"], "sha256": entry["sha256"]}
            for fixture, entry in manifests.items()
        },
        "summary": {
            "pairs": len(rows),
            "solved": len(solved_rows),
            "refused": len(rows) - len(solved_rows),
            "class_correct": class_correct,
            "false_solves": len(solved_rows) - class_correct,
            "precision_among_solves": (
                class_correct / len(solved_rows) if solved_rows else None
            ),
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
            "refusal_reasons": dict(refusal_reasons),
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
            "Evaluation only. The manifest is scaffolding and was opened only "
            "after joint_estimates.json and the manifest-free graph artifact were "
            "written; no matcher or solver parameter was changed after this "
            "comparison, and the graph diagnostics above are derived from the "
            "frozen pair outputs, never from the manifest. The baseline direction "
            "is compared after removing the four-fold yaw gauge the single-view "
            "calibrator leaves free."
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


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="stop after the blind all-pairs phase; do not open any manifest",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="compute manifest-free graph diagnostics from frozen estimates only",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="graph diagnostics plus reference evaluation from frozen estimates",
    )
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    selected = [args.extract_only, args.graph_only, args.evaluate_only]
    if sum(1 for value in selected if value) > 1:
        parser.error("--extract-only, --graph-only and --evaluate-only are mutually exclusive")

    estimates_path = args.out_dir / "joint_estimates.json"
    if args.graph_only or args.evaluate_only:
        if not estimates_path.exists():
            parser.error(f"no frozen estimates to digest: {estimates_path}")
        frozen = json.loads(estimates_path.read_text(encoding="utf-8"))
    else:
        frozen = run_extraction(args.out_dir)
        solved = sum(1 for record in frozen["pairs"] if record["joint"]["solved"])
        print(
            f"[blind] {len(frozen['pairs'])} unordered pairs over "
            f"{len(frozen['frame_sets'])} sets; solved {solved}; "
            f"canonical digest {frozen['canonical_pairs_digest']}; "
            f"frozen {frozen['_path']} sha256 {frozen['_digest']}"
        )
        if args.extract_only:
            return 0

    graph = run_graph_diagnostics(args.out_dir, frozen)
    print(
        f"[graph] solved edges {graph['totals']['solved_edges']}/"
        f"{graph['totals']['pairs']}; all-solved triangles "
        f"{graph['totals']['all_solved_triangles']}; class-cycle closed "
        f"{graph['totals']['class_cycle_closed']}; closure within tolerance "
        f"{graph['totals']['closure_within_tolerance']}; wrote "
        f"{graph['_path']} sha256 {graph['_digest']}"
    )
    if args.graph_only:
        return 0

    evaluation = run_reference_evaluation(args.out_dir, frozen, graph)
    summary = evaluation["summary"]
    print(
        f"[evaluate] solved {summary['solved']}/{summary['pairs']}; "
        f"class correct {summary['class_correct']}; "
        f"precision among solves {summary['precision_among_solves']}; "
        f"false solves {summary['false_solves']}; "
        f"median rotation error {summary['median_relative_rotation_error_deg']}"
    )
    print(f"[evaluate] wrote {evaluation['_path']} sha256 {evaluation['_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
