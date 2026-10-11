"""Frozen real-pair correspondence diagnostic for the connected-camera stack.

This script has two phases with a hard separation between them, because the
M3 pair solvers can only be judged if the estimate was made before the answer
was known:

1. **Extraction and estimation (blind).** Read a frozen list of image pairs from
   Template 2, Template 6 and the frozen ``t3-sparse-8`` subset, load the RGB
   frames, and take each frame's camera from the *image-derived* M3 calibration
   artifact. Run SIFT correspondence extraction (manifest-free, camera-free) and,
   for every pair whose correspondence set is feedable, feed it to DS-11's
   :func:`~mc_photogrammetry.pipeline.connected_camera.solve_pair`. Write
   ``estimates.json`` and ``estimates.csv``. No ``cameras.json``, reference
   litematic, lattice or asserted metadata is opened in this phase.

2. **Reference evaluation (separate).** Only after phase 1 has been written and
   its SHA-256 recorded, load the fixture manifests and compare the solved
   relative yaw class and baseline direction. Write ``evaluation.json`` and
   ``evaluation.csv``. The comparison is labelled as evaluation scaffolding; no
   extraction or solver parameter is changed after this point.

The frozen pair list is fixed by a rule stated here, not chosen by outcome: for
each fixture's calibration artifact, take the first two adjacent frame pairs in
sorted-filename order. Nothing about a reconstruction result was consulted.

Run with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_connected_pairs.py

``--extract-only`` stops after phase 1, which is the honest way to inspect the
frozen estimate before deciding whether to open the manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair  # noqa: E402
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import (  # noqa: E402
    CorrespondenceSettings,
    PairMatches,
    extract_pair_matches,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    camera_rotation_matrix,
)

OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds12"
REPORTS = ROOT / "docs" / "reports"

# Frozen pairs: first two adjacent pairs, sorted filename order, per fixture.
# The calibration artifacts are image-derived (scripts/calibrate_frames.py); they
# carry no manifest data. ``fixture`` names the screenshot folder.
PAIR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "templates 2",
        "M3-calibration-templates-2.json",
        "2026-09-02_13.22.14.png",
        "2026-09-02_13.24.24.png",
    ),
    (
        "templates 2",
        "M3-calibration-templates-2.json",
        "2026-09-02_13.24.24.png",
        "2026-09-02_13.24.33.png",
    ),
    (
        "templates 6",
        "M3-calibration-templates-6.json",
        "2026-09-05_18.01.09.png",
        "2026-09-05_18.01.14.png",
    ),
    (
        "templates 6",
        "M3-calibration-templates-6.json",
        "2026-09-05_18.01.14.png",
        "2026-09-05_18.01.18.png",
    ),
    (
        "templates 3",
        "M3-calibration-templates-3-t3-sparse-8.json",
        "2026-09-04_20.03.32.png",
        "2026-09-04_20.03.55.png",
    ),
    (
        "templates 3",
        "M3-calibration-templates-3-t3-sparse-8.json",
        "2026-09-04_20.03.55.png",
        "2026-09-04_20.04.18.png",
    ),
)

_GAUGE_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True, slots=True)
class _Estimate:
    """The image-derived camera summary carried into the frozen artifact."""

    frame: str
    width: int
    height: int
    focal_pixels: float
    principal_point: tuple[float, float]
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def source_revision() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    trace = np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def load_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_estimates(calibration_path: Path) -> dict[str, _Estimate]:
    """Image-derived per-frame estimates; this never touches a manifest."""
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    sizes = {name: row["size"] for name, row in payload["inputs"].items()}
    estimates: dict[str, _Estimate] = {}
    for row in payload["estimates"]:
        frame = row["frame"]
        width, height = (int(value) for value in sizes[frame])
        focal = float(row["focal_pixels"])
        principal = row.get("principal_point")
        centre = (
            (float(principal[0]), float(principal[1]))
            if principal
            else (width / 2.0, height / 2.0)
        )
        estimates[frame] = _Estimate(
            frame=frame,
            width=width,
            height=height,
            focal_pixels=focal,
            principal_point=centre,
            yaw_degrees=float(row["yaw_degrees_mod90"]),
            pitch_degrees=float(row["pitch_degrees"]),
            roll_degrees=float(row.get("roll_degrees") or 0.0),
        )
    return estimates


def to_camera_estimate(estimate: _Estimate) -> CameraEstimate:
    intrinsics = CameraIntrinsics(
        image_size=(estimate.width, estimate.height),
        focal_length_pixels=(estimate.focal_pixels, estimate.focal_pixels),
        principal_point_pixels=estimate.principal_point,
    )
    pose = CameraPose(
        position=None,
        yaw_degrees=estimate.yaw_degrees,
        pitch_degrees=estimate.pitch_degrees,
        roll_degrees=estimate.roll_degrees,
    )
    return CameraEstimate(intrinsics=intrinsics, pose=pose)


def to_camera_model(estimate: _Estimate, *, yaw_degrees: float | None = None) -> CameraModel:
    """A manifest-free camera for the public rotation API (position is irrelevant)."""
    return CameraModel(
        position=(0.0, 0.0, 0.0),
        yaw_degrees=estimate.yaw_degrees if yaw_degrees is None else yaw_degrees,
        pitch_degrees=estimate.pitch_degrees,
        roll_degrees=estimate.roll_degrees,
        eye_height=0.0,
    )


def _correspondence_record(result: PairMatches) -> dict:
    left, right = result.left, result.right
    return {
        "status": result.status,
        "reason": result.reason,
        "feedable": result.feedable,
        "matches": int(result.matches.shape[0]),
        "raw_keypoints": {"left": len(left.points), "right": len(right.points)},
        "usable_pixels": {"left": left.usable_pixels, "right": right.usable_pixels},
        "ratio_survivors": {
            "left": result.selection.ratio_left,
            "right": result.selection.ratio_right,
        },
        "mutual_survivors": result.selection.mutual,
        "mutual_fraction": result.mutual_fraction,
        "discarded_non_finite": result.selection.discarded_non_finite,
        "discarded_duplicate": result.selection.discarded_duplicate,
        "spatial_bins": {
            "left": result.spatial_bins_left,
            "right": result.spatial_bins_right,
            "total": result.spatial_bins_total,
            "coverage": result.spatial_coverage,
        },
        "descriptor_distance": {
            "min": result.selection.distance_min,
            "median": result.selection.distance_median,
            "mean": result.selection.distance_mean,
            "max": result.selection.distance_max,
        },
    }


def _class_record(score) -> dict:
    return {
        "yaw_class": score.yaw_class,
        "passed": score.passed,
        "refusal": score.refusal,
        "inliers": score.inliers,
        "informative": score.informative,
        "inlier_fraction": score.inlier_fraction,
        "median_rho_degrees": score.median_rho_degrees,
        "parallax_degrees": score.parallax_degrees,
        "cheirality_fraction": score.cheirality_fraction,
        "singular_ratio": score.singular_ratio,
    }


def _solve_record(pose) -> dict:
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
            list(pose.baseline_direction)
            if pose.baseline_direction is not None
            else None
        ),
        "inliers": pose.inliers,
        "inlier_fraction": pose.inlier_fraction,
        "median_rho_degrees": pose.median_rho_degrees,
        "parallax_degrees": pose.parallax_degrees,
        "cheirality_fraction": pose.cheirality_fraction,
        "classes": [_class_record(score) for score in pose.classes],
    }


def run_extraction(out_dir: Path) -> dict:
    """Phase 1: extract correspondences and solve pairs with no manifest access."""
    settings = CorrespondenceSettings()
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
        record: dict = {
            "fixture": fixture,
            "calibration_artifact": str(calibration_path.relative_to(ROOT)),
            "left": left_name,
            "right": right_name,
            "left_estimate": asdict(estimates[left_name]),
            "right_estimate": asdict(estimates[right_name]),
        }
        started = time.perf_counter()
        result = extract_pair_matches(
            left_name,
            left_rgb,
            right_name,
            right_rgb,
            usable_masks=None,
            settings=settings,
        )
        extracted = time.perf_counter() - started

        record["correspondence"] = _correspondence_record(result)
        if result.feedable:
            solve_started = time.perf_counter()
            pose = solve_pair(
                to_camera_estimate(estimates[left_name]),
                to_camera_estimate(estimates[right_name]),
                result.matches,
                left_id=left_name,
                right_id=right_name,
                gates=gates,
            )
            record["solve"] = _solve_record(pose)
            record["solve_seconds"] = time.perf_counter() - solve_started
        else:
            record["solve"] = {
                "solved": False,
                "reason": f"correspondence_{result.status}",
                "right_yaw_class": None,
                "relative_rotation": None,
                "baseline_direction": None,
                "inliers": 0,
                "inlier_fraction": 0.0,
                "median_rho_degrees": None,
                "parallax_degrees": None,
                "cheirality_fraction": None,
                "classes": [],
            }
            record["solve_seconds"] = 0.0
        record["extraction_seconds"] = extracted
        record["matches"] = result.matches.tolist()
        pairs.append(record)

    payload = {
        "schema": "ds12-1",
        "phase": "extraction_and_estimation_blind",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "settings": {
            "correspondence": asdict(settings),
            "pair_gates": asdict(gates),
            "usable_masks": "None (derived ~create_gui_mask per frame)",
        },
        "calibration_artifacts": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted(estimates_cache)
        },
        "inputs": input_records,
        "pairs": pairs,
        "recipe": " ".join(sys.argv),
        "note": (
            "No cameras.json manifest, reference litematic, lattice, coordinate or "
            "asserted geometry was opened while producing this file. The reference "
            "evaluation runs only after it is written; evaluation.json records the "
            "SHA-256 of these exact bytes."
        ),
    }
    estimates_path = out_dir / "estimates.json"
    estimates_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_estimates_csv(out_dir / "estimates.csv", pairs)
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
        "matches",
        "raw_left",
        "raw_right",
        "mutual",
        "spatial_coverage",
        "solved",
        "solve_reason",
        "right_yaw_class",
        "extraction_seconds",
        "solve_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in pairs:
            correspondence = record["correspondence"]
            solve = record["solve"]
            writer.writerow(
                {
                    "fixture": record["fixture"],
                    "left": record["left"],
                    "right": record["right"],
                    "status": correspondence["status"],
                    "reason": correspondence["reason"],
                    "matches": correspondence["matches"],
                    "raw_left": correspondence["raw_keypoints"]["left"],
                    "raw_right": correspondence["raw_keypoints"]["right"],
                    "mutual": correspondence["mutual_survivors"],
                    "spatial_coverage": round(correspondence["spatial_bins"]["coverage"], 4),
                    "solved": solve["solved"],
                    "solve_reason": solve["reason"],
                    "right_yaw_class": solve["right_yaw_class"],
                    "extraction_seconds": round(record["extraction_seconds"], 3),
                    "solve_seconds": round(record["solve_seconds"], 3),
                }
            )


def _estimate_from_record(record: dict) -> _Estimate:
    return _Estimate(**record)


def _reference_class(
    left: _Estimate,
    right: _Estimate,
    truth_left: CameraModel,
    truth_right: CameraModel,
) -> tuple[int, int]:
    """Reference yaw class two ways: manifest floors and best public-API match."""
    true_rotation = camera_rotation_matrix(truth_right) @ camera_rotation_matrix(truth_left).T
    left_rotation = camera_rotation_matrix(to_camera_model(left))
    errors = {}
    for quadrant in range(4):
        candidate = (
            camera_rotation_matrix(to_camera_model(right, yaw_degrees=right.yaw_degrees + 90.0 * quadrant))
            @ left_rotation.T
        )
        errors[quadrant] = angle_between(candidate, true_rotation)
    direct = min(errors, key=lambda quadrant: (errors[quadrant], quadrant))
    floors = (
        int(np.floor(truth_right.yaw_degrees / 90.0))
        - int(np.floor(truth_left.yaw_degrees / 90.0))
    ) % 4
    return direct, floors


def run_evaluation(out_dir: Path, frozen: dict) -> dict:
    """Phase 2: the only place a manifest is opened."""
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest

    estimates_path = out_dir / "estimates.json"
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
        solved = record["solve"]["solved"]
        row = {
            "fixture": fixture,
            "left": left_name,
            "right": right_name,
            "correspondence_status": record["correspondence"]["status"],
            "matches": record["correspondence"]["matches"],
            "solved": solved,
            "solve_reason": record["solve"]["reason"],
            "solved_yaw_class": record["solve"]["right_yaw_class"],
            "reference_yaw_class": class_direct,
            "reference_yaw_class_floors": class_floors,
            "class_correct": (
                bool(solved and record["solve"]["right_yaw_class"] == class_direct)
            ),
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
        if solved and record["solve"]["relative_rotation"] is not None:
            true_rotation = (
                camera_rotation_matrix(truth_right) @ camera_rotation_matrix(truth_left).T
            )
            got_rotation = np.asarray(record["solve"]["relative_rotation"], dtype=float)
            row["relative_rotation_error_deg"] = angle_between(got_rotation, true_rotation)

            baseline_true = (
                np.asarray(truth_right.position, dtype=float)
                - np.asarray(truth_left.position, dtype=float)
            )
            length = float(np.linalg.norm(baseline_true))
            if length > 1e-9:
                baseline_true = baseline_true / length
                baseline_got = np.asarray(record["solve"]["baseline_direction"], dtype=float)
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
                                float(
                                    np.clip(
                                        np.dot(candidate, baseline_got), -1.0, 1.0
                                    )
                                )
                            )
                        )
                    )
                best = min(errors, key=lambda step: (errors[step], step))
                row["gauge_index"] = best // 90
                row["baseline_error_deg"] = errors[best]
                row["baseline_error_without_gauge_deg"] = errors[0]
                cosine = float(np.clip(np.dot(baseline_true, baseline_got), -1.0, 1.0))
                row["baseline_error_signless_deg"] = float(
                    np.degrees(np.arccos(abs(cosine)))
                )
        rows.append(row)

    solved_rows = [row for row in rows if row["solved"]]
    class_rows = [row for row in solved_rows if row["solved_yaw_class"] is not None]
    rotation_errors = [
        row["relative_rotation_error_deg"]
        for row in solved_rows
        if row["relative_rotation_error_deg"] is not None
    ]
    baseline_errors = [
        row["baseline_error_deg"]
        for row in solved_rows
        if row["baseline_error_deg"] is not None
    ]
    payload = {
        "schema": "ds12-1",
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
            "correspondence_feedable": sum(
                1 for row in rows if row["correspondence_status"] == "ok"
            ),
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
            "estimates.json was written; no correspondence or solver threshold was "
            "changed after this comparison. The baseline direction is compared after "
            "removing the four-fold yaw gauge the single-view calibrator leaves free."
        ),
        "pairs": rows,
    }
    evaluation_path = out_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_evaluation_csv(out_dir / "evaluation.csv", rows)
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
        help="evaluate the existing frozen estimates.json without re-extracting",
    )
    args = parser.parse_args(argv)
    if args.extract_only and args.evaluate_only:
        parser.error("--extract-only and --evaluate-only are mutually exclusive")

    if args.evaluate_only:
        estimates_path = args.out_dir / "estimates.json"
        if not estimates_path.exists():
            parser.error(f"no frozen estimates to evaluate: {estimates_path}")
        frozen = json.loads(estimates_path.read_text(encoding="utf-8"))
    else:
        frozen = run_extraction(args.out_dir)
        print(
            f"[blind] {len(frozen['pairs'])} pairs; "
            f"feedable {sum(1 for p in frozen['pairs'] if p['correspondence']['feedable'])}; "
            f"solved {sum(1 for p in frozen['pairs'] if p['solve']['solved'])}; "
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
