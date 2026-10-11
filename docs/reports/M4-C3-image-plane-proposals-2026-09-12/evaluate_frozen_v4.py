"""One-shot evaluation of frozen image-only plane proposals against F3 v4.

The production proposer is invoked on RGB screenshot pixels, the standard
derived GUI usable mask, and a fresh calibration estimated from those same
screenshots. The annotation is read only after the caller has frozen and
recorded the source/test hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARGETS = ROOT / "docs/reports/M4-C2-boundary-targets-v4-2026-09-12.json"
FREEZE = OUT / "pre_score_freeze.json"
FROZEN_SOURCES = (
    "src/mc_photogrammetry/reconstruction/boundary_candidates.py",
    "src/mc_photogrammetry/reconstruction/surfaces.py",
    "src/mc_photogrammetry/reconstruction/calibration.py",
    "tests/test_boundary_candidates.py",
    "tests/test_surfaces.py",
    "tests/test_calibration.py",
    "docs/reports/M4-C3-image-plane-proposals-2026-09-12/evaluate_frozen_v4.py",
)
FROZEN_TARGETS_SHA256 = "f4a0f86938561b3fc81b096fc614d1ae404c1372e65c925811c1368ce56a0da5"
FRAME_NAMES = (
    "2026-09-02_13.22.14.png",
    "2026-09-02_13.24.24.png",
    "2026-09-02_13.24.33.png",
)
SOURCE_IMAGE_SHA256 = {
    "2026-09-02_13.22.14.png": "eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46",
    "2026-09-02_13.24.24.png": "e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae",
    "2026-09-02_13.24.33.png": "35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc",
}
K_VALUES = (1, 5, 10, 32)
RECALL_CONTINUATION_FLOOR = 0.50
PRECISION_CONTINUATION_FLOOR = 0.50


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes():
    rows = {}
    aggregate = hashlib.sha256()
    for relative in FROZEN_SOURCES:
        raw = (ROOT / relative).read_bytes()
        digest = _sha256(raw)
        rows[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    return rows, aggregate.hexdigest()


def _metric(candidate, positive, domain, unknown, rank):
    from mc_photogrammetry.reconstruction.boundary_scoring import region_metrics

    return {
        "rank": rank,
        "axis": candidate.axis,
        "side": candidate.side,
        "strength": float(candidate.strength),
        "sources": list(candidate.sources),
        "coverage": float(candidate.coverage),
        "polygon": [[float(x), float(y)] for x, y in candidate.polygon],
        **region_metrics(candidate.mask, positive, domain, unknown),
    }


def _recall_at_k(rows):
    result = {}
    for requested_k in K_VALUES:
        considered = rows[:requested_k]
        recall = max((row["recall"] for row in considered), default=0.0)
        result[str(requested_k)] = {
            "max_single_proposal_recall": recall,
            "candidate_count_considered": len(considered),
            "recall_at_least_0_50": recall >= RECALL_CONTINUATION_FLOOR,
        }
    return result


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import cv2
    import numpy as np
    from mc_photogrammetry.reconstruction.boundary_candidates import propose_image_planes
    from mc_photogrammetry.reconstruction.boundary_scoring import target_masks
    from mc_photogrammetry.reconstruction.calibration import estimate_session
    from mc_photogrammetry.vision.gui_mask import create_gui_mask

    if not FREEZE.exists():
        raise SystemExit("pre-score freeze record is missing; refusing label access")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    source_hashes, aggregate_hash = _source_hashes()
    if (source_hashes != frozen.get("source_files_sha256")
            or aggregate_hash != frozen.get("pre_score_source_sha256")):
        raise SystemExit("pre-score source hash mismatch; refusing evaluation")
    target_bytes = TARGETS.read_bytes()
    target_digest = _sha256(target_bytes)
    if target_digest != FROZEN_TARGETS_SHA256:
        raise SystemExit(
            f"frozen v4 hash mismatch: expected {FROZEN_TARGETS_SHA256}, got {target_digest}")
    targets_by_frame = {item["frame"]: item for item in json.loads(target_bytes)["targets"]}
    if set(targets_by_frame) != set(FRAME_NAMES) or len(targets_by_frame) != 3:
        raise SystemExit("the frozen v4 target set is not the required three-frame set")

    images_rgb = {}
    images_grey = {}
    image_hashes = {}
    for frame in FRAME_NAMES:
        path = ROOT / "templates 2" / frame
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise SystemExit(f"could not load source screenshot: {frame}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        digest = _sha256(path.read_bytes())
        if SOURCE_IMAGE_SHA256.get(frame) != digest:
            raise SystemExit(f"source screenshot hash mismatch for {frame}")
        images_rgb[frame] = image_rgb
        images_grey[frame] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        image_hashes[frame] = digest

    lock_path = OUT / "evaluation_started.json"
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "target_sha256": target_digest,
                "frames": list(FRAME_NAMES),
                "mode": "single_frozen_evaluation",
            }, indent=2) + "\n")
    except FileExistsError as exc:
        raise SystemExit("one-shot evaluation marker already exists; refusing a second run") from exc

    calibrations = estimate_session(images_grey)
    frame_rows = []
    overlay_dir = OUT / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for frame in FRAME_NAMES:
        image = images_rgb[frame]
        height, width = image.shape[:2]
        domain = ~create_gui_mask(height, width)
        calibration = calibrations.get(frame)
        proposal_result = propose_image_planes(
            image, domain, calibration, max_candidates=32,
        )
        positive, unknown = target_masks(targets_by_frame[frame], width, height, domain)
        metrics = [
            _metric(candidate, positive, domain, unknown, rank)
            for rank, candidate in enumerate(proposal_result.proposals)
        ]
        top1 = metrics[0] if metrics else None
        best_iou = (max(metrics, key=lambda row: (row["iou"], -row["rank"]))
                    if metrics else None)
        best_recall = (max(metrics, key=lambda row: (row["recall"], row["iou"], -row["rank"]))
                       if metrics else None)
        calibration_record = None if calibration is None else {
            "focal_pixels": float(calibration.focal_pixels),
            "yaw_degrees_mod_90": float(calibration.yaw_degrees),
            "pitch_degrees": float(calibration.pitch_degrees),
            "roll_degrees": float(calibration.roll_degrees),
            "principal_point": (None if calibration.principal_point is None
                                else [float(value) for value in calibration.principal_point]),
            "inliers": int(calibration.inliers),
            "segments": int(calibration.segments),
            "median_residual": float(calibration.median_residual),
        }
        unknown_fraction = int(unknown.sum()) / max(int(domain.sum()), 1)
        frame_rows.append({
            "frame": frame,
            "target_id": targets_by_frame[frame]["target_id"],
            "source_image_sha256": image_hashes[frame],
            "calibration_from_image_pixels_only": calibration_record,
            "proposal_status": proposal_result.status,
            "proposal_reason": proposal_result.reason,
            "proposal_stats": proposal_result.stats,
            "candidate_count": len(metrics),
            "proposal_recall_at_k": _recall_at_k(metrics),
            "best_candidate_by_iou_evaluator_only": best_iou,
            "best_candidate_by_recall_evaluator_only": best_recall,
            "top1_algorithm_ranked": top1,
            "positive_area": int(positive.sum()),
            "unknown_area": int(unknown.sum()),
            "unknown_fraction_of_usable_domain": unknown_fraction,
            "candidates_union_performed": False,
            "candidate_metrics": metrics,
        })

        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        tint = overlay.copy()
        tint[positive] = (0, 150, 0)
        tint[unknown] = (0, 200, 220)
        overlay = cv2.addWeighted(tint, 0.30, overlay, 0.70, 0)
        for row in metrics:
            color = (0, 140, 255) if row["rank"] == 0 else (0, 220, 220)
            polygon = np.rint(np.asarray(row["polygon"], dtype=float)).astype(np.int32)
            if len(polygon) >= 3:
                cv2.polylines(overlay, [polygon.reshape((-1, 1, 2))], True,
                              color, 3 if row["rank"] == 0 else 1, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"{Path(frame).stem}_proposals.png"), overlay)

    recall_ok = all(
        row["best_candidate_by_recall_evaluator_only"] is not None
        and row["best_candidate_by_recall_evaluator_only"]["recall"]
        >= RECALL_CONTINUATION_FLOOR
        for row in frame_rows
    )
    precision_frames = sum(
        row["best_candidate_by_iou_evaluator_only"] is not None
        and row["best_candidate_by_iou_evaluator_only"]["precision"]
        >= PRECISION_CONTINUATION_FLOOR
        for row in frame_rows
    )
    all_have_candidate = all(row["candidate_count"] > 0 for row in frame_rows)
    no_cross_plane_union = all(not row["candidates_union_performed"] for row in frame_rows)
    continuation = bool(
        len(frame_rows) == 3 and all_have_candidate and recall_ok
        and precision_frames >= 2 and no_cross_plane_union
    )
    result = {
        "status": "meets_preregistered_diagnostic_continuation" if continuation
                  else "negative_preregistered_diagnostic",
        "targets_v4_sha256": target_digest,
        "pre_score_source_sha256": aggregate_hash,
        "source_inputs": "RGB screenshot pixels, derived usable GUI mask, and fresh image-derived Manhattan calibration only",
        "usable_mask": "logical not of create_gui_mask(height,width) with mask_debug_area default false",
        "candidate_cap_per_frame": 32,
        "proposal_recall_at_k_definition": "maximum individual candidate recall among the first K strength-ranked proposals; proposal masks are never unioned",
        "best_candidate_definition": "maximum IoU among returned candidates, evaluator-only tie-broken by lower rank",
        "best_recall_definition": "maximum individual recall among returned candidates; evaluator-only tie-broken by IoU then lower rank",
        "top1_definition": "rank zero from the image-only proposer, scored independently",
        "continuation_criteria": {
            "candidate_exists_all_three_frames": all_have_candidate,
            "best_individual_recall_at_least_0_50_all_three": recall_ok,
            "best_iou_candidate_precision_at_least_0_50_frames": precision_frames,
            "required_precision_frames": 2,
            "cross_plane_candidate_mask_unions_performed": 0,
            "passed": continuation,
        },
        "frames_total": 3,
        "frames": frame_rows,
    }
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "targets_v4_sha256": target_digest,
        "frames_total": 3,
        "continuation_criteria": result["continuation_criteria"],
        "frames": [{
            "frame": row["frame"],
            "candidate_count": row["candidate_count"],
            "proposal_recall_at_k": {
                key: value["max_single_proposal_recall"]
                for key, value in row["proposal_recall_at_k"].items()
            },
            "best_candidate_by_iou": None if row["best_candidate_by_iou_evaluator_only"] is None
                else {key: row["best_candidate_by_iou_evaluator_only"][key]
                      for key in ("rank", "precision", "recall", "iou")},
            "best_candidate_by_recall": None if row["best_candidate_by_recall_evaluator_only"] is None
                else {key: row["best_candidate_by_recall_evaluator_only"][key]
                      for key in ("rank", "precision", "recall", "iou")},
            "top1": None if row["top1_algorithm_ranked"] is None
                else {key: row["top1_algorithm_ranked"][key]
                      for key in ("rank", "precision", "recall", "iou")},
        } for row in frame_rows],
    }, indent=2))


if __name__ == "__main__":
    main()
