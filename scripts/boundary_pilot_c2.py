"""M4 C2: corrected structural-boundary proposal run, per the lead follow-up
review's S1/S2 assignment (`LEAD-review-2026-09-09-followup.md`).

Supersedes `boundary_pilot.py` (preserved as an "invalidly targeted
historical diagnostic," per the review - its evaluation target described a
safe strip inside the visible wall, not the wall's boundary, and its
proposer paired role-by-image-centre edges across different world axes).
This run:

- Uses the corrected `boundary_candidates.py` (S1): opposite edges paired
  within one calibrated axis's own pencil, degenerate/self-intersecting/
  unsupported combinations rejected rather than substituted, chains and
  candidates distributed across axes and image regions rather than a pure
  global top-N.
- Scores against `M4-C2-boundary-targets-2026-09-09.json` (S2): the wall's
  full visible extent, with explicit positive/negative/uncertain regions -
  uncertain pixels excluded from scoring in both directions, never treated
  as a negative just because they were not confidently labelled positive.
- Derives its frame list and each image's own width/height independently
  from the calibration artifact and the loaded image itself - never from
  the evaluation target file, which the proposal path never opens.
- Crops to the usable domain (excludes the F3 HUD band and the hotbar/
  screenshot-notice band) *before* feature extraction, so the mask
  boundary itself cannot manufacture a detectable line the way blacking
  out those bands in place would.
- Reports the algorithm's own top-ranked candidate (`top1`, by supporting
  strength alone) separately from `best_matched` (the highest-IoU
  candidate among the full returned set) - `best_matched` is an evaluator-
  only diagnostic ("is a correct answer anywhere in the candidate set"),
  never a stand-in for what the proposer would actually output, and the
  local continuation gate is evaluated on `top1` alone.

    python scripts/boundary_pilot_c2.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGETS_PATH = ROOT / "docs/reports/M4-C2-boundary-targets-2026-09-09.json"
FROZEN_SHA256 = "2b116093a253cc46e4f173e54694ff37c4417c380ee5369d6f8cb489ea982306"
CALIBRATION_PATH = ROOT / "docs/reports/M3-calibration-templates-2.json"
FIXTURE = ROOT / "templates 2"

TOP_EXCLUDE_PIXELS = 130
BOTTOM_EXCLUDE_PIXELS = 130
"""Widened from C1's 120 to comfortably clear the third frame's
"Saved screenshot as ..." line as well as the hotbar, since this run
crops to the domain rather than only masking it at scoring time."""

PRECISION_GATE = 0.90
RECALL_GATE = 0.50


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shift_chain(chain, dy: float):
    return replace(chain, point=(chain.point[0], chain.point[1] + dy))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = TARGETS_PATH.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_SHA256:
        raise SystemExit(
            f"target file changed since it was frozen: expected {FROZEN_SHA256}, "
            f"got {actual_hash}. Not scoring a moved target.")
    targets_by_frame = {t["frame"]: t for t in json.loads(raw)["targets"]}
    print(f"targets frozen and verified: {FROZEN_SHA256}")

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.boundary_candidates import (
        find_chains, propose_boundary_candidates,
    )
    from mc_photogrammetry.reconstruction.boundary_scoring import (
        boundary_error, fill_polygon, region_metrics, target_masks, usable_domain,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    calibration_hash = _file_hash(CALIBRATION_PATH)
    # The frame list and every image's own dimensions come from the
    # calibration artifact and the loaded file - never from the
    # evaluation-target file, which stays unopened until scoring.
    frame_inventory = [entry["frame"] for entry in calibration["estimates"]]

    frame_results = []
    for frame_id in frame_inventory:
        image_path = FIXTURE / frame_id
        source_hash = _file_hash(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read {image_path}")
        height, width = bgr.shape[:2]
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        domain = usable_domain(width, height, top_exclude=TOP_EXCLUDE_PIXELS,
                               bottom_exclude=BOTTOM_EXCLUDE_PIXELS)

        entry = next(e for e in calibration["estimates"] if e["frame"] == frame_id)
        principal_point = entry.get("principal_point")
        principal_point = tuple(principal_point) if principal_point else None
        if principal_point is not None:
            # Cropping the top band shifts the optical centre's y-coordinate
            # by exactly the crop offset; an unshifted principal point would
            # silently mis-calibrate every cropped-frame detection.
            principal_point = (principal_point[0], principal_point[1] - TOP_EXCLUDE_PIXELS)

        # Crop to the usable domain *before* detection - masking it in
        # place instead would draw a rectangle whose own edge is a strong,
        # fully artificial line for the detector to find.
        cropped = image[TOP_EXCLUDE_PIXELS: height - BOTTOM_EXCLUDE_PIXELS, :, :]
        chains = find_chains(
            cropped, focal=entry["focal_pixels"], yaw_degrees=entry["yaw_degrees_mod90"],
            pitch_degrees=entry["pitch_degrees"], roll_degrees=entry.get("roll_degrees", 0.0),
            principal_point=principal_point,
        )
        chains = [_shift_chain(chain, float(TOP_EXCLUDE_PIXELS)) for chain in chains]

        stats: dict = {}
        candidates = propose_boundary_candidates(chains, width, height, stats=stats)

        target = targets_by_frame.get(frame_id)
        if target is None:
            frame_results.append({"frame": frame_id, "source_image_sha256": source_hash,
                                  "chain_count": len(chains), "proposal_stats": stats,
                                  "note": "no frozen target for this frame"})
            continue

        positive_mask, unknown_mask = target_masks(target, width, height, domain)

        candidate_records = []
        for rank, candidate in enumerate(candidates):
            candidate_mask = fill_polygon(candidate.polygon, width, height)
            metrics = region_metrics(candidate_mask, positive_mask, domain, unknown_mask)
            error = boundary_error(candidate_mask, positive_mask, domain, unknown_mask,
                                   width, height)
            candidate_records.append({
                "rank": rank, "strength": candidate.strength, "complete": candidate.complete,
                "axis_pair": list(candidate.axis_pair), "polygon": candidate.polygon,
                **metrics, "normalized_boundary_error": error,
            })

        top1 = candidate_records[0] if candidate_records else None
        best_matched = (max(candidate_records, key=lambda record: record["iou"])
                        if candidate_records else None)
        passed = bool(top1 and top1["precision"] >= PRECISION_GATE
                      and top1["recall"] >= RECALL_GATE)

        overlay = bgr.copy()
        def draw(polygon, color, thickness):
            points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [points], True, color, thickness, cv2.LINE_AA)
        draw(target["positive_polygon"], (0, 220, 0), 3)
        for polygon in target.get("uncertain_polygons", []):
            draw(polygon, (0, 220, 220), 2)
        for group in target.get("negative_polygons", {}).values():
            for polygon in group:
                draw(polygon, (0, 0, 220), 1)
        for record in candidate_records:
            if top1 is not None and record["rank"] == top1["rank"]:
                continue
            draw(record["polygon"], (0, 200, 220), 1)
        if top1 is not None:
            draw(top1["polygon"], (0, 140, 255), 3)
        if best_matched is not None and best_matched["rank"] != (top1["rank"] if top1 else -1):
            draw(best_matched["polygon"], (255, 0, 180), 2)
        cv2.line(overlay, (0, TOP_EXCLUDE_PIXELS), (width, TOP_EXCLUDE_PIXELS), (128, 128, 128), 1)
        cv2.line(overlay, (0, height - BOTTOM_EXCLUDE_PIXELS),
                 (width, height - BOTTOM_EXCLUDE_PIXELS), (128, 128, 128), 1)
        label = (f"top1: P={top1['precision']:.2f} R={top1['recall']:.2f} "
                f"IoU={top1['iou']:.2f} {'PASS' if passed else 'fail'}"
                if top1 else "no candidates")
        cv2.putText(overlay, label, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (0, 140, 255) if passed else (0, 0, 220), 2, cv2.LINE_AA)
        overlay_path = args.out_dir / f"overlay_{frame_id}"
        cv2.imwrite(str(overlay_path), overlay)

        frame_results.append({
            "frame": frame_id, "source_image_sha256": source_hash,
            "chain_count": len(chains), "proposal_stats": stats,
            "excluded_unknown_area": int(unknown_mask.sum()),
            "positive_area": int(positive_mask.sum()),
            "top1": top1, "best_matched_evaluator_only": best_matched,
            "candidates": candidate_records, "gate_passed": passed,
        })
        if top1:
            print(f"{frame_id}: {len(chains)} chains, {stats['accepted']} accepted / "
                 f"{stats['combinations_tried']} combinations tried, "
                 f"{len(candidates)} candidates returned; top1 P={top1['precision']:.3f} "
                 f"R={top1['recall']:.3f} IoU={top1['iou']:.3f} "
                 f"boundary_err={top1['normalized_boundary_error']:.4f} "
                 f"-> {'PASS' if passed else 'fail'} "
                 f"(best_matched IoU={best_matched['iou']:.3f}, evaluator-only)")
        else:
            print(f"{frame_id}: no candidates")

    scored = [f for f in frame_results if "top1" in f]
    passed_count = sum(1 for f in scored if f["gate_passed"])
    continuation = passed_count >= 2 and len(scored) >= 2
    print(f"local continuation condition (>=2 of {len(scored)} scored frames at "
         f"P>=0.90,R>=0.50 on top1): {passed_count} passed -> "
         f"{'MET' if continuation else 'NOT MET'}")

    results = {
        "targets_sha256": FROZEN_SHA256, "calibration_artifact_sha256": calibration_hash,
        "precision_gate": PRECISION_GATE, "recall_gate": RECALL_GATE,
        "top_exclude_pixels": TOP_EXCLUDE_PIXELS, "bottom_exclude_pixels": BOTTOM_EXCLUDE_PIXELS,
        "note": "top1 is the algorithm's own highest-strength candidate and is what the "
                "continuation gate is evaluated on. best_matched_evaluator_only is the "
                "highest-IoU candidate anywhere in the returned set - a diagnostic for "
                "whether a correct answer exists among the alternatives, never a stand-in "
                "for the proposer's actual output and never used to select it.",
        "frames": frame_results, "frames_passed": passed_count,
        "continuation_condition_met": continuation,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote results.json and overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
