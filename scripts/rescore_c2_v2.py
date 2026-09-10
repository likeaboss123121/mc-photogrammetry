"""M4 E1: re-score the already-saved C2 candidate polygons against the
corrected v2 annotation - a labeled evaluator correction, not a new
proposer run.

Per the lead follow-up review: "re-score the already saved C2 candidate
polygons once against the corrected annotation as a labeled evaluator
correction. Preserve original targets/results. No new C3 proposer or cue
is assigned this turn." This script never calls `find_chains` or
`propose_boundary_candidates` - it reads the candidate polygons
`boundary_pilot_c2.py` already wrote to
`M4-C2-boundary-pilot-2026-09-09/results.json` and re-computes
`boundary_scoring.region_metrics`/`boundary_error` for each one against
`M4-C2-boundary-targets-v2-2026-09-09.json` instead of the original
target file. Both the original results.json and the original v1 target
file are left untouched.

    python scripts/rescore_c2_v2.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ORIGINAL_RESULTS_PATH = ROOT / "docs/reports/M4-C2-boundary-pilot-2026-09-09/results.json"
TARGETS_V2_PATH = ROOT / "docs/reports/M4-C2-boundary-targets-v2-2026-09-09.json"
FROZEN_SHA256 = "193086408c57ad75e145c9636f34b8bda4b1adf7b3fd8e49ff22f9fa475cd574"
FIXTURE = ROOT / "templates 2"

PRECISION_GATE = 0.90
RECALL_GATE = 0.50


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = TARGETS_V2_PATH.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_SHA256:
        raise SystemExit(
            f"v2 target file changed since it was frozen: expected {FROZEN_SHA256}, "
            f"got {actual_hash}. Not re-scoring against a moved target.")
    targets_by_frame = {t["frame"]: t for t in json.loads(raw)["targets"]}
    print(f"v2 targets frozen and verified: {FROZEN_SHA256}")

    if not ORIGINAL_RESULTS_PATH.exists():
        raise SystemExit(f"original C2 results not found at {ORIGINAL_RESULTS_PATH}")
    original = json.loads(ORIGINAL_RESULTS_PATH.read_text(encoding="utf-8"))

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.boundary_scoring import (
        boundary_error, fill_polygon, region_metrics, target_masks, usable_domain,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_exclude = original["top_exclude_pixels"]
    bottom_exclude = original["bottom_exclude_pixels"]

    frame_results = []
    for original_frame in original["frames"]:
        frame_id = original_frame["frame"]
        target = targets_by_frame.get(frame_id)
        if target is None or "top1" not in original_frame:
            frame_results.append({"frame": frame_id, "note": "no target/candidates to re-score"})
            continue
        if original_frame.get("top1") is None:
            frame_results.append({"frame": frame_id, "note": "original C2 run had no candidates",
                                  "gate_passed": False})
            print(f"{frame_id}: no candidates in the original C2 run - nothing to re-score")
            continue
        width, height = target["image_size"]
        domain = usable_domain(width, height, top_exclude=top_exclude, bottom_exclude=bottom_exclude)
        positive_mask, unknown_mask = target_masks(target, width, height, domain)

        rescored_candidates = []
        for candidate in original_frame["candidates"]:
            candidate_mask = fill_polygon(candidate["polygon"], width, height)
            metrics = region_metrics(candidate_mask, positive_mask, domain, unknown_mask)
            error = boundary_error(candidate_mask, positive_mask, domain, unknown_mask,
                                   width, height)
            rescored_candidates.append({
                "rank": candidate["rank"], "strength": candidate["strength"],
                "complete": candidate.get("complete"), "polygon": candidate["polygon"],
                **metrics, "normalized_boundary_error": error,
            })

        top1 = next((c for c in rescored_candidates if c["rank"] == original_frame["top1"]["rank"]),
                    None)
        best_matched = max(rescored_candidates, key=lambda c: c["iou"]) if rescored_candidates else None
        passed = bool(top1 and top1["precision"] >= PRECISION_GATE and top1["recall"] >= RECALL_GATE)

        image_path = FIXTURE / frame_id
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        overlay = bgr.copy()
        overlay[positive_mask] = (0, 200, 0)
        overlay[unknown_mask] = (0, 200, 200)
        negative_domain = domain & ~positive_mask & ~unknown_mask
        overlay[negative_domain] = (0, 0, 200)
        blended = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)

        def draw(polygon, color, thickness):
            points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(blended, [points], True, color, thickness, cv2.LINE_AA)
        for record in rescored_candidates:
            if top1 is not None and record["rank"] == top1["rank"]:
                continue
            draw(record["polygon"], (0, 200, 220), 1)
        if top1 is not None:
            draw(top1["polygon"], (0, 140, 255), 3)
        label = (f"v2-rescored top1: P={top1['precision']:.2f} R={top1['recall']:.2f} "
                f"IoU={top1['iou']:.2f} {'PASS' if passed else 'fail'}"
                if top1 else "no candidates")
        cv2.putText(blended, label, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (0, 140, 255) if passed else (0, 0, 220), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"overlay_v2_{frame_id}"), blended)

        old_top1 = original_frame["top1"]
        frame_results.append({
            "frame": frame_id, "top1": top1, "best_matched_evaluator_only": best_matched,
            "candidates": rescored_candidates, "gate_passed": passed,
            "v1_top1_for_comparison": {
                "precision": old_top1["precision"], "recall": old_top1["recall"],
                "iou": old_top1["iou"],
            },
        })
        print(f"{frame_id}: v1 top1 P/R/IoU={old_top1['precision']:.3f}/"
             f"{old_top1['recall']:.3f}/{old_top1['iou']:.3f} -> v2-rescored "
             f"P/R/IoU={top1['precision']:.3f}/{top1['recall']:.3f}/{top1['iou']:.3f} "
             f"boundary_err={top1['normalized_boundary_error']:.4f} "
             f"-> {'PASS' if passed else 'fail'} (best_matched IoU={best_matched['iou']:.3f})")

    scored = [f for f in frame_results if "top1" in f]
    passed_count = sum(1 for f in scored if f["gate_passed"])
    continuation = passed_count >= 2 and len(scored) >= 2
    print(f"v2-rescored local continuation condition: {passed_count} of {len(scored)} "
         f"scored frames passed -> {'MET' if continuation else 'NOT MET'}")

    results = {
        "note": "Re-scores boundary_pilot_c2.py's already-saved candidate polygons "
                "(read from M4-C2-boundary-pilot-2026-09-09/results.json, never "
                "regenerated) against the corrected v2 annotation. No proposer code "
                "ran in this script.",
        "targets_v2_sha256": FROZEN_SHA256,
        "original_results_path": str(ORIGINAL_RESULTS_PATH.relative_to(ROOT)),
        "precision_gate": PRECISION_GATE, "recall_gate": RECALL_GATE,
        "frames": frame_results, "frames_passed": passed_count,
        "continuation_condition_met": continuation,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote results.json and overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
