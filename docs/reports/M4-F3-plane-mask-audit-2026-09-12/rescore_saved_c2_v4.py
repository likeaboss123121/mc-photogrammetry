"""Re-score only the saved C2 polygons against the frozen v4 annotation.

This script does not call the boundary proposer or generate candidates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "rescored_c2"
TARGETS = ROOT / "docs/reports/M4-C2-boundary-targets-v4-2026-09-12.json"
SAVED = ROOT / "docs/reports/M4-C2-boundary-pilot-2026-09-09/results.json"
FRAMES = ROOT / "templates 2"
FROZEN_SHA256 = "f4a0f86938561b3fc81b096fc614d1ae404c1372e65c925811c1368ce56a0da5"
PRECISION_GATE = 0.90
RECALL_GATE = 0.50


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import cv2
    import numpy as np
    from mc_photogrammetry.reconstruction.boundary_scoring import (
        boundary_error, fill_polygon, region_metrics, target_masks, usable_domain,
    )

    raw = TARGETS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_SHA256:
        raise SystemExit(f"v4 hash mismatch: expected {FROZEN_SHA256}, got {digest}")
    targets = {t["frame"]: t for t in json.loads(raw)["targets"]}
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    if len(saved["frames"]) != 3 or set(targets) != {f["frame"] for f in saved["frames"]}:
        raise SystemExit("the saved C2 three-frame set no longer matches the frozen v4 targets")
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for old in saved["frames"]:
        frame = old["frame"]
        target = targets[frame]
        width, height = target["image_size"]
        domain = usable_domain(
            width, height,
            top_exclude=saved["top_exclude_pixels"],
            bottom_exclude=saved["bottom_exclude_pixels"],
        )
        positive, unknown = target_masks(target, width, height, domain)
        if old.get("top1") is None:
            records.append({
                "frame": frame, "target_id": target["target_id"], "top1": None,
                "best_matched_evaluator_only": None, "candidates": [],
                "gate_passed": False, "note": "saved C2 run had no candidates; retained as a failure",
                "positive_area": int(positive.sum()), "excluded_unknown_area": int(unknown.sum()),
            })
            continue

        candidates = []
        for candidate in old["candidates"]:
            mask = fill_polygon(candidate["polygon"], width, height)
            metrics = region_metrics(mask, positive, domain, unknown)
            metrics["normalized_boundary_error"] = boundary_error(
                mask, positive, domain, unknown, width, height,
            )
            candidates.append({
                "rank": candidate["rank"], "strength": candidate["strength"],
                "complete": candidate.get("complete"), "polygon": candidate["polygon"],
                **metrics,
            })
        top1 = next((c for c in candidates if c["rank"] == old["top1"]["rank"]), None)
        best = max(candidates, key=lambda c: c["iou"]) if candidates else None
        passed = bool(
            top1 and top1["precision"] >= PRECISION_GATE and top1["recall"] >= RECALL_GATE
        )

        image = cv2.imread(str(FRAMES / frame), cv2.IMREAD_COLOR)
        overlay = image.copy()
        overlay[positive] = (0, 190, 0)
        overlay[unknown] = (0, 200, 220)
        overlay[domain & ~positive & ~unknown] = (0, 0, 190)
        blended = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
        for candidate in candidates:
            color = (0, 220, 220) if top1 and candidate["rank"] != top1["rank"] else (0, 140, 255)
            points = np.asarray(candidate["polygon"], dtype=np.float32).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(blended, [points], True, color, 2 if top1 and candidate["rank"] == top1["rank"] else 1, cv2.LINE_AA)
        label = (
            f"top1 P={top1['precision']:.3f} R={top1['recall']:.3f} IoU={top1['iou']:.3f}; "
            f"evaluator-only best IoU={best['iou']:.3f}"
        ) if top1 and best else "no saved candidate"
        cv2.putText(blended, label, (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{Path(frame).stem}_v4.png"), blended)
        records.append({
            "frame": frame, "target_id": target["target_id"], "top1": top1,
            "best_matched_evaluator_only": best, "candidates": candidates,
            "gate_passed": passed, "positive_area": int(positive.sum()),
            "excluded_unknown_area": int(unknown.sum()),
        })

    passed = sum(bool(row["gate_passed"]) for row in records)
    result = {
        "note": "Only the already-saved C2 candidate polygons were re-scored. This script never calls the proposer. A no-candidate frame remains in the three-frame denominator and fails.",
        "targets_v4_sha256": digest,
        "saved_candidate_source": str(SAVED.relative_to(ROOT)),
        "precision_gate": PRECISION_GATE, "recall_gate": RECALL_GATE,
        "frames": records, "frames_passed": passed, "frames_total": 3,
        "continuation_condition_met": passed >= 2 and len(records) == 3,
        "status": "provisional_annotation_dependent_diagnostic",
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "targets_v4_sha256": digest, "frames_passed": passed, "frames_total": 3,
        "frames": [{
            "frame": row["frame"], "target_id": row["target_id"],
            "top1": None if not row["top1"] else {k: row["top1"][k] for k in ("rank", "precision", "recall", "iou")},
            "best_matched_evaluator_only": None if not row["best_matched_evaluator_only"] else {k: row["best_matched_evaluator_only"][k] for k in ("rank", "precision", "recall", "iou")},
            "gate_passed": row["gate_passed"], "candidate_count": len(row["candidates"]),
        } for row in records],
    }, indent=2))


if __name__ == "__main__":
    main()
