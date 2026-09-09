"""M4 C: one automatic structural-boundary proposal run on all three T2 frames.

Authorized directly by the lead review's conditional next step, after the
G1/G2/G3 corrections (`docs/reports/M4-G1-G2-G3-fixes-2026-09-09.md`) -
"unresolved real block phase/scale need not prevent this mask experiment;
it prevents exact-cell claims." This is 2D footprint evidence only: does a
candidate boundary for a visible plane fall out of image-derived vanishing
directions and line chains, not camera/block-scale recovery.

Per-frame target masks (`M4-C-boundary-targets-2026-09-09.json`) are a
frozen, hand-drawn evaluation control - HUD, foreground occlusion and
openings distinguished, verified by content hash before scoring. Nothing in
`boundary_candidates.py` or this script's proposal step reads that file:
the automatic proposal runs first, from the photograph and its
already-calibrated, image-derived camera alone
(`docs/reports/M3-calibration-templates-2.json`, itself built from
`calibration.py`'s line-segment/vanishing-point machinery - never a
manifest), and only the resulting candidates are compared against the
frozen target afterward.

One configuration, run once, on all three frames: no threshold sweep, no
per-frame tuning, no seam/period occupancy gate, no new dependency. Stops
after these three runs regardless of outcome, per the review's explicit
instruction.

    python scripts/boundary_pilot.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGETS_PATH = ROOT / "docs/reports/M4-C-boundary-targets-2026-09-09.json"
FROZEN_SHA256 = "206ef350eeb659748d8f995963b640a409ae7bd821036696ddaf217b453aaa84"
CALIBRATION_PATH = ROOT / "docs/reports/M3-calibration-templates-2.json"
FIXTURE = ROOT / "templates 2"

TOP_EXCLUDE_PIXELS = 130
"""F3 debug text band, established in the M4 B real pilot (partA/B reports)."""
BOTTOM_EXCLUDE_PIXELS = 120
"""Hotbar plus, on the third frame, the 'Saved screenshot as ...' HUD line."""

PRECISION_GATE = 0.90
RECALL_GATE = 0.50


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fill_mask(polygon, width: int, height: int):
    import numpy as np
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], 255)
    return mask > 0


def _boundary_mask(polygon, width: int, height: int):
    import numpy as np
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [points], isClosed=True, color=255, thickness=1)
    return mask


def _usable_domain(width: int, height: int):
    import numpy as np

    domain = np.ones((height, width), dtype=bool)
    domain[:TOP_EXCLUDE_PIXELS, :] = False
    domain[height - BOTTOM_EXCLUDE_PIXELS:, :] = False
    return domain


def _region_metrics(candidate_mask, target_mask):
    intersection = int((candidate_mask & target_mask).sum())
    candidate_area = int(candidate_mask.sum())
    target_area = int(target_mask.sum())
    union = int((candidate_mask | target_mask).sum())
    precision = intersection / candidate_area if candidate_area else 0.0
    recall = intersection / target_area if target_area else 0.0
    iou = intersection / union if union else 0.0
    return {"precision": precision, "recall": recall, "iou": iou,
           "candidate_area": candidate_area, "target_area": target_area,
           "intersection_area": intersection}


def _boundary_error(candidate_polygon, target_polygon, width: int, height: int) -> float:
    import cv2

    candidate_boundary = _boundary_mask(candidate_polygon, width, height)
    target_boundary = _boundary_mask(target_polygon, width, height)
    diagonal = math.hypot(width, height)
    distance_to_target = cv2.distanceTransform(255 - target_boundary, cv2.DIST_L2, 5)
    distance_to_candidate = cv2.distanceTransform(255 - candidate_boundary, cv2.DIST_L2, 5)
    candidate_points = candidate_boundary > 0
    target_points = target_boundary > 0
    if not candidate_points.any() or not target_points.any():
        return float("nan")
    mean_a = float(distance_to_target[candidate_points].mean())
    mean_b = float(distance_to_candidate[target_points].mean())
    return (mean_a + mean_b) / 2.0 / diagonal


def _draw_polygon(image, polygon, color, thickness, width, height):
    import numpy as np
    import cv2

    points = np.array([(max(-2 * width, min(3 * width, p[0])),
                        max(-2 * height, min(3 * height, p[1]))) for p in polygon],
                      dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness,
                 lineType=cv2.LINE_AA)


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
    targets = json.loads(raw)
    print(f"targets frozen and verified: {FROZEN_SHA256}")

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.boundary_candidates import (
        find_chains, propose_boundary_candidates,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    calibration_hash = _file_hash(CALIBRATION_PATH)

    frame_results = []
    for target in targets["targets"]:
        frame_id = target["frame"]
        width, height = target["image_size"]
        image_path = FIXTURE / frame_id
        source_hash = _file_hash(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        entry = next(e for e in calibration["estimates"] if e["frame"] == frame_id)
        principal_point = entry.get("principal_point")
        principal_point = tuple(principal_point) if principal_point else None

        chains = find_chains(
            image, focal=entry["focal_pixels"], yaw_degrees=entry["yaw_degrees_mod90"],
            pitch_degrees=entry["pitch_degrees"], roll_degrees=entry.get("roll_degrees", 0.0),
            principal_point=principal_point,
        )
        candidates = propose_boundary_candidates(chains, width, height)

        domain = _usable_domain(width, height)
        visible = _fill_mask(target["visible_extent_polygon"], width, height)
        excluded = np.zeros((height, width), dtype=bool)
        for polygon in target.get("foreground_occlusion_polygons", []):
            excluded |= _fill_mask(polygon, width, height)
        for polygon in target.get("opening_polygons", []):
            excluded |= _fill_mask(polygon, width, height)
        target_mask = visible & ~excluded & domain

        candidate_records = []
        for rank, candidate in enumerate(candidates):
            candidate_mask = _fill_mask(candidate.polygon, width, height) & domain
            metrics = _region_metrics(candidate_mask, target_mask)
            boundary_error = _boundary_error(
                candidate.polygon, target["visible_extent_polygon"], width, height)
            candidate_records.append({
                "rank": rank, "strength": candidate.strength, "polygon": candidate.polygon,
                "edges": {side: (None if chain is None else {
                    "axis": chain.axis, "strength": chain.strength, "point": chain.point,
                    "direction": chain.direction, "segment_count": chain.segment_count,
                }) for side, chain in candidate.edges.items()},
                **metrics, "normalized_boundary_error": boundary_error,
            })

        chosen = candidate_records[0] if candidate_records else None
        passed = bool(chosen and chosen["precision"] >= PRECISION_GATE
                      and chosen["recall"] >= RECALL_GATE)

        overlay = bgr.copy()
        _draw_polygon(overlay, target["visible_extent_polygon"], (0, 220, 0), 3, width, height)
        for polygon in target.get("foreground_occlusion_polygons", []):
            _draw_polygon(overlay, polygon, (0, 0, 220), 2, width, height)
        for polygon in target.get("opening_polygons", []):
            _draw_polygon(overlay, polygon, (220, 160, 0), 2, width, height)
        for record in candidate_records[1:]:
            _draw_polygon(overlay, record["polygon"], (0, 200, 220), 1, width, height)
        if chosen is not None:
            _draw_polygon(overlay, chosen["polygon"], (0, 140, 255), 3, width, height)
        cv2.line(overlay, (0, TOP_EXCLUDE_PIXELS), (width, TOP_EXCLUDE_PIXELS), (128, 128, 128), 1)
        cv2.line(overlay, (0, height - BOTTOM_EXCLUDE_PIXELS),
                 (width, height - BOTTOM_EXCLUDE_PIXELS), (128, 128, 128), 1)
        label = (f"chosen: P={chosen['precision']:.2f} R={chosen['recall']:.2f} "
                f"IoU={chosen['iou']:.2f} {'PASS' if passed else 'fail'}"
                if chosen else "no candidates")
        cv2.putText(overlay, label, (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (0, 140, 255) if passed else (0, 0, 220), 2, cv2.LINE_AA)
        overlay_path = args.out_dir / f"overlay_{frame_id}"
        cv2.imwrite(str(overlay_path), overlay)

        frame_results.append({
            "frame": frame_id, "source_image_sha256": source_hash,
            "chain_count": len(chains), "candidate_count": len(candidates),
            "target_area_pixels": int(target_mask.sum()),
            "chosen": chosen, "candidates": candidate_records,
            "gate_passed": passed,
        })
        print(f"{frame_id}: {len(chains)} chains, {len(candidates)} candidates, "
             f"chosen P={chosen['precision']:.3f} R={chosen['recall']:.3f} "
             f"IoU={chosen['iou']:.3f} boundary_err={chosen['normalized_boundary_error']:.4f} "
             f"-> {'PASS' if passed else 'fail'}" if chosen else f"{frame_id}: no candidates")

    passed_count = sum(1 for f in frame_results if f["gate_passed"])
    continuation = passed_count >= 2
    print(f"local continuation condition (>=2 of 3 at P>=0.90,R>=0.50): "
         f"{passed_count} of 3 passed -> {'MET' if continuation else 'NOT MET'}")

    results = {
        "targets_sha256": FROZEN_SHA256, "calibration_artifact_sha256": calibration_hash,
        "precision_gate": PRECISION_GATE, "recall_gate": RECALL_GATE,
        "top_exclude_pixels": TOP_EXCLUDE_PIXELS, "bottom_exclude_pixels": BOTTOM_EXCLUDE_PIXELS,
        "frames": frame_results, "frames_passed": passed_count,
        "continuation_condition_met": continuation,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote results.json and {len(frame_results)} overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
