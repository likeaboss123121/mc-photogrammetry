"""Render the Part B annotation and per-cell verdict overlays a lead review asked for.

Two images per frame:

* ``<frame>.annotation.png`` - the source photograph with the hand-drawn
  visible-extent polygon (green), occlusion regions (red hatch), and the two
  stated-length points (yellow, joined by a line) - so the annotation in
  `M4-partB-annotations-templates-2.json` can be checked by eye against what
  it was supposed to mark, not just trusted as JSON.
* ``<frame>.verdicts.png`` - the rectified patch, downsampled, with each cell
  tinted by its verdict: green (supported), red (contradicted), orange
  (foreground_occlusion), grey (weak_evidence), black (outside_image).

    python scripts/mask_to_cell_overlay.py \\
        --calibration docs/reports/M3-calibration-templates-2.json \\
        --annotations docs/reports/M4-partB-annotations-templates-2.json \\
        --out-dir docs/reports/M4-partB-templates-2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VERDICT_COLOURS = {
    "supported": (60, 200, 60),
    "contradicted": (60, 60, 220),
    "foreground_occlusion": (0, 165, 255),
    "weak_evidence": (150, 150, 150),
    "outside_image": (20, 20, 20),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.calibration import axis_directions, enhance
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length
    from mc_photogrammetry.reconstruction.supported import plane_from_evidence

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    fixture = ROOT / "templates 2"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in calibration["estimates"]:
        frame_id = entry["frame"]
        spec = annotations["frames"].get(frame_id)
        if spec is None:
            continue
        geometry_path = args.out_dir / f"{frame_id}.geometry.json"
        if not geometry_path.exists():
            print(f"{frame_id}: no geometry file, run mask_to_cell_control.py first")
            continue
        resolved = json.loads(geometry_path.read_text(encoding="utf-8"))
        axis, side = resolved["axis"], resolved["side"]

        size = tuple(calibration["inputs"][frame_id]["size"])
        colour_image = cv2.imread(str(fixture / frame_id), cv2.IMREAD_COLOR)

        # --- annotation overlay ---
        annotated = colour_image.copy()
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [np.array(spec["visible_extent_polygon"], dtype=np.int32)],
                    (0, 200, 0))
        for region in spec.get("occlusion_regions_image_px", []):
            cv2.fillPoly(overlay, [np.array(region["polygon"], dtype=np.int32)], (0, 0, 200))
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
        points = [tuple(map(int, p)) for p in spec["points"]]
        cv2.line(annotated, points[0], points[1], (0, 255, 255), 3)
        for p in points:
            cv2.circle(annotated, p, 10, (0, 255, 255), -1)
        cv2.putText(annotated, f"{spec['blocks_apart']:.0f} blocks (image-drawn estimate)",
                   (points[0][0] + 15, points[0][1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                   1.1, (0, 255, 255), 3)
        out_path = args.out_dir / f"{frame_id}.annotation.png"
        cv2.imwrite(str(out_path), annotated)

        # --- verdict overlay: recompute the plane exactly as the control did ---
        grey = enhance(cv2.imread(str(fixture / frame_id), cv2.IMREAD_GRAYSCALE))
        axes = axis_directions(entry["yaw_degrees_mod90"], entry["pitch_degrees"],
                               entry.get("roll_degrees", 0.0))
        focal = entry["focal_pixels"]
        centre_pp = entry.get("principal_point")
        principal_point = tuple(centre_pp) if centre_pp else None
        visible_mask = np.zeros(size[::-1], dtype=np.float32)
        cv2.fillPoly(visible_mask, [np.array(spec["visible_extent_polygon"], dtype=np.int32)], 1.0)
        for region in spec.get("occlusion_regions_image_px", []):
            cv2.fillPoly(visible_mask, [np.array(region["polygon"], dtype=np.int32)], 0.0)
        points_f = [tuple(p) for p in spec["points"]]
        constraint = scale_from_known_length(
            points_f[0], points_f[1], float(spec["blocks_apart"]), axis, axes, focal, size,
            principal_point=principal_point)
        plane = plane_from_evidence(grey, axis, axes, focal, size, distance=constraint.distance,
                                    side=side, mask=visible_mask, principal_point=principal_point)
        if plane is None:
            print(f"{frame_id}: plane vanished on re-run, skipping verdict overlay")
            continue

        rows = max(c.cell[0] for c in plane.cells) + 1
        cols = max(c.cell[1] for c in plane.cells) + 1
        cell_px = 24
        verdict_image = np.full((rows * cell_px, cols * cell_px, 3), 255, dtype=np.uint8)
        for cell in plane.cells:
            r, c = cell.cell
            colour = VERDICT_COLOURS.get(cell.verdict, (255, 0, 255))
            verdict_image[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = colour
        legend_y = 20
        for verdict, colour in VERDICT_COLOURS.items():
            cv2.rectangle(verdict_image, (10, legend_y - 12), (30, legend_y + 4), colour, -1)
            cv2.putText(verdict_image, verdict, (36, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            legend_y += 24
        out_verdicts = args.out_dir / f"{frame_id}.verdicts.png"
        cv2.imwrite(str(out_verdicts), verdict_image)
        print(f"{frame_id}: wrote {out_path.name}, {out_verdicts.name} ({rows}x{cols} cells)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
