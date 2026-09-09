"""M4 B, step 6: one corrected real-image Template 2 pilot.

Authorized only after the synthetic checks in `footprint_pilot.py` passed.
Every geometric input comes from
`docs/reports/M4-partB-pilot-annotation-2026-09-09.json` - picked by looking
directly at the screenshot, frozen with a SHA-256 before this script ever
scores anything against the reference. Camera optics come from image-derived
calibration (`axis_directions`, `focal_pixels`), never the manifest. The
manifest is read only at the very end, to report - not to build anything
above it.

Corrections from the superseded pilot, per the lead's review:

* The visible-extent polygon no longer overlaps the F3 debug text block.
* The stated length is carried through with an explicit LOW confidence flag
  rather than asserted as precise - the annotation JSON documents why.
* Only one frame, one plane - not three frames worth of best-effort guesses.

    python scripts/footprint_pilot_real.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ANNOTATION_PATH = ROOT / "docs/reports/M4-partB-pilot-annotation-2026-09-09.json"
FROZEN_SHA256 = "9621bc5a07b1c69f649a8e3792128708157c16c50ef9c09135a5d87078248978"
CALIBRATION_PATH = ROOT / "docs/reports/M3-calibration-templates-2.json"
FIXTURE = ROOT / "templates 2"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    raw = ANNOTATION_PATH.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_SHA256:
        raise SystemExit(
            f"annotation file changed since it was frozen: expected {FROZEN_SHA256}, "
            f"got {actual_hash}. Not scoring a moved target.")
    annotation = json.loads(raw)
    print(f"annotation frozen and verified: {FROZEN_SHA256}")

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.calibration import axis_directions, enhance
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length
    from mc_photogrammetry.reconstruction.footprint import (
        FOREGROUND_OCCLUSION, GEOMETRY_OPENING, OPENING, SUPPORTED, UNKNOWN,
        VISIBLE, build_footprint, cell_to_block, footprint_to_volume)
    from mc_photogrammetry.reconstruction.footprint_adapter import (
        LatticeTransform, RegionEvidence, cell_corners_pixels,
        declared_labels_from_evidence, rasterize_cells)
    from mc_photogrammetry.export.litematic import LitematicExporter

    frame_id = annotation["frame"]
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    entry = next(e for e in calibration["estimates"] if e["frame"] == frame_id)
    size = tuple(calibration["inputs"][frame_id]["size"])
    axes = axis_directions(entry["yaw_degrees_mod90"], entry["pitch_degrees"],
                           entry.get("roll_degrees", 0.0))
    focal = entry["focal_pixels"]
    centre_pp = entry.get("principal_point")
    principal_point = tuple(centre_pp) if centre_pp else (size[0] / 2.0, size[1] / 2.0)

    length = annotation["length_estimate"]
    points = [tuple(p) for p in length["points"]]
    blocks_apart = float(length["blocks_apart"])

    resolved = None
    for axis in annotation["axis_candidates"]:
        for side in (1.0, -1.0):
            constraint = scale_from_known_length(
                points[0], points[1], blocks_apart, axis, axes, focal, size,
                principal_point=principal_point)
            if constraint.distance is None:
                continue
            if resolved is None:
                resolved = (axis, side, constraint.distance)
    if resolved is None:
        print("STATED LENGTH DID NOT RESOLVE on any axis/side candidate.")
        print("Per the assignment: reporting this as unresolved ambiguity rather than")
        print("inventing a distance or block indices. No cells are constructed.")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "result.json").write_text(json.dumps({
            "status": "length_unresolved", "annotation_sha256": FROZEN_SHA256,
        }, indent=2), encoding="utf-8")
        return 0

    axis, side, distance = resolved
    print(f"length resolved: axis={axis} side={side:+.0f} distance={distance:.3f} blocks "
         f"[confidence: {length['confidence']}]")

    transform = LatticeTransform(axis=axis, side=side, distance=distance, axis_matrix=axes,
                                 focal=focal, principal_point=principal_point,
                                 lateral_origin=(0.0, 0.0), period=1.0)

    regions = [RegionEvidence(VISIBLE, tuple(tuple(p) for p in annotation["visible_extent_polygon"]))]
    for region in annotation.get("occlusion_regions_image_px", []):
        regions.append(RegionEvidence(FOREGROUND_OCCLUSION, tuple(tuple(p) for p in region["polygon"])))

    # A candidate cell range wide enough to cover the declared extent -
    # derived from the extent polygon's own pixel bounds via the inverse
    # mapping, not an arbitrary guess.
    from mc_photogrammetry.reconstruction.footprint_adapter import pixel_to_lateral
    lateral_corners = [pixel_to_lateral(transform, tuple(p))
                       for p in annotation["visible_extent_polygon"]]
    lateral_corners = [c for c in lateral_corners if c is not None]
    if not lateral_corners:
        print("STATED PLANE DOES NOT INTERSECT THE DECLARED EXTENT in front of the camera.")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "result.json").write_text(json.dumps({
            "status": "extent_behind_camera", "annotation_sha256": FROZEN_SHA256,
        }, indent=2), encoding="utf-8")
        return 0
    us = [c[0] for c in lateral_corners]
    vs = [c[1] for c in lateral_corners]
    columns = range(int(np.floor(min(us))) - 1, int(np.ceil(max(us))) + 1)
    rows = range(int(np.floor(min(vs))) - 1, int(np.ceil(max(vs))) + 1)
    print(f"candidate grid: {len(rows)} rows x {len(columns)} columns")

    evidence = rasterize_cells(transform, regions, image_size=size, rows=rows, columns=columns)
    recovered_labels = declared_labels_from_evidence(evidence)
    footprint = build_footprint(axis=axis, face_coordinate=0, side=side,
                               lateral_origin=(min(rows), min(columns)),
                               rows=len(rows), columns=len(columns),
                               declared={(r - min(rows), c - min(columns)): label
                                        for (r, c), label in recovered_labels.items()})

    from collections import Counter
    totals = Counter(c.geometry for c in footprint.cells)
    print(f"resolved: {dict(totals)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "resolved", "annotation_sha256": FROZEN_SHA256, "frame": frame_id,
        "axis": axis, "side": side, "distance_blocks": distance,
        "length_confidence": length["confidence"],
        "verdict_totals": dict(totals),
        # Cell indices are in the *transform's own* (0, 0)-origin lattice
        # frame - i.e. directly usable with cell_corners_pixels/cell_to_block
        # given this same LatticeTransform - not footprint.py's local,
        # 0-indexed frame. That local remapping exists only because
        # build_footprint's declared dict is 0-indexed internally; undoing it
        # here means a reader never has to separately recover
        # (min(rows), min(columns)) to make sense of these numbers.
        "lattice_origin_offset": [min(rows), min(columns)],
        "supported_cells": [[c.cell[0] + min(rows), c.cell[1] + min(columns)]
                            for c in footprint.of(SUPPORTED)],
        "opening_cells": [[c.cell[0] + min(rows), c.cell[1] + min(columns)]
                          for c in footprint.of(GEOMETRY_OPENING)],
        "unknown_cells_in_extent": len(footprint.of(UNKNOWN)),
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if footprint.of(SUPPORTED):
        volume = footprint_to_volume(footprint)
        LitematicExporter().export(volume, args.out_dir / "pilot.litematic",
                                   name="footprint pilot, real image",
                                   description="M4 B real-image pilot, Template 2")
        print(f"wrote pilot.litematic ({len(footprint.of(SUPPORTED))} solids)")

    # Overlay for review: the extent/occlusion regions and every candidate
    # cell's resolved geometry, drawn on the actual photograph.
    image = cv2.imread(str(FIXTURE / frame_id), cv2.IMREAD_COLOR)
    overlay = image.copy()
    cv2.fillPoly(overlay, [np.array(annotation["visible_extent_polygon"], dtype=np.int32)],
                (0, 200, 0))
    for region in annotation.get("occlusion_regions_image_px", []):
        cv2.fillPoly(overlay, [np.array(region["polygon"], dtype=np.int32)], (0, 0, 200))
    image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)
    verdict_colours = {SUPPORTED: (60, 200, 60), GEOMETRY_OPENING: (60, 60, 220), UNKNOWN: (140, 140, 140)}
    for cell in footprint.cells:
        world_cell = (cell.cell[0] + min(rows), cell.cell[1] + min(columns))
        corners = cell_corners_pixels(transform, world_cell)
        if corners is None:
            continue
        pts = np.array([[round(x), round(y)] for x, y in corners], dtype=np.int32)
        cv2.polylines(image, [pts], True, verdict_colours.get(cell.geometry, (0, 0, 0)), 2)
    p0, p1 = [round(v) for v in points[0]], [round(v) for v in points[1]]
    cv2.line(image, p0, p1, (0, 255, 255), 3)
    cv2.putText(image, f"{blocks_apart:.0f} blocks ({length['confidence']} confidence)",
               (p0[0] + 15, p0[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imwrite(str(args.out_dir / "overlay.png"), image)
    print(f"wrote result.json, overlay.png to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
