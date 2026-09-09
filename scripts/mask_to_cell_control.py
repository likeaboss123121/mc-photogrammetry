"""M4 Part B: assisted mask-to-cell control, from genuinely image-drawn input.

Every geometric input here - the two points for a stated length, the block
count, and the occlusion region - was picked by looking at the actual
screenshot (see `docs/reports/M4-partB-annotations-templates-2.json`), not
derived from the reference litematic or the camera manifest. Camera
orientation and focal length still come from calibration (`axis_directions`,
`focal_pixels`), which is itself image-derived - never from `cameras.json`.
The manifest is read only at the very end, to score, and is absent from this
script's own decisions entirely otherwise.

The point of this control is to isolate implementation bugs from unsolved
automatic detection: if the pipeline is correct, feeding it real, honestly
imprecise human annotations on a genuinely visible surface should produce a
sensible, irregular, evidence-backed footprint - not a filled rectangle and
not a silently-empty one. It does not represent, and is not being proposed
as, the production workflow: Astra's binding scope clarification (2026-09-08)
is that normal mode neither requires nor accepts manual geometric input.

Two outputs per frame:

* a geometry file - only cells the evidence actually resolved. `supported`
  cells get a neutral placeholder block; `contradicted` cells are recorded as
  an explicit hole (an inferred opening, not a default absence). Everything
  else (`weak_evidence`, `outside_image`, `foreground_occlusion`) is left out
  of the geometry file entirely - it is not a hole and not a fill, it is
  unresolved, and the geometry file says nothing rather than guessing.
* an evidence sidecar - every judged cell's verdict, score and coverage, so
  the geometry file's claims can be checked against what actually produced
  them.

    python scripts/mask_to_cell_control.py \\
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.calibration import axis_directions, enhance
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest, camera_eye
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length
    from mc_photogrammetry.reconstruction.supported import (
        plane_from_evidence, SUPPORTED, CONTRADICTED, WEAK_EVIDENCE,
        FOREGROUND_OCCLUSION, OUTSIDE_IMAGE)

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    assert annotations["provenance"] == "image_drawn", (
        "this control refuses to run on anything but image-drawn annotations")

    fixture = ROOT / "templates 2"
    cameras = load_camera_manifest(fixture / "cameras.json") if args.score else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in calibration["estimates"]:
        frame_id = entry["frame"]
        spec = annotations["frames"].get(frame_id)
        if spec is None:
            print(f"{frame_id}: no annotation, skipped")
            continue
        size = tuple(calibration["inputs"][frame_id]["size"])
        width, height = size
        image_path = fixture / frame_id
        grey = enhance(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE))
        axes = axis_directions(entry["yaw_degrees_mod90"], entry["pitch_degrees"],
                               entry.get("roll_degrees", 0.0))
        focal = entry["focal_pixels"]
        centre_pp = entry.get("principal_point")
        principal_point = tuple(centre_pp) if centre_pp else None

        # A visibility mask scoped to the plane actually being measured, not
        # "everywhere the photograph shows something." An earlier version
        # used the whole frame minus only the occlusion polygons, which fed
        # plane_window regions showing entirely different, near-grazing
        # surfaces for this axis (the ceiling, the opposite wall) and blew up
        # its window computation (span ~283145, block_px ~0.003, silently
        # returning None downstream). `visible_extent_polygon` is the rough,
        # hand-drawn region that actually shows this wall; occlusion regions
        # are then subtracted from inside it.
        visible_mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(visible_mask, [np.array(spec["visible_extent_polygon"], dtype=np.int32)], 1.0)
        for region in spec.get("occlusion_regions_image_px", []):
            polygon = np.array(region["polygon"], dtype=np.int32)
            cv2.fillPoly(visible_mask, [polygon], 0.0)

        points = [tuple(p) for p in spec["points"]]
        blocks = float(spec["blocks_apart"])

        best = None
        for axis in spec["axis_candidates"]:
            for side in (1.0, -1.0):
                constraint = scale_from_known_length(
                    points[0], points[1], blocks, axis, axes, focal, size,
                    principal_point=principal_point)
                if constraint.distance is None:
                    continue
                plane = plane_from_evidence(
                    grey, axis, axes, focal, size, distance=constraint.distance,
                    side=side, mask=visible_mask, principal_point=principal_point)
                if plane is None:
                    continue
                supported = len(plane.supported)
                # Prefer whichever (axis, side) combination the evidence
                # itself supports best - the most cells actually judged
                # `supported`, a plain, pre-registered tie-breaker rather than
                # one chosen after seeing which looked best against ground
                # truth.
                if best is None or supported > best[0]:
                    best = (supported, axis, side, constraint, plane)

        if best is None:
            print(f"{frame_id}: no (axis, side) combination resolved, skipped")
            continue
        supported_count, axis, side, constraint, plane = best
        print(f"{frame_id}: axis={axis} side={side:+.0f} distance={plane.distance:.3f} "
              f"supported={supported_count}/{len(plane.cells)}")

        bounds = plane.bounds()
        holes = {c.cell for c in plane.holes()}
        geometry = {}
        sidecar = []
        for cell in plane.cells:
            sidecar.append({
                "cell": list(cell.cell), "verdict": cell.verdict,
                "score": cell.score, "coverage": cell.coverage,
            })
            if cell.verdict == SUPPORTED:
                geometry[f"{cell.cell[0]},{cell.cell[1]}"] = "neutral_solid"
            elif cell.cell in holes:
                geometry[f"{cell.cell[0]},{cell.cell[1]}"] = "inferred_opening"
            # weak_evidence, outside_image, foreground_occlusion, and any
            # contradicted cell outside the supported bounds: omitted. Not a
            # hole, not a fill - unresolved, and the geometry file says
            # nothing rather than guessing.

        from collections import Counter
        verdict_totals = Counter(c.verdict for c in plane.cells)

        payload = {
            "frame": frame_id, "provenance": "image_drawn",
            "annotation": spec, "axis": axis, "side": side,
            "distance_blocks": plane.distance,
            "constraint_status": constraint.status,
            "bounds": bounds,
            "verdict_totals": dict(verdict_totals),
            "geometry_cell_count": len(geometry),
            "resolved_geometry": geometry,
        }
        if args.score and frame_id in cameras:
            payload["reference_note"] = (
                "camera pose read here only, to report - not used to build "
                "any of the above")
            payload["camera_eye_for_scoring_only"] = list(camera_eye(cameras[frame_id]))

        out_geometry = args.out_dir / f"{frame_id}.geometry.json"
        out_sidecar = args.out_dir / f"{frame_id}.evidence.json"
        out_geometry.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        out_sidecar.write_text(json.dumps(sidecar, indent=2, default=float), encoding="utf-8")
        print(f"  wrote {out_geometry.name}, {out_sidecar.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
