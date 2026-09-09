"""Build supported cells for one local plane, per image, and score them.

Runs identical settings on each frame separately. Nothing here relates one view
to another: a single view gives a local frame and no position, so placing two
planes in a common frame would need evidence this does not have.

Two paths are reported side by side and never averaged:

* **automatic** - the distance comes from the lattice period, which carries the
  block-versus-texel ambiguity and can be wrong by a factor of sixteen while
  looking perfectly reasonable.
* **assisted** - the distance comes from two points a user names on the surface
  and how many blocks apart they are. That is the one constraint Minecraft's
  repetition cannot defeat, and it is a real production input: it needs no
  manifest, no second view and no access to the world.

Bounds come from the supported cells. Reference geometry is read only to score,
after every decision has been made.

``--simulate-annotations`` manufactures the assisted path's two points by
projecting reference corners through the manifest pose. That is *scaffolding*
standing in for a user's mouse, and it is recorded as such in the artifact: the
pixel coordinates are simulated, the pipeline downstream of them is not.

    python scripts/supported_plane.py "templates 2" \\
        --calibration docs/reports/M3-calibration-templates-2.json \\
        --axis 1 --simulate-annotations \\
        --out docs/reports/M4-supported-plane-templates-2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.reconstruction.calibration import axis_directions  # noqa: E402
from mc_photogrammetry.reconstruction.supported import (  # noqa: E402
    CONTRADICTED,
    UNOBSERVED,
    plane_from_evidence,
)

AXIS_NAMES = ("X", "Y", "Z")
SEPARATION_BLOCKS = 4.0
"""How far apart the two simulated annotation points are asked to be.

Far enough that a click's own error is small against it, near enough that both
points stay in frame on a wide-angle interior shot.
"""


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def simulate_annotation(reference, scene, camera, axis, size):
    """Stand in for a user clicking two points on the most prominent surface.

    Scaffolding, not measurement. The reference and the manifest are read here
    and nowhere else in the assisted path: what leaves this function is two pixel
    coordinates and an integer, which is exactly what a user would supply.
    """
    from mc_photogrammetry.reconstruction.camera import camera_eye, project_points

    width, height = size
    eye = np.asarray(camera_eye(camera), dtype=float)
    levels = Counter()
    for coordinate in reference.blocks:
        world = scene.to_world(coordinate)
        levels[world[axis]] += 1
    # The surface a user would pick: the busiest plane on this axis, on whichever
    # side of the eye holds more of the build.
    candidates = sorted(levels, key=lambda level: -levels[level])
    for level in candidates:
        if abs(level - eye[axis]) < 1.0:
            continue
        # A face of that plane, seen from this side.
        face = float(level) + (1.0 if level < eye[axis] else 0.0)
        on_plane = [scene.to_world(c) for c in reference.blocks
                    if scene.to_world(c)[axis] == level]
        others = [i for i in range(3) if i != axis]
        best = None
        for first in on_plane:
            for direction in others:
                second = list(first)
                second[direction] += int(SEPARATION_BLOCKS)
                points = []
                for corner in (first, second):
                    world = [float(corner[0]) + 0.5, float(corner[1]) + 0.5,
                             float(corner[2]) + 0.5]
                    world[axis] = face
                    points.append(tuple(world))
                pixels = project_points(points, camera, width=width, height=height)
                if any(p is None for p in pixels):
                    continue
                margin = 0.04
                if not all(width * margin <= p[0] <= width * (1 - margin)
                           and height * margin <= p[1] <= height * (1 - margin)
                           for p in pixels):
                    continue
                gap = float(np.hypot(pixels[0][0] - pixels[1][0],
                                     pixels[0][1] - pixels[1][1]))
                if best is None or gap > best[0]:
                    best = (gap, [[float(pixels[0][0]), float(pixels[0][1])],
                                  [float(pixels[1][0]), float(pixels[1][1])]],
                            face)
        if best is not None:
            return {"points": best[1], "blocks": SEPARATION_BLOCKS,
                    "provenance": "simulated: reference corners projected "
                                  "through the manifest pose, standing in for a "
                                  "user's two clicks",
                    "pixel_separation": best[0],
                    "surface_world_coordinate": best[2],
                    "side": 1.0 if best[2] > eye[axis] else -1.0}
    return None


def reference_distance(reference, scene, camera, axis, side):
    """How far the surface straight along the axis really is. Evaluation only.

    The window sits on the view centre of the axis, so what it looks at is the
    surface directly above or below the eye - not merely the nearest one. Taking
    the column of reference blocks the eye stands in answers that exactly.
    """
    from mc_photogrammetry.reconstruction.camera import camera_eye

    eye = np.asarray(camera_eye(camera), dtype=float)
    others = [i for i in range(3) if i != axis]
    best = None
    for coordinate in reference.blocks:
        world = scene.to_world(coordinate)
        if not all(world[i] <= eye[i] <= world[i] + 1.0 for i in others):
            continue
        # The face turned towards the eye.
        face = float(world[axis]) + (1.0 if world[axis] < eye[axis] else 0.0)
        gap = (face - eye[axis]) * side
        if gap <= 0.0:
            continue
        if best is None or gap < best:
            best = gap
    return best


def reference_cells(reference, scene, camera, axis, distance, palette, size, frame):
    """Cells the reference really has on the plane at that distance, seen here.

    Evaluation only. Used after every decision, to score - never to choose a
    plane, a distance or a cell.
    """
    from mc_photogrammetry.export.visibility import visible_cells
    from mc_photogrammetry.reconstruction.camera import camera_eye

    eye = np.asarray(camera_eye(camera), dtype=float)
    reachable = visible_cells(reference.blocks, scene, {frame: camera},
                              {frame: size}, palette=palette)
    found = set()
    for coordinate in reachable:
        world = scene.to_world(coordinate)
        for offset in (0.0, 1.0):
            if abs(abs(world[axis] + offset - eye[axis]) - distance) < 0.51:
                found.add(coordinate)
                break
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--axis", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--annotations", type=Path,
                        help="JSON of {frame: {points: [[x,y],[x,y]], blocks: n}} "
                             "for the assisted path")
    parser.add_argument("--simulate-annotations", action="store_true",
                        help="stand in for a user's clicks by projecting "
                             "reference corners; scaffolding, recorded as such")
    parser.add_argument("--yaw-quadrant", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    import cv2

    from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.reconstruction.calibration import enhance
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length
    from mc_photogrammetry.reconstruction.scene import load_scene_transform

    folder = ROOT / args.fixture
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    scene = load_scene_transform(folder / "scene.json")
    reference = load_litematic(next(folder.glob("*.litematic")))
    cameras = load_camera_manifest(folder / "cameras.json")

    annotations = {}
    if args.annotations:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))

    palette = None
    if args.score:
        provider = VanillaJarTextureProvider()
        provider.load_jar(Path(os.environ["APPDATA"])
                          / ".minecraft/versions/1.21.5/1.21.5.jar")
        palette = provider.complete_palette()

    rows = []
    print(f"  {'frame':<26}{'path':>14}{'dist':>8}{'cells':>7}{'sup':>5}"
          f"{'contra':>7}{'unobs':>7}{'holes':>7}{'ref':>6}")
    for entry in calibration["estimates"]:
        frame_id = entry["frame"]
        path = folder / frame_id
        if not path.exists():
            continue
        size = tuple(calibration["inputs"][frame_id]["size"])
        grey = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2GRAY)
        boosted = enhance(grey)
        yaw = entry["yaw_degrees_mod90"] + 90.0 * args.yaw_quadrant
        axes = axis_directions(yaw, entry["pitch_degrees"],
                               entry.get("roll_degrees", 0.0))
        centre = entry.get("principal_point")
        centre = tuple(centre) if centre else None

        annotation = annotations.get(frame_id)
        if annotation is None and args.simulate_annotations:
            annotation = simulate_annotation(reference, scene, cameras[frame_id],
                                             args.axis, size)

        stated = None
        constraint_status = "not attempted"
        if annotation:
            constraint = scale_from_known_length(
                annotation["points"][0], annotation["points"][1],
                float(annotation["blocks"]), args.axis, axes,
                entry["focal_pixels"], size, principal_point=centre)
            constraint_status = constraint.status
            stated = constraint.distance

        annotated_side = (annotation or {}).get("side", 1.0)
        attempts = [(f"automatic +{AXIS_NAMES[args.axis]}", None, 1.0),
                    (f"automatic -{AXIS_NAMES[args.axis]}", None, -1.0),
                    ("assisted", stated, annotated_side)]
        for label, distance, side in attempts:
            if label == "assisted" and distance is None:
                if annotation:
                    print(f"  {frame_id:<26}{label:>14}   stated length did not "
                          f"resolve: {constraint_status}")
                    rows.append({"frame": frame_id, "path": label, "plane": None,
                                 "annotation": annotation,
                                 "constraint_status": constraint_status})
                continue
            plane = plane_from_evidence(boosted, args.axis, axes,
                                        entry["focal_pixels"], size,
                                        distance=distance, side=side,
                                        principal_point=centre)
            if plane is None:
                print(f"  {frame_id:<26}{label:>14}   no plane measured")
                rows.append({"frame": frame_id, "path": label, "plane": None})
                continue

            supported = len(plane.supported)
            contradicted = len(plane.of(CONTRADICTED))
            unobserved = len(plane.of(UNOBSERVED))
            record = {
                "frame": frame_id, "path": label, "axis": args.axis,
                "side": side, "distance_blocks": plane.distance,
                "phase_pixels": list(plane.phase),
                "cells_judged": len(plane.cells),
                "supported": supported, "contradicted": contradicted,
                "unobserved": unobserved,
                "bounds": plane.bounds(),
                "holes_inside_extent": len(plane.holes()),
                "notes": list(plane.notes),
            }
            # The verdict counts alone cannot tell "the cue said no" from "the
            # cue said nothing". The score distribution can: chance is 0.378.
            judged = [c.score for c in plane.cells if c.verdict != UNOBSERVED]
            if judged:
                record["score_quantiles"] = {
                    "min": float(np.min(judged)),
                    "p50": float(np.percentile(judged, 50)),
                    "p90": float(np.percentile(judged, 90)),
                    "max": float(np.max(judged)),
                    "chance_level": 0.378,
                }
            if label == "assisted":
                record["annotation"] = annotation
                record["constraint_status"] = constraint_status

            truth = None
            if args.score:
                actual = reference_distance(reference, scene, cameras[frame_id],
                                            args.axis, side)
                record["reference_distance_blocks"] = actual
                if actual:
                    record["distance_relative_error"] = abs(
                        plane.distance - actual) / actual
                cells = reference_cells(reference, scene, cameras[frame_id],
                                        args.axis, plane.distance, palette,
                                        size, frame_id)
                truth = len(cells)
                record["reference_cells_on_plane"] = truth
                if truth:
                    overlap = min(supported, truth)
                    record["count_recall"] = overlap / truth
                    record["count_precision"] = overlap / supported if supported else 0.0
                record["scoring_note"] = (
                    "counts only: a local frame without position cannot be put "
                    "into world coordinates, so per-cell correspondence is not "
                    "claimed and these are not per-cell precision and recall")

            rows.append(record)
            print(f"  {frame_id:<26}{label:>14}{plane.distance:>8.2f}"
                  f"{len(plane.cells):>7}{supported:>5}{contradicted:>7}"
                  f"{unobserved:>7}{len(plane.holes()):>7}"
                  f"{'-' if truth is None else truth:>6}")

    payload = {
        "fixture": args.fixture,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "axis": args.axis, "axis_name": AXIS_NAMES[args.axis],
        "yaw_quadrant": args.yaw_quadrant,
        "settings": {
            "minimum_coverage": __import__(
                "mc_photogrammetry.reconstruction.supported",
                fromlist=["MINIMUM_COVERAGE"]).MINIMUM_COVERAGE,
            "phase_tolerance": __import__(
                "mc_photogrammetry.reconstruction.supported",
                fromlist=["PHASE_TOLERANCE"]).PHASE_TOLERANCE,
            "support_threshold": __import__(
                "mc_photogrammetry.reconstruction.supported",
                fromlist=["SUPPORT_THRESHOLD"]).SUPPORT_THRESHOLD,
        },
        "annotations_simulated": bool(args.simulate_annotations),
        "calibration_artifact": {"path": str(args.calibration),
                                 "sha256": digest(args.calibration)},
        "note": ("each frame is treated independently in its own local cell "
                 "frame; no relative placement between views is claimed, since "
                 "a single view recovers no position"),
        "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float),
                            encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
