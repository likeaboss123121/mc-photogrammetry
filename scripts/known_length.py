"""Resolve a plane's distance from a length the user states across it.

One image fixes a plane's shape and never its size, so scale has to come from
outside the picture. This is the cheapest thing to ask for: two points on one
surface and how many blocks apart they are. It needs no correspondence, no second
view and no feature matching, which is why it survives the repetition that
defeats appearance-based methods on Minecraft interiors.

The points are an *input*, so they are recorded in the artifact alongside the
answer. A distance whose points are not written down cannot be checked, and the
earlier ceiling measurement had exactly that problem: a real number with no
reproducible invocation behind it.

An impossible constraint returns its status and no distance. Two points on
opposite sides of a plane's vanishing line cannot both lie on one plane in front
of the camera, and answering anyway hands downstream a number it cannot tell from
a measured one.

    python scripts/known_length.py "templates 2" 2026-09-02_13.22.14.png \\
        --calibration docs/reports/M3-calibration-templates-2.json \\
        --axis 1 --point 1200,300 --point 1500,300 --blocks 4 --out report.json

``--score`` compares the answer against the reference geometry afterwards and is
evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.reconstruction.calibration import axis_directions  # noqa: E402
from mc_photogrammetry.reconstruction.lattice import (  # noqa: E402
    GRAZING_INCIDENCE,
    MINIMUM_SEPARATION,
    scale_from_known_length,
)

AXIS_NAMES = ("X", "Y", "Z")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def as_point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a point as x,y but got {text!r}") from error
    return (x, y)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("frame")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--axis", type=int, required=True, choices=(0, 1, 2),
                        help="world axis the surface is perpendicular to")
    parser.add_argument("--point", type=as_point, action="append", required=True,
                        help="an image point on the surface; give exactly two")
    parser.add_argument("--blocks", type=float, required=True,
                        help="how many blocks apart the two points are")
    parser.add_argument("--yaw-quadrant", type=int, default=0, choices=(0, 1, 2, 3),
                        help="quarter turns to add; one view fixes yaw only "
                             "modulo 90 degrees")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--score", action="store_true",
                        help="compare against the reference; evaluation only")
    args = parser.parse_args(argv)

    if len(args.point) != 2:
        raise SystemExit(f"need exactly two points, got {len(args.point)}")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    row = next((r for r in calibration["estimates"] if r["frame"] == args.frame), None)
    if row is None:
        raise SystemExit(f"{args.frame} is not in {args.calibration}")
    size = calibration["inputs"][args.frame]["size"]

    axes = axis_directions(row["yaw_degrees_mod90"] + 90.0 * args.yaw_quadrant,
                           row["pitch_degrees"], row.get("roll_degrees", 0.0))
    centre = row.get("principal_point")
    found = scale_from_known_length(
        args.point[0], args.point[1], args.blocks, args.axis, axes,
        row["focal_pixels"], tuple(size),
        principal_point=tuple(centre) if centre else None)

    print(f"{args.fixture} / {args.frame}")
    print(f"  surface perpendicular to {AXIS_NAMES[args.axis]}, "
          f"points {args.point[0]} and {args.point[1]}, stated {args.blocks} blocks apart")
    print(f"  yaw {row['yaw_degrees_mod90']:.2f} + {90 * args.yaw_quadrant} deg, "
          f"pitch {row['pitch_degrees']:.2f}, focal {row['focal_pixels']:.1f}")
    if found.resolved:
        print(f"  -> plane distance {found.distance:.3f} blocks")
    else:
        print(f"  -> UNRESOLVED ({found.status}): {found.detail}")

    payload = {
        "fixture": args.fixture,
        "frame": args.frame,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration_artifact": {"path": str(args.calibration),
                                 "sha256": digest(args.calibration)},
        "inputs": {
            "axis": args.axis,
            "axis_name": AXIS_NAMES[args.axis],
            "points": [list(p) for p in args.point],
            "blocks_apart": args.blocks,
            "yaw_quadrant": args.yaw_quadrant,
            "image_size": size,
        },
        "camera_used": {"yaw_degrees": row["yaw_degrees_mod90"] + 90.0 * args.yaw_quadrant,
                        "pitch_degrees": row["pitch_degrees"],
                        "roll_degrees": row.get("roll_degrees", 0.0),
                        "focal_pixels": row["focal_pixels"],
                        "principal_point": centre},
        "thresholds": {"grazing_incidence": GRAZING_INCIDENCE,
                       "minimum_separation": MINIMUM_SEPARATION},
        "result": {"status": found.status, "distance_blocks": found.distance,
                   "detail": found.detail},
        "scored": None,
    }

    if args.score:
        from mc_photogrammetry.export.compare import load_litematic
        from mc_photogrammetry.reconstruction.camera import load_camera_manifest
        from mc_photogrammetry.reconstruction.scene import load_scene_transform

        folder = ROOT / args.fixture
        manifest = folder / "cameras.json"
        truth = load_camera_manifest(manifest)[args.frame]
        scene = load_scene_transform(folder / "scene.json")
        reference = load_litematic(next(folder.glob("*.litematic")))
        eye = (np.asarray(truth.position, dtype=float)
               + np.asarray([0.0, truth.eye_height, 0.0]))

        # The nearest occupied cell wall along that axis, as a reference distance.
        walls = sorted({scene.to_world(c)[args.axis] for c in reference.blocks}
                       | {scene.to_world(c)[args.axis] + 1.0 for c in reference.blocks})
        nearest = min(walls, key=lambda w: abs(w - eye[args.axis]) if found.resolved
                      and abs(abs(w - eye[args.axis]) - found.distance) < 2.0
                      else float("inf")) if found.resolved else None
        payload["scored"] = {
            "note": "evaluation only; the manifest and reference are scaffolding "
                    "and are not read while resolving the constraint",
            "manifest_sha256": digest(manifest),
            "eye_on_axis": float(eye[args.axis]),
            "nearest_matching_wall": None if nearest is None else float(nearest),
            "reference_distance_blocks": None if nearest is None
            else float(abs(nearest - eye[args.axis])),
        }
        reference_distance = payload["scored"]["reference_distance_blocks"]
        if found.resolved and reference_distance:
            error = abs(found.distance - reference_distance) / reference_distance
            payload["scored"]["relative_error"] = float(error)
            print(f"  reference distance {reference_distance:.3f} blocks, "
                  f"error {error:.2%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0 if found.resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
