"""Register camera translations across views that share an asserted plane.

Builds on `supported_plane.py`'s assisted distance, which places a plane to
within 0.4% from a single image. This script adds the one thing a single image
cannot give: a *second* view of the same physical surface, asserted by the
caller via `--link`, from which `registration.register_axis` solves the two
views' relative position along that axis.

On the vertical axis that is the whole story - "up" means the same thing in
every screenshot regardless of which way the player was facing. The two
horizontal axes need one more fact first: `axis_directions` only fixes a view's
yaw modulo 90 degrees, so "axis 0" in one view and "axis 0" in another may not
be the same wall of the room, or may point along it backwards. `--quadrant`
states the missing quarter-turn per view, the same way `--link` states plane
identity - a fact a person can read off two photographs at a glance, that
appearance-matching cannot be trusted to recover on its own.

The manifest is read only after registration, to score - never to build the
links, choose a quadrant, or choose which plane a view is looking at.

    python scripts/register_planes.py "templates 2" \\
        --calibration docs/reports/M3-calibration-templates-2.json \\
        --axis 1 --side -1 --simulate-annotations \\
        --link 2026-09-02_13.22.14.png,2026-09-02_13.24.24.png \\
        --anchor 2026-09-02_13.22.14.png \\
        --out docs/reports/M4-registration-templates-2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.reconstruction.calibration import axis_directions  # noqa: E402
from mc_photogrammetry.reconstruction.registration import (  # noqa: E402
    PlaneObservation,
    SharedPlaneLink,
    register_axis,
)
from mc_photogrammetry.reconstruction.supported import plane_from_evidence  # noqa: E402

# Reuse supported_plane's simulated-annotation scaffolding rather than
# duplicating the "stand in for a user's two clicks" logic.
sys.path.insert(0, str(ROOT / "scripts"))
from supported_plane import simulate_annotation  # noqa: E402


def as_link(text: str) -> tuple[str, str]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected two frame ids separated by a comma, got {text!r}")
    return (parts[0], parts[1])


def as_quadrant(text: str) -> tuple[str, int]:
    frame, _, value = text.partition("=")
    if not value:
        raise argparse.ArgumentTypeError(
            f"expected frame_id=n, got {text!r}")
    try:
        return (frame, int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--axis", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--side", type=float, required=True, choices=(-1.0, 1.0))
    parser.add_argument("--link", type=as_link, action="append", default=[],
                        help="frame_a,frame_b: a stated assertion that these "
                             "two views see the same plane")
    parser.add_argument("--anchor", required=True,
                        help="the view fixed at position 0 on this axis")
    parser.add_argument("--quadrant", type=as_quadrant, action="append", default=[],
                        help="frame_id=n: quarter turns to add to this view's "
                             "yaw so its horizontal axes agree with the others' "
                             "(irrelevant for --axis 1, the vertical axis)")
    parser.add_argument("--simulate-annotations", action="store_true")
    parser.add_argument("--simulate-quadrant", action="store_true",
                        help="derive --quadrant from the manifest yaw instead "
                             "of requiring it on the command line; scaffolding, "
                             "standing in for a person reading it off two "
                             "photographs, recorded as such in the artifact")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    import cv2

    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.reconstruction.calibration import enhance
    from mc_photogrammetry.reconstruction.camera import (
        camera_eye, load_camera_manifest)
    from mc_photogrammetry.reconstruction.scene import load_scene_transform

    folder = ROOT / args.fixture
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    cameras = load_camera_manifest(folder / "cameras.json")
    scene = load_scene_transform(folder / "scene.json")
    reference = load_litematic(next(folder.glob("*.litematic")))

    quadrants = dict(args.quadrant)
    if args.simulate_quadrant:
        if quadrants:
            print("--quadrant and --simulate-quadrant both given; explicit "
                  "values win")
        for entry in calibration["estimates"]:
            frame_id = entry["frame"]
            if frame_id in quadrants:
                continue
            manifest_yaw = cameras[frame_id].yaw_degrees
            quadrants[frame_id] = round(
                (manifest_yaw - entry["yaw_degrees_mod90"]) / 90.0)

    observations = []
    plane_notes = {}
    for entry in calibration["estimates"]:
        frame_id = entry["frame"]
        path = folder / frame_id
        if not path.exists():
            continue
        size = tuple(calibration["inputs"][frame_id]["size"])
        grey = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2GRAY)
        boosted = enhance(grey)
        quadrant = quadrants.get(frame_id, 0)
        if args.axis != 1 and frame_id not in quadrants:
            print(f"  {frame_id}: no stated quadrant for a horizontal axis; "
                  f"assuming 0, which is not a claim this script can justify "
                  f"on its own")
        axes = axis_directions(entry["yaw_degrees_mod90"] + 90.0 * quadrant,
                               entry["pitch_degrees"],
                               entry.get("roll_degrees", 0.0))
        centre = entry.get("principal_point")
        centre = tuple(centre) if centre else None

        stated = None
        if args.simulate_annotations:
            annotation = simulate_annotation(reference, scene, cameras[frame_id],
                                              args.axis, size)
            if annotation and annotation.get("side") == args.side:
                from mc_photogrammetry.reconstruction.lattice import (
                    scale_from_known_length)
                constraint = scale_from_known_length(
                    annotation["points"][0], annotation["points"][1],
                    float(annotation["blocks"]), args.axis, axes,
                    entry["focal_pixels"], size, principal_point=centre)
                stated = constraint.distance

        if stated is None:
            print(f"  {frame_id}: no assisted distance on the requested side; "
                  f"skipped")
            continue

        plane = plane_from_evidence(boosted, args.axis, axes,
                                    entry["focal_pixels"], size,
                                    distance=stated, side=args.side,
                                    principal_point=centre)
        if plane is None:
            print(f"  {frame_id}: no plane measured; skipped")
            continue
        observations.append(PlaneObservation(frame_id, args.axis, args.side,
                                             plane.distance))
        plane_notes[frame_id] = plane.distance

    links = [SharedPlaneLink(a, b, args.axis) for a, b in args.link]
    result = register_axis(observations, links, args.axis, args.anchor)
    if result is None:
        print("\n  anchor made no observation on this axis; nothing to register")
        return 1

    print(f"\n  registered axis {args.axis}, side {args.side}, "
          f"anchor {args.anchor}")
    rows = []
    for view, position in sorted(result.positions.items()):
        row = {"view": view, "distance_to_plane": plane_notes.get(view),
              "solved_position": position}
        if args.score:
            actual = camera_eye(cameras[view])[args.axis]
            row["reference_eye_coordinate"] = float(actual)
        rows.append(row)

    if args.score and result.positions:
        anchor_actual = camera_eye(cameras[args.anchor])[args.axis]
        for row in rows:
            row["reference_offset_from_anchor"] = (
                row["reference_eye_coordinate"] - anchor_actual)
            row["registration_error_blocks"] = (
                row["solved_position"] - row["reference_offset_from_anchor"])

    for row in rows:
        line = (f"    {row['view']:<26} plane {row['distance_to_plane']:.3f}"
                f"  solved {row['solved_position']:+.3f}")
        if "reference_offset_from_anchor" in row:
            line += (f"  actual {row['reference_offset_from_anchor']:+.3f}"
                     f"  error {row['registration_error_blocks']:+.3f}")
        print(line)

    payload = {
        "fixture": args.fixture,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "axis": args.axis, "side": args.side, "anchor": args.anchor,
        "quadrants": quadrants,
        "quadrants_simulated": bool(args.simulate_quadrant),
        "links": [{"view_a": a, "view_b": b} for a, b in args.link],
        "redundant_links": [{"view_a": a, "view_b": b}
                            for a, b in result.redundant_links],
        "note": ("links are a stated assertion that two views see the same "
                 "plane, not something this script infers from appearance; "
                 "a wrong assertion between two single-observation views "
                 "cannot be caught by this module, see registration.py"),
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float),
                            encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
