"""Freeze seam landmarks, then measure how far a camera puts them from the truth.

The gate is a median seam displacement of 2 pixels at a 1,920-pixel long edge. I
previously reported an angle converted into a nominal displacement and argued
from it that no camera could pass; the lead ruled that a fitting residual is not
an independently measured seam reprojection error, and that ruling is accepted.
This measures the thing the gate actually names.

Two commands, deliberately separated so the second cannot influence the first:

``freeze``
    Choose 3D seam landmarks from the reference and write them with a digest.
    The rule reads only the reference geometry and the manifest cameras - never
    an estimate, a residual or a score - so membership cannot be steered by
    which points a candidate happens to get right. Run once; it refuses to
    overwrite.

``score``
    Project each frozen landmark twice and measure the pixel distance between
    the two images of it: once through the manifest camera, once through the
    estimated one. Normalised by ``1920 / max(width, height)`` as the ruling
    specifies.

**The translation is evaluation scaffolding and is labelled as such.** Vanishing
points recover orientation and focal length and no position at all, so the
estimated camera is given the manifest's eye. That isolates the orientation and
focal error, which is what an orientation-and-focal diagnostic should report, and
it is not a claim that position is known. A full reconstruction will later use
its own pose and must be measured again.

Landmarks are block corners on faces some camera can see - the corners *are* the
seams, being where one block's face meets the next. Nothing is dropped for being
short, hard or badly predicted.
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

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS  # noqa: E402
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    load_camera_manifest,
    project_points,
)
from mc_photogrammetry.reconstruction.scene import load_scene_transform  # noqa: E402

REFERENCE_EDGE = 1920.0
"""The long edge the gate's two pixels are quoted at."""

PER_FRAME = 400
"""Landmarks kept per frame.

Enough that a median is stable, few enough that the frozen list stays readable.
Chosen by a fixed stride through a sorted list, so it is a thinning and not a
selection.
"""


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def visible_cells_for(reference, scene, cameras, sizes, palette):
    from mc_photogrammetry.export.visibility import (
        full_cube_keys,
        shape_table,
        visible_cells,
    )

    return visible_cells(reference.blocks, scene, cameras, sizes,
                         full_cubes=full_cube_keys(palette),
                         shapes=shape_table(palette))


def corners_of(coordinate, scene) -> list[tuple[float, float, float]]:
    """The eight corners of one cell, in world coordinates.

    A block corner is where seams meet, so it is the sharpest thing in frame and
    the least ambiguous to reproject.
    """
    origin = np.asarray(scene.to_world(coordinate), dtype=float)
    return [tuple(origin + np.asarray(offset, dtype=float))
            for offset in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                           (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))]


def freeze(args) -> int:
    import os

    from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a freeze is never re-run")

    folder = ROOT / args.fixture
    reference = load_litematic(next(folder.glob("*.litematic")))
    scene = load_scene_transform(folder / "scene.json")
    cameras = load_camera_manifest(folder / "cameras.json")

    from PIL import Image
    sizes, frames = {}, {}
    for path in sorted(folder.glob("*.png")):
        if path.name in cameras:
            sizes[path.name] = Image.open(path).size
            frames[path.name] = cameras[path.name]
    if not frames:
        raise SystemExit("no frames with cameras")

    provider = VanillaJarTextureProvider()
    provider.load_jar(Path(os.environ["APPDATA"])
                      / ".minecraft/versions/1.21.5/1.21.5.jar")
    palette = provider.complete_palette()
    reachable = visible_cells_for(reference, scene, frames, sizes, palette)

    per_frame = {}
    for frame_id, camera in sorted(frames.items()):
        width, height = sizes[frame_id]
        found = []
        # Sorted so the order depends on the fixture alone, then thinned by a
        # fixed stride. No score, residual or estimate is consulted.
        for coordinate in sorted(reachable):
            for point in corners_of(coordinate, scene):
                projected = project_points([point], camera,
                                           width=width, height=height)[0]
                if projected is None:
                    continue
                x, y, depth = projected
                if not (0 <= x < width and 0 <= y < height) or depth <= 0.5:
                    continue
                found.append({"world": list(point), "pixel": [x, y],
                              "depth": depth, "cell": list(coordinate)})
        if not found:
            continue
        stride = max(1, len(found) // args.per_frame)
        per_frame[frame_id] = found[::stride][:args.per_frame]

    payload = {
        "fixture": args.fixture,
        "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": ("block corners of camera-reachable cells, in sorted cell order, "
                 "thinned by a fixed stride; no estimate, residual or score was "
                 "consulted"),
        "per_frame_cap": args.per_frame,
        "reference_edge_pixels": REFERENCE_EDGE,
        "sizes": {k: list(v) for k, v in sizes.items()},
        "landmarks": per_frame,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["digest"] = digest_bytes(body.encode("utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total = sum(len(v) for v in per_frame.values())
    print(f"{args.fixture}: froze {total} landmarks across {len(per_frame)} frames "
          f"({', '.join(f'{k[-12:]}:{len(v)}' for k, v in sorted(per_frame.items()))})")
    print(f"  digest {payload['digest'][:16]}  ->  {out}")
    return 0


def score(args) -> int:
    frozen = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    estimates = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    folder = ROOT / frozen["fixture"]
    cameras = load_camera_manifest(folder / "cameras.json")
    rows = {row["frame"]: row for row in estimates["estimates"]}

    print(f"{frozen['fixture']}: {sum(len(v) for v in frozen['landmarks'].values())} "
          f"frozen landmarks, digest {frozen.get('digest', '?')[:16]}")
    print(f"\n{'frame':<30}{'n':>6}{'median px':>11}{'p90':>8}"
          f"{'normalised':>12}{'gate':>7}")

    everything, per_frame = [], {}
    for frame_id, landmarks in sorted(frozen["landmarks"].items()):
        truth = cameras.get(frame_id)
        estimate = rows.get(frame_id)
        if truth is None or estimate is None:
            continue
        width, height = frozen["sizes"][frame_id]
        scale = REFERENCE_EDGE / max(width, height)

        world = [tuple(item["world"]) for item in landmarks]
        wanted = project_points(world, truth, width=width, height=height)

        # One view recovers yaw only modulo 90 degrees, because the two
        # horizontal lattice axes are interchangeable without semantic cues. The
        # gate says to account for lattice-axis symmetries, so the four quarter
        # turns are tried and the best is used. That is a discrete symmetry of
        # the world - four options, not a free parameter - and the quadrant that
        # was chosen is reported per frame rather than hidden.
        best = None
        for quadrant in range(4):
            candidate = CameraModel(
                # The manifest's eye, held equal for both cameras. Vanishing
                # points recover no position at all, so this isolates the
                # orientation and focal error. Evaluation scaffolding, not a
                # claim that position is known.
                position=truth.position,
                yaw_degrees=estimate["yaw_degrees_mod90"] + 90.0 * quadrant,
                pitch_degrees=estimate["pitch_degrees"],
                fov_degrees=estimate["fov_degrees"],
                roll_degrees=estimate.get("roll_degrees", 0.0),
                eye_height=truth.eye_height,
            )
            got = project_points(world, candidate, width=width, height=height)
            found = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
                     for a, b in zip(wanted, got) if a is not None and b is not None]
            if not found:
                continue
            if best is None or np.median(found) < np.median(best[1]):
                best = (quadrant, found)
        if best is None:
            print(f"  {frame_id:<28}{0:>6}   nothing projected by both cameras")
            continue
        quadrant, displacements = best

        values = np.asarray(displacements)
        normalised = values * scale
        median = float(np.median(normalised))
        passed = median <= 2.0
        per_frame[frame_id] = {
            "yaw_quadrant_chosen": quadrant,
            "landmarks": len(landmarks),
            "projected_by_both": int(values.size),
            "coverage": float(values.size / len(landmarks)),
            "median_px": float(np.median(values)),
            "p90_px": float(np.percentile(values, 90)),
            "median_normalised_px": median,
            "within_gate": bool(passed),
        }
        everything.extend(normalised.tolist())
        print(f"  {frame_id:<28}{values.size:>6}{np.median(values):>11.2f}"
              f"{np.percentile(values, 90):>8.2f}{median:>12.2f}"
              f"{'PASS' if passed else 'FAIL':>7}   +{90 * quadrant}deg")

    if everything:
        pooled = float(np.median(everything))
        print(f"\n  pooled median normalised displacement over "
              f"{len(everything)} landmarks: {pooled:.2f} px "
              f"({'PASS' if pooled <= 2.0 else 'FAIL'} against the 2 px gate)")
        print("  each frame's yaw quadrant is chosen from the four quarter turns "
              "the lattice permits and reported above;")
        print("  no landmark was dropped for being short, hard or badly predicted")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "fixture": frozen["fixture"],
            "landmark_digest": frozen.get("digest"),
            "calibration_artifact": str(args.calibration),
            "gate": {"median_normalised_px": 2.0,
                     "reference_edge": REFERENCE_EDGE},
            "translation": "manifest eye used for both cameras; evaluation "
                           "scaffolding, since vanishing points recover no position",
            "per_frame": per_frame,
            "pooled_median_normalised_px": float(np.median(everything))
            if everything else None,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("freeze", help="choose and record landmarks; run once")
    a.add_argument("fixture")
    a.add_argument("--out", required=True)
    a.add_argument("--per-frame", type=int, default=PER_FRAME)
    a.set_defaults(func=freeze)

    b = sub.add_parser("score", help="measure displacement against frozen landmarks")
    b.add_argument("--landmarks", required=True)
    b.add_argument("--calibration", required=True)
    b.add_argument("--out")
    b.set_defaults(func=score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
