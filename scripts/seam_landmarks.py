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
    camera_space_points,
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


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def project_with_intrinsics(points, camera, focal, principal_point):
    """Project through an explicit focal length and optical centre.

    `CameraModel` has nowhere to put a principal point, so rebuilding a candidate
    from its field of view silently drops the measured optical centre and puts it
    back at the image middle. On a 100x100 crop with centre (30, 50) that is 20
    raw pixels of error - 384 once normalised - on a camera that is otherwise
    exactly right. Every intrinsic has to be carried, so the projection is done
    here rather than through a model that cannot hold them.
    """
    centre_x, centre_y = principal_point
    out = []
    for x, y, depth in camera_space_points(points, camera):
        if depth <= 1e-9:
            out.append(None)
            continue
        out.append((centre_x + x * focal / depth, centre_y - y * focal / depth, depth))
    return out


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


def visible_depths(reference, scene, camera, width, height, palette, scale=0.35):
    """A depth buffer for one frame, so a corner can be tested for occlusion."""
    from mc_photogrammetry.export.attribution import (
        face_colours,
        face_shapes,
        render_volume,
    )

    _canvas, _owner, _index, depth = render_volume(
        reference.blocks, scene, camera, width, height,
        face_colours(palette), face_shapes(palette), scale)
    return depth, scale


def freeze(args) -> int:
    import os

    from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a freeze is never re-run")

    folder = ROOT / args.fixture
    # The reference is named, not discovered. `next(glob(...))` silently picks
    # whichever litematic the filesystem offers first, and a fixture folder can
    # hold more than one.
    if args.reference:
        reference_path = Path(args.reference)
    else:
        candidates = sorted(folder.glob("*.litematic"))
        if len(candidates) != 1:
            raise SystemExit(
                f"{len(candidates)} litematics in {folder}; name one with --reference")
        reference_path = candidates[0]
    reference = load_litematic(reference_path)
    scene = load_scene_transform(folder / "scene.json")
    cameras = load_camera_manifest(folder / "cameras.json")

    from PIL import Image
    sizes, frames = {}, {}
    for path in sorted(folder.glob("*.png")):
        if path.name in cameras:
            sizes[path.name] = Image.open(path).size
            frames[path.name] = cameras[path.name]
    if args.frames:
        wanted = set(args.frames)
        missing = wanted - set(frames)
        if missing:
            raise SystemExit(f"requested frames are absent: {sorted(missing)}")
        frames = {k: v for k, v in frames.items() if k in wanted}
        sizes = {k: v for k, v in sizes.items() if k in wanted}
    if not frames:
        raise SystemExit("no frames with cameras")

    provider = VanillaJarTextureProvider()
    jar = Path(os.environ["APPDATA"]) / ".minecraft/versions/1.21.5/1.21.5.jar"
    provider.load_jar(jar)
    palette = provider.complete_palette()

    per_frame, rejected = {}, {}
    for frame_id, camera in sorted(frames.items()):
        width, height = sizes[frame_id]
        # Visibility is decided *per frame*. Unioning it across frames admitted
        # corners that some other camera could see, which is not the same claim.
        reachable = visible_cells_for(reference, scene, {frame_id: camera},
                                      {frame_id: (width, height)}, palette)
        depth_buffer, scale = visible_depths(reference, scene, camera,
                                             width, height, palette)
        found, occluded = [], 0
        for coordinate in sorted(reachable):
            for point in corners_of(coordinate, scene):
                projected = project_points([point], camera,
                                           width=width, height=height)[0]
                if projected is None:
                    continue
                x, y, depth = projected
                if not (0 <= x < width and 0 <= y < height) or depth <= 0.5:
                    continue
                # And the corner itself must be the nearest thing along its ray,
                # not merely a corner of a cell that is visible somewhere else.
                row = int(min(max(y * scale, 0), depth_buffer.shape[0] - 1))
                column = int(min(max(x * scale, 0), depth_buffer.shape[1] - 1))
                nearest = float(depth_buffer[row, column])
                if np.isfinite(nearest) and depth > nearest + args.depth_slack:
                    occluded += 1
                    continue
                found.append({"world": list(point), "pixel": [x, y],
                              "depth": depth, "cell": list(coordinate)})
        rejected[frame_id] = occluded
        if not found:
            continue
        stride = max(1, len(found) // args.per_frame)
        per_frame[frame_id] = found[::stride][:args.per_frame]

    payload = {
        "fixture": args.fixture,
        "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": ("block corners of cells visible *in that frame*, each tested "
                 "against the frame's own depth buffer for occlusion, in sorted "
                 "cell order, thinned by a fixed stride; no estimate, residual "
                 "or score was consulted"),
        "expected_frames": sorted(per_frame),
        "per_frame_cap": args.per_frame,
        "depth_slack": args.depth_slack,
        "reference_edge_pixels": REFERENCE_EDGE,
        "sizes": {k: list(v) for k, v in sizes.items()},
        "sources": {
            "reference": {"path": str(reference_path),
                          "sha256": digest_file(reference_path)},
            "scene": {"sha256": digest_file(folder / "scene.json")},
            "cameras": {"sha256": digest_file(folder / "cameras.json")},
            "jar": {"sha256": digest_file(jar)},
        },
        "occluded_corners_rejected": rejected,
        "landmarks": per_frame,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["digest"] = digest_bytes(body.encode("utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total = sum(len(v) for v in per_frame.values())
    print(f"{args.fixture}: froze {total} landmarks across {len(per_frame)} frames; "
          f"{sum(rejected.values())} occluded corners rejected")
    print(f"  reference {reference_path.name}, digest {payload['digest'][:16]} -> {out}")
    return 0


def score(args) -> int:
    landmark_path = Path(args.landmarks)
    frozen = json.loads(landmark_path.read_text(encoding="utf-8"))

    # The digest is verified, not merely stored. A frozen set that can be edited
    # after the fact is not frozen.
    stored = frozen.pop("digest", None)
    recomputed = digest_bytes(
        json.dumps(frozen, indent=2, sort_keys=True).encode("utf-8"))
    frozen["digest"] = stored
    if stored is None:
        raise SystemExit(f"{landmark_path} carries no digest")
    if stored != recomputed:
        raise SystemExit(
            f"{landmark_path} has been modified since it was frozen:\n"
            f"  stored     {stored}\n  recomputed {recomputed}")

    estimates = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    folder = ROOT / frozen["fixture"]
    cameras = load_camera_manifest(folder / "cameras.json")
    rows = {row["frame"]: row for row in estimates["estimates"]}

    expected = frozen.get("expected_frames") or sorted(frozen["landmarks"])
    print(f"{frozen['fixture']}: {sum(len(v) for v in frozen['landmarks'].values())} "
          f"frozen landmarks over {len(expected)} frames")
    print(f"  digest verified {stored[:16]}")
    header = ("frame", "n", "lost", "cover", "median px", "p90", "normalised", "gate")
    print(f"\n{header[0]:<30}{header[1]:>6}{header[2]:>6}{header[3]:>7}"
          f"{header[4]:>11}{header[5]:>8}{header[6]:>12}{header[7]:>7}   yaw quadrant")

    everything, per_frame, failures = [], {}, []
    for frame_id in expected:
        landmarks = frozen["landmarks"].get(frame_id, [])
        truth = cameras.get(frame_id)
        estimate = rows.get(frame_id)
        if not landmarks or truth is None or estimate is None:
            # A frame the frozen set expects and the candidate cannot answer is a
            # failure of the candidate. Skipping it would let a run improve its
            # score by producing fewer estimates.
            reason = ("no landmarks" if not landmarks else
                      "no manifest camera" if truth is None else
                      "no calibration estimate")
            failures.append({"frame": frame_id, "reason": reason})
            print(f"  {frame_id:<28}{len(landmarks):>6}{'-':>6}{'-':>7}"
                  f"   FAILED: {reason}")
            continue

        width, height = frozen["sizes"][frame_id]
        scale = REFERENCE_EDGE / max(width, height)
        world = [tuple(item["world"]) for item in landmarks]
        wanted = project_points(world, truth, width=width, height=height)

        focal = estimate["focal_pixels"]
        centre = estimate.get("principal_point") or [width / 2.0, height / 2.0]
        best = None
        for quadrant in range(4):
            candidate = CameraModel(
                position=truth.position,
                yaw_degrees=estimate["yaw_degrees_mod90"] + 90.0 * quadrant,
                pitch_degrees=estimate["pitch_degrees"],
                fov_degrees=estimate["fov_degrees"],
                roll_degrees=estimate.get("roll_degrees", 0.0),
                eye_height=truth.eye_height,
            )
            # Every intrinsic is carried, including the measured optical centre,
            # which a field of view cannot express.
            got = project_with_intrinsics(world, candidate, focal, centre)
            found = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
                     for a, b in zip(wanted, got) if a is not None and b is not None]
            if not found:
                continue
            if best is None or np.median(found) < np.median(best[1]):
                best = (quadrant, found)
        if best is None:
            failures.append({"frame": frame_id, "reason": "nothing projected"})
            print(f"  {frame_id:<28}{len(landmarks):>6}{len(landmarks):>6}"
                  f"{0.0:>7.0%}   FAILED: nothing projected")
            continue

        quadrant, displacements = best
        values = np.asarray(displacements)
        lost = len(landmarks) - values.size
        coverage = values.size / len(landmarks)
        normalised = values * scale
        median = float(np.median(normalised))
        # Coverage is part of the verdict. A frame answering a tenth of its
        # landmarks has not passed, however small that tenth's displacement.
        passed = median <= 2.0 and coverage >= args.minimum_coverage
        per_frame[frame_id] = {
            "yaw_quadrant_chosen": quadrant,
            "landmarks": len(landmarks),
            "projected_by_both": int(values.size),
            "not_projected": int(lost),
            "coverage": float(coverage),
            "median_px": float(np.median(values)),
            "p90_px": float(np.percentile(values, 90)),
            "median_normalised_px": median,
            "within_gate": bool(passed),
        }
        everything.extend(normalised.tolist())
        print(f"  {frame_id:<28}{values.size:>6}{lost:>6}{coverage:>7.0%}"
              f"{np.median(values):>11.2f}{np.percentile(values, 90):>8.2f}"
              f"{median:>12.2f}{'PASS' if passed else 'FAIL':>7}   +{90 * quadrant}deg")

    passed_frames = sum(1 for v in per_frame.values() if v["within_gate"])
    print(f"\n  {passed_frames}/{len(expected)} frames pass "
          f"(coverage floor {args.minimum_coverage:.0%}); "
          f"{len(failures)} could not be scored and count as failures")
    if everything:
        pooled = float(np.median(everything))
        print(f"  pooled median normalised displacement over {len(everything)} "
              f"landmarks: {pooled:.2f} px against the 2 px gate")
    print("  each frame's yaw quadrant is chosen from the four quarter turns the "
          "lattice permits and is reported above;")
    print("  no landmark was dropped for being short, hard or badly predicted")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "fixture": frozen["fixture"],
            "landmark_digest": stored,
            "digest_verified": True,
            "expected_frames": expected,
            "sources": frozen.get("sources"),
            "calibration_artifact": {"path": str(args.calibration),
                                     "sha256": digest_file(args.calibration)},
            "gate": {"median_normalised_px": 2.0,
                     "reference_edge": REFERENCE_EDGE,
                     "minimum_coverage": args.minimum_coverage},
            "translation": "manifest eye used for both cameras; evaluation "
                           "scaffolding, since vanishing points recover no position",
            "frames_passed": passed_frames,
            "frames_expected": len(expected),
            "failures": failures,
            "per_frame": per_frame,
            "pooled_median_normalised_px": float(np.median(everything))
            if everything else None,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0 if passed_frames == len(expected) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("freeze", help="choose and record landmarks; run once")
    a.add_argument("fixture")
    a.add_argument("--out", required=True)
    a.add_argument("--per-frame", type=int, default=PER_FRAME)
    a.add_argument("--reference", help="the litematic to freeze against, "
                                       "named rather than discovered")
    a.add_argument("--frames", nargs="*", help="restrict to these frames")
    a.add_argument("--depth-slack", type=float, default=0.25,
                   help="blocks of depth tolerance when testing a corner "
                        "for occlusion against its frame depth buffer")
    a.set_defaults(func=freeze)

    b = sub.add_parser("score", help="measure displacement against frozen landmarks")
    b.add_argument("--landmarks", required=True)
    b.add_argument("--calibration", required=True)
    b.add_argument("--out")
    b.add_argument("--minimum-coverage", type=float, default=0.9,
                   help="fraction of a frame's landmarks that must "
                        "project through both cameras before it can pass")
    b.set_defaults(func=score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
