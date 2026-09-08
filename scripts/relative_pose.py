"""Recover relative pose between views with CPU PyCOLMAP, or say why it cannot.

Calibration fixes where each camera points and how wide its lens is, and says
nothing about where any of them stands. Two views with correspondences between
them fix the direction from one to the other - never its length, which no pair of
images can supply - and that is what this measures.

Minecraft is close to the worst case for feature matching. A wall of forty
identical blocks offers forty identical descriptors, so a match is as likely to
join the wrong pair of corners as the right one. This is therefore written to
report failure precisely rather than to produce a number: PyCOLMAP's two-view
configuration distinguishes a genuine calibrated pair from a **panoramic** one
(the cameras only rotated, so no translation exists to find), a **planar** one
(everything visible lies on one surface, so the geometry is ambiguous) and a
**degenerate** one. Those come back as unresolved, not as a fabricated baseline.

Three configurations are declared here, before any of them has been run, so that
the one that happens to work cannot be chosen afterwards and presented as the
plan:

* ``baseline`` - PyCOLMAP's own defaults.
* ``dense`` - more features at a lower peak threshold, because these frames are
  dim and low-contrast; a fixture shot at gamma 100% means about 34 of 255.
* ``strict`` - a tighter ratio test with cross-checking, aimed squarely at the
  repetition, accepting fewer matches to reduce wrong ones.

Runs under the isolated CPU SfM environment, which has only pycolmap and numpy::

    .venv-sfm/Scripts/python.exe scripts/relative_pose.py "templates 2" \\
        --calibration docs/reports/M3-calibration-templates-2.json --out report.json

Intrinsics come from that calibration artifact - image-derived, never a manifest.
``--score`` compares the recovered *rotations* against the manifest afterwards
and labels the comparison as evaluation scaffolding.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]

CONFIGURATIONS = {
    "baseline": {
        "extraction": {},
        "matching": {},
        "why": "PyCOLMAP defaults, so the others have something to be compared against",
    },
    "dense": {
        "extraction": {"max_num_features": 16384, "peak_threshold": 0.004,
                       "edge_threshold": 12.0},
        "matching": {},
        "why": "these frames are dim and low contrast; more, weaker features",
    },
    "strict": {
        "extraction": {},
        "matching": {"max_ratio": 0.7, "cross_check": True},
        "why": "Minecraft repeats textures, so trade recall for fewer wrong matches",
    },
}

# `TwoViewGeometry.config` comes back as a plain int, so comparing it against a
# set of names silently matches nothing and every configuration reads as
# resolved. A PLANAR_OR_PANORAMIC pair was accepted that way before this map
# existed.
CONFIG_NAMES = {
    int(getattr(pycolmap.TwoViewGeometryConfiguration, name)): name
    for name in ("UNDEFINED", "DEGENERATE", "CALIBRATED", "UNCALIBRATED",
                 "PLANAR", "PANORAMIC", "PLANAR_OR_PANORAMIC", "WATERMARK",
                 "MULTIPLE")
}

UNRESOLVED = {"PANORAMIC", "PLANAR", "PLANAR_OR_PANORAMIC", "DEGENERATE",
              "WATERMARK", "UNDEFINED", "MULTIPLE"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def camera_for(entry: dict) -> pycolmap.Camera:
    """A pinhole camera from the image-derived calibration, not from a manifest."""
    width, height = entry["size"]
    focal = entry["focal_pixels"]
    centre = entry["principal_point"] or [width / 2.0, height / 2.0]
    camera = pycolmap.Camera.create(1, "PINHOLE", focal, width, height)
    # PINHOLE takes (fx, fy, cx, cy); the measured optical centre goes in here
    # rather than the image middle, which is the whole point of refining it.
    camera.params = [focal, focal, float(centre[0]), float(centre[1])]
    return camera


def rotation_from(yaw_degrees, pitch_degrees, roll_degrees=0.0) -> np.ndarray:
    """World-to-camera rotation, matching the project's stated convention.

    Restated here rather than imported: this runs under an isolated interpreter
    that has neither the project nor OpenCV installed.
    """
    yaw, pitch, roll = np.radians([yaw_degrees, pitch_degrees, roll_degrees])
    forward = np.array([-np.sin(yaw), 0.0, np.cos(yaw)])
    right = np.array([-np.cos(yaw), 0.0, -np.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    pitched_forward = np.cos(pitch) * forward - np.sin(pitch) * up
    pitched_up = np.sin(pitch) * forward + np.cos(pitch) * up
    rolled_right = np.cos(roll) * right - np.sin(roll) * pitched_up
    rolled_up = np.sin(roll) * right + np.cos(roll) * pitched_up
    return np.stack([rolled_right, rolled_up, pitched_forward])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    trace = np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def run(fixture: Path, calibration: dict, name: str, workspace: Path,
        masks: Path | None = None) -> dict:
    options = CONFIGURATIONS[name]
    database = workspace / f"{name}.db"
    if database.exists():
        database.unlink()

    entries = {row["frame"]: row for row in calibration["estimates"]}
    for frame, row in entries.items():
        row["size"] = calibration["inputs"][frame]["size"]
    names = sorted(entries)

    started = time.time()
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = False
    for key, value in options["extraction"].items():
        setattr(extraction.sift, key, value)
    reader = pycolmap.ImageReaderOptions()
    if masks is not None:
        reader.mask_path = str(masks)
    pycolmap.extract_features(database, fixture, names,
                              camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              reader_options=reader,
                              extraction_options=extraction,
                              device=pycolmap.Device.cpu)
    extracted = time.time() - started

    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = False
    for key, value in options["matching"].items():
        setattr(matching.sift, key, value)
    pycolmap.match_exhaustive(database, matching_options=matching)
    matched = time.time() - started - extracted

    handle = pycolmap.Database.open(str(database))
    images = {image.name: image for image in handle.read_all_images()}
    keypoints = {name_: handle.read_keypoints(images[name_].image_id) for name_ in names
                 if name_ in images}
    features = {name_: len(points) for name_, points in keypoints.items()}

    pairs, edges = [], []
    for first, second in itertools.combinations(names, 2):
        if first not in images or second not in images:
            continue
        raw = handle.read_matches(images[first].image_id, images[second].image_id)
        record = {"views": [first, second], "raw_matches": int(len(raw))}
        if len(raw) < 8:
            record.update(resolved=False, reason="too few raw matches",
                          inliers=0, configuration="NONE")
            pairs.append(record)
            continue

        points1 = np.asarray(keypoints[first], dtype=np.float64)[:, :2]
        points2 = np.asarray(keypoints[second], dtype=np.float64)[:, :2]
        geometry = pycolmap.estimate_calibrated_two_view_geometry(
            camera_for(entries[first]), points1,
            camera_for(entries[second]), points2,
            np.asarray(raw, dtype=np.uint32))
        configuration = CONFIG_NAMES.get(int(geometry.config),
                                         f"UNKNOWN_{int(geometry.config)}")
        inliers = int(len(geometry.inlier_matches))
        record.update(inliers=inliers, configuration=configuration,
                      triangulation_angle_deg=float(geometry.tri_angle))

        if configuration in UNRESOLVED or inliers < 15:
            record.update(
                resolved=False,
                reason=("no translation exists between these views"
                        if configuration.startswith("PAN")
                        else f"configuration {configuration} with {inliers} inliers"),
            )
            pairs.append(record)
            continue

        # The geometry carries E, F and H; turning one into a pose is a separate
        # step that can fail on its own, and a pair whose pose cannot be
        # recovered is unresolved rather than an error.
        if geometry.cam2_from_cam1 is None:
            pycolmap.estimate_two_view_geometry_pose(
                camera_for(entries[first]), points1,
                camera_for(entries[second]), points2, geometry)
        pose = geometry.cam2_from_cam1
        if pose is None:
            record.update(resolved=False,
                          reason=f"{configuration} geometry yielded no pose")
            pairs.append(record)
            continue

        direction = np.asarray(pose.translation, dtype=float)
        norm = float(np.linalg.norm(direction))
        record.update(
            resolved=True,
            relative_rotation=np.asarray(pose.rotation.matrix()).tolist(),
            translation_direction=(direction / norm).tolist() if norm > 0 else None,
            scale="unresolved: two views fix direction only",
        )
        edges.append((first, second))
        pairs.append(record)
    handle.close()

    # Connected components over the pairs that actually resolved.
    parent = {name_: name_ for name_ in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    components: dict[str, list[str]] = {}
    for name_ in names:
        components.setdefault(find(name_), []).append(name_)
    grouped = sorted(components.values(), key=len, reverse=True)

    return {
        "configuration": name,
        "why": options["why"],
        "settings": {"extraction": options["extraction"],
                     "matching": options["matching"]},
        "seconds": {"extraction": extracted, "matching": matched},
        "features": features,
        "pairs": pairs,
        "resolved_pairs": len(edges),
        "components": grouped,
        "largest_component": len(grouped[0]) if grouped else 0,
        "registered": sorted(set(itertools.chain.from_iterable(
            group for group in grouped if len(group) > 1))),
        "excluded": sorted(group[0] for group in grouped if len(group) == 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="artifact from scripts/calibrate_frames.py")
    parser.add_argument("--config", choices=sorted(CONFIGURATIONS) + ["all"],
                        default="all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--masks", type=Path,
                        help="directory of COLMAP masks, from "
                             "scripts/overlay_mask.py; excludes the debug overlay")
    parser.add_argument("--score", action="store_true",
                        help="compare recovered rotations with cameras.json; "
                             "evaluation only")
    args = parser.parse_args(argv)

    folder = ROOT / args.fixture
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    wanted = sorted(CONFIGURATIONS) if args.config == "all" else [args.config]

    workspace = Path(tempfile.mkdtemp(prefix="sfm-"))
    runs = []
    try:
        for name in wanted:
            print(f"[{name}] {CONFIGURATIONS[name]['why']}", flush=True)
            result = run(folder, calibration, name, workspace, args.masks)
            runs.append(result)
            print(f"    {result['resolved_pairs']}/{len(result['pairs'])} pairs resolved, "
                  f"largest component {result['largest_component']}/"
                  f"{len(result['features'])} views, "
                  f"{result['seconds']['extraction'] + result['seconds']['matching']:.0f}s",
                  flush=True)
            for pair in result["pairs"]:
                mark = "ok  " if pair.get("resolved") else "SKIP"
                print(f"      {mark} {pair['views'][0][-12:]} - {pair['views'][1][-12:]}"
                      f"  raw {pair['raw_matches']:5d}  inliers {pair.get('inliers', 0):5d}"
                      f"  {pair.get('configuration', '-'):<20}"
                      f"  {pair.get('reason', '')}", flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    payload = {
        "fixture": args.fixture,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {"python": platform.python_version(),
                        "pycolmap": pycolmap.__version__,
                        "numpy": np.__version__, "device": "cpu"},
        "calibration_artifact": {"path": str(args.calibration),
                                 "sha256": digest(args.calibration)},
        "declared_configurations": {k: v["why"] for k, v in CONFIGURATIONS.items()},
        "masks": str(args.masks) if args.masks else None,
        "note": "translation is recovered as a direction only; two views cannot "
                "fix its length, and no scale is claimed",
        "runs": runs,
    }

    if args.score:
        manifest = folder / "cameras.json"
        poses = json.loads(manifest.read_text(encoding="utf-8"))
        scored = []
        for result in runs:
            for pair in result["pairs"]:
                if not pair.get("resolved"):
                    continue
                a, b = pair["views"]
                if a not in poses or b not in poses:
                    continue
                first = rotation_from(poses[a]["yaw_degrees"], poses[a]["pitch_degrees"])
                second = rotation_from(poses[b]["yaw_degrees"], poses[b]["pitch_degrees"])
                got = np.asarray(pair["relative_rotation"], dtype=float)
                row = {"configuration": result["configuration"],
                       "views": [a, b],
                       "two_view_config": pair["configuration"],
                       "inliers": pair["inliers"],
                       "rotation_error_deg": angle_between(got, second @ first.T)}

                # Translation has no scale, but its *direction* is comparable.
                # The true baseline in the first camera's frame is the second
                # eye minus the first, rotated into that frame.
                if pair.get("translation_direction"):
                    baseline = (np.asarray(poses[b]["position"], dtype=float)
                                - np.asarray(poses[a]["position"], dtype=float))
                    length = float(np.linalg.norm(baseline))
                    if length > 1e-6:
                        wanted = first @ (baseline / length)
                        mine = np.asarray(pair["translation_direction"], dtype=float)
                        cosine = float(np.clip(np.dot(mine, wanted), -1.0, 1.0))
                        row["baseline_metres"] = length
                        row["translation_direction_error_deg"] = float(
                            np.degrees(np.arccos(abs(cosine))))
                        row["direction_sign_ambiguous"] = bool(cosine < 0)
                scored.append(row)
        payload["scored"] = {
            "note": "evaluation only; the manifest is scaffolding, never read "
                    "during estimation. Translation is compared as a direction "
                    "and only up to sign, because two views fix no scale.",
            "manifest_sha256": digest(manifest),
            "pairs": scored,
        }
        if scored:
            rotations = [row["rotation_error_deg"] for row in scored]
            directions = [row["translation_direction_error_deg"] for row in scored
                          if "translation_direction_error_deg" in row]
            print(f"\n  scored {len(scored)} resolved pairs; median relative "
                  f"rotation error {np.median(rotations):.2f} deg")
            if directions:
                print(f"  median translation direction error "
                      f"{np.median(directions):.2f} deg over {len(directions)} pairs")
        else:
            print("\n  nothing resolved, so nothing to score")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
