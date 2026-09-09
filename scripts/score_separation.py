"""Does a per-cell occupancy cue separate real cells from not? Measured broadly.

The earlier prototype (`supported_plane.py`) found a promising score in one
hand-picked window. That is a selected example, not a sample, and this script
is the correction: for every cell the cue actually renders a judgment on
(`supported`, `contradicted` or `foreground_occlusion` - not `weak_evidence` or
`outside_image`, which are the cue declining to guess), reconstruct its world
location from the real camera pose (read here only, to evaluate) and ask
whether a genuine, camera-reachable reference block face sits there. Then
compare the score distribution of truth-positive against truth-negative cells
across every view with usable data, and report the AUC - 0.5 is no separation
at all, 1.0 is perfect, computed with proper tie handling (see `_average_ranks`
- the plain argsort version silently gave 0.0 for an all-tied input whose
correct AUC is 0.5).

The world-location reconstruction uses `lattice.rectified_pixel_to_plane_coordinates`,
`lattice.plane_coordinates_to_camera_point` and `camera.camera_point_to_world`
rather than a hand-rolled copy of that mapping - those are tested against
`rectify_plane`'s own internal construction in `test_lattice.py` and
`test_camera.py`.

Two cues are implemented, selected with ``--cue``:

* ``phase`` - local-window phase agreement, the corrected cue from
  `supported.py`'s addendum.
* ``brightness`` - does a cell's mean brightness match the median of its
  immediate neighbours, on the reasoning that a doorway onto a different,
  differently-lit room (or a window onto the sky) should be a local outlier.

Both measured at chance (phase: AUC 0.523; brightness: AUC 0.502) on Template
2's axis 1, side -1 floor, before the five-way verdict contract and the AUC
tie fix - see the second addendum in
`docs/reports/M4-supported-plane-2026-09-08.md`. Numbers here supersede that
run; re-measure rather than trust the old ones after either fix.

    python scripts/score_separation.py "templates 2" \\
        --calibration docs/reports/M3-calibration-templates-2.json \\
        --axis 1 --side -1 --cue phase
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _average_ranks(values):
    """1-indexed ranks with ties given the mean of the positions they span.

    Plain ``argsort``-based ranking breaks ties by array position, which is
    silently wrong for the Mann-Whitney AUC formula: with the positive class
    concatenated first, a degenerate all-tied input (every score identical)
    got every positive the lowest ranks and every negative the highest,
    scoring AUC 0.0 for a case with zero actual separation - the correct
    answer is 0.5. Verified against that exact case in this script's own tests.
    """
    import numpy as np

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _auc_with_ties(pos, neg) -> float:
    """Mann-Whitney AUC, correct under ties (see `_average_ranks`)."""
    import numpy as np

    combined = np.concatenate([pos, neg])
    ranks = _average_ranks(combined)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--axis", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--side", type=float, default=-1.0, choices=(-1.0, 1.0))
    parser.add_argument("--cue", choices=("phase", "brightness"), default="phase")
    parser.add_argument("--local-window-periods", type=float, default=1.5)
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.calibration import axis_directions, enhance
    from mc_photogrammetry.reconstruction.camera import (
        load_camera_manifest, camera_point_to_world, camera_eye)
    from mc_photogrammetry.reconstruction.scene import load_scene_transform
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.export.visibility import visible_cells
    from mc_photogrammetry.reconstruction.lattice import (
        measure_plane, rectify_plane, scale_from_known_length,
        rectified_pixel_to_plane_coordinates, plane_coordinates_to_camera_point)
    from mc_photogrammetry.reconstruction.supported import (
        support_cells, _edge_profiles, phase_at_period,
        SUPPORTED, CONTRADICTED, FOREGROUND_OCCLUSION)
    from supported_plane import simulate_annotation

    JUDGED_VERDICTS = {SUPPORTED, CONTRADICTED, FOREGROUND_OCCLUSION}

    folder = ROOT / args.fixture
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    cameras = load_camera_manifest(folder / "cameras.json")
    scene = load_scene_transform(folder / "scene.json")
    reference = load_litematic(next(folder.glob("*.litematic")))
    axis, side = args.axis, args.side

    all_pos: list[float] = []
    all_neg: list[float] = []
    verdict_totals: Counter = Counter()
    for entry in calibration["estimates"]:
        frame_id = entry["frame"]
        path = folder / frame_id
        if not path.exists():
            continue
        size = tuple(calibration["inputs"][frame_id]["size"])
        camera = cameras[frame_id]
        grey = enhance(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        axes = axis_directions(entry["yaw_degrees_mod90"], entry["pitch_degrees"],
                               entry.get("roll_degrees", 0.0))
        focal = entry["focal_pixels"]

        annotation = simulate_annotation(reference, scene, camera, axis, size)
        if annotation is None or annotation.get("side") != side:
            print(f"{frame_id}: no side {side} annotation, skipped")
            continue
        centre_pp = entry.get("principal_point")
        principal_point = tuple(centre_pp) if centre_pp else (size[0] / 2.0, size[1] / 2.0)
        constraint = scale_from_known_length(
            annotation["points"][0], annotation["points"][1],
            float(annotation["blocks"]), axis, axes, focal, size,
            principal_point=centre_pp)
        if constraint.distance is None:
            print(f"{frame_id}: constraint unresolved, skipped")
            continue
        distance = constraint.distance

        # Distance is already known (assisted), so a failed automatic period
        # detection should not discard the frame - see supported.py's fix for
        # plane_from_evidence. measure_plane still returns a usable sampling
        # even when it found no period.
        found, samples, span = measure_plane(grey, axis, axes, focal, extent=1.0,
                                             side=side)
        patch = rectify_plane(grey, axis, axes, focal, extent=span, samples=samples,
                              side=side)
        if patch is None:
            print(f"{frame_id}: rectification failed, skipped")
            continue
        inside = rectify_plane(np.ones(grey.shape, np.float32), axis, axes, focal,
                               extent=span, samples=samples, side=side)
        coverage = np.asarray(inside, float)

        per_pixel = 2.0 * span / samples
        block_px = (1.0 / distance) / per_pixel
        if block_px < 2.0:
            print(f"{frame_id}: block period too small, skipped")
            continue

        cells_down = int(patch.shape[0] // block_px)
        cells_across = int(patch.shape[1] // block_px)

        if args.cue == "phase":
            across, down = _edge_profiles(patch)
            global_phase = (
                phase_at_period(across.mean(axis=0), np.arange(across.shape[1]), block_px),
                phase_at_period(down.mean(axis=1), np.arange(down.shape[0]), block_px))
            cells = support_cells(patch, coverage, block_px, global_phase,
                                  local_window_periods=args.local_window_periods)
            verdict_by_cell = {c.cell: c.verdict for c in cells}
            cue_score = {c.cell: c.score for c in cells if c.verdict in JUDGED_VERDICTS}
        else:
            # Photometric alternative: a cell's mean brightness against the
            # median of its immediate neighbours. A depth discontinuity - a
            # doorway onto a different, differently-lit room, or a window onto
            # the sky - should be a local outlier; a genuine continuation of
            # the same material should not. This cue has no notion of
            # weak/foreground yet, so every cell with enough neighbours to
            # compare against counts as judged.
            cell_mean = np.full((cells_down, cells_across), np.nan)
            for row in range(cells_down):
                for column in range(cells_across):
                    v0, v1 = int(row * block_px), int((row + 1) * block_px)
                    u0, u1 = int(column * block_px), int((column + 1) * block_px)
                    if coverage[v0:v1, u0:u1].mean() < 0.6:
                        continue
                    cell_mean[row, column] = patch[v0:v1, u0:u1].mean()
            verdict_by_cell = {}
            cue_score = {}
            for row in range(cells_down):
                for column in range(cells_across):
                    if np.isnan(cell_mean[row, column]):
                        continue
                    r0, r1 = max(0, row - 2), min(cells_down, row + 3)
                    c0, c1 = max(0, column - 2), min(cells_across, column + 3)
                    neighbourhood = cell_mean[r0:r1, c0:c1]
                    neighbours = neighbourhood[~np.isnan(neighbourhood)]
                    neighbours = neighbours[neighbours != cell_mean[row, column]]
                    if neighbours.size < 3:
                        continue
                    median = np.median(neighbours)
                    spread = np.median(np.abs(neighbours - median)) + 1e-6
                    z = abs(cell_mean[row, column] - median) / spread
                    cue_score[(row, column)] = 1.0 / (1.0 + z)
                    verdict_by_cell[(row, column)] = "judged"

        verdict_totals.update(verdict_by_cell.values())

        # Ground truth: is there a genuine, camera-reachable reference block
        # face at the world location this cell's centre maps to? Camera pose
        # is read here only, to evaluate - never to build the cue's score.
        reachable = visible_cells(reference.blocks, scene, {frame_id: camera},
                                  {frame_id: size})
        others = [i for i in range(3) if i != axis]
        by_lateral: dict[tuple[int, int], list[float]] = {}
        for coordinate in reachable:
            w = scene.to_world(coordinate)
            by_lateral.setdefault((w[others[0]], w[others[1]]), []).append(w[axis])

        eye = camera_eye(camera)
        plane_coord = eye[axis] + side * distance

        n_judged = n_truth_pos = 0
        for (row, column), score in cue_score.items():
            # Cell indices are in units of the block period; the mapping
            # function wants rectified-patch *pixel* coordinates.
            u, v = rectified_pixel_to_plane_coordinates(
                ((column + 0.5) * block_px, (row + 0.5) * block_px),
                extent=span, samples=samples)
            camera_point = plane_coordinates_to_camera_point(axis, axes, side, distance, u, v)
            world_point = camera_point_to_world(camera_point, camera)

            lateral_key = (int(np.floor(world_point[others[0]])),
                          int(np.floor(world_point[others[1]])))
            is_truth = any(
                abs((axis_value + offset) - plane_coord) < 0.51
                for axis_value in by_lateral.get(lateral_key, [])
                for offset in (0.0, 1.0))

            n_judged += 1
            (all_pos if is_truth else all_neg).append(score)
            n_truth_pos += is_truth

        frame_verdicts = Counter(verdict_by_cell.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(frame_verdicts.items()))
        print(f"{frame_id}: judged={n_judged} truth_positive={n_truth_pos} "
              f"({100*n_truth_pos/max(n_judged,1):.1f}%) [{args.cue} cue] [{breakdown}]")

    print(f"\nverdict totals across all frames: {dict(verdict_totals)}")
    pos, neg = np.array(all_pos), np.array(all_neg)
    print(f"TOTAL ({args.cue}): {len(pos)} truth-positive, {len(neg)} truth-negative "
          f"judged cells (weak_evidence/outside_image excluded)")
    if len(pos) == 0 or len(neg) == 0:
        print("  not enough of both classes to measure separation")
        return 0
    print(f"  positive: mean={pos.mean():.3f} p50={np.percentile(pos,50):.3f}")
    print(f"  negative: mean={neg.mean():.3f} p50={np.percentile(neg,50):.3f}")
    auc = _auc_with_ties(pos, neg)
    print(f"  AUC (0.5 = no separation, 1.0 = perfect): {auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
