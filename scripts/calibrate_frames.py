"""Estimate cameras from images alone, and optionally score them.

The M3 and M4 results were produced by harnesses that lived in a scratchpad, so
the numbers in those reports could be read but not re-run. This is the
maintained entry point: it records the settings it used and a SHA-256 of every
input it touched, so a result can be tied to the exact bytes that produced it.

Estimation never reads a manifest. Scoring does, and says so - the manifest is
evaluation scaffolding, and in production nobody has the F3 overlay that
produced it. Passing ``--score`` writes the comparison into a clearly separate
part of the artifact; leaving it off produces an estimate that could have been
made from a stranger's screenshots.

Examples
--------
Estimate every frame of a fixture, image-only::

    python scripts/calibrate_frames.py "templates 2" --out report.json

Estimate and score against the manifest, for development only::

    python scripts/calibrate_frames.py "templates 2" --score --out report.json

Score a frozen subset by name, as the milestone requires for Template 3::

    python scripts/calibrate_frames.py "templates 3" --subset t3-sparse-8 --score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.reconstruction import calibration as C  # noqa: E402
from mc_photogrammetry.reconstruction.calibration import (  # noqa: E402
    estimate_session,
    to_camera_solution,
)

SUBSETS = ROOT / "docs" / "reports" / "M3-frozen-subsets.json"


def plain(value):
    """Convert numpy scalars so the artifact is ordinary JSON.

    Several fields arrive as numpy types from the fit, and an artifact that
    cannot be written is worse than one that loses a type distinction nothing
    downstream depends on.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def digest(path: Path) -> str:
    """SHA-256 of one file, so a number can be tied to the bytes behind it."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_revision() -> dict:
    """The commit this ran at, and whether the tree was dirty."""
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return ""

    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def frames_for(folder: Path, subset: str | None) -> list[Path]:
    if subset is None:
        return sorted(folder.glob("*.png"))
    if not SUBSETS.exists():
        raise SystemExit(f"no frozen subsets at {SUBSETS}")
    frozen = json.loads(SUBSETS.read_text(encoding="utf-8"))
    names = frozen["subsets"].get(subset)
    if names is None:
        raise SystemExit(f"unknown subset {subset!r}; "
                         f"have {sorted(frozen['subsets'])}")
    return [folder / name for name in names]


def fold(degrees: float) -> float:
    """Signed difference on (-45, 45]: one view cannot name the horizontal axes."""
    return (degrees + 45.0) % 90.0 - 45.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture", help="folder holding the screenshots")
    parser.add_argument("--subset", help="a named subset from the frozen list")
    parser.add_argument("--out", type=Path, help="where to write the JSON artifact")
    parser.add_argument("--score", action="store_true",
                        help="compare against cameras.json; evaluation only, "
                             "never part of estimation")
    parser.add_argument("--per-view", action="store_true", default=True,
                        help="include a row per frame (default on)")
    args = parser.parse_args(argv)

    folder = ROOT / args.fixture
    if not folder.is_dir():
        raise SystemExit(f"no such fixture: {folder}")
    paths = [p for p in frames_for(folder, args.subset) if p.exists()]
    if not paths:
        raise SystemExit("no frames found")

    import cv2

    greys, inputs = {}, {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        greys[path.name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inputs[path.name] = {"sha256": digest(path),
                             "size": [int(image.shape[1]), int(image.shape[0])]}
    if not greys:
        raise SystemExit("no readable frames")

    started = time.time()
    results = estimate_session(greys)
    elapsed = time.time() - started

    settings = {
        "minimum_length_px": C.MINIMUM_LENGTH,
        "inlier_px": C.INLIER_PIXELS,
        "detection_scales": list(C.DETECTION_SCALES),
        "coarse_segments": C.COARSE_SEGMENTS,
        "gamma": C.GAMMA,
        "clahe_clip": C.CLAHE_CLIP,
        "clahe_grid": C.CLAHE_GRID,
        "refine_margin": C.REFINE_MARGIN,
        "roll_limit_deg": C.ROLL_LIMIT,
        "principal_limit_fraction": C.PRINCIPAL_LIMIT,
        "weight_by_contrast": False,
    }

    rows = []
    for frame_id, result in sorted(results.items()):
        row = {
            "frame": frame_id,
            "focal_pixels": result.focal_pixels,
            "fov_degrees": result.fov_degrees,
            "yaw_degrees_mod90": result.yaw_degrees,
            "pitch_degrees": result.pitch_degrees,
            "roll_degrees": result.roll_degrees,
            "principal_point": (list(result.principal_point)
                                if result.principal_point else None),
            "segments": result.segments,
            "inliers": result.inliers,
            "inlier_fraction": result.inlier_fraction,
            "median_residual_px": result.median_residual,
            "notes": list(result.notes),
        }
        rows.append(row)

    payload = {
        "fixture": args.fixture,
        "subset": args.subset,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "opencv": cv2.__version__},
        "settings": settings,
        "inputs": inputs,
        "seconds": elapsed,
        "frames_estimated": len(rows),
        "estimates": rows if args.per_view else [],
        "unresolved": list(to_camera_solution(results).unresolved),
        "scored": None,
    }

    print(f"{args.fixture}"
          f"{' / ' + args.subset if args.subset else ''}: "
          f"{len(rows)}/{len(greys)} frames estimated in {elapsed:.0f}s")

    if args.score:
        from mc_photogrammetry.reconstruction.camera import load_camera_manifest

        manifest = folder / "cameras.json"
        cameras = load_camera_manifest(manifest)
        comparisons, passed = [], 0
        for row in rows:
            truth = cameras.get(row["frame"])
            if truth is None:
                continue
            height = inputs[row["frame"]]["size"][1]
            true_focal = (height / 2.0) / np.tan(np.radians(truth.fov_degrees) / 2.0)
            pitch = row["pitch_degrees"] - truth.pitch_degrees
            yaw = fold(row["yaw_degrees_mod90"] - truth.yaw_degrees)
            focal = (row["focal_pixels"] - true_focal) / true_focal
            ok = abs(pitch) <= 0.5 and abs(yaw) <= 0.5 and abs(focal) <= 0.02
            passed += ok
            comparisons.append({"frame": row["frame"], "pitch_error": pitch,
                                "yaw_error_mod90": yaw, "focal_error": focal,
                                "within_gate": bool(ok)})
        payload["scored"] = {
            "note": "evaluation only; the manifest is scaffolding and is never "
                    "read during estimation",
            "manifest_sha256": digest(manifest),
            "gate": {"orientation_deg": 0.5, "focal_fraction": 0.02},
            "passed": passed,
            "compared": len(comparisons),
            "per_view": comparisons,
        }
        if comparisons:
            print(f"  scored {passed}/{len(comparisons)} within the gate; "
                  f"median |pitch| "
                  f"{np.median([abs(c['pitch_error']) for c in comparisons]):.3f} deg, "
                  f"median |yaw| "
                  f"{np.median([abs(c['yaw_error_mod90']) for c in comparisons]):.3f} deg")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=plain),
                            encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
