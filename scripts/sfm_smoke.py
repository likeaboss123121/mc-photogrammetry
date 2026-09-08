"""Run a small CPU-only PyCOLMAP feature-to-mapping smoke test.

This utility is intentionally separate from the reconstruction pipeline.  It
verifies the isolated SfM environment and provides a repeatable handoff point
for the M3 multiview experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy
import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--image-name",
        action="append",
        default=[],
        help="Image filename to include; repeat up to the desired small set.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("single", "auto", "per-folder", "per-image"),
        default="single",
        help="COLMAP camera sharing mode (default: single for one screenshot session).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    names = args.image_name or sorted(
        path.name for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(names) < 2:
        raise SystemExit("at least two image names are required")
    missing = [name for name in names if not (image_dir / name).is_file()]
    if missing:
        raise SystemExit(f"image names are absent from {image_dir}: {', '.join(missing)}")

    if output_dir.exists():
        raise SystemExit(f"refusing to replace existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    database_path = output_dir / "database.db"
    mapping_path = output_dir / "sparse"
    camera_modes = {
        "single": pycolmap.CameraMode.SINGLE,
        "auto": pycolmap.CameraMode.AUTO,
        "per-folder": pycolmap.CameraMode.PER_FOLDER,
        "per-image": pycolmap.CameraMode.PER_IMAGE,
    }

    try:
        pycolmap.extract_features(
            database_path,
            image_dir,
            image_names=names,
            camera_mode=camera_modes[args.camera_mode],
            device=pycolmap.Device.cpu,
        )
        pycolmap.match_exhaustive(database_path, device=pycolmap.Device.cpu)
        reconstructions = pycolmap.incremental_mapping(
            database_path,
            image_dir,
            mapping_path,
        )
        database = pycolmap.Database.open(database_path)
        summary = {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pycolmap": pycolmap.__version__,
            "device": "cpu",
            "camera_mode": args.camera_mode,
            "image_dir": str(image_dir),
            "image_names": names,
            "database": str(database_path),
            "num_cameras": database.num_cameras(),
            "num_images": database.num_images(),
            "num_keypoints": database.num_keypoints(),
            "num_matches": database.num_matches(),
            "num_verified_pairs": database.num_verified_image_pairs(),
            "num_reconstructions": len(reconstructions),
            "registered_images": [reconstruction.num_reg_images() for reconstruction in reconstructions.values()],
            "points3d": [reconstruction.num_points3D() for reconstruction in reconstructions.values()],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
