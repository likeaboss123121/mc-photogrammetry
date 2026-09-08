"""Measure how a frozen evaluator mask changes the reference denominator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_templates import _load_usable_masks
from mc_photogrammetry.export.compare import load_litematic
from mc_photogrammetry.export.visibility import visible_cells
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.scene import load_scene_transform


def paths_for(frames, directory: Path) -> dict[str, Path]:
    return {name: directory / f"{Path(name).stem}.json" for name in frames}


def summary(reference, scene, cameras, sizes, frames, paths) -> dict[str, int]:
    visible = visible_cells(
        reference, scene, cameras, sizes, usable_masks=_load_usable_masks(paths, frames),
    )
    return {"visible_reference_cells": len(visible), "visible_reference_pixels": sum(visible.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    args = parser.parse_args()
    frames = {path.name: load_image(path) for path in sorted(args.images.glob("*.png"))}
    sizes = {name: (image.shape[1], image.shape[0]) for name, image in frames.items()}
    reference = load_litematic(args.reference).blocks
    cameras, scene = load_camera_manifest(args.cameras), load_scene_transform(args.scene)
    result = {
        "all_decoded_pixels": summary(reference, scene, cameras, sizes, frames, {}),
        "historical_v1": summary(reference, scene, cameras, sizes, frames, {name: args.v1 for name in frames}),
        "reviewed_v2": summary(reference, scene, cameras, sizes, frames, paths_for(frames, args.v2)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
