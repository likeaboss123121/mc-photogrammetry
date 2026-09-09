"""Render a real fixture view with the shared CPU renderer, for GPU-buffer parity.

Luna's M6 render-buffer seam (`compute.render_buffers`) has only been checked
against synthetic random noise at a 37x29 toy resolution
(`probe_wgpu_render_buffers.py`). This is the other half of "Sonnet
integrates this seam at the renderer boundary": produce a real `RenderResult`
from the actual shared renderer (`export.attribution.render_result`) on a
real fixture, at production scale, and save its arrays so the GPU side of the
parity check (`verify_render_buffer_parity.py`, run in `.venv-gpu`, which has
no OpenCV/nbtlib) can load them without either environment needing the
other's dependencies.

Runs in the regular `.venv` - the one with the full rendering stack.

    python scripts/export_render_for_gpu_parity.py --out <file>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fixture", default="templates 2")
    parser.add_argument("--frame", default="2026-09-02_13.22.14.png")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import os
    import numpy as np

    from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
    from mc_photogrammetry.export.attribution import (
        face_colours, face_shapes, face_occluding_slots, face_model_faces, render_result)
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.reconstruction.camera import load_camera_manifest
    from mc_photogrammetry.reconstruction.scene import load_scene_transform

    fixture = ROOT / args.fixture
    reference = load_litematic(next(fixture.glob("*.litematic")))
    scene = load_scene_transform(fixture / "scene.json")
    cameras = load_camera_manifest(fixture / "cameras.json")
    camera = cameras[args.frame]

    provider = VanillaJarTextureProvider()
    provider.load_jar(Path(os.environ["APPDATA"]) / ".minecraft/versions/1.21.5/1.21.5.jar")
    palette = provider.complete_palette()

    colours = face_colours(palette)
    shapes = face_shapes(palette)
    occluding_slots = face_occluding_slots(palette)
    model_faces = face_model_faces(palette)

    # A real-scale render, not a toy resolution - this is the whole point of
    # extending Luna's synthetic-only probe.
    width, height = 640, 342
    result, index = render_result(
        reference.blocks, scene, camera, width, height, colours,
        shapes=shapes, scale=1.0, ambient=True,
        occluding_slots=occluding_slots, model_faces=model_faces)

    valid_fraction = float(np.mean(result.valid_pixels)) if result.valid_pixels is not None else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, colour=result.colour, depth=result.depth,
            ownership=result.ownership, uv=result.uv,
            valid_pixels=result.valid_pixels)
    meta = {
        "fixture": args.fixture, "frame": args.frame, "width": width, "height": height,
        "blocks_in_reference": len(reference.blocks), "faces_in_index": len(index),
        "valid_pixel_fraction": valid_fraction,
        "colour_shape": list(result.colour.shape), "depth_shape": list(result.depth.shape),
        "ownership_range": [int(result.ownership.min()), int(result.ownership.max())],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rendered {width}x{height}, {len(index)} faces, "
         f"{valid_fraction:.1%} valid pixels, ownership range {meta['ownership_range']}")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
