"""Verify GPU buffer parity against a real renderer output, not synthetic noise.

The other half of `export_render_for_gpu_parity.py`. Run in `.venv-gpu`
(has `wgpu`, not the rendering stack) against the `.npz` that script produces
in the regular `.venv` (has the rendering stack, not `wgpu`). Neither
environment needs the other's dependencies.

This is real production-scale renderer output - 640x342, hundreds of real
faces, real depth/UV/ownership values from an actual Template 2 view - not
`probe_wgpu_render_buffers.py`'s 37x29 random noise. Same contract
(`pack_render_result` / `WgpuRenderBufferContext`), same assertions, real
data.

    <regular venv> python scripts/export_render_for_gpu_parity.py --out render.npz
    <.venv-gpu>    python scripts/verify_render_buffer_parity.py --render render.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--render", type=Path, required=True)
    args = parser.parse_args(argv)

    import numpy as np

    from mc_photogrammetry.compute import WgpuRenderBufferContext, pack_render_result
    from mc_photogrammetry.pipeline.contracts import RenderResult

    data = np.load(args.render, allow_pickle=True)
    # np.savez stores a Python None as a 0-d object array; a real buffer never
    # has shape () (every renderer buffer is at least (height, width)).
    uv = data["uv"] if data["uv"].shape != () else None
    valid = data["valid_pixels"] if data["valid_pixels"].shape != () else None
    result = RenderResult(colour=data["colour"], depth=data["depth"],
                          ownership=data["ownership"], uv=uv, valid_pixels=valid)

    meta_path = args.render.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"source: {meta['fixture']}/{meta['frame']}, {meta['width']}x{meta['height']}, "
             f"{meta['faces_in_index']} faces, {meta['valid_pixel_fraction']:.1%} valid pixels")

    packed = pack_render_result(result)
    context = WgpuRenderBufferContext()
    started = perf_counter()
    layout = context.upload(result)
    uploaded = perf_counter() - started
    started = perf_counter()
    roundtrip = context.readback()
    readback = perf_counter() - started

    colour_error = float(np.max(np.abs(roundtrip.colour - packed.colour)))
    depth_error = float(np.max(np.abs(roundtrip.depth - packed.depth)))
    uv_error = (float(np.max(np.abs(roundtrip.uv - packed.uv))) if packed.uv is not None else None)
    ownership_matches = bool(np.array_equal(roundtrip.ownership, packed.ownership))
    valid_matches = bool(np.array_equal(roundtrip.valid_pixels, packed.valid_pixels.astype(bool)))

    print(f"device={context.device.provider} name={context.device.name}")
    print(f"shape={layout.shape} has_uv={layout.has_uv} bytes={sum(layout.byte_sizes.values())}")
    print(f"upload_seconds={uploaded:.6f} readback_seconds={readback:.6f}")
    print(f"max_abs_error colour={colour_error:.3g} depth={depth_error:.3g} "
         f"uv={'n/a' if uv_error is None else f'{uv_error:.3g}'}")
    print(f"ownership_exact={ownership_matches} valid_pixels_exact={valid_matches}")

    ok = (colour_error == 0.0 and depth_error == 0.0 and (uv_error in (None, 0.0))
         and ownership_matches and valid_matches)
    if not ok:
        print("PARITY FAILED on real renderer output")
        return 1
    print("PARITY OK on real renderer output (not synthetic noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
