"""Probe RenderResult upload/readback parity on the selected wgpu adapter."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from mc_photogrammetry.compute import WgpuRenderBufferContext, pack_render_result
from mc_photogrammetry.pipeline.contracts import RenderResult


def main() -> None:
    height, width = 37, 29
    rng = np.random.default_rng(20260909)
    ownership = rng.integers(-1, 48, size=(height, width), dtype=np.int64)
    result = RenderResult(
        colour=rng.random((height, width, 3), dtype=np.float64),
        depth=rng.random((height, width), dtype=np.float64),
        ownership=ownership,
        uv=rng.random((height, width, 2), dtype=np.float64),
        valid_pixels=(ownership % 3 != 0),
    )
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
    uv_error = float(np.max(np.abs(roundtrip.uv - packed.uv)))
    np.testing.assert_array_equal(roundtrip.ownership, packed.ownership)
    np.testing.assert_array_equal(roundtrip.valid_pixels, packed.valid_pixels.astype(bool))
    assert colour_error == 0.0 and depth_error == 0.0 and uv_error == 0.0
    print(f"device={context.device.provider} name={context.device.name}")
    print(f"shape={layout.shape} has_uv={layout.has_uv} bytes={layout.byte_sizes}")
    print(f"upload_seconds={uploaded:.6f} readback_seconds={readback:.6f}")
    print(f"max_abs_error colour={colour_error:.3g} depth={depth_error:.3g} uv={uv_error:.3g}")


if __name__ == "__main__":
    main()
