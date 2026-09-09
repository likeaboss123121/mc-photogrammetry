# M6 — render-buffer seam integrated, verified on real renderer output

**Claude Sonnet 5, 2026-09-09.** Luna's M6 report drew a clear boundary:
production call-site wiring (CLI/desktop pipeline, overlays, review controls)
depends on M2/M4/M5 closing and stays untouched, but named one piece as ready
now - *"Sonnet can integrate this seam at the renderer boundary before any GPU
rasterisation is enabled."* This is that piece.

No library code changed - `src/` and the 445-test suite are untouched, exactly
as intended for something explicitly scoped as safe before upstream gates
close. Two new scripts.

## What was missing

`probe_wgpu_render_buffers.py` (Luna's) verifies the `pack_render_result` /
`WgpuRenderBufferContext` contract against synthetic random noise at a 37x29
toy resolution. That's a real check of the contract's *shape*, but it can't
catch anything specific to real renderer output - actual depth ranges, real
UV distributions, the ownership/validity patterns an actual scene produces,
or production scale.

## The integration

Two scripts, split across the two environments this project already
maintains for exactly this reason (the regular `.venv` has the rendering
stack and no `wgpu` - it needs Python 3.11+; `.venv-gpu` has `wgpu` and
nothing else):

- **`export_render_for_gpu_parity.py`** (regular `.venv`) calls the real
  shared renderer, `export.attribution.render_result`, on an actual Template
  2 view - real reference blocks, real scene, real camera, the real palette
  from the vanilla jar - at production scale (640x342, not 37x29), and saves
  the resulting `RenderResult` arrays.
- **`verify_render_buffer_parity.py`** (`.venv-gpu`) loads them, runs the
  exact same `pack_render_result` / `WgpuRenderBufferContext` upload/readback
  Luna's probe uses, and asserts parity.

## Result

```
source: templates 2/2026-09-02_13.22.14.png, 640x342, 767 faces, 100.0% valid pixels
device=wgpu:d3d12 name=AMD Radeon RX 9070 XT
shape=(342, 640) has_uv=True bytes=6347520
upload_seconds=0.030268 readback_seconds=0.051055
max_abs_error colour=0 depth=0 uv=0
ownership_exact=True valid_pixels_exact=True
PARITY OK on real renderer output (not synthetic noise)
```

Exact parity - zero error on colour, depth, and UV, exact equality on
ownership and validity - on a real render roughly 200x the pixel count of
the synthetic probe, with real geometry (767 distinct faces) rather than
uniform random noise.

## What this does and does not establish

**Does:** confirm the render-buffer contract holds under real data
characteristics the synthetic probe couldn't exercise - real depth
continuity, real UV layout, a realistic ownership id range, production
resolution.

**Does not:** enable GPU rasterization, change any CPU default, or touch the
production call-site wiring Luna's boundary correctly keeps blocked on
M2/M4/M5. Both scripts are read-only with respect to the renderer and the
reconstruction pipeline - they call the existing renderer once and pack its
already-computed output; nothing about how or when the renderer normally runs
changed.

## Why two scripts instead of one pytest test

No single environment currently has both the rendering stack and `wgpu`
together - the regular `.venv` is Python 3.10.11 and `wgpu` needs 3.11+, so
it can't be installed there; `.venv-gpu` was deliberately scoped narrowly by
Luna and doesn't have OpenCV/nbtlib. Adding either dependency to the other's
environment is an environment-ownership decision, not something to do
unilaterally while integrating a seam. The two-script, save-then-load design
follows the same split-environment pattern this project already uses for
CPU-SfM (`.venv-sfm`), rather than forcing a combined test that isn't
possible to run as-is.
