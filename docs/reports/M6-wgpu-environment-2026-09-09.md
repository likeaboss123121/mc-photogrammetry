# M6 wgpu environment groundwork

GPT-5.6 Luna, 2026-09-09.

An isolated `.venv-gpu` was created with Python 3.12 and `wgpu==0.32.0`.
The existing Python 3.10 `.venv` and the separate `.venv-sfm` environment were
not changed. The environment is ignored by Git.

`scripts/probe_wgpu.py` enumerated seven adapters, including the discrete AMD
Radeon RX 9070 XT through Vulkan and D3D12, integrated AMD adapters, a software
CPU adapter, and an OpenGL adapter. Deterministic selection chose:

```text
provider:     wgpu:d3d12
device:       AMD Radeon RX 9070 XT
adapter type: DiscreteGPU
backend:      D3D12
vendor id:    4098 (AMD)
```

The probe successfully created a native `GPUDevice` from the selected adapter.
No production render or scoring pipeline uses it yet. CPU rendering remains the
correctness path until an independently compared GPU implementation exists.

The provider seam is in `src/mc_photogrammetry/compute/wgpu_provider.py` and
exports serialisable adapter metadata, stable device selection, optional import
behaviour, and native device creation. `LocalComputeProvider` now discovers these
wgpu devices before its DirectML/CPU fallback.

The repository regression suite was green after the provider groundwork with
**428 tests passed in 174.56 seconds**. After the scoring-context refactor it
remains green at **434 tests passed in 142.94 seconds**. The provider-specific tests also pass in the regular
Python 3.10 environment where wgpu is absent, preserving the optional-dependency
CPU path.

The first batched-scoring parity probe (`scripts/probe_wgpu_scoring.py`) ran on
the selected RX 9070 XT D3D12 device. Three fixed candidate vectors matched the
CPU reference to a maximum absolute error of `1.91e-6` (GPU dispatch elapsed
`0.174 s` in the initial latest run, including setup and readback). The reusable
context now keeps the native device and pipeline: the latest two-call probe
measured `1.048 s` for context creation, `0.092 s` for the first dispatch, and
`0.00061 s` for the second dispatch. These are smoke-test timings, not a production performance
claim; image rendering and texture batches still use the CPU path.

The M6 cache groundwork also adds `FramePyramid`/`FramePyramidCache`: immutable
half-resolution levels and an LRU bounded by total bytes, with conservative
validation for invalid limits and oversized entries. This gives future GPU
uploads a reusable, memory-bounded source without loading every full-resolution
frame into an accelerator at once.

## Renderer buffer parity seam

`compute.render_buffers` now defines the wire contract for the existing
`RenderResult` without changing the renderer. Colour, depth, and optional UV are
contiguous `float32`; ownership is checked and packed as `int32`; validity is
packed as `uint8` and derives from ownership when the source omits it. Shapes are
bounded to one `(height, width)` output and GPU allocations are padded to four
bytes, including the validity mask. `WgpuRenderBufferContext` reuses allocations
for a stable shape and provides a staging-buffer readback for independent parity
checks; the default application path is still CPU.

The isolated GPU probe (`scripts/probe_wgpu_render_buffers.py`) completed on the
selected AMD Radeon RX 9070 XT D3D12 adapter using a 37x29 result with UV and
explicit validity. It reported 0 maximum absolute error for colour, depth, and
UV, exact ownership and validity equality, `0.032116 s` upload, and `0.050615 s`
readback. These timings cover a small transfer smoke test rather than a renderer
performance claim. Sonnet can integrate this seam at the renderer boundary before
any GPU rasterisation is enabled.

The current repository suite contains 441 collected tests; the full `.venv`
run completed successfully after this seam was added.

`BoundedArtifactCache` extends the frame cache contract to arbitrary feature,
geometry, descriptor, and accepted-buffer artifacts. It accounts for each
entry's byte size, updates LRU order on access, evicts whole entries when the
configured budget is exceeded, and rejects values without a measurable size
unless the caller supplies one. Focused cache and buffer tests pass.

## Closure boundary for review

The Luna-owned M6 acceleration bundle is complete: isolated adapter discovery,
reusable batched scoring, bounded frame and artifact caches, and renderer-buffer
transfer parity are implemented and regression-tested. Production call-site
wiring (one CLI/desktop pipeline, overlays, review controls, and resume-aware
application state) remains the Sonnet integration boundary and depends on the
M2/M4/M5 production contracts. No CPU default was changed while those upstream
quality gates remain open.
