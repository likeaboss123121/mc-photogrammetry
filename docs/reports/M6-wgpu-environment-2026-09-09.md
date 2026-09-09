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

The repository regression suite remains green after this groundwork: **428 tests
passed in 174.56 seconds**. The provider-specific tests also pass in the regular
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
