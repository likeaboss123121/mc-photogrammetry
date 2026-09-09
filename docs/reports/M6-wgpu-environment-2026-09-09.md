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
No render or scoring kernel is enabled yet. CPU rendering remains the correctness
path until an independently compared GPU implementation exists.

The provider seam is in `src/mc_photogrammetry/compute/wgpu_provider.py` and
exports serialisable adapter metadata, stable device selection, optional import
behaviour, and native device creation. `LocalComputeProvider` now discovers these
wgpu devices before its DirectML/CPU fallback.
