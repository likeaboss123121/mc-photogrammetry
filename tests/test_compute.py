from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.compute import (
    LocalComputeProvider,
    WgpuComputeProvider,
    rgb_l2_cpu,
    score_batch,
)
from mc_photogrammetry.pipeline import ComputeDevice


def _adapter(name, backend, adapter_type, vendor_id=0x1002):
    info = SimpleNamespace(
        device=name, backend_type=backend, adapter_type=adapter_type,
        vendor="AMD proprietary driver", vendor_id=vendor_id, device_id=1,
    )
    return SimpleNamespace(
        info=info, features={"timestamp-query", "float32-filterable"},
        request_device_sync=lambda required_features: {"features": required_features},
    )


def test_wgpu_provider_prefers_discrete_amd_d3d12_and_deduplicates(monkeypatch):
    adapters = [
        _adapter("AMD Radeon RX 9070 XT", "Vulkan", "DiscreteGPU"),
        _adapter("AMD Radeon RX 9070 XT", "D3D12", "DiscreteGPU"),
        _adapter("AMD Radeon RX 9070 XT", "D3D12", "DiscreteGPU"),
        _adapter("AMD Radeon(TM) Graphics", "D3D12", "IntegratedGPU", vendor_id=0x1002),
    ]
    fake_wgpu = SimpleNamespace(
        gpu=SimpleNamespace(enumerate_adapters_sync=lambda: adapters),
    )
    monkeypatch.setitem(__import__("sys").modules, "wgpu", fake_wgpu)

    provider = WgpuComputeProvider()
    found = provider.available_adapters()

    assert len(found) == 3
    selected = provider.select()
    assert selected.provider == "wgpu:d3d12"
    assert provider.metadata(selected).adapter_type == "DiscreteGPU"
    assert provider.request_device(selected) == {"features": []}


def test_wgpu_provider_is_optional_when_the_module_is_absent(monkeypatch):
    monkeypatch.setattr(WgpuComputeProvider, "_module", staticmethod(lambda: None))

    provider = WgpuComputeProvider()

    assert provider.available_adapters() == []
    assert provider.available_devices() == []


def test_local_provider_keeps_a_cpu_fallback(monkeypatch):
    monkeypatch.setattr(WgpuComputeProvider, "available_devices", lambda self: [])

    devices = LocalComputeProvider().available_devices()

    assert devices[-1].provider == "cpu"
    assert devices[-1].supports_gpu is False


def test_local_provider_prefers_the_discrete_d3d12_wgpu_path(monkeypatch):
    monkeypatch.setattr(
        WgpuComputeProvider, "available_devices",
        lambda self: [
            ComputeDevice("wgpu:vulkan", "AMD", True),
            ComputeDevice("wgpu:d3d12", "AMD", True),
        ],
    )

    assert LocalComputeProvider().select().provider == "wgpu:d3d12"


def test_cpu_batch_score_is_deterministic_and_reports_cpu_provenance():
    observed = np.asarray([0, 10, 20, 30], dtype=np.float32)
    candidates = np.asarray([[0, 10, 20, 30], [4, 14, 24, 34]], dtype=np.float32)

    result = score_batch(observed, candidates)

    np.testing.assert_allclose(result.scores, [0.0, 4.0])
    np.testing.assert_allclose(result.scores, rgb_l2_cpu(observed, candidates))
    assert result.device.provider == "cpu"
    assert result.elapsed_seconds >= 0.0


def test_batch_score_rejects_mismatched_features():
    with pytest.raises(ValueError, match="shape"):
        rgb_l2_cpu(np.zeros(3), np.zeros((2, 4)))
