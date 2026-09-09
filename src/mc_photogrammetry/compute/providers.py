from __future__ import annotations

from dataclasses import dataclass

from mc_photogrammetry.pipeline.contracts import ComputeDevice

from .wgpu_provider import WgpuComputeProvider


@dataclass(slots=True)
class LocalComputeProvider:
    """Discover optional wgpu/DirectML devices without hard dependencies."""

    def available_devices(self) -> list[ComputeDevice]:
        devices: list[ComputeDevice] = []
        devices.extend(WgpuComputeProvider().available_devices())
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None

        if ort is not None and "DmlExecutionProvider" in ort.get_available_providers():
            devices.append(ComputeDevice("directml", "DirectML GPU", supports_gpu=True))
        devices.append(ComputeDevice("cpu", "CPU", supports_gpu=False))
        return devices

    def select(self, preferred: str | None = None) -> ComputeDevice:
        devices = self.available_devices()
        if preferred:
            for device in devices:
                if device.provider == preferred:
                    return device
        # M6's default is the discrete AMD D3D12 path.  Wgpu's own selector
        # applies the same policy; preserve it when this composite provider is
        # asked to choose without an explicit preference.
        for device in devices:
            if device.provider == "wgpu:d3d12" and device.supports_gpu:
                return device
        return devices[0]
