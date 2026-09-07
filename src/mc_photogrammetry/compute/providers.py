from __future__ import annotations

from dataclasses import dataclass

from mc_photogrammetry.pipeline.contracts import ComputeDevice


@dataclass(slots=True)
class LocalComputeProvider:
    """Discover optional DirectML without making it a hard dependency."""

    def available_devices(self) -> list[ComputeDevice]:
        devices: list[ComputeDevice] = []
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
        return devices[0]
