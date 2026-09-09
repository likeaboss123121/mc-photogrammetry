"""Optional wgpu adapter discovery for the M6 acceleration seam.

The provider deliberately owns discovery and device selection only.  Rendering
and scoring remain on the CPU until a GPU kernel has an independently checked
CPU equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mc_photogrammetry.pipeline.contracts import ComputeDevice


def _field(value: Any, name: str, default: Any = "") -> Any:
    """Read a field from wgpu's versioned adapter-info object or mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True, slots=True)
class WgpuAdapterInfo:
    """Stable, serialisable metadata for one enumerated wgpu adapter."""

    name: str
    backend: str
    adapter_type: str
    vendor: str = ""
    vendor_id: int = 0
    device_id: int = 0
    features: tuple[str, ...] = ()

    @property
    def supports_gpu(self) -> bool:
        return self.adapter_type.lower() not in {"cpu", "software"}

    @property
    def provider(self) -> str:
        return f"wgpu:{self.backend.lower()}"

    def as_compute_device(self) -> ComputeDevice:
        return ComputeDevice(self.provider, self.name, self.supports_gpu)


class WgpuComputeProvider:
    """Discover and select optional wgpu adapters without a hard dependency."""

    def __init__(self, *, preferred_backend: str = "d3d12") -> None:
        self.preferred_backend = preferred_backend.lower()
        self._adapters: dict[ComputeDevice, Any] = {}
        self._metadata: dict[ComputeDevice, WgpuAdapterInfo] = {}

    @staticmethod
    def _module() -> Any | None:
        try:
            import wgpu
        except ImportError:
            return None
        return wgpu

    def available_adapters(self) -> list[WgpuAdapterInfo]:
        """Enumerate adapters and retain their native handles for device creation."""
        wgpu = self._module()
        self._adapters.clear()
        self._metadata.clear()
        if wgpu is None:
            return []
        gpu = wgpu.gpu
        enumerate_sync = getattr(gpu, "enumerate_adapters_sync", None)
        adapters = enumerate_sync() if enumerate_sync is not None else gpu.enumerate_adapters()
        result: list[WgpuAdapterInfo] = []
        seen: set[tuple[Any, ...]] = set()
        for adapter in adapters:
            raw = getattr(adapter, "info", {})
            name = str(_field(raw, "device", "Unknown adapter"))
            backend = str(_field(raw, "backend_type", "unknown"))
            adapter_type = str(_field(raw, "adapter_type", "Unknown"))
            vendor = str(_field(raw, "vendor", ""))
            vendor_id = int(_field(raw, "vendor_id", 0) or 0)
            device_id = int(_field(raw, "device_id", 0) or 0)
            key = (name, backend.lower(), adapter_type.lower(), vendor_id, device_id)
            if key in seen:
                continue
            seen.add(key)
            features = tuple(sorted(str(feature) for feature in getattr(adapter, "features", ())))
            info = WgpuAdapterInfo(
                name=name, backend=backend, adapter_type=adapter_type,
                vendor=vendor, vendor_id=vendor_id, device_id=device_id,
                features=features,
            )
            device = info.as_compute_device()
            self._adapters[device] = adapter
            self._metadata[device] = info
            result.append(info)
        return result

    def available_devices(self) -> list[ComputeDevice]:
        if not self._metadata:
            self.available_adapters()
        return list(self._metadata)

    @staticmethod
    def _score(info: WgpuAdapterInfo, preferred_backend: str) -> tuple[int, int, int]:
        amd = info.vendor_id == 0x1002 or "amd" in info.vendor.lower() or "radeon" in info.name.lower()
        discrete = info.adapter_type.lower() == "discretegpu"
        backend = info.backend.lower() == preferred_backend
        return (int(info.supports_gpu), int(discrete and amd), int(backend))

    def select(self, preferred: str | None = None) -> ComputeDevice:
        """Select a discrete AMD adapter, preferring the configured backend."""
        if not self._metadata:
            self.available_adapters()
        devices = list(self._metadata)
        if not devices:
            raise RuntimeError("wgpu is unavailable or exposes no adapters")
        if preferred:
            exact = [device for device in devices if device.provider == preferred]
            if exact:
                return exact[0]
        return max(
            devices,
            key=lambda device: self._score(self._metadata[device], self.preferred_backend),
        )

    def metadata(self, device: ComputeDevice | None = None) -> WgpuAdapterInfo:
        """Return metadata for the selected device."""
        if not self._metadata:
            self.available_adapters()
        selected = device or self.select()
        try:
            return self._metadata[selected]
        except KeyError as exc:
            raise ValueError("device was not enumerated by this provider") from exc

    def request_device(self, device: ComputeDevice | None = None) -> Any:
        """Create a native GPUDevice for a selected adapter.

        This is the only operation that touches the native device.  Kernel
        submission is intentionally left to later M6 work.
        """
        if not self._adapters:
            self.available_adapters()
        selected = device or self.select()
        try:
            adapter = self._adapters[selected]
        except KeyError as exc:
            raise ValueError("device was not enumerated by this provider") from exc
        request_sync = getattr(adapter, "request_device_sync", None)
        if request_sync is not None:
            return request_sync(required_features=[])
        return adapter.request_device(required_features=[])
