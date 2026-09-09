"""Report optional wgpu adapters and validate native device creation."""
from __future__ import annotations

import json

from mc_photogrammetry.compute import WgpuComputeProvider


def main() -> int:
    provider = WgpuComputeProvider(preferred_backend="d3d12")
    adapters = provider.available_adapters()
    payload: dict[str, object] = {
        "available": bool(adapters),
        "adapters": [
            {
                "name": info.name,
                "backend": info.backend,
                "adapter_type": info.adapter_type,
                "vendor": info.vendor,
                "vendor_id": info.vendor_id,
                "device_id": info.device_id,
                "supports_gpu": info.supports_gpu,
                "features": list(info.features),
            }
            for info in adapters
        ],
    }
    if adapters:
        selected = provider.select()
        metadata = provider.metadata(selected)
        provider.request_device(selected)
        payload["selected"] = {
            "provider": selected.provider,
            "name": selected.name,
            "backend": metadata.backend,
            "adapter_type": metadata.adapter_type,
            "device_created": True,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
