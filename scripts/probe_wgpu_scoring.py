"""Run the first CPU/wgpu batched-scoring parity probe."""
from __future__ import annotations

import json

import numpy as np

from mc_photogrammetry.compute import rgb_l2_cpu, rgb_l2_wgpu


def main() -> int:
    observed = np.asarray([0, 10, 20, 30, 40, 50], dtype=np.float32)
    candidates = np.asarray([
        [0, 10, 20, 30, 40, 50],
        [6, 16, 26, 36, 46, 56],
        [10, 0, 20, 0, 10, 0],
    ], dtype=np.float32)
    cpu = rgb_l2_cpu(observed, candidates)
    gpu = rgb_l2_wgpu(observed, candidates)
    payload = {
        "device": {
            "provider": gpu.device.provider,
            "name": gpu.device.name,
            "supports_gpu": gpu.device.supports_gpu,
        },
        "cpu_scores": cpu.tolist(),
        "gpu_scores": gpu.scores.tolist(),
        "max_absolute_error": float(np.max(np.abs(cpu - gpu.scores))),
        "gpu_elapsed_seconds": gpu.elapsed_seconds,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
