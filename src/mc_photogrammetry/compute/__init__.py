"""Compute backend discovery."""

from .providers import LocalComputeProvider
from .scoring import ScoreBatch, rgb_l2_cpu, rgb_l2_wgpu, score_batch
from .wgpu_provider import WgpuAdapterInfo, WgpuComputeProvider

__all__ = [
    "LocalComputeProvider", "ScoreBatch", "WgpuAdapterInfo", "WgpuComputeProvider",
    "rgb_l2_cpu", "rgb_l2_wgpu", "score_batch",
]
