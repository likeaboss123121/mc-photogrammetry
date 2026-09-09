"""Compute backend discovery."""

from .providers import LocalComputeProvider
from .cache import FramePyramid, FramePyramidCache, build_frame_pyramid
from .scoring import ScoreBatch, WgpuScoringContext, rgb_l2_cpu, rgb_l2_wgpu, score_batch
from .wgpu_provider import WgpuAdapterInfo, WgpuComputeProvider

__all__ = [
    "LocalComputeProvider", "ScoreBatch", "WgpuAdapterInfo", "WgpuComputeProvider",
    "WgpuScoringContext",
    "FramePyramid", "FramePyramidCache", "build_frame_pyramid",
    "rgb_l2_cpu", "rgb_l2_wgpu", "score_batch",
]
