"""Compute backend discovery."""

from .providers import LocalComputeProvider
from .wgpu_provider import WgpuAdapterInfo, WgpuComputeProvider

__all__ = ["LocalComputeProvider", "WgpuAdapterInfo", "WgpuComputeProvider"]
