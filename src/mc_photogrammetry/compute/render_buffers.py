"""Stable CPU/GPU buffer packing for renderer output.

The CPU renderer remains the source of truth.  This module gives later GPU
kernels a bounded, documented representation of :class:`RenderResult` and a
small upload/readback harness for checking that representation independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mc_photogrammetry.pipeline.contracts import ComputeDevice, RenderResult


def _aligned_size(size: int, alignment: int = 4) -> int:
    return max(alignment, (size + alignment - 1) // alignment * alignment)


@dataclass(frozen=True, slots=True)
class RenderBufferLayout:
    """Shape and canonical dtypes shared by CPU and GPU renderer buffers."""

    height: int
    width: int
    has_uv: bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def byte_sizes(self) -> dict[str, int]:
        pixels = self.height * self.width
        sizes = {
            "colour": pixels * 3 * np.dtype(np.float32).itemsize,
            "depth": pixels * np.dtype(np.float32).itemsize,
            "ownership": pixels * np.dtype(np.int32).itemsize,
            "valid_pixels": pixels * np.dtype(np.uint8).itemsize,
        }
        if self.has_uv:
            sizes["uv"] = pixels * 2 * np.dtype(np.float32).itemsize
        return sizes


@dataclass(frozen=True, slots=True)
class PackedRenderBuffers:
    """Contiguous, read-only arrays ready for transfer to a GPU buffer."""

    layout: RenderBufferLayout
    colour: np.ndarray
    depth: np.ndarray
    ownership: np.ndarray
    valid_pixels: np.ndarray
    uv: np.ndarray | None = None

    @property
    def nbytes(self) -> int:
        """Total unpadded bytes held by this CPU-side buffer snapshot."""
        arrays = (self.colour, self.depth, self.ownership, self.valid_pixels)
        if self.uv is not None:
            arrays += (self.uv,)
        return sum(int(array.nbytes) for array in arrays)


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def pack_render_result(result: RenderResult) -> PackedRenderBuffers:
    """Canonicalise a renderer result without changing its source arrays."""
    colour = np.asarray(result.colour)
    depth = np.asarray(result.depth)
    ownership = np.asarray(result.ownership)
    if colour.ndim != 3 or colour.shape[2] != 3:
        raise ValueError("colour must have shape (height, width, 3)")
    if depth.ndim != 2 or ownership.ndim != 2:
        raise ValueError("depth and ownership must have shape (height, width)")
    if depth.shape != colour.shape[:2] or ownership.shape != colour.shape[:2]:
        raise ValueError("all renderer buffers must have matching height and width")
    height, width = colour.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("renderer buffers must not be empty")
    if not np.issubdtype(ownership.dtype, np.integer):
        raise ValueError("ownership must use an integer dtype")
    info = np.iinfo(np.int32)
    if ownership.size and (ownership.min() < info.min or ownership.max() > info.max):
        raise ValueError("ownership values must fit in int32")

    uv: np.ndarray | None
    if result.uv is None:
        uv = None
    else:
        uv_array = np.asarray(result.uv)
        if uv_array.shape != (height, width, 2):
            raise ValueError("uv must have shape (height, width, 2)")
        uv = _readonly(uv_array.astype(np.float32, copy=False))

    if result.valid_pixels is None:
        valid = ownership >= 0
    else:
        valid = np.asarray(result.valid_pixels)
        if valid.shape != (height, width):
            raise ValueError("valid_pixels must have shape (height, width)")
        valid = valid.astype(bool, copy=False)

    return PackedRenderBuffers(
        layout=RenderBufferLayout(height, width, uv is not None),
        colour=_readonly(colour.astype(np.float32, copy=False)),
        depth=_readonly(depth.astype(np.float32, copy=False)),
        ownership=_readonly(ownership.astype(np.int32, copy=False)),
        valid_pixels=_readonly(valid.astype(np.uint8, copy=False)),
        uv=uv,
    )


def unpack_render_buffers(buffers: PackedRenderBuffers) -> RenderResult:
    """Return a normal ``RenderResult`` with the packed canonical values."""
    return RenderResult(
        colour=np.array(buffers.colour, copy=True),
        depth=np.array(buffers.depth, copy=True),
        ownership=np.array(buffers.ownership, copy=True),
        uv=None if buffers.uv is None else np.array(buffers.uv, copy=True),
        valid_pixels=np.asarray(buffers.valid_pixels, dtype=bool).copy(),
    )


class WgpuRenderBufferContext:
    """Reusable wgpu storage for one bounded renderer output shape.

    The context deliberately performs no rasterisation.  It uploads the CPU
    result, then reads the same bytes through a staging buffer so layout and
    transfer parity can be tested before a GPU renderer is enabled.
    """

    def __init__(
        self,
        *,
        provider: Any | None = None,
        device: ComputeDevice | None = None,
    ) -> None:
        from .wgpu_provider import WgpuComputeProvider

        provider = provider or WgpuComputeProvider()
        selected = device or provider.select()
        self.provider = provider
        self.device = selected
        self._native = provider.request_device(selected)
        self._buffers: dict[str, tuple[Any, int, int]] = {}
        self._packed: PackedRenderBuffers | None = None

    @staticmethod
    def _raw_arrays(packed: PackedRenderBuffers) -> dict[str, np.ndarray]:
        arrays = {
            "colour": packed.colour,
            "depth": packed.depth,
            "ownership": packed.ownership,
            "valid_pixels": packed.valid_pixels,
        }
        if packed.uv is not None:
            arrays["uv"] = packed.uv
        return arrays

    def _ensure_buffers(self, packed: PackedRenderBuffers) -> None:
        import wgpu

        raw_arrays = self._raw_arrays(packed)
        if set(self._buffers) == set(raw_arrays) and all(
            entry[1] == _aligned_size(array.nbytes) and entry[2] == array.nbytes
            for name, array in raw_arrays.items()
            for entry in [self._buffers[name]]
        ):
            return
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        self._buffers = {
            name: (self._native.create_buffer(size=_aligned_size(array.nbytes), usage=usage),
                   _aligned_size(array.nbytes), array.nbytes)
            for name, array in raw_arrays.items()
        }

    def upload(self, result: RenderResult) -> RenderBufferLayout:
        """Upload a result, reusing allocations while its layout is unchanged."""
        packed = pack_render_result(result)
        self._ensure_buffers(packed)
        for name, array in self._raw_arrays(packed).items():
            buffer, padded_size, raw_size = self._buffers[name]
            payload = array.tobytes()
            if padded_size != raw_size:
                payload += bytes(padded_size - raw_size)
            self._native.queue.write_buffer(buffer, 0, payload)
        self._packed = packed
        return packed.layout

    def _read(self, buffer: Any, padded_size: int, raw_size: int) -> bytes:
        import wgpu

        staging = self._native.create_buffer(
            size=padded_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = self._native.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, padded_size)
        self._native.queue.submit([encoder.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        data = bytes(staging.read_mapped())[:raw_size]
        staging.unmap()
        return data

    def readback(self) -> RenderResult:
        """Read the last upload back as a semantically equivalent ``RenderResult``."""
        if self._packed is None:
            raise RuntimeError("upload must be called before readback")
        arrays: dict[str, np.ndarray] = {}
        for name, (buffer, padded_size, raw_size) in self._buffers.items():
            data = self._read(buffer, padded_size, raw_size)
            source = self._raw_arrays(self._packed)[name]
            arrays[name] = np.frombuffer(data, dtype=source.dtype).reshape(source.shape).copy()
        return RenderResult(
            colour=arrays["colour"], depth=arrays["depth"], ownership=arrays["ownership"],
            valid_pixels=arrays["valid_pixels"].astype(bool),
            uv=arrays.get("uv"),
        )
