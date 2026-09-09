"""Batched colour scoring with a CPU reference and optional wgpu execution."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from mc_photogrammetry.pipeline.contracts import ComputeDevice

from .wgpu_provider import WgpuComputeProvider

_WGSL = """
struct Params {
    candidate_count: u32,
    item_count: u32,
}

@group(0) @binding(0) var<storage, read> candidates: array<f32>;
@group(0) @binding(1) var<storage, read> observed: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let candidate = global_id.x;
    if (candidate >= params.candidate_count) {
        return;
    }
    var sum = 0.0;
    let start = candidate * params.item_count;
    for (var item = 0u; item < params.item_count; item = item + 1u) {
        let delta = candidates[start + item] - observed[item];
        sum = sum + delta * delta;
    }
    scores[candidate] = sqrt(sum / f32(params.item_count));
}
"""


@dataclass(frozen=True, slots=True)
class ScoreBatch:
    """Scores plus execution provenance for a batched comparison."""

    scores: np.ndarray
    device: ComputeDevice
    elapsed_seconds: float


def _normalise_inputs(observed: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if observed.ndim != 1:
        raise ValueError("observed must be a one-dimensional flattened feature")
    if candidates.ndim != 2 or candidates.shape[1] != observed.shape[0]:
        raise ValueError("candidates must have shape (count, observed.size)")
    if not observed.size:
        raise ValueError("observed must not be empty")
    return np.ascontiguousarray(observed), np.ascontiguousarray(candidates)


def rgb_l2_cpu(observed: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Reference root-mean-square RGB distance for flattened candidate batches."""
    observed, candidates = _normalise_inputs(observed, candidates)
    return np.sqrt(np.mean((candidates - observed[None, :]) ** 2, axis=1, dtype=np.float32))


def _buffer_data(device: Any, data: np.ndarray, usage: int) -> Any:
    return device.create_buffer_with_data(data=data.tobytes(), usage=usage)


class WgpuScoringContext:
    """Reusable device/pipeline context for repeated batched scoring calls."""

    def __init__(
        self,
        *,
        provider: WgpuComputeProvider | None = None,
        device: ComputeDevice | None = None,
    ) -> None:
        provider = provider or WgpuComputeProvider()
        selected = device or provider.select()
        native = provider.request_device(selected)
        self.provider = provider
        self.device = selected
        self._native = native
        self._pipeline, self._layout = self._create_pipeline(native)

    @staticmethod
    def _create_pipeline(native: Any) -> tuple[Any, Any]:
        import wgpu

        shader = native.create_shader_module(code=_WGSL)
        layout = native.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "uniform"}},
        ])
        pipeline = native.create_compute_pipeline(
            layout=native.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={"module": shader, "entry_point": "main"},
        )
        return pipeline, layout

    def score(self, observed: np.ndarray, candidates: np.ndarray) -> ScoreBatch:
        """Run one batch using the cached native device and compute pipeline."""
        observed, candidates = _normalise_inputs(observed, candidates)
        import wgpu

        native = self._native
        started = perf_counter()
        if candidates.shape[0] == 0:
            return ScoreBatch(np.empty(0, dtype=np.float32), self.device, perf_counter() - started)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        candidate_buffer = _buffer_data(native, candidates, usage)
        observed_buffer = _buffer_data(native, observed, usage)
        output_size = max(4, candidates.shape[0] * 4)
        output_buffer = native.create_buffer(
            size=output_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        params = np.asarray((candidates.shape[0], candidates.shape[1]), dtype=np.uint32)
        params_buffer = _buffer_data(
            native, params, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        staging = native.create_buffer(
            size=output_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        bind_group = native.create_bind_group(layout=self._layout, entries=[
            {"binding": 0, "resource": {"buffer": candidate_buffer}},
            {"binding": 1, "resource": {"buffer": observed_buffer}},
            {"binding": 2, "resource": {"buffer": output_buffer}},
            {"binding": 3, "resource": {"buffer": params_buffer}},
        ])
        encoder = native.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups = (candidates.shape[0] + 63) // 64
        compute_pass.dispatch_workgroups(workgroups)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging, 0, output_size)
        native.queue.submit([encoder.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        result = np.frombuffer(
            staging.read_mapped(), dtype=np.float32, count=candidates.shape[0],
        ).copy()
        staging.unmap()
        return ScoreBatch(result, self.device, perf_counter() - started)


def rgb_l2_wgpu(
    observed: np.ndarray,
    candidates: np.ndarray,
    *,
    provider: WgpuComputeProvider | None = None,
    device: ComputeDevice | None = None,
    context: WgpuScoringContext | None = None,
) -> ScoreBatch:
    """Run the batched RMS distance kernel on an enumerated wgpu device."""
    if context is not None:
        return context.score(observed, candidates)
    context = WgpuScoringContext(provider=provider, device=device)
    return context.score(observed, candidates)


def score_batch(
    observed: np.ndarray,
    candidates: np.ndarray,
    *,
    backend: str = "cpu",
    provider: WgpuComputeProvider | None = None,
    context: WgpuScoringContext | None = None,
) -> ScoreBatch:
    """Score a batch on CPU, wgpu, or auto-selected wgpu with CPU fallback."""
    started = perf_counter()
    if backend not in {"cpu", "wgpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'wgpu', or 'auto'")
    if backend in {"wgpu", "auto"}:
        try:
            return rgb_l2_wgpu(observed, candidates, provider=provider, context=context)
        except (ImportError, RuntimeError, AttributeError, TypeError):
            if backend == "wgpu":
                raise
    scores = rgb_l2_cpu(observed, candidates)
    return ScoreBatch(scores, ComputeDevice("cpu", "CPU", False), perf_counter() - started)
