from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from mc_photogrammetry.domain.models import BlockVolume, ScanProject


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """The reversible crop/resize mapping applied to one source frame.

    Coordinates in the source image first subtract ``crop_origin`` and then use
    ``scale`` to reach the frame consumed by reconstruction.  Keeping this with
    the frame prevents camera intrinsics being silently reused after a crop.
    """

    source_size: tuple[int, int]
    crop_origin: tuple[int, int] = (0, 0)
    crop_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None

    @property
    def resolved_crop_size(self) -> tuple[int, int]:
        return self.crop_size or self.source_size

    @property
    def resolved_output_size(self) -> tuple[int, int]:
        return self.output_size or self.resolved_crop_size


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Stable image evidence passed between camera, renderer, and reconstruction."""

    frame_id: str
    source_path: Path
    transform: FrameTransform
    quality: float = 1.0
    mask_path: Path | None = None
    feature_cache_key: str | None = None


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    image_size: tuple[int, int]
    focal_length_pixels: tuple[float, float]
    principal_point_pixels: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CameraPose:
    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraEstimate:
    intrinsics: CameraIntrinsics
    pose: CameraPose | None
    residual_pixels: float | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraSolution:
    """Image-derived camera estimates, including unresolved components."""

    cameras: Mapping[str, CameraEstimate]
    components: tuple[tuple[str, ...], ...] = ()
    unresolved: tuple[str, ...] = ()
    scale_known: bool = False


@dataclass(frozen=True, slots=True)
class LatticeScene:
    """A lattice coordinate frame and evidence-backed planar surface proposals."""

    origin: tuple[float, float, float]
    scale: float | None
    planes: tuple[object, ...] = ()
    alternatives: tuple[object, ...] = ()
    observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Renderer outputs; every buffer has the same output image dimensions."""

    colour: np.ndarray
    depth: np.ndarray
    ownership: np.ndarray
    uv: np.ndarray | None = None
    valid_pixels: np.ndarray | None = None


@dataclass(slots=True)
class ReconstructionResult:
    """The best exportable estimate plus alternatives and review evidence."""

    volume: BlockVolume
    alternatives: list[object] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)
    review_items: list[str] = field(default_factory=list)
    timings_seconds: dict[str, float] = field(default_factory=dict)
    complete: bool = False


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    provider: str
    name: str
    supports_gpu: bool


class ComputeProvider(Protocol):
    def available_devices(self) -> Sequence[ComputeDevice]: ...

    def select(self, preferred: str | None = None) -> ComputeDevice: ...


class DepthEstimator(Protocol):
    def estimate(self, image: np.ndarray, device: ComputeDevice) -> np.ndarray: ...


class CameraSolver(Protocol):
    def solve(self, frames: Sequence[np.ndarray]) -> CameraSolution: ...


class TextureProvider(Protocol):
    def resolve(self, block_name: str) -> object | None: ...

    def load_jar(self, path: Path) -> None: ...


class BlockClassifier(Protocol):
    def classify(self, image: np.ndarray, mask: np.ndarray) -> str | None: ...


class Reconstructor(Protocol):
    def reconstruct(self, project: ScanProject) -> BlockVolume: ...


class StructureExporter(Protocol):
    def export(self, volume: BlockVolume, destination: Path) -> None: ...
