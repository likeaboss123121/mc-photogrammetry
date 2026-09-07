from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from mc_photogrammetry.domain.models import BlockVolume, ScanProject


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
    def solve(self, frames: Sequence[np.ndarray]) -> object: ...


class TextureProvider(Protocol):
    def resolve(self, block_name: str) -> object | None: ...

    def load_jar(self, path: Path) -> None: ...


class BlockClassifier(Protocol):
    def classify(self, image: np.ndarray, mask: np.ndarray) -> str | None: ...


class Reconstructor(Protocol):
    def reconstruct(self, project: ScanProject) -> BlockVolume: ...


class StructureExporter(Protocol):
    def export(self, volume: BlockVolume, destination: Path) -> None: ...
