from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SceneTransform:
    """Mapping between litematic-local coordinates and Minecraft world coordinates."""

    world_origin: tuple[int, int, int]
    size: tuple[int, int, int]

    def to_world(self, coordinate: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(coordinate[index] + self.world_origin[index] for index in range(3))

    def to_local(self, coordinate: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(coordinate[index] - self.world_origin[index] for index in range(3))

    def contains_world(self, coordinate: tuple[int, int, int]) -> bool:
        local = self.to_local(coordinate)
        return all(0 <= local[index] < self.size[index] for index in range(3))


def load_scene_transform(path: str | Path) -> SceneTransform:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    region = data["reference_region"]
    minimum = tuple(int(value) for value in region["world_min"])
    maximum = tuple(int(value) for value in region["world_max_exclusive"])
    size = tuple(maximum[index] - minimum[index] for index in range(3))
    if len(minimum) != 3 or any(value <= 0 for value in size):
        raise ValueError("reference region must define a positive 3D volume")
    return SceneTransform(world_origin=minimum, size=size)
