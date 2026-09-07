from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.domain.models import BlockState, BlockVolume


class LitematicExportError(ValueError):
    """Raised when a volume cannot be represented as a valid litematic."""


def pack_palette_indices(indices: Iterable[int], bits_per_entry: int) -> list[int]:
    """Pack palette indices into signed 64-bit long-array values."""
    if bits_per_entry < 1 or bits_per_entry > 63:
        raise ValueError("bits_per_entry must be between 1 and 63")
    values = list(indices)
    limit = 1 << bits_per_entry
    if any(value < 0 or value >= limit for value in values):
        raise ValueError("palette index does not fit the requested bit width")

    longs = [0] * math.ceil(len(values) * bits_per_entry / 64)
    long_mask = (1 << 64) - 1
    for index, value in enumerate(values):
        bit_index = index * bits_per_entry
        long_index, offset = divmod(bit_index, 64)
        longs[long_index] |= (value << offset) & long_mask
        if offset + bits_per_entry > 64:
            longs[long_index + 1] |= value >> (64 - offset)
    return [_as_signed_long(value) for value in longs]


def _as_signed_long(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


class LitematicExporter:
    """Write a single-region Litematica file for a Minecraft 1.21.x project."""

    def __init__(self, minecraft_data_version: int = 4440, schematic_version: int = 7) -> None:
        self.minecraft_data_version = minecraft_data_version
        self.schematic_version = schematic_version

    def export(
        self,
        volume: BlockVolume,
        destination: Path | str,
        *,
        name: str = "Minecraft scan",
        description: str = "Reconstructed Minecraft build",
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    ) -> None:
        # A string destination used to reach `destination.parent` and raise
        # `AttributeError` after the work was done, which threw away a two and a
        # half hour reconstruction of Template 5. Accept either.
        destination = Path(destination)
        if not volume.blocks:
            raise LitematicExportError("cannot export an empty volume")
        if not name:
            raise LitematicExportError("export name cannot be empty")

        # A subset of some other litematic needs that litematic's region, not
        # its own tight one, or its local coordinates come out shifted and it
        # stops being a drop-in reference to compare against.
        if bounds is None:
            minimum, maximum = self._bounds(volume)
        else:
            minimum, maximum = (tuple(int(v) for v in bounds[0]),
                                tuple(int(v) for v in bounds[1]))
            tight_low, tight_high = self._bounds(volume)
            if any(tight_low[i] < minimum[i] or tight_high[i] > maximum[i]
                   for i in range(3)):
                raise LitematicExportError(
                    "volume has blocks outside the requested bounds")
        size = tuple(maximum[index] - minimum[index] + 1 for index in range(3))
        palette_states = [BlockState("minecraft:air")]
        for _, observation in sorted(volume.items()):
            if observation.state not in palette_states:
                palette_states.append(observation.state)
        palette = {state: index for index, state in enumerate(palette_states)}
        indices = []
        for y in range(size[1]):
            for z in range(size[2]):
                for x in range(size[0]):
                    observation = volume.get((minimum[0] + x, minimum[1] + y, minimum[2] + z))
                    indices.append(0 if observation is None else palette[observation.state])
        bits = max(2, (len(palette) - 1).bit_length())
        block_states = pack_palette_indices(indices, bits)

        try:
            import nbtlib
            from nbtlib import tag
        except ImportError as error:
            raise LitematicExportError(
                "litematic export requires nbtlib; install project dependencies first"
            ) from error

        now = int(time.time() * 1000)
        root = {
            "Version": tag.Int(self.schematic_version),
            "SubVersion": tag.Int(1),
            "MinecraftDataVersion": tag.Int(self.minecraft_data_version),
            "Metadata": tag.Compound(
                {
                    "Author": tag.String("mc-photogrammetry"),
                    "Description": tag.String(description),
                    "Name": tag.String(name),
                    "RegionCount": tag.Int(1),
                    "TimeCreated": tag.Long(now),
                    "TimeModified": tag.Long(now),
                    "TotalBlocks": tag.Int(len(volume.blocks)),
                    "TotalVolume": tag.Int(size[0] * size[1] * size[2]),
                    "EnclosingSize": tag.Compound({
                        "x": tag.Int(size[0]), "y": tag.Int(size[1]), "z": tag.Int(size[2])
                    }),
                }
            ),
            "Regions": tag.Compound(
                {
                    "scan": tag.Compound(
                        {
                            "Position": tag.Compound({
                                "x": tag.Int(0), "y": tag.Int(0), "z": tag.Int(0)
                            }),
                            "Size": tag.Compound({
                                "x": tag.Int(size[0]), "y": tag.Int(size[1]), "z": tag.Int(size[2])
                            }),
                            "BlockStatePalette": tag.List[tag.Compound](
                                [_state_tag(state, tag) for state in palette_states]
                            ),
                            "BlockStates": tag.LongArray(block_states),
                            "PendingBlockTicks": tag.List[tag.Compound]([]),
                            "PendingFluidTicks": tag.List[tag.Compound]([]),
                            "TileEntities": tag.List[tag.Compound]([]),
                            "Entities": tag.List[tag.Compound]([]),
                        }
                    )
                }
            ),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbtlib.File(tag.Compound(root), gzipped=True).save(destination)

    @staticmethod
    def _bounds(volume: BlockVolume) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        coordinates = list(volume.blocks)
        minimum = tuple(min(coordinate[index] for coordinate in coordinates) for index in range(3))
        maximum = tuple(max(coordinate[index] for coordinate in coordinates) for index in range(3))
        return minimum, maximum


def _state_tag(state: BlockState, tag: object) -> object:
    state_tag = {"Name": tag.String(state.name)}  # type: ignore[attr-defined]
    if state.properties:
        state_tag["Properties"] = tag.Compound({key: tag.String(value) for key, value in state.properties})  # type: ignore[attr-defined]
    return tag.Compound(state_tag)  # type: ignore[attr-defined]
