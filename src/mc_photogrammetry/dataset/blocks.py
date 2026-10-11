"""Exact blockstate JSON -> BlockVolume -> reference.litematic.

The mod exports the server world's real ``BlockState`` names and property maps
for the declared region; nothing here is inferred from an image. Air is dropped
when building the sparse volume because ``BlockVolume`` represents empty space
by absence, but it stays in the recorded JSON so a diff can still see it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mc_photogrammetry.dataset.schemas import (
    BlockRecord,
    DatasetSchemaError,
    GroundTruthExport,
    RegionBounds,
)
from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.litematic import LitematicExporter

AIR_BLOCKS = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")


def parse_ground_truth(payload: Mapping[str, Any]) -> GroundTruthExport:
    return GroundTruthExport.from_dict(payload)


def block_state_for(record: BlockRecord) -> BlockState:
    """Convert one exported record into the project's normalised block state."""
    properties = tuple(sorted((str(key), str(value)) for key, value in record.properties.items()))
    return BlockState(name=record.name, properties=properties)


def volume_from_ground_truth(export: GroundTruthExport, *,
                             include_air: bool = False) -> BlockVolume:
    """Build a sparse volume; by default air coordinates stay absent."""
    volume = BlockVolume()
    for record in export.blocks:
        if not include_air and record.name in AIR_BLOCKS:
            continue
        volume.set(record.position, BlockObservation(state=block_state_for(record)))
    return volume


def write_reference_litematic(
    export: GroundTruthExport,
    destination: Path | str,
    *,
    name: str | None = None,
    description: str = "Minecraft capture ground truth",
) -> Path:
    """Write ``reference.litematic`` pinned to the declared region bounds.

    The bounds are passed explicitly so the litematic keeps the region's
    coordinates even when the region has air-only margins. Without that, the
    exporter shrink-wraps to the blocks present and the reference stops being
    coordinate-comparable to the camera positions that produced it.
    """
    destination = Path(destination)
    volume = volume_from_ground_truth(export)
    if not volume.blocks:
        raise DatasetSchemaError("ground truth contains no non-air blocks")
    if destination.exists():
        raise DatasetSchemaError(f"refusing to overwrite {destination}")
    LitematicExporter().export(
        volume,
        destination,
        name=name or export.scene_id,
        description=description,
        bounds=(export.region.minimum, export.region.maximum),
    )
    return destination


def ground_truth_for_volume(
    *,
    scene_id: str,
    world_identity: str,
    region: RegionBounds,
    volume: BlockVolume,
) -> GroundTruthExport:
    """Fixture helper: an export record for an in-memory volume."""
    blocks: list[BlockRecord] = []
    for coordinate, observation in sorted(volume.items()):
        if not region.contains(coordinate):
            raise DatasetSchemaError(f"block {coordinate} lies outside the declared region")
        blocks.append(BlockRecord(
            position=coordinate,
            name=observation.state.name,
            properties={key: value for key, value in observation.state.properties},
        ))
    return GroundTruthExport(
        scene_id=scene_id,
        world_identity=world_identity,
        region=region,
        blocks=tuple(blocks),
    )
