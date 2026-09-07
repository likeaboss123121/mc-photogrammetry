"""Write a reconstructed volume as a Minecraft world that can be opened directly.

A litematic needs a mod to view. A world folder does not: zip it, drop it in the
saves directory, and the build is standing there. The format is Anvil - one region
file per 512-block square, holding up to 1024 chunks, each chunk a compressed NBT
compound whose sections carry a block palette and the indices into it.

Two details of the format are easy to get wrong and silently produce a world that
loads as empty air:

* Since 1.16 the packed indices do **not** span longs. Each long holds as many
  whole entries as fit and the spare high bits are left unused, unlike the
  litematic format, which packs continuously.
* A region file is addressed in 4096-byte sectors, and the two header sectors
  count, so the first chunk begins at sector two.
"""

from __future__ import annotations

import math
import time
import zipfile
from io import BytesIO
from pathlib import Path

from mc_photogrammetry.domain.models import BlockState, BlockVolume

SECTOR = 4096
"""Region files are addressed in sectors of this many bytes."""

SECTION = 16
"""A chunk section is a 16-block cube."""

AIR = BlockState("minecraft:air")


class AnvilExportError(ValueError):
    """Raised when a volume cannot be written as a world."""


def pack_section_indices(indices: list[int], bits: int) -> list[int]:
    """Pack palette indices into longs without letting an entry span two longs.

    This is what separates the chunk format from the litematic one. Packing
    continuously, as a litematic does, shifts every entry after the first
    boundary and the chunk decodes as noise.
    """
    if bits <= 0:
        raise AnvilExportError("bits per entry must be positive")
    per_long = 64 // bits
    longs: list[int] = []
    for start in range(0, len(indices), per_long):
        value = 0
        for offset, index in enumerate(indices[start : start + per_long]):
            value |= (index & ((1 << bits) - 1)) << (offset * bits)
        longs.append(value - (1 << 64) if value >= (1 << 63) else value)
    return longs


def _state_tag(state: BlockState, tag) -> object:
    entry = tag.Compound({"Name": tag.String(state.name)})
    if state.properties:
        entry["Properties"] = tag.Compound(
            {str(key): tag.String(str(value)) for key, value in state.properties}
        )
    return entry


def _section_tag(states: list[BlockState], tag) -> object:
    """One section's block data: its palette and the indices into it."""
    palette: list[BlockState] = []
    position: dict[BlockState, int] = {}
    indices: list[int] = []
    for state in states:
        found = position.get(state)
        if found is None:
            found = len(palette)
            position[state] = found
            palette.append(state)
        indices.append(found)
    block_states = tag.Compound(
        {"palette": tag.List[tag.Compound]([_state_tag(s, tag) for s in palette])}
    )
    if len(palette) > 1:
        bits = max(4, (len(palette) - 1).bit_length())
        block_states["data"] = tag.LongArray(pack_section_indices(indices, bits))
    return block_states


class AnvilExporter:
    """Write a block volume as a zipped Minecraft world."""

    def __init__(self, minecraft_data_version: int = 4440, version_name: str = "1.21.5") -> None:
        self.minecraft_data_version = minecraft_data_version
        self.version_name = version_name

    def export(
        self,
        volume: BlockVolume,
        destination: Path | str,
        *,
        name: str = "Minecraft scan",
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Write ``volume`` to ``destination`` as a world.zip.

        ``origin`` is the world position the volume's own (0, 0, 0) is placed at,
        so a reconstruction can be written back where it was found.
        """
        destination = Path(destination)
        if not volume.blocks:
            raise AnvilExportError("cannot export an empty volume")
        if not name:
            raise AnvilExportError("export name cannot be empty")
        try:
            import nbtlib
            from nbtlib import tag
        except ImportError as error:
            raise AnvilExportError(
                "world export requires nbtlib; install project dependencies first"
            ) from error

        placed = {
            (
                coordinate[0] + origin[0],
                coordinate[1] + origin[1],
                coordinate[2] + origin[2],
            ): observation.state
            for coordinate, observation in volume.items()
        }
        by_chunk: dict[tuple[int, int], dict[tuple[int, int, int], BlockState]] = {}
        for coordinate, state in placed.items():
            key = (coordinate[0] >> 4, coordinate[2] >> 4)
            by_chunk.setdefault(key, {})[coordinate] = state

        by_region: dict[tuple[int, int], dict[tuple[int, int], bytes]] = {}
        for (chunk_x, chunk_z), blocks in by_chunk.items():
            payload = self._chunk_bytes(chunk_x, chunk_z, blocks, nbtlib, tag)
            by_region.setdefault((chunk_x >> 5, chunk_z >> 5), {})[(chunk_x, chunk_z)] = payload

        spawn = min(placed) if placed else (0, 64, 0)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{name}/level.dat", self._level_bytes(name, spawn, nbtlib, tag))
            for (region_x, region_z), chunks in by_region.items():
                archive.writestr(
                    f"{name}/region/r.{region_x}.{region_z}.mca", _region_bytes(chunks)
                )

    def _chunk_bytes(self, chunk_x, chunk_z, blocks, nbtlib, tag) -> bytes:
        lowest = min(coordinate[1] for coordinate in blocks)
        highest = max(coordinate[1] for coordinate in blocks)
        sections = []
        for section_y in range(math.floor(lowest / SECTION), math.floor(highest / SECTION) + 1):
            base = section_y * SECTION
            states = [
                blocks.get((chunk_x * 16 + x, base + y, chunk_z * 16 + z), AIR)
                # a section is stored y, then z, then x
                for y in range(SECTION)
                for z in range(SECTION)
                for x in range(SECTION)
            ]
            sections.append(
                tag.Compound(
                    {
                        "Y": tag.Byte(section_y),
                        "block_states": _section_tag(states, tag),
                        "biomes": tag.Compound(
                            {"palette": tag.List[tag.String]([tag.String("minecraft:plains")])}
                        ),
                    }
                )
            )
        chunk = tag.Compound(
            {
                "DataVersion": tag.Int(self.minecraft_data_version),
                "xPos": tag.Int(chunk_x),
                "yPos": tag.Int(math.floor(lowest / SECTION)),
                "zPos": tag.Int(chunk_z),
                "Status": tag.String("minecraft:full"),
                "sections": tag.List[tag.Compound](sections),
                "block_entities": tag.List[tag.Compound]([]),
                "block_ticks": tag.List[tag.Compound]([]),
                "fluid_ticks": tag.List[tag.Compound]([]),
                "Heightmaps": tag.Compound({}),
                "structures": tag.Compound(
                    {"References": tag.Compound({}), "starts": tag.Compound({})}
                ),
            }
        )
        buffer = BytesIO()
        nbtlib.File(chunk).write(buffer, byteorder="big")
        return buffer.getvalue()

    def _level_bytes(self, name, spawn, nbtlib, tag) -> bytes:
        flat = tag.Compound(
            {
                "type": tag.String("minecraft:flat"),
                "settings": tag.Compound(
                    {
                        "biome": tag.String("minecraft:plains"),
                        "lakes": tag.Byte(0),
                        "features": tag.Byte(0),
                        "layers": tag.List[tag.Compound](
                            [
                                tag.Compound(
                                    {
                                        "block": tag.String("minecraft:air"),
                                        "height": tag.Int(1),
                                    }
                                )
                            ]
                        ),
                    }
                ),
            }
        )
        data = tag.Compound(
            {
                "DataVersion": tag.Int(self.minecraft_data_version),
                "version": tag.Compound(
                    {
                        "Id": tag.Int(self.minecraft_data_version),
                        "Name": tag.String(self.version_name),
                        "Snapshot": tag.Byte(0),
                    }
                ),
                "LevelName": tag.String(name),
                "GameType": tag.Int(1),
                "Difficulty": tag.Byte(0),
                "allowCommands": tag.Byte(1),
                "hardcore": tag.Byte(0),
                "initialized": tag.Byte(1),
                "raining": tag.Byte(0),
                "thundering": tag.Byte(0),
                "Time": tag.Long(0),
                "DayTime": tag.Long(6000),
                "LastPlayed": tag.Long(int(time.time() * 1000)),
                "SpawnX": tag.Int(int(spawn[0])),
                "SpawnY": tag.Int(int(spawn[1]) + 2),
                "SpawnZ": tag.Int(int(spawn[2])),
                "SpawnAngle": tag.Float(0.0),
                "WorldGenSettings": tag.Compound(
                    {
                        "seed": tag.Long(0),
                        "generate_features": tag.Byte(0),
                        "bonus_chest": tag.Byte(0),
                        "dimensions": tag.Compound(
                            {
                                "minecraft:overworld": tag.Compound(
                                    {
                                        "type": tag.String("minecraft:overworld"),
                                        "generator": flat,
                                    }
                                )
                            }
                        ),
                    }
                ),
                "DataPacks": tag.Compound(
                    {"Enabled": tag.List[tag.String]([tag.String("vanilla")]),
                     "Disabled": tag.List[tag.String]([])}
                ),
                "GameRules": tag.Compound({"doDaylightCycle": tag.String("false")}),
            }
        )
        buffer = BytesIO()
        nbtlib.File(tag.Compound({"Data": data})).write(buffer, byteorder="big")
        raw = buffer.getvalue()
        import gzip

        return gzip.compress(raw)


def _region_bytes(chunks: dict[tuple[int, int], bytes]) -> bytes:
    """Assemble one region file from its chunks.

    The first two sectors are the location and timestamp tables, so chunk data
    starts at sector two; a location entry pointing at sector zero means the chunk
    is absent.
    """
    import zlib

    locations = bytearray(SECTOR)
    timestamps = bytearray(SECTOR)
    body = bytearray()
    next_sector = 2
    stamp = int(time.time())
    for (chunk_x, chunk_z), payload in sorted(chunks.items()):
        compressed = zlib.compress(payload)
        framed = len(compressed) + 1
        block = bytearray(framed.to_bytes(4, "big") + bytes([2]) + compressed)
        used = math.ceil(len(block) / SECTOR)
        block.extend(b"\x00" * (used * SECTOR - len(block)))
        index = ((chunk_x & 31) + (chunk_z & 31) * 32) * 4
        locations[index : index + 3] = next_sector.to_bytes(3, "big")
        locations[index + 3] = used
        timestamps[index : index + 4] = stamp.to_bytes(4, "big")
        body.extend(block)
        next_sector += used
    return bytes(locations) + bytes(timestamps) + bytes(body)


def read_world(path: Path) -> dict[tuple[int, int, int], BlockState]:
    """Read a world.zip back, so an export can be checked against what it meant.

    Deliberately independent of the writer: it walks the region files and unpacks
    the sections rather than trusting anything the exporter remembered.
    """
    import zlib

    import nbtlib

    blocks: dict[tuple[int, int, int], BlockState] = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            if not entry.endswith(".mca"):
                continue
            raw = archive.read(entry)
            locations = raw[:SECTOR]
            for slot in range(1024):
                offset = int.from_bytes(locations[slot * 4 : slot * 4 + 3], "big")
                count = locations[slot * 4 + 3]
                if offset == 0 or count == 0:
                    continue
                start = offset * SECTOR
                length = int.from_bytes(raw[start : start + 4], "big")
                payload = raw[start + 5 : start + 4 + length]
                chunk = nbtlib.File.parse(BytesIO(zlib.decompress(payload)), byteorder="big")
                root = chunk.get("", chunk)
                chunk_x, chunk_z = int(root["xPos"]), int(root["zPos"])
                for section in root["sections"]:
                    states = section.get("block_states")
                    if states is None or "palette" not in states:
                        continue
                    palette = [
                        BlockState(
                            str(item["Name"]),
                            tuple(sorted(
                                (str(k), str(v)) for k, v in item.get("Properties", {}).items()
                            )),
                        )
                        for item in states["palette"]
                    ]
                    base = int(section["Y"]) * SECTION
                    data = states.get("data")
                    if data is None:
                        indices = [0] * (SECTION**3)
                    else:
                        bits = max(4, (len(palette) - 1).bit_length())
                        per_long = 64 // bits
                        mask = (1 << bits) - 1
                        indices = []
                        for value in data:
                            unsigned = int(value) & ((1 << 64) - 1)
                            for slot_index in range(per_long):
                                indices.append((unsigned >> (slot_index * bits)) & mask)
                        indices = indices[: SECTION**3]
                    for position, index in enumerate(indices):
                        state = palette[index] if index < len(palette) else AIR
                        if state.name == "minecraft:air":
                            continue
                        y, rest = divmod(position, SECTION * SECTION)
                        z, x = divmod(rest, SECTION)
                        blocks[(chunk_x * 16 + x, base + y, chunk_z * 16 + z)] = state
    return blocks
