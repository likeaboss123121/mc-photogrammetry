"""Exporting a reconstruction as a Minecraft world.

A litematic needs a mod to look at; a world folder does not. The Anvil format has
two traps that produce a world which loads cleanly as empty air, so both are
pinned here: packed indices must not span longs, and region files are addressed in
sectors with the two header sectors counted.
"""
import gzip
import zipfile
from io import BytesIO

import pytest

from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.anvil import (
    AnvilExportError,
    AnvilExporter,
    pack_section_indices,
    read_world,
)

STONE = BlockState("minecraft:stone")
LOG = BlockState("minecraft:oak_log", (("axis", "x"),))


def _volume(cells):
    volume = BlockVolume()
    for coordinate, state in cells.items():
        volume.set(coordinate, BlockObservation(state=state))
    return volume


def _write(tmp_path, cells, **kwargs):
    path = tmp_path / "world.zip"
    AnvilExporter().export(_volume(cells), path, **kwargs)
    return path


def test_entries_are_written_where_minecraft_looks_for_them(tmp_path):
    path = _write(tmp_path, {(0, 0, 0): STONE}, name="Scan")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    assert "Scan/level.dat" in names
    assert any(name.startswith("Scan/region/r.") and name.endswith(".mca") for name in names)


def test_a_volume_survives_the_round_trip(tmp_path):
    cells = {
        (x, y, z): (LOG if (x + y + z) % 3 == 0 else STONE)
        for x in range(4)
        for y in range(3)
        for z in range(5)
    }
    path = _write(tmp_path, cells)
    assert read_world(path) == cells


def test_block_properties_survive(tmp_path):
    path = _write(tmp_path, {(1, 2, 3): LOG})
    assert read_world(path)[(1, 2, 3)] == LOG


def test_the_volume_lands_at_the_requested_origin(tmp_path):
    path = _write(tmp_path, {(0, 0, 0): STONE, (1, 0, 0): LOG}, origin=(100, 64, -40))
    assert read_world(path) == {(100, 64, -40): STONE, (101, 64, -40): LOG}


def test_negative_coordinates_reach_the_right_region(tmp_path):
    cells = {(-40, 8, -70): STONE, (-1, 8, -1): LOG}
    path = _write(tmp_path, cells)
    with zipfile.ZipFile(path) as archive:
        regions = {n for n in archive.namelist() if n.endswith(".mca")}
    assert len(regions) >= 1
    assert read_world(path) == cells


def test_blocks_spanning_several_regions_all_arrive(tmp_path):
    # A region covers 512 blocks, so these land in three different files and
    # exercise the sector accounting independently in each.
    cells = {(0, 0, 0): STONE, (600, 0, 0): LOG, (0, 0, 600): STONE}
    path = _write(tmp_path, cells)
    with zipfile.ZipFile(path) as archive:
        assert len({n for n in archive.namelist() if n.endswith(".mca")}) == 3
    assert read_world(path) == cells


def test_a_palette_too_large_for_four_bits_still_decodes(tmp_path):
    # Above sixteen entries the packing widens, which is where an implementation
    # that lets entries span longs starts returning noise.
    cells = {
        (x, 0, 0): BlockState(f"minecraft:test_block_{x}")
        for x in range(20)
    }
    path = _write(tmp_path, cells)
    assert read_world(path) == cells


def test_indices_do_not_span_two_longs():
    # Twelve five-bit entries fit as twelve per long only if the spare four bits
    # are wasted; packing continuously would fit twelve and four fifths.
    packed = pack_section_indices([1] * 12, 5)
    assert len(packed) == 1
    value = packed[0] & ((1 << 64) - 1)
    assert [(value >> (i * 5)) & 31 for i in range(12)] == [1] * 12
    assert value >> 60 == 0


def test_packing_starts_a_new_long_rather_than_overflowing():
    packed = pack_section_indices([3] * 13, 5)
    assert len(packed) == 2


def test_level_dat_is_gzipped_nbt_naming_the_world(tmp_path):
    import nbtlib

    path = _write(tmp_path, {(0, 0, 0): STONE}, name="Scan")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("Scan/level.dat")
    parsed = nbtlib.File.parse(BytesIO(gzip.decompress(raw)), byteorder="big")
    data = parsed.get("", parsed)["Data"]
    assert str(data["LevelName"]) == "Scan"
    assert "WorldGenSettings" in data
    assert int(data["DataVersion"]) > 0


def test_the_spawn_sits_at_the_build(tmp_path):
    import nbtlib

    path = _write(tmp_path, {(5, 70, 9): STONE}, name="Scan")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("Scan/level.dat")
    parsed = nbtlib.File.parse(BytesIO(gzip.decompress(raw)), byteorder="big")
    data = parsed.get("", parsed)["Data"]
    assert (int(data["SpawnX"]), int(data["SpawnZ"])) == (5, 9)
    assert int(data["SpawnY"]) >= 70


def test_an_empty_volume_is_refused(tmp_path):
    with pytest.raises(AnvilExportError):
        AnvilExporter().export(BlockVolume(), tmp_path / "world.zip")


def test_an_unnamed_world_is_refused(tmp_path):
    with pytest.raises(AnvilExportError):
        AnvilExporter().export(_volume({(0, 0, 0): STONE}), tmp_path / "world.zip", name="")


def test_a_string_destination_is_accepted(tmp_path):
    """A `str` used to raise `AttributeError` only after the volume was built.

    That is the worst possible place for a type error: on a large fixture the
    reconstruction can take hours, and all of it was lost at the final line.
    """
    path = tmp_path / "world.zip"
    AnvilExporter().export(_volume({(0, 0, 0): STONE}), str(path), name="Scan")
    assert read_world(path) == {(0, 0, 0): STONE}
