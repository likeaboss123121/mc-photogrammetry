import pytest

from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.litematic import LitematicExporter, pack_palette_indices
from mc_photogrammetry.export.compare import compare_litematics
from mc_photogrammetry.export.validation import validate_litematic


def test_palette_indices_pack_across_long_boundary():
    packed = pack_palette_indices([0, 1, 2, 3], bits_per_entry=2)

    assert packed == [228]


def test_palette_indices_reject_values_that_do_not_fit():
    with pytest.raises(ValueError, match="fit"):
        pack_palette_indices([4], bits_per_entry=2)


def test_palette_indices_reject_invalid_bit_width():
    with pytest.raises(ValueError, match="between"):
        pack_palette_indices([0], bits_per_entry=0)


def test_palette_indices_round_trip_across_long_boundaries():
    values = [0, 17, 3, 31, 8, 22, 1, 30, 14, 5, 27, 2, 19]
    packed = pack_palette_indices(values, bits_per_entry=5)
    decoded = []
    mask = (1 << 5) - 1
    for index in range(len(values)):
        bit_index = index * 5
        long_index, offset = divmod(bit_index, 64)
        value = (packed[long_index] & ((1 << 64) - 1)) >> offset
        if offset + 5 > 64:
            value |= (packed[long_index + 1] & ((1 << 64) - 1)) << (64 - offset)
        decoded.append(value & mask)
    assert decoded == values


def test_exporter_writes_gzipped_litematic(tmp_path):
    import nbtlib

    volume = BlockVolume()
    volume.set((4, 64, -2), BlockObservation(BlockState("minecraft:stone")))
    volume.set((5, 64, -2), BlockObservation(BlockState("minecraft:oak_log", (("axis", "y"),))))
    destination = tmp_path / "scan.litematic"

    LitematicExporter().export(volume, destination, name="Fixture")
    result = validate_litematic(destination)

    assert result["dimensions"] == (2, 1, 1)
    assert result["position"] == (0, 0, 0)
    assert result["palette_size"] == 3

    loaded = nbtlib.load(destination, gzipped=True)
    root = loaded
    assert root["Version"] == 7
    assert root["SubVersion"] == 1
    assert root["MinecraftDataVersion"] == 4440

    comparison = compare_litematics(destination, destination)
    assert comparison.exact_coordinate_matches == 2
    assert comparison.exact_state_matches == 2
    assert comparison.coordinate_recall == 1.0
    assert comparison.best_translation == (0, 0, 0)
    assert comparison.direct_coordinate_matches == 2
    assert comparison.coordinate_precision == 1.0
    assert comparison.coordinate_f1 == 1.0


def test_a_string_destination_is_accepted(tmp_path):
    """A `str` used to raise `AttributeError` only after the volume was built.

    That is the worst place for a type error: on a large fixture the
    reconstruction takes hours, and all of it was lost at the final line.
    """
    volume = BlockVolume()
    volume.set((0, 0, 0), BlockObservation(state=BlockState("minecraft:stone")))
    path = tmp_path / "scan.litematic"
    LitematicExporter().export(volume, str(path), name="Scan")
    assert path.exists()


def test_explicit_bounds_keep_a_subset_in_its_parent_region(tmp_path):
    """A camera-reachable subset must stay comparable to the litematic it came from.

    Without this the exporter shrink-wraps the region to the blocks present, the
    subset's local coordinates shift by however far the box moved, and it stops
    being a drop-in reference. Templates 1 and 3 both shift: their reachable
    blocks start at y=14 and (y=2, z=23) rather than at the origin.
    """
    from mc_photogrammetry.export.compare import load_litematic

    volume = BlockVolume()
    volume.set((4, 5, 6), BlockObservation(state=BlockState("minecraft:stone")))
    destination = tmp_path / "subset.litematic"
    LitematicExporter().export(volume, destination, name="subset",
                               bounds=((0, 0, 0), (9, 9, 9)))
    loaded = load_litematic(destination)
    assert loaded.size == (10, 10, 10)
    assert (4, 5, 6) in loaded.blocks, "the block moved when the region was pinned"


def test_bounds_that_would_drop_a_block_are_refused(tmp_path):
    from mc_photogrammetry.export.litematic import LitematicExportError

    volume = BlockVolume()
    volume.set((4, 5, 6), BlockObservation(state=BlockState("minecraft:stone")))
    with pytest.raises(LitematicExportError, match="outside"):
        LitematicExporter().export(volume, tmp_path / "bad.litematic", name="bad",
                                   bounds=((0, 0, 0), (2, 2, 2)))
