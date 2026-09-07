from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.reconstruction.raycast import _remove_enclosed_blocks


def test_enclosed_blocks_are_removed_from_visible_surface():
    volume = BlockVolume()
    for x in range(3):
        for y in range(3):
            for z in range(3):
                volume.set((x, y, z), BlockObservation(BlockState("minecraft:stone")))

    _remove_enclosed_blocks(volume)

    assert (1, 1, 1) not in volume.blocks
    assert len(volume.blocks) == 26