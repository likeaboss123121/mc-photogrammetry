import pytest

from mc_photogrammetry.domain.models import BlockObservation, BlockState


def test_block_state_requires_namespace():
    with pytest.raises(ValueError, match="namespace"):
        BlockState("stone")


def test_confidence_is_bounded():
    with pytest.raises(ValueError, match="between 0 and 1"):
        BlockObservation(BlockState("minecraft:stone"), confidence=1.1)
