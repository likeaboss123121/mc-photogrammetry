import numpy as np

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.raycast import (
    _choose_state_with_local_family_support,
    _palette_candidates,
    _patch_descriptor,
    _rectify_projected_face,
)


def test_palette_candidates_follow_plain_color_distance_without_wood_bias():
    palette = (
        PaletteColor(BlockState("minecraft:stone"), (100, 100, 100), family="cube"),
        PaletteColor(BlockState("minecraft:dark_oak_planks"), (90, 70, 43), family="log"),
        PaletteColor(BlockState("minecraft:birch_planks"), (160, 140, 100), family="log"),
    )

    candidates = _palette_candidates(
        np.array([100, 100, 100], dtype=np.float32),
        palette,
        family_priors={"log": 1.0, "cube": 1.0, "other": 1.0},
        limit=3,
    )

    assert candidates[0][0] == BlockState("minecraft:stone")


def test_choose_state_with_no_family_context_uses_best_score():
    candidate_scores = {
        BlockState("minecraft:dirt"): 21.0,
        BlockState("minecraft:stone"): 15.0,
        BlockState("minecraft:oak_planks"): 20.0,
    }
    palette_families = {
        BlockState("minecraft:dirt"): "cube",
        BlockState("minecraft:stone"): "cube",
        BlockState("minecraft:oak_planks"): "log",
    }

    chosen = _choose_state_with_local_family_support(
        (0, 0, 0),
        candidate_scores,
        {},
        palette_families,
    )

    assert chosen == BlockState("minecraft:dirt")


def test_patch_descriptor_supports_flattened_face_samples():
    descriptor = _patch_descriptor(np.full((9, 3), 125, dtype=np.float32))

    assert len(descriptor) == 7
    assert all(np.isfinite(value) for value in descriptor)


def test_rectify_projected_face_returns_square_patch():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (20, 40, 60)
    corners = np.asarray([[8, 8], [23, 8], [23, 23], [8, 23]], dtype=np.float32)

    patch = _rectify_projected_face(image, corners)

    assert patch is not None
    assert patch.shape == (16, 16, 3)
    assert tuple(patch[8, 8]) == (20, 40, 60)
