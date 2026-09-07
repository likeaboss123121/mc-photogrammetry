import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.lighting import (
    LightResponse,
    blocks_light,
    brightness,
    emission_of,
    fit_response,
    light_field,
    propagate,
)


def test_emission_matches_vanilla_levels():
    assert emission_of(BlockState("minecraft:glowstone")) == 15
    assert emission_of(BlockState("minecraft:torch")) == 14
    assert emission_of(BlockState("minecraft:soul_torch")) == 10
    assert emission_of(BlockState("minecraft:magma_block")) == 3
    assert emission_of(BlockState("minecraft:dark_oak_planks")) == 0


def test_copper_bulbs_dim_as_they_oxidise_and_only_while_lit():
    lit = (("lit", "true"),)
    assert emission_of(BlockState("minecraft:waxed_copper_bulb", lit)) == 15
    assert emission_of(BlockState("minecraft:waxed_exposed_copper_bulb", lit)) == 12
    assert emission_of(BlockState("minecraft:waxed_weathered_copper_bulb", lit)) == 8
    assert emission_of(BlockState("minecraft:waxed_oxidized_copper_bulb", lit)) == 4
    assert emission_of(BlockState("minecraft:waxed_copper_bulb", (("lit", "false"),))) == 0


def test_soul_torch_beats_the_shorter_torch_key():
    # Longest-token matching, or "soul_torch" would pick up plain torch's 14.
    assert emission_of(BlockState("minecraft:soul_torch")) < emission_of(BlockState("minecraft:torch"))


def test_light_sources_that_are_not_solid_do_not_block_light():
    assert not blocks_light(BlockState("minecraft:torch"))
    assert not blocks_light(BlockState("minecraft:glass"))
    assert blocks_light(BlockState("minecraft:glowstone"))
    assert blocks_light(BlockState("minecraft:stone"))


def test_brightness_follows_the_vanilla_curve():
    assert brightness(15) == pytest.approx(1.0)
    assert brightness(0) == pytest.approx(0.0)
    assert brightness(14) == pytest.approx(0.7778, abs=1e-3)
    assert brightness(10) == pytest.approx(0.3333, abs=1e-3)
    # Strictly increasing, so a brighter cell is never scored darker.
    levels = [brightness(level) for level in range(16)]
    assert all(b > a for a, b in zip(levels, levels[1:]))


def test_propagation_loses_one_level_per_block():
    levels = propagate((9, 1, 1), {(4, 0, 0): 15}, set())
    assert levels[4, 0, 0] == 15
    assert levels[5, 0, 0] == 14
    assert levels[8, 0, 0] == 11
    assert levels[0, 0, 0] == 11


def test_opaque_blocks_cast_a_shadow():
    # A wall at x=5 stops the fill; everything past it stays dark.
    levels = propagate((9, 1, 1), {(0, 0, 0): 15}, {(5, 0, 0)})
    assert levels[4, 0, 0] == 11
    assert levels[5, 0, 0] == 0
    assert levels[8, 0, 0] == 0


def test_light_field_reads_the_cell_a_face_points_into():
    blocks = {(1, 0, 0): BlockState("minecraft:stone"), (3, 0, 0): BlockState("minecraft:glowstone")}
    field = light_field((5, 1, 1), blocks)
    # The stone's east face looks at (2,0,0), one step from the glowstone.
    assert field.face_level((1, 0, 0), "east") == 14
    # Its west face looks away into darkness.
    assert field.face_level((1, 0, 0), "west") == 0


def test_a_scene_with_no_light_source_leaves_brightness_alone():
    blocks = {(0, 0, 0): BlockState("minecraft:stone")}
    field = light_field((3, 3, 3), blocks)
    assert int(field.levels.max()) == 0
    # The default response must not darken anything, or an unlit scene would
    # drive every expected colour to black.
    assert field.face_factor((0, 0, 0), "up") == pytest.approx(1.0)


def test_response_falls_back_to_a_flat_gain_without_level_spread():
    response = fit_response([(0, 0.6)] * 30)
    assert response.factor(0) == pytest.approx(0.6)
    assert response.factor(15) == pytest.approx(0.6)


def test_response_recovers_a_known_curve():
    measurements = [(level, LightResponse(floor=0.1, scale=0.9).factor(level)) for level in range(16)]
    response = fit_response(measurements * 2)
    assert response.floor == pytest.approx(0.1, abs=0.02)
    assert response.scale == pytest.approx(0.9, abs=0.02)


def test_extra_emitters_light_a_cell_the_blocks_do_not():
    blocks = {(0, 0, 0): BlockState("minecraft:stone")}
    field = light_field((5, 1, 1), blocks, extra_emitters={(2, 0, 0): 14})
    assert field.level_at((2, 0, 0)) == 14
    assert field.level_at((3, 0, 0)) == 13
