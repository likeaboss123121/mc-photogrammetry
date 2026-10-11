"""Neutral appearance scoring: records, compatibility surface and typed face scoring."""

from __future__ import annotations

import ast
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks import appearance as blocks_appearance
from mc_photogrammetry.blocks.appearance import (
    DEFAULT_PALETTE,
    PaletteColor,
    palette_face_colors,
    slot_coverage,
)
from mc_photogrammetry.blocks.model_atlas import FACE_SHADE, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction import appearance_scoring, facade, raycast
from mc_photogrammetry.reconstruction.appearance_scoring import (
    AppearanceScorer,
    PaletteIndex,
    observation_costs,
    palette_face_textures,
    score_face_observation,
)
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame-0000"


def _cube(name: str, rgb: tuple[int, int, int], family: str = "cube") -> PaletteColor:
    return PaletteColor(
        BlockState(f"minecraft:{name}"),
        rgb,
        face_rgbs=(rgb,),
        family=family,
        face_map=tuple((slot, rgb) for slot in FACE_SLOTS),
        full_cube=True,
    )


def _slab(name: str, rgb: tuple[int, int, int]) -> PaletteColor:
    return PaletteColor(
        BlockState(f"minecraft:{name}"),
        rgb,
        face_rgbs=(rgb,),
        face_map=tuple((slot, rgb) for slot in FACE_SLOTS),
        full_cube=False,
        shape_boxes=(((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),),
    )


def _observation(patch: np.ndarray, **overrides: object) -> FaceObservation:
    slot = str(overrides.get("slot", "north"))
    fields: dict[str, object] = {
        "cell": (0, 0, 0),
        "slot": slot,
        "frame_id": FRAME,
        "normal": NORMAL_OF_SLOT[slot],
        "patch": patch,
        "usable_coverage": 1.0,
        "support_frames": (FRAME,),
    }
    fields.update(overrides)
    return FaceObservation(**fields)  # type: ignore[arg-type]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


# --- records and compatibility surface -------------------------------------------------


def test_palette_color_fields_defaults_and_equality_are_preserved():
    record = PaletteColor(BlockState("minecraft:stone"), (125, 125, 125))
    assert record.face_rgbs == ()
    assert record.family == "other"
    assert record.descriptors == ()
    assert record.face_map == ()
    assert record.face_thumbnails == ()
    assert record.face_corners == ()
    assert record.model_faces == ()
    assert record.full_cube is True
    assert record.shape_boxes == ()
    assert record.covered_slots == ()
    assert record.occupies == 1.0
    assert record == PaletteColor(BlockState("minecraft:stone"), (125, 125, 125))


def test_default_palette_is_state_keyed_and_unchanged():
    assert [item.state for item in DEFAULT_PALETTE] == [
        BlockState("minecraft:stone"),
        BlockState("minecraft:dirt"),
        BlockState("minecraft:oak_planks"),
        BlockState("minecraft:grass_block"),
        BlockState("minecraft:bricks"),
        BlockState("minecraft:glass"),
        BlockState("minecraft:white_concrete"),
        BlockState("minecraft:black_concrete"),
    ]
    assert DEFAULT_PALETTE[0].rgb == (125, 125, 125)


def test_legacy_import_paths_still_resolve_to_the_neutral_records():
    assert facade.PaletteColor is blocks_appearance.PaletteColor
    assert facade.DEFAULT_PALETTE == blocks_appearance.DEFAULT_PALETTE
    assert raycast.PaletteColor is blocks_appearance.PaletteColor
    assert raycast.DEFAULT_PALETTE == blocks_appearance.DEFAULT_PALETTE
    assert raycast.slot_coverage is blocks_appearance.slot_coverage
    assert raycast.PaletteIndex is PaletteIndex
    assert raycast._palette_candidates is blocks_appearance.palette_candidates
    assert raycast._patch_descriptor is blocks_appearance.patch_descriptor
    assert raycast._palette_face_colors is blocks_appearance.palette_face_colors
    assert raycast._palette_face_textures is palette_face_textures
    assert raycast._observation_costs is observation_costs


def test_new_scorer_does_not_import_facade_or_raycast():
    for relative in (
        "src/mc_photogrammetry/blocks/appearance.py",
        "src/mc_photogrammetry/reconstruction/appearance_scoring.py",
        "src/mc_photogrammetry/reconstruction/face_observation.py",
    ):
        modules = _imported_modules(ROOT / relative)
        assert not any(
            module.endswith("facade") or module.endswith("raycast") for module in modules
        ), (relative, sorted(modules))
    # The neutral record module must not reach back into the reconstruction package.
    modules = _imported_modules(ROOT / "src/mc_photogrammetry/blocks/appearance.py")
    assert not any("reconstruction" in module for module in modules), sorted(modules)


# --- scoring primitives -----------------------------------------------------------------


def test_palette_face_colors_falls_back_when_face_metadata_is_missing():
    item = PaletteColor(BlockState("minecraft:stone"), (100, 100, 100), face_rgbs=((90, 90, 90),))
    colors = palette_face_colors((item,))
    assert set(colors[item.state]) == set(FACE_SLOTS)
    for slot in FACE_SLOTS:
        expected = np.asarray((90, 90, 90), dtype=np.float32) * FACE_SHADE[slot]
        assert np.allclose(colors[item.state][slot], expected)

    bare = PaletteColor(BlockState("minecraft:dirt"), (50, 60, 70))
    colors = palette_face_colors((bare,))
    assert np.allclose(colors[bare.state]["up"], (50, 60, 70))


def test_partial_shape_slot_coverage_is_preserved():
    slab = _slab("oak_slab", (120, 120, 120))
    assert slot_coverage(slab, "up") == pytest.approx(1.0)
    assert slot_coverage(slab, "north") == pytest.approx(0.5)
    index = PaletteIndex((slab,), palette_face_colors((slab,)), None, 0.0)
    assert index.coverage["up"][0] == pytest.approx(1.0)
    assert index.coverage["north"][0] == pytest.approx(0.5)


def test_costs_are_relative_capped_and_keyed_by_state():
    stone, dirt = _cube("stone", (125, 125, 125)), _cube("dirt", (134, 96, 67))
    index = PaletteIndex((stone, dirt), palette_face_colors((stone, dirt)), None, 0.0)
    costs, best = index.costs(
        np.asarray([125.0, 125.0, 125.0]), (), "north", outlier_ceiling=24.0
    )
    assert index.states[int(np.argmin(costs))] == stone.state
    assert float(costs.min()) == pytest.approx(0.0)
    assert float(costs.max()) <= 24.0 + 1e-9

    far, _best = index.costs(
        np.asarray([255.0, 255.0, 255.0]), (), "north", outlier_ceiling=24.0
    )
    assert float(far.max()) == pytest.approx(24.0)


def test_observation_costs_returns_every_state_and_abstains_when_empty():
    stone = _cube("stone", (125, 125, 125))
    face_colors = palette_face_colors((stone,))
    costs, best = observation_costs(
        np.asarray([125.0, 125.0, 125.0]), (), "north", (stone,), face_colors, None, None,
        True, outlier_ceiling=24.0,
    )
    assert set(costs) == {stone.state}
    assert costs[stone.state] == pytest.approx(0.0)
    assert best == pytest.approx(
        blocks_appearance.appearance_distance(
            np.asarray([125.0, 125.0, 125.0]), face_colors[stone.state]["north"]
        )
    )

    empty, empty_best = observation_costs(
        np.asarray([125.0, 125.0, 125.0]), (), "north", (), {}, None, None, True,
        outlier_ceiling=24.0,
    )
    assert empty == {}
    assert empty_best == float("inf")


def test_palette_face_textures_reads_normalised_slot_patterns():
    Image = pytest.importorskip("PIL.Image")

    def thumbnail(invert: bool) -> str:
        pixels = np.zeros((16, 16), dtype=np.uint8)
        pixels[::2, :] = 0 if invert else 255
        pixels[:, ::2] = 255 if invert else 0
        rgb = np.stack((pixels, pixels, pixels), axis=-1)
        buffer = BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    items = tuple(
        PaletteColor(
            BlockState(f"minecraft:pattern_{index}"),
            (120, 120, 120),
            face_rgbs=((120, 120, 120),),
            face_map=tuple((slot, (120, 120, 120)) for slot in FACE_SLOTS),
            face_thumbnails=(("north", thumbnail(bool(index))),),
        )
        for index in range(2)
    )
    textures = palette_face_textures(items)
    assert set(textures["north"][0]) == {items[0].state, items[1].state}
    matrix = textures["north"][1]
    assert matrix.shape == (2, 256)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    index = PaletteIndex(items, palette_face_colors(items), None, texture_weight=1.0)
    aligned, mask = index.textures["north"]
    assert mask.tolist() == [True, True]
    assert aligned.shape == (2, 256)


# --- typed face scoring -----------------------------------------------------------------


def test_usable_observation_scores_the_right_material():
    stone, dirt = _cube("stone", (125, 125, 125)), _cube("dirt", (134, 96, 67))
    scorer = AppearanceScorer((stone, dirt))
    grey = _observation(np.full((4, 4, 3), 125.0, dtype=np.float32))
    score = scorer.score(grey)
    assert not score.abstained
    assert score.ranked()[0][0] == stone.state
    assert score.cost_for(stone.state) == pytest.approx(0.0)
    assert score.cost_for(BlockState("minecraft:absent")) is None


def test_zero_coverage_and_invalid_observations_abstain_typed():
    stone = _cube("stone", (125, 125, 125))
    scorer = AppearanceScorer((stone,))
    patch = np.full((4, 4, 3), 125.0, dtype=np.float32)

    zero = scorer.score(_observation(patch, usable_coverage=0.0))
    assert zero.abstained and zero.reason == "zero_coverage" and zero.costs == ()

    invalid = scorer.score(_observation(patch, valid=np.zeros((4, 4), dtype=bool)))
    assert invalid.abstained and invalid.reason == "no_valid_pixels"

    class _EmptyIndex:
        count = 0

    empty = score_face_observation(_observation(patch), _EmptyIndex())  # type: ignore[arg-type]
    assert empty.abstained and empty.reason == "empty_palette"


def test_zero_effective_coverage_abstains_even_with_positive_usable_coverage():
    stone = _cube("stone", (125, 125, 125))
    scorer = AppearanceScorer((stone,))
    patch = np.full((4, 4, 3), 125.0, dtype=np.float32)

    zero_coverage = scorer.score(_observation(patch, coverage=np.zeros((4, 4))))
    assert zero_coverage.abstained
    assert zero_coverage.reason == "zero_coverage"
    assert zero_coverage.costs == ()

    disjoint = scorer.score(
        _observation(
            patch,
            valid=np.asarray([[True, True, False, False]] * 4),
            coverage=np.asarray([[0.0, 0.0, 1.0, 1.0]] * 4),
        )
    )
    assert disjoint.abstained
    assert disjoint.reason == "zero_coverage"


def test_excluded_pixels_cannot_affect_a_face_score():
    stone, dirt = _cube("stone", (125, 125, 125)), _cube("dirt", (134, 96, 67))
    scorer = AppearanceScorer((stone, dirt))
    valid = np.asarray([[True, True], [True, False]])
    coverage = np.asarray([[1.0, 1.0], [1.0, 0.0]])
    clean = np.full((2, 2, 3), 125.0, dtype=np.float32)
    dirty = clean.copy()
    dirty[~valid] = (0.0, 255.0, 0.0)

    a = scorer.score(_observation(clean, valid=valid, coverage=coverage))
    b = scorer.score(_observation(dirty, valid=valid, coverage=coverage))
    assert a == b
    assert a.ranked()[0][0] == stone.state


def test_coverage_zero_pixels_cannot_affect_a_face_score():
    stone, dirt = _cube("stone", (125, 125, 125)), _cube("dirt", (134, 96, 67))
    scorer = AppearanceScorer((stone, dirt))
    coverage = np.asarray([[1.0, 1.0], [1.0, 0.0]])
    clean = np.full((2, 2, 3), 125.0, dtype=np.float32)
    dirty = clean.copy()
    dirty[1, 1] = (0.0, 255.0, 0.0)

    a = scorer.score(_observation(clean, coverage=coverage))
    b = scorer.score(_observation(dirty, coverage=coverage))
    assert not a.abstained
    assert a == b
    assert a.ranked()[0][0] == stone.state


def test_ranked_ties_are_deterministic_regardless_of_palette_order():
    first = _cube("aaa", (125, 125, 125))
    second = _cube("bbb", (125, 125, 125))
    patch = np.full((2, 2, 3), 125.0, dtype=np.float32)

    forward = AppearanceScorer((first, second)).score(_observation(patch))
    reverse = AppearanceScorer((second, first)).score(_observation(patch))
    assert forward.ranked() == reverse.ranked()
    assert forward.ranked()[0][0] != forward.ranked()[1][0]
    assert forward.ranked()[0][0] == reverse.ranked()[0][0] == first.state


def test_texture_weight_changes_the_ranking_but_flat_metadata_is_neutral():
    stone, dirt = _cube("stone", (125, 125, 125)), _cube("dirt", (134, 96, 67))
    index = PaletteIndex((stone, dirt), palette_face_colors((stone, dirt)), None, 0.0)
    # No thumbnails at all: the texture term must be exactly neutral.
    without = score_face_observation(
        _observation(np.full((4, 4, 3), 125.0, dtype=np.float32)),
        index,
        texture_weight=40.0,
        use_pattern=True,
    )
    assert without.cost_for(stone.state) == pytest.approx(0.0)
