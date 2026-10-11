"""The residual pass must use the same usable-pixel authority as the shape fit.

DS-3 R1.  ``face_residuals`` and ``residual_proposals`` accept an optional,
evaluator-owned ``usable_masks`` mapping.  ``None`` keeps the historical unmasked
behaviour; a supplied mapping is authoritative: a frame missing from it is
unusable, a wrong-shaped mask raises, and at render scale only pixels whose whole
source footprint is usable may reach the resized photo, the frame gain, the
typical residual, a face's score or a proposal.

The controls here are synthetic and analytic.  A photograph built from the
renderer's own canvas agrees with it exactly, so any disagreement that shows up
is one the test put there, not one copied out of the implementation.
"""
import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import (
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_result,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.residual import face_residuals, residual_proposals
from mc_photogrammetry.reconstruction.scene import SceneTransform

SLOTS = ("up", "down", "north", "south", "east", "west")
CARPET = (((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0)),)
PHOTO_HEIGHT = 100
PHOTO_WIDTH = 100


def _entry(name, rgb, boxes=()):
    return PaletteColor(
        BlockState(f"minecraft:{name}", ()), rgb,
        face_map=tuple((slot, rgb) for slot in SLOTS),
        full_cube=not boxes, shape_boxes=boxes,
    )


def _palette():
    return (
        _entry("stone", (120, 120, 120)),
        _entry("white_carpet", (245, 245, 245), CARPET),
    )


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(6, 6, 6))


def _camera():
    return CameraModel(position=(2.5, 4.0, -4.0), yaw_degrees=0.0,
                       pitch_degrees=40.0, fov_degrees=90.0)


def _floor():
    return {(x, 1, z): BlockState("minecraft:stone") for x in range(6) for z in range(6)}


def _render(blocks, palette, scale):
    """The same renderer call the residual pass makes, exposed for control setup."""
    colours, shapes = face_colours(palette), face_shapes(palette)
    rendered, index = render_result(
        blocks, _scene(), _camera(), PHOTO_WIDTH, PHOTO_HEIGHT, colours, shapes, scale,
        ambient=True, occluding_slots=face_occluding_slots(palette),
        model_faces=face_model_faces(palette),
    )
    return rendered, index


def _photo_agreeing_with(rendered):
    """A uint8 frame that reproduces the render exactly (its colours are integral)."""
    return np.clip(np.rint(rendered.colour), 0, 255).astype(np.uint8)


def _random_photo(seed):
    return np.random.default_rng(seed).integers(0, 256, (PHOTO_HEIGHT, PHOTO_WIDTH, 3),
                                                dtype=np.uint8)


def _run(blocks, frames, palette, **kwargs):
    return face_residuals(blocks, _scene(), {name: _camera() for name in frames},
                          frames, palette, **kwargs)


def test_none_keeps_the_historical_residuals_and_proposals():
    palette, floor = _palette(), _floor()
    photo = _random_photo(3)
    historical = _run(floor, {"a.png": photo}, palette, scale=1.0)
    explicit = _run(floor, {"a.png": photo}, palette, scale=1.0, usable_masks=None)
    assert explicit == historical
    assert historical, "the fixture must produce faces"

    proposals = residual_proposals(floor, _scene(), {"a.png": _camera()}, {"a.png": photo},
                                   palette, scale=1.0, threshold=0.0)
    masked_none = residual_proposals(floor, _scene(), {"a.png": _camera()}, {"a.png": photo},
                                     palette, scale=1.0, threshold=0.0, usable_masks=None)
    assert masked_none == proposals
    assert proposals


def test_a_mask_missing_a_frame_makes_that_frame_unusable():
    palette, floor = _palette(), _floor()
    quiet, loud = _random_photo(4), np.full((PHOTO_HEIGHT, PHOTO_WIDTH, 3), 255, np.uint8)
    mask = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)

    alone = _run(floor, {"a.png": quiet}, palette, scale=1.0,
                 usable_masks={"a.png": mask})
    with_loud = _run(floor, {"a.png": quiet, "b.png": loud}, palette, scale=1.0,
                     usable_masks={"a.png": mask})
    assert with_loud == alone
    assert alone

    # Without the mapping the second frame is evidence, so the control is not vacuous.
    unmasked = _run(floor, {"a.png": quiet, "b.png": loud}, palette, scale=1.0)
    assert unmasked != alone


def test_a_wrong_shaped_mask_is_rejected():
    palette, floor = _palette(), _floor()
    photo = _random_photo(5)
    wrong = np.ones((PHOTO_HEIGHT + 1, PHOTO_WIDTH), dtype=bool)
    with pytest.raises(ValueError):
        _run(floor, {"a.png": photo}, palette, scale=1.0, usable_masks={"a.png": wrong})
    with pytest.raises(ValueError):
        residual_proposals(floor, _scene(), {"a.png": _camera()}, {"a.png": photo}, palette,
                           scale=1.0, usable_masks={"a.png": wrong})


def test_excluded_pixel_colours_cannot_reach_residuals_or_proposals():
    palette, floor = _palette(), _floor()
    rendered, _index = _render(floor, palette, 1.0)
    base = _photo_agreeing_with(rendered)
    seen = rendered.ownership >= 0
    rows, columns = np.nonzero(seen)
    # The left half of the owned footprint, which therefore contains owned pixels.
    left = int(columns.min())
    right = left + max(1, (int(columns.max()) - left + 1) // 2)
    top, bottom = int(rows.min()), int(rows.max()) + 1

    low, high = base.copy(), base.copy()
    low[top:bottom, left:right] = 0
    high[top:bottom, left:right] = 255
    mask = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)
    mask[top:bottom, left:right] = False

    masked_low = _run(floor, {"a.png": low}, palette, scale=1.0, usable_masks={"a.png": mask})
    masked_high = _run(floor, {"a.png": high}, palette, scale=1.0, usable_masks={"a.png": mask})
    assert masked_low == masked_high
    assert masked_low, "the usable half must still score faces"
    # The photo agrees with the render wherever it is usable, so nothing is doubted.
    assert set(masked_low.values()) == {0.0}
    assert residual_proposals(
        floor, _scene(), {"a.png": _camera()}, {"a.png": low}, palette, scale=1.0,
        usable_masks={"a.png": mask}) == {}
    assert residual_proposals(
        floor, _scene(), {"a.png": _camera()}, {"a.png": high}, palette, scale=1.0,
        usable_masks={"a.png": mask}) == {}

    # Unmasked, the same excluded rectangle does change the scores and proposals.
    unmasked_low = _run(floor, {"a.png": low}, palette, scale=1.0)
    unmasked_high = _run(floor, {"a.png": high}, palette, scale=1.0)
    assert unmasked_low != unmasked_high


def test_a_partly_usable_render_pixel_is_not_evidence():
    """A render pixel needs its whole source footprint, not most of it.

    At scale 0.5 each render pixel is exactly one 2x2 source block.  Leaving one
    of the four source pixels usable and leaving none must score identically:
    either way that render pixel carries no evidence.  An implementation that
    accepted partial coverage would let the two masks disagree.
    """
    palette, floor = _palette(), _floor()
    rendered, _index = _render(floor, palette, 0.5)
    photo = _random_photo(6)
    seen = rendered.ownership >= 0
    rows, columns = np.nonzero(seen)
    row, column = int(rows[len(rows) // 2]), int(columns[len(columns) // 2])
    top, left = row - row % 2, column - column % 2
    block = (slice(top, top + 2), slice(left, left + 2))

    partly = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)
    partly[top, left] = False
    wholly = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)
    wholly[block] = False

    partly_scored = _run(floor, {"a.png": photo}, palette, scale=0.5,
                         usable_masks={"a.png": partly})
    wholly_scored = _run(floor, {"a.png": photo}, palette, scale=0.5,
                         usable_masks={"a.png": wholly})
    assert wholly_scored == partly_scored
    assert wholly_scored, "the fixture must leave other faces scoreable"


def test_minimum_pixel_counts_use_only_usable_pixels():
    palette, floor = _palette(), _floor()
    rendered, index = _render(floor, palette, 1.0)
    photo = _random_photo(7)
    seen = rendered.ownership >= 0
    ids, counts = np.unique(rendered.ownership[seen], return_counts=True)
    richest = int(ids[int(np.argmax(counts))])
    face = index[richest]
    owned_rows, owned_columns = np.nonzero(seen & (rendered.ownership == richest))
    total = int(owned_rows.size)
    assert total > 1

    every_pixel = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)
    with_all = _run(floor, {"a.png": photo}, palette, scale=1.0, minimum_pixels=total,
                    usable_masks={"a.png": every_pixel})
    assert face in with_all

    # Keep only half of that face's pixels usable; the same minimum must now reject it.
    half = np.ones((PHOTO_HEIGHT, PHOTO_WIDTH), dtype=bool)
    half[owned_rows[total // 2:], owned_columns[total // 2:]] = False
    with_half = _run(floor, {"a.png": photo}, palette, scale=1.0, minimum_pixels=total,
                     usable_masks={"a.png": half})
    assert face not in with_half
    assert _run(floor, {"a.png": photo}, palette, scale=1.0, minimum_pixels=1,
                usable_masks={"a.png": half}), "the remaining pixels still carry faces"


def test_reconstruct_scene_surfaces_forwards_usable_masks(monkeypatch):
    """The call site, not just the API, must hand the masks to the residual pass."""
    import mc_photogrammetry.reconstruction.raycast as raycast
    import mc_photogrammetry.reconstruction.residual as residual

    captured = {}

    def spy(blocks, scene, cameras, frames, palette, **kwargs):
        captured["usable_masks"] = kwargs.get("usable_masks")
        return {}

    monkeypatch.setattr(residual, "residual_proposals", spy)
    palette = _palette()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    raycast.reconstruct_scene_surfaces(
        {"a.png": image}, {"a.png": _camera()}, _scene(), palette=palette,
        propose_colour_gap=0.0, adjacency_strength=0.0, simulate_lighting=False,
        correct_light_cast=False, reconcile_exposure=False, self_calibrate=False,
        surface_only=False, fit_partial_shapes=True, usable_masks={"a.png": mask},
    )
    assert captured["usable_masks"] is not None
    assert captured["usable_masks"]["a.png"] is mask
