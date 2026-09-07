"""The classifier's vocabulary must contain every block, partial shapes included.

The palette was built from full cubes only, so a reconstruction contained exactly
zero stairs, slabs, carpets, fences and torches - not because the classifier judged
them badly but because it had no way to say them. Template 2's reference is 8.1%
partial blocks, which made its true ceiling 91.9% and nothing in the classifier
could have reached it.

Shape has to be carried and it has to cost something. A stair is cut from the same
texture as the planks it sits on, so on colour alone the two are identical, and the
palette holds twenty-four stair orientations against one entry for planks.

The cost is the shape's **projected footprint** along the face's axis - how much of
that face it hides. Measuring wall contact instead was the first attempt and it is
wrong where it matters: a carpet's top surface sits at y=0.0625 and never touches
the ceiling of its cell, so it scored zero on the very face a camera sees it
through, took the maximum penalty, and could never win a cell.
"""
import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.raycast import PaletteIndex, slot_coverage

FULL = (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)
BOTTOM_SLAB = (((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),)
CARPET = (((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0)),)
FENCE_POST = (((0.375, 0.0, 0.375), (0.625, 1.0, 0.625)),)


def _item(name, boxes=(), full_cube=False, rgb=(120, 120, 120)):
    return PaletteColor(
        BlockState(f"minecraft:{name}"), rgb,
        face_rgbs=(rgb,),
        face_map=tuple((slot, rgb) for slot in
                       ("up", "down", "north", "south", "east", "west")),
        full_cube=full_cube, shape_boxes=boxes,
    )


def test_a_full_cube_fills_every_face():
    cube = _item("stone", FULL, full_cube=True)
    for slot in ("up", "down", "north", "south", "east", "west"):
        assert slot_coverage(cube, slot) == 1.0


def test_a_bottom_slab_hides_a_whole_floor_and_half_a_wall():
    # Seen from above or below it covers the entire cross section; edge on, half.
    slab = _item("oak_slab", BOTTOM_SLAB)
    assert slot_coverage(slab, "down") == pytest.approx(1.0)
    assert slot_coverage(slab, "up") == pytest.approx(1.0)
    assert slot_coverage(slab, "north") == pytest.approx(0.5)
    assert slot_coverage(slab, "east") == pytest.approx(0.5)


def test_a_carpet_covers_the_floor_it_lies_on_and_almost_nothing_edge_on():
    # The case the wall-contact measure got wrong: a carpet is what you see when
    # you look down at a carpeted floor, so its `up` coverage must be one.
    carpet = _item("white_carpet", CARPET)
    assert slot_coverage(carpet, "up") == pytest.approx(1.0)
    assert slot_coverage(carpet, "down") == pytest.approx(1.0)
    assert slot_coverage(carpet, "north") == pytest.approx(0.0625)


def test_a_fence_post_is_thin_from_above_and_narrow_from_the_side():
    fence = _item("oak_fence", FENCE_POST)
    assert slot_coverage(fence, "up") == pytest.approx(0.0625)
    assert slot_coverage(fence, "north") == pytest.approx(0.25)


def test_disjoint_boxes_on_one_face_add_up():
    pane = _item("glass_pane", (
        ((0.0, 0.0, 0.0), (0.25, 1.0, 1.0)),
        ((0.75, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ))
    assert slot_coverage(pane, "down") == pytest.approx(0.5)


def test_overlapping_boxes_are_not_counted_twice():
    # A model may describe one face with boxes that meet or overlap; the covered
    # area is their union, and double counting would push coverage above one.
    doubled = _item("weird", (
        ((0.0, 0.0, 0.0), (0.6, 1.0, 1.0)),
        ((0.4, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ))
    assert slot_coverage(doubled, "down") == pytest.approx(1.0)


def test_a_shape_touching_no_wall_still_hides_its_own_cross_section():
    # It touches nothing, but a camera looking along any axis still sees it.
    floating = _item("floating", (((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)),))
    for slot in ("up", "down", "north", "south", "east", "west"):
        assert slot_coverage(floating, slot) == pytest.approx(0.25)


def test_coverage_depends_on_the_axis_not_the_direction():
    # A shape hides the same cross section whichever way you look along it.
    slab = _item("oak_slab", BOTTOM_SLAB)
    for near, far in (("up", "down"), ("north", "south"), ("east", "west")):
        assert slot_coverage(slab, near) == pytest.approx(slot_coverage(slab, far))


def _index(items):
    return PaletteIndex(tuple(items), {
        item.state: {slot: np.asarray(colour, dtype=np.float64)
                     for slot, colour in item.face_map}
        for item in items
    }, {}, 0.0)


def test_shape_cost_prefers_the_block_that_fills_the_face():
    # Identical colour, so only shape separates them: edge on, the planks explain
    # the whole wall and the slab only half of it.
    planks = _item("oak_planks", FULL, full_cube=True)
    slab = _item("oak_slab", BOTTOM_SLAB)
    index = _index([planks, slab])
    costs, _best = index.costs(np.asarray([120.0, 120.0, 120.0]), (), "north",
                               shape_weight=6.0)
    assert index.states[int(np.argmin(costs))] == planks.state


def test_a_carpet_is_not_penalised_for_the_floor_it_covers():
    # The regression this measure exists to prevent: seen from above a carpet
    # hides as much as the plank floor under it, so it must pay nothing there and
    # stay a live candidate.
    planks = _item("oak_planks", FULL, full_cube=True)
    carpet = _item("white_carpet", CARPET)
    index = _index([planks, carpet])
    costs, _best = index.costs(np.asarray([120.0, 120.0, 120.0]), (), "up",
                               shape_weight=6.0)
    assert costs[0] == pytest.approx(costs[1])


def test_without_a_shape_cost_the_two_are_indistinguishable():
    planks = _item("oak_planks", FULL, full_cube=True)
    slab = _item("oak_slab", BOTTOM_SLAB)
    index = _index([planks, slab])
    costs, _best = index.costs(np.asarray([120.0, 120.0, 120.0]), (), "up", shape_weight=0.0)
    assert costs[0] == pytest.approx(costs[1])


def test_the_cost_does_not_punish_a_face_the_shape_does_fill():
    # A slab seen from below hides that face completely and must pay nothing,
    # otherwise the penalty is a blanket prior against partial blocks.
    planks = _item("oak_planks", FULL, full_cube=True)
    slab = _item("oak_slab", BOTTOM_SLAB)
    index = _index([planks, slab])
    costs, _best = index.costs(np.asarray([120.0, 120.0, 120.0]), (), "down", shape_weight=6.0)
    assert costs[0] == pytest.approx(costs[1])


def test_restricting_the_palette_keeps_coverage_aligned():
    items = [_item("oak_planks", FULL, full_cube=True), _item("oak_slab", BOTTOM_SLAB),
             _item("white_carpet", CARPET)]
    index = _index(items)
    kept = index.restricted(np.array([0, 2], dtype=np.intp))
    # Entry 0 is the full block, entry 2 the carpet: both hide a whole floor from
    # above, and edge on they part company.
    assert kept.coverage["up"][0] == pytest.approx(1.0)
    assert kept.coverage["up"][1] == pytest.approx(1.0)
    assert kept.coverage["north"][1] == pytest.approx(0.0625)
