"""Multipart blockstates: fences, walls, panes and bars.

These blocks are not described as a list of variants but as a set of pieces with
conditions - a post always, a side for each neighbour that connects - so the
variant reader sees no `variants` key and returns nothing. That is why
`dark_oak_fence` was absent from the palette entirely.
"""
import json

from mc_photogrammetry.blocks.model_atlas import (
    ModelGeometry,
    blockstate_multipart,
    blockstate_variants,
    compose,
)

FENCE = {
    "multipart": [
        {"apply": {"model": "block/fence_post"}},
        {"apply": {"model": "block/fence_side"}, "when": {"north": "true"}},
        {"apply": {"model": "block/fence_side", "y": 90}, "when": {"east": "true"}},
    ]
}

WALL = {
    "multipart": [
        {"apply": {"model": "block/wall_post"}, "when": {"up": "true"}},
        {"apply": {"model": "block/wall_side"}, "when": {"north": "low"}},
        {"apply": {"model": "block/wall_tall"}, "when": {"north": "tall"}},
    ]
}

EITHER = {
    "multipart": [
        {"apply": {"model": "block/either"},
         "when": {"OR": [{"north": "true"}, {"south": "true"}]}},
    ]
}


def _properties(entries):
    return {tuple(sorted(properties)) for properties, _parts in entries}


def test_a_variants_reader_finds_nothing_in_a_multipart_blockstate():
    assert blockstate_variants(FENCE) == []


def test_both_sides_of_a_boolean_are_enumerated():
    # A condition only ever names the value that switches a piece on, so the off
    # value is implied by its absence. Reading the domain literally leaves the
    # single combination where every side connects.
    entries = blockstate_multipart(FENCE)
    assert len(entries) == 4
    values = {dict(properties)["north"] for properties, _parts in entries}
    assert values == {"true", "false"}


def test_a_piece_applies_only_when_its_condition_holds():
    entries = dict(
        (tuple(sorted(properties)), parts) for properties, parts in blockstate_multipart(FENCE)
    )
    bare = entries[(("east", "false"), ("north", "false"))]
    assert [model for model, _x, _y in bare] == ["block/fence_post"]
    both = entries[(("east", "true"), ("north", "true"))]
    assert [model for model, _x, _y in both].count("block/fence_side") == 2


def test_a_rotated_piece_keeps_its_rotation():
    entries = dict(
        (tuple(sorted(properties)), parts) for properties, parts in blockstate_multipart(FENCE)
    )
    east = entries[(("east", "true"), ("north", "false"))]
    assert ("block/fence_side", 0.0, 90.0) in east


def test_a_wall_gains_the_disconnected_value():
    # Walls speak of "low" and "tall" and never of "none", which is the state of a
    # side that joins nothing.
    entries = blockstate_multipart(WALL)
    values = {dict(properties).get("north") for properties, _parts in entries}
    assert values == {"low", "tall", "none"}


def test_an_or_condition_accepts_either_alternative():
    entries = dict(
        (tuple(sorted(properties)), parts) for properties, parts in blockstate_multipart(EITHER)
    )
    assert entries[(("north", "true"), ("south", "false"))]
    assert entries[(("north", "false"), ("south", "true"))]
    assert (("north", "false"), ("south", "false")) not in entries


def test_composing_pieces_unions_their_shape():
    post = ModelGeometry((), (0.4, 0.0, 0.4), (0.6, 1.0, 0.6),
                         (((0.4, 0.0, 0.4), (0.6, 1.0, 0.6)),))
    side = ModelGeometry((), (0.4, 0.4, 0.0), (0.6, 0.9, 0.4),
                         (((0.4, 0.4, 0.0), (0.6, 0.9, 0.4)),))
    joined = compose([post, side])
    assert joined is not None
    assert len(joined.boxes) == 2
    assert joined.minimum == (0.4, 0.0, 0.0)
    assert joined.maximum == (0.6, 1.0, 0.6)
    # A fence is thin, so it hides nothing behind it whatever it joins.
    assert not joined.full_cube
    assert not any(joined.covers(slot) for slot in ("down", "up", "north", "south"))


def test_composing_nothing_yields_nothing():
    assert compose([]) is None
    assert compose([None]) is None


def test_the_combination_count_is_capped():
    many = {"multipart": [
        {"apply": {"model": "block/x"}, "when": {name: "true"}}
        for name in ("a", "b", "c", "d", "e", "f", "g", "h")
    ]}
    assert len(blockstate_multipart(many, limit=16)) <= 16
