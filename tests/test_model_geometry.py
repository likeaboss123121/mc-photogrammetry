import json
import zipfile

import numpy as np

from mc_photogrammetry.blocks.model_atlas import (
    FACE_BASIS,
    FACE_SLOTS,
    ElementFace,
    ModelAtlas,
    ModelGeometry,
    element_face_corners,
)

FULL = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
SLAB = ((0.0, 0.0, 0.0), (1.0, 0.5, 1.0))
CARPET = ((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0))
POST = ((0.375, 0.0, 0.375), (0.625, 1.0, 0.625))


def _geometry(*boxes):
    minimum = tuple(min(box[0][axis] for box in boxes) for axis in range(3))
    maximum = tuple(max(box[1][axis] for box in boxes) for axis in range(3))
    return ModelGeometry((), minimum, maximum, tuple(boxes))


def test_full_cube_face_corners_match_the_unit_cube_basis():
    for slot in FACE_SLOTS:
        normal, corners = element_face_corners(np.zeros(3), np.ones(3), slot)
        expected_normal, origin, edge_u, edge_v = (np.asarray(v, float) for v in FACE_BASIS[slot])
        assert np.allclose(normal, expected_normal)
        assert np.allclose(corners[0], origin)
        assert np.allclose(corners[1], origin + edge_u)
        assert np.allclose(corners[2], origin + edge_u + edge_v)
        assert np.allclose(corners[3], origin + edge_v)


def test_partial_face_corners_span_only_the_element():
    _normal, corners = element_face_corners(np.asarray(SLAB[0]), np.asarray(SLAB[1]), "north")
    # A slab's side reaches from the cell floor to halfway up, not to the ceiling.
    assert np.isclose(corners[:, 1].min(), -0.5)
    assert np.isclose(corners[:, 1].max(), 0.0)


def test_a_full_cube_covers_every_face():
    geometry = _geometry(FULL)
    assert geometry.full_cube
    assert all(geometry.covers(slot) for slot in FACE_SLOTS)


def test_a_slab_covers_the_face_it_rests_on_but_not_the_one_above():
    geometry = _geometry(SLAB)
    assert not geometry.full_cube
    assert geometry.covers("down")
    assert not geometry.covers("up")
    assert not geometry.covers("north")
    assert np.isclose(geometry.height, 0.5)


def test_a_carpet_is_thin_but_still_hides_the_face_beneath_it():
    geometry = _geometry(CARPET)
    # The block underneath a carpet is completely hidden even though the carpet
    # is a sixteenth of a block tall, which is why its upward face reads as the
    # carpet's material rather than the floor's.
    assert geometry.covers("down")
    assert not geometry.covers("up")
    assert geometry.height < 0.1


def test_a_fence_post_covers_nothing_because_it_touches_no_face_fully():
    geometry = _geometry(POST)
    assert not geometry.full_cube
    assert not any(geometry.covers(slot) for slot in FACE_SLOTS)


def test_a_staircase_bounding_box_fills_the_cell_but_the_shape_does_not():
    # Lower slab plus the upper half-depth step: the box spans the whole cell,
    # so judging coverage from bounds alone would wrongly hide the block above.
    geometry = _geometry(((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)), ((0.0, 0.5, 0.5), (1.0, 1.0, 1.0)))
    assert geometry.minimum == (0.0, 0.0, 0.0)
    assert geometry.maximum == (1.0, 1.0, 1.0)
    assert not geometry.full_cube
    assert geometry.covers("down")
    assert geometry.covers("south")
    assert not geometry.covers("up")
    assert not geometry.covers("north")


def _slab_jar(path):
    texture = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000001000000010080600000"
        "01ff3ff610000002249444154789c63fccfc0f09f8118a8c1c8c0c000"
        "00ffff03000a1b01e3a1ae1f4d0000000049454e44ae426082"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("assets/minecraft/textures/block/oak_planks.png", texture)
        archive.writestr(
            "assets/minecraft/models/block/oak_slab.json",
            json.dumps({
                "textures": {"side": "minecraft:block/oak_planks"},
                "elements": [{
                    "from": [0, 0, 0],
                    "to": [16, 8, 16],
                    "faces": {
                        slot: {"texture": "#side", "cullface": slot} for slot in FACE_SLOTS
                    },
                }],
            }),
        )


def test_geometry_reads_a_partial_element_from_a_jar(tmp_path):
    jar = tmp_path / "slab.jar"
    _slab_jar(jar)
    with zipfile.ZipFile(jar) as archive:
        geometry = ModelAtlas(archive).geometry("block/oak_slab")
    assert geometry is not None
    assert np.isclose(geometry.height, 0.5)
    assert geometry.covers("down") and not geometry.covers("up")
    assert {face.slot for face in geometry.faces} == set(FACE_SLOTS)


def test_variant_rotation_turns_which_face_is_covered(tmp_path):
    jar = tmp_path / "slab.jar"
    _slab_jar(jar)
    with zipfile.ZipFile(jar) as archive:
        atlas = ModelAtlas(archive)
        # Rotating a slab about X stands it on end, so the face it covers moves
        # from the cell floor to a side wall.
        upright = atlas.geometry("block/oak_slab", x_degrees=90.0)
    assert upright is not None
    assert not upright.covers("down")
    assert any(upright.covers(slot) for slot in ("north", "south"))


def test_bake_still_refuses_a_partial_model(tmp_path):
    jar = tmp_path / "slab.jar"
    _slab_jar(jar)
    with zipfile.ZipFile(jar) as archive:
        # The classifier palette compares full-cube appearances, so the partial
        # path must stay separate from it rather than quietly widening it.
        assert ModelAtlas(archive).bake("block/oak_slab") is None
