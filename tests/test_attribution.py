"""Attributing visible error to blocks and to causes.

Cell-counting measures weight a carpet covering a floor the same as an edge-on face
forty blocks away. This instrument weights by what the camera actually sees, which
is how Template 2 turned out to be 8.1% partial blocks by count and 76.3% by error.

The four causes come from *where each ray stopped*. Asking instead whether each ray
hit anything is useless - both volumes are usually closed, so both almost always
hit something, and ``missing`` and ``phantom`` sit at zero while their content
hides inside a catch-all.
"""
import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import (
    MATERIAL,
    MISSING,
    PHANTOM,
    attribute_error,
    box_face_corners,
    face_colours,
    face_shapes,
    render_volume,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.camera import project_points
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.scene import SceneTransform

WHITE = BlockState("minecraft:white_wool")
BLACK = BlockState("minecraft:black_wool")
CARPET = BlockState("minecraft:white_carpet")

SLOTS = ("up", "down", "north", "south", "east", "west")


def _palette():
    def entry(state, rgb, boxes=(), full=True):
        return PaletteColor(
            state, rgb,
            face_map=tuple((slot, rgb) for slot in SLOTS),
            full_cube=full, shape_boxes=boxes,
        )
    return (
        entry(WHITE, (240, 240, 240)),
        entry(BLACK, (12, 12, 12)),
        entry(CARPET, (240, 240, 240), (((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0)),), full=False),
    )


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(8, 8, 8))


def _camera():
    return CameraModel(position=(3.5, 2.0, -6.0), yaw_degrees=0.0, pitch_degrees=0.0,
                       fov_degrees=90.0)


def _frames():
    return {"a.png": np.zeros((120, 120, 3), dtype=np.uint8)}


def _cameras():
    return {"a.png": _camera()}


def _wall(state):
    return {(x, y, 4): state for x in range(2, 6) for y in range(1, 4)}


def test_a_volume_identical_to_itself_has_no_error():
    wall = _wall(WHITE)
    report = attribute_error(wall, dict(wall), _scene(), _cameras(), _frames(), _palette())
    assert report.total == 0.0


def test_a_recoloured_wall_is_a_material_error():
    report = attribute_error(_wall(WHITE), _wall(BLACK), _scene(), _cameras(),
                             _frames(), _palette())
    assert report.total > 0.0
    assert max(report.share(), key=report.share().get) == MATERIAL
    assert report.worst_blocks(1)[0][0] == WHITE.name


def test_a_block_we_never_placed_is_missing_not_material():
    # The reference has a wall in front of a far wall; the candidate has only the
    # far one, so its ray travels further.
    far = {(x, y, 7): BLACK for x in range(2, 6) for y in range(1, 4)}
    reference = {**far, **_wall(WHITE)}
    report = attribute_error(reference, far, _scene(), _cameras(), _frames(), _palette())
    assert report.energy.get(MISSING, 0.0) > 0.0
    assert report.energy.get(PHANTOM, 0.0) == 0.0


def test_a_block_we_invented_is_phantom_not_material():
    far = {(x, y, 7): BLACK for x in range(2, 6) for y in range(1, 4)}
    candidate = {**far, **_wall(WHITE)}
    report = attribute_error(far, candidate, _scene(), _cameras(), _frames(), _palette())
    assert report.energy.get(PHANTOM, 0.0) > 0.0
    assert report.energy.get(MISSING, 0.0) == 0.0


def test_the_blame_names_the_reference_block_not_ours():
    # A report is read to decide what to work on, so it must name the thing that
    # should have been there.
    report = attribute_error(_wall(WHITE), _wall(BLACK), _scene(), _cameras(),
                             _frames(), _palette())
    assert report.worst_confusions(1)[0][:2] == (WHITE.name, BLACK.name)


def test_shares_sum_to_one_and_are_ordered():
    report = attribute_error(_wall(WHITE), _wall(BLACK), _scene(), _cameras(),
                             _frames(), _palette())
    shares = list(report.share().values())
    assert sum(shares) == pytest.approx(1.0)
    assert shares == sorted(shares, reverse=True)


def test_a_carpet_is_drawn_as_a_carpet_and_not_as_a_cube():
    # The flaw that made the residual loop reject its own correct answers: drawn
    # as a cube, a carpet covers a whole cell face it never fills.
    scene, camera = _scene(), _camera()
    blocks = {(3, 2, 4): CARPET}
    palette = _palette()
    colours, shapes = face_colours(palette), face_shapes(palette)
    _cube, cube_owner, _ci, _cd = render_volume(blocks, scene, camera, 120, 120, colours)
    _box, box_owner, _bi, _bd = render_volume(blocks, scene, camera, 120, 120,
                                              colours, shapes)
    assert (cube_owner >= 0).sum() > (box_owner >= 0).sum()


def test_depth_buffer_keeps_the_nearer_surface_at_an_overlapping_pixel():
    scene = _scene()
    camera = _camera()
    colours = face_colours(_palette())
    near = (3, 2, 4)
    far = (3, 2, 6)
    _canvas, owner, index, depth = render_volume(
        {near: WHITE, far: BLACK}, scene, camera, 120, 120, colours, scale=1.0
    )
    centre = np.asarray(scene.to_world(near), dtype=float) + 0.5
    pixels = project_points(centre + box_face_corners((0, 0, 0), (1, 1, 1), "north"), camera, width=120, height=120)
    column, row = (int(round(value)) for value in np.mean([[pixel[0], pixel[1]] for pixel in pixels], axis=0))
    assert owner[row, column] >= 0
    assert index[owner[row, column]][0] == near
    assert depth[row, column] == pytest.approx(10.0, abs=0.1)


def test_a_box_face_lies_in_its_own_plane():
    corners = box_face_corners((0.0, 0.0, 0.0), (1.0, 0.5, 1.0), "up")
    assert np.allclose(corners[:, 1], 0.0)  # y = 0.5, less the half-cell offset
    assert corners.shape == (4, 3)


def test_an_unknown_face_slot_is_refused():
    with pytest.raises(ValueError):
        box_face_corners((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), "sideways")


def test_a_bad_scale_is_refused():
    with pytest.raises(ValueError):
        render_volume({}, _scene(), _camera(), 120, 120, {}, None, 0.0)


def test_a_frame_without_a_camera_is_skipped():
    report = attribute_error(_wall(WHITE), _wall(BLACK), _scene(), {}, _frames(), _palette())
    assert report.total == 0.0
