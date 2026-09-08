"""Casting the reference through the cameras to find its reachable blocks.

The result becomes a denominator, so the properties worth pinning are the ones
that would quietly inflate or deflate it: that occlusion is real, that a partial
block does not occlude like a cube, and that the ray parameter really is the
camera depth the projection reports.
"""
import base64
from io import BytesIO

import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.visibility import (
    full_cube_keys,
    ray_directions,
    shape_table,
    theoretical_maximum,
    visible_cells,
)
from mc_photogrammetry.pipeline.contracts import CameraIntrinsics
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.export.attribution import box_face_corners

SIZE = (256, 256)
CARPET = (((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0)),)
SLAB = (((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),)
CUBE = (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)
STONE = BlockState("minecraft:stone")


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(8, 8, 8))


def _camera(eye_y=2.5):
    """Standing back along -Z, looking toward +Z, which is Minecraft yaw 0."""
    return CameraModel(position=(3.5, eye_y - 1.62, -4.0), yaw_degrees=0.0,
                       pitch_degrees=0.0, fov_degrees=70.0)


def _look(blocks, shapes=None, full_cubes=None, camera=None):
    return visible_cells(
        blocks, _scene(), {"a.png": camera or _camera()}, {"a.png": SIZE},
        shapes=shapes, full_cubes=full_cubes)


def test_the_ray_parameter_is_the_camera_depth():
    # The depth buffer compares this against `project_points`, so the two must
    # be the same quantity, not merely proportional.
    camera = CameraModel(position=(1.0, 2.0, 3.0), yaw_degrees=37.0,
                         pitch_degrees=-12.0, fov_degrees=80.0)
    width, height = 640, 480
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    point = eye + np.asarray([2.0, -1.0, 9.0])
    (projected,) = project_points([tuple(point)], camera, width=width, height=height)
    x, y, depth = projected
    direction = ray_directions(np.array([x]), np.array([y]), camera, width, height)[0]
    assert np.allclose(eye + depth * direction, point, atol=1e-9)


def test_visibility_projection_and_rays_honour_explicit_intrinsics_and_roll():
    from mc_photogrammetry.export.visibility import _project_clipped

    camera = CameraModel(
        position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0, fov_degrees=90.0,
        eye_height=0.0, roll_degrees=10.0,
        intrinsics=CameraIntrinsics((100, 100), (50.0, 50.0), (30.0, 40.0)),
    )
    corners = np.asarray(((-0.5, -0.5, 4.0), (0.5, -0.5, 4.0),
                          (0.5, 0.5, 4.0), (-0.5, 0.5, 4.0)))
    shared = np.asarray(project_points(corners, camera, width=100, height=100), dtype=float)
    assert np.allclose(_project_clipped(corners, camera, 100, 100), shared[:, :2])
    direction = ray_directions(shared[:, 0], shared[:, 1], camera, 100, 100)
    assert np.allclose(direction * shared[:, 2, None], corners, atol=1e-9)


def test_a_block_in_front_of_the_camera_is_reached():
    found = _look({(3, 2, 2): STONE})
    assert found.get((3, 2, 2), 0) > 0


def test_frozen_usable_mask_restricts_the_visibility_denominator():
    blocks = {(3, 2, 2): STONE}
    unrestricted = _look(blocks)
    masked = visible_cells(
        blocks, _scene(), {"a.png": _camera()}, {"a.png": SIZE},
        usable_masks={"a.png": np.zeros((SIZE[1], SIZE[0]), dtype=bool)},
    )
    assert unrestricted[(3, 2, 2)] > 0
    assert masked == {}


def test_usable_mask_dimensions_are_checked():
    with pytest.raises(ValueError, match="expected"):
        visible_cells(
            {(3, 2, 2): STONE}, _scene(), {"a.png": _camera()}, {"a.png": SIZE},
            usable_masks={"a.png": np.ones((10, 10), dtype=bool)},
        )


def test_a_block_outside_the_frustum_is_not_reached():
    # Behind the camera entirely.
    assert _look({(3, 2, 2): STONE}, camera=CameraModel(
        position=(3.5, 0.88, -4.0), yaw_degrees=180.0,
        pitch_degrees=0.0, fov_degrees=70.0)) == {}


def test_a_full_cube_hides_what_stands_behind_it():
    blocks = {(3, 2, 2): STONE, (3, 2, 3): STONE}
    found = _look(blocks, full_cubes={"minecraft:stone"})
    assert found.get((3, 2, 2), 0) > 0, "the near block should be reached"
    assert (3, 2, 3) not in found, "the far block is squarely behind the near one"


def test_a_carpet_does_not_hide_the_block_behind_it():
    """The reason this cannot reuse the attribution renderer.

    `attribution` culls a cell-wall face whenever *any* neighbour is occupied,
    which is right for a diagnostic and would delete the floor under every
    carpet here.
    """
    blocks = {(3, 2, 2): BlockState("minecraft:white_carpet"), (3, 2, 3): STONE}
    shapes = {"minecraft:white_carpet": CARPET}
    found = _look(blocks, shapes=shapes, full_cubes={"minecraft:stone"})
    assert found.get((3, 2, 3), 0) > 0, "a carpet occludes a sixteenth of a cube"


def test_palette_visibility_uses_exact_alpha_tested_model_faces():
    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[:, 8:] = 255
    buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    white, black = BlockState("minecraft:white_wool"), BlockState("minecraft:black_wool")
    corners = tuple(tuple(float(v) for v in point) for point in box_face_corners((0, 0, 0), (1, 1, 1), "north"))
    palette = (
        PaletteColor(white, (240, 240, 240), model_faces=(("north", corners, (0, 0, -1), None, encoded),)),
        PaletteColor(black, (12, 12, 12)),
    )
    near, far = (3, 2, 2), (3, 2, 4)
    found = visible_cells(
        {near: white, far: black}, _scene(), {"a.png": _camera()}, {"a.png": SIZE}, palette=palette
    )
    assert found.get(near, 0) > 0 and found.get(far, 0) > 0


def test_depth_decides_rather_than_draw_order():
    """A large far face must not overwrite a small near one.

    Painter's algorithm sorted by a face's mean depth gets this wrong, and the
    consequence would be deleting a visible block from the maximum.
    """
    wall = {(x, y, 6): STONE for x in range(8) for y in range(8)}
    near = {(3, 2, 1): STONE}
    found = _look({**wall, **near}, full_cubes={"minecraft:stone"})
    assert found.get((3, 2, 1), 0) > 0, "the near block was overpainted by the wall"


def test_a_block_walled_in_on_every_side_is_unreachable():
    """The blocks this whole instrument exists to stop charging for.

    A cell with six full-cube neighbours cannot be seen from anywhere, so no
    camera placement and no algorithm will ever recover it.
    """
    solid = {(x, y, z): STONE
             for x in range(2, 5) for y in range(1, 4) for z in range(2, 5)}
    found = _look(solid, full_cubes={"minecraft:stone"})
    assert found, "the shell itself should be reached"
    assert (3, 2, 3) not in found, "the buried centre cell was reported visible"


def test_the_maximum_is_the_reference_restricted_to_what_was_reached():
    blocks = {(1, 1, 1): STONE, (2, 2, 2): BlockState("minecraft:dirt")}
    counts = {(1, 1, 1): 40, (2, 2, 2): 3}
    kept = theoretical_maximum(blocks, counts, minimum_pixels=1)
    assert kept == blocks
    strict = theoretical_maximum(blocks, counts, minimum_pixels=10)
    assert strict == {(1, 1, 1): STONE}
    assert strict[(1, 1, 1)] is blocks[(1, 1, 1)], "states must survive unchanged"


def test_a_cell_reached_by_nothing_never_appears():
    assert theoretical_maximum({(1, 1, 1): STONE}, {}) == {}


def test_full_cube_keys_carry_both_the_state_and_the_name():
    palette = (
        PaletteColor(BlockState("minecraft:stone"), (120, 120, 120), full_cube=True),
        PaletteColor(BlockState("minecraft:white_carpet"), (240, 240, 240),
                     full_cube=False, shape_boxes=CARPET),
    )
    keys = full_cube_keys(palette)
    assert BlockState("minecraft:stone") in keys
    assert "minecraft:stone" in keys
    assert "minecraft:white_carpet" not in keys


def test_a_name_shared_by_a_cube_and_a_half_block_is_not_admitted_by_name():
    """A slab is a whole cube as `type=double` and half a block otherwise.

    Admitting the bare name would cull the face behind every slab in the fixture
    and understate the maximum, so only the exact double state may occlude.
    """
    double = BlockState("minecraft:oak_slab", (("type", "double"),))
    bottom = BlockState("minecraft:oak_slab", (("type", "bottom"),))
    palette = (
        PaletteColor(double, (160, 130, 80), full_cube=True),
        PaletteColor(bottom, (160, 130, 80), full_cube=False, shape_boxes=SLAB),
    )
    keys = full_cube_keys(palette)
    assert double in keys
    assert bottom not in keys
    assert "minecraft:oak_slab" not in keys, "the name must not stand in for the state"


def test_the_shape_table_prefers_the_exact_state_over_the_name():
    double = BlockState("minecraft:oak_slab", (("type", "double"),))
    bottom = BlockState("minecraft:oak_slab", (("type", "bottom"),))
    palette = (
        PaletteColor(double, (160, 130, 80), full_cube=True),
        PaletteColor(bottom, (160, 130, 80), full_cube=False, shape_boxes=SLAB),
    )
    table = shape_table(palette)
    assert table[bottom] == SLAB
    assert table[double] == CUBE


def test_a_frame_with_no_resolution_is_skipped_rather_than_guessed():
    found = visible_cells({(3, 2, 2): STONE}, _scene(),
                          {"a.png": _camera()}, {})
    assert found == {}


def test_no_blocks_means_no_visibility():
    assert visible_cells({}, _scene(), {"a.png": _camera()}, {"a.png": SIZE}) == {}


def test_a_face_straddling_the_eye_plane_survives_as_its_visible_part():
    """Dropping it would delete the near wall that occludes the room.

    A face with one corner behind the eye cannot be projected. Skipping the
    whole face is not conservative in the direction that matters: with cameras
    standing inside a room, as every fixture here has, it removes exactly the
    surfaces doing the occluding and buried blocks leak into the maximum.
    """
    from mc_photogrammetry.export.visibility import _camera_space, _project_clipped

    camera = CameraModel(position=(3.5, 0.88, 3.0), yaw_degrees=0.0,
                         pitch_degrees=0.0, fov_degrees=70.0)
    # A floor quad running from behind the camera to well in front of it.
    corners = [(3.0, 2.0, 1.0), (4.0, 2.0, 1.0), (4.0, 2.0, 9.0), (3.0, 2.0, 9.0)]
    depths = _camera_space(corners, camera)[:, 2]
    assert depths.min() < 0 < depths.max(), "the fixture must actually straddle"
    polygon = _project_clipped(corners, camera, 256, 256)
    assert polygon.shape[0] >= 3, "the visible half of the face was thrown away"


def test_a_face_against_the_lens_stays_inside_the_frame():
    """Rasterised coordinates follow the image, not the camera's proximity.

    A face a twentieth of a block from the eye projects past +/-80,000 pixels.
    That is within what OpenCV fills correctly, so this is not the bug that was
    found - it just keeps the polygon the size of the region it covers.
    """
    from mc_photogrammetry.export.visibility import _project_clipped

    camera = CameraModel(position=(3.5, 0.88, 2.4), yaw_degrees=0.0,
                         pitch_degrees=0.0, fov_degrees=70.0)
    corners = [(0.0, 0.0, 2.5), (8.0, 0.0, 2.5), (8.0, 8.0, 2.5), (0.0, 8.0, 2.5)]
    polygon = _project_clipped(corners, camera, 256, 256)
    assert polygon.shape[0] >= 3
    assert polygon[:, 0].min() >= -1e-6 and polygon[:, 0].max() <= 256 + 1e-6
    assert polygon[:, 1].min() >= -1e-6 and polygon[:, 1].max() <= 256 + 1e-6


def test_a_wall_the_camera_is_pressed_against_still_occludes():
    """The end-to-end consequence of the near clip, and the bug that was found.

    The wall's side faces straddle the eye plane at this range. Dropping such
    faces, as the first version did, removed the occluder and reported the
    hidden block as plainly visible - 33,243 pixels for a sealed cell in
    Template 4. 4.55% of that fixture's candidate faces straddle.
    """
    wall = {(x, y, 3): STONE for x in range(8) for y in range(8)}
    hidden = (3, 2, 5)
    # Five hundredths of a block from the wall, at a real screenshot's size, so
    # the projected quad genuinely reaches the coordinates that overflow.
    close = CameraModel(position=(3.5, 0.88, 2.95), yaw_degrees=0.0,
                        pitch_degrees=0.0, fov_degrees=70.0)
    found = visible_cells({**wall, hidden: STONE}, _scene(), {"a.png": close},
                          {"a.png": (2560, 1369)}, full_cubes={"minecraft:stone"})
    assert found, "the wall itself must be reached"
    assert hidden not in found, "a block behind a wall pressed against the lens"


def test_no_ray_leaks_through_a_solid_double_wall():
    """Two full layers of stone must stop every ray.

    Watertightness at the seams between adjacent quads is the property; the
    rasteriser is trusted for it rather than patched, because drawing each
    face's outline as well changed nothing measurable on any fixture and would
    have widened every face by a pixel.
    """
    wall = {(x, y, 4): STONE for x in range(8) for y in range(8)}
    behind = {(x, y, 5): STONE for x in range(8) for y in range(8)}
    found = visible_cells({**wall, **behind}, _scene(), {"a.png": _camera()},
                          {"a.png": SIZE}, full_cubes={"minecraft:stone"})
    leaked = [cell for cell in found if cell[2] == 5]
    assert not leaked, f"rays leaked through the wall onto {leaked[:5]}"
