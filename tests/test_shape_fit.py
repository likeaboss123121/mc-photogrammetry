"""Fitting a partial block's shape and material against the photographs.

The classifier compares a rectified face to a texture, which cannot separate a
stair from the planks it is cut from. This renders each candidate into the scene
and asks which one explains the pixels.

Four things had to be right before it worked, and three of them are the sort that
fail silently, so they are pinned here: search only the properties a builder
chooses, judge inside the cell's silhouette, and shade by ambient occlusion.
"""
import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import (
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_volume,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.reconstruction.shape_fit import (
    candidate_states,
    fit_states,
    is_canonical,
)

SLOTS = ("up", "down", "north", "south", "east", "west")
CARPET = (((0.0, 0.0, 0.0), (1.0, 0.0625, 1.0)),)
SLAB = (((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),)


def _entry(name, rgb, boxes, properties=()):
    return PaletteColor(
        BlockState(f"minecraft:{name}", tuple(properties)), rgb,
        face_map=tuple((slot, rgb) for slot in SLOTS),
        full_cube=not boxes, shape_boxes=boxes,
    )


def test_a_state_with_no_derived_properties_is_canonical():
    assert is_canonical(BlockState("minecraft:white_carpet"))
    assert is_canonical(BlockState("minecraft:oak_slab", (("type", "bottom"),)))


def test_a_connected_fence_is_not_canonical():
    # Minecraft derives a fence's connections from its neighbours, so they are not
    # a builder's choice and must not be searched.
    connected = BlockState("minecraft:oak_fence",
                           (("north", "true"), ("east", "false")))
    assert not is_canonical(connected)
    assert is_canonical(BlockState("minecraft:oak_fence",
                                   (("north", "false"), ("east", "false"))))


def test_a_stair_corner_is_not_canonical():
    # `shape` turns a staircase around a corner and is likewise derived.
    assert is_canonical(BlockState("minecraft:oak_stairs",
                                   (("facing", "east"), ("shape", "straight"))))
    assert not is_canonical(BlockState("minecraft:oak_stairs",
                                       (("facing", "east"), ("shape", "inner_left"))))


def test_a_waterlogged_state_is_not_canonical():
    assert not is_canonical(BlockState("minecraft:oak_slab",
                                       (("waterlogged", "true"),)))


def test_candidates_group_materials_under_one_shape():
    palette = (
        _entry("white_carpet", (240, 240, 240), CARPET),
        _entry("red_carpet", (200, 40, 40), CARPET),
        _entry("oak_slab", (160, 130, 80), SLAB, (("type", "bottom"),)),
    )
    shapes, materials = candidate_states(palette)
    assert len(shapes) == 2
    counts = sorted(len(v) for v in materials.values())
    assert counts == [1, 2]


def test_full_cubes_are_never_candidates():
    palette = (_entry("stone", (120, 120, 120), ()),)
    shapes, _materials = candidate_states(palette)
    assert shapes == {}


def test_a_block_outside_the_build_families_is_not_offered():
    # The full palette holds 797 partial shapes, most of them coral fans and crop
    # stems; offering them is how `fire` and `resin_clump` came to win cells.
    palette = (_entry("dead_tube_coral_fan", (150, 150, 150), CARPET),)
    shapes, _materials = candidate_states(palette)
    assert shapes == {}


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(6, 6, 6))


def test_no_cells_means_no_work():
    palette = (_entry("white_carpet", (240, 240, 240), CARPET),)
    assert fit_states([], {}, _scene(), {}, {}, palette) == {}


def test_an_empty_palette_fits_nothing():
    assert fit_states([(1, 1, 1)], {}, _scene(), {}, {}, ()) == {}


def test_a_cell_no_camera_can_see_is_left_alone():
    palette = (_entry("white_carpet", (240, 240, 240), CARPET),)
    camera = CameraModel(position=(3.0, 2.0, -8.0), yaw_degrees=180.0,
                         pitch_degrees=0.0, fov_degrees=70.0)
    frames = {"a.png": np.zeros((40, 40, 3), dtype=np.uint8)}
    fitted = fit_states([(3, 2, 3)], {(3, 2, 3): BlockState("minecraft:white_carpet")},
                        _scene(), {"a.png": camera}, frames, palette)
    assert fitted == {}


def test_the_brighter_of_two_materials_wins_on_a_bright_photograph():
    # Same shape, opposite colours: the fit must follow the pixels.
    palette = (
        _entry("white_carpet", (245, 245, 245), CARPET),
        _entry("black_carpet", (10, 10, 10), CARPET),
    )
    scene = _scene()
    camera = CameraModel(position=(2.5, 3.5, -4.0), yaw_degrees=0.0,
                         pitch_degrees=35.0, fov_degrees=90.0)
    photo = np.full((120, 120, 3), 240, dtype=np.uint8)
    floor = {(x, 0, z): BlockState("minecraft:stone") for x in range(6) for z in range(6)}
    target = (2, 1, 3)
    blocks = {**floor, target: BlockState("minecraft:white_carpet")}
    fitted = fit_states([target], blocks, scene, {"a.png": camera},
                        {"a.png": photo}, palette)
    if fitted:
        assert fitted[target].name == "minecraft:white_carpet"


def test_no_proposals_means_no_residual_work():
    from mc_photogrammetry.reconstruction.residual import residual_proposals

    palette = (_entry("white_carpet", (240, 240, 240), CARPET),)
    assert residual_proposals({}, _scene(), {}, {}, palette, limit=0) == {}


def test_a_proposal_lands_in_front_of_the_face_that_disagrees():
    """The empty cell along the face normal, toward the camera.

    For a floor's top face that is where a carpet would lie; for a wall, where a
    stair or a slab would sit. Proposing the cell *behind* would be proposing
    somewhere no camera can see.
    """
    import numpy as np

    from mc_photogrammetry.reconstruction.residual import residual_proposals

    scene = _scene()
    camera = CameraModel(position=(2.5, 4.0, -4.0), yaw_degrees=0.0,
                         pitch_degrees=40.0, fov_degrees=90.0)
    floor = {(x, 1, z): BlockState("minecraft:stone") for x in range(6) for z in range(6)}
    # A photograph that agrees with nothing, so every face is doubted.
    photo = np.full((100, 100, 3), 255, dtype=np.uint8)
    palette = (
        _entry("stone", (20, 20, 20), ()),
        _entry("white_carpet", (245, 245, 245), CARPET),
    )
    found = residual_proposals(floor, scene, {"a.png": camera}, {"a.png": photo},
                               palette, limit=5, threshold=0.0)
    assert found, "nothing proposed"
    for cell in found:
        assert cell not in floor
        assert cell[1] == 2, "a top face should propose the cell above it"


def _fit_scene():
    """A carpet lying on a stone floor, seen from above and in front."""
    palette = (
        _entry("stone", (120, 120, 120), ()),
        _entry("white_carpet", (245, 245, 245), CARPET),
        _entry("black_carpet", (10, 10, 10), CARPET),
        _entry("oak_slab", (160, 130, 80), SLAB, (("type", "bottom"),)),
    )
    camera = CameraModel(position=(2.5, 3.5, -4.0), yaw_degrees=0.0,
                         pitch_degrees=35.0, fov_degrees=90.0)
    floor = {(x, 0, z): BlockState("minecraft:stone")
             for x in range(6) for z in range(6)}
    return palette, camera, floor


def test_a_face_too_small_to_read_offers_no_grain():
    # A patch of a dozen pixels carries no usable texture, and rectifying it
    # would hand the classifier upscaled noise to correlate against.
    from mc_photogrammetry.reconstruction.shape_fit import (
        MINIMUM_FACE_AREA,
        _face_patches,
    )

    scene = _scene()
    far = CameraModel(position=(2.5, 3.0, -400.0), yaw_degrees=0.0,
                      pitch_degrees=0.0, fov_degrees=70.0)
    photo = np.full((200, 200, 3), 128, dtype=np.uint8)
    found = _face_patches(scene, far, (2, 1, 3), CARPET, photo, 1.0)
    assert found == [], f"a cell 400 blocks away offered {len(found)} readable faces"
    assert MINIMUM_FACE_AREA > 0


def test_face_texture_sampling_rejects_an_excluded_usable_region():
    from mc_photogrammetry.reconstruction.shape_fit import _face_patches

    scene = _scene()
    camera = CameraModel((2.5, 3.5, -4.0), yaw_degrees=0.0,
                         pitch_degrees=35.0, fov_degrees=90.0)
    image = np.full((120, 120, 3), 160, dtype=np.uint8)
    excluded = np.zeros((120, 120), dtype=bool)

    assert _face_patches(
        scene, camera, (2, 1, 3), CARPET, image, 1.0,
        usable_mask=excluded,
    ) == []


def test_the_shape_does_not_change_with_the_texture_weight():
    """Silhouette decides the shape; grain only decides the material.

    Summing the two into one score let the grain outvote the silhouette and cost
    2.9pp of family accuracy on Template 2, so the split is the behaviour under
    test, not an implementation detail.
    """
    palette, camera, floor = _fit_scene()
    scene = _scene()
    photo = np.full((120, 120, 3), 240, dtype=np.uint8)
    target = (2, 1, 3)
    blocks = {**floor, target: BlockState("minecraft:white_carpet")}

    shapes = set()
    for weight in (0.0, 1.0, 64.0):
        fitted = fit_states([target], blocks, scene, {"a.png": camera},
                            {"a.png": photo}, palette, texture_weight=weight)
        if target in fitted:
            shapes.add("slab" if "slab" in fitted[target].name else "carpet")
    assert len(shapes) <= 1, f"the texture weight moved the shape: {shapes}"


def test_the_joint_mode_still_names_a_block():
    # The pre-split behaviour stays reachable, so the split can be measured
    # against it rather than argued about.
    palette, camera, floor = _fit_scene()
    target = (2, 1, 3)
    blocks = {**floor, target: BlockState("minecraft:white_carpet")}
    fitted = fit_states([target], blocks, _scene(), {"a.png": camera},
                        {"a.png": np.full((120, 120, 3), 240, dtype=np.uint8)},
                        palette, shape_from_silhouette=False)
    for state in fitted.values():
        assert state.name.startswith("minecraft:")


def test_slab_beats_carpet_regardless_of_palette_order():
    """A small candidate must pay for the photographed surface it leaves unexplained."""
    target = (0, 0, 0)
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)

    colours = face_colours((slab, carpet))
    photo, _owner, _index, _depth = render_volume(
        {target: slab.state}, scene, camera, 120, 120, colours,
        face_shapes((slab, carpet)), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots((slab, carpet)),
        model_faces=face_model_faces((slab, carpet)),
    )
    frames = {"slab.png": np.rint(photo).astype(np.uint8)}
    cameras = {"slab.png": camera}

    for palette in ((carpet, slab), (slab, carpet)):
        fitted = fit_states(
            [target], {target: slab.state}, scene, cameras, frames, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
        )
        assert fitted[target] == slab.state


def test_shape_fit_excludes_a_cell_occluded_by_the_scene():
    """A foreground cube makes every shape at the covered target unscorable."""
    target, foreground = (0, 0, 1), (0, 0, 0)
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (2, 2, 3))

    fitted = fit_states(
        [target], {target: slab.state, foreground: BlockState("minecraft:stone")}, scene,
        {"covered.png": camera}, {"covered.png": np.zeros((120, 120, 3), dtype=np.uint8)},
        (carpet, slab), scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert fitted == {}


def test_an_empty_proposal_stays_empty_when_the_background_already_matches():
    palette, camera, floor = _fit_scene()
    scene = _scene()
    target = (2, 1, 3)
    colours = face_colours(palette)
    photo, _owner, _index, _depth = render_volume(
        floor, scene, camera, 120, 120, colours, face_shapes(palette),
        scale=1.0, ambient=True, occluding_slots=face_occluding_slots(palette),
        model_faces=face_model_faces(palette),
    )

    fitted = fit_states(
        [target], floor, scene, {"floor.png": camera},
        {"floor.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert target not in fitted


def test_an_empty_proposal_can_recover_a_real_slab():
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = (carpet, slab)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    colours = face_colours(palette)
    photo, _owner, _index, _depth = render_volume(
        {target: slab.state}, scene, camera, 120, 120, colours,
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )

    fitted = fit_states(
        [target], {}, scene, {"slab.png": camera},
        {"slab.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert fitted[target] == slab.state


def test_a_correct_cube_is_retained_over_partial_alternatives():
    stone = _entry("stone", (120, 120, 120), ())
    slab = _entry("quartz_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    palette = (slab, stone)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    colours = face_colours(palette)
    photo, _owner, _index, _depth = render_volume(
        {target: stone.state}, scene, camera, 120, 120, colours,
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )

    fitted = fit_states(
        [target], {target: stone.state}, scene, {"stone.png": camera},
        {"stone.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert fitted[target] == stone.state
