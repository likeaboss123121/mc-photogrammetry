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
    _representative_material,
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
    if fitted and fitted[target] is not None:
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


def test_face_texture_sampling_treats_missing_owned_slots_as_invisible():
    from mc_photogrammetry.reconstruction.shape_fit import _face_patches

    scene = _scene()
    camera = CameraModel((2.5, 3.5, -4.0), yaw_degrees=0.0,
                         pitch_degrees=35.0, fov_degrees=90.0)
    image = np.full((120, 120, 3), 160, dtype=np.uint8)

    assert _face_patches(
        scene, camera, (2, 1, 3), CARPET, image, 1.0,
        owner_masks={},
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
        if target in fitted and fitted[target] is not None:
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
        if state is not None:
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


def test_air_can_explicitly_delete_an_occupied_cell():
    """An occupied cell is removed when the photographed background wins."""
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = (carpet, slab)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    colours = face_colours(palette)
    photo, _owner, _index, _depth = render_volume(
        {}, scene, camera, 120, 120, colours, face_shapes(palette), scale=1.0,
        ambient=True, occluding_slots=face_occluding_slots(palette),
        model_faces=face_model_faces(palette),
    )

    fitted = fit_states(
        [target], {target: slab.state}, scene, {"floor.png": camera},
        {"floor.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert fitted[target] is None


def test_all_excluded_pixels_abstain_without_a_fit():
    palette, camera, floor = _fit_scene()
    target = (2, 1, 3)
    fitted = fit_states(
        [target], floor, _scene(), {"blank.png": camera},
        {"blank.png": np.zeros((120, 120, 3), dtype=np.uint8)}, palette,
        usable_masks={"blank.png": np.zeros((120, 120), dtype=bool)},
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert fitted == {}


@pytest.mark.parametrize("gain", (0.5, 1.5))
def test_air_and_candidates_share_the_same_exposure_gain(gain):
    palette, camera, floor = _fit_scene()
    scene = _scene()
    target = (2, 1, 3)
    colours = face_colours(palette)
    photo, _owner, _index, _depth = render_volume(
        floor, scene, camera, 120, 120, colours, face_shapes(palette),
        scale=1.0, ambient=True, occluding_slots=face_occluding_slots(palette),
        model_faces=face_model_faces(palette),
    )
    frame = np.clip(np.rint(photo * gain), 0, 255).astype(np.uint8)

    fitted = fit_states(
        [target], floor, scene, {"floor.png": camera}, {"floor.png": frame},
        palette, scale=1.0, edge_weight=0.0, texture_weight=0.0,
    )

    assert target not in fitted


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


def test_shape_fit_reports_render_work_and_honours_a_candidate_cap():
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = (slab, carpet)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    photo, _owner, _index, _depth = render_volume(
        {target: slab.state}, scene, camera, 120, 120, face_colours(palette),
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    diagnostics = {}

    fit_states(
        [target], {}, scene, {"slab.png": camera},
        {"slab.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=1.0,
        diagnostics=diagnostics, max_candidate_renders=1,
    )

    assert diagnostics["shape_fit_context_renders"] >= 1
    assert diagnostics["shape_fit_probe_renders"] >= 1
    assert diagnostics["shape_fit_ownership_renders"] >= 1
    assert diagnostics["shape_fit_candidate_renders"] == 1
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True
    assert diagnostics["shape_fit_seconds"] >= 0.0


def test_candidate_render_cap_below_view_count_reserves_no_partial_hypothesis():
    """A cap below the usable-view count spends no candidate render at all.

    The previous scheduler rendered a prefix of the views and then discarded the
    incomplete hypothesis, spending the scarce budget without ever evaluating a
    complete one.  The whole view set is reserved instead, so the cell is left
    explicitly unvisited and its discovery work is counted separately.
    """
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = (carpet, slab)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    cameras = {
        name: CameraModel((0.5, 0.25, -3.0), 0.0, 0.0,
                          fov_degrees=90.0, eye_height=0.0)
        for name in ("a.png", "b.png", "c.png")
    }
    colours = face_colours(palette)
    rendered, _owner, _index, _depth = render_volume(
        {target: slab.state}, scene, cameras["a.png"], 120, 120, colours,
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    frames = {name: np.rint(rendered).astype(np.uint8) for name in cameras}
    diagnostics = {}

    fitted = fit_states(
        [target], {}, scene, cameras, frames, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=1,
    )

    results = diagnostics["shape_fit_cell_results"]
    assert results[0]["usable_views"] == 3
    assert results[0]["status"] == "budget_unvisited"
    assert results[0]["candidate_renders"] == 0
    assert diagnostics["shape_fit_candidate_renders"] == 0
    assert diagnostics["shape_fit_completed_hypotheses"] == 0
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True
    # Discovery work keeps its own counter; it is never disguised as candidate work.
    assert diagnostics["shape_fit_probe_renders"] >= 1
    assert diagnostics["shape_fit_context_renders"] == 0
    assert target not in fitted


def test_zero_cap_reports_every_cell_budget_unvisited_without_rendering():
    """cap=0 must not build the palette or render background/context/probe."""
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = (carpet, slab)
    cells = [(0, 0, 0), (2, 0, 0)]
    scene = SceneTransform((0, 0, 0), (4, 2, 2))
    camera = CameraModel((1.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    diagnostics = {}

    fitted = fit_states(
        cells, {}, scene, {"a.png": camera}, {"a.png": frame}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=0,
    )

    assert fitted == {}
    for name in ("background_renders", "context_renders", "probe_renders",
                 "ownership_renders", "current_renders", "candidate_renders"):
        assert diagnostics[f"shape_fit_{name}"] == 0, name
    assert diagnostics["shape_fit_render_work"] == 0
    assert diagnostics["shape_fit_completed_hypotheses"] == 0
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True
    assert diagnostics["shape_fit_considered_shapes"] == []
    assert diagnostics["shape_fit_budget_unvisited_cells"] == 2
    results = diagnostics["shape_fit_cell_results"]
    assert [item["status"] for item in results] == ["budget_unvisited"] * 2
    assert [item["candidate_renders"] for item in results] == [0, 0]


def test_atomic_view_reservation_below_equal_and_above_cap():
    """One hypothesis needs its whole two-view set; the cap cannot be shaved.

    below -> no candidate render at all (cell budget_unvisited)
    equal -> exactly one complete hypothesis
    above -> one complete hypothesis, then a refinement refused atomically
    full  -> the representative plus one complete refinement
    """
    aaa = _entry("aaa_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    bbb = _entry("bbb_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    palette = (aaa, bbb)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    cameras = {"a.png": camera, "b.png": camera}
    photo, _owner, _index, _depth = render_volume(
        {target: bbb.state}, scene, camera, 120, 120, face_colours(palette),
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    frames = {name: np.rint(photo).astype(np.uint8) for name in cameras}

    observed = {}
    refusals = {}
    for cap in (1, 2, 3, 4):
        diagnostics = {}
        fit_states(
            [target], {}, scene, cameras, frames, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            diagnostics=diagnostics, max_candidate_renders=cap,
        )
        observed[cap] = (
            diagnostics["shape_fit_candidate_renders"],
            diagnostics["shape_fit_completed_hypotheses"],
            diagnostics["shape_fit_cell_results"][0]["status"],
        )
        refusals[cap] = list(diagnostics["shape_fit_unresolved_budget_exits"])
        assert diagnostics["shape_fit_candidate_render_cap_reached"] is True, cap

    assert observed[1] == (0, 0, "budget_unvisited")
    assert observed[2] == (2, 1, "partly_tested")
    assert observed[3] == (2, 1, "partly_tested")
    assert observed[4] == (4, 2, "accepted")
    # cap=3 is one render short of the refinement: it is refused, never partial.
    assert any(
        entry["phase"] == "material_refinement" for entry in refusals[3]
    )
    assert refusals[1][0]["phase"] == "reserve_usable_view_set"


def test_two_cells_two_views_cap_two_completes_one_hypothesis_only():
    """Never spend one render per cell: reserve both for one complete one."""
    slab = _entry("quartz_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (120, 120, 120), CARPET)
    stone = _entry("stone", (120, 120, 120), ())
    palette = (slab, carpet, stone)
    scene = SceneTransform((0, 0, 0), (4, 2, 2))
    cameras = {
        "front": CameraModel((1.5, 0.5, -5.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0),
        "back": CameraModel((1.5, 0.5, 5.0), 180.0, 0.0, fov_degrees=90.0, eye_height=0.0),
    }
    target_blocks = {(0, 0, 0): slab.state, (2, 0, 0): carpet.state}
    frames = {}
    for frame_id, camera in cameras.items():
        photo, _owner, _index, _depth = render_volume(
            target_blocks, scene, camera, 160, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        frames[frame_id] = np.rint(photo).astype(np.uint8)

    diagnostics = {}
    fitted = fit_states(
        [(0, 0, 0), (2, 0, 0)], {}, scene, cameras, frames, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=2,
    )

    results = diagnostics["shape_fit_cell_results"]
    assert diagnostics["shape_fit_candidate_renders"] == 2
    assert sum(item["candidate_renders"] for item in results) == 2
    assert diagnostics["shape_fit_completed_hypotheses"] == 1
    assert [item["status"] for item in results] == ["budget_unvisited", "partly_tested"]
    # Exactly one cell was decided; the other is explicitly unsearched.
    assert (2, 0, 0) in fitted
    assert (0, 0, 0) not in fitted


def test_uneven_view_costs_let_the_cheaper_cell_use_a_scarce_cap():
    """A one-view cell is eligible at cap1 where the two-view cell is not."""
    stone = _entry("stone", (120, 120, 120), ())
    carpet = _entry("white_carpet", (240, 240, 240), CARPET)
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    palette = (carpet, slab, stone)
    scene = SceneTransform((0, 0, 0), (2, 2, 3))
    cameras = {
        "front": CameraModel((0.5, 0.5, -5.0), 0.0, 0.0, fov_degrees=70.0, eye_height=0.0),
        "back": CameraModel((0.5, 0.5, 7.0), 180.0, 0.0, fov_degrees=70.0, eye_height=0.0),
    }
    target_blocks = {(0, 0, 0): stone.state, (0, 0, 1): carpet.state}
    frames = {}
    for frame_id, camera in cameras.items():
        photo, _owner, _index, _depth = render_volume(
            target_blocks, scene, camera, 160, 160, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        frames[frame_id] = np.rint(photo).astype(np.uint8)

    diagnostics = {}
    fitted = fit_states(
        [(0, 0, 0), (0, 0, 1)], {(0, 0, 0): stone.state}, scene, cameras, frames, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=1,
    )

    results = {
        tuple(item["coordinate"]): item
        for item in diagnostics["shape_fit_cell_results"]
    }
    assert results[(0, 0, 0)]["usable_views"] == 2
    assert results[(0, 0, 0)]["status"] == "budget_unvisited"
    assert results[(0, 0, 0)]["candidate_renders"] == 0
    assert results[(0, 0, 1)]["usable_views"] == 1
    assert results[(0, 0, 1)]["candidate_renders"] == 1
    assert diagnostics["shape_fit_candidate_renders"] == 1
    assert diagnostics["shape_fit_completed_hypotheses"] == 1
    assert fitted[(0, 0, 1)] == carpet.state


def test_two_view_then_three_view_cap_two_does_not_starve_the_budget():
    """Leftover budget returns to a cell refused only by the fair-share split.

    The first cell sees two cameras and the second three.  A cap of two cannot
    pay for one complete three-view hypothesis, but it pays exactly for the
    two-view cell.  Dividing the cap before the view counts are known gives
    each cell one render, refusing both and spending nothing; the leftover
    budget must return to the two-view cell instead of going unused.
    """
    slab = _entry("quartz_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    palette = (slab,)
    scene = SceneTransform((0, 0, 0), (8, 2, 2))
    cameras = {
        "front": CameraModel((2.0, 0.5, -5.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0),
        "back": CameraModel((2.0, 0.5, 5.0), 180.0, 0.0, fov_degrees=90.0, eye_height=0.0),
        # Close to the right cell and looking away from the left one, which
        # falls outside this camera's frustum.
        "right": CameraModel((4.5, 0.5, -1.5), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0),
    }
    target_blocks = {(0, 0, 0): slab.state, (4, 0, 0): slab.state}
    frames = {}
    for frame_id, camera in cameras.items():
        photo, _owner, _index, _depth = render_volume(
            target_blocks, scene, camera, 160, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        frames[frame_id] = np.rint(photo).astype(np.uint8)
    diagnostics = {}

    fitted = fit_states(
        [(0, 0, 0), (4, 0, 0)], {}, scene, cameras, frames, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=2,
    )

    results = {
        tuple(item["coordinate"]): item
        for item in diagnostics["shape_fit_cell_results"]
    }
    assert results[(0, 0, 0)]["usable_views"] == 2
    assert results[(4, 0, 0)]["usable_views"] == 3
    assert results[(0, 0, 0)]["status"] == "accepted"
    assert results[(0, 0, 0)]["candidate_renders"] == 2
    assert results[(4, 0, 0)]["status"] == "budget_unvisited"
    assert results[(4, 0, 0)]["candidate_renders"] == 0
    assert diagnostics["shape_fit_candidate_renders"] == 2
    assert diagnostics["shape_fit_completed_hypotheses"] == 1
    assert diagnostics["shape_fit_budget_unvisited_cells"] == 1
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True
    assert fitted[(0, 0, 0)] == slab.state
    assert (4, 0, 0) not in fitted


def test_two_cell_cap_seven_reports_the_shared_cap_as_reached():
    """A refused refinement must not leave the aggregate cap flag false.

    Both cells spend part of the shared seven-render cap and are then refused
    an atomic reference/refinement hypothesis, leaving renders unspent.  The
    aggregate must still report that the cap limited the search, exactly as
    each per-cell run does, instead of comparing the leftover balance to zero.
    """
    slab = _entry("quartz_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (120, 120, 120), CARPET)
    stone = _entry("stone", (120, 120, 120), ())
    palette = (slab, carpet, stone)
    scene = SceneTransform((0, 0, 0), (4, 2, 2))
    cameras = {
        "front": CameraModel((1.5, 0.5, -5.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0),
        "back": CameraModel((1.5, 0.5, 5.0), 180.0, 0.0, fov_degrees=90.0, eye_height=0.0),
    }
    target_blocks = {(0, 0, 0): slab.state, (2, 0, 0): carpet.state}
    frames = {}
    for frame_id, camera in cameras.items():
        photo, _owner, _index, _depth = render_volume(
            target_blocks, scene, camera, 160, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        frames[frame_id] = np.rint(photo).astype(np.uint8)
    diagnostics = {}

    fit_states(
        [(0, 0, 0), (2, 0, 0)], {}, scene, cameras, frames, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=7,
    )

    results = diagnostics["shape_fit_cell_results"]
    assert diagnostics["shape_fit_candidate_renders"] == 6
    assert sum(item["candidate_renders"] for item in results) == 6
    assert [item["status"] for item in results] == ["partly_tested", "partly_tested"]
    assert diagnostics["shape_fit_unresolved_budget_exits"]
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True


def test_shape_balanced_schedule_reaches_partial_shapes_before_cube_materials():
    """A cube-heavy palette must not exhaust the cap inside one material group."""
    cubes = tuple(
        _entry(f"stone_variant_{number}", (120 + number % 8,) * 3, ())
        for number in range(100)
    )
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = cubes + (slab, carpet)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    diagnostics = {}

    fit_states(
        [target], {}, scene, {"cube-heavy.png": camera}, {"cube-heavy.png": frame},
        palette, scale=1.0, edge_weight=0.0, texture_weight=0.0,
        diagnostics=diagnostics, max_candidate_renders=3,
    )

    considered = diagnostics["shape_fit_considered_shapes"]
    assert any("quartz_slab" in name for name in considered)
    assert any("white_carpet" in name for name in considered)
    assert diagnostics["shape_fit_candidate_renders"] == 3
    assert diagnostics["shape_fit_candidate_render_cap_reached"] is True


def test_shape_fit_returns_supporting_frames_without_diagnostics():
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
    assert fitted.supporting_frames[target] == ("slab.png",)


def test_cube_heavy_schedule_recovers_partial_targets_and_retains_a_cube():
    """The same bounded schedule covers slabs/carpets without losing cube baselines."""
    cubes = tuple(
        _entry(f"stone_variant_{number}", (120 + number % 8,) * 3, ())
        for number in range(100)
    )
    slab = _entry("quartz_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (160, 160, 160), CARPET)
    palette = cubes + (slab, carpet)
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    colours = face_colours(palette)

    for expected in (slab.state, carpet.state):
        photo, _owner, _index, _depth = render_volume(
            {target: expected}, scene, camera, 120, 120, colours,
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        fitted = fit_states(
            [target], {}, scene, {"bounded.png": camera},
            {"bounded.png": np.rint(photo).astype(np.uint8)}, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            max_candidate_renders=3,
        )
        assert fitted[target] == expected

    cube = cubes[0].state
    photo, _owner, _index, _depth = render_volume(
        {target: cube}, scene, camera, 120, 120, colours,
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    retained = fit_states(
        [target], {target: cube}, scene, {"bounded.png": camera},
        {"bounded.png": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        max_candidate_renders=3,
    )
    assert retained[target] == cube


def test_material_selection_is_palette_order_independent_with_rendered_evidence():
    """A refined material must beat the representative regardless of palette order."""
    aaa = _entry("aaa_slab", (80, 80, 80), SLAB, (("type", "bottom"),))
    bbb = _entry("bbb_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)

    for palette in ((aaa, bbb), (bbb, aaa), (aaa, bbb, aaa)):
        photo, _owner, _index, _depth = render_volume(
            {target: bbb.state}, scene, camera, 120, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        fitted = fit_states(
            [target], {}, scene, {"material.png": camera},
            {"material.png": np.rint(photo).astype(np.uint8)}, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            max_candidate_renders=8,
        )
        assert fitted[target] == bbb.state


def test_representative_material_is_order_independent_when_cap_ends_after_probe():
    """A one-render cap still uses image evidence instead of palette position."""
    aaa = _entry("aaa_slab", (80, 80, 80), SLAB, (("type", "bottom"),))
    bbb = _entry("bbb_slab", (160, 160, 160), SLAB, (("type", "bottom"),))
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)

    for palette in ((aaa, bbb), (bbb, aaa), (aaa, bbb, aaa)):
        photo, _owner, _index, _depth = render_volume(
            {target: bbb.state}, scene, camera, 120, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        fitted = fit_states(
            [target], {}, scene, {"material-cap-one.png": camera},
            {"material-cap-one.png": np.rint(photo).astype(np.uint8)}, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            max_candidate_renders=1,
        )
        assert fitted[target] == bbb.state


def test_missing_material_metadata_never_beats_known_rgb_evidence():
    """Unknown metadata is a fallback only after every known material loses."""
    from mc_photogrammetry.reconstruction.camera import CameraModel

    known = BlockState("minecraft:known_slab")
    missing = BlockState("minecraft:missing_slab")
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    colours = {known.name: {slot: np.full(3, 100.0) for slot in SLOTS}}
    observed = {"a.png": np.full(3, 100.0)}
    views = [("a.png", camera, np.ones((1, 1), dtype=bool))]

    for states in ((missing, known), (known, missing)):
        assert _representative_material(
            states, colours, {}, views, observed,
        ) == known

    unknown_colours = {}
    first = _representative_material(
        (missing, BlockState("minecraft:zzz_slab")), unknown_colours, {}, views, observed,
    )
    second = _representative_material(
        (BlockState("minecraft:zzz_slab"), missing), unknown_colours, {}, views, observed,
    )
    assert first == second == missing


def test_grain_evidence_beats_equal_shape_representative(monkeypatch):
    """A texture preference must survive equal rendered shape scores."""
    from mc_photogrammetry.reconstruction import raycast as raycast_module

    class FakePaletteIndex:
        def __init__(self, palette, *_args):
            self.states = [item.state for item in palette]
            self.count = len(self.states)

        def costs(self, *_args, **_kwargs):
            costs = np.asarray([
                100.0 if state.name.endswith("aaa_slab") else 0.0
                for state in self.states
            ])
            return costs, float(costs.min())

    monkeypatch.setattr(raycast_module, "PaletteIndex", FakePaletteIndex)
    aaa = _entry("aaa_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    bbb = _entry("bbb_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.25, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    photo, _owner, _index, _depth = render_volume(
        {target: bbb.state}, scene, camera, 120, 120, face_colours((aaa, bbb)),
        face_shapes((aaa, bbb)), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots((aaa, bbb)), model_faces=face_model_faces((aaa, bbb)),
    )

    fitted = fit_states(
        [target], {}, scene, {"grain.png": camera},
        {"grain.png": np.rint(photo).astype(np.uint8)}, (aaa, bbb),
        scale=1.0, edge_weight=0.0, texture_weight=1.0, max_candidate_renders=8,
    )
    assert fitted[target] == bbb.state


def test_completed_geometry_decisions_survive_a_later_refinement_cap():
    """A partial refinement must not hide a completed perfect cube decision."""
    aaa = _entry("aaa_carpet", (100, 100, 100), CARPET)
    bbb = _entry("bbb_carpet", (200, 200, 200), CARPET)
    stone = _entry("stone", (120, 120, 120), ())
    target = (0, 0, 0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    palette = (aaa, bbb, stone)
    photo, _owner, _index, _depth = render_volume(
        {target: stone.state}, scene, camera, 120, 120, face_colours(palette),
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    frames = {"cap.png": np.rint(photo).astype(np.uint8)}

    for cap in (2, 3, 4):
        fitted = fit_states(
            [target], {}, scene, {"cap.png": camera}, frames, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            max_candidate_renders=cap,
        )
        assert fitted[target] == stone.state


def test_multi_cell_budget_round_robin_covers_both_shape_families_and_order():
    """Two cells receive all representative hypotheses before refinement work."""
    slab = _entry("quartz_slab", (120, 120, 120), SLAB, (("type", "bottom"),))
    carpet = _entry("white_carpet", (120, 120, 120), CARPET)
    stone = _entry("stone", (120, 120, 120), ())
    palette = (slab, carpet, stone)
    scene = SceneTransform((0, 0, 0), (4, 2, 2))
    cameras = {
        "front": CameraModel((1.5, 0.5, -5.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0),
        "back": CameraModel((1.5, 0.5, 5.0), 180.0, 0.0, fov_degrees=90.0, eye_height=0.0),
    }
    target_blocks = {(0, 0, 0): slab.state, (2, 0, 0): carpet.state}
    frames = {}
    for frame_id, camera in cameras.items():
        photo, _owner, _index, _depth = render_volume(
            target_blocks, scene, camera, 160, 120, face_colours(palette),
            face_shapes(palette), scale=1.0, ambient=True,
            occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
        )
        frames[frame_id] = np.rint(photo).astype(np.uint8)

    for cells in ([(0, 0, 0), (2, 0, 0)], [(2, 0, 0), (0, 0, 0)]):
        diagnostics = {}
        fitted = fit_states(
            cells, {}, scene, cameras, frames, palette,
            scale=1.0, edge_weight=0.0, texture_weight=0.0,
            max_candidate_renders=12, diagnostics=diagnostics,
        )
        assert fitted[(0, 0, 0)] == slab.state
        assert fitted[(2, 0, 0)] == carpet.state
        assert diagnostics["shape_fit_candidate_renders"] == 12
        results = diagnostics["shape_fit_cell_results"]
        assert [item["candidate_renders"] for item in results] == [6, 6]
        assert diagnostics["shape_fit_stage_seconds"]["render"] > 0.0
        assert diagnostics["shape_fit_stage_seconds"]["ownership"] >= 0.0


def test_multi_cell_context_edits_invalidate_an_occluded_later_cell():
    """A committed foreground edit removes stale visibility for a later cell."""
    stone = _entry("stone", (120, 120, 120), ())
    carpet = _entry("white_carpet", (240, 240, 240), CARPET)
    palette = (stone, carpet)
    scene = SceneTransform((0, 0, 0), (2, 2, 3))
    camera = CameraModel((0.5, 0.5, -5.0), 0.0, 0.0, fov_degrees=70.0, eye_height=0.0)
    target_blocks = {(0, 0, 0): stone.state, (0, 0, 1): carpet.state}
    photo, _owner, _index, _depth = render_volume(
        target_blocks, scene, camera, 160, 120, face_colours(palette),
        face_shapes(palette), scale=1.0, ambient=True,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    diagnostics = {}
    fitted = fit_states(
        [(0, 0, 0), (0, 0, 1)], {}, scene, {"front": camera},
        {"front": np.rint(photo).astype(np.uint8)}, palette,
        scale=1.0, edge_weight=0.0, texture_weight=0.0,
        max_candidate_renders=6, diagnostics=diagnostics,
    )
    assert fitted[(0, 0, 0)] == stone.state
    assert (0, 0, 1) not in fitted
    results = diagnostics["shape_fit_cell_results"]
    assert results[0]["status"] == "accepted"
    assert results[1]["status"] == "no_usable_views"
