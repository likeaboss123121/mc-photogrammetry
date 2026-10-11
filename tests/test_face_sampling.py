import base64
from io import BytesIO

import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import (
    box_face_corners,
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_result,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.face_sampling import (
    CUBE_GEOMETRY,
    DEPTH_SCALE,
    EXACT_MODEL,
    MODEL_METADATA_UNAVAILABLE,
    PATCH_SIZE,
    PASS_SURFACE,
    PASS_SURFACE_PROPOSED,
    STATES_INCOMPLETE,
    STATES_UNAVAILABLE,
    SamplingGeometryDecision,
    _nearest_surface,
    _owner_pixels_by_face,
    default_face_corners,
    exact_sampling_blocks,
    quad_area,
    rectify_face,
    rectify_face_valid,
    resolve_sampling_geometry,
    sample_visible_faces,
    validate_sampling_states,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.vision.gui_mask import create_gui_mask


def _striped_image(height, width, period=8):
    """Horizontal stripes, so averaging over them is visibly different from sampling one."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    rows = (np.arange(height) // period) % 2
    image[rows == 0] = 240
    image[rows == 1] = 20
    return image


def test_rectify_face_area_averages_instead_of_point_sampling():
    # A large striped face must average to mid grey. Point sampling would land on
    # individual stripes and return near-black or near-white noise instead.
    image = _striped_image(600, 600, period=8)
    quad = np.asarray([[50, 50], [550, 50], [550, 550], [50, 550]], dtype=np.float32)

    patch = rectify_face(image, quad)

    assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert 100.0 < float(patch.mean()) < 160.0
    assert float(patch.std()) < 40.0


def test_face_sampling_defaults_to_source_pixel_depth_ownership():
    assert DEPTH_SCALE == 1.0


def test_rectify_face_preserves_a_pattern_it_can_resolve():
    # Eight stripes across sixteen output rows stays resolvable; a finer period
    # would alias to flat grey no matter how faithful the resampling is.
    image = _striped_image(320, 320, period=40)
    quad = np.asarray([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32)

    patch = rectify_face(image, quad).mean(axis=2)

    assert patch.max() - patch.min() > 150.0


def test_rectify_face_rejects_malformed_input():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        rectify_face(image, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        rectify_face(image, np.zeros((4, 2), dtype=np.float32), size=1)


def test_rectify_face_valid_divides_rgb_by_surviving_coverage():
    """Invalid black pixels must not darken a valid resampled face."""
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    valid = np.zeros((128, 128), dtype=bool)
    valid[:, ::8] = True
    image[valid] = 160
    quad = np.asarray([[0, 0], [128, 0], [128, 128], [0, 128]], dtype=np.float32)

    patch, output_valid, coverage = rectify_face_valid(image, quad, valid)

    assert output_valid.all()
    assert np.allclose(patch, 160.0, atol=1.0)
    assert float(coverage.mean()) == pytest.approx(0.125, abs=0.01)


def test_rectify_face_valid_is_invariant_to_masked_source_colour():
    valid = np.zeros((64, 64), dtype=bool)
    valid[:, ::4] = True
    first = np.zeros((64, 64, 3), dtype=np.uint8)
    second = np.full((64, 64, 3), 17, dtype=np.uint8)
    first[valid] = second[valid] = (160, 120, 80)
    quad = np.asarray([[0.25, 0.25], [63.25, 0.25], [63.25, 63.25], [0.25, 63.25]], dtype=np.float32)

    first_patch, first_valid, first_coverage = rectify_face_valid(first, quad, valid)
    second_patch, second_valid, second_coverage = rectify_face_valid(second, quad, valid)

    assert np.array_equal(first_valid, second_valid)
    assert np.allclose(first_coverage, second_coverage)
    assert np.allclose(first_patch[first_valid], second_patch[second_valid])


def test_rectify_face_valid_keeps_the_observed_part_at_an_image_edge():
    image = np.full((128, 128, 3), 160, dtype=np.uint8)
    valid = np.ones((128, 128), dtype=bool)
    quad = np.asarray([[-32, 0], [96, 0], [96, 128], [-32, 128]], dtype=np.float32)

    patch, output_valid, coverage = rectify_face_valid(image, quad, valid)

    assert output_valid.any() and not output_valid.all()
    assert np.allclose(patch[output_valid], 160.0, atol=1.0)
    assert float(coverage[output_valid].min()) > 0.01


def test_default_face_corners_lie_on_their_own_face():
    for slot, axis, sign in (("up", 1, 0.5), ("down", 1, -0.5), ("east", 0, 0.5), ("west", 0, -0.5)):
        corners = default_face_corners(slot)
        assert corners.shape == (4, 3)
        assert np.allclose(corners[:, axis], sign)


def test_sample_visible_faces_returns_only_unoccluded_front_faces():
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    camera = CameraModel((2.0, 2.0, -6.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (4, 4, 4))
    # Two blocks in a line away from the camera; the far one hides behind the near one.
    coordinates = {(2, 2, 1), (2, 2, 2)}

    samples = sample_visible_faces(coordinates, camera, scene, image, minimum_area=1.0)

    slots = {(sample.coordinate, sample.slot) for sample in samples}
    assert ((2, 2, 1), "north") in slots
    # The shared boundary between the two blocks is never sampled from either side.
    assert ((2, 2, 2), "north") not in slots
    assert ((2, 2, 1), "south") not in slots
    assert all(sample.patch.shape == (PATCH_SIZE, PATCH_SIZE, 3) for sample in samples)


def test_sample_visible_faces_skips_faces_under_the_hud():
    image = np.full((400, 400, 3), 128, dtype=np.uint8)
    camera = CameraModel((2.0, 2.0, -6.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (4, 4, 4))
    coordinates = {(2, 2, 1)}

    without_mask = sample_visible_faces(coordinates, camera, scene, image, minimum_area=1.0)
    covered = np.ones((400, 400), dtype=bool)
    with_mask = sample_visible_faces(
        coordinates, camera, scene, image, gui_mask=covered, minimum_area=1.0
    )

    assert without_mask
    assert with_mask == []


def test_sample_visible_faces_keeps_the_unmasked_part_of_a_hud_overlapped_face():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (2, 2, 2))
    mask = np.zeros((100, 100), dtype=bool)
    mask[42:44, 42:59] = True
    image[mask] = 0
    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, scene, image, gui_mask=mask, minimum_area=1.0, shrink=1.0
    )

    assert samples
    north = next(sample for sample in samples if sample.slot == "north")
    assert north.valid is not None and not north.valid.all()
    assert north.pixels.shape[0] < north.patch.shape[0] * north.patch.shape[1]
    assert np.allclose(north.rgb, 128.0, atol=1.0)


def test_sample_visible_faces_uses_exact_alpha_tested_model_ownership():
    """A model cut-out must not become sampling evidence just because its cell is occupied."""
    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[:, 8:] = 255
    encoded_buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(encoded_buffer, format="PNG")
    encoded = base64.b64encode(encoded_buffer.getvalue()).decode("ascii")
    state = BlockState("minecraft:cutout")
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    palette = (
        PaletteColor(state, (160, 160, 160), model_faces=(
            ("north", corners, (0, 0, -1), None, encoded),
        )),
    )
    image = np.full((100, 100, 3), 160, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)

    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, SceneTransform((0, 0, 0), (2, 2, 2)), image,
        blocks={(0, 0, 0): state}, palette=palette, minimum_area=1.0,
        minimum_visible_fraction=0.1, shrink=1.0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.slot == "north"
    assert sample.valid is not None and sample.valid.any() and not sample.valid.all()
    assert np.allclose(sample.patch[sample.valid], 160.0, atol=1.0)


def test_sample_visible_faces_rectifies_the_front_part_of_a_near_plane_face():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -0.1), 60.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    rejections = {}
    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, SceneTransform((0, 0, 0), (2, 2, 2)), image,
        minimum_area=1.0, shrink=1.0, rejections=rejections,
    )

    assert any(sample.slot == "north" and sample.pixels.size for sample in samples)
    assert rejections.get("behind_camera", 0) == 0


def test_quad_area_matches_a_known_rectangle():
    quad = np.asarray([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float32)
    assert quad_area(quad) == pytest.approx(40.0)


def test_gui_mask_covers_the_held_item_corner():
    mask = create_gui_mask(1000, 2000)
    assert bool(mask[930, 1900])
    # Well clear of the HUD, the crosshair and the held item.
    assert not bool(mask[300, 1100])


def _cube_scene():
    state = BlockState("minecraft:stone")
    palette = (PaletteColor(state, (125, 125, 125)),)
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    return state, palette, camera, SceneTransform((0, 0, 0), (2, 2, 2))


def _nearest_entry():
    """One synthetic face spanning a known rectangle on a 101x99 frame."""
    quad = np.asarray([[10.0, 10.0], [90.0, 10.0], [90.0, 80.0], [10.0, 80.0]], dtype=np.float32)
    return (5.0, quad, np.full(4, 5.0), (0, 0, 0), "north", 1.0, quad)


@pytest.mark.parametrize(
    "scale, expected",
    [(1.0, (99, 101)), (0.75, (74, 75)), (0.5, (49, 50)), (0.4, (39, 40))],
)
def test_fallback_ownership_buffer_truncates_like_the_renderer(scale, expected):
    """Round would give 40 rows at scale 0.4 and 50 at 0.5, disagreeing with the renderer."""
    owner = _nearest_surface([_nearest_entry()], 101, 99, scale)

    state, palette, camera, scene = _cube_scene()
    rendered, _index = render_result(
        {(0, 0, 0): state}, scene, camera, 101, 99, face_colours(palette), face_shapes(palette),
        scale=scale, occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )

    assert owner.shape == expected
    assert owner.shape == rendered.ownership.shape


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.4])
def test_fractional_scale_keeps_fallback_and_exact_face_sets_in_step(scale):
    """The cube fallback and the exact renderer must accept the same faces at a shared scale."""
    image = np.full((99, 101, 3), 128, dtype=np.uint8)
    state, palette, camera, scene = _cube_scene()
    coordinates = {(0, 0, 0)}

    fallback = sample_visible_faces(
        coordinates, camera, scene, image, minimum_area=1.0, shrink=1.0, depth_scale=scale
    )
    exact = sample_visible_faces(
        coordinates, camera, scene, image, blocks={(0, 0, 0): state}, palette=palette,
        minimum_area=1.0, minimum_visible_fraction=0.1, shrink=1.0, depth_scale=scale,
    )

    fallback_faces = {(sample.coordinate, sample.slot) for sample in fallback}
    exact_faces = {(sample.coordinate, sample.slot) for sample in exact}
    assert fallback_faces == exact_faces
    assert fallback_faces == {((0, 0, 0), "north")}


def test_exact_missing_owner_fallback_uses_the_rendered_ownership_shape(monkeypatch):
    """A face the renderer culled must still carry the renderer's buffer shape, not a rounded one."""
    import cv2

    width, height, scale = 79, 99, 0.5
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    state, palette, _camera, scene = _cube_scene()
    # Camera off the near block's shared wall: the renderer culls that face, while
    # sampling still enumerates it and falls back to an all-false owner.
    camera = CameraModel((4.0, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    blocks = {(0, 0, 0): state, (1, 0, 0): state}
    rendered, _index = render_result(
        blocks, scene, camera, width, height, face_colours(palette), face_shapes(palette),
        scale=scale, occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    sources = []
    real_resize = cv2.resize

    def record(source, size, *args, **kwargs):
        sources.append((tuple(source.shape), tuple(size)))
        return real_resize(source, size, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", record)
    sample_visible_faces(
        set(blocks), camera, scene, image, blocks=blocks, palette=palette,
        minimum_area=1.0, minimum_visible_fraction=0.0, shrink=1.0, depth_scale=scale,
    )

    # Face patches are resampled to (PATCH_SIZE, PATCH_SIZE); every ownership
    # array resized up to the frame must come from the renderer's buffer.
    ownership_sources = {source for source, size in sources if size != (PATCH_SIZE, PATCH_SIZE)}
    assert ownership_sources == {rendered.ownership.shape}


@pytest.mark.parametrize("scale", [0.0, -0.25, 1.5])
def test_sample_visible_faces_rejects_an_out_of_range_depth_scale(scale):
    """Both ownership paths need a scale within (0, 1], so it is refused up front."""
    image = np.full((99, 101, 3), 128, dtype=np.uint8)
    state, palette, camera, scene = _cube_scene()

    with pytest.raises(ValueError):
        sample_visible_faces(
            {(0, 0, 0)}, camera, scene, image, minimum_area=1.0, depth_scale=scale
        )
    with pytest.raises(ValueError):
        sample_visible_faces(
            {(0, 0, 0)}, camera, scene, image, blocks={(0, 0, 0): state}, palette=palette,
            minimum_area=1.0, depth_scale=scale,
        )


# ---------------------------------------------------------------------------
# Geometry resolution: the accepted sampling-states contract.
#
# A call may use exact baked-state geometry only when every one of *its own*
# targets has a supplied state carrying exact model polygons in the palette.
# Anything less keeps the whole call on the cube geometry and names the missing
# prerequisite. Nothing here invents a state, falls back to a name, chooses the
# first palette entry, mixes the two depth buffers, or drops a target.


def _encoded_alpha(columns=slice(0, 8)):
    """A transparent cut-out face texture; opaque only in ``columns``."""
    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[:, columns] = 255
    buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _modelled_state(name="minecraft:modelled"):
    return BlockState(name)


def _sampling_palette():
    """A modelled state plus a plain one, so metadata is controllably missing."""
    modelled = _modelled_state()
    plain = BlockState("minecraft:plain")
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    palette = (
        PaletteColor(
            modelled, (250, 250, 250),
            model_faces=(("north", corners, (0, 0, -1), None, _encoded_alpha(slice(0, 4))),),
        ),
        PaletteColor(plain, (20, 20, 20)),
    )
    return modelled, plain, palette


def test_validate_sampling_states_accepts_none_and_normalises_coordinates():
    state = BlockState("minecraft:stone")

    assert validate_sampling_states(None) is None
    validated = validate_sampling_states({(0, 0, 0): state, (2, 3, 4): state})

    assert validated == {(0, 0, 0): state, (2, 3, 4): state}
    assert type(validated) is dict


@pytest.mark.parametrize("bad", [[], (), "states", 7, [((0, 0, 0), BlockState("minecraft:stone"))]])
def test_validate_sampling_states_rejects_a_non_mapping(bad):
    with pytest.raises(TypeError):
        validate_sampling_states(bad)


@pytest.mark.parametrize(
    "coordinate",
    [(0, 0), (0, 0, 0, 0), (0.0, 0, 0), ("0", 0, 0), (True, 0, 0), "000", 7, None],
)
def test_validate_sampling_states_rejects_malformed_coordinates(coordinate):
    with pytest.raises(ValueError):
        validate_sampling_states({coordinate: BlockState("minecraft:stone")})


def test_validate_sampling_states_rejects_a_non_state_value():
    with pytest.raises(TypeError):
        validate_sampling_states({(0, 0, 0): "minecraft:stone"})


def test_resolve_without_a_snapshot_stays_on_cube_and_reports_unavailable():
    _modelled, _plain, palette = _sampling_palette()

    decision = resolve_sampling_geometry(
        {(0, 0, 0), (1, 0, 0)}, None, palette, PASS_SURFACE)

    assert decision == SamplingGeometryDecision(
        pass_name=PASS_SURFACE,
        mode=CUBE_GEOMETRY,
        target_count=2,
        missing_state_count=2,
        missing_model_count=0,
        fallback_reason=STATES_UNAVAILABLE,
    )


def test_resolve_with_an_empty_snapshot_is_incomplete_not_unavailable():
    _modelled, _plain, palette = _sampling_palette()

    decision = resolve_sampling_geometry({(0, 0, 0)}, {}, palette, PASS_SURFACE)

    assert decision.mode == CUBE_GEOMETRY
    assert decision.fallback_reason == STATES_INCOMPLETE
    assert decision.missing_state_count == 1


def test_resolve_falls_back_when_a_target_has_no_state():
    modelled, _plain, palette = _sampling_palette()

    decision = resolve_sampling_geometry(
        {(0, 0, 0), (1, 0, 0)}, {(0, 0, 0): modelled}, palette, PASS_SURFACE_PROPOSED)

    assert decision.mode == CUBE_GEOMETRY
    assert decision.fallback_reason == STATES_INCOMPLETE
    assert (decision.target_count, decision.missing_state_count, decision.missing_model_count) == (2, 1, 0)


def test_resolve_falls_back_when_a_state_has_no_exact_model_metadata():
    modelled, plain, palette = _sampling_palette()

    decision = resolve_sampling_geometry(
        {(0, 0, 0), (1, 0, 0)}, {(0, 0, 0): modelled, (1, 0, 0): plain},
        palette, PASS_SURFACE_PROPOSED)

    assert decision.mode == CUBE_GEOMETRY
    assert decision.fallback_reason == MODEL_METADATA_UNAVAILABLE
    assert (decision.target_count, decision.missing_state_count, decision.missing_model_count) == (2, 0, 1)


def test_resolve_selects_exact_model_and_ignores_entries_outside_the_targets():
    modelled, plain, palette = _sampling_palette()

    decision = resolve_sampling_geometry(
        {(0, 0, 0)},
        {(0, 0, 0): modelled, (7, 7, 7): plain},
        palette,
        PASS_SURFACE,
    )

    assert decision.mode == EXACT_MODEL
    assert decision.fallback_reason is None
    assert (
        decision.target_count,
        decision.missing_state_count,
        decision.missing_model_count,
    ) == (1, 0, 0)


def test_resolve_does_not_accept_a_name_only_model_match():
    """A sibling state sharing a name must not lend another state's geometry."""
    _modelled, _plain, palette = _sampling_palette()
    sibling = BlockState("minecraft:modelled", (("variant", "other"),))

    decision = resolve_sampling_geometry(
        {(0, 0, 0)}, {(0, 0, 0): sibling}, palette, PASS_SURFACE)

    assert decision.mode == CUBE_GEOMETRY
    assert decision.fallback_reason == MODEL_METADATA_UNAVAILABLE


def test_resolve_is_a_frozen_typed_record():
    import dataclasses

    _modelled, _plain, palette = _sampling_palette()
    decision = resolve_sampling_geometry({(0, 0, 0)}, None, palette, PASS_SURFACE)

    assert isinstance(decision, SamplingGeometryDecision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.mode = EXACT_MODEL


def test_decision_to_record_is_primitive_and_json_round_trips():
    """Diagnostics hold records, so only JSON primitives may leave a decision."""
    import json

    modelled, plain, palette = _sampling_palette()
    decisions = (
        resolve_sampling_geometry({(0, 0, 0)}, None, palette, PASS_SURFACE),
        resolve_sampling_geometry({(0, 0, 0)}, {}, palette, PASS_SURFACE),
        resolve_sampling_geometry(
            {(0, 0, 0), (1, 0, 0)}, {(0, 0, 0): modelled}, palette, PASS_SURFACE_PROPOSED),
        resolve_sampling_geometry(
            {(0, 0, 0), (1, 0, 0)}, {(0, 0, 0): modelled, (1, 0, 0): plain},
            palette, PASS_SURFACE_PROPOSED),
        resolve_sampling_geometry(
            {(0, 0, 0)}, {(0, 0, 0): modelled}, palette, PASS_SURFACE),
    )
    expected_reasons = (
        STATES_UNAVAILABLE,
        STATES_INCOMPLETE,
        STATES_INCOMPLETE,
        MODEL_METADATA_UNAVAILABLE,
        None,
    )
    expected_keys = {
        "pass_name",
        "mode",
        "target_count",
        "missing_state_count",
        "missing_model_count",
        "fallback_reason",
    }

    for decision, reason in zip(decisions, expected_reasons):
        record = decision.to_record()

        assert set(record) == expected_keys
        assert record == {
            "pass_name": decision.pass_name,
            "mode": decision.mode,
            "target_count": decision.target_count,
            "missing_state_count": decision.missing_state_count,
            "missing_model_count": decision.missing_model_count,
            "fallback_reason": reason,
        }
        # Only str/int/None: no live decision, numpy count or bool survives.
        assert all(
            value is None or type(value) in (str, int) for value in record.values()
        )
        assert json.loads(json.dumps(record, sort_keys=True)) == record


def test_exact_sampling_blocks_is_none_for_cube_and_covers_every_target_for_exact():
    modelled, plain, palette = _sampling_palette()
    targets = {(0, 0, 0), (1, 0, 0)}

    cube = resolve_sampling_geometry(targets, None, palette, PASS_SURFACE)
    assert exact_sampling_blocks(cube, targets, None) is None

    exact = resolve_sampling_geometry(
        targets, {(0, 0, 0): modelled, (1, 0, 0): modelled}, palette, PASS_SURFACE)
    assert exact.mode == EXACT_MODEL
    assert exact_sampling_blocks(exact, targets, {(0, 0, 0): modelled, (1, 0, 0): modelled}) == {
        (0, 0, 0): modelled,
        (1, 0, 0): modelled,
    }


# ---------------------------------------------------------------------------
# Exact-model sampling through the real renderer: cut-out ownership and the
# transparency/coverage invariance the contract requires.


def _cutout_scene():
    """A single full-cube cell whose north face is a transparent cut-out.

    The model keeps only its rightmost quarter on screen; the rest of the face is
    transparent, so the pixels there belong to whatever stands behind it. Those
    pixels must not enter the sample.
    """
    state = BlockState("minecraft:cutout")
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    palette = (
        PaletteColor(
            state, (250, 250, 250),
            model_faces=(("north", corners, (0, 0, -1), None, _encoded_alpha(slice(0, 4))),),
        ),
    )
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    return state, palette, camera, SceneTransform((0, 0, 0), (1, 1, 1))


def _cutout_frame(background: int) -> np.ndarray:
    """Opaque quarter at screen right in the block's face colour, rest background."""
    image = np.full((128, 128, 3), background, dtype=np.uint8)
    image[:, 70:] = 200
    return image


def test_exact_model_cutout_excludes_transparent_pixels_and_keeps_coverage():
    state, palette, camera, scene = _cutout_scene()

    exact = [
        sample_visible_faces(
            {(0, 0, 0)}, camera, scene, _cutout_frame(background), blocks={(0, 0, 0): state},
            palette=palette, minimum_area=1.0, minimum_visible_fraction=0.1, shrink=1.0,
        )[0]
        for background in (20, 100)
    ]
    cube = [
        sample_visible_faces(
            {(0, 0, 0)}, camera, scene, _cutout_frame(background),
            minimum_area=1.0, minimum_visible_fraction=0.1, shrink=1.0,
        )[0]
        for background in (20, 100)
    ]

    # Exact ownership is fixed by the cut-out, so the transparent background is
    # invisible to the sample: same owned pixels, same colour.
    assert exact[0].valid is not None
    assert np.array_equal(exact[0].valid, exact[1].valid)
    assert np.array_equal(exact[0].coverage, exact[1].coverage)
    assert np.allclose(exact[0].rgb, exact[1].rgb)
    assert np.allclose(exact[0].rgb, 200.0, atol=1.0)

    # Cube ownership spans the whole face, so the same pixels change the sample.
    assert not np.allclose(cube[0].rgb, cube[1].rgb)
    assert int(exact[0].valid.sum()) < int(cube[0].valid.sum())  # type: ignore[union-attr]


def test_exact_model_cutout_honours_a_masked_owned_pixel_domain():
    state, palette, camera, scene = _cutout_scene()
    mask = np.zeros((128, 128), dtype=bool)
    # Mask the lower half of the frame, which removes the lower half of the owned
    # quarter but leaves its upper half as evidence.
    mask[64:, :] = True

    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, scene, _cutout_frame(20), gui_mask=mask,
        blocks={(0, 0, 0): state}, palette=palette,
        minimum_area=1.0, minimum_visible_fraction=0.1, shrink=1.0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.valid is not None and sample.valid.any()
    # Masked pixels are excluded; the surviving owned pixels are still the
    # block's colour, not the background.
    assert np.allclose(sample.patch[sample.valid], 200.0, atol=1.0)


# ---------------------------------------------------------------------------
# Sparse owner index: the exact path used to retain one H x W boolean mask per
# rendered face, which reached tens of gigabytes on a full factory scene. The
# index stores each owner's pixel coordinates instead, must aggregate several
# renderer identifiers that share one (coordinate, slot), and must reconstruct
# exactly the masks the old dense ``logical_or`` produced.


def _dense_owner_mask(shape, pixels):
    mask = np.zeros(shape, dtype=bool)
    if pixels is not None:
        mask.reshape(-1)[pixels] = True
    return mask


def test_owner_pixels_by_face_aggregates_duplicate_face_ids():
    ownership = np.asarray([[0, 0, 1], [2, 2, 1]], dtype=np.int64)
    face_index = [((0, 0, 0), "north"), ((0, 0, 0), "north"), ((1, 0, 0), "east")]

    pixels = _owner_pixels_by_face(ownership, face_index)

    assert set(pixels) == {((0, 0, 0), "north"), ((1, 0, 0), "east")}
    # Identifiers 0 and 1 share the north key, so their disjoint pixels union.
    assert np.array_equal(pixels[((0, 0, 0), "north")], np.asarray([0, 1, 2, 5]))
    assert np.array_equal(pixels[((1, 0, 0), "east")], np.asarray([3, 4]))


def test_owner_pixels_by_face_matches_full_frame_logical_or_masks():
    # A fixed buffer with background pixels and several identifiers sharing one
    # owner key, so the union is genuinely exercised rather than vacuous.
    ownership = np.full((5, 7), -1, dtype=np.int64)
    ownership[0, 0:3] = 0
    ownership[0, 3:5] = 1
    ownership[1, :] = 2
    ownership[2, 0:2] = 1
    ownership[2, 5:] = 3
    ownership[3, 1:6] = 4
    ownership[4, 2:4] = 5
    face_index = [
        ((0, 0, 0), "north"), ((0, 0, 0), "north"), ((1, 0, 0), "east"),
        ((0, 0, 0), "north"), ((2, 0, 0), "up"), ((3, 0, 0), "west"),
    ]

    pixels = _owner_pixels_by_face(ownership, face_index)

    assert set(pixels) <= set(face_index)
    for key in set(face_index):
        dense = np.zeros(ownership.shape, dtype=bool)
        for identifier, owned in enumerate(face_index):
            if owned == key:
                dense |= ownership == identifier
        assert np.array_equal(_dense_owner_mask(ownership.shape, pixels.get(key)), dense)
    # The north key is split across three identifiers that all own pixels, so
    # the aggregated mask is a real union rather than one identifier's mask.
    split = [key for key in set(face_index) if face_index.count(key) > 1]
    assert split
    for key in split:
        identifiers = [index for index, owned in enumerate(face_index) if owned == key]
        per_identifier = [ownership == identifier for identifier in identifiers]
        assert all(bool(mask.any()) for mask in per_identifier)
        dense = _dense_owner_mask(ownership.shape, pixels[key])
        assert all(not np.array_equal(dense, mask) for mask in per_identifier)


def test_owner_pixels_by_face_drops_background_pixels():
    ownership = np.full((4, 5), -1, dtype=np.int64)
    face_index = [((0, 0, 0), "north")]

    assert _owner_pixels_by_face(ownership, face_index) == {}


def _duplicate_north_scene():
    """One cell whose exact model exposes two north faces under one owner key."""
    state = BlockState("minecraft:double_north")
    local = box_face_corners((0, 0, 0), (1, 1, 1), "north")
    offset = local + np.asarray([1.0, 0.0, 0.0])
    records = tuple(
        (
            "north",
            tuple(tuple(float(value) for value in point) for point in corners),
            (0.0, 0.0, -1.0),
            None,
            None,
        )
        for corners in (local, offset)
    )
    palette = (PaletteColor(state, (210, 210, 210), model_faces=records),)
    camera = CameraModel((0.5, 0.5, -4.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    return state, palette, camera, SceneTransform((0, 0, 0), (2, 2, 2))


def test_exact_sampling_aggregates_duplicate_owner_ids_into_one_mask(monkeypatch):
    import mc_photogrammetry.reconstruction.face_sampling as face_sampling

    state, palette, camera, scene = _duplicate_north_scene()
    width, height = 128, 128
    image = np.full((height, width, 3), 160, dtype=np.uint8)
    rendered, face_index = render_result(
        {(0, 0, 0): state}, scene, camera, width, height,
        face_colours(palette), face_shapes(palette),
        scale=DEPTH_SCALE,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    owner_pixels = _owner_pixels_by_face(rendered.ownership, face_index)
    expected = _dense_owner_mask(rendered.ownership.shape, owner_pixels[((0, 0, 0), "north")])
    first = rendered.ownership == 0
    second = rendered.ownership == 1
    assert first.any() and second.any()
    assert not np.array_equal(expected, first) and not np.array_equal(expected, second)

    captured = []
    real_rectify = face_sampling.rectify_face_valid

    def record(image_arg, quad, source_valid, *args, **kwargs):
        captured.append(source_valid.copy())
        return real_rectify(image_arg, quad, source_valid, *args, **kwargs)

    monkeypatch.setattr(face_sampling, "rectify_face_valid", record)
    samples = sample_visible_faces(
        {(0, 0, 0)}, camera, scene, image, blocks={(0, 0, 0): state}, palette=palette,
        minimum_area=1.0, minimum_visible_fraction=0.0, shrink=1.0,
    )

    assert len(samples) == 2
    assert len(captured) == 2
    for mask in captured:
        assert np.array_equal(mask, expected)


def test_exact_owner_scratch_does_not_bleed_across_missing_owners(monkeypatch):
    import mc_photogrammetry.reconstruction.face_sampling as face_sampling

    width, height = 79, 99
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    state, palette, _camera, scene = _cube_scene()
    camera = CameraModel((4.0, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    blocks = {(0, 0, 0): state, (1, 0, 0): state}
    rendered, face_index = render_result(
        blocks, scene, camera, width, height, face_colours(palette), face_shapes(palette),
        scale=DEPTH_SCALE,
        occluding_slots=face_occluding_slots(palette), model_faces=face_model_faces(palette),
    )
    owner_pixels = _owner_pixels_by_face(rendered.ownership, face_index)
    accepted_masks = {
        _dense_owner_mask(rendered.ownership.shape, pixels).tobytes()
        for pixels in owner_pixels.values()
    }
    empty = np.zeros(rendered.ownership.shape, dtype=bool).tobytes()

    captured = []
    real_rectify = face_sampling.rectify_face_valid

    def record(image_arg, quad, source_valid, *args, **kwargs):
        captured.append(source_valid.copy())
        return real_rectify(image_arg, quad, source_valid, *args, **kwargs)

    monkeypatch.setattr(face_sampling, "rectify_face_valid", record)
    sample_visible_faces(
        set(blocks), camera, scene, image, blocks=blocks, palette=palette,
        minimum_area=1.0, minimum_visible_fraction=0.0, shrink=1.0,
    )

    assert captured
    # The renderer culls the shared wall while sampling still enumerates it, so
    # at least one face reaches the rectifier with no owner at all. It must read
    # all-false rather than inherit the previous face's scratch contents.
    assert any(not mask.any() for mask in captured)
    assert all(mask.shape == rendered.ownership.shape for mask in captured)
    for mask in captured:
        assert mask.tobytes() == empty or mask.tobytes() in accepted_masks
