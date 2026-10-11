from __future__ import annotations

import base64
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from mc_photogrammetry.dataset.schemas import BlockRecord, GroundTruthExport, RegionBounds
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.blocks.textures import TextureSample
from mc_photogrammetry.export import attribution
from mc_photogrammetry.reconstruction.camera import CameraModel, intrinsics_for
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.vision.gui_mask import create_gui_mask
import mc_photogrammetry.evaluation_frozen_planes as evaluator


def _alpha_png(left_opaque: bool = True) -> str:
    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    if left_opaque:
        alpha[:, :8] = 255
    else:
        alpha[:, 8:] = 255
    buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _single_block_truth(state: BlockState | None = None) -> GroundTruthExport:
    state = state or BlockState("minecraft:stone")
    return GroundTruthExport(
        scene_id="synthetic-scene", world_identity="synthetic-world",
        region=RegionBounds((0, 0, 4), (0, 0, 4)),
        blocks=(BlockRecord((0, 0, 4), state.name, dict(state.properties)),),
    )


def _camera() -> CameraModel:
    return CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0,
                       pitch_degrees=0.0, fov_degrees=70.0, eye_height=0.5)


def test_frozen_plane_image_ingress_accepts_more_than_eight_frames(tmp_path):
    capture_path = tmp_path / "capture.json"
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    ok, encoded = cv2.imencode(".png", np.zeros((1, 1, 3), dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    frames = []
    capture_frames = []
    screenshots = []
    for index in range(9):
        basename = f"frame-{index:02}.png"
        (capture_dir / basename).write_bytes(payload)
        frames.append({"frame_id": basename, "input": {
            "basename": basename, "width": 1, "height": 1, "sha256": digest,
        }})
        capture_frames.append({"filename": basename, "frame_id": Path(basename).stem,
                               "captured": True})
        screenshots.append({"filename": basename, "sha256": digest, "bytes": len(payload)})
    freeze = {"frames": frames}
    capture = {
        "frames": capture_frames,
        "cameras": [{} for _ in frames],
        "resolutions": [[1, 1] for _ in frames],
        "screenshots": screenshots,
    }

    images, cameras, resolutions = evaluator._verify_capture_images(
        freeze, capture, capture_path)

    assert len(images) == len(cameras) == len(resolutions) == 9


def test_face_slot_plane_grouping_and_usable_mask_exclude_gui_pixels():
    ownership = np.asarray([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
    ], dtype=np.int32)
    owners = [((2, 3, 5), "north"), ((4, 3, 5), "north"),
              ((2, 3, 5), "east"), ((2, 3, 6), "north")]
    usable = np.asarray([[True, False, True, True],
                         [True, True, True, True]], dtype=bool)
    groups = evaluator.group_visible_owners(ownership, owners, usable)
    assert set(groups) == {("north", 5), ("east", 3), ("north", 6)}
    assert int(groups[("north", 5)].sum()) == 3
    assert not groups[("north", 5)][0, 1]
    assert int(sum(mask.sum() for mask in groups.values())) == int((ownership >= 0)[usable].sum())


def test_full_cube_primary_excludes_visible_second_subface_when_first_is_culled():
    partial_state = BlockState("minecraft:partial_projection_test")
    cube_state = BlockState("minecraft:stone")
    first_face = ("north", ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
                             (0.5, 0.5, -0.5)), (0, 0, -1), None, None)
    second_face = ("north", ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0),
                              (0.5, 0.5, 0.0)), (0, 0, -1), None, None)
    palette = (
        PaletteColor(partial_state, (90, 90, 90), full_cube=False,
                     model_faces=(first_face, second_face)),
        PaletteColor(cube_state, (120, 120, 120), full_cube=True),
    )
    # The owner table still contains both same-slot polygons, but the first has
    # no raster ownership (culled/fully occluded); only polygon 1 has pixels.
    owner_index = [((0, 0, 4), "north"), ((0, 0, 4), "north"),
                   ((0, 0, 5), "north")]
    ownership = np.asarray([[-1, 1, 2]], dtype=np.int32)
    usable = np.ones_like(ownership, dtype=bool)
    states = {(0, 0, 4): partial_state, (0, 0, 5): cube_state}

    groups, scoring_valid, audit = evaluator.group_full_cube_owners(
        ownership, owner_index, usable, block_states=states, palette=palette)

    assert set(groups) == {("north", 5)}
    assert groups[("north", 5)].tolist() == [[False, False, True]]
    assert scoring_valid.tolist() == [[True, False, True]]
    assert audit["excluded_non_full_cube_owner_pixel_count"] == 1
    assert audit["excluded_non_full_cube_states"] == [
        {"state": partial_state.name, "pixel_count": 1}]
    prediction = np.ones_like(ownership, dtype=bool)
    scored = evaluator.score_frame_planes(
        [("p", {}, prediction & scoring_valid)], groups)
    assert scored["proposals"][0]["metrics"]["prediction_pixels"] == 2
    assert scored["micro"]["fp_pixels"] == 1


def test_full_cube_grouping_uses_cell_boundary_for_repeated_face_offsets():
    state = BlockState("minecraft:offset_full_cube")
    faces = tuple(("north", ((-0.5, -0.5, z), (0.5, -0.5, z),
                             (0.5, 0.5, z)), (0, 0, -1), None, None)
                  for z in (-0.5, 0.0))
    palette = (PaletteColor(state, (80, 80, 80), model_faces=faces, full_cube=True),)
    ownership = np.asarray([[0, 1]], dtype=np.int32)
    groups, valid, _audit = evaluator.group_full_cube_owners(
        ownership, [((2, 3, 5), "north"), ((2, 3, 5), "north")],
        np.ones_like(ownership, dtype=bool), block_states={(2, 3, 5): state},
        palette=palette)
    assert set(groups) == {("north", 5)}
    assert groups[("north", 5)].tolist() == [[True, True]]
    assert valid.all()


def test_renderer_uses_capture_camera_and_full_screenshot_coordinates(monkeypatch):
    state = BlockState("minecraft:stone")
    truth = _single_block_truth(state)
    camera = _camera()
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    palette = (PaletteColor(state, (200, 200, 200)),)
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda _jar, _cache, _states: (palette, "a" * 64, "b" * 64, []))
    calls = []
    render = attribution.render_result

    def record_camera(blocks, scene, supplied_camera, width, height, colours, **kwargs):
        calls.append((supplied_camera, width, height, kwargs["scale"]))
        return render(blocks, scene, supplied_camera, width, height, colours, **kwargs)

    monkeypatch.setattr(evaluator.attribution, "render_result", record_camera)
    planes = evaluator.render_truth_planes(
        truth, camera, image, vanilla_jar_path="vanilla.jar",
        palette_cache_path="palette.cache")
    assert calls == [(camera, 96, 72, 1.0)]
    assert ("north", 4) in planes
    assert planes[("north", 4)].shape == (72, 96)
    assert planes[("north", 4)].any()
    assert not np.any(planes[("north", 4)] & create_gui_mask(72, 96))


def test_partial_alpha_model_path_changes_visible_face_pixels(monkeypatch):
    state = BlockState("minecraft:cutout")
    truth = _single_block_truth(state)
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    corners = tuple(tuple(float(value) for value in point)
                    for point in attribution.box_face_corners((0, 0, 0), (1, 1, 1), "north"))
    palette = (PaletteColor(state, (240, 240, 240), full_cube=False,
                            model_faces=(("north", corners, (0, 0, -1), None,
                                          _alpha_png()),)),)
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda _jar, _cache, _states: (palette, "a" * 64, "b" * 64, []))
    cutout = evaluator.render_truth_planes(
        truth, _camera(), image, vanilla_jar_path="vanilla.jar",
        palette_cache_path="palette.cache")
    full_palette = (PaletteColor(state, (240, 240, 240)),)
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda _jar, _cache, _states: (full_palette, "a" * 64, "b" * 64, []))
    opaque = evaluator.render_truth_planes(
        truth, _camera(), image, vanilla_jar_path="vanilla.jar",
        palette_cache_path="palette.cache")
    assert ("north", 4) in cutout and ("north", 4) in opaque
    assert 0 < cutout[("north", 4)].sum() < opaque[("north", 4)].sum()


def test_explicit_air_records_are_not_sent_to_palette_or_renderer(monkeypatch):
    truth = GroundTruthExport(
        scene_id="synthetic-scene", world_identity="synthetic-world",
        region=RegionBounds((0, 0, 4), (0, 0, 4)),
        blocks=(BlockRecord((0, 0, 4), "minecraft:air", {}),),
    )
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda *_args: pytest.fail("air must not require a palette entry"))
    planes = evaluator.render_truth_planes(
        truth, _camera(), np.zeros((72, 96, 3), dtype=np.uint8),
        vanilla_jar_path="vanilla.jar", palette_cache_path="palette.cache")
    assert planes == {}


def test_unique_subset_compatible_state_projection_rekeys_and_keeps_partial_geometry():
    target = BlockState("minecraft:projection_test", (("facing", "north"),
                                                        ("waterlogged", "false")))
    cached_state = BlockState("minecraft:projection_test", (("facing", "north"),))
    model_faces = (("north", ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
                               (0.5, 0.5, -0.5)), (0, 0, -1), None, None),)
    boxes = (((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),)
    sample = TextureSample(cached_state, (40, 50, 60), "",
                           model_faces=model_faces, shape_boxes=boxes, full_cube=False)
    palette, projected = evaluator._project_palette_states((target,), (sample,), ())
    assert len(palette) == 1
    assert palette[0].state == target
    assert palette[0].model_faces == model_faces
    assert palette[0].shape_boxes == boxes
    assert projected == [{
        "target_state": target.name,
        "target_properties": dict(target.properties),
        "cached_variant_properties": dict(cached_state.properties),
        "rule": "one unique same-name cache sample whose stored blockstate-variant properties are an exact subset of GT properties",
    }]


def test_subset_projection_rejects_ambiguous_or_conflicting_variants():
    target = BlockState("minecraft:projection_test", (("facing", "north"),))
    empty = TextureSample(BlockState(target.name), (1, 2, 3), "")
    facing = TextureSample(target, (4, 5, 6), "")
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="ambiguous"):
        evaluator._project_palette_states((target,), (empty, facing), ())
    conflicting = TextureSample(
        BlockState(target.name, (("facing", "south"),)), (1, 2, 3), "")
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="no compatible"):
        evaluator._project_palette_states((target,), (conflicting,), ())


def _variant_blockstate(*, selector_waterlogged: bool = False) -> dict:
    if selector_waterlogged:
        return {"variants": {
            "type=top,waterlogged=false": {"model": "minecraft:block/top_dry"},
            "type=top,waterlogged=true": {"model": "minecraft:block/top_wet"},
            "type=bottom,waterlogged=false": {"model": "minecraft:block/bottom_dry"},
            "type=bottom,waterlogged=true": {"model": "minecraft:block/bottom_wet"},
        }}
    return {"variants": {
        "type=top": {"model": "minecraft:block/top"},
        "type=bottom": {"model": "minecraft:block/bottom"},
    }}


def _selector_resource(name: str, document: dict) -> tuple[str, bytes, str]:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (f"assets/minecraft/blockstates/{name.removeprefix('minecraft:')}.json",
            payload, hashlib.sha256(payload).hexdigest())


def _slab_sample(state: BlockState) -> TextureSample:
    model_faces = (("north", ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
                                (0.5, 0.0, -0.5)), (0, 0, -1), None, None),)
    boxes = (((0.0, 0.5, 0.0), (1.0, 1.0, 1.0)),)
    return TextureSample(state, (52, 92, 76), "",
                         model_faces=model_faces, shape_boxes=boxes, full_cube=False)


def test_jar_selector_projection_ignores_only_unreferenced_waterlogged_property():
    name = "minecraft:dark_prismarine_slab"
    target = BlockState(name, (("type", "top"), ("waterlogged", "true")))
    cached = BlockState(name, (("type", "top"), ("waterlogged", "false")))
    resource = _selector_resource(name, _variant_blockstate())
    palette, projection = evaluator._project_palette_states(
        (target,), (_slab_sample(cached),), (),
        blockstate_resources={name: resource}, vanilla_jar_sha256="a" * 64)

    assert palette[0].state == target
    assert palette[0].shape_boxes == _slab_sample(cached).shape_boxes
    assert projection == [{
        "target_state": name,
        "target_properties": dict(target.properties),
        "cached_variant_properties": dict(cached.properties),
        "rule": "unique same-name cache sample with identical selected vanilla blockstate variant/multipart signature; every differing GT property is absent from all selector conditions",
        "blockstate_asset_path": "assets/minecraft/blockstates/dark_prismarine_slab.json",
        "blockstate_asset_sha256": resource[2],
        "vanilla_jar_sha256": "a" * 64,
        "selector_properties": ["type"],
        "selector_signature": {
            "variant": {"key": "type=top", "apply_sha256": hashlib.sha256(
                b'{"model":"minecraft:block/top"}').hexdigest()},
        },
        "ignored_properties": [{
            "name": "waterlogged", "ground_truth": "true", "cached_variant": "false",
        }],
        "rendering_limitation": (
            "the baked block model projection does not render a waterlogged fluid overlay"),
    }]


def test_selector_projection_refuses_type_and_referenced_property_mismatches():
    name = "minecraft:dark_prismarine_slab"
    target = BlockState(name, (("type", "top"), ("waterlogged", "true")))
    bottom = BlockState(name, (("type", "bottom"), ("waterlogged", "false")))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError,
                       match="no selector-compatible"):
        evaluator._project_palette_states(
            (target,), (_slab_sample(bottom),), (),
            blockstate_resources={name: _selector_resource(name, _variant_blockstate())})

    top_dry = BlockState(name, (("type", "top"), ("waterlogged", "false")))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError,
                       match="no selector-compatible"):
        evaluator._project_palette_states(
            (target,), (_slab_sample(top_dry),), (),
            blockstate_resources={
                name: _selector_resource(name, _variant_blockstate(selector_waterlogged=True))})


def test_selector_projection_refuses_ambiguous_matching_cached_models():
    name = "minecraft:dark_prismarine_slab"
    target = BlockState(name, (("type", "top"), ("waterlogged", "true")))
    cached_a = BlockState(name, (("type", "top"), ("waterlogged", "false"),
                                 ("alternate", "a")))
    cached_b = BlockState(name, (("type", "top"), ("waterlogged", "false"),
                                 ("alternate", "b")))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError,
                       match="ambiguous selector-compatible"):
        evaluator._project_palette_states(
            (target,), (_slab_sample(cached_a), _slab_sample(cached_b)), (),
            blockstate_resources={name: _selector_resource(name, _variant_blockstate())})


def test_multipart_selector_properties_are_compared_from_bound_blockstate():
    name = "minecraft:multipart_projection_test"
    target = BlockState(name, (("north", "false"), ("south", "false"),
                               ("waterlogged", "true")))
    cached = BlockState(name, (("north", "false"), ("south", "false"),
                               ("waterlogged", "false")))
    multipart = {"multipart": [
        {"when": {"OR": [{"north": "true"}, {"south": "true"}]},
         "apply": {"model": "minecraft:block/side"}},
        {"apply": {"model": "minecraft:block/post"}},
    ]}
    palette, projection = evaluator._project_palette_states(
        (target,), (_slab_sample(cached),), (),
        blockstate_resources={name: _selector_resource(name, multipart)})
    assert palette[0].state == target
    assert projection[0]["selector_properties"] == ["north", "south"]
    assert projection[0]["ignored_properties"] == [
        {"name": "waterlogged", "ground_truth": "true", "cached_variant": "false"}]
    assert projection[0]["selector_signature"]["multipart"] == [{
        "index": 1,
        "apply_sha256": hashlib.sha256(b'{"model":"minecraft:block/post"}').hexdigest(),
    }]


def test_palette_selector_projection_reads_and_hashes_the_supplied_vanilla_jar(
        tmp_path, monkeypatch):
    name = "minecraft:dark_prismarine_slab"
    target = BlockState(name, (("type", "top"), ("waterlogged", "true")))
    cached = BlockState(name, (("type", "top"), ("waterlogged", "false")))
    sample = _slab_sample(cached)
    monkeypatch.setattr(evaluator, "_load_cached_samples", lambda *_args: [sample])
    jar = tmp_path / "vanilla.jar"
    cache = tmp_path / "palette.json"
    cache.write_bytes(b"synthetic pinned cache")
    asset = "assets/minecraft/blockstates/dark_prismarine_slab.json"
    raw_blockstate = json.dumps(_variant_blockstate(), sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr(asset, raw_blockstate)

    palette, jar_hash, cache_hash, projection = evaluator._load_palette(
        jar, cache, (target,))
    assert palette[0].state == target
    assert jar_hash == hashlib.sha256(jar.read_bytes()).hexdigest()
    assert cache_hash == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert projection[0]["blockstate_asset_path"] == asset
    assert projection[0]["blockstate_asset_sha256"] == hashlib.sha256(
        raw_blockstate).hexdigest()
    assert projection[0]["vanilla_jar_sha256"] == jar_hash

    missing_asset_jar = tmp_path / "missing-blockstate.jar"
    with zipfile.ZipFile(missing_asset_jar, "w"):
        pass
    with pytest.raises(evaluator.FrozenPlaneEvaluationError,
                       match="no compatible cached variant or bound vanilla blockstate asset"):
        evaluator._load_palette(missing_asset_jar, cache, (target,))


def test_exact_cached_state_match_precedes_subset_projection():
    target = BlockState("minecraft:projection_test", (("facing", "north"),))
    exact = PaletteColor(target, (10, 20, 30))
    compatible_but_not_exact = TextureSample(BlockState(target.name), (1, 2, 3), "")
    palette, projected = evaluator._project_palette_states(
        (target,), (compatible_but_not_exact,), (exact,))
    assert palette == (exact,)
    assert projected == []


def test_maximum_overlap_is_one_to_one_and_micro_does_not_count_truth_twice():
    truth = np.zeros((8, 8), dtype=bool)
    truth[2:5, 2:5] = True
    pred_a = truth.copy()
    pred_b = truth.copy()
    result = evaluator.score_frame_planes(
        [("p0", {}, pred_a), ("p1", {}, pred_b)], {("north", 4): truth})
    assert len(result["matches"]) == 1
    assert result["one_to_one_assignment_count"] == 1
    assert len(result["unmatched_predictions"]) == 1
    assert result["micro"]["tp_pixels"] == int(truth.sum())
    assert result["micro"]["fn_pixels"] == 0
    assert result["micro"]["fp_pixels"] == int(truth.sum())


def test_empty_prediction_and_empty_truth_masks_are_reported_without_assignment():
    blank = np.zeros((4, 4), dtype=bool)
    truth = blank.copy()
    truth[1:3, 1:3] = True
    result = evaluator.score_frame_planes([("empty", {}, blank)], {("up", 2): truth})
    assert result["matches"] == []
    assert result["unmatched_predictions"] == ["empty"]
    assert result["unmatched_truth_planes"] == ["up:2"]
    assert result["micro"]["tp_pixels"] == 0
    assert result["micro"]["fp_pixels"] == 0
    assert result["micro"]["fn_pixels"] == 4
    assert evaluator._maximum_overlap_assignment(np.zeros((0, 2), dtype=np.int64)) == []


def test_axis_calibration_comparison_is_permutation_and_sign_invariant():
    result = evaluator._axis_line_errors(
        {"yaw_degrees": 90.0, "pitch_degrees": 0.0, "roll_degrees": 0.0,
         "focal_pixels": intrinsics_for(_camera(), 96, 72).focal_length_pixels[1],
         "width": 96, "height": 72}, _camera())
    assert result["max_axis_line_error_degrees"] < 1e-6
    assert result["focal_percent_error_vs_vertical_focal"] == pytest.approx(0.0)
    assert "best absolute-dot" in result["gauge_note"]


def _encoded_mask(mask: np.ndarray, offset: int = 0) -> tuple[bytes, dict]:
    payload = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    return payload, {"offset": offset, "length": len(payload),
                     "shape": list(mask.shape), "dtype": "bool",
                     "encoding": "packbits-little",
                     "sha256": hashlib.sha256(payload).hexdigest()}


def _evaluation_fixture(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    captures = scene / "captures"
    captures.mkdir(parents=True)
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    image[20:55, 25:70] = (70, 100, 130)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    raw = encoded.tobytes()
    name = "view-synthetic-00.png"
    (captures / name).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    cam = _camera().to_dict()
    camera_dict = {"position": list(cam["position"]), "yaw_degrees": cam["yaw_degrees"],
                   "pitch_degrees": cam["pitch_degrees"], "fov_degrees": cam["fov_degrees"],
                   "eye_height": cam["eye_height"], "roll_degrees": 0.0,
                   "intrinsics": None}
    capture = {
        "scene_id": "synthetic-scene", "world_identity": "synthetic-world",
        "region": {"min": [0, 0, 4], "max": [0, 0, 4]},
        "frames": [{"filename": name, "frame_id": "view-synthetic-00", "captured": True,
                    "actual": {"framebuffer_width": 96, "framebuffer_height": 72}}],
        "screenshots": [{"filename": name, "frame_id": "view-synthetic-00",
                         "sha256": digest, "bytes": len(raw)}],
        "cameras": [camera_dict], "resolutions": [[96, 72]],
    }
    capture_path = scene / "capture.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    truth = _single_block_truth().to_dict()
    truth_path = scene / "ground_truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    proposal = np.zeros((72, 96), dtype=bool)
    proposal[20:55, 25:70] = True
    payload, descriptor = _encoded_mask(proposal)
    frame = {"frame_id": name,
             "input": {"basename": name, "width": 96, "height": 72, "sha256": digest},
             "calibration": None,
             "proposal": {"proposals": [{"axis": 0, "side": 1, "mask": descriptor}]}}
    sidecar_name = "freeze.json." + hashlib.sha256(payload).hexdigest() + ".masks.bin"
    freeze = {"schema_version": 1, "artifact_id": "synthetic-freeze-id",
              "labels_present": False,
              "configuration": {"gui_mask": {name: getattr(evaluator.GuiMaskConfig(), name)
                                                  for name in evaluator.GuiMaskConfig.__dataclass_fields__}},
              "frames": [frame],
              "mask_sidecar": {"file": sidecar_name, "sha256": hashlib.sha256(payload).hexdigest(),
                               "size_bytes": len(payload)}}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    freeze_path.with_name(sidecar_name).write_bytes(payload)
    monkeypatch.setattr(evaluator, "load_and_verify_frozen_image",
                        lambda _path: freeze)
    jar_path = tmp_path / "vanilla.jar"
    palette_path = tmp_path / "palette.cache"
    jar_path.write_bytes(b"synthetic jar identity")
    palette_path.write_bytes(b"synthetic palette identity")
    state = BlockState("minecraft:stone")
    palette = (PaletteColor(state, (125, 125, 125)),)
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda _jar, _cache, _states: (palette, "a" * 64, "b" * 64, []))
    return freeze_path, capture_path, truth_path, jar_path, palette_path, freeze


def _context_document(paths, *, region=None):
    freeze_path, capture_path, truth_path, _jar_path, _palette_path, freeze = paths
    capture_bytes = capture_path.read_bytes()
    target_bytes = truth_path.read_bytes()
    capture = json.loads(capture_bytes)
    target = GroundTruthExport.from_dict(json.loads(target_bytes))
    ordered_images = []
    for frame in freeze["frames"]:
        item = frame["input"]
        ordered_images.append({key: item[key] for key in ("basename", "width", "height", "sha256")})
    context_region = region or RegionBounds((-1, -1, 3), (1, 1, 5))
    return {
        "schema_version": 1,
        "artifact_type": evaluator._CONTEXT_ARTIFACT_TYPE,
        "scene_id": target.scene_id,
        "world_identity": target.world_identity,
        "capture_metadata_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "target_ground_truth_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "ordered_images": ordered_images,
        "target_region": target.region.to_dict(),
        "region": context_region.to_dict(),
        "blocks": target.to_dict()["blocks"],
    }


def _validate_fixture_context(paths, document=None):
    _freeze_path, capture_path, truth_path, _jar_path, _palette_path, freeze = paths
    capture_bytes = capture_path.read_bytes()
    target_bytes = truth_path.read_bytes()
    capture = json.loads(capture_bytes)
    target = GroundTruthExport.from_dict(json.loads(target_bytes))
    rows = [frame["input"] for frame in freeze["frames"]]
    return evaluator._validate_context_ground_truth(
        document if document is not None else _context_document(paths),
        capture_sha256=hashlib.sha256(capture_bytes).hexdigest(),
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        capture_doc=capture, input_rows=rows, target_truth=target)


def test_context_schema_accepts_equal_or_larger_region_and_rejects_bad_volume(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    target = GroundTruthExport.from_dict(json.loads(paths[2].read_text(encoding="utf-8")))
    equal = _context_document(paths, region=target.region)
    assert _validate_fixture_context(paths, equal).region == target.region
    larger = _context_document(paths)
    assert _validate_fixture_context(paths, larger).region == RegionBounds((-1, -1, 3), (1, 1, 5))

    larger["region"] = {"min": [1, 0, 4], "max": [1, 0, 4]}
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="does not contain"):
        _validate_fixture_context(paths, larger)

    oversized = _context_document(paths)
    oversized["region"] = {"min": [0, 0, 0], "max": [62, 62, 62]}
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="region violates region limits"):
        _validate_fixture_context(paths, oversized)


@pytest.mark.parametrize(("field", "replacement", "message"), [
    ("capture_metadata_sha256", "0" * 64, "capture metadata hash differs"),
    ("target_ground_truth_sha256", "0" * 64, "target ground-truth hash differs"),
    ("scene_id", "other-scene", "scene ID differs"),
    ("world_identity", "other-world", "world identity differs"),
])
def test_context_schema_rejects_capture_target_identity_mismatches(
        tmp_path, monkeypatch, field, replacement, message):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    document = _context_document(paths)
    document[field] = replacement
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match=message):
        _validate_fixture_context(paths, document)


@pytest.mark.parametrize(("field", "replacement"), [
    ("basename", "view-other.png"),
    ("width", 95),
    ("height", 71),
    ("sha256", "0" * 64),
])
def test_context_schema_rejects_ordered_image_identity_mismatches(
        tmp_path, monkeypatch, field, replacement):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    document = _context_document(paths)
    document["ordered_images"][0][field] = replacement
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="ordered image"):
        _validate_fixture_context(paths, document)


def test_context_schema_rejects_reordered_image_inventory(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    document = _context_document(paths)
    first = dict(document["ordered_images"][0])
    second = {**first, "basename": "view-synthetic-01.png", "sha256": "1" * 64}
    expected = [first, second]
    document["ordered_images"] = [second, first]
    freeze_path, capture_path, truth_path, _jar, _palette, _freeze = paths
    capture_bytes, target_bytes = capture_path.read_bytes(), truth_path.read_bytes()
    capture = json.loads(capture_bytes)
    target = GroundTruthExport.from_dict(json.loads(target_bytes))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="ordered image"):
        evaluator._validate_context_ground_truth(
            document, capture_sha256=hashlib.sha256(capture_bytes).hexdigest(),
            target_sha256=hashlib.sha256(target_bytes).hexdigest(),
            capture_doc=capture, input_rows=expected, target_truth=target)


def test_context_schema_rejects_unknown_keys_bad_types_and_missing_target_records(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    document = _context_document(paths)
    document["unexpected"] = "metadata"
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="unknown or missing"):
        _validate_fixture_context(paths, document)

    document = _context_document(paths)
    document["ordered_images"][0]["width"] = True
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="ordered image"):
        _validate_fixture_context(paths, document)

    document = _context_document(paths)
    document["blocks"] = []
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="preserve every target"):
        _validate_fixture_context(paths, document)


def test_context_pixel_domain_excludes_unknown_symmetrically_and_keeps_partial_as_negative():
    ownership = np.asarray([[0, 1, -1, 0, 1]], dtype=np.int32)
    usable = np.asarray([[True, True, True, True, False]], dtype=bool)
    full = np.asarray([[True, False, False, True, False]], dtype=bool)
    known, counts = evaluator._context_pixel_domain(ownership, usable, {("north", 4): full})
    assert known.tolist() == [[True, True, False, True, False]]
    assert counts == {
        "image_pixels": 5,
        "usable_pixels": 4,
        "known_context_owned_pixels": 3,
        "unknown_no_owner_usable_pixels": 1,
        "excluded_unknown_pixels": 1,
        "excluded_gui_masked_pixels": 1,
        "full_cube_positive_pixels": 2,
        "other_context_owned_known_negative_pixels": 1,
    }
    prediction = np.ones_like(usable)
    scored = evaluator.score_frame_planes(
        [("proposal", {}, prediction & known)], {("north", 4): full})
    assert scored["micro"]["tp_pixels"] == 2
    assert scored["micro"]["fp_pixels"] == 1
    assert scored["micro"]["tp_pixels"] + scored["micro"]["fp_pixels"] == 3


def test_context_scoring_uses_exact_capture_space_renderer_and_marks_no_owner_unknown():
    truth = _single_block_truth()
    state = BlockState("minecraft:stone")
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    palette = (PaletteColor(state, (180, 190, 200), full_cube=True),)
    provisional, primary, valid, audit = evaluator._render_block_map_evidence(
        {truth.blocks[0].position: state}, (state,), truth, _camera(), image,
        gui_mask_config=None, palette=palette, context_scoring=True)
    domain = audit["context_pixel_domain"]
    assert sum(int(mask.sum()) for mask in primary.values()) == domain["full_cube_positive_pixels"]
    assert domain["known_context_owned_pixels"] == int(valid.sum())
    assert domain["known_context_owned_pixels"] > 0
    assert domain["unknown_no_owner_usable_pixels"] > 0
    assert domain["excluded_gui_masked_pixels"] > 0
    assert domain["other_context_owned_known_negative_pixels"] == 0
    assert provisional and primary


def test_context_evaluation_binds_provenance_and_scores_only_known_domain(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    freeze_path, capture_path, truth_path, jar_path, palette_path, _freeze = paths
    context = _context_document(paths)
    context_path = capture_path.parent / "context-ground-truth.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")

    def render_context(_blocks, _states, _truth, _camera, image, *, gui_mask_config,
                       palette=None, provider_assets=None, context_scoring=False):
        assert context_scoring is True
        height, width = image.shape[:2]
        usable = ~create_gui_mask(height, width, gui_mask_config or evaluator.GuiMaskConfig())
        ownership = np.full((height, width), -1, dtype=np.int32)
        ownership[20:25, 25:30] = 0
        ownership[25:30, 25:30] = 1
        primary = np.zeros((height, width), dtype=bool)
        primary[20:25, 25:30] = True
        partial = np.zeros_like(primary)
        partial[25:30, 25:30] = True
        scoring_valid, domain = evaluator._context_pixel_domain(
            ownership, usable, {("north", 4): primary})
        audit = {
            "excluded_non_full_cube_owner_pixel_count": 25,
            "excluded_non_full_cube_states": [{"state": "minecraft:partial_test", "pixel_count": 25}],
            "potentially_ambiguous_owner_pixel_count": 0,
            "potentially_ambiguous_owner_states": [],
            "context_pixel_domain": domain,
        }
        return {("north", 4): primary, ("east", 1): partial}, {("north", 4): primary}, scoring_valid, audit

    monkeypatch.setattr(evaluator, "_render_block_map_evidence", render_context)
    one = evaluator.evaluate_frozen_planes(
        freeze_path, capture_path, truth_path, jar_path, palette_path,
        context_ground_truth_path=context_path)
    output_path = tmp_path / "context-result.json"
    two = evaluator.evaluate_frozen_planes(
        freeze_path, capture_path, truth_path, jar_path, palette_path,
        output_path, context_ground_truth_path=context_path)
    assert one["artifact_id"] == two["artifact_id"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == two
    assert two["method"]["ground_truth_mode"] == "context_region_known_owner_domain"
    assert two["inputs"]["ground_truth_provenance"]["target"]["region"] == context["target_region"]
    assert two["inputs"]["ground_truth_provenance"]["context"]["region"] == context["region"]
    assert two["frames"][0]["scoring"]["micro"]["tp_pixels"] == 25
    assert two["frames"][0]["scoring"]["micro"]["fp_pixels"] == 25
    assert (two["frames"][0]["scoring"]["micro"]["tp_pixels"]
            + two["frames"][0]["scoring"]["micro"]["fp_pixels"] == 50)
    domain = two["scoring_domain_summary"]
    assert domain["known_context_owned_pixels"] == 50
    assert domain["full_cube_positive_pixels"] == 25
    assert domain["other_context_owned_known_negative_pixels"] == 25
    assert domain["unknown_no_owner_usable_pixels"] > 0
    assert two["promotion"]["promotion_eligible"] is False
    assert two["promotion"]["context_boundary_completeness_proven"] is False


def test_invalid_freeze_is_rejected_before_capture_or_ground_truth_reads(tmp_path, monkeypatch):
    called = []

    def reject(_path):
        called.append("freeze")
        raise FrozenImageError("bad freeze")

    class FrozenImageError(ValueError):
        pass

    monkeypatch.setattr(evaluator, "load_and_verify_frozen_image", reject)
    with pytest.raises(FrozenImageError):
        evaluator.evaluate_frozen_planes(
            tmp_path / "missing-freeze.json", tmp_path / "missing-capture.json",
            tmp_path / "secret-ground-truth.json", tmp_path / "missing.jar",
            tmp_path / "missing.cache")
    assert called == ["freeze"]


def test_image_hash_mismatch_stops_before_ground_truth_open(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    freeze_path, capture_path, truth_path, jar_path, palette_path, freeze = paths
    image_path = capture_path.parent / "captures" / "view-synthetic-00.png"
    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    image[0, 0] = (1, 2, 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    image_path.write_bytes(encoded.tobytes())
    monkeypatch.setattr(evaluator, "_load_truth",
                        lambda _path: pytest.fail("GT was opened before PNG verification"))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="bytes differ from freeze"):
        evaluator.evaluate_frozen_planes(
            freeze_path, capture_path, truth_path, jar_path, palette_path)


def test_capture_dimension_mismatch_stops_before_ground_truth_open(tmp_path, monkeypatch):
    paths = _evaluation_fixture(tmp_path, monkeypatch)
    freeze_path, capture_path, truth_path, jar_path, palette_path, freeze = paths
    capture = json.loads(capture_path.read_text(encoding="utf8"))
    capture["resolutions"][0] = [95, 72]
    capture_path.write_text(json.dumps(capture), encoding="utf8")
    monkeypatch.setattr(evaluator, "_load_truth",
                        lambda _path: pytest.fail("GT was opened before dimension verification"))
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="resolution differs"):
        evaluator.evaluate_frozen_planes(
            freeze_path, capture_path, truth_path, jar_path, palette_path)


def test_evaluation_artifact_is_deterministic_and_warns_about_cropped_occluders(tmp_path, monkeypatch):
    freeze_path, capture_path, truth_path, jar_path, palette_path, freeze = _evaluation_fixture(
        tmp_path, monkeypatch)
    one = evaluator.evaluate_frozen_planes(
        freeze_path, capture_path, truth_path, jar_path, palette_path)
    output = tmp_path / "result.json"
    two = evaluator.evaluate_frozen_planes(
        freeze_path, capture_path, truth_path, jar_path, palette_path, output)
    assert one["artifact_id"] == two["artifact_id"]
    assert json.loads(output.read_text(encoding="utf8")) == two
    assert two["promotion"]["promotion_eligible"] is False
    assert two["promotion"]["cropped_occluder_warning"] is True
    assert two["method"]["ground_truth_mode"] == "target_bounded_legacy_provisional"
    assert two["scoring_domain_summary"] is None
    assert two["method"]["proposal_masks_mutated_or_recomputed"] is False
    assert two["frames"][0]["scoring"]["micro"]["tp_pixels"] > 0
    expected_proposal_pixels = np.zeros((72, 96), dtype=bool)
    expected_proposal_pixels[20:55, 25:70] = True
    expected_proposal_pixels &= ~create_gui_mask(72, 96)
    assert (two["frames"][0]["scoring"]["proposals"][0]["metrics"]["prediction_pixels"]
            == int(expected_proposal_pixels.sum()))


def test_input_paths_are_excluded_from_deterministic_artifact_identity(tmp_path, monkeypatch):
    first = _evaluation_fixture(tmp_path / "a", monkeypatch)
    second = _evaluation_fixture(tmp_path / "b", monkeypatch)
    one = evaluator.evaluate_frozen_planes(
        first[0], first[1], first[2], first[3], first[4])
    two = evaluator.evaluate_frozen_planes(
        second[0], second[1], second[2], second[3], second[4])
    assert one["artifact_id"] == two["artifact_id"]
    assert str(tmp_path) not in json.dumps(one)
    assert str(tmp_path) not in json.dumps(two)


def test_image_only_production_modules_do_not_import_the_evaluator():
    root = Path(evaluator.__file__).parents[2]
    paths = [
        root / "src/mc_photogrammetry/reconstruction/frozen_image_runner.py",
        root / "src/mc_photogrammetry/reconstruction/region_growth.py",
        root / "src/mc_photogrammetry/reconstruction/lattice_planes.py",
        root / "src/mc_photogrammetry/reconstruction/plane_graph.py",
        root / "src/mc_photogrammetry/cli.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any("evaluation_frozen_planes" in name for name in imports), path.name
