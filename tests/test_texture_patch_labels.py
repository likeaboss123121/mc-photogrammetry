"""Focused tests for the evaluation-only E1 patch-label harness.

Every test here runs on synthetic geometry, masks and pins in ``tmp_path``. No
test loads a Template 2 / Template 6 corpus frame, runs a corpus evaluator, or
writes a patch artifact outside ``tmp_path``.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.textures import TextureSample
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import box_face_corners
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.scene import SceneTransform

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "evaluate_texture_patch_labels.py"
)
_spec = importlib.util.spec_from_file_location("evaluate_texture_patch_labels_under_test", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
e1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e1  # dataclasses(slots=True) resolves the module by name
_spec.loader.exec_module(e1)

STATE = BlockState("minecraft:e1_test_cube")
OTHER = BlockState("minecraft:e1_other_cube")
PLAIN = BlockState("minecraft:e1_unmodelled")


def _exact_palette(state: BlockState = STATE, *, alpha: str | None = None) -> tuple[PaletteColor, ...]:
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    record = ("north", corners, (0.0, 0.0, -1.0), None, alpha)
    return (PaletteColor(state, (160, 160, 160), model_faces=(record,)),)


def _encoded_alpha(columns: slice) -> str:
    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[:, columns] = 255
    buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _synthetic_frame() -> tuple[np.ndarray, CameraModel, SceneTransform]:
    image = np.full((100, 100, 3), 160, dtype=np.uint8)
    camera = CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)
    return image, camera, SceneTransform((0, 0, 0), (2, 2, 2))


def _derive(
    *,
    state: BlockState = STATE,
    palette: tuple[PaletteColor, ...] | None = None,
    usable: np.ndarray | None = None,
    coordinate: tuple[int, int, int] = (0, 0, 0),
):
    image, camera, scene = _synthetic_frame()
    if usable is None:
        usable = np.ones((100, 100), dtype=bool)
    if palette is None:
        palette = _exact_palette(state)
    return e1.derive_frame_labels(
        "frame.png",
        image,
        camera,
        scene,
        {coordinate: state},
        palette,
        usable,
        minimum_area=1.0,
        minimum_visible_fraction=0.1,
        shrink=1.0,
    )


def _sample(
    coordinate: tuple[int, int, int] = (0, 0, 0),
    *,
    value: float = 160.0,
    weight: float = 1.0,
    patch: np.ndarray | None = None,
) -> "e1.PatchSample":
    if patch is None:
        patch = np.full((e1.PATCH_SIZE, e1.PATCH_SIZE, 3), value, dtype=np.float32)
    else:
        patch = np.asarray(patch, dtype=np.float32)
    valid = np.ones((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=bool)
    coverage = np.ones((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=np.float32)
    return e1.PatchSample(
        frame_id="frame.png",
        coordinate=coordinate,
        slot="north",
        truth_state=STATE,
        weight=weight,
        area=100.0,
        depth=3.0,
        visible=1.0,
        facing=1.0,
        valid_fraction=1.0,
        coverage_mean=1.0,
        partial=False,
        patch=patch,
        valid=valid,
        coverage=coverage,
    )


def _pin() -> "e1.PinVerification":
    return e1.PinVerification(
        role="template-6",
        fixture="templates 6",
        reviewed=True,
        pinned=True,
        reference_sha256="a" * 64,
        frames=(("frame.png", "b" * 64),),
        usable_masks=(("frame.png", "c" * 64, "d" * 64),),
        missing_fields=(),
        mismatched_fields=(),
    )


def _artifact(patches) -> dict:
    return e1.build_artifact(
        role="template-6",
        pin=_pin(),
        inputs={"fixture_dir": "synthetic"},
        sampler={"patch_size": e1.PATCH_SIZE},
        denominators={
            "all_visible_cells": 1,
            "scorer_usable_cells": 1,
            "geometry_labelable_cells": 1,
            "model_vocabulary_cells": 1,
        },
        coverage={"frozen_patches": len(patches)},
        patches=patches,
        scorer={"available": False, "reason": "synthetic"},
        reconciliation={"claimed_equal": False},
    )


# --------------------------------------------------------------------------- #
# State / face labels
# --------------------------------------------------------------------------- #


def test_exact_cube_derives_one_state_face_label():
    result = _derive()

    assert len(result.samples) == 1
    patch = result.samples[0]
    assert patch.coordinate == (0, 0, 0)
    assert patch.slot == "north"
    assert patch.truth_state == STATE
    assert patch.valid_fraction == pytest.approx(1.0)
    assert patch.partial is False
    assert patch.patch.shape == (e1.PATCH_SIZE, e1.PATCH_SIZE, 3)
    assert np.allclose(patch.patch[patch.valid], 160.0, atol=1.0)
    assert result.exact_cells == 1
    assert result.refused_cells == 0


def test_hud_mask_abstains_without_labelling():
    result = _derive(usable=np.zeros((100, 100), dtype=bool))

    assert result.samples == ()
    assert result.sampler_events.get("gui_mask", 0) >= 1


def test_partial_geometry_is_refused_not_cube_labelled():
    palette = (PaletteColor(PLAIN, (160, 160, 160)),)  # no exact model polygons

    result = _derive(state=PLAIN, palette=palette)

    assert result.samples == ()
    assert result.refusals.get(e1.MODEL_METADATA_UNAVAILABLE) == 1
    assert result.exact_cells == 0
    assert result.refused_cells == 1
    assert result.refused_state_faces >= 1


def test_unmodelled_cell_still_occludes_so_an_exact_face_is_not_labelled():
    # PLAIN sits between the camera and STATE's north face. Because the refused
    # cell is still rendered (conservatively as a cube), STATE must not be
    # labelled through it; if the refused cell were dropped from the render, the
    # occluded exact face would be silently labelled.
    image, camera, scene = _synthetic_frame()
    palette = (
        _exact_palette()[0],
        PaletteColor(PLAIN, (20, 20, 20)),
    )

    result = e1.derive_frame_labels(
        "frame.png",
        image,
        camera,
        scene,
        {(0, 0, 0): STATE, (0, 0, -2): PLAIN},
        palette,
        np.ones((100, 100), dtype=bool),
        minimum_area=1.0,
        minimum_visible_fraction=0.1,
        shrink=1.0,
    )

    assert result.samples == ()
    assert result.refusals.get(e1.MODEL_METADATA_UNAVAILABLE) == 1
    assert result.sampler_events.get("visible_fraction", 0) >= 1


def test_cutout_pixels_are_excluded_and_marked_partial():
    palette = _exact_palette(alpha=_encoded_alpha(slice(4, 16)))

    result = _derive(palette=palette)

    assert len(result.samples) == 1
    patch = result.samples[0]
    assert patch.partial is True
    assert 0.5 <= patch.valid_fraction < 1.0
    assert int(patch.valid.sum()) < e1.PATCH_SIZE * e1.PATCH_SIZE
    assert not patch.valid.all()


def test_classify_sample_refuses_sparse_ownership():
    valid = np.zeros((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=bool)
    valid[0, 0] = True
    fake = types.SimpleNamespace(valid=valid, coverage=valid.astype(np.float32))

    ok, reason, valid_fraction, _coverage = e1.classify_sample(fake)

    assert ok is False
    assert reason == e1.REASON_INSUFFICIENT_VALID_COVERAGE
    assert valid_fraction < e1.MINIMUM_PATCH_VALID_FRACTION


def test_patch_sample_refuses_unrelated_image_pixels():
    big = np.zeros((100, 100, 3), dtype=np.float32)
    valid = np.ones((100, 100), dtype=bool)
    coverage = np.ones((100, 100), dtype=np.float32)

    with pytest.raises(e1.PatchLabelError):
        e1.PatchSample(
            frame_id="frame.png",
            coordinate=(0, 0, 0),
            slot="north",
            truth_state=STATE,
            weight=1.0,
            area=1.0,
            depth=1.0,
            visible=1.0,
            facing=1.0,
            valid_fraction=1.0,
            coverage_mean=1.0,
            partial=False,
            patch=big,
            valid=valid,
            coverage=coverage,
        )


def test_observation_weight_exactly_matches_production_formula_including_uniformity():
    # A quadrant-disagreeing patch with fractional coverage: the production
    # divisor is 1 + _face_uniformity(observed_patch, coverage)/UNIFORMITY_SCALE.
    patch = np.zeros((e1.PATCH_SIZE, e1.PATCH_SIZE, 3), dtype=np.float32)
    patch[0:8, 0:8] = 100.0
    patch[0:8, 8:16] = 120.0
    patch[8:16, 0:8] = 140.0
    patch[8:16, 8:16] = 160.0
    coverage = np.ones((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=np.float32)
    valid = np.ones((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=bool)
    sample = types.SimpleNamespace(
        patch=patch, coverage=coverage, valid=valid,
        area=100.0, depth=3.0, visible=0.8, facing=0.5,
    )

    assert e1.raycast.VISIBILITY_POWER == 0.0
    assert e1.raycast.OBLIQUITY_POWER == 0.0
    observed = e1.raycast._face_uniformity(patch.astype(np.float64), coverage.astype(np.float64))
    expected = 100.0 ** e1.raycast.AREA_POWER / (1.0 + 3.0 / 16.0)
    if e1.raycast.VISIBILITY_POWER:
        expected *= 0.8 ** e1.raycast.VISIBILITY_POWER
    if e1.raycast.OBLIQUITY_POWER:
        expected *= 0.5 ** e1.raycast.OBLIQUITY_POWER
    expected /= 1.0 + observed / float(e1.raycast.UNIFORMITY_SCALE)

    assert observed > 0.0
    assert e1.observation_weight(sample) == pytest.approx(expected, rel=1e-12)
    # The divisor is not decorative: omitting it changes the recorded weight.
    omitted = expected * (1.0 + observed / float(e1.raycast.UNIFORMITY_SCALE))
    assert e1.observation_weight(sample) != pytest.approx(omitted, rel=1e-12)


def test_observation_weight_reproduces_production_pipeline_weight():
    patch = np.full((e1.PATCH_SIZE, e1.PATCH_SIZE, 3), 160.0, dtype=np.float32)
    patch[8:16, 8:16] = 40.0
    coverage = np.full((e1.PATCH_SIZE, e1.PATCH_SIZE), 0.75, dtype=np.float32)
    valid = np.ones((e1.PATCH_SIZE, e1.PATCH_SIZE), dtype=bool)
    valid[0, 0] = False  # one unowned pixel, replaced by the face mean
    sample = types.SimpleNamespace(
        patch=patch, coverage=coverage, valid=valid,
        area=42.0, depth=7.5, visible=0.42, facing=0.9,
    )

    # Literal transcription of raycast.reconstruct_scene_surfaces' weight block.
    pixels = patch[valid]
    weights = coverage[valid]
    rgb = np.average(pixels, axis=0, weights=weights)
    observed_patch = patch.copy()
    observed_patch[~valid] = rgb
    uniformity = e1.raycast._face_uniformity(observed_patch, coverage)
    weight = float(sample.area ** e1.raycast.AREA_POWER) / (1.0 + sample.depth / 16.0)
    if e1.raycast.VISIBILITY_POWER:
        weight *= float(max(sample.visible, 0.0) ** e1.raycast.VISIBILITY_POWER)
    if e1.raycast.OBLIQUITY_POWER:
        weight *= float(max(sample.facing, 0.0) ** e1.raycast.OBLIQUITY_POWER)
    weight /= 1.0 + uniformity / max(float(e1.raycast.UNIFORMITY_SCALE), 1e-6)

    assert e1.observation_weight(sample) == pytest.approx(weight, rel=1e-12)


# --------------------------------------------------------------------------- #
# Scorer read over frozen patches
# --------------------------------------------------------------------------- #


def test_existing_scorer_scores_frozen_patches_and_keeps_denominators_separate():
    palette = (
        PaletteColor(STATE, (160, 160, 160), model_faces=_exact_palette()[0].model_faces),
        PaletteColor(OTHER, (10, 10, 10)),
    )

    scorer = e1.score_patches([_sample()], palette)

    assert scorer["available"] is True
    assert scorer["scored_patches"] == 1
    assert scorer["truth_in_vocabulary_patches"] == 1
    assert scorer["truth_out_of_vocabulary_patches"] == 0
    assert scorer["ordered_top_k"]["1"]["matches"] == 1
    assert scorer["ordered_top_k"]["8"]["matches"] == 1
    assert scorer["ordered_top_k"]["1"]["denominator"] == 1
    assert scorer["production_set_membership_claimed"] is False
    assert scorer["texture_weight"] == e1.raycast.TEXTURE_WEIGHT


def test_scorer_texture_weight_defaults_to_production_and_is_overridable():
    palette = (PaletteColor(STATE, (160, 160, 160)),)

    assert e1.score_patches([_sample()], palette)["texture_weight"] == e1.raycast.TEXTURE_WEIGHT
    assert e1.score_patches([_sample()], palette, texture_weight=1.5)["texture_weight"] == 1.5


def test_truth_state_absent_from_canonical_scoring_palette_is_out_of_vocabulary():
    # canonical scoring palette deliberately omits STATE's truth
    scoring_palette = (PaletteColor(OTHER, (10, 10, 10)),)

    scorer = e1.score_patches([_sample()], scoring_palette)

    assert scorer["available"] is True
    assert scorer["scored_patches"] == 1
    assert scorer["truth_out_of_vocabulary_patches"] == 1
    assert scorer["truth_in_vocabulary_patches"] == 0
    for k in e1.TOP_K:
        assert scorer["ordered_top_k"][str(k)]["matches"] == 0
        assert scorer["ordered_top_k"][str(k)]["denominator"] == 0
        assert scorer["ordered_top_k"][str(k)]["recall"] is None


def test_geometry_labelable_truth_outside_scoring_vocabulary_is_never_an_oracle_candidate():
    # STATE is exact-model geometry (geometry_palette carries model_faces) but the
    # scoring vocabulary omits it: geometry presence must not admit it to scoring.
    geometry_palette = _exact_palette(STATE)
    scoring_palette = (PaletteColor(OTHER, (10, 10, 10)),)

    assert STATE in e1.exact_model_states(geometry_palette)
    assert STATE not in e1.palette_states(scoring_palette)

    scorer = e1.score_patches([_sample()], scoring_palette)

    assert scorer["truth_in_vocabulary_patches"] == 0
    assert scorer["truth_out_of_vocabulary_patches"] == 1
    assert scorer["ordered_top_k"]["8"]["matches"] == 0


def test_truth_state_present_only_in_geometry_palette_is_not_scored_as_vocabulary():
    # Regression guard for the palette swap: building the scoring palette from
    # reference truth admits STATE, so the same OOV check fails to fire.
    truth_built_palette = (
        PaletteColor(STATE, (160, 160, 160), model_faces=_exact_palette()[0].model_faces),
        PaletteColor(OTHER, (10, 10, 10)),
    )

    scorer = e1.score_patches([_sample()], truth_built_palette)

    assert scorer["truth_in_vocabulary_patches"] == 1
    assert scorer["truth_out_of_vocabulary_patches"] == 0


def test_scorer_reports_empty_palette_as_unavailable():
    assert e1.score_patches([_sample()], ()) == {"available": False, "reason": "empty_palette"}


def _half_thumbnail(*, top_bright: bool) -> str:
    """A 16x16 RGB face texture bright on one half and dark on the other.

    The two orientations are exact complements, so their zero-mean, unit-norm
    luminance vectors correlate at ``-1``: a patch matching one is maximally
    mismatched against the other.
    """
    from PIL import Image

    pixels = np.zeros((e1.PATCH_SIZE, e1.PATCH_SIZE, 3), dtype=np.uint8)
    half = e1.PATCH_SIZE // 2
    pixels[:half] = 255 if top_bright else 0
    pixels[half:] = 0 if top_bright else 255
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _patterned_palette() -> tuple[PaletteColor, ...]:
    """STATE and OTHER share one colour and differ only in their north pattern."""
    return (
        PaletteColor(
            STATE,
            (128, 128, 128),
            face_thumbnails=(("north", _half_thumbnail(top_bright=True)),),
        ),
        PaletteColor(
            OTHER,
            (128, 128, 128),
            face_thumbnails=(("north", _half_thumbnail(top_bright=False)),),
        ),
    )


def _top_bright_patch() -> np.ndarray:
    patch = np.zeros((e1.PATCH_SIZE, e1.PATCH_SIZE, 3), dtype=np.float32)
    patch[: e1.PATCH_SIZE // 2] = 255.0
    return patch


def test_score_patches_requests_the_texture_pattern_term():
    scorer = e1.score_patches([_sample()], _patterned_palette())

    assert scorer["texture_weight"] == e1.raycast.TEXTURE_WEIGHT
    assert scorer["use_pattern"] is True


def test_texture_pattern_term_changes_frozen_patch_ranking():
    # STATE and OTHER share a colour, so colour alone leaves them tied and the
    # canonical tie-break puts OTHER ("...e1_other_cube") ahead of the truth
    # STATE ("...e1_test_cube"). Only the texture pattern separates them, so the
    # truth can only reach rank 1 when the pattern term is actually applied.
    palette = _patterned_palette()
    sample = _sample(patch=_top_bright_patch())

    with_pattern = e1.score_patches([sample], palette)
    without_pattern = e1.score_patches([sample], palette, texture_weight=0.0)

    assert with_pattern["use_pattern"] is True
    assert with_pattern["truth_in_vocabulary_patches"] == 1
    assert with_pattern["ordered_top_k"]["1"]["matches"] == 1
    assert with_pattern["ordered_top_k"]["8"]["matches"] == 1

    # texture_weight=0 disables the pattern term inside the scorer, reproducing
    # the colour-tie ranking the harness produced for every weight while
    # use_pattern stayed False.
    assert without_pattern["texture_weight"] == 0.0
    assert without_pattern["ordered_top_k"]["1"]["matches"] == 0
    assert without_pattern["ordered_top_k"]["8"]["matches"] == 1


def test_score_patches_pattern_flag_matches_the_underlying_api():
    # Pin the API contract the harness relies on: the same frozen observation
    # ranked second by the default scorer (use_pattern=False) is ranked first
    # once the option is requested.
    from mc_photogrammetry.reconstruction.appearance_scoring import AppearanceScorer

    palette = _patterned_palette()
    sample = _sample(patch=_top_bright_patch())
    observation = e1.face_observation_from_sample(
        sample,
        frame_id=sample.frame_id,
        provenance=("e1-frozen-patch", f"frame:{sample.frame_id}"),
        weight=sample.weight,
    )
    scorer = AppearanceScorer(tuple(palette), texture_weight=float(e1.raycast.TEXTURE_WEIGHT))

    default_ranked = [state for state, _cost in scorer.score(observation).ranked()]
    pattern_ranked = [
        state for state, _cost in scorer.score(observation, use_pattern=True).ranked()
    ]

    assert default_ranked[0] == OTHER
    assert pattern_ranked[0] == STATE


# --------------------------------------------------------------------------- #
# Deterministic seal and frozen pixel domain
# --------------------------------------------------------------------------- #


def test_seal_is_deterministic_and_order_independent():
    first = _sample((0, 0, 0))
    second = _sample((1, 0, 0))

    forward = _artifact([first, second])
    reversed_input = _artifact([second, first])

    assert forward["artifact_sha256"] == reversed_input["artifact_sha256"]
    order = [tuple(entry["coordinate"]) for entry in forward["patches"]]
    assert order == sorted(order)


def test_seal_changes_when_a_patch_changes():
    baseline = _artifact([_sample((0, 0, 0))])
    tampered = _artifact([_sample((0, 0, 0), weight=2.0)])

    assert baseline["artifact_sha256"] != tampered["artifact_sha256"]
    assert e1.seal_record({key: value for key, value in baseline.items() if key != "artifact_sha256"}) == baseline["artifact_sha256"]


def test_patch_domain_refuses_non_frozen_shapes():
    record = _artifact([_sample()])
    assert e1.patch_domain_problems(record) == []

    damaged = copy.deepcopy(record)
    damaged["patches"][0]["patch"]["shape"] = [32, 32, 3]
    assert e1.patch_domain_problems(damaged)


def test_write_artifact_refuses_overwrite(tmp_path):
    record = _artifact([_sample()])
    target = tmp_path / "labels.json"

    e1.write_artifact(target, record)
    before = target.read_bytes()

    with pytest.raises(e1.PatchLabelError):
        e1.write_artifact(target, record)

    assert target.read_bytes() == before


# --------------------------------------------------------------------------- #
# Pin, mask and preflight guards
# --------------------------------------------------------------------------- #


def test_pin_manifest_rejects_non_integer_schema(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(
        json.dumps({"manifest_schema": True, "roles": {"template-2": {"reviewed": True}}}),
        encoding="utf-8",
    )
    assert e1.load_pin_manifest(path) == {}

    path.write_text(
        json.dumps(
            {
                "manifest_schema": 1,
                "roles": {"template-2": {"reviewed": True, "fixture": "templates 2"}},
            }
        ),
        encoding="utf-8",
    )
    assert set(e1.load_pin_manifest(path)) == {"template-2"}


def test_verify_role_pins_matches_and_flags_mismatch():
    specification = {"fixture": "templates 2"}
    expected = {
        "reviewed": True,
        "fixture": "templates 2",
        "reference_sha256": "a" * 64,
        "frames": [{"name": "frame.png", "sha256": "b" * 64}],
        "usable_masks": [
            {"frame": "frame.png", "domain_sha256": "c" * 64, "file_sha256": "d" * 64}
        ],
    }
    frames = [{"name": "frame.png", "sha256": "b" * 64}]

    assert e1.verify_role_pins(
        specification, expected, "a" * 64, frames, {"frame.png": "c" * 64}, {"frame.png": "d" * 64}
    ) == ([], [])

    missing, mismatched = e1.verify_role_pins(
        specification, expected, "e" * 64, frames, {"frame.png": "c" * 64}, {"frame.png": "d" * 64}
    )
    assert missing == []
    assert mismatched == ["reference_sha256"]

    unreviewed = dict(expected, reviewed=False)
    missing, mismatched = e1.verify_role_pins(
        specification, unreviewed, "a" * 64, frames, {"frame.png": "c" * 64}, {"frame.png": "d" * 64}
    )
    assert missing == ["reviewed"]
    assert mismatched == []

    assert e1.verify_role_pins(specification, None, "a" * 64, frames, {}, {})[0] == list(
        e1.REQUIRED_PIN_FIELDS
    )


def test_load_mask_spec_domain_and_bounds(tmp_path):
    spec = tmp_path / "mask.json"
    spec.write_text(
        json.dumps({"image_size": [10, 8], "excluded_rectangles": [[0, 0, 2, 1]]}),
        encoding="utf-8",
    )
    mask = e1.load_mask_spec(spec)
    assert mask.shape == (8, 10)
    assert not bool(mask[0, 0])
    assert bool(mask[7, 9])
    assert e1.mask_domain_sha256(mask) == e1.mask_domain_sha256(mask.copy())

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"image_size": [10, 8], "excluded_rectangles": [[0, 0, 99, 1]]}),
        encoding="utf-8",
    )
    with pytest.raises(e1.PatchLabelError):
        e1.load_mask_spec(bad)


_FRAME_NAME = "2026-09-02_13.22.14.png"
"""A pinned frame name with the real ``<timestamp>.png`` shape (dots included)."""

_FRAME_MASK_SPEC = "2026-09-02_13.22.14.json"
"""The tracked spec for that frame: same stem, ``.json``, never ``.png.json``."""


def _preflight_fixture(tmp_path: Path) -> dict[str, Path]:
    fixture = tmp_path / "templates 2"
    fixture.mkdir()
    (fixture / "Template 2.litematic").write_bytes(b"synthetic reference bytes")
    (fixture / "cameras.json").write_text("{}", encoding="utf-8")
    (fixture / "scene.json").write_text("{}", encoding="utf-8")
    (fixture / _FRAME_NAME).write_bytes(b"synthetic frame bytes")
    mask_dir = tmp_path / "masks" / "templates-2"
    mask_dir.mkdir(parents=True)
    (mask_dir / _FRAME_MASK_SPEC).write_text(
        json.dumps({"image_size": [10, 8]}), encoding="utf-8"
    )
    jar = tmp_path / "1.21.5.jar"
    jar.write_bytes(b"synthetic jar bytes")
    return {
        "fixture_dir": fixture,
        "reference": fixture / "Template 2.litematic",
        "camera_manifest": fixture / "cameras.json",
        "scene": fixture / "scene.json",
        "jar": jar,
        "mask_root": tmp_path / "masks",
    }


def test_preflight_refuses_unknown_role(tmp_path):
    with pytest.raises(e1.PatchLabelError):
        e1.preflight("template-99", **_preflight_fixture(tmp_path))


def _asset_pins(paths: dict[str, Path], role: str = "template-2") -> dict[str, dict[str, str]]:
    return {
        role: {
            "cameras": e1.sha256_file(paths["camera_manifest"]),
            "scene": e1.sha256_file(paths["scene"]),
        },
        e1.LOCAL_JAR_PIN_KEY: {"jar": e1.sha256_file(paths["jar"])},
    }


def _manifest(paths: dict[str, Path], role: str = "template-2") -> dict[str, dict]:
    return {
        role: {
            "reviewed": True,
            "fixture": "templates 2",
            "reference_sha256": e1.sha256_file(paths["reference"]),
            "frames": [
                {"name": _FRAME_NAME, "sha256": e1.sha256_file(paths["fixture_dir"] / _FRAME_NAME)}
            ],
            "usable_masks": [
                {"frame": _FRAME_NAME, "domain_sha256": e1.mask_domain_sha256(np.ones((8, 10), dtype=bool))}
            ],
        }
    }


def test_preflight_accepts_matching_asset_pins(tmp_path):
    paths = _preflight_fixture(tmp_path)

    pre = e1.preflight(
        "template-2",
        pin_manifest=_manifest(paths),
        expected_asset_pins=_asset_pins(paths),
        **paths,
    )

    assert pre.asset_sha256["cameras"] == e1.sha256_file(paths["camera_manifest"])
    assert pre.asset_sha256["scene"] == e1.sha256_file(paths["scene"])
    assert pre.asset_sha256["jar"] == e1.sha256_file(paths["jar"])
    assert pre.asset_pin_problems == ()
    assert pre.pin.pinned is True


def test_preflight_maps_png_frames_to_stem_matched_mask_specs(tmp_path):
    # Pinned frames are `<timestamp>.png` but the tracked specs are
    # `<timestamp>.json`. Requesting `<timestamp>.png.json` refused every pinned
    # frame; the mapping must use the frame stem, not the frame name.
    paths = _preflight_fixture(tmp_path)

    pre = e1.preflight(
        "template-2",
        pin_manifest=_manifest(paths),
        expected_asset_pins=_asset_pins(paths),
        **paths,
    )

    mask_dir = paths["mask_root"] / "templates-2"
    assert pre.mask_paths[_FRAME_NAME] == mask_dir / _FRAME_MASK_SPEC
    assert pre.mask_paths[_FRAME_NAME].is_file()
    assert not (mask_dir / f"{_FRAME_NAME}.json").exists()
    assert e1.mask_spec_name_for_frame(_FRAME_NAME) == _FRAME_MASK_SPEC
    # A naive split on the first dot would lose the timestamp's later components.
    assert e1.mask_spec_name_for_frame(_FRAME_NAME) != f"{_FRAME_NAME.split('.')[0]}.json"


def test_preflight_refuses_missing_stem_matched_mask_spec(tmp_path):
    paths = _preflight_fixture(tmp_path)
    mask_dir = paths["mask_root"] / "templates-2"
    # Leave only the doubled-suffix spec the buggy mapping used to request: the
    # missing stem-matched spec must still refuse before any image decode.
    (mask_dir / _FRAME_MASK_SPEC).rename(mask_dir / f"{_FRAME_NAME}.json")

    with pytest.raises(e1.PatchLabelError) as error:
        e1.preflight(
            "template-2",
            pin_manifest=_manifest(paths),
            expected_asset_pins=_asset_pins(paths),
            **paths,
        )

    assert _FRAME_MASK_SPEC in str(error.value)
    assert not (mask_dir / _FRAME_MASK_SPEC).exists()


@pytest.mark.parametrize("asset", ["cameras", "scene", "jar"])
def test_preflight_refuses_asset_pin_mismatch(tmp_path, asset):
    paths = _preflight_fixture(tmp_path)
    pins = _asset_pins(paths)
    key = e1.LOCAL_JAR_PIN_KEY if asset == "jar" else "template-2"
    pins[key] = dict(pins[key], **{asset: "f" * 64})

    with pytest.raises(e1.PatchLabelError) as error:
        e1.preflight("template-2", expected_asset_pins=pins, **paths)

    assert asset in str(error.value)


def test_preflight_refuses_role_with_no_declared_asset_pins(tmp_path):
    paths = _preflight_fixture(tmp_path)

    with pytest.raises(e1.PatchLabelError) as error:
        e1.preflight("template-2", expected_asset_pins={}, **paths)

    assert "asset pins" in str(error.value)


def test_verify_asset_pins_reports_each_mismatched_asset():
    observed = {"cameras": "a" * 64, "scene": "b" * 64, "jar": "c" * 64}
    pins = {
        "template-6": {"cameras": "a" * 64, "scene": "0" * 64},
        e1.LOCAL_JAR_PIN_KEY: {"jar": "1" * 64},
    }

    problems = e1.verify_asset_pins("template-6", observed, pins)

    assert len(problems) == 2
    assert any("scene" in item for item in problems)
    assert any("jar" in item for item in problems)


def test_run_creates_no_output_when_pins_mismatch(tmp_path):
    paths = _preflight_fixture(tmp_path)
    manifest = _manifest(paths)
    manifest["template-2"]["reference_sha256"] = "0" * 64  # not the supplied digest
    output = tmp_path / "labels.json"

    with pytest.raises(e1.PatchLabelError):
        e1.run(
            "template-2",
            output=output,
            pin_manifest=manifest,
            expected_asset_pins=_asset_pins(paths),
            **paths,
        )

    assert not output.exists()


@pytest.mark.parametrize("asset", ["cameras", "scene", "jar"])
def test_run_creates_no_output_when_asset_pin_mismatches(tmp_path, asset):
    paths = _preflight_fixture(tmp_path)
    pins = _asset_pins(paths)
    key = e1.LOCAL_JAR_PIN_KEY if asset == "jar" else "template-2"
    pins[key] = dict(pins[key], **{asset: "f" * 64})
    output = tmp_path / "labels.json"

    with pytest.raises(e1.PatchLabelError):
        e1.run(
            "template-2",
            output=output,
            pin_manifest=_manifest(paths),
            expected_asset_pins=pins,
            **paths,
        )

    assert not output.exists()


def test_run_refuses_existing_output_before_preflight(tmp_path):
    paths = _preflight_fixture(tmp_path)
    output = tmp_path / "labels.json"
    output.write_text("already here", encoding="utf-8")

    with pytest.raises(e1.PatchLabelError):
        e1.run("template-2", output=output, pin_manifest={}, **paths)

    assert output.read_text(encoding="utf-8") == "already here"


# --------------------------------------------------------------------------- #
# Production visibility palette vs reference geometry palette
# --------------------------------------------------------------------------- #

_SHARED_NAME = "minecraft:e1_shared_name"
_SHARED_FULL = BlockState(_SHARED_NAME, (("type", "double"),))
_SHARED_PARTIAL = BlockState(_SHARED_NAME, (("type", "bottom"),))


def _visibility_samples() -> list[TextureSample]:
    """The raw extraction table: one full-cube and one partial state share a name.

    A reference restricted to the full-cube state makes the bare name look
    unconditionally solid, which is exactly how a geometry-palette visibility
    domain silently culls the face behind the partial sibling.
    """
    return [
        TextureSample(
            _SHARED_FULL,
            (128, 128, 128),
            "assets/minecraft/textures/block/e1_shared_full.png",
            full_cube=True,
        ),
        TextureSample(
            _SHARED_PARTIAL,
            (128, 128, 128),
            "assets/minecraft/textures/block/e1_shared_partial.png",
            full_cube=False,
            shape_boxes=(((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),),
        ),
    ]


def test_production_visibility_palette_is_raw_samples_and_refuses_empty():
    samples = _visibility_samples()

    assert e1.production_visibility_palette(types.SimpleNamespace(samples=samples)) is samples

    with pytest.raises(e1.PatchLabelError):
        e1.production_visibility_palette(types.SimpleNamespace(samples=[]))
    with pytest.raises(e1.PatchLabelError):
        e1.production_visibility_palette(types.SimpleNamespace())


def test_visibility_domain_is_built_from_production_samples_not_geometry_palette():
    samples = _visibility_samples()
    provider = types.SimpleNamespace(samples=samples)
    # A reference-only geometry palette holds just the full-cube state.
    geometry_palette = _exact_palette(_SHARED_FULL)

    arguments = e1.visibility_domain_arguments(e1.production_visibility_palette(provider))

    assert arguments["palette"] is samples
    assert arguments["full_cubes"] == e1.full_cube_keys(samples)
    assert arguments["shapes"] == e1.shape_table(samples)
    # The partial sibling is present, so the bare name is not an unconditional
    # full cube. The geometry palette loses the sibling and would cull more faces.
    assert _SHARED_NAME not in arguments["full_cubes"]
    assert _SHARED_NAME in e1.full_cube_keys(geometry_palette)
    assert _SHARED_PARTIAL in arguments["shapes"]
    assert _SHARED_PARTIAL not in e1.shape_table(geometry_palette)
    assert arguments["full_cubes"] != e1.full_cube_keys(geometry_palette)


def test_run_renders_visibility_from_production_samples_and_labels_from_geometry(
    monkeypatch, tmp_path
):
    paths = _preflight_fixture(tmp_path)
    samples = _visibility_samples()
    geometry_palette = _exact_palette(_SHARED_FULL)
    scoring_palette = (PaletteColor(STATE, (160, 160, 160)),)
    snapshot = types.SimpleNamespace(palette=(_SHARED_FULL,), blocks={(0, 0, 0): _SHARED_FULL})
    captured: dict[str, object] = {}

    class _FakeProvider:
        def __init__(self):
            self.samples = samples

        def load_jar(self, path):
            return len(self.samples)

        def palette_for_states(self, states, *, include_partial):
            return geometry_palette

        def canonical_palette(self):
            return scoring_palette

    def fake_derive(frame_id, image, camera, scene, blocks, palette, usable_mask, **kwargs):
        captured["label_palette"] = palette
        return e1.FrameLabelResult(
            frame_id=frame_id,
            samples=(),
            refusals={},
            sampler_events={},
            sample_rejections={},
            image_pixels=int(image.shape[0] * image.shape[1]),
            usable_pixels=int(np.count_nonzero(usable_mask)),
            exact_cells=0,
            refused_cells=0,
            refused_state_faces=0,
            duplicate_faces=0,
        )

    def fake_visible(blocks, scene, cameras, resolutions, **kwargs):
        captured["visible_palette"] = kwargs["palette"]
        captured["full_cubes"] = kwargs["full_cubes"]
        captured["shapes"] = kwargs["shapes"]
        return {}

    monkeypatch.setattr(e1, "VanillaJarTextureProvider", _FakeProvider)
    monkeypatch.setattr(e1, "load_litematic", lambda path: snapshot)
    monkeypatch.setattr(
        e1,
        "load_camera_manifest",
        lambda path: {
            _FRAME_NAME: CameraModel(
                (0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0
            )
        },
    )
    monkeypatch.setattr(e1, "load_scene_transform", lambda path: SceneTransform((0, 0, 0), (2, 2, 2)))
    monkeypatch.setattr(e1, "load_image", lambda path: np.full((8, 10, 3), 160, dtype=np.uint8))
    monkeypatch.setattr(e1, "derive_frame_labels", fake_derive)
    monkeypatch.setattr(e1, "visible_cells", fake_visible)

    output = tmp_path / "labels.json"
    e1.run(
        "template-2",
        output=output,
        pin_manifest=_manifest(paths),
        expected_asset_pins=_asset_pins(paths),
        include_scorer=False,
        **paths,
    )

    # Face-label ownership stays on the exact reference-state geometry palette...
    assert captured["label_palette"] is geometry_palette
    # ...while the visible-cell domain uses the production extraction table.
    assert captured["visible_palette"] is samples
    assert captured["full_cubes"] == e1.full_cube_keys(samples)
    assert captured["shapes"] == e1.shape_table(samples)
    assert captured["full_cubes"] != e1.full_cube_keys(geometry_palette)
