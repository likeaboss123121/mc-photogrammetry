"""Focused tests for the evaluation-only E3 factory patch-dataset exporter.

Every test here runs on synthetic geometry, synthetic screenshots and synthetic
palettes in ``tmp_path``. No test loads a factory corpus, a real reference
Litematic, camera manifest, scene definition or JAR, decodes a real image, runs a
training model or writes outside ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import box_face_corners

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "export_factory_texture_patch_dataset.py"
)
_spec = importlib.util.spec_from_file_location(
    "export_factory_texture_patch_dataset_under_test", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
e3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e3  # dataclasses(slots=True) resolves the module by name
_spec.loader.exec_module(e3)

STATE = BlockState("minecraft:e3_test_cube")
PLAIN = BlockState("minecraft:e3_unmodelled")

IMAGE_SIZE = (100, 100)


# --------------------------------------------------------------------------- #
# Synthetic palette, geometry and image
# --------------------------------------------------------------------------- #


def _exact_palette(state: BlockState = STATE, *, rgb: tuple[int, int, int] = (160, 160, 160)):
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    record = ("north", corners, (0.0, 0.0, -1.0), None, None)
    return PaletteColor(state, rgb, model_faces=(record,))


def _provider(*states: BlockState):
    palette = tuple(_exact_palette(state) for state in (states or (STATE,)))

    class _FakeProvider:
        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.entries = palette

        def load_jar(self, path) -> int:
            self.loaded.append(str(path))
            return len(self.entries)

        def palette_for_states(self, states, *, include_partial: bool = True):
            wanted = set(states)
            return tuple(item for item in self.entries if item.state in wanted)

    return _FakeProvider()


class _Snapshot:
    def __init__(self, palette, blocks) -> None:
        self.palette = tuple(palette)
        self.blocks = dict(blocks)


def _image_loader(image: np.ndarray | None = None):
    pixels = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 160, dtype=np.uint8)
    if image is not None:
        pixels = np.asarray(image, dtype=np.uint8)

    def load(path):
        return pixels.copy()

    return load


def _reference_loader(snapshots):
    def load(path):
        scene_id = Path(path).parent.name
        return snapshots[scene_id]

    return load


# --------------------------------------------------------------------------- #
# Synthetic dataset fixture
# --------------------------------------------------------------------------- #


_MISSING = object()


def _valid_actual(**overrides):
    actual = {
        "feet_position": [0.5, -1.12, -3.0],
        "eye_position": [0.5, 0.5, -3.0],
        "yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
        "fov_degrees": 90.0,
        "gamma": 0.0,
        "hud_hidden": True,
        "window_width": IMAGE_SIZE[0],
        "window_height": IMAGE_SIZE[1],
        "framebuffer_width": IMAGE_SIZE[0],
        "framebuffer_height": IMAGE_SIZE[1],
        "time_of_day": 6000,
        "weather": "clear",
        "active_resource_packs": ["vanilla"],
        "resource_pack_applied": False,
        "shader_pack_applied": False,
        "requested": {
            "resource_pack": None,
            "shader_pack": None,
            "fov_degrees": 90,
            "gamma": 0.0,
            "hud_hidden": True,
            "time_of_day": 6000,
            "weather": "clear",
            "framebuffer_width": IMAGE_SIZE[0],
            "framebuffer_height": IMAGE_SIZE[1],
        },
    }
    for key, value in overrides.items():
        if value is _MISSING:
            actual.pop(key, None)
        else:
            actual[key] = value
    return actual


def _frame(frame_id: str = "frame-0", filename: str = "frame-0.png", *,
           actual=_MISSING, include_actual: bool = True, captured: bool = True):
    frame = {
        "schema_version": 1,
        "frame_id": frame_id,
        "filename": filename,
        "captured": captured,
        "screenshot": f"screenshots/{filename}",
        "settled_ticks": 20,
        "errors": [],
    }
    if include_actual:
        frame["actual"] = _valid_actual() if actual is _MISSING else actual
    return frame


def _scene(scene_id: str, *, group: str | None = None, split: str = "train", blocks=None,
           palette_states=None, frames=None, region_min=(0, 0, 0), region_max=(1, 1, 1),
           staging_token: str | None = None, world_identity: str | None = None,
           drop_files=(), directory: str | None = None, reference_blocks=None,
           planned_cameras=None):
    resolved_blocks = blocks if blocks is not None else {(0, 0, 0): STATE}
    return {
        "scene_id": scene_id,
        "group": group if group is not None else f"group-{scene_id}",
        "split": split,
        "blocks": resolved_blocks,
        "reference_blocks": reference_blocks if reference_blocks is not None else resolved_blocks,
        "palette_states": tuple(palette_states) if palette_states is not None
        else tuple(resolved_blocks.values()),
        "frames": frames if frames is not None else [_frame()],
        "planned_cameras": planned_cameras,
        "region_min": tuple(region_min),
        "region_max": tuple(region_max),
        "staging_token": staging_token if staging_token is not None
        else hashlib.sha256(f"staging-{scene_id}".encode()).hexdigest(),
        "world_identity": world_identity if world_identity is not None
        else f"world-{scene_id}",
        "drop_files": tuple(drop_files),
        "directory": directory if directory is not None else f"scenes/{scene_id}",
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_dataset(tmp_path: Path, specs, *, advanced_only: bool = True):
    root = tmp_path / "dataset"
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, _Snapshot] = {}
    index_scenes = []
    for spec in specs:
        index_scenes.append(_write_scene(root, spec, snapshots))
    index = {
        "schema_version": 1,
        "generator_version": "ds17r-1",
        "mod_contract_version": 1,
        "advanced_only": advanced_only,
        "usage": "developer/evaluation ground truth only; never normal reconstruction input",
        "summary": {"scenes": len(index_scenes)},
        "scenes": index_scenes,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return root, snapshots


def _write_scene(root: Path, spec, snapshots) -> dict:
    scene_id = spec["scene_id"]
    minimum = spec["region_min"]
    maximum = spec["region_max"]
    scene_dir = root / "scenes" / scene_id
    (scene_dir / "captures").mkdir(parents=True, exist_ok=True)

    blocks = spec["blocks"]
    ground_truth = {
        "schema_version": 1,
        "mod_contract_version": 1,
        "scene_id": scene_id,
        "world_identity": spec["world_identity"],
        "region": {"min": list(minimum), "max": list(maximum)},
        "blocks": [
            {
                "pos": [coordinate[axis] + minimum[axis] for axis in range(3)],
                "name": state.name,
                "properties": {key: value for key, value in state.properties},
            }
            for coordinate, state in sorted(blocks.items())
        ],
    }
    frames = spec["frames"]
    screenshot_rows = []
    for frame in frames:
        filename = frame["filename"]
        payload = b"synthetic-png:" + f"{scene_id}/{filename}".encode()
        (scene_dir / "captures" / filename).write_bytes(payload)
        digest = _sha256_bytes(payload)
        screenshot_rows.append({
            "filename": filename,
            "sha256": digest,
            "source_sha256": digest,
            "bytes": len(payload),
            "frame_id": frame["frame_id"],
        })
    resolutions = []
    for frame in frames:
        actual = frame.get("actual")
        if isinstance(actual, dict) and isinstance(actual.get("framebuffer_width"), int):
            resolutions.append([actual["framebuffer_width"], actual["framebuffer_height"]])
        else:
            resolutions.append([IMAGE_SIZE[0], IMAGE_SIZE[1]])
    capture = {
        "schema_version": 1,
        "dataset_id": root.name,
        "scene_id": scene_id,
        "plan_id": f"plan-{scene_id}",
        "world_identity": spec["world_identity"],
        "region": {"min": list(minimum), "max": list(maximum)},
        "advanced_only": True,
        "usage": "developer/evaluation ground truth only; never normal reconstruction input",
        "frames": frames,
        "screenshots": screenshot_rows,
        "cameras": spec["planned_cameras"] if spec.get("planned_cameras") is not None
        else [{} for _ in frames],
        "resolutions": resolutions,
    }
    size = tuple(maximum[axis] - minimum[axis] + 1 for axis in range(3))
    recipe = {
        "schema_version": 1,
        "seed": 0,
        "family": "custom",
        "dimensions": list(size),
        "palette": [],
        "features": [],
        "source": "procedural",
    }
    reference_bytes = b"synthetic-litematic:" + scene_id.encode()
    (scene_dir / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    (scene_dir / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (scene_dir / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (scene_dir / "reference.litematic").write_bytes(reference_bytes)

    names = ["capture.json", "ground_truth.json", "recipe.json", "reference.litematic"]
    hashes = {}
    for name in names:
        if name in spec["drop_files"]:
            continue
        path = scene_dir / name
        if path.is_file():
            hashes[name] = _sha256_bytes(path.read_bytes())

    snapshots[scene_id] = _Snapshot(spec["palette_states"], spec["reference_blocks"])
    return {
        "scene_id": scene_id,
        "split": spec["split"],
        "family": "custom",
        "seed": 0,
        "mod_contract_version": 1,
        "captures": len(frames),
        "directory": spec["directory"],
        "topology_digest": hashlib.sha256(f"topo-{scene_id}".encode()).hexdigest(),
        "scene_group": spec["group"],
        "hashes": hashes,
        "screenshots": sorted(screenshot_rows, key=lambda row: row["filename"]),
        "provenance": {
            "generator_version": "ds17r-1",
            "source": "real-minecraft-capture",
            "world_identity": spec["world_identity"],
            "plan_id": f"plan-{scene_id}",
            "staging_token": spec["staging_token"],
            "split_identity": "scene-topology",
            "advanced_only": True,
            "usage": "developer/evaluation ground truth only; never normal reconstruction input",
        },
    }


def _export(tmp_path: Path, specs, *, provider=None, image_loader=None,
            output_name: str = "out.json", **kwargs):
    root, snapshots = _write_dataset(tmp_path, specs)
    provider = provider if provider is not None else _provider(STATE)
    loader = image_loader if image_loader is not None else _image_loader()
    record = e3.export_dataset(
        root,
        output=tmp_path / output_name,
        provider=provider,
        image_loader=loader,
        reference_loader=_reference_loader(snapshots),
        **kwargs,
    )
    return record, root, snapshots


# --------------------------------------------------------------------------- #
# Schema, integrity and patch domain
# --------------------------------------------------------------------------- #


def test_export_writes_sealed_evaluation_only_artifact(tmp_path):
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha")])

    assert record["schema"] == e3.SCHEMA
    assert record["engine_version"] == e3.ENGINE_VERSION
    assert record["evaluation_only"] is True
    assert record["training_permitted"] is False
    assert record["t2_t6_excluded"] is True
    assert record["artifact_sha256"] == e3.seal_record(record)
    assert (tmp_path / "out.json").is_file()


def test_example_carries_exact_label_arrays_and_digests(tmp_path):
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha")])

    assert record["coverage"]["examples"] == 1
    example = record["examples"][0]
    assert example["label"]["name"] == STATE.name
    assert example["slot"] == "north"
    assert example["coordinate"] == [0, 0, 0]
    assert example["world_coordinate"] == [0, 0, 0]
    assert example["scene_group"] == "group-alpha"
    assert example["build"]["staging_token"] == hashlib.sha256(b"staging-alpha").hexdigest()
    assert example["build"]["world_identity"] == "world-alpha"
    assert example["patch"]["shape"] == [e3.PATCH_SIZE, e3.PATCH_SIZE, 3]
    assert example["patch"]["dtype"] == np.dtype(np.float32).str
    assert example["valid"]["shape"] == [e3.PATCH_SIZE, e3.PATCH_SIZE]
    assert example["valid"]["dtype"] == np.dtype(bool).str
    assert example["coverage"]["shape"] == [e3.PATCH_SIZE, e3.PATCH_SIZE]
    assert example["coverage"]["dtype"] == np.dtype(np.float32).str

    patch = e3.array_from_record(example["patch"])
    valid = e3.array_from_record(example["valid"])
    coverage = e3.array_from_record(example["coverage"])
    assert example["digests"]["patch_sha256"] == _sha256_bytes(patch.tobytes(order="C"))
    assert example["digests"]["valid_sha256"] == _sha256_bytes(valid.tobytes(order="C"))
    assert example["digests"]["coverage_sha256"] == _sha256_bytes(coverage.tobytes(order="C"))

    recomputed = json.loads(json.dumps(example))
    recomputed["digests"].pop("example_sha256")
    assert example["digests"]["example_sha256"] == _sha256_bytes(e3.canonical_bytes(recomputed))


def test_local_label_coordinate_is_transformed_to_world_bounds(tmp_path):
    frames = [_frame(actual=_valid_actual(eye_position=[10.5, 60.5, -8.0]))]
    spec = _scene("shifted", blocks={(0, 0, 0): STATE}, frames=frames,
                  region_min=(10, 60, -5), region_max=(11, 61, -4))
    record, _root, _snapshots = _export(tmp_path, [spec])

    assert record["coverage"]["examples"] == 1
    example = record["examples"][0]
    assert example["coordinate"] == [0, 0, 0]
    assert example["world_coordinate"] == [10, 60, -5]
    scene_row = record["admission"]["scenes"][0]
    assert scene_row["region_min"] == [10, 60, -5]
    assert scene_row["region_max"] == [11, 61, -4]


def test_artifact_is_byte_deterministic_across_two_exports(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    provider = _provider(STATE)
    loader = _reference_loader(snapshots)
    first = e3.export_dataset(root, output=tmp_path / "first.json", provider=provider,
                              image_loader=_image_loader(), reference_loader=loader)
    second = e3.export_dataset(root, output=tmp_path / "second.json", provider=provider,
                               image_loader=_image_loader(), reference_loader=loader)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


def test_array_record_round_trip_and_malformed_refusal():
    array = np.arange(e3.PATCH_SIZE * e3.PATCH_SIZE * 3, dtype=np.float32).reshape(
        e3.PATCH_SIZE, e3.PATCH_SIZE, 3)
    record = e3.array_record(array)
    assert np.array_equal(e3.array_from_record(record), array)
    with pytest.raises(e3.FactoryPatchExportError):
        e3.array_from_record({"dtype": "<f4", "shape": [2, 2], "base64": "AAAA"})
    with pytest.raises(e3.FactoryPatchExportError):
        e3.array_from_record("not-an-object")


def test_patch_domain_refuses_non_frozen_shapes():
    good = e3.array_record(np.zeros((e3.PATCH_SIZE, e3.PATCH_SIZE, 3), dtype=np.float32))
    bad_record = {"examples": [{"patch": good, "valid": good, "coverage": good}]}
    assert e3.patch_domain_problems(bad_record)
    assert e3.patch_domain_problems({"examples": []}) == []
    assert e3.patch_domain_problems({}) == ["artifact has no example list"]


# --------------------------------------------------------------------------- #
# Finalized/scene admission and integrity
# --------------------------------------------------------------------------- #


def test_refuses_staging_unindexed_scene_directory(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    ghost = root / "scenes" / "ghost-scene"
    ghost.mkdir(parents=True)
    (ghost / "plan.json").write_text("{}", encoding="utf-8")

    with pytest.raises(e3.FactoryPatchExportError, match="staging/unindexed"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_incomplete_scene_missing_ground_truth(tmp_path):
    spec = _scene("alpha", drop_files=("ground_truth.json",))
    root, snapshots = _write_dataset(tmp_path, [spec])
    (root / "scenes" / "alpha" / "ground_truth.json").unlink(missing_ok=True)

    with pytest.raises(e3.FactoryPatchExportError, match="incomplete scene"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_tampered_scene_file_hash(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    (root / "scenes" / "alpha" / "reference.litematic").write_bytes(b"tampered")

    with pytest.raises(e3.FactoryPatchExportError, match="refusing dataset root"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_tampered_screenshot_hash(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    (root / "scenes" / "alpha" / "captures" / "frame-0.png").write_bytes(b"tampered")

    with pytest.raises(e3.FactoryPatchExportError, match="refusing dataset root"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_scene_directory_escape(tmp_path):
    root, snapshots = _write_dataset(
        tmp_path,
        [_scene("alpha", directory="../outside")],
    )
    # The escape target exists with matching hashes and screenshots, so only the
    # scene-directory path check can catch it.
    outside = tmp_path / "outside"
    shutil.copytree(root / "scenes" / "alpha", outside)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    hashes = {}
    for name in ("capture.json", "ground_truth.json", "recipe.json", "reference.litematic"):
        hashes[name] = _sha256_bytes((outside / name).read_bytes())
    index["scenes"][0]["hashes"] = hashes
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(e3.FactoryPatchExportError,
                       match="unsafe scene directory"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_escaping_index_paths_before_shared_discovery(tmp_path, monkeypatch):
    root, _snapshots = _write_dataset(
        tmp_path, [_scene("alpha", directory="../outside")])
    outside = tmp_path / "outside"
    shutil.copytree(root / "scenes" / "alpha", outside)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["scenes"][0]["hashes"] = {
        name: _sha256_bytes((outside / name).read_bytes())
        for name in ("capture.json", "ground_truth.json", "recipe.json", "reference.litematic")
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")

    def discovery_must_not_run(_root):
        raise AssertionError("shared discovery ran before path preflight")

    monkeypatch.setattr(e3, "discover_finalized_entries", discovery_must_not_run)
    with pytest.raises(e3.FactoryPatchExportError, match="unsafe scene directory"):
        e3.load_finalized_scenes(root)


def test_refuses_escaping_screenshot_before_shared_discovery(tmp_path, monkeypatch):
    root, _snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["scenes"][0]["screenshots"][0]["filename"] = "../outside.png"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    def discovery_must_not_run(_root):
        raise AssertionError("shared discovery ran before screenshot-path preflight")

    monkeypatch.setattr(e3, "discover_finalized_entries", discovery_must_not_run)
    with pytest.raises(e3.FactoryPatchExportError, match="unsafe screenshot filename"):
        e3.load_finalized_scenes(root)


def test_refuses_missing_index(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(e3.FactoryPatchExportError, match="refusing dataset root"):
        e3.export_dataset(empty, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(), reference_loader=_reference_loader({}))


def test_refuses_t2_t6_fixture_root(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    fixture_root = tmp_path / "templates 6"
    shutil.move(str(root), str(fixture_root))

    with pytest.raises(e3.FactoryPatchExportError, match="excluded T2/T6"):
        e3.export_dataset(fixture_root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_unknown_index_split(tmp_path):
    with pytest.raises(e3.FactoryPatchExportError, match="unknown index split"):
        _export(tmp_path, [_scene("alpha", split="holdout")])


def test_refuses_missing_staging_token(tmp_path):
    spec = _scene("alpha")
    spec["staging_token"] = "not-a-digest"
    with pytest.raises(e3.FactoryPatchExportError, match="staging_token"):
        _export(tmp_path, [spec])


def test_refuses_non_advanced_index(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")], advanced_only=False)
    with pytest.raises(e3.FactoryPatchExportError, match="advanced-only"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


# --------------------------------------------------------------------------- #
# Capture admission: actual pose, HUD and vanilla packs
# --------------------------------------------------------------------------- #


def test_hud_visible_capture_is_refused_and_recorded(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png", actual=_valid_actual(hud_hidden=False))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])

    assert record["coverage"]["examples"] == 1
    assert record["admission"]["refusals"] == {e3.REASON_HUD_NOT_HIDDEN: 1}
    frame_rows = {row["frame_id"]: row for row in record["admission"]["frames"]}
    assert frame_rows["frame-0"]["admitted"] is True
    assert frame_rows["frame-1"]["admitted"] is False
    assert frame_rows["frame-1"]["reason"] == e3.REASON_HUD_NOT_HIDDEN


def test_missing_actual_pose_is_refused_not_planned(tmp_path):
    planned = {"position": [0.5, 0.5, -3.0], "yaw_degrees": 0.0, "pitch_degrees": 0.0,
               "fov_degrees": 90.0, "eye_height": 1.62}
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png",
                     actual=_valid_actual(eye_position=_MISSING))]
    spec = _scene("alpha", frames=frames, planned_cameras=[planned, planned])
    record, _root, _snapshots = _export(tmp_path, [spec])

    # A valid planned camera exists in capture.json, yet the frame is refused
    # because only the recorded actual pose is admissible.
    assert record["admission"]["refusals"] == {e3.REASON_MISSING_POSE: 1}
    refused = [row for row in record["admission"]["frames"] if not row["admitted"]][0]
    assert refused["reason"] == e3.REASON_MISSING_POSE


def test_missing_actual_metadata_is_refused(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png", include_actual=False)]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {e3.REASON_MISSING_ACTUAL: 1}


def test_non_vanilla_active_pack_is_refused(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png",
                     actual=_valid_actual(active_resource_packs=["vanilla", "faithful.zip"]))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {e3.REASON_NON_VANILLA_PACK: 1}


def test_unapplied_requested_shader_pack_remains_vanilla(tmp_path):
    requested = dict(_valid_actual()["requested"])
    requested["shader_pack"] = "bsl.zip"
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png",
                     actual=_valid_actual(requested=requested))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {}
    assert record["coverage"]["admitted_frames"] == 2


def test_unknown_shader_applied_state_is_refused(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png",
                     actual=_valid_actual(shader_pack_applied=_MISSING))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {e3.REASON_UNKNOWN_PACK: 1}


def test_unknown_pack_metadata_is_refused(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png",
                     actual=_valid_actual(active_resource_packs=_MISSING))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {e3.REASON_UNKNOWN_PACK: 1}


def test_empty_active_pack_list_is_refused(tmp_path):
    frames = [_frame("frame-0", "frame-0.png"),
              _frame("frame-1", "frame-1.png", actual=_valid_actual(active_resource_packs=[]))]
    record, _root, _snapshots = _export(tmp_path, [_scene("alpha", frames=frames)])
    assert record["admission"]["refusals"] == {e3.REASON_UNKNOWN_PACK: 1}


def test_scene_with_only_refused_captures_is_recorded_and_export_refuses_if_empty(tmp_path):
    refused = _frame("frame-0", "frame-0.png", actual=_valid_actual(hud_hidden=False))
    with pytest.raises(e3.FactoryPatchExportError, match="empty export"):
        _export(tmp_path, [_scene("only-refused", frames=[refused])])
    good = _scene("good")
    bad = _scene("bad", frames=[refused])
    record, _root, _snapshots = _export(tmp_path / "second", [good, bad])
    scenes = {row["scene_id"]: row for row in record["admission"]["scenes"]}
    assert scenes["bad"]["admitted"] is False
    assert scenes["good"]["admitted"] is True
    assert record["coverage"]["refused_scenes"] == 1


def test_empty_labels_distinguish_admitted_but_out_of_view_captures(tmp_path):
    out_of_view = _frame(
        "frame-0", "frame-0.png", actual=_valid_actual(yaw_degrees=180.0))
    with pytest.raises(e3.FactoryPatchExportError, match="empty label export.*no exact-model"):
        _export(tmp_path, [_scene("out-of-view", frames=[out_of_view])])


def test_example_cap_is_enforced(tmp_path):
    with pytest.raises(e3.FactoryPatchExportError, match="above the cap"):
        _export(tmp_path, [_scene("alpha")], max_examples=0)


# --------------------------------------------------------------------------- #
# Group isolation
# --------------------------------------------------------------------------- #


def test_same_scene_group_cannot_cross_folds(tmp_path):
    first = _scene("alpha-one", group="shared-topology", split="train")
    second = _scene("alpha-two", group="shared-topology", split="train")
    record, _root, _snapshots = _export(tmp_path, [first, second])

    assert record["grouping"]["grouping_key"] == "scene_group"
    assert record["grouping"]["leakage_issues"] == []
    assert record["grouping"]["fold_assignment"] == {"shared-topology": "train"}
    for example in record["examples"]:
        assert example["scene_group"] == "shared-topology"
        assert example["fold"] == "train"
    manifest = record["grouping"]["manifest"]
    assert manifest["group_count"] == 1
    held = manifest["folds"][0]["held_out_scene_group"]
    assert held == "shared-topology"
    # Both captures of the group are held out together: they never straddle folds.
    assert len(manifest["folds"][0]["held_out_entries"]) == 2


def test_split_leakage_across_one_group_is_refused(tmp_path):
    first = _scene("alpha-one", group="shared-topology", split="train")
    second = _scene("alpha-two", group="shared-topology", split="test")
    with pytest.raises(e3.FactoryPatchExportError, match="refusing grouped export"):
        _export(tmp_path, [first, second])


# --------------------------------------------------------------------------- #
# Bounds and exact-model labelling
# --------------------------------------------------------------------------- #


def test_reference_block_outside_region_is_refused(tmp_path):
    spec = _scene("alpha", blocks={(0, 0, 0): STATE},
                  reference_blocks={(0, 0, 0): STATE, (5, 0, 0): STATE},
                  palette_states=(STATE,))
    with pytest.raises(e3.FactoryPatchExportError, match="outside the region size"):
        _export(tmp_path, [spec], provider=_provider(STATE))


def test_unmodelled_state_is_refused_not_cube_labelled(tmp_path):
    blocks = {(0, 0, 0): STATE, (1, 0, 0): PLAIN}
    spec = _scene("alpha", blocks=blocks, palette_states=(STATE, PLAIN),
                  region_max=(2, 0, 2))
    record, _root, _snapshots = _export(tmp_path, [spec], provider=_provider(STATE))

    labels = {example["label"]["name"] for example in record["examples"]}
    assert labels == {STATE.name}
    frame_rows = [row for row in record["admission"]["frames"] if row["admitted"]]
    assert frame_rows[0]["refusals"] == {e3.MODEL_METADATA_UNAVAILABLE: 1}


def test_geometry_palette_states_are_exact_model_only(tmp_path):
    with pytest.raises(e3.FactoryPatchExportError, match="exact-model palette is empty"):
        _export(tmp_path, [_scene("alpha", blocks={(0, 0, 0): PLAIN},
                                  palette_states=(PLAIN,))],
                provider=_provider(STATE))


# --------------------------------------------------------------------------- #
# Overwrite, output paths and provider/JAR seam
# --------------------------------------------------------------------------- #


def test_refuses_to_overwrite_existing_artifact(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    target = tmp_path / "out.json"
    target.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(e3.FactoryPatchExportError, match="overwrite"):
        e3.export_dataset(root, output=target, provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))
    assert target.read_text(encoding="utf-8") == "pre-existing"


def test_atomic_artifact_create_does_not_overwrite_racing_target(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    original_link = e3.os.link

    def create_target_then_link(source, destination):
        Path(destination).write_text("racing writer", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(e3.os, "link", create_target_then_link)
    record = e3.seal_artifact({"schema": "synthetic", "examples": []})
    with pytest.raises(e3.FactoryPatchExportError, match="overwrite"):
        e3.write_artifact(target, record)
    assert target.read_text(encoding="utf-8") == "racing writer"


def test_refuses_output_inside_dataset_root(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    with pytest.raises(e3.FactoryPatchExportError, match="inside the dataset root"):
        e3.export_dataset(root, output=root / "out.json", provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_refuses_unsafe_output_path(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    unsafe = tmp_path / "sub" / ".." / "out.json"
    with pytest.raises(e3.FactoryPatchExportError, match="unsafe output path"):
        e3.export_dataset(root, output=unsafe, provider=_provider(STATE),
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_requires_jar_or_provider(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    with pytest.raises(e3.FactoryPatchExportError, match="vanilla JAR"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=None, jar=None,
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_missing_jar_file_is_refused(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    with pytest.raises(e3.FactoryPatchExportError, match="JAR is missing"):
        e3.export_dataset(root, output=tmp_path / "out.json", provider=_provider(STATE),
                          jar=tmp_path / "missing.jar",
                          image_loader=_image_loader(),
                          reference_loader=_reference_loader(snapshots))


def test_jar_is_hashed_and_loaded_when_supplied(tmp_path):
    root, snapshots = _write_dataset(tmp_path, [_scene("alpha")])
    jar = tmp_path / "1.21.5.jar"
    jar.write_bytes(b"synthetic-jar")
    provider = _provider(STATE)
    record = e3.export_dataset(root, output=tmp_path / "out.json", provider=provider, jar=jar,
                               image_loader=_image_loader(),
                               reference_loader=_reference_loader(snapshots))
    assert record["inputs"]["minecraft_jar"] == {
        "path": str(jar), "sha256": _sha256_bytes(b"synthetic-jar")}
    assert provider.loaded == [str(jar)]
