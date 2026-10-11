"""Focused tests for the launch-free bulk capture queue (DS-27).

Every test here runs without Minecraft and without executing either factory
script: the queue only plans, fingerprints source saves read-only, and writes its
own manifest/region bookkeeping.  The categories the work package requires are
covered explicitly: deterministic idempotence, corrupted-state refusal,
path-containment, duplicate/leakage, status transitions, Unicode, and
command-argument generation.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mc_photogrammetry.dataset import (
    DatasetSchemaError,
    RegionBounds,
    load_batch_spec,
    load_queue,
    load_region_requests,
    manifest_path,
    materialize_queue,
    plan_batch,
    queue_commands,
    resume_queue,
    set_status,
    write_queue_lines,
)
from mc_photogrammetry.dataset.hashing import sha256_file
from mc_photogrammetry.dataset.hashing import digest_json
from mc_photogrammetry.dataset.batch import _immutable_payload

ROOT = Path(__file__).resolve().parents[1]

CJK_LEVEL_NAME = "\u9ad8\u538b\u9ad8\u706b\u529b\u4e0b\u5f39\u4e09\u5411\u8f70\u70b8\u673a"
CJK_SAVE_NAME = "\u9ad8\u538b\u5b58\u6863"


# --- fixtures ---------------------------------------------------------------


def _level_dat(path: Path, *, level_name: str, spawn=(3, 71, -4),
               with_spawn_y: bool = True) -> None:
    import nbtlib
    from nbtlib import tag

    data = {
        "LevelName": tag.String(level_name),
        "LastPlayed": tag.Long(1_700_000_000_000),
        "SpawnX": tag.Int(spawn[0]),
        "SpawnZ": tag.Int(spawn[2]),
    }
    if with_spawn_y:
        data["SpawnY"] = tag.Int(spawn[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(tag.Compound({"Data": tag.Compound(data)}), gzipped=True).save(path)


def _save(game: Path, name: str, *, level_name: str | None = None,
          payload: bytes = b"region", with_spawn_y: bool = True) -> Path:
    world = game / "saves" / name
    (world / "region").mkdir(parents=True, exist_ok=True)
    (world / "region" / "r.0.0.mca").write_bytes(payload)
    if level_name is None:
        (world / "level.dat").write_bytes(b"not-an-nbt")
    else:
        _level_dat(world / "level.dat", level_name=level_name, with_spawn_y=with_spawn_y)
    return world


def _game(tmp_path: Path) -> Path:
    game = tmp_path / "game"
    _save(game, "Alpha", level_name="Alpha World", payload=b"alpha-region")
    _save(game, "Beta", level_name="Beta World", payload=b"beta-region")
    return game


def _region(name: str = "house", lo=(0, 60, 0), hi=(15, 75, 15)) -> dict:
    return {"name": name, "min": list(lo), "max": list(hi)}


def _task(save: str = "Alpha", split: str = "train", region: dict | None = None,
          **over) -> dict:
    task = {"task_id": f"{save.lower()}-{over.get('name', 'task')}",
            "save": save, "declared_split": split, "region": region or _region()}
    task.update(over)
    return task


def _spec(game: Path, dataset: Path, tasks: list, **over) -> dict:
    payload = {
        "schema_version": 1,
        "batch_id": "ds27-test",
        "game_dir": str(game),
        "dataset": str(dataset),
        "seed": 7,
        "views": 6,
        "tasks": tasks,
    }
    payload.update(over)
    return payload


def _write_spec(tmp_path: Path, payload: dict, name: str = "batch.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _materialized(tmp_path: Path, tasks: list, **over):
    game = over.pop("game", None) or _game(tmp_path)
    dataset = tmp_path / "ds"
    plan = plan_batch(_spec(game, dataset, tasks, **over))
    path = materialize_queue(plan)
    return game, dataset, plan, path


def _arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _tree(root: Path) -> list[str]:
    return sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))


def _strict_text_stream(encoding: str):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, errors="strict", newline="")
    return stream, buffer


def _batch_cli_module():
    path = ROOT / "scripts" / "plan_capture_batch.py"
    spec = importlib.util.spec_from_file_location("ds27_plan_capture_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- deterministic idempotence ----------------------------------------------


def test_plan_is_deterministic_and_materialization_is_byte_idempotent(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        _task("Alpha", name="a", region=_region("house")),
        _task("Beta", split="dev", name="b", region=_region("yard")),
    ])
    first = plan_batch(spec)
    second = plan_batch(spec)
    assert first.manifest_id == second.manifest_id
    assert [scene.scene_id for scene in first.scenes] == ["alpha-house", "beta-yard"]

    path = materialize_queue(first)
    written = path.read_bytes()
    materialize_queue(second)
    assert path.read_bytes() == written
    assert load_queue(dataset / "queue")["manifest_id"] == first.manifest_id


def test_source_identity_tracks_save_content_and_changes_the_manifest(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [_task("Alpha")])
    first = plan_batch(spec)
    (game / "saves" / "Alpha" / "region" / "r.0.0.mca").write_bytes(b"changed")
    second = plan_batch(spec)
    assert first.scenes[0].source_identity != second.scenes[0].source_identity
    assert first.scenes[0].build_id != second.scenes[0].build_id
    assert first.manifest_id != second.manifest_id


def test_existing_manifest_with_a_different_id_is_refused(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    materialize_queue(plan_batch(_spec(game, dataset, [_task("Alpha")])))
    other = plan_batch(_spec(game, dataset, [
        _task("Beta", split="dev", region=_region("yard"))]))
    with pytest.raises(DatasetSchemaError, match="immutable queue"):
        materialize_queue(other)


# --- corrupted-state refusal -------------------------------------------------


def test_corrupted_or_tampered_manifest_is_refused(tmp_path):
    _, dataset, _, path = _materialized(tmp_path, [_task("Alpha")])
    queue = dataset / "queue"
    assert load_queue(queue)["scenes"][0]["status"] == "pending"
    good = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(DatasetSchemaError, match="no queue manifest"):
        load_queue(dataset / "absent-queue")

    tampered = copy.deepcopy(good)
    tampered["scenes"][0]["region"]["min"][0] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="does not hash to its manifest_id"):
        load_queue(queue)

    wrong_schema = copy.deepcopy(good)
    wrong_schema["schema_version"] = 2
    path.write_text(json.dumps(wrong_schema), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="unsupported schema_version"):
        load_queue(queue)

    unknown_status = copy.deepcopy(good)
    unknown_status["scenes"][0]["status"] = "queued"
    path.write_text(json.dumps(unknown_status), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="unknown status"):
        load_queue(queue)

    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="corrupted queue manifest"):
        load_queue(queue)


def test_a_tampered_region_file_is_refused_on_rematerialize(tmp_path):
    game, dataset, plan, _ = _materialized(tmp_path, [_task("Alpha")])
    region_file = dataset / "queue" / "regions" / "alpha-house.json"
    region_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="immutable region file"):
        materialize_queue(plan)


# --- duplicate and leakage refusal ------------------------------------------


def test_duplicate_scene_ids_and_identical_work_are_refused(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    duplicate_id = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
        {"task_id": "b", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
    ])
    with pytest.raises(DatasetSchemaError, match="duplicate scene_id"):
        plan_batch(duplicate_id)

    duplicate_work = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
        {"task_id": "b", "save": "Alpha", "declared_split": "train",
         "region": _region("house-copy")},
    ])
    with pytest.raises(DatasetSchemaError, match="duplicates the same source/region/variant"):
        plan_batch(duplicate_work)


def test_one_source_may_not_straddle_two_declared_splits(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
        {"task_id": "b", "save": "Alpha", "declared_split": "test",
         "region": _region("yard", (32, 60, 0), (47, 75, 15))},
    ])
    with pytest.raises(DatasetSchemaError, match="must not straddle"):
        plan_batch(spec)


def test_multi_variant_explicit_region_gets_distinct_scenes_but_one_build(tmp_path):
    """Two variants of one explicit-region build must not collide on scene id.

    Each variant gets its own region name/file, staging directory and commands,
    while the bounds and the source/build identity stay identical so the two
    variants still group as one build for split-leakage refusal.
    """
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = {
        "task_id": "alpha-multi", "save": "Alpha", "declared_split": "train",
        "region": _region("house", (0, 60, 0), (15, 75, 15)),
        "variants": [
            {"variant_id": "day", "seed": 1, "views": 5},
            {"variant_id": "night", "seed": 2, "views": 6},
        ],
    }
    plan = plan_batch(_spec(game, dataset, [task]))
    assert [scene.scene_id for scene in plan.scenes] == [
        "alpha-house-day", "alpha-house-night"]
    day, night = plan.scenes
    assert day.region_name == "house-day" and night.region_name == "house-night"
    assert day.region == night.region           # identical bounds
    assert day.build_id == night.build_id       # one common build identity
    assert day.source_identity == night.source_identity
    assert day.declared_split == night.declared_split == "train"

    materialize_queue(plan)
    payload = load_queue(plan.queue_dir)
    rows = {row["scene_id"]: row for row in payload["scenes"]}
    assert set(rows) == {"alpha-house-day", "alpha-house-night"}
    day_row, night_row = rows["alpha-house-day"], rows["alpha-house-night"]
    assert day_row["region_file"] != night_row["region_file"]
    assert day_row["staging_game_dir"] != night_row["staging_game_dir"]
    assert day_row["prepare_argv"] != night_row["prepare_argv"]
    assert day_row["finalize_argv"] != night_row["finalize_argv"]
    assert _arg(day_row["prepare_argv"], "--seed") == "1"
    assert _arg(night_row["prepare_argv"], "--seed") == "2"
    assert _arg(day_row["prepare_argv"], "--views") == "5"
    assert _arg(night_row["prepare_argv"], "--views") == "6"
    assert _arg(day_row["finalize_argv"], "--scene") == "alpha-house-day"
    assert _arg(night_row["finalize_argv"], "--scene") == "alpha-house-night"
    for row, name in ((day_row, "house-day"), (night_row, "house-night")):
        request = load_region_requests(Path(row["region_file"]))[0]
        assert request.name == name
        assert request.bounds == day.region     # same bounds, per-variant file
    assert [row["scene_id"] for row in queue_commands(payload)] == [
        "alpha-house-day", "alpha-house-night"]

    # The variants share one build id, so a second task that puts the same save
    # under another declared split is refused as leakage across splits.
    leak = _spec(game, dataset, [task, {
        "task_id": "alpha-test", "save": "Alpha", "declared_split": "test",
        "region": _region("yard", (32, 60, 0), (47, 75, 15))}])
    with pytest.raises(DatasetSchemaError, match="must not straddle"):
        plan_batch(leak)


def test_existing_dataset_index_leakage_and_repeats_are_refused(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = {"task_id": "a", "save": "Alpha", "declared_split": "train",
            "region": _region("house")}
    token = plan_batch(_spec(game, dataset, [task])).scenes[0].source_identity

    dataset.mkdir(parents=True, exist_ok=True)
    index = dataset / "index.json"
    index.write_text(json.dumps({
        "schema_version": 1,
        "scenes": [{"scene_id": "alpha-other", "split": "dev",
                    "provenance": {"staging_token": token}}],
    }), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="already finalized under split"):
        plan_batch(_spec(game, dataset, [task]))

    index.write_text(json.dumps({
        "schema_version": 1,
        "scenes": [{"scene_id": "alpha-house", "split": "train",
                    "provenance": {"staging_token": token}}],
    }), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="already exists"):
        plan_batch(_spec(game, dataset, [task]))

    index.write_text("{ not json", encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="unreadable"):
        plan_batch(_spec(game, dataset, [task]))


# --- path containment --------------------------------------------------------


@pytest.mark.parametrize("case", ["dataset_in_game", "staging_in_save",
                                  "dataset_is_save", "staging_is_game"])
def test_output_roots_may_not_overlap_the_game_dir_or_a_source_save(tmp_path, case):
    game = _game(tmp_path)
    spec = _spec(game, tmp_path / "ds", [_task("Alpha")])
    if case == "dataset_in_game":
        spec["dataset"] = str(game / "ds")
    elif case == "staging_in_save":
        spec["staging_game_dir"] = str(game / "saves" / "Alpha" / "staging")
    elif case == "dataset_is_save":
        spec["dataset"] = str(game / "saves" / "Alpha")
    elif case == "staging_is_game":
        spec["staging_game_dir"] = str(game)
    with pytest.raises(DatasetSchemaError, match="overlaps"):
        plan_batch(spec)


def test_staging_may_not_sit_inside_the_dataset_scenes_directory(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [_task("Alpha")])
    spec["staging_game_dir"] = str(dataset / "scenes" / "staging")
    with pytest.raises(DatasetSchemaError, match="must not sit inside"):
        plan_batch(spec)


# --- command arguments and region files --------------------------------------


def test_generated_commands_and_region_files_match_the_factory_contract(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    plan = plan_batch(_spec(game, dataset, [{
        "task_id": "a", "save": "Alpha", "declared_split": "train",
        "region": _region("house", (0, 60, 0), (15, 75, 15)),
        "views": 5, "seed": 11}]))
    scene = plan.scenes[0]
    materialize_queue(plan)
    payload = load_queue(plan.queue_dir)
    row = payload["scenes"][0]
    staging = str(plan.staging_game_dir / scene.scene_id)

    prepare = row["prepare_argv"]
    assert prepare[:2] == ["python", "scripts/prepare_capture_dataset.py"]
    assert _arg(prepare, "--game-dir") == str(plan.game_dir)
    assert _arg(prepare, "--save") == "Alpha"
    assert _arg(prepare, "--dataset") == str(plan.dataset)
    assert _arg(prepare, "--dataset-id") == "ds27-test"
    assert _arg(prepare, "--staging-game-dir") == staging
    assert _arg(prepare, "--views") == "5"
    assert _arg(prepare, "--seed") == "11"
    assert "--stratified-settings" in prepare
    assert row["settings_mode"] == "stratified"
    assert row["settings_requested"] is None
    assert row["settings_effective_request"] == {
        "strategy": "view_setting_strata", "view_count": 5}
    assert "--allow-overwrite" in prepare
    assert "--landscape" not in prepare
    assert "--camera" not in prepare and "--coords" not in prepare

    finalize = row["finalize_argv"]
    assert finalize[:2] == ["python", "scripts/finalize_capture_dataset.py"]
    assert _arg(finalize, "--scene") == scene.scene_id
    assert _arg(finalize, "--staging-game-dir") == staging
    assert _arg(finalize, "--dataset") == str(plan.dataset)

    regions = load_region_requests(Path(row["region_file"]))
    assert [request.name for request in regions] == ["house"]
    assert regions[0].bounds == scene.region
    assert regions[0].views == 5


def test_queue_commands_select_actionable_scenes_only(tmp_path):
    _, dataset, _, _ = _materialized(tmp_path, [
        _task("Alpha", name="a", region=_region("house")),
        _task("Beta", split="dev", name="b", region=_region("yard")),
    ])
    queue = dataset / "queue"
    set_status(queue, "alpha-house", "running")
    set_status(queue, "alpha-house", "complete")
    rows = queue_commands(load_queue(queue))
    assert [row["scene_id"] for row in rows] == ["beta-yard"]
    assert rows[0]["status"] == "pending"


def test_save_lookup_is_case_insensitive_and_unknown_saves_are_refused(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [{"task_id": "a", "save": "alpha",
                                  "declared_split": "train", "region": _region()}])
    assert plan_batch(spec).scenes[0].save_name == "Alpha"
    with pytest.raises(DatasetSchemaError, match="no save named"):
        plan_batch(_spec(game, dataset, [{"task_id": "a", "save": "Gamma",
                                          "declared_split": "train",
                                          "region": _region()}]))

    spec["game_dir"] = str(tmp_path / "absent")
    with pytest.raises(DatasetSchemaError, match="no such game directory"):
        plan_batch(spec)


# --- advanced-only input guard ----------------------------------------------


@pytest.mark.parametrize("field", ["camera", "f3", "coords", "position", "target", "yaw"])
def test_camera_f3_and_coordinate_fields_are_refused(tmp_path, field):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = _task("Alpha")
    task[field] = {"x": 1}
    spec_path = _write_spec(tmp_path, _spec(game, dataset, [task]))
    with pytest.raises(DatasetSchemaError, match="refused"):
        load_batch_spec(spec_path)
    with pytest.raises(DatasetSchemaError, match="refused"):
        plan_batch(_spec(game, dataset, [task]))


def test_packs_are_requested_provenance_only(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = _task("Alpha", resource_pack="faithful.zip", shader_pack="bsl.zip")
    plan = plan_batch(_spec(game, dataset, [task]))
    materialize_queue(plan)
    payload = load_queue(plan.queue_dir)
    scene = payload["scenes"][0]
    assert scene["resource_pack_requested"] == "faithful.zip"
    assert scene["shader_pack_requested"] == "bsl.zip"
    assert scene["resource_pack_applied"] is False
    assert scene["shader_pack_applied"] is False
    assert "no runtime pack switch" in scene["pack_note"]
    assert scene["settings_requested"]["resource_pack"] == "faithful.zip"
    assert payload["advanced_only"] is True
    assert "never a normal reconstruction input" in payload["usage"]


def test_fixed_settings_are_argv_safe_merged_and_manifest_consistent(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    pack = 'pack "quoted" $literal; & name.zip'
    task = _task(
        "Alpha", settings={"gamma": 0.45, "time_of_day": 12000,
                           "weather": "rain", "hud_hidden": True,
                           "resource_pack": "nested-task.zip"},
        resource_pack="top-task.zip",
        variants=[{
            "variant_id": "variant", "settings": {"gamma": 0.7},
            "resource_pack": pack,
        }],
    )
    plan = plan_batch(_spec(game, dataset, [task]))
    scene = plan.scenes[0]
    assert scene.settings_mode == "fixed"
    assert scene.settings.gamma == 0.7
    assert scene.settings.time_of_day == 12000
    assert scene.settings.weather == "rain"
    assert scene.settings.hud_hidden is True
    assert scene.settings.resource_pack == pack

    materialize_queue(plan)
    row = load_queue(plan.queue_dir)["scenes"][0]
    argv = row["prepare_argv"]
    encoded = _arg(argv, "--fixed-settings-json")
    assert len([arg for arg in argv if arg == "--fixed-settings-json"]) == 1
    assert "--stratified-settings" not in argv
    assert json.loads(encoded) == row["settings_requested"]
    assert row["settings_requested"] == row["settings_effective_request"]
    assert row["settings_requested"]["resource_pack"] == pack


def test_empty_fixed_settings_and_explicit_null_pack_have_distinct_modes(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    fixed = plan_batch(_spec(game, dataset, [_task("Alpha", settings={})]))
    assert fixed.scenes[0].settings_mode == "fixed"
    assert fixed.scenes[0].settings.to_dict() == {
        "gamma": 0.0, "time_of_day": 6000, "weather": "clear",
        "hud_hidden": False, "resource_pack": None, "shader_pack": None,
    }

    stratified = plan_batch(_spec(game, dataset, [_task("Alpha")]))
    assert stratified.scenes[0].settings_mode == "stratified"
    assert plan_batch(_spec(game, tmp_path / "null-pack", [
        _task("Alpha", resource_pack="inherited.zip", variants=[
            {"resource_pack": None}])
    ])).scenes[0].settings.resource_pack is None
    explicit_null_pack = plan_batch(_spec(game, tmp_path / "explicit-null", [
        _task("Alpha", resource_pack=None)
    ]))
    assert explicit_null_pack.scenes[0].settings_mode == "fixed"


def test_variant_settings_merge_by_key_and_null_clears_nested_inheritance(tmp_path):
    game = _game(tmp_path)
    task = _task(
        "Alpha", settings={"gamma": 0.25, "weather": "rain",
                           "resource_pack": "nested-task.zip"},
        resource_pack="top-task.zip",
        variants=[
            {"variant_id": "merge", "settings": {
                "gamma": 0.8, "resource_pack": "nested-variant.zip"}},
            {"variant_id": "clear", "settings": None},
        ],
    )
    plan = plan_batch(_spec(game, tmp_path / "ds", [task]))
    merge, clear = plan.scenes
    assert merge.settings.gamma == 0.8
    assert merge.settings.weather == "rain"
    # A top-level task pack overrides both nested task and variant pack values.
    assert merge.settings.resource_pack == "top-task.zip"
    # Explicit settings:null clears the nested gamma/weather, while the separate
    # task-level pack field still applies according to the documented precedence.
    assert clear.settings.gamma == 0.0
    assert clear.settings.weather == "clear"
    assert clear.settings.resource_pack == "top-task.zip"


@pytest.mark.parametrize("settings", [
    {"hud_hidden": "false"},
    {"gamma": True},
    {"time_of_day": 6000.5},
    {"weather": 3},
    {"unexpected": 1},
    {"resource_pack": 123},
    {"shader_pack": "bad\nname"},
])
def test_batch_settings_reject_permissive_or_unsafe_json_types(tmp_path, settings):
    game = _game(tmp_path)
    with pytest.raises(DatasetSchemaError):
        plan_batch(_spec(game, tmp_path / "ds", [_task("Alpha", settings=settings)]))


def test_manifest_loader_rejects_plan_settings_that_disagree_with_argv(tmp_path):
    _, _, plan, _ = _materialized(tmp_path, [_task("Alpha", settings={"gamma": 0.6})])
    payload = load_queue(plan.queue_dir)
    payload["scenes"][0]["settings_effective_request"]["gamma"] = 0.2
    payload["manifest_id"] = digest_json(_immutable_payload(payload))
    manifest_path(plan.queue_dir).write_text(
        json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="settings manifest and prepare argv disagree"):
        load_queue(plan.queue_dir)


def test_legacy_v1_manifest_without_settings_mode_still_loads(tmp_path):
    _, _, plan, _ = _materialized(tmp_path, [_task("Alpha")])
    payload = load_queue(plan.queue_dir)
    row = payload["scenes"][0]
    for key in ("settings_mode", "settings_requested", "settings_effective_request"):
        row.pop(key)
    row["prepare_argv"] = [
        arg for arg in row["prepare_argv"]
        if arg != "--stratified-settings"
    ]
    payload["manifest_id"] = digest_json(_immutable_payload(payload))
    manifest_path(plan.queue_dir).write_text(json.dumps(payload), encoding="utf-8")
    assert load_queue(plan.queue_dir)["scenes"][0]["scene_id"] == "alpha-house"


# --- statuses, attempts, resume ---------------------------------------------


def test_status_transitions_and_attempt_evidence_are_enforced(tmp_path):
    _, dataset, _, _ = _materialized(tmp_path, [
        _task("Alpha", name="a", region=_region("house")),
        _task("Beta", split="dev", name="b", region=_region("yard")),
    ])
    queue = dataset / "queue"

    payload = set_status(queue, "alpha-house", "running")
    assert payload["scenes"][0]["status"] == "running"
    payload = set_status(queue, "alpha-house", "complete", evidence={"stage": "finalize"})
    assert payload["scenes"][0]["attempts"] == [
        {"attempt": 1, "status": "running", "evidence": {}},
        {"attempt": 2, "status": "complete", "evidence": {"stage": "finalize"}},
    ]

    with pytest.raises(DatasetSchemaError, match="illegal status transition"):
        set_status(queue, "alpha-house", "running")
    with pytest.raises(DatasetSchemaError, match="already"):
        set_status(queue, "alpha-house", "complete")
    with pytest.raises(DatasetSchemaError, match="illegal status transition"):
        set_status(queue, "beta-yard", "complete")
    with pytest.raises(DatasetSchemaError, match="unknown status"):
        set_status(queue, "beta-yard", "queued")
    with pytest.raises(DatasetSchemaError, match="no scene"):
        set_status(queue, "missing-scene", "running")
    with pytest.raises(DatasetSchemaError, match="JSON-serializable"):
        set_status(queue, "beta-yard", "running", evidence={"bad": object()})
    # A rejected transition must not have written a partial update.
    assert load_queue(queue)["scenes"][1]["status"] == "pending"


def test_load_queue_fully_validates_mutable_attempt_history(tmp_path):
    """Status/attempts are stripped from manifest_id, so load_queue is their only gate.

    Every corruption below leaves ``manifest_id`` valid (the mutable fields are
    excluded from the digest) and must still be refused by history validation.
    """
    _, dataset, _, path = _materialized(tmp_path, [
        _task("Alpha", name="a", region=_region("house"))])
    queue = dataset / "queue"
    good = json.loads(path.read_text(encoding="utf-8"))

    def rewrite(mutate):
        payload = copy.deepcopy(good)
        mutate(payload["scenes"][0])
        path.write_text(json.dumps(payload), encoding="utf-8")

    # Ordinals must be exactly 1..N: a gap is refused.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": 1, "status": "running", "evidence": {}},
                     {"attempt": 3, "status": "failed", "evidence": {}}],
        "status": "failed"}))
    with pytest.raises(DatasetSchemaError, match="ordinals must be exactly 1..N"):
        load_queue(queue)

    # ... and so is a history that does not start at 1.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": 2, "status": "running", "evidence": {}}],
        "status": "running"}))
    with pytest.raises(DatasetSchemaError, match="ordinals must be exactly 1..N"):
        load_queue(queue)

    # A boolean is not an integer ordinal.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": True, "status": "running", "evidence": {}}],
        "status": "running"}))
    with pytest.raises(DatasetSchemaError, match="positive integer attempt ordinal"):
        load_queue(queue)

    # The history must start from pending: pending -> complete is illegal.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": 1, "status": "complete", "evidence": {}}],
        "status": "complete"}))
    with pytest.raises(DatasetSchemaError, match="illegal transition"):
        load_queue(queue)

    # A same-status repeat is not a legal transition either.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": 1, "status": "running", "evidence": {}},
                     {"attempt": 2, "status": "running", "evidence": {}}],
        "status": "running"}))
    with pytest.raises(DatasetSchemaError, match="illegal transition"):
        load_queue(queue)

    # The recorded scene status must equal the last attempt status.
    rewrite(lambda scene: scene.update({
        "attempts": [{"attempt": 1, "status": "running", "evidence": {}}],
        "status": "complete"}))
    with pytest.raises(DatasetSchemaError, match="disagrees"):
        load_queue(queue)

    # A non-pending scene must carry at least one attempt.
    rewrite(lambda scene: scene.update({"attempts": [], "status": "running"}))
    with pytest.raises(DatasetSchemaError, match="no attempts"):
        load_queue(queue)

    # A pristine pending scene with zero attempts is the one legal empty history.
    rewrite(lambda scene: scene.update({"attempts": [], "status": "pending"}))
    assert load_queue(queue)["scenes"][0]["status"] == "pending"

    # A legal ladder (including a retry after failure) still loads.
    rewrite(lambda scene: scene.update({
        "attempts": [
            {"attempt": 1, "status": "running", "evidence": {}},
            {"attempt": 2, "status": "failed", "evidence": {"reason": "interrupted"}},
            {"attempt": 3, "status": "running", "evidence": {}},
        ],
        "status": "running"}))
    assert load_queue(queue)["scenes"][0]["status"] == "running"


def test_resume_resets_stale_running_and_is_idempotent(tmp_path):
    _, dataset, _, _ = _materialized(tmp_path, [
        _task("Alpha", name="a", region=_region("house")),
        _task("Beta", split="dev", name="b", region=_region("yard")),
    ])
    queue = dataset / "queue"
    set_status(queue, "alpha-house", "running")
    set_status(queue, "beta-yard", "running")
    set_status(queue, "beta-yard", "complete")

    first = resume_queue(queue)
    assert first["reset"] == ["alpha-house"]
    assert first["actionable"] == ["alpha-house"]
    assert first["complete"] == ["beta-yard"]
    alpha = load_queue(queue)["scenes"][0]
    assert alpha["status"] == "failed"
    assert alpha["attempts"][-1]["evidence"]["reason"].startswith("stale running")

    second = resume_queue(queue)
    assert second["reset"] == []
    assert second["actionable"] == ["alpha-house"]
    assert len(load_queue(queue)["scenes"][0]["attempts"]) == 2


# --- dry run and source-save safety -----------------------------------------


def test_dry_run_writes_nothing_and_never_touches_a_source_save(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    save_file = game / "saves" / "Alpha" / "region" / "r.0.0.mca"
    before_tree = _tree(game)
    before_digest = sha256_file(save_file)
    before_mtime = save_file.stat().st_mtime_ns

    plan = plan_batch(_spec(game, dataset, [_task("Alpha")]))
    assert not dataset.exists()
    assert _tree(game) == before_tree

    materialize_queue(plan)
    assert sha256_file(save_file) == before_digest
    assert save_file.stat().st_mtime_ns == before_mtime
    assert _tree(game) == before_tree
    assert not (dataset / "scenes").exists()


# --- Unicode -----------------------------------------------------------------


def test_unicode_world_identity_round_trips_and_renders_under_a_strict_codec(tmp_path):
    game = tmp_path / "game"
    _save(game, "Alpha", level_name=CJK_LEVEL_NAME, payload=b"alpha")
    dataset = tmp_path / "ds"
    plan = plan_batch(_spec(game, dataset, [_task("Alpha")]))
    materialize_queue(plan)
    payload = load_queue(plan.queue_dir)
    assert payload["scenes"][0]["world_identity"] == CJK_LEVEL_NAME

    raw = (dataset / "queue" / "manifest.json").read_text(encoding="utf-8")
    # The canonical JSON writer is ASCII-escaped, so the manifest bytes stay
    # byte-stable on any console while the identity still round-trips exactly.
    assert "\\u9ad8\\u538b" in raw
    assert json.loads(raw)["scenes"][0]["world_identity"] == CJK_LEVEL_NAME

    stream, buffer = _strict_text_stream("cp1252")
    write_queue_lines(payload, stream=stream)
    stream.flush()
    rendered = buffer.getvalue().decode("cp1252")
    assert "alpha-house" in rendered
    assert "\\u9ad8\\u538b" in rendered  # escaped, not dropped or crashed
    stream.detach()


def test_non_ascii_save_directory_is_refused_because_frame_ids_must_be_ascii(tmp_path):
    game = tmp_path / "game"
    _save(game, CJK_SAVE_NAME, level_name="High Voltage", payload=b"alpha")
    spec = _spec(game, tmp_path / "ds", [{"task_id": "a", "save": CJK_SAVE_NAME,
                                          "declared_split": "train",
                                          "region": _region()}])
    with pytest.raises(DatasetSchemaError, match="frame-id contract"):
        plan_batch(spec)


# --- landscape convenience ---------------------------------------------------


def test_landscape_region_is_derived_from_spawn_without_manual_geometry(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = {"task_id": "a", "save": "Alpha", "declared_split": "train",
            "landscape": {"radius": 4, "vertical_below": 2, "vertical_above": 6}}
    scene = plan_batch(_spec(game, dataset, [task])).scenes[0]
    assert scene.region_name == "spawn-3--4"
    assert scene.region == RegionBounds((3 - 4, 71 - 2, -4 - 4), (3 + 4, 71 + 6, -4 + 4))
    assert scene.scene_id == "alpha-spawn-3-4"


def test_multi_variant_landscape_regions_are_also_distinct(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    task = {"task_id": "alpha-land", "save": "Alpha", "declared_split": "train",
            "landscape": {"radius": 4, "vertical_below": 2, "vertical_above": 6},
            "variants": [{"variant_id": "a", "seed": 1},
                         {"variant_id": "b", "seed": 2}]}
    first, second = plan_batch(_spec(game, dataset, [task])).scenes
    assert first.scene_id == "alpha-spawn-3-4-a"
    assert second.scene_id == "alpha-spawn-3-4-b"
    assert first.region == second.region
    assert first.build_id == second.build_id


def test_landscape_refuses_a_save_without_a_spawn_height(tmp_path):
    game = tmp_path / "game"
    _save(game, "NoY", level_name="NoY", payload=b"alpha", with_spawn_y=False)
    task = {"task_id": "a", "save": "NoY", "declared_split": "train",
            "landscape": {"radius": 4}}
    with pytest.raises(DatasetSchemaError, match="SpawnY"):
        plan_batch(_spec(game, tmp_path / "ds", [task]))


def test_region_caps_are_enforced_by_the_region_schema(tmp_path):
    game = _game(tmp_path)
    task = _task("Alpha", region=_region("huge", (0, 0, 0), (600, 10, 10)))
    with pytest.raises(DatasetSchemaError, match="region edge"):
        plan_batch(_spec(game, tmp_path / "ds", [task]))


# --- CLI ---------------------------------------------------------------------


def test_cli_dry_run_materialize_status_mark_and_resume(tmp_path, capsys):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec_path = _write_spec(tmp_path, _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
        {"task_id": "b", "save": "Beta", "declared_split": "dev",
         "region": _region("yard")},
    ]))
    cli = _batch_cli_module()

    assert cli.main(["--spec", str(spec_path)]) == 0
    assert "dry run" in capsys.readouterr().out
    assert not dataset.exists()

    assert cli.main(["--spec", str(spec_path), "--materialize"]) == 0
    capsys.readouterr()
    assert manifest_path(dataset / "queue").is_file()
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "pending"

    assert cli.main(["--spec", str(spec_path), "--status"]) == 0
    assert "alpha-house" in capsys.readouterr().out

    assert cli.main(["--spec", str(spec_path), "--mark", "alpha-house", "running"]) == 0
    capsys.readouterr()
    assert cli.main(["--spec", str(spec_path), "--resume"]) == 0
    assert "alpha-house" in capsys.readouterr().out
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "failed"

    assert cli.main(["--spec", str(spec_path), "--commands"]) == 0
    out = capsys.readouterr().out
    assert "prepare" in out and "finalize" in out and "beta-yard" in out

    with pytest.raises(SystemExit, match="JSON object"):
        cli.main(["--spec", str(spec_path), "--mark", "beta-yard", "running",
                  "--evidence", "[]"])
    with pytest.raises(DatasetSchemaError, match="illegal status transition"):
        cli.main(["--spec", str(spec_path), "--mark", "beta-yard", "complete"])


def test_cli_materializes_a_unicode_identity_on_a_strict_cp1252_console(tmp_path):
    game = tmp_path / "game"
    _save(game, "Alpha", level_name=CJK_LEVEL_NAME, payload=b"alpha")
    dataset = tmp_path / "ds"
    spec_path = _write_spec(tmp_path, _spec(game, dataset, [_task("Alpha")]))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plan_capture_batch.py"),
         "--spec", str(spec_path), "--materialize"],
        cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="cp1252"),
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert "alpha-house" in completed.stdout
    assert load_queue(dataset / "queue")["scenes"][0]["world_identity"] == CJK_LEVEL_NAME
