from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mc_photogrammetry.dataset.executor import BatchExecutor
from mc_photogrammetry.dataset.finalize import finalize_scene, write_scene_plan
from mc_photogrammetry.dataset.schemas import (
    BlockRecord,
    CapturePlan,
    DatasetSchemaError,
    GroundTruthExport,
    RegionBounds,
)
from mc_photogrammetry.evaluation_context_export import (
    ContextExportError,
    build_bound_context_document,
    context_export_plan,
    validate_context_export_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    target_region = RegionBounds((1, 0, 0), (1, 0, 0))
    context_region = RegionBounds((0, 0, 0), (1, 1, 1))
    target = GroundTruthExport(
        "synthetic-scene", "synthetic-world", target_region,
        (BlockRecord((1, 0, 0), "minecraft:stone", {}),),
    )
    records = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                name = "minecraft:stone" if (x, y, z) == (1, 0, 0) else "minecraft:air"
                records.append({"pos": [x, y, z], "name": name, "properties": {}})
    raw = {
        "schema_version": 1,
        "mod_contract_version": 1,
        "scene_id": "synthetic-scene",
        "world_identity": "synthetic-world",
        "region": context_region.to_dict(),
        "blocks": records,
    }
    capture = {
        "scene_id": "synthetic-scene",
        "world_identity": "synthetic-world",
        "region": target_region.to_dict(),
    }
    images = [{
        "basename": "frame-a.png", "width": 8, "height": 6,
        "sha256": "c" * 64,
    }]
    return target_region, context_region, target, raw, capture, images


def _bind(raw):
    target_region, context_region, target, _, capture, images = _fixture()
    return build_bound_context_document(
        raw,
        capture_sha256="a" * 64,
        target_ground_truth_sha256="b" * 64,
        capture_document=capture,
        ordered_images=images,
        target_ground_truth=target,
        context_region=context_region,
    )


def test_zero_job_plan_is_explicit_deterministic_and_round_trips():
    region = RegionBounds((140, 60, 340), (190, 130, 400))
    first = context_export_plan(
        dataset_id="context-test", scene_id="synthetic-scene",
        world_identity="synthetic-world", region=region, staging_token="d" * 64,
    )
    second = context_export_plan(
        dataset_id="context-test", scene_id="synthetic-scene",
        world_identity="synthetic-world", region=region, staging_token="d" * 64,
    )
    assert first == second
    payload = first.to_dict()
    assert payload["context_export_only"] is True
    assert payload["jobs"] == []
    assert CapturePlan.from_dict(payload) == first
    assert region.size == (51, 71, 61)
    assert region.size[0] * region.size[1] * region.size[2] == 220_881


def test_empty_plan_requires_strict_context_flag_and_capture_jobs_are_forbidden():
    region = RegionBounds((0, 0, 0), (1, 1, 1))
    with pytest.raises(DatasetSchemaError, match="at least one capture job"):
        CapturePlan("p", "s", "d", "w", region, (), "d" * 64)
    plan = context_export_plan(
        dataset_id="d", scene_id="s", world_identity="w", region=region,
        staging_token="d" * 64,
    )
    with pytest.raises(DatasetSchemaError, match="context_export_only must be a boolean"):
        CapturePlan.from_dict({**plan.to_dict(), "context_export_only": "true"})
    with pytest.raises(DatasetSchemaError, match="must not contain capture jobs"):
        CapturePlan.from_dict({**plan.to_dict(), "jobs": [{"bad": "capture"}]})


def test_status_requires_zero_pose_exact_world_region_and_empty_results():
    region = RegionBounds((0, 0, 0), (1, 1, 1))
    plan = context_export_plan(
        dataset_id="d", scene_id="s", world_identity="w", region=region,
        staging_token="e" * 64,
    )
    status = {
        "context_export_only": True,
        "run_failed": False,
        "all_frames_captured": False,
        "ground_truth_exported": True,
        "plan_id": plan.plan_id,
        "staging_token": plan.staging_token,
        "errors": [],
        "context_export": {
            "scene_id": "s", "world_identity": "w", "region": region.to_dict(),
            "camera_pose_applied": False, "result_count": 0,
        },
    }
    validate_context_export_status(status, plan=plan)
    for key, value in (("context_export_only", False), ("ground_truth_exported", False),
                       ("all_frames_captured", True)):
        with pytest.raises(ContextExportError):
            validate_context_export_status({**status, key: value}, plan=plan)
    changed = json.loads(json.dumps(status))
    changed["context_export"]["camera_pose_applied"] = True
    with pytest.raises(ContextExportError, match="identity/region/zero-pose"):
        validate_context_export_status(changed, plan=plan)


def test_context_document_is_complete_sorted_and_deterministic():
    _, context_region, target, raw, capture, images = _fixture()
    kwargs = dict(
        capture_sha256="a" * 64,
        target_ground_truth_sha256="b" * 64,
        capture_document=capture,
        ordered_images=images,
        target_ground_truth=target,
        context_region=context_region,
    )
    first = build_bound_context_document(raw, **kwargs)
    reversed_raw = {**raw, "blocks": list(reversed(raw["blocks"]))}
    second = build_bound_context_document(reversed_raw, **kwargs)
    assert first == second
    assert [tuple(row["pos"]) for row in first["blocks"]] == sorted(
        tuple(row["pos"]) for row in first["blocks"])
    assert first["target_region"] == target.region.to_dict()
    assert first["region"] == context_region.to_dict()
    assert len(first["blocks"]) == 8


def test_incomplete_context_export_is_refused():
    _, _, _, raw, _, _ = _fixture()
    with pytest.raises(ContextExportError, match="records for a 8-block region"):
        _bind({**raw, "blocks": raw["blocks"][:-1]})


def test_context_plan_cannot_enter_normal_dataset_finalizer(tmp_path):
    dataset = tmp_path / "dataset"
    scene = "synthetic-scene"
    plan = context_export_plan(
        dataset_id="d", scene_id=scene, world_identity="w",
        region=RegionBounds((0, 0, 0), (1, 1, 1)), staging_token="f" * 64,
    )
    write_scene_plan(dataset, plan)
    with pytest.raises(DatasetSchemaError, match="cannot be finalized as a capture dataset"):
        finalize_scene(dataset, scene, tmp_path / "staging-game")


def test_context_plan_is_rejected_by_standard_batch_capture_path(tmp_path):
    dataset = tmp_path / "dataset"
    scene = "synthetic-scene"
    plan = context_export_plan(
        dataset_id="d", scene_id=scene, world_identity="w",
        region=RegionBounds((0, 0, 0), (1, 1, 1)), staging_token="f" * 64,
    )
    write_scene_plan(dataset, plan)
    executor = BatchExecutor(tmp_path / "queue", dry_run=True)
    with pytest.raises(DatasetSchemaError, match="isolated evaluation exporter"):
        executor._capture_request({
            "scene_id": scene,
            "staging_game_dir": str(tmp_path / "staging-game"),
        }, dataset)


def test_context_export_is_not_imported_by_reconstruction_or_cli():
    paths = list((ROOT / "src/mc_photogrammetry/reconstruction").rglob("*.py"))
    paths.extend([ROOT / "src/mc_photogrammetry/cli.py"])
    paths.extend((ROOT / "src/mc_photogrammetry/ui").rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "mc_photogrammetry.evaluation_context_export" not in imported, path
