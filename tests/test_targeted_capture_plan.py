from dataclasses import replace

import pytest

from mc_photogrammetry.dataset.plan import build_plan
from mc_photogrammetry.dataset.schemas import DatasetSchemaError, RegionBounds
from mc_photogrammetry.dataset.targeted_capture_plan import (
    PRODUCER, ROLES, SCHEMA, TargetedCapturePlan, TargetedJob, TargetQuality,
    materialize_capture_plan, png_only_handoff)

D = "a" * 64


def _plan():
    region = RegionBounds((0, 0, 0), (7, 7, 7))
    base = build_plan(dataset_id="targeted-eval", scene_id="axis0-scene",
                      world_identity="Canal Town", region=region, seed=4,
                      staging_token=D, view_count=4)
    jobs = tuple(TargetedJob(role, "wall-x0" if index < 3 else None, capture,
                             64, .5, (-24, 0, 24, 0)[index], .15, 3.0)
                 for index, (role, capture) in enumerate(zip(ROLES, base.jobs)))
    return TargetedCapturePlan(SCHEMA, PRODUCER, "targeted-eval", "axis0-scene",
                               "Canal Town", region, jobs, TargetQuality(),
                               D, "b" * 64, "c" * 64, "d" * 64, "e" * 64)


def test_canonical_digest_replay_and_materialization():
    plan = _plan()
    assert TargetedCapturePlan.from_record(plan.to_record()) == plan
    assert TargetedCapturePlan.from_record(plan.to_record()).plan_sha256 == plan.plan_sha256
    capture = materialize_capture_plan(plan)
    assert capture.jobs == tuple(job.capture for job in plan.jobs)
    assert capture.staging_token == plan.source_sha256


@pytest.mark.parametrize("field,value", [
    ("schema", "v0"), ("producer", "other"), ("source_sha256", "0"),
])
def test_version_and_hash_tampering_fail(field, value):
    with pytest.raises(DatasetSchemaError):
        replace(_plan(), **{field: value})


def test_digest_tamper_and_unknown_fields_fail_closed():
    record = _plan().to_record()
    record["plan_sha256"] = "0" * 64
    with pytest.raises(DatasetSchemaError, match="digest"):
        TargetedCapturePlan.from_record(record)
    record = _plan().to_record()
    record["manual_world_coordinates"] = [1, 2, 3]
    with pytest.raises(DatasetSchemaError, match="exactly"):
        TargetedCapturePlan.from_record(record)


@pytest.mark.parametrize("field,value", [
    ("predicted_support_cells", 63),
    ("predicted_support_fraction", .499),
    ("predicted_displacement_cells", 24.01),
    ("predicted_structural_share", .149),
    ("predicted_psr", 2.99),
])
def test_quality_safety_failures(field, value):
    plan = _plan()
    jobs = (replace(plan.jobs[0], **{field: value}),) + plan.jobs[1:]
    with pytest.raises(DatasetSchemaError):
        replace(plan, jobs=jobs, plan_sha256="")


def test_roles_and_surface_binding_are_exact():
    plan = _plan()
    with pytest.raises(DatasetSchemaError):
        replace(plan, jobs=(replace(plan.jobs[0], role="context"),) + plan.jobs[1:],
                plan_sha256="")
    with pytest.raises(DatasetSchemaError):
        replace(plan, jobs=(replace(plan.jobs[0], surface_id="other"),) + plan.jobs[1:],
                plan_sha256="")


def test_png_only_handoff_strips_all_evaluation_metadata():
    plan = _plan()
    paths = {job.capture.frame_id: f"images/{job.capture.frame_id}.png"
             for job in plan.jobs}
    handoff = png_only_handoff(plan, paths)
    assert handoff == tuple(paths[job.capture.frame_id] for job in plan.jobs)
    assert all(isinstance(value, str) and value.endswith(".png") for value in handoff)
    with pytest.raises(DatasetSchemaError):
        png_only_handoff(plan, {**paths, "extra": "extra.png"})
    paths[plan.jobs[0].capture.frame_id] = "capture.json"
    with pytest.raises(DatasetSchemaError, match="PNG"):
        png_only_handoff(plan, paths)


def test_module_is_isolated_from_batch_executor_and_production_pipeline():
    source = __import__(
        "mc_photogrammetry.dataset.targeted_capture_plan", fromlist=["x"]).__file__
    text = open(source, encoding="utf-8").read()
    assert "dataset.batch" not in text
    assert "dataset.executor" not in text
    assert "mc_photogrammetry.pipeline" not in text
    assert "subprocess" not in text
