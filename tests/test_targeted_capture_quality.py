from types import SimpleNamespace

from PIL import Image
import pytest

from mc_photogrammetry.dataset.schemas import DatasetSchemaError
from mc_photogrammetry.dataset import targeted_capture_quality as subject
from mc_photogrammetry.reconstruction.plane_graph import PlanePairTrace
from test_targeted_capture_plan import _plan


def _pngs(tmp_path, plan):
    paths = []
    for job in plan.jobs:
        path = tmp_path / f"{job.capture.frame_id}.png"
        Image.new("RGB", (8, 8), (index := len(paths), 2, 3)).save(path)
        paths.append(path)
    return tuple(paths)


def _row(plan, left, right, *, cells=64, fraction=.5, shift=(0, 24), psr=3.0):
    frames = [job.capture.frame_id for job in plan.jobs]
    return PlanePairTrace(
        frames[left], frames[right], 0, 0, 0, True, "",
        shift, 1.0, cells, shift, 1.0, cells,
        left_plane=0, right_plane=0, selected=True, psr=psr, score=.8,
        overlap_cells=cells, valid_cells=128,
        overlap_fraction=fraction, cell_shift_rc=shift,
        left_structural_share=.15, right_structural_share=.2)


def _trace(plan, **overrides):
    rows = [_row(plan, 0, 1), _row(plan, 1, 2), _row(plan, 0, 2)]
    if overrides:
        rows[0] = _row(plan, 0, 1, **overrides)
    return SimpleNamespace(pair_attribution=tuple(rows), attribution_sha256="f" * 64)


def test_report_passes_only_with_all_three_measured_pairs():
    plan = _plan()
    production = SimpleNamespace(result_sha256="a" * 64)
    graph = SimpleNamespace(result_sha256="b" * 64, status="assembled", reason="")
    report = subject._report(plan, tuple(job.capture.frame_id for job in plan.jobs),
                             production, graph, _trace(plan))
    assert report.status == "pass"
    assert len(report.pairs) == 3 and report.axes[0].axis == 0
    assert subject.TargetedCaptureQualityReport(**{
        field: getattr(report, field) for field in report.__dataclass_fields__}) == report


@pytest.mark.parametrize("metric", [
    {"cells": 63}, {"fraction": .49}, {"shift": (0, 25)}, {"psr": 2.9},
])
def test_measured_target_failure_is_typed(metric):
    plan = _plan()
    report = subject._report(
        plan, tuple(job.capture.frame_id for job in plan.jobs),
        SimpleNamespace(result_sha256="a" * 64),
        SimpleNamespace(result_sha256="b" * 64, status="assembled", reason=""),
        _trace(plan, **metric))
    assert report.status == "fail" and report.reason == "quality_target_failed"


def test_atomic_solver_abstention_with_no_selected_pairs_is_insufficient():
    plan = _plan()
    trace = SimpleNamespace(pair_attribution=(), attribution_sha256="f" * 64)
    graph = SimpleNamespace(result_sha256="b" * 64, status="unavailable",
                            reason="ambiguous_quadrant")
    report = subject._report(plan, tuple(job.capture.frame_id for job in plan.jobs),
                             SimpleNamespace(result_sha256="a" * 64), graph, trace)
    assert report.status == "insufficient_evidence"
    assert report.solver_reason == "ambiguous_quadrant"


def test_png_handoff_rejects_metadata_and_checks_exact_readable_frames(tmp_path):
    plan = _plan()
    paths = _pngs(tmp_path, plan)
    with pytest.raises(DatasetSchemaError, match="PNG-path sequence"):
        subject._paths(plan, {"paths": paths, "camera": "forbidden"})
    assert subject._paths(plan, paths) == paths
    paths[0].write_bytes(b"not png")
    with pytest.raises(DatasetSchemaError, match="unreadable"):
        subject._paths(plan, paths)


def test_normal_production_receives_png_paths_only_and_unavailable_is_typed(
        tmp_path, monkeypatch):
    plan = _plan()
    paths = _pngs(tmp_path, plan)
    seen = {}
    def run(image_inputs, *, project_path):
        seen["inputs"] = tuple(image_inputs)
        seen["project"] = project_path
        return SimpleNamespace(result_sha256="a" * 64, lattice=None)
    monkeypatch.setattr(subject.production_mod, "derive_image_output_from_paths", run)
    report = subject.evaluate_targeted_capture_quality(
        plan=plan, png_handoff=paths, project_path=tmp_path / "project.json")
    assert seen == {"inputs": paths, "project": tmp_path / "project.json"}
    assert report.status == "insufficient_evidence"
    assert not report.pairs


def test_positive_infinity_psr_has_canonical_tag():
    plan = _plan()
    report = subject._report(
        plan, tuple(job.capture.frame_id for job in plan.jobs),
        SimpleNamespace(result_sha256="a" * 64),
        SimpleNamespace(result_sha256="b" * 64, status="assembled", reason=""),
        _trace(plan, psr=float("inf")))
    assert report.pairs[0].psr == "positive_infinity"
    assert len(report.report_sha256) == 64
