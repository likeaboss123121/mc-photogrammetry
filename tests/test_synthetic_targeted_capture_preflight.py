from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as subject
from mc_photogrammetry.reconstruction.camera import CameraModel
from test_evaluation_reading_region_mask import _reading


def _frozen(value):
    result = np.asarray(value).copy()
    result.flags.writeable = False
    return result


def _evidence(mask):
    rgb = _frozen(np.zeros((*mask.shape, 3), dtype=np.uint8))
    ownership = _frozen(np.where(mask, 0, -1).astype(np.int64))
    usable = _frozen(np.ones(mask.shape, dtype=bool))
    surface = truth_mod.RenderedTruthSurface("east", 10.0, _frozen(mask))
    return truth_mod.RenderedTruthEvidence(
        "mc-photogrammetry-rendered-truth-evidence-v1",
        CameraModel((12., 5., 5.), 90., 0.), "a" * 64, "b" * 64, "c" * 64,
        rgb, ownership, (((0, 0, 0), "east"),), (surface,), usable)


def test_readings_bind_only_by_exact_unique_truth_containment():
    reading, mask = _reading()
    frames = ("f0", "f1", "f2", "context")
    readings = tuple(replace(reading, frame=frame) for frame in frames[:3])
    evidences = tuple(_evidence(mask) for _ in frames)
    bindings = subject._surface_bindings(evidences, frames, readings)
    assert bindings == {(frame, 0): "east:10" for frame in frames[:3]}
    ambiguous = replace(
        evidences[0], surfaces=evidences[0].surfaces +
        (truth_mod.RenderedTruthSurface("east", 11., _frozen(mask)),),
        evidence_sha256="")
    bindings = subject._surface_bindings(
        (ambiguous,) + evidences[1:], frames, readings)
    assert ("f0", 0) not in bindings


def test_binding_diagnostics_report_reverse_coverage_without_binding():
    reading, mask = _reading()
    frame = "f0"
    reading = replace(reading, frame=frame)
    # A broad reading may fully cover a tiny truth owner surface, but that is
    # diagnostic evidence only and cannot bind whole-reading match metrics.
    truth_subset = np.zeros_like(mask)
    covered = np.argwhere(mask)
    truth_subset[tuple(covered[0])] = True
    broad = _evidence(truth_subset)
    bindings, rows = subject._surface_binding_diagnostics(
        (broad,) * 4, (frame, "f1", "f2", "context"), (reading,))
    assert bindings == {}
    assert rows[0].rejection_reason == "partial"
    intersection = rows[0].intersections[0]
    assert intersection.surface_pixels == 1
    assert intersection.truth_coverage == 1.0
    assert 0 < intersection.reading_purity < 1

    # A truth surface extending beyond the reading is only partially covered.
    partial_mask = mask.copy()
    outside = np.argwhere(~mask)
    partial_mask[tuple(covered[0])] = False
    partial_mask[tuple(outside[0])] = True
    partial = _evidence(partial_mask)
    _bindings, rows = subject._surface_binding_diagnostics(
        (partial,) * 4, (frame, "f1", "f2", "context"), (reading,))
    assert rows[0].rejection_reason == "partial"
    assert rows[0].intersections[0].surface_id == "east:10"
    assert 0 < rows[0].intersections[0].reading_purity < 1
    assert 0 < rows[0].intersections[0].truth_coverage < 1

    exact = _evidence(mask)
    second = truth_mod.RenderedTruthSurface("east", 171., _frozen(mask))
    multiple = replace(exact, surfaces=exact.surfaces + (second,),
                       evidence_sha256="")
    _bindings, rows = subject._surface_binding_diagnostics(
        (multiple,) * 4, (frame, "f1", "f2", "context"), (reading,))
    assert rows[0].rejection_reason == "multiple_exact"
    assert rows[0].exact_candidates == ("east:10", "east:171")

    empty = _evidence(np.zeros_like(mask))
    _bindings, rows = subject._surface_binding_diagnostics(
        (empty,) * 4, (frame, "f1", "f2", "context"), (reading,))
    assert rows[0].rejection_reason == "zero"


def test_surface_permutation_is_diagnostic_invariant_and_disjoint_sets_remain_disjoint():
    reading, mask = _reading()
    frames = ("f0", "f1", "f2", "context")
    readings = tuple(replace(reading, frame=frame) for frame in frames[:3])
    base = _evidence(mask)
    other = truth_mod.RenderedTruthSurface("east", 171., _frozen(mask))
    first = replace(base, surfaces=(base.surfaces[0], other), evidence_sha256="")
    reversed_first = replace(base, surfaces=(other, base.surfaces[0]),
                             evidence_sha256="")
    assert subject._surface_binding_diagnostics(
        (first, base, base, base), frames, readings)[1] == \
        subject._surface_binding_diagnostics(
            (reversed_first, base, base, base), frames, readings)[1]

    surface_171 = replace(base, surfaces=(other,), evidence_sha256="")
    bindings, _rows = subject._surface_binding_diagnostics(
        (base, surface_171, base, base), frames, readings)
    sets = [{surface for (bound_frame, _), surface in bindings.items()
             if bound_frame == frame} for frame in frames[:3]]
    assert set.intersection(*sets) == set()


def test_frame_and_feet_eye_camera_identity_are_preserved():
    reading, mask = _reading()
    evidence = _evidence(mask)
    assert evidence.camera.position == (12., 5., 5.)
    assert evidence.camera.eye_height == 1.62
    _bindings, rows = subject._surface_binding_diagnostics(
        (evidence,) * 4, ("frame-a", "f1", "f2", "context"), (reading,))
    assert rows[0].frame_id == reading.frame == "frame-a"


def test_entry_abstention_is_typed_and_temporary_pngs_are_removed(monkeypatch, tmp_path):
    _reading_value, mask = _reading()
    evidences = tuple(_evidence(mask) for _ in range(4))
    seen = []
    def entry(paths, *, project_path):
        seen.extend(Path(path) for path in paths)
        return SimpleNamespace(
            graph=SimpleNamespace(status="unavailable"),
            pose_graph=SimpleNamespace(status="unavailable"),
            reason="no_connections", detail="", entry_sha256="d" * 64)
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry", entry)
    report = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("f0", "f1", "f2", "context"),
        project_path=tmp_path / "project.json")
    assert report.status == "insufficient_evidence"
    assert report.reason == "no_connections"
    assert seen and all(not path.exists() for path in seen)


def test_png_cap_fails_before_pipeline_and_cleans_up(monkeypatch, tmp_path):
    _reading_value, mask = _reading()
    evidences = tuple(_evidence(mask) for _ in range(4))
    monkeypatch.setattr(
        subject.entry_mod, "build_image_derived_entry",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not run")))
    report = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("f0", "f1", "f2", "context"),
        project_path=tmp_path / "project.json", max_png_bytes=1)
    assert report.status == "insufficient_evidence" and report.reason == "png_byte_cap"


def test_report_digest_replay_and_no_delivery_or_launcher_coupling(monkeypatch, tmp_path):
    _reading_value, mask = _reading()
    evidences = tuple(_evidence(mask) for _ in range(4))
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry", lambda *_a, **_k:
        SimpleNamespace(graph=SimpleNamespace(status="unavailable"),
                        pose_graph=SimpleNamespace(status="unavailable"),
                        reason="no_connections", detail="", entry_sha256="d" * 64))
    report = subject.preflight_synthetic_targeted_capture(
        evidences=evidences, frame_ids=("f0", "f1", "f2", "context"),
        project_path=tmp_path / "project.json")
    replay = subject.SyntheticTargetedPreflightReport(**{
        name: getattr(report, name) for name in report.__dataclass_fields__})
    assert replay == report
    text = open(subject.__file__, encoding="utf-8").read()
    assert "delivery" not in text and "launcher" not in text
