"""Focused controls for normal-mode image-derived production wiring."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_entry import _refused_result, _valid_result  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production as subject  # noqa: E402
from mc_photogrammetry.project.store import load_project  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_real_entry_retains_live_pixels_through_lattice(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    assert all(frame.image is not None for frame in entry.session.frames)
    original = subject.lattice_mod.measure_image_derived_lattice
    seen = []

    def measure(**kwargs):
        seen.append(kwargs["session"])
        assert kwargs["session"] is entry.session
        assert all(frame.image is not None for frame in kwargs["session"].frames)
        return original(**kwargs)

    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    monkeypatch.setattr(subject.lattice_mod,
                        "measure_image_derived_lattice", measure)
    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "other.mcscan")
    assert seen == [entry.session]
    assert isinstance(result, subject.ImageDerivedProductionResult)
    assert result.stage in ("lattice", "geometry", "output")
    # The durable identity is rebound to the requested logical target, while the
    # live pixel-carrying session is the same accepted object throughout.
    assert result.entry is not entry
    assert result.entry.project_path == str(tmp_path / "other.mcscan")
    assert result.entry.session is entry.session
    assert all(frame.image is not None for frame in result.entry.session.frames)
    assert result.entry_sha256 == result.entry.entry_sha256


def test_entry_refusal_short_circuits_lattice(monkeypatch, tmp_path):
    refused = _refused_result(monkeypatch, tmp_path)
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: refused)
    monkeypatch.setattr(subject.lattice_mod, "measure_image_derived_lattice",
                        lambda **_k: pytest.fail("lattice must not run"))
    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "x.mcscan")
    assert result.stage == "entry"
    assert result.status == "geometry_unavailable"


def test_lattice_refusal_propagates_exactly(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    refusal = plane_placement._make_refusal("invalid_input", {"field": "test"})
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    monkeypatch.setattr(subject.lattice_mod, "measure_image_derived_lattice",
                        lambda **_k: refusal)
    assert subject.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "x.mcscan") is refusal


def test_production_forwards_the_exact_accepted_entry_graph(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    lattice = type("Lattice", (), {
        "status": "assembled", "result_sha256": "c" * 64})()
    refusal = plane_placement._make_refusal("invalid_input", {"field": "stop"})
    seen = []
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    monkeypatch.setattr(subject.lattice_mod, "measure_image_derived_lattice",
                        lambda **_k: lattice)

    def geometry(**kwargs):
        seen.append(kwargs["connection_graph"])
        return refusal

    monkeypatch.setattr(subject.geometry_mod, "derive_image_derived_geometry", geometry)
    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "x.mcscan")
    assert result is refusal
    assert seen == [entry.graph]


def test_public_api_has_only_paths_project_and_m7_caps():
    import inspect
    names = tuple(inspect.signature(
        subject.derive_image_output_from_paths).parameters)
    assert names == (
        "image_inputs", "project_path", "recursive", "max_frames",
        "max_pairs", "max_total_support", "max_edges", "max_cycle_checks",
        "rotation_tolerance_degrees", "direction_tolerance")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manual_metadata", "destination", "palette", "bounds",
                      "camera_manifest", "scene", "exporter"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Durable schema-3 project destination
# ---------------------------------------------------------------------------

def test_an_existing_project_target_is_refused_before_entry_calls(
        monkeypatch, tmp_path):
    target = tmp_path / "existing.mcscan"
    target.write_text("prior bytes", encoding="utf-8")
    calls = []
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: calls.append(1))

    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=target)

    assert calls == []
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.detail["reason"] == "target_exists"
    assert target.read_text(encoding="utf-8") == "prior bytes"


def test_a_dangling_symlink_target_is_refused_before_entry_calls(
        monkeypatch, tmp_path):
    target = tmp_path / "dangling.mcscan"
    missing = tmp_path / "missing-predecessor.mcscan"
    try:
        target.symlink_to(missing)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable in this environment")
    assert target.is_symlink()
    assert not target.exists()

    calls = []
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: calls.append(1))

    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=target)

    assert calls == []
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.detail["reason"] == "target_exists"
    # The dangling link itself is neither followed, replaced nor removed.
    assert target.is_symlink()
    assert not missing.exists()


def test_a_fresh_refusal_publishes_a_zero_block_schema3_terminal(
        monkeypatch, tmp_path):
    entry = _refused_result(monkeypatch, tmp_path)
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    target = tmp_path / "review.mcscan"

    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=target)

    assert result.status == "geometry_unavailable"
    loaded = load_project(target)
    assert loaded.schema_version == 3
    assert loaded.volume.to_dict()["blocks"] == []
    assert loaded.image_derived_session is not None
    record = loaded.image_derived_production
    assert record["status"] == "geometry_unavailable"
    assert record["promotion"] is None


def test_a_target_race_preserves_the_racer_bytes(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    target = tmp_path / "race.mcscan"

    def racing_entry(*_args, **_kwargs):
        target.write_text("racer bytes", encoding="utf-8")
        return entry

    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        racing_entry)

    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=target)

    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.detail["reason"] == "publish_failed"
    assert target.read_text(encoding="utf-8") == "racer bytes"


def test_an_exception_after_a_race_preserves_the_predecessor(
        monkeypatch, tmp_path):
    target = tmp_path / "predecessor.mcscan"

    def exploding_entry(*_args, **_kwargs):
        target.write_text("predecessor", encoding="utf-8")
        raise subject.entry_mod.ImageDerivedEntryError(
            subject.entry_mod.ENTRY_PERSIST_FAILED, "checkpoint failed")

    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        exploding_entry)

    with pytest.raises(subject.entry_mod.ImageDerivedEntryError):
        subject.derive_image_output_from_paths([tmp_path], project_path=target)

    assert target.read_text(encoding="utf-8") == "predecessor"


def test_a_final_save_failure_returns_a_typed_refusal(monkeypatch, tmp_path):
    entry = _refused_result(monkeypatch, tmp_path)
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    target = tmp_path / "failure.mcscan"

    def failing_publish(project, path):
        raise subject.ProjectStoreError("disk full")

    monkeypatch.setattr(subject, "publish_project_no_clobber", failing_publish)

    result = subject.derive_image_output_from_paths(
        [tmp_path], project_path=target)

    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.detail["reason"] == "publish_failed"
    assert not target.exists()


def test_the_staged_checkpoint_is_removed_after_publication(monkeypatch, tmp_path):
    entry = _refused_result(monkeypatch, tmp_path)
    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    seen = {}
    original = subject.record_mod.checkpoint_path

    def recording_checkpoint(target):
        checkpoint = original(target)
        seen["checkpoint"] = checkpoint
        return checkpoint

    monkeypatch.setattr(subject.record_mod, "checkpoint_path",
                        recording_checkpoint)
    target = tmp_path / "staged.mcscan"

    subject.derive_image_output_from_paths([tmp_path], project_path=target)

    assert "checkpoint" in seen
    assert seen["checkpoint"] != target
    assert not seen["checkpoint"].exists()
    assert not seen["checkpoint"].parent.exists()
    assert target.exists()


def test_entry_identity_binds_the_logical_target_not_the_checkpoint(
        monkeypatch, tmp_path):
    """Two equivalent runs must not persist the deleted random checkpoint identity."""
    from test_image_derived_entry import _patch_matcher, _patch_solver, _two_images

    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    real_build = subject.entry_mod.build_image_derived_entry
    checkpoints = []

    def builder(image_inputs, *, project_path, **kwargs):
        checkpoints.append(Path(project_path))
        kwargs.pop("max_total_support", None)
        return real_build(image_inputs, project_path=project_path,
                          max_total_support=0, **kwargs)

    monkeypatch.setattr(subject.entry_mod, "build_image_derived_entry", builder)
    target = tmp_path / "logical.mcscan"

    first = subject.derive_image_output_from_paths([tmp_path], project_path=target)

    assert isinstance(first, subject.ImageDerivedProductionResult)
    assert len(checkpoints) == 1
    # The entry was built at a random temporary checkpoint, not the final target.
    assert checkpoints[0] != target
    assert not checkpoints[0].exists()
    assert first.entry.project_path == str(target)
    assert first.entry_sha256 == first.entry.entry_sha256
    first_record = load_project(target).image_derived_production
    assert first_record["entry_sha256"] == first.entry.entry_sha256

    # A second equivalent run stages through a *different* random checkpoint but
    # must persist exactly the same logical entry identity.
    target.unlink()
    second = subject.derive_image_output_from_paths([tmp_path], project_path=target)

    assert len(checkpoints) == 2
    assert checkpoints[1] != checkpoints[0]
    assert not checkpoints[1].exists()
    assert second.entry_sha256 == first.entry_sha256
    second_record = load_project(target).image_derived_production
    assert second_record["entry_sha256"] == first_record["entry_sha256"]
