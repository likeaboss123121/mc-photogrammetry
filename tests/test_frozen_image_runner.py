from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mc_photogrammetry.reconstruction import frozen_image_runner as runner
from mc_photogrammetry.reconstruction import (
    lattice, lattice_planes, plane_graph, supported,
)
from mc_photogrammetry.reconstruction.boundary_candidates import (
    PlaneProposal,
    PlaneProposalResult,
)
from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult,
    estimate_image_calibrations,
)
from mc_photogrammetry.vision.gui_mask import create_gui_mask


def _write_png(path: Path, value: int, *, width: int = 96, height: int = 72) -> None:
    rgb = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _calibration(width: int, height: int, focal: float = 80.0) -> CalibrationResult:
    return CalibrationResult(
        focal_pixels=focal, yaw_degrees=12.0, pitch_degrees=-8.0,
        width=width, height=height, inliers=24, segments=31,
        median_residual=0.8, per_axis_inliers=(9, 8, 7),
    )


def _reading(frame: str, width: int, height: int) -> lattice_planes.PlaneReading:
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=80.0, principal_point=(width / 2.0, height / 2.0), centre=(0.0, 0.0),
    )
    return lattice_planes.PlaneReading(
        frame=frame, axis=0, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 3.0, 1.0),
                      lattice_planes.DistanceAlternative(2.0, 6.0, 0.75)),
        period_pixels=48.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=64,
        polygon=((10.0, 10.0), (80.0, 10.0), (80.0, 60.0), (10.0, 60.0)),
        coverage_cells=16, structural_share=0.6, confidence=0.8,
        rectification=rectification,
    )


def _install_synthetic_stages(monkeypatch) -> None:
    def calibrate(frames):
        return {name: _calibration(image.shape[1], image.shape[0],
                                   70.0 + float(image[20, 20, 0]))
                for name, image in frames.items()}

    def proposals(image, usable, calibration, **_kwargs):
        mask = usable.copy()
        if int(image[20, 20, 0]) > 40:
            mask[:, :3] = False
        mask.setflags(write=False)
        proposal = PlaneProposal(
            axis=0, side=1, polygon=((4.0, 4.0), (80.0, 4.0),
                                     (80.0, 60.0), (4.0, 60.0)),
            mask=mask, strength=0.8, sources=("synthetic_pixels",), coverage=0.7,
        )
        return PlaneProposalResult("ok", "", (proposal,), {"atoms": 4, "elapsed_ms": 2.0})

    def readings(image, usable, calibration, proposal_result, *, frame, **_kwargs):
        return (_reading(frame, image.shape[1], image.shape[0]),)

    def signature(image, reading, b, **_kwargs):
        cells = np.asarray([[1.0, 0.5], [0.25, 0.0]], dtype=np.float32) * b
        return lattice_planes.PlaneSignature(
            reading=reading, blocks_per_period=b, cells=cells,
            interior=cells * 0.1, coverage=np.ones_like(cells), origin_uv=(0.0, 0.0),
        )

    monkeypatch.setattr(runner, "estimate_image_calibrations", calibrate)
    monkeypatch.setattr(runner.region_growth, "propose_image_planes_v2", proposals)
    monkeypatch.setattr(lattice_planes, "frame_plane_readings_from_proposals", readings)
    monkeypatch.setattr(lattice_planes, "plane_signature", signature)
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *_a, **_k: ())


def test_synthetic_pipeline_freezes_all_stages_and_proposal_signatures(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    paths = [tmp_path / "source" / "view-00.png", tmp_path / "source" / "view-01.png"]
    _write_png(paths[0], 30)
    _write_png(paths[1], 45)

    output = tmp_path / "freeze.json"
    frozen = runner.run_frozen_image_pipeline(paths, output)
    loaded = runner.load_and_verify_frozen_image(output)

    assert loaded["artifact_id"] == frozen["artifact_id"]
    assert loaded["labels_present"] is False
    assert len(loaded["frames"]) == 2
    assert all(frame["calibration"]["focal_pixels"] in (100.0, 115.0)
               for frame in loaded["frames"])
    assert all(len(frame["proposal"]["proposals"]) == 1 for frame in loaded["frames"])
    assert all(len(frame["readings"]) == 1 for frame in loaded["frames"])
    assert all(len(frame["signatures"]) == 2 for frame in loaded["frames"])
    assert set(loaded["code_hashes"]) == runner._CODE_HASH_KEYS
    assert loaded["code_hashes"]["lattice"] == hashlib.sha256(
        Path(lattice.__file__).read_bytes()).hexdigest()
    assert loaded["code_hashes"]["supported"] == hashlib.sha256(
        Path(supported.__file__).read_bytes()).hexdigest()
    assert loaded["configuration"]["calibration"]["grouping"] == "independent"
    assert loaded["graph"]["solution"]["reason"] == "no_matches"
    assert loaded["mask_sidecar"]["size_bytes"] > 0
    assert "elapsed_ms" not in loaded["frames"][0]["proposal"]["stats"]


def test_runner_passes_exact_complete_reading_sequence_to_graph_matcher(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    source_reader = lattice_planes.frame_plane_readings_from_proposals
    staged_readings = []

    def capture_readings(*args, **kwargs):
        result = source_reader(*args, **kwargs)
        staged_readings.extend(result)
        return result

    graph_calls = []

    def capture_graph(readings, signatures, **kwargs):
        graph_calls.append((readings, signatures))
        return ()

    monkeypatch.setattr(lattice_planes, "frame_plane_readings_from_proposals",
                        capture_readings)
    monkeypatch.setattr(plane_graph, "match_plane_graph", capture_graph)
    paths = [tmp_path / "capture" / "v0.png", tmp_path / "capture" / "v1.png"]
    _write_png(paths[0], 30)
    _write_png(paths[1], 45)
    runner.run_frozen_image_pipeline(paths, tmp_path / "complete-universe.json")

    assert len(graph_calls) == 1
    passed_readings, passed_signatures = graph_calls[0]
    assert len(passed_readings) == len(staged_readings) == 2
    assert all(actual is expected for actual, expected
               in zip(passed_readings, staged_readings))
    assert len(passed_signatures) == 4
    assert all(any(signature.reading is reading for reading in passed_readings)
               for signature in passed_signatures)


def test_artifact_id_ignores_path_and_output_name_but_preserves_input_order(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    first = [tmp_path / "a" / "v0.png", tmp_path / "a" / "v1.png"]
    second = [tmp_path / "b" / "v0.png", tmp_path / "b" / "v1.png"]
    for path, value in zip(first, (20, 50)):
        _write_png(path, value)
    for path, source in zip(second, first):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    one = runner.run_frozen_image_pipeline(first, tmp_path / "one.json")
    two = runner.run_frozen_image_pipeline(second, tmp_path / "elsewhere" / "two.json")
    swapped = runner.run_frozen_image_pipeline(second[::-1], tmp_path / "swapped.json")
    assert one["artifact_id"] == two["artifact_id"]
    assert one["artifact_id"] != swapped["artifact_id"]


def test_freeze_verifier_rejects_unknown_forbidden_and_tampered_payloads(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    image = tmp_path / "v0.png"
    _write_png(image, 40)
    freeze = tmp_path / "freeze.json"
    runner.run_frozen_image_pipeline([image], freeze)
    original = freeze.read_bytes()

    document = json.loads(original)
    document["unexpected"] = True
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(runner.FrozenImageRunError, match="top-level"):
        runner.load_and_verify_frozen_image(freeze)

    document = json.loads(original)
    document["frames"][0]["unexpected"] = True
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(runner.FrozenImageRunError, match="artifact id|keys"):
        runner.load_and_verify_frozen_image(freeze)

    # Recompute a valid artifact id after injecting a forbidden key to prove the
    # forbidden-key guard is independent of the digest guard.
    document = json.loads(original)
    document["frames"][0]["proposal"]["stats"]["ground_truth"] = 1
    document["artifact_id"] = runner._artifact_id(document)
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(runner.FrozenImageRunError, match="forbidden"):
        runner.load_and_verify_frozen_image(freeze)

    document = json.loads(original)
    document["code_hashes"]["unexpected_module"] = "0" * 64
    document["artifact_id"] = runner._artifact_id(document)
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(runner.FrozenImageRunError, match="provenance keys"):
        runner.load_and_verify_frozen_image(freeze)

    freeze.write_bytes(original)
    sidecar_name = json.loads(original)["mask_sidecar"]["file"]
    sidecar = freeze.with_name(sidecar_name)
    corrupted = bytearray(sidecar.read_bytes())
    corrupted[0] ^= 1
    sidecar.write_bytes(corrupted)
    with pytest.raises(runner.FrozenImageRunError, match="digest|payload"):
        runner.load_and_verify_frozen_image(freeze)


def test_verifier_rejects_nonfinite_json_and_oversized_array_shape(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    image = tmp_path / "v0.png"
    _write_png(image, 40)
    freeze = tmp_path / "freeze.json"
    runner.run_frozen_image_pipeline([image], freeze)
    original = freeze.read_bytes()

    nonfinite = original.replace(b'"labels_present":false', b'"labels_present":NaN')
    assert nonfinite != original
    freeze.write_bytes(nonfinite)
    with pytest.raises(runner.FrozenImageRunError, match="non-finite"):
        runner.load_and_verify_frozen_image(freeze)

    document = json.loads(original)
    document["frames"][0]["signatures"][0]["arrays"]["cells"]["shape"] = [
        runner.MAX_SIGNATURE_CELLS + 1, 2]
    document["artifact_id"] = runner._artifact_id(document)
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(runner.FrozenImageRunError, match="cell cap"):
        runner.load_and_verify_frozen_image(freeze)


def test_freeze_files_publish_atomically_with_json_last(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    image = tmp_path / "v0.png"
    _write_png(image, 40)
    output = tmp_path / "atomic.json"
    replaced = []
    original_replace = runner.os.replace

    def observe_replace(source, destination):
        replaced.append(Path(destination).name)
        return original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", observe_replace)
    runner.run_frozen_image_pipeline([image], output)
    old_document = json.loads(output.read_bytes())
    old_sidecar = output.with_name(old_document["mask_sidecar"]["file"])
    assert replaced == [old_document["mask_sidecar"]["file"], output.name]
    assert not list(tmp_path.glob("*.tmp"))

    # A failed replacement for changed content leaves the previous JSON and its
    # content-addressed sidecar readable as one complete pair.
    previous_freeze = output.read_bytes()
    _write_png(image, 41)
    def fail_json_replace(source, destination):
        if Path(destination) == output:
            raise OSError("synthetic atomic publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_json_replace)
    with pytest.raises(OSError, match="synthetic atomic"):
        runner.run_frozen_image_pipeline([image], output)
    assert output.read_bytes() == previous_freeze
    assert old_sidecar.exists()
    runner.load_and_verify_frozen_image(output)
    assert not list(tmp_path.glob(".*.tmp"))


def test_forbidden_proposal_stats_become_typed_abstention_not_invalid_freeze(
    tmp_path, monkeypatch,
):
    _install_synthetic_stages(monkeypatch)
    image = tmp_path / "v0.png"
    _write_png(image, 40)
    monkeypatch.setattr(
        runner.region_growth, "propose_image_planes_v2",
        lambda *_args, **_kwargs: PlaneProposalResult(
            "ok", "", (), {"reference": "poison"}),
    )
    freeze = tmp_path / "forbidden-stats.json"
    result = runner.run_frozen_image_pipeline([image], freeze)
    assert result["frames"][0]["proposal"]["reason"] == "invalid_proposal_stats"
    verified = runner.load_and_verify_frozen_image(freeze)
    assert verified["frames"][0]["proposal"]["stats"] == {}


def test_blank_pixel_inputs_create_typed_abstentions_without_expected_data_escape(
    tmp_path, monkeypatch,
):
    # Calibration is exercised for real here; a featureless image has too few
    # structural segments and safely abstains before proposal growth.
    image = tmp_path / "blank.png"
    blank = np.zeros((96, 128, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    image.write_bytes(encoded.tobytes())

    result = runner.run_frozen_image_pipeline([image], tmp_path / "blank-freeze.json")
    frame = result["frames"][0]
    assert frame["calibration"] is None
    assert frame["calibration_reason"] == "insufficient_image_evidence"
    assert frame["proposal"]["status"] == "abstain"
    assert frame["readings"] == []
    assert result["graph"]["solution"]["reason"] == "no_plane_reading"
    runner.load_and_verify_frozen_image(tmp_path / "blank-freeze.json")


def test_image_calibration_is_independent_even_for_equal_resolution(monkeypatch):
    estimates = []

    def estimate(grey):
        focal = 120.0 if int(np.median(grey)) < 80 else 240.0
        estimates.append(focal)
        return _calibration(grey.shape[1], grey.shape[0], focal)

    monkeypatch.setattr("mc_photogrammetry.reconstruction.calibration.estimate_camera", estimate)
    monkeypatch.setattr(
        "mc_photogrammetry.reconstruction.calibration.estimate_session",
        lambda *_a, **_k: pytest.fail("same-resolution frames must not share a fitted FOV"),
    )
    frames = {
        "dark": np.full((80, 120, 3), 40, dtype=np.uint8),
        "bright": np.full((80, 120, 3), 210, dtype=np.uint8),
    }
    results = estimate_image_calibrations(frames)
    assert results["dark"].focal_pixels == 120.0
    assert results["bright"].focal_pixels == 240.0
    assert estimates == [240.0, 120.0]  # deterministic sorted frame-id order


def test_calibration_zero_fill_makes_excluded_gui_pixels_irrelevant(monkeypatch):
    observed = []

    def estimate(grey):
        observed.append(grey.copy())
        return _calibration(grey.shape[1], grey.shape[0])

    monkeypatch.setattr("mc_photogrammetry.reconstruction.calibration.estimate_camera", estimate)
    base = np.full((100, 160, 3), 100, dtype=np.uint8)
    changed = base.copy()
    excluded = create_gui_mask(100, 160)
    changed[excluded] = 220

    estimate_image_calibrations({"base": base, "changed": changed})
    np.testing.assert_array_equal(observed[0], observed[1])
    # The estimator still receives zero-fill boundaries around the excluded
    # GUI shapes; this test pins pixel authority, not an alternate edge policy.


def test_runner_public_surface_is_only_paths_and_freeze_and_has_no_metadata_imports():
    signature = inspect.signature(runner.run_frozen_image_pipeline)
    assert list(signature.parameters) == ["image_paths", "freeze_path"]
    source_path = Path(runner.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name.startswith(("mc_photogrammetry.dataset", "mc_photogrammetry.pipeline"))
                   for name in imported)
    assert "apply_manual_mask" not in source_path.read_text(encoding="utf-8")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == runner._code_hashes()["runner"]


@pytest.mark.parametrize("path_value", ["single.png", Path("single.png")])
def test_single_path_instead_of_ordered_sequence_is_a_typed_argument_error(tmp_path, path_value):
    with pytest.raises(runner.FrozenImageRunError):
        runner.run_frozen_image_pipeline(path_value, tmp_path / "bad.json")


def test_prior_r4_freeze_remains_schema_v1_loadable_without_mutation():
    freeze = (Path(__file__).resolve().parents[1] / "exports" / "evaluation"
              / "frozen-image-run-r4-l1b-canal-town-2026-09-13" / "freeze.json")
    assert freeze.is_file()
    document_before = freeze.read_bytes()
    document = json.loads(document_before)
    sidecar = freeze.with_name(document["mask_sidecar"]["file"])
    sidecar_before = sidecar.read_bytes()

    loaded = runner.load_and_verify_frozen_image(freeze)

    assert loaded["schema_version"] == 1
    assert loaded["artifact_id"] == "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
    assert len(loaded["frames"]) == 8
    assert loaded["labels_present"] is False
    assert freeze.read_bytes() == document_before
    assert sidecar.read_bytes() == sidecar_before


def test_additive_match_shift_is_excluded_from_frozen_v1_record():
    match = plane_graph.PlaneMatch(
        left="a", right="b", left_axis=0, right_axis=0,
        left_b=1.0, right_b=1.0, quadrant_of_right=0, psr=4.0,
        peak=1.0, second_peak=0.1, overlap_cells=32,
        integer_residual=None, score=0.8, left_plane=0, right_plane=0,
        cell_shift_rc=(-3, 4),
    )
    record = runner._match_record(match)
    assert set(record) == runner._MATCH_KEYS
    assert "cell_shift_rc" not in record
    assert record["left"] == "a" and record["right"] == "b"


@pytest.mark.parametrize("count", (9, 32))
def test_extended_frame_bound_preserves_every_ordered_input(tmp_path, monkeypatch, count):
    _install_synthetic_stages(monkeypatch)
    paths = [tmp_path / f"view-{i:02}.png" for i in range(count)][::-1]
    for i, path in enumerate(paths): _write_png(path, 20 + i)
    seen = []
    original = runner.estimate_image_calibrations
    monkeypatch.setattr(runner, "estimate_image_calibrations", lambda frames: (
        seen.extend(frames), original(frames))[1])
    output = tmp_path / "extended.json"
    document = runner.run_frozen_image_pipeline(iter(paths), output)
    verified = runner.load_and_verify_frozen_image(output)
    expected = [path.name for path in paths]
    assert seen == expected
    assert [item["basename"] for item in document["inputs"]] == expected
    assert [item["frame_id"] for item in document["frames"]] == expected
    assert len(verified["frames"]) == count
    assert document["configuration"]["input"]["max_frames"] == runner.MAX_INPUT_FRAMES == 32
    assert set(document["configuration"]["input"]) == {"max_frames", "max_pixels_per_frame", "order"}
    assert document["schema_version"] == 1 and verified["artifact_id"] == document["artifact_id"]


def test_thirty_third_input_refuses_before_decode_or_unbounded_consumption(tmp_path, monkeypatch):
    seen = []
    def paths():
        for i in range(33):
            seen.append(i)
            yield tmp_path / f"view-{i:02}.png"
        pytest.fail("consumer requested a thirty-fourth path")
    monkeypatch.setattr(runner, "_load_rgb", lambda *a: pytest.fail("count refusal decoded input"))
    output = tmp_path / "refused.json"
    with pytest.raises(runner.FrozenImageRunError, match="between 1 and 32"):
        runner.run_frozen_image_pipeline(paths(), output)
    assert seen == list(range(33))
    assert not output.exists() and not list(tmp_path.iterdir())


def test_per_frame_pixel_refusal_does_not_retain_or_calibrate_image(tmp_path, monkeypatch):
    path = tmp_path / "oversize.png"; _write_png(path, 40)
    monkeypatch.setattr(runner, "MAX_INPUT_PIXELS", 96 * 72 - 1)
    monkeypatch.setattr(runner, "create_gui_mask", lambda *a: pytest.fail("over-cap frame retained mask"))
    monkeypatch.setattr(runner, "estimate_image_calibrations", lambda *a: pytest.fail("over-cap frame calibrated"))
    document = runner.run_frozen_image_pipeline([path], tmp_path / "refusal.json")
    assert document["inputs"][0]["width"] == 96 and document["inputs"][0]["height"] == 72
    assert document["frames"][0]["image_reason"] == "image_pixel_cap"
    assert document["frames"][0]["calibration"] is None
    assert document["frames"][0]["readings"] == document["frames"][0]["signatures"] == []


def test_aggregate_pixel_bound_rejects_before_retaining_overflow_frame(tmp_path, monkeypatch):
    assert runner.MAX_TOTAL_INPUT_PIXELS == 48_000_000 and runner.MAX_INPUT_PIXELS == 6_000_000
    paths = [tmp_path / f"view-{i}.png" for i in range(4)]
    for path in paths: _write_png(path, 40)
    monkeypatch.setattr(runner, "MAX_TOTAL_INPUT_PIXELS", 2 * 96 * 72)
    decoded, retained = [], []
    load, mask = runner._load_rgb, runner.create_gui_mask
    def tracked_load(path):
        decoded.append(path.name); return load(path)
    def tracked_mask(*args):
        retained.append(args); return mask(*args)
    monkeypatch.setattr(runner, "_load_rgb", tracked_load)
    monkeypatch.setattr(runner, "create_gui_mask", tracked_mask)
    monkeypatch.setattr(runner, "estimate_image_calibrations", lambda *a: pytest.fail("failed intake reached stages"))
    output = tmp_path / "output" / "refused.json"
    with pytest.raises(runner.FrozenImageRunError, match="aggregate retained image pixels"):
        runner.run_frozen_image_pipeline(paths, output)
    assert decoded == [path.name for path in paths[:3]] and len(retained) == 2
    assert not output.parent.exists()


def test_sidecar_budget_rejects_before_buffer_growth(monkeypatch):
    monkeypatch.setattr(runner, "MAX_SIDECAR_BYTES", 8)
    sidecar = runner._Sidecar()
    sidecar.add(b"12345678", shape=(8,), dtype="uint8")
    before = bytes(sidecar.data), list(sidecar.ranges)
    with pytest.raises(runner.FrozenImageRunError, match="sidecar.*size cap"):
        sidecar.add(b"9", shape=(1,), dtype="uint8")
    assert (bytes(sidecar.data), sidecar.ranges) == before


@pytest.mark.parametrize("stage", ("proposal", "signature", "json"))
def test_writer_budget_failure_publishes_no_complete_or_partial_freeze(tmp_path, monkeypatch, stage):
    _install_synthetic_stages(monkeypatch)
    source = tmp_path / "source.png"; _write_png(source, 40)
    output = tmp_path / "not-created" / "freeze.json"
    if stage == "json": monkeypatch.setattr(runner, "MAX_FREEZE_JSON_BYTES", 32)
    else:
        # Proposal is864 packed bytes. Signature cell/interior arrays are16
        # bytes each; the second array must fail after the first was admitted.
        monkeypatch.setattr(runner, "MAX_SIDECAR_BYTES", 863 if stage == "proposal" else 864 + 16)
    monkeypatch.setattr(runner, "_atomic_write", lambda *a: pytest.fail("budget failure reached publication"))
    with pytest.raises(runner.FrozenImageRunError, match="size cap"):
        runner.run_frozen_image_pipeline([source], output)
    assert not output.parent.exists()


@pytest.mark.parametrize("stage", ("sidecar", "json"))
def test_writer_budget_failure_preserves_previous_complete_pair(tmp_path, monkeypatch, stage):
    _install_synthetic_stages(monkeypatch)
    source = tmp_path / "source.png"; _write_png(source, 40)
    output = tmp_path / "freeze.json"
    previous = runner.run_frozen_image_pipeline([source], output)
    old_json = output.read_bytes(); old_sidecar = output.with_name(previous["mask_sidecar"]["file"])
    old_binary = old_sidecar.read_bytes(); inventory = {p.name for p in tmp_path.iterdir()}
    _write_png(source, 41)
    with monkeypatch.context() as bounds:
        bounds.setattr(runner, "MAX_SIDECAR_BYTES" if stage == "sidecar" else "MAX_FREEZE_JSON_BYTES", 32)
        with pytest.raises(runner.FrozenImageRunError, match="size cap"):
            runner.run_frozen_image_pipeline([source], output)
    assert output.read_bytes() == old_json and old_sidecar.read_bytes() == old_binary
    assert {p.name for p in tmp_path.iterdir()} == inventory
    assert runner.load_and_verify_frozen_image(output)["artifact_id"] == previous["artifact_id"]


@pytest.mark.parametrize("value", ({"emoji": "\U0001f600" * 5}, {"escapes": "\n\"\\" * 6},
                                  {"floats": [.1, -0., 1.e30], "unicode": "caf\u00e9"}))
def test_bounded_json_preserves_exact_utf8_canonical_bytes_and_boundary(value):
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    assert runner._canonical_json(value, max_bytes=len(expected)) == expected
    with pytest.raises(runner.FrozenImageRunError, match="JSON.*size cap"):
        runner._canonical_json(value, max_bytes=len(expected) - 1)


@pytest.mark.parametrize("count", (1, 8))
def test_one_to_eight_stage_records_are_unchanged_except_configuration_value(tmp_path, monkeypatch, count):
    _install_synthetic_stages(monkeypatch)
    paths = [tmp_path / f"view-{i}.png" for i in range(count)]
    for path in paths: _write_png(path, 40)
    with monkeypatch.context() as old_bound:
        old_bound.setattr(runner, "MAX_INPUT_FRAMES", 8)
        old = runner.run_frozen_image_pipeline(paths, tmp_path / "old.json")
    current = runner.run_frozen_image_pipeline(paths, tmp_path / "current.json")
    assert old["configuration"]["input"]["max_frames"] == 8
    assert current["configuration"]["input"]["max_frames"] == 32
    assert old["artifact_id"] != current["artifact_id"]
    for doc in (old, current):
        doc.pop("artifact_id"); doc["mask_sidecar"].pop("file")
        doc["configuration"]["input"]["max_frames"] = 32
    assert old == current


def test_nine_frame_solver_assignment_refusal_is_unchanged(tmp_path, monkeypatch):
    _install_synthetic_stages(monkeypatch)
    paths = [tmp_path / f"view-{i}.png" for i in range(9)]
    for path in paths: _write_png(path, 40)
    def matches(readings, signatures, **kwargs):
        assert len(readings) == 9 and len(signatures) == 18
        return tuple(plane_graph.PlaneMatch(
            left=left.frame, right=right.frame, left_axis=0, right_axis=0,
            left_b=1., right_b=1., quadrant_of_right=0, psr=5., peak=1., second_peak=.1,
            overlap_cells=16, integer_residual=None, score=.8, left_plane=0, right_plane=0)
            for left, right in zip(readings, readings[1:]))
    monkeypatch.setattr(plane_graph, "match_plane_graph", matches)
    output = tmp_path / "nine.json"
    document = runner.run_frozen_image_pipeline(paths, output)
    assert plane_graph.MAX_ASSIGNMENTS == 16_384
    assert len(document["frames"]) == 9 and len(document["graph"]["matches"]) == 8
    assert document["graph"]["solution"]["reason"] == "component_too_large"
    assert document["graph"]["solution"]["scale_known"] is False
    assert runner.load_and_verify_frozen_image(output)["graph"]["solution"]["reason"] == "component_too_large"
