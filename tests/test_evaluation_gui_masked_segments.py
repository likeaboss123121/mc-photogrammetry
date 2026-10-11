import inspect
import os
import sys
import base64
import hashlib
import json
import gzip
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.reconstruction.evaluation_gui_masked_segments import (
    MAX_SAMPLES_PER_SEGMENT,
    ParityRefusal,
    SegmentSupportResult,
    canonical_dtype,
    detector_parity_diagnostics,
    execute_one_shot_attempt,
    paired_segment_inputs,
    require_detector_parity,
    split_at_excluded_mask,
)
from mc_photogrammetry.reconstruction import evaluation_gui_masked_segments as study


def test_clean_support_is_retained_with_native_coordinates_and_parent_provenance():
    source = np.asarray([[11.0, 8.0, 92.0, 8.0]], dtype=np.float64)
    source_before = source.copy()
    excluded = np.zeros((32, 128), dtype=bool)
    result = split_at_excluded_mask(source, excluded)

    assert isinstance(result, SegmentSupportResult)
    assert result.segments.shape == (1, 4)
    assert result.segments[0] == pytest.approx(source[0])
    assert result.segments.dtype == source.dtype
    assert not result.segments.flags.writeable
    assert np.array_equal(source, source_before)
    assert result.intervals[0].parent_index == 0
    assert result.intervals[0].interval_index == 0
    assert result.intervals[0].t_start == 0.0
    assert result.intervals[0].t_end == 1.0
    assert result.intervals[0].length == pytest.approx(81.0)
    assert result.sample_count == 163


def test_pure_split_module_does_not_import_native_cv2():
    assert "cv2" not in sys.modules


def test_gui_edge_artifact_only_inside_excluded_pixels_is_removed():
    excluded = np.zeros((40, 80), dtype=bool)
    excluded[:, 20:60] = True
    source = np.asarray([[24.0, 15.0, 55.0, 15.0]])

    result = split_at_excluded_mask(source, excluded)

    assert result.retained_count == 0
    assert result.segments.shape == (0, 4)


def test_line_crossing_gui_mask_is_split_into_long_clean_runs():
    excluded = np.zeros((40, 128), dtype=bool)
    excluded[:, 50:61] = True
    source = np.asarray([[0.0, 12.0, 119.0, 12.0]])

    result = split_at_excluded_mask(source, excluded)

    assert result.retained_count == 2
    assert result.segments[:, 0] == pytest.approx([0.0, 60.5])
    assert result.segments[:, 2] == pytest.approx([49.5, 119.0])
    assert [item.parent_index for item in result.intervals] == [0, 0]
    assert [item.interval_index for item in result.intervals] == [0, 1]
    assert result.intervals[0].t_end < result.intervals[1].t_start
    assert result.intervals[0].length >= 40.0
    assert result.intervals[1].length >= 40.0


def test_partly_masked_diagonal_keeps_distinct_support_intervals_in_order():
    excluded = np.zeros((100, 100), dtype=bool)
    excluded[35:56, 35:56] = True
    source = np.asarray([[5.0, 5.0, 90.0, 90.0]])

    result = split_at_excluded_mask(source, excluded)

    assert result.retained_count == 2
    assert result.segments[0, 0] < result.segments[1, 0]
    assert result.segments[0, 1] < result.segments[1, 1]
    assert result.intervals[0].t_end < result.intervals[1].t_start
    assert all(item.length >= 40.0 for item in result.intervals)


def test_boundary_band_marks_both_transition_sides_and_exact_chebyshev_radius():
    mask = np.zeros((9, 10), dtype=bool)
    mask[3:5, 4:6] = True
    before = mask.copy()
    seeds, band = study.mask_transition_band(mask)

    expected_seeds = np.zeros(mask.shape, dtype=bool)
    expected_seeds[3:5, 3] = True
    expected_seeds[3:5, 4] = True
    expected_seeds[3:5, 5] = True
    expected_seeds[3:5, 6] = True
    expected_seeds[2, 4:6] = True
    expected_seeds[3:5, 4:6] = True
    expected_seeds[5, 4:6] = True
    expected_band = np.zeros(mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(expected_seeds)):
        expected_band[max(0, y - 3):min(mask.shape[0], y + 4),
                      max(0, x - 3):min(mask.shape[1], x + 4)] = True

    assert np.array_equal(seeds, expected_seeds)
    assert np.array_equal(band, expected_band)
    assert band[0, 0] and band[7, 9] and not band[8, 9]
    assert np.array_equal(mask, before)
    assert not seeds.flags.writeable and not band.flags.writeable


def test_boundary_band_does_not_seed_outside_image_but_keeps_real_corner_edge_transition():
    uniform = np.ones((5, 6), dtype=bool)
    uniform_seeds, uniform_band = study.mask_transition_band(uniform)
    assert not uniform_seeds.any()
    assert not uniform_band.any()

    corner = np.zeros((5, 6), dtype=bool)
    corner[0, 0] = True
    seeds, band = study.mask_transition_band(corner)
    assert set(zip(*np.nonzero(seeds))) == {(0, 0), (0, 1), (1, 0)}
    assert band[0, 0] and band[0, 4] and band[3, 3]
    assert not band[4, 4]


def test_boundary_split_b_expands_only_split_exclusion_and_preserves_native_interval_order():
    mask = np.zeros((24, 40), dtype=bool)
    mask[:, 12:16] = True
    source = np.asarray([[0.0, 12.0, 31.0, 12.0]], dtype=np.float64)
    original_mask = mask.copy()

    old_a = split_at_excluded_mask(source, mask, minimum_length=2.5)
    result_a, meta_a = study.split_with_boundary_band(
        source, mask, apply_boundary_band=False, minimum_length=2.5)
    result_b, meta_b = study.split_with_boundary_band(
        source, mask, apply_boundary_band=True, minimum_length=2.5)

    assert np.array_equal(result_a.segments, old_a.segments)
    assert result_a.intervals == old_a.intervals
    assert meta_a["applied"] is False and meta_b["applied"] is True
    assert np.array_equal(meta_a["split_exclusion_mask"], mask)
    assert np.array_equal(meta_b["expanded_mask"], np.logical_or(mask, meta_b["boundary_band"]))
    assert np.array_equal(meta_b["split_exclusion_mask"], meta_b["expanded_mask"])
    assert result_a.segments[:, 0] == pytest.approx([0.0, 15.5])
    assert result_a.segments[:, 2] == pytest.approx([11.0, 31.0])
    assert result_b.segments[:, 0] == pytest.approx([0.0, 19.5])
    assert result_b.segments[:, 2] == pytest.approx([7.0, 31.0])
    assert all(a.t_end < b.t_start for a, b in zip(result_b.intervals, result_b.intervals[1:]))
    assert [item.parent_index for item in result_b.intervals] == [0, 0]
    assert [item.interval_index for item in result_b.intervals] == [0, 1]
    assert np.array_equal(mask, original_mask)
    assert min(item.x0 for item in result_b.intervals) == 0.0
    assert max(item.x1 for item in result_b.intervals) == 31.0


def test_boundary_mask_and_splitter_resource_caps_fail_closed():
    with pytest.raises(ValueError, match="boundary mask pixel cap"):
        study.mask_transition_band(np.zeros((3, 3), dtype=bool), max_mask_pixels=8)
    source = np.asarray([[0.0, 0.0, MAX_SAMPLES_PER_SEGMENT, 0.0]])
    mask = np.zeros((4, MAX_SAMPLES_PER_SEGMENT + 2), dtype=bool)
    with pytest.raises(ValueError, match="per-segment sample cap"):
        study.split_with_boundary_band(source, mask, apply_boundary_band=True)


def test_r3_recipes_freeze_deterministic_family_cameras_and_paired_stress_views():
    recipes = study._fresh_r3_scene_recipes(None)
    repeated = study._fresh_r3_scene_recipes(None)
    assert len(recipes) == 12
    assert len({row["identity"] for row in recipes}) == 12
    assert [row["recipe_json"] for row in recipes] == [row["recipe_json"] for row in repeated]
    ordinary = {row["identity"]: json.loads(row["recipe_json"])
                for row in recipes if row["stratum"] == "ordinary"}
    cameras = {}
    r2 = {row["identity"]: json.loads(row["recipe_json"])
          for row in study._fresh_r2_scene_recipes(None) if row["stratum"] == "ordinary"}
    yaw_range = (min(item["yaw"] for item in r2.values()), max(item["yaw"] for item in r2.values()))
    pitch_range = (min(item["pitch"] for item in r2.values()), max(item["pitch"] for item in r2.values()))
    focal_range = (min(item["focal"] for item in r2.values()), max(item["focal"] for item in r2.values()))
    for row in recipes:
        recipe = json.loads(row["recipe_json"])
        assert recipe["r3_camera_parent_seed"] == 2026100603
        assert yaw_range[0] <= recipe["yaw"] <= yaw_range[1]
        assert pitch_range[0] <= recipe["pitch"] <= pitch_range[1]
        assert focal_range[0] <= recipe["focal"] <= focal_range[1]
        assert recipe["eye"] == [1.25, 3.15, -0.5]
        cameras.setdefault(row["family"], set()).add(
            (tuple(recipe["eye"]), recipe["yaw"], recipe["pitch"], recipe["focal"]))
        if row["base_identity"] is not None:
            base = ordinary[row["base_identity"]]
            assert (recipe["eye"], recipe["yaw"], recipe["pitch"], recipe["focal"]) == (
                base["eye"], base["yaw"], base["pitch"], base["focal"])
    assert len(cameras) == 6
    assert all(len(values) == 1 for values in cameras.values())
    assert len({next(iter(values)) for values in cameras.values()}) == 6


def test_r3_trace_wrapper_keeps_expanded_mask_out_of_public_mask_and_detector_path():
    automatic = np.zeros((24, 110), dtype=bool)
    automatic[:, 40:50] = True
    automatic_before = automatic.copy()
    raw = np.asarray([[0.0, 12.0, 99.0, 12.0]], dtype=np.float64)
    image = np.zeros((24, 110, 3), dtype=np.uint8)
    public_masks = []
    detector_inputs = []

    class Calibration:
        @staticmethod
        def detect_segments(grey):
            detector_inputs.append(grey.copy())
            return raw.copy()

    class GuiMask:
        @staticmethod
        def create_gui_mask(height, width):
            assert (height, width) == automatic.shape
            return automatic.copy()

    def trace_rgb(rgb, frame_id):
        observed = gui.create_gui_mask(*rgb.shape[:2])
        public_masks.append(observed.copy())
        grey = np.zeros(observed.shape, dtype=np.uint8)
        calibration.detect_segments(grey)
        return {"status": "returned", "frame_id": frame_id}

    calibration, gui = Calibration(), GuiMask()
    _, support_a = study.trace_candidate_with_support(
        image, "crop-local", apply_boundary_band=False, record_boundary_masks=True,
        trace_rgb=trace_rgb, calibration_module=calibration, gui_mask_module=gui)
    _, support_b = study.trace_candidate_with_support(
        image, "crop-local", apply_boundary_band=True, record_boundary_masks=True,
        trace_rgb=trace_rgb, calibration_module=calibration, gui_mask_module=gui)

    assert len(public_masks) == len(detector_inputs) == 2
    assert all(np.array_equal(item, automatic_before) for item in public_masks)
    assert np.array_equal(support_a["mask"], support_b["mask"])
    assert np.array_equal(support_a["boundary_band"], support_b["boundary_band"])
    assert np.array_equal(support_b["expanded_mask"],
                          np.logical_or(support_b["mask"], support_b["boundary_band"]))
    assert support_a["split_exclusion_mask_sha256"] == support_a["mask_sha256"]
    assert support_b["split_exclusion_mask_sha256"] == support_b["expanded_mask_sha256"]
    assert support_a["boundary_band_applied"] is False
    assert support_b["boundary_band_applied"] is True
    assert not np.array_equal(support_a["split"].segments, support_b["split"].segments)
    assert np.array_equal(automatic, automatic_before)


def test_output_stays_in_crop_native_pixel_coordinates_without_rebasing():
    excluded = np.zeros((180, 240), dtype=bool)
    source = np.asarray([[101.0, 52.0, 203.0, 52.0]], dtype=np.float32)

    result = split_at_excluded_mask(source, excluded)

    assert result.segments[0] == pytest.approx(source[0])
    assert result.intervals[0].x0 == 101.0
    assert result.intervals[0].x1 == 203.0


def test_gui_only_rgb_is_outside_contract_and_does_not_affect_segment_result():
    parameters = inspect.signature(split_at_excluded_mask).parameters
    assert "rgb" not in parameters
    assert "camera" not in parameters
    assert "truth" not in parameters
    assert "manual_mask" not in parameters

    excluded = np.zeros((60, 100), dtype=bool)
    excluded[:, 40:50] = True
    source = np.asarray([[3.0, 20.0, 96.0, 20.0]])
    first = split_at_excluded_mask(source, excluded)
    # RGB mutation cannot enter this segment-and-mask-only transform.
    gui_rgb_a = np.zeros((60, 100, 3), dtype=np.uint8)
    gui_rgb_b = gui_rgb_a.copy()
    gui_rgb_b[excluded] = (255, 0, 255)
    second = split_at_excluded_mask(source, excluded)

    assert np.array_equal(gui_rgb_a[~excluded], gui_rgb_b[~excluded])
    assert np.array_equal(first.segments, second.segments)
    assert first.intervals == second.intervals


def test_per_parent_intervals_are_nonoverlapping_and_order_is_deterministic():
    excluded = np.zeros((80, 220), dtype=bool)
    excluded[:, 40:50] = True
    excluded[:, 130:140] = True
    source = np.asarray([
        [0.0, 10.0, 200.0, 10.0],
        [0.0, 20.0, 200.0, 20.0],
    ])

    first = split_at_excluded_mask(source, excluded)
    second = split_at_excluded_mask(source.copy(), excluded.copy())

    assert np.array_equal(first.segments, second.segments)
    assert first.intervals == second.intervals
    assert [item.parent_index for item in first.intervals] == [0, 0, 1, 1]
    for parent in (0, 1):
        intervals = [item for item in first.intervals if item.parent_index == parent]
        assert all(a.t_end < b.t_start for a, b in zip(intervals, intervals[1:]))


def test_paired_helper_preserves_baseline_object_and_only_derives_candidate():
    source = np.asarray([[0.0, 15.0, 119.0, 15.0]])
    source_before = source.copy()
    excluded = np.zeros((40, 128), dtype=bool)
    excluded[:, 50:61] = True
    mask_before = excluded.copy()

    baseline, candidate = paired_segment_inputs(source, excluded)

    assert baseline is source
    assert np.array_equal(baseline, source_before)
    assert np.array_equal(excluded, mask_before)
    assert candidate.retained_count == 2
    assert not np.array_equal(candidate.segments, baseline)


def test_work_caps_fail_closed_before_sampling():
    source = np.asarray([[0.0, 0.0, MAX_SAMPLES_PER_SEGMENT, 0.0]])
    excluded = np.zeros((4, MAX_SAMPLES_PER_SEGMENT + 2), dtype=bool)

    with pytest.raises(ValueError, match="per-segment sample cap"):
        split_at_excluded_mask(source, excluded)
    with pytest.raises(ValueError, match="segment input cap"):
        split_at_excluded_mask(
            np.zeros((2, 4)), np.zeros((2, 2), dtype=bool), max_input_segments=1)
    with pytest.raises(ValueError, match="aggregate sample cap"):
        split_at_excluded_mask(
            np.asarray([[0.0, 0.0, 10.0, 0.0]]),
            np.zeros((2, 12), dtype=bool), max_total_samples=10)


@pytest.mark.parametrize("bad_mask", [np.zeros((0, 2), bool), np.zeros((2, 2), np.uint8)])
def test_invalid_masks_are_rejected(bad_mask):
    with pytest.raises(ValueError, match="excluded_mask"):
        split_at_excluded_mask(np.zeros((0, 4)), bad_mask)


def test_invalid_segment_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        split_at_excluded_mask(
            np.asarray([[0.0, 0.0, np.inf, 0.0]]), np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="numeric"):
        split_at_excluded_mask(np.asarray([["a", "b", "c", "d"]]),
                               np.zeros((4, 4), dtype=bool))


def test_detector_parity_accepts_equivalent_numpy_dtype_spellings():
    source = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

    assert canonical_dtype("float64") == canonical_dtype(source.dtype.str)
    result = require_detector_parity(
        source, "float64", [1, 4], __import__("hashlib").sha256(source.tobytes()).hexdigest())

    assert all(check["passed"] for check in result.values())


@pytest.mark.parametrize("field, candidate", [
    ("dtype", "float32"),
    ("shape", [4, 1]),
    ("raw_data_sha256", "0" * 64),
])
def test_detector_parity_independently_reports_each_mismatch(field, candidate):
    import hashlib

    source = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    values = {
        "dtype": "float64",
        "shape": [1, 4],
        "raw_data_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
    }
    values[field] = candidate

    diagnostics = detector_parity_diagnostics(
        source, values["dtype"], values["shape"], values["raw_data_sha256"])

    assert diagnostics[field]["passed"] is False
    assert sum(check["passed"] is False for check in diagnostics.values()) == 1
    with pytest.raises(ParityRefusal) as refusal:
        require_detector_parity(
            source, values["dtype"], values["shape"], values["raw_data_sha256"])
    assert refusal.value.diagnostics == diagnostics


def _attempt_plan():
    return {"execution_state": "NOT_STARTED", "sealed_predictions": [], "evaluations": None,
            "preregistration_sha256": "test-prereg"}


def _member_store(tmp_path, plan, **kwargs):
    path = tmp_path / "payloads.gz"
    path.touch()
    store = study.GzipMemberStore(path, total_bytes=10_000_000,
                                  reserve_bytes=0, **kwargs)
    return store, (lambda case, arm, payload, state: store.append(payload, state, case, arm)), store.validate_all


def _canonical_payload(value):
    return study._canonical(value).encode("utf-8")


def test_gzip_store_accepts_exact_caps_and_rejects_compressed_or_decoded_overflow(tmp_path):
    noise = random.Random(19).randbytes(2048)
    payload = _canonical_payload({"noise_hex": noise.hex()})
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)

    exact_path = tmp_path / "exact.gz"
    exact_path.touch()
    exact = study.GzipMemberStore(
        exact_path, max_decoded_bytes=len(payload),
        max_compressed_bytes=len(compressed), total_bytes=10_000_000, reserve_bytes=0)
    plan = {"sealed_predictions": []}
    descriptor = exact.append(payload, plan, "case", "baseline")
    assert descriptor["decoded_bytes"] == len(payload)
    assert descriptor["compressed_bytes"] == len(compressed)
    assert exact.read_member(descriptor) == payload

    decoded_path = tmp_path / "decoded-over.gz"
    decoded_path.touch()
    decoded_over = study.GzipMemberStore(
        decoded_path, max_decoded_bytes=len(payload) - 1,
        total_bytes=10_000_000, reserve_bytes=0)
    with pytest.raises(RuntimeError, match="decoded_member_byte_cap"):
        decoded_over.append(payload, {"sealed_predictions": []}, "case", "baseline")
    assert decoded_path.stat().st_size == 0

    compressed_path = tmp_path / "compressed-over.gz"
    compressed_path.touch()
    compressed_over = study.GzipMemberStore(
        compressed_path, max_decoded_bytes=len(payload),
        max_compressed_bytes=len(compressed) - 1,
        total_bytes=10_000_000, reserve_bytes=0)
    with pytest.raises(RuntimeError, match="compressed_member_byte_cap"):
        compressed_over.append(payload, {"sealed_predictions": []}, "case", "baseline")
    assert compressed_path.stat().st_size == 0


def test_gzip_store_rejects_incompressible_payload_over_member_cap(tmp_path):
    noise = random.Random(27).randbytes(8192)
    payload = _canonical_payload({"noise_base64": base64.b64encode(noise).decode("ascii")})
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    assert len(compressed) > len(payload) * 0.5
    path = tmp_path / "incompressible.gz"
    path.touch()
    store = study.GzipMemberStore(
        path, max_decoded_bytes=len(payload), max_compressed_bytes=len(compressed) - 1,
        total_bytes=10_000_000, reserve_bytes=0)
    with pytest.raises(RuntimeError, match="compressed_member_byte_cap"):
        store.append(payload, {"sealed_predictions": []}, "case", "baseline")


def test_gzip_store_aggregate_cap_counts_json_sidecar_and_reserve_exactly(tmp_path):
    payload = _canonical_payload({"identity": "case", "arm": "baseline"})
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    descriptor = {
        "identity": "case", "arm": "baseline", "member_index": 0, "offset": 0,
        "compressed_bytes": len(compressed), "decoded_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
    }
    candidate_plan = {"sealed_predictions": [descriptor]}
    json_bytes = len((study._canonical(candidate_plan) + "\n").encode("utf-8"))
    reserve = 64
    exact_total = json_bytes + len(compressed) + reserve

    exact_path = tmp_path / "aggregate-exact.gz"
    exact_path.touch()
    exact = study.GzipMemberStore(exact_path, total_bytes=exact_total,
                                  reserve_bytes=reserve)
    accepted_plan = {"sealed_predictions": []}
    accepted_plan["sealed_predictions"].append(
        exact.append(payload, accepted_plan, "case", "baseline"))
    assert accepted_plan["sealed_predictions"] == [descriptor]

    over_path = tmp_path / "aggregate-over.gz"
    over_path.touch()
    over = study.GzipMemberStore(over_path, total_bytes=exact_total - 1,
                                 reserve_bytes=reserve)
    with pytest.raises(RuntimeError, match="result_byte_cap"):
        over.append(payload, {"sealed_predictions": []}, "case", "baseline")
    assert over_path.stat().st_size == 0


def test_gzip_store_detects_truncation_crc_and_payload_hash_corruption(tmp_path):
    payload = _canonical_payload({"arm": "baseline", "value": 4})
    path = tmp_path / "corruption.gz"
    path.touch()
    store = study.GzipMemberStore(path, total_bytes=10_000_000, reserve_bytes=0)
    descriptor = store.append(payload, {"sealed_predictions": []}, "case", "baseline")

    original = path.read_bytes()
    path.write_bytes(original[:-2])
    with pytest.raises(RuntimeError, match="truncated_sidecar_member"):
        store.read_member(descriptor)

    path.write_bytes(original)
    corrupted = bytearray(original)
    corrupted[-8] ^= 0x01  # gzip member CRC32 trailer
    path.write_bytes(corrupted)
    bad_crc_descriptor = dict(descriptor, compressed_sha256=hashlib.sha256(corrupted).hexdigest())
    with pytest.raises(RuntimeError, match="gzip_member_corrupt"):
        store.read_member(bad_crc_descriptor)

    path.write_bytes(original)
    bad_payload_hash = dict(descriptor, payload_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="payload_hash_mismatch"):
        store.read_member(bad_payload_hash)


def test_short_sidecar_write_preserves_prior_complete_member(tmp_path):
    path = tmp_path / "short.gz"
    path.touch()
    plan = {"sealed_predictions": []}
    good = study.GzipMemberStore(path, total_bytes=10_000_000, reserve_bytes=0)
    first = good.append(_canonical_payload({"n": 1}), plan, "one", "baseline")
    plan["sealed_predictions"].append(first)
    first_bytes = path.read_bytes()

    def short_writer(target, blob):
        with target.open("ab") as stream:
            partial = blob[:-1]
            stream.write(partial)
            stream.flush()
            os.fsync(stream.fileno())
        return len(blob) - 1

    failed = study.GzipMemberStore(
        path, total_bytes=10_000_000, reserve_bytes=0, writer=short_writer)
    with pytest.raises(RuntimeError, match="short_sidecar_write"):
        failed.append(_canonical_payload({"n": 2}), plan, "two", "candidate")
    assert path.read_bytes().startswith(first_bytes)
    assert good.read_member(first) == _canonical_payload({"n": 1})


def test_coordinator_publishes_descriptor_after_append_fsync_and_readback(tmp_path):
    plan = _attempt_plan()
    path = tmp_path / "ordered.gz"
    path.touch()
    events = []

    def writer(target, blob):
        with target.open("ab") as stream:
            count = stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        events.append("append_fsync")
        return count

    store = study.GzipMemberStore(path, total_bytes=10_000_000, reserve_bytes=0, writer=writer)
    original_read = store.read_member

    def readback(descriptor):
        result = original_read(descriptor)
        events.append("readback_validate")
        return result

    store.read_member = readback

    def persist(state):
        if state["sealed_predictions"]:
            events.append("descriptor_publish")
            assert path.stat().st_size == sum(
                item["compressed_bytes"] for item in state["sealed_predictions"])

    with pytest.raises(ParityRefusal):
        execute_one_shot_attempt(
            plan, ("case",), trace_arm=lambda case, arm: {"case": case, "arm": arm},
            validate_pair=lambda *args: {**_passing_diagnostics(), "arm_status": {"passed": False}},
            seal_payload=lambda case, arm, value: value,
            store_payload=lambda case, arm, payload, state: store.append(payload, state, case, arm),
            validate_payloads=store.validate_all, persist=persist,
            score_seals=lambda seals: pytest.fail("truth must not run"),
        )
    assert events[:3] == ["append_fsync", "readback_validate", "descriptor_publish"]
    assert events[3:6] == ["append_fsync", "readback_validate", "descriptor_publish"]


def _passing_diagnostics():
    return {key: {"passed": True} for key in (
        "dtype", "shape", "raw_data_sha256", "arm_status", "rgb_pair")}


def test_one_shot_runner_persists_baseline_before_candidate_and_retains_exception_receipt(tmp_path):
    plan = _attempt_plan()
    store, store_payload, validate_payloads = _member_store(tmp_path, plan)
    snapshots = []
    arm_events = []

    class CandidateError(RuntimeError):
        partial_receipt = {"stages": [{"stage": "create_gui_mask", "input_sha256": "abc"}]}

    def persist(state):
        snapshots.append((state["execution_state"], len(state["sealed_predictions"])))

    def trace_arm(case, arm):
        arm_events.append((case, arm, len(plan["sealed_predictions"])))
        if arm == "baseline":
            return {"trace_result": {"status": "returned", "trace": {"stages": []}}}
        raise CandidateError("candidate crashed")

    def validate(case, baseline, candidate):
        assert baseline["trace_result"]["status"] == "returned"
        assert candidate["trace_result"]["status"] == "trace_refusal"
        assert candidate["trace_result"]["trace"]["stages"][0]["stage"] == "create_gui_mask"
        result = _passing_diagnostics()
        result["arm_status"] = {"passed": False, "reason": candidate["trace_result"]["reason"]}
        return result

    with pytest.raises(ParityRefusal):
        execute_one_shot_attempt(
            plan, ("case-a",), trace_arm=trace_arm, validate_pair=validate,
            seal_payload=lambda case, arm, output: output,
            store_payload=store_payload, validate_payloads=validate_payloads,
            persist=persist,
            score_seals=lambda seals: pytest.fail("truth must not run after refusal"),
        )

    assert arm_events == [("case-a", "baseline", 0), ("case-a", "candidate", 1)]
    assert [(row["identity"], row["arm"]) for row in plan["sealed_predictions"]] == [
        ("case-a", "baseline"), ("case-a", "candidate")]
    refusal = json.loads(store.read_member(plan["sealed_predictions"][1]))
    assert refusal["trace_result"]["status"] == "trace_refusal"
    assert refusal["trace_result"]["trace"]["partial_stage_count"] == 1
    assert plan["execution_state"] == "FAILED"
    assert plan["failure"]["post_hoc_truth_accessed"] is False
    assert plan["failure"]["retry_performed"] is False
    assert snapshots[-1] == ("FAILED", 2)


@pytest.mark.parametrize("field,candidate", [
    ("dtype", "float32"), ("shape", [4, 1]), ("raw_data_sha256", "0" * 64),
])
def test_runner_persists_each_independent_parity_failure_before_refusal(field, candidate, tmp_path):
    import hashlib

    source = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    values = {"dtype": "float64", "shape": [1, 4],
              "raw_data_sha256": hashlib.sha256(source.tobytes()).hexdigest()}
    values[field] = candidate
    plan = _attempt_plan()
    store, store_payload, validate_payloads = _member_store(tmp_path, plan)
    persisted_diagnostics = []

    def validate_pair(*_):
        return detector_parity_diagnostics(
            source, values["dtype"], values["shape"], values["raw_data_sha256"])

    with pytest.raises(ParityRefusal):
        execute_one_shot_attempt(
            plan, ("case",), trace_arm=lambda case, arm: {"status": "returned"},
            validate_pair=validate_pair,
            seal_payload=lambda case, arm, output: {"case": case, "arm": arm},
            store_payload=store_payload, validate_payloads=validate_payloads,
            persist=lambda state: persisted_diagnostics.append(
                json.loads(json.dumps(state["parity_diagnostics"]))),
            score_seals=lambda seals: pytest.fail("truth must not run after parity failure"),
        )
    assert plan["failure"]["parity_diagnostics"][field]["passed"] is False
    assert persisted_diagnostics[-2][-1][field]["passed"] is False
    assert len(plan["sealed_predictions"]) == 2


@pytest.mark.parametrize("failure", ["returned_trace_refusal", "rgb_mismatch", "decode_failure", "support_failure"])
def test_runner_seals_both_arms_before_pair_or_decode_refusal(failure, tmp_path):
    plan = _attempt_plan()
    store, store_payload, validate_payloads = _member_store(tmp_path, plan)
    persisted_arm_counts = []

    def trace_arm(case, arm):
        if failure == "returned_trace_refusal" and arm == "candidate":
            return {"trace_result": {"status": "trace_refusal", "reason": "stage_cap",
                                     "trace": {"trace_complete": False, "stages": []}}}
        if failure == "rgb_mismatch" and arm == "candidate":
            return {"trace_result": {"status": "returned", "rgb_sha256": "wrong"}}
        return {"trace_result": {"status": "returned", "rgb_sha256": "same"}}

    def validate(case, baseline, candidate):
        if failure == "decode_failure":
            raise RuntimeError("decode_failed")
        if failure == "support_failure":
            raise RuntimeError("candidate_support_missing")
        checks = _passing_diagnostics()
        if failure == "returned_trace_refusal":
            checks["arm_status"] = {"passed": False}
        elif failure == "rgb_mismatch":
            checks["rgb_pair"] = {"passed": False}
        return checks

    with pytest.raises((ParityRefusal, RuntimeError)):
        execute_one_shot_attempt(
            plan, ("case",), trace_arm=trace_arm, validate_pair=validate,
            seal_payload=lambda case, arm, output: output,
            store_payload=store_payload, validate_payloads=validate_payloads,
            persist=lambda state: persisted_arm_counts.append(
                (state["execution_state"], len(state["sealed_predictions"]))),
            score_seals=lambda seals: pytest.fail("truth must not run after pair refusal"),
        )
    assert persisted_arm_counts[1] == ("RUNNING", 1)
    assert persisted_arm_counts[2] == ("RUNNING", 2)
    assert plan["execution_state"] == "FAILED"
    assert len(plan["sealed_predictions"]) == 2


def test_one_shot_runner_validates_persisted_24_seals_before_truth_loader(tmp_path):
    plan = _attempt_plan()
    store, store_payload, validate_payloads = _member_store(tmp_path, plan)
    events = []
    case_ids = tuple(f"case-{index}" for index in range(12))

    def trace_arm(case, arm):
        events.append(("arm", case, arm))
        return {"status": "returned", "case": case, "arm": arm}

    def persist(state):
        events.append(("persist", state["execution_state"], len(state["sealed_predictions"])))

    def truth_loader(seals):
        assert plan["execution_state"] == "PREDICTIONS_VALIDATED"
        assert len(seals) == 24
        assert plan["seal_validation"]["validated"] is True
        events.append(("truth", len(seals)))
        return {"evaluations": [], "summary": {"synthetic": True}}

    result = execute_one_shot_attempt(
        plan, case_ids, trace_arm=trace_arm,
        validate_pair=lambda *args: _passing_diagnostics(),
        seal_payload=lambda case, arm, value: value,
        store_payload=store_payload, validate_payloads=validate_payloads,
        persist=persist, score_seals=truth_loader)

    truth_index = next(index for index, event in enumerate(events) if event[0] == "truth")
    assert events[truth_index - 1] == ("persist", "PREDICTIONS_VALIDATED", 24)
    assert result["execution_state"] == "COMPLETE"
    assert len(result["sealed_predictions"]) == 24
    assert result["evaluations"] == []


def test_retry_after_failure_is_refused_by_same_one_shot_path(tmp_path):
    plan = _attempt_plan()
    store, store_payload, validate_payloads = _member_store(tmp_path, plan)
    def reject_refusal(_case, _baseline, _candidate):
        diagnostics = _passing_diagnostics()
        diagnostics["arm_status"] = {"passed": False}
        diagnostics["reason"] = "first failure"
        return diagnostics

    with pytest.raises(ParityRefusal):
        execute_one_shot_attempt(
            plan, ("one",), trace_arm=lambda case, arm: (_ for _ in ()).throw(ValueError("first failure")),
            validate_pair=reject_refusal,
            seal_payload=lambda case, arm, value: value,
            store_payload=store_payload, validate_payloads=validate_payloads,
            persist=lambda state: None,
            score_seals=lambda seals: pytest.fail("scoring must not run"),
        )
    assert plan["execution_state"] == "FAILED"
    with pytest.raises(RuntimeError, match="already_claimed"):
        execute_one_shot_attempt(
            plan, ("one",), trace_arm=lambda *args: pytest.fail("retry attempted"),
            validate_pair=lambda *args: {}, seal_payload=lambda *args: {},
            store_payload=store_payload, validate_payloads=validate_payloads,
            persist=lambda state: pytest.fail("retry persisted"),
            score_seals=lambda seals: pytest.fail("retry scored"),
        )


def test_image_adapters_accept_only_readonly_rgb_and_frame_id():
    assert list(inspect.signature(study._run_baseline_arm).parameters) == ["rgb", "frame_id"]
    assert list(inspect.signature(study._run_candidate_arm).parameters) == ["rgb", "frame_id"]
    assert list(inspect.signature(study._run_r3_a_arm).parameters) == ["rgb", "frame_id"]
    assert list(inspect.signature(study._run_r3_b_arm).parameters) == ["rgb", "frame_id"]


def test_run_study_uses_the_tested_one_shot_path_and_defers_truth_until_24_seals(
        tmp_path, monkeypatch):
    report_path = tmp_path / "study.md"
    json_path = tmp_path / "study.json"
    sidecar_path = tmp_path / "study.payloads.gz"
    sidecar_path.touch()
    identities = tuple(f"case-{index}" for index in range(12))
    strata = ["ordinary"] * 6 + ["gui_contamination"] * 2 + ["crop_boundary"] * 2 + ["texture_geometry"] * 2
    renderer = []
    truth_registry = []
    for index, (identity, stratum) in enumerate(zip(identities, strata)):
        rgb = np.full((2, 2, 3), index, dtype=np.uint8)
        recipe_json = json.dumps({"renderer_only_camera_tag": index}, sort_keys=True)
        recipe_sha = hashlib.sha256(recipe_json.encode()).hexdigest()
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        base_identity = "case-0" if index == 6 else ("case-4" if index == 7 else None)
        renderer.append({"identity": identity, "stratum": stratum,
                         "base_identity": base_identity, "recipe_json": recipe_json,
                         "recipe_sha256": recipe_sha, "rgb_sha256": rgb_sha,
                         "rgb_shape": list(rgb.shape),
                         "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii")})
        truth_registry.append({"identity": identity, "stratum": stratum,
                               "base_identity": base_identity,
                               "renderer_recipe_sha256": recipe_sha})
    plan = {"schema": "test", "study": "CAM-SUPPORT-02R2", "execution_state": "NOT_STARTED",
            "supersedes": {"status": "FAILED; preserve permanently"},
            "dependencies": {}, "renderer_registry": renderer,
            "evaluator_truth_registry": truth_registry,
            "policy": {"camera_gates": {}, "success": {}},
            "sealed_predictions": [], "parity_diagnostics": [],
            "seal_validation": None, "evaluations": None, "summary": None,
            "preregistration_sha256": None}
    plan["preregistration_sha256"] = hashlib.sha256(
        study._canonical({k: v for k, v in plan.items()
                          if k != "preregistration_sha256"}).encode()).hexdigest()
    json_path.write_text(study._canonical(plan) + "\n", encoding="utf-8")
    report_path.write_text("CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n", encoding="utf-8")
    monkeypatch.setattr(study, "_active_study_paths", lambda: (report_path, json_path))
    monkeypatch.setattr(study, "_r2_paths", lambda: (report_path, json_path, sidecar_path))
    monkeypatch.setattr(study, "_expected_dependencies", lambda harness: {})

    class Harness:
        def CameraRecipe(self, *, identity, physical, recipe_json):
            on_truth.append(("camera", identity, read_plan()["execution_state"]))
            return {"identity": identity, "recipe_json": recipe_json}

        def decode_array(self, trace, output):
            return {"segments": segments, "mask": mask}[output]

        def truth_camera(self, scene):
            on_truth.append(("truth", scene["identity"], read_plan()["execution_state"]))
            return object()

        def camera_metrics(self, prediction, truth):
            return {"diagnostic_pass": True}

    def read_plan():
        return json.loads(json_path.read_text(encoding="utf-8"))

    on_truth = []
    monkeypatch.setattr(study, "_study_harness", lambda: Harness())
    mask = np.zeros((2, 2), dtype=bool)
    segments = np.asarray([[0.0, 0.0, 50.0, 0.0]], dtype=np.float64)
    detector_sha = hashlib.sha256(segments.tobytes()).hexdigest()
    mask_sha = hashlib.sha256(mask.tobytes()).hexdigest()
    candidate_support = study.split_at_excluded_mask(segments, mask)

    def prediction(image, frame_id):
        assert image.flags.writeable is False
        assert len(inspect.signature(study._run_baseline_arm).parameters) == 2
        return {"status": "returned", "reason": "", "prediction": {
                    "inliers": 1, "segments": 1, "per_axis_inliers": [1, 1, 1]},
                "trace": {"stages": [
                    {"stage": "detect_segments", "output": "segments"},
                    {"stage": "create_gui_mask", "output": "mask"}]},
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "rgb_shape": list(image.shape)}

    def baseline(image, frame_id):
        state = read_plan()
        assert len(state["sealed_predictions"]) % 2 == 0
        return {"trace_result": prediction(image, frame_id), "support": None}

    def candidate(image, frame_id):
        state = read_plan()
        assert len(state["sealed_predictions"]) % 2 == 1
        assert state["sealed_predictions"][-1]["arm"] == "baseline"
        support = {"mask": mask, "mask_sha256": mask_sha,
                   "raw_segment_sha256": detector_sha,
                   "raw_segment_dtype": "float64", "raw_segment_shape": [1, 4],
                   "split": candidate_support}
        return {"trace_result": prediction(image, frame_id), "support": support}

    monkeypatch.setattr(study, "_run_baseline_arm", baseline)
    monkeypatch.setattr(study, "_run_candidate_arm", candidate)
    result = study.run_study()

    assert result["execution_state"] == "COMPLETE"
    assert len(result["sealed_predictions"]) == 24
    assert all("payload_json" not in descriptor for descriptor in result["sealed_predictions"])
    assert on_truth and all(item[2] == "PREDICTIONS_VALIDATED" for item in on_truth)
    assert read_plan()["seal_validation"]["validated"] is True
    assert "CAM_SUPPORT_EXECUTION_STATE: COMPLETE" in report_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="already_claimed"):
        study.run_study()


def test_r3_driver_pairs_a_and_b_under_same_one_shot_truth_barrier(tmp_path, monkeypatch):
    report_path, json_path, sidecar_path = (
        tmp_path / "r3.md", tmp_path / "r3.json", tmp_path / "r3.payloads.gz")
    sidecar_path.touch()
    identities = tuple(f"r3-case-{index}" for index in range(12))
    strata = (["ordinary"] * 6 + ["gui_contamination"] * 2
              + ["crop_boundary"] * 2 + ["texture_geometry"] * 2)
    renderer, truth_registry = [], []
    for index, (identity, stratum) in enumerate(zip(identities, strata)):
        rgb = np.full((2, 2, 3), index, dtype=np.uint8)
        recipe_json = json.dumps({"renderer_tag": index}, sort_keys=True)
        recipe_sha = hashlib.sha256(recipe_json.encode()).hexdigest()
        base_identity = "r3-case-0" if index == 6 else ("r3-case-4" if index == 7 else None)
        renderer.append({
            "identity": identity, "stratum": stratum, "base_identity": base_identity,
            "family": f"family-{index if index < 6 else (0 if index == 6 else 4 if index == 7 else index)}",
            "recipe_json": recipe_json, "recipe_sha256": recipe_sha,
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "rgb_shape": list(rgb.shape),
            "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
        })
        truth_registry.append({"identity": identity, "stratum": stratum,
                               "base_identity": base_identity,
                               "family": renderer[-1]["family"],
                               "renderer_recipe_sha256": recipe_sha})
    policy = {"camera_gates": {"focal_relative_error": 0.05,
                               "orientation_degrees_mod90": 1.0,
                               "principal_error_over_diagonal": 0.01},
              "success": {"candidate_passes_min": 9, "net_added_passes_min": 3,
                          "baseline_passes_lost_max": 0,
                          "newly_wrong_stress_cases_max": 0,
                          "candidate_stress_wrong_count_le_baseline": True}}
    plan = {"schema": "test-r3", "study": "CAM-BOUNDARY-SPLIT-03",
            "execution_state": "NOT_STARTED", "dependencies": {},
            "treatment_reference": {"fixture": "pinned"},
            "renderer_registry": renderer, "evaluator_truth_registry": truth_registry,
            "policy": policy, "sealed_predictions": [], "parity_diagnostics": [],
            "seal_validation": None, "evaluations": None, "summary": None,
            "preregistration_sha256": None}
    plan["preregistration_sha256"] = hashlib.sha256(
        study._canonical({k: v for k, v in plan.items()
                          if k != "preregistration_sha256"}).encode()).hexdigest()
    json_path.write_text(study._canonical(plan) + "\n", encoding="utf-8")
    report_path.write_text("CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n", encoding="utf-8")
    monkeypatch.setattr(study, "_active_study_paths", lambda: (report_path, json_path))
    monkeypatch.setattr(study, "_r3_paths", lambda: (report_path, json_path, sidecar_path))
    monkeypatch.setattr(study, "_expected_dependencies", lambda harness: {})
    monkeypatch.setattr(study, "_gui_border_reference_pins", lambda: {"fixture": "pinned"})

    mask = np.zeros((60, 110), dtype=bool)
    raw = np.asarray([[0.0, 20.0, 99.0, 20.0]], dtype=np.float64)
    enhance = np.zeros(mask.shape, dtype=np.uint8)
    segment_results = {}
    for apply, label in ((False, "A"), (True, "B")):
        split, derived = study.split_with_boundary_band(
            raw, mask, apply_boundary_band=apply)
        segment_results[label] = split.segments
    camera_events, arm_events = [], []

    def read_plan():
        return json.loads(json_path.read_text(encoding="utf-8"))

    class Harness:
        def CameraRecipe(self, *, identity, physical, recipe_json):
            camera_events.append(("camera", identity, read_plan()["execution_state"]))
            return {"identity": identity, "recipe_json": recipe_json}

        def decode_array(self, trace, output):
            if output == "mask":
                return mask
            if output == "enhance":
                return enhance
            if output == "segments":
                return segment_results["A"]
            raise KeyError(output)

        def truth_camera(self, scene):
            camera_events.append(("truth", scene["identity"], read_plan()["execution_state"]))
            return object()

        def camera_metrics(self, prediction, truth):
            return {"diagnostic_pass": True}

    monkeypatch.setattr(study, "_study_harness", lambda: Harness())

    def adapter(image, frame_id, apply_boundary_band, label):
        state = read_plan()
        expected_seals = 0 if label == "A" else 1
        assert len(state["sealed_predictions"]) % 2 == expected_seals
        arm_events.append(label)
        split, derived = study.split_with_boundary_band(
            raw, mask, apply_boundary_band=apply_boundary_band)
        raw_copy = raw.copy()
        return {
            "adapter_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "trace_result": {
                "status": "returned", "reason": "",
                "prediction": {"inliers": 1, "segments": 1,
                               "per_axis_inliers": [1, 1, 1]},
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "rgb_shape": list(image.shape),
                "trace": {"stages": [
                    {"stage": "detect_segments", "output": "segments"},
                    {"stage": "create_gui_mask", "output": "mask"},
                    {"stage": "enhance", "output": "enhance"}]}},
            "support": {
                "mask": mask.copy(), "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "raw_segments": raw_copy, "raw_segment_sha256": hashlib.sha256(raw_copy.tobytes()).hexdigest(),
                "raw_segment_dtype": str(raw.dtype), "raw_segment_shape": list(raw.shape),
                "split": split, "boundary_seeds": derived["boundary_seeds"],
                "boundary_band": derived["boundary_band"],
                "expanded_mask": derived["expanded_mask"],
                "split_exclusion_mask": derived["split_exclusion_mask"],
                "boundary_band_sha256": hashlib.sha256(derived["boundary_band"].tobytes()).hexdigest(),
                "expanded_mask_sha256": hashlib.sha256(derived["expanded_mask"].tobytes()).hexdigest(),
                "split_exclusion_mask_sha256": hashlib.sha256(derived["split_exclusion_mask"].tobytes()).hexdigest(),
                "boundary_band_applied": apply_boundary_band}}

    monkeypatch.setattr(study, "_run_r3_a_arm",
                        lambda image, frame_id: adapter(image, frame_id, False, "A"))
    monkeypatch.setattr(study, "_run_r3_b_arm",
                        lambda image, frame_id: adapter(image, frame_id, True, "B"))
    result = study.run_study()

    assert result["execution_state"] == "COMPLETE"
    assert len(result["sealed_predictions"]) == 24
    assert arm_events == [arm for _ in identities for arm in ("A", "B")]
    assert result["seal_validation"]["validated"] is True
    assert camera_events and all(state == "PREDICTIONS_VALIDATED"
                                 for _, _, state in camera_events)
    payloads = [json.loads(study.GzipMemberStore(sidecar_path).read_member(item))
                for item in result["sealed_predictions"]]
    assert [item["arm"] for item in payloads[:2]] == ["baseline", "candidate"]
    assert all("boundary_band_sha256" in item["support"] for item in payloads)
    assert "| Case | Stratum | A | B |" in report_path.read_text(encoding="utf-8")


# VALID-NATIVE-CHAIN-01 analytic adapter/coordinator/storage controls.
def _valid_bindings(rgb,arm="candidate"):
    return {"identity":"analytic-frame","arm":arm,"family":"analytic-family",
            "base_identity":None,"stratum":"analytic-control",
            "rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),"rgb_shape":list(rgb.shape),
            "preregistration_sha256":"test-prereg","source_sha256":"test-source",
            "runtime_sha256":"test-runtime"}


def _valid_fake_trace(rgb,mask=None,raw=None,status="ordinary_refusal",candidate=True,config=None):
    if mask is None:
        mask=np.zeros(rgb.shape[:2],dtype=bool)
    if raw is None:
        raw=np.asarray([[2.,32.,98.,32.]],dtype=np.float32)
    events=[]
    def detect(grey,**kwargs):
        events.append(("original_detector",grey.copy(),kwargs))
        return raw
    def create(height,width):
        events.append(("mask",height,width))
        return mask.copy()
    calibration=SimpleNamespace(detect_segments=detect)
    gui=SimpleNamespace(create_gui_mask=create)
    seen=[]
    def trace(image,frame_id):
        assert image is rgb and frame_id=="analytic-frame"
        public_mask=gui.create_gui_mask(*image.shape[:2])
        # Deliberately erase valid source evidence in the enhanced trace input.
        # Candidate must use original RGB; control must receive this exact input.
        enhanced=np.zeros(public_mask.shape,dtype=np.uint8)
        rows=calibration.detect_segments(enhanced,scales=(1.,0.5))
        seen.append(rows)
        return {"status":status,"prediction":None,"trace":{"arrays":{},"stages":[]}}
    kwargs={} if config is None else {"config":config}
    result,support=study.trace_valid_acquisition(rgb,"analytic-frame",candidate=candidate,
        trace_rgb=trace,calibration_module=calibration,gui_mask_module=gui,**kwargs)
    assert calibration.detect_segments is detect and gui.create_gui_mask is create
    return result,support,events,seen


def _analytic_rgb():
    rgb=np.zeros((65,101,3),dtype=np.uint8)
    rgb[32:]=255
    rgb.setflags(write=False)
    return rgb


def test_valid_adapter_uses_original_rgb_and_control_preserves_detector_dtype_bytes():
    rgb=_analytic_rgb()
    candidate,support,events,seen=_valid_fake_trace(rgb)
    assert support["raw_segments"].dtype==np.float64
    assert support["raw_segments"][0]==pytest.approx([2.,32.,98.,32.])
    assert not any(event[0]=="original_detector" for event in events)
    assert seen[0] is support["split"].segments
    raw=np.asarray([[2.,32.,98.,32.]],dtype=np.float32)
    baseline,control,events,seen=_valid_fake_trace(rgb,raw=raw,candidate=False)
    assert control["raw_segments"].dtype==raw.dtype
    assert control["raw_segments"].shape==raw.shape
    assert control["raw_segments"].tobytes()==raw.tobytes()
    assert len([event for event in events if event[0]=="original_detector"])==1
    assert events[1][2]=={"scales":(1.,0.5)}
    assert support["config_sha256"]==control["config_sha256"]
    for name in ("mask","boundary_band","expanded_mask"):
        assert np.array_equal(support[name],control[name])
    assert control["pre_length_observability"]=="unobservable"
    assert baseline["status"]==candidate["status"]=="ordinary_refusal"


def test_valid_payload_contains_exact_replay_masks_detector_runs_and_structured_fields():
    rgb=_analytic_rgb()
    traced,support,_,_=_valid_fake_trace(rgb)
    payload=study.valid_acquisition_payload(_valid_bindings(rgb),traced,support)
    decoded=json.loads(payload)
    assert decoded["control_pre_length"]=="unobservable"
    arrays=decoded["arrays"]
    for name,expected in (("original_mask",support["mask"]),
                          ("expanded_mask",support["expanded_mask"]),
                          ("candidate_pre_length",support["extraction"].pre_length),
                          ("detector_segments",support["raw_segments"]),
                          ("emitted_segments",support["split"].segments)):
        data=base64.b64decode(arrays[name]["base64"],validate=True)
        assert data==expected.tobytes()
        assert arrays[name]["sha256"]==hashlib.sha256(data).hexdigest()
        assert arrays[name]["byte_count"]==len(data)
        assert arrays[name]["shape"]==list(expected.shape)
        assert arrays[name]["dtype"]==expected.dtype.str
    assert arrays["edgels"]["dtype_descr"] is not None
    assert arrays["run_ledger"]["dtype_descr"] is not None
    assert decoded["split_counts"]["supported"]==193
    assert decoded["split_counts"]["emitted_samples"]==193
    assert decoded["split_counts"]["singleton_samples"]==0
    assert decoded["split_counts"]["short_samples"]==0
    assert "camera" not in decoded and "truth" not in decoded


def test_valid_support_audit_includes_singletons_short_runs_and_disjoint_mask_reasons():
    from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import AllocationBudget
    original=np.zeros((5,100),dtype=bool)
    original[:,4:6]=True
    expanded=original.copy(); expanded[:,2:4]=True
    rows=np.asarray([[-1.,2.,9.,2.]],dtype=np.float32)
    split=study.split_at_excluded_mask(rows,expanded)
    audit=study.audit_valid_support(rows,original,expanded,split,budget=AllocationBudget(1_000_000))
    counts=audit["counts"]
    assert counts["frame"]>0 and counts["original"]>0 and counts["band"]>0
    assert counts["supported"]==counts["short_samples"]+counts["singleton_samples"]+counts["emitted_samples"]
    assert counts["samples"]==sum(counts[k] for k in ("frame","original","band","supported"))
    assert len(audit["runs"])==2
    isolated=np.ones((3,3),dtype=bool); isolated[1,1]=False
    rows=np.asarray([[0.,1.,2.,1.]])
    split=study.split_at_excluded_mask(rows,isolated)
    audit=study.audit_valid_support(rows,isolated,isolated,split,budget=AllocationBudget(1_000_000))
    # Three half-pixel positions round to the one valid center: a short run.
    assert audit["counts"]["short_runs"]==1
    rows=np.asarray([[1.,0.,1.,0.]])
    mask=np.zeros((1,3),dtype=bool)
    split=study.split_at_excluded_mask(rows,mask)
    audit=study.audit_valid_support(rows,mask,mask,split,budget=AllocationBudget(1_000_000))
    assert audit["counts"]["short_runs"]==1


@pytest.mark.parametrize("kind",[study.AcquisitionCapError,study.AcquisitionIntegrityError])
def test_typed_terminal_bypasses_refusal_preserves_prior_member_and_blocks_truth(kind,tmp_path):
    plan=_attempt_plan()
    store,store_payload,validate_payloads=_member_store(tmp_path,plan)
    events=[]
    def trace(case,arm):
        if arm=="candidate":
            raise kind("analytic_failure",{"witness":17})
        return {"status":"ordinary_refusal"}
    with pytest.raises(kind,match="analytic_failure"):
        execute_one_shot_attempt(plan,("first","second"),trace_arm=trace,
            validate_pair=lambda *args: _passing_diagnostics(),
            seal_payload=lambda case,arm,value:value,store_payload=store_payload,
            validate_payloads=validate_payloads,persist=lambda state:events.append(state["execution_state"]),
            score_seals=lambda seals:pytest.fail("truth must remain unavailable"))
    assert plan["execution_state"]=="FAILED"
    assert len(plan["sealed_predictions"])==1
    assert plan["failure"]["terminal_evidence"]=={"witness":17}
    assert plan["failure"]["terminal_kind"]==kind.__name__
    assert plan["failure"]["post_hoc_truth_accessed"] is False
    assert json.loads(store.read_member(plan["sealed_predictions"][0]))["status"]=="ordinary_refusal"


def test_ordinary_acquisition_refusal_continues_all_24_arms_and_scores_after_barrier(tmp_path):
    plan=_attempt_plan()
    store,store_payload,validate_payloads=_member_store(tmp_path,plan)
    calls=[]
    def trace(case,arm):
        calls.append((case,arm))
        return {"trace_result":{"status":"ordinary_refusal","prediction":None}}
    def scorer(seals):
        assert len(seals)==24 and plan["execution_state"]=="PREDICTIONS_VALIDATED"
        assert plan["seal_validation"]["validated"] is True
        return {"evaluations":[],"summary":{"refusals":24}}
    execute_one_shot_attempt(plan,tuple(f"analytic-{i}" for i in range(12)),trace_arm=trace,
        validate_pair=lambda *args: _passing_diagnostics(),seal_payload=lambda case,arm,value:value,
        store_payload=store_payload,validate_payloads=validate_payloads,persist=lambda state:None,
        score_seals=scorer)
    assert len(calls)==24 and plan["execution_state"]=="COMPLETE"
    assert plan["summary"]["refusals"]==24


def test_valid_adapter_terminal_error_escapes_and_restores_patches_after_cap():
    from dataclasses import replace
    from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import FROZEN_CONFIG
    rgb=_analytic_rgb()
    with pytest.raises(study.AcquisitionCapError,match="edgel_cap") as exc:
        _valid_fake_trace(rgb,config=replace(FROZEN_CONFIG,max_edgels=0))
    assert exc.value.evidence["mask_calls"]==1
    assert exc.value.evidence["detector_calls"]==1
    assert exc.value.evidence["managed_peak_bytes"]>0


def test_payload_cap_is_checked_before_base64_encoding(monkeypatch):
    from dataclasses import replace
    from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import FROZEN_CONFIG
    rgb=_analytic_rgb()
    traced,support,_,_=_valid_fake_trace(rgb)
    monkeypatch.setattr(study.base64,"b64encode",lambda *args:pytest.fail("encoding occurred before cap"))
    with pytest.raises(study.AcquisitionCapError,match="raw_array_cap"):
        study.valid_acquisition_payload(_valid_bindings(rgb),traced,support,
                                        config=replace(FROZEN_CONFIG,max_raw_bytes=0))


def test_valid_store_short_write_has_typed_integrity_and_no_published_descriptor(tmp_path):
    path=tmp_path/"valid-short.gz"; path.touch()
    def short(target,blob):
        with target.open("ab") as stream:
            stream.write(blob[:-1]); stream.flush(); os.fsync(stream.fileno())
        return len(blob)-1
    store=study.ValidAcquisitionMemberStore(path,writer=short,total_bytes=1_000_000,reserve_bytes=0)
    plan={"sealed_predictions":[]}
    with pytest.raises(study.AcquisitionIntegrityError,match="store_failure") as exc:
        store.append(_canonical_payload({"status":"ordinary_refusal"}),plan,"analytic","candidate")
    assert plan["sealed_predictions"]==[]
    assert path.stat().st_size>0
    assert exc.value.evidence["sidecar_bytes"]==path.stat().st_size


def test_guarded_inherited_recorder_charges_before_raw_hex_allocation_and_keeps_bytes():
    from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import AllocationBudget
    calls=[]
    class Recorder:
        def __init__(self,policy):
            self.policy=policy; self.arrays={}; self.array_bytes=0
        def array(self,value,*,segments=False):
            calls.append("array")
            raw=value.tobytes()
            self.arrays["observed"]={"dtype":value.dtype.str,"shape":list(value.shape),"bytes_hex":raw.hex()}
            return "observed"
        def add(self,collection,event,cap):
            calls.append("add"); collection.append(event)
    policy=SimpleNamespace(max_score_events=2,max_stage_events=2,max_array_bytes=100_000)
    budget=AllocationBudget(1_000_000)
    recorder=study._valid_recorder_class(Recorder,budget)(policy)
    array=np.asarray([[2.,3.,4.,5.]],dtype=np.float32)
    assert recorder.array(array,segments=True)=="observed"
    assert recorder.arrays["observed"]["bytes_hex"]==array.tobytes().hex()
    assert budget.live>array.nbytes
    smaller=AllocationBudget(65536+512*1024+4096+100)
    guarded=study._valid_recorder_class(Recorder,smaller)(policy)
    calls.clear()
    with pytest.raises(study.AcquisitionCapError):
        guarded.array(array,segments=True)
    assert calls==[]


def test_singleton_supported_sample_is_preserved_in_run_accounting():
    from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import AllocationBudget
    rows=np.asarray([[0.49,1.,1.49,1.]])
    mask=np.ones((3,3),dtype=bool); mask[1,0]=False
    split=study.split_at_excluded_mask(rows,mask)
    audit=study.audit_valid_support(rows,mask,mask,split,budget=AllocationBudget(1_000_000))
    assert audit["counts"]["singleton_samples"]==1
    assert audit["counts"]["singleton_runs"]==1
    assert audit["counts"]["emitted_samples"]==0
    assert len(audit["runs"])==1 and int(audit["runs"][0]["reason"])==0


@pytest.mark.parametrize("field,value",[("identity","other-frame"),("arm","baseline"),
                                      ("rgb_sha256","wrong"),("rgb_shape",[1,1,3])])
def test_valid_payload_rejects_wrong_case_arm_rgb_binding(field,value):
    rgb=_analytic_rgb(); traced,support,_,_=_valid_fake_trace(rgb)
    bindings=_valid_bindings(rgb); bindings[field]=value
    with pytest.raises(study.AcquisitionIntegrityError,match="binding"):
        study.valid_acquisition_payload(bindings,traced,support)


def test_exact_valid_member_bytes_roundtrip_through_existing_coordinator_with_shared_budget(tmp_path):
    rgb=_analytic_rgb()
    plan=_attempt_plan()
    path=tmp_path/"analytic-members.gz"; path.touch()
    store=study.ValidAcquisitionMemberStore(path,total_bytes=5_000_000,reserve_bytes=0)
    supports={}
    payloads={}
    def trace(case,arm):
        traced,support,_,_=_valid_fake_trace(rgb,candidate=arm=="candidate")
        supports[arm]=support
        return {"trace_result":traced,"support":support}
    def seal(case,arm,value):
        encoded=study.valid_acquisition_payload(_valid_bindings(rgb,arm),value["trace_result"],value["support"])
        payloads[arm]=encoded
        return encoded
    def persist(state):
        for descriptor in state["sealed_predictions"]:
            assert store.read_member(descriptor)==payloads[descriptor["arm"]]
    def scored(seals):
        assert plan["execution_state"]=="PREDICTIONS_VALIDATED"
        assert len(seals)==2
        for descriptor in seals:
            decoded=json.loads(store.read_member(descriptor))
            assert decoded["identity"]==descriptor["identity"]
            assert decoded["arm"]==descriptor["arm"]
            assert decoded["trace_result"]["status"]=="ordinary_refusal"
        return {"evaluations":[],"summary":{"refusals":2}}
    execute_one_shot_attempt(plan,("analytic-frame",),trace_arm=trace,
        validate_pair=lambda *args:_passing_diagnostics(),seal_payload=seal,
        encode_payload=study.valid_sealed_bytes,
        store_payload=lambda case,arm,encoded,state:store.append(encoded,state,case,arm,budget=supports[arm]["budget"]),
        validate_payloads=store.validate_all,persist=persist,score_seals=scored)
    assert plan["execution_state"]=="COMPLETE"
    assert plan["summary"]["refusals"]==2


def test_inherited_integrity_refusal_is_terminal_and_unknown_exception_does_not_become_ordinary():
    with pytest.raises(study.AcquisitionIntegrityError,match="inherited_trace_refusal"):
        _valid_fake_trace(_analytic_rgb(),status="trace_refusal")


def test_adapter_restores_detector_mask_and_recorder_when_typed_exception_escapes():
    rgb=_analytic_rgb()
    def detector(grey,**kwargs):
        raise study.AcquisitionCapError("injected-detector-cap")
    def mask(height,width):
        return np.zeros((height,width),dtype=bool)
    calibration=SimpleNamespace(detect_segments=detector)
    gui=SimpleNamespace(create_gui_mask=mask)
    class Recorder:
        pass
    namespace=SimpleNamespace(TraceRecorder=Recorder)
    def trace(image,identity):
        gui.create_gui_mask(*image.shape[:2])
        calibration.detect_segments(np.zeros(image.shape[:2],dtype=np.uint8))
        pytest.fail("terminal was swallowed")
    with pytest.raises(study.AcquisitionCapError,match="injected-detector-cap"):
        study.trace_valid_acquisition(rgb,"analytic-frame",candidate=False,
            trace_rgb=trace,trace_module=namespace,calibration_module=calibration,gui_mask_module=gui)
    assert calibration.detect_segments is detector
    assert gui.create_gui_mask is mask
    assert namespace.TraceRecorder is Recorder


def test_json_parser_guard_counts_structural_objects_without_allocating_them():
    simple=b'"'+b'x'*4096+b'"'
    structured=b'[{"a":0},'*400+b'{"a":0}]'
    assert study._json_allocation_bound(structured)>study._json_allocation_bound(simple)
    with pytest.raises(study.AcquisitionCapError):
        study.valid_sealed_bytes({"unexpected":"unencoded"})



def _source_trace_data_classes():
    # Execute only the source's data/policy classes, with pure injected helpers.
    # Neither its module imports nor any camera function execute.
    import ast
    from dataclasses import asdict,dataclass
    path=Path(__file__).resolve().parents[1]/"scripts/evaluation/run_camera_mixed_depth.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    selected=[node for node in tree.body if isinstance(node,ast.ClassDef)
              and node.name in ("TracePolicy","TraceRecorder")]
    assert len(selected)==2
    def require(condition,reason):
        if not condition:
            raise study.AcquisitionIntegrityError(reason)
    namespace={"dataclass":dataclass,"asdict":asdict,"np":np,
               "_require":require,"_json":study._canonical,"_sha":study._sha256}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["TracePolicy"],namespace["TraceRecorder"]


def _source_shaped_score(index,segments_sha="a"*64,weights_sha="b"*64):
    # Full accepted camera score schema: all fields/winner values preserved.
    focal=240.+(index%19)*12.5
    yaw=float((index//19)%361-180)
    pitch=float((index//(19*361))%67-33)
    return {"site":"coarse","span":[291,292],"segments_sha256":segments_sha,
            "weights_sha256":weights_sha,
            "camera":{"focal":focal,"yaw":yaw,"pitch":pitch,"width":384,"height":384,
                      "threshold":0.09,"roll":0.,"principal_point":None},
            "winner":{"best_before":None if index==0 else [0.754326973137135,13.5349729831321,22.97561926841352,240.3962786234871]},
            "score":float((index*13)%10000)/10001.}


def test_realistic_76095_full_score_records_serialize_readback_and_validate_under_unchanged_caps(tmp_path):
    rgb=_analytic_rgb(); traced,support,_,_=_valid_fake_trace(rgb)
    budget=support["budget"]
    Policy,Recorder=_source_trace_data_classes()
    recorder=study._valid_recorder_class(Recorder,budget)(Policy())
    # Source-shaped raw/enhanced grayscale, GUI mask, segment and weight snapshots.
    for value in (np.full((384,384),80,np.uint8),np.full((384,384),81,np.uint8),
                  np.zeros((384,384),bool)):
        recorder.array(value)
    coordinates=np.tile(np.asarray([[2.,32.,98.,32.]],np.float32),(256,1))
    segment_sha=recorder.array(coordinates,segments=True)
    weight_sha=recorder.array(np.full(256,96.,np.float32))
    for i in range(76_095):
        event=_source_shaped_score(i,segment_sha,weight_sha)
        recorder.add(recorder.scores,dict(event,score=0.),recorder.policy.max_score_events)
        extra=len(study._canonical(event))-len(study._canonical(recorder.scores[-1]))
        assert recorder.trace_bytes+extra<=recorder.policy.max_trace_bytes
        recorder.trace_bytes+=extra
        recorder.scores[-1]=event
    packet=recorder.payload()
    traced["trace"]=packet
    # Realistic 12-row preregistry with >7 MiB embedded RGB and case bindings.
    pixels=bytes(384*384*3)
    frame_base64=base64.b64encode(pixels).decode("ascii")
    registry=[{"identity":f"fixture-{i}","family":f"family-{i%6}","stratum":"ordinary",
               "base_identity":None,"rgb_shape":[384,384,3],"rgb_base64":frame_base64,
               "rgb_sha256":hashlib.sha256(pixels).hexdigest(),
               "recipe_sha256":"c"*64,"recipe_json":"{}"} for i in range(12)]
    plan={"execution_state":"RUNNING","renderer_registry":registry,"sealed_predictions":[],
          "preregistration_sha256":"d"*64,"evaluations":None}
    encoded=study.valid_acquisition_payload(_valid_bindings(rgb),traced,support)
    path=tmp_path/"realistic-members.gz"; path.touch()
    store=study.ValidAcquisitionMemberStore(path,total_bytes=512_000_000,reserve_bytes=16*1024*1024)
    descriptor=store.append(encoded,plan,"analytic-frame","candidate",budget=budget)
    plan["sealed_predictions"].append(descriptor)
    assert store.read_member(descriptor)==encoded
    validation=store.validate_all(plan)
    assert validation["member_count"]==1
    payload=json.loads(encoded)
    witnessed=[]
    stream=payload["trace_result"]["trace"]["scores"]
    def compare(i,record):
        assert record==_source_shaped_score(i,segment_sha,weight_sha)
        if i in (0,1,76094):
            witnessed.append(i)
    replay=study.validate_score_stream(stream,study.AllocationBudget(128*1024*1024),
                                       expected_records=76095,on_record=compare)
    assert witnessed==[0,1,76094] and replay["records"]==76095
    assert stream["raw_bytes"]>30_000_000
    assert len(encoded)<=64*1024*1024 and descriptor["compressed_bytes"]<=16*1024*1024
    assert store.last_append_peak<=128*1024*1024 and store.last_read_peak<=128*1024*1024
    assert sum(d["compressed_bytes"] for d in plan["sealed_predictions"])+len(study._canonical(plan).encode())+16*1024*1024<=512_000_000
    print("REALISTIC",{"records":replay["records"],"score_json_bytes":stream["raw_bytes"],
        "score_gzip_bytes":stream["compressed_bytes"],"member_json_bytes":len(encoded),
        "member_gzip_bytes":descriptor["compressed_bytes"],"plan_json_bytes":len(study._canonical(plan).encode()),
        "append_managed_peak":store.last_append_peak,"read_managed_peak":store.last_read_peak})
    # Same realistic member plus 23 finite ordinary-refusal receipts exercise
    # the real coordinator's full ordered barrier, without a study runner.
    barrier_plan=_attempt_plan(); barrier_plan["renderer_registry"]=registry
    barrier_path=tmp_path/"realistic-barrier.gz"; barrier_path.touch()
    barrier_store=study.ValidAcquisitionMemberStore(barrier_path,total_bytes=512_000_000,
                                                   reserve_bytes=16*1024*1024)
    identities=("analytic-frame",)+tuple(f"refusal-{i}" for i in range(11))
    def seal(case,arm,value):
        if case=="analytic-frame" and arm=="candidate":
            return encoded
        return study._canonical({"identity":case,"arm":arm,"status":"ordinary_refusal"}).encode()
    def score(seals):
        assert len(seals)==24 and barrier_plan["execution_state"]=="PREDICTIONS_VALIDATED"
        assert barrier_plan["seal_validation"]["validated"] is True
        return {"evaluations":[],"summary":{"barrier":24}}
    execute_one_shot_attempt(barrier_plan,identities,trace_arm=lambda *args:{"status":"ordinary_refusal"},
        validate_pair=lambda *args:_passing_diagnostics(),seal_payload=seal,
        encode_payload=study.valid_sealed_bytes,
        store_payload=lambda case,arm,data,state:barrier_store.append(data,state,case,arm),
        validate_payloads=barrier_store.validate_all,persist=lambda state:None,score_seals=score)
    assert barrier_plan["summary"]["barrier"]==24
    # A hash-forged inner score descriptor is rejected before its outer member
    # descriptor is published and before any evaluator callback is entered.
    tampered=json.loads(encoded)
    tampered["trace_result"]["trace"]["scores"]["raw_sha256"]="0"*64
    bad=study._canonical(tampered).encode()
    failed_plan=_attempt_plan()
    failed_path=tmp_path/"realistic-tamper.gz"; failed_path.touch()
    failed_store=study.ValidAcquisitionMemberStore(failed_path,total_bytes=512_000_000,reserve_bytes=0)
    with pytest.raises(study.AcquisitionIntegrityError):
        execute_one_shot_attempt(failed_plan,identities,trace_arm=lambda *args:{"status":"ordinary_refusal"},
            validate_pair=lambda *args:_passing_diagnostics(),
            seal_payload=lambda case,arm,value:bad if arm=="candidate" else seal(case,arm,value),
            encode_payload=study.valid_sealed_bytes,
            store_payload=lambda case,arm,data,state:failed_store.append(data,state,case,arm),
            validate_payloads=failed_store.validate_all,persist=lambda state:None,
            score_seals=lambda seals:pytest.fail("truth entered after inner score tamper"))
    assert failed_plan["execution_state"]=="FAILED" and len(failed_plan["sealed_predictions"])==1
    assert failed_plan["failure"]["post_hoc_truth_accessed"] is False
    assert failed_path.stat().st_size>failed_plan["sealed_predictions"][0]["compressed_bytes"]
    assert "cv2" not in sys.modules


def test_lossless_stream_tamper_record_count_crc_and_member_array_structure_are_rejected():
    budget=study.AllocationBudget(128*1024*1024)
    stream=study.LosslessScoreStream(budget)
    stream.append(_source_shaped_score(0)); packet=stream.finish()
    for key,bad in (("record_count",2),("raw_sha256","0"*64),("compressed_sha256","0"*64)):
        changed=dict(packet); changed[key]=bad
        with pytest.raises(study.AcquisitionIntegrityError):
            study.validate_score_stream(changed,study.AllocationBudget(128*1024*1024))
    compressed=bytearray(base64.b64decode(packet["base64"])); compressed[-8]^=1
    changed=dict(packet,base64=base64.b64encode(compressed).decode(),
                 compressed_sha256=hashlib.sha256(compressed).hexdigest())
    with pytest.raises(study.AcquisitionIntegrityError):
        study.validate_score_stream(changed,study.AllocationBudget(128*1024*1024))
    rgb=_analytic_rgb(); traced,support,_,_=_valid_fake_trace(rgb)
    encoded=study.valid_acquisition_payload(_valid_bindings(rgb),traced,support)
    payload=json.loads(encoded); payload["arrays"]["detector_segments"]["shape"]=[4,1]
    with pytest.raises(study.AcquisitionIntegrityError,match="coordinate"):
        study._validate_valid_member_blob(study._canonical(payload).encode(),study.AllocationBudget(128*1024*1024))


@pytest.mark.parametrize("value",["quotes: \" \\ \n\t", "é\U0001f600",{"v":[1,2.25,None,True]}])
def test_plan_json_length_preflight_matches_canonical_encoding_exactly(value):
    assert study._exact_json_size(value)==len(study._canonical(value).encode())




def test_score_stream_record_cap_cannot_be_relaxed_before_allocating_workspace():
    budget=study.AllocationBudget(0)
    with pytest.raises(study.AcquisitionCapError,match="event_cap"):
        study.LosslessScoreStream(budget,max_records=120001)
    assert budget.live==0 and budget.peak==0
