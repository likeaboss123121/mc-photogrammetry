import numpy as np
import pytest

from mc_photogrammetry.reconstruction.camera import (
    CameraModel, camera_projection_matrix, camera_rotation_matrix, project_points,
)
from mc_photogrammetry.pipeline.contracts import CameraIntrinsics
from mc_photogrammetry.reconstruction.learned_geometry_adapter import (
    PreprocessingTransform, ScaleUncertainty, canonicalize_pose,
    canonical_roundtrip_roundoff_estimate, quaternion_rotation_error_bound, project_world, repository_camera,
    source_intrinsic, unproject_axial, validate_camera, validate_depth_confidence,
    vggt_transforms,
)


def test_half_pixel_crop_resize_pad_roundtrip():
    transform = PreprocessingTransform((1200, 800), (600, 280), (13, 17, 588, 261), (9, 21, 12, 8))
    pixels = np.array([[0, 0], [51.25, 128.5], [1199, 799]])
    expected = np.column_stack((pixels[:, 0] * .5 - .25 - 13 + 9,
                                pixels[:, 1] * .35 - .325 - 17 + 21))
    np.testing.assert_allclose(transform.to_model(pixels), expected)
    np.testing.assert_allclose(transform.to_source(expected), pixels, atol=1e-10)
    mask = transform.valid_source_mask()
    assert mask.sum() == 575 * 244
    assert not mask[:21].any() and not mask[:, :9].any()


@pytest.mark.parametrize("mode", ["crop", "pad"])
def test_non_square_and_batch_padding_roundtrip(mode):
    transforms = vggt_transforms([(1920, 1080), (800, 1600), (900, 901)], mode)
    assert len(set(t.model_size for t in transforms)) == 1
    for t in transforms:
        xy = np.array([[20.3, 40.5], [333.25, 200.25]])
        np.testing.assert_allclose(t.to_source(t.to_model(xy)), xy, atol=1e-10)
        k = np.array([[712., 0, 137.], [0, 803., 202.], [0, 0, 1]])
        np.testing.assert_allclose(t.affine @ source_intrinsic(k, t), k)
        assert t.valid_source_mask().any()


@pytest.mark.parametrize("yaw,pitch,roll", [(37, 12, 28), (-120, -79, -42), (17, 90, 33), (-22, -90, -71)])
def test_off_center_rolled_vertical_camera_depth_projection_parity(yaw, pitch, roll):
    k = np.array([[813., 0, 233.], [0, 921., 172.], [0, 0, 1]])
    old = CameraModel((2, -4, 7), yaw, pitch, eye_height=0, roll_degrees=roll,
                      intrinsics=CameraIntrinsics((640, 480), (813, 921), (233, 172)))
    r = np.diag([1, -1, 1]) @ camera_rotation_matrix(old)
    e = np.column_stack((r, -r @ np.array(old.position)))
    converted = repository_camera(e, k, (640, 480))
    pixels = np.array([[21, 71], [199.3, 123.2], [509, 375]])
    depths = np.array([3., 11., 7.])
    points = unproject_axial(pixels, depths, e, k)
    reprojection, z = project_world(points, e, k)
    np.testing.assert_allclose(reprojection, pixels, atol=1e-10)
    np.testing.assert_allclose(z, depths, atol=1e-10)
    np.testing.assert_allclose(np.array(project_points(points, converted, width=640, height=480)),
                               np.column_stack((pixels, depths)), atol=1e-9)
    np.testing.assert_allclose(camera_projection_matrix(converted, width=640, height=480), k @ e, atol=1e-9)
    assert converted.eye_height == 0


@pytest.mark.parametrize("depth,confidence", [(0, 1), (-1, 1), (np.nan, 1), (np.inf, 1), (1, -1), (1, np.nan), (1, np.inf)])
def test_invalid_observed_depth_confidence_refused(depth, confidence):
    with pytest.raises(ValueError):
        validate_depth_confidence(np.full((2, 3), depth), np.full((2, 3), confidence), np.ones((2, 3), bool))


def test_padding_is_never_source_evidence_and_confidence_is_not_probability():
    t = PreprocessingTransform((3, 2), (3, 2), (0, 0, 3, 2), (1, 1, 1, 1))
    mask = t.valid_source_mask()
    d, c = np.full(mask.shape, np.nan), np.full(mask.shape, np.nan)
    d[mask], c[mask] = 1, 7.5
    assert np.array_equal(validate_depth_confidence(d, c, mask), mask)


def test_unknown_metric_and_block_scale_remain_distinct_from_aliases():
    assert ScaleUncertainty().to_dict()["blocks_per_model_unit"] is None
    aliases = ScaleUncertainty(block_scale_alternatives=(.5, 1, 2))
    assert aliases.blocks_per_model_unit is None and aliases.metric_units_per_model_unit is None
    with pytest.raises(ValueError):
        ScaleUncertainty(blocks_per_model_unit=1, block_scale_alternatives=(.5, 1, 2))
    with pytest.raises(ValueError):
        ScaleUncertainty(blocks_per_model_unit=np.nan)


def test_mirrored_camera_is_refused():
    with pytest.raises(ValueError, match="proper rotation"):
        validate_camera(np.column_stack((np.diag([1, -1, 1]), np.zeros(3))), np.eye(3))


def test_source_reprojection_matches_model_coordinate_mapping():
    t = PreprocessingTransform((1920, 1080), (518, 294), (0, 0, 518, 294), (0, 14, 0, 0))
    e = np.column_stack((np.eye(3), np.array([.3, -.7, 1.])))
    k = np.array([[421., 0, 197.], [0, 408., 139.], [0, 0, 1]])
    xy = np.array([[15., 32.], [499., 281.]])
    world = unproject_axial(xy, np.array([4., 6.]), e, k)
    source, _ = project_world(world, e, source_intrinsic(k, t))
    np.testing.assert_allclose(source, t.to_source(xy), atol=1e-10)


def synthetic_pose(yaw=37, pitch=12, roll=28, translation=(.3, -.7, 1)):
    camera = CameraModel((0, 0, 0), yaw, pitch, eye_height=0, roll_degrees=roll)
    r = np.diag([1, -1, 1]) @ camera_rotation_matrix(camera)
    raw = np.column_stack((r, translation)).astype(np.float32)
    k = np.array([[813., 0, 233.], [0, 921., 172.], [0, 0, 1]], dtype=np.float32)
    return raw, k


def test_analytic_float32_counterexample_is_not_a_canonical_algebra_failure():
    delta = 2. ** -23
    raw = np.column_stack((np.diag([1 + delta, 1, 1]), np.zeros(3))).astype(np.float32)
    original = raw.copy()
    k = np.array([[1000., 0, 10.], [0, 1000., 17.], [0, 0, 1]], dtype=np.float32)
    # Old transpose-based contract: displacement is analytic, not helper-derived.
    old_error = 500 * ((1 + delta) ** 2 - 1)
    assert old_error > 1e-5
    with pytest.raises(ValueError, match="explicit canonicalize"):
        validate_camera(raw, k)
    pose = canonicalize_pose(raw, k)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(pose.raw_extrinsic, original)
    assert not pose.raw_extrinsic.flags.writeable and not pose.extrinsic.flags.writeable
    np.testing.assert_allclose(pose.extrinsic[:, :3], np.eye(3), atol=1e-15)
    p = np.array([[510., 17.]])
    world = unproject_axial(p, np.array([4.]), pose.extrinsic, pose.intrinsic)
    projected, _ = project_world(world, pose.extrinsic, pose.intrinsic)
    np.testing.assert_allclose(projected, p, atol=1e-10)
    assert pose.diagnostics["correction_frobenius"] < pose.diagnostics["correction_bound_frobenius"]


@pytest.mark.parametrize("pitch", [89.999999, -89.999999, 89.999, -89.999, 90, -90])
def test_float32_near_vertical_rolled_pose_uses_one_canonical_rotation(pitch):
    raw, k = synthetic_pose(pitch=pitch, roll=73, translation=(.2, 3., -.8))
    pose = canonicalize_pose(raw, k)
    camera = repository_camera(pose.extrinsic, pose.intrinsic, (640, 480))
    pixels = np.array([[12, 29], [199.5, 122.25], [590, 443]])
    depth = np.array([3., 11., 7.])
    world = unproject_axial(pixels, depth, pose.extrinsic, pose.intrinsic)
    projected, axial = project_world(world, pose.extrinsic, pose.intrinsic)
    np.testing.assert_allclose(projected, pixels, atol=1e-9, rtol=0)
    np.testing.assert_allclose(axial, depth, atol=1e-12, rtol=0)
    repo = np.asarray(project_points(world, camera, width=640, height=480))
    np.testing.assert_allclose(repo, np.column_stack((pixels, depth)), atol=1e-9, rtol=0)
    estimate = canonical_roundtrip_roundoff_estimate(pixels, depth, pose)
    assert np.isfinite(estimate).all() and (estimate >= 0).all()


@pytest.mark.parametrize("drift", [1e-4, .01, .3])
def test_materially_nonrigid_rotations_cannot_be_normalized_away(drift):
    raw, k = synthetic_pose()
    raw[:, :3] *= 1 + drift
    with pytest.raises(ValueError, match="over-correction"):
        canonicalize_pose(raw, k)


def test_one_ulp_perturbation_accepted_but_float64_drift_has_no_float32_budget():
    raw, k = synthetic_pose()
    raw[0, 0] = np.nextafter(raw[0, 0], np.float32(np.inf))
    pose = canonicalize_pose(raw, k)
    assert np.linalg.norm(pose.extrinsic[:, :3] @ pose.extrinsic[:, :3].T - np.eye(3)) < 1e-14
    with pytest.raises(ValueError, match="over-correction"):
        canonicalize_pose(raw.astype(np.float64), k)


@pytest.mark.parametrize("scale", [.01, 1., 1000.])
def test_float32_quaternion_arithmetic_stays_within_derived_error_envelope(scale):
    # Pinned upstream arithmetic, independently compared with Rodrigues in float64.
    q = (np.array([.3, -.4, .7, .5]) * scale).astype(np.float32)
    i, j, k, w = q
    two = np.float32(2) / np.sum(q * q, dtype=np.float32)
    actual = np.array([[1 - two * (j*j + k*k), two * (i*j - k*w), two * (i*k + j*w)],
                       [two * (i*j + k*w), 1 - two * (i*i + k*k), two * (j*k - i*w)],
                       [two * (i*k - j*w), two * (j*k + i*w), 1 - two * (i*i + j*j)]], dtype=np.float32)
    vector = q[:3].astype(np.float64)
    axis = vector / np.linalg.norm(vector)
    theta = 2 * np.arctan2(np.linalg.norm(vector), float(q[3]))
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    expected = np.eye(3) + np.sin(theta)*skew + (1-np.cos(theta))*(skew @ skew)
    bound = quaternion_rotation_error_bound(np.finfo(np.float32).eps / 2)
    assert np.linalg.norm(actual.astype(np.float64) - expected) <= bound
    canonicalize_pose(np.column_stack((actual, np.zeros(3, dtype=np.float32))), np.eye(3, dtype=np.float32))


def load_numeric_runner():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/evaluation/run_learned_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("synthetic_vggt_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports definitions only; never invokes main/model
    return module


def synthetic_artifact_case(translation=(.3, -.7, 1), forced_projector=None, invalid_depth=False,
                            nonrigid=False, rotation=None):
    import copy
    import json
    from pathlib import Path
    runner = load_numeric_runner()
    raw, k = synthetic_pose(pitch=89.999, translation=translation)
    if nonrigid:
        raw[:, :3] *= np.float32(1.0001)
    if rotation is not None:
        raw = np.column_stack((rotation, np.zeros(3))).astype(np.float64)
    e, intrinsics = np.stack([raw, raw]), np.stack([k, k])
    depth = np.full((2, 3, 5), 7., dtype=np.float32)
    if invalid_depth:
        depth[0, 1, 2] = np.nan
    confidence = np.ones_like(depth)
    transforms = [PreprocessingTransform((5, 3), (5, 3), (0, 0, 5, 3))] * 2
    state = {"status": "STARTED", "frames": [{"frame_id": "synthetic-a"}, {"frame_id": "synthetic-b"}]}
    artifacts, receipts = {}, []

    def writer(path, **arrays):
        artifacts[path.name] = {key: np.asarray(value).copy() for key, value in arrays.items()}

    def receipt_writer(path, value):
        json.dumps(value, allow_nan=False)  # enforce real JSON contract
        receipts.append(copy.deepcopy(value))

    try:
        runner.process_geometry_predictions(Path("synthetic-memory-only"), state, e, intrinsics,
                                            depth, confidence, transforms, writer, receipt_writer, forced_projector)
    except ValueError:
        pass
    return state, artifacts, receipts


def test_synthetic_success_has_separate_raw_canonical_and_delta_diagnostics():
    state, artifacts, receipts = synthetic_artifact_case()
    assert state["status"] == "GEOMETRY_VALIDATED"
    assert len(artifacts) == 6
    for frame in state["frames"]:
        assert frame["canonical_roundtrip"]["maximum_pixels"] < 1e-5
        assert frame["canonical_repository_parity"]["maximum_pixels"] < 1e-5
        assert frame["raw_matrix_inverse_roundtrip"]["maximum_pixels"] < 1e-5
        assert frame["roundoff_estimate_policy"]["certified"] is False
        assert frame["roundoff_estimate_policy"]["used_for_acceptance"] is False
    assert not any(r["status"] == "COMPLETE" for r in receipts)
    assert receipts[0]["stage"] == "raw_prediction_preservation"


def test_forced_parity_failure_keeps_all_raw_frames_exact_residual_and_blocked_receipt():
    def projector(points, camera, **kwargs):
        values = np.asarray(project_points(points, camera, **kwargs))
        values[:, 0] += .25
        return values

    state, artifacts, receipts = synthetic_artifact_case(forced_projector=projector)
    assert state["status"] == "BLOCKED" and state["stage"] == "canonical_repository_parity"
    assert "frame-00-raw.npz" in artifacts and "frame-01-raw.npz" in artifacts
    assert artifacts["frame-00-raw.npz"]["extrinsic"].dtype == np.float32
    assert "frame-00-canonical.npz" in artifacts and "frame-00-diagnostics.npz" in artifacts
    assert state["frames"][0]["canonical_repository_parity"]["maximum_pixels"] == pytest.approx(.25, abs=1e-10)
    assert "argmax_model_pixel" in state["frames"][0]["canonical_repository_parity"]
    assert receipts[-1]["status"] == "BLOCKED" and receipts[-1]["failure_frame_id"] == "synthetic-a"


def test_invalid_raw_output_is_preserved_before_refusal():
    state, artifacts, receipts = synthetic_artifact_case(invalid_depth=True)
    assert state["status"] == "BLOCKED" and state["stage"] == "raw_finite_validation"
    assert len(artifacts) == 2 and np.isnan(artifacts["frame-00-raw.npz"]["depth"][1, 2])
    assert receipts[-1]["status"] == "BLOCKED"


def test_over_correction_refusal_preserves_the_correction_and_bound():
    state, artifacts, receipts = synthetic_artifact_case(nonrigid=True)
    assert state["status"] == "BLOCKED" and state["stage"] == "rotation_canonicalization"
    diagnostic = receipts[-1]["frames"][0]["rotation_policy"]
    assert diagnostic["correction_frobenius"] > diagnostic["correction_bound_frobenius"]
    assert diagnostic["orthogonality_frobenius"] > 0
    assert "frame-00-raw.npz" in artifacts and "frame-01-raw.npz" in artifacts


def test_large_translation_depth_conditioning_is_explicitly_refused():
    state, artifacts, receipts = synthetic_artifact_case(translation=(1e14, -2e14, 3e14))
    assert state["status"] == "BLOCKED"
    assert state["stage"] == "raw_to_canonical_projection"
    assert "nonfinite/nonpositive" in state["detail"]
    assert "frame-00-raw.npz" in artifacts and "frame-00-canonical.npz" in artifacts
    assert receipts[-1]["status"] == "BLOCKED"


def test_moderate_large_translation_uses_measured_gates_not_estimate_certificate():
    state, artifacts, receipts = synthetic_artifact_case(translation=(1e5, -2e5, 3e5))
    assert state["status"] == "GEOMETRY_VALIDATED"
    assert state["frames"][0]["roundoff_estimate_policy"]["used_for_acceptance"] is False
    assert state["frames"][0]["raw_to_canonical_projection_delta"]["maximum_pixels"] > 1e-5
    # Input normalization delta is explicitly distinct from algebra parity.
    assert state["frames"][0]["canonical_repository_parity"]["maximum_pixels"] < 1e-5


def test_json_receipt_is_published_only_by_atomic_replace():
    import json
    runner = load_numeric_runner()
    events, stored = [], {}

    class MemoryPath:
        suffix = ".json"
        def __init__(self, name):
            self.name = name
        def with_suffix(self, suffix):
            return MemoryPath("run" + suffix)
        def write_text(self, data, encoding):
            events.append(("write", self.name))
            stored[self.name] = data
        def replace(self, target):
            events.append(("replace", self.name, target.name))
            stored[target.name] = stored.pop(self.name)

    runner.write_json(MemoryPath("run.json"), {"status": "BLOCKED", "stage": "forced"})
    assert events == [("write", "run.json.tmp"), ("replace", "run.json.tmp", "run.json")]
    assert json.loads(stored["run.json"]) == {"status": "BLOCKED", "stage": "forced"}


def test_raw_array_artifact_atomic_serialization_keeps_dtype_and_invalid_values():
    import io
    runner = load_numeric_runner()
    events, stored = [], {}

    class MemoryBuffer(io.BytesIO):
        def close(self):
            stored["frame.npz.tmp"] = self.getvalue()
            super().close()

    class MemoryPath:
        suffix = ".npz"
        def __init__(self, name):
            self.name = name
        def with_suffix(self, suffix):
            return MemoryPath("frame" + suffix)
        def open(self, mode):
            assert mode == "wb"
            events.append(("write", self.name))
            return MemoryBuffer()
        def replace(self, target):
            events.append(("replace", self.name, target.name))
            stored[target.name] = stored.pop(self.name)

    runner.write_arrays(MemoryPath("frame.npz"), depth=np.array([1., np.nan], dtype=np.float32))
    assert events == [("write", "frame.npz.tmp"), ("replace", "frame.npz.tmp", "frame.npz")]
    with np.load(io.BytesIO(stored["frame.npz"]), allow_pickle=False) as artifact:
        assert artifact["depth"].dtype == np.float32 and np.isnan(artifact["depth"][1])


@pytest.mark.parametrize("rotation", [np.zeros((3, 3)), np.diag([1., 1., 0.]),
                                      np.eye(3) * 1e308])
def test_singular_and_overflow_diagnostics_preserve_all_raw_and_finite_terminal_json(rotation):
    import json
    state, artifacts, receipts = synthetic_artifact_case(rotation=rotation)
    assert state["status"] == "BLOCKED" and receipts[-1]["status"] == "BLOCKED"
    assert len(artifacts) == 2
    for index in range(2):
        np.testing.assert_array_equal(artifacts[f"frame-{index:02d}-raw.npz"]["extrinsic"][:, :3], rotation)
        diagnostic = state["frames"][index]["raw_rotation"]
        assert diagnostic["unavailable"]
        assert any(diagnostic[name] is None for name in diagnostic["unavailable"])
    json.dumps(receipts, allow_nan=False)


def test_failed_svd_is_optional_until_all_raw_evidence_is_preserved(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("injected SVD unavailable")
    monkeypatch.setattr(np.linalg, "svd", fail)
    state, artifacts, receipts = synthetic_artifact_case()
    assert len(artifacts) == 2 and receipts[-1]["status"] == "BLOCKED"
    assert "SVD" in state["detail"]
    assert "SVD unavailable" in state["frames"][0]["rotation_policy"]["canonicalization_unavailable"]


@pytest.mark.parametrize("principal,translation,depth", [
    ((1e8, -2e8), (0., 0., 0.), 7.),
    ((233., 172.), (1e5, -2e5, 3e5), 7.),
    ((-4e6, 7e6), (.3, -.7, 1.), .001)])
def test_independent_cancellation_crop_upscale_and_translation_depth_controls(principal, translation, depth):
    # Identity rotation isolates homogeneous cancellation from Euler/polar arithmetic.
    from decimal import Decimal, localcontext
    transform = PreprocessingTransform((29, 17), (87, 85), (41, 33, 46, 36))
    e = np.column_stack((np.eye(3), translation))
    k = np.array([[813., 0, principal[0]], [0, 921., principal[1]], [0, 0, 1.]])
    pose = canonicalize_pose(e, k)
    pixels = np.array([[0., 0.], [2., 1.], [4., 2.]])
    depths = np.full(3, depth)
    world = unproject_axial(pixels, depths, e, k)
    projected, _ = project_world(world, e, source_intrinsic(k, transform))
    # Decimal reference derives inverse resize/crop directly, without adapter mapping.
    with localcontext() as context:
        context.prec = 60
        expected = np.array([[float((Decimal(str(x)) + 40) / 3),
                              float((Decimal(str(y)) + 31) / 5)] for x, y in pixels])
    errors = np.linalg.norm(projected - expected, axis=1)
    assert np.isfinite(errors).all() and errors.max() < 1e-5
    repo = np.asarray(project_points(world, repository_camera(e, source_intrinsic(k, transform),
                                       transform.source_size), width=29, height=17))
    assert np.linalg.norm(repo[:, :2] - expected, axis=1).max() < 1e-5
    assert "NOT a certified" in canonical_roundtrip_roundoff_estimate.__doc__


def test_estimate_understatement_cannot_bypass_measured_parity_gate(monkeypatch):
    import mc_photogrammetry.reconstruction.learned_geometry_adapter as adapter
    monkeypatch.setattr(adapter, "canonical_roundtrip_roundoff_estimate", lambda pixels, depth, pose: np.zeros(len(pixels)))
    def corrupt(points, camera, **kwargs):
        result = np.asarray(project_points(points, camera, **kwargs))
        result[:, 0] += 2e-5
        return result
    state, _, receipts = synthetic_artifact_case(forced_projector=corrupt)
    assert receipts[-1]["status"] == "BLOCKED" and state["stage"] == "canonical_repository_parity"
    assert state["frames"][0]["canonical_roundoff_estimate"]["maximum_pixels"] == 0
    assert state["frames"][0]["roundoff_estimate_policy"]["used_for_acceptance"] is False


def test_receipt_competing_writers_are_serialized_and_resource_failure_is_sticky():
    import copy
    import threading
    from pathlib import Path
    runner = load_numeric_runner()
    entered, release, competing = threading.Event(), threading.Event(), threading.Event()
    writes, failures = [], []
    def writer(path, value):
        if value["status"] == "STARTED":
            entered.set()
            assert release.wait(2)
        writes.append(copy.deepcopy(value))
    publisher = runner.ReceiptPublisher(Path("memory-only/run.json"), writer)
    def first():
        try:
            publisher.publish({"status": "STARTED", "stage": "forward"})
        except Exception as exc:
            failures.append(exc)
    def resource():
        competing.set()
        publisher.resource_failure("CPU_COMPUTE_CAP", {"cpu_seconds": 3600})
    threads = [threading.Thread(target=first), threading.Thread(target=resource)]
    try:
        threads[0].start()
        assert entered.wait(2)
        threads[1].start()
        assert competing.wait(2)
        assert writes == []  # second writer cannot touch the shared temporary path
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(2)
    assert not failures and not any(thread.is_alive() for thread in threads)
    assert [value["status"] for value in writes] == ["STARTED", "BLOCKED"]
    publisher.publish({"status": "GEOMETRY_VALIDATED"})
    publisher.publish({"status": "COMPLETE"})
    publisher.publish({"status": "BLOCKED", "failure": "stale generic exception"})
    assert len(writes) == 2 and publisher.snapshot["failure"] == "CPU_COMPUTE_CAP"


def test_resource_termination_survives_receipt_io_failure():
    from pathlib import Path
    runner = load_numeric_runner()
    def fail(path, value):
        raise OSError("injected receipt failure")
    publisher = runner.ReceiptPublisher(Path("memory-only/run.json"), fail)
    exits = []
    with pytest.raises(OSError, match="injected"):
        runner.stop_for_resource_failure(publisher, "RSS_CAP", {}, exits.append)
    assert exits == [3] and publisher.snapshot["status"] == "BLOCKED"
    publisher.publish({"status": "COMPLETE"})
    assert publisher.snapshot["failure"] == "RSS_CAP"


@pytest.mark.parametrize("failure", ["gallery", "validation", None])
def test_outer_completion_requires_all_galleries_and_validated_artifacts(failure):
    import copy
    from pathlib import Path
    runner = load_numeric_runner()
    state, artifacts, receipts = synthetic_artifact_case()
    writes = []
    publisher = runner.ReceiptPublisher(Path("memory-only/run.json"),
                                       lambda path, value: writes.append(copy.deepcopy(value)))
    def build(index, frame):
        assert not any(value["status"] == "COMPLETE" for value in writes)
        if failure == "gallery" and index == 1:
            raise OSError("injected gallery failure")
        artifacts[f"frame-{index:02d}-reprojection.png"] = "synthetic-gallery"
    checked = []
    def validate(path, kind):
        assert not any(value["status"] == "COMPLETE" for value in writes)
        assert path.name in artifacts
        checked.append((path.name, kind))
        if failure == "validation" and kind == "gallery":
            raise ValueError("injected invalid gallery")
    if failure:
        with pytest.raises((ValueError, OSError), match="injected"):
            runner.complete_required_artifacts(Path("memory-only"), state, publisher, build, validate)
        assert writes[-1]["status"] == "BLOCKED"
        assert not any(value["status"] == "COMPLETE" for value in receipts + writes)
    else:
        runner.complete_required_artifacts(Path("memory-only"), state, publisher, build, validate)
        assert len(checked) == 8 and writes[-1]["status"] == "COMPLETE"


@pytest.mark.parametrize("invalid", [False, True])
def test_actual_artifact_validator_rejects_nonfinite_diagnostics_in_memory(invalid):
    import io
    from types import SimpleNamespace
    runner = load_numeric_runner()
    class MemoryArtifact(io.BytesIO):
        name = "synthetic-diagnostics.npz"
        def is_file(self):
            return True
        def stat(self):
            return SimpleNamespace(st_size=len(self.getvalue()))
    path = MemoryArtifact()
    np.savez_compressed(path, canonical_roundtrip_errors=np.array([np.nan if invalid else 0.]),
                        repository_parity_errors=np.array([0.]))
    path.seek(0)
    if invalid:
        with pytest.raises(ValueError, match="invalid diagnostics artifact"):
            runner.validate_output_artifact(path, "diagnostics")
    else:
        runner.validate_output_artifact(path, "diagnostics")


def test_outer_completion_cannot_override_prior_resource_refusal():
    import copy
    from pathlib import Path
    runner = load_numeric_runner()
    state, _, _ = synthetic_artifact_case()
    writes = []
    publisher = runner.ReceiptPublisher(Path("memory-only/run.json"),
                                       lambda path, value: writes.append(copy.deepcopy(value)))
    publisher.publish(state)
    publisher.resource_failure("WALL_TIME_CAP", {"wall_seconds": 1200})
    runner.complete_required_artifacts(Path("memory-only"), state, publisher,
                                      lambda index, frame: None, lambda path, kind: None)
    assert state["status"] == "BLOCKED" and state["failure"] == "WALL_TIME_CAP"
    assert not any(value["status"] == "COMPLETE" for value in writes)


@pytest.mark.parametrize("resource", [False, True], ids=["numeric-blocked", "resource-blocked"])
def test_terminal_final_telemetry_preserves_original_failure_and_priority(resource):
    import copy
    import json
    from pathlib import Path
    runner = load_numeric_runner()
    writes = []
    def writer(path, value):
        json.dumps(value, allow_nan=False)
        writes.append(copy.deepcopy(value))
    publisher = runner.ReceiptPublisher(Path("memory-only/run.json"), writer)
    state = {"status": "BLOCKED", "failure": "RSS_CAP" if resource else "ValueError",
             "detail": "original failure", "stage": "canonical_repository_parity",
             "failure_frame_id": "synthetic-a", "wall_seconds": 0, "cpu_seconds": 0,
             "peak_rss_bytes": 0, "new_disk_bytes": 0}
    publisher.publish(state, resource_failure=resource)
    priority = publisher.terminal_priority
    original = copy.deepcopy(state)
    metrics = {"wall_seconds": 17.5, "cpu_seconds": 9.25,
               "peak_rss_bytes": 1024, "new_disk_bytes": 512,
               "status": "COMPLETE", "failure": "replacement", "detail": "stale detail",
               "stage": "complete", "failure_frame_id": "synthetic-b", "arbitrary": "ignored"}
    state.update(publisher.finalize_telemetry(metrics))
    for key in ("wall_seconds", "cpu_seconds", "peak_rss_bytes", "new_disk_bytes"):
        assert writes[-1][key] == state[key] == metrics[key] > 0
    for key in ("status", "failure", "detail", "stage", "failure_frame_id"):
        assert writes[-1][key] == state[key] == original[key]
    assert "arbitrary" not in writes[-1] and publisher.terminal_priority == priority
    publisher.publish({"status": "COMPLETE", "stage": "stale completion"})
    publisher.publish({"status": "STARTED", "stage": "stale progress"})
    assert len(writes) == 2 and publisher.snapshot == state
    with pytest.raises(ValueError, match="finite nonnegative"):
        publisher.finalize_telemetry({"wall_seconds": float("nan")})
    assert len(writes) == 2 and publisher.snapshot == state


def test_fresh_binding_accepts_only_packet_paths_and_pins():
    runner = load_numeric_runner()
    capture = runner.ROOT / "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures"
    images = tuple(capture / f"view-canal-town-architectural-component-bbox-{index:02d}.png" for index in range(4))
    output = runner.ROOT / "exports/diagnostics/vggt-c1-fresh-canal-town-framed-2026-10-07"
    assert runner.bind_inputs(images, output) == images
    assert runner.INPUT_SHA256 == (
        "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c",
        "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696",
        "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95",
        "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9")


def test_fresh_receipt_explicitly_records_developer_only_train_role_without_running_main():
    import ast
    import inspect
    runner = load_numeric_runner()
    roles = [ast.literal_eval(value) for node in ast.walk(ast.parse(inspect.getsource(runner.main)))
             if isinstance(node, ast.Dict) for key, value in zip(node.keys, node.values)
             if isinstance(key, ast.Constant) and key.value == "input_role"]
    assert roles == [{"run": "developer_runtime_validation", "split": "train", "advanced_only": True,
                      "normal_mode_validation": False, "accuracy_claim": False}]


@pytest.mark.parametrize("hostile", ["original", "reordered", "duplicate", "other-four", "sidecar", "short", "other-root"])
def test_fresh_binding_rejects_hostile_image_sets(hostile):
    runner = load_numeric_runner()
    images = [runner.DATA / name for name in runner.NAMES]
    if hostile == "original":
        capture = runner.ROOT / "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures"
        images = [capture / f"view-train-station-spawn-208-96-orbit-a-{index:02d}.png" for index in range(4)]
    elif hostile == "reordered":
        images.reverse()
    elif hostile == "duplicate":
        images[3] = images[0]
    elif hostile == "other-four":
        images = [runner.DATA / f"view-canal-town-architectural-component-bbox-{index:02d}.png" for index in range(4, 8)]
    elif hostile == "sidecar":
        images[0] = runner.DATA / "ground_truth.json"
    elif hostile == "short":
        images.pop()
    else:
        images[0] = runner.ROOT / "substitute" / runner.NAMES[0]
    with pytest.raises(ValueError, match="four exact ordered"):
        runner.bind_inputs(images, runner.OUTPUT)


@pytest.mark.parametrize("hostile", ["consumed", "child", "other-root"])
def test_fresh_binding_rejects_hostile_output_paths(hostile):
    runner = load_numeric_runner()
    output = {"consumed": runner.ROOT / "exports/diagnostics/vggt-c1-2026-10-07",
              "child": runner.OUTPUT / "child", "other-root": runner.ROOT / "substitute"}[hostile]
    with pytest.raises(ValueError, match="fresh output"):
        runner.bind_inputs([runner.DATA / name for name in runner.NAMES], output)


@pytest.mark.parametrize("redirected", ["input", "output"])
def test_fresh_binding_refuses_redirected_exact_paths_without_creating_links(monkeypatch, redirected):
    from pathlib import Path
    runner = load_numeric_runner()
    images = tuple(runner.DATA / name for name in runner.NAMES)
    target = images[0] if redirected == "input" else runner.OUTPUT
    resolve = Path.resolve
    def redirected_resolve(path, *args, **kwargs):
        if path == target:
            return runner.ROOT / "exports/diagnostics/vggt-c1-2026-10-07"
        return resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    with pytest.raises(ValueError, match="redirected"):
        runner.bind_inputs(images, runner.OUTPUT)
