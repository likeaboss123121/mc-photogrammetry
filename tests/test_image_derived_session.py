"""Tests for the typed image-derived session producer/deserializer (M4 seam).

Most tests monkeypatch ``pipeline.image_only.solve_image_cameras`` rather than
running the real M3 calibrator. The all-abstain regression uses the real seam
with an independent-estimator stub. Delegation, immutability, digest binding,
and the forbidden-claim/refusal contract are what this module owns.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline import image_derived_session as session_module
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    IMAGE_DERIVED_SESSION_SCHEMA,
    IMAGE_DERIVED_SESSION_VERSION,
    SESSION_CAMERA_UNRESOLVED,
    SESSION_DIGEST_MISMATCH,
    SESSION_FORBIDDEN_CLAIM,
    SESSION_FRAME_INVALID,
    SESSION_INPUT_INVALID,
    SESSION_RECORD_INVALID,
    SESSION_SCHEMA_UNSUPPORTED,
    SESSION_UNRESOLVED_FACT_MISSING,
    ImageDerivedFrame,
    ImageDerivedSession,
    ImageDerivedSessionError,
    build_image_derived_session,
    image_derived_session_from_record,
    image_derived_session_to_record,
    image_sha256,
)
from mc_photogrammetry.pipeline.image_only import LOCAL_FRAME_ORIGIN


_FRAME_DIGEST_KEYS = {"frame_id", "source_label", "image_shape", "image_sha256"}
_FORBIDDEN_PIXEL_KEYS = {"image", "pixels", "array", "pixel_data", "image_data"}


def _contains_pixel_payload(value) -> bool:
    """True when a record carries pixel bytes or an array rather than a digest."""
    if isinstance(value, dict):
        return any(
            key in _FORBIDDEN_PIXEL_KEYS or _contains_pixel_payload(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_pixel_payload(item) for item in value)
    return isinstance(value, np.ndarray)


def _estimate(image_size=(60, 40), focal=(100.0, 100.0), principal=(30.0, 20.0)):
    return CameraEstimate(
        intrinsics=CameraIntrinsics(tuple(image_size), tuple(focal), tuple(principal)),
        pose=CameraPose(
            position=None, yaw_degrees=14.0, pitch_degrees=17.0, roll_degrees=3.5),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _solution(frame_ids, **estimate_kwargs):
    cameras = {frame_id: _estimate(**estimate_kwargs) for frame_id in frame_ids}
    unresolved: list[str] = []
    for frame_id in frame_ids:
        unresolved.append(f"{frame_id}: position (vanishing points fix no translation)")
        unresolved.append(f"{frame_id}: scale (needs the lattice period)")
        unresolved.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return CameraSolution(cameras=cameras, unresolved=tuple(unresolved), scale_known=False)


def _rgb(shape=(40, 60)):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:, : shape[1] // 2] = (255, 0, 0)
    image[:, shape[1] // 2 :] = (0, 0, 255)
    return image


def _sealed(image: np.ndarray) -> np.ndarray:
    """A genuinely non-reopenable frame buffer for direct-construction tests."""
    return np.frombuffer(image.tobytes(order="C"), dtype=np.uint8).reshape(image.shape)


def _patched_solution(monkeypatch, frame_ids=None, **estimate_kwargs):
    """Install a recording fake solver; returns the seen mapping holder."""
    seen: dict[str, np.ndarray] = {}

    def fake_solve(frames):
        seen.update(frames)
        return _solution(list(frames) if frame_ids is None else frame_ids,
                         **estimate_kwargs)

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    return seen


def _reseal(record: dict) -> dict:
    payload = {key: value for key, value in record.items() if key != "session_sha256"}
    resealed = dict(record)
    resealed["session_sha256"] = session_module._digest(payload)
    return resealed


# --- Production: immutable copies, digests, delegation -----------------------

def test_the_producer_copies_frames_immutably_and_freezes_the_digest(monkeypatch):
    _patched_solution(monkeypatch)
    original = _rgb()

    session = build_image_derived_session({"frame.png": original})
    captured = session.frames[0]

    assert captured.image is not original
    assert captured.image.flags.writeable is False
    assert captured.image_shape == (40, 60, 3)
    assert captured.image_sha256 == image_sha256(captured.image)
    assert captured.image_sha256 == image_sha256(original)

    digest_before = session.session_sha256
    expected = original.copy()
    original[:] = 255
    assert np.array_equal(captured.image, expected)
    assert session.session_sha256 == digest_before


def test_rgb_and_gray_frames_are_both_admitted(monkeypatch):
    monkeypatch.setattr(
        image_only, "solve_image_cameras",
        lambda frames: _solution(list(frames), image_size=(30, 20), principal=(15.0, 10.0)))

    session = build_image_derived_session({
        "rgb.png": _rgb((20, 30)),
        "gray.png": np.zeros((20, 30), dtype=np.uint8),
    })

    shapes = {frame.frame_id: frame.image_shape for frame in session.frames}
    assert shapes == {"rgb.png": (20, 30, 3), "gray.png": (20, 30)}
    assert all(frame.image.flags.writeable is False for frame in session.frames)


def test_solving_receives_read_only_copies_in_frame_order(monkeypatch):
    seen = _patched_solution(monkeypatch)
    first, second = _rgb((20, 30)), np.full((20, 30), 7, dtype=np.uint8)

    build_image_derived_session({"first.png": first, "second.png": second})

    assert list(seen) == ["first.png", "second.png"]
    assert seen["first.png"].flags.writeable is False
    assert seen["second.png"].flags.writeable is False
    assert seen["first.png"] is not first
    assert seen["second.png"] is not second


def test_a_produced_frame_buffer_cannot_be_reopened(monkeypatch):
    _patched_solution(monkeypatch)

    session = build_image_derived_session({"frame.png": _rgb()})
    image = session.frames[0].image

    assert image.flags.writeable is False
    # The buffer is borrowed from immutable bytes, not owned by numpy.
    assert image.flags.owndata is False
    with pytest.raises(ValueError):
        image.setflags(write=True)
    assert image.flags.writeable is False
    with pytest.raises(ValueError):
        image[0, 0, 0] = 0
    assert session.frames[0].image_sha256 == image_sha256(image)


def test_a_delegated_solver_cannot_reopen_or_mutate_a_captured_frame(monkeypatch):
    observed: dict[str, object] = {}

    def hostile_solver(frames):
        frame = frames["frame.png"]
        reopened = False
        reopen_error: Exception | None = None
        write_error: Exception | None = None
        try:
            frame.setflags(write=True)
            reopened = True
        except ValueError as error:
            reopen_error = error
        try:
            frame[:] = 255
        except ValueError as error:
            write_error = error
        observed.update(
            reopened=reopened,
            reopen_error=reopen_error,
            write_error=write_error,
            writeable_after=frame.flags.writeable,
        )
        return _solution(list(frames))

    monkeypatch.setattr(image_only, "solve_image_cameras", hostile_solver)
    original = _rgb()

    session = build_image_derived_session({"frame.png": original})
    captured = session.frames[0]

    assert observed["reopened"] is False
    assert isinstance(observed["reopen_error"], ValueError)
    assert isinstance(observed["write_error"], ValueError)
    assert observed["writeable_after"] is False
    assert np.array_equal(captured.image, original)
    assert captured.image_sha256 == image_sha256(original)


def test_post_build_digest_and_session_integrity_remain_valid(monkeypatch):
    _patched_solution(monkeypatch)
    original = _rgb()

    session = build_image_derived_session({"frame.png": original})
    captured = session.frames[0]
    record = image_derived_session_to_record(session)
    payload = {key: value for key, value in record.items() if key != "session_sha256"}

    assert captured.image_sha256 == image_sha256(captured.image)
    assert record["session_sha256"] == session.session_sha256
    assert session.session_sha256 == session_module._digest(payload)

    rebuilt = image_derived_session_from_record(record)
    assert rebuilt.session_sha256 == session.session_sha256
    assert [frame.image_sha256 for frame in rebuilt.frames] == [
        frame.image_sha256 for frame in session.frames]
    assert rebuilt.has_pixels is False

    original[:] = 0
    assert np.array_equal(captured.image, _rgb())
    assert captured.image_sha256 == image_sha256(_rgb())
    assert session.session_sha256 == session_module._digest(payload)


def test_recovery_delegates_to_the_shared_seam_and_adapts_at_the_local_origin(monkeypatch):
    _patched_solution(monkeypatch)
    positions: list[tuple[float, float, float]] = []
    real_adapt = image_only.camera_model_from_estimate

    def spy(estimate, *, position):
        positions.append(position)
        return real_adapt(estimate, position=position)

    monkeypatch.setattr(image_only, "camera_model_from_estimate", spy)

    session = build_image_derived_session({"frame.png": _rgb()})

    assert positions, "camera_model_from_estimate was never consulted"
    assert set(positions) == {LOCAL_FRAME_ORIGIN}
    camera = session.local_cameras["frame.png"]
    assert camera.position == LOCAL_FRAME_ORIGIN
    assert camera.yaw_degrees == pytest.approx(14.0)
    assert camera.pitch_degrees == pytest.approx(17.0)
    assert camera.roll_degrees == pytest.approx(3.5)
    assert camera.intrinsics is session.cameras["frame.png"].intrinsics


def test_unresolved_scale_translation_and_yaw_quadrant_facts_are_preserved(monkeypatch):
    _patched_solution(monkeypatch)

    session = build_image_derived_session({"frame.png": _rgb()})

    assert session.scale_known is False
    assert session.cameras["frame.png"].pose.position is None
    assert session.unresolved == (
        "frame.png: position (vanishing points fix no translation)",
        "frame.png: scale (needs the lattice period)",
        "frame.png: yaw quadrant (recovered modulo 90 degrees)",
    )


# --- Producer refusal surface ------------------------------------------------

def test_a_frame_without_a_solved_camera_is_retained_as_a_typed_exclusion(monkeypatch):
    monkeypatch.setattr(
        image_only, "solve_image_cameras",
        lambda frames: _solution(["one.png"]))

    session = build_image_derived_session({"one.png": _rgb(), "two.png": _rgb()})
    assert session.version == 2 and [frame.frame_id for frame in session.frames] == ["one.png"]
    assert [frame.frame_id for frame in session.input_frames] == ["one.png", "two.png"]
    assert session.admissions[1].reason == "camera_estimate_unavailable"


def test_real_camera_seam_all_abstentions_preserve_inputs_and_typed_exclusions(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    frames = {"image-0000": np.full((20, 30), 7, dtype=np.uint8),
              "image-0001": np.full((20, 30), 11, dtype=np.uint8)}
    real_solve = image_only.solve_image_cameras
    seen = []

    def abstain(grey):
        seen.append(grey)
        return None

    def forbidden(*args, **kwargs):
        pytest.fail("all-abstain session used shared/GUI fitting or adapted an absent estimate")

    monkeypatch.setattr(calibration, "estimate_camera", abstain)
    monkeypatch.setattr(calibration, "estimate_session", forbidden)
    monkeypatch.setattr(calibration, "estimate_image_calibrations", forbidden)
    monkeypatch.setattr(calibration, "to_camera_solution", forbidden)

    session = build_image_derived_session(frames)

    assert image_only.solve_image_cameras is real_solve
    assert session.version == 2
    assert session.frames == ()
    assert dict(session.cameras) == {} and dict(session.local_cameras) == {}
    assert session.components == () and session.scale_known is False
    assert [frame.frame_id for frame in session.input_frames] == list(frames)
    assert len(seen) == len(frames)
    for grey, captured, original in zip(seen, session.input_frames, frames.values()):
        assert grey is captured.image
        assert np.array_equal(grey, original)
        assert captured.image_sha256 == image_sha256(original)
    assert [(outcome.frame_id, outcome.status, outcome.stage, outcome.reason)
            for outcome in session.admissions] == [
        (frame_id, "excluded", "camera_recovery", "camera_estimate_unavailable")
        for frame_id in frames
    ]
    assert session.unresolved == tuple(
        f"{frame_id}: camera estimate unavailable; position, scale and orientation unknown"
        for frame_id in frames
    )
    record = image_derived_session_to_record(session)
    assert record["active_frame_ids"] == []
    assert record["camera_pooling_caveat"] == session_module.CAMERA_INDEPENDENT_CAVEAT
    assert [frame["frame_id"] for frame in record["frames"]] == list(frames)
    assert image_derived_session_to_record(image_derived_session_from_record(record)) == record
    assert not _contains_pixel_payload(record)


def test_a_solution_that_claims_scale_is_refused(monkeypatch):
    def claiming(frames):
        solution = _solution(list(frames))
        return CameraSolution(
            cameras=solution.cameras, unresolved=solution.unresolved, scale_known=True)

    monkeypatch.setattr(image_only, "solve_image_cameras", claiming)

    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session({"frame.png": _rgb()})

    assert raised.value.reason == SESSION_FORBIDDEN_CLAIM


def test_an_estimate_without_a_usable_camera_is_refused(monkeypatch):
    def broken(frames):
        solution = _solution(list(frames))
        frame_id = next(iter(solution.cameras))
        estimate = solution.cameras[frame_id]
        return CameraSolution(
            cameras={frame_id: CameraEstimate(intrinsics=None, pose=estimate.pose)},
            unresolved=solution.unresolved, scale_known=False)

    monkeypatch.setattr(image_only, "solve_image_cameras", broken)

    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session({"frame.png": _rgb()})

    assert raised.value.reason == SESSION_CAMERA_UNRESOLVED


def test_an_empty_frame_mapping_is_refused():
    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session({})

    assert raised.value.reason == SESSION_INPUT_INVALID


def test_frames_must_be_an_ordered_mapping():
    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session([("frame.png", _rgb())])  # type: ignore[arg-type]

    assert raised.value.reason == SESSION_INPUT_INVALID


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 1), dtype=np.uint8),
        np.zeros((0, 10), dtype=np.uint8),
        "not an image",
    ],
)
def test_malformed_frames_are_refused(image):
    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session({"frame.png": image})

    assert raised.value.reason == SESSION_FRAME_INVALID


def test_empty_frame_ids_are_refused():
    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session({"": _rgb()})

    assert raised.value.reason == SESSION_INPUT_INVALID


def test_source_labels_only_name_known_frames_and_never_carry_geometry(monkeypatch):
    _patched_solution(monkeypatch, image_size=(30, 20), principal=(15.0, 10.0))

    session = build_image_derived_session(
        {"frame.png": _rgb((20, 30))},
        source_paths={"frame.png": "captures/frame.png"},
    )
    assert session.frames[0].source_label == "captures/frame.png"

    with pytest.raises(ImageDerivedSessionError) as raised:
        build_image_derived_session(
            {"frame.png": _rgb((20, 30))}, source_paths={"other.png": "x.png"})
    assert raised.value.reason == SESSION_INPUT_INVALID


def test_an_absent_label_stays_null_and_a_path_label_is_accepted(monkeypatch):
    from pathlib import Path

    _patched_solution(monkeypatch, image_size=(30, 20), principal=(15.0, 10.0))

    session = build_image_derived_session(
        {"a.png": _rgb((20, 30)), "b.png": _rgb((20, 30))},
        source_paths={"b.png": Path("captures") / "b.png"},
    )

    labels = {frame.frame_id: frame.source_label for frame in session.frames}
    assert labels == {"a.png": None, "b.png": str(Path("captures") / "b.png")}


def test_the_public_entry_point_accepts_no_metadata_parameter():
    signature = inspect.signature(build_image_derived_session)

    assert list(signature.parameters) == ["frames", "source_paths", "ingress"]
    # The ingress pin is optional, defaults off and adds no manual metadata: the
    # default session digest is unchanged when it is omitted.
    assert signature.parameters["ingress"].default is None
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values())


def test_manifest_and_scene_loaders_are_never_consulted(monkeypatch):
    def poison(*args, **kwargs):
        raise AssertionError("a metadata loader was consulted")

    monkeypatch.setattr("mc_photogrammetry.reconstruction.camera.load_camera_manifest", poison)
    monkeypatch.setattr("mc_photogrammetry.reconstruction.scene.load_scene_transform", poison)
    _patched_solution(monkeypatch)

    session = build_image_derived_session({"frame.png": _rgb()})

    assert session.local_cameras["frame.png"].position == LOCAL_FRAME_ORIGIN


# --- Serialization surface ---------------------------------------------------

def test_serialization_carries_only_the_permitted_keys(monkeypatch):
    _patched_solution(monkeypatch)
    session = build_image_derived_session({"frame.png": _rgb()})

    record = image_derived_session_to_record(session)

    assert set(record) == {
        "schema", "version", "frames", "cameras", "components", "unresolved",
        "scale_known", "local_cameras", "session_sha256",
    }
    assert set(record["frames"][0]) == _FRAME_DIGEST_KEYS
    assert record["scale_known"] is False
    assert record["cameras"][0]["pose"]["position"] is None
    # No pixel payload of any kind may travel in the record.
    json.dumps(record, allow_nan=False)
    assert not _contains_pixel_payload(record)


def test_round_trip_preserves_digests_and_does_not_fabricate_pixels(monkeypatch):
    _patched_solution(monkeypatch, image_size=(30, 20), principal=(15.0, 10.0))
    session = build_image_derived_session(
        {"a.png": _rgb((20, 30)), "b.png": np.full((20, 30), 3, dtype=np.uint8)},
        source_paths={"b.png": "b.png"},
    )

    record = image_derived_session_to_record(session)
    rebuilt = image_derived_session_from_record(record)

    assert isinstance(rebuilt, ImageDerivedSession)
    assert rebuilt.session_sha256 == session.session_sha256
    assert rebuilt.has_pixels is False
    assert all(frame.image is None for frame in rebuilt.frames)
    assert [frame.source_label for frame in rebuilt.frames] == [None, "b.png"]
    assert [frame.image_shape for frame in rebuilt.frames] == [(20, 30, 3), (20, 30)]
    assert [frame.image_sha256 for frame in rebuilt.frames] == [
        frame.image_sha256 for frame in session.frames]
    assert rebuilt.scale_known is False
    assert rebuilt.unresolved == session.unresolved
    assert rebuilt.cameras == session.cameras
    assert rebuilt.local_cameras == session.local_cameras
    assert image_derived_session_to_record(rebuilt) == record


def test_the_session_digest_is_deterministic_and_order_sensitive(monkeypatch):
    _patched_solution(monkeypatch, image_size=(30, 20), principal=(15.0, 10.0))
    frames = {"a.png": _rgb((20, 30)), "b.png": np.full((20, 30), 3, dtype=np.uint8)}

    first = build_image_derived_session(dict(frames))
    second = build_image_derived_session(dict(frames))
    reversed_order = build_image_derived_session(
        {"b.png": frames["b.png"], "a.png": frames["a.png"]})

    assert image_derived_session_to_record(first) == image_derived_session_to_record(second)
    assert first.session_sha256 == second.session_sha256
    assert reversed_order.session_sha256 != first.session_sha256


def test_a_non_finite_residual_is_recorded_as_null_and_round_trips(monkeypatch):
    def nan_residual(frames):
        solution = _solution(list(frames))
        frame_id = next(iter(solution.cameras))
        estimate = solution.cameras[frame_id]
        solution.cameras[frame_id] = CameraEstimate(
            intrinsics=estimate.intrinsics, pose=estimate.pose,
            residual_pixels=float("nan"), confidence=0.0)
        return solution

    monkeypatch.setattr(image_only, "solve_image_cameras", nan_residual)
    session = build_image_derived_session({"frame.png": _rgb()})

    record = image_derived_session_to_record(session)
    assert record["cameras"][0]["residual_pixels"] is None
    rebuilt = image_derived_session_from_record(record)
    assert rebuilt.cameras["frame.png"].residual_pixels is None


def test_the_image_digest_binds_shape_and_bytes():
    assert image_sha256(np.zeros((2, 4), dtype=np.uint8)) != image_sha256(
        np.zeros((4, 2), dtype=np.uint8))
    assert image_sha256(np.zeros((2, 4), dtype=np.uint8)) != image_sha256(
        np.ones((2, 4), dtype=np.uint8))


# --- Deserializer integrity and typed refusals -------------------------------

def test_a_tampered_frame_digest_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    record["frames"][0]["image_sha256"] = "0" * 64

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_DIGEST_MISMATCH


def test_a_resealed_record_claiming_scale_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    record["scale_known"] = True
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_FORBIDDEN_CLAIM


def test_a_record_claiming_camera_translation_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    record["cameras"][0]["pose"]["position"] = [1.0, 2.0, 3.0]

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_FORBIDDEN_CLAIM


def test_a_resealed_record_missing_an_unresolved_fact_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))
    assert _reseal(record)["session_sha256"] == record["session_sha256"]

    record["unresolved"] = [
        note for note in record["unresolved"] if "scale" not in note]
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_UNRESOLVED_FACT_MISSING


def test_a_resealed_record_with_only_resolved_wording_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    # The frame-id prefix and fact nouns survive, but the unresolved wording does
    # not; the old substring check accepted these as if the facts were recorded.
    record["unresolved"] = [
        "frame.png: position",
        "frame.png: scale",
        "frame.png: yaw quadrant",
    ]
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_UNRESOLVED_FACT_MISSING


@pytest.mark.parametrize(
    "extra_note",
    [
        "frame.png: position",
        "frame.png: scale (needs the lattice period) resolved",
        "frame.png: yaw quadrant (recovered modulo 90 degrees)",
    ],
)
def test_a_resealed_record_with_a_misleading_prefixed_note_is_refused(
        monkeypatch, extra_note):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    # Canonical notes stay present, but a resealed extra or duplicate prefixed
    # note must not be able to claim resolution for this frame.
    record["unresolved"].append(extra_note)
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_UNRESOLVED_FACT_MISSING


def test_a_resealed_record_with_a_forged_local_camera_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    record["local_cameras"][0]["fov_degrees"] = 100.0
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_RECORD_INVALID


def test_a_resealed_record_with_a_forged_local_position_is_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    record["local_cameras"][0]["position"] = [1.0, 0.0, 0.0]
    record = _reseal(record)

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)

    assert raised.value.reason == SESSION_FORBIDDEN_CLAIM


def test_unknown_or_missing_record_keys_are_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    with_extra = dict(record, bounds=[0, 0, 0])
    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(with_extra)
    assert raised.value.reason == SESSION_RECORD_INVALID

    missing = {key: value for key, value in record.items() if key != "unresolved"}
    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(missing)
    assert raised.value.reason == SESSION_RECORD_INVALID


def test_unsupported_schema_and_version_are_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    assert record["schema"] == IMAGE_DERIVED_SESSION_SCHEMA
    assert record["version"] == IMAGE_DERIVED_SESSION_VERSION

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(dict(record, schema="something-else"))
    assert raised.value.reason == SESSION_SCHEMA_UNSUPPORTED

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(dict(record, version=99))
    assert raised.value.reason == SESSION_SCHEMA_UNSUPPORTED


def test_empty_and_duplicate_frame_records_are_refused(monkeypatch):
    _patched_solution(monkeypatch)
    record = image_derived_session_to_record(
        build_image_derived_session({"frame.png": _rgb()}))

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(dict(record, frames=[]))
    assert raised.value.reason == SESSION_RECORD_INVALID

    duplicate = dict(record)
    duplicate["frames"] = [record["frames"][0], record["frames"][0]]
    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(duplicate)
    assert raised.value.reason == SESSION_RECORD_INVALID


def test_a_non_mapping_record_is_refused():
    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(["not", "a", "record"])  # type: ignore[arg-type]

    assert raised.value.reason == SESSION_RECORD_INVALID


# --- Frame contract ----------------------------------------------------------

def test_a_frame_rejects_a_writeable_or_mismatched_image():
    image = _rgb()
    with pytest.raises(ImageDerivedSessionError) as raised:
        ImageDerivedFrame(
            frame_id="frame.png", source_label=None, image=image,
            image_shape=image.shape, image_sha256=image_sha256(image))
    assert raised.value.reason == SESSION_FRAME_INVALID

    sealed = _sealed(image)
    with pytest.raises(ImageDerivedSessionError) as raised:
        ImageDerivedFrame(
            frame_id="frame.png", source_label=None, image=sealed,
            image_shape=sealed.shape, image_sha256="0" * 64)
    assert raised.value.reason == SESSION_DIGEST_MISMATCH


def test_a_frame_refuses_a_reopenable_read_only_copy():
    image = _rgb()
    reopenable = np.array(image, copy=True)
    reopenable.setflags(write=False)
    assert reopenable.flags.writeable is False
    # numpy owns this buffer, so it can be reopened; the frame must refuse it.
    assert reopenable.setflags(write=True) is None
    reopenable.setflags(write=False)

    with pytest.raises(ImageDerivedSessionError) as raised:
        ImageDerivedFrame(
            frame_id="frame.png", source_label=None, image=reopenable,
            image_shape=reopenable.shape, image_sha256=image_sha256(reopenable))

    assert raised.value.reason == SESSION_FRAME_INVALID


def test_a_deserialized_frame_reports_no_pixels():
    frame = ImageDerivedFrame(
        frame_id="frame.png", source_label=None, image=None,
        image_shape=(4, 4), image_sha256="0" * 64)

    assert frame.has_pixels is False


def _partial_session(monkeypatch, admitted=("image-0000", "image-0002")):
    ids = ("image-0000", "image-0001", "image-0002")
    seen = _patched_solution(monkeypatch, frame_ids=admitted)
    session = build_image_derived_session({name: _rgb() for name in ids},
                                         source_paths={name: f"synthetic/{name}.png" for name in ids})
    return session, seen


def test_partial_inventory_digest_and_original_id_gaps_round_trip(monkeypatch):
    session, seen = _partial_session(monkeypatch)
    assert tuple(seen) == ("image-0000", "image-0001", "image-0002")  # no reduced rerun
    assert tuple(session.cameras) == tuple(session.local_cameras) == ("image-0000", "image-0002")
    assert tuple(frame.frame_id for frame in session.frames) == ("image-0000", "image-0002")
    assert session.scale_known is False
    assert all(camera.pose.position is None for camera in session.cameras.values())
    assert all(camera.position == LOCAL_FRAME_ORIGIN for camera in session.local_cameras.values())
    record = image_derived_session_to_record(session)
    assert [item["frame_id"] for item in record["frames"]] == list(seen)
    assert record["active_frame_ids"] == ["image-0000", "image-0002"]
    assert record["camera_pooling_caveat"] == session_module.CAMERA_INDEPENDENT_CAVEAT
    assert record["admissions"][1] == {"frame_id": "image-0001", "status": "excluded",
                                        "stage": "camera_recovery", "reason": "camera_estimate_unavailable"}
    rebuilt = image_derived_session_from_record(record)
    assert image_derived_session_to_record(rebuilt) == record
    assert not rebuilt.has_pixels and all(frame.image is None for frame in rebuilt.input_frames)
    assert rebuilt.admission_summary == session.admission_summary
    assert not _contains_pixel_payload(record)


@pytest.mark.parametrize("tamper", ["excluded-hash", "excluded-reason", "excluded-stage", "partition", "ordering", "caveat"])
def test_partial_record_binds_excluded_identity_reason_partition_and_order(monkeypatch, tamper):
    session, _ = _partial_session(monkeypatch)
    record = image_derived_session_to_record(session)
    if tamper == "excluded-hash":
        record["frames"][1]["image_sha256"] = "0" * 64
    elif tamper == "excluded-reason":
        record["admissions"][1]["reason"] = "unproved-estimator-cause"
    elif tamper == "excluded-stage":
        record["admissions"][1]["stage"] = "geometry"
    elif tamper == "partition":
        record["active_frame_ids"] = ["image-0000", "image-0001"]
    elif tamper == "ordering":
        record["frames"].reverse()
    else:
        record["camera_pooling_caveat"] = "excluded inputs have no influence"
    with pytest.raises(ImageDerivedSessionError):
        image_derived_session_from_record(record)
    if tamper != "excluded-hash":
        with pytest.raises(ImageDerivedSessionError):
            image_derived_session_from_record(_reseal(record))


@pytest.mark.parametrize("admitted", [(), ("image-0002",)])
def test_zero_and_one_admitted_retain_every_input_unknown_and_pixel_free_record(monkeypatch, admitted):
    session, _ = _partial_session(monkeypatch, admitted)
    assert tuple(session.cameras) == admitted
    assert len(session.input_frames) == 3 and len(session.frames) == len(admitted)
    assert session.admission_summary["excluded"] == 3 - len(admitted)
    assert "position, scale and orientation unknown" in " ".join(session.unresolved)
    record = image_derived_session_to_record(session)
    rebuilt = image_derived_session_from_record(record)
    assert rebuilt.session_sha256 == session.session_sha256 and not rebuilt.has_pixels
    assert image_derived_session_to_record(rebuilt) == record


@pytest.mark.parametrize("bad", ["foreign", "present-null", "nonfinite-optics", "runtime"])
def test_only_absent_estimates_are_recoverable(monkeypatch, bad):
    def solve(frames):
        if bad == "runtime":
            raise RuntimeError("injected programming failure")
        solution = _solution(["image-0000"])
        if bad == "foreign":
            solution.cameras["foreign"] = _estimate()
        elif bad == "present-null":
            solution.cameras["image-0001"] = None
        else:
            solution.cameras["image-0000"] = _estimate(focal=(float("nan"), 100.))
        return solution
    monkeypatch.setattr(image_only, "solve_image_cameras", solve)
    with pytest.raises(ImageDerivedSessionError):
        build_image_derived_session({"image-0000": _rgb(), "image-0001": _rgb()})


def test_v1_exact_payload_bytes_hash_and_reader_stay_unchanged(monkeypatch):
    import hashlib
    import math
    _patched_solution(monkeypatch)
    image = _rgb()
    pixel_hash = hashlib.sha256(b"mc-photogrammetry-image-derived-session-frame-v1" + bytes([3])
                               + b"".join(size.to_bytes(8, "little") for size in (40, 60, 3))
                               + b"|u1" + image.tobytes()).hexdigest()
    intrinsics = {"image_size": [60, 40], "focal_length_pixels": [100., 100.], "principal_point_pixels": [30., 20.]}
    legacy = {"schema": IMAGE_DERIVED_SESSION_SCHEMA, "version": 1,
              "frames": [{"frame_id": "frame.png", "source_label": None, "image_shape": [40, 60, 3], "image_sha256": pixel_hash}],
              "cameras": [{"frame_id": "frame.png", "intrinsics": intrinsics,
                           "pose": {"position": None, "yaw_degrees": 14., "pitch_degrees": 17., "roll_degrees": 3.5},
                           "residual_pixels": .4, "confidence": 0.}],
              "components": [], "unresolved": list(_solution(["frame.png"]).unresolved), "scale_known": False,
              "local_cameras": [{"frame_id": "frame.png", "position": [0., 0., 0.], "yaw_degrees": 14., "pitch_degrees": 17.,
                                 "fov_degrees": math.degrees(2 * math.atan(40 / 2 / 100)), "eye_height": 1.62,
                                 "roll_degrees": 3.5, "intrinsics": intrinsics}]}
    encoded = json.dumps(legacy, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    legacy["session_sha256"] = hashlib.sha256(encoded).hexdigest()
    produced = image_derived_session_to_record(build_image_derived_session({"frame.png": image}))
    assert produced == legacy
    assert json.dumps(produced, sort_keys=True, separators=(",", ":")) == json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    assert image_derived_session_to_record(image_derived_session_from_record(legacy)) == legacy


@pytest.mark.parametrize("admitted", [(), ("image-0000",)], ids=["all-excluded", "partial"])
def test_legacy_v2_caveat_preserves_complete_record_and_digest(monkeypatch, admitted):
    import hashlib

    session, _ = _partial_session(monkeypatch, admitted)
    legacy = image_derived_session_to_record(session)
    legacy["camera_pooling_caveat"] = (
        "Excluded inputs may have influenced pooled focal estimation; "
        "no reduced-subset rerun or independence claim."
    )
    assert session_module.CAMERA_POOLING_CAVEAT == legacy["camera_pooling_caveat"]
    assert session_module.CAMERA_POOLING_CAVEAT_LEGACY == legacy["camera_pooling_caveat"]
    legacy = _reseal(legacy)
    encoded = json.dumps(legacy, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode()
    rebuilt = image_derived_session_from_record(legacy)
    serialized = image_derived_session_to_record(rebuilt)

    assert rebuilt.camera_pooling_caveat == legacy["camera_pooling_caveat"]
    assert serialized == legacy
    assert serialized["session_sha256"] == legacy["session_sha256"]
    assert hashlib.sha256(json.dumps(serialized, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).digest() == hashlib.sha256(encoded).digest()


@pytest.mark.parametrize("admitted", [(), ("image-0000",)], ids=["all-excluded", "partial"])
def test_new_v2_sessions_bind_exact_independent_wording(monkeypatch, admitted):
    session, _ = _partial_session(monkeypatch, admitted)
    record = image_derived_session_to_record(session)
    expected = (
        "Camera estimates were fitted independently per input frame; "
        "no focal pooling or reduced-subset rerun was used."
    )
    assert session_module.CAMERA_INDEPENDENT_CAVEAT == expected
    assert session.camera_pooling_caveat == expected
    assert record["camera_pooling_caveat"] == expected
    rebuilt = image_derived_session_from_record(record)
    assert rebuilt.camera_pooling_caveat == expected
    assert image_derived_session_to_record(rebuilt) == record


@pytest.mark.parametrize("unknown", ["unknown", None, True, ["legacy"]],
                         ids=["unknown-string", "null", "boolean", "list"])
def test_unknown_caveats_refuse_even_when_resealed_or_constructed(monkeypatch, unknown):
    from dataclasses import replace

    session, _ = _partial_session(monkeypatch)
    record = image_derived_session_to_record(session)
    record["camera_pooling_caveat"] = unknown
    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(_reseal(record))
    assert raised.value.reason == SESSION_RECORD_INVALID
    with pytest.raises(ImageDerivedSessionError) as raised:
        replace(session, camera_pooling_caveat=unknown)
    assert raised.value.reason == SESSION_RECORD_INVALID


@pytest.mark.parametrize("original_caveat", ["legacy", "independent"])
def test_allowed_caveat_change_without_resealing_fails_digest(monkeypatch, original_caveat):
    session, _ = _partial_session(monkeypatch)
    record = image_derived_session_to_record(session)
    legacy = session_module.CAMERA_POOLING_CAVEAT_LEGACY
    independent = session_module.CAMERA_INDEPENDENT_CAVEAT
    record["camera_pooling_caveat"] = legacy if original_caveat == "legacy" else independent
    record = _reseal(record)
    record["camera_pooling_caveat"] = independent if original_caveat == "legacy" else legacy

    with pytest.raises(ImageDerivedSessionError) as raised:
        image_derived_session_from_record(record)
    assert raised.value.reason == SESSION_DIGEST_MISMATCH


@pytest.mark.parametrize("version", [1, 2], ids=["v1", "v2"])
def test_existing_positional_constructor_uses_conservative_legacy_default(monkeypatch, version):
    if version == 1:
        _patched_solution(monkeypatch)
        session = build_image_derived_session({"frame.png": _rgb()})
    else:
        session, _ = _partial_session(monkeypatch)
    # Preserve all eleven historical positional slots, including the digest,
    # full inventory and admissions; the new field is appended after them.
    original_record = image_derived_session_to_record(session)
    expected = dict(original_record)
    if version == 2:
        expected["camera_pooling_caveat"] = session_module.CAMERA_POOLING_CAVEAT_LEGACY
        expected = _reseal(expected)
    compatible = ImageDerivedSession(
        session.schema, session.version, session.frames, session.cameras,
        session.components, session.unresolved, session.scale_known,
        session.local_cameras, expected["session_sha256"], session.input_frames,
        session.admissions,
    )
    assert compatible.camera_pooling_caveat == session_module.CAMERA_POOLING_CAVEAT_LEGACY
    assert image_derived_session_to_record(compatible) == expected
    if version == 1:
        assert "camera_pooling_caveat" not in original_record
        assert expected == original_record
        assert compatible.session_sha256 == session.session_sha256
