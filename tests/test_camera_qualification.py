"""Pure synthetic camera QA: no PNG/world/fixture reads or production imports.

UV constants below are hand-derived analytic test values. They are explicitly
NOT manual annotations or qualification evidence for any actual capture.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluation/camera_qualification.py"
_SPEC = importlib.util.spec_from_file_location("camera_qualification_pure_tests", _PATH)
qa = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qa
_SPEC.loader.exec_module(qa)

# Identity RH camera, depth=-world_z, fx=fy=50, center=(50,50).
XYZ = ((-1, 1, -2), (1, 1, -2), (-1, -1, -2), (1, -1, -2),
       (-1, 1, -4), (1, 1, -4), (-1, -1, -4), (1, -1, -4))
UV = ((25, 25), (75, 25), (25, 75), (75, 75),
      (37.5, 37.5), (62.5, 37.5), (37.5, 62.5), (62.5, 62.5))
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def binding(**changes):
    values = dict(frame_id="synthetic-only-frame-a", png_sha256="a" * 64, width=100, height=100,
                  capture_id="synthetic-only-fresh-pair", world_sha256="b" * 64,
                  model_sha256="c" * 64, context_sha256="d" * 64)
    values.update(changes)
    return qa.FrameBinding(**values)


def plan_doc():
    return dict(schema=qa.PLAN_SCHEMA, evaluation_only=True, policy_sha256=qa.POLICY_SHA256,
                world_sha256="b" * 64, model_sha256="c" * 64, context_sha256="d" * 64,
                landmarks=[dict(landmark_id=f"natural-test-{i}", world_xyz=list(xyz),
                                surface_id="near" if i < 4 else "far",
                                structural_identity=f"test-corner-{i}",
                                identity_cue=f"synthetic corner {i}", support_class="natural_corner",
                                required=True) for i, xyz in enumerate(XYZ)])


def load_plan(doc=None):
    payload = raw(plan_doc() if doc is None else doc)
    return qa.load_landmark_plan(payload, expected_sha256=digest(payload))


def packet_doc(plan, bound=None, pixels=UV):
    bound = binding() if bound is None else bound
    return dict(schema=qa.PACKET_SCHEMA, evaluation_only=True, method="manual_png_observation",
                observer_id="synthetic-not-a-real-annotator", tool_version="synthetic-test-v1",
                protocol=qa.PROTOCOL, pixel_convention=qa.PIXEL_CONVENTION,
                projected_coordinates_exposed=False, frozen_before_projection=True,
                landmark_plan_sha256=plan.sha256, policy_sha256=qa.POLICY_SHA256,
                frame_id=bound.frame_id, png_sha256=bound.png_sha256,
                dimensions=[bound.width, bound.height], capture_id=bound.capture_id,
                world_sha256=bound.world_sha256, model_sha256=bound.model_sha256,
                context_sha256=bound.context_sha256,
                observations=[dict(landmark_id=point.landmark_id, disposition="observed", uv=list(uv),
                                   uncertainty_pixels=0.25, uncertainty_method="manual_localization_bound",
                                   support_segments=[[[uv[0] - 1, uv[1]], [uv[0] + 1, uv[1]]]],
                                   second_annotation=None)
                              for point, uv in zip(plan.landmarks, pixels)])


def seal(plan, doc=None, bound=None):
    bound = binding() if bound is None else bound
    payload = raw(packet_doc(plan, bound) if doc is None else doc)
    return qa.seal_observations(payload, plan, expected_packet_sha256=digest(payload),
                                expected_binding=bound)


def camera(bound=None, **changes):
    bound = binding() if bound is None else bound
    values = dict(binding=bound, eye=(0., 0., 0.), rotation=IDENTITY, focal_pixels=(50., 50.),
                  principal_point=(50., 50.), viewport=(0, 0, bound.width, bound.height))
    values.update(changes)
    return qa.CameraHypothesis(**values)


def assess(plan=None, observation="default", candidate=None, bound=None):
    plan = load_plan() if plan is None else plan
    bound = binding() if bound is None else bound
    observation = seal(plan, bound=bound) if observation == "default" else observation
    return qa.assess_camera(plan, observation, camera(bound) if candidate is None else candidate,
                            expected_binding=bound, expected_validator_sha256=qa.validator_code_sha256(),
                            expected_plan_sha256=plan.sha256,
                            expected_observation_sha256=None if observation is None else observation.sha256)


def refuses(code, action):
    with pytest.raises(qa.QualificationError) as caught:
        action()
    assert caught.value.code == code


def pair(first, first_camera, second, second_camera, *, first_plan=None, second_plan=None,
         first_packet="default", second_packet="default", **anchors):
    """Test anchors come from synthetic acquisition inputs, never receipt fields."""
    first_plan = load_plan() if first_plan is None else first_plan
    second_plan = load_plan() if second_plan is None else second_plan
    first_packet = seal(first_plan, bound=first_camera.binding) if first_packet == "default" else first_packet
    second_packet = seal(second_plan, bound=second_camera.binding) if second_packet == "default" else second_packet
    inputs = dict(first_plan_raw=first_plan.raw, second_plan_raw=second_plan.raw,
                  first_packet_raw=None if first_packet is None else first_packet.raw,
                  second_packet_raw=None if second_packet is None else second_packet.raw,
                  expected_first_binding=first_camera.binding, expected_second_binding=second_camera.binding,
                  expected_first_plan_sha256=first_plan.sha256, expected_second_plan_sha256=second_plan.sha256,
                  expected_first_packet_sha256=None if first_packet is None else first_packet.sha256,
                  expected_second_packet_sha256=None if second_packet is None else second_packet.sha256,
                  expected_validator_sha256=qa.validator_code_sha256())
    inputs.update(anchors)
    return qa.assess_pair(first, first_camera, second, second_camera, **inputs)


def test_analytic_camera_pass_is_never_visibility_pass():
    receipt = assess()
    assert receipt.status == "PASS" and receipt.reasons == ()
    assert receipt.observed_count == 8 and len(receipt.residuals) == 8
    assert receipt.maximum_residual_pixels == receipt.median_residual_pixels == 0
    assert len(receipt.coverage_bins) >= 4
    assert [r.projected_uv for r in receipt.residuals] == list(UV)
    assert [r.signed_depth for r in receipt.residuals] == [2] * 4 + [4] * 4
    assert receipt.visibility_status == "UNAVAILABLE"
    calls = []
    refuses("complete_outside_occluder_context_unverified",
            lambda: qa.sample_certified_visibility(receipt, lambda: calls.append("sampler")))
    assert calls == []


@pytest.mark.parametrize("status", ["FAIL", "UNAVAILABLE", "invented"])
def test_camera_refusal_precedes_sampler(status):
    calls = []
    receipt = replace(assess(), status=status)
    refuses("camera_not_qualified", lambda: qa.sample_certified_visibility(receipt, lambda: calls.append(1)))
    assert calls == []


def test_self_asserted_complete_context_or_pass_does_not_open_visibility():
    receipt = replace(assess(), visibility_status="PASS", visibility_reason="complete=True")
    refuses("complete_outside_occluder_context_unverified", lambda: qa.require_certified_visibility(receipt))


def test_missing_observations_and_single_point_are_unavailable_not_exception():
    receipt = assess(observation=None)
    assert receipt.status == "UNAVAILABLE" and "landmark_observations_missing" in receipt.reasons
    doc = plan_doc()
    doc["landmarks"] = doc["landmarks"][:1]
    receipt = assess(load_plan(doc), observation=None)
    assert "insufficient_noncoplanar_support" in receipt.reasons


@pytest.mark.parametrize("field,value,code", [
    ("method", "manifest_projection", "observation_source_forbidden"),
    ("method", "mod_actual", "observation_source_forbidden"),
    ("method", "sift_without_world_association", "observation_source_forbidden"),
    ("projected_coordinates_exposed", True, "observation_not_blinded"),
    ("frozen_before_projection", False, "observation_not_blinded"),
    ("protocol", "residual_overlay", "observation_convention"),
    ("pixel_convention", "bottom-left", "observation_convention"),
    ("policy_sha256", "e" * 64, "policy_mismatch"),
    ("landmark_plan_sha256", "e" * 64, "landmark_plan_mismatch"),
    ("evaluation_only", False, "observation_schema"),
    ("schema", "other", "observation_schema"),
])
def test_observation_source_blinding_and_policy_controls(field, value, code):
    plan = load_plan()
    doc = packet_doc(plan)
    doc[field] = value
    refuses(code, lambda: seal(plan, doc))


@pytest.mark.parametrize("field,value", [
    ("frame_id", "stale-frame"), ("capture_id", "other-capture"), ("png_sha256", "e" * 64),
    ("dimensions", [100, 101]), ("dimensions", [100., 100]),
    ("world_sha256", "e" * 64), ("model_sha256", "e" * 64), ("context_sha256", "e" * 64),
])
def test_exact_frame_png_world_bindings(field, value):
    plan = load_plan()
    doc = packet_doc(plan)
    doc[field] = value
    refuses("frame_binding_mismatch", lambda: seal(plan, doc))


@pytest.mark.parametrize("extra", ["predicted_uv", "actual", "requested", "camera", "fov", "residuals"])
def test_blinded_packet_cannot_contain_prediction_metadata(extra):
    plan = load_plan()
    doc = packet_doc(plan)
    doc[extra] = {}
    refuses("invalid_fields", lambda: seal(plan, doc))
    planned = plan_doc()
    planned["landmarks"][0][extra] = {}
    refuses("invalid_fields", lambda: load_plan(planned))


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "foreign"])
def test_exact_ordered_membership_never_filters_bad_landmarks(change):
    plan = load_plan()
    doc = packet_doc(plan)
    if change == "missing":
        doc["observations"].pop()
    elif change == "duplicate":
        doc["observations"][1] = copy.deepcopy(doc["observations"][0])
    elif change == "reordered":
        doc["observations"].reverse()
    else:
        doc["observations"][0]["landmark_id"] = "foreign"
    refuses("observation_membership", lambda: seal(plan, doc))


@pytest.mark.parametrize("disposition", ["ambiguous", "occluded", "out_of_frame", "unresolvable"])
def test_all_dispositions_retained_and_required_absence_abstains(disposition):
    plan = load_plan()
    doc = packet_doc(plan)
    doc["observations"][0] = dict(landmark_id=plan.landmarks[0].landmark_id,
                                   disposition=disposition, reason="synthetic unavailable raster feature")
    receipt = assess(plan, seal(plan, doc))
    assert receipt.status == "UNAVAILABLE" and f"landmark_{disposition}" in receipt.reasons
    assert len(receipt.residuals) == 8 and receipt.residuals[0].disposition == disposition
    assert receipt.observed_count == 7


@pytest.mark.parametrize("field,value,code", [
    ("uv", [-1, 25], "pixel_out_of_bounds"), ("uv", [100, 25], "pixel_out_of_bounds"),
    ("uv", [25, True], "invalid_number"), ("uncertainty_pixels", -1, "invalid_uncertainty"),
    ("uncertainty_method", "unknown", "uncertainty_method"),
    ("support_segments", [], "raster_support_cap"),
    ("support_segments", [[[25, 25], [25, 25]]], "invalid_raster_support"),
    ("support_segments", [[[25, 25], [100, 25]]], "pixel_out_of_bounds"),
])
def test_raster_measurement_contract(field, value, code):
    plan = load_plan()
    doc = packet_doc(plan)
    doc["observations"][0][field] = value
    refuses(code, lambda: seal(plan, doc))


def test_two_annotators_disagreement_is_not_averaged():
    plan = load_plan()
    doc = packet_doc(plan)
    doc["observations"][0]["second_annotation"] = dict(observer_id="independent-test-observer",
                                                       uv=[26, 25], uncertainty_pixels=0.25)
    receipt = assess(plan, seal(plan, doc))
    assert receipt.status == "UNAVAILABLE" and "independent_annotations_disagree" in receipt.reasons
    assert receipt.residuals[0].observed_uv == (25, 25)
    doc["observations"][0]["second_annotation"]["observer_id"] = doc["observer_id"]
    refuses("second_observer_not_independent", lambda: seal(plan, doc))


@pytest.mark.parametrize("delta,uncertainty,status,reason", [
    (0.5, 0.25, "PASS", None), (1.1, 0.25, "UNAVAILABLE", "residual_uncertainty_indeterminate"),
    (1.4, 0.25, "FAIL", "residual_gate_failed"),
    (0., 1.1, "UNAVAILABLE", "localization_uncertainty_excessive"),
])
def test_fixed_median_residual_and_uncertainty_gates(delta, uncertainty, status, reason):
    plan = load_plan()
    doc = packet_doc(plan)
    for row in doc["observations"]:
        row["uv"][0] += delta
        row["uncertainty_pixels"] = uncertainty
    receipt = assess(plan, seal(plan, doc))
    assert receipt.status == status
    if reason:
        assert reason in receipt.reasons


def test_maximum_residual_cannot_be_hidden_by_low_median():
    plan = load_plan()
    doc = packet_doc(plan)
    doc["observations"][0]["uv"][0] += 3
    receipt = assess(plan, seal(plan, doc))
    assert receipt.status == "FAIL" and receipt.median_residual_pixels == 0
    assert receipt.maximum_residual_pixels == 3


@pytest.mark.parametrize("changes,code", [
    ({"rotation": ((-1., 0., 0.), (0., 1., 0.), (0., 0., 1.))}, "improper_rotation"),
    ({"rotation": ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.))}, "improper_rotation"),
    ({"rotation": ((1., .001, 0.), (0., 1., 0.), (0., 0., 1.))}, "nonrigid_rotation"),
    ({"focal_pixels": (0., 50.)}, "invalid_intrinsics"),
    ({"focal_pixels": (math.inf, 50.)}, "invalid_number"),
    ({"eye": (math.nan, 0., 0.)}, "invalid_number"),
    ({"convention": "camera-to-world"}, "unsupported_camera_convention"),
    ({"viewport": (1, 0, 99, 100)}, "unsupported_viewport"),
    ({"viewport": (0, 0, 100., 100)}, "unsupported_viewport"),
    ({"effects": ("view_bob",)}, "unsupported_camera_effects"),
    ({"effects": ("shader_distortion",)}, "unsupported_camera_effects"),
    ({"principal_point": (-1., 50.)}, "invalid_intrinsics"),
    ({"source": "trusted_mod_actual"}, "unsupported_camera_source"),
])
def test_camera_malformed_and_unsupported_refusals(changes, code):
    refuses(code, lambda: assess(candidate=camera(**changes)))


@pytest.mark.parametrize("changes", [
    {"eye": (0., 1.62, 0.)}, {"eye": (1., 0., 0.)},
    {"focal_pixels": (100., 50.)}, {"focal_pixels": (50., 25.)},
    {"rotation": ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))},
    {"rotation": ((1., 0., 0.), (0., -1., 0.), (0., 0., -1.))},
])
def test_axis_fov_eye_offset_or_back_facing_hypotheses_cannot_pass(changes):
    receipt = assess(candidate=camera(**changes))
    assert receipt.status == "FAIL"


def test_half_pixel_center_is_measured_without_promising_subpixel_rejection():
    receipt = assess(candidate=camera(principal_point=(50.5, 50.5)))
    assert receipt.maximum_residual_pixels == pytest.approx(math.sqrt(0.5))
    assert receipt.status == "PASS"  # falls within the frozen 1px median+0.25 bound


def test_same_resolution_different_focal_length_is_not_pooled():
    plan = load_plan()
    pixels = ((20, 20), (80, 20), (20, 80), (80, 80),
              (35, 35), (65, 35), (35, 65), (65, 65))  # independent analytic fx=fy=60
    packet = seal(plan, packet_doc(plan, pixels=pixels))
    assert assess(plan, packet, camera(focal_pixels=(60., 60.))).status == "PASS"
    assert assess(plan, packet, camera(focal_pixels=(50., 50.))).status == "FAIL"


def test_rotation_transpose_yaw_wrap_and_near_vertical_analytic_controls():
    quarter = ((0., 0., -1.), (0., 1., 0.), (1., 0., 0.))
    uv = qa.project_hypothesis(camera(rotation=quarter), (-2., 1., -1.))
    assert uv == (75., 25., 2.)
    wrong_order = tuple(tuple(quarter[j][i] for j in range(3)) for i in range(3))
    refuses("nonpositive_depth", lambda: qa.project_hypothesis(camera(rotation=wrong_order), (-2., 1., -1.)))
    wrapped = ((math.cos(2 * math.pi), 0., math.sin(2 * math.pi)), (0., 1., 0.),
               (-math.sin(2 * math.pi), 0., math.cos(2 * math.pi)))
    assert qa.project_hypothesis(camera(rotation=wrapped), XYZ[0]) == pytest.approx((*UV[0], 2.))
    vertical = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
    assert qa.project_hypothesis(camera(rotation=vertical), (1., -2., -1.)) == (75., 25., 2.)


def test_thin_support_insufficient_bins_and_coplanarity_abstain():
    doc = plan_doc()
    for i, point in enumerate(doc["landmarks"]):
        point["world_xyz"] = [i / 100., 0., -2.]
        point["surface_id"] = "one-plane"
    plan = load_plan(doc)
    pixels = tuple((50 + i / 4., 50) for i in range(8))
    receipt = assess(plan, seal(plan, packet_doc(plan, pixels=pixels)))
    assert receipt.status == "UNAVAILABLE"
    assert set(receipt.reasons) >= {"insufficient_image_coverage", "insufficient_noncoplanar_support",
                                   "insufficient_depth_separation"}


def test_sealed_content_and_external_digest_anchors_detect_mutations():
    plan = load_plan()
    packet = seal(plan)
    changed = replace(packet.observations[0], uv=(26., 25.))
    tampered = replace(packet, observations=(changed,) + packet.observations[1:])
    refuses("observations_tampered", lambda: assess(plan, tampered))
    forged_plan = replace(plan, landmarks=(replace(plan.landmarks[0], world_xyz=(0., 0., -2.)),)
                          + plan.landmarks[1:])
    refuses("landmark_plan_tampered", lambda: assess(forged_plan, packet))
    refuses("packet_digest_mismatch", lambda: qa.seal_observations(
        packet.raw + b" ", plan, expected_packet_sha256=packet.sha256, expected_binding=binding()))
    refuses("validator_code_mismatch", lambda: qa.assess_camera(
        plan, packet, camera(), expected_binding=binding(), expected_validator_sha256="e" * 64,
        expected_plan_sha256=plan.sha256, expected_observation_sha256=packet.sha256))
    refuses("landmark_plan_mismatch", lambda: qa.assess_camera(
        plan, packet, camera(), expected_binding=binding(), expected_validator_sha256=qa.validator_code_sha256(),
        expected_plan_sha256="e" * 64, expected_observation_sha256=packet.sha256))
    refuses("observation_digest_mismatch", lambda: qa.assess_camera(
        plan, packet, camera(), expected_binding=binding(), expected_validator_sha256=qa.validator_code_sha256(),
        expected_plan_sha256=plan.sha256, expected_observation_sha256="e" * 64))


@pytest.mark.parametrize("change,code", [
    ("duplicate_id", "duplicate_landmark"), ("duplicate_xyz", "duplicate_world_landmark"),
    ("optional", "optional_landmark_forbidden"), ("many", "landmark_count_cap"),
])
def test_plan_unique_fixed_membership_caps(change, code):
    doc = plan_doc()
    if change == "duplicate_id":
        doc["landmarks"][1]["landmark_id"] = doc["landmarks"][0]["landmark_id"]
    elif change == "duplicate_xyz":
        doc["landmarks"][1]["world_xyz"] = doc["landmarks"][0]["world_xyz"]
    elif change == "optional":
        doc["landmarks"][0]["required"] = False
    else:
        doc["landmarks"] *= 5
    refuses(code, lambda: load_plan(doc))


@pytest.mark.parametrize("payload,code", [
    (b" " * (qa.MAX_PACKET_BYTES + 1), "packet_byte_cap"),
    (b'{"x":' + b'[' * 13 + b'0' + b']' * 13 + b'}', "packet_depth_cap"),
    (b'{"x":[' + b'0,' * 4097 + b'0]}', "packet_token_cap"),
    (b'{"x":1,"x":2}', "duplicate_json_key"), (b'{"x":NaN}', "invalid_number"),
    (b'{"x":1e999}', "invalid_number"), (b'{"x":' + b'9' * 21 + b'}', "integer_text_cap"),
    (b'{"x":', "invalid_json"), (b'{"x":"\xff"}', "invalid_json"),
], ids=["bytes", "depth", "tokens", "duplicate-key", "nan", "infinity", "integer", "syntax", "utf8"])
def test_preparse_malformed_and_resource_caps(payload, code):
    refuses(code, lambda: qa.load_landmark_plan(payload, expected_sha256=digest(payload)))


def test_pair_world_gauge_baseline_sign_and_candidate_binding():
    plan = load_plan()
    first_cam = camera()
    first = assess(plan, candidate=first_cam)
    bound = binding(frame_id="synthetic-only-frame-b", png_sha256="e" * 64)
    # Eye=(.2,0,0): hand-derived near shift -5px; far shift -2.5px.
    pixels = ((20, 25), (70, 25), (20, 75), (70, 75),
              (35, 37.5), (60, 37.5), (35, 62.5), (60, 62.5))
    second_cam = camera(bound, eye=(.2, 0., 0.))
    second_packet = seal(plan, packet_doc(plan, bound, pixels), bound)
    second = assess(plan, second_packet, second_cam, bound)
    assert second.status == "PASS"
    result = pair(first, first_cam, second, second_cam, second_packet=second_packet)
    assert result.status == "PASS" and result.baseline_blocks == .2
    assert result.relative_rotation == IDENTITY and result.relative_translation == (-.2, 0., 0.)
    assert result.visibility_status == "UNAVAILABLE"
    refuses("pair_receipt_evidence_mismatch",
            lambda: pair(first, first_cam, second, replace(second_cam, eye=(-.2, 0., 0.)),
                         second_packet=second_packet))
    reversed_receipt = assess(plan, seal(plan, packet_doc(plan, bound, pixels), bound),
                              camera(bound, eye=(-.2, 0., 0.)), bound)
    assert reversed_receipt.status == "FAIL"
    same_eye = camera(bound)
    same_receipt = assess(plan, seal(plan, packet_doc(plan, bound), bound), same_eye, bound)
    assert pair(first, first_cam, same_receipt, same_eye).reasons == ("zero_baseline",)
    refuses("pair_frame_reused", lambda: pair(first, first_cam, first, first_cam))


def test_pixel_gate_never_fits_camera_or_rewrites_sealed_measurements():
    plan = load_plan()
    packet = seal(plan)
    original = packet.raw
    candidate = camera(eye=(1., 0., 0.))
    assert assess(plan, packet, candidate).status == "FAIL"
    assert packet.raw is original and candidate.eye == (1., 0., 0.)
    assert [x.uv for x in packet.observations] == list(UV)


def test_signed_depth_failures_remain_in_complete_evidence():
    receipt = assess(candidate=camera(rotation=((1., 0., 0.), (0., -1., 0.), (0., 0., -1.))))
    assert receipt.status == "FAIL"
    assert [row.signed_depth for row in receipt.residuals] == [-2] * 4 + [-4] * 4
    assert all(row.reason == "nonpositive_depth" for row in receipt.residuals)
    assert all(row.projected_uv is None for row in receipt.residuals)


def test_repeated_ray_landmarks_cannot_count_as_independent_visible_pixels():
    doc = plan_doc()
    for point in doc["landmarks"][4:]:
        point["world_xyz"][0] *= 2
        point["world_xyz"][1] *= 2
    plan = load_plan(doc)
    packet = seal(plan, packet_doc(plan, pixels=UV[:4] + UV[:4]))
    receipt = assess(plan, packet)
    assert receipt.status == "UNAVAILABLE" and "indistinguishable_raster_landmarks" in receipt.reasons


def test_32_landmark_envelope_keeps_every_record_and_fixed_coverage():
    doc = plan_doc()
    doc["landmarks"] = []
    pixels = []
    # Independent hand arithmetic: fx=50; near scale25 and far scale12.5.
    for surface, z, world_steps, pixel_steps in (
            ("near", -2, (-1.2, -.4, .4, 1.2), (20, 40, 60, 80)),
            ("far", -4, (-1.5, -.5, .5, 1.5), (31.25, 43.75, 56.25, 68.75))):
        for i, x in enumerate(world_steps):
            for j, y in enumerate(world_steps):
                index = len(doc["landmarks"])
                doc["landmarks"].append(dict(landmark_id=f"envelope-{index}", world_xyz=[x, y, z],
                    surface_id=surface, structural_identity=f"corner-{index}", identity_cue=f"test {index}",
                    support_class="natural_corner", required=True))
                pixels.append((pixel_steps[i], 100 - pixel_steps[j]))
    plan = load_plan(doc)
    packet = seal(plan, packet_doc(plan, pixels=pixels))
    receipt = assess(plan, packet)
    assert receipt.status == "PASS" and receipt.observed_count == len(receipt.residuals) == 32
    assert len(receipt.coverage_bins) == 16
    assert len(plan.raw) <= qa.MAX_PACKET_BYTES and len(packet.raw) <= qa.MAX_PACKET_BYTES
    print(f"CAMERA_PACKET_ENVELOPE plan_bytes={len(plan.raw)} packet_bytes={len(packet.raw)} "
          f"landmarks={len(receipt.residuals)} bins={len(receipt.coverage_bins)}")


def test_native_center_and_nonsquare_intrinsics_are_explicit():
    bound = binding(width=200, height=100)
    candidate = camera(bound, focal_pixels=(100., 50.), principal_point=(99.5, 49.5))
    assert qa.project_hypothesis(candidate, (-1., 1., -2.)) == (49.5, 24.5, 2.)
    # Horizontal and vertical FOV/focal axes cannot be interchanged silently.
    swapped = replace(candidate, focal_pixels=(50., 100.))
    assert qa.project_hypothesis(swapped, (-1., 1., -2.)) == (74.5, -.5, 2.)


def test_maximum_dimension_and_context_binding_controls():
    refuses("image_dimension_cap", lambda: assess(bound=binding(width=16384, height=16384)))
    refuses("invalid_integer", lambda: assess(bound=binding(width=True)))
    refuses("invalid_sha256", lambda: assess(bound=binding(png_sha256="A" * 64)))
    refuses("plan_context_mismatch", lambda: assess(bound=binding(world_sha256="e" * 64)))
    refuses("camera_frame_mismatch", lambda: assess(candidate=camera(binding(frame_id="other-frame"))))


def test_successful_second_annotation_is_retained_without_fitting():
    plan = load_plan()
    doc = packet_doc(plan)
    doc["observations"][0]["second_annotation"] = dict(observer_id="independent-test-observer",
                                                       uv=[25.1, 25], uncertainty_pixels=.25)
    packet = seal(plan, doc)
    assert assess(plan, packet).status == "PASS"
    assert packet.observations[0].second_uv == (25.1, 25)
    assert packet.observations[0].uv == (25, 25)


def test_source_code_size_bound_and_unavailable_pair(monkeypatch):
    import io
    class SourcePath:
        def __init__(self, unused):
            pass
        def open(self, mode):
            assert mode == "rb"
            return io.BytesIO(b"x" * (qa.MAX_CODE_BYTES + 1))
    with monkeypatch.context() as isolated:
        isolated.setattr(qa, "Path", SourcePath)
        refuses("validator_code_cap", qa.validator_code_sha256)
    first = assess()
    missing = assess(observation=None)
    assert pair(first, camera(), missing, camera(), second_packet=None).status == "UNAVAILABLE"


def test_legacy_finalizer_and_exporter_remain_unmodified_and_unqualified():
    # Read source bytes only; do not import either module or read old artifacts.
    root = _PATH.parents[2]
    assert digest((root / "src/mc_photogrammetry/dataset/finalize.py").read_bytes()) == (
        "3e5d2e6136158e0c680ecf0a853b8bbdf1b320b5608844b393e6d657d5fe8cb2")
    exporter = root / "scripts/evaluation/export_factory_texture_patch_dataset.py"
    assert digest(exporter.read_bytes()) == "290720b253b7da4d4c7e77bc9431b0b6ecd44c40ed6e5cb17a73293e301a8c94"


@pytest.mark.parametrize("field,value,code", [
    ("schema", "forged", "receipt_schema_mismatch"),
    ("policy_sha256", "e" * 64, "policy_mismatch"),
    ("validator_sha256", "e" * 64, "validator_code_mismatch"),
])
def test_pair_does_not_accept_shared_but_wrong_receipt_policy_or_code(field, value, code):
    receipt = replace(assess(), **{field: value})
    refuses(code, lambda: pair(receipt, camera(), receipt, camera()))


def pair_evidence():
    plan = load_plan()
    bound = binding(frame_id="synthetic-only-frame-b", png_sha256="e" * 64)
    first_camera = camera()
    second_camera = camera(bound, eye=(.2, 0., 0.))
    pixels = ((20, 25), (70, 25), (20, 75), (70, 75),
              (35, 37.5), (60, 37.5), (35, 62.5), (60, 62.5))
    first_packet = seal(plan)
    second_packet = seal(plan, packet_doc(plan, bound, pixels), bound)
    first = assess(plan, first_packet, first_camera)
    second = assess(plan, second_packet, second_camera, bound)
    return plan, first_camera, second_camera, first_packet, second_packet, first, second


def test_astra_missing_observations_forged_pair_pass_is_rejected():
    plan, first_camera, second_camera, _, _, _, _ = pair_evidence()
    first = assess(plan, None, first_camera)
    second = assess(plan, None, second_camera, second_camera.binding)
    assert first.status == second.status == "UNAVAILABLE"
    honest = pair(first, first_camera, second, second_camera, first_packet=None, second_packet=None)
    assert honest.status == "UNAVAILABLE"
    forged_first = replace(first, status="PASS", reasons=())
    forged_second = replace(second, status="PASS", reasons=())
    refuses("pair_receipt_evidence_mismatch", lambda: pair(
        forged_first, first_camera, forged_second, second_camera, first_packet=None, second_packet=None))


def test_failed_residual_forged_pair_pass_is_rejected():
    plan, first_camera, second_camera, first_packet, second_packet, _, _ = pair_evidence()
    bad_first = replace(first_camera, eye=(1., 0., 0.))
    bad_second = replace(second_camera, eye=(1.2, 0., 0.))
    first = assess(plan, first_packet, bad_first)
    second = assess(plan, second_packet, bad_second, bad_second.binding)
    assert first.status == second.status == "FAIL"
    assert "residual_gate_failed" in first.reasons
    forged_first = replace(first, status="PASS", reasons=(), median_residual_pixels=0., maximum_residual_pixels=0.)
    forged_second = replace(second, status="PASS", reasons=(), median_residual_pixels=0., maximum_residual_pixels=0.)
    refuses("pair_receipt_evidence_mismatch", lambda: pair(
        forged_first, bad_first, forged_second, bad_second, first_packet=first_packet, second_packet=second_packet))


@pytest.mark.parametrize("change", ["status", "reasons", "residuals", "count", "bins", "median", "maximum",
                                  "observation_hash", "plan_hash", "candidate_hash", "binding"])
def test_complete_receipt_replay_refuses_each_altered_evidence_component(change):
    plan, c1, c2, p1, p2, r1, r2 = pair_evidence()
    altered = {
        "status": dict(status="FAIL"), "reasons": dict(reasons=("forged",)),
        "residuals": dict(residuals=(replace(r1.residuals[0], projected_uv=(26., 25.)),) + r1.residuals[1:]),
        "count": dict(observed_count=0), "bins": dict(coverage_bins=()),
        "median": dict(median_residual_pixels=0.5), "maximum": dict(maximum_residual_pixels=0.5),
        "observation_hash": dict(observation_sha256="f" * 64),
        "plan_hash": dict(plan_sha256="f" * 64), "candidate_hash": dict(candidate_sha256="f" * 64),
        "binding": dict(binding=replace(r1.binding, frame_id="stale")),
    }[change]
    refuses("pair_receipt_evidence_mismatch", lambda: pair(
        replace(r1, **altered), c1, r2, c2, first_plan=plan, second_plan=plan,
        first_packet=p1, second_packet=p2))


@pytest.mark.parametrize("anchor", ["expected_first_plan_sha256", "expected_second_plan_sha256",
                                  "expected_first_packet_sha256", "expected_second_packet_sha256"])
def test_pair_external_raw_digest_anchor_is_not_derived_from_supplied_receipt(anchor):
    plan, c1, c2, p1, p2, r1, r2 = pair_evidence()
    refuses("packet_digest_mismatch", lambda: pair(r1, c1, r2, c2,
        first_plan=plan, second_plan=plan, first_packet=p1, second_packet=p2, **{anchor: "f" * 64}))


@pytest.mark.parametrize("change", ["swap_packets", "swap_bindings", "alter_bytes", "swap_candidates",
                                  "missing_with_pin", "wrong_code", "overcap_plan", "overcap_packet"])
def test_pair_replay_swapped_missing_tampered_or_overcap_inputs(change):
    plan, c1, c2, p1, p2, r1, r2 = pair_evidence()
    kwargs = dict(first_plan=plan, second_plan=plan, first_packet=p1, second_packet=p2)
    code = "frame_binding_mismatch"
    if change == "swap_packets":
        kwargs.update(first_packet_raw=p2.raw, expected_first_packet_sha256=p2.sha256,
                      second_packet_raw=p1.raw, expected_second_packet_sha256=p1.sha256)
    elif change == "swap_bindings":
        kwargs.update(expected_first_binding=c2.binding, expected_second_binding=c1.binding)
    elif change == "alter_bytes":
        kwargs.update(first_packet_raw=p1.raw + b" ")
        code = "packet_digest_mismatch"
    elif change == "swap_candidates":
        c1, c2 = c2, c1
    elif change == "missing_with_pin":
        kwargs.update(first_packet_raw=None)
        code = "observation_binding_mismatch"
    elif change == "wrong_code":
        kwargs.update(expected_validator_sha256="f" * 64)
        code = "validator_code_mismatch"
    elif change == "overcap_plan":
        kwargs.update(first_plan_raw=b" " * (qa.MAX_PACKET_BYTES + 1))
        code = "packet_byte_cap"
    else:
        kwargs.update(first_packet_raw=b" " * (qa.MAX_PACKET_BYTES + 1))
        code = "packet_byte_cap"
    refuses(code, lambda: pair(r1, c1, r2, c2, **kwargs))


def test_pair_has_no_qualifying_receipt_only_api():
    _, c1, c2, _, _, r1, r2 = pair_evidence()
    with pytest.raises(TypeError):
        qa.assess_pair(r1, c1, r2, c2)


def test_pair_replays_each_camera_once_and_cap_refuses_before_assessment(monkeypatch):
    plan, c1, c2, p1, p2, r1, r2 = pair_evidence()
    original = qa.assess_camera
    calls = []
    def replay(*args, **kwargs):
        calls.append(kwargs["expected_binding"].frame_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(qa, "assess_camera", replay)
    result = pair(r1, c1, r2, c2, first_plan=plan, second_plan=plan, first_packet=p1, second_packet=p2)
    assert result.status == "PASS" and calls == [c1.binding.frame_id, c2.binding.frame_id]
    calls.clear()
    refuses("packet_byte_cap", lambda: pair(r1, c1, r2, c2,
        first_plan=plan, second_plan=plan, first_packet=p1, second_packet=p2,
        first_packet_raw=b" " * (qa.MAX_PACKET_BYTES + 1)))
    assert calls == []


def test_production_source_count_and_manifest_are_restored():
    root = _PATH.parents[2]
    assert not (root / "src/mc_photogrammetry/dataset/camera_qualification.py").exists()
    rows = []
    for path in sorted((root / "src/mc_photogrammetry").rglob("*.py")):
        source = path.read_bytes()
        rows.append(dict(path=path.relative_to(root).as_posix(), bytes=len(source), sha256=digest(source)))
    assert len(rows) == 168
    assert digest(raw(rows)) == "5c341d990bf78da39af208e3fc3b8a3a956e404050e975b6e288ace5004efa35"


def test_no_production_camera_import_or_image_world_loader_in_module():
    import ast
    tree = ast.parse(_PATH.read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.startswith("mc_photogrammetry") for name in modules)
    assert not any(isinstance(node, ast.Import) and any(alias.name in ("numpy", "cv2", "PIL")
                                                     for alias in node.names) for node in ast.walk(tree))
