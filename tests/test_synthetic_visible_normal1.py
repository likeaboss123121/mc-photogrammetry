"""Isolated generated renderer/default-camera/proposal controls; no production."""
from pathlib import Path
import ast
import importlib.util

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("visible_normal1_control", ROOT /
    "scripts/evaluation/probe_synthetic_visible_normal1.py")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    folder = tmp_path_factory.mktemp("visible-roof") / "fresh-control"
    original_fit, original_proposal = helper.cal.estimate_camera, helper.rg.propose_image_planes_v2
    calls = {"camera": 0, "proposal": 0}
    def estimate(grey, *args, **kwargs):
        assert isinstance(grey, np.ndarray) and grey.shape == (helper.HEIGHT, helper.WIDTH)
        assert args == () and kwargs == {}  # no oracle camera, focal, scale or injected segments
        calls["camera"] += 1
        return original_fit(grey)
    def propose(rgb, usable, fit, *args, **kwargs):
        assert args == () and kwargs == {}  # every source proposal default retained
        assert rgb.shape == (helper.HEIGHT, helper.WIDTH, 3)
        assert usable.dtype == np.bool_
        calls["proposal"] += 1
        return original_proposal(rgb, usable, fit)
    original_graph = helper.rg._build_graph
    try:
        helper.cal.estimate_camera = estimate
        helper.rg.propose_image_planes_v2 = propose
        result = helper.run_controls(folder)
    finally:
        helper.cal.estimate_camera = original_fit
        helper.rg.propose_image_planes_v2 = original_proposal
    assert helper.rg._build_graph is original_graph
    return result, calls


@pytest.mark.parametrize("index", [0, 1, 2])
def test_independent_oracle_has_both_seam_families_in_one_usable_unoccluded_patch(control, index):
    result, _ = control
    row = result["views"][index]
    oracle = row["oracle"]
    assert oracle["unoccluded_by_convex_box"] and oracle["renderer_face_binding"]
    assert oracle["oracle_qualifies"]
    assert {r["tangent_axis"] for r in oracle["seam_families"]} == {0, 2}
    assert all(len(r["contiguous_usable_extents_pixels"]) == 3 for r in oracle["seam_families"])
    assert all(min(r["contiguous_usable_extents_pixels"]) >= helper.cal.MINIMUM_LENGTH for r in oracle["seam_families"])
    assert min(oracle["stripe_width_pixels"]) >= 2
    assert min(oracle["one_unit_cell_pitch_pixels"]) > 2 * max(oracle["stripe_width_pixels"])
    assert oracle["usable_oracle_patch_pixels"] <= oracle["oracle_patch_pixels"]
    assert row["pixels_unchanged"]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_same_pass_default_estimator_and_proposal_metrics_are_honest(control, index):
    result, _ = control
    measured = result["views"][index]["estimated_default_control"]
    if measured["stage"] == "camera":
        assert measured["status"] == "camera_estimate_unavailable" and not measured["qualifies"]
        return
    assert measured["same_pass_build_graph_calls"] == 1
    assert measured["camera_inliers"] == sum(measured["camera_axis_inliers"])
    assert measured["camera_inliers"] <= measured["camera_segments"]
    assert len(measured["seed_predicates"]) == 6 and measured["observer_complete"]
    assert measured["normal1_eligible_seeds"] == sum(
        (r["eligible_seed_count"] or 0) for r in measured["seed_predicates"] if r["axis"] == 1)
    assert measured["qualifies"] == (measured["normal1_eligible_seeds"] > 0 and measured["proposal_normal_counts"][1] > 0)
    assert measured["support"]["roof_patch_atoms"] > 0
    assert len(measured["support"]["tangent_support_mass"]) == 2
    assert "pixels" not in measured and "endpoints" not in measured


def test_only_fresh_png_inputs_and_qualified_packet_decision_no_production(control):
    result, calls = control
    assert calls["camera"] == 3
    assert calls["proposal"] == sum(r["estimated_default_control"]["stage"] == "proposal" for r in result["views"])
    assert result["production_invocations"] == 0 and not result["oracle_passed_to_estimator_or_proposal"]
    assert result["qualifying_view_count"] == sum(r["qualifies"] for r in result["views"])
    assert result["packet_eligible"] == (result["qualifying_view_count"] >= 2 and calls["proposal"] == 3)
    directory = Path(result["views"][0]["path"]).parent
    assert sorted(p.suffix for p in directory.iterdir()) == [".png"] * 3
    record = dict(result); claimed = record.pop("result_sha256")
    assert helper.digest(record) == claimed
    record["views"] = list(record["views"]); record["qualifying_view_count"] += 1
    assert helper.digest(record) != claimed


@pytest.mark.parametrize("index", [0, 1, 2])
def test_oracle_projection_uses_actual_explicit_eye_and_camera_basis(index):
    camera = helper.render_camera(index)
    np.testing.assert_array_equal(helper.camera_eye(camera), helper.EYES[index])
    points = np.array([[3., 4., 3.], [5., 4., 3.], [3., 4., 5.]])
    projected, depth = helper.oracle_project(index, points)
    forward = (points - helper.camera_eye(camera)) @ helper.camera_rotation_matrix(camera).T
    fx, fy = helper.intrinsics_for(camera, helper.WIDTH, helper.HEIGHT).focal_length_pixels
    expected = np.column_stack((helper.WIDTH / 2 + fx * forward[:, 0] / forward[:, 2],
                                helper.HEIGHT / 2 - fy * forward[:, 1] / forward[:, 2]))
    np.testing.assert_allclose(projected, expected, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(depth, forward[:, 2], rtol=1e-13)


def test_false_render_face_binding_cannot_qualify_oracle():
    faces = np.zeros((helper.HEIGHT, helper.WIDTH), dtype=np.int8)
    oracle, _ = helper.roof_oracle(0, faces, np.ones_like(faces, dtype=bool))
    assert not oracle["renderer_face_binding"] and not oracle["oracle_qualifies"]


def test_generator_contains_no_production_route_or_model_truth_adapter():
    tree = ast.parse(Path(helper.__file__).read_bytes())
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("image_derived_production" in name or "session" in name for name in imports)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("derive_image")
                   for n in ast.walk(tree))


@pytest.mark.parametrize("hash_case", ["lower", "upper", "mixed"])
@pytest.mark.parametrize("mismatch", [None, "hash", "bytes"])
def test_frozen_packet_actual_normalized_hash_guard_authored_bytes(tmp_path, hash_case, mismatch):
    """Execute only the actual driver pin loop, never its imports or production."""
    import hashlib
    driver = ROOT / "agent swarm instantiation template/runs/SOL-SYNTHETIC-VISIBLE-NORMAL1-CONTROL-01.amendment02.driver.py"
    tree = ast.parse(driver.read_bytes())
    loops = [node for node in tree.body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Tuple)
             and [item.id for item in node.target.elts] == ["row", "path"]]
    assert len(loops) == 1
    guard = loops[0]
    # This subset contains only read_bytes, hash/length comparison and assert.
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(guard))
    assert not any(isinstance(node, ast.Name) and node.id == "derive_image_output_from_paths"
                   for node in ast.walk(guard))
    payload = b"authored hash-case control; no PNG or production input"
    authored = tmp_path / "pin.bin"
    authored.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    if hash_case == "upper": expected = expected.upper()
    if hash_case == "mixed": expected = "".join(c.upper() if i % 2 else c for i, c in enumerate(expected))
    row = {"bytes": len(payload), "sha256": expected}
    if mismatch == "hash": row["sha256"] = "0" * 64
    if mismatch == "bytes": row["bytes"] += 1
    code = compile(ast.Module(body=[guard], type_ignores=[]), "<frozen-packet-hash-guard>", "exec")
    namespace = {"hashlib": hashlib, "pins": [row], "paths": [authored]}
    if mismatch is None:
        exec(code, namespace)
    else:
        with pytest.raises(AssertionError): exec(code, namespace)
