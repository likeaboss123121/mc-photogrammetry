"""DS-29 entry-point audit: the new plane/lattice modules stay image-only.

This file is the G8 boundary proof for the bootstrap package. It does three
things the other DS-29 tests do not:

* a **source audit** over the two new module files (AST imports plus a raw token
  scan) showing that no manifest, scene transform, litematic, reference,
  evaluation mask, v4 annotation or evaluator accessor is named;
* a **poisoned-access exercise**: hostile replacements for the manifest/scene
  loaders and for file reads under a forbidden path are installed and the graph
  controls are re-run, which must be bit-identical;
* an **import audit** in a clean subprocess showing the reference/evaluator
  modules are not even loaded.

No Luna annotation and no reference artifact is read here.
"""
from __future__ import annotations

import ast
import builtins
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(SRC),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import test_plane_graph as graph_helpers  # noqa: E402  (shared DS-29 controls)

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph  # noqa: E402

MODULE_PATHS = (
    SRC / "mc_photogrammetry" / "reconstruction" / "lattice_planes.py",
    SRC / "mc_photogrammetry" / "reconstruction" / "plane_graph.py",
)

FORBIDDEN_IDENTIFIERS = (
    "load_camera_manifest",
    "load_scene_transform",
    "litematic",
    "luna_annotation",
    "attribution",
    "visibility",
    "oracle",
    "truth",
    "reference_class",
    "expected_class",
    "mask_debug_area",
    "usable_mask_for",
    "render_volume",
    "_evaluate_pose",
    "_reference_class",
)

FORBIDDEN_STRING_PATTERNS = (
    ".masks",
    "cameras.json",
    "scene.json",
    "M4-C2-boundary-targets-v4",
    ".litematic",
    "reference.litematic",
)

FORBIDDEN_MODULE_MARKERS = (
    "diagnose_match_attribution",
    "diagnose_connected_pairs",
    "diagnose_oracle_pose_failures",
    "diagnose_blind_pose_policy",
    "diagnose_ds24",
    "diagnose_ds25",
    "diagnose_plane_lattice_pose",
    "blind_pose_policy",
)


def _source(name: str) -> str:
    return name.read_text(encoding="utf-8")


def _identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def test_new_module_sources_name_no_forbidden_accessor():
    for path in MODULE_PATHS:
        tree = ast.parse(_source(path))
        names = _identifiers(tree)
        for token in FORBIDDEN_IDENTIFIERS:
            assert token not in names, f"{path.name} uses the identifier {token!r}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in FORBIDDEN_STRING_PATTERNS:
                    assert pattern not in node.value, f"{path.name} embeds {pattern!r}"


def test_new_module_imports_are_image_only():
    for path in MODULE_PATHS:
        tree = ast.parse(_source(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(alias.name for alias in node.names)
        joined = " ".join(imported)
        for token in FORBIDDEN_IDENTIFIERS:
            assert token.lower() not in joined.lower(), f"{path.name} imports {token!r}"
        assert "numpy.random" not in joined
        assert "random" not in joined.split()


def test_new_modules_use_no_random_generator():
    for path in MODULE_PATHS:
        text = _source(path)
        assert "default_rng" not in text
        assert "np.random" not in text
        assert "import random" not in text


def test_importing_the_new_modules_loads_no_reference_or_evaluator_module():
    program = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import mc_photogrammetry.reconstruction.lattice_planes as lp;"
        "import mc_photogrammetry.reconstruction.plane_graph as pg;"
        "bad=[m for m in sys.modules if any(k in m for k in {markers})];"
        "print('BAD' if bad else 'CLEAN');"
    ).format(src=str(SRC), markers=repr(list(FORBIDDEN_MODULE_MARKERS)))
    completed = subprocess.run([sys.executable, "-c", program],
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("CLEAN"), completed.stdout


def test_poisoned_manifest_scene_and_file_access_cannot_change_a_solution(monkeypatch):
    readings, matches, _eyes = graph_helpers.lattice_component()
    baseline = plane_graph.solve_component(readings, matches)

    class _Poison(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise _Poison("forbidden reference/manifest access attempted")

    import mc_photogrammetry.reconstruction.camera as camera_module
    import mc_photogrammetry.reconstruction.scene as scene_module

    monkeypatch.setattr(camera_module, "load_camera_manifest", _boom, raising=False)
    monkeypatch.setattr(scene_module, "load_scene_transform", _boom, raising=False)

    real_read_text = Path.read_text

    def _guarded_read_text(self, *args, **kwargs):
        lowered = str(self).lower()
        if any(token in lowered for token in (".masks", "litematic", "cameras.json",
                                              "scene.json", "targets-v4")):
            raise _Poison(f"forbidden read of {self}")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read_text)
    real_open = builtins.open

    def _guarded_open(file, *args, **kwargs):
        lowered = str(file).lower()
        if any(token in lowered for token in (".masks", "litematic", "cameras.json",
                                              "scene.json", "targets-v4")):
            raise _Poison(f"forbidden open of {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)

    repeated = plane_graph.solve_component(readings, matches)
    assert repeated.reason == baseline.reason
    assert dict(repeated.quadrants) == dict(baseline.quadrants)
    for frame in baseline.frames:
        assert repeated.translations[frame] == pytest.approx(baseline.translations[frame])


def test_class_constrained_bestof_is_deterministic_and_one_to_one():
    neighbors, features, count = _synthetic_features()
    first = plane_graph.class_constrained_bestof(features, neighbors, 1)
    second = plane_graph.class_constrained_bestof(features, neighbors, 1)
    assert np.array_equal(first, second)
    left = [tuple(row[:2]) for row in first]
    right = [tuple(row[2:]) for row in first]
    assert len(set(left)) == len(first)
    assert len(set(right)) == len(first)
    assert len(first) >= 8


def test_resolved_class_margin_flags_an_ambiguous_component():
    eye = {"v0": (5.0, 2.0, 5.0), "v1": (6.0, 3.0, 7.0)}
    base = graph_helpers.make_reading("v0", 1, -1.0, [(1.0, 2.0, 1.0)])
    other = graph_helpers.make_reading("v1", 1, -1.0, [(1.0, 3.0, 1.0)])
    matches = [
        graph_helpers.make_match("v0", "v1", 1, 1, 0, 0, 0),
        graph_helpers.make_match("v0", "v1", 1, 1, 2, 0, 0),
    ]
    solution = plane_graph.solve_component([base, other], matches)
    assert solution.reason == "ambiguous_quadrant"
    _quadrant, margin = plane_graph.resolved_class(solution, "v1", {"v0": object(), "v1": object()})
    assert margin < 0.25
    assert eye  # the eye mapping documents the geometry used above


def test_public_entry_points_keep_their_declared_arity():
    assert list(inspect.signature(lattice_planes.frame_plane_readings).parameters)[:3] == \
        ["image", "usable", "calibration_result"]
    assert list(inspect.signature(lattice_planes.plane_signature).parameters)[:3] == \
        ["image", "reading", "blocks_per_period"]
    assert list(inspect.signature(plane_graph.match_plane_pair).parameters)[:2] == \
        ["left", "right"]
    assert list(inspect.signature(plane_graph.match_plane_graph).parameters)[:2] == \
        ["readings", "signatures"]
    assert list(inspect.signature(plane_graph.solve_component).parameters)[:2] == \
        ["readings", "matches"]


def _synthetic_features():
    from mc_photogrammetry.reconstruction.camera import (
        CameraModel,
        camera_rotation_matrix,
        intrinsics_for,
        project_points,
    )

    width, height = 720, 480
    left_model = CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0,
                             fov_degrees=70.0)
    right_model = CameraModel(position=(3.0, 0.0, 1.0), yaw_degrees=90.0, pitch_degrees=0.0,
                              fov_degrees=70.0)
    right_calibrated = CameraModel(position=(3.0, 0.0, 1.0), yaw_degrees=0.0,
                                   pitch_degrees=0.0, fov_degrees=70.0)
    rng = np.random.default_rng(29)
    points = rng.uniform(-6.0, 6.0, (24, 3)) + np.asarray((0.0, 0.0, 20.0))
    left = project_points([tuple(point) for point in points], left_model, width=width, height=height)
    right = project_points([tuple(point) for point in points], right_model, width=width, height=height)
    paired = [(l[:2], r[:2]) for l, r in zip(left, right) if l is not None and r is not None][:18]
    left_points = np.asarray([pair[0] for pair in paired], dtype=float)
    right_points = np.asarray([pair[1] for pair in paired], dtype=float)
    count = len(paired)
    left_intrinsics = intrinsics_for(left_model, width, height)
    right_intrinsics = intrinsics_for(right_model, width, height)
    edges = [(index, index) for index in range(count)]
    for index in range(count):
        if index % 4 == 0:
            edges.append((index, (index + 5) % count))
    neighbors = graph_helpers._Neighbors(
        np.asarray(edges, dtype=np.int64), np.asarray([0.1] * len(edges), dtype=float))
    features = plane_graph.ClassConstrainedFeatures(
        left_keypoints=left_points, right_keypoints=right_points,
        left_rays=_rays(left_points, left_intrinsics),
        right_rays=_rays(right_points, right_intrinsics),
        rotation_left=camera_rotation_matrix(left_model),
        rotation_right=camera_rotation_matrix(right_calibrated))
    return neighbors, features, count


def _rays(points, intrinsics):
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (points[:, 0] - centre_x) / focal_x
    y = -(points[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(points))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)
