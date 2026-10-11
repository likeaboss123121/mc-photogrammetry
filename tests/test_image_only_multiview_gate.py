"""One preregistered generated two-view pair through the normal production API."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from test_image_derived_p0_pixel_bridge import PRIOR, _profiles
from test_lattice_planes import ROOM, mixed_material, render_room
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.project.store import load_project
from mc_photogrammetry.reconstruction.camera import CameraModel


ROOT = Path(__file__).resolve().parents[1]
PINS = dict(PRIOR, **{
    "tests/test_image_derived_p0_pixel_bridge.py":
        "e800dea4f23b11c7f4d65b4944b3e2b90fb0d1b447991243072eb05dfb174d2b",
    "docs/reports/SOL-IMG-P0-PIXEL-BRIDGE-20261005.json":
        "81a3d305a5a1683f8af875c7024c7e3a6f907635c3a33c2692d396045f1770a1",
    "docs/reports/SOL-IMG-P0-PIXEL-BRIDGE-20261005.txt":
        "6df91d75e7a8c167c421443a7d0b7e34fc9f8ae310e4d3b4c8b729cb5f159c31",
})
# Generator truth only: these values are never arguments to production.
VIEWS = (((3.6, .7, 3.6), 10., -25.), ((5., .7, 5.), 10., -25.))


def _file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _record(value):
    if hasattr(value, "to_record"):
        return value.to_record()
    module = importlib.import_module(type(value).__module__)
    payload = getattr(module, "_payload", None) or getattr(module, "_result_payload", None)
    assert payload is not None, type(value)
    result = payload(value)
    for field in ("result_sha256", "entry_sha256", "graph_sha256"):
        if hasattr(value, field):
            result[field] = getattr(value, field)
    return result


def test_default_image_only_multiview_gate(tmp_path):
    assert {path: _file_sha(path) for path in PINS} == PINS
    source_paths = ["src/mc_photogrammetry/" + name + ".py" for name in (
        "pipeline/image_derived_production", "pipeline/image_derived_entry",
        "pipeline/image_derived_geometry_orchestration", "pipeline/image_derived_output",
        "pipeline/image_only", "pipeline/image_derived_session",
        "pipeline/image_derived_connections", "pipeline/image_derived_pose_graph",
        "pipeline/image_derived_lattice_measure", "pipeline/image_derived_signatures",
        "pipeline/image_derived_graph", "reconstruction/calibration",
        "reconstruction/lattice_planes", "reconstruction/plane_graph")]
    source_pins = {path: _file_sha(path) for path in source_paths}
    source_pins.update({path: _file_sha(path) for path in (
        "tests/test_lattice_planes.py", "tests/test_image_derived_lattice_measure.py")})
    camera_solver = image_only.solve_image_cameras
    pair_matcher = entry.match_and_solve_pair
    assert camera_solver.__module__ == "mc_photogrammetry.pipeline.image_only"
    inputs, rendered = [], []
    for index, (position, yaw, pitch) in enumerate(VIEWS):
        camera = CameraModel(position=position, yaw_degrees=yaw,
                             pitch_degrees=pitch, fov_degrees=70.)
        grey = render_room(camera, 360, 240, ROOM, materials=mixed_material)
        rgb = np.repeat(grey[..., None], 3, axis=2)
        path = tmp_path / f"view{index}.png"
        Image.fromarray(rgb).save(path)
        inputs.append(path)
        rendered.append({"filename": path.name, "rgb_sha256": sessions.image_sha256(rgb),
                         "encoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "shape": list(rgb.shape)})
    assert rendered[0]["rgb_sha256"] != rendered[1]["rgb_sha256"]
    target = tmp_path / "multiview.mcscan"
    # Public normal-mode CLI boundary, with DEFAULT matcher, camera solver,
    # policies, caps and gates. It receives only generated PNG paths + output path.
    outcome = production.derive_image_output_from_paths(inputs, project_path=target)
    assert isinstance(outcome, production.ImageDerivedProductionResult), outcome
    assert len(outcome.entry.session.frames) == 2
    assert outcome.entry.graph.pair_count == len(outcome.entry.graph.evidence) == 1
    assert outcome.entry.session.scale_known is False
    assert image_only.solve_image_cameras is camera_solver
    assert entry.match_and_solve_pair is pair_matcher
    persisted = load_project(target)
    assert persisted.schema_version == 3
    assert persisted.image_derived_session == outcome.entry.session_record
    assert persisted.image_derived_production["status"] == outcome.status
    stages = [] if outcome.geometry is None else [
        {"stage": stage, "record": _record(result)}
        for stage, result in outcome.geometry.records]
    profiles = [] if outcome.lattice is None else _profiles(
        outcome.lattice, outcome.entry.session)
    proof = {"scope": "one generated two-view pair; default public image-only production",
        "decision": "abstain", "prior_pins": PINS, "source_pins": source_pins,
        "input_files": rendered, "generator_truth_posthoc_only": {
            "room": ROOM, "views": VIEWS, "fov_degrees": 70.,
            "material_rule": "(floor(u)*3 + floor(v)*5) mod 3",
            "truth_was_production_input": False},
        "production": _record(outcome), "entry": _record(outcome.entry),
        "session": outcome.entry.session_record,
        "connections": outcome.entry.graph.to_record(),
        "pose": outcome.entry.pose_graph.to_record(),
        "lattice": None if outcome.lattice is None else outcome.lattice.to_record(),
        "p0_profiles": profiles,
        "geometry": None if outcome.geometry is None else _record(outcome.geometry),
        "geometry_stages": stages,
        "output": None if outcome.output is None else _record(outcome.output),
        "project": {"schema_version": persisted.schema_version,
            "image_derived_production": persisted.image_derived_production,
            "blocks": len(persisted.volume.blocks),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "metrics": {"geometry_accuracy": None, "exact_state_accuracy": None},
        "export": "not called; no supported promoted geometry/state output",
        "limits": "two fixed 360x240 RGB frames and one pair; production default policies; no hard runtime containment"}
    print("MULTIVIEW_OUTCOME", json.dumps({"stage": outcome.stage,
        "status": outcome.status, "reason": outcome.reason, "detail": outcome.detail,
        "pair": outcome.entry.graph.to_record()["evidence"],
        "geometry_stages": stages}, sort_keys=True))
    assert outcome.status == "geometry_unavailable"
    assert not persisted.volume.blocks
    assert outcome.output is None
    assert {path: _file_sha(path) for path in PINS} == PINS
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    encoded = json.dumps(proof, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    assert len(encoded) < 2_000_000  # cooperative publication guard only
    artifact = tmp_path / "multiview-gate.json"
    artifact.write_bytes(encoded)
    print("MULTIVIEW_ARTIFACT", str(artifact), len(encoded),
          hashlib.sha256(encoded).hexdigest())
