"""Focused A1 tests: the image-only route frontier record contract.

This module tests the evaluation-only A1 implementation of
``docs/reports/image-only-loss-attribution-design-2026-10-03.md`` section 5.  The
tests are synthetic and self-contained: they read no image, mask, reference,
manifest or JAR, they never import ``cli``, ``project.store`` or any production
orchestration, and they produce no visible-truth number.

Coverage:

* module isolation (forbidden imports, IO, exporter and reconstruction calls);
* adapter signatures carry no truth/reference/alignment/path parameter;
* the five labelled spaces (G, O, P, C, E) plus the camera diagnostic;
* stage prefixes for both routes and the fail-closed typed-unavailable contract
  (no zero is ever emitted for a missing record);
* deterministic serialization and digest binding;
* tampered-record refusal that builds no funnel;
* typed route-B failures (lattice/geometry that do not bind, a terminal record
  that contradicts the carried prefix);
* a synthetic recipe that exercises the real public volume/proposal/promotion
  tail constructors while explicitly labelling its route-B envelope records as
  synthetic.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_image_only_frontier as frontier
from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export import compare as compare_mod
from mc_photogrammetry.pipeline import image_derived_production_record as record_mod
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals_mod
from mc_photogrammetry.reconstruction import plane_face_observation as face_mod
from mc_photogrammetry.reconstruction import plane_volume as volume_mod
from mc_photogrammetry.reconstruction import state_promotion as promotion_mod
from mc_photogrammetry.reconstruction.face_observation import (
    NORMAL_OF_SLOT,
    FaceObservation,
)

SOURCE = Path(frontier.__file__)
REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_B_RUNNER = REPO_ROOT / "scripts" / "evaluation" / "run_image_derived_frontier.py"
ROUTE_A_RUNNER = REPO_ROOT / "scripts" / "evaluation" / "run_image_only_route_a.py"

FORBIDDEN_SIGNATURE_TOKENS = (
    "path", "manifest", "reference", "visible_reference", "truth", "alignment",
    "coordinate", "bounds",
)
FORBIDDEN_SOURCE_TOKENS = (
    "open(", "Path", "LitematicExporter", "load_litematic", "load_image",
    "image_derived_production", "import cli", "project.store", "argparse",
    "subprocess", "resolve_face_placement", "build_plane_volume",
    "propose_face_states", "promote_plane_states",
)
#: The Route A adapter's only permitted production-record parameters.
ROUTE_A_RECORD_PARAMETERS = frozenset(
    {"admission", "camera", "facade", "palette", "decoded_export"})

_HEX = {name: character * 64 for name, character in (
    ("entry", "a"), ("session", "b"), ("lattice", "c"), ("geometry", "d"),
    ("output", "e"))}


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------

class _Volume:
    """A minimal block-volume shaped record with a deterministic digest."""

    def __init__(self, blocks):
        self.blocks = dict(blocks)

    def to_dict(self):
        return json.dumps(
            {"blocks": [[list(key), str(value)]
                        for key, value in sorted(self.blocks.items())]},
            sort_keys=True, separators=(",", ":"))


class _Undigestible:
    """A record that carries no accepted digest at all."""


def _frame(entry=("a", "b"), palette=((0, 0, 0), (255, 255, 255))):
    slots = ("east", "west", "up", "down", "south", "north")
    return SimpleNamespace(
        state=BlockState("minecraft:stone"), rgb=entry,
        face_rgbs=(entry,), family="cube",
        face_map=tuple((slot, entry) for slot in slots), full_cube=True)


def _entry_record(*, status="ok", reason="", frames=3):
    return SimpleNamespace(
        entry_sha256=_HEX["entry"], status=status, reason=reason,
        session=SimpleNamespace(session_sha256=_HEX["session"],
                                frames=tuple(range(frames))))


def _lattice_record(*, status="assembled", reason="", detail="lattice detail"):
    return SimpleNamespace(
        status=status, reason=reason, detail=detail,
        result_sha256=_HEX["lattice"], session_sha256=_HEX["session"])


def _geometry_record(*, status="assembled", reason="", detail="plane_integer_residual"):
    return SimpleNamespace(
        status=status, reason=reason, detail=detail,
        result_sha256=_HEX["geometry"], lattice_sha256=_HEX["lattice"])


def _output_record(*, status="review", reason="proposals_review",
                   detail="proposal detail"):
    return SimpleNamespace(
        status=status, reason=reason, detail=detail,
        result_sha256=_HEX["output"], geometry_sha256=_HEX["geometry"])


def _production_result(*, stage, status, reason, detail="",
                       entry_sha256=_HEX["entry"],
                       lattice_sha256=_HEX["lattice"],
                       geometry_sha256=_HEX["geometry"],
                       output_sha256=_HEX["output"]):
    return SimpleNamespace(
        schema="mc-photogrammetry-image-derived-production", producer="producer",
        stage=stage, status=status, reason=reason, detail=detail,
        entry_sha256=entry_sha256, lattice_sha256=lattice_sha256,
        geometry_sha256=geometry_sha256, output_sha256=output_sha256,
        result_sha256=_HEX["output"])


def _full_route_b(**overrides):
    records = {
        "entry": _entry_record(),
        "lattice": _lattice_record(),
        "geometry": _geometry_record(),
        "output": _output_record(),
    }
    records.update(overrides)
    return records


def _route_a_records(*, facade_blocks=(((0, 0, 0), "minecraft:stone"),),
                     with_camera=True):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    camera = None
    if with_camera:
        camera = SimpleNamespace(position=(0.0, 0.0, 0.0), yaw_degrees=1.5,
                                 pitch_degrees=-2.0, roll_degrees=0.0,
                                 fov_degrees=70.0)
    facade = _Volume(facade_blocks)
    palette = (_frame(),)
    decoded = _Volume(facade_blocks)
    return {"admission": image, "camera": camera, "facade": facade,
            "palette": palette, "decoded_export": decoded}


def _rebuild_dataclass(record, **changes):
    """Copy a frozen record past its validating constructor."""
    clone = object.__new__(type(record))
    for field in fields(record):
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _blocked(frontier_record):
    return [item.blocked for item in frontier_record.checkpoints]


# ---------------------------------------------------------------------------
# Module isolation and adapter signatures
# ---------------------------------------------------------------------------

def test_module_has_no_io_no_production_orchestration_and_no_reconstruction_calls():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("mc_photogrammetry.cli", "mc_photogrammetry.project",
                      "mc_photogrammetry.pipeline", "mc_photogrammetry.export",
                      "mc_photogrammetry.reconstruction", "pathlib", "os",
                      "tempfile"):
        assert not any(name == forbidden or name.startswith(forbidden + ".")
                       for name in imported), (forbidden, imported)
    # The module docstring names the forbidden concepts in order to forbid them;
    # the executed code must not reference them.  Walk real identifiers, not text.
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name)
    for token in ("LitematicExporter", "load_litematic", "load_image",
                  "image_derived_production", "argparse",
                  "subprocess", "resolve_face_placement", "build_plane_volume",
                  "propose_face_states", "promote_plane_states", "open",
                  "Path"):
        assert token not in identifiers, token


def test_adapters_take_no_truth_reference_alignment_path_or_manifest_parameter():
    for name in ("route_a_frontier", "route_b_frontier", "frontier_unavailable",
                 "first_observable_failure"):
        signature = inspect.signature(getattr(frontier, name))
        for parameter in signature.parameters:
            lowered = parameter.lower()
            assert not any(token in lowered
                           for token in FORBIDDEN_SIGNATURE_TOKENS), (name, parameter)
    route_a = set(inspect.signature(frontier.route_a_frontier).parameters)
    route_b = set(inspect.signature(frontier.route_b_frontier).parameters)
    assert route_a & ROUTE_A_RECORD_PARAMETERS == ROUTE_A_RECORD_PARAMETERS
    assert "admission" not in FORBIDDEN_SIGNATURE_TOKENS
    assert not any(token in name for name in route_b
                   for token in FORBIDDEN_SIGNATURE_TOKENS), route_b


def _load_runner(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_scripts_import_and_expose_run_and_main():
    for name, path in (("_a1_route_b_runner", ROUTE_B_RUNNER),
                       ("_a1_route_a_runner", ROUTE_A_RUNNER)):
        assert path.is_file(), path
        module = _load_runner(name, path)
        assert callable(module.run)
        assert callable(module.main)
        assert module.FRONTIER_PRODUCER == frontier.FRONTIER_PRODUCER


def test_route_a_runner_defaults_mirror_the_normal_cli_parameters():
    """The runner's defaults are the normal-mode `reconstruct` defaults.

    ``cli.py:94-101`` gives the normal-mode ``reconstruct`` command
    ``--columns 32 --rows 32 --depth-layers 8``; the runner must default to the
    same, so a frozen CLI-vs-runner parity run needs no parameter overrides.
    """
    runner = _load_runner("_a1_route_a_runner_defaults", ROUTE_A_RUNNER)
    args = runner.build_parser().parse_args(["--image", "frozen.png"])
    assert (args.columns, args.rows, args.depth_layers) == (32, 32, 8)
    assert args.palette_jar is None


class _State:
    """A minimal block-observation stand-in carrying one exact state."""

    def __init__(self, name):
        self.state = BlockState(name)


class _RunnerVolume:
    """A minimal ``BlockVolume``-shaped record for the Route-A runner summary."""

    def __init__(self, blocks):
        self.blocks = dict(blocks)


def _route_a_run_outcome(runner, *, with_export):
    records = _route_a_records()
    if with_export:
        facade = _RunnerVolume({(0, 0, 0): _State("minecraft:stone")})
        decoded = _RunnerVolume({(0, 0, 0): _State("minecraft:stone")})
        blocked = ""
    else:
        facade = None
        decoded = None
        blocked = "export_unavailable"
    route_a = frontier.route_a_frontier(
        admission=records["admission"], camera=records["camera"],
        facade=facade, palette=records["palette"], decoded_export=decoded,
        camera_solved=records["camera"] is not None, blocked_reason=blocked)
    return runner.RouteARun(
        route_a=route_a, image=records["admission"], mask=None,
        camera=records["camera"], facade=facade, palette=records["palette"],
        decoded_export=decoded, project_path=Path("route-a.mcscan"),
        litematic_path=Path("route-a.litematic"))


def test_route_a_runner_summary_uses_the_actual_terminal_and_nulls_no_export(
        monkeypatch):
    runner = _load_runner("_a1_route_a_runner_summary", ROUTE_A_RUNNER)
    # Export ran: the round trip is actually known.
    monkeypatch.setattr(
        runner, "replay",
        lambda *args, **kwargs: _route_a_run_outcome(runner, with_export=True))
    summary = runner.run("frozen.png")
    assert summary["terminal_stage"] == "terminal"
    assert summary["export_round_trip_exact"] is True
    # Export never ran (an earlier stage failed): the round trip is null, not a
    # false-positive exact result from comparing two empty maps.
    monkeypatch.setattr(
        runner, "replay",
        lambda *args, **kwargs: _route_a_run_outcome(runner, with_export=False))
    failed = runner.run("frozen.png")
    assert failed["export_round_trip_exact"] is None
    assert failed["terminal_stage"] != "terminal"
    assert failed["frontier"]["terminal_policy"]["stage"] != "terminal"


# ---------------------------------------------------------------------------
# Route-B typed PlacementRefusal attribution (no production run)
# ---------------------------------------------------------------------------

_REFUSAL_DIGEST = "a" * 64


def _route_b_runner():
    return _load_runner("_a1_route_b_runner_refusal", ROUTE_B_RUNNER)


def _refusal_outcome(*, reason="invalid_input", refusal_sha256=_REFUSAL_DIGEST):
    """A synthetic stand-in for a returned ``plane_placement.PlacementRefusal``."""
    return SimpleNamespace(reason=reason, refusal_sha256=refusal_sha256,
                           detail_sha256="b" * 64)


def _refusal_envelope(*, stage="geometry", reason="invalid_input",
                      refusal_sha256=_REFUSAL_DIGEST):
    """A real, digest-bound schema-3 refusal envelope (no project was written)."""
    return record_mod.build_production_envelope(
        stage=stage, status="refused", reason=reason, detail="b" * 64,
        entry_sha256="1" * 64, lattice_sha256="2" * 64,
        session_sha256="3" * 64, refusal_sha256=refusal_sha256)


def _non_refusal_envelope():
    """A real, digest-bound schema-3 *normal* terminal (status is not refused)."""
    return record_mod.build_production_envelope(
        stage="geometry", status="geometry_unavailable", reason="g_unavailable",
        detail="", entry_sha256="1" * 64, lattice_sha256="2" * 64,
        geometry_sha256="5" * 64, session_sha256="3" * 64)


def test_route_b_placement_refusal_attributes_its_bound_refusal_envelope(
        monkeypatch, tmp_path):
    runner = _route_b_runner()
    target = tmp_path / "route-b.mcscan"
    target.write_bytes(b"x")  # an existent path, so the runner inspects it
    envelope = _refusal_envelope(stage="geometry")
    monkeypatch.setattr(runner, "_persisted_envelope", lambda path: envelope)
    summary = runner._refusal_summary(_refusal_outcome(), target)
    assert summary["route"] == "route_b"
    assert summary["frontier_available"] is False
    assert summary["frontier"] is None
    assert summary["terminal_stage"] == "geometry"
    assert summary["terminal_status"] == "refused"
    assert summary["terminal_reason"] == "invalid_input"
    assert summary["terminal_detail"] == "b" * 64
    assert summary["refusal_reason"] == "invalid_input"
    assert summary["refusal_sha256"] == _REFUSAL_DIGEST
    assert summary["envelope_sha256"] == envelope.record_sha256


def test_route_b_placement_refusal_mismatched_digest_fails_closed(
        monkeypatch, tmp_path):
    runner = _route_b_runner()
    target = tmp_path / "route-b.mcscan"
    target.write_bytes(b"x")
    stale = _refusal_envelope(refusal_sha256="c" * 64)
    monkeypatch.setattr(runner, "_persisted_envelope", lambda path: stale)
    with pytest.raises(frontier.FrontierRefusal):
        runner._refusal_summary(
            _refusal_outcome(refusal_sha256=_REFUSAL_DIGEST), target)


def test_route_b_placement_refusal_mismatched_reason_fails_closed(
        monkeypatch, tmp_path):
    runner = _route_b_runner()
    target = tmp_path / "route-b.mcscan"
    target.write_bytes(b"x")
    other = _refusal_envelope(reason="budget_exhausted")
    monkeypatch.setattr(runner, "_persisted_envelope", lambda path: other)
    with pytest.raises(frontier.FrontierRefusal):
        runner._refusal_summary(
            _refusal_outcome(reason="invalid_input"), target)


def test_route_b_placement_refusal_non_refusal_envelope_fails_closed(
        monkeypatch, tmp_path):
    runner = _route_b_runner()
    target = tmp_path / "route-b.mcscan"
    target.write_bytes(b"x")
    monkeypatch.setattr(runner, "_persisted_envelope",
                        lambda path: _non_refusal_envelope())
    with pytest.raises(frontier.FrontierRefusal):
        runner._refusal_summary(_refusal_outcome(), target)


def test_route_b_placement_refusal_without_envelope_is_typed_unavailable(
        monkeypatch, tmp_path):
    runner = _route_b_runner()
    target = tmp_path / "route-b.mcscan"  # never written: no persisted envelope
    monkeypatch.setattr(
        runner, "_persisted_envelope",
        lambda path: pytest.fail("no envelope may be read for an absent target"))
    summary = runner._refusal_summary(_refusal_outcome(), target)
    assert summary["frontier_available"] is False
    assert summary["frontier"] is None
    assert summary["terminal_stage"] is None
    assert summary["terminal_status"] is None
    assert summary["terminal_reason"] is None
    assert summary["envelope_sha256"] is None
    assert summary["refusal_reason"] == "invalid_input"
    assert summary["refusal_sha256"] == _REFUSAL_DIGEST


# ---------------------------------------------------------------------------
# Contract: labelled spaces, fixed universes, no cell metric in A1
# ---------------------------------------------------------------------------

def test_five_labelled_spaces_are_named_and_never_interchanged():
    assert [item.value for item in frontier.RecordKind] == [
        "G", "O", "P", "C", "E", "camera"]
    assert {item.value for item in frontier.Universe} == {
        "frozen_reference_cells", "frozen_frame_pixels",
        "cell_slot_frame_face_observations", "frozen_pixel_pair_identities",
        "none"}
    route_b = frontier.route_b_frontier(**_full_route_b())
    universes = {item.stage: item.universe for item in route_b.checkpoints}
    assert universes["entry"] is frontier.Universe.PIXEL_PAIRS
    for stage in ("lattice", "geometry", "output"):
        assert universes[stage] is frontier.Universe.REFERENCE_CELLS
    route_a = frontier.route_a_frontier(**_route_a_records())
    route_a_universes = {item.stage: item.universe
                         for item in route_a.frontier.checkpoints}
    assert route_a_universes["admission"] is frontier.Universe.FRAME_PIXELS
    assert route_a_universes["camera"] is frontier.Universe.NONE
    for stage in ("facade_grid", "palette_label", "terminal"):
        assert route_a_universes[stage] is frontier.Universe.REFERENCE_CELLS


def test_every_checkpoint_is_typed_and_carries_no_cell_metric_in_a1():
    for record in (frontier.route_b_frontier(**_full_route_b()),
                   frontier.route_a_frontier(**_route_a_records()).frontier):
        for item in record.checkpoints:
            assert item.metric is None
            assert isinstance(item.kind, frontier.RecordKind)
            assert isinstance(item.universe, frontier.Universe)
            assert "universe" in item.to_record()
            assert item.to_record()["metric"] is None
    for record in (frontier.route_b_frontier(**_full_route_b()),
                   frontier.route_a_frontier(**_route_a_records()).frontier):
        assert record.to_record()["transitions"] is not None


def test_camera_diagnostic_has_no_cell_denominator_and_no_transition():
    route_a = frontier.route_a_frontier(**_route_a_records()).frontier
    camera = next(item for item in route_a.checkpoints if item.stage == "camera")
    assert camera.kind is frontier.RecordKind.CAMERA_DIAGNOSTIC
    assert camera.universe is frontier.Universe.NONE
    assert camera.metric is None
    assert all(transition.previous_stage != "camera"
               and transition.next_stage != "camera"
               for transition in route_a.transitions)
    tampered = _rebuild_dataclass(camera, universe=frontier.Universe.REFERENCE_CELLS)
    with pytest.raises(frontier.FrontierRefusal):
        frontier.FrontierCheckpoint(**{
            field.name: getattr(tampered, field.name)
            for field in fields(tampered)})


def test_route_b_transitions_stay_inside_one_comparable_universe():
    record = frontier.route_b_frontier(**_full_route_b())
    assert record.transitions, "the assembled tail must carry its gain/loss steps"
    for transition in record.transitions:
        assert transition.universe is frontier.Universe.REFERENCE_CELLS
        assert transition.gained is None and transition.lost is None
        assert transition.previous_stage != "entry"


# ---------------------------------------------------------------------------
# Stage prefixes and typed failure
# ---------------------------------------------------------------------------

def test_route_b_canonical_prefixes_are_reachability_prefixes():
    assert frontier.route_b_frontier(
        **_full_route_b()).reached_stages == ("entry", "lattice", "geometry", "output")
    assert frontier.route_b_frontier(
        entry=_entry_record(), lattice=_lattice_record(),
        geometry=_geometry_record()).reached_stages == (
            "entry", "lattice", "geometry")
    assert frontier.route_b_frontier(
        entry=_entry_record(), lattice=_lattice_record()).reached_stages == (
            "entry", "lattice")
    assert frontier.route_b_frontier(entry=_entry_record()).reached_stages == (
        "entry",)


@pytest.mark.parametrize("stage,expected", [
    ("lattice", "lattice_record_absent"),
    ("geometry", "lattice_lattice_record_absent"),
    ("output", "lattice_lattice_record_absent"),
])
def test_route_b_unreached_stages_are_typed_unavailable_and_never_zero(stage, expected):
    record = frontier.route_b_frontier(entry=_entry_record())
    row = next(item for item in record.checkpoints if item.stage == stage)
    assert row.reachable is False
    assert row.blocked == expected
    assert row.typed_reason == ""
    assert row.unavailable_reason is None
    assert row.record_digest == ""
    assert row.metric is None
    assert row.record_state == "unavailable"


def test_route_b_empty_entry_reports_entry_only_and_a_typed_frontier():
    record = frontier.route_b_frontier()
    assert record.reached_stages == ()
    assert [item.stage for item in record.checkpoints] == list(
        frontier.ROUTE_B_STAGES)
    assert record.checkpoints[0].blocked == "entry_record_absent"
    assert record.frontier_stop is record.checkpoints[0]
    failure = frontier.first_observable_failure(record)
    # No accepted invocation record: the frontier's own terminal reading is the
    # stage it stopped at, reported separately from the first unattempted row.
    assert failure["terminal_stage"] == "entry"
    assert failure["terminal_status"] == "unavailable"
    assert failure["terminal_reason"] == "entry_record_absent"
    assert failure["first_unattempted_checkpoint"]["stage"] == "entry"
    assert failure["first_unattempted_checkpoint"]["blocked"] == \
        "entry_record_absent"
    assert failure["evidence_digest"] == ""
    assert failure["metric"] is None


def test_route_a_empty_and_partial_prefixes_report_typed_frontiers():
    empty = frontier.route_a_frontier().frontier
    assert empty.reached_stages == ()
    assert [item.stage for item in empty.checkpoints] == list(
        frontier.ROUTE_A_STAGES)
    assert empty.checkpoints[0].blocked == "admission_record_absent"
    assert all(item.metric is None for item in empty.checkpoints)

    admission_only = frontier.route_a_frontier(admission=np.zeros(
        (4, 5, 3), dtype=np.uint8)).frontier
    assert admission_only.reached_stages == ("admission",)
    assert _blocked(admission_only)[1] == "camera_record_absent"
    assert _blocked(admission_only)[2] == "camera_camera_record_absent"
    assert admission_only.terminal_policy["stage"] == "camera"
    failure = frontier.first_observable_failure(admission_only)
    assert failure["terminal_stage"] == "camera"
    assert failure["first_unattempted_checkpoint"]["stage"] == "camera"
    assert failure["first_unattempted_checkpoint"]["kind"] == \
        frontier.RecordKind.CAMERA_DIAGNOSTIC.value
    assert failure["first_unattempted_checkpoint"]["universe"] == \
        frontier.Universe.NONE.value
    assert failure["evidence_digest"] == \
        admission_only.checkpoints[0].record_digest


def test_route_a_camera_solve_failure_is_a_typed_unavailable_camera():
    result = frontier.route_a_frontier(
        admission=np.zeros((4, 5, 3), dtype=np.uint8), camera=None,
        camera_solved=False, blocked_reason="camera_unavailable")
    assert result.frontier.reached_stages == ("admission",)
    camera = result.frontier.checkpoints[1]
    assert camera.blocked == "camera_unavailable"
    assert camera.unavailable_reason == "camera_unavailable"
    assert camera.metric is None
    assert [item.blocked for item in result.frontier.checkpoints[2:]] == [
        "camera_camera_unavailable"] * 3
    # The terminal is the actual failed stage, not the planned last checkpoint.
    assert result.frontier.terminal_policy["stage"] == "camera"
    assert result.frontier.terminal_status == "unavailable"
    assert result.frontier.terminal_reason == "camera_unavailable"
    assert result.frontier.frontier_stop is camera


def test_route_a_complete_route_terminates_at_the_planned_terminal():
    result = frontier.route_a_frontier(**_route_a_records())
    assert result.frontier.complete is True
    assert result.frontier.frontier_stop is None
    assert result.frontier.terminal_policy["stage"] == "terminal"

def test_route_a_empty_facade_is_a_typed_unavailable_grid():
    records = _route_a_records(facade_blocks=())
    result = frontier.route_a_frontier(blocked_reason="no_visible_blocks", **records)
    assert result.frontier.reached_stages == ("admission", "camera")
    assert result.frontier.checkpoints[2].blocked == "no_visible_blocks"
    assert result.frontier.checkpoints[2].unavailable_reason == "no_visible_blocks"
    assert [item.blocked for item in result.frontier.checkpoints[3:]] == [
        "facade_grid_no_visible_blocks"] * 2

def test_malformed_or_out_of_order_records_build_no_funnel():
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=None, lattice=_lattice_record())
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(), lattice=None,
                                  geometry=_geometry_record())
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(), lattice=_lattice_record(),
                                  output=_output_record())
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_a_frontier(admission=np.zeros((2, 2, 3), dtype=np.uint8),
                                  facade=_Volume({}))
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_Undigestible())
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_a_frontier(admission=object(), camera=object(),
                                  facade=object(), palette=_frame())


def test_frontier_unavailable_is_typed_and_contributes_no_zero():
    section = frontier.frontier_unavailable(
        "entry_refused", stage="entry", detail="graph_not_ok")
    assert section == {"available": False, "stage": "entry",
                       "reason": "entry_refused", "detail": "graph_not_ok",
                       "metric": None}
    with pytest.raises(frontier.FrontierRefusal):
        frontier.frontier_unavailable("")
    with pytest.raises(frontier.FrontierRefusal):
        frontier.frontier_unavailable("two words")


# ---------------------------------------------------------------------------
# Determinism and digest binding
# ---------------------------------------------------------------------------

def test_identical_records_serialize_byte_identically_with_stable_digests():
    first = frontier.route_b_frontier(**_full_route_b())
    second = frontier.route_b_frontier(**_full_route_b())
    assert first.to_record() == second.to_record()
    assert frontier.canonical_json(first.to_record()) == \
        frontier.canonical_json(second.to_record())
    assert first.frontier_sha256 == second.frontier_sha256
    for left, right in zip(first.checkpoints, second.checkpoints):
        assert left.to_record() == right.to_record()
        assert left.checkpoint_sha256 == right.checkpoint_sha256

    route_a_one = frontier.route_a_frontier(**_route_a_records())
    route_a_two = frontier.route_a_frontier(**_route_a_records())
    assert route_a_one.to_record() == route_a_two.to_record()
    assert route_a_one.frontier.to_record() == route_a_two.frontier.to_record()


def test_shortest_failure_is_the_first_unreached_checkpoint():
    complete = frontier.route_b_frontier(**_full_route_b())
    failure = frontier.first_observable_failure(complete)
    assert failure["first_unattempted_checkpoint"] is None
    assert failure["terminal_stage"] == "output"
    assert failure["evidence_digest"] == \
        complete.checkpoints[-1].record_digest


# ---------------------------------------------------------------------------
# Tampered records
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("changes", [
    {"stage": "geometry_extra"},
    {"kind": "G"},
    {"universe": "reference_cells"},
    {"reachable": "yes"},
    {"record_digest": "not-a-digest"},
    {"metric": 0},
    {"checkpoint_sha256": "0" * 64},
])
def test_a_tampered_checkpoint_refuses_and_builds_no_funnel(changes):
    record = frontier.route_b_frontier(**_full_route_b())
    forged = _rebuild_dataclass(record.checkpoints[2], **changes)
    try:
        rebuilt = frontier.FrontierCheckpoint(**{
            field.name: getattr(forged, field.name) for field in fields(forged)})
    except frontier.FrontierRefusal:
        # Rejected by the checkpoint's own validating constructor.
        return
    # A re-sealed checkpoint that is self-consistent must still not repair the
    # frontier: its carried transitions bind the payload the frontier recorded.
    with pytest.raises(frontier.FrontierRefusal):
        frontier.ImageOnlyFrontier(
            route=record.route,
            checkpoints=(record.checkpoints[0], record.checkpoints[1], rebuilt,
                         record.checkpoints[3]),
            transitions=record.transitions,
            terminal_policy=record.terminal_policy,
            terminal_policy_agrees=True,
            terminal_status=record.terminal_status,
            terminal_reason=record.terminal_reason)


def test_a_self_consistent_but_unbound_checkpoint_is_refused_by_the_frontier():
    """A re-sealed checkpoint cannot silently replace a transition-bound payload."""
    record = frontier.route_b_frontier(**_full_route_b())
    target = record.checkpoints[2]
    forged = _rebuild_dataclass(target, record_state="smuggled_state",
                                checkpoint_sha256="")
    resealed = frontier.FrontierCheckpoint(**{
        field.name: getattr(forged, field.name) for field in fields(forged)})
    assert resealed.record_state == "smuggled_state"
    with pytest.raises(frontier.FrontierRefusal):
        frontier.ImageOnlyFrontier(
            route=record.route,
            checkpoints=(record.checkpoints[0], record.checkpoints[1], resealed,
                         record.checkpoints[3]),
            transitions=record.transitions,
            terminal_policy=record.terminal_policy,
            terminal_policy_agrees=True,
            terminal_status=record.terminal_status,
            terminal_reason=record.terminal_reason)


def test_tampered_reachability_and_transition_binding_are_refused():
    record = frontier.route_b_frontier(**_full_route_b())
    unreached = frontier.route_b_frontier(entry=_entry_record())
    # The first checkpoint the truncated route did not reach, followed by a
    # reached one, is not a reachability prefix.
    assert unreached.checkpoints[1].reachable is False
    with pytest.raises(frontier.FrontierRefusal):
        frontier.ImageOnlyFrontier(
            route=unreached.route,
            checkpoints=(unreached.checkpoints[1], record.checkpoints[2]))
    # A transition that does not bind its two checkpoints is refused.
    with pytest.raises(frontier.FrontierRefusal):
        frontier.ImageOnlyFrontier(
            route=record.route, checkpoints=record.checkpoints,
            transitions=(_rebuild_dataclass(
                record.transitions[0], next_sha256="0" * 64),
                record.transitions[1]),
            terminal_policy=record.terminal_policy)
    # A1 carries no numeric transition fields: the counts stay null and the
    # module performs no gain/loss arithmetic or net reconciliation.
    for transition in record.transitions:
        assert transition.previous is None
        assert transition.gained is None
        assert transition.lost is None
        assert transition.net is None
        assert transition.geometry_lost is None
        assert transition.state_lost is None
    null_counts = frontier.FrontierTransition(
        previous_stage="lattice", next_stage="geometry",
        universe=frontier.Universe.REFERENCE_CELLS)
    assert null_counts.net is None


@pytest.mark.parametrize("name", ["previous", "gained", "lost", "net",
                                  "geometry_lost", "state_lost"])
@pytest.mark.parametrize("value", [0, 1])
def test_a1_transition_refuses_any_non_null_numeric_field(name, value):
    """A1 transition counts are ``None``; even ``0`` is a typed refusal."""
    with pytest.raises(frontier.FrontierRefusal):
        frontier.FrontierTransition(
            previous_stage="lattice", next_stage="geometry",
            universe=frontier.Universe.REFERENCE_CELLS, **{name: value})


# ---------------------------------------------------------------------------
# Route-B typed bindings
# ---------------------------------------------------------------------------

def test_route_b_refuses_a_lattice_that_does_not_bind_the_live_session():
    lattice = _lattice_record()
    object.__setattr__(lattice, "session_sha256", "f" * 64)
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(), lattice=lattice)


def test_route_b_refuses_geometry_or_output_with_a_broken_chain():
    geometry = _geometry_record()
    object.__setattr__(geometry, "lattice_sha256", "f" * 64)
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(),
                                  lattice=_lattice_record(), geometry=geometry)
    output = _output_record()
    object.__setattr__(output, "geometry_sha256", "f" * 64)
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(), lattice=_lattice_record(),
                                  geometry=_geometry_record(), output=output)


def test_route_b_terminal_policy_comes_from_the_accepted_invocation_record():
    records = _full_route_b()
    agreeing = _production_result(stage="output", status="review",
                                  reason="proposals_review")
    built = frontier.route_b_frontier(result=agreeing, **records)
    assert built.terminal_policy_agrees is True
    assert built.terminal_policy["stage"] == "output"
    assert built.terminal_status == "review"
    assert built.terminal_reason == "proposals_review"


def test_route_b_failed_geometry_is_the_terminal_while_output_is_unattempted():
    """The exact recorded Route-B stop: geometry/unavailable/g_unavailable.

    A failed geometry record is still *carried* (``reachable`` means the record
    exists), so ``geometry`` is the accepted terminal failure while ``output`` -
    which was never attempted - is the first unattempted checkpoint.
    """
    result = _production_result(stage="geometry", status="geometry_unavailable",
                                reason="g_unavailable", detail="plane_integer_residual")
    built = frontier.route_b_frontier(
        entry=_entry_record(), lattice=_lattice_record(),
        geometry=_geometry_record(status="unavailable",
                                  reason="plane_integer_residual"),
        result=result)
    geometry = next(item for item in built.checkpoints if item.stage == "geometry")
    assert geometry.reachable is True
    assert geometry.record_state == "unavailable"
    assert built.reached_stages == ("entry", "lattice", "geometry")
    assert built.terminal_policy["stage"] == "geometry"
    assert built.terminal_status == "geometry_unavailable"
    assert built.terminal_reason == "g_unavailable"
    assert built.terminal_detail == "plane_integer_residual"
    assert built.frontier_stop is not None
    assert built.frontier_stop.stage == "output"
    failure = frontier.first_observable_failure(built)
    assert failure["terminal_stage"] == "geometry"
    assert failure["terminal_status"] == "geometry_unavailable"
    assert failure["terminal_reason"] == "g_unavailable"
    assert failure["first_unattempted_checkpoint"]["stage"] == "output"
    assert failure["first_unattempted_checkpoint"]["blocked"] == \
        "output_record_absent"


def test_route_b_result_terminal_stage_must_be_the_last_carried_record():
    # A result that claims a terminal stage whose record was never carried (here
    # the geometry record is absent) contradicts the carried prefix.
    result = _production_result(stage="geometry", status="geometry_unavailable",
                                reason="g_unavailable")
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(entry=_entry_record(),
                                  lattice=_lattice_record(), result=result)
    # A result whose terminal stage is not a route-B stage is refused.
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(
            **_full_route_b(),
            result=_production_result(stage="placement", status="review",
                                      reason="x"))
    # A result that does not bind the carried terminal record's digest is refused.
    with pytest.raises(frontier.FrontierRefusal):
        frontier.route_b_frontier(
            entry=_entry_record(), lattice=_lattice_record(),
            geometry=_geometry_record(),
            result=_production_result(stage="geometry",
                                      status="geometry_unavailable",
                                      reason="g_unavailable",
                                      geometry_sha256="f" * 64))


# ---------------------------------------------------------------------------
# Synthetic recipe: real public tail constructors with synthetic route-B records
# ---------------------------------------------------------------------------

def _recipe_tail_records():
    """Real volume/proposal/promotion records from accepted public constructors.

    These are genuine production tail constructors; the route-B envelope records
    built around them below are synthetic ``SimpleNamespace`` stand-ins, not real
    end-to-end production constructors (no fixture, image or JAR is read).
    """
    coordinate = (0, 0, 0)
    claim = volume_mod.CellClaim.build(
        frame="f0", plane_index=0, reading_axis=0, absolute_quadrant=0,
        source_region_id="a" * 64, reading_identity_sha256="b" * 64,
        support_sha256="c" * 64, face_sha256="d" * 64,
        row=0, column=0, world_normal_axis=0, world_normal_sign=-1,
        outward_sign=-1, face=0, normal_block=0, tangent_u_axis=1, lateral_u=0,
        tangent_v_axis=2, lateral_v=0, coordinate=coordinate)
    cell = volume_mod.SupportedCellObservation.build(
        coordinate=coordinate, claims=(claim,), frame_ids=("f0",),
        conflict_kind=None)
    bindings = tuple(sorted(({
        "frame": claim.frame, "plane_index": 0, "reading_axis": 0,
        "reading_identity_sha256": claim.reading_identity_sha256,
        "support_sha256": claim.support_sha256, "face_sha256": claim.face_sha256,
        "source_region_id": claim.source_region_id,
        "alternative_distance": [0, 1], "phase_anchor_sha256": "e" * 64,
        "support_cells": 1, "supported_cells": 1, "unknown_cells": 0,
    },), key=volume_mod._binding_sort_key))

    work = volume_mod._make_work({
        "faces": 1, "support_cells": 1, "contributions": 1, "coordinates": 1,
        "conflicts": 0, "omitted_unknown": 0, "dense_bounds_cells": 1})
    volume = volume_mod.PlaneVolumeResult.build(
        schema_version=volume_mod.V3_VOLUME_SCHEMA,
        producer_version=volume_mod.V3_VOLUME_PRODUCER_VERSION,
        gauge_state=volume_mod.V3_VOLUME_GAUGE_STATE, gauge_axes=(0, 1, 2),
        alpha=((0, 1), (0, 1), (0, 1)), placement_sha256="f" * 64,
        support_bindings=bindings, cells=(cell,), conflicts=(), omitted=(),
        cell_bounds=((0, 0, 0), (0, 0, 0)), dense_bounds_cells=1,
        normalization_offset=(0, 0, 0), export_bounds=None, complete=False,
        work=work)
    assert isinstance(volume, volume_mod.PlaneVolumeResult), volume
    observation = FaceObservation(
        cell=coordinate, slot="north", frame_id="f0",
        normal=NORMAL_OF_SLOT["north"],
        patch=np.full((4, 4, 3), 125.0, dtype=np.float32),
        usable_coverage=1.0, support_frames=("f0",), provenance=("cam/a",))
    proposals = proposals_mod.propose_face_states(
        (observation,),
        (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),),
        max_cost=1e9, min_margin=0.0,
        max_observations=1000, max_candidate_evaluations=100_000,
        max_payload_bytes=10_000_000)
    assert proposals.status == proposals_mod.PROPOSED, proposals
    promotion = promotion_mod.promote_plane_states(
        volume, proposals, budget=promotion_mod.make_promotion_budget(
            max_cells=16, max_proposals=16, max_payload_bytes=1_000_000))
    assert isinstance(promotion, promotion_mod.PromotionResult), promotion
    assert promotion.exact_state_eligible is True
    return volume, proposals, promotion


def test_synthetic_recipe_tail_quotes_real_volume_records_beside_synthetic_envelope():
    volume, proposals, promotion = _recipe_tail_records()
    # Explicitly synthetic route-B envelope records (SimpleNamespace stand-ins),
    # not real end-to-end production constructors.
    entry = _entry_record()
    lattice = SimpleNamespace(
        status="assembled", reason="", detail="",
        result_sha256=_HEX["lattice"], session_sha256=_HEX["session"])
    geometry = SimpleNamespace(
        status="assembled", reason="", detail="",
        result_sha256=_HEX["geometry"], lattice_sha256=_HEX["lattice"])
    output = SimpleNamespace(
        status="promoted", reason="", detail="",
        result_sha256=_HEX["output"], geometry_sha256=_HEX["geometry"])
    result = _production_result(stage="output", status="promoted", reason="")
    record = frontier.route_b_frontier(
        entry=entry, lattice=lattice, geometry=geometry, output=output,
        result=result)
    assert record.reached_stages == (
        "entry", "lattice", "geometry", "output")
    assert record.terminal_status == "promoted"
    assert record.checkpoints[-1].record_digest == _HEX["output"]
    assert record.transitions[-1].universe is frontier.Universe.REFERENCE_CELLS
    # The genuine volume/proposal/promotion records are digest-bound and replay
    # through their own public constructors; the route-B envelope above is
    # synthetic test scaffolding, not a production integration proof.
    assert promotion.volume_sha256 == volume.volume_sha256
    assert promotion.proposal_result_digest == proposals.result_digest
    assert promotion.to_block_volume().blocks
    assert compare_mod is not None
