"""Focused synthetic tests for the DS-R4-V2-ROUTEB-RUNNER one-shot runner.

These tests run no real capture, no image pixel decode of real screenshots, no
evaluator/reference/litematic/truth/camera-manifest/F3/direction/mask source and
no production CLI. They operate only on the runner's own injectable API seam and
on in-memory synthetic V2 material built from the accepted
``candidate_support_provenance`` test helpers. The runner module itself is loaded
from its script path, exactly like the accepted support-ownership harness test.

Pinned invariants:

* exactly one production pass - one entry, lattice, signature and graph call,
  with the signature-witness sink collected during the same signature pass and
  forwarded with the same accepted session/lattice/signature objects into the
  graph seam;
* the entry gate mirrors the production readiness contract: the status-less,
  empty-reason ``ImageDerivedEntryResult`` terminal with graph and pose graph
  ``ok`` passes to lattice, while a refused graph or pose graph refuses at entry
  with its typed reason/detail preserved and never reaches lattice;
* single-read ingress identity - the recorded source hash is the entry's own
  ``FrameIngressProvenanceV1.byte_sha256`` and the runner never pre-reads or
  pre-hashes an input path; the pins are bound to the supplied paths
  positionally in contract order, so a permuted pin order refuses;
* exactly the frozen three-image input; a fourth image refuses before entry;
* the fixed interface: screenshot paths plus one output root, with graph settings
  frozen to the production defaults and no caller-supplied hash, identity pin,
  settings mapping or record-mode CLI;
* the frozen 1,928-row count, complete admitted order and sealed
  ``whole_universe_identity_sha256`` gate refuse atomically on any count, order,
  content or cap mismatch, and write no record;
* staged entry output and no-clobber publication: a failed run leaves no
  ``record.json``, staged ``route_b.mcscan``, ``.partial`` or staging directory,
  and a symlink/reparse output root or ancestor, or an uninspectable or
  file-parent component, is refused closed; and
* the production default/public path stays byte-identical (the runner changes
  no production source and adds no default-pass allocation).
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.pipeline import candidate_support_provenance as csp
from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2
from mc_photogrammetry.pipeline import frame_ingress as ingress_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.reconstruction import plane_graph
from test_candidate_support_provenance import (
    _HEX,
    _HEX_B,
    _binding,
    _match,
    _observation,
    _payload_row,
    _pixels,
    _png_bytes,
    _readings_for,
    _refusal_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "evaluation" / "run_route_b_diagnostic.py"
EXPECTED_ROW_COUNT = 1_928

# The accepted production modules this runner consumes must stay byte-identical.
# The V2 pin includes the accepted sparse-sidecar replay repair recorded in
# DS-R4-V2-SPARSE-SIDECAR-PAGE-INDEX-2026-10-04.md.
PRODUCTION_PINS = {
    "src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py":
        "F5CEE3CFE9D0247069637992FB29D1F9A8533ABC9383AF841CFC9A726313CCD2",
    "src/mc_photogrammetry/pipeline/image_derived_graph.py":
        "32747546C934F8C218E43646CC995F310FF383E522C8A10F878E691F54954188",
    "src/mc_photogrammetry/pipeline/image_derived_signatures.py":
        "8F7D24C018395A32A1D2A3CDA616E2E7A7EED1873DCEFC74FE3D673911CF7555",
    "src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py":
        "8441E91CF158D5403676D249FE6EBDEBEEB723BD4A9967FDEB8371BEB1C9CB81",
}

_ASSIGNMENT = hashlib.sha256(b"route-b-diagnostic-assignment").hexdigest()

# The real ``ImageDerivedEntryResult`` terminal has no ``status`` field: when the
# graph and pose graph are both ``ok`` its ``reason`` is empty and its ``detail``
# is exactly this informational image-only statement, which is *not* a refusal
# and must proceed to the lattice/signature/graph chain.
_GEOMETRY_UNAVAILABLE_DETAIL = (
    "image-derived session and relative camera evidence are established; "
    "V1/V2/V3/M5 geometry is unavailable from images alone in normal mode")


# ---------------------------------------------------------------------------
# Runner module + synthetic capture material
# ---------------------------------------------------------------------------

def _runner():
    name = "_route_b_diagnostic_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _distance_reading(frame_id, plane):
    from mc_photogrammetry.reconstruction import lattice_planes as lattice_planes_mod

    return lattice_planes_mod.PlaneReading(
        frame=frame_id, axis=1, side=1.0,
        alternatives=(lattice_planes_mod.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0 + 4.0 * plane, phase_pixels=(0.0, float(plane)),
        extent=1.0, samples=17, polygon=None, coverage_cells=4,
        structural_share=0.6, confidence=0.8,
        rectification=lattice_planes_mod.PlaneRectification(
            tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            12.0, (16.0, 16.0), (0.0, 0.0)))


def _pin_witness_binding(pin, signature_sha256, seed):
    """One real sampler witness over a synthetic frame of the pin's own shape.

    The witness is produced by the ordinary sampler over synthetic pixels shaped
    exactly like the pinned frame, so ``SourceTapWitnessBindingV1`` binds its
    rectification ``source_shape`` to that root pin. This is what makes the
    runner's written record internally consistent: every bound signature witness
    names a frame that is in the capture root's ordered ingress.
    """
    from mc_photogrammetry.reconstruction import lattice_planes as lattice_planes_mod

    pixels = _pixels(seed, shape=tuple(pin.source_shape) + (3,))
    sink: list = []
    lattice_planes_mod.plane_signature(
        pixels, _distance_reading(pin.frame_id, 0), 1.0, witness_sink=sink)
    return csp.SourceTapWitnessBindingV1(
        frame_id=pin.frame_id, canonical_index=0, alternative_index=0,
        signature_sha256=signature_sha256, ingress=pin,
        mask_sha256=_HEX_B, masks_sha256=_HEX,
        max_cells=lattice_planes_mod.MAX_SIGNATURE_CELLS,
        max_rectified_side=lattice_planes_mod.MAX_RECTIFIED_SIDE,
        pixels_per_cell=lattice_planes_mod.SIGNATURE_PIXELS_PER_CELL,
        witness=sink[0])


def _route_b_witness_bindings(pins=None):
    """The two ordered route-B signature witnesses.

    With ``pins`` supplied (the entry session's ordered ingress) each binding uses
    that pin as its ingress and a witness over a synthetic frame of the pin's own
    shape, so the record's sampling section is fully root-pinned. Without pins the
    legacy standalone synthetic bindings are returned.
    """
    if pins is None:
        return (
            _binding("a", _pixels(211), _png_bytes((1, 2, 3)),
                     signature_sha256=_HEX),
            _binding("b", _pixels(212), _png_bytes((4, 5, 6)),
                     signature_sha256=_HEX_B),
        )
    pins = tuple(pins)
    return tuple(
        _pin_witness_binding(pin, signature, 211 + index)
        for index, (pin, signature) in enumerate(zip(pins, (_HEX, _HEX_B))))


def _route_b_universe(count, present):
    """A synthetic Route B diagnostic universe over the shipped sink material.

    The first ``present`` candidates are kept and carry real support payloads;
    every other admitted candidate is dropped with a typed row-local refusal.
    """
    candidates = tuple(
        _match("a", "b", left_plane=0, right_plane=0, score=0.9 - index * 1e-9)
        for index in range(count))
    dispositions = tuple(
        plane_graph.SAME_AXIS_DISPOSITION_KEPT if index < present
        else plane_graph.SAME_AXIS_DISPOSITION_DROPPED for index in range(count))
    rows = []
    for index, (candidate, disposition) in enumerate(zip(candidates, dispositions)):
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            rows.append(_payload_row(candidate, index=index,
                                     disposition=disposition, post_index=index))
        else:
            rows.append(_refusal_row(candidate, index=index,
                                     disposition=disposition, post_index=None))
    return candidates, candidates[:present], tuple(rows), dispositions


def _synthetic_capture(count=EXPECTED_ROW_COUNT, present=2, *, plan=None,
                       ingress=None):
    """Collect one valid in-memory V2 capture directly from the accepted seam."""
    candidates, matches, rows, dispositions = _route_b_universe(count, present)
    if ingress is None:
        bindings = _route_b_witness_bindings()
        ingress = tuple(item.ingress for item in bindings)
    else:
        ingress = tuple(ingress)
        bindings = _route_b_witness_bindings(ingress)
    observation = _observation(candidates, dispositions)
    kwargs = {} if plan is None else {"plan": plan}
    return v2.collect_route_b_diagnostic_support(
        _readings_for(candidates), matches, bindings, rows,
        observation, ingress=ingress,
        failed_assignment_sha256=_ASSIGNMENT, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic inputs and the entry's own ingress pins
# ---------------------------------------------------------------------------

def _synthetic_pngs(directory, count=3):
    """Real PNG-suffixed byte files; only metadata is used by the runner."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"ordinary-view-{index}.png"
        path.write_bytes(_png_bytes((index + 1, 2, 3)))
        paths.append(path)
    return tuple(paths)


def _session_pins(paths):
    """One entry-style ingress pin per input, from each file's actual bytes.

    This mirrors what ``build_image_derived_entry(capture_ingress=True)`` attaches
    to its session: the pin's ``byte_sha256`` comes from the exact encoded buffer
    the entry read and decoded.
    """
    pins = []
    for index, path in enumerate(paths):
        data = Path(path).read_bytes()
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        pins.append(ingress_mod.build_frame_ingress_provenance(
            frame_id=f"image-{index:04d}",
            source_label=str(Path(path).resolve()),
            encoded_bytes=data, decoded_pixels=pixels,
            transform=FrameTransform(source_size=(4, 4)),
            decoded_pixel_sha256=session_mod.image_sha256(pixels)))
    return tuple(pins)


def _session_for(paths, pins=None):
    """The entry-shaped session carrying each input's own ingress pin, in order."""
    pins = _session_pins(paths) if pins is None else tuple(pins)
    return SimpleNamespace(
        frames=tuple(SimpleNamespace(frame_id=pin.frame_id, ingress=pin)
                     for pin in pins),
        session_sha256="1" * 64)


def _entry_terminal(session, *, graph_status="ok", pose_status="ok",
                    reason="", detail=None):
    """An ``ImageDerivedEntryResult``-shaped terminal with **no** ``status``.

    The real terminal is always ``geometry_unavailable``; with both graphs ``ok``
    its ``reason`` is empty and its ``detail`` is the informational image-only
    statement. Passing ``graph_status``/``pose_status`` other than ``ok`` models
    the real derived terminal, whose ``reason``/``detail`` come from the failed
    graph or pose graph.
    """
    if detail is None:
        detail = _GEOMETRY_UNAVAILABLE_DETAIL
    return SimpleNamespace(
        terminal="geometry_unavailable", reason=reason, detail=detail,
        session=session, entry_sha256="2" * 64,
        graph=SimpleNamespace(status=graph_status),
        pose_graph=SimpleNamespace(status=pose_status))


def _capture_for(paths, *, count=EXPECTED_ROW_COUNT, present=2, plan=None):
    """A synthetic capture whose ordered root ingress is the entry's own pins.

    The runner cross-binds entry session pins, ``capture.root.ingress``, the
    serialized top-level ``ingress`` and the positional ``inputs`` rows, so a
    capture that is meant to reach publication must carry the same ordered pins
    the fake entry attaches to its session.
    """
    return _synthetic_capture(count, present, plan=plan,
                              ingress=_session_pins(paths))


# ---------------------------------------------------------------------------
# Injectable API seam: one pass, with counters
# ---------------------------------------------------------------------------

def _write_synthetic_project(path, data, cap=None, retain=False):
    if not retain:
        Path(path).write_bytes(data)
        return None
    from mc_photogrammetry.project.store import (
        ProjectWriteLease, _WindowsRetainedFile, _WindowsRootAnchor, _write_bounded)
    if type(cap) is not int or len(data) > cap:
        from mc_photogrammetry.pipeline.image_derived_entry import ImageDerivedEntryError, ENTRY_PERSIST_FAILED
        raise ImageDerivedEntryError(ENTRY_PERSIST_FAILED, "project_over_cap")
    anchor = _WindowsRootAnchor(Path(path).parent)
    held = _WindowsRetainedFile(path, create=True)
    try:
        _write_bounded(held, data, cap)
        held.flush()
        return ProjectWriteLease(held, anchor, cap, data)
    except BaseException:
        try:
            held.delete_owned()
        finally:
            held.close()
            anchor.close()
        raise


def _fake_api(capture, paths, *, calls, pins=None, terminal=None):
    pins = _session_pins(paths) if pins is None else tuple(pins)
    bindings = _route_b_witness_bindings(pins)
    session = _session_for(paths, pins)

    def entry(input_paths, *, project_path=None, capture_ingress=False, project_max_bytes=None, retain_project_lease=False):
        calls.append("entry")
        # The ordinary entry persists its schema-2 project before returning; the
        # runner stages that exact output. The returned terminal is status-less,
        # exactly like the real ``ImageDerivedEntryResult``.
        lease = _write_synthetic_project(project_path, b'{"schema":"synthetic-route-b-project"}',
                                         project_max_bytes, retain_project_lease)
        result = _entry_terminal(session) if terminal is None else terminal
        result.project_lease = lease
        return result

    def lattice(*, session, pose_graph):
        calls.append("lattice")
        return SimpleNamespace(status="assembled", reason="", result_sha256="3" * 64,
                               readings=(_distance_reading("a", 0),
                                         _distance_reading("b", 0)))

    def signatures(*, session, lattice_result, signature_witness_sink=None):
        calls.append("signatures")
        assert signature_witness_sink is not None
        signature_witness_sink.extend(bindings)
        return SimpleNamespace(status="assembled", reason="", result_sha256="4" * 64)

    def graph(*, session, lattice_result, signatures_result, connection_graph,
              signature_witnesses=None, complete_support_sink=None, **settings):
        calls.append("graph")
        assert complete_support_sink is not None
        assert signature_witnesses is not None
        complete_support_sink(capture)
        return (SimpleNamespace(result_sha256="5" * 64),
                SimpleNamespace(attribution_sha256="6" * 64))

    return (entry, lattice, signatures, graph)


def _entry_failure_api(paths, *, calls, graph_status="ok", pose_status="ok",
                       reason="", detail=""):
    """A type-preserving API whose entry terminal has a refused graph or pose.

    The later stages record their own name and then fail loudly, so a regression
    that wrongly accepts the refused entry cannot pass this control by silently
    running the remaining chain.
    """
    session = _session_for(paths)
    terminal = _entry_terminal(session, graph_status=graph_status,
                               pose_status=pose_status, reason=reason,
                               detail=detail)

    def entry(input_paths, *, project_path=None, capture_ingress=False, project_max_bytes=None, retain_project_lease=False):
        calls.append("entry")
        Path(project_path).write_bytes(b'{"schema":"synthetic-route-b-project"}')
        return terminal

    def unreachable(stage):
        def call(**kwargs):
            calls.append(stage)
            raise AssertionError(
                f"{stage} must not run after a typed entry refusal")
        return call

    return (entry, unreachable("lattice"), unreachable("signatures"),
            unreachable("graph"))


def _execute(module, tmp_path, capture, *, calls=None, root=None, api=None,
             paths=None):
    paths = _synthetic_pngs(tmp_path) if paths is None else paths
    calls = [] if calls is None else calls
    return module.execute_route_b_diagnostic(
        image_paths=[str(path) for path in paths],
        root=(root if root is not None else tmp_path / "run"),
        api=(api if api is not None else _fake_api(capture, paths, calls=calls)))


def _no_failure_artifacts(root):
    root = Path(root)
    assert not (root / "record.json").exists()
    assert not (root / "record.json.partial").exists()
    assert not (root / "route_b.mcscan").exists()
    assert not (root / "route_b_staging").exists()


# ---------------------------------------------------------------------------
# One-pass and input-surface invariants
# ---------------------------------------------------------------------------

def test_single_pass_calls_each_stage_once_and_collects_witnesses_in_that_pass(
        tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    pins = _session_pins(paths)
    calls = []
    report = _execute(module, tmp_path, capture, calls=calls, paths=paths)

    assert report["status"] == "complete"
    assert calls == ["entry", "lattice", "signatures", "graph"]
    assert report["row_count"] == EXPECTED_ROW_COUNT
    assert report["signature_witness_count"] == 2
    assert report["calls"] == ["entry", "lattice", "signatures", "graph"]
    # The written record binds the same accepted session ingress pins, in order.
    body = json.loads((tmp_path / "run" / "record.json").read_text(encoding="utf-8"))
    assert body["ingress"] == [pin.to_record() for pin in pins]
    assert [row["frame_id"] for row in body["inputs"]] == [
        pin.frame_id for pin in pins]
    assert [row["path"] for row in body["inputs"]] == [
        pin.source_label for pin in pins]
    assert [row["sha256"] for row in body["inputs"]] == [
        pin.byte_sha256 for pin in pins]
    assert body["capture"]["root"]["row_count"] == EXPECTED_ROW_COUNT
    # The entry's staged project is published on success and no staging remains.
    assert (tmp_path / "run" / "route_b.mcscan").is_file()
    assert not (tmp_path / "run" / "route_b_staging").exists()
    assert not (tmp_path / "run" / "record.json.partial").exists()


def test_status_less_geometry_unavailable_entry_passes_gate_and_calls_lattice(
        tmp_path):
    """The real status-less entry terminal is accepted, not read as a refusal."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    terminal = _entry_terminal(_session_for(paths))
    # ``ImageDerivedEntryResult`` has no ``status`` field at all; with both graphs
    # ok its reason is empty and its detail is the informational image-only
    # statement, which must proceed instead of refusing.
    assert not hasattr(terminal, "status")
    assert terminal.terminal == "geometry_unavailable"
    assert terminal.reason == ""
    assert terminal.detail == _GEOMETRY_UNAVAILABLE_DETAIL

    calls = []
    report = _execute(
        module, tmp_path, capture, calls=calls, paths=paths,
        api=_fake_api(capture, paths, calls=calls, terminal=terminal))

    assert report["status"] == "complete"
    assert calls == ["entry", "lattice", "signatures", "graph"]
    assert report["calls"] == ["entry", "lattice", "signatures", "graph"]


def test_entry_graph_failure_refuses_before_lattice_and_preserves_reason_detail(
        tmp_path):
    """A refused connection graph refuses at entry, before any later stage."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls = []
    api = _entry_failure_api(
        paths, calls=calls, graph_status="refused",
        reason="entry_connection_graph_failed",
        detail="the accepted connection adapter reported a contract violation")
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, None, calls=calls, paths=paths, api=api)
    assert error.value.reason == "entry_connection_graph_failed"
    assert error.value.field == "entry"
    assert error.value.detail == (
        "the accepted connection adapter reported a contract violation")
    assert calls == ["entry"]
    _no_failure_artifacts(tmp_path / "run")
    refusal = json.loads((tmp_path / "run" / "refusal.json").read_text(
        encoding="utf-8"))
    assert refusal["reason"] == "entry_connection_graph_failed"


def test_entry_pose_failure_refuses_before_lattice_and_preserves_reason_detail(
        tmp_path):
    """A refused pose graph refuses at entry, before any later stage."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls = []
    api = _entry_failure_api(
        paths, calls=calls, graph_status="ok", pose_status="refused",
        reason="entry_pose_graph_failed",
        detail="the accepted pose-graph composer reported a contract violation")
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, None, calls=calls, paths=paths, api=api)
    assert error.value.reason == "entry_pose_graph_failed"
    assert error.value.field == "entry"
    assert error.value.detail == (
        "the accepted pose-graph composer reported a contract violation")
    assert calls == ["entry"]
    _no_failure_artifacts(tmp_path / "run")
    refusal = json.loads((tmp_path / "run" / "refusal.json").read_text(
        encoding="utf-8"))
    assert refusal["reason"] == "entry_pose_graph_failed"


def test_runner_has_no_truth_surface_and_accepts_only_screenshot_paths(tmp_path):
    module = _runner()
    source = RUNNER_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("evaluator", "ground_truth", "litematic",
                      "camera_manifest"):
        assert forbidden not in source, forbidden
    # Only ordinary .png inputs are accepted; a truth-like path refuses before
    # any capture is attempted.
    truth_paths = []
    for index in range(3):
        path = tmp_path / f"evaluator_truth_{index}.json"
        path.write_text("{}", encoding="utf-8")
        truth_paths.append(path)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.resolve_images([str(path) for path in truth_paths])
    assert error.value.reason == "invalid_input"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in truth_paths],
            root=tmp_path / "run",
            api=_fake_api(_synthetic_capture(), (), calls=[]))
    assert error.value.reason == "invalid_input"


def test_interface_rejects_caller_hashes_pin_settings_and_record_modes(tmp_path):
    """The fixed interface exposes only screenshot paths and one output root.

    The Phase A harness adds four inert-by-default seams (``expected_pins``,
    ``checkpoint_sink``, ``observer``, ``project_max_bytes``). They are not a
    caller-controlled identity surface: the harness always passes the fixed
    accepted T2 pin profile, and no caller hash, settings mapping, universe
    digest or record-mode switch exists on the runner or its CLI.
    """
    import inspect

    module = _runner()
    parameters = inspect.signature(
        module.execute_route_b_diagnostic).parameters
    assert set(parameters) == {
        "image_paths", "root", "api", "expected_pins", "checkpoint_sink",
        "observer", "project_max_bytes", "output_budget", "invocation_ownership"}, parameters
    assert parameters["expected_pins"].default is None
    assert parameters["project_max_bytes"].default is None
    assert parameters["checkpoint_sink"].default is None
    assert parameters["observer"].default is None

    images = _synthetic_pngs(tmp_path)
    base: list[str] = []
    for path in images:
        base.extend(["--image", str(path)])
    base.extend(["--root", str(tmp_path / "run")])

    for option, value in (
            ("--sha256", "0" * 64),
            ("--whole-universe-digest", "0" * 64),
            ("--settings", "{}"),
            ("--record", str(tmp_path / "record.json"))):
        with pytest.raises(SystemExit) as error:
            module._parse_args(base + [option, value])
        assert error.value.code == 2, option

    parsed = module._parse_args(base)
    assert parsed.image is not None and parsed.root is not None
    for absent in ("sha256", "whole_universe_digest", "settings", "record"):
        assert not hasattr(parsed, absent), absent

    # A stale caller cannot reintroduce the removed pin helper either.
    assert not hasattr(module, "check_whole_universe_pin")


# ---------------------------------------------------------------------------
# Frozen three-image input
# ---------------------------------------------------------------------------

def test_allowed_frame_count_is_exactly_three():
    module = _runner()
    assert tuple(module.ALLOWED_FRAME_COUNTS) == (3,)


def test_fourth_image_is_refused_before_entry(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path, count=4)
    calls = []
    root = tmp_path / "run"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=calls))
    assert error.value.reason == "invalid_input"
    assert error.value.field == "png_count"
    assert calls == []
    assert not root.exists()


# ---------------------------------------------------------------------------
# Single-read ingress identity
# ---------------------------------------------------------------------------

def test_source_identity_is_entry_ingress_and_paths_are_never_pre_read(
        tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("resolve_images must not read input bytes")

    monkeypatch.setattr(Path, "read_bytes", explode)
    resolved = module.resolve_images([str(path) for path in paths])
    monkeypatch.undo()
    assert [Path(path) for path in resolved] == list(paths)

    pins = _session_pins(paths)
    report = _execute(module, tmp_path, _capture_for(paths), paths=paths)
    body = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    assert [row["sha256"] for row in body["inputs"]] == [
        pin.byte_sha256 for pin in pins]
    # The internal `_file_sha256` is used only on a published artifact, never on
    # an input path; there is no caller-supplied pin to assert a stale value.
    assert not hasattr(module, "check_whole_universe_pin")


def test_changed_source_bytes_never_change_the_ingress_bound_identity(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    pins = _session_pins(paths)
    # Build the entry seam over the original bytes, then mutate every input file
    # on disk. The entry's pins remain the only recorded identity.
    calls: list = []
    api = _fake_api(capture, paths, calls=calls, pins=pins)
    for path in paths:
        Path(path).write_bytes(_png_bytes((7, 7, 7)))
    report = module.execute_route_b_diagnostic(
        image_paths=[str(path) for path in paths],
        root=tmp_path / "run", api=api)
    body = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    assert [row["sha256"] for row in body["inputs"]] == [
        pin.byte_sha256 for pin in pins]
    reread = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
    assert body["inputs"][0]["sha256"] != reread


def test_mismatched_ingress_source_binding_refuses_atomically(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _synthetic_capture()
    # Pins taken from a different path set must not be recorded as this run.
    foreign = _synthetic_pngs(tmp_path / "foreign")
    foreign_pins = _session_pins(foreign)
    root = tmp_path / "run"
    api = _fake_api(capture, paths, calls=[], pins=foreign_pins)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root, api=api)
    assert error.value.reason == "ingress_binding_mismatch"
    _no_failure_artifacts(root)


def test_permuted_ingress_source_binding_refuses_atomically(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _synthetic_capture()
    # The exact same three source identities as the contract order, but the entry
    # returns them permuted. A set comparison would accept this; the positional
    # contract-order binding must refuse rather than record a reordered identity.
    pins = _session_pins(paths)
    permuted = (pins[1], pins[0], pins[2])
    assert {pin.source_label for pin in permuted} == {
        pin.source_label for pin in _session_pins(paths)}
    root = tmp_path / "run"
    api = _fake_api(capture, paths, calls=[], pins=permuted)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root, api=api)
    assert error.value.reason == "ingress_binding_mismatch"
    assert error.value.field == "source_label"
    _no_failure_artifacts(root)


def test_duplicate_ingress_frame_ids_refuse(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _synthetic_capture()
    pins = list(_session_pins(paths))
    # A repeated frame identity is not a valid ordered binding even when the
    # source labels still name the supplied files.
    duplicate = ingress_mod.build_frame_ingress_provenance(
        frame_id=pins[1].frame_id, source_label=pins[0].source_label,
        encoded_bytes=Path(paths[0]).read_bytes(),
        decoded_pixels=np.zeros((4, 4, 3), dtype=np.uint8),
        transform=FrameTransform(source_size=(4, 4)),
        decoded_pixel_sha256=session_mod.image_sha256(
            np.zeros((4, 4, 3), dtype=np.uint8)))
    ordered = (duplicate, pins[1], pins[2])
    root = tmp_path / "run"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[], pins=ordered))
    assert error.value.reason == "ingress_binding_mismatch"
    _no_failure_artifacts(root)


def test_foreign_capture_root_ingress_refuses_atomically(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    # The entry's own pins match the supplied paths, but the capture root carries
    # a foreign ordered ingress. The pre-write cross-bind must refuse rather than
    # pair the entry's correct outer rows with foreign captured bytes.
    foreign = _synthetic_pngs(tmp_path / "foreign")
    capture = _synthetic_capture(ingress=_session_pins(foreign))
    root = tmp_path / "run"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[]))
    assert error.value.reason == "ingress_binding_mismatch"
    assert error.value.field == "record.capture_root"
    _no_failure_artifacts(root)
    assert (root / "refusal.json").is_file()


def test_permuted_capture_root_ingress_refuses_atomically(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    pins = _session_pins(paths)
    # The exact same ordered source identities as the contract, but the capture
    # root returns them permuted; the positional cross-bind must refuse.
    capture = _synthetic_capture(ingress=(pins[1], pins[0], pins[2]))
    root = tmp_path / "run"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[]))
    assert error.value.reason == "ingress_binding_mismatch"
    assert error.value.field == "record.capture_root"
    _no_failure_artifacts(root)

# ---------------------------------------------------------------------------
# Output-root safety and no-clobber publication
# ---------------------------------------------------------------------------

def test_symlink_or_reparse_output_root_is_refused(tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls = []
    root = tmp_path / "run"
    # Reparse detection is read-only metadata; the synthetic probe avoids any
    # native symlink/reparse creation.
    monkeypatch.setattr(
        module, "_reparse_tag",
        lambda path: 0xA0000003 if Path(path) == root else 0)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=calls))
    assert error.value.reason == "root_reparse"
    assert calls == []
    assert not root.exists()


def test_reparse_ancestor_component_is_refused(tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls = []
    root = tmp_path / "run"
    ancestor = tmp_path
    # Metadata-only mock: an existing ancestor of the requested root reports a
    # synthetic reparse tag, so the runner refuses the whole component chain
    # without creating any native symlink/junction.
    monkeypatch.setattr(
        module, "_reparse_tag",
        lambda path: 0xA0000003 if Path(path) == ancestor else 0)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=calls))
    assert error.value.reason == "root_reparse"
    assert calls == []
    assert not root.exists()


def test_uninspectable_root_metadata_refuses_closed(tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"
    locked = tmp_path
    original = module.os.lstat

    def failing(path, *args, **kwargs):
        if Path(path) == locked:
            raise PermissionError("synthetic uninspectable metadata")
        return original(path, *args, **kwargs)

    # Metadata-only mock: an lstat error on the chain must refuse closed instead
    # of being reported as an ordinary path.
    monkeypatch.setattr(module.os, "lstat", failing)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=[]))
    assert error.value.reason == "root_uninspectable"
    assert not root.exists()


def test_file_parent_component_refuses_uninspectable(tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls = []
    # A path component that is an ordinary file cannot be inspected: the OS
    # reports ``NotADirectoryError`` when a descendant path is probed. That is an
    # inspection failure, not an absent component, so the runner must refuse
    # closed rather than treat the root as ordinary and continue. The parent file
    # is real; only the descendant probe is mapped, so the control is
    # deterministic and no reparse point is created or inspected.
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_bytes(b"ordinary file, not a directory")
    root = parent_file / "run"
    original = module.os.lstat

    def file_parent_lstat(path, *args, **kwargs):
        if Path(path) == root:
            raise NotADirectoryError(f"not a directory: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "lstat", file_parent_lstat)
    with pytest.raises(module.DiagnosticRefusal) as direct:
        module._reparse_tag(root)
    assert direct.value.reason == "root_uninspectable"
    assert direct.value.field == "root"
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=calls))
    assert error.value.reason == "root_uninspectable"
    assert calls == []
    assert not root.exists()
    assert parent_file.read_bytes() == b"ordinary file, not a directory"


def test_existing_record_target_is_refused_and_not_overwritten(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    target = root / "record.json"
    target.write_text('{"foreign": true}', encoding="utf-8")
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(_synthetic_capture(), paths, calls=[]))
    assert error.value.reason == "output_exists"
    assert target.read_text(encoding="utf-8") == '{"foreign": true}'
    assert not (root / "record.json.partial").exists()


def test_atomic_publish_link_race_is_refused_and_cleans_partial(tmp_path,
                                                               monkeypatch):
    module = _runner()
    target = tmp_path / "record.json"

    def racing_link(source, destination, held):
        Path(destination).write_text("foreign", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(module, "_publish_owned_partial", racing_link)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._atomic_publish(target, {"schema": "synthetic"})
    assert error.value.reason == "output_exists"
    assert target.read_text(encoding="utf-8") == "foreign"
    assert not (tmp_path / "record.json.partial").exists()


def test_partial_name_collision_preserves_preexisting_bytes(tmp_path):
    module = _runner()
    target = tmp_path / "record.json"
    partial = tmp_path / "record.json.partial"
    partial.write_bytes(b"preexisting foreign partial")
    # A concurrent/pre-existing partial name must refuse and be left untouched:
    # this invocation never owned it, because its exclusive open did not succeed.
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._atomic_publish(target, {"schema": "synthetic"})
    assert error.value.reason == "output_exists"
    assert error.value.field == "record"
    assert partial.read_bytes() == b"preexisting foreign partial"
    assert not target.exists()


def test_record_target_race_after_precheck_leaves_target_untouched(tmp_path,
                                                                   monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    root = tmp_path / "run"
    target = root / "record.json"
    original = module._write_partial

    def racing(temporary, data):
        held = original(temporary, data)
        if not target.exists():
            target.write_text('{"foreign": true}', encoding="utf-8")
        return held

    monkeypatch.setattr(module, "_write_partial", racing)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[]))
    assert error.value.reason == "output_exists"
    assert target.read_text(encoding="utf-8") == '{"foreign": true}'
    assert not (root / "record.json.partial").exists()
    assert not (root / "route_b.mcscan").exists()
    assert not (root / "route_b_staging").exists()


# ---------------------------------------------------------------------------
# Failure atomicity / staging cleanup
# ---------------------------------------------------------------------------

def test_persisted_entry_output_is_cleaned_up_on_downstream_refusal(
        tmp_path, monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _synthetic_capture()
    root = tmp_path / "run"

    def exploding(capture, **kwargs):
        raise module.DiagnosticRefusal("replay_mismatch", "capture")

    monkeypatch.setattr(module, "replay_capture", exploding)
    with pytest.raises(module.DiagnosticRefusal):
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[]))
    _no_failure_artifacts(root)
    assert (root / "refusal.json").is_file()


def test_temporary_write_failure_leaves_no_partial_or_record(tmp_path,
                                                             monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    root = tmp_path / "run"

    def failing_fsync(descriptor):
        raise OSError("synthetic temporary write failure")

    # The real exclusive-open seam creates the partial, then the failing flush
    # removes exactly the file this invocation created.
    if os.name == "nt":
        from mc_photogrammetry.project.store import _WindowsRetainedFile
        monkeypatch.setattr(_WindowsRetainedFile, "flush", failing_fsync)
    else:
        monkeypatch.setattr(module.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        module.execute_route_b_diagnostic(
            image_paths=[str(path) for path in paths], root=root,
            api=_fake_api(capture, paths, calls=[]))
    _no_failure_artifacts(root)


def test_staging_cleanup_is_nonrecursive_and_only_exact_names(tmp_path):
    module = _runner()
    staging = tmp_path / "route_b_staging"
    staging.mkdir()
    staged = staging / "route_b.mcscan"
    staged.write_bytes(b"staged")
    (staging / "route_b.mcscan.tmp").write_bytes(b"partial")
    module._discard_staging(staging, staged)
    assert not staged.exists()
    assert not (staging / "route_b.mcscan.tmp").exists()
    assert not staging.exists()


# ---------------------------------------------------------------------------
# Count / order / content tamper refusal
# ---------------------------------------------------------------------------

def test_short_universe_refuses_atomically_and_writes_no_record(tmp_path):
    module = _runner()
    # The frozen collector itself owns the exact 1,928 expectation, so a short
    # universe is refused by the collector before it ever assembles pages.
    capture = _synthetic_capture(3, 1)
    assert capture.status == "refused"
    assert capture.root.reason == "universe_count_mismatch"
    assert capture.root.field == "row_count"
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, capture)
    assert error.value.reason == "universe_count_mismatch"
    assert error.value.field == "row_count"
    _no_failure_artifacts(tmp_path / "run")
    refusal = json.loads((tmp_path / "run" / "refusal.json").read_text(
        encoding="utf-8"))
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "universe_count_mismatch"


def test_reordered_admitted_order_refuses_atomically(tmp_path):
    module = _runner()
    capture = _synthetic_capture()
    rows = list(module.index_rows(capture))
    rows[0], rows[7] = rows[7], rows[0]
    object.__setattr__(capture.index_pages[0], "rows", tuple(rows))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, capture)
    assert error.value.reason in ("replay_mismatch", "universe_identity_mismatch")
    _no_failure_artifacts(tmp_path / "run")


def test_reordered_kept_content_refuses_the_sealed_universe_identity(tmp_path):
    module = _runner()
    capture = _synthetic_capture()
    rows = list(module.index_rows(capture))
    object.__setattr__(rows[3], "left", "tampered")
    object.__setattr__(capture.index_pages[0], "rows", tuple(rows))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, capture)
    assert error.value.reason == "replay_mismatch"
    _no_failure_artifacts(tmp_path / "run")


def test_sealed_whole_universe_identity_mismatch_refuses_atomically(tmp_path):
    module = _runner()
    capture = _synthetic_capture()
    object.__setattr__(capture.root, "record_sha256", "")
    object.__setattr__(capture.root, "whole_universe_identity_sha256", "0" * 64)
    object.__setattr__(capture.root, "record_sha256", v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(capture.root, include_digest=False)))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, capture)
    assert error.value.reason == "replay_mismatch"
    _no_failure_artifacts(tmp_path / "run")


def test_sealed_universe_identity_is_the_only_content_gate(tmp_path):
    """No caller pin exists; the sealed/replayed digest plus count is the gate."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)
    body = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    assert report["expected_row_count"] == EXPECTED_ROW_COUNT
    assert body["expected_row_count"] == EXPECTED_ROW_COUNT
    assert report["whole_universe_identity_sha256"] == (
        capture.root.whole_universe_identity_sha256)
    assert body["whole_universe_identity_sha256"] == (
        capture.root.whole_universe_identity_sha256)
    # The runner independently recomputes the same digest from the persisted
    # ordered index rows; nothing external can weaken or replace it.
    recomputed = v2.complete_support_universe_digest(module.index_rows(capture))
    assert body["whole_universe_identity_sha256"] == recomputed


# ---------------------------------------------------------------------------
# Cap atomicity
# ---------------------------------------------------------------------------

def test_cap_failure_refuses_atomically_and_writes_no_record(tmp_path,
                                                            monkeypatch):
    module = _runner()
    cases = (
        ("index_page_over_cap", {"index_page_max_bytes": 10}),
        ("digest_page_over_cap", {"digest_page_max_bytes": 10}),
        ("sidecar_page_over_cap", {"sidecar_page_max_bytes": 10}),
    )
    for index, (reason, overrides) in enumerate(cases):
        over_cap = v2.CompleteSupportCapPlanV1(**{
            **v2.ROUTE_B_DIAGNOSTIC_CAP_PLAN.as_record(), **overrides})
        capture = _synthetic_capture(plan=over_cap)
        assert capture.status == "refused", reason
        assert capture.root.reason == reason, reason
        assert capture.row_count == 0 and capture.index_pages == ()
        root = tmp_path / f"case-{index}"
        with pytest.raises(module.DiagnosticRefusal) as error:
            _execute(module, tmp_path, capture, root=root)
        assert error.value.reason == reason
        _no_failure_artifacts(root)
        refusal = json.loads((root / "refusal.json").read_text(encoding="utf-8"))
        assert refusal["reason"] == reason


def test_record_is_written_only_after_the_replay_gate(tmp_path, monkeypatch):
    module = _runner()

    def exploding(candidate, **kwargs):
        raise module.DiagnosticRefusal("replay_mismatch", "capture")

    monkeypatch.setattr(module, "replay_capture", exploding)
    with pytest.raises(module.DiagnosticRefusal):
        _execute(module, tmp_path, _synthetic_capture())
    _no_failure_artifacts(tmp_path / "run")
    assert (tmp_path / "run" / "refusal.json").is_file()


# ---------------------------------------------------------------------------
# Real seam parity and production pins
# ---------------------------------------------------------------------------

def test_real_graph_seam_is_output_neutral_with_the_v2_sink(tmp_path):
    """Runtime default-parity control on the real seam (no capture, no pixels)."""
    module = _runner()
    from test_candidate_support_provenance_v2_preflight import (
        _graph_seam_material, _graph_seam_inputs, _install_graph_seam_fakes)
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod

    patch, inputs = _graph_seam_inputs()
    try:
        matches, rows, _dispositions, bindings, observation, solution = (
            _graph_seam_material(EXPECTED_ROW_COUNT, 2))
        _install_graph_seam_fakes(patch, matches, rows, observation, solution)
        plain = graph_mod._derive_image_derived_graph(**inputs)
        captured: list = []
        with_sink = graph_mod._derive_image_derived_graph(
            **inputs, complete_support_sink=captured.append,
            signature_witnesses=bindings)
        assert with_sink == plain
        assert len(captured) == 1
        assert captured[0].row_count == EXPECTED_ROW_COUNT
        assert module.replay_capture(captured[0]) is not None
    finally:
        patch.undo()


def test_real_graph_seam_observer_is_output_neutral_with_the_v2_sink(tmp_path):
    """The same-pass Phase A observer changes no result, capture or digest."""
    module = _runner()
    from test_candidate_support_provenance_v2_preflight import (
        _graph_seam_material, _graph_seam_inputs, _install_graph_seam_fakes)
    from mc_photogrammetry.pipeline import (
        image_derived_graph_attribution as attribution_mod)

    patch, inputs = _graph_seam_inputs()
    try:
        matches, rows, _dispositions, bindings, observation, solution = (
            _graph_seam_material(EXPECTED_ROW_COUNT, 2))
        _install_graph_seam_fakes(patch, matches, rows, observation, solution)
        plain_result, plain_trace = (
            attribution_mod.derive_image_derived_graph_attribution(
                **inputs, signature_witnesses=bindings))
        observed_captures: list = []
        seen: list = []
        observed_result, observed_trace = (
            attribution_mod.derive_image_derived_graph_attribution(
                **inputs, signature_witnesses=bindings,
                complete_support_sink=observed_captures.append,
                solver_failure_observer=seen.append))
        neutral_captures: list = []
        neutral_result, neutral_trace = (
            attribution_mod.derive_image_derived_graph_attribution(
                **inputs, signature_witnesses=bindings,
                complete_support_sink=neutral_captures.append))
        assert observed_result.result_sha256 == neutral_result.result_sha256
        assert observed_trace.to_record() == neutral_trace.to_record()
        assert plain_trace.to_record() == neutral_trace.to_record()
        assert len(observed_captures) == len(neutral_captures) == 1
        assert (module.replay_capture(observed_captures[0])
                == module.replay_capture(neutral_captures[0]))
        # The observer fired exactly once from the same abstaining solver pass.
        assert len(seen) == 1
        assert observed_trace.solver_failure is not None
    finally:
        patch.undo()


def test_production_default_and_public_sources_stay_byte_identical(tmp_path):
    module = _runner()
    for relative, digest in PRODUCTION_PINS.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual.upper() == digest, relative
    source = RUNNER_PATH.read_text(encoding="utf-8")
    # Exactly one call per ordinary derivation; the runner adds no second pass.
    for name in ("derive_image_derived_signatures(",
                 "derive_image_derived_graph_attribution("):
        assert source.count(name) == 1, name
    assert "complete_support_sink=captured.append" in source
    assert "signature_witness_sink=signature_witnesses" in source
    # The public graph producer is never called with the diagnostic sink.
    assert "derive_image_derived_graph(" not in source
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    parameters = graph_mod.derive_image_derived_graph.__code__.co_varnames
    assert "complete_support_sink" not in parameters


# ---------------------------------------------------------------------------
# Serialized record replay validation
# ---------------------------------------------------------------------------

def test_written_record_replays_and_tampering_after_serialization_refuses(
        tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)
    record_path = Path(report["record"])
    document = module.validate_record(record_path)
    assert document["expected_row_count"] == EXPECTED_ROW_COUNT
    assert document["capture"]["root"]["row_count"] == EXPECTED_ROW_COUNT

    # A content tamper after serialization fails the rebuild/replay, not a stale
    # digest.
    tampered = json.loads(record_path.read_text(encoding="utf-8"))
    entry = tampered["capture"]["sidecar_pages"][0]["entries"][0]
    entry["payload"]["logical_pairs"]["values"][0][0] += 1
    body = dict(tampered)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = module._digest(
        {key: value for key, value in body.items() if key != "artifact_sha256"})
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.DiagnosticRefusal):
        module.validate_record(tampered_path)

    # A top-level binding tamper is caught by the artifact digest first.
    broken = json.loads(record_path.read_text(encoding="utf-8"))
    broken["whole_universe_identity_sha256"] = "0" * 64
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.validate_record(broken_path)
    assert error.value.reason == "replay_mismatch"
    assert error.value.field == "artifact_sha256"


def _rewrite_with_recomputed_digest(module, document, path):
    """Write ``document`` after recomputing its outer artifact digest."""
    body = {key: value for key, value in document.items()
            if key != "artifact_sha256"}
    document["artifact_sha256"] = module._digest(body)
    Path(path).write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return Path(path)


def test_serialized_outer_ingress_tamper_refuses_after_digest_recompute(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)
    record_path = Path(report["record"])
    module.validate_record(record_path)

    # Change one outer positional input row and recompute the outer artifact
    # digest: the persisted capture-root ingress no longer matches it.
    inputs_tampered = json.loads(record_path.read_text(encoding="utf-8"))
    foreign = _synthetic_pngs(tmp_path / "foreign")
    for path in foreign:
        Path(path).write_bytes(_png_bytes((9, 9, 9)))
    foreign_pins = _session_pins(foreign)
    inputs_tampered["inputs"][0]["sha256"] = foreign_pins[0].byte_sha256
    inputs_path = _rewrite_with_recomputed_digest(
        module, inputs_tampered, tmp_path / "inputs-tampered.json")
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.validate_record(inputs_path)
    assert error.value.reason == "ingress_binding_mismatch"
    assert error.value.field == "record.inputs"

    # Permute the serialized top-level ingress records and recompute the digest:
    # the positional cross-bind refuses the reordered outer rows.
    ingress_tampered = json.loads(record_path.read_text(encoding="utf-8"))
    ingress_tampered["ingress"][0], ingress_tampered["ingress"][1] = (
        ingress_tampered["ingress"][1], ingress_tampered["ingress"][0])
    ingress_path = _rewrite_with_recomputed_digest(
        module, ingress_tampered, tmp_path / "ingress-tampered.json")
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.validate_record(ingress_path)
    assert error.value.reason == "ingress_binding_mismatch"
    assert error.value.field == "record.ingress"


# ---------------------------------------------------------------------------
# V2 sampling-witness persistence (gate 2) and the bounded replay path
# ---------------------------------------------------------------------------

def test_v2_record_persists_the_exact_sampler_witnesses_without_changing_support(
        tmp_path):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    # The record's two ordered witness bindings are the entry session's own root
    # pins, so the same bindings decode the persisted section.
    bindings = _route_b_witness_bindings(_session_pins(paths))
    report = _execute(module, tmp_path, capture, paths=paths)

    document = json.loads(
        (tmp_path / "run" / "record.json").read_text(encoding="utf-8"))
    assert document["schema"] == module.RECORD_SCHEMA == (
        "mc-photogrammetry-route-b-diagnostic-record-v2")
    # One pass, one record: the support capture and every disposition are
    # byte-identical to the accepted V2 capture.
    assert document["capture"] == module.replay_capture(capture)
    assert report["calls"] == ["entry", "lattice", "signatures", "graph"]
    assert report["signature_witness_count"] == 2
    assert report["sampling_persistence"]["status"] == "complete"
    assert report["sampling_persistence"]["entry_count"] == 2
    assert report["sampling_persistence"]["bound_count"] == 2
    assert report["sampling_persistence"]["persistence_refused_count"] == 0

    section = document["sampling_persistence"]
    persistence.cross_bind_witness_summaries(
        section, document["signature_witnesses"])
    decoded = persistence.decode_section(
        section, ingress_by_frame={item.ingress.frame_id: item.ingress
                                   for item in bindings})
    assert decoded.bound_count == 2
    for entry, binding in zip(decoded.entries, bindings):
        assert entry.frame_id == binding.frame_id
        assert entry.witness.witness_sha256 == binding.witness.witness_sha256
        np.testing.assert_array_equal(entry.witness.cell_source_ptr,
                                      binding.witness.cell_source_ptr)
        np.testing.assert_array_equal(entry.witness.cell_source_indices,
                                      binding.witness.cell_source_indices)
        assert (entry.witness.rgb_rectification.witness_sha256
                == binding.witness.rgb_rectification.witness_sha256)
        np.testing.assert_array_equal(
            entry.witness.rgb_rectification.tap_source_indices,
            binding.witness.rgb_rectification.tap_source_indices)

    # The on-disk replay validator re-derives the same section.
    replayed = module.validate_record(report["record"])
    assert replayed["sampling_persistence"] == section


def test_v1_legacy_record_validates_without_a_sampling_section(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)
    document = json.loads((tmp_path / "run" / "record.json").read_text(
        encoding="utf-8"))
    document["schema"] = module.LEGACY_RECORD_SCHEMA
    document.pop("sampling_persistence")
    legacy = _rewrite_with_recomputed_digest(
        module, document, tmp_path / "legacy.json")
    replayed = module.validate_record(legacy)
    assert replayed["schema"] == module.LEGACY_RECORD_SCHEMA
    assert "sampling_persistence" not in replayed


def test_resigned_sampling_array_tamper_refuses_on_replay(tmp_path):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)
    document = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    section = document["sampling_persistence"]
    array = section["entries"][0]["witness"]["rgb_rectification"]["map_x"]
    raw = bytearray(base64.b64decode(array["b64"]))
    raw[0] ^= 0x01
    array["b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    array["sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
    # Re-sign both the section and the outer artifact: the tamper still refuses.
    section["section_sha256"] = persistence.section_digest(section)
    document["artifact_sha256"] = module._digest(
        {key: value for key, value in document.items()
         if key != "artifact_sha256"})
    tampered = tmp_path / "tampered-sampling.json"
    tampered.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.DiagnosticRefusal):
        module.validate_record(tampered)


def test_validate_record_uses_a_bounded_read_not_read_text(tmp_path,
                                                           monkeypatch):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _execute(module, tmp_path, capture, paths=paths)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("unbounded Path.read_text must not be used")

    monkeypatch.setattr(Path, "read_text", _forbidden)
    document = module.validate_record(report["record"])
    assert document["schema"] == module.RECORD_SCHEMA


def test_post_write_sampling_replay_failure_rolls_back_the_owned_record(
        tmp_path, monkeypatch):
    """A failure *after* publication removes the record this call created."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    root = tmp_path / "run"
    real = module._replay_sampling_section
    calls = {"count": 0}

    def flaky(document, capture=None):
        calls["count"] += 1
        if calls["count"] >= 2:
            # The first replay is the pre-publication gate; the second is the
            # post-write replay that owns the just-published artifact.
            raise module.DiagnosticRefusal(
                "persistence_digest_mismatch", "section_sha256")
        return real(document, capture)

    monkeypatch.setattr(module, "_replay_sampling_section", flaky)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _execute(module, tmp_path, capture, paths=paths, root=root)
    assert error.value.reason == "persistence_digest_mismatch"
    assert calls["count"] == 2
    # The second replay failed after ``_write_record`` published the artifact, so
    # the record must be rolled back by its own writer before the caller's own
    # rollback (which only knows about the project) runs.
    assert not (root / "record.json").exists()
    assert not (root / "record.json.partial").exists()
    _no_failure_artifacts(root)
    assert (root / "refusal.json").is_file()


def test_record_cap_is_decided_before_the_complete_record_is_serialized(
        tmp_path, monkeypatch):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    report = _execute(module, tmp_path, _capture_for(paths), paths=paths)
    payload = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    # The declared size used for the cap is exactly the serialized size.
    assert module._record_declared_size(payload) == len(module._record_bytes(payload))

    full = module._record_declared_size(payload)
    monkeypatch.setattr(module, "MAX_RECORD_BYTES", full - 1)
    real = module._canonical_bytes
    sizes = []

    def spy(value):
        sizes.append(persistence.canonical_json_size(value))
        return real(value)

    monkeypatch.setattr(module, "_canonical_bytes", spy)
    fitted = module._fit_sampling_section(payload)
    assert fitted["sampling_persistence"]["status"] == "refused"
    assert fitted["sampling_persistence"]["reason"] == (
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP)
    assert module._record_declared_size(fitted) < full
    # The over-cap complete payload was never serialized by the runner at all:
    # the whole decision (including the degraded section's own digest) went
    # through the declared-size path and the persistence module's small record.
    assert sizes == []


def test_write_record_rejects_over_cap_before_the_artifact_digest(
        tmp_path, monkeypatch):
    """A direct ``_write_record`` call must cap before the artifact digest runs."""
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    report = _execute(module, tmp_path, _capture_for(paths), paths=paths)
    payload = json.loads(Path(report["record"]).read_text(encoding="utf-8"))
    payload.pop("artifact_sha256")
    declared = module._record_declared_size(payload)
    root = tmp_path / "caproot"
    root.mkdir()
    # The payload alone fits the cap exactly, so ``_fit_sampling_section`` is a
    # no-op, but the artifact digest's fixed 64-hex field pushes the complete
    # record over; ``_write_record`` must reject before serializing anything.
    monkeypatch.setattr(module, "MAX_RECORD_BYTES", declared)
    real_bytes = module._canonical_bytes
    real_digest = module._digest
    calls = {"count": 0}

    def spy_bytes(value):
        calls["count"] += 1
        return real_bytes(value)

    def spy_digest(value):
        calls["count"] += 1
        return real_digest(value)

    monkeypatch.setattr(module, "_canonical_bytes", spy_bytes)
    monkeypatch.setattr(module, "_digest", spy_digest)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._write_record(root, payload, canonical_capture=None)
    assert error.value.reason == "record_over_cap"
    assert error.value.field == "record"
    assert calls["count"] == 0
    assert not (root / "record.json").exists()
    assert not (root / "record.json.partial").exists()


def test_validate_record_caps_canonical_size_before_the_artifact_digest(
        tmp_path, monkeypatch):
    """A bounded raw read can still expand canonically past the record cap."""
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    module = _runner()
    # 6,000,000 U+00E9 code points are 12,000,000 raw UTF-8 bytes but 36,000,000
    # escaped JSON bytes under ``ensure_ascii=True``, so the bounded raw read
    # passes while the canonical record does not.
    document = {
        "schema": module.RECORD_SCHEMA,
        "status": "complete",
        "artifact_sha256": "0" * 64,
        "note": "\u00e9" * 6_000_000,
    }
    raw = json.dumps(document, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    assert len(raw) < module.MAX_RECORD_BYTES
    assert persistence.canonical_json_size(document) + 1 > module.MAX_RECORD_BYTES
    path = tmp_path / "expanded.json"
    path.write_bytes(b"{}")

    snapshot = persistence.BoundedRecordSnapshotV1(
        data=raw, sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw), max_json_depth=2, json_tokens=5)
    monkeypatch.setattr(persistence, "read_bounded_record_bytes",
                        lambda *_args, **_kwargs: snapshot)
    digest_calls = {"count": 0}

    def trap(payload):
        digest_calls["count"] += 1
        raise AssertionError(
            "the artifact digest must not serialize an over-cap record")

    monkeypatch.setattr(module, "_digest", trap)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.validate_record(path)
    assert error.value.reason == (
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP)
    assert error.value.field == "record"
    assert digest_calls["count"] == 0


# ---------------------------------------------------------------------------
# Corrective DS-IMG-T2-V2-REAL-STAGE-MEASUREMENT regressions (Req 5)
# ---------------------------------------------------------------------------

def _solver_snapshot_axes():
    return SimpleNamespace(
        reason="plane_integer_residual", invalid_field=None,
        frames=(("image-0000",),), quadrants=(),
        axis_resultants=((0, 0.5), (1, 0.6)),
        axis_residuals=((0, 0.125),), solve_context=None,
        active_edge_aliases=None, boundary_saturation=None)


def _pass_api(paths, *, calls, fire=0, lattice=None, signatures=None,
              graph_status="assembled"):
    """A minimal single-pass API whose graph fires ``fire`` complete captures."""
    pins = _session_pins(paths)
    bindings = _route_b_witness_bindings(pins)
    session = _session_for(paths, pins)

    def entry(input_paths, *, project_path=None, capture_ingress=False, project_max_bytes=None, retain_project_lease=False):
        calls.append("entry")
        Path(project_path).write_bytes(b'{"schema":"synthetic-route-b-project"}')
        return _entry_terminal(session)

    def lattice_call(*, session, pose_graph):
        calls.append("lattice")
        if lattice is not None:
            return lattice
        return SimpleNamespace(status="assembled", reason="",
                               result_sha256="3" * 64, readings=(1, 2, 3))

    def signatures_call(*, session, lattice_result, signature_witness_sink=None):
        calls.append("signatures")
        if signatures is not None:
            return signatures
        signature_witness_sink.extend(bindings)
        return SimpleNamespace(status="assembled", reason="",
                               result_sha256="4" * 64,
                               signature_count=len(bindings))

    def graph_call(*, session, lattice_result, signatures_result,
                   connection_graph, signature_witnesses=None,
                   complete_support_sink=None, solver_failure_observer=None,
                   **settings):
        calls.append("graph")
        if solver_failure_observer is not None:
            solver_failure_observer(_solver_snapshot_axes())
        for _ in range(fire):
            complete_support_sink(
                _capture_for(paths) if fire == 1 else object())
        return (SimpleNamespace(status=graph_status,
                                reason="plane_integer_residual", detail="",
                                result_sha256="5" * 64),
                SimpleNamespace(attribution_sha256="6" * 64, stage="solver",
                                post_admission_candidates=3,
                                post_same_axis_candidates=2,
                                proposal_count=7))

    return (entry, lattice_call, signatures_call, graph_call)


def test_runner_uses_the_legacy_capture_refusal_reason_not_a_phase_a_terminal(
        tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls: list = []
    checkpoints: list = []
    snapshots: list = []
    api = _pass_api(paths, calls=calls, fire=0)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._run_single_pass(
            resolved=paths, project_path=tmp_path / "project",
            settings=module.resolve_settings(), api=module.RouteBApi(*api),
            checkpoint=checkpoints.append, observer=snapshots)
    # The *legacy* reason/field are restored; the runner never invents the
    # Phase A assembled terminal itself.
    assert error.value.reason == "capture_unavailable"
    assert error.value.field == "capture"
    assert calls == ["entry", "lattice", "signatures", "graph"]
    assert calls.count("graph") == 1


def test_runner_persists_the_same_pass_graph_checkpoint_before_the_refusal(
        tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls: list = []
    checkpoints: list = []
    api = _pass_api(paths, calls=calls, fire=0)
    with pytest.raises(module.DiagnosticRefusal):
        module._run_single_pass(
            resolved=paths, project_path=tmp_path / "project",
            settings=module.resolve_settings(), api=module.RouteBApi(*api),
            checkpoint=checkpoints.append, observer=[],
            expected_pins=tuple(pin.byte_sha256 for pin in _session_pins(paths)))
    graph_checkpoints = [c for c in checkpoints if c["stage"] == "graph"]
    assert len(graph_checkpoints) == 1
    graph = graph_checkpoints[0]
    # The wrapper's assembled-terminal classification reads exactly this.
    assert graph["status"] == "assembled"
    assert graph["captured_count"] == 0
    assert graph["proposal_count"] == 7
    # Numeric axis residual/resultant diagnostics are present, bounded and are
    # the same pass's own numbers (no second solver pass).
    assert graph["solver"][0]["axis_resultants"] == [[0, 0.5], [1, 0.6]]
    assert graph["solver"][0]["axis_residuals"] == [[0, 0.125]]
    assert [c["stage"] for c in checkpoints] == [
        "entry", "pins", "lattice", "signatures", "graph"]


def test_runner_lattice_and_signature_checkpoints_precede_their_refusals(
        tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls: list = []
    checkpoints: list = []

    # A refused lattice still leaves its own reading-count checkpoint.
    refused_lattice = SimpleNamespace(status="refused", reason="lattice_refused",
                                      detail="no lattice", result_sha256="",
                                      readings=())
    api = _pass_api(paths, calls=calls, lattice=refused_lattice)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._run_single_pass(
            resolved=paths, project_path=tmp_path / "project",
            settings=module.resolve_settings(), api=module.RouteBApi(*api),
            checkpoint=checkpoints.append)
    assert error.value.reason == "lattice_refused"
    lattice_cp = [c for c in checkpoints if c["stage"] == "lattice"]
    assert len(lattice_cp) == 1
    assert lattice_cp[0]["status"] == "refused"
    assert lattice_cp[0]["reading_count"] == 0
    assert ["signatures", "graph"] != [c["stage"] for c in checkpoints]

    # A refused signature still leaves its own signature-count checkpoint.
    calls.clear()
    checkpoints.clear()
    refused_signatures = SimpleNamespace(status="refused",
                                         reason="signature_refused",
                                         detail="no signatures",
                                         result_sha256="")
    api = _pass_api(paths, calls=calls, signatures=refused_signatures)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._run_single_pass(
            resolved=paths, project_path=tmp_path / "project",
            settings=module.resolve_settings(), api=module.RouteBApi(*api),
            checkpoint=checkpoints.append)
    assert error.value.reason == "signature_refused"
    signature_cp = [c for c in checkpoints if c["stage"] == "signatures"]
    assert len(signature_cp) == 1
    assert signature_cp[0]["status"] == "refused"
    assert signature_cp[0]["witness_count"] == 0
    assert "graph" not in [c["stage"] for c in checkpoints]


def test_runner_duplicate_capture_is_the_legacy_refusal(tmp_path):
    module = _runner()
    paths = _synthetic_pngs(tmp_path)
    calls: list = []
    checkpoints: list = []
    api = _pass_api(paths, calls=calls, fire=2)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._run_single_pass(
            resolved=paths, project_path=tmp_path / "project",
            settings=module.resolve_settings(), api=module.RouteBApi(*api),
            checkpoint=checkpoints.append)
    assert error.value.reason == "capture_unavailable"
    graph = [c for c in checkpoints if c["stage"] == "graph"][0]
    assert graph["captured_count"] == 2


def test_solver_snapshot_summary_bounds_axis_diagnostics():
    module = _runner()
    snapshot = SimpleNamespace(
        reason="r", invalid_field=None, frames=(), quadrants=(),
        axis_resultants=tuple((axis, axis / 3.0) for axis in range(20)),
        axis_residuals=(), solve_context=None, active_edge_aliases=None,
        boundary_saturation=None)
    summary = module._solver_snapshot_summary(snapshot, max_axes=4)
    assert summary["axis_resultant_keys"] == [0, 1, 2, 3]
    assert len(summary["axis_resultants"]) == 4



# Native ownership tests use fixture regular files only, Events, and bounded joins.
def _native_barrier_pair(operation, race):
    from threading import Event, Thread
    ready, complete = Event(), Event()
    errors = []
    def racer():
        try:
            if not ready.wait(3):
                raise AssertionError("native mutation barrier was not reached")
            race()
        except BaseException as error:
            errors.append(error)
        finally:
            complete.set()
    thread = Thread(target=racer)
    thread.start()
    def barrier():
        ready.set()
        assert complete.wait(3), "native racer did not finish"
    try:
        result = operation(barrier)
    finally:
        ready.set()
        thread.join(3)
        assert not thread.is_alive()
    if errors:
        raise errors[0]
    return result


def _assert_native_replacement_blocked(target, proposed, saved):
    import os
    data = Path(proposed).read_bytes()
    for index, operation in enumerate((lambda: Path(target).rename(saved),
                      lambda: os.replace(proposed, target),
                      lambda: Path(target).write_bytes(data))):
        with pytest.raises(OSError) as error:
            operation()
        if index < 2:
            assert error.value.winerror in (5, 32), error.value
        else:
            assert error.value.errno == 13, error.value


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
def test_native_owned_partial_cleanup_blocks_equal_byte_replacement(tmp_path, monkeypatch):
    from mc_photogrammetry.project.store import _WindowsRetainedFile
    module = _runner()
    target = tmp_path / "record.json"
    partial = tmp_path / "record.json.partial"
    proposed = tmp_path / "foreign.json"
    proposed.write_bytes(b'{"x":1}\n')
    original = _WindowsRetainedFile.delete_owned
    def operation(barrier):
        def delete(held):
            if held.path == partial:
                held._verify()  # native ID, not a mocked pathname stat
                assert len(held.identity[1]) == 16
                barrier()
            return original(held)
        monkeypatch.setattr(_WindowsRetainedFile, "delete_owned", delete)
        def refusal(*args):
            raise FileExistsError("injected publication refusal")
        monkeypatch.setattr(module, "_publish_owned_partial", refusal)
        with pytest.raises(module.DiagnosticRefusal) as error:
            module._atomic_publish(target, {"x": 1})
        assert error.value.reason == "output_exists"
    _native_barrier_pair(operation, lambda: _assert_native_replacement_blocked(
        partial, proposed, tmp_path / "saved.json"))
    assert proposed.read_bytes() == b'{"x":1}\n'
    assert not partial.exists() and not target.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
def test_native_owned_rollback_blocks_equal_byte_replacement(tmp_path, monkeypatch):
    from mc_photogrammetry.project.store import _WindowsRetainedFile
    module = _runner()
    target, proposed = tmp_path / "record.json", tmp_path / "foreign.json"
    proposed.write_bytes(b'{"x":1}\n')
    ownership = module.InvocationOwnership(native=True)
    ownership.anchor(tmp_path)
    original = _WindowsRetainedFile.delete_owned
    try:
        module._atomic_publish(target, {"x": 1}, ownership=ownership)
        identity = ownership._handles[target].identity
        assert len(identity[1]) == 16
        def operation(barrier):
            def delete(held):
                if held.path == target:
                    assert held._identity() == identity
                    barrier()
                return original(held)
            monkeypatch.setattr(_WindowsRetainedFile, "delete_owned", delete)
            ownership.rollback()
        _native_barrier_pair(operation, lambda: _assert_native_replacement_blocked(
            target, proposed, tmp_path / "saved.json"))
    finally:
        ownership.close()
    assert not target.exists()
    assert proposed.read_bytes() == b'{"x":1}\n'


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
def test_native_no_replace_publication_preserves_racing_foreign_target(tmp_path, monkeypatch):
    from mc_photogrammetry.project.store import _WindowsRetainedFile
    module = _runner()
    target = tmp_path / "record.json"
    original = _WindowsRetainedFile.rename_no_replace
    def operation(barrier):
        def rename(held, destination):
            if held.path.name == "record.json.partial":
                held._verify()
                barrier()
            return original(held, destination)
        monkeypatch.setattr(_WindowsRetainedFile, "rename_no_replace", rename)
        with pytest.raises(module.DiagnosticRefusal) as error:
            module._atomic_publish(target, {"x": 1})
        assert error.value.reason == "output_exists"
    _native_barrier_pair(operation, lambda: target.write_bytes(b"foreign"))
    assert target.read_bytes() == b"foreign"
    assert not target.with_name(target.name + ".partial").exists()
