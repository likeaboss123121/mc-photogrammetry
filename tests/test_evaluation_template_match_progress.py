"""Focused synthetic tests for the Phase A template-match progress harness.

These tests run no real capture, no image inference, no evaluator/truth/pixel
source or Minecraft/Prism/network path. Fixed harmless child tests exercise
actual Windows Job/pipe controls separately; reconstruction harness semantics
use a synthetic fake supervisor and never launch image inference.

**Interim fail-closed state (Sol's final recommendation).** The default
supervisor and the ``--worker`` CLI entry both refuse the typed
``resource_enforcement_unavailable`` before ``api.entry``/``_execute_pass``; the
real ``WindowsJobSupervisor`` is disabled. Only the private synthetic test seam
may exercise the private harness semantics, and only when the injected supervisor
**and** the injected API both carry the explicit ``synthetic_test_only is True``
marker (a code-enforced test-only convention, not an authentication boundary).
This file adds regressions for the default pre-entry refusal, the disabled
Windows supervisor, the arbitrary-supervisor/``api=None`` refusal, the unmarked
API refusal, the ``_execute_pass(api=None)`` refusal, the direct worker refusal
and the zero-write root collision. The private seam itself is not claimed
accepted: its unresolved internal defects stay documented as re-enable blockers.

Coverage maps to the accepted Phase A requirements plus the corrective revision:
ordered same-pass T2 pin mismatch, all five terminals with no retry,
observer/output neutrality, checkpoint retention, caps and pre-entry resource
refusal, freeze-before-evaluator loading, missing/foreign alignment refusal,
persisted witness immutability, the ``-1``/missing UNKNOWN sentinel, overlapping
footprint dedup, hard whole-run containment refusal, ownership/no-clobber,
terminal-exact freeze, the renamed support-grid counter and the categorically
unavailable Phase B gate.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_template_match_progress as progress
from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

from test_route_b_diagnostic_runner import (
    EXPECTED_ROW_COUNT,
    _HEX,
    _HEX_B,
    _capture_for,
    _entry_terminal,
    _route_b_witness_bindings,
    _session_for,
    _session_pins,
    _synthetic_capture,
    _synthetic_pngs,
    _write_synthetic_project,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (REPO_ROOT / "scripts" / "evaluation"
                / "run_template_match_progress.py")
RUNNER_PATH = REPO_ROOT / "scripts" / "evaluation" / "run_route_b_diagnostic.py"
REPORT_PATH = (REPO_ROOT / "docs" / "reports"
               / "image-only-route-b-stage-loss-run-2026-10-04.md")


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
def test_native_harmless_job_assignment_membership_and_failure():
    module = _harness()
    supervisor = module.WindowsJobSupervisor()
    good = supervisor.run_control("ok")
    assert good.exit_code == 0
    assert good.job_assigned_before_start and good.job_terminated
    assert good.report["specific_job_verified"]
    assert good.report["memory_metric"] == "committed_bytes"
    assert b"OPTIMIZE=1\n" in good.report["output"]
    failed = supervisor.run_control("ok", fail_assignment=True)
    assert not failed.job_assigned_before_start
    assert failed.exit_code == 1 and failed.stream_bytes == 0
    assert module.WindowsJobSupervisor.disabled  # native proof never enables inference


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
def test_native_harmless_job_timeout_and_root_memory_limit():
    module = _harness()
    supervisor = module.WindowsJobSupervisor()
    timeout = supervisor.run_control("timeout", wall_seconds=.15)
    assert timeout.timed_out and timeout.job_terminated
    memory = supervisor.run_control("memory", memory_bytes=96 * 1024 * 1024)
    assert memory.memory_exceeded and memory.report["specific_job_verified"]



@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
@pytest.mark.parametrize("membership_check,exit_code", [("wrong", 22), ("missing", 21)])
@pytest.mark.parametrize("optimization", ["1", "2"])
def test_native_harmless_job_membership_failure_under_optimization(
        monkeypatch, membership_check, exit_code, optimization):
    monkeypatch.setenv("PYTHONOPTIMIZE", optimization)
    module = _harness()
    supervisor = module.WindowsJobSupervisor()
    good = supervisor.run_control("ok")
    assert good.exit_code == 0 and good.report["specific_job_verified"]
    # Explicit -O is used even without the environment; the verifier runs optimized.
    assert ("OPTIMIZE=" + optimization + "\n").encode() in good.report["output"]
    failed = supervisor.run_control("ok", membership_check=membership_check)
    assert failed.job_assigned_before_start and failed.job_terminated
    assert failed.exit_code == exit_code
    assert failed.stream_bytes == 0 and failed.report["output"] == b""
    assert not failed.report["specific_job_verified"]
    assert module.WindowsJobSupervisor.disabled


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
@pytest.mark.parametrize("mode", ["burst", "burst_reverse"])
def test_native_harmless_job_isolated_combined_stream_overflow(mode):
    module = _harness()
    cap = 64
    # Marker+stdout=32; stderr=33. Neither stream independently crosses cap.
    stdout_total = (cap + 1) // 2
    stderr_total = cap + 1 - stdout_total
    assert len(b"JOB_OK\n") <= stdout_total <= cap and stderr_total <= cap
    burst = module.WindowsJobSupervisor().run_control(mode, stream_max_bytes=cap)
    assert burst.stream_exceeded and burst.job_terminated
    assert not burst.timed_out
    assert burst.stream_bytes == cap + 1
    assert len(burst.report["output"]) <= cap
    assert module.WindowsJobSupervisor.disabled


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
def test_native_harmless_job_descendant_aggregate_committed_memory_limit():
    module = _harness()
    limit = 128 * 1024 * 1024
    result = module.WindowsJobSupervisor().run_control(
        "memory_tree", memory_bytes=limit, wall_seconds=3)
    assert result.exit_code == 0 and not result.timed_out and not result.stream_exceeded
    assert result.job_assigned_before_start and result.job_terminated
    assert result.report["specific_job_verified"] and result.memory_exceeded
    output = result.report["output"]
    assert output.count(b"JOB_OK\n") == 2  # root AND descendant verify this specific Job
    values = dict(line.split(b"=", 1) for line in output.splitlines() if b"=" in line)
    root_commit = int(values[b"TREE_ROOT_COMMIT"])
    leaf_projected = int(values[b"TREE_LEAF_PROJECTED"])
    request = int(values[b"TREE_REQUEST"])
    assert root_commit < limit and leaf_projected < limit and request < limit
    assert root_commit + leaf_projected > limit
    assert b"TREE_MEMORY_REFUSED\n" in output
    assert b"TREE_ALLOCATION_SUCCEEDED" not in output
    # Native completion-port messages distinguish Job aggregate from process limit.
    assert result.report["job_memory_limit_observed"]
    assert not result.report["process_memory_limit_observed"]
    # Windows peak telemetry can exceed the configured limit for this refused run;
    # the native notification/refusal proves disposition, not a peak/RSS ceiling.
    assert result.peak_job_bytes > 0
    assert result.report["memory_metric"] == "committed_bytes"
    assert module.WindowsJobSupervisor.disabled


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence required")
def test_native_harmless_job_kills_descendant_after_primary_exit():
    import ctypes
    from ctypes import wintypes as w
    module = _harness()
    result = module.WindowsJobSupervisor().run_control("descendant")
    assert result.exit_code == 0 and result.job_terminated
    pid = int(result.report["output"].split(b"DESCENDANT=")[1].splitlines()[0])
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.GetExitCodeProcess.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        try:
            code = w.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            assert code.value != 259
        finally:
            kernel.CloseHandle(handle)


def _harness():
    name = "_template_match_progress_harness_under_test"
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_passes_limit_into_actual_entry_contract(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    api = _instrumented_api(capture, paths, calls=[])
    seen = []
    def entry(paths, **kwargs):
        seen.append(kwargs["project_max_bytes"])
        return api[0](paths, **kwargs)
    wrapped = _SyntheticApi((entry, *api[1:]))
    _run(module, tmp_path, capture, paths=paths, api=wrapped)
    assert seen == [module.ResourceLimits().project_bytes]


def test_caller_root_and_every_extra_partial_are_rejected(tmp_path):
    module = _harness()
    original = tmp_path / "original"
    original.mkdir()
    _refusal_root(original)
    module.freeze_root(original, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "freeze.json").write_bytes((original / "freeze.json").read_bytes())
    with pytest.raises(progress.TemplateMatchProgressError, match="caller root mismatch"):
        module.load_freeze(copied)
    for name in ("foreign", "diagnostics.json.partial"):
        extra = original / name
        extra.write_bytes(b"foreign")
        with pytest.raises(progress.TemplateMatchProgressError, match="unexpected"):
            module.load_freeze(original)
        assert extra.read_bytes() == b"foreign"
        extra.unlink()


def test_identity_rollback_preserves_equal_byte_foreign_replacement(tmp_path):
    runner = _runner()
    target = tmp_path / "owned.json"
    target.write_bytes(b"identical")
    owned = runner.InvocationOwnership()
    owned.register(target)
    target.rename(tmp_path / "original.json")
    target.write_bytes(b"identical")
    owned.rollback()
    assert target.read_bytes() == b"identical"
    assert (tmp_path / "original.json").read_bytes() == b"identical"


def test_output_replacement_peak_uses_actual_old_size():
    module = _harness()
    budget = module.OutputBudget(module.ResourceLimits(total_output_bytes=10))
    budget.charge(8, field="diagnostics.json")
    with pytest.raises(module.DiagnosticRefusal, match="replacement peak"):
        budget.replace(3, field="diagnostics.json")
    assert budget.total == 8
    budget.replace(2, field="diagnostics.json")
    assert budget.total == 2 and budget.peak_bytes == 10


def test_journal_prescan_refuses_before_json_serialization(tmp_path, monkeypatch):
    module = _harness()
    journal = module.DiagnosticsJournal(tmp_path / "diagnostics.json", max_bytes=10,
                                        terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    def trap(*args, **kwargs):
        raise AssertionError("oversized JSON must not serialize")
    monkeypatch.setattr(module.json, "dumps", trap)
    with pytest.raises(module.DiagnosticRefusal, match="diagnostics_over_cap"):
        journal.append({"detail": "x" * 100})
    assert not list(tmp_path.iterdir())


def test_resealed_scorecard_cannot_change_record_derived_counts(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    report = _run(module, tmp_path, capture, paths=paths)
    root = tmp_path / "run"
    score = json.loads((root / "scorecard.json").read_text())
    score["candidate_rows"]["admitted_total"] += 1
    score["scorecard_sha256"] = progress.scorecard_digest(score)
    (root / "scorecard.json").write_text(json.dumps(score), encoding="utf-8")
    forged = json.loads((root / "freeze.json").read_text())
    for entry in forged["artifacts"]:
        path = root / entry["path"]
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["size"] = path.stat().st_size
    forged["freeze_sha256"] = progress._sha256_of(forged, drop="freeze_sha256")
    (root / "freeze.json").write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(progress.TemplateMatchProgressError, match="binding"):
        module.revalidate_freeze(forged, root=root)


def test_parent_refuses_nonzero_exit_despite_complete_frozen_artifacts(tmp_path):
    module = _harness()
    class FailedExit(_FakeSupervisor):
        def run(self, spec):
            result = super().run(spec)
            result.exit_code = 7
            return result
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, paths=paths,
             supervisor=FailedExit(module))
    assert error.value.reason == "child_abnormal_exit"


def test_bounded_json_reader_types_deep_and_nonobject_refusals(tmp_path):
    module = _harness()
    target = tmp_path / "diagnostics.json"
    for data in (b"true", b"[" * 1100 + b"0" + b"]" * 1100):
        target.write_bytes(data)
        with pytest.raises(module.DiagnosticRefusal) as error:
            module._read_json_bounded(target)
        assert error.value.reason == "artifact_malformed"


def test_manifest_cannot_raise_trusted_artifact_cap(tmp_path):
    module = _harness()
    _refusal_root(tmp_path)
    freeze = module.freeze_root(tmp_path, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    freeze["artifacts"][0]["cap_bytes"] = 10 ** 12
    freeze["artifacts"][0]["size"] = progress.FREEZE_ARTIFACT_CAPS["diagnostics.json"] + 1
    freeze["freeze_sha256"] = progress._sha256_of(freeze, drop="freeze_sha256")
    (tmp_path / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(progress.TemplateMatchProgressError, match="typed cap"):
        module.revalidate_freeze(freeze, root=tmp_path)


def test_bounded_entry_persistence_refusal_keeps_typed_terminal(tmp_path):
    from mc_photogrammetry.pipeline.image_derived_entry import ImageDerivedEntryError
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    api = _instrumented_api(capture, paths, calls=[])
    def entry(paths, **kwargs):
        raise ImageDerivedEntryError("entry_persist_failed", "project_over_cap")
    wrapped = _SyntheticApi((entry, *api[1:]))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, paths=paths, api=wrapped)
    assert error.value.reason == "entry_persist_failed"
    root = tmp_path / "run"
    journal = json.loads((root / "diagnostics.json").read_text())
    assert journal["checkpoints"][0]["reason"] == "entry_persist_failed"
    assert module.load_freeze(root)["terminal"] == progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE


def _runner():
    name = "_route_b_runner_for_phase_a_tests"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSupervisor:
    """Synthetic fake supervisor: no real child, no real inference.

    It records that the worker is only entered after an (emulated) job
    assignment, and it can emulate an OS-level deadline/memory/stream abort that
    is observed externally - exactly the shape the harness must consume.
    """

    kind = "synthetic_fake_supervisor"
    spawns_child = False
    can_contain = True
    # The code-enforced private-seam marker: only an object that explicitly
    # self-identifies as a test double may open the harness pass.
    synthetic_test_only = True

    def __init__(self, module, *, available=True, timed_out=False,
                 memory_exceeded=False, stream_exceeded=False,
                 omit_assignment=False, peak_job_bytes=0, stream_bytes=0):
        self.module = module
        self.available = available
        self.timed_out = timed_out
        self.memory_exceeded = memory_exceeded
        self.stream_exceeded = stream_exceeded
        self.omit_assignment = omit_assignment
        self.peak_job_bytes = peak_job_bytes
        self.stream_bytes = stream_bytes
        self.spec = None
        self.ensure_calls = 0
        self.run_calls = 0
        self.worker_ran = False
        self.assigned_before_worker = False

    def ensure_available(self):
        self.ensure_calls += 1
        if not self.available:
            raise self.module.ContainmentUnavailable("synthetic unavailable")

    def run(self, spec):
        self.run_calls += 1
        self.spec = spec
        if self.timed_out or self.memory_exceeded or self.stream_exceeded:
            return self.module.SupervisedOutcome(
                job_assigned_before_start=True, timed_out=self.timed_out,
                memory_exceeded=self.memory_exceeded,
                stream_exceeded=self.stream_exceeded,
                peak_job_bytes=self.peak_job_bytes,
                stream_bytes=self.stream_bytes)
        assigned = not self.omit_assignment
        self.assigned_before_worker = assigned
        outcome = self.module.SupervisedOutcome(
            job_assigned_before_start=assigned, exit_code=0)
        self.worker_ran = True
        try:
            outcome.report = spec.worker()
        except BaseException as exc:  # noqa: BLE001 - synthetic capture
            outcome.error = exc
            outcome.exit_code = 1
        return outcome


class _SyntheticApi(tuple):
    """Length-4 synthetic route-B api tuple carrying the test-seam marker.

    The runner resolves any plain length-4 tuple; the harness refuses it unless it
    (and the supervisor) explicitly self-identify as test doubles. Keeping the
    marker on the tuple subclass preserves the runner's tuple contract unchanged.
    """

    synthetic_test_only = True


def _pin(value):
    return SimpleNamespace(byte_sha256=value)


def _row(index, *, disposition="kept", support="present", values=None,
         warp=None):
    row = {"admitted_row_index": index, "disposition": disposition}
    if support == "present":
        payload = {}
        if values is not None:
            payload["logical_pairs"] = {"values": values}
        if warp is not None:
            payload["warp_source_rows"] = {"values": warp}
            payload["warp_source_columns"] = {"values": warp}
        row["support"] = payload
        row["support_refusal_reason"] = None
    elif support == "refused":
        row["support"] = None
        row["support_refusal_reason"] = "matcher_refused"
    else:
        row["support"] = None
        row["support_refusal_reason"] = None
    return row


def _synthetic_bindings(paths):
    pins = _session_pins(paths)
    return pins, _route_b_witness_bindings(pins)


def _section_and_ingress(bindings):
    section = persistence.persist_signature_witnesses(bindings)
    ingress = {binding.ingress.frame_id: binding.ingress for binding in bindings}
    return section, ingress


class _AbortingGovernor:
    def __init__(self, module):
        self.module = module
        self.limits = module.ResourceLimits()
        self.checks = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def check(self):
        self.checks += 1
        raise self.module.ResourceAbort(
            "resource_wall_exceeded", "wall_seconds", "synthetic abort")

    def record(self):
        return {"limits": self.limits.record(), "checks": self.checks,
                "elapsed_seconds": 0.0, "peak_process_tree_bytes": 0}


def _snapshot():
    return SimpleNamespace(
        reason="plane_integer_residual", invalid_field=None,
        frames=(("image-0000",),), quadrants=(),
        axis_resultants=((0, 0.5), (1, 0.6), (2, 0.7)),
        axis_residuals=((0, 0.1),), solve_context=None,
        active_edge_aliases=None, boundary_saturation=None)


def _instrumented_api(capture, paths, *, calls, snapshot=None, fire=True):
    pins = _session_pins(paths)
    bindings = _route_b_witness_bindings(pins)
    session = _session_for(paths, pins)

    def entry(input_paths, *, project_path=None, capture_ingress=False, project_max_bytes=None, retain_project_lease=False):
        calls.append("entry")
        payload = json.dumps({"schema_version": 2,
            "source_paths": [str(path) for path in paths],
            "image_derived_session": {"session_sha256": session.session_sha256,
                "frames": [{"frame_id": pin.frame_id, "ingress": pin.to_record()} for pin in pins]}}).encode("utf-8")
        lease = _write_synthetic_project(project_path, payload, project_max_bytes, retain_project_lease)
        result = _entry_terminal(session)
        result.project_lease = lease
        return result

    def lattice(*, session, pose_graph):
        calls.append("lattice")
        return SimpleNamespace(status="assembled", reason="",
                               result_sha256="3" * 64,
                               readings=(SimpleNamespace(axis=1),
                                         SimpleNamespace(axis=2)))

    def signatures(*, session, lattice_result, signature_witness_sink=None):
        calls.append("signatures")
        signature_witness_sink.extend(bindings)
        return SimpleNamespace(status="assembled", reason="",
                               result_sha256="4" * 64,
                               signature_count=len(bindings))

    def graph(*, session, lattice_result, signatures_result, connection_graph,
              signature_witnesses=None, complete_support_sink=None,
              solver_failure_observer=None, **settings):
        calls.append("graph")
        if fire:
            complete_support_sink(capture)
        if snapshot is not None and solver_failure_observer is not None:
            solver_failure_observer(snapshot)
        return (SimpleNamespace(status="assembled" if not fire else "refused",
                                reason="plane_integer_residual", detail="",
                                result_sha256="5" * 64),
                SimpleNamespace(attribution_sha256="6" * 64, stage="solver",
                                post_admission_candidates=3,
                                post_same_axis_candidates=2,
                                proposal_count=4))

    return _SyntheticApi((entry, lattice, signatures, graph))


def _run(module, tmp_path, capture, *, api=None, expected_pins=None,
         governor=None, root=None, paths=None, limits=None, supervisor=None):
    paths = _synthetic_pngs(tmp_path) if paths is None else paths
    pins = _session_pins(paths)
    if expected_pins is None:
        expected_pins = tuple(pin.byte_sha256 for pin in pins)
    if api is None:
        api = _instrumented_api(capture, paths, calls=[])
    if governor is None:
        governor = module.ResourceGovernor(
            limits=limits or module.ResourceLimits(),
            process_tree_probe=lambda: 1024)
    if supervisor is None:
        supervisor = _FakeSupervisor(module)
    return module.execute_template_match_progress(
        image_paths=[str(path) for path in paths],
        root=(root if root is not None else tmp_path / "run"),
        api=api, expected_pins=expected_pins, governor=governor, limits=limits,
        supervisor=supervisor)


# ---------------------------------------------------------------------------
# Pin identity
# ---------------------------------------------------------------------------

def test_t2_pin_profile_matches_the_accepted_report_lines():
    lines = REPORT_PATH.read_text(encoding="utf-8").splitlines()
    accepted = [line for line in lines[40:43]]
    assert len(accepted) == 3
    for (label, digest, size), line in zip(progress.T2_PIN_PROFILE, accepted):
        assert digest in line, label
        assert str(size) not in line  # the report table carries hashes only
        assert label.split("/")[-1] in line
    assert progress.T2_PIN_REFERENCE.endswith("2026-10-04.md:41-43")
    assert len(set(progress.T2_PIN_SHA256S)) == 3


def test_check_pin_profile_accepts_only_the_ordered_t2_pins():
    accepted = tuple(_pin(value) for value in progress.T2_PIN_SHA256S)
    assert progress.check_pin_profile(accepted) == progress.T2_PIN_SHA256S

    permuted = (accepted[1], accepted[0], accepted[2])
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.check_pin_profile(permuted)
    assert error.value.reason == "input_pin_mismatch"
    assert error.value.field == "pin.0"

    arbitrary = tuple(_pin(hashlib.sha256(b"arbitrary").hexdigest())
                      for _ in range(3))
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.check_pin_profile(arbitrary)
    assert error.value.reason == "input_pin_mismatch"

    with pytest.raises(progress.TemplateMatchProgressError):
        progress.check_pin_profile(accepted[:2])
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.check_pin_profile((_pin("not-a-digest"),) + accepted[1:])
    assert error.value.field == "pin.0"


# ---------------------------------------------------------------------------
# Terminals
# ---------------------------------------------------------------------------

def test_terminal_states_are_exactly_the_five_accepted():
    assert set(progress.TERMINAL_STATES) == {
        "captured_solver_abstention", "assembled_no_support_capture",
        "early_refusal_no_capture", "profile_drift_no_capture",
        "resource_aborted_no_capture"}


@pytest.mark.parametrize("reason,expected", [
    ("row_count_mismatch", progress.TERMINAL_PROFILE_DRIFT_NO_CAPTURE),
    ("universe_count_mismatch", progress.TERMINAL_PROFILE_DRIFT_NO_CAPTURE),
    ("universe_identity_mismatch", progress.TERMINAL_PROFILE_DRIFT_NO_CAPTURE),
    ("resource_aborted", progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE),
    ("invalid_input", progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE),
    ("input_pin_mismatch", progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE),
    # The legacy capture refusal is an early refusal unless the same-pass graph
    # checkpoint says the solver assembled.
    ("capture_unavailable", progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE),
])
def test_classify_refusal_maps_each_terminal(reason, expected):
    assert progress.classify_refusal(reason) == expected


def test_classify_outcome_requires_the_same_pass_assembled_graph_checkpoint():
    assembled = {"stage": "graph", "status": "assembled"}
    assert progress.classify_outcome(
        "capture_unavailable", "capture",
        graph_checkpoint=assembled) == (
            progress.TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE)
    # No checkpoint, a non-graph checkpoint, or a non-assembled status never
    # fabricates the terminal.
    assert progress.classify_outcome(
        "capture_unavailable", "capture") == (
            progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    assert progress.classify_outcome(
        "capture_unavailable", "capture",
        graph_checkpoint={"stage": "graph", "status": "refused"}) == (
            progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    assert progress.classify_outcome(
        "capture_unavailable", "capture",
        graph_checkpoint={"stage": "entry", "status": "assembled"}) == (
            progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

def test_candidate_row_universe_cross_tabs_present_refused_and_unknown():
    rows = (_row(0, disposition="kept", support="present", values=[]),
            _row(1, disposition="dropped", support="refused"),
            _row(2, disposition="dropped", support="missing"))
    universe = progress.candidate_row_universe(rows)
    assert universe["admitted_total"] == 3
    assert universe["kept"] == 1 and universe["dropped"] == 2
    assert universe["support_present"] == 1
    assert universe["support_refused"] == 1
    assert universe["support_unknown"] == 1
    assert universe["denominators"] == {"all_admitted": 3,
                                        "conditional_support_present": 1}
    assert universe["cross_tab"]["kept"][progress.PRESENT] == 1
    assert universe["cross_tab"]["dropped"][progress.UNKNOWN] == 1


def test_logical_cell_pair_universe_counts_declared_unique_and_unknown():
    rows = (_row(0, values=[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]),
            _row(1, values=[[-1, 0, 0, 0], [2, 2, 2, 2]]),
            _row(2, support="present", values=None),
            _row(3, support="refused"))
    universe = progress.logical_cell_pair_universe(rows)
    assert universe["state"] == "known"
    assert universe["declared_pairs"] == 5
    assert universe["unique_pairs"] == 3
    assert universe["unknown_entries"] == 1
    assert universe["rows_with_pairs"] == 2
    assert universe["rows_unknown_payload"] == 1
    assert universe["rows_without_payload"] == 1
    assert universe["identity"] == "candidate_ordinal_and_four_coordinates"


def test_logical_pair_identity_includes_the_candidate_ordinal():
    # The same four coordinates on two different candidates are two pairs.
    rows = (_row(0, values=[[7, 8, 9, 10]]),
            _row(1, values=[[7, 8, 9, 10]]))
    universe = progress.logical_cell_pair_universe(rows)
    assert universe["declared_pairs"] == 2
    assert universe["unique_pairs"] == 2
    # The same coordinates repeated on one candidate still collapse to one.
    rows = (_row(0, values=[[7, 8, 9, 10], [7, 8, 9, 10]]),)
    universe = progress.logical_cell_pair_universe(rows)
    assert universe["declared_pairs"] == 2
    assert universe["unique_pairs"] == 1


def test_dependency_universe_projects_real_decoded_witnesses(tmp_path):
    paths = _synthetic_pngs(tmp_path)
    _, bindings = _synthetic_bindings(paths)
    section, ingress = _section_and_ingress(bindings)
    universe = progress.dependency_universe(section, ingress_by_frame=ingress)
    assert universe["state"] == progress.UNKNOWN
    assert universe["partial_entries"] > 0
    assert universe["projected_entries"] == 2
    cells = universe["cell_dependencies"]
    assert cells["frames_total"] == 2
    assert cells["declared"] > 0
    assert cells["union_total"] > 0
    assert cells["unknown_sentinel"] > 0
    # Every decoded entry is in the denominators, so coverage can never claim
    # completeness from bound entries alone.
    assert universe["denominators"]["all_entries"] == 2
    assert universe["denominators"]["entries_with_payload"] == 2
    assert universe["entries_total"] == 2
    # The projected union is exactly the deduplicated decoded indices.
    for binding in bindings:
        decoded = binding.witness.cell_source_indices.tolist()
        expected = {value for value in decoded if value != -1}
        assert cells["per_frame"][binding.frame_id] == len(expected)


def test_dependency_universe_dedups_overlapping_footprints_on_one_frame(
        tmp_path):
    paths = _synthetic_pngs(tmp_path)
    pins, bindings = _synthetic_bindings(paths)
    first = bindings[0]
    second = dataclasses.replace(first, canonical_index=first.canonical_index + 1,
                                 alternative_index=0, signature_sha256=_HEX_B,
                                 witness_sha256="")
    assert second.witness.witness_sha256 == first.witness.witness_sha256
    section = persistence.persist_signature_witnesses((first, second))
    ingress = {first.ingress.frame_id: first.ingress}
    universe = progress.dependency_universe(section, ingress_by_frame=ingress)
    assert universe["state"] == progress.UNKNOWN
    assert universe["partial_entries"] > 0
    cells = universe["cell_dependencies"]
    decoded = {value for value in first.witness.cell_source_indices.tolist()
               if value != -1}
    # Two overlapping witnesses on the same frame: the union is one footprint,
    # strictly smaller than the doubled declared sum.
    assert cells["frames_total"] == 1
    assert cells["union_total"] == len(decoded)
    assert cells["declared"] == 2 * first.witness.cell_source_indices.size
    assert cells["union_total"] < cells["declared"]


def test_dependency_universe_counts_border_sentinel_as_unknown_not_dependency(
        tmp_path):
    paths = _synthetic_pngs(tmp_path)
    pins, bindings = _synthetic_bindings(paths)
    first = bindings[0]
    raw = first.witness.cell_source_indices.copy()
    positions = np.nonzero(raw != -1)[0]
    assert positions.size
    raw[positions[0]] = -1
    raw.flags.writeable = False
    witness = dataclasses.replace(first.witness, cell_source_indices=raw,
                                  witness_sha256="")
    tampered = dataclasses.replace(first, witness=witness, witness_sha256="")
    section = persistence.persist_signature_witnesses((tampered,))
    universe = progress.dependency_universe(
        section, ingress_by_frame={first.ingress.frame_id: first.ingress})
    cells = universe["cell_dependencies"]
    assert cells["unknown_sentinel"] >= 1
    expected = {value for value in raw.tolist() if value != -1}
    assert cells["union_total"] == len(expected)
    assert -1 not in expected
    # A sentinel-bearing witness is a *partial* entry and stays in the
    # denominators rather than being dropped.
    assert universe["partial_entries"] >= 1
    assert universe["denominators"]["partial_entries"] >= 1


def test_dependency_universe_combines_channels_but_keeps_them_separate(
        tmp_path):
    paths = _synthetic_pngs(tmp_path)
    _, bindings = _synthetic_bindings(paths)
    section, ingress = _section_and_ingress(bindings)
    universe = progress.dependency_universe(section, ingress_by_frame=ingress)
    cell = universe["channel_totals_separate"]["cell"]
    tap = universe["channel_totals_separate"]["tap"]
    assert cell == universe["cell_dependencies"]["union_total"]
    assert tap == universe["tap_dependencies"]["union_total"]
    assert universe["union_total"] == universe["union_pairs"]
    # The actual union of (frame, pixel) pairs is never larger than the sum of
    # the separate channel totals.
    assert universe["union_pairs"] <= universe["union_total"]


def test_dependency_universe_unknown_for_absent_or_refused_section():
    assert progress.dependency_universe(None) == {
        "state": progress.UNKNOWN, "reason": "sampling_section_unavailable"}
    refused = persistence.refused_section_record(
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, field="record",
        detail="over")
    universe = progress.dependency_universe(refused, ingress_by_frame={})
    assert universe["state"] == progress.UNKNOWN


def test_dependency_universe_uses_the_real_decoder_not_a_stub(tmp_path,
                                                              monkeypatch):
    paths = _synthetic_pngs(tmp_path)
    _, bindings = _synthetic_bindings(paths)
    section, ingress = _section_and_ingress(bindings)
    seen = []
    real = persistence.decode_section

    def spy(*args, **kwargs):
        seen.append(kwargs.get("require_ingress"))
        return real(*args, **kwargs)

    monkeypatch.setattr(persistence, "decode_section", spy)
    progress.dependency_universe(section, ingress_by_frame=ingress)
    assert seen == [True]


def test_persisted_witness_arrays_are_immutable_after_decode(tmp_path):
    paths = _synthetic_pngs(tmp_path)
    _, bindings = _synthetic_bindings(paths)
    section, ingress = _section_and_ingress(bindings)
    decoded = persistence.decode_section(section, ingress_by_frame=ingress)
    entry = decoded.entries[0]
    with pytest.raises(ValueError):
        entry.witness.cell_source_indices[0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.witness.subsample_factor = 2


def test_support_grid_availability_reports_denominators_and_sentinel_unknown():
    rows = (_row(0, values=[[0, 0, 0, 0]],
                 warp=[[-1, 3], [4, -1]]),
            _row(1, support="refused"),
            _row(2, support="missing"))
    availability = progress.support_grid_availability(rows)
    assert availability["metric"] == "support_grid_availability"
    assert availability["rows_total"] == 3
    assert availability["present"] == 1
    assert availability["refused"] == 1
    assert availability["unknown"] == 1
    assert availability["cells_available"] == 2
    assert availability["cells_unknown"] == 2
    assert availability["denominators"] == {"all_admitted": 3,
                                            "conditional_support_present": 1}
    assert availability["per_row"]["2"] == progress.UNKNOWN


def test_source_screenshot_footprint_is_null_until_the_witness_csr_join():
    footprint = progress.source_screenshot_footprint()
    assert footprint["state"] == "unavailable"
    assert footprint["reason"] == progress.SOURCE_FOOTPRINT_REASON
    assert footprint["available"] is False
    assert footprint["join_present"] is False
    for value in footprint["metrics"].values():
        assert value is None
    # Even a claimed join does not fabricate metrics in this task.
    claimed = progress.source_screenshot_footprint(
        endpoint_witness_csr_join={"synthetic": True})
    assert claimed["state"] == "unavailable"
    assert claimed["metrics"]["source_pixels"] is None


def test_scorecard_downstream_metrics_are_null_for_abstention():
    scorecard = progress.build_progress_scorecard(
        capture_record={"index_pages": [{"rows": []}]},
        terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
    assert scorecard["complete_support_capture"] is True
    assert scorecard["downstream"]["state"] == "unavailable"
    for key in progress.DOWNSTREAM_METRIC_KEYS:
        assert scorecard["downstream"][key] is None
    assert scorecard["unique_dependencies"]["state"] == progress.UNKNOWN
    assert "support_grid_availability" in scorecard
    assert "footprint_availability" not in scorecard
    assert scorecard["source_screenshot_footprint"]["state"] == "unavailable"


def test_scorecard_digest_is_stable_and_malformed_input_is_refused():
    scorecard = progress.build_progress_scorecard(
        capture_record={"index_pages": [{"rows": []}]})
    assert scorecard["scorecard_sha256"] == progress.scorecard_digest(scorecard)
    with pytest.raises(progress.TemplateMatchProgressError):
        progress.build_progress_scorecard(
            capture_record={"index_pages": "not-a-list"})
    with pytest.raises(progress.TemplateMatchProgressError):
        progress.build_progress_scorecard(
            capture_record={"index_pages": [{"rows": []}]},
            terminal="not-a-terminal")


# ---------------------------------------------------------------------------
# Resource limits and budgets
# ---------------------------------------------------------------------------

def test_resource_limits_declared_budget_fits_and_sidecar_allowance_is_unused():
    limits = progress.ResourceLimits()
    assert limits.record() == {
        "wall_seconds": 900.0,
        "process_tree_bytes": 8 * 1024 * 1024 * 1024,
        "total_output_bytes": 128 * 1024 * 1024,
        "record_bytes": 32 * 1024 * 1024,
        "project_bytes": 4 * 1024 * 1024,
        "diagnostics_bytes": 2 * 1024 * 1024,
        "stream_bytes": 4 * 1024 * 1024,
    }
    assert limits.declared_budget_bytes() <= limits.total_output_bytes
    sidecar = 512 * 1024 * 1024
    assert limits.process_tree_bytes != sidecar
    assert limits.total_output_bytes != sidecar
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "512 * 1024" not in source


def test_establish_governor_refuses_when_the_memory_probe_is_unavailable():
    module = _harness()

    def unavailable():
        raise OSError("no probe")

    with pytest.raises(module.DiagnosticRefusal) as error:
        module.establish_governor(process_tree_probe=unavailable)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "process_tree_memory"


def test_governor_aborts_on_wall_and_memory():
    module = _harness()
    clock = {"now": 0.0}
    limits = module.ResourceLimits(wall_seconds=10.0, process_tree_bytes=100)
    governor = module.ResourceGovernor(
        limits=limits, clock=lambda: clock["now"],
        process_tree_probe=lambda: 50)
    governor.start()
    assert governor.check()["process_tree_bytes"] == 50
    clock["now"] = 11.0
    with pytest.raises(module.ResourceAbort) as error:
        governor.check()
    assert error.value.reason == "resource_wall_exceeded"

    clock["now"] = 0.0
    heavy = module.ResourceGovernor(
        limits=limits, clock=lambda: clock["now"],
        process_tree_probe=lambda: 101)
    heavy.start()
    with pytest.raises(module.ResourceAbort) as error:
        heavy.check()
    assert error.value.reason == "resource_memory_exceeded"


def test_output_and_stream_budgets_refuse_over_cap():
    module = _harness()
    limits = module.ResourceLimits(total_output_bytes=10, stream_bytes=4)
    budget = module.OutputBudget(limits)
    budget.charge(6, field="a")
    with pytest.raises(module.DiagnosticRefusal) as error:
        budget.charge(5, field="b")
    assert error.value.reason == "output_over_cap"

    class _Stream:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

    stream = _Stream()
    stream_budget = module.StreamBudget(4)
    stream_budget.write(stream, "abc")
    with pytest.raises(module.DiagnosticRefusal) as error:
        stream_budget.write(stream, "de")
    assert error.value.reason == "stream_over_cap"


def test_output_budget_accounts_temporary_and_replacement_bytes():
    module = _harness()
    limits = module.ResourceLimits(total_output_bytes=10)
    budget = module.OutputBudget(limits)
    budget.charge(6, field="record.json")
    # Replacing with 6 bytes peaks at old (6) + temporary (6) = 12 > 10.
    with pytest.raises(module.DiagnosticRefusal) as error:
        budget.replace(6, field="record.json")
    assert error.value.reason == "output_over_cap"
    budget.replace(2, field="record.json")
    record = budget.record()
    assert record["temporary_bytes"] == 2
    assert record["replacement_bytes"] == 2


def test_diagnostics_journal_refuses_over_cap(tmp_path):
    module = _harness()
    journal = module.DiagnosticsJournal(
        tmp_path / "diagnostics.json", max_bytes=120,
        terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    with pytest.raises(module.DiagnosticRefusal) as error:
        journal.append({"stage": "entry", "payload": "x" * 200})
    assert error.value.reason == "diagnostics_over_cap"
    assert not (tmp_path / "diagnostics.json").exists()


def test_diagnostics_journal_never_clobbers_a_foreign_target(tmp_path):
    module = _harness()
    target = tmp_path / "diagnostics.json"
    target.write_bytes(b"foreign bytes")
    journal = module.DiagnosticsJournal(
        target, max_bytes=1024,
        terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    with pytest.raises(module.DiagnosticRefusal) as error:
        journal.append({"stage": "entry"})
    assert error.value.reason == "journal_not_owned"
    assert target.read_bytes() == b"foreign bytes"


# ---------------------------------------------------------------------------
# Freeze (terminal-exact) and Phase B gates
# ---------------------------------------------------------------------------

def _refusal_root(tmp_path):
    diagnostics = {
        "schema": _harness().DIAGNOSTICS_SCHEMA,
        "terminal": progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE,
        "checkpoints": [],
    }
    (tmp_path / "diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8")
    (tmp_path / "refusal.json").write_text(json.dumps({
        "schema": _runner().REFUSAL_SCHEMA, "status": "refused", "reason": "entry_refused", "field": "entry"}), encoding="utf-8")
    return tmp_path


def test_freeze_round_trip_and_tamper_refusal(tmp_path):
    module = _harness()
    _refusal_root(tmp_path)
    freeze = module.freeze_root(
        tmp_path, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    assert freeze["schema"] == progress.FREEZE_SCHEMA
    assert freeze["artifact_set"] == "terminal_exact"
    assert module.revalidate_freeze(freeze, root=tmp_path)["artifact_count"] == 2
    (tmp_path / "refusal.json").write_bytes(b"changed")
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        module.revalidate_freeze(freeze, root=tmp_path)
    assert error.value.reason == "phase_a_not_frozen"


def test_freeze_rejects_missing_extra_foreign_and_over_cap(tmp_path):
    module = _harness()
    _refusal_root(tmp_path)
    module.freeze_root(tmp_path,
                       terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    # A missing terminal artifact refuses.
    (tmp_path / "refusal.json").unlink()
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.freeze_root(tmp_path,
                           terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    assert error.value.reason == "freeze_artifact_missing"
    # A foreign/extra artifact refuses.
    (tmp_path / "refusal.json").write_bytes(b"{}")
    (tmp_path / "foreign.bin").write_bytes(b"x")
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.freeze_root(tmp_path,
                           terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    assert error.value.reason == "freeze_artifact_foreign"


def test_freeze_record_rejects_foreign_names_duplicates_and_paths():
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.artifact_freeze_record(
            [("record.json", "record.json", "a" * 64, 1),
             ("foreign.bin", "foreign.bin", "b" * 64, 1)],
            terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
    assert error.value.reason == "freeze_artifact_foreign"
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.artifact_freeze_record(
            [("record.json", "record.json", "a" * 64, 1),
             ("record.json", "sub/record.json", "b" * 64, 1),
             ("route_b.mcscan", "route_b.mcscan", "c" * 64, 1),
             ("diagnostics.json", "diagnostics.json", "d" * 64, 1),
             ("scorecard.json", "scorecard.json", "e" * 64, 1)],
            terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
    assert error.value.reason == "freeze_artifact_duplicate"
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.artifact_freeze_record(
            [("record.json", "../record.json", "a" * 64, 1),
             ("route_b.mcscan", "route_b.mcscan", "c" * 64, 1),
             ("diagnostics.json", "diagnostics.json", "d" * 64, 1),
             ("scorecard.json", "scorecard.json", "e" * 64, 1)],
            terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
    assert error.value.reason == "freeze_path_invalid"
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.artifact_freeze_record(
            [("record.json", "record.json", "a" * 64, 999 * 1024 * 1024),
             ("route_b.mcscan", "route_b.mcscan", "c" * 64, 1),
             ("diagnostics.json", "diagnostics.json", "d" * 64, 1),
             ("scorecard.json", "scorecard.json", "e" * 64, 1)],
            terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
    assert error.value.reason == "freeze_artifact_over_cap"


def test_assert_phase_a_frozen_refuses_missing_and_malformed(tmp_path):
    module = _harness()
    _refusal_root(tmp_path)
    freeze = module.freeze_root(
        tmp_path, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    (tmp_path / "refusal.json").unlink()
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        module.revalidate_freeze(freeze, root=tmp_path)
    assert error.value.reason == "phase_a_not_frozen"
    with pytest.raises(progress.TemplateMatchProgressError):
        module.revalidate_freeze({"schema": "wrong"})
    tampered = dict(freeze)
    tampered["freeze_sha256"] = "0" * 64
    with pytest.raises(progress.TemplateMatchProgressError):
        module.revalidate_freeze(tampered, root=tmp_path)


def _phase_b_evidence():
    return {
        field: {"binding": "source_bound", "source": "ordinary_image_pass",
                "evidence_sha256": hashlib.sha256(field.encode()).hexdigest(),
                "detail": {"measurement": field}}
        for field in progress.PHASE_B_EVIDENCE_FIELDS}


def _phase_b_adapter_join():
    return {
        "freeze_revalidated": True,
        "same_pass": True,
        "fields": _phase_b_evidence(),
    }


def test_phase_b_gate_is_categorically_unavailable():
    # No adapter join at all.
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.phase_b_gate(_phase_b_evidence())
    assert error.value.reason == progress.PHASE_B_ALIGNMENT_UNAVAILABLE
    # Even a structurally complete adapter-join claim is refused: no real
    # adapter exists in this task, so a valid freeze alone never authorizes.
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.phase_b_gate(_phase_b_evidence(),
                              adapter_join=_phase_b_adapter_join())
    assert error.value.reason == progress.PHASE_B_ALIGNMENT_UNAVAILABLE
    assert "real_adapter" in error.value.detail
    # The contract is explicit and names every required join component.
    contract = progress.phase_b_join_contract()
    assert set(progress.PHASE_B_JOIN_REQUIREMENTS) <= set(
        contract["requirements"])


def test_phase_b_gate_refuses_manual_metadata():
    evidence = _phase_b_evidence()
    evidence["manual_metadata"] = {"f3": [1, 2, 3]}
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.phase_b_gate(evidence)
    assert error.value.reason == "phase_b_manual_metadata_rejected"

    foreign = _phase_b_evidence()
    foreign["pose"] = dict(foreign["pose"], source="camera_manifest")
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        progress.phase_b_gate(foreign)
    assert error.value.reason == "phase_b_manual_metadata_rejected"


# ---------------------------------------------------------------------------
# Containment (synthetic fake supervisor only)
# ---------------------------------------------------------------------------

def test_containment_is_established_before_entry_and_assigned_before_start(
        tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    supervisor = _FakeSupervisor(module)
    report = _run(module, tmp_path, capture, api=api, paths=paths,
                  supervisor=supervisor)
    assert supervisor.ensure_calls == 1
    assert supervisor.run_calls == 1
    assert supervisor.assigned_before_worker is True
    assert supervisor.worker_ran is True
    assert calls == ["entry", "lattice", "signatures", "graph"]
    assert report["supervision"]["job_assigned_before_start"] is True


def test_harness_refuses_before_entry_when_containment_is_unavailable(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    supervisor = _FakeSupervisor(module, available=False)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, api=api, paths=paths,
             supervisor=supervisor)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "containment"
    assert calls == []  # entry never ran
    assert supervisor.run_calls == 0
    root = tmp_path / "run"
    assert not (root / "record.json").exists()
    assert (root / "refusal.json").is_file()


def test_default_supervisor_refuses_before_entry_with_zero_stage_calls(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_template_match_progress(
            image_paths=[str(path) for path in paths],
            root=tmp_path / "run", api=api)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "containment"
    # The default supervisor is disabled, so no stage call ever runs.
    assert calls == []
    root = tmp_path / "run"
    assert not (root / "record.json").exists()
    assert not (root / "route_b.mcscan").exists()
    assert not (root / "route_b_staging").exists()
    # The typed pre-entry refusal is still persisted as evidence.
    assert (root / "refusal.json").is_file()


def test_disabled_windows_job_supervisor_refuses_before_entry(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    supervisor = module.WindowsJobSupervisor()
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_template_match_progress(
            image_paths=[str(path) for path in paths],
            root=tmp_path / "run", api=api, supervisor=supervisor)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "containment"
    assert calls == []
    # The OS supervisor is disabled: even its capability probe refuses.
    with pytest.raises(module.ContainmentUnavailable):
        supervisor.ensure_available()
    assert module.WindowsJobSupervisor.disabled is True
    assert not (tmp_path / "run" / "record.json").exists()


def test_arbitrary_injected_supervisor_with_api_none_refuses_before_pass(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"

    class _ArbitrarySupervisor:
        """A caller-shaped supervisor that is *not* the sanctioned test double."""

        kind = "arbitrary_injected"
        spawns_child = False
        can_contain = True

        def __init__(self):
            self.ensure_calls = 0
            self.run_calls = 0

        def ensure_available(self):
            self.ensure_calls += 1

        def run(self, spec):  # pragma: no cover - must never be reached
            self.run_calls += 1
            raise AssertionError("an arbitrary supervisor must never run the pass")

    supervisor = _ArbitrarySupervisor()
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_template_match_progress(
            image_paths=[str(path) for path in paths], root=root, api=None,
            supervisor=supervisor)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "containment"
    # The arbitrary supervisor never launched the pass, and no record/project was
    # produced (only the typed pre-entry refusal evidence is persisted).
    assert supervisor.ensure_calls == 1
    assert supervisor.run_calls == 0
    assert not (root / "record.json").exists()
    assert not (root / "route_b.mcscan").exists()


def test_unmarked_synthetic_api_refuses_before_entry(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    marked = _instrumented_api(capture, paths, calls=calls)
    # Strip the test-seam marker: the runner would still accept this plain tuple,
    # but the harness must refuse it before any stage runs.
    unmarked_api = tuple(marked)
    assert not getattr(unmarked_api, "synthetic_test_only", False)
    supervisor = _FakeSupervisor(module)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_template_match_progress(
            image_paths=[str(path) for path in paths], root=tmp_path / "run",
            api=unmarked_api, supervisor=supervisor)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "api"
    assert calls == []  # entry/lattice/signatures/graph never ran
    assert not (tmp_path / "run" / "record.json").exists()


def test_execute_pass_refuses_api_none_before_the_route_runner(tmp_path,
                                                              monkeypatch):
    module = _harness()
    reached: list = []

    def trap(**kwargs):  # pragma: no cover - must never be reached
        reached.append("runner")
        raise AssertionError("the route runner must not be entered without an api")

    monkeypatch.setattr(module.route_b, "execute_route_b_diagnostic", trap)
    with pytest.raises(module.DiagnosticRefusal) as error:
        module._execute_pass(
            root=tmp_path / "run", resolved=[], api=None,
            limits=module.ResourceLimits(), governor=None, expected_pins=(),
            clock=time.monotonic, process_tree_probe=None)
    assert error.value.reason == "resource_enforcement_unavailable"
    assert error.value.field == "api"
    assert reached == []
    # The refusal happens before any journal/artifact write.
    assert not (tmp_path / "run").exists()


def test_worker_cli_refuses_before_execute_pass(tmp_path, capsys):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"
    argv = ["--worker", "--root", str(root)]
    for path in paths:
        argv.extend(["--image", str(path)])
    calls: list = []
    original = module._execute_pass

    def trap(**kwargs):
        calls.append("pass")
        raise AssertionError("the disabled worker entry must not run a pass")

    module._execute_pass = trap
    try:
        code = module.main(argv)
    finally:
        module._execute_pass = original
    assert code == 1
    assert calls == []
    payload = json.loads(capsys.readouterr().err)
    assert payload["reason"] == "resource_enforcement_unavailable"
    assert payload["field"] == "containment"
    # No unauthenticated worker path: the root is not even created.
    assert not root.exists()


def test_default_cli_refuses_before_entry_and_writes_no_record(tmp_path, capsys):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"
    argv = ["--root", str(root)]
    for path in paths:
        argv.extend(["--image", str(path)])
    code = module.main(argv)
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["reason"] == "resource_enforcement_unavailable"
    assert payload["field"] == "containment"
    assert not (root / "record.json").exists()
    assert not (root / "route_b.mcscan").exists()
    assert not (root / "route_b_staging").exists()


def test_default_harness_root_collision_writes_nothing(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    (root / "record.json").write_bytes(b"foreign pre-existing bytes")
    before = sorted(os.listdir(root))
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.execute_template_match_progress(
            image_paths=[str(path) for path in paths], root=root)
    assert error.value.reason == "resource_enforcement_unavailable"
    # Zero writes: the pre-existing root is entry-for-entry unchanged and no
    # refusal, diagnostics or staging artifact is written into a foreign root.
    assert sorted(os.listdir(root)) == before
    assert (root / "record.json").read_bytes() == b"foreign pre-existing bytes"
    assert not (root / "diagnostics.json").exists()
    assert not (root / "refusal.json").exists()
    assert not (root / "route_b_staging").exists()


def test_harness_refuses_when_pre_start_assignment_is_not_demonstrated(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    supervisor = _FakeSupervisor(module, omit_assignment=True)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, paths=paths, supervisor=supervisor)
    assert error.value.reason == "resource_enforcement_unavailable"


def test_supervisor_deadline_abort_yields_resource_terminal_not_a_record(
        tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    supervisor = _FakeSupervisor(module, timed_out=True, peak_job_bytes=99)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, paths=paths, supervisor=supervisor)
    assert error.value.reason == "resource_wall_exceeded"
    root = tmp_path / "run"
    diagnostics = json.loads((root / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == (
        progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE)
    assert diagnostics["supervisor_abort"]["timed_out"] is True
    assert not (root / "record.json").exists()
    assert module.load_freeze(root)["status"] == "frozen"


def test_supervisor_memory_and_stream_aborts_are_typed(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    for kwargs, reason in (
            ({"memory_exceeded": True}, "resource_memory_exceeded"),
            ({"stream_exceeded": True}, "stream_over_cap")):
        supervisor = _FakeSupervisor(module, **kwargs)
        with pytest.raises(module.DiagnosticRefusal) as error:
            _run(module, tmp_path, capture, paths=paths,
                 root=tmp_path / reason, supervisor=supervisor)
        assert error.value.reason == reason


# ---------------------------------------------------------------------------
# Harness end-to-end
# ---------------------------------------------------------------------------

def test_harness_runs_one_instrumented_pass_and_freezes_artifacts(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls, snapshot=_snapshot())
    report = _run(module, tmp_path, capture, api=api, paths=paths)

    assert report["status"] == "complete"
    assert report["terminal"] == progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION
    assert calls == ["entry", "lattice", "signatures", "graph"]
    assert report["row_count"] == EXPECTED_ROW_COUNT
    assert len(report["solver_diagnostics"]) == 1
    assert report["solver_diagnostics"][0]["reason"] == "plane_integer_residual"
    assert report["solver_diagnostics"][0]["axis_resultant_keys"] == [0, 1, 2]
    assert report["solver_diagnostics"][0]["axis_resultants"] == [
        [0, 0.5], [1, 0.6], [2, 0.7]]
    assert report["solver_diagnostics"][0]["axis_residuals"] == [[0, 0.1]]

    root = tmp_path / "run"
    for name in ("record.json", "route_b.mcscan", "diagnostics.json",
                 "scorecard.json", "freeze.json"):
        assert (root / name).is_file(), name
    diagnostics = json.loads((root / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION
    stages = [record["stage"] for record in diagnostics["checkpoints"]]
    assert stages == ["entry", "pins", "lattice", "signatures", "graph",
                      "capture", "record"]
    assert diagnostics["scorecard"]["candidate_rows"]["admitted_total"] == (
        EXPECTED_ROW_COUNT)
    assert diagnostics["scorecard"]["candidate_rows"]["support_present"] == 2
    for key in progress.DOWNSTREAM_METRIC_KEYS:
        assert diagnostics["scorecard"]["downstream"][key] is None
    lattice_cp = [c for c in diagnostics["checkpoints"]
                  if c["stage"] == "lattice"][0]
    assert lattice_cp["reading_count"] == 2
    signature_cp = [c for c in diagnostics["checkpoints"]
                    if c["stage"] == "signatures"][0]
    assert signature_cp["signature_count"] == 2
    graph_cp = [c for c in diagnostics["checkpoints"]
                if c["stage"] == "graph"][0]
    assert graph_cp["proposal_count"] == 4
    assert graph_cp["solver"][0]["axis_resultants"] == [
        [0, 0.5], [1, 0.6], [2, 0.7]]
    assert report["resource"]["checks"] >= 7
    assert module.load_freeze(root)["status"] == "frozen"


def test_harness_observer_is_output_neutral(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls_a: list = []
    calls_b: list = []
    _run(module, tmp_path, capture, paths=paths, root=tmp_path / "run-a",
         api=_instrumented_api(capture, paths, calls=calls_a,
                               snapshot=_snapshot()))
    _run(module, tmp_path, capture, paths=paths, root=tmp_path / "run-b",
         api=_instrumented_api(capture, paths, calls=calls_b))
    first = (tmp_path / "run-a" / "record.json").read_bytes()
    second = (tmp_path / "run-b" / "record.json").read_bytes()
    assert first == second


def test_harness_early_entry_refusal_keeps_checkpoint_evidence(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    calls: list = []
    pins = _session_pins(paths)
    session = _session_for(paths, pins)

    def entry(input_paths, *, project_path=None, capture_ingress=False, project_max_bytes=None, retain_project_lease=False):
        calls.append("entry")
        lease = _write_synthetic_project(project_path, b"{}", project_max_bytes, retain_project_lease)
        result = _entry_terminal(session, graph_status="refused",
                               reason="entry_refused", detail="no entry")
        result.project_lease = lease
        return result

    def unreachable(stage):
        def call(**kwargs):
            calls.append(stage)
            raise AssertionError("must not run")
        return call

    api = _SyntheticApi((entry, unreachable("lattice"), unreachable("signatures"),
                         unreachable("graph")))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, None, api=api, paths=paths)
    assert error.value.reason == "entry_refused"
    assert calls == ["entry"]
    root = tmp_path / "run"
    assert not (root / "record.json").exists()
    diagnostics = json.loads((root / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE
    assert [record["stage"] for record in diagnostics["checkpoints"]] == ["entry"]
    refusal = json.loads((root / "refusal.json").read_text(encoding="utf-8"))
    assert refusal["reason"] == "entry_refused"


def test_harness_assembled_terminal_is_not_an_early_refusal(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls, fire=False)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, api=api, paths=paths)
    # The legacy runner emits its own legacy reason; only the wrapper classifies
    # the assembled terminal from the same-pass graph checkpoint.
    assert error.value.reason == "capture_unavailable"
    assert calls == ["entry", "lattice", "signatures", "graph"]
    diagnostics = json.loads((tmp_path / "run" / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == (
        progress.TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE)
    graph_cp = [c for c in diagnostics["checkpoints"]
                if c["stage"] == "graph"][0]
    assert graph_cp["status"] == "assembled"
    assert not (tmp_path / "run" / "record.json").exists()


def test_harness_profile_drift_terminal_for_wrong_universe(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    short = _synthetic_capture(3, 1)
    calls: list = []
    api = _instrumented_api(short, paths, calls=calls)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, short, api=api, paths=paths)
    assert error.value.reason == "universe_count_mismatch"
    diagnostics = json.loads((tmp_path / "run" / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == progress.TERMINAL_PROFILE_DRIFT_NO_CAPTURE
    assert calls.count("graph") == 1
    assert not (tmp_path / "run" / "record.json").exists()


def test_harness_resource_abort_terminal_and_no_record(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    governor = _AbortingGovernor(module)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, api=api, paths=paths,
             governor=governor)
    assert error.value.reason == "resource_wall_exceeded"
    assert calls == ["entry"]
    root = tmp_path / "run"
    diagnostics = json.loads((root / "diagnostics.json").read_text(
        encoding="utf-8"))
    assert diagnostics["terminal"] == (
        progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE)
    assert not (root / "record.json").exists()
    assert not (root / "route_b.mcscan").exists()
    # The watchdog abort still produces the abort terminal's exact artifact set
    # (diagnostics + typed refusal) and a revalidated freeze.
    assert (root / "refusal.json").is_file()
    assert module.load_freeze(root)["status"] == "frozen"


def test_harness_pin_gate_mismatch_refuses_before_lattice(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    wrong = tuple(hashlib.sha256(b"wrong").hexdigest() for _ in range(3))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, api=api, paths=paths,
             expected_pins=wrong)
    assert error.value.reason == "input_pin_mismatch"
    assert error.value.field == "pin.0"
    assert calls == ["entry"]
    root = tmp_path / "run"
    refusal = json.loads((root / "refusal.json").read_text(encoding="utf-8"))
    assert refusal["reason"] == "input_pin_mismatch"
    assert not (root / "record.json").exists()


def test_harness_project_cap_refuses_and_rolls_back(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls: list = []
    api = _instrumented_api(capture, paths, calls=calls)
    limits = module.ResourceLimits(project_bytes=4)
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, api=api, paths=paths, limits=limits)
    assert error.value.reason == "entry_persist_failed"
    assert calls == ["entry"]
    root = tmp_path / "run"
    assert not (root / "record.json").exists()
    assert not (root / "route_b.mcscan").exists()
    assert not (root / "route_b_staging").exists()


def test_harness_root_collision_writes_nothing_including_diagnostics(tmp_path):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    root = tmp_path / "run"
    root.mkdir()
    (root / "record.json").write_bytes(b"foreign pre-existing bytes")
    before = sorted(os.listdir(root))
    with pytest.raises(module.DiagnosticRefusal) as error:
        _run(module, tmp_path, capture, paths=paths, root=root)
    assert error.value.reason == "output_exists"
    # Zero writes: the pre-existing root is byte-for-byte and entry-for-entry
    # unchanged, with no diagnostics, refusal, staging or partial artifact.
    assert sorted(os.listdir(root)) == before
    assert (root / "record.json").read_bytes() == b"foreign pre-existing bytes"
    assert not (root / "diagnostics.json").exists()
    assert not (root / "refusal.json").exists()
    assert not (root / "route_b_staging").exists()


def test_load_freeze_refuses_before_a_run_and_after_artifact_change(tmp_path):
    module = _harness()
    with pytest.raises(module.DiagnosticRefusal) as error:
        module.load_freeze(tmp_path / "missing")
    assert error.value.reason == "phase_a_not_frozen"

    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    _run(module, tmp_path, capture, paths=paths, root=tmp_path / "run")
    root = tmp_path / "run"
    assert module.load_freeze(root)["status"] == "frozen"
    record = root / "record.json"
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(progress.TemplateMatchProgressError) as error:
        module.load_freeze(root)
    assert error.value.reason == "phase_a_not_frozen"


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
def test_native_journal_checkpoint_update_blocks_equal_byte_replacement(tmp_path, monkeypatch):
    from mc_photogrammetry.project.store import _WindowsRetainedFile
    from test_route_b_diagnostic_runner import _native_barrier_pair, _assert_native_replacement_blocked
    module = _harness()
    target, proposed = tmp_path / "diagnostics.json", tmp_path / "foreign.json"
    ownership = module.route_b.InvocationOwnership(native=True)
    journal = module.DiagnosticsJournal(target, max_bytes=4096, ownership=ownership, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    original = _WindowsRetainedFile.rename_no_replace
    try:
        journal.append({"stage": "entry"})
        previous_bytes = ownership.read_bytes(target, 4096)
        proposed.write_bytes(previous_bytes)
        identity = ownership._handles[target].identity
        def operation(barrier):
            def rename(held, destination):
                if held.path == target:
                    assert held._identity() == identity
                    barrier()
                return original(held, destination)
            monkeypatch.setattr(_WindowsRetainedFile, "rename_no_replace", rename)
            journal.append({"stage": "lattice"})
        _native_barrier_pair(operation, lambda: _assert_native_replacement_blocked(
            target, proposed, tmp_path / "saved.json"))
        assert len(json.loads(ownership.read_bytes(target, 4096))["checkpoints"]) == 2
    finally:
        ownership.close()
    assert proposed.read_bytes() == previous_bytes
    assert not target.with_name(target.name + ".partial").exists()
    assert not target.with_name(target.name + ".previous").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
def test_native_journal_gap_refuses_and_preserves_foreign_and_saved_checkpoint(tmp_path, monkeypatch):
    from mc_photogrammetry.project.store import _WindowsRetainedFile
    from test_route_b_diagnostic_runner import _native_barrier_pair
    module = _harness()
    target = tmp_path / "diagnostics.json"
    ownership = module.route_b.InvocationOwnership(native=True)
    journal = module.DiagnosticsJournal(target, max_bytes=4096, ownership=ownership, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    original = _WindowsRetainedFile.rename_no_replace
    try:
        journal.append({"stage": "entry"})
        original_bytes = ownership.read_bytes(target, 4096)
        def operation(barrier):
            def rename(held, destination):
                if held.path.name == "diagnostics.json.partial":
                    barrier()  # old checkpoint already moved to its retained backup
                return original(held, destination)
            monkeypatch.setattr(_WindowsRetainedFile, "rename_no_replace", rename)
            with pytest.raises(module.DiagnosticRefusal) as error:
                journal.append({"stage": "lattice"})
            assert error.value.reason == "output_exists"
        _native_barrier_pair(operation, lambda: target.write_bytes(b"foreign"))
    finally:
        ownership.close()
    assert target.read_bytes() == b"foreign"
    assert target.with_name(target.name + ".previous").read_bytes() == original_bytes
    assert not target.with_name(target.name + ".partial").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows file identity required")
@pytest.mark.parametrize("insert_extra", [False, True])
def test_native_freeze_retains_file_and_root_identity_but_namespace_remains_hold(
        tmp_path, monkeypatch, insert_extra):
    from test_route_b_diagnostic_runner import _native_barrier_pair, _assert_native_replacement_blocked
    module = _harness()
    root = tmp_path / "root"
    root.mkdir()
    _refusal_root(root)
    freeze = module.freeze_root(root, terminal=progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE)
    proposed = tmp_path / "foreign.json"
    original_bytes = (root / "diagnostics.json").read_bytes()
    proposed.write_bytes(original_bytes)
    original = module._freeze_binding_ok
    seen = []
    def operation(barrier):
        def bind(record, base, *, snapshots=None):
            assert snapshots is not None
            assert snapshots["diagnostics.json"] == original_bytes
            barrier()  # all file/directory handles are held through semantic validation
            return original(record, base, snapshots=snapshots)
        monkeypatch.setattr(module, "_freeze_binding_ok", bind)
        if insert_extra:
            with pytest.raises(progress.TemplateMatchProgressError):
                module.revalidate_freeze(freeze, root=root)
        else:
            assert module.revalidate_freeze(freeze, root=root)["artifact_count"] == 2
    def race():
        _assert_native_replacement_blocked(root / "diagnostics.json", proposed,
                                         tmp_path / "saved.json")
        with pytest.raises(OSError) as error:
            root.rename(tmp_path / "saved-root")
        assert error.value.winerror in (5, 32)
        seen.append("root rename denied")
        if insert_extra:
            (root / "foreign.bin").write_bytes(b"foreign")
    _native_barrier_pair(operation, race)
    assert seen == ["root rename denied"]
    assert proposed.read_bytes() == original_bytes
    assert (root / "diagnostics.json").read_bytes() == original_bytes
    assert module.RetainedFreezeSnapshot.namespace_exclusive is False
    if insert_extra:
        assert (root / "foreign.bin").read_bytes() == b"foreign"


@pytest.mark.skipif(os.name != "nt", reason="native Windows lease required")
@pytest.mark.parametrize("failure", ["missing", "released", "wrong_path", "wrong_cap"])
def test_native_capture_refuses_absent_or_invalid_birth_lease_before_lattice(tmp_path, failure):
    module = _harness()
    paths = _synthetic_pngs(tmp_path)
    capture = _capture_for(paths)
    calls = []
    api = _instrumented_api(capture, paths, calls=calls)
    leases = []
    def entry(paths, **kwargs):
        result = api[0](paths, **kwargs)
        lease = result.project_lease
        leases.append(lease)
        if failure == "missing":
            lease.close()
            result.project_lease = None
        elif failure == "released":
            lease.close()
        elif failure == "wrong_path":
            lease.path = lease.path.with_name("wrong.mcscan")
        else:
            lease.max_bytes += 1
        return result
    wrapped = _SyntheticApi((entry, *api[1:]))
    try:
        with pytest.raises(module.DiagnosticRefusal) as error:
            _run(module, tmp_path, capture, paths=paths, api=wrapped)
        assert error.value.reason in ("entry_project_lease_unavailable", "entry_project_lease_invalid")
        assert calls == ["entry"]
        assert not (tmp_path / "run" / "record.json").exists()
    finally:
        # Explicitly release corrupted fixture leases; never reopen/adopt them.
        for lease in leases:
            lease.close()
