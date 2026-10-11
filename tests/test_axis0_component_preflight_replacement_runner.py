import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / ".tmp-run-axis0-component-preflight-replacement.py"
SPEC = importlib.util.spec_from_file_location("axis0_replacement_runner", RUNNER)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


class TerminationKernel:
    def __init__(self, waits=(subject.WAIT_TIMEOUT, subject.WAIT_OBJECT_0),
                 job_result=False, process_result=True):
        self.waits = list(waits); self.job_result = job_result
        self.process_result = process_result; self.calls = []

    def WaitForSingleObject(self, process, timeout):
        self.calls.append(("wait", timeout))
        return self.waits.pop(0) if self.waits else subject.WAIT_OBJECT_0

    def TerminateJobObject(self, job, code):
        self.calls.append(("job", code)); return self.job_result

    def TerminateProcess(self, process, code):
        self.calls.append(("process", code)); return self.process_result


def test_assignment_failure_cleanup_explicitly_kills_and_reaps_child():
    kernel = TerminationKernel()
    result = subject._terminate_and_reap(kernel, 7, 8, 125)
    assert result == {"job_terminated": False, "process_terminated": True,
                      "process_reaped": True}
    assert ("process", 125) in kernel.calls
    assert all(timeout != subject.INFINITE for kind, timeout in kernel.calls if kind == "wait")


def test_timeout_termination_failure_is_bounded_and_reported():
    kernel = TerminationKernel(
        waits=(subject.WAIT_TIMEOUT, subject.WAIT_TIMEOUT, subject.WAIT_TIMEOUT),
        job_result=False, process_result=False)
    result = subject._terminate_and_reap(kernel, 7, 8, 124)
    assert result == {"job_terminated": False, "process_terminated": False,
                      "process_reaped": False}
    assert [row for row in kernel.calls if row[0] == "wait"][-2:] == [
        ("wait", 5000), ("wait", 5000)]


class JobKernel:
    def __init__(self, counts): self.counts = list(counts)
    def QueryInformationJobObject(self, job, kind, buffer, size, returned):
        words = ctypes.cast(buffer, ctypes.POINTER(subject.wintypes.DWORD))
        words[0] = self.counts[0]; words[1] = self.counts.pop(0)
        return True


class CleanupKernel(TerminationKernel):
    def __init__(self, counts, **kwargs):
        super().__init__(**kwargs); self.counts = list(counts)
    def QueryInformationJobObject(self, job, kind, buffer, size, returned):
        words = ctypes.cast(buffer, ctypes.POINTER(subject.wintypes.DWORD))
        words[0] = self.counts[0]; words[1] = self.counts.pop(0)
        return True


def test_descendant_surviving_root_exit_blocks_success():
    assert subject._wait_job_empty(JobKernel([1]), 7, seconds=0.0) is False
    assert subject._wait_job_empty(JobKernel([1, 0]), 7, seconds=0.2) is True


def test_timeout_cleanup_verifies_whole_job_empty_before_publication():
    complete = subject._cleanup_job(CleanupKernel([1, 0]), 7, 8, 124)
    assert complete["job_empty"] is True
    incomplete = subject._cleanup_job(CleanupKernel([1] * 200), 7, 8, 124)
    assert incomplete["job_empty"] is False


def test_report_schema_and_digest_are_recomputed(tmp_path):
    report = {"schema": "mc-photogrammetry-axis0-component-synthetic-preflight-v1",
              "status": "unavailable"}
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["run_sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "report.json"; path.write_text(json.dumps(report))
    assert subject._validate_worker_report(path) == report
    report["status"] = "complete"; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="digest mismatch"):
        subject._validate_worker_report(path)
    report["schema"] = "foreign"; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="schema"):
        subject._validate_worker_report(path)


def test_project_artifact_is_replayed_and_malformed_input_refuses(tmp_path):
    accepted = ROOT / ".tmp-axis0-component-synthetic-preflight.mcscan"
    if accepted.is_file():
        subject._validate_project(accepted, ROOT)
    malformed = tmp_path / "bad.mcscan"; malformed.write_text("{}")
    with pytest.raises(Exception):
        subject._validate_project(malformed, ROOT)


def test_retained_png_set_requires_exact_frames_and_bytes(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"payload"
    for index in range(4):
        (tmp_path / f"image-{index:04d}.png").write_bytes(png + bytes([index]))
    rows = subject._validate_png_inputs(tmp_path)
    assert len(rows) == 4 and all(row["sha256"] for row in rows)
    (tmp_path / "image-0003.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="malformed"):
        subject._validate_png_inputs(tmp_path)


def test_manifest_precedes_native_launch_and_job_has_kill_on_close():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.index('_atomic_json(paths["manifest"]') < source.index("kernel.CreateProcessW(")
    manifest = subject.validate(ROOT, "static-control-attempt")
    assert manifest["job_limit_flags"] == 0x2300
    assert manifest["creation_flags"] & subject.CREATE_SUSPENDED
    assert manifest["input_ledger"]["witness_budget"] == [
        12_582_912, 50_331_648, 67_108_864, 1_256_194_072]
    assert len(manifest["input_ledger"]["cameras"]) == 4
    assert not any(Path(value).exists() for value in manifest["paths"].values())


def test_retention_seam_uses_private_tempfile_proxy_and_restores_module():
    paths = subject._paths(ROOT, "proxy-static-control")
    source = (ROOT / ".tmp-run-axis0-component-preflight.py").read_text(encoding="utf-8-sig")
    patched = subject._patched_worker(source, paths)
    assert "SimpleNamespace(TemporaryDirectory=_RetainedPngDirectory)" in patched
    assert "pre.tempfile=_original_pre_tempfile" in patched
    assert "pre.tempfile.TemporaryDirectory=_RetainedPngDirectory" not in patched


def test_frozen_ledger_rejects_file_and_camera_drift():
    worker = ROOT / ".tmp-run-axis0-component-preflight.py"
    ledger = subject._frozen_ledger(ROOT, worker, "a" * 64, ROOT / "future-worker.py")
    subject._verify_ledger(ledger, require_patched_worker=False)
    altered = json.loads(json.dumps(ledger)); altered["files"][0]["sha256"] = "0" * 64
    replay = dict(altered); replay.pop("ledger_sha256")
    altered["ledger_sha256"] = hashlib.sha256(json.dumps(replay, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(RuntimeError, match="input drift"):
        subject._verify_ledger(altered, require_patched_worker=False)


def test_prelaunch_failure_gets_atomic_typed_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "validate",
                        lambda root, attempt: (_ for _ in ()).throw(RuntimeError("setup")))
    assert subject.run(tmp_path, "prelaunch-control") == 1
    terminal = tmp_path / ".tmp-axis0-component-preflight-prelaunch-control.terminal.json"
    value = json.loads(terminal.read_text())
    assert value["status"] == "refused"
    assert value["reason"] == "prelaunch_setup_failed"
    assert not terminal.with_name(terminal.name + ".partial").exists()
