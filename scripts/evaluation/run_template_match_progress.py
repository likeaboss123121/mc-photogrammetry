"""Phase A controls implementation with reconstruction release disabled.

The default API, CLI and direct worker always refuse before reconstruction.
Only explicitly marked synthetic APIs enter the test seam. Native Windows
controls are tested separately by run_control, which accepts only fixed harmless
inline child modes and no caller command, image, output path or reconstruction
callable. Native Job limits constrain committed memory, not RSS.

The actual entry/store writer measures encoded project bytes before serialization
and disk creation, then enforces every write. Artifact/journal serializers prescan
size and reserve output before writing. Freeze reload anchors to the caller root,
rejects extra/partial entries and recomputes record-derived scorecards plus
project/session/ingress and checkpoint/refusal joins. Missing/sentinel source
dependencies remain UNKNOWN; Phase B and source-footprint metrics stay unavailable.

This is not complete release readiness. Native owned files use retained identity,
no-replace handle publication and handle disposition. Directory handles block
rename of held objects, but permit WRITE and child-name creation; inventories are
observed rather than namespace-sealed. Directory birth, reparse mutation and
empty-directory cleanup remain release gates. Legacy cleanup is outside this
native contract. Deterministic barriers prove only the stated held-object scope.
The harmless controls are not wired to a released reconstruction worker. The
private test marker is a convention, not an authentication boundary. No real
image reconstruction or evaluator is authorized by this implementation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_route_b_diagnostic as route_b  # noqa: E402
from mc_photogrammetry import evaluation_template_match_progress as progress  # noqa: E402

SCHEMA = "mc-photogrammetry-template-match-progress-harness-v2"
DIAGNOSTICS_SCHEMA = "mc-photogrammetry-template-match-progress-diagnostics-v1"
DIAGNOSTICS_NAME = "diagnostics.json"
FREEZE_NAME = "freeze.json"
SCORE_NAME = "scorecard.json"
# The supervised combined stdout/stderr capture is harness infrastructure, not a
# Phase A artifact: it lives under the fresh root but is excluded from the
# terminal-exact freeze set (its own byte cap is enforced separately).
STREAM_NAME = "stdout_stderr.log"
# The sealed freeze record's own serialized cap (it is not a Phase A artifact).
MAX_FREEZE_BYTES = 1024 * 1024
ResourceLimits = progress.ResourceLimits
DiagnosticRefusal = route_b.DiagnosticRefusal

# The interim fail-closed reason. The OS Job Object supervisor may not launch a
# contained run until every outstanding release gate is fixed and independently
# reviewed. The public CLI and the default ``execute_template_match_progress``
# (``supervisor is None``) refuse with this typed reason before ``api.entry``.
SUPERVISOR_DISABLED_REASON = (
    "reconstruction release is disabled pending independent review: native "
    "controls only run harmless children; concurrent filesystem replacement/"
    "exclusive namespace, directory birth, and production worker guarantees remain unresolved")

# The explicit marker every private-seam test double must carry. This is a
# **test-only convention enforced in code**: the harness refuses to run a pass
# unless both the injected supervisor and the injected API set
# ``synthetic_test_only is True``. It is not an authentication boundary (a caller
# could set the attribute), it exists so the public API, the CLI and every
# non-test call refuse before ``_execute_pass``.
SYNTHETIC_TEST_ONLY = "synthetic_test_only"

# Explicit re-enable blockers for the private synthetic seam. These are
# unresolved internal defects in the pass body reachable only through that seam.
# The default/CLI boundary is verified fail-closed, but the private seam is NOT
# accepted for real inputs until every one of these is fixed and independently
# reviewed. They are deliberately not silently fixed in this narrow correction.
PRIVATE_SEAM_REENABLE_BLOCKERS = (
    "Directory WRITE/reparse/child-name mutation and empty-directory cleanup are not an exclusive "
    "namespace contract; native handles protect held objects, not every possible name",
    "Root/staging creation before retained acquisition is not a demonstrated directory "
    "birth-identity contract; namespace inventory is observed, never sealed",
    "No-replace journal backup publication preserves a racing foreign name and the "
    "previous checkpoint, but is not a crash-atomic replace-style journal",
    "Freeze hashes and joins use retained immutable bytes on Windows, but full typed "
    "canonical project/session replay and populated-root abort finalization remain gates",
    "Fixed harmless native controls are not wired to the reconstruction worker",
    "Native Job allocation refusal/configuration is not a hard observed peak or RSS ceiling",
    "Legacy/non-Windows runner cleanup remains outside the native capture lease contract",
)


def _is_synthetic_test_only(candidate) -> bool:
    """True only when a candidate explicitly self-identifies as a test double."""
    return getattr(candidate, SYNTHETIC_TEST_ONLY, None) is True


class ResourceAbort(RuntimeError):
    """One bounded resource-limit abort (wall or process-tree memory)."""

    def __init__(self, reason: str, field: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.field = field
        self.detail = detail


# ---------------------------------------------------------------------------
# Hard containment: a supervised child assigned to an OS Job Object before it
# ever runs. The in-process checkpoint governor below is a bounded watchdog and
# telemetry channel; it is *not* the containment mechanism.
# ---------------------------------------------------------------------------

class ContainmentUnavailable(RuntimeError):
    """The OS-level containment properties could not be established."""


@dataclass(frozen=True)
class SupervisedSpec:
    """Everything the supervisor needs to run one contained child."""

    command: tuple
    cwd: str
    root: str
    stdout_path: str
    stderr_path: str
    stream_path: str
    wall_seconds: float
    memory_bytes: int
    stream_max_bytes: int
    worker: Callable[[], object]


@dataclass
class SupervisedOutcome:
    """One supervised child's externally observed result."""

    exit_code: int | None = None
    timed_out: bool = False
    memory_exceeded: bool = False
    stream_exceeded: bool = False
    job_assigned_before_start: bool = False
    job_terminated: bool = False
    peak_job_bytes: int = 0
    stream_bytes: int = 0
    root_entries: tuple = ()
    report: object | None = None
    error: BaseException | None = None

    @property
    def killed(self) -> bool:
        return bool(self.timed_out or self.memory_exceeded
                    or self.stream_exceeded)

    @property
    def breach_reason(self) -> str:
        if self.timed_out:
            return "resource_wall_exceeded"
        if self.memory_exceeded:
            return "resource_memory_exceeded"
        if self.stream_exceeded:
            return "stream_over_cap"
        return "resource_aborted"


class RunSupervisor:
    """Containment contract. The base class is deliberately unavailable.

    Only an explicitly injected synthetic test supervisor may pass
    ``ensure_available`` in the interim state; the default is refused before
    ``execute_template_match_progress`` ever resolves an api. ``run`` must return
    an outcome whose ``job_assigned_before_start`` is true only when the child was
    assigned to the OS containment object before it was allowed to execute.
    """

    kind = "unavailable"
    spawns_child = True
    disabled = True

    def ensure_available(self) -> None:
        raise ContainmentUnavailable(
            "no OS-level containment supervisor is available")

    def run(self, spec: SupervisedSpec) -> SupervisedOutcome:  # pragma: no cover
        raise ContainmentUnavailable(
            "no OS-level containment supervisor is available")

    def describe(self) -> dict:
        return {"kind": self.kind, "spawns_child": self.spawns_child,
                "disabled": getattr(self, "disabled", False)}


class WindowsJobSupervisor(RunSupervisor):
    """Reconstruction disabled; run_control tests native harmless children only.

    ensure_available/run refuse production specs. run_control is a separate,
    fixed-mode native test API with no caller command or reconstruction path.
    Native containment evidence does not authorize real capture or filesystem
    guarantees. The old uncapped file-polling reference launcher is removed.
    """

    kind = "windows_job_object"
    spawns_child = True
    disabled = True

    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
    _JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT = 9
    _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT = 10
    _CREATE_SUSPENDED = 0x00000004
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_NO_WINDOW = 0x08000000
    _STARTF_USESTDHANDLES = 0x00000100
    _STARTF_USESHOWWINDOW = 0x00000001
    _SW_HIDE = 0
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258
    _POLL_MS = 200

    def __init__(self, *, poll_ms: int = _POLL_MS) -> None:
        self._poll_ms = int(poll_ms)
        self._kernel32 = None

    # -- ctypes plumbing -----------------------------------------------------

    @staticmethod
    def _load_kernel32():
        from ctypes import wintypes

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation",
                 _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        return kernel32, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION

    def _create_job(self, kernel32, info_type, memory_bytes: int, name=None):
        job = kernel32.CreateJobObjectW(None, name)
        if not job:
            raise ContainmentUnavailable(
                f"CreateJobObjectW failed: {ctypes.get_last_error()}")
        info = info_type()
        info.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | self._JOB_OBJECT_LIMIT_JOB_MEMORY
            | self._JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        info.ProcessMemoryLimit = int(memory_bytes)
        info.JobMemoryLimit = int(memory_bytes)
        if not kernel32.SetInformationJobObject(
                job, self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            raise ContainmentUnavailable(
                f"SetInformationJobObject failed: {ctypes.get_last_error()}")
        return job

    def ensure_available(self) -> None:
        """Always refuse: the OS containment path is disabled (see class docs)."""
        raise ContainmentUnavailable(SUPERVISOR_DISABLED_REASON)

    def _peak_job_bytes(self, job) -> int:
        kernel32, info_type = self._kernel32
        info = info_type()
        returned = ctypes.c_ulong(0)
        if not kernel32.QueryInformationJobObject(
                job, self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
            raise ContainmentUnavailable("native job memory query failed")
        return int(info.PeakJobMemoryUsed)

    def run(self, spec: SupervisedSpec) -> SupervisedOutcome:  # pragma: no cover
        """Always refuse: no real child may be launched in the interim state."""
        raise ContainmentUnavailable(SUPERVISOR_DISABLED_REASON)

    def run_control(self, mode: str, *, wall_seconds=3.0,
                    memory_bytes=128 * 1024 * 1024,
                    stream_max_bytes=1024, fail_assignment=False,
                    membership_check="correct"):
        """Exercise actual Windows controls using fixed harmless inline children.

        This API accepts no command, callable, image, output path or worker mode.
        It cannot enable reconstruction. Captured pipe bytes stay bounded in RAM.
        Explicit child checks run under -O and cannot disappear as asserts.
        Wrong/missing membership checks only test refusal in fixed harmless modes.
        Job notifications distinguish aggregate from process allocation refusal;
        neither configured limits nor peak telemetry prove a hard RSS ceiling.
        """
        if mode not in {"ok", "burst", "burst_reverse", "timeout",
                        "memory", "memory_tree", "descendant"}:
            raise ContainmentUnavailable("invalid harmless control mode")
        if os.name != "nt":
            raise ContainmentUnavailable("native controls require Windows")
        if (not 0 < wall_seconds <= 10 or type(memory_bytes) is not int
                or not 32 * 1024 * 1024 <= memory_bytes <= 256 * 1024 * 1024
                or type(stream_max_bytes) is not int
                or not 1 <= stream_max_bytes <= 1024 * 1024):
            raise ContainmentUnavailable("invalid harmless control limits")
        if membership_check not in {"correct", "wrong", "missing"}:
            raise ContainmentUnavailable("invalid harmless membership check")
        if mode in {"burst", "burst_reverse"} and stream_max_bytes < 16:
            raise ContainmentUnavailable("combined burst needs a cap of at least 16")
        from ctypes import wintypes as w
        import uuid
        kernel, info_type = self._load_kernel32()
        self._kernel32 = (kernel, info_type)
        handle = w.HANDLE
        class Security(ctypes.Structure):
            _fields_ = [("length", w.DWORD), ("descriptor", ctypes.c_void_p),
                        ("inherit", w.BOOL)]
        class CompletionPortAssociation(ctypes.Structure):
            _fields_ = [("key", ctypes.c_void_p), ("port", handle)]
        class Startup(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR),
                        ("desktop", w.LPWSTR), ("title", w.LPWSTR),
                        ("x", w.DWORD), ("y", w.DWORD), ("xs", w.DWORD),
                        ("ys", w.DWORD), ("xc", w.DWORD), ("yc", w.DWORD),
                        ("fill", w.DWORD), ("flags", w.DWORD),
                        ("show", w.WORD), ("reserved2", w.WORD),
                        ("reservedptr", ctypes.c_void_p), ("stdin", handle),
                        ("stdout", handle), ("stderr", handle)]
        class StartupEx(ctypes.Structure):
            _fields_ = [("startup", Startup), ("attributes", ctypes.c_void_p)]
        class Process(ctypes.Structure):
            _fields_ = [("process", handle), ("thread", handle),
                        ("pid", w.DWORD), ("tid", w.DWORD)]
        signatures = {
            "CloseHandle": ([handle], w.BOOL),
            "CreateIoCompletionPort": ([handle, handle, ctypes.c_size_t, w.DWORD], handle),
            "GetQueuedCompletionStatus": ([handle, ctypes.POINTER(w.DWORD),
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p),
                w.DWORD], w.BOOL),
            "CreatePipe": ([ctypes.POINTER(handle), ctypes.POINTER(handle),
                            ctypes.POINTER(Security), w.DWORD], w.BOOL),
            "SetHandleInformation": ([handle, w.DWORD, w.DWORD], w.BOOL),
            "InitializeProcThreadAttributeList": ([ctypes.c_void_p, w.DWORD,
                w.DWORD, ctypes.POINTER(ctypes.c_size_t)], w.BOOL),
            "UpdateProcThreadAttribute": ([ctypes.c_void_p, w.DWORD,
                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p], w.BOOL),
            "DeleteProcThreadAttributeList": ([ctypes.c_void_p], None),
            "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, ctypes.c_void_p,
                ctypes.c_void_p, w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR,
                ctypes.c_void_p, ctypes.POINTER(Process)], w.BOOL),
            "ResumeThread": ([handle], w.DWORD),
            "TerminateProcess": ([handle, w.UINT], w.BOOL),
            "WaitForSingleObject": ([handle, w.DWORD], w.DWORD),
            "GetExitCodeProcess": ([handle, ctypes.POINTER(w.DWORD)], w.BOOL),
            "PeekNamedPipe": ([handle, ctypes.c_void_p, w.DWORD,
                ctypes.c_void_p, ctypes.POINTER(w.DWORD), ctypes.c_void_p], w.BOOL),
            "ReadFile": ([handle, ctypes.c_void_p, w.DWORD,
                ctypes.POINTER(w.DWORD), ctypes.c_void_p], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = args, result
        verifier = r'''
import ctypes, os, subprocess, sys, time
from ctypes import wintypes as w
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenJobObjectW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];k.OpenJobObjectW.restype=w.HANDLE
k.GetCurrentProcess.restype=w.HANDLE
k.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)];k.IsProcessInJob.restype=w.BOOL
k.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p];k.QueryInformationJobObject.restype=w.BOOL
k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
j=k.OpenJobObjectW(4,False,sys.argv[1]);member=w.BOOL()
if not j: sys.exit(21)
if not k.IsProcessInJob(k.GetCurrentProcess(),j,ctypes.byref(member)) or not member.value:
    k.CloseHandle(j);sys.exit(22)
class Basic(ctypes.Structure):
    _fields_=[("t1",ctypes.c_longlong),("t2",ctypes.c_longlong),("flags",w.DWORD),("min",ctypes.c_size_t),("max",ctypes.c_size_t),("active",w.DWORD),("aff",ctypes.c_size_t),("priority",w.DWORD),("sched",w.DWORD)]
class Info(ctypes.Structure):
    _fields_=[("basic",Basic),("io",ctypes.c_ulonglong*6),("process",ctypes.c_size_t),("job",ctypes.c_size_t),("peakprocess",ctypes.c_size_t),("peakjob",ctypes.c_size_t)]
i=Info()
if not k.QueryInformationJobObject(j,9,ctypes.byref(i),ctypes.sizeof(i),None):
    k.CloseHandle(j);sys.exit(23)
if i.process!=int(sys.argv[2]) or i.job!=int(sys.argv[2]) or i.basic.flags & 0x2300 != 0x2300:
    k.CloseHandle(j);sys.exit(24)
k.CloseHandle(j);os.write(1,b"JOB_OK\n")
class MemoryCounters(ctypes.Structure):
    _fields_=[("cb",w.DWORD),("faults",w.DWORD)]+[(name,ctypes.c_size_t) for name in ("peak_ws","ws","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile","private")]
k.K32GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD];k.K32GetProcessMemoryInfo.restype=w.BOOL
def private_bytes():
    counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
    if not k.K32GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(counters),counters.cb):sys.exit(25)
    return counters.private
'''
        child = verifier + '\nVERIFIER=' + repr(verifier) + r'''
mode=sys.argv[3]
if mode=="ok":os.write(1,("OPTIMIZE="+str(sys.flags.optimize)+"\n").encode())
elif mode in ("burst","burst_reverse"):
    cap=int(sys.argv[4]);out_total=(cap+1)//2;err_total=cap+1-out_total
    out_payload=out_total-len(b"JOB_OK\n")
    out_first=out_payload//2;err_first=err_total//2
    if mode=="burst":
        os.write(1,b"x"*out_first);os.write(2,b"y"*err_first)
        os.write(1,b"x"*(out_payload-out_first));os.write(2,b"y"*(err_total-err_first))
    else:
        os.write(2,b"y"*err_first);os.write(1,b"x"*out_first)
        os.write(2,b"y"*(err_total-err_first));os.write(1,b"x"*(out_payload-out_first))
elif mode=="timeout":time.sleep(30)
elif mode=="memory":
    try:data=bytearray(int(sys.argv[2])*2)
    except MemoryError:os.write(1,b"MEMORY_REFUSED\n")
    else:sys.exit(9)
elif mode=="memory_tree":
    limit=int(sys.argv[2]);requested=limit*3//5
    data=bytearray(requested)
    root_commit=private_bytes()
    if root_commit>=limit:sys.exit(26)
    os.write(1,("TREE_ROOT_COMMIT="+str(root_commit)+"\nTREE_REQUEST="+str(requested)+"\n").encode())
    leaf=VERIFIER+r"""
requested=int(sys.argv[2])*3//5
projected=private_bytes()+requested
os.write(1,("TREE_LEAF_PROJECTED="+str(projected)+"\n").encode())
if projected>=int(sys.argv[2]):sys.exit(27)
try:data=bytearray(requested)
except MemoryError:os.write(1,b"TREE_MEMORY_REFUSED\n")
else:os.write(1,b"TREE_ALLOCATION_SUCCEEDED\n");sys.exit(9)
"""
    descendant=subprocess.Popen([sys.executable,"-B","-O","-c",leaf,sys.argv[1],sys.argv[2]],stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,close_fds=True,creationflags=0x08000000)
    sys.exit(descendant.wait(timeout=2))
elif mode=="descendant":
    descendant=subprocess.Popen([sys.executable,"-B","-O","-c","import time; time.sleep(30)"],stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,close_fds=True,creationflags=0x08000000)
    os.write(1,("DESCENDANT="+str(descendant.pid)+"\n").encode())
'''
        job = verifier_job = completion_port = read = write = process = thread = None
        attributes = None
        attributes_initialized = False
        retained = bytearray()
        result = SupervisedOutcome()
        job_name = "Local\\mc-photo-controls-" + uuid.uuid4().hex
        def drain():
            available = w.DWORD()
            if not kernel.PeekNamedPipe(read, None, 0, None,
                                        ctypes.byref(available), None):
                if ctypes.get_last_error() in (109, 232):
                    return True
                raise ContainmentUnavailable("pipe inspection failed")
            if available.value:
                # One crossing byte detects overflow; it is never retained.
                count = min(65536, available.value,
                            max(1, stream_max_bytes - len(retained) + 1))
                buffer = ctypes.create_string_buffer(count)
                received = w.DWORD()
                if not kernel.ReadFile(read, buffer, count,
                                        ctypes.byref(received), None):
                    raise ContainmentUnavailable("pipe read failed")
                result.stream_bytes += received.value
                if len(retained) + received.value > stream_max_bytes:
                    result.stream_exceeded = True
                else:
                    retained.extend(buffer.raw[:received.value])
            return False
        try:
            job = self._create_job(kernel, info_type, memory_bytes, job_name)
            verification_name = job_name
            if membership_check == "wrong":
                verification_name = job_name + "-wrong"
                verifier_job = self._create_job(kernel, info_type, memory_bytes,
                                                verification_name)
            elif membership_check == "missing":
                verification_name = job_name + "-missing"
            if mode == "memory_tree":
                completion_port = kernel.CreateIoCompletionPort(
                    handle(-1), None, 0, 1)
                if not completion_port:
                    raise ContainmentUnavailable("memory notification port creation failed")
                association = CompletionPortAssociation(1, completion_port)
                if not kernel.SetInformationJobObject(job,
                        self._JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION,
                        ctypes.byref(association), ctypes.sizeof(association)):
                    raise ContainmentUnavailable("job memory notification association failed")
            rh, wh = handle(), handle()
            security = Security(ctypes.sizeof(Security), None, True)
            if not kernel.CreatePipe(ctypes.byref(rh), ctypes.byref(wh),
                                     ctypes.byref(security), 4096):
                raise ContainmentUnavailable("CreatePipe failed")
            read, write = rh.value, wh.value
            if not kernel.SetHandleInformation(read, 1, 0):
                raise ContainmentUnavailable("pipe inheritance setup failed")
            size = ctypes.c_size_t()
            kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            attributes = ctypes.create_string_buffer(size.value)
            if not kernel.InitializeProcThreadAttributeList(attributes, 1, 0,
                                                            ctypes.byref(size)):
                raise ContainmentUnavailable("handle allowlist initialization failed")
            attributes_initialized = True
            allowed = (handle * 1)(write)
            if not kernel.UpdateProcThreadAttribute(attributes, 0, 0x20002,
                    ctypes.byref(allowed), ctypes.sizeof(allowed), None, None):
                raise ContainmentUnavailable("handle allowlist failed")
            startup = StartupEx()
            startup.startup.cb = ctypes.sizeof(startup)
            startup.startup.flags = 0x100
            startup.startup.stdout = startup.startup.stderr = write
            startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline([
                sys.executable, "-B", "-O", "-c", child, verification_name, str(memory_bytes),
                mode, str(stream_max_bytes)]))
            pi = Process()
            if not kernel.CreateProcessW(None, command, None, None, True,
                    0x80000 | self._CREATE_SUSPENDED | self._CREATE_NO_WINDOW,
                    None, str(REPO_ROOT), ctypes.byref(startup), ctypes.byref(pi)):
                raise ContainmentUnavailable(
                    f"control CreateProcessW failed: {ctypes.get_last_error()}")
            process, thread = pi.process, pi.thread
            if fail_assignment or not kernel.AssignProcessToJobObject(job, process):
                kernel.TerminateProcess(process, 1)
                if kernel.WaitForSingleObject(process, 3000) != 0:
                    raise ContainmentUnavailable("unassigned child did not terminate")
                result.exit_code = 1
                return result
            result.job_assigned_before_start = True
            if kernel.ResumeThread(thread) == 0xffffffff:
                raise ContainmentUnavailable("control resume failed")
            kernel.CloseHandle(write)
            write = None
            deadline = time.monotonic() + wall_seconds
            while True:
                drain()
                if result.stream_exceeded:
                    break
                if kernel.WaitForSingleObject(process, 10) == 0:
                    break
                if time.monotonic() >= deadline:
                    result.timed_out = True
                    break
            result.peak_job_bytes = self._peak_job_bytes(job)
            # Terminate the full tree even on primary-child success, before EOF.
            result.job_terminated = bool(kernel.TerminateJobObject(job, 1))
            if not result.job_terminated or kernel.WaitForSingleObject(process, 3000) != 0:
                raise ContainmentUnavailable("control job teardown failed")
            drain_deadline = time.monotonic() + 3
            while not result.stream_exceeded and not drain():
                if time.monotonic() > drain_deadline:
                    raise ContainmentUnavailable("control pipe did not close")
                time.sleep(.005)
            code = w.DWORD()
            if not kernel.GetExitCodeProcess(process, ctypes.byref(code)):
                raise ContainmentUnavailable("control exit query failed")
            result.exit_code = int(code.value)
            notifications = []
            if completion_port is not None:
                for _ in range(64):
                    message, key, pointer = w.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
                    if not kernel.GetQueuedCompletionStatus(completion_port,
                            ctypes.byref(message), ctypes.byref(key),
                            ctypes.byref(pointer), 0):
                        if ctypes.get_last_error() == 258:
                            break
                        raise ContainmentUnavailable("job memory notification query failed")
                    if key.value != 1:
                        raise ContainmentUnavailable("foreign job notification key")
                    notifications.append(int(message.value))
                else:
                    raise ContainmentUnavailable("job memory notification cap exceeded")
            result.memory_exceeded = (b"MEMORY_REFUSED\n" in retained
                                      or b"TREE_MEMORY_REFUSED\n" in retained)
            result.report = {"output": bytes(retained),
                             "specific_job_verified": b"JOB_OK\n" in retained,
                             "process_id": pi.pid,
                             "memory_metric": "committed_bytes",
                             "job_memory_limit_observed":
                                 self._JOB_OBJECT_MSG_JOB_MEMORY_LIMIT in notifications,
                             "process_memory_limit_observed":
                                 self._JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT in notifications,
                             "job_notifications": tuple(notifications)}
            return result
        finally:
            if job is not None:
                kernel.TerminateJobObject(job, 1)
            if attributes_initialized:
                kernel.DeleteProcThreadAttributeList(attributes)
            for owned_handle in (thread, process, write, read, completion_port,
                                 verifier_job, job):
                if owned_handle is not None:
                    kernel.CloseHandle(owned_handle)


def _root_entries(root) -> tuple:
    try:
        return tuple(sorted(os.listdir(root)))
    except OSError:
        return ()


# ---------------------------------------------------------------------------
# Process-tree memory probe (Windows). The supervised child is the run, so the
# probe below is the in-process watchdog's own-view telemetry; the hard memory
# cap is the job object set by the supervisor.
# ---------------------------------------------------------------------------

def _default_process_tree_probe() -> int:
    if os.name != "nt":
        raise OSError("process-tree memory probe requires Windows")
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                      counters.cb):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(counters.WorkingSetSize)


class ResourceGovernor:
    """Bounded watchdog/telemetry at stage checkpoints - *not* enforcement.

    The governor records elapsed time and working-set telemetry and raises a
    typed abort so a refusal is preserved, but checkpoint polling cannot preempt
    a long pure-Python call. Hard wall/memory containment is the supervisor's OS
    job object; the report states that distinction rather than claiming the
    governor enforces anything.
    """

    def __init__(self, *, limits: ResourceLimits | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 process_tree_probe: Callable[[], int] | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self._clock = clock
        self._probe = (process_tree_probe
                       if process_tree_probe is not None
                       else _default_process_tree_probe)
        self._started: float | None = None
        self.checks = 0
        self.peak_process_tree_bytes = 0

    def __enter__(self) -> "ResourceGovernor":
        self.start()
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def start(self) -> None:
        if self._started is None:
            self._started = self._clock()

    def elapsed(self) -> float:
        if self._started is None:
            return 0.0
        return max(0.0, self._clock() - self._started)

    def check(self) -> dict:
        self.checks += 1
        elapsed = self.elapsed()
        if elapsed > self.limits.wall_seconds:
            raise ResourceAbort(
                "resource_wall_exceeded", "wall_seconds",
                f"elapsed {elapsed:.3f}s exceeds {self.limits.wall_seconds}s")
        value = int(self._probe())
        self.peak_process_tree_bytes = max(self.peak_process_tree_bytes, value)
        if value > self.limits.process_tree_bytes:
            raise ResourceAbort(
                "resource_memory_exceeded", "process_tree_bytes",
                f"{value} bytes exceeds {self.limits.process_tree_bytes}")
        return {"elapsed_seconds": elapsed, "process_tree_bytes": value}

    def record(self) -> dict:
        return {
            "limits": self.limits.record(),
            "checks": self.checks,
            "elapsed_seconds": self.elapsed(),
            "peak_process_tree_bytes": self.peak_process_tree_bytes,
        }


def establish_governor(*, limits: ResourceLimits | None = None,
                       clock: Callable[[], float] = time.monotonic,
                       process_tree_probe: Callable[[], int] | None = None,
                       ) -> ResourceGovernor:
    """Build the bounded watchdog; refuse before launch if it cannot start."""
    limits = limits or ResourceLimits()
    if limits.declared_budget_bytes() > limits.total_output_bytes:
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "total_output_bytes",
            "the per-artifact caps exceed the declared total output budget")
    governor = ResourceGovernor(limits=limits, clock=clock,
                                process_tree_probe=process_tree_probe)
    try:
        governor.start()
        governor.check()
    except ResourceAbort:
        raise
    except Exception as error:  # probe/clock not establishable -> refuse
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "process_tree_memory",
            type(error).__name__) from error
    return governor


def establish_containment(supervisor: RunSupervisor | None) -> RunSupervisor:
    """Return an *injected, proven* supervisor or refuse before entry.

    The default (``supervisor is None``) is refused outright: the OS supervisor
    path is disabled in this interim fail-closed state, so the public API and CLI
    refuse ``resource_enforcement_unavailable`` before ``api.entry`` with zero
    entry/lattice/signature/graph calls. A caller-supplied supervisor must prove
    itself through ``ensure_available``; the disabled ``WindowsJobSupervisor``
    always fails that probe. Only an explicitly injected synthetic test
    supervisor can pass, and only with synthetic test APIs.
    """
    if supervisor is None:
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "containment",
            SUPERVISOR_DISABLED_REASON)
    candidate = supervisor
    try:
        candidate.ensure_available()
    except ContainmentUnavailable as error:
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "containment",
            str(error) or type(error).__name__) from error
    if not getattr(candidate, "can_contain", True):
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "containment",
            "the supervisor does not provide OS-level containment")
    return candidate


def require_synthetic_test_seam(supervisor, api) -> None:
    """Refuse unless BOTH the supervisor and the API are explicit test doubles.

    The public ``execute_template_match_progress`` and the CLI must never run a
    pass for an arbitrary injected supervisor or a real/default/unmarked API. The
    only reachable pass is the private synthetic seam, and it opens only when the
    supervisor *and* the API both set ``synthetic_test_only is True``. The marker
    is a code-enforced **test-only convention**, not an authentication boundary.
    """
    if not _is_synthetic_test_only(supervisor):
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "containment",
            "the injected supervisor is not marked synthetic_test_only=True; the "
            "public harness only runs under the explicit synthetic test seam")
    if not _is_synthetic_test_only(api):
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "api",
            "the injected api is not marked synthetic_test_only=True; a real, "
            "default or unmarked route-B api is never used by this harness")


# ---------------------------------------------------------------------------
# Bounded diagnostics journal and output/stream budgets
# ---------------------------------------------------------------------------

class DiagnosticsJournal:
    """Append-only bounded stage journal, exclusively published on each record.

    The journal writes a uniquely created ``.partial`` (``xb``) and then replaces
    its *own proven* target. Before any replace the target must either be absent
    (first flush) or byte-identical to the exact document this journal last
    wrote; a foreign or divergent file refuses ``journal_not_owned`` instead of
    being clobbered. Every flush reserves the exact declared size, and the
    output budget is charged for both the temporary bytes and the replacement.
    """

    def __init__(self, path: Path, *, max_bytes: int, terminal: str,
                 ownership=None, output=None) -> None:
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.terminal = terminal
        self.checkpoints: list = []
        self.extra: dict = {}
        self.written = 0
        self._last_data: bytes | None = None
        self._private_ownership = ownership is None
        self._ownership = ownership or route_b.InvocationOwnership(native=(os.name == "nt"))
        self._output = output

    def document(self) -> dict:
        return {
            "schema": DIAGNOSTICS_SCHEMA,
            "terminal": self.terminal,
            "checkpoints": self.checkpoints,
            **self.extra,
        }

    def append(self, record: dict) -> dict:
        self.checkpoints.append(record)
        return self._flush()

    def set_extra(self, **values) -> dict:
        self.extra.update(values)
        return self._flush()

    def set_terminal(self, terminal: str) -> dict:
        self.terminal = terminal
        return self._flush()

    def _flush(self) -> dict:
        size = route_b.persistence_canonical_size(self.document()) + 1
        if size > self.max_bytes:
            raise DiagnosticRefusal("diagnostics_over_cap", "diagnostics")
        data = json.dumps(self.document(), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
        if len(data) + 1 > self.max_bytes:
            raise DiagnosticRefusal(
                "diagnostics_over_cap", "diagnostics",
                f"{len(data) + 1} > {self.max_bytes} bytes")
        payload = data + b"\n"
        # Reserve the temporary + replacement bytes *before* touching disk.
        if self._output is not None:
            self._output.replace(len(payload), field=self.path.name)
        if self._ownership.native:
            return self._flush_native(payload)
        if os.path.lexists(self.path):
            if not self._proves_own_file():
                raise DiagnosticRefusal(
                    "journal_not_owned", "diagnostics",
                    "the diagnostics target is not this invocation's own file")
        temporary = self.path.with_name(self.path.name + ".partial")
        if os.path.lexists(temporary):
            raise DiagnosticRefusal("output_exists", "diagnostics")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Prove ownership once more immediately before the replace.
        if os.path.lexists(self.path) and not self._proves_own_file():
            route_b._unlink_if_present(temporary)
            raise DiagnosticRefusal("journal_not_owned", "diagnostics")
        os.replace(temporary, self.path)
        self._last_data = payload
        self.written = len(payload)
        if self._ownership is not None:
            self._ownership.register(self.path)
        return self.document()

    def _flush_native(self, payload):
        self._ownership.anchor(self.path.parent)
        resolved = Path(os.path.abspath(self.path))
        old = self._ownership._handles.get(resolved)
        if (old is None and os.path.lexists(self.path)) or (
                old is not None and (self._last_data is None or
                    old.read_bounded(self.max_bytes) != self._last_data)):
            raise DiagnosticRefusal("journal_not_owned", "diagnostics")
        temporary = self.path.with_name(self.path.name + ".partial")
        new = None
        previous = self.path.with_name(self.path.name + ".previous")
        old_moved = False
        try:
            new = route_b._write_partial(temporary, payload)
            if old is not None:
                old.rename_no_replace(previous)
                self._ownership.take_handle(self.path)
                old_moved = True
            new.rename_no_replace(self.path)
            self._ownership.register_handle(self.path, new)
            new = None
            if old is not None:
                old.delete_owned()
                old.close()
                old = None
        except OSError as error:
            reason = "output_exists" if isinstance(error, FileExistsError) or getattr(error, "winerror", None) in (80, 183) else "ownership_unavailable"
            raise DiagnosticRefusal(reason, "diagnostics", str(error)) from error
        finally:
            if new is not None:
                try:
                    new.delete_owned()
                finally:
                    new.close()
            # A racing target is preserved. The old checkpoint remains under the
            # no-replace backup name on failure; never path-replace it back.
            if old_moved and old is not None:
                old.close()
        self._last_data = payload
        self.written = len(payload)
        return self.document()

    def close(self):
        if self._private_ownership:
            self._ownership.close()

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass

    def _proves_own_file(self) -> bool:
        if self._last_data is None:
            return False
        try:
            current = read_bounded_bytes(self.path, self.max_bytes)
        except (OSError, DiagnosticRefusal):
            return False
        return current == self._last_data


class OutputBudget:
    """Cumulative on-disk output accounting against the 128 MiB total cap.

    ``charge`` registers a new artifact; ``replace`` accounts the peak of an
    in-place replacement (the old version plus the new temporary), so temporary
    and replacement bytes are never invisible. Only the current size of each
    field counts toward ``total``; ``peak_bytes`` records the worst transient.
    """

    def __init__(self, limits: ResourceLimits) -> None:
        self.limits = limits
        self._sizes: dict = {}
        self.temporary_bytes = 0
        self.replacement_bytes = 0
        self.peak_bytes = 0

    @property
    def total(self) -> int:
        return sum(self._sizes.values())

    def _check(self, value: int, *, field: str) -> None:
        if type(value) is not int or value < 0:
            raise DiagnosticRefusal("output_over_cap", field)
        if value > self.limits.total_output_bytes:
            raise DiagnosticRefusal(
                "output_over_cap", field,
                f"{value} > {self.limits.total_output_bytes} bytes")

    def charge(self, size: int, *, field: str) -> int:
        self._check(size, field=field)
        current = self._sizes.get(field, 0)
        if self.total - current + int(size) > self.limits.total_output_bytes:
            raise DiagnosticRefusal(
                "output_over_cap", field,
                f"{self.total - current + int(size)} > "
                f"{self.limits.total_output_bytes} bytes")
        self._sizes[field] = int(size)
        self.peak_bytes = max(self.peak_bytes, self.total)
        return self.total

    def replace(self, size: int, *, field: str) -> int:
        self._check(size, field=field)
        current = self._sizes.get(field, 0)
        peak = self.total + int(size)
        if peak > self.limits.total_output_bytes:
            raise DiagnosticRefusal(
                "output_over_cap", field,
                f"replacement peak {peak} > {self.limits.total_output_bytes}")
        self._sizes[field] = int(size)
        self.temporary_bytes += int(size)
        self.replacement_bytes += int(size)
        self.peak_bytes = max(self.peak_bytes, peak)
        return self.total

    def release(self, *, field: str) -> int:
        self._sizes.pop(field, None)
        return self.total

    def record(self) -> dict:
        return {
            "total_bytes": self.total,
            "peak_bytes": self.peak_bytes,
            "temporary_bytes": self.temporary_bytes,
            "replacement_bytes": self.replacement_bytes,
        }


class StreamBudget:
    """Bound one stdout/stderr stream to the declared stream cap."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self.written = 0

    def write(self, stream, data: str) -> int:
        payload = data.encode("utf-8")
        if self.written + len(payload) > self.max_bytes:
            raise DiagnosticRefusal(
                "stream_over_cap", "stdout_stderr",
                f"{self.written + len(payload)} > {self.max_bytes} bytes")
        stream.write(data)
        self.written += len(payload)
        return self.written


# ---------------------------------------------------------------------------
# Bounded stable artifact reads
# ---------------------------------------------------------------------------

def _is_reparse_point(path) -> bool:
    try:
        return bool(route_b._reparse_tag(Path(path)))
    except DiagnosticRefusal:
        return True


def read_bounded_bytes(path, max_bytes: int) -> bytes:
    """Bounded, stable, non-reparse read of one artifact, returning its bytes.

    The read is refused when the path is a reparse point, when the file grows
    past ``max_bytes`` while reading, or when its size/mtime identity changes
    during the read (a same-size in-place writer is outside the supported
    Windows contract).
    """
    path = Path(path)
    if type(max_bytes) is not int or max_bytes < 0:
        raise DiagnosticRefusal("artifact_over_cap", path.name)
    if os.name == "nt":
        from mc_photogrammetry.project.store import _WindowsRetainedFile, _WindowsRootAnchor, ProjectStoreError
        try:
            with _WindowsRootAnchor(path.parent):
                with _WindowsRetainedFile(path) as held:
                    return held.read_bounded(max_bytes)
        except ProjectStoreError as error:
            raise DiagnosticRefusal("artifact_over_cap", path.name, str(error)) from error
    route_b._refuse_reparse_components(path)
    if _is_reparse_point(path):
        raise DiagnosticRefusal("artifact_reparse", path.name)
    info = os.lstat(path)
    if info.st_size > max_bytes:
        raise DiagnosticRefusal(
            "artifact_over_cap", path.name,
            f"{info.st_size} > {max_bytes} bytes")
    chunks = []
    total = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise DiagnosticRefusal("artifact_unstable", path.name)
        while True:
            chunk = stream.read(min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise DiagnosticRefusal("artifact_over_cap", path.name)
            chunks.append(chunk)
        closed = os.fstat(stream.fileno())
    after = os.lstat(path)
    if (route_b._reparse_tag(path) or
            (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns,
             closed.st_ctime_ns) !=
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns) !=
            (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns,
             closed.st_ctime_ns)):
        raise DiagnosticRefusal("artifact_unstable", path.name)
    return b"".join(chunks)


def read_bounded_stable(path, max_bytes: int):
    """Bounded, stable, non-reparse read; returns ``(sha256_hex, size)``."""
    data = read_bounded_bytes(path, max_bytes)
    return hashlib.sha256(data).hexdigest(), len(data)


def _read_json_bounded(path) -> dict:
    cap = progress.FREEZE_ARTIFACT_CAPS.get(Path(path).name,
                                            MAX_FREEZE_BYTES)
    try:
        document = json.loads(read_bounded_bytes(path, cap).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, OSError, RecursionError) as error:
        raise DiagnosticRefusal("artifact_malformed", Path(path).name) from error
    if type(document) is not dict:
        raise DiagnosticRefusal("artifact_malformed", Path(path).name)
    return document


def _sha256_file(path) -> str:
    return read_bounded_stable(
        path, progress.FREEZE_ARTIFACT_CAPS.get(Path(path).name,
                                                32 * 1024 * 1024))[0]


# ---------------------------------------------------------------------------
# Artifact freeze / revalidation
# ---------------------------------------------------------------------------

def freeze_root(root, *, terminal, limits: ResourceLimits | None = None,
                ownership=None, output_budget=None) -> dict:
    """Write, return and *immediately revalidate* the Phase A freeze.

    The root must contain exactly the terminal's artifact set: a missing, extra,
    duplicate, foreign or reparse artifact refuses before the freeze is sealed.
    """
    root = Path(root)
    expected = tuple(progress.FREEZE_ARTIFACTS_BY_TERMINAL.get(terminal, ()))
    if not expected:
        raise DiagnosticRefusal("freeze_terminal_invalid", "terminal",
                                str(terminal))
    actual = []
    for name in sorted(os.listdir(root)):
        if name == FREEZE_NAME:
            continue
        actual.append(name)
    extras = [name for name in actual if name not in expected]
    if extras:
        raise DiagnosticRefusal("freeze_artifact_foreign", extras[0],
                                "unexpected artifact in the frozen root")
    for name in expected:
        path = root / name
        if not path.is_file():
            raise DiagnosticRefusal("freeze_artifact_missing", name)
        if _is_reparse_point(path):
            raise DiagnosticRefusal("freeze_artifact_reparse", name)
    caps = dict(progress.FREEZE_ARTIFACT_CAPS)
    if limits is not None:
        caps["record.json"] = limits.record_bytes
        caps["route_b.mcscan"] = limits.project_bytes
        caps["diagnostics.json"] = limits.diagnostics_bytes
    entries = []
    for name in expected:
        if ownership is not None and ownership.native and ownership.owns(root / name):
            data = ownership.read_bytes(root / name, caps.get(name, 32 * 1024 * 1024))
            sha256, size = hashlib.sha256(data).hexdigest(), len(data)
        else:
            sha256, size = read_bounded_stable(root / name,
                                               caps.get(name, 32 * 1024 * 1024))
        entries.append((name, name, sha256, size))
    record = progress.artifact_freeze_record(
        entries, root=str(root), terminal=terminal, limits=limits)
    route_b._atomic_publish(root / FREEZE_NAME, record, field="freeze",
                            max_bytes=MAX_FREEZE_BYTES, ownership=ownership,
                            output_budget=output_budget)
    # Immediate revalidation: the seal is never trusted without re-reading it.
    revalidate_freeze(record, root=root, ownership=ownership)
    return record


def _freeze_binding_unchecked(freeze, base, *, snapshots=None) -> dict:
    """Cross-validate record/scorecard/journal bindings against the freeze."""
    def read_json(path):
        if snapshots is None:
            return _read_json_bounded(path)
        document = json.loads(snapshots[Path(path).name].decode("utf-8"))
        if type(document) is not dict:
            raise DiagnosticRefusal("artifact_malformed", Path(path).name)
        return document
    def hash_file(path):
        return _sha256_file(path) if snapshots is None else hashlib.sha256(snapshots[Path(path).name]).hexdigest()
    base = Path(base)
    terminal = freeze.get("terminal")
    if terminal == progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION:
        try:
            document = route_b.validate_record(base / route_b.RECORD_NAME, snapshot_bytes=None if snapshots is None else snapshots[route_b.RECORD_NAME])
        except DiagnosticRefusal as error:
            return {"ok": False, "reason": error.reason}
        if document.get("schema") not in route_b.RECORD_SCHEMAS:
            return {"ok": False, "reason": "record_schema"}
        scorecard_path = base / SCORE_NAME
        try:
            scorecard = read_json(scorecard_path)
        except (DiagnosticRefusal, OSError, json.JSONDecodeError,
                UnicodeDecodeError):
            return {"ok": False, "reason": "scorecard_unreadable"}
        if scorecard.get("scorecard_sha256") != progress.scorecard_digest(
                scorecard):
            return {"ok": False, "reason": "scorecard_digest"}
        if scorecard.get("terminal") != terminal:
            return {"ok": False, "reason": "scorecard_terminal"}
        from mc_photogrammetry.pipeline.frame_ingress import FrameIngressProvenanceV1
        expected_scorecard = progress.build_progress_scorecard(
            capture_record=document["capture"],
            section=document.get("sampling_persistence"),
            ingress_by_frame={item["frame_id"]: FrameIngressProvenanceV1.from_record(item)
                              for item in document.get("ingress", ())}, terminal=terminal)
        if scorecard != expected_scorecard:
            return {"ok": False, "reason": "scorecard_record_mismatch"}
        project = read_json(base / route_b.PROJECT_NAME)
        session = project.get("image_derived_session")
        if (project.get("schema_version") != 2 or type(session) is not dict
                or session.get("session_sha256") !=
                document.get("stage_digests", {}).get("session_sha256")
                or [frame.get("ingress") for frame in session.get("frames", ())]
                != document.get("ingress")
                or project.get("source_paths") !=
                [item["path"] for item in document.get("inputs", ())]):
            return {"ok": False, "reason": "project_record_mismatch"}
    diagnostics = None
    try:
        diagnostics = read_json(base / DIAGNOSTICS_NAME)
    except (DiagnosticRefusal, OSError, json.JSONDecodeError,
            UnicodeDecodeError):
        return {"ok": False, "reason": "journal_unreadable"}
    if diagnostics.get("terminal") != terminal:
        return {"ok": False, "reason": "journal_terminal"}
    if diagnostics.get("schema") != DIAGNOSTICS_SCHEMA:
        return {"ok": False, "reason": "journal_schema"}
    frozen = {entry["name"]: entry["sha256"]
              for entry in freeze.get("artifacts", [])}
    if frozen.get(DIAGNOSTICS_NAME) != hash_file(base / DIAGNOSTICS_NAME):
        return {"ok": False, "reason": "journal_digest"}
    if terminal == progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION:
        if frozen.get(SCORE_NAME) != hash_file(base / SCORE_NAME):
            return {"ok": False, "reason": "scorecard_digest_file"}
        if diagnostics.get("scorecard_sha256") != (
                read_json(base / SCORE_NAME).get("scorecard_sha256")):
            return {"ok": False, "reason": "journal_scorecard"}
        record_stage = None
        for item in diagnostics.get("checkpoints", ()):
            if type(item) is dict and item.get("stage") == "record":
                record_stage = item
        if (type(record_stage) is not dict
                or record_stage.get("record_file_sha256")
                != frozen.get(route_b.RECORD_NAME)):
            return {"ok": False, "reason": "journal_record"}
        checkpoints = {item.get("stage"): item for item in diagnostics.get("checkpoints", ())
                       if type(item) is dict}
        entry = checkpoints.get("entry", {})
        stages = document.get("stage_digests", {})
        if (entry.get("project_file_sha256") != frozen.get(route_b.PROJECT_NAME)
                or entry.get("session_sha256") != stages.get("session_sha256")
                or entry.get("entry_sha256") != stages.get("entry_sha256")):
            return {"ok": False, "reason": "journal_project_entry"}
        for stage in ("lattice", "signatures", "graph"):
            if checkpoints.get(stage, {}).get("result_sha256") != stages.get(stage + "_sha256"):
                return {"ok": False, "reason": "journal_stage"}
        if document.get("settings") != route_b.resolve_settings():
            return {"ok": False, "reason": "record_settings"}
    else:
        refusal_path = base / route_b.REFUSAL_NAME
        if not refusal_path.is_file():
            return {"ok": False, "reason": "refusal_missing"}
        if frozen.get(route_b.REFUSAL_NAME) != hash_file(refusal_path):
            return {"ok": False, "reason": "refusal_digest_file"}
        refusal = read_json(refusal_path)
        if (refusal.get("schema") != route_b.REFUSAL_SCHEMA
                or refusal.get("status") != "refused"
                or not isinstance(refusal.get("reason"), str)):
            return {"ok": False, "reason": "refusal_schema"}
        graph = next((item for item in diagnostics.get("checkpoints", ())
                      if type(item) is dict and item.get("stage") == "graph"), None)
        expected_terminal = progress.classify_outcome(
            refusal["reason"], refusal.get("field", ""), graph_checkpoint=graph)
        if refusal["reason"] in {"resource_wall_exceeded", "resource_memory_exceeded",
                                 "stream_over_cap", "resource_aborted"}:
            expected_terminal = progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE
        if expected_terminal != terminal:
            return {"ok": False, "reason": "refusal_terminal"}
    return {"ok": True}


def _freeze_binding_ok(freeze, base, *, snapshots=None) -> dict:
    try:
        return _freeze_binding_unchecked(freeze, base, snapshots=snapshots)
    except (DiagnosticRefusal, ValueError, OSError, TypeError, KeyError,
            AttributeError) as error:
        return {"ok": False, "reason": type(error).__name__}


class RetainedFreezeSnapshot:
    """Retain roots/files and one byte snapshot for all digest/semantic checks.

    File contents and root identity are excluded from foreign mutation while held.
    Child-name creation remains possible; double inventory observation cannot seal
    a hostile directory namespace. This deliberate limit keeps release HOLD.
    """
    namespace_exclusive = False

    def __init__(self, root, names, *, ownership=None):
        from mc_photogrammetry.project.store import _WindowsRootAnchor, _WindowsRetainedFile
        self.root, self.names = Path(root), set(names)
        self.anchor = _WindowsRootAnchor(root)
        self.held, self.bytes, self.identities = [], {}, {}
        try:
            self.check_inventory()
            opened = {}
            for name in sorted(self.names):
                path = Path(os.path.abspath(self.root / name))
                held = None if ownership is None else ownership._handles.get(path)
                if held is None:
                    held = _WindowsRetainedFile(path)
                    self.held.append(held)
                held._verify()
                opened[name] = held
                self.identities[name] = held.identity
            for name, held in opened.items():
                cap = MAX_FREEZE_BYTES if name == FREEZE_NAME else progress.FREEZE_ARTIFACT_CAPS[name]
                self.bytes[name] = held.read_bounded(cap)
            self.check_inventory()
        except BaseException:
            self.close()
            raise

    def check_inventory(self):
        self.anchor.verify()
        if set(os.listdir(self.root)) != self.names:
            raise progress.TemplateMatchProgressError("phase_a_not_frozen", "root", "inventory changed/foreign")

    def close(self):
        for held in reversed(self.held):
            held.close()
        self.held.clear()
        self.anchor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def revalidate_freeze(freeze: dict, *, root=None, ownership=None) -> dict:
    """The gate every evaluator loader must pass before opening Phase A output."""
    if (type(freeze) is not dict or freeze.get("schema") != progress.FREEZE_SCHEMA
            or not isinstance(freeze.get("root"), str) or not freeze["root"]):
        raise progress.TemplateMatchProgressError("phase_a_not_frozen", "freeze")
    if root is None:
        raise progress.TemplateMatchProgressError("phase_a_not_frozen", "root", "caller root required")
    base = Path(root)
    route_b._refuse_reparse_components(base)
    if os.path.normcase(os.path.abspath(base)) != os.path.normcase(
            os.path.abspath(str(freeze.get("root")))):
        raise progress.TemplateMatchProgressError("phase_a_not_frozen", "root", "caller root mismatch")
    expected = set(progress.FREEZE_ARTIFACTS_BY_TERMINAL.get(freeze.get("terminal"), ()))
    if set(os.listdir(base)) != expected | {FREEZE_NAME}:
        raise progress.TemplateMatchProgressError("phase_a_not_frozen", "root", "unexpected/missing artifact")
    if os.name == "nt":
        try:
            with RetainedFreezeSnapshot(base, expected | {FREEZE_NAME}, ownership=ownership) as snapshot:
                if json.loads(snapshot.bytes[FREEZE_NAME].decode("utf-8")) != freeze:
                    raise progress.TemplateMatchProgressError("phase_a_not_frozen", "freeze", "seal mismatch")
                result = progress.assert_phase_a_frozen(
                    freeze, root=base,
                    read_sha256=lambda path: hashlib.sha256(snapshot.bytes[Path(path).name]).hexdigest(),
                    read_size=lambda path: len(snapshot.bytes[Path(path).name]),
                    is_reparse=lambda path: False,
                    read_binding=lambda record, path: _freeze_binding_ok(record, path, snapshots=snapshot.bytes))
                snapshot.check_inventory()
                return result
        except (OSError, ValueError, KeyError, RecursionError) as error:
            if isinstance(error, progress.TemplateMatchProgressError):
                raise
            raise progress.TemplateMatchProgressError("phase_a_not_frozen", "root", str(error)) from error
    return progress.assert_phase_a_frozen(
        freeze, read_sha256=_sha256_file,
        root=base,
        read_size=lambda path: os.stat(path).st_size,
        is_reparse=_is_reparse_point,
        read_binding=_freeze_binding_ok)


def load_freeze(root: Path) -> dict:
    """Load and revalidate the frozen Phase A artifact identity."""
    path = Path(root) / FREEZE_NAME
    if not path.is_file():
        raise DiagnosticRefusal("phase_a_not_frozen", "freeze",
                                "no Phase A freeze record exists")
    try:
        freeze = _read_json_bounded(path)
    except (DiagnosticRefusal, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DiagnosticRefusal("phase_a_not_frozen", "freeze",
                                type(error).__name__) from error
    return revalidate_freeze(freeze, root=root)


# ---------------------------------------------------------------------------
# The single in-process pass (the supervised worker body)
# ---------------------------------------------------------------------------

def _prelaunch_refusal(root, resolved, settings, error) -> None:
    """Persist a pre-launch refusal in a prepared fresh root (best effort)."""
    try:
        root, staging = route_b._prepare_root(Path(root))
    except DiagnosticRefusal:
        return
    route_b._discard_staging(staging, staging / route_b.PROJECT_NAME)
    route_b._write_refusal(root, error, resolved, settings)
    try:
        journal = DiagnosticsJournal(
            root / DIAGNOSTICS_NAME,
            max_bytes=ResourceLimits().diagnostics_bytes,
            terminal=progress.classify_refusal(error.reason, error.field))
        journal.set_extra(prelaunch_refusal={
            "reason": error.reason, "field": error.field,
            "detail": error.detail})
    except DiagnosticRefusal:
        pass


def _execute_pass(*, root, resolved, api, limits, governor, expected_pins,
                  clock, process_tree_probe, settings=None) -> dict:
    """Run the one bounded pass and persist diagnostics/scorecard/freeze.

    This is the worker body: it is executed inside the supervised child in the
    real path and in-process by a synthetic fake supervisor in tests. It refuses
    a ``None`` or unmarked API *before* the route runner can resolve one, so the
    production/default route-B API can never be selected through this path.
    """
    if api is None or not _is_synthetic_test_only(api):
        raise DiagnosticRefusal(
            "resource_enforcement_unavailable", "api",
            "the private pass refuses to resolve a default or production api; "
            "only an explicitly marked synthetic test api may be used")
    root = Path(root)
    output = OutputBudget(limits)
    # Reserve the actual producer's maximum staged project before entry. Its
    # bounded store never writes beyond this reservation, including temporary.
    output.charge(limits.project_bytes, field=route_b.PROJECT_NAME)
    ownership = route_b.InvocationOwnership(native=(os.name == "nt"))
    try:
        snapshots: list = []
        terminal = progress.TERMINAL_EARLY_REFUSAL_NO_CAPTURE
        journal = DiagnosticsJournal(root / DIAGNOSTICS_NAME,
                                     max_bytes=limits.diagnostics_bytes,
                                     terminal=terminal, ownership=ownership,
                                     output=output)
        journal.extra.update(limits=limits.record(), pins=list(expected_pins))

        def checkpoint(record: dict) -> None:
            if governor is not None:
                probe = governor.check()
                record = dict(record)
                record.update(probe)
            journal.append(record)

        graph_checkpoint: dict = {}

        def graph_aware_checkpoint(record: dict) -> None:
            if record.get("stage") == "graph":
                graph_checkpoint.clear()
                graph_checkpoint.update(record)
            checkpoint(record)

        try:
            report = route_b.execute_route_b_diagnostic(
                image_paths=[str(path) for path in resolved], root=root, api=api,
                expected_pins=expected_pins,
                checkpoint_sink=graph_aware_checkpoint,
                observer=snapshots, project_max_bytes=limits.project_bytes,
                output_budget=output, invocation_ownership=ownership)
        except ResourceAbort as error:
            terminal = progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE
            _set_terminal_best_effort(journal, terminal)
            _set_extra_best_effort(journal, resource_abort={
                "reason": error.reason, "field": error.field,
                "detail": error.detail})
            # The legacy runner does not write a refusal for a watchdog abort, so the
            # harness writes the typed refusal record itself; the abort terminal's
            # exact artifact set (diagnostics + refusal) can then be sealed.
            if settings is not None:
                route_b._write_refusal(
                    root, DiagnosticRefusal(error.reason, error.field,
                                            error.detail),
                    resolved, settings, ownership=ownership)
            _seal_refusal_terminal(root, terminal, limits, ownership=ownership)
            raise DiagnosticRefusal(error.reason, error.field, error.detail) from error
        except DiagnosticRefusal as error:
            if _root_refusal_blocks_writes(error):
                # A colliding, non-fresh, reparse or uninspectable root gets *zero*
                # writes, including failure diagnostics: there is no journal flush,
                # no refusal record and no freeze in a root this call does not own.
                raise
            terminal = progress.classify_outcome(
                error.reason, error.field, graph_checkpoint=graph_checkpoint)
            _set_terminal_best_effort(journal, terminal)
            _seal_refusal_terminal(root, terminal, limits, ownership=ownership)
            raise

        published = (route_b.RECORD_NAME, route_b.PROJECT_NAME)
        try:
            document = route_b.validate_record(report["record"], snapshot_bytes=ownership.read_bytes(report["record"], limits.record_bytes) if ownership.native else None)
            from mc_photogrammetry.pipeline.frame_ingress import (
                FrameIngressProvenanceV1)
            ingress_by_frame = {
                item["frame_id"]: FrameIngressProvenanceV1.from_record(item)
                for item in document.get("ingress", ())}
            scorecard = progress.build_progress_scorecard(
                capture_record=document["capture"],
                section=document.get("sampling_persistence"),
                ingress_by_frame=ingress_by_frame,
                terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)
            score_path = root / SCORE_NAME
            route_b._atomic_publish(
                score_path, scorecard, field="scorecard", ownership=ownership,
                max_bytes=progress.FREEZE_ARTIFACT_CAPS["scorecard.json"],
                output_budget=output)
            output.charge(score_path.stat().st_size, field=SCORE_NAME)
            journal.set_extra(scorecard_sha256=scorecard["scorecard_sha256"],
                              scorecard=scorecard)
            journal.set_terminal(progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION)

            for name in (route_b.RECORD_NAME, route_b.PROJECT_NAME,
                         DIAGNOSTICS_NAME):
                output.charge((root / name).stat().st_size, field=name)
            freeze = freeze_root(
                root, terminal=progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION,
                limits=limits, ownership=ownership, output_budget=output)
            output.charge((root / FREEZE_NAME).stat().st_size, field=FREEZE_NAME)
        except BaseException:
            _rollback_published(root, report, ownership)
            raise

        report = dict(report)
        report.update({
            "schema": SCHEMA,
            "terminal": progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION,
            "scorecard": scorecard,
            "scorecard_path": str(score_path),
            "diagnostics_path": str(root / DIAGNOSTICS_NAME),
            "freeze_path": str(root / FREEZE_NAME),
            "freeze": freeze,
            "resource": None if governor is None else governor.record(),
            "output": output.record(),
            "output_total_bytes": output.total,
            "solver_diagnostics": list(report.get("solver_diagnostics", [])),
        })
        return report

    finally:
        ownership.close()


def _rollback_published(root, report, ownership) -> None:
    """Delete exactly this invocation's record/project/scorecard/freeze.

    Native captures delete through their retained birth handles. Legacy callers
    have only the narrower sequential pathname-identity protection.
    """
    ownership.rollback()


def _seal_refusal_terminal(root, terminal, limits, *, ownership=None) -> None:
    """Best-effort terminal-exact freeze for a refused/aborted run."""
    try:
        freeze_root(root, terminal=terminal, limits=limits, ownership=ownership)
    except (DiagnosticRefusal, OSError):
        pass


def _root_refusal_blocks_writes(error) -> bool:
    """True when a refusal means the root is not this invocation's to write."""
    return (error.reason in ("output_exists", "root_not_fresh", "root_reparse",
                             "root_uninspectable")
            and error.field in ("root", "", None))


def _set_terminal_best_effort(journal: DiagnosticsJournal, terminal: str) -> None:
    try:
        journal.set_terminal(terminal)
    except DiagnosticRefusal:
        pass


def _set_extra_best_effort(journal: DiagnosticsJournal, **values) -> None:
    try:
        journal.set_extra(**values)
    except DiagnosticRefusal:
        pass


def _record_supervisor_abort(root, resolved, settings, outcome, limits) -> None:
    """Persist an externally observed abort; only ever into a fresh root."""
    terminal = progress.TERMINAL_RESOURCE_ABORTED_NO_CAPTURE
    try:
        root, staging = route_b._prepare_root(Path(root))
    except DiagnosticRefusal:
        return
    route_b._discard_staging(staging, staging / route_b.PROJECT_NAME)
    refusal = DiagnosticRefusal(outcome.breach_reason, "containment",
                                f"supervisor observed {outcome.breach_reason}")
    route_b._write_refusal(root, refusal, resolved, settings)
    journal = None
    try:
        journal = DiagnosticsJournal(
            root / DIAGNOSTICS_NAME, max_bytes=limits.diagnostics_bytes,
            terminal=terminal)
        journal.set_extra(supervisor_abort={
            "reason": outcome.breach_reason,
            "timed_out": outcome.timed_out,
            "memory_exceeded": outcome.memory_exceeded,
            "stream_exceeded": outcome.stream_exceeded,
            "peak_job_bytes": outcome.peak_job_bytes,
            "stream_bytes": outcome.stream_bytes,
            "root_entries": list(outcome.root_entries),
        })
    except DiagnosticRefusal:
        pass
    try:
        _seal_refusal_terminal(root, terminal, limits,
                               ownership=None if journal is None else journal._ownership)
    finally:
        if journal is not None:
            journal.close()


# ---------------------------------------------------------------------------
# The harness entry point
# ---------------------------------------------------------------------------

def _worker_command(resolved, root) -> tuple:
    """Private argv for the disabled worker entry (it now always refuses)."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--root", str(root)]
    for path in resolved:
        command.extend(["--image", str(path)])
    return tuple(command)


def execute_template_match_progress(
    *,
    image_paths,
    root,
    api=None,
    limits: ResourceLimits | None = None,
    supervisor: RunSupervisor | None = None,
    governor: ResourceGovernor | None = None,
    clock: Callable[[], float] = time.monotonic,
    process_tree_probe: Callable[[], int] | None = None,
    expected_pins=progress.T2_PIN_SHA256S,
) -> dict:
    """Run the one bounded Phase A pass under an injected, proven supervisor.

    **Interim fail-closed state:** with the default ``supervisor=None`` this
    refuses the typed ``resource_enforcement_unavailable`` **before**
    ``api.entry`` and never calls entry/lattice/signatures/graph, because the OS
    supervisor path is disabled. The public API, the CLI and every non-test call
    also refuse before ``_execute_pass``. Only the private synthetic test seam may
    reach the pass, and only when the injected supervisor **and** the injected API
    both carry the explicit ``synthetic_test_only is True`` marker (a code-enforced
    test-only convention, not an authentication boundary). The private seam itself
    still has unresolved internal defects and is not accepted for real inputs; see
    ``PRIVATE_SEAM_REENABLE_BLOCKERS``.

    For that injected seam, containment is proven **before** ``api.entry``. The
    pass itself is the accepted single-pass runner; this harness adds the
    pre-entry containment gate, the ordered T2 pin gate, the bounded stage
    journal, the same-pass solver observer, the Phase A scorecard, the
    terminal-exact freeze and the external root/stream supervision. The runner is
    entered exactly once.
    """
    limits = limits or ResourceLimits()
    resolved = route_b.resolve_images(image_paths)
    settings = route_b.resolve_settings()

    try:
        supervisor = establish_containment(supervisor)
        require_synthetic_test_seam(supervisor, api)
    except DiagnosticRefusal as error:
        _prelaunch_refusal(root, resolved, settings, error)
        raise

    if governor is None:
        try:
            governor = establish_governor(
                limits=limits, clock=clock,
                process_tree_probe=process_tree_probe)
        except DiagnosticRefusal as error:
            _prelaunch_refusal(root, resolved, settings, error)
            raise

    root = Path(os.path.abspath(root))
    worker = (lambda: _execute_pass(
        root=root, resolved=resolved, api=api, limits=limits,
        governor=governor, expected_pins=expected_pins, clock=clock,
        process_tree_probe=process_tree_probe, settings=settings))
    # The combined stdout/stderr capture lives in a bounded, invocation-owned
    # sibling directory (created before launch, removed after) because the fresh
    # output root is created by the contained child itself and must contain only
    # terminal artifacts. Nothing is written outside an owned path.
    stream_dir = None
    stream_path = str(root / STREAM_NAME)
    if getattr(supervisor, "spawns_child", False):
        stream_dir = root.parent / (root.name + ".supervisor")
        try:
            stream_dir.mkdir()
        except FileExistsError as error:
            refusal = DiagnosticRefusal(
                "resource_enforcement_unavailable", "containment",
                "the supervisor stream area already exists")
            _prelaunch_refusal(root, resolved, settings, refusal)
            raise refusal from error
        stream_path = str(stream_dir / STREAM_NAME)
    spec = SupervisedSpec(
        command=_worker_command(resolved, root), cwd=str(REPO_ROOT),
        root=str(root), stdout_path=stream_path, stderr_path=stream_path,
        stream_path=stream_path,
        wall_seconds=limits.wall_seconds,
        memory_bytes=limits.process_tree_bytes,
        stream_max_bytes=limits.stream_bytes,
        worker=worker)
    try:
        outcome = supervisor.run(spec)
    finally:
        if stream_dir is not None:
            route_b._unlink_if_present(stream_dir / STREAM_NAME)
            route_b._rmdir_if_empty(stream_dir)

    if outcome.error is not None:
        raise outcome.error
    if outcome.killed:
        _record_supervisor_abort(root, resolved, settings, outcome, limits)
        raise DiagnosticRefusal(
            outcome.breach_reason, "containment",
            f"supervisor aborted the run: {outcome.breach_reason} "
            f"(job_assigned_before_start={outcome.job_assigned_before_start})")

    if not outcome.job_assigned_before_start:
        raise DiagnosticRefusal("resource_enforcement_unavailable", "containment",
                                "the supervisor did not assign the child before it started")
    if outcome.exit_code != 0:
        # Never turn an exited child's ordinary refusal into record_missing.
        if (root / route_b.REFUSAL_NAME).is_file():
            refusal = _read_json_bounded(root / route_b.REFUSAL_NAME)
            raise DiagnosticRefusal(refusal.get("reason", "child_refused"),
                                    refusal.get("field", "child"),
                                    refusal.get("detail", ""))
        raise DiagnosticRefusal("child_abnormal_exit", "child", str(outcome.exit_code))
    _supervise_root(root, limits)
    load_freeze(root)
    report = outcome.report
    if report is None:
        report = _report_from_artifacts(root)
    report = dict(report)
    report["supervision"] = {
        "kind": supervisor.kind,
        "job_assigned_before_start": outcome.job_assigned_before_start,
        "exit_code": outcome.exit_code,
        "peak_job_bytes": outcome.peak_job_bytes,
        "stream_bytes": outcome.stream_bytes,
        "root_entries": list(outcome.root_entries),
    }
    return report


def _supervise_root(root, limits=None) -> None:
    """External root check: exact terminal-exact set, no reparse, no foreign."""
    root = Path(root)
    if _is_reparse_point(root):
        raise DiagnosticRefusal("root_reparse", "root")
    accepted = {FREEZE_NAME, STREAM_NAME}
    for names in progress.FREEZE_ARTIFACTS_BY_TERMINAL.values():
        accepted.update(names)
    for name in os.listdir(root):
        if name in accepted:
            continue
        raise DiagnosticRefusal("root_foreign_artifact", name,
                                "supervised root contains an unexpected entry")
    if (root / STREAM_NAME).is_file():
        size = (root / STREAM_NAME).stat().st_size
        cap = (limits or ResourceLimits()).stream_bytes
        if size > cap:
            raise DiagnosticRefusal("stream_over_cap", "stdout_stderr",
                                    f"{size} bytes")


def _report_from_artifacts(root) -> dict:
    """Reconstruct the minimal report for a real supervised child."""
    root = Path(root)
    record_path = root / route_b.RECORD_NAME
    if not record_path.is_file():
        raise DiagnosticRefusal("record_missing", "record")
    document = route_b.validate_record(record_path)
    diagnostics = _read_json_bounded(root / DIAGNOSTICS_NAME)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "terminal": diagnostics.get("terminal"),
        "root": str(root),
        "record": str(record_path),
        "record_file_sha256": _sha256_file(record_path),
        "expected_row_count": document.get("expected_row_count"),
        "row_count": document.get("capture", {}).get("row_count"),
        "solver_diagnostics": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase A bounded Template 2 Route B stage-measurement harness")
    parser.add_argument("--image", action="append", default=None,
                        help="ordinary screenshot path (repeatable, in order)")
    parser.add_argument("--root", default=None, help="fresh output root")
    parser.add_argument("--check", action="store_true",
                        help="revalidate the freeze record in --root and exit")
    parser.add_argument("--worker", action="store_true",
                        help="disabled; the direct worker entry always refuses")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.check:
        if args.root is None:
            print("--root is required with --check", file=sys.stderr)
            return 2
        try:
            result = load_freeze(Path(args.root))
        except (DiagnosticRefusal, progress.TemplateMatchProgressError) as refusal:
            print(json.dumps({"status": "refused", "reason": refusal.reason,
                              "field": refusal.field, "detail": refusal.detail},
                             sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps({"status": result["status"],
                          "artifact_count": result["artifact_count"]},
                         sort_keys=True))
        return 0
    if args.worker:
        # Disabled: the direct worker entry point refuses closed before
        # ``_execute_pass``, so there is no unauthenticated worker path.
        refusal = DiagnosticRefusal(
            "resource_enforcement_unavailable", "containment",
            SUPERVISOR_DISABLED_REASON)
        print(json.dumps({"status": "refused", "reason": refusal.reason,
                          "field": refusal.field, "detail": refusal.detail},
                         sort_keys=True), file=sys.stderr)
        return 1
    if args.image is None or args.root is None:
        print("--image and --root are required to run", file=sys.stderr)
        return 2
    try:
        report = execute_template_match_progress(
            image_paths=args.image, root=Path(args.root))
    except DiagnosticRefusal as refusal:
        print(json.dumps({"status": "refused", "reason": refusal.reason,
                          "field": refusal.field, "detail": refusal.detail},
                         sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"],
                      "terminal": report["terminal"],
                      "record": report["record"],
                      "row_count": report["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
