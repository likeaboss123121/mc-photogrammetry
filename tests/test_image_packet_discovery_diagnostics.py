"""Exercise extracted supervisor functions against authored providers, never images."""

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_image_packet.ps1"


def run_control(tmp_path, body, *, fake_handles=False):
    # Parsing declarations does not execute the supervisor or its launch path.
    source = str(RUNNER).replace("'", "''")
    bootstrap = r"""
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('__RUNNER__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'source_parse_failure' }
$names = @('Get-PacketField', 'Assert-Packet', 'Get-PacketDiagnosticCode',
           'Get-PacketMonitorSeconds', 'Get-PacketBoundedInterval', 'Update-PacketSnapshotTelemetry',
           'Get-PacketProcessSnapshot', 'Get-PacketCreation', 'Update-PacketExit',
           'Register-PacketProcess', 'Get-PacketSample')
if ('__FAKE_HANDLES__' -eq 'True') { $names += 'Stop-PacketTree' }
foreach ($name in $names) {
    $nodes = @($ast.FindAll({ param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $true))
    if ($nodes.Count -ne 1) { throw 'function_not_unique' }
    $functionText = $nodes[0].Extent.Text
    if ('__FAKE_HANDLES__' -eq 'True' -and $name -in @('Get-PacketSample', 'Stop-PacketTree')) {
        # Test-only OS binding seam. Every original join/identity/cleanup statement stays exact.
        $needle = if ($name -eq 'Get-PacketSample') {
            '[Diagnostics.Process]::GetProcessById($entry.Pid)'
        } else { '[Diagnostics.Process]::GetProcessById($tracked.Pid)' }
        if ($functionText.Split([string[]]@($needle), [StringSplitOptions]::None).Count -ne 2) {
            throw 'handle_binding_seam_not_unique'
        }
        $replacement = if ($name -eq 'Get-PacketSample') {
            'Get-ControlProcessById $entry.Pid'
        } else { 'Get-ControlProcessById $tracked.Pid' }
        $functionText = $functionText.Replace($needle, $replacement)
    }
    . ([ScriptBlock]::Create($functionText))
}
$mainTry = @($ast.FindAll({ param($node)
    $node -is [Management.Automation.Language.TryStatementAst] -and
    $node.CatchClauses.Count -eq 1 -and
    $node.CatchClauses[0].Extent.Text.Contains('$packetRunnerReceipt.error_type =')
}, $true))
if ($mainTry.Count -ne 1) { throw 'main_catch_not_unique' }
$actualCatch = $mainTry[0].CatchClauses[0].Extent.Text
$actualFinally = $mainTry[0].Finally.Extent.Text
$packetRunnerReceipt = [ordered]@{ discovery_failure = $null; accounting_complete = $true;
    status = 'running'; reason = $null; launched = $true; launch_attempts = 1;
    peak_aggregate_working_set_bytes = 0; cleanup_attempted = $false;
    process_snapshot_telemetry = $null; cleanup_errors = @(); cleanup_complete = $null;
    cleanup_elapsed_seconds = 0.0; cleanup_overshoot_seconds = 0.0;
    root_exit_observed = $false; root_exit_code = $null; monitor_elapsed_seconds = 0.0;
    drain_elapsed_seconds = 0.0; drain_overshoot_seconds = 0.0; drain_complete = $false;
    failure_reason_before_drain = $null; logs = @(); outputs = @(); pre_inputs = @(); post_inputs = @() }
$packetRunnerTracked = [ordered]@{}
$packetRunnerSamples = New-Object 'Collections.Generic.List[object]'
$packetRunnerClock = [Diagnostics.Stopwatch]::StartNew()
$script:packetRunnerPreviousSnapshotStartSeconds = $null
$script:providerCalls = 0; $script:cleanupCalls = 0; $script:writes = 0
function Write-PacketReceipt { $script:writes++ }
if ('__FAKE_HANDLES__' -ne 'True') {
    function Stop-PacketTree { $script:cleanupCalls++; $packetRunnerReceipt.cleanup_attempted = $true }
}
function Invoke-ActualCatch([ScriptBlock]$Work) {
    . ([ScriptBlock]::Create('try { & $Work } ' + $actualCatch))
}
function Invoke-ActualFinally {
    . ([ScriptBlock]::Create('try {} finally ' + $actualFinally))
}
""".replace("__RUNNER__", source).replace("__FAKE_HANDLES__", str(fake_handles))
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "control.ps1"
    script.write_text(bootstrap + body, encoding="utf-8")
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_prior_good_sample_then_cim_failure_is_not_empty_or_stale(tmp_path):
    record = run_control(tmp_path, r"""
$creation = [DateTime]::UtcNow.AddMinutes(-1)
$fakeHandle = [pscustomobject]@{ HasExited = $false; StartTime = $creation; WorkingSet64 = 12345 }
$fakeHandle | Add-Member ScriptMethod Refresh { }
$key = '101|' + (Get-PacketCreation $creation)
$packetRunnerTracked[$key] = [pscustomobject]@{ Pid = 101; Creation = Get-PacketCreation $creation;
    ParentPid = 0; ParentCreation = ''; Handle = $fakeHandle; LastSeen = $null;
    ExitObserved = $false; ExitCode = $null }
$script:originalFailure = New-Object Microsoft.Management.Infrastructure.CimException('sensitive/path ' + ('x' * 50000))
$script:forwarded = $null
function Get-CimInstance {
    param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++
    $script:forwarded = @($ClassName) + @($Property) + @($OperationTimeoutSec, $ErrorAction)
    if ($script:providerCalls -eq 1) {
        return [pscustomobject]@{ ProcessId = 101; ParentProcessId = 0; CreationDate = $creation }
    }
    throw $script:originalFailure
}
$good = Get-PacketSample
$script:sameException = $false
Invoke-ActualCatch {
    try { Get-PacketSample } catch {
        $script:sameException = [Object]::ReferenceEquals($_.Exception, $script:originalFailure)
        throw
    }
}
@{ receipt = $packetRunnerReceipt; calls = $script:providerCalls; forwarded = $script:forwarded;
   samples = @($packetRunnerSamples.ToArray()); same_exception = $script:sameException;
   cleanup_calls = $script:cleanupCalls; writes = $script:writes } | ConvertTo-Json -Depth 12
""")
    failure = record["receipt"]["discovery_failure"]
    assert record["calls"] == 2
    assert record["forwarded"] == ["Win32_Process", "ProcessId", "ParentProcessId", "CreationDate", 1, "Stop"]
    assert record["same_exception"] is True
    assert record["cleanup_calls"] == 1
    assert record["writes"] == 1
    assert len(record["samples"]) == 1
    assert record["samples"][0]["aggregate_working_set_bytes"] == 12345
    assert record["samples"][0]["live_processes"] == 1
    assert record["receipt"]["peak_aggregate_working_set_bytes"] == 12345
    assert record["receipt"]["launch_attempts"] == 1
    assert record["receipt"]["status"] == "supervisor_failed"
    assert record["receipt"]["reason"] == "process_snapshot_failed"
    assert record["receipt"]["termination_reason"] == "supervisor_exception"
    assert record["receipt"]["accounting_complete"] is False
    assert failure["stage"] == "process_snapshot"
    assert failure["completed_samples"] == 1
    assert failure["exception_type"] == "Microsoft.Management.Infrastructure.CimException"
    assert set(failure) == {"stage", "reason", "exception_type", "hresult",
                            "native_error_code", "status_code", "completed_samples", "elapsed_seconds", "query_telemetry"}
    assert len(json.dumps(failure)) < 1024
    assert "sensitive" not in json.dumps(record)
    assert "xxxxx" not in json.dumps(record)


def test_valid_empty_snapshot_is_a_successful_new_sample(tmp_path):
    record = run_control(tmp_path, r"""
function Get-CimInstance { param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++; return @() }
$sample = Get-PacketSample
@{ receipt = $packetRunnerReceipt; calls = $script:providerCalls;
   samples = @($packetRunnerSamples.ToArray()); cleanup_calls = $script:cleanupCalls } | ConvertTo-Json -Depth 8
""")
    assert record["calls"] == 1
    assert record["cleanup_calls"] == 0
    assert record["receipt"]["discovery_failure"] is None
    assert record["receipt"]["accounting_complete"] is True
    assert record["receipt"]["status"] == "running"
    assert len(record["samples"]) == 1
    assert record["samples"][0]["aggregate_working_set_bytes"] == 0
    assert record["samples"][0]["live_processes"] == 0
    assert record["samples"][0]["processes"] == []


def test_unexpected_provider_exception_is_bounded_and_failed(tmp_path):
    record = run_control(tmp_path, r"""
function Get-CimInstance { param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++; throw (New-Object InvalidOperationException('private ' + ('q' * 100000))) }
Invoke-ActualCatch { Get-PacketSample }
@{ receipt = $packetRunnerReceipt; calls = $script:providerCalls;
   count = $packetRunnerSamples.Count; cleanup_calls = $script:cleanupCalls } | ConvertTo-Json -Depth 8
""")
    assert record["calls"] == 1
    assert record["count"] == 0
    assert record["cleanup_calls"] == 1
    assert record["receipt"]["status"] == "supervisor_failed"
    assert record["receipt"]["accounting_complete"] is False
    failure = record["receipt"]["discovery_failure"]
    assert failure["exception_type"] == "System.InvalidOperationException"
    assert failure["native_error_code"] is None
    assert failure["status_code"] is None
    assert len(json.dumps(failure)) < 1024
    assert "private" not in json.dumps(record)


@pytest.mark.parametrize("value, expected", [
    ("[int32]-2147483648", -2147483648), ("[uint32]4294967295", 4294967295),
    ("[long]4294967296", None), ("[long]-2147483649", None),
    ("[double]1.5", None), ("'123'", None), ("$null", None),
    ("[DayOfWeek]::Friday", 5),
], ids=["signed-min", "unsigned-max", "too-large", "too-small", "float", "string", "missing", "enum"])
def test_native_metadata_accepts_only_bounded_integer_codes(tmp_path, value, expected):
    record = run_control(tmp_path, """
$fake = [pscustomobject]@{ NativeErrorCode = __VALUE__ }
@{ code = Get-PacketDiagnosticCode $fake 'NativeErrorCode' } | ConvertTo-Json
""".replace("__VALUE__", value))
    assert record["code"] == expected


def test_throwing_metadata_getter_cannot_replace_error(tmp_path):
    record = run_control(tmp_path, r"""
$fake = [pscustomobject]@{}
$fake | Add-Member ScriptProperty NativeErrorCode { throw 'private_metadata' }
@{ code = Get-PacketDiagnosticCode $fake 'NativeErrorCode' } | ConvertTo-Json
""")
    assert record["code"] is None


def test_snapshot_forwards_exact_objects_without_extra_query(tmp_path):
    record = run_control(tmp_path, r"""
$script:row = [pscustomobject]@{ ProcessId = 123; ParentProcessId = 0; CreationDate = [DateTime]::UtcNow }
function Get-CimInstance { param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++; return $script:row }
$returned = @(Get-PacketProcessSnapshot)
@{ identical = [Object]::ReferenceEquals($returned[0], $script:row); calls = $script:providerCalls;
   failure = $packetRunnerReceipt.discovery_failure } | ConvertTo-Json
""")
    assert record == {"identical": True, "calls": 1, "failure": None}


# Only the two native GetProcessById expressions in extracted sample/cleanup
# declarations are replaced with the authored handle provider. All production
# ancestry/creation/exit/working-set/cleanup/finally statements execute unchanged.
FAKE_PROCESS_MODEL = r"""
$script:base = [DateTime]'2020-01-01T00:00:00Z'
$script:fakeHandles = @{}
$script:fakeGetCalls = 0
$script:killLog = New-Object 'Collections.Generic.List[int]'
$packetRunnerCleanupSeconds = 5
$packetRunnerDrainSeconds = 10
$packetRunnerLogCap = 4194304
$packetRunnerClock = [pscustomobject]@{ Elapsed = [pscustomobject]@{ TotalSeconds = [double]0 } }
function New-ControlHandle([int]$ProcessId, [double]$Start, [long]$Working = 100) {
    $handle = [pscustomobject]@{ Id = $ProcessId; Handle = 1;
        StartTime = $script:base.AddSeconds($Start); HasExited = $false;
        ExitTime = $script:base.AddSeconds(60); ExitCode = 0;
        WorkingSet64 = $Working; Disposals = 0 }
    $handle | Add-Member ScriptMethod Refresh { }
    $handle | Add-Member ScriptMethod Dispose { $this.Disposals++ }
    $handle | Add-Member ScriptMethod Kill {
        $script:killLog.Add($this.Id); $this.HasExited = $true; $this.ExitCode = -1
    }
    $script:fakeHandles[$ProcessId] = $handle
    return $handle
}
function Get-ControlProcessById([int]$ProcessId) {
    $script:fakeGetCalls++
    if (-not $script:fakeHandles.ContainsKey($ProcessId)) {
        throw (New-Object ArgumentException('authored unavailable handle'))
    }
    return $script:fakeHandles[$ProcessId]
}
function New-ControlRow([int]$ProcessId, [int]$ParentId, [double]$Created) {
    return [pscustomobject]@{ ProcessId = $ProcessId; ParentProcessId = $ParentId;
        CreationDate = $script:base.AddSeconds($Created); UnusedPrivatePayload = 'private/' + ('x' * 10000) }
}
$root = New-ControlHandle 100 0 100
$packetRunnerProcess = $root
Register-PacketProcess 100 (Get-PacketCreation $root.StartTime) 0 '' 0 $root
"""


MODEL_FIXTURES = {
    "nested": r"""
$child = New-ControlHandle 200 1 200
$grandchild = New-ControlHandle 300 2 300
$script:rows = @((New-ControlRow 100 0 0), (New-ControlRow 200 100 1),
                 (New-ControlRow 300 200 2), (New-ControlRow 900 0 3))
""",
    "nested-reordered": r"""
$child = New-ControlHandle 200 1 200
$grandchild = New-ControlHandle 300 2 300
$script:rows = @((New-ControlRow 900 0 3), (New-ControlRow 300 200 2),
                 (New-ControlRow 200 100 1), (New-ControlRow 100 0 0))
""",
    "exited-parent": r"""
$root.HasExited = $true; $root.ExitTime = $script:base.AddSeconds(10)
$child = New-ControlHandle 200 1 200
$child.HasExited = $true; $child.ExitTime = $script:base.AddSeconds(8)
Register-PacketProcess 200 (Get-PacketCreation $child.StartTime) 100 (Get-PacketCreation $root.StartTime) 1 $child
$grandchild = New-ControlHandle 300 7 300
$script:rows = @((New-ControlRow 300 200 7))
""",
    "after-parent-exit": r"""
$root.HasExited = $true; $root.ExitTime = $script:base.AddSeconds(10)
$child = New-ControlHandle 200 11 200
$script:rows = @((New-ControlRow 200 100 11))
""",
    "reused-parent": r"""
$root.HasExited = $true; $root.ExitTime = $script:base.AddSeconds(10)
$replacement = New-ControlHandle 100 20 999
$child = New-ControlHandle 200 21 200
$script:rows = @((New-ControlRow 100 0 20), (New-ControlRow 200 100 21))
""",
    "reused-child-race": r"""
$replacement = New-ControlHandle 200 8 999
$script:rows = @((New-ControlRow 100 0 0), (New-ControlRow 200 100 1))
""",
    "vanished-child-race": r"""
$script:rows = @((New-ControlRow 100 0 0), (New-ControlRow 200 100 1))
""",
}


def sample_model(tmp_path, scenario, *, full):
    body = FAKE_PROCESS_MODEL + MODEL_FIXTURES[scenario] + r"""
$script:fullRows = __FULL__
function Get-CimInstance {
    param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++
    $script:forwarded = @($ClassName) + @($Property) + @($OperationTimeoutSec, $ErrorAction)
    if ($args.Count -ne 0) { throw 'unexpected_query_argument' }
    foreach ($row in $script:rows) {
        if ($script:fullRows) { $row } else {
            [pscustomobject]@{ ProcessId = $row.ProcessId; ParentProcessId = $row.ParentProcessId;
                CreationDate = $row.CreationDate }
        }
    }
}
Invoke-ActualCatch { Get-PacketSample | Out-Null }
$sampleRows = @($packetRunnerSamples.ToArray() | ForEach-Object {
    @{ elapsed_seconds = $_.elapsed_seconds; aggregate_working_set_bytes = $_.aggregate_working_set_bytes;
       live_processes = $_.live_processes; processes = @($_.processes | Sort-Object pid);
       discovery_telemetry = $_.discovery_telemetry }
})
$trackedRows = @($packetRunnerTracked.Values | Sort-Object Pid | ForEach-Object {
    @{ pid = $_.Pid; creation = $_.Creation; parent = $_.ParentPid;
       parent_creation = $_.ParentCreation; depth = $_.Depth; exited = $_.ExitObserved }
})
@{ receipt = $packetRunnerReceipt; samples = $sampleRows; tracked = $trackedRows;
   calls = $script:providerCalls; forwarded = $script:forwarded;
   handle_calls = $script:fakeGetCalls; kills = @($script:killLog.ToArray()) } | ConvertTo-Json -Depth 20
""".replace("__FULL__", "$true" if full else "$false")
    return run_control(tmp_path, body, fake_handles=True)


@pytest.mark.parametrize("scenario", tuple(MODEL_FIXTURES))
def test_full_and_projected_rows_preserve_joins_accounting_and_races(tmp_path, scenario):
    full = sample_model(tmp_path / "full", scenario, full=True)
    projected = sample_model(tmp_path / "projected", scenario, full=False)
    # Cleanup deadline measurements vary; compare every semantic/accounting field.
    for record in (full, projected):
        for elapsed in ("cleanup_elapsed_seconds", "cleanup_overshoot_seconds"):
            assert record["receipt"].pop(elapsed) >= 0
        # Bound-handle cleanup polls until its unchanged deadline; poll count is timing-dependent.
        assert record.pop("handle_calls") >= 0
    assert full == projected
    assert projected["calls"] == 1
    assert projected["forwarded"] == ["Win32_Process", "ProcessId", "ParentProcessId", "CreationDate", 1, "Stop"]
    assert "private" not in json.dumps(projected)
    expected_ids = {"nested": [100, 200, 300], "nested-reordered": [100, 200, 300],
                    "exited-parent": [100, 200, 300], "after-parent-exit": [100],
                    "reused-parent": [100], "reused-child-race": [100],
                    "vanished-child-race": [100, 200]}[scenario]
    assert [row["pid"] for row in projected["tracked"]] == expected_ids
    if scenario == "vanished-child-race":
        assert projected["receipt"]["status"] == "supervisor_failed"
        assert projected["receipt"]["reason"] == "tracked_process_handle_unavailable"
        assert projected["receipt"]["accounting_complete"] is False
        assert projected["samples"] == []
        assert projected["kills"] == [100]  # foreign/unavailable child handle is never killed
    else:
        assert projected["receipt"]["status"] == "running"
        assert projected["receipt"]["accounting_complete"] is True
        sample = projected["samples"][0]
        expected_live, expected_working = {
            "nested": (3, 600), "nested-reordered": (3, 600),
            "exited-parent": (1, 300), "after-parent-exit": (0, 0),
            "reused-parent": (0, 0), "reused-child-race": (1, 100)}[scenario]
        assert (sample["live_processes"], sample["aggregate_working_set_bytes"]) == (expected_live, expected_working)
        if scenario == "reused-child-race":
            # The existing creation mismatch continues before registration.
            assert projected["kills"] == []
        if scenario.startswith("nested") or scenario == "exited-parent":
            assert [(row["pid"], row["parent"], row["depth"]) for row in projected["tracked"]] == [
                (100, 0, 0), (200, 100, 1), (300, 200, 2)]
            assert projected["tracked"][2]["parent_creation"] == projected["tracked"][1]["creation"]


FINALIZER_MODEL = r"""
$packetRunnerStdoutPath = 'authored-stdout'; $packetRunnerStderrPath = 'authored-stderr'
$packetRunnerStdoutFile = $null; $packetRunnerStderrFile = $null
$packetRunnerInputPaths = @(); $packetRunnerOutputPaths = @()
$packetRunnerReceiptStream = [pscustomobject]@{}
$packetRunnerReceiptStream | Add-Member ScriptMethod Dispose { }
function Get-PacketFileRecord($Path) {
    return [ordered]@{ path = $Path; bytes = 0; sha256 = 'authored-empty'; error = $null }
}
$packetRunnerStdoutDrain = [pscustomobject]@{ Completion = [pscustomobject]@{ IsCompleted = $true };
    Received = 0; Stored = 0; Discarded = 0; ErrorType = $null }
$packetRunnerStderrDrain = [pscustomobject]@{ Completion = [pscustomobject]@{ IsCompleted = $true };
    Received = 0; Stored = 0; Discarded = 0; ErrorType = $null }
"""


@pytest.mark.parametrize("partial", [False, True], ids=["thrown", "partial-enumeration"])
def test_projected_query_failure_preserves_exception_bound_cleanup_and_drain(tmp_path, partial):
    body = FAKE_PROCESS_MODEL + MODEL_FIXTURES["nested"] + FINALIZER_MODEL + r"""
$script:partial = __PARTIAL__
$script:failure = New-Object Microsoft.Management.Infrastructure.CimException('private/' + ('q' * 50000))
function Get-CimInstance {
    param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++
    if ($script:providerCalls -eq 1) { return $script:rows }
    if ($script:partial) { $script:rows[0] }
    throw $script:failure
}
$good = Get-PacketSample
$script:originalException = $false
Invoke-ActualCatch {
    try { Get-PacketSample | Out-Null } catch {
        $script:originalException = [Object]::ReferenceEquals($_.Exception, $script:failure)
        throw
    }
}
Invoke-ActualFinally
@{ receipt = $packetRunnerReceipt; sample_count = $packetRunnerSamples.Count;
   calls = $script:providerCalls; same_exception = $script:originalException;
   kills = @($script:killLog.ToArray()); writes = $script:writes } | ConvertTo-Json -Depth 20
""".replace("__PARTIAL__", "$true" if partial else "$false")
    result = run_control(tmp_path, body, fake_handles=True)
    assert result["calls"] == 2  # one good query, one failed query; no retry
    assert result["sample_count"] == 1 and result["same_exception"] is True
    assert result["kills"] == [300, 200, 100]
    receipt = result["receipt"]
    assert receipt["status"] == "supervisor_failed" and receipt["reason"] == "process_snapshot_failed"
    assert receipt["accounting_complete"] is False
    assert receipt["cleanup_complete"] is receipt["drain_complete"] is True
    assert receipt["root_exit_observed"] is True and receipt["root_exit_code"] == -1
    assert len(receipt["logs"]) == 2 and all(log["completed"] for log in receipt["logs"])
    assert receipt["discovery_failure"]["query_telemetry"]["outcome"] == "failed"
    assert "private" not in json.dumps(result)
    assert len(json.dumps(receipt["discovery_failure"])) < 1024


@pytest.mark.parametrize("started, finished, state, seconds", [
    ("[double]0", "[double]0", "measured", 0),
    ("[double]5", "[double]7.5", "measured", 2.5),
    ("[double]0", "[double]100000", "capped", 86400),
    ("[double]-1", "[double]3", "unavailable", None),
    ("[double]3", "[double]2", "unavailable", None),
    ("[double]::NaN", "[double]3", "unavailable", None),
    ("[double]0", "[double]::PositiveInfinity", "unavailable", None),
    ("$null", "[double]0", "unavailable", None),
    ("'private/path'", "[double]0", "unavailable", None),
], ids=["zero", "measured", "cap", "negative", "clock-reversed", "nan", "inf", "missing", "string"])
def test_snapshot_timing_is_finite_bounded_and_redacted(tmp_path, started, finished, state, seconds):
    record = run_control(tmp_path, """
$interval = Get-PacketBoundedInterval (__START__) (__END__)
@{ interval = $interval } | ConvertTo-Json
""".replace("__START__", started).replace("__END__", finished))
    assert record == {"interval": {"seconds": seconds, "state": state}}
    assert "private" not in json.dumps(record)


def test_snapshot_duration_and_start_gap_use_same_original_query(tmp_path):
    record = run_control(tmp_path, r"""
$script:ticks = New-Object 'Collections.Generic.Queue[double]'
foreach ($tick in @(10.0, 10.25, 12.0, 12.5)) { $script:ticks.Enqueue($tick) }
$elapsed = [pscustomobject]@{}
$elapsed | Add-Member ScriptProperty TotalSeconds { $script:ticks.Dequeue() }
$packetRunnerClock = [pscustomobject]@{ Elapsed = $elapsed }
$script:row = [pscustomobject]@{ ProcessId = 1; ParentProcessId = 0; CreationDate = [DateTime]'2020-01-01' }
function Get-CimInstance {
    param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++; return $script:row
}
$first = @(Get-PacketProcessSnapshot); $a = $packetRunnerReceipt.process_snapshot_telemetry
$second = @(Get-PacketProcessSnapshot); $b = $packetRunnerReceipt.process_snapshot_telemetry
@{ calls = $script:providerCalls; first = $a; second = $b;
   same_rows = ([Object]::ReferenceEquals($first[0], $script:row) -and
                [Object]::ReferenceEquals($second[0], $script:row)); remaining_ticks = $script:ticks.Count } | ConvertTo-Json
""")
    assert record["calls"] == 2 and record["same_rows"] is True and record["remaining_ticks"] == 0
    assert record["first"] == {"outcome": "returned", "query_duration_seconds": 0.25,
                               "query_duration_state": "measured", "query_start_gap_seconds": None,
                               "query_start_gap_state": "unavailable"}
    assert record["second"] == {"outcome": "returned", "query_duration_seconds": 0.5,
                                "query_duration_state": "measured", "query_start_gap_seconds": 2,
                                "query_start_gap_state": "measured"}


@pytest.mark.parametrize("failed", [False, True], ids=["success", "provider-failure"])
def test_telemetry_failure_never_changes_query_or_original_exception(tmp_path, failed):
    record = run_control(tmp_path, r"""
$script:failed = __FAILED__
$script:failure = New-Object InvalidOperationException('authored provider failure')
$script:row = [pscustomobject]@{ ProcessId = 1; ParentProcessId = 0; CreationDate = [DateTime]'2020-01-01' }
$packetRunnerReceipt.process_snapshot_telemetry = @{ outcome = 'returned'; query_duration_seconds = 99 }
function Update-PacketSnapshotTelemetry { throw 'private_telemetry_error' }
function Get-CimInstance {
    param($ClassName, $Property, $OperationTimeoutSec, $ErrorAction)
    $script:providerCalls++
    if ($script:failed) { throw $script:failure }
    return $script:row
}
$identical = $false; $sameException = $false
try { $rows = @(Get-PacketProcessSnapshot); $identical = [Object]::ReferenceEquals($rows[0], $script:row) }
catch { $sameException = [Object]::ReferenceEquals($_.Exception, $script:failure) }
@{ calls = $script:providerCalls; identical = $identical; same_exception = $sameException;
   telemetry = $packetRunnerReceipt.process_snapshot_telemetry } | ConvertTo-Json
""".replace("__FAILED__", "$true" if failed else "$false"))
    assert record == {"calls": 1, "identical": not failed, "same_exception": failed, "telemetry": None}


def test_invalid_telemetry_outcome_redacts_without_serializing_payload(tmp_path):
    record = run_control(tmp_path, r"""
Update-PacketSnapshotTelemetry ([double]0) ([double]1) ('private/' + ('x' * 100000))
@{ telemetry = $packetRunnerReceipt.process_snapshot_telemetry } | ConvertTo-Json
""")
    assert record == {"telemetry": None}
