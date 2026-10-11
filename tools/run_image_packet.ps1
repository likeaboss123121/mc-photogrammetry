# Windows PowerShell 5.1. Sampled process supervision, not OS containment.
#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PacketPath,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedPacketSha256,
    [Parameter(Mandatory = $true)][string]$MonitorReceiptPath,
    [switch]$SelfTest,
    [switch]$InjectSupervisorException,
    [switch]$SelfTestHoldDrains
)
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$packetRunnerRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$packetRunnerRuns = [IO.Path]::GetFullPath((Join-Path $packetRunnerRoot 'agent swarm instantiation template\runs'))
$packetRunnerLogCap = 4 * 1024 * 1024
$packetRunnerDrainSeconds = 10
$packetRunnerCleanupSeconds = 5

function Get-PacketField($Record, [string]$Name, $Default = $null) {
    if ($null -ne $Record -and $null -ne $Record.PSObject.Properties[$Name]) { return $Record.$Name }
    return $Default
}
function Assert-Packet([bool]$Condition, [string]$Reason) {
    if (-not $Condition) { throw $Reason }
}
function Resolve-PacketPath([string]$Value, [switch]$RepoOnly) {
    Assert-Packet (-not [string]::IsNullOrWhiteSpace($Value)) 'empty_path'
    $resolved = if ([IO.Path]::IsPathRooted($Value)) {
        [IO.Path]::GetFullPath($Value)
    } else { [IO.Path]::GetFullPath((Join-Path $packetRunnerRoot $Value)) }
    if ($RepoOnly) {
        Assert-Packet ($resolved.StartsWith($packetRunnerRoot + '\', [StringComparison]::OrdinalIgnoreCase)) 'path_outside_repository'
    }
    return $resolved
}
function Get-PacketHash([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}
function Get-PacketFileRecord([string]$Path) {
    $record = [ordered]@{ path = $Path; exists = $false; bytes = $null; sha256 = $null; error = $null }
    try {
        if ([IO.File]::Exists($Path)) {
            $record.exists = $true
            $record.bytes = (Get-Item -LiteralPath $Path).Length
            $record.sha256 = Get-PacketHash $Path
        }
    } catch { $record.error = $_.Exception.GetType().FullName }
    return $record
}
function ConvertTo-PacketArgument([string]$Value) {
    # Windows CRT quoting: double backslashes before quotes and the final quote.
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (2 * $slashes + 1)))
            [void]$builder.Append('"')
        } else {
            [void]$builder.Append(('\' * $slashes))
            [void]$builder.Append($character)
        }
        $slashes = 0
    }
    [void]$builder.Append(('\' * (2 * $slashes)))
    [void]$builder.Append('"')
    return $builder.ToString()
}
function Get-PacketCreation([datetime]$Value) {
    # CIM's creation date has microsecond precision; normalize both APIs equally.
    return $Value.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ')
}
function Write-PacketReceipt {
    $packetRunnerReceipt.updated_utc = [DateTime]::UtcNow.ToString('o')
    $packetRunnerReceipt.processes = @($packetRunnerTracked.Values | ForEach-Object {
        [ordered]@{ pid = $_.Pid; creation_utc = $_.Creation; parent_pid = $_.ParentPid;
            parent_creation_utc = $_.ParentCreation; depth = $_.Depth;
            first_seen_utc = $_.FirstSeen; last_seen_utc = $_.LastSeen;
            exit_observed = $_.ExitObserved; exit_code = $_.ExitCode;
            exit_observed_utc = $_.ExitUtc; exit_time_utc = $_.ExitTimeUtc;
            termination_attempted = $_.TerminationAttempted }
    })
    $packetRunnerReceipt.samples = @($packetRunnerSamples.ToArray())
    $serialized = $packetRunnerReceipt | ConvertTo-Json -Depth 20
    $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($serialized + "`n")
    $packetRunnerReceiptStream.Position = 0
    $packetRunnerReceiptStream.SetLength(0)
    $packetRunnerReceiptStream.Write($bytes, 0, $bytes.Length)
    $packetRunnerReceiptStream.Flush($true)
}
function Register-PacketProcess([int]$ProcessId, [string]$Creation, [int]$ParentId,
                                [string]$ParentCreation, [int]$Depth, $Handle) {
    $key = [string]$ProcessId + '|' + $Creation
    if (-not $packetRunnerTracked.Contains($key)) {
        $packetRunnerTracked[$key] = [pscustomobject]@{
            Pid = $ProcessId; Creation = $Creation; ParentPid = $ParentId;
            ParentCreation = $ParentCreation; Depth = $Depth; Handle = $Handle;
            FirstSeen = [DateTime]::UtcNow.ToString('o'); LastSeen = $null;
            ExitObserved = $false; ExitCode = $null; ExitUtc = $null;
            ExitTimeUtc = $null;
            TerminationAttempted = $false
        }
    }
}
function Update-PacketExit($Tracked) {
    if ($null -ne $Tracked.Handle -and -not $Tracked.ExitObserved) {
        try {
            if ($Tracked.Handle.HasExited) {
                $Tracked.ExitObserved = $true
                $Tracked.ExitUtc = [DateTime]::UtcNow.ToString('o')
                try { $Tracked.ExitCode = $Tracked.Handle.ExitCode } catch { $Tracked.ExitCode = $null }
                try { $Tracked.ExitTimeUtc = Get-PacketCreation $Tracked.Handle.ExitTime } catch { $Tracked.ExitTimeUtc = $null }
            }
        } catch { $packetRunnerReceipt.accounting_complete = $false }
    }
}
function Get-PacketDiagnosticCode($Exception, [string]$Name) {
    # Only bounded numeric codes, never message/ErrorData/stack/provider payload.
    try {
        $value = Get-PacketField $Exception $Name
        if ($value -is [Enum] -or $value -is [sbyte] -or $value -is [byte] -or
            $value -is [int16] -or $value -is [uint16] -or $value -is [int32] -or
            $value -is [uint32] -or $value -is [int64] -or $value -is [uint64]) {
            $code = [long]$value
            if ($code -ge -2147483648 -and $code -le 4294967295) { return $code }
        }
    } catch { } # Unavailable/throwing metadata cannot replace the original error.
    return $null
}
function Get-PacketMonitorSeconds {
    try {
        $value = $packetRunnerClock.Elapsed.TotalSeconds
        if ($value -is [double] -and -not [double]::IsNaN($value) -and
            -not [double]::IsInfinity($value) -and $value -ge 0) { return $value }
    } catch { } # Timing observation must not affect query/result/exception.
    return $null
}
function Get-PacketBoundedInterval($Started, $Finished) {
    $seconds = $null; $state = 'unavailable'
    try {
        if ($Started -is [double] -and $Finished -is [double] -and
            -not [double]::IsNaN($Started) -and -not [double]::IsInfinity($Started) -and
            -not [double]::IsNaN($Finished) -and -not [double]::IsInfinity($Finished) -and
            $Started -ge 0 -and $Finished -ge $Started) {
            $delta = $Finished - $Started
            if (-not [double]::IsInfinity($delta)) {
                $seconds = [Math]::Min($delta, 86400.0)
                $state = if ($delta -gt 86400.0) { 'capped' } else { 'measured' }
            }
        }
    } catch { } # Only finite bounded seconds/state, never clock/error payload.
    return [ordered]@{ seconds = $seconds; state = $state }
}
function Update-PacketSnapshotTelemetry($Started, $Finished, [string]$Outcome) {
    try {
        if ($Outcome -notin @('returned', 'failed')) { throw 'invalid_timing_outcome' }
        $duration = Get-PacketBoundedInterval $Started $Finished
        # Attempt-start gap, not evidence of uninterrupted descendant coverage.
        $gap = Get-PacketBoundedInterval $script:packetRunnerPreviousSnapshotStartSeconds $Started
        $packetRunnerReceipt['process_snapshot_telemetry'] = [ordered]@{
            outcome = $Outcome; query_duration_seconds = $duration.seconds;
            query_duration_state = $duration.state;
            query_start_gap_seconds = $gap.seconds; query_start_gap_state = $gap.state
        }
        $script:packetRunnerPreviousSnapshotStartSeconds = $Started
    } catch {
        try {
            $packetRunnerReceipt['process_snapshot_telemetry'] = $null
            $script:packetRunnerPreviousSnapshotStartSeconds = $null
        } catch { } # Telemetry failures never replace the original query outcome.
    }
}
function Get-PacketProcessSnapshot {
    $started = $null; $failed = $false
    try { $started = Get-PacketMonitorSeconds } catch { }
    try {
        # All processes, only consumed properties; same deadline/objects/attempt.
        Get-CimInstance -ClassName Win32_Process -Property ProcessId,ParentProcessId,CreationDate -OperationTimeoutSec 1 -ErrorAction Stop
    } catch {
        $failed = $true
        $failure = $_.Exception
        $failureType = $failure.GetType().FullName
        if ($failureType.Length -gt 160 -or $failureType -notmatch '^[A-Za-z0-9_.+]+$') {
            $failureType = $null
        }
        $packetRunnerReceipt.discovery_failure = [ordered]@{
            stage = 'process_snapshot'; reason = 'process_snapshot_failed';
            exception_type = $failureType;
            hresult = Get-PacketDiagnosticCode $failure 'HResult';
            native_error_code = Get-PacketDiagnosticCode $failure 'NativeErrorCode';
            status_code = Get-PacketDiagnosticCode $failure 'StatusCode';
            completed_samples = $packetRunnerSamples.Count;
            elapsed_seconds = $packetRunnerClock.Elapsed.TotalSeconds
        }
        throw # Fail closed; no empty/cached snapshot, retry, or exception substitution.
    } finally {
        try {
            $finished = Get-PacketMonitorSeconds
            $outcome = if ($failed) { 'failed' } else { 'returned' }
            Update-PacketSnapshotTelemetry $started $finished $outcome
            if ($failed) {
                $packetRunnerReceipt.discovery_failure['query_telemetry'] = $packetRunnerReceipt['process_snapshot_telemetry']
            }
        } catch {
            try {
                $packetRunnerReceipt['process_snapshot_telemetry'] = $null
                if ($failed) { $packetRunnerReceipt.discovery_failure['query_telemetry'] = $null }
            } catch { }
        } # Best-effort scalars; preserve exact returned rows or exception.
    }
}
function Get-PacketSample {
    $now = [DateTime]::UtcNow.ToString('o')
    $snapshot = @(Get-PacketProcessSnapshot)
    $byPid = @{}
    foreach ($entry in $snapshot) {
        if ($null -ne $entry.CreationDate) {
            $byPid[[int]$entry.ProcessId] = [pscustomobject]@{
                Pid = [int]$entry.ProcessId; ParentPid = [int]$entry.ParentProcessId;
                Creation = Get-PacketCreation ([datetime]$entry.CreationDate)
            }
        }
    }
    foreach ($tracked in @($packetRunnerTracked.Values)) { Update-PacketExit $tracked }
    # Discover the transitive descendant closure in this single sampled snapshot.
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($entry in $byPid.Values) {
            $entryKey = [string]$entry.Pid + '|' + $entry.Creation
            if ($packetRunnerTracked.Contains($entryKey)) { continue }
            foreach ($parent in @($packetRunnerTracked.Values)) {
                if ($entry.ParentPid -ne $parent.Pid -or $entry.Creation -lt $parent.Creation) { continue }
                # A reused live parent PID cannot be used to join a descendant.
                if ($byPid.ContainsKey($parent.Pid)) {
                    if ($byPid[$parent.Pid].Creation -ne $parent.Creation) { continue }
                } elseif ($parent.ExitObserved) {
                    # Use the actual bound parent's exit time, not the later
                    # observation time; a reused parent PID could live in between.
                    if ($null -eq $parent.ExitTimeUtc -or $entry.Creation -gt $parent.ExitTimeUtc) { continue }
                } else { continue }
                $handle = $null
                try {
                    $handle = [Diagnostics.Process]::GetProcessById($entry.Pid)
                    [void]$handle.Handle
                    if ((Get-PacketCreation $handle.StartTime) -ne $entry.Creation) {
                        $handle.Dispose(); $handle = $null; continue
                    }
                } catch {
                    if ($null -ne $handle) { $handle.Dispose() }
                    $handle = $null
                }
                Register-PacketProcess $entry.Pid $entry.Creation $parent.Pid $parent.Creation ($parent.Depth + 1) $handle
                $changed = $true
                break
            }
        }
    }
    $rows = New-Object 'Collections.Generic.List[object]'
    $aggregate = [long]0
    $live = 0
    foreach ($tracked in @($packetRunnerTracked.Values)) {
        $alive = $byPid.ContainsKey($tracked.Pid) -and $byPid[$tracked.Pid].Creation -eq $tracked.Creation
        $working = $null
        if ($alive) {
            $live++
            $tracked.LastSeen = $now
            if ($null -eq $tracked.Handle) {
                throw 'tracked_process_handle_unavailable'
            }
            try {
                $tracked.Handle.Refresh()
                if (-not $tracked.Handle.HasExited) {
                    Assert-Packet ((Get-PacketCreation $tracked.Handle.StartTime) -eq $tracked.Creation) 'sample_creation_mismatch'
                    $working = [long]$tracked.Handle.WorkingSet64
                    $aggregate += $working
                } else { $alive = $false; $live--; Update-PacketExit $tracked }
            } catch {
                if ($tracked.Handle.HasExited) { $alive = $false; $live--; Update-PacketExit $tracked }
                else { throw }
            }
        } else {
            Update-PacketExit $tracked
            if (-not $tracked.ExitObserved) {
                $packetRunnerReceipt.accounting_complete = $false
            }
        }
        $rows.Add([ordered]@{ pid = $tracked.Pid; creation_utc = $tracked.Creation;
            parent_pid = $tracked.ParentPid; parent_creation_utc = $tracked.ParentCreation;
            alive = $alive; working_set_bytes = $working; exit_observed = $tracked.ExitObserved;
            exit_code = $tracked.ExitCode })
    }
    $sample = [ordered]@{ timestamp_utc = $now; elapsed_seconds = $packetRunnerClock.Elapsed.TotalSeconds;
        aggregate_working_set_bytes = $aggregate; live_processes = $live; processes = @($rows.ToArray());
        discovery_telemetry = $packetRunnerReceipt['process_snapshot_telemetry'] }
    $packetRunnerSamples.Add($sample)
    if ($aggregate -gt $packetRunnerReceipt.peak_aggregate_working_set_bytes) {
        $packetRunnerReceipt.peak_aggregate_working_set_bytes = $aggregate
    }
    Write-PacketReceipt
    return $sample
}
function Stop-PacketTree {
    $cleanupClock = [Diagnostics.Stopwatch]::StartNew()
    $cleanupRootUnbound = $false
    $packetRunnerReceipt.cleanup_attempted = $true
    if ($packetRunnerTracked.Count -eq 0 -and $null -ne $packetRunnerProcess -and $packetRunnerReceipt.launched) {
        try {
            $rootCreation = Get-PacketCreation $packetRunnerProcess.StartTime
            $packetRunnerReceipt.root_pid = $packetRunnerProcess.Id
            $packetRunnerReceipt.root_creation_utc = $rootCreation
            Register-PacketProcess $packetRunnerProcess.Id $rootCreation 0 '' 0 $packetRunnerProcess
        } catch {
            $cleanupRootUnbound = $true
            $packetRunnerReceipt.accounting_complete = $false
            $packetRunnerReceipt.cleanup_errors += 'root_identity_unavailable'
        }
    }
    $remaining = $packetRunnerTracked.Count
    do {
        # Cleanup uses already bound handles; no blocking CIM discovery or receipt
        # hashing/write is placed inside its five-second attempt deadline.
        foreach ($tracked in @($packetRunnerTracked.Values | Sort-Object Depth -Descending)) {
            if ($cleanupClock.Elapsed.TotalSeconds -ge $packetRunnerCleanupSeconds) { break }
            $candidate = $null
            try {
                $candidate = [Diagnostics.Process]::GetProcessById($tracked.Pid)
                # Force an actual process handle before the identity check and kill.
                [void]$candidate.Handle
                if ((Get-PacketCreation $candidate.StartTime) -eq $tracked.Creation -and -not $candidate.HasExited) {
                    $tracked.TerminationAttempted = $true
                    $candidate.Kill()
                }
            } catch [ArgumentException] { } catch [InvalidOperationException] { } catch {
                $packetRunnerReceipt.cleanup_errors += $_.Exception.GetType().FullName
                $packetRunnerReceipt.accounting_complete = $false
            } finally { if ($null -ne $candidate) { $candidate.Dispose() } }
        }
        $remaining = $packetRunnerTracked.Count
        foreach ($tracked in @($packetRunnerTracked.Values)) {
            if ($cleanupClock.Elapsed.TotalSeconds -ge $packetRunnerCleanupSeconds) { break }
            Update-PacketExit $tracked
            if ($tracked.ExitObserved) { $remaining-- }
        }
        if ($remaining -eq 0) { break }
        $cleanupSleep = [int][Math]::Min(100, [Math]::Max(0, 1000 * ($packetRunnerCleanupSeconds - $cleanupClock.Elapsed.TotalSeconds)))
        if ($cleanupSleep -gt 0) { Start-Sleep -Milliseconds $cleanupSleep }
    } while ($cleanupClock.Elapsed.TotalSeconds -lt $packetRunnerCleanupSeconds)
    $packetRunnerReceipt.cleanup_elapsed_seconds = $cleanupClock.Elapsed.TotalSeconds
    $packetRunnerReceipt.cleanup_overshoot_seconds = [Math]::Max(0, $cleanupClock.Elapsed.TotalSeconds - $packetRunnerCleanupSeconds)
    $packetRunnerReceipt.cleanup_complete = ($remaining -eq 0 -and -not $cleanupRootUnbound)
    if (-not $packetRunnerReceipt.cleanup_complete) { $packetRunnerReceipt.accounting_complete = $false }
}

$packetRunnerReceiptFull = Resolve-PacketPath $MonitorReceiptPath -RepoOnly
Assert-Packet ($packetRunnerReceiptFull.StartsWith($packetRunnerRuns + '\', [StringComparison]::OrdinalIgnoreCase)) 'receipt_outside_runs'
# Reserve and flush the receipt before reading any packet or attempting launch.
$packetRunnerReceiptStream = [IO.File]::Open($packetRunnerReceiptFull, [IO.FileMode]::CreateNew,
    [IO.FileAccess]::ReadWrite, [IO.FileShare]::Read)
$packetRunnerReceipt = [ordered]@{
    schema = 'mc-photogrammetry-windows-packet-monitor-v1'; status = 'reserved'; reason = $null;
    packet_path = $PacketPath; expected_packet_sha256 = $ExpectedPacketSha256.ToUpperInvariant();
    actual_packet_sha256 = $null; created_utc = [DateTime]::UtcNow.ToString('o'); updated_utc = $null;
    self_test = [bool]$SelfTest; launch_attempts = 0; launched = $false; root_pid = $null;
    root_creation_utc = $null; root_exit_observed = $false; root_exit_code = $null;
    termination_reason = $null; accounting_complete = $true;
    failure_reason_before_drain = $null;
    coverage = 'sampled descendant discovery once per second; short-lived descendants may be missed; not hard containment';
    interpreter = $null; interpreter_sha256 = $null; arguments = @(); working_directory = $null;
    limits = $null; pre_inputs = @(); post_inputs = @(); outputs = @(); logs = @(); samples = @(); processes = @();
    peak_aggregate_working_set_bytes = [long]0; monitor_elapsed_seconds = 0.0;
    limit_observed_elapsed_seconds = $null; memory_overshoot_bytes = [long]0; wall_overshoot_seconds = 0.0;
    drain_deadline_seconds = $packetRunnerDrainSeconds; drain_elapsed_seconds = 0.0; drain_complete = $false;
    cleanup_deadline_seconds = $packetRunnerCleanupSeconds; cleanup_attempted = $false;
    cleanup_elapsed_seconds = 0.0; cleanup_complete = $null; cleanup_errors = @(); error_type = $null
    cleanup_overshoot_seconds = 0.0; drain_overshoot_seconds = 0.0; discovery_failure = $null
    process_snapshot_telemetry = $null
}
$packetRunnerTracked = [ordered]@{}
$packetRunnerSamples = New-Object 'Collections.Generic.List[object]'
$packetRunnerClock = [Diagnostics.Stopwatch]::StartNew()
$script:packetRunnerPreviousSnapshotStartSeconds = $null
$packetRunnerProcess = $null
$packetRunnerStdoutFile = $null; $packetRunnerStderrFile = $null
$packetRunnerStdoutDrain = $null; $packetRunnerStderrDrain = $null
$packetRunnerInputPaths = New-Object 'Collections.Generic.List[string]'
$packetRunnerOutputPaths = New-Object 'Collections.Generic.List[string]'
$packetRunnerStdoutPath = $null; $packetRunnerStderrPath = $null
$packetRunnerRootTracked = $null
try {
    Write-PacketReceipt
    Assert-Packet ((-not $InjectSupervisorException -and -not $SelfTestHoldDrains) -or $SelfTest) 'self_test_injection_requires_self_test'
    $packetRunnerPacketFull = Resolve-PacketPath $PacketPath -RepoOnly
    Assert-Packet ([IO.Path]::GetExtension($packetRunnerPacketFull) -ieq '.json') 'packet_json_path_required'
    if ($SelfTest) {
        Assert-Packet ($packetRunnerPacketFull.StartsWith((Join-Path $packetRunnerRuns 'SOL-WINDOWS-BOUNDED-PACKET-RUNNER-01.smoke-'), [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($packetRunnerPacketFull) -ceq 'synthetic-packet.json') 'synthetic_packet_path_required'
    }
    $packetRunnerReceipt.actual_packet_sha256 = Get-PacketHash $packetRunnerPacketFull
    Assert-Packet ($packetRunnerReceipt.actual_packet_sha256 -eq $ExpectedPacketSha256) 'packet_hash_mismatch'
    $packetRunnerPacket = Get-Content -LiteralPath $packetRunnerPacketFull -Raw -Encoding UTF8 | ConvertFrom-Json
    $packetRunnerSchema = Get-PacketField $packetRunnerPacket 'schema'
    $packetRunnerInputs = @(Get-PacketField $packetRunnerPacket 'inputs')
    if ($SelfTest) {
        Assert-Packet ($packetRunnerSchema -ceq 'mc-photogrammetry-synthetic-run-packet-v1') 'synthetic_schema_required'
        Assert-Packet ((Get-PacketField $packetRunnerPacket 'synthetic_only') -is [bool] -and $packetRunnerPacket.synthetic_only) 'synthetic_marker_required'
        Assert-Packet ($packetRunnerInputs.Count -eq 0) 'self_test_inputs_forbidden'
    } else {
        Assert-Packet ($packetRunnerSchema -ceq 'mc-photogrammetry-image-run-packet-v1') 'packet_schema_invalid'
        Assert-Packet ($packetRunnerInputs.Count -gt 0) 'input_inventory_missing'
    }
    Assert-Packet ((Get-PacketField $packetRunnerPacket 'lead_authorized') -is [bool] -and $packetRunnerPacket.lead_authorized) 'lead_authorization_missing'
    $packetRunnerCwd = Resolve-PacketPath (Get-PacketField $packetRunnerPacket 'working_directory') -RepoOnly:$false
    Assert-Packet ($packetRunnerCwd -eq $packetRunnerRoot -and [IO.Directory]::Exists($packetRunnerCwd)) 'working_directory_mismatch'
    $packetRunnerInterpreter = Resolve-PacketPath (Get-PacketField $packetRunnerPacket 'interpreter')
    Assert-Packet ([IO.Path]::GetFileName($packetRunnerInterpreter) -ieq 'python.exe') 'direct_python_interpreter_required'
    $packetRunnerInterpreterHash = Get-PacketHash $packetRunnerInterpreter
    Assert-Packet ($packetRunnerInterpreterHash -eq (Get-PacketField $packetRunnerPacket 'interpreter_sha256')) 'interpreter_hash_mismatch'
    $packetRunnerArguments = @(Get-PacketField $packetRunnerPacket 'arguments')
    Assert-Packet ($packetRunnerArguments.Count -gt 0) 'arguments_missing'
    foreach ($argument in $packetRunnerArguments) {
        Assert-Packet ($argument -is [string] -and $argument.IndexOf([char]0) -lt 0) 'argument_invalid'
    }
    $packetRunnerClosurePath = Resolve-PacketPath (Get-PacketField $packetRunnerPacket 'source_closure_path') -RepoOnly
    if ($SelfTest) {
        Assert-Packet ($packetRunnerClosurePath.StartsWith((Join-Path $packetRunnerRuns 'SOL-WINDOWS-BOUNDED-PACKET-RUNNER-01.smoke-'), [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($packetRunnerClosurePath) -ceq 'synthetic-source-closure.json') 'synthetic_closure_path_required'
    }
    Assert-Packet ((Get-PacketHash $packetRunnerClosurePath) -eq (Get-PacketField $packetRunnerPacket 'source_closure_sha256')) 'source_closure_hash_mismatch'
    $packetRunnerClosure = Get-Content -LiteralPath $packetRunnerClosurePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $packetRunnerPins = @(Get-PacketField $packetRunnerClosure 'repo_pins')
    Assert-Packet ($packetRunnerPins.Count -gt 0 -and $packetRunnerPins.Count -eq (Get-PacketField $packetRunnerClosure 'repo_node_count')) 'source_closure_count_mismatch'
    $packetRunnerPinnedPaths = @{}
    $syntheticPinPaths = @(
        'tools/run_image_packet.ps1',
        'agent swarm instantiation template/runs/SOL-WINDOWS-BOUNDED-PACKET-RUNNER-01.synthetic-worker.py',
        'agent swarm instantiation template/runs/SOL-WINDOWS-BOUNDED-PACKET-RUNNER-01.synthetic-smoke.ps1'
    )
    if ($SelfTest) { Assert-Packet ($packetRunnerPins.Count -eq $syntheticPinPaths.Count) 'synthetic_pin_count_invalid' }
    foreach ($pin in $packetRunnerPins) {
        $pinnedPath = Resolve-PacketPath (Get-PacketField $pin 'path') -RepoOnly
        if ($SelfTest) {
            Assert-Packet ((Get-PacketField $pin 'path') -cin $syntheticPinPaths) 'synthetic_pin_path_forbidden'
        }
        Assert-Packet (-not $packetRunnerPinnedPaths.ContainsKey($pinnedPath)) 'duplicate_source_pin'
        $packetRunnerPinnedPaths[$pinnedPath] = $true
        Assert-Packet ((Get-PacketHash $pinnedPath) -eq (Get-PacketField $pin 'sha256')) 'source_pin_hash_mismatch'
        Assert-Packet ((Get-Item -LiteralPath $pinnedPath).Length -eq (Get-PacketField $pin 'bytes')) 'source_pin_length_mismatch'
    }
    if ($SelfTest) {
        $fixtureRelative = 'agent swarm instantiation template/runs/SOL-WINDOWS-BOUNDED-PACKET-RUNNER-01.synthetic-worker.py'
        Assert-Packet ($packetRunnerArguments.Count -ge 3 -and $packetRunnerArguments[0] -ceq '-B' -and
            $packetRunnerArguments[1] -ceq $fixtureRelative) 'synthetic_worker_required'
        Assert-Packet ($packetRunnerPinnedPaths.ContainsKey((Resolve-PacketPath $fixtureRelative -RepoOnly))) 'synthetic_worker_pin_required'
        Assert-Packet ($packetRunnerArguments[2] -cin @('echo', 'timeout', 'memory')) 'synthetic_mode_invalid'
        foreach ($argument in $packetRunnerArguments) {
            Assert-Packet ($argument -notmatch '(?i)\.(png|jpe?g|bmp|gif|webp|tiff?|exr)(?:$|[\s"''])') 'self_test_image_argument_forbidden'
        }
    }
    $packetRunnerLimits = Get-PacketField $packetRunnerPacket 'limits'
    $sampling = Get-PacketField $packetRunnerLimits 'sampling_seconds'
    $wall = Get-PacketField $packetRunnerLimits 'wall_timeout_seconds'
    $memory = Get-PacketField $packetRunnerLimits 'working_set_threshold_bytes'
    Assert-Packet (($sampling -is [int] -or $sampling -is [long]) -and $sampling -eq 1 -and
        ($wall -is [int] -or $wall -is [long]) -and $wall -ge 1 -and $wall -le 86400 -and
        ($memory -is [int] -or $memory -is [long]) -and $memory -gt 0) 'limits_invalid'
    $packetRunnerReceipt.limits = [ordered]@{ sampling_seconds = 1; wall_timeout_seconds = $wall;
        working_set_threshold_bytes = $memory; log_cap_bytes_each = $packetRunnerLogCap }
    $packetRunnerReceipt.interpreter = $packetRunnerInterpreter
    $packetRunnerReceipt.interpreter_sha256 = $packetRunnerInterpreterHash
    $packetRunnerReceipt.arguments = $packetRunnerArguments
    $packetRunnerReceipt.working_directory = $packetRunnerCwd
    $packetRunnerProtected = @{}
    foreach ($path in @($packetRunnerPacketFull, $packetRunnerClosurePath, $packetRunnerInterpreter, $packetRunnerReceiptFull) + @($packetRunnerPinnedPaths.Keys)) {
        $packetRunnerProtected[$path] = $true
    }
    foreach ($inputRecord in $packetRunnerInputs) {
        $inputPath = Resolve-PacketPath (Get-PacketField $inputRecord 'path') -RepoOnly
        $declaredAbsolute = Get-PacketField $inputRecord 'absolute_path'
        if ($null -ne $declaredAbsolute) {
            Assert-Packet ((Resolve-PacketPath $declaredAbsolute) -eq $inputPath) 'input_absolute_path_mismatch'
        }
        Assert-Packet (-not $packetRunnerProtected.ContainsKey($inputPath)) 'duplicate_or_protected_input'
        $packetRunnerProtected[$inputPath] = $true
        $packetRunnerInputPaths.Add($inputPath)
        $actualInput = Get-PacketFileRecord $inputPath
        $packetRunnerReceipt.pre_inputs += $actualInput
        Assert-Packet ($actualInput.exists -and $null -eq $actualInput.error -and $actualInput.bytes -eq (Get-PacketField $inputRecord 'bytes') -and
            $actualInput.sha256 -eq (Get-PacketField $inputRecord 'sha256')) 'input_pin_mismatch'
    }
    $packetRunnerStdoutPath = Resolve-PacketPath (Get-PacketField $packetRunnerPacket 'stdout_log') -RepoOnly
    $packetRunnerStderrPath = Resolve-PacketPath (Get-PacketField $packetRunnerPacket 'stderr_log') -RepoOnly
    if ($SelfTest) {
        Assert-Packet ($null -eq (Get-PacketField $packetRunnerPacket 'project_output') -and
            $null -eq (Get-PacketField $packetRunnerPacket 'json_output') -and
            @(Get-PacketField $packetRunnerPacket 'output_paths' @()).Count -eq 0) 'synthetic_outputs_forbidden'
        Assert-Packet ([IO.Path]::GetDirectoryName($packetRunnerStdoutPath) -eq [IO.Path]::GetDirectoryName($packetRunnerPacketFull) -and
            [IO.Path]::GetDirectoryName($packetRunnerStderrPath) -eq [IO.Path]::GetDirectoryName($packetRunnerPacketFull) -and
            [IO.Path]::GetFileName($packetRunnerStdoutPath) -ceq 'stdout.log' -and
            [IO.Path]::GetFileName($packetRunnerStderrPath) -ceq 'stderr.log') 'synthetic_log_paths_required'
    }
    $declaredOutputs = New-Object 'Collections.Generic.List[string]'
    foreach ($field in @('project_output', 'json_output')) {
        $value = Get-PacketField $packetRunnerPacket $field
        if ($null -ne $value) { $declaredOutputs.Add((Resolve-PacketPath $value -RepoOnly)) }
        elseif (-not $SelfTest) { throw 'declared_output_missing' }
    }
    foreach ($value in @(Get-PacketField $packetRunnerPacket 'output_paths' @())) {
        $declaredOutputs.Add((Resolve-PacketPath $value -RepoOnly))
    }
    $declaredOutputs.Add($packetRunnerStdoutPath); $declaredOutputs.Add($packetRunnerStderrPath)
    foreach ($path in $declaredOutputs) {
        Assert-Packet (-not $packetRunnerProtected.ContainsKey($path)) 'duplicate_or_protected_output'
        $packetRunnerProtected[$path] = $true
        $packetRunnerOutputPaths.Add($path)
        Assert-Packet (-not (Test-Path -LiteralPath $path)) 'output_already_exists'
        Assert-Packet ([IO.Directory]::Exists([IO.Path]::GetDirectoryName($path))) 'output_parent_missing'
    }
    $packetRunnerReceipt.status = 'preflight_passed'
    Write-PacketReceipt
    if (-not ('PacketRunnerV1.BinaryDrain' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
namespace PacketRunnerV1 {
    public sealed class BinaryDrain {
        private readonly Stream input;
        private readonly FileStream output;
        private readonly long cap;
        private readonly int selfTestDelay;
        private long received, stored;
        public readonly Task Completion;
        public string ErrorType;
        public BinaryDrain(Stream input, FileStream output, long cap, int selfTestDelay) {
            this.input = input; this.output = output; this.cap = cap; this.selfTestDelay = selfTestDelay;
            Completion = Task.Run((Action)Pump);
        }
        public long Received { get { return Interlocked.Read(ref received); } }
        public long Stored { get { return Interlocked.Read(ref stored); } }
        public long Discarded { get { return Received - Stored; } }
        private void Pump() {
            try {
                byte[] buffer = new byte[65536];
                int count;
                while ((count = input.Read(buffer, 0, buffer.Length)) > 0) {
                    Interlocked.Add(ref received, count);
                    int keep = (int)Math.Min(count, Math.Max(0L, cap - Stored));
                    if (keep > 0) {
                        output.Write(buffer, 0, keep);
                        // Keep written bytes visible even if a deadline aborts
                        // the drain before its final durable flush.
                        output.Flush();
                        Interlocked.Add(ref stored, keep);
                    }
                }
            } catch (Exception error) { ErrorType = error.GetType().FullName; }
            finally {
                if (selfTestDelay > 0) Thread.Sleep(selfTestDelay);
                try { output.Flush(true); } catch (Exception error) { ErrorType = error.GetType().FullName; }
                try { output.Dispose(); } catch (Exception error) { ErrorType = error.GetType().FullName; }
            }
        }
        public void Abort() { try { input.Dispose(); } catch { } }
    }
}
'@
    }
    $packetRunnerStdoutFile = [IO.File]::Open($packetRunnerStdoutPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $packetRunnerStderrFile = [IO.File]::Open($packetRunnerStderrPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $packetRunnerStart = New-Object Diagnostics.ProcessStartInfo
    $packetRunnerStart.FileName = $packetRunnerInterpreter
    $packetRunnerStart.Arguments = (@($packetRunnerArguments | ForEach-Object { ConvertTo-PacketArgument $_ }) -join ' ')
    $packetRunnerStart.WorkingDirectory = $packetRunnerCwd
    $packetRunnerStart.UseShellExecute = $false
    $packetRunnerStart.CreateNoWindow = $true
    $packetRunnerStart.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $packetRunnerStart.RedirectStandardOutput = $true
    $packetRunnerStart.RedirectStandardError = $true
    $packetRunnerProcess = New-Object Diagnostics.Process
    $packetRunnerProcess.StartInfo = $packetRunnerStart
    $packetRunnerReceipt.launch_attempts = 1
    $packetRunnerReceipt.status = 'launching'
    Write-PacketReceipt
    Assert-Packet ($packetRunnerProcess.Start()) 'process_start_failed'
    $packetRunnerReceipt.launched = $true
    $packetRunnerClock.Restart()
    $packetRunnerReceipt.root_pid = $packetRunnerProcess.Id
    [void]$packetRunnerProcess.Handle
    $packetRunnerReceipt.root_creation_utc = Get-PacketCreation $packetRunnerProcess.StartTime
    Register-PacketProcess $packetRunnerProcess.Id $packetRunnerReceipt.root_creation_utc 0 '' 0 $packetRunnerProcess
    $packetRunnerRootTracked = $packetRunnerTracked[[string]$packetRunnerProcess.Id + '|' + $packetRunnerReceipt.root_creation_utc]
    $packetRunnerDrainDelay = if ($SelfTestHoldDrains) { 12000 } else { 0 }
    $packetRunnerStdoutDrain = New-Object PacketRunnerV1.BinaryDrain($packetRunnerProcess.StandardOutput.BaseStream, $packetRunnerStdoutFile, $packetRunnerLogCap, $packetRunnerDrainDelay)
    $packetRunnerStderrDrain = New-Object PacketRunnerV1.BinaryDrain($packetRunnerProcess.StandardError.BaseStream, $packetRunnerStderrFile, $packetRunnerLogCap, $packetRunnerDrainDelay)
    $packetRunnerReceipt.status = 'running'
    Write-PacketReceipt
    $nextSample = 0.0
    while ($true) {
        $sample = Get-PacketSample
        if ($SelfTest -and $InjectSupervisorException -and $packetRunnerClock.Elapsed.TotalSeconds -ge 2) { throw 'injected_post_launch_supervisor_exception' }
        if ($sample.aggregate_working_set_bytes -gt $memory) {
            $packetRunnerReceipt.limit_observed_elapsed_seconds = $packetRunnerClock.Elapsed.TotalSeconds
            $packetRunnerReceipt.memory_overshoot_bytes = $sample.aggregate_working_set_bytes - $memory
            $packetRunnerReceipt.termination_reason = 'aggregate_working_set_limit'
            $packetRunnerReceipt.status = 'terminated'; $packetRunnerReceipt.reason = 'aggregate_working_set_limit'
            Stop-PacketTree
            break
        }
        if ($packetRunnerClock.Elapsed.TotalSeconds -ge $wall) {
            $packetRunnerReceipt.limit_observed_elapsed_seconds = $packetRunnerClock.Elapsed.TotalSeconds
            $packetRunnerReceipt.wall_overshoot_seconds = $packetRunnerClock.Elapsed.TotalSeconds - $wall
            $packetRunnerReceipt.termination_reason = 'wall_timeout'
            $packetRunnerReceipt.status = 'terminated'; $packetRunnerReceipt.reason = 'wall_timeout'
            Stop-PacketTree
            break
        }
        if ($packetRunnerProcess.HasExited -and $sample.live_processes -eq 0) { break }
        $nextSample += 1.0
        $sleepMilliseconds = [int][Math]::Max(0, 1000 * ($nextSample - $packetRunnerClock.Elapsed.TotalSeconds))
        if ($sleepMilliseconds -gt 0) { Start-Sleep -Milliseconds $sleepMilliseconds }
    }
} catch {
    $packetRunnerReceipt.error_type = $_.Exception.GetType().FullName
    $packetRunnerReceipt.reason = if ($null -ne $packetRunnerReceipt.discovery_failure) { 'process_snapshot_failed' }
        elseif ($_.Exception.Message -match '^[a-z_]+$') { $_.Exception.Message } else { 'supervisor_exception' }
    $packetRunnerReceipt.status = if ($packetRunnerReceipt.launched) { 'supervisor_failed' } else { 'refused' }
    if ($packetRunnerReceipt.launched) {
        $packetRunnerReceipt.accounting_complete = $false
        $packetRunnerReceipt.failure_reason_before_drain = $packetRunnerReceipt.reason
        $packetRunnerReceipt.termination_reason = 'supervisor_exception'
        Stop-PacketTree
    }
} finally {
    $packetRunnerReceipt.monitor_elapsed_seconds = $packetRunnerClock.Elapsed.TotalSeconds
    $drainClock = [Diagnostics.Stopwatch]::StartNew()
    $drains = @(@($packetRunnerStdoutDrain, $packetRunnerStderrDrain) | Where-Object { $null -ne $_ })
    while (@($drains | Where-Object { -not $_.Completion.IsCompleted }).Count -gt 0 -and $drainClock.Elapsed.TotalSeconds -lt $packetRunnerDrainSeconds) {
        $drainSleep = [int][Math]::Min(50, [Math]::Max(0, 1000 * ($packetRunnerDrainSeconds - $drainClock.Elapsed.TotalSeconds)))
        if ($drainSleep -gt 0) { Start-Sleep -Milliseconds $drainSleep }
    }
    $packetRunnerReceipt.drain_elapsed_seconds = $drainClock.Elapsed.TotalSeconds
    $packetRunnerReceipt.drain_overshoot_seconds = [Math]::Max(0, $drainClock.Elapsed.TotalSeconds - $packetRunnerDrainSeconds)
    $packetRunnerReceipt.drain_complete = ($drains.Count -eq 2 -and @($drains | Where-Object { -not $_.Completion.IsCompleted }).Count -eq 0)
    if ($packetRunnerReceipt.launched -and -not $packetRunnerReceipt.drain_complete) {
        $packetRunnerReceipt.accounting_complete = $false
        $packetRunnerReceipt.failure_reason_before_drain = $packetRunnerReceipt.reason
        $packetRunnerReceipt.status = 'accounting_incomplete'
        $packetRunnerReceipt.reason = 'drain_deadline'
        foreach ($drain in $drains) { $drain.Abort() }
        if (-not $packetRunnerReceipt.cleanup_attempted) { Stop-PacketTree }
    }
    foreach ($item in @(@($packetRunnerStdoutPath, $packetRunnerStdoutDrain), @($packetRunnerStderrPath, $packetRunnerStderrDrain))) {
        if ($null -ne $item[0]) {
            $drain = $item[1]
            $logSnapshot = Get-PacketFileRecord $item[0]
            $packetRunnerReceipt.logs += [ordered]@{ path = $item[0]; cap_bytes = $packetRunnerLogCap;
                bytes = $logSnapshot.bytes; sha256 = $logSnapshot.sha256; hash_error = $logSnapshot.error;
                snapshot_final = ($null -ne $drain -and $drain.Completion.IsCompleted);
                received_bytes = $(if ($null -ne $drain) { $drain.Received } else { $null });
                stored_bytes = $(if ($null -ne $drain) { $drain.Stored } else { $null });
                discarded_bytes = $(if ($null -ne $drain) { $drain.Discarded } else { $null });
                completed = ($null -ne $drain -and $drain.Completion.IsCompleted);
                error_type = $(if ($null -ne $drain) { $drain.ErrorType } else { $null }) }
            if ($null -ne $drain -and $null -ne $drain.ErrorType) { $packetRunnerReceipt.accounting_complete = $false }
        }
    }
    if ($null -eq $packetRunnerStdoutDrain -and $null -ne $packetRunnerStdoutFile) { $packetRunnerStdoutFile.Dispose() }
    if ($null -eq $packetRunnerStderrDrain -and $null -ne $packetRunnerStderrFile) { $packetRunnerStderrFile.Dispose() }
    if ($null -ne $packetRunnerProcess -and $packetRunnerReceipt.launched) {
        try {
            $packetRunnerReceipt.root_exit_observed = $packetRunnerProcess.HasExited
            if ($packetRunnerReceipt.root_exit_observed) { $packetRunnerReceipt.root_exit_code = $packetRunnerProcess.ExitCode }
        } catch { $packetRunnerReceipt.accounting_complete = $false }
    }
    if ($packetRunnerReceipt.status -eq 'running') {
        if (-not $packetRunnerReceipt.accounting_complete -or -not $packetRunnerReceipt.root_exit_observed) {
            $packetRunnerReceipt.status = 'accounting_incomplete'; $packetRunnerReceipt.reason = 'exit_or_stream_accounting_unknown'
        } elseif ($packetRunnerReceipt.root_exit_code -eq 0) {
            $packetRunnerReceipt.status = 'completed'; $packetRunnerReceipt.reason = 'exit_zero'
        } else { $packetRunnerReceipt.status = 'child_failed'; $packetRunnerReceipt.reason = 'exit_nonzero' }
    }
    foreach ($path in $packetRunnerInputPaths) { $packetRunnerReceipt.post_inputs += (Get-PacketFileRecord $path) }
    foreach ($path in $packetRunnerOutputPaths) {
        $outputRecord = Get-PacketFileRecord $path
        $outputRecord.snapshot_final = ($packetRunnerReceipt.drain_complete -and
            ($packetRunnerReceipt.cleanup_complete -ne $false))
        $packetRunnerReceipt.outputs += $outputRecord
        if ($null -ne $outputRecord.error) { $packetRunnerReceipt.accounting_complete = $false }
    }
    if ($packetRunnerReceipt.status -eq 'completed' -and -not $packetRunnerReceipt.accounting_complete) {
        $packetRunnerReceipt.status = 'accounting_incomplete'; $packetRunnerReceipt.reason = 'postflight_accounting_failure'
    }
    if ($packetRunnerReceipt.pre_inputs.Count -eq $packetRunnerReceipt.post_inputs.Count) {
        for ($ordinal = 0; $ordinal -lt $packetRunnerReceipt.pre_inputs.Count; $ordinal++) {
            $before = $packetRunnerReceipt.pre_inputs[$ordinal]; $after = $packetRunnerReceipt.post_inputs[$ordinal]
            if ($null -ne $after.error -or $before.sha256 -ne $after.sha256 -or $before.bytes -ne $after.bytes) {
                $packetRunnerReceipt.accounting_complete = $false
                if ($packetRunnerReceipt.status -eq 'completed') {
                    $packetRunnerReceipt.status = 'accounting_incomplete'; $packetRunnerReceipt.reason = 'post_input_mismatch'
                }
            }
        }
    }
    Write-PacketReceipt
    $packetRunnerReceiptStream.Dispose()
    foreach ($tracked in @($packetRunnerTracked.Values)) {
        if ($null -ne $tracked.Handle -and $tracked.Handle -ne $packetRunnerProcess) { $tracked.Handle.Dispose() }
    }
    if ($null -ne $packetRunnerProcess) { $packetRunnerProcess.Dispose() }
}
Write-Output $packetRunnerReceiptFull
if ($packetRunnerReceipt.status -ne 'completed') { exit 1 }
exit 0
