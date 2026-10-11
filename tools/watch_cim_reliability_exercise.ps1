# Independent sampled watchdog; no CIM, images or packet runner invocation.
#requires -Version 5.1
[CmdletBinding()]
param([switch]$Execute, [string]$GatePath, [string]$ExpectedGateSha256,
      [int]$ControllerId, [string]$ControllerCreation,
      [int]$ChildId, [string]$ChildCreation, [string]$DeadlineUtc)
$script:cimWatchStartedUtc=[DateTime]::UtcNow
$script:cimWatchClock=[Diagnostics.Stopwatch]::StartNew()
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Get-CimExerciseCreation($Handle) {
    return $Handle.StartTime.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ')
}
function Get-CimExerciseRemainingDeadline([string]$Deadline,[datetime]$WatchStartedUtc) {
    $parsed=[datetime]::MinValue
    if(-not [datetime]::TryParseExact($Deadline,'o',[Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind,[ref]$parsed) -or $parsed.Kind -ne [DateTimeKind]::Utc){throw 'watchdog_deadline'}
    $remaining=($parsed-$WatchStartedUtc.ToUniversalTime()).TotalSeconds
    if([double]::IsNaN($remaining) -or [double]::IsInfinity($remaining) -or $remaining -le 0 -or $remaining -gt 450){throw 'watchdog_deadline'}
    return $remaining
}
function Update-CimExerciseWatchdogProtocol($Record,$Task,$Buffer,$Output,$InputStream) {
    if(-not $Record.controller_terminal_committed -and $Task.IsCompleted){
        if($Task.GetAwaiter().GetResult() -ne 1){throw 'watchdog_protocol'}
        if(-not $Record.controller_terminal_acknowledged){
            if($Buffer[0] -ne 68){throw 'watchdog_protocol'}
            $Record.controller_terminal_acknowledged=$true
            # A never releases the monitor. Observe the separate final-flush marker next.
            $Output.WriteByte(65);$Output.Flush();[Array]::Clear($Buffer,0,$Buffer.Length)
            return $InputStream.ReadAsync($Buffer,0,1)
        }
        if($Buffer[0] -ne 70){throw 'watchdog_protocol'}
        $Record.controller_terminal_committed=$true
        [Array]::Clear($Buffer,0,$Buffer.Length)
    }
    return $Task
}
function Bind-CimExerciseHandle($Handle, [int]$ExpectedId, [string]$ExpectedCreation) {
    [void]$Handle.Handle
    if ($Handle.Id -ne $ExpectedId -or (Get-CimExerciseCreation $Handle) -ne $ExpectedCreation) {
        throw 'identity_mismatch'
    }
    return $Handle
}
function Stop-CimExerciseHandles($Child, $Controller, $Clock, [double]$Seconds) {
    $complete = $true
    foreach ($handle in @($Child, $Controller)) {
        if ($Clock.Elapsed.TotalSeconds -ge $Seconds) { $complete = $false; continue }
        if ($null -eq $handle) { $complete = $false; continue }
        try {
            if (-not $handle.HasExited) { $handle.Kill() }
        } catch { $complete = $false }
    }
    foreach ($handle in @($Child, $Controller)) {
        if ($null -eq $handle) { continue }
        try {
            $remaining = [Math]::Max(0, [int](1000 * ($Seconds - $Clock.Elapsed.TotalSeconds)))
            if (-not $handle.WaitForExit($remaining)) { $complete = $false }
        } catch { $complete = $false }
    }
    return $complete
}
function Get-CimExerciseMemory($Controller, $Child, $Watchdog) {
    $controllerBytes = [long]0; $childBytes = [long]0
    foreach ($handle in @($Controller, $Watchdog)) {
        $handle.Refresh()
        if (-not $handle.HasExited) {
            $bytes=[long]$handle.WorkingSet64
            if ($bytes -lt 0) { throw 'memory_unavailable' }
            $controllerBytes += $bytes
        }
    }
    $Child.Refresh()
    if (-not $Child.HasExited) { $childBytes = [long]$Child.WorkingSet64 }
    if ($childBytes -lt 0) { throw 'memory_unavailable' }
    return [ordered]@{ controller_and_watchdog_working_set_bytes = $controllerBytes;
        child_working_set_bytes = $childBytes; combined_working_set_bytes = $controllerBytes + $childBytes;
        exceeded = ($controllerBytes -gt 536870912 -or $childBytes -gt 268435456 -or
                    $controllerBytes + $childBytes -gt 805306368) }
}
function Assert-CimExerciseGate([string]$Path, [string]$Expected, [string]$RepoRoot) {
    if ($Expected -notmatch '^[A-Fa-f0-9]{64}$') { throw 'invalid_gate_hash' }
    $full = [IO.Path]::GetFullPath($Path)
    $runs = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'agent swarm instantiation template\runs'))
    if (-not $full.StartsWith($runs + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'gate_outside_runs' }
    if ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash -ne $Expected) { throw 'gate_hash_mismatch' }
    $gate = [IO.File]::ReadAllText($full) | ConvertFrom-Json
    $keys = @($gate.PSObject.Properties.Name | Sort-Object)
    $allowed = @('astra_go','executable','launch_allowed','lead_authorized','output_base','powershell_path',
                 'powershell_sha256','schema','sources') | Sort-Object
    if (($keys -join ',') -ne ($allowed -join ',') -or $gate.schema -ne 'native-cim-exercise-gate-v1') { throw 'gate_schema' }
    foreach ($name in @('astra_go','executable','launch_allowed','lead_authorized')) {
        if ($gate.$name -isnot [bool] -or -not $gate.$name) { throw 'gate_not_authorized' }
    }
    if ($gate.output_base -ne 'agent swarm instantiation template/runs/SOL-CIM-NATIVE-RELIABILITY-EXERCISE-01') { throw 'gate_output_base' }
    $expectedPaths = @('tools/run_image_packet.ps1','tools/run_cim_reliability_exercise.ps1',
                      'tools/watch_cim_reliability_exercise.ps1','tests/test_cim_reliability_exercise.py')
    if (@($gate.sources).Count -ne $expectedPaths.Count) { throw 'gate_source_count' }
    for ($index = 0; $index -lt $expectedPaths.Count; $index++) {
        $pin = $gate.sources[$index]
        if ((@($pin.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'path,sha256' -or
            $pin.path -ne $expectedPaths[$index] -or $pin.sha256 -notmatch '^[A-Fa-f0-9]{64}$') { throw 'gate_source_schema' }
        if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $pin.path) -Algorithm SHA256).Hash -ne $pin.sha256) { throw 'gate_source_mismatch' }
    }
    if ($gate.sources[0].sha256 -ne '11980AEBA3AFE4DFBD7986A90BE351310ECC4EE85A467BA286652989E5C5B6FE') { throw 'accepted_runner_pin' }
    $nativeHost = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if ([IO.Path]::GetFullPath($gate.powershell_path) -ne [IO.Path]::GetFullPath($nativeHost) -or
        (Get-FileHash -LiteralPath $nativeHost -Algorithm SHA256).Hash -ne $gate.powershell_sha256) { throw 'gate_interpreter_pin' }
    return $gate
}
function Write-CimExerciseWatchdog([string]$Path, $Record) {
    $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes(($Record | ConvertTo-Json -Compress) + [Environment]::NewLine)
    if ($bytes.Length -gt 65536) { throw 'watchdog_receipt_cap' }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        if ($stream.Length -ne 0) { throw 'watchdog_reserved_output_changed' }
        $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true)
    } finally { $stream.Dispose() }
}

if (-not $Execute) { throw 'native_exercise_not_authorized' }
$deadlineSeconds=Get-CimExerciseRemainingDeadline $DeadlineUtc $script:cimWatchStartedUtc
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gate = Assert-CimExerciseGate $GatePath $ExpectedGateSha256 $root
$controller = $null; $child = $null; $self = $null
$clock = $script:cimWatchClock
$record = [ordered]@{ timestamp_utc = [DateTime]::UtcNow.ToString('o'); status = 'not_ready';
    elapsed_seconds = 0.0; cleanup_complete = $false; accounting_complete = $false;
    peak_combined_working_set_bytes = [long]0; controller_terminal_acknowledged=$false;
    controller_terminal_committed=$false }
$receipt = Join-Path $root ($gate.output_base + '.watchdog.json')
try {
    $controller = Bind-CimExerciseHandle ([Diagnostics.Process]::GetProcessById($ControllerId)) $ControllerId $ControllerCreation
    $child = Bind-CimExerciseHandle ([Diagnostics.Process]::GetProcessById($ChildId)) $ChildId $ChildCreation
    $self = [Diagnostics.Process]::GetCurrentProcess()
    # One byte only; readiness means both retained handles are identity bound before any query.
    $out = [Console]::OpenStandardOutput(); $out.WriteByte(82); $out.Flush()
    $watchInputStream = [Console]::OpenStandardInput(); $stopBuffer = New-Object byte[] 1
    $stopRead = $watchInputStream.ReadAsync($stopBuffer, 0, 1)
    while ($true) {
        $stopRead=Update-CimExerciseWatchdogProtocol $record $stopRead $stopBuffer $out $watchInputStream
        if ($controller.HasExited) {
            if($record.controller_terminal_acknowledged -and -not $record.controller_terminal_committed){
                # The controller may exit just before the asynchronous F read callback completes.
                $remainingMilliseconds=[int][Math]::Max(0,[Math]::Min(100,1000*($deadlineSeconds-$clock.Elapsed.TotalSeconds)))
                if($stopRead.Wait($remainingMilliseconds)){
                    $stopRead=Update-CimExerciseWatchdogProtocol $record $stopRead $stopBuffer $out $watchInputStream
                }
            }
            $record.status = 'controller_exited';
            $record.cleanup_complete = Stop-CimExerciseHandles $child $controller ([Diagnostics.Stopwatch]::StartNew()) 5
            $record.accounting_complete=($record.cleanup_complete -and $record.controller_terminal_committed)
            break
        }
        $memory = Get-CimExerciseMemory $controller $child $self
        $record.peak_combined_working_set_bytes = [Math]::Max($record.peak_combined_working_set_bytes, $memory.combined_working_set_bytes)
        if ($memory.exceeded -or $clock.Elapsed.TotalSeconds -ge $DeadlineSeconds) {
            $record.status = if ($memory.exceeded) { 'memory_limit_exceeded' } else { 'watchdog_stopped' }
            $record.cleanup_complete = Stop-CimExerciseHandles $child $controller ([Diagnostics.Stopwatch]::StartNew()) 5
            break
        }
        Start-Sleep -Milliseconds 100
    }
} catch {
    $record.status = 'watchdog_incomplete'
    # No exception text; kill only handles already successfully identity bound.
    $record.cleanup_complete = Stop-CimExerciseHandles $child $controller ([Diagnostics.Stopwatch]::StartNew()) 5
} finally {
    $record.elapsed_seconds = [Math]::Min(86400.0, $clock.Elapsed.TotalSeconds)
    try { Write-CimExerciseWatchdog $receipt $record } finally {
        foreach ($handle in @($controller,$child,$self)) { if ($null -ne $handle) { $handle.Dispose() } }
    }
}
if (-not $record.accounting_complete) { exit 1 }
exit 0
