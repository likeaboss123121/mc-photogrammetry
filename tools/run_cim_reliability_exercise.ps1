# Native local CIM observation only. Never invoke image-packet top-level code.
#requires -Version 5.1
[CmdletBinding()]
param([switch]$Execute, [string]$GatePath, [string]$ExpectedGateSha256)
# Deadline origin precedes gate/source/output preflight and either process startup.
$script:cimExerciseStartedUtc=[DateTime]::UtcNow
$script:cimExerciseClock=[Diagnostics.Stopwatch]::StartNew()
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Import-CimExerciseFunctions([string]$RunnerPath) {
    if ((Get-FileHash -LiteralPath $RunnerPath -Algorithm SHA256).Hash -ne
        '11980AEBA3AFE4DFBD7986A90BE351310ECC4EE85A467BA286652989E5C5B6FE') { throw 'accepted_runner_pin' }
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($RunnerPath, [ref]$tokens, [ref]$errors)
    if ($errors.Count -ne 0) { throw 'accepted_runner_parse' }
    $names = @('Get-PacketField','Assert-Packet','ConvertTo-PacketArgument','Get-PacketCreation',
        'Register-PacketProcess','Update-PacketExit','Get-PacketDiagnosticCode','Get-PacketMonitorSeconds',
        'Get-PacketBoundedInterval','Update-PacketSnapshotTelemetry','Get-PacketProcessSnapshot','Get-PacketSample','Stop-PacketTree')
    foreach ($name in $names) {
        $nodes = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true))
        if ($nodes.Count -ne 1) { throw 'accepted_function_not_unique' }
        # Explicit script scope, exact original scriptblock, no top-level runner execution.
        $declaration=$nodes[0].Extent.Text.Replace(('function ' + $name), ('function script:' + $name))
        . ([ScriptBlock]::Create($declaration))
    }
}
function New-CimExerciseProcess([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = (@($Arguments | ForEach-Object { ConvertTo-PacketArgument $_ }) -join ' ')
    $info.WorkingDirectory = $WorkingDirectory; $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true; $info.RedirectStandardInput = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    if (-not $process.Start()) { $process.Dispose(); throw 'synthetic_launch_failed' }
    return $process
}
function Get-CimExerciseController { return [Diagnostics.Process]::GetCurrentProcess() }
function Open-CimExerciseOutput([string]$Path) {
    return [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite)
}
function New-CimExerciseClock { return [Diagnostics.Stopwatch]::StartNew() }
function Wait-CimExercise([int]$Milliseconds) { if($Milliseconds -gt 0){Start-Sleep -Milliseconds $Milliseconds} }
function New-CimExerciseDrain($Stream) {
    $buffer = New-Object byte[] 4096
    return [pscustomobject]@{ Stream=$Stream; Buffer=$buffer; Task=$Stream.ReadAsync($buffer,0,$buffer.Length);
        Count=[long]0; Complete=$false; Failed=$false }
}
function Update-CimExerciseDrain($Drain) {
    if ($Drain.Complete -or $Drain.Failed) { return }
    try {
        while ($Drain.Task.IsCompleted) {
            $count = $Drain.Task.GetAwaiter().GetResult()
            $Drain.Count += $count
            [Array]::Clear($Drain.Buffer,0,$Drain.Buffer.Length)
            if ($Drain.Count -gt 65536) { throw 'stream_cap' }
            if ($count -eq 0) { $Drain.Complete=$true; break }
            $Drain.Task=$Drain.Stream.ReadAsync($Drain.Buffer,0,$Drain.Buffer.Length)
        }
    } catch { $Drain.Failed=$true; [Array]::Clear($Drain.Buffer,0,$Drain.Buffer.Length); throw }
}
function Write-CimExerciseScalars($Stream, $Record, [int]$Cap) {
    $allowed = @('index','timestamp_utc','elapsed_seconds','query_duration_seconds','query_duration_state',
        'query_start_gap_seconds','query_start_gap_state','global_row_count','tracked_count','live_count',
        'child_working_set_bytes','controller_and_watchdog_working_set_bytes','combined_working_set_bytes',
        'status','child_identity_join_valid','child_exit_observed','accounting_complete','hresult',
        'native_error_code','status_code','attempted_samples','completed_samples','last_acknowledged_sample',
        'peak_combined_working_set_bytes','child_launch_count','child_exit_code','cleanup_attempted',
        'cleanup_complete','cleanup_elapsed_seconds','cleanup_overshoot_seconds','drain_complete',
        'drain_elapsed_seconds','drain_overshoot_seconds','stdout_bytes','stderr_bytes','watchdog_ready',
        'watchdog_stdout_bytes','watchdog_stderr_bytes','watchdog_completion_pending')
    foreach ($key in $Record.Keys) {
        if ($key -notin $allowed) { throw 'scalar_key' }
        $value=$Record[$key]
        $booleanKeys=@('child_identity_join_valid','child_exit_observed','accounting_complete',
            'cleanup_attempted','cleanup_complete','drain_complete','watchdog_ready','watchdog_completion_pending')
        if ($key -in $booleanKeys) {
            if ($value -isnot [bool]) { throw 'scalar_boolean' }
            continue
        }
        if ($null -eq $value) {
            if ($key -notin @('query_duration_seconds','query_start_gap_seconds','global_row_count',
                'child_working_set_bytes','controller_and_watchdog_working_set_bytes','combined_working_set_bytes',
                'hresult','native_error_code','status_code','last_acknowledged_sample','child_exit_code')) { throw 'scalar_null' }
            continue
        }
        if ($value -is [bool]) { throw 'scalar_type' }
        if ($value -is [string]) {
            if ($key -eq 'timestamp_utc') {
                if ($value -notmatch '^\d{4}-\d{2}-\d{2}T[0-9:.]+Z$' -or $value.Length -gt 40) { throw 'scalar_timestamp' }
            } elseif ($key -in @('query_duration_state','query_start_gap_state')) {
                if ($value -notin @('measured','capped','unavailable')) { throw 'scalar_enum' }
            } elseif ($key -eq 'status') {
                if ($value -notin @('returned','native_query_failed','accounting_failed','row_cap_exceeded',
                    'memory_limit_exceeded','wall_limit_exceeded','telemetry_incomplete','cancelled',
                    'not_started','observed_no_failure','exercise_incomplete','finalizing')) { throw 'scalar_enum' }
            } else { throw 'scalar_string' }
        } elseif ($value -is [int] -or $value -is [long] -or $value -is [double]) {
            if ($key -in @('status','timestamp_utc','query_duration_state','query_start_gap_state')) { throw 'scalar_type' }
            if (-not $key.EndsWith('_seconds') -and $value -is [double]) { throw 'scalar_integral' }
            if ([double]::IsNaN([double]$value) -or [double]::IsInfinity([double]$value) -or
                [double]$value -lt -2147483648 -or [double]$value -gt 4294967295) { throw 'scalar_number' }
            if ($key -notin @('hresult','native_error_code','status_code','child_exit_code') -and $value -lt 0) { throw 'scalar_negative' }
            if ($key -eq 'child_exit_code' -and ($value -lt -2147483648 -or $value -gt 2147483647)) { throw 'scalar_exit_code' }
            if ($key.EndsWith('_seconds') -and $value -gt 86400) { throw 'scalar_seconds' }
            if ($key -in @('index','last_acknowledged_sample') -and $value -gt 299) { throw 'scalar_index' }
            if ($key -in @('attempted_samples','completed_samples') -and $value -gt 300) { throw 'scalar_sample_count' }
            if ($key -eq 'global_row_count' -and $value -gt 8192) { throw 'scalar_rows' }
        } else { throw 'scalar_payload' }
    }
    $bytes=(New-Object Text.UTF8Encoding($false)).GetBytes(($Record|ConvertTo-Json -Compress)+[Environment]::NewLine)
    if ($Stream.Length + $bytes.Length -gt $Cap) { throw 'receipt_cap' }
    $Stream.Write($bytes,0,$bytes.Length);$Stream.Flush($true)
}
function Get-CimExerciseSampleProjection($Sample, [int]$Index, [int]$GlobalRows, $Memory) {
    if ($Index -lt 0 -or $Index -ge 300 -or $GlobalRows -lt 0 -or $GlobalRows -gt 8192) { throw 'sample_cap' }
    $timing=$packetRunnerReceipt.process_snapshot_telemetry
    if ($null -eq $timing -or $timing.query_duration_state -ne 'measured' -or
        ($Index -gt 0 -and $timing.query_start_gap_state -ne 'measured')) { throw 'telemetry_incomplete' }
    return [ordered]@{index=$Index;timestamp_utc=$Sample.timestamp_utc;elapsed_seconds=$Sample.elapsed_seconds;
        query_duration_seconds=$timing.query_duration_seconds;query_duration_state=$timing.query_duration_state;
        query_start_gap_seconds=$timing.query_start_gap_seconds;query_start_gap_state=$timing.query_start_gap_state;
        global_row_count=$GlobalRows;tracked_count=$packetRunnerTracked.Count;live_count=$Sample.live_processes;
        child_working_set_bytes=$Memory.child_working_set_bytes;
        controller_and_watchdog_working_set_bytes=$Memory.controller_and_watchdog_working_set_bytes;
        combined_working_set_bytes=$Memory.combined_working_set_bytes;status='returned';child_identity_join_valid=$true;
        child_exit_observed=$packetRunnerProcess.HasExited;accounting_complete=$packetRunnerReceipt.accounting_complete;
        hresult=$null;native_error_code=$null;status_code=$null}
}
function Assert-CimExerciseSampleAccounting($Sample, $Process, $Receipt, [int]$GlobalRows) {
    # The controller is alive: global enumeration cannot honestly be an empty table.
    # A zero tracked-live set after observed child exit is a separate valid condition.
    if ($GlobalRows -lt 1 -or -not $Receipt.accounting_complete -or
        (-not $Process.HasExited -and $Sample.live_processes -ne 1) -or
        $Sample.live_processes -lt 0 -or $Sample.live_processes -gt 1) { throw 'accounting_failed' }
}
function Update-CimExerciseKnownStreamCounts($Terminal, $Drains) {
    # A drain may throw after advancing Count. Copy acknowledged counts unconditionally.
    if($Drains.Count -ge 2){$Terminal.stdout_bytes=$Drains[0].Count;$Terminal.stderr_bytes=$Drains[1].Count}
    if($Drains.Count -ge 3){$Terminal.watchdog_stderr_bytes=$Drains[2].Count}
}
function Set-CimExerciseTerminalFailure($Terminal) {
    $Terminal.accounting_complete=$false
    if($Terminal.status -in @('observed_no_failure','finalizing','not_started')){$Terminal.status='exercise_incomplete'}
}
function Publish-CimExerciseTerminal($Stream,$Terminal) {
    # Pending acknowledgement, then terminal; one bounded failure publication if finalization faults.
    # The watchdog still monitors the controller through both durable flushes.
    $Stream.Position=0;$Stream.SetLength(0)
    Write-CimExerciseScalars $Stream $Terminal 65536
}
function Confirm-CimExerciseWatchdog($Watchdog,$Terminal) {
    if($Watchdog.HasExited){throw 'watchdog_exited_early'}
    $Watchdog.StandardInput.BaseStream.WriteByte(68);$Watchdog.StandardInput.BaseStream.Flush()
    $ack=New-Object byte[] 1
    $task=$Watchdog.StandardOutput.BaseStream.ReadAsync($ack,0,1)
    if(-not $task.Wait(5000)){throw 'watchdog_ack_timeout'}
    $count=$task.GetAwaiter().GetResult()
    $Terminal.watchdog_stdout_bytes += $count
    try {if($count -ne 1 -or $ack[0] -ne 65 -or $Watchdog.HasExited){throw 'watchdog_ack_invalid'}}
    finally {[Array]::Clear($ack,0,$ack.Length)}
    # A acknowledges receipt of D, not watchdog exit/drain. It remains live until controller exit.
    $Terminal.watchdog_completion_pending=$true
}

function Invoke-CimReliabilityExercise {
if (-not $Execute) { throw 'native_exercise_not_authorized' }
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# Load declarations only from the watchdog; never execute its top-level launch path.
$tokens=$null;$errors=$null
$watchdogPath=Join-Path $PSScriptRoot 'watch_cim_reliability_exercise.ps1'
$watchAst=[Management.Automation.Language.Parser]::ParseFile($watchdogPath,[ref]$tokens,[ref]$errors)
if ($errors.Count -ne 0) { throw 'watchdog_parse' }
foreach ($name in @('Assert-CimExerciseGate','Get-CimExerciseCreation','Bind-CimExerciseHandle','Get-CimExerciseMemory')) {
    $nodes=@($watchAst.FindAll({param($node)$node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true))
    if ($nodes.Count -ne 1) { throw 'watchdog_function_not_unique' }
    . ([ScriptBlock]::Create($nodes[0].Extent.Text))
}
$gate=Assert-CimExerciseGate $GatePath $ExpectedGateSha256 $root
Import-CimExerciseFunctions (Join-Path $PSScriptRoot 'run_image_packet.ps1')
Invoke-CimExerciseLifecycle $gate $root $watchdogPath $script:cimExerciseClock $script:cimExerciseStartedUtc
}
function Invoke-CimExerciseLifecycle($Gate,[string]$Root,[string]$WatchdogPath,$Clock,[datetime]$StartedUtc) {
$gate=$Gate;$root=$Root;$watchdogPath=$WatchdogPath
$base=Join-Path $root $gate.output_base
$paths=@('.samples.jsonl','.terminal.json','.stdout.log','.stderr.log','.watchdog.json')|ForEach-Object{$base+$_}
foreach ($path in $paths) { if ([IO.File]::Exists($path) -or [IO.Directory]::Exists($path)) { throw 'output_exists' } }
$streams=New-Object 'Collections.Generic.List[object]'
$controller=$null;$watchdog=$null;$packetRunnerProcess=$null;$drains=@();$originalError=$null
$packetRunnerClock=$Clock
$packetRunnerCleanupSeconds=5
$packetRunnerTracked=[ordered]@{}
$packetRunnerSamples=New-Object 'Collections.Generic.List[object]'
$script:packetRunnerPreviousSnapshotStartSeconds=$null
$packetRunnerReceipt=[ordered]@{launched=$false;accounting_complete=$true;process_snapshot_telemetry=$null;
    discovery_failure=$null;peak_aggregate_working_set_bytes=[long]0;cleanup_attempted=$false;
    cleanup_complete=$false;cleanup_elapsed_seconds=0.0;cleanup_overshoot_seconds=0.0;
    cleanup_errors=@();root_pid=$null;root_creation_utc=$null}
# Exact original sampler gets a scalar-only persistence entry point, no packet/file side effects.
function Write-PacketReceipt { }
$terminal=[ordered]@{timestamp_utc=[DateTime]::UtcNow.ToString('o');status='not_started';attempted_samples=0;
    completed_samples=0;last_acknowledged_sample=$null;elapsed_seconds=0.0;
    peak_combined_working_set_bytes=[long]0;child_launch_count=0;child_exit_observed=$false;child_exit_code=$null;
    cleanup_attempted=$false;cleanup_complete=$false;cleanup_elapsed_seconds=0.0;cleanup_overshoot_seconds=0.0;
    drain_complete=$false;drain_elapsed_seconds=0.0;drain_overshoot_seconds=0.0;accounting_complete=$false;
    stdout_bytes=[long]0;stderr_bytes=[long]0;watchdog_stdout_bytes=[long]0;watchdog_stderr_bytes=[long]0;
    watchdog_ready=$false;watchdog_completion_pending=$false;hresult=$null;native_error_code=$null;status_code=$null}
try {
    foreach ($path in $paths) { $streams.Add((Open-CimExerciseOutput $path)) }
    # Independent watchdog owns its reserved receipt; no inherited writable object.
    $streams[4].Dispose()
    $controller=Get-CimExerciseController
    [void]$controller.Handle
    $controllerCreation=Get-CimExerciseCreation $controller
    $terminal.child_launch_count=1
    $packetRunnerProcess=New-CimExerciseProcess $gate.powershell_path @('-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 240; exit 0') $root
    $packetRunnerReceipt.launched=$true
    [void]$packetRunnerProcess.Handle
    $childCreation=Get-PacketCreation $packetRunnerProcess.StartTime
    Register-PacketProcess $packetRunnerProcess.Id $childCreation 0 '' 0 $packetRunnerProcess
    $drains=@(New-CimExerciseDrain $packetRunnerProcess.StandardOutput.BaseStream;New-CimExerciseDrain $packetRunnerProcess.StandardError.BaseStream)
    $deadlineUtc=$StartedUtc.ToUniversalTime().AddSeconds(450).ToString('o')
    $watchdog=New-CimExerciseProcess $gate.powershell_path @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$watchdogPath,
        '-Execute','-GatePath',$GatePath,'-ExpectedGateSha256',$ExpectedGateSha256,
        '-ControllerId',[string]$controller.Id,'-ControllerCreation',$controllerCreation,
        '-ChildId',[string]$packetRunnerProcess.Id,'-ChildCreation',$childCreation,
        '-DeadlineUtc',$deadlineUtc) $root
    [void]$watchdog.Handle
    $ready=New-Object byte[] 1
    $readyTask=$watchdog.StandardOutput.BaseStream.ReadAsync($ready,0,1)
    if (-not $readyTask.Wait(5000) -or $readyTask.GetAwaiter().GetResult() -ne 1 -or $ready[0] -ne 82) { throw 'watchdog_not_ready' }
    $terminal.watchdog_ready=$true
    $terminal.watchdog_stdout_bytes=1
    # Watchdog stdout is a two-byte protocol (R then A), never an EOF drain before controller exit.
    $drains += New-CimExerciseDrain $watchdog.StandardError.BaseStream
    # Snapshot wrapper materializes the real query exactly once. Original rows/exception unchanged.
    $script:actualSnapshot=${function:Get-PacketProcessSnapshot}
    $script:globalRows=$null
    function Get-PacketProcessSnapshot {
        $script:globalRows=$null
        $rows=@(& $script:actualSnapshot)
        $script:globalRows=$rows.Count
        if ($rows.Count -gt 8192) { throw 'row_cap_exceeded' }
        foreach ($row in $rows) { Write-Output -NoEnumerate $row }
    }
    $previousStart=$null
    for ($index=0;$index -lt 300;$index++) {
        if ($null -ne $previousStart) {
            $wait=1-($packetRunnerClock.Elapsed.TotalSeconds-$previousStart)
            if ($wait -gt 0) { Wait-CimExercise ([int][Math]::Ceiling(1000*$wait)) }
        }
        if ($packetRunnerClock.Elapsed.TotalSeconds -ge 420) { throw 'wall_limit_exceeded' }
        if ($watchdog.HasExited) { throw 'watchdog_exited_early' }
        $previousStart=$packetRunnerClock.Elapsed.TotalSeconds
        $terminal.attempted_samples++
        $sample=Get-PacketSample
        if ($packetRunnerClock.Elapsed.TotalSeconds -ge 420) { throw 'wall_limit_exceeded' }
        # Accepted sampler allows unknown missing-live observations; exercise refuses immediately.
        Assert-CimExerciseSampleAccounting $sample $packetRunnerProcess $packetRunnerReceipt $script:globalRows
        $memory=Get-CimExerciseMemory $controller $packetRunnerProcess $watchdog
        $terminal.peak_combined_working_set_bytes=[Math]::Max($terminal.peak_combined_working_set_bytes,$memory.combined_working_set_bytes)
        if ($memory.exceeded) { throw 'memory_limit_exceeded' }
        foreach ($drain in $drains) { Update-CimExerciseDrain $drain; if ($drain.Count -ne 0) { throw 'unexpected_stream_bytes' } }
        $projection=Get-CimExerciseSampleProjection $sample $index $script:globalRows $memory
        Write-CimExerciseScalars $streams[0] $projection 524288
        $terminal.completed_samples++;$terminal.last_acknowledged_sample=$index
        $terminal.peak_combined_working_set_bytes=[Math]::Max($terminal.peak_combined_working_set_bytes,$memory.combined_working_set_bytes)
        $sample=$null;$projection=$null
    }
    if (-not $packetRunnerProcess.HasExited -or $packetRunnerProcess.ExitCode -ne 0) { throw 'workload_incomplete' }
    $terminal.status='observed_no_failure';$terminal.accounting_complete=$true
} catch {
    $originalError=$_
    $terminal.status=if ($_.Exception -is [OperationCanceledException]) {'cancelled'} elseif ($null -ne $packetRunnerReceipt.discovery_failure) {'native_query_failed'} else {'exercise_incomplete'}
    $terminal.accounting_complete=$false
    if ($null -ne $packetRunnerReceipt.discovery_failure) {
        $terminal.hresult=$packetRunnerReceipt.discovery_failure.hresult
        $terminal.native_error_code=$packetRunnerReceipt.discovery_failure.native_error_code
        $terminal.status_code=$packetRunnerReceipt.discovery_failure.status_code
    }
} finally {
    try {
        Stop-PacketTree
        $terminal.cleanup_attempted=$packetRunnerReceipt.cleanup_attempted
        $terminal.cleanup_complete=$packetRunnerReceipt.cleanup_complete
        $terminal.cleanup_elapsed_seconds=$packetRunnerReceipt.cleanup_elapsed_seconds
        $terminal.cleanup_overshoot_seconds=$packetRunnerReceipt.cleanup_overshoot_seconds
    } catch { $terminal.cleanup_complete=$false;Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_} }
    $drainClock=New-CimExerciseClock
    try {
        do {
            foreach ($drain in $drains) { Update-CimExerciseDrain $drain }
            if ($drains.Count -ge 2 -and $drains[0].Complete -and $drains[1].Complete) { break }
            Wait-CimExercise 20
        } while ($drainClock.Elapsed.TotalSeconds -lt 10)
        $terminal.drain_complete=($drains.Count -eq 3 -and $drains[0].Complete -and $drains[1].Complete -and
            @($drains|Where-Object{$_.Failed}).Count -eq 0)
        if (@($drains|Where-Object{$_.Count -gt 0}).Count -gt 0) {$terminal.accounting_complete=$false}
    } catch { $terminal.drain_complete=$false;Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_} }
    finally {Update-CimExerciseKnownStreamCounts $terminal $drains}
    $terminal.drain_elapsed_seconds=$drainClock.Elapsed.TotalSeconds
    $terminal.drain_overshoot_seconds=[Math]::Max(0,$terminal.drain_elapsed_seconds-10)
    if ($null -ne $packetRunnerProcess) {
        try {$terminal.child_exit_observed=$packetRunnerProcess.HasExited;if($terminal.child_exit_observed){$terminal.child_exit_code=$packetRunnerProcess.ExitCode}}catch{$terminal.accounting_complete=$false}
    }
    if (-not $terminal.cleanup_complete -or -not $terminal.drain_complete -or -not $terminal.accounting_complete) { Set-CimExerciseTerminalFailure $terminal }
    $terminal.elapsed_seconds=$packetRunnerClock.Elapsed.TotalSeconds
    try {
        if($streams.Count -ge 2){
            $pending=[ordered]@{};foreach($key in $terminal.Keys){$pending[$key]=$terminal[$key]}
            $pending.accounting_complete=$false
            if($pending.status -eq 'observed_no_failure'){$pending.status='finalizing'}
            try {
                Publish-CimExerciseTerminal $streams[1] $pending
                if($terminal.watchdog_ready){Confirm-CimExerciseWatchdog $watchdog $terminal}
                else {throw 'watchdog_not_ready'}
                foreach($drain in $drains){Update-CimExerciseDrain $drain}
                if(@($drains|Where-Object{$_.Count -gt 0}).Count -gt 0){throw 'unexpected_stream_bytes'}
                if($watchdog.HasExited){throw 'watchdog_exited_early'}
            } catch {Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_}}
            finally {Update-CimExerciseKnownStreamCounts $terminal $drains}
            Publish-CimExerciseTerminal $streams[1] $terminal
            if($terminal.watchdog_completion_pending){
                # F follows successful final durable flush; watchdog still lives until controller exit.
                $watchdog.StandardInput.BaseStream.WriteByte(70);$watchdog.StandardInput.BaseStream.Flush()
            }
        }
    } catch {
        Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_}
        if($streams.Count -ge 2){
            try{Publish-CimExerciseTerminal $streams[1] $terminal}
            catch{
                # No durable publication can be promised. Best effort invalidation; watchdog sees no F.
                try{$streams[1].SetLength(0);$streams[1].Flush($true)}catch{}
            }
        }
    }
    finally {
        foreach($drain in $drains){[Array]::Clear($drain.Buffer,0,$drain.Buffer.Length)}
        # Retain terminal stream until other resource faults can be recorded; preserve first error.
        for($streamIndex=0;$streamIndex -lt $streams.Count;$streamIndex++){
            if($streamIndex -eq 1){continue}
            try{$streams[$streamIndex].Dispose()}catch{Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_}}
        }
        foreach($handle in @($watchdog,$packetRunnerProcess,$controller)){
            if($null -ne $handle){try{$handle.Dispose()}catch{Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_}}}
        }
        if($streams.Count -ge 2){
            if($null -ne $originalError){try{Publish-CimExerciseTerminal $streams[1] $terminal}catch{}}
            try{$streams[1].Dispose()}catch{Set-CimExerciseTerminalFailure $terminal;if($null -eq $originalError){$originalError=$_}}
        }
        $packetRunnerTracked.Clear();$packetRunnerSamples.Clear();$script:actualSnapshot=$null
    }
}
if ($null -ne $originalError) { throw $originalError }
if (-not $terminal.accounting_complete) { throw 'exercise_incomplete' }
}
# Exceptions remain original inside the callable harness; the CLI never echoes raw errors.
try { Invoke-CimReliabilityExercise } catch { exit 1 }
exit 0
