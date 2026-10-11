# Dedicated outer supervisor. Native worker stages are monitored independently
# by the accepted, pinned Windows process-tree monitor. No matching in this host.
#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PacketPath,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedPacketSha256,
    [Parameter(Mandatory=$true)][string]$MonitorReceiptPath,
    [switch]$FinalizeOnly
)
Set-StrictMode -Version 2.0
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
function Resolve-TaskPath([string]$taskValue) {
    if ([IO.Path]::IsPathRooted($taskValue)) {return [IO.Path]::GetFullPath($taskValue)}
    return [IO.Path]::GetFullPath((Join-Path $taskRoot $taskValue))
}
$taskPacketFull=Resolve-TaskPath $PacketPath
if ((Get-FileHash -LiteralPath $taskPacketFull -Algorithm SHA256).Hash -ne $ExpectedPacketSha256) {
    throw 'packet_seal_mismatch'
}
$taskPacket=Get-Content -LiteralPath $taskPacketFull -Raw -Encoding UTF8 | ConvertFrom-Json
function Write-ExclusiveScalar([string]$taskOutput,$taskValue) {
    $taskPayload=[Text.Encoding]::ASCII.GetBytes(($taskValue | ConvertTo-Json -Depth 6 -Compress))
    if ($taskPayload.Length -gt 16384) {throw 'scalar_report_cap'}
    $taskWriter=[IO.File]::Open($taskOutput,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$taskWriter.Write($taskPayload,0,$taskPayload.Length)} finally {$taskWriter.Dispose()}
}
if (-not $FinalizeOnly) {
    foreach($taskField in @('lead_authorized','astra_authorized','launch_allowed','execute')) {
        if ($taskPacket.$taskField -isnot [bool] -or -not $taskPacket.$taskField) {throw 'authority_missing'}
    }
    if ($taskPacket.maximum_invocations -ne 1 -or $taskPacket.retry_allowed -ne $false) {throw 'attempt_contract'}
    if ((Resolve-TaskPath $MonitorReceiptPath) -ne (Resolve-TaskPath $taskPacket.monitor_receipt)) {throw 'monitor_path_mismatch'}
    if ($taskPacket.limits.wall_timeout_seconds -ne 120 -or
        $taskPacket.limits.working_set_threshold_bytes -ne 1073741824) {throw 'supervisor_limits'}
    Write-ExclusiveScalar (Resolve-TaskPath $taskPacket.outer_receipt) ([ordered]@{
        status='attempt_reserved';diagnostic_only=$true;seed_authority=$false;maximum_invocations=1;retry_allowed=$false})
    # Final runtime freshness and startup-path invariants are mandatory BEFORE
    # the worker interpreter starts. The stock monitor does not check these.
    $taskRuntimePath=Resolve-TaskPath $taskPacket.runtime_closure_path
    if ((Get-FileHash -LiteralPath $taskRuntimePath -Algorithm SHA256).Hash -ne $taskPacket.runtime_closure_sha256) {
        throw 'runtime_closure_seal'
    }
    $taskRuntime=Get-Content -LiteralPath $taskRuntimePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (@($taskRuntime.files).Count -ne $taskRuntime.node_count) {throw 'runtime_closure_count'}
    foreach($taskPin in $taskRuntime.files) {
        if ((Get-Item -LiteralPath $taskPin.path).Length -ne $taskPin.bytes -or
            (Get-FileHash -LiteralPath $taskPin.path -Algorithm SHA256).Hash -ne $taskPin.sha256) {throw 'runtime_pin_mismatch'}
    }
    foreach($taskInvariant in $taskPacket.environment_invariants) {
        if (Test-Path -LiteralPath $taskInvariant.path) {throw 'alternate_source_root_present'}
    }
    foreach($taskEnvironment in $taskPacket.startup_environment.PSObject.Properties) {
        if ([Environment]::GetEnvironmentVariable($taskEnvironment.Name,'Process') -cne $taskEnvironment.Value) {
            throw 'startup_environment_changed'
        }
    }
    $taskOutputDirectory=Resolve-TaskPath $taskPacket.report_directory
    if (Test-Path -LiteralPath $taskOutputDirectory) {throw 'report_directory_already_exists'}
    # New-Item refuses an existing directory; no -Force/retry/delete.
    [void](New-Item -ItemType Directory -Path $taskOutputDirectory -ErrorAction Stop)
    $taskMonitor=Resolve-TaskPath $MonitorReceiptPath
    $taskSupervisor=Resolve-TaskPath $taskPacket.supervisor.path
    if ((Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash -ne $taskPacket.supervisor.sha256) {throw 'supervisor_source_seal'}
    # Separate PowerShell child: its exit cannot skip this finalizer. Exactly
    # one monitor call, which makes exactly one worker launch, no retry.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $taskSupervisor -PacketPath $taskPacketFull `
        -ExpectedPacketSha256 $ExpectedPacketSha256 -MonitorReceiptPath $taskMonitor
    $taskMonitorExit=$LASTEXITCODE
} else {
    # Offline scalar-finalizer fixtures only. This branch NEVER launches worker,
    # loads/hashes images, creates directories or validates source runtime.
    if ($taskPacket.schema -ne 'mc-cross-view-finalizer-offline-fixture-v1' -or
        $taskPacket.task -notlike 'synthetic-*' -or @($taskPacket.inputs).Count -ne 0) {throw 'offline_fixture_only'}
    $taskMonitor=Resolve-TaskPath $MonitorReceiptPath
    $taskMonitorExit=0
}
$taskStatus='incomplete';$taskReason='monitor_missing';$taskPeak=0;$taskWall=0
if (Test-Path -LiteralPath $taskMonitor) {
    $taskMonitorValue=Get-Content -LiteralPath $taskMonitor -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskPeak=$taskMonitorValue.peak_aggregate_working_set_bytes
    $taskWall=$taskMonitorValue.monitor_elapsed_seconds
    $taskReason='supervisor_incomplete'
    if ($taskMonitorValue.status -eq 'completed' -and $taskMonitorValue.accounting_complete -eq $true -and
        $taskMonitorValue.root_exit_observed -eq $true -and $taskMonitorValue.root_exit_code -eq 0 -and
        $taskMonitorValue.drain_complete -eq $true -and $taskMonitorExit -eq 0) {
        $taskWorkerPath=Resolve-TaskPath $taskPacket.json_output
        if (Test-Path -LiteralPath $taskWorkerPath) {
            if ((Get-Item -LiteralPath $taskWorkerPath).Length -le 16384) {
                $taskWorker=Get-Content -LiteralPath $taskWorkerPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($taskWorker.status -eq 'complete' -and $taskWorker.seed_authority -eq $false) {
                    $taskStatus='complete';$taskReason='appearance_diagnostic_completed'
                } else {$taskReason='worker_incomplete'}
            } else {$taskReason='worker_scalar_cap'}
        } else {$taskReason='worker_report_missing'}
    } elseif ($taskMonitorValue.reason -eq 'wall_timeout') {$taskReason='wall_timeout'
    } elseif ($taskMonitorValue.reason -eq 'aggregate_working_set_limit') {$taskReason='aggregate_working_set_limit'}
}
$taskScalar=[ordered]@{status=$taskStatus;reason=$taskReason;diagnostic_only=$true;seed_authority=$false;
    reconstruction_gain_claimed=$false;sampled_peak_aggregate_rss_bytes=$taskPeak;supervisor_wall_seconds=$taskWall;
    sampled_supervision=$true;os_containment=$false;maximum_invocations=1;retry_allowed=$false}
foreach($taskOutput in @($taskPacket.project_output,$taskPacket.json_output)) {
    $taskOutputFull=Resolve-TaskPath $taskOutput
    if (-not (Test-Path -LiteralPath $taskOutputFull)) {Write-ExclusiveScalar $taskOutputFull $taskScalar}
}
Write-ExclusiveScalar (Resolve-TaskPath $taskPacket.supervisor_report) $taskScalar
Write-Output ($taskScalar | ConvertTo-Json -Compress)
if ($taskStatus -ne 'complete') {exit 1}
exit 0
