"""Isolated extracted declarations and authored handles; never native CIM/launch paths.

These controls are prepared for independent review, not executed by implementation.
An explicit future pytest invocation may launch only the short extraction host.
"""

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools/run_cim_reliability_exercise.ps1"
WATCHDOG = ROOT / "tools/watch_cim_reliability_exercise.ps1"
RUNNER = ROOT / "tools/run_image_packet.ps1"


def run_fake(tmp_path, body, names=()):
    bootstrap = r"""
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$paths = @('__HARNESS__','__WATCHDOG__')
$names = @('Get-CimExerciseCreation','Bind-CimExerciseHandle','Stop-CimExerciseHandles',
    'Get-CimExerciseMemory','New-CimExerciseDrain','Update-CimExerciseDrain',
    'Write-CimExerciseScalars','Get-CimExerciseSampleProjection','Assert-CimExerciseSampleAccounting','__EXTRA__')
foreach ($path in $paths) {
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
    if($errors.Count -ne 0){throw 'parse_error'}
    foreach($node in @($ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true))) {
        if($node.Name -in $names){ . ([ScriptBlock]::Create($node.Extent.Text)) }
    }
}
function New-ControlHandle([int]$Id,[long]$Bytes,[bool]$Exited=$false) {
    $handle=[pscustomobject]@{Id=$Id;Handle=1;StartTime=[datetime]'2026-10-10T01:00:00Z';
        WorkingSet64=$Bytes;HasExited=$Exited;Killed=$false;Waited=$false;Fault=$false}
    $handle|Add-Member ScriptMethod Refresh {if($this.Fault){throw 'authored_fault'}}
    $handle|Add-Member ScriptMethod Kill {if($this.Fault){throw 'authored_fault'};$this.Killed=$true;$this.HasExited=$true}
    $handle|Add-Member ScriptMethod WaitForExit {param($ms)$this.Waited=$true;return $this.HasExited}
    return $handle
}
function New-ControlTask([int]$Result,[bool]$Complete=$true) {
    $task=[pscustomobject]@{IsCompleted=$Complete;Result=$Result;Fault=$false}
    $task|Add-Member ScriptMethod GetAwaiter {return $this}
    $task|Add-Member ScriptMethod GetResult {if($this.Fault){throw 'authored_fault'};return $this.Result}
    $task|Add-Member ScriptMethod Wait {param($ms)return $this.IsCompleted}
    return $task
}
function New-ControlStream([int[]]$Chunks) {
    $stream=[pscustomobject]@{Chunks=[int[]]@($Chunks|Where-Object{$null -ne $_});Offset=0;Length=[long]0;Payload=$null;Fault=$false;Flushed=$false}
    $stream|Add-Member ScriptMethod ReadAsync {
        param($buffer,$start,$length)
        if($this.Offset -ge $this.Chunks.Count){return (New-ControlTask 0 $false)}
        $count=$this.Chunks[$this.Offset];$this.Offset++
        for($i=0;$i -lt $count;$i++){$buffer[$i]=83}
        return (New-ControlTask $count)
    }
    $stream|Add-Member ScriptMethod Write {
        param($buffer,$start,$length)
        if($this.Fault){throw 'authored_persistence_fault'}
        $this.Length+=$length;$this.Payload=[Text.Encoding]::UTF8.GetString($buffer,$start,$length)
    }
    $stream|Add-Member ScriptMethod Flush {param($durable)$this.Flushed=$durable}
    return $stream
}
"""
    for marker, path in (("__HARNESS__", HARNESS), ("__WATCHDOG__", WATCHDOG)):
        bootstrap = bootstrap.replace(marker, str(path).replace("'", "''"))
    bootstrap = bootstrap.replace("'__EXTRA__'", ",".join(f"'{name}'" for name in names) or "''")
    script = tmp_path / "fake.ps1"
    script.write_text(bootstrap + body, encoding="utf-8")
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_all_prepared_sources_parse_and_no_native_query_in_watchdog(tmp_path):
    result = run_fake(tmp_path, "@{ parsed=$true }|ConvertTo-Json")
    assert result == {"parsed": True}
    assert "Get-CimInstance" not in WATCHDOG.read_text()


@pytest.mark.parametrize("wrong_id,wrong_creation", [(False, False), (True, False), (False, True), (True, True)])
def test_identity_binding_accepts_only_exact_bound_handle(tmp_path, wrong_id, wrong_creation):
    result = run_fake(tmp_path, rf"""
$handle=New-ControlHandle 7 0
$id=7; $creation=Get-CimExerciseCreation $handle
if({'$true' if wrong_id else '$false'}){{$id=8}}
if({'$true' if wrong_creation else '$false'}){{$creation='2026-10-10T01:00:01.000000Z'}}
$refused=$false;$same=$false
try{{$bound=Bind-CimExerciseHandle $handle $id $creation;$same=[Object]::ReferenceEquals($bound,$handle)}}catch{{$refused=$true}}
@{{refused=$refused;same=$same;killed=$handle.Killed}}|ConvertTo-Json
""")
    assert result == {"refused": wrong_id or wrong_creation, "same": not (wrong_id or wrong_creation), "killed": False}


@pytest.mark.parametrize("child,controller,watchdog,exceeded", [
    (0, 0, 0, False), (268435456, 268435456, 268435456, False),
    (268435457, 0, 0, True), (0, 536870912, 1, True),
])
def test_combined_memory_counts_watchdog_and_fixed_boundaries(tmp_path, child, controller, watchdog, exceeded):
    result = run_fake(tmp_path, rf"""
$a=New-ControlHandle 1 {controller};$b=New-ControlHandle 2 {child};$c=New-ControlHandle 3 {watchdog}
Get-CimExerciseMemory $a $b $c|ConvertTo-Json
""")
    assert result == {"controller_and_watchdog_working_set_bytes": controller + watchdog,
                      "child_working_set_bytes": child, "combined_working_set_bytes": child + controller + watchdog,
                      "exceeded": exceeded}


def test_exited_child_counts_zero_but_foreign_negative_memory_refuses(tmp_path):
    result = run_fake(tmp_path, r"""
$a=New-ControlHandle 1 5;$b=New-ControlHandle 2 999 $true;$c=New-ControlHandle 3 7
$zero=(Get-CimExerciseMemory $a $b $c).child_working_set_bytes
$a.WorkingSet64=-1;$failed=$false
try{Get-CimExerciseMemory $a $b $c|Out-Null}catch{$failed=$true}
@{zero=$zero;failed=$failed}|ConvertTo-Json
""")
    assert result == {"zero": 0, "failed": True}


@pytest.mark.parametrize("fault,expired", [(False, False), (True, False), (False, True)])
def test_stop_uses_only_authored_bound_handles_and_clock_budget(tmp_path, fault, expired):
    result = run_fake(tmp_path, rf"""
$child=New-ControlHandle 2 0;$parent=New-ControlHandle 1 0
$child.Fault={'$true' if fault else '$false'}
$clock=[pscustomobject]@{{Elapsed=[pscustomobject]@{{TotalSeconds={5 if expired else 0}}}}}
$complete=Stop-CimExerciseHandles $child $parent $clock 5
@{{complete=$complete;child_killed=$child.Killed;parent_killed=$parent.Killed}}|ConvertTo-Json
""")
    assert result["complete"] is (not fault and not expired)
    assert result["child_killed"] is (not fault and not expired)
    assert result["parent_killed"] is (not expired)


@pytest.mark.parametrize("chunks", [[0], [4, 3, 0]])
def test_async_stream_projection_counts_then_clears_raw_bytes(tmp_path, chunks):
    result = run_fake(tmp_path, rf"""
$stream=New-ControlStream @({','.join(map(str, chunks))})
$drain=New-CimExerciseDrain $stream
Update-CimExerciseDrain $drain
@{{count=$drain.Count;complete=$drain.Complete;failed=$drain.Failed;buffer_nonzero=@($drain.Buffer|Where-Object{{$_ -ne 0}}).Count}}|ConvertTo-Json
""")
    assert result == {"count": sum(chunks), "complete": True, "failed": False, "buffer_nonzero": 0}


def test_stream_overflow_and_read_exception_fail_closed(tmp_path):
    result = run_fake(tmp_path, r"""
$stream=New-ControlStream @(1,0);$drain=New-CimExerciseDrain $stream;$drain.Count=65536
$overflow=$false;try{Update-CimExerciseDrain $drain}catch{$overflow=$true}
$other=New-CimExerciseDrain (New-ControlStream @(1,0));$other.Task.Fault=$true
$fault=$false;try{Update-CimExerciseDrain $other}catch{$fault=$true}
@{overflow=$overflow;overflow_failed=$drain.Failed;fault=$fault;fault_failed=$other.Failed}|ConvertTo-Json
""")
    assert result == {"overflow": True, "overflow_failed": True, "fault": True, "fault_failed": True}


@pytest.mark.parametrize("record", [
    "@{path='sensitive'}", "@{status='foreign_enum'}", "@{status='private/path'}",
    "@{global_row_count=8193}", "@{index=300}", "@{completed_samples=301}",
    "@{elapsed_seconds=86401.0}", "@{elapsed_seconds=[double]::NaN}",
    "@{elapsed_seconds=[double]::PositiveInfinity}", "@{global_row_count=-1}",
    "@{status=@{nested='payload'}}",
    "@{status=$true}", "@{status=1}", "@{accounting_complete=1}",
    "@{global_row_count=1.5}", "@{status=$null}",
    "@{child_exit_code=2147483648}", "@{child_exit_code=-2147483649}",
])
def test_scalar_schema_rejects_payload_foreign_states_nonfinite_and_caps(tmp_path, record):
    result = run_fake(tmp_path, rf"""
$stream=New-ControlStream @();$failed=$false
try{{Write-CimExerciseScalars $stream {record} 65536}}catch{{$failed=$true}}
@{{failed=$failed;bytes=$stream.Length}}|ConvertTo-Json
""")
    assert result == {"failed": True, "bytes": 0}


def test_scalar_valid_null_codes_durable_flush_and_byte_cap(tmp_path):
    result = run_fake(tmp_path, r"""
$stream=New-ControlStream @()
$record=[ordered]@{index=0;status='returned';global_row_count=0;hresult=$null;native_error_code=1;
    query_duration_seconds=0.25;query_duration_state='measured';accounting_complete=$true}
Write-CimExerciseScalars $stream $record 65536
$parsed=$stream.Payload|ConvertFrom-Json
$cap=$false;try{Write-CimExerciseScalars $stream $record 1}catch{$cap=$true}
@{record=$parsed;flushed=$stream.Flushed;cap=$cap}|ConvertTo-Json
""")
    assert result["record"]["hresult"] is None
    assert result["record"]["global_row_count"] == 0
    assert result["flushed"] is True and result["cap"] is True


@pytest.mark.parametrize("code", [-2147483648, -1, 0, 2147483647])
def test_signed_exit_code_is_preserved_in_terminal_receipt(tmp_path, code):
    result = run_fake(tmp_path, rf"""
$stream=New-ControlStream @()
Write-CimExerciseScalars $stream @{{status='exercise_incomplete';child_exit_code={code};accounting_complete=$false}} 65536
$stream.Payload|ConvertFrom-Json|ConvertTo-Json
""")
    assert result["child_exit_code"] == code
    assert result["status"] == "exercise_incomplete"


@pytest.mark.parametrize("offset,refused,remaining", [(0, False, 450), (20, False, 430), (449, False, 1),
                                                      (450, True, None), (451, True, None), (-1, True, None)])
def test_watchdog_deadline_includes_startup_and_preflight(tmp_path, offset, refused, remaining):
    result = run_fake(tmp_path, rf"""
$origin=[datetime]'2026-10-10T01:00:00Z'
$deadline=$origin.ToUniversalTime().AddSeconds(450).ToString('o')
$failed=$false;$remaining=$null
try{{$remaining=Get-CimExerciseRemainingDeadline $deadline ($origin.AddSeconds({offset}))}}catch{{$failed=$true}}
@{{failed=$failed;remaining=$remaining}}|ConvertTo-Json
""", names=("Get-CimExerciseRemainingDeadline",))
    assert result == {"failed": refused, "remaining": remaining}


def test_persistence_exception_is_original_and_not_success(tmp_path):
    result = run_fake(tmp_path, r"""
$stream=New-ControlStream @();$stream.Fault=$true;$failed=$false
try{Write-CimExerciseScalars $stream @{status='returned'} 65536}catch{$failed=$true}
@{failed=$failed;flushed=$stream.Flushed;bytes=$stream.Length}|ConvertTo-Json
""")
    assert result == {"failed": True, "flushed": False, "bytes": 0}


@pytest.mark.parametrize("mode", ["good", "missing", "bad_duration", "bad_gap"])
def test_sample_projection_requires_complete_same_pass_timing(tmp_path, mode):
    result = run_fake(tmp_path, rf"""
$packetRunnerReceipt=[ordered]@{{accounting_complete=$true;process_snapshot_telemetry=[ordered]@{{
 query_duration_seconds=0.1;query_duration_state='measured';query_start_gap_seconds=1.0;query_start_gap_state='measured'}}}}
$packetRunnerTracked=[ordered]@{{a=1}};$packetRunnerProcess=New-ControlHandle 1 0
$sample=[ordered]@{{timestamp_utc='2026-10-10T01:00:00.000000Z';elapsed_seconds=1.0;live_processes=1}}
$memory=[ordered]@{{child_working_set_bytes=0;controller_and_watchdog_working_set_bytes=0;combined_working_set_bytes=0}}
if('{mode}' -eq 'missing'){{$packetRunnerReceipt.process_snapshot_telemetry=$null}}
if('{mode}' -eq 'bad_duration'){{$packetRunnerReceipt.process_snapshot_telemetry.query_duration_state='unavailable'}}
if('{mode}' -eq 'bad_gap'){{$packetRunnerReceipt.process_snapshot_telemetry.query_start_gap_state='unavailable'}}
$failed=$false;$projection=$null
try{{$projection=Get-CimExerciseSampleProjection $sample 1 10 $memory}}catch{{$failed=$true}}
@{{failed=$failed;projection=$projection}}|ConvertTo-Json -Depth 6
""")
    assert result["failed"] is (mode != "good")
    if mode == "good":
        assert result["projection"]["global_row_count"] == 10
        assert result["projection"]["status"] == "returned"
        assert "processes" not in result["projection"]


@pytest.mark.parametrize("mode", ["return", "throw", "partial", "overflow", "cancel"])
def test_actual_snapshot_wrapper_one_call_exact_objects_or_exception(tmp_path, mode):
    result = run_fake(tmp_path, rf"""
$script:calls=0;$script:original=New-Object InvalidOperationException('authored private detail')
if('{mode}' -eq 'cancel'){{$script:original=New-Object OperationCanceledException('authored cancellation')}}
$script:row=[pscustomobject]@{{ProcessId=1;ParentProcessId=0;CreationDate=[datetime]'2026-10-10T01:00:00Z'}}
$script:actualSnapshot={{
 $script:calls++
 if('{mode}' -in @('throw','cancel')){{throw $script:original}}
 if('{mode}' -eq 'overflow'){{for($i=0;$i -lt 8193;$i++){{$script:row}};return}}
 $script:row
 if('{mode}' -eq 'partial'){{throw $script:original}}
}}
$failed=$false;$same=$false;$sameException=$false;$rows=$null
try{{$rows=@(Get-PacketProcessSnapshot);$same=[Object]::ReferenceEquals($rows[0],$script:row)}}
catch{{$failed=$true;$sameException=[Object]::ReferenceEquals($_.Exception,$script:original)}}
@{{calls=$script:calls;failed=$failed;same=$same;same_exception=$sameException;global_rows=$script:globalRows}}|ConvertTo-Json
""", names=("Get-PacketProcessSnapshot",))
    assert result["calls"] == 1
    assert result["failed"] is (mode != "return")
    if mode == "return":
        assert result["same"] is True and result["global_rows"] == 1
    elif mode in {"throw", "partial", "cancel"}:
        assert result["same_exception"] is True and result["global_rows"] is None


@pytest.mark.parametrize("exited,live,global_rows,accounted,refused", [
    (False, 1, 2, True, False), (False, 0, 2, True, True),
    (True, 0, 2, True, False), (True, 0, 0, True, True),
    (False, 1, 2, False, True), (False, 2, 2, True, True),
])
def test_live_child_empty_post_exit_and_incomplete_accounting_are_distinct(
    tmp_path, exited, live, global_rows, accounted, refused,
):
    result = run_fake(tmp_path, rf"""
$process=New-ControlHandle 1 0 {'$true' if exited else '$false'}
$sample=[ordered]@{{live_processes={live}}};$receipt=[ordered]@{{accounting_complete={'$true' if accounted else '$false'}}}
$refused=$false
try{{Assert-CimExerciseSampleAccounting $sample $process $receipt {global_rows}}}catch{{$refused=$true}}
@{{refused=$refused}}|ConvertTo-Json
""")
    assert result == {"refused": refused}


def test_execution_gate_false_ends_before_any_launch_or_query(tmp_path):
    result = run_fake(tmp_path, r"""
$Execute=$false;$calls=0
function New-CimExerciseProcess {$script:calls++;throw 'unexpected_launch'}
function Get-CimInstance {$script:calls++;throw 'unexpected_query'}
$failed=$false;try{Invoke-CimReliabilityExercise}catch{$failed=$true}
@{failed=$failed;calls=$calls}|ConvertTo-Json
""", names=("Invoke-CimReliabilityExercise",))
    assert result == {"failed": True, "calls": 0}


def test_static_lifecycle_brackets_single_query_and_readiness_no_packet_invocation():
    source = HARNESS.read_text()
    assert source.index("$terminal.watchdog_ready=$true") < source.index("$sample=Get-PacketSample")
    assert source.count("$sample=Get-PacketSample") == 1
    assert "$rows=@(& $script:actualSnapshot)" in source
    assert "$script:globalRows=$null" in source
    assert "for ($index=0;$index -lt 300;$index++)" in source
    assert "Stop-PacketTree" in source and "if ($null -ne $originalError) { throw $originalError }" in source
    assert "try { Invoke-CimReliabilityExercise } catch { exit 1 }" in source
    assert "ReadAllText($full) | ConvertFrom-Json" in WATCHDOG.read_text()
    assert "-File',$watchdogPath" in source
    assert "-File','run_image_packet" not in source
    assert "Get-CimInstance" not in source  # Native command comes only from pinned declarations.


def test_import_declaration_keeps_typed_header_and_exact_original_body():
    source = HARNESS.read_text()
    assert "$nodes[0].Extent.Text.Replace(('function ' + $name), ('function script:' + $name))" in source
    assert ". ([ScriptBlock]::Create($declaration))" in source
    assert "Body.Extent.Text.Substring" not in source
    accepted = RUNNER.read_text()
    assert accepted.count("Get-CimInstance -ClassName Win32_Process -Property ProcessId,ParentProcessId,CreationDate -OperationTimeoutSec 1 -ErrorAction Stop") == 1


def test_real_declaration_import_preserves_query_arguments_and_fake_sampler_scope(tmp_path):
    body = r"""
Import-CimExerciseFunctions '__RUNNER__'
$packetRunnerClock=[pscustomobject]@{Elapsed=[pscustomobject]@{TotalSeconds=[double]1.0}}
$packetRunnerTracked=[ordered]@{};$packetRunnerSamples=New-Object 'Collections.Generic.List[object]'
$script:packetRunnerPreviousSnapshotStartSeconds=$null
$packetRunnerReceipt=[ordered]@{accounting_complete=$true;discovery_failure=$null;
    process_snapshot_telemetry=$null;peak_aggregate_working_set_bytes=[long]0}
$handle=New-ControlHandle 7 12
$creation=Get-PacketCreation $handle.StartTime
Register-PacketProcess 7 $creation 0 '' 0 $handle
$script:calls=0;$script:argsSeen=$null;$writes=0
function Write-PacketReceipt {$script:writes++}
function Get-CimInstance {
 param($ClassName,$Property,$OperationTimeoutSec,$ErrorAction)
 $script:calls++;$script:argsSeen=@($ClassName)+@($Property)+@($OperationTimeoutSec,$ErrorAction)
 [pscustomobject]@{ProcessId=7;ParentProcessId=0;CreationDate=$handle.StartTime}
}
$sample=Get-PacketSample
@{calls=$script:calls;args=$script:argsSeen;live=$sample.live_processes;
    bytes=$sample.aggregate_working_set_bytes;count=$packetRunnerSamples.Count;writes=$script:writes}|ConvertTo-Json
""".replace("__RUNNER__", str(RUNNER).replace("'", "''"))
    result = run_fake(tmp_path, body, names=("Import-CimExerciseFunctions",))
    assert result == {"calls": 1, "args": ["Win32_Process", "ProcessId", "ParentProcessId", "CreationDate", 1, "Stop"],
                      "live": 1, "bytes": 12, "count": 1, "writes": 1}


LIFECYCLE_FAKE = r"""
$script:mode='__MODE__';$script:lifeSeconds=[double]20
$script:watchReadyRead=$false
$GatePath='authored-gate.json';$ExpectedGateSha256=('0'*64)
$script:queryCalls=0;$script:launchCalls=0;$script:inCleanup=$false
$script:original=New-Object InvalidOperationException('private authored first failure')
if($script:mode -eq 'cancel'){$script:original=New-Object OperationCanceledException('private cancellation')}
$script:secondary=New-Object InvalidOperationException('private secondary failure')
$script:trace=New-Object 'Collections.Generic.List[object]'
$script:files=[ordered]@{};$script:deadlineSeen=$null
function New-LifeClock([double]$Origin) {
 $elapsed=[pscustomobject]@{Origin=$Origin}
 $elapsed|Add-Member ScriptProperty TotalSeconds {[double]($script:lifeSeconds-$this.Origin)}
 return [pscustomobject]@{Elapsed=$elapsed}
}
function New-CimExerciseClock {return (New-LifeClock $script:lifeSeconds)}
function Wait-CimExercise([int]$Milliseconds) {
 $script:lifeSeconds += [double]$Milliseconds/1000
 if($null -ne $script:child -and $script:lifeSeconds -ge 240){$script:child.HasExited=$true;$script:child.ExitCode=0}
}
function New-LifeStream([string]$Name) {
 $stream=New-ControlStream @()
 $stream|Add-Member NoteProperty Name $Name
 $stream|Add-Member NoteProperty Position ([long]0)
 $stream|Add-Member NoteProperty FlushCount 0
 $stream|Add-Member NoteProperty Closed $false
 $stream|Add-Member ScriptMethod Dispose {$this.Closed=$true}
 $stream|Add-Member ScriptMethod SetLength {param($n)$this.Length=[long]$n;$this.Payload=$null}
 $stream|Add-Member ScriptMethod ReadAsync -Force {
  param($buffer,$start,$length)
  $task=New-ControlTask 0 $false
  if($this.Name -eq 'watch_out'){
   $byte=if(-not $script:watchReadyRead){82}else{65}
   $script:watchReadyRead=$true;$buffer[$start]=$byte
   $task=New-ControlTask 1 ($script:mode -ne 'readiness')
  }elseif($this.Name -in @('child_out','child_err')){
   $task.PSObject.Properties.Remove('IsCompleted')
   $task|Add-Member ScriptProperty IsCompleted {$script:child.HasExited}
  }
  return $task
 }
 $stream|Add-Member ScriptMethod WriteByte {
  param($value)
  $script:trace.Add([ordered]@{event='protocol';value=$value;watch_alive=(-not $script:watch.HasExited)})
  if($value -eq 68 -and $script:mode -in @('protocol','query_protocol')){throw $script:secondary}
 }
 $stream|Add-Member ScriptMethod Flush -Force {
  param($durable)
  $this.FlushCount++
  if($this.Name -eq '.terminal.json'){
   $script:trace.Add([ordered]@{event='terminal_flush';watch_alive=(-not $script:watch.HasExited);
     status=($this.Payload|ConvertFrom-Json).status;flush_count=$this.FlushCount})
   if(($script:mode -eq 'persistence' -and $this.FlushCount -eq 1) -or
      ($script:mode -eq 'final_persistence' -and $this.FlushCount -eq 2)){throw $script:original}
  }
  $this.Flushed=$durable
 }
 return $stream
}
function New-LifeHandle([int]$Id,[string]$Role) {
 $handle=New-ControlHandle $Id 12
 $handle|Add-Member NoteProperty ExitCode 0
 $handle|Add-Member NoteProperty StandardOutput ([pscustomobject]@{BaseStream=(New-LifeStream ($Role+'_out'))})
 $handle|Add-Member NoteProperty StandardError ([pscustomobject]@{BaseStream=(New-LifeStream ($Role+'_err'))})
 $handle|Add-Member NoteProperty StandardInput ([pscustomobject]@{BaseStream=(New-LifeStream ($Role+'_in'))})
 $handle|Add-Member ScriptMethod Dispose {}
 return $handle
}
$script:child=$null;$script:watch=$null;$script:controller=New-LifeHandle 1 'controller'
function Get-CimExerciseController {return $script:controller}
function Open-CimExerciseOutput([string]$Path) {
 $suffix=@('.samples.jsonl','.terminal.json','.stdout.log','.stderr.log','.watchdog.json')|
   Where-Object{$Path.EndsWith($_)}
 $stream=New-LifeStream $suffix;$script:files[$suffix]=$stream;return $stream
}
function New-CimExerciseProcess([string]$Executable,[string[]]$Arguments,[string]$WorkingDirectory) {
 $script:launchCalls++
 if($script:launchCalls -eq 1){$script:child=New-LifeHandle 2 'child';return $script:child}
 if($script:launchCalls -ne 2){throw 'unexpected_additional_launch'}
 $offset=[Array]::IndexOf($Arguments,'-DeadlineUtc')
 $script:deadlineSeen=$Arguments[$offset+1]
 $script:watch=New-LifeHandle 3 'watch';return $script:watch
}
function Get-PacketCreation([datetime]$Value){return $Value.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ')}
function Register-PacketProcess($Id,$Creation,$Parent,$ParentCreation,$Depth,$Handle){$packetRunnerTracked['child']=$Handle}
function Get-PacketProcessSnapshot {
 $script:queryCalls++
 $packetRunnerReceipt.process_snapshot_telemetry=[ordered]@{query_duration_seconds=0.1;query_duration_state='measured';
  query_start_gap_seconds=1.0;query_start_gap_state='measured'}
 if($script:queryCalls -eq 2 -and $script:mode -in @('query','partial','cancel','query_protocol')){
  if($script:mode -ne 'cancel'){$packetRunnerReceipt.discovery_failure=[ordered]@{hresult=-2146233088;native_error_code=1;status_code=1}}
  if($script:mode -eq 'partial'){[pscustomobject]@{ProcessId=2}}
  throw $script:original
 }
 [pscustomobject]@{ProcessId=2}
}
function Get-PacketSample {
 $rows=@(Get-PacketProcessSnapshot)
 $sample=[ordered]@{timestamp_utc=[DateTime]::UtcNow.ToString('o');elapsed_seconds=$packetRunnerClock.Elapsed.TotalSeconds;
   live_processes=([int](-not $script:child.HasExited))}
 $packetRunnerSamples.Add($sample);return $sample
}
function Stop-PacketTree {
 $script:inCleanup=$true
 if(-not $script:child.HasExited){$script:child.HasExited=$true;$script:child.ExitCode=-1}
 $packetRunnerReceipt.cleanup_attempted=$true
 $packetRunnerReceipt.cleanup_elapsed_seconds=0.0;$packetRunnerReceipt.cleanup_overshoot_seconds=0.0
 if($script:mode -eq 'cleanup'){throw $script:original}
 $packetRunnerReceipt.cleanup_complete=$true
}
$script:realDrain=${function:Update-CimExerciseDrain}
function Update-CimExerciseDrain($Drain) {
 if($script:inCleanup -and $script:mode -eq 'drain' -and $Drain.Stream.Name -eq 'child_out'){
  $Drain.Count=7;throw $script:original
 }
 & $script:realDrain $Drain
}
$failed=$false;$same=$false;$captured_error=$null
$started=[datetime]'2026-10-10T01:00:00Z'
$gate=[pscustomobject]@{output_base='planned';powershell_path='authored-fake.exe'}
try{Invoke-CimExerciseLifecycle $gate '__ROOT__' 'authored-watch.ps1' (New-LifeClock 0) $started}
catch{
 $failed=$true;$errorValue=$_.Exception
 $captured_error=[ordered]@{type=$errorValue.GetType().FullName;hresult=[int]$errorValue.HResult}
 for($depth=0;$depth -lt 8 -and $null -ne $errorValue;$depth++){
  if([Object]::ReferenceEquals($errorValue,$script:original)){$same=$true;break}
  $errorValue=$errorValue.InnerException
 }
}
$terminal=$null
if($null -ne $script:files['.terminal.json'].Payload){$terminal=$script:files['.terminal.json'].Payload|ConvertFrom-Json}
@{failed=$failed;original_preserved=$same;error=$captured_error;queries=$script:queryCalls;launches=$script:launchCalls;
  terminal=$terminal;trace=@($script:trace.ToArray());deadline=$script:deadlineSeen}|ConvertTo-Json -Depth 10
"""


@pytest.mark.parametrize("mode", ["normal", "readiness", "query", "partial", "cancel", "cleanup", "drain",
                                  "protocol", "query_protocol", "persistence", "final_persistence"])
def test_actual_controller_lifecycle_is_fake_only_and_fail_closed(tmp_path, mode):
    body = LIFECYCLE_FAKE.replace("__MODE__", mode).replace("__ROOT__", str(tmp_path).replace("'", "''"))
    result = run_fake(tmp_path, body, names=("Invoke-CimExerciseLifecycle", "Update-CimExerciseKnownStreamCounts",
                                           "Set-CimExerciseTerminalFailure", "Publish-CimExerciseTerminal",
                                           "Confirm-CimExerciseWatchdog"))
    terminal = result["terminal"]
    assert result["launches"] == 2, result["error"]  # Authored factory calls only, not OS launches.
    assert result["deadline"] == "2026-10-10T01:07:30.0000000Z"
    assert terminal is not None
    assert all(event["watch_alive"] for event in result["trace"])
    if mode == "normal":
        assert result["failed"] is False
        assert result["queries"] == 300
        assert terminal["completed_samples"] == 300 and terminal["last_acknowledged_sample"] == 299
        assert terminal["status"] == "observed_no_failure" and terminal["accounting_complete"] is True
        assert terminal["watchdog_completion_pending"] is True
        events = result["trace"]
        final_flush = next(index for index, event in enumerate(events)
                           if event["event"] == "terminal_flush" and event["status"] == "observed_no_failure")
        final_marker = next(index for index, event in enumerate(events)
                            if event["event"] == "protocol" and event["value"] == 70)
        assert final_flush < final_marker
    else:
        assert result["failed"] is True
        assert terminal["status"] != "observed_no_failure" and terminal["accounting_complete"] is False
        if mode in {"query", "partial", "cancel", "query_protocol"}:
            assert result["queries"] == 2 and terminal["completed_samples"] == 1
            assert terminal["last_acknowledged_sample"] == 0
            assert result["original_preserved"] is True
        if mode == "readiness":
            assert result["queries"] == 0 and terminal["child_exit_code"] == -1
        if mode == "drain":
            assert terminal["stdout_bytes"] == 7 and terminal["drain_complete"] is False
        if mode in {"cleanup", "drain", "persistence", "final_persistence"}:
            assert result["original_preserved"] is True
        if mode == "final_persistence":
            assert not any(event["event"] == "protocol" and event["value"] == 70 for event in result["trace"])


def test_watchdog_stays_active_after_ack_and_controller_clock_precedes_preflight():
    controller = HARNESS.read_text()
    watchdog = WATCHDOG.read_text()
    assert controller.index("$script:cimExerciseClock=") < controller.index("$gate=Assert-CimExerciseGate")
    assert watchdog.index("$script:cimWatchClock=") < watchdog.index("$gate = Assert-CimExerciseGate")
    assert "'-DeadlineUtc',$deadlineUtc" in controller
    assert "$StartedUtc.ToUniversalTime().AddSeconds(450)" in controller
    protocol = watchdog[watchdog.index("function Update-CimExerciseWatchdogProtocol"):watchdog.index("function Bind-CimExerciseHandle")]
    assert "break" not in protocol
    assert "$Output.WriteByte(65)" in protocol
    assert "$record.controller_terminal_committed" in watchdog


@pytest.mark.parametrize("acknowledged,byte,count,refused,committed", [
    (False, 68, 1, False, False), (True, 70, 1, False, True),
    (False, 70, 1, True, False), (True, 68, 1, True, False),
    (False, 0, 0, True, False), (False, 99, 1, True, False),
])
def test_actual_watchdog_protocol_never_releases_on_pending_terminal_ack(
    tmp_path, acknowledged, byte, count, refused, committed,
):
    result = run_fake(tmp_path, rf"""
$record=[ordered]@{{controller_terminal_acknowledged={'$true' if acknowledged else '$false'};controller_terminal_committed=$false}}
$task=New-ControlTask {count};$buffer=New-Object byte[] 1;$buffer[0]={byte}
$output=[pscustomobject]@{{Bytes=(New-Object 'Collections.Generic.List[int]')}}
$output|Add-Member ScriptMethod WriteByte {{param($value)$this.Bytes.Add($value)}}
$output|Add-Member ScriptMethod Flush {{}}
$inputStream=New-ControlStream @()
$failed=$false
try{{$task=Update-CimExerciseWatchdogProtocol $record $task $buffer $output $inputStream}}catch{{$failed=$true}}
@{{failed=$failed;acknowledged=$record.controller_terminal_acknowledged;committed=$record.controller_terminal_committed;bytes=@($output.Bytes.ToArray())}}|ConvertTo-Json
""", names=("Update-CimExerciseWatchdogProtocol",))
    assert result["failed"] is refused and result["committed"] is committed
    if not refused and not acknowledged:
        assert result["acknowledged"] is True and result["bytes"] == [65]


IMPORTED_SAMPLER_INTEGRATION_FAKE = r"""
$script:integrationMode='__MODE__'
$script:integrationSeconds=[double]20
$script:integrationLaunchSeconds=$null;$script:integrationExpiry=$null
$script:integrationChild=$null;$script:integrationWatch=$null
$script:integrationReadyReads=0;$script:integrationFactories=0
$script:integrationProviderCalls=0;$script:integrationBindingCalls=0
$script:integrationMemoryCount=$null;$script:integrationMemoryTail=$null
$script:integrationTrackedCount=$null;$script:integrationTrackedIdentityValid=$null
$script:integrationFirstError=$null;$script:integrationStage=$null
$script:integrationAccountingReason=$null;$script:integrationNaturalExitSeconds=$null
$script:integrationError=New-Object InvalidOperationException('authored provider failure')
$script:integrationFiles=[ordered]@{}
$script:integrationTrace=New-Object 'Collections.Generic.List[object]'
$script:integrationRecords=New-Object 'Collections.Generic.List[object]'
$script:integrationIndices=New-Object 'Collections.Generic.List[int]'
$script:integrationArguments=New-Object 'Collections.Generic.List[object]'
$GatePath='authored-gate.json';$ExpectedGateSha256=('0'*64)
function New-IntegrationClock([double]$Origin) {
 $elapsed=[pscustomobject]@{Origin=$Origin}
 $elapsed|Add-Member ScriptProperty TotalSeconds {[double]($script:integrationSeconds-$this.Origin)}
 return [pscustomobject]@{Elapsed=$elapsed}
}
function New-CimExerciseClock {return (New-IntegrationClock $script:integrationSeconds)}
function Wait-CimExercise([int]$Milliseconds) {
 $script:integrationSeconds += [double]$Milliseconds/1000
 if($null -ne $script:integrationChild -and -not $script:integrationChild.HasExited -and
    $script:integrationSeconds -ge $script:integrationExpiry){
  $script:integrationChild.HasExited=$true;$script:integrationChild.ExitCode=0
  $script:integrationNaturalExitSeconds=$script:integrationSeconds
 }
}
function New-IntegrationStream([string]$Name) {
 $stream=New-ControlStream @()
 $stream|Add-Member NoteProperty Name $Name
 $stream|Add-Member NoteProperty Position ([long]0)
 $stream|Add-Member ScriptMethod Dispose {}
 $stream|Add-Member ScriptMethod SetLength {param($n)$this.Length=[long]$n;$this.Payload=$null}
 $stream|Add-Member ScriptMethod ReadAsync -Force {
  param($buffer,$start,$length)
  $task=New-ControlTask 0 $false
  if($this.Name -eq 'watch_out'){
   $script:integrationReadyReads++
   $buffer[$start]=if($script:integrationReadyReads -eq 1){82}else{65}
   if($script:integrationReadyReads -eq 1){$script:integrationTrace.Add(@{event='ready'})}
   return (New-ControlTask 1)
  }
  if($this.Name -in @('child_out','child_err')){
   $task.PSObject.Properties.Remove('IsCompleted')
   $task|Add-Member ScriptProperty IsCompleted {$script:integrationChild.HasExited}
  }
  return $task
 }
 $stream|Add-Member ScriptMethod WriteByte {
  param($value)
  $script:integrationTrace.Add(@{event='protocol';value=[int]$value;
    watch_alive=(-not $script:integrationWatch.HasExited)})
 }
 $stream|Add-Member ScriptMethod Flush -Force {
  param($durable)
  $this.Flushed=$durable
  if($this.Name -eq '.samples.jsonl'){
   $record=$this.Payload|ConvertFrom-Json
   $script:integrationRecords.Add($record)
   $script:integrationIndices.Add([int]$record.index)
   $script:integrationTrace.Add(@{event='sample_flush';index=[int]$record.index;durable=[bool]$durable})
  }elseif($this.Name -eq '.terminal.json'){
   $script:integrationTrace.Add(@{event='terminal_flush';status=($this.Payload|ConvertFrom-Json).status;
     durable=[bool]$durable;watch_alive=(-not $script:integrationWatch.HasExited)})
  }
 }
 return $stream
}
function New-IntegrationHandle([int]$Id,[string]$Role) {
 $handle=New-ControlHandle $Id 12
 $handle|Add-Member NoteProperty ExitCode 0
 $handle|Add-Member NoteProperty StandardOutput ([pscustomobject]@{BaseStream=(New-IntegrationStream ($Role+'_out'))})
 $handle|Add-Member NoteProperty StandardError ([pscustomobject]@{BaseStream=(New-IntegrationStream ($Role+'_err'))})
 $handle|Add-Member NoteProperty StandardInput ([pscustomobject]@{BaseStream=(New-IntegrationStream ($Role+'_in'))})
 $handle|Add-Member ScriptMethod Dispose {}
 return $handle
}
$script:integrationController=New-IntegrationHandle 1 'controller'
function Get-CimExerciseController {return $script:integrationController}
function Open-CimExerciseOutput([string]$Path) {
 $suffix=@('.samples.jsonl','.terminal.json','.stdout.log','.stderr.log','.watchdog.json')|
  Where-Object{$Path.EndsWith($_)}
 $stream=New-IntegrationStream $suffix;$script:integrationFiles[$suffix]=$stream;return $stream
}
function New-CimExerciseProcess([string]$Executable,[string[]]$Arguments,[string]$WorkingDirectory) {
 $script:integrationFactories++
 if($script:integrationFactories -eq 1){
  $script:integrationLaunchSeconds=$script:integrationSeconds
  $script:integrationExpiry=$script:integrationLaunchSeconds+240
  $script:integrationChild=New-IntegrationHandle 2 'child';return $script:integrationChild
 }
 if($script:integrationFactories -ne 2){throw 'unexpected_fake_factory'}
 $script:integrationWatch=New-IntegrationHandle 3 'watch';return $script:integrationWatch
}
Import-CimExerciseFunctions '__RUNNER__'
# Test-copy binding seam only; every other original sampler byte remains exact.
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile('__RUNNER__',[ref]$tokens,[ref]$errors)
if($errors.Count -ne 0){throw 'source_parse'}
$nodes=@($ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and
 $n.Name -eq 'Get-PacketSample'},$true))
if($nodes.Count -ne 1){throw 'source_sampler_count'}
$original=$nodes[0].Extent.Text
$needle='[Diagnostics.Process]::GetProcessById($entry.Pid)'
$replacement='Invoke-CimIntegrationRejectingBinding $entry.Pid'
$replacementCount=[regex]::Matches($original,[regex]::Escape($needle)).Count
if($replacementCount -ne 1 -or $original.Contains($replacement)){throw 'source_binding_count'}
$patched=$original.Replace($needle,$replacement)
$declaration=$patched.Replace('function Get-PacketSample','function script:Get-PacketSample')
$inverse=$declaration.Replace('function script:Get-PacketSample','function Get-PacketSample').Replace($replacement,$needle)
$equivalent=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($inverse)) -ceq
 [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($original))
if(-not $equivalent){throw 'source_equivalence'}
. ([ScriptBlock]::Create($declaration))
function Invoke-CimIntegrationRejectingBinding([int]$Id) {
 $script:integrationBindingCalls++;throw 'unexpected_native_binding_seam'
}
# Must override AFTER import: the real cleanup also contains a native binding.
function Stop-PacketTree {
 $script:integrationMemoryCount=$packetRunnerSamples.Count
 $script:integrationMemoryTail=@($packetRunnerSamples|Where-Object{$_.live_processes -eq 0}).Count
 $script:integrationTrackedCount=$packetRunnerTracked.Count
 $tracked=@($packetRunnerTracked.Values)
 $script:integrationTrackedIdentityValid=($tracked.Count -eq 1 -and
   $tracked[0].Pid -eq $script:integrationChild.Id -and
   $tracked[0].Creation -ceq (Get-PacketCreation $script:integrationChild.StartTime) -and
   [Object]::ReferenceEquals($tracked[0].Handle,$script:integrationChild))
 if(-not $script:integrationChild.HasExited){
  $script:integrationChild.Kill();$script:integrationChild.ExitCode=-1
 }
 $packetRunnerReceipt.cleanup_attempted=$true;$packetRunnerReceipt.cleanup_complete=$true
 $packetRunnerReceipt.cleanup_elapsed_seconds=0.0;$packetRunnerReceipt.cleanup_overshoot_seconds=0.0
}
function Get-CimInstance {
 param($ClassName,$Property,$OperationTimeoutSec,$ErrorAction)
 $script:integrationProviderCalls++
 $script:integrationArguments.Add(@{class=$ClassName;properties=@($Property);
   timeout=$OperationTimeoutSec;action=$ErrorAction})
 $script:integrationTrace.Add(@{event='provider';ready=($script:integrationReadyReads -eq 1)})
 if($script:integrationMode -eq 'throw'){
  $script:integrationFirstError=$script:integrationError;throw $script:integrationError
 }
 if($script:integrationMode -eq 'empty'){return}
 [pscustomobject]@{ProcessId=1;ParentProcessId=0;CreationDate=$script:integrationController.StartTime}
 if(-not $script:integrationChild.HasExited){
  $creation=$script:integrationChild.StartTime
  if($script:integrationMode -eq 'identity_mismatch'){$creation=$creation.AddSeconds(1)}
  [pscustomobject]@{ProcessId=2;ParentProcessId=0;CreationDate=$creation}
 }
}
# Exact-forwarding diagnostics stay in the fake host, outside production schemas.
$script:integrationSampler=${function:Get-PacketSample}
function Get-PacketSample {
 try { & $script:integrationSampler }
 catch {
  $script:integrationStage='sampler'
  if($null -eq $script:integrationFirstError){$script:integrationFirstError=$_.Exception}
  throw
 }
}
$script:integrationAccounting=${function:Assert-CimExerciseSampleAccounting}
function Assert-CimExerciseSampleAccounting($Sample,$Process,$Receipt,[int]$GlobalRows) {
 try { & $script:integrationAccounting $Sample $Process $Receipt $GlobalRows }
 catch {
  $script:integrationStage='accounting'
  if($_.Exception.Message -ceq 'accounting_failed'){$script:integrationAccountingReason='accounting_failed'}
  if($null -eq $script:integrationFirstError){$script:integrationFirstError=$_.Exception}
  throw
 }
}
$failed=$false;$same=$false;$errorCode=$null
$gate=[pscustomobject]@{output_base='planned';powershell_path='authored-fake.exe'}
try{Invoke-CimExerciseLifecycle $gate '__ROOT__' 'authored-watch.ps1' (New-IntegrationClock 0) ([datetime]'2026-10-10T01:00:00Z')}
catch {
 $failed=$true;$failure=$_.Exception
 for($depth=0;$depth -lt 16 -and $null -ne $failure;$depth++){
  if([Object]::ReferenceEquals($failure,$script:integrationFirstError)){$same=$true;break}
  $failure=$failure.InnerException
 }
 if($null -ne $script:integrationFirstError){$errorCode=[int]$script:integrationFirstError.HResult}
}
$terminal=$script:integrationFiles['.terminal.json'].Payload|ConvertFrom-Json
@{failed=$failed;original_preserved=$same;error_hresult=$errorCode;
 provider_hresult=[int]$script:integrationError.HResult;stage=$script:integrationStage;
 accounting_reason=$script:integrationAccountingReason;provider_calls=$script:integrationProviderCalls;
 factories=$script:integrationFactories;binding_calls=$script:integrationBindingCalls;
 replacement_count=$replacementCount;source_equivalent=$equivalent;
 memory_count=$script:integrationMemoryCount;memory_tail=$script:integrationMemoryTail;
 wrapper_rows=$script:globalRows;tracked_count=$script:integrationTrackedCount;
 tracked_identity_valid=$script:integrationTrackedIdentityValid;
 launch_seconds=$script:integrationLaunchSeconds;expiry_seconds=$script:integrationExpiry;
 natural_exit_seconds=$script:integrationNaturalExitSeconds;terminal=$terminal;
 records=@($script:integrationRecords.ToArray());indices=@($script:integrationIndices.ToArray());
 arguments=@($script:integrationArguments.ToArray());trace=@($script:integrationTrace.ToArray());
 sample_bytes=$script:integrationFiles['.samples.jsonl'].Length}|ConvertTo-Json -Depth 12
"""


@pytest.mark.parametrize("mode", ["valid", "empty", "identity_mismatch", "throw"])
def test_imported_sampler_lifecycle_fake_integration(tmp_path, mode):
    body = IMPORTED_SAMPLER_INTEGRATION_FAKE.replace("__MODE__", mode)
    body = body.replace("__RUNNER__", str(RUNNER).replace("'", "''"))
    body = body.replace("__ROOT__", str(tmp_path).replace("'", "''"))
    result = run_fake(tmp_path, body, names=(
        "Import-CimExerciseFunctions", "Invoke-CimExerciseLifecycle",
        "Update-CimExerciseKnownStreamCounts", "Set-CimExerciseTerminalFailure",
        "Publish-CimExerciseTerminal", "Confirm-CimExerciseWatchdog",
    ))
    terminal = result["terminal"]
    assert result["replacement_count"] == 1 and result["source_equivalent"] is True
    assert result["binding_calls"] == 0 and result["factories"] == 2
    assert result["tracked_count"] == 1 and result["tracked_identity_valid"] is True
    assert result["wrapper_rows"] == {"valid": 1, "empty": 0, "identity_mismatch": 2, "throw": None}[mode]
    assert result["launch_seconds"] == 20 and result["expiry_seconds"] == 260
    assert result["sample_bytes"] <= 524288
    assert terminal["watchdog_ready"] and terminal["cleanup_complete"] and terminal["drain_complete"]
    assert terminal["watchdog_completion_pending"] is True
    assert terminal["stdout_bytes"] == terminal["stderr_bytes"] == terminal["watchdog_stderr_bytes"] == 0
    events = result["trace"]
    assert next(i for i, event in enumerate(events) if event["event"] == "ready") < next(
        i for i, event in enumerate(events) if event["event"] == "provider"
    )
    assert all(event["ready"] for event in events if event["event"] == "provider")
    expected_args = {"class": "Win32_Process", "properties": ["ProcessId", "ParentProcessId", "CreationDate"],
                     "timeout": 1, "action": "Stop"}
    assert all(arguments == expected_args for arguments in result["arguments"])
    assert len(result["arguments"]) == result["provider_calls"]
    flushes = [event for event in events if event["event"] == "sample_flush"]
    assert all(event["durable"] for event in flushes)
    assert [event["index"] for event in flushes] == result["indices"]
    assert len(flushes) == len(result["records"]) == terminal["completed_samples"]
    if mode == "valid":
        assert result["failed"] is False, result["stage"]
        assert result["provider_calls"] == result["memory_count"] == terminal["attempted_samples"] == 300
        assert terminal["completed_samples"] == 300 and terminal["last_acknowledged_sample"] == 299
        assert terminal["status"] == "observed_no_failure" and terminal["accounting_complete"] is True
        assert terminal["child_exit_code"] == 0 and terminal["child_exit_observed"] is True
        assert result["natural_exit_seconds"] == result["launch_seconds"] + 240
        assert result["memory_tail"] == 60
        assert result["indices"] == list(range(300))
        assert [record["global_row_count"] for record in result["records"]] == [2] * 240 + [1] * 60
        assert [record["live_count"] for record in result["records"]] == [1] * 240 + [0] * 60
        assert [record["elapsed_seconds"] for record in result["records"]] == list(range(20, 320))
        assert all(record["query_duration_state"] == "measured" for record in result["records"])
        assert result["records"][0]["query_start_gap_state"] == "unavailable"
        assert all(record["query_start_gap_state"] == "measured" and record["query_start_gap_seconds"] == 1
                   for record in result["records"][1:])
    else:
        assert result["failed"] and result["original_preserved"]
        assert result["provider_calls"] == terminal["attempted_samples"] == 1
        assert result["memory_count"] == (0 if mode == "throw" else 1)
        assert terminal["completed_samples"] == 0 and terminal["last_acknowledged_sample"] is None
        assert terminal["accounting_complete"] is False and terminal["child_exit_code"] == -1
        assert terminal["peak_combined_working_set_bytes"] == 0
        assert result["indices"] == result["records"] == []
        assert result["sample_bytes"] == 0 and result["natural_exit_seconds"] is None
        if mode == "throw":
            assert terminal["status"] == "native_query_failed" and result["stage"] == "sampler"
            assert terminal["hresult"] == result["provider_hresult"] == result["error_hresult"]
        else:
            assert terminal["status"] == "exercise_incomplete" and result["stage"] == "accounting"
            assert result["accounting_reason"] == "accounting_failed"
            assert terminal["hresult"] is None and terminal["native_error_code"] is None
    assert all(event.get("watch_alive", True) for event in events)
    pending_flush = next(i for i, event in enumerate(events) if event["event"] == "terminal_flush")
    pending_marker = next(i for i, event in enumerate(events) if event["event"] == "protocol" and event["value"] == 68)
    committed_marker = next(i for i, event in enumerate(events) if event["event"] == "protocol" and event["value"] == 70)
    assert pending_flush < pending_marker < committed_marker
    assert any(event["event"] == "terminal_flush" and event["durable"] and pending_marker < i < committed_marker
               for i, event in enumerate(events))
