param([switch]$Preview)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stateFile = Join-Path $projectRoot 'PROJECT_PROGRESS_STATE.json'
$logFile = Join-Path $projectRoot 'PROJECT_PROGRESS.md'
$state = Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
$nowLocal = Get-Date
$timestamp = $nowLocal.ToString('yyyy-MM-dd HH:mm:ss zzz')
function Clip-250([string]$value) {
    if ($value.Length -le 250) { return $value }
    return $value.Substring(0, 247) + '...'
}
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("## Update $timestamp")
$lines.Add('')
$lines.Add("Status: $($state.project_status)")
$lines.Add('')
$lines.Add('| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |')
$lines.Add('|---|---:|---:|---:|---:|---:|')
foreach ($metric in $state.metrics) {
    if ($null -eq $metric.exact -or $null -eq $metric.visible -or $null -eq $metric.candidate_ceiling) {
        $lines.Add("| $($metric.label) | N/A | N/A | N/A | N/A | N/A |")
        continue
    }
    $exactPct = 100.0 * [double]$metric.exact / [double]$metric.visible
    $ceilingPct = 100.0 * [double]$metric.candidate_ceiling / [double]$metric.visible
    $relative = if ($ceilingPct -gt 0) { 100.0 * $exactPct / $ceilingPct } else { 0.0 }
    $previous = $state.last_reported_exact.PSObject.Properties[$metric.key]
    $gain = if ($null -eq $previous) { 'baseline' } else { '{0:+0.00;-0.00;0.00} pp' -f ($exactPct - [double]$previous.Value * 100.0 / [double]$metric.visible) }
    $expected = '{0:+0.00;-0.00;0.00} pp' -f [double]$state.expected_gain_next_update_pp
    $lines.Add(('| {0} | {1:0.00}% ({2}/{3}) | {4:0.00}% ({5}/{3}) | {6:0.00}% | {7} | {8} |' -f $metric.label,$exactPct,$metric.exact,$metric.visible,$ceilingPct,$metric.candidate_ceiling,$relative,$gain,$expected))
}
$lines.Add('')
$lines.Add('Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.')
$lines.Add('')
$lines.Add("Image-only production: $($state.image_only_status)")
$lines.Add('')
$lines.Add("What happened (max 250 chars): $(Clip-250 ([string]$state.what_happened))")
$lines.Add('')
$lines.Add("What happens next (max 250 chars): $(Clip-250 ([string]$state.what_next))")
$lines.Add('')
$lines.Add("Forecast basis: $($state.expected_gain_basis)")
$lines.Add('')
$lines.Add('---')
$block = ($lines -join [Environment]::NewLine) + [Environment]::NewLine
if ($Preview) { Write-Output $block; exit 0 }
$stream = [System.IO.File]::Open($logFile,[System.IO.FileMode]::Append,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
try { $writer=[System.IO.StreamWriter]::new($stream,[System.Text.UTF8Encoding]::new($false)); $writer.Write($block); $writer.Flush() } finally { $stream.Dispose() }
foreach ($metric in $state.metrics) {
    if ($null -ne $metric.exact -and $null -ne $metric.visible) {
        $property=$state.last_reported_exact.PSObject.Properties[$metric.key]
        if ($null -eq $property) { $state.last_reported_exact | Add-Member -NotePropertyName $metric.key -NotePropertyValue ([int]$metric.exact) }
        else { $property.Value=[int]$metric.exact }
    }
}
$state.last_updated_local=$timestamp
[System.IO.File]::WriteAllText($stateFile,($state|ConvertTo-Json -Depth 8)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))