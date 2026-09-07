$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv sync --extra desktop
    uv run mc-photogrammetry desktop
    exit $LASTEXITCODE
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python or uv is required. Install Python 3.10+ or uv first."
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$projectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $projectPython) {
    & $projectPython -m mc_photogrammetry desktop
} else {
    py -m mc_photogrammetry desktop
}
exit $LASTEXITCODE
