@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    uv sync --extra desktop
    uv run mc-photogrammetry desktop
    exit /b %ERRORLEVEL%
)

set PYTHONPATH=%~dp0src
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m mc_photogrammetry desktop
    exit /b %ERRORLEVEL%
)
py -m mc_photogrammetry desktop
exit /b %ERRORLEVEL%
