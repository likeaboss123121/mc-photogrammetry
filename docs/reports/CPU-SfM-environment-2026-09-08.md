# CPU-SfM environment — 2026-09-08

## Purpose

Provide an isolated, CPU-only PyCOLMAP environment for M3 multiview experiments
without changing the project's Python 3.10 test environment.

## Installed environment

| Item | Actual value |
| --- | --- |
| Python executable | `C:\Users\Likea\Documents\mc photogrammetry project\.venv-sfm\Scripts\python.exe` |
| Python | 3.12.10 |
| pip | 26.2.1 |
| NumPy | 2.5.3 |
| PyCOLMAP | 4.2.0 |
| Device used | `pycolmap.Device.cpu` |

The environment was made with `py -3.12 -m venv .venv-sfm`; the workspace
`.venv` remains Python 3.10.11 and unchanged. `.venv-sfm/` is ignored.

Install commands:

```powershell
py -3.12 -m venv .venv-sfm
.\.venv-sfm\Scripts\python.exe -m pip install --upgrade pip
.\.venv-sfm\Scripts\python.exe -m pip install pycolmap==4.2.0 numpy
```

The import check succeeded. `pycolmap.extract_features`,
`pycolmap.match_exhaustive`, and `pycolmap.incremental_mapping` are all exposed
by this wheel.

## Small CPU smoke test

`scripts/sfm_smoke.py` performed feature extraction, exhaustive matching with
geometric verification, and incremental mapping on all three Template 2 PNGs.
It uses one shared camera (`CameraMode.SINGLE`) and explicitly selects the CPU:

```powershell
.\.venv-sfm\Scripts\python.exe scripts\sfm_smoke.py 'templates 2' `
  'exports\m2-review\sfm-template2-single-camera-cpu-20260908' `
  --image-name 2026-09-02_13.22.14.png `
  --image-name 2026-09-02_13.24.24.png `
  --image-name 2026-09-02_13.24.33.png
```

The run completed in 5.4 seconds and wrote an ignored database and summary.
It found 21,669 keypoints, three verified image pairs, and 1,200 database
matches. Incremental mapping ran but returned zero reconstructions because
COLMAP could not select an initial pair. This is a useful M3 experiment result,
not an environment failure: extraction, matching, verification, database access,
and the mapper all completed under the intended CPU interpreter.

## M3 invocation

Use the executable and `scripts/sfm_smoke.py` command above for bounded
experiments. The script refuses to replace an existing output directory,
records package and database counts in `summary.json`, and keeps artifacts under
the already ignored `exports/` tree. A mapping result with zero registered
images must be reported as such; it is not a recovered pose.
