# M2 renderer review diagnostics — 2026-09-08

These are fixed-settings, supplied-camera diagnostics run after the M2 renderer
review fixes. They are evaluation evidence only: both runs consume fixture camera
manifests and scene bounds, so neither qualifies for the image-only promotion gate.

The visibility denominator now uses exact cached model polygons, alpha-tested
cut-outs, explicit intrinsics, principal points, and roll where a palette is
available. It therefore supersedes the prior box-only denominator; it is not a
quality improvement claim and should not be compared to the schema-13 baseline
without rerunning that baseline under this evaluator.

| Fixture | Visible exact state | Visible geometry F1 | Precision / recall | Visible cells | Runtime |
|---|---:|---:|---:|---:|---:|
| Template 2 | 78.74% (526/668) | 92.66% | 94.69% / 90.72% | 668 | 31.6 s reconstruct; 102.4 s total |
| Template 6 | 0.00% (0/2,007) | 85.86% | 82.47% / 89.54% | 2,007 | 135.0 s reconstruct; 200.0 s total |

Both runs used cache schema 15, `adjacency_strength=0.55`, unbounded adjacency,
and the local 1.21.5 client JAR. Their generated JSON reports are deliberately
ignored under `exports/m2-review/`; they retain source-frame, camera-manifest,
scene, and JAR hashes plus all effective settings.

Commands:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_templates.py 'templates 2' 'exports\m2-review\template2.litematic' --mode supplied-camera --reference 'templates 2\Template 2.litematic' --camera-manifest 'templates 2\cameras.json' --scene 'templates 2\scene.json' --minecraft-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar" --report 'exports\m2-review\template2.json'
.\.venv\Scripts\python.exe scripts\benchmark_templates.py 'templates 6' 'exports\m2-review\template6.litematic' --mode supplied-camera --reference 'templates 6\Template 6.litematic' --camera-manifest 'templates 6\cameras.json' --scene 'templates 6\scene.json' --minecraft-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar" --report 'exports\m2-review\template6.json'
```

The Template 2 diagnostic misses the 80% exact-state gate by 1.26 points despite
clearing geometry; Template 6 misses both. This identifies reconstruction quality
work, not a renderer correctness failure.

## R6 frozen-mask rerun

R6 is now implemented in the maintained harness. The evaluator accepts a
candidate-independent usable-pixel mask and applies it only after exact
alpha-tested ownership rendering. The frozen fixture mask is
[`vanilla-hud-2560x1369-v1.json`](../evaluation_masks/vanilla-hud-2560x1369-v1.json):
it contains integer rectangles for the status bars, chat, hotbar, crosshair and
held-item overlay, rather than calling the mutable production GUI-mask helper.
Its SHA-256 is
`11651fc4d5f368fdcdfcabc86531a283d1e0817b0d119cf0d6fd0d12fadba797`.

Both fixture reports were rerun under source-tree SHA-256
`c18a2fcdba837d5a6cbf6a48b78c1f4bdc22547c7e6503239222c9543efc4694`.
They record the exact candidate hash, Git revision/dirty state, every input and
mask hash, effective settings, phase timings, permitted rotation/state-aware
alignment, and visible breakdowns by state and pixel area.

| Fixture | Visible exact state | Visible geometry F1 | Precision / recall | Visible cells | Runtime |
|---|---:|---:|---:|---:|---:|
| Template 2 | 78.52% (519/661) | 92.58% | 94.63% / 90.62% | 661 | 33.5 s reconstruct; 59.0 s total |
| Template 6 | 0.00% (0/1,963) | 85.55% | 82.11% / 89.30% | 1,963 | 157.2 s reconstruct; 239.1 s total |

The frozen-mask rerun remains a supplied-camera diagnostic and therefore cannot
pass the image-only production gate. Its values differ from the unmasked table
because the denominator is deliberately fixed to usable pixels; the change is
not an algorithm-quality comparison.
