# DS-R3-AC: targeted Canal Town axis-0 capture contract

Status: **planned contract; no capture launched** (2026-09-20).

The current frozen Canal Town queue is complete and immutable. Its batch spec generates generic orbit cameras and cannot express a targeted shared-surface axis-0 evidence loop. The next capture must use a separate evaluation-only plan and hand only finalized PNG paths to normal reconstruction.

## Required plan

- Schema `mc-photogrammetry-evaluation-targeted-capture`, version 1, usage `evaluation_only`, canonical SHA-256 digest.
- Four ordered jobs: `axis0-primary`, `axis0-overlap-a`, `axis0-overlap-b`, `context`.
- The first three jobs share one distinctive world-axis-0 surface; the context job must not claim that surface.
- Freeze vanilla, clear noon, HUD hidden, one FOV/resolution family where possible, and zero roll. Preserve genuine block-scale cues; do not add markers or supplied scale aliases.
- Quality targets: pair support >=64 cells and >=0.5 fraction, predicted displacement <=24 cells, structural share >=0.15, PSR >=3. These are acceptance targets, not evidence until measured from rendered images.
- Bind source-save, worker-template, capture-mod, predecessor manifest and predecessor capture SHA-256 values. Bind plan hash to materialized `CapturePlan`, staged-world token, completion sentinel, pose/PNG/GT hashes.

## Candidate evaluation geometry

A defensible candidate derives from predecessor views 01/02/03 for the shared east-facing surface, plus view 00 as context, after the previously evaluated inward radial reframe:

| role | feet position | yaw | pitch | FOV | resolution |
|---|---|---:|---:|---:|---|
| axis0-primary | (173.758, 88.966, 379.706) | 133.5612 | 40.1070 | 90 | 1600x900 |
| axis0-overlap-a | (172.408, 101.381, 369.362) | 92.0916 | 68.7549 | 70 | 1920x1080 |
| axis0-overlap-b | (172.695, 94.533, 358.268) | 43.5284 | 51.5662 | 80 | 1920x1080 |
| context | (164.372, 88.024, 386.652) | 173.9453 | 34.3775 | 90 | 1280x720 |

These poses satisfy prior framing/clearance checks, but common visible-surface identity and support/PSR targets remain unverified. Do not materialize this candidate as a launch plan until a read-only rendered-image preflight or fresh capture measures the targets.

## Safety and isolation

Use a fresh disposable Prism worker, hidden Desktop 2, master volume zero, `pauseOnLostFocus=false`, `rawMouseInput=false`, no foreground activation or input injection, and confirmed teardown. Never overwrite the frozen queue or source save. Production receives only ordered PNG paths through `png_only_handoff`; poses, bounds, GT, masks and target identities remain evaluation-only.

Current local candidate template: `B:\prismInstances\MCPhotoCapture-Template-20260920`, containing only regular `instance.cfg` and `mmc-pack.json`; digest `11126d1f83530c6dea8515f15bd08fe88270c67cb7f51f7c1bb1a9c4e6c028ea`. It is not yet a certified launch binding until executor verification and plan preflight succeed.
