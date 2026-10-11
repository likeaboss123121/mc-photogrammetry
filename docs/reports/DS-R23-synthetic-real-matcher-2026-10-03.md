# DS-R23 synthetic real-matcher experiment — 2026-10-03

Bounded foreground experiment authorized by the DS-R23 entry in `AGENT_COMMS.md`
(2026-10-03 11:34 UTC). The objective was to run the **actual**
`plane_graph.match_plane_graph` and the **actual** `plane_graph.solve_component`
on a deterministic in-memory synthetic two-frame session/lattice after real
signature derivation, using the accepted DS-R22B/R22C fixture construction and a
valid synthetic connection/pose graph only to bind relative yaw. No source or
production file was edited; this report is the only repository addition.

## What was executed

- Working directory: repository root.
- Exact command (project venv, Python 3.10.11):

  ```powershell
  & .\.venv\Scripts\python.exe "$env:TEMP\ds_r23_probe.py"
  ```

- Driver: `%TEMP%\ds_r23_probe.py`, 10021 bytes,
  SHA-256 `976E84710D05143295534E88AB9B90EBE99D1F4829065CB38D817BDA9D23E3AB`
  (outside the repository; no repository file was created by the run).
- The driver imports the accepted test module
  `tests/test_image_derived_axis_diagnostics_assembled.py` and reuses its fixture
  helpers (`_synthetic_two_frame_session`, `_accepted_graphs`, `_entry`,
  `production_mod._make`). The only environment substitution is the session's own
  declared camera-solver seam (`image_only.solve_image_cameras`), which is part of
  the accepted fixture. `plane_graph.match_plane_graph` and
  `plane_graph.solve_component` are **not** patched or spied on: the run calls
  the real functions.
- Scene: `ROOM = (0,12,0,6,0,12)`, `mixed_material`, 360×240, two frames
  `f0` at (3.6,0.7,3.6) and `f1` at (5.0,0.7,5.0), both yaw 10°, pitch −25°,
  fov 70°. The connection graph and pose graph are the accepted synthetic
  construction (unit-baseline identity-rotation pair, yaw class 0), used only to
  supply the relative-yaw constraint.
- Synthetic provenance digests: session
  `015138379c4bc09a11eb9084a93deb1e7555d00e61cccd396bded3e42cf0474e`,
  graph `655673dc9a106c5e919a115c98aec9f110902919afd8129b4fb6384e91927bf2`
  (in-memory artifacts, not reads of any real file).

## Per-frame readings (measured lattice)

Lattice status `assembled` (reason empty). 4 readings total.

| frame | axis | side | structural_share | retained b alternatives |
|-------|------|------|------------------|-------------------------|
| f0 | 0 | −1 | 0.6467 | 1.0, 0.5 |
| f0 | 2 | +1 | 0.7731 | 1.0, 0.25, 0.5, 0.125 |
| f1 | 1 | +1 | 0.1277 | 1.0 |
| f1 | 2 | +1 | 0.7446 | 1.0, 0.25, 0.5, 0.125 |

Per-frame axis counts — f0: axis0 ×1, axis2 ×1; f1: axis1 ×1, axis2 ×1.

## Per-frame signatures (real derivation)

`derive_image_derived_signatures` returned status `assembled`. 11 signatures
total (one per reading alternative).

| frame | axis0 | axis1 | axis2 | total |
|-------|-------|-------|-------|-------|
| f0 | 2 | 0 | 4 | 6 |
| f1 | 0 | 1 | 4 | 5 |

## Real matcher (`match_plane_graph`, `max_shift_cells=32`)

Trace counters: 120 pre-admission candidates (30 signature pairs × 4 quadrants),
37 pair-correlation admissions, 4 after the same-axis equality filter, 120 pair
rows. Refusals: `axis_mismatch` 72, `same_axis_mismatch` 33,
`repetition_unresolved` 5, `no_signal` 4, `insufficient_overlap` 2
(83 pre-admission refusals + 37 admissions = 120; 33 drops + 4 kept = 37).

**Admitted matches: 4.** Per-axis distribution of the admitted endpoints:
left axis {0: 2, 2: 2}, right axis {2: 4}. Under the constraint-consistent
quadrants (f0 = f1 = 0) the distribution is world axis {2: 1}.

| # | left | right | Laxis | Raxis | Lplane | Rplane | left_b | right_b | quadrant | psr | score | overlap | shift rc |
|---|------|-------|-------|-------|--------|--------|--------|---------|----------|-----|-------|---------|----------|
| 1 | f0 | f1 | 0 | 2 | 0 | 1 | 1.0 | 0.5 | 1 | 5.4247 | 0.6562 | 31 | (3, 31) |
| 2 | f0 | f1 | 0 | 2 | 0 | 1 | 1.0 | 1.0 | 3 | 4.7922 | 0.5506 | 267 | (−20, 21) |
| 3 | f0 | f1 | 2 | 2 | 1 | 1 | 1.0 | 0.5 | 2 | 7.5252 | 0.7114 | 575 | (1, −2) |
| 4 | f0 | f1 | 2 | 2 | 1 | 1 | 1.0 | 1.0 | **0** | 7.9208 | 0.7208 | **2415** | (1, 3) |

Match 4 is the only identity consistent with the synthetic relative yaw
(quadrant 0) and the true identity relative rotation; it also carries by far the
largest overlap and highest score. Matches 1–3 are false identities that the
matcher admits under unconstrained quadrant hypotheses — including the same
f0-axis-2 / f1-axis-2 plane pair with `right_b = 0.5` (match 3). The oracle yaw
constraint below is what discards them.

## Graph constraints (`_connection_constraints`)

Status `ok`. One solved evidence edge: `f0 → f1`, status `solved`, support 6,
`right_yaw_class` 0. Emitted constraints:
`[["f0", "f1", 0]]` (`solved_edge_count` 1, `yaw_constraint_count` 1).

## Real solver (`solve_component`, thresholds 0.15 / 0.90 / 0.02)

Baseline, with `quadrant_constraints = (("f0","f1",0))`:

- gate (`reason`): **`insufficient_axis_coverage`**
- matched axes: `[2]` (resultants `{2: 0.9999999999999999}`)
- residuals: `{2: 3.8981718325193755e-17}`
- quadrants: `{f0: 0, f1: 0}`; retained alternatives 1; `scale_known` false

The one surviving match (match 4) is numerically clean, but the component covers
only one world axis, below the solver's three-axis requirement, so it abstains
honestly rather than asserting a pose.

Constraint ablation (identical readings and ordered matches, only
`quadrant_constraints` changed to `()`):

- baseline `insufficient_axis_coverage`; ablated `ambiguous_quadrant`;
  `changed = true`.

So the oracle yaw constraint alone changes the typed outcome: without it, the
three spurious matches create a rival quadrant assignment within the 0.02
margin (`ambiguous_quadrant`); with it, they are filtered and the gate becomes
insufficient axis coverage. The end-to-end diagnostic agreed:
`fired_gate = insufficient_axis_coverage`, `matched_pair_counts {0:0, 1:0, 2:1}`,
`selected_match_count` 4, `solved_edge_count` 1, `yaw_constraint_count` 1,
`normal_mode.unchanged = true`.

## Deterministic repeat

- In-process: two consecutive `diagnose_production_result` payloads produced
  identical canonical JSON (`deterministic_in_process = true`).
- Cross-process: two independent invocations produced byte-identical stdout,
  SHA-256 `C74A9E0982916F11486151439A866F3C9678C63602B17CBD202AC7059F7B8855`;
  the probe summary canonical digest is
  `22840cbbdc171008b86ca6062cf44abf5acfb61e8863c8349011ef3266296d2b`.

## Limitations

- Synthetic geometry only: a procedural ray-cast room rendered from constants;
  no real image, capture, fixture, reference or mask file was read, hashed or
  processed. This is not real-camera or real-fixture accuracy, and not an
  image-causality claim.
- The true camera is injected through the session's declared solver seam, and the
  connection/pose graph is the accepted **synthetic** construction used only to
  bind relative yaw. The reported solver behaviour is conditioned on that oracle
  yaw; it does not demonstrate image-derived connection-graph accuracy.
- The lattice covers only 4 readings over 2 frames and 2 axes, so a three-axis
  solve is unreachable here by construction; `insufficient_axis_coverage` is an
  expected typed abstention, not a matcher or solver defect.
- The real matcher admits spurious same-plane candidates under wrong quadrants
  (3 of 4 here); this experiment does not test the pipeline's ability to reject
  them without the connection-graph constraint.
- No live diagnostic, benchmark, Minecraft/v5 path, gate/threshold change,
  background/persistence or system modification was involved.

Reported for independent lead/Jev review. No self-acceptance.
