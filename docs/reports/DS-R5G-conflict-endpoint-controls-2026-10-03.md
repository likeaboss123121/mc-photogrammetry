# DS-R5G per-endpoint removal versus matched-size controls — 2026-10-03

## Scope

Evaluation-only, one foreground contained replay that recomputed the frozen v6
chain from the four frozen PNGs, failed closed through DS-R5F's full
trace/graph/match-identity replay checks, and then, for each of the eight
conflicting `(frame_id, plane)` endpoints, solved after removing only the
matches incident to that endpoint (treatment) and after 20 deterministic,
without-replacement, matched-size control removals per endpoint. No production
module, existing evaluator, threshold, gate, camera, image, frozen project, or
frozen artifact was changed. The new R5G utility and tests plus this report and
one output were created. DS-R5F output was neither reused nor overwritten.
The one run output is `.tmp-ds-r5g-20261003.json`.

Utility: `scripts/evaluation/replay_graph_endpoint_controls.py` (SHA-256
`983640e2de7205a4640c1b31bd72ac88b0a62d7057ec95540ea4d7c6cce6ac67`).
Focused tests: `tests/test_graph_endpoint_controls.py` (SHA-256
`7d542351953225bd0d7e9faa2d707b01d0bfa19f688d96729cf6d19ed1c3e58c`).

Command (one invocation, no retry):

```
.\.venv\Scripts\python.exe scripts\evaluation\replay_graph_endpoint_controls.py
```

## Pre-run gate recheck (DeepSeek)

The three named output paths were absent immediately before the run:

| Path | Before | After |
|---|---|---|
| `.tmp-ds-r5g-20261003.json` | absent | written (complete) |
| `.tmp-ds-r5g-20261003.json.partial` | absent | absent (no staging left) |
| `docs/reports/DS-R5G-conflict-endpoint-controls-2026-10-03.md` | absent | written after the run |

Frozen producer hashes at run time: utility `983640e2…ac67`, focused test
`7d542351…e58c`. Unchanged input hashes: frozen v6 artifact file
`7688538f…56c1` (internal seal `c5c62515…f6b3`), project `a85fe65b…f87`, four
ordered PNGs `0087a714…ba07`, `0c309299…95f6`, `874fe746…8049`, `2c75a0df…c853`.
DS-R5F output `.tmp-ds-r5f-20261003.json` remains `de68ddb7…c7c9`. Per the
session instruction, only the single contained replay was run; the focused
suite was not rerun (the lead pre-run gate already records 27/27 focus tests and
`py_compile` pass on these exact hashes).

## Replay validation (fail-closed)

The parent created the already-reviewed 8-GiB Job through
`scripts/run_image_derived_graph_stage.py::_native_execute` (which loads the
reviewed `.tmp-run-axis0-component-preflight-replacement.py` Job helper) and
launched the utility itself as a contained worker under a 600-second deadline.
The worker refused before image processing if it was not inside the Job. Exit 0;
wall time 147.387 s; no `.partial` remained; no other path was created.

Every check below matched the v6 artifact, so the endpoint/control stage was
reached:

| Evidence | Value |
|---|---|
| Artifact self-digest | `c5c6251504f74c0f9c16a9666834d74b185cc13e180d6a30dec7fea6b312f6b3` |
| Session / lattice / signatures | `db924383…0c4e` / `98784b0c…0ded` / `e5875d66…8412` |
| Policy | `501ada9e…b08a` |
| Graph result (inner, outer, trace join) | `83eb7fb3b995b9d89bdccedef7480f40daf0793564191386d9f4024c81226ae4` |
| Pair-trace record (all 7,476 rows, full record equality) | `eb2fabfa1e2acc907e1397e61430cbbb9cb2a2e785ce516b8c66b8d60b2ac176` |
| Candidate funnel pre / post-admission / post-same-axis | 7,476 / 1,495 / 54 |
| Selected match identity (alias and boundary domains) | `e6f9e955…842a` and `7fe96f18…d9a9`, count 54 |
| Boundary kept counts / radius | `[1, 1, 0]` / 32 |
| Baseline reason | `plane_integer_residual` |
| Baseline per-axis (resultant, residual, observations) | `(0, .39146, .17397, 12)`, `(1, .24717, .20172, 6)`, `(2, .24521, .18721, 10)` |
| Conflicting endpoints | 8 (`image-0000` 0/1, `image-0001` 1/3/4, `image-0002` 0/2/3) |

The recomputation used the accepted chain with `_persist_session` temporarily
replaced by a no-op, so the frozen project was read and never rewritten. The
non-invasive observer captured the exact `readings`, the exact post-same-axis
`PlaneMatch` tuple and the exact solver keyword arguments (including the
image-derived quadrant constraints); all three treatments and controls reused
those captured values, the production solver and the unchanged gates.

## Experiment design as executed

For endpoint `(frame_id, plane)`:

- **treatment** removes exactly the matches incident to that endpoint;
- **controls** are 20 deterministic, without-replacement removals per endpoint,
  matching the treatment's removed-match count per baseline world axis. The
  baseline world axis of a match is
  `plane_graph.axis_transform(match.left_axis, baseline_quadrants[match.left])`
  using the baseline best quadrants; controls are drawn only from matches not
  incident to the target endpoint, are exactly stratified by that axis, and are
  ranked by `SHA-256(replicate_seed || canonical match identity)`.
- The replicate seed is `SHA-256("<frame_id> = <plane_index> # <replicate_index>")`
  on explicit UTF-8 text, recorded per replicate as `seed_sha256`.
- A bin without enough eligible controls is recorded as a typed
  `control_unavailable`; matching is never relaxed and no substitute row is
  fabricated.

All 8 treatments solved with full axis coverage `0,1,2` and still ended
`plane_integer_residual`. Control availability, control reason and the number of
distinct control removal sets:

| Endpoint | Required per axis (0/1/2) | Eligible controls (0/1/2) | Control status | Control reasons | Distinct control sets |
|---|---|---|---|---|---|
| `(image-0000, 0)` | 12 / 0 / 0 | 12 / 9 / 21 | 20 solved | `plane_integer_residual` ×20 | 1 |
| `(image-0000, 1)` | 0 / 6 / 0 | 24 / 3 / 21 | 20 unavailable | `control_unavailable` ×20 (axis 1: 3 < 6) | 1 |
| `(image-0001, 1)` | 1 / 0 / 5 | 23 / 9 / 16 | 20 solved | `plane_integer_residual` ×20 | 20 |
| `(image-0001, 3)` | 0 / 3 / 0 | 24 / 6 / 21 | 20 solved | `plane_integer_residual` ×20 | 13 |
| `(image-0001, 4)` | 10 / 0 / 2 | 14 / 9 / 19 | 20 solved | `plane_integer_residual` ×20 | 20 |
| `(image-0002, 0)` | 4 / 0 / 7 | 20 / 9 / 14 | 20 solved | `plane_integer_residual` ×20 | 20 |
| `(image-0002, 2)` | 0 / 6 / 0 | 24 / 3 / 21 | 20 unavailable | `control_unavailable` ×20 (axis 1: 3 < 6) | 1 |
| `(image-0002, 3)` | 8 / 0 / 4 | 16 / 9 / 17 | 20 solved | `plane_integer_residual` ×19, `ambiguous_quadrant` ×1 | 20 |

The `control_unavailable` rows record `removed_match_count = 0` and
`retained_match_count = 54` (the untouched baseline); nothing was removed.

## Treatment versus control results (descriptive)

Per-axis `(resultant, residual, observations)`; control columns are the range
over the 20 replicates where solved (n = 20 unless noted).

| Endpoint | Axis | Treatment | Control n | Control resultant | Control residual | Control obs |
|---|---:|---|---:|---|---|---|
| `(image-0000, 0)` | 0 | `(0.926217, 0.0577868, 6)` | 20 | `[0.167682, 0.167682]` | `[0.227573, 0.227573]` | `[6, 6]` |
| `(image-0000, 0)` | 1 | `(0.247172, 0.201723, 6)` | 20 | `[0.247172, 0.247172]` | `[0.201723, 0.201723]` | `[6, 6]` |
| `(image-0000, 0)` | 2 | `(0.134354, 0.215621, 12)` | 20 | `[0.245206, 0.245206]` | `[0.187211, 0.187211]` | `[10, 10]` |
| `(image-0000, 1)` | 0 | `(0.391458, 0.173970, 12)` | 0 | — | — | — |
| `(image-0000, 1)` | 1 | `(1.000000, 3.6046e-15, 2)` | 0 | — | — | — |
| `(image-0000, 1)` | 2 | `(0.245206, 0.187211, 10)` | 0 | — | — | — |
| `(image-0001, 1)` | 0 | `(0.391458, 0.173970, 12)` | 20 | `[0.132310, 0.391458]` | `[0.173970, 0.222952]` | `[10, 12]` |
| `(image-0001, 1)` | 1 | `(0.247172, 0.201723, 6)` | 20 | `[0.247172, 0.247172]` | `[0.201723, 0.201723]` | `[6, 6]` |
| `(image-0001, 1)` | 2 | `(1.000000, 2.5091e-15, 6)` | 20 | `[0.217992, 0.478366]` | `[0.145814, 0.223985]` | `[4, 10]` |
| `(image-0001, 3)` | 0 | `(0.391458, 0.173970, 12)` | 20 | `[0.391458, 0.391458]` | `[0.173970, 0.173970]` | `[12, 12]` |
| `(image-0001, 3)` | 1 | `(1.000000, 0.000000, 2)` | 20 | `[0.247172, 0.821135]` | `[0.096671, 0.201723]` | `[4, 6]` |
| `(image-0001, 3)` | 2 | `(0.245206, 0.187211, 10)` | 20 | `[0.245206, 0.245206]` | `[0.187211, 0.187211]` | `[10, 10]` |
| `(image-0001, 4)` | 0 | `(0.360056, 0.176306, 6)` | 20 | `[0.113465, 0.567187]` | `[0.122105, 0.228442]` | `[6, 10]` |
| `(image-0001, 4)` | 1 | `(0.247172, 0.201723, 6)` | 20 | `[0.247172, 0.247172]` | `[0.201723, 0.201723]` | `[6, 6]` |
| `(image-0001, 4)` | 2 | `(0.245206, 0.187211, 10)` | 20 | `[0.162582, 1.000000]` | `[2.5091e-15, 0.214819]` | `[6, 12]` |
| `(image-0002, 0)` | 0 | `(0.391458, 0.173970, 12)` | 20 | `[0.062880, 0.999062]` | `[0.006501, 0.240705]` | `[6, 12]` |
| `(image-0002, 0)` | 1 | `(0.247172, 0.201723, 6)` | 20 | `[0.247172, 0.247172]` | `[0.201723, 0.201723]` | `[6, 6]` |
| `(image-0002, 0)` | 2 | `(0.439238, 0.177624, 4)` | 20 | `[0.245233, 0.544946]` | `[0.133306, 0.202053]` | `[6, 8]` |
| `(image-0002, 2)` | 0 | `(0.391458, 0.173970, 12)` | 0 | — | — | — |
| `(image-0002, 2)` | 1 | `(1.000000, 3.8982e-17, 2)` | 0 | — | — | — |
| `(image-0002, 2)` | 2 | `(0.245206, 0.187211, 10)` | 0 | — | — | — |
| `(image-0002, 3)` | 0 | `(0.393098, 0.167475, 6)` | 20 | `[0.203149, 0.938133]` | `[0.048361, 0.212090]` | `[6, 12]` |
| `(image-0002, 3)` | 1 | `(0.247172, 0.201723, 6)` | 20 | `[0.247172, 0.247172]` | `[0.201723, 0.201723]` | `[6, 6]` |
| `(image-0002, 3)` | 2 | `(0.245206, 0.187211, 10)` | 20 | `[0.044510, 1.000000]` | `[2.4738e-16, 0.243005]` | `[6, 10]` |

Descriptive observations only, no significance or causality claim:

- **Removing just the incident matches does not change the terminal reason.**
  All 8 treatments remain `plane_integer_residual` with full `0,1,2` coverage,
  as do all solved controls; the matched-count control never turned the fixture
  into `insufficient_axis_coverage`.
- **On the targeted/required axis the treatment often desaturates residual
  structure.** Where the treated axis still has >= 2 retained observations, its
  resultant rises toward 1.0 and its residual falls toward 0
  (`(image-0000, 1)`, `(image-0001, 1)`, `(image-0001, 3)`, `(image-0002, 2)`)
  while observation counts fall (e.g. axis-1 observations 6 -> 2). That is the
  expected effect of removing evidence, and the same removal count also applies
  to each control.
- **Where the required-axis eligible pool equals the treatment count, the
  control draw is degenerate.** For `(image-0000, 0)` the axis-0 eligible pool is
  exactly 12 = required, so all 20 replicates remove the same 12 axis-0 matches
  and the control range is a single point; treatment (`0.9262`) sits outside that
  single control value (`0.1677`). This control cannot separate identity from
  evidence because it must remove the whole bin.
- **Two endpoints have no legitimate control at all.** For `(image-0000, 1)` and
  `(image-0002, 2)` the required axis is 1 with 6 removed matches but only 3
  non-incident axis-1 matches exist, so all 20 replicates are typed
  `control_unavailable`; these endpoints carry a descriptive treatment record but
  no control comparison.
- **One matched-size control changes the terminal reason.** Control-0009 of
  `(image-0002, 3)` ends `ambiguous_quadrant` (coverage still `0,1,2`,
  axes `(0.312644, 0.187651, 10)`, `(0.247172, 0.201723, 6)`,
  `(1.000000, 2.5091e-15, 6)`); the other 19 controls end `plane_integer_residual`.
  A non-conflicting matched-size removal can therefore land on different gates,
  which is exactly why the treatment values are reported alongside the control
  ranges rather than against a single baseline.

## Retained / removed identities

Treatments removed only incident matches; all other treatment/control identity
tuples and per-replicate seeds are in `.tmp-ds-r5g-20261003.json`. Treatment
removed index sets and identities:

- `(image-0000, 0)` — indices `0,1,2,3,10,11,12,13,20,21,22,23` (12 removed,
  42 retained):
  `[image-0000 0 0 image-0001 0 0 2 0.5 3.0]`, `[image-0000 0 0 image-0001 1 0 0 0.125 0.5]`,
  `[image-0000 0 0 image-0001 4 2 1 1.0 0.5]`, `[image-0000 0 0 image-0001 4 2 3 2.0 4.0]`,
  `[image-0000 0 0 image-0002 0 0 0 0.125 0.5]`, `[image-0000 0 0 image-0002 0 0 2 0.125 2.0]`,
  `[image-0000 0 0 image-0002 3 2 1 0.25 1.0]`, `[image-0000 0 0 image-0002 3 2 3 0.5 1.0]`,
  `[image-0000 0 0 image-0003 0 0 0 0.125 0.125]`, `[image-0000 0 0 image-0003 0 0 2 0.125 0.5]`,
  `[image-0000 0 0 image-0003 1 2 1 0.125 0.5]`, `[image-0000 0 0 image-0003 1 2 3 0.125 0.5]`
- `(image-0000, 1)` — indices `4,5,6,14,15,16` (6 removed, 48 retained):
  `[image-0000 1 1 image-0001 2 1 2 0.5 0.5]`, `[image-0000 1 1 image-0001 3 1 1 0.5 0.25]`,
  `[image-0000 1 1 image-0001 3 1 0 1.0 3.0]`, `[image-0000 1 1 image-0002 2 1 1 0.25 1.0]`,
  `[image-0000 1 1 image-0002 2 1 0 0.5 1.0]`, `[image-0000 1 1 image-0002 2 1 2 1.0 0.5]`
- `(image-0001, 1)` — indices `1,7,28,29,30,41` (6 removed, 48 retained):
  `[image-0000 0 0 image-0001 1 0 0 0.125 0.5]`, `[image-0000 3 2 image-0001 1 0 1 0.5 0.25]`,
  `[image-0001 1 0 image-0002 0 0 0 0.5 0.5]`, `[image-0001 1 0 image-0002 3 2 1 0.25 1.0]`,
  `[image-0001 1 0 image-0002 3 2 3 0.5 0.5]`, `[image-0001 1 0 image-0003 1 2 3 0.5 0.25]`
- `(image-0001, 3)` — indices `5,6,33` (3 removed, 51 retained):
  `[image-0000 1 1 image-0001 3 1 1 0.5 0.25]`, `[image-0000 1 1 image-0001 3 1 0 1.0 3.0]`,
  `[image-0001 3 1 image-0002 2 1 0 3.0 0.5]`
- `(image-0001, 4)` — indices `2,3,8,9,34,35,36,37,42,43,44,45` (12 removed, 42 retained):
  `[image-0000 0 0 image-0001 4 2 1 1.0 0.5]`, `[image-0000 0 0 image-0001 4 2 3 2.0 4.0]`,
  `[image-0000 2 2 image-0001 4 2 2 0.5 3.0]`, `[image-0000 3 2 image-0001 4 2 0 0.5 4.0]`,
  `[image-0001 4 2 image-0002 0 0 1 4.0 0.25]`, `[image-0001 4 2 image-0002 0 0 3 4.0 0.5]`,
  `[image-0001 4 2 image-0002 3 2 0 0.125 1.0]`, `[image-0001 4 2 image-0002 3 2 2 0.5 1.0]`,
  `[image-0001 4 2 image-0003 0 0 1 2.0 3.0]`, `[image-0001 4 2 image-0003 0 0 3 4.0 0.5]`,
  `[image-0001 4 2 image-0003 1 2 2 3.0 0.5]`, `[image-0001 4 2 image-0003 1 2 0 3.0 1.0]`
- `(image-0002, 0)` — indices `10,11,17,27,28,34,35,46,47,48,49` (11 removed, 43 retained):
  `[image-0000 0 0 image-0002 0 0 0 0.125 0.5]`, `[image-0000 0 0 image-0002 0 0 2 0.125 2.0]`,
  `[image-0000 3 2 image-0002 0 0 1 0.5 0.25]`, `[image-0001 0 0 image-0002 0 0 2 0.25 2.0]`,
  `[image-0001 1 0 image-0002 0 0 0 0.5 0.5]`, `[image-0001 4 2 image-0002 0 0 1 4.0 0.25]`,
  `[image-0001 4 2 image-0002 0 0 3 4.0 0.5]`, `[image-0002 0 0 image-0003 0 0 2 0.25 1.0]`,
  `[image-0002 0 0 image-0003 0 0 0 1.0 0.5]`, `[image-0002 0 0 image-0003 1 2 3 1.0 1.0]`,
  `[image-0002 0 0 image-0003 1 2 1 1.0 2.0]`
- `(image-0002, 2)` — indices `14,15,16,31,32,33` (6 removed, 48 retained):
  `[image-0000 1 1 image-0002 2 1 1 0.25 1.0]`, `[image-0000 1 1 image-0002 2 1 0 0.5 1.0]`,
  `[image-0000 1 1 image-0002 2 1 2 1.0 0.5]`, `[image-0001 2 1 image-0002 2 1 3 1.0 1.0]`,
  `[image-0001 2 1 image-0002 2 1 1 2.0 0.5]`, `[image-0001 3 1 image-0002 2 1 0 3.0 0.5]`
- `(image-0002, 3)` — indices `12,13,18,19,29,30,36,37,50,51,52,53` (12 removed, 42 retained):
  `[image-0000 0 0 image-0002 3 2 1 0.25 1.0]`, `[image-0000 0 0 image-0002 3 2 3 0.5 1.0]`,
  `[image-0000 2 2 image-0002 3 2 0 0.25 1.0]`, `[image-0000 3 2 image-0002 3 2 2 1.0 0.5]`,
  `[image-0001 1 0 image-0002 3 2 1 0.25 1.0]`, `[image-0001 1 0 image-0002 3 2 3 0.5 0.5]`,
  `[image-0001 4 2 image-0002 3 2 0 0.125 1.0]`, `[image-0001 4 2 image-0002 3 2 2 0.5 1.0]`,
  `[image-0002 3 2 image-0003 0 0 3 0.125 0.5]`, `[image-0002 3 2 image-0003 0 0 1 1.0 2.0]`,
  `[image-0002 3 2 image-0003 1 2 0 0.125 0.25]`, `[image-0002 3 2 image-0003 1 2 2 0.125 0.5]`

## Limitations

- Pair hypotheses, not independent image correspondences or truth recall; no
  ground truth, evaluator, label, or visible-state score was used.
- One fixture, one run. Treatment and control ranges are descriptive; no
  significance, effect size, or causal claim is made.
- Controls are matched on removed count per baseline world axis only, not on
  endpoint degree or residual contribution, so residual differences are not
  identity-controlled.
- Two of eight endpoints admit no legitimate matched control (`image-0000`
  plane 1, `image-0002` plane 2): their axis-1 eligible pool (3) is smaller than
  the treatment count (6), recorded as typed `control_unavailable`.
- Where the required-axis eligible pool equals the treatment count
  (`image-0000` plane 0) the control draw is degenerate, so that endpoint's
  treatment lies outside a single-point control range by construction.
- Nothing here authorizes a matcher, threshold, gate, solver, production,
  project, or promotion change; geometry remains unavailable for this fixture.

## Provenance

- Output `.tmp-ds-r5g-20261003.json`: 440,460 bytes, file SHA-256
  `fd6ed849d9ccb20f2b27e05d810ed8cf676ac674aad6d7e56675fd0d700010bc`; its
  internal canonical `artifact_sha256` seal independently replays as
  `50a5e06e0c88d3df2771df0583aa580eef13f5829e7f2d2be0cb3a21bcf6fae6`
  (status `complete`).
- Producer hashes recorded in the output: utility `983640e2…ac67`, focused test
  `7d542351…e58c`, contained runner `901d73d4…ef8f7`, Job helper
  `e33456d8…e67b6`, DS-R5F utility `eb1eb7f2…8252`, DS-R5F test
  `4db83073…83e6`, entry `5ced5b2b…f0962`, lattice measure `c782edbf…16e673`,
  signatures `86172efb…a69e245`, graph producer `7d4579f8…cb5acb`, graph
  attribution `65dffda3…e2438b`, `plane_graph.py` `9b19b63d…d1d90`,
  `lattice_planes.py` `ddf95ff9…02d01`.
- Containment: `limit_bytes=8589934592`, `deadline_seconds=600`, reviewed
  contained runner and Job helper; worker verified `IsProcessInJob` before image
  processing.
- Gates unchanged: `max_shift_cells=32`, `integer_tolerance=0.15`,
  `min_resultant=0.9`, `ambiguous_margin=0.02`, `min_overlap_cells=9`,
  `min_structural_share=0.15`.
- Focused tests were not rerun in this session by explicit session instruction;
  the lead pre-run gate records `27 passed` plus `py_compile` on these exact
  utility/test hashes.

## Status

Evaluation-only evidence complete and stopped for lead review. No matcher,
solver, threshold, production wiring, project, or artifact change is authorized
by this result. Geometry, promotion and visible acceptance remain unmet.
