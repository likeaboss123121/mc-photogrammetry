# M4 G1F-A: physical axis-solve budget accounting

Date: 2026-09-13

G1F-A changes internal selector accounting so `max_axis_solves` limits unique
per-axis objective computations on cache misses. `axis_solve_requests` reports
all requested per-axis results, including those supplied by cache hits. A miss
preflights its exact number of axis solves before invoking the objective; if
the bound would be exceeded, the selector returns whole-search abstention with
no partial candidate, Pareto set, or certificate. The public natural-order
path stays the default. No production wiring, default budget, or schema
migration was made.

## Validation and synthetic profile

Focused regression command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

Result: **117 passed**, exit 0. Selector-only tests also passed. `py_compile`
passed for the selector, selector tests, synthetic profile runner, and R4 probe.
Regressions cover no-cache request/solve equivalence; distinct q+2 and row-set
cache keys; exact miss-limit preflight without partial work; request-budget
versus physical-budget behavior against the exhaustive oracle; deterministic
permutation behavior; and a shared multi-pair reading with b-domain sizes 4,
5, 6, and 7.

Synthetic profile command:

```text
.venv\Scripts\python.exe scripts\profile_m4_g1f_a_axis_budget.py --output exports\evaluation\m4-g1f-a-axis-accounting-profile-2026-09-13.json
```

The paired profile ran natural and constraint-first orders with old request
accounting and new physical accounting. On cache-repeat-4 with limit 189,
request accounting stopped at 189 requests/84 physical solves; physical
accounting completed the exact 91-candidate `b_unresolved` Pareto result at
189 physical solves, 19,149 requests, 18,960 physical cache hits, and 189
misses. Its ordered candidates and certificate matched the unbounded oracle.
On shared-subset-tie-4 with limit 512, request accounting stopped at 510
requests/210 solves, while physical accounting completed the exact
64-candidate `b_unresolved` Pareto result at 264 solves and 3,768 cache hits.
Natural/constraint-first outputs matched for both modes on all eight
synthetic workloads. Shared-reading b domains 4–7 generated 1,740, 2,745,
3,978, and 5,439 requests against 108, 135, 162, and 189 physical solves.
These synthetic results verify accounting and search parity; they do not
justify a default production budget.

## One selector-only R4 call

The single preregistered command was:

```text
.venv\Scripts\python.exe scripts\probe_m4_g1f_a_r4_selector.py
```

It loaded only the frozen R4 JSON's eight frame, 33 reading, and 200 match
records. The R4 freeze SHA-256 was
`96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, artifact
ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`; the
sidecar was digest-checked as
`f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac` without
loading its payload. No PNG, GT, evaluator, capture manifest, or label was
opened. The same seven numeric limits as G1D/G1E-B were used, including
`max_axis_solves=512`; this run used the public natural-order selector.

The call returned `budget_exhausted` at `compatibility_checks=1,000,000` after
3,492 nodes and 1,509 complete labels. It recorded 11,532 objective rows,
4,527 axis-solve requests, 168 physical solves across 56 objective signatures,
4,359 physical axis-cache hits, and 1,453 objective-cache hits. No candidate,
Pareto member, or certificate was returned. Runtime was 0.098994 seconds.
Thus the physical axis-solve limit did not bind; the compatibility-check
limit stopped this selector traversal first. For context, G1D and G1E-B under
the same numeric limits stopped at the former request-counted axis limit after
510 requests, with 384/230 nodes and 171 complete labels. The new R4 run is an
incomplete computation, not a solution-quality finding.

## Frozen artifacts

- Preregistration: `docs/reports/M4-G1F-A-r4-selector-prereg-2026-09-13.md`,
  SHA-256 `44b7ab24e33809073a830d71257a10bfdcc8ef5d1c4d84b72be78bc18779618b`.
- Selector: `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`,
  SHA-256 `04a0114a184d622d4a8e6bb669377a672c3c35c35e1d7aff8416d572400bf485`.
- Tests: `tests/test_joint_plane_selector.py`, SHA-256
  `ee877c3d4a61e5bffb79a171c4620c6a35563636ce7ecc6f388aa0256692dff4`.
- Synthetic profile runner: `scripts/profile_m4_g1f_a_axis_budget.py`,
  SHA-256 `74ee89ad7a6c068e07414f500c6be60fe8b95eb9dc1ce3b23712a9217f0d86a2`.
- Synthetic profile: `exports/evaluation/m4-g1f-a-axis-accounting-profile-2026-09-13.json`,
  SHA-256 `0604bba5f193a1ad1e9ff9bf5c7b0e8bbd2e0844e2c4ca3a15054e93c4d0f123`.
- Probe script: `scripts/probe_m4_g1f_a_r4_selector.py`, SHA-256
  `280e7dde2d3b8f92d37fb2ded6f2050dfb71562bad9e7a945d68b1c97c58b717`.
- Probe output: `exports/evaluation/m4-g1f-a-r4-axis-accounting-probe.json`,
  SHA-256 `8aea1e7cccb2e979744a611f0c7c6685960cc9fd236b741c1c3066f2b543b48d`.
- Prior G1D and G1E-B outputs remain unchanged at SHA-256
  `2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c` and
  `a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3`.

The probe rechecked all pinned inputs after execution. R1-R4, E1-E3, G1D,
G1E-B, and image artifacts were not modified. No R5/evaluation, production
integration, stage, or commit was performed.
