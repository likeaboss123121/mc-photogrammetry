# M4 G1F-A preregistration: physical axis-solve budget accounting

Date: 2026-09-13

## Frozen implementation and synthetic checks

This package changes only internal G1C selector accounting. `max_axis_solves`
now limits unique per-axis objective computations admitted on objective-cache
misses. `axis_solve_requests` separately records per-axis requests, including
requests satisfied from the per-call cache. Cache hits consume no physical
solve budget. Before a cache miss enters the objective, its exact number of
axis solves is computed and the whole amount is preflighted; exhaustion must
not expose a candidate, Pareto member, or certificate. Public selection keeps
the natural traversal as its default. The constraint-first path and the
uncached/reference path remain diagnostic-only.

Frozen source SHA-256 values before the R4 call:

- Selector `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`:
  `04a0114a184d622d4a8e6bb669377a672c3c35c35e1d7aff8416d572400bf485`.
- Focused tests `tests/test_joint_plane_selector.py`:
  `ee877c3d4a61e5bffb79a171c4620c6a35563636ce7ecc6f388aa0256692dff4`.
- Synthetic profiler `scripts/profile_m4_g1f_a_axis_budget.py`:
  `74ee89ad7a6c068e07414f500c6be60fe8b95eb9dc1ce3b23712a9217f0d86a2`.
- Synthetic fixture source `scripts/benchmark_m4_g1c_selector.py`:
  `65170810a4786a17f31117461bf4690c38968476bd07f56e37a69e4a3a0be933`.
- Paired profile `exports/evaluation/m4-g1f-a-axis-accounting-profile-2026-09-13.json`:
  `0604bba5f193a1ad1e9ff9bf5c7b0e8bbd2e0844e2c4ca3a15054e93c4d0f123`.

Focused command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

It passed 117 tests. `py_compile` passed for the selector, selector tests, and
synthetic profiler. Tests cover uncached request/solve parity, exact miss
preflight, cache-hit reuse, q+2 and distinct row-subset cache separation,
whole-result parity against the Cartesian oracle, permutation determinism,
and a shared multi-pair reading with b domains 4 through 7.

The paired synthetic profile compares the same numeric `max_axis_solves`
limits under prior request accounting and new physical accounting, for natural
and constraint-first variable order. On cache-repeat-4 with limit 189, old
request accounting abstains after 189 requests/84 physical solves, while
physical accounting completes the exact 91-member `b_unresolved` Pareto result
at 189 physical solves, 19,149 requests, 18,960 cache hits, and 189 misses;
its certificate and ordered candidates match the unbounded oracle. On
shared-subset-tie-4 with limit 512, request accounting stops after 510
requests/210 solves, while physical accounting completes the same 64-member
`b_unresolved` result at 264 solves, 4,032 requests, 3,768 hits, and 264
misses. Natural and constraint-first full outputs matched under both modes on
all eight profiled workloads. Shared-reading b domains 4/5/6/7 produced
1,740/2,745/3,978/5,439 requests but only 108/135/162/189 physical solves.
Profiled runtime and peak traced memory are recorded in the profile artifact;
these synthetic measurements do not establish a production default budget.

## Single same-numeric-budget R4 selector call

The sole input is the frozen R4 JSON record set at
`exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json`,
SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`,
artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`.
The mask-sidecar digest is checked without deserializing its payload:
`f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`.
Expected selector records are 8 frames, 33 readings, and 200 match rows.
Existing G1D and G1E-B selector outputs remain read-only and are pinned at
SHA-256 `2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c`
and `a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3`,
respectively. G1D/G1E-B used the same numeric limits but interpreted the
axis limit as requests.

Exactly those limits are used once for this probe: 4,096 search nodes;
1,000,000 compatibility checks; 20,000 objective rows; 512 axis solves;
64 Pareto comparisons; 16 candidate certificates; and 1,024 buffered match
rows. The public natural-order selector is invoked once. The fresh output
`exports/evaluation/m4-g1f-a-r4-axis-accounting-probe.json` and its `.tmp`
must both be absent before execution. The probe reads only serialized
PlaneReading/PlaneMatch records, and hash-checks the sidecar without opening
its payload. It must not load PNGs, context/target ground truth, evaluator
results, capture manifests, or labels. It records requests, physical solves,
cache signatures/hits/misses, all other work counters, status/reason, elapsed
time, candidate/Pareto counts, and certificate. If the new physical limit
exhausts, the result must be wholly empty. Existing R1-R4, E1-E3, G1D,
G1E-B, and captured-image artifacts are preserved. No R5, evaluator call,
production wiring, default-budget/schema change, staging, or commit is allowed.

The exact probe script and preregistration hashes, result, and post-call input
checks will be recorded in the completion report after the single call.
