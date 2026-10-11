# M4 G1E-B: constraint-first q/b ordering and incremental row masks

Date: 2026-09-13

G1E-B adds an isolated exact search path beside the retained G1E-A natural-order path. It orders every non-constant q/b variable statically by `(domain_size, -incident_canonical_row_count, kind_rank, canonical_key)`. The anchored first-frame q and singleton b values are applied at the root; each branch intersects the current canonical feasible-row bitmask with only the changed variable's constraints. Partial possible-axis pruning still uses the exact pair/quadrant/axis masks, so its set matches the prior reference bound. At complete labels, rows are emitted in canonical order and the existing candidate, objective, Pareto, certificate, and whole-search-abstention code remains in force. Missing labels remain wildcards. Production callers continue using the G1E-A natural-order path.

Focused regressions cover systematic quadrant and b-domain cases, q/q+2 ambiguity, three-frame multi-b search, tree/disconnected evidence, frame relabel and input permutation, deterministic repeated runs, every finite budget exhaustion, and exhaustive partial-mask versus reference-axis/active-row parity on three- and four-frame inputs. Full outputs match the Cartesian oracle, including status/reason/certificate and ordered Pareto records. No label/image/evaluator data enters the synthetic profile.

Focused command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

Result: **108 passed**, exit 0. `py_compile` passed for the selector, tests, profile runner, and R4 probe. The paired profile compares G1E-A against G1E-B on four synthetic workloads. The shared-subset tie-4 matched all 64 Pareto candidates and certificate; runtime was 0.129883 s versus 0.127125 s and peak traced memory 3,133,296 B versus 2,820,092 B. The binary-b tree-4 matched all 592 candidates and certificate; runtime was 4.306218 s versus 4.233761 s, with 1,195 versus 1,375 visited nodes, 27 versus 432 insufficient-axis prunes, and 9,392,383 B versus 8,495,369 B peak traced memory. Compatibility checks rose from 32,166 to 35,406 on that workload. The strong-prune case and eight-frame 64-certificate budget abstention also matched exactly. These results show exact ordering/mask parity and a small workload-dependent memory/time change, not a general speedup.

## One same-budget R4 comparison

Before the single selector call, the preregistration pinned the R4 freeze SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, sidecar SHA-256 `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`, selector/tests/profile hashes, synthetic profile, G1D parsing helper, and preserved G1D output. All seven G1D limits were reused. The probe parsed only the frozen JSON reading/match records and verified the sidecar digest without opening its payload. It opened no PNG, GT, evaluator, manifest, or labels.

The G1E-B probe returned `budget_exhausted` at `axis_solves`, with no candidate, Pareto entries, or certificate. Relative to G1D under the same limits, it reached the same 171 complete labels and 510 axis solves in 0.018426 s instead of 0.074387 s; search nodes fell from 384 to 230 and compatibility checks from 111,000 to 80,200. Objective-row counts were 1,337 versus 1,335. G1D had 37 partial-axis prunes; G1E-B had none. The constraint-first traversal changed where the same budget was spent, and this remains an incomplete computational diagnostic, not evidence of a valid or invalid R4 solution. The 19 unique objective signatures had 152 cache hits; 57 physical axis-result misses produced 456 hits.

For R4, the deterministic order had 31 variables: 13 b variables with domains of size 2 or 3 first, all seven non-anchor q variables next, then the remaining 11 b variables with domains of size 4–7. Within each group, incident-row degree and canonical key broke ties.

## Frozen artifacts and hashes

- Selector: `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`, SHA-256 `c562fe2324562294be6b68cd4bb0d95234038965d2315f13dcc7a86b5183757e`.
- Tests: `tests/test_joint_plane_selector.py`, SHA-256 `7823f0a519b04b227405e9f915f8198ccc66b0e35d7a480dcd564efea3b0d4fd`.
- Profile runner: `scripts/profile_m4_g1e_b_selector.py`, SHA-256 `e168cc0002883ca292943c10525986ac85fb06ebc0a4bd397a89ea5201e5d194`.
- Paired synthetic profile: `exports/evaluation/m4-g1e-b-paired-profile-2026-09-13.json`, SHA-256 `2a9376605c64930f7cc3d816709d6efa6b77ff8ea62a36f277138b470573bfd5`.
- Preregistration: `docs/reports/M4-G1E-B-r4-selector-prereg-2026-09-13.md`, SHA-256 `74c1f0760258f2843da5b5f96c74242ff1299ac67533bc301e44316f4d80b31e`.
- R4 probe: `scripts/probe_m4_g1e_b_r4_selector.py`, SHA-256 `4fe79159ac9688095892b299aae4e0b1ecf381f2691a205fe986e8f677e6c8dc`.
- R4 probe output: `exports/evaluation/m4-g1e-b-r4-selector-probe.json`, SHA-256 `a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3`.
- G1D comparison output remained SHA-256 `2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c`.

The independently recorded protected inventory reports all 20 R1–R4 freeze/sidecar, E1–E3 result, and ordered PNG hashes unchanged. No production wiring, default budget, schema migration, evaluator, image/GT scoring, or R5 was performed. No staging or commit.
