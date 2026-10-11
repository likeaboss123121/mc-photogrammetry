# M4 G1E-A: canonical row masks and exact objective memoization

Date: 2026-09-13

## Scope and exactness

The isolated bounded selector now compiles each prepared run's canonically sorted unique match rows into integer bit masks for frame-pair quadrant compatibility, world-axis support, and assigned reading-scale compatibility. Candidate extraction visits set bits in increasing canonical row index, preserving the former row order. Partial possible-axis checks use the corresponding precompiled masks. The exhaustive oracle and an uncached bounded reference path retain the original row-by-row logic.

Objective results are memoized only within one `_Work` call. Keys contain the complete anchored absolute quadrant tuple in prepared frame order and the exact ordered canonical row-index tuple passed to the objective. Pareto shared-subset calls use the exact filtered row sequence in that key, so pair-group equality alone cannot collide. Null objective results are cached too. No q/b variable ordering, objective, gate, candidate/Pareto rule, production wiring, or default budget changed.

Logical counters and limits retain G1C meaning. The mask path charges the same per-row compatibility work as the canonical scan, including the partial counter value at a finite-limit stop. Objective-row, axis-solve, shared-evaluation and Pareto charges still occur on cache hits as if the underlying computation were repeated. New physical counters are diagnostic only: `objective_cache_hits/misses/unique_signatures` record whole-objective signatures; `physical_axis_cache_hits/misses` count axis result sets reused from or computed on objective cache hits/misses. These counters do not consume a logical budget.

## Tests and paired synthetic profile

Focused command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

All **96 tests passed** (exit 0). `py_compile` passed for the selector, focused tests, and G1E-A profile script. Added mask-vs-scan parity for complete and partial q/b labels; optimized-vs-uncached bounded parity and logical counter checks; exact q/q+2, same-pair row-subset and cross-call cache-key controls; a full repeated-signature workload; adjacent-float geometry gate controls; and whole-search abstention parity at every finite work limit, including a mid-scan compatibility limit.

The paired synthetic profile compares the mask/cache path against the uncached canonical scan using the existing strong-prune, tied four-frame, and eight-frame tie-budget controls. All three matched status, reason, certificate, candidate records, and logical counters.

| Synthetic workload | Legacy time / peak heap | Mask+cache time / peak heap | Physical cache activity |
|---|---:|---:|---|
| Strong-prune, 8 frames | 0.000884 s / 39,968 B | 0.008428 s / 447,984 B | 0 hits, 0 misses; mask preparation dominates this tiny workload |
| Shared-subset tie, 4 frames | 1.053141 s / 2,611,253 B | 0.163103 s / 3,242,136 B | 1,256 objective hits / 88 misses; 3,768 axis-result hits / 264 misses |
| Tie-budget probe, 8 frames | 0.268182 s / 1,444,704 B | 0.259077 s / 1,940,555 B | 0 hits / 71 misses; both abstained at the same certificate-buffer limit |

The cache helps the tie-heavy workload by about 6.5x in this run, at higher peak heap. It adds noticeable setup cost to a case rejected immediately by the possible-axis bound, so this profile supports only the exact-search compute benefit on repeated objectives, not a universal speed or memory claim. The eight-frame case remains an identical budget abstention; no larger cap is proposed.

Authoritative paired profile: `exports/evaluation/m4-g1e-a-paired-profile-2026-09-13.json`, SHA-256 `bab100bf5fce47876ab77458eea2a3be1ee89ce49d879d3db00578f763ebd80a`. A preceding optimized-only profile in `exports/evaluation/m4-g1e-a-synthetic-profile-2026-09-13.json` is superseded by this paired comparison.

## Source hashes and stopping point

- Selector: `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`, SHA-256 `68e1f76dd77f10c18e70b4e91bed2f5fa3ec7980fccf49a502129e1aa201e89c`.
- Tests: `tests/test_joint_plane_selector.py`, SHA-256 `ab9c6e703421269121119cf5b91abe4f42f3e4ce10d70c89f65bd3fbd1af0f49`.
- Profile runner: `scripts/profile_m4_g1e_a_selector.py`, SHA-256 `32264b1483127c137022c355bcb9969d07136a1f4c502dea229978166fd8de49`.
- Synthetic fixture source: `scripts/benchmark_m4_g1c_selector.py`, SHA-256 `65170810a4786a17f31117461bf4690c38968476bd07f56e37a69e4a3a0be933`.

R4/R5, images, labels, evaluators, and production graph wiring were not used or changed. No default cap was selected. This package stops for lead review.
