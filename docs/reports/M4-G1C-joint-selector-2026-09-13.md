# M4 G1C: isolated exact joint selector

Date: 2026-09-13

## Scope and contract

Implemented an internal prototype in `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`. It is not exported or imported by reconstruction production modules. The selector jointly enumerates an anchored quadrant per frame and one blocks-per-period value per touched reading. Every complete label induces all compatible pair records; callers cannot select or discard edges independently.

`SearchBudget` has no default values and is required by the bounded API. It separately limits search nodes, compatibility checks, objective rows, axis solves, Pareto comparisons, candidate certificates, and buffered match rows. Any exceeded counter returns `budget_exhausted` with no candidate, Pareto set, or certificate hash. Exact duplicate evidence rows are idempotently removed and counted. Conflicting rows sharing pair/quadrant/b labels are rejected as invalid rather than double-weighted or max-selected.

The oracle traverses the full Cartesian q/b label product without the bounded search's feasibility pruning. Both paths use the existing frozen `plane_graph._evaluate_assignment` geometry/appearance rules, then apply the G1B-R all-pairs Pareto relation. All-pairs comparison retains every candidate until reduction because shared-evidence dominance can be non-transitive. Coverage inclusion is necessary; it is not a score. A strict improvement on at least one shared geometry/appearance objective is also required, so extra-only groups and exact ties do not dominate.

## Verification

Focused command:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\reconstruction\_joint_plane_selector.py scripts\benchmark_m4_g1c_selector.py tests\test_joint_plane_selector.py
.\.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

Result after review corrections: 87 passed. The selector suite covers unique/no candidate, q+2, tree and disconnected inputs, false-node/dense-wrong/superset/tie/tradeoff cases, duplicate rules, deterministic permutations, low budgets for nodes, certificates, buffered rows, compatibility checks, objective rows, axis solves, and Pareto comparisons, a 12-case systematic q/b covering set, 12 fixed-seed generated two-frame bounded-versus-Cartesian comparisons, and a three-frame multi-b bounded-versus-Cartesian case. An inconsistent `PlaneMatch` axis is rejected against its referenced `PlaneReading` before either bounded pruning or legacy objective evaluation, and both paths return the same typed `invalid_input`. The dense-worse and false-node controls check all-pairs reduction under a deliberately non-transitive dominance relation. An AST guard verifies the internal selector is not imported anywhere else under `src/mc_photogrammetry`.

## Synthetic full-stage profile

`scripts/benchmark_m4_g1c_selector.py` constructs synthetic readings/matches only. It reads no capture, manifest, reference, R4 freeze, evaluator, or label. Run command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_m4_g1c_selector.py --output exports\evaluation\m4-g1c-synthetic-profile-2026-09-13.json
```

The recorded profile uses Python 3.10.11 on Windows 10, with `tracemalloc` enabled. It includes induced-record enumeration, the existing least-squares/resultant/residual evaluation, shared-group objective recomputation, all-pairs Pareto comparisons, and certificate construction where the caller budget permits:

| Workload | Result | Main work | Time | Peak traced memory |
|---|---|---|---:|---:|
| 8-frame strong prune, 28 records | `no_candidate / insufficient_axis_coverage` | 1 node, 28 checks, 0 objective rows/axis solves | 0.001015 s | 62,295 B |
| 4-frame worst tie, 36 records | `b_unresolved`, all 64 candidates retained | 85 nodes, 64 labels, 4,032 Pareto comparisons, 1,280 shared evaluations, 12,096 objective rows, 4,032 axis solves | 1.047283 s | 2,751,811 B |
| 8-frame tie budget probe, 168 records | `budget_exhausted / candidate_certificates_buffered` | Caller limit 64; 71 complete labels, 65 candidates evaluated, 64 buffered; 3,723 buffered rows; Pareto did not start | 0.247897 s | 1,230,199 B |

The 8-frame tie probe demonstrates deterministic whole-search abstention without a partial certificate; it is not represented as a full Pareto-stage timing. The full tie stage is profiled on four frames to keep the all-pairs workload bounded.

For lead review only, a conservative explicit caller budget candidate is: 32,768 search nodes; 5,000,000 compatibility checks; 200,000 objective rows; 10,000 axis solves; 1,560 Pareto comparisons; 40 candidate certificates; and 2,520 buffered match rows. The proposed candidate/row/comparison limits correspond to at most 40 buffered candidates with 63 compatible rows each and 40×39 ordered comparisons. At the measured 4-frame tie rates, 200,000 objective rows are about 17 seconds; 5 million compatibility checks at the 8-frame probe rate are about 55 seconds. Allowing an 8× axis-solve cost for the increase from four to eight frames keeps the summed rough estimate below 100 seconds. This is an extrapolation, not a guarantee: the 8-frame tie profile intentionally stops before Pareto work, so these values are **not activated** and need lead review plus representative 8-frame profiling before any production caller is considered.

## Artifacts and hashes

- Selector: `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`, SHA-256 `30338b9ed5851526277a5d65d49a592583ac51f10756904cc79a33f82b4d3bba`.
- Tests: `tests/test_joint_plane_selector.py`, SHA-256 `24b6282a55e7a32e31bfdf9bbfd9589526ba256dce99eaa6baa2c3c27b15f515`.
- Synthetic benchmark: `scripts/benchmark_m4_g1c_selector.py`, SHA-256 `65170810a4786a17f31117461bf4690c38968476bd07f56e37a69e4a3a0be933`.
- Profile JSON: `exports/evaluation/m4-g1c-synthetic-profile-2026-09-13.json`, SHA-256 `dede128db00582003a6234e9125e85b4747893ef6fe219c86aee35cb9bc55064`. It embeds the corrected selector and benchmark hashes.

No production solver/runner wiring, schema changes, capture, evaluator, R5, staging, or commit was performed. R1-R4/E1-E3 remain untouched.
