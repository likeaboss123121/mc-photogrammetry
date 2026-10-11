# M4 G1D: low-budget R4 selector-only probe

Date: 2026-09-13

## Frozen protocol and inputs

The single selector call followed [the preregistration](M4-G1D-r4-selector-probe-prereg-2026-09-13.md), SHA-256 `f470d4be43b478a07a840a5c850afc24cd7408eafe0aaed2162edf6a56d8087b`. It deserialized only the frozen R4 records and hash-verified the sidecar. It did not open any PNG or ground truth and did not call the evaluator. The strict-hash probe rechecked the freeze and sidecar before and after the selector call.

R4 freeze SHA-256: `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`; artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`. Sidecar SHA-256: `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`. Corrected isolated selector SHA-256: `367329861d3838ed3df360bde9b2735242f611ce0e25bfecb68cf26ac2f0298b`. Selector test SHA-256: `62eb03d5d252163663ab26fd7370de511fcb6dc4feb4ecc3fb45427d1f408933`. Probe script SHA-256: `917f5d51bffab1dec7fa65a120ab7769ba67c68ea70cf0e761fc8a7c02ec9dee`.

The first script attempt stopped in input preflight before calling the selector because six records have `psr=null`. R4 has `integer_residual=null` in all 200 matches. The preregistration was amended before any selector call. The isolated selector now preserves optional null PSR as a tagged sort/dedup field without substituting a numeric value; score, axes, and `b` remain finite-required. Added a bounded/oracle/permutation regression. Focused G1C command `\.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q` passed **88 tests** (independent clean run, exit 0). The null-PSR selector test command passed **20 tests**.

## One-call result

The probe ran once with the preregistered limits: 4,096 search nodes, 1,000,000 compatibility checks, 20,000 objective rows, 512 axis solves, 64 Pareto comparisons, 16 buffered candidate certificates, and 1,024 buffered match rows. Input size was 8 frames, 33 readings, and 200 matches. It terminated `budget_exhausted` at `axis_solves` after 0.0744 seconds. Counters at termination were:

| Counter | Result |
|---|---:|
| Search nodes | 384 |
| Complete labels | 171 |
| Pruned for insufficient possible axes | 37 |
| Compatibility checks | 111,000 |
| Objective rows | 1,335 |
| Axis solves | 510 of 512 |
| Evaluated candidates | 0 |
| Buffered certificates / match rows | 0 / 0 |
| Pareto comparisons | 0 |

The result contained no candidate, Pareto entries, or certificate (`candidate_count=0`, certificate null), satisfying the whole-search abstention contract. This is an incomplete search, so it establishes neither that a full-budget candidate exists nor that no candidate exists. Probe JSON SHA-256: `2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c` at `exports/evaluation/m4-g1d-r4-selector-probe.json`.

## Exact acceleration design for review

No optimization code was edited. The probe shows that the low budget reached the axis-solve ceiling before candidate buffering or Pareto comparison. A behavior-preserving implementation candidate is to compile each canonical match row's quadrant/b compatibility into immutable row-index bitsets. At each partial assignment, intersect those masks to derive the same possible-axis set; at complete labels, enumerate the surviving rows in canonical order to produce exactly the current active-match tuple. This changes representation and lookup cost only: it must retain every label combination, candidate, duplicate rule, geometry gate, and Pareto comparison.

A second safe cache candidate is exact memoization of objective calculations by the full anchored quadrant tuple and the canonical active-match row IDs, with a separate exact key for each shared-row subset used by Pareto comparison. Cached values may replace only repeated calls to the unchanged objective function. Candidate labels, translations, all incomparable candidates, and q+2 alternatives must still be emitted and compared. The low-budget probe did not reach Pareto processing and therefore gives no evidence that shared-subset caching helps this R4 input.

Constraint-first q/b variable ordering is another exact completed-search option: deterministically order all original variables by a fixed constraint-degree/domain-size rule, enumerate every value, and retain whole-search abstention on any budget exhaustion. Completion must match the exhaustive G1C oracle exactly; budget outcomes are deterministic for the chosen ordering, but their node/counter path may differ from the present frame-then-b order. No top-q/top-b trimming, score pruning, q+2 quotienting, candidate deletion, cap selection, or production wiring is authorized by this probe. Any implementation requires focused oracle parity plus full-stage profiling; no numeric production budget is proposed here.
