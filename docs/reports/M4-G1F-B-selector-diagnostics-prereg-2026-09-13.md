# M4 G1F-B preregistration: bounded selector diagnostics

Date: 2026-09-13

## Scope and frozen inputs

This is an output-neutral diagnostic change to the isolated exact selector. It adds typed mutually exclusive terminal-gate counters and bounded exact census of partial states and complete leaves. Signatures are observed only; they never prune, merge, reorder, or skip search work. The public natural traversal, complete labels, objective, candidates, certificates, Pareto ordering, and caller budget semantics remain fixed. Signature storage is separately capped at 4,096 per signature class; over-cap observations increment an explicit truncation counter and do not affect search.

Frozen source/test hashes before the single R4 call:

- `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`: `ff33bf73cea5efa690b4e2a171df3f08c7dc9a82036542893bd9baaf0e420d3b`
- `tests/test_joint_plane_selector.py`: `a09215ef188edf156ade78e090a783041411dcda79fad0360489e572308aa627`
- `exports/evaluation/m4-g1f-a-r4-axis-accounting-probe.json`: `8aea1e7cccb2e979744a611f0c7c6685960cc9fd236b741c1c3066f2b543b48d`
- R4 freeze: `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`; artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`
- R4 mask-sidecar digest, checked but payload never read: `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`
- G1D and G1E-B same-numeric-budget outputs: `2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c` and `a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3`.

The single call uses the exact same numeric limits as G1F-A/G1D/G1E-B: 4,096 nodes; 1,000,000 compatibility checks; 20,000 objective rows; 512 physical axis solves; 64 Pareto comparisons; 16 certificates; 1,024 buffered match rows. Only frozen serialized reading/match records are loaded. The mask-sidecar file is hash-checked without opening its payload. No images, GT, evaluator, manifest, or labels are opened. Fresh output and `.tmp` must be absent; no overwrite is permitted.

## Diagnostic definitions and acceptance checks

Each complete-label attempt has one terminal diagnostic gate, with precedence: no compatible records; objective rejected; fewer than three axes; minimum resultant; integer-tolerance/residual; accepted. A work-budget exception while evaluating a label is `budget_aborted`. These counters sum to complete labels when a budget stop occurs inside a label; search-node/compatibility exhaustion before a label attempt is represented only by the ordinary status/reason. Existing status/reason/candidate behavior is preserved.

A partial semantic signature records frame order/depth, exact assigned and unassigned q/b labels, the canonical feasible-row bitmask, active ordered candidate-row indices, and possible axes. A leaf signature records complete q/b labels, feasible mask, active ordered rows, and objective-relevant row fields (row index, assigned world axis, signed face coordinates, match score). Exact equality defines repeats; the census records observations, stored unique signatures, repeats, and truncation. No signature is used as an equivalence class for search.

Tests cover mutually exclusive gate precedence and typed budget aborts; independently reconstruct signature masks/axes/row ordering; distinguish leaves with identical active rows but different exact b labels; bound census storage; assert the diagnostics preserve oracle output; and compare predicted objective-axis count with actual objective axes across empty, infeasible, duplicate, one-, two-, and three-axis evidence with deterministic seeded subsets. Focused verification command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

The preregistered selector-only probe will record the diagnostic gate histogram, partial/leaf observations and duplicate rates, existing logical counters, and terminal limit. All prior frozen artifacts remain immutable. No production wiring/default budget/schema migration, R5, evaluator, staging, or commit is in scope.
## Frozen single-call driver and output

The frozen driver is `scripts/probe_m4_g1f_b_r4_diagnostics.py`, SHA-256 `afdbd8db8ca4b1735f1ae57f9e5c02f0a1b8d5b5a46b46d61dc367fff4e8b55a`. It requires the operator to supply the independently calculated preregistration and driver hashes; both are checked before the call, and the driver checks all other hashes listed above again afterward. The sole command is:

```text
.venv\Scripts\python.exe scripts\probe_m4_g1f_b_r4_diagnostics.py --expected-prereg-sha256 <this preregistration SHA-256> --expected-probe-sha256 afdbd8db8ca4b1735f1ae57f9e5c02f0a1b8d5b5a46b46d61dc367fff4e8b55a
```

Fresh output `exports/evaluation/m4-g1f-b-r4-diagnostics-probe.json` and its `.tmp` must be absent before this command. The driver asserts exact equality of every preexisting `SearchCounters` field, status/reason, Pareto count, and certificate against the immutable G1F-A same-budget output before publishing its result.