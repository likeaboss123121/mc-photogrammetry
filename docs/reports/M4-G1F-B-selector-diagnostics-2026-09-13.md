# M4 G1F-B: selector diagnostics census

Date: 2026-09-13

G1F-B added observational diagnostics to the isolated exact selector: mutually exclusive complete-label gate counts and bounded exact signatures for partial states and complete leaves. Neither signature class prunes or merges search states. Search order, objective, candidate/certificate/Pareto handling, and budget behavior were preserved. The signature census stores at most 4,096 signatures per class and reports any overflow separately.

The focused verification passed: targeted G1F-B tests, 11 passed; selector/graph/lattice-entry/supported/lattice-planes regression command, 124 passed; `py_compile` passed for selector source, selector tests, and the probe script. The full command was:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

One preregistered selector-only call consumed the frozen R4 serialized records (8 frames, 33 readings, 200 matches) with the same numeric budget as G1F-A. The terminal status was `budget_exhausted`, reason `compatibility_checks`; the limit was 1,000,000. It visited 3,492 nodes and reached 1,509 complete-label attempts. Terminal gates were mutually exclusive and reconciled: 1,509 minimum-resultant rejections; all other gates, including accepted and budget-aborted, were zero. The partial-state census observed/stored 3,492/3,492 unique signatures; the complete-leaf census observed/stored 1,509/1,509. Both had zero repeats and zero truncations (0% repeated observations).

Work counters were 11,532 objective rows; 4,527 axis requests; 168 physical axis solves (4,359 axis-cache hits); 56 objective signatures (1,453 objective-cache hits); and zero candidates, Pareto comparisons, certificates, or buffered rows. The selector returned no partial candidate state. All legacy counters, status/reason, Pareto count, and certificate matched the immutable G1F-A same-budget result exactly, so this diagnostic change was output-neutral on R4. This exhausted search is incomplete computation and does not establish geometric quality.

The probe did not open images, GT, evaluator outputs, manifests, or labels. The mask sidecar was hash-checked without opening its payload. R4 freeze and G1F-A/G1D/G1E-B outputs remained byte-identical. No production wiring/default budget/schema change, R5, staging, or commit occurred.

Files and hashes:

- Selector: `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py` — `ff33bf73cea5efa690b4e2a171df3f08c7dc9a82036542893bd9baaf0e420d3b`
- Tests: `tests/test_joint_plane_selector.py` — `a09215ef188edf156ade78e090a783041411dcda79fad0360489e572308aa627`
- Preregistration: `docs/reports/M4-G1F-B-selector-diagnostics-prereg-2026-09-13.md` — `5e334c269b515e3de0850b112e2bcbb1991d6be82f6b4b3690135ccdb33ec72c`
- Probe driver: `scripts/probe_m4_g1f_b_r4_diagnostics.py` — `afdbd8db8ca4b1735f1ae57f9e5c02f0a1b8d5b5a46b46d61dc367fff4e8b55a`
- Probe output: `exports/evaluation/m4-g1f-b-r4-diagnostics-probe.json` — `f237439d544f4f0f674b20e2765eae286b36800c683d18ce0acd8467a48f31a1`
- Frozen input: `exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json` — `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`; artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`.