# M4 G1E-B preregistration: ordered incremental-mask selector probe

Date: 2026-09-13

## Frozen implementation and synthetic evidence

This comparison evaluates only the isolated G1E-B bounded selector against
the same-budget G1D call. The selector source is frozen at SHA-256
`c562fe2324562294be6b68cd4bb0d95234038965d2315f13dcc7a86b5183757e` and its
focused test source at `7823f0a519b04b227405e9f915f8198ccc66b0e35d7a480dcd564efea3b0d4fd`.
The paired synthetic profile runner is
`e168cc0002883ca292943c10525986ac85fb06ebc0a4bd397a89ea5201e5d194`; its
output is `2a9376605c64930f7cc3d816709d6efa6b77ff8ea62a36f277138b470573bfd5`.
It compared G1E-A natural order with the constraint-first path on four
synthetic workloads and matched full selector outputs on every workload.
The frozen order is `(domain_size, -incident_canonical_row_count, kind_rank,
canonical_key)`. The mask path applies the anchored q and singleton b values
at root, incrementally intersects each assigned label, and preserves the
reference partial-axis set with exact pair/q/axis masks. No candidate or
quadrant/b label is removed by ordering.

The focused suite is 108 tests across selector, graph, lattice-entry,
supported-plane, and lattice-plane modules. Command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_supported.py tests\test_lattice_planes.py -q
```

It passed 108/108. `py_compile` passed for the selector, selector tests, and
profile runner. The 4-frame binary-b tree workload retained the same 592
Pareto candidates/certificate; the ordered path visited 1,375 nodes versus
1,195, with 432 versus 27 axis prunes, so this is an exactness experiment,
not a claimed speed win. The 4-frame shared-subset tie and 8-frame tie-budget
outputs/abstention also matched. Profile hash above is pinned.

## R4 input and one-call protocol

The sole R4 input is the already-frozen selector record set in
`exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json`
(SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`,
artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`).
Its content-addressed mask sidecar is hash-verified only, never read: SHA-256
`f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`.
Expected solver records remain 8 frames, 33 readings, and 200 match rows.
The prior G1D same-budget output is preserved at
`exports/evaluation/m4-g1d-r4-selector-probe.json`, SHA-256
`2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c`.
It exhausted at `axis_solves` after 384 nodes/171 complete labels and 510 of
512 axis solves (0.0743875 seconds), returning no partial candidate.

Exactly the G1D limits are reused: 4,096 search nodes; 1,000,000 compatibility
checks; 20,000 objective rows; 512 axis solves; 64 Pareto comparisons; 16
candidate certificates; and 1,024 buffered match rows. A single invocation
uses only serialized PlaneReading/PlaneMatch records. The R4 JSON and sidecar
are identity/hash checked; no PNG, sidecar payload, ground truth, evaluator,
capture manifest, or labels are opened. Output path
`exports/evaluation/m4-g1e-b-r4-selector-probe.json` must be absent before the
call. Budget exhaustion must expose no candidate, Pareto entry, or certificate.
R4, G1D, and all frozen image/evaluation artifacts are read-only. No production
wiring, default budget, schema migration, or R5 is authorized.

The exact probe script hash and this byte-exact preregistration hash will be
written into the output and completion report after the script is finalized.
