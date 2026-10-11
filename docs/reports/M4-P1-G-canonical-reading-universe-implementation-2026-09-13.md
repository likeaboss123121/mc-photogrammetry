# M4 P1-G canonical reading universe implementation

Date: 2026-09-13  
Status: implemented and tested; stopped for lead review  
Contract: [M4-P1-G-canonical-reading-universe-contract-2026-09-13.md](M4-P1-G-canonical-reading-universe-contract-2026-09-13.md), SHA-256 `aa9ad4f6921156220e31704f8d0f10427be91a0ca05975ac6d7f14bf7f2fd50d`

## Change

`match_plane_graph` now requires both the complete `PlaneReading` universe and signatures. It validates the entire universe before correlation: reading type, nonempty string frame, canonical axis, finite/orderable `reading_order_key`, no repeated reading object, and no distinct tied order keys within a frame. It verifies every signature points by exact object identity into that universe and has a finite positive b. Matches receive indices from all readings, including signatureless ones. Each published endpoint is checked against the canonical index, source object, frame, axis, and selected b before and after same-axis filtering. The `PlaneMatch` fields and frozen schema-v1 record are unchanged.

P1-D now requires the same complete readings sequence as keyword-only `readings=` on both `build_shift_domains` and `validate_shift_domain_inputs`. It resolves endpoints through the shared canonical map and then locates the selected signature by exact reading object and canonical b. There is no signature-derived fallback. The all-or-nothing result, budgets, domain enumeration, scoring, correlation replay, and refusal behavior are unchanged except that a previously ambiguous endpoint now resolves correctly or refuses.

Production runner passes its exact `all_readings` sequence. Direct graph callers in `tests/test_plane_graph.py`, `tests/test_lattice_planes.py`, and the runner are migrated. P1-D calls are migrated in `tests/test_plane_shift_domain.py`, `tests/test_plane_grid_map_p1f_integration.py`, `tests/test_c2r_vector_domain_proof.py`, and `tests/test_c2r_vector_domain_matrix.py`. A source/scripts/tests search found no legacy one-argument graph call; the single one-argument call in tests is an intentional assertion that Python raises `TypeError`. P1-D calls likewise all pass `readings=`; the only omissions are intentional interface-rejection tests.

## Identity proof and validation

The retained leading-signatureless case puts unsigned axis-0 readings before signed axis-1 and axis-2 endpoints. The complete expected q0 match set is asserted exactly as `{(1,1,1,1), (2,2,2,2)}` in `(left_axis,left_plane,right_axis,right_plane)` order. A wrapper spy verifies the same-axis filter receives the complete canonical map, including the exact signatureless objects at index 0 and the signed objects at indices 1 and 2. Both matches proceed through the solver and integer-residual annotator, C1 maps the selected identities at indices 1 and 2, and P1-D completes with both indices. A forged legacy-renumbered index-0 match is rejected by solver/C1 and yields an all-or-nothing P1-D refusal. Additional controls cover leading, interleaved, trailing and multiple signatureless readings, input permutations, multiple valid-axis readings with unique order keys, same-axis filtering, repeated alternative signatures for one reading, foreign/missing readings, duplicate objects, tied keys, malformed frames, nonfinite ordering fields, and bool/nonfinite match b values.

The runner spy verifies object-by-object that the exact ordered `all_readings` sequence reaches `match_plane_graph`. A separate spy proves malformed complete-universe data is rejected before correlation begins. A P1-G AST dependency/call test scans both `plane_graph.py` and `plane_shift_domain.py` for forbidden metadata, manifest, F3/coordinate/bounds, reference/GT/evaluator, renderer, CLI/export, filesystem/network/process, and placement surfaces; it inspects imports and call targets, so harmless docstring prose does not trigger it. Frozen schema-v1 runner regressions passed.

The broad focused command covering graph, P1-B, C1, P1-D, P1-E/F, C2 proof regressions, selector, lattice, and frozen runner completed **288 passed in 112.69 seconds**. After the lead review amendments, the focused P1-G command covering graph, P1-D, full-universe runner spy, and public-boundary tests completed **81 passed in 26.66 seconds**, including the exact match-set, same-axis map, and AST dependency/call assertions. `compileall` passed for the changed sources/tests. Ruff is not installed in the workspace; no lint result is claimed.

## Hashes

Implementation and principal call-site test hashes:

- `plane_graph.py`: `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92`
- `plane_shift_domain.py`: `512c6fb1c9d3491e3deb4ea4db11353e7f8167dcf8aaaabe8fd17ea17303d91b`
- `plane_axis_support.py` (unchanged by P1-G): `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f`
- `frozen_image_runner.py` (passes existing all-readings list): `8221c52b3e89e2c136a0bb779a01b45121677b01bf52abf4b26114e1ce668b61`
- `test_plane_graph.py`: `2ccd95fc0e0c307be4a09a710e0e92fa502c2f9b17aa57bd4ff19eb192fca4f1`
- `test_plane_shift_domain.py`: `a859f146ab756c85908e7c8a223b350031d185385730741b9fcc8d00a6564a69`
- `test_plane_lattice_entry.py`: `8fb4cc79b52cf1cbf15d7063b1a71c9775ec309b3e8fe1dafea129bbda415a63`
- `test_lattice_planes.py`: `76351e29dd63ab1231d4655950ef2f0f4123663165749b5c4d35cfa64da3dcf1`
- `test_frozen_image_runner.py`: `cfe7405aae2596e481c6ac0246eb839195cae3ec75db67c1aae455c23947ecdd`
- `test_plane_grid_map_p1f_integration.py`: `f90550c4fea6c2be370132c6566fd0af3168982bf8d550053dcaf8cf3011b945`
- `test_c2r_vector_domain_proof.py`: `151d32ca811c2a6688c5fbe3d4c13269d158e58ee76463dc51d118aa2687781b`
- `test_c2r_vector_domain_matrix.py`: `ee3116129f932469ac2a122e35c0792e7cb615a5d591a5aecbc968d1b64aa4a0`

Protected historical artifacts still match their accepted hashes: R4 freeze `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, R4 mask sidecar `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`, R5 freeze `8dfc8b41bad1dff018fa08db42411c5f91572233cb1e7718cc17a082ab0c218b`, R5 sidecar `087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b`, and R5 exhaustive diff `9731923b092d28a0d52453a44fadcb8875dd94974c2b4b80339c7cbbbe5653089`.

No real-fixture run, evaluation, schema change, scoring or solver policy change, C2-R builder, R6, V2, CLI, renderer, export, or placement work was performed. No commit was created.
