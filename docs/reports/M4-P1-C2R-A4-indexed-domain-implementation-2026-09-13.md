# M4 P1 C2-R-A4 indexed-domain implementation closeout

Date: 2026-09-13  
Status: implemented; stopped for lead review  
Contract: [A4 indexed-domain contract](M4-P1-C2R-A4-indexed-domain-contract-2026-09-13.md), SHA-256 `6f1714eb33aa524798220594a9be51dfb23c358e699d8a33522e27d220b4ea1f`

The isolated module [plane_vector_domain.py](../../src/mc_photogrammetry/reconstruction/plane_vector_domain.py) stores match domains as ordered factor tables and reconstructs records by mixed-radix index. The normal build, serialization, and validation paths construct no candidates. `materialize_lateral_vector_domains` is the sole full-expansion API and preflights domain and candidate limits before allocating anything. Result records distinguish planned from represented work and preserve all-or-nothing outcomes. A mixed valid/refused multi-identity request returns only real refusal records for failed identities and publishes no domains.

The 18 focused tests cover the eager test-local oracle, mixed-radix bijection and large indices, duplicate numeric vectors with distinct provenance, direct A3 truth addressing, five work budgets, atomic materialization limits, permutations, typed refusals, constructor/replace attacks, source compatibility/digest mutations, runtime-count mismatch, serialization limits, zero-candidate normal paths, and the resolved-module import/call boundary.

The exact 68-case A3 census indexes all `2,195,063` candidates without expansion. Its strata are 48 orientation cases / 3,888 candidates, 10 equal-b cases / 1,747,750 candidates, and 10 cyclic unequal-b cases / 443,425 candidates. Every independently generated physical truth is addressed directly by index. Aggregate serialized results are 1,483,156 bytes; maximum per-case traced allocation is 418,302 bytes and maximum measured Windows working-set increase is 1,277,952 bytes. Census elapsed time was 4.335 seconds. The two-domain stress fixture additionally represents 2,351,104 candidates and remains within the test's 8 MiB serialized / 32 MiB traced-allocation bounds.

Validation passed:

```text
pytest -q tests/test_plane_vector_domain.py
18 passed

pytest -q tests/test_plane_vector_domain.py tests/test_plane_residue.py tests/test_plane_axis_support.py tests/test_plane_shift_domain.py tests/test_plane_graph.py tests/test_plane_grid_map_p1f_proof.py tests/test_plane_grid_map_p1f_integration.py tests/test_lattice_planes.py tests/test_joint_plane_selector.py tests/test_frozen_image_runner.py tests/test_c2r_vector_domain_a3_proof.py tests/test_c2r_vector_domain_proof.py tests/test_c2r_vector_domain_matrix.py
423 passed, 133.793 seconds

python -m py_compile src/mc_photogrammetry/reconstruction/plane_vector_domain.py tests/test_plane_vector_domain.py
passed
```

SHA-256: implementation `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5`; tests `a7ede5bbd2fc39667ab735e9650e82f6de6501a9aea66de355272c73d27f9e4a`.

Protected inputs were rechecked unchanged: R4 freeze `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, R4 sidecar `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`, R5 freeze `8dfc8b41bad1dff018fa08db42411c5f91572233cb1e7718cc17a082ab0c218b`, and R5 sidecar `087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b`. The eight preregistered R5 PNG SHA-256 values, in filename order 00–07, remain `48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c`, `8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696`, `2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95`, `e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9`, `8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f`, `6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b`, `53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923`, and `fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556`. No protected artifact or upstream module was changed; no commit was created.
