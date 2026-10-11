# M4 P1 C2-R-B cycle-closure implementation

Date: 2026-09-13
Status: accepted after remediation-2, lead rerun, and DeepSeek frozen re-audit
Contract: [C2-R-B cycle-closure contract](M4-P1-C2R-B-cycle-closure-contract-2026-09-13.md), current verified SHA-256
`8929b19c3be1f729c20bc2be5849df3f7ca54dfc25fd5fa528ef96bc19492ce5`.
The earlier `df6a…`, `baa22…`, `71fc…`, `25…`, and `448…` entries in
`AGENT_COMMS.md` are superseded history, not competing current hashes.

## Implementation

The isolated solver in [plane_relative_translation.py](../../src/mc_photogrammetry/reconstruction/plane_relative_translation.py)
performs graph-wide directed relative-translation closure over public A4 indexed
domains. It canonicalizes components and spanning trees, propagates exact dyadic
translations, uses one shared assignment for all cycles, and checks chords by
factorized membership. It preserves ambiguity, contradiction, tree-only status,
disconnected gauges, provenance multiplicity, explicit budgets, and atomic typed
refusal or exhaustion.

Remediation-2 closes the frozen-review findings:

- **N=1 endpoint proof.** The `represented_candidate_count == 1` shortcut was
  removed from `_factorized_single_delta_class`. The public single-vector A4
  cycle now executes the factorized endpoint proof itself: 4 membership probes
  per domain (two factors times two rounded corners) and 12 probes for the
  three-edge cycle, still with zero candidate views and zero `candidate_at`
  calls. The module documents why a represented count greater than one cannot
  produce a numeric singleton under A4's strictly increasing residue tables and
  unique row-major P1-D entries, so no impossible multi-represented singleton
  fixture is fabricated; the theorem test pins those invariants instead.
- **Record/result validation.** Ambiguous translation certificates must have
  distinct translation keys. Unique and ambiguous cycle residuals must have
  positive matching provenance. Refused results must publish exactly zero
  deterministic and runtime work. The enclosing complete result now checks its
  feasible cycle residual bounds against its own `closure_tolerance`; component
  records still carry no hidden tolerance and are unchanged in shape.
- **Falsifiable zero materialization.** A monkeypatch test raises on
  `candidate_at`, `iter_candidates`, `materialize_lateral_vector_domains`, and
  `plane_graph.solve_component` while the public ambiguous 289-candidate and the
  public unique single-vector solves both complete. The module AST test denies
  those names plus selector/placement/export surfaces and allows only
  `plane_graph.PlaneMatch`/`axis_transform`.
- **Public lateral-sign mutant.** A rebuilt public A4 cycle with all normal
  deltas zero and a sign-flipped lateral half-period is a contradiction. The
  nearest A4 candidate mismatches the required lateral component by more than
  1/4 while the normal component matches exactly, so the failure cannot be a
  normal or nearest-candidate artifact.
- **Honest performance evidence.** The 1,000-edge timing/RSS check is explicitly
  kernel-only private synthetic coverage, not a public A4 path. A public
  five-edge all-singleton A4 ring now measures the production singleton path
  (`unique_validated/ready`, 20 probes, zero views) under the 5 s / 32 MiB gate.
- **Budget boundaries.** The public singleton path has an exact-observed budget
  that succeeds and a one-below `max_membership_probes` run that exhausts
  atomically with zero components, zero views, and zero `candidate_at` calls.

No A4 source, test, or report changed. No runner, schema, graph, C1/P1-D,
selector, evaluator, CLI, renderer, export, or default-path wiring was added. No
images or labels were evaluated, and no commit, stage, reset, or clean was
performed.

## Validation

Measurements below were rerun for remediation-2. DeepSeek ran only the focused file and `py_compile`; the lead then independently ran the complete accepted A4 dependency set plus C2-R-B.

- Focused solver suite — **36 passed in 2.85–2.93 s** across reruns:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_plane_relative_translation.py
  ```

- `py_compile` passed for the solver and its test with the repository
  interpreter.

- Independent combined dependency regression ? **459 passed in 138.35 s**:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_plane_relative_translation.py tests/test_plane_vector_domain.py tests/test_plane_residue.py tests/test_plane_axis_support.py tests/test_plane_shift_domain.py tests/test_plane_graph.py tests/test_plane_grid_map_p1f_proof.py tests/test_plane_grid_map_p1f_integration.py tests/test_lattice_planes.py tests/test_joint_plane_selector.py tests/test_frozen_image_runner.py tests/test_c2r_vector_domain_a3_proof.py tests/test_c2r_vector_domain_proof.py tests/test_c2r_vector_domain_matrix.py
  ```

Observed call durations from that run include 0.19 s for the >2-million-represented-vector
factorized preflight/eligibility case, 0.18 s for the public five-edge singleton
ring, and 0.17 s for the kernel-only 1,000-edge synthetic check. The public
two-hundred-eighty-nine-candidate ambiguous fixture still records 289 compact
views, 2 partial assignments, 584 membership probes, 289 memoized terms, 2
provenance witnesses, zero `candidate_at` calls, and a 5,194-byte result. No
1,000 replay-valid public A4 edges were constructed.

## Frozen hashes

- Contract (current verified): `8929b19c3be1f729c20bc2be5849df3f7ca54dfc25fd5fa528ef96bc19492ce5`
- Superseded contract history: `df6a434dc9621d6068ef69beba38518b527eb54bb9ac6cb72d5c9622bf585904`
- Solver source: `F703A28D275D55E4A85FAD7FDE0BFD612928F8D54A5489C0EA097FB42EFBCD3E`
- Solver tests: `A52BEEB0F11A1B42FF64B3FC3AC9F6B2004498187B2A58E757D8057591E97CCD`
- Accepted A4 source (unchanged): `72EE29FB936CFFD7F8A154C18D6F6BAB18876D57AD66F45BC316E39F7D45E7F5`
- Accepted A4 tests (unchanged): `A7EDE5BBD2FC39667AB735E9650E82F6DE6501A9AEA66DE355272C73D27F9E4A`
- Accepted A4 report (unchanged): `D48AA6F3541CD79B58E1B23471C53724FD5A38ACB2DE1D8172A7EA1CF12AE15F`
