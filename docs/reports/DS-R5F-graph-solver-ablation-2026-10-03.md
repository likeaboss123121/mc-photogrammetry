# DS-R5F exact graph-solver replay and counterfactual — 2026-10-03

## Scope

Evaluation-only, one foreground contained replay that recomputed entry, lattice,
signatures, matcher and the exact 54 `PlaneMatch` solver inputs from the four
frozen v6 PNGs, failed closed unless every recomputed digest, the complete
ordered pair-trace record, the selected match identities, the graph digest, the
baseline reason and the per-axis diagnostics reproduced the frozen artifact, and
then ran the four preregistered solver-only counterfactuals with the original
image-derived connection constraints. No production module, threshold, gate,
camera, image, evaluator, project, or artifact was edited. The one output is
`.tmp-ds-r5f-20261003.json`.

Utility: `scripts/evaluation/replay_graph_solver_ablation.py` (SHA-256
`eb1eb7f20b21cb5a51e354750b9a099c769dc65d7f6d7db63cf59a387c8c8252`).
Focused tests: `tests/test_graph_solver_ablation.py` (SHA-256
`4db83073745b1fa4ffd823d049c375fd4a26dbe76ee864686be96e00dd9283e6`).

Command (one invocation, no retry):

```
.\.venv\Scripts\python.exe scripts\evaluation\replay_graph_solver_ablation.py
```

## Replay validation (fail-closed)

The parent created the already-reviewed 8-GiB Job through
`scripts/run_image_derived_graph_stage.py::_native_execute` (which loads the
reviewed `.tmp-run-axis0-component-preflight-replacement.py` Job helper) and
launched the utility itself as a contained worker under a 600-second deadline.
The worker refused before image processing if it was not inside the Job. Exit 0;
wall time 146.128 s; no `.partial` remained; no other path was created.

Every check below matched the v6 artifact, so the ablation stage was reached:

| Evidence | Value |
|---|---|
| Artifact self-digest (both replay directions) | `c5c6251504f74c0f9c16a9666834d74b185cc13e180d6a30dec7fea6b312f6b3` |
| Session / lattice / signatures | `db924383…0c4e` / `98784b0c…0ded` / `e5875d66…8412` |
| Policy | `501ada9e…b08a` |
| Graph result (inner, outer, trace join) | `83eb7fb3b995b9d89bdccedef7480f40daf0793564191386d9f4024c81226ae4` |
| Pair-trace record (all 7,476 rows, full record equality) | `eb2fabfa1e2acc907e1397e61430cbbb9cb2a2e785ce516b8c66b8d60b2ac176` |
| Candidate funnel pre / post-admission / post-same-axis | 7,476 / 1,495 / 54 |
| Selected match identity (alias and boundary domains) | `e6f9e955…842a` and `7fe96f18…d9a9`, count 54 |
| Boundary kept counts / radius | `[1, 1, 0]` / 32 |
| Baseline reason | `plane_integer_residual` |
| Baseline per-axis (resultant, residual, observations) | `(0, .39146, .17397, 12)`, `(1, .24717, .20172, 6)`, `(2, .24521, .18721, 10)` |
| Conflicting endpoints | 8 (`image-0000` 0/1, `image-0001` 1/3/4, `image-0002` 0/2/3) |
| Project / frozen PNGs | project `a85fe65b…f87`; the four recorded PNG SHA-256 values unchanged |

The recomputation used the accepted chain with `_persist_session` temporarily
replaced by a no-op, so the frozen project was read and never rewritten. A
non-invasive observer on `plane_graph.solve_component` copied the exact
`readings`, the exact post-filter `PlaneMatch` tuple and the exact solver
keyword arguments (including `quadrant_constraints`) that the production graph
seam passed to the solver, then delegated to the real solver unchanged. The
replay therefore establishes that the exact readings/matches are reconstructible
from the frozen PNGs and current producers: no new evaluator seam is required
for this solver-only ablation.

## Counterfactual results

Removal sets, baseline and three counterfactuals. All four reuse the captured
readings, the captured image-derived quadrant constraints and the production
gates (`max_shift_cells=32`, `integer_tolerance=.15`, `min_resultant=.90`,
`ambiguous_margin=.02`, `min_overlap_cells=9`, `min_structural_share=.15`).
`removed rows` are counted with the same identity projection that
`active_edge_aliases` binds.

| Ablation | Removed | Retained | Reason | Per-axis (resultant, residual, observations) |
|---|---:|---:|---|---|
| baseline | 0 | 54 | `plane_integer_residual` | `(0, .39146, .17397, 12)`, `(1, .24717, .20172, 6)`, `(2, .24521, .18721, 10)` |
| boundary `±32` | 2 | 52 | `plane_integer_residual` | identical to baseline on all three axes |
| alias-conflicting | 48 | 6 | `insufficient_axis_coverage` | `(0, —, —, 0)`, `(1, —, —, 0)`, `(2, 0.9999999999999999, 6.7145e-16, 4)` |
| union | 48 | 6 | `insufficient_axis_coverage` | identical to alias-conflicting |

Exact boundary rows removed (indices 9 and 15):

- `[image-0000 p3 axis2 b0.5 -> image-0001 p4 axis2 b4.0 q0]`
- `[image-0000 p1 axis1 b0.5 -> image-0002 p2 axis1 b1.0 q0]`

The union equals the alias-conflicting set: those two boundary matches are
inside the conflict-incident population, so boundary saturation adds no
additional removed row.

## Interpretation

- **Boundary saturation is not the cause of the abstention in this fixture.**
  Removing the only two `±32`-boundary matches leaves the solver's returned best
  assignment and all three per-axis resultants, residuals and observation counts
  bit-for-bit identical. The DS-R5E boundary-saturation hypothesis is not
  supported by this measurement; widening the radius is not motivated.
- **Axis-0 and axis-1 evidence is exactly the alias-conflicted evidence.** After
  removing the 48 matches incident to the 8 conflicting endpoints, only 6
  matches remain and all of them are axis 2; the solver then abstains because
  axes 0 and 1 have no coverage. The surviving axis-2 matches are internally
  perfect (resultant ≈ 1.0, residual ≈ 6.7e-16), so their identity geometry is
  consistent, but they cannot supply the missing axes.
- **This is not proof that the conflicting matches are wrong.** Removing 48/54
  matches trivially destroys the two low-coherence axes, so the counterfactual
  confounds wrong-identity evidence with evidence starvation. It establishes a
  structural fact (the failing axes are precisely the alias-conflicted axes),
  not a causal attribution.

## Limitations

- Pair hypotheses, not independent image correspondences or truth recall; no
  ground truth, evaluator, label, or visible-state score was used.
- One fixture, one run. The result is diagnostic and does not establish accuracy.
- The alias counterfactual removes 89% of the solver evidence, so it cannot
  separate wrong identities from missing coverage. A next test would remove
  conflict classes one endpoint group at a time, or pair each removal with an
  equal-size non-conflicting control, before any matcher or threshold change.
- No production output, gate, score, camera, or promotion changed; `acceptance`
  semantics are unchanged and geometry remains unavailable for this fixture.

## Provenance

- Output `.tmp-ds-r5f-20261003.json`: 37,721 bytes, file SHA-256
  `de68ddb7765d49d13e40f7e6cb17433cc95b64443ba00a19e581d2c8daf0c7c9`;
  its internal canonical `artifact_sha256` seal replays as
  `a896b9a3f5caf9c5cdc5d10a284ba91925818af8157643afa6ff5ce09c283fb6`
  (status `complete`). The v6 artifact follows the same convention: file
  `7688538f…56c1`, internal seal `c5c62515…f6b3`.
- Frozen v6 artifact `c5c6251504f74c0f9c16a9666834d74b185cc13e180d6a30dec7fea6b312f6b3`;
  project `a85fe65babfd4d07e257b3451dacdb58e6b5f527c078313fd86d58cbeb727f87`.
- Ordered PNG SHA-256: `0087a714…ba07`, `0c309299…95f6`, `874fe746…8049`,
  `2c75a0df…c853`.
- Producer hashes recorded in the output: contained runner `901d73d4…ef8f7`;
  Job helper `e33456d8…e67b6`; entry `5ced5b2b…f0962`; lattice measure
  `c782edbf…b16e673`; signatures `86172efb…a69e245`; graph producer
  `7d4579f8…cb5acb`; graph attribution `65dffda3…e2438b`; `plane_graph.py`
  `9b19b63d…d1d90`; `lattice_planes.py` `ddf95ff9…02d01`.
- Containment: `limit_bytes=8589934592`, `deadline_seconds=600`, reviewed
  contained runner and Job helper; worker verified `IsProcessInJob` before
  image processing.
- Focused tests: `12 passed in 0.35s`
  (`pytest -o addopts='' -q tests/test_graph_solver_ablation.py`), covering
  boundary/conflict selection, union semantics, solver-kwargs reuse, the
  captured-baseline path, digest-bound output, overwrite refusal, and
  fail-closed validator mismatches.

## Status

Evaluation-only evidence complete and stopped for lead review. No matcher,
solver, threshold, production wiring, project, or artifact change is authorized
by this result. Geometry, promotion and visible acceptance remain unmet.
