# PLANE-EDGE-ATTRIBUTION-01

Status: **complete; replay hardening finished; stop for renewed Astra/root review.** This is an
offline diagnostic of the accepted R8 observer artifact, not a production change,
accuracy measurement, solver refit, or promotion result.

## Frozen input and binding contract

The sole data input is
`.tmp-geom-phase-capture-r8-20261004/observation.json`. No image, camera, truth,
reference structure, mask, or other run artifact is opened. The artifact and embedded
records are pinned as follows:

| Record | SHA-256 |
|---|---|
| Observation file bytes | `c3d2ee4ad3e803754d7297ab0c09b7994b17c24de345fda281c0b5cb62888ff8` |
| Container payload digest | `db4b1100d34169570bd7878de05216c6d50a7df76d391087eedd84c7b7a94ddd` |
| Solver-failure observation | `a9d15a99ce58a6b42c183dc4ba63662afe1ec5e12ed8609298e5883ac20fc3f1` |
| Phase capture | `5cbe0943a225abe565ad9f49bf7b0e136f305a992a52d48e51bdb3ed867b6535` |
| Solver-input bindings | `25c8f1460311846cacfec929ec10f703ad8f47af9e8f41eeb5304bbe4f97c99a` |
| Failed assignment | `ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd` |
| Ordered input-match identity | `28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105` |
| Graph result | `0a64c5e73b62b376f41e337a756c61af880d72cee1a90992001a42d00d9f39cd` |
| Attribution trace | `738c9dc09dc17cf64a74e91472cb59a45bbc9917de479cd4f33f80551becd736` |

The run module independently recomputes the graph payload digest and the trace digest
(excluding only attribution_sha256), and compares both against their inner claims and
frozen pins. Existing graph validators validate alias, phase-capture and solver-input
records. The full ordered 38-match identity is independently replayed from every captured
input edge, including inactive edges: referenced readings supply frame/plane/axis, captured
endpoint values supply selected b, and the edge supplies relative quadrant. The canonical
input_match_identity domain and failed-active-edge-aliases-v1 schema bind the exact ordered
scalar projection. No match or missing measurement is reconstructed. It also
cross-checks: observer input-image hashes against the container; entry/lattice/signature/
graph digests against the production records; projected refusal against the attribution
trace; phase and solver-input frame/assignment identity; failed-assignment and ordered
input-match identity across aliases, phase capture and solver-input binding; and every
active edge's two phase rows against the exact endpoint reading/alternative/b/side/sign
rows. A missing, over-cap, malformed or digest-mismatched field aborts before metrics.

## Frozen method

The unit is one active solver match (`match_index`), processed in ascending index order.
For that match, remove both captured endpoint face scalars from the matching world axis,
hold the returned baseline quadrant assignment, least-squares translations and every
remaining face scalar fixed, and call the unchanged
`image_derived_graph._recompose_phase_rows` to recompute the axis circular resultant and
mean absolute wrapped phase residual. The full set is recomposed first and must match the
captured phase record to six decimal places on each axis.

This is fixed-fit phase-consensus influence attribution. It does **not** remove an edge
and re-solve translations, re-rank quadrant assignments, rerun `solve_component`, or
estimate a refitted solver's output. R8 does not retain the complete `PlaneMatch` scoring
fields or exact solve-call quadrant constraints needed for that experiment; its effective
trace settings do retain the frozen gate values. No missing value is reconstructed.

Report for each active match: its index/world axis, remaining face count, recomposed
resultant and residual, whether each crosses the frozen `0.90` and `0.15` gates, and
delta from that axis's full-set recomposition. No aggregation is used to change a gate.
The ten per-match rows are descriptive, dependent counterfactuals on one captured fit,
not independent trials or causal accuracy estimates. If deleting one match makes the
frozen axis clear both gates, label it a single-edge sufficient influence under this
fixed-fit diagnostic; otherwise report the observed values without attribution language.

Frozen numerical policy: Python binary64 values from the sealed JSON; production
recomposition helper; output ordering by integer match index; six-decimal full-set
reconciliation; no tunable parameters. Exact production settings from the bound trace:
`max_shift_cells=32`, `integer_tolerance=0.15`, `min_resultant=0.90`,
`ambiguous_margin=0.02`. LOO does not alter these settings or evaluate a new quadrant
assignment.

## Caps and source pins

Fail closed above 16 MiB artifact bytes, 8 frames, 64 active edges, 128 phase-face rows,
or the existing nested-record caps enforced by the production validators. No output
artifact is written; the CLI prints one JSON record. Inputs and source dependencies are
content-addressed. These four documented immutable source hashes are enforced before
opening the data and before every production replay-helper load; missing or changed
sources refuse, and loaded helper module paths must resolve to the pinned repository:

| Source | SHA-256 |
|---|---|
| `scripts/run_image_derived_graph_stage.py` | `d5244bc4f5fb4648dadf22b14b66ef3694b60781d688864679a274f5058ada97` |
| `src/mc_photogrammetry/pipeline/image_derived_graph.py` | `32747546c934f8c218e43646cc995f310ff383e522c8a10f878e691f54954188` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `f697c2f4ffb512ebb3e7af8f9e497328b4fe828197f62d761b9b0f850ef8972f` |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `ddf95ff9f0dfb66f3bb2c080643b467652194611eaf6237cd28160d73e002d01` |

The analysis hypothesis is that one or a small number of active match edges have enough
phase influence to move at least one axis across a frozen geometry gate. The null outcome
is that every leave-one-edge-out axis still fails or that any gate changes are isolated
to axes that remain unusable overall. Neither outcome establishes why the measurements
disagree or predicts a re-fitted solution.

## Validation order and required tests

Before LOO, validate all file/container/observer/nested digests and cross-bindings above.
The R8 seals and all active match/phase endpoint identities validated before analysis;
validation-only CLI output is `status=validated` for the pinned digests. Numerical
analysis and deterministic replay are complete; the accepted numerical findings below
remain unchanged after the binding hardening.

Focused tests cover source drift/missing source/foreign helper path refusal before
artifact or metric access; resealed outer-container graph/trace payload tampering; fully
resealed structurally valid inactive endpoint-b and inactive ordered-match tampering that
the generic input-record validator alone accepts; exact 38-row identity replay and order
refusal; caps; direct endpoint cross-binding; six-decimal full-set and every accepted LOO
row; deterministic ordering; and original artifact-byte preservation. Resealed adversarial
tests lift enclosing file/container/observation pins in memory only so the new inner checks
are exercised rather than vacuously rejecting at the outer file pin. Replay functions
also enforce source pins when called directly. The analysis immutability test forbids
`solve_component`; the offline analysis never invokes a solver or reads images/truth.
These are replay/binding tests, not fresh reconstruction or observer parity tests.

## Result

All container, observer, phase-capture, solver-input, alias, endpoint, and cross-record
bindings validated before LOO. Validation-only command:

```powershell
.venv\Scripts\python.exe -B scripts\evaluation\attribute_plane_residuals.py --validate-only
```

It returned `status=validated`, ten active edges, and the pinned phase and solver-input
digests. The full phase set recomposed exactly within the frozen 6-decimal tolerance:

| World axis | Faces | Resultant | Residual |
|---:|---:|---:|---:|
| 0 | 8 | 0.789357 | 0.093065 |
| 1 | 6 | 0.013942 | 0.247102 |
| 2 | 6 | 0.876999 | 0.067258 |

No single-edge deletion clears all three axes. Match 4 removal moves axis 0 across both
gates (`R=0.916964`, residual `0.044599`). Match 23 and match 26 removal each move axis 2
across both gates (`R=0.938134` / `0.901812`, residual `0.050726`). Every deletion on
axis 1 leaves it far below the resultant gate (`0.011848–0.016969`) and above the
residual gate (`0.247102`, within floating-point noise). The per-edge values are in the
CLI record; test execution also checks identical ordered LOO output on repeated calls.

Interpretation is deliberately narrow: under the captured baseline translations, the
phase consensus on axes 0 and 2 is sensitive to individual edges; axis 1's incoherence is
not removed by any single-edge deletion. This does not establish which match is wrong,
why phases disagree, how a re-fit would behave, or any geometry/scene accuracy. In
particular, each deletion keeps the original least-squares translations fixed, so this is
not a solver counterfactual. `solve_component` was not called and no fit or image work
occurred.

No-fit test command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='src'; .venv\Scripts\python.exe -B -m pytest tests/test_plane_residual_attribution.py -o addopts= -p no:cacheprovider -q --no-header
```

Result after hardening: **21 passed in 2.14 s**, exit 0 (earlier expanded checkpoint:
21 passed in 2.11 s). The pinned R8 validation-only CLI returned status=validated;
deterministic replay command below returned status=complete, exit 0, all three baseline
axes and the same ten LOO rows. No added counterfactual, refit, reconstruction, image/truth
access, or output artifact. The only new calculations are digest/binding replays.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='src'; .venv\Scripts\python.exe -B scripts\evaluation\attribute_plane_residuals.py
```

The artifact file hash and all four production source pins were unchanged. Byte
trailing-whitespace/newline checks, AST parsing and owned git diff --check are verified
at final handoff. Source pins bind these unchanged helpers, not a new fitted result.

R8 contains only the observer-on result; this package does not claim a fresh empirical
trace-on/trace-off production parity comparison. The offline no-mutation/no-solver test
proves only that this analysis preserves the sealed container and its graph/trace
digests. Production observer neutrality remains the existing code/test contract.

Final SHA-256 values:

| File | SHA-256 |
|---|---|
| `scripts/evaluation/attribute_plane_residuals.py` | `30ea9a6f50b667b445195b92ee9f0c37a4e325f3d7e49ee556ca97ef53a489f6` |
| `tests/test_plane_residual_attribution.py` | `8a0d58aca0faccde38731a93a88cfaa19d681fed364ddde1c36a6ed42c1cbdc5` |

Stop here for independent Astra/root review. No production edit or follow-on intervention
is authorized by this diagnostic.
