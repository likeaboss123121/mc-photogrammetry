# M4 P1-C1 public axis transform and translation-support topology

Status: approved implementation contract after P1-B acceptance. This is the first bounded part of P1-C; it does not place blocks or choose a phase residue.

## Purpose

P1-B preserves per-reading phase-residue candidate sets and the match-relative integer cell shift. P1-C must put that evidence into the accepted component's common axis frame before V2 can use it. C1 establishes the shared axis transform, maps the integer shift as evidence, and records which frame/axis translations are actually constrained. C2 will combine phase candidates through the homography-aware frame relation and retain every consistent fractional assignment.

## Inputs and acceptance boundary

C1 consumes only:

- an accepted `plane_graph.ComponentSolution` with `reason == ""`;
- the same ordered image-derived `PlaneReading`, `PlaneSignature`, and admitted `PlaneMatch` values used to solve and form it;
- P1-B `HypothesisPhaseShiftEvidence` regenerated in memory from those values.

A match contributes only when both reading identities exist, its relative quadrant agrees with the accepted absolute quadrant labels, and its left/right `b` values uniquely match both `solution.blocks_per_period` entries through the existing canonical tolerance. Competing admitted q/b hypotheses that the accepted solution did not select are filtered, not treated as corrupt input. At least one selected row must remain. Each selected endpoint must also resolve to exactly one of the actual supplied signatures by ordered reading identity and selected canonical `b`; readings and matches alone cannot reconstruct signature origins or the cell homography. Duplicates are deterministic and cannot increase connectivity or support counts.

An abstained solution, missing selected identity, inconsistent selected provenance, malformed evidence, or nonfinite value yields a typed refusal. Translations stored on an abstained solution are never evidence.

## Public axis transform

Expose one public pure transform in `plane_graph`:

```text
axis_transform(local_axis, quadrant) -> (world_axis, world_sign)
```

- `local_axis` is exactly `0`, `1`, or `2`.
- `quadrant` is exactly `0`, `1`, `2`, or `3`; booleans and implicit modulo normalization are rejected.
- `world_axis` is an integer in `0..2`.
- `world_sign` is an integer `-1` or `+1`.
- It represents `Ry(-90 degrees * quadrant) * e_local = world_sign * e_world`.

The frozen table is:

| q | local 0 | local 1 | local 2 |
|---:|---|---|---|
| 0 | `(0,+1)` | `(1,+1)` | `(2,+1)` |
| 1 | `(2,+1)` | `(1,+1)` | `(0,-1)` |
| 2 | `(0,-1)` | `(1,+1)` | `(2,-1)` |
| 3 | `(2,-1)` | `(1,+1)` | `(0,+1)` |

Verify the table independently from a literal expected table, not by reproducing the implementation formula in the test. Migrate `plane_graph` and `_joint_plane_selector` callers to the public function. Remove production use of `_axis_index` and `_axis_sign`; compatibility aliases are allowed only if tests prove no independent logic remains. Valid solver candidates, ordering, certificates, resultants, residuals, translations, and hashes must remain byte-identical.

## Local plane axes

For a reading with normal label `n`, `lattice.plane_basis` fixes:

```text
local u label = sorted({0,1,2} - {n})[0]
local v label = sorted({0,1,2} - {n})[1]
array column = u
array row = v
```

Map each label through `axis_transform(label, frame_quadrant)`. The two mapped lateral world axes must be distinct and must exclude the mapped normal axis. Any disagreement is a typed transform refusal.

## Evidence records

C1 adds deterministic in-memory records with strict construction checks and JSON-safe `to_record()` output:

### `MappedCellShiftAxis`

- full match identity: frames, plane indices, reading axes, original b pair, relative quadrant;
- right local axis: `u` or `v`, plus its numeric local label;
- exact source component: column for `u`, row for `v`;
- copied signed integer cell shift in the right grid;
- mapped world axis and right-frame world sign;
- `signed_world_integer = world_sign * right_grid_shift`;
- copied P1-B support identity and refusal; no new score or threshold;
- deterministic left/right signature-geometry identities and normalized finite right-to-left 3x3 cell-homography digest.

The value is evidence for the integer component of `t_right[A] - t_left[A]`. It is not a complete translation and must not be added directly to `ComponentSolution.translations` before C2 accounts for within-cell phase.

The sign convention is pinned by:

```text
warped_left[x + shift] ~= right[x]
cell_shift_rc = (row, column) = (v, u)
```

Because `_cell_homography` resamples the left signature on the right grid, interpret both components in the right reading's local plane basis. Do not interpret the raw left and right windows as sharing axes. C1 receives the exact signatures used for matching and binds them uniquely; it never regenerates signatures from images or readings. Its geometry identity covers only endpoint identity, canonical `b`, `origin_uv`, grid shape, and rectification/distance inputs used by the homography. A digest is provenance, not evidence. Normalize the finite homography deterministically before hashing. C2 can then validate the fractional relation rather than reconstructing a guessed transform.

### `NormalFaceConstraint`

For each selected match, retain its full identity, mapped world normal axis, left/right world signs, copied camera sides, selected distances, and finite `delta = sign_left * side_left * distance_left - sign_right * side_right * distance_right`. This is the numeric evidence for `t_right[A] - t_left[A]`; later consumers must not reconstruct it from unrelated or reordered readings. Construction checks bind every field and recompute `delta` exactly.

### `AxisSupportComponent`

For each world axis and each connected component of the accepted constraint graph, record:

- sorted frames;
- deterministic gauge anchor: lexicographically first frame in that axis component;
- sorted unique contributing match identities;
- separate counts for normal-face and lateral-shift evidence;
- whether the component contains the global component anchor `solution.frames[0]`;
- typed status: `supported`, `disconnected_from_global_gauge`, or `no_constraints`.

Normal-face evidence is the typed equation:

```text
t_right[A] - t_left[A]
  = sign_left * side_left * distance_left
  - sign_right * side_right * distance_right
```

Lateral evidence comes only from non-refused P1-B cell shifts, one row and one column record per accepted match. A match may therefore support its normal world axis and both distinct lateral world axes. Typed normal constraints and mapped lateral records remain separate in serialization and diagnostics.

### `FrameAxisSupport`

For every `(frame, world_axis)` pair, record its axis-component anchor, connected-to-global-anchor flag, contributing edge count, independent frame-pair count, evidence classes, and refusal. A numeric entry in `solution.translations` does not imply support. V2 may use a frame axis only when it is connected to the global anchor on that axis and C2 resolves its fractional candidates.

## Gauge

All constraints are relative. For each connected axis component, adding the same integer to every frame translation on that axis changes no observation. C1 declares this integer gauge and fixes the reporting representative by setting that axis component's deterministic anchor integer to zero. It must not claim Minecraft world coordinates or merge disconnected axis components. The component-wide anchor establishes one common output frame only for axes whose support graph reaches it.

## C1 refusals

At minimum: `solution_abstained`, `frame_missing`, `reading_missing`, `signature_missing`, `signature_ambiguous`, `signature_identity_mismatch`, `homography_invalid`, `quadrant_invalid`, `selected_provenance_missing`, `phase_shift_identity_mismatch`, `cell_shift_absent`, `nonfinite_support`, `axis_transform_invalid`, `disconnected_from_global_gauge`, and `no_constraints`. Unselected competing hypotheses are skipped. Missing selected data never becomes zero.

## Decisive controls

1. Literal axis/sign table for all 12 `(axis,q)` inputs, plus booleans/out-of-range/nonfinite rejection.
2. Existing graph and selector suites prove byte-identical behavior after migration.
3. All three plane normals, four quadrants, both sides, and asymmetric nonzero shifts prove row=v and column=u without transpose.
4. `right=np.roll(left,(3,-4))`, whose admitted shift is `(-3,+4)`, maps each component through the right frame's u/v labels and sign table.
5. Quadrants `q` and `q+2` preserve axis index and reverse horizontal signs; vertical sign remains positive.
6. A synthetic three-frame cycle produces deterministic unique edges and one connected component per supported axis; normal records reproduce the solver equation exactly.
7. A graph with an axis edge only between two non-anchor frames reports both as disconnected from the global gauge and refuses the anchor on that axis.
8. An accepted solution whose least-squares map contains a zero for an unconstrained frame axis still reports that frame axis unsupported.
9. A refused P1-B shift contributes normal-face evidence only and never lateral support.
10. Reordered readings/matches and duplicate identical matches produce byte-identical records and counts.
11. Competing unselected q/b rows are skipped; missing selected provenance, mismatched nested identity, nonfinite support, missing/duplicate/foreign signature, invalid homography, or abstained solution fails closed.
12. Import/isolation checks forbid dataset, evaluator, manifest, reference, F3, supplied camera, supplied bounds, CLI, renderer, export, and placement dependencies.

## C1 exit criteria

- Public transform and every existing caller use one implementation.
- Typed mapped-shift and support-topology records enforce their own invariants.
- Tests 1-12 and the focused graph/selector/P1-B/frozen-runner suites pass.
- Valid pre-C1 graph and selector outputs remain unchanged.
- No frozen schema/artifact, evaluator, threshold, ordering, production runner, CLI, renderer, export, or placement change.

## C2 boundary

C2 must use the actual cell homography/signature origins plus the C1 transform to compare the left and right P1-B residue sets on the same world axis. It will retain all consistent full relative-translation candidates, solve cycles without choosing an absolute seam, and report ambiguity or contradiction. It must not combine raw local residues with the integer shift by an assumed sign formula; asymmetric synthetic homographies must establish the formula first. V2 remains blocked until C2 is accepted.

Manual cameras, camera manifests, F3 coordinates, directions, scene bounds, references, schematics, and ground truth remain advanced/developer/evaluation-only and are forbidden as default reconstruction inputs.

## Implementation closeout — 2026-09-13

Published `plane_graph.axis_transform` from the frozen literal table and migrated graph/selector axis consumers. Added in-memory C1 records in `plane_axis_support.py` for selected normal-face equations, exact right-grid integer-shift mappings, signature/homography provenance, and per-world-axis connected support. The builder filters losing q/b hypotheses, binds exact supplied signatures by reading identity and canonical b, and returns typed refusals for missing or malformed selected evidence. Cross-record validation requires each selected normal identity to have two matching u/v mappings or one retained P1-B refusal.

Focused validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_plane_axis_support.py tests/test_plane_graph.py tests/test_joint_plane_selector.py tests/test_plane_residue.py tests/test_lattice_planes.py tests/test_supported.py tests/test_frozen_image_runner.py -q -o addopts=''
```

Result: 253 passed in 113.10s. `py_compile` passed for changed source/tests; `git diff --check` passed. A source scan found no remaining `_axis_index` or `_axis_sign` use in `plane_graph.py` or `_joint_plane_selector.py`. Tests cover the literal transform, four quadrants, both camera sides, asymmetric `(-3,+4)` shift, selected-hypothesis filtering, cycle/disconnected gauges, signature failures, retained nonfinite P1-B evidence, permutation determinism, replacement attacks, and frozen-runner compatibility. No capture, labels, evaluator, or frozen artifacts were changed.

SHA-256:

- `plane_graph.py`: `3818f85a1947c3270b479f2f1938e8f6edeeec4f6be7b2d614ec87aed0e8e70f`
- `_joint_plane_selector.py`: `f292f08d1b14a5cadbcb395bcc25ce6904a16b353ee6244e5b955de1aed99b5c`
- `plane_axis_support.py`: `ff74c70fd9641e188a791c547e06b694328e67f116630a28a8e3187acb125504`
- `__init__.py`: `5f98378c92de84379b4cdffaa33b7ffd38313963e556a6b687d543a7e2b9fa9e1`
- `test_plane_axis_support.py`: `4b2614eb1b583fa0e8039acf07425b64cd96050c63c512f5179fc8173b5ef988`
- `test_joint_plane_selector.py`: `c6a55ef7986233c0fa2047e712e3b650ac217e339603b5ee9d7c97ec85a0f0a0`
