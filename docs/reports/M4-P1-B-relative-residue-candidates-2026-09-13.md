# M4 P1-B relative residue candidates

Date: 2026-09-13
Status: implementation complete; stopped for lead review
Scope: in-memory evidence only; no cross-view fusion, selection, placement, or wiring

## Result

Added `reconstruction/plane_residue.py` with a pure function that takes one
admitted `PlaneMatch` and the two addressed `PlaneReading` values. For each
reading and local axis (`u` is column, `v` is row), it preserves the complete
periodic-energy residue candidate set for that reading's uniquely matched,
canonical retained `b` alternative. It separately copies the match's exact
`(row, column)` cell shift and existing support diagnostics. Candidate phases
are not fused across views, mapped to world axes, or reduced to a selected
residue. No seam identity, placement, or absolute translation is produced.

For reduced `b=p/q`, phase fraction `f=phase_pixels/period_pixels`, and the
continuous leading edge of the untrimmed measurement window
`L=distance_blocks*(rectification.centre[axis]-reading.extent)`, the output is
the sorted unique set

```text
((L + f*b + k/q) mod 1) for k in 0..q-1
```

`L` is computed directly from the reading geometry. A forced-trim test uses the
real `plane_signature` grid/origin logic to show that its capped, shifted
`origin_uv` does not enter the residue calculation. The rectifier alone is
stubbed in this control; the signature origin, trimming arithmetic and residue
formula are exercised directly.

The supported distance-alternative table is the only rational lookup. Match
`b` values within the existing retained-alternative tolerance are canonicalized
to that alternative for phase calculations, while the original match `b` pair
remains in the top-level identity. Missing, ambiguous, or unsupported values
refuse with typed reasons. Missing P1-A phase is never interpreted as zero; a
measured zero phase with nonzero response remains valid. A missing shift is
reported separately as `cell_shift_absent`. The optional peak margin was
omitted, so this change introduces no derived support score or threshold.

## Evidence

All ten values in `DISTANCE_ALTERNATIVES` were tested against period-index
enumeration spanning positive and negative indices over multiple `q` cycles.
The controls also cover exact modulo endpoints and near-one residues, phase
perturbation, a one-measured-period window translation using the endpoint
pixel-step and `distance * world_period = b`, forced signature trimming, source
and reading identity, near-equal canonical `b`, all six P1-A unavailability
reasons, missing/duplicate alternatives, finite-value abstention, signed
asymmetric shifts, deterministic records, and forbidden metadata/output
fields.

Worked synthetic `u` examples use phase zero and `L=9.5`:

- `b=1`: `q=1`, candidate `{0.5}`.
- `b=1/2`: `q=2`, candidates `{0, 0.5}`.
- `b=1/16`: `q=16`, all multiples of `1/16` in `[0,1)`.

A representative serialized match record (left `b=1/2`, right `b=1/16`, shift
`(-3, 4)`) has SHA-256
`d01b3d2576a48748d71bc5a055526055fdc32e7ca4eb9ea0308e9b4e600a7d58`.

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plane_residue.py tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_joint_plane_selector.py tests\test_frozen_image_runner.py tests\test_supported.py
```

Result: 210 collected, all passed (exit 0). The focused residue suite passed
43/43; the targeted endpoint/window-shift subset passed 2/2. `py_compile` passed for the new source and test file;
an explicit trailing-whitespace/final-newline scan passed for all three owned
files.

## Frozen-record invariant closure

Direct construction and `dataclasses.replace` now reject inconsistent frozen
records: quality metrics must be finite and in range with known refusal reasons;
residue candidates must bind the supported canonical rational, exact amplification,
candidate formula/cardinality, and refusal/quality state; reading axes are part of
candidate identity; and hypothesis phase/shift evidence must match the full frame,
plane, axis, b-pair, and quadrant identity. Refusal semantics are checked at the
record boundary, including nonfinite-field reporting and absent shifts. Valid
factory output, candidate equation, selector/graph behavior, and serializer schema
remain unchanged. The verified sample record hash appears once in the Evidence
section above.

The final defensive closure returns typed `candidate_set_invalid` when a phase
value is missing without an unavailable reason (a combination forbidden by P1-A),
instead of fabricating an unavailable reason. Negative overlap counts normalize to
`None` and produce `nonfinite_derived_value`; adversarial tests bypass upstream
validation to exercise both boundaries. The final six-file suite collected 210 tests
and passed all; the residue suite passed 43/43.

## Files and hashes

- `src/mc_photogrammetry/reconstruction/plane_residue.py` — SHA-256
  `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d`
- `tests/test_plane_residue.py` — SHA-256
  `4d7d00cd1b4d31ec70d964ee153bf13e648f8325e8b284f0e46a130eec43a36f`

No existing solver, serializer/schema, frozen artifact, evaluator, CLI, or
production path was edited or run. P1-C owns any cross-view comparison or
translation support; P1-B stops with the candidate ambiguity intact.
