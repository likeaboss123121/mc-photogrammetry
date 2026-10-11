# M4 P1-D complete bounded shift-domain implementation

Date: 2026-09-13
Status: implementation complete; stopped for lead review
Scope: isolated in-memory evidence retention; no C2-R/V2, evaluation, schema, CLI, renderer, or export wiring

## Result

Added `reconstruction/plane_shift_domain.py` with immutable, self-validating budget,
work, entry, domain, refusal, and result records. `build_shift_domains` accepts only
accepted C1 mapped support, its selected `PlaneMatch` rows, and the exact addressed
`PlaneSignature` objects. C1's distinct mapped u/v identities define the request; other
valid graph hypotheses are filtered. Duplicate or missing selected matches refuse.

For each selected pair the builder snapshots and hashes complete current signature
contents (cells, interior, coverage, dtypes/shapes, origins, canonical b, ordered reading
identity, and region footprint), recomputes C1 geometry and normalized homography
digests, rebuilds the existing warped fixed intersection, and replays the unchanged
correlation surface. It refuses unless the stored shift, peak, second peak, PSR, overlap,
and score match exactly. A raw match b within C1's unique retained alternative tolerance
keeps its exact match identity while the content digest binds canonical b. Positive-infinite PSR is tagged in records; no public record
contains a non-finite number. The full finite surface is bound by a SHA-256 digest.

Before allocating a correlation surface, request-wide preflight checks explicit positive
budget limits for domain count, padded FFT cells, shift positions, clipped overlap-cell
visits, and maximum buffered entries. Enumeration retains every row-major joint
`(row,column)=(v,u)` shift with at least the existing nine-cell support, counting the
fixed whitening intersection shifted against itself. Surface values are diagnostics only.
The result is all-or-nothing: complete domains cover every C1 mapped identity; refusal
and budget exhaustion return no domains; budget exhaustion carries no partial refusals.
`validate_shift_domain_inputs` detects changes to signature content after a record was
made. As approved, P1-D cannot establish historical content that C1 did not hash, and a
declared radius does not certify that physical truth lies within it.

The retained C2-A fixture now shows that top `(0,+1)` remains unchanged while lower
`(+1,+1)` is retained with support above nine and restores the v candidate from `-0.2` to
`+0.8`. The independent scalar reference also covers a non-square pair with holes,
unequal coverages, and nonzero origins. Exact nine-cell shifted support is retained;
eight is excluded.

## Verification

Focused P1-D tests: 19 passed. Combined P1-D, C2-A, C1, P1-B, graph, selector, lattice,
supported, and frozen-runner regressions: **273 passed in 114.96s**. `py_compile` and
owned-file final-newline/trailing-whitespace checks passed.

Accepted P1-B/C1 files retained their recorded hashes: `plane_graph.py`
`3818f85a1947c3270b479f2f1938e8f6edeeec4f6be7b2d614ec87aed0e8e70f`,
`_joint_plane_selector.py`
`f292f08d1b14a5cadbcb395bcc25ce6904a16b353ee6244e5b955de1aed99b5c`,
`plane_axis_support.py`
`ff74c70fd9641e188a791c547e06b694328e67f116630a28a8e3187acb125504`,
`plane_residue.py`
`55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d`,
`test_plane_axis_support.py`
`4b2614eb1b583fa0e8039acf07425b64cd96050c63c512f5179fc8173b5ef988`,
`test_plane_residue.py`
`4d7d00cd1b4d31ec70d964ee153bf13e648f8325e8b284f0e46a130eec43a36f`, and
`test_joint_plane_selector.py`
`c6a55ef7986233c0fa2047e712e3b650ac217e339603b5ee9d7c97ec85a0f0a0`. The C2-A
proof test hash is `6e6a87e8203d9b506b2992cfefe03b0cf2dc7ef19fa8a12d10e1324357b18fb0`;
the frozen-runner test hash is
`760f7b9fd75747e326dd2d7407834e327236e21ed1a2e47c0a47243a9ffd79d4`.

P1-D source SHA-256: `af1c86ec7bfbf79652e198b3dd6b8535b911004a9e9a0beb399c3ac8f1c38640`.
P1-D test SHA-256: `96d28592e960ca099cd193062e2b441dd246ad04bb8f39484a4a5cccd50f949f`.

No existing `PlaneMatch`, graph, selector, frozen schema-v1, or protected artifact file
was edited. No scoring/evaluation, image/GT/manual-metadata path, C2-R, V2, CLI,
renderer, or export path was added. Stopped for lead review.
