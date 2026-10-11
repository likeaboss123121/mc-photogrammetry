# P1-D v2 shifted-support contract

Status: approved contract amendment for bounded implementation; production promotion remains gated.

## Kernel

`correlation_algorithm_id` is `p1d-separate-mask-phase-correlation-v2`. For right-grid shape
`(h,w)` and radius `R`, use a halo canvas with origin `(-R,-R)` and shape
`(h+2R,w+2R)`, and FFT shape `(h+4R,w+4R)`. Preserve separate left and right coverage masks.
Whiten each map over its own covered samples, then zero uncovered samples. Do not intersect
masks before whitening. Enumerate shifts row-major in `[-R,R]^2` and define

`N(dr,dc) = sum MR(r,c) * ML(r+dr,c+dc)`.

Zero-shift overlap is diagnostic only. Preserve peak, 3x3 sidelobe, PSR, structural-share,
and score rules. Require the selected peak to have `N >= MIN_OVERLAP_CELLS`; refuse
`insufficient_overlap` without substituting a secondary peak. Match overlap and score use the
selected `N`, with denominator `min(left_coverage_cells,right_coverage_cells)`. Existing
radius 32 and `.90/.15/.02` gates remain unchanged.

## Evidence and versioning

Matcher and P1-D use one kernel. v2 match evidence binds algorithm ID, radius, grid shape,
canvas origin/shape, FFT shape, left/right coverage counts, zero-shift support, whitening
means, mask/map/surface/support digests, signature/homography digests, and canonical little-
endian array encodings. `ShiftDomainResult` and `MatchShiftDomain` bind schema version 2,
algorithm ID, kernel evidence, and exact support counts. Every shift with `N >= 9` is retained,
including non-top shifts; entries are unique and row-major.

The v2 producer sequence is graph-2, axis-support-2, shift-policy-2, graph-policy-2,
shift-domains-2, and phase-evidence-2. V2 boundaries reject v1 or unknown evidence with typed
`provenance_mismatch`; they never reinterpret old records or silently fall back. Historical v1
artifacts remain immutable and are replayed only by an explicit legacy implementation.

## Infinite PSR

Canonical serialization is `{value: finite, positive_infinity: false}` or
`{value: null, positive_infinity: true}`. NaN, negative infinity, and inconsistent tags are
invalid. The tag must survive graph, phase/C1, P1-D, and vector-domain replay without clamping.

## Work accounting and acceptance

Per valid pair, planned work is `P=(2R+1)^2`, FFT cells `(h+4R)(w+4R)`, overlap visits
`P*h*w`, and planned buffered entries `P`. Existing caps remain 10 domains, 2,000,000 FFT
cells, 100,000 shifts, 100,000,000 overlap visits, and 100,000 buffered entries. Preflight
must occur before allocation; planned exhaustion is atomic with zero consumed counters.

Acceptance requires both DS-R3-N counterexamples to recover exact shifts/supports (0,-24)/1200
and (0,-20)/2400; support-oracle coverage for holes, borders, unequal grids/origins, all
quadrants, zero overlap, support 8/9/10, and ±32; deterministic reverse/permutation and
input-immutability checks; repeated/constant texture refusals; exact evidence replay and
tamper/version rejection; infinite-PSR end-to-end replay; budget boundary tests; adjacent
regressions; and fresh Template 2/6 runs. No threshold relaxation or synthetic fallback is
allowed.
