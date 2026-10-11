# M4 P0 Endpoint-Inclusive Spacing Audit

Date: 2026-09-13
Role: DeepSeek read-only/no-shell audit, reviewed by lead
Status: implementation prerequisite; no tests or artifacts executed by DeepSeek

## Defect

`rectify_plane` samples `linspace(centre-extent, centre+extent, samples)`, including both endpoints
(`lattice.py:466-468`). Its exact adjacent-sample step is therefore
`2*extent/(samples-1)`, already implemented by `rectified_pixel_to_plane_coordinates`
(`lattice.py:493-505`). Several consumers instead use `2*extent/samples`.

Production inventory:

1. `lattice.period_hypotheses` at `lattice.py:849-857`.
2. `lattice_planes._distance_alternatives` at `lattice_planes.py:795,812`.
3. the legacy branch of `lattice_planes.plane_signature` at `lattice_planes.py:1401-1405`.
4. `plane_graph._cell_homography` at `plane_graph.py:222-225`.
5. `supported.plane_from_evidence` at `supported.py:556-558`.
6. Evaluation-only `scripts/score_separation.py:169-170`, which later uses the exact coordinate map.

Stale prose appears at `lattice.py:446,827` and `lattice_planes.py:121`. `plane_window` is annotated
as a three-float tuple at `lattice.py:370` but returns four values at `lattice.py:415`.
`sampling_for` requests a sample count and is not a spacing consumer.

## Exact effect

For `s=samples`:

- `step_old/step_new = (s-1)/s`.
- `distance_old/distance_new = s/(s-1)`.
- `distance_new = distance_old*(s-1)/s`.

At 64 samples the distance decreases 1.5625%; at 512 it decreases 0.1953125%; at 2048 it
decreases 0.048828%. Thresholds based on block pixels, integer cell counts, origin floors and graph
overlap can still change at boundaries.

## Required contract

Add one public validated helper in `lattice.py` returning `2*extent/(samples-1)`. It rejects boolean
or non-integral samples, samples below two, and non-positive/non-finite extent. The existing coordinate
inverse and every period/spacing consumer call it.

`_distance_alternatives` and `period_hypotheses` produce distances with that helper. Once a reading
exists, `plane_signature` and `_cell_homography` use the unique canonical retained
`DistanceAlternative.distance_blocks`; they do not re-derive distance. Zero or multiple b matches
are typed failures. The selector and other graph face arithmetic already use `distance_for` and need
no change.

This stored-distance rule is also the compatibility boundary: old frozen artifacts retain their
original distance values, so current code must consume those values rather than reinterpret their
period/extent/sample fields under a newer convention.

## Decisive tests

- Analytic helper and endpoint mapping at several sample counts; invalid input rejection.
- Exact `period_hypotheses` and `_distance_alternatives` values against an independent
  `2*extent/(samples-1)` calculation. Existing loose tests must not hide the old formula.
- Sentinel stored distance inconsistent with period metadata is authoritative in both signature and
  graph homography paths; missing/ambiguous alternatives refuse.
- Two representations of one physical plane with different sample counts map a known block point
  identically through the graph homography, checked against an independent analytic oracle.
- `supported.plane_from_evidence` uses the corrected block-pixel value and tests its two-pixel boundary.
- Static source guard ensures no production copy of the old spacing expression remains.
- `plane_window` return shape/type is four values.
- A current immutable freeze loads with its bytes and artifact ID unchanged.

## Persistence

P0 changes numerical production behavior but not schema v1 or its strict key sets. The loader verifies
an artifact against its own content ID and does not compare embedded code hashes with current source,
so existing artifacts remain loadable. New freezes will have changed source hashes/artifact IDs and
corrected numeric values. Old and new readings/signatures must not be mixed in one graph.

P1 periodic-phase evidence and V2 placement remain blocked until this spacing contract is accepted.
