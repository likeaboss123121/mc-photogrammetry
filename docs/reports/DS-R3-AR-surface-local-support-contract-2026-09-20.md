# DS-R3-AR: Surface-local support attribution contract

**Date:** 2026-09-20
**Status:** accepted evaluation-only contract; implementation planned

## Purpose
Attribute selected matcher support cells to rendered physical surfaces without binding an entire broad `PlaneReading` to a surface. Existing PSR, structural share, score, and overlap remain `whole_reading_metrics`; they are not surface-local quality evidence.

## Inputs and provenance
Consume two frozen `RenderedTruthEvidence` records, original RGB/usable arrays, canonical readings and selected alternatives, exact `PlaneSignature` records, and the selected match identity, quadrant, shift, homography, correlation algorithm, and radius. Replay signatures and matcher support. Bind truth, camera, reading, signature, match, sampling, ownership, and source hashes.

## Exact sampling witness
Reconstruct the actual source-pixel dependencies for every signature cell: rectification bilinear interpolation and nonzero taps, `np.gradient` neighbour samples, cell aggregation, matcher nearest-neighbour cell mapping, and selected shift convention. Cell centres, bounding polygons, majority ownership, and interpolated owner IDs are insufficient substitutes.

## Ownership and identity
A cell is `pure(S)` only when every contributing source pixel is valid, usable, and unambiguously owned by the same physical plane `S`. Unknown owner, border contribution, or unresolved transparency yields `unknown`; multiple known planes yields `mixed`. Surface identity is `(truth_sha256, normal_axis, outward_sign, exact dyadic plane coordinate)`. Resolve owners through bound block/model geometry; ambiguous subfaces remain unknown.

## Pair accounting
Let `P` be the exact logical cell pairs counted by the selected matcher shift; require `|P| == match.overlap_cells`. For surface `S`, `N_S` counts pairs whose two endpoints are pure `S`. Report `surface_support_cells=N_S`, distinct endpoint counts, `surface_share_of_pair_support=N_S/|P|`, and `surface_overlap_fraction=N_S/min(A_S,B_S)` where `A_S/B_S` are pure-S covered cells in each matcher canvas; zero denominators are null. Partition every pair into same-surface, different-surface, mixed, or unknown, with precedence unknown > mixed > different > same; counts must sum to `|P|`.

## Output and refusal
Use schema `mc-photogrammetry-surface-local-support-v1`, including input bindings, sampling/version IDs, selected match/shift, footprint and ownership digests, explicitly labelled whole-reading metrics, per-surface counts/fractions, exclusion counts, and canonical report hash. Refuse malformed or stale inputs, signature/support replay mismatch, unavailable exact dependencies, uncertifiable ownership, and budget exhaustion atomically. Do not convert global PSR/structural-share or the `64/.5/.15/3` capture targets into surface-local evidence.

## Required tests
Pure single-plane; tiny surface inside broad reading; two planes in one cell; bilinear boundary mixing; gradient-neighbour crossing; unknown/background/transparent ownership; ambiguous partial-block subfaces; all quadrants, reversed pairs, borders, and shifts ±32; exact conservation; tamper/version/budget refusal; deterministic ordering; unchanged production outputs.

## Implementation boundary
Add `evaluation_surface_support.py` and focused tests. Integrate only with evaluation preflight and version its report. If exact sampling dependencies cannot be recovered through existing helpers, stop with a typed refusal and report a separately reviewed output-neutral seam. No production matcher, thresholds, camera plans, launcher, or metadata path changes.