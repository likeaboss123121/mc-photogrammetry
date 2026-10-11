# M4-C3 image-only plane-proposal recall repair

Status: **negative preregistered diagnostic**. This bounded diagnostic repairs proposal coverage only; it does not integrate plane proposals into reconstruction or change the default image-only product path.

The proposal generator consumes screenshot pixels, an explicitly derived usable mask, and a fresh image-derived Manhattan `CalibrationResult`. Production source contains no annotation, manifest, reference-world, or fixture-camera access. Existing `find_chains` and `propose_boundary_candidates` call signatures remain stable. The new `PlaneProposalResult` reports abstentions with a reason when calibration or usable coverage is insufficient.

The frozen algorithm detects line segments once, clips them to usable pixel runs, and reuses the evidence for planar-region votes at 24, 48, and 96 pixel tiles plus same-axis line-chain closures. Region area is held at a 13,824-pixel minimum and limited to 12 regions per scale with axis/spatial coverage. The chain pool is limited to 12 per Manhattan axis (36 total); closures are limited to 32. Proposals are deduplicated only when their masks share the same axis, signed side, and at least 0.55 IoU. Evidence may add a source tag and score, but masks are never unioned. Final ranking spreads candidates across axis, calibrated side, and a 3x3 image grid, with a hard cap of 32 per frame.

Synthetic controls cover frame-truncated walls, a furniture occluder, low-contrast seams, same-texture adjacent orthogonal planes, calibration roll propagation, view-order determinism, typed abstentions, production-source isolation, the 32-candidate cap, and bounded runtime. The orthogonal-wall test uses exact synthetic line detections to isolate plane-axis association from LSD variance. Test and source evidence is in `pre_score_freeze.json`.

## Frozen source and evaluation

| Item | Evidence |
|---|---|
| Pre-score source aggregate SHA-256 | `c015bffc5df0efd0a77f928527577c451e0fbc0a14680ee0969715a7fc2636ed` |
| Frozen v4 target SHA-256 | `f4a0f86938561b3fc81b096fc614d1ae404c1372e65c925811c1368ce56a0da5` |
| Evaluation command | `.\.venv\Scripts\python.exe docs\reports\M4-C3-image-plane-proposals-2026-09-12\evaluate_frozen_v4.py` |
| Evaluation count | one scored run against all three T2 frames; the preceding schema-validation attempt exited before writing the one-shot marker and scored nothing |
| Pixel-evidence overlays | `M4-C3-image-plane-proposals-2026-09-12/overlays/` |

Proposal recall@K is the maximum **individual** candidate recall among the first K strength-ranked proposals; candidate masks are never unioned. The evaluator reports the maximum-IoU candidate and its precision/recall/IoU, the maximum-recall candidate, and actual top1 separately. The derived usable mask is `~create_gui_mask(height, width)` with the standard debug-area setting off. Each per-frame denominator remains all three v4 frames, including abstentions.

| Frame | Candidates | Recall@1 | Recall@5 | Recall@10 | Recall@32 | Best-IoU P/R/IoU | Top1 P/R/IoU |
|---|---:|---:|---:|---:|---:|---|---|
| 2026-09-02 13:22:14 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 2026-09-02 13:24:24 | 3 | 0.000 | 0.025 | 0.025 | 0.025 | 1.000 / 0.025 / 0.025 | 0.000 / 0.000 / 0.000 |
| 2026-09-02 13:24:33 | 8 | 0.019 | 0.060 | 0.060 | 0.060 | 1.000 / 0.060 / 0.060 | 1.000 / 0.019 / 0.019 |

The preregistered continuation condition is: a candidate in every frame; maximum individual recall at least 0.50 in all three; maximum-IoU candidate precision at least 0.50 in at least two frames; and zero candidate-mask unions across planes. Candidates existed in all three frames, best-IoU precision met the requirement in two frames, and no masks were unioned. Best individual recall was below 0.50 in every frame, so the gate **failed**.

Pixel audit of the saved overlays agrees with these scores: frame 1's sole orange proposal is on the right-hand wall while the v4 positive mask covers the broad left wall; frame 2's best-overlap proposal is a small tile on the positive wall while top1 lies elsewhere; frame 3's best-overlap region touches the positive left wall but covers only a small fraction, while top1 is a separate small patch. Candidate masks remain separate in each overlay and in the scorer.

## Validation and limits

Focused regression: `tests/test_boundary_candidates.py`, `tests/test_calibration.py`, and `tests/test_surfaces.py` — **60 passed in 88.02s**. `py_compile` passed for the proposer, surfaces helper, and corrected evaluation script. The evaluator-only loader correction now validates screenshot byte hashes pinned by the accepted audit because target rows do not contain a source-image-hash field; production code, tests, and algorithm constants did not change, and all source hashes were re-frozen before the one scored run. No camera/pose metadata, annotation thresholds, search sweep, DS-29 files, reconstruction integration, dataset-factory code, git staging, or commit are part of this package. The result is diagnostic-only and annotation-dependent. A next bounded investigation should explain the absent broad planar-region proposals and the small-tile fragmentation from image evidence before proposing any new thresholds or changing the frozen generator.
