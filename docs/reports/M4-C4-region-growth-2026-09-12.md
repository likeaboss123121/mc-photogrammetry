# M4-C4 image-only plane region growth — 2026-09-12

## Result

Added an experimental, deterministic second plane proposer that turns a usable image mask into local Lab+XY atoms, an edge-aware atom graph, and axis/side-specific candidate masks. The new API is exported as `propose_image_planes_v2` and adapts to the existing `PlaneProposalResult`. The default CLI and reconstruction entry path are unchanged; manifests, camera references, labels, and evaluation masks are not production inputs.

Graph edges retain their dominant image boundary orientation. Growth can cross a boundary aligned with either tangent Manhattan axis when the target carries candidate-plane tangent support and there is no strong competing plane. For a supported seam atom to exit into an adjacent low-support wall atom, the latter must resemble one of the candidate's seed Lab appearances. The seed palette is quantized and capped at 512 colors. This allows dark repeated block seams to bridge same-plane wall regions while unsupported, dissimilar furniture remains behind the Lab barrier. Candidate masks remain separate by normal axis and ray side.

The synthetic nonzero-roll fixture was corrected to generate projected line segments whenever roll is nonzero. Previously it drew the unrolled horizontal/vertical scene despite supplying a rolled calibration. With the correction, the rolled case supplies 10 projected segments; all are clipped and assigned, six atoms seed the expected axis-2 plane, and the proposer returns a candidate.

## Fixed resource limits

- Working image max side: 960 pixels; input cap: 6,000,000 pixels.
- Atom spacing: 16 pixels; atom cap: 20,000; SLIC-like assignment: 3 iterations.
- Segment cap: 5,000; at most 128 samples per segment.
- Seed Lab palette: at most 512 entries; returned candidates: at most 32 per frame.
- No `scikit-image` or `cv2.ximgproc` dependency.

The synthetic runtime control passes with the 960×640 input and enforces the 8-second limit. Outputs are deterministic under segment-order changes and contain no pixels outside the supplied usable mask.

## Synthetic evidence and checks

`tests/test_region_growth.py`: **12 passed**. The controls cover a frame-truncated wall, same-texture orthogonal walls, a furniture hole with reconnected visible wall samples, low contrast, repeated texture, competing axis evidence, nonzero roll, view/segment-order determinism, mask purity, abstention, proposal cap, runtime, and a source audit against annotation or external-data access. The truncated front plane grows to 39,668 pixels in the synthetic wall; cross-axis mask overlap stays below 10% of the smaller candidate. The furniture interior remains excluded under the test's 8% hole limit.

Focused regressions across region growth, boundary candidates, calibration, independent calibration, and surfaces: **88 passed in 162.69 seconds** (pytest exit code 0).

An axis-2 synthetic growth trace recorded 727 edge attempts, 487 structurally supported tangent continuations, 417 Lab-barrier overrides backed by both source/target tangent support and seed-color similarity, 645 accepted edges, and 82 remaining hard-edge/Lab rejections. The global Lab threshold was not raised. The source audit found no manifest/reference/v4 reads, and no real or v4 masks were opened or scored for C4.

## Frozen source hashes

SHA-256 values for the implementation and synthetic test source at closeout:

- `src/mc_photogrammetry/reconstruction/region_growth.py`: `c3f35babf8128e41c090ad0415314b3653e2377010f26249ae651db9881c2734`
- `src/mc_photogrammetry/reconstruction/__init__.py`: `f24a9a85ca6f9c0aa9246777b5f1db6fd7f6ecb6c6cd34c2a1bfb022efec4bff`
- `tests/test_region_growth.py`: `bd3b6ca33bf10f7f8060c222ad7bcd1abc152b0ab062d1dec6cf41a72c03b864`

No fresh factory holdout is available in this task. This is synthetic-only implementation evidence; no real-capture proposal recall or precision is claimed. Default image-only product behavior remains unchanged pending lead review and a fresh holdout.
