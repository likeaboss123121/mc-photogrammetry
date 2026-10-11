# M4 Evaluator E1: full-cube-only primary score

E1 strictly loads and verifies the immutable R1 freeze before opening evaluation metadata. It then checks all capture-frame basenames, dimensions, camera/resolution cardinality, screenshot sizes, and PNG hashes before reading ground truth. Rendering uses the recorded capture camera, the exact-state palette/shape cache, screenshot coordinates, and R1's default derived usable-pixel mask. Frozen proposals are copied and never recomputed or mutated. Promotion remains fixed false because the region-bounded GT omits exterior world occluders.

## Ownership correction

Renderer ownership identifies only `(cell, face slot)`, not a source polygon. The initial all-model result assigned repeated same-slot baked-face offsets by visible occurrence, which can shift when an earlier polygon was culled or fully occluded. That result is now retained only under `provisional_owner_face_ambiguity` in the revised artifact; its earlier standalone `evaluation.json` remains unchanged for provenance and is superseded as the primary score.

The primary `full_cube_only` score resolves each owner through its cell state and includes only palette entries with `full_cube=true`. It groups those visible owners at integer cell boundaries, avoiding sub-face occurrence assignment. Pixels owned by non-full-cube states are removed from both proposal and truth scoring validity; usable background remains valid so unrelated prediction pixels still count as false positives. Across the eight views, 232,755 owner pixels from non-full-cube blocks are excluded. The all-face audit also reports 59,865 pixels belonging to state/slot groups with multiple cached same-slot offsets, a subset that may be misassigned in the provisional grouping. The synthetic culled-face test keeps both owner-table entries but gives pixels only to the second same-slot sub-face; primary scoring excludes the partial-block pixel and retains only the independent full-cube cell plane.

## Revised result

The revised scoring run completed on the same immutable eight-frame R1 freeze and the same explicitly supplied capture, GT, jar, and palette cache. It took about 110 seconds. Of 6,256 GT records, 4,002 explicit air/cave-air/void-air records were filtered; 89 distinct non-air states remained and 48 used the approved unique subset-compatible cached render-variant projection. No palette cache or jar was written.

The primary renderer produced 296 full-cube visible plane groups across all views. R1 has three proposals in frame 00 and none in frames 01–07. The three frame-00 masks match 3/42 primary groups (recall@3 0.0714); there are no matched planes in the other frames. Across eight frames, primary micro TP/FP/FN are 21,287 / 1,219,342 / 482,495, giving precision 0.0172, recall 0.0423, F1 0.0244, and IoU 0.0124. Per-frame non-full-cube exclusions are 24,618; 35,645; 42,549; 44,512; 19,489; 20,308; 29,244; and 16,390 pixels. The output separately retains the previous all-model metrics as provisional: TP/FP/FN 26,597 / 1,276,900 / 709,940; P/R/F1/IoU 0.0204 / 0.0361 / 0.0261 / 0.0132. Those numbers are not the primary score.

Calibration comparison is unchanged: vertical-FOV focal error <=2% on 5/8 frames; best-permutation absolute-dot Manhattan axis-line error <=0.5 degrees on 5/8; median focal error 1.4579%; pooled median axis-line error 0.1738 degrees. Promotion is false for both evaluator and provisional result.

## Validation and provenance

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frozen_plane_evaluation.py tests\test_frozen_image_runner.py tests\test_attribution.py tests\test_visibility.py tests\test_camera.py tests\test_shape_persistence.py -o addopts= -p no:cacheprovider -q
```

Result: **98 passed**. `py_compile` passed for the evaluator and wrapper. `git diff --check` over the whole shared worktree reported pre-existing CRLF/trailing-whitespace issues in unrelated historical files; no formatting result is claimed for that broad scan. The evaluator's production-direction AST guard remains in the focused tests.

Revised artifact: `exports/evaluation/frozen-plane-e1-canal-town-2026-09-12/evaluation-full-cube-primary.json`, 289,806 bytes, SHA-256 `a7a8a8e93e0921d0fc5f1dfd4a008d44de3fc396fc1205dbb5a4f4a3cd7557fa`, artifact ID `6ca613fcb413c24bc0d80363f1d4aa5bb63778b01d8e03a8e2c143bcd38b095e`. It binds R1 artifact ID `193536b469d60fec210d7571c8f2537816ac3a6c3102c0889a4040c3eee1734e`; freeze SHA-256 `60eb97973473596ca7ba44c9c305e386615043ff336d9bd81d30e2de28744231`; sidecar SHA-256 `e55383ac4279a64137497823ff4f7ca07199fc56fff980e577a9dcb36767115c`; capture SHA-256 `c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`; GT SHA-256 `36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`; jar SHA-256 `522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11`; palette-cache SHA-256 `d119381cf3717d6e300c7da73701c3d0f8358a5b7b9d6091697bee2c1e2c1934`.

Evaluator source SHA-256 `2c032d1798ca962a2361e480019b222e8080e7b96ceeb5397c2b78153a1fe031`; focused test SHA-256 `5c31af04249b93b7158297c669099bfdcc9c254dd775a64e05aa9b25ce2092ae`. R1 freeze/sidecar and capture artifacts were not modified. No reconstruction/CLI, attribution renderer, thresholds, capture, or stage/commit actions were made.
