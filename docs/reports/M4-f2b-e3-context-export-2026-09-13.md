# M4 F2B context export and E3 evaluation

Date: 2026-09-13. This is an evaluator-only context artifact; it is not reconstruction input. Promotion remains false unless context boundary and occluder completeness are separately proven.

## Frozen inputs and context export

- Scene/world: `canal-town-architectural-component-bbox` / `Canal Town`.
- Capture SHA-256: `c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`.
- Target GT SHA-256: `36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`; inclusive region `[154,66,361]..[170,88,376]`.
- R2 freeze SHA-256: `3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f`; its eight ordered PNG basenames, hashes, and dimensions match capture metadata.
- Context region: inclusive `[140,60,340]..[190,130,400]`, dimensions `51x71x61`, volume `220,881` (under the `250,000` cap), strictly contains target GT. All eight camera feet are inside the context bounds.
- Source save fingerprint before and after the successful context export: `b4aeeb844025284939e51754be3eba6cce16a876d2500bb9d14ab4ba8a48e54e`.
- Successful context-only run used the distinct per-scene worker `MCPhotoCapture-Worke-canal-town-architect-1846fafc6f8db3dba29e`, Desktop 2, and the staged muted/background-safe options. Mod status proved context-only mode, zero jobs/results/frames, and successful GT export; no PNG was produced and process-tree teardown was confirmed.
- Strict F2A artifact: `exports/evaluation/context-gt-f2b-canal-town-2026-09-12-attempt2/context-ground-truth.json`, 28,685,500 bytes, SHA-256 `180e49099e3e26ee957671dd94c3410a9fbe1124802d4f8976d6a4eac3bf2f8d`. It contains 220,881 complete ordered block records and binds capture, target, scene/world, region, and ordered R2 image hashes/dimensions.

## Aborted launch and deployment correction

The first context-only launch was aborted before export. The plan correctly had `context_export_only: true` and zero jobs, but the staged capture JAR was stale: its `PlanAudit.class` lacked the explicit mode. The log refused the plan with “plan must contain at least one job.” The launcher was interrupted through its teardown path; a read-only process check found no remaining process. That attempt produced no status/results/context artifact and was not scored. Its staged output and worker are preserved as failure evidence.

The source tests had passed, but the plain `jar` task left the old remapped deployment JAR in place. The verified deployment build is:

```powershell
$env:JAVA_HOME='C:\Program Files\Java\jdk-22'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat --no-daemon clean test remapJar
```

It passed the Java suite, including `PlanAuditTest.acceptsZeroJobsOnlyForExplicitContextExportOnlyPlans`. ZIP inspection confirmed the final remapped `PlanAudit.class` and `CaptureController.class` contain the explicit context-only contract and terminal status fields. JAR: 75,672 bytes, SHA-256 `d63a7eaf5d6b413fee3cc2aa25269fad3fefcf6b0af66b96dac3e751a559597c`.

## Preregistered selector-equivalence fallback

The first E3 evaluator invocation wrote no result because context GT contains 50 `minecraft:dark_prismarine_slab{type=top,waterlogged=true}` records with no exact/strict-subset cache entry. The unchanged palette cache has three same-name samples (`type=bottom|double|top`), all `waterlogged=false`. The supplied vanilla JAR asset `assets/minecraft/blockstates/dark_prismarine_slab.json` (SHA-256 `b91e6454c83b8ee55e3aef8865d9aa0c18097214a1d1d70e375bb424e3ac8f42`) has variants selected by `type` only.

For the single follow-up score, the evaluator may project an otherwise-unavailable exact state only if the exact supplied JAR blockstate JSON proves that (a) target and cached sample select the same unique variant/multipart signature, (b) every property referenced by those selectors is present and equal in both states, and (c) every differing property is absent from all selector conditions. Missing JAR assets, malformed selectors, mismatches, or zero/multiple matching cache samples refuse. The output records the JAR and blockstate asset hashes, selected signature, ignored properties, and exact rule. For `waterlogged`, the baked-block projection does not render the fluid overlay; this limitation is reported and promotion remains false. Exact context GT rows, palette cache, captures, masks, thresholds, and production modules remain unchanged.

Focused tests passed: `tests/test_frozen_plane_evaluation.py` (37 passed); `py_compile` passed. Frozen evaluator source SHA-256 `3918388075dcfab7b9cde0788bbe930df7ef295041a6871bfd045a0594aee91c`; test SHA-256 `7407c1d43d30e0537d12aa7819f8d131948c3839cb974060026abdaee04c02b0`.

The one post-fix E3 invocation was preregistered to write only to the fresh path `exports/evaluation/frozen-plane-e3-context-canal-town-2026-09-12-attempt3/evaluation-r2-context.json`. Before that invocation, R1/R2, E1/E2, capture, target GT, context artifact, vanilla JAR, palette cache, and evaluator-source hashes were verified.

## Single post-fix E3 result

The evaluator ran once after the selector fallback was frozen. R2, capture/target GT, context artifact, vanilla JAR, and palette cache were read-only inputs. Result: `exports/evaluation/frozen-plane-e3-context-canal-town-2026-09-12-attempt3/evaluation-r2-context.json`, 676,834 bytes, SHA-256 `fd276b821e96757f39afc902f97ffa8a6e866ada317ad08a16bb0ef2d36a600b`, artifact ID `8fd3435659cfce7e44e9b44df68afce8e797386cfc836aba91996f6289d3d168`. Output code hash for the evaluator matches the frozen source SHA above. The state-projection record binds the supplied vanilla JAR SHA `522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11` and blockstate asset SHA `b91e6454c83b8ee55e3aef8865d9aa0c18097214a1d1d70e375bb424e3ac8f42`; it records `waterlogged=true` versus cached `false` as ignored because that property is absent from selectors, and explicitly notes the unrendered fluid overlay.

The primary E3 context-known-domain totals are TP/FP/FN `980,088 / 3,511,450 / 1,036,912`, precision `0.218208`, recall `0.485914`, F1 `0.301170`, IoU `0.177281`. The E2 target-bounded totals are TP/FP/FN `157,521 / 10,761,705 / 346,261`, precision `0.014426`, recall `0.312677`, F1 `0.027580`, IoU `0.013983`. These metrics use different known domains: E3 treats pixels with a nearest context owner as known and excludes no-owner usable pixels from both prediction and truth. The precision/recall changes are descriptive and are not a like-for-like quality comparison.

E3 domain totals: 11,232,000 image pixels; 6,850,796 usable; 2,574,529 known context-owned (2,017,000 full-cube positive and 557,529 other-owner known-negative); 4,276,267 no-owner usable/unknown and symmetrically excluded; 4,381,204 GUI-masked/excluded pixels. The retained per-frame denominators and primary metrics are:

| Frame | Proposals | Visible planes | Known pixels | Unknown pixels | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 3 | 67 | 126,767 | 435,180 | 0.143555 | 0.476944 | 0.220686 |
| 01 | 5 | 66 | 193,695 | 684,984 | 0.213399 | 0.510053 | 0.300904 |
| 02 | 2 | 73 | 513,896 | 750,563 | 0.227466 | 0.532642 | 0.318792 |
| 03 | 6 | 93 | 359,201 | 905,258 | 0.236187 | 0.253480 | 0.244528 |
| 04 | 4 | 93 | 257,956 | 303,991 | 0.201205 | 0.527199 | 0.291254 |
| 05 | 7 | 89 | 451,098 | 427,581 | 0.288223 | 0.481195 | 0.360510 |
| 06 | 4 | 101 | 444,982 | 433,697 | 0.186553 | 0.638174 | 0.288710 |
| 07 | 4 | 88 | 226,934 | 335,013 | 0.282149 | 0.378166 | 0.323176 |

All eight frozen views remain in the denominator. Calibration summary matches E2: median focal error `1.4579%`, median Manhattan axis-line error `0.1738°`, with each tolerance met on 5/8 frames. Promotion is false: context boundary/exterior-occluder completeness is unproven and the waterlogged fluid overlay is not rendered.

Post-score verification found unchanged R1/R2 JSON and sidecars, E1/E2, capture, target GT, vanilla JAR, palette cache, and source-save fingerprint. R2 SHA/sidecar remain `3c4e8e1b...a6c1c90f` / `68277b11...03bc74db`; capture/target/context SHA remain `c6c44d43...b812abe6` / `36114fee...7a4e35a` / `180e4909...3bf2f8d`; vanilla/cache remain `522672ad...95a8c11` / `d119381c...1e2c1934`. No screenshot, queue, reconstruction, graph, CLI, or source-save artifact was changed.
