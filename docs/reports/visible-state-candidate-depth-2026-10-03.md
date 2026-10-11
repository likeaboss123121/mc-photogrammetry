# T6 candidate-depth attribution — 2026-10-03

## Scope

Supplied-camera diagnostic on pinned Template 6, reviewed per-frame JSON masks, vanilla 1.21.5 JAR, and the same no-global-pruning override (`palette_support=0.0`) as the prior ablation. This measures the existing appearance-cost ranking on known cameras; it is neither image-only performance nor an ML-model result. The override makes acceptance ineligible by design.

The trace limit was raised to 32 for evaluator-only capture. Production still builds and uses its existing top-8 candidate set and winner. The trace separately records the ordered top-32 states from the same per-cell mean-cost vector; no diagnostic list feeds reconstruction.

## Findings

- Visible exact state remained 511/2,007 (25.46%); visible geometry F1 remained 85.39%. This reproduces the prior no-pruning output score, consistent with trace-only behavior.
- The visible reference states cover 1,920/2,007 cells in the canonical palette (95.67%); 87 references are outside it. One visible cell had no candidate trace record.
- Existing production top-8 set membership remains 788/2,007 (39.26%). A separately sorted full-cost ranking gives 793/1,920 (41.30%) at K=8, 850/1,920 (44.27%) at K=16 and 932/1,920 (48.54%) at K=32. These ordered-ranking rows are conditional on truth being in the canonical palette; they are not end-to-end exact recovery.
- K=8 differs by five cells from production set membership. Posthoc inspection of the saved full-cost trace and exact visible truth confirmed all five expected states have cost 24.0, equal to `OUTLIER_CEILING` and tied at the production top-8 boundary; multiple other states share that cost. This explains the membership delta as tie selection at the clipped-cost boundary. The five cells are `(12,20,4)`, `(2,20,14)`, `(10,20,12)`, `(1,20,6)`, `(2,20,21)`. Keep the two metrics distinct: the ordered K8 list remains a deterministic diagnostic ranking, while production set membership follows `argpartition`.
- Under the existing scorer, widening from K=8 to K=32 recovers 139 additional in-palette visible states. This is a current-ranking diagnostic ceiling, not a limit on a learned model that scores the full palette or proposes states differently.
- Main remaining losses in this ablation: support-to-candidate loses 1,132 visible exact states (1 geometry absence, 1,131 state); candidate-to-raw-winner loses 318; smoothing nets +49 (59 gains, 10 losses); placement omits 8.

## Implication for a local recognizer

A local texture model is worth a bounded vanilla POC as an auxiliary material/state proposer and scorer, not as a general-purpose AI replacement for the pipeline. It must be evaluated on source-build-held-out data, against the current scorer on identical patches. Measure material top-1/top-k, full-BlockState candidate recall, calibration/abstention and end-to-end pinned metrics separately. The current factory has only three independent build groups and no held-out fold, so its existing captures cannot support a strong generalization claim; first expand grouped capture data or report leave-one-group-out as exploratory only. Preserve geometric state properties as deterministic decisions where possible.

## Reproduction and provenance

The original command used `.venv/Scripts/python.exe scripts/benchmark_templates.py "templates 6" ... --mode supplied-camera --reference "templates 6/Template 6.litematic" --camera-manifest "templates 6/cameras.json" --scene "templates 6/scene.json" --fixture-role template-6 --palette-support-override 0.0 --stage-trace ... --stage-trace-candidate-limit 32 --report ...`, the local 1.21.5 JAR and all four tracked `docs/evaluation_masks/templates-6/*.json` specs. Reconstruction time was 409.53 s; total report time was 480.93 s.

- Report JSON: `.tmp-ds-r36-t6-noprune-report.json`, SHA-256 `C5D8C80DE04E332E522244058A898A6600D39A4B6247FE62A272462231FBA4A6`
- Trace JSON: `.tmp-ds-r36-t6-noprune-trace.json`, SHA-256 `49269C8D4DB7323464AA80E8E7F0B525605CA6DB462AA008AF365EC0BBA6E7B7`
- Candidate Litematic: `.tmp-ds-r36-t6-noprune-output.litematic`, SHA-256 `D72AD1A35EE3A45F5DE04DE850AEF8112D6E0944779FEC3F3E2EAD42B66B3280`

Lead reran focused verification after the trace extension: `tests/test_raycast.py -k stage_trace` 10 passed; `tests/test_benchmark_templates.py -k 'candidate_availability or stage_trace_candidate_limit'` 16 passed; py_compile passed. The K=8 delta remains a review note, not a production change or acceptance-gate result.
