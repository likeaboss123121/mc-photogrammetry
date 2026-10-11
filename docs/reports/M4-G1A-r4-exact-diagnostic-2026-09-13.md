# M4 G1A: exact 8-view solver unblocked; R4 remains ambiguous

Date: 2026-09-13

G1A changed only the deterministic assignment cap in plane_graph.solve_component from 256 to 16,384, enough to exhaust all 4^(8-1) quadrant assignments with the anchor fixed at zero. The objective, reading-index identity, selected b values, match/quadrant constraints, translations, cost terms, tie order, ambiguity logic, thresholds, alternative limit, output schema, and abstention vocabulary are unchanged. Nine-view components still stop before enumeration with component_too_large. No assignment is pruned.

The label-blind preregistered diagnostic strict-loaded R4’s freeze and mask sidecar and used only its serialized PlaneReading and PlaneMatch records; it did not open PNG pixels or any label, capture, manifest, reference, or evaluator data. Baseline output m4-g1a-r4-baseline.json confirmed the old cap 256 rejected the 16,384-assignment case with component_too_large before any assignment evaluation. Under the new cap, all 16,384 assignments were attempted and all 16,384 produced an evaluable assignment. Runtime was 20.199 seconds. The frozen graph supplied 8 frames, 33 readings, 200 matches, and 71,216 match refusals.

The exact solve stops at ambiguous_quadrant. Best cost is 60.04393342612463; runner-up cost is 60.04393342612466; margin is 2.842170943040401e-14, far below the unchanged 0.02 ambiguity margin. Their quadrant tuples differ at one frame: best [0,3,1,0,2,3,1,3], runner-up [0,3,3,0,2,3,1,3]. Best resultants for axes 0/1/2 are 0.498715/0.245562/0.411718; residuals are 0.144720/0.197856/0.163475. The exact selected per-reading b values, best/runner-up records, and retained alternatives are in the diagnostic JSON. Because the existing objective remains ambiguous and resultant support is below the unchanged 0.90 requirement, G1A stops here: no R5, objective change, threshold tuning, or redesign is authorized by this package.

Focused tests passed: 17 passed in 27.30 seconds for the complete plane-graph test file. The tests compare exact public-field/dataclass outputs with the legacy 256 cap for existing small-component fixtures; exercise an eight-view unique star, eight-view vertical q/q+2 ambiguity, nine-view typed abstention, order invariance and repeatable serialization; and assert exactly 16,384 assignment evaluations, runtime below 120 seconds, and traced Python peak memory no more than 128 MiB on the largest synthetic solve. The 16,384 cap is therefore both exact for eight views and explicitly bounded. py_compile passed for the modified solver and diagnostic.

Protected hashes cover 20 files: R1-R4 freezes and sidecars, E1-E3 result JSONs, and the eight ordered R4 input PNGs. All 20 pre/post values match. No R5 artifact exists. No evaluator/GT, capture, graph match generation, staging, or commit occurred.

## Artifacts and hashes

- Preregistration: docs/reports/M4-G1A-prereg-2026-09-13.md, SHA-256 ac5ac010a0d707e7e0a7e43b6b0b223719d030cc506d163aa828be0f2e52e40f.
- Pre-edit R4 baseline: exports/evaluation/m4-g1a-r4-baseline.json, SHA-256 b122a1f2989b1d90a7b43c84ab18dc601865f479e078e3007f205e4201eed3d2.
- Exact R4 diagnostic: exports/evaluation/m4-g1a-r4-exact.json, SHA-256 4c0117015c9f21766c1baae671648c795d5d9cfa0b370a3098e96cc0da38bc681.
- Protected hash record: docs/reports/M4-G1A-protected-hashes.json, SHA-256 72c51b6301c73706a4b73453a7bf3485016c3e754404c0131fcde3fbcda05d28.
- plane_graph.py: SHA-256 961459b5a9f7fcb640448cd74be804e03a1b2c1b24a4522fdf7fb89395bd1e16.
- tests/test_plane_graph.py: SHA-256 1b5e5b4554aa369015c18cd38dcd4160d2fd35f15ac00e7f837cbaab22d0aba6.
- scripts/diagnose_m4_g1a.py: SHA-256 fdf067a575ac5a6e9f91c581fa9bc39ae82978dbff29701a04ae3fd1cde46783.

R4 itself remains immutable: freeze SHA-256 96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d; sidecar SHA-256 f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac. R1/R2/R3 and E1-E3 are unchanged.