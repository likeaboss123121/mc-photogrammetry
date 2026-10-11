# Minecraft Photogrammetry Progress

Updates append every three hours while Windows is awake and the user session is available. The local scheduled task reads only PROJECT_PROGRESS_STATE.json and appends a snapshot; it does not run an AI model, reconstruction, or Minecraft. The lead updates that state after reviewed project evidence. If Windows sleeps through a run, the next available scheduled run appends one current snapshot; it does not backfill every missed interval.

Each entry reports exact-state match, the fixed-pipeline candidate ceiling, match as a percentage of that ceiling, change since the prior entry, and expected point gain before the next entry. The ceiling means exact state existed in the candidate set at the aligned coordinate; it is not a theoretical maximum for future algorithms. N/A means no comparable score exists. First row is the baseline; its gain is not comparable to an earlier progress-log entry.

## Update 2026-10-07 07:19:37 -05:00

Status: Image-only path has not yet produced a scored reconstruction; the documented release gates remain unmet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | Baseline | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2,007) | 19.58% (393/2,007) | 89.06% | Baseline | 0.00 pp |
| Template 6 no-pruning ablation | 25.46% (511/2,007) | 39.26% (788/2,007) | 64.85% | Baseline | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

These are historical supplied-camera diagnostics from 2026-10-03. They do not measure normal image-only performance. The latest image-only attempt refused before a usable output or score. The 0.00 pp forecast reflects source-only work and no authorized image-scored run; it is not a prediction that future algorithm work cannot improve accuracy.

What happened (max 250 chars): Reviewed architecture and factory readiness. Astra rejected a VGGT-to-lattice wrapper; DeepSeek found only two factory builds, train-only and evaluation-only. No image-scored reconstruction output changed.

What happens next (max 250 chars): Finish the world-split contract review and, if approved, route its additive source implementation. No score gain is expected until an authorized image-scored end-to-end run.

---## Update 2026-10-07 10:21:13 -05:00

Status: Image-only reconstruction still refuses before scored geometry or litematic delivery; historical supplied-camera scores are unchanged.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Parsed the saved graph capture: 70 input matches became 437 candidate rows across 16 groups; 421 were dropped by same-axis filtering. An axis-0 alternative closes, but full three-axis feasibility is unproven. No search or score ran.

What happens next (max 250 chars): Keep v1 unchanged and the diagnostic search opt-in. With authorization, run the bounded full-axis feasibility check; if it closes, design a capped selector that abstains on ambiguity and preserves production gates.

Forecast basis: Artifact-only review; no search or reconstruction scoring authorized.

---
## Update 2026-10-07 13:21:14 -05:00

Status: Image-only reconstruction still refuses before scored geometry or litematic delivery; historical supplied-camera scores are unchanged.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Parsed the saved graph capture: 70 input matches became 437 candidate rows across 16 groups; 421 were dropped by same-axis filtering. An axis-0 alternative closes, but full three-axis feasibility is unproven. No search or score ran.

What happens next (max 250 chars): Keep v1 unchanged and the diagnostic search opt-in. With authorization, run the bounded full-axis feasibility check; if it closes, design a capped selector that abstains on ambiguity and preserves production gates.

Forecast basis: Artifact-only review; no search or reconstruction scoring authorized.

---
## Update 2026-10-07 16:21:14 -05:00

Status: Image-only reconstruction still refuses before scored geometry or litematic delivery; historical supplied-camera scores are unchanged.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Parsed the saved graph capture: 70 input matches became 437 candidate rows across 16 groups; 421 were dropped by same-axis filtering. An axis-0 alternative closes, but full three-axis feasibility is unproven. No search or score ran.

What happens next (max 250 chars): Keep v1 unchanged and the diagnostic search opt-in. With authorization, run the bounded full-axis feasibility check; if it closes, design a capped selector that abstains on ambiguity and preserves production gates.

Forecast basis: Artifact-only review; no search or reconstruction scoring authorized.

---
## Update 2026-10-07 19:21:15 -05:00

Status: Image-only reconstruction still refuses before scored geometry or litematic delivery; historical supplied-camera scores are unchanged.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Parsed the saved graph capture: 70 input matches became 437 candidate rows across 16 groups; 421 were dropped by same-axis filtering. An axis-0 alternative closes, but full three-axis feasibility is unproven. No search or score ran.

What happens next (max 250 chars): Keep v1 unchanged and the diagnostic search opt-in. With authorization, run the bounded full-axis feasibility check; if it closes, design a capped selector that abstains on ambiguity and preserves production gates.

Forecast basis: Artifact-only review; no search or reconstruction scoring authorized.

---
## Update 2026-10-07 22:21:15 -05:00

Status: Renderer-in-loop implementation is not yet viable: the normal image-derived path refuses before a candidate scene; staged feasibility plan saved.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Global 8/8 attempt cap is revoked. R5 recovered child-local periods but parent readings stayed unchanged and geometry still refused; diagnostic is incomplete, with no litematic or image-only score.

What happens next (max 250 chars): Trace why local child periods disagree across windows; design a separately reviewed consistency test without auto-promoting child periods. Then target the next measured geometry loss.

Forecast basis: Diagnostic-only period support cannot establish accuracy. The global 8/8 attempt cap is revoked; per-run input, memory, and evidence limits remain.

---
## Update 2026-10-08 01:21:14 -05:00

Status: Global 8/8 attempt cap revoked. R5 recovered local child periods but parent readings and graph refusal were unchanged; Astra-approved local-lattice bridge implementation is assigned. No litematic or exact-state score yet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Astra accepted the local-lattice bridge task after mask/evidence/candidate-accounting amendments. Sol owns implementation; DeepSeek mapped source and Luna checked scope. No new accuracy score.

What happens next (max 250 chars): Complete synthetic full-assembler/signature/matcher gates and Astra review. Then one six-image comparison only if accepted; require unchanged-gate endpoint admission or stop.

Forecast basis: This experiment can improve image-derived geometry evidence, but will not itself prove quality. Exact-state score remains unavailable until default-mode placement, state promotion and .litematic delivery succeed. Global attempt cap revoked; local run limits remain.

---

## Update 2026-10-08 07:17:21 -05:00

Status: Global 8/8 image-attempt ceiling is revoked. Accepted observer completed one ordinary six-image run, but G7 had zero eligible assignments; no `.litematic` or image-only accuracy score.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of visible truth cells whose exact state exists in the current candidate set at the allowed aligned coordinate. These assisted/template metrics are historical and are not image-only production scores.

Image-only production: 6/6 images admitted. The selection proof exhausted 256/256 assignments; all failed before numerical geometry. Reason counts: 240 `unsupported_reading_endpoint`, 16 `axis_0_disconnected`. No `.litematic`; exact-state score unavailable.

What happened (max 250 chars): Removed the global 8/8 attempt ceiling, accepted the bounded observer, and ran six ordinary photos once. All 256 endpoint assignments failed before numerical geometry: 240 lacked active endpoint support; 16 had disconnected world axis 0. No accuracy change.

What happens next (max 250 chars): Diagnose candidate topology and per-assignment endpoint support. Preserve production's three-axis observability gate; implement the smallest candidate-generation correction with synthetic controls, then compare once on the same six photos after review.

Forecast basis: C5 measured the failure but produced no usable structure or exact-state metric. Expected image-only gain by the next report is 0.00 pp until a new default-path run yields eligible geometry and a valid `.litematic` that can be scored.

---
## Update 2026-10-08 04:21:14 -05:00

Status: Global 8/8 attempt cap revoked. R5 recovered local child periods but parent readings and graph refusal were unchanged; Astra-approved local-lattice bridge implementation is assigned. No litematic or exact-state score yet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Candidate07 image-only attempt refused before a usable litematic/score; no current image-only exact-state metric.

What happened (max 250 chars): Astra accepted the local-lattice bridge task after mask/evidence/candidate-accounting amendments. Sol owns implementation; DeepSeek mapped source and Luna checked scope. No new accuracy score.

What happens next (max 250 chars): Complete synthetic full-assembler/signature/matcher gates and Astra review. Then one six-image comparison only if accepted; require unchanged-gate endpoint admission or stop.

Forecast basis: This experiment can improve image-derived geometry evidence, but will not itself prove quality. Exact-state score remains unavailable until default-mode placement, state promotion and .litematic delivery succeed. Global attempt cap revoked; local run limits remain.

---
## Update 2026-10-08 07:21:14 -05:00

Status: Global 8/8 attempt cap revoked. C5 admitted six ordinary images but exhausted 256 endpoint assignments before numerical geometry; no litematic or image-only exact-state score.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Six of six images admitted; zero of 256 endpoint assignments eligible; numerical geometry not entered; no litematic or exact-state score.

What happened (max 250 chars): Accepted G7 observer ran once on six ordinary photos. All 256 assignments failed: 240 unsupported_reading_endpoint, 16 axis_0_disconnected. No accuracy change.

What happens next (max 250 chars): Diagnose candidate topology and per-assignment endpoint support; preserve downstream observability gates. Implement a source-backed candidate-generation fix with synthetic controls, then compare once on the same six photos after review.

Forecast basis: C5 is a failure diagnostic, not a reconstruction or score. Expect 0.00 pp until eligible geometry yields a valid litematic that can be scored.

---
## Update 2026-10-08 10:21:14 -05:00

Status: Global 8/8 attempt cap revoked. C5 admitted six ordinary images but exhausted 256 endpoint assignments before numerical geometry; no litematic or image-only exact-state score.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Six of six images admitted; zero of 256 endpoint assignments eligible; numerical geometry not entered; no litematic or exact-state score.

What happened (max 250 chars): Accepted G7 observer ran once on six ordinary photos. All 256 assignments failed: 240 unsupported_reading_endpoint, 16 axis_0_disconnected. No accuracy change.

What happens next (max 250 chars): Diagnose candidate topology and per-assignment endpoint support; preserve downstream observability gates. Implement a source-backed candidate-generation fix with synthetic controls, then compare once on the same six photos after review.

Forecast basis: C5 is a failure diagnostic, not a reconstruction or score. Expect 0.00 pp until eligible geometry yields a valid litematic that can be scored.

---
## Update 2026-10-08 13:21:14 -05:00

Status: Global 8/8 attempt cap revoked. C5 admitted six ordinary images but exhausted 256 endpoint assignments before numerical geometry; no litematic or image-only exact-state score.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Six of six images admitted; zero of 256 endpoint assignments eligible; numerical geometry not entered; no litematic or exact-state score.

What happened (max 250 chars): Accepted G7 observer ran once on six ordinary photos. All 256 assignments failed: 240 unsupported_reading_endpoint, 16 axis_0_disconnected. No accuracy change.

What happens next (max 250 chars): Diagnose candidate topology and per-assignment endpoint support; preserve downstream observability gates. Implement a source-backed candidate-generation fix with synthetic controls, then compare once on the same six photos after review.

Forecast basis: C5 is a failure diagnostic, not a reconstruction or score. Expect 0.00 pp until eligible geometry yields a valid litematic that can be scored.

---
## Update 2026-10-08 16:05:00 -05:00

Status: Global 8/8 attempt cap revoked. C12 processed six ordinary images, but the joint endpoint proof rejected all 256 assignments. No litematic or image-only exact-state score exists.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a limit on future algorithms.

Image-only production: C12 admitted all six images; 100 same-axis rows reached joint selection; the exhaustive 256-assignment proof found zero eligible assignments. Diagnostic status incomplete; no geometry score, exact-state score, or litematic.

What happened (max 250 chars): C12 processed six ordinary photos in 280s. 100 same-axis rows reached joint selection; all 256 assignments were excluded (0 eligible). Input-context fix passed 58 tests. No score or litematic.

What happens next (max 250 chars): Analyze C12 exclusions to isolate endpoint constraints versus candidate topology. Keep thresholds/caps frozen; implement and test one image-derived fix, then rerun after review.

Forecast basis: 0.00 pp expected; no eligible geometry reaches a scoreable litematic.

---## Update 2026-10-08 16:21:14 -05:00

Status: Global 8/8 attempt cap revoked. C12 processed six ordinary images; joint endpoint proof exhaustively rejected all 256 assignments. No litematic or image-only exact-state score.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: C12: six inputs admitted; 100 same-axis rows; exhaustive endpoint proof rejected 256/256 assignments (0 eligible); diagnostic incomplete; no litematic or exact-state score.

What happened (max 250 chars): C12 processed six ordinary photos in 280s. 100 same-axis rows reached joint selection; all 256 assignments were excluded (0 eligible). Input-context fix passed 58 tests. No score or litematic.

What happens next (max 250 chars): Analyze C12 exclusions to isolate endpoint constraints versus candidate topology. Keep thresholds/caps frozen; implement and test one image-derived fix, then rerun after review.

Forecast basis: No eligible endpoint assignment means no numerical geometry, valid litematic, or score. Forecast 0.00 pp until a scored default-mode output exists.

---
## Update 2026-10-08 17:10:00 -05:00

Status: The global 8/8 cap is revoked. The quota-split geometry repair is reviewed; its one authorized synthetic run failed before publication and was not retried. No production accuracy score changed.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. These supplied-camera metrics are assisted diagnostics, not default-mode acceptance scores.

Image-only production: latest real-image evidence remains C12: six inputs; 100 same-axis rows; all 256 endpoint assignments rejected. No valid litematic or image-only exact-state score exists. The quota-v1 synthetic calibration exited at report validation; it produced no result and no accuracy conclusion.

What happened (max 250 chars): The quota split exposed and fixed two report-validator gaps. Astra, Luna and DeepSeek reviewed; 211 focused+historical tests pass. One synthetic run failed before report publication and was not retried. No image score changed.

What happens next (max 250 chars): Prepare a versioned quota-v2 evaluator on four fresh dev seeds, review it, then request one dev-only run. It measures geometric support only; default-mode state accuracy still needs a scoreable litematic.

Forecast basis: 0.00 pp; the next task is a synthetic geometry diagnostic and cannot change the measured screenshot-to-litematic exact-state score.

---
## Update 2026-10-08 17:22:00 -05:00

Status: Global 8/8 cap revoked. Fresh quota-v2 synthetic calibration completed once and returned no-go. Production exact-state metrics and candidate ceilings remain unchanged.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. Supplied-camera rows are assisted and do not measure normal-mode acceptance.

Image-only production remains at the prior C12 result: 100 same-axis rows and 0/256 eligible endpoint assignments; no scoreable litematic exists. On synthetic V2, quota reduced starved pairs 26 to 0 and raised RANSAC 64 to 90, but strict-MNN positive support stayed 4/16 and invalid structural cases stayed 3/8. The run failed its support-gain and structural-safety gates.

What happened (max 250 chars): Fresh 36-case synthetic quota-v2 run completed once: starved pairs 26→0 and RANSAC 64→90, but positive support stayed 4/16 and 3/8 physical corners mixed planes. No-go; no state score changed.

What happens next (max 250 chars): Attribute losses by family and test a multi-plane contamination fix without relaxing gates. Any new synthetic trial needs fresh seeds and review; exact-state accuracy needs a valid default-mode litematic.

Forecast basis: 0.00 pp; loss attribution and geometry experiments are synthetic diagnostics with no scoreable production output.

---
## Update 2026-10-08 19:21:14 -05:00

Status: Global 8/8 cap revoked. Fresh 36-case synthetic quota-v2 calibration completed once and returned no-go: strict positive support unchanged at 4/16; structural corner invalidity 3/8. Production exact-state metrics unchanged; no valid image-only litematic.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest real-image result remains C12: six inputs, 100 same-axis rows, 0/256 eligible endpoint assignments. No image-only litematic or exact-state score.

What happened (max 250 chars): Quota eliminated eligible pair starvation (26 to 0) and raised RANSAC reach (64 to 90), but accepted models, true support and positive three-view support did not improve; 3/8 physical-corner cases had mixed-plane support.

What happens next (max 250 chars): Attribute family-level losses and design a falsifiable multi-plane contamination fix without gate relaxation. Any new synthetic run requires fresh seeds and review; exact-state accuracy still requires scoreable default-mode output.

Forecast basis: 0.00 pp: current work is synthetic diagnosis; no default-mode litematic or image-derived exact-state score exists.

---
## Update 2026-10-08 22:21:15 -05:00

Status: Global 8/8 cap revoked. Fresh 36-case synthetic quota-v2 calibration completed once and returned no-go: strict positive support unchanged at 4/16; structural corner invalidity 3/8. Production exact-state metrics unchanged; no valid image-only litematic.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest real-image result remains C12: six inputs, 100 same-axis rows, 0/256 eligible endpoint assignments. No image-only litematic or exact-state score.

What happened (max 250 chars): Quota eliminated eligible pair starvation (26 to 0) and raised RANSAC reach (64 to 90), but accepted models, true support and positive three-view support did not improve; 3/8 physical-corner cases had mixed-plane support.

What happens next (max 250 chars): Attribute family-level losses and design a falsifiable multi-plane contamination fix without gate relaxation. Any new synthetic run requires fresh seeds and review; exact-state accuracy still requires scoreable default-mode output.

Forecast basis: 0.00 pp: current work is synthetic diagnosis; no default-mode litematic or image-derived exact-state score exists.

---

---
## Update 2026-10-08 22:21:14 -05:00

Status: RUN-04 admitted both ordinary images but ended `geometry_unavailable / canonical_constraints_inconsistent`; no `.litematic` or image-only accuracy score. The source-consistent A/B/C falsifier stopped at its first unchanged local gate before matcher execution. A bounded provenance observer is implemented and awaiting independent review; no production policy has changed.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

These exact-state and candidate-ceiling values are historical supplied-camera/reference-assisted measurements from 2026-10-03. The ceiling is the fraction of visible truth cells whose exact state exists in the fixed pipeline's candidate set at aligned coordinates; it is not a theoretical maximum for future algorithms. No comparable image-only score exists, so image-only match/ceiling and point gain are N/A/unmeasured.

Image-only production: RUN-04 admitted 2/2 inputs. Geometry selection was exhaustively `canonical_constraints_inconsistent`; there is no output geometry or `.litematic`, so visible-geometry F1, image-only exact-state match and candidate ceiling cannot be computed.

What happened (max 250 chars): RUN-04 admitted 2/2 ordinary images but geometry ended canonical_constraints_inconsistent. The frozen A/B/C fixture stopped before matching at structural_share 0.1144 < 0.15. No litematic or image-only score.

What happens next (max 250 chars): The bounded observer is implemented and awaiting review. After acceptance, capture exact joined provenance from a separately authorized pinned run; then assess conversion, support and correspondence causes before any production policy change.

Forecast basis: 0.00 pp. Observer work is diagnostic only and has not yielded a scoreable default-mode `.litematic`; no image-only accuracy baseline exists.## Update 2026-10-09 01:21:15 -05:00

Status: RUN-04 admitted 2/2 ordinary images but geometry was exhaustively canonical_constraints_inconsistent; no .litematic or image-only score. The frozen source-consistent A/B/C falsifier stopped before matching at an unchanged structural-share gate. Bounded provenance observer implemented; independent review pending.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: RUN-04: two ordinary images admitted; selection exhaustively returned canonical_constraints_inconsistent. No geometry, .litematic, image-only F1/exact-state score, candidate ceiling or match/theoretical-maximum value can be computed.

What happened (max 250 chars): RUN-04 admitted both images but produced no geometry. The frozen A/B/C fixture stopped at structural_share 0.1144 < 0.15 before matching; no retry or tuning. Diagnostic observer implementation passed 139 scoped tests; no image-only score.

What happens next (max 250 chars): Complete Astra and DeepSeek observer review; after acceptance, authorize one source-pinned provenance run to examine exact witness measurements and candidate joins. Preserve caps/gates and advanced-only manual metadata; do not change production po...

Forecast basis: 0.00 pp: the observer is diagnostic only and no scoreable default-mode .litematic or image-only baseline exists.

---
## Update 2026-10-09 04:21:14 -05:00

Status: RUN-04 admitted 2/2 ordinary images but geometry was exhaustively canonical_constraints_inconsistent; no .litematic or image-only score. The frozen source-consistent A/B/C falsifier stopped before matching at an unchanged structural-share gate. Bounded provenance observer implemented; independent review pending.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: RUN-04: two ordinary images admitted; selection exhaustively returned canonical_constraints_inconsistent. No geometry, .litematic, image-only F1/exact-state score, candidate ceiling or match/theoretical-maximum value can be computed.

What happened (max 250 chars): RUN-04 admitted both images but produced no geometry. The frozen A/B/C fixture stopped at structural_share 0.1144 < 0.15 before matching; no retry or tuning. Diagnostic observer implementation passed 139 scoped tests; no image-only score.

What happens next (max 250 chars): Complete Astra and DeepSeek observer review; after acceptance, authorize one source-pinned provenance run to examine exact witness measurements and candidate joins. Preserve caps/gates and advanced-only manual metadata; do not change production po...

Forecast basis: 0.00 pp: the observer is diagnostic only and no scoreable default-mode .litematic or image-only baseline exists.

---
## Update 2026-10-09 07:21:14 -05:00

Status: RUN-04 admitted 2/2 ordinary images but geometry was exhaustively canonical_constraints_inconsistent; no .litematic or image-only score. The frozen source-consistent A/B/C falsifier stopped before matching at an unchanged structural-share gate. Bounded provenance observer implemented; independent review pending.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: RUN-04: two ordinary images admitted; selection exhaustively returned canonical_constraints_inconsistent. No geometry, .litematic, image-only F1/exact-state score, candidate ceiling or match/theoretical-maximum value can be computed.

What happened (max 250 chars): RUN-04 admitted both images but produced no geometry. The frozen A/B/C fixture stopped at structural_share 0.1144 < 0.15 before matching; no retry or tuning. Diagnostic observer implementation passed 139 scoped tests; no image-only score.

What happens next (max 250 chars): Complete Astra and DeepSeek observer review; after acceptance, authorize one source-pinned provenance run to examine exact witness measurements and candidate joins. Preserve caps/gates and advanced-only manual metadata; do not change production po...

Forecast basis: 0.00 pp: the observer is diagnostic only and no scoreable default-mode .litematic or image-only baseline exists.

---
## Update 2026-10-09 10:21:14 -05:00

Status: No new image-only exact-state score or geometry F1; acceptance gates unmet. Frame cap and synthetic adapter tests pass. Current graph-10 full-universe path is under source-bound integration review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: RUN-04: two ordinary images admitted; selection exhaustively returned canonical_constraints_inconsistent. No geometry, .litematic, image-only F1/exact-state score, candidate ceiling or match/theoretical-maximum value can be computed.

What happened (max 250 chars): Removed the 8-image cap and verified nine-/32-frame intake. Synthetic adapters pass, but add no accuracy. C10 neighborhood pooling was falsified; historical C13 evidence is stale for current graph-10.

What happens next (max 250 chars): Review Sol current graph-10 integration tests. Then run a same-pass kept-vs-full candidate study with identical CSP budgets; proceed to geometry only if selection is unique and all fixed gates pass.

Forecast basis: No scoreable normal-mode reconstruction exists yet; plumbing, synthetic tests, and selector evidence do not count as accuracy gain.

---
## Update 2026-10-09 11:02:34 -05:00

Status: No new image-only exact-state score or geometry F1; acceptance gates unmet. The cap and camera-accounting diagnostics are implemented, but the latest same-input run still produces no lattice geometry.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Graph10 independent-camera diagnostic: 1/4 admitted; 3 camera_estimate_unavailable. Admitted frame had 15 segments and 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. Zero proposals, no matcher/CSP, no .litematic or comparable score.

What happened (max 250 chars): Added scale-aware segment admission, independent cameras, compatible session records, and truthful accounting. Same-input run found 1/4 cameras admitted; 9/12 inliers blocked lattice. No score/export. Monitor missed exit/memory telemetry; no retry.

What happens next (max 250 chars): Record same-pass raw-per-scale and retained segment counts plus calibration rejection reasons. Fix and smoke-test the bounded hidden-process monitor before any new image run. Keep thresholds unchanged.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics cannot be computed. Next diagnostic improves failure attribution; forecast accuracy gain remains 0 pp until measured.

---
## Update 2026-10-09 11:18:41 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-09 13:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-09 16:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-09 19:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-09 22:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-10 01:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-10 04:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-10 07:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-10 10:21:14 -05:00

Status: No image-only exact-state score or geometry F1; release gates unmet. Same-input diagnostic identifies calibration support and the image-only camera observer now has same-pass telemetry under synthetic review.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest Graph10 diagnostic: 1/4 admitted; 3 camera abstentions. Admitted frame had 15 segments, 9/12 calibration inliers (axes 5,4,0), then insufficient_calibration_inliers. No proposals, matcher/CSP, geometry score, or .litematic. Run artifacts validated; process exit/memory not captured.

What happened (max 250 chars): Added scale-aware admission and independent-camera/session accounting; same-pass detector/inlier observer is frozen for review. Latest diagnostic: 1/4 cameras admitted, 9/12 inliers; no lattice proposals or output. No accuracy score.

What happens next (max 250 chars): Review and test same-pass telemetry, then review the synthetic-only process monitor. Use evidence to distinguish detector filtering from insufficient calibration support; keep thresholds fixed and do not run images until the runner is audited.

Forecast basis: No geometry or .litematic candidate exists, so exact-state and visible-geometry metrics remain uncomputable. Next task measures failure evidence, not accuracy; expected gain is 0 pp until scored output exists.

---
## Update 2026-10-10 13:21:15 -05:00

Status: Six-image actual-unit-weight full-delivery run reached image-derived endpoint selection but refused its bounded budget; no scoreable geometry or litematic, so release gates remain unmet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest exact-pinned full-delivery run: image-derived endpoint selector refused with endpoint_budget_refused; telemetry reports 4,584 candidates, 22 readings, 91 signatures, zero selected. Report observers incomplete downstream; no geometry, litematic, or accuracy score. Monitor: exit 1, 472.15s, 574.9 MiB peak, one launch, within stated run limits.

What happened (max 250 chars): Ran one pinned six-image unit-weight delivery: endpoint selection refused at budget (4,584 candidates, 22 readings, 91 signatures, zero selected). 575 MiB peak; 472 s. No litematic or accuracy score.

What happens next (max 250 chars): Sol is implementing an opt-in source-bound partial-observability selection certificate with explicit unknown offsets, exhaustive uniqueness and unchanged full-mode gates; Astra will review it before downstream geometry integration.

Forecast basis: No partial result reaches geometry/export yet, so image-only exact-state and geometry-F1 gains remain unmeasurable; next step is source-bound selection groundwork.

---
## Update 2026-10-10 15:41:15 -05:00

Status: Normal-mode six-view reconstruction still has no scoreable geometry or litematic; latest selector diagnostic refused at its bounded partial budget, and accepted AC-3 has been restored.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Latest bitset diagnostic: budget_refused at 524288 after 9 eligible assignments and 1523 branches; physical assignments/core=0, C1 not reached, no geometry/litematic. 6/6 input hashes unchanged. Production restored to accepted AC-3; image-only match remains unscored.

What happened (max 250 chars): The bitset solver run refused at 524,288 visits: 9 assignments/1,523 branches, versus AC-3 11/1,750. No geometry or litematic; restored accepted AC-3. Six input seals unchanged.

What happens next (max 250 chars): Profile and optimize the AC-3 image-path bottleneck using the refusal counters. Require exact differential tests and Astra review before one bounded rerun; keep caps and geometry/physics gates unchanged.

Forecast basis: No scoreable geometry exists, so exact-state and geometry-F1 point gains are unmeasurable. The next work targets search progress; no numeric match forecast is supported.

---
## Update 2026-10-10 16:15:24 -05:00

Status: Image-derived six-view run still has no scoreable geometry or litematic. Forced-arc AC3 regression passed but the sealed fixture budget-refused at 524,288 visits with fewer assignments than baseline and zero support-pair rejections/domain reductions. The 240-hour progress gate is unmet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: SOL-PARTIAL-FORCED-ARCS-IMAGE-DIAGNOSTIC-02: exit0; selection budget_refused at 524288. 8/1024 assignments, 1295 branches, 378 forced-pair checks, 0 pair rejections/domain reductions, 0 physical assignments/core; C1 not reached. No geometry/litematic/score. Runtime 447.24s, peak 625840128 B; monitor accounting/drain complete; six input seals unchanged.

What happened (max 250 chars): One same-cap run completed in 447s (626 MB): 8 assignments/1,295 branches, 378 forced checks, zero pair rejections/domain reductions, no physical assignment or geometry. No accuracy gain; six input seals matched.

What happens next (max 250 chars): Use actual-run counters to assess and likely remove/rework forced-arc overhead; target the first image-path rejection with a falsifiable source-bound fix before another image run. Keep caps and output gates fixed.

Forecast basis: There is no scoreable image-only geometry or exact-state score. Diagnostic activity and synthetic work savings do not establish match gain; report 0 pp until an artifact is independently scored.

---
## Update 2026-10-10 16:21:14 -05:00

Status: Image-derived six-view run still has no scoreable geometry or litematic. Forced-arc AC3 regression passed but the sealed fixture budget-refused at 524,288 visits with fewer assignments than baseline and zero support-pair rejections/domain reductions. The 240-hour progress gate is unmet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: SOL-PARTIAL-FORCED-ARCS-IMAGE-DIAGNOSTIC-02: exit0; selection budget_refused at 524288. 8/1024 assignments, 1295 branches, 378 forced-pair checks, 0 pair rejections/domain reductions, 0 physical assignments/core; C1 not reached. No geometry/litematic/score. Runtime 447.24s, peak 625840128 B; monitor accounting/drain complete; six input seals unchanged.

What happened (max 250 chars): One same-cap run completed in 447s (626 MB): 8 assignments/1,295 branches, 378 forced checks, zero pair rejections/domain reductions, no physical assignment or geometry. No accuracy gain; six input seals matched.

What happens next (max 250 chars): Use actual-run counters to assess and likely remove/rework forced-arc overhead; target the first image-path rejection with a falsifiable source-bound fix before another image run. Keep caps and output gates fixed.

Forecast basis: There is no scoreable image-only geometry or exact-state score. Diagnostic activity and synthetic work savings do not establish match gain; report 0 pp until an artifact is independently scored.

---
## Update 2026-10-10 19:21:14 -05:00

Status: Image-derived six-view run still has no scoreable geometry or litematic. Forced-arc AC3 regression passed but the sealed fixture budget-refused at 524,288 visits with fewer assignments than baseline and zero support-pair rejections/domain reductions. The 240-hour progress gate is unmet.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: SOL-PARTIAL-FORCED-ARCS-IMAGE-DIAGNOSTIC-02: exit0; selection budget_refused at 524288. 8/1024 assignments, 1295 branches, 378 forced-pair checks, 0 pair rejections/domain reductions, 0 physical assignments/core; C1 not reached. No geometry/litematic/score. Runtime 447.24s, peak 625840128 B; monitor accounting/drain complete; six input seals unchanged.

What happened (max 250 chars): One same-cap run completed in 447s (626 MB): 8 assignments/1,295 branches, 378 forced checks, zero pair rejections/domain reductions, no physical assignment or geometry. No accuracy gain; six input seals matched.

What happens next (max 250 chars): Use actual-run counters to assess and likely remove/rework forced-arc overhead; target the first image-path rejection with a falsifiable source-bound fix before another image run. Keep caps and output gates fixed.

Forecast basis: There is no scoreable image-only geometry or exact-state score. Diagnostic activity and synthetic work savings do not establish match gain; report 0 pp until an artifact is independently scored.

---
## Update 2026-10-10 22:21:15 -05:00

Status: Astra approved disabled packet readiness only. Exact packet, 198 source pins, 21,130 runtime pins, nine ordered inputs, limits, and external pre/post runtime checks reviewed. Launch flags false; hours unresolved. No evaluator run, 0 pp gain, no geometry or .litematic.

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Production image-only path: GUI-mask zero-fill fix implemented; 17 focused tests pass. No post-change image camera evaluation, geometry F1, exact-state score, or .litematic. Prior production run still budget-refused before geometry. Camera-basin attempt01 failed pre-freeze and cannot be retried.

What happened (max 250 chars): Completed packet prep and Astra exact-pin review for one camera-only measurement. All nine consumed inputs are sealed pre/post; source/runtime closure checks are external to the stock runner. No image, capture truth, or AB05 payload was accessed d...

What happens next (max 250 chars): Reconcile the actual tracked-hour total. If still below 240, make a separate lead launch decision for the exact reviewed packet; if at/above 240 without visible progress or a solvable image-specific gap, stop per user cutoff.

Forecast basis: No image-scored result exists. Camera measurement may resolve one camera hypothesis but cannot itself satisfy visible geometry/exact-state delivery gates.

---
## Update 2026-10-11 07:10:24 -05:00

Status: Astra approved refreshed source-closure successor for disabled readiness only. 198/198 source pins verify. Hours remain unresolved; launch flags false. No evaluation, image-derived geometry, .litematic, or score gain (0 pp).

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Production image-only path: GUI-mask zero-fill fix implemented; 17 focused tests pass. No post-change image camera evaluation, geometry F1, exact-state score, or .litematic. Prior production run still budget-refused before geometry. Camera-basin attempt01 failed pre-freeze and cannot be retried.

What happened (max 250 chars): Refreshed the stale packet-task pin; verifier passed all 198 source files and Astra approved disabled readiness. No image/truth payload or evaluator was accessed.

What happens next (max 250 chars): Reconcile current tracked hours. If below 240, refresh/review remaining runtime and input seals and make a separate lead launch decision; otherwise stop per cutoff.

Forecast basis: No image-scored result exists. The one-shot camera evaluation remains disabled and cannot establish geometry or exact-state delivery.

---
## Update 2026-10-11 07:21:14 -05:00

Status: Astra approved refreshed source-closure successor for disabled readiness only. 198/198 source pins verify. Hours remain unresolved; launch flags false. No evaluation, image-derived geometry, .litematic, or score gain (0 pp).

| Project | Exact-state match | Current candidate ceiling | Match / ceiling | Point gain since prior update | Expected gain by next update |
|---|---:|---:|---:|---:|---:|
| Template 2 supplied-camera | 71.96% (480/667) | 84.11% (561/667) | 85.56% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera default | 17.44% (350/2007) | 19.58% (393/2007) | 89.06% | 0.00 pp | 0.00 pp |
| Template 6 supplied-camera no-pruning ablation | 25.46% (511/2007) | 39.26% (788/2007) | 64.85% | 0.00 pp | 0.00 pp |
| Template 3 frozen subset | N/A | N/A | N/A | N/A | N/A |
| Templates 4-5 terrain | N/A | N/A | N/A | N/A | N/A |

Ceiling definition: fraction of all visible truth cells whose exact state exists in the current candidate set at the aligned coordinate. This is a fixed-pipeline ranker ceiling, not a bound on future end-to-end algorithms.

Image-only production: Production image-only path: GUI-mask zero-fill fix implemented; 17 focused tests pass. No post-change image camera evaluation, geometry F1, exact-state score, or .litematic. Prior production run still budget-refused before geometry. Camera-basin attempt01 failed pre-freeze and cannot be retried.

What happened (max 250 chars): Refreshed the stale packet-task pin; verifier passed all 198 source files and Astra approved disabled readiness. No image/truth payload or evaluator was accessed.

What happens next (max 250 chars): Reconcile current tracked hours. If below 240, refresh/review remaining runtime and input seals and make a separate lead launch decision; otherwise stop per cutoff.

Forecast basis: No image-scored result exists. The one-shot camera evaluation remains disabled and cannot establish geometry or exact-state delivery.

---
