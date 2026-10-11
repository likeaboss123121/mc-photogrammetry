# Image-only Route B stage-loss run - one pinned run

Task ID: `image-only-route-b-stage-loss-run-2026-10-04`.
Owner: DeepSeek (bounded run/report). Run date: 2026-10-04. Status: **one authorized run complete; stopped for Luna verification and Astra read-only acceptance.**

This report records exactly one ordinary-image Route B run of the accepted A2 evaluation
runner on the pinned three-view Template 2 fixture, its terminal outcome, the required
stage ledger, immutable artifact hashes and the limits of the measurement. It is a
diagnostic stage ledger, not a promotion run, tuning sweep, or accuracy estimate.

## 1. Result summary

- Exit code `0`; wall time `185.597 s`; stderr empty.
- Production ran **before** any evaluator file was opened (runner order: production,
  then pin manifest, then truth), and no camera/F3/direction/coordinate/bounds/truth/mask
  input reached production inference.
- The run is a **geometry-stage abstention**: terminal
  `stage=geometry`, `status=geometry_unavailable`, `reason=g_unavailable`,
  `detail=plane_integer_residual`.
- Frontier: `entry` reached, `lattice` reached/assembled, `geometry` reached but
  unavailable, `output` not reached. Policy reach `3/4` planned checkpoints (`0.75`),
  refusal `1/4` (`0.25`).
- **No numeric G/O/P/C/E accuracy metric is produced by this run.** Every accuracy row in
  the artifact is typed unavailable, never a measured zero. The only measured evidence is
  the frontier/policy reach and the terminal typed refusal.
- The produced `.mcscan` carries an empty volume and no `.litematic` was written; export
  is typed unavailable (`output_sha256` empty, no output record).

## 2. Preflight verification (before the run)

The preflight hashed the accepted runner/module/tests and three source screenshots,
checked expected fixture pin values, and confirmed the fresh artifact paths were absent.
It did not open the reference, scene, camera manifest, or mask files; those evaluator
inputs were opened by the runner only after production returned.

| Input | SHA-256 |
|---|---|
| runner `scripts/evaluation/report_image_only_attribution.py` | `EE976B6B9E79A8535A412F9430FDF5E0CA8D1C16CFFEA3DF342E130EECF22805` |
| module `src/mc_photogrammetry/evaluation_image_only_attribution.py` | `4FED6AA8E22DD881E69063A33B56DF2480B1700BFE62CC7D86D69EE54A8FE2A3` |
| tests `tests/test_evaluation_image_only_attribution.py` | `3B94D058DF7A896550C5265D4B037700EA310B6DAF21048609ED620CA0611CC5` |
| `templates 2/2026-09-02_13.22.14.png` | `eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46` |
| `templates 2/2026-09-02_13.24.24.png` | `e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae` |
| `templates 2/2026-09-02_13.24.33.png` | `35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc` |
| Expected reference pin from the reviewed manifest | `519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416` (value only; reference file not opened in preflight) |

The three screenshot hashes equal the reviewed `template-2` pins. The artifact root `.tmp-routeb-stage-loss-20261004` and its
project/report/stdout/stderr targets were confirmed absent before the run. The accepted
A2 report records the three mask-domain pins as matching; the runner re-verified them at
runtime (`mask_pins_match=true`, `mask_pinned_domains_checked=3`).

## 2.1 Post-production evaluator input integrity

The following evaluator-file hashes were computed after production returned, not during
preflight. The runner then loaded reference/scene/camera/mask data for scoring; the
reference file hash matched the pin value checked during preflight.

| Evaluator input | SHA-256 |
|---|---|
| `templates 2/Template 2.litematic` | `519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416` |
| `templates 2/scene.json` | `9620608A2544E9460240C2EFD0FF83DA354D3FFCBEE8C01E61E2D9A05B627881` |
| `templates 2/cameras.json` | `4F743E4FC478430F608E0E879F73CD4A3DC9CE50C726841D2DA897BE78BE6126` |
| `docs/reports/evaluator-fixture-pins.json` | `C564743A7AB63009E2105BAAE3D591338D6D3B8A694B0633DA3B0EBA6F019FD1` |
| `docs/evaluation_masks/templates-2/2026-09-02_13.22.14.json` | `2B0E18CFC26D225C6BB5CB1043792CDCC252983215B5FB7FD09480E88622850E` |
| `docs/evaluation_masks/templates-2/2026-09-02_13.24.24.json` | `068C7FE623E66D770E77AF98538916EAFE34A7F4E7B0F3D7080ED5FA5D5687C7` |
| `docs/evaluation_masks/templates-2/2026-09-02_13.24.33.json` | `212A5D7E49B66D14EED26A829F75440E15B20B8281B433907C7F4319FD830DD2` |

## 3. Exact command and settings

Executed once, foreground, from the repository root:

```
.\.venv\Scripts\python.exe scripts\evaluation\report_image_only_attribution.py ^
  --route route-b --fixture "templates 2" ^
  --sources "templates 2\2026-09-02_13.22.14.png" ^
            "templates 2\2026-09-02_13.24.24.png" ^
            "templates 2\2026-09-02_13.24.33.png" ^
  --reference "templates 2\Template 2.litematic" --scene "templates 2\scene.json" ^
  --cameras "templates 2\cameras.json" ^
  --masks "docs\evaluation_masks\templates-2" ^
  --pins docs\reports\evaluator-fixture-pins.json --fixture-role template-2 ^
  --project .tmp-routeb-stage-loss-20261004\route-b.mcscan ^
  --report .tmp-routeb-stage-loss-20261004\attribution.json
```

stdout and stderr were redirected separately to
`.tmp-routeb-stage-loss-20261004\stdout.txt` and
`.tmp-routeb-stage-loss-20261004\stderr.txt` (Windows PowerShell 5.1 native redirection
writes UTF-16LE for stdout; the decoded text is byte-identical to `attribution.json`,
verified below).

Settings are the accepted defaults: pinned mask specifications, `--require-pins` implied
by `--pins`, no `--cli-parity`, no `--allow-pin-mismatch`, no threshold, palette, radius,
matcher or solver override, no `--workdir`. No retry, sweep or second run was performed.

The project target did not exist before the run; `run_route_b` refuses an existing target,
and the report target was written once to a path confirmed absent. The runner is the
evaluation-only A2 artifact and no production/source/test file was edited.

## 4. Immutable run artifacts

| Artifact | Bytes | SHA-256 |
|---|---|---|
| `.tmp-routeb-stage-loss-20261004/attribution.json` | 16,816 | `5CD4F15A62F6FB505DBC6A162643658EA48C3C9B738742D8BD99663D52DCF19B` |
| `.tmp-routeb-stage-loss-20261004/route-b.mcscan` | 7,254 | `FD508D9F0E20846314D3E48723617D3771151CC496457F8A1096B828C6EE30EC` |
| `.tmp-routeb-stage-loss-20261004/stdout.txt` (UTF-16LE capture) | 33,634 | `95A98CC3F6A204660AFA5A5B039FD51F8CB36BF295AAA78B7AA7C31B4F315EBC` |
| `.tmp-routeb-stage-loss-20261004/stderr.txt` (empty) | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` |

stdout capture integrity: decoding `stdout.txt` from UTF-16LE to UTF-8 bytes yields
SHA-256 `5cd4f15a62f6fb505dbc6a162643658ea48c3c9b738742d8bd99663d52dcf19b`, which equals
`attribution.json` byte-for-byte; the JSON objects are equal.

After the run the runner, module and tests still hash to their preflight values, and the
three screenshots and reference still hash to their pins, so the run did not mutate any
source or protected input.

Internal content addresses reported by the artifact (the artifact cannot embed its own
file hash): `frontier_sha256 cfe6ee399ae2da1155c4853ac0c0b53c0182ff154c8506b7a7337b18b6f6702f`,
`attribution_sha256 7f720c43449fb7307e28c3190cbdf877f3b8f89e922509278136b1918ba3f633`,
`result_sha256 7fe7472f51c39331f50b720366605ee85217e98c7057e469a9aed26f30b6ee02`,
`entry_sha256 83c6fef5483ec4f67577d6a6a543a0d3a22205b7c3225d3c09cea1671a78a3aa`,
`lattice_sha256 6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4`,
`geometry_sha256 0b9680a168dddd24a56e05d5b7ccf832a9ef63c248c69a1070ce2a84c5be4f2d`,
`output_sha256 ""` (no output record),
`session_sha256 3b6d15bf2a74b672d88d5a7a40f1bf5298d00ede770735309694d97770be9fb7`,
`envelope_sha256 30dcd1566865e9e6e4ad77520dc0f42fa5dd663d06bfe0727e707756f74d95f8`,
`envelope_replays_result true`.

## 5. Production/evaluator separation evidence

- The A2 runner orders production inference before `load_pins` (pin manifest) and before
  `load_frozen_truth`; this run's artifact was produced under that order.
- Production received only the three screenshot paths and the project target. The
  evaluator-only reference, scene, camera manifest, pinned mask specifications and pins
  were opened only after `derive_image_output_from_paths` returned.
- The persisted schema-3 envelope replays the returned result
  (`envelope_replays_result=true`; envelope stage/status/reason and entry/lattice/geometry
  digests agree with the outcome), so carried records are bound to the same production
  outcome the evaluator scored.
- The runner's source-level call boundary and production < pin-loader < truth-loader
  sentinel establish the intended evaluator separation: production receives source paths
  and a project target before pins/truth load. The absent tail records
  (`placement_cells=0`, `volume_cells=0`, `promoted_cells=0`, `proposal_cells=0`) are
  consistent with the geometry stop, but are not by themselves proof of separation.

## 6. Visible truth domain (evaluator-only, loaded after production)

| Field | Value |
|---|---|
| Visible reference cells `V` | 621 |
| Reference cells | 852 |
| Pixel-counted cells | 621 |
| Minimum usable pixels | 1 |
| Frames | 3 (2560x1369 each) |
| Mask origin | `tracked_json_specification` |
| Mask pins match | `true` (3 domains checked) |
| Reference pin matches | `true` |
| Truth source | `evaluator_raycast_pinned_reference_cameras_frozen_masks` |

The primary denominator for G/C/E is the fixed visible domain `|V| = 621`; the evaluator
pair control uses its own usable-pixel mask and is never divided by `|V|`.

## 7. Required stage ledger

Universes and conditional denominators are kept separate. "Unavailable" means the
artifact carries no bound record and the metric is `null`; it is never reported as `0`.

| Stage | Checkpoint | Numerator / denominator | Availability | First refusal / typed reason | Provenance |
|---|---|---|---|---|---|
| Camera/bounds | entry | 3 pairs; `match_count_total=424` (context only) | Reached; no numeric camera accuracy or scale metric | entry reached (`record_state=ok`), 2 pairs `solved` / 1 `refused`; P metric typed unavailable | `entry_sha256 83c6fef5...`, `record_digest 83c6fef5...` |
| G geometry | lattice | `null` / `|V|=621` | Reached / assembled, metric unavailable | lattice assembled; no accuracy metric | `lattice_sha256 6f4cba95...` |
| G geometry | placed | `null` / `|V|=621` | geometry checkpoint reached, record absent | `g_unavailable`; terminal detail `plane_integer_residual` | `geometry_sha256 0b9680a1...` |
| G geometry | volume_cell | `null` / `|V|=621` | Not reached (output absent) | `g_unavailable` | `production_records.volume_cells=0` |
| O ownership | (none) | - | Unavailable; no rows | no bound evaluator surface-ownership artifact | empty O section, `frozen_frame_pixels` |
| P correspondence | entry | `null` / `frozen_pixel_pair_identities` | Reached, metric unavailable | `entry_match_identities_not_retained_or_digest_unbound` | retained `RelativeCameraEvidence` (counts + `matches_sha256`), no pair identities |
| C candidates | palette | `null` / `|V|=621` | Not reached | `palette_unknown` | output record absent |
| C candidates | candidate_k | `null` / `|V|=621` | Not reached | `palette_unknown` | output record absent |
| E exact state | raw_winner | `null` / `|V|=621` | Not reached | `record_absent` | output record absent |
| E exact state | promoted | `null` / `|V|=621` | Metric unavailable; substage reach unknown (`checkpoint_reached=null`) | `record_absent` | no promotion record |
| E exact state | terminal | `null` / `|V|=621` | Metric unavailable; substage reach unknown (`checkpoint_reached=null`) | `record_absent` | no terminal state record |
| Export | decoded `.litematic` | - | Unavailable | no output record; `output_sha256` empty | no export file in the run root |

### 7.1 Frontier and terminal

| Frontier checkpoint | Kind | Universe | Reachable | Record state | Metric | Typed reason / detail |
|---|---|---|---|---|---|---|
| entry | P | `frozen_pixel_pair_identities` | `true` | `ok` | `null` | `frames=3` |
| lattice | G | `frozen_reference_cells` | `true` | `assembled` | `null` | lattice readings sealed; V1/V2/V3/M5 geometry unavailable |
| geometry | G | `frozen_reference_cells` | `true` | `unavailable` | `null` | `g_unavailable` / `plane_integer_residual` |
| output | E | `frozen_reference_cells` | `false` | `unavailable` | `null` | not reached |

- `frontier_stop`: kind `E`, stage `output`, reason `output_record_absent`.
- Terminal: `stage=geometry`, `status=geometry_unavailable`, `reason=g_unavailable`,
  `detail=plane_integer_residual`; `terminal_policy_agrees=true`.
- Policy rates: planned `4`, reached `3`, completion `3/4=0.75`, refusal `1/4=0.25`,
  terminal stage `geometry`. These are policy rates, never accuracy.
- One transition is serialized: `lattice -> geometry`, universe
  `frozen_reference_cells`, all gain/loss fields `null` (not numeric), transition SHA-256
  `dcbf434d379c636f8ee62ec2b6f9948b9f201a80dfe40c2e3965b0547c0418f8`.

### 7.2 Retained entry match evidence (P context only, not a P metric)

The Route B entry retained match digests but not pixel/keypoint-pair identities, so no
bit-identical joint match set can be bound to the accepted digest and the P entry row
stays typed unavailable. Retained context:

| Pair | Status | `match_count` | `mutual_survivors` | `matches_sha256` |
|---|---|---|---|---|
| image-0000 / image-0001 | solved | 106 | 17 | `06f93a8bb031b44d5f2034597d3c439570b34bdbff6381ea679c0274979b853e` |
| image-0000 / image-0002 | refused | 176 | 7 | `082a4a61911bdd73ed175d3f6dacc8006cc081fec002cce28d340bb115ab43a4` |
| image-0001 / image-0002 | solved | 142 | 7 | `10a553277a3a31c86d6695f9f9985114c635f2e3a0722c8d2821e2ca51e7ba25` |

These entry records retain pair match counts and digests, but not joint identities. The
separate evaluator control's truth-pair coverage/keypoint counts are described below;
they are not entry connectivity and are not an achieved P metric.

### 7.3 Evaluator pair control (separate, excluded)

`evaluator_pair_control` ran `evaluate_scene/select_mutual_matches`, a different matcher
from the Route B entry's `joint_matching.match_and_solve_pair`. It is labeled
`role=control`, `excluded_from_route_b_attribution=true`, and is never attached to the
entry row: returned 33, truth pairs 232, truth pairs returned 0, positive 0 over 3 pairs.
Its truth-pair coverage is 3 frames, 3 pairs, 3,773 keypoints (1,973 with truth). These
are evaluator-control counts from the alternate matcher only. It is not a Route B P
metric and was not used to fill P.

## 8. First-stop signal and next falsifiable one-factor test

**Measured loss: none.** This run provides no numeric stage-loss measurement. It records a single
typed frontier outcome: the run reaches entry and assembles the lattice, then the
**geometry stage abstains with `plane_integer_residual`**; output is the first
unreached checkpoint. Because every G/O/P/C/E accuracy row is typed unavailable, no
largest numeric or percentage loss was measured. The geometry abstention is the current
production blocker/first-stop signal, not a loss score, and must not be restated as
`0/621` or any other zero.

**Next falsifiable one-factor test (not authorized by this task).** An evaluation-only,
single-factor diagnostic on this exact pinned three-view fixture that exposes the
already-designed first-failed geometry predicate and per-axis integer resultants at the
lattice -> geometry transition, with every production gate, threshold, palette, matcher
and solver frozen. It would test one hypothesis: whether the stop is an axis
resultant/coverage predicate under sparse image-only support. It must not lower or retune
a gate, and it requires a separate authorization and its own single pinned run.

## 9. Limitations

- One fixture, one three-view run, one machine. No general performance claim.
- No end-to-end, geometry, ownership, correspondence, candidate or exact-state accuracy
  was measured; unavailable rows are not zeros and must not be pooled with Route A or
  with assisted-camera diagnostics.
- P remains unavailable because the entry retains match digests but not joint identities;
  entry connectivity is not a truth-pair count.
- O remains unavailable because the frozen A2 truth assets provide no bound
  (cell, slot, frame) surface-ownership artifact.
- Export is unavailable: the run stopped before any output record, so no `.litematic`
  exists to decode or round-trip.
- No alignment was measurable (`alignment=null`) because this run carried no candidate
  coordinates; no geometry/state candidate row could be scored. `metric_scale_known=False`
  remains an independent scale limitation. Unknown metric scale alone would not prohibit
  scoring a carried candidate under the permitted lattice/integer alignment family.
- `plane_integer_residual` is a terminal typed reason from the accepted production path;
  this artifact does not localize which internal predicate failed, which is why the next
  test above is limited to instrumentation.
- The stdout capture is UTF-16LE because of Windows PowerShell 5.1 native redirection; its
  decoded text is byte-identical to `attribution.json` (verified in section 4).

## 10. Checks performed in this task

- Preflight hashes of the accepted runner/module/tests and three screenshots, plus pin
  value/path checks (section 2). Evaluator content hashes are explicitly post-production
  (section 2.1).
- Confirmed the artifact root and project/report/stdout/stderr targets were absent before
  the run.
- Exactly one foreground run; exit `0`; stderr empty; wall `185.597 s` (section 3).
- Read-only parse of the produced `attribution.json` to extract the stage ledger
  (sections 6-7); no second run, matcher, solver or workload.
- Post-run re-hash of the runner/module/tests, screenshots and reference: unchanged.
- stdout capture decoded and compared byte-for-byte with `attribution.json`.

No production/source/test edit, threshold or gate change, palette/matcher/solver change,
alternate matcher substitution, additional run, sweep, retry, training, dataset mutation,
launcher/client automation, install/download, background job, recursive deletion,
junction/reparse operation, encoded command or commit was performed.
