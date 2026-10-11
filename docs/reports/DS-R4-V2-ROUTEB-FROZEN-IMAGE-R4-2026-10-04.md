# DS-R4-V2-ROUTEB-FROZEN-IMAGE-R4 - frozen-image Route B diagnostic (completed once)

- Task: `AGENT_COMMS.md` 2026-10-04 LEAD -> DeepSeek `DS-R4-V2-ROUTEB-FROZEN-IMAGE-R4`.
- Owner: DeepSeek CLI worker (direct shell).
- Environment: `.venv\Scripts\python.exe` (Python 3.10.11), direct foreground invocation.
- Result: **status `complete`**, `row_count` **1928**, record SHA-256
  `006E8B6B16A312CDAA61A17B4336CE27C1ABAC590435645908EA9CFEF2E998EE`. No refusal.

This is a diagnostic run record. It reports that the accepted sparse-sidecar-page
index and overflow-guard state let the one-shot Route B chain reach a complete
capture and record. It makes **no accuracy, exact-match, promotion or milestone
claim**; reference/truth/evaluator data was never consulted.

## 1. Preflight (all pins matched before invocation)

| artifact | assigned SHA-256 | observed | bytes | match |
|---|---|---|---|---|
| `scripts/evaluation/run_route_b_diagnostic.py` | `050AC0123386333B55E7321FE9D121DCF9CAC3E2635097931D2869BA9D0E8EC4` | same | 50,755 | yes |
| `tests/test_route_b_diagnostic_runner.py` | `D9A595EA84886B645C95182472D6B2E34EC057C410F9AE47195FFA17B54F04F3` | same | 51,487 | yes |
| `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py` | `F5CEE3CFE9D0247069637992FB29D1F9A8533ABC9383AF841CFC9A726313CCD2` | same | 122,069 | yes |
| `tests/test_candidate_support_provenance_v2_preflight.py` | `F1895E5C40148C49C77247CC13E488F7691CCAEC36003619401DF5101A8DBD96` | same | 114,420 | yes |
| `templates 2/2026-09-02_13.22.14.png` | `EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46` | same | 1,609,624 | yes |
| `templates 2/2026-09-02_13.24.24.png` | `E2FD9FD6C83BF9005FE0973987DA9E46D213A4351EC91393CE578F8E6D3961AE` | same | 1,461,426 | yes |
| `templates 2/2026-09-02_13.24.33.png` | `35D9DEBB206C2505F65D3BE29183F4A4F3D2BF3F9C05644CE75570F48381D1FC` | same | 1,336,191 | yes |

Assigned root `.tmp-routeb-v2-frozen-r4-20261004` was **absent immediately before
the run** (`Test-Path` = False), so the root was fresh. `.venv\Scripts\python.exe`
existed; no bare-python attempt was made. All seven pins were re-checked after the
run and were unchanged.

## 2. Invocation and result

Exactly one invocation, foreground, with only the three ordered image paths and
the fresh root; no settings, manual metadata, camera/F3/direction/coordinate/mask,
reference/truth/evaluator/litematic/scene input, and no retry or tuning:

```
.venv\Scripts\python.exe scripts/evaluation/run_route_b_diagnostic.py ^
  --image "templates 2\2026-09-02_13.22.14.png" ^
  --image "templates 2\2026-09-02_13.24.24.png" ^
  --image "templates 2\2026-09-02_13.24.33.png" ^
  --root .tmp-routeb-v2-frozen-r4-20261004
```

- Exit code: **0** (success path; no `DiagnosticRefusal`).
- stdout (single JSON line):
  `{"record": "<abs>\.tmp-routeb-v2-frozen-r4-20261004\record.json", "record_file_sha256": "006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee", "row_count": 1928, "status": "complete"}`
- stderr: **empty**.
- Refusal field: **none emitted** - no `refusal.json` exists, and the runner's
  refused branch (`run_route_b_diagnostic.py:1107-1111`) did not execute.

## 3. Stage sequence (source-derived, not emitted or observed)

No per-stage runtime trace was retained: the CLI `main` prints only `status`,
`record`, `row_count` and `record_file_sha256`, it drops the `calls` list
returned by `execute_route_b_diagnostic`, and no call list or stage ledger is
persisted in `record.json`. The stage order below is therefore **derived from
`_run_single_pass` source** (`ChainRun.calls` is appended immediately before each
direct stage invocation, `_run_single_pass:447-504`), not emitted or observed by
this run: **`entry` -> `lattice` -> `signatures` -> `graph`**. On the source's
only-success path each stage must complete and each typed status gate must pass
(`_entry_refusal`, `_status_refusal("lattice"|"signatures")`,
`PlacementRefusal`/`trace` checks), and the complete-support sink must capture
exactly once (`len(captured) == 1`) with a `CompleteSupportCaptureV2`
(schema `mc-photogrammetry-complete-support-capture-v2`). The persisted evidence
is the published `complete` record with `row_count 1928` and its row/page counts
(Sections 4-7); it is consistent with that source path, but no emitted per-stage
runtime trace and no directly observed sink invocation count is claimed.

Post-pass validation is likewise **source-derived from the source's only-success
path plus the published record artifact** (the record exists, so the path that
publishes it rather than the rollback path must have been taken); this CLI run did
not retain a runtime trace of it:

1. `replay_capture(run.capture)` (`:837`) - module validator
   `candidate_support_provenance_v2.validate_complete_support_capture` plus the
   runner's exact `expected_row_count = 1928` gate.
2. `_cross_bind_ingress(...)` (`:844`) between the entry's ordered pins, the
   capture root's own ingress, the serialized top-level `ingress`, and the
   positional `inputs`.
3. Post-write replay inside `_write_record` (`:703-714`) - the on-disk
   `record.json` must re-parse to the exact canonical capture and the artifact
   digest must recompute.

The source publishes the record only when these checks pass and otherwise rolls
back to a refusal, so the published `complete` record is source-derived evidence
that this success path, including those checks, was taken.
(The script also contains a separate `validate_record(record_path)` replayer
(`:888-935`); it is reachable only through direct calls, and this CLI run never
invokes it, so no claim is made about it here.)

## 4. Root file list and artifact hashes

Root `.tmp-routeb-v2-frozen-r4-20261004` contains exactly two files:

| file | bytes | SHA-256 | mtime |
|---|---|---|---|
| `route_b.mcscan` | 9,319 | `EE7C56BB5BD922D42BD2AD782E0FDFF202AD9DBE61CD331B423087C5A6407057` | 2026-10-04T19:52:17.4605450-05:00 |
| `record.json` | 16,896,513 | `006E8B6B16A312CDAA61A17B4336CE27C1ABAC590435645908EA9CFEF2E998EE` | 2026-10-04T20:00:53.2234088-05:00 |

Verified absent: `record.json.partial`, `refusal.json`, `refusal.json.partial`,
`route_b.mcscan.tmp`, and `route_b_staging/`. The staging directory was removed
after publication.

`route_b.mcscan` is the staged schema-2 project published by the entry stage
(`schema_version: 2`, `name: route_b`, `minecraft_version: 1.21.1`) and carries
an empty `volume.blocks` array with an `image_derived_session` block. It is not a
geometry result and is not scored here.

## 5. Record identity (as written to `record.json`)

| field | value |
|---|---|
| `schema` | `mc-photogrammetry-route-b-diagnostic-record-v1` |
| `status` | `complete` |
| `expected_row_count` | 1928 (equals actual `row_count`) |
| `signature_witness_count` | 85 |
| `whole_universe_identity_sha256` | `a5ddc8a65c0414e99b6026b5c2d7f4770e942f495ac9ebd8f26e59368d9c0efc` |
| `input_match_identity_sha256` | `28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105` |
| `failed_assignment_sha256` | `ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd` |
| `record_sha256` (in-record content digest) | `6eddf7f2bb60c1bdeac053d86e592dcd3c5c1b4bdd91a056c960c6f8db8ba797` |
| `artifact_sha256` | `678475d64a95a2814b5cce6864cd55853cc8f8b5e45cdf8d1b86a7023d5d41af` |

Note the distinction: `record_sha256` is the in-record content digest of the
capture, while the file's own SHA-256 is
`006E8B6B16A312CDAA61A17B4336CE27C1ABAC590435645908EA9CFEF2E998EE` (reported by
the runner as `record_file_sha256`).

Ingress pins recorded by the entry itself (ordered `image-0000`, `image-0001`,
`image-0002`), each decoded 2560x1369x3 `|u1` by
`input.image.load_image:exif_transpose+convert('RGB')` with no crop:

| frame | byte SHA-256 | byte length | decoded pixel SHA-256 |
|---|---|---|---|
| image-0000 | `eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46` | 1,609,624 | `209064e8cc5b95cd8403fddafa6e64758ffd8e1cf6be41e58118ac73e386b9ff` |
| image-0001 | `e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae` | 1,461,426 | `72bbe0594f432863b6de5f7d262a576c78404b65e495a68ec40a0c791b72e63f` |
| image-0002 | `35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc` | 1,336,191 | `0696521b4aa97a15cbc90dee26888024e11b4c0f238326d4f36730ae443aecc7` |

The byte hashes equal the operator-side PNG pins in Section 1, and the input
order is preserved.

## 6. Stage digests

| stage | SHA-256 |
|---|---|
| `entry_sha256` | `8635ea2e896f80c6e5ab5b9a80ddf8ff8d75c095fb8ff02417d3baf253198d70` |
| `lattice_sha256` | `cd8e4368d084b9e9b494384e3cb099ceb2862a7b28f73c57881481068ee71b6d` |
| `signatures_sha256` | `3e5530b4af3b734d3431a987fac33b97b6484c8fb2f82e044433defb91b55ef2` |
| `graph_sha256` | `49e2ed39c92ab8e9ccd3e1f07ed0961a1eeabdbb07e4ac11d2cd33d6d98a0224` |
| `session_sha256` | `d9a59ff8c14aef4d63c646e6823383e83804529ed31af503caf482d32459a9a5` |
| `trace_sha256` | `ec89f851fc41ed22bb1fe2f8002c68a6484e6f77eb4e1151269345dcd584e9ac` |

## 7. Capture shape (evidence that the sparse-page path was exercised)

`record.capture` uses schema `mc-photogrammetry-complete-support-capture-v2`:

- root: `row_count 1928`, `status complete`, `sidecar_page_count 67`,
  `index_page_count 5`, `digest_page_count 6`,
  `declared_universe_cap 1928`, `universe_max_rows 1928`.
- sidecar pages: 67 pages covering **971 entries**; page-level
  `first_admitted_row_index` ranges `0` .. `1887`.
- the sparse case is real and validated: page 63 starts at admitted index `1803`
  and page 64 starts at `1839`, so several admitted indices are absent between
  sidecar entries, yet the capture still passed the strict one-to-one present-row
  binding. This is the behaviour Luna accepted in
  `DS-R4-V2-SPARSE-SIDECAR-PAGE-INDEX` plus the overflow guard; R4 is consistent
  with it.
- index pages: 5 pages, row counts `469, 467, 465, 466, 61` = 1,928.
- digest pages: 6 sections x 1 page, `digest_count 1928` each.

## 8. Timing and cleanup

- The project artifact `route_b.mcscan` was published at
  `2026-10-04T19:52:17.4605450-05:00`; the record was published at
  `2026-10-04T20:00:53.2234088-05:00`. The `route_b.mcscan`-to-`record.json` file
  mtime interval is therefore an artifact-publication delta of about 8m36s (it is
  the gap between those two published artifacts, not the run's elapsed time).
  Separately, the Python process start was independently observed at
  `2026-10-04T19:51:48-05:00`, so the observed process start to `record.json` mtime
  spans about 9m05s. Neither figure is an exact monotonic runtime or exit
  measurement: this shell wrapper does not surface the runner's internal
  `elapsed_seconds`, and process-start observation is wall-clock only. The
  invocation was run foreground and was awaited to completion in-session; no
  background or persistent process was started. Timing is reported for provenance
  only and no throughput claim is made.
- Cleanup: root holds exactly the two published files; all `.partial` names, the
  `refusal.json` name, and `route_b_staging/` are verified absent.

## 9. What this does and does not establish

Established:

- The accepted `candidate_support_provenance_v2` state (sparse page index fix +
  overflow guard, pinned above) carries the real three-image Route B chain through
  the complete-support capture, through the source-required strict replay
  validation (source-derived from the only-success path, Section 3), and to a
  published `complete` record with `row_count 1928`.
- The R3 failure path (`refusal.json` at the capture-status gate) did not recur in
  this single run.

Not established / not claimed:

- No accuracy, exact-match, visible-state, geometry or promotion claim. No truth,
  reference litematic, scene, camera, F3, direction, mask, coordinate or evaluator
  input was consulted, so no match claim is possible from this run.
- This is one run of one frozen input set. It does not prove determinism,
  generalisation to other screenshots, or absence of the R3 class under other
  inputs.
- `route_b.mcscan` carries no blocks (`volume.blocks: []`); this run is evidence
  about the diagnostic chain and its provenance, not about reconstructed
  geometry.

Stopped for lead/Luna review.

## 10. Direct-shell compliance

Per `docs/deepseek-direct-shell.patch.yml` / `docs/deepseek-worker.md`: no
junction or reparse creation/manipulation, no recursive deletion, no
encoded/obfuscated commands, no generated executables or installers, no
background or persistent processes, no Defender or system changes, no external
temp inspection, and no client automation. Production code was not edited; only
this report and the concise `AGENT_COMMS.md` result were written.
