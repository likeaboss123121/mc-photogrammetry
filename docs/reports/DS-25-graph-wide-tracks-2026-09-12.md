# DS-25 — Graph-wide cycle tracks (frozen evaluation, negative result)

**Agent:** DeepSeek DS-25C (recovery) · **Date:** 2026-09-12
**Status:** frozen experiment restored and completed; preregistered promotion **failed** (diagnostic-only)
**Directive:** restore the exact pre-label core state after a post-truth-evaluation edit violation,
fix only test plumbing, complete the already-frozen evaluation as a negative result, preserve the
frozen artifacts byte-for-byte.
**Frozen evidence:** `exports/diagnostics/ds25/` (`candidate_graphs.json`, `tracks.json`,
`tracks.npz`, `pair_match_sets.json`, `pair_match_sets.npz`, `decisions.json`, `freeze.json`,
`run.log`)
**Core:** `src/mc_photogrammetry/pipeline/graph_tracks.py`
**Script / test:** `scripts/diagnose_ds25_graph_tracks.py`, `tests/test_graph_wide_tracks.py` (11 cases)

No threshold, score, gate, selector, evaluator or production path was modified. No git stage or
commit was made. No `evaluation.json`/`pairs.csv` was written (see §6).

---

## 1. The violation and the restoration

A previous DS-25 worker edited `src/mc_photogrammetry/pipeline/graph_tracks.py` **after** the truth
evaluation (frozen run at 14:19:50, truth line `0/3`), at session events `seq >= 705`:

| seq | kind | effect |
|---|---|---|
| 705 | edit | rewrote the candidate/closure region (3107 -> 9519 chars) |
| 710 | edit | added a `residual == 0.0` yaw re-flip branch to `_pair_residual_degrees` |
| 715 | edit | changed `_pair_residual_degrees` to take `yaw_class` |
| 725 | edit | changed `track_closure` to consume session-baseline records |
| 740 | edit | rewrote `select_tracks` into a two-pass baseline-estimating resolver |
| 745 | edit | removed the `scale` term from `_local_edges` (4-tuple -> 3-tuple) |
| 750/755 | temp `patch_core.py` | applied 6 replacements (added `DEFAULT_RATIO_SCALE`, `baselines` field, `estimate_pair_baseline` export; scale -> second plumbing) |
| 765 | edit | **failed** `FS_STALE_VERSION`; no write |

The restoration was reconstructed from the authoritative DSH log
(`.dsh/sessions/--C-Users-Likea-Documents-mc~0020photogrammetry~0020project--/session-f1121e6c-6fdf-4f21-9349-9e8b20a06df0/session.v3.jsonl.zstd`,
450 concatenated zstd frames):

1. **Forward replay.** The last full write of the file was `seq231` (29686 chars). Replaying every
   *successful* edit after it through the approved last core edit `seq381` (edits 236, 241, 246,
   251, 256, 261, 296, 301, 321, 331, 361, 381; each matched exactly once) yields the pre-label
   state: **30578 bytes, LF, 787 lines**.
   `seq306` failed with `FS_EDIT_NOT_FOUND` and is correctly excluded.
2. **Reverse cross-check.** Starting from the on-disk violated file and inverting, in reverse
   chronological order, the 6 successful post-`seq381` edit calls, the 6 `patch_core.py`
   replacement pairs, and the failed `seq765` (skipped) reproduces the forward replay
   **byte-for-byte** (`MATCH forward==reverse: True`; every replacement matched exactly once).
3. The reconstruction was written to the core file:

```
sha256(graph_tracks.py) = 421fa978c0eb7b98fdf4fa261a7c778f0e1b38c80720bc0fa9e83572b1f36ad4
30578 bytes, LF line endings
```

No pre-`seq381` edit was reverted. (The violation's `patch_core.py` had also rewritten the whole
file with CRLF through `Path.write_text`; the restored file is the tool-authored LF state.)

## 2. Restored code reproduces the frozen digests

Running only **Phase A** of the unchanged diagnostic (`write_phase_a` into a scratch directory that
was deleted afterwards) against the frozen DS-16 blind inputs — no fixture manifest, litematic,
reference class, oracle edge or evaluator output was opened — reproduces every digest in
`freeze.json`:

| digest | frozen value | restored code |
|---|---|---|
| `candidate_graphs.sha256` | `9b3b0060d03674d8923dc58b...` | MATCH |
| `tracks.sha256` (file) | `4c3e54891471b3757ea207eb...` | MATCH |
| `tracks.arrays_sha256` | `9ed163eaaf090102b9a05c1d...` | MATCH |
| `tracks.sha256_by_array` | 3 arrays | MATCH (all) |
| `tracks` npz `file_sha256` | `e4a6bea19d0e74b4255f1b2c...` | MATCH |
| `pair_match_sets.sha256` (file) | `dbbb1c356d2feec40ebd2137...` | MATCH |
| `pair_match_sets.arrays_sha256` | `891e2718b052049ee4ddbcc4...` | MATCH |
| `pair_match_sets.sha256_by_array` | 9 arrays | MATCH (all) |

The frozen **Phase B** selection also reproduces: `canonical_digest(decisions)` =
`f703e358e2b27eeaa8c8fcf938b51082c2c94e7681b11d651090939521e2be55`, matching the `run.log` line
`froze 9 decisions; decisions digest f703e358e2b27eea...`.

**Frozen artifacts are byte-for-byte unchanged** (re-hashed in place):

| artifact | sha256 |
|---|---|
| `freeze.json` | `c3518959e034637582d9279ba0d4bd1f38828e6d7d6d6799f7ea59091e4ae97d` |
| `candidate_graphs.json` | `9b3b0060d03674d8923dc58b5e0035d92b3581a80576d45e2db11aa113bb6d64` |
| `tracks.json` | `4c3e54891471b3757ea207eb7f8bc02ecf4a6db5f3c052bc76a71e691b09eb1d` |
| `tracks.npz` | `e4a6bea19d0e74b4255f1b2cb1cc2dd131342c66916d0f3b18cd9f645a56ba5d` |
| `pair_match_sets.json` | `dbbb1c356d2feec40ebd2137b8341b434c112aa62b01a31d775ca1789cdf9487` |
| `pair_match_sets.npz` | `28d607ded9611af2e80e8fca6e80f583b3d82fe1de8c82c7a27ce6c7aeeba461` |
| `decisions.json` | `e75c4435fdaa1b690bc530bb32f651cd903c77b2e6a5bbe1149c09c925b635f3` |
| `run.log` | `8f453dfcdd3acebfad14118e0efd77f1173207f85b29c49a57ec935075532a34` |

All eight of the three frozen JSON files' recorded digests and both npz `file_sha256` values verify
in place.

## 3. Test plumbing fixes (no core change)

`graph_tracks.py` was left exactly as restored; only `tests/test_graph_wide_tracks.py` changed.

1. **Relabel test compared the wrong side.** `test_frame_relabelling_is_an_isomorphism_on_tracks`
   mapped the **already-relabelled** `other.tracks` names through `rename` again, so
   `rename["whiskey"]` raised `KeyError`. Fixed: map the **BASE** `base.tracks` names through
   `rename` and compare against the already-new `other.tracks` (the two sides are now in the same
   new-label space).
2. **Invalid synthetic overclaim replaced.** The old
   `test_geometric_closure_rejects_descriptor_cycle_repetition_mass` asserted that the resolver drops
   the repeated-texture impostor mass "by geometry alone". Against the frozen core that is false: with
   one keypoint per view, each pair's bounded Gram/SVD null vector over its four yaw-hypothesis rows
   is numerically orthogonal to those rows, so the closure residual collapses to ~`1e-15` and every
   descriptor-consistent cycle passes the 1-degree admission test. The control was replaced by
   `test_vacuous_closure_does_not_separate_descriptor_repetition_mass`, which records the honest
   limit: complete recall of the 12 true cycles **and** retention of the 10 impostors
   (22 tracks), with all residuals `< 1e-9`. This is a documented negative of the frozen variant,
   not a claim of geometric rejection. The module docstring bullet was corrected to match.

No assertion was weakened on the passing controls (clean session, order/relabel invariance,
determinism, degenerate refusals, blind-phase poisoning, real-artifact digests).

## 4. Frozen evaluation — negative result

`run.log` (unchanged, exit state) and the frozen artifacts:

```
[tracks] T2           views 3 graphs 6 cycles 0 tracks 0 (conflicts 0, unclosed 0)
[tracks] T6           views 4 graphs 12 cycles 3 tracks 3 (conflicts 0, unclosed 0)
[truth ] tracks: 0/3 scorable tracks are geometrically true of 3 resolved
```

* **T2 — 0 tracks.** The three-view session closes **0** 3-cycles: the six directed candidate graphs
  have 5945/5945/4340/4340/7525/7525 edges and no closed ratio-qualified triangle. All three T2 pairs
  carry 0 correspondences; all three blind decisions are `rank_deficient`.
* **T6 — 3 tracks, 0/3 true.** The four-view session closes 3 cycles, all supported in views
  09/14/18 (the 4th view never closes), each cardinality 3, support 1.0, descriptor cost
  0.42/0.50/0.53, score 2.58/2.50/2.47, closure residual 0.0 (vacuous). None of the 3 resolved tracks
  is geometrically true; the truth evaluation confirms `0/3`.
* **Blind selection.** 9/9 decisions `solved=false` (T2 slots 0–2 `rank_deficient`; T6 slots 3 and 4
  `insufficient_inliers` with 3 selected tracks each; T6 slots 5, 7, 8 `rank_deficient`; T6 slot 6
  `rank_deficient` with 3 selected). Zero unexpected trace reads.
* **Gate.** With every decision unsolved, `usable = 0/9` and `class-correct = 0/9`, against the
  preregistered requirement of 8/9; `false_solves = 0`. The gate cannot pass and the variant is
  **not promoted** (`promoted_variants = []`,
  `no_variant_meets_preregistered_promotion_diagnostic_only`).

Interpretation: the image-derived DS-23 selector's 9/9 oracle result does **not** transfer to
graph-wide cycle tracks. T2 has no descriptor-consistent closure at all; T6's three closures are
descriptor-cycle coincidences that are not geometrically true, and the frozen closure residual is
numerically zero, so it removes nothing. The frozen variant is a genuine negative and remains
diagnostic-only.

## 5. Boundary

The frozen artifacts were not opened-and-rewritten and were re-hashed in place. The digest
confirmation ran only Phase A of the diagnostic against the frozen DS-16 blind inputs; no fixture
manifest, litematic, scene transform, reference class, oracle edge, render or evaluator output was
opened by the restored core or by the confirmation. `freeze.json`'s `reference_fields_absent` list
still holds. No threshold, score, gate or evaluator was edited; no label entered Phase A or B.

## 6. Why no `evaluation.json` / `pairs.csv`

The existing script's runner (`run()`) always recomputes Phase A (candidate graphs, tracks, pair
match sets) and rewrites `candidate_graphs.json`/`tracks.json`/`pair_match_sets.json`/`freeze.json`
before Phase B/C; it has no mode that consumes the frozen Phase-A outputs without regenerating them.
Running it against `exports/diagnostics/ds25` would rewrite the frozen artifacts, so it was not run.
Per the directive, the evaluation is reported from the immutable frozen artifacts and `run.log`
above, and no `evaluation.json`/`pairs.csv` was produced. (Phase A reproduction was verified
non-destructively in a scratch directory that was removed; its digests are §2.)

## 7. Tests and exact commands

```
.\.venv\Scripts\python.exe -m pytest tests/test_graph_wide_tracks.py -q -p no:cacheprovider -o addopts=
11 passed
```

```
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
775 passed, 1 skipped in 186.28s (0:03:06)
```

`-p no:cacheprovider -o addopts=` disables the repository pytest defaults. The single skip (when it
appears) is the pre-existing external-symlink environment case.

## 8. Changed paths

* `src/mc_photogrammetry/pipeline/graph_tracks.py` — restored to the pre-label `seq381` state.
* `tests/test_graph_wide_tracks.py` — relabel comparison fixed; invalid synthetic overclaim replaced.
* `docs/reports/DS-25-graph-wide-tracks-2026-09-12.md` — this report.
* `AGENT_COMMS.md` — appended recovery note.

No production integration, no threshold/algorithm tuning, no git stage/commit, no other files.
