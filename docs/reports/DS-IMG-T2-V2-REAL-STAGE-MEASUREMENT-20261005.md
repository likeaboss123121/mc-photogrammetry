# DS-IMG-T2-V2-REAL-STAGE-MEASUREMENT - ownership checkpoint

Implementation task: `DS-IMG-PHASEA-OWNERSHIP-20261005`. Date: 2026-10-05.

**HOLD: partial ownership implementation, pending Luna/Astra independent review. Reconstruction release stays disabled.** Public/default/CLI/direct worker paths refuse before reconstruction; the private synthetic marker is a test convention, not an authentication boundary. No real image stage, evaluator, source footprint, geometry or exact-state score exists. Phase B remains `phase_b_alignment_unavailable`; source-footprint/downstream metrics remain unavailable/null.

The capped Windows store writer now retains its CREATE_NEW birth handle through bounded writes, flush and no-replace handle rename. Native FILE_ID_INFO binds identity. A runtime-only typed ProjectWriteLease drops WRITE rights without a reopen gap, supports bounded immutable reads/full typed producer replay and transfers the same retained object through ImageDerivedEntryResult into Route B. The lease is excluded from the canonical entry payload and digest. Missing, released, wrong-path or wrong-cap leases refuse before lattice work. Closing a lease releases handles; discarding it disposes only the owned file object.

Native capture publication and cleanup use retained born-file handles: no-replace SetFileInformationByHandle rename, or disposition on the same handle. Partial, project and journal reads/writes stay capped. Journal update moves the prior retained checkpoint to a no-replace backup before publishing the new born file. A racing foreign destination causes refusal while preserving foreign bytes and the previous checkpoint. This is not crash-atomic replacement, and a failed update can retain the backup name. Ordinary legacy/non-Windows runner cleanup retains its weaker sequential identity-check behavior and is outside the native capture contract.

Freeze validation retains every terminal file and seal handle across bounded reads, hashes and semantic joins, borrowing invocation birth handles when available. All checks use those immutable byte snapshots; the supplied freeze must exactly match its retained seal. Caller-root, trusted caps, record-derived scorecard, selective project/session/ingress, same-pass stage/checkpoint hashes and terminal refusal checks remain enforced. Initial/final inventory observations reject observed extras and partials. Full typed canonical frozen project/session replay is still a re-enable gate; full producer replay already uses the retained project lease.

**Directory namespace remains unsealed.** Retained directory/ancestor handles block rename of those held objects, including an EMPTY root; their compatible sharing allows directory WRITE opens and child-name creation. The native regression explicitly demonstrates both allowed operations, and `namespace_exclusive=false` is recorded. Reparse/type checks and inventory observations do not atomically exclude mutation or transient names. Root/staging creation before handle acquisition has no proven directory birth contract; empty-directory pathname cleanup remains race-sensitive. Full-path publication is not a root-relative namespace capability. Denying directory WRITE sharing blocked the existing publisher's rename path, so no such exclusion is claimed. The lead explicitly approved truthful partial HOLD instead of a false namespace-safety claim.

Deterministic native Event barriers, with bounded foreground thread joins and no timing sleeps/junctions, race same-size/same-byte rename/replacement/write against partial cleanup, rollback, store temporary cleanup, project lease handoff, journal update and freeze validation. Held-file mutation and held-root rename are blocked; racing foreign final names are preserved; observed extra child names refuse freeze validation and survive. A direct WriteFile on the sealed retained lease handle fails with access denied and zero bytes. The JSON maps each ownership requirement to exact test names. These proofs concern held file objects and observed inventory, not hostile-writer namespace sealing or reparse exclusion.

The previous native controls evidence remains: ten parametrized harmless-child cases in six tests cover suspended creation, assignment-before-resume, restricted inherited handles, explicit optimized-Python membership failures, wall timeout, combined stdout/stderr overflow, aggregate descendant Job committed-memory allocation refusal and descendant teardown. Wrong-specific Job exits 22 and missing-specific Job exits 21 with zero work/stream bytes. Each combined-stream ordering yields stdout=32, stderr=33, aggregate=65 at cap=64 while neither stream alone crosses cap. Root/descendant each verify the same Job and request 60% of 128 MiB; aggregate memory notification 10 occurs without process-limit notification 9. No fixed native child mode runs reconstruction.

Memory proof remains configured committed-memory limits/allocation-refusal evidence, not a hard observed peak or RSS ceiling. Preserve the earlier development observation `PeakJobMemoryUsed=184000512` at configured `134217728`; its cause is not inferred. Production resource limits are unchanged. The harmless native supervisor is not integrated with the production reconstruction worker. Fresh-root external abort is covered, but safe finalized resource-terminal freeze on an already populated root remains a gate.

Final current-pin nine-module suite: **466 passed in 61.23s**, exit 0, including 18 added ownership cases and all prior harmless native controls. The JSON records the exact command/environment, requirement-to-test mapping and historical prior results. Bytecode/cache writes were disabled; fixtures use pytest-managed tmp_path. No ad-hoc scripts/probes/logs, real inference/evaluator or real-input image/client/JAR/world/reference/network access occurred. Synthetic entry fixtures use only generated pytest-managed PNGs.

Remaining release gates:

1. Exclusive directory namespace/birth identity, atomic reparse exclusion and safe empty-directory cleanup are unproved; transient hostile child-name mutation remains possible.
2. Journal backup/new-file publication is not crash-atomic replacement.
3. Full typed canonical frozen project/session replay and populated-root abort finalization remain incomplete.
4. Real reconstruction worker integration and hard observed memory peak/RSS containment remain unproved.
5. Legacy/non-Windows cleanup is not covered by the native birth-file lease contract.

Owned/dependency source and test pins (the progress evaluator itself is unchanged):

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| src/mc_photogrammetry/evaluation_template_match_progress.py | 54527 | `2759f15245ac651122acd8e0204a9be170ad667ca0f06d93197567d27dc86c4f` |
| scripts/evaluation/run_template_match_progress.py | 89909 | `6523a8e94a940697f3e648a199170962e61208f30565f2191c92bc0550d04032` |
| scripts/evaluation/run_route_b_diagnostic.py | 88979 | `59ae30b79e028fe3d3aa0d93c79d4a8a59e274fe582b9d91b8ea8c0f6e6246d7` |
| src/mc_photogrammetry/pipeline/image_derived_entry.py | 48434 | `9926312ae58d862ad95ef8754d0c49e1ff8d2e86e45a51674a84046208f5cd02` |
| src/mc_photogrammetry/project/store.py | 30318 | `bd984ac51079c4b01144814db35f38e14efe6d2f64787e9c03b6177f3d559559` |
| tests/test_evaluation_template_match_progress.py | 80588 | `081ccdda85725a2637bcc552a5f9e2eb1a9cdbdf4e33d31aa1ce858c2bbb69fe` |
| tests/test_route_b_diagnostic_runner.py | 82915 | `bf061384f99b6a5d976bab46ad06533195238319e05cfea375cd53f3ae66aeca` |
| tests/test_image_derived_entry.py | 41006 | `b6e1e6fcda88f55c5f0defd6a3eb5f69f4c9dbe4bcc8016ce87b8719d49f3ac9` |
| tests/test_project_store.py | 23959 | `f3d0d8dfcb4bbbde652a28751526b2d0e1d7166bc5e37d65f58c8fbb88f00d32` |

All four matching/support dependency pins in the companion JSON were rechecked unchanged. No matcher, filter, numerical solver, schema or image-derived camera policy was changed. Historical R4 artifacts were not reread/migrated/rerun/edited. AGENT_HANDOFF.md was not edited by this task. Stop for Luna/Astra review; synthetic/native success does not authorize capture or imply accuracy/containment acceptance.
