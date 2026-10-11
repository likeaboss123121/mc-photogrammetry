# SOL-DENSE-UPRIGHT-SIFT-AMBIGUITY-01 ? terminal one-shot resource refusal

The exact separately authorized command ran once and stopped with **worker_failure1 / descriptor_site_limit_before_compute**, exit1. This is **inconclusive**, not an experimental failure of dense upright SIFT: at least one frame's planned unique eligible descriptor sites exceeded1024, and no descriptor computation occurred. The log does not retain the limiting frame or its exact site count. No tuning, cap change, retry, test, source edit or additional image access followed.

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluation/probe_dense_upright_sift_ambiguity.py
```

## Exact first stop and available evidence

Worker.log630B records the call at `probe_dense_upright_sift_ambiguity.py:622` to `plan_sites(frames,masks)`, then line212 raising `Refusal("descriptor_site_limit_before_compute")`. The prescribed planning guard checks both frame unique-site counts after exact center-mask/usable membership and left-site deduplication and before either descriptor compute. The recorded successful preflight and source call chain establish both native frames were bound/decoded and the three exact masks loaded before this stop. This is a source-path inference; decoded arrays/site plans were not persisted.

No `ambiguity.json` or `completion.json` exists. Only `exports/diagnostics/dense-upright-sift-ambiguity-2026-10-09/worker.log` was created. Thus no sealed supply counts, descriptor inventory, discrimination evidence, directed funnels, comparison-work counters or common-four witness status exists. `found`, `exhaustive_no_witness` and `budget_inconclusive` are unavailable because the search was never reached; the experimental outcome is resource-refused/inconclusive. Do not report no witness or insufficient feature supply from this refusal.

The source path has zero descriptor/detector calls, zero cross-frame comparison arithmetic and zero common-witness nodes before the stop. These are source-path inferences, not persisted counter closure. Ordinary screenshot decodes were exactly the two authorized native frame binds; no later/alternate/post-command screenshot access occurred.

## Preflight and unchanged artifacts

Authorization receipt SHA256 `2EB21E22EF8DF2DC7DEB0112A5F47F6F1D429DAA8A05A5CEFD84B6145BA8AE59` matched. All124non-image source/code/history/task/auth pins matched before the command and afterward. Both fixed encoded PNG hashes/byte lengths were checked before invocation, with no post-command image hash/read/decode by the operator. Exact diagnostic outer seal, three endpoint IDs/header+packed/full-native digests, both native ingress shapes/dtypes/frame IDs, treatment and limit bindings matched. Runtime output was absent before launch. All prior accepted/failure/synthetic receipts remain intact; code/test hashes are unchanged.

The frozen treatment remains dense upright SIFT on unmodified RGB2GRAY, with exact source/usable center admission and baseline pad0 (no added erosion). **Descriptor neighborhoods may cross source-region and usable-mask boundaries**; this preprocessing differs from production detect_features zeroing. No actual descriptor evidence was produced under that treatment, and no physical association, geometry/accuracy or appearance-ambiguity conclusion is justified.

## Resource record and limits

Operator wall 1.453000s, below180s; timeoutfalse. Final output630B, below8MiB. Supervisor source uses the accepted180s/RSS/artifact monitors and2GiB kernel child-tree committed-memory limit. Its sampling peaks were not persisted on failure, so peak RSS/job committed/CPU and exact eligible site counts remain unknown. No completion means resource peaks or site/membership counter closure cannot be independently replayed. The terminal guard is the1024unique-site-per-frame ceiling; there is no timeout or memory/output termination reported.

No cap was altered. Maximum2frames/2.88Mpixels/11200grid centers/16800memberships/1024sites per frame/1048576comparisons/4096common tuples/32768determinants/180s/2GiB/4MiB evidence/8MiB artifacts remain frozen. Code execution is consumed; any further work requires a separate scoped task and independent review.

## SHA256

| Artifact | SHA256 |
|---|---|
| `scripts/evaluation/probe_dense_upright_sift_ambiguity.py` | `23AFB6C5ADFFEB198548E0C7D6778DE6CD35FFA76AD97072254D18A661B037C4` |
| `tests/test_dense_upright_sift_ambiguity.py` | `58BF7C49B866AF83BDB78FC1700F198330BE56E8130024BA8EE4B8C300C42136` |
| `exports/diagnostics/dense-upright-sift-ambiguity-2026-10-09/worker.log` | `2E6B3DC11B4D0135529838D77BD9BCE7C6F54746A0DF5FB2D979A2CA4542FA26` |
| `agent swarm instantiation template/runs/SOL-DENSE-UPRIGHT-SIFT-AMBIGUITY-01.runtime.preflight.json` | `22B1D08AEB0D30B3D49F6DB5FCBB1CCD51C92DE47B0DF11A9F4CB908C86BEB96` |
| `agent swarm instantiation template/runs/SOL-DENSE-UPRIGHT-SIFT-AMBIGUITY-01.runtime.command.json` | `21D48F65CA9E7C172F3106430B544CB48E3921DF04C4A41BEEF1DD951AF9D529` |
| `agent swarm instantiation template/runs/SOL-DENSE-UPRIGHT-SIFT-AMBIGUITY-01.runtime.stdout.txt` | `24985E004DB42D80E799592AE2F1C6DE4BD9B03D39EFAAFFB211275459FD5A31` |

Stop for independent lead review. Preserve the log and terminal refusal; no retry or follow-on image work.
