# Fresh VGGT developer validation — COMPLETE, consumed one-shot

Task `completion-c1-postfix-validation-2026-10-07`, completion-plan developer attempt 5/8. Phase 1 and its narrow receipt-role correction completed on 2026-10-06 approximately 21:55–22:02 CDT, about eight aggregate agent-minutes. The lead's explicit 22:09 CDT one-shot GO authorized Phase 2, executed exactly once around 22:10 CDT. The fresh developer-only run exited 0, **COMPLETE**, with all required artifacts validated; its authorization is consumed. **STOP for lead review; no retry or follow-on work.** Original C1 remains BLOCKED/consumed. No reconstruction accuracy, normal-mode validation, scoring, scale, C2, capture, training, integration, or promotion claim.

## Exact prepared binding

Fixed capture root: `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/`.

| Ordered PNG | Fixed SHA256 |
| --- | --- |
| `view-canal-town-architectural-component-bbox-00.png` | `48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c` |
| `view-canal-town-architectural-component-bbox-01.png` | `8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696` |
| `view-canal-town-architectural-component-bbox-02.png` | `2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95` |
| `view-canal-town-architectural-component-bbox-03.png` | `e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9` |

These pins come from the task's lead-verified provenance. During Phase 1 this developer did not open/hash/decode screenshots or read the index. Fresh exclusive output: `exports/diagnostics/vggt-c1-fresh-canal-town-framed-2026-10-07/`. Output absence was confirmed after Phase 1 and again at 22:10:10 CDT immediately before the sole authorized runtime. Phase 2 created this exclusive output and read only the four exact PNG pixel inputs; all runtime hashes matched.

The receipt's structured `input_role` explicitly states `run=developer_runtime_validation`, `split=train`, `advanced_only=true`, `normal_mode_validation=false`, and `accuracy_claim=false`. A narrow AST assertion checked the literal receipt field without invoking main during Phase 1. The real terminal receipt retains these restrictions.

Binding compares normalized lexical paths to fixed paths, then rejects resolved redirection of inputs/output. It rejects original C1 paths/output, alternate roots, reordered/duplicate/short groups, captures 04–07, sidecar substitutions, child/other output paths, and redirected exact paths. Tests simulate redirection with an injected resolver, creating no links or filesystem artifacts. Runtime output reservation uses exclusive directory creation and exclusive receipt creation.

Per the lead's Phase 1 clarification, prepared runtime now verifies only the exact existing cached checkpoint, offline; old download, partial file, cache-directory creation, and download-metadata writes were removed. Checkpoint absence/size/hash mismatch refuses without fallback. PNG hash verification precedes each pixel decode; fixed expected hashes plus runtime runner/adapter hashes are recorded before forward. Socket connections are denied. Audited file writes must resolve inside the fixed fresh output, including temporary files; TEMP/TMP are directed there. Additional disk telemetry counts the fresh output rather than existing immutable environment/cache bytes. Python3.12.10, CPU Torch2.9.1+cpu, torchvision0.24.1+cpu, source/revision/license pins, one process/one forward, 20-minute wall/60 CPU-minute/24GiB RSS/9GiB additional-disk caps, all numerical gates, noncertified estimate policy, and terminal telemetry remain unchanged. None of these prepared runtime branches were executed in Phase 1.

## Verification and source pins

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv-vggt\Scripts\python.exe' -B -m pytest -p no:cacheprovider tests/test_learned_geometry_adapter.py
```

**72 passed in 0.79 seconds.** Includes 13 new positive/hostile binding cases, one explicit receipt-role assertion, and all 58 existing numerical, artifact, receipt, and telemetry cases. Initial verification passed 69 cases and found two displaced prior telemetry-test assertions during insertion; placement was corrected before 71 passed. Following the lead's narrow role-label correction, the final 72-case suite passed. `py_compile.compile(..., doraise=True)` passed for both changed Python files, with `_write_atomic` intercepted and source-path `cfile`, producing no bytecode/cache writes. No broad suite or runtime workload was run.

| Source/report | Final SHA256 |
| --- | --- |
| `scripts/evaluation/run_learned_geometry_dev.py` | `d4e8efc4231f40bee3fc35243f0a373dd8f43333bce58f267af320aaab776450` |
| `tests/test_learned_geometry_adapter.py` | `1ddd8d118d4e1c1be996c9978c249832efdddee4f19f326bd2027bf28a667a8a` |
| Unchanged `src/mc_photogrammetry/reconstruction/learned_geometry_adapter.py` | `bac64581a01e89d8645590a71f86077b3d1467f4f58d9a474da73d59d5236afa` |
| `docs/reports/VGGT-C1-postfix-validation-2026-10-07.md` | Supplied separately in lead handoff to avoid self-reference. |

During Phase 1, exactly the runner, focused tests, and this report were mutated, within its three-file write scope. Unrelated dirty state was preserved. Phase 1 performed no runner-main/model/forward, checkpoint/image access, output creation, truth/metadata/sidecar/evaluator access, original C1 access/mutation, environment/cache change, download/install/network contact, capture/Minecraft/Prism/Java/UI/input/audio, system/driver action, cleanup, or delegation. Phase 2 report and Comms updates are separately authorized by the runtime handoff; task/run bookkeeping remains for the lead unless an exact run path is supplied.

## Phase 2 — sole authorized runtime result

Under the lead's 22:09 CDT GO, exactly one foreground CPU Python process ran the following command from `C:\Users\Likea\Documents\vscode projects\mc photogrammetry project`:

```powershell
& '.\.venv-vggt\Scripts\python.exe' -B 'scripts/evaluation/run_learned_geometry_dev.py' --images 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-00.png' 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-01.png' 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-02.png' 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-03.png' --output-dir 'exports/diagnostics/vggt-c1-fresh-canal-town-framed-2026-10-07'
```

The GO-pinned runner and adapter hashes above matched before launch and in the runtime receipt. Python3.12.10, Torch2.9.1+cpu and torchvision0.24.1+cpu were verified in-process; pinned code/revision/CC-BY-NC-4.0 private-research checkpoint remained unchanged. Checkpoint size 5,026,367,224 bytes and SHA256 `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` were verified only inside this authorized process, offline with no cache writes.

One four-view batch `(4, 3, 294, 518)` and one CPU camera/depth forward completed in **14.953 seconds**. Terminal receipt: **COMPLETE**, stage complete, four VALIDATED frames, all finite/domain checks passed, zero GPU forwards. Final telemetry: **31.281 seconds wall**, **142.515625 CPU-seconds (2.375 CPU-minutes)**, **10,380,234,752 bytes peak RSS (9.667 GiB)**, **40,962,444 bytes sampled fresh disk (39.065 MiB)**. All are below fixed 20-minute wall/60 CPU-minute/24GiB RSS/9GiB additional-disk caps. Disk sampling precedes the final receipt rewrite, so it is the recorded sample rather than a claim of exact final filesystem size.

Every frame preserves 152,292 valid-source pixels. Maximum rotation correction across four frames is `1.2180717590226114e-7`, below unchanged `3.0398382691038855e-6` policy. Worst raw true-inverse roundtrip, canonical source roundtrip, and canonical repository parity are respectively `1.1368683772161603e-12`, `2.2393709973121313e-12`, and `1.3258049636636544e-12` pixels; all pass unchanged **1e-5** measured gates. Raw-to-canonical correction deltas are separately `8.656298420980208e-6`, `7.219633392198162e-5`, `1.2277404716456712e-4`, and `1.9780806034552738e-4` pixels. They describe explicit normalization, not scene error; estimates remain noncertified and unused for acceptance.

The process validated all four raw NPZs, four canonical NPZs, four diagnostic NPZs and four reprojection PNGs before terminal COMPLETE. Dataset-open audit records exactly 16 opens, four per approved PNG (hash/header, size, upstream preprocessing, gallery), no other dataset path. No dataset metadata/truth/camera/plan/recipe/mask/queue/manifest/palette/reference/sidecar or original C1 payload was read. No network/download/install/cache mutation, retry/substitution, source edit during runtime, GPU/backend/driver/system change, Minecraft/Prism/Java/UI/input/audio/capture, or follow-on work occurred. Upstream emitted a FutureWarning for deprecated CUDA-autocast syntax with `enabled=False`; zero GPU forwards remained recorded. No execution deviation or refusal occurred. Console output was captured by the tool; no separate log files were generated.

### New output SHA256 manifest

All filenames below are under the exact fresh output only.

| File | SHA256 |
| --- | --- |
| `run.json` | `4e3a20f7c8d36b07c238fb13ca5166045f562aa83f428a7aa4391dfc314872f3` |
| `frame-00-raw.npz` | `d116549490f09fc097fa090edb42bd09298de71d3b46467567d525c4daa46290` |
| `frame-00-canonical.npz` | `ded9f98683e2a630e78b5b065c10e979da6b64bc7b5ab0338820d728953ceecf` |
| `frame-00-diagnostics.npz` | `81c869df3207ffce7386b21e0274547c6aee376bc5e31b1e7bb3f960483a48c6` |
| `frame-00-reprojection.png` | `df5517da57cba3447b01a15b3e2d9a65bcb4c780d06c7123970b4bbc6d0009be` |
| `frame-01-raw.npz` | `5ed2c74d7022ce92acd2c4cb9939a33afcfe33eb45991a95d03ba63592081af5` |
| `frame-01-canonical.npz` | `db998d292ca8047199541f1f3f710cf716e5542219101cfbe01f94998461a2f2` |
| `frame-01-diagnostics.npz` | `f0efeafbd3a5f58a5d2719026c67f10b95976151aa0e6a40e13558558658beb8` |
| `frame-01-reprojection.png` | `a46af09d55f7c7a076b2315b7229b74281b83c0c692229d34738bb08fce5d7a7` |
| `frame-02-raw.npz` | `4bfde97df557d18c104943b4be5deaf495e0e423dfc3618108e65cdd530b1320` |
| `frame-02-canonical.npz` | `475051a199f47c592ce2258741d52dc013a68485d31756e9c1a106302f8b5607` |
| `frame-02-diagnostics.npz` | `b1f6201ef70e03b5639be34eb5c5b0ccc2a8e7b8f4830d8fdd891b51fa23b939` |
| `frame-02-reprojection.png` | `2fa5d13528248be4b492a1de89c2222c0c0263ebe7b2b1ef5cc6e94844ee9bbf` |
| `frame-03-raw.npz` | `0eef9d57ab51a029782192edb3574b68447db0c08adbd7215188378438793b56` |
| `frame-03-canonical.npz` | `bf9ddd524e4c0ded3479bfb15570d32cac0ac7f807e9c4b78d33cd1eaf3a4635` |
| `frame-03-diagnostics.npz` | `df741cab57b8570fa75a45c7b8396067725b04d9b3a4a52681235883a590a370` |
| `frame-03-reprojection.png` | `6f6b6507477ec7aa5ce23f1972f84d7b7f4ac3a18b4b6e0112cc1a6822c41cb7` |

Independent post-run filename/size inventory confirms 17 files totaling 40,962,511 bytes; the 67-byte difference from sampled disk telemetry is the final receipt rewrite. COMPLETE establishes same-view numerical conversion, preprocessing and required-artifact publication on this distinct developer train input group only. Camera/depth accuracy, cross-view consistency, metric/block scale, occupancy, states, visible acceptance and release gates remain unknown. Original C1 is still BLOCKED/consumed. This fresh GO is now consumed; STOP.
