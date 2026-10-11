# C1 review corrections — synthetic evidence, awaiting Astra review

Developer attempt 3, scoped to `completion-c1-review-corrections-2026-10-07.md` and Astra's 2026-10-06 21:27 CDT R1–R4 rejection. Implementation and verification completed on 2026-10-06, approximately 21:31–21:42 CDT, below the two aggregate agent-hour cap. STOP for independent review. The real C1 run remains **BLOCKED and consumed**; its discarded predictions, residual, and failing branch remain unknown. No real-run pass, reconstruction accuracy, metric/block scale, C2, capture GO, integration, or promotion is established.

## Changes

- **R1:** All raw frames are written before optional rotation diagnostics. Each determinant/orthogonality/condition diagnostic independently catches overflow, nonfinite results, and linear-algebra failure, records `null` plus an explicit reason, and retains strict `allow_nan=False` JSON. Canonicalization refusals carry finite diagnostics. Zero, singular, extreme finite rotations and injected SVD failure preserve both raw frames and a serialized terminal BLOCKED receipt. Nonfinite derived source intrinsics refuse, and overflowed translation norms are unavailable diagnostics.
- **R2:** Renamed the partial canonical roundoff estimate and explicitly documented omissions: rounded inverse affine, source-intrinsic construction/application, homogeneous cancellation, repository centre/subtraction, and Euler arithmetic. Both canonical and raw-to-canonical pixel estimates are labelled noncertified and are never acceptance thresholds. Unavailable estimates are `null` with reasons in receipts and NaN sentinels in optional NPZ estimate arrays. The perspective lemma is evaluated in ordinary float64 and is not an outward certificate. Raw true-inverse roundtrip, canonical source roundtrip, and repository parity retain their measured **1e-5 pixel** gates. The previously accepted quaternion input-precision correction envelope and rotation residual tolerance are unchanged; measured SVD/orthogonality checks do not become rigorous rounding certificates. Raw-to-canonical correction displacement remains a separate diagnostic, not a scene-error or acceptance claim.
- **R3:** Main, helper, and watchdog share one `ReceiptPublisher` and one synchronized JSON publication path. Deep-copied snapshots and terminal priority prevent stale progress, COMPLETE, or generic failure from replacing a resource refusal. Resource sampling does not mutate the main/helper state from the monitor thread. Terminal ownership is retained even when receipt writing fails; monitor termination executes in `finally`, independently of receipt success. Atomic replacement alone is not presented as durability or fsync assurance.
- **R4:** The numeric helper publishes only GEOMETRY_VALIDATED. The outer completion gate builds every gallery, then validates every frame's raw, canonical, diagnostics, and gallery artifact before publishing COMPLETE. Missing/empty outputs, nonfinite required numeric payloads, invalid gallery decoding, and downstream exceptions refuse. A prior resource refusal also dominates outer completion.

## Checks

Final focused command, with bytecode and pytest cache writes disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv-vggt\Scripts\python.exe' -B -m pytest -p no:cacheprovider tests/test_learned_geometry_adapter.py
```

**56 passed in 0.71 seconds.** Existing independent analytic one-ULP and Rodrigues controls remain. Added zero/singular/overflow/SVD refusal evidence, Decimal references for cancellation-sensitive off-centre K and crop-offset/upscale coordinates, translation/depth controls, deliberate underestimated-estimate plus 2e-5 pixel corruption, competing receipt writers, sticky resource status, receipt-I/O failure with injected termination, gallery construction/validation failure, actual in-memory NPZ validation, and outer completion after resource refusal. The concurrency check uses two bounded, joined test threads and injected events/writers; no background process or model workload is started. All test artifacts are in memory.

`py_compile.compile(..., doraise=True)` passed for the three changed Python files. `_write_atomic` was intercepted and `cfile` set to each source path, so compilation generated no bytecode files or cache directories. Earlier checks found two test/stage expectations, then one remaining stage expectation; these were corrected within this same implementation attempt before the final passing run. No extra workload or broad test suite was run.

Task-local mutations are exactly the following four declared files. They are pre-existing untracked source/test paths plus this new report, so Git's tracked-file diff does not represent the task baseline. Scoped mutation history and source hashes identify this change; the pre-existing unrelated dirty tree was preserved. No task/run/Comms status file is in this assignment's write scope; lead must record the review-ready status and Comms evidence.

| File | Final SHA256 |
| --- | --- |
| `src/mc_photogrammetry/reconstruction/learned_geometry_adapter.py` | `bac64581a01e89d8645590a71f86077b3d1467f4f58d9a474da73d59d5236afa` |
| `scripts/evaluation/run_learned_geometry_dev.py` | `19a2727f528d652050a568e9fef5465b16585bc3bd9790c8dd1e71b3b1465593` |
| `tests/test_learned_geometry_adapter.py` | `88f94b87d4c37ca9d37696daf6272b3011f864f29c8747ac53ded8a07c3c08f0` |
| `docs/reports/VGGT-C1-review-corrections-2026-10-07.md` | Report hash supplied separately in the developer handoff to avoid a self-reference. |

## Preservation boundary

Per the task and lead's 21:30 CDT clarification, the lead independently verified the consumed C1 receipt, C1 report, and prior remediation report before this assignment against their recorded hashes. This developer did **not** reopen or rehash those artifacts or checkpoint weights. The prior checkpoint immutability record stands; no fresh checkpoint byte-verification claim is made. Recorded receipt: `c4e12492756b0e64e1ef9b027d82c27fd04d072d4c4518f4f36519b1594d5cb2`; C1 report: `f233a1016b25a4050dba7e3fec48fedea05b765d181a5c2b8e0caf1cf5a2d358`; prior remediation report: `08f3377621c6ada7dca522e21664b4d3fb980568b1904cf5af997682b0b7ec92`.

No screenshot, dataset metadata/truth/mask/camera manifest, evaluator payload, checkpoint, consumed output/log/report, or previous remediation report was opened. No model load/forward, runner-main launch, retry, download/install/environment/cache mutation, capture, Minecraft/Prism/Java/UI/input/mouse/audio, driver/system action, cleanup, or delegation occurred. The new Astra R1–R4 guidance was present in Comms and is the authority for this correction. Synthetic checks validate these source boundaries only; they do not recover or validate the consumed C1 result.
