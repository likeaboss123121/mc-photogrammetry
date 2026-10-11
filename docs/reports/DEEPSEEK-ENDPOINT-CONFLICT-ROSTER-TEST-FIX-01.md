# DEEPSEEK-ENDPOINT-CONFLICT-ROSTER-TEST-FIX-01

**Owner:** DeepSeek (Sol-equivalent CLI worker). **Lead:** Luna. **Date:** 2026-10-09.
**Parent failure:** `SOL-ENDPOINT-CONFLICT-NATIVE-ROSTER-01` (8 passed / 1 failed / 38 unreached; exit 1).

## Scope executed

Repair of one invalid synthetic substitution control only:

- File: `tests/test_endpoint_conflict_native_roster.py`
- Test: `test_decoded_digest_substitution_refuses_with_one_decode`
- Before: the substitute decoder returned `np.zeros((64, 64, 3), np.uint8)`, byte-identical to the all-zero
  frozen source frame, so `decoded_pixel_sha256` matched the ingress pin and `Stop("decoded_ingress_mismatch")`
  was correctly never raised.
- After: the substitute returns the same `uint8` `64x64x3` array with one channel byte set
  (`pixels[0, 0, 0] = 1`), a deterministic single-byte difference from the frozen source.
  The `pytest.raises(subject.Stop, match="decoded_ingress_mismatch")` and `assert len(calls) == 1`
  assertions are unchanged.

No other line, file, driver, receipt, pin, or production module was touched.

## Hashes

| Item | Before | After |
| --- | --- | --- |
| `tests/test_endpoint_conflict_native_roster.py` | `E20AF276C190BFBD4A51B9D6535C13E4602D9475B4A317CB93E2076A828CA238` | `28F2994FB25AB72F94BDA8D473B7F7A9AEE60356182E2367DF5AA740DB65C73B` |
| `scripts/evaluation/acquire_endpoint_conflict_native_roster.py` | `0A204799D07A022903DB017C3307C3966C0C1B617669B24689D9048A7680BB35` | `0A204799D07A022903DB017C3307C3966C0C1B617669B24689D9048A7680BB35` (unchanged) |

The new test hash was frozen before the sole command. The driver hash is unchanged in preflight and
postflight.

## Sole focused command (run exactly once)

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q -x --tb=short tests/test_endpoint_conflict_native_roster.py
```

Result: **47 passed, 0 failed, 0 errors, 0 unreached; exit code 0** (1.71 s pytest, 2.05 s wall).
All 47 preregistered synthetic controls now execute, including the previously unreached 38.

Raw output is preserved at
`agent swarm instantiation template/runs/DEEPSEEK-ENDPOINT-CONFLICT-ROSTER-TEST-FIX-01.stdout.txt`.

## Preservation and boundaries

- The prior failed run artifacts (`SOL-ENDPOINT-CONFLICT-NATIVE-ROSTER-01.*`) and the lead review receipt
  were re-hashed before and after; all match. `8 passed / 1 failed / 38 unreached` remains the recorded
  terminal result of that run.
- `exports/diagnostics/endpoint-conflict-native-roster-2026-10-09` remains absent.
- Zero ordinary-image reads, decodes, hashes, helper acquisitions, detector runs, supervisor invocations,
  camera/projective fits, checkers, truth/scoring, `.litematic`, or output creations.
- No retry: the focused command was invoked exactly once.

## Explicit limitation for lead review

Editing the frozen test changed its code pin. The existing lead review receipt
(`RESULT-LEAD-ENDPOINT-CONFLICT-NATIVE-ROSTER-DRIVER-REVIEW-01-2026-10-09.json`) still carries the old test
pin `E20AF276…`; a live `authorize()` against that receipt would therefore refuse with `source_pin_mismatch`
until the lead independently re-issues the review with the new pin. This worker did not edit that receipt.

A green suite validates synthetic driver controls only. It does not accept the driver and does not authorize
any live acquisition or screenshot read/decode. Stop for lead review.
