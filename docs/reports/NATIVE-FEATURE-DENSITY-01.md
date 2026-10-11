# SOL-NATIVE-FEATURE-DENSITY-01 ? terminal synthetic failure

The sole preregistered synthetic module run stopped at its first failure: **48 passed, 1 failed, 1 not reached of 50 frozen cases**, pytest 1.98 seconds, operator wall 2.359000 seconds, exit 1. No fix or retry was performed. The probe is pending independent Astra/lead review; this result does not authorize an image run.

## Scope and implementation

Only `scripts/evaluation/probe_native_feature_density.py` and `tests/test_native_feature_density.py` were added as code. Production and the accepted 512-feature roster helper/driver were unchanged. The standalone probe requires a fresh absent authorization binding its final code/test/input/diagnostic/baseline hashes before runtime I/O. Its fixed runtime contract permits one image-0000 decode and one existing SIFT detection at max_features=2048, with all other accepted correspondence settings, native resolution and GUI usable mask unchanged. It reads the exact historical left plane2/plane3 mask bundle, retains framed ingress digest versus raw array digest semantics, and never performs cross-view matching or physical inference.

The inventory has a distinct 2048-feature identity and records raw/retained/duplicate/boundary counts, exact-mask eligible classes/coordinates and 4x4 bins. General-position testing is an exact binary-coordinate boolean with bounded work. Limits remain one frame, 2048 raw returned features including boundary drops, 4096 mask checks, zero cross-view comparisons, 180 seconds, 2 GiB job committed, 4 MiB evidence and 8 MiB artifacts. The proposed runtime supervisor reuses accepted job/publisher budget primitives. Windows job enforcement and an ordinary runtime acquisition were not exercised by this synthetic task.

## Frozen run and first failure

The code and 50 controls were sealed in `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.freeze.json` before the single command:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q -x --tb=short tests/test_native_feature_density.py
```

Failure: `tests/test_native_feature_density.py:350`, in `test_evidence_bound_before_publish_and_no_clobber_roundtrip`, at `assert subject.sealed_source(path.read_bytes()) == result`.

The exact 4 MiB/one-byte-over evidence checks and canonical publisher call at line349 were reached before this assertion failed. JSON readback converts tuple-valued runtime settings to lists; `sealed_source` uses ordinary JSON parsing (probe lines94?106), whereas the inventory retains runtime settings in memory. The log identifies only the inventory as unequal. Source inspection supports a tuple/list equality mismatch as the likely explanation; it does not independently establish that this is the sole differing value. Canonical digest and record validation passed during publication, but this failed assertion remains unresolved. No production regression is demonstrated by this result.

The subsequent no-clobber assertions (test lines351?352) were not reached. The final `test_preserved_budget_infrastructure_and_no_density_runtime_invocation` was not executed because `-x` stopped the module. No additional test collection, imports, runtime command or rerun followed the failure.

## Passing controls and practical limits

The 48 passing cases covered absent/invalid/positive synthetic authorization, code/mask/baseline binding without ordinary pixels, exact max_features-only settings, one genuine SIFT detection on a synthetic zero PNG with matching/acquisition APIs forbidden, exact mask and native-frame corruption refusals, distinct inventory/provenance/bins, duplicates/coincident coordinates, mask holes/outside points, 2048 retained rows and exactly 4096 charged membership checks, raw overflow before support work, boundary counts, invalid feature tables, exact general-position comparison against all 336 subsets of sizes4?6 from a 3x3 grid plus adversarial collinear controls, and output seal/provenance/counter corruptions.

The successful full-cap synthetic inventory closed raw/retained counts and requested=charged=4096, unprocessed=0, cross_view=0; no truncation was used. That evidence is synthetic and says nothing about actual row197 feature supply. The genuine SIFT zero-image control had zero features/membership checks and an insufficient supply result. Runtime metadata, budgets and evidence semantics still require independent source review.

## Integrity and no-run boundary

All **86 read-only pins** from `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.preflight.json` matched before freeze and after the run. No source/test edits occurred after freeze. Compile-to-memory and trailing-whitespace checks passed for both owned files after the run. No screenshot was accessed, hashed or decoded; synthetic PNGs only were created in pytest temporary paths. Accepted sealed artifacts and masks were read as provenance during preflight, unchanged.

Fresh runtime authorization remains absent: `agent swarm instantiation template/runs/RESULT-LEAD-NATIVE-FEATURE-DENSITY-AUTH-01-2026-10-09.json`.
Fresh runtime output remains absent: `exports/diagnostics/native-feature-density-2026-10-09`.
No worker, supervisor, ordinary feature extraction, cross-view comparison, camera/geometry operation, truth/checker/score or production integration was run.

## SHA256

| Artifact | SHA256 |
|---|---|
| Probe | `FF2A8F647362C2CC101706FE0A660732681D1792F37D13848360B3BC4E84A467` |
| Synthetic test | `67FB18827A275063F9C71D28067669191A8AA44B86633794A6C58A3B78B26D05` |
| `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.preflight.json` | `998247FDB531B3E28AC9C2A53A1695AF0AE01DBFD3FCA196639E1B1AABBC1FCC` |
| `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.freeze.json` | `BD1726D2C553FBE65947A9E73F531D9D156921DE2AD14F70EA9399CACC344684` |
| `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.test-run.json` | `569F192062771AA7CF5108B1BBB1DC03383B561768214C1525E89AF97834599F` |
| `agent swarm instantiation template/runs/SOL-NATIVE-FEATURE-DENSITY-01.stdout.txt` | `403B5A89027ED4878077EA5315E8AC192F2D0210867844B26D5C037186256822` |

Terminal stop: preserve the frozen failure; any fixture repair or new run needs a separately scoped task.
