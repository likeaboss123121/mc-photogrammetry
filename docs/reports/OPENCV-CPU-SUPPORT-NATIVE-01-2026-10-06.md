# OPENCV-CPU-SUPPORT-NATIVE-01 frozen fake-only harness

State: FROZEN_REVISED / PURE_TESTS_PASS / NATIVE_NOT_STARTED. Astra R1/R2/R3 remediation is complete pending independent re-review. Stop for Astra technical and Luna scope review. No native cv2/NumPy import/call, image/truth/evaluator, observer/capture, production/gate/.venv/build/temp changes. A-SCALE remains held. Only the three assigned paths and append-only Comms were written; pytest uses tmp_path for guard artifacts. No native evidence or run marker was created.

The harness implements bounded orchestration around the accepted schema-4 symbolic evaluator, with fake provider injection and an actual stdlib-only synthetic subprocess mode. `FakeProvider` emits deterministic zero bytes; these are deliberately not OpenCV interpolation outputs. Every receipt/result is `SYNTHETIC_ONLY`; `PASS` or native-mode substitutions refuse. The default native entry unconditionally returns `unsupported_attestor` before marker consumption or native entry. The real attestor allowlist is empty.

## Attestation boundary and remaining blocker

Preflight requires provider CAPABILITY, not a preexisting smoke/trace receipt. Capability must cover the exact pinned import/DLL/source/config identity, requested dtype-specific entry, declared branch and atomic call-boundary FP/selected-entry observation. FakeProvider's single call generates fresh before/after evidence and output together. Verification checks exact call metadata/request digest, a fresh parent-issued 256-bit attempt nonce in transport/capability/both phases/receipt, exact actual spawned PID, typed actual-event records, order, binary32 policy, MXCSR control bits (RNE; FTZ/DAZ off), x87 rounding and scalar arithmetic declaration. Planned lane/block/tail counts never establish actual reachability. Real dispatch/FP/scalar-vector observations still require a separately reviewed trusted provider.

Existing reference artifacts prove build availability and historical smoke/import identities. They do not supply actual selected-function address, MXCSR at the native boundary, scalar/vector execution provenance or atomically bound output. Labels, environment CPU masks, old trace worker entry or output equality cannot replace these observations. Therefore **NO-GO for all native cases**. No attestor design/rebuild is implemented. An instrumented rebuild would be a NEW reference identity requiring separate scope and authorization; it cannot silently preserve the existing artifact pins or old-R8 identity.

## Exact producer contracts

The accepted producer inventory is `docs/reports/OPENCV-CPU-SUPPORT-PRODUCER-DTYPES-2026-10-05.md`, SHA `f9b854cb06affc350c383b2f6ef1a48d7885ccc73817ea388710b91d78a36056`. Its numerical contract received Astra ACCEPT and its corrected callsite inventory Luna ACCEPT in Comms. This harness makes no adapter/coercion changes.

| Producer anchors | Conditional contract |
| --- | --- |
| lattice_planes.py:937-940,:1821; lattice.py:1306 | Actual uint8 C1 or tightly interleaved uint8 C3 image; dtype forwarded unchanged |
| lattice_planes.py:942-944,:1531 | float32 C1 region 0/1 |
| lattice_planes.py:1824-1827,:1863 | float32 C1 coverage/region; +0 or positive normal [0,1] |
| lattice_planes.py:1220-1237,:1381-1383; image_derived_lattice_measure.py:481-486,:506 | float32 C1 structural raster; default confidences [.25,1], +0 background; every actual source pixel still validated |
| lattice.py:1313 | Caller mask only if exact admitted uint8 C1/C3 or bounded float32 C1 metadata/domain matches |
| supported.py:539-541,:549-552,:558-560 | Legacy caller image conditional uint8 C1/C3; generated float32 ones admitted; unchanged caller mask only exact admitted contracts |
| image_derived_unit_face.py:259-262 | Advanced/evaluation crop image conditional uint8 C1/C3; exact crop origin/strides/address/padded byte span must bind native view; existing V1 witness only |

All require accurate Mat remap, independent absolute planar float32 maps, INTER_LINEAR, BORDER_CONSTANT and strict source/map/output layout/alias/address gates. New C3/f32 contracts require integer +0 border; C1 retains integer uint8 border saturation. Float sources admit raw +0 or positive normal <=1; -0/subnormal/negative/out-of-domain/NaN/Inf are refused even at zero symbolic coefficients. Float64, uint16/int, bool source arrays, float C3, planar/arbitrary-stride RGB, relative/interleaved maps, unsupported border/FP/dispatch are refused. Post-remap float64 conversion/averaging in supported.py does not change the source contract. Arbitrary caller masks/images and unverified native crop metadata have no universal coverage claim. Symbolic same-channel support and float arithmetic remain conditional on trusted ingress; output shape/bytes do not prove source dependency.

## Frozen finite synthetic plan

Plan SHA-256 `6097f8adbf1f6fa0023977c91e991fae87be25a591247b4bf35faa7f7e429964`.

492 unique recipes: 486 combinations plus 6 saturated C1 borders. Each dtype/declared-dispatch pair tests widths immediately below/at/above source-planned uf=2*float_lanes (BASELINE/SSE4_1 planned8; AVX2 planned16), contiguous/padded source and separate padded planar maps, two output rows and scalar tails. Patterns are integer, half, nextafter below1, negative-near-zero minimum-normal, mixed border, all OOB (-1,-1), degenerate1x1, and smallest-normal coordinate products. C3 adds one impulse per channel with zero-valued other channels. Float controls contain0,.25,.5,1. C1 borders -17 and300 cover uint8 saturation; all new-type borders +0. These are source-derived plans and fake events, not observed native branch coverage. `matrix_pin()` binds recipes, harness bytes, accepted symbolic SHA, all reference/config/source pins and caps.

No native expected-output fixture, epsilon arithmetic model, shadow estimator or native dependency proof is produced. The independent accepted symbolic worker builds/replays its CSR before fake backend entry. Receipt verification separately checks dtype/channels/shape/row stride/output byte span/digest and the worker's bounded support summary. Packed witness remains child-owned; parent receives only its seal/size, not a second full witness. Actual output numeric replay remains NOT_STARTED; fake zeros and transport equality cannot imply native correctness or pixel influence.

## Resource and lifetime envelope

Worker uses unchanged schema4 N<=262144, each native source/map/destination span<=16MiB and permanent aggregate ledger<=256MiB. For I source+map bytes and W=4096+184N+8: R=2I+2W+16MiB, formal Wmax48,238,600 and schema4 Rmax213,917,712. One Worker accepts one request; it is consumed before capability/build/call work and refuses reuse after success/refusal/exception/interruption. Native/Synthetic subprocess orchestration never resets/releases production ledger. Existing accepted module remains unchanged. Test fixture ledger replacement is explicit test-only modeling of fresh child lifetimes, not a production reset.

Parent reservation P=8I+8O+3*(MAX_WIRE_BYTES+8)+8MiB <=64MiB, where O is exact output native byte span. The added fixed allowance covers up to16MiB bounded incoming frame, parser text, and parsed long-string representation independently of a tiny expected request/output. `unframe` returns a memoryview rather than another full slice copy; conversion/text and parsing have a maximum three-wire-buffer peak. All existing64MiB parent/16MiB wire/256MiB child caps stay unchanged; admission becomes stricter instead of enlarging caps. P implies I+O<=1,048,573 bytes. Parent may refuse otherwise valid schema4 inputs, including maximum512x512 maps, under this independent transport envelope; accepted symbolic caps/behavior remain unchanged. No all-traces/witnesses memory accumulation. Metadata<=64KiB; JSON structural punctuation<=2048/depth<=12/unquoted primitive<=64 precede parser allocation, with bounded shape/source-pin types before tuple conversion. Packet/parser temporaries are dropped before witness allocation. Child prelude remains conservatively<=64MiB; witness/output stage R+8O+16MiB <=138,420,216 bytes (<256MiB). Loose simultaneous application-buffer bound with parent64MiB is <=205,529,080 bytes. These are representation reservations, not measured RSS, interpreter/DLL/OS/thread-stack commitments or native/multi-view ownership proof. Native max-span execution remains absent.

One child at a time; a ChildOwner context exists before spawn and encloses all post-spawn encoding/thread construction/start/verification work. Cleanup independently attempts poll/kill/reap, every acquired pipe close and each started thread join/aliveness check after errors or interruption; incomplete teardown latches refusal and prevents successful receipt return. Each context accepts at most one receipt for its nonce and actual Popen.pid. stdout/stderr share one bounded read cap; fixed read/write threads;30s timeout. Request/output frames reject truncation, extra data and oversized counts. This is best-effort bounded teardown, not immunity to an arbitrary OS/factory failure inside Popen before a process object is handed to the context. Failure never reopens consumed run status. Durable exclusive-create IN_PROGRESS precedes the first fake plan transport, is fsynced and survives failure/crash; COMPLETE is an append only after all validated synthetic summaries. Seeing either marker state refuses rerun. Guards write only caller-authorized paths; tests use tmp_path. No project evidence/marker path is authorized/created here; future native artifacts/authorization need independent lead scope. Fake plan folds only summary hashes and releases case-local output, without retaining trace arrays.

## Pure validation

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv/Scripts/python.exe -B -m pytest tests/test_cpu_support_reference.py -o addopts= -p no:cacheprovider -q --no-header
```

Final revised focused result: **594 passed in 3.10s**, exit 0. Prior pre-review freeze555/2.11s was REVISE. Initial remediation run591 passed/1 failed (2.35s): exact child PID binding correctly refused the Windows .venv python.exe redirector/actual-child mismatch. The correction launches the pinned same base interpreter directly; the subsequent complete focused suite passed594/3.10s. All source/test edits are frozen. No native call/import occurred during either run or the bounded stdlib-only diagnostic child. Prior coherent draft539 passed in 1.93s; subsequent changes added typed-event/receipt checks, pre-parser caps and guard wiring. Only this focused fake-mode module ran. Tests cover 492 recipes; pre-call zero-backend failures; capability/old-receipt distinction; wrong FP/branch/import identity/call metadata; source nonfinite/subnormal/overflow and zero-weight source hazards; alias/alignment/address; parent/wire caps and decoder pre-allocation sentinel; tamper/digest/order/type; worker nested/refusal/failure/KeyboardInterrupt/SystemExit; marker order/completion/rerun/later failure; and real stdlib-only synthetic child plus timeout termination. Synthetic native declarations cannot yield a native PASS.

AST parsing/no bare integrity asserts/native imports and UTF8/LF/final newline/no-tabs/trailing-whitespace QA passed at final freeze. Read-only rehash of all 21 existing reference files passed. No symbolic suite rerun or production tests/native imports in this assignment; prior schema4 286-pass receipts remain historical and unchanged.

## Exact review remediation

R1: authoritative child context is established before spawn; every post-spawn acquisition is enclosed. Fake-process tests inject failure after spawn, each of two Thread constructors and starts, kill/reap, each stdin/stdout/stderr close, both joins and incomplete thread teardown. All remaining cleanup attempts occur, incomplete teardown refuses, and another receipt cannot be accepted. Consumed matrix status remains IN_PROGRESS on failure; successful summaries append COMPLETE only afterward.

R2: every outer/request/output/support/attempt digest is exact64 lowercase hex before serialization; constant-schema identity/string/int validation rejects metadata substitutions. Output hex length is bounded by the exact expected span before seal. Matrix summaries have exact seven keys, typed PID and canonical digests; extra/oversized fields refuse before JSON allocation. Sentinel tests patch serializers to fail if reached for oversized/noncanonical digest or extra summary fields. Worker record encoding is reused for sealing rather than serializing a second time. Fixed maximum receive/parse allowance is explicit in the stricter unchanged-cap envelope above.

R3: parent generates a fresh256-bit nonce for each child; the envelope binds it and the actual spawned PID. Worker requires envelope PID==os.getpid before backend entry. Capability, both atomic phase records and sealed receipt carry the same nonce/PID; parent compares them to its owned context, not a non-parent-PID heuristic. Exact child receipt is accepted at most once; matrix summaries reject duplicate attempt nonces. Replay of a stale same-request receipt, wrong child PID, resealed substituted phase nonce/PID and duplicated nonce summaries refuse. This is protocol freshness, not authentication against an adversarial local backend.

Windows launcher correction: `.venv/Scripts/python.exe` is a redirector and its Popen.pid differed from the executing Python PID. Synthetic workers now run the same pinned base executable directly with `-I -S -B` (isolated, no site hooks, no bytecode). Before spawn, harness verifies CPython3.10.11 exact sys.version string, win32/64-bit layout, base path, executable/DLL SHA, and stdlib-only AST imports of harness+accepted symbolic module. Base path `C:/Users/Likea/AppData/Local/Programs/Python/Python310/python.exe`, SHA `3cce33d75d6fdae4e004d0bdf149320b3147482a9caf370079dcb9c191a1b260`; adjacent python310.dll SHA `14b06796f288bc6599e458fb23a944ab0c843e9868058f02a91d4606533505ed`; version `3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]`. This pins the pure worker interpreter; it does not add native FP/dispatch attestation or modify any environment/build/.venv artifact.

## Frozen pins

- scripts/evaluation/validate_cpu_support_reference.py: `183411c393590d57304d644dab34fdd1873d542ab63c064c36f815bd347f5d8b`
- tests/test_cpu_support_reference.py: `8f573d87ae0dfa6f682b93191638198970f91274f8338a583daa62afc9257ce7`

Accepted symbolic module SHA `7f332788762ec308ad396aa3c8c82a4364b81ccd09d6311200824ce0c844769d`; accepted symbolic tests `36a39b02d7f93e74825f9fdf7defce622f8ecfe0df5eb6e4257e0b1f553e9bfa`. Source commit 40738fb16ceddb5fb3fea747585f7ce6abb0605b, existing OpenCV 5.0.0 reference root C:/Users/Likea/AppData/Local/Temp/mc-photo-opencv5-cpu. Historical reference artifact hashes are checked without loading/importing binaries. Raw reference-build-information.txt file hash below differs from its historical decoded logical build-information digest20e38ff4f7bb09560492f2b4bca416f6d626feb7762b62dfe89abda66f8e7eab; both identities retain their distinct meanings.

| Relative reference path | SHA-256 |
| --- | --- |
| provenance/final-artifacts-sha256.json | `46d3b51cd4b3ea2efa2592e47b3204ae079652ef9ac7a6c3cb31d5f38ffe3421` |
| provenance/precompile-gates.json | `4db1525235cab2261a30a29e09e4fdd19ed9fc4415bbeb0cff19aa33482cb077` |
| provenance/generated-static-gates.json | `d2c33f2d5807baa20d56295cd8f58db58f19bc5bb5738edc19fe0995dd0aed57` |
| provenance/built-binary-gates.json | `d1b950a3f1ee3fee85089d350c98fba36e591bbda7043dff3a81f5c84305d2bc` |
| provenance/import-probe.json | `e2f86db59653cd4c607f4467f5cd320b04bfb5003b2e814956204683fbc8199b` |
| provenance/reference-build-information.txt | `d7449aeff4c79db372adbe3bc778a0dde2c5d218aafbeca57ac2c57860eccc68` |
| build-clean/CMakeCache.txt | `7932fade553a41cf4f11acdb9fb7cfc157be53d7c880e7662766d62bef241bbb` |
| build-clean/cvconfig.h | `36b3f6b4f3a56625322b53ec4648fbb0f6ff4a7e17117bf9c8e451701f38ee68` |
| build-clean/cv_cpu_config.h | `4e9cda34d3df32950977d63b30b3a921847ce0aac7bbff83f781614a3c61d3ea` |
| build-clean/opencv_python_config.cmake | `d0cd0958eb88909562d202f158b87887120b816afd823ef29850158baabd27a0` |
| build-clean/compile_commands.json | `f7b7540b3afff8edea808dee333c0d4d43640aeedfc95a2a2f7e92e047d24280` |
| build-clean/python_loader/cv2/config-3.10.py | `ed08d9584d5d3f26d9d06f66c961acdac5eed47e373ef2e8c6de1c6be74ed5de` |
| build-clean/lib/python3/cv2.cp310-win_amd64.pyd | `15e8c9f063076e4e6c07bbba053b0ce02fcfb201eff35145ce2395aae686e4e0` |
| build-clean/bin/opencv_core500.dll | `2149d6ca8914330ac9d2ebfbbade70af4e9d58f6a427fb8010664fd45fb55159` |
| build-clean/bin/opencv_imgproc500.dll | `6a7be1e21436a9f909eed1dfdf12f9277fe472010de51b82108537cbae8d1276` |
| build-clean/bin/opencv_geometry500.dll | `df7263c34d850e58390d80618de97bfa7b1a62b90c47c617bfb34ffa2bc415be` |
| build-clean/bin/opencv_flann500.dll | `9b8c6315a0ab8fab1007ebb927928402d0007d9ebb44ffc5d9eea27ad336c314` |
| source/modules/imgproc/src/imgwarp.cpp | `b7a95e4bb5c2c5adc5378c43151fa4bc6e461ec3bb267d9d72eda91a004595ae` |
| source/modules/imgproc/src/warp_kernels.simd.hpp | `3e19fdf116c99babd38775f543301276db59ae8bbfa50edaaba7afe89c1a7c73` |
| source/modules/imgproc/src/warp_common.scalar.hpp | `3da7138a38c611aedd4f4805b35f35cde9e5e038107bedce04c4295fe1f35af6` |
| source/modules/imgproc/src/warp_common.vector.hpp | `ca5314ca884189cf790afc4d14ef121f3fd30f7b25bf3054f977cef81160df21` |

Production calibration `2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc`, lattice `3077b4c380b293fe7b97a7a127cd09e42e79fc7277de654168a8ec0274a87bd7`, observer runner `7d17734cbce8cbbc5490b6f82363d72edecccd5ff098b4a4a63f3a0f2d23e34d`, historical V2 report `24b44c5a25659be249e92391abbc913b3f8f83be98dac3388c9315ef93c84b54` and accepted producer report are unchanged. Observer evidence/run directory absent; capture remains NOT_STARTED and production unsupported_kernel remains closed.

Acceptance of this package can establish only bounded fake orchestration. No native execution is authorized by this report or its tests. Installed custom-HAL certification, reference-to-old-R8 equivalence, observer readiness/capture, multi-view lifecycle, production integration, reconstruction/image-only accuracy and advanced manual-input promotion remain unestablished. Default geometry/exact-state gates and advanced-only camera/F3/coordinate policy are unchanged.
