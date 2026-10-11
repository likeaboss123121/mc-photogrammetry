# PLANE-PERIOD-SUPPORT-01 observer revision

PLANE_PERIOD_SUPPORT_EXECUTION_STATE: NOT_STARTED

Implementation review package; runtime NO-GO. No real R8, source RGB, reconstruction, truth, or registered capture has been loaded or executed. Evidence and future run directory are absent. This package cannot consume its one-shot marker: unsupported-kernel preflight unconditionally refuses before claim_run. Independent Astra acceptance and separate lead authorization remain required; neither would bypass the unresolved kernel gate.

## Scope and fixed bindings

A separate private ContextVar carries only observer evidence. Legacy V1 witness sinks, numerical decisions, estimator inputs, defaults, production gates and advanced-only camera/F3/manual metadata policy remain unchanged. Observer snapshots have detached immutable byte-backed arrays. Ordinary diagnostic exceptions latch incomplete; KeyboardInterrupt/SystemExit propagate; ownership is exclusive by run/thread, nesting refuses, and context resets in finally. No extra production measurement, remap, profile, period, signature, matcher or solve call is introduced. Solver_result captures the already returned existing solution.

Four fixed reading selectors and three full endpoint/axis/alternative-b/relative-quadrant selectors, original R8 raw/core/ordered38 pins, source closure, runtime, limits, path/format and one-shot semantics are bound in the registry below. Identity is frame plus canonical reading_order_key position throughout private signature/matcher/replay; legacy SignatureEvidence global iteration index remains unchanged and is joined separately to preserve production hashes. Every retained alternative requires its own signature binding.

## Semantic replay and seal barrier

Each append is one independently finalized gzip member, flush/fsync followed by complete readback; order, native dtype/endian/shape/raw hashes, CRC/trailer, exact expected offsets/lengths and domain-separated nested event seals are checked. Earlier complete members are preserved on failure; incomplete tails refuse. A first complete bounded structural/native replay and final full-stream digest are prerequisites to the second semantic pass. No numerical interpretation precedes the first full-stream seal barrier; final compressed digest must still match afterward.

Replay joins trusted ingress to decoded RGB; original RGB to masked grayscale; captured detector segments to contrast weights and rasterized structural image; captured calibration to basis/focal/principal; actual candidate region to usable support/window/projection maps and native rectification output; patch to directional profiles, normalized correlation, peaks, harmonic choice and phase; supra signal to the winning directional profile; pixel/world scale to all evaluated and retained b supports; reading projection, phase evidence, structural share, confidence and selection order. Retained signatures bind their reading/alternative, actual remap outputs, derivatives/cell sums/coverage, full mask set, legacy V1 cell membership and final signature identity. Matcher binds emitted signatures, declared grid/quadrant homography, warp/coverage, logical selected pairs, final match identities and the existing solver call/return. Frozen R8 core comparisons remain mandatory for any future real capture; baseline drift refuses rather than replacing a selector.

Evaluator-only post-production cv2.remap over detached sealed source/maps/arguments requires exact dtype, shape and bytes; no epsilon. That check proves output replay only. V1 tap/cell membership checks establish legacy-table internal consistency, not installed OpenCV5 actual kernel support. Semantic finish also refuses unsupported_kernel even if every other join succeeds. No support-complete claim is made.

## Memory and caps

One monotone pre-allocation ledger joins callback snapshots and journal work, with a fixed default 256 MiB limit. Freed objects remain charged, including retained callback records. Plans cover nested scalar/container/native snapshot payloads, targeted witness construction, encoding/canonical buffers, compression state and overlapping readback, compressed slices/decode tails/chunks, joined decoded bytes/JSON tree, descriptor/event hash buffers and semantic replay temporaries. Native-array and structural depth/node caps apply before copy or hash allocation; bytes/strings are counted. Conservative reservations may refuse far below declared format maxima; a cap refusal never truncates evidence or changes production return.

The host-object and witness envelopes have synthetic boundary/pre-allocation tests. They still require independent audit; this report does not claim a proven whole-process allocator/RSS bound for opaque OpenCV/native backend workspaces. Actual active-kernel V2 support, its allocation proof and native backend verification remain unimplemented. Runtime remains blocked before any real capture.

## Active-kernel blocker

Installed opencv-python 5.0.0.93 is a CI build; cv2.pyd is 86,293,504 bytes with SHA f1e76ab01cf4d33af780adf8ada1a26c5eaeecc55d2cbe8a5e288195e8f9b0dd. Custom HAL is ipp 0.0.1, IPP 2026.0.0; AVX512 dispatch is present. Build hash and all runtime flags are frozen below. IPP runtime toggling is not active-path proof. V1 fixed-map replay differed from native float-map output by 5.485086441040039 on a bounded synthetic stripe patch; ideal float arithmetic also fails bitwise equality. Both are diagnostics, not acceptance tolerances.

Official opencv-python tag93 resolves to commit b83046cda41133f1bf2e73e99dba16a1248f103a. Its Windows release run28506231080 completed successfully but exposes no remaining artifacts; release assets are empty and logs endpoint returned403. Installed wheel has no PDB/CMake cache. Windows workflow uploads wheels, not PDB. [Release](https://github.com/opencv/opencv-python/releases/tag/93), [workflow](https://github.com/opencv/opencv-python/blob/93/.github/workflows/build_wheels_windows.yml).

Static dumpbin confirms zero COFF symbols and CodeView PDB GUID A7249F5F-CF40-431D-9DEE-38757EF97616 age1 at the opencv-python Release build path. No named remap/HAL export or import resolves the exercised control flow. These findings do not prove absent HAL or IPP_CALLS_ENFORCED. Matched PDB/build provenance or an audited exact stripped-binary control-flow analysis is still needed. No speculative support table or V2 sidecar is implemented; no active-kernel completeness claim.

Conditional future V2 design remains subject to Astra: verified reachable dtype/channel/CPU/HAL dispatch, native rounded float32 fractions/endpoint handling, symbolic nonzero coefficients, fail-closed unverified subnormal/overflow, canonical CSR, all borders/degenerate/SIMD cases and (-1,-1) invalid maps as constant/empty-source rows. Source zeros cannot erase support. Output bitwise replay alone cannot establish these properties.

## Synthetic validation and limits

Focused command, with PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src:

```text
.venv/Scripts/python.exe -B -m pytest tests/test_plane_period_support.py -o addopts= -p no:cacheprovider -q --no-header
```

115 passed in30.14s, exit0. Includes genuine unchanged controlled lattice-to-signature-to-graph call/argument/output/hash parity with ordinary and failing observer; controlled graph success/solver refusal/no-match solve-count and solver-return capture; canonical ordering/permutations; stage finalization/refusal/exception/interruption reset; native/endian immutability; gzip/order/seal/tamper/caps; resealed foreign-pixel, period/phase, candidate-window, structural-weight/raster/camera, matcher-signature/warp substitutions; all retained alternatives; hash/decode/witness pre-allocation and unsupported-kernel-before-one-shot controls. Genuine chain uses two frames from the existing bounded synthetic session fixture with max_cells=8 fixed in both arms. Nonempty canonical reading and exact matcher selectors are selected from the unchanged baseline output, and successful capture must emit actual signature_sampling and matcher_pair records with no incomplete latch; this is synthetic development QA, not R8 or renderer-truth evaluation. The four-cell trial honestly produced no matches and failed nonvacuity; no gate or production default was relaxed. Controlled graph solver outcome tests use returned synthetic solutions; they do not claim new reconstruction success. Real target selector completeness and baseline parity have not been exercised.

AST and raw-byte trailing-whitespace/newline checks passed on the eight source/test paths. Calibration source remains 2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc. Expanded frozen-source suite: 475 passed in179.15s (0:02:59), exit0; session16444 completed. Exact command:

```text
.venv/Scripts/python.exe -B -m pytest tests/test_lattice.py tests/test_lattice_planes.py tests/test_plane_graph.py tests/test_image_derived_lattice_measure.py tests/test_image_derived_signatures.py tests/test_image_derived_graph.py tests/test_image_derived_graph_phase_capture.py tests/test_image_derived_graph_solver_input_bindings.py tests/test_image_derived_graph_stage_harness.py tests/test_plane_period_support.py -o addopts= -p no:cacheprovider -q --no-header
```

Both suites used PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src, Python -B, with pytest bytecode/cache disabled. Runner/report/source/test/preregistration are frozen for a third independent Astra review; runtime remains NO-GO. No retry/capture/evidence has occurred.

## Third-review hardening

Before any linspace/meshgrid/asarray projection work, scalar samples must be an integer in the bounded domain and match all sealed float32 map/bool-validity shapes; centre/principal/axis/side/focal/extent and finite 3-by-3 scalar axes are checked. Reservation is computed from the validated planned n-by-n shapes. Candidate region image dimensions must equal trusted decoded-source dimensions; crop/bbox/count/packed lengths and pixel caps are checked before constructors or decode. Whitelisted PlaneReading reconstruction requires the trusted frame shape and performs the same planned region reservation before its constructor can decode. Reading-order, signature materialization and signature-region paths use this guard. Allocation sentinels prove resealed tiny maps/huge samples, huge region image shapes, missing trust and insufficient planned budget refuse before grid or full-mask allocation. Exact planned-budget equality is accepted; one byte below refuses.

The actual candidate eligibility bit is stored, checked against structural_share >= the preregistered effective threshold, and joined to every ranked selection row. Frozen source policy is min_structural_share=.15, max_readings=6, max_per_axis_side=2, max_rectified_side=512; source and selection records must both bind it. Resealed candidate/ranked flips and threshold/cap changes refuse.

Solver-call replay retains integer_tolerance, min_resultant, ambiguous_margin and the ordered quadrant_constraints. The sealed complete solve_context is structurally/digest/frame validated and must equal all captured effective values and constraint rows. Resealed context or call substitutions refuse before phase/input replay; a sentinel confirms the matching context reaches the existing next replay stage. No extra production solver or measurement call.

## Owned source/test pins

```json
{
  "scripts/evaluation/capture_plane_period_support.py": "7d17734cbce8cbbc5490b6f82363d72edecccd5ff098b4a4a63f3a0f2d23e34d",
  "scripts/run_image_derived_graph_stage.py": "514ca58662d946666e5593beb2c82c5a34bb7f2dcd88b680b7a9aa36da92f2e9",
  "src/mc_photogrammetry/pipeline/image_derived_graph.py": "5d42b4feafd282b8a4f15a500120970f0bc69ca0469d885e55d3d8a77c5ab1ac",
  "src/mc_photogrammetry/pipeline/image_derived_signatures.py": "fa7a0973d701c3986873ad1f19751b1ed6671b3bb69f52928a2f2e13bbe26125",
  "src/mc_photogrammetry/reconstruction/lattice.py": "3077b4c380b293fe7b97a7a127cd09e42e79fc7277de654168a8ec0274a87bd7",
  "src/mc_photogrammetry/reconstruction/lattice_planes.py": "4c6853a0ad24589c244a3d0f472039ea53e9c079095e7069f0af930686a22f10",
  "src/mc_photogrammetry/reconstruction/plane_graph.py": "d8896f272aaf79850d3ae105dc002ba1e71dddac1d111ce91de29bdba147603b",
  "tests/test_plane_period_support.py": "ccef168e7cc8a96c44a5b555e9dcab1a6eae09e2cf2a6943dfda28510c459b20"
}
```

<!-- PLANE_PERIOD_SUPPORT_PREREG -->
```json
{
  "baseline": {
    "file_sha256": "c3d2ee4ad3e803754d7297ab0c09b7994b17c24de345fda281c0b5cb62888ff8",
    "graph_sha256": "0a64c5e73b62b376f41e337a756c61af880d72cee1a90992001a42d00d9f39cd",
    "ordered_match_identity_sha256": "28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105",
    "path": ".tmp-geom-phase-capture-r8-20261004/observation.json",
    "trace_sha256": "738c9dc09dc17cf64a74e91472cb59a45bbc9917de479cd4f33f80551becd736"
  },
  "caps": {
    "alternatives": 16,
    "candidates": 64,
    "compressed_member": 33554432,
    "compressed_total": 134217728,
    "decoded_member": 134217728,
    "decoded_total": 536870912,
    "events": 4096,
    "extra_memory": 268435456,
    "frames": 3,
    "matcher_pairs": 1048576,
    "matcher_witnesses": 3,
    "native_array": 33554432,
    "rectification_taps": 1048576,
    "rectified_samples": 262144,
    "rounds": 3,
    "signature_memberships": 1048576,
    "signature_witnesses": 64,
    "target_readings": 4
  },
  "evidence_path": "docs/reports/PLANE-PERIOD-SUPPORT-01-evidence-2026-10-05.jsonl.gz",
  "future_stage_directory": "docs/reports/PLANE-PERIOD-SUPPORT-01-run-2026-10-05",
  "mode": "observer_only",
  "preregistration_sha256": "490cad6ebb5fa75ab0765e5296bcc93f8a03cfb714dc7e7f6400c62ee1c8bcd9",
  "protocol": {
    "diagnostic_work": "only target witness mapping from existing locals; failures latch incomplete",
    "interpretation": "require identical sealed R8 core/failure/38 ordered matches; otherwise baseline_drift; no replacement selection",
    "memory": "single monotone 256MiB preallocation ledger per capture/replay; scalar/container/native snapshots, witness work, encoding, readback, bounded decode, JSON tree and semantic temporaries charged; freed buffers remain charged; conservative early cap refusal",
    "numerical_calls": "unchanged args/counts/data/decision/return/core hashes; no legacy sink activation",
    "one_shot": "exclusive durable NOT_STARTED->RUNNING before baseline/RGB load; no retry",
    "private_context": "run/thread owned; no nesting; finally reset; interrupts propagate",
    "remap_output": "post-production detached cv2.remap replay only after first complete stream verification; native dtype/shape/bytes equality; no epsilon and no extra production sampling",
    "replay": "complete bounded gzip/CRC/order/seal/native/semantic replay before interpretation; final digest recheck",
    "support_kernel": "UNSUPPORTED: exact installed HAL/CPU dispatch and nonzero kernel support unproven; unconditional unsupported_kernel before claim_run and semantic completion; V1 preserved as legacy-table witness only; V2 not implemented"
  },
  "reading_policy": {
    "max_per_axis_side": 2,
    "max_readings": 6,
    "max_rectified_side": 512,
    "min_structural_share": 0.15
  },
  "runtime": {
    "cpu_features": "SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *AVX *FP16 *AVX2 *AVX512-SKX",
    "implementation": "CPython",
    "ipp_enabled": true,
    "ipp_not_exact": false,
    "ipp_version": "ippIP AVX-512F/CD/BW/DQ/VL (k0) 2026.0.0 (-) Mar 27 2026",
    "numpy": "2.2.6",
    "opencl_enabled": true,
    "opencv": "5.0.0",
    "opencv_binaries": {
      "cv2.pyd": {
        "byte_length": 86293504,
        "sha256": "f1e76ab01cf4d33af780adf8ada1a26c5eaeecc55d2cbe8a5e288195e8f9b0dd"
      }
    },
    "opencv_build_sha256": "f71df2f163323276b66a0c26f605490476e207bf50042bae19501cff4f1d865a",
    "optimized": true,
    "pillow": "12.3.0",
    "python": "3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]"
  },
  "schema": "mc-photogrammetry-plane-period-support-v1",
  "source_sha256": {
    "scripts/evaluation/capture_plane_period_support.py": "7d17734cbce8cbbc5490b6f82363d72edecccd5ff098b4a4a63f3a0f2d23e34d",
    "scripts/run_image_derived_graph_stage.py": "514ca58662d946666e5593beb2c82c5a34bb7f2dcd88b680b7a9aa36da92f2e9",
    "src/mc_photogrammetry/__init__.py": "e54953216fa986cd12def3b31c30d8e200c26378057e006117f33f49a81b192f",
    "src/mc_photogrammetry/blocks/__init__.py": "d791f35c103aee8a9561f039a1f2a8493e21014692e9329fe84628404a332069",
    "src/mc_photogrammetry/blocks/appearance.py": "2380f1676e713d340432dfa28f2146b924206157bbb41f79f8c39ed897c621b8",
    "src/mc_photogrammetry/blocks/model_atlas.py": "f53e8878c55fc902d76ef79816e92f0a3e70e2a20eb26458f1e81ae697403c54",
    "src/mc_photogrammetry/blocks/textures.py": "f735ae866d949bdcf6f24c90b2a604b4c2b8a0b2d5661561aa9f4ff47c37cc45",
    "src/mc_photogrammetry/domain/__init__.py": "f223d7149f7c3a3f320f7b33705820f77288f9fd256a4c284c520cf20f357d35",
    "src/mc_photogrammetry/domain/models.py": "6a87d543661cc62ada9a154bcdaa089ca844f41691ff3b53207ad89e3f05a478",
    "src/mc_photogrammetry/export/__init__.py": "259fac03dcbb676b1f518aa515d7f38b26efd499071c0f29b0db232fb07e1b3e",
    "src/mc_photogrammetry/export/attribution.py": "4e053555f215c7305fa384775e5186be2f93d6d03c0bc9b4cb210180a6a47f96",
    "src/mc_photogrammetry/export/compare.py": "6d1cc9ef3bc7d5e7dc08ef2a7fb269cb4bf7e3ee0f4cc2e01114f47625d2e738",
    "src/mc_photogrammetry/export/litematic.py": "1917451e46ccb451dcbf2e5194161edfbea0862dc55e62489fcdeeb14f3db794",
    "src/mc_photogrammetry/export/validation.py": "9e53d5ca8388b42eac19d8eee47e80fd97dbc162d8a0ba73c7282841fef331cb",
    "src/mc_photogrammetry/input/__init__.py": "14ee634d9fab59deffdc5c48b800fd1619773909a26213ae487f8c043855e64f",
    "src/mc_photogrammetry/input/frames.py": "ed9b166d29887ef849d3d601fdef89306fccac4d40cae4d39a48a4a6ff290d8f",
    "src/mc_photogrammetry/input/image.py": "43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732",
    "src/mc_photogrammetry/input/media.py": "8f872b165283fb65a01e7470062cb51040feeee46ecb25ed26cdb0b67b7c2f89",
    "src/mc_photogrammetry/input/video.py": "8265ada8f8d9153d54b94365fb50a643a8c892edbb0ab21176fe9af01d629d04",
    "src/mc_photogrammetry/pipeline/__init__.py": "28dd13121ddce9923020ab7493c416e4146520e4f734f8588d02389008312f32",
    "src/mc_photogrammetry/pipeline/candidate_support_provenance.py": "a8bd3102652720513ed85e2d3b7e2c81a79a01a7621c41238b09f6e741604562",
    "src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py": "f5cee3cfe9d0247069637992fb29d1f9a8533abc9383af841cfc9a726313ccd2",
    "src/mc_photogrammetry/pipeline/connected_camera.py": "3b5df735084daddccb7f25511af1b8549c0300d02e4841ade39e39461b8275a6",
    "src/mc_photogrammetry/pipeline/contracts.py": "db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc",
    "src/mc_photogrammetry/pipeline/correspondence.py": "192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1",
    "src/mc_photogrammetry/pipeline/frame_ingress.py": "3411b4067281b19a6ac2e530ece5dfc2e53e0f63b978e83e09ffadb27f715f0a",
    "src/mc_photogrammetry/pipeline/image_derived_connections.py": "a2d166bf711d6cd77c169f044da5ca28d00073e66e60b9bff992572fb56ba211",
    "src/mc_photogrammetry/pipeline/image_derived_entry.py": "9926312ae58d862ad95ef8754d0c49e1ff8d2e86e45a51674a84046208f5cd02",
    "src/mc_photogrammetry/pipeline/image_derived_graph.py": "5d42b4feafd282b8a4f15a500120970f0bc69ca0469d885e55d3d8a77c5ab1ac",
    "src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py": "8441e91cf158d5403676d249fe6ebdebeeb723bd4a9967fdeb8371beb1c9cb81",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py": "491841a58e5faca5b07200d88a0115c3aca2d25c6328206755b3295e0b785d27",
    "src/mc_photogrammetry/pipeline/image_derived_pose_graph.py": "b80b75034ff133ab8dd25ea328081863c97abd7969c9d03e35b751e01447a74e",
    "src/mc_photogrammetry/pipeline/image_derived_production_record.py": "e0cac4470fdc4129f21b6589cb7a4650a335fa27b88fba0dd5f61ef9ea914411",
    "src/mc_photogrammetry/pipeline/image_derived_session.py": "a97d5ce04ef91dba6501dff9d0436d5a09e2f74539c411f5cdafd05e85720060",
    "src/mc_photogrammetry/pipeline/image_derived_shift_domain_policy.py": "271e684610c0a4f85ea94b86d4713a317fb84e9ffd345b507d96164b626ab633",
    "src/mc_photogrammetry/pipeline/image_derived_signatures.py": "fa7a0973d701c3986873ad1f19751b1ed6671b3bb69f52928a2f2e13bbe26125",
    "src/mc_photogrammetry/pipeline/image_only.py": "5a6e99b66e7d580d9082f00347514d286937e9a6917dd4a36b8bcc9f5591d391",
    "src/mc_photogrammetry/pipeline/joint_matching.py": "9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3",
    "src/mc_photogrammetry/project/__init__.py": "3cf716d6d4b09e1a26420b9557b07fe26325eac9a1d326494cb72f20decd70be",
    "src/mc_photogrammetry/project/store.py": "bd984ac51079c4b01144814db35f38e14efe6d2f64787e9c03b6177f3d559559",
    "src/mc_photogrammetry/reconstruction/__init__.py": "7f80f67bfe5ce8f93a4cf140f093e5d23458b6b194814a2c0a8d2973f99c5d09",
    "src/mc_photogrammetry/reconstruction/appearance_scoring.py": "84f767fb781bebb08c39cb45fa0ec41990655b3562dcfa090ff8f4defda9ed02",
    "src/mc_photogrammetry/reconstruction/appearance_state_proposals.py": "39ecfc70415472a7770a76fe492cc251d42233a43a557ecbf60305ae8846876a",
    "src/mc_photogrammetry/reconstruction/boundary_candidates.py": "95e550a44bd65a19044b2a35a99d9a503b389d066085dccb464b0f18fa89216c",
    "src/mc_photogrammetry/reconstruction/calibration.py": "2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc",
    "src/mc_photogrammetry/reconstruction/camera.py": "0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680",
    "src/mc_photogrammetry/reconstruction/facade.py": "c324946e27c604f62f3397af7d8f331febfdd43fa8fa8a757a0277e64344a6cf",
    "src/mc_photogrammetry/reconstruction/face_observation.py": "03710aaec58d1e2fffb769174db34f826c7d8dd830bedb8d9074bd231bda8c1e",
    "src/mc_photogrammetry/reconstruction/face_sampling.py": "14be0f406364c2175db84aa4909825a2d059511b6d39c0c33debf4b3d2ecba6e",
    "src/mc_photogrammetry/reconstruction/grid.py": "401888823cc50db2112f8b78ab5a91f3ca35010bf65bfda51ecbec61f2e0a266",
    "src/mc_photogrammetry/reconstruction/lattice.py": "3077b4c380b293fe7b97a7a127cd09e42e79fc7277de654168a8ec0274a87bd7",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py": "4c6853a0ad24589c244a3d0f472039ea53e9c079095e7069f0af930686a22f10",
    "src/mc_photogrammetry/reconstruction/lighting.py": "7e99b691f258d6fca055b4620309e069c0be700b8ca20b8f8d6fd6d8c9f5c148",
    "src/mc_photogrammetry/reconstruction/multiview.py": "acfc0e4c4e6b45102a3911c19b6e4d6ec86c455428eea1a2ad19af8c5b287d8a",
    "src/mc_photogrammetry/reconstruction/occupancy.py": "81b32acfc4cf370f303a67d83dcab9fff2a246d7a5d0733c80b973f212fba59b",
    "src/mc_photogrammetry/reconstruction/plane_axis_support.py": "18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f",
    "src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py": "3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7",
    "src/mc_photogrammetry/reconstruction/plane_geometry_provenance.py": "1d51ffb706be2826af7bd9393bec8537f2bb9499937d71baec49c100338c8605",
    "src/mc_photogrammetry/reconstruction/plane_graph.py": "d8896f272aaf79850d3ae105dc002ba1e71dddac1d111ce91de29bdba147603b",
    "src/mc_photogrammetry/reconstruction/plane_phase_anchor.py": "36d8d24541a856d97476ae440ee37fb43173672cfff18340538ca92b56bdeefc",
    "src/mc_photogrammetry/reconstruction/plane_placement.py": "743fd95a18651a342642ced43701803cda379a2d392604557dc6069ba3909880",
    "src/mc_photogrammetry/reconstruction/plane_region_support.py": "45a5b02947e3f95ee5d160334b5172e3a547b16c5893b9c757f964839d8896e3",
    "src/mc_photogrammetry/reconstruction/plane_relative_translation.py": "6b0d971e1badc4ae25001850670d248d3bd19f342cc4605c0513b3cc758ce113",
    "src/mc_photogrammetry/reconstruction/plane_residue.py": "a3bc4365aeefb0773ff020685825c1d4e2e21650aaa0515dfd03c9f1280d87a4",
    "src/mc_photogrammetry/reconstruction/plane_shift_domain.py": "7f3cbab5e4d665e620b0d0bff2bf1b11f6fa344eb41977cfd308273f9c11bc91",
    "src/mc_photogrammetry/reconstruction/plane_vector_domain.py": "fd60bda08d8304eb378dc7eb568a6ea6365031e69ccd9bd4c3590e01f92411f0",
    "src/mc_photogrammetry/reconstruction/plane_volume.py": "872b6e5317f7cf0f2c729ccb57f0ad8c52368290397facb0eb34b37eed25a7bd",
    "src/mc_photogrammetry/reconstruction/raycast.py": "44864cbeb56ae1d171acb9cb4c95256c9c456896fe4127f676010a550eb88ceb",
    "src/mc_photogrammetry/reconstruction/region_growth.py": "5375bc247defa370baee530434ef9751d9aec8a477ad143599d8fadc5b007e8a",
    "src/mc_photogrammetry/reconstruction/residual.py": "44a0d05afac68d11822bb8dbf1ed5e50fe97ae7e2557f1eb74d7680ad846791b",
    "src/mc_photogrammetry/reconstruction/scene.py": "cd72ead7f553c0157bbdf80b3651e5c6d31faf379af068f3ae235a5e4e47e53c",
    "src/mc_photogrammetry/reconstruction/sfm.py": "256cf5a882bb1b535838c638c16b2e4290bd802531e72cc3db1bf7bab459a181",
    "src/mc_photogrammetry/reconstruction/shape_fit.py": "cb08402f5b0230a615fab70b57982b539b3f361c9269f4197554a479b8ad0b97",
    "src/mc_photogrammetry/reconstruction/state_promotion.py": "65e361ebdc0efaa91a38ff15ddf2780f49791b536b3caa73a41575b8fe5a9824",
    "src/mc_photogrammetry/reconstruction/stereo.py": "7dfef297603b317075134d6f60c1e345ae796acb3e92a18dc04f6d172eab9c38",
    "src/mc_photogrammetry/reconstruction/supported.py": "4c94564d3842f1c3a03883f6f95a78a4ac08260150eb9fc12251bf708e35f29c",
    "src/mc_photogrammetry/reconstruction/surfaces.py": "888ebf1ab837baccd845b85256fb2cc7b2cf518352c371eac58fbc48264ec11c",
    "src/mc_photogrammetry/reconstruction/texture_calibration.py": "15b987e645463a84b0e021d3f2181f89bb1d6db83fa4225a9f07f4bfec117d4a",
    "src/mc_photogrammetry/vision/__init__.py": "10a37a6abd7ba1f3ac3e58194a08500a9b01795206e62866d5e7f566871b0ce6",
    "src/mc_photogrammetry/vision/gui_mask.py": "1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9"
  },
  "study": "PLANE-PERIOD-SUPPORT-01",
  "targets": {
    "matches": [
      [
        "image-0000",
        2,
        1,
        0.25,
        "image-0001",
        3,
        1,
        0.125,
        3
      ],
      [
        "image-0000",
        2,
        1,
        0.5,
        "image-0002",
        2,
        1,
        0.25,
        1
      ],
      [
        "image-0001",
        3,
        1,
        3.0,
        "image-0002",
        3,
        1,
        0.125,
        2
      ]
    ],
    "readings": [
      [
        "image-0000",
        2
      ],
      [
        "image-0001",
        3
      ],
      [
        "image-0002",
        2
      ],
      [
        "image-0002",
        3
      ]
    ]
  }
}
```
