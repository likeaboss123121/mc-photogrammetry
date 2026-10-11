# UNIT-FACE-DENSITY-01 paired density ablation

Preregistered before either arm execution. The inspected INPUT-01 v1 RGB scene
is development evidence only, with its unchanged constant/mixed-depth controls.
One disjoint heldout identity is `heldout-v2-identifiable`: 800 x 600, focal730,
yaw31, pitch-17, distance3.61, phases(.337,.619), native16 asymmetric Fourier
texture with NEW fixed coefficients/phases, new line anchors(25,85,160,220,695,775)
and rows(100,155,210,265,320,375,430,485). The intended face rectangle stays
(270,90)-(650,480). Full immutable recipe JSON is defined in the new harness.
The heldout pair will execute once; no unit test executes either heldout arm.

Arm A is the original20x20 support. Arm B retains ALL A support and adds EVERY
eligible generation-only complete stencil of a fixed endpoint-inclusive40x40
grid over the SAME inferred plane window. Native source pixels deduplicate
exactly; validation source pixels stay identical. Calibration/proposals/crop,
RGB, optics, catalogue, partition, exposure/residual/uncertainty policy and
budgets stay fixed. No top-K, validation additions, re-proposal or candidate
retry. Both supports and their manifest seal precede either numerical fit;
both fit outputs seal before any evaluator labels/truth.

Astra-approved preparation instrumentation is single-threaded and temporary:
delegate genuine accepted `_fit_prepared`, intercept its actual fitter arguments
once, raise only a private BaseException sentinel, restore both accepted
callables in finally, and propagate all other exceptions. No fit occurs during
capture. Ordinary preparation refusal stays a result; known optics never replace
it. Each complete arm then calls the restored unchanged fitter once.

Primary falsifiable success: dense B meets the existing strict width gates,
retains joint independent truth and passes the identical validation pixels.
Otherwise report negative/inconclusive; a narrower failing cover is descriptive.
If A and B both succeed, report both successes; this is not evidence that
density caused a recovery. The success gate itself remains unchanged.
Mixed-depth control has no single-plane truth score. Validation noninterference
is conditional on frozen upstream context; no placement/generalization claim.

Preregistration SHA256:
`87f0e54c8cd721c625c9ca59bc86e4cb3a24bddbd19083bf4449c9df27f1b0d3`.
Heldout scene/recipe SHA256:
`2e7d8d9f57c9ffcd8837977ee3c4b35e50cdc21a3530d9df78f73aafed79559c`.
These pins were printed/recorded before either arm ran. This expanded
preregistration supersedes the earlier expanded pre-execution pin
`8afc5d4890639f52e7795d2e4e7fbeb18a04e59d4e1e7ea9fbc976466c261aa7`
and provisional pre-execution pin
`cb76458b22056e202800f48c454800c649462fd3a1bf152258a52da8b5b0a9d0`.
The scene recipe is unchanged. The new pin binds the exact support/evaluator
function sources, accepted dependency hashes, complete ContractLimits and
SearchPolicy values, strict distance width/lower < .10 and circular phase
width < .2 success rules, joint truth/axis/side/gauge/source ownership,
nonempty identical validation, durable one-shot execution rule, full sealed
pair evidence append/replay sources, evidence filename/format/order, and
exercised local dependency closure plus runtime versions below.
The recipe, support rules and fixed policy will not be revised after evaluator
access.

HELDOUT_EXECUTION_STATE: COMPLETE

The harness must exclusively lock this report and durably replace NOT_STARTED
with RUNNING before rendering or fitting any study arm. COMPLETE or FAILED
then retains the result; RUNNING after a crash also refuses rerun. No other
marker artifact is written. Unit tests exercise pure state transitions and a
mocked claimed-run refusal, without claiming or running the actual study.

Append-only evidence path:
`docs/reports/UNIT-FACE-DENSITY-01-evidence-2026-10-05.jsonl`.
Actual evidence status is **ABSENT, expected before first study claim**;
there is no evidence artifact hash or measured sealed pair yet. Existing
evidence refuses execution before the report claim. After the durable report
claim, exclusive creation writes/fsyncs a preregistration-bound JSONL header.
Immediately after each `seal_pair`, the harness appends the full exact
`sealed_pair_payload_json`, its SHA256, scene identity and preregistration pin,
flushes/fsyncs, then replays the complete retained prefix before evaluation.
Missing, truncated, mismatched or tampered evidence refuses evaluation. Earlier
sealed pairs remain if a later scene/evaluator fails. COMPLETE/FAILED report
records include the artifact SHA256 and retained pair hashes. Nothing truncates
or reuses the evidence artifact.

The local closure follows the exercised adapter imports: calibration uses
camera for its shared axis basis; camera imports pipeline contracts and domain
models; region_growth uses boundary_candidates; GUI exclusion and native
catalogue/renderer paths are pinned separately. Uncalled boundary_candidates
`propose_image_planes`/surfaces and unrelated package exports are not part of
the exercised numerical/proposal path. Exact dependency source pins:

| Source | SHA256 |
| --- | --- |
| scripts/evaluation/run_unit_face_input.py | 52ea25ead4c0862230b9a72d807e0f5baa170dfbfd8c418f78d353504ec08832 |
| src/mc_photogrammetry/blocks/unit_face_catalogue.py | ab828d782c7a7d5b83f66968b5900e392263c7839ea6569f8b3e8b9c81eda33d |
| src/mc_photogrammetry/domain/models.py | 6a87d543661cc62ada9a154bcdaa089ca844f41691ff3b53207ad89e3f05a478 |
| src/mc_photogrammetry/pipeline/contracts.py | db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc |
| src/mc_photogrammetry/pipeline/image_derived_unit_face.py | 912d70b21ca24f5277391766bf92a76700135db8fc0216e44e40f52520163307 |
| src/mc_photogrammetry/reconstruction/boundary_candidates.py | 95e550a44bd65a19044b2a35a99d9a503b389d066085dccb464b0f18fa89216c |
| src/mc_photogrammetry/reconstruction/calibration.py | 2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc |
| src/mc_photogrammetry/reconstruction/camera.py | 0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680 |
| src/mc_photogrammetry/reconstruction/lattice.py | 0e16c2231c511dc830b3cf8efca38a9c410a634052b39a696cb587a66ecc747c |
| src/mc_photogrammetry/reconstruction/region_growth.py | 5375bc247defa370baee530434ef9751d9aec8a477ad143599d8fadc5b007e8a |
| src/mc_photogrammetry/reconstruction/unit_face_estimator.py | 6d428b3f84c60b7bcd6f5ec44deb6a515d9f77b78298ed9e15592896c1285c25 |
| src/mc_photogrammetry/reconstruction/unit_face_lattice.py | f0be670be5dcb9921342fe028cbe8203e90ae29ab057a37ca645133e52c84622 |
| src/mc_photogrammetry/vision/gui_mask.py | 1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9 |

Runtime pins: Python `3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17)
[MSC v.1929 64 bit (AMD64)]`; NumPy `2.2.6`; OpenCV `5.0.0`; Pillow `12.3.0`.
Pure tests mutate each newly required upstream source and each runtime version
without writing files and verify rejection before rendering or claiming the
study. Evidence tests use only an in-memory artifact, real prepared records,
fitter stubs and evaluator spies; no actual evidence file, fit or truth score
is produced by these tests.

DeepSeek review changes, all before any numerical fit: integrity checks now
raise typed errors explicitly and survive Python -O; source proofs replay and
actual manifest/context/support/policy/callable bindings are checked; dense
tap indices include the source upper bound; controls seal their actual case;
baseline witness sample, worst-case tap and payload consumption is deducted
before extension. Pure tests include independent heldout rendering at native16
and native8, real prepared nonempty same-validation evidence, and a refused B
arm that never calls the fitter. The heldout rendering test does not fit or
score truth.

No density arm or evaluator truth scoring has executed. Numerical development
validation-mutation testing and the one-shot paired study remain pending
parent re-review clearance. No-fit focused verification:
`python -B -m pytest tests/test_unit_face_density.py -k 'not numerical_validation' -o addopts= -p no:cacheprovider -q`
with PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src produced **51 passed,
1 deselected in 8.41s**, exit 0. The deselected test is the real numerical
development validation-mutation check. The preceding no-fit run had one
JSON-list/dataclass-tuple assertion mismatch (28 passed, 1 deselected); only
that test comparison was normalized before this passing rerun. No harness
policy or scene changed in response to an arm result.


Durable one-shot result:

```json
{"evidence_path":"UNIT-FACE-DENSITY-01-evidence-2026-10-05.jsonl","evidence_sha256":"c5474fc06f3da7baadece34387516b6298f520ef2740acc33fa287fc956a5415","generalization_claim":false,"placement_supported":false,"preregistration_sha256":"87f0e54c8cd721c625c9ca59bc86e4cb3a24bddbd19083bf4449c9df27f1b0d3","scenes":[{"arms":{"A":{"alias_count":1,"candidate_count":1,"counts":{"crop_pixels":65536,"generation_pixels":202,"proposals":6,"region_pixels":54965,"rejected_border":133,"rejected_cross_role":43,"rejected_depth":0,"rejected_excluded":40,"retained_samples":184,"validation_pixels":508,"witness_bytes":23539,"witness_samples":400},"cover_geometries":["d76441d67be0a11b84b018b95744e410735c9cffe7df9c5ad67d335473209ca4"],"fit":{"actual_pixel_work":2759926,"compatible_faces":1,"distance_interval":[3.037109375,3.501953125],"domain_search_complete":true,"estimated_pixel_work":20202430,"phase_intervals":[[0.046875,0.2734375],[0.171875,0.1875]],"pixel_work_budget":250000000,"policy_sha256":"7727cae0fe2cce72da39a20e6cfdd6fc92b2f054c8951e9e1244ff4f8d1b2223","retained_boxes":2176,"search_complete":true,"source_generation_sha256":"74e4f71761d1540d881f383f19b943091e9c995b2aa547b045549cd6c91fd7be","source_validation_sha256":"","status":"abstained","unique_templates":1,"work_budget":100000,"work_used":13663},"reason":"scale_or_phase_ambiguous","seconds":3.480642900103703,"support_sha256":"e47572ce464fe5a5ad1c218424f9fa2cecdf072960e22121b423ef9180930663"},"B":{"alias_count":1,"candidate_count":1,"counts":{"crop_pixels":65536,"generation_pixels":1708,"proposals":6,"region_pixels":54965,"rejected_border":133,"rejected_cross_role":43,"rejected_depth":0,"rejected_excluded":40,"retained_samples":573,"validation_pixels":508,"witness_bytes":113929,"witness_samples":2000},"cover_geometries":["d76441d67be0a11b84b018b95744e410735c9cffe7df9c5ad67d335473209ca4"],"fit":{"actual_pixel_work":18639404,"compatible_faces":1,"distance_interval":[3.064453125,3.447265625],"domain_search_complete":true,"estimated_pixel_work":170812972,"phase_intervals":[[0.0625,0.2265625],[0.1953125,0.1640625]],"pixel_work_budget":250000000,"policy_sha256":"7727cae0fe2cce72da39a20e6cfdd6fc92b2f054c8951e9e1244ff4f8d1b2223","retained_boxes":1802,"search_complete":true,"source_generation_sha256":"89f11f78528836e4cb10c49feaab820b55e3003986a5e566d254f59498403397","source_validation_sha256":"","status":"abstained","unique_templates":1,"work_budget":100000,"work_used":10913},"reason":"scale_or_phase_ambiguous","seconds":5.5298490999266505,"support_sha256":"5d609bb0beb3881b9bce82a35b1947cc53c461dc14c5c7117adde688792ec201"}},"dense_counts":{"added_generation_pixels":1506,"baseline_witness_samples":400,"baseline_worst_case_taps":1600,"eligible_generation_stencils":389,"omitted_validation":386,"planned_generation_pixels":1708,"reason":"","rejected_border":483,"rejected_cross_role":168,"rejected_depth":0,"rejected_excluded":174,"remaining_witness_bytes":1025037,"remaining_witness_samples":3696,"remaining_witness_taps":14784,"status":"complete","window":[-0.621606092050449,0.3873124819785499,0.27980723866684365,1.0]},"evaluation":{"arms":{"A":{"distance_covered":true,"joint_covered":true,"ownership_verified":true,"phase_arcs":[[0.046875,0.2734375],[0.171875,0.1875]],"phase_covered":[true,true],"primary_success":false,"reason":"","relative_distance_width":0.15305466237942122,"status":"evaluated"},"B":{"distance_covered":true,"joint_covered":true,"ownership_verified":true,"phase_arcs":[[0.0625,0.2265625],[0.1953125,0.1640625]],"phase_covered":[true,true],"primary_success":false,"reason":"","relative_distance_width":0.12492033142128744,"status":"evaluated"}},"primary_success":false,"reason":"","same_validation":true,"status":"evaluated"},"manifest_sha256":"3224691b580f1cccafee786361aee7f1f5df13a783b15bb052071021113b02b1","preparation_seconds":5.958858099998906,"scene":{"case":"identifiable","identity":"development-v1-identifiable","purpose":"development","recipe_json":"{\"controls\":[\"identifiable\",\"mixed_depth\",\"ambiguous\"],\"distance\":3.23,\"face_rectangle\":[270,90,650,480],\"line_anchors\":[30,100,180,230,700,760],\"line_rows\":[95,145,195,245,295,345,395,445,495],\"native_side\":16,\"phase\":[0.173,0.287],\"renderer_focal\":700.0,\"renderer_pitch\":-14.0,\"renderer_yaw\":23.0,\"shape\":[600,800],\"structure_band_edges\":[260,660],\"version\":\"UNIT-FACE-INPUT-01-generated-v1\"}"},"scene_recipe_sha256":"c5380e0b8f02ca43ec4241db7e93d2721dc3171eb607d9ef2d805aebfb8ac9d0","sealed_pair_sha256":"680c3ca6cf4873fe76a86ac1edcfd80d685454a869d6619326601c6c82bf9dce"},{"arms":{"A":{"alias_count":0,"candidate_count":0,"counts":{"crop_pixels":65536,"generation_pixels":485,"proposals":6,"region_pixels":57383,"rejected_border":125,"rejected_cross_role":56,"rejected_depth":0,"rejected_excluded":31,"retained_samples":188,"validation_pixels":235,"witness_bytes":23359,"witness_samples":400},"cover_geometries":[],"fit":{"actual_pixel_work":0,"compatible_faces":1,"distance_interval":null,"domain_search_complete":false,"estimated_pixel_work":48503865,"phase_intervals":null,"pixel_work_budget":250000000,"policy_sha256":"7727cae0fe2cce72da39a20e6cfdd6fc92b2f054c8951e9e1244ff4f8d1b2223","retained_boxes":0,"search_complete":false,"source_generation_sha256":"86ea110a5ada361e755ffb72731f72b6d9a32431c20b2cfc5775930a20c2f248","source_validation_sha256":"","status":"abstained","unique_templates":1,"work_budget":100000,"work_used":0},"reason":"flat_or_uninformative_evidence","seconds":0.018048600060865283,"support_sha256":"1ce0821feba64f90fe221fe994a6544a8e04e65eabf285edc762c1d96346cd4a"},"B":{"alias_count":0,"candidate_count":0,"counts":{"crop_pixels":65536,"generation_pixels":1967,"proposals":6,"region_pixels":57383,"rejected_border":125,"rejected_cross_role":56,"rejected_depth":0,"rejected_excluded":31,"retained_samples":578,"validation_pixels":235,"witness_bytes":113429,"witness_samples":2000},"cover_geometries":[],"fit":{"actual_pixel_work":0,"compatible_faces":1,"distance_interval":null,"domain_search_complete":false,"estimated_pixel_work":196714239,"phase_intervals":null,"pixel_work_budget":250000000,"policy_sha256":"7727cae0fe2cce72da39a20e6cfdd6fc92b2f054c8951e9e1244ff4f8d1b2223","retained_boxes":0,"search_complete":false,"source_generation_sha256":"0c08aaf42694d658cf42ec64052822fde0503a06bf94a692e09232deeabe4675","source_validation_sha256":"","status":"abstained","unique_templates":1,"work_budget":100000,"work_used":0},"reason":"flat_or_uninformative_evidence","seconds":0.040055799996480346,"support_sha256":"ff2b1540c738e998b43a92e48a26e36744990220f6670b77362790ec75582469"}},"dense_counts":{"added_generation_pixels":1482,"baseline_witness_samples":400,"baseline_worst_case_taps":1600,"eligible_generation_stencils":390,"omitted_validation":394,"planned_generation_pixels":1967,"reason":"","rejected_border":470,"rejected_cross_role":203,"rejected_depth":0,"rejected_excluded":143,"remaining_witness_bytes":1025217,"remaining_witness_samples":3696,"remaining_witness_taps":14784,"status":"complete","window":[-0.5410372140403386,0.3764908800056799,0.2678343805957186,1.0]},"evaluation":{"arms":null,"primary_success":false,"reason":"constant_texture_control","status":"not_applicable"},"manifest_sha256":"af44db63de40dac47f935fe99960175b9dbf7da988608edf4c6f014e97e00f80","preparation_seconds":5.891017299843952,"scene":{"case":"ambiguous","identity":"development-v1-ambiguous","purpose":"development","recipe_json":"{\"controls\":[\"identifiable\",\"mixed_depth\",\"ambiguous\"],\"distance\":3.23,\"face_rectangle\":[270,90,650,480],\"line_anchors\":[30,100,180,230,700,760],\"line_rows\":[95,145,195,245,295,345,395,445,495],\"native_side\":16,\"phase\":[0.173,0.287],\"renderer_focal\":700.0,\"renderer_pitch\":-14.0,\"renderer_yaw\":23.0,\"shape\":[600,800],\"structure_band_edges\":[260,660],\"version\":\"UNIT-FACE-INPUT-01-generated-v1\"}"},"scene_recipe_sha256":"f7e98fe527331f3130a033befca3d0f9ba12fe303ea967ae012542c8114cb341","sealed_pair_sha256":"dcfecd787e387c6f389490c0fee7b03579387ca5cd21bed2b594b0085281d2ae"},{"arms":{"A":{"alias_count":0,"candidate_count":0,"counts":null,"cover_geometries":[],"fit":null,"reason":"insufficient_role_source_pixels","seconds":0.0,"support_sha256":null},"B":{"alias_count":0,"candidate_count":0,"counts":null,"cover_geometries":[],"fit":null,"reason":"insufficient_role_source_pixels","seconds":0.0,"support_sha256":null}},"dense_counts":null,"evaluation":{"arms":null,"primary_success":false,"reason":"mixed_depth_has_no_single_plane_truth","status":"not_applicable"},"manifest_sha256":"20c2cf562c939dc1e0ddda1aae29d737b4485790b560c6c45e64e172daac4200","preparation_seconds":6.321418999927118,"scene":{"case":"mixed_depth","identity":"development-v1-mixed_depth","purpose":"development","recipe_json":"{\"controls\":[\"identifiable\",\"mixed_depth\",\"ambiguous\"],\"distance\":3.23,\"face_rectangle\":[270,90,650,480],\"line_anchors\":[30,100,180,230,700,760],\"line_rows\":[95,145,195,245,295,345,395,445,495],\"native_side\":16,\"phase\":[0.173,0.287],\"renderer_focal\":700.0,\"renderer_pitch\":-14.0,\"renderer_yaw\":23.0,\"shape\":[600,800],\"structure_band_edges\":[260,660],\"version\":\"UNIT-FACE-INPUT-01-generated-v1\"}"},"scene_recipe_sha256":"9e417501a855a25d1ae1778e79ff17a03750eb548e71fff18613d24c9663be2e","sealed_pair_sha256":"08d0ad0dd20d6ac63d755b57169522e81df106d48932157ef070c5542554eb4e"},{"arms":{"A":{"alias_count":1,"candidate_count":1,"counts":{"crop_pixels":65536,"generation_pixels":416,"proposals":9,"region_pixels":60607,"rejected_border":76,"rejected_cross_role":99,"rejected_depth":0,"rejected_excluded":22,"retained_samples":203,"validation_pixels":378,"witness_bytes":23639,"witness_samples":400},"cover_geometries":[],"fit":{"actual_pixel_work":1302496,"compatible_faces":1,"distance_interval":null,"domain_search_complete":true,"estimated_pixel_work":41603668,"phase_intervals":null,"pixel_work_budget":250000000,"policy_sha256":"7727cae0fe2cce72da39a20e6cfdd6fc92b2f054c8951e9e1244ff4f8d1b2223","retained_boxes":0,"search_complete":true,"source_generation_sha256":"d1e4da72b931571a321a216980f9250eb6cb89762129941fb622ed3e1eb53078","source_validation_sha256":"","status":"abstained","unique_templates":1,"work_budget":100000,"work_used":3131},"reason":"generation_residual_failure","seconds":0.8984749000519514,"support_sha256":"1bc3e7b7d2af97e0644c6dc913bc923c3f93afd72d4686596de3d5926abf80a5"},"B":{"alias_count":0,"candidate_count":0,"counts":null,"cover_geometries":[],"fit":null,"reason":"density_role_source_pixel_cap","seconds":0.0,"support_sha256":null}},"dense_counts":{"added_generation_pixels":1857,"baseline_witness_samples":400,"baseline_worst_case_taps":1600,"eligible_generation_stencils":488,"omitted_validation":485,"planned_generation_pixels":2273,"reason":"density_role_source_pixel_cap","rejected_border":231,"rejected_cross_role":295,"rejected_depth":0,"rejected_excluded":101,"remaining_witness_bytes":1024937,"remaining_witness_samples":3696,"remaining_witness_taps":14784,"status":"refused","window":[-0.03688152672015624,0.044836008198714646,0.051733108778192594,1.0]},"evaluation":{"arms":{"A":{"distance_covered":null,"joint_covered":null,"ownership_verified":false,"phase_arcs":null,"phase_covered":null,"primary_success":false,"reason":"target_phase_gauge_mismatch","relative_distance_width":null,"status":"unavailable"},"B":{"distance_covered":null,"joint_covered":null,"ownership_verified":false,"phase_arcs":null,"phase_covered":null,"primary_success":false,"reason":"prepared_fit_unavailable","relative_distance_width":null,"status":"unavailable"}},"primary_success":false,"reason":"","same_validation":false,"status":"evaluated"},"manifest_sha256":"254b061eb82969bc09e21b8f81dbb03400b523a334b4740fce604f963c9ec508","preparation_seconds":5.5237759000156075,"scene":{"case":"identifiable","identity":"heldout-v2-identifiable","purpose":"heldout","recipe_json":"{\"distance\":3.61,\"face_rectangle\":[270,90,650,480],\"fourier\":[[54.0,0.91,31.0,-0.4,14.0,0.28],[47.0,-0.8,37.0,0.51,18.0,0.61],[44.0,-0.35,40.0,0.2,19.0,0.49]],\"line_anchors\":[25,85,160,220,695,775],\"line_rows\":[100,155,210,265,320,375,430,485],\"native_side\":16,\"phase\":[0.337,0.619],\"renderer_focal\":730.0,\"renderer_pitch\":-17.0,\"renderer_yaw\":31.0,\"shape\":[600,800],\"structure_band_edges\":[260,660],\"version\":\"UNIT-FACE-DENSITY-01-heldout-v2\"}"},"scene_recipe_sha256":"2e7d8d9f57c9ffcd8837977ee3c4b35e50cdc21a3530d9df78f73aafed79559c","sealed_pair_sha256":"050bf14940e3079f1a3db5b55b593a55f6b0dcf4428d2296bc9fbb2e395b2e32"}],"sealed_pairs":["680c3ca6cf4873fe76a86ac1edcfd80d685454a869d6619326601c6c82bf9dce","dcfecd787e387c6f389490c0fee7b03579387ca5cd21bed2b594b0085281d2ae","08d0ad0dd20d6ac63d755b57169522e81df106d48932157ef070c5542554eb4e","050bf14940e3079f1a3db5b55b593a55f6b0dcf4428d2296bc9fbb2e395b2e32"]}
```
