# UNIT-FACE-INPUT-01 generated RGB experiment

Preregistered before the first execution: recipe
`UNIT-FACE-INPUT-01-generated-v1` in `scripts/evaluation/run_unit_face_input.py`.
One 800 x 600 RGB scene family uses independently rendered Manhattan line
pencils in fixed side bands and a native 16 x 16 asymmetric Fourier texture
on a front plane. Renderer focal 700, yaw 23 degrees, pitch -14 degrees,
distance 3.23, phases .173/.287. The fixed controls alternate depth 3.23/4.6835
in 48-pixel strips or replace the texture with constant RGB 128. These are
generated test optics, consumed only by the renderer and post-seal evaluator.
No known camera, region, plane, distance or phase enters the adapter.

Policy is fixed before observations: independent default GUI-masked image
calibration; existing default region-growth proposals; select by descending
strength/area, then axis/side/mask hash; one centered bounding-box native crop
with each dimension capped at 256; original-parent 8-pixel checker roles;
20 x 20 rectification witness. Source and witness ceilings are lowerable;
fitting uses the unchanged accepted default bounded SearchPolicy, with only
work ceilings lowered by the adapter policy. No failed candidate retry.

All case outputs must be sealed as immutable JSON with SHA256 before the
evaluator checks generated truth. Calibration/proposal failure is a result;
no true-optics or manual-region fallback is permitted. Validation is a veto
conditional on frozen image-derived upstream context. There is no independent
camera holdout, optical uncertainty propagation, real-image accuracy claim,
placement, occupancy or export.

The recipe SHA256 is
`98227b8b880a1f8588841907acf5efc81a0d101e13f8e47d87b069cc6ea33cf0`.
It was printed before the first execution. The recipe was not revised after
observing calibration, proposals, numerical fits or evaluator coverage.

## Actual RGB execution

Every case completed actual independent RGB calibration and actual image-only
region growth, returning six proposals. No supplied segments, camera,
geometry or mask entered these stages. One proposal was selected before the
fitter ran. All three final outputs abstained; no representative was reported
and no held-out validation pixels were observed by the fitter.

| Case | Final refusal | Crop pixels | Region pixels | Generation pixels | Validation pixels |
| --- | --- | ---: | ---: | ---: | ---: |
| Identifiable native16 | `scale_or_phase_ambiguous` | 65,536 | 54,965 | 202 | 508 |
| Mixed depth | `insufficient_role_source_pixels` | 13,572 | 10,590 | 48 | 36 |
| Ambiguous constant native16 | `flat_or_uninformative_evidence` | 65,536 | 57,383 | 485 | 235 |

Every witness has 400 endpoint-inclusive sample locations. Complete stencils
that touch a crop's final row/column are rejected as borders because the
unchanged fitter requires neighboring rays for its footprint check. This fixed
rule was added after the first reporting execution exposed that existing
contract; it was not an appearance-dependent selection or gate change.

| Case | Retained samples | Depth rejected | Border rejected | Region/GUI rejected | Cross-role rejected | Full witness payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifiable | 184 | 0 | 133 | 40 | 43 | 23,539 |
| Mixed depth | 21 | 0 | 371 | 4 | 4 | 23,418 |
| Ambiguous | 188 | 0 | 125 | 31 | 56 | 23,359 |

The genuine identifiable calibration has focal 702.7527016071156, yaw
22.849999999999998 degrees, pitch -14.049999999999995 degrees, roll 0,
158 inliers / 182 segments, axis counts (75,12,71), median residual
.14656228885216688, and no measured principal-point displacement. Its native
crop is (x=363,y=136,width=256,height=256), principal point (37,164), with
the same focal and retained calibration counts. The ambiguous calibration
has the same optics, 154/167 inliers/segments, axis counts (72,14,68), residual
.1580159365931129; its crop is (324,136,256,256), principal point (76,164).

The mixed-depth RGB calibration is **wrong relative to the independent
renderer**, despite 146/234 inliers/segments: focal 1926.481635928744,
yaw 89.75 degrees, pitch .30000000000000016 degrees, principal point
(496,228), axis counts (37,69,40), residual .3098475482769747. The genuine
result is retained in the refusal diagnostic. No true-optics fallback is used.
Preparation then lacks the fixed minimum 48 validation source pixels and
never calls the fitter. This control does not prove a downstream mixed-depth
residual veto or reliable camera refusal.

Identifiable search completes its entire fixed compatible-template domain:
one compatible face / one unique geometry / 1,024 native RGBA bytes,
13,663 evaluations of the 100,000 ceiling and 2,759,926 actual sampled-pixel
visits of the 250,000,000 ceiling. Conservative estimated visits are
20,202,430. The final original objective admission threshold is
.007196227155517909 and the returned joint cover has 2,176 boxes.
Distance interval [3.037109375,3.501953125], phase-U arc
(start=.046875,width=.2734375), phase-V arc (start=.171875,width=.1875).
Independent generated distance and both phases lie in the returned marginal
enclosures **and a returned joint box**. The distance width exceeds
.10 * the lower endpoint and U-phase width exceeds .2, so the strict
accuracy gate refuses. There is no point estimate, point error or validation
execution to report. Domain completeness is conditional on computed optics
and the fixed forward model; it does not authenticate physical scale.

Ambiguous search refuses after observing generation RGB for flatness, before
any model evaluation: 0 evaluations/actual sampled-pixel visits, conservative
estimated visits 48,503,865, incomplete search, no cover or intervals. Mixed
depth has no numerical search. All validation observation hashes are empty.

All three original INPUT-01 diagnostic JSON outputs were sealed before any
evaluator coverage check. Their original SHA256 values, superseded by the
INPUT-01R tap-evidence seals below, are:

- Identifiable: `89f179abac426f4192aba32c62236bbcee4573d01f11feef3315cf1ec04ffcab`
- Mixed depth: `52cc2a0c1d54a99c4ada629bd1a4859a79086e9611ef094d791018cd2ba180c2`
- Ambiguous: `b829188467b7ac6a0b5b40d447220a85e6fb9d7b663fc6b9754f59a5ccc24b06`

Identifiable parent-frame binding is
`c66b632232dab88e0c688615056d2e66471ff264cbd5197522d1f5ad4cd49870`;
selected proposal hash
`0e04265c91bf2060da69f8cd28bf07fc93195f937d4338c136a8797472784ddf`;
preparation binding
`025d43131a9497461ac1acdfe7b305b524b44d3d4a9b9d4ef59ef65fdedb052c`;
replayed support hash
`1bcf7a5427fcb687aeaeac7a3b8e3dbbb24dfb6716076d342e7befd53209fefc`.
Fixed adapter policy hash is
`0241af6b4e92d4d0a56ed12570ebb79bf66648e51282747687e85ce69a4b2f14`.

## Verification and scope

Focused check before the final assertion refinements: 45 passed in 29.85s.
Final focused/regression check, exit 0:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
.venv/Scripts/python.exe -B -m pytest tests/test_image_derived_unit_face.py tests/test_unit_face_estimator.py tests/test_unit_face_lattice.py tests/test_calibration.py tests/test_region_growth.py -o addopts= -p no:cacheprovider -q --no-header
265 passed in 254.37s (0:04:14)
```

The 45 new tests check native crop/ray equivalence, off-center translation and
unchanged focal/counts; holes/default GUI exclusion; complete tap replay,
role/source disjointness and integer parent-pixel ownership; deterministic
appearance-independent selection; immutable snapshots; stale/forged source,
calibration, proposal, crop, context, support, policy/order and rehash controls;
frame/crop/proposal/role/witness-sample/tap/dependency/payload caps;
root/pixel/search work refusal with no certificate; failed upstream/preparation
never calling the fitter; rejected manual/evaluator arguments; all output seals
preceding truth evaluation; and the actual RGB scene/control path.

The frozen-native16 isolation check is non-vacuous: both searches complete,
generation-ranked candidate count is nonzero and unchanged, retained joint
cover and actual work match, generation-owned RGB is unchanged, validation RGB
changes, generation hashes/parameters/exposure/rank match exactly, and both
validation hashes are empty. Its ambiguity gate precedes validation, so it
proves generation noninterference under frozen context, not execution of a
validation veto. The unchanged accepted estimator regression
`test_heldout_mutation_changes_veto_only_generation_rank_and_fit_are_frozen`
separately checks a supported fit and validation veto.

Two earlier isolation assertions incorrectly demanded support from a small
visible crop (first native16, then a separate native4 stimulus); both honestly
refused ambiguity. The first partial run was 43 passed/1 failed/1 deselected
in 4.04s; the second full run was 44 passed/1 failed in 29.12s. Lead/Astra
reviewed the corrected, narrow completed-search assertion. The alternative
stimulus was removed, and no scene/threshold was changed to achieve support.
The initial pre-border reporting invocation refused `border_source_evidence`
for identifiable/ambiguous and insufficient support for mixed depth.

AST/explicit whitespace scans and scoped `git diff --check` pass. New paths
are untracked, so explicit byte scans are the operative whitespace check:
zero tabs/trailing spaces, final LF. Original INPUT-01 source/test/evaluator
pins (revision pins are below):

- Adapter (377 lines, 17,517 bytes): `912d70b21ca24f5277391766bf92a76700135db8fc0216e44e40f52520163307`
- Tests (328 lines, 17,950 bytes): `b75fec7be3a0eb29db048e00930ad6704cafd5cdf85a45a9f421c8ce3c06038c`
- Evaluator (196 lines, 10,019 bytes): `75424907ab3d37b2bfe18a8767765e84b491192aa1e97e48f9cd6290157c0924`

The accepted fitter and its tests remain byte-for-byte pinned respectively at
`6d428b3f84c60b7bcd6f5ec44deb6a515d9f77b78298ed9e15592896c1285c25`
and `de494d0e8ee02a4c38990476eb599a6404bc7e64224882248e32a37bd728ad07`.

Limits: upstream calibration/proposals consume the whole GUI-masked image,
including the subsequently named validation role. This is conditional held-out
fitting evidence, not independent optical/proposal selection evidence. No
calibration uncertainty or horizontal-axis naming ambiguity is resolved.
Selected-region/crop truncation can remove useful evidence; there is no
retry/rerank using fitter appearance. Bindings/replay detect stale snapshots;
they do not cryptographically authenticate an adversarial fabrication of every
private preparation record. Caps bound declared retained payload/operation
work, not RSS or wall time. The point-sampled periodic bilinear model does not
establish Minecraft lighting/mipmap/shader fidelity or arbitrary contamination
detection. No unique unit fit succeeded on this scene family. No real input,
JAR, screenshot, evaluator fixture, saved mcscan ownership, client/network,
production integration, graph/XYZ/occupancy, placement or export was used.
Only the four assigned new paths and an AGENT_COMMS append were written.
Stopped for independent Astra review; no readiness/promotion claim.

## INPUT-01R evaluation hardening

This revision changes only the focused tests, private evaluator and report.
The adapter, recipe, rendered RGB, selection rule, crop, source partition,
search/accuracy policy and accepted fitter remain unchanged. The actual
identifiable frame binding stays
`c66b632232dab88e0c688615056d2e66471ff264cbd5197522d1f5ad4cd49870`.

Each prepared diagnostic now seals both roles' complete sorted original crop
and parent pixel indices, their actual partition role values, every retained
tap sample and every dependency group. The post-seal evaluator checks the
integer crop-to-parent transform, disjoint roles, complete sample/group closure
and equality of tap unions to the reported owned pixels. Only then does it
derive independent renderer target labels and require **every** generation and
validation parent tap to belong to the intended face. It never uses mere crop
bbox overlap. These evaluator labels never enter calibration, proposals, crop
selection, roles, fitting or ranking.

Scoring also requires target normal axis 2 / side +1, ascending in-plane axes
(0,1), the world-axis block-center modulo-one phase convention, and compatible
signed axis naming against the independent renderer gauge. Signed correspondence
uses the largest absolute axis dot product and its positive sign; it checks
axis identity, not exact optics or optical uncertainty. Recipe binding is
validated before scoring. Any mismatched source, region/crop, axis/side, phase
convention or recipe receives unavailable truth scoring with a stable reason.

Mixed depth is explicitly `not_applicable`, reason
`mixed_depth_has_no_single_plane_truth`, **even if an attempted diagnostic
contains a complete cover and point estimate**. Constant ambiguous texture is
also `not_applicable`, reason
`ambiguous_scene_has_no_identifiable_unit_anchor`. Neither case calls the
single-plane scorer or reads single-plane renderer labels. There are no scalar
distance/phase errors or coverage results for these controls.

The actual identifiable-case test now requires preparation and fitter reach;
completed domain/search; nonempty candidates and joint cover; positive search
and pixel work; independently verified ownership; marginal **and joint** truth
containment; only `scale_or_phase_ambiguous`; distance width at least .10 times
its lower endpoint and U-phase width at least .2; no representative, validation
observation or candidate validation residual. An earlier preparation/search
regression cannot silently remain green. The strict accuracy rule is preserved.

The final unchanged-recipe run again returns exactly the prior counts,
calibrations, work, cover and refusals. Identifiable ownership verifies all
202 generation and 508 validation parent source pixels across 184 retained
samples and 184 complete dependency groups. Its coverage is evaluated under
`identifiable_single_plane_target`, distance/both-phase/joint covered; point
errors remain unavailable because ambiguity prevents a representative.
Mixed depth has no prepared support or search; ambiguous seals 188 samples
and groups but receives no single-plane scoring. All fitter validation hashes
remain empty; no unique fit or validation execution is newly claimed.

Current diagnostic seals:

- Identifiable: `febc841c897d027291f337b8b24e38d72d8af102bbbbaa5d4a0996bbc480d7c6`
- Mixed depth: `52cc2a0c1d54a99c4ada629bd1a4859a79086e9611ef094d791018cd2ba180c2`
- Ambiguous: `521b8b2e7288ac40edfbbc1de6186865778193be2bfb9f0e3c4d27d7d073f69e`

Focused revision check: **58 passed in 29.71s**, exit 0. Final same-five-module
focused/regression command above: **278 passed in 254.40s (0:04:14)**, exit 0,
session 21645. The 13 added tests cover actual tap/role ownership, control
scope with forged complete covers, and wrong crop with target-bbox overlap,
parent-index binding, axis/side, phase-axis order/origin/signed orientation,
role values, tap closure and recipe binding. The seal-order test now instruments
the actual independent renderer-label reader and requires all three outputs
sealed before any label access. No additional implementation changes followed
that final regression. Reporting invocation session 92630 exited 0.

Final AST/whitespace/scoped diff QA passes. Current pins:

- Adapter unchanged (377 lines, 17,517 bytes): `912d70b21ca24f5277391766bf92a76700135db8fc0216e44e40f52520163307`
- Revised tests (448 lines, 25,055 bytes): `a1c7c23d5c0d7ce03b134169de271ee3df67e1ff5e7372ba07585b4a857cb146`
- Revised evaluator (330 lines, 17,712 bytes): `52ea25ead4c0862230b9a72d807e0f5baa170dfbfd8c418f78d353504ec08832`

All earlier conditional-inference, calibration, physical-model, budget and
no-placement limitations remain. Source ownership here is generated renderer
label evidence for this fixed scene family, not a real-image surface oracle or
cryptographic source-origin certificate. No source/evaluator payload, scene
recipe, calibration fallback, thresholds, candidate reranking or production
path was introduced. Stopped for independent Astra review of INPUT-01R.
