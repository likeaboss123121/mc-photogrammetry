SOL-ENDPOINT-WITNESS-ROOT-CAUSE-01 ? source trace, 2026-10-08

Disposition: no production or test patch. No exact bookkeeping, endpoint identity, canonical direction, Q orientation, selected alternative, or distance-conversion defect is demonstrated by the pinned artifact/source. The model correctly finds the two captured same-plane assertions inconsistent. Whether those assertions are true physical associations, or whether the right reading contains several physical surfaces, remains unobserved. Changing mandatory association semantics or adding unmeasured plane offsets would require a separate design, not this bounded bug repair.

Authority is the amended task card SHA256 `5A5326B4087DF0A5385B2988847B37BBED230FBEE852F53AC9B5D228001C258E`. The original task SHA `7C2415E5FFC093AF4612412F66C20F7973848B3C40484C2A605D996F96EAF467` was verified first. Endpoint factor construction was initially outside its three-source list; the lead expressly added endpoint_hypothesis.py, whose pin was verified before inspection. All nine amended artifact/source/test pins matched and were rechecked before this report. No project modules were imported or executed.

The captured diagnostic SHA256 is `FB474C15C6CFEFF691B52C8DCB2A1F9E1F2FADF600718FC3F94C54E37A368C49`. Its strict canonical JSON digests independently recomputed from stored JSON are outer `6b31ce23689639c7084574b51467841666c02ce0d7d6e9c0240977bed22d5e78` and nested joint observation `3a619fbdd806afd0817dda0b01a4fd0ac405f5796649cbc1c3f051bbd739b678`. The authorized result JSON agrees with the actual outer digest. The earlier 22:28 AGENT_COMMS prose transcribed the outer digest incorrectly; use the artifact/result values above. This is a reporting transcription, not a production identity defect.

Exact witness identity

Universe SHA256 `77afc4fa8a1d0466b055cf029ea347cb4ba8fa4c1dbc92b3bb3e6f603478143b`; proof SHA256 `263f104a5d14299679c0454e2862896d4a1209e0dd1b274fe85b79c1947befcd`; source digest `9d33ca1ce4cbdbbf3dd4cdfeca80d5570cca74f4eab0273a0a24fb7170f2cae5`. These bind this original row universe only. Partial exclusion index 0, assignment index 0, quadrants image-0000=0/image-0001=0. Both rows have local axes [1,1], mapped world axis Y=1, selected b [1,1], original and canonical alternative indices [0,0], and unchanged original/canonical frame direction. Their original signature hashes exactly agree with the joined reading alternative hashes in the stored projection.

| Original row | Original/canonical endpoint direction | Row SHA256 | Signed delta |
| --- | --- | --- | --- |
| 176 | image-0000/2 -> image-0001/1 | `5a4a41f9bfb0e6bb4d569d8a2a84b084355605f7cbfd123a7403c9aa199d5751` | -122.53593922433798 |
| 197 | image-0000/3 -> image-0001/1 | `ed3b5869fd56d09c25e0a4ecf547dab49e32535c56a061045d4e9d4be6796028` | -19.50616981305791 |

The first event retains separate forced factors `[2,2,0,7,0,[176]]` and `[3,3,0,7,0,[197]]`, plus a third forced factor with source row 250. The factor IDs are 2 and 3; the second/fourth fields are ordered endpoint-variable positions, not portable row/reading IDs. They represent different left endpoints sharing one right endpoint. No deduplication merges the two left readings into one endpoint. The right endpoint's global reading index is 7 while its per-frame canonical plane index is 1; these are explicitly different index spaces.

| Joined endpoint | Global reading index | Selected distance | Period pixels | Extent | Samples | Retained distance roster |
| --- | --- | --- | --- | --- | --- | --- |
| ['image-0000', 2] | 2 | 146.15787988715914 | 9.056887606054975 | 0.19301439128951037 | 512 | 146.15787988715914, 73.07893994357957, 36.539469971789785, 438.47363966147736 |
| ['image-0000', 3] | 3 | 43.12811047587907 | 34.277272239249015 | 0.1728320253847461 | 512 | 43.12811047587907 |
| ['image-0001', 1] | 7 | 23.62194066282116 | 7.588774437883119 | 1.4252914693041312 | 512 | 23.62194066282116, 11.81097033141058, 5.90548516570529 |

All three sides are -1. Their reading-binding hashes are, in table order, `0a2623668586ebf2aee8348fef301a1b160d112eaa61da7156586dbf67d12478`, `260d8932adf83a47a3d9e7b06901487fa5843cb498735e80ac0cd0c4bffb167a`, and `00b86d2dae51fef6e959f85791a41326442a0bd51bcb1c8e532dc3044d405722`. The left source-region IDs are different (`338a47a2...d25098` and `61b80361...77d55d`), with disjoint bounding boxes [627,415,864,640] and [1360,320,1600,495]. The shared right region is `d28eb153...32d33`, bbox [0,0,1600,792], pixel_count848172. These are image-derived region identities; disjoint/large image regions do not establish distinct/common physical planes. The two left readings retain the same axis matrix, focal779.1295821110922 and principal point [800,450], with different rectification centres. There is no captured calibration swap between them.

Numerical consistency and physical interpretation

For all eight stored alternatives, independent scalar readback gives exactly the stored rectified step `2*extent/(samples-1)`, world period `period_pixels*step`, and distance `b/world_period`, with residual0.0. This rules out a mismatch between the recorded distance and its recorded conversion parameters; it does not verify the estimated period, physical b, region geometry or correct harmonic.

With world sign+1 for Y and side-1, the source equation is:

```
delta176 = -146.15787988715914 + 23.62194066282116 = -122.53593922433798
delta197 =  -43.12811047587907 + 23.62194066282116 =  -19.50616981305791
delta197 - delta176 = 103.02976941128007
146.15787988715914 - 43.12811047587907 = 103.02976941128007
```

The equality is exact under these stored floating-point operations. The shared right term cancels. The closest pair in the complete left distance rosters is alternative2 (36.539469971789785) versus alternative0 (43.12811047587907), gap6.588640504089284, still above tolerance0.15. Thus no retained left alternatives can make these two endpoint pairs share one normal displacement when both are active and bind the same right alternative. This conditional statement does not extrapolate the first witness to omitted events or unrepresented Q plans.

Let a physical surface have coordinate h, a camera coordinate C, and a signed camera-to-plane distance s*d. Then `h=C+s*d`. A row asserting the same physical surface on both endpoints requires `Cright-Cleft=sleft*dleft-sright*dright`. Distinct surfaces may have different h while producing the same camera translation when each is matched to its own correct right-hand surface. Both captured rows instead target the same right reading, which carries one normal distance. If each left endpoint actually represents a different physical surface, the assertions cannot both be same-plane matches with accurate distances. Allowing a deliberate cross-plane offset would add the unknown term `hleft-hright` to the row interpretation; no such offset is measured or represented by this PlaneMatch. The observed gap therefore does not itself prove a wrong camera-translation equation or a valid state that the source mistakenly rejects.

Exact source trace

1. `plane_graph.py:218-262` creates content-ordered per-frame canonical reading indices and rejects repeated objects or ambiguous canonical keys. `:265-296` validates frame/axis/object and selected b identity. Matcher construction `:1554-1570` recovers both canonical plane indices from the actual signature reading objects and checks those objects. The original source-row order is retained by `AdmittedCandidateUniverse` (`:94-104`, `:1596-1601`). It includes the complete local-admission universe with legacy kept/dropped dispositions; the selector deliberately uses that complete universe, not scored filtering. Exact legacy dispositions for these two rows are not in this first-witness projection.
2. Matcher admission `plane_graph.py:1246-1309` requires supported overlap>=9, structural_share>=0.15, defined PSR>=3, and retains a scored local correlation cue. Rows176/197 have overlaps1166/116, PSR4.717645623448209/4.402303568464688, score0.5325010917463767/0.2900980888817555. Both pass those recorded local gates. Row197 has a close second peak, but that scalar does not prove a false correspondence. `cell_grid_homography` (`:843-848`, `:888-899`) maps plane-tangent grids; their shifts [18,7] and [-2,-10] are tangent shifts, not normal-plane offsets. For a Y-normal plane, tangents are X/Z.
3. `image_derived_joint_graph.py:250-260` takes `universe.candidates` without changing row identity and passes the complete readings/signature universe to the selector. `image_derived_endpoint_selection.py:96-132` binds endpoint/global reading index, reading key hash, full alternative roster and exact signature hashes. `:533-546` validates candidate indices and calls the endpoint CSP. `:263` in joint_graph only evaluates selected geometry after uniqueness; this no-solution result never reaches that stage.
4. `endpoint_hypothesis.py:453-483` binds every source row to selected alternatives by actual alternative identity, normalizes endpoint order and inverse Q only when required, and keys the registry by `(u,v,relativeQ)`. Rows176/197 have keys `((image-0000,2),(image-0001,1),0)` and `((image-0000,3),(image-0001,1),0)`. Rows are alternative support within one factor, but distinct endpoint-pair factors remain separate obligations. `:508-521` activates factors compatible with Q and builds alternative domains. `:540-554` intersects each domain with supported alternative pairs across every active factor, then checks forced relations.
5. `plane_graph.py:107-141` binds exact alternatives, transforms each axis/sign, multiplies sign*side*distance, and canonicalizes frame order once. Its literal Y mapping is `(1,+1)` for every yaw quadrant (`:816-835`). Original and canonical directions here are both forward; no sign reversal occurs. No phase, tangent-grid origin, half-cell or block-face term is silently inserted into this normal equation.
6. `endpoint_hypothesis.py:333-359` processes only factors with singleton endpoints, resolves their genuine source-supported rows through that shared signed relation, and merges extrema by `(left_frame,right_frame,world_axis)`. `:364-378` rejects a forced diameter>0.15 and records exact source extrema. This is the first physical contradiction check for this branch; it rejects the simultaneous same-plane assertions, not the mere presence of distinct reading IDs. Reversible state restoration is explicit at `:381-389` and `:523-530`. There is no evidence of stale sibling state in this captured event.
7. `image_derived_endpoint_selection.py:305-336` requires unique ordered forced-factor IDs, singleton domains, source-supported extrema and a real over-tolerance diameter. `:628-649` independently requires every selected endpoint-pair factor, refusing omitted support; `:658-668` replays the entire proof. The independent final physical output guard also compares each signed relation to one camera displacement and checks the same diameter (`plane_graph.py:152-170`). Removing just the pruning check would not repair the physical inconsistency.

No pinned operation is demonstrated to turn a physically valid association state into an invalid one. The operation that makes admitted cues mandatory is registry activation plus all-factor propagation. Calling that an implementation defect would first require evidence that these cues are intended to be mutually exclusive physical-association hypotheses, rather than separate mandatory factors under the accepted contract. That evidence/design is absent. Skipping row197, choosing the higher score, making either factor optional, adding an arbitrary per-plane offset, changing side/Q, or relaxing tolerance would alter consistency/uncertainty semantics and could accept a genuinely contradictory control. None is authorized or implemented.

Precise remaining evidence gap and smallest next diagnostic

The missing fact is whether the shared right reading represents one physical plane associated correctly with both left regions, a composite region spanning different planes, or an appearance alias unrelated to one of them. The projection retains source mask/sample-count hashes and selected scalar metrics, but not spatially located matched support on the shared right region or independent physical correspondence evidence. Its winning_round, ranked_support_floor_candidates and raw_supported_correlation_maxima groups are explicitly unavailable, so wrong harmonic/estimator choice and competing-peak support also cannot be diagnosed from this capture.

The smallest next diagnostic is a separately authorized, source-pinned spatial witness for exactly these two original-row identities using the existing matcher witness mechanism: retain each selected cell/pixel correspondence footprint and its exact source-region membership, especially where rows176 and197 land inside the shared right region. Bind universe/row/proof/signature hashes and original directions; keep existing budgets and fail closed if unavailable. Readback of source-produced support should not rematch, remeasure, rerank or fabricate geometry. Such a capture can test merged-region or disjoint-support hypotheses; even disjoint support does not prove distinct physical planes. A production repair additionally needs independently constrained image-derived physical association/normal geometry, or a new preregistered genuine-producer synthetic falsifier that demonstrates this exact mandatory-factor failure on known distinct surfaces. That is a new design/task, not permission to reopen or tune the consumed first-gate-failed experiment.

Verification and stop

Static strict-JSON digest checks and stored-scalar arithmetic passed. In-memory compile passed for all four pinned sources and four pinned test files; no imports, tests or production inference ran. Tests executed:0. Per assignment, implementation/tests are conditional on demonstrating an exact defect, which this trace did not establish; therefore the four focused modules were not executed and no test-pass claim is made. Only this report and append-only AGENT_COMMS are written. There are no code/test changes, cap/gate changes, new observer hooks, image/truth/metadata accesses, reruns, captures or output claims. The first retained event is representative; totals remain four Q plans visited, three eligible,432 partial prunes (128 retained+304 omitted),61871 relation visits, zero solutions. Stop for independent lead review.

Final unchanged source/test SHA256 values:

| Path | SHA256 |
| --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_joint_graph.py` | `60F4F55BC887BEF50C60427A00FCA54F5AD854A1785EBD5B1319C4B823DF25AB` |
| `src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py` | `F8065946EAEDB95CDBB8D2BFCEACE0246BF1369D2F8683F2FD555AA1904A7C0A` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `6868AC42C657C2CA270E9D91A36C4F7958E2AD4AD9014B391BFF415CE4A646A8` |
| `src/mc_photogrammetry/reconstruction/endpoint_hypothesis.py` | `F197D8242148EBD3B49B2A5F01FA2BAB117731EF01F51A36423DA3CA8EE2F08C` |
| `tests/test_image_derived_joint_graph.py` | `7D167660DEDFA1E66BFA84CB85B691C25056FFB76D19DC1314C987E4CF98C1F9` |
| `tests/test_image_derived_endpoint_selection.py` | `7033D739E769B64D14D9A6286C4A2561C3E334F5A01A5B51933D0A19E1D21F29` |
| `tests/test_plane_graph.py` | `E2197A5CF0D925F93EB69AFDA0CA79151C79C5E10672664D8E54D8671011AFAA` |
| `tests/test_endpoint_hypothesis.py` | `789893CF482E22934D07CFFBE957A0DCB7DF83A195C91D41467599D0E66C72B3` |

