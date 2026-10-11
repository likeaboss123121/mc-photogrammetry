# SOL-PLANE-MODE-REPROJECTION-FALSIFIER-02

Preregistered before sole execution. Both v1 code artifacts archived byte-for-byte outside discovery before edits; all eleven historical archive/report/receipt pins match. New report/evidence/result targets were absent.

Frozen v2 helper SHA256 4554DBB2953C289F1863E917F4E1112DAFAD93BCDCA195D4E54E2AE5C554A67B; frozen v2 test SHA256 DBF1093BCED0819C68D2DE8D08F65C3558D2226D27A6B184A6BF6936E30113B7. V2 evidence path is distinct from v1. Complete numeric interface and unchanged geometric controls are declared in FROZEN/test source and evidence. Static AST/whitespace checks passed before execution. First failure stops; no retry or tuning.

Sole execution completed with all frozen controls passing; no retry or tuning.

**86 passed in 0.74s**, wrapper wall 1.0s, exit 0. One focused run only:

```text
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q -x --tb=short tests/test_plane_mode_reprojection.py
```

The first 61 tests validate the corrected scalar/source interface before the 25 geometric and failure controls: NumPy float32/float64/int32/int64 are accepted in every numeric field ; 56 combinations of bool/numpy.bool_, NaN/Inf, complex/numpy.complex128, string and arbitrary object across calibration/rotation/translation/normal/distance/left-coordinate/right-coordinate refuse atomically with zero point checks. Integer canonicalization preserves adjacent values 2**53 and 2**53+1 as distinct source seals while matching equivalent NumPy/builtin integer values. Conversion accepts only declared real scalar types, rejects bool explicitly, checks finiteness/shape before numerical arrays, and normalizes to built-in int/float; source IDs and all supplied integer coordinate bits remain bound.

The geometry follows `X_right=R X_left+t`, `n_left.T X_left=d`, `H=K_right(R+t n_left.T/d) inverse(K_left)`. For camera-centre displacement delta expressed in left-camera coordinates, `t=-R delta`; world displacement requires left-camera rotation first. The quarter-turn rotation control verifies this conversion independently. Physical world points and true pose are used by the test's direct projection generator/assertions only. The checker receives proposed parameters and sealed continuous pixel pairs. Fixture mode identifiers `true`, `mixed`, `scaled` are diagnostic keys only: the checker sorts/echoes them and never consults their meaning, ranks modes or selects a preferred explanation.

| Plane | Maximum true symmetric residual (pixels) | Minimum mixed symmetric residual (pixels) | Maximum mixed symmetric residual (pixels) |
|---|---:|---:|---:|
|X|1.7975467359112708e-13|317.6173289318809|2941.839525219178|
|Y|1.1544952946900308e-13|188.3411126938081|1675.9028241507413|
|Z|5.684341886080802e-14|39.59745461640557|171.14157185835205|

All true residuals meet the preregistered 1e-10 numerical assertion. All mixed-mode point residuals exceed the preregistered 1px falsifier assertion; these values are test assertions, not checker acceptance thresholds. Candidate X/Y distances remain 8/4; left Z distance 12 becomes 6, changing proposed centre delta Z from 0.5 to ?5.5 while preserving right plane normal distance 11.5. Independent continuous point projections expose the alias, including its cross-axis effects. The checker returns both raw mode rosters and no production acceptance/rejection flag.

Uniform scale 4 of every distance and translation leaves all homographies and forward/backward residuals exactly identical while depths scale; output explicitly says `unresolved_uniform_global_scale`. Degenerate calibration/plane/pose/homography/rays, nonfinite inputs and behind-camera points return typed abstention with no partial modes/residuals. Identity-preserving plane/pair permutations keep source seal/results invariant; swapping pixel assignments and resealing cannot replace the independently retained source anchor. Source-ID changes refuse. A rotated repeated-feature correspondence roster remains a second explanation even with larger residuals; all twelve Z roster pairs are retained without oracle selection.

The unchanged hard limits apply cumulatively to each complete request: 2 modes, 3 planes, 32 pairs per plane across rosters, 192 mode-point checks and 65536 canonical diagnostic bytes. Exact-bound control emits 192 points after 192 checks with 52121 bytes. Maximal legal identifier lengths produce atomic `output_limit` after 192 checks and an empty mode list; no partial diagnostic survives. Over-limit mode/plane/pair/combined-roster input and malformed shape/source/duplicate identity/plane mismatch refuse with zero point checks. Stored full-packet closure: true+mixed 36 emitted = 36 charged; true+uniform-scale 36 = 36; repeated explanations 24 = 24. No budget reset or cap change.

Real `py_compile.compile` passed for both updated files with bytecode captured in memory/no .pyc or source execution. Direct whitespace/scoped diff pass (files untracked; direct whitespace is substantive). Canonical JSON replay independently recomputes the full source seal and both proposed-mode seals; all stored point counts close and packet outputs stay<=65536bytes. V1 helper/test archive bytes, v1 report/evidence/result and all task02/archive seals remain exactly unchanged: eleven historical pins checked before/after. V2 helper/test hashes still match preregistration. Distinct evidence path prevents v1 receipt overwrite.

Passing establishes only the frozen synthetic algebraic falsifier. It does not establish availability of independent continuous image correspondences, rejection of the previously observed actual image mode, identity of physical planes, product accuracy, or production disambiguation. Image-only acquisition, ambiguity/visibility/source/held-out handling and numerical uncertainty require separate review before any image study or integration. No ordinary images, truth files, archived execution, full suite, production caller/selector/schema/policy/threshold change, score ranking, certificate/export or accuracy gain. Preserve ordinary image-only behavior and 90% visible geometry F1 / 80% visible exact-state gates. STOP for independent lead/Astra review.

Evidence SHA256: C81D5B91AF088B17A8EDDB134B58CCC3F041E8A81ADA053C8F7559EAD3394989.

Exact command output:

```text
........................................................................ [ 83%]
..............                                                           [100%]
86 passed in 0.74s
```

The single-use v2 test remains under pytest discovery and writes its fixed v2 evidence path. A separate archival or execution-guard task is needed before an ordinary suite can rerun it without overwriting the sealed receipt; no such follow-on change is included here.
