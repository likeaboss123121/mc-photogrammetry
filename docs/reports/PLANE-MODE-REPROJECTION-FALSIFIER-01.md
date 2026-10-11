# SOL-PLANE-MODE-REPROJECTION-FALSIFIER-01

Preregistration sealed before sole focused execution. All six historical archive/report/receipt pins match; consumed original test is absent, and all five new task targets were absent before creation.

Frozen helper SHA256 9E00EDE0510E8F9AD9BBD0EBE9E54E9120DD7CDA6E26C5EA28B07236EBA94916; frozen test SHA256 E6CB0321472FF5EB84FFF5947C4FBE1AB9136C84729E53916A828B28F0D41996. Complete numeric controls, camera conventions, source anchors, invalid cases and bounds are frozen in test source and evidence. Static AST/whitespace checks pass. No checker/test execution or image access has occurred.

Sole execution stopped at the first failure. Frozen helper/test bytes were preserved; no retry or parameter change.

The sole focused command was:

```text
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q -x --tb=short tests/test_plane_mode_reprojection.py
```

**1 failed in 0.26s**, wall 0.609s, exit 1. The first true/mixed control expected raw residual evidence and 36 point checks, but returned `abstained/malformed_numeric`, `modes=[]`, `point_checks=0`. Atomic refusal passed as behavior; the intended geometric positive control did not execute. No other test or control was run. Uniform-scale invariance, invalid/depth cases, provenance/permutation, repeated-feature explanations and exact-budget/64KiB cases remain unverified.

The source trace is precise: test line 68 creates `delta=np.asarray(...)`; test line 90 constructs `tuple(-v for v in delta)`, retaining NumPy floating scalars in the proposed true translation. Checker line 196 passes that translation to `_vector`, whose line 81 requires exact built-in `int` or `float` types. This scalar restriction is a local implementation choice, not a production gate or task requirement. JSON serializes these scalars as ordinary numeric values, so the source construction, not JSON alone, identifies the mismatch. The failure is a frozen fixture/checker interface typing error, not evidence that the homography cannot distinguish a mixed-scale alias. No projective residual was computed and no geometry or production regression is established. Repair/retry requires a separately authorized task; no code was changed after the freeze.

The written checker declares `X_right=R X_left+t`, `n_left.T X_left=d`, and `H=K_right(R+t n_left.T/d) inverse(K_left)`. If camera-centre delta is expressed in left camera coordinates, substitution gives `t=-R delta`; world displacement first needs left-camera rotation. The checker has no oracle point/true-mode selector input, score, outlier deletion or thresholded production acceptance. It is designed to preserve all correspondence rosters and emit source/candidate seals plus finite/depth validity and symmetric residuals, with atomic refusal and unresolved uniform global scale. These paths beyond scalar preflight are not validated by this failed run.

All six historical archival/report/receipt pins match after execution, and consumed original test02 remains absent. Real `py_compile.compile` passed for both new files using an in-memory bytecode sink; neither was imported/executed by this verification and no .pyc was written. Direct whitespace and scoped `git diff --check` pass (files are untracked, so direct whitespace is substantive). Strict JSON roundtrip and canonical evidence bound<=65536bytes pass. No full suite, images, consumed test execution, production/helper integration, policy/threshold/cap/schema change, mode ranking, certificate or accuracy/export claim. Preserve image-only normal mode and 90% visible geometry F1 / 80% visible exact-state gates. STOP for independent lead/Astra review.

Evidence SHA256: F9C9B2D2F1ED4062E0A03F2DF088B58D22B0B9BD1860311C8F83C7F0D81E36B0.

Exact command output:

```text
F
================================== FAILURES ===================================
__________________ test_true_mixed_and_global_scale_controls __________________
tests\test_plane_mode_reprojection.py:118: in test_true_mixed_and_global_scale_controls
    assert result["status"] == "raw_evidence" and result["point_checks"] == 36
E   AssertionError: assert ('abstained' == 'raw_evidence'
E
E     - raw_evidence
E     + abstained)
=========================== short test summary info ===========================
FAILED tests/test_plane_mode_reprojection.py::test_true_mixed_and_global_scale_controls
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.26s
```
