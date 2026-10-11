# C1 float32 remediation: synthetic-only implementation

Status: implemented for independent review, 2026-10-06 21:22 CDT. **C1 remains BLOCKED and consumed.** No actual prediction was recovered, re-evaluated or regenerated. Work used about10 aggregate agent-minutes, below the lead's stricter1hour cap (the task file otherwise says2hours).

Changes are confined to the declared adapter, runner, tests, this report and Comms append. No runner main, model, screenshot/dataset/truth/manifest/mask read, checkpoint-byte read, package install/download, environment/cache/C1 output/report modification, capture, UI/input/audio, driver change or product integration occurred.

## Explicit rotation policy and derived correction bound

Pinned upstream source `.venv-vggt/vendor/vggt/utils/rotation.py` defines `quat_to_mat` using four squared quaternion components, their sum, `two_s=2/sum(q*q)`, quadratic pairs and final products/subtractions. Source SHA256 `ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070`, code commit `a288dd0f14786c93483e45524328726ab7b1b4ce`. Only this source was read, never weights.

Let u be declared precision's unit roundoff, gamma_n=n*u/(1-n*u), and assume finite normal arithmetic with no overflow/underflow in the quaternion construction. Components already stored at that precision are the exact inputs to this derivation; quaternion normalization need not be exact because the upstream formula divides by its norm.

- Four products plus a sum give a relative norm-squared error bounded by gamma_4.
- Division gives relative `two_s` error a=(gamma_4+u)/(1-gamma_4).
- A quadratic pair has absolute error at most gamma_2 times its sum of absolute products. Diagonal pairs are <=||q||^2; off-diagonal product-pair sums are <=||q||^2/2, by2|ab|<=a^2+b^2.
- Including final multiplication, b=a+gamma_2+u+a*gamma_2+a*u+gamma_2*u+a*gamma_2*u. The exact diagonal correction has magnitude<=2; final subtraction contributes at most u*(1+2*b), since an exact rotation entry is within[-1,1]. Every entry is therefore bounded by eta=2*b+u*(1+2*b).
- Nine entries imply `||R_raw-R_exact||_F <= B=3*eta`. Float32 u=2^-24 gives **B=3.0398382691038855e-6**. Float64 gives5.662137425588303e-15. This comes from source arithmetic, not the discarded real residual or a threshold sweep.

`canonicalize_pose` preserves raw matrix/intrinsic copies and dtype, rejects nonfinite/reflected/materially non-rigid matrices, and applies one float64 polar/SVD correction `R_c=U V^T` with translation unchanged. A proper polar rotation minimizes Frobenius distance to SO(3), so a raw input explainable by the declared precision should require no more than B correction. The implementation rejects measured correction>B with **no extra empirical SVD correction allowance**. Singular values, determinant, raw/canonical orthogonality, correction and bound are recorded, including over-correction refusals.

Canonical matrices separately certify float64 orthogonality within64*epsilon64. This is a fail-closed numerical certificate for the computed polar/Euler seam, not a guarantee that every LAPACK/trigonometric implementation attains that residual. Measured canonical orthogonality also enters the projection error envelope. Geometry primitives refuse raw near-rotations rather than silently treating transpose as inverse. Both inverse geometry and repository-camera conversion receive the same explicit canonical extrinsic. Pitch uses atan2(vertical,hypot(horizontal)) to avoid asin conditioning near vertical; eye_height remains0 and camera conventions are unchanged.

Assumptions/limits: the precision envelope covers the pinned formula under normal finite arithmetic, not arbitrary learned non-rigid matrices or unknown arithmetic provenance. An input requiring larger correction is refused, even if visually plausible. No angular/world/block accuracy or metric scale is established by normalization.

## Separate numeric diagnostics and conditioning

Future geometry artifacts now distinguish:

1. **Raw-matrix source round-trip:** use the raw matrix's true inverse, retain its residual; never pretend a drifted raw transpose is the exact inverse.
2. **Canonical algebra/source and repository parity:** canonical transpose inversion, direct matrix projection and repository Euler seam. The existing **1e-5 pixel gates remain unchanged**; raw-matrix inverse parity also uses1e-5.
3. **Raw-to-canonical projection delta:** project the same canonical world point through both matrices, preserving raw values and the observed delta separately. It is governed by its derived input-precision envelope, not mislabeled as canonical algebra error or assumed below1e-5.

For world point X and measured correction c=||R_raw-R_c||_F, camera displacement is bounded by d=c*||X|| plus float64 dot/add rounding terms. For canonical camera coordinates(x,y,z), require z>d. Then:

`|delta_u| <= fx*d*(z+|x|)/(z*(z-d))`, similarly forv. The Euclidean pixel envelope combines both axes. Nonfinite values or a depth envelope crossingz=0 are explicitly refused.

The canonical round-trip conditioning check uses gamma_3/gamma_4 float64 bounds for pixel subtraction/division/depth multiplication, inverse/forward dot products and translation addition, plus the measured `||R_c R_c^T-I||` term. Resize inverse scale maps this into source pixels. If that calculated envelope reaches1e-5, the view is rejected as ill-conditioned rather than weakening the parity gate. Observed raw/canonical deltas must also lie within their per-point bound. Large translation/depth ratios therefore have an explicit decision; tests show1e5 translation can pass while1e14 translation is refused.

These are algebra/numerical diagnostics, never camera-accuracy, cross-view visibility or scene-quality metrics. Existing90% geometryF1/80% exact-state release criteria and all other acceptance thresholds are untouched.

## Persistence before refusal

The existing small artifact boundary writes **all raw frames first**, preserving identity, raw extrinsic/intrinsic/depth/confidence and original dtype even for nonfinite/refused data. Atomic NPZ writes use temporary-file replacement; receipts remain atomic JSON replacement. No observer framework was added.

Canonical matrices and correction records are written before depth/parity validation. Numeric diagnostic artifacts retain source pixels, raw inverse and canonical reprojections, canonical error envelope, raw-to-canonical delta/envelope and repository sampled projections/errors. Frame receipts retain maxima, argmax indices/model pixels and exact failure stage/frame. Over-correction failures retain their measured correction and limit. A forced parity refusal leaves all raw frames, canonical matrices, computed residuals and a terminal BLOCKED receipt. Early domain/conditioning failures retain raw evidence and the diagnostics available before refusal; no unavailable residual is fabricated.

No write was made to the consumed C1 artifact directory. The unchanged exclusive-receipt guard still prevents replay of that packet. Future inference requires a separate lead-reviewed packet and fresh authorized output/input scope.

## Checks and preservation evidence

Focused command: `.venv-vggt/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_learned_geometry_adapter.py` -> **40 passed in0.52s**. Intermediate runs were repeated only after added behavior/checks or relevant edits. New tests include the analytic one-ULP counterexample, float32-rounded/perturbed rotations, three quaternion magnitude controls versus independent Rodrigues rotations, rejected non-rigid matrices, near/exact vertical rolls/off-centre intrinsics, accepted/rejected large-translation conditioning, separate raw/canonical/delta checks, retained over-correction diagnostics, nonfinite raw preservation, forced0.25px repository failure with argmax/BLOCKED receipt, and actual in-memory atomic JSON/NPZ serialization including float32NaN retention.

Tests import runner definitions without invoking main; no model framework is imported by the numeric artifact helper. Artifact writers are injected in-memory; no synthetic files were written outside the listed scope.

`py_compile.compile(...,doraise=True)` passed on all3Python files. Bytecode publication was intercepted via `unittest.mock.patch('importlib._bootstrap_external._write_atomic')` so this required compiler check wrote no pyc or environment/cache files. Standard compilation still executed and produced valid code bytes in memory.

Final changed-source SHA256:

| File | SHA256 |
| --- | --- |
| learned_geometry_adapter.py | a7d20dd015872193eab07e709c6920e35639fb838f94aaa6e1ee61f90dfebb16 |
| run_learned_geometry_dev.py | 30e626d0ab769efbe65e30f42c0571ea2aeb25fd1ffdf4dd8601d145e63385a0 |
| test_learned_geometry_adapter.py | bc8b867fe13a71ead207101d64e529c030f445d6f4f544c80deca42b05525404 |

Consumed files were hashed before and after; all unchanged:

| C1 file | SHA256 |
| --- | --- |
| Original C1 report | f233a1016b25a4050dba7e3fec48fedea05b765d181a5c2b8e0caf1cf5a2d358 |
| run.json | c4e12492756b0e64e1ef9b027d82c27fd04d072d4c4518f4f36519b1594d5cb2 |
| stdout.log | cb1eb338505f2f52a3ab81be61a5071212ccf4a68717182c0f59bc736b7647ac |
| stderr.log | 96a226cfd106dfd501f63c0a75ddab8a199cf05a506e57204981eeb854598b9f |

Checkpoint metadata before/after is identical: length5,026,367,224; CreationTimeUtc epoch-ms1791338392963; LastWriteTimeUtc epoch-ms1791338829431. Its previously verified SHA256 remains recorded asf164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e. **No fresh byte hash was computed:** opening weights would violate this remediation's hard constraint. Metadata identity and the absence of checkpoint writes are the limited preservation evidence, not a new cryptographic attestation.

Astra's21:10 read-only failure review was read and implemented as the architecture basis; it corrects the claim that one specific old parity branch was proved. The discarded C1 failure branch/residual remains unknown. This remediation supports only independent review and a later decision about fresh authority. It grants no retry, capture, C2, integration or promotion. STOP.
