# M4 P1-A — periodic phase evidence and relative cell shift

Date: 2026-09-13

## Scope

This change adds two hypothesis-scoped, in-memory measurements. It does not
resolve phase into block residues, seams, lateral origins, or world placement.
It does not alter match admission, selector scoring/order, thresholds, or the
frozen image-run schema.

`PlaneReading.phase_evidence` is an optional frozen
`PeriodicEnergyPhaseEvidence`. Its `(u, v)` entries are raw patch-pixel phases
from the rectified measurement patch: `u` is the column profile and `v` is the
row profile. It retains the period, rectification center, extent, sample count,
per-axis response magnitude/profile mass/concentration, unavailability reason,
and optional source-region ID. Available phases use the existing degeneracy
gates only: finite nonnegative profiles, period at least 2 pixels, mass above
`1e-9`, response above `1e-12`. A modulo endpoint within `1e-12` of the period
is canonicalized to zero. This preserves the legacy field's meaning while
distinguishing a measured zero (nonzero response) from its old `(0, 0)`
unavailable sentinel. Evidence validates its geometry, source identity and
phase values against the owning reading. It is excluded from
`reading_order_key` and schema-v1 records.

`PlaneMatch.cell_shift_rc` is an optional integer `(row, column)` residual on
an admitted in-memory match. Its sign is pinned by
`warped_left[x + shift] ~= right[x]`; for `right = np.roll(left, (3, -4))`,
the shift is `(-3, +4)`. `match_plane_graph` preserves it when copying an
admitted match. The selector and graph objectives ignore this additive cue.
The frozen runner writes the same explicit legacy match-key set and omits
`cell_shift_rc`.

## Bounded closure

The raw complex Fourier resultant is now calculated by the single private
`supported._periodic_energy_resultant` helper. Both the legacy
`supported.phase_at_period` caller and lattice phase evidence use that exact
complex core; profile validation, phase sentinels, mass/response gates,
concentration and near-period canonicalization remain caller-specific.

New frozen runs include `lattice.py` and `supported.py` in `code_hashes`. The
strict loader accepts exactly the historical six-module provenance set or the
current eight-module set, preserving old schema-v1 artifacts while rejecting
unknown provenance keys. Top-level, reading and match schema-v1 key sets are
unchanged. The existing R4 freeze and sidecar were strict-loaded and byte
verified without mutation.

## Validation

The focused command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_joint_plane_selector.py tests\test_frozen_image_runner.py tests\test_supported.py
```

Result: **167 passed in 113.19 seconds**. Coverage includes shared Fourier-core
parity on nonzero phase, exact mass/response boundaries, near-period
canonicalization, zero-versus-unavailable, concentration and malformed
evidence, source/geometry binding, evidence-stripped signature parity, signed
zero/interior/maximum shifts on asymmetric non-square controls, graph and
integer-residual annotation propagation, selector/legacy-solver parity,
schema-v1 key exclusion, new provenance hashes, unknown-hash rejection, and
strict loading of the immutable R4 freeze/sidecar. Six production modules
passed `py_compile`; `git diff --check` on owned source/test paths passed.

No R1-R4/E1-E3 freeze, sidecar, evaluator, or score was modified or rerun.

## Source hashes

SHA-256 values after the implementation and tests:

| File | SHA-256 |
|---|---|
| `src/mc_photogrammetry/reconstruction/supported.py` | `4c94564d3842f1c3a03883f6f95a78a4ac08260150eb9fc12251bf708e35f29c` |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `eb25a2f5d6f97f38f7c572b9dca8a0820db19f7d712db6530df88cb3225c008e` |
| `src/mc_photogrammetry/reconstruction/frozen_image_runner.py` | `8ed08a27393728f04a767d40736d399d6952baaacceebad149dc34a6b909fbde` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `fefbb7277165549fabac52602190c721ab65b80125c5dd4b94313d37b21dc1c1` |
| `src/mc_photogrammetry/reconstruction/lattice.py` | `6fcde3a7e8d451e207ec0eed70c26a3f1746ab89f9ac8cf7bba5bcb8fd7bb17e` |
| `src/mc_photogrammetry/reconstruction/__init__.py` | `2e6a68657455b581b36afae83c3be2b31818cccc4503767e5104c5f48f953cce` |
| `tests/test_supported.py` | `6cf4f8d967f1c41da659314fc600029025afebc59709c36205905bc4590aded2` |
| `tests/test_lattice_planes.py` | `f2d52d0440e3d099832a89a5e877e88fc20aadf0901d771a2e39ad77c662108f` |
| `tests/test_plane_graph.py` | `e45d6e803dc6290b01d1636d92c5293969b7f30a0ec68e91c61a26e871ea1f31` |
| `tests/test_joint_plane_selector.py` | `89c85b0e229dce71774dc594d7453b8987182b9333370d820dd7c428dc4f7feb` |
| `tests/test_frozen_image_runner.py` | `760f7b9fd75747e326dd2d7407834e327236e21ed1a2e47c0a47243a9ffd79d4` |
