# SOL-SOURCE-CONSISTENT-ASSOCIATION-02

Preregistration before sole execution: all nine frozen source/reference pins match; the test target was absent. Task card SHA256 C518A995BBCF021C45BD08B4B75FD52E68C1FBE73C8AA2F65F470829DE30BF19.

Frozen test SHA256 6BE53C795D7A80EF603A4F2CEB75077B43BF5F019D3D53A3BA4E47AD03EB32D2. Source compiled in memory and direct whitespace check passed before any fixture execution.

The complete FROZEN declaration specifies all three arms, finite-panel ray casting/occlusion, material arithmetic, exact cameras/calibrations/proposal masks, expected statuses and oracle assertions. Genuine producer kernels and exact immutable signature binding are used without mocks. All alternatives and the full pre-filter admitted universe remain method inputs; true labels/coordinates/poses are used only for synthetic calibration/proposals and separate assertions. No ordinary image run, production edits, retries, cap/gate changes or production certificates are authorized.

Execution completed once and stopped at the first mismatch. The frozen test was not edited or retried.

The sole focused command returned **1 failed in 30.96s** (wrapper wall time 31.25s, exit 1):

```text
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q --tb=short tests/test_source_consistent_plane_association_02.py
```

First mismatch: arm A / baseline_selector / preregistered baseline disposition. Expected `unique`; observed `multiple`, reason `multiple_endpoint_hypotheses`. This is a failed positive-control expectation, not a demonstrated production regression. The genuine producer chain reaches complete candidate admission but does not establish the required unique assignment. No counterfactual was called; B/C rasters and downstream stages were not executed. Nineteen prior predicates passed. No retry, regeneration, seed search, score preference or candidate removal occurred.

| Frame | Axis | Structural share | Retained alternatives |
|---|---:|---:|---:|
|a|X|0.6433991385030939|6|
|a|Y|0.4484669733327459|6|
|a|Z|0.6717426042366385|8|
|b|X|0.6382808990861588|6|
|b|Y|0.4565367567988684|6|
|b|Z|0.556705850313621|6|

All six required readings clear the unchanged 0.15 structural gate, exactly bind their proposal mask, and preserve the exact distance formula `b/(period_pixels*rectified_pixel_step(extent,samples))` for every alternative. Segments 473/434, no measurement abstentions. There are 38 real signature alternatives,444 complete pre-filter admitted candidates, and 12 legacy-kept candidates. The baseline receives all 444 and their typed dispositions.

The proof stores two Q=0 modes. The first selects b=1 at every endpoint; signed XYZ displacements are `(0.7566354308824774,0.49893800027089963,0.4873349393708999)`, agreeing with frozen true translation `(0.75,0.5,0.5)` within the unchanged 0.15 tolerance. The second changes only left Z to retained b=0.5, distance 11.981455663569777; right Z remains b=1, distance 23.475576387768655. It yields Z displacement ?11.494120724198877. No oracle camera translation enters either selector.

| Source row | Relation | PSR | Overlap | Cell shift | Signed delta |
|---:|---|---:|---:|---|---:|
|23|X b=1/b=1|13.008452332792256|387|(1,0)|0.7566354308824774|
|183|Y b=1/b=1|31.954946394604526|427|(0,1)|0.49893800027089963|
|388|Z b=1/b=1|18.8307214036431|400|(0,1)|0.4873349393708999|
|377|Z b=0.5/b=1|3.4605608727136614|104|(6,?4)|?11.494120724198877|

The false-scale row 377 is genuine matcher output that clears existing admission requirements. Each chosen mode has one active factor per axis; with two frames, this fixture supplies no redundant signed relation on an axis or independent frame cycle to contradict the rival Z displacement. Selecting the highest-score row or using the supplied true translation would change the authorized label-blind semantics. The smallest missing capability evidenced here is source-derived harmonic/distance disambiguation sufficient to rule out this already-admitted mixed-scale mode. A separately authorized fixture with redundant same-axis constraints or independent image-derived displacement evidence may test that capability; this consumed fixture is not repaired or retried. B distinct-plane/cross-candidate and C many-to-many realizability remain unknown.

Baseline work: relation visits 15742/262144; branches 8/16384; Q plans 4/16384; endpoints 6/128; max alternatives 8/16; rows 444/16384; registry factors 12/4096; scalar payload 238877/4194304bytes. There is no denial. Two feasible assignments were retained; exhaustive=false because baseline correctly stopped after its second solution. Assignment partition 1 visited + 3 unvisited = 4 planned; physical 2 feasible + 0 rejected = 2 checked; eligibility 1 eligible + 0 excluded = 1 hashed. Scalar categories 236648 source + 575 relation + 370 disposition + 0 exclusion + 1284 support = 238877; scalar 918 attempted = 918 committed + 0 uncommitted; search 2 attempted = 2 committed + 0 uncommitted. Every recorded closure passes. Counterfactual counters are unavailable because its stage was not executed.

In-memory compile, direct whitespace and strict bounded JSON readback pass. All nine pinned sources/references and the task card match after execution; frozen test bytes match their preregistered digest. Evidence contains complete source-row records, measurement scalars/alternatives, both modes, exact stdout/stderr and explicit unexecuted stages. No ordinary photo/image inference, production/helper edits, cap/gate changes, truth input to methods, accuracy gain or output export occurred. Keep ordinary image-only behavior and 90% visible geometry F1 / 80% visible exact-state acceptance gates intact. STOP for independent lead/Astra review.

Evidence SHA256: 1D53C0BC83521498C5B243D3575C82632117E0CC01C1DE811DB2BE1E09FA7CD1.

Source/reference postflight pins:
- `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` SHA256 `FB061BBFAA14BFB969E4349F1A01C080EE338A558D3CD6521425EB1D8231ED8E`.
- `src/mc_photogrammetry/pipeline/image_derived_signatures.py` SHA256 `7E287AECB6C1E6BB111C5D5A6C33ABC876B6083CE48ABDF3C4FFF1306819E43F`.
- `src/mc_photogrammetry/reconstruction/plane_graph.py` SHA256 `6868AC42C657C2CA270E9D91A36C4F7958E2AD4AD9014B391BFF415CE4A646A8`.
- `src/mc_photogrammetry/reconstruction/endpoint_hypothesis.py` SHA256 `F197D8242148EBD3B49B2A5F01FA2BAB117731EF01F51A36423DA3CA8EE2F08C`.
- `src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py` SHA256 `F8065946EAEDB95CDBB8D2BFCEACE0246BF1369D2F8683F2FD555AA1904A7C0A`.
- `scripts/evaluation/endpoint_association_counterfactual.py` SHA256 `97B05EA9A4700DEE29EF82CE615FCCDF71B14D7BDC3EF806C85199FDB2CFB664`.
- `tests/test_image_derived_lattice_measure.py` SHA256 `045B31348DD1F7A0BFC519D386697248E79DA411053436AEEBB3E7BFBB67E240`.
- `tests/test_image_derived_signatures.py` SHA256 `40A5EF36D58ECBF804434AA4C0711EE6CE1E28658CABB8E41829B19A9AA8F455`.
- `tests/test_plane_graph.py` SHA256 `E2197A5CF0D925F93EB69AFDA0CA79151C79C5E10672664D8E54D8671011AFAA`.

Exact command output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
____________ test_frozen_source_consistent_association_full_chain _____________
tests\test_source_consistent_plane_association_02.py:256: in test_frozen_source_consistent_association_full_chain
    _require(record, arm, "baseline_selector", "preregistered baseline disposition", baseline.status == FROZEN["arms"][arm]["baseline"])
tests\test_source_consistent_plane_association_02.py:101: in _require
    raise AssertionError(f"{arm}/{stage}: {predicate}; frozen experiment stops without retry")
E   AssertionError: A/baseline_selector: preregistered baseline disposition; frozen experiment stops without retry
=========================== short test summary info ===========================
FAILED tests/test_source_consistent_plane_association_02.py::test_frozen_source_consistent_association_full_chain
1 failed in 30.96s
```
