"""Focused controls for the narrow M4 geometry provenance bridge (DS-M9-B).

Every fixture is an explicit in-memory record built from accepted lattice
helpers; digests are recomputed from record data, never read from the bridge.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields as dc_fields, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes as lp
from mc_photogrammetry.reconstruction import plane_geometry_provenance as m9b
from mc_photogrammetry.reconstruction import plane_placement as placement

FRAME, AXIS, INDEX = "frame-0", 2, 0
EXTENT, SAMPLES, PERIOD, CENTRE = 4.0, 16, 4.0, (2.0, 2.0)
SIDE, POLY = 8, ((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0))
SPLIT_H, SPLIT_W, SPLIT_S, SPLIT_EXTENT, SPLIT_PERIOD = 12, 10, 8, 1.0, 4.0
_MISSING = object()

def _mask() -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[0, 0] = mask[-1, -1] = mask[2, 3] = True
    return mask

def _region():
    return lp._make_region_evidence(_mask(), AXIS, POLY, ("synthetic-region",))

def _structural_patch(offset: float = 1.0) -> np.ndarray:
    rows, cols = np.indices((SAMPLES, SAMPLES))
    values = offset + np.cos(2 * np.pi * rows / PERIOD) + np.sin(2 * np.pi * cols / PERIOD)
    return np.ascontiguousarray(np.abs(values), dtype="<f4")

def _phase(patch, region):
    return lp._periodic_energy_phase_evidence(patch, PERIOD, centre=CENTRE,
        extent=EXTENT, samples=SAMPLES, source_region_id=region.source_region_id)

def _reading(region=None, axis: int = AXIS, frame: str = FRAME):
    region = _region() if region is None else region
    patch = _structural_patch()
    phase = _phase(patch, region)
    rect = lp.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=20.0, principal_point=(512.0, 512.0), centre=CENTRE)
    return lp.PlaneReading(
        frame=frame, axis=axis, side=1.0,
        alternatives=(lp.DistanceAlternative(1.0, 0.5, 1.0),),
        period_pixels=PERIOD,
        phase_pixels=tuple(0.0 if p is None else float(p) for p in phase.phase_uv_pixels),
        extent=EXTENT, samples=SAMPLES, polygon=POLY,
        coverage_cells=int(_mask().sum()), structural_share=float(
            lp._structural_share(patch, PERIOD)),
        confidence=1.0, rectification=rect, region_evidence=region,
        phase_evidence=phase)

def _weights(reading, patch=None, *, frame: str = FRAME,
             canonical_index: int = INDEX, **overrides) -> m9b.StructuralWeightsEvidence:
    patch = _structural_patch() if patch is None else patch
    region = reading.region_evidence
    fields = dict(
        frame=frame, canonical_index=canonical_index, reading_axis=reading.axis,
        source_region_id=region.source_region_id, patch_rows=int(patch.shape[0]),
        patch_columns=int(patch.shape[1]),
        patch_float32_le=patch.tobytes(order="C"),
        structural_share=float(lp._structural_share(patch, PERIOD)),
        period_pixels=PERIOD, extent=EXTENT, samples=SAMPLES, centre_uv=CENTRE,
        phase_evidence=_phase(patch, region))
    fields.update(overrides)
    return m9b.StructuralWeightsEvidence(**fields)

def _split_patch() -> np.ndarray:
    return np.abs(np.random.default_rng(7).normal(size=(SPLIT_H, SPLIT_W))).astype("<f8")

def _split_source(region, *, source_region_id=_MISSING, patch=None,
                  frame: str = FRAME, plane_index: int = INDEX):
    patch = _split_patch() if patch is None else patch
    budget = lp._make_budget({name: 1_000_000 for name in lp._SPLIT_BUDGET_FIELDS})
    geometry = lp.SplitPhaseGeometry(
        centre=(0.0, 0.0), extent=SPLIT_EXTENT, samples=SPLIT_S,
        rectified_pixel_step=lattice.rectified_pixel_step(SPLIT_EXTENT, SPLIT_S),
        period_pixels=SPLIT_PERIOD)
    result = lp.produce_split_phase_evidence(
        patch, geometry=geometry, frame=frame, plane_index=plane_index,
        reading_axis=AXIS,
        source_region_id=(region.source_region_id
                          if source_region_id is _MISSING else source_region_id),
        partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
    assert result.status == "complete"
    return m9b.SplitSourceEvidence(
        frame=frame, canonical_index=plane_index, reading_axis=AXIS, result=result,
        budget=budget, patch_float64_le=patch.tobytes(order="C"),
        patch_rows=int(patch.shape[0]), patch_columns=int(patch.shape[1]))

def _bind(readings, weights, sources):
    return m9b.bind_geometry_provenance(
        readings=readings, structural_weights=weights, split_sources=sources)

def _forge(record, **changes):
    """Bypass __post_init__ to model a tampered on-the-wire record."""
    forged = object.__new__(type(record))
    for field in dc_fields(record):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(record, field.name)))
    return forged

def test_bind_seals_exact_records_and_materialises_the_exact_split_source():
    reading = _reading()
    weights = _weights(reading)
    source = _split_source(reading.region_evidence)
    bridge = _bind((reading,), (weights,), (source,))
    assert isinstance(bridge, m9b.GeometryProvenanceBridge)
    assert bridge.readings[0] is reading
    assert bridge.structural_weights[0] is weights
    assert bridge.bridge_sha256 == placement._sha256(m9b._bridge_payload(
        universe=bridge.reading_universe_sha256, readings=bridge.readings,
        structural_weights=bridge.structural_weights, split_sources=bridge.split_sources))
    result, patch, budget = bridge.split_sources_for_assembly()[0]
    assert result is source.result and budget is source.budget
    assert np.array_equal(patch, source.to_source()[1])

def test_record_digests_are_exact_and_a_foreign_digest_is_rejected():
    reading = _reading()
    weights = _weights(reading)
    source = _split_source(reading.region_evidence)
    assert weights.patch_sha256 == m9b._structural_patch_sha256(
        weights.patch_rows, weights.patch_columns, weights.patch_float32_le)
    assert weights.evidence_sha256 == placement._sha256(
        m9b._structural_payload(weights))
    assert source.source_sha256 == placement._sha256(m9b._split_payload(source))
    with pytest.raises(ValueError):
        replace(weights, evidence_sha256="f" * 64)
    with pytest.raises(ValueError):
        replace(source, source_sha256="f" * 64)

def test_tampered_structural_share_is_refused_by_replay():
    reading = _reading()
    share = float(reading.structural_share)
    tampered = _weights(reading, structural_share=(0.0 if share != 0.0 else 1.0))
    refusal = _bind((reading,), (tampered,),
                    (_split_source(reading.region_evidence),))
    assert isinstance(refusal, placement.PlacementRefusal)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "structural_weights.structural_share"

def test_non_reproducing_structural_patch_is_refused_at_binding():
    reading = _reading()
    weights = _weights(reading, _structural_patch(offset=3.0))
    refusal = _bind((reading,), (weights,),
                    (_split_source(reading.region_evidence),))
    assert isinstance(refusal, placement.PlacementRefusal)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "structural_weights.reading_binding"

def test_forged_patch_digest_without_reesal_is_refused():
    reading = _reading()
    forged = _forge(_weights(reading), patch_sha256="0" * 64)
    refusal = _bind((reading,), (forged,),
                    (_split_source(reading.region_evidence),))
    assert isinstance(refusal, placement.PlacementRefusal)
    assert refusal.detail["field"] == "structural_weights.patch_sha256"

def test_foreign_reading_identity_and_duplicate_weights_are_refused():
    reading = _reading()
    weights = _weights(reading)
    source = _split_source(reading.region_evidence)
    foreign = _bind((reading,), (_weights(reading, frame="frame-other"),),
                    (source,))
    assert isinstance(foreign, placement.PlacementRefusal)
    assert foreign.reason == "missing_provenance"
    assert foreign.detail["field"] == "structural_weights.reading"
    duplicate = _bind((reading,), (weights, weights), (source,))
    assert duplicate.reason == "provenance_mismatch"
    assert duplicate.detail["field"] == "structural_weights.duplicate"

def test_split_source_cross_binding_and_replay_tamper_are_refused():
    reading = _reading()
    weights = _weights(reading)
    mismatched = _split_source(reading.region_evidence, source_region_id=None)
    cross = _bind((reading,), (weights,), (mismatched,))
    assert isinstance(cross, placement.PlacementRefusal)
    assert cross.detail["field"] == "split_sources.source_region_id"
    zeroed = _forge(_split_source(reading.region_evidence),
                    patch_float64_le=b"\x00" * (SPLIT_H * SPLIT_W * 8))
    replayed = _bind((reading,), (weights,), (zeroed,))
    assert replayed.reason == "replay_mismatch"
    assert replayed.detail["field"] == "split_sources.patch"

def test_missing_region_phase_evidence_and_missing_coverage_are_refused():
    reading = _reading()
    weights = _weights(reading)
    source = _split_source(reading.region_evidence)
    bare = replace(reading, region_evidence=None, phase_evidence=None)
    missing = _bind((bare,), (weights,), (source,))
    assert isinstance(missing, placement.PlacementRefusal)
    assert missing.reason == "missing_provenance"
    second = replace(reading, frame="frame-1")
    coverage = _bind((reading, second), (weights,), (source,))
    assert coverage.reason == "missing_provenance"
    assert coverage.detail["field"] == "structural_weights.coverage"

def test_refusals_are_typed_and_records_are_frozen():
    assert set(m9b.GEOMETRY_PROVENANCE_REFUSAL_REASONS) <= set(
        placement.V2_PLACEMENT_REFUSAL_REASONS)
    reading = _reading()
    empty = _bind((reading,), (), ())
    assert isinstance(empty, placement.PlacementRefusal)
    assert empty.reason in m9b.GEOMETRY_PROVENANCE_REFUSAL_REASONS
    weights = _weights(reading)
    source = _split_source(reading.region_evidence)
    bridge = _bind((reading,), (weights,), (source,))
    assert bridge.readings[0] is reading
    for record, field in ((weights, "frame"), (source, "frame"),
                          (bridge, "readings")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)
    assert not np.frombuffer(weights.patch_float32_le, dtype="<f4").flags.writeable

def test_constructor_bypassed_split_budget_stale_digest_is_refused():
    reading = _reading()
    source = _split_source(reading.region_evidence)
    forged = _forge(source, budget=_forge(source.budget, max_source_records=999_999))
    object.__setattr__(forged, "source_sha256", placement._sha256(
        m9b._split_payload(forged)))
    refusal = _bind((reading,), (_weights(reading),), (forged,))
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "split_sources.budget"

def test_constructor_bypassed_patch_geometry_is_refused_with_valid_digests():
    reading = _reading()
    with pytest.raises(ValueError):
        _weights(reading, _structural_patch().reshape(8, 32))
    forged = _forge(_weights(reading), patch_rows=8, patch_columns=32)
    array = m9b._structural_array(forged)
    object.__setattr__(forged, "structural_share",
                       float(lp._structural_share(array, PERIOD)))
    object.__setattr__(forged, "phase_evidence", _phase(array, reading.region_evidence))
    object.__setattr__(forged, "patch_sha256", m9b._structural_patch_sha256(
        8, 32, forged.patch_float32_le))
    object.__setattr__(forged, "evidence_sha256",
                       placement._sha256(m9b._structural_payload(forged)))
    refusal = _bind((reading,), (forged,), (_split_source(reading.region_evidence),))
    assert isinstance(refusal, placement.PlacementRefusal)
    assert refusal.detail["field"] == "structural_weights.patch_geometry"
