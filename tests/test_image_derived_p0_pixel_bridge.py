"""Generated RGB -> unmodified camera/reader/P0; no injected scene metadata."""
from __future__ import annotations

import base64
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from test_image_derived_lattice_measure import _scene_rgb, _pose_graph
from test_b0prime_d_redistribution import PINS
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.pipeline import image_derived_signatures as signatures
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.reconstruction import lattice, lattice_planes as lp, plane_graph
from mc_photogrammetry.reconstruction import plane_placement
from mc_photogrammetry.vision.gui_mask import create_gui_mask


ROOT = Path(__file__).resolve().parents[1]
PRIOR = dict(PINS, **{
    "tests/test_b0prime_d_redistribution.py":
        "eb6eea92fdb662d457710bb433ed7530e7507c915fd9859828606e3a62715087",
    "docs/reports/DS-IMG-B0PRIME-D-REDISTRIBUTION-20261005.json":
        "304a67ddde63bf162429a095ddd9b7b7de0e1f84888e677c8728fa7cd7287ee9",
    "docs/reports/DS-IMG-B0PRIME-D-REDISTRIBUTION-20261005.txt":
        "643b6e4059516258e508d5bea325c229c81cc8a29e6d7f469309cbbc34559818",
})


def _file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run_rgb(rgb):
    # Only pixels enter the accepted session: no patched camera solver, render
    # calibration, supplied reading, mask, rectified patch or geometry.
    session = sessions.build_image_derived_session({"f0": rgb})
    pose = _pose_graph(session)
    result = measure.measure_image_derived_lattice(session=session, pose_graph=pose)
    record = {"session": sessions.image_derived_session_to_record(session),
              "pose": pose.to_record(), "lattice": result.to_record()}
    if isinstance(result, plane_placement.PlacementRefusal):
        return session, result, None, record
    assert result.terminal == "geometry_unavailable"
    if result.status != "assembled":
        return session, result, None, record
    derived = signatures.derive_image_derived_signatures(session=session,
                                                       lattice_result=result)
    record["signatures"] = dict(signatures._result_payload(derived),
                                result_sha256=derived.result_sha256)
    record["signature_records"] = [signatures._signature_payload(s)
                                   for s in derived.signatures]
    if derived.status == "assembled":
        sink = []
        matches = plane_graph.match_plane_graph(result.readings,
            tuple(s.signature for s in derived.signatures),
            max_shift_cells=32, candidate_observation_sink=sink)
        assert matches == () and len(sink) == 1
        assert sink[0].candidates == sink[0].dispositions == ()
        record["matcher"] = {"max_shift_cells": 32, "matches": [], "pool": []}
    return session, result, derived, record


def _profiles(result, session):
    records = []
    if not isinstance(result, measure.ImageDerivedLatticeResult) or result.bridge is None:
        return records
    structural = {(s.frame, s.canonical_index): s for s in result.bridge.structural_weights}
    by_index = measure._canonical_indices(result.readings)
    readings = {(r.frame, by_index[id(r)]): r for r in result.readings}
    weight_images = {}
    for frame in session.frames:
        rgb = measure._reader_rgb(frame.image)
        usable = ~create_gui_mask(*rgb.shape[:2])
        weight_images[frame.frame_id] = measure._frame_weight_image(rgb, usable)
    for source in result.bridge.split_sources:
        key = (source.frame, source.canonical_index)
        reading = readings[key]
        p0, patch, budget = source.to_source()
        evidence = structural[key]
        assert patch.shape == (reading.samples, reading.samples)
        assert reading.samples <= measure.LATTICE_MAX_RECTIFIED_SIDE
        rebuilt, rebuilt_patch = measure._structural_evidence(
            reading, weight_images[reading.frame], source.canonical_index)
        assert rebuilt.evidence_sha256 == evidence.evidence_sha256
        assert np.array_equal(rebuilt_patch, patch)
        assert lp.validate_split_phase_evidence(p0, patch, geometry=p0.geometry,
            partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
        measured_share = lp._structural_share(patch, reading.period_pixels)
        measured_phase = lp._periodic_energy_phase_evidence(patch,
            reading.period_pixels, centre=reading.rectification.centre,
            extent=reading.extent, samples=reading.samples,
            source_region_id=reading.region_evidence.source_region_id)
        assert measured_share == evidence.structural_share == reading.structural_share
        assert measured_phase == evidence.phase_evidence == reading.phase_evidence
        rows = []
        for seam in p0.seam_profile_evidence:
            samples = np.frombuffer(base64.b64decode(
                seam.profile_samples_float64_le_base64), dtype="<f8")
            weights = np.frombuffer(base64.b64decode(
                seam.profile_weights_float64_le_base64), dtype="<f8")
            assert len(samples) == len(weights) == reading.samples
            lane = next(s for s in p0.lane_sets
                        if s.local_axis == seam.local_axis and s.role == "D")
            expected = (patch[lane.row_start:lane.row_stop, :].sum(axis=0)
                if seam.local_axis == "u" else
                patch[:, lane.column_start:lane.column_stop].sum(axis=1))
            assert np.array_equal(samples, expected)
            # Local diagnostic, not an absolute-world V2-B anchor prediction.
            # Zero coordinate origin is declared here; no endpoint/translation
            # is supplied, and every retained b alternative is reported.
            bands = []
            for alternative in reading.alternatives:
                b = Fraction(alternative.blocks_per_period)
                coordinates = [(b * i / Fraction(seam.period_pixels)) % 1
                               for i in range(len(samples))]
                for anchor in (Fraction(), Fraction(1, 2)):
                    def distance(x, a):
                        d = abs(x - a)
                        return min(d, 1-d)
                    boundary = [i for i, x in enumerate(coordinates)
                        if distance(x, anchor) <= Fraction(1, 8)]
                    interior = [i for i, x in enumerate(coordinates)
                        if distance(x, (anchor + Fraction(1, 2)) % 1) <= Fraction(1, 8)]
                    assert not set(boundary) & set(interior)
                    def energy(indices):
                        denominator = sum(int(weights[i]) for i in indices)
                        numerator = sum((Fraction(float(samples[i])) for i in indices), Fraction())
                        value = numerator / denominator if denominator else None
                        return {"sample_sum": [numerator.numerator, numerator.denominator],
                            "weight_sum": denominator, "energy": None if value is None else
                            [value.numerator, value.denominator]}
                    be, ie = energy(boundary), energy(interior)
                    margin = (Fraction(*be["energy"]) - Fraction(*ie["energy"])) \
                        if be["energy"] is not None and ie["energy"] is not None else None
                    bands.append({"b": float(b), "anchor": float(anchor),
                        "boundary_indices": boundary, "interior_indices": interior,
                        "boundary": be, "interior": ie,
                        "margin": None if margin is None else
                                  [margin.numerator, margin.denominator]})
            rows.append({"seam": seam.to_record(), "samples": samples.tolist(),
                         "weights": weights.tolist(), "local_band_diagnostics": bands})
        records.append({"identity": p0.reading_identity,
            "structural_share": measured_share, "full_patch_phase": asdict(measured_phase),
            "generation": p0.generation_evidence[0].to_record(),
            "p0_sha256": p0.result_sha256, "source_sha256": source.source_sha256,
            "structural_sha256": evidence.evidence_sha256,
            "rgb_weight_image_sha256": _sha(weight_images[reading.frame]),
            "patch_sha256": _sha(patch), "lanes": [s.to_record() for s in p0.lane_sets],
            "geometry": p0.geometry.to_record(), "profiles": rows})
    return records


def _d_only_rgb_patch(pixels, baseline):
    """One measured D/D square, disjoint from conservative baseline G taps.

    Select the first canonical reading with a positive D/D sample whose 5x5
    projected square is inside the normal usable area and outside all G tap
    neighbourhoods. Rank its samples by descending structural weight then row/
    column. This is a pixel intervention rule, not a successful-arm search.
    """
    height, width = pixels.shape[:2]
    usable = ~create_gui_mask(height, width)
    index = measure._canonical_indices(baseline.readings)
    by_key = {(s.frame, s.canonical_index): s for s in baseline.bridge.split_sources}
    for reading in baseline.readings:
        source = by_key[(reading.frame, index[id(reading)])]
        p0, patch, _ = source.to_source()
        n = reading.samples
        normal, u_basis, v_basis = lattice.plane_basis(reading.axis,
            np.asarray(reading.rectification.axis_matrix))
        u, v = np.meshgrid(
            np.linspace(reading.rectification.centre[0]-reading.extent,
                        reading.rectification.centre[0]+reading.extent, n),
            np.linspace(reading.rectification.centre[1]-reading.extent,
                        reading.rectification.centre[1]+reading.extent, n))
        points = reading.side*normal + u[..., None]*u_basis + v[..., None]*v_basis
        depth = points[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = (reading.rectification.principal_point[0] +
                  points[..., 0]*reading.rectification.focal/depth).astype(np.float32)
            ys = (reading.rectification.principal_point[1] -
                  points[..., 1]*reading.rectification.focal/depth).astype(np.float32)
        valid = (depth > 1e-6) & np.isfinite(xs) & np.isfinite(ys)
        # Same sampling maps as production rectify_plane; the wider [-1,+2]
        # neighbourhood conservatively includes OpenCV's quantized bilinear taps.
        floors_x = np.floor(np.where(valid, xs, -10.)).astype(np.int64)
        floors_y = np.floor(np.where(valid, ys, -10.)).astype(np.int64)
        g_u = next(s for s in p0.lane_sets if s.local_axis == "u" and s.role == "G")
        g_v = next(s for s in p0.lane_sets if s.local_axis == "v" and s.role == "G")
        generation = np.zeros((n, n), dtype=bool)
        generation[g_u.row_start:g_u.row_stop, :] = True
        generation[:, g_v.column_start:g_v.column_stop] = True
        generation &= valid
        g_pixels = np.zeros((height, width), dtype=bool)
        for dy in (-1, 0, 1, 2):
            for dx in (-1, 0, 1, 2):
                rows, columns = floors_y[generation]+dy, floors_x[generation]+dx
                inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
                g_pixels[rows[inside], columns[inside]] = True
        d_u = next(s for s in p0.lane_sets if s.local_axis == "u" and s.role == "D")
        d_v = next(s for s in p0.lane_sets if s.local_axis == "v" and s.role == "D")
        candidates = [(r, c) for r in range(d_u.row_start, d_u.row_stop)
            for c in range(d_v.column_start, d_v.column_stop)
            if valid[r, c] and patch[r, c] > 0.]
        candidates.sort(key=lambda rc: (-float(patch[rc]), *rc))
        for r, c in candidates:
            row, column = int(round(float(ys[r, c]))), int(round(float(xs[r, c])))
            if not (2 <= row < height-2 and 2 <= column < width-2):
                continue
            bounds = (row-2, row+3, column-2, column+3)
            r0, r1, c0, c1 = bounds
            if g_pixels[r0:r1, c0:c1].any() or not usable[r0:r1, c0:c1].all():
                continue
            intervention = pixels.copy()
            intervention[r0:r1, c0:c1] //= 2
            assert not np.array_equal(intervention, pixels)
            assert np.count_nonzero(np.any(intervention != pixels, axis=2)) <= 25
            return intervention, reading, source, {
                "rule": "halve RGB uint8 in the first eligible 5x5 projected D/D square",
                "canonical_index": index[id(reading)], "reading_axis": reading.axis,
                "rectified_sample_rc": [r, c], "rgb_bounds_r0_r1_c0_c1": bounds,
                "baseline_sample_weight": float(patch[r, c]),
                "baseline_G_source_tap_overlap": 0,
                "conservative_G_source_pixels": int(g_pixels.sum()),
                "before_rgb": pixels[r0:r1, c0:c1].tolist(),
                "after_rgb": intervention[r0:r1, c0:c1].tolist()}
    raise AssertionError("No nonzero D/D projected square is isolated from baseline G taps")


def test_generated_rgb_image_only_p0_bridge(tmp_path):
    assert {path: _file_sha(path) for path in PRIOR} == PRIOR
    production_paths = ["src/mc_photogrammetry/" + name + ".py" for name in (
        "pipeline/image_only", "pipeline/image_derived_session",
        "pipeline/image_derived_lattice_measure", "pipeline/image_derived_signatures",
        "reconstruction/lattice", "reconstruction/lattice_planes",
        "reconstruction/calibration", "reconstruction/plane_geometry_provenance",
        "reconstruction/plane_phase_anchor", "reconstruction/plane_graph")]
    source_pins = {path: _file_sha(path) for path in production_paths}
    solver = image_only.solve_image_cameras
    assert solver.__module__ == "mc_photogrammetry.pipeline.image_only"
    pixels = np.array(_scene_rgb(), copy=True)
    baseline = _run_rgb(pixels)
    profile_records = _profiles(baseline[1], baseline[0])
    assert profile_records, "generated RGB camera/reader/P0 baseline did not assemble"
    altered, reading, old_source, mutation = _d_only_rgb_patch(pixels, baseline[1])
    intervention = _run_rgb(altered)
    intervention_profiles = _profiles(intervention[1], intervention[0])
    assert intervention_profiles, "RGB intervention did not reach a fresh P0 bridge"
    # A measured frame modification must never reuse old image/session/source
    # seals. Exact camera and reading invariants are reported, not forced.
    assert sessions.image_sha256(pixels) != sessions.image_sha256(altered)
    assert baseline[0].session_sha256 != intervention[0].session_sha256
    assert baseline[1].bridge.bridge_sha256 != intervention[1].bridge.bridge_sha256
    usable = ~create_gui_mask(*pixels.shape[:2])
    old_weight_image = measure._frame_weight_image(pixels, usable)
    new_weight_image = measure._frame_weight_image(altered, usable)
    r = reading.rectification
    held_geometry_patch = lattice.rectify_plane(new_weight_image, reading.axis,
        np.asarray(r.axis_matrix), r.focal, extent=reading.extent,
        samples=reading.samples, centre=r.centre, principal_point=r.principal_point,
        side=reading.side)
    old_patch = old_source.to_source()[1]
    assert held_geometry_patch is not None
    changed_share = lp._structural_share(held_geometry_patch, reading.period_pixels)
    changed_phase = lp._periodic_energy_phase_evidence(held_geometry_patch,
        reading.period_pixels, centre=r.centre, extent=reading.extent,
        samples=reading.samples, source_region_id=reading.region_evidence.source_region_id)
    binding_refusal = None
    try:
        measure._structural_evidence(reading, new_weight_image,
                                     mutation["canonical_index"])
    except measure._InputRefusal as refused:
        binding_refusal = {"reason": refused.reason, "detail": refused.detail}
    assert binding_refusal == {"reason": "provenance_mismatch",
                               "detail": {"field": "structural_weights.reading_binding"}}
    lanes = old_source.result.lane_sets
    lane_changes = []
    for lane in lanes:
        sl = (slice(lane.row_start, lane.row_stop), slice(None)) if lane.local_axis == "u" \
            else (slice(None), slice(lane.column_start, lane.column_stop))
        lane_changes.append({"axis": lane.local_axis, "role": lane.role,
            "changed_samples": int(np.count_nonzero(old_patch[sl] != held_geometry_patch[sl])),
            "before_sha256": _sha(np.asarray(old_patch[sl], dtype="<f8")),
            "after_sha256": _sha(np.asarray(held_geometry_patch[sl], dtype="<f8")),
            "digest_encoding": "float64-le"})
    assert [(s["role"], s["changed_samples"]) for s in lane_changes] == [
        ("G", 0), ("D", 1340), ("G", 0), ("D", 1340)]
    assert all(s["before_sha256"] == s["after_sha256"] for s in lane_changes if s["role"] == "G")
    comparisons = {
        "camera_records_equal": baseline[3]["session"]["cameras"] == intervention[3]["session"]["cameras"],
        "reading_universe_equal": baseline[1].bridge.reading_universe_sha256 ==
                                  intervention[1].bridge.reading_universe_sha256,
        "signature_records_equal": baseline[3]["signature_records"] == intervention[3]["signature_records"],
        "matcher_equal": baseline[3]["matcher"] == intervention[3]["matcher"],
        "target_full_phase_equal_at_held_geometry": reading.phase_evidence == changed_phase,
        "target_structural_share_equal_at_held_geometry": reading.structural_share == changed_share}
    assert not comparisons["target_full_phase_equal_at_held_geometry"] or \
           not comparisons["target_structural_share_equal_at_held_geometry"]
    assert not comparisons["reading_universe_equal"]
    assert image_only.solve_image_cameras is solver
    assert {path: _file_sha(path) for path in PRIOR} == PRIOR
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    record = {"baseline": baseline[3], "baseline_profiles": profile_records,
        "intervention": intervention[3], "intervention_profiles": intervention_profiles,
        "mutation": mutation, "comparisons": comparisons,
        "held_geometry_negative_control": {"binding_refusal": binding_refusal,
            "old_structural_share": reading.structural_share,
            "changed_structural_share": changed_share,
            "old_full_phase": asdict(reading.phase_evidence),
            "changed_full_phase": asdict(changed_phase), "lane_changes": lane_changes,
            "old_weight_image_sha256": _sha(old_weight_image),
            "changed_weight_image_sha256": _sha(new_weight_image)},
        "prior_pins": PRIOR, "production_source_pins": source_pins,
        "decision": "abstain",
        "geometry": "unavailable; no placement/volume called",
        "camera_input": "only generated RGB, unmodified production solve_image_cameras",
        "limitations": "single generated frame; empty inter-frame matcher control; local D energies only, no world anchor or label selection; obstruction applies to this intervention only"}
    artifact = tmp_path / "image-derived-p0-pixel-bridge.json"
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    assert len(encoded) < 1_000_000  # cooperative publication guard only
    artifact.write_bytes(encoded)
    print("PIXEL_BRIDGE_ARTIFACT", str(artifact), artifact.stat().st_size,
          hashlib.sha256(artifact.read_bytes()).hexdigest())
    print("PIXEL_BRIDGE_RESULT", json.dumps({"comparisons": comparisons,
        "lane_changes": lane_changes, "binding_refusal": binding_refusal,
        "reading_counts": [len(profile_records), len(intervention_profiles)]}, sort_keys=True))
