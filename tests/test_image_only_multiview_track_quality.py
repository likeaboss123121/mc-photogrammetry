"""Frozen research tracks on one fresh native aperiodic production measurement."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from itertools import combinations

import numpy as np

from test_image_only_aperiodic_material_gate import (
    PINS as OLD_PINS, _plain, _array_sha, _truth_counts,
    test_default_image_only_aperiodic_material_gate as _run_native_harness)
from test_image_only_g_prefilter_capture import _file_sha, _semantic
from test_image_only_multiview_gate import ROOT
from mc_photogrammetry.pipeline import graph_tracks as tracks_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_only, joint_matching as joint


BASELINE = "docs/reports/SOL-IMG-APERIODIC-MATERIAL-GATE-20261005.json"
PINS = dict(OLD_PINS, **{
    "tests/test_image_only_aperiodic_material_gate.py":
        "c2e0084fe4ad9c2a30f0dec483bb900c4a1c7d6916db2f0e9c91da51a96162cd",
    BASELINE: "fe6128ed6760a5d1ab9f6b0afbff8fd79d2fb74dd02fc1348191c60a97cfa7d1",
    "docs/reports/SOL-IMG-APERIODIC-MATERIAL-GATE-20261005.txt":
        "2b6e82e649d60edc52ffcb21ded8d498dd2e4d3a1147776501ac51ae5610ceee",
    "src/mc_photogrammetry/pipeline/graph_tracks.py":
        "421fa978c0eb7b98fdf4fa261a7c778f0e1b38c80720bc0fa9e83572b1f36ad4",
})
# Exact DS25 diagnostic defaults, fixed before this run and any truth evaluation.
SETTINGS = dict(neighbors=5, max_edges=8192, ratio_test=.75,
                epipolar_degrees=1., max_cycles=None)


def _encode(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _edges(vertices):
    for (left, a), (right, b) in combinations(sorted(vertices), 2):
        yield (left, right), (int(a), int(b))


def test_default_image_only_multiview_track_quality(tmp_path, monkeypatch):
    assert {path: _file_sha(path) for path in PINS} == PINS
    source_pins = {path: _file_sha(path) for path in (
        "src/mc_photogrammetry/pipeline/graph_tracks.py",
        "src/mc_photogrammetry/pipeline/directed_matching.py",
        "scripts/diagnose_ds25_graph_tracks.py", "tests/test_graph_wide_tracks.py",
        "docs/reports/DS-25-graph-wide-tracks-2026-09-12.md")}
    assert SETTINGS == dict(neighbors=tracks_mod.DEFAULT_NEIGHBORS,
        max_edges=tracks_mod.DEFAULT_MAX_EDGES, ratio_test=tracks_mod.DEFAULT_RATIO_TEST,
        epipolar_degrees=tracks_mod.DEFAULT_EPIPOLAR_DEGREES, max_cycles=None)
    originals = (joint.detect_features, entry.match_and_solve_pair)
    features, cameras, identity_rows, blind = {}, {}, [], {}
    pair_count = 0

    def detect(frame, image, **kwargs):
        result = originals[0](frame, image, **kwargs)
        if frame in features:
            first = features[frame]
            # Hashes supplement exact array equality; keypoint indices are never
            # joined from approximate coordinates, descriptor distance or rank.
            same_points = np.array_equal(first.points, result.points)
            same_descriptors = np.array_equal(first.descriptors, result.descriptors)
            identity_rows.append({"frame": frame, "points_equal": same_points,
                "descriptors_equal": same_descriptors,
                "points_sha256": _array_sha(result.points),
                "descriptors_sha256": _array_sha(result.descriptors)})
            assert same_points and same_descriptors, "unbound_identity: no cycle score permitted"
        else:
            features[frame] = result
        return result

    def freeze_tracks():
        assert pair_count == 6 and len(features) == len(cameras) == 4
        before = {f: (_array_sha(v.points), _array_sha(v.descriptors)) for f, v in features.items()}
        views = []
        for frame in sorted(features):
            estimate = cameras[frame]
            assert estimate.pose.position is None
            model = image_only.camera_model_from_estimate(estimate, position=(0., 0., 0.))
            views.append(tracks_mod.SessionView(frame, features[frame], model,
                (estimate.intrinsics.focal_length_pixels, estimate.intrinsics.principal_point_pixels)))
        graphs = tracks_mod.build_directed_graphs(views, neighbors=SETTINGS["neighbors"],
                                                max_edges=SETTINGS["max_edges"])
        captured_cycles, candidates = [], []
        original_cycles, original_candidate = tracks_mod.closed_cycles, tracks_mod._Candidate

        def cycle_observer(*args, **kwargs):
            result = original_cycles(*args, **kwargs)
            captured_cycles.extend(result)
            return result

        def candidate_observer(*args, **kwargs):
            result = original_candidate(*args, **kwargs)
            candidates.append(result)
            return result

        with monkeypatch.context() as research:
            research.setattr(tracks_mod, "closed_cycles", cycle_observer)
            research.setattr(tracks_mod, "_Candidate", candidate_observer)
            result = tracks_mod.select_tracks(views, graphs,
                ratio_test=SETTINGS["ratio_test"], epipolar_degrees=SETTINGS["epipolar_degrees"],
                max_cycles=SETTINGS["max_cycles"])
        assert tracks_mod.closed_cycles is original_cycles and tracks_mod._Candidate is original_candidate
        assert result.candidates == len(captured_cycles)
        assert before == {f: (_array_sha(v.points), _array_sha(v.descriptors)) for f, v in features.items()}
        derived_pairs = []
        for left, right in combinations(sorted(features), 2):
            vertices = [(t.point_of(left), t.point_of(right)) for t in result.tracks
                        if t.point_of(left) is not None and t.point_of(right) is not None]
            exact_rows = np.asarray([(*features[left].points[a], *features[right].points[b])
                                     for a, b in vertices], dtype=float).reshape((-1, 4))
            order = np.lexsort((exact_rows[:, 2], exact_rows[:, 3], exact_rows[:, 0], exact_rows[:, 1]))
            native_rows = tracks_mod.tracks_to_matches(result.tracks, left, right)
            assert np.array_equal(native_rows, exact_rows[order])
            derived_pairs.append({"left": left, "right": right,
                "endpoint_indices_in_native_output_order": [vertices[int(i)] for i in order],
                "matches": native_rows.tolist(),
                "duplicate_pixel_rows": len(native_rows)-len({tuple(row) for row in native_rows})})
        record = {"settings": SETTINGS, "phase": "frozen before native pair6 returns and before renderer truth",
            "identity_guard": identity_rows, "views": [{"name": v.name,
                "camera": asdict(v.camera), "intrinsics": v.intrinsics,
                "native_pipeline_estimate": asdict(cameras[v.name]),
                "features": asdict(v.features), "array_sha256": before[v.name]} for v in views],
            "directed_graphs": [{"left": l, "right": r, "graph": asdict(graph)} for (l, r), graph in graphs.items()],
            "qualified": [{"left": l, "right": r, "edges": rows} for (l, r), rows in result.qualified.items()],
            "closed_cycles": captured_cycles, "candidates": [asdict(c) for c in candidates],
            "tracks": [asdict(t) for t in result.tracks], "derived_pair_matches": derived_pairs,
            "counts": {"closed_cycles": result.candidates, "candidate_tracks": len(candidates),
                "selected_tracks": len(result.tracks), "discarded_conflicts": result.discarded_conflicts,
                "discarded_unclosed": result.discarded_unclosed}}
        frozen = _encode(record)
        blind.update(record=json.loads(frozen), bytes=frozen,
                     sha256=hashlib.sha256(frozen).hexdigest())

    def pair(left, left_image, right, right_image, **kwargs):
        nonlocal pair_count
        for f, key in ((left, "left_camera"), (right, "right_camera")):
            if f in cameras:
                assert cameras[f] is kwargs[key], "unbound_pipeline_camera_identity"
            else:
                cameras[f] = kwargs[key]
        result = originals[1](left, left_image, right, right_image, **kwargs)
        pair_count += 1
        if pair_count == 6:
            freeze_tracks()  # research output is never handed back to a solver
        return result

    # Reuse the pinned producer harness without changing its behavior. Its own
    # delegates call these outer original-return observers; it still executes
    # one default public call and retains all native camera/G modes and rows.
    with monkeypatch.context() as observe:
        observe.setattr(joint, "detect_features", detect)
        observe.setattr(entry, "match_and_solve_pair", pair)
        _run_native_harness(tmp_path, monkeypatch)
    assert (joint.detect_features, entry.match_and_solve_pair) == originals
    assert len(identity_rows) == 8 and pair_count == 6 and blind
    assert _encode(blind["record"]) == blind["bytes"]
    native_path = tmp_path/"aperiodic-material-gate.json"
    native_bytes = native_path.read_bytes()
    native = json.loads(native_bytes)
    baseline = json.loads((ROOT/BASELINE).read_bytes())
    assert _semantic(native) == _semantic(baseline), "authoritative unchanged default measurement"
    # Only now join/score the frozen research output with the post-hoc oracle.
    packets = {(p["left"], p["right"]): p for p in native["pair_observations"]}
    lookups = {key: {tuple(edge): i for i, edge in enumerate(p["native_graph"]["edges"])}
               for key, p in packets.items()}
    cycle_ids = {key: set() for key in packets}
    track_ids = {key: set() for key in packets}
    unmapped = []

    def joined(vertices, destination):
        rows = []
        for key, indices in _edges(vertices):
            row = lookups[key].get(indices)
            if row is None:
                unmapped.append({"pair": key, "endpoint_indices": indices})
                rows.append({"pair": key, "endpoint_indices": indices, "native_row": None, "truth": None})
            else:
                destination[key].add(row)
                rows.append({"pair": key, "endpoint_indices": indices, "native_row": row,
                             "truth": packets[key]["edge_truth"][row]})
        return rows

    cycle_rows = [{"vertices": vertices, "mapped_edges": joined(vertices, cycle_ids)}
                  for vertices in blind["record"]["closed_cycles"]]
    track_rows = []
    for i, track in enumerate(blind["record"]["tracks"]):
        mapped = joined(zip(track["views"], track["points"]), track_ids)
        known = [e["truth"] for e in mapped if e["truth"] is not None]
        scorable = len(known) == len(mapped) and all(row["scorable"] for row in known)
        track_rows.append({"track_index": i, "native_track": track, "mapped_edges": mapped,
            "fully_scorable": scorable,
            "truth_correct_all_pairs": None if not scorable else all(row["correct_within_2px"] for row in known)})
    derived_joins = []
    for row in blind["record"]["derived_pair_matches"]:
        key = row["left"], row["right"]
        ids = [lookups[key].get(tuple(edge)) for edge in row["endpoint_indices_in_native_output_order"]]
        derived_joins.append(dict(row, mapped_native_rows=ids))
        assert all(i is not None for i in ids), "no alternate graph edges may replace production candidates"
    summary = []
    for key, p in packets.items():
        truth = p["edge_truth"]
        winner_indices = p["modes"][0]["native_mode"]["selection"] if p["modes"] else []
        summary.append({"left": key[0], "right": key[1], "all_candidates": _truth_counts(truth),
            "cycle_supported_native_rows": sorted(cycle_ids[key]),
            "cycle_supported_candidates": _truth_counts([truth[i] for i in sorted(cycle_ids[key])]),
            "selected_track_native_rows": sorted(track_ids[key]),
            "selected_track_candidates": _truth_counts([truth[i] for i in sorted(track_ids[key])]),
            "native_winner": _truth_counts([truth[i] for i in winner_indices]),
            "native_camera_status": p["native_result"]["status"], "native_camera_reason": p["native_result"]["reason"]})
    totals = {}
    for name in ("all_candidates", "cycle_supported_candidates", "selected_track_candidates", "native_winner"):
        total = {k: sum(p[name][k] for p in summary) for k in ("total", "scorable", "correct", "incorrect", "boundary_excluded")}
        total["precision_on_scorable"] = None if not total["scorable"] else total["correct"]/total["scorable"]
        totals[name] = total
    scorable_tracks = sum(t["fully_scorable"] for t in track_rows)
    correct_tracks = sum(t["truth_correct_all_pairs"] is True for t in track_rows)
    proof = {"scope": "research-only exact-ID tracks; one unchanged native aperiodic measurement",
        "decision": "abstain", "prior_pins": PINS, "source_pins": source_pins,
        "native_measurement": native, "native_artifact_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "semantic_baseline_equal": True, "native_pair_solver_inputs_unchanged": True,
        "blind_freeze_sha256": blind["sha256"], "blind_tracks": blind["record"],
        "cycle_rows": cycle_rows, "track_rows": track_rows, "derived_pair_exact_joins": derived_joins,
        "unmapped_edges": unmapped, "pair_summary": summary, "totals": totals,
        "track_truth": {"total": len(track_rows), "scorable": scorable_tracks,
            "correct": correct_tracks,
            "precision_on_scorable": None if not scorable_tracks else correct_tracks/scorable_tracks},
        "limits": "one public call; frozen DS25 defaults,12 research directed graphs/one select_tracks; <=256cycles/256tracks/8192edges-per-graph/8MB cooperative assertions, not hard runtime containment or typed universal overflow",
        "prior_negative": "DS25 T6 selected3/scorable3/true0; vacuous closure is documented; no presumed geometric discrimination",
        "geometry": "authoritative default result; research matches not supplied to any solver, no promotion/export claim"}
    assert {path: _file_sha(path) for path in PINS} == PINS
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    encoded = _encode(proof)
    artifact = tmp_path/"multiview-track-quality.json"
    artifact.write_bytes(encoded)
    print("TRACK_QUALITY_ARTIFACT", str(artifact), len(encoded), hashlib.sha256(encoded).hexdigest())
    print("TRACK_QUALITY_RESULT", json.dumps({"identity_comparisons": len(identity_rows),
        "counts": blind["record"]["counts"], "totals": totals, "track_truth": proof["track_truth"],
        "unmapped": len(unmapped), "production_status": native["production"]["status"]}, sort_keys=True))
    assert not unmapped
    assert len(encoded) < 8_000_000
    assert len(cycle_rows) <= 256 and len(track_rows) <= 256
    assert all(len(g["graph"]["edges"]) <=8192 for g in blind["record"]["directed_graphs"])
