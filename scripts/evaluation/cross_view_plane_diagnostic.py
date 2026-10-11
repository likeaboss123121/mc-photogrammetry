"""One-shot, scalar-only cross-view appearance diagnostic; never seed authority.

The dedicated runner temporarily bypasses FILE persistence, replaying the same
accepted session record in memory. Matching, cameras, proposal, masks and lattice
measurement use ordinary defaults. No mcscan, litematic, render or export occurs.
Windows supervisor enforcement is sampled, not OS memory containment.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline.image_derived_connections import (
    _PairCorrespondenceObservation, _array_sha256, _MATCH_DIGEST_DOMAIN)
from mc_photogrammetry.reconstruction import region_growth as rg
from scripts.evaluation.cross_view_plane_consistency import diagnose_plane_union

LIMITS = dict(frames=6, pairs=15, native_pixels_per_frame=6_000_000,
              working_side=960, snapshot_bytes=128*1024*1024,
              matches_per_observation=64, unions_per_observation=16,
              unions_total=128, models_per_union=128,
              wall_seconds=120, rss_bytes=1024*1024*1024, report_bytes=16384)
SELECTION = "adjacent_source_atoms_min6_each_total_support_desc_then_left_right_atom_ids"
INVALID_JOIN_REASONS = frozenset({
    "incomplete_correspondences", "missing_source_atom_snapshot",
    "source_frame_binding_mismatch", "invalid_atom_snapshot_bound",
    "invalid_native_mask_bound", "atom_snapshot_digest_mismatch",
    "native_gui_mask_digest_mismatch", "source_coordinate_mapping_mismatch",
    "native_working_mask_mismatch", "atom_mask_label_mismatch",
    "pair_coordinate_digest_mismatch", "pair_coordinates_outside_native_frame",
    "correspondence_in_unusable_atom"})


def inherited_input_seals():
    """Read source literal only; do not import another diagnostic or open images."""
    tree = ast.parse((REPO_ROOT / "scripts/evaluation/partial_core_image_diagnostic.py").read_text())
    rows = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(name, ast.Name) and name.id == "INPUT_SEALS" for name in node.targets)]
    if len(rows) != 1 or len(rows[0]) != LIMITS["frames"]:
        raise ValueError("inherited six-input manifest missing")
    return rows[0]


def _rss_bytes():
    """Current process working set; the outer monitor also accounts descendants."""
    if sys.platform != "win32":
        raise RuntimeError("Windows supervisor required")
    class Memory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak_working", ctypes.c_size_t), ("working", ctypes.c_size_t),
                    ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                    ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    memory = Memory(); memory.cb = ctypes.sizeof(memory)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
        raise OSError("working-set census failed")
    return int(memory.working)


class StopDiagnostic(Exception):
    pass


class Watch:
    def __init__(self, *, clock=time.monotonic, rss=_rss_bytes):
        self.clock, self.rss = clock, rss
        self.start, self.peak = clock(), 0

    def check(self):
        elapsed, working = self.clock()-self.start, self.rss()
        self.peak = max(self.peak, working)
        if elapsed >= LIMITS["wall_seconds"]:
            raise StopDiagnostic("wall_cap")
        if working > LIMITS["rss_bytes"]:
            raise StopDiagnostic("rss_cap")


class Capture:
    def __init__(self, watch):
        self.watch = watch
        self.pairs, self.atoms = {}, {}
        self.snapshot_bytes, self.denials = 0, Counter()

    def discard(self, reason):
        self.denials["observer_error"] += 1

    def pair(self, observation):
        try:
            self.watch.check()
            if type(observation) is not _PairCorrespondenceObservation:
                raise StopDiagnostic("pair_type")
            key = (observation.left_id, observation.right_id)
            if key in self.pairs or len(self.pairs) >= LIMITS["pairs"]:
                raise StopDiagnostic("duplicate_or_excess_pair")
            self.pairs[key] = observation
        except Exception as error:
            self.denials[str(error) if type(error) is StopDiagnostic else "observer_error"] += 1

    def atom(self, snapshot):
        try:
            self.watch.check()
            if type(snapshot) is not rg._DiagnosticAtomSnapshot:
                raise StopDiagnostic("snapshot_type")
            if snapshot.frame_id in self.atoms or len(self.atoms) >= LIMITS["frames"]:
                raise StopDiagnostic("duplicate_or_excess_snapshot")
            size = sum(len(value) for value in (snapshot.labels_bytes, snapshot.usable_bytes,
                                                snapshot.native_usable_bytes))
            if self.snapshot_bytes+size > LIMITS["snapshot_bytes"]:
                raise StopDiagnostic("snapshot_payload_cap")
            self.atoms[snapshot.frame_id] = snapshot
            self.snapshot_bytes += size
        except Exception as error:
            self.denials[str(error) if type(error) is StopDiagnostic else "observer_error"] += 1

    def clear(self):
        self.pairs.clear(); self.atoms.clear()


@contextmanager
def _memory_entry(seals):
    """Dedicated single-thread diagnostic scope; restore both hooks in finally."""
    original_discover, original_persist = entry_mod._discover, entry_mod._persist_session
    expected = {str((REPO_ROOT / row["path"]).resolve()): row for row in seals}
    seen = set()
    def discover(paths, *, recursive, ingress_sink):
        def sink(path, frame, data):
            row = expected.get(str(Path(path).resolve()))
            if (row is None or path in seen or frame.frame_id != row["frame"]
                    or len(data) != row["bytes"]
                    or hashlib.sha256(data).hexdigest().upper() != row["sha256"]):
                raise StopDiagnostic("input_ingress_seal")
            with Image.open(io.BytesIO(data)) as header:
                if math.prod(header.size) > LIMITS["native_pixels_per_frame"]:
                    raise StopDiagnostic("native_pixel_cap")
            seen.add(path)
            ingress_sink(path, frame, data)
        result = original_discover(paths, recursive=recursive, ingress_sink=sink)
        if len(seen) != LIMITS["frames"]:
            raise StopDiagnostic("missing_input_ingress")
        return result
    def persist(*, session, **kwargs):
        # No file is created. Validate the identical accepted record in memory.
        record = session_mod.image_derived_session_to_record(session)
        rebuilt = session_mod.image_derived_session_from_record(record)
        if rebuilt.session_sha256 != session.session_sha256:
            raise StopDiagnostic("memory_session_replay")
        return None
    entry_mod._discover, entry_mod._persist_session = discover, persist
    try:
        yield
    finally:
        entry_mod._discover, entry_mod._persist_session = original_discover, original_persist


def _capture_state(observation):
    """Census-only truncation is not usable evidence; never promote its prefix.

    Full source digest is joined to the canonical same-pass graph in _reconcile.
    A truncated prefix cannot independently replay that digest and is not modeled.
    Complete captures must replay every ordered source row exactly.
    """
    if (type(observation.source_count) is not int or observation.source_count < 0
            or type(observation.complete) is not bool or type(observation.matches) is not tuple
            or len(observation.matches) > LIMITS["matches_per_observation"]):
        raise StopDiagnostic("malformed_correspondence_capture")
    if any(type(row) is not tuple or len(row) != 4
            or any(type(value) is not float or not math.isfinite(value) for value in row)
            for row in observation.matches):
        raise StopDiagnostic("malformed_correspondence_rows")
    if observation.source_count > LIMITS["matches_per_observation"]:
        if len(observation.matches) == LIMITS["matches_per_observation"] and observation.complete is False:
            return "capture_truncated_over64"
        raise StopDiagnostic("malformed_correspondence_capture")
    if observation.complete is not True or len(observation.matches) != observation.source_count:
        raise StopDiagnostic("incomplete_correspondences")
    values = np.asarray(observation.matches, dtype=np.float64).reshape(-1, 4)
    if _array_sha256(values, _MATCH_DIGEST_DOMAIN) != observation.matches_sha256:
        raise StopDiagnostic("pair_coordinate_digest_mismatch")
    return "complete"


def _reconcile(capture, entry):
    frames = {frame.frame_id: frame for frame in entry.session.frames}
    if len(frames) != LIMITS["frames"] or set(capture.atoms) != set(frames):
        raise StopDiagnostic("missing_or_foreign_atom_snapshot")
    for name, frame in frames.items():
        snapshot = capture.atoms[name]
        if (snapshot.frame_id != name or snapshot.session_sha256 != entry.session.session_sha256
                or snapshot.pixel_sha256 != frame.image_sha256 or snapshot.source_shape != frame.image_shape):
            raise StopDiagnostic("source_atom_frame_mismatch")
        if (hashlib.sha256(snapshot.labels_bytes).hexdigest() != snapshot.labels_sha256
                or hashlib.sha256(snapshot.usable_bytes).hexdigest() != snapshot.usable_sha256
                or rg._diagnostic_native_mask_digest(name, snapshot.source_shape,
                    snapshot.native_usable_bytes) != snapshot.native_usable_sha256
                or rg._diagnostic_snapshot_digest(snapshot) != snapshot.snapshot_sha256):
            raise StopDiagnostic("atom_snapshot_digest_mismatch")
    graph = entry.graph
    if graph is None or graph.status != "ok":
        raise StopDiagnostic("source_pair_graph_refused")
    expected = {(row.left_id, row.right_id): row for row in graph.evidence}
    if set(capture.pairs) != set(expected):
        raise StopDiagnostic("missing_or_foreign_pair_observation")
    states = {}
    for key, observation in sorted(capture.pairs.items()):
        row = expected[key]
        if ((observation.left_id, observation.right_id) != key
                or observation.evidence_sha256 != row.evidence_sha256
                or observation.matches_sha256 != row.matches_sha256
                or observation.source_count != row.match_count
                or observation.status != row.status
                or observation.reason != row.reason
                or observation.parallax_degrees != row.parallax_degrees
                or observation.ambiguity_margin != row.ambiguity_margin
                or observation.session_sha256 != entry.session.session_sha256):
            raise StopDiagnostic("source_pair_evidence_mismatch")
        for name, pixel, shape in ((observation.left_id, observation.left_pixel_sha256,
                                    observation.left_shape),
                                   (observation.right_id, observation.right_pixel_sha256,
                                    observation.right_shape)):
            frame = frames.get(name)
            if frame is None or pixel != frame.image_sha256 or shape != frame.image_shape:
                raise StopDiagnostic("source_pair_frame_mismatch")
        states[key] = _capture_state(observation)
    return states


def _candidates(capture):
    ranked = []
    for key, observation in sorted(capture.pairs.items()):
        capture.watch.check()
        if (observation.status != "solved" or observation.complete is not True
                or observation.source_count > LIMITS["matches_per_observation"]):
            continue
        snapshot = capture.atoms[observation.left_id]
        # The accepted helper validates ALL source bytes, masks, coordinate rows,
        # identities and mapping before its adjacency/model/holdout checks.
        if (type(snapshot.working_shape) is not tuple or len(snapshot.working_shape) != 2
                or any(type(value) is not int or not 0 < value <= 960 for value in snapshot.working_shape)
                or type(snapshot.labels_bytes) is not bytes
                or len(snapshot.labels_bytes) != math.prod(snapshot.working_shape)*4):
            raise StopDiagnostic("invalid_atom_snapshot_bound")
        bounded_labels = np.frombuffer(snapshot.labels_bytes, "<i4").reshape(snapshot.working_shape)
        present = set(int(node) for node in np.unique(bounded_labels))
        absent = tuple(node for node in range(20_000) if node not in present)[:2]
        if len(absent) != 2:
            raise StopDiagnostic("atom_validation_probe_unavailable")
        # Absent ids force adjacency refusal, so this validation call executes
        # ZERO model hypotheses and cannot inspect holdout or choose a union.
        audit = diagnose_plane_union(observation, snapshot, absent)
        if audit["reason"] in INVALID_JOIN_REASONS:
            raise StopDiagnostic(audit["reason"])
        labels = np.frombuffer(snapshot.labels_bytes, "<i4").reshape(snapshot.working_shape)
        points = np.asarray(observation.matches, np.float64).reshape(-1, 4)
        memberships = rg._diagnostic_atom_membership(labels,
            rg._diagnostic_native_to_working(points[:, :2], snapshot.native_to_working_scale))
        counts = Counter(memberships)
        supported = sorted(node for node, count in counts.items() if node >= 0 and count >= 6)
        local = []
        for i, first in enumerate(supported):
            for second in supported[i+1:]:
                adjacent = any(np.any(((a == first) & (b == second)) | ((a == second) & (b == first)))
                    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])))
                if adjacent:
                    local.append((-(counts[first]+counts[second]), *key, first, second))
        if len(local) > LIMITS["unions_per_observation"]:
            raise StopDiagnostic("union_observation_cap")
        ranked.extend(local)
        if len(ranked) > LIMITS["unions_total"]:
            raise StopDiagnostic("union_total_cap")
    return sorted(ranked)


def summarize(capture, entry):
    report = dict(status="incomplete", diagnostic_only=True, seed_authority=False,
                  appearance_identity_verified=False, physical_coplanarity_proven=False,
                  reconstruction_gain_claimed=False, selection_rule=SELECTION,
                  limits=dict(LIMITS), frame_count=len(entry.session.frames),
                  pair_observations=len(capture.pairs), atom_snapshots=len(capture.atoms),
                  snapshot_payload_bytes=capture.snapshot_bytes, unions_evaluated=0,
                  correspondences_evaluated=0, hypotheses_evaluated=0,
                  fitting_hypotheses=0, status_counts={}, reason_counts={}, denials={},
                  pair_capture_counts={}, pair_skip_counts={}, pair_rows=[],
                  complete_solved_pairs=0, capture_census_reconciled=False)
    statuses, reasons = Counter(), Counter()
    pair_statuses, pair_reasons = Counter(), Counter()
    try:
        if capture.denials:
            raise StopDiagnostic("observer_incomplete")
        capture_states = _reconcile(capture, entry)
        skips = Counter()
        report["capture_census_reconciled"] = True
        report["pair_capture_counts"] = dict(sorted(Counter(capture_states.values()).items()))
        for key, observation in sorted(capture.pairs.items()):
            pair_statuses[observation.status] += 1
            state = capture_states[key]
            skip = (state if state != "complete" else
                    "source_pair_refused" if observation.status != "solved" else None)
            if skip is not None:
                skips[skip] += 1
            else:
                report["complete_solved_pairs"] += 1
            report["pair_rows"].append(dict(left_id=key[0], right_id=key[1],
                source_count=observation.source_count, captured_count=len(observation.matches),
                status=observation.status, source_reason=observation.reason,
                capture_status=state, model_skip_reason=skip))
            if observation.status != "solved":
                pair_reasons[observation.reason] += 1
        report["pair_skip_counts"] = dict(sorted(skips.items()))
        for _, left, right, first, second in _candidates(capture):
            capture.watch.check()
            row = diagnose_plane_union(capture.pairs[left, right], capture.atoms[left], (first, second))
            report["unions_evaluated"] += 1
            report["correspondences_evaluated"] += row["correspondences"]
            report["hypotheses_evaluated"] += row["hypotheses"]
            report["fitting_hypotheses"] += row["fitting_hypotheses"]
            statuses[row["status"]] += 1; reasons[row["reason"]] += 1
            if row["reason"] in INVALID_JOIN_REASONS or row["reason"] == "hypothesis_budget_exhausted":
                raise StopDiagnostic(row["reason"])
        capture.watch.check()
        report["status"] = "complete"
    except StopDiagnostic as error:
        capture.denials[str(error)] += 1
    report.update(status_counts=dict(sorted(statuses.items())), reason_counts=dict(sorted(reasons.items())),
                  pair_status_counts=dict(sorted(pair_statuses.items())),
                  refused_pair_reason_counts=dict(sorted(pair_reasons.items())),
                  denials=dict(sorted(capture.denials.items())))
    return report


def _write_exclusive(path, report):
    content = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(content) > LIMITS["report_bytes"]:
        raise StopDiagnostic("report_byte_cap")
    with Path(path).open("xb") as output:
        output.write(content)


def run_diagnostic(report_output, ingress_output, *, watch=None):
    watch = Watch() if watch is None else watch
    capture, entry = Capture(watch), None
    report = None
    try:
        watch.check()
        seals = inherited_input_seals()
        with _memory_entry(seals):
            entry = entry_mod._build_image_derived_entry(
                [(REPO_ROOT / row["path"]).resolve() for row in seals],
                project_path=ingress_output, capture_ingress=True,
                _correspondence_observer=capture.pair)
        watch.check()
        measure_mod.measure_image_derived_lattice(session=entry.session, pose_graph=entry.pose_graph,
                                                  _atom_snapshot_observer=capture.atom)
        report = summarize(capture, entry)
    except Exception as error:
        reason = str(error) if type(error) is StopDiagnostic else "source_stage_exception"
        report = dict(status="incomplete", diagnostic_only=True, seed_authority=False,
                      reconstruction_gain_claimed=False, denials={reason: 1},
                      pair_observations=len(capture.pairs), atom_snapshots=len(capture.atoms))
    finally:
        capture.clear()
        entry = None
    report.update(schema="mc-cross-view-plane-diagnostic-report-v2",
                  wall_seconds=round(watch.clock()-watch.start, 6), sampled_peak_rss_bytes=watch.peak)
    # Supervisor's project_output slot holds only scalar accounting, never a
    # persisted production session/camera record. Both files use exclusive create.
    _write_exclusive(ingress_output, dict(status=report["status"], diagnostic_only=True,
                                        pair_observations=report["pair_observations"],
                                        atom_snapshots=report["atom_snapshots"], seed_authority=False))
    _write_exclusive(report_output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--ingress-output", required=True)
    args = parser.parse_args(argv)
    report_path, ingress_path = Path(args.report_output).resolve(), Path(args.ingress_output).resolve()
    if (report_path == ingress_path or report_path.exists() or ingress_path.exists()
            or report_path.parent != ingress_path.parent or not report_path.parent.is_dir()):
        parser.error("distinct absent outputs in pre-reserved fresh directory required")
    report = run_diagnostic(report_path, ingress_path)
    print(json.dumps({"status": report["status"], "seed_authority": False}, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
