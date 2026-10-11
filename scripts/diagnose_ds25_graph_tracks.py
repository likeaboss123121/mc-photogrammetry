"""DS-25 - graph-wide multi-view cycle tracks, frozen before any truth read.

Directive (user, 2026-09-12): DS-24 showed that image-derived *pairwise* match
sets never reach the preregistered gate under the frozen DS-23 blind selector.
DS-25 tests a different representation: build graph-wide keypoint tracks over a
complete session from the frozen directed top-k candidate graphs, admit only
tracks backed by closed 3-cycles on exact keypoint identities, resolve competing
repeated-texture tracks with a label-blind score, and then hand the derived
pairwise match sets to the frozen DS-23 selector.

Phase order inside one process:

* **Phase A (blind).** Reads the digest-verified DS-16 frozen features/estimates
  only (pixels + image-derived M3 geometry). Builds, for every complete session
  (T2's three views, T6's four), the frozen directed top-k candidate graphs,
  enumerates closed 3-cycles under the frozen descriptor ratio test, resolves
  competing tracks with the preregistered label-blind score and derives the nine
  pairs' one-to-one match sets. Writes ``candidate_graphs.json``, ``tracks.json``,
  ``tracks.npz``, ``pair_match_sets.json`` and ``pair_match_sets.npz``. No
  manifest, litematic, scene transform, reference class, oracle edge, evaluator
  output or fixture constant is opened.
* **Freeze.** ``freeze.json`` records the SHA-256 of the track arrays and of the
  pair-match-set arrays; phase B refuses to start if either digest moved.
* **Phase B (blind).** Re-reads the frozen tracks/pair sets from disk and applies
  ``scripts/blind_pose_policy.select_blind_pose`` to each pair. Writes
  ``decisions.json``.
* **Phase C (post-freeze).** The only place reference data is opened: the fixture
  camera manifests (reference yaw classes), the reference renders used for the
  post-hoc track truth fraction, the immutable DS-16B evaluator, and the DS-24
  A/B comparator (recomputed from the frozen features so the comparison is not a
  hard-coded number). Nothing from phase C feeds back into phase A or B.

Preregistered promotion (single variant, fixed before the run): usable poses
>= 8/9, yaw class correct >= 8/9, gauge-aligned baseline <= 5 degrees, zero
regressions among DS-14's four class-correct pairs, <= 1 false solve, and the
existing DS-23 synthetic degeneracy controls. A failed result stays diagnostic;
nothing is tuned after a truth read.

Run once with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts\\diagnose_ds25_graph_tracks.py

No production source, artifact, schema, CLI or desktop file is modified; no
commit and no delegation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair  # noqa: E402
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import FrameFeatures  # noqa: E402
from mc_photogrammetry.pipeline.directed_matching import (  # noqa: E402
    match_feature_frames as directed_match_feature_frames,
)
from mc_photogrammetry.pipeline.graph_tracks import (  # noqa: E402
    DEFAULT_EPIPOLAR_DEGREES,
    DEFAULT_MAX_EDGES,
    DEFAULT_NEIGHBORS,
    DEFAULT_RATIO_TEST,
    SessionView,
    Track,
    build_directed_graphs,
    select_tracks,
    tracks_to_matches,
    tracks_to_records,
)
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings  # noqa: E402
from mc_photogrammetry.pipeline.joint_matching import (  # noqa: E402
    match_feature_frames as joint_match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import CameraModel  # noqa: E402

# Blind-script helpers only. Reference accessors are resolved lazily inside the
# phase C functions so a poisoning test can intercept them on their source
# modules and prove the blind phases never reach them.
import blind_pose_policy as blind_policy  # noqa: E402
import diagnose_blind_pose_policy as ds23  # noqa: E402
import diagnose_connected_pairs as connected  # noqa: E402
import diagnose_ds24_image_derived_pose as ds24  # noqa: E402
import diagnose_match_attribution as attribution  # noqa: E402

DS16_DIR = ROOT / "exports" / "diagnostics" / "ds16"
DS23_DIR = ROOT / "exports" / "diagnostics" / "ds23"
DS24_DIR = ROOT / "exports" / "diagnostics" / "ds24"
OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds25"

# --------------------------------------------------------------------------- #
# Preregistered constants. Frozen before the run; none of them is tuned after a
# reference read.
# --------------------------------------------------------------------------- #
PAIR_COUNT = 9  # DS-16 indices 0..8 = T2's three views and T6's four views.
SESSIONS = (("T2", 3), ("T6", 4))
VARIANT = "graph_wide_cycle_tracks"

NEIGHBORS = DEFAULT_NEIGHBORS
MAX_EDGES = DEFAULT_MAX_EDGES
RATIO_TEST = DEFAULT_RATIO_TEST
EPIPOLAR_DEGREES = DEFAULT_EPIPOLAR_DEGREES

RENDER_SCALE = attribution.REFERENCE_RENDER_SCALE  # 0.35, frozen DS-16/DS-23
WORLD_HIT_DISTANCE_BLOCKS = 0.5  # frozen DS-16 oracle tolerance
FACE_NORMAL_COSINE_MIN = float(np.cos(np.radians(30.0)))
ALIGNMENT_MEDIAN_PX_MAX = 10.0

PROMOTION_USABLE_REQUIRED = 8
PROMOTION_CLASS_CORRECT_REQUIRED = 8
PROMOTION_GAUGE_BASELINE_DEGREES_MAX = 5.0
PROMOTION_FALSE_SOLVES_MAX = 1
PROMOTION_REGRESSIONS_MAX = 0

DS14_BASELINE_SOLVED_REQUIRED = 5
DS14_BASELINE_CLASS_CORRECT_REQUIRED = 4
DS23_ORACLE_BASELINE_SOLVED_REQUIRED = 6
DS23_ORACLE_BLIND_USABLE_REQUIRED = 9
DS24_A_USABLE_RECORDED = 1
DS24_B_USABLE_RECORDED = 1

RSS_LIMIT_BYTES = 1_800 * 1024 * 1024

# Digest-pinned blind inputs read before any reference access.
BLIND_INPUT_PATHS = {
    "exports/diagnostics/ds16/match_candidates.json": (
        "e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93"
    ),
    "exports/diagnostics/ds16/match_candidates.npz": (
        "3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4"
    ),
}
# The immutable evaluator and the frozen selector boundary.
EVALUATOR_PATHS = {
    "scripts/diagnose_match_attribution.py": (
        "aa822efa4c40b44fd227c3c6d8448567817b062e4600ce24450fddd006be9340"
    ),
    "scripts/diagnose_connected_pairs.py": (
        "43c7c4769f576cf8e7c69f5d0c634e95b89b53248d95c292f81584f4e55d8abd"
    ),
    "scripts/blind_pose_policy.py": (
        "2c5edb1ff088b81a5b6e196542af4eb7725d1afacc0a91c376734f5a946c7d4c"
    ),
}
DS23_TAPE_PATHS = {
    "exports/diagnostics/ds23/blind_tape.json": (
        "72a1259d9307ee21646eb87c2a72bc581ec25a551658a4fe9e648a6d193e8a23"
    ),
    "exports/diagnostics/ds23/blind_tape.npz": (
        "763933ef890414d28516af4c0e67ad21d8c4f0687921b63fb1e5d1849cd603a5"
    ),
}

PAIR_CSV_COLUMNS = (
    "pair", "set", "fixture", "left", "right",
    "left_keypoints", "right_keypoints",
    "tracks_touching_pair", "selected_correspondences",
    "decision_solved", "decision_reason", "decision_class",
    "decision_inliers", "decision_inlier_fraction",
    "decision_median_rho_degrees", "decision_parallax_degrees",
    "decision_cheirality_fraction", "decision_singular_ratio",
    "decision_class_margin", "trace_reads",
    "reference_class", "reference_class_floors",
    "evaluated_class_correct", "evaluated_within_5deg",
    "evaluated_rotation_error_degrees",
    "evaluated_baseline_error_degrees_gauge_aligned",
    "evaluated_baseline_error_degrees_signless", "evaluated_gauge_index",
    "usable", "false_solve",
    "selected_scorable", "selected_true", "selected_truth_fraction",
    "ds24_A_usable", "ds24_A_class_correct", "ds24_B_usable", "ds24_B_class_correct",
)


class MemoryBudgetExceeded(RuntimeError):
    """The run cannot stay inside the preregistered RSS budget."""


class _Log:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8")
        self.peak_rss_bytes: int | None = None

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def checkpoint(self, label: str) -> None:
        import ctypes
        from ctypes import wintypes

        if sys.platform != "win32":
            return
        if not hasattr(self, "_probe"):
            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = ctypes.c_void_p(kernel32.GetCurrentProcess())

            def probe():
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return int(counters.PeakWorkingSetSize)

            self._probe = probe
        peak = self._probe()
        if peak is None:
            return
        if self.peak_rss_bytes is None or peak > self.peak_rss_bytes:
            self.peak_rss_bytes = peak
        if peak > RSS_LIMIT_BYTES:
            self(f"[abort ] peak RSS {peak / 2**20:.1f} MiB exceeds "
                  f"{RSS_LIMIT_BYTES / 2**20:.0f} MiB at {label}; stopping")
            raise MemoryBudgetExceeded(f"peak RSS {peak / 2**20:.1f} MiB at {label}")

    def close(self) -> None:
        self._handle.close()


def relative_str(path: Path) -> str:
    """Repo-relative path when the file lives under the repo, else absolute."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def digest_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_digest(payload) -> str:
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=connected.plain)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_digests(records: dict[str, str], label: str, log) -> dict:
    verified: dict[str, dict] = {}
    for relative, expected in records.items():
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"blocker: {label} input is missing: {relative}")
        actual = connected.digest(path)
        verified[relative] = {"sha256": actual, "expected": expected, "ok": actual == expected}
        if actual != expected:
            raise SystemExit(
                f"blocker: {label} input {relative} was modified: sha256 {actual} "
                f"!= preregistered {expected}"
            )
    log(f"[verify] {label}: {len(records)} digest(s) match their preregistered value")
    return verified


# --------------------------------------------------------------------------- #
# Blind geometry helpers (pixels + image-derived M3 estimates only).
# --------------------------------------------------------------------------- #
def image_estimate(record: dict) -> CameraEstimate:
    """Image-derived (M3) estimate carried in a DS-16 frozen pair record."""
    principal = record["principal_point"]
    return CameraEstimate(
        intrinsics=CameraIntrinsics(
            image_size=(int(record["width"]), int(record["height"])),
            focal_length_pixels=(float(record["focal_pixels"]), float(record["focal_pixels"])),
            principal_point_pixels=(float(principal[0]), float(principal[1])),
        ),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(record["yaw_degrees"]),
            pitch_degrees=float(record["pitch_degrees"]),
            roll_degrees=float(record["roll_degrees"]),
        ),
    )


def pose_model(estimate: CameraEstimate) -> CameraModel:
    pose = estimate.pose
    return CameraModel(
        position=(0.0, 0.0, 0.0),
        yaw_degrees=float(pose.yaw_degrees),
        pitch_degrees=float(pose.pitch_degrees),
        roll_degrees=float(pose.roll_degrees),
        eye_height=0.0,
    )


def frame_features(record: dict, arrays) -> FrameFeatures:
    key = record["key"]
    return FrameFeatures(
        frame_id=record["frame"],
        points=np.asarray(arrays[f"feat_points_{key}"], dtype=np.float64),
        descriptors=np.asarray(arrays[f"feat_desc_{key}"], dtype=np.float64),
        width=int(record["width"]),
        height=int(record["height"]),
        usable_pixels=int(record["usable_pixels"]),
        sizes=np.asarray(arrays[f"feat_sizes_{key}"], dtype=np.float64),
        boundary_dropped=int(record["boundary_dropped"]),
    )


def session_views(session: dict, frozen: dict, arrays) -> list[SessionView]:
    """One ``SessionView`` per frame of a frozen frame set (blind inputs only)."""
    frames = {record["frame"]: record for record in frozen["frames"]}
    frame_set = next(
        entry for entry in frozen["frame_sets"] if entry["set"] == session["set"]
    )
    pairs = [
        pair for pair in frozen["pairs"]
        if pair["index"] < PAIR_COUNT and pair["set"] == session["set"]
    ]
    estimates: dict[str, dict] = {}
    for pair in pairs:
        for name, record in ((pair["left"], pair["left_estimate"]),
                             (pair["right"], pair["right_estimate"])):
            previous = estimates.setdefault(name, record)
            if previous != record:
                raise SystemExit(
                    f"blocker: frame {name} carries two different frozen estimates"
                )
    views: list[SessionView] = []
    for name in sorted(frame_set["frames"]):
        record = frames[name]
        estimate = image_estimate(estimates[name])
        views.append(
            SessionView(
                name=name,
                features=frame_features(record, arrays),
                camera=pose_model(estimate),
                intrinsics=(
                    tuple(float(value) for value in estimate.intrinsics.focal_length_pixels),
                    tuple(float(value) for value in estimate.intrinsics.principal_point_pixels),
                ),
            )
        )
    return views


def _tracks_from_records(records: list[dict]) -> list[Track]:
    tracks = []
    for row in records:
        tracks.append(
            Track(
                views=tuple(row["views"]),
                points=tuple(int(value) for value in row["points"]),
                cycles=int(row["cycles"]),
                descriptor_cost=float(row["descriptor_cost"]),
                epipolar_residual_degrees=float(row["epipolar_residual_degrees"]),
                cycle_support_ratio=float(row["cycle_support_ratio"]),
                epipolar_test=float(row["epipolar_test"]),
                score=float(row["score"]),
                pixels=tuple((float(p[0]), float(p[1])) for p in row["pixels"]),
            )
        )
    return tracks


def _intrinsics_record(estimate: dict) -> dict:
    return {
        "image_size": [int(estimate["width"]), int(estimate["height"])],
        "focal_length_pixels": [float(estimate["focal_pixels"])] * 2,
        "principal_point_pixels": [float(value) for value in estimate["principal_point"]],
    }


def _pose_record(estimate: dict) -> dict:
    return {
        "position": None,
        "yaw_degrees": float(estimate["yaw_degrees"]),
        "pitch_degrees": float(estimate["pitch_degrees"]),
        "roll_degrees": float(estimate["roll_degrees"]),
    }


def _view_support(tracks) -> dict:
    support: dict[str, int] = {}
    for track in tracks:
        key = str(track.cardinality)
        support[key] = support.get(key, 0) + 1
    return dict(sorted(support.items()))


# --------------------------------------------------------------------------- #
# Phase A: blind candidate graphs, tracks and pair match sets.
# --------------------------------------------------------------------------- #
def build_session_tracks(
    frozen: dict, arrays, *, session_filter=None, neighbors=NEIGHBORS,
    max_edges=MAX_EDGES, ratio_test=RATIO_TEST, epipolar_degrees=EPIPOLAR_DEGREES,
    log=None, checkpoint=None,
) -> tuple[dict, dict, list[dict], dict, list[dict], list[dict]]:
    """Build every session's frozen graphs and tracks (blind, digest-agnostic)."""
    session_records: list[dict] = []
    array_payload: dict[str, np.ndarray] = {}
    graph_payload: dict[str, dict] = {}
    expected_sessions = [
        {"set": entry["set"], "views": len(entry["frames"]), "fixture": entry["fixture"],
         "frames": list(entry["frames"])}
        for entry in frozen["frame_sets"]
        if entry["set"] in {name for name, _count in SESSIONS}
    ]
    expected_sessions = sorted(expected_sessions, key=lambda entry: entry["set"])
    if session_filter is not None:
        wanted = {str(value) for value in session_filter}
        expected_sessions = [entry for entry in expected_sessions if entry["set"] in wanted]
    if not expected_sessions:
        raise SystemExit("blocker: no complete session selected")

    track_records: list[dict] = []
    offset = 0
    for session in expected_sessions:
        label = session["set"]
        views = session_views(session, frozen, arrays)
        graphs = build_directed_graphs(views, neighbors=neighbors, max_edges=max_edges)
        result = select_tracks(
            views, graphs, ratio_test=ratio_test, epipolar_degrees=epipolar_degrees
        )
        if log is not None:
            log(
                f"[tracks] {label:12s} views {len(views)} graphs {len(graphs)} "
                f"cycles {result.candidates} tracks {len(result.tracks)} "
                f"(conflicts {result.discarded_conflicts}, "
                f"unclosed {result.discarded_unclosed})"
            )
        graph_payload[label] = {
            "views": [view.name for view in views],
            "pairs": {
                f"{left}->{right}": {
                    "edges": int(len(graph.edges)),
                    "total_edges": int(graph.total_edges),
                    "truncated": int(graph.truncated),
                }
                for (left, right), graph in sorted(graphs.items())
            },
        }
        for track in result.tracks:
            record = tracks_to_records([track])[0]
            record["slot"] = offset + len(track_records)
            record["session"] = label
            track_records.append(record)
            array_payload[f"track_points_{record['slot']}"] = np.asarray(
                track.points, dtype=np.int64
            )
        session_records.append(
            {
                "set": label,
                "fixture": session["fixture"],
                "views": [view.name for view in views],
                "graphs": len(graphs),
                "cycles": int(result.candidates),
                "tracks": len(result.tracks),
                "discarded_conflicts": int(result.discarded_conflicts),
                "discarded_unclosed": int(result.discarded_unclosed),
                "view_support": _view_support(result.tracks),
            }
        )
        offset += len(result.tracks)
        if checkpoint is not None:
            checkpoint(f"session {label}")

    pair_records: list[dict] = []
    pair_arrays: dict[str, np.ndarray] = {}
    for pair in [entry for entry in frozen["pairs"] if entry["index"] < PAIR_COUNT]:
        slot = int(pair["index"])
        session_label = pair["set"]
        records = [row for row in track_records if row["session"] == session_label]
        tracks = _tracks_from_records(records)
        matches = tracks_to_matches(tracks, pair["left"], pair["right"])
        pair_arrays[f"pair_matches_{slot}"] = matches
        pair_records.append(
            {
                "slot": slot,
                "set": session_label,
                "fixture": pair["fixture"],
                "left": pair["left"],
                "right": pair["right"],
                "width": int(pair["left_estimate"]["width"]),
                "height": int(pair["left_estimate"]["height"]),
                "left_estimate": pair["left_estimate"],
                "right_estimate": pair["right_estimate"],
                "left_intrinsics": _intrinsics_record(pair["left_estimate"]),
                "right_intrinsics": _intrinsics_record(pair["right_estimate"]),
                "left_pose": _pose_record(pair["left_estimate"]),
                "right_pose": _pose_record(pair["right_estimate"]),
                "tracks_touching_pair": int(
                    sum(
                        1
                        for row in records
                        if pair["left"] in row["views"] and pair["right"] in row["views"]
                    )
                ),
                "selected_correspondences": int(len(matches)),
                "matches_sha256": digest_array(matches),
            }
        )
    return (
        graph_payload, array_payload, pair_records, pair_arrays,
        session_records, track_records,
    )


def write_phase_a(
    out_dir: Path, frozen: dict, arrays, *, session_filter=None, log=None,
    checkpoint=None,
) -> dict:
    """Phase A: write and digest the blind graphs, tracks and pair match sets."""
    (
        graphs, track_arrays, pair_records, pair_arrays, sessions, track_records,
    ) = build_session_tracks(
        frozen, arrays, session_filter=session_filter, log=log, checkpoint=checkpoint
    )

    graphs_path = out_dir / "candidate_graphs.json"
    graphs_payload = {
        "schema": "ds25-candidate-graphs-1",
        "phase": "pre_reference_frozen_directed_candidate_graphs",
        "settings": {
            "neighbors": NEIGHBORS,
            "max_edges": MAX_EDGES,
            "distance_ceiling": None,
            "descriptor": "frozen DS-16 SIFT descriptors; no re-detection, no retuning",
        },
        "sessions": graphs,
        "forbidden_in_graphs": [
            "fixture manifests / reference camera poses",
            "litematics, scene transforms and rendered block labels",
            "reference yaw classes, oracle edge sets and per-pair truth counts",
            "frozen evaluator outputs or any reference score",
        ],
    }
    graphs_path.write_text(
        json.dumps(graphs_payload, indent=2, default=connected.plain), encoding="utf-8"
    )

    tracks_path = out_dir / "tracks.npz"
    np.savez_compressed(tracks_path, **track_arrays)
    track_array_digests = {
        key: digest_array(value) for key, value in sorted(track_arrays.items())
    }
    track_payload = {
        "schema": "ds25-tracks-1",
        "phase": "pre_reference_resolved_tracks",
        "rules": {
            "candidate_graph": (
                "directed left->right top-k (k=5, cap 8192, no distance ceiling) over the "
                "frozen DS-16 descriptors; no reciprocity requirement"
            ),
            "cycle": (
                "closed 3-cycle on exact keypoint identities across three distinct frames; "
                "all six directed edges must be ratio-qualified at the frozen 0.75 test"
            ),
            "track": (
                "union of overlapping closed cycles; T2 (three views) requires support in "
                "all three views; T6 (four views) requires at least one closed triangle "
                "and consistency across >=3 views"
            ),
            "closure": (
                "worst image-derived epipolar residual over the track's view pairs, each "
                "pair's baseline direction the bounded 3x3 Gram/SVD null vector of its "
                "d-as-cross-d rows (DS-18/DS-19 mechanism, no RANSAC)"
            ),
            "score": (
                "cycle_support_ratio + max(0, 1 - worst_residual/1deg) + "
                "max(0, 1 - mean normalized descriptor cost); ties break on residual, "
                "descriptor cost, view names and keypoint indices"
            ),
            "conflict_rule": (
                "greedy global one-to-one: best score first; a track is dropped when any "
                "view already claimed its keypoint"
            ),
        },
        "constants": {
            "neighbors": NEIGHBORS,
            "max_edges": MAX_EDGES,
            "ratio_test": RATIO_TEST,
            "epipolar_inlier_degrees": EPIPOLAR_DEGREES,
        },
        "sessions": sessions,
        "track_count": len(track_records),
        "tracks": track_records,
        "arrays": {
            "path": relative_str(tracks_path),
            "file_sha256": connected.digest(tracks_path),
            "sha256_by_array": track_array_digests,
            "arrays_sha256": canonical_digest(track_array_digests),
        },
    }
    tracks_json = out_dir / "tracks.json"
    tracks_json.write_text(
        json.dumps(track_payload, indent=2, default=connected.plain), encoding="utf-8"
    )

    pairs_path = out_dir / "pair_match_sets.npz"
    np.savez_compressed(pairs_path, **pair_arrays)
    pair_array_digests = {
        key: digest_array(value) for key, value in sorted(pair_arrays.items())
    }
    pairs_payload = {
        "schema": "ds25-pair-match-sets-1",
        "phase": "pre_selection_pair_match_sets_derived_from_frozen_tracks",
        "rule": (
            "one correspondence per resolved track that carries a keypoint in both views; "
            "no pairwise matcher, no pairwise threshold and no re-detection"
        ),
        "arrays": {
            "path": relative_str(pairs_path),
            "file_sha256": connected.digest(pairs_path),
            "sha256_by_array": pair_array_digests,
            "arrays_sha256": canonical_digest(pair_array_digests),
        },
        "pairs": pair_records,
    }
    pairs_json = out_dir / "pair_match_sets.json"
    pairs_json.write_text(
        json.dumps(pairs_payload, indent=2, default=connected.plain), encoding="utf-8"
    )
    if log is not None:
        log(
            f"[tape  ] candidate_graphs.json {connected.digest(graphs_path)[:16]}...; "
            f"tracks.json {connected.digest(tracks_json)[:16]}...; "
            f"pair_match_sets.json {connected.digest(pairs_json)[:16]}..."
        )
        log(
            f"[tape  ] track arrays {track_payload['arrays']['arrays_sha256'][:16]}...; "
            f"pair arrays {pairs_payload['arrays']['arrays_sha256'][:16]}..."
        )
    return {
        "graphs": {"path": relative_str(graphs_path),
                   "sha256": connected.digest(graphs_path)},
        "tracks": {"path": relative_str(tracks_json),
                   "sha256": connected.digest(tracks_json),
                   "npz": relative_str(tracks_path),
                   "arrays_sha256": track_payload["arrays"]["arrays_sha256"],
                   "sha256_by_array": track_array_digests},
        "pair_match_sets": {
            "path": relative_str(pairs_json),
            "sha256": connected.digest(pairs_json),
            "npz": relative_str(pairs_path),
            "arrays_sha256": pairs_payload["arrays"]["arrays_sha256"],
            "sha256_by_array": pair_array_digests,
        },
        "sessions": sessions,
        "pair_records": pair_records,
        "track_records": track_records,
    }


def read_frozen_phase_a(out_dir: Path, log) -> tuple[dict, dict, dict, dict]:
    """Re-read and verify the frozen tracks/pair sets before selection."""
    tracks_json = out_dir / "tracks.json"
    tracks_npz = out_dir / "tracks.npz"
    pairs_json = out_dir / "pair_match_sets.json"
    pairs_npz = out_dir / "pair_match_sets.npz"
    track_payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    pair_payload = json.loads(pairs_json.read_text(encoding="utf-8"))
    for payload, npz, label in (
        (track_payload, tracks_npz, "tracks"),
        (pair_payload, pairs_npz, "pair match sets"),
    ):
        if connected.digest(npz) != payload["arrays"]["file_sha256"]:
            raise SystemExit(f"blocker: {label} archive changed after it was written")
        arrays = dict(np.load(npz))
        actual = {key: digest_array(value) for key, value in sorted(arrays.items())}
        if actual != payload["arrays"]["sha256_by_array"]:
            raise SystemExit(f"blocker: {label} arrays changed after they were frozen")
        if canonical_digest(actual) != payload["arrays"]["arrays_sha256"]:
            raise SystemExit(f"blocker: {label} array digest does not reproduce")
    track_arrays = dict(np.load(tracks_npz))
    pair_arrays = dict(np.load(pairs_npz))
    log(
        f"[tape  ] re-read frozen tracks arrays "
        f"{track_payload['arrays']['arrays_sha256'][:16]}... and pair arrays "
        f"{pair_payload['arrays']['arrays_sha256'][:16]}... verified"
    )
    return track_payload, track_arrays, pair_payload, pair_arrays


def write_freeze(out_dir: Path, phase_a: dict, log) -> dict:
    payload = {
        "schema": "ds25-freeze-1",
        "phase": "tracks_and_pair_match_sets_frozen_before_any_truth_read",
        "candidate_graphs": phase_a["graphs"],
        "tracks": phase_a["tracks"],
        "pair_match_sets": phase_a["pair_match_sets"],
        "reference_fields_absent": [
            "reference_class", "reference_class_floors", "oracle", "truth",
            "evaluator", "usable", "class_equals_expected", "within_5deg",
            "baseline_error_degrees_gauge_aligned", "baseline_error_degrees_signless",
        ],
        "note": (
            "Phase B refuses to start unless every digest above reproduces; no phase A/B "
            "function opens a fixture manifest, litematic, scene transform, reference "
            "class, oracle edge or evaluator output."
        ),
    }
    path = out_dir / "freeze.json"
    path.write_text(json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8")
    payload["file_sha256"] = connected.digest(path)
    log(f"[freeze] tracks + pair sets hashed and frozen: freeze.json "
        f"{payload['file_sha256'][:16]}...; reference evaluation may start now")
    return payload


# --------------------------------------------------------------------------- #
# Phase B: the frozen DS-23 selector, on the frozen tracks only.
# --------------------------------------------------------------------------- #
def tape_blind_input(pair: dict, matches: np.ndarray):
    """Build the selector input from the *written* pair record only."""

    def _intrinsics(record: dict) -> CameraIntrinsics:
        return CameraIntrinsics(
            image_size=tuple(int(value) for value in record["image_size"]),
            focal_length_pixels=tuple(float(value) for value in record["focal_length_pixels"]),
            principal_point_pixels=tuple(
                float(value) for value in record["principal_point_pixels"]
            ),
        )

    def _pose(record: dict) -> CameraPose:
        return CameraPose(
            position=None,
            yaw_degrees=float(record["yaw_degrees"]),
            pitch_degrees=float(record["pitch_degrees"]),
            roll_degrees=float(record["roll_degrees"]),
        )

    return blind_policy.to_blind_input(
        left_intrinsics=_intrinsics(pair["left_intrinsics"]),
        left_pose=_pose(pair["left_pose"]),
        right_intrinsics=_intrinsics(pair["right_intrinsics"]),
        right_pose=_pose(pair["right_pose"]),
        matches=np.asarray(matches, dtype=np.float64).reshape(-1, 4),
        image_size=(int(pair["width"]), int(pair["height"])),
    )


def select_pairs(pair_payload: dict, pair_arrays: dict) -> tuple[list[dict], list[str]]:
    """Run the frozen DS-23 selector on every derived pair match set."""
    decisions: list[dict] = []
    read_union: set[str] = set()
    for pair in pair_payload["pairs"]:
        slot = int(pair["slot"])
        matches = pair_arrays[f"pair_matches_{slot}"]
        blind = tape_blind_input(pair, matches)
        blind_policy.start_trace(blind, blind.left_intrinsics, blind.right_intrinsics)
        decision = blind_policy.select_blind_pose(blind)
        trace = blind_policy.stop_trace()
        reads = tuple(sorted(set(trace.names())))
        read_union.update(reads)
        decisions.append(
            {
                "slot": slot,
                "set": pair["set"],
                "pair": f"{pair['left']}->{pair['right']}",
                "solved": bool(decision.solved),
                "reason": decision.reason,
                "right_yaw_class": (
                    None if decision.right_yaw_class is None
                    else int(decision.right_yaw_class)
                ),
                "baseline_direction": (
                    None if decision.baseline_direction is None
                    else [float(value) for value in decision.baseline_direction]
                ),
                "inliers": int(decision.inliers),
                "inlier_fraction": float(decision.inlier_fraction),
                "median_rho_degrees": decision.median_rho_degrees,
                "parallax_degrees": decision.parallax_degrees,
                "cheirality_fraction": decision.cheirality_fraction,
                "singular_ratio": decision.singular_ratio,
                "class_margin": decision.class_margin,
                "stability": decision.stability,
                "selected_correspondences": int(pair["selected_correspondences"]),
                "matches_sha256": pair["matches_sha256"],
                "trace_reads": list(reads),
                "unexpected_reads": [
                    name for name in reads
                    if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
                ],
            }
        )
    return decisions, sorted(read_union)


def write_decisions(out_dir: Path, decisions: list[dict], read_union: list[str], log) -> dict:
    payload = {
        "schema": "ds25-blind-decisions-1",
        "phase": "pre_reference_blind_selection",
        "selector": "scripts/blind_pose_policy.select_blind_pose",
        "frozen_policy": {
            "edge_consensus_degrees": blind_policy.FROZEN_POLICY.edge_consensus_degrees,
            "edge_consensus_min": blind_policy.FROZEN_POLICY.edge_consensus_min,
            "edge_consensus_fraction": blind_policy.FROZEN_POLICY.edge_consensus_fraction,
            "edge_parallax_min_degrees": blind_policy.FROZEN_POLICY.edge_parallax_min_degrees,
            "edge_cheirality_min": blind_policy.FROZEN_POLICY.edge_cheirality_min,
            "edge_rank_min_ratio": blind_policy.FROZEN_POLICY.edge_rank_min_ratio,
            "class_margin_min": blind_policy.FROZEN_POLICY.class_margin_min,
            "class_margin_fraction": blind_policy.FROZEN_POLICY.class_margin_fraction,
            "stability_degrees": blind_policy.FROZEN_POLICY.stability_degrees,
            "stratum_min_edges": blind_policy.FROZEN_POLICY.stratum_min_edges,
            "spatial_grid": list(blind_policy.FROZEN_POLICY.spatial_grid),
        },
        "trace_reads_union": read_union,
        "trace_unexpected_reads": sorted(
            name for name in read_union
            if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
        ),
        "decisions": decisions,
    }
    path = out_dir / "decisions.json"
    path.write_text(json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8")
    digests = {
        "path": relative_str(path),
        "file_sha256": connected.digest(path),
        "decisions_sha256": canonical_digest(decisions),
    }
    log(
        f"[blind ] froze {len(decisions)} decisions; decisions digest "
        f"{digests['decisions_sha256'][:16]}..."
    )
    return digests


# --------------------------------------------------------------------------- #
# Phase C: reference-only evaluation (manifests, renders, frozen evaluator).
# --------------------------------------------------------------------------- #
def phase_c_reference(pair_payload: dict, log) -> tuple[dict, dict]:
    """Load the reference manifests and yaw classes for the nine pairs."""
    import mc_photogrammetry.reconstruction.camera as camera_module

    fixtures: dict[str, dict] = {}
    for pair in pair_payload["pairs"]:
        fixture = pair["fixture"]
        if fixture in fixtures:
            continue
        manifest_path = ROOT / fixture / "cameras.json"
        if not manifest_path.exists():
            raise SystemExit(f"blocker: reference manifest missing: {manifest_path}")
        fixtures[fixture] = {
            "path": relative_str(manifest_path),
            "sha256": connected.digest(manifest_path),
            "cameras": camera_module.load_camera_manifest(manifest_path),
        }
    references: dict[int, dict] = {}
    for pair in pair_payload["pairs"]:
        cameras = fixtures[pair["fixture"]]["cameras"]
        left_model = cameras[pair["left"]]
        right_model = cameras[pair["right"]]
        left_estimate = connected._estimate_from_record(pair["left_estimate"])
        right_estimate = connected._estimate_from_record(pair["right_estimate"])
        direct, floors = connected._reference_class(
            left_estimate, right_estimate, left_model, right_model
        )
        references[int(pair["slot"])] = {
            "reference_class": int(direct),
            "reference_class_floors": int(floors),
            "evaluator_ambiguity": bool(direct != floors),
            "left_model": left_model,
            "right_model": right_model,
        }
    log(f"[ref   ] opened {len(fixtures)} reference camera manifest(s); "
        f"floor/direct disagreements "
        f"{sum(1 for row in references.values() if row['evaluator_ambiguity'])}/9")
    return fixtures, references


def reference_truth(
    pair_payload: dict, track_payload: dict, frozen: dict, fixtures: dict, log,
) -> dict:
    """Post-freeze reference rendering: per-track and per-pair geometric truth.

    Implements the frozen DS-16 geometric-true rule post hoc: a drawn edge is
    true when both endpoints land on reference pixels and their world points are
    within the frozen tolerance with compatible face normals. A track is true
    when every scorable view pair of the track is true.
    """
    import os

    from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
    from mc_photogrammetry.export.attribution import (
        face_colours,
        face_model_faces,
        face_occluding_slots,
        face_shapes,
        render_result,
    )
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.reconstruction.scene import load_scene_transform
    from mc_photogrammetry.vision.gui_mask import create_gui_mask

    provider = VanillaJarTextureProvider()
    jar_path = Path(os.environ["APPDATA"]) / ".minecraft/versions/1.21.5/1.21.5.jar"
    provider.load_jar(jar_path)
    palette = provider.complete_palette()
    colours = face_colours(palette)
    shapes = face_shapes(palette)
    occluding_slots = face_occluding_slots(palette)
    model_faces = face_model_faces(palette)

    frames = {record["frame"]: record for record in frozen["frames"]}
    arrays = dict(np.load(ROOT / frozen["candidates_npz"]["path"]))
    needed = sorted({pair["left"] for pair in pair_payload["pairs"]}
                    | {pair["right"] for pair in pair_payload["pairs"]})
    labels: dict[str, dict] = {}
    indices: dict[str, dict[tuple[float, float], int]] = {}
    alignment: list[dict] = []
    for name in needed:
        record = frames[name]
        fixture = fixtures[record["fixture"]]
        if "reference" not in fixture:
            folder = ROOT / record["fixture"]
            litematic_path = next(folder.glob("*.litematic"))
            scene_path = folder / "scene.json"
            fixture["reference"] = {
                "litematic": relative_str(litematic_path),
                "litematic_sha256": connected.digest(litematic_path),
                "scene": relative_str(scene_path),
                "scene_sha256": connected.digest(scene_path),
                "blocks": load_litematic(litematic_path).blocks,
                "scene_transform": load_scene_transform(scene_path),
            }
        camera = fixture["cameras"][name]
        width, height = int(record["width"]), int(record["height"])
        dimensioned, index = render_result(
            fixture["reference"]["blocks"],
            fixture["reference"]["scene_transform"],
            camera,
            width,
            height,
            colours,
            shapes=shapes,
            scale=RENDER_SCALE,
            occluding_slots=occluding_slots,
            model_faces=model_faces,
        )
        points = np.asarray(
            arrays[f"feat_points_{record['key']}"], dtype=np.float64
        )
        usable = ~create_gui_mask(height, width)
        labels[name] = ds23.label_keypoints(
            points, dimensioned, index, camera, RENDER_SCALE, usable, width, height
        )
        indices[name] = {
            (round(float(point[0]), 4), round(float(point[1]), 4)): position
            for position, point in enumerate(points)
        }
        errors = labels[name]["alignment_px"]
        alignment.append(
            {
                "frame": name,
                "fixture": record["fixture"],
                "keypoints": int(len(points)),
                "hits": int(np.count_nonzero(labels[name]["hit"])),
                "alignment_median_px": float(np.median(errors)) if len(errors) else None,
            }
        )
        log(f"[truth ] rendered {name[:24]:24s} kp {len(points):5d} "
            f"hits {alignment[-1]['hits']:5d}")
    medians = [row["alignment_median_px"] for row in alignment
               if row["alignment_median_px"] is not None]
    alignment_median = float(np.median(medians)) if medians else None
    available = bool(medians) and alignment_median <= ALIGNMENT_MEDIAN_PX_MAX
    if not available:
        log(f"[truth ] reference labels unavailable: alignment median "
            f"{alignment_median} px > {ALIGNMENT_MEDIAN_PX_MAX}")

    records = track_payload["tracks"]
    view_of = {row["slot"]: row for row in records}
    truth_flags: dict[int, list[bool]] = {}
    truth_edges: dict[int, list[bool]] = {}
    for slot, row in view_of.items():
        flags: list[bool] = []
        edges: list[bool] = []
        names = sorted(row["views"])
        hits = {}
        for name in names:
            source = indices[name]
            points = row["pixels"][row["views"].index(name)]
            position = source.get((round(float(points[0]), 4), round(float(points[1]), 4)))
            hits[name] = position
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                left_name, right_name = names[first], names[second]
                left_position, right_position = hits[left_name], hits[right_name]
                scorable = (
                    left_position is not None
                    and right_position is not None
                    and bool(labels[left_name]["hit"][left_position])
                    and bool(labels[right_name]["hit"][right_position])
                )
                if not scorable:
                    continue
                left_world = labels[left_name]["world"][left_position]
                right_world = labels[right_name]["world"][right_position]
                distance = float(np.linalg.norm(left_world - right_world))
                normal_dot = float(
                    np.dot(
                        labels[left_name]["normal"][left_position],
                        labels[right_name]["normal"][right_position],
                    )
                )
                value = bool(
                    distance <= WORLD_HIT_DISTANCE_BLOCKS
                    and normal_dot >= FACE_NORMAL_COSINE_MIN
                )
                edges.append(value)
                flags.append(value)
        truth_edges[slot] = edges
        truth_flags[slot] = flags

    scorable_tracks = [slot for slot, flags in truth_flags.items() if flags]
    true_tracks = [slot for slot, flags in truth_flags.items() if flags and all(flags)]
    per_pair: dict[str, dict] = {}
    for pair in pair_payload["pairs"]:
        slot = int(pair["slot"])
        left_label = labels[pair["left"]]
        right_label = labels[pair["right"]]
        truth = ds23.oracle_edges(
            left_label, right_label, WORLD_HIT_DISTANCE_BLOCKS, FACE_NORMAL_COSINE_MIN
        )
        right_total = int(max(len(right_label["hit"]), 1))
        truth_codes = set(
            int(value) for value in truth[:, 0] * right_total + truth[:, 1]
        ) if len(truth) else set()
        pairs_of_tracks = [
            row for row in records
            if pair["left"] in row["views"] and pair["right"] in row["views"]
        ]
        scorable = 0
        true_edges = 0
        for row in pairs_of_tracks:
            left_position = indices[pair["left"]].get(
                (round(float(row["pixels"][row["views"].index(pair["left"])][0]), 4),
                 round(float(row["pixels"][row["views"].index(pair["left"])][1]), 4))
            )
            right_position = indices[pair["right"]].get(
                (round(float(row["pixels"][row["views"].index(pair["right"])][0]), 4),
                 round(float(row["pixels"][row["views"].index(pair["right"])][1]), 4))
            )
            if left_position is None or right_position is None:
                continue
            if not labels[pair["left"]]["hit"][left_position]:
                continue
            if not labels[pair["right"]]["hit"][right_position]:
                continue
            scorable += 1
            if int(left_position) * right_total + int(right_position) in truth_codes:
                true_edges += 1
        per_pair[str(slot)] = {
            "oracle_truth_edges": int(len(truth)),
            "left_keypoints": int(np.count_nonzero(left_label["hit"])),
            "right_keypoints": int(np.count_nonzero(right_label["hit"])),
            "tracks": int(len(pairs_of_tracks)),
            "selected_scorable": int(scorable),
            "selected_true": int(true_edges),
            "selected_truth_fraction": (
                float(true_edges / scorable) if available and scorable else None
            ),
        }
    log(
        f"[truth ] tracks: {len(true_tracks)}/{len(scorable_tracks)} scorable tracks are "
        f"geometrically true of {len(records)} resolved"
    )
    return {
        "available": available,
        "alignment_median_px": alignment_median,
        "alignment": alignment,
        "rule": {
            "world_hit_distance_blocks": WORLD_HIT_DISTANCE_BLOCKS,
            "face_normal_cosine_min": FACE_NORMAL_COSINE_MIN,
            "render_scale": RENDER_SCALE,
            "note": "DS-16 geometric-true rule, applied post-freeze only",
        },
        "track_truth": {
            "available": available,
            "tracks": len(records),
            "scorable_tracks": len(scorable_tracks),
            "true_tracks": len(true_tracks),
            "true_fraction_of_scorable": (
                float(len(true_tracks) / len(scorable_tracks)) if scorable_tracks else None
            ),
            "true_fraction_of_all": (
                float(len(true_tracks) / len(records)) if records else None
            ),
        },
        "per_pair": per_pair,
    }


# --------------------------------------------------------------------------- #
# Reproduction of the DS-24 A/B baselines (post-freeze, same evaluator).
# --------------------------------------------------------------------------- #
def ds14_class_correct_slots(frozen: dict, arrays, references: dict) -> list[int]:
    """Slots whose frozen DS-14 reciprocal set is solved and class-correct."""
    frames = {record["frame"]: record for record in frozen["frames"]}
    slots: list[int] = []
    for pair in [entry for entry in frozen["pairs"] if entry["index"] < PAIR_COUNT]:
        slot = int(pair["index"])
        left = frame_features(frames[pair["left"]], arrays)
        right = frame_features(frames[pair["right"]], arrays)
        left_camera = image_estimate(pair["left_estimate"])
        right_camera = image_estimate(pair["right_estimate"])
        blind = blind_policy.to_blind_input(
            left_intrinsics=left_camera.intrinsics,
            left_pose=left_camera.pose,
            right_intrinsics=right_camera.intrinsics,
            right_pose=right_camera.pose,
            matches=np.asarray(arrays[f"sel_{slot}"], dtype=np.float64).reshape(-1, 4),
            image_size=(
                int(pair["left_estimate"]["width"]),
                int(pair["left_estimate"]["height"]),
            ),
        )
        decision = blind_policy.select_blind_pose(blind)
        evaluation = _score_blind_decision(
            decision, blind,
            {
                "left_model": references[slot]["left_model"],
                "right_model": references[slot]["right_model"],
                "reference_class": references[slot]["reference_class"],
            },
        )
        if decision.solved and evaluation["class_equals_expected"]:
            slots.append(slot)
    return slots


def ds24_baselines(pair_payload: dict, frozen: dict, arrays, references: dict, log) -> dict:
    """Recompute DS-24's A (reciprocal) and B (directed) baselines from frozen data."""
    frames = {record["frame"]: record for record in frozen["frames"]}
    rows = {"A": [], "B": []}
    for pair in pair_payload["pairs"]:
        slot = int(pair["slot"])
        left = frame_features(frames[pair["left"]], arrays)
        right = frame_features(frames[pair["right"]], arrays)
        left_camera = image_estimate(pair["left_estimate"])
        right_camera = image_estimate(pair["right_estimate"])
        joint = joint_match_feature_frames(
            left, right, left_camera, right_camera,
            matcher=JointMatchSettings(), gates=PairGates(), ratio_test=RATIO_TEST,
        )
        directed = directed_match_feature_frames(
            left, right, left_camera, right_camera,
            matcher=JointMatchSettings(), gates=PairGates(), ratio_test=RATIO_TEST,
        )
        reference = {
            "left_model": references[slot]["left_model"],
            "right_model": references[slot]["right_model"],
            "reference_class": references[slot]["reference_class"],
        }
        for label, result in (("A", joint), ("B", directed)):
            blind = blind_policy.to_blind_input(
                left_intrinsics=left_camera.intrinsics,
                left_pose=left_camera.pose,
                right_intrinsics=right_camera.intrinsics,
                right_pose=right_camera.pose,
                matches=np.asarray(result.matches, dtype=np.float64).reshape(-1, 4),
                image_size=(int(pair["width"]), int(pair["height"])),
            )
            decision = blind_policy.select_blind_pose(blind)
            evaluation = _score_blind_decision(decision, blind, reference)
            rows[label].append(
                {
                    "slot": slot,
                    "solved": bool(decision.solved),
                    "class_correct": bool(evaluation["class_equals_expected"]),
                    "usable": _usable(evaluation),
                    "selected": int(len(result.matches)),
                }
            )
    summary = {}
    for label in ("A", "B"):
        summary[label] = {
            "usable": int(sum(row["usable"] for row in rows[label])),
            "class_correct": int(sum(row["class_correct"] for row in rows[label])),
            "solved": int(sum(row["solved"] for row in rows[label])),
            "rows": rows[label],
        }
    log(
        f"[repro ] DS-24 baselines recomputed: A usable {summary['A']['usable']}/9 "
        f"class-correct {summary['A']['class_correct']}/9; B usable "
        f"{summary['B']['usable']}/9 class-correct {summary['B']['class_correct']}/9 "
        f"(DS-24 recorded A {DS24_A_USABLE_RECORDED} usable, B {DS24_B_USABLE_RECORDED})"
    )
    return summary


def _decision_view(decision):
    """``ds23._DecisionView`` accepts dicts; ``blind_policy`` returns objects."""
    if isinstance(decision, dict):
        return ds23._DecisionView(decision)
    return decision


def _score_blind_decision(decision, blind, reference: dict) -> dict:
    if not decision.solved:
        return {
            "solved": False,
            "reason": decision.reason,
            "right_yaw_class": None,
            "class_equals_expected": False,
            "within_5deg": False,
            "rotation_error_degrees": None,
            "baseline_error_degrees_gauge_aligned": None,
            "baseline_error_degrees_signless": None,
            "gauge_index": None,
            "inliers": 0,
            "inlier_fraction": 0.0,
            "median_rho_degrees": None,
            "parallax_degrees": None,
            "cheirality_fraction": None,
        }
    handle = ds23._DecisionPose(_decision_view(decision), blind)
    return attribution._evaluate_pose(
        handle, reference["left_model"], reference["right_model"],
        reference["reference_class"],
    )


def _score_decision_record(decision: dict, blind, reference: dict) -> dict:
    return _score_blind_decision(ds23._DecisionView(decision), blind, reference)


def _usable(evaluation: dict) -> bool:
    baseline = evaluation["baseline_error_degrees_gauge_aligned"]
    return bool(
        evaluation["class_equals_expected"]
        and evaluation["within_5deg"]
        and baseline is not None
        and baseline <= PROMOTION_GAUGE_BASELINE_DEGREES_MAX
    )


def evaluate_gates(rows: list[dict], ds14_slots: list[int], synthetic: dict, log) -> dict:
    usable = sum(row["usable"] for row in rows)
    class_correct = sum(row["evaluation"]["class_equals_expected"] for row in rows)
    gauge_ok = sum(
        row["usable"]
        and row["evaluation"]["baseline_error_degrees_gauge_aligned"]
        <= PROMOTION_GAUGE_BASELINE_DEGREES_MAX
        for row in rows
    )
    false_solves = sum(row["false_solve"] for row in rows)
    regressions = [
        row["pair"] for row in rows
        if row["slot"] in ds14_slots and not row["usable"]
    ]
    passed = bool(
        usable >= PROMOTION_USABLE_REQUIRED
        and class_correct >= PROMOTION_CLASS_CORRECT_REQUIRED
        and gauge_ok >= PROMOTION_USABLE_REQUIRED
        and len(regressions) <= PROMOTION_REGRESSIONS_MAX
        and false_solves <= PROMOTION_FALSE_SOLVES_MAX
        and synthetic.get("all_controls_ok", False)
    )
    gate = {
        "usable": int(usable),
        "usable_required": PROMOTION_USABLE_REQUIRED,
        "class_correct": int(class_correct),
        "class_correct_required": PROMOTION_CLASS_CORRECT_REQUIRED,
        "gauge_baseline_within_5deg": int(gauge_ok),
        "false_solves": int(false_solves),
        "false_solves_max": PROMOTION_FALSE_SOLVES_MAX,
        "regressions_among_ds14_class_correct_pairs": regressions,
        "synthetic_degeneracy_controls_ok": bool(synthetic.get("all_controls_ok", False)),
        "passed": passed,
    }
    log(
        f"[gate  ] tracks usable {usable}/9 class {class_correct}/9 false {false_solves} "
        f"regressions {len(regressions)} -> {'PASS' if passed else 'fail'}"
    )
    return {
        "preregistered": {
            "usable_required": PROMOTION_USABLE_REQUIRED,
            "class_correct_required": PROMOTION_CLASS_CORRECT_REQUIRED,
            "gauge_baseline_degrees_max": PROMOTION_GAUGE_BASELINE_DEGREES_MAX,
            "false_solves_max": PROMOTION_FALSE_SOLVES_MAX,
            "regressions_among_ds14_correct_max": PROMOTION_REGRESSIONS_MAX,
            "synthetic_degeneracy_controls_required": True,
        },
        "variant": VARIANT,
        "result": gate,
        "promoted_variants": [VARIANT] if passed else [],
        "status": (
            "variant_meets_preregistered_promotion"
            if passed
            else "no_variant_meets_preregistered_promotion_diagnostic_only"
        ),
    }


def reproduce_ds23_oracle(fixtures: dict, log) -> dict:
    """Reproduce DS-23 on its own frozen oracle tape (post-freeze reference read)."""
    verify_digests(DS23_TAPE_PATHS, "DS-23 oracle tape", log)
    ds23_tape = json.loads((DS23_DIR / "blind_tape.json").read_text(encoding="utf-8"))
    ds23_arrays = dict(np.load(DS23_DIR / "blind_tape.npz"))
    fixture_cameras = {fixture: record["cameras"] for fixture, record in fixtures.items()}
    baseline_rows: list[dict] = []
    blind_rows: list[dict] = []
    for pair in ds23_tape["pairs"]:
        blind, matches = ds23.tape_blind_input(pair, ds23_arrays)
        left_estimate, right_estimate = ds23.tape_cameras(pair)
        cameras = fixture_cameras[pair["fixture"]]
        left_model = cameras[pair["left"]]
        right_model = cameras[pair["right"]]

        def _record(name, intrinsics, pose):
            return connected._Estimate(
                frame=name,
                width=int(pair["width"]),
                height=int(pair["height"]),
                focal_pixels=float(intrinsics["focal_length_pixels"][0]),
                principal_point=tuple(
                    float(value) for value in intrinsics["principal_point_pixels"]
                ),
                yaw_degrees=float(pose["yaw_degrees"]),
                pitch_degrees=float(pose["pitch_degrees"]),
                roll_degrees=float(pose["roll_degrees"]),
            )

        reference_class, _floors = connected._reference_class(
            _record(pair["left"], pair["left_intrinsics"], pair["left_pose"]),
            _record(pair["right"], pair["right_intrinsics"], pair["right_pose"]),
            left_model,
            right_model,
        )
        reference = {
            "left_model": left_model,
            "right_model": right_model,
            "reference_class": int(reference_class),
        }
        pose = solve_pair(
            left_estimate, right_estimate, matches,
            left_id="left", right_id="right", gates=PairGates(),
        )
        baseline_rows.append(
            {
                "pair": f"{pair['left']}->{pair['right']}",
                "solved": bool(pose.solved),
                "right_yaw_class": pose.right_yaw_class,
                "class_equals_expected": bool(
                    pose.solved and pose.right_yaw_class == reference_class
                ),
            }
        )
        decision = blind_policy.select_blind_pose(blind)
        evaluation = _score_blind_decision(decision, blind, reference)
        blind_rows.append(
            {
                "pair": f"{pair['left']}->{pair['right']}",
                "solved": bool(decision.solved),
                "right_yaw_class": decision.right_yaw_class,
                "usable": _usable(evaluation),
            }
        )
    solved = sum(row["solved"] for row in baseline_rows)
    class_correct = sum(row["class_equals_expected"] for row in baseline_rows)
    usable = sum(row["usable"] for row in blind_rows)
    log(
        f"[repro ] DS-23 oracle tape: DS-11 baseline {solved}/9 solved, {class_correct}/9 "
        f"class-correct; blind selector {usable}/9 usable"
    )
    return {
        "ds11_oracle_baseline_solved": int(solved),
        "ds11_oracle_baseline_class_correct": int(class_correct),
        "blind_selector_usable": int(usable),
        "reproduced": bool(
            solved == DS23_ORACLE_BASELINE_SOLVED_REQUIRED
            and usable >= DS23_ORACLE_BLIND_USABLE_REQUIRED
        ),
    }


# --------------------------------------------------------------------------- #
# Runner.
# --------------------------------------------------------------------------- #
def evaluate_pairs(
    pair_payload: dict, pair_arrays: dict, decisions: list[dict], references: dict,
    truth: dict,
) -> list[dict]:
    by_slot = {int(row["slot"]): row for row in decisions}
    rows: list[dict] = []
    for pair in pair_payload["pairs"]:
        slot = int(pair["slot"])
        decision = by_slot[slot]
        matches = pair_arrays[f"pair_matches_{slot}"]
        blind = tape_blind_input(pair, matches)
        evaluation = _score_decision_record(decision, blind, references[slot])
        rows.append(
            {
                "slot": slot,
                "pair": f"{pair['left']}->{pair['right']}",
                "set": pair["set"],
                "fixture": pair["fixture"],
                "left": pair["left"],
                "right": pair["right"],
                "tracks_touching_pair": int(pair["tracks_touching_pair"]),
                "selected": int(len(matches)),
                "decision": decision,
                "evaluation": evaluation,
                "usable": _usable(evaluation),
                "false_solve": bool(decision["solved"] and not _usable(evaluation)),
                "truth": truth["per_pair"][str(slot)],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], references: dict, ds24: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            decision = row["decision"]
            evaluation = row["evaluation"]
            writer.writerow(
                {
                    "pair": row["pair"],
                    "set": row["set"],
                    "fixture": row["fixture"],
                    "left": row["left"],
                    "right": row["right"],
                    "left_keypoints": row["truth"]["left_keypoints"],
                    "right_keypoints": row["truth"]["right_keypoints"],
                    "tracks_touching_pair": row["tracks_touching_pair"],
                    "selected_correspondences": row["selected"],
                    "decision_solved": decision["solved"],
                    "decision_reason": decision["reason"],
                    "decision_class": decision["right_yaw_class"],
                    "decision_inliers": decision["inliers"],
                    "decision_inlier_fraction": decision["inlier_fraction"],
                    "decision_median_rho_degrees": decision["median_rho_degrees"],
                    "decision_parallax_degrees": decision["parallax_degrees"],
                    "decision_cheirality_fraction": decision["cheirality_fraction"],
                    "decision_singular_ratio": decision["singular_ratio"],
                    "decision_class_margin": decision["class_margin"],
                    "trace_reads": ";".join(decision["trace_reads"]),
                    "reference_class": references[row["slot"]]["reference_class"],
                    "reference_class_floors":
                        references[row["slot"]]["reference_class_floors"],
                    "evaluated_class_correct": evaluation["class_equals_expected"],
                    "evaluated_within_5deg": evaluation["within_5deg"],
                    "evaluated_rotation_error_degrees": evaluation["rotation_error_degrees"],
                    "evaluated_baseline_error_degrees_gauge_aligned":
                        evaluation["baseline_error_degrees_gauge_aligned"],
                    "evaluated_baseline_error_degrees_signless":
                        evaluation["baseline_error_degrees_signless"],
                    "evaluated_gauge_index": evaluation["gauge_index"],
                    "usable": row["usable"],
                    "false_solve": row["false_solve"],
                    "selected_scorable": row["truth"]["selected_scorable"],
                    "selected_true": row["truth"]["selected_true"],
                    "selected_truth_fraction": row["truth"]["selected_truth_fraction"],
                    "ds24_A_usable": ds24["A"]["usable"],
                    "ds24_A_class_correct": ds24["A"]["class_correct"],
                    "ds24_B_usable": ds24["B"]["usable"],
                    "ds24_B_class_correct": ds24["B"]["class_correct"],
                }
            )


def run(out_dir: Path, log, *, session_filter=None) -> dict:
    started_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Phase A: blind graphs, tracks and pair sets ---------- #
    log("[phaseA] frozen directed candidate graphs, closed 3-cycles and tracks")
    blind_verified = verify_digests(BLIND_INPUT_PATHS, "DS-16 blind", log)
    frozen = attribution.load_frozen(DS16_DIR)
    arrays = dict(np.load(ROOT / frozen["candidates_npz"]["path"]))
    if len([pair for pair in frozen["pairs"] if pair["index"] < PAIR_COUNT]) != PAIR_COUNT:
        raise SystemExit("blocker: the DS-16 frozen artifact does not carry the nine pairs")
    phase_a = write_phase_a(
        out_dir, frozen, arrays, session_filter=session_filter, log=log,
        checkpoint=log.checkpoint,
    )
    track_payload, track_arrays, pair_payload, pair_arrays = read_frozen_phase_a(
        out_dir, log
    )
    freeze = write_freeze(out_dir, phase_a, log)

    # ---------------- Phase B: frozen DS-23 selector ----------------------- #
    log("[phaseB] applying the frozen DS-23 selector to the frozen pair sets")
    decisions, read_union = select_pairs(pair_payload, pair_arrays)
    decisions_digests = write_decisions(out_dir, decisions, read_union, log)
    selection_finished = time.perf_counter()

    # ---------------- Phase C: reference evaluation ------------------------ #
    log("[phaseC] reference evaluation (post-freeze only)")
    evaluator_verified = verify_digests(EVALUATOR_PATHS, "immutable evaluator", log)
    fixtures, references = phase_c_reference(pair_payload, log)
    truth = reference_truth(pair_payload, track_payload, frozen, fixtures, log)
    rows = evaluate_pairs(pair_payload, pair_arrays, decisions, references, truth)
    ds14_slots = ds14_class_correct_slots(frozen, arrays, references)
    synthetic = ds23.synthetic_controls(
        log, lambda decision, blind, case: ds23._synthetic_evaluate(
            attribution, decision, blind, case
        )
    )
    gates = evaluate_gates(rows, ds14_slots, synthetic, log)
    ds24 = ds24_baselines(pair_payload, frozen, arrays, references, log)
    oracle = reproduce_ds23_oracle(fixtures, log)
    runtime = time.perf_counter() - started_all

    csv_path = out_dir / "pairs.csv"
    write_csv(csv_path, rows, references, ds24)

    payload = {
        "schema": "ds25-graph-tracks-1",
        "phase": "blind_graph_wide_cycle_tracks_under_frozen_ds23_selector",
        "status": "complete",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": connected.source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "directive": (
            "DS-25: freeze high-recall directed top-k candidate graphs for each complete "
            "session (T2 three views, T6 four), build graph-wide tracks by exact keypoint "
            "identities and closed 3-cycles, resolve competing repeated-texture tracks with "
            "a predeclared label-blind score, hash tracks and derived pair match sets "
            "before scoring, apply the frozen DS-23 selector post-track, and evaluate the "
            "same nine pairs post-freeze"
        ),
        "boundary": {
            "phases": [
                "A blind candidate graphs + tracks + derived pair match sets",
                "freeze (tracks and pair sets hashed)",
                "B blind selection (frozen tape only)",
                "C reference evaluation (manifests, renders, frozen evaluator)",
            ],
            "blind_inputs": blind_verified,
            "evaluator": evaluator_verified,
            "freeze": freeze,
            "trace_reads_union": read_union,
            "trace_unexpected_reads": sorted(
                name for name in read_union
                if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
            ),
        },
        "settings": {
            "variant": VARIANT,
            "neighbors": NEIGHBORS,
            "max_edges": MAX_EDGES,
            "ratio_test": RATIO_TEST,
            "epipolar_inlier_degrees": EPIPOLAR_DEGREES,
            "world_hit_distance_blocks": WORLD_HIT_DISTANCE_BLOCKS,
            "face_normal_cosine_min": FACE_NORMAL_COSINE_MIN,
            "render_scale": RENDER_SCALE,
        },
        "sessions": phase_a["sessions"],
        "tracks": {
            "arrays_sha256": phase_a["tracks"]["arrays_sha256"],
            "digest": phase_a["tracks"],
            "per_session": phase_a["sessions"],
            "count": len(phase_a["track_records"]),
        },
        "pair_match_sets": {
            "arrays_sha256": phase_a["pair_match_sets"]["arrays_sha256"],
            "digest": phase_a["pair_match_sets"],
            "per_pair": pair_payload["pairs"],
        },
        "decisions": decisions_digests,
        "run": {
            "runtime_seconds": runtime,
            "selection_seconds": selection_finished - started_all,
            "peak_rss_bytes": log.peak_rss_bytes,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
        },
        "reproduction": {
            "ds14_baseline": {
                "class_correct_slots": ds14_slots,
                "class_correct": len(ds14_slots),
                "expected_class_correct": DS14_BASELINE_CLASS_CORRECT_REQUIRED,
                "reproduced": len(ds14_slots) == DS14_BASELINE_CLASS_CORRECT_REQUIRED,
            },
            "ds23_oracle": oracle,
            "ds24_match_sets": {
                "A_reciprocal_usable": ds24["A"]["usable"],
                "A_reciprocal_class_correct": ds24["A"]["class_correct"],
                "B_directed_usable": ds24["B"]["usable"],
                "B_directed_class_correct": ds24["B"]["class_correct"],
                "recorded_A_usable": DS24_A_USABLE_RECORDED,
                "recorded_B_usable": DS24_B_USABLE_RECORDED,
                "rows": {"A": ds24["A"]["rows"], "B": ds24["B"]["rows"]},
            },
        },
        "synthetic_controls": synthetic,
        "threshold": gates,
        "truth": truth,
        "rows": rows,
        "reference_inputs": {
            fixture: {
                "manifest": record["path"],
                "manifest_sha256": record["sha256"],
                "litematic": record.get("reference", {}).get("litematic"),
                "litematic_sha256": record.get("reference", {}).get("litematic_sha256"),
                "scene": record.get("reference", {}).get("scene"),
                "scene_sha256": record.get("reference", {}).get("scene_sha256"),
            }
            for fixture, record in fixtures.items()
        },
        "artifacts": {},
    }
    for name in ("candidate_graphs.json", "tracks.json", "tracks.npz",
                 "pair_match_sets.json", "pair_match_sets.npz", "decisions.json",
                 "freeze.json", "pairs.csv"):
        artifact = out_dir / name
        payload["artifacts"][name] = {
            "bytes": artifact.stat().st_size,
            "sha256": connected.digest(artifact),
        }
    evaluation_path = out_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8"
    )
    payload["artifacts"]["evaluation.json"] = {
        "bytes": evaluation_path.stat().st_size,
        "sha256": connected.digest(evaluation_path),
    }
    payload["_path"] = relative_str(evaluation_path)
    payload["_digest"] = payload["artifacts"]["evaluation.json"]["sha256"]
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--sessions",
        default="",
        help="comma-separated session filter (debug/smoke only; default all)",
    )
    arguments = parser.parse_args(argv)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_filter = (
        [value.strip() for value in arguments.sessions.split(",") if value.strip()]
        if arguments.sessions
        else None
    )
    log = _Log(out_dir / "run.log")
    try:
        payload = run(out_dir, log, session_filter=session_filter)
    except MemoryBudgetExceeded as error:
        log(f"[abort ] {error}")
        log.close()
        return 4
    log(
        f"[done  ] status {payload['threshold']['status']}; usable "
        f"{payload['threshold']['result']['usable']}/9, class-correct "
        f"{payload['threshold']['result']['class_correct']}/9, false "
        f"{payload['threshold']['result']['false_solves']}; runtime "
        f"{payload['run']['runtime_seconds']:.1f}s; peak RSS "
        f"{(payload['run']['peak_rss_bytes'] or 0) / 2**20:.1f} MiB"
    )
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
