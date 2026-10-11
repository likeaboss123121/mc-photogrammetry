"""DS-24 - image-derived match sets under the frozen DS-23 blind pose selector.

Directive (user, 2026-09-12): DS-23 proved that the label-blind support tie-break
scores 9/9 on frozen *oracle* correspondences. DS-24 asks whether the same frozen
selector keeps that behaviour when the correspondences come only from screenshot
pixels plus the image-derived M3 cameras.

Three per-pair match sets, all blind (frozen pixels + M3 geometry only):

* ``A_reciprocal_ds14`` - the current reciprocal matcher's selected one-to-one
  support (``joint_matching.match_feature_frames`` on the frozen DS-16 features);
* ``B_directed_ds18`` - the DS-18 directed matcher's selected one-to-one support;
* ``C_bestof_ratio_epipolar`` - the predeclared production-safe candidate
  selection: directed top-k, per-left best-of, the frozen descriptor ratio test
  (0.75), deterministic right-collision resolution, and a geometric-consistency
  filter (the frozen DS-11 1.0 degree epipolar inlier residual under at least one
  of the four M3 rotation hypotheses, each class's baseline direction being
  DS-19's bounded 3x3 Gram/SVD null vector). No new threshold is introduced.

Hard phase boundary, in this order, inside one process:

* **Phase A (blind).** Builds the three match sets from the digest-verified
  DS-16 frozen features and image-derived M3 estimates and writes
  ``match_sets.json`` + ``match_sets.npz``. No manifest, litematic, scene
  transform, reference class, evaluator output or oracle attribution is opened.
* **Phase B (blind).** Re-reads the frozen tape from disk, runs
  ``scripts/blind_pose_policy.select_blind_pose`` for every pair and variant, and
  writes ``decisions.json``.
* **Freeze.** ``freeze.json`` records the SHA-256 of the match-set arrays and of
  the decision payload, so match sets and pose decisions are hashed and frozen
  *before* any truth read. Phase C refuses to start if either digest moved.
* **Phase C (post-freeze).** The only place reference data is opened: the fixture
  camera manifests (reference classes and the frozen evaluator), the reference
  renders used for the post-hoc truth fraction, and the DS-23 oracle tape used to
  reproduce the 6/9 DS-11 oracle baseline and the 9/9 blind selection. Nothing
  from phase C feeds back into phase A or B.

Preregistered promotion (per variant, fixed before the run): usable poses
>= 8/9, yaw class correct >= 8/9, gauge-aligned baseline <= 5 degrees, zero
regressions among DS-14's four class-correct pairs, <= 1 false solve, and the
existing DS-23 synthetic degeneracy controls. No threshold is tuned after truth;
a failed result stays a diagnostic.

Run once with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts\\diagnose_ds24_image_derived_pose.py

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
    directed_topk_graph,
)
from mc_photogrammetry.pipeline.directed_matching import (  # noqa: E402
    match_feature_frames as directed_match_feature_frames,
)
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings  # noqa: E402
from mc_photogrammetry.pipeline.joint_matching import (  # noqa: E402
    match_feature_frames as joint_match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    camera_rotation_matrix,
)

# Blind-script helpers only. Reference accessors are imported lazily inside
# phase C so a poisoning test can intercept them on their source modules.
import blind_pose_policy as blind_policy  # noqa: E402
import diagnose_blind_pose_policy as ds23  # noqa: E402
import diagnose_connected_pairs as connected  # noqa: E402
import diagnose_match_attribution as attribution  # noqa: E402

DS16_DIR = ROOT / "exports" / "diagnostics" / "ds16"
DS18_DIR = ROOT / "exports" / "diagnostics" / "ds18"
DS23_DIR = ROOT / "exports" / "diagnostics" / "ds23"
OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds24"

# --------------------------------------------------------------------------- #
# Preregistered constants. All frozen before the run; none is tuned afterwards.
# --------------------------------------------------------------------------- #
PAIR_COUNT = 9  # DS-19/DS-23's exact nine T2/T6 pairs (DS-16 indices 0..8)
VARIANTS = ("A_reciprocal_ds14", "B_directed_ds18", "C_bestof_ratio_epipolar")

VARIANT_NEIGHBORS = 5
VARIANT_MAX_EDGES = 8192
VARIANT_RATIO_TEST = 0.75  # frozen CorrespondenceSettings.ratio_test
VARIANT_INLIER_DEGREES = 1.0  # frozen DS-11 inlier_residual_degrees

RENDER_SCALE = attribution.REFERENCE_RENDER_SCALE  # 0.35, frozen DS-16/DS-19
WORLD_HIT_DISTANCE_BLOCKS = 0.5  # frozen DS-16 oracle tolerance
FACE_NORMAL_COSINE_MIN = float(np.cos(np.radians(30.0)))
ALIGNMENT_MEDIAN_PX_MAX = 10.0

PROMOTION_USABLE_REQUIRED = 8
PROMOTION_CLASS_CORRECT_REQUIRED = 8
PROMOTION_GAUGE_BASELINE_DEGREES_MAX = 5.0
PROMOTION_FALSE_SOLVES_MAX = 1
PROMOTION_REGRESSIONS_MAX = 0
SYNTHETIC_REPEATS = 24

DS14_BASELINE_SOLVED_REQUIRED = 5
DS14_BASELINE_CLASS_CORRECT_REQUIRED = 4
DS23_ORACLE_BASELINE_SOLVED_REQUIRED = 6
DS23_ORACLE_BLIND_USABLE_REQUIRED = 9

DECISION_TAU = 1e-9
RSS_LIMIT_BYTES = 1_500 * 1024 * 1024

# Digest-pinned blind inputs read before any reference access.
BLIND_INPUT_PATHS = {
    "exports/diagnostics/ds16/match_candidates.json": (
        "e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93"
    ),
    "exports/diagnostics/ds16/match_candidates.npz": (
        "3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4"
    ),
    "exports/diagnostics/ds18/directed_estimates.json": (
        "bd5a34214bf1992283bda817eba41a781d2939945c429bb8cc5e9c77b611cc53"
    ),
}
# The immutable evaluator and the frozen selector boundaries.
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
    "pair", "set", "variant", "candidate_graph_edges", "selected_correspondences",
    "decision_solved", "decision_reason", "decision_class", "decision_inliers",
    "decision_inlier_fraction", "decision_median_rho_degrees",
    "decision_parallax_degrees", "decision_cheirality_fraction",
    "decision_singular_ratio", "decision_class_margin", "trace_reads",
    "reference_class", "reference_class_floors",
    "evaluated_class_correct", "evaluated_within_5deg",
    "evaluated_rotation_error_degrees",
    "evaluated_baseline_error_degrees_gauge_aligned",
    "evaluated_baseline_error_degrees_signless", "evaluated_gauge_index",
    "usable", "false_solve",
    "candidate_scorable", "candidate_true", "candidate_truth_fraction",
    "selected_scorable", "selected_true", "selected_truth_fraction",
    "ds14_baseline_solved", "ds14_baseline_class", "ds14_baseline_class_correct",
    "ds14_baseline_usable",
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
        import sys as _sys
        from ctypes import wintypes

        if _sys.platform != "win32":
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
# Blind geometry helpers (pixels + M3 cameras only).
# --------------------------------------------------------------------------- #
def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (yaw_class % 4))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _unit_rays(pixels: np.ndarray, intrinsics) -> np.ndarray:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (pixels[:, 0] - centre_x) / focal_x
    y = -(pixels[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotation_from_estimate(estimate: CameraEstimate) -> np.ndarray:
    pose = estimate.pose
    return camera_rotation_matrix(
        CameraModel(
            position=(0.0, 0.0, 0.0),
            yaw_degrees=float(pose.yaw_degrees),
            pitch_degrees=float(pose.pitch_degrees),
            roll_degrees=float(pose.roll_degrees),
            eye_height=0.0,
        )
    )


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


def frame_features(record: dict, arrays) -> FrameFeatures:
    key = record["key"]
    return FrameFeatures(
        frame_id=record["frame"],
        points=np.asarray(arrays[f"feat_points_{key}"], dtype=np.float64),
        descriptors=np.asarray(arrays[f"feat_desc_{key}"], dtype=np.float32),
        width=int(record["width"]),
        height=int(record["height"]),
        usable_pixels=int(record["usable_pixels"]),
        sizes=np.asarray(arrays[f"feat_sizes_{key}"], dtype=np.float64),
        boundary_dropped=int(record["boundary_dropped"]),
    )


def variant_c_stages(
    left: FrameFeatures,
    right: FrameFeatures,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Predeclared production-safe candidate selection (descriptors + geometry).

    Stage 1 ``graph``: directed left->right top-k over the frozen descriptors.
    Stage 2 ``ratio``: one best edge per left keypoint, kept only when the frozen
    descriptor ratio test (0.75) accepts it.
    Stage 3 ``one_to_one``: deterministic right-collision resolution by
    ``(descriptor distance, left index)``.
    Stage 4 ``selected``: keep edges whose epipolar residual is <= the frozen
    DS-11 inlier residual (1.0 degree) under at least one of the four M3 rotation
    hypotheses, each hypothesis' baseline direction being the bounded 3x3
    Gram/SVD null vector of the current rows (DS-19's non-RANSAC mechanism).
    """
    graph = directed_topk_graph(
        left.descriptors,
        right.descriptors,
        neighbors=VARIANT_NEIGHBORS,
        max_edges=VARIANT_MAX_EDGES,
        distance_ceiling=None,
    )
    edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    distances = np.asarray(graph.distances, dtype=np.float64).reshape(-1)
    empty = np.empty((0, 2), dtype=np.int64)
    stats = {
        "candidate_graph_edges": int(len(edges)),
        "total_graph_edges": int(graph.total_edges),
        "truncated": int(graph.truncated),
        "ratio_edges": 0,
        "one_to_one_edges": 0,
        "selected_correspondences": 0,
        "per_class_consensus": {},
    }
    if len(edges) == 0:
        return empty, {"graph": empty, "ratio": empty, "one_to_one": empty}, stats

    best: dict[int, tuple[float, int]] = {}
    second: dict[int, float] = {}
    for row in range(len(edges)):
        left_index = int(edges[row, 0])
        value = float(distances[row])
        if left_index not in best:
            best[left_index] = (value, int(edges[row, 1]))
        elif left_index not in second:
            second[left_index] = value
    keep_rows = [
        row
        for row in range(len(edges))
        if int(edges[row, 0]) not in second
        or best[int(edges[row, 0])][0] <= VARIANT_RATIO_TEST * second[int(edges[row, 0])]
    ]
    ratio_edges = edges[keep_rows] if keep_rows else empty
    stats["ratio_edges"] = int(len(ratio_edges))

    if len(ratio_edges):
        rows = np.asarray(keep_rows, dtype=np.int64)
        order = np.lexsort((edges[rows, 0], distances[rows], edges[rows, 1]))
        rows = rows[order]
        _unique, first = np.unique(edges[rows, 1], return_index=True)
        rows = rows[np.sort(first)]
        one_to_one_edges = edges[rows]
    else:
        one_to_one_edges = empty
    stats["one_to_one_edges"] = int(len(one_to_one_edges))

    selected = empty
    if len(one_to_one_edges):
        left_rays = _unit_rays(left.points[one_to_one_edges[:, 0]], left_camera.intrinsics)
        right_rays = _unit_rays(right.points[one_to_one_edges[:, 1]], right_camera.intrinsics)
        left_rotation = _rotation_from_estimate(left_camera)
        right_rotation = _rotation_from_estimate(right_camera)
        worst = np.full(len(one_to_one_edges), np.inf, dtype=np.float64)
        per_class: dict[str, int] = {}
        for yaw_class in range(4):
            relative = (right_rotation @ _ry90(yaw_class)) @ left_rotation.T
            rows = np.cross(left_rays @ relative.T, right_rays)
            norms = np.linalg.norm(rows, axis=1)
            informative = norms > 1e-12
            if not np.any(informative):
                per_class[str(yaw_class)] = 0
                continue
            block = rows[informative]
            eigenvalues, vectors = np.linalg.eigh(block.T @ block)
            direction = vectors[:, 0]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                per_class[str(yaw_class)] = 0
                continue
            direction = direction / norm
            projection = np.abs(rows @ direction) / np.maximum(norms, 1e-12)
            residual = np.degrees(np.arcsin(np.clip(projection, 0.0, 1.0)))
            per_class[str(yaw_class)] = int(
                np.count_nonzero(residual <= VARIANT_INLIER_DEGREES)
            )
            worst = np.minimum(worst, residual)
        selected = one_to_one_edges[worst <= VARIANT_INLIER_DEGREES]
        stats["per_class_consensus"] = per_class
    stats["selected_correspondences"] = int(len(selected))
    return selected, {"graph": edges, "ratio": ratio_edges, "one_to_one": one_to_one_edges}, stats


def _matches_from_edges(
    left_points: np.ndarray, right_points: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.column_stack(
        (
            left_points[edges[:, 0], 0],
            left_points[edges[:, 0], 1],
            right_points[edges[:, 1], 0],
            right_points[edges[:, 1], 1],
        )
    )


def _pixel_index(points: np.ndarray) -> dict[tuple[float, float], int]:
    index: dict[tuple[float, float], int] = {}
    for position, point in enumerate(points):
        key = (float(point[0]), float(point[1]))
        index.setdefault(key, position)
    return index


def _edges_from_matches(
    matches: np.ndarray, left_points: np.ndarray, right_points: np.ndarray
) -> tuple[np.ndarray, int]:
    """Recover keypoint-index edges for a match array (same pixel -> same world)."""
    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.int64), 0
    left_index = _pixel_index(left_points)
    right_index = _pixel_index(right_points)
    rows: list[tuple[int, int]] = []
    missing = 0
    for match in matches:
        left_key = (float(match[0]), float(match[1]))
        right_key = (float(match[2]), float(match[3]))
        position = left_index.get(left_key)
        other = right_index.get(right_key)
        if position is None or other is None:
            missing += 1
            continue
        rows.append((position, other))
    if not rows:
        return np.empty((0, 2), dtype=np.int64), missing
    return np.asarray(rows, dtype=np.int64), missing


# --------------------------------------------------------------------------- #
# Phase A: blind match-set construction.
# --------------------------------------------------------------------------- #
def build_match_sets(frozen: dict, arrays, pair_indices=None) -> tuple[dict, dict]:
    """Build the three blind match sets; reads pixels/features and M3 geometry only."""
    ds18 = json.loads((DS18_DIR / "directed_estimates.json").read_text(encoding="utf-8"))
    ds18_by_key = {(row["left"], row["right"]): row for row in ds18["pairs"]}
    frames = {record["frame"]: record for record in frozen["frames"]}
    pairs = [pair for pair in frozen["pairs"] if pair["index"] < PAIR_COUNT]
    if pair_indices is not None:
        wanted = {int(value) for value in pair_indices}
        pairs = [pair for pair in pairs if pair["index"] in wanted]

    tape_arrays: dict[str, np.ndarray] = {}
    tape_pairs: list[dict] = []
    for slot, pair in enumerate(pairs):
        left_record = frames[pair["left"]]
        right_record = frames[pair["right"]]
        left = frame_features(left_record, arrays)
        right = frame_features(right_record, arrays)
        left_camera = image_estimate(pair["left_estimate"])
        right_camera = image_estimate(pair["right_estimate"])

        joint = joint_match_feature_frames(
            left, right, left_camera, right_camera,
            matcher=JointMatchSettings(), gates=PairGates(),
            ratio_test=VARIANT_RATIO_TEST,
        )
        directed = directed_match_feature_frames(
            left, right, left_camera, right_camera,
            matcher=JointMatchSettings(), gates=PairGates(),
            ratio_test=VARIANT_RATIO_TEST,
        )
        c_selected, c_stages, c_stats = variant_c_stages(
            left, right, left_camera, right_camera
        )

        a_matches = np.asarray(joint.matches, dtype=np.float64).reshape(-1, 4)
        b_matches = np.asarray(directed.matches, dtype=np.float64).reshape(-1, 4)
        c_matches = _matches_from_edges(left.points, right.points, c_selected)
        a_edges, a_missing = _edges_from_matches(a_matches, left.points, right.points)
        b_edges, b_missing = _edges_from_matches(b_matches, left.points, right.points)

        # Reproduction controls against the frozen blind artifacts.
        frozen_sel = np.asarray(arrays[f"sel_{pair['index']}"], dtype=np.float64).reshape(-1, 4)
        a_reproduces = a_matches.shape == frozen_sel.shape and np.array_equal(
            a_matches, frozen_sel
        )
        frozen_directed_row = ds18_by_key.get((pair["left"], pair["right"]))
        if frozen_directed_row is None:
            b_reproduces = False
        else:
            frozen_directed = np.asarray(
                frozen_directed_row["directed"]["matches"], dtype=np.float64
            ).reshape(-1, 4)
            b_reproduces = b_matches.shape == frozen_directed.shape and np.array_equal(
                b_matches, frozen_directed
            )
        if not a_reproduces:
            raise SystemExit(
                f"blocker: pair {left.frame_id}->{right.frame_id} does not reproduce the "
                "frozen DS-16/DS-14 selected set; refusing to freeze a new match set"
            )
        if not b_reproduces:
            raise SystemExit(
                f"blocker: pair {left.frame_id}->{right.frame_id} does not reproduce the "
                "frozen DS-18 directed selected set; refusing to freeze a new match set"
            )

        tape_arrays[f"A_matches_{slot}"] = a_matches
        tape_arrays[f"A_edges_{slot}"] = a_edges
        tape_arrays[f"A_graph_{slot}"] = np.asarray(
            arrays[f"cand_{pair['index']}"], dtype=np.float64
        ).reshape(-1, 3)[:, :2].astype(np.int64)
        tape_arrays[f"B_matches_{slot}"] = b_matches
        tape_arrays[f"B_edges_{slot}"] = b_edges
        tape_arrays[f"B_graph_{slot}"] = c_stages["graph"]
        tape_arrays[f"C_matches_{slot}"] = c_matches
        tape_arrays[f"C_edges_{slot}"] = c_selected
        tape_arrays[f"C_ratio_{slot}"] = c_stages["ratio"]
        tape_arrays[f"C_one_to_one_{slot}"] = c_stages["one_to_one"]
        tape_arrays[f"C_graph_{slot}"] = c_stages["graph"]

        tape_pairs.append(
            {
                "slot": int(slot),
                "ds16_index": int(pair["index"]),
                "set": pair["set"],
                "fixture": pair["fixture"],
                "left": pair["left"],
                "right": pair["right"],
                "width": int(left_record["width"]),
                "height": int(left_record["height"]),
                "left_estimate": pair["left_estimate"],
                "right_estimate": pair["right_estimate"],
                "left_intrinsics": {
                    "image_size": list(left_camera.intrinsics.image_size),
                    "focal_length_pixels": list(left_camera.intrinsics.focal_length_pixels),
                    "principal_point_pixels": list(
                        left_camera.intrinsics.principal_point_pixels
                    ),
                },
                "right_intrinsics": {
                    "image_size": list(right_camera.intrinsics.image_size),
                    "focal_length_pixels": list(right_camera.intrinsics.focal_length_pixels),
                    "principal_point_pixels": list(
                        right_camera.intrinsics.principal_point_pixels
                    ),
                },
                "left_pose": {
                    "position": None,
                    "yaw_degrees": float(left_camera.pose.yaw_degrees),
                    "pitch_degrees": float(left_camera.pose.pitch_degrees),
                    "roll_degrees": float(left_camera.pose.roll_degrees),
                },
                "right_pose": {
                    "position": None,
                    "yaw_degrees": float(right_camera.pose.yaw_degrees),
                    "pitch_degrees": float(right_camera.pose.pitch_degrees),
                    "roll_degrees": float(right_camera.pose.roll_degrees),
                },
                "variants": {
                    "A_reciprocal_ds14": {
                        "candidate_graph_edges": int(len(tape_arrays[f"A_graph_{slot}"])),
                        "selected_correspondences": int(len(a_matches)),
                        "matches_sha256": digest_array(a_matches),
                        "edges_sha256": digest_array(a_edges),
                        "matcher_status": joint.status,
                        "matcher_reason": joint.reason,
                        "matcher_solved": bool(joint.solved),
                        "matcher_class": (
                            None if joint.right_yaw_class is None
                            else int(joint.right_yaw_class)
                        ),
                        "reproduces_frozen_ds16_selected_set": bool(a_reproduces),
                        "index_lookup_missing": int(a_missing),
                    },
                    "B_directed_ds18": {
                        "candidate_graph_edges": int(len(tape_arrays[f"B_graph_{slot}"])),
                        "selected_correspondences": int(len(b_matches)),
                        "matches_sha256": digest_array(b_matches),
                        "edges_sha256": digest_array(b_edges),
                        "matcher_status": directed.status,
                        "matcher_reason": directed.reason,
                        "matcher_solved": bool(directed.solved),
                        "matcher_class": (
                            None if directed.right_yaw_class is None
                            else int(directed.right_yaw_class)
                        ),
                        "reproduces_frozen_ds18_selected_set": bool(b_reproduces),
                        "index_lookup_missing": int(b_missing),
                    },
                    "C_bestof_ratio_epipolar": {
                        "candidate_graph_edges": int(c_stats["candidate_graph_edges"]),
                        "ratio_edges": int(c_stats["ratio_edges"]),
                        "one_to_one_edges": int(c_stats["one_to_one_edges"]),
                        "selected_correspondences": int(len(c_matches)),
                        "matches_sha256": digest_array(c_matches),
                        "edges_sha256": digest_array(c_selected),
                        "per_class_consensus": {
                            key: int(value)
                            for key, value in c_stats["per_class_consensus"].items()
                        },
                        "index_lookup_missing": 0,
                    },
                },
            }
        )
    tape = {
        "schema": "ds24-blind-match-sets-1",
        "phase": "pre_selection_blind_match_sets",
        "camera_regime": "image_derived_m3",
        "pair_universe": "DS-16 indices 0..8 (T2 3 + T6 6), the DS-19/DS-23 nine pairs",
        "variants": list(VARIANTS),
        "variant_rule": {
            "A_reciprocal_ds14": (
                "joint_matching.match_feature_frames selected one-to-one support on the "
                "frozen DS-16 features"
            ),
            "B_directed_ds18": (
                "directed_matching.match_feature_frames selected one-to-one support on the "
                "frozen DS-16 features"
            ),
            "C_bestof_ratio_epipolar": (
                "directed top-k=5 -> per-left best-of -> frozen ratio test 0.75 -> "
                "deterministic right-collision resolution -> frozen DS-11 1.0 degree "
                "epipolar inlier residual under at least one of four M3 rotation hypotheses "
                "(DS-19 bounded 3x3 Gram/SVD null vector); no new threshold"
            ),
        },
        "forbidden_in_tape": [
            "fixture manifests / reference camera poses",
            "litematics, scene transforms and rendered block labels",
            "reference yaw classes, oracle edge sets and per-pair truth counts",
            "frozen evaluator outputs or any reference score",
        ],
        "pairs": tape_pairs,
    }
    return tape, tape_arrays


def write_match_sets(out_dir: Path, tape: dict, tape_arrays: dict, log) -> dict:
    arrays_path = out_dir / "match_sets.npz"
    np.savez_compressed(arrays_path, **tape_arrays)
    arrays_sha256 = {key: digest_array(value) for key, value in sorted(tape_arrays.items())}
    tape = dict(tape)
    tape["arrays"] = {
        "path": str(arrays_path.relative_to(ROOT)),
        "file_sha256": connected.digest(arrays_path),
        "sha256_by_array": arrays_sha256,
        "arrays_sha256": canonical_digest(arrays_sha256),
    }
    json_path = out_dir / "match_sets.json"
    json_path.write_text(json.dumps(tape, indent=2, default=connected.plain), encoding="utf-8")
    log(
        f"[tape  ] wrote {json_path.name} ({json_path.stat().st_size} B) and "
        f"{arrays_path.name} ({arrays_path.stat().st_size} B); arrays digest "
        f"{tape['arrays']['arrays_sha256'][:16]}..."
    )
    return tape


def read_match_sets(out_dir: Path, log) -> tuple[dict, dict]:
    json_path = out_dir / "match_sets.json"
    arrays_path = out_dir / "match_sets.npz"
    tape = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(arrays_path))
    recorded = tape["arrays"]
    if connected.digest(arrays_path) != recorded["file_sha256"]:
        raise SystemExit("blocker: blind match-set archive changed after it was written")
    actual = {key: digest_array(value) for key, value in sorted(arrays.items())}
    if actual != recorded["sha256_by_array"]:
        raise SystemExit(
            "blocker: blind match-set arrays changed after they were frozen; refusing to select"
        )
    if canonical_digest(actual) != recorded["arrays_sha256"]:
        raise SystemExit("blocker: blind match-set array digest does not reproduce")
    log(
        f"[tape  ] re-read frozen {json_path.name} + {arrays_path.name}; array digest "
        f"{recorded['arrays_sha256'][:16]}... verified"
    )
    return tape, arrays


# --------------------------------------------------------------------------- #
# Phase B: the frozen blind selector, on the frozen tape only.
# --------------------------------------------------------------------------- #
def tape_blind_input(pair: dict, matches: np.ndarray):
    """Build the selector input from the *written* tape fields only."""
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


def select_match_sets(tape: dict, arrays: dict) -> tuple[list[dict], list[str]]:
    """Run the frozen DS-23 selector on every frozen match set."""
    decisions: list[dict] = []
    read_union: set[str] = set()
    for pair in tape["pairs"]:
        slot = int(pair["slot"])
        for variant in VARIANTS:
            matches = arrays[f"{variant[0]}_matches_{slot}"]
            blind = tape_blind_input(pair, matches)
            trace = blind_policy.start_trace(
                blind, blind.left_intrinsics, blind.right_intrinsics
            )
            decision = blind_policy.select_blind_pose(blind)
            trace = blind_policy.stop_trace()
            reads = tuple(sorted(set(trace.names())))
            read_union.update(reads)
            unexpected = [
                name for name in reads if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
            ]
            decisions.append(
                {
                    "slot": slot,
                    "variant": variant,
                    "solved": bool(decision.solved),
                    "reason": decision.reason,
                    "right_yaw_class": (
                        None if decision.right_yaw_class is None
                        else int(decision.right_yaw_class)
                    ),
                    "baseline_direction": (
                        None
                        if decision.baseline_direction is None
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
                    "trace_reads": list(reads),
                    "unexpected_reads": unexpected,
                    "matches_sha256": pair["variants"][variant]["matches_sha256"],
                    "per_class": {
                        key: {
                            "gate_pass": bool(value["gate_pass"]),
                            "gate_refusal": value["gate_refusal"],
                            "consensus_count": int(value["consensus_count"]),
                            "consensus_fraction": float(value["consensus_fraction"]),
                            "median_rho_degrees": value["median_rho_degrees"],
                            "parallax_degrees": value["parallax_degrees"],
                            "cheirality_fraction": value["cheirality_fraction"],
                            "singular_ratio": value["singular_ratio"],
                            "stability": value["stability"],
                        }
                        for key, value in decision.per_class.items()
                    },
                }
            )
    return decisions, sorted(read_union)


def write_decisions(out_dir: Path, decisions: list[dict], read_union: list[str], log) -> dict:
    payload = {
        "schema": "ds24-blind-decisions-1",
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
            name for name in read_union if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
        ),
        "decisions": decisions,
    }
    path = out_dir / "decisions.json"
    path.write_text(json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8")
    digests = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": connected.digest(path),
        "decisions_sha256": canonical_digest(decisions),
    }
    log(
        f"[blind ] froze {len(decisions)} decisions across {len(VARIANTS)} variants; "
        f"decisions digest {digests['decisions_sha256'][:16]}..."
    )
    return digests


def write_freeze(
    out_dir: Path, tape: dict, decisions_digests: dict, decisions: list[dict], log
) -> dict:
    payload = {
        "schema": "ds24-freeze-1",
        "phase": "match_sets_and_pose_decisions_frozen_before_any_truth_read",
        "match_sets": {
            "json": str((out_dir / "match_sets.json").relative_to(ROOT)),
            "json_sha256": connected.digest(out_dir / "match_sets.json"),
            "npz": str((out_dir / "match_sets.npz").relative_to(ROOT)),
            "npz_sha256": connected.digest(out_dir / "match_sets.npz"),
            "arrays_sha256": tape["arrays"]["arrays_sha256"],
            "per_array_sha256": tape["arrays"]["sha256_by_array"],
        },
        "decisions": decisions_digests,
        "pair_decisions_sha256": {
            f"{pair['slot']}|{variant}": canonical_digest(
                [
                    row
                    for row in decisions
                    if row["slot"] == pair["slot"] and row["variant"] == variant
                ]
            )
            for pair in tape["pairs"]
            for variant in VARIANTS
        },
        "reference_fields_absent": [
            "reference_class", "reference_class_floors", "oracle", "truth",
            "evaluator", "usable", "class_equals_expected", "within_5deg",
            "baseline_error_degrees_gauge_aligned", "baseline_error_degrees_signless",
        ],
        "note": (
            "Phase C refuses to start unless both digests above reproduce; the decision "
            "path never reads a fixture manifest, litematic, scene transform, reference "
            "class, oracle edge or evaluator output."
        ),
    }
    path = out_dir / "freeze.json"
    path.write_text(json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8")
    digest = connected.digest(path)
    payload["file_sha256"] = digest
    log(
        f"[freeze] match sets + {len(tape['pairs']) * len(VARIANTS)} decisions hashed and "
        f"frozen: freeze.json {digest[:16]}...; reference evaluation starts now"
    )
    return payload


# --------------------------------------------------------------------------- #
# Phase C: reference-only evaluation (manifests, renders, frozen evaluator).
# --------------------------------------------------------------------------- #
def reference_models(tape: dict, log) -> dict:
    import mc_photogrammetry.reconstruction.camera as camera_module

    fixtures: dict[str, dict] = {}
    for pair in tape["pairs"]:
        fixture = pair["fixture"]
        if fixture in fixtures:
            continue
        manifest_path = ROOT / fixture / "cameras.json"
        if not manifest_path.exists():
            raise SystemExit(f"blocker: reference manifest missing: {manifest_path}")
        fixtures[fixture] = {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": connected.digest(manifest_path),
            "cameras": camera_module.load_camera_manifest(manifest_path),
        }
    log(f"[ref   ] opened {len(fixtures)} reference camera manifest(s)")
    return fixtures


def reference_classes(tape: dict, fixtures: dict) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for pair in tape["pairs"]:
        cameras = fixtures[pair["fixture"]]["cameras"]
        left_model = cameras[pair["left"]]
        right_model = cameras[pair["right"]]
        left_estimate = connected._estimate_from_record(pair["left_estimate"])
        right_estimate = connected._estimate_from_record(pair["right_estimate"])
        direct, floors = connected._reference_class(
            left_estimate, right_estimate, left_model, right_model
        )
        rows[int(pair["slot"])] = {
            "reference_class": int(direct),
            "reference_class_floors": int(floors),
            "evaluator_ambiguity": bool(direct != floors),
            "left_model": left_model,
            "right_model": right_model,
            "left_estimate": image_estimate(pair["left_estimate"]),
            "right_estimate": image_estimate(pair["right_estimate"]),
        }
    return rows


def _score_pose(pose, reference: dict) -> dict:
    return attribution._evaluate_pose(
        pose, reference["left_model"], reference["right_model"],
        reference["reference_class"],
    )


def _score_decision(decision: dict, blind, reference: dict) -> dict:
    if not decision["solved"]:
        return {
            "solved": False,
            "reason": decision["reason"],
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
    handle = ds23._DecisionPose(ds23._DecisionView(decision), blind)
    return attribution._evaluate_pose(
        handle, reference["left_model"], reference["right_model"],
        reference["reference_class"],
    )


def _usable(evaluation: dict) -> bool:
    baseline = evaluation["baseline_error_degrees_gauge_aligned"]
    return bool(
        evaluation["class_equals_expected"]
        and evaluation["within_5deg"]
        and baseline is not None
        and baseline <= PROMOTION_GAUGE_BASELINE_DEGREES_MAX
    )


def reference_truth_fractions(
    tape: dict, tape_arrays: dict, feature_arrays: dict, fixtures: dict,
    frozen: dict, log,
) -> dict:
    """Post-freeze reference rendering: per-edge geometric truth, DS-16 rule."""
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
    needed = sorted({pair["left"] for pair in tape["pairs"]}
                    | {pair["right"] for pair in tape["pairs"]})
    labels: dict[str, dict] = {}
    alignment: list[dict] = []
    for name in needed:
        record = frames[name]
        fixture = fixtures[record["fixture"]]
        if "reference" not in fixture:
            folder = ROOT / record["fixture"]
            litematic_path = next(folder.glob("*.litematic"))
            scene_path = folder / "scene.json"
            fixture["reference"] = {
                "litematic": str(litematic_path.relative_to(ROOT)),
                "litematic_sha256": connected.digest(litematic_path),
                "scene": str(scene_path.relative_to(ROOT)),
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
        points = np.asarray(feature_arrays[f"feat_points_{record['key']}"], dtype=np.float64)
        usable = ~create_gui_mask(height, width)
        labels[name] = ds23.label_keypoints(
            points, dimensioned, index, camera, RENDER_SCALE, usable, width, height
        )
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
        log(f"[truth ] reference labels unavailable: alignment median {alignment_median} px "
            f"> {ALIGNMENT_MEDIAN_PX_MAX}")

    fractions: dict[str, dict] = {}
    for pair in tape["pairs"]:
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

        def _measure(edges: np.ndarray, left=left_label, right=right_label,
                     codes_set=truth_codes, total=right_total) -> dict:
            if not available or len(edges) == 0:
                return {"scorable": 0, "true": 0, "fraction": None}
            left_index = np.asarray(edges[:, 0], dtype=np.int64)
            right_index = np.asarray(edges[:, 1], dtype=np.int64)
            scorable = left["hit"][left_index] & right["hit"][right_index]
            codes = left_index * total + right_index
            true = np.asarray(
                [bool(scorable[row]) and int(codes[row]) in codes_set
                 for row in range(len(edges))],
                dtype=bool,
            )
            scorable_count = int(np.count_nonzero(scorable))
            return {
                "scorable": scorable_count,
                "true": int(np.count_nonzero(true)),
                "fraction": (
                    float(np.count_nonzero(true) / scorable_count)
                    if scorable_count else None
                ),
            }

        fractions[f"{slot}"] = {
            "A_candidate": _measure(tape_arrays[f"A_graph_{slot}"]),
            "A_selected": _measure(tape_arrays[f"A_edges_{slot}"]),
            "B_candidate": _measure(tape_arrays[f"B_graph_{slot}"]),
            "B_selected": _measure(tape_arrays[f"B_edges_{slot}"]),
            "C_candidate": _measure(tape_arrays[f"C_graph_{slot}"]),
            "C_ratio": _measure(tape_arrays[f"C_ratio_{slot}"]),
            "C_one_to_one": _measure(tape_arrays[f"C_one_to_one_{slot}"]),
            "C_selected": _measure(tape_arrays[f"C_edges_{slot}"]),
            "oracle_truth_edges": int(len(truth)),
        }
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
        "per_pair": fractions,
    }


def reproduce_ds23_oracle(tape: dict, fixtures: dict, log) -> dict:
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
        evaluation = _score_decision(
            {
                "solved": bool(decision.solved),
                "reason": decision.reason,
                "right_yaw_class": decision.right_yaw_class,
                "baseline_direction": (
                    None if decision.baseline_direction is None
                    else [float(value) for value in decision.baseline_direction]
                ),
                "inliers": int(decision.inliers),
                "inlier_fraction": float(decision.inlier_fraction),
                "median_rho_degrees": decision.median_rho_degrees,
                "parallax_degrees": decision.parallax_degrees,
                "cheirality_fraction": decision.cheirality_fraction,
            },
            blind,
            reference,
        )
        blind_rows.append(
            {
                "pair": f"{pair['left']}->{pair['right']}",
                "solved": bool(decision.solved),
                "right_yaw_class": decision.right_yaw_class,
                "usable": _usable(evaluation),
                "class_equals_expected": evaluation["class_equals_expected"],
                "baseline_error_degrees_gauge_aligned":
                    evaluation["baseline_error_degrees_gauge_aligned"],
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
        "baseline_rows": baseline_rows,
        "blind_rows": blind_rows,
        "reproduced": bool(
            solved == DS23_ORACLE_BASELINE_SOLVED_REQUIRED
            and usable >= DS23_ORACLE_BLIND_USABLE_REQUIRED
        ),
    }


def evaluate_variants(tape: dict, arrays: dict, decisions: list[dict], references: dict, log) -> dict:
    by_key = {(row["slot"], row["variant"]): row for row in decisions}
    variant_rows: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    for pair in tape["pairs"]:
        slot = int(pair["slot"])
        reference = references[slot]
        for variant in VARIANTS:
            decision = by_key[(slot, variant)]
            matches = arrays[f"{variant[0]}_matches_{slot}"]
            blind = tape_blind_input(pair, matches)
            evaluation = _score_decision(decision, blind, reference)
            variant_rows[variant].append(
                {
                    "slot": slot,
                    "pair": f"{pair['left']}->{pair['right']}",
                    "set": pair["set"],
                    "fixture": pair["fixture"],
                    "selected": int(len(matches)),
                    "decision": decision,
                    "evaluation": evaluation,
                    "usable": _usable(evaluation),
                    "false_solve": bool(decision["solved"] and not _usable(evaluation)),
                }
            )

    ds14 = variant_rows["A_reciprocal_ds14"]
    ds14_solved = sum(row["decision"]["solved"] for row in ds14)
    ds14_class_correct = sum(
        bool(row["decision"]["solved"] and row["evaluation"]["class_equals_expected"])
        for row in ds14
    )
    ds14_false = sum(
        bool(row["decision"]["solved"] and not row["evaluation"]["class_equals_expected"])
        for row in ds14
    )
    ds14_correct_slots = [
        row["slot"] for row in ds14
        if row["decision"]["solved"] and row["evaluation"]["class_equals_expected"]
    ]
    ds14_usable_slots = [row["slot"] for row in ds14 if row["usable"]]
    baseline = {
        "solved": int(ds14_solved),
        "class_correct": int(ds14_class_correct),
        "false_class": int(ds14_false),
        "usable": int(len(ds14_usable_slots)),
        "class_correct_slots": ds14_correct_slots,
        "usable_slots": ds14_usable_slots,
        "expected_solved": DS14_BASELINE_SOLVED_REQUIRED,
        "expected_class_correct": DS14_BASELINE_CLASS_CORRECT_REQUIRED,
        "reproduced": bool(
            ds14_solved == DS14_BASELINE_SOLVED_REQUIRED
            and ds14_class_correct == DS14_BASELINE_CLASS_CORRECT_REQUIRED
        ),
    }
    log(
        f"[repro ] DS-14 baseline on the nine pairs: solved {ds14_solved}/9, class-correct "
        f"{ds14_class_correct}/9, usable {len(ds14_usable_slots)}/9 "
        f"(expected {DS14_BASELINE_SOLVED_REQUIRED}/{DS14_BASELINE_CLASS_CORRECT_REQUIRED})"
    )
    return {"ds14_baseline": baseline, "rows": variant_rows}


def evaluate_gates(variant_rows: dict, ds14_correct_slots: list[int], synthetic: dict, log) -> dict:
    gates: dict[str, dict] = {}
    for variant, rows in variant_rows.items():
        usable = sum(row["usable"] for row in rows)
        class_correct = sum(
            bool(row["evaluation"]["class_equals_expected"]) for row in rows
        )
        gauge_ok = sum(
            row["usable"]
            and row["evaluation"]["baseline_error_degrees_gauge_aligned"]
            <= PROMOTION_GAUGE_BASELINE_DEGREES_MAX
            for row in rows
        )
        false_solves = sum(row["false_solve"] for row in rows)
        regressions_correct = [
            row["pair"] for row in rows
            if row["slot"] in ds14_correct_slots and not row["usable"]
        ]
        regressions_strict = [
            row["pair"] for row in rows
            if row["slot"] in {
                other["slot"] for other in variant_rows["A_reciprocal_ds14"] if other["usable"]
            }
            and not row["usable"]
        ]
        passed = bool(
            usable >= PROMOTION_USABLE_REQUIRED
            and class_correct >= PROMOTION_CLASS_CORRECT_REQUIRED
            and gauge_ok >= PROMOTION_USABLE_REQUIRED
            and len(regressions_correct) <= PROMOTION_REGRESSIONS_MAX
            and false_solves <= PROMOTION_FALSE_SOLVES_MAX
            and synthetic.get("all_controls_ok", False)
        )
        gates[variant] = {
            "usable": int(usable),
            "usable_required": PROMOTION_USABLE_REQUIRED,
            "class_correct": int(class_correct),
            "class_correct_required": PROMOTION_CLASS_CORRECT_REQUIRED,
            "gauge_baseline_within_5deg": int(gauge_ok),
            "false_solves": int(false_solves),
            "false_solves_max": PROMOTION_FALSE_SOLVES_MAX,
            "regressions_among_ds14_class_correct_pairs": regressions_correct,
            "regressions_strict_where_ds14_usable": regressions_strict,
            "synthetic_degeneracy_controls_ok": bool(
                synthetic.get("all_controls_ok", False)
            ),
            "passed": passed,
        }
        log(
            f"[gate  ] {variant:26s} usable {usable}/9 class {class_correct}/9 "
            f"false {false_solves} regressions(ds14-correct) {len(regressions_correct)} "
            f"-> {'PASS' if passed else 'fail'}"
        )
    promoted = [variant for variant, gate in gates.items() if gate["passed"]]
    return {
        "preregistered": {
            "usable_required": PROMOTION_USABLE_REQUIRED,
            "class_correct_required": PROMOTION_CLASS_CORRECT_REQUIRED,
            "gauge_baseline_degrees_max": PROMOTION_GAUGE_BASELINE_DEGREES_MAX,
            "false_solves_max": PROMOTION_FALSE_SOLVES_MAX,
            "regressions_among_ds14_correct_max": PROMOTION_REGRESSIONS_MAX,
            "synthetic_degeneracy_controls_required": True,
        },
        "per_variant": gates,
        "promoted_variants": promoted,
        "status": (
            "variant_meets_preregistered_promotion"
            if promoted
            else "no_variant_meets_preregistered_promotion_diagnostic_only"
        ),
    }


def payload_variant_rows(
    variant: str, variant_eval: dict, references: dict, fraction_by_slot: dict
) -> list[dict]:
    """Serialize one variant's evaluated rows for the payload.

    The rows are read by name from ``variant_eval["rows"]``. The CSV loop in
    :func:`run` leaves ``rows`` bound to the last variant, so the payload must
    never reuse that leaked name.
    """
    prefix = variant[0]
    return [
        {
            "slot": row["slot"],
            "pair": row["pair"],
            "selected": row["selected"],
            "decision_solved": row["decision"]["solved"],
            "decision_reason": row["decision"]["reason"],
            "decision_class": row["decision"]["right_yaw_class"],
            "decision_inliers": row["decision"]["inliers"],
            "decision_inlier_fraction": row["decision"]["inlier_fraction"],
            "decision_class_margin": row["decision"]["class_margin"],
            "reference_class": references[row["slot"]]["reference_class"],
            "evaluation": row["evaluation"],
            "usable": row["usable"],
            "false_solve": row["false_solve"],
            "truth_fraction": fraction_by_slot[str(row["slot"])][
                f"{prefix}_selected"
            ],
        }
        for row in variant_eval["rows"][variant]
    ]


# --------------------------------------------------------------------------- #
# Runner.
# --------------------------------------------------------------------------- #
def run(out_dir: Path, log) -> dict:
    started_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    blind_verified = verify_digests(BLIND_INPUT_PATHS, "DS-16/DS-18 blind", log)
    frozen = attribution.load_frozen(DS16_DIR)
    arrays = dict(np.load(ROOT / frozen["candidates_npz"]["path"]))
    if len([pair for pair in frozen["pairs"] if pair["index"] < PAIR_COUNT]) != PAIR_COUNT:
        raise SystemExit("blocker: the DS-16 frozen artifact does not carry the nine pairs")

    # ---------------- Phase A: blind match sets ---------------------------- #
    log("[phaseA] building blind match sets (pixels + M3 geometry only)")
    tape, tape_arrays = build_match_sets(frozen, arrays)
    for pair in tape["pairs"]:
        variants = pair["variants"]
        log(
            f"[tape  ] {pair['left'][11:19]}->{pair['right'][11:19]} "
            f"A {variants['A_reciprocal_ds14']['selected_correspondences']:4d} "
            f"B {variants['B_directed_ds18']['selected_correspondences']:4d} "
            f"C {variants['C_bestof_ratio_epipolar']['selected_correspondences']:4d} "
            f"(graphs A {variants['A_reciprocal_ds14']['candidate_graph_edges']} "
            f"B {variants['B_directed_ds18']['candidate_graph_edges']} "
            f"C {variants['C_bestof_ratio_epipolar']['candidate_graph_edges']})"
        )
        log.checkpoint(f"match sets {pair['slot']}")
    tape = write_match_sets(out_dir, tape, tape_arrays, log)

    # ---------------- Phase B: frozen blind selector ----------------------- #
    log("[phaseB] re-reading the frozen tape and applying the frozen DS-23 selector")
    tape_read, arrays_read = read_match_sets(out_dir, log)
    decisions, read_union = select_match_sets(tape_read, arrays_read)
    decisions_digests = write_decisions(out_dir, decisions, read_union, log)
    freeze = write_freeze(out_dir, tape_read, decisions_digests, decisions, log)
    selection_finished = time.perf_counter()

    # ---------------- Phase C: reference evaluation ------------------------ #
    log("[phaseC] reference evaluation (post-freeze only)")
    evaluator_verified = verify_digests(EVALUATOR_PATHS, "immutable evaluator", log)
    fixtures = reference_models(tape_read, log)
    references = reference_classes(tape_read, fixtures)
    ambiguity = sum(row["evaluator_ambiguity"] for row in references.values())
    log(f"[ref   ] reference classes resolved; floor/direct disagreements {ambiguity}/9")

    truth = reference_truth_fractions(
        tape_read, arrays_read, arrays, fixtures, frozen, log
    )
    variant_eval = evaluate_variants(tape_read, arrays_read, decisions, references, log)
    synthetic = ds23.synthetic_controls(
        log, lambda decision, blind, case: ds23._synthetic_evaluate(
            attribution, decision, blind, case
        )
    )
    gates = evaluate_gates(
        variant_eval["rows"], variant_eval["ds14_baseline"]["class_correct_slots"],
        synthetic, log,
    )
    oracle = reproduce_ds23_oracle(tape_read, fixtures, log)

    runtime = time.perf_counter() - started_all

    # ---------------- artifacts ------------------------------------------- #
    fraction_by_slot = truth["per_pair"]
    csv_rows: list[dict] = []
    for variant, rows in variant_eval["rows"].items():
        prefix = variant[0]
        for row in rows:
            slot = row["slot"]
            decision = row["decision"]
            evaluation = row["evaluation"]
            measured = fraction_by_slot[str(slot)]
            baseline_row = next(
                other for other in variant_eval["rows"]["A_reciprocal_ds14"]
                if other["slot"] == slot
            )
            csv_rows.append(
                {
                    "pair": row["pair"],
                    "set": row["set"],
                    "variant": variant,
                    "candidate_graph_edges": tape_read["pairs"][slot]["variants"][variant][
                        "candidate_graph_edges"
                    ],
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
                    "reference_class": references[slot]["reference_class"],
                    "reference_class_floors": references[slot]["reference_class_floors"],
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
                    "candidate_scorable": measured[f"{prefix}_candidate"]["scorable"],
                    "candidate_true": measured[f"{prefix}_candidate"]["true"],
                    "candidate_truth_fraction": measured[f"{prefix}_candidate"]["fraction"],
                    "selected_scorable": measured[f"{prefix}_selected"]["scorable"],
                    "selected_true": measured[f"{prefix}_selected"]["true"],
                    "selected_truth_fraction": measured[f"{prefix}_selected"]["fraction"],
                    "ds14_baseline_solved": baseline_row["decision"]["solved"],
                    "ds14_baseline_class": baseline_row["decision"]["right_yaw_class"],
                    "ds14_baseline_class_correct":
                        baseline_row["evaluation"]["class_equals_expected"],
                    "ds14_baseline_usable": baseline_row["usable"],
                }
            )
    csv_path = out_dir / "pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    payload = {
        "schema": "ds24-image-derived-pose-policy-1",
        "phase": "blind_match_sets_under_frozen_ds23_selector",
        "status": "complete",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": connected.source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "directive": (
            "DS-24: freeze image-derived match sets (current reciprocal/DS-14, DS-18 "
            "directed, one predeclared production-safe candidate-selection variant) before "
            "any truth read; apply the frozen DS-23 blind selector; score post-freeze with "
            "the immutable evaluator; reproduce the DS-14 baseline; preregistered gate "
            ">=8/9 usable, <=5 deg gauge baseline, zero regressions among DS-14's four "
            "class-correct pairs, <=1 false solve, existing degeneracy controls"
        ),
        "boundary": {
            "phases": [
                "A blind match-set construction (pixels + image-derived M3 cameras)",
                "B blind selection (frozen tape only)",
                "freeze (match-set arrays + decisions hashed)",
                "C reference evaluation (manifests, renders, frozen evaluator)",
            ],
            "variants": list(VARIANTS),
            "variant_C_rule": tape_read["variant_rule"]["C_bestof_ratio_epipolar"],
            "blind_inputs": blind_verified,
            "evaluator": evaluator_verified,
            "trace_reads_union": read_union,
            "trace_unexpected_reads": sorted(
                name for name in read_union
                if name not in blind_policy.GEOMETRIC_INPUT_WHITELIST
            ),
            "freeze": freeze,
        },
        "match_sets": {
            "json": str((out_dir / "match_sets.json").relative_to(ROOT)),
            "npz": str((out_dir / "match_sets.npz").relative_to(ROOT)),
            "arrays_sha256": tape_read["arrays"]["arrays_sha256"],
            "per_array_sha256": tape_read["arrays"]["sha256_by_array"],
        },
        "decisions": decisions_digests,
        "reproduction": {
            "ds14_baseline": variant_eval["ds14_baseline"],
            "ds23_oracle": {
                "ds11_oracle_baseline_solved": oracle["ds11_oracle_baseline_solved"],
                "ds11_oracle_baseline_class_correct":
                    oracle["ds11_oracle_baseline_class_correct"],
                "blind_selector_usable": oracle["blind_selector_usable"],
                "reproduced": oracle["reproduced"],
            },
        },
        "synthetic_controls": synthetic,
        "threshold": gates,
        "truth_fraction": {
            "available": truth["available"],
            "alignment_median_px": truth["alignment_median_px"],
            "rule": truth["rule"],
            "per_pair": truth["per_pair"],
        },
        "per_variant": {
            variant: {
                "usable": gates["per_variant"][variant]["usable"],
                "class_correct": gates["per_variant"][variant]["class_correct"],
                "false_solves": gates["per_variant"][variant]["false_solves"],
                "passed": gates["per_variant"][variant]["passed"],
                "rows": payload_variant_rows(
                    variant, variant_eval, references, fraction_by_slot
                ),
            }
            for variant in VARIANTS
        },
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
        "run": {
            "runtime_seconds": runtime,
            "selection_seconds": selection_finished - started_all,
            "peak_rss_bytes": log.peak_rss_bytes,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
        },
        "artifacts": {},
    }
    payload["artifacts"]["match_sets.json"] = {
        "bytes": (out_dir / "match_sets.json").stat().st_size,
        "sha256": connected.digest(out_dir / "match_sets.json"),
    }
    payload["artifacts"]["match_sets.npz"] = {
        "bytes": (out_dir / "match_sets.npz").stat().st_size,
        "sha256": connected.digest(out_dir / "match_sets.npz"),
    }
    payload["artifacts"]["decisions.json"] = {
        "bytes": (out_dir / "decisions.json").stat().st_size,
        "sha256": connected.digest(out_dir / "decisions.json"),
    }
    payload["artifacts"]["freeze.json"] = {
        "bytes": (out_dir / "freeze.json").stat().st_size,
        "sha256": freeze["file_sha256"],
    }
    payload["artifacts"]["pairs.csv"] = {
        "bytes": csv_path.stat().st_size,
        "sha256": connected.digest(csv_path),
    }
    evaluation_path = out_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(payload, indent=2, default=connected.plain), encoding="utf-8"
    )
    payload["artifacts"]["evaluation.json"] = {
        "bytes": evaluation_path.stat().st_size,
        "sha256": connected.digest(evaluation_path),
    }
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    arguments = parser.parse_args(argv)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(out_dir / "run.log")
    try:
        payload = run(out_dir, log)
    except MemoryBudgetExceeded as error:
        log(f"[abort ] {error}")
        log.close()
        return 4
    log(
        f"[done  ] status {payload['threshold']['status']}; DS-14 baseline "
        f"{payload['reproduction']['ds14_baseline']['solved']}/9 solved, "
        f"{payload['reproduction']['ds14_baseline']['class_correct']}/9 class-correct; "
        f"runtime {payload['run']['runtime_seconds']:.1f}s; peak RSS "
        f"{(payload['run']['peak_rss_bytes'] or 0) / 2**20:.1f} MiB"
    )
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
