"""DS-23 - label-blind pose-policy audit of the DS-19 support tie-break.

**Two phases, one process, hard boundary.** DS-19 showed that a support-margin
tie-break repairs the three ``ambiguous_yaw_class`` DS-11 refusals on frozen
oracle edges and scores 9/9 under the frozen DS-16B evaluator. DS-19's
selection, however, was only *argued* to be label-free; DS-23 proves it.

Phase A - **tape construction (reference-assisted, pre-selection).**
Reads the digest-verified DS-16 frozen candidates and the frozen DS-16/DS-16B
artifacts, and (only here) the fixture manifests/litematics/scene transforms to
render reference pixel ownership. It materialises, per pair:

* the frozen oracle best-of correspondence set, produced by the frozen DS-16B
  geometric rule re-implemented locally (predicate and bounded chunking copied
  from DS-16/DS-19, cross-checked against the frozen DS-16 and DS-16B counts),
* the image-derived camera geometry (intrinsics + orientation) for both views.

Nothing else. The tape carries no pair identity, no reference class, no
manifest camera, no evaluator output: see ``blind_tape.json`` +
``blind_tape.npz``, digest-frozen before phase B.

Phase B - **selection (label-blind).**
Runs ``scripts/blind_pose_policy.select_blind_pose`` on the tape only. The
selector is a clean-room refactor of DS-19's ``adjudicate`` /
``class_hypothesis`` / ``fit_class`` / ``choose_sign`` / ``_gate_check`` maths,
verified bit-identical to DS-19 on the same geometry (equivalence certificate in
the payload). It receives ``BlindPairInput`` (intrinsics, poses, pixel matches,
image size) and the frozen policy constants; the audit trace records every
attribute it reads.

Phase C - **evaluation (post-selection, reference-only).**
Only now are reference labels touched: the frozen DS-16B evaluator
(``diagnose_match_attribution._evaluate_pose``) scores each frozen decision, the
reference yaw class is compared, the DS-11 baseline read-out is reproduced, and
the DS-19 artifact is cross-checked. No decision is recomputed after scoring.

Preregistered gates (fixed before the run, unchanged during it):

1. reproduce the frozen DS-11 oracle baseline **6/9** solved / class-correct;
2. blind selection **>= 8/9** usable (correct reference yaw class + frozen
   ``within_5deg`` + frozen gauge-aligned baseline error <= 5 deg);
3. **zero regressions** among the original six class-correct pairs;
4. **all existing synthetic degeneracy refusals** hold (valid/parallel solved,
   pure rotation refused).

Run once with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts\\diagnose_blind_pose_policy.py

No production, DS-11..DS-22 source, artifact or threshold is modified. No
commit, no delegation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider  # noqa: E402
from mc_photogrammetry.export.attribution import (  # noqa: E402
    FACE_BASIS,
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_result,
)
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.pipeline.connected_camera import (  # noqa: E402
    PairGates,
    solve_pair,
)
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    camera_eye,
    camera_rotation_matrix,
    intrinsics_for,
    load_camera_manifest,
    project_points,
)
from mc_photogrammetry.reconstruction.scene import load_scene_transform  # noqa: E402
from mc_photogrammetry.vision.gui_mask import create_gui_mask  # noqa: E402

# Phase A helpers reused from the DS-16/DS-19 diagnostic scripts (geometry only).
from diagnose_connected_pairs import digest, plain, source_revision  # noqa: E402
from diagnose_joint_pairs import _estimate_from_record  # noqa: E402
from diagnose_match_attribution import (  # noqa: E402
    FACE_SLOTS_INDEX,
    REFERENCE_RENDER_SCALE,
    load_frozen,
)

# Phase B - the boundary under audit.
from blind_pose_policy import (  # noqa: E402
    FORBIDDEN_DECISION_INPUTS,
    FROZEN_POLICY,
    GEOMETRIC_INPUT_WHITELIST,
    select_blind_pose,
    start_trace,
    stop_trace,
    to_blind_input,
)

DS16_DIR = ROOT / "exports" / "diagnostics" / "ds16"
DS16B_DIR = ROOT / "exports" / "diagnostics" / "ds16b"
DS19_DIR = ROOT / "exports" / "diagnostics" / "ds19"
OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds23"

EXPECTED_CANDIDATES_JSON_SHA256 = (
    "e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93"
)
EXPECTED_CANDIDATES_NPZ_SHA256 = (
    "3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4"
)
EXPECTED_ATTRIBUTION_JSON_SHA256 = (
    "72da240a1992b31640ddbfb4e4bd0b6fb6c1b02e4ede1abd6a7507b74ea7a038"
)
EXPECTED_SWEEP_JSON_SHA256 = (
    "397294502f902f84d54abee047bc1a52939f843fd8016147577e5020289f1cf0"
)
EXPECTED_DS19_JSON_SHA256 = (
    "0664facd955676a25ba302c9361fbcc57746d56595e94afdc97c2c9204002e79"
)

BASE_DISTANCE_BLOCKS = 0.5
BASE_NORMAL_DEGREES = 30.0
RENDER_SCALE = REFERENCE_RENDER_SCALE
SELECTED_SETS = ("T2", "T6")
PAIR_COUNT = 9
BASE_CELL = "d0.5_n30"

TRUTH_CHUNK = 64
TRUTH_MATERIALIZE_MAX_ROWS = 5_000_000
RSS_LIMIT_BYTES = 1_800 * 1024 * 1024
ALIGNMENT_MEDIAN_PX_MAX = 10.0
SYNTHETIC_REPEATS = 24
SYNTHETIC_SEED = 19
DECISION_TAU = 1e-9  # frozen-decision equivalence tolerance

BASELINE_SOLVES_REQUIRED = 6
PROMOTION_USABLE_REQUIRED = 8

PAIR_CSV_COLUMNS = (
    "pair", "set", "correspondences", "decision_solved", "decision_reason",
    "decision_class", "decision_inliers", "decision_inlier_fraction",
    "decision_median_rho_degrees", "decision_parallax_degrees",
    "decision_cheirality_fraction", "decision_singular_ratio",
    "decision_class_margin", "trace_reads",
    "reference_class", "reference_class_floors",
    "evaluated_class_correct", "evaluated_within_5deg",
    "evaluated_rotation_error_degrees",
    "evaluated_baseline_error_degrees_gauge_aligned",
    "evaluated_baseline_error_degrees_signless", "evaluated_gauge_index",
    "usable", "production_bestof_solved", "production_bestof_class",
    "production_bestof_class_correct", "production_bestof_usable",
    "ds19_artifact_class_agrees", "ds19_artifact_baseline_max_abs_diff",
    "production_failure_class",
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


def _normal_cosine(degrees: float) -> float:
    return float(np.cos(np.radians(degrees)))


# --------------------------------------------------------------------------- #
# Phase A: reference render -> per-keypoint ownership/world (pre-selection).
# --------------------------------------------------------------------------- #
def _scaled_unproject(col: int, row: int, depth: float, camera, scale: float,
                      width: int, height: int) -> np.ndarray:
    calibration = intrinsics_for(camera, width, height)
    focal_x, focal_y = calibration.focal_length_pixels
    centre_x, centre_y = calibration.principal_point_pixels
    view = np.asarray(
        (
            (col - centre_x * scale) / (focal_x * scale) * depth,
            -(row - centre_y * scale) / (focal_y * scale) * depth,
            depth,
        ),
        dtype=np.float64,
    )
    return camera_rotation_matrix(camera).T @ view + camera_eye(camera)


def _alignment_errors(points, world, camera, width, height) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    projected = project_points(
        [tuple(value) for value in world], camera, width=width, height=height
    )
    errors = []
    for (pixel_x, pixel_y), point in zip(points, projected):
        if point is None:
            errors.append(np.inf)
        else:
            errors.append(float(np.hypot(point[0] - pixel_x, point[1] - pixel_y)))
    return np.asarray(errors, dtype=np.float64)


def label_keypoints(points, result, index, camera, scale, usable_full,
                    width, height) -> dict:
    """Map keypoints to reference-hit world points (rule copied from DS-16)."""
    rows, columns = result.ownership.shape
    count = len(points)
    hit = np.zeros(count, dtype=bool)
    normal = np.zeros((count, 3), dtype=np.float64)
    world = np.full((count, 3), np.nan, dtype=np.float64)
    if count == 0:
        return {"hit": hit, "normal": normal, "world": world,
                "alignment_px": np.empty(0, dtype=np.float64)}
    columns_index = np.clip(np.rint(points[:, 0] * scale).astype(np.int64), 0, columns - 1)
    rows_index = np.clip(np.rint(points[:, 1] * scale).astype(np.int64), 0, rows - 1)
    owner = result.ownership[rows_index, columns_index]
    depth = np.asarray(result.depth[rows_index, columns_index], dtype=np.float64)
    pixel_x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    pixel_y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    hit = (owner >= 0) & np.isfinite(depth) & usable_full[pixel_y, pixel_x]
    for position in np.flatnonzero(hit):
        _coordinate, face_slot = index[int(owner[position])]
        normal[position] = FACE_BASIS[face_slot][0]
        world[position] = _scaled_unproject(
            int(columns_index[position]), int(rows_index[position]),
            float(depth[position]), camera, scale, width, height,
        )
    return {
        "hit": hit,
        "normal": normal,
        "world": world,
        "alignment_px": _alignment_errors(points[hit], world[hit], camera, width, height),
    }


def oracle_chunks(left: dict, right: dict, tolerance: float, normal_cosine: float,
                  chunk: int = TRUTH_CHUNK):
    """Frozen DS-16 oracle predicate, bounded per left-keypoint chunk."""
    left_index = np.flatnonzero(left["hit"])
    right_index = np.flatnonzero(right["hit"])
    if len(left_index) == 0 or len(right_index) == 0:
        return
    left_world, left_normal = left["world"][left_index], left["normal"][left_index]
    right_world, right_normal = right["world"][right_index], right["normal"][right_index]
    right_count = int(max(len(right["hit"]), 1))
    cell = max(float(tolerance), 1e-6)
    right_cell = np.floor(right_world / cell).astype(np.int64)
    left_cell = np.floor(left_world / cell).astype(np.int64)
    minimum = np.minimum(right_cell.min(axis=0), left_cell.min(axis=0)) - 1
    dims = np.maximum(right_cell.max(axis=0), left_cell.max(axis=0)) - minimum + 2

    def encode(block: np.ndarray) -> np.ndarray:
        return ((block[:, 0] - minimum[0]) * dims[1] + (block[:, 1] - minimum[1])) * dims[2] \
            + (block[:, 2] - minimum[2])

    right_key = encode(right_cell)
    order = np.argsort(right_key, kind="stable")
    right_key_sorted = right_key[order]
    shifts = [
        (dx * dims[1] + dy) * dims[2] + dz
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for start in range(0, len(left_index), chunk):
        world = left_world[start:start + chunk]
        normal = left_normal[start:start + chunk]
        origin = left_index[start:start + chunk]
        key = encode(np.floor(world / cell).astype(np.int64))
        codes_parts: list[np.ndarray] = []
        for shift in shifts:
            shifted = key + shift
            low = np.searchsorted(right_key_sorted, shifted, side="left")
            high = np.searchsorted(right_key_sorted, shifted, side="right")
            counts = high - low
            total = int(counts.sum())
            if total == 0:
                continue
            left_rows = np.repeat(np.arange(len(shifted), dtype=np.int64), counts)
            base = np.repeat(np.cumsum(counts) - counts, counts)
            gathered = np.repeat(low, counts) + (np.arange(total) - base)
            right_rows = order[gathered]
            distance = np.linalg.norm(world[left_rows] - right_world[right_rows], axis=1)
            dot = np.einsum("ij,ij->i", normal[left_rows], right_normal[right_rows])
            keep = (distance <= tolerance) & (dot >= normal_cosine)
            if np.any(keep):
                codes_parts.append(
                    np.unique(origin[left_rows[keep]] * right_count + right_index[right_rows[keep]])
                )
        del world, normal, key, shifted, low, high, counts
        if not codes_parts:
            continue
        codes = np.unique(np.concatenate(codes_parts))
        yield int(origin[0]), int(origin[-1]), np.column_stack(
            (codes // right_count, codes % right_count)
        )


def oracle_edges(left: dict, right: dict, tolerance: float, normal_cosine: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    rows = 0
    for _lo, _hi, edges in oracle_chunks(left, right, tolerance, normal_cosine):
        parts.append(edges)
        rows += len(edges)
        if rows > TRUTH_MATERIALIZE_MAX_ROWS:
            raise MemoryBudgetExceeded(
                f"one pair's oracle set exceeded {TRUTH_MATERIALIZE_MAX_ROWS} rows"
            )
    if not parts:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(parts)


def best_of_edges(truth: np.ndarray, left_world: np.ndarray,
                  right_world: np.ndarray) -> np.ndarray:
    """One oracle edge per left keypoint: nearest world hit, index tie-break."""
    if len(truth) == 0:
        return np.empty((0, 2), dtype=np.int64)
    distance = np.linalg.norm(left_world[truth[:, 0]] - right_world[truth[:, 1]], axis=1)
    order = np.lexsort((truth[:, 1], distance, truth[:, 0]))
    ordered = truth[order]
    first = np.ones(len(ordered), dtype=bool)
    first[1:] = ordered[1:, 0] != ordered[:-1, 0]
    return ordered[first]


def matches_from_edges(left_points: np.ndarray, right_points: np.ndarray,
                       edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.float64)
    left_i, right_j = edges[:, 0], edges[:, 1]
    return np.column_stack(
        (
            left_points[left_i, 0],
            left_points[left_i, 1],
            right_points[right_j, 0],
            right_points[right_j, 1],
        )
    )


def camera_estimate(model: CameraModel, width: int, height: int) -> CameraEstimate:
    return CameraEstimate(
        intrinsics=intrinsics_for(model, width, height),
        pose=CameraPose(
            position=model.position,
            yaw_degrees=model.yaw_degrees,
            pitch_degrees=model.pitch_degrees,
            roll_degrees=model.roll_degrees,
        ),
    )


def image_estimate(record: dict) -> CameraEstimate:
    """Image-derived (M3) estimate from the frozen DS-16 record - geometry only."""
    estimate = _estimate_from_record(record)
    return CameraEstimate(
        intrinsics=CameraIntrinsics(
            image_size=(estimate.width, estimate.height),
            focal_length_pixels=(estimate.focal_pixels, estimate.focal_pixels),
            principal_point_pixels=estimate.principal_point,
        ),
        pose=CameraPose(
            position=None,
            yaw_degrees=estimate.yaw_degrees,
            pitch_degrees=estimate.pitch_degrees,
            roll_degrees=estimate.roll_degrees,
        ),
    )


#: Geometry allowlist for the frozen blind tape: every other field the tape
#: carries is evidence bookkeeping and is never touched by the decision path.
TAPE_GEOMETRIC_FIELDS = (
    "slot",
    "width",
    "height",
    "correspondences",
    "left_intrinsics",
    "left_pose",
    "right_intrinsics",
    "right_pose",
)


def tape_blind_input(pair: dict, tape_arrays) -> tuple:
    """Rebuild the decision input from the *written* tape only.

    Reads no in-memory camera object from phase A: intrinsics and poses are
    reconstructed from ``blind_tape.json`` and the correspondences from
    ``blind_tape.npz``, so the selection can only see the frozen tape.
    """
    matches = np.asarray(tape_arrays[f"tp_matches_{pair['slot']}"], dtype=np.float64)
    if digest_array(matches) != pair["matches_sha256"]:
        raise SystemExit(
            f"blocker: blind tape slot {pair['slot']} correspondences do not match their "
            "recorded digest; refusing to select"
        )

    def _intrinsics(record: dict):
        return CameraIntrinsics(
            image_size=tuple(int(value) for value in record["image_size"]),
            focal_length_pixels=tuple(
                float(value) for value in record["focal_length_pixels"]
            ),
            principal_point_pixels=tuple(
                float(value) for value in record["principal_point_pixels"]
            ),
        )

    def _pose(record: dict):
        return CameraPose(
            position=None,
            yaw_degrees=float(record["yaw_degrees"]),
            pitch_degrees=float(record["pitch_degrees"]),
            roll_degrees=float(record["roll_degrees"]),
        )

    left_intrinsics = _intrinsics(pair["left_intrinsics"])
    right_intrinsics = _intrinsics(pair["right_intrinsics"])
    blind = to_blind_input(
        left_intrinsics=left_intrinsics,
        left_pose=_pose(pair["left_pose"]),
        right_intrinsics=right_intrinsics,
        right_pose=_pose(pair["right_pose"]),
        matches=matches,
        image_size=(int(pair["width"]), int(pair["height"])),
    )
    return blind, matches


def tape_cameras(pair: dict) -> tuple[CameraEstimate, CameraEstimate]:
    """Reconstruct the two image-derived camera estimates from tape fields only."""
    def _estimate(intrinsics_record: dict, pose_record: dict) -> CameraEstimate:
        return CameraEstimate(
            intrinsics=CameraIntrinsics(
                image_size=tuple(int(value) for value in intrinsics_record["image_size"]),
                focal_length_pixels=tuple(
                    float(value) for value in intrinsics_record["focal_length_pixels"]
                ),
                principal_point_pixels=tuple(
                    float(value) for value in intrinsics_record["principal_point_pixels"]
                ),
            ),
            pose=CameraPose(
                position=None,
                yaw_degrees=float(pose_record["yaw_degrees"]),
                pitch_degrees=float(pose_record["pitch_degrees"]),
                roll_degrees=float(pose_record["roll_degrees"]),
            ),
        )

    return (
        _estimate(pair["left_intrinsics"], pair["left_pose"]),
        _estimate(pair["right_intrinsics"], pair["right_pose"]),
    )


# --------------------------------------------------------------------------- #
# Phase B: blind selection record -> the frozen evaluator's pose handle.
# --------------------------------------------------------------------------- #
class _DecisionPose:
    """Carry a frozen blind decision to the frozen evaluator (fields only).

    Rotation and baseline use exactly the ``solve_pair`` conventions DS-19
    established for this hand-over, so the frozen metric reads the blind pose in
    the same gauge it reads a production pose. No evaluation maths lives here.
    """

    def __init__(self, decision, blind) -> None:
        class_index = int(decision.right_yaw_class)
        rotation_right_class = camera_rotation_matrix(
            CameraModel(
                position=(0.0, 0.0, 0.0),
                yaw_degrees=blind.right_yaw_degrees,
                pitch_degrees=blind.right_pitch_degrees,
                roll_degrees=blind.right_roll_degrees,
                eye_height=0.0,
            )
        ) @ _ry90(class_index)
        rotation_left = camera_rotation_matrix(
            CameraModel(
                position=(0.0, 0.0, 0.0),
                yaw_degrees=blind.left_yaw_degrees,
                pitch_degrees=blind.left_pitch_degrees,
                roll_degrees=blind.left_roll_degrees,
                eye_height=0.0,
            )
        )
        self.solved = True
        self.reason = ""
        self.right_yaw_class = class_index
        self.relative_rotation = rotation_right_class @ rotation_left.T
        direction = np.asarray(decision.baseline_direction, dtype=np.float64)
        self.baseline_direction = tuple(float(value) for value in direction)
        self.inliers = int(decision.inliers)
        self.inlier_fraction = float(decision.inlier_fraction)
        self.median_rho_degrees = decision.median_rho_degrees
        self.parallax_degrees = decision.parallax_degrees
        self.cheirality_fraction = decision.cheirality_fraction


def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (yaw_class % 4))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# Synthetic degeneracy controls (frozen expectation, DS-19 families).
# --------------------------------------------------------------------------- #
SYNTHETIC_POSES = {
    "valid_baseline": ((0.0, 0.0), (0.0, 0.0), False),
    "parallel_axes": ((30.0, 0.0), (30.0, 0.0), False),
    "pure_rotation": ((0.0, 0.0), (25.0, 0.0), True),
}
SYNTHETIC_EXPECTATION = {
    "valid_baseline": "solved",
    "parallel_axes": "solved",
    "pure_rotation": "refused",
}


def _synthetic_model(position, yaw_degrees, pitch_degrees) -> CameraModel:
    return CameraModel(
        position=tuple(float(value) for value in position),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        roll_degrees=0.0,
        eye_height=1.62,
    )


def _synthetic_case(kind: str, rng: np.random.Generator) -> dict:
    (yaw_left, pitch_left), (yaw_right, pitch_right), same_eye = SYNTHETIC_POSES[kind]
    width, height = 640, 480
    left_position = np.asarray((-3.0, 0.0, 0.0), dtype=np.float64)
    right_position = left_position.copy() if same_eye else np.asarray((3.0, 0.0, 0.0), dtype=np.float64)
    left_model = _synthetic_model(left_position, yaw_left, pitch_left)
    right_model = _synthetic_model(right_position, yaw_right, pitch_right)
    intrinsics = intrinsics_for(left_model, width, height)

    def estimate(position, yaw, pitch):
        return CameraEstimate(
            intrinsics=intrinsics,
            pose=CameraPose(
                position=(float(position[0]), float(position[1] + 1.62), float(position[2])),
                yaw_degrees=float(yaw),
                pitch_degrees=float(pitch),
                roll_degrees=0.0,
            ),
        )

    left_estimate = estimate(left_position, yaw_left, pitch_left)
    right_estimate = estimate(right_position, yaw_right, pitch_right)
    centre = rng.uniform(-12.0, 12.0, 3) + np.asarray((0.0, 0.0, 48.0))
    points = centre + rng.uniform(-22.0, 22.0, (360, 3))
    left_pixels = project_points([tuple(point) for point in points], left_model,
                                 width=width, height=height)
    right_pixels = project_points([tuple(point) for point in points], right_model,
                                  width=width, height=height)
    pairs = [
        (float(lp[0]), float(lp[1]), float(rp[0]), float(rp[1]))
        for lp, rp in zip(left_pixels, right_pixels)
        if lp is not None and rp is not None
    ]
    matches = np.asarray(pairs, dtype=np.float64).reshape(-1, 4)
    return {
        "kind": kind,
        "left_camera": left_estimate,
        "right_camera": right_estimate,
        "left_model": left_model,
        "right_model": right_model,
        "matches": matches,
        "width": width,
        "height": height,
    }


def synthetic_controls(log: _Log, evaluate) -> dict:
    """Blind selector on ideal-correspondence degenerate/normal geometry."""
    rows: list[dict] = []
    for kind, expectation in SYNTHETIC_EXPECTATION.items():
        solved = 0
        usable = 0
        reasons: Counter = Counter()
        classes: Counter = Counter()
        for repeat in range(SYNTHETIC_REPEATS):
            rng = np.random.default_rng(SYNTHETIC_SEED + 1000 * repeat)
            case = _synthetic_case(kind, rng)
            matches = case["matches"]
            if len(matches) < FROZEN_POLICY.edge_consensus_min:
                reasons["insufficient_correspondences"] += 1
                continue
            blind = to_blind_input(
                left_intrinsics=case["left_camera"].intrinsics,
                left_pose=case["left_camera"].pose,
                right_intrinsics=case["right_camera"].intrinsics,
                right_pose=case["right_camera"].pose,
                matches=matches,
                image_size=(case["width"], case["height"]),
            )
            decision = select_blind_pose(blind)
            if decision.solved:
                solved += 1
                classes[decision.right_yaw_class] += 1
                usable += int(evaluate(decision, blind, case))
            else:
                reasons[decision.reason] += 1
        if expectation == "refused":
            control_ok = solved == 0
        else:
            control_ok = usable == SYNTHETIC_REPEATS
        rows.append(
            {
                "kind": kind,
                "expectation": expectation,
                "repeats": SYNTHETIC_REPEATS,
                "solved": solved,
                "usable": usable,
                "refused": SYNTHETIC_REPEATS - solved,
                "refusals": dict(sorted(reasons.items())),
                "classes": {str(key): value for key, value in sorted(classes.items())},
                "control_ok": bool(control_ok),
            }
        )
        log(
            f"[synth ] {kind:15s} expect {expectation:7s} usable {usable}/{SYNTHETIC_REPEATS} "
            f"refusals {dict(sorted(reasons.items()))} control "
            f"{'ok' if control_ok else 'FAILED'}"
        )
    return {
        "families": rows,
        "all_controls_ok": all(row["control_ok"] for row in rows),
    }


# --------------------------------------------------------------------------- #
# Runner.
# --------------------------------------------------------------------------- #
def run(out_dir: Path, log: _Log) -> dict:
    started_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tape_json_path = out_dir / "blind_tape.json"
    tape_npz_path = out_dir / "blind_tape.npz"

    # ---------------- frozen-input verification (before any label) --------- #
    candidates_json = DS16_DIR / "match_candidates.json"
    attribution_json = DS16_DIR / "match_attribution.json"
    sweep_json = DS16B_DIR / "match_attribution_sweep.json"
    ds19_json = DS19_DIR / "oracle_pose_failures.json"
    digests = {
        "match_candidates.json": {
            "path": str(candidates_json.relative_to(ROOT)),
            "actual": digest(candidates_json),
            "expected": EXPECTED_CANDIDATES_JSON_SHA256,
        },
        "match_candidates.npz": {
            "path": str((DS16_DIR / "match_candidates.npz").relative_to(ROOT)),
            "actual": digest(DS16_DIR / "match_candidates.npz"),
            "expected": EXPECTED_CANDIDATES_NPZ_SHA256,
        },
        "match_attribution.json": {
            "path": str(attribution_json.relative_to(ROOT)),
            "actual": digest(attribution_json),
            "expected": EXPECTED_ATTRIBUTION_JSON_SHA256,
        },
        "match_attribution_sweep.json": {
            "path": str(sweep_json.relative_to(ROOT)),
            "actual": digest(sweep_json),
            "expected": EXPECTED_SWEEP_JSON_SHA256,
        },
        "oracle_pose_failures.json": {
            "path": str(ds19_json.relative_to(ROOT)),
            "actual": digest(ds19_json),
            "expected": EXPECTED_DS19_JSON_SHA256,
        },
    }
    for name, record in digests.items():
        record["ok"] = record["actual"] == record["expected"]
        if not record["ok"]:
            raise SystemExit(
                f"frozen input {name} was modified: sha256 {record['actual']} "
                f"!= preregistered {record['expected']}"
            )
    log("[verify] all five frozen DS-16/DS-16B/DS-19 digests match")

    frozen = load_frozen(DS16_DIR)
    frozen_attribution = json.loads(attribution_json.read_text(encoding="utf-8"))
    known_attribution = {
        (row["fixture"], row["left"], row["right"]): row
        for row in frozen_attribution["pairs"]
    }
    known_sweep = json.loads(sweep_json.read_text(encoding="utf-8"))
    sweep_cells = {cell["cell"]: cell for cell in known_sweep["cells"]}
    if BASE_CELL not in sweep_cells:
        raise SystemExit(f"frozen DS-16B sweep has no {BASE_CELL} cell; refusing to run")
    known_pairs = {
        (row["fixture"], row["left"], row["right"]): row
        for row in sweep_cells[BASE_CELL]["pairs"]
    }
    frozen_ds19 = json.loads(ds19_json.read_text(encoding="utf-8"))
    ds19_pairs = {
        f"{row['left']}->{row['right']}": row for row in frozen_ds19["cell"]["pairs"]
    }

    pairs = [pair for pair in frozen["pairs"] if pair["set"] in SELECTED_SETS]
    if len(pairs) != PAIR_COUNT:
        raise SystemExit(
            f"selection rule expected the nine T2/T6 pairs, found {len(pairs)}; refusing to run"
        )
    log(f"[select] {len(pairs)} T2/T6 pairs at the frozen base cell {BASE_CELL}")

    # ---------------- Phase A: tape construction --------------------------- #
    import os

    provider = VanillaJarTextureProvider()
    provider.load_jar(Path(os.environ["APPDATA"]) / ".minecraft/versions/1.21.5/1.21.5.jar")
    palette = provider.complete_palette()
    colours = face_colours(palette)
    shapes = face_shapes(palette)
    occluding_slots = face_occluding_slots(palette)
    model_faces = face_model_faces(palette)

    arrays = np.load(ROOT / frozen["candidates_npz"]["path"])
    frames = {record["frame"]: record for record in frozen["frames"]}

    fixtures: dict[str, dict] = {}
    for pair in pairs:
        fixture = pair["fixture"]
        if fixture in fixtures:
            continue
        folder = ROOT / fixture
        litematic_path = next(folder.glob("*.litematic"))
        manifest_path = folder / "cameras.json"
        scene_path = folder / "scene.json"
        fixtures[fixture] = {
            "folder": fixture,
            "litematic": str(litematic_path.relative_to(ROOT)),
            "litematic_sha256": digest(litematic_path),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(manifest_path),
            "scene": str(scene_path.relative_to(ROOT)),
            "scene_sha256": digest(scene_path),
            "reference": load_litematic(litematic_path),
            "scene_transform": load_scene_transform(scene_path),
            "cameras": load_camera_manifest(manifest_path),
        }

    needed_frames = sorted({pair["left"] for pair in pairs} | {pair["right"] for pair in pairs})
    labels: dict[str, dict] = {}
    alignment: list[dict] = []
    for name in needed_frames:
        record = frames[name]
        fixture = fixtures[record["fixture"]]
        camera = fixture["cameras"][name]
        width, height = int(record["width"]), int(record["height"])
        started = time.perf_counter()
        result, index = render_result(
            fixture["reference"].blocks,
            fixture["scene_transform"],
            camera,
            width,
            height,
            colours,
            shapes=shapes,
            scale=RENDER_SCALE,
            occluding_slots=occluding_slots,
            model_faces=model_faces,
        )
        points = np.asarray(arrays[f"feat_points_{record['key']}"], dtype=np.float64)
        usable = ~create_gui_mask(height, width)
        labels[name] = label_keypoints(
            points, result, index, camera, RENDER_SCALE, usable, width, height
        )
        errors = labels[name]["alignment_px"]
        alignment.append(
            {
                "frame": name,
                "fixture": record["fixture"],
                "render_seconds": time.perf_counter() - started,
                "keypoints": int(len(points)),
                "hits": int(np.count_nonzero(labels[name]["hit"])),
                "alignment_median_px": float(np.median(errors)) if len(errors) else None,
                "alignment_p90_px": float(np.percentile(errors, 90)) if len(errors) else None,
            }
        )
        log(f"[render] {name[:24]:24s} kp {len(points):5d} "
            f"render {alignment[-1]['render_seconds']:.2f}s")
        del result, index, points, usable
        log.checkpoint(f"render {name}")

    medians = [row["alignment_median_px"] for row in alignment
               if row["alignment_median_px"] is not None]
    alignment_median = float(np.median(medians)) if medians else None
    if not medians or alignment_median > ALIGNMENT_MEDIAN_PX_MAX:
        raise SystemExit(
            "blocker: reference renderer ownership does not align with the manifest cameras "
            f"(median re-projection error {alignment_median} px); refusing to trust oracle labels"
        )
    log(f"[align] median frame re-projection error {alignment_median:.4f} px "
        f"(<= {ALIGNMENT_MEDIAN_PX_MAX})")

    normal_cosine = _normal_cosine(BASE_NORMAL_DEGREES)
    tape_pairs: list[dict] = []
    tape_arrays: dict[str, np.ndarray] = {}
    for position, pair in enumerate(pairs):
        fixture = fixtures[pair["fixture"]]
        left_name, right_name = pair["left"], pair["right"]
        left_record, right_record = frames[left_name], frames[right_name]
        left_points = np.asarray(arrays[f"feat_points_{left_record['key']}"], dtype=np.float64)
        right_points = np.asarray(arrays[f"feat_points_{right_record['key']}"], dtype=np.float64)
        left_label, right_label = labels[left_name], labels[right_name]
        width, height = int(left_record["width"]), int(left_record["height"])

        truth = oracle_edges(left_label, right_label, BASE_DISTANCE_BLOCKS, normal_cosine)
        best = best_of_edges(truth, left_label["world"], right_label["world"])

        reference_row = known_attribution.get((pair["fixture"], left_name, right_name))
        sweep_row = known_pairs.get((pair["fixture"], left_name, right_name))
        cross = {
            "ds16_oracle_truths": None if reference_row is None
            else reference_row["candidate_stats"]["oracle_truths"],
            "ds16b_oracle_truths": None if sweep_row is None
            else sweep_row["oracle_truths"],
            "ds16b_oracle_bestof_edges": None if sweep_row is None
            else sweep_row["oracle_bestof_edges"],
            "recomputed_oracle_truths": int(len(truth)),
            "recomputed_oracle_bestof_edges": int(len(best)),
        }
        cross["oracle_truths_match"] = (
            cross["recomputed_oracle_truths"] == cross["ds16_oracle_truths"]
            and cross["recomputed_oracle_truths"] == cross["ds16b_oracle_truths"]
        )
        cross["oracle_bestof_match"] = (
            cross["recomputed_oracle_bestof_edges"] == cross["ds16b_oracle_bestof_edges"]
        )
        if not (cross["oracle_truths_match"] and cross["oracle_bestof_match"]):
            raise SystemExit(
                f"blocker: pair {left_name}->{right_name} does not reproduce the frozen "
                f"DS-16/DS-16B truth counts: {json.dumps(cross, default=plain)}"
            )

        matches = matches_from_edges(left_points, right_points, best)
        left_est = image_estimate(pair["left_estimate"])
        right_est = image_estimate(pair["right_estimate"])
        left_model, right_model = fixture["cameras"][left_name], fixture["cameras"][right_name]
        tape_arrays[f"tp_matches_{position}"] = matches
        tape_pairs.append(
            {
                "slot": position,
                "set": pair["set"],
                "fixture": pair["fixture"],
                "left": left_name,
                "right": right_name,
                "width": width,
                "height": height,
                "correspondences": int(len(matches)),
                "oracle_truths_frozen_crosscheck_only": int(len(truth)),
                "left_intrinsics": {
                    "image_size": list(left_est.intrinsics.image_size),
                    "focal_length_pixels": list(left_est.intrinsics.focal_length_pixels),
                    "principal_point_pixels": list(left_est.intrinsics.principal_point_pixels),
                },
                "right_intrinsics": {
                    "image_size": list(right_est.intrinsics.image_size),
                    "focal_length_pixels": list(right_est.intrinsics.focal_length_pixels),
                    "principal_point_pixels": list(right_est.intrinsics.principal_point_pixels),
                },
                "left_pose": {
                    "position": None,
                    "yaw_degrees": float(left_est.pose.yaw_degrees),
                    "pitch_degrees": float(left_est.pose.pitch_degrees),
                    "roll_degrees": float(left_est.pose.roll_degrees),
                },
                "right_pose": {
                    "position": None,
                    "yaw_degrees": float(right_est.pose.yaw_degrees),
                    "pitch_degrees": float(right_est.pose.pitch_degrees),
                    "roll_degrees": float(right_est.pose.roll_degrees),
                },
                "matches_sha256": digest_array(matches),
                "camera_regime": "image_derived_m3",
                "count_crosscheck": cross,
                "_left_model": left_model,
                "_right_model": right_model,
            }
        )
        log(f"[tape  ] {left_name[11:19]}->{right_name[11:19]} truths {len(truth):6d} "
            f"bestof {len(best):5d} correspondences {len(matches):5d}")
        log.checkpoint(f"tape {left_name}")

    np.savez_compressed(tape_npz_path, **tape_arrays)
    tape_npz_sha = digest(tape_npz_path)
    tape_json = {
        "schema": "ds23-blind-input-tape-1",
        "phase": "pre_selection_frozen_blind_inputs",
        "cell": BASE_CELL,
        "distance_blocks": BASE_DISTANCE_BLOCKS,
        "normal_degrees": BASE_NORMAL_DEGREES,
        "normal_cosine": normal_cosine,
        "render_scale": RENDER_SCALE,
        "camera_regime": "image_derived_m3",
        "tape_arrays": {
            "path": str(tape_npz_path.relative_to(ROOT)),
            "sha256": tape_npz_sha,
            "arrays": {"tp_matches_<slot>": "[N,4] pixel correspondences (lx, ly, rx, ry)"},
        },
        "pairs": [
            {key: value for key, value in pair.items() if not key.startswith("_")}
            for pair in tape_pairs
        ],
        "forbidden_in_tape": [
            "fixture manifests / reference poses / reference yaw classes",
            "litematics, scene transforms and rendered block labels",
            "frozen evaluator outputs or any reference score",
            "pair identity beyond the tape's own slot/evidence bookkeeping",
            "oracle truth edge sets and per-pair truth counts (count cross-check only, "
            "recorded after the tape's correspondence set is fixed)",
        ],
        "note": (
            "The decision path never receives the 'left'/'right'/'fixture'/'set' bookkeeping "
            "fields; every one of them is dropped by the DS-23 adapter and is absent from "
            "BlindPairInput, and 'oracle_truths_frozen_crosscheck_only' is a count of a "
            "reference label, never a decision input."
        ),
    }
    tape_json_path.write_text(
        json.dumps(tape_json, indent=2, default=plain), encoding="utf-8"
    )
    tape_json_sha = digest(tape_json_path)
    log(f"[tape  ] frozen tape written: {tape_json_path.name} {tape_json_sha[:16]}... "
        f"{tape_npz_path.name} {tape_npz_sha[:16]}...")

    # ---------------- Phase B: blind selection ----------------------------- #
    # The tape is now closed on disk. Phase B re-reads it and nothing else:
    # no manifest, no litematic, no evaluator, no reference class, no pair
    # identity enters the decision path. Decisions are frozen here, before any
    # reference label is read below.
    tape_json_read = json.loads(tape_json_path.read_text(encoding="utf-8"))
    tape_arrays_read = np.load(tape_npz_path)
    if digest(tape_npz_path) != tape_npz_sha:
        raise SystemExit("blocker: blind tape archive changed between write and read")
    tape_rows = tape_json_read["pairs"]
    if len(tape_rows) != PAIR_COUNT:
        raise SystemExit(
            f"blocker: tape carries {len(tape_rows)} pairs, expected {PAIR_COUNT}"
        )
    decisions: list[dict] = []
    trace_unions: list[list[str]] = []
    for pair in tape_rows:
        blind, matches = tape_blind_input(pair, tape_arrays_read)
        trace = start_trace(blind, blind.left_intrinsics, blind.right_intrinsics)
        decision = select_blind_pose(blind)
        trace = stop_trace()
        reads = tuple(sorted(set(trace.names())))
        trace_unions.append(list(reads))
        unexpected_reads = [
            name for name in reads
            if name not in GEOMETRIC_INPUT_WHITELIST
        ]
        decisions.append(
            {
                "slot": int(pair["slot"]),
                "solved": bool(decision.solved),
                "reason": decision.reason,
                "right_yaw_class": decision.right_yaw_class,
                "baseline_direction": None if decision.baseline_direction is None
                else [float(value) for value in decision.baseline_direction],
                "inliers": int(decision.inliers),
                "inlier_fraction": float(decision.inlier_fraction),
                "median_rho_degrees": decision.median_rho_degrees,
                "parallax_degrees": decision.parallax_degrees,
                "cheirality_fraction": decision.cheirality_fraction,
                "singular_ratio": decision.singular_ratio,
                "class_margin": decision.class_margin,
                "stability": decision.stability,
                "trace_reads": list(reads),
                "unexpected_reads": unexpected_reads,
                "tape_matches_sha256": pair["matches_sha256"],
                "per_class": {
                    key: {
                        "gate_pass": value["gate_pass"],
                        "gate_refusal": value["gate_refusal"],
                        "consensus_count": value["consensus_count"],
                        "consensus_fraction": value["consensus_fraction"],
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
        log(f"[blind ] {pair['left'][11:19]}->{pair['right'][11:19]} "
            f"n {len(matches):5d} class {decision.right_yaw_class} "
            f"inliers {decision.inliers:5d} reason "
            f"{decision.reason or 'SOLVED'} reads {len(reads)}")
        log.checkpoint(f"blind {pair['left']}")
    selection_finished = time.perf_counter()
    log(f"[blind ] selection frozen for all {len(decisions)} pairs in "
        f"{selection_finished - started_all:.1f}s; reference evaluation starts now")

    all_reads = sorted({name for reads in trace_unions for name in reads})
    unexpected = sorted(
        name for name in all_reads if name not in GEOMETRIC_INPUT_WHITELIST
    )

    # ---------------- Phase C: reference evaluation ------------------------ #
    import diagnose_match_attribution as frozen_evaluator

    # (a) DS-19 equivalence on the same geometry: run DS-19's own adjudicator
    #     in memory on the tape geometry and compare field-for-field with the
    #     clean-room decision that was already frozen.
    import diagnose_oracle_pose_failures as ds19_module

    equivalence: list[dict] = []
    for position, pair in enumerate(tape_rows):
        matches = np.asarray(
            tape_arrays_read[f"tp_matches_{position}"], dtype=np.float64
        )
        left_est, right_est = tape_cameras(pair)
        ds19_geometry = ds19_module.class_rows(left_est, right_est, matches)
        ds19_decision = ds19_module.adjudicate(
            ds19_geometry, left_est, right_est,
            matches[:, 0:2], matches[:, 2:4],
            np.arange(len(matches)), np.arange(len(matches)),
            int(pair["width"]), int(pair["height"]),
        )
        decision = decisions[position]
        class_equal = ds19_decision["right_yaw_class"] == decision["right_yaw_class"]
        solved_equal = bool(ds19_decision["solved"]) == decision["solved"]
        reason_equal = ds19_decision["reason"] == decision["reason"]
        if ds19_decision["baseline_direction"] is None:
            baseline_gap = 0.0 if decision["baseline_direction"] is None else float("inf")
        else:
            baseline_gap = float(
                np.max(
                    np.abs(
                        np.asarray(ds19_decision["baseline_direction"], dtype=np.float64)
                        - np.asarray(decision["baseline_direction"], dtype=np.float64)
                    )
                )
            )
        inliers_equal = int(ds19_decision["inliers"]) == decision["inliers"]
        equivalence.append(
            {
                "slot": position,
                "pair": f"{pair['left']}->{pair['right']}",
                "solved_equal": solved_equal,
                "reason_equal": reason_equal,
                "class_equal": class_equal,
                "inliers_equal": inliers_equal,
                "baseline_max_abs_diff": baseline_gap,
                "equivalent": bool(
                    solved_equal and reason_equal and class_equal and inliers_equal
                    and baseline_gap <= DECISION_TAU
                ),
            }
        )
    equivalence_ok = all(row["equivalent"] for row in equivalence)
    log(f"[equiv ] clean-room selector vs DS-19 adjudicator on the same geometry: "
        f"{sum(row['equivalent'] for row in equivalence)}/{len(equivalence)} identical "
        f"(<= {DECISION_TAU:g})")

    # Reference/truth models for scoring only (never visible to the selection):
    # taken from the phase-A manifest cache using the tape's bookkeeping names.
    pair_models = {
        position: (
            fixtures[pair["fixture"]]["cameras"][pair["left"]],
            fixtures[pair["fixture"]]["cameras"][pair["right"]],
        )
        for position, pair in enumerate(tape_rows)
    }

    # (b) frozen DS-11 baseline read-out (the 6/9 reproduction).
    baseline_rows: list[dict] = []
    for position, pair in enumerate(tape_rows):
        matches = np.asarray(
            tape_arrays_read[f"tp_matches_{position}"], dtype=np.float64
        )
        left_est, right_est = tape_cameras(pair)
        reference_class = ds19_pairs[f"{pair['left']}->{pair['right']}"]["reference_class"]
        pose = solve_pair(
            left_est, right_est, matches,
            left_id="left", right_id="right", gates=PairGates(),
        )
        record = frozen_evaluator._evaluate_pose(
            pose, pair_models[position][0], pair_models[position][1], reference_class
        )
        baseline_rows.append(record)
    baseline_solved = [row for row in baseline_rows if row["solved"]]
    baseline_class_correct = [row for row in baseline_rows if row["class_equals_expected"]]
    if len(baseline_solved) != BASELINE_SOLVES_REQUIRED:
        raise SystemExit(
            f"blocker: expected the frozen {BASELINE_SOLVES_REQUIRED}/9 DS-11 oracle solves, "
            f"measured {len(baseline_solved)}/9; refusing to report"
        )
    log(f"[basel ] reproduced the frozen DS-11 oracle baseline "
        f"({len(baseline_solved)}/9 solved, {len(baseline_class_correct)}/9 class-correct)")

    # (c) score each frozen blind decision with the frozen evaluator.
    evaluated: list[dict] = []
    usable_count = 0
    class_correct_count = 0
    for position, pair in enumerate(tape_rows):
        decision = decisions[position]
        reference_class = ds19_pairs[f"{pair['left']}->{pair['right']}"]["reference_class"]
        reference_floors = ds19_pairs[f"{pair['left']}->{pair['right']}"]["reference_class_floors"]
        if not decision["solved"]:
            evaluation = {
                "solved": False,
                "reason": decision["reason"],
                "class_equals_expected": False,
                "within_5deg": False,
                "rotation_error_degrees": None,
                "baseline_error_degrees_gauge_aligned": None,
                "baseline_error_degrees_signless": None,
                "gauge_index": None,
            }
        else:
            blind, _matches = tape_blind_input(pair, tape_arrays_read)
            handle = _DecisionPose(_DecisionView(decision), blind)
            evaluation = frozen_evaluator._evaluate_pose(
                handle, pair_models[position][0], pair_models[position][1], reference_class
            )
        usable = bool(
            evaluation["class_equals_expected"]
            and evaluation["within_5deg"]
            and evaluation["baseline_error_degrees_gauge_aligned"] is not None
            and evaluation["baseline_error_degrees_gauge_aligned"] <= 5.0
        )
        usable_count += int(usable)
        class_correct_count += int(bool(evaluation["class_equals_expected"]))
        ds19_row = ds19_pairs[f"{pair['left']}->{pair['right']}"]
        ds19_adjudicated = ds19_row["adjudicated"]
        agrees = ds19_adjudicated["right_yaw_class"] == decision["right_yaw_class"]
        if ds19_adjudicated["baseline_direction"] is None:
            baseline_gap = 0.0 if decision["baseline_direction"] is None else float("inf")
        else:
            baseline_gap = float(
                np.max(
                    np.abs(
                        np.asarray(ds19_adjudicated["baseline_direction"], dtype=np.float64)
                        - np.asarray(decision["baseline_direction"], dtype=np.float64)
                    )
                )
            )
        production_failure = ds19_row["attribution"]["production_failure_class"]
        evaluated.append(
            {
                "slot": position,
                "pair": f"{pair['left']}->{pair['right']}",
                "set": pair["set"],
                "fixture": pair["fixture"],
                "correspondences": pair["correspondences"],
                "reference_class": reference_class,
                "reference_class_floors": reference_floors,
                "evaluation": evaluation,
                "usable": usable,
                "baseline": baseline_rows[position],
                "ds19_artifact_class_agrees": bool(agrees),
                "ds19_artifact_baseline_max_abs_diff": baseline_gap,
                "production_failure_class": production_failure,
            }
        )

    original_six = [
        row["pair"] for row in evaluated
        if row["baseline"]["class_equals_expected"] and row["baseline"]["solved"]
    ]
    regressions = [
        row["pair"] for row in evaluated
        if row["pair"] in original_six and not row["usable"]
    ]
    synthetic = synthetic_controls(
        log,
        lambda decision, blind, case: _synthetic_evaluate(
            frozen_evaluator, decision, blind, case
        ),
    )

    threshold = {
        "reproduce_6_of_9_baseline": {
            "value": len(baseline_solved),
            "required": BASELINE_SOLVES_REQUIRED,
            "of": PAIR_COUNT,
            "class_correct": len(baseline_class_correct),
            "passed": len(baseline_solved) == BASELINE_SOLVES_REQUIRED,
        },
        "blind_selection_usable": {
            "value": usable_count,
            "required": PROMOTION_USABLE_REQUIRED,
            "of": PAIR_COUNT,
            "passed": usable_count >= PROMOTION_USABLE_REQUIRED,
        },
        "blind_selection_yaw_class_correct": {
            "value": class_correct_count,
            "required": PROMOTION_USABLE_REQUIRED,
            "of": PAIR_COUNT,
            "passed": class_correct_count >= PROMOTION_USABLE_REQUIRED,
        },
        "blind_selection_gauge_baseline_within_5deg": {
            "value": sum(
                1 for row in evaluated
                if row["usable"]
                and row["evaluation"]["baseline_error_degrees_gauge_aligned"] <= 5.0
            ),
            "required": PROMOTION_USABLE_REQUIRED,
            "of": PAIR_COUNT,
            "passed": usable_count >= PROMOTION_USABLE_REQUIRED,
        },
        "zero_regressions_among_original_six": {
            "value": len(regressions),
            "required": 0,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        },
        "existing_synthetic_degeneracy_refusals": {
            "value": synthetic["all_controls_ok"],
            "required": True,
            "detail": {
                row["kind"]: {
                    "expectation": row["expectation"],
                    "usable": row["usable"],
                    "refused": row["refused"],
                    "refusals": row["refusals"],
                    "control_ok": row["control_ok"],
                }
                for row in synthetic["families"]
            },
            "passed": bool(synthetic["all_controls_ok"]),
        },
        "apiboundary_no_unexpected_reads": {
            "value": len(unexpected),
            "required": 0,
            "unexpected": unexpected,
            "passed": len(unexpected) == 0,
        },
        "ds19_equivalence": {
            "value": sum(row["equivalent"] for row in equivalence),
            "required": PAIR_COUNT,
            "of": PAIR_COUNT,
            "passed": bool(equivalence_ok),
        },
    }
    promotion_keys = (
        "reproduce_6_of_9_baseline",
        "blind_selection_usable",
        "blind_selection_yaw_class_correct",
        "blind_selection_gauge_baseline_within_5deg",
        "zero_regressions_among_original_six",
        "existing_synthetic_degeneracy_refusals",
        "apiboundary_no_unexpected_reads",
    )
    threshold["all_passed"] = all(threshold[key]["passed"] for key in promotion_keys)
    threshold["status"] = (
        "blind_policy_meets_preregistered_threshold"
        if threshold["all_passed"]
        else "blind_policy_fails_preregistered_threshold"
    )
    ds19_survives = bool(
        usable_count >= PROMOTION_USABLE_REQUIRED
        and class_correct_count >= PROMOTION_USABLE_REQUIRED
        and not regressions
        and synthetic["all_controls_ok"]
    )
    log(f"[thresh] {threshold['status']}: baseline {len(baseline_solved)}/9, "
        f"blind usable {usable_count}/9, class-correct {class_correct_count}/9, "
        f"regressions {len(regressions)}, DS-19 9/9 survives: {ds19_survives}")

    # ---------------- artifacts -------------------------------------------- #
    csv_rows = []
    for row in evaluated:
        decision = decisions[row["slot"]]
        evaluation = row["evaluation"]
        baseline = row["baseline"]
        csv_rows.append(
            {
                "pair": row["pair"],
                "set": row["set"],
                "correspondences": row["correspondences"],
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
                "reference_class": row["reference_class"],
                "reference_class_floors": row["reference_class_floors"],
                "evaluated_class_correct": evaluation["class_equals_expected"],
                "evaluated_within_5deg": evaluation["within_5deg"],
                "evaluated_rotation_error_degrees": evaluation["rotation_error_degrees"],
                "evaluated_baseline_error_degrees_gauge_aligned":
                    evaluation["baseline_error_degrees_gauge_aligned"],
                "evaluated_baseline_error_degrees_signless":
                    evaluation["baseline_error_degrees_signless"],
                "evaluated_gauge_index": evaluation["gauge_index"],
                "usable": row["usable"],
                "production_bestof_solved": baseline["solved"],
                "production_bestof_class": baseline["right_yaw_class"],
                "production_bestof_class_correct": baseline["class_equals_expected"],
                "production_bestof_usable": bool(
                    baseline["class_equals_expected"] and baseline["within_5deg"]
                    and baseline["baseline_error_degrees_gauge_aligned"] is not None
                    and baseline["baseline_error_degrees_gauge_aligned"] <= 5.0
                ),
                "ds19_artifact_class_agrees": row["ds19_artifact_class_agrees"],
                "ds19_artifact_baseline_max_abs_diff":
                    row["ds19_artifact_baseline_max_abs_diff"],
                "production_failure_class": row["production_failure_class"],
            }
        )
    csv_path = out_dir / "blind_pose_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    runtime = time.perf_counter() - started_all
    payload = {
        "schema": "ds23-blind-pose-policy-1",
        "phase": "label_blind_pose_policy_audit",
        "status": "complete",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "directive": (
            "AGENT_COMMS.md 2026-09-12 13:38 CDT (DS-23): prove the support tie-break is "
            "label-blind before any production edit; frozen 9-pair + synthetic degeneracy "
            "controls; report >=8/9 usable, <=5 deg gauge baseline, zero regressions"
        ),
        "boundary": {
            "selector": "scripts/blind_pose_policy.select_blind_pose",
            "selector_inputs": sorted(GEOMETRIC_INPUT_WHITELIST),
            "forbidden_before_selection": list(FORBIDDEN_DECISION_INPUTS),
            "frozen_policy": {
                "edge_consensus_degrees": FROZEN_POLICY.edge_consensus_degrees,
                "edge_consensus_min": FROZEN_POLICY.edge_consensus_min,
                "edge_consensus_fraction": FROZEN_POLICY.edge_consensus_fraction,
                "edge_parallax_min_degrees": FROZEN_POLICY.edge_parallax_min_degrees,
                "edge_cheirality_min": FROZEN_POLICY.edge_cheirality_min,
                "edge_rank_min_ratio": FROZEN_POLICY.edge_rank_min_ratio,
                "class_margin_min": FROZEN_POLICY.class_margin_min,
                "class_margin_fraction": FROZEN_POLICY.class_margin_fraction,
                "stability_degrees": FROZEN_POLICY.stability_degrees,
                "stratum_min_edges": FROZEN_POLICY.stratum_min_edges,
                "spatial_grid": list(FROZEN_POLICY.spatial_grid),
            },
            "decision_inputs_per_pair": [
                "left/right camera intrinsics (image-derived M3)",
                "left/right camera orientation (image-derived M3)",
                "the frozen oracle best-of pixel correspondences",
                "image size",
            ],
            "tape": {
                "json": str(tape_json_path.relative_to(ROOT)),
                "json_sha256": tape_json_sha,
                "npz": str(tape_npz_path.relative_to(ROOT)),
                "npz_sha256": tape_npz_sha,
            },
            "trace_reads_union": all_reads,
            "trace_unexpected_reads": unexpected,
            "evaluator": {
                "module": "scripts/diagnose_match_attribution.py",
                "function": "_evaluate_pose",
                "frozen": True,
                "used_after_selection_only": True,
            },
        },
        "equivalence_with_ds19": {
            "note": (
                "scripts/diagnose_oracle_pose_failures.py is unedited; its own adjudicator is "
                "executed in memory on the same geometry and compared field-for-field with the "
                "clean-room selector before any reference label is read"
            ),
            "per_pair": equivalence,
            "all_equivalent": bool(equivalence_ok),
        },
        "reproduction": {
            "production_oracle_bestof_solved": len(baseline_solved),
            "production_oracle_bestof_class_correct": len(baseline_class_correct),
            "frozen_ds19_artifact_reported_solved":
                frozen_ds19["reproduction"]["production_oracle_bestof_solved"],
            "frozen_ds19_artifact_reported_class_correct":
                frozen_ds19["reproduction"]["production_oracle_bestof_class_correct"],
        },
        "blind_selection": {
            "usable": usable_count,
            "class_correct": class_correct_count,
            "of": PAIR_COUNT,
            "ds19_9_of_9_survives": ds19_survives,
            "notes": (
                "the selector's decision for each pair is frozen in 'decisions' before any "
                "reference label is read; 'evaluated' is the frozen evaluator's post-selection "
                "read-out and was never fed back into selection"
            ),
        },
        "threshold": threshold,
        "synthetic_controls": synthetic,
        "decisions": decisions,
        "evaluated": [
            {
                key: value for key, value in row.items()
                if key not in ("baseline",)
            }
            for row in evaluated
        ],
        "baseline_readout": [
            {
                "pair": evaluated[position]["pair"],
                "solved": baseline_rows[position]["solved"],
                "reason": baseline_rows[position]["reason"],
                "right_yaw_class": baseline_rows[position]["right_yaw_class"],
                "class_equals_expected": baseline_rows[position]["class_equals_expected"],
                "within_5deg": baseline_rows[position]["within_5deg"],
                "baseline_error_degrees_gauge_aligned":
                    baseline_rows[position]["baseline_error_degrees_gauge_aligned"],
            }
            for position in range(len(evaluated))
        ],
        "alignment": alignment,
        "alignment_median_px": alignment_median,
        "frozen_inputs": digests,
        "reference_inputs": {
            fixture: {
                "litematic": record["litematic"],
                "litematic_sha256": record["litematic_sha256"],
                "manifest": record["manifest"],
                "manifest_sha256": record["manifest_sha256"],
                "scene": record["scene"],
                "scene_sha256": record["scene_sha256"],
            }
            for fixture, record in fixtures.items()
        },
        "runtime_seconds": runtime,
        "selection_seconds": selection_finished - started_all,
        "peak_rss_bytes": log.peak_rss_bytes,
        "artifacts": {},
    }
    payload["artifacts"]["blind_tape.json"] = {
        "bytes": tape_json_path.stat().st_size, "sha256": tape_json_sha,
    }
    payload["artifacts"]["blind_tape.npz"] = {
        "bytes": tape_npz_path.stat().st_size, "sha256": tape_npz_sha,
    }
    payload["artifacts"]["blind_pose_pairs.csv"] = {
        "bytes": csv_path.stat().st_size, "sha256": digest(csv_path),
    }
    decision_json = out_dir / "blind_pose_policy.json"
    decision_json.write_text(
        json.dumps(payload, indent=2, default=plain), encoding="utf-8"
    )
    payload["artifacts"]["blind_pose_policy.json"] = {
        "bytes": decision_json.stat().st_size, "sha256": digest(decision_json),
    }
    payload["artifacts"]["blind_pose_run.log"] = {
        "note": "log is still open while the payload is written; digest recorded in the report",
    }
    return payload


class _DecisionView:
    """Minimal ``BlindDecision``-shaped view for the frozen evaluator handle."""

    def __init__(self, record: dict) -> None:
        self.solved = record["solved"]
        self.reason = record["reason"]
        self.right_yaw_class = record["right_yaw_class"]
        self.baseline_direction = record["baseline_direction"]
        self.inliers = record["inliers"]
        self.inlier_fraction = record["inlier_fraction"]
        self.median_rho_degrees = record["median_rho_degrees"]
        self.parallax_degrees = record["parallax_degrees"]
        self.cheirality_fraction = record["cheirality_fraction"]


def _synthetic_evaluate(frozen_evaluator, decision, blind, case) -> bool:
    handle = _DecisionPose(decision, blind)
    evaluation = frozen_evaluator._evaluate_pose(
        handle, case["left_model"], case["right_model"], 0
    )
    return bool(
        evaluation["class_equals_expected"]
        and evaluation["within_5deg"]
        and evaluation["baseline_error_degrees_gauge_aligned"] is not None
        and evaluation["baseline_error_degrees_gauge_aligned"] <= 5.0
    )


def digest_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    arguments = parser.parse_args(argv)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(out_dir / "blind_pose_run.log")
    try:
        payload = run(out_dir, log)
    except MemoryBudgetExceeded as error:
        log(f"[abort ] {error}")
        log.close()
        return 4
    log(f"[done  ] baseline {payload['reproduction']['production_oracle_bestof_solved']}/9; "
        f"blind usable {payload['blind_selection']['usable']}/9; "
        f"threshold {payload['threshold']['status']}; "
        f"runtime {payload['runtime_seconds']:.1f}s; peak RSS "
        f"{(payload['peak_rss_bytes'] or 0) / 2**20:.1f} MiB")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
