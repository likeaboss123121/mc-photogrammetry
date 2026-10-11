"""DS-19 - isolate the DS-16B oracle-edge pose-adjudication failures.

**Evaluation-only and reference-assisted.** Every label used here comes from the
frozen fixture manifests/litematics and must never enter a production path. This
script reads, and never re-detects or re-matches:

* ``exports/diagnostics/ds16/match_candidates.{json,npz}`` - the blind DS-16
  phase-1 freeze (keypoints, descriptors, reciprocal candidates, DS-14
  selections), digest-verified before anything else happens;
* ``exports/diagnostics/ds16/match_attribution.json`` - DS-16's authoritative
  reference attribution (cross-check only);
* ``exports/diagnostics/ds16b/match_attribution_sweep.json`` - DS-16B's frozen
  per-cell oracle read-out (cross-check only).

Scope (DS-19 directive, ``AGENT_COMMS.md`` 2026-09-12 13:04 CDT):

1. Reproduce the 6/9 DS-11 oracle solves on the exact nine T2/T6 pairs, under
   both current (M3 image-derived) and reference (manifest) cameras, on the
   *frozen* DS-16B oracle truth correspondences at the frozen base cell
   (distance <= 0.5 blocks, face-normal cos >= cos 30 deg, render scale 0.35).
2. Attribute every failure to one of: yaw-class scoring, baseline null-space
   conditioning, cheirality, parallax, correspondence geometry, evaluator
   ambiguity.
3. Test **one** physically justified adjudicator against the same frozen oracle
   edges. It changes exactly one thing - how the baseline direction and the
   winning yaw class are selected - and leaves every DS-11 gate degree
   unchanged. Preregistered candidate threshold (fixed before the run): >= 8/9
   usable poses with the yaw class correct and gauge-aligned baseline error
   <= 5 deg, zero regressions among the original six solves, plus refusal on
   synthetic degenerate configurations.

No descriptor, production, test, DS-11..DS-18 source, artifact or threshold is
touched. Run once with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts\\diagnose_oracle_pose_failures.py

Correction (``AGENT_COMMS.md`` 2026-09-12 13:16 CDT), applied after the lead
interrupt: the DS-16B reference yaw-class and gauge-aligned baseline-error
evaluator is frozen. This script does **not** score the adjudicated pose with
any local formula - it hands the adjudicated pose to
``diagnose_match_attribution._evaluate_pose`` (the exact function DS-16/DS-16B
used) and copies its read-out. When the floor heuristic disagrees with the
matrix-based reference class the pair is reported as evaluator ambiguity; the
disagreement is never promoted into an accepted alternate label. The only
addition is the preregistered ``usable`` conjunction, built from the frozen
evaluator's own ``class_equals_expected``, ``within_5deg`` and
``baseline_error_degrees_gauge_aligned`` <= 5 deg.
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
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    camera_rotation_matrix,
    load_camera_manifest,
)
from mc_photogrammetry.reconstruction.scene import load_scene_transform  # noqa: E402
from mc_photogrammetry.vision.gui_mask import create_gui_mask  # noqa: E402

from diagnose_connected_pairs import (  # noqa: E402
    _reference_class,
    digest,
    plain,
    source_revision,
    to_camera_estimate,
)
from diagnose_joint_pairs import _estimate_from_record  # noqa: E402
import diagnose_match_attribution as _frozen_evaluator  # noqa: E402
from diagnose_match_attribution import (  # noqa: E402
    ALIGNMENT_MEDIAN_PX_MAX,
    REFERENCE_RENDER_SCALE,
    _camera_estimate,
    _matches_from_edges,
    label_keypoints,
    load_frozen,
)

DS16_DIR = ROOT / "exports" / "diagnostics" / "ds16"
DS16B_DIR = ROOT / "exports" / "diagnostics" / "ds16b"
OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds19"

# Frozen DS-16 digests (DS-16 report section 1). Any mismatch aborts before a
# single label is produced.
EXPECTED_CANDIDATES_JSON_SHA256 = (
    "e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93"
)
EXPECTED_CANDIDATES_NPZ_SHA256 = (
    "3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4"
)
EXPECTED_ATTRIBUTION_JSON_SHA256 = (
    "72da240a1992b31640ddbfb4e4bd0b6fb6c1b02e4ede1abd6a7507b74ea7a038"
)
# Frozen DS-16B authoritative sweep (DS-16B report section 6), cross-check only.
EXPECTED_SWEEP_JSON_SHA256 = (
    "397294502f902f84d54abee047bc1a52939f843fd8016147577e5020289f1cf0"
)

# The frozen DS-16B base cell that produced the 6/9 read-out.
BASE_DISTANCE_BLOCKS = 0.5
BASE_NORMAL_DEGREES = 30.0
RENDER_SCALE = REFERENCE_RENDER_SCALE  # 0.35
SELECTED_SETS = ("T2", "T6")
PAIR_COUNT = 9
BASE_CELL = "d0.5_n30"

# Boundedness, inherited from DS-16BR: DS-11's full-matrix SVD is 8*N**2 bytes.
SOLVER_ROW_CAP = 4096
TRUTH_CHUNK = 64
TRUTH_MATERIALIZE_MAX_ROWS = 5_000_000
RSS_LIMIT_BYTES = 1_500 * 1024 * 1024
OFFLINE_CAP = 2_000_000  # offline Gram cap: DSM is 3x3 regardless of N

# Preregistered adjudicator constants, fixed here before the single run.
EDGE_CONSENSUS_DEGREES = 1.0          # equals PairGates.inlier_residual_degrees
EDGE_CONSENSUS_MIN = 8                # equals PairGates.min_inliers
EDGE_CONSENSUS_FRACTION = 0.5         # equals PairGates.min_inlier_fraction
EDGE_PARALLAX_MIN_DEGREES = 3.0       # equals PairGates.min_parallax_degrees
EDGE_CHEIRALITY_MIN = 0.75            # equals PairGates.min_cheirality_fraction
EDGE_RANK_MIN_RATIO = 1e-3            # equals PairGates.min_singular_ratio
CLASS_MARGIN_MIN = 3                  # consensus-edge margin over the runner-up
CLASS_MARGIN_FRACTION = 0.25          # ... or this fraction of the winner
STABILITY_DEGREES = 5.0               # equals the evaluation "usable pose" bound
STRATUM_MIN_EDGES = 8                 # a stratum is only trustworthy at this size
SPATIAL_GRID = (4, 4)
GAUGE_COMPARABLE_DEGREES = 2.0   # adjudicated vs frozen DS-11 rotation-error gap bound
BOOTSTRAP_REPEATS = 200
BOOTSTRAP_SEED = 0
BOOTSTRAP_CLASS_TOLERANCE = 0.05      # 5% of resamples may disagree
SYNTHETIC_REPEATS = 24                # per degenerate family
SYNTHETIC_SEED = 19

DIAGNOSTIC_LABELS = (
    "yaw_class_scoring",
    "baseline_null_space_conditioning",
    "cheirality",
    "parallax",
    "correspondence_geometry",
    "evaluator_ambiguity",
    "adjudicated_solved",
    "unknown",
)

PAIR_CSV_COLUMNS = (
    "pair", "set", "fixture", "left", "right",
    "reference_class", "reference_class_floors", "oracle_truths", "oracle_bestof_edges",
    "production_bestof_current_solved", "production_bestof_current_reason",
    "production_bestof_current_class", "production_bestof_current_within_5deg",
    "production_bestof_current_rotation_error_degrees",
    "production_bestof_current_baseline_error_degrees",
    "production_bestof_reference_solved", "production_bestof_reference_reason",
    "production_bestof_reference_class",
    "production_full_current_solved", "production_full_current_reason",
    "production_full_reference_solved", "production_full_reference_reason",
    "adjudicated_solved", "adjudicated_reason", "adjudicated_class",
    "adjudicated_class_correct", "adjudicated_within_5deg",
    "adjudicated_rotation_error_degrees", "adjudicated_baseline_error_degrees",
    "adjudicated_baseline_error_signless_degrees", "adjudicated_gauge_index",
    "adjudicated_inliers", "adjudicated_inlier_fraction",
    "adjudicated_median_rho_degrees", "adjudicated_parallax_degrees",
    "adjudicated_cheirality_fraction", "adjudicated_singular_ratio",
    "diagnosis", "production_failure_class", "diagnosis_detail",
)


class MemoryBudgetExceeded(RuntimeError):
    """The run cannot stay inside the preregistered RSS budget."""


class _Log:
    """Line-buffered stdout + ``oracle_pose_run.log`` tee with an RSS watchdog."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8")
        self.peak_rss_bytes: int | None = None

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def checkpoint(self, label: str) -> None:
        current, peak = _rss_bytes()
        if peak is None:
            return
        if self.peak_rss_bytes is None or peak > self.peak_rss_bytes:
            self.peak_rss_bytes = peak
        if peak > RSS_LIMIT_BYTES:
            self(
                f"[abort ] peak RSS {peak / 2**20:.1f} MiB exceeds the "
                f"{RSS_LIMIT_BYTES / 2**20:.0f} MiB budget at {label}; stopping"
            )
            raise MemoryBudgetExceeded(f"peak RSS {peak / 2**20:.1f} MiB at {label}")

    def close(self) -> None:
        self._handle.close()


def _make_rss_probe():
    """OS working-set probe: exact peak, no third-party dependency."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

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

        def probe() -> tuple[int | None, int | None]:
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None, None
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

        return probe

    def probe() -> tuple[int | None, int | None]:
        try:
            fields = {
                line.split(":", 1)[0]: line.split(":", 1)[1]
                for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
                if ":" in line
            }
            return (
                int(fields["VmRSS"].split()[0]) * 1024,
                int(fields["VmHWM"].split()[0]) * 1024,
            )
        except Exception:
            return None, None

    return probe


_RSS_PROBE = _make_rss_probe()


def _rss_bytes() -> tuple[int | None, int | None]:
    return _RSS_PROBE()


def _mib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 2**20:.1f} MiB"


def _normal_cosine(degrees: float) -> float:
    return float(np.cos(np.radians(degrees)))


# --------------------------------------------------------------------------- #
# Frozen oracle truth correspondence reconstruction (copied from DS-16B, with
# an explicit cross-check against DS-16's own many-to-many oracle counts).
# --------------------------------------------------------------------------- #
def oracle_chunks(left: dict, right: dict, tolerance: float, normal_cosine: float,
                  chunk: int = TRUTH_CHUNK):
    """Yield ``(lo, hi, edges)``: unique oracle pairs of one left-keypoint chunk.

    Identical predicate and boundedness to DS-16B's ``truth_chunks`` (DS-16's
    ``_truth_edges``, chunk-partitioned and de-duplicated per chunk). Copied
    rather than imported so this diagnostic's truth label is auditable on its
    own; the frozen-count cross-check below verifies it reproduces DS-16.
    """
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
        return (
            (block[:, 0] - minimum[0]) * dims[1] + (block[:, 1] - minimum[1])
        ) * dims[2] + (block[:, 2] - minimum[2])

    right_key = encode(right_cell)
    order = np.argsort(right_key, kind="stable")
    right_key_sorted = right_key[order]
    shifts = [
        (dx * dims[1] + dy) * dims[2] + dz
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for start in range(0, len(left_index), chunk):
        world = left_world[start : start + chunk]
        normal = left_normal[start : start + chunk]
        origin = left_index[start : start + chunk]
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
        del codes_parts
        yield int(origin[0]), int(origin[-1]), np.column_stack(
            (codes // right_count, codes % right_count)
        )


def oracle_edges(left: dict, right: dict, tolerance: float, normal_cosine: float) -> np.ndarray:
    """Complete unique oracle truth set, assembled from bounded chunks."""
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


def best_of_edges(truth: np.ndarray, left_world: np.ndarray, right_world: np.ndarray) -> np.ndarray:
    """One oracle edge per left keypoint: nearest world hit, index tie-break."""
    if len(truth) == 0:
        return np.empty((0, 2), dtype=np.int64)
    distance = np.linalg.norm(left_world[truth[:, 0]] - right_world[truth[:, 1]], axis=1)
    order = np.lexsort((truth[:, 1], distance, truth[:, 0]))
    ordered = truth[order]
    first = np.ones(len(ordered), dtype=bool)
    first[1:] = ordered[1:, 0] != ordered[:-1, 0]
    return ordered[first]


def _bounded_rows(edges: np.ndarray, cap: int = SOLVER_ROW_CAP) -> tuple[np.ndarray, bool]:
    """Deterministic stride bound on a DS-11 input; ``(rows, sampled)``."""
    rows = int(len(edges))
    if rows <= cap:
        return edges, False
    step = (rows + cap - 1) // cap
    return edges[::step], True


def _encode_codes(edges: np.ndarray, right_count: int) -> np.ndarray:
    return edges[:, 0].astype(np.int64) * int(max(right_count, 1)) + edges[:, 1].astype(np.int64)


# --------------------------------------------------------------------------- #
# Offline geometry: bounded, production-source-free reimplementation of the
# DS-11 adjudication maths. Values are verified against `solve_pair` output.
# --------------------------------------------------------------------------- #
def _rays(pixels: np.ndarray, intrinsics) -> np.ndarray:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (pixels[:, 0] - centre_x) / focal_x
    y = -(pixels[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (yaw_class % 4))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _residual_degrees(rows: np.ndarray, direction: np.ndarray) -> np.ndarray:
    projections = np.abs(rows @ direction)
    norms = np.linalg.norm(rows, axis=1)
    values = np.ones(len(rows), dtype=np.float64)
    safe = norms > 1e-9
    values[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    return np.degrees(np.arcsin(values))


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    if rows.size == 0:
        return None
    _u, _s, vt = np.linalg.svd(rows, full_matrices=False)
    vector = vt[-1]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _singular_values(rows: np.ndarray) -> tuple[float, float, float]:
    if rows.size == 0:
        return (0.0, 0.0, 0.0)
    _u, singular, _vt = np.linalg.svd(rows, full_matrices=False)
    padded = np.zeros(3, dtype=np.float64)
    padded[: min(3, len(singular))] = singular[:3]
    return (float(padded[0]), float(padded[1]), float(padded[2]))


def _positive_depth_fraction(world_left: np.ndarray, world_right: np.ndarray,
                             baseline: np.ndarray) -> float:
    if len(world_left) == 0:
        return 0.0
    cosine = np.clip(np.sum(world_left * world_right, axis=1), -1.0, 1.0)
    determinant = 1.0 - cosine * cosine
    safe = determinant > 1e-12
    if not np.any(safe):
        return 0.0
    first = world_left @ baseline
    second = -(world_right @ baseline)
    lambda_left = (first[safe] + cosine[safe] * second[safe]) / determinant[safe]
    lambda_right = (second[safe] + cosine[safe] * first[safe]) / determinant[safe]
    positive = int(np.count_nonzero((lambda_left > 0.0) & (lambda_right > 0.0)))
    return positive / float(len(world_left))


def _median_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dots = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(dots))))


def class_rows(left_camera, right_camera, matches: np.ndarray) -> dict:
    """Per-yaw-class ``a = d_l x d_r`` rows for a fixed correspondence set."""
    rotation_left = camera_rotation_matrix(to_camera_model_from_estimate(left_camera))
    rotation_right = camera_rotation_matrix(to_camera_model_from_estimate(right_camera))
    rays_left = _rays(matches[:, 0:2], left_camera.intrinsics)
    rays_right = _rays(matches[:, 2:4], right_camera.intrinsics)
    world_left = rays_left @ rotation_left
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for yaw_class in (0, 1, 2, 3):
        rotation_right_class = rotation_right @ _ry90(yaw_class)
        relative = rotation_right_class @ rotation_left.T
        left_in_right = rays_left @ relative.T
        rows[yaw_class] = (np.cross(left_in_right, rays_right), rays_right @ rotation_right_class)
    return {"rays_left": rays_left, "world_left": world_left, "rows": rows}


def to_camera_model_from_estimate(camera):
    """Manifest-free `CameraModel` for the public rotation API (position unused)."""
    from mc_photogrammetry.reconstruction.camera import CameraModel

    pose = camera.pose
    return CameraModel(
        position=(0.0, 0.0, 0.0),
        yaw_degrees=float(pose.yaw_degrees),
        pitch_degrees=float(pose.pitch_degrees),
        roll_degrees=float(pose.roll_degrees),
        eye_height=0.0,
    )


def fit_class(rows: np.ndarray, consensus_degrees: float = EDGE_CONSENSUS_DEGREES) -> dict:
    """One yaw class: bounded Gram/SVD consensus fit, no RANSAC, no sampling.

    The direction is the total-least-squares null vector of the *recomputed*
    consensus rows (bounded 3x3 Gram eigen-decomposition), never a subsample.
    """
    norms = np.linalg.norm(rows, axis=1)
    informative = norms > 1e-9
    informative_count = int(np.count_nonzero(informative))
    if informative_count:
        block = rows[informative]
        direction = _null_vector(block)
    else:
        direction = None
    fit: dict = {
        "informative_count": informative_count,
        "row_max": float(norms.max()) if norms.size else 0.0,
        "direction": None,
        "consensus": np.zeros(0, dtype=np.int64),
        "consensus_count": 0,
        "consensus_fraction": 0.0,
        "median_rho_degrees": None,
        "median_all_residual_degrees": None,
        "singular_values": None,
        "singular_ratio": None,
        "row_scale_min": None,
        "row_scale_max": None,
        "baseline": None,
        "parallax_degrees": None,
        "cheirality_fraction": None,
    }
    if direction is None:
        return fit
    residual = _residual_degrees(rows, direction)
    consensus = np.flatnonzero(residual <= consensus_degrees)
    for _ in range(10):
        if len(consensus) == 0:
            break
        candidate = _null_vector(rows[consensus])
        if candidate is None:
            break
        refined = np.flatnonzero(_residual_degrees(rows, candidate) <= consensus_degrees)
        converged = len(refined) == len(consensus) and np.array_equal(refined, consensus)
        direction, consensus = candidate, refined
        if converged:
            break
    if len(consensus) == 0:
        return fit
    # Bounded Gram/SVD refit on the consensus subset only. The smallest
    # eigenvector is the direction; it is only discarded when it is not finite.
    singular_values = _singular_values(rows[consensus])
    gram = rows[consensus].T @ rows[consensus]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if not np.all(np.isfinite(eigenvectors[:, 0])):
        return fit
    candidate = eigenvectors[:, 0]
    reference = _null_vector(rows[consensus])
    if reference is not None and float(np.dot(candidate, reference)) < 0.0:
        candidate = -candidate
    direction = candidate
    fit["direction"] = direction
    fit["consensus"] = consensus
    fit["consensus_count"] = int(len(consensus))
    fit["consensus_fraction"] = len(consensus) / max(informative_count, 1)
    fit["median_rho_degrees"] = float(np.median(residual[consensus]))
    fit["median_all_residual_degrees"] = (
        float(np.median(residual[informative])) if informative_count else None
    )
    fit["singular_values"] = singular_values
    fit["singular_ratio"] = (
        singular_values[1] / singular_values[0] if singular_values[0] > 1e-12 else 0.0
    )
    scales = np.linalg.norm(rows[informative], axis=1)
    fit["row_scale_min"] = float(scales.min()) if len(scales) else None
    fit["row_scale_max"] = float(scales.max()) if len(scales) else None
    return fit


def _spatial_bins(points: np.ndarray, width: int, height: int,
                  grid: tuple[int, int] = SPATIAL_GRID) -> np.ndarray:
    """Bin index per keypoint on a fixed image grid (the DS-16 grid)."""
    columns, rows = grid
    x = np.clip((points[:, 0] / max(width, 1) * columns).astype(np.int64), 0, columns - 1)
    y = np.clip((points[:, 1] / max(height, 1) * rows).astype(np.int64), 0, rows - 1)
    return y * columns + x


def stratify(inliers: np.ndarray, left_pixels: np.ndarray, right_pixels: np.ndarray,
             width: int, height: int,
             grid: tuple[int, int] = SPATIAL_GRID) -> list[tuple[int, np.ndarray]]:
    """Split a correspondence set by left/right spatial bin intersection.

    ``left_pixels`` and ``right_pixels`` are per-correspondence image points and
    ``inliers`` are positional indices into them.
    """
    if len(inliers) == 0:
        return []
    left_bins = _spatial_bins(left_pixels[inliers], width, height, grid)
    right_bins = _spatial_bins(right_pixels[inliers], width, height, grid)
    key = left_bins * (grid[0] * grid[1]) + right_bins
    strata: list[tuple[int, np.ndarray]] = []
    for value in np.unique(key):
        rows = inliers[key == value]
        if len(rows) >= 2:
            strata.append((int(value), rows))
    return strata


def stability_degrees(rows: np.ndarray, inliers: np.ndarray,
                      left_pixels: np.ndarray, right_pixels: np.ndarray,
                      width: int, height: int, minimum: int = STRATUM_MIN_EDGES) -> dict:
    """Largest direction disagreement between trustworthy spatial strata.

    A baseline that is only determined by one spatial band is ill-conditioned:
    refits on disjoint image regions disagree. Deterministic and bounded.
    """
    strata = stratify(inliers, left_pixels, right_pixels, width, height)
    usable = [(key, rows_) for key, rows_ in strata if len(rows_) >= minimum]
    if len(usable) < 2:
        return {
            "strata": len(strata),
            "usable_strata": len(usable),
            "pairwise_max_degrees": None,
            "available": False,
            "disagreeing_pairs": 0,
            "pairs": 0,
        }
    directions: list[np.ndarray] = []
    for _key, subset in usable:
        vector = _null_vector(rows[subset])
        if vector is None:
            continue
        if directions and float(np.dot(vector, directions[0])) < 0.0:
            vector = -vector
        directions.append(vector)
    if len(directions) < 2:
        return {
            "strata": len(strata),
            "usable_strata": len(usable),
            "pairwise_max_degrees": None,
            "available": False,
            "disagreeing_pairs": 0,
            "pairs": 0,
        }
    worst = 0.0
    disagreeing = 0
    pairs = 0
    for first in range(len(directions)):
        for second in range(first + 1, len(directions)):
            value = float(
                np.degrees(
                    np.arccos(float(np.clip(np.dot(directions[first], directions[second]), -1.0, 1.0)))
                )
            )
            worst = max(worst, value)
            pairs += 1
            if value > STABILITY_DEGREES:
                disagreeing += 1
    return {
        "strata": len(strata),
        "usable_strata": len(usable),
        "pairwise_max_degrees": worst,
        "available": True,
        "disagreeing_pairs": disagreeing,
        "pairs": pairs,
    }


def choose_sign(rows_direction: np.ndarray, rotation_right_class: np.ndarray,
                world_left: np.ndarray, world_right: np.ndarray,
                inliers: np.ndarray) -> dict:
    """Pick the baseline sign by depth positivity; report both fractions."""
    candidate = -(rotation_right_class.T @ rows_direction)
    positive = _positive_depth_fraction(world_left[inliers], world_right[inliers], candidate)
    negative = _positive_depth_fraction(world_left[inliers], world_right[inliers], -candidate)
    if negative > positive:
        return {"baseline": -candidate, "cheirality_fraction": negative,
                "positive": positive, "negative": negative}
    return {"baseline": candidate, "cheirality_fraction": positive,
            "positive": positive, "negative": negative}


def class_hypothesis(geometry: dict, yaw_class: int, left_camera, right_camera,
                     left_points: np.ndarray, right_points: np.ndarray,
                     left_edges: np.ndarray, right_edges: np.ndarray,
                     width: int, height: int) -> dict:
    """One complete offline yaw-class hypothesis, bounded and deterministic.

    ``left_points`` and ``right_points`` here are the per-correspondence pixel
    arrays (``left_edges``/``right_edges`` are retained for interface parity with
    the DS-11 edge sets but are not needed by the geometry).
    """
    rows, world_right = geometry["rows"][yaw_class]
    fit = fit_class(rows)
    record: dict = {
        "yaw_class": yaw_class,
        "informative_count": fit["informative_count"],
        "row_max": fit["row_max"],
        "row_scale_min": fit["row_scale_min"],
        "row_scale_max": fit["row_scale_max"],
        "consensus_count": fit["consensus_count"],
        "consensus_fraction": fit["consensus_fraction"],
        "median_rho_degrees": fit["median_rho_degrees"],
        "median_all_residual_degrees": fit["median_all_residual_degrees"],
        "singular_values": fit["singular_values"],
        "singular_ratio": fit["singular_ratio"],
        "consensus_singular_values": None,
        "rank_direction_ratio": None,
        "baseline": None,
        "baseline_direction": None,
        "parallax_degrees": None,
        "cheirality_fraction": None,
        "cheirality_positive": None,
        "cheirality_negative": None,
        "stability": None,
        "gate_pass": False,
        "gate_refusal": "rank_deficient",
    }
    if fit["direction"] is None or fit["consensus_count"] == 0:
        return record
    consensus = fit["consensus"]
    # The raw rows already contain the geometric baseline, so their singular
    # spectrum is an unconditional rank test: a second singular value at the
    # floating-point floor means the one-dimensional null space the estimator
    # needs does not exist (classic coplanar / collinear degeneracy).
    consensus_singular = _singular_values(rows[consensus])
    record["consensus_singular_values"] = consensus_singular
    record["rank_direction_ratio"] = (
        consensus_singular[2] / consensus_singular[0]
        if consensus_singular[0] > 1e-12
        else 0.0
    )
    rotation_right_class = camera_rotation_matrix(to_camera_model_from_estimate(right_camera)) @ _ry90(yaw_class)
    sign = choose_sign(fit["direction"], rotation_right_class, geometry["world_left"],
                       world_right, consensus)
    record["baseline"] = sign["baseline"]
    record["baseline_direction"] = sign["baseline"] / max(float(np.linalg.norm(sign["baseline"])), 1e-12)
    record["cheirality_fraction"] = sign["cheirality_fraction"]
    record["cheirality_positive"] = sign["positive"]
    record["cheirality_negative"] = sign["negative"]
    record["parallax_degrees"] = _median_angle_degrees(
        geometry["world_left"][consensus], world_right[consensus]
    )
    record["stability"] = stability_degrees(
        rows, consensus, left_points, right_points, width, height,
    )
    record["gate_pass"], record["gate_refusal"] = _gate_check(record)
    return record


def _gate_check(hypothesis: dict) -> tuple[bool, str]:
    """The unchanged DS-11 class gates, applied to an offline hypothesis.

    The rank gate is the frozen ``min_singular_ratio`` applied to the class's
    informative rows, exactly as DS-11 applies it to its inlier support.
    """
    if hypothesis["row_max"] <= 1e-9 or hypothesis["informative_count"] < 2:
        return False, "pure_rotation"
    if hypothesis["singular_ratio"] is None or hypothesis["singular_ratio"] < EDGE_RANK_MIN_RATIO:
        return False, "rank_deficient"
    if hypothesis["parallax_degrees"] is None or hypothesis["parallax_degrees"] < EDGE_PARALLAX_MIN_DEGREES:
        return False, "pure_rotation"
    if hypothesis["consensus_count"] < EDGE_CONSENSUS_MIN:
        return False, "insufficient_inliers"
    if hypothesis["consensus_fraction"] < EDGE_CONSENSUS_FRACTION:
        return False, "low_inlier_fraction"
    if hypothesis["median_rho_degrees"] is None or hypothesis["median_rho_degrees"] > EDGE_CONSENSUS_DEGREES:
        return False, "high_residual"
    if hypothesis["cheirality_fraction"] is None or hypothesis["cheirality_fraction"] < EDGE_CHEIRALITY_MIN:
        return False, "cheirality_failed"
    return True, ""


def adjudicate(geometry: dict, left_camera, right_camera,
               left_points: np.ndarray, right_points: np.ndarray,
               left_edges: np.ndarray, right_edges: np.ndarray,
               width: int, height: int) -> dict:
    """Consensus yaw-class selection with a support margin, all gates unchanged.

    Everything except (a) the yaw-class selection rule, (b) the baseline sign
    rule and (c) the baseline fit (recomputed consensus subset instead of a
    RANSAC support) is the frozen DS-11 gate set above. The spatial-stratum
    consistency of the winner is measured and reported, but is not a refusal
    condition: it is a diagnostic, not a gate.
    """
    hypotheses = {
        yaw_class: class_hypothesis(
            geometry, yaw_class, left_camera, right_camera,
            left_points, right_points, left_edges, right_edges, width, height,
        )
        for yaw_class in (0, 1, 2, 3)
    }
    record: dict = {
        "solved": False,
        "reason": "no_yaw_class_passes_gates",
        "right_yaw_class": None,
        "baseline_direction": None,
        "inliers": 0,
        "inlier_fraction": 0.0,
        "median_rho_degrees": None,
        "parallax_degrees": None,
        "cheirality_fraction": None,
        "singular_ratio": None,
        "stability": None,
        "class_margin": None,
        "classes": {str(key): value for key, value in hypotheses.items()},
    }
    passing = [value for value in hypotheses.values() if value["gate_pass"]]
    if not passing:
        refused = sorted(
            hypotheses.values(),
            key=lambda item: (
                item["median_all_residual_degrees"] is None,
                0.0 if item["median_all_residual_degrees"] is None else item["median_all_residual_degrees"],
                item["yaw_class"],
            ),
        )
        record["reason"] = refused[0]["gate_refusal"] or "no_yaw_class_passes_gates"
        return record
    winner = sorted(
        passing,
        key=lambda item: (-item["consensus_count"], item["median_rho_degrees"], item["yaw_class"]),
    )[0]
    competitors = [value for value in passing if value["yaw_class"] != winner["yaw_class"]]
    margin = None
    if competitors:
        runner_up = sorted(
            competitors,
            key=lambda item: (-item["consensus_count"], item["median_rho_degrees"], item["yaw_class"]),
        )[0]
        margin = winner["consensus_count"] - runner_up["consensus_count"]
        required = max(CLASS_MARGIN_MIN, CLASS_MARGIN_FRACTION * winner["consensus_count"])
        record["class_margin"] = int(margin)
        if margin < required:
            record["reason"] = "ambiguous_yaw_class"
            return record
    record["right_yaw_class"] = winner["yaw_class"]
    record["baseline_direction"] = winner["baseline_direction"]
    record["inliers"] = winner["consensus_count"]
    record["inlier_fraction"] = winner["consensus_fraction"]
    record["median_rho_degrees"] = winner["median_rho_degrees"]
    record["parallax_degrees"] = winner["parallax_degrees"]
    record["cheirality_fraction"] = winner["cheirality_fraction"]
    record["singular_ratio"] = winner["singular_ratio"]
    record["stability"] = winner["stability"]
    record["solved"] = True
    record["reason"] = ""
    return record


# --------------------------------------------------------------------------- #
# Reference evaluation (the only place reference metadata is used).
# --------------------------------------------------------------------------- #
class _AdjudicatedPose:
    """The six fields the frozen DS-16B pose evaluator reads from a ``PairPose``.

    The online evaluator is :func:`diagnose_match_attribution._evaluate_pose`; this
    object only *carries* the offline adjudicator's decision to it. Rotation and
    baseline direction use exactly the ``solve_pair`` conventions
    (``relative_rotation = R_right_est @ Ry90(class) @ R_left_est.T`` and
    ``baseline_direction = normalize(-(R_right_est @ Ry90(class).T @ t))``, which
    is the vector the adjudicator already reports), so the frozen metric reads the
    adjudicated pose in the same gauge it reads the production pose. No evaluation
    maths lives here.
    """

    def __init__(self, record: dict, left_camera, right_camera) -> None:
        class_index = int(record["right_yaw_class"])
        rotation_right_class = (
            camera_rotation_matrix(to_camera_model_from_estimate(right_camera))
            @ _ry90(class_index)
        )
        self.solved = True
        self.reason = ""
        self.right_yaw_class = class_index
        self.relative_rotation = (
            rotation_right_class
            @ camera_rotation_matrix(to_camera_model_from_estimate(left_camera)).T
        )
        direction = np.asarray(record["baseline_direction"], dtype=np.float64)
        self.baseline_direction = tuple(float(value) for value in direction)
        self.inliers = int(record["inliers"])
        self.inlier_fraction = float(record["inlier_fraction"])
        self.median_rho_degrees = record["median_rho_degrees"]
        self.parallax_degrees = record["parallax_degrees"]
        self.cheirality_fraction = record["cheirality_fraction"]


def evaluate_offline(record: dict, left_camera, right_camera,
                     left_model, right_model, expected_class: int) -> dict:
    """Score an offline adjudicator decision with the **frozen** DS-16B evaluator.

    ``_evaluate_pose`` is imported verbatim from ``diagnose_match_attribution``
    (DS-16/DS-16B) and is *not* reimplemented, re-thresholded or re-gauged here.
    The only diagnostic addition is ``usable``: the DS-19 preregistered
    conjunction of a correct reference yaw class, ``within_5deg`` (the frozen
    evaluator's own rotation bound) and the frozen evaluator's own gauge-aligned
    baseline error <= 5 deg. ``within_5deg`` and
    ``baseline_error_degrees_gauge_aligned`` come straight from the frozen
    evaluator and are never edited.
    """
    if not record["solved"] or record["right_yaw_class"] is None:
        out = {
            "solved": False,
            "reason": record["reason"],
            "right_yaw_class": record["right_yaw_class"],
            "class_equals_expected": False,
            "inliers": int(record["inliers"]),
            "inlier_fraction": float(record["inlier_fraction"]),
            "median_rho_degrees": record["median_rho_degrees"],
            "parallax_degrees": record["parallax_degrees"],
            "cheirality_fraction": record["cheirality_fraction"],
            "singular_ratio": record["singular_ratio"],
            "rotation_error_degrees": None,
            "within_5deg": False,
            "usable": False,
            "baseline_error_degrees_gauge_aligned": None,
            "baseline_error_degrees_signless": None,
            "gauge_index": None,
        }
        return out
    frozen = _frozen_evaluator._evaluate_pose(
        _AdjudicatedPose(record, left_camera, right_camera),
        left_model, right_model, expected_class,
    )
    out = {
        "solved": True,
        "reason": record["reason"],
        "right_yaw_class": record["right_yaw_class"],
        "class_equals_expected": frozen["class_equals_expected"],
        "inliers": int(frozen["inliers"]),
        "inlier_fraction": float(frozen["inlier_fraction"]),
        "median_rho_degrees": frozen["median_rho_degrees"],
        "parallax_degrees": frozen["parallax_degrees"],
        "cheirality_fraction": frozen["cheirality_fraction"],
        "singular_ratio": record["singular_ratio"],
        "rotation_error_degrees": frozen["rotation_error_degrees"],
        "within_5deg": bool(frozen["within_5deg"]),
        "baseline_error_degrees_gauge_aligned": frozen["baseline_error_degrees_gauge_aligned"],
        "baseline_error_degrees_signless": frozen["baseline_error_degrees_signless"],
        "gauge_index": frozen["gauge_index"],
    }
    out["usable"] = bool(
        out["class_equals_expected"]
        and out["within_5deg"]
        and out["baseline_error_degrees_gauge_aligned"] is not None
        and out["baseline_error_degrees_gauge_aligned"] <= STABILITY_DEGREES
    )
    return out


# --------------------------------------------------------------------------- #
# Synthetic degeneracy controls.
# --------------------------------------------------------------------------- #
SYNTHETIC_POSES = {
    # kind: ((left yaw, left pitch), (right yaw, right pitch), same eye?)
    "valid_baseline": ((0.0, 0.0), (0.0, 0.0), False),
    "parallel_axes": ((30.0, 0.0), (30.0, 0.0), False),
    "pure_rotation": ((0.0, 0.0), (25.0, 0.0), True),
}
SYNTHETIC_EXPECTATION = {
    "valid_baseline": "solved",
    "parallel_axes": "solved",
    "pure_rotation": "refused",
}


def _synthetic_case(kind: str, rng: np.random.Generator) -> dict:
    """Build one synthetic pair with ideal correspondences from a shared cloud.

    Camera yaw follows the Minecraft convention (forward is +X at yaw 0 and +Z
    at yaw 90); the estimate intrinsics are taken from the same model that
    produced the pixels, so rays and pixels are exactly consistent.
    """
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraPose
    from mc_photogrammetry.reconstruction.camera import intrinsics_for, project_points

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

    # A distant cloud so both views keep every point inside their frames.
    centre = rng.uniform(-12.0, 12.0, 3) + np.asarray((0.0, 0.0, 48.0))
    points = centre + rng.uniform(-22.0, 22.0, (360, 3))
    left_pixels = project_points([tuple(point) for point in points], left_model,
                                 width=width, height=height)
    right_pixels = project_points([tuple(point) for point in points], right_model,
                                  width=width, height=height)
    pairs = [
        (float(left_pixel[0]), float(left_pixel[1]), float(right_pixel[0]), float(right_pixel[1]))
        for left_pixel, right_pixel in zip(left_pixels, right_pixels)
        if left_pixel is not None and right_pixel is not None
    ]
    matches = (
        np.asarray(pairs, dtype=np.float64).reshape(-1, 4)
        if pairs
        else np.empty((0, 4), dtype=np.float64)
    )
    return {
        "kind": kind,
        "left_camera": left_estimate,
        "right_camera": right_estimate,
        "left_model": left_model,
        "right_model": right_model,
        "matches": matches,
        "width": width,
        "height": height,
        "expected_class": 0,
    }


def _synthetic_model(position, yaw_degrees, pitch_degrees):
    from mc_photogrammetry.reconstruction.camera import CameraModel

    return CameraModel(
        position=tuple(float(value) for value in position),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        roll_degrees=0.0,
        eye_height=1.62,
    )


def synthetic_controls(log: _Log) -> dict:
    """Adjudicator behaviour on ideal-correspondence synthetic geometry.

    ``valid_baseline`` and ``parallel_axes`` are fully observable and must be
    solved; ``pure_rotation`` puts both eyes at the same point so no baseline
    can be observed and must be refused. The expectation is fixed before the
    run; a family that does not match it is reported as a failed control rather
    than tuned away.
    """
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
            if len(matches) < EDGE_CONSENSUS_MIN:
                reasons["insufficient_correspondences"] += 1
                continue
            edges = np.column_stack(
                (np.arange(len(matches)), np.arange(len(matches)))
            ).astype(np.int64)
            geometry = class_rows(case["left_camera"], case["right_camera"], matches)
            decision = adjudicate(
                geometry, case["left_camera"], case["right_camera"],
                matches[:, 0:2], matches[:, 2:4], edges[:, 0], edges[:, 1],
                case["width"], case["height"],
            )
            if decision["solved"]:
                solved += 1
                classes[decision["right_yaw_class"]] += 1
                evaluation = evaluate_offline(
                    decision, case["left_camera"], case["right_camera"],
                    case["left_model"], case["right_model"], case["expected_class"],
                )
                if evaluation["usable"]:
                    usable += 1
            else:
                reasons[decision["reason"]] += 1
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
            f"refusals {dict(sorted(reasons.items()))} control {'ok' if control_ok else 'FAILED'}"
        )
    return {
        "families": rows,
        "all_controls_ok": all(row["control_ok"] for row in rows),
    }


# --------------------------------------------------------------------------- #
# Attribution.
# --------------------------------------------------------------------------- #
def _gate_status(classes: list[dict], expected_class: int) -> dict:
    """Which gate each yaw class fails, with the true class highlighted.

    ``classes`` is the DS-11 ``YawClassScore`` read-out (``refusal``/``passed``).
    The offline adjudicator's per-class records use ``gate_refusal``/``gate_pass``
    instead, and are normalised here for the attribution step.
    """
    normalised = [
        {
            "yaw_class": item["yaw_class"],
            "passed": item.get("passed", item.get("gate_pass", False)),
            "refusal": item.get("refusal", item.get("gate_refusal")),
            "inliers": item.get("inliers", item.get("consensus_count")),
            "median_rho_degrees": item.get("median_rho_degrees"),
        }
        for item in classes
    ]
    true_class = next((item for item in normalised if item["yaw_class"] == expected_class), None)
    others = [item for item in normalised if item["yaw_class"] != expected_class]
    return {
        "true_class_refusal": None if true_class is None else true_class["refusal"],
        "true_class_passed": bool(true_class and true_class["passed"]),
        "true_class_inliers": None if true_class is None else true_class["inliers"],
        "true_class_median_rho_degrees": None if true_class is None else true_class["median_rho_degrees"],
        "other_refusals": sorted({item["refusal"] for item in others}),
        "any_other_passed": any(item["passed"] for item in others),
    }


def attribute_pair(pair_report: dict) -> dict:
    """Diagnose one pair.

    ``diagnosis`` is the DS-19 attribution when the *production* reader still
    failed (i.e. what the adjudicator repaired); ``production_failure_class``
    keeps that same production diagnosis for every pair, including the ones the
    adjudicator now solves, so the tally stays informative. When the production
    reader already class-matched, there is no production failure to attribute and
    ``production_failure_class`` is ``None``.
    """
    base = pair_report["production"]["oracle_bestof_current"]
    full = pair_report["production"]["oracle_full_current"]
    adjudicated = pair_report["adjudicated"]
    geometry = pair_report["geometry"]
    reference_class = pair_report["reference_class"]
    reference_floors = pair_report["reference_class_floors"]

    production_classification = classify_production_failure(
        base, full, adjudicated, geometry, reference_class, reference_floors
    )
    production_failed = not base["class_equals_expected"]
    if production_failed:
        return {
            **production_classification,
            "production_failure_class": production_classification["diagnosis"],
        }
    return {
        "diagnosis": (
            "adjudicated_solved"
            if adjudicated["solved"] and adjudicated["evaluation"]["usable"]
            else "production_already_class_correct"
        ),
        "detail": (
            "the production DS-11 reader already reports the reference yaw class, so there is "
            "no production failure to attribute; the adjudicator "
            + (
                "keeps the pose usable under the frozen evaluator"
                if adjudicated["solved"] and adjudicated["evaluation"]["usable"]
                else "loses it"
            )
        ),
        "production_reason": base["reason"],
        "production_full_reason": full["reason"],
        "production_failure_class": None,
    }


def classify_production_failure(
    base: dict, full: dict, adjudicated: dict, geometry: dict,
    reference_class: int, reference_floors: int,
) -> dict:
    """Assign the production failure diagnosis from the measured evidence."""
    # 1. Evaluator ambiguity: the reference class is not unique between its two
    #    derivations (manifest floors vs best public-API match), or the adjudicator
    #    fails to reproduce the frozen evaluator's verdict. Reported, never
    #    promoted into an accepted alternate label.
    if reference_class != reference_floors:
        return {
            "diagnosis": "evaluator_ambiguity",
            "detail": (
                "reference yaw class disagrees between manifest floors "
                f"({reference_floors}) and best public-API match ({reference_class}); "
                "reported as evaluator ambiguity, not an accepted alternate answer"
            ),
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    gate = _gate_status(base.get("classes", []), reference_class)
    # An unsolved DS-11 read still carries its per-class diagnostics in the full
    # read-out; use it to classify the true class's gate status when the best-of
    # read-out refused before returning scores.
    if not base.get("classes") and full.get("classes"):
        gate = _gate_status(full.get("classes", []), reference_class)
    detail = (
        f"true class {reference_class} refusal="
        f"{gate['true_class_refusal']!r} inliers={gate['true_class_inliers']} "
        f"median_rho={gate['true_class_median_rho_degrees']}"
    )

    # 2. Yaw-class scoring: the true class passes but loses, or the residual
    #    margin to a passing competitor is below the frozen margin.
    if gate["true_class_passed"] and base["reason"] == "ambiguous_yaw_class":
        return {
            "diagnosis": "yaw_class_scoring",
            "detail": detail + "; the true class passes every gate but the frozen "
            "residual-margin rule rejects it as ambiguous",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    # 3. Baseline null-space conditioning.
    stability = adjudicated.get("stability") or geometry.get("stability")
    if stability and stability.get("available") and stability.get("disagreeing_pairs"):
        return {
            "diagnosis": "baseline_null_space_conditioning",
            "detail": detail + f"; spatial-stratum refits disagree by "
            f"{stability['pairwise_max_degrees']:.2f} deg "
            f"({stability['disagreeing_pairs']}/{stability['pairs']} pairs > {STABILITY_DEGREES:g} deg)",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }
    singular = geometry.get("singular_ratio")
    if singular is not None and singular < EDGE_RANK_MIN_RATIO * 10:
        return {
            "diagnosis": "baseline_null_space_conditioning",
            "detail": detail + f"; true-class singular ratio {singular:.3e} is within one "
            f"decade of the frozen rank gate {EDGE_RANK_MIN_RATIO:g}",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    # 4. Cheirality.
    if gate["true_class_refusal"] == "cheirality_failed" or base["reason"] == "cheirality_failed":
        return {
            "diagnosis": "cheirality",
            "detail": detail + "; the depth-positivity fraction is below the frozen 0.75 gate",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    # 5. Parallax.
    if gate["true_class_refusal"] == "pure_rotation" or base["reason"] == "pure_rotation":
        return {
            "diagnosis": "parallax",
            "detail": detail + "; the median parallax is below the frozen 3 deg gate",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    # 6. Correspondence geometry: resolvable edges exist but the set is thin or
    #    spatially confined, so no estimator can pin the baseline.
    used_bins = geometry.get("used_bins")
    if base["rows_used"] < 2 * EDGE_CONSENSUS_MIN or (used_bins is not None and used_bins < 4):
        return {
            "diagnosis": "correspondence_geometry",
            "detail": detail + f"; only {base['rows_used']} oracle-bestof edges over "
            f"{used_bins} spatial bins",
            "production_reason": base["reason"],
            "production_full_reason": full["reason"],
        }

    return {
        "diagnosis": "unknown",
        "detail": detail + f"; production reason {base['reason']!r}, adjudicator reason "
        f"{adjudicated['reason']!r}",
        "production_reason": base["reason"],
        "production_full_reason": full["reason"],
    }


# --------------------------------------------------------------------------- #
# Main diagnostic.
# --------------------------------------------------------------------------- #
def run(out_dir: Path, log: _Log) -> dict:
    started_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen-input verification (before any reference label) ------------- #
    candidates_json = DS16_DIR / "match_candidates.json"
    attribution_json = DS16_DIR / "match_attribution.json"
    sweep_json = DS16B_DIR / "match_attribution_sweep.json"
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
    }
    for name, record in digests.items():
        record["ok"] = record["actual"] == record["expected"]
        if not record["ok"]:
            raise SystemExit(
                f"frozen input {name} was modified: sha256 {record['actual']} "
                f"!= preregistered {record['expected']}"
            )
    log("[verify] all four frozen DS-16/DS-16B digests match")

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

    pairs = [pair for pair in frozen["pairs"] if pair["set"] in SELECTED_SETS]
    if len(pairs) != PAIR_COUNT:
        raise SystemExit(
            f"selection rule expected the nine T2/T6 pairs, found {len(pairs)}; refusing to run"
        )
    log(f"[select] {len(pairs)} T2/T6 pairs at the frozen base cell {BASE_CELL}")

    # --- reference fixtures (the only reference-assisted step) -------------- #
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
        log(
            f"[render] {name[:24]:24s} kp {len(points):5d} "
            f"render {alignment[-1]['render_seconds']:.2f}s"
        )
        del result, index, points, usable
        log.checkpoint(f"render {name}")

    medians = [row["alignment_median_px"] for row in alignment if row["alignment_median_px"] is not None]
    alignment_median = float(np.median(medians)) if medians else None
    if not medians or alignment_median > ALIGNMENT_MEDIAN_PX_MAX:
        raise SystemExit(
            "blocker: reference renderer ownership does not align with the manifest cameras "
            f"(median re-projection error {alignment_median} px); refusing to trust oracle labels"
        )
    log(f"[align] median frame re-projection error {alignment_median:.4f} px (<= {ALIGNMENT_MEDIAN_PX_MAX})")

    # --- per-pair oracle truth and solve read-outs --------------------------- #
    normal_cosine = _normal_cosine(BASE_NORMAL_DEGREES)
    pair_reports: list[dict] = []
    for pair in pairs:
        fixture = fixtures[pair["fixture"]]
        left_name, right_name = pair["left"], pair["right"]
        left_record, right_record = frames[left_name], frames[right_name]
        left_points = np.asarray(arrays[f"feat_points_{left_record['key']}"], dtype=np.float64)
        right_points = np.asarray(arrays[f"feat_points_{right_record['key']}"], dtype=np.float64)
        left_label, right_label = labels[left_name], labels[right_name]
        left_world = left_label["world"]
        right_world = right_label["world"]
        width, height = int(left_record["width"]), int(left_record["height"])

        left_est = _estimate_from_record(pair["left_estimate"])
        right_est = _estimate_from_record(pair["right_estimate"])
        left_model, right_model = fixture["cameras"][left_name], fixture["cameras"][right_name]
        reference_class, reference_floors = _reference_class(
            left_est, right_est, left_model, right_model
        )
        current_left = to_camera_estimate(left_est)
        current_right = to_camera_estimate(right_est)
        reference_left = _camera_estimate(left_model, width, height)
        reference_right = _camera_estimate(right_model, width, height)

        truth = oracle_edges(left_label, right_label, BASE_DISTANCE_BLOCKS, normal_cosine)
        best = best_of_edges(truth, left_world, right_world)

        # Frozen cross-checks: DS-16 many-to-many count and DS-16B cell counts.
        reference_row = known_attribution.get((pair["fixture"], left_name, right_name))
        sweep_row = known_pairs.get((pair["fixture"], left_name, right_name))
        cross = {
            "ds16_oracle_truths": None if reference_row is None else reference_row["candidate_stats"]["oracle_truths"],
            "ds16b_oracle_truths": None if sweep_row is None else sweep_row["oracle_truths"],
            "ds16b_oracle_bestof_edges": None if sweep_row is None else sweep_row["oracle_bestof_edges"],
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

        # Production DS-11 read-outs on the frozen oracle edge sets.
        production: dict[str, dict] = {}
        for key, edges, left_camera, right_camera, expected in (
            ("oracle_bestof_current", best, current_left, current_right, reference_class),
            ("oracle_bestof_reference", best, reference_left, reference_right, 0),
            ("oracle_full_current", truth, current_left, current_right, reference_class),
            ("oracle_full_reference", truth, reference_left, reference_right, 0),
        ):
            rows, sampled = _bounded_rows(edges)
            matches = _matches_from_edges(left_points, right_points, rows)
            if len(matches) < PairGates().min_correspondences:
                record = {
                    "solved": False,
                    "reason": "insufficient_correspondences",
                    "right_yaw_class": None,
                    "class_equals_expected": False,
                    "inliers": 0,
                    "inlier_fraction": 0.0,
                    "median_rho_degrees": None,
                    "parallax_degrees": None,
                    "cheirality_fraction": None,
                    "rotation_error_degrees": None,
                    "within_5deg": False,
                    "baseline_error_degrees_gauge_aligned": None,
                    "baseline_error_degrees_signless": None,
                    "gauge_index": None,
                    "rows_used": int(len(matches)),
                    "rows_full": int(len(edges)),
                    "sampled": bool(sampled),
                    "classes": [],
                    "per_class": None,
                }
            else:
                pose = solve_pair(
                    left_camera, right_camera, matches,
                    left_id=left_name, right_id=right_name, gates=PairGates(),
                )
                record = _frozen_evaluator._evaluate_pose(pose, left_model, right_model, expected)
                record["rows_used"] = int(len(matches))
                record["rows_full"] = int(len(edges))
                record["sampled"] = bool(sampled)
                record["classes"] = [
                    {
                        "yaw_class": score.yaw_class,
                        "passed": bool(score.passed),
                        "refusal": score.refusal,
                        "inliers": int(score.inliers),
                        "informative": int(score.informative),
                        "inlier_fraction": float(score.inlier_fraction),
                        "median_rho_degrees": score.median_rho_degrees,
                        "parallax_degrees": score.parallax_degrees,
                        "cheirality_fraction": score.cheirality_fraction,
                        "singular_ratio": score.singular_ratio,
                    }
                    for score in pose.classes
                ]
                record["per_class"] = {
                    "inlier_fraction_denominator": "informative rows",
                    "classes": record["classes"],
                }
            production[key] = record

        # Offline adjudicator on the same frozen oracle-bestof edges.
        edges_best = best
        geometry = class_rows(current_left, current_right,
                              _matches_from_edges(left_points, right_points, edges_best)) \
            if len(edges_best) >= EDGE_CONSENSUS_MIN else None
        if geometry is None:
            adjudicated = {
                "solved": False,
                "reason": "insufficient_correspondences",
                "right_yaw_class": None,
                "baseline_direction": None,
                "inliers": 0,
                "inlier_fraction": 0.0,
                "median_rho_degrees": None,
                "parallax_degrees": None,
                "cheirality_fraction": None,
                "singular_ratio": None,
                "stability": None,
                "class_margin": None,
                "classes": {},
                "evaluation": {
                    "solved": False,
                    "reason": "insufficient_correspondences",
                    "class_equals_expected": False,
                    "within_5deg": False,
                    "rotation_error_degrees": None,
                    "baseline_error_degrees_gauge_aligned": None,
                },
            }
        else:
            decision = adjudicate(
                geometry, current_left, current_right,
                left_points, right_points, edges_best[:, 0], edges_best[:, 1], width, height,
            )
            decision["evaluation"] = evaluate_offline(
                decision, current_left, current_right, left_model, right_model, reference_class
            )
            adjudicated = decision

        # Reference-camera offline read-out (gauge comparison only).
        reference_geometry = class_rows(
            reference_left, reference_right,
            _matches_from_edges(left_points, right_points, edges_best),
        ) if len(edges_best) >= EDGE_CONSENSUS_MIN else None
        if reference_geometry is None:
            adjudicated_reference = {
                "solved": False,
                "reason": "insufficient_correspondences",
                "evaluation": {
                    "solved": False,
                    "class_equals_expected": False,
                    "within_5deg": False,
                    "rotation_error_degrees": None,
                    "baseline_error_degrees_gauge_aligned": None,
                },
            }
        else:
            decision = adjudicate(
                reference_geometry, reference_left, reference_right,
                left_points, right_points, edges_best[:, 0], edges_best[:, 1], width, height,
            )
            decision["evaluation"] = evaluate_offline(
                decision, reference_left, reference_right, left_model, right_model, 0
            )
            adjudicated_reference = decision

        # Measured geometry of the frozen oracle-bestof set.
        left_bins = _spatial_bins(left_points[edges_best[:, 0]], width, height) if len(edges_best) else np.zeros(0, dtype=np.int64)
        right_bins = _spatial_bins(right_points[edges_best[:, 1]], width, height) if len(edges_best) else np.zeros(0, dtype=np.int64)
        geometry_summary = {
            "truth_edges": int(len(truth)),
            "bestof_edges": int(len(best)),
            "left_keypoints_with_hit": int(np.count_nonzero(left_label["hit"])),
            "used_bins": int(len(np.unique(left_bins * 16 + right_bins))) if len(left_bins) else 0,
            "left_bins": int(len(np.unique(left_bins))) if len(left_bins) else 0,
            "right_bins": int(len(np.unique(right_bins))) if len(right_bins) else 0,
            "singular_ratio": None,
            "stability": None,
            "truth_available": int(len(truth)),
        }
        if adjudicated["classes"]:
            true_key = str(reference_class)
            true_hypothesis = adjudicated["classes"].get(true_key, {})
            geometry_summary["singular_ratio"] = true_hypothesis.get("singular_ratio")
            geometry_summary["stability"] = true_hypothesis.get("stability")
            geometry_summary["truth_rows"] = int(len(truth))
        # Gauge cross-check: when the adjudicator and the frozen DS-11 reader
        # agree on the yaw class, the frozen evaluator must read both poses in
        # one gauge, so the measured errors have to be the same order. A large
        # gap would mean the adjudicated baseline was handed over in a different
        # frame and the score would be meaningless.
        gauge_check = {"comparable": None, "detail": "one of the two did not solve"}
        base_record = production["oracle_bestof_current"]
        if (
            adjudicated["solved"]
            and base_record["solved"]
            and adjudicated["right_yaw_class"] == base_record["right_yaw_class"]
            and adjudicated["evaluation"]["rotation_error_degrees"] is not None
            and base_record["rotation_error_degrees"] is not None
            and adjudicated["evaluation"]["baseline_error_degrees_gauge_aligned"] is not None
            and base_record["baseline_error_degrees_gauge_aligned"] is not None
        ):
            rotation_gap = abs(
                adjudicated["evaluation"]["rotation_error_degrees"]
                - base_record["rotation_error_degrees"]
            )
            baseline_gap = abs(
                adjudicated["evaluation"]["baseline_error_degrees_gauge_aligned"]
                - base_record["baseline_error_degrees_gauge_aligned"]
            )
            gauge_check = {
                "comparable": bool(rotation_gap <= GAUGE_COMPARABLE_DEGREES),
                "rotation_gap_degrees": rotation_gap,
                "baseline_gap_degrees": baseline_gap,
                "detail": "same yaw class as the frozen DS-11 reader",
            }
        pair_reports.append(
            {
                "pair": f"{left_name}->{right_name}",
                "set": pair["set"],
                "fixture": pair["fixture"],
                "left": left_name,
                "right": right_name,
                "reference_class": reference_class,
                "reference_class_floors": reference_floors,
                "cross_check": cross,
                "gauge_check": gauge_check,
                "production": production,
                "adjudicated": adjudicated,
                "adjudicated_reference": adjudicated_reference,
                "geometry": geometry_summary,
            }
        )
        solved_current = sum(
            1 for key in ("oracle_bestof_current", "oracle_full_current") if production[key]["solved"]
        )
        log(
            f"[pair  ] {left_name[11:19]}->{right_name[11:19]} truths {len(truth):6d} "
            f"bestof {len(best):5d} prod-bestof {production['oracle_bestof_current']['reason'] or 'SOLVED':24s} "
            f"prod-full {production['oracle_full_current']['reason'] or 'SOLVED':24s} "
            f"adj {adjudicated['reason'] or 'SOLVED'} (cur {solved_current})"
        )
        del truth, best, geometry, reference_geometry, left_world, right_world
        log.checkpoint(f"pair {left_name}")

    # --- attribution and threshold decision --------------------------------- #
    for report in pair_reports:
        report["attribution"] = attribute_pair(report)

    base_solved = [
        report for report in pair_reports
        if report["production"]["oracle_bestof_current"]["solved"]
    ]
    base_correct = [
        report for report in pair_reports
        if report["production"]["oracle_bestof_current"]["class_equals_expected"]
    ]
    full_solved = [
        report for report in pair_reports
        if report["production"]["oracle_full_current"]["solved"]
    ]
    reference_correct = [
        report for report in pair_reports
        if report["production"]["oracle_bestof_reference"]["class_equals_expected"]
    ]
    if len(base_solved) != 6:
        raise SystemExit(
            f"blocker: expected the frozen 6/9 DS-11 oracle best-of solves, measured "
            f"{len(base_solved)}/9; refusing to attribute"
        )
    log(
        f"[verify] reproduced the 6/9 DS-11 oracle read-out "
        f"(best-of {len(base_solved)} solved, {len(base_correct)} class-correct; "
        f"full {len(full_solved)} solved; reference {len(reference_correct)} class-correct)"
    )

    original_correct_pairs = {
        report["pair"] for report in pair_reports
        if report["production"]["oracle_bestof_current"]["class_equals_expected"]
    }
    adjudicated_usable = [
        report for report in pair_reports
        if report["adjudicated"]["evaluation"]["usable"]
    ]
    adjudicated_correct = [
        report for report in pair_reports
        if report["adjudicated"]["evaluation"]["class_equals_expected"]
    ]
    regressions = [
        report["pair"] for report in pair_reports
        if report["pair"] in original_correct_pairs
        and not report["adjudicated"]["evaluation"]["usable"]
    ]
    gauge_checks = [report["gauge_check"] for report in pair_reports]
    gauge_comparable = [
        check for check in gauge_checks if check.get("comparable") is True
    ]
    gauge_incomparable = [
        check for check in gauge_checks if check.get("comparable") is False
    ]
    reference_class_checks = {
        report["pair"]: {
            "direct": report["reference_class"],
            "floors": report["reference_class_floors"],
            "disagree": report["reference_class"] != report["reference_class_floors"],
        }
        for report in pair_reports
    }
    evaluator_disagreements = [
        pair for pair, check in reference_class_checks.items() if check["disagree"]
    ]
    log(
        f"[evalu ] floor/reference-class disagreements: {len(evaluator_disagreements)}/9 "
        f"(reported as evaluator ambiguity, never an accepted alternate label)"
    )
    log(
        f"[gauge ] adjudicated and frozen DS-11 poses agree in gauge on "
        f"{len(gauge_comparable)} pairs, disagree on {len(gauge_incomparable)} "
        f"(bound {GAUGE_COMPARABLE_DEGREES:g} deg)"
    )
    synthetic = synthetic_controls(log)

    threshold = {
        # Frozen DS-16B evaluator read-out: correct reference yaw class, its own
        # ``within_5deg`` rotation bound, and its own gauge-aligned baseline error
        # <= 5 deg. Nothing here is computed by this diagnostic.
        "usable_poses": {
            "value": len(adjudicated_usable),
            "required": 8,
            "of": PAIR_COUNT,
            "passed": len(adjudicated_usable) >= 8,
        },
        "class_correct": {
            "value": len(adjudicated_correct),
            "required": 8,
            "of": PAIR_COUNT,
            "passed": len(adjudicated_correct) >= 8,
        },
        "gauge_aligned_baseline_within_5deg": {
            "value": sum(
                1 for report in adjudicated_usable
                if report["adjudicated"]["evaluation"]["baseline_error_degrees_gauge_aligned"] <= 5.0
            ),
            "required": 8,
            "of": PAIR_COUNT,
            "passed": len(adjudicated_usable) >= 8,
        },
        "zero_regressions_among_original_six": {
            "value": len(regressions),
            "required": 0,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        },
        "synthetic_degeneracy_refusal": {
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
    }
    threshold["all_passed"] = all(item["passed"] for item in threshold.values())
    threshold["status"] = (
        "adjudicator_meets_preregistered_threshold"
        if threshold["all_passed"]
        else "adjudicator_fails_preregistered_threshold"
    )
    log(f"[thresh] {threshold['status']}: usable {len(adjudicated_usable)}/9, "
        f"class-correct {len(adjudicated_correct)}/9, regressions {len(regressions)}, "
        f"synthetic {threshold['synthetic_degeneracy_refusal']['value']}")

    # --- write artifacts ----------------------------------------------------- #
    cell = {
        "cell": BASE_CELL,
        "distance_blocks": BASE_DISTANCE_BLOCKS,
        "normal_degrees": BASE_NORMAL_DEGREES,
        "normal_cosine": normal_cosine,
        "render_scale": RENDER_SCALE,
        "oracle_truth_rule": (
            "frozen DS-16 oracle predicate: render-scale ownership hit on both sides, "
            "world-hit distance <= distance_blocks, face-normal dot >= cos(normal_degrees); "
            "one nearest oracle partner per left keypoint (oracle_bestof), and the complete "
            "many-to-many set (oracle_full) for the reproduction cross-check"
        ),
        "pairs": pair_reports,
    }
    payload = {
        "schema": "ds19-oracle-pose-failures-1",
        "phase": "reference_assisted_oracle_pose_failure_diagnostic",
        "status": "complete",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "directive": (
            "AGENT_COMMS.md 2026-09-12 13:04 CDT (DS-19): reproduce the 6/9 DS-11 oracle "
            "solves, attribute the failures, test one physically justified adjudicator against "
            "the frozen oracle edges, >=8/9 usable poses with the yaw class correct and "
            "gauge-aligned baseline error <=5 deg, zero regressions, synthetic degeneracy refusal"
        ),
        "correction": (
            "AGENT_COMMS.md 2026-09-12 13:16 CDT: the DS-16B reference yaw-class and "
            "gauge-aligned baseline-error evaluator is frozen and is used verbatim via "
            "diagnose_match_attribution._evaluate_pose. A floor-heuristic/reference-class "
            "disagreement is reported as evaluator ambiguity and is not an accepted alternate "
            "label. The preregistered criterion is unchanged."
        ),
        "blindness_boundary": {
            "blind": [
                "frozen DS-16 phase-1 candidate keypoints/descriptors (digest-verified)",
                "frozen DS-14 selected matches",
                "the adjacent camera estimates used as the current (M3) regime",
            ],
            "reference_assisted_here_only": [
                "fixture manifests (cameras.json), litematics and scene transforms used to "
                "render ownership/depth labels",
                "manifest cameras used as the reference-camera regime and for the pose error "
                "score",
                "DS-16/DS-16B frozen artifacts, read only as frozen count cross-checks",
            ],
            "evaluator": {
                "module": "scripts/diagnose_match_attribution.py",
                "function": "_evaluate_pose",
                "frozen": True,
                "reimplemented": False,
                "reference_class_source": "scripts/diagnose_connected_pairs._reference_class",
                "gauge_discovery": (
                    "the frozen evaluator's own 4-way about-y search; reported as "
                    "gauge_index, never used to accept an alternate label"
                ),
            },
            "never_enters_production": True,
            "no_redetection": True,
            "no_threshold_change": True,
        },
        "reproduction": {
            "production_oracle_bestof_solved": len(base_solved),
            "production_oracle_bestof_class_correct": len(base_correct),
            "production_oracle_bestof_reference_class_correct": len(reference_correct),
            "production_oracle_full_solved": len(full_solved),
            "frozen_ds16b_reported_bestof_solved_current": (
                sweep_cells[BASE_CELL]["aggregate"]["cells"]["oracle_bestof_current"]["solved"]
            ),
            "frozen_ds16b_reported_bestof_solved_reference": (
                sweep_cells[BASE_CELL]["aggregate"]["cells"]["oracle_bestof_reference"]["solved"]
            ),
            "frozen_ds16b_reported_full_solved_current": (
                sweep_cells[BASE_CELL]["aggregate"]["cells"]["oracle_full_current"]["solved"]
            ),
            "frozen_ds16b_reported_full_solved_reference": (
                sweep_cells[BASE_CELL]["aggregate"]["cells"]["oracle_full_reference"]["solved"]
            ),
        },
        "attribution_tally": dict(
            sorted(Counter(report["attribution"]["diagnosis"] for report in pair_reports).items())
        ),
        "production_failure_tally": dict(
            sorted(
                Counter(
                    report["attribution"]["production_failure_class"]
                    for report in pair_reports
                    if report["attribution"]["production_failure_class"] is not None
                ).items()
            )
        ),
        "adjudicator": {
            "name": "consensus_yaw_with_stratum_stability",
            "preregistered": True,
            "constants": {
                "edge_consensus_degrees": EDGE_CONSENSUS_DEGREES,
                "edge_consensus_min": EDGE_CONSENSUS_MIN,
                "edge_consensus_fraction": EDGE_CONSENSUS_FRACTION,
                "edge_parallax_min_degrees": EDGE_PARALLAX_MIN_DEGREES,
                "edge_cheirality_min": EDGE_CHEIRALITY_MIN,
                "edge_rank_min_ratio": EDGE_RANK_MIN_RATIO,
                "class_margin_min": CLASS_MARGIN_MIN,
                "class_margin_fraction": CLASS_MARGIN_FRACTION,
                "stability_degrees": STABILITY_DEGREES,
                "stratum_min_edges": STRATUM_MIN_EDGES,
                "spatial_grid": list(SPATIAL_GRID),
            },
            "changed": [
                "baseline direction is the bounded Gram/SVD null vector of the recomputed "
                "residual-consensus subset (not a subsample and not a RANSAC support)",
                "the yaw class is selected by consensus-edge count with an explicit margin "
                "over the runner-up, instead of by the frozen residual-margin rule",
                "the baseline sign is chosen by maximum depth positivity",
            ],
            "diagnostic_only": [
                "spatial-stratum direction consistency of the winner is measured and "
                "reported, never used as a refusal condition",
            ],
            "unchanged": [
                "every DS-11 gate degree (1.0 deg residual, 8 inliers, 0.5 inlier fraction, "
                "3.0 deg parallax, 0.75 cheirality, 1e-3 singular ratio, 0.25 deg margin)",
                "the frozen oracle edges, costs and pair set",
            ],
            "threshold": threshold,
            "evaluator_class_check": {
                "disagreements": len(evaluator_disagreements),
                "disagreement_pairs": evaluator_disagreements,
                "note": (
                    "floor heuristic vs matrix-based reference class; a disagreement is "
                    "reported as evaluator ambiguity and is not an accepted alternate label"
                ),
                "per_pair": reference_class_checks,
            },
            "gauge_handover_self_check": {
                "bound_degrees": GAUGE_COMPARABLE_DEGREES,
                "comparable": len(gauge_comparable),
                "incomparable": len(gauge_incomparable),
                "incomparable_pairs": [report["pair"] for report in pair_reports
                                       if report["gauge_check"].get("comparable") is False],
                "per_pair": {report["pair"]: report["gauge_check"] for report in pair_reports},
            },
            "per_pair": {
                report["pair"]: {
                    "solved": report["adjudicated"]["solved"],
                    "reason": report["adjudicated"]["reason"],
                    "class_equals_expected": report["adjudicated"]["evaluation"]["class_equals_expected"],
                    "within_5deg": report["adjudicated"]["evaluation"]["within_5deg"],
                    "rotation_error_degrees": report["adjudicated"]["evaluation"]["rotation_error_degrees"],
                    "baseline_error_degrees_gauge_aligned": report["adjudicated"]["evaluation"]["baseline_error_degrees_gauge_aligned"],
                    "baseline_error_degrees_signless": report["adjudicated"]["evaluation"]["baseline_error_degrees_signless"],
                    "gauge_index": report["adjudicated"]["evaluation"]["gauge_index"],
                }
                for report in pair_reports
            },
        },
        "synthetic_controls": synthetic,
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
        "cell": cell,
    }

    json_path = out_dir / "oracle_pose_failures.json"
    csv_path = out_dir / "oracle_pose_pairs.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        for report in pair_reports:
            base = report["production"]["oracle_bestof_current"]
            full = report["production"]["oracle_full_current"]
            reference = report["production"]["oracle_bestof_reference"]
            adjudicated = report["adjudicated"]
            evaluation = adjudicated["evaluation"]
            writer.writerow(
                {
                    "pair": report["pair"],
                    "set": report["set"],
                    "fixture": report["fixture"],
                    "left": report["left"],
                    "right": report["right"],
                    "reference_class": report["reference_class"],
                    "reference_class_floors": report["reference_class_floors"],
                    "oracle_truths": report["geometry"]["truth_edges"],
                    "oracle_bestof_edges": report["geometry"]["bestof_edges"],
                    "production_bestof_current_solved": base["solved"],
                    "production_bestof_current_reason": base["reason"],
                    "production_bestof_current_class": base["right_yaw_class"],
                    "production_bestof_current_within_5deg": base["within_5deg"],
                    "production_bestof_current_rotation_error_degrees": base["rotation_error_degrees"],
                    "production_bestof_current_baseline_error_degrees": base["baseline_error_degrees_gauge_aligned"],
                    "production_bestof_reference_solved": reference["solved"],
                    "production_bestof_reference_reason": reference["reason"],
                    "production_bestof_reference_class": reference["right_yaw_class"],
                    "production_full_current_solved": full["solved"],
                    "production_full_current_reason": full["reason"],
                    "production_full_reference_solved": report["production"]["oracle_full_reference"]["solved"],
                    "production_full_reference_reason": report["production"]["oracle_full_reference"]["reason"],
                    "adjudicated_solved": adjudicated["solved"],
                    "adjudicated_reason": adjudicated["reason"],
                    "adjudicated_class": adjudicated["right_yaw_class"],
                    "adjudicated_class_correct": evaluation["class_equals_expected"],
                    "adjudicated_within_5deg": evaluation["within_5deg"],
                    "adjudicated_rotation_error_degrees": evaluation["rotation_error_degrees"],
                    "adjudicated_baseline_error_degrees": evaluation["baseline_error_degrees_gauge_aligned"],
                    "adjudicated_baseline_error_signless_degrees": evaluation["baseline_error_degrees_signless"],
                    "adjudicated_gauge_index": evaluation["gauge_index"],
                    "adjudicated_inliers": adjudicated["inliers"],
                    "adjudicated_inlier_fraction": adjudicated["inlier_fraction"],
                    "adjudicated_median_rho_degrees": adjudicated["median_rho_degrees"],
                    "adjudicated_parallax_degrees": adjudicated["parallax_degrees"],
                    "adjudicated_cheirality_fraction": adjudicated["cheirality_fraction"],
                    "adjudicated_singular_ratio": adjudicated["singular_ratio"],
                    "diagnosis": report["attribution"]["diagnosis"],
                    "production_failure_class": report["attribution"]["production_failure_class"],
                    "diagnosis_detail": report["attribution"]["detail"],
                }
            )
    runtime = time.perf_counter() - started_all
    payload["runtime_seconds"] = runtime
    payload["peak_rss_bytes"] = log.peak_rss_bytes
    payload["artifacts"] = {
        "oracle_pose_pairs.csv": {
            "bytes": csv_path.stat().st_size,
            "sha256": digest(csv_path),
        },
        "oracle_pose_run.log": {
            "note": "log is still open while the payload is written; its digest is "
            "recorded in the DS-19 report after the run closes it",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    payload["artifacts"]["oracle_pose_failures.json"] = {
        "bytes": json_path.stat().st_size,
        "sha256": digest(json_path),
        "note": "digest computed after the final write; re-verified with Get-FileHash",
    }
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    arguments = parser.parse_args(argv)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(out_dir / "oracle_pose_run.log")
    try:
        payload = run(out_dir, log)
    except MemoryBudgetExceeded as error:
        log(f"[abort ] {error}")
        log.close()
        return 4
    log(
        f"[done  ] {payload['reproduction']['production_oracle_bestof_solved']}/9 reproduced; "
        f"adjudicator threshold {payload['adjudicator']['threshold']['status']}; "
        f"runtime {payload['runtime_seconds']:.1f}s; peak RSS {_mib(payload['peak_rss_bytes'])}"
    )
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
