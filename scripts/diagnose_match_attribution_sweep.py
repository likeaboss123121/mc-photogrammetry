"""DS-16B preregistered oracle-tolerance sweep over the frozen DS-16 candidates.

This is the bounded follow-up that DS-16 section 6 preregistered, and nothing
else. It is **evaluation-only** and reference-assisted: every label it produces
comes from fixture manifests/litematics and must never enter a production path.

Fixed inputs (verified by SHA-256 before anything else happens):

* ``exports/diagnostics/ds16/match_candidates.json`` - blind phase-1 freeze.
* ``exports/diagnostics/ds16/match_candidates.npz``  - keypoints, descriptors,
  reciprocal candidate edges and DS-14 selected matches.
* ``exports/diagnostics/ds16/match_attribution.json`` - DS-16's authoritative
  reference attribution, read **only** as an independent cross-check.

Nothing is re-detected: no image is opened, no SIFT run, no DS-14/DS-15 run and
no DS-11/14/15/16 source, test, artifact or threshold is touched. The sweep
re-renders the seven T2/T6 reference frames from their manifests (a pure
function of the frozen fixtures) purely to re-label the **already frozen**
keypoints under the grid below.

Preregistered grid (12 cells, no cell is selected or tuned):

    hit distance in {0.25, 0.5, 1.0, 2.0} blocks
    x face-normal in {cos 30 deg, cos 45 deg, cos 60 deg}, render scale 0.35.

Pair set: the nine frozen T2/T6 pairs (DS-16's ``oracle_truths >= 8`` set).
The 28 ``t3-sparse-8`` pairs are excluded from every conclusion.

Boundedness (DS-16BR correction). The first DS-16B run was terminated at
3.8 GiB RSS. Two independent causes, both fixed here without touching any
threshold, gate, cell or outcome:

* DS-11's ``_robust_baseline`` (production code, deliberately unmodified) calls
  ``np.linalg.svd(rows, full_matrices=True)``. For an ``(N, 3)`` row matrix that
  returns an ``N x N`` left-singular matrix: ``8 * N**2`` bytes, so 19617 rows
  asked for 3.08 GiB in a single allocation. Every DS-11 input in this sweep is
  now deterministically bounded to ``SOLVER_ROW_CAP`` rows (134 MiB for that
  matrix at 4096 rows; the local probe measured 253 ms and a 282.7 MiB peak RSS
  for the call). The cap is the smallest power of two above the largest frozen
  frame's keypoint count (3837), so the complete per-left-keypoint best-of
  oracle cell is never subsampled. Raw oracle counts stay exact: only the
  *solver* input is capped, and each cell reports ``rows_full``, ``rows_used``
  and ``sampled``.
* The many-to-many oracle search used to accumulate every pre-dedup hit of a
  whole cell in one list before ``np.unique``. It now de-duplicates inside a
  fixed left-keypoint chunk, so its temporaries never exceed one chunk, and each
  pair's truth set is released before the next pair; no edge payload survives a
  cell. The oracle set itself is still the complete unique truth set.

A process-wide RSS watchdog reads the OS peak working set at every checkpoint
and aborts the run, writing partial evidence, instead of continuing above
``RSS_LIMIT_BYTES`` (1.5 GiB).

Per cell and pair we report, beside DS-16's many-to-many read-outs:

* exact oracle truth count, left-keypoint best-of denominator, candidate labels
  (``geometric``/``strict``/``coordinate``) and the unscored/no-hit count;
* per-left-keypoint **best-of** directed / reciprocal / selected recall and the
  unchanged many-to-many recalls;
* DS-11 ``solve_pair`` cells on ``oracle_bestof`` (one oracle edge per left
  keypoint, nearest world hit, deterministic) and ``oracle_full`` (the whole
  many-to-many truth set, deterministically stride-bounded), each under current
  (M3) and reference cameras;
* the ``candidate-intersection`` cell: reciprocal candidate graph INTERSECTED
  with the oracle truth set.

Stop rule. The DS-16B directive's written rule ("oracle ceiling <= 8/9 OR
best-of directed recall < 5% in every cell => correspondence representation is
binding") was rejected by lead review: a low oracle ceiling is evidence of pose
adjudication failure, not of correspondence representation, so the disjunction
conflated two independent mechanisms. This script reports the two predicates
**separately**, adds the preregistered decomposition (ideal-edge oracle failure
=> pose adjudication; zero best-of directed support => detection/candidate
representation; directed support present but reciprocal/selected absent =>
graph filtering/selection), and records the OR conclusion as rejected. No
threshold, gate or cell is changed and nothing is tuned either way.

Run once with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_match_attribution_sweep.py
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
from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair  # noqa: E402
from mc_photogrammetry.reconstruction.camera import load_camera_manifest  # noqa: E402
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
from diagnose_match_attribution import (  # noqa: E402
    ALIGNMENT_MEDIAN_PX_MAX,
    REFERENCE_RENDER_SCALE,
    _camera_estimate,
    _directed_topk,
    _evaluate_pose,
    _matches_from_edges,
    _selected_edge_set,
    label_keypoints,
    load_frozen,
)

DS16_DIR = ROOT / "exports" / "diagnostics" / "ds16"
OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds16b"

# DS-16 frozen digests, copied from the DS-16 report tables. Any mismatch aborts
# before a single label is produced.
EXPECTED_CANDIDATES_JSON_SHA256 = (
    "e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93"
)
EXPECTED_CANDIDATES_NPZ_SHA256 = (
    "3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4"
)
EXPECTED_ATTRIBUTION_JSON_SHA256 = (
    "72da240a1992b31640ddbfb4e4bd0b6fb6c1b02e4ede1abd6a7507b74ea7a038"
)

# Preregistered grid and bounds. Fixed here before the single run.
DISTANCES_BLOCKS = (0.25, 0.5, 1.0, 2.0)
NORMAL_DEGREES = (30.0, 45.0, 60.0)
RENDER_SCALE = REFERENCE_RENDER_SCALE  # 0.35, the DS-16 attribution scale
SELECTED_SETS = ("T2", "T6")
BASE_DISTANCE_BLOCKS = 0.5
BASE_NORMAL_DEGREES = 30.0
SWEEP_SEED_NOTE = "no random sampling: best-of is nearest-hit with index tie-break"

# DS-16BR boundedness constants (memory bounds, not outcome thresholds).
TRUTH_CHUNK = 64  # left keypoints per oracle-search chunk
SOLVER_ROW_CAP = 4096  # DS-11 rows per solve cell: 8 * N**2 = 134 MiB at 4096
SOLVER_SAMPLING_RULE = (
    "deterministic stride subsample of the lexicographically sorted unique edge set: "
    "step = ceil(rows / cap), keep edges[::step]; no RNG, identical for every cell"
)
TRUTH_MATERIALIZE_MAX_ROWS = 5_000_000  # per-pair guard; observed maximum ~2.7e5
RSS_LIMIT_BYTES = 1_500 * 1024 * 1024

CELL_KEYS = (
    "oracle_bestof_current",
    "oracle_bestof_reference",
    "oracle_full_current",
    "oracle_full_reference",
    "intersection_current",
    "intersection_reference",
)
ORACLE_CELL_KEYS = (
    "oracle_bestof_current",
    "oracle_bestof_reference",
    "oracle_full_current",
    "oracle_full_reference",
)
COMPLETE_ORACLE_KEYS = ("oracle_bestof_current", "oracle_bestof_reference")
RECALL_KEYS = ("directed", "reciprocal", "selected", "intersection")
PAIR_COUNT = 9

CELL_CSV_COLUMNS = (
    "cell", "distance_blocks", "normal_degrees", "set", "fixture", "left", "right",
    "reference_yaw_class", "oracle_truths", "left_truth_keypoints", "oracle_bestof_edges",
    "candidate_edges", "candidate_hits", "candidate_unscored", "geometric_true",
    "strict_true", "coordinate_true", "intersection_edges",
    "mm_directed", "mm_reciprocal", "mm_selected", "mm_intersection",
    "bo_directed", "bo_reciprocal", "bo_selected", "bo_intersection",
    "oracle_bestof_current_solved", "oracle_bestof_current_within_5deg",
    "oracle_bestof_reference_solved", "oracle_bestof_reference_within_5deg",
    "oracle_full_current_rows_used", "oracle_full_current_sampled",
    "oracle_full_current_solved", "oracle_full_current_within_5deg",
    "oracle_full_reference_rows_used", "oracle_full_reference_sampled",
    "oracle_full_reference_solved", "oracle_full_reference_within_5deg",
    "intersection_current_solved", "intersection_current_within_5deg",
    "intersection_reference_solved", "intersection_reference_within_5deg",
)

AGGREGATE_CSV_COLUMNS = (
    "cell", "distance_blocks", "normal_degrees", "oracle_truths", "left_truth_keypoints",
    "geometric_true", "geometric_prevalence", "intersection_edges",
    "mm_directed", "mm_reciprocal", "mm_selected", "bo_directed", "bo_reciprocal", "bo_selected",
    "oracle_bestof_current_evaluable", "oracle_bestof_current_sampled",
    "oracle_bestof_current_solved", "oracle_bestof_current_within_5deg",
    "oracle_bestof_reference_evaluable", "oracle_bestof_reference_sampled",
    "oracle_bestof_reference_solved", "oracle_bestof_reference_within_5deg",
    "oracle_full_current_evaluable", "oracle_full_current_sampled", "oracle_full_current_solved",
    "oracle_full_current_within_5deg", "oracle_full_reference_evaluable",
    "oracle_full_reference_sampled", "oracle_full_reference_solved",
    "oracle_full_reference_within_5deg",
    "intersection_current_evaluable", "intersection_current_sampled",
    "intersection_current_solved", "intersection_current_within_5deg",
    "intersection_reference_evaluable", "intersection_reference_sampled",
    "intersection_reference_solved", "intersection_reference_within_5deg",
    "peak_rss_mib",
)


class MemoryBudgetExceeded(RuntimeError):
    """The run cannot stay inside the preregistered RSS/row budget."""


class _Log:
    """Line-buffered stdout + ``sweep_run.log`` tee."""

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
                f"{RSS_LIMIT_BYTES / 2**20:.0f} MiB budget at {label}; stopping instead of continuing"
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


def _normal_cosine(degrees: float) -> float:
    return float(np.cos(np.radians(degrees)))


def _cell_id(distance: float, degrees: float) -> str:
    return f"d{distance:g}_n{degrees:g}"


def _encode_codes(edges: np.ndarray, right_count: int) -> np.ndarray:
    return edges[:, 0].astype(np.int64) * int(max(right_count, 1)) + edges[:, 1].astype(np.int64)


def _count_members(codes: np.ndarray, sorted_codes: np.ndarray) -> int:
    """Number of ``codes`` present in the sorted ``sorted_codes`` array."""
    if len(codes) == 0 or len(sorted_codes) == 0:
        return 0
    position = np.searchsorted(sorted_codes, codes)
    clipped = np.minimum(position, len(sorted_codes) - 1)
    return int(np.count_nonzero((position < len(sorted_codes)) & (sorted_codes[clipped] == codes)))


def _bounded_rows(edges: np.ndarray, cap: int = SOLVER_ROW_CAP) -> tuple[np.ndarray, bool]:
    """Deterministic stride bound on a DS-11 input; ``(rows, sampled)``."""
    rows = int(len(edges))
    if rows <= cap:
        return edges, False
    step = (rows + cap - 1) // cap
    return edges[::step], True


# --------------------------------------------------------------------------- #
# Reference relabelling (frozen keypoints, manifest render only).
# --------------------------------------------------------------------------- #
def truth_chunks(left: dict, right: dict, tolerance: float, normal_cosine: float,
                 chunk: int = TRUTH_CHUNK):
    """Yield ``(lo, hi, edges)``: the unique oracle pairs of one left-keypoint chunk.

    Same predicate as DS-16's ``_truth_edges`` (hit on both sides, world hit
    distance ``<= tolerance``, face-normal dot ``>= normal_cosine`` via a
    tolerance-sized spatial hash), but every temporary is bounded by one chunk
    of ``chunk`` left keypoints and de-duplication happens per chunk. Chunks
    partition the left keypoints, so their pair sets are disjoint: concatenating
    them yields exactly the global ``np.unique`` set, while the pre-dedup
    products (the 3.8 GiB failure mode) never exist at once.
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
        yield int(origin[0]), int(origin[-1]), np.column_stack((codes // right_count, codes % right_count))


def oracle_edges(left: dict, right: dict, tolerance: float, normal_cosine: float) -> np.ndarray:
    """Complete unique oracle truth set, assembled from bounded chunks."""
    parts: list[np.ndarray] = []
    rows = 0
    for _lo, _hi, edges in truth_chunks(left, right, tolerance, normal_cosine):
        parts.append(edges)
        rows += len(edges)
        if rows > TRUTH_MATERIALIZE_MAX_ROWS:
            raise MemoryBudgetExceeded(
                f"one pair's oracle set exceeded {TRUTH_MATERIALIZE_MAX_ROWS} rows; "
                "refusing to materialize it"
            )
    if not parts:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(parts)


def best_of_edges(truth: np.ndarray, left_world: np.ndarray, right_world: np.ndarray) -> np.ndarray:
    """One oracle edge per left keypoint: nearest world hit, index tie-break.

    Deterministic and cell-consistent. ``truth`` is already sorted by
    ``(left, right)``; the lexsort makes the retained row a pure function of the
    frozen data (no RNG).
    """
    if len(truth) == 0:
        return np.empty((0, 2), dtype=np.int64)
    distance = np.linalg.norm(left_world[truth[:, 0]] - right_world[truth[:, 1]], axis=1)
    order = np.lexsort((truth[:, 1], distance, truth[:, 0]))
    ordered = truth[order]
    first = np.ones(len(ordered), dtype=bool)
    first[1:] = ordered[1:, 0] != ordered[:-1, 0]
    return ordered[first]


def candidate_labels(left: dict, right: dict, edges: np.ndarray, tolerance: float,
                     normal_cosine: float) -> dict:
    """DS-16's geometric/strict/coordinate labels at *this cell's* tolerance."""
    count = len(edges)
    if count == 0:
        return {
            "hit": np.zeros(0, dtype=bool),
            "distance": np.empty(0, dtype=np.float64),
            "normal_dot": np.empty(0, dtype=np.float64),
            "geometric": np.zeros(0, dtype=bool),
            "strict": np.zeros(0, dtype=bool),
            "coordinate": np.zeros(0, dtype=bool),
        }
    left_i, right_j = edges[:, 0], edges[:, 1]
    hit = left["hit"][left_i] & right["hit"][right_j]
    distance = np.linalg.norm(left["world"][left_i] - right["world"][right_j], axis=1)
    normal_dot = np.einsum("ij,ij->i", left["normal"][left_i], right["normal"][right_j])
    same_block = left["block_code"][left_i] == right["block_code"][right_j]
    same_slot = left["slot"][left_i] == right["slot"][right_j]
    near = hit & (distance <= tolerance)
    return {
        "hit": hit,
        "distance": distance,
        "normal_dot": normal_dot,
        "geometric": near & (normal_dot >= normal_cosine),
        "strict": near & same_block & same_slot,
        "coordinate": near & same_block,
    }


def recall_block(truth: np.ndarray, edge_sets: dict[str, np.ndarray], right_count: int) -> dict:
    """Many-to-many and per-left-keypoint best-of recall for each edge set."""
    if len(truth) == 0:
        return {
            "denominator_many_to_many": 0,
            "denominator_best_of": 0,
            "sets": {
                name: {
                    "edges": int(len(edges)),
                    "many_to_many_numerator": 0,
                    "many_to_many": 0.0,
                    "best_of_numerator": 0,
                    "best_of": 0.0,
                }
                for name, edges in edge_sets.items()
            },
        }
    truth_codes = np.sort(_encode_codes(truth, right_count))
    left_truth = np.unique(truth[:, 0])
    sets: dict[str, dict] = {}
    for name, edges in edge_sets.items():
        if len(edges) == 0:
            sets[name] = {
                "edges": 0,
                "many_to_many_numerator": 0,
                "many_to_many": 0.0,
                "best_of_numerator": 0,
                "best_of": 0.0,
            }
            continue
        many = _count_members(_encode_codes(edges, right_count), truth_codes)
        best = _count_members(np.unique(edges[:, 0]).astype(np.int64), left_truth.astype(np.int64))
        sets[name] = {
            "edges": int(len(edges)),
            "many_to_many_numerator": int(many),
            "many_to_many": many / len(truth),
            "best_of_numerator": int(best),
            "best_of": best / len(left_truth),
        }
    return {
        "denominator_many_to_many": int(len(truth)),
        "denominator_best_of": int(len(left_truth)),
        "sets": sets,
    }


def solve_cell(left_points, right_points, edges, left_camera, right_camera, left_model,
               right_model, expected_class, left_name, right_name) -> dict:
    """One DS-11 ``solve_pair`` read-out on a deterministically bounded edge set."""
    rows_full = int(len(edges))
    bounded, sampled = _bounded_rows(edges)
    matches = _matches_from_edges(left_points, right_points, bounded)
    minimum = PairGates().min_correspondences
    blank = {
        "rows": int(len(matches)),
        "rows_full": rows_full,
        "rows_used": int(len(matches)),
        "sampled": bool(sampled),
        "row_cap": SOLVER_ROW_CAP,
        "evaluated": False,
        "skipped": None,
        "solved": False,
        "reason": "insufficient_correspondences",
        "right_yaw_class": None,
        "class_equals_expected": False,
        "within_5deg": False,
        "inliers": 0,
        "inlier_fraction": 0.0,
        "median_rho_degrees": None,
        "parallax_degrees": None,
        "cheirality_fraction": None,
        "rotation_error_degrees": None,
        "baseline_error_degrees_gauge_aligned": None,
        "baseline_error_degrees_signless": None,
        "gauge_index": None,
    }
    if len(matches) < minimum:
        return blank
    pose = solve_pair(
        left_camera,
        right_camera,
        matches,
        left_id=left_name,
        right_id=right_name,
        gates=PairGates(),
    )
    record = _evaluate_pose(pose, left_model, right_model, expected_class)
    record["rows"] = int(len(matches))
    record["rows_full"] = rows_full
    record["rows_used"] = int(len(matches))
    record["sampled"] = bool(sampled)
    record["row_cap"] = SOLVER_ROW_CAP
    record["evaluated"] = True
    record["skipped"] = None
    return record


# --------------------------------------------------------------------------- #
# Sweep.
# --------------------------------------------------------------------------- #
def run_sweep(out_dir: Path, log: _Log) -> dict:
    started_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen-input verification ----------------------------------------- #
    candidates_json = DS16_DIR / "match_candidates.json"
    attribution_json = DS16_DIR / "match_attribution.json"
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
    }
    for name, record in digests.items():
        record["ok"] = record["actual"] == record["expected"]
        if not record["ok"]:
            raise SystemExit(
                f"frozen DS-16 input {name} was modified: sha256 {record['actual']} "
                f"!= preregistered {record['expected']}"
            )
    log(f"[verify] all three DS-16 digests match; {digests['match_candidates.npz']['actual'][:12]}...")

    frozen = load_frozen(DS16_DIR)
    frozen_attribution = json.loads(attribution_json.read_text(encoding="utf-8"))
    known_attribution = {
        (row["fixture"], row["left"], row["right"]): row for row in frozen_attribution["pairs"]
    }

    pairs = [pair for pair in frozen["pairs"] if pair["set"] in SELECTED_SETS]
    if len(pairs) != PAIR_COUNT:
        raise SystemExit(
            f"selection rule expected the nine T2/T6 pairs, found {len(pairs)}; refusing to sweep"
        )
    log(f"[select] {len(pairs)} T2/T6 pairs; t3-sparse-8 excluded from every conclusion")

    # --- reference fixtures (manifests/litematics are reference-assisted) ---- #
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
    needed_frames = sorted({pair["left"] for pair in pairs} | {pair["right"] for pair in pairs})

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

    # Per-frame reference labels: one render each, released before the next
    # frame's render is built.
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
                "hit_fraction": float(np.mean(labels[name]["hit"])) if len(points) else None,
                "alignment_median_px": float(np.median(errors)) if len(errors) else None,
                "alignment_p90_px": float(np.percentile(errors, 90)) if len(errors) else None,
            }
        )
        log(f"[render] {name[:24]:24s} kp {len(points):5d} render {alignment[-1]['render_seconds']:.2f}s")
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

    # --- per-pair prepared state (no truth, no retained descriptors) -------- #
    grid = [
        (distance, degrees, _cell_id(distance, degrees), _normal_cosine(degrees))
        for distance in DISTANCES_BLOCKS
        for degrees in NORMAL_DEGREES
    ]
    base_cell = _cell_id(BASE_DISTANCE_BLOCKS, BASE_NORMAL_DEGREES)

    prepared: list[dict] = []
    for pair in pairs:
        fixture = fixtures[pair["fixture"]]
        left_name, right_name = pair["left"], pair["right"]
        left_record, right_record = frames[left_name], frames[right_name]
        left_points = np.asarray(arrays[f"feat_points_{left_record['key']}"], dtype=np.float64)
        right_points = np.asarray(arrays[f"feat_points_{right_record['key']}"], dtype=np.float64)
        left_descriptors = np.asarray(arrays[f"feat_desc_{left_record['key']}"], dtype=np.float64)
        right_descriptors = np.asarray(arrays[f"feat_desc_{right_record['key']}"], dtype=np.float64)
        candidate_edges = np.asarray(arrays[f"cand_{pair['index']}"], dtype=np.float64).reshape(-1, 3)
        edges = candidate_edges[:, :2].astype(np.int64)
        selected = np.asarray(arrays[f"sel_{pair['index']}"], dtype=np.float64).reshape(-1, 4)

        left_est = _estimate_from_record(pair["left_estimate"])
        right_est = _estimate_from_record(pair["right_estimate"])
        left_model, right_model = fixture["cameras"][left_name], fixture["cameras"][right_name]
        width, height = left_record["width"], left_record["height"]
        reference_class, reference_floors = _reference_class(left_est, right_est, left_model, right_model)

        directed = _directed_topk(
            left_descriptors, right_descriptors, frozen["settings"]["joint_matcher"]["neighbors"]
        )
        directed_edges = np.asarray(
            [
                (origin, int(target))
                for origin in range(directed.shape[0])
                for target in directed[origin]
                if target >= 0
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        selected_set = _selected_edge_set(left_points, right_points, selected)
        selected_edges = (
            np.asarray(sorted(selected_set), dtype=np.int64).reshape(-1, 2)
            if selected_set
            else np.empty((0, 2), dtype=np.int64)
        )

        prepared.append(
            {
                "pair": pair,
                "fixture": fixture,
                "left_name": left_name,
                "right_name": right_name,
                "left_points": left_points,
                "right_points": right_points,
                "edges": edges,
                "left_label": labels[left_name],
                "right_label": labels[right_name],
                "left_est": left_est,
                "right_est": right_est,
                "left_model": left_model,
                "right_model": right_model,
                "width": width,
                "height": height,
                "reference_class": reference_class,
                "reference_floors": reference_floors,
                "current_left": to_camera_estimate(left_est),
                "current_right": to_camera_estimate(right_est),
                "reference_left": _camera_estimate(left_model, width, height),
                "reference_right": _camera_estimate(right_model, width, height),
                "directed_edges": directed_edges,
                "selected_edges": selected_edges,
            }
        )
        # Descriptors are only needed for the directed top-k above; do not retain
        # them for the whole sweep.
        del left_descriptors, right_descriptors, directed, selected, selected_set, candidate_edges

    # --- base-cell cross-check against frozen DS-16 (bounded, exact counts) - #
    cross_check: list[dict] = []
    for item in prepared:
        pair = item["pair"]
        truth = oracle_edges(
            item["left_label"], item["right_label"], BASE_DISTANCE_BLOCKS, _normal_cosine(BASE_NORMAL_DEGREES)
        )
        labels_base = candidate_labels(
            item["left_label"], item["right_label"], item["edges"],
            BASE_DISTANCE_BLOCKS, _normal_cosine(BASE_NORMAL_DEGREES),
        )
        reference = known_attribution.get((pair["fixture"], pair["left"], pair["right"]))
        record = {
            "fixture": pair["fixture"],
            "left": pair["left"],
            "right": pair["right"],
            "recomputed_oracle_truths": int(len(truth)),
            "ds16_oracle_truths": None if reference is None else reference["candidate_stats"]["oracle_truths"],
            "recomputed_geometric_true": int(np.count_nonzero(labels_base["geometric"])),
            "ds16_geometric_true": None if reference is None else reference["candidate_stats"]["geometric_true"],
            "recomputed_hits": int(np.count_nonzero(labels_base["hit"])),
            "ds16_hits": None if reference is None else reference["candidate_stats"]["hits"],
        }
        record["oracle_truths_match"] = record["recomputed_oracle_truths"] == record["ds16_oracle_truths"]
        record["geometric_true_match"] = record["recomputed_geometric_true"] == record["ds16_geometric_true"]
        record["hits_match"] = record["recomputed_hits"] == record["ds16_hits"]
        cross_check.append(record)
        del truth, labels_base
        log.checkpoint(f"cross-check {pair['left'][:12]}->{pair['right'][:12]}")
    bad = [r for r in cross_check if not (r["oracle_truths_match"] and r["geometric_true_match"] and r["hits_match"])]
    if bad:
        raise SystemExit(
            "blocker: the sweep does not reproduce the frozen DS-16 base cell; "
            f"first mismatch {json.dumps(bad[0], default=plain)}"
        )
    log(f"[verify] base cell {base_cell} reproduces all {len(prepared)} frozen DS-16 pairs exactly")

    # --- grid evaluation: one pair/cell at a time, streamed to CSV ---------- #
    cells: list[dict] = []
    truth_seconds = 0.0
    solve_seconds = 0.0
    sampled_cells: Counter = Counter()
    cell_path = out_dir / "oracle_sweep_cells.csv"
    aggregate_path = out_dir / "oracle_sweep_aggregate.csv"
    try:
        with cell_path.open("w", newline="", encoding="utf-8") as cell_handle, \
                aggregate_path.open("w", newline="", encoding="utf-8") as aggregate_handle:
            cell_writer = csv.DictWriter(cell_handle, fieldnames=CELL_CSV_COLUMNS)
            cell_writer.writeheader()
            aggregate_writer = csv.DictWriter(aggregate_handle, fieldnames=AGGREGATE_CSV_COLUMNS)
            aggregate_writer.writeheader()
            for distance, degrees, cell_id, normal_cosine in grid:
                log(f"[cell  ] {cell_id} distance {distance} normal cos{degrees:g} ...")
                pair_rows: list[dict] = []
                for item in prepared:
                    pair = item["pair"]
                    left_world = item["left_label"]["world"]
                    right_world = item["right_label"]["world"]
                    right_count = len(item["right_points"])

                    started = time.perf_counter()
                    truth = oracle_edges(item["left_label"], item["right_label"], distance, normal_cosine)
                    truth_seconds += time.perf_counter() - started
                    if len(truth) and not np.all(np.diff(truth[:, 0]) >= 0):
                        raise MemoryBudgetExceeded("chunked oracle assembly produced unsorted pairs")

                    labels_cell = candidate_labels(
                        item["left_label"], item["right_label"], item["edges"], distance, normal_cosine
                    )
                    best = best_of_edges(truth, left_world, right_world)
                    graph_codes = _encode_codes(item["edges"], right_count)
                    intersection = (
                        truth[np.isin(_encode_codes(truth, right_count), graph_codes)]
                        if len(truth)
                        else np.empty((0, 2), dtype=np.int64)
                    )
                    recall = recall_block(
                        truth,
                        {
                            "directed": item["directed_edges"],
                            "reciprocal": item["edges"],
                            "selected": item["selected_edges"],
                            "intersection": intersection,
                        },
                        right_count,
                    )
                    started = time.perf_counter()
                    cells_record = {
                        "oracle_bestof_current": solve_cell(
                            item["left_points"], item["right_points"], best, item["current_left"],
                            item["current_right"], item["left_model"], item["right_model"],
                            item["reference_class"], item["left_name"], item["right_name"],
                        ),
                        "oracle_bestof_reference": solve_cell(
                            item["left_points"], item["right_points"], best, item["reference_left"],
                            item["reference_right"], item["left_model"], item["right_model"],
                            0, item["left_name"], item["right_name"],
                        ),
                        "intersection_current": solve_cell(
                            item["left_points"], item["right_points"], intersection, item["current_left"],
                            item["current_right"], item["left_model"], item["right_model"],
                            item["reference_class"], item["left_name"], item["right_name"],
                        ),
                        "intersection_reference": solve_cell(
                            item["left_points"], item["right_points"], intersection, item["reference_left"],
                            item["reference_right"], item["left_model"], item["right_model"],
                            0, item["left_name"], item["right_name"],
                        ),
                        "oracle_full_current": solve_cell(
                            item["left_points"], item["right_points"], truth, item["current_left"],
                            item["current_right"], item["left_model"], item["right_model"],
                            item["reference_class"], item["left_name"], item["right_name"],
                        ),
                        "oracle_full_reference": solve_cell(
                            item["left_points"], item["right_points"], truth, item["reference_left"],
                            item["reference_right"], item["left_model"], item["right_model"],
                            0, item["left_name"], item["right_name"],
                        ),
                    }
                    solve_seconds += time.perf_counter() - started
                    for key, record in cells_record.items():
                        if record["sampled"]:
                            sampled_cells[key] += 1
                    hits = int(np.count_nonzero(labels_cell["hit"]))
                    row = {
                        "cell": cell_id,
                        "distance_blocks": distance,
                        "normal_degrees": degrees,
                        "set": pair["set"],
                        "fixture": pair["fixture"],
                        "left": item["left_name"],
                        "right": item["right_name"],
                        "reference_yaw_class": item["reference_class"],
                        "oracle_truths": int(len(truth)),
                        "left_truth_keypoints": int(recall["denominator_best_of"]),
                        "oracle_bestof_edges": int(len(best)),
                        "candidate_edges": int(len(item["edges"])),
                        "candidate_hits": hits,
                        "candidate_unscored": int(len(item["edges"]) - hits),
                        "geometric_true": int(np.count_nonzero(labels_cell["geometric"])),
                        "strict_true": int(np.count_nonzero(labels_cell["strict"])),
                        "coordinate_true": int(np.count_nonzero(labels_cell["coordinate"])),
                        "intersection_edges": int(len(intersection)),
                        "recall": recall,
                        "cells": cells_record,
                    }
                    pair_rows.append(row)
                    cell_writer.writerow(_cell_csv_row(row))
                    cell_handle.flush()
                    # Release every per-pair payload before the next pair.
                    del truth, best, intersection, labels_cell, graph_codes, cells_record
                    log.checkpoint(f"{cell_id} {item['left_name'][:12]}->{item['right_name'][:12]}")
                aggregate = _aggregate(pair_rows)
                cell_record = {
                    "cell": cell_id,
                    "distance_blocks": distance,
                    "normal_degrees": degrees,
                    "normal_cosine": normal_cosine,
                    "render_scale": RENDER_SCALE,
                    "aggregate": aggregate,
                    "pairs": pair_rows,
                }
                cells.append(cell_record)
                aggregate_writer.writerow(_aggregate_csv_row(cell_record, log.peak_rss_bytes))
                aggregate_handle.flush()
                log(
                    f"[cell  ] {cell_id:10s} truths {aggregate['oracle_truths']:8d} "
                    f"best-of dir {aggregate['recall']['directed']['best_of']:.5f} "
                    f"oracle solved ceiling info: bestof {aggregate['cells']['oracle_bestof_current']['solved']}/"
                    f"{aggregate['cells']['oracle_bestof_reference']['solved']} "
                    f"full {aggregate['cells']['oracle_full_current']['solved']}/"
                    f"{aggregate['cells']['oracle_full_reference']['solved']} "
                    f"rss {_mib(log.peak_rss_bytes)}"
                )
    except MemoryBudgetExceeded as error:
        partial = {
            "schema": "ds16b-oracle-sweep-1",
            "phase": "reference_assisted_oracle_tolerance_sweep",
            "status": "aborted_memory_budget",
            "aborted": {
                "reason": str(error),
                "rss_limit_bytes": RSS_LIMIT_BYTES,
                "peak_rss_bytes": log.peak_rss_bytes,
                "cells_completed": [cell["cell"] for cell in cells],
            },
            "frozen_inputs": digests,
            "grid": _grid_payload(grid, base_cell),
            "cells": cells,
        }
        partial_path = out_dir / "match_attribution_sweep.aborted.json"
        partial_path.write_text(json.dumps(partial, indent=2, default=plain), encoding="utf-8")
        log(f"[abort ] partial evidence written to {partial_path.relative_to(ROOT)}")
        raise SystemExit(4)

    stop_rule = _stop_rule(cells)
    log(f"[stop  ] {json.dumps(stop_rule, default=plain)}")

    payload = {
        "schema": "ds16b-oracle-sweep-1",
        "phase": "reference_assisted_oracle_tolerance_sweep",
        "status": "complete",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "frozen_inputs": digests,
        "frozen_candidates": {
            "json": frozen["_path"],
            "json_sha256": frozen["_digest"],
            "npz": frozen["candidates_npz"]["path"],
            "npz_sha256": frozen["candidates_npz"]["sha256"],
            "canonical_predictions_digest": frozen["canonical_predictions_digest"],
            "ds15_cross_check": frozen["ds15_cross_check"],
        },
        "reference_inputs": {
            fixture: {
                key: entry[key]
                for key in ("litematic", "litematic_sha256", "manifest", "manifest_sha256", "scene", "scene_sha256")
            }
            for fixture, entry in fixtures.items()
        },
        "selection": {
            "rule": (
                "frozen pairs whose set is T2 or T6 and whose recomputed oracle truth count at the "
                "DS-16 base cell (0.5 blocks, cos 30 deg) is >= 8"
            ),
            "sets_included": list(SELECTED_SETS),
            "sets_excluded": ["t3-sparse-8"],
            "t3_excluded_from_conclusions": True,
            "pairs": [
                {
                    "index": item["pair"]["index"],
                    "set": item["pair"]["set"],
                    "fixture": item["pair"]["fixture"],
                    "left": item["left_name"],
                    "right": item["right_name"],
                    "ds16_attribution_label": known_attribution[
                        (item["pair"]["fixture"], item["left_name"], item["right_name"])
                    ]["attribution"]["label"],
                }
                for item in prepared
            ],
            "count": len(prepared),
        },
        "grid": _grid_payload(grid, base_cell),
        "best_of_definition": (
            "per left keypoint: the oracle-compatible right keypoint with the smallest world-hit "
            "distance, ties broken by right index; " + SWEEP_SEED_NOTE
        ),
        "mask_and_hit_rules": (
            "unchanged from DS-16: GUI-mask-excluded pixels and reference no-hit pixels are "
            "unscored (hit=false); oracle truths, candidate labels and every recall denominator "
            "are computed only over hit keypoints"
        ),
        "boundedness": {
            "solver_row_cap": SOLVER_ROW_CAP,
            "solver_sampling_rule": SOLVER_SAMPLING_RULE,
            "solver_cap_rationale": (
                "DS-11 _robust_baseline calls np.linalg.svd(rows, full_matrices=True), whose "
                "left-singular matrix is rows x rows float64 = 8*rows**2 bytes; 19617 rows needs "
                "3.08 GiB. 4096 rows needs 134 MiB for that matrix (253 ms, 282.7 MiB measured peak "
                "RSS for the call), and is the smallest power of two above the largest frozen frame's "
                "keypoint count (3837), so the complete per-left best-of oracle cell is never "
                "subsampled. Raw oracle counts and both recall denominators stay exact."
            ),
            "truth_chunk_left_keypoints": TRUTH_CHUNK,
            "truth_materialize_max_rows": TRUTH_MATERIALIZE_MAX_ROWS,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "peak_rss_bytes": log.peak_rss_bytes,
            "sampled_cells": {key: int(count) for key, count in sorted(sampled_cells.items())},
        },
        "alignment": alignment,
        "alignment_median_px": alignment_median,
        "ds16_base_cross_check": {
            "path": digests["match_attribution.json"]["path"],
            "sha256": digests["match_attribution.json"]["actual"],
            "base_cell": base_cell,
            "pairs": cross_check,
            "mismatch_count": sum(
                1
                for row in cross_check
                if not (row["oracle_truths_match"] and row["geometric_true_match"] and row["hits_match"])
            ),
        },
        "cells": cells,
        "stop_rule": stop_rule,
        "runtime_seconds": {
            "render_and_label": sum(row["render_seconds"] for row in alignment),
            "oracle_truth_search": truth_seconds,
            "ds11_solves": solve_seconds,
            "total": time.perf_counter() - started_all,
        },
        "note": (
            "Evaluation only, reference-assisted. The candidate graph, keypoints, descriptors and "
            "DS-14 selections are the frozen DS-16 bytes; nothing was re-detected and no DS-11/14/15/16 "
            "source, test, artifact or threshold was modified. No cell is selected as best and no "
            "threshold is tuned. T3 is excluded from every conclusion."
        ),
        "recipe": " ".join(sys.argv),
    }
    json_path = out_dir / "match_attribution_sweep.json"
    payload["_path"] = str(json_path.relative_to(ROOT))
    json_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    payload["_digest"] = digest(json_path)
    log(f"[write ] {payload['_path']} sha256 {payload['_digest']}")
    log(
        f"[write ] runtime render {payload['runtime_seconds']['render_and_label']:.1f}s "
        f"truth {payload['runtime_seconds']['oracle_truth_search']:.1f}s "
        f"solve {payload['runtime_seconds']['ds11_solves']:.1f}s "
        f"total {payload['runtime_seconds']['total']:.1f}s peak rss {_mib(log.peak_rss_bytes)}"
    )
    return payload


def _mib(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 2**20:.1f} MiB"


def _grid_payload(grid: list[tuple], base_cell: str) -> dict:
    return {
        "distances_blocks": list(DISTANCES_BLOCKS),
        "normal_degrees": list(NORMAL_DEGREES),
        "normal_cosines": [_normal_cosine(value) for value in NORMAL_DEGREES],
        "render_scale": RENDER_SCALE,
        "cells": len(grid),
        "base_cell": base_cell,
        "solver_row_cap": SOLVER_ROW_CAP,
        "solver_sampling_rule": SOLVER_SAMPLING_RULE,
        "bound_note": (
            "DS-11 solve_pair takes a full-matrix SVD over its informative rows, so the many-to-many "
            "oracle cell is deterministically stride-subsampled down to the row cap; raw oracle "
            "counts stay exact and each cell reports rows_full/rows_used/sampled. The best-of oracle "
            "cell (one edge per left keypoint) is the complete oracle read-out."
        ),
    }


def _aggregate(pair_rows: list[dict]) -> dict:
    oracle_truths = sum(row["oracle_truths"] for row in pair_rows)
    left_truth = sum(row["left_truth_keypoints"] for row in pair_rows)
    recall: dict[str, dict] = {}
    for name in RECALL_KEYS:
        many_num = sum(row["recall"]["sets"][name]["many_to_many_numerator"] for row in pair_rows)
        best_num = sum(row["recall"]["sets"][name]["best_of_numerator"] for row in pair_rows)
        recall[name] = {
            "many_to_many_numerator": many_num,
            "many_to_many": many_num / oracle_truths if oracle_truths else None,
            "best_of_numerator": best_num,
            "best_of": best_num / left_truth if left_truth else None,
        }
    cells: dict[str, dict] = {}
    for key in CELL_KEYS:
        records = [row["cells"][key] for row in pair_rows]
        evaluated = [record for record in records if record["evaluated"]]
        solved = [record for record in evaluated if record["solved"]]
        within = [record for record in solved if record["within_5deg"]]
        rotation = [record["rotation_error_degrees"] for record in solved if record["rotation_error_degrees"] is not None]
        baseline = [
            record["baseline_error_degrees_gauge_aligned"]
            for record in solved
            if record["baseline_error_degrees_gauge_aligned"] is not None
        ]
        cells[key] = {
            "pairs": len(records),
            "evaluable": len(evaluated),
            "skipped": sum(1 for record in records if record["skipped"]),
            "sampled_pairs": sum(1 for record in records if record["sampled"]),
            "cap": SOLVER_ROW_CAP,
            "rows_full_total": sum(record["rows_full"] for record in records),
            "solved": len(solved),
            "within_5deg": len(within),
            "precision_within_5deg": len(within) / len(solved) if solved else None,
            "median_rotation_error_degrees": float(np.median(rotation)) if rotation else None,
            "median_baseline_error_degrees_gauge_aligned": float(np.median(baseline)) if baseline else None,
            "reason_counts": dict(
                Counter(record["reason"] for record in evaluated if not record["solved"])
            ),
            "rows_total": sum(record["rows"] for record in records),
        }
    hits = sum(row["candidate_hits"] for row in pair_rows)
    return {
        "pairs": len(pair_rows),
        "oracle_truths": oracle_truths,
        "left_truth_keypoints": left_truth,
        "oracle_bestof_edges": sum(row["oracle_bestof_edges"] for row in pair_rows),
        "candidate_edges": sum(row["candidate_edges"] for row in pair_rows),
        "candidate_hits": hits,
        "candidate_unscored": sum(row["candidate_unscored"] for row in pair_rows),
        "geometric_true": sum(row["geometric_true"] for row in pair_rows),
        "strict_true": sum(row["strict_true"] for row in pair_rows),
        "coordinate_true": sum(row["coordinate_true"] for row in pair_rows),
        "geometric_prevalence": (
            sum(row["geometric_true"] for row in pair_rows) / hits if hits else None
        ),
        "intersection_edges": sum(row["intersection_edges"] for row in pair_rows),
        "recall": recall,
        "cells": cells,
    }


def _pair_key(row: dict) -> list:
    return [row["fixture"], row["left"], row["right"]]


def _stop_rule(cells: list[dict]) -> dict:
    """Report the two rejected-disjunction predicates separately, plus decomposition."""
    oracle_ceiling = 0
    ceiling_cells: list[str] = []
    complete_ceiling = 0
    complete_cells: list[str] = []
    sampled_cells: list[dict] = []
    for cell in cells:
        for key in ORACLE_CELL_KEYS:
            record = cell["aggregate"]["cells"][key]
            if record["solved"] > oracle_ceiling:
                oracle_ceiling = record["solved"]
                ceiling_cells = [f"{cell['cell']}:{key}"]
            elif record["solved"] == oracle_ceiling:
                ceiling_cells.append(f"{cell['cell']}:{key}")
            if key in COMPLETE_ORACLE_KEYS:
                if record["solved"] > complete_ceiling:
                    complete_ceiling = record["solved"]
                    complete_cells = [f"{cell['cell']}:{key}"]
                elif record["solved"] == complete_ceiling:
                    complete_cells.append(f"{cell['cell']}:{key}")
            if record["sampled_pairs"]:
                sampled_cells.append(
                    {
                        "cell": cell["cell"],
                        "key": key,
                        "sampled_pairs": record["sampled_pairs"],
                        "rows_full_total": record["rows_full_total"],
                        "rows_used_total": record["rows_total"],
                    }
                )
    directed_by_cell = {
        cell["cell"]: cell["aggregate"]["recall"]["directed"]["best_of"] for cell in cells
    }
    directed_values = [value for value in directed_by_cell.values() if value is not None]
    predicate_1 = bool(cells) and oracle_ceiling <= PAIR_COUNT - 1
    predicate_2 = bool(directed_values) and max(directed_values) < 0.05

    adjudication: Counter = Counter()
    representation: Counter = Counter()
    filtering: Counter = Counter()
    adjudication_pairs: dict[tuple, list[str]] = {}
    representation_pairs: dict[tuple, list[str]] = {}
    filtering_pairs: dict[tuple, list[str]] = {}
    for cell in cells:
        for row in cell["pairs"]:
            key = (row["fixture"], row["left"], row["right"])
            if not row["cells"]["oracle_bestof_reference"]["solved"]:
                adjudication[cell["cell"]] += 1
                adjudication_pairs.setdefault(key, []).append(cell["cell"])
            best_of_directed = row["recall"]["sets"]["directed"]["best_of"]
            best_of_reciprocal = row["recall"]["sets"]["reciprocal"]["best_of"]
            best_of_selected = row["recall"]["sets"]["selected"]["best_of"]
            if best_of_directed <= 0.0:
                representation[cell["cell"]] += 1
                representation_pairs.setdefault(key, []).append(cell["cell"])
            elif best_of_reciprocal <= 0.0 and best_of_selected <= 0.0:
                filtering[cell["cell"]] += 1
                filtering_pairs.setdefault(key, []).append(cell["cell"])

    def _tally(counter: Counter, grouped: dict) -> dict:
        return {
            "pairs_by_cell": {cell["cell"]: int(counter[cell["cell"]]) for cell in cells},
            "pair_union_count": len(grouped),
            "pairs_in_union": [
                {"fixture": key[0], "left": key[1], "right": key[2], "cells": sorted(value)}
                for key, value in sorted(grouped.items())
            ],
        }

    original_verdict = (
        "correspondence_representation_binding"
        if (predicate_1 or predicate_2)
        else "not_binding_within_this_grid"
    )
    return {
        "corrected_by": "DS-16BR lead review",
        "original_combined_conclusion": {
            "formula": (
                "oracle solved ceiling <= 8/9 OR aggregate best-of directed recall < 5% in every "
                "cell => correspondence representation is binding"
            ),
            "as_written_verdict": original_verdict,
            "status": "rejected_by_lead_review",
            "reason": (
                "a low oracle ceiling is evidence of pose adjudication failure (ideal edges + true "
                "cameras still do not produce a pose), not of correspondence representation, so the "
                "disjunction conflated two independent mechanisms and its OR conclusion is not "
                "reported as a single verdict here"
            ),
        },
        "predicate_1_oracle_solved_ceiling": {
            "definition": "maximum solved count over the four oracle cells <= 8 of 9 pairs",
            "holds": predicate_1,
            "oracle_solved_ceiling": oracle_ceiling,
            "pairs": PAIR_COUNT,
            "cells_at_ceiling": ceiling_cells,
            "complete_bestof_ceiling": complete_ceiling,
            "complete_bestof_cells_at_ceiling": complete_cells,
            "diagnoses": "pose_adjudication",
            "reading": (
                "oracle-consistent edges (complete per-left best-of; sampled many-to-many where the "
                "row cap applied) fail typed DS-11 gates under current and reference cameras"
            ),
        },
        "predicate_2_bestof_directed_recall": {
            "definition": "aggregate per-left best-of directed recall < 5% in every grid cell",
            "holds": predicate_2,
            "threshold": 0.05,
            "min": min(directed_values) if directed_values else None,
            "max": max(directed_values) if directed_values else None,
            "by_cell": directed_by_cell,
            "diagnoses": "detection_or_candidate_representation",
            "reading": (
                "the directed top-k candidate graph carries almost no oracle-consistent edge for any "
                "left keypoint"
            ),
        },
        "decomposition": {
            "rule": (
                "per pair and cell: oracle_bestof_reference unsolved with complete ideal edges => "
                "pose adjudication; best-of directed recall numerator 0 => detection/candidate "
                "representation; directed support > 0 but reciprocal and selected support 0 => graph "
                "filtering/selection"
            ),
            "pose_adjudication": _tally(adjudication, adjudication_pairs),
            "detection_or_candidate_representation": _tally(representation, representation_pairs),
            "graph_filtering_or_selection": _tally(filtering, filtering_pairs),
        },
        "sampled_solver_cells": sampled_cells,
        "action": "do not tune matcher gates; report the decomposition",
    }


def _cell_csv_row(row: dict) -> dict:
    recall = row["recall"]["sets"]
    cells = row["cells"]
    values = {
        "cell": row["cell"],
        "distance_blocks": row["distance_blocks"],
        "normal_degrees": row["normal_degrees"],
        "set": row["set"],
        "fixture": row["fixture"],
        "left": row["left"],
        "right": row["right"],
        "reference_yaw_class": row["reference_yaw_class"],
        "oracle_truths": row["oracle_truths"],
        "left_truth_keypoints": row["left_truth_keypoints"],
        "oracle_bestof_edges": row["oracle_bestof_edges"],
        "candidate_edges": row["candidate_edges"],
        "candidate_hits": row["candidate_hits"],
        "candidate_unscored": row["candidate_unscored"],
        "geometric_true": row["geometric_true"],
        "strict_true": row["strict_true"],
        "coordinate_true": row["coordinate_true"],
        "intersection_edges": row["intersection_edges"],
    }
    for name in RECALL_KEYS:
        values[f"mm_{name}"] = recall[name]["many_to_many"]
        values[f"bo_{name}"] = recall[name]["best_of"]
    for key in CELL_KEYS:
        values[f"{key}_solved"] = cells[key]["solved"]
        values[f"{key}_within_5deg"] = cells[key]["within_5deg"]
    values["oracle_full_current_rows_used"] = cells["oracle_full_current"]["rows"]
    values["oracle_full_current_sampled"] = cells["oracle_full_current"]["sampled"]
    values["oracle_full_reference_rows_used"] = cells["oracle_full_reference"]["rows"]
    values["oracle_full_reference_sampled"] = cells["oracle_full_reference"]["sampled"]
    return values


def _aggregate_csv_row(cell: dict, peak_rss_bytes: int | None) -> dict:
    aggregate = cell["aggregate"]
    row = {
        "cell": cell["cell"],
        "distance_blocks": cell["distance_blocks"],
        "normal_degrees": cell["normal_degrees"],
        "oracle_truths": aggregate["oracle_truths"],
        "left_truth_keypoints": aggregate["left_truth_keypoints"],
        "geometric_true": aggregate["geometric_true"],
        "geometric_prevalence": aggregate["geometric_prevalence"],
        "intersection_edges": aggregate["intersection_edges"],
    }
    for name in RECALL_KEYS[:3]:
        row[f"mm_{name}"] = aggregate["recall"][name]["many_to_many"]
        row[f"bo_{name}"] = aggregate["recall"][name]["best_of"]
    for key in CELL_KEYS:
        row[f"{key}_evaluable"] = aggregate["cells"][key]["evaluable"]
        row[f"{key}_sampled"] = aggregate["cells"][key]["sampled_pairs"]
        row[f"{key}_solved"] = aggregate["cells"][key]["solved"]
        row[f"{key}_within_5deg"] = aggregate["cells"][key]["within_5deg"]
    row["peak_rss_mib"] = None if peak_rss_bytes is None else round(peak_rss_bytes / 2**20, 1)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(out_dir / "sweep_run.log")
    try:
        run_sweep(out_dir, log)
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
