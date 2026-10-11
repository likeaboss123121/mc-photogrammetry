"""Evaluator-only replay reader: sealed E1 patch labels vs production stage-trace v2.

Joins a sealed E1 patch-label artifact
(``mc-photogrammetry-texture-patch-labels-v1``, written by
``scripts/evaluation/evaluate_texture_patch_labels.py``) to a production stage
trace (``schema == 2``, written by ``raycast.reconstruct_scene_surfaces`` when a
caller passes ``stage_trace``). The E1 artifact carries reference-assisted
``(frame, coordinate, slot)`` truth labels and the exact rectified production
patches; the trace carries the reference-agnostic observations, their final
scored cost vectors, the applied effective weights and the per-coordinate
aggregation.

The module parses two JSON documents, validates both fail-closed, joins them and
returns one JSON-safe summary. An optional third document - the E2C production
patch sidecar (``mc-photogrammetry-texture-patch-sidecar-v1``) - can be supplied
with ``--sidecar``; it is validated against the trace's exact file bytes,
ordinals, identities and digests and then joined to E1 so raw-patch,
observed-patch, valid-mask and coverage parity each get their own denominator.
Without ``--sidecar`` the output is unchanged. The reader reads no screenshot,
reference litematic, camera manifest, mask or JAR, runs no benchmark or corpus
evaluation, writes no artifact and changes no production behaviour, threshold,
palette or gate.

Identity contract
-----------------

The trace digests ``FaceSample.observed_patch`` (invalid pixels replaced by the
coverage-weighted ``FaceSample.rgb`` in float32), while E1 stores the raw float32
patch plus ``valid`` and ``coverage``. For a like-for-like comparison the reader
reconstructs the production float32 observed patch from the E1 raw arrays using
production's own arithmetic - it does not reuse E1's ``_observed_patch``, which
deliberately promotes to float64 for its weight calculation. For all-valid
patches the raw and observed arrays coincide. Both comparisons are reported, and
the digest contract is production's
``sha256(f"{shape}|{dtype.str}|" + C-order bytes)``.

Face-patch join
---------------

Only observations whose ``source`` is ``resolved_face`` carry a rectified face
patch and can be joined to E1's ``(frame, coordinate, slot)`` patch keys. A
``centre_patch_fallback`` observation may share an E1 key, but it is not a face
patch, so patch-key intersection, digest parity and the per-observation truth
rank are restricted to ``resolved_face``. Fallback observations are counted
separately as production-only evidence, including any key collision with an E1
face key, and are never counted as E1 face joins. Fallback evidence still feeds
the production coordinate trace and therefore the per-cell outcomes, whose
denominator is stated explicitly. ``load_trace`` enforces the digest half of this
contract on input: ``observed_patch``, ``valid``, ``coverage`` and ``pattern``
must be null for a ``centre_patch_fallback`` observation, and a non-null value is
refused.

Duplicate identities
--------------------

``raycast`` retains at most one resolved-face observation per
``(frame_id, coordinate, slot)`` per frame through its ``seen_here`` set, its
centre-patch fallback only runs for coordinates no resolved face covered, and E1
``deduplicate_samples`` keeps one patch per key. Duplicated observation
identities are therefore a source contradiction and are refused, never collapsed.

Denominators
------------

Every rate carries ``{numerator, denominator, rate}`` and an empty denominator
yields ``rate: null``. The visible-cell total is never inferred from patch rows:
``per_cell`` is explicitly the E1-labeled coordinate subset and the E1 artifact's
own ``denominators`` mapping is carried through untouched. Results are
conditional on the supplied cameras and reference geometry the E1 artifact was
produced with; this is not an image-only product score.

Optional production patch sidecar
---------------------------------

`--sidecar` adds the E2C resolved-face payload document. It is refused unless it
declares ``mc-photogrammetry-texture-patch-sidecar-v1``, trace schema 2, a
SHA-256 and byte count matching the supplied trace file exactly, and header
counts that reconcile to the trace (``observation_count``,
``resolved_face_records``, ``excluded_fallback_observations``). Exactly one row
per ``resolved_face`` trace observation is required; each row's ordinal,
identity and ``source`` must agree with the trace observation at that ordinal,
and its recomputed ``observed_patch``/``valid``/``coverage`` digests must equal
the trace's. Payload arrays are decoded at E2C's frozen shape/dtype and every
digest is recomputed from the decoded bytes. The fixed E2C bounds (10,000
records, 64 MiB) are enforced before decoding. ``raw_patch`` has no trace
counterpart by design. The sidecar is then joined to E1 by exact
``(frame_id, coordinate, slot)`` and raw-patch, observed-patch, valid-mask and
coverage parity are reported over that join with separate identity-join,
E1-only and sidecar-only counts: an identity match is never reported as a pixel
match.

Numeric similarity (E2F)
------------------------

Beside the exact digest parity, ``_reconcile_sidecar`` reports a bounded,
deterministic per-row numeric comparison of the same joined decoded arrays:
raw-patch and reconstructed-observed RGB MAE over all pixels and over the
intersection valid in both, coverage MAE over all pixels, and the fraction of
pixels whose valid masks agree. Per-row metrics are aggregated with
min/median/p90/p99/max and explicit row/pixel denominators; empty both-valid
intersections are counted and their intersection MAE is serialized as null. A
shape mismatch is refused before aggregation. Tolerant RGB similarity is not
semantic parity: it never replaces the exact digest parity section, the E1 truth
rank or any production gate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # direct-script execution without PYTHONPATH=src
    sys.path.insert(0, str(_SRC))

from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS  # noqa: E402

E1_SCHEMA = "mc-photogrammetry-texture-patch-labels-v1"
TRACE_SCHEMA = 2
REPLAY_SCHEMA = "mc-photogrammetry-texture-observation-replay-v1"
MODE = "assisted/supplied-camera evaluation"
ADMITTED_ROLES = ("template-2", "template-6")

PATCH_SIZE = 16
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE, 3)
MASK_SHAPE = (PATCH_SIZE, PATCH_SIZE)
PATCH_DTYPE = np.dtype(np.float32).str
MASK_DTYPE = np.dtype(bool).str
COVERAGE_DTYPE = np.dtype(np.float32).str

PRODUCTION_TOP_K = 8
OBSERVATION_SOURCES = ("resolved_face", "centre_patch_fallback")
MAX_EXAMPLES = 10
"""How many example rows the mismatch/top-K/rank lists keep; totals stay exact."""

REPLAY_RTOL = 1e-9
REPLAY_ATOL = 1e-12

SIDECAR_SCHEMA = "mc-photogrammetry-texture-patch-sidecar-v1"
SIDECAR_MAX_RECORDS = 10_000
"""E2C's fixed resolved-face payload record bound; above it the sidecar is refused."""
SIDECAR_MAX_BYTES = 64 * 1024 * 1024
"""E2C's fixed serialized sidecar byte bound; above it the sidecar is refused."""
SIDECAR_SOURCE = "resolved_face"
"""The only observation source E2C writes a payload record for."""
SIDECAR_ARRAY_FIELDS = ("raw_patch", "observed_patch", "valid", "coverage")
"""Payload arrays E2C carries, in the order the reader validates them."""

NUMERIC_ROW_METRICS = (
    "raw_rgb_mae_all_pixels",
    "raw_rgb_mae_both_valid",
    "observed_rgb_mae_all_pixels",
    "observed_rgb_mae_both_valid",
    "coverage_mae",
    "valid_mask_agreement",
)
"""Per-row numeric similarity metrics that get an aggregate distribution."""

NUMERIC_QUANTILE_LEVELS = ((90.0, "p90"), (99.0, "p99"))
"""Fixed percentile levels aggregated beside min/median/max; deterministic only."""


class TraceReplayError(ValueError):
    """Invalid, unverifiable or contradictory E1/trace replay input."""


# --------------------------------------------------------------------------- #
# Canonical JSON, hashing and primitive validation
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes used by every digest; byte-identical to E1's seal."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def seal_record(record: Mapping[str, Any]) -> str:
    """The E1 artifact seal: canonical JSON of the record minus its own digest."""
    content = {key: value for key, value in record.items() if key != "artifact_sha256"}
    return sha256_bytes(canonical_json(content))


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TraceReplayError(f"{label} is missing: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise TraceReplayError(f"{label} is not readable JSON: {source}") from error
    if not isinstance(document, dict):
        raise TraceReplayError(f"{label} must be a JSON object: {source}")
    return document


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceReplayError(f"{label} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise TraceReplayError(f"{label} must be finite")
    return number


def _require_non_negative(value: object, *, label: str) -> float:
    number = _require_finite(value, label=label)
    if number < 0.0:
        raise TraceReplayError(f"{label} must be non-negative")
    return number


def _require_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TraceReplayError(f"{label} must be an integer")
    return value


def _require_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceReplayError(f"{label} must be a non-empty string")
    return value


def _require_coordinate(value: object, *, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise TraceReplayError(f"{label} must be a 3-element list")
    return (
        _require_int(value[0], label=f"{label}[0]"),
        _require_int(value[1], label=f"{label}[1]"),
        _require_int(value[2], label=f"{label}[2]"),
    )


def state_record(value: object, *, label: str) -> dict[str, Any]:
    """Validate and normalise one JSON block-state record."""
    if not isinstance(value, Mapping):
        raise TraceReplayError(f"{label} must be an object")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise TraceReplayError(f"{label}.name must be a non-empty string")
    raw_properties = value.get("properties", [])
    if not isinstance(raw_properties, list):
        raise TraceReplayError(f"{label}.properties must be a list")
    properties: list[list[str]] = []
    for position, item in enumerate(raw_properties):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TraceReplayError(f"{label}.properties[{position}] must be a [key, value] pair")
        properties.append([str(item[0]), str(item[1])])
    return {"name": name, "properties": properties}


def state_key(record: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Hashable, order-insensitive identity of a validated state record."""
    return (
        str(record["name"]),
        tuple(sorted((str(pair[0]), str(pair[1])) for pair in record["properties"])),
    )


def array_digest(array: np.ndarray) -> str:
    """Production's ``_array_digest``: shape/dtype-bound SHA-256 of C-order bytes."""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(f"{value.shape}|{value.dtype.str}|".encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _decode_array(
    record: object, *, shape: tuple[int, ...], dtype_str: str, label: str
) -> np.ndarray:
    """Decode one bounded base64 array record with its exact frozen shape/dtype."""
    if not isinstance(record, Mapping):
        raise TraceReplayError(f"{label} must be an array object")
    if record.get("dtype") != dtype_str:
        raise TraceReplayError(f"{label} dtype must be exactly {dtype_str!r}")
    raw_shape = record.get("shape")
    if (
        not isinstance(raw_shape, list)
        or any(type(size) is not int for size in raw_shape)
        or raw_shape != list(shape)
    ):
        raise TraceReplayError(f"{label} shape must be exactly {list(shape)}")
    payload = record.get("base64")
    if not isinstance(payload, str):
        raise TraceReplayError(f"{label}.base64 must be a string")
    dtype = np.dtype(dtype_str)
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if len(payload) > 2 * expected_bytes + 8:
        raise TraceReplayError(f"{label} base64 payload is larger than its declared shape")
    try:
        raw = base64.b64decode(payload, validate=True)
    except ValueError as error:
        raise TraceReplayError(f"{label} is not valid base64") from error
    if len(raw) != expected_bytes:
        raise TraceReplayError(f"{label} byte length does not match its declared shape")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise TraceReplayError(f"{label} contains non-finite values")
    return array


def production_observed_patch(
    patch: np.ndarray, valid: np.ndarray, coverage: np.ndarray
) -> np.ndarray:
    """Rebuild production's ``FaceSample.observed_patch`` from raw E1 arrays.

    Production replaces every pixel outside ``valid`` with the coverage-weighted
    mean colour, in the stored float32 dtype (``FaceSample.rgb``), and returns the
    raw patch unchanged when ``valid`` is all true. This mirrors that arithmetic
    exactly; E1's own float64 ``_observed_patch`` is deliberately not reused.
    """
    patch = np.asarray(patch)
    valid = np.asarray(valid, dtype=bool)
    coverage = np.asarray(coverage)
    if valid.all():
        return np.ascontiguousarray(patch)
    result = patch.copy()
    pixels = patch[valid]
    if pixels.size == 0:
        result[~valid] = 0.0
        return np.ascontiguousarray(result)
    weights = coverage[valid]
    if float(weights.sum()) <= 1e-6:
        fill = np.zeros(3, dtype=np.float32)
    else:
        fill = np.average(pixels, axis=0, weights=weights).astype(np.float32)
    result[~valid] = fill
    return np.ascontiguousarray(result)


def _similarity_arrays(record: Mapping[str, Any], *, label: str) -> dict[str, np.ndarray]:
    """Pull the four decoded arrays a numeric comparison needs, or refuse."""
    if not isinstance(record, Mapping):
        raise TraceReplayError(f"{label} must be an array mapping")
    arrays: dict[str, np.ndarray] = {}
    for key in ("patch", "observed_patch", "valid", "coverage"):
        if key not in record:
            raise TraceReplayError(f"{label} is missing the {key!r} array")
        arrays[key] = np.asarray(record[key])
    return arrays


def _mean_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute channel difference in float64 over two same-shape arrays."""
    delta = np.abs(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    )
    return float(np.mean(delta))


def numeric_patch_similarity(
    e1_arrays: Mapping[str, Any],
    sidecar_arrays: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Bounded deterministic per-row numeric comparison of one joined face pair.

    Compares the decoded sidecar payload arrays against the sealed E1 arrays:
    raw-patch and reconstructed-observed RGB MAE over all pixels and over the
    intersection valid in both, coverage MAE over all pixels and the fraction of
    pixels whose valid masks agree. Every returned number is finite. The shape
    contract is explicit - patch/observed patch ``(H, W, 3)``, valid/coverage
    ``(H, W)`` - and any mismatch raises ``TraceReplayError`` rather than
    returning a number. An empty both-valid intersection is a real, counted
    outcome whose intersection MAE is ``None``. This is tolerant numeric
    similarity, never semantic parity and never a substitute for digest parity.
    """
    e1 = _similarity_arrays(e1_arrays, label=f"{label}.e1")
    sidecar = _similarity_arrays(sidecar_arrays, label=f"{label}.sidecar")
    patch_shape = e1["patch"].shape
    if e1["patch"].ndim != 3 or patch_shape[2] != 3:
        raise TraceReplayError(
            f"{label} shape mismatch: e1 patch shape {patch_shape} is not an HxWx3 RGB patch"
        )
    if sidecar["patch"].shape != patch_shape:
        raise TraceReplayError(
            f"{label} shape mismatch: raw patches {patch_shape} vs {sidecar['patch'].shape}"
        )
    if (
        e1["observed_patch"].shape != patch_shape
        or sidecar["observed_patch"].shape != patch_shape
    ):
        raise TraceReplayError(
            f"{label} shape mismatch: observed patches {e1['observed_patch'].shape} vs "
            f"{sidecar['observed_patch'].shape}, expected {patch_shape}"
        )
    pixel_shape = patch_shape[:2]
    if 0 in pixel_shape:
        raise TraceReplayError(f"{label} shape mismatch: patch has no pixels {patch_shape}")
    for name, arrays in (("e1", e1), ("sidecar", sidecar)):
        for mask in ("valid", "coverage"):
            if arrays[mask].shape != pixel_shape:
                raise TraceReplayError(
                    f"{label} shape mismatch: {name} {mask} shape {arrays[mask].shape} "
                    f"does not match the patch pixels {pixel_shape}"
                )
        for field in ("patch", "observed_patch", "coverage"):
            if not np.isfinite(np.asarray(arrays[field], dtype=np.float64)).all():
                raise TraceReplayError(f"{label}.{name}.{field} contains non-finite values")

    e1_valid = np.asarray(e1["valid"], dtype=bool)
    sidecar_valid = np.asarray(sidecar["valid"], dtype=bool)
    both_valid = e1_valid & sidecar_valid
    pixel_count = int(pixel_shape[0]) * int(pixel_shape[1])
    both_valid_count = int(np.count_nonzero(both_valid))
    metrics: dict[str, Any] = {
        "raw_rgb_mae_all_pixels": _mean_abs_difference(e1["patch"], sidecar["patch"]),
        "raw_rgb_mae_both_valid": (
            _mean_abs_difference(e1["patch"][both_valid], sidecar["patch"][both_valid])
            if both_valid_count
            else None
        ),
        "observed_rgb_mae_all_pixels": _mean_abs_difference(
            e1["observed_patch"], sidecar["observed_patch"]
        ),
        "observed_rgb_mae_both_valid": (
            _mean_abs_difference(
                e1["observed_patch"][both_valid], sidecar["observed_patch"][both_valid]
            )
            if both_valid_count
            else None
        ),
        "coverage_mae": _mean_abs_difference(e1["coverage"], sidecar["coverage"]),
        "valid_mask_agreement": float(np.mean(e1_valid == sidecar_valid)),
        "patch_pixel_count": pixel_count,
        "both_valid_pixel_count": both_valid_count,
    }
    for name, value in metrics.items():
        if value is not None and not np.isfinite(float(value)):
            raise TraceReplayError(f"{label}.{name} is not finite")
    return metrics


def _numeric_distribution(values: list[float]) -> dict[str, Any]:
    """Finite deterministic min/median/p90/p99/max with an explicit denominator."""
    if not values:
        summary: dict[str, Any] = {"row_denominator": 0}
        for name in ("min", "median", "p90", "p99", "max"):
            summary[name] = None
        return summary
    array = np.sort(np.asarray(values, dtype=np.float64))
    summary = {
        "row_denominator": int(array.size),
        "min": float(array[0]),
        "median": float(np.median(array)),
        "max": float(array[-1]),
    }
    for level, name in NUMERIC_QUANTILE_LEVELS:
        summary[name] = float(np.percentile(array, level))
    return summary


def summarize_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-row numeric metrics, keeping every denominator explicit.

    ``rows`` is one :func:`numeric_patch_similarity` result per joined resolved
    face, each carrying its own key. Shape mismatches are refused by that
    function before they can reach this aggregate, so ``rows_with_shape_mismatch``
    is an explicit zero-by-contract count rather than a silent drop.

    ``patch_pixels_per_row`` is derived from the measured ``patch_pixel_count`` of
    the joined rows, never assumed from the 16x16 production patch: uniform rows
    report their one measured size, and rows that mix shapes report ``None`` with
    the distinct measured sizes listed in ``distinct_patch_pixel_counts``.
    """
    joined = len(rows)
    empty_intersections = sum(1 for row in rows if row["both_valid_pixel_count"] == 0)
    distinct_patch_pixel_counts = sorted({int(row["patch_pixel_count"]) for row in rows})
    patch_pixels = sum(int(row["patch_pixel_count"]) for row in rows)
    both_valid_pixels = sum(int(row["both_valid_pixel_count"]) for row in rows)
    uniform_patch_pixels_per_row = (
        distinct_patch_pixel_counts[0] if len(distinct_patch_pixel_counts) == 1 else None
    )
    distributions = {
        metric: _numeric_distribution(
            [float(row[metric]) for row in rows if row[metric] is not None]
        )
        for metric in NUMERIC_ROW_METRICS
    }
    return {
        "joined_key_denominator": joined,
        "rows_with_shape_mismatch": 0,
        "rows_with_empty_both_valid_intersection": empty_intersections,
        "pixel_denominators": {
            "patch_pixels_per_row": uniform_patch_pixels_per_row,
            "patch_pixels_per_row_uniform": uniform_patch_pixels_per_row is not None,
            "distinct_patch_pixel_counts": distinct_patch_pixel_counts,
            "patch_pixel_comparisons": patch_pixels,
            "coverage_pixel_comparisons": patch_pixels,
            "valid_mask_pixel_comparisons": patch_pixels,
            "both_valid_pixel_comparisons": both_valid_pixels,
        },
        "per_row_metrics": distributions,
        "rows": rows,
        "shape_contract": (
            "every joined row measures its own patch and observed_patch (H, W, 3) <f4 with "
            "valid (H, W) |b1 and coverage (H, W) <f4; production resolved-face rows are "
            f"fixed {PATCH_SIZE}x{PATCH_SIZE}x3 and so measure {PATCH_SIZE * PATCH_SIZE} "
            "patch pixels per row; a within-row shape mismatch is refused before aggregation; "
            "patch_pixels_per_row is the measured uniform row size and is null with the mixed "
            "sizes listed in distinct_patch_pixel_counts when joined rows differ in shape"
        ),
        "note": (
            "Tolerant numeric similarity over a joined (frame_id, coordinate, slot) pair, "
            "reported separately from exact digest parity: raw/observed RGB MAE in channel-value "
            "units over all pixels and over the valid intersection, coverage MAE over all pixels, "
            "and the fraction of pixels whose valid masks agree. A both-valid metric is null and "
            "counted when the intersection is empty. Tolerant RGB similarity is not semantic "
            "parity and never replaces the digest parity section or the E1 truth rank."
        ),
    }


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    """A rate that always carries its own denominator, ``None`` when empty."""
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (float(numerator) / int(denominator)) if denominator else None,
    }


def _tally(bucket: list[int], matched: bool) -> None:
    """``bucket[0]`` is the denominator, ``bucket[1]`` the matches."""
    bucket[0] += 1
    if matched:
        bucket[1] += 1


def _rate_tally(bucket: list[int]) -> dict[str, Any]:
    return rate(bucket[1], bucket[0])


def _examples(rows: list[Any], total: int) -> dict[str, Any]:
    return {"count": int(total), "examples": rows[:MAX_EXAMPLES]}


# --------------------------------------------------------------------------- #
# E1 artifact loading
# --------------------------------------------------------------------------- #


def load_e1_artifact(path: str | Path, *, role: str) -> dict[str, Any]:
    """Validate and parse a sealed E1 patch-label artifact, or refuse.

    The embedded seal is recomputed from canonical JSON, arrays are decoded into
    their exact frozen domains and hashed with production's contract, and the raw
    patch and the reconstructed production observed patch are both retained.
    """
    if role not in ADMITTED_ROLES:
        raise TraceReplayError(
            f"role {role!r} is not a reviewed E1 fixture role; admitted roles are {list(ADMITTED_ROLES)}"
        )
    document = _load_json_object(path, "E1 patch-label artifact")
    if document.get("schema") != E1_SCHEMA:
        raise TraceReplayError(
            f"unknown E1 schema {document.get('schema')!r}; expected {E1_SCHEMA!r}"
        )
    if document.get("evaluation_only") is not True:
        raise TraceReplayError("E1 artifact is not marked evaluation_only")
    if document.get("role") != role:
        raise TraceReplayError(
            f"E1 artifact role {document.get('role')!r} does not match requested role {role!r}"
        )
    embedded = document.get("artifact_sha256")
    if not is_sha256(embedded):
        raise TraceReplayError("E1 artifact has no valid artifact_sha256")
    recomputed = seal_record(document)
    if recomputed != embedded:
        raise TraceReplayError(
            f"E1 artifact seal mismatch: embedded {embedded}, recomputed {recomputed}"
        )
    denominators = document.get("denominators")
    if not isinstance(denominators, Mapping):
        raise TraceReplayError("E1 artifact has no denominators mapping")
    patches = document.get("patches")
    if not isinstance(patches, list) or not patches:
        raise TraceReplayError("E1 artifact has no patch rows")

    entries: dict[tuple[str, tuple[int, int, int], str], dict[str, Any]] = {}
    coordinate_truth: dict[tuple[int, int, int], tuple[str, tuple[tuple[str, str], ...]]] = {}
    for position, record in enumerate(patches):
        label = f"E1 patches[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        frame_id = _require_str(record.get("frame_id"), label=f"{label}.frame_id")
        coordinate = _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate")
        slot = _require_str(record.get("slot"), label=f"{label}.slot")
        if slot not in FACE_SLOTS:
            raise TraceReplayError(f"{label}.slot must be one of {list(FACE_SLOTS)}")
        truth = state_record(record.get("truth_state"), label=f"{label}.truth_state")
        truth_key = state_key(truth)
        partial = record.get("partial")
        if not isinstance(partial, bool):
            raise TraceReplayError(f"{label}.partial must be a boolean")
        valid_fraction = _require_finite(record.get("valid_fraction"), label=f"{label}.valid_fraction")
        if not 0.0 <= valid_fraction <= 1.0:
            raise TraceReplayError(f"{label}.valid_fraction must lie in [0, 1]")
        patch = _decode_array(
            record.get("patch"), shape=PATCH_SHAPE, dtype_str=PATCH_DTYPE, label=f"{label}.patch"
        )
        valid = _decode_array(
            record.get("valid"), shape=MASK_SHAPE, dtype_str=MASK_DTYPE, label=f"{label}.valid"
        )
        coverage = _decode_array(
            record.get("coverage"), shape=MASK_SHAPE, dtype_str=COVERAGE_DTYPE, label=f"{label}.coverage"
        )
        key = (frame_id, coordinate, slot)
        if key in entries:
            raise TraceReplayError(
                f"duplicate E1 patch identity (frame, coordinate, slot)={key}; refusing an ambiguous join"
            )
        previous_truth = coordinate_truth.get(coordinate)
        if previous_truth is not None and previous_truth != truth_key:
            raise TraceReplayError(
                f"E1 patches for coordinate {coordinate} disagree on the truth state"
            )
        coordinate_truth[coordinate] = truth_key
        observed = production_observed_patch(patch, valid, coverage)
        entries[key] = {
            "frame_id": frame_id,
            "coordinate": coordinate,
            "slot": slot,
            "truth": truth_key,
            "partial": partial,
            # Decoded arrays retained for the optional E2F numeric comparison;
            # the E2B/E2D summary itself keeps using only the digests below.
            "arrays": {
                "patch": patch,
                "observed_patch": observed,
                "valid": valid,
                "coverage": coverage,
            },
            "digests": {
                "patch": array_digest(patch),
                "observed_patch": array_digest(observed),
                "valid": array_digest(valid),
                "coverage": array_digest(coverage),
            },
        }
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "artifact_sha256": str(embedded),
        "role": role,
        "denominators": {str(key): value for key, value in denominators.items()},
        "entries": entries,
        "coordinate_truth": coordinate_truth,
    }


# --------------------------------------------------------------------------- #
# Production trace loading
# --------------------------------------------------------------------------- #


def _digest_field(
    value: object, *, label: str, required: bool, forbidden: bool = False
) -> str | None:
    if value is None:
        if required:
            raise TraceReplayError(f"{label} is required for a resolved face")
        return None
    if forbidden:
        raise TraceReplayError(
            f"{label} must be null for a centre_patch_fallback observation"
        )
    if not is_sha256(value):
        raise TraceReplayError(f"{label} must be a sha256 digest or null")
    return str(value)


def load_trace(path: str | Path) -> dict[str, Any]:
    """Validate and parse a schema-2 production stage trace, or refuse."""
    document = _load_json_object(path, "production stage trace")
    schema = document.get("schema")
    if type(schema) is not int or schema != TRACE_SCHEMA:
        raise TraceReplayError(
            f"unknown trace schema {schema!r}; this reader admits schema {TRACE_SCHEMA} only"
        )
    required = (
        "candidate_limit", "palette", "support_rounds", "final_palette", "coordinates",
        "coordinate_evidence", "observations", "observation_costs", "aggregation",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise TraceReplayError(f"trace is missing required keys: {', '.join(missing)}")
    _require_int(document.get("candidate_limit"), label="trace.candidate_limit")

    final_palette = document.get("final_palette")
    if not isinstance(final_palette, Mapping):
        raise TraceReplayError("trace.final_palette must be an object")
    raw_states = final_palette.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise TraceReplayError("trace.final_palette.states must be a non-empty list")
    states = [state_record(state, label=f"final_palette.states[{i}]") for i, state in enumerate(raw_states)]
    size = _require_int(final_palette.get("size"), label="final_palette.size")
    if size != len(states):
        raise TraceReplayError("final_palette.size does not match its state list")
    final_keys = [state_key(state) for state in states]
    if len(set(final_keys)) != len(final_keys):
        raise TraceReplayError("final_palette.states contains duplicate states")
    final_index = {key: position for position, key in enumerate(final_keys)}

    observations = document.get("observations")
    if not isinstance(observations, list):
        raise TraceReplayError("trace.observations must be a list")
    cost_records = document.get("observation_costs")
    if not isinstance(cost_records, list):
        raise TraceReplayError("trace.observation_costs must be a list")
    if len(cost_records) != len(observations):
        raise TraceReplayError(
            "observation_costs and observations have different lengths; ordinals cannot align"
        )

    validated: list[dict[str, Any]] = []
    observations_by_key: dict[tuple[str, tuple[int, int, int], str], int] = {}
    for position, record in enumerate(observations):
        label = f"observations[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        ordinal = _require_int(record.get("ordinal"), label=f"{label}.ordinal")
        if ordinal != position:
            raise TraceReplayError(f"{label}.ordinal {ordinal} is not its position {position}")
        frame_id = _require_str(record.get("frame_id"), label=f"{label}.frame_id")
        coordinate = _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate")
        slot = _require_str(record.get("slot"), label=f"{label}.slot")
        if slot not in FACE_SLOTS:
            raise TraceReplayError(f"{label}.slot must be one of {list(FACE_SLOTS)}")
        source = _require_str(record.get("source"), label=f"{label}.source")
        if source not in OBSERVATION_SOURCES:
            raise TraceReplayError(f"{label}.source must be one of {list(OBSERVATION_SOURCES)}")
        resolved = source == "resolved_face"
        fallback = not resolved
        key = (frame_id, coordinate, slot)
        if key in observations_by_key:
            # raycast's seen_here retains one resolved face per key per frame and
            # its fallback only runs where no face covered the coordinate, so a
            # duplicate is a source contradiction, never something to collapse.
            raise TraceReplayError(
                f"duplicate production observation identity (frame, coordinate, slot)={key}"
            )
        observations_by_key[key] = position
        validated.append(
            {
                "ordinal": position,
                "frame_id": frame_id,
                "coordinate": coordinate,
                "slot": slot,
                "source": source,
                "observed_patch": _digest_field(
                    record.get("observed_patch"),
                    label=f"{label}.observed_patch",
                    required=resolved,
                    forbidden=fallback,
                ),
                "valid": _digest_field(
                    record.get("valid"),
                    label=f"{label}.valid",
                    required=resolved,
                    forbidden=fallback,
                ),
                "coverage": _digest_field(
                    record.get("coverage"),
                    label=f"{label}.coverage",
                    required=resolved,
                    forbidden=fallback,
                ),
                "rgb": _digest_field(record.get("rgb"), label=f"{label}.rgb", required=True),
                "descriptor": _digest_field(
                    record.get("descriptor"), label=f"{label}.descriptor", required=True
                ),
                "pattern": _digest_field(
                    record.get("pattern"),
                    label=f"{label}.pattern",
                    required=False,
                    forbidden=fallback,
                ),
            }
        )

    for position, record in enumerate(cost_records):
        label = f"observation_costs[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        observation = validated[position]
        if _require_int(record.get("ordinal"), label=f"{label}.ordinal") != position:
            raise TraceReplayError(f"{label}.ordinal does not match its observation")
        if _require_str(record.get("frame_id"), label=f"{label}.frame_id") != observation["frame_id"]:
            raise TraceReplayError(f"{label}.frame_id does not match its observation")
        if _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate") != observation["coordinate"]:
            raise TraceReplayError(f"{label}.coordinate does not match its observation")
        if _require_str(record.get("slot"), label=f"{label}.slot") != observation["slot"]:
            raise TraceReplayError(f"{label}.slot does not match its observation")
        if _require_str(record.get("source"), label=f"{label}.source") != observation["source"]:
            raise TraceReplayError(f"{label}.source does not match its observation")
        palette_order = record.get("palette_order")
        if not isinstance(palette_order, list):
            raise TraceReplayError(f"{label}.palette_order must be a list")
        normalised_order = [
            state_record(state, label=f"{label}.palette_order[{i}]")
            for i, state in enumerate(palette_order)
        ]
        if normalised_order != states:
            raise TraceReplayError(
                f"{label}.palette_order does not equal final_palette.states; "
                "the cost vector is not interpretable"
            )
        raw_costs = record.get("costs")
        if not isinstance(raw_costs, list):
            raise TraceReplayError(f"{label}.costs must be a list")
        if len(raw_costs) != size:
            raise TraceReplayError(
                f"{label}.costs has length {len(raw_costs)}, expected the palette size {size}"
            )
        costs = [_require_finite(value, label=f"{label}.costs[{i}]") for i, value in enumerate(raw_costs)]
        best_cost = _require_finite(record.get("best_cost"), label=f"{label}.best_cost")
        weight = _require_non_negative(record.get("weight"), label=f"{label}.weight")
        view_weight = _require_non_negative(record.get("view_weight"), label=f"{label}.view_weight")
        effective = _require_non_negative(
            record.get("effective_weight"), label=f"{label}.effective_weight"
        )
        if not np.isclose(effective, weight * view_weight, rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
            raise TraceReplayError(f"{label}.effective_weight is not weight * view_weight")
        # The scored vector is ``min(distance - best, ceiling)``, so its minimum is
        # exactly zero. A vector with no zero cannot have come from the scorer.
        if size and abs(min(costs)) > 1e-9:
            raise TraceReplayError(f"{label}.costs has no zero entry; not a production cost vector")
        observation.update(
            {
                "costs": costs,
                "best_cost": best_cost,
                "weight": weight,
                "view_weight": view_weight,
                "effective_weight": effective,
            }
        )

    raw_aggregation = document.get("aggregation")
    if not isinstance(raw_aggregation, list):
        raise TraceReplayError("trace.aggregation must be a list")
    aggregation: dict[tuple[int, int, int], dict[str, Any]] = {}
    for position, record in enumerate(raw_aggregation):
        label = f"aggregation[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        coordinate = _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate")
        if coordinate in aggregation:
            raise TraceReplayError(f"{label}.coordinate {coordinate} is duplicated")
        raw_total = record.get("total")
        if not isinstance(raw_total, list) or len(raw_total) != size:
            raise TraceReplayError(f"{label}.total must have the palette length {size}")
        aggregation[coordinate] = {
            "coordinate": coordinate,
            "total": [_require_finite(value, label=f"{label}.total[{i}]") for i, value in enumerate(raw_total)],
            "weight": _require_non_negative(record.get("weight"), label=f"{label}.weight"),
        }

    raw_coordinates = document.get("coordinates")
    if not isinstance(raw_coordinates, list):
        raise TraceReplayError("trace.coordinates must be a list")
    coordinate_records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for position, record in enumerate(raw_coordinates):
        label = f"coordinates[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        coordinate = _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate")
        if coordinate in coordinate_records:
            raise TraceReplayError(f"{label}.coordinate {coordinate} is duplicated")
        coordinate_records[coordinate] = dict(record)

    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "schema": TRACE_SCHEMA,
        "candidate_limit": int(document["candidate_limit"]),
        "final_palette_size": size,
        "states": states,
        "final_keys": final_keys,
        "final_index": final_index,
        "observations": validated,
        "observations_by_key": observations_by_key,
        "aggregation": aggregation,
        "coordinate_records": coordinate_records,
    }


# --------------------------------------------------------------------------- #
# Optional E2C production patch sidecar loading
# --------------------------------------------------------------------------- #


def _sidecar_stored_digests(record: Mapping[str, Any], *, label: str) -> dict[str, str]:
    """Validate the four stored sidecar digests E2C writes beside its payloads."""
    digests = record.get("digests")
    if not isinstance(digests, Mapping):
        raise TraceReplayError(f"{label}.digests must be an object")
    stored: dict[str, str] = {}
    for field in SIDECAR_ARRAY_FIELDS:
        value = digests.get(field)
        if not is_sha256(value):
            raise TraceReplayError(f"{label}.digests.{field} must be a sha256 digest")
        stored[field] = str(value).lower()
    return stored


def load_sidecar(
    path: str | Path, *, trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and parse an E2C production patch sidecar, or refuse.

    The sidecar is optional input bound to one already-loaded schema-2 trace. It
    is checked against that trace's exact file bytes (SHA-256 and byte count) and
    against its observation ordinals, identities and digests; the fixed E2C
    record/byte bounds are enforced before any payload array is decoded. Every
    payload is decoded losslessly at its frozen shape/dtype and the reader
    recomputes its shape/dtype-bound digest rather than trusting the stored one.
    Nothing is mutated and no reference label or reconstruction decision is read.
    """
    source = Path(path)
    if not source.is_file():
        raise TraceReplayError(f"production patch sidecar is missing: {source}")
    size = int(source.stat().st_size)
    if size > SIDECAR_MAX_BYTES:
        raise TraceReplayError(
            f"production patch sidecar is {size} bytes, above the {SIDECAR_MAX_BYTES}-byte bound"
        )
    document = _load_json_object(source, "production patch sidecar")
    if document.get("schema") != SIDECAR_SCHEMA:
        raise TraceReplayError(
            f"unknown sidecar schema {document.get('schema')!r}; expected {SIDECAR_SCHEMA!r}"
        )
    trace_schema = document.get("trace_schema")
    if type(trace_schema) is not int or trace_schema != TRACE_SCHEMA:
        raise TraceReplayError(
            f"sidecar trace_schema {trace_schema!r} does not declare trace schema {TRACE_SCHEMA}"
        )
    bound_sha = document.get("trace_sha256")
    if not is_sha256(bound_sha):
        raise TraceReplayError("sidecar has no valid trace_sha256")
    bound_bytes = _require_int(document.get("trace_bytes"), label="sidecar.trace_bytes")
    if bound_bytes < 0:
        raise TraceReplayError("sidecar.trace_bytes must be non-negative")
    trace_path = Path(str(trace["path"]))
    if not trace_path.is_file():
        raise TraceReplayError(f"the sidecar's bound trace is missing: {trace_path}")
    actual_bytes = int(trace_path.stat().st_size)
    if bound_bytes != actual_bytes:
        raise TraceReplayError(
            f"sidecar trace byte count {bound_bytes} does not match the trace file size "
            f"{actual_bytes}"
        )
    if str(bound_sha).lower() != str(trace["sha256"]).lower():
        raise TraceReplayError(
            f"sidecar trace digest {bound_sha} does not match the trace bytes {trace['sha256']}"
        )
    observation_count = _require_int(
        document.get("observation_count"), label="sidecar.observation_count"
    )
    resolved_headers = _require_int(
        document.get("resolved_face_records"), label="sidecar.resolved_face_records"
    )
    excluded_fallbacks = _require_int(
        document.get("excluded_fallback_observations"),
        label="sidecar.excluded_fallback_observations",
    )
    for name, value in (
        ("observation_count", observation_count),
        ("resolved_face_records", resolved_headers),
        ("excluded_fallback_observations", excluded_fallbacks),
    ):
        if value < 0:
            raise TraceReplayError(f"sidecar.{name} must be non-negative")
    records = document.get("records")
    if not isinstance(records, list):
        raise TraceReplayError("sidecar.records must be a list")
    if len(records) > SIDECAR_MAX_RECORDS:
        raise TraceReplayError(
            f"sidecar carries {len(records)} records, above the {SIDECAR_MAX_RECORDS}-record bound"
        )

    observations = trace["observations"]
    resolved_positions = {
        observation["ordinal"]
        for observation in observations
        if observation["source"] == SIDECAR_SOURCE
    }
    fallback_total = sum(
        1 for observation in observations if observation["source"] != SIDECAR_SOURCE
    )
    if observation_count != len(observations):
        raise TraceReplayError(
            f"sidecar observation_count {observation_count} does not match the trace's "
            f"{len(observations)} observations"
        )
    if excluded_fallbacks != fallback_total:
        raise TraceReplayError(
            f"sidecar excluded_fallback_observations {excluded_fallbacks} does not match the "
            f"trace's {fallback_total} fallback observations"
        )
    if resolved_headers != len(records):
        raise TraceReplayError(
            f"sidecar resolved_face_records {resolved_headers} does not match its "
            f"{len(records)} record rows"
        )
    if len(records) != len(resolved_positions):
        raise TraceReplayError(
            f"sidecar has {len(records)} rows but the trace has {len(resolved_positions)} "
            "resolved_face observations; exactly one row per resolved face is required"
        )

    validated: list[dict[str, Any]] = []
    by_ordinal: dict[int, dict[str, Any]] = {}
    by_key: dict[tuple[str, tuple[int, int, int], str], dict[str, Any]] = {}
    for position, record in enumerate(records):
        label = f"sidecar.records[{position}]"
        if not isinstance(record, Mapping):
            raise TraceReplayError(f"{label} must be an object")
        ordinal = _require_int(record.get("ordinal"), label=f"{label}.ordinal")
        if ordinal in by_ordinal:
            raise TraceReplayError(f"{label}.ordinal {ordinal} is duplicated in the sidecar")
        frame_id = _require_str(record.get("frame_id"), label=f"{label}.frame_id")
        coordinate = _require_coordinate(record.get("coordinate"), label=f"{label}.coordinate")
        slot = _require_str(record.get("slot"), label=f"{label}.slot")
        if slot not in FACE_SLOTS:
            raise TraceReplayError(f"{label}.slot must be one of {list(FACE_SLOTS)}")
        if record.get("source") != SIDECAR_SOURCE:
            raise TraceReplayError(
                f"{label}.source must be {SIDECAR_SOURCE!r}; a fallback carries no face payload"
            )
        key = (frame_id, coordinate, slot)
        if key in by_key:
            raise TraceReplayError(
                f"duplicate sidecar identity (frame, coordinate, slot)={key}; "
                "refusing an ambiguous join"
            )
        if ordinal not in resolved_positions:
            raise TraceReplayError(
                f"{label}.ordinal {ordinal} is not a resolved_face trace observation"
            )
        observation = observations[ordinal]
        if (frame_id, coordinate, slot) != (
            observation["frame_id"],
            observation["coordinate"],
            observation["slot"],
        ):
            raise TraceReplayError(
                f"{label} identity (frame, coordinate, slot)={key} disagrees with the trace "
                "observation at the same ordinal"
            )
        stored = _sidecar_stored_digests(record, label=label)
        raw_patch = _decode_array(
            record.get("raw_patch"), shape=PATCH_SHAPE, dtype_str=PATCH_DTYPE,
            label=f"{label}.raw_patch",
        )
        observed_patch = _decode_array(
            record.get("observed_patch"), shape=PATCH_SHAPE, dtype_str=PATCH_DTYPE,
            label=f"{label}.observed_patch",
        )
        valid = _decode_array(
            record.get("valid"), shape=MASK_SHAPE, dtype_str=MASK_DTYPE, label=f"{label}.valid"
        )
        coverage = _decode_array(
            record.get("coverage"), shape=MASK_SHAPE, dtype_str=COVERAGE_DTYPE,
            label=f"{label}.coverage",
        )
        # Recompute every digest from the decoded bytes; the stored digest is only
        # compared, never trusted.
        computed = {
            "raw_patch": array_digest(raw_patch),
            "observed_patch": array_digest(observed_patch),
            "valid": array_digest(valid),
            "coverage": array_digest(coverage),
        }
        for field in SIDECAR_ARRAY_FIELDS:
            if stored[field] != computed[field]:
                raise TraceReplayError(
                    f"{label}.digests.{field} {stored[field]} does not match its decoded bytes "
                    f"{computed[field]}"
                )
        for field in ("observed_patch", "valid", "coverage"):
            trace_digest = str(observation[field]).lower()
            if computed[field] != trace_digest:
                raise TraceReplayError(
                    f"{label}.{field} digest {computed[field]} disagrees with trace observation "
                    f"ordinal {ordinal} ({trace_digest})"
                )
        entry = {
            "ordinal": ordinal,
            "frame_id": frame_id,
            "coordinate": coordinate,
            "slot": slot,
            # Decoded arrays retained for the optional E2F numeric comparison;
            # digest parity below is unchanged and recomputed from these bytes.
            "arrays": {
                "patch": raw_patch,
                "observed_patch": observed_patch,
                "valid": valid,
                "coverage": coverage,
            },
            "digests": computed,
        }
        validated.append(entry)
        by_ordinal[ordinal] = entry
        by_key[key] = entry

    if set(by_ordinal) != resolved_positions:
        missing = sorted(resolved_positions - set(by_ordinal))
        extra = sorted(set(by_ordinal) - resolved_positions)
        raise TraceReplayError(
            "sidecar ordinal set does not cover the resolved_face trace observations "
            f"(missing {missing[:3]}, extra {extra[:3]})"
        )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "schema": SIDECAR_SCHEMA,
        "trace_sha256": str(bound_sha).lower(),
        "trace_bytes": bound_bytes,
        "observation_count": observation_count,
        "resolved_face_records": resolved_headers,
        "excluded_fallback_observations": excluded_fallbacks,
        "records": validated,
        "by_ordinal": by_ordinal,
        "by_key": by_key,
    }


# --------------------------------------------------------------------------- #
# Aggregation replay and production top-K parity
# --------------------------------------------------------------------------- #


def _replay_aggregation(
    observations: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], float]]:
    """Rebuild ``_accumulate_costs`` from the stored vectors and effective weights."""
    totals: dict[tuple[int, int, int], np.ndarray] = {}
    weights: dict[tuple[int, int, int], float] = {}
    for record in observations:
        effective = float(record["effective_weight"])
        if effective <= 0.0:
            continue
        coordinate = record["coordinate"]
        costs = np.asarray(record["costs"], dtype=np.float64)
        if coordinate in totals:
            totals[coordinate] = totals[coordinate] + costs * effective
        else:
            totals[coordinate] = costs * effective
        weights[coordinate] = weights.get(coordinate, 0.0) + effective
    return totals, weights


def _validate_aggregation(
    trace: dict[str, Any],
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], float], dict[str, Any]]:
    """Replay the aggregation and require float64-tolerance equality, or refuse."""
    totals, weights = _replay_aggregation(trace["observations"])
    aggregation = trace["aggregation"]
    if set(aggregation) != set(totals):
        missing = sorted(set(aggregation) - set(totals))
        extra = sorted(set(totals) - set(aggregation))
        raise TraceReplayError(
            "inconsistent aggregate replay: coordinate sets differ "
            f"(missing from replay {missing[:3]}, absent from trace {extra[:3]})"
        )
    max_weight_error = 0.0
    max_total_error = 0.0
    weight_rows = 0
    total_rows = 0
    for coordinate, record in aggregation.items():
        weight_error = abs(float(record["weight"]) - float(weights[coordinate]))
        max_weight_error = max(max_weight_error, weight_error)
        if not np.isclose(record["weight"], weights[coordinate], rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
            raise TraceReplayError(
                f"inconsistent aggregate replay: weight mismatch at {coordinate} "
                f"(trace {record['weight']}, replay {weights[coordinate]})"
            )
        weight_rows += 1
        stored_total = np.asarray(record["total"], dtype=np.float64)
        if stored_total.shape != totals[coordinate].shape:
            raise TraceReplayError(f"inconsistent aggregate replay: total length mismatch at {coordinate}")
        error = float(np.max(np.abs(totals[coordinate] - stored_total))) if stored_total.size else 0.0
        max_total_error = max(max_total_error, error)
        if not np.allclose(totals[coordinate], stored_total, rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
            raise TraceReplayError(
                f"inconsistent aggregate replay: total mismatch at {coordinate} (max |delta| {error})"
            )
        total_rows += 1
    return totals, weights, {
        "coordinates": len(aggregation),
        "weights_equal": rate(weight_rows, len(aggregation)),
        "totals_equal": rate(total_rows, len(aggregation)),
        "max_abs_weight_error": max_weight_error,
        "max_abs_total_error": max_total_error,
    }


def _check_production_top_k(
    trace: dict[str, Any],
    totals: dict[tuple[int, int, int], np.ndarray],
    weights: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    """Re-derive production's top-K set from the replayed means and compare.

    Production's ``_top_candidates`` is ``np.argpartition(mean, keep - 1)[:keep]``
    over the per-coordinate mean cost vector; this rebuilds that set entirely
    separately from any E1 per-face ranking.
    """
    final_keys = trace["final_keys"]
    checked = 0
    agreed = 0
    disagreements: list[dict[str, Any]] = []
    for coordinate, record in trace["coordinate_records"].items():
        if not record.get("raw_winner") or coordinate not in totals:
            continue
        mean = totals[coordinate] / max(float(weights[coordinate]), 1e-6)
        keep = min(PRODUCTION_TOP_K, int(mean.size))
        if keep <= 0:
            continue
        derived = {final_keys[int(position)] for position in np.argpartition(mean, keep - 1)[:keep]}
        raw_stored = record.get("top_candidates")
        stored: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        if isinstance(raw_stored, list):
            for item in raw_stored:
                if isinstance(item, Mapping) and isinstance(item.get("state"), Mapping):
                    stored.add(state_key(state_record(item["state"], label="top_candidates.state")))
        checked += 1
        if derived == stored:
            agreed += 1
        elif len(disagreements) < MAX_EXAMPLES:
            disagreements.append(
                {
                    "coordinate": list(coordinate),
                    "trace_only": [list(key) for key in sorted(stored - derived)],
                    "replay_only": [list(key) for key in sorted(derived - stored)],
                }
            )
    return {
        "checked": rate(agreed, checked),
        "disagreements": _examples(disagreements, checked - agreed),
        "contract": "np.argpartition(mean, min(8, size) - 1)[:min(8, size)]",
    }


# --------------------------------------------------------------------------- #
# Replay and reconciliation
# --------------------------------------------------------------------------- #


def _group_counts(
    keys: list[tuple[str, tuple[int, int, int], str]],
) -> tuple[dict[str, int], dict[str, int]]:
    frames: dict[str, int] = {}
    slots: dict[str, int] = {}
    for frame_id, _coordinate, slot in keys:
        frames[frame_id] = frames.get(frame_id, 0) + 1
        slots[slot] = slots.get(slot, 0) + 1
    return frames, slots


def _reconcile_sidecar(
    sidecar: Mapping[str, Any], e1: Mapping[str, Any], trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Join the validated sidecar to E1 and split identity from pixel parity.

    The join key is the exact ``(frame_id, coordinate, slot)``; the sidecar key
    set already equals the trace's resolved-face key set by ``load_sidecar``.
    Identity joins are counted separately from the pixel parity rates, and the
    stored sidecar digests are never used here - ``load_sidecar`` replaced them
    with digests recomputed from the decoded bytes. The separate
    ``numeric_similarity`` section reuses the decoded arrays for a bounded,
    deterministic tolerant MAE comparison; it never feeds the exact parity rates,
    the E1 truth rank or any production gate.
    """
    e1_entries = e1["entries"]
    sidecar_entries = sidecar["by_key"]
    e1_keys = set(e1_entries)
    sidecar_keys = set(sidecar_entries)
    joins = sidecar_keys & e1_keys
    sidecar_only = sidecar_keys - e1_keys
    e1_only = e1_keys - sidecar_keys
    trace_resolved = sum(
        1 for observation in trace["observations"] if observation["source"] == SIDECAR_SOURCE
    )

    parity = {field: [0, 0] for field in (*SIDECAR_ARRAY_FIELDS, "all_four")}
    mismatches: list[dict[str, Any]] = []
    mismatch_total = 0
    for key in sorted(joins):
        record = sidecar_entries[key]
        entry = e1_entries[key]
        e1_digests = entry["digests"]
        matches = {
            "raw_patch": record["digests"]["raw_patch"] == e1_digests["patch"],
            "observed_patch": record["digests"]["observed_patch"] == e1_digests["observed_patch"],
            "valid": record["digests"]["valid"] == e1_digests["valid"],
            "coverage": record["digests"]["coverage"] == e1_digests["coverage"],
        }
        for field, matched in matches.items():
            _tally(parity[field], matched)
        all_four = all(matches.values())
        _tally(parity["all_four"], all_four)
        if not all_four:
            mismatch_total += 1
            if len(mismatches) < MAX_EXAMPLES:
                mismatches.append(
                    {
                        "key": {
                            "frame_id": key[0],
                            "coordinate": list(key[1]),
                            "slot": key[2],
                        },
                        "matched": matches,
                        "e1_partial": entry["partial"],
                        "sidecar": dict(record["digests"]),
                        "e1": {
                            "raw_patch": e1_digests["patch"],
                            "observed_patch": e1_digests["observed_patch"],
                            "valid": e1_digests["valid"],
                            "coverage": e1_digests["coverage"],
                        },
                    }
                )

    # --- Bounded deterministic numeric similarity, separate from parity --- #
    # The decoded sidecar/E1 arrays are compared per joined row; a shape
    # mismatch is refused inside ``numeric_patch_similarity`` before any number
    # is produced, so it can never be silently aggregated.
    numeric_rows: list[dict[str, Any]] = []
    for key in sorted(joins):
        metrics = numeric_patch_similarity(
            e1_entries[key]["arrays"],
            sidecar_entries[key]["arrays"],
            label=f"sidecar numeric row (frame={key[0]!r}, coordinate={list(key[1])}, slot={key[2]!r})",
        )
        metrics["key"] = {"frame_id": key[0], "coordinate": list(key[1]), "slot": key[2]}
        numeric_rows.append(metrics)
    numeric_similarity = summarize_numeric_rows(numeric_rows)

    return {
        "path": sidecar["path"],
        "sha256": sidecar["sha256"],
        "schema": SIDECAR_SCHEMA,
        "trace_sha256_bound": sidecar["trace_sha256"],
        "trace_bytes_bound": sidecar["trace_bytes"],
        "record_count": len(sidecar["records"]),
        "trace_observation_count": sidecar["observation_count"],
        "trace_resolved_face_observations": trace_resolved,
        "fallback_observations_excluded": sidecar["excluded_fallback_observations"],
        "ordinal_identity_reconciliation": {
            "reconciled_rows": len(sidecar["records"]),
            "resolved_face_trace_observations": trace_resolved,
            "fallback_observations_excluded": sidecar["excluded_fallback_observations"],
            "note": (
                "Every sidecar row was checked against the trace observation at the same "
                "ordinal for identity, source and observed/valid/coverage digest agreement; "
                "exactly one row per resolved_face observation is required."
            ),
        },
        "counts": {
            "sidecar_resolved_rows": len(sidecar_keys),
            "trace_resolved_face_rows": trace_resolved,
            "e1_patch_keys": len(e1_keys),
            "identity_joins": len(joins),
            "e1_only_keys": len(e1_only),
            "sidecar_only_keys": len(sidecar_only),
            "note": (
                "identity_joins is an exact (frame_id, coordinate, slot) join only; "
                "e1_only/sidecar_only rows are legitimate sampler-domain differences"
            ),
        },
        "parity": {
            "joined_key_denominator": len(joins),
            "raw_patch": _rate_tally(parity["raw_patch"]),
            "observed_patch": _rate_tally(parity["observed_patch"]),
            "valid": _rate_tally(parity["valid"]),
            "coverage": _rate_tally(parity["coverage"]),
            "all_four": _rate_tally(parity["all_four"]),
            "mismatches": _examples(mismatches, mismatch_total),
            "note": (
                "raw_patch compares decoded sidecar bytes to E1's stored raw patch; "
                "observed_patch compares decoded sidecar bytes to E1's reconstructed "
                "production float32 observed input; identity match does not imply pixel match"
            ),
        },
        "numeric_similarity": numeric_similarity,
    }


def replay(
    e1_path: str | Path,
    trace_path: str | Path,
    *,
    role: str,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile one sealed E1 artifact with one schema-2 trace, or refuse.

    Aggregate replay is validated before any E1 join; an inconsistent trace is a
    refusal, not a number. ``sidecar_path`` is the optional E2C production patch
    sidecar: when supplied it is validated against the exact trace bytes and the
    trace's ordinals/identities/digests, joined to E1 by exact key, and reported as
    a separate parity section plus a separate tolerant numeric similarity section.
    When omitted the summary is byte-for-byte the existing E2B output.
    """
    e1 = load_e1_artifact(e1_path, role=role)
    trace = load_trace(trace_path)
    totals, weights, aggregation_replay = _validate_aggregation(trace)
    top_k = _check_production_top_k(trace, totals, weights)
    sidecar_summary = None
    if sidecar_path is not None:
        sidecar = load_sidecar(sidecar_path, trace=trace)
        sidecar_summary = _reconcile_sidecar(sidecar, e1, trace)

    e1_entries: dict[tuple[str, tuple[int, int, int], str], dict[str, Any]] = e1["entries"]
    observations_by_key = trace["observations_by_key"]
    observations = trace["observations"]
    e1_keys = set(e1_entries)
    production_keys = set(observations_by_key)
    # Only resolved-face observations carry a rectified face patch. A
    # centre_patch_fallback observation may collide with an E1 key, but it is not
    # a face patch, so patch-key joins, digest parity and truth rank are all
    # restricted to source == resolved_face; fallback keys are counted separately.
    resolved_keys = {
        key
        for key, position in observations_by_key.items()
        if observations[position]["source"] == "resolved_face"
    }
    fallback_keys = production_keys - resolved_keys
    intersection = e1_keys & resolved_keys
    e1_only = e1_keys - resolved_keys
    production_only = resolved_keys - e1_keys
    fallback_only = fallback_keys - e1_keys
    fallback_collisions = fallback_keys & e1_keys

    e1_frames, e1_slots = _group_counts(sorted(e1_keys))
    production_frames, production_slots = _group_counts(sorted(production_keys))
    resolved_frames, resolved_slots = _group_counts(sorted(resolved_keys))
    fallback_frames, fallback_slots = _group_counts(sorted(fallback_keys))
    intersection_frames, intersection_slots = _group_counts(sorted(intersection))
    e1_only_frames, e1_only_slots = _group_counts(sorted(e1_only))
    production_only_frames, production_only_slots = _group_counts(sorted(production_only))

    per_frame = {
        frame: {
            "e1": int(e1_frames.get(frame, 0)),
            "production": int(production_frames.get(frame, 0)),
            "resolved_face": int(resolved_frames.get(frame, 0)),
            "fallback": int(fallback_frames.get(frame, 0)),
            "intersection": int(intersection_frames.get(frame, 0)),
            "e1_only": int(e1_only_frames.get(frame, 0)),
            "production_only": int(production_only_frames.get(frame, 0)),
        }
        for frame in sorted(set(e1_frames) | set(production_frames))
    }
    per_slot = {
        slot: {
            "e1": int(e1_slots.get(slot, 0)),
            "production": int(production_slots.get(slot, 0)),
            "resolved_face": int(resolved_slots.get(slot, 0)),
            "fallback": int(fallback_slots.get(slot, 0)),
            "intersection": int(intersection_slots.get(slot, 0)),
            "e1_only": int(e1_only_slots.get(slot, 0)),
            "production_only": int(production_only_slots.get(slot, 0)),
        }
        for slot in sorted(set(e1_slots) | set(production_slots))
    }

    # --- Digest identity over joined resolved-face keys -------------------- #
    digest = {
        "patch_observed_reconstruction": [0, 0],
        "patch_raw_vs_observed": [0, 0],
        "valid": [0, 0],
        "coverage": [0, 0],
        "all_three_observed_contract": [0, 0],
    }
    partial_rows = [0, 0]
    fully_valid_rows = [0, 0]
    mismatch_rows: list[dict[str, Any]] = []
    mismatch_total = 0
    for key in sorted(intersection):
        entry = e1_entries[key]
        observation = trace["observations"][observations_by_key[key]]
        e1_observed = entry["digests"]["observed_patch"]
        e1_valid = entry["digests"]["valid"]
        e1_coverage = entry["digests"]["coverage"]
        reconstruction_match = e1_observed == observation["observed_patch"]
        raw_match = entry["digests"]["patch"] == observation["observed_patch"]
        all_three = (
            reconstruction_match
            and e1_valid == observation["valid"]
            and e1_coverage == observation["coverage"]
        )
        _tally(digest["patch_observed_reconstruction"], reconstruction_match)
        _tally(digest["patch_raw_vs_observed"], raw_match)
        _tally(digest["valid"], e1_valid == observation["valid"])
        _tally(digest["coverage"], e1_coverage == observation["coverage"])
        _tally(digest["all_three_observed_contract"], all_three)
        _tally(partial_rows if entry["partial"] else fully_valid_rows, reconstruction_match)
        if not all_three:
            mismatch_total += 1
            if len(mismatch_rows) < MAX_EXAMPLES:
                mismatch_rows.append(
                    {
                        "key": {"frame_id": key[0], "coordinate": list(key[1]), "slot": key[2]},
                        "e1_partial": entry["partial"],
                        "e1": {
                            "observed_patch": e1_observed,
                            "valid": e1_valid,
                            "coverage": e1_coverage,
                        },
                        "production": {
                            "observed_patch": observation["observed_patch"],
                            "valid": observation["valid"],
                            "coverage": observation["coverage"],
                        },
                    }
                )

    # --- Per-observation truth rank from the final cost vector ------------- #
    # Restricted to joined resolved-face keys: a fallback observation has no
    # rectified face patch and must not contribute a rank row.
    rank_rows: list[dict[str, Any]] = []
    truth_in_palette = 0
    truth_oov = 0
    rank_hits = {depth: 0 for depth in (1, PRODUCTION_TOP_K, 16, 32)}
    for key in sorted(intersection):
        entry = e1_entries[key]
        truth = entry["truth"]
        if truth not in trace["final_index"]:
            truth_oov += 1
            continue
        truth_in_palette += 1
        position = trace["final_index"][truth]
        observation = trace["observations"][observations_by_key[key]]
        costs = np.asarray(observation["costs"], dtype=np.float64)
        truth_cost = float(costs[position])
        # Cheaper first, ties broken by palette position - the order the cost
        # vector is recorded in. This is not the argpartition candidate set.
        better = int(np.sum(costs < truth_cost))
        tied_before = int(np.sum((costs == truth_cost) & (np.arange(costs.size) < position)))
        truth_rank = better + tied_before + 1
        for depth in rank_hits:
            if truth_rank <= depth:
                rank_hits[depth] += 1
        if len(rank_rows) < MAX_EXAMPLES:
            rank_rows.append(
                {
                    "key": {"frame_id": key[0], "coordinate": list(key[1]), "slot": key[2]},
                    "truth_rank": truth_rank,
                    "truth_cost": truth_cost,
                    "best_cost": float(observation["best_cost"]),
                }
            )

    # --- Per-cell aggregates over the E1-labeled coordinate subset --------- #
    cell_metrics = {
        "top8_membership": [0, 0],
        "raw_winner_match": [0, 0],
        "post_smoothing_match": [0, 0],
        "selected_state_match": [0, 0],
        "final_state_match": [0, 0],
    }
    e1_labeled_cells = len(e1["coordinate_truth"])
    cells_with_record = 0
    cells_missing_record = 0
    truth_available_cells = 0
    truth_oov_cells = 0
    candidate_abstentions = 0
    placement_abstentions = 0
    # Fallback evidence is part of the actual production cell result, so it is
    # reported here even though it is never joined as an E1 face patch. The
    # observation-row count and the distinct-coordinate count are kept separate:
    # several fallback observations can sit at one E1-labeled coordinate (for
    # example different slots or frames), and the previous single field name
    # claimed observations while counting coordinates.
    fallback_labeled_observations = sum(
        1 for key in fallback_keys if key[1] in e1["coordinate_truth"]
    )
    fallback_labeled_coordinates = {
        key[1] for key in fallback_keys if key[1] in e1["coordinate_truth"]
    }
    for coordinate in sorted(e1["coordinate_truth"]):
        truth = e1["coordinate_truth"][coordinate]
        record = trace["coordinate_records"].get(coordinate)
        if record is None:
            cells_missing_record += 1
            continue
        cells_with_record += 1
        truth_in_final = truth in trace["final_index"]
        if truth_in_final:
            truth_available_cells += 1
        else:
            truth_oov_cells += 1
        top_candidates = record.get("top_candidates")
        if not isinstance(top_candidates, list) or not top_candidates:
            candidate_abstentions += 1
        if record.get("final_state") is None:
            placement_abstentions += 1
        if not truth_in_final:
            continue
        if isinstance(top_candidates, list) and top_candidates:
            stored = {
                state_key(state_record(item["state"], label="top_candidates.state"))
                for item in top_candidates
                if isinstance(item, Mapping) and isinstance(item.get("state"), Mapping)
            }
            _tally(cell_metrics["top8_membership"], truth in stored)
        raw_winner = record.get("raw_winner")
        if raw_winner is not None:
            _tally(
                cell_metrics["raw_winner_match"],
                state_key(state_record(raw_winner, label="raw_winner")) == truth,
            )
        for field, bucket in (
            ("post_smoothing_label", "post_smoothing_match"),
            ("selected_state", "selected_state_match"),
            ("final_state", "final_state_match"),
        ):
            value = record.get(field)
            if value is not None:
                _tally(
                    cell_metrics[bucket],
                    state_key(state_record(value, label=field)) == truth,
                )

    limitations = [
        "Conditional on the supplied cameras and reference geometry the E1 artifact was produced with; not an image-only product score.",
        "E1 patch rows are the reference-assisted exact-model sampler subset and are never a visible-cell total.",
        "The parity column reconstructs production's float32 observed patch from E1 raw arrays; E1's float64 _observed_patch is not reused.",
        "Only resolved_face observations carry patch digests and are joined to E1 face keys; a centre_patch_fallback observation must carry null observed_patch/valid/coverage/pattern, is refused otherwise, and is counted separately as production-only evidence, including any key collision, and is never counted as an E1 face join or rank row.",
        "Per-cell metrics cover only E1-labeled coordinates whose trace record provides the field; the production cell result still includes fallback evidence. Abstentions, missing records and OOV are counted separately.",
        "No recognizer is trained, no threshold is changed and nothing is promoted by this reader.",
    ]
    if sidecar_summary is not None:
        limitations.append(
            "The optional sidecar section joins decoded production payload bytes to E1 by exact "
            "(frame_id, coordinate, slot); identity joins are counted separately from raw/observed/"
            "valid/coverage pixel parity, so an identity match is never read as a pixel match."
        )
        limitations.append(
            "The optional sidecar numeric_similarity section reports bounded tolerant float32 "
            "RGB/coverage MAE and valid-mask pixel agreement per joined row, separate from exact "
            "digest parity; a low MAE is numeric similarity only and is never treated as semantic "
            "parity or as a correct state label."
        )

    summary = {
        "schema": REPLAY_SCHEMA,
        "evaluation_only": True,
        "mode": MODE,
        "role": role,
        "e1": {
            "path": e1["path"],
            "sha256": e1["sha256"],
            "artifact_sha256": e1["artifact_sha256"],
            "patch_count": len(e1_entries),
            "distinct_coordinates": e1_labeled_cells,
            "denominators": e1["denominators"],
        },
        "trace": {
            "path": trace["path"],
            "sha256": trace["sha256"],
            "schema": trace["schema"],
            "candidate_limit": trace["candidate_limit"],
            "final_palette_size": trace["final_palette_size"],
            "observation_count": len(trace["observations"]),
            "coordinate_count": len(trace["coordinate_records"]),
            "aggregation_coordinate_count": len(trace["aggregation"]),
        },
        "join": {
            "e1_patch_keys": len(e1_keys),
            "production_observation_keys": len(production_keys),
            "resolved_face_keys": len(resolved_keys),
            "fallback_observation_keys": len(fallback_keys),
            "intersection_keys": len(intersection),
            "e1_only_keys": len(e1_only),
            "production_only_keys": len(production_only),
            "fallback_only_keys": len(fallback_only),
            "fallback_key_collisions_with_e1": len(fallback_collisions),
            "face_patch_join_source": "resolved_face",
            "per_frame": per_frame,
            "per_slot": per_slot,
            "note": (
                "intersection/e1_only/production_only cover resolved_face observations only; "
                "centre_patch_fallback keys are counted separately (fallback_observation_keys, "
                "fallback_only_keys, fallback_key_collisions_with_e1) and are never E1 face joins"
            ),
        },
        "digest_equality": {
            "joined_key_denominator": len(intersection),
            "patch_observed_reconstruction": _rate_tally(digest["patch_observed_reconstruction"]),
            "patch_raw_vs_observed": _rate_tally(digest["patch_raw_vs_observed"]),
            "valid": _rate_tally(digest["valid"]),
            "coverage": _rate_tally(digest["coverage"]),
            "all_three_observed_contract": _rate_tally(digest["all_three_observed_contract"]),
            "partial_rows_reconstruction": _rate_tally(partial_rows),
            "fully_valid_rows_reconstruction": _rate_tally(fully_valid_rows),
            "mismatches": _examples(mismatch_rows, mismatch_total),
            "digest_contract": 'sha256(f"{shape}|{dtype.str}|" + C-order bytes)',
            "note": (
                "patch_observed_reconstruction is the like-for-like identity check; "
                "patch_raw_vs_observed compares E1's stored raw patch directly and differs "
                "by design on partially owned rows"
            ),
        },
        "truth_rank": {
            "joined_key_denominator": len(intersection),
            "truth_in_final_palette": truth_in_palette,
            "truth_out_of_vocabulary": truth_oov,
            "top1": rate(rank_hits[1], truth_in_palette),
            "top8": rate(rank_hits[PRODUCTION_TOP_K], truth_in_palette),
            "top16": rate(rank_hits[16], truth_in_palette),
            "top32": rate(rank_hits[32], truth_in_palette),
            "examples": _examples(rank_rows, truth_in_palette),
            "note": "Rank comes from the final scored cost vector only (cheaper first, then palette position) and only for joined resolved_face keys.",
        },
        "per_cell": {
            "denominator": (
                "E1-labeled distinct coordinates with a production coordinate record; each "
                "metric narrows further to rows where the truth is in the final palette and "
                "the field is present"
            ),
            "e1_labeled_coordinates": e1_labeled_cells,
            "coordinates_with_trace_record": cells_with_record,
            "coordinates_missing_trace_record": cells_missing_record,
            "fallback_observation_rows_in_labeled_coordinates": fallback_labeled_observations,
            "fallback_labeled_coordinates_with_evidence": len(fallback_labeled_coordinates),
            "truth_available_in_final_palette": truth_available_cells,
            "truth_out_of_vocabulary": truth_oov_cells,
            "candidate_abstentions": candidate_abstentions,
            "placement_abstentions": placement_abstentions,
            "top8_membership": _rate_tally(cell_metrics["top8_membership"]),
            "raw_winner_match": _rate_tally(cell_metrics["raw_winner_match"]),
            "post_smoothing_match": _rate_tally(cell_metrics["post_smoothing_match"]),
            "selected_state_match": _rate_tally(cell_metrics["selected_state_match"]),
            "final_state_match": _rate_tally(cell_metrics["final_state_match"]),
            "note": (
                "Outcomes come from the production coordinate trace, so fallback evidence "
                "remains part of the actual production cell result; each metric carries its own "
                "denominator and never infers the visible-cell total from patch rows."
            ),
        },
        "aggregation_replay": aggregation_replay,
        "production_top_k_check": top_k,
        "limitations": limitations,
    }
    if sidecar_summary is not None:
        summary["sidecar"] = sidecar_summary
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluator-only replay reader. Joins one sealed E1 patch-label artifact to one "
            "schema-2 production stage trace and prints a JSON reconciliation summary. The "
            "optional --sidecar validates and joins the E2C production patch sidecar, reporting "
            "exact digest parity and a separate bounded numeric similarity section. Reads "
            "no images, references, cameras, masks or JARs and writes no file."
        )
    )
    parser.add_argument("e1_artifact", type=Path, help="sealed E1 patch-label artifact JSON")
    parser.add_argument("trace", type=Path, help="schema-2 production stage-trace JSON")
    parser.add_argument(
        "--role",
        required=True,
        choices=list(ADMITTED_ROLES),
        help="reviewed E1 fixture role the artifact must declare",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "optional E2C production patch sidecar; validated against the exact trace bytes "
            "and joined to E1 by exact key"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = replay(
            args.e1_artifact, args.trace, role=args.role, sidecar_path=args.sidecar
        )
    except TraceReplayError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
