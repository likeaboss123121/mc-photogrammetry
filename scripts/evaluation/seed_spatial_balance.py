"""Standalone synthetic one-hop balance falsifier. No production imports or image IO.

Image adjacency is not evidence of physical coplanarity. The byte-identical
positive/corner pair intentionally falsifies this local aggregation hypothesis.
Charges are conservative logical retained-state charges, not process RSS.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np


TASK_SHA256 = "b753b1ad8570b9dd5a43096bcf4046f0483cc9807eb7c80864d36435318744d1"
SOURCE_SHA256 = "abfc4c10cdfff0b1d6b5bc9dc315b240ec4c94521d915a49fb36c84407dc957c"
CAPS = MappingProxyType(dict(frames=32, atoms_per_frame=20000, atoms=32000,
    edges=131072, visits=262144, retained_bytes=8 * 1024**2,
    output_bytes=4 * 1024**2, counter_max=2**63 - 1))
SCHEMA = "synthetic-spatial-balance-v1"


class Refusal(ValueError):
    pass


def _integer(value):
    return type(value) is int and 0 <= value <= CAPS["counter_max"]


class Budget:
    """Check before retaining/visiting; a refusal poisons this invocation."""
    def __init__(self, limits=None):
        self.limits = dict(CAPS)
        if limits is not None:
            if type(limits) is not dict or set(limits) - set(CAPS):
                raise Refusal("limits")
            for key, value in limits.items():
                if not _integer(value) or value > CAPS[key]:
                    raise Refusal("limits")
                self.limits[key] = value
        self.used = dict.fromkeys(CAPS, 0)
        self.failed = False

    def charge(self, key, count):
        if self.failed:
            raise Refusal("sticky refusal")
        if key not in self.used or not _integer(count):
            self.failed = True
            raise Refusal("integer overflow")
        prior = self.used[key]
        if count > self.limits["counter_max"] - prior or count > self.limits[key] - prior:
            self.failed = True
            raise Refusal("cap: " + key)
        self.used[key] = prior + count


@dataclass(frozen=True)
class Frame:
    """Already materialized synthetic graph fields; row index is canonical node ID."""
    supports: np.ndarray
    plane_scores: np.ndarray
    sides: np.ndarray
    node_area: np.ndarray
    node_lab: np.ndarray
    node_texture: np.ndarray
    node_xy: np.ndarray
    node_ids: np.ndarray
    adjacency: tuple


def _factors(supports, axis):
    b, c = [other for other in range(3) if other != axis]
    sa, sb, sc = supports[:, axis], supports[:, b], supports[:, c]
    in_plane = sb + sc
    total = in_plane + sa
    balance = 2.0 * np.sqrt(sb * sc) / np.maximum(in_plane, 1e-6)
    suppression = in_plane / np.maximum(total, 1e-6)
    reliability = in_plane / (in_plane + 24.0)
    return balance, suppression, reliability, in_plane


def source_scores(supports):
    """Independent literal region_growth._normal_scores expression, dtype/order intact."""
    if (type(supports) is not np.ndarray or supports.dtype not in (np.float32, np.float64)
            or supports.ndim != 2 or supports.shape[1] != 3):
        raise Refusal("support dtype/shape")
    if supports.shape[0] > CAPS["atoms_per_frame"]:
        raise Refusal("cap: atoms_per_frame")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scores = np.zeros_like(supports, dtype=np.float32)
            for axis in range(3):
                balance, suppression, reliability, _ = _factors(supports, axis)
                scores[:, axis] = np.clip(balance * suppression * reliability, 0.0, 1.0)
            if not np.isfinite(scores).all():
                raise Refusal("invalid score")
            return scores
    except FloatingPointError as exc:
        raise Refusal("invalid arithmetic") from exc


def _array(array, shape, dtypes, name):
    if (type(array) is not np.ndarray or array.shape != shape
            or array.dtype not in dtypes or not array.flags.c_contiguous):
        raise Refusal("array: " + name)
    if not np.isfinite(array).all():
        raise Refusal("nonfinite: " + name)


def _validate(frame, budget):
    if type(frame) is not Frame or type(frame.adjacency) is not tuple:
        raise Refusal("frame")
    if type(frame.supports) is not np.ndarray or frame.supports.ndim != 2:
        raise Refusal("supports")
    n = frame.supports.shape[0]
    if n > budget.limits["atoms_per_frame"]:
        raise Refusal("cap: atoms_per_frame")
    budget.used["atoms_per_frame"] = max(budget.used["atoms_per_frame"], n)
    budget.charge("atoms", n)
    if len(frame.adjacency) != n:
        raise Refusal("adjacency shape")
    # Before any array-sized validation masks, replay arrays, dictionaries or output rows.
    budget.charge("retained_bytes", n * 16384 + 1024)
    floats = (np.dtype("float32"), np.dtype("float64"))
    _array(frame.supports, (n, 3), floats, "supports")
    _array(frame.plane_scores, (n, 3), (np.dtype("float32"),), "plane_scores")
    _array(frame.sides, (n, 3), (np.dtype("int8"), np.dtype("int32"), np.dtype("int64")), "sides")
    _array(frame.node_area, (n,), (np.dtype("int32"), np.dtype("int64")), "node_area")
    _array(frame.node_lab, (n, 3), floats, "node_lab")
    _array(frame.node_texture, (n,), floats, "node_texture")
    _array(frame.node_xy, (n, 2), floats, "node_xy")
    _array(frame.node_ids, (n,), (np.dtype("int32"), np.dtype("int64")), "node_ids")
    if not np.array_equal(frame.node_ids, np.arange(n)):
        raise Refusal("duplicate/out-of-range/noncanonical node IDs")
    if ((frame.supports < 0).any() or (frame.node_area <= 0).any()
            or (frame.sides < -1).any() or (frame.sides > 1).any()
            or (frame.node_texture < 0).any()):
        raise Refusal("range/area")
    baseline = source_scores(frame.supports)
    if baseline.tobytes() != frame.plane_scores.tobytes():
        raise Refusal("baseline byte mismatch")
    edges = []
    for node, rows in enumerate(frame.adjacency):
        if type(rows) is not tuple:
            raise Refusal("adjacency rows")
        # Check size without retaining an over-cap row or dictionary.
        if len(rows) > n - 1:
            raise Refusal("duplicate/out-of-range adjacency")
        budget.charge("retained_bytes", len(rows) * 1024 + 64)
        rowmap = {}
        for row in rows:
            budget.charge("visits", 1)
            if type(row) is not tuple or len(row) != 6 or not _integer(row[0]):
                raise Refusal("adjacency type")
            neighbor = row[0]
            if neighbor >= n or neighbor == node or neighbor in rowmap:
                raise Refusal("duplicate/out-of-range adjacency")
            if any(type(x) not in (float, int) or not math.isfinite(x) for x in row[1:]):
                raise Refusal("nonfinite adjacency")
            if any(x < 0 for x in row[1:4]) or any(abs(x) > 1 for x in row[4:]):
                raise Refusal("adjacency range")
            if node < neighbor:
                budget.charge("edges", 1)
            rowmap[neighbor] = row[1:]
        edges.append(rowmap)
    # Reverse lookups are O(1); every retained directed row was charged above.
    for node, rows in enumerate(edges):
        for neighbor, values in rows.items():
            budget.charge("visits", 1)
            if edges[neighbor].get(node) != values:
                raise Refusal("asymmetric adjacency")
    return edges


def competing(scores, node, axis):
    candidate = float(scores[node, axis])
    other = max(float(scores[node, a]) for a in range(3) if a != axis)
    # Deliberately source Python-float addition, NOT float32 subtraction.
    return other >= 0.26 and other >= candidate + 0.035


def _reject(frame, node, neighbor, axis, values, spacing):
    if frame.sides[node, axis] == 0 or frame.sides[neighbor, axis] == 0:
        return "zero_side"
    if frame.sides[node, axis] != frame.sides[neighbor, axis]:
        return "side"
    # Float64 image-space centroid distance; overflow is refused by caller.
    delta = frame.node_xy[node].astype(np.float64) - frame.node_xy[neighbor].astype(np.float64)
    distance = np.sqrt(np.sum(delta * delta, dtype=np.float64))
    if not np.isfinite(distance):
        raise Refusal("invalid distance")
    if distance > 2.0 * spacing:
        return "distance"
    for value, threshold, reason in zip(values[:3], (48.0, 40.0, 28.0),
                                       ("gradient", "lab", "texture")):
        if value >= threshold:
            return reason
    if competing(frame.plane_scores, node, axis):
        return "target_competing"
    if competing(frame.plane_scores, neighbor, axis):
        return "donor_competing"
    return None


def _seed(score, opponent, tangent_support, side):
    # Operate on float32 arrays just as source does, including scalar comparisons.
    margin = score - opponent
    eligible = ((score >= 0.26) & (margin >= 0.035)
                & (tangent_support >= 12.0) & (side != 0))
    return margin, eligible


def replay(frames, *, atom_spacing=16.0, limits=None):
    """Return bounded provenance. Inputs contain no evaluator truth or world metadata."""
    budget = Budget(limits)
    if type(frames) is not tuple:
        raise Refusal("frames must be materialized tuple")
    budget.charge("frames", len(frames))
    if (type(atom_spacing) not in (float, int) or not math.isfinite(atom_spacing)
            or atom_spacing != 16.0):
        raise Refusal("frozen atom spacing")
    output = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for frame_id, frame in enumerate(frames):
                edges = _validate(frame, budget)
                n = len(frame.node_area)
                factors = [_factors(frame.supports, axis) for axis in range(3)]
                opponents = [np.max(np.delete(frame.plane_scores, axis, axis=1), axis=1)
                             for axis in range(3)]
                rows = []
                for node in range(n):
                    donors = [[node] for _ in range(3)]
                    rejects = [[] for _ in range(3)]
                    for neighbor in sorted(edges[node]):
                        budget.charge("visits", 1)  # one directed pooling visit, all 3 axes
                        for axis in range(3):
                            reason = _reject(frame, node, neighbor, axis,
                                             edges[node][neighbor], float(atom_spacing))
                            if reason is None:
                                donors[axis].append(neighbor)
                            else:
                                rejects[axis].append([neighbor, reason])
                    for axis in range(3):
                        admitted = sorted(donors[axis])
                        b, c = [a for a in range(3) if a != axis]
                        db, dc = np.float64(0), np.float64(0)
                        for donor in admitted:
                            area = np.float64(frame.node_area[donor])
                            db = db + np.float64(frame.supports[donor, b]) / area
                            dc = dc + np.float64(frame.supports[donor, c]) / area
                        db = db / np.float64(len(admitted))
                        dc = dc / np.float64(len(admitted))
                        pooled = 2.0 * np.sqrt(db * dc) / np.maximum(db + dc, 1e-6)
                        if not np.isfinite(pooled):
                            raise Refusal("invalid pooling")
                        # Length-one slices preserve numpy source dtype and operation order.
                        balance, suppression, reliability, support = factors[axis]
                        new_balance = np.maximum(balance[node:node+1],
                            np.asarray([pooled], dtype=frame.supports.dtype))
                        candidate = np.asarray(np.clip(new_balance * suppression[node:node+1]
                            * reliability[node:node+1], 0.0, 1.0), dtype=np.float32)
                        original = frame.plane_scores[node:node+1, axis]
                        opponent = opponents[axis][node:node+1]
                        baseline_margin, baseline_seed = _seed(original, opponent,
                            support[node:node+1], frame.sides[node:node+1, axis])
                        margin, seed = _seed(candidate, opponent, support[node:node+1],
                                             frame.sides[node:node+1, axis])
                        rows.append(dict(node=node, axis=axis, side=int(frame.sides[node, axis]),
                            area=int(frame.node_area[node]), donors=admitted, rejected=rejects[axis],
                            original_balance=float(balance[node]), pooled_balance=float(pooled),
                            balance=float(new_balance[0]), suppression=float(suppression[node]),
                            reliability=float(reliability[node]), tangent_support=float(support[node]),
                            normal_support=float(frame.supports[node, axis]),
                            opponent=float(opponent[0]), baseline_score=float(original[0]),
                            score=float(candidate[0]), baseline_margin=float(baseline_margin[0]),
                            margin=float(margin[0]), baseline_seed=bool(baseline_seed[0]),
                            seed=bool(seed[0])))
                # Mass is an unchanged-array fingerprint and original support sum, not pooled density mass.
                output.append(dict(frame=frame_id, support_dtype=frame.supports.dtype.name,
                    support_sha256=hashlib.sha256(frame.supports.tobytes()).hexdigest(),
                    support_mass=[float(np.sum(frame.supports[:, a], dtype=np.float64)) for a in range(3)],
                    rows=rows))
    except FloatingPointError as exc:
        raise Refusal("invalid arithmetic") from exc
    report = dict(schema=SCHEMA, task_sha256=TASK_SHA256, source_sha256=SOURCE_SHA256,
                  limits=budget.limits, resources=budget.used, frames=output)
    # Enforce canonical output bound before returning the diagnostic object.
    for _ in range(8):
        size = _canonical_size(report, budget.limits["output_bytes"])
        if size == budget.used["output_bytes"]:
            break
        budget.used["output_bytes"] = size
    else:
        raise Refusal("output accounting")
    seal(report, output_limit=budget.limits["output_bytes"])
    return report


@dataclass(frozen=True)
class Sealed:
    payload: bytes
    sha256: str


def _canonical_scalars(value, depth=0, work=None):
    if work is None:
        work = [0]
    if work[0] >= CAPS["visits"]:
        raise Refusal("canonical work")
    work[0] += 1
    if depth > 12:
        raise Refusal("canonical depth")
    if value is None or type(value) is bool:
        return
    if type(value) is int and abs(value) <= CAPS["counter_max"]:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and len(value) <= 256:
        return
    if type(value) in (list, dict):
        if len(value) > CAPS["visits"]:
            raise Refusal("canonical container")
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str or len(key) > 64:
                    raise Refusal("canonical key")
                _canonical_scalars(child, depth + 1, work)
        else:
            for child in value:
                _canonical_scalars(child, depth + 1, work)
        return
    raise Refusal("canonical scalar")


def _canonical_size(report, output_limit):
    _canonical_scalars(report)
    count = 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(report):
        size = len(chunk.encode("utf-8"))
        if size > output_limit - count:
            raise Refusal("cap: output_bytes")
        count += size
    return count


def seal(report, *, output_limit=CAPS["output_bytes"]):
    if not _integer(output_limit) or output_limit > CAPS["output_bytes"]:
        raise Refusal("output limit")
    _canonical_size(report, output_limit)  # no retained output until full bounded preflight
    payload_buffer, count = bytearray(), 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        for chunk in encoder.iterencode(report):
            encoded = chunk.encode("utf-8")
            if len(encoded) > output_limit - count:
                raise Refusal("cap: output_bytes")
            count += len(encoded)
            payload_buffer.extend(encoded)
    except (ValueError, TypeError, RecursionError) as exc:
        raise Refusal("noncanonical output") from exc
    payload = bytes(payload_buffer)
    return Sealed(payload, hashlib.sha256(payload).hexdigest())


def _readback(path, payload):
    with open(path, "rb") as handle:
        actual = handle.read(len(payload) + 1)
    if actual != payload:
        raise Refusal("readback mismatch")


def publish(sealed, destination):
    """Create-new, fsync, hard-link publication. Never overwrite an existing output."""
    if (type(sealed) is not Sealed or type(sealed.payload) is not bytes
            or len(sealed.payload) > CAPS["output_bytes"]
            or hashlib.sha256(sealed.payload).hexdigest() != sealed.sha256):
        raise Refusal("invalid seal")
    destination = Path(destination)
    if destination.exists():
        raise Refusal("destination exists")
    temporary, published = None, False
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".spatial-", dir=destination.parent)
        with os.fdopen(descriptor, "wb") as handle:
            if handle.write(sealed.payload) != len(sealed.payload):
                raise Refusal("short write")
            handle.flush()
            os.fsync(handle.fileno())
        _readback(temporary, sealed.payload)
        os.link(temporary, destination)
        published = True
        _readback(destination, sealed.payload)
    except (OSError, Refusal) as exc:
        if published and destination.exists() and os.path.samefile(temporary, destination):
            destination.unlink()
        raise Refusal("publication failure") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class Truth:
    """Evaluator-only per-node plane identity and signed Cartesian plane normal."""
    plane_ids: tuple
    normals: tuple
    positive: bool


def evaluate(sealed, truth):
    """Accept truth only after helper output has been canonicalized and hashed."""
    if (type(sealed) is not Sealed or type(sealed.payload) is not bytes
            or len(sealed.payload) > CAPS["output_bytes"]
            or hashlib.sha256(sealed.payload).hexdigest() != sealed.sha256):
        raise Refusal("unsealed evaluator input")
    try:
        report = json.loads(sealed.payload)
    except (ValueError, TypeError) as exc:
        raise Refusal("invalid seal JSON") from exc
    if seal(report) != sealed or report.get("schema") != SCHEMA:
        raise Refusal("noncanonical seal")
    if type(truth) is not tuple or len(truth) != len(report["frames"]):
        raise Refusal("truth closure")
    cases, new, correct_new, false_new = [], 0, 0, 0
    baseline_retained = True
    for frame, labels in zip(report["frames"], truth):
        if type(labels) is not Truth or type(labels.positive) is not bool:
            raise Refusal("truth type")
        n = len(frame["rows"]) // 3
        if len(labels.plane_ids) != n or len(labels.normals) != n:
            raise Refusal("truth closure")
        if any(type(p) is not int or p < 0 for p in labels.plane_ids):
            raise Refusal("truth plane IDs")
        if any(type(v) is not tuple or len(v) != 3 or any(type(x) is not int or x not in (-1, 0, 1)
               for x in v) or sum(abs(x) for x in v) != 1 for v in labels.normals):
            raise Refusal("truth normals")
        positive_area, before, after, false = 0, 0, 0, 0
        for row in frame["rows"]:
            node, axis = row["node"], row["axis"]
            matches = labels.normals[node][axis] == row["side"] and row["side"] != 0
            coplanar = all(labels.plane_ids[d] == labels.plane_ids[node] for d in row["donors"])
            good = matches and coplanar
            if matches:
                positive_area += row["area"]
                before += row["area"] * row["baseline_seed"]
                after += row["area"] * (row["seed"] and good)
            baseline_retained &= not row["baseline_seed"] or row["seed"]
            if row["seed"] and not row["baseline_seed"]:
                new += 1
                correct_new += int(good)
                false_new += int(not good)
                false += int(not good)
        cases.append(dict(positive=labels.positive, false_new_seeds=false,
            recall_gain=(after - before) / positive_area if positive_area else 0.0))
    gates = dict(positive_recall=sum(c["positive"] and c["recall_gain"] >= 0.1 for c in cases) >= 3,
        new_seed_precision=(correct_new / new if new else 1.0) >= 0.99,
        zero_false_plane_seeds=all(c["false_new_seeds"] == 0 for c in cases if not c["positive"]),
        baseline_retained=bool(baseline_retained))
    return dict(helper_sha256=sealed.sha256, cases=cases, new_seeds=new,
        correct_new_seeds=correct_new, false_new_seeds=false_new, gates=gates,
        hypothesis_falsified=not gates["zero_false_plane_seeds"],
        disposition="stop_before_image_capture" if not all(gates.values()) else "review_only")


def _synthetic(supports, *, edges=((0, 1),), sides=None, areas=None, gradient=0.0):
    support = np.array(supports, dtype=np.float32)
    n = len(support)
    rows = [[] for _ in range(n)]
    for a, b in edges:
        rows[a].append((b, gradient, 0.0, 0.0, 1.0, 0.0))
        rows[b].append((a, gradient, 0.0, 0.0, 1.0, 0.0))
    return Frame(support, source_scores(support),
        np.array(sides if sides is not None else [[1, 1, 1]] * n, dtype=np.int8),
        np.array(areas if areas is not None else [256] * n, dtype=np.int32),
        np.zeros((n, 3), np.float32), np.zeros(n, np.float32),
        np.array([[16 * i, 0] for i in range(n)], dtype=np.float32),
        np.arange(n, dtype=np.int32), tuple(tuple(row) for row in rows))


FIXTURE_NAMES = ("planar_balanced", "planar_imbalanced", "planar_split", "planar_unequal_area",
    "stripe_b", "stripe_c", "sharp_corner", "indistinguishable_same_color_corner",
    "disconnected", "noise", "boundary", "zero_side")


def synthetic_matrix():
    """Frozen 12 cases. Returned truth is separate and cannot be passed to replay."""
    split = [(0, 24, 0), (0, 0, 24)]
    frames = (
        _synthetic([(0, 12, 12), (0, 12, 12)]),
        _synthetic([(0, 23.9, 0.1), (0, 0.1, 23.9)]),
        _synthetic(split),
        _synthetic([(0, 24, 0), (0, 0, 48)], areas=[256, 512]),
        _synthetic([(0, 24, 0), (0, 24, 0)]),
        _synthetic([(0, 0, 24), (0, 0, 24)]),
        _synthetic(split, gradient=48.0),
        _synthetic(split),  # EXACT helper-visible paired copy of planar_split
        _synthetic(split, edges=()),
        _synthetic([(0, 0.1, 0.1), (0, 0.2, 0.2)]),
        _synthetic(split, gradient=48.0),
        _synthetic(split, sides=[[0, 0, 0], [0, 0, 0]]),
    )
    truth = tuple(Truth((0, 1) if i in (6, 7) else (0, 0),
                  ((0, 1, 0), (0, 0, 1)) if i in (6, 7) else ((1, 0, 0), (1, 0, 0)), i < 4)
                  for i in range(12))
    return frames, truth
