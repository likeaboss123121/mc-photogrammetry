"""Private, bounded receipt for one normally completed admitted matcher universe.

No production caller uses this module. Completeness is limited to PlaneMatch
admission in the owned call, and is a correctness contract, not authentication.
Input and evidence reservations are logical conservative bounds, not RSS claims.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from mc_photogrammetry.pipeline import image_derived_connections as connections
from mc_photogrammetry.pipeline import image_derived_graph as graph_module
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_module
from mc_photogrammetry.pipeline import image_derived_session as session_module
from mc_photogrammetry.pipeline import image_derived_signatures as signature_module
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph

SCHEMA = "mc-photogrammetry-admitted-matcher-call-receipt-v1"
DOMAIN = "one-normal-return-complete-admitted-plane-match-universe-v1"
CAP_VERSION = "dscale-07-separate-logical-workspace-v1"
# Same frozen definition as DSCALE ConstraintCaps.canonical_input_bytes. The
# search module is deliberately not imported into this producer seam.
INPUT_EVIDENCE_BYTES = 16 * 1024 * 1024
CORRELATION_WORKSPACE_BYTES = 32 * 1024 * 1024
SOURCE_PINS = (
    ("src/mc_photogrammetry/reconstruction/plane_graph.py", "6868ac42c657c2ca270e9d91a36c4f7958e2ad4ad9014b391bff415ce4a646a8"),
    ("src/mc_photogrammetry/pipeline/image_derived_graph.py", "7549c7177a82f8ef452ae9fa4581f6a443d290a5e3ae9af9403de5335eb3a6b6"),
    ("src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py", "cbd2d8e2e8e744799628159b4eb307207024435272441d4b90215bb6f10add1e"),
    ("src/mc_photogrammetry/reconstruction/_same_axis_constraint_search.py", "e3440b8225dd525f7c25b7eab3b634861da522ef809cd2bf91e4e499396bd00a"),
    # Matching, reading identity and every canonical context serializer used
    # below, including delegated ingress records and local-origin semantics.
    ("src/mc_photogrammetry/reconstruction/lattice_planes.py", "c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b"),
    ("src/mc_photogrammetry/reconstruction/lattice.py", "76a0c0f6a0bc057fbbc75216b7dbcdfef637d69f7e484d92ba373c3a6eb45c9b"),
    ("src/mc_photogrammetry/reconstruction/plane_placement.py", "743fd95a18651a342642ced43701803cda379a2d392604557dc6069ba3909880"),
    ("src/mc_photogrammetry/reconstruction/camera.py", "0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680"),
    ("src/mc_photogrammetry/reconstruction/calibration.py", "2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc"),
    ("src/mc_photogrammetry/pipeline/image_derived_session.py", "e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e"),
    ("src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py", "fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e"),
    ("src/mc_photogrammetry/pipeline/image_derived_signatures.py", "7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f"),
    ("src/mc_photogrammetry/pipeline/image_derived_connections.py", "b46d51a17f45a292666ef089199dca120cc11b91125d028e5114d0c6a010aedd"),
    ("src/mc_photogrammetry/pipeline/frame_ingress.py", "3411b4067281b19a6ac2e530ece5dfc2e53e0f63b978e83e09ffadb27f715f0a"),
    ("src/mc_photogrammetry/pipeline/contracts.py", "db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc"),
    ("src/mc_photogrammetry/pipeline/image_only.py", "ca28afd0e9dc36d2b827a28807b736d635374acb2787506c2d2f3d07c929c93c"),
    ("src/mc_photogrammetry/pipeline/joint_matching.py", "9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3"),
)


@dataclass(frozen=True, slots=True)
class SeamCaps:
    readings: int = 256
    frames: int = 32
    alternatives_per_reading: int = 64
    candidate_identities: int = 4096
    hypotheses: int = 16384
    attempts: int = 16384
    input_evidence_bytes: int = INPUT_EVIDENCE_BYTES
    correlation_workspace_bytes: int = CORRELATION_WORKSPACE_BYTES


HARD_CAPS = SeamCaps()


@dataclass(frozen=True, slots=True)
class AcceptedCallContext:
    """Already validated upstream records and independently retained pins.

    The adapter verifies content pins/relations without reading frame pixels,
    structural patches, or re-evaluating upstream numerical producers.
    """
    session: session_module.ImageDerivedSession
    lattice_result: lattice_module.ImageDerivedLatticeResult
    signatures_result: signature_module.ImageDerivedSignaturesResult
    connection_graph: connections.ImageDerivedConnectionGraph
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    graph_sha256: str


@dataclass(frozen=True, slots=True)
class Reservation:
    hypotheses: int
    structural_skips: int
    attempts: int
    potential_candidates: int
    input_evidence_bytes: int
    same_axis_pair_checks: int
    correlation_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class Ledger:
    hypotheses: int
    structural_skips: int
    attempts: int
    admitted: int
    kept: int
    dropped: int
    pair_refusal_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class MatcherCallReceipt:
    schema: str
    domain: str
    cap_version: str
    input_record: bytes
    evidence_record: bytes
    input_sha256: str
    admitted_sha256: str
    kept_sha256: str
    receipt_sha256: str
    ledger: Ledger
    reservation: Reservation
    candidates: tuple[plane_graph.PlaneMatch, ...]
    dispositions: tuple[str, ...]
    returned_matches: tuple[plane_graph.PlaneMatch, ...]
    refusals: tuple[plane_graph.PlaneMatchRefusal, ...]
    reconstruction_authorized: bool = False


@dataclass(frozen=True, slots=True)
class SeamResult:
    status: str
    reason: str = ""
    receipt: MatcherCallReceipt | None = None
    matches: tuple[plane_graph.PlaneMatch, ...] = ()
    refusals: tuple[plane_graph.PlaneMatchRefusal, ...] = ()
    reconstruction_authorized: bool = False


class _Refusal(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _Refusal(reason)


def _json(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(record):
    return hashlib.sha256(_json(record)).hexdigest()


def _scan_bound(value, limit):
    """Size-only bounded traversal; never a serializer or canonical workspace.

    A depth-first iterator stack has fixed bounded depth. Repeated references
    are charged repeatedly; cycles/depth and unsupported values refuse.
    """
    stack = [(iter((value,)), 0)]
    size = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        _require(depth <= 24, "invalid_input")
        size += max(64, sys.getsizeof(item))
        if isinstance(item, np.ndarray):
            _require(item.dtype.kind in "bfiu" and item.ndim == 2,
                     "invalid_signature")
            size += 4 * item.nbytes
        elif type(item) is str:
            size += 6 * len(item)
        elif type(item) is bytes:
            size += 8 * len(item)
        elif item is None or type(item) is bool:
            pass
        elif isinstance(item, (int, float, np.integer, np.floating)):
            _require(not isinstance(item, np.bool_) and math.isfinite(float(item)),
                     "invalid_input")
        elif type(item) is tuple or type(item) is list:
            stack.append((iter(item), depth + 1))
        elif isinstance(item, Mapping):
            stack.append((iter(item.items()), depth + 1))
        elif is_dataclass(item) and not isinstance(item, type):
            _require(item.__dataclass_params__.frozen, "invalid_input")
            stack.append((iter(tuple(getattr(item, field.name) for field in fields(item))),
                          depth + 1))
        else:
            raise _Refusal("unpinable_input")
        _require(size <= limit, "input_evidence_bytes_cap")
    return size


def _context_scan_surface(context):
    if context is None:
        return ()
    _require(type(context) is AcceptedCallContext, "invalid_context")
    session, lattice, signatures, graph = (context.session, context.lattice_result,
        context.signatures_result, context.connection_graph)
    _require(type(session) is session_module.ImageDerivedSession
             and type(lattice) is lattice_module.ImageDerivedLatticeResult
             and type(signatures) is signature_module.ImageDerivedSignaturesResult
             and type(graph) is connections.ImageDerivedConnectionGraph,
             "invalid_context")
    _require(lattice.status == signatures.status == "assembled"
             and graph.status == "ok", "upstream_refusal")
    _require(len(session.frames) <= 32 and len(session.input_frames) <= 32
             and len(graph.frame_ids) <= 32 and len(graph.evidence) <= 496,
             "frames_cap")
    # All pixel/patch buffers stay outside this seam's read surface.
    frame_metadata = tuple((frame.frame_id, frame.source_label, frame.image_shape,
                            frame.image_sha256, frame.ingress)
                           for frame in session.input_frames)
    return (frame_metadata, session.schema, session.version, session.cameras,
            session.local_cameras, session.components, session.unresolved,
            session.admissions, lattice.camera_calibrations, lattice.abstentions,
            lattice.reason, lattice.detail, signatures.abstentions,
            signatures.reason, signatures.detail, graph,
            context.session_sha256, context.lattice_sha256,
            context.signatures_sha256, context.graph_sha256)


def _workspace_add(left, right):
    _require(type(left) is int and type(right) is int
             and 0 <= left <= sys.maxsize and 0 <= right <= sys.maxsize - left,
             "correlation_workspace_overflow")
    return left + right


def _workspace_mul(left, right):
    _require(type(left) is int and type(right) is int
             and 0 <= left <= sys.maxsize and 0 <= right <= sys.maxsize
             and (left == 0 or right <= sys.maxsize // left),
             "correlation_workspace_overflow")
    return left * right


def _pair_workspace(shape, itemsize, shift):
    """Source-accounted logical arrays, not available RAM or process RSS.

    N=R*C, P=(R+4S)*(C+4S), F=padded_R*(padded_C//2+1),
    T=(2S+1)**2. B is the largest cells/coverage itemsize of the pair.
    Sum across phases (rather than relying on timely frees):
    * warp_signature: 16N grids, 48N stack inputs/output, 24N mapped,
      16N divided coordinates, 8N boolean expressions, 5N output,
      48N indices/round/clip/flat temporaries, 2BN source gathers;
    * match_plane_pair: 5N retained warp, (24+B)N where/cast maps,
      16N indexed/std temporaries;
    * correlation: 2N masks, 16N values, 16N where temporaries,
      32N indexed mean/subtract temporaries; 18P padded maps/masks,
      8P inverse result, 3P roll/and temporaries;
    * spectra: (32+16+16+8+1+8+16+16)F for both spectra, cross,
      conjugate, magnitude, safe, denominator, divide and replacement;
      another 32F covers transform intermediate arrays;
    * surface/support/finite/admissible/ranked/mask/sidelobes/std:
      (16+1+1+8+1+8+8)T; eight 3x3 float64 matrix equivalents
      cover the fixed homography expressions.
    No witness sinks are supplied, and active period capture is refused.
    FFT-library scratch, allocator retention, unrelated callers, Python/OS
    overhead are unmeasured/excluded. This is a logical policy reservation.
    All arithmetic stays in checked Python integers before any NumPy work.
    """
    _require(type(shape) in (tuple, list) and len(shape) == 2
             and all(type(value) is int and value > 0 for value in shape)
             and type(itemsize) is int and itemsize > 0
             and type(shift) is int and shift >= 0,
             "correlation_workspace_overflow")
    rows, columns = shape
    n = _workspace_mul(rows, columns)
    halo = _workspace_mul(4, shift)
    padded_rows = _workspace_add(rows, halo)
    padded_columns = _workspace_add(columns, halo)
    p = _workspace_mul(padded_rows, padded_columns)
    f = _workspace_mul(padded_rows, _workspace_add(padded_columns // 2, 1))
    span = _workspace_add(_workspace_mul(2, shift), 1)
    t = _workspace_mul(span, span)
    total = 576
    for count, width in ((n, _workspace_add(276, _workspace_mul(3, itemsize))),
                         (p, 29), (f, 145), (t, 43)):
        total = _workspace_add(total, _workspace_mul(count, width))
    return total


def _preflight(readings, signatures, caps, context, max_shift_cells, threshold):
    _require(lattice_planes.lattice._period_support_state() is None,
             "active_period_support_capture")
    _require(type(caps) is SeamCaps, "invalid_caps")
    for field in fields(caps):
        value = getattr(caps, field.name)
        _require(type(value) is int and 0 < value <= getattr(HARD_CAPS, field.name),
                 "invalid_caps")
    _require(type(readings) is tuple and type(signatures) is tuple, "invalid_input")
    _require(len(readings) <= caps.readings, "readings_cap")
    # This also bounds signatureless/single-frame signature occurrences before
    # scanning; a frame cannot supply an unbounded input with H=0.
    _require(len(signatures) <= caps.readings * caps.alternatives_per_reading,
             "signature_occurrences_cap")
    _require(type(max_shift_cells) is int and max_shift_cells >= 0,
             "invalid_settings")
    _require(type(threshold) in (int, float) and math.isfinite(threshold),
             "invalid_settings")
    frames = set()
    for reading in readings:
        _require(type(reading) is lattice_planes.PlaneReading
                 and type(reading.alternatives) is tuple, "invalid_reading")
        _require(len(reading.alternatives) <= caps.alternatives_per_reading,
                 "alternatives_per_reading_cap")
        _require(type(reading.frame) is str and bool(reading.frame), "invalid_reading")
        frames.add(reading.frame)
        _require(len(frames) <= caps.frames, "frames_cap")
    size = _scan_bound((readings, _context_scan_surface(context)),
                       caps.input_evidence_bytes)
    counts = {frame: 0 for frame in frames}
    for signature in signatures:
        _require(type(signature) is lattice_planes.PlaneSignature
                 and any(signature.reading is reading for reading in readings),
                 "foreign_signature")
        _require(plane_graph._is_finite_positive_number(signature.blocks_per_period),
                 "invalid_signature")
        size += _scan_bound((signature.blocks_per_period, signature.cells,
            signature.interior, signature.coverage, signature.origin_uv,
            signature.region_footprint), caps.input_evidence_bytes - size)
        _require(all(type(array) is np.ndarray and array.ndim == 2
                     and array.shape == signature.cells.shape
                     for array in (signature.cells, signature.interior, signature.coverage))
                 and signature.cells.size > 0, "invalid_signature")
        counts[signature.reading.frame] += 1
    ordered_frames = sorted(frames)
    h = 4 * sum(counts[left] * counts[right]
                for i, left in enumerate(ordered_frames) for right in ordered_frames[i + 1:])
    _require(h <= caps.hypotheses, "hypotheses_cap")
    _require(8 * size + 65536 <= caps.input_evidence_bytes,
             "input_evidence_bytes_cap")
    by_frame = {frame: [] for frame in ordered_frames}
    for signature in signatures:
        by_frame[signature.reading.frame].append(signature)
    e = possible = workspace = 0
    # Bounded H loop, before trace/sinks/candidate workspace exist.
    for i, left_frame in enumerate(ordered_frames):
        for right_frame in ordered_frames[i + 1:]:
            for left in by_frame[left_frame]:
                for right in by_frame[right_frame]:
                    if (left.reading.structural_share < threshold
                            and right.reading.structural_share < threshold):
                        e += 4
                        continue
                    for yaw in range(4):
                        axis, _sign = plane_graph.axis_transform(left.reading.axis, yaw)
                        if (axis == right.reading.axis
                                and left.reading.distance_for(left.blocks_per_period) is not None
                                and right.reading.distance_for(right.blocks_per_period) is not None):
                            workspace = max(workspace, _pair_workspace(right.cells.shape,
                                max(8, *(array.itemsize for row in (left, right)
                                    for array in (row.cells, row.coverage))), max_shift_cells))
                        if (axis == right.reading.axis
                                and min(left.reading.structural_share, right.reading.structural_share)
                                    >= plane_graph.MIN_STRUCTURAL_SHARE
                                and left.reading.distance_for(left.blocks_per_period) is not None
                                and right.reading.distance_for(right.blocks_per_period) is not None):
                            possible += 1
    a = h - e
    _require(a <= caps.attempts, "attempts_cap")
    _require(possible <= caps.candidate_identities, "potential_candidate_identities_cap")
    # Includes copies, canonical JSON, trace rows/refusal detail, all candidates,
    # digest temporaries and bounded filter/validation workspace. Correlation
    # arrays have their independent logical cap; matcher pairs run sequentially.
    identifier_bytes = max((6 * len(reading.frame) for reading in readings), default=0)
    reserved = (8 * size + a * (4096 + 16 * identifier_bytes)
                + possible * (8192 + 24 * identifier_bytes)
                + 65536)
    _require(reserved <= caps.input_evidence_bytes, "input_evidence_bytes_cap")
    _require(workspace <= caps.correlation_workspace_bytes,
             "correlation_workspace_bytes_cap")
    return Reservation(h, e, a, possible, reserved, possible * (possible - 1) // 2, workspace)


def _reading_payload(reading):
    """Fixed PlaneReading schema, including every retained field/evidence."""
    rectification = reading.rectification
    return {"frame": reading.frame, "axis": reading.axis, "side": reading.side,
        "alternatives": [[alt.blocks_per_period, alt.distance_blocks, alt.support]
                         for alt in reading.alternatives],
        "period_pixels": reading.period_pixels, "phase_pixels": reading.phase_pixels,
        "extent": reading.extent, "samples": reading.samples, "polygon": reading.polygon,
        "coverage_cells": reading.coverage_cells, "structural_share": reading.structural_share,
        "confidence": reading.confidence,
        "rectification": {"axis_matrix": rectification.axis_matrix,
            "focal": rectification.focal, "principal_point": rectification.principal_point,
            "centre": rectification.centre},
        "region_evidence": None if reading.region_evidence is None
            else reading.region_evidence.to_record(),
        "phase_evidence": None if reading.phase_evidence is None
            else asdict(reading.phase_evidence)}


def _signature_payload(signature, input_index, local_index):
    reading = signature.reading
    alternatives = tuple(index for index, alt in enumerate(reading.alternatives)
                         if alt in reading.matching_alternatives(signature.blocks_per_period))
    # Reuse the existing canonical signature serializer. Invalid/ambiguous b
    # bindings are pinned as such and may produce normal pair refusals.
    proxy = SimpleNamespace(signature=signature, frame_id=reading.frame,
        canonical_index=input_index, reading_axis=reading.axis,
        alternative_index=alternatives[0] if len(alternatives) == 1 else -1,
        blocks_per_period=signature.blocks_per_period)
    payload = signature_module._signature_payload(proxy)
    return {"payload": payload, "sha256": _digest(payload), "frame": reading.frame,
            "plane": local_index, "axis": reading.axis, "b": signature.blocks_per_period,
            "alternative_indices": alternatives, "structural_share": reading.structural_share}


def _context_payload(context, readings, signature_rows):
    if context is None:
        return None
    session, lattice, signatures, graph = (context.session, context.lattice_result,
        context.signatures_result, context.connection_graph)
    pins = (context.session_sha256, context.lattice_sha256,
            context.signatures_sha256, context.graph_sha256)
    _require(all(plane_graph._sha256_hex(pin) for pin in pins), "invalid_pin")
    session_payload = session_module._session_payload(session)
    lattice_payload = lattice_module._result_payload(lattice)
    signatures_payload = signature_module._result_payload(signatures)
    _require(session_module._digest(session_payload) == context.session_sha256
             == session.session_sha256, "stale_session_pin")
    _require(_digest(lattice_payload) == context.lattice_sha256 == lattice.result_sha256,
             "stale_lattice_pin")
    _require(_digest(signatures_payload) == context.signatures_sha256 == signatures.result_sha256,
             "stale_signatures_pin")
    frame_ids = tuple(frame.frame_id for frame in session.frames)
    _require(lattice.session_sha256 == signatures.session_sha256 == graph.session_sha256
             == context.session_sha256 and lattice.frame_ids == signatures.frame_ids
             == graph.frame_ids == frame_ids and signatures.lattice_sha256
             == context.lattice_sha256 and lattice.graph_sha256 == context.graph_sha256,
             "context_binding_mismatch")
    _require(len(lattice.readings) == len(readings)
             and all(a is b for a, b in zip(lattice.readings, readings)), "foreign_readings")
    expected = [(i, j) for i, reading in enumerate(readings)
                for j, _alternative in enumerate(reading.alternatives)]
    _require([(row.canonical_index, row.alternative_index) for row in signatures.signatures]
             == expected and len(signatures.signatures) == len(signature_rows),
             "incomplete_upstream_signatures")
    for evidence, row in zip(signatures.signatures, signature_rows):
        _require(evidence.signature.reading is readings[evidence.canonical_index]
                 and evidence.signature_sha256 == row["sha256"], "stale_signature_pin")
    _require(all(record.session_sha256 == context.session_sha256 for record in graph.evidence),
             "context_binding_mismatch")
    _require(signature_module._signatures_digest(signatures.signatures)
             == signatures.signatures_sha256, "stale_signature_pin")
    replace(graph.budgets)
    for evidence in graph.evidence:
        for diagnostic in evidence.hypotheses:
            replace(diagnostic)
        replace(evidence)
    replace(graph)
    graph_payload = connections._graph_payload(graph)
    _require(connections._digest(graph_payload) == context.graph_sha256 == graph.graph_sha256,
             "stale_graph_pin")
    edges = tuple((row.left_id, row.right_id, row.right_yaw_class)
                  for row in graph.evidence if row.status == "solved")
    projected = graph_module._reading_present_constraint_projection(edges, frame_ids, readings)
    _require(type(projected) is tuple, "invalid_graph_relations")
    return {"pins": pins, "session": session_payload, "lattice": lattice_payload,
            "signatures": signatures_payload, "connection_graph": graph_payload,
            "projected_relations": projected,
            "omitted_frame_ids": tuple(frame for frame in frame_ids
                if frame not in {reading.frame for reading in readings})}


def _source_manifest():
    root = Path(__file__).resolve().parents[3]
    for relative, pin in SOURCE_PINS:
        _require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == pin,
                 "source_pin_mismatch")
    return (*SOURCE_PINS,
            ("adapter", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def _input_payload(readings, signatures, caps, reservation, context, shift,
                   structural_share, solver_settings, manifest):
    index, by_object, frames = plane_graph._canonical_reading_index(readings)
    reading_rows = [{"input_index": i, "canonical_identity": by_object[id(reading)],
                     "order_key": lattice_planes.reading_order_key(reading),
                     "reading": _reading_payload(reading)}
                    for i, reading in enumerate(readings)]
    input_by_object = {id(reading): i for i, reading in enumerate(readings)}
    signature_rows = [_signature_payload(signature, input_by_object[id(signature.reading)],
                                        by_object[id(signature.reading)][1])
                      for signature in signatures]
    return {"schema": SCHEMA, "domain": DOMAIN, "cap_version": CAP_VERSION,
        "caps": asdict(caps), "reservation": asdict(reservation), "sources": manifest,
        "settings": {"max_shift_cells": shift, "max_structural_share_argument": structural_share,
            "effective_structural_skip_threshold": plane_graph.MIN_STRUCTURAL_SHARE
                if structural_share is None else structural_share,
            "pair_min_structural_share": plane_graph.MIN_STRUCTURAL_SHARE,
            "min_overlap_cells": plane_graph.MIN_OVERLAP_CELLS,
            "psr_min": 3.0, "psr_scale": plane_graph.PSR_SCALE,
            "score_weights": (0.5, 0.3, 0.2, -0.5),
            "same_axis_tolerance": plane_graph.SAME_AXIS_TOLERANCE,
            "same_axis_score_tie_tolerance": 1e-12,
            "algorithm": plane_graph.PLANE_CORRELATION_ALGORITHM,
            "structural_skip_rule": "both-below-threshold",
            "same_axis_rule": "drop-lower-score-conflict-and-both-on-tie",
            "selected_b_rule": "unique-distance-for-v1",
            "pair_refusal_reasons": plane_graph.MATCH_REFUSAL_REASONS},
        "solver_downstream_context": solver_settings,
        "frames": frames, "readings": reading_rows, "signatures": signature_rows,
        "accepted_context": _context_payload(context, readings, signature_rows)}


def _attempts(payload):
    rows = payload["signatures"]
    threshold = payload["settings"]["effective_structural_skip_threshold"]
    frames = sorted({row["frame"] for row in rows})
    for i, left_frame in enumerate(frames):
        for right_frame in frames[i + 1:]:
            for left in rows:
                if left["frame"] != left_frame:
                    continue
                for right in rows:
                    if right["frame"] != right_frame:
                        continue
                    if left["structural_share"] < threshold and right["structural_share"] < threshold:
                        continue
                    for yaw in range(4):
                        yield left, right, yaw


def _candidate_identity(match):
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, match.left_b, match.right_b,
            match.quadrant_of_right, match.cell_shift_rc)


def _match_payload(match):
    # Existing witness serializer has no solver residual field. Add it explicitly
    # so this receipt binds every PlaneMatch field, including infinity PSR.
    return {**plane_graph._indexed_match_payload(match),
            "integer_residual": match.integer_residual}


def _trace_payload(row):
    result = asdict(row)
    if row.psr == float("inf"):
        result["psr"] = "positive_infinity"
    return result


def _reconcile(payload, trace, observation, returned, refusals, index):
    reservation = payload["reservation"]
    _require(type(observation) is plane_graph.SameAxisCandidateObservation
             and type(observation.candidates) is tuple
             and type(observation.dispositions) is tuple, "invalid_observation")
    candidates, dispositions = observation.candidates, observation.dispositions
    _require(type(trace) is plane_graph.PlaneGraphTrace and type(trace.pairs) is list
             and len(trace.pairs) <= payload["caps"]["attempts"], "invalid_trace")
    _require(len(candidates) == len(dispositions), "unpaired_observation")
    _require(len(candidates) <= payload["caps"]["candidate_identities"], "candidate_identities_cap")
    _require(type(returned) is tuple and all(type(m) is plane_graph.PlaneMatch for m in returned),
             "invalid_return")
    kept = tuple(match for match, disposition in zip(candidates, dispositions)
                 if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    dropped = tuple(match for match, disposition in zip(candidates, dispositions)
                    if disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
    _require(len(kept) + len(dropped) == len(candidates), "invalid_disposition")
    _require(len(kept) == len(returned) and all(a is b and a == b for a, b in zip(kept, returned)),
             "ordered_return_mismatch")
    # plane_graph marks trace selection by this key, which deliberately omits
    # alternative b. A dropped candidate can share a selected trace key with
    # a kept candidate; its exact disposition remains independently bound.
    selected_keys = {(match.left, match.right, match.left_axis, match.right_axis,
                      match.left_plane, match.right_plane,
                      match.quadrant_of_right, match.score) for match in kept}
    _require(tuple(sorted(candidates, key=plane_graph._match_order)) == candidates,
             "candidate_order_mismatch")
    identities = set()
    candidate_by_attempt = {}
    for candidate, disposition in zip(candidates, dispositions):
        plane_graph._validate_indexed_match(candidate, index, require_selected_b=True)
        identity = _candidate_identity(candidate)
        _require(identity not in identities, "duplicate_candidate_identity")
        identities.add(identity)
        key = identity[:9]
        _require(key not in candidate_by_attempt, "conflicting_candidate_identity")
        candidate_by_attempt[key] = (candidate, disposition)
        _json(_match_payload(candidate))
    a, m, k, d = reservation["attempts"], len(candidates), len(kept), len(dropped)
    signature_counts = {}
    for row in payload["signatures"]:
        signature_counts[row["frame"]] = signature_counts.get(row["frame"], 0) + 1
    frames = sorted(signature_counts)
    h = 4 * sum(signature_counts[left] * signature_counts[right]
                for i, left in enumerate(frames) for right in frames[i + 1:])
    _require(h == reservation["hypotheses"], "ledger_mismatch")
    _require(trace.pre_admission_candidates == len(trace.pairs) == a
             and trace.post_admission_candidates == m
             and trace.post_same_axis_candidates == k, "ledger_mismatch")
    _require(a <= payload["caps"]["attempts"] and m <= reservation["potential_candidates"],
             "ledger_mismatch")
    refusal_counts = {reason: 0 for reason in plane_graph.MATCH_REFUSAL_REASONS
                      if reason != plane_graph.SAME_AXIS_DISPOSITION_DROPPED}
    pair_refusals = []
    seen_admitted = set()
    attempted_count = 0
    for expected, row in zip(_attempts(payload), trace.pairs):
        attempted_count += 1
        left, right, yaw = expected
        _require(type(row) is plane_graph.PlanePairTrace, "invalid_trace")
        _require((row.left, row.right, row.left_axis, row.right_axis, row.quadrant)
                 == (left["frame"], right["frame"], left["axis"], right["axis"], yaw),
                 "attempt_identity_mismatch")
        if row.admitted:
            key = (left["frame"], right["frame"], left["plane"], right["plane"],
                   left["axis"], right["axis"], left["b"], right["b"], yaw)
            _require(key in candidate_by_attempt and key not in seen_admitted, "admitted_trace_mismatch")
            seen_admitted.add(key)
            candidate, _disposition = candidate_by_attempt[key]
            trace_key = (row.left, row.right, row.left_axis, row.right_axis,
                         row.left_plane, row.right_plane, row.quadrant, row.score)
            _require(row.reason == "" and row.left_plane == candidate.left_plane
                     and row.right_plane == candidate.right_plane
                     and row.psr == candidate.psr and row.score == candidate.score
                     and row.overlap_cells == candidate.overlap_cells
                     and row.cell_shift_rc == candidate.cell_shift_rc
                     and type(row.selected) is bool
                     and row.selected == (trace_key in selected_keys),
                     "admitted_trace_mismatch")
        else:
            _require(row.reason in refusal_counts and not row.selected,
                     "invalid_pair_refusal")
            refusal_counts[row.reason] += 1
            pair_refusals.append((row.left, row.right, row.left_axis, row.right_axis,
                                  row.quadrant, row.reason))
    _require(attempted_count == a and len(seen_admitted) == m
             and a == m + sum(refusal_counts.values()), "ledger_mismatch")
    expected_refusals = pair_refusals + [(match.left, match.right, match.left_axis,
        match.right_axis, match.quadrant_of_right, plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
        for match in dropped]
    _require(len(refusals) == len(expected_refusals), "refusal_ledger_mismatch")
    for refusal, expected in zip(refusals, expected_refusals):
        _require(type(refusal) is plane_graph.PlaneMatchRefusal
                 and type(refusal.detail) is str
                 and (refusal.left, refusal.right, refusal.left_axis, refusal.right_axis,
                      refusal.quadrant_of_right, refusal.reason) == expected,
                 "refusal_ledger_mismatch")
    ledger = Ledger(reservation["hypotheses"], reservation["structural_skips"], a, m, k, d,
                    tuple(refusal_counts.items()))
    _require(ledger.hypotheses == ledger.structural_skips + a and m == k + d,
             "ledger_mismatch")
    return ledger


def _seal(payload, reservation, trace, observation, returned, refusals, index):
    ledger = _reconcile(payload, trace, observation, returned, refusals, index)
    input_record = _json(payload)
    admitted = [_match_payload(match) for match in observation.candidates]
    kept = [_match_payload(match) for match in returned]
    evidence = {"ledger": asdict(ledger), "candidates": admitted,
                "dispositions": observation.dispositions, "returned_matches": kept,
                "refusals": [asdict(row) for row in refusals],
                "trace": {"pairs": [_trace_payload(row) for row in trace.pairs],
                    "pre_admission_candidates": trace.pre_admission_candidates,
                    "post_admission_candidates": trace.post_admission_candidates,
                    "post_same_axis_candidates": trace.post_same_axis_candidates},
                "normal_return": True, "matcher_calls": 1, "reconstruction_authorized": False}
    evidence_record = _json(evidence)
    _require(len(input_record) + len(evidence_record) <= reservation.input_evidence_bytes,
             "reservation_mismatch")
    input_pin = hashlib.sha256(input_record).hexdigest()
    admitted_pin, kept_pin = _digest(admitted), _digest(kept)
    receipt_pin = _digest([SCHEMA, DOMAIN, CAP_VERSION, input_pin,
        hashlib.sha256(evidence_record).hexdigest(), admitted_pin, kept_pin])
    return MatcherCallReceipt(SCHEMA, DOMAIN, CAP_VERSION, input_record, evidence_record,
        input_pin, admitted_pin, kept_pin, receipt_pin, ledger, reservation,
        observation.candidates, observation.dispositions, returned, tuple(refusals))


def collect_matcher_call(readings, signatures, *, max_shift_cells=32,
                         max_structural_share=None, context=None, expected_input_sha256=None,
                         integer_tolerance=0.15, min_resultant=0.90, ambiguous_margin=0.02,
                         caps=HARD_CAPS):
    """Own one unchanged matcher invocation; refuse without partial evidence.

    Caller sinks/observations are never accepted. Solver settings are recorded
    solely as downstream context; no solver evaluation occurs.
    """
    try:
        _require(lattice_planes.lattice._period_support_state() is None,
                 "active_period_support_capture")
        threshold = plane_graph.MIN_STRUCTURAL_SHARE if max_structural_share is None else max_structural_share
        solver_settings = (integer_tolerance, min_resultant, ambiguous_margin)
        _require(all(type(value) in (int, float) and math.isfinite(value)
                     for value in solver_settings), "invalid_settings")
        reservation = _preflight(readings, signatures, caps, context, max_shift_cells, threshold)
        manifest = _source_manifest()
        payload = _input_payload(readings, signatures, caps, reservation, context,
                                 max_shift_cells, max_structural_share, solver_settings, manifest)
        input_pin = _digest(payload)
        if expected_input_sha256 is not None:
            _require(plane_graph._sha256_hex(expected_input_sha256)
                     and expected_input_sha256 == input_pin, "stale_input_pin")
        index, _by_object, _frames = plane_graph._canonical_reading_index(readings)
        # All cardinality/byte/work checks have completed before these objects.
        trace = plane_graph.PlaneGraphTrace([])
        refusals, observations = [], []
        try:
            returned = plane_graph.match_plane_graph(readings, signatures,
                max_shift_cells=max_shift_cells, max_structural_share=max_structural_share,
                refusals=refusals, diagnostic_sink=trace, candidate_observation_sink=observations)
        except BaseException:
            return SeamResult("refused", "matcher_exception_or_cancellation")
        _require(len(observations) == 1, "observation_count_mismatch")
        # Detect mutation/stale input/source context across the invocation.
        after = _input_payload(readings, signatures, caps, reservation, context,
                               max_shift_cells, max_structural_share, solver_settings, _source_manifest())
        _require(_digest(after) == input_pin, "input_changed_during_call")
        receipt = _seal(payload, reservation, trace, observations[0], returned, refusals, index)
        return SeamResult("complete", receipt=receipt, matches=returned, refusals=tuple(refusals))
    except _Refusal as error:
        return SeamResult("refused", str(error))
    except (AttributeError, TypeError, ValueError, IndexError, KeyError, OverflowError,
            MemoryError, OSError):
        return SeamResult("refused", "invalid_or_incomplete_evidence")


def _validate_workspace_reservation(payload, readings, reservation, caps):
    _require(type(reservation) is Reservation
             and all(type(getattr(reservation, field.name)) is int
                     and getattr(reservation, field.name) >= 0 for field in fields(reservation)),
             "reservation_mismatch")
    _require(reservation.input_evidence_bytes <= caps.input_evidence_bytes,
             "input_evidence_bytes_cap")
    _require(reservation.correlation_workspace_bytes <= caps.correlation_workspace_bytes,
             "correlation_workspace_bytes_cap")
    shift = payload["settings"]["max_shift_cells"]
    _require(type(shift) is int and shift >= 0, "invalid_settings")
    # Array metadata only: replay never allocates or converts source arrays.
    metadata = {}
    for row in payload["signatures"]:
        arrays = row["payload"][5:8]
        shape = arrays[0]["shape"]
        _require(type(shape) is list and len(shape) == 2
                 and all(type(value) is int and value > 0 for value in shape),
                 "reservation_mismatch")
        n = _workspace_mul(*shape)
        widths = []
        for array in arrays:
            size = array["bytes"]
            _require(array["shape"] == shape and type(size) is int and size > 0
                     and size % n == 0, "reservation_mismatch")
            widths.append(size // n)
        metadata[id(row)] = (shape, max(8, widths[0], widths[2]))
    workspace = 0
    for left, right, yaw in _attempts(payload):
        lreading, rreading = (readings[row["payload"][1]] for row in (left, right))
        axis, _sign = plane_graph.axis_transform(lreading.axis, yaw)
        if (axis == rreading.axis and lreading.distance_for(left["b"]) is not None
                and rreading.distance_for(right["b"]) is not None):
            shape, width = metadata[id(right)]
            workspace = max(workspace, _pair_workspace(shape,
                max(width, metadata[id(left)][1]), shift))
    _require(workspace == reservation.correlation_workspace_bytes,
             "reservation_mismatch")


def validate_receipt(receipt):
    """Replay sealed evidence without matcher/solver calls or input payload reads."""
    try:
        _require(type(receipt) is MatcherCallReceipt and not receipt.reconstruction_authorized,
                 "invalid_receipt")
        _require((receipt.schema, receipt.domain, receipt.cap_version) == (SCHEMA, DOMAIN, CAP_VERSION),
                 "invalid_receipt")
        _require(type(receipt.input_record) is bytes and type(receipt.evidence_record) is bytes
                 and len(receipt.input_record) + len(receipt.evidence_record) <= INPUT_EVIDENCE_BYTES,
                 "invalid_receipt")
        payload, evidence = json.loads(receipt.input_record), json.loads(receipt.evidence_record)
        # Recheck dependencies before replaying with current reading/serializer
        # semantics. Canonical JSON compares decoded lists with manifest tuples
        # while retaining the exact dependency set, hashes and order.
        _require(_json(payload.get("sources")) == _json(_source_manifest()),
                 "source_manifest_mismatch")
        _require(payload["schema"] == SCHEMA and payload["domain"] == DOMAIN
                 and payload["cap_version"] == CAP_VERSION
                 and payload["reservation"] == asdict(receipt.reservation), "invalid_receipt")
        profile = SeamCaps(**payload["caps"])
        _require(all(type(getattr(profile, field.name)) is int
                     and 0 < getattr(profile, field.name) <= getattr(HARD_CAPS, field.name)
                     for field in fields(profile)), "invalid_caps")
        _require(len(receipt.candidates) <= profile.candidate_identities
                 and len(payload["readings"]) <= profile.readings
                 and len(evidence["trace"]["pairs"]) <= profile.attempts,
                 "invalid_receipt")
        _require(_json(payload) == receipt.input_record and _json(evidence) == receipt.evidence_record,
                 "noncanonical_receipt")
        _require(evidence["normal_return"] is True and evidence["matcher_calls"] == 1
                 and evidence["reconstruction_authorized"] is False, "incomplete_receipt")
        readings = tuple(lattice_planes.PlaneReading(
            **{**row["reading"], "alternatives": tuple(lattice_planes.DistanceAlternative(*alt)
                        for alt in row["reading"]["alternatives"]),
               "rectification": lattice_planes.PlaneRectification(**row["reading"]["rectification"]),
               "phase_evidence": None, "region_evidence": None}) for row in payload["readings"])
        _validate_workspace_reservation(payload, readings, receipt.reservation, profile)
        # Replay uses the pinned canonical identities directly; reading key/order
        # was validated before the owned call and all content is digest-bound.
        index = {tuple(row["canonical_identity"]): reading
                 for row, reading in zip(payload["readings"], readings)}
        trace_payload = evidence["trace"]
        trace_rows = []
        for row in trace_payload["pairs"]:
            values = dict(row)
            if values["psr"] == "positive_infinity":
                values["psr"] = float("inf")
            for name in ("raw_max_shift_rc", "supported_max_shift_rc", "cell_shift_rc"):
                if values[name] is not None:
                    values[name] = tuple(values[name])
            trace_rows.append(plane_graph.PlanePairTrace(**values))
        trace = plane_graph.PlaneGraphTrace(trace_rows, trace_payload["pre_admission_candidates"],
            trace_payload["post_admission_candidates"], trace_payload["post_same_axis_candidates"])
        observation = plane_graph.SameAxisCandidateObservation(receipt.candidates, receipt.dispositions)
        ledger = _reconcile(payload, trace, observation, receipt.returned_matches, receipt.refusals, index)
        rebuilt = _seal(payload, receipt.reservation, trace, observation,
                        receipt.returned_matches, receipt.refusals, index)
        _require(ledger == receipt.ledger and rebuilt == receipt, "receipt_digest_mismatch")
        return SeamResult("complete", receipt=receipt, matches=receipt.returned_matches,
                          refusals=receipt.refusals)
    except _Refusal as error:
        return SeamResult("refused", str(error))
    except (AttributeError, TypeError, ValueError, IndexError, KeyError, OverflowError,
            MemoryError, OSError):
        return SeamResult("refused", "invalid_or_incomplete_evidence")
