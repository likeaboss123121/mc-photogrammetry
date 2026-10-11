"""One bounded, unmocked ordinary-image production probe over generated PNGs.

Scene/camera parameters belong to rendering only. Reconstruction receives paths;
there is no renderer truth, calibrated camera or expected coordinate adapter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from mc_photogrammetry.reconstruction.camera import (
    CameraModel, camera_eye, camera_rotation_matrix, intrinsics_for,
)

WIDTH, HEIGHT = 720, 480
SEED = 20261009
TEXTURE_PROFILE = "orthogonal-long-seams-v2"
BOX = ((0.0, 6.0), (0.0, 4.0), (0.0, 6.0))
VIEWS = (
    ((-5.0, 4.0, -6.0), -40.0, 25.0),
    ((-4.0, 3.8, -6.5), -35.0, 24.0),
    ((-5.5, 4.2, -5.0), -45.0, 27.0),
)


def canonical_digest(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def trace_corner(eye: np.ndarray, directions: np.ndarray):
    """Nearest positive finite face intersection; back faces cannot overwrite it."""
    shape = directions.shape[:-1]
    distance = np.full(shape, np.inf)
    points = np.zeros((*shape, 3), dtype=np.float64)
    faces = np.full(shape, -1, dtype=np.int8)
    for axis in range(3):
        for side in range(2):
            with np.errstate(divide="ignore", invalid="ignore"):
                candidate = (BOX[axis][side] - eye[axis]) / directions[..., axis]
            usable = np.isfinite(candidate) & (candidate > 1e-6) & (candidate < distance)
            hit = eye + np.where(usable, candidate, 0.0)[..., None] * directions
            for other in range(3):
                if other != axis:
                    usable &= ((hit[..., other] >= BOX[other][0] - 1e-9)
                               & (hit[..., other] <= BOX[other][1] + 1e-9))
            distance[usable] = candidate[usable]
            points[usable] = hit[usable]
            faces[usable] = 2 * axis + side
    return points, faces


def _texture(face: int, first: int, second: int) -> np.ndarray:
    """Stable features and long parallel seam families on orthogonal faces.

    Four texture stripes per cell are appearance, not asserted unit boundaries.
    Their phase is continuous across neighbouring cells. Different face normals
    expose the three axis directions without fragmenting every line at a grid
    crossing. Block period and occupancy remain reconstruction's responsibility.
    """
    rng = np.random.default_rng(np.random.SeedSequence([SEED, face, first, second]))
    texture = np.full((32, 32, 3), 155, dtype=np.int16)
    texture += rng.integers(-7, 8, size=(32, 32, 1), dtype=np.int16)
    for _ in range(5):
        row, column = rng.integers(5, 24, size=2)
        texture[row:row + 4, column:column + 4] += int(rng.choice([-30, 25]))
    # Small per-cell colour shifts identify correspondence without replacing seams.
    texture[..., 0] += (first * 7 + second * 3 + face) % 17 - 8
    texture[..., 2] += (first * 3 + second * 11 + face) % 13 - 6
    for start in (0, 8, 16, 24):
        if face // 2 == 1:
            texture[start:start + 2, :] = 28
        else:
            texture[:, start:start + 2] = 28
    return np.clip(texture, 0, 255).astype(np.uint8)


def render_view(index: int) -> np.ndarray:
    if type(index) is not int or not 0 <= index < len(VIEWS):
        raise ValueError("view outside frozen fixture")
    position, yaw, pitch = VIEWS[index]
    camera = CameraModel(position=position, yaw_degrees=yaw,
                         pitch_degrees=pitch, fov_degrees=70.0)
    intrinsics = intrinsics_for(camera, WIDTH, HEIGHT)
    fx, fy = intrinsics.focal_length_pixels
    cx, cy = intrinsics.principal_point_pixels
    xs, ys = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    directions = np.stack(((xs - cx) / fx, -(ys - cy) / fy,
                           np.ones_like(xs, dtype=np.float64)), axis=-1)
    points, faces = trace_corner(camera_eye(camera),
                                directions @ camera_rotation_matrix(camera))
    rgb = np.full((HEIGHT, WIDTH, 3), (70, 85, 100), dtype=np.uint8)
    for face in range(6):
        mask = faces == face
        if not mask.any():
            continue
        tangents = [axis for axis in range(3) if axis != face // 2]
        u, v = points[..., tangents[0]], points[..., tangents[1]]
        first, second = np.floor(u).astype(int), np.floor(v).astype(int)
        tu = np.minimum((np.mod(u, 1.0) * 32).astype(int), 31)
        tv = np.minimum((np.mod(v, 1.0) * 32).astype(int), 31)
        cells = np.unique(np.stack((first[mask], second[mask]), axis=-1), axis=0)
        for a, b in cells:
            selected = mask & (first == a) & (second == b)
            rgb[selected] = _texture(face, int(a), int(b))[tv[selected], tu[selected]]
    return rgb


def write_fixture(destination: Path) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    for index in range(len(VIEWS)):
        path = destination / f"view-{index:02d}.png"
        Image.fromarray(render_view(index)).save(path, format="PNG")
        paths.append(path)
    return tuple(paths)


def detector_counts(paths: tuple[Path, ...]) -> dict:
    """Separate generated-only detector diagnostic, never a production call.

    Uses the exact current enhancement/detection defaults. Arrays and endpoints
    stay local; only scalar counts/length ranges and encoded provenance leave.
    """
    import cv2
    from mc_photogrammetry.reconstruction import calibration

    if len(paths) != len(VIEWS):
        raise ValueError("diagnostic requires the frozen three generated views")
    rows = []
    for index, path in enumerate(paths):
        encoded = path.read_bytes()
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != (HEIGHT, WIDTH, 3):
            raise ValueError("generated view has foreign dimensions")
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        segments = calibration.detect_segments(calibration.enhance(grey))
        lengths = np.hypot(segments[:, 2] - segments[:, 0],
                           segments[:, 3] - segments[:, 1])
        rows.append({"ordinal": index, "label": path.name,
                     "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                     "byte_length": len(encoded), "retained_segments": len(segments),
                     "meets_camera_gate": len(segments) >= 12,
                     "minimum_retained_length": float(lengths.min()) if len(lengths) else None,
                     "maximum_retained_length": float(lengths.max()) if len(lengths) else None})
    record = {"schema": "synthetic-corner-detector-counts-v1",
              "texture_profile": TEXTURE_PROFILE, "seed": SEED,
              "production_invocations": 0, "row_arrays_persisted": False,
              "settings": {"scales": list(calibration.DETECTION_SCALES),
                           "minimum_length": calibration.MINIMUM_LENGTH,
                           "camera_gate": 12}, "cv2_version": cv2.__version__, "rows": rows}
    record["report_sha256"] = canonical_digest(record)
    return record


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _stage(value) -> dict | None:
    if value is None:
        return None
    return {"status": str(value.status), "reason": str(value.reason),
            "detail": str(getattr(value, "detail", ""))[:512]}


# Observer-only limits; no production certificate validation or search replay.
_SELECTION_COUNTERS = frozenset("""
endpoint_variables frames alternatives_per_endpoint input_match_rows registry_factors
quadrant_assignments branch_nodes relation_visits solution_rows scalar_payload_bytes scalar_row_bytes
planned_assignments visited_assignments solutions_found active_factor_visits eligible_assignments
excluded_assignments eligibility_hashed_assignments retained_exclusions omitted_exclusions
physical_assignments physical_feasible physical_rejected physical_relation_visits
physical_cycle_relaxations physical_hashed_assignments retained_physical_exclusions omitted_physical_exclusions
partial_checks partial_prunes retained_partial_exclusions omitted_partial_exclusions
search_events_attempted search_events_committed search_events_uncommitted
preflight_visits incidence_visits partial_support_visits partial_update_visits partial_queue_visits
partial_journal_visits partial_undo_visits partial_event_visits source_bytes relation_bytes
 disposition_bytes exclusion_bytes support_bytes scalar_records_attempted scalar_records_committed
scalar_records_uncommitted eligibility_unclassified_assignments unvisited_assignments
""".split())
_ELIGIBILITY_CODES = frozenset(("no_active_factors", "active_axis_mismatch",
    "unsupported_reading_endpoint", "axis_0_disconnected", "axis_1_disconnected", "axis_2_disconnected"))
_PHYSICAL_CODES = frozenset(("physical_displacement_diameter", "physical_frame_cycle",
                           "unverified_physical_displacement"))


def _count(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("invalid observer count")
    return value


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid observer digest")
    return value


def _reason_code(value):
    if (type(value) is not str or not 1 <= len(value) <= 96
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value)):
        raise ValueError("invalid observer reason")
    return value


def _unique_keys(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate observer key")
        result[key] = value
    return result


def _reason_counts(value, allowed):
    if type(value) is not dict or not value.keys() <= allowed:
        raise ValueError("invalid observer histogram")
    return {key: _count(count) for key, count in value.items()}


def _endpoint_selection_record(graph):
    """Project the carried sealed proof; never construct/replay a certificate."""
    if graph is None:
        return {"available": False, "incomplete": False, "reason": "stage_not_reached"}
    try:
        cert = graph.selection
        raw = cert.proof_json
        # Cap before decoding/copying arbitrary proof payload; no raw proof escapes.
        if type(raw) is not str or len(raw) > 4 * 1024 * 1024 or not raw.isascii():
            raise ValueError("observer proof exceeds bound")
        proof = json.loads(raw, object_pairs_hook=_unique_keys)
        if type(proof) is not dict or len(proof) > 32:
            raise ValueError("invalid observer proof")
        status = cert.status
        if status not in ("no", "unique", "multiple", "unresolved_evidence", "budget_refused") or proof["status"] != status:
            raise ValueError("observer status differs")
        frames = cert.frame_ids
        if (type(frames) is not tuple or not 1 <= len(frames) <= 32
                or len(set(frames)) != len(frames) or frames != graph.frame_ids
                or proof["session_frames"] != list(frames)
                or any(type(f) is not str or len(f) != 10 or not f.startswith("image-")
                       or not f[6:].isascii() or not f[6:].isdigit() or int(f[6:]) >= 32 for f in frames)):
            raise ValueError("observer frame binding differs")
        sources = {}
        for key in ("session_sha256", "lattice_sha256", "signatures_sha256", "connection_sha256"):
            value = getattr(cert, key)
            if value != getattr(graph, key):
                raise ValueError("observer source binding differs")
            sources[key] = "" if key == "connection_sha256" and value == "" else _digest(value)
        counters = proof["counters"]
        if type(counters) is not dict or counters.keys() != _SELECTION_COUNTERS:
            raise ValueError("observer counter schema differs")
        counters = {key: (None if key in ("planned_assignments", "unvisited_assignments") and value is None
                          else _count(value)) for key, value in counters.items()}
        eligibility = _reason_counts(proof["eligibility_reasons"], _ELIGIBILITY_CODES)
        physical = _reason_counts(proof["physical_reasons"], _PHYSICAL_CODES)
        c = counters
        if (c["eligible_assignments"] + c["excluded_assignments"] != c["eligibility_hashed_assignments"]
                or c["eligibility_hashed_assignments"] + c["eligibility_unclassified_assignments"] != c["visited_assignments"]
                or c["retained_exclusions"] + c["omitted_exclusions"] != c["excluded_assignments"]
                or sum(eligibility.values()) != c["excluded_assignments"]
                or c["physical_feasible"] + c["physical_rejected"] != c["physical_hashed_assignments"]
                or not c["physical_hashed_assignments"] <= c["physical_assignments"] <= c["branch_nodes"]
                or c["retained_physical_exclusions"] + c["omitted_physical_exclusions"] != c["physical_rejected"]
                or sum(physical.values()) != c["physical_rejected"]
                or (c["planned_assignments"] is None) != (c["unvisited_assignments"] is None)
                or c["planned_assignments"] is not None and c["visited_assignments"] + c["unvisited_assignments"] != c["planned_assignments"]):
            raise ValueError("contradictory observer counters")
        exhaustive = proof["exhaustive"]
        if type(exhaustive) is not bool or exhaustive and (c["unvisited_assignments"] != 0 or c["eligibility_unclassified_assignments"] != 0):
            raise ValueError("contradictory observer completion")
        result = {"available": True, "incomplete": False, "status": status,
                  "reason": _reason_code(cert.reason), "exhaustive": exhaustive,
                  "frame_ids": list(frames), "sources": sources,
                  "graph_sha256": _digest(graph.result_sha256),
                  "selection_sha256": _digest(cert.result_sha256),
                  "proof_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
                  "candidate_count": _count(cert.candidate_count),
                  "counters": counters, "eligibility_reasons": eligibility,
                  "physical_reasons": physical}
        if result["candidate_count"] > 16384:
            raise ValueError("observer candidate count exceeds bound")
        result["projection_sha256"] = canonical_digest(result)
        return result
    except Exception as error:
        return {"available": False, "incomplete": True, "reason": "invalid_selection_projection",
                "error_type": type(error).__name__}


def _pair_spatial_record(pair):
    """Carry existing counts only; per-hypothesis bins are not pair aggregates."""
    try:
        direct = {key: (None if getattr(pair, key, None) is None else _count(getattr(pair, key)))
                  for key in ("spatial_bins_left", "spatial_bins_right", "spatial_bins_total")}
        hypotheses = getattr(pair, "hypotheses", None)
        rows = None
        if hypotheses is not None:
            if type(hypotheses) is not tuple or len(hypotheses) > 64:
                raise ValueError("observer hypothesis count exceeds bound")
            rows = []
            for ordinal, row in enumerate(hypotheses):
                yaw = row.yaw_class
                if type(yaw) is not int or yaw not in range(4):
                    raise ValueError("invalid observer yaw class")
                rows.append({"ordinal": ordinal, "yaw_class": yaw, "spatial_bins": _count(row.spatial_bins)})
        return {"incomplete": False, "pair_aggregate_available": any(v is not None for v in direct.values()),
                **direct, "hypothesis_scope": "per_yaw_hypothesis", "hypotheses": rows}
    except Exception as error:
        return {"incomplete": True, "pair_aggregate_available": False,
                "reason": "invalid_pair_spatial_projection", "error_type": type(error).__name__}


_SEED_COUNTS = ("node_count", "side_match_count", "normal_score_pass_count",
    "margin_pass_count", "tangent_support_pass_count", "after_side_count",
    "after_score_count", "after_margin_count", "after_support_count", "eligible_seed_count")
_SEED_MAXIMA = ("max_side_normal_score", "max_side_margin", "max_side_tangent_support")
_LATTICE_DISPOSITIONS = {
    "seed": frozenset(("available", "absent", "not_reached")),
    "proposal": frozenset(("mask_empty", "area_rejected", "polygon_rejected",
                           "deduplicated", "returned", "diversity_cap")),
    "reading": frozenset(("proposal_validation", "frame_rejection", "reading_stage_failed",
        "window_incidence", "window_invalid", "periodicity", "structural_energy",
        "distance_alternative", "coverage", "per_axis_side_cap", "accepted", "reading_cap")),
}


def _new_lattice_observer():
    from mc_photogrammetry.reconstruction import region_growth as region

    class ScalarJoins(region._ScalarAttritionObserver):
        def __init__(self):
            super().__init__(max_items=16384, max_bytes=4 * 1024 * 1024)
            self.scalar_rows = {name: {} for name in ("seed", "proposal", "reading", "frame")}
            self.scalar_incomplete = False
            self.lattice_calls = 0

        def finish(self, stage, identity, disposition, **fields):
            # Preserve the original observer call/return and inference behavior.
            result = super().finish(stage, identity, disposition, **fields)
            capture = stage in ("seed", "proposal", "reading") or (
                stage == "frame" and type(identity) is tuple and len(identity) == 2 and identity[1] == "proposal")
            if not capture:
                return result
            try:
                if type(identity) is not tuple or not identity or type(identity[0]) is not str:
                    raise ValueError("invalid scalar identity")
                if sum(len(rows) for rows in self.scalar_rows.values()) >= 4096:
                    raise ValueError("scalar join cap")
                if identity in self.scalar_rows[stage]:
                    raise ValueError("duplicate scalar terminal")
                code = _reason_code(disposition)
                if stage != "frame" and code not in _LATTICE_DISPOSITIONS[stage]:
                    raise ValueError("foreign scalar disposition")
                names = (_SEED_COUNTS + _SEED_MAXIMA + ("axis", "side") if stage == "seed" else
                         ("axis", "side", "plane_index") if stage == "reading" else
                         ("axis", "side") if stage == "proposal" else ("returned_count",))
                row = {"disposition": code}
                for name in names:
                    value = fields.get(name)
                    if value is not None and (type(value) not in (int, float)
                            or not math.isfinite(value) or abs(value) > 2**31 - 1):
                        raise ValueError("invalid scalar field")
                    row[name] = value
                self.scalar_rows[stage][identity] = row
            except Exception:
                self.scalar_incomplete = True
            return result

    return ScalarJoins()


@contextmanager
def observe_lattice_pass(module):
    """Observe only the original lattice delegate, restoring it on every exit."""
    from mc_photogrammetry.reconstruction import region_growth as region

    observer = _new_lattice_observer()
    original = module.measure_image_derived_lattice

    def forwarding(*args, **kwargs):
        observer.lattice_calls += 1
        with region._scalar_attrition_context(observer):
            return original(*args, **kwargs)

    module.measure_image_derived_lattice = forwarding
    try:
        yield observer
    finally:
        if module.measure_image_derived_lattice is not forwarding:
            observer.scalar_incomplete = True
        module.measure_image_derived_lattice = original


def _lattice_frame_ids(active):
    if (type(active) is not tuple or not 1 <= len(active) <= len(VIEWS)
            or len(set(active)) != len(active)
            or any(type(f) is not str or len(f) != 10 or not f.startswith("image-")
                or not f[6:].isascii() or not f[6:].isdigit() or int(f[6:]) >= 32 for f in active)):
        raise ValueError("invalid lattice frame binding")
    return active


def _lattice_pass_record(lattice, observer, active):
    """Join existing scalar terminals/handoffs and carried readings, no replay."""
    if lattice is None and (observer is None or observer.lattice_calls == 0):
        return {"status": "not_reached", "incomplete": False, "groups": None}
    if observer is None:
        return {"status": "observer_not_enabled", "incomplete": False, "groups": None}
    try:
        active = _lattice_frame_ids(active)
        if lattice is None or lattice.frame_ids != active:
            raise ValueError("missing lattice delivery")
        readings = lattice.readings
        if type(readings) is not tuple or len(readings) > 6 * len(active):
            raise ValueError("reading cap")
        keys = [(f, axis, side) for f in active for axis in range(3) for side in (-1, 1)]
        groups = {key: {"frame_id": key[0], "normal_axis": key[1], "side": key[2], "proposal_side": key[2],
            "accepted_readings_at_side": 0, "seed": None, "proposals_expected": 0,
            "proposals_observed": 0, "proposal_dispositions": {}, "proposal_returned": 0,
            "handoffs": 0, "reader_expected": 0, "reader_observed": 0,
            "reader_dispositions": {}, "reader_side_transitions": {}} for key in keys}
        live = {}
        for reading in readings:
            key = (reading.frame, reading.axis, reading.side)
            if type(reading.axis) is not int or type(reading.side) not in (int, float) or key not in groups:
                raise ValueError("foreign reading join")
            if id(reading) in live:
                raise ValueError("duplicate live reading")
            live[id(reading)] = key
            groups[key]["accepted_readings_at_side"] += 1
        normal_counts = [{"frame_id": f, "normal_counts": [sum(groups[f, a, side]["accepted_readings_at_side"]
            for side in (-1, 1)) for a in range(3)]} for f in active]
        if observer.lattice_calls != 1 or observer.scalar_incomplete:
            raise ValueError("incomplete scalar delivery")
        record = observer.record()
        if record["status"] != "complete" or record["omitted_accounting_events"] or record["omitted_proposal_handoffs"]:
            raise ValueError("incomplete original observer")
        relevant = {name: observer.stages.get(name, {"expected": {}, "terminal": {}})
                    for name in ("seed", "proposal", "reading")}
        for name, stage in relevant.items():
            expected = set(stage["expected"])
            rows = observer.scalar_rows[name]
            if expected != set(stage["terminal"]) or expected != {observer._encode(i).decode() for i in rows}:
                raise ValueError("incomplete scalar stage")
        seed_rows = observer.scalar_rows["seed"]
        if set(seed_rows) != set(keys):
            raise ValueError("missing seed cells")
        for key, row in seed_rows.items():
            if row["axis"] != key[1] or row["side"] != key[2]:
                raise ValueError("foreign seed cell")
            seed = dict(row)
            if row["disposition"] == "not_reached":
                seed["first_failure_counts"] = None
            else:
                for field in _SEED_COUNTS:
                    _count(row[field])
                n, side, score, margin, support = (row[field] for field in (
                    "node_count", "after_side_count", "after_score_count", "after_margin_count", "after_support_count"))
                if (not n >= side >= score >= margin >= support >= 0
                        or any(row[field] > n for field in _SEED_COUNTS[1:])
                        or side != row["side_match_count"] or support != row["eligible_seed_count"]):
                    raise ValueError("contradictory seed predicates")
                if (row["disposition"] == "available") != (support > 0):
                    raise ValueError("contradictory seed disposition")
                seed["first_failure_counts"] = {"side": n - side, "normal_score": side - score,
                                                "margin": score - margin, "tangent_support": margin - support}
            groups[key]["seed"] = seed
        proposal_rows = observer.scalar_rows["proposal"]
        returned = {}
        for identity, row in proposal_rows.items():
            if len(identity) != 2 or type(identity[1]) is not int or identity[1] < 0:
                raise ValueError("foreign proposal identity")
            axis, side = row["axis"], row["side"]
            if type(axis) is not int or type(side) is not int or (identity[0], axis, side) not in groups:
                raise ValueError("foreign proposal cell")
            cell = groups[identity[0], axis, side]
            cell["proposals_expected"] += 1
            cell["proposals_observed"] += 1
            code = row["disposition"]
            cell["proposal_dispositions"][code] = cell["proposal_dispositions"].get(code, 0) + 1
            if code == "returned":
                cell["proposal_returned"] += 1
                returned[identity] = (identity[0], axis, side)
        handoffs = {}
        for key, value in observer.bindings.items():
            if key[0] != "proposal_return":
                continue
            generated = (key[1], key[2])
            if generated not in returned or type(value) is not tuple or len(value) != 2 or value[0] != generated[0] or type(value[1]) is not int or value[1] < 0 or value in handoffs:
                raise ValueError("foreign or duplicate handoff")
            cell_key = returned[generated]
            handoffs[value] = cell_key
            groups[cell_key]["handoffs"] += 1
        reader_rows = observer.scalar_rows["reading"]
        if len(handoffs) != len(returned) or set(handoffs) != set(reader_rows):
            raise ValueError("incomplete proposal reader handoff")
        accepted = {}
        for identity, row in reader_rows.items():
            cell_key = handoffs[identity]
            cell = groups[cell_key]
            if row["axis"] is not None and (type(row["axis"]) is not int or row["axis"] != cell_key[1]):
                raise ValueError("reader normal changed")
            side = row["side"]
            if side is not None and (type(side) not in (int, float) or side not in (-1, 1)):
                raise ValueError("invalid reader side")
            code = row["disposition"]
            cell["reader_expected"] += 1
            cell["reader_observed"] += 1
            cell["reader_dispositions"][code] = cell["reader_dispositions"].get(code, 0) + 1
            label = "unknown" if side is None else str(int(side))
            cell["reader_side_transitions"][label] = cell["reader_side_transitions"].get(label, 0) + 1
            if code == "accepted":
                plane_index = row["plane_index"]
                if type(plane_index) is not int or plane_index < 0 or side is None or (identity[0], plane_index) in accepted:
                    raise ValueError("invalid accepted reading join")
                accepted[identity[0], plane_index] = (identity[0], cell_key[1], side)
        bound_live = {}
        for identity, cell_key in live.items():
            binding = observer.bindings.get(("reading", identity))
            if type(binding) is not tuple or len(binding) != 2 or binding in bound_live:
                raise ValueError("missing live reading binding")
            bound_live[binding] = cell_key
        if accepted != bound_live:
            raise ValueError("accepted reading conservation failed")
        if set(observer.scalar_rows["frame"]) != {(frame, "proposal") for frame in active}:
            raise ValueError("foreign frame observation")
        frames = []
        for frame in active:
            row = observer.scalar_rows["frame"].get((frame, "proposal"))
            if row is None:
                raise ValueError("missing frame return count")
            count = _count(row["returned_count"])
            returned_count = sum(cell["proposal_returned"] for key, cell in groups.items() if key[0] == frame)
            if count > 32 or count != returned_count or set(i[1] for i in handoffs if i[0] == frame) != set(range(count)):
                raise ValueError("frame return count differs")
            frames.append({**next(r for r in normal_counts if r["frame_id"] == frame),
                           "proposal_outcome": row["disposition"], "proposal_returned": count})
        allowed_checks = {prefix + ":" + frame for frame in active for prefix in
                          ("proposal_generation", "proposal_returned", "proposal_to_reading")}
        checks = {}
        for key, value in record["checks"].items():
            if key == "local_period_child_accounting":
                continue
            if type(key) is not str or key not in allowed_checks or type(value) is not bool or not value:
                raise ValueError("foreign or failed original check")
            checks[key] = value
        for frame in active:
            if any(i[0] == frame for i in proposal_rows):
                if any(checks.get(prefix + ":" + frame) is not True for prefix in ("proposal_generation", "proposal_returned")):
                    raise ValueError("missing proposal checks")
            if any(i[0] == frame for i in handoffs) and checks.get("proposal_to_reading:" + frame) is not True:
                raise ValueError("missing reader check")
        result = {"schema": "synthetic-lattice-scalar-joins-v1", "status": "complete", "incomplete": False,
            "lattice_calls": 1, "lattice_sha256": _digest(lattice.result_sha256),
            "frames": frames, "groups": list(groups.values()),
            "totals": {"seed_expected": len(keys), "seed_observed": len(seed_rows),
                "proposal_expected": len(proposal_rows), "proposal_observed": len(proposal_rows),
                "proposal_returned": len(returned), "handoffs": len(handoffs),
                "reader_expected": len(reader_rows), "reader_observed": len(reader_rows),
                "accepted_readings": len(readings)},
            "original_stage_digests": {name: _digest(stage["digest"].hexdigest()) for name, stage in observer.stages.items()
                                       if name in relevant},
            "checks": checks,
            "limits": {"scalar_rows": 4096, "accounting_items": observer.max_items,
                       "accounting_bytes": observer.max_bytes}}
        result["projection_sha256"] = canonical_digest(result)
        return result
    except Exception as error:
        return {"status": "incomplete", "incomplete": True, "groups": None,
                "reason": "lattice_scalar_projection_failed", "error_type": type(error).__name__}


def _calibration_rows(lattice, active_ids: tuple[str, ...]) -> tuple[bool, list[dict]]:
    """Join ordered calibration values to their actual active-frame identities."""
    from mc_photogrammetry.reconstruction.calibration import CalibrationResult

    frame_ids = lattice.frame_ids
    calibrations = lattice.camera_calibrations
    if (type(frame_ids) is not tuple or frame_ids != active_ids
            or len(frame_ids) > len(VIEWS) or len(set(frame_ids)) != len(frame_ids)
            or type(calibrations) is not tuple):
        raise ValueError("invalid calibration frame binding")
    if not calibrations:
        return False, []
    if len(calibrations) != len(frame_ids):
        raise ValueError("incomplete calibration frame binding")
    rows = []
    for frame_id, value in zip(frame_ids, calibrations):
        if not isinstance(value, CalibrationResult):
            raise ValueError("foreign calibration record")
        counts = value.per_axis_inliers
        if (type(value.segments) is not int or type(value.inliers) is not int
                or not 0 <= value.inliers <= value.segments
                or type(counts) is not tuple or len(counts) != 3
                or any(type(count) is not int or count < 0 for count in counts)
                or sum(counts) != value.inliers):
            raise ValueError("contradictory calibration counters")
        rows.append({"frame_id": frame_id, "segments": value.segments,
                     "inliers": value.inliers, "axis_inliers": list(counts)})
    return True, rows


def summarize(outcome, project) -> dict:
    """Project carried facts only; no extra camera/matcher/lattice pass."""
    entry = getattr(outcome, "entry", None)
    record = project.image_derived_production
    result = {
        "stage": record["stage"], "status": record["status"],
        "reason": record["reason"], "detail": record["detail"],
        "entry_sha256": record["entry_sha256"],
        "session_sha256": record["session_sha256"],
        "lattice_sha256": record["lattice_sha256"],
        "geometry_sha256": record["geometry_sha256"],
        "output_sha256": record["output_sha256"],
        "production_result_sha256": getattr(outcome, "result_sha256", None),
        "block_count": len(project.volume.blocks),
        "observation_kinds": sorted({cell.kind.value for _, cell in project.volume.items()}),
        "litematic_created": False,
        "source_labels": list(project.source_paths),
        "camera_admissions": [], "cameras": [], "pairs": [],
        "connections": None, "pose_graph": None, "lattice": None,
        "geometry_prefix": [], "output": None,
        "endpoint_selection": _endpoint_selection_record(None),
    }
    if entry is None:
        # Typed refusal may retain only its persisted stage envelope, not live inputs.
        result["live_accounting_available"] = False
        return result
    result["live_accounting_available"] = True
    session = entry.session
    result["input_frame_ids"] = [frame.frame_id for frame in session.input_frames]
    result["active_frame_ids"] = [frame.frame_id for frame in session.frames]
    result["scale_known"] = session.scale_known
    result["camera_admissions"] = ([{"frame_id": row.frame_id, "status": row.status,
                                      "stage": row.stage, "reason": row.reason}
                                     for row in session.admissions]
                                    if session.version == 2 else
                                    [{"frame_id": frame.frame_id, "status": "admitted",
                                      "stage": "camera_recovery", "reason": ""}
                                     for frame in session.frames])
    for frame_id, camera in session.cameras.items():
        result["cameras"].append({
            "frame_id": frame_id, "focal_pixels": list(camera.intrinsics.focal_length_pixels),
            "yaw_degrees": _number(camera.pose.yaw_degrees),
            "pitch_degrees": _number(camera.pose.pitch_degrees),
            "roll_degrees": _number(camera.pose.roll_degrees),
            "position_resolved": camera.pose.position is not None,
            "residual_pixels": _number(camera.residual_pixels),
        })
    result["connections"] = _stage(entry.graph)
    result["pose_graph"] = _stage(entry.pose_graph)
    if entry.graph is not None:
        for row in entry.graph.evidence:
            result["pairs"].append({
                "left_id": row.left_id, "right_id": row.right_id,
                "status": row.status, "reason": row.reason,
                "match_count": row.match_count, "support": row.support,
                "raw_keypoints": list(row.raw_keypoints),
                "spatial_bin_accounting": _pair_spatial_record(row),
                "parallax_degrees": _number(row.parallax_degrees),
                "median_selected_residual_degrees": _number(row.median_selected_residual_degrees),
            })
    lattice = outcome.lattice
    if lattice is not None:
        available, calibrations = _calibration_rows(lattice, tuple(result["active_frame_ids"]))
        result["lattice"] = {
            **_stage(lattice), "terminal": lattice.terminal,
            "reading_count": len(lattice.readings),
            "reading_axes": [int(row.axis) for row in lattice.readings],
            "calibration_accounting_available": available,
            "calibrations": calibrations,
            "abstentions": [{"frame_id": row.frame, "reason": row.reason,
                              "detail": row.detail[:512]} for row in lattice.abstentions],
        }
    geometry = outcome.geometry
    if geometry is not None:
        result["geometry_prefix"] = [{"name": name, **_stage(value)}
                                     for name, value in geometry.records]
        graphs = [value for name, value in geometry.records if name == "g"]
        if len(graphs) > 1:
            raise ValueError("duplicate carried graph stage")
        result["endpoint_selection"] = _endpoint_selection_record(graphs[0] if graphs else None)
    result["output"] = _stage(outcome.output)
    return result


def summarize_safely(outcome, project, *, lattice_observer=None) -> dict:
    """Preserve persisted production terminal even when diagnostics cannot project.

    Reporting failure has a distinct disposition; it never changes the original
    production status or promotes/reconstructs a cell. Only exception class is
    recorded, so traceback text and diagnostic payloads cannot escape.
    """
    try:
        result = summarize(outcome, project)
    except Exception as error:
        record = project.image_derived_production
        result = {key: record[key] for key in (
            "stage", "status", "reason", "entry_sha256", "session_sha256",
            "lattice_sha256", "geometry_sha256", "output_sha256")}
        result.update({
            "detail": record.get("detail", "")[:512],
            "detail_truncated": len(record.get("detail", "")) > 512,
            "block_count": len(project.volume.blocks),
            "live_accounting_available": False, "litematic_created": False,
            "diagnostic_status": "incomplete",
            "diagnostic_failure": {"reason": "diagnostic_projection_failed",
                                   "error_type": type(error).__name__},
        })
        return result
    result["lattice_attrition"] = _lattice_pass_record(
        getattr(outcome, "lattice", None), lattice_observer, tuple(result.get("active_frame_ids", ())))
    incomplete = result["lattice_attrition"]["incomplete"] or result["endpoint_selection"]["incomplete"] or any(
        row["spatial_bin_accounting"]["incomplete"] for row in result["pairs"])
    result["diagnostic_status"] = "incomplete" if incomplete else "complete"
    result["diagnostic_failure"] = ({"reason": "observer_projection_incomplete"} if incomplete else None)
    return result


def run_probe(destination: Path) -> dict:
    """One original production invocation with path-only inputs and fixed defaults."""
    from mc_photogrammetry.pipeline.image_derived_production import derive_image_output_from_paths
    from mc_photogrammetry.project.store import load_project

    destination.mkdir(parents=True, exist_ok=False)
    paths = write_fixture(destination / "inputs")
    project_path = destination / "corner.mcscan"
    ingress = [{"ordinal": index, "label": path.name,
                "byte_length": path.stat().st_size,
                "encoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for index, path in enumerate(paths)]
    started = time.monotonic()
    try:
        outcome = derive_image_output_from_paths(paths, project_path=project_path)
    except Exception as error:
        # Preserve a failed invocation honestly without leaking arrays or traceback text.
        terminal = {"stage": None, "status": "exception",
                    "reason": type(error).__name__, "block_count": None,
                    "live_accounting_available": False, "litematic_created": False}
    else:
        terminal = (summarize_safely(outcome, load_project(project_path))
                    if project_path.exists() else
                    {"stage": None, "status": "unpersisted_refusal",
                     "reason": getattr(outcome, "reason", "unknown"),
                     "block_count": None, "live_accounting_available": False,
                     "litematic_created": False})
    record = {
        "schema": "synthetic-image-production-corner-probe-v1",
        "fixture": {"seed": SEED, "width": WIDTH, "height": HEIGHT,
                    "view_count": len(paths), "generated_only": True,
                    "texture_profile": TEXTURE_PROFILE},
        "inputs": ingress, "production_invocations": 1,
        "production_defaults_unchanged": True,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "terminal": terminal,
        "artifacts": {"project": "corner.mcscan" if project_path.exists() else None},
        "claims": {"accuracy": False, "world_coordinates": False,
                   "hidden_blocks": False, "complete_build": False},
    }
    record["report_sha256"] = canonical_digest(record)
    with (destination / "report.json").open("x", encoding="ascii", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_probe(args.output)
    print(json.dumps({"report": str(args.output / "report.json"),
                      "terminal": report["terminal"],
                      "report_sha256": report["report_sha256"]}, sort_keys=True))
    terminal = report["terminal"]
    return 1 if (terminal["status"] in ("exception", "unpersisted_refusal")
                 or terminal.get("diagnostic_status") == "incomplete") else 0


if __name__ == "__main__":
    raise SystemExit(main())
