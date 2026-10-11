"""Single-pass kept/full development diagnostic; never promotes the kept arm."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

SCHEMA = "mc-photogrammetry-graph10-paired-universe-diagnostic-v1"
COHORT = "previously used train/development data; not fresh or held out"
SETTINGS = (32, 0.15, 0.90, 0.02)
MAX_IMAGES = 32
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_REPORT_ITEMS = 1_048_576
MAX_CSP_CALLS = 32
MAX_SOURCE_LABEL_CHARS = 4096
MAX_UPSTREAM_TEXT_CHARS = 1024
MAX_DIAGNOSTIC_ROWS = 512
MAX_DIAGNOSTIC_COUNT = 1_048_576
MAX_LINE_PASS_ROWS = 4096
BASE_DETECTOR_SCALES = (1.0, 0.5)
CANDIDATE_DETECTOR_SCALES = (1.0, 0.75, 0.5)
ATTRITION_ITEMS = 4096
ATTRITION_BYTES = 4 * 1024 * 1024
UPSTREAM_DETAIL_FIELDS = frozenset(("field", "detail", "stage", "cap", "limit", "observed"))
INGRESS_METADATA_FIELDS = frozenset(("schema", "frame_id", "source_label", "byte_sha256", "byte_length",
    "decoded_shape", "decoded_dtype", "decoded_pixel_sha256", "transform", "decoder"))
UPSTREAM_REFUSAL_FIELDS = {
    "invalid_inputs": ("ordered_png_paths", "png_path", "order_or_duplicates", "ordinary_png_file"),
    "input_inventory_changed": ("entry_order_or_deduplication", "source_label_cap", "entry_session_digests",
        "ingress_join", "ingress_metadata", "ingress_byte_binding"),
    "upstream_unavailable": ("lattice", "signatures"),
    "diagnostic_accounting_incomplete": ("upstream_accounting",),
    "report_cap": ("structure", "items", "bytes"),
    "invalid_report": ("key", "integer", "nonfinite", "scalar"),
}
_OBSERVATION_LOCK = threading.Lock()


class PairedRefusal(ValueError):
    def __init__(self, reason, field):
        self.reason, self.field = reason, field
        super().__init__(reason)


def _require(condition, reason, field):
    if not condition:
        raise PairedRefusal(reason, field)


def _modules():
    # All image/numerical imports are lazy; importing this script reads no inputs.
    from mc_photogrammetry.pipeline import image_derived_joint_graph as graph
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from mc_photogrammetry.pipeline import image_derived_entry as entry
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    from mc_photogrammetry.reconstruction import plane_graph as matcher
    from mc_photogrammetry.reconstruction import plane_placement as placement
    from mc_photogrammetry.pipeline import image_only
    from mc_photogrammetry.reconstruction import calibration, region_growth
    return SimpleNamespace(graph=graph, selection=selection, entry=entry, lattice=lattice,
        signatures=signatures, csp=csp, matcher=matcher, placement=placement,
        image_only=image_only, calibration=calibration, region_growth=region_growth)


def canonical_bytes(record):
    stack, count, lower = [(record, 0)], 0, 0
    while stack:
        value, depth = stack.pop()
        count += 1
        _require(count <= MAX_REPORT_ITEMS and depth <= 32, "report_cap", "structure")
        if type(value) is dict:
            _require(all(type(key) is str for key in value), "invalid_report", "key")
            _require(len(value) <= MAX_REPORT_ITEMS - count - len(stack), "report_cap", "items")
            lower += sum(len(key) for key in value)
            stack.extend((item, depth + 1) for item in value.values())
        elif type(value) in (list, tuple):
            _require(len(value) <= MAX_REPORT_ITEMS - count - len(stack), "report_cap", "items")
            stack.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            lower += len(value)
        elif type(value) is int:
            _require(value.bit_length() <= 64, "invalid_report", "integer")
        elif type(value) is float:
            import math
            _require(math.isfinite(value), "invalid_report", "nonfinite")
        else:
            _require(value is None or type(value) is bool, "invalid_report", "scalar")
        _require(lower + count <= MAX_REPORT_BYTES, "report_cap", "bytes")
    data = bytearray()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(record):
        raw = part.encode("utf-8")
        _require(len(data) + len(raw) <= MAX_REPORT_BYTES, "report_cap", "bytes")
        data.extend(raw)
    return bytes(data)


def _digest(record):
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


@dataclass(frozen=True)
class PairedResult:
    record: dict
    full_graph: object = None
    kept_evaluation: object = None


class _ObservedCalls:
    def __init__(self):
        self.phase = "full.production"
        self.validation_depth = 0
        self.matcher_calls = 0
        self.universe = self.kept = self.readings = self.signatures = None
        self.csp_calls, self.validations = [], []


@contextmanager
def _observe(modules, observed):
    matcher = modules.matcher.match_plane_graph
    solver = modules.csp.solve_endpoint_hypotheses
    validator = modules.selection.validate_candidate_certificate
    _require(threading.active_count() == 1, "inconclusive", "single_thread_required")
    _require(_OBSERVATION_LOCK.acquire(blocking=False), "inconclusive", "reentrant_observation")

    def match(readings, signatures, **kwargs):
        _require(observed.matcher_calls == 0, "matcher_multiplicity", "matcher")
        observed.matcher_calls += 1
        observed.readings, observed.signatures = readings, signatures
        kept = matcher(readings, signatures, **kwargs)
        sink = kwargs.get("admitted_candidate_sink")
        _require(type(sink) is list and len(sink) == 1, "source_mismatch", "admitted_universe")
        _require(type(kept) is tuple, "source_mismatch", "kept_return")
        observed.universe, observed.kept = sink[0], kept
        return kept  # exact original object, unchanged

    def solve(*args, **kwargs):
        _require(len(observed.csp_calls) < MAX_CSP_CALLS, "inconclusive", "csp_call_cap")
        result = solver(*args, **kwargs)
        observed.csp_calls.append({"phase": observed.phase,
            "purpose": "validation" if observed.validation_depth else "solve",
            "status": result.status, "counters": dict(result.counters),
            "denial": result.denial, "exhaustive": result.exhaustive})
        return result

    def validate(certificate, *args, **kwargs):
        start = len(observed.csp_calls)
        observed.validation_depth += 1
        accepted = False
        try:
            result = validator(certificate, *args, **kwargs)
            accepted = True
            return result
        finally:
            observed.validation_depth -= 1
            observed.validations.append({"phase": observed.phase, "status": certificate.status,
                "accepted": accepted, "csp_recomputations": len(observed.csp_calls) - start})

    try:
        modules.matcher.match_plane_graph = match
        modules.csp.solve_endpoint_hypotheses = solve
        modules.selection.validate_candidate_certificate = validate
        yield
    finally:
        modules.selection.validate_candidate_certificate = validator
        modules.csp.solve_endpoint_hypotheses = solver
        modules.matcher.match_plane_graph = matcher
        _OBSERVATION_LOCK.release()


def _verify_mapping(modules, universe, kept, row_map):
    _require(type(universe) is modules.matcher.AdmittedCandidateUniverse,
        "source_mismatch", "universe_type")
    _require(universe.source_rows == tuple(range(len(universe.candidates)))
        and all(type(row) is int for row in universe.source_rows), "source_mismatch", "source_rows")
    expected = tuple(i for i, disposition in enumerate(universe.dispositions)
        if disposition == modules.matcher.SAME_AXIS_DISPOSITION_KEPT)
    _require(type(row_map) is tuple and all(type(row) is int for row in row_map)
        and row_map == expected and all(a < b for a, b in zip(row_map, row_map[1:])),
        "source_mismatch", "kept_row_map")
    _require(type(kept) is tuple and len(kept) == len(row_map)
        and all(row is universe.candidates[index] for row, index in zip(kept, row_map)),
        "source_mismatch", "kept_return_join")
    for row, index in zip(kept, row_map):
        _require(modules.graph.legacy._match_payload(row) ==
            modules.graph.legacy._match_payload(universe.candidates[index]),
            "source_mismatch", "kept_payload")


def _validate_arm(modules, certificate, graph, candidates, dispositions, budget):
    expected = (graph.session_sha256, graph.lattice_sha256, graph.signatures_sha256, graph.connection_sha256)
    actual = (certificate.session_sha256, certificate.lattice_sha256,
        certificate.signatures_sha256, certificate.connection_sha256)
    _require(actual == expected and certificate.frame_ids == graph.frame_ids
        and certificate.quadrant_constraints == graph.selection.quadrant_constraints
        and certificate.budget == budget and certificate.candidate_dispositions == dispositions,
        "source_mismatch", "certificate_context")
    modules.selection.validate_candidate_certificate(certificate, graph.readings, graph.signatures, candidates)


def _arm_record(modules, certificate, status, reason, detail, matches, solution, calls, row_map):
    proof = certificate.proof_record()
    representatives = []
    for rep in certificate.representatives:
        _require(type(rep.source_row) is int and 0 <= rep.source_row < len(row_map),
            "source_mismatch", "representative_row")
        representatives.append({"factor": rep.factor_index, "local_source_row": rep.source_row,
            "full_source_row": row_map[rep.source_row],
            "match": modules.graph.legacy._match_payload(rep.match)})
    return {"status": status, "reason": reason, "detail": detail,
        "selection_status": certificate.status, "certificate_sha256": certificate.result_sha256,
        "certificate": modules.selection._payload(certificate), "representatives": representatives,
        "counters": proof["counters"], "denial": proof["denial"], "exhaustive": proof["exhaustive"],
        "geometry_evaluations": calls, "accepted_matches": [modules.graph.legacy._match_payload(row) for row in matches],
        "component": modules.graph.legacy._solution_payload(solution)}


def _root_record(observed):
    return {"schema": SCHEMA, "cohort": COHORT, "purpose": "paired consistency/availability diagnostic reuse",
        "accuracy_or_generalization_claim": False, "kept_production_authorized": False,
        "matcher_calls": observed.matcher_calls, "csp_calls": observed.csp_calls,
        "certificate_validations": observed.validations}


def compare_sources(*, session, lattice_result, signatures_result, connection_graph=None, budget=None):
    """Evaluation-only source-record seam; the CLI never accepts these records."""
    modules, observed = _modules(), _ObservedCalls()
    full = None
    try:
        budget = modules.csp.EndpointBudget() if budget is None else budget
        _require(type(budget) is modules.csp.EndpointBudget, "invalid_configuration", "budget")
        with _observe(modules, observed):
            full = modules.graph.derive_image_derived_graph(session=session, lattice_result=lattice_result,
                signatures_result=signatures_result, connection_graph=connection_graph,
                max_shift_cells=SETTINGS[0], integer_tolerance=SETTINGS[1],
                min_resultant=SETTINGS[2], ambiguous_margin=SETTINGS[3], budget=budget)
            if isinstance(full, modules.placement.PlacementRefusal):
                raise PairedRefusal(full.reason, "full.production")
            _require(type(full) is modules.graph.ImageDerivedGraphResult and observed.matcher_calls == 1,
                "source_mismatch", "full_graph")
            universe = observed.universe
            _require(full.candidate_matches is universe.candidates
                and full.selection.candidate_dispositions == universe.dispositions
                and observed.readings is full.readings
                and len(observed.signatures) == len(full.signatures)
                and all(raw is evidence.signature for raw, evidence in zip(observed.signatures, full.signatures)),
                "source_mismatch", "full_capture_join")
            _require((full.session_sha256, full.lattice_sha256, full.signatures_sha256, full.connection_sha256) ==
                (session.session_sha256, lattice_result.result_sha256, signatures_result.result_sha256,
                 "" if connection_graph is None else connection_graph.graph_sha256),
                "source_mismatch", "upstream_pins")
            row_map = tuple(i for i, value in enumerate(universe.dispositions)
                if value == modules.matcher.SAME_AXIS_DISPOSITION_KEPT)
            _verify_mapping(modules, universe, observed.kept, row_map)
            observed.phase = "full.source_replay"
            modules.graph._require_live(full)
            observed.phase = "full.explicit_validation"
            _validate_arm(modules, full.selection, full, universe.candidates, universe.dispositions, budget)
            observed.phase = "kept.selection"
            dispositions = tuple(universe.dispositions[index] for index in row_map)
            kept_certificate = modules.selection.select_endpoint_hypotheses(readings=full.readings,
                signatures=full.signatures, candidate_matches=observed.kept, frame_ids=full.frame_ids,
                quadrant_constraints=full.selection.quadrant_constraints, budget=budget,
                session_sha256=full.session_sha256, lattice_sha256=full.lattice_sha256,
                signatures_sha256=full.signatures_sha256, connection_sha256=full.connection_sha256,
                candidate_dispositions=dispositions)
            observed.phase = "kept.explicit_validation"
            _validate_arm(modules, kept_certificate, full, observed.kept, dispositions, budget)
            observed.phase = "kept.shared_evaluation"
            kept = modules.graph._evaluate_selection(readings=full.readings, signatures=full.signatures,
                candidate_matches=observed.kept, frame_ids=full.frame_ids, certificate=kept_certificate)
        full_rows = tuple(range(len(universe.candidates)))
        full_record = _arm_record(modules, full.selection, full.status, full.reason, full.detail,
            full.matches, full.component_solution, full.geometry_evaluations, full_rows)
        kept_record = _arm_record(modules, kept_certificate, "assembled" if kept.solution is not None else "unavailable",
            kept.reason, kept.detail, kept.matches, kept.solution, kept.calls, row_map)
        exhausted = full.selection.status == "budget_refused" or kept_certificate.status == "budget_refused"
        record = dict(_root_record(observed), status="inconclusive" if exhausted else "complete",
            reason="budget_exhausted" if exhausted else "", settings=SETTINGS, budget=asdict(budget),
            full_graph_sha256=full.result_sha256, full_graph_payload=modules.graph._payload(full),
            full=full_record, kept=kept_record, kept_to_full_source_rows=row_map,
            universe={"rows": [modules.graph.legacy._match_payload(row) for row in universe.candidates],
                "dispositions": universe.dispositions, "source_rows": universe.source_rows},
            comparison="inconclusive" if exhausted else "same" if (full.status, full.reason, full.selection.status) ==
                (kept_record["status"], kept.reason, kept_certificate.status) else "different_consistency_or_availability")
        record["record_sha256"] = _digest(record)
        return PairedResult(record, full, kept)
    except (PairedRefusal, ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError) as error:
        reason = error.reason if isinstance(error, PairedRefusal) else "validation_or_evaluation_refused"
        field = error.field if isinstance(error, PairedRefusal) else type(error).__name__
        record = dict(_root_record(observed), status="inconclusive", reason=reason, field=field)
        record["record_sha256"] = _digest(record)
        return PairedResult(record, full)


def _png_paths(images):
    _require(type(images) in (tuple, list) and 2 <= len(images) <= MAX_IMAGES,
        "invalid_inputs", "ordered_png_paths")
    _require(all(type(item) is str or isinstance(item, Path) for item in images),
        "invalid_inputs", "png_path")
    paths = tuple(Path(item).resolve() for item in images)
    labels = tuple(str(path) for path in paths)
    _require(labels == tuple(sorted(set(labels), key=str.casefold)), "invalid_inputs", "order_or_duplicates")
    _require(all(path.suffix.lower() == ".png" and path.is_file() for path in paths),
        "invalid_inputs", "ordinary_png_file")
    return paths


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _plain_scalar(value, *, text_limit=MAX_UPSTREAM_TEXT_CHARS):
    """No repr, exception text, paths or containers in stage diagnostic fields."""
    if value is None or type(value) is bool:
        return True, value
    if type(value) is int:
        return value.bit_length() <= 64, value
    if type(value) is float:
        import math
        return math.isfinite(value), value
    if type(value) is str:
        valid = len(value) <= text_limit and ".png" not in value.casefold() and all(
            char.isascii() and (char.isalnum() or char in "_. ,;()-=") for char in value)
        return valid, value
    return False, None


def _upstream_stage(result, placement):
    refused = isinstance(result, placement.PlacementRefusal)
    summary = {"kind": "placement_refusal" if refused else "result"}
    for name in ("status", "reason", "terminal"):
        accepted, value = _plain_scalar(getattr(result, name, None), text_limit=128)
        summary[name] = value if accepted and (value is None or type(value) is str) else None
    digest = getattr(result, "refusal_sha256" if refused else "result_sha256", None)
    summary["source_result_sha256"] = digest if _is_sha256(digest) else None
    detail = getattr(result, "detail", None)
    if type(detail) is dict:
        if len(detail) > 8:
            summary.update(detail=None, detail_capture="omitted_cap")
        else:
            captured, complete = {}, True
            for name, value in detail.items():
                if type(name) is not str or name not in UPSTREAM_DETAIL_FIELDS:
                    complete = False
                    continue
                accepted, scalar = _plain_scalar(value)
                if accepted:
                    captured[name] = scalar
                else:
                    complete = False
            summary.update(detail=captured, detail_capture="complete" if complete else "partial")
    else:
        accepted, scalar = _plain_scalar(detail)
        summary.update(detail=scalar if accepted else None,
                       detail_capture="unavailable" if detail is None else "complete" if accepted else "omitted_unsupported")
    return summary


def _ingress_metadata(pin, label):
    record = pin.to_record()
    _require(type(record) is dict and len(record) <= len(INGRESS_METADATA_FIELDS)
        and all(type(key) is str and key in INGRESS_METADATA_FIELDS for key in record),
        "input_inventory_changed", "ingress_metadata")
    _require(record.get("source_label") == label and _is_sha256(record.get("byte_sha256"))
        and type(record.get("byte_length")) is int and 0 < record["byte_length"] <= (1 << 63) - 1,
        "input_inventory_changed", "ingress_byte_binding")
    _require(record["byte_sha256"] == pin.byte_sha256 and record["byte_length"] == pin.byte_length,
        "input_inventory_changed", "ingress_byte_binding")
    captured = dict(record)
    for key in ("schema", "frame_id", "decoded_dtype", "decoder"):
        if key in record:
            value = record[key]
            _require(type(value) is str and len(value) <= 128 and ".png" not in value.casefold()
                and "/" not in value and "\\" not in value,
                "input_inventory_changed", "ingress_metadata")
    if "frame_id" in record:
        _require(record["frame_id"] == pin.frame_id, "input_inventory_changed", "ingress_metadata")
    if "decoded_pixel_sha256" in record:
        _require(_is_sha256(record["decoded_pixel_sha256"]), "input_inventory_changed", "ingress_metadata")
    if "decoded_shape" in record:
        shape = record["decoded_shape"]
        _require(type(shape) in (tuple, list) and len(shape) in (2, 3)
            and all(type(size) is int and 0 < size <= (1 << 31) - 1 for size in shape),
            "input_inventory_changed", "ingress_metadata")
        captured["decoded_shape"] = list(shape)
    if "transform" in record:
        transform = record["transform"]
        _require(type(transform) is dict and len(transform) <= 4
            and all(type(key) is str and key in ("source_size", "crop_origin", "crop_size", "output_size")
                for key in transform), "input_inventory_changed", "ingress_metadata")
        copied = {}
        for key, pair in transform.items():
            _require(pair is None or type(pair) in (tuple, list) and len(pair) == 2
                and all(type(size) is int and 0 <= size <= (1 << 31) - 1 for size in pair),
                "input_inventory_changed", "ingress_metadata")
            copied[key] = None if pair is None else list(pair)
        captured["transform"] = copied
    return captured


def _count(value, limit=MAX_DIAGNOSTIC_COUNT):
    return value if type(value) is int and 0 <= value <= limit else None


def _code(value):
    return (value if type(value) is str and 0 < len(value) <= 96 and
            all(char.isascii() and (char.isalnum() or char == "_") for char in value) else None)


def _frame_id(value):
    """The bounded production discover_media identifier, separate from codes."""
    return (value if type(value) is str and len(value) == 10 and value.startswith("image-")
            and all("0" <= char <= "9" for char in value[6:])
            and int(value[6:]) < MAX_IMAGES else None)


def _line_pass_metrics(result, scale):
    """Scalar projection of the original detector return; no finite-row filter.

    Arithmetic deliberately matches calibration.detect_segments, including its
    NaN/Inf comparison behavior. Only finite lifted rows enter length bins.
    """
    keys = ("raw_detection_count", "finite_row_count", "nonfinite_row_count",
            "exact_ge40_count", "nonfinite_eligible_count")
    unknown = dict.fromkeys(keys)
    unknown["lifted_length_bins"] = None
    try:
        if type(scale) not in (int, float) or scale not in CANDIDATE_DETECTOR_SCALES:
            return unknown
        if type(result) is not tuple or not 1 <= len(result) <= 4:
            return unknown
        found = result[0]
        if found is None:
            return dict.fromkeys(keys, 0) | {"lifted_length_bins": dict.fromkeys(
                ("lt39", "39_to_lt40", "40_to_lt41", "ge41"), 0)}
        import numpy as np
        # Bound native shape/dtype before lifting or vectorized arithmetic.
        if (type(found) is not np.ndarray
                or not ((found.ndim == 2 and found.shape[1:] == (4,))
                        or (found.ndim == 3 and found.shape[1:] == (1, 4)))
                or _count(found.shape[0], MAX_LINE_PASS_ROWS) is None
                or found.dtype not in (np.dtype("float32"), np.dtype("float64"))):
            return unknown
        with np.errstate(all="ignore"):
            lifted = found.reshape(-1, 4) / scale
            lengths = np.hypot(lifted[:, 2] - lifted[:, 0], lifted[:, 3] - lifted[:, 1])
            finite = np.isfinite(lifted).all(axis=1)
            eligible = lengths >= 40.0
            bins = {"lt39": int((finite & (lengths < 39.0)).sum()),
                    "39_to_lt40": int((finite & (lengths >= 39.0) & (lengths < 40.0)).sum()),
                    "40_to_lt41": int((finite & (lengths >= 40.0) & (lengths < 41.0)).sum()),
                    "ge41": int((finite & (lengths >= 41.0)).sum())}
            finite_count = int(finite.sum())
            exact = int(eligible.sum())
            nonfinite_eligible = int((~finite & eligible).sum())
        if (sum(bins.values()) != finite_count
                or bins["40_to_lt41"] + bins["ge41"] + nonfinite_eligible != exact):
            return unknown
        return {"raw_detection_count": int(found.shape[0]), "finite_row_count": finite_count,
                "nonfinite_row_count": int(found.shape[0]) - finite_count,
                "lifted_length_bins": bins, "exact_ge40_count": exact,
                "nonfinite_eligible_count": nonfinite_eligible}
    except Exception:
        # Do not publish exception text, coordinates, arrays or partial metrics.
        return unknown


@contextmanager
def _observe_fit_evidence(modules, original_gray, evidence, *, expected_scales=BASE_DETECTOR_SCALES):
    """Join actual transform/detector objects; retain only scalar evidence."""
    calibration = modules.calibration
    enhance, detect = getattr(calibration, "enhance", None), getattr(calibration, "detect_segments", None)
    evidence.update(status="incomplete", raw_detections_per_pass=[], retained_segment_count=None,
                    result_segments=None, inliers=None, per_axis_inliers=None, abstention_reason=None,
                    enhance_calls=0, segment_calls=0, detector_calls=0, resize_calls=0,
                    identity_references_cleared=False)
    if enhance is None or detect is None:
        evidence["identity_references_cleared"] = True
        yield
        return
    cv2 = getattr(modules, "cv2", None)
    if cv2 is None:
        import cv2
    resize, factory = cv2.resize, cv2.createLineSegmentDetector
    state = {"complete": True, "detecting": False, "factory_calls": 0,
             "scales": (), "enhanced": None, "resized": None, "pass_inputs": []}

    def invalidate():
        state["complete"] = False

    def enhance_observed(*args, **kwargs):
        evidence["enhance_calls"] += 1
        if len(args) != 1 or args[0] is not original_gray or kwargs or evidence["enhance_calls"] != 1:
            invalidate()
        result = enhance(*args, **kwargs)
        if result is original_gray or result is None:
            invalidate()
        if evidence["enhance_calls"] == 1:
            state["enhanced"] = result
        return result

    def resize_observed(*args, **kwargs):
        evidence["resize_calls"] += 1
        ordinal = evidence["detector_calls"]
        scales = state["scales"]
        if (not state["detecting"] or ordinal >= len(scales) or scales[ordinal] == 1.0
                or len(args) != 2 or args[0] is not state["enhanced"] or args[1] is not None
                or set(kwargs) != {"fx", "fy", "interpolation"}
                or type(kwargs.get("fx")) not in (int, float) or type(kwargs.get("fy")) not in (int, float)
                or type(kwargs.get("interpolation")) is not int
                or kwargs["fx"] != scales[ordinal] or kwargs["fy"] != scales[ordinal]
                or kwargs["interpolation"] != cv2.INTER_AREA
                or state["resized"] is not None):
            invalidate()
        result = resize(*args, **kwargs)
        if result is original_gray or result is state["enhanced"] or any(result is item for item in state["pass_inputs"]):
            invalidate()
        state["resized"] = result
        return result

    class DetectorProxy:
        def __init__(self, delegate):
            self.delegate = delegate

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def detect(self, *args, **kwargs):
            ordinal = evidence["detector_calls"]
            evidence["detector_calls"] += 1
            scales = state["scales"]
            scale = scales[ordinal] if ordinal < len(scales) else None
            expected = state["enhanced"] if scale == 1.0 else state["resized"]
            if (not state["detecting"] or scale is None or expected is None
                    or len(args) != 1 or args[0] is not expected or kwargs
                    or any(args[0] is item for item in state["pass_inputs"])):
                invalidate()
            if args and len(state["pass_inputs"]) < 4:
                state["pass_inputs"].append(args[0])
            result = self.delegate.detect(*args, **kwargs)
            metrics = _line_pass_metrics(result, scale)
            if metrics["raw_detection_count"] is None:
                invalidate()
            if len(evidence["raw_detections_per_pass"]) < 4:
                evidence["raw_detections_per_pass"].append({"scale": scale, **metrics})
            else:
                invalidate()
            state["resized"] = None
            return result

    def factory_observed(*args, **kwargs):
        state["factory_calls"] += 1
        if not state["detecting"] or state["factory_calls"] != 1:
            invalidate()
        return DetectorProxy(factory(*args, **kwargs))

    def detect_observed(*args, **kwargs):
        evidence["segment_calls"] += 1
        scales = kwargs.get("scales", getattr(calibration, "DETECTION_SCALES", None))
        known_scales = (type(scales) is tuple and all(type(value) in (int, float) for value in scales)
                        and scales == expected_scales)
        if (len(args) != 1 or args[0] is not state["enhanced"] or state["enhanced"] is None
                or evidence["segment_calls"] != 1 or state["detecting"] or not known_scales
                or not set(kwargs) <= {"scales", "minimum_length"}
                or type(kwargs.get("minimum_length", 40.0)) not in (int, float)
                or kwargs.get("minimum_length", 40.0) != 40.0):
            invalidate()
        state["scales"] = scales if known_scales else ()
        state["detecting"] = True
        try:
            result = detect(*args, **kwargs)
        finally:
            state["detecting"] = False
        try:
            retained = _count(len(result))
        except Exception:
            retained = None
        if evidence["segment_calls"] == 1:
            evidence["retained_segment_count"] = retained
        if retained is None:
            invalidate()
        return result

    try:
        calibration.enhance, calibration.detect_segments = enhance_observed, detect_observed
        cv2.resize, cv2.createLineSegmentDetector = resize_observed, factory_observed
        try:
            yield
        except BaseException:
            invalidate()
            raise
    finally:
        cv2.createLineSegmentDetector, cv2.resize = factory, resize
        calibration.detect_segments, calibration.enhance = detect, enhance
        passes = evidence["raw_detections_per_pass"]
        retained = evidence["retained_segment_count"]
        valid = (state["complete"] and evidence["enhance_calls"] == evidence["segment_calls"] == state["factory_calls"] == 1
                 and evidence["detector_calls"] == len(expected_scales)
                 and evidence["resize_calls"] == sum(scale != 1.0 for scale in expected_scales)
                 and tuple(row["scale"] for row in passes) == expected_scales
                 and all(row["raw_detection_count"] is not None for row in passes)
                 and retained is not None and retained == sum(row["exact_ge40_count"] for row in passes))
        state["enhanced"] = state["resized"] = None
        state["pass_inputs"].clear()
        evidence["identity_references_cleared"] = True
        evidence["status"] = "complete" if valid else "incomplete"


def _finish_fit_evidence(evidence, result):
    retained = evidence["retained_segment_count"]
    if result is None:
        if evidence["status"] == "complete" and retained < 12:
            evidence["abstention_reason"] = "insufficient_segment_count"
        else:
            evidence["status"] = "incomplete"
        return
    try:
        segments = _count(getattr(result, "segments", None))
        inliers = _count(getattr(result, "inliers", None))
        axes = getattr(result, "per_axis_inliers", None)
        valid = (segments is not None and inliers is not None and segments == retained and segments >= 12
                 and inliers <= segments and type(axes) is tuple and len(axes) == 3
                 and all(_count(value) is not None for value in axes) and sum(axes) == inliers)
        evidence["result_segments"], evidence["inliers"] = segments, inliers
        evidence["per_axis_inliers"] = list(axes) if valid else None
    except Exception:
        valid = False
    if not valid:
        evidence["status"] = "incomplete"


class _CameraAccounting:
    def __init__(self):
        self.complete = True
        self.solve_calls = self.session_calls = self.fit_calls = 0
        self.frame_ids, self.events, self.conversion_order = (), [], []
        self.inputs, self.grays, self.converted = {}, {}, set()
        self.active = False


@contextmanager
def _observe_camera(modules, accounting, *, detector_scales=None):
    """Observe independent existing fits by identity, never a detector pass."""
    _require(detector_scales is None or (type(detector_scales) is tuple
        and all(type(value) is float for value in detector_scales)
        and detector_scales == CANDIDATE_DETECTOR_SCALES),
        "diagnostic_accounting_incomplete", "upstream_accounting")
    expected_scales = BASE_DETECTOR_SCALES if detector_scales is None else detector_scales
    image_only, calibration = getattr(modules, "image_only", None), getattr(modules, "calibration", None)
    if image_only is None or calibration is None:
        accounting.complete = False
        yield
        return
    solve, gray = image_only.solve_image_cameras, image_only._to_grayscale
    session, fit = calibration.estimate_session, calibration.estimate_camera
    _require(threading.active_count() == 1 and _OBSERVATION_LOCK.acquire(blocking=False),
             "diagnostic_accounting_incomplete", "upstream_accounting")

    def solve_observed(frames, *args, **kwargs):
        accounting.solve_calls += 1
        try:
            if accounting.active or accounting.solve_calls != 1 or type(frames) is not dict or not 1 <= len(frames) <= MAX_IMAGES:
                accounting.complete = False
            else:
                accounting.frame_ids = tuple(frames)
                for frame, image in frames.items():
                    if _frame_id(frame) is None or id(image) in accounting.inputs:
                        accounting.complete = False
                    else:
                        accounting.inputs[id(image)] = (frame, image)
        except Exception:
            accounting.complete = False
        previous, accounting.active = accounting.active, True
        try:
            return solve(frames, *args, **kwargs)
        except BaseException:
            accounting.complete = False
            raise
        finally:
            accounting.active = previous
            if not previous:
                accounting.inputs.clear()
                accounting.grays.clear()

    def gray_observed(image, *args, **kwargs):
        try:
            result = gray(image, *args, **kwargs)
        except BaseException:
            if accounting.active:
                accounting.complete = False
            raise
        if accounting.active:
            binding = accounting.inputs.get(id(image))
            if binding is None or binding[1] is not image or binding[0] in accounting.converted or id(result) in accounting.grays:
                accounting.complete = False
            else:
                accounting.converted.add(binding[0])
                accounting.conversion_order.append(binding[0])
                accounting.grays[id(result)] = (binding[0], result)
        return result

    def session_observed(frames, *args, **kwargs):
        if accounting.active:
            accounting.session_calls += 1
            accounting.complete = False
        return session(frames, *args, **kwargs)

    def fit_observed(*args, **kwargs):
        event = None
        if accounting.active:
            accounting.fit_calls += 1
            image = args[0] if args else kwargs.get("grey")
            binding = accounting.grays.get(id(image))
            frame = binding[0] if binding is not None and binding[1] is image else None
            ordinal = accounting.fit_calls - 1
            if (frame is None or ordinal >= len(accounting.frame_ids)
                    or frame != accounting.frame_ids[ordinal] or len(args) != 1 or kwargs):
                accounting.complete = False
            if len(accounting.events) >= 2 * MAX_IMAGES:
                accounting.complete = False
            else:
                event = {"frame_id": frame,
                         "phase": "fixed_focal" if kwargs.get("focal") is not None else "free_focal",
                         "segments": None, "disposition": "raised"}
                accounting.events.append(event)
        try:
            if event is None:
                result = fit(*args, **kwargs)
            else:
                event["calibration_evidence"] = {}
                with _observe_fit_evidence(modules, image, event["calibration_evidence"],
                                           expected_scales=expected_scales):
                    # Experimental override belongs only to this diagnostic's
                    # active independent fit. Never modify calibration defaults.
                    if detector_scales is not None and len(args) == 1 and not kwargs:
                        result = fit(*args, scales=detector_scales)
                    else:
                        result = fit(*args, **kwargs)
                _finish_fit_evidence(event["calibration_evidence"], result)
                if event["calibration_evidence"]["status"] != "complete":
                    accounting.complete = False
        except BaseException:
            if accounting.active:
                accounting.complete = False
            raise
        if event is not None:
            event["disposition"] = "unavailable" if result is None else "returned"
        return result

    try:
        image_only.solve_image_cameras, image_only._to_grayscale = solve_observed, gray_observed
        calibration.estimate_session, calibration.estimate_camera = session_observed, fit_observed
        yield
    finally:
        calibration.estimate_camera, calibration.estimate_session = fit, session
        image_only._to_grayscale, image_only.solve_image_cameras = gray, solve
        # Drop object references before any report serialization.
        accounting.inputs.clear()
        accounting.grays.clear()
        _OBSERVATION_LOCK.release()


def _camera_record(session, accounting, ingress):
    base = {"status": "incomplete", "admission_evidence": "unavailable", "active_frame_ids": [],
            "frames": [], "fitting_mode": "independent_per_frame",
            "solve_calls": accounting.solve_calls, "session_calls": accounting.session_calls,
            "fit_calls": accounting.fit_calls}
    if (type(ingress) is not list or not 1 <= len(ingress) <= MAX_IMAGES
            or any(type(row) is not dict for row in ingress)):
        return base
    ids = tuple(row.get("frame_id") for row in ingress)
    if not all(_frame_id(frame) for frame in ids) or len(set(ids)) != len(ids):
        return base
    complete = (accounting.complete and accounting.solve_calls == 1 and accounting.session_calls == 0
                and accounting.fit_calls == len(ids) and accounting.frame_ids == ids
                and accounting.converted == set(ids) and tuple(accounting.conversion_order) == ids
                and tuple(event["frame_id"] for event in accounting.events) == ids)
    frames, admissions = getattr(session, "frames", None), getattr(session, "admissions", None)
    inputs = getattr(session, "input_frames", None)
    complete = complete and (type(inputs) is tuple and len(inputs) == len(ids)
                              and tuple(frame.frame_id for frame in inputs) == ids)
    active = tuple(frame.frame_id for frame in frames) if type(frames) is tuple and len(frames) <= MAX_IMAGES else ()
    version = getattr(session, "version", None)
    admission_basis = "explicit_session_admissions"
    if version == 1 and type(admissions) is tuple and not admissions and active == ids:
        rows = [{"frame_id": frame, "status": "admitted", "stage": "camera_recovery", "reason": ""} for frame in ids]
        admission_basis = "session_v1_all_admitted"
    elif version == 2 and type(admissions) is tuple and len(admissions) == len(ids):
        rows = [{name: getattr(row, name, None) for name in ("frame_id", "status", "stage", "reason")} for row in admissions]
    else:
        rows, complete = [], False
    joined = tuple(row["frame_id"] for row in rows) == ids
    joined = joined and all(row["status"] in ("admitted", "excluded") and row["stage"] == "camera_recovery"
        and row["reason"] == ("camera_estimate_unavailable" if row["status"] == "excluded" else "") for row in rows)
    joined = joined and active == tuple(row["frame_id"] for row in rows if row["status"] == "admitted")
    if not joined:
        rows, active, complete = [], (), False
    output = []
    for index, frame in enumerate(ids):
        events = [event for event in accounting.events if event["frame_id"] == frame]
        valid = accounting.complete and len(events) == 1 and events[0]["phase"] == "free_focal"
        valid = valid and events[0]["disposition"] in ("returned", "unavailable")
        fit_evidence = events[0].get("calibration_evidence", {}) if len(events) == 1 else {}
        valid = valid and fit_evidence.get("status") == "complete" and fit_evidence.get("identity_references_cleared") is True
        admission = rows[index] if joined else {"frame_id": frame, "status": None, "stage": "camera_recovery", "reason": None}
        valid = valid and joined and (admission["status"] == "admitted") == (events[0]["disposition"] == "returned")
        complete = complete and valid
        output.append(dict(admission, segments=fit_evidence.get("retained_segment_count") if valid else None,
            segments_unavailable_reason=None if valid else "same_pass_evidence_incomplete",
            calibration_evidence=fit_evidence,
            initial_disposition=events[0]["disposition"] if len(events) == 1 else None,
            fixed_focal_disposition=None,
            capture="complete" if complete else "incomplete",
            lattice_stage="pending" if admission["status"] == "admitted" else "not_reached" if admission["status"] == "excluded" else "unknown"))
    if not complete:
        for row in output:
            row["capture"] = "incomplete"
    return dict(base, status="complete" if complete else "incomplete", admission_evidence=admission_basis,
                active_frame_ids=list(active), frames=output)


def _lattice_record(result, active):
    output = {"status": "incomplete", "frames": []}
    if (type(active) not in (tuple, list) or len(active) > MAX_IMAGES or not all(_frame_id(frame) for frame in active)
            or len(set(active)) != len(active)):
        return output
    frame_ids, calibrations = getattr(result, "frame_ids", None), getattr(result, "camera_calibrations", None)
    abstentions = getattr(result, "abstentions", None)
    if (type(frame_ids) is not tuple or frame_ids != tuple(active) or type(calibrations) is not tuple
            or len(calibrations) != len(active) or type(abstentions) is not tuple or len(abstentions) > MAX_DIAGNOSTIC_ROWS):
        return output
    reasons = {frame: {} for frame in active}
    for abstention in abstentions:
        frame, reason = getattr(abstention, "frame", None), _code(getattr(abstention, "reason", None))
        if _frame_id(frame) is None or frame not in reasons or reason is None:
            return output
        reasons[frame][reason] = reasons[frame].get(reason, 0) + 1
    for frame, calibration in zip(active, calibrations):
        segments, inliers = _count(getattr(calibration, "segments", None)), _count(getattr(calibration, "inliers", None))
        axes = getattr(calibration, "per_axis_inliers", None)
        if (segments is None or inliers is None or inliers > segments or type(axes) is not tuple or len(axes) != 3
                or any(_count(value) is None for value in axes) or sum(axes) != inliers):
            return {"status": "incomplete", "frames": []}
        output["frames"].append({"frame_id": frame, "segments": segments, "inliers": inliers,
                                  "per_axis_inliers": list(axes), "abstention_reasons": reasons[frame]})
    output["status"] = "complete"
    return output


def _attrition_record(record, active):
    """Project aggregates; never traverse examples, handoffs or detailed stages."""
    if (type(active) not in (tuple, list) or len(active) > MAX_IMAGES or not all(_frame_id(frame) for frame in active)
            or len(set(active)) != len(active)):
        return {"status": "incomplete", "stages": {}, "checks": {}, "frames": []}
    output = {"status": "incomplete", "stages": {}, "checks": {}, "frames": [
        {"frame_id": frame, "proposal_returned_count": None, "proposal_outcome": None,
         "reader_stage": "unknown", "reader_classified": None, "reader_accepted": None,
         "reader_dispositions": {}} for frame in active]}
    if type(record) is not dict:
        return output
    stages, checks = record.get("stages"), record.get("checks")
    if type(stages) is not dict or len(stages) > 32 or type(checks) is not dict or len(checks) > 4 * MAX_IMAGES:
        return output
    omitted, handoffs = _count(record.get("omitted_accounting_events")), _count(record.get("omitted_proposal_handoffs"))
    complete = record.get("status") == "complete" and omitted == handoffs == 0
    output.update(omitted_accounting_events=omitted, omitted_proposal_handoffs=handoffs)
    allowed_checks = {prefix + ":" + frame for frame in active for prefix in
                      ("proposal_generation", "proposal_returned", "proposal_to_reading")}
    allowed_checks.add("local_period_child_accounting")
    for name, value in checks.items():
        if type(name) is not str or name not in allowed_checks or type(value) is not bool:
            complete = False
        else:
            output["checks"][name] = value
            complete = complete and value
    groups, rows = {}, 0
    count_keys = ("expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events")
    for name in ("proposal", "reading", "frame"):
        stage = stages.get(name)
        if stage is None:
            continue
        if type(stage) is not dict or any(_count(stage.get(key), ATTRITION_ITEMS) is None for key in count_keys):
            complete = False
            continue
        counts = stage.get("counts")
        if type(counts) is not list or len(counts) > MAX_DIAGNOSTIC_ROWS - rows:
            complete = False
            continue
        rows += len(counts)
        captured, total = [], 0
        for group in counts:
            if type(group) is not dict:
                complete = False
                continue
            frame, code, count = group.get("frame"), _code(group.get("disposition")), _count(group.get("count"), ATTRITION_ITEMS)
            if _frame_id(frame) is None or frame not in active or code is None or count is None:
                complete = False
                continue
            metric = None
            if name == "frame":
                metrics = group.get("metrics")
                if type(metrics) is not dict or len(metrics) > 32:
                    complete = False
                    continue
                if "returned_count" in metrics:
                    candidate = metrics["returned_count"]
                    if (type(candidate) is dict and all(_count(candidate.get(key), 32) is not None
                            for key in ("count", "minimum", "maximum", "total")) and candidate["count"] == count == 1
                            and candidate.get("minimum") == candidate.get("maximum") == candidate["total"]):
                        metric = candidate["total"]
                    else:
                        complete = False
            captured.append({"frame_id": frame, "disposition": code, "count": count,
                             **({"returned_count": metric} if metric is not None else {})})
            groups.setdefault((name, frame), []).append((code, count, metric))
            total += count
        complete = complete and total == stage["classified"] == stage["expected"] and all(stage[key] == 0 for key in count_keys[2:])
        output["stages"][name] = {key: stage[key] for key in count_keys}
        output["stages"][name]["counts"] = captured
    for frame in output["frames"]:
        frame_id = frame["frame_id"]
        proposals = [(code, count, metric) for code, count, metric in groups.get(("frame", frame_id), ()) if metric is not None]
        if len(proposals) == 1:
            frame["proposal_outcome"], _, frame["proposal_returned_count"] = proposals[0]
        else:
            failures = [code for code, count, _ in groups.get(("frame", frame_id), ()) if code == "proposal_stage_failed" and count == 1]
            frame["proposal_outcome"] = failures[0] if len(failures) == 1 else None
            complete = False  # unknown return count is never replaced with zero
        reading = groups.get(("reading", frame_id), ())
        if reading or frame["proposal_returned_count"] == 0:
            frame["reader_classified"] = sum(count for _, count, _ in reading)
            frame["reader_accepted"] = sum(count for code, count, _ in reading if code == "accepted")
            for code, count, _ in reading:
                frame["reader_dispositions"][code] = frame["reader_dispositions"].get(code, 0) + count
        if frame["proposal_returned_count"] not in (None, 0):
            frame["reader_stage"] = "reached"
            generated_returned = sum(count for code, count, _ in groups.get(("proposal", frame_id), ()) if code == "returned")
            complete = complete and generated_returned == frame["reader_classified"] == frame["proposal_returned_count"] and all(
                output["checks"].get(prefix + ":" + frame_id) is True for prefix in
                ("proposal_generation", "proposal_returned", "proposal_to_reading"))
        elif frame["proposal_returned_count"] == 0:
            frame["reader_stage"] = "not_reached"
            complete = complete and frame["reader_classified"] == 0
    output["status"] = "complete" if complete else "incomplete"
    return output


@contextmanager
def _observe_lattice(modules, facts, active):
    region = getattr(modules, "region_growth", None)
    if region is None:
        facts["lattice_attrition"] = _attrition_record(None, active)
        yield
        return
    observer = region._ScalarAttritionObserver(max_items=ATTRITION_ITEMS, max_bytes=ATTRITION_BYTES)
    try:
        with region._scalar_attrition_context(observer):
            yield
    finally:
        try:
            facts["lattice_attrition"] = _attrition_record(observer.record(), active)
        except Exception:
            # Observer failure does not replace the original producer exception.
            facts["lattice_attrition"] = _attrition_record(None, active)


def evaluate_pngs(images, *, project_path, candidate_detector_scales=False):
    """Called only by a separately authorized image run, never during import."""
    modules, observed = _modules(), _ObservedCalls()
    facts, stages, stage = {}, {}, "inputs"
    try:
        _require(type(candidate_detector_scales) is bool,
                 "diagnostic_accounting_incomplete", "upstream_accounting")
        if candidate_detector_scales:
            facts["experiment_settings"] = {"detector_scales": list(CANDIDATE_DETECTOR_SCALES),
                "baseline_detector_scales": list(BASE_DETECTOR_SCALES),
                "scale_override_provenance": "scoped independent estimate_camera keyword; defaults unchanged",
                "minimum_length": 40.0, "camera_segment_gate": 12, "lattice_inlier_gate": 12,
                "fitting_mode": "independent_per_frame", "evaluation_scope": "camera_lattice_only"}
            facts.update(geometry_promoted=False, state_scoring_performed=False)
        paths = _png_paths(images)
        stage = "entry"
        camera = _CameraAccounting()
        with _observe_camera(modules, camera, detector_scales=(
                CANDIDATE_DETECTOR_SCALES if candidate_detector_scales else None)):
            entry = modules.entry.build_image_derived_entry(paths, project_path=project_path,
                recursive=False, capture_ingress=True, max_frames=MAX_IMAGES)
        _require(tuple(entry.source_paths) == tuple(str(path) for path in paths),
            "input_inventory_changed", "entry_order_or_deduplication")
        labels = [str(path) for path in paths]
        _require(all(len(label) <= MAX_SOURCE_LABEL_CHARS for label in labels),
            "input_inventory_changed", "source_label_cap")
        facts["ordered_png_paths"] = labels
        _require(_is_sha256(entry.entry_sha256) and _is_sha256(entry.session.session_sha256),
            "input_inventory_changed", "entry_session_digests")
        facts.update(entry_sha256=entry.entry_sha256, session_sha256=entry.session.session_sha256)
        ingress = tuple(frame.ingress for frame in entry.session.input_frames)
        _require(len(ingress) == len(paths) and all(pin is not None and
            pin.source_label == str(path) for pin, path in zip(ingress, paths)),
            "input_inventory_changed", "ingress_join")
        facts["input_ingress"] = [_ingress_metadata(pin, label) for pin, label in zip(ingress, labels)]
        facts["camera_accounting"] = _camera_record(entry.session, camera, facts["input_ingress"])
        active = facts["camera_accounting"]["active_frame_ids"]
        facts["diagnostic_accounting_status"] = "incomplete"
        stage = "lattice"
        for frame in facts["camera_accounting"]["frames"]:
            if frame["status"] == "admitted":
                frame["lattice_stage"] = "reached"
        with _observe_lattice(modules, facts, active):
            lattice = modules.lattice.measure_image_derived_lattice(session=entry.session, pose_graph=entry.pose_graph)
        stages[stage] = _upstream_stage(lattice, modules.placement)
        facts["lattice_accounting"] = _lattice_record(lattice, active)
        facts["diagnostic_accounting_status"] = ("complete" if all(facts[key]["status"] == "complete"
            for key in ("camera_accounting", "lattice_accounting", "lattice_attrition")) else "incomplete")
        _require(not isinstance(lattice, modules.placement.PlacementRefusal) and lattice.status == "assembled",
            "upstream_unavailable", "lattice")
        if candidate_detector_scales:
            _require(facts["diagnostic_accounting_status"] == "complete",
                     "diagnostic_accounting_incomplete", "upstream_accounting")
            record = dict(_root_record(observed), status="inconclusive",
                reason="candidate_camera_lattice_diagnostic", field="lattice", upstream_stages=stages)
            record.update(facts)
            record["record_sha256"] = _digest(record)
            return PairedResult(record)
        stage = "signatures"
        signatures = modules.signatures.derive_image_derived_signatures(session=entry.session, lattice_result=lattice)
        stages[stage] = _upstream_stage(signatures, modules.placement)
        _require(not isinstance(signatures, modules.placement.PlacementRefusal) and signatures.status == "assembled",
            "upstream_unavailable", "signatures")
        _require(facts["diagnostic_accounting_status"] == "complete",
                 "diagnostic_accounting_incomplete", "upstream_accounting")
        stage = "paired"
        paired = compare_sources(session=entry.session, lattice_result=lattice, signatures_result=signatures,
            connection_graph=entry.graph)
        record = dict(paired.record)
        record.pop("record_sha256", None)
        record.update(facts, upstream_stages=stages)
        record["record_sha256"] = _digest(record)
        return PairedResult(record, paired.full_graph, paired.kept_evaluation)
    except Exception as error:
        if stage in ("lattice", "signatures") and stage not in stages:
            stages[stage] = {"kind": "exception", "status": None, "reason": None,
                "detail": None, "detail_capture": "not_returned", "source_result_sha256": None}
        internal = type(error) is PairedRefusal and type(error.reason) is str and type(error.field) is str
        internal = internal and error.reason in UPSTREAM_REFUSAL_FIELDS and error.field in UPSTREAM_REFUSAL_FIELDS[error.reason]
        record = dict(_root_record(observed), status="inconclusive",
            reason=error.reason if internal else "upstream_error",
            field=error.field if internal else stage, upstream_stages=stages)
        if not internal:
            name = type(error).__name__
            accepted, name = _plain_scalar(name, text_limit=128)
            record["exception_class"] = name if accepted else "Exception"
        record.update(facts)
        record["record_sha256"] = _digest(record)
        return PairedResult(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-detector-scales", action="store_true",
        help="Diagnostic only: use scales 1.0,0.75,0.5 and stop after camera/lattice accounting.")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    options = {"project_path": args.project}
    if args.candidate_detector_scales:
        options["candidate_detector_scales"] = True
    result = evaluate_pngs(args.images, **options)
    with Path(args.output).open("xb") as handle:
        handle.write(canonical_bytes(result.record) + b"\n")
    return 0 if result.record["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
