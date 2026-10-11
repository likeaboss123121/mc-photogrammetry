"""Developer-only scalar attrition around one unchanged default delivery."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from mc_photogrammetry.reconstruction import region_growth as growth

_spec = importlib.util.spec_from_file_location(
    "_proposal_attrition_g7", Path(__file__).with_name("diagnose_g7_selection_failure.py"))
g7 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g7
_spec.loader.exec_module(g7)

SCHEMA = "mc-photogrammetry-proposal-attrition-diagnostic-v1"
PRODUCER = "proposal-attrition-diagnostic-1"
TASK = "DS-PROPOSAL-ATTRITION-DIAG-01"
TASK_SHA256 = "499454274de7ada4fddf7e2bf6bbe7d99b73dc7731d9cf91d54999b5d12a7f88"
MAX_RECORDS = 4096
MAX_BYTES = 4 * 1024 * 1024
MAX_NODES = 65536
MAX_DEPTH = 16
SOURCE_PINS = {
    "scripts/evaluation/diagnose_g7_selection_failure.py": "97bd1bac570074d4ff96692b17a236ebf06d79eca0338f556d9c4c440132aa1d",
    "src/mc_photogrammetry/reconstruction/region_growth.py": "b38ea177a161ff0beb093f9ee9b420ea9e31705c82a63fa2ed07cf0f3da3d5f9",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py": "fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py": "1be1c60ca8184e7cd1c62e7bfe11fcfea60c5e52e2aa080e8495cd3ea1035c04",
    "src/mc_photogrammetry/pipeline/image_derived_production_delivery.py": "4305c7156a451224523c4d691c933539c741b874569c5fe5341802061cb202e8",
    **g7.SOURCE_PINS,
}
NOTES = ["Aggregate scalar measurements only; no pixels, masks, positions or manual hints.",
         "G7 reasons count first failed predicates, not every defect.",
         "Partial execution-seam source pins; not a complete environment identity."]


class AttritionError(RuntimeError):
    def __init__(self, reason, field=""):
        self.reason, self.field = reason, field
        super().__init__(f"{reason}: {field}")


def canonical_bytes(value):
    """Charge escaped JSON before encoding; no changes to the G7 serializer."""
    stack, nodes, size = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise AttritionError("sidecar_budget", "structure")
        if type(item) in (dict, list, tuple):
            if nodes + len(stack) + len(item) > MAX_NODES:
                raise AttritionError("sidecar_budget", "items")
            if type(item) is dict:
                if any(type(k) is not str for k in item):
                    raise AttritionError("malformed_sidecar", "keys")
                for key in item:
                    size += g7._escaped_string_size(key) + 1
                    if size > MAX_BYTES:
                        raise AttritionError("sidecar_budget", "bytes")
                children = item.values()
            else:
                children = item
            size += 2 + max(0, len(item) - 1)
            stack.extend((child, depth + 1) for child in children)
        elif type(item) is str:
            size += g7._escaped_string_size(item)
        elif type(item) is bool or item is None:
            size += 4 if item is True or item is None else 5
        elif type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise AttritionError("sidecar_budget", "integer")
            size += len(str(item))
        elif type(item) is float and math.isfinite(item):
            size += len(repr(item))
        else:
            raise AttritionError("malformed_sidecar", "scalar")
        if size > MAX_BYTES:
            raise AttritionError("sidecar_budget", "bytes")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _seal(report):
    report = dict(report)
    report.pop("result_sha256", None)
    report["result_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def _predecode(raw):
    """Bound lexical nesting/items/scalar token length before JSON decoding."""
    if len(raw) > MAX_BYTES:
        raise AttritionError("sidecar_budget", "readback_bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AttritionError("malformed_sidecar", "canonical_ascii") from error
    stack, nodes, token = [], 0, 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char in " \r\n\t,:]}[{\"":
            token = 0
        else:
            token += 1
            if token > 64:
                raise AttritionError("sidecar_budget", "scalar_token")
        if char in '[{"' or token == 1:
            nodes += 1
            if nodes > MAX_NODES:
                raise AttritionError("sidecar_budget", "readback_items")
        if char in "[{":
            stack.append(char)
            if len(stack) > MAX_DEPTH:
                raise AttritionError("sidecar_budget", "readback_depth")
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise AttritionError("malformed_sidecar", "readback_structure")
        elif char == '"':
            quoted = True
    if stack or quoted:
        raise AttritionError("malformed_sidecar", "readback_structure")


class DeliveryCapture:
    """The context is limited to the synchronous original delivery call."""
    def __init__(self, delivery, observer=None):
        self.delivery, self.calls = delivery, 0
        self.observer = observer if observer is not None else growth._ScalarAttritionObserver(
            max_items=MAX_RECORDS, max_bytes=MAX_BYTES)
        if (type(self.observer) is not growth._ScalarAttritionObserver
                or not 0 < self.observer.max_items <= MAX_RECORDS
                or not 0 < self.observer.max_bytes <= MAX_BYTES):
            raise AttritionError("observer_budget", "limits")

    def __call__(self, *args, **kwargs):
        self.calls += 1
        with growth._scalar_attrition_context(self.observer):
            return self.delivery(*args, **kwargs)


def _g7_binding(report):
    g7.validate_report(report)
    if report["status"] not in ("complete", "no_failure") or report["g_calls"] != 1:
        raise AttritionError("g7_incomplete", report["reason"])
    graph = report["graph"]
    cert = graph["selection"]
    proof = cert.get("proof")
    if proof is not None and not proof["exhaustive"]:
        raise AttritionError("g7_incomplete", "partial_proof")
    binding = dict(diagnostic_sha256=report["result_sha256"], status=report["status"],
                graph_sha256=graph["result_sha256"], selection_sha256=cert["result_sha256"],
                selection_status=cert["status"], selection_reason=cert.get("reason", ""),
                frame_ids=graph["frame_ids"], counts=graph["counts"],
                proof_scope="compact_no_failure" if proof is None else "observed_exhaustive",
                assignment_counts=None if proof is None else proof["counters"],
                first_failure_reasons={} if proof is None else proof["eligibility_reasons"],
                geometry_evaluations=graph["geometry_evaluations"])
    # Live records can contain tuples; the sidecar always uses JSON containers.
    return json.loads(canonical_bytes(binding))


def _observer_record(observer):
    raw = observer.record()
    stages = {}
    for name, stage in raw["stages"].items():
        rows = []
        for row in stage["counts"]:
            item = dict(frame=row["frame"], axis=row["axis"], disposition=row["disposition"], count=row["count"])
            if name == "seed":
                item["eligible_seed_count"] = row["metrics"].get("eligible_seed_count")
            rows.append(item)
        stages[name] = {key: stage[key] for key in (
            "expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events")}
        stages[name].update(rows=rows, terminal_sha256=stage["terminal_sha256"])
    items = sum(len(s["expected"]) + len(s["terminal"]) + len(s["counts"])
                for s in observer.stages.values()) + len(observer.bindings)
    return dict(schema=raw["schema"], status=raw["status"], items=items, bytes=observer.bytes,
                overflow=observer.incomplete or observer.omitted > 0,
                omitted_accounting_events=raw["omitted_accounting_events"],
                omitted_proposal_handoffs=raw["omitted_proposal_handoffs"],
                checks=raw["checks"], stages=stages, proposal_handoffs=raw["proposal_handoffs"],
                limits=dict(items=observer.max_items, bytes=observer.max_bytes))


def build_report(capture, report):
    binding = _g7_binding(report)
    observer = _observer_record(capture.observer)
    if capture.calls != 1:
        raise AttritionError("delivery_call_count", str(capture.calls))
    if observer["status"] != "complete" or observer["overflow"]:
        raise AttritionError("observer_incomplete", "overflow_or_accounting")
    result = _seal(dict(schema=SCHEMA, producer=PRODUCER, status="complete",
        task=dict(id=TASK, implementation_sha256=TASK_SHA256), source_sha256=SOURCE_PINS,
        runner_sha256=g7._file_hash(Path(__file__)),
        inputs=[dict(index=item["index"], sha256=item["sha256"]) for item in report["run"]["inputs"]],
        g7=binding, observer=observer,
        notes=NOTES))
    return validate_report(result, report)


def _integer(value):
    return type(value) is int and 0 <= value < 2**63


def validate_report(report, g7_report):
    """Explicit G7 binding validation, never matcher/selector/geometry replay."""
    canonical_bytes(report)
    fields = {"schema", "producer", "status", "task", "source_sha256", "runner_sha256",
              "inputs", "g7", "observer", "notes", "result_sha256"}
    if type(report) is not dict or set(report) != fields or report["schema"] != SCHEMA or report["producer"] != PRODUCER or report["status"] != "complete":
        raise AttritionError("malformed_sidecar", "identity")
    if report["task"] != dict(id=TASK, implementation_sha256=TASK_SHA256) or report["source_sha256"] != SOURCE_PINS:
        raise AttritionError("binding_mismatch", "task_sources")
    if report["runner_sha256"] != g7._file_hash(Path(__file__)) or report["notes"] != NOTES:
        raise AttritionError("binding_mismatch", "runner")
    if report["g7"] != _g7_binding(g7_report):
        raise AttritionError("binding_mismatch", "g7")
    if report["inputs"] != [dict(index=i["index"], sha256=i["sha256"]) for i in g7_report["run"]["inputs"]]:
        raise AttritionError("binding_mismatch", "inputs")
    observer = report["observer"]
    required = {"schema", "status", "items", "bytes", "overflow", "omitted_accounting_events",
                "omitted_proposal_handoffs", "checks", "stages", "proposal_handoffs", "limits"}
    if type(observer) is not dict or set(observer) != required or observer["schema"] != "mc-photogrammetry-scalar-attrition-v1" or observer["status"] != "complete" or observer["overflow"] is not False:
        raise AttritionError("observer_incomplete", "identity")
    if (not _integer(observer["items"]) or not _integer(observer["bytes"])
            or observer["limits"] != dict(items=MAX_RECORDS, bytes=MAX_BYTES)
            or observer["items"] > MAX_RECORDS or observer["bytes"] > MAX_BYTES
            or observer["omitted_accounting_events"] != 0 or observer["omitted_proposal_handoffs"] != 0):
        raise AttritionError("observer_incomplete", "budget")
    if type(observer["checks"]) is not dict or not observer["checks"] or any(value is not True for value in observer["checks"].values()):
        raise AttritionError("observer_incomplete", "handoff_checks")
    stages, frames = observer["stages"], set(report["g7"]["frame_ids"])
    if type(stages) is not dict or not {"seed", "frame"}.issubset(stages):
        raise AttritionError("observer_incomplete", "stages")
    allowed = {"proposal", "reading", "signature", "pair", "seed", "frame", *growth._DETAILED_SCALAR_STAGES}
    for name, stage in stages.items():
        if name not in allowed or type(stage) is not dict or set(stage) != {"expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events", "rows", "terminal_sha256"}:
            raise AttritionError("malformed_sidecar", "stage")
        if not all(_integer(stage[k]) for k in ("expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events")):
            raise AttritionError("malformed_sidecar", "counts")
        if stage["expected"] > MAX_RECORDS or stage["classified"] > MAX_RECORDS:
            raise AttritionError("observer_incomplete", "stage_budget")
        if stage["expected"] != stage["classified"] or any(stage[k] for k in ("unclassified", "unexpected", "multiply_classified", "unexpected_events")) or not g7.selection._hex(stage["terminal_sha256"]):
            raise AttritionError("observer_incomplete", "stage_closure")
        if type(stage["rows"]) is not list or len(stage["rows"]) > MAX_RECORDS:
            raise AttritionError("malformed_sidecar", "rows")
        for row in stage["rows"]:
            keys = {"frame", "axis", "disposition", "count"} | ({"eligible_seed_count"} if name == "seed" else set())
            if type(row) is not dict or set(row) != keys or row["frame"] not in frames or not (row["axis"] is None or type(row["axis"]) is int and row["axis"] in (0, 1, 2)) or type(row["disposition"]) is not str or not _integer(row["count"]) or not 0 < row["count"] <= MAX_RECORDS:
                raise AttritionError("malformed_sidecar", "row")
            if name == "seed":
                metric = row["eligible_seed_count"]
                if row["axis"] is None or row["disposition"] not in ("available", "absent", "not_reached"):
                    raise AttritionError("malformed_sidecar", "seed")
                if metric is not None and (type(metric) is not dict or set(metric) != {"count", "minimum", "maximum", "total"} or not all(_integer(v) for v in metric.values()) or metric["count"] != row["count"] or not metric["minimum"] <= metric["maximum"] or not metric["minimum"] * metric["count"] <= metric["total"] <= metric["maximum"] * metric["count"]):
                    raise AttritionError("malformed_sidecar", "seed_metric")
                if row["disposition"] == "available" and (metric is None or metric["minimum"] == 0) or row["disposition"] == "absent" and (metric is None or metric["total"] != 0):
                    raise AttritionError("malformed_sidecar", "seed_disposition")
        if sum(row["count"] for row in stage["rows"]) != stage["classified"]:
            raise AttritionError("observer_incomplete", "aggregate_closure")
    for frame in frames:
        for axis in range(3):
            if sum(row["count"] for row in stages["seed"]["rows"] if row["frame"] == frame and row["axis"] == axis) != 2:
                raise AttritionError("observer_incomplete", "seed_axes")
    handoffs = observer["proposal_handoffs"]
    if type(handoffs) is not list or len(handoffs) > MAX_RECORDS:
        raise AttritionError("malformed_sidecar", "handoffs")
    generated, returned = [], []
    for row in handoffs:
        if type(row) is not dict or set(row) != {"generated_identity", "returned_identity"}:
            raise AttritionError("malformed_sidecar", "handoff")
        for identity in row.values():
            if type(identity) not in (list, tuple) or len(identity) != 2 or identity[0] not in frames or not _integer(identity[1]):
                raise AttritionError("malformed_sidecar", "handoff_identity")
        if row["generated_identity"][0] != row["returned_identity"][0]:
            raise AttritionError("binding_mismatch", "handoff_frame")
        generated.append(tuple(row["generated_identity"]))
        returned.append(tuple(row["returned_identity"]))
    if len(set(generated)) != len(generated) or len(set(returned)) != len(returned):
        raise AttritionError("binding_mismatch", "handoff_duplicates")
    for frame in frames:
        count = sum(row["count"] for row in stages.get("proposal", {}).get("rows", []) if row["frame"] == frame and row["disposition"] == "returned")
        if set(p for p in returned if p[0] == frame) != {(frame, ordinal) for ordinal in range(count)} or sum(row["count"] for row in stages.get("reading", {}).get("rows", []) if row["frame"] == frame) != count:
            raise AttritionError("observer_incomplete", "proposal_reading_join")
        for axis in (None, 0, 1, 2):
            proposals = sum(row["count"] for row in stages.get("proposal", {}).get("rows", [])
                if row["frame"] == frame and row["axis"] == axis and row["disposition"] == "returned")
            readings = sum(row["count"] for row in stages.get("reading", {}).get("rows", [])
                if row["frame"] == frame and row["axis"] == axis)
            if proposals != readings:
                raise AttritionError("observer_incomplete", "proposal_reading_axis_join")
    minimum_items = sum(s["expected"] + s["classified"] + len(s["rows"]) for s in stages.values()) + len(handoffs)
    if observer["items"] < minimum_items:
        raise AttritionError("observer_incomplete", "item_accounting")
    accepted = sum(row["count"] for row in stages.get("reading", {}).get("rows", []) if row["disposition"] == "accepted")
    if accepted != report["g7"]["counts"]["readings"]:
        raise AttritionError("binding_mismatch", "accepted_readings")
    for row in stages.get("proposal", {}).get("rows", []):
        if not any(seed["frame"] == row["frame"] and seed["axis"] == row["axis"] and seed["disposition"] == "available" for seed in stages["seed"]["rows"]):
            raise AttritionError("binding_mismatch", "seed_proposal_join")
    if _seal(report)["result_sha256"] != report["result_sha256"]:
        raise AttritionError("binding_mismatch", "digest")
    return report


def _readback(path, g7_report):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    _predecode(raw)
    report = json.loads(raw)
    validate_report(report, g7_report)
    if canonical_bytes(report) != raw:
        raise AttritionError("readback_mismatch", "canonical")
    return raw


def publish_report(path, report, g7_report):
    path = Path(path)
    data = canonical_bytes(validate_report(report, g7_report))
    if path.exists():
        raise AttritionError("output_collision", "sidecar")
    temporary, published = None, False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".proposal-attrition-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            if stream.write(data) != len(data):
                raise AttritionError("publication_failed", "short_write")
            stream.flush()
            os.fsync(stream.fileno())
        if _readback(temporary, g7_report) != data:
            raise AttritionError("readback_mismatch", "temporary")
        os.link(temporary, path)
        published = True
        if _readback(path, g7_report) != data:
            raise AttritionError("readback_mismatch", "final")
    except FileExistsError as error:
        raise AttritionError("output_collision", "sidecar") from error
    finally:
        if temporary is not None:
            if published and sys.exc_info()[0] is not None and path.exists() and os.path.samefile(temporary, path):
                path.unlink()
            temporary.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class Outcome:
    g7_outcome: object
    report: dict | None
    diagnostic_error: AttritionError | None
    observer_state: dict


def run_once(images, *, project, destination, diagnostic, sidecar, name, description, delivery=None):
    sidecar = Path(sidecar)
    outputs = tuple(Path(p).resolve() for p in (project, destination, diagnostic, sidecar))
    if len(set(outputs)) != 4 or sidecar.exists():
        raise AttritionError("output_collision", "sidecar")
    bounded = []
    for image in images:
        if len(bounded) == 32:
            raise AttritionError("invalid_input", "image_count")
        bounded.append(Path(image))
    if sidecar.resolve() in (p.resolve() for p in bounded):
        raise AttritionError("output_collision", "sidecar_input")
    if any(g7._file_hash(REPO_ROOT / path) != pin for path, pin in SOURCE_PINS.items()):
        raise AttritionError("source_pin_mismatch", "runtime")
    capture = DeliveryCapture(g7.derive_and_deliver_image_output if delivery is None else delivery)
    observed = g7.run_once(bounded, project=project, destination=destination, diagnostic=diagnostic,
                          name=name, description=description, delivery=capture)
    report, error = None, None
    try:
        if observed.diagnostic_error is not None or observed.production_error is not None:
            raise AttritionError("g7_incomplete", "production_or_diagnostic_error")
        report = build_report(capture, observed.report)
        publish_report(sidecar, report, observed.report)
    except (AttritionError, g7.DiagnosticError, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as failure:
        error = failure if isinstance(failure, AttritionError) else AttritionError("diagnostic_failed", type(failure).__name__)
    state = dict(status="incomplete" if error is not None else "complete",
                 bytes=capture.observer.bytes, items=sum(len(s["expected"]) + len(s["terminal"]) + len(s["counts"]) for s in capture.observer.stages.values()) + len(capture.observer.bindings),
                 omitted=capture.observer.omitted, sidecar_published=error is None)
    return Outcome(observed, report if error is None else None, error, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    for key in ("project", "destination", "diagnostic", "sidecar"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args(argv)
    try:
        outcome = run_once(args.image, project=args.project, destination=args.destination,
            diagnostic=args.diagnostic, sidecar=args.sidecar, name=args.name, description=args.description)
    except (AttritionError, g7.DiagnosticError, OSError) as error:
        print(json.dumps(dict(status="refused", reason=getattr(error, "reason", "preflight_io_failed"), field=getattr(error, "field", type(error).__name__))))
        return 1
    error = outcome.diagnostic_error
    print(json.dumps(dict(status="complete" if error is None else "refused",
        reason="" if error is None else error.reason, field="" if error is None else error.field,
        result_sha256=None if outcome.report is None else outcome.report["result_sha256"],
        observer=outcome.observer_state, production=outcome.g7_outcome.report["run"].get("production", {})), sort_keys=True))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
