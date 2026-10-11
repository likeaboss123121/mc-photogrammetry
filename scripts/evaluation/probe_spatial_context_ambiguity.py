"""Injected paired-experiment coordinator; no CLI, discovery, or native adapters.

All payload access, authorization, process isolation and publication are supplied
by the caller. This module supplies no OS resource enforcement or native launch.
The synthetic task does not authorize constructing those adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Callable

import numpy as np
from scripts.evaluation import spatial_context_evidence as codec

scd = codec.scd
CONTRACT_PIN = codec.CONTRACT_SHA256
FEASIBILITY_PIN = "29e4fb299158377fd791b0f24185dce8c6b6f55df9650c691942a863d58f46fb"
DEPENDENCY_PINS = {
    "descriptor": codec.ACCEPTED_SOURCE_SHA256,
    "evidence": "825d93f8660f34c09a022ea229d9173e83b984e5186b002494848e4c5efa046d",
}
ACCEPTED_SOURCE_PINS = (
    ("scripts/evaluation/spatial_context_descriptor.py", DEPENDENCY_PINS["descriptor"]),
    ("scripts/evaluation/spatial_context_evidence.py", DEPENDENCY_PINS["evidence"]),
)
NATIVE_SHAPE = (900, 1600, 3)
ENCODED_CAP = 8 * 1024 * 1024
DIAGNOSTIC_CAP = 1024 * 1024
EXPECTED_SUPPLY = (0, 37, 2570)
FEASIBILITY_IDENTITIES = ("row176", "row197", "shared-right")
PREPARED_ROLES = (0, 0, 1)


@dataclass(frozen=True)
class Limits:
    wall_seconds: int = 180
    committed_bytes: int = 2 * 1024 ** 3
    log_bytes: int = 65536
    comparisons: int = 1048576
    tuples: int = 4096
    determinants: int = 32768
    footprint_predicates: int = 16800
    footprint_operands: int = 77414400
    extraction_windows: int = 4352
    extraction_pixels: int = 10027008
    gradient_sites: int = 7676928
    derivation_elements: int = 1392640
    distance_matrix_bytes: int = 8388608
    component_contributions: int = 335544320
    evidence_bytes: int = 16777216
    model_bytes: int = 10092544
    publication_bytes: int = 20381696
    aggregate_bytes: int = 37748736
    small_bytes: int = 65536


FROZEN_LIMITS = Limits()
FROZEN_LIMIT_ITEMS = tuple(vars(FROZEN_LIMITS).items())


@dataclass(frozen=True)
class Request:
    """Explicit preregistration metadata, never paths or hidden discovery."""
    bindings: dict
    runtime: dict
    metadata: dict
    row176: dict
    diagnostic_sha256: str
    diagnostic_length: int
    label_associations: list
    feasibility_sha256: str = FEASIBILITY_PIN
    dependency_pins: dict = field(default_factory=lambda: dict(DEPENDENCY_PINS))
    limits: Limits = FROZEN_LIMITS


@dataclass(frozen=True)
class Dependencies:
    """Injected worker-local boundaries.

    in_memory_inventory is retained for constructor compatibility only; the
    coordinator assembles row176 from checked left objects and never calls it.
    """
    capability: object
    in_memory_inventory: Callable
    source_metadata: Callable
    read_payload: Callable
    inspect_diagnostic: Callable
    decode_rgb: Callable
    ingress_digest: Callable
    decode_mask: Callable
    usable_mask: Callable
    prepare_shared_frame: Callable
    extract_prepared_frame: Callable
    match: Callable
    supervisor: object
    publication: object


@dataclass
class Result:
    status: str
    code: str
    consumed: bool = False
    completion: bool = False
    arm_dispositions: tuple = ()
    counters: dict = field(default_factory=dict)
    row176: dict = field(default_factory=lambda: {"status": "untestable_footprint", "sites": 0})
    evidence_receipt: dict | None = None


@dataclass(frozen=True)
class DiagnosticInspection:
    """Parsed, validated identities reported by the injected diagnostic boundary.

    sources is the ordered sealed inventory (sentinel, left, right), including
    physical endpoints, region IDs, and every source/usable-mask identity.
    The boundary must validate the outer/nested seals in the parsed payload.
    """
    payload_sha256: str
    outer_sha256: str
    nested_sha256: str
    sources: tuple


@dataclass(frozen=True)
class StagedTerminal:
    """Worker outcome; completion bytes have not been committed."""
    status: str
    completion_bytes: bytes | None


def require(ok, code):
    codec.require(ok, code)


def check_bound(value, ceiling, code):
    require(type(value) is int and 0 <= value <= ceiling, code)


class Inconclusive(scd.Refusal):
    """Consumed attempt lacking the sealed collective feasibility binding."""


def reservations(counts=EXPECTED_SUPPLY, grid_positions=16800, limits=FROZEN_LIMITS):
    """Metadata-only work/storage reservation, before source or array access."""
    require(type(counts) is tuple and len(counts) == 3
            and all(type(n) is int and n >= 0 for n in counts), "supply_layout")
    scd.check_site_caps(counts[1], counts[2])
    count = counts[1] + counts[2]
    comparison = counts[1] * counts[2]
    operands = grid_positions * 2 * 2304
    values = {"plans": 3, "grid_positions": grid_positions,
        "boolean_operands": operands, "extractions": 2,
        "extraction_windows": count, "extraction_pixels": count * 2304,
        "gradient_sites": count * 9 * 196, "derivation_elements": count * 320,
        "comparisons_per_arm": comparison, "comparisons_total": comparison * 2,
        "distance_matrix_bytes": comparison * 8, "component_contributions": comparison * 320,
        "tuples_per_arm": limits.tuples, "determinants_per_arm": limits.determinants,
        "tuples_total": limits.tuples * 2, "determinants_total": limits.determinants * 2,
        "model_bytes": codec.size_model()["evidence"],
        "publication_bytes": codec.size_model()["aggregate"],
        "wall_seconds": limits.wall_seconds, "committed_bytes": limits.committed_bytes,
        "log_bytes": limits.log_bytes}
    for key, cap in (("grid_positions", limits.footprint_predicates),
            ("boolean_operands", limits.footprint_operands),
            ("extraction_windows", limits.extraction_windows),
            ("extraction_pixels", limits.extraction_pixels), ("gradient_sites", limits.gradient_sites),
            ("derivation_elements", limits.derivation_elements),
            ("comparisons_per_arm", limits.comparisons),
            ("distance_matrix_bytes", limits.distance_matrix_bytes),
            ("component_contributions", limits.component_contributions),
            ("model_bytes", limits.model_bytes), ("model_bytes", limits.evidence_bytes),
            ("publication_bytes", limits.publication_bytes),
            ("publication_bytes", limits.aggregate_bytes)):
        check_bound(values[key], cap, "reservation_" + key)
    return values


def work(work_record, cap):
    """Preserve attempted/charged/processed tuples, including refused batches."""
    codec._work(work_record, cap)
    return copy.deepcopy(work_record)


def arm_work(arm, limits=FROZEN_LIMITS):
    summary = arm["summary"]
    comparison = work(summary["comparison_work"], limits.comparisons)
    witness = summary["witness"]
    node = work(witness["node_work"], limits.tuples)
    determinant = work(witness["determinant_work"], limits.determinants)
    require(determinant["processed"] == 8 * node["processed"], "determinant_node_closure")
    require(determinant["charged"] % 8 == 0 and determinant["requested"] % 8 == 0,
            "determinant_atomic_batch")
    available = math.comb(len(arm["pairs"]), 4)
    require(node["requested"] <= available, "witness_available_nodes")
    status = witness["status"]
    if status == "exhaustive_no_witness":
        require(node["processed"] == available and all(w[k] == 0 for w in (node, determinant)
                for k in ("uncharged", "unprocessed")), "incomplete_exhaustive_witness")
        disposition = "complete_no_witness"
    elif status == "found":
        disposition = "complete_witness"
    else:
        require(status == "budget_inconclusive", "witness_status")
        disposition = "budget_inconclusive"
    if summary["status"] == "insufficient_support":
        disposition = "insufficient_support"
    return disposition, {"comparisons": comparison, "tuples": node, "determinants": determinant}


def bounded_request(request):
    """Close every caller-controlled shape before serialization/copy/equality.

    The codec traversal serializes only scalars; schema checks first ensure
    even those scalars and every container have a fixed, small upper bound.
    The complete envelope shares the existing source-section budget.
    """
    def keys(value, expected, code):
        require(type(value) is dict and len(value) == len(expected), code)
        require(all(type(k) is str and len(k) <= 128 and k in expected
                    for k in value), code)

    def sequence(value, count, code):
        require(type(value) is list and len(value) == count, code)

    def text(value, cap, code):
        require(type(value) is str and 0 < len(value) <= cap and value.isascii(), code)

    def digest(value):
        require(type(value) is str and len(value) == 64, "digest_type")
        codec.digest(value)

    def pins(value):
        sequence(value, 2, "accepted_source_pins")
        for row in value:
            sequence(row, 2, "accepted_source_pins")
            text(row[0], 300, "accepted_source_pins")
            digest(row[1])

    def dependency(value):
        keys(value, ("descriptor", "evidence"), "dependency_pin")
        for item in value.values():
            digest(item)

    def endpoint(value):
        sequence(value, 2, "physical_endpoint")
        text(value[0], 128, "frame_id")
        check_bound(value[1], 65535, "physical_plane_id")

    def source(value):
        keys(value, ("frame_id", "endpoint", "region_id", "encoded_sha256", "encoded_length",
            "decoded_shape", "decoded_dtype", "decoded_rgb_sha256", "ingress_sha256",
            "packed_mask_sha256", "full_native_bool_sha256", "usable_mask_sha256",
            "usable_helper_sha256"), "source_keys")
        text(value["frame_id"], 128, "frame_id")
        endpoint(value["endpoint"])
        sequence(value["decoded_shape"], 3, "native_metadata")
        for dimension in value["decoded_shape"]:
            check_bound(dimension, 1600, "native_metadata")
        text(value["decoded_dtype"], 3, "native_metadata")
        check_bound(value["encoded_length"], ENCODED_CAP, "encoded_length")
        for key in ("encoded_sha256", "decoded_rgb_sha256", "ingress_sha256", "region_id",
                "packed_mask_sha256", "full_native_bool_sha256", "usable_mask_sha256",
                "usable_helper_sha256"):
            digest(value[key])

    def associations(value):
        sequence(value, 3, "label_associations")
        for row in value:
            keys(row, ("label", "frame_id", "endpoint", "region_id"), "label_associations")
            text(row["label"], 128, "label_associations")
            text(row["frame_id"], 128, "frame_id")
            endpoint(row["endpoint"])
            digest(row["region_id"])

    require(type(request) is Request, "request_type")
    keys(vars(request), ("bindings", "runtime", "metadata", "row176", "diagnostic_sha256",
        "diagnostic_length", "label_associations", "feasibility_sha256", "dependency_pins",
        "limits"), "request_keys")
    require(type(request.limits) is Limits, "frozen_limits")
    keys(vars(request.limits), tuple(key for key, _ in FROZEN_LIMIT_ITEMS), "limit_keys")
    for value in vars(request.limits).values():
        check_bound(value, 2 ** 63 - 1, "limit_value")
    dependency(request.dependency_pins)
    digest(request.feasibility_sha256)
    digest(request.diagnostic_sha256)
    check_bound(request.diagnostic_length, DIAGNOSTIC_CAP, "diagnostic_length")
    binding = request.bindings
    keys(binding, ("sources", "source_pins", "diagnostic_outer_sha256", "diagnostic_nested_sha256",
        "contract_sha256", "source_sha256", "runtime_sha256", "authorization_sha256",
        "binding_sha256"), "binding_keys")
    sequence(binding["sources"], 2, "frame_count")
    for item in binding["sources"]:
        source(item)
    pins(binding["source_pins"])
    for key in binding:
        if key not in ("sources", "source_pins"):
            digest(binding[key])
    runtime = request.runtime
    require(type(runtime) is dict and len(runtime) <= len(codec.RUNTIME_KEYS), "runtime_keys")
    require(all(type(k) is str and len(k) <= 128 and k in codec.RUNTIME_KEYS
                for k in runtime), "runtime_keys")
    for key, value in runtime.items():
        text(value, 128, "runtime_value")
        if key.endswith("sha256"):
            digest(value)
    source(request.row176)
    associations(request.label_associations)
    metadata = request.metadata
    keys(metadata, ("expected_supply", "feasibility_identities", "reservation", "dependency_pins",
        "source_pins", "contract_sha256", "source_sha256", "source_inventory", "label_associations",
        "inventory_sha256", "diagnostic_payload_sha256", "diagnostic_outer_sha256",
        "diagnostic_nested_sha256", "runtime_sha256"), "metadata_keys")
    sequence(metadata["expected_supply"], 3, "frozen_supply")
    for count in metadata["expected_supply"]:
        check_bound(count, 4352, "frozen_supply")
    sequence(metadata["feasibility_identities"], 3, "feasibility_order")
    for label in metadata["feasibility_identities"]:
        text(label, 128, "feasibility_order")
    reservation = metadata["reservation"]
    keys(reservation, ("plans", "grid_positions", "boolean_operands", "extractions",
        "extraction_windows", "extraction_pixels", "gradient_sites", "derivation_elements",
        "comparisons_per_arm", "comparisons_total", "distance_matrix_bytes",
        "component_contributions", "tuples_per_arm", "determinants_per_arm", "tuples_total",
        "determinants_total", "model_bytes", "publication_bytes", "wall_seconds",
        "committed_bytes", "log_bytes"), "reservation_keys")
    for value in reservation.values():
        check_bound(value, 2 ** 63 - 1, "reservation_value")
    dependency(metadata["dependency_pins"])
    pins(metadata["source_pins"])
    sequence(metadata["source_inventory"], 3, "sealed_source_inventory")
    for item in metadata["source_inventory"]:
        source(item)
    associations(metadata["label_associations"])
    for key in ("contract_sha256", "source_sha256", "inventory_sha256", "diagnostic_payload_sha256",
            "diagnostic_outer_sha256", "diagnostic_nested_sha256", "runtime_sha256"):
        digest(metadata[key])
    envelope = {**vars(request), "limits": vars(request.limits)}
    codec._bounded_object(envelope, codec.SECTION_CAPS["source"])
    return envelope


def validate_request(request):
    bounded_request(request)
    # These checks inspect caller-supplied metadata only, before any accessor.
    require(request.feasibility_sha256 == FEASIBILITY_PIN, "feasibility_pin")
    require(request.dependency_pins == DEPENDENCY_PINS, "dependency_pin")
    require(request.bindings.get("source_pins") == [list(row) for row in ACCEPTED_SOURCE_PINS],
            "accepted_source_pins")
    require(request.metadata.get("dependency_pins") == request.dependency_pins,
            "metadata_dependency_pins")
    require(request.metadata.get("source_pins") == request.bindings["source_pins"],
            "metadata_source_pins")
    require(request.metadata.get("contract_sha256") == CONTRACT_PIN
            and request.metadata.get("source_sha256") == DEPENDENCY_PINS["descriptor"],
            "metadata_contract_source")
    require(request.bindings.get("contract_sha256") == CONTRACT_PIN, "contract_pin")
    require(request.bindings.get("source_sha256") == DEPENDENCY_PINS["descriptor"], "source_pin")
    require(all(getattr(request.limits, key) == value for key, value in FROZEN_LIMIT_ITEMS),
            "frozen_limits")
    require(codec.ROW_CAPS == {"site": 128, "directed": 256, "pair": 192}, "codec_row_caps")
    require(codec.SECTION_CAPS == {"source": 262144, "summary": 131072,
            "failure": 65536, "framing": 65536}, "codec_section_caps")
    require((codec.MAX_MODEL, codec.MAX_PUBLICATION, codec.EVIDENCE_CAP, codec.AGGREGATE_CAP,
            codec.SMALL_CAP) == (10092544, 20381696, 16777216, 37748736, 65536), "codec_caps")
    require(codec.size_model() == {"evidence": 10092544, "aggregate": 20381696,
            "hard_evidence": 16777216, "hard_aggregate": 37748736}, "codec_model")
    sources = request.bindings.get("sources")
    require(type(sources) is list and len(sources) == 2, "frame_count")
    require(sources[0]["frame_id"] != sources[1]["frame_id"], "frame_identity")
    inventory = request.metadata.get("source_inventory")
    require(type(inventory) is list and inventory == [request.row176, *sources],
            "sealed_source_inventory")
    associations = [{"label": label, "frame_id": source["frame_id"],
                     "endpoint": source["endpoint"], "region_id": source["region_id"]}
                    for label, source in zip(FEASIBILITY_IDENTITIES, inventory)]
    require(type(request.label_associations) is list and request.label_associations == associations
            and request.metadata.get("label_associations") == associations, "label_associations")
    require(request.metadata.get("inventory_sha256") == codec.sha(codec.canonical(
            {"sources": inventory, "label_associations": associations})), "inventory_pin")
    require(len({tuple(s["endpoint"]) for s in inventory}) == 3
            and len({s["region_id"] for s in inventory}) == 3, "inventory_identity")
    require(request.metadata["expected_supply"] == list(EXPECTED_SUPPLY), "frozen_supply")
    require(request.metadata["feasibility_identities"] == list(FEASIBILITY_IDENTITIES),
            "feasibility_order")
    require(request.metadata["reservation"] == reservations(limits=request.limits), "frozen_reservation")
    require(request.row176["frame_id"] == sources[0]["frame_id"], "row176_frame")
    for source in [request.row176, *sources]:
        require(type(source["frame_id"]) is str and source["frame_id"].isascii()
                and 0 < len(source["frame_id"]) <= 128, "frame_id")
        endpoint = source["endpoint"]
        require(type(endpoint) is list and len(endpoint) == 2
                and endpoint[0] == source["frame_id"], "physical_endpoint")
        check_bound(endpoint[1], 65535, "physical_plane_id")
        require(source["decoded_shape"] == list(NATIVE_SHAPE) and source["decoded_dtype"] == "|u1",
                "native_metadata")
        check_bound(source["encoded_length"], ENCODED_CAP, "encoded_length")
        for key in ("encoded_sha256", "decoded_rgb_sha256", "ingress_sha256", "region_id",
                "packed_mask_sha256", "full_native_bool_sha256", "usable_mask_sha256",
                "usable_helper_sha256"):
            codec.digest(source[key])
    require(request.row176["decoded_rgb_sha256"] == sources[0]["decoded_rgb_sha256"]
            and request.row176["ingress_sha256"] == sources[0]["ingress_sha256"], "row176_rgb_identity")
    require(request.row176["usable_mask_sha256"] == sources[0]["usable_mask_sha256"]
            and request.row176["usable_helper_sha256"] == sources[0]["usable_helper_sha256"],
            "row176_usable_identity")
    check_bound(request.diagnostic_length, DIAGNOSTIC_CAP, "diagnostic_length")
    codec.digest(request.diagnostic_sha256)
    # Validate every remaining metadata identity without opening its payload.
    for key in ("diagnostic_outer_sha256", "diagnostic_nested_sha256", "contract_sha256",
                "source_sha256", "runtime_sha256", "authorization_sha256", "binding_sha256"):
        codec.digest(request.bindings[key])
    require(request.bindings["runtime_sha256"] == codec.sha(codec.canonical(request.runtime)),
            "runtime_pin")
    require(request.metadata.get("runtime_sha256") == request.bindings["runtime_sha256"],
            "metadata_runtime_pin")
    require(request.bindings["binding_sha256"] == codec.sha(codec.canonical({k: v for k, v
            in request.bindings.items() if k != "binding_sha256"})), "binding_pin")


def checked_diagnostic(inspection, payload, request):
    require(type(inspection) is DiagnosticInspection, "diagnostic_inspection_type")
    for digest in (inspection.payload_sha256, inspection.outer_sha256, inspection.nested_sha256):
        codec.digest(digest)
    require(inspection.payload_sha256 == request.diagnostic_sha256 == codec.sha(payload)
            == request.metadata.get("diagnostic_payload_sha256"),
            "diagnostic_payload_identity")
    require(inspection.outer_sha256 == request.bindings["diagnostic_outer_sha256"]
            == request.metadata.get("diagnostic_outer_sha256"),
            "diagnostic_outer_identity")
    require(inspection.nested_sha256 == request.bindings["diagnostic_nested_sha256"]
            == request.metadata.get("diagnostic_nested_sha256"),
            "diagnostic_nested_identity")
    require(type(inspection.sources) is tuple
            and list(inspection.sources) == request.metadata["source_inventory"]
            and list(inspection.sources) == [request.row176, *request.bindings["sources"]],
            "diagnostic_source_inventory")


def packed_digest(mask):
    """Explicit native row-major bit packing; never substitutes for bool bytes."""
    return codec.sha(np.packbits(mask.reshape(-1), bitorder="little").tobytes())


def checked_inputs(items, source, dependencies, *, ingress=True):
    require(type(items) is tuple and len(items) == 3, "inventory_tuple")
    pixels, mask, usable = items
    require(isinstance(pixels, np.ndarray) and pixels.shape == NATIVE_SHAPE
            and pixels.dtype == np.dtype("uint8"), "decoded_layout")
    scd.validate_mask(mask, NATIVE_SHAPE[:2], "source_mask")
    scd.validate_mask(usable, NATIVE_SHAPE[:2], "usable_mask")
    require(codec._source_digest(*items) == (source["decoded_rgb_sha256"],
            source["full_native_bool_sha256"], source["usable_mask_sha256"]), "input_digest")
    require(packed_digest(mask) == source["packed_mask_sha256"], "source_mask_digest")
    if ingress:
        require(dependencies.ingress_digest(pixels) == source["ingress_sha256"], "ingress_digest")


def checked_preparations(prepared, arrays, sources, limits):
    """Collective count/work gate; no extraction, derivation, or replanning."""
    counts = tuple(p.plan.eligible_count for p in prepared)
    require(counts == EXPECTED_SUPPLY and all(type(n) is int for n in counts),
            "footprint_supply_mismatch")
    require(tuple(p.role for p in prepared) == PREPARED_ROLES, "prepared_role_order")
    require(len({id(p) for p in prepared}) == 3, "prepared_identity")
    grid = sum(p.plan.grid_centers for p in prepared)
    actual = reservations(counts, grid, limits)
    require(actual == reservations(limits=limits), "actual_reservation")
    for receipt, items, source, role in zip(prepared, arrays, sources, PREPARED_ROLES):
        require(receipt.version == codec.PREPARED_FRAME_VERSION, "prepared_version")
        plan = receipt.plan
        codec._plan_content(plan, role)
        require(all(site.site_id == scd._site_id(plan.frame_sha256, plan.source_mask_sha256,
                plan.usable_mask_sha256, site.x, site.y) for site in plan.sites), "prepared_site_identity")
        require(plan.shape == NATIVE_SHAPE[:2], "prepared_native_shape")
        require((plan.frame_sha256, plan.source_mask_sha256, plan.usable_mask_sha256)
                == (source["decoded_rgb_sha256"], source["full_native_bool_sha256"],
                    source["usable_mask_sha256"]), "prepared_source_binding")
        require(codec._source_digest(*items) == (plan.frame_sha256, plan.source_mask_sha256,
                plan.usable_mask_sha256), "prepared_input_mutation")
    return actual


def check_resource(supervisor, stage, limits):
    telemetry = supervisor.checkpoint(stage)
    require(type(telemetry) is dict and len(telemetry) == 4
            and all(type(k) is str and len(k) <= 32
                    and k in ("status", "wall_seconds", "committed_bytes", "log_bytes")
                    for k in telemetry), "supervisor_telemetry")
    require(type(telemetry.get("status")) is str
            and telemetry["status"] == "running", "supervisor_resource_stop")
    require(type(telemetry.get("wall_seconds")) in (int, float)
            and math.isfinite(telemetry["wall_seconds"])
            and 0 <= telemetry["wall_seconds"] <= limits.wall_seconds, "supervisor_timeout")
    check_bound(telemetry.get("committed_bytes"), limits.committed_bytes, "supervisor_memory")
    check_bound(telemetry.get("log_bytes"), limits.log_bytes, "supervisor_log")


def checked_table(frame, plan, arrays):
    require(type(frame) is codec.SharedFrame and frame.plan is plan, "shared_plan_binding")
    require(type(frame.source_refs) is tuple and len(frame.source_refs) == 3
            and all(a is b for a, b in zip(frame.source_refs, arrays)),
            "shared_source_identity")
    count = len(plan.sites)
    for array, dtype, shape in ((frame.tiles, np.dtype("<f4"), (count, 9, 32)),
            (frame.central_zero, np.dtype(bool), (count,)),
            (frame.neighbor_zero_counts, np.dtype(np.int64), (count,))):
        require(isinstance(array, np.ndarray) and array.dtype == dtype and array.shape == shape,
                "table_layout")
        require(not array.flags.writeable, "mutable_table")
    require(frame.tiles.flags.c_contiguous, "table_contiguous")
    require(codec.sha(frame.tiles.tobytes()) == frame.frozen_sha256, "table_mutation")
    return tuple(codec.sha(a.tobytes()) for a in
                 (frame.tiles, frame.central_zero, frame.neighbor_zero_counts))


def coordinate(request, dependencies):
    """One fake/injected attempt, returning typed refusal/incomplete on any failure.

    Supervisor.run must synchronously supervise evidence publication/readback.
    The coordinator commits completion only after accepting its terminal return.
    The entire coordinator, its receipts, closure and StagedTerminal remain in
    one worker process; none is an IPC/pickle contract. A future native adapter
    must make publication.complete a nonauthoritative worker candidate. Its
    separately reviewed OS controller owns external authoritative completion.
    No OS isolation implementation is provided here.
    publication.evidence receives the accepted codec.publish function explicitly;
    a future separately approved adapter must call it rather than fork its codec.
    """
    result = Result("refused", "not_started")
    snapshots = None
    try:
        validate_request(request)
        require(type(dependencies) is Dependencies, "dependencies_type")
        snapshots = copy.deepcopy(request)
        capability = dependencies.capability
        require(capability.valid(request.bindings["authorization_sha256"]), "authorization_invalid")
        require(capability.consume(request.bindings["authorization_sha256"]), "authorization_reused")
        result.consumed = True
        staged = None
        staged_value = None
        staged_result = None
        commit_postflight = None

        def worker():
            nonlocal staged, staged_value, staged_result, commit_postflight
            limits = request.limits
            check_resource(dependencies.supervisor, "preaccess", limits)
            require(dependencies.source_metadata() == request.metadata, "source_metadata_mismatch")

            def read(kind, key, length, digest):
                payload = dependencies.read_payload(kind, key, length + 1)
                require(type(payload) is bytes and len(payload) == length, "payload_length")
                require(codec.sha(payload) == digest, "payload_digest")
                return payload

            diagnostic = read("diagnostic", "sealed_source", request.diagnostic_length,
                              request.diagnostic_sha256)
            checked_diagnostic(dependencies.inspect_diagnostic(diagnostic), diagnostic, request)
            arrays = []
            for role, source in enumerate(request.bindings["sources"]):
                check_resource(dependencies.supervisor, "decode_" + str(role), limits)
                encoded = read("encoded", source["frame_id"], source["encoded_length"],
                               source["encoded_sha256"])
                pixels = dependencies.decode_rgb(encoded, source["frame_id"])
                require(isinstance(pixels, np.ndarray) and pixels.shape == NATIVE_SHAPE
                        and pixels.dtype == np.dtype("uint8"), "decoded_layout")
                require(codec.sha(pixels.tobytes(order="C")) == source["decoded_rgb_sha256"],
                        "decoded_digest")
                require(dependencies.ingress_digest(pixels) == source["ingress_sha256"],
                        "ingress_digest")
                mask = dependencies.decode_mask(diagnostic, source["region_id"])
                scd.validate_mask(mask, NATIVE_SHAPE[:2], "source_mask")
                require(codec.sha(mask.tobytes()) == source["full_native_bool_sha256"]
                        and packed_digest(mask) == source["packed_mask_sha256"], "source_mask_digest")
                usable = dependencies.usable_mask(pixels, source["usable_helper_sha256"])
                scd.validate_mask(usable, NATIVE_SHAPE[:2], "usable_mask")
                require(codec.sha(usable.tobytes()) == source["usable_mask_sha256"], "usable_mask_digest")
                arrays.append((pixels, mask, usable))
            # One sentinel-mask unpack from the checked diagnostic, after the
            # two matching decodes. Reuse the exact left RGB/usable objects.
            check_resource(dependencies.supervisor, "row176_mask", limits)
            sentinel_mask = dependencies.decode_mask(diagnostic, request.row176["region_id"])
            absent_items = (arrays[0][0], sentinel_mask, arrays[0][2])
            checked_inputs(absent_items, request.row176, dependencies, ingress=False)
            feasibility_arrays = [absent_items, *arrays]
            feasibility_sources = [request.row176, *request.bindings["sources"]]

            # Reserve full stage coverage before the first planning predicate.
            result.counters["footprint"] = {"unit": "full_logical_window_predicate",
                "limit": limits.footprint_predicates, "requested": 16800, "charged": 16800,
                "processed": 0, "uncharged": 0, "unprocessed": 16800,
                "reserved_logical_operands": limits.footprint_operands}
            prepared = []
            try:
                for identity, items, role in zip(FEASIBILITY_IDENTITIES, feasibility_arrays, PREPARED_ROLES):
                    check_resource(dependencies.supervisor, "prepare_" + identity, limits)
                    prepared.append(dependencies.prepare_shared_frame(
                        codec.prepare_shared_frame, *items, role=role))
                    grid = prepared[-1].plan.grid_centers
                    check_bound(grid, 5600, "prepared_grid_work")
                    result.counters["footprint"]["processed"] += grid
                    result.counters["footprint"]["unprocessed"] -= grid
                result.counters["reservation"] = checked_preparations(
                    prepared, feasibility_arrays, feasibility_sources, limits)
            except scd.Refusal as error:
                raise Inconclusive(str(error)) from None
            result.counters["footprint"].update(processed=16800, unprocessed=0)
            plans = [p.plan for p in prepared[1:]]
            bound = codec.size_model()
            check_bound(bound["evidence"], limits.model_bytes, "model_reservation")
            check_bound(bound["aggregate"], limits.publication_bytes, "publication_reservation")
            frames = []
            frozen_buffers = []
            for role, (receipt, plan, items) in enumerate(zip(prepared[1:], plans, arrays)):
                check_resource(dependencies.supervisor, "extract_" + str(role), limits)
                frame = dependencies.extract_prepared_frame(codec.extract_prepared_frame, receipt)
                frozen_buffers.append(checked_table(frame, plan, items))
                frames.append(frame)
            arms = []
            closure = {}; dispositions = []
            result.counters["arms"] = closure
            for name in ("A", "B"):
                check_resource(dependencies.supervisor, "match_" + name, limits)
                for frame, plan, items, frozen in zip(frames, plans, arrays, frozen_buffers):
                    require(checked_table(frame, plan, items) == frozen, "table_metadata_mutation")
                descriptors = [codec.derive(f, name) for f in frames]
                require(all(d.tile_tables is f.tiles for d, f in zip(descriptors, frames)),
                        "shared_table_identity")
                match = None
                try:
                    match = dependencies.match(scd.match_arms, *descriptors,
                        comparison_limit=limits.comparisons, witness_node_limit=limits.tuples,
                        witness_determinant_limit=limits.determinants,
                        source_binding_sha256=scd.source_binding(*plans))
                    check_bound(match.distance_matrix.nbytes, limits.distance_matrix_bytes, "matrix_bytes")
                    arm = codec.compact_match(match, frames, descriptors)
                    disposition, closed = arm_work(arm, limits)
                    arms.append(arm); dispositions.append(disposition); closure[name] = closed
                    result.arm_dispositions = tuple(dispositions)
                finally:
                    # The coordinator retains no matrix through the next arm.
                    if match is not None:
                        match.distance_matrix = None
                    del match, descriptors
                for frame, plan, items, frozen in zip(frames, plans, arrays, frozen_buffers):
                    require(checked_table(frame, plan, items) == frozen, "table_metadata_mutation")
            result.arm_dispositions = tuple(dispositions)
            result.counters["arms"] = closure
            result.counters["combined"] = {}
            for name, cap in (("comparisons", 2097152), ("tuples", 8192), ("determinants", 65536)):
                combined = {key: sum(closure[a][name][key] for a in ("A", "B"))
                            for key in ("limit", "requested", "charged", "processed", "uncharged", "unprocessed")}
                # Requested includes the one refused atomic reservation per arm.
                require(combined["charged"] <= cap and combined["processed"] <= cap, "combined_work_cap")
                result.counters["combined"][name] = combined
            body = {"schema": codec.SCHEMA, "treatment": codec.TREATMENT,
                "bindings": request.bindings, "runtime": request.runtime, "size_bound": bound,
                "frames": [codec.frame_record(f, i) for i, f in enumerate(frames)], "arms": arms,
                "outcomes": [a["summary"]["status"] for a in arms],
                "counters": {"storage_replay": "no_matching_or_common_search"}, "failure": {"code": "none"}}
            record = codec.seal(body)
            wire = codec.dumps(record)  # Accepted complete validation, no matching rerun.
            check_bound(len(wire), limits.model_bytes, "evidence_model_bytes")
            check_bound(len(wire), limits.evidence_bytes, "evidence_hard_bytes")
            bounded_request(request)
            require(request == snapshots and dependencies.source_metadata() == request.metadata,
                    "postflight_metadata")
            for frame, plan, items, frozen in zip(frames, plans, arrays, frozen_buffers):
                require(checked_table(frame, plan, items) == frozen, "table_metadata_mutation")
                require(codec._source_digest(*items) == (plan.frame_sha256, plan.source_mask_sha256,
                        plan.usable_mask_sha256), "postflight_input_mutation")
            checked_inputs(absent_items, request.row176, dependencies, ingress=False)
            check_resource(dependencies.supervisor, "publication", limits)
            current = dependencies.publication.artifact_bytes()
            check_bound(current + 2 * len(wire) + 3 * limits.small_bytes,
                        limits.publication_bytes, "publication_growth")
            check_bound(current + 2 * len(wire) + 3 * limits.small_bytes,
                        limits.aggregate_bytes, "aggregate_growth")
            receipt = dependencies.publication.evidence(codec.publish, record, limits)
            require(receipt["bytes"] == len(wire) and receipt["sha256"] == codec.sha(wire),
                    "publication_receipt")
            check_bound(receipt["aggregate_peak"], limits.publication_bytes, "publication_peak")
            readback = dependencies.publication.readback(len(wire) + 1)
            require(type(readback) is bytes and readback == wire, "publication_readback")
            require(codec.loads(readback) == record, "publication_canonical_readback")
            check_bound(dependencies.publication.artifact_bytes(), limits.aggregate_bytes, "aggregate_postflight")
            result.evidence_receipt = receipt
            if any(d in ("budget_inconclusive", "insufficient_support") for d in dispositions):
                result.status = "inconclusive"; result.code = "arm_inconclusive"
                staged = StagedTerminal("inconclusive", None)
                staged_value = copy.deepcopy(staged)
                staged_result = copy.deepcopy(result)
                return staged
            check_resource(dependencies.supervisor, "completion", limits)
            bounded_request(request)
            require(dependencies.source_metadata() == request.metadata and request == snapshots,
                    "completion_postflight_metadata")
            for frame, plan, items, frozen in zip(frames, plans, arrays, frozen_buffers):
                require(checked_table(frame, plan, items) == frozen, "table_metadata_mutation")
                require(codec._source_digest(*items) == (plan.frame_sha256, plan.source_mask_sha256,
                        plan.usable_mask_sha256), "completion_postflight_input")
            checked_inputs(absent_items, request.row176, dependencies, ingress=False)
            # Canonical readback itself must be included in resource supervision.
            check_resource(dependencies.supervisor, "completion_commit", limits)
            def postflight():
                bounded_request(request)
                require(request == snapshots and dependencies.source_metadata() == request.metadata,
                        "terminal_postflight_metadata")
                for frame, plan, items, frozen in zip(frames, plans, arrays, frozen_buffers):
                    require(checked_table(frame, plan, items) == frozen, "table_metadata_mutation")
                    require(codec._source_digest(*items) == (plan.frame_sha256, plan.source_mask_sha256,
                            plan.usable_mask_sha256), "terminal_postflight_input")
                checked_inputs(absent_items, request.row176, dependencies, ingress=False)
            commit_postflight = postflight
            completion = {"status": "complete", "evidence_sha256": receipt["sha256"],
                          "arm_dispositions": dispositions, "row176": result.row176}
            small = codec.canonical(completion) + b"\n"
            check_bound(len(small), limits.small_bytes, "completion_bytes")
            result.status = "staged_complete"; result.code = "awaiting_supervisor"
            staged = StagedTerminal("staged_complete", small)
            staged_value = copy.deepcopy(staged)
            staged_result = copy.deepcopy(result)
            return staged

        supervised = dependencies.supervisor.run(worker, request.limits)
        require(staged is not None and supervised is staged and type(supervised) is StagedTerminal
                and staged == staged_value and result == staged_result and not result.completion,
                "supervisor_result")
        if staged.status == "inconclusive":
            require(result.status == "inconclusive" and staged.completion_bytes is None,
                    "supervisor_terminal_status")
            return result
        require(staged.status == result.status == "staged_complete"
                and type(staged.completion_bytes) is bytes and commit_postflight is not None,
                "supervisor_terminal_status")
        commit_postflight()
        check_resource(dependencies.supervisor, "terminal_commit", snapshots.limits)
        require(dependencies.publication.complete(staged.completion_bytes, request.limits) is True,
                "completion_publication")
        result.completion = True; result.status = "complete"; result.code = "none"
        return result
    except Exception as error:
        # No arbitrary exception text/stack or environment data escapes to evidence.
        code = str(error) if isinstance(error, scd.Refusal) else "injected_boundary_failure"
        if not (code.isascii() and code.replace("_", "").isalnum() and len(code) <= 128):
            code = "injected_boundary_failure"
        result.status = ("inconclusive" if isinstance(error, Inconclusive)
                         else "incomplete" if result.consumed else "refused")
        result.code = code; result.completion = False
        if result.consumed:
            try:
                dependencies.supervisor.terminate(code)
            except Exception:
                pass
            try:
                failure = codec.canonical({"status": result.status, "code": code, "consumed": True}) + b"\n"
                check_bound(len(failure), FROZEN_LIMITS.small_bytes, "failure_bytes")
                dependencies.publication.failure(failure, FROZEN_LIMITS)
            except Exception:
                pass
        return result
