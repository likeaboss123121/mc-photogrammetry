"""Injected parent/worker protocol. Import performs no launch or native imports.

The OS API must prove suspended assignment and kernel limits before resume.
This module supplies the controller state machine, not an unreviewed OS backend.
The synthetic suite uses fake API implementations exclusively.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import struct
import time

STAGES = ("worker_bootstrap", "preaccess", "decode_0", "decode_1", "row176_mask",
    "prepare_row176", "prepare_row197", "prepare_shared-right", "extract_0", "extract_1",
    "match_A", "match_B", "publication", "completion", "completion_commit", "terminal_commit")
CODES = frozenset(("none", "refused", "incomplete", "inconclusive", "boundary_failure",
    "channel_failure", "resource_limit", "child_failure", "publication_failure", "failure_publication_unavailable"))
MESSAGE_CAP = 4096
CONTROL_CAP = 65536


def io():
    from scripts.evaluation import spatial_context_native_io
    return spatial_context_native_io


class ControlBudget:
    def __init__(self):
        self.bytes = 0

    def charge(self, count):
        io().integer(count, MESSAGE_CAP + 4)
        io().require(self.bytes + count <= CONTROL_CAP, "control_budget")
        self.bytes += count


def work(record, limit):
    m = io()
    m.keys(record, ("limit", "requested", "charged", "processed", "uncharged", "unprocessed"))
    for value in record.values():
        m.integer(value)
    m.require(record["limit"] == limit and record["requested"] == record["charged"] + record["uncharged"]
              and record["charged"] == record["processed"] + record["unprocessed"]
              and record["charged"] <= limit, "work_closure")


def counters(record, *, stage=True):
    if record is None:
        return
    m = io()
    m.keys(record, ("stage", "footprint", "arms") if stage else ("footprint", "arms"))
    if stage:
        m.require(record["stage"] in STAGES, "counter_stage")
    footprint = record["footprint"]
    if footprint is not None:
        m.keys(footprint, ("requested", "charged", "processed", "uncharged", "unprocessed"))
        for value in footprint.values():
            m.integer(value, 16800)
        m.require(footprint["requested"] == footprint["charged"] + footprint["uncharged"]
                  and footprint["charged"] == footprint["processed"] + footprint["unprocessed"], "footprint_closure")
    m.keys(record["arms"], ("A", "B"))
    for arm in record["arms"].values():
        if arm is not None:
            m.keys(arm, ("comparisons", "tuples", "determinants"))
            for key, cap in (("comparisons", 1048576), ("tuples", 4096), ("determinants", 32768)):
                work(arm[key], cap)


def final_counters(result):
    """Project only observed accepted quantities; absent work stays unavailable."""
    m = io()
    raw = getattr(result, "counters", None)
    if raw is None:
        return None
    m.bound_object(raw, cap=MESSAGE_CAP, depth=8, items=512, strings=256)
    m.require(type(raw) is dict, "result_counters")
    if not raw:
        return None
    footprint = raw.get("footprint")
    if footprint is not None:
        footprint = {key:footprint[key] for key in ("requested", "charged", "processed", "uncharged", "unprocessed")}
    arms = raw.get("arms", {})
    m.require(type(arms) is dict and all(name in ("A","B") for name in arms), "result_counter_arms")
    projected = dict(footprint=footprint, arms={name:arms.get(name) for name in ("A","B")})
    counters(projected, stage=False)
    return projected


def monotonic_counters(previous, current):
    counters(current, stage=False)
    if previous is None:
        return current
    if current is None:
        return previous
    m = io()
    def compare(before, after):
        if before is None:
            return
        m.require(after is not None, "counter_availability_regression")
        for key in ("requested", "charged", "processed"):
            m.require(after[key] >= before[key], "counter_regression")
    compare(previous["footprint"], current["footprint"])
    for name in ("A","B"):
        before, after = previous["arms"][name], current["arms"][name]
        if before is not None:
            m.require(after is not None, "counter_availability_regression")
            for key in ("comparisons","tuples","determinants"):
                compare(before[key], after[key])
    return current


def staged_candidate(record):
    m = io()
    m.bound_object(record, cap=65536, depth=8, items=512, strings=256)
    m.keys(record, ("status", "evidence_sha256", "arm_dispositions", "row176", "attempt_id",
                    "manifest_file_sha256", "authorization_file_sha256", "runtime_attestation_sha256"))
    m.require(record["status"] == "complete", "candidate_status")
    m.sequence(record["arm_dispositions"], 2)
    m.require(all(type(value) is str and value in ("complete_witness","complete_no_witness") for value in record["arm_dispositions"]), "candidate_arms")
    m.keys(record["row176"], ("status","sites"))
    m.require(record["row176"]["status"] == "untestable_footprint" and type(record["row176"]["sites"]) is int
              and record["row176"]["sites"] == 0, "candidate_sentinel")
    m.text(record["attempt_id"])
    for key in ("evidence_sha256","manifest_file_sha256","authorization_file_sha256","runtime_attestation_sha256"):
        m.digest(record[key])


def completed_counters(record):
    counters(record, stage=False)
    m = io()
    m.require(record is not None and record["footprint"] is not None
              and all(record["footprint"][key] == 16800 for key in ("requested","charged","processed"))
              and all(record["footprint"][key] == 0 for key in ("uncharged","unprocessed"))
              and all(record["arms"][name] is not None for name in ("A","B")), "complete_counts_unavailable")
    for arm in record["arms"].values():
        m.require(all(work["uncharged"] == work["unprocessed"] == 0 for work in arm.values())
                  and arm["determinants"]["processed"] == 8 * arm["tuples"]["processed"], "complete_arm_work")


def failure_code(error):
    if isinstance(error, io().Refusal):
        value = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str and len(error.args[0]) <= 128 else ""
        if any(word in value for word in ("deadline", "limit", "budget", "cap")):
            return "resource_limit"
        if any(word in value for word in ("channel", "nonce", "control", "counter", "checkpoint")):
            return "channel_failure"
        if any(word in value for word in ("candidate", "publication", "receipt", "readback")):
            return "publication_failure"
        return "refused"
    return "child_failure"


def durable_small(path, data, ledger):
    """One exclusive alias for log/failure; flush and bounded readback required."""
    m = io()
    m.require(type(data) is bytes and len(data) <= 65536, "small_cap")
    ledger.precharge(len(data))
    ledger.small_aliases(1)
    with Path(path).open("xb") as stream:
        m.require(stream.write(data) == len(data), "small_short_write")
        stream.flush()
        os.fsync(stream.fileno())
    with Path(path).open("rb") as stream:
        m.require(stream.read(len(data)+1) == data, "small_readback")
    return dict(bytes=len(data), sha256=m.sha(data)), m.identity(Path(path).lstat())


def failure_record(record):
    m = io()
    m.bound_object(record, cap=65536, depth=8, items=512, strings=256)
    m.keys(record, ("schema","status","code","attempt_id","manifest_file_sha256",
                    "authorization_file_sha256","last_stage","counters","counter_tail",
                    "peak_committed_bytes","sampled_rss","cleanup_failures","publication_state","log"))
    m.require(record["schema"] == "spatial-context-native-failure-v1" and record["status"] == "incomplete"
              and type(record["code"]) is str and record["code"] in CODES
              and record["code"] not in ("none","inconclusive"), "failure_schema")
    m.text(record["attempt_id"])
    for key in ("manifest_file_sha256","authorization_file_sha256"):
        m.digest(record[key])
    m.require(record["last_stage"] is None or record["last_stage"] in STAGES, "failure_stage")
    counters(record["counters"], stage=False)
    m.require(record["counter_tail"] in ("unknown","observed_terminal")
              and (record["counter_tail"] != "observed_terminal" or record["counters"] is not None), "failure_counter_tail")
    if record["peak_committed_bytes"] is not None:
        m.integer(record["peak_committed_bytes"])
    m.require(record["sampled_rss"] is None, "failure_rss_unavailable")
    m.require(type(record["cleanup_failures"]) is list and len(record["cleanup_failures"]) <= 2
              and all(type(value) is str and value in ("terminate","close") for value in record["cleanup_failures"])
              and len(set(record["cleanup_failures"])) == len(record["cleanup_failures"]), "failure_cleanup")
    m.keys(record["publication_state"], ("completion_present","temporary_present","log_present","failure_present"))
    m.require(all(type(value) is bool for value in record["publication_state"].values()), "failure_publication_state")
    if record["log"] is not None:
        m.keys(record["log"], ("bytes","sha256","retired"))
        m.integer(record["log"]["bytes"], 65536)
        m.digest(record["log"]["sha256"])
        m.require(type(record["log"]["retired"]) is bool, "failure_log_summary")


def message_shape(kind, payload):
    m = io()
    if kind == "hello":
        m.keys(payload, ("manifest_file_sha256", "authorization_file_sha256", "nonce"))
        for value in payload.values():
            m.digest(value)
    elif kind == "ack":
        m.keys(payload, ("accepted", "deadline_ms"))
        m.require(payload["accepted"] is True and type(payload["deadline_ms"]) is int
                  and payload["deadline_ms"] == 180000, "ack")
    elif kind == "checkpoint":
        m.keys(payload, ("stage",))
        m.require(payload["stage"] in STAGES, "stage")
    elif kind == "telemetry":
        m.keys(payload, ("stage", "status", "elapsed_ms", "committed_bytes", "log_bytes"))
        m.require(payload["stage"] in STAGES and payload["status"] in ("running", "terminated", "limit"), "telemetry")
        for key in ("elapsed_ms", "committed_bytes", "log_bytes"):
            m.integer(payload[key])
    elif kind == "counters":
        counters(payload)
    elif kind == "terminal":
        m.keys(payload, ("status", "code", "worker_exit_intent", "evidence", "candidate", "counters"))
        m.require(type(payload["status"]) is str and payload["status"] in ("staged_complete", "inconclusive", "incomplete")
                  and type(payload["code"]) is str and payload["code"] in CODES, "terminal_status")
        intent = {"staged_complete": 0, "inconclusive": 2, "incomplete": 1}[payload["status"]]
        m.require(type(payload["worker_exit_intent"]) is int and payload["worker_exit_intent"] == intent, "terminal_intent")
        m.require(payload["code"] == "none" if intent == 0 else payload["code"] == "inconclusive" if intent == 2
                  else payload["code"] not in ("none","inconclusive"), "terminal_code")
        for key, cap in (("evidence", 10092544), ("candidate", 65536)):
            if payload[key] is not None:
                m.keys(payload[key], ("bytes", "sha256"))
                m.integer(payload[key]["bytes"], cap, 1)
                m.digest(payload[key]["sha256"])
        m.require((payload["candidate"] is not None) == (payload["status"] == "staged_complete"), "terminal_candidate")
        counters(payload["counters"], stage=False)
    else:
        raise m.Refusal("message_type")


class Channel:
    """Strict complete frames; transport owns bounded incremental reads."""
    def __init__(self, attempt_id, budget=None):
        io().text(attempt_id)
        self.attempt_id = attempt_id
        self.budget = budget or ControlBudget()
        self.sent = self.received = 0

    def encode(self, kind, payload):
        m = io()
        m.bound_object(payload, cap=MESSAGE_CAP, depth=8, items=512, strings=256)
        message_shape(kind, payload)
        data = m.canonical(dict(schema="spatial-context-control-v1", attempt_id=self.attempt_id,
                                sequence=self.sent, type=kind, payload=payload), MESSAGE_CAP)
        self.budget.charge(4 + len(data))
        self.sent += 1
        return struct.pack("<I", len(data)) + data

    def decode(self, frame):
        m = io()
        m.require(type(frame) is bytes and 4 <= len(frame) <= MESSAGE_CAP + 4, "channel_frame")
        length = struct.unpack_from("<I", frame)[0]
        m.require(0 < length <= MESSAGE_CAP and len(frame) == length + 4, "channel_length")
        self.budget.charge(len(frame))
        record = m.parse(frame[4:], cap=MESSAGE_CAP, depth=8, items=512, strings=256)
        m.keys(record, ("schema", "attempt_id", "sequence", "type", "payload"))
        m.require(record["schema"] == "spatial-context-control-v1" and record["attempt_id"] == self.attempt_id
                  and type(record["sequence"]) is int and record["sequence"] == self.received, "channel_sequence")
        message_shape(record["type"], record["payload"])
        m.require(m.canonical(record, MESSAGE_CAP) == frame[4:], "channel_canonical")
        self.received += 1
        return record["type"], record["payload"]


def authorize(data, manifest, command_sha256):
    m = io()
    record = m.parse(data, cap=65536)
    m.keys(record, ("schema", "attempt_id", "manifest_file_sha256", "command_sha256", "review_receipts", "single_use", "disposition", "authorization_sha256"))
    m.require(record["schema"] == "spatial-context-native-auth-v1"
              and record["attempt_id"] == manifest.record["attempt_id"]
              and record["manifest_file_sha256"] == manifest.file_sha256
              and record["command_sha256"] == command_sha256
              and record["single_use"] is True and record["disposition"] == "AUTHORIZE_ONE_SPATIAL_CONTEXT_RUN", "authorization")
    m.sequence(record["review_receipts"], 2)
    for pin, reviewer in zip(record["review_receipts"], ("Luna", "Astra")):
        m.keys(pin, ("reviewer", "path", "sha256"))
        m.require(pin["reviewer"] == reviewer, "review_order")
        m.windows_path(pin["path"])
        m.digest(pin["sha256"])
    m.sealed(record, "authorization_sha256", 65536)
    m.require(m.canonical(record, 65536) + b"\n" == data, "auth_canonical")
    return record, m.sha(data)


class Ledger:
    def __init__(self, directory):
        self.directory = Path(directory)

    def current(self):
        total = 0
        for path in self.directory.iterdir():
            total += io().identity(path.lstat())[2]
        return total

    def precharge(self, added):
        m = io()
        m.integer(added)
        total = self.current() + added
        m.require(total <= m.LIMITS["publication_bytes"] and total <= m.LIMITS["aggregate_bytes"], "artifact_budget")
        return total

    def small_aliases(self, added):
        names = {"worker-staged-candidate.json", "completion.json", "failure.json", "worker.log"}
        count = sum(path.name in names or path.name in {"."+name+".partial" for name in names}
                    for path in self.directory.iterdir())
        io().require(count + added <= 3, "small_alias_budget")


def exclusive_bytes(path, wire, cap, ledger=None, *, final=False, validate=None):
    """Exclusive hard-link commit with checked readback and rollback of ours."""
    m = io()
    m.require(type(wire) is bytes and 0 < len(wire) <= cap, "receipt_bytes")
    destination = Path(path)
    temporary = destination.with_name("." + destination.name + ".partial")
    m.require(not destination.exists() and not temporary.exists(), "publication_collision")
    if ledger is not None:
        ledger.precharge(2 * len(wire))
        ledger.small_aliases(2)
    created = linked = False
    try:
        with temporary.open("xb") as stream:
            created = True
            m.require(stream.write(wire) == len(wire), "short_write")
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            m.require(stream.read(len(wire) + 1) == wire, "temporary_readback")
        if validate is not None:
            validate()
        os.link(temporary, destination)
        linked = True
        with destination.open("rb") as stream:
            m.require(stream.read(len(wire) + 1) == wire, "final_readback")
        if validate is not None:
            validate()
        temporary.unlink()  # A final cleanup failure enters rollback below.
        created = False
    except Exception:
        cleanup_failed = False
        if final and linked:
            try:
                destination.unlink()
            except Exception:
                cleanup_failed = True
        if created:
            try:
                temporary.unlink()
            except Exception:
                cleanup_failed = True
        if cleanup_failed:
            raise m.Refusal("publication_cleanup_failure") from None
        raise


class WorkerGrant:
    """Private transport proof, consumed once; no argv/environment credential."""
    def __init__(self, proof, authorization_sha256):
        io().require(proof is not None and proof.authenticated and proof.inherited and proof.job_enforced, "private_worker_proof")
        io().digest(authorization_sha256)
        self.proof, self.authorization_sha256 = proof, authorization_sha256
        self.active = False
        self.job_enforced = proof.job_enforced

    def valid(self, identity):
        return identity == self.authorization_sha256 and not self.active and self.proof.authenticated

    def consume(self, identity):
        if not self.valid(identity):
            return False
        self.active = True
        return True


class WorkerSupervisor:
    def __init__(self, exchange):
        self.exchange, self.stage = exchange, -1

    def run(self, callback, limits):
        return callback()  # Local closure/receipt/terminal identity never crosses IPC.

    def checkpoint(self, stage):
        m = io()
        index = STAGES.index(stage)
        m.require(index == self.stage + 1, "checkpoint_order")
        response = self.exchange("checkpoint", dict(stage=stage))
        message_shape("telemetry", response)
        m.require(response["stage"] == stage, "stale_telemetry")
        m.require(response["status"] == "running" and response["elapsed_ms"] <= 180000
                  and response["committed_bytes"] <= 2147483648 and response["log_bytes"] <= 65536,
                  "worker_resource_limit")
        self.stage = index
        return dict(status="running" if response["status"] == "running" else "terminated",
                    wall_seconds=response["elapsed_ms"] / 1000,
                    committed_bytes=response["committed_bytes"], log_bytes=response["log_bytes"])

    def terminate(self, code):
        # Actual parent watchdog/Job API owns termination, even if worker blocks.
        self.stopped = True  # worker() emits the sole typed terminal afterward.


class WorkerPublication:
    def __init__(self, manifest, services, authorization_sha256):
        self.manifest, self.services = manifest, services
        self.artifacts = manifest.record["artifacts"]
        self.ledger = Ledger(self.artifacts["output_directory"])
        self.authorization_sha256 = authorization_sha256

    def artifact_bytes(self):
        return self.ledger.current()

    def evidence(self, accepted, record, limits):
        m = io()
        wire = self.services.driver.codec.dumps(record)
        m.require(len(wire) <= m.LIMITS["model_bytes"], "evidence_model")
        self.ledger.precharge(2 * len(wire) + 3 * 65536)
        return accepted(self.artifacts["evidence"], record)

    def readback(self, limit):
        with Path(self.artifacts["evidence"]).open("rb") as stream:
            return stream.read(limit)

    def complete(self, wire, limits):
        record = io().parse(wire, cap=65536)
        record.update(attempt_id=self.manifest.record["attempt_id"], manifest_file_sha256=self.manifest.file_sha256,
                      authorization_file_sha256=self.authorization_sha256,
                      runtime_attestation_sha256=self.manifest.record["runtime"]["attestation"]["sha256"])
        staged_candidate(record)
        candidate = io().canonical(record, 65536) + b"\n"
        exclusive_bytes(self.artifacts["worker_candidate"], candidate, 65536, self.ledger)
        return True

    def failure(self, wire, limits):
        # Parent alone publishes failure; worker failure remains a typed terminal.
        return None


def _worker_run(manifest, grant, exchange, loader, state):
    """Entry requires private authenticated proof; no CLI flag can supply it."""
    m = io()
    m.require(type(grant) is WorkerGrant and grant.valid(grant.authorization_sha256), "worker_grant")
    grant.consume(grant.authorization_sha256)
    supervisor = WorkerSupervisor(exchange)
    supervisor.checkpoint("worker_bootstrap")
    services = (loader or m.lazy_services)(manifest, grant)
    driver = services.driver
    template = manifest.record["request_template"]
    binding = {**template["bindings"], "authorization_sha256": grant.authorization_sha256}
    binding["binding_sha256"] = driver.codec.sha(driver.codec.canonical(binding))
    request = driver.Request(**{**template, "bindings": binding, "limits": driver.Limits(**template["limits"])})
    driver.validate_request(request)
    # The coordinator retains its one-shot gate; private consumption is already
    # fixed above, and this local gate cannot be reused for a second coordinator.
    class LocalCapability:
        used = False
        def valid(self, identity):
            return grant.active and identity == grant.authorization_sha256 and not self.used
        def consume(self, identity):
            if not self.valid(identity):
                return False
            self.used = True
            return True
    readers = m.FixedReaders(manifest, grant)
    diagnostic = m.DiagnosticAdapter(manifest, services)
    rgb = m.RGBAdapter(manifest, services)
    publication = WorkerPublication(manifest, services, grant.authorization_sha256)
    def no_inventory():
        raise m.Refusal("preload_forbidden")
    dependencies = driver.Dependencies(LocalCapability(), no_inventory,
        lambda: request.metadata, readers.read_payload, diagnostic.inspect,
        rgb.decode_rgb, rgb.ingress_digest, diagnostic.decode_mask, rgb.usable_mask,
        lambda accepted,*args,**kwargs: accepted(*args,**kwargs),
        lambda accepted,*args,**kwargs: accepted(*args,**kwargs),
        lambda accepted,*args,**kwargs: accepted(*args,**kwargs), supervisor, publication)
    result = driver.coordinate(request, dependencies)
    state["result"] = result
    status = "staged_complete" if result.completion else "inconclusive" if result.status == "inconclusive" else "incomplete"
    candidate = None
    if result.completion:
        pin = Path(publication.artifacts["worker_candidate"])
        with pin.open("rb") as stream:
            data = stream.read(m.SMALL_CAP+1)
        m.require(len(data) <= m.SMALL_CAP, "candidate_cap")
        candidate = dict(bytes=len(data), sha256=m.sha(data))
    terminal = dict(status=status, code="none" if result.completion else status,
                    worker_exit_intent=0 if result.completion else 2 if status == "inconclusive" else 1,
                    evidence={k:result.evidence_receipt[k] for k in ("bytes", "sha256")} if result.evidence_receipt else None,
                    candidate=candidate, counters=final_counters(result))
    message_shape("terminal", terminal)
    state["terminal_attempted"] = True
    exchange("terminal", terminal)
    return terminal["worker_exit_intent"]


def worker(manifest, grant, exchange, loader=None):
    state = {}
    try:
        return _worker_run(manifest, grant, exchange, loader, state)
    except Exception as error:
        if not state.get("terminal_attempted"):
            try:
                observed = final_counters(state.get("result"))
            except Exception:
                observed = None
            terminal = dict(status="incomplete", code=failure_code(error), worker_exit_intent=1,
                            evidence=None, candidate=None, counters=observed)
            try:
                message_shape("terminal", terminal)
                exchange("terminal", terminal)
            except Exception:
                pass  # Broken private channel leaves the parent an unknown tail.
        return 1


class Controller:
    """OS API protocol requires real proof; fake tests prove ordering, not OS limits.

    API: private_channel(nonce), create_job(), create_suspended(launch,channel),
    attach(job,child), enforcement(job,child), resume(child), next_event(child,
    max_bytes), send(channel,frame), telemetry(job), terminate(job), close(job).
    An event is (stdout|stderr|control|exit|idle, bounded bytes or integer exit).
    """
    def __init__(self, manifest, api, clock=time.monotonic):
        self.manifest, self.api, self.clock = manifest, api, clock
        self.started = clock()
        self.log = bytearray()
        self.terminal = None
        self.last_counters = None
        self.last_stage = None
        self.last_elapsed = 0
        self.peak_commit = None
        self.next_stage = 0
        self.job = self.child = None
        self.descendants = None
        self.log_receipt = self.log_identity = None
        self.log_retired = False

    def telemetry(self, stage):
        m = io()
        record = self.api.telemetry(self.job)
        m.keys(record, ("status", "committed_bytes", "peak_committed_bytes", "descendants"))
        for key in ("committed_bytes", "peak_committed_bytes", "descendants"):
            m.integer(record[key])
        elapsed = self.clock() - self.started
        m.require(type(elapsed) in (int, float) and 0 <= elapsed <= 180, "deadline")
        m.require(record["status"] == "running" and record["committed_bytes"] <= 2147483648
                  and record["peak_committed_bytes"] <= 2147483648, "job_limit")
        m.require(len(self.log) <= 65536, "log_limit")
        self.peak_commit = record["peak_committed_bytes"]
        self.descendants = record["descendants"]
        return dict(stage=stage, status="running", elapsed_ms=int(elapsed * 1000),
                    committed_bytes=record["committed_bytes"], log_bytes=len(self.log))

    def publication_state(self):
        artifacts = self.manifest.record["artifacts"]
        return dict(completion_present=Path(artifacts["completion"]).exists(),
                    temporary_present=Path(artifacts["completion"]).with_name(".completion.json.partial").exists(),
                    log_present=Path(artifacts["log"]).exists(), failure_present=Path(artifacts["failure"]).exists())

    def retire_owned_log(self):
        m = io()
        m.require(self.log_receipt is not None and self.log_identity is not None, "log_ownership_unavailable")
        path = Path(self.manifest.record["artifacts"]["log"])
        m.no_reparse(path)
        m.require(m.identity(path.lstat()) == self.log_identity, "log_retirement_identity")
        with path.open("rb") as stream:
            m.require(m.identity(os.fstat(stream.fileno())) == self.log_identity, "log_retirement_handle")
            wire = stream.read(65537)
            m.require(m.identity(os.fstat(stream.fileno())) == self.log_identity, "log_retirement_mutation")
        m.require(m.identity(path.lstat()) == self.log_identity and len(wire) == self.log_receipt["bytes"]
                  and m.sha(wire) == self.log_receipt["sha256"], "log_retirement_pin")
        path.unlink()
        self.log_retired = True

    def run(self, authorization_bytes):
        m = io()
        launch = self.manifest.record["launch"]
        _, authorization_sha = authorize(authorization_bytes, self.manifest, m.sha(m.canonical(launch)))
        artifacts = self.manifest.record["artifacts"]
        output = Path(artifacts["output_directory"])
        m.require(not output.exists(), "output_collision")
        # Durable consume before any child capable of opening input payloads.
        marker = m.canonical(dict(schema="spatial-context-native-consumed-v1",
            attempt_id=self.manifest.record["attempt_id"], manifest_file_sha256=self.manifest.file_sha256,
            authorization_file_sha256=authorization_sha), 65536) + b"\n"
        with Path(artifacts["consumed_path"]).open("xb") as stream:
            m.require(stream.write(marker) == len(marker), "consume_short_write")
            stream.flush()
            os.fsync(stream.fileno())
        with Path(artifacts["consumed_path"]).open("rb") as stream:
            readback = stream.read(m.SMALL_CAP+1)
        m.require(readback == marker, "consume_readback")
        output.mkdir(exist_ok=False)
        nonce = secrets.token_hex(32)
        budget = ControlBudget()
        channel = Channel(self.manifest.record["attempt_id"], budget)
        hello = False
        try:
            private = self.api.private_channel(nonce)
            self.job = self.api.create_job(dict(committed_bytes=2147483648, kill_on_close=True,
                                                 no_breakaway=True, descendant_containment=True))
            self.child = self.api.create_suspended(launch, private)
            self.api.attach(self.job, self.child)
            proof = self.api.enforcement(self.job, self.child)
            m.keys(proof, ("suspended", "attached", "committed_bytes", "kill_on_close", "no_breakaway", "descendant_containment", "private_inherited"))
            m.require(proof == dict(suspended=True, attached=True, committed_bytes=2147483648,
                kill_on_close=True, no_breakaway=True, descendant_containment=True, private_inherited=True), "job_enforcement")
            self.telemetry("worker_bootstrap")
            self.api.resume(self.child)
            exit_code = None
            while exit_code is None:
                self.telemetry(self.last_stage or "worker_bootstrap")
                kind, value = self.api.next_event(self.child, MESSAGE_CAP + 4)
                if kind in ("stdout", "stderr"):
                    m.require(type(value) is bytes and len(value) <= MESSAGE_CAP + 4
                              and len(self.log) + len(value) <= 65536, "log_limit")
                    self.log.extend(value)
                elif kind == "control":
                    name, payload = channel.decode(value)
                    if not hello:
                        m.require(name == "hello" and payload == dict(manifest_file_sha256=self.manifest.file_sha256,
                            authorization_file_sha256=authorization_sha, nonce=nonce), "private_hello")
                        self.api.send(private, channel.encode("ack", dict(accepted=True, deadline_ms=180000)))
                        hello = True
                    elif name == "checkpoint":
                        m.require(self.terminal is None and self.next_stage < len(STAGES)
                                  and payload["stage"] == STAGES[self.next_stage], "checkpoint_order")
                        self.next_stage += 1
                        self.last_stage = payload["stage"]
                        response = self.telemetry(payload["stage"])
                        self.last_elapsed = response["elapsed_ms"]
                        self.api.send(private, channel.encode("telemetry", response))
                    elif name == "counters":
                        m.require(self.terminal is None and payload["stage"] == self.last_stage, "counter_stage")
                        observed = {key:payload[key] for key in ("footprint","arms")}
                        self.last_counters = monotonic_counters(self.last_counters, observed)
                    elif name == "terminal":
                        m.require(self.terminal is None, "duplicate_terminal")
                        self.last_counters = monotonic_counters(self.last_counters, payload["counters"])
                        self.terminal = payload
                    else:
                        raise m.Refusal("control_order")
                elif kind == "exit":
                    m.require(type(value) is int, "exit_type")
                    exit_code = value
                else:
                    m.require(kind == "idle" and value is None, "event_type")
            self.telemetry("terminal_commit")
            m.require(hello and self.terminal is not None and exit_code == 0
                      and self.terminal["status"] == "staged_complete"
                      and self.terminal["worker_exit_intent"] == 0 and self.next_stage == len(STAGES)
                      and self.descendants == 0, "child_terminal")
            terminal = self.terminal
            m.require(terminal["evidence"] is not None and terminal["candidate"] is not None, "terminal_artifacts")
            ledger = Ledger(output)
            evidence = m.read_fixed(dict(path=artifacts["evidence"], **terminal["evidence"]), 10092544)
            candidate = m.read_fixed(dict(path=artifacts["worker_candidate"], **terminal["candidate"]), 65536)
            record = m.parse(candidate, cap=65536)
            staged_candidate(record)
            completed_counters(terminal["counters"])
            m.require(record.get("attempt_id") == self.manifest.record["attempt_id"]
                      and record.get("manifest_file_sha256") == self.manifest.file_sha256
                      and record.get("authorization_file_sha256") == authorization_sha
                      and record.get("runtime_attestation_sha256") == self.manifest.record["runtime"]["attestation"]["sha256"]
                      and record.get("evidence_sha256") == m.sha(evidence), "candidate_join")
            # Transition from candidate+log to final temporary/final+log, never
            # retaining four small aliases; remove only this verified candidate.
            Path(artifacts["worker_candidate"]).unlink()
            receipt = m.canonical(dict(schema="spatial-context-native-completion-v1", status="complete",
                attempt_id=self.manifest.record["attempt_id"], manifest_file_sha256=self.manifest.file_sha256,
                authorization_file_sha256=authorization_sha, evidence=terminal["evidence"], candidate=terminal["candidate"],
                runtime_attestation_sha256=self.manifest.record["runtime"]["attestation"]["sha256"],
                worker_exit=exit_code, last_stage=self.last_stage, peak_committed_bytes=self.peak_commit,
                counters=terminal["counters"]), 65536) + b"\n"
            self.log_receipt, self.log_identity = durable_small(artifacts["log"], bytes(self.log), ledger)
            checks = 0
            def commit_check():
                nonlocal checks
                self.telemetry("terminal_commit")
                checks += 1
                if checks == 2:
                    self.api.close(self.job)
                    self.job = None
                    m.require(0 <= self.clock()-self.started <= 180, "deadline")
            exclusive_bytes(artifacts["completion"], receipt, 65536, ledger, final=True,
                            validate=commit_check)
            return dict(status="complete", completion=True)
        except Exception as error:
            code = failure_code(error)
            cleanup = []
            if self.job is not None:
                for action in ("terminate","close"):
                    try:
                        getattr(self.api, action)(self.job)
                    except Exception:
                        cleanup.append(action)
                self.job = None
            ledger = Ledger(output)
            try:
                if not Path(artifacts["log"]).exists():
                    self.log_receipt, self.log_identity = durable_small(artifacts["log"], bytes(self.log), ledger)
            except Exception:
                code = "publication_failure"
            tail = "observed_terminal" if self.terminal is not None and self.terminal["counters"] is not None else "unknown"
            publication = "unavailable"
            try:
                try:
                    ledger.small_aliases(1)
                except m.Refusal:
                    self.retire_owned_log()
                    ledger.small_aliases(1)
                state = self.publication_state()
                state["failure_present"] = True  # The receipt describes its committed state.
                summary = {**self.log_receipt, "retired":self.log_retired} if self.log_receipt is not None else None
                failure = dict(schema="spatial-context-native-failure-v1",status="incomplete",code=code,
                    attempt_id=self.manifest.record["attempt_id"],manifest_file_sha256=self.manifest.file_sha256,
                    authorization_file_sha256=authorization_sha,last_stage=self.last_stage,counters=self.last_counters,
                    counter_tail=tail,peak_committed_bytes=self.peak_commit,sampled_rss=None,cleanup_failures=cleanup,
                    publication_state=state,log=summary)
                failure_record(failure)
                durable_small(artifacts["failure"], m.canonical(failure,65536)+b"\n", ledger)
                publication = "published"
            except Exception:
                code = "failure_publication_unavailable"
            summary = {**self.log_receipt, "retired":self.log_retired} if self.log_receipt is not None else None
            return dict(status="incomplete", completion=False, code=code, counters=self.last_counters,
                        last_stage=self.last_stage, counter_tail=tail, sampled_rss=None,
                        failure_publication=publication, publication_state=self.publication_state(), log=summary)



def main(argv=None, *, api=None):
    """No worker flag, hidden defaults, environment capability, or OS fallback."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("manifest_sha256")
    parser.add_argument("authorization")
    arguments = parser.parse_args(argv)
    m = io()
    m.require(api is not None, "os_api_required")
    m.windows_path(arguments.manifest)
    m.windows_path(arguments.authorization)
    m.digest(arguments.manifest_sha256)
    def metadata(path, cap):
        m.no_reparse(path)
        with Path(path).open("rb") as stream:
            data = stream.read(cap + 1)
        m.require(len(data) <= cap, "metadata_cap")
        return data
    manifest = m.Manifest(metadata(arguments.manifest, 262144), arguments.manifest_sha256)
    m.require(arguments.authorization == manifest.record["artifacts"]["authorization_path"], "authorization_path")
    result = Controller(manifest, api).run(metadata(arguments.authorization, 65536))
    return 0 if result["completion"] else 1
