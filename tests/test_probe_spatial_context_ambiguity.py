"""Only authored in-memory arrays and deterministic injected boundaries.

No input paths, production imports, real capabilities, OS workers, or native code.
"""
from dataclasses import replace
import ast
import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import probe_spatial_context_ambiguity as driver

ev = driver.codec
scd = driver.scd
WORKER_CANDIDATE = "worker-staged-candidate.json"
PUBLIC_COMPLETION = "completion.json"


def authored_arrays(count):
    pixels = np.zeros(driver.NATIVE_SHAPE, np.uint8)
    mask = np.zeros(driver.NATIVE_SHAPE[:2], bool)
    usable = np.ones(driver.NATIVE_SHAPE[:2], bool)
    if count:
        columns = min(95, count)
        rows, extra = divmod(count, columns)
        height = 48 + 16 * (rows - 1)
        mask[:height, :48 + 16 * (columns - 1)] = True
        if extra:
            mask[height:height + 16, :48 + 16 * (extra - 1)] = True
    return pixels, mask, usable


class FakeCapability:
    def __init__(self, harness):
        self.h = harness
        self.consumed = False
        self.invalid = False

    def valid(self, identity):
        self.h.events.append(("valid", identity))
        return not self.invalid

    def consume(self, identity):
        self.h.events.append(("consume", identity))
        if self.consumed:
            return False
        self.consumed = True
        return True


class FakeSupervisor:
    """Synchronous only: represents ceilings, makes no OS isolation claim."""
    def __init__(self, harness):
        self.h = harness
        self.stopped = []
        self.telemetry = {}
        self.hooks = {}
        self.after_callback = None

    def run(self, callback, limits):
        self.h.access("supervisor_run")
        assert limits.wall_seconds == 180 and limits.committed_bytes == 2 * 1024 ** 3
        terminal = callback()
        self.h.access("supervisor_terminal", terminal.status)
        assert WORKER_CANDIDATE not in self.h.publication.artifacts
        return self.after_callback(terminal) if self.after_callback else terminal

    def checkpoint(self, stage):
        self.h.access("checkpoint", stage)
        if stage in self.hooks:
            self.hooks[stage]()
        return {"status": "running", "wall_seconds": 1,
                "committed_bytes": 1024, "log_bytes": 0, **self.telemetry.get(stage, {})}

    def terminate(self, code):
        self.h.access("terminate", code)
        self.stopped.append(code)


class FakePublication:
    """Worker-local candidate transaction; it has no external completion authority."""
    def __init__(self, harness):
        self.h = harness
        self.artifacts = {}
        self.partial = None
        self.mode = None
        self.current = None
        self.read_limit = None
        self.failures = []

    def artifact_bytes(self):
        self.h.access("artifact_bytes")
        return sum(map(len, self.artifacts.values())) if self.current is None else self.current

    def evidence(self, accepted_publish, record, limits):
        self.h.access("evidence")
        assert accepted_publish is ev.publish and limits == driver.FROZEN_LIMITS
        if "evidence" in self.artifacts or self.partial is not None:
            raise scd.Refusal("publication_collision")
        wire = ev.dumps(record)
        old = dict(self.artifacts)
        try:
            self.partial = wire
            if self.mode in ("short_write", "flush", "fsync", "link", "crash"):
                raise scd.Refusal("publication_" + self.mode)
            if self.mode == "atomic_readback":
                raise scd.Refusal("publication_readback")
            self.artifacts["evidence"] = wire
            return {"bytes": len(wire), "sha256": ev.sha(wire),
                    "aggregate_peak": 2 * len(wire)}
        except Exception:
            self.artifacts = old
            raise
        finally:
            self.partial = None

    def readback(self, limit):
        self.h.access("readback", limit)
        self.read_limit = limit
        wire = self.artifacts["evidence"]
        assert limit == len(wire) + 1
        return (wire + b"x")[:limit] if self.mode == "external_readback" else wire[:limit]

    def complete(self, wire, limits):
        self.h.access("complete")
        assert len(wire) <= limits.small_bytes
        if WORKER_CANDIDATE in self.artifacts:
            raise scd.Refusal("completion_collision")
        if self.mode == "completion_short_write":
            return False
        if self.mode == "completion_crash":
            raise RuntimeError("synthetic crash")
        self.artifacts[WORKER_CANDIDATE] = wire
        return True

    def failure(self, wire, limits):
        self.h.access("failure")
        assert len(wire) <= limits.small_bytes
        self.failures.append(json.loads(wire))
        # No overwrite of existing status or evidence.
        self.artifacts.setdefault("failure", wire)


class FakeParentController:
    """Synthetic authority model; no launch, IPC, OS controller or file access.

    The coordinator and all local handles live inside run_worker. The parent
    sees only a bounded scalar terminal message and worker candidate bytes.
    """
    def __init__(self, harness):
        self.h = harness
        self.parent_identity = object()
        self.authoritative = None
        self.artifacts = {}
        self.message = None
        self.events = []
        self.telemetry = {}

    def checkpoint(self, stage):
        self.events.append(("checkpoint", stage))
        return {"status": "running", "wall_seconds": 1, "committed_bytes": 1024,
                "log_bytes": 0, **self.telemetry}

    def run_worker(self):
        assert self.h.worker_identity is not self.parent_identity
        self.events.append(("worker_start",))
        result = self.h.run()
        assert PUBLIC_COMPLETION not in self.h.publication.artifacts
        candidate = self.h.publication.artifacts.get(WORKER_CANDIDATE)
        self.message = ev.canonical({"status": result.status, "completion": result.completion,
            "candidate_sha256": ev.sha(candidate) if candidate else None,
            "evidence_sha256": result.evidence_receipt["sha256"] if result.evidence_receipt else None})
        assert len(self.message) <= driver.FROZEN_LIMITS.small_bytes
        self.events.append(("worker_terminal",))
        terminal = json.loads(self.message)
        if terminal["status"] != "complete" or not terminal["completion"]:
            return result
        assert type(candidate) is bytes and terminal["candidate_sha256"] == ev.sha(candidate)
        assert any(e == ("checkpoint", "terminal_commit") for e in self.h.events)
        driver.check_resource(self, "authoritative_completion", driver.FROZEN_LIMITS)
        assert self.authoritative is None
        self.authoritative = candidate
        self.artifacts[PUBLIC_COMPLETION] = candidate
        self.events.append(("authoritative_completion",))
        return result


def closed_work(limit, processed=0):
    return {"limit": limit, "requested": processed, "charged": processed,
            "processed": processed, "uncharged": 0, "unprocessed": 0}


def authored_diagnostic(sources, *, note="authored"):
    nested = {"sources": copy.deepcopy(sources), "note": note}
    nested["sha256"] = ev.sha(ev.canonical(nested))
    outer = {"nested": nested}
    outer["sha256"] = ev.sha(ev.canonical(outer))
    return ev.canonical(outer)


def reseal_binding(request):
    request.bindings["binding_sha256"] = ev.sha(ev.canonical(
        {k: v for k, v in request.bindings.items() if k != "binding_sha256"}))


class Harness:
    def __init__(self):
        self.events = []
        self.inject = None
        self.arrays = [authored_arrays(n) for n in driver.EXPECTED_SUPPLY]
        # Expected sentinel objects; the coordinator must construct this tuple
        # after its permitted matching-frame decodes, with no preload accessor.
        self.arrays[0] = (self.arrays[1][0], self.arrays[0][1], self.arrays[1][2])
        self.encoded = {"authored-left": b"authored-encoded-left",
                        "authored-right": b"authored-encoded-right"}
        self.capability = FakeCapability(self)
        self.supervisor = FakeSupervisor(self)
        self.publication = FakePublication(self)
        self.prepared = []
        self.frames = []
        self.extracted = set()
        self.matches = []
        self.arm_tables = []
        self.actual_counts = None
        self.prepare_edit = None
        self.extract_edit = None
        self.match_edit = None
        self.encoded_reads = set()
        self.rgb_decodes = set()
        self.mask_unpacks = set()
        self.worker_identity = object()
        self.receipt_workers = {}
        self.empty_domains = False
        # Physical plane IDs deliberately differ from feasibility row labels.
        sentinel = self.source(0, "authored-left", 7)
        sources = [self.source(i, name, plane) for i, name, plane in
                   ((1, "authored-left", 11), (2, "authored-right", 29))]
        inventory = [sentinel, *sources]
        associations = [{"label": label, "frame_id": source["frame_id"],
                         "endpoint": source["endpoint"], "region_id": source["region_id"]}
                        for label, source in zip(driver.FEASIBILITY_IDENTITIES, inventory)]
        self.diagnostic = authored_diagnostic(inventory)
        diagnostic_object = json.loads(self.diagnostic)
        runtime = {"implementation": "authored-injected-only", "numpy": "sealed-fake-runtime"}
        binding = {"sources": sources,
            "source_pins": [list(row) for row in driver.ACCEPTED_SOURCE_PINS],
            "diagnostic_outer_sha256": diagnostic_object["sha256"],
            "diagnostic_nested_sha256": diagnostic_object["nested"]["sha256"],
            "contract_sha256": driver.CONTRACT_PIN, "source_sha256": driver.DEPENDENCY_PINS["descriptor"],
            "runtime_sha256": ev.sha(ev.canonical(runtime)), "authorization_sha256": ev.sha(b"fake-capability")}
        binding["binding_sha256"] = ev.sha(ev.canonical(binding))
        metadata = {"expected_supply": list(driver.EXPECTED_SUPPLY),
            "feasibility_identities": list(driver.FEASIBILITY_IDENTITIES),
            "reservation": driver.reservations(), "dependency_pins": dict(driver.DEPENDENCY_PINS),
            "source_pins": copy.deepcopy(binding["source_pins"]),
            "contract_sha256": binding["contract_sha256"], "source_sha256": binding["source_sha256"],
            "source_inventory": copy.deepcopy(inventory), "label_associations": copy.deepcopy(associations),
            "inventory_sha256": ev.sha(ev.canonical({"sources": inventory, "label_associations": associations})),
            "diagnostic_payload_sha256": ev.sha(self.diagnostic),
            "diagnostic_outer_sha256": binding["diagnostic_outer_sha256"],
            "diagnostic_nested_sha256": binding["diagnostic_nested_sha256"],
            "runtime_sha256": binding["runtime_sha256"]}
        self.request = driver.Request(binding, runtime, metadata,
            sentinel, ev.sha(self.diagnostic), len(self.diagnostic), copy.deepcopy(associations))
        self.dependencies = driver.Dependencies(self.capability, self.inventory, self.metadata,
            self.read, self.inspect_diagnostic, self.decode_rgb, self.ingress,
            self.decode_mask, self.usable, self.prepare, self.extract, self.match,
            self.supervisor, self.publication)

    def source(self, index, name, endpoint):
        pixels, mask, usable = self.arrays[index]
        return {"frame_id": name, "endpoint": [name, endpoint], "region_id": ev.sha(f"region-{index}".encode()),
            "encoded_sha256": ev.sha(self.encoded[name]), "encoded_length": len(self.encoded[name]),
            "decoded_shape": list(driver.NATIVE_SHAPE), "decoded_dtype": "|u1",
            "decoded_rgb_sha256": ev.sha(pixels.tobytes()), "ingress_sha256": ev.sha(pixels.tobytes()),
            "packed_mask_sha256": driver.packed_digest(mask), "full_native_bool_sha256": ev.sha(mask.tobytes()),
            "usable_mask_sha256": ev.sha(usable.tobytes()), "usable_helper_sha256": ev.sha(b"fake-usable-helper")}

    def access(self, name, *detail):
        assert self.capability.consumed, f"access before authorization: {name}"
        self.events.append((name, *detail))
        if self.inject == name:
            raise RuntimeError("untrusted exception text must not escape")

    def inventory(self):
        self.access("inventory")
        raise AssertionError("hidden preload inventory must never be accessed")

    def metadata(self):
        self.access("metadata")
        return copy.deepcopy(self.request.metadata)

    def read(self, kind, key, limit):
        self.access("read", kind, key, limit)
        if kind == "encoded":
            if key not in self.encoded:
                raise scd.Refusal("undeclared_rgb_read")
            if key in self.encoded_reads:
                raise scd.Refusal("duplicate_rgb_read")
            self.encoded_reads.add(key)
        payload = self.diagnostic if kind == "diagnostic" else self.encoded[key]
        assert limit == len(payload) + 1
        return payload[:limit]

    def inspect_diagnostic(self, payload):
        self.access("inspect_diagnostic")
        assert payload == self.diagnostic
        parsed = json.loads(payload)
        nested = parsed["nested"]
        assert parsed["sha256"] == ev.sha(ev.canonical({k: v for k, v in parsed.items() if k != "sha256"}))
        assert nested["sha256"] == ev.sha(ev.canonical({k: v for k, v in nested.items() if k != "sha256"}))
        return driver.DiagnosticInspection(ev.sha(payload), parsed["sha256"], nested["sha256"],
                                           tuple(nested["sources"]))

    def decode_rgb(self, encoded, name):
        self.access("decode_rgb", name)
        if name not in self.encoded:
            raise scd.Refusal("undeclared_rgb_decode")
        if name in self.rgb_decodes:
            raise scd.Refusal("duplicate_rgb_decode")
        self.rgb_decodes.add(name)
        assert any(e[0] == "inspect_diagnostic" for e in self.events)
        assert encoded == self.encoded[name]
        return self.arrays[1 if name == "authored-left" else 2][0]

    def ingress(self, pixels):
        self.access("ingress")
        return ev.sha(pixels.tobytes())

    def decode_mask(self, diagnostic, region):
        self.access("decode_mask", region)
        assert diagnostic == self.diagnostic
        if region in self.mask_unpacks:
            raise scd.Refusal("duplicate_mask_unpack")
        self.mask_unpacks.add(region)
        for i, source in enumerate([self.request.row176, *self.request.bindings["sources"]]):
            if source["region_id"] == region:
                if i == 0:
                    assert self.rgb_decodes == set(self.encoded)
                return self.arrays[i][1]
        raise scd.Refusal("undeclared_mask_region")

    def usable(self, pixels, helper):
        self.access("usable")
        assert helper == ev.sha(b"fake-usable-helper")
        return next(a[2] for a in self.arrays[1:] if a[0] is pixels)

    def prepare(self, accepted_prepare, *items, role):
        index = len(self.prepared)
        self.access("prepare", driver.FEASIBILITY_IDENTITIES[index], role)
        assert accepted_prepare is ev.prepare_shared_frame
        assert all(a is b for a, b in zip(items, self.arrays[index]))
        # The accepted pure planner is allowed; no extraction/native decoding.
        receipt = accepted_prepare(*items, role=role)
        self.receipt_workers[id(receipt)] = self.worker_identity
        self.prepared.append(receipt)
        if self.actual_counts is not None:
            plan = replace(receipt.plan, eligible_count=self.actual_counts[index])
            receipt._plan = plan
        if self.prepare_edit:
            self.prepare_edit(index, receipt)
        return receipt

    def extract(self, accepted_extract, receipt):
        self.access("extract", driver.FEASIBILITY_IDENTITIES[self.prepared.index(receipt)])
        assert accepted_extract is ev.extract_prepared_frame and len(self.prepared) == 3
        assert receipt is not self.prepared[0] and id(receipt) not in self.extracted
        assert self.receipt_workers[id(receipt)] is self.worker_identity
        assert tuple(p.plan.eligible_count for p in self.prepared) == driver.EXPECTED_SUPPLY
        self.extracted.add(id(receipt))
        # Deterministic authored tile table; no luma, tile sampling or descriptors from pixels.
        count = len(receipt.plan.sites)
        tiles = np.zeros((count, 9, 32), dtype="<f4")
        if not self.empty_domains:
            tiles[:, :, 0] = 1
        central = np.full(count, self.empty_domains, bool)
        neighbors = np.full(count, 8 if self.empty_domains else 0, np.int64)
        for a in (tiles, central, neighbors):
            a.setflags(write=False)
        frame = ev.SharedFrame(receipt.plan, tiles, central, neighbors, ev.sha(tiles.tobytes()), receipt._source_refs)
        self.frames.append(frame)
        return self.extract_edit(frame) if self.extract_edit else frame

    def match(self, accepted_match, left, right, **limits):
        self.access("match", left.arm)
        assert accepted_match is scd.match_arms and left.arm == right.arm
        if self.matches:
            assert self.matches[-1].distance_matrix is None, "matrix must be released before next arm"
        assert left.tile_tables is self.frames[0].tiles and right.tile_tables is self.frames[1].tiles
        self.arm_tables.append((left.tile_tables, right.tile_tables))
        ids = [[a.sites[int(i)].site_id for i in a.shared_indices] for a in (left, right)]
        matrix = np.zeros(tuple(map(len, ids)), np.float64)
        def tied(origin, target):
            counts = {"origins": len(origin), "absent_second_neighbor": 0,
                      "nearest_distance_tie": len(origin) if all(ids) else 0,
                      "ratio_rejected": 0, "ratio_accepted": 0}
            if not all(ids):
                return counts, ()
            rows = tuple({"origin": i, "nearest": target[0], "second": target[1],
                          "nearest_squared": 0.0, "second_squared": 0.0,
                          "status": "nearest_distance_tie"} for i in origin)
            return counts, rows
        forward, fe = tied(ids[0], ids[1])
        reverse, re = tied(ids[1], ids[0])
        witness = {"status": "exhaustive_no_witness", "pair_ids": [],
            "source_binding_sha256": limits["source_binding_sha256"],
            "node_work": closed_work(limits["witness_node_limit"]),
            "determinant_work": closed_work(limits["witness_determinant_limit"])}
        match = scd.ArmMatch(left.arm, "no_admitted_pairs" if all(ids) else "insufficient_support", tuple(ids[0]), tuple(ids[1]),
            forward, reverse, fe, re, (), witness,
            closed_work(limits["comparison_limit"], matrix.size), matrix, limits["source_binding_sha256"])
        self.matches.append(match)
        if self.match_edit:
            self.match_edit(match)
        return match

    def run(self):
        return driver.coordinate(self.request, self.dependencies)


@pytest.fixture
def harness():
    return Harness()


def no_completion(h, result):
    assert not result.completion and WORKER_CANDIDATE not in h.publication.artifacts
    assert PUBLIC_COMPLETION not in h.publication.artifacts
    if result.consumed:
        assert h.capability.consumed


def test_complete_authored_flow_exact_counts_tables_release_and_wire(harness, monkeypatch):
    h = harness
    calls = []
    original = scd.plan_sites
    def plan(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(scd, "plan_sites", plan)
    def forbidden(*args, **kwargs):
        raise AssertionError("native or duplicate extraction reached")
    for name in ("luma_from_rgb", "tile_table", "match_arms"):
        monkeypatch.setattr(scd, name, forbidden)
    # No file open is needed by the coordinator or any fake boundary.
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = h.run()
    assert result.status == "complete" and result.code == "none" and result.completion
    assert len(calls) == 3 and [len(p.plan.sites) for p in h.prepared] == [0, 37, 2570]
    assert [e[1:] for e in h.events if e[0] == "prepare"] == [("row176", 0), ("row197", 0), ("shared-right", 1)]
    assert [e[1] for e in h.events if e[0] == "extract"] == ["row197", "shared-right"]
    assert [e[1] for e in h.events if e[0] == "match"] == ["A", "B"]
    assert [e[1] for e in h.events if e[0] == "decode_rgb"] == ["authored-left", "authored-right"]
    assert len([e for e in h.events if e[0] == "decode_mask"]) == 3
    assert len([e for e in h.events if e[0] == "usable"]) == 2
    assert len([e for e in h.events if e[0] == "ingress"]) == 2
    assert not any(e[0] == "inventory" for e in h.events)
    inspection_index = next(i for i, e in enumerate(h.events) if e[0] == "inspect_diagnostic")
    rgb_indices = [i for i, e in enumerate(h.events) if e[0] == "decode_rgb"]
    sentinel_mask_index = next(i for i, e in enumerate(h.events)
                               if e == ("decode_mask", h.request.row176["region_id"]))
    assert inspection_index < min(rgb_indices) <= max(rgb_indices) < sentinel_mask_index
    assert h.prepared[0]._source_refs[0] is h.prepared[1]._source_refs[0]
    assert h.prepared[0]._source_refs[2] is h.prepared[1]._source_refs[2]
    assert h.prepared[0]._source_refs[1] is h.arrays[0][1]
    assert len([e for e in h.events if e[0] == "read"]) == 3
    assert all(a is b for a, b in zip(h.arm_tables[0], h.arm_tables[1]))
    assert all(m.distance_matrix is None for m in h.matches)
    assert all(not a.flags.writeable for f in h.frames for a in (f.tiles, f.central_zero, f.neighbor_zero_counts))
    assert result.row176 == {"status": "untestable_footprint", "sites": 0}
    assert result.counters["reservation"] == driver.reservations()
    assert result.counters["footprint"]["processed"] == 16800
    assert result.counters["combined"]["comparisons"]["processed"] == 190180
    wire = h.publication.artifacts["evidence"]
    record = ev.loads(wire)
    assert record["treatment"] == ev.TREATMENT and record["outcomes"] == ["no_admitted_pairs"] * 2
    assert all(not a["pairs"] and a["summary"]["forward_counts"]["ratio_accepted"] == 0 for a in record["arms"])
    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert "distance_matrix" not in keys(record) and "distance_matrix_sha256" in keys(record)
    assert h.publication.read_limit == len(wire) + 1
    assert json.loads(h.publication.artifacts[WORKER_CANDIDATE])["evidence_sha256"] == ev.sha(wire)
    assert PUBLIC_COMPLETION not in h.publication.artifacts
    terminal_index = next(i for i, e in enumerate(h.events) if e[0] == "supervisor_terminal")
    completion_index = next(i for i, e in enumerate(h.events) if e[0] == "complete")
    checkpoint_index = next(i for i, e in enumerate(h.events) if e == ("checkpoint", "terminal_commit"))
    assert terminal_index < checkpoint_index < completion_index
    assert h.events[completion_index - 1] == ("checkpoint", "terminal_commit")
    assert record["bindings"]["sources"][0]["endpoint"] == ["authored-left", 11]
    assert record["bindings"]["sources"][1]["endpoint"] == ["authored-right", 29]


@pytest.mark.parametrize("kind", ["missing", "swapped", "endpoint", "frame", "region", "extra", "forged_source"])
def test_label_associations_join_frozen_physical_inventory_before_access(harness, kind):
    h = harness
    associations = h.request.label_associations
    if kind == "missing":
        associations.pop()
    elif kind == "swapped":
        associations[0], associations[1] = associations[1], associations[0]
    elif kind in ("endpoint", "frame", "region"):
        key = {"endpoint": "endpoint", "frame": "frame_id", "region": "region_id"}[kind]
        associations[1][key] = {"endpoint": ["authored-left", 197],
                                "frame": "forged-frame", "region": "f" * 64}[kind]
    elif kind == "extra":
        associations.append(copy.deepcopy(associations[1]))
    else:
        h.request.bindings["sources"][0]["endpoint"][1] = 197
    reseal_binding(h.request)
    result = h.run()
    assert result.status == "refused" and h.events == []
    assert h.prepared == [] and h.frames == [] and h.matches == []
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["missing", "substituted_path", "substituted_hash", "duplicate", "reordered", "extra"])
def test_resealed_source_pin_list_is_exact_and_ordered_before_access(harness, kind):
    h = harness
    pins = h.request.bindings["source_pins"]
    if kind == "missing":
        pins.pop()
    elif kind == "substituted_path":
        pins[0][0] = "scripts/evaluation/forged.py"
    elif kind == "substituted_hash":
        pins[1][1] = "f" * 64
    elif kind == "duplicate":
        pins[1] = copy.deepcopy(pins[0])
    elif kind == "reordered":
        pins.reverse()
    else:
        pins.append(["scripts/evaluation/extra.py", "f" * 64])
    # Even coherent caller metadata and a valid reseal cannot replace accepted pins.
    h.request.metadata["source_pins"] = copy.deepcopy(pins)
    reseal_binding(h.request)
    result = h.run()
    assert result.code == "accepted_source_pins" and result.status == "refused"
    assert h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("field", ["dependency_pins", "source_pins", "contract_sha256", "source_sha256", "runtime_sha256"])
def test_source_metadata_pin_consistency_before_access(harness, field):
    h = harness
    h.request.metadata[field] = {} if field == "dependency_pins" else [] if field == "source_pins" else "f" * 64
    result = h.run()
    assert result.status == "refused" and h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["untyped", "payload", "outer", "nested", "missing_source", "source_order",
    "endpoint", "region_id", "packed_mask_sha256", "full_native_bool_sha256", "usable_mask_sha256",
    "usable_helper_sha256"])
def test_parsed_diagnostic_inventory_identity_mismatch_before_decoding(harness, kind):
    h = harness
    original = h.dependencies.inspect_diagnostic
    def inspect_payload(payload):
        inspection = original(payload)
        if kind == "untyped":
            return copy.deepcopy(h.request.metadata)
        if kind in ("payload", "outer", "nested"):
            return replace(inspection, **{kind + "_sha256": "f" * 64})
        sources = copy.deepcopy(list(inspection.sources))
        if kind == "missing_source":
            sources.pop(0)
        elif kind == "source_order":
            sources.reverse()
        else:
            sources[2][kind] = ["authored-right", 197] if kind == "endpoint" else "f" * 64
        return replace(inspection, sources=tuple(sources))
    h.dependencies = replace(h.dependencies, inspect_diagnostic=inspect_payload)
    result = h.run()
    assert result.status == "incomplete"
    assert not any(e[0] in ("decode_rgb", "decode_mask", "prepare", "extract", "match", "complete") for e in h.events)
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["outer", "nested", "endpoint", "region_id", "packed_mask_sha256",
    "full_native_bool_sha256", "usable_mask_sha256", "usable_helper_sha256"])
def test_coherently_resealed_diagnostic_substitution_cannot_replace_frozen_inventory(harness, kind):
    h = harness
    sources = copy.deepcopy(h.request.metadata["source_inventory"])
    if kind not in ("outer", "nested"):
        sources[2][kind] = ["authored-right", 197] if kind == "endpoint" else "f" * 64
    h.diagnostic = authored_diagnostic(sources, note="coherently-substituted")
    parsed = json.loads(h.diagnostic)
    if kind == "outer":
        parsed["outer_note"] = "substitution"
        parsed["sha256"] = ev.sha(ev.canonical({k: v for k, v in parsed.items() if k != "sha256"}))
        h.diagnostic = ev.canonical(parsed)
    h.request = replace(h.request, diagnostic_sha256=ev.sha(h.diagnostic), diagnostic_length=len(h.diagnostic))
    h.request.bindings["diagnostic_outer_sha256"] = parsed["sha256"]
    h.request.bindings["diagnostic_nested_sha256"] = parsed["nested"]["sha256"]
    # Admit the new bounded payload digest while retaining the independent
    # frozen outer/nested or source inventory identity targeted by this case.
    h.request.metadata["diagnostic_payload_sha256"] = ev.sha(h.diagnostic)
    if kind != "outer":
        h.request.metadata["diagnostic_outer_sha256"] = parsed["sha256"]
    if kind not in ("outer", "nested"):
        h.request.metadata["diagnostic_nested_sha256"] = parsed["nested"]["sha256"]
    reseal_binding(h.request)
    result = h.run()
    expected = {"outer": "diagnostic_outer_identity", "nested": "diagnostic_nested_identity"}
    assert result.code == expected.get(kind, "diagnostic_source_inventory") and result.status == "incomplete"
    assert any(e[0] == "inspect_diagnostic" for e in h.events)
    assert not any(e[0] in ("decode_rgb", "decode_mask", "prepare", "extract", "match", "complete") for e in h.events)
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["exception", "termination", "wrong_identity", "wrong_status", "changed_bytes", "late_metadata"])
def test_supervisor_failure_after_staged_success_never_commits_completion(harness, kind):
    h = harness
    def after_callback(terminal):
        assert terminal.status == "staged_complete"
        assert "evidence" in h.publication.artifacts and WORKER_CANDIDATE not in h.publication.artifacts
        if kind == "exception":
            raise RuntimeError("late supervisor failure")
        if kind == "termination":
            raise scd.Refusal("supervisor_resource_stop")
        if kind == "wrong_identity":
            return replace(terminal)
        if kind == "wrong_status":
            object.__setattr__(terminal, "status", "terminated")
        elif kind == "changed_bytes":
            object.__setattr__(terminal, "completion_bytes", b"forged-completion")
        else:
            h.request.runtime["implementation"] = "late-tamper"
        return terminal
    h.supervisor.after_callback = after_callback
    result = h.run()
    assert result.status == "incomplete" and result.evidence_receipt is not None
    assert result.counters["combined"]["comparisons"]["processed"] == 190180
    assert h.supervisor.stopped == [result.code]
    assert not any(e[0] == "complete" for e in h.events)
    no_completion(h, result)


@pytest.mark.parametrize("target", ["request", "limits", "bindings", "runtime", "metadata",
    "source", "row176", "inventory", "association", "metadata_association",
    "dependency", "metadata_dependency", "reservation"])
@pytest.mark.parametrize("kind", ["oversized", "unknown", "nested", "cardinality"])
def test_entire_request_schema_refuses_before_canonical_copy_or_access(harness, monkeypatch, target, kind):
    h = harness
    targets = {
        "request": (vars(h.request), "diagnostic_sha256"),
        "limits": (vars(h.request.limits), "wall_seconds"),
        "bindings": (h.request.bindings, "runtime_sha256"),
        "runtime": (h.request.runtime, "implementation"),
        "metadata": (h.request.metadata, "inventory_sha256"),
        "source": (h.request.bindings["sources"][0], "region_id"),
        "row176": (h.request.row176, "region_id"),
        "inventory": (h.request.metadata["source_inventory"][0], "region_id"),
        "association": (h.request.label_associations[0], "region_id"),
        "metadata_association": (h.request.metadata["label_associations"][0], "region_id"),
        "dependency": (h.request.dependency_pins, "evidence"),
        "metadata_dependency": (h.request.metadata["dependency_pins"], "evidence"),
        "reservation": (h.request.metadata["reservation"], "plans"),
    }
    # Limits is shared by default; isolate this fixture before deliberate mutation.
    if target == "limits":
        h.request = replace(h.request, limits=replace(driver.FROZEN_LIMITS))
        targets[target] = (vars(h.request.limits), "wall_seconds")
    record, field = targets[target]
    if kind == "oversized":
        record[field] = "x" * (ev.SECTION_CAPS["source"] + 1)
    elif kind == "unknown":
        record["unexpected"] = "unknown"
    elif kind == "nested":
        value = "nested"
        for _ in range(ev.DEPTH_CAP + 1):
            value = {"nested": value}
        record[field] = value
    else:
        record.update({f"extra_{i}": i for i in range(257)})
    reached = []
    def forbidden(name):
        def trap(*args, **kwargs):
            reached.append(name)
            raise AssertionError("request guard must precede " + name)
        return trap
    monkeypatch.setattr(ev, "canonical", forbidden("canonical"))
    monkeypatch.setattr(ev, "sha", forbidden("digest_serialization"))
    monkeypatch.setattr(driver.copy, "deepcopy", forbidden("deepcopy"))
    result = h.run()
    assert result.status == "refused" and result.code != "injected_boundary_failure"
    assert reached == [] and h.events == [] and not result.consumed
    assert h.prepared == [] and h.frames == [] and h.matches == []
    assert h.publication.artifacts == {}
    no_completion(h, result)


@pytest.mark.parametrize("target", ["sources", "inventory", "associations", "metadata_associations",
    "source_pins", "metadata_source_pins", "source_pin_row", "supply", "identities",
    "source_endpoint", "row176_endpoint", "source_shape", "inventory_endpoint", "association_endpoint"])
def test_request_list_cardinalities_refuse_before_serialization_copy_or_access(harness, monkeypatch, target):
    h = harness
    targets = {
        "sources": h.request.bindings["sources"],
        "inventory": h.request.metadata["source_inventory"],
        "associations": h.request.label_associations,
        "metadata_associations": h.request.metadata["label_associations"],
        "source_pins": h.request.bindings["source_pins"],
        "metadata_source_pins": h.request.metadata["source_pins"],
        "source_pin_row": h.request.bindings["source_pins"][0],
        "supply": h.request.metadata["expected_supply"],
        "identities": h.request.metadata["feasibility_identities"],
        "source_endpoint": h.request.bindings["sources"][0]["endpoint"],
        "row176_endpoint": h.request.row176["endpoint"],
        "source_shape": h.request.bindings["sources"][0]["decoded_shape"],
        "inventory_endpoint": h.request.metadata["source_inventory"][0]["endpoint"],
        "association_endpoint": h.request.label_associations[0]["endpoint"],
    }
    sequence = targets[target]
    # Extra entries repeat an existing identity as well as exceeding cardinality.
    sequence.extend([sequence[0]] * 8705)
    reached = []
    def forbidden(*args, **kwargs):
        reached.append(True)
        raise AssertionError("unbounded request reached serialization or copy")
    monkeypatch.setattr(ev, "canonical", forbidden)
    monkeypatch.setattr(ev, "sha", forbidden)
    monkeypatch.setattr(driver.copy, "deepcopy", forbidden)
    result = h.run()
    assert result.status == "refused" and result.code != "injected_boundary_failure"
    assert reached == [] and h.events == [] and not result.consumed
    assert h.publication.artifacts == {}
    no_completion(h, result)


def test_complete_request_envelope_is_capped_before_hash_copy_and_access(harness, monkeypatch):
    h = harness
    bounded = ev._bounded_object
    visits = []
    def inspect_envelope(value, cap):
        if type(value) is dict and "metadata" in value and "row176" in value:
            visits.append(value)
            assert set(value) == set(vars(h.request))
            assert value["metadata"] is h.request.metadata
            assert value["row176"] is h.request.row176
            assert value["label_associations"] is h.request.label_associations
            assert value["limits"] == vars(h.request.limits)
            assert cap == 262144 == ev.SECTION_CAPS["source"]
            # Exercise the accepted traversal's refusal without increasing caps.
            bounded({"guard": "x" * ev.STRING_CAP}, 1)
        return bounded(value, cap)
    monkeypatch.setattr(ev, "_bounded_object", inspect_envelope)
    def forbidden_copy(*args, **kwargs):
        raise AssertionError("copy preceded the complete-envelope guard")
    monkeypatch.setattr(driver.copy, "deepcopy", forbidden_copy)
    result = h.run()
    assert result.code == "section_byte_limit" and result.status == "refused"
    assert len(visits) == 1 and h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("telemetry,code", [
    ({"status": "terminated"}, "supervisor_resource_stop"),
    ({"status": "stopped"}, "supervisor_resource_stop"),
    ({"wall_seconds": 181}, "supervisor_timeout"),
    ({"wall_seconds": float("nan")}, "supervisor_timeout"),
    ({"committed_bytes": 2 * 1024 ** 3 + 1}, "supervisor_memory"),
    ({"log_bytes": 65537}, "supervisor_log"),
    ({"committed_bytes": None}, "supervisor_memory"),
    ({"wall_seconds": "181"}, "supervisor_timeout"),
    ({"extra": "malformed"}, "supervisor_telemetry"),
])
def test_unchanged_staged_terminal_cannot_hide_late_resource_stop(harness, telemetry, code):
    h = harness
    returned = []
    def after_callback(terminal):
        assert terminal.status == "staged_complete"
        assert "evidence" in h.publication.artifacts and WORKER_CANDIDATE not in h.publication.artifacts
        returned.append(terminal)
        h.supervisor.telemetry["terminal_commit"] = telemetry
        return terminal
    h.supervisor.after_callback = after_callback
    result = h.run()
    assert len(returned) == 1 and returned[0].status == "staged_complete"
    assert result.status == "incomplete" and result.code == code
    assert result.evidence_receipt is not None and "evidence" in h.publication.artifacts
    assert result.counters["combined"]["comparisons"]["processed"] == 190180
    assert result.arm_dispositions == ("complete_no_witness", "complete_no_witness")
    assert h.supervisor.stopped == [code]
    assert ("checkpoint", "terminal_commit") in h.events
    assert not any(e[0] == "complete" for e in h.events)
    no_completion(h, result)


def test_invalid_authorization_and_one_shot_reuse(harness):
    h = harness
    h.capability.invalid = True
    refused = h.run()
    assert refused.status == "refused" and not refused.consumed
    assert [e[0] for e in h.events] == ["valid"]
    h.capability.invalid = False
    h.actual_counts = (1, 37, 2570)
    first = h.run()
    assert first.status == "inconclusive" and first.consumed
    h.events.clear()
    second = h.run()
    assert second.code == "authorization_reused" and second.status == "refused"
    assert [e[0] for e in h.events] == ["valid", "consume"]


@pytest.mark.parametrize("index", range(3))
def test_sealed_count_mismatch_zero_access(harness, index):
    h = harness
    h.request.metadata["expected_supply"][index] += 1
    result = h.run()
    assert result.code == "frozen_supply" and result.status == "refused" and h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("field", list(driver.reservations()))
def test_each_sealed_reservation_mismatch_zero_access(harness, field):
    h = harness
    h.request.metadata["reservation"][field] += 1
    result = h.run()
    assert result.code == "frozen_reservation" and h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("field", list(driver.Limits.__dataclass_fields__))
def test_every_frozen_ceiling_change_refused_before_access(harness, field):
    h = harness
    h.request = replace(h.request, limits=replace(h.request.limits, **{field: getattr(h.request.limits, field) + 1}))
    result = h.run()
    assert result.code == "frozen_limits" and h.events == []


def test_sentinel_mask_unpack_failure_after_two_decodes_before_preparation(harness):
    h = harness
    original = h.dependencies.decode_mask
    def unpack(diagnostic, region):
        if region == h.request.row176["region_id"]:
            raise scd.Refusal("sentinel_mask_unavailable")
        return original(diagnostic, region)
    h.dependencies = replace(h.dependencies, decode_mask=unpack)
    result = h.run()
    assert result.status == "incomplete" and result.code == "sentinel_mask_unavailable"
    assert h.rgb_decodes == set(h.encoded) and not any(e[0] == "inventory" for e in h.events)
    assert not any(e[0] in ("prepare", "extract", "match", "evidence") for e in h.events)
    no_completion(h, result)


def test_diagnostic_binding_precedes_all_decodes_and_no_preload_access(harness, monkeypatch):
    h = harness
    checked = driver.checked_diagnostic
    def validate(*args):
        checked(*args)
        h.events.append(("diagnostic_validated",))
    monkeypatch.setattr(driver, "checked_diagnostic", validate)
    def never_inventory():
        raise AssertionError("preload accessor reached")
    h.dependencies = replace(h.dependencies, in_memory_inventory=never_inventory)
    result = h.run()
    assert result.completion
    validated = h.events.index(("diagnostic_validated",))
    assert all(validated < i for i, e in enumerate(h.events) if e[0] in ("decode_rgb", "decode_mask", "usable"))
    assert h.encoded_reads == h.rgb_decodes == set(h.encoded)
    assert h.mask_unpacks == {s["region_id"] for s in [h.request.row176, *h.request.bindings["sources"]]}
    assert h.prepared[0]._source_refs[0] is h.frames[0].source_refs[0]
    assert h.prepared[0]._source_refs[2] is h.frames[0].source_refs[2]


@pytest.mark.parametrize("kind,code", [("duplicate_read", "duplicate_rgb_read"),
    ("third_read", "undeclared_rgb_read"), ("duplicate_decode", "duplicate_rgb_decode"),
    ("third_decode", "undeclared_rgb_decode")])
def test_duplicate_or_third_rgb_ingress_is_trapped_before_preparation(harness, kind, code):
    h = harness
    original = h.dependencies.decode_rgb
    def decode(payload, name):
        pixels = original(payload, name)
        if kind == "duplicate_read":
            h.read("encoded", name, len(payload) + 1)
        elif kind == "third_read":
            h.read("encoded", "sentinel-third-image", len(payload) + 1)
        elif kind == "duplicate_decode":
            original(payload, name)
        else:
            original(payload, "sentinel-third-image")
        return pixels
    h.dependencies = replace(h.dependencies, decode_rgb=decode)
    result = h.run()
    assert result.status == "incomplete" and result.code == code
    assert h.prepared == [] and h.frames == [] and h.matches == []
    no_completion(h, result)


@pytest.mark.parametrize("field", ["decoded_rgb_sha256", "ingress_sha256", "usable_mask_sha256", "usable_helper_sha256"])
def test_sentinel_shared_left_identity_drift_refuses_before_input_access(harness, field):
    h = harness
    h.request.row176[field] = "f" * 64
    h.request.metadata["source_inventory"][0][field] = "f" * 64
    h.request.metadata["inventory_sha256"] = ev.sha(ev.canonical({
        "sources": h.request.metadata["source_inventory"], "label_associations": h.request.label_associations}))
    result = h.run()
    expected = "row176_usable_identity" if field.startswith("usable_") else "row176_rgb_identity"
    assert result.status == "refused" and result.code == expected and h.events == []
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["wrong_mask", "packed_pin", "resource_stop"])
def test_sentinel_mask_binding_and_resource_stop_fail_before_three_plans(harness, kind):
    h = harness
    original = h.dependencies.decode_mask
    if kind == "resource_stop":
        h.supervisor.telemetry["row176_mask"] = {"wall_seconds": 181}
    elif kind == "wrong_mask":
        h.dependencies = replace(h.dependencies, decode_mask=lambda diagnostic, region:
            h.arrays[1][1] if region == h.request.row176["region_id"] else original(diagnostic, region))
    else:
        h.request.row176["packed_mask_sha256"] = "f" * 64
        h.request.metadata["source_inventory"][0]["packed_mask_sha256"] = "f" * 64
        h.request.metadata["inventory_sha256"] = ev.sha(ev.canonical({
            "sources": h.request.metadata["source_inventory"], "label_associations": h.request.label_associations}))
        # Keep diagnostic inventory consistent so the independently pinned
        # full/packed mask check, rather than inspection, is the failing seam.
        h.diagnostic = authored_diagnostic(h.request.metadata["source_inventory"])
        parsed = json.loads(h.diagnostic)
        h.request = replace(h.request, diagnostic_sha256=ev.sha(h.diagnostic), diagnostic_length=len(h.diagnostic))
        for key, value in (("diagnostic_outer_sha256", parsed["sha256"]),
                           ("diagnostic_nested_sha256", parsed["nested"]["sha256"])):
            h.request.bindings[key] = h.request.metadata[key] = value
        h.request.metadata["diagnostic_payload_sha256"] = ev.sha(h.diagnostic)
        reseal_binding(h.request)
    result = h.run()
    expected = "supervisor_timeout" if kind == "resource_stop" else "input_digest" if kind == "wrong_mask" else "source_mask_digest"
    assert result.status == "incomplete" and result.code == expected
    assert h.rgb_decodes == set(h.encoded) and h.prepared == [] and h.frames == [] and h.matches == []
    no_completion(h, result)


@pytest.mark.parametrize("terminal_mode", ["accepted", "wrong_identity", "late_stop"])
def test_worker_local_handles_never_cross_parent_authority_boundary(harness, monkeypatch, terminal_mode):
    h = harness
    parent = FakeParentController(h)
    local_receipts = []
    copied = driver.copy.deepcopy
    def no_pickle(*args, **kwargs):
        raise AssertionError("worker-local closure/receipt/terminal must never be pickled")
    monkeypatch.setattr(pickle, "dumps", no_pickle)
    monkeypatch.setattr(pickle, "loads", no_pickle)
    def bounded_copy(value, *args, **kwargs):
        assert not any(value is receipt for receipt in local_receipts)
        assert not any(value is frame for frame in h.frames)
        return copied(value, *args, **kwargs)
    monkeypatch.setattr(driver.copy, "deepcopy", bounded_copy)
    def local_preparation(index, receipt):
        assert parent.authoritative is None
        assert h.receipt_workers[id(receipt)] is h.worker_identity
        local_receipts.append(receipt)
    h.prepare_edit = local_preparation
    def after_callback(terminal):
        assert parent.authoritative is None and WORKER_CANDIDATE not in h.publication.artifacts
        assert PUBLIC_COMPLETION not in parent.artifacts and PUBLIC_COMPLETION not in h.publication.artifacts
        assert len(local_receipts) == 3 and len(h.frames) == 2
        assert set(vars(terminal)) == {"status", "completion_bytes"}
        assert terminal.status == "staged_complete" and type(terminal.completion_bytes) is bytes
        assert all(h.receipt_workers[id(receipt)] is h.worker_identity for receipt in local_receipts)
        if terminal_mode == "wrong_identity":
            return replace(terminal)
        if terminal_mode == "late_stop":
            h.supervisor.telemetry["terminal_commit"] = {"status": "terminated"}
        return terminal
    h.supervisor.after_callback = after_callback
    result = parent.run_worker()
    assert set(json.loads(parent.message)) == {"status", "completion", "candidate_sha256", "evidence_sha256"}
    assert all(str(id(receipt)).encode() not in parent.message for receipt in local_receipts)
    if terminal_mode == "accepted":
        assert result.completion and parent.authoritative == h.publication.artifacts[WORKER_CANDIDATE]
        assert parent.artifacts[PUBLIC_COMPLETION] == h.publication.artifacts[WORKER_CANDIDATE]
        assert PUBLIC_COMPLETION not in h.publication.artifacts
        assert parent.events[-2:] == [("checkpoint", "authoritative_completion"), ("authoritative_completion",)]
    else:
        assert result.status == "incomplete" and parent.authoritative is None
        no_completion(h, result)


def test_internal_worker_completion_is_not_external_authority_on_parent_resource_stop(harness):
    h = harness
    parent = FakeParentController(h)
    parent.telemetry = {"status": "terminated"}
    with pytest.raises(scd.Refusal, match="supervisor_resource_stop"):
        parent.run_worker()
    assert WORKER_CANDIDATE in h.publication.artifacts and parent.authoritative is None
    assert PUBLIC_COMPLETION not in h.publication.artifacts and PUBLIC_COMPLETION not in parent.artifacts
    assert json.loads(parent.message)["completion"] is True


@pytest.mark.parametrize("index", range(3))
def test_collective_actual_count_mismatch_no_extraction(harness, index):
    h = harness
    h.actual_counts = list(driver.EXPECTED_SUPPLY)
    h.actual_counts[index] += 1
    result = h.run()
    assert result.status == "inconclusive" and result.code == "footprint_supply_mismatch"
    assert len(h.prepared) == 3 and h.frames == [] and h.matches == []
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["role", "order", "version", "source", "usable", "rgb", "grid", "site_order", "site_id"])
def test_collective_prepared_binding_work_and_order_failures(harness, kind):
    h = harness
    def edit(index, receipt):
        if index != 2:
            return
        if kind == "role":
            receipt._role = 0
        elif kind == "order":
            receipt._plan = h.prepared[1].plan
        elif kind == "version":
            receipt._version = "incorrect"
        else:
            changes = {"source": {"source_mask_sha256": "f" * 64},
                       "usable": {"usable_mask_sha256": "f" * 64}, "rgb": {"frame_sha256": "f" * 64},
                       "grid": {"grid_centers": 5601},
                       "site_order": {"sites": tuple(reversed(receipt.plan.sites))},
                       "site_id": {"sites": (replace(receipt.plan.sites[0], site_id="f" * 64), *receipt.plan.sites[1:])}}
            receipt._plan = replace(receipt.plan, **changes[kind])
    h.prepare_edit = edit
    result = h.run()
    assert result.status == "inconclusive" and len(h.prepared) == 3
    assert h.frames == [] and h.matches == [] and "evidence" not in h.publication.artifacts
    no_completion(h, result)


@pytest.mark.parametrize("boundary", ["metadata", "read", "inspect_diagnostic", "decode_rgb", "ingress",
    "decode_mask", "usable", "prepare", "extract", "match", "supervisor_run", "checkpoint",
    "artifact_bytes", "evidence", "readback", "complete"])
def test_independent_failure_at_each_injected_boundary(harness, boundary):
    h = harness
    h.inject = boundary
    result = h.run()
    assert result.status == "incomplete" and result.code == "injected_boundary_failure"
    assert h.supervisor.stopped == [result.code]
    assert h.publication.failures[-1]["status"] == "incomplete"
    no_completion(h, result)


@pytest.mark.parametrize("field,value,code", [
    ("status", "terminated", "supervisor_resource_stop"),
    ("wall_seconds", 181, "supervisor_timeout"), ("wall_seconds", float("nan"), "supervisor_timeout"),
    ("committed_bytes", 2 * 1024 ** 3 + 1, "supervisor_memory"),
    ("log_bytes", 65537, "supervisor_log")])
@pytest.mark.parametrize("stage", ["preaccess", "extract_0", "match_B", "completion_commit"])
def test_supervisor_stops_and_termination(harness, field, value, code, stage):
    h = harness
    h.supervisor.telemetry[stage] = {field: value}
    result = h.run()
    assert result.code == code and h.supervisor.stopped == [code]
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["layout", "decoded_digest", "ingress", "mask", "packed", "usable", "diagnostic", "encoded"])
def test_source_boundary_identity_tampering_before_preparation(harness, kind):
    h = harness
    d = h.dependencies
    if kind == "layout":
        h.dependencies = replace(d, decode_rgb=lambda *a: np.zeros((2, 2, 3), np.uint8))
    elif kind == "decoded_digest":
        h.arrays[2][0][0, 0] = 1
    elif kind == "ingress":
        h.dependencies = replace(d, ingress_digest=lambda *a: "f" * 64)
    elif kind in ("mask", "packed", "usable"):
        key = {"mask": "full_native_bool_sha256", "packed": "packed_mask_sha256", "usable": "usable_mask_sha256"}[kind]
        h.request.bindings["sources"][1][key] = "f" * 64
        h.request.metadata["source_inventory"][2][key] = "f" * 64
        h.request.metadata["inventory_sha256"] = ev.sha(ev.canonical({
            "sources": h.request.metadata["source_inventory"],
            "label_associations": h.request.label_associations}))
        reseal_binding(h.request)
    elif kind == "diagnostic":
        h.dependencies = replace(d, read_payload=lambda kind, key, limit: b"x" * (limit - 1))
    else:
        original = d.read_payload
        h.dependencies = replace(d, read_payload=lambda kind, key, limit: original(kind, key, limit) + (b"x" if kind == "encoded" else b""))
    result = h.run()
    assert result.status == "incomplete" and h.prepared == []
    no_completion(h, result)


@pytest.mark.parametrize("kind", ["mutable", "shape", "plan", "refs", "table_digest"])
def test_extracted_table_boundary_refusal(harness, kind):
    h = harness
    def edit(frame):
        if kind == "mutable":
            frame.tiles.setflags(write=True)
        elif kind == "shape":
            return replace(frame, tiles=frame.tiles[:, :8])
        elif kind == "plan":
            return replace(frame, plan=replace(frame.plan))
        elif kind == "refs":
            return replace(frame, source_refs=frame.source_refs[:2])
        else:
            return replace(frame, frozen_sha256="f" * 64)
        return frame
    h.extract_edit = edit
    result = h.run()
    assert result.status == "incomplete" and h.matches == []
    no_completion(h, result)


@pytest.mark.parametrize("which", ["pixels", "source", "usable", "row176", "table", "central", "neighbors", "metadata", "runtime", "code_pin"])
@pytest.mark.parametrize("stage", ["match_B", "completion"])
def test_postflight_tamper_refuses_completion(harness, which, stage):
    h = harness
    def mutate():
        if which in ("pixels", "source", "usable"):
            a = h.arrays[2][("pixels", "source", "usable").index(which)]
            a.flat[0] = 1 if which == "pixels" else not a.flat[0]
        elif which == "row176":
            h.arrays[0][1].flat[0] = True
        elif which in ("table", "central", "neighbors"):
            a = getattr(h.frames[0], {"table": "tiles", "central": "central_zero", "neighbors": "neighbor_zero_counts"}[which])
            a.setflags(write=True)
            a.flat[0] = 2 if which != "central" else True
            a.setflags(write=False)
        elif which == "metadata":
            h.request.metadata["expected_supply"][1] = 38
        elif which == "runtime":
            h.request.runtime["implementation"] = "tampered"
        else:
            h.request.metadata["dependency_pins"]["evidence"] = "f" * 64
    h.supervisor.hooks[stage] = mutate
    result = h.run()
    assert result.status == "incomplete"
    no_completion(h, result)


@pytest.mark.parametrize("mode", ["short_write", "flush", "fsync", "link", "crash", "atomic_readback",
                                  "external_readback", "completion_short_write", "completion_crash"])
def test_transaction_failure_preserves_status_and_no_false_completion(harness, mode):
    h = harness
    h.publication.mode = mode
    h.publication.artifacts["unrelated"] = b"preserved"
    result = h.run()
    assert result.status == "incomplete" and h.publication.artifacts["unrelated"] == b"preserved"
    assert h.publication.partial is None
    if mode in ("short_write", "flush", "fsync", "link", "crash", "atomic_readback"):
        assert "evidence" not in h.publication.artifacts
    no_completion(h, result)


@pytest.mark.parametrize("collision", ["evidence", "partial", WORKER_CANDIDATE])
def test_no_clobber_collisions(harness, collision):
    h = harness
    if collision == "partial":
        h.publication.partial = b"existing-partial"
    else:
        h.publication.artifacts[collision] = b"existing"
    result = h.run()
    assert result.status == "incomplete" and not result.completion
    if collision == "partial":
        assert h.publication.partial == b"existing-partial"
    else:
        assert h.publication.artifacts[collision] == b"existing"


@pytest.mark.parametrize("field", ["encoded_length", "diagnostic_length"])
def test_payload_size_cap_plus_one_before_access(harness, field):
    h = harness
    if field == "encoded_length":
        h.request.bindings["sources"][0][field] = driver.ENCODED_CAP + 1
    else:
        h.request = replace(h.request, diagnostic_length=driver.DIAGNOSTIC_CAP + 1)
    result = h.run()
    assert result.code == field and h.events == []


@pytest.mark.parametrize("field,required", [
    ("footprint_predicates", 16800), ("footprint_operands", 77414400),
    ("extraction_windows", 4352), ("extraction_pixels", 10027008),
    ("gradient_sites", 7676928), ("derivation_elements", 1392640),
    ("comparisons", 1048576), ("distance_matrix_bytes", 8388608),
    ("component_contributions", 335544320), ("model_bytes", 10092544),
    ("evidence_bytes", 10092544), ("publication_bytes", 20381696),
    ("aggregate_bytes", 20381696)])
def test_reservation_exact_cap_and_one_unit_exhaustion_before_arrays(field, required):
    exact = replace(driver.FROZEN_LIMITS, **{field: required})
    driver.reservations((0, 256, 4096), limits=exact)
    exhausted = replace(exact, **{field: required - 1})
    with pytest.raises(scd.Refusal, match="reservation_"):
        driver.reservations((0, 256, 4096), limits=exhausted)


@pytest.mark.parametrize("counts,grid", [((0, 257, 2570), 16800), ((0, 37, 4097), 16800),
                                       ((0, 37, 2570), 16801), ((0, 256, 4096), 28000)])
def test_site_and_five_plan_reservations_refuse_without_array_work(counts, grid):
    with pytest.raises(scd.Refusal):
        driver.reservations(counts, grid)


def test_maximum_site_work_and_storage_reservation_arithmetic():
    r = driver.reservations((0, 256, 4096))
    assert r["extraction_windows"] == 4352 and r["extraction_pixels"] == 10027008
    assert r["gradient_sites"] == 7676928 and r["derivation_elements"] == 1392640
    assert r["comparisons_per_arm"] == 1048576 and r["distance_matrix_bytes"] == 8388608
    assert r["component_contributions"] == 335544320
    assert r["model_bytes"] == 10092544 and r["publication_bytes"] == 20381696


def test_work_refusal_counters_preserved_and_determinant_atomicity():
    nw = {"limit": 4096, "requested": 4097, "charged": 4096, "processed": 4096, "uncharged": 1, "unprocessed": 0}
    dw = closed_work(32768, 32768)
    arm = {"pairs": [None] * 20, "summary": {"status": "witness_budget_inconclusive",
        "comparison_work": closed_work(1048576, 95090),
        "witness": {"status": "budget_inconclusive", "node_work": nw, "determinant_work": dw}}}
    status, work = driver.arm_work(arm)
    assert status == "budget_inconclusive" and work["tuples"] == nw and work["determinants"] == dw
    arm["summary"]["witness"]["determinant_work"] = closed_work(32768, 32767)
    with pytest.raises(scd.Refusal, match="determinant_node_closure"):
        driver.arm_work(arm)


def test_publication_growth_cap_refuses_before_transaction(harness):
    h = harness
    h.publication.current = driver.FROZEN_LIMITS.publication_bytes
    result = h.run()
    assert result.code == "publication_growth" and "evidence" not in h.publication.artifacts
    no_completion(h, result)


def test_failure_sink_and_termination_exceptions_do_not_mask_typed_result(harness):
    h = harness
    h.actual_counts = (1, 37, 2570)
    def crash(*args):
        raise RuntimeError("failure sink unavailable")
    h.supervisor.terminate = crash
    h.publication.failure = crash
    result = h.run()
    assert result.status == "inconclusive" and result.code == "footprint_supply_mismatch"
    no_completion(h, result)


@pytest.mark.parametrize("boundary", ["valid", "consume"])
def test_capability_exception_precedes_all_access(harness, boundary):
    h = harness
    def crash(identity):
        raise RuntimeError("fake invalid capability")
    setattr(h.capability, boundary, crash)
    result = h.run()
    assert result.status == "refused" and result.code == "injected_boundary_failure"
    assert not result.consumed and not any(e[0] == "inventory" for e in h.events)


@pytest.mark.parametrize("kind", ["dependency", "feasibility", "contract", "source", "runtime", "binding", "frames", "endpoint", "order"])
def test_preflight_exact_identity_mismatch_refuses_without_accessor(harness, kind):
    h = harness
    if kind == "dependency":
        h.request.dependency_pins["evidence"] = "f" * 64
    elif kind == "feasibility":
        h.request = replace(h.request, feasibility_sha256="f" * 64)
    elif kind == "frames":
        h.request.bindings["sources"].append(copy.deepcopy(h.request.bindings["sources"][0]))
    elif kind == "endpoint":
        h.request.bindings["sources"][0]["endpoint"][1] = 176
    elif kind == "order":
        h.request.metadata["feasibility_identities"].reverse()
    else:
        key = {"contract": "contract_sha256", "source": "source_sha256", "runtime": "runtime_sha256", "binding": "binding_sha256"}[kind]
        h.request.bindings[key] = "f" * 64
    result = h.run()
    assert result.status == "refused" and h.events == []
    no_completion(h, result)


def test_matrix_cap_plus_one_refuses_before_compaction_and_releases(harness):
    h = harness
    def edit(match):
        match.distance_matrix = np.broadcast_to(np.zeros(1, np.float64), (1048577,))
    h.match_edit = edit
    result = h.run()
    assert result.code == "matrix_bytes" and h.matches[0].distance_matrix is None
    assert len(h.matches) == 1 and "evidence" not in h.publication.artifacts
    no_completion(h, result)


def test_model_cap_plus_one_refuses_before_publication(harness, monkeypatch):
    h = harness
    class LengthSentinel:
        def __len__(self):
            return 10092545
    monkeypatch.setattr(ev, "dumps", lambda record: LengthSentinel())
    result = h.run()
    assert result.code == "evidence_model_bytes" and "evidence" not in h.publication.artifacts
    no_completion(h, result)


def test_completed_arm_counters_survive_next_arm_stop(harness):
    h = harness
    h.supervisor.telemetry["match_B"] = {"wall_seconds": 181}
    result = h.run()
    assert result.code == "supervisor_timeout" and set(result.counters["arms"]) == {"A"}
    assert result.counters["arms"]["A"]["comparisons"]["processed"] == 95090
    assert result.arm_dispositions == ("complete_no_witness",)
    no_completion(h, result)


def test_insufficient_shared_supply_is_typed_inconclusive_with_evidence(harness):
    h = harness
    h.empty_domains = True
    result = h.run()
    assert result.status == "inconclusive" and result.code == "arm_inconclusive"
    assert result.arm_dispositions == ("insufficient_support", "insufficient_support")
    assert "evidence" in h.publication.artifacts and result.evidence_receipt is not None
    assert result.counters["combined"]["comparisons"]["processed"] == 0
    no_completion(h, result)


def test_no_cli_hidden_paths_production_imports_or_compatibility_wrapper():
    source = inspect.getsource(driver)
    tree = ast.parse(source)
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert "scripts.evaluation" in imports and not any((n or "").startswith("mc_photogrammetry") for n in imports)
    assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main" for n in ast.walk(tree))
    forbidden = {"shared_frame", "plan_sites", "in_memory_inventory", "open", "read_bytes", "getenv", "Popen", "system", "eval", "exec"}
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
             for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert forbidden.isdisjoint(calls)
    assert not any(isinstance(n, ast.Import) and any(a.name in ("pickle", "multiprocessing", "subprocess")
                   for a in n.names) for n in ast.walk(tree))
    own = ast.parse(inspect.getsource(inspect.getmodule(test_no_cli_hidden_paths_production_imports_or_compatibility_wrapper)))
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("mc_photogrammetry", "tests.test_spatial")) for n in ast.walk(own))
    assert "if __name__" not in source
