"""Developer-only exact seed-predicate attribution; no recognition changes."""
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
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from mc_photogrammetry.reconstruction import region_growth as growth

_spec = importlib.util.spec_from_file_location(
    "_proposal_attrition_g7", Path(__file__).with_name("diagnose_g7_selection_failure.py"))
g7 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g7
_spec.loader.exec_module(g7)

SCHEMA = "mc-photogrammetry-seed-gate-attribution-v1"
PRODUCER = "seed-gate-attribution-1"
TASK = "DS-SEED-GATE-ATTRIBUTION-C9-01"
TASK_SHA256 = "46434dadc2d2099bd183992941573271db2fe3e998086923633a2bdff875c8be"
MAX_RECORDS = 16384
MAX_BYTES = 4 * 1024 * 1024
MAX_NODES = 65536
MAX_DEPTH = 16
STREAM_LIMITS = dict(live_identities=16384, bindings=8192, aggregate_buckets=4096,
    events=262144, retained_bytes=MAX_BYTES, sidecar_bytes=MAX_BYTES,
    identity_bytes=2048, identity_depth=8, identity_nodes=128,
    closed_frame_scopes=32, counter_max=2**63 - 1)
EARLY_STAGES = {"frame", "seed", "proposal", "reading", *growth._DETAILED_SCALAR_STAGES}
SOURCE_PINS = {
    "scripts/evaluation/diagnose_g7_selection_failure.py": "97bd1bac570074d4ff96692b17a236ebf06d79eca0338f556d9c4c440132aa1d",
    "src/mc_photogrammetry/reconstruction/region_growth.py": "abfc4c10cdfff0b1d6b5bc9dc315b240ec4c94521d915a49fb36c84407dc957c",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py": "fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py": "1be1c60ca8184e7cd1c62e7bfe11fcfea60c5e52e2aa080e8495cd3ea1035c04",
    "src/mc_photogrammetry/pipeline/image_derived_production_delivery.py": "4305c7156a451224523c4d691c933539c741b874569c5fe5341802061cb202e8",
    "scripts/evaluation/diagnose_image_proposal_attrition.py": "9ce9b38fbef87659bb8cbd037ca415832d7bd6b68935b51312ae0916a0449512",
    **g7.SOURCE_PINS,
}
NOTES = ["Aggregate scalar measurements only; no pixels, masks, positions or manual hints.",
         "G7 reasons count first failed predicates, not every defect.",
         "Partial execution-seam source pins; not a complete environment identity.",
         "Stream retains exact stage/axis counts and seed metrics; other scalar values enter ordered digests, not detail/example rows.",
         "Observer bytes measure charged retained diagnostic state, not process RSS; independent identity/binding/bucket/event limits also apply.",
         "Seed attribution is fixed16-mask aggregate with diagnostic precedence; it does not prove visibility or justify thresholds."]


class AttritionError(RuntimeError):
    def __init__(self, reason, field=""):
        self.reason, self.field = reason, field
        super().__init__(f"{reason}: {field}")


def canonical_bytes(value, *, byte_limit=MAX_BYTES, node_limit=MAX_NODES, depth_limit=MAX_DEPTH):
    """Charge escaped JSON before encoding; no changes to the G7 serializer."""
    stack, nodes, size = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > node_limit or depth > depth_limit:
            raise AttritionError("sidecar_budget", "structure")
        if type(item) in (dict, list, tuple):
            if nodes + len(stack) + len(item) > node_limit:
                raise AttritionError("sidecar_budget", "items")
            if type(item) is dict:
                if any(type(k) is not str for k in item):
                    raise AttritionError("malformed_sidecar", "keys")
                for key in item:
                    size += g7._escaped_string_size(key) + 1
                    if size > byte_limit:
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
        if size > byte_limit:
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


class _BaseStreamingObserver:
    """Diagnostic-only aggregates, exact live ledgers, and closed-frame epochs.

    Canonical string dictionaries preserve private hook membership/iteration.
    No raw examples or identity-specific period detail rows are retained.
    Early ledgers are released only after exact frame closure. Global signature
    and pair ledgers remain intact for downstream universe/product checks.
    """
    def __init__(self, *, limits=None):
        self.limits = dict(STREAM_LIMITS)
        if limits is not None:
            if type(limits) is not dict or set(limits) - set(self.limits):
                raise AttritionError("observer_budget", "limits")
            for key, value in limits.items():
                if type(value) is not int or not 0 < value <= self.limits[key]:
                    raise AttritionError("observer_budget", key)
                self.limits[key] = value
        self.limits = MappingProxyType(self.limits)
        self.max_items = self.limits["live_identities"]
        self.max_bytes = self.limits["retained_bytes"]
        self.bytes = self.peak_bytes = self.events = self.omitted = 0
        self.incomplete = False
        self.denials = {}
        self.stages, self.bindings, self.checks = {}, {}, {}
        self._binding_costs = {}
        self.active_frames, self.closed_frames = set(), set()
        self.peak_identities = self.peak_bindings = self.peak_buckets = 0

    def _deny(self, cause):
        self.incomplete = True
        self.omitted = min(self.limits["counter_max"], self.omitted + 1)
        # Fixed vocabulary only; denial reporting cannot grow with bad inputs.
        cause = cause if cause in STREAM_LIMITS or cause in {
            "missing", "duplicate", "unexpected", "late", "scope", "scalar", "checks"} else "scalar"
        self.denials[cause] = min(self.limits["counter_max"], self.denials.get(cause, 0) + 1)

    def _event(self):
        if self.events >= self.limits["counter_max"]:
            self._deny("counter_max")
            return False
        if self.events >= self.limits["events"]:
            self._deny("events")
            return False
        self.events += 1
        if self.incomplete:
            return False
        return True

    def _encode(self, value):
        return canonical_bytes(value, byte_limit=self.limits["identity_bytes"],
            node_limit=self.limits["identity_nodes"], depth_limit=self.limits["identity_depth"])

    def _live(self):
        return sum(len(s["expected"]) for s in self.stages.values())

    def _buckets(self):
        return sum(len(s["counts"]) for s in self.stages.values())

    def _reserve(self, size, *, identities=0, bindings=0, buckets=0):
        counts = dict(live_identities=self._live() + identities,
                      bindings=len(self.bindings) + bindings,
                      aggregate_buckets=self._buckets() + buckets,
                      retained_bytes=self.bytes + size)
        if self.incomplete:
            return False
        for field, value in counts.items():
            if value > self.limits[field]:
                self._deny(field)
                return False
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.peak_identities = max(self.peak_identities, counts["live_identities"])
        self.peak_bindings = max(self.peak_bindings, counts["bindings"])
        self.peak_buckets = max(self.peak_buckets, counts["aggregate_buckets"])
        return True

    def _stage(self, stage):
        if stage not in {"proposal", "reading", "signature", "pair", "seed", "frame", *growth._DETAILED_SCALAR_STAGES}:
            raise AttritionError("malformed_observer", "stage")
        if stage not in self.stages:
            if not self._reserve(256):
                raise AttritionError("observer_incomplete", "stage")
            self.stages[stage] = dict(expected={}, terminal={}, counts={}, digest=hashlib.sha256(),
                expected_cost={}, terminal_cost={}, evicted_expected=0, evicted_terminal=0,
                duplicates=0, unexpected=0)
        return self.stages[stage]

    def begin_frame(self, frame):
        if not self._event():
            return
        try:
            self._encode(frame)
            if type(frame) is not str or not frame or frame in self.closed_frames or frame in self.active_frames:
                self._deny("scope")
                return
            if len(self.active_frames) + len(self.closed_frames) >= self.limits["closed_frame_scopes"]:
                self._deny("closed_frame_scopes")
                return
            if self._reserve(128 + len(frame.encode("utf-8"))):
                self.active_frames.add(frame)
        except Exception:
            self._deny("scalar")

    def end_frame(self, frame):
        if not self._event():
            return
        if frame not in self.active_frames:
            self._deny("scope")
            return
        for name, stage in self.stages.items():
            if name not in EARLY_STAGES:
                continue
            keys = [key for key, identity in stage["expected"].items() if identity[0] == frame]
            if any(key not in stage["terminal"] for key in keys):
                self._deny("missing")
                return
        # This frame's raw ledgers and transient bindings are no longer read by
        # proposal/reader hooks. Aggregates/digests and closed-frame names remain.
        for name, stage in self.stages.items():
            if name not in EARLY_STAGES:
                continue
            keys = [key for key, identity in stage["expected"].items() if identity[0] == frame]
            for key in keys:
                self.bytes -= stage["expected_cost"].pop(key) + stage["terminal_cost"].pop(key)
                stage["expected"].pop(key)
                stage["terminal"].pop(key)
            stage["evicted_expected"] += len(keys)
            stage["evicted_terminal"] += len(keys)
        for key in tuple(self.bindings):
            if key[0] in ("proposal", "candidate") and key[1] == frame:
                self.bytes -= self._binding_costs.pop(key)
                self.bindings.pop(key)
        self.active_frames.remove(frame)
        self.closed_frames.add(frame)

    def _identity(self, stage, identity):
        if type(identity) not in (tuple, list) or not identity or type(identity[0]) is not str:
            raise AttritionError("malformed_observer", "identity")
        key = self._encode(identity).decode("ascii")
        if stage in EARLY_STAGES and identity[0] not in self.active_frames:
            self._deny("late" if identity[0] in self.closed_frames else "scope")
            return None
        return key

    def expect(self, stage, identity):
        if not self._event():
            return
        try:
            key = self._identity(stage, identity)
            if key is None:
                return
            data = self._stage(stage)
            if key in data["expected"]:
                data["duplicates"] += 1
                self._deny("duplicate")
                return
            size = 192 + 2 * len(key)
            if self._reserve(size, identities=1):
                data["expected"][key] = identity
                data["expected_cost"][key] = size
        except Exception:
            self._deny("scalar")

    def finish(self, stage, identity, disposition, **fields):
        if not self._event():
            return
        try:
            key = self._identity(stage, identity)
            if key is None:
                return
            row = dict(identity=identity, disposition=disposition, **fields)
            encoded = self._encode(row)
            if type(disposition) is not str or not disposition or len(disposition) > 128:
                raise AttritionError("malformed_observer", "disposition")
            data = self._stage(stage)
            if key not in data["expected"]:
                data["unexpected"] += 1
                self._deny("unexpected")
                return
            if key in data["terminal"]:
                data["duplicates"] += 1
                self._deny("duplicate")
                return
            group = self._encode((identity[0], fields.get("axis"), disposition,
                identity[4] if stage == "pair" else None, fields.get("right_axis"),
                fields.get("subcause"), fields.get("signal_failure_operand"))).decode("ascii")
            if group not in data["counts"]:
                if not self._reserve(256 + len(group), buckets=1):
                    return
                data["counts"][group] = dict(count=0, metrics={})
            metric = fields.get("eligible_seed_count") if stage == "seed" else None
            aggregate = data["counts"][group]
            if aggregate["count"] >= self.limits["counter_max"]:
                self._deny("counter_max")
                return
            if metric is not None:
                if type(metric) is not int or not 0 <= metric <= self.limits["counter_max"]:
                    self._deny("counter_max")
                    return
                prior = aggregate["metrics"].get("eligible_seed_count")
                if prior is None:
                    if not self._reserve(192):
                        return
                    aggregate["metrics"]["eligible_seed_count"] = dict(count=0, minimum=metric, maximum=metric, total=0)
                    prior = aggregate["metrics"]["eligible_seed_count"]
                if prior["total"] > self.limits["counter_max"] - metric:
                    self._deny("counter_max")
                    return
            size = 96 + len(key) + len(disposition)
            if not self._reserve(size):
                return
            data["terminal"][key] = disposition
            data["terminal_cost"][key] = size
            aggregate["count"] += 1
            if metric is not None:
                prior.update(count=prior["count"] + 1, minimum=min(prior["minimum"], metric),
                             maximum=max(prior["maximum"], metric), total=prior["total"] + metric)
            data["digest"].update(encoded + b"\n")
            if stage == "pair":
                # Deferred admissions finish after the matcher reads these
                # bindings; rejected traces have no later source consumer.
                for binding in (("defer", tuple(identity)), ("pair_metrics", tuple(identity))):
                    if binding in self.bindings:
                        self.bytes -= self._binding_costs.pop(binding)
                        self.bindings.pop(binding)
        except Exception:
            self._deny("scalar")

    def bind(self, key, identity):
        if not self._event():
            return
        try:
            if type(key) is not tuple or not key:
                raise AttritionError("malformed_observer", "binding")
            if key in self.bindings:
                if self.bindings[key] != identity:
                    self._deny("duplicate")
                return
            size = 128 + len(self._encode(key)) + len(self._encode(identity))
            if self._reserve(size, bindings=1):
                self.bindings[key] = identity
                self._binding_costs[key] = size
        except Exception:
            self._deny("scalar")

    def pending(self, stage, frame=None):
        data = self._stage(stage)
        return tuple(identity for key, identity in data["expected"].items()
                     if key not in data["terminal"] and (frame is None or identity[0] == frame))

    def check(self, name, condition):
        if not self._event():
            return
        try:
            encoded = self._encode(name)
            if type(name) is not str or len(name) > 256:
                raise AttritionError("malformed_observer", "check")
            if name not in self.checks and not self._reserve(128 + len(encoded)):
                return
            self.checks[name] = bool(condition) and self.checks.get(name, True)
            if not self.checks[name]:
                self._deny("checks")
        except Exception:
            self._deny("scalar")

    def record(self):
        stages = {}
        for name, data in sorted(self.stages.items()):
            expected, terminal = set(data["expected"]), set(data["terminal"])
            stages[name] = dict(expected=len(expected) + data["evicted_expected"],
                classified=len(terminal) + data["evicted_terminal"],
                unclassified=len(expected - terminal), unexpected=len(terminal - expected),
                multiply_classified=data["duplicates"], unexpected_events=data["unexpected"],
                counts=[dict(frame=json.loads(key)[0], axis=json.loads(key)[1],
                    disposition=json.loads(key)[2], **value) for key, value in sorted(data["counts"].items())],
                terminal_sha256=data["digest"].hexdigest())
        handoffs = [dict(generated_identity=(key[1], key[2]), returned_identity=value)
                    for key, value in self.bindings.items() if key[0] == "proposal_return"]
        handoffs.sort(key=lambda item: item["generated_identity"])
        closed = bool(stages) and not self.active_frames and all(not s[k] for s in stages.values()
            for k in ("unclassified", "unexpected", "multiply_classified", "unexpected_events"))
        return dict(schema="mc-photogrammetry-scalar-attrition-stream-v1",
            status="complete" if closed and not self.incomplete and all(self.checks.values()) else "incomplete",
            stages=stages, checks=dict(sorted(self.checks.items())), proposal_handoffs=handoffs,
            omitted_accounting_events=self.omitted, omitted_proposal_handoffs=0,
            resources=dict(events=self.events, live_identities=self._live(), peak_identities=self.peak_identities,
                bindings=len(self.bindings), peak_bindings=self.peak_bindings, aggregate_buckets=self._buckets(),
                peak_buckets=self.peak_buckets, retained_bytes=self.bytes, peak_bytes=self.peak_bytes,
                active_frames=sorted(self.active_frames), closed_frames=sorted(self.closed_frames), denials=dict(self.denials)),
            limits=dict(self.limits))


SEED_LIMITS = dict(frames=32, records=192, atoms_per_record=20000,
    atom_visits=3840000, predicate_visits=15360000, histogram_bins=16,
    counter_max=2**63 - 1, aggregate_bytes=262144, chunk_atoms=256)
SEED_ORDER = ["score", "margin", "support", "side"]
# Default-only diagnostic; these values come from the hash-pinned source config.
SEED_THRESHOLDS = tuple(getattr(growth.DEFAULT_REGION_GROWTH_CONFIG, name)
                        for name in ("min_seed_score", "min_seed_margin", "min_seed_support"))


def seed_counts(histogram):
    return dict(passes=[sum(n for mask, n in enumerate(histogram) if mask & (1 << bit)) for bit in range(4)],
        cumulative=[sum(n for mask, n in enumerate(histogram) if mask & ((1 << bits) - 1) == (1 << bits) - 1) for bits in range(1, 5)],
        first_failures=[sum(n for mask, n in enumerate(histogram) if mask & ((1 << bit) - 1) == (1 << bit) - 1 and not mask & (1 << bit)) for bit in range(4)],
        all_pass=histogram[15])


class StreamingObserver(_BaseStreamingObserver):
    """Separate fixed seed aggregates; never retain source arrays."""
    def __init__(self, *, seed_limits=None, **kwargs):
        super().__init__(**kwargs)
        values = dict(SEED_LIMITS)
        if seed_limits is not None:
            if type(seed_limits) is not dict or not set(seed_limits).issubset(values):
                raise AttritionError("seed_budget", "limits")
            for name, value in seed_limits.items():
                if (type(value) is not int or not 0 < value <= values[name]
                        or name == "histogram_bins" and value != 16):
                    raise AttritionError("seed_budget", "limits")
                values[name] = value
        self.seed_limits = MappingProxyType(values)
        self.seed_rows, self.seed_scopes = {}, set()
        self.seed_bytes = self.atom_visits = self.predicate_visits = 0

    def begin_frame(self, frame):
        if self.incomplete:
            return
        if type(frame) is not str or not frame or frame in self.seed_scopes:
            self._deny("seed_scope")
            return
        try:
            cost = 6 * (1024 + len(self._encode(frame)))
        except Exception:
            self._deny("seed_scalar")
            return
        if (len(self.seed_scopes) >= self.seed_limits["frames"]
                or len(self.seed_rows) + 6 > self.seed_limits["records"]
                or self.seed_bytes + cost > self.seed_limits["aggregate_bytes"]):
            self._deny("seed_budget")
            return
        super().begin_frame(frame)
        if self.incomplete:
            return
        self.seed_bytes += cost
        self.seed_scopes.add(frame)
        for axis in range(3):
            for side in (-1, 1):
                self.seed_rows[(frame, axis, side)] = dict(frame=frame, axis=axis, side=side,
                    state="pending", source_disposition=None, reason="", denominator=None,
                    histogram=None, counts=None, eligible_seed_count=None, thresholds=None)

    def finish(self, stage, identity, disposition, **fields):
        super().finish(stage, identity, disposition, **fields)
        if self.incomplete or stage != "seed":
            return
        key = tuple(identity)
        row = self.seed_rows.get(key)
        if row is None or row["source_disposition"] is not None:
            self._deny("seed_source")
            return
        row["source_disposition"] = disposition
        row["eligible_seed_count"] = fields.get("eligible_seed_count")
        if disposition == "not_reached":
            reason = fields.get("reason")
            if type(reason) is not str or not reason or len(self._encode(reason)) > 256:
                self._deny("seed_source")
                return
            row.update(state="not_reached", reason=reason)

    def seed_predicates(self, identity, *, count, best, other, seed_support, sides, seed, config):
        import numpy as np
        if self.incomplete:
            return
        key = tuple(identity)
        row = self.seed_rows.get(key)
        if row is None or row["state"] != "pending" or row["source_disposition"] not in ("available", "absent") or key[0] not in self.active_frames:
            self._deny("seed_source")
            return
        if type(count) is not int or not 0 <= count <= self.seed_limits["atoms_per_record"]:
            self._deny("seed_atoms")
            return
        if (count > self.seed_limits["atom_visits"] - self.atom_visits
                or 4 * count > self.seed_limits["predicate_visits"] - self.predicate_visits):
            self._deny("seed_work")
            return
        thresholds = [config.min_seed_score, config.min_seed_margin, config.min_seed_support]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in thresholds):
            self._deny("seed_threshold")
            return
        arrays = (best, other, seed_support, sides, seed)
        if any(type(value) is not np.ndarray or value.ndim != 1 or len(value) != count for value in arrays) or seed.dtype != np.bool_:
            self._deny("seed_source")
            return
        self.atom_visits += count
        self.predicate_visits += 4 * count
        histogram = [0] * 16
        for start in range(0, count, self.seed_limits["chunk_atoms"]):
            stop = min(count, start + self.seed_limits["chunk_atoms"])
            predicates = (best[start:stop] >= thresholds[0],
                          (best[start:stop] - other[start:stop]) >= thresholds[1],
                          seed_support[start:stop] >= thresholds[2], sides[start:stop] == key[2])
            mask = np.zeros(stop - start, dtype=np.uint8)
            for bit, predicate in enumerate(predicates):
                mask |= predicate.astype(np.uint8) << bit
            if not np.array_equal(mask == 15, seed[start:stop]):
                self._deny("seed_predicate_drift")
                return
            bins = np.bincount(mask, minlength=16)
            for index in range(16):
                n = int(bins[index])
                if n > self.seed_limits["counter_max"] - histogram[index]:
                    self._deny("seed_counter")
                    return
                histogram[index] += n
        counts = seed_counts(histogram)
        if sum(histogram) > self.seed_limits["counter_max"] or counts["all_pass"] != row["eligible_seed_count"]:
            self._deny("seed_counter_or_join")
            return
        row.update(state="reached", denominator=count, histogram=histogram,
                   counts=counts, thresholds=thresholds)

    def end_frame(self, frame):
        if any(row["state"] == "pending" for key, row in self.seed_rows.items() if key[0] == frame):
            self._deny("seed_missing")
        super().end_frame(frame)

    def seed_record(self):
        return dict(schema="mc-photogrammetry-seed-predicate-aggregate-v1", precedence=SEED_ORDER,
            precedence_meaning="diagnostic precedence; NumPy predicates eagerly evaluated",
            limits=dict(self.seed_limits), resources=dict(frames=len(self.seed_scopes), records=len(self.seed_rows),
                aggregate_bytes=self.seed_bytes, atom_visits=self.atom_visits, predicate_visits=self.predicate_visits),
            rows=[dict(row) for _, row in sorted(self.seed_rows.items())])


class DeliveryCapture:
    """The context is limited to the synchronous original delivery call."""
    def __init__(self, delivery, observer=None):
        self.delivery, self.calls = delivery, 0
        self.observer = observer if observer is not None else StreamingObserver()
        if (type(self.observer) is not StreamingObserver
                or not 0 < self.observer.max_items <= MAX_RECORDS
                or not 0 < self.observer.max_bytes <= MAX_BYTES):
            raise AttritionError("observer_budget", "limits")

    def __call__(self, *args, **kwargs):
        self.calls += 1
        from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
        original = measure._measure_frame
        def scoped(*args, **kwargs):
            frame = args[0] if args else kwargs["frame"]
            self.observer.begin_frame(str(frame.frame_id))
            try:
                return original(*args, **kwargs)
            finally:
                self.observer.end_frame(str(frame.frame_id))
        with growth._scalar_attrition_context(self.observer):
            measure._measure_frame = scoped
            try:
                return self.delivery(*args, **kwargs)
            finally:
                measure._measure_frame = original


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
    items = observer._live() if type(observer) is StreamingObserver else sum(
        len(s["expected"]) + len(s["terminal"]) + len(s["counts"])
        for s in observer.stages.values()) + len(observer.bindings)
    return dict(schema=raw["schema"], status=raw["status"], items=items, bytes=observer.bytes,
                overflow=observer.incomplete or observer.omitted > 0,
                omitted_accounting_events=raw["omitted_accounting_events"],
                omitted_proposal_handoffs=raw["omitted_proposal_handoffs"],
                checks=raw["checks"], stages=stages, proposal_handoffs=raw["proposal_handoffs"],
                limits=dict(observer.limits) if type(observer) is StreamingObserver else dict(items=observer.max_items, bytes=observer.max_bytes),
                resources=raw.get("resources"))


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
        g7=binding, observer=observer, seed_attribution=capture.observer.seed_record(),
        notes=NOTES))
    return validate_report(result, report)


def _integer(value):
    return type(value) is int and 0 <= value < 2**63


def validate_report(report, g7_report):
    """Explicit G7 binding validation, never matcher/selector/geometry replay."""
    canonical_bytes(report)
    fields = {"schema", "producer", "status", "task", "source_sha256", "runner_sha256",
              "inputs", "g7", "observer", "seed_attribution", "notes", "result_sha256"}
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
                "omitted_proposal_handoffs", "checks", "stages", "proposal_handoffs", "limits", "resources"}
    if type(observer) is not dict or set(observer) != required or observer["schema"] != "mc-photogrammetry-scalar-attrition-stream-v1" or observer["status"] != "complete" or observer["overflow"] is not False:
        raise AttritionError("observer_incomplete", "identity")
    if (not _integer(observer["items"]) or not _integer(observer["bytes"])
            or observer["limits"] != STREAM_LIMITS
            or observer["items"] > MAX_RECORDS or observer["bytes"] > MAX_BYTES
            or observer["omitted_accounting_events"] != 0 or observer["omitted_proposal_handoffs"] != 0):
        raise AttritionError("observer_incomplete", "budget")
    if type(observer["checks"]) is not dict or not observer["checks"] or any(value is not True for value in observer["checks"].values()):
        raise AttritionError("observer_incomplete", "handoff_checks")
    stages, frames = observer["stages"], set(report["g7"]["frame_ids"])
    if type(stages) is not dict or not {"seed", "frame", "signature", "pair"}.issubset(stages):
        raise AttritionError("observer_incomplete", "stages")
    allowed = {"proposal", "reading", "signature", "pair", "seed", "frame", *growth._DETAILED_SCALAR_STAGES}
    for name, stage in stages.items():
        if name not in allowed or type(stage) is not dict or set(stage) != {"expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events", "rows", "terminal_sha256"}:
            raise AttritionError("malformed_sidecar", "stage")
        if not all(_integer(stage[k]) for k in ("expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events")):
            raise AttritionError("malformed_sidecar", "counts")
        if stage["expected"] > STREAM_LIMITS["events"] or stage["classified"] > STREAM_LIMITS["events"]:
            raise AttritionError("observer_incomplete", "stage_budget")
        if stage["expected"] != stage["classified"] or any(stage[k] for k in ("unclassified", "unexpected", "multiply_classified", "unexpected_events")) or not g7.selection._hex(stage["terminal_sha256"]):
            raise AttritionError("observer_incomplete", "stage_closure")
        if type(stage["rows"]) is not list or len(stage["rows"]) > STREAM_LIMITS["aggregate_buckets"]:
            raise AttritionError("malformed_sidecar", "rows")
        for row in stage["rows"]:
            keys = {"frame", "axis", "disposition", "count"} | ({"eligible_seed_count"} if name == "seed" else set())
            if type(row) is not dict or set(row) != keys or row["frame"] not in frames or not (row["axis"] is None or type(row["axis"]) is int and row["axis"] in (0, 1, 2)) or type(row["disposition"]) is not str or not _integer(row["count"]) or not 0 < row["count"] <= STREAM_LIMITS["events"]:
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
    # Check presence is part of the source-hook contract, not just truth of
    # whichever checks a resealed record happens to retain.
    checks = observer["checks"]
    mandatory = {"reading_universe", "signature_universe", "pair_product"}
    for frame in frames:
        if any(row["frame"] == frame for row in stages.get("proposal", {}).get("rows", [])):
            mandatory.update(prefix + ":" + frame for prefix in
                ("proposal_generation", "proposal_returned", "proposal_to_reading"))
    if report["g7"]["counts"]["candidates"] or any(row["disposition"] in
            ("selected_admitted", "same_axis_dropped") for row in stages["pair"]["rows"]):
        mandatory.add("pair_admission_traces")
    if not mandatory.issubset(checks):
        raise AttritionError("observer_incomplete", "required_source_checks")
    built = [row for row in stages["signature"]["rows"] if row["disposition"] == "built"]
    if sum(row["count"] for row in built) != report["g7"]["counts"]["signatures"]:
        raise AttritionError("binding_mismatch", "signature_universe_count")
    # Failure G7 records retain independently validated endpoint incidence.
    # Compact success records retain only source totals: do not invent detail.
    if g7_report["graph"]["kind"] == "failure":
        incidence = g7_report["graph"]["endpoint_incidence"]
        for frame in frames:
            for axis in range(3):
                observed = sum(row["count"] for row in built
                    if row["frame"] == frame and row["axis"] == axis)
                source = sum(row["signatures"] for row in incidence
                    if row["frame"] == frame and row["local_axis"] == axis)
                if observed != source:
                    raise AttritionError("binding_mismatch", "signature_frame_axis")
    signature_counts = [sum(row["count"] for row in built if row["frame"] == frame)
                        for frame in sorted(frames)]
    expected_pairs = 4 * sum(left * right for i, left in enumerate(signature_counts)
                             for right in signature_counts[i + 1:])
    if stages["pair"]["expected"] != expected_pairs:
        raise AttritionError("binding_mismatch", "pair_product_count")
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
    resources = observer["resources"]
    resource_fields = {"events", "live_identities", "peak_identities", "bindings", "peak_bindings",
        "aggregate_buckets", "peak_buckets", "retained_bytes", "peak_bytes", "active_frames", "closed_frames", "denials"}
    if type(resources) is not dict or set(resources) != resource_fields or resources["active_frames"] != [] or resources["denials"] != {}:
        raise AttritionError("observer_incomplete", "resource_state")
    for field, cap in (("events", "events"), ("live_identities", "live_identities"),
        ("peak_identities", "live_identities"), ("bindings", "bindings"), ("peak_bindings", "bindings"),
        ("aggregate_buckets", "aggregate_buckets"), ("peak_buckets", "aggregate_buckets"),
        ("retained_bytes", "retained_bytes"), ("peak_bytes", "retained_bytes")):
        if not _integer(resources[field]) or resources[field] > STREAM_LIMITS[cap]:
            raise AttritionError("observer_incomplete", "resource_budget")
    if (type(resources["closed_frames"]) is not list or len(resources["closed_frames"]) > STREAM_LIMITS["closed_frame_scopes"]
            or resources["closed_frames"] != sorted(frames)
            or resources["live_identities"] != observer["items"] or resources["retained_bytes"] != observer["bytes"]
            or resources["aggregate_buckets"] != sum(len(s["rows"]) for s in stages.values())
            or resources["live_identities"] != sum(s["expected"] for name, s in stages.items() if name not in EARLY_STAGES)
            or resources["bindings"] < len(handoffs)
            or resources["events"] < sum(s["expected"] + s["classified"] for s in stages.values())):
        raise AttritionError("observer_incomplete", "resource_accounting")
    for live, peak in (("live_identities", "peak_identities"), ("bindings", "peak_bindings"),
                       ("aggregate_buckets", "peak_buckets"), ("retained_bytes", "peak_bytes")):
        if resources[live] > resources[peak]:
            raise AttritionError("observer_incomplete", "peak_accounting")
    accepted = sum(row["count"] for row in stages.get("reading", {}).get("rows", []) if row["disposition"] == "accepted")
    if accepted != report["g7"]["counts"]["readings"]:
        raise AttritionError("binding_mismatch", "accepted_readings")
    for row in stages.get("proposal", {}).get("rows", []):
        if not any(seed["frame"] == row["frame"] and seed["axis"] == row["axis"] and seed["disposition"] == "available" for seed in stages["seed"]["rows"]):
            raise AttritionError("binding_mismatch", "seed_proposal_join")
    validate_seed_record(report["seed_attribution"], observer, frames)
    if _seal(report)["result_sha256"] != report["result_sha256"]:
        raise AttritionError("binding_mismatch", "digest")
    return report


def validate_seed_record(record, observer, frames):
    required = {"schema", "precedence", "precedence_meaning", "limits", "resources", "rows"}
    if (type(record) is not dict or set(record) != required
            or record["schema"] != "mc-photogrammetry-seed-predicate-aggregate-v1"
            or record["precedence"] != SEED_ORDER
            or record["precedence_meaning"] != "diagnostic precedence; NumPy predicates eagerly evaluated"
            or record["limits"] != SEED_LIMITS):
        raise AttritionError("seed_incomplete", "identity")
    rows = record["rows"]
    if type(rows) is not list or len(rows) != 6 * len(frames) or len(rows) > SEED_LIMITS["records"]:
        raise AttritionError("seed_incomplete", "records")
    fields = {"frame", "axis", "side", "state", "source_disposition", "reason", "denominator",
              "histogram", "counts", "eligible_seed_count", "thresholds"}
    identities, work = [], 0
    for row in rows:
        if (type(row) is not dict or set(row) != fields or row["frame"] not in frames
                or type(row["axis"]) is not int or row["axis"] not in (0, 1, 2)
                or type(row["side"]) is not int or row["side"] not in (-1, 1)):
            raise AttritionError("seed_incomplete", "row")
        identities.append((row["frame"], row["axis"], row["side"]))
        if row["state"] == "not_reached":
            if (row["source_disposition"] != "not_reached" or type(row["reason"]) is not str
                    or not row["reason"] or len(canonical_bytes(row["reason"])) > 256
                    or any(row[name] is not None for name in
                           ("denominator", "histogram", "counts", "eligible_seed_count", "thresholds"))
                    or not any(item["frame"] == row["frame"] and item["disposition"] == row["reason"]
                               for item in observer["stages"]["frame"]["rows"])):
                raise AttritionError("seed_incomplete", "not_reached")
            continue
        n, histogram = row["denominator"], row["histogram"]
        if (row["state"] != "reached" or row["reason"] != "" or not _integer(n)
                or n > SEED_LIMITS["atoms_per_record"] or type(histogram) is not list or len(histogram) != 16
                or any(not _integer(v) or v > SEED_LIMITS["counter_max"] for v in histogram)
                or sum(histogram) != n):
            raise AttritionError("seed_incomplete", "histogram")
        counts = seed_counts(histogram)
        supplied = row["counts"]
        if (type(supplied) is not dict or set(supplied) != set(counts)
                or any(type(supplied[k]) is not list or len(supplied[k]) != 4
                       or any(not _integer(v) for v in supplied[k])
                       for k in ("passes", "cumulative", "first_failures"))
                or not _integer(supplied["all_pass"])):
            raise AttritionError("seed_incomplete", "count_types")
        if (supplied != counts or not _integer(row["eligible_seed_count"])
                or row["eligible_seed_count"] != counts["all_pass"]
                or row["source_disposition"] != ("available" if counts["all_pass"] else "absent")
                or sum(counts["first_failures"]) + counts["all_pass"] != n
                or any(a < b for a, b in zip(counts["cumulative"], counts["cumulative"][1:]))):
            raise AttritionError("seed_incomplete", "counts")
        if (type(row["thresholds"]) is not list or len(row["thresholds"]) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in row["thresholds"])
                or row["thresholds"] != list(SEED_THRESHOLDS)
                or any(type(v) is not type(expected)
                       for v, expected in zip(row["thresholds"], SEED_THRESHOLDS))):
            raise AttritionError("seed_incomplete", "thresholds")
        work += n
    expected = sorted((frame, axis, side) for frame in frames for axis in range(3) for side in (-1, 1))
    if identities != expected:
        raise AttritionError("seed_incomplete", "identity_closure")
    by_identity = {(row["frame"], row["axis"], row["side"]): row for row in rows}
    for frame in frames:
        frame_rows = [row for row in rows if row["frame"] == frame]
        if len({row["state"] for row in frame_rows}) != 1:
            raise AttritionError("seed_incomplete", "shared_frame_reachability")
        if (frame_rows[0]["state"] == "not_reached"
                and len({row["reason"] for row in frame_rows}) != 1):
            raise AttritionError("seed_incomplete", "shared_frame_terminal_reason")
        reached = [row for row in frame_rows if row["state"] == "reached"]
        if reached and any(row["denominator"] != reached[0]["denominator"] for row in reached):
            raise AttritionError("seed_incomplete", "shared_frame_denominator")
        for axis in range(3):
            plus, minus = by_identity[(frame, axis, 1)], by_identity[(frame, axis, -1)]
            if plus["state"] != minus["state"]:
                raise AttritionError("seed_incomplete", "paired_side_state")
            if plus["state"] != "reached":
                continue
            if plus["thresholds"] != minus["thresholds"]:
                raise AttritionError("seed_incomplete", "paired_side_thresholds")
            p, m = plus["histogram"], minus["histogram"]
            for base in range(8):
                # Source sides are np.sign values: zeros fail BOTH side tests.
                # Preserve that non-+/-1 mass, rather than assume complements.
                remainder = p[base] - m[base + 8]
                if remainder < 0 or remainder != m[base] - p[base + 8]:
                    raise AttritionError("seed_incomplete", "paired_side_mask_join")
    resources = record["resources"]
    cost = sum(6 * (1024 + len(canonical_bytes(frame))) for frame in frames)
    if (type(resources) is not dict or any(not _integer(v) for v in resources.values())
            or resources != dict(frames=len(frames), records=len(rows),
            aggregate_bytes=cost, atom_visits=work, predicate_visits=4 * work)
            or any(resources[k] > SEED_LIMITS[k] for k in resources)):
        raise AttritionError("seed_incomplete", "resource_accounting")
    for frame in frames:
        for axis in range(3):
            for disposition in ("available", "absent", "not_reached"):
                selected = [row for row in rows if row["frame"] == frame and row["axis"] == axis
                            and row["source_disposition"] == disposition]
                aggregates = [row for row in observer["stages"]["seed"]["rows"] if row["frame"] == frame
                              and row["axis"] == axis and row["disposition"] == disposition]
                if sum(row["count"] for row in aggregates) != len(selected) or len(aggregates) > 1:
                    raise AttritionError("seed_incomplete", "source_join")
                if selected:
                    if disposition == "not_reached":
                        metric = None
                    else:
                        values = [row["eligible_seed_count"] for row in selected]
                        metric = dict(count=len(values), minimum=min(values), maximum=max(values), total=sum(values))
                    if aggregates[0]["eligible_seed_count"] != metric:
                        raise AttritionError("seed_incomplete", "eligible_seed_join")
    return record


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
                 bytes=capture.observer.bytes, items=capture.observer._live(),
                 omitted=capture.observer.omitted, sidecar_published=error is None)
    state["resources"] = capture.observer.record()["resources"]
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
