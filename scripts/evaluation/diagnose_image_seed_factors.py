"""Diagnostic-only exact support-factor replay; no production adoption."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mc_photogrammetry.reconstruction import region_growth as growth

_spec = importlib.util.spec_from_file_location("_factor_c9", Path(__file__).with_name("diagnose_image_seed_gate.py"))
c9 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = c9
_spec.loader.exec_module(c9)

TASK = "DS-SEED-SUPPORT-FACTOR-REPLAY-01"
TASK_SHA256 = "0116f09fd945b278f0fa9b6ca558550c5903c867ac2e7c8fed01cb684a3ca1e7"
SCHEMA = "mc-photogrammetry-seed-factor-replay-v1"
PRODUCER = "seed-factor-replay-1"
CAPS = MappingProxyType(dict(frames=32, atoms_per_frame=20000, atoms=640000,
    unique_bins=16384, aggregate_bytes=8*1024*1024, predicate_work=61440000,
    sidecar_bytes=4*1024*1024, chunk_atoms=256, counter_max=2**63-1))
VARIANTS = ("baseline", "balance", "normal_suppression", "reliability")
NOTES = ["One optimistic factor set to1 at a time; original opposing scores and all other seed predicates retained.",
         "Attribution upper bounds only; no estimator, proposal, geometry or threshold recommendation.",
         "Raw source-dtype support/score/sign bytes grouped by tuple; no pixels, atom identities or positions.",
         "Aggregate bytes are conservative logical charges, not RSS; source pins cover execution seams."]
Error = c9.AttritionError
REFERENCE_GRAPH = "569458a09ba497ffc1475816b06b7525e6f0ccd87528e008875c03e71bfb1ad9"
REFERENCE_SELECTION = "a0b58962548f3c7c645bea355c13347f342cb14242e3aad65d6148c27099263a"
REFERENCE_CANDIDATES = "b2c7c92656e1f47be5ffc33c3d7c60c00525642c9b3517f09551eefc014c6554"
REFERENCE_COUNTS = dict(readings=16, signatures=50, candidates=100, selected=0)
REFERENCE_SEEDS = "839f61b5addfa3b057040b84d80cd23cbe65304671aad2a91d624ddbb88997c8"
REFERENCE_INPUTS = ('19e7399f449c339ac2dc38dd8f6b1c549e1616be010b0682bb15ca26d8d99389', 'aa33d6b6ddee52f071629c5eac3116ec6f82467ff1285285122f4549e94b67e4', '005e29a296bd096544100ca132d5ecc813451c60e7d9ef41a778323546ec3bf8', '8e774ca86d4b305e1e177879d2bfd1529ba64d77edab5831412b3c433540e9ad', '370f19e63486f7fa5d1a2835e3b621ed3f39fbd48b30b7447a2ec26e614b8148', '1e2319d0f01ef100e183aa8c8de5fb0751f9a67078681e49938cc6ec7e3d92d2')


def _hash(path):
    return c9.g7._file_hash(Path(path))


def source_binding():
    pins = dict(c9.SOURCE_PINS)
    pins["scripts/evaluation/diagnose_image_seed_gate.py"] = "95ec1e8e582172c0c0a3893fcd97639beb4c6f505a3ad305eacd441e07ec7841"
    for name, expected in pins.items():
        if _hash(ROOT / name) != expected:
            raise Error("source_pin_mismatch", name)
    for name in ("scripts/evaluation/diagnose_image_seed_factors.py", "tests/test_diagnose_image_seed_factors.py"):
        pins[name] = _hash(ROOT / name)
    return pins


def _int(value):
    return type(value) is int and 0 <= value <= CAPS["counter_max"]


def scores_from_support(supports, variant="baseline"):
    """Independent expression, same source dtypes/order and float32 output cast."""
    if variant not in VARIANTS:
        raise Error("malformed_factors", "variant")
    result = np.zeros_like(supports, dtype=np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for axis in range(3):
            a, b = [other for other in range(3) if other != axis]
            sa, sb, sc = supports[:, axis], supports[:, a], supports[:, b]
            in_plane = sb + sc
            total = in_plane + sa
            balance = 2.0 * np.sqrt(sb * sc) / np.maximum(in_plane, 1e-6)
            normal_suppression = in_plane / np.maximum(total, 1e-6)
            reliability = in_plane / (in_plane + 24.0)
            if variant == "balance": balance = np.ones_like(balance)
            if variant == "normal_suppression": normal_suppression = np.ones_like(normal_suppression)
            if variant == "reliability": reliability = np.ones_like(reliability)
            result[:, axis] = np.clip(balance * normal_suppression * reliability, 0.0, 1.0)
    return result


class FactorCollector:
    """Bounded tuple census. Diagnostic failures are sticky and never mutate graphs."""
    def __init__(self, frame_ids, *, limits=None):
        self.limits = dict(CAPS)
        if limits is not None:
            if type(limits) is not dict or set(limits)-set(CAPS):
                raise Error("factor_budget", "limits")
            for name, value in limits.items():
                if not _int(value) or value > CAPS[name] or name == "chunk_atoms" and value == 0:
                    raise Error("factor_budget", name)
                self.limits[name] = value
        self.limits = MappingProxyType(self.limits)
        ids = []
        for frame in frame_ids:
            if len(ids) == self.limits["frames"]:
                raise Error("factor_budget", "frames")
            if type(frame) is not str or not frame or len(c9.canonical_bytes(frame)) > 128 or frame in ids:
                raise Error("malformed_factors", "frame_identity")
            ids.append(frame)
        if not ids: raise Error("malformed_factors", "frames")
        self.frame_ids = tuple(ids)
        cost = sum(512 + len(c9.canonical_bytes(frame)) for frame in ids)
        if cost > self.limits["aggregate_bytes"]: raise Error("factor_budget", "aggregate_bytes")
        self.resources = dict(frames=len(ids), atoms=0, unique_bins=0, aggregate_bytes=cost, predicate_work=0)
        self.frames = {frame: dict(state="pending", dtype=None, atoms=0, bins={}) for frame in ids}
        self.error = None

    def refuse(self, reason, field):
        if self.error is None: self.error = Error(reason, field)

    def capture_graph(self, frame, graph):
        if self.error is not None: return
        try:
            if frame not in self.frames or self.frames[frame]["state"] != "pending":
                raise Error("factor_incomplete", "duplicate_or_unexpected_graph")
            data = self.frames[frame]
            if graph is None:
                data["state"] = "not_reached"
                return
            support, scores, sides = graph.supports, graph.plane_scores, graph.sides
            n = len(graph.node_area)
            if (type(support) is not np.ndarray or support.dtype.str not in ("<f4", "<f8")
                    or type(scores) is not np.ndarray or scores.dtype.str != "<f4"
                    or type(sides) is not np.ndarray or sides.dtype.str != "|i1"
                    or support.shape != (n,3) or scores.shape != (n,3) or sides.shape != (n,3)):
                raise Error("malformed_factors", "source_shapes_or_dtypes")
            if n > self.limits["atoms_per_frame"]:
                raise Error("factor_budget", "atoms_per_frame")
            for name, amount in (("atoms", n), ("predicate_work", 96*n)):
                if self.resources[name] > self.limits[name]-amount:
                    raise Error("factor_budget", name)
            # Reserve the full source census and all four six-side replay passes.
            self.resources["atoms"] += n
            self.resources["predicate_work"] += 96*n
            data.update(atoms=n, dtype=support.dtype.str)
            for start in range(0,n,self.limits["chunk_atoms"]):
                stop = min(n,start+self.limits["chunk_atoms"])
                if (not np.isfinite(support[start:stop]).all() or (support[start:stop] < 0).any()
                        or not np.isfinite(scores[start:stop]).all() or ((scores[start:stop] < 0) | (scores[start:stop] > 1)).any()
                        or not np.isin(sides[start:stop], (-1,0,1)).all()):
                    raise Error("malformed_factors", "source_values")
                for index in range(start,stop):
                    key = support[index].tobytes()+scores[index].tobytes()+sides[index].tobytes()
                    if key not in data["bins"]:
                        charge = 256 + 3*len(key) + len(c9.canonical_bytes(frame))
                        if self.resources["unique_bins"] == self.limits["unique_bins"]:
                            raise Error("factor_budget", "unique_bins")
                        if self.resources["aggregate_bytes"] > self.limits["aggregate_bytes"]-charge:
                            raise Error("factor_budget", "aggregate_bytes")
                        self.resources["unique_bins"] += 1
                        self.resources["aggregate_bytes"] += charge
                        data["bins"][key] = 0
                    if data["bins"][key] == self.limits["counter_max"]:
                        raise Error("factor_budget", "counter_max")
                    data["bins"][key] += 1
            data["state"] = "reached"
        except Exception as error:
            self.refuse(getattr(error,"reason","factor_capture_failed"),getattr(error,"field",type(error).__name__))

    @contextmanager
    def scoped_source(self):
        original = growth._build_graph
        def scoped(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                self.capture_graph(growth._attrition_frame.get(), result[0])
            except Exception as error:
                self.refuse("factor_capture_failed", type(error).__name__)
            return result
        growth._build_graph = scoped
        try:
            yield self
        finally:
            growth._build_graph = original

    def record(self, baseline_rows):
        if self.error is not None: raise self.error
        rows = []
        for frame in self.frame_ids:
            data = self.frames[frame]
            source = [row for row in baseline_rows if row["frame"] == frame]
            if len(source) != 6: raise Error("factor_incomplete", "seed_identity_closure")
            if data["state"] == "pending":
                if not all(row["state"] == "not_reached" for row in source):
                    raise Error("factor_incomplete", "missing_graph")
                state = "not_reached"
            else: state = data["state"]
            if any(row["state"] != state for row in source):
                raise Error("factor_incomplete", "graph_seed_state_join")
            rows.append(dict(frame=frame,state=state,dtype=data["dtype"],atoms=data["atoms"],
                bins=[[key.hex(),count] for key,count in sorted(data["bins"].items())]))
        return dict(limits=dict(CAPS),resources=dict(self.resources),frames=rows)


def _decode_bin(key, dtype):
    size = 3*np.dtype(dtype).itemsize
    raw = bytes.fromhex(key)
    if len(raw) != size+15 or raw.hex() != key:
        raise Error("malformed_factors", "tuple_bytes")
    support = np.frombuffer(raw[:size],dtype=dtype).reshape(1,3)
    scores = np.frombuffer(raw[size:size+12],dtype="<f4").reshape(1,3)
    sides = np.frombuffer(raw[size+12:],dtype="|i1").reshape(1,3)
    if (not np.isfinite(support).all() or (support<0).any() or not np.isfinite(scores).all()
            or ((scores<0)|(scores>1)).any() or not np.isin(sides,(-1,0,1)).all()):
        raise Error("malformed_factors", "tuple_values")
    return support,scores,sides


def replay(record, baseline_rows):
    """First prove every baseline row, then evaluate individual optimistic factors."""
    if type(record) is not dict or set(record) != {"limits","resources","frames"} or record["limits"] != dict(CAPS):
        raise Error("malformed_factors", "record")
    frames = record["frames"]
    if type(frames) is not list or not 1 <= len(frames) <= CAPS["frames"]:
        raise Error("factor_budget", "frames")
    ids = [frame.get("frame") for frame in frames if type(frame) is dict]
    expected_ids = sorted({row["frame"] for row in baseline_rows})
    if ids != expected_ids or len(baseline_rows) != 6*len(frames):
        raise Error("factor_incomplete", "frame_identity_closure")
    resources = dict(frames=len(frames),atoms=0,unique_bins=0,
        aggregate_bytes=sum(512+len(c9.canonical_bytes(frame)) for frame in ids),predicate_work=0)
    histograms = {name: {} for name in VARIANTS}
    for frame in frames:
        if set(frame) != {"frame","state","dtype","atoms","bins"} or not _int(frame["atoms"]):
            raise Error("malformed_factors", "frame")
        name,n,bins = frame["frame"],frame["atoms"],frame["bins"]
        if type(bins) is not list or len(bins) > CAPS["unique_bins"] or n > CAPS["atoms_per_frame"]:
            raise Error("factor_budget", "frame_bins")
        if frame["state"] == "not_reached":
            if n != 0 or bins or frame["dtype"] is not None:
                raise Error("malformed_factors", "not_reached")
            for variant in VARIANTS:
                for axis in range(3):
                    for side in (-1,1):histograms[variant][(name,axis,side)] = None
            continue
        if frame["state"] != "reached" or frame["dtype"] not in ("<f4","<f8"):
            raise Error("malformed_factors", "dtype_or_state")
        resources["atoms"] += n;resources["predicate_work"] += 96*n
        resources["unique_bins"] += len(bins)
        if any(resources[k] > CAPS[k] for k in resources):raise Error("factor_budget", "resources")
        previous,total = None,0
        for axis in range(3):
            for side in (-1,1):histograms["baseline"][(name,axis,side)] = [0]*16
        for item in bins:
            if (type(item) is not list or len(item) != 2 or type(item[0]) is not str
                    or len(item[0]) != (3*np.dtype(frame["dtype"]).itemsize+15)*2
                    or not _int(item[1]) or item[1] == 0 or item[1] > n
                    or previous is not None and item[0] <= previous):
                raise Error("malformed_factors", "canonical_bins")
            previous = item[0];total += item[1]
            resources["aggregate_bytes"] += 256 + 3*(len(item[0])//2) + len(c9.canonical_bytes(name))
            if total > n or resources["aggregate_bytes"] > CAPS["aggregate_bytes"]:
                raise Error("factor_budget", "aggregation")
            # Fixed scalar tuples only; no per-atom expansion by count.
            support,original,sides = _decode_bin(item[0],frame["dtype"])
            recomputed = scores_from_support(support)
            if recomputed.tobytes() != original.tobytes():
                raise Error("baseline_mismatch", "normal_score_bytes")
            _add_masks(histograms["baseline"],name,support,original,original,sides,item[1])
        if total != n:raise Error("factor_incomplete", "aggregation_conservation")
        # Do not expose any counterfactual until all frames' baselines pass below.
    if record["resources"] != resources or any(not _int(v) for v in record["resources"].values()):
        raise Error("factor_incomplete", "resource_accounting")
    for row in baseline_rows:
        identity=(row["frame"],row["axis"],row["side"])
        hist=histograms["baseline"].get(identity)
        if row["state"] == "not_reached":
            if hist is not None:raise Error("baseline_mismatch", "not_reached")
        elif (hist != row["histogram"] or sum(hist) != row["denominator"]
                or c9.seed_counts(hist) != row["counts"] or hist[15] != row["eligible_seed_count"]
                or row["thresholds"] != list(c9.SEED_THRESHOLDS)):
            raise Error("baseline_mismatch", "seed_histogram")
    for frame in frames:
        if frame["state"] != "reached":continue
        name=frame["frame"]
        for variant in VARIANTS[1:]:
            for axis in range(3):
                for side in (-1,1):histograms[variant][(name,axis,side)] = [0]*16
            for key,count in frame["bins"]:
                support,original,sides=_decode_bin(key,frame["dtype"])
                _add_masks(histograms[variant],name,support,scores_from_support(support,variant),original,sides,count)
    output={variant:[] for variant in VARIANTS}
    for variant in VARIANTS:
        for row in baseline_rows:
            key=(row["frame"],row["axis"],row["side"]);hist=histograms[variant][key]
            eligible=None if hist is None else hist[15]
            baseline=row["eligible_seed_count"]
            if eligible is not None and eligible < baseline:raise Error("baseline_mismatch", "non_optimistic")
            output[variant].append(dict(frame=key[0],axis=key[1],side=key[2],state=row["state"],
                denominator=row["denominator"],histogram=hist,eligible=eligible,
                newly_eligible=None if hist is None else eligible-baseline))
    return output


def _add_masks(histograms, frame, support, changed, original, sides, count):
    config=growth.DEFAULT_REGION_GROWTH_CONFIG
    for axis in range(3):
        tangent=[a for a in range(3) if a != axis]
        best=changed[:,axis]
        # The competing evidence is never counterfactually modified.
        other=np.max(np.delete(original,axis,axis=1),axis=1)
        total=support[:,tangent].sum(axis=1)
        first=int(best[0]>=config.min_seed_score)+2*int((best-other)[0]>=config.min_seed_margin)+4*int(total[0]>=config.min_seed_support)
        for side in (-1,1):
            mask=first+8*int(sides[0,axis]==side)
            histograms[(frame,axis,side)][mask] += count


def require_c9_reference(baseline, graph_report):
    """Future ordinary capture is fixed to accepted C8/C9 evidence, never a sidecar input."""
    graph=graph_report["graph"]
    if (tuple(item["sha256"] for item in baseline["inputs"]) != REFERENCE_INPUTS
            or graph["result_sha256"] != REFERENCE_GRAPH
            or graph["selection"]["result_sha256"] != REFERENCE_SELECTION
            or graph["selection"]["candidate_matches_sha256"] != REFERENCE_CANDIDATES
            or graph["counts"] != REFERENCE_COUNTS
            or hashlib.sha256(c9.canonical_bytes(baseline["seed_attribution"])).hexdigest() != REFERENCE_SEEDS):
        raise Error("reference_mismatch", "c8_c9_parity")


def _seal(report):
    report=dict(report);report.pop("result_sha256",None)
    report["result_sha256"]=hashlib.sha256(c9.canonical_bytes(report)).hexdigest()
    return report


def build_report(collector, baseline, graph_report, pins):
    c9.validate_report(baseline,graph_report)
    factors=collector.record(baseline["seed_attribution"]["rows"])
    results=replay(factors,baseline["seed_attribution"]["rows"])
    report=_seal(dict(schema=SCHEMA,producer=PRODUCER,status="complete",task_sha256=TASK_SHA256,
        source_sha256=pins,inputs=baseline["inputs"],baseline_sha256=baseline["result_sha256"],
        graph_sha256=baseline["g7"]["graph_sha256"],selection_sha256=baseline["g7"]["selection_sha256"],
        factors=factors,variants=results,notes=NOTES))
    return validate_report(report,baseline,graph_report)


def validate_report(report, baseline, graph_report):
    c9.validate_report(baseline,graph_report)
    required={"schema","producer","status","task_sha256","source_sha256","inputs","baseline_sha256",
        "graph_sha256","selection_sha256","factors","variants","notes","result_sha256"}
    if (type(report) is not dict or set(report)!=required or report["schema"]!=SCHEMA or report["producer"]!=PRODUCER
            or report["status"]!="complete" or report["task_sha256"]!=TASK_SHA256 or report["notes"]!=NOTES
            or report["source_sha256"]!=source_binding() or report!=_seal(report)):
        raise Error("factor_binding_mismatch","identity")
    if (report["inputs"]!=baseline["inputs"] or report["baseline_sha256"]!=baseline["result_sha256"]
            or report["graph_sha256"]!=baseline["g7"]["graph_sha256"]
            or report["selection_sha256"]!=baseline["g7"]["selection_sha256"]):
        raise Error("factor_binding_mismatch","source_join")
    expected=replay(report["factors"],baseline["seed_attribution"]["rows"])
    if report["variants"]!=expected or c9.canonical_bytes(report["variants"])!=c9.canonical_bytes(expected):
        raise Error("factor_binding_mismatch","replay")
    return report


def _readback(path, baseline, graph_report):
    with Path(path).open("rb") as stream:raw=stream.read(CAPS["sidecar_bytes"]+1)
    c9._predecode(raw)
    report=json.loads(raw)
    validate_report(report,baseline,graph_report)
    if c9.canonical_bytes(report)!=raw:raise Error("readback_mismatch","canonical")
    return raw


def publish_report(path, report, baseline, graph_report):
    path=Path(path);data=c9.canonical_bytes(validate_report(report,baseline,graph_report))
    if len(data) > CAPS["sidecar_bytes"]:
        raise Error("factor_budget", "sidecar_bytes")
    if path.exists():raise Error("output_collision","factors")
    temporary,published=None,False
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix=".seed-factors-",suffix=".tmp",dir=path.parent);temporary=Path(name)
        with os.fdopen(fd,"wb") as stream:
            if stream.write(data)!=len(data):raise Error("publication_failed","short_write")
            stream.flush();os.fsync(stream.fileno())
        if _readback(temporary,baseline,graph_report)!=data:raise Error("readback_mismatch","temporary")
        os.link(temporary,path);published=True
        if _readback(path,baseline,graph_report)!=data:raise Error("readback_mismatch","final")
    except FileExistsError as error:raise Error("output_collision","factors") from error
    finally:
        if temporary is not None:
            if published and sys.exc_info()[0] is not None and path.exists() and os.path.samefile(temporary,path):path.unlink()
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Outcome:
    seed_outcome: object
    report: dict | None
    diagnostic_error: Error | None


def run_once(images, *, project, destination, diagnostic, sidecar, factors, name, description, delivery=None):
    bounded=[]
    for image in images:
        if len(bounded)==CAPS["frames"]:raise Error("factor_budget","frames")
        bounded.append(Path(image))
    if not bounded:raise Error("malformed_factors","frames")
    outputs=[Path(p).resolve() for p in (project,destination,diagnostic,sidecar,factors)]
    if len(set(outputs))!=5 or any(p.exists() for p in outputs) or any(p.resolve() in outputs for p in bounded):
        raise Error("output_collision","outputs")
    pins=source_binding()
    inputs=[_hash(path) for path in bounded]
    if tuple(inputs) != REFERENCE_INPUTS:raise Error("reference_mismatch","ordered_inputs")
    collector=FactorCollector(f"image-{i:04d}" for i in range(len(bounded)))
    with collector.scoped_source():
        observed=c9.run_once(bounded,project=project,destination=destination,diagnostic=diagnostic,sidecar=sidecar,name=name,description=description,delivery=delivery)
    report,error=None,None
    try:
        if source_binding()!=pins or [_hash(path) for path in bounded]!=inputs:raise Error("source_pin_mismatch","postflight")
        if observed.diagnostic_error is not None:raise Error("factor_incomplete","seed_diagnostic")
        require_c9_reference(observed.report,observed.g7_outcome.report)
        report=build_report(collector,observed.report,observed.g7_outcome.report,pins)
        publish_report(factors,report,observed.report,observed.g7_outcome.report)
    except Exception as failure:
        error=failure if isinstance(failure,Error) else Error("factor_diagnostic_failed",type(failure).__name__)
    return Outcome(observed,report if error is None else None,error)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",action="append",type=Path,required=True)
    for key in ("project","destination","diagnostic","sidecar","factors"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--name",required=True);parser.add_argument("--description",required=True)
    args=parser.parse_args(argv)
    try:
        outcome=run_once(args.image,project=args.project,destination=args.destination,diagnostic=args.diagnostic,
            sidecar=args.sidecar,factors=args.factors,name=args.name,description=args.description)
        error=outcome.diagnostic_error
        print(json.dumps(dict(status="complete" if error is None else "refused",reason="" if error is None else error.reason,
            field="" if error is None else error.field,result_sha256=None if outcome.report is None else outcome.report["result_sha256"])))
        return int(error is not None)
    except (Error,OSError) as error:
        print(json.dumps(dict(status="refused",reason=getattr(error,"reason","io_error"),field=getattr(error,"field",type(error).__name__))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
