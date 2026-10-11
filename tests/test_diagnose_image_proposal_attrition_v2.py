"""Synthetic-only wrapper identity, scalar accounting and no-clobber controls."""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import region_growth as growth, lattice_planes
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from test_diagnose_g7_selection_failure import controlled_graph
import test_region_growth as room

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation/diagnose_image_proposal_attrition_v2.py"
spec = importlib.util.spec_from_file_location("proposal_attrition_v2_test_subject", SCRIPT)
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def scalar_events(observer, frames=("a", "b")):
    """Explicit synthetic hook events for serialization/publication unit tests."""
    for frame in frames:
        if type(observer) is subject.StreamingObserver:
            observer.begin_frame(frame)
        observer.expect("frame", (frame, "proposal"))
        observer.finish("frame", (frame, "proposal"), "synthetic")
        for axis in range(3):
            for side in (-1, 1):
                identity = (frame, axis, side)
                observer.expect("seed", identity)
                observer.finish("seed", identity, "available" if side == 1 else "absent",
                    axis=axis, eligible_seed_count=3 if side == 1 else 0)
            identity = (frame, axis)
            observer.expect("proposal", identity)
            observer.finish("proposal", identity, "returned", axis=axis, returned_ordinal=axis)
            observer.bind(("proposal_return", frame, axis), identity)
            observer.expect("reading", identity)
            observer.finish("reading", identity, "accepted", axis=axis)
        for prefix in ("proposal_generation", "proposal_returned", "proposal_to_reading"):
            observer.check(prefix + ":" + frame, True)
        if type(observer) is subject.StreamingObserver:
            observer.end_frame(frame)


def run(tmp_path, monkeypatch, *, kind="no_active_factors", pixels=False):
    graph, sources = controlled_graph(kind)
    monkeypatch.setattr(subject.g7.orchestration.g_mod, "derive_image_derived_graph", lambda **kwargs: graph)
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic input; no decoder called")
    calls = []
    def delivery(images, **kwargs):
        calls.append((images, kwargs, growth._attrition_state.get()))
        if pixels:
            rgb, _, segments, _ = room._synthetic_room()
            monkeypatch.setattr(growth, "detect_segments", lambda _: segments)
            for frame in graph.frame_ids:
                readings, abstentions = measure._measure_frame(
                    SimpleNamespace(frame_id=frame, image=rgb), room._calibration())
                assert len(readings) == 3 and not abstentions
        else:
            scalar_events(growth._attrition_state.get(), graph.frame_ids)
        # Explicit synthetic F1/matcher hook census for this stubbed-G unit.
        # Real default-driver and unique-source tests exercise native hooks.
        observer = growth._attrition_state.get()
        signatures = list(sources["signatures_result"].signatures)
        for ordinal, evidence in enumerate(signatures):
            identity = (evidence.frame_id, ordinal, "synthetic_signature")
            observer.expect("signature", identity)
            observer.finish("signature", identity, "built", axis=evidence.reading_axis)
        for left_index, left in enumerate(signatures):
            for right_index, right in enumerate(signatures):
                if left.frame_id >= right.frame_id:
                    continue
                for quadrant in range(4):
                    identity = (left.frame_id, left_index, left.reading_axis, 1.,
                                right.frame_id, right_index, right.reading_axis, 1., quadrant)
                    observer.expect("pair", identity)
                    observer.finish("pair", identity, "synthetic_rejected",
                                    axis=left.reading_axis, right_axis=right.reading_axis)
        for name in ("reading_universe", "signature_universe", "pair_product"):
            observer.check(name, True)
        if graph.candidate_matches:
            observer.check("pair_admission_traces", True)
        observed = subject.g7.orchestration.g_mod.derive_image_derived_graph(**sources)
        Path(kwargs["project_path"]).write_bytes(b"synthetic production output")
        return observed
    outcome = subject.run_once([image], project=tmp_path / "out.mcscan", destination=tmp_path / "out.litematic",
        diagnostic=tmp_path / "g7.json", sidecar=tmp_path / "attrition.json", name="synthetic", description="synthetic", delivery=delivery)
    return outcome, graph, calls


def test_real_measure_frame_context_accounting_and_g7_identity(tmp_path, monkeypatch):
    outer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(outer):
        outcome, graph, calls = run(tmp_path, monkeypatch, pixels=True)
        assert growth._attrition_state.get() is outer
    assert growth._attrition_state.get() is None
    assert len(calls) == 1 and calls[0][2] is not outer
    assert outcome.g7_outcome.production_result is graph
    assert outcome.g7_outcome.diagnostic_error is None and outcome.diagnostic_error is None
    assert outcome.report["observer"]["status"] == "complete"
    stages = outcome.report["observer"]["stages"]
    assert stages["seed"]["expected"] == 12
    assert stages["proposal"]["classified"] == stages["reading"]["classified"] == 6
    assert outcome.report["g7"]["counts"]["readings"] == 6
    assert len(outcome.report["observer"]["proposal_handoffs"]) == 6
    assert (tmp_path / "g7.json").read_bytes() == subject.g7.canonical_bytes(outcome.g7_outcome.report)
    assert subject._readback(tmp_path / "attrition.json", outcome.g7_outcome.report) == subject.canonical_bytes(outcome.report)
    assert (tmp_path / "out.mcscan").read_bytes() == b"synthetic production output"
    assert not (tmp_path / "out.litematic").exists()


def test_genuine_absence_does_not_manufacture_axis_or_reading(monkeypatch):
    rgb, usable, segments, _ = room._synthetic_room()
    monkeypatch.setattr(growth, "detect_segments", lambda _: segments)
    observer = growth._ScalarAttritionObserver(max_items=4096, max_bytes=subject.MAX_BYTES)
    with growth._scalar_attrition_context(observer, frame="a"):
        proposals = growth.propose_image_planes_v2(rgb, usable, room._calibration())
        readings = lattice_planes.frame_plane_readings_from_proposals(rgb, usable, room._calibration(), proposals, frame="a")
    assert {p.axis for p in proposals.proposals} == {r.axis for r in readings} == {0, 2}
    raw = subject._observer_record(observer)
    absent = [r for r in raw["stages"]["seed"]["rows"] if r["axis"] == 1]
    assert len(absent) == 1 and absent[0]["count"] == 2
    assert absent[0]["disposition"] == "absent" and absent[0]["eligible_seed_count"]["total"] == 0


def test_real_default_driver_off_on_pixel_universe_and_kernel_parity(tmp_path, monkeypatch):
    """Known synthetic calibration/connection only; real readers through delivery."""
    import cv2
    import test_image_derived_lattice_measure as fixture
    from mc_photogrammetry.pipeline import image_derived_production as production
    from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
    from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_mod
    from mc_photogrammetry.reconstruction import plane_graph
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    rgb, _, segments, _ = room._synthetic_room()
    monkeypatch.setattr(growth, "detect_segments", lambda _: segments)
    monkeypatch.setattr(fixture, "_true_calibration", room._calibration)
    session = fixture._build_session(monkeypatch, image=rgb, frame_ids=("a", "b"))
    frames = tuple(dataclasses.replace(f, source_label=str(tmp_path / (f.frame_id + ".png"))) for f in session.frames)
    session = dataclasses.replace(session, frames=frames, input_frames=frames, session_sha256=None)
    connection = fixture.connections_mod.build_image_derived_connections(session,
        budgets=fixture.connections_mod.make_connection_budgets(max_pairs=2, max_total_support=64),
        matcher=lambda left, _li, right, _ri, **kw: fixture._solved_pair(left, right))
    pose = pose_mod.compose_image_derived_pose_graph(connection, budgets=pose_mod.PoseGraphBudgets(128, 1000, 1., .1))
    assert connection.status == pose.status == "ok"
    reason, detail = entry_mod._terminal_reason(connection, pose)
    entry = entry_mod.ImageDerivedEntryResult(entry_mod.IMAGE_DERIVED_ENTRY_SCHEMA,
        entry_mod.IMAGE_DERIVED_ENTRY_PRODUCER, entry_mod.GEOMETRY_UNAVAILABLE, reason, detail,
        str(tmp_path / "out.mcscan"), tuple(f.source_label for f in frames),
        connection.budgets, pose.budgets, session, connection, pose)
    monkeypatch.setattr(production.entry_mod, "build_image_derived_entry", lambda *a, **k: entry)
    images = tuple(Path(f.source_label) for f in frames)
    ok, encoded = cv2.imencode(".png", rgb)
    assert ok
    for image in images:
        image.write_bytes(encoded.tobytes())
    calls = dict(matcher=0, selector=0, evaluator=0)
    for key, module, name in (("matcher", plane_graph, "match_plane_graph"),
                              ("selector", selection, "select_endpoint_hypotheses"),
                              ("evaluator", plane_graph, "_evaluate_assignment")):
        saved = getattr(module, name)
        def count(*args, _key=key, _saved=saved, **kwargs):
            calls[_key] += 1
            return _saved(*args, **kwargs)
        monkeypatch.setattr(module, name, count)
    contexts = []
    original_measure = measure._measure_frame
    def observed_measure(*args, **kwargs):
        contexts.append(growth._attrition_state.get())
        return original_measure(*args, **kwargs)
    monkeypatch.setattr(measure, "_measure_frame", observed_measure)
    kwargs = dict(project=tmp_path / "out.mcscan", destination=tmp_path / "out.litematic",
        diagnostic=tmp_path / "g7.json", name="synthetic", description="synthetic")
    off = subject.g7.run_once(images, **kwargs)
    assert off.production_error is None and off.diagnostic_error is None
    assert off.report["status"] in ("complete", "no_failure")
    off_counts = dict(calls)
    assert off_counts["matcher"] == off_counts["selector"] == 1
    assert contexts == [None, None]
    # Only exact files produced in this test's temporary directory are removed.
    for key in ("project", "destination", "diagnostic"):
        kwargs[key].unlink(missing_ok=True)
    on = subject.run_once(images, sidecar=tmp_path / "attrition.json", **kwargs)
    assert on.g7_outcome.production_error is None
    # The unchanged selector stops once ambiguity is proven. Its partial proof
    # cannot authorize a complete attrition sidecar, even with closed hooks.
    assert on.diagnostic_error.reason == "g7_incomplete"
    assert on.diagnostic_error.field == "partial_proof"
    assert not (tmp_path / "attrition.json").exists()
    assert contexts[-1].record()["status"] == "complete"
    assert calls == {key: 2 * value for key, value in off_counts.items()}
    assert len(contexts) == 4 and all(c is not None for c in contexts[2:])
    assert contexts[2] is contexts[3] and growth._attrition_state.get() is None
    assert subject.g7.canonical_bytes(off.report["graph"]) == subject.g7.canonical_bytes(on.g7_outcome.report["graph"])
    assert off.production_result.status == on.g7_outcome.production_result.status
    assert off.production_result.reason == on.g7_outcome.production_result.reason
    assert off.production_result.result_sha256 == on.g7_outcome.production_result.result_sha256
    assert on.g7_outcome.report["graph"]["counts"]["readings"] == 6
    assert on.g7_outcome.report["graph"]["counts"]["signatures"] == 10
    assert on.g7_outcome.report["graph"]["counts"]["candidates"] == 13
    assert contexts[-1].record()["stages"]["reading"]["classified"] == 6


def test_genuine_unique_g7_is_compact_no_failure_without_invented_proof(tmp_path, monkeypatch):
    from test_image_derived_graph_v3 import _source_records
    from mc_photogrammetry.reconstruction import plane_graph
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    sources = _source_records(monkeypatch)
    counts = dict(matcher=0, selector=0, evaluator=0)
    for key, module, name in (("matcher", plane_graph, "match_plane_graph"),
                              ("selector", selection, "select_endpoint_hypotheses"),
                              ("evaluator", plane_graph, "_evaluate_assignment")):
        saved = getattr(module, name)
        def counted(*args, _key=key, _saved=saved, **kwargs):
            counts[_key] += 1
            return _saved(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic source records; no decode")
    delivered, observers = [], []
    def delivery(images, **kwargs):
        observers.append(growth._attrition_state.get())
        # Explicit scalar fixture: this unit checks compact-success serialization.
        scalar_events(growth._attrition_state.get(), tuple(f.frame_id for f in sources["session"].frames))
        canonical, indices, _ = plane_graph._canonical_reading_index(sources["lattice_result"].readings)
        observer = observers[-1]
        for (frame, index), reading in canonical.items():
            observer.bind(("reading", id(reading)), (frame, index))
        for evidence in sources["signatures_result"].signatures:
            signature = evidence.signature
            identity = plane_graph._attrition_signature_identity(signature, indices[id(signature.reading)][1])
            observer.expect("signature", identity)
            observer.finish("signature", identity, "built", axis=signature.reading.axis)
        result = subject.g7.orchestration.g_mod.derive_image_derived_graph(**sources)
        delivered.append(result)
        Path(kwargs["project_path"]).write_bytes(result.result_sha256.encode())
        return result
    outcome = subject.run_once([image], project=tmp_path / "out.mcscan", destination=tmp_path / "out.litematic",
        diagnostic=tmp_path / "g7.json", sidecar=tmp_path / "attrition.json", name="synthetic", description="synthetic", delivery=delivery)
    assert outcome.g7_outcome.production_error is None, outcome.g7_outcome.production_error
    assert counts == dict(matcher=1, selector=1, evaluator=1)
    assert delivered[0].status == "assembled" and outcome.g7_outcome.production_result is delivered[0]
    assert outcome.diagnostic_error is None, (observers[0].record()["checks"], {k: {x: v[x] for x in ("unclassified", "unexpected", "multiply_classified")} for k, v in observers[0].record()["stages"].items()})
    assert outcome.g7_outcome.report["status"] == "no_failure"
    assert outcome.report["g7"]["proof_scope"] == "compact_no_failure"
    assert outcome.report["g7"]["assignment_counts"] is None
    assert outcome.report["g7"]["first_failure_reasons"] == {}
    assert subject._readback(tmp_path / "attrition.json", outcome.g7_outcome.report) == subject.canonical_bytes(outcome.report)


def test_delivery_wrapper_exact_args_result_and_exception_restoration():
    sentinel, argument, kwargs, seen = object(), object(), dict(value=object()), []
    def delivery(*args, **kw):
        seen.append((args, kw, growth._attrition_state.get()))
        return sentinel
    capture = subject.DeliveryCapture(delivery)
    assert capture(argument, **kwargs) is sentinel
    assert seen[0][0][0] is argument and seen[0][1]["value"] is kwargs["value"]
    assert seen[0][2] is capture.observer and growth._attrition_state.get() is None
    error = RuntimeError("native")
    def failing(*args, **kwargs):
        assert growth._attrition_state.get() is capture.observer
        raise error
    capture.delivery = failing
    with pytest.raises(RuntimeError) as exc:
        capture(argument, **kwargs)
    assert exc.value is error and growth._attrition_state.get() is None


@pytest.mark.parametrize("mutation", ["task", "source", "runner", "input", "g7", "stage", "metric", "handoff", "notes", "accepted", "seed", "cardinality", "axis_redistribution", "generated_duplicate"])
def test_resealed_malformed_foreign_and_binding_reports_refuse(mutation, tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    assert outcome.diagnostic_error is None
    report = copy.deepcopy(outcome.report)
    if mutation == "task": report["task"]["id"] = "foreign"
    elif mutation == "source": report["source_sha256"][next(iter(report["source_sha256"]))] = "0" * 64
    elif mutation == "runner": report["runner_sha256"] = "0" * 64
    elif mutation == "input": report["inputs"][0]["sha256"] = "0" * 64
    elif mutation == "g7": report["g7"]["selection_sha256"] = "0" * 64
    elif mutation == "stage": report["observer"]["stages"]["reading"]["unclassified"] = 1
    elif mutation == "metric": report["observer"]["stages"]["seed"]["rows"][0]["eligible_seed_count"]["total"] = 99
    elif mutation == "handoff": report["observer"]["proposal_handoffs"][0]["returned_identity"] = ("foreign", 0)
    elif mutation == "notes": report["notes"] = [{"position": [1, 2, 3]}]
    elif mutation == "accepted": report["observer"]["stages"]["reading"]["rows"][0]["disposition"] = "no_period"
    elif mutation == "seed": report["observer"]["stages"]["seed"]["rows"][1]["disposition"] = "absent"
    elif mutation == "cardinality":
        stage = report["observer"]["stages"]["proposal"]
        stage["expected"] = stage["classified"] = stage["rows"][0]["count"] = 2**62
    elif mutation == "axis_redistribution":
        rows = report["observer"]["stages"]["reading"]["rows"]
        row = next(r for r in rows if r["frame"] == "a" and r["axis"] == 0)
        row["axis"] = 1
        assert sum(r["count"] for r in rows if r["frame"] == "a") == 3
    elif mutation == "generated_duplicate":
        handoffs = report["observer"]["proposal_handoffs"]
        first = handoffs[0]
        second = next(h for h in handoffs[1:] if h["generated_identity"][0] == first["generated_identity"][0])
        second["generated_identity"] = first["generated_identity"]
        assert second["returned_identity"] != first["returned_identity"]
    report = subject._seal(report)
    with pytest.raises(subject.AttritionError):
        subject.validate_report(report, outcome.g7_outcome.report)


@pytest.mark.parametrize("mode", ["missing", "overflow", "bytes", "false_check"])
def test_incomplete_observer_preserves_production_and_g7(mode, tmp_path, monkeypatch):
    original = subject.DeliveryCapture.__call__
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if mode == "missing": self.observer.expect("reading", ("a", 99))
        elif mode == "false_check": self.observer.check("synthetic", False)
        else:
            self.observer.incomplete = True
            self.observer.omitted += 1
        return result
    monkeypatch.setattr(subject.DeliveryCapture, "__call__", wrapped)
    outcome, graph, _ = run(tmp_path, monkeypatch)
    assert outcome.g7_outcome.production_result is graph
    assert outcome.diagnostic_error.reason == "observer_incomplete"
    assert not (tmp_path / "attrition.json").exists()
    assert (tmp_path / "g7.json").exists() and (tmp_path / "out.mcscan").exists()


def test_actual_observer_budget_and_partial_g7_refuse_without_replay(tmp_path, monkeypatch):
    observer = growth._ScalarAttritionObserver(max_items=2, max_bytes=1024)
    scalar_events(observer)
    assert observer.incomplete and observer.omitted > 0
    byte_limited = growth._ScalarAttritionObserver(max_items=4096, max_bytes=32)
    scalar_events(byte_limited)
    assert byte_limited.incomplete and byte_limited.omitted > 0
    with pytest.raises(subject.AttritionError):
        subject.DeliveryCapture(lambda: None, growth._ScalarAttritionObserver(max_items=4097))
    outcome, graph, _ = run(tmp_path, monkeypatch, kind="budget_refused")
    assert outcome.g7_outcome.production_result is graph
    assert outcome.diagnostic_error.reason == "g7_incomplete"
    assert not (tmp_path / "attrition.json").exists()
    assert outcome.g7_outcome.report["graph"]["proof_scope"] == "partial"


def test_native_delivery_error_retains_g7_and_created_output(tmp_path):
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic only")
    native = RuntimeError("native delivery error after output")
    seen = []
    def delivery(images, **kwargs):
        seen.append(growth._attrition_state.get())
        Path(kwargs["project_path"]).write_bytes(b"created before native error")
        raise native
    outcome = subject.run_once([image], project=tmp_path / "out.mcscan", destination=tmp_path / "out.litematic",
        diagnostic=tmp_path / "g7.json", sidecar=tmp_path / "attrition.json", name="s", description="s", delivery=delivery)
    assert len(seen) == 1 and seen[0] is not None and growth._attrition_state.get() is None
    assert outcome.g7_outcome.production_error is native
    assert outcome.diagnostic_error.reason == "g7_incomplete"
    assert (tmp_path / "out.mcscan").read_bytes() == b"created before native error"
    assert (tmp_path / "g7.json").exists() and not (tmp_path / "attrition.json").exists()


def test_runtime_source_mismatch_stops_before_delivery(tmp_path, monkeypatch):
    original = subject.g7._file_hash
    pinned = subject.REPO_ROOT / next(iter(subject.SOURCE_PINS))
    monkeypatch.setattr(subject.g7, "_file_hash", lambda p: "0" * 64 if Path(p) == pinned else original(p))
    with pytest.raises(subject.AttritionError) as exc:
        subject.run_once([], project=tmp_path / "out.mcscan", destination=tmp_path / "out.litematic",
            diagnostic=tmp_path / "g7.json", sidecar=tmp_path / "attrition.json", name="s", description="s",
            delivery=lambda *a, **k: pytest.fail("delivery entered"))
    assert exc.value.reason == "source_pin_mismatch" and not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["short_write", "fsync", "link", "race", "temporary_readback", "final_readback"])
def test_atomic_failure_preserves_production_and_foreign_racer(failure, tmp_path, monkeypatch):
    original = subject.publish_report
    reached = []
    def publish(path, report, g7_report):
        def fail(*args):
            reached.append(failure)
            raise OSError(failure)
        if failure == "short_write":
            fdopen = subject.os.fdopen
            class ShortStream:
                def __init__(self, stream): self.stream = stream
                def __enter__(self): return self
                def __exit__(self, *args): return self.stream.__exit__(*args)
                def write(self, data):
                    reached.append(failure)
                    return self.stream.write(data[:-1])
            monkeypatch.setattr(subject.os, "fdopen", lambda *a, **k: ShortStream(fdopen(*a, **k)))
        elif failure == "fsync": monkeypatch.setattr(subject.os, "fsync", fail)
        elif failure == "link": monkeypatch.setattr(subject.os, "link", fail)
        elif failure == "race":
            def race(source, target):
                reached.append(failure)
                Path(target).write_bytes(b"foreign")
                raise FileExistsError("race")
            monkeypatch.setattr(subject.os, "link", race)
        else:
            read = subject._readback
            def readback(p, g):
                if (failure == "temporary_readback") == (Path(p) != Path(path)):
                    reached.append(failure)
                    raise OSError("readback")
                return read(p, g)
            monkeypatch.setattr(subject, "_readback", readback)
        return original(path, report, g7_report)
    monkeypatch.setattr(subject, "publish_report", publish)
    outcome, graph, _ = run(tmp_path, monkeypatch)
    assert reached == [failure]
    if failure == "short_write":
        assert outcome.diagnostic_error.reason == "publication_failed"
        assert outcome.diagnostic_error.field == "short_write"
    assert outcome.g7_outcome.production_result is graph and outcome.diagnostic_error is not None
    assert (tmp_path / "out.mcscan").read_bytes() == b"synthetic production output"
    assert (tmp_path / "g7.json").exists()
    target = tmp_path / "attrition.json"
    assert target.read_bytes() == b"foreign" if failure == "race" else not target.exists()
    assert not list(tmp_path.glob(".proposal-attrition-*.tmp"))


def test_success_publication_not_visible_until_fsync_and_exact_readback(tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    target = tmp_path / "second.json"
    original = os.fsync
    def fsync(fd):
        assert not target.exists()
        return original(fd)
    monkeypatch.setattr(subject.os, "fsync", fsync)
    subject.publish_report(target, outcome.report, outcome.g7_outcome.report)
    assert subject._readback(target, outcome.g7_outcome.report) == subject.canonical_bytes(outcome.report)
    with pytest.raises(subject.AttritionError):
        subject.publish_report(target, outcome.report, outcome.g7_outcome.report)


def test_occupied_sidecar_and_33rd_image_stop_before_delivery(tmp_path):
    target = tmp_path / "occupied.json"; target.write_bytes(b"foreign")
    kwargs = dict(project=tmp_path / "a.mcscan", destination=tmp_path / "a.litematic", diagnostic=tmp_path / "g7.json", sidecar=target, name="s", description="s", delivery=lambda *a, **k: pytest.fail("delivery entered"))
    with pytest.raises(subject.AttritionError): subject.run_once([], **kwargs)
    assert target.read_bytes() == b"foreign"
    kwargs["sidecar"] = tmp_path / "new.json"
    def images():
        for i in range(33): yield tmp_path / f"{i}.png"
        pytest.fail("unbounded image traversal")
    with pytest.raises(subject.AttritionError): subject.run_once(images(), **kwargs)


@pytest.mark.parametrize("kind", ["depth", "items", "integer"])
def test_bounded_predecode_never_enters_json_decoder(kind, tmp_path, monkeypatch):
    if kind == "depth": raw = b"[" * 17 + b"0" + b"]" * 17
    elif kind == "items": raw = b"[" + b"0," * subject.MAX_NODES + b"0]"
    else: raw = b"1" * 100
    path = tmp_path / "unsafe.json"; path.write_bytes(raw)
    monkeypatch.setattr(subject.json, "loads", lambda *a, **k: pytest.fail("decoder entered"))
    with pytest.raises(subject.AttritionError): subject._readback(path, {})


def test_escaped_byte_precharge_and_huge_int_never_enter_encoder(monkeypatch):
    monkeypatch.setattr(subject.json, "dumps", lambda *a, **k: pytest.fail("encoder entered"))
    for value in ({"value": "\x01" * (subject.MAX_BYTES // 6 + 1)}, {"value": 2**100}):
        with pytest.raises(subject.AttritionError): subject.canonical_bytes(value)


def test_cli_exposes_no_metadata_or_budget_override():
    for flag in ("--bounds", "--camera", "--local-period-support", "--max-records"):
        with pytest.raises(SystemExit) as exc: subject.main([flag])
        assert exc.value.code == 2


def frame_event(observer, frame="a"):
    observer.begin_frame(frame)
    observer.expect("frame", (frame, "proposal"))
    observer.finish("frame", (frame, "proposal"), "synthetic")
    observer.end_frame(frame)


def test_streams_more_than_c7_volume_with_bounded_closed_scope_eviction():
    observer = subject.StreamingObserver()
    for frame_index in range(32):
        frame = "frame-" + str(frame_index)
        observer.begin_frame(frame)
        for identity_index in range(512):
            identity = (frame, 0, identity_index, "row")
            observer.expect("period_profile", identity)
            observer.finish("period_profile", identity, "synthetic", axis=identity_index % 3)
        assert len(observer._stage("period_profile")["expected"]) == 512
        observer.end_frame(frame)
        assert observer._stage("period_profile")["expected"] == {}
        assert observer._stage("period_profile")["terminal"] == {}
    record = observer.record()
    assert record["status"] == "complete"
    assert record["resources"]["events"] == 32832
    assert record["resources"]["peak_identities"] == 512
    assert record["resources"]["live_identities"] == 0
    assert record["resources"]["peak_bytes"] < subject.MAX_BYTES
    stage = record["stages"]["period_profile"]
    assert stage["expected"] == stage["classified"] == 16384
    assert sum(row["count"] for row in stage["counts"]) == 16384
    assert len(stage["counts"]) == 96
    assert len(record["resources"]["closed_frames"]) == 32
    assert subject.canonical_bytes(record) == subject.canonical_bytes(observer.record())


def test_c7_shaped_pair_volume_keeps_exact_private_product_contract():
    observer = subject.StreamingObserver()
    sizes = (9, 12, 8, 8, 10, 3)
    for frame_index in range(6):
        frame_event(observer, "f" + str(frame_index))
    expected = 0
    for left in range(6):
        for right in range(left + 1, 6):
            for a in range(sizes[left]):
                for b in range(sizes[right]):
                    for quadrant in range(4):
                        identity = ("f" + str(left), a, 0, 1., "f" + str(right), b, 0, 1., quadrant)
                        observer.expect("pair", identity)
                        observer.bind(("defer", identity), identity)
                        assert ("defer", identity) in observer.bindings
                        observer.finish("pair", identity, "synthetic_refused", axis=a % 3, right_axis=b % 3)
                        assert ("defer", identity) not in observer.bindings
                        expected += 1
    assert expected == 4076
    census = observer._stage("pair")
    assert len(census["expected"]) == len(census["terminal"]) == expected
    observer.check("pair_product", len(census["expected"]) == expected)
    record = observer.record()
    assert record["status"] == "complete"
    assert record["checks"]["pair_product"] is True
    assert record["stages"]["pair"]["expected"] == 4076
    assert sum(row["count"] for row in record["stages"]["pair"]["counts"]) == 4076
    assert record["resources"]["peak_bytes"] < subject.MAX_BYTES


@pytest.mark.parametrize("failure", ["duplicate_expected", "duplicate_terminal", "unexpected_terminal", "missing", "late", "reopen", "false_check"])
def test_exact_closure_failures_are_sticky(failure):
    observer = subject.StreamingObserver()
    observer.begin_frame("a")
    identity = ("a", 0, 1)
    observer.expect("seed", identity)
    if failure == "duplicate_expected": observer.expect("seed", identity)
    elif failure == "unexpected_terminal": observer.finish("seed", ("a", 0, -1), "absent", axis=0, eligible_seed_count=0)
    elif failure == "missing": observer.end_frame("a")
    else:
        observer.finish("seed", identity, "available", axis=0, eligible_seed_count=1)
        if failure == "duplicate_terminal": observer.finish("seed", identity, "available", axis=0, eligible_seed_count=1)
        elif failure == "false_check": observer.check("synthetic", False)
        else:
            observer.end_frame("a")
            assert observer._stage("seed")["expected"] == {}
            if failure == "late": observer.finish("seed", identity, "available", axis=0, eligible_seed_count=1)
            elif failure == "reopen": observer.begin_frame("a")
    assert observer.incomplete and observer.record()["status"] == "incomplete"
    assert observer.record()["resources"]["denials"]
    before = observer.bytes
    observer.begin_frame("replacement")
    observer.expect("signature", ("replacement", 0, 0, 1.))
    assert observer.incomplete and observer.bytes == before


def test_unclosed_scope_is_not_evicted_and_closed_scope_tombstone_persists():
    observer = subject.StreamingObserver()
    observer.begin_frame("a")
    observer.expect("reading", ("a", 0))
    observer.end_frame("a")
    assert "a" in observer.active_frames and "a" not in observer.closed_frames
    assert observer.pending("reading", "a") == (("a", 0),)
    assert observer.record()["status"] == "incomplete"
    closed = subject.StreamingObserver()
    frame_event(closed)
    assert closed._stage("frame")["expected"] == closed._stage("frame")["terminal"] == {}
    assert closed.record()["stages"]["frame"]["expected"] == 1
    assert "a" in closed.closed_frames
    closed.expect("frame", ("a", "new"))
    assert closed.record()["resources"]["denials"] == {"late": 1}


@pytest.mark.parametrize("resource", ["live_identities", "bindings", "aggregate_buckets", "events", "closed_frame_scopes", "counter_max", "retained_bytes"])
def test_resource_exact_limit_and_one_over_fail_closed(resource):
    if resource == "retained_bytes":
        baseline = subject.StreamingObserver()
        frame_event(baseline)
        required = baseline.peak_bytes
        exact = subject.StreamingObserver(limits={resource: required})
        frame_event(exact)
        assert exact.record()["status"] == "complete" and exact.peak_bytes == required
        short = subject.StreamingObserver(limits={resource: required - 1})
        frame_event(short)
        assert short.incomplete and short.peak_bytes < required
        return
    if resource in ("events", "counter_max"):
        exact = subject.StreamingObserver(limits={resource: 4})
        frame_event(exact)
        assert exact.record()["status"] == "complete" and exact.events == 4
        frame_event(exact, "b")
        assert exact.incomplete and exact.record()["resources"]["denials"].get(resource)
        return
    observer = subject.StreamingObserver(limits={resource: 2})
    if resource == "closed_frame_scopes":
        frame_event(observer, "a"); frame_event(observer, "b")
        assert observer.record()["status"] == "complete"
        observer.begin_frame("c")
    elif resource == "bindings":
        for i in range(2): observer.bind(("synthetic", i), ("a", i))
        assert len(observer.bindings) == 2 and not observer.incomplete
        observer.bind(("synthetic", 2), ("a", 2))
    elif resource == "live_identities":
        for i in range(2): observer.expect("signature", ("a", i, 0, 1.))
        assert observer._live() == 2 and not observer.incomplete
        observer.expect("signature", ("a", 2, 0, 1.))
    else:
        for i in range(2):
            identity = ("a", i, 0, 1.)
            observer.expect("signature", identity)
            observer.finish("signature", identity, "built", axis=i)
        assert observer._buckets() == 2 and not observer.incomplete
        observer.expect("signature", ("a", 2, 0, 1.))
        observer.finish("signature", ("a", 2, 0, 1.), "built", axis=2)
    assert observer.incomplete and observer.record()["resources"]["denials"].get(resource)
    assert observer.bytes <= observer.max_bytes
    assert observer._live() <= observer.limits["live_identities"]
    assert len(observer.bindings) <= observer.limits["bindings"]
    assert observer._buckets() <= observer.limits["aggregate_buckets"]


@pytest.mark.parametrize("bound", ["identity_bytes", "identity_nodes", "identity_depth"])
def test_identity_exact_boundary_and_one_over_encoder_precharge(bound, monkeypatch):
    observer = subject.StreamingObserver()
    if bound == "identity_bytes":
        value = "a" * 20
        required = len(subject.canonical_bytes(value))
    elif bound == "identity_nodes":
        value = [1, 2, 3]
        required = 4
    else:
        value = [[[1]]]
        required = 3
    exact = subject.StreamingObserver(limits={bound: required})
    assert exact._encode(value) == subject.canonical_bytes(value)
    short = subject.StreamingObserver(limits={bound: required - 1})
    called = []
    monkeypatch.setattr(subject.json, "dumps", lambda *a, **k: called.append(1))
    with pytest.raises(subject.AttritionError): short._encode(value)
    assert called == []


def test_seed_metric_counter_overflow_is_sticky_without_wrap():
    observer = subject.StreamingObserver(limits={"counter_max": 10})
    observer.begin_frame("a")
    for side in (-1, 1):
        identity = ("a", 0, side)
        observer.expect("seed", identity)
        observer.finish("seed", identity, "available", axis=0, eligible_seed_count=6)
    assert observer.incomplete and observer.denials.get("counter_max") == 1
    metric = next(iter(observer._stage("seed")["counts"].values()))["metrics"]["eligible_seed_count"]
    assert metric["total"] == 6 and metric["count"] == 1


@pytest.mark.parametrize("field", ["events", "live_identities", "aggregate_buckets", "bindings", "active_frames", "closed_frames", "peak_bytes", "denials", "limits"])
def test_resealed_stream_resource_accounting_refuses(field, tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    assert outcome.diagnostic_error is None
    report = copy.deepcopy(outcome.report)
    resources = report["observer"]["resources"]
    if field in ("events", "bindings"): resources[field] = 0
    elif field in ("live_identities", "aggregate_buckets"): resources[field] += 1
    elif field == "active_frames": resources[field] = ["a"]
    elif field == "closed_frames": resources[field] = ["a"]
    elif field == "peak_bytes": resources[field] = 0
    elif field == "denials": resources[field] = {"events": 1}
    else: report["observer"]["limits"]["events"] += 1
    report = subject._seal(report)
    with pytest.raises(subject.AttritionError): subject.validate_report(report, outcome.g7_outcome.report)


def test_stream_canonical_projection_equals_unchanged_legacy_observer():
    legacy = growth._ScalarAttritionObserver()
    stream = subject.StreamingObserver()
    for observer in (legacy, stream):
        for frame in ("a", "b"):
            if observer is stream: observer.begin_frame(frame)
            for axis in range(3):
                for side in (-1, 1):
                    identity = (frame, axis, side)
                    observer.expect("seed", identity)
                    observer.finish("seed", identity, "available", axis=axis, eligible_seed_count=axis + 1)
                for ordinal in range(4):
                    identity = (frame, axis, ordinal, "row")
                    observer.expect("period_profile", identity)
                    observer.finish("period_profile", identity, "synthetic", axis=axis, standard_deviation=0.25)
            if observer is stream: observer.end_frame(frame)
    def projection(record):
        result = {}
        for name, stage in record["stages"].items():
            rows = {}
            for row in stage["counts"]:
                key = (row["frame"], row["axis"], row["disposition"])
                value = rows.setdefault(key, dict(count=0, seed_count=0, seed_total=0, seed_minimum=None, seed_maximum=None))
                value["count"] += row["count"]
                metric = row["metrics"].get("eligible_seed_count")
                if metric:
                    value["seed_count"] += metric["count"]
                    value["seed_total"] += metric["total"]
                    value["seed_minimum"] = metric["minimum"] if value["seed_minimum"] is None else min(value["seed_minimum"], metric["minimum"])
                    value["seed_maximum"] = metric["maximum"] if value["seed_maximum"] is None else max(value["seed_maximum"], metric["maximum"])
            result[name] = {key: stage[key] for key in ("expected", "classified", "unclassified", "unexpected", "multiply_classified", "unexpected_events", "terminal_sha256")}
            result[name]["rows"] = [[*key, value] for key, value in sorted(rows.items())]
        return result
    a, b = projection(legacy.record()), projection(stream.record())
    assert subject.canonical_bytes(a) == subject.canonical_bytes(b)
    assert legacy.record()["status"] == stream.record()["status"] == "complete"
    assert growth._ScalarAttritionObserver is type(legacy)
    assert legacy.stages["period_profile"]["expected"]
    assert stream.stages["period_profile"]["expected"] == {}


def test_limits_immutable_and_overflow_cannot_recover_after_eviction():
    observer = subject.StreamingObserver(limits={"events": 5})
    with pytest.raises(TypeError): observer.limits["events"] = 99
    frame_event(observer)
    assert observer.record()["status"] == "complete"
    observer.begin_frame("b")
    observer.expect("frame", ("b", "proposal"))
    assert observer.incomplete
    events, retained = observer.events, observer.bytes
    observer.end_frame("b")
    observer.begin_frame("c")
    observer.finish("frame", ("b", "proposal"), "synthetic")
    assert observer.events == events and observer.bytes == retained
    assert observer.record()["status"] == "incomplete"
    assert observer.record()["resources"]["denials"]["events"] >= 1


def test_measure_hook_and_native_error_identity_restore_without_calls():
    original = measure._measure_frame
    native = RuntimeError("synthetic")
    def delivery(*args, **kwargs):
        assert measure._measure_frame is not original
        raise native
    capture = subject.DeliveryCapture(delivery)
    with pytest.raises(RuntimeError) as exc: capture(object(), argument=object())
    assert exc.value is native and capture.calls == 1
    assert measure._measure_frame is original and growth._attrition_state.get() is None


def test_sidecar_exact_four_mib_and_one_over_precharge(monkeypatch):
    exact = "a" * (subject.MAX_BYTES - 2)
    assert len(subject.canonical_bytes(exact)) == subject.MAX_BYTES
    called = []
    monkeypatch.setattr(subject.json, "dumps", lambda *a, **k: called.append(1))
    with pytest.raises(subject.AttritionError): subject.canonical_bytes(exact + "a")
    assert called == []


@pytest.mark.parametrize("name", ["reading_universe", "signature_universe", "pair_product",
    "proposal_generation:a", "proposal_returned:a", "proposal_to_reading:a",
    "proposal_generation:b", "proposal_returned:b", "proposal_to_reading:b"])
def test_resealed_required_source_check_omission_refuses(name, tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    assert outcome.diagnostic_error is None
    report = copy.deepcopy(outcome.report)
    del report["observer"]["checks"][name]
    assert report["observer"]["checks"]
    with pytest.raises(subject.AttritionError, match="required_source_checks"):
        subject.validate_report(subject._seal(report), outcome.g7_outcome.report)


@pytest.mark.parametrize("name", ["signature", "pair"])
def test_resealed_stage_deletion_with_adjusted_resources_refuses(name, tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    assert outcome.diagnostic_error is None
    report = copy.deepcopy(outcome.report)
    observer = report["observer"]
    stage = observer["stages"].pop(name)
    observer["items"] -= stage["expected"]
    observer["resources"]["live_identities"] = observer["items"]
    observer["resources"]["aggregate_buckets"] -= len(stage["rows"])
    observer["resources"]["events"] -= stage["expected"] + stage["classified"]
    with pytest.raises(subject.AttritionError, match="stages"):
        subject.validate_report(subject._seal(report), outcome.g7_outcome.report)


@pytest.mark.parametrize("name", ["signature", "pair"])
def test_resealed_universe_census_reduction_refuses(name, tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    report = copy.deepcopy(outcome.report)
    observer = report["observer"]
    stage = observer["stages"][name]
    stage["rows"][0]["count"] -= 1
    stage["expected"] -= 1
    stage["classified"] -= 1
    if not stage["rows"][0]["count"]:
        stage["rows"].pop(0)
        observer["resources"]["aggregate_buckets"] -= 1
    observer["items"] -= 1
    observer["resources"]["live_identities"] -= 1
    observer["resources"]["events"] -= 2
    with pytest.raises(subject.AttritionError, match="signature_universe_count|pair_product_count"):
        subject.validate_report(subject._seal(report), outcome.g7_outcome.report)


def test_resealed_admission_trace_check_omission_refuses(tmp_path, monkeypatch):
    outcome, graph, _ = run(tmp_path, monkeypatch, kind="unsupported_reading_endpoint")
    assert graph.candidate_matches and outcome.diagnostic_error is None
    report = copy.deepcopy(outcome.report)
    del report["observer"]["checks"]["pair_admission_traces"]
    with pytest.raises(subject.AttritionError, match="required_source_checks"):
        subject.validate_report(subject._seal(report), outcome.g7_outcome.report)


def test_resealed_signature_axis_redistribution_preserving_totals_refuses(tmp_path, monkeypatch):
    outcome, _, _ = run(tmp_path, monkeypatch)
    report = copy.deepcopy(outcome.report)
    rows = report["observer"]["stages"]["signature"]["rows"]
    row = next(row for row in rows if row["frame"] == "a" and row["axis"] == 0)
    row["axis"] = 2
    with pytest.raises(subject.AttritionError, match="signature_frame_axis"):
        subject.validate_report(subject._seal(report), outcome.g7_outcome.report)
