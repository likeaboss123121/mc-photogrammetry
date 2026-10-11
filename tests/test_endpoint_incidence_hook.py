"""Synthetic source-record controls; never invokes image ingress or production."""
from dataclasses import replace
from types import MappingProxyType

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph
from test_endpoint_hypothesis import connected_fixture
from test_image_derived_endpoint_selection import _evidence
from test_image_derived_graph_v3 import _source_records


def _fixture(state="unique"):
    readings, rows = connected_fixture("image-0000", "image-0001")
    if state == "no":
        rows = []
    elif state == "multiple":
        rows += [replace(row, quadrant_of_right=2) for row in rows]
    elif state == "unresolved_evidence":
        rows.append(replace(rows[0], cell_shift_rc=(1, 2)))
    budget = csp.EndpointBudget(branch_nodes=0) if state == "budget_refused" else csp.EndpointBudget()
    return tuple(readings), tuple(rows), dict(session_frames=("image-0000", "image-0001"), budget=budget)


def _capture(state="unique"):
    readings, rows, options = _fixture(state)
    events = []
    result = csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda event, payload: events.append((event, payload)))
    return result, events, readings, rows, options


def _event(events, name):
    return [payload for event, payload in events if event == name]


@pytest.mark.parametrize("state", ["unique", "no", "multiple", "unresolved_evidence", "budget_refused"])
def test_none_and_observer_leave_exact_result_counters_and_digests_unchanged(state):
    result, events, readings, rows, options = _capture(state)
    assert result.status == state
    assert csp.solve_endpoint_hypotheses(readings, rows, **options) == result
    assert csp.solve_endpoint_hypotheses(readings, rows, **options, endpoint_incidence_sink=None) == result
    final = _event(events, "finished")[0]
    assert final["observer_complete"] and final["registry_complete"]
    assert dict(final["counters"]) == result.counters
    assert final["source_digest"] == result.source_digest
    assert final["exhaustive"] is result.exhaustive
    assert dict(final["observed"])["classified"] == result.counters["eligibility_hashed_assignments"]


def test_actual_registry_rows_quadrants_active_ids_and_local_normals_reconcile():
    result, events, readings, rows, _ = _capture()
    registry = _event(events, "registry")[0]
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    assert registry["endpoints"] == tuple((f, i, r.axis) for (f, i), r in sorted(indexed.items()))
    assert sorted(i for _, _, _, _, indices in registry["factors"] for i in indices) == list(range(len(rows)))
    by_fid = {fid: (u, v, relative, indices) for fid, u, v, relative, indices in registry["factors"]}
    for _, u, v, relative, indices in registry["factors"]:
        for i in indices:
            row = rows[i]
            assert (u, v, relative) == ((row.left, row.left_plane), (row.right, row.right_plane), row.quadrant_of_right)
    visited = _event(events, "visited")
    entered = _event(events, "entered")
    assert len(visited) == len(entered) == 4
    assert [p["quadrants"] for p in entered] == [p["quadrants"] for p in visited]
    for call in entered:
        q = dict(call["quadrants"])
        ids = call["active_factor_ids"]
        assert ids == tuple(fid for fid, (u, v, relative, _) in by_fid.items()
                            if (q[v[0]] - q[u[0]]) % 4 == relative)
        degrees = {key: 0 for key in indexed}
        for fid in ids:
            u, v, _, _ = by_fid[fid]
            degrees[u] += 1
            degrees[v] += 1
        assert sum(degrees.values()) == 2 * len(ids)
    returned = _event(events, "returned")
    assert [p["reason"] for p in returned] == [None, "no_active_factors", "no_active_factors", "no_active_factors"]
    assert all(p["result_available"] for p in returned)
    assert sum(len(p["active_factor_ids"]) for p in entered) == result.counters["active_factor_visits"]


def test_payload_is_immutable_and_has_no_live_model_or_relation_references():
    _, events, _, _, _ = _capture()
    def scalar(value):
        assert type(value) in (str, int, bool, type(None), tuple)
        if type(value) is tuple:
            for item in value: scalar(item)
    for _, payload in events:
        assert type(payload) is MappingProxyType
        with pytest.raises(TypeError): payload["changed"] = True
        for value in payload.values(): scalar(value)


@pytest.mark.parametrize("error", [RuntimeError("private/path"), KeyboardInterrupt(), SystemExit()])
def test_throwing_sink_cannot_change_exact_result_or_escape(error):
    readings, rows, options = _fixture()
    expected = csp.solve_endpoint_hypotheses(readings, rows, **options)
    calls = []
    def sink(event, payload):
        calls.append(event)
        raise error
    assert csp.solve_endpoint_hypotheses(readings, rows, **options, endpoint_incidence_sink=sink) == expected
    assert calls == ["registry", "incomplete", "finished"]


@pytest.mark.parametrize("cap,value", [("MAX_ASSIGNMENTS", 0), ("MAX_BYTES", 128)])
def test_observer_cap_only_marks_incomplete_never_changes_work(monkeypatch, cap, value):
    readings, rows, options = _fixture()
    expected = csp.solve_endpoint_hypotheses(readings, rows, **options)
    monkeypatch.setattr(csp._EndpointIncidenceHooks, cap, value)
    events = []
    result = csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda e, p: events.append((e, p)))
    assert result == expected
    if cap == "MAX_ASSIGNMENTS":
        assert _event(events, "finished")[0]["observer_complete"] is False
        assert not _event(events, "visited")
    else:
        assert not _event(events, "registry")
        assert not any(p.get("observer_complete") is True for _, p in events)


def test_budget_before_registry_does_not_fabricate_full_snapshot():
    readings, rows, options = _fixture()
    options["budget"] = csp.EndpointBudget(input_match_rows=0)
    events = []
    result = csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda e, p: events.append((e, p)))
    assert result.status == "budget_refused" and not _event(events, "registry")
    assert _event(events, "finished")[0]["registry_complete"] is False
    assert not _event(events, "returned")


def test_budget_between_visited_and_entered_retains_partial_states(monkeypatch):
    original = csp._Work.charge
    def charge(work, name, amount):
        if work.counts["visited_assignments"] and name == "relation_visits":
            raise csp._Exhausted
        return original(work, name, amount)
    monkeypatch.setattr(csp._Work, "charge", charge)
    result, events, readings, rows, options = _capture()
    assert result == csp.solve_endpoint_hypotheses(readings, rows, **options)
    assert result.status == "budget_refused"
    assert len(_event(events, "visited")) == 1
    assert not _event(events, "entered") and not _event(events, "returned")
    assert result.counters["eligibility_unclassified_assignments"] == 1


@pytest.mark.parametrize("stage", ["eligibility", "classification"])
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.parametrize("state", ["unique", "no"])
def test_exact_original_exception_and_returned_vs_classified_cleanup(monkeypatch, stage, exhausted, state):
    error = csp._Exhausted() if exhausted else ValueError("sensitive/private/path")
    original = csp._eligibility_reason if stage == "eligibility" else csp._Work.eligibility
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise error
    if stage == "eligibility": monkeypatch.setattr(csp, "_eligibility_reason", fail)
    else: monkeypatch.setattr(csp._Work, "eligibility", fail)
    events, hooks = [], []
    close = csp._EndpointIncidenceHooks.close
    def closed(hook):
        close(hook)
        hooks.append(hook)
    monkeypatch.setattr(csp._EndpointIncidenceHooks, "close", closed)
    readings, rows, options = _fixture(state)
    if exhausted:
        result = csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda e, p: events.append((e, p)))
        assert result.status == "budget_refused"
        assert result.counters["eligibility_unclassified_assignments"] == 1
    else:
        with pytest.raises(ValueError) as caught:
            csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda e, p: events.append((e, p)))
        assert caught.value is error
    assert len(calls) == 1 and calls[0][1] == {}
    assert len(_event(events, "entered")) == 1
    assert len(_event(events, "returned")) == (stage == "classification")
    assert not _event(events, "classified")
    failure = _event(events, "exception")[0]
    assert failure["stage"] == stage and failure["assignment_index"] == 0
    assert failure["result_available"] is (stage == "classification")
    if stage == "classification":
        returned = _event(events, "returned")[0]
        assert returned["assignment_index"] == failure["assignment_index"]
        assert failure["reason"] == returned["reason"] == (None if state == "unique" else "no_active_factors")
    else:
        assert "reason" not in failure
    assert "sensitive" not in repr(events)
    assert hooks and hooks[0].sink is None and hooks[0].relations == {}
    # The integration never installs/replaces the original function.
    monkeypatch.undo()
    assert (csp._eligibility_reason if stage == "eligibility" else csp._Work.eligibility) is original
    assert csp.solve_endpoint_hypotheses(readings, rows, **options).status == state


def test_classification_exception_preserves_nonzero_assignment_and_original_refusal(monkeypatch):
    error = ValueError("private/path")
    original = csp._Work.eligibility
    calls = []
    def eligibility(work, index, q, reason):
        calls.append(index)
        if index == 1:
            raise error
        return original(work, index, q, reason)
    monkeypatch.setattr(csp._Work, "eligibility", eligibility)
    readings, rows, options = _fixture()
    events = []
    with pytest.raises(ValueError) as caught:
        csp.solve_endpoint_hypotheses(readings, rows, **options,
                        endpoint_incidence_sink=lambda e, p: events.append((e, p)))
    assert caught.value is error and calls == [0, 1]
    assert _event(events, "exception") == [{"stage": "classification", "assignment_index": 1,
                                          "result_available": True, "reason": "no_active_factors"}]
    assert [p["assignment_index"] for p in _event(events, "returned")] == [0, 1]
    assert [p["assignment_index"] for p in _event(events, "classified")] == [0]
    assert "private/path" not in repr(events)


@pytest.mark.parametrize("kind", ["foreign", "duplicate", "alias", "path"])
def test_relation_identity_join_or_foreign_labels_fail_closed(kind):
    readings, rows, _ = _fixture()
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    relation = {(0, 0): [0]}
    u, v = ("image-0000", 0), ("image-0001", 0)
    registry = {(u, v, 0): relation}
    events = []
    hook = csp._EndpointIncidenceHooks(lambda e, p: events.append((e, p)))
    if kind == "alias": registry[(u, v, 1)] = relation
    if kind == "path":
        indexed = {("private/path", 0): next(iter(indexed.values()))}
    hook.capture("registry", registry, indexed, 1, "a" * 64)
    if kind in ("foreign", "duplicate"):
        active = [(u, v, dict(relation))] if kind == "foreign" else [(u, v, relation)] * 2
        hook.capture("entered", 0, {"image-0000": 0, "image-0001": 0}, active)
    assert not hook.complete and _event(events, "incomplete")
    assert not _event(events, "entered")
    hook.close()


def test_selector_primary_capture_and_validation_replay_absence_with_proof_parity(monkeypatch):
    readings, rows, options = _fixture()
    arguments = dict(readings=readings, signatures=_evidence(readings), candidate_matches=rows,
        frame_ids=options["session_frames"], quadrant_constraints=(), budget=options["budget"],
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    original = csp.solve_endpoint_hypotheses
    calls = []
    def solve(*args, **kwargs):
        calls.append(kwargs.get("endpoint_incidence_sink"))
        return original(*args, **kwargs)
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", solve)
    expected = selection.select_endpoint_hypotheses(**arguments)
    events = []
    sink = lambda e, p: events.append((e, p))
    actual = selection.select_endpoint_hypotheses(**arguments, endpoint_incidence_sink=sink)
    assert actual == expected and actual.proof_json == expected.proof_json
    assert actual.result_sha256 == expected.result_sha256
    captured = tuple(events)
    selection.validate_candidate_certificate(actual, readings, arguments["signatures"], rows)
    assert tuple(events) == captured and calls == [None, sink, None]
    assert len(_event(events, "registry")) == len(_event(events, "finished")) == 1


@pytest.mark.parametrize("alternatives", [False, True])
def test_graph_primary_only_capture_preserves_payload_digest_and_call_counts(monkeypatch, alternatives):
    sources = _source_records(monkeypatch, alternatives=alternatives)
    counts = dict(matcher=0, selector=0, evaluator=0)
    for name, module, attr in (("matcher", plane_graph, "match_plane_graph"),
                               ("selector", selection, "select_endpoint_hypotheses"),
                               ("evaluator", plane_graph, "_evaluate_assignment")):
        original = getattr(module, attr)
        def wrapper(*args, _name=name, _fn=original, **kwargs):
            counts[_name] += 1
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, attr, wrapper)
    original_solve, solve_sinks = csp.solve_endpoint_hypotheses, []
    def solve(*args, **kwargs):
        solve_sinks.append(kwargs.get("endpoint_incidence_sink"))
        return original_solve(*args, **kwargs)
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", solve)
    expected = graph.derive_image_derived_graph(**sources)
    expected_counts = dict(counts)
    before = len(solve_sinks)
    events = []
    sink = lambda e, p: events.append((e, p))
    actual = graph.derive_image_derived_graph(**sources, endpoint_incidence_sink=sink)
    assert actual.result_sha256 == expected.result_sha256
    assert graph._payload(actual) == graph._payload(expected)
    assert actual.selection.proof_json == expected.selection.proof_json
    assert counts == {key: 2 * value for key, value in expected_counts.items()}
    assert expected_counts == dict(matcher=1, selector=1, evaluator=0 if alternatives else 1)
    assert solve_sinks[before] is sink and all(s is None for s in solve_sinks[before + 1:])
    assert len(_event(events, "registry")) == len(_event(events, "finished")) == 1
