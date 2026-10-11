"""Authored-scalar reference controls for a proposed observer; no model imports."""
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import json

import pytest

MAX_ENDPOINTS = 128
MAX_ROWS = 16384
MAX_FACTORS = 4096
MAX_ASSIGNMENTS = 64
MAX_SOURCE_MEMBERS = 262144


def _frame(value):
    assert type(value) is str and len(value) == 10 and value.startswith("image-")
    assert value[6:].isascii() and value[6:].isdigit() and int(value[6:]) < 32
    return value


def _integer(value, cap):
    assert type(value) is int and 0 <= value <= cap
    return value


def _key(value):
    assert type(value) is tuple and len(value) == 2
    return (_frame(value[0]), _integer(value[1], MAX_ENDPOINTS - 1))


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def reference_projection(endpoints, registry, assignments, counts, *, capture_complete=True, source_exhaustive=True):
    """Consume direct same-pass scalar snapshots; never construct/activate factors."""
    assert type(capture_complete) is bool and type(source_exhaustive) is bool
    assert type(endpoints) is tuple and 0 < len(endpoints) <= MAX_ENDPOINTS
    assert type(assignments) is tuple and len(assignments) <= MAX_ASSIGNMENTS
    assert type(counts) is dict and set(counts) == {
        "endpoints", "candidate_rows", "registry_factors", "assignments", "active_factor_visits",
        "entered", "returned", "classified", "unclassified"}
    cap = dict(endpoints=MAX_ENDPOINTS, candidate_rows=MAX_ROWS, registry_factors=MAX_FACTORS,
               assignments=MAX_ROWS, entered=MAX_ROWS, returned=MAX_ROWS,
               classified=MAX_ROWS, unclassified=MAX_ROWS, active_factor_visits=MAX_ROWS * MAX_FACTORS)
    for name, count in counts.items(): _integer(count, cap[name])
    assert counts["endpoints"] == len(endpoints)
    assert counts["classified"] <= counts["returned"] <= counts["entered"] <= counts["assignments"]
    assert counts["unclassified"] == counts["assignments"] - counts["classified"]
    axes = {}
    for row in endpoints:
        assert type(row) is dict and set(row) == {"endpoint", "local_normal"}
        key = _key(row["endpoint"]); assert key not in axes
        axes[key] = _integer(row["local_normal"], 2)
    frames = set(key[0] for key in axes)
    candidate, degree, factor_endpoints = Counter(), Counter(), {}; edges = []
    if registry is not None:
        assert type(registry) is tuple and len(registry) <= MAX_FACTORS
        assert len(registry) == counts["registry_factors"]
        all_rows = set(); source_members = 0
        for row in registry:
            assert type(row) is dict and set(row) == {"factor_id", "endpoints", "relative_quadrant", "source_rows"}
            fid = _integer(row["factor_id"], MAX_FACTORS - 1); assert fid not in factor_endpoints
            relative = _integer(row["relative_quadrant"], 3)
            assert type(row["endpoints"]) is tuple and len(row["endpoints"]) == 2
            u, v = map(_key, row["endpoints"])
            assert u in axes and v in axes and u[0] != v[0]
            ids = row["source_rows"]
            assert type(ids) is tuple and 0 < len(ids) <= MAX_ROWS
            source_members += len(ids); assert source_members <= MAX_SOURCE_MEMBERS
            unique = {_integer(i, counts["candidate_rows"] - 1) for i in ids}
            assert not unique & all_rows
            all_rows.update(unique); factor_endpoints[fid] = (u, v)
            edges.append(dict(factor_id=fid, endpoints=[list(u), list(v)], relative_quadrant=relative, source_rows=sorted(unique)))
            for key in (u, v): candidate[key] += len(unique); degree[key] += 1
        assert all_rows == set(range(counts["candidate_rows"]))
        assert sum(candidate.values()) == 2 * counts["candidate_rows"]
        assert sum(degree.values()) == 2 * counts["registry_factors"]
    table = []; active_visits = 0; seen = set(); dispositions = []; observed = Counter()
    levels = {"visited": 0, "entered": 1, "returned": 2, "classified": 3}
    for assignment in assignments:
        assert type(assignment) is dict and set(assignment) == {
            "assignment", "quadrants", "active_factors", "state", "reason"}
        index = _integer(assignment["assignment"], MAX_ROWS - 1)
        assert index < counts["assignments"] and index not in seen
        seen.add(index)
        state = assignment["state"]; assert state in levels
        level = levels[state]
        observed["assignments"] += 1
        for name, cutoff in (("entered", 1), ("returned", 2), ("classified", 3)):
            observed[name] += level >= cutoff
        q = assignment["quadrants"]
        assert type(q) is tuple and len(q) == len(frames)
        q_frames = []
        for item in q:
            assert type(item) is tuple and len(item) == 2
            q_frames.append(_frame(item[0])); _integer(item[1], 3)
        assert len(set(q_frames)) == len(q_frames) and set(q_frames) == frames
        # Preserve exact original tuple order, not an independently sorted map.
        reason = assignment["reason"]
        assert reason in (None, "unsupported_reading_endpoint", "no_active_factors", "active_axis_mismatch",
                          "axis_0_disconnected", "axis_1_disconnected", "axis_2_disconnected")
        returned = level >= 2
        assert returned or reason is None
        active = assignment["active_factors"]; support = Counter()
        if level == 0:
            assert active is None  # pre-eligibility factor state is genuinely unknown
        else:
            assert type(active) is tuple and len(active) <= MAX_FACTORS
            assert all(type(fid) is int for fid in active) and len(set(active)) == len(active)
            if registry is not None:
                for fid in active:
                    assert fid in factor_endpoints
                    for key in factor_endpoints[fid]: support[key] += 1
            else:
                assert active == ()
            active_visits += len(active)
        known_active = registry is not None and level >= 1
        dispositions.append(dict(assignment=index, quadrants=[list(item) for item in q], state=state,
            active_factor_ids=list(active) if known_active else None,
            active_factor_count=len(active) if known_active else None,
            result_available=returned, classified=level == 3,
            first_failure_reason=reason if returned else None))
        for key in sorted(axes):
            table.append(dict(assignment=index, frame_id=key[0], reading_index=key[1], local_normal=axes[key],
                candidate_row_incidence=candidate[key] if registry is not None else None,
                registry_factor_degree=degree[key] if registry is not None else None,
                active_factor_degree=support[key] if known_active else None,
                unsupported=(support[key] == 0) if known_active else None))
    for name in ("assignments", "entered", "returned", "classified"):
        assert observed[name] <= counts[name]
    assert active_visits <= counts["active_factor_visits"]
    stage_counts_match = all(observed[name] == counts[name] for name in ("assignments", "entered", "returned", "classified"))
    effective_complete = registry is not None and capture_complete and stage_counts_match
    full = effective_complete and source_exhaustive and counts["unclassified"] == 0
    if full:
        assert all(observed[name] == counts[name] for name in ("assignments", "entered", "returned", "classified"))
        assert active_visits == counts["active_factor_visits"]
        assert sum(row["active_factor_degree"] for row in table) == 2 * active_visits
        assert len(table) == counts["endpoints"] * counts["assignments"]
    else:
        assert sum(row["active_factor_degree"] or 0 for row in table) == 2 * active_visits if registry is not None else True
    status = "unavailable" if registry is None else "complete" if full else "partial" if effective_complete else "incomplete"
    record = dict(schema="endpoint-incidence-observer-proposal-v2", authored_control=True, historical_capture=False,
        status=status, capture_complete=effective_complete, capture_declared_complete=capture_complete, source_exhaustive=source_exhaustive,
        full_conservation_applied=full, unavailable_reason="historical_identities_not_captured" if registry is None else None,
        counts=counts.copy(), observed_counts=dict(observed), observed_active_factor_visits=active_visits,
        observed_unclassified=observed["assignments"] - observed["classified"],
        missing_counts={name: counts[name] - observed[name] for name in ("assignments", "entered", "returned", "classified")},
        registry_snapshot_complete=registry is not None,
        rows=table, assignments=dispositions, registry_incidence_edges=edges if registry is not None else None)
    record["projection_sha256"] = _digest(record)
    return record


def reference_identity_join(registry_relations, active_relations):
    """Exact same-object join; no equality, planner, row grouping or activation."""
    assert len(registry_relations) <= MAX_FACTORS and len(active_relations) <= MAX_FACTORS
    ids = {id(value): index for index, value in enumerate(registry_relations)}
    assert len(ids) == len(registry_relations)
    selected = tuple(ids[id(value)] for value in active_relations)
    assert len(set(selected)) == len(selected)
    return selected


def authored():
    keys = tuple((f"image-{frame:04d}", index) for frame, size in enumerate((4, 4, 3)) for index in range(size))
    endpoints = tuple(dict(endpoint=key, local_normal=index % 3) for index, key in enumerate(keys))
    pair01 = tuple((u, v) for u in keys for v in keys if u[0] == "image-0000" and v[0] == "image-0001")
    groups = []; assignments = []; offset = 0
    for i, size in enumerate((6, 6, 6, 7)):
        groups.extend((pair, i) for pair in pair01[:size])
        assignments.append(dict(assignment=i, quadrants=(("image-0000", 0), ("image-0001", i), ("image-0002", 0)),
            active_factors=tuple(range(offset, offset + size)), state="classified", reason="unsupported_reading_endpoint"))
        offset += size
    extras = tuple((("image-0000", index), ("image-0002", right)) for index in (3, 2) for right in range(3))[:5]
    groups.extend((pair, 1) for pair in extras)
    registry = tuple(dict(factor_id=index, endpoints=pair, relative_quadrant=relative,
        source_rows=tuple(i for i in range(103) if i % 30 == index)) for index, (pair, relative) in enumerate(groups))
    counts = dict(endpoints=11, candidate_rows=103, registry_factors=30, assignments=4, active_factor_visits=25,
                  entered=4, returned=4, classified=4, unclassified=0)
    return endpoints, registry, tuple(assignments), counts


def test_authored_11_by_4_table_reconciles_rows_registry_and_active_degrees():
    record = reference_projection(*authored())
    assert record["authored_control"] and not record["historical_capture"]
    assert len(record["rows"]) == 44 and record["status"] == "complete"
    for i in range(4):
        cells = [r for r in record["rows"] if r["assignment"] == i]
        assert len(cells) == 11
        assert sum(r["candidate_row_incidence"] for r in cells) == 206
        assert sum(r["registry_factor_degree"] for r in cells) == 60
        assert any(r["unsupported"] for r in cells)
    assert sum(r["active_factor_degree"] for r in record["rows"]) == 50
    digest = record.pop("projection_sha256"); assert _digest(record) == digest
    record["rows"][0]["active_factor_degree"] += 1; assert _digest(record) != digest
    assert not any(word in json.dumps(record) for word in ("pixels", "positions", "camera", "path"))


def test_historical_aggregate_counts_are_not_a_reconstructed_table():
    endpoints, _, assignments, counts = authored()
    assignments = tuple(dict(row, active_factors=()) for row in assignments)
    result = reference_projection(endpoints, None, assignments, counts)
    assert result["status"] == "unavailable" and not result["historical_capture"]
    assert all(row[key] is None for row in result["rows"] for key in (
        "candidate_row_incidence", "registry_factor_degree", "active_factor_degree", "unsupported"))


def test_same_factor_alternative_repetition_does_not_duplicate_source_rows():
    endpoints, registry, assignments, counts = authored()
    registry = list(registry); registry[0] = dict(registry[0], source_rows=registry[0]["source_rows"] * 2)
    record = reference_projection(endpoints, tuple(registry), assignments, counts)
    assert sum(row["candidate_row_incidence"] for row in record["rows"] if row["assignment"] == 0) == 206


@pytest.mark.parametrize("kind", ["foreign_frame", "path", "duplicate_endpoint", "foreign_normal", "bool",
    "candidate_missing", "candidate_cross_factor", "foreign_factor", "duplicate_active", "duplicate_assignment",
    "unknown_field", "factor_count", "active_count", "endpoint_cap", "assignment_cap", "foreign_reason"])
def test_projection_refuses_inconsistent_or_unbounded_authored_capture(kind):
    endpoints, registry, assignments, counts = authored()
    if kind in ("foreign_frame", "path", "foreign_normal", "bool", "unknown_field"):
        endpoints = list(endpoints); endpoints[0] = dict(endpoints[0])
        if kind == "foreign_frame": endpoints[0]["endpoint"] = ("image-0032", 0)
        if kind == "path": endpoints[0]["endpoint"] = ("SECRET/path", 0)
        if kind == "foreign_normal": endpoints[0]["local_normal"] = 3
        if kind == "bool": endpoints[0]["local_normal"] = True
        if kind == "unknown_field": endpoints[0]["payload"] = "SECRET"
        endpoints = tuple(endpoints)
    if kind == "duplicate_endpoint": endpoints = endpoints[:-1] + endpoints[:1]
    if kind == "candidate_missing": registry = registry[1:]; counts = dict(counts, registry_factors=29)
    if kind == "candidate_cross_factor": registry = registry[:-1] + (dict(registry[-1], source_rows=registry[0]["source_rows"]),)
    if kind in ("foreign_factor", "duplicate_active", "duplicate_assignment"):
        assignments = list(assignments); assignments[1] = dict(assignments[1])
        if kind == "foreign_factor": assignments[1]["active_factors"] = (99,)
        if kind == "duplicate_active": assignments[1]["active_factors"] = (0, 0)
        if kind == "duplicate_assignment": assignments[1]["assignment"] = 0
        assignments = tuple(assignments)
    if kind == "factor_count": counts = dict(counts, registry_factors=31)
    if kind == "active_count": counts = dict(counts, active_factor_visits=26)
    if kind == "endpoint_cap": endpoints = endpoints * 12
    if kind == "assignment_cap": assignments = assignments * 17
    if kind == "foreign_reason": assignments = (dict(assignments[0], reason="SECRET/path"),) + assignments[1:]
    with pytest.raises(AssertionError): reference_projection(endpoints, registry, assignments, counts)


def test_relation_identity_join_rejects_alias_duplicate_and_foreign_without_equality():
    class IdentityOnly:
        def __eq__(self, other): raise AssertionError("equality must not be used")
    a, b, foreign = IdentityOnly(), IdentityOnly(), IdentityOnly()
    assert reference_identity_join((a, b), (b,)) == (1,)
    with pytest.raises(AssertionError): reference_identity_join((a, a), (a,))
    with pytest.raises(AssertionError): reference_identity_join((a, b), (b, b))
    with pytest.raises(KeyError): reference_identity_join((a, b), (foreign,))


def test_total_relation_members_are_bounded_before_set_or_json_growth():
    endpoints, registry, assignments, counts = authored()
    registry = tuple(dict(registry[i], source_rows=(i,) * MAX_ROWS) for i in range(17))
    counts = dict(counts, candidate_rows=17, registry_factors=17)
    with pytest.raises(AssertionError): reference_projection(endpoints, registry, assignments, counts)


@contextmanager
def reference_fake_scope(module, observe):
    """Forwarding design proof only: this test module never patches production."""
    original = module._eligibility_reason
    telemetry = {"status": "complete", "original_calls": 0, "eligibility_entered": 0,
                 "eligibility_returned": 0, "classification_accounting": "not_observed"}
    def forward(*args, **kwargs):
        telemetry["original_calls"] += 1
        telemetry["eligibility_entered"] += 1
        try:
            result = original(*args, **kwargs)
        except BaseException:
            telemetry["status"] = "incomplete"
            telemetry["reason"] = "original_call_exception"
            raise
        telemetry["eligibility_returned"] += 1
        try:
            observe(args, kwargs, result)
        except Exception:
            telemetry["status"] = "incomplete"
            telemetry["reason"] = "observer_failed"
        return result
    module._eligibility_reason = forward
    try:
        yield telemetry
    finally:
        module._eligibility_reason = original


@pytest.mark.parametrize("failure", [None, "original", "observer", "overflow"])
def test_proposed_wrapper_forwards_once_preserves_exact_result_exception_and_restores(failure):
    args = tuple(object() for _ in range(5)); keyword = object(); result = object()
    source_exception = ValueError("SECRET/exception-path")
    observed = []; calls = []
    def original(*actual, **kwargs):
        calls.append((actual, kwargs))
        assert all(a is b for a, b in zip(actual, args)) and kwargs["sentinel"] is keyword
        if failure == "original": raise source_exception
        return result
    def observer(actual, kwargs, returned):
        assert all(a is b for a, b in zip(actual, args)) and kwargs["sentinel"] is keyword
        assert returned is result
        observed.append(True)
        if failure == "observer": raise ValueError("SECRET/observer-path")
        if failure == "overflow": raise OverflowError("SECRET/overflow-path")
    module = SimpleNamespace(_eligibility_reason=original)
    with reference_fake_scope(module, observer) as telemetry:
        if failure == "original":
            with pytest.raises(ValueError) as caught: module._eligibility_reason(*args, sentinel=keyword)
            assert caught.value is source_exception
        else:
            assert module._eligibility_reason(*args, sentinel=keyword) is result
    assert module._eligibility_reason is original and len(calls) == telemetry["original_calls"] == 1
    assert telemetry["status"] == ("complete" if failure is None else "incomplete")
    assert len(observed) == (0 if failure == "original" else 1)
    assert telemetry["eligibility_entered"] == 1
    assert telemetry["eligibility_returned"] == (0 if failure == "original" else 1)
    assert telemetry["classification_accounting"] == "not_observed"
    assert "SECRET" not in json.dumps(telemetry)


def test_no_candidate_incidence_and_inactive_candidate_support_are_distinct():
    endpoints, registry, assignments, counts = authored()
    endpoints += (dict(endpoint=("image-0002", 3), local_normal=1),)
    counts = dict(counts, endpoints=12)
    result = reference_projection(endpoints, registry, assignments, counts)
    zero = next(r for r in result["rows"] if r["assignment"] == 0 and r["frame_id"] == "image-0002" and r["reading_index"] == 3)
    inactive = next(r for r in result["rows"] if r["assignment"] == 0 and r["frame_id"] == "image-0000" and r["reading_index"] == 3)
    assert zero["candidate_row_incidence"] == zero["registry_factor_degree"] == zero["active_factor_degree"] == 0
    assert inactive["candidate_row_incidence"] > 0 and inactive["registry_factor_degree"] > 0
    assert inactive["active_factor_degree"] == 0 and zero["unsupported"] and inactive["unsupported"]
    assert len(result["registry_incidence_edges"]) == 30
    assert sum(len(edge["source_rows"]) for edge in result["registry_incidence_edges"]) == 103


def test_original_first_failure_reason_is_carried_without_inference_from_degrees():
    endpoints, registry, assignments, counts = authored()
    assignments = (dict(assignments[0], reason="active_axis_mismatch"),) + assignments[1:]
    result = reference_projection(endpoints, registry, assignments, counts)
    assert result["assignments"][0]["first_failure_reason"] == "active_axis_mismatch"
    assert any(r["unsupported"] for r in result["rows"] if r["assignment"] == 0)


def test_exact_quadrant_tuple_order_relative_quadrants_and_active_factor_ids_survive():
    endpoints, registry, assignments, counts = authored()
    assignments = (dict(assignments[0], quadrants=tuple(reversed(assignments[0]["quadrants"]))),) + assignments[1:]
    result = reference_projection(endpoints, registry, assignments, counts)
    assert result["assignments"][0]["quadrants"] == [list(item) for item in assignments[0]["quadrants"]]
    assert [r["active_factor_ids"] for r in result["assignments"]] == [list(a["active_factors"]) for a in assignments]
    assert [r["relative_quadrant"] for r in result["registry_incidence_edges"]] == [r["relative_quadrant"] for r in registry]
    assert result["full_conservation_applied"]
    # The authored source is coherent: active identity is supplied, not selected by projector.
    for assignment in assignments:
        q = dict(assignment["quadrants"])
        for fid in assignment["active_factors"]:
            row = registry[fid]; u, v = row["endpoints"]
            assert (q[v[0]] - q[u[0]]) % 4 == row["relative_quadrant"]


@pytest.mark.parametrize("phase", ["before_entry", "eligibility_exception", "before_hash"])
def test_partial_source_states_distinguish_visited_entered_returned_and_classified(phase):
    endpoints, registry, assignments, counts = authored()
    last = dict(assignments[-1], reason=None)
    if phase == "before_entry":
        last.update(state="visited", active_factors=None)
        counts = dict(counts, entered=3, returned=3, classified=3, unclassified=1, active_factor_visits=18)
    elif phase == "eligibility_exception":
        last.update(state="entered")
        counts = dict(counts, returned=3, classified=3, unclassified=1)
    else:
        last.update(state="returned")
        counts = dict(counts, classified=3, unclassified=1)
    assignments = assignments[:-1] + (last,)
    result = reference_projection(endpoints, registry, assignments, counts, source_exhaustive=False)
    row = result["assignments"][-1]
    assert result["status"] == "partial" and not result["full_conservation_applied"]
    assert result["counts"]["assignments"] == 4 and result["counts"]["unclassified"] == 1
    assert row["classified"] is False and row["first_failure_reason"] is None
    assert row["result_available"] == (phase == "before_hash")
    assert row["active_factor_ids"] is None if phase == "before_entry" else row["active_factor_ids"] == list(last["active_factors"])
    if phase == "before_entry":
        assert all(r["active_factor_degree"] is None and r["unsupported"] is None
                   for r in result["rows"] if r["assignment"] == 3)


def test_observer_truncation_retains_missing_stage_counts_without_full_conservation():
    endpoints, registry, assignments, counts = authored()
    result = reference_projection(endpoints, registry, assignments[:2], counts, capture_complete=False)
    assert result["status"] == "incomplete" and not result["full_conservation_applied"]
    assert len(result["rows"]) == 22 and result["missing_counts"] == dict(assignments=2, entered=2, returned=2, classified=2)
    assert result["observed_active_factor_visits"] == 12 < result["counts"]["active_factor_visits"] == 25


def test_partial_capture_does_not_require_unknown_source_active_tail_to_balance():
    endpoints, registry, assignments, counts = authored()
    result = reference_projection(endpoints, registry, assignments[:3], dict(counts, active_factor_visits=26),
                                  capture_complete=False, source_exhaustive=False)
    assert not result["full_conservation_applied"] and result["observed_active_factor_visits"] == 18
    assert result["counts"]["active_factor_visits"] == 26


@pytest.mark.parametrize("kind", ["bool", "range", "foreign", "duplicate", "missing", "container"])
def test_quadrant_tuple_schema_is_closed_without_normalizing_source_order(kind):
    endpoints, registry, assignments, counts = authored()
    row = dict(assignments[0]); q = row["quadrants"]
    if kind == "bool": q = ((q[0][0], True),) + q[1:]
    if kind == "range": q = ((q[0][0], 4),) + q[1:]
    if kind == "foreign": q = (("image-0032", 0),) + q[1:]
    if kind == "duplicate": q = (q[0], q[0], q[2])
    if kind == "missing": q = q[:-1]
    if kind == "container": q = list(q)
    row["quadrants"] = q
    with pytest.raises(AssertionError): reference_projection(endpoints, registry, (row,) + assignments[1:], counts)


@pytest.mark.parametrize("kind", ["relative_bool", "relative_range", "foreign_state", "missing_result",
                                 "unclassified_count", "returned_before_entered", "classified_before_returned", "visited_active_zero"])
def test_factor_or_lifecycle_contradictions_are_rejected(kind):
    endpoints, registry, assignments, counts = authored()
    if kind.startswith("relative_"):
        registry = (dict(registry[0], relative_quadrant=True if kind == "relative_bool" else 4),) + registry[1:]
    if kind == "foreign_state": assignments = (dict(assignments[0], state="SECRET/path"),) + assignments[1:]
    if kind == "missing_result": assignments = (dict(assignments[0], state="entered"),) + assignments[1:]
    if kind == "unclassified_count": counts = dict(counts, unclassified=1)
    if kind == "returned_before_entered": counts = dict(counts, entered=3)
    if kind == "classified_before_returned": counts = dict(counts, returned=3)
    if kind == "visited_active_zero": assignments = (dict(assignments[0], state="visited", active_factors=(), reason=None),) + assignments[1:]
    with pytest.raises(AssertionError): reference_projection(endpoints, registry, assignments, counts)
