"""Synthetic only: load disconnected module without dataset's production exports."""

import ast
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SOURCE = Path(__file__).parents[1] / "src/mc_photogrammetry/dataset/_private_capture_contract.py"
SPEC = importlib.util.spec_from_file_location("_synthetic_private_capture_contract", SOURCE)
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)

Action = contract.Action
Resource = contract.Resource
Control = contract.Control
Evidence = contract.Evidence
Budget = contract.Budget
ControlEvidence = contract.ControlEvidence
AdmissionPolicy = contract.AdmissionPolicy
Event = contract.Event
Usage = contract.Usage
Report = contract.Report
start = contract.start
advance = contract.advance


def policy():
    # All numbers/identities are synthetic assertions, not host recommendations.
    evidence = Evidence("synthetic-reviewed-source", "a" * 64, "synthetic-run",
                        "private-test-boundary", True, True)
    return AdmissionPolicy(evidence.run_id, evidence.boundary_id,
                           tuple(ControlEvidence(c, evidence) for c in Control),
                           tuple(Budget(r, 100, 100, evidence) for r in Resource))


def event(state, **changes):
    values = dict(action=state.action, now_ns=state.last_ns + 1,
                  supervisor_alive=True, supervisor_owns_job_handle=True,
                  boundary_id=state.policy.private_boundary_id,
                  private_boundary_confirmed=True, suspended_confirmed=True,
                  launcher_handle_owned=True, membership_confirmed=True,
                  job_limits_confirmed=True, no_breakaway_confirmed=True,
                  kill_on_close_confirmed=True, termination_confirmed=True,
                  owned_launcher_tree_reaped_confirmed=True,
                  usage=Usage(active_members=0 if state.action in {
                      Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED} else 1))
    values.update(changes)
    return Event(**values)


def at(action):
    state = start(policy(), 0)
    for _ in range(6):
        if state.action is action:
            return state
        state = advance(state, event(state))
    raise AssertionError("setup action not reached")


def finish(state):
    if state.action is Action.TEARDOWN:
        state = advance(state, event(state))
    if state.action is Action.VERIFY_ZERO:
        state = advance(state, event(state, usage=Usage(active_members=0)))
    assert state.action is Action.EMIT
    return advance(state, event(state))


def test_correct_order_and_complete_report_replay():
    state = start(policy(), 40)
    actual = []
    for expected in (Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED, Action.ASSIGN_JOB,
                     Action.CONFIRM_JOB, Action.RESUME):
        assert state.action is expected
        actual.append(state.action)
        state = advance(state, event(state))
    assert state.action is Action.OBSERVE and state.resumed
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    assert state.action is Action.TEARDOWN
    state = advance(state, event(state, usage=Usage(active_members=0)))
    assert state.action is Action.VERIFY_ZERO  # Termination alone never certifies zero.
    state = advance(state, event(state, usage=Usage(active_members=0)))
    assert state.action is Action.EMIT and state.report is None
    state = advance(state, event(state))
    assert state.action is Action.DONE
    assert state.report.outcome == "synthetic_complete"
    assert state.report.teardown_verified
    assert Report.from_json(state.report.to_json()) == state.report
    assert len(state.report.to_json()) <= contract.MAX_REPORT_BYTES
    with pytest.raises(ValueError, match="terminal"):
        advance(state, event(state))


@pytest.mark.parametrize("action", [Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED,
                                   Action.ASSIGN_JOB, Action.CONFIRM_JOB, Action.RESUME])
def test_failure_at_each_setup_edge_never_resumes(action):
    state = at(action)
    state = advance(state, event(state, success=False))
    assert not state.resumed
    state = finish(state)
    assert state.report.outcome == "refused" and not state.report.resumed


@pytest.mark.parametrize("action,flag", [
    (Action.SELECT_PRIVATE, "private_boundary_confirmed"),
    (Action.CREATE_SUSPENDED, "suspended_confirmed"),
    (Action.CREATE_SUSPENDED, "launcher_handle_owned"),
    (Action.CREATE_SUSPENDED, "private_boundary_confirmed"),
    (Action.ASSIGN_JOB, "membership_confirmed"),
    (Action.CONFIRM_JOB, "job_limits_confirmed"),
    (Action.CONFIRM_JOB, "membership_confirmed"),
    (Action.CONFIRM_JOB, "no_breakaway_confirmed"),
    (Action.CONFIRM_JOB, "kill_on_close_confirmed"),
    (Action.RESUME, "membership_confirmed"),
])
def test_positive_setup_attestations_required(action, flag):
    state = at(action)
    failed = advance(state, event(state, **{flag: False}))
    assert failed.reason and not failed.resumed


@pytest.mark.parametrize("control", list(Control))
def test_missing_each_control_refuses_before_creation(control):
    original = policy()
    missing = replace(original, controls=tuple(c for c in original.controls
                                             if c.control != control))
    state = start(missing, 0)
    assert state.action is Action.EMIT and not state.process_may_exist
    assert finish(state).report.reason == "missing_control_evidence"


@pytest.mark.parametrize("resource", list(Resource))
def test_missing_each_cap_refuses_before_creation(resource):
    original = policy()
    missing = replace(original, budgets=tuple(b for b in original.budgets
                                            if b.resource != resource))
    state = start(missing, 0)
    assert state.action is Action.EMIT and not state.process_may_exist
    assert finish(state).report.reason == "missing_budget_evidence"


@pytest.mark.parametrize("resource", list(Resource))
@pytest.mark.parametrize("value,admitted", [(-1, False), (0, False), (1, True),
                                          (100, True), (101, False)])
def test_invalid_exact_and_one_over_budgets(resource, value, admitted):
    original = policy()
    modified = replace(original, budgets=tuple(replace(b, limit=value)
                       if b.resource is resource else b for b in original.budgets))
    state = start(modified, 0)
    assert (state.action is Action.SELECT_PRIVATE) is admitted
    assert not state.process_may_exist


@pytest.mark.parametrize("field,value", [("authority", ""), ("artifact_sha256", "bad")])
def test_missing_provenance_is_structurally_rejected(field, value):
    original = policy()
    evidence = replace(original.controls[0].evidence, **{field: value})
    with pytest.raises(ValueError):
        replace(original, controls=(replace(original.controls[0], evidence=evidence),)
                                  + original.controls[1:])


@pytest.mark.parametrize("field,value", [("run_id", "other"), ("boundary_id", "other"),
    ("established_before_creation", False), ("covers_launcher_and_all_descendants", False)])
@pytest.mark.parametrize("kind", ["controls", "budgets"])
def test_evidence_scope_and_preexistence_required(field, value, kind):
    original = policy()
    items = getattr(original, kind)
    items = (replace(items[0], evidence=replace(items[0].evidence, **{field: value})),) + items[1:]
    state = start(replace(original, **{kind: items}), 0)
    assert state.action is Action.EMIT and state.reason == "unqualified_evidence"


@pytest.mark.parametrize("kind", ["controls", "budgets"])
def test_duplicate_evidence_tags_do_not_replace_missing_control(kind):
    original = policy()
    items = getattr(original, kind)
    state = start(replace(original, **{kind: items[:-1] + (items[0],)}), 0)
    assert state.action is Action.EMIT


def test_launcher_exit_with_surviving_java_is_not_completion():
    state = at(Action.OBSERVE)
    state = advance(state, event(state, launcher_exited=True, client_seen=True,
                                 usage=Usage(active_members=2)))
    assert state.action is Action.OBSERVE
    assert not state.teardown_verified and state.report is None
    state = advance(state, event(state, launcher_exited=True, sentinel_complete=True,
                                 usage=Usage(active_members=1)))
    assert finish(state).report.outcome == "synthetic_complete"


def test_sentinel_without_client_does_not_complete():
    state = at(Action.OBSERVE)
    state = advance(state, event(state, sentinel_complete=True))
    assert state.action is Action.OBSERVE
    state = advance(state, event(state, usage=Usage(active_members=0)))
    assert finish(state).report.outcome == "refused"


@pytest.mark.parametrize("action", [Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED,
    Action.ASSIGN_JOB, Action.CONFIRM_JOB, Action.RESUME, Action.OBSERVE])
@pytest.mark.parametrize("field", ["supervisor_alive", "supervisor_owns_job_handle"])
def test_supervisor_loss_never_claims_verified_teardown(action, field):
    state = at(action)
    confirmed = state.containment_confirmed
    failed = advance(state, event(state, **{field: False}))
    final = finish(failed)
    assert final.report.outcome == "refused"
    assert not final.report.teardown_verified
    assert final.report.kill_on_close_assumed is confirmed


@pytest.mark.parametrize("action", [Action.CREATE_SUSPENDED, Action.ASSIGN_JOB])
def test_unassigned_suspended_process_needs_owned_tree_reaping(action):
    state = at(action)
    state = advance(state, event(state, success=False))
    assert state.action is Action.TEARDOWN and not state.job_assigned
    state = advance(state, event(state, owned_launcher_tree_reaped_confirmed=False,
                                 usage=Usage(active_members=0)))
    assert not finish(state).report.teardown_verified


@pytest.mark.parametrize("members", [None, 1, 2])
def test_uncertain_or_nonzero_teardown_is_refusal(members):
    state = at(Action.OBSERVE)
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    state = advance(state, event(state))
    state = advance(state, event(state, usage=Usage(active_members=members)))
    result = finish(state).report
    assert result.outcome == "refused" and not result.teardown_verified


@pytest.mark.parametrize("field", ["success", "termination_confirmed"])
def test_failed_termination_is_refusal(field):
    state = at(Action.OBSERVE)
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    state = advance(state, event(state, **{field: False}))
    assert not finish(state).report.teardown_verified


def test_wall_deadline_starts_before_first_setup_and_covers_setup():
    state = start(policy(), 500)
    state = advance(state, event(state, now_ns=601))
    assert state.reason == "wall_deadline" and not state.process_may_exist
    assert not finish(state).report.resumed
    state = start(policy(), 500)
    state = advance(state, event(state, now_ns=600))
    assert state.action is Action.CREATE_SUSPENDED  # Exact boundary accepted.
    state = advance(state, event(state, now_ns=601))
    assert state.reason == "wall_deadline" and not state.resumed


@pytest.mark.parametrize("elapsed,verified", [(100, True), (101, False)])
def test_cleanup_has_separate_bounded_time(elapsed, verified):
    state = at(Action.OBSERVE)
    state = advance(state, event(state, now_ns=100, client_seen=True, sentinel_complete=True))
    state = advance(state, event(state, now_ns=100 + elapsed))
    if state.action is Action.VERIFY_ZERO:
        state = advance(state, event(state, now_ns=100 + elapsed, usage=Usage(active_members=0)))
    result = finish(state).report
    assert result.teardown_verified is verified
    assert (result.outcome == "synthetic_complete") is verified


def test_cleanup_bad_order_does_not_restart_cleanup_deadline():
    state = at(Action.OBSERVE)
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    began = state.cleanup_started_ns
    state = advance(state, event(state, action=Action.OBSERVE))
    assert state.action is Action.EMIT and state.cleanup_started_ns == began
    assert finish(state).report.outcome == "refused"


def test_incremental_output_exact_cap_and_one_over_are_bounded():
    state = at(Action.OBSERVE)
    state = advance(state, event(state, output_bytes=40))
    state = advance(state, event(state, output_bytes=60))
    assert state.output_bytes == 100 and state.action is Action.OBSERVE
    state = advance(state, event(state, output_bytes=1))
    assert state.action is Action.TEARDOWN and state.output_overflow
    result = finish(state).report
    assert result.output_bytes == 100 and result.output_overflow
    assert result.outcome == "refused"


def test_huge_output_increment_is_not_buffered_or_retained():
    state = at(Action.OBSERVE)
    state = advance(state, event(state, output_bytes=contract.MAX_INTEGER))
    assert state.output_bytes == 100 and state.output_overflow
    assert len(finish(state).report.to_json()) < contract.MAX_REPORT_BYTES


def test_failed_setup_output_is_counted_before_failure():
    state = at(Action.CREATE_SUSPENDED)
    state = advance(state, event(state, success=False, output_bytes=30))
    assert finish(state).report.output_bytes == 30


@pytest.mark.parametrize("field", ["memory_bytes", "active_members", "cpu_rate_units",
                                   "cpu_time_ns", "storage_bytes"])
@pytest.mark.parametrize("reading,refused", [(100, False), (101, True)])
def test_aggregate_cap_exact_and_one_over_including_storage(field, reading, refused):
    state = at(Action.OBSERVE)
    usage = replace(Usage(active_members=1), **{field: reading})
    state = advance(state, event(state, usage=usage))
    assert bool(state.reason) is refused
    if refused:
        assert finish(state).report.reason == "resource_cap_overrun"


def test_unknown_membership_and_cpu_time_regression_refuse():
    state = at(Action.OBSERVE)
    failed = advance(state, event(state, usage=Usage(active_members=None)))
    assert failed.reason == "uncertain_membership"
    state = advance(state, event(state, usage=Usage(active_members=1, cpu_time_ns=50)))
    state = advance(state, event(state, usage=Usage(active_members=1, cpu_time_ns=49)))
    assert state.reason == "cpu_time_regression"


@pytest.mark.parametrize("action", [Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED,
    Action.ASSIGN_JOB, Action.CONFIRM_JOB, Action.RESUME, Action.OBSERVE])
def test_interactive_fallback_at_every_active_edge_refuses(action):
    state = at(action)
    state = advance(state, event(state, interactive_fallback=True))
    assert finish(state).report.outcome == "refused"


def test_policy_interactive_fallback_refuses_before_setup():
    state = start(replace(policy(), allow_interactive_fallback=True), 0)
    assert state.action is Action.EMIT and state.reason == "interactive_fallback"


@pytest.mark.parametrize("action", [Action.SELECT_PRIVATE, Action.CREATE_SUSPENDED])
def test_selected_boundary_cannot_be_replaced(action):
    state = at(action)
    state = advance(state, event(state, boundary_id="interactive-or-other"))
    assert state.reason and not state.resumed


def test_out_of_order_resume_and_nonmonotonic_events_refuse():
    state = at(Action.CREATE_SUSPENDED)
    failed = advance(state, event(state, action=Action.RESUME))
    assert failed.reason == "transition_order" and not failed.resumed
    failed = advance(state, event(state, now_ns=state.last_ns - 1))
    assert failed.reason == "nonmonotonic_time" and not failed.resumed


def test_immutability_no_mutable_inputs_and_deterministic_replay():
    original = policy()
    objects = (original, original.controls[0], original.controls[0].evidence,
               original.budgets[0], start(original, 0), Event(Action.SELECT_PRIVATE, 0), Usage())
    for obj in objects:
        field = next(iter(obj.__dataclass_fields__))
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)
    with pytest.raises(ValueError):
        replace(original, controls=list(original.controls))
    with pytest.raises(ValueError):
        replace(original, budgets=list(original.budgets))
    assert original.identity() == policy().identity()
    altered = replace(original, budgets=tuple(replace(b, limit=99)
                      if b.resource is Resource.OUTPUT else b for b in original.budgets))
    assert original.identity() != altered.identity()
    first = start(original, 10)
    second = start(original, 10)
    injected = event(first)
    assert advance(first, injected) == advance(second, injected)
    assert first.action is Action.SELECT_PRIVATE


@pytest.mark.parametrize("value", [True, 1.5, "1", -1, contract.MAX_INTEGER + 1])
def test_malformed_counters_are_rejected_before_state(value):
    with pytest.raises(ValueError):
        Event(Action.OBSERVE, 1, output_bytes=value)
    with pytest.raises(ValueError):
        Event(Action.OBSERVE, 1, usage=Usage(storage_bytes=value))


@pytest.mark.parametrize("mutation", ["extra", "missing", "duplicate", "nested", "overflow",
                                      "false_completion", "bad_boolean"])
def test_bounded_report_replay_refuses_malformed_records(mutation):
    state = start(replace(policy(), allow_interactive_fallback=True), 0)
    record = json.loads(finish(state).report.to_json())
    if mutation == "extra":
        record["extra"] = 1
    elif mutation == "missing":
        del record["reason"]
    elif mutation == "duplicate":
        raw = '{"run_id":"a",' + json.dumps(record)[1:]
    elif mutation == "nested":
        record["outcome"] = []
    elif mutation == "overflow":
        raw = " " * (contract.MAX_REPORT_BYTES + 1)
    elif mutation == "false_completion":
        record["outcome"] = "synthetic_complete"
        record["reason"] = ""
    elif mutation == "bad_boolean":
        record["resumed"] = 0
    if mutation not in {"duplicate", "overflow"}:
        raw = json.dumps(record)
    with pytest.raises(ValueError):
        Report.from_json(raw)


def test_private_contract_has_only_pure_standard_library_dependencies():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {"__future__", "dataclasses", "enum", "hashlib", "json"}
    imports = {node.module.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom)}
    imports |= {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert imports <= allowed
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "exec", "eval", "__import__"}


@pytest.mark.parametrize("resource", list(Resource))
@pytest.mark.parametrize("ceiling", [-1, 0])
def test_nonpositive_reviewed_ceilings_refuse(resource, ceiling):
    original = policy()
    original = replace(original, budgets=tuple(replace(b, reviewed_ceiling=ceiling)
                       if b.resource is resource else b for b in original.budgets))
    state = start(original, 0)
    assert state.action is Action.EMIT and state.reason == "invalid_budget"


@pytest.mark.parametrize("field", ["memory_bytes", "active_members", "cpu_rate_units",
                                   "cpu_time_ns", "storage_bytes"])
def test_resource_overruns_during_cleanup_cannot_report_completion(field):
    state = at(Action.OBSERVE)
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    state = advance(state, event(state, usage=replace(Usage(active_members=1), **{field: 101})))
    result = finish(state).report
    assert result.outcome == "refused" and result.reason == "resource_cap_overrun"


@pytest.mark.parametrize("action", [Action.TEARDOWN, Action.VERIFY_ZERO, Action.EMIT])
def test_fallback_during_cleanup_or_emission_refuses(action):
    state = at(Action.OBSERVE)
    state = advance(state, event(state, client_seen=True, sentinel_complete=True))
    if action in {Action.VERIFY_ZERO, Action.EMIT}:
        state = advance(state, event(state))
    if action is Action.EMIT:
        state = advance(state, event(state, usage=Usage(active_members=0)))
    state = advance(state, event(state, interactive_fallback=True))
    if state.action is not Action.DONE:
        state = finish(state)
    assert state.report.outcome == "refused"


def test_nonmonotonic_create_still_requires_cleanup_for_possible_process():
    state = at(Action.CREATE_SUSPENDED)
    state = advance(state, event(state, now_ns=0))
    assert state.action is Action.TEARDOWN and state.process_may_exist
    assert not state.resumed


def test_report_immutability_and_schema_version():
    report = finish(start(replace(policy(), allow_interactive_fallback=True), 0)).report
    with pytest.raises(FrozenInstanceError):
        report.reason = "altered"
    record = json.loads(report.to_json())
    record["contract_version"] = "another-version"
    with pytest.raises(ValueError):
        Report.from_json(json.dumps(record))


def test_structural_bounds_are_independent_of_resource_caps():
    original = policy()
    with pytest.raises(ValueError):
        replace(original, run_id="r" * (contract.MAX_TEXT + 1))
    with pytest.raises(ValueError):
        replace(original, controls=original.controls + (original.controls[0],))
    with pytest.raises(ValueError):
        replace(original, budgets=original.budgets + (original.budgets[0],))
    with pytest.raises(ValueError):
        replace(original, budgets=(replace(original.budgets[0], limit=True),)
                                  + original.budgets[1:])
    with pytest.raises(ValueError):
        replace(original, budgets=(replace(original.budgets[0], limit=contract.MAX_INTEGER + 1),)
                                  + original.budgets[1:])


def test_absent_supervisor_attestations_refuse_closed():
    state = start(policy(), 0)
    state = advance(state, Event(Action.SELECT_PRIVATE, 1))
    assert state.action is Action.EMIT and state.reason == "supervisor_or_owner_lost"
    assert not finish(state).report.resumed


def test_zero_preassignment_then_confirmed_positive_membership_can_resume():
    state = at(Action.CREATE_SUSPENDED)
    state = advance(state, event(state, membership_confirmed=False,
                                 usage=Usage(active_members=0)))
    assert state.action is Action.ASSIGN_JOB and state.process_may_exist
    assert not state.reason and not state.job_assigned and not state.resumed
    state = advance(state, event(state, membership_confirmed=True,
                                 usage=Usage(active_members=1)))
    assert state.action is Action.CONFIRM_JOB and state.job_assigned
    assert not state.containment_confirmed and not state.resumed
    state = advance(state, event(state, usage=Usage(active_members=1)))
    assert state.action is Action.RESUME and state.containment_confirmed
    state = advance(state, event(state, usage=Usage(active_members=1)))
    assert state.action is Action.OBSERVE and state.resumed and not state.reason


@pytest.mark.parametrize("members,confirmed,success,reason", [
    (0, True, True, "job_assignment_unconfirmed"),
    (1, False, True, "job_assignment_unconfirmed"),
    (None, True, True, "uncertain_membership"),
    (1, True, False, "setup_or_observation_failed"),
])
@pytest.mark.parametrize("reaped", [False, True])
def test_failed_assignment_requires_independent_unassigned_tree_reaping(
        members, confirmed, success, reason, reaped):
    state = at(Action.ASSIGN_JOB)
    state = advance(state, event(state, membership_confirmed=confirmed, success=success,
                                 usage=Usage(active_members=members)))
    assert state.action is Action.TEARDOWN and state.reason == reason
    assert state.process_may_exist and not state.job_assigned and not state.resumed
    state = advance(state, event(state, owned_launcher_tree_reaped_confirmed=reaped,
                                 usage=Usage(active_members=0)))
    assert state.action is (Action.VERIFY_ZERO if reaped else Action.EMIT)
    result = finish(state).report
    assert result.outcome == "refused" and result.reason == reason
    assert result.teardown_verified is reaped and not result.resumed


@pytest.mark.parametrize("action", [Action.CONFIRM_JOB, Action.RESUME])
def test_postassignment_setup_requires_positive_job_membership(action):
    state = at(action)
    assert state.job_assigned
    state = advance(state, event(state, usage=Usage(active_members=0)))
    assert state.reason and not state.resumed


def test_rejected_transition_output_still_counts():
    state = at(Action.CREATE_SUSPENDED)
    state = advance(state, event(state, action=Action.RESUME, output_bytes=101))
    assert state.output_bytes == 100 and state.output_overflow
    assert not finish(state).report.resumed
