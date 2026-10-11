"""Disconnected synthetic orchestration contract; contains no native adapter.

Actions are inert descriptions. Callers inject completed transition evidence and
aggregate telemetry, never commands or callbacks. Admission is an assertion of
externally reviewed evidence, not proof that any host implements these controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json


MAX_TEXT = 128  # Structural record bounds, never host resource allowances.
MAX_INTEGER = (1 << 63) - 1
MAX_REPORT_BYTES = 4096
CONTRACT_VERSION = "private-capture-synthetic-v1"


class Control(str, Enum):
    PRIVATE_DESKTOP = "private_desktop_window_station_no_fallback"
    NO_HOST_AUDIO = "preexisting_no_host_audio"
    INPUT_MOUSE = "input_mouse_boundary"
    DATA_ISOLATION = "identity_access_ipc_data_root_boundary"
    OWNED_HANDLES = "owned_launcher_job_supervisor_handles"
    CONTAINMENT = "no_breakaway_kill_on_close"


class Resource(str, Enum):
    MEMORY = "aggregate_committed_memory_bytes"
    PROCESSES = "active_processes"
    CPU_RATE = "aggregate_cpu_rate_units"
    CPU_TIME = "aggregate_cpu_time_ns"
    WALL = "wall_deadline_ns"
    CLEANUP = "cleanup_time_ns"
    OUTPUT = "cumulative_output_bytes"
    STORAGE = "writable_storage_bytes"


def _text(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= MAX_TEXT


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= MAX_INTEGER


def _digest(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


@dataclass(frozen=True, slots=True)
class Evidence:
    """Opaque reviewed evidence identity, covering launcher and all descendants."""

    authority: str
    artifact_sha256: str
    run_id: str
    boundary_id: str
    established_before_creation: bool
    covers_launcher_and_all_descendants: bool

    def valid_for(self, run_id: str, boundary_id: str) -> bool:
        return (_text(self.authority) and _digest(self.artifact_sha256)
                and self.run_id == run_id and self.boundary_id == boundary_id
                and self.established_before_creation is True
                and self.covers_launcher_and_all_descendants is True)


@dataclass(frozen=True, slots=True)
class ControlEvidence:
    control: Control
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class Budget:
    """Requested cap and reviewed maximum, in the named resource's units.

    Both are supplied explicitly. A ceiling is an external allowance assertion,
    not a guessed host capacity; its evidence must cover aggregate enforcement.
    """

    resource: Resource
    limit: int
    reviewed_ceiling: int
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    run_id: str
    private_boundary_id: str
    controls: tuple[ControlEvidence, ...]
    budgets: tuple[Budget, ...]
    allow_interactive_fallback: bool = False

    def __post_init__(self) -> None:
        # No mutable containers, subclasses or arbitrary user objects enter state.
        if (type(self.controls) is not tuple or len(self.controls) > len(Control)
                or any(type(c) is not ControlEvidence for c in self.controls)
                or type(self.budgets) is not tuple or len(self.budgets) > len(Resource)
                or any(type(b) is not Budget for b in self.budgets)):
            raise ValueError("policy requires bounded exact typed tuples")
        for item in self.controls + self.budgets:
            if type(item.evidence) is not Evidence:
                raise ValueError("policy requires immutable evidence")
        for evidence in (item.evidence for item in self.controls + self.budgets):
            if (not all(_text(v) for v in (evidence.authority, evidence.run_id,
                                          evidence.boundary_id))
                    or not _digest(evidence.artifact_sha256)
                    or type(evidence.established_before_creation) is not bool
                    or type(evidence.covers_launcher_and_all_descendants) is not bool):
                raise ValueError("invalid bounded evidence record")
        if not _text(self.run_id) or not _text(self.private_boundary_id):
            raise ValueError("invalid policy identity")
        if type(self.allow_interactive_fallback) is not bool:
            raise ValueError("fallback flag must be boolean")
        if any(type(c.control) is not Control for c in self.controls):
            raise ValueError("invalid control tag")
        if any(type(b.resource) is not Resource or type(b.limit) is not int
               or type(b.reviewed_ceiling) is not int
               or abs(b.limit) > MAX_INTEGER or abs(b.reviewed_ceiling) > MAX_INTEGER
               for b in self.budgets):
            raise ValueError("invalid bounded budget record")

    def refusal(self) -> str:
        if self.allow_interactive_fallback:
            return "interactive_fallback"
        if {c.control for c in self.controls} != set(Control) or len(self.controls) != len(Control):
            return "missing_control_evidence"
        if ({b.resource for b in self.budgets} != set(Resource)
                or len(self.budgets) != len(Resource)):
            return "missing_budget_evidence"
        if any(not item.evidence.valid_for(self.run_id, self.private_boundary_id)
               for item in self.controls + self.budgets):
            return "unqualified_evidence"
        if any(not _integer(b.limit, 1) or not _integer(b.reviewed_ceiling, 1)
               or b.limit > b.reviewed_ceiling for b in self.budgets):
            return "invalid_budget"
        return ""

    def cap(self, resource: Resource) -> int:
        return next(b.limit for b in self.budgets if b.resource is resource)

    def identity(self) -> str:
        raw = json.dumps({"contract_version": CONTRACT_VERSION, "policy": asdict(self)},
                         sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


class Action(str, Enum):
    SELECT_PRIVATE = "select_or_create_private_desktop"
    CREATE_SUSPENDED = "create_prism_suspended"
    ASSIGN_JOB = "assign_process_to_job"
    CONFIRM_JOB = "confirm_job_limits_and_containment"
    RESUME = "resume"
    OBSERVE = "observe_client_and_sentinel"
    TEARDOWN = "initiate_job_teardown"
    VERIFY_ZERO = "verify_zero_active_members"
    EMIT = "emit_bounded_evidence"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class Usage:
    """Aggregate readings for the entire job; CPU time is cumulative.

    None for membership means uncertainty and is never treated as zero.
    Storage is the whole writable boundary's occupied bytes, including ancestors.
    """

    memory_bytes: int = 0
    active_members: int | None = 0
    cpu_rate_units: int = 0
    cpu_time_ns: int = 0
    storage_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Event:
    action: Action
    now_ns: int
    success: bool = True
    boundary_id: str = ""
    interactive_fallback: bool = False
    supervisor_alive: bool = False
    supervisor_owns_job_handle: bool = False
    launcher_handle_owned: bool = False
    private_boundary_confirmed: bool = False
    suspended_confirmed: bool = False
    membership_confirmed: bool = False
    job_limits_confirmed: bool = False
    no_breakaway_confirmed: bool = False
    kill_on_close_confirmed: bool = False
    termination_confirmed: bool = False
    owned_launcher_tree_reaped_confirmed: bool = False
    client_seen: bool = False
    sentinel_complete: bool = False
    launcher_exited: bool = False
    output_bytes: int = 0
    usage: Usage = Usage()

    def __post_init__(self) -> None:
        if type(self.action) is not Action or not _integer(self.now_ns):
            raise ValueError("invalid action/time")
        if type(self.usage) is not Usage or not _integer(self.output_bytes):
            raise ValueError("invalid usage/output")
        if type(self.boundary_id) is not str or len(self.boundary_id) > MAX_TEXT:
            raise ValueError("invalid boundary identity")
        for name in self.__dataclass_fields__:
            if name not in {"action", "now_ns", "boundary_id", "output_bytes", "usage"}:
                if type(getattr(self, name)) is not bool:
                    raise ValueError("event flags must be boolean")
        for name in self.usage.__dataclass_fields__:
            value = getattr(self.usage, name)
            if name == "active_members" and value is None:
                continue
            if not _integer(value):
                raise ValueError("invalid aggregate reading")


@dataclass(frozen=True, slots=True)
class Report:
    """Fixed-size synthetic summary. Replay checks structure, not host truth."""

    run_id: str
    policy_sha256: str
    outcome: str
    reason: str
    started_ns: int
    ended_ns: int
    output_bytes: int
    output_overflow: bool
    resumed: bool
    teardown_verified: bool
    kill_on_close_assumed: bool
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (type(self.contract_version) is not str or self.contract_version != CONTRACT_VERSION
                or not _text(self.run_id) or not _digest(self.policy_sha256)
                or type(self.outcome) is not str
                or self.outcome not in {"synthetic_complete", "refused"}
                or type(self.reason) is not str or len(self.reason) > MAX_TEXT
                or any(not _integer(v) for v in (self.started_ns, self.ended_ns,
                                                 self.output_bytes))
                or self.ended_ns < self.started_ns
                or any(type(v) is not bool for v in (self.output_overflow, self.resumed,
                          self.teardown_verified, self.kill_on_close_assumed))):
            raise ValueError("invalid bounded report")
        if self.outcome == "synthetic_complete" and (
                self.reason or not self.resumed or not self.teardown_verified
                or self.output_overflow or self.kill_on_close_assumed):
            raise ValueError("completion requires confirmed teardown")
        if self.outcome == "refused" and not self.reason:
            raise ValueError("refusal requires reason")

    def to_json(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if len(raw.encode("ascii")) > MAX_REPORT_BYTES:
            raise ValueError("report exceeds structural bound")
        return raw

    @classmethod
    def from_json(cls, raw: str) -> Report:
        if type(raw) is not str or len(raw) > MAX_REPORT_BYTES or not raw.isascii():
            raise ValueError("invalid bounded replay input")
        def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate report key")
                result[key] = value
            return result
        try:
            record = json.loads(raw, object_pairs_hook=pairs)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid report JSON") from exc
        if type(record) is not dict or set(record) != set(cls.__dataclass_fields__):
            raise ValueError("invalid report schema")
        return cls(**record)


@dataclass(frozen=True, slots=True)
class Supervisor:
    policy: AdmissionPolicy
    action: Action
    started_ns: int
    last_ns: int
    cleanup_started_ns: int | None = None
    process_may_exist: bool = False
    job_assigned: bool = False
    containment_confirmed: bool = False
    resumed: bool = False
    client_seen: bool = False
    output_bytes: int = 0
    output_overflow: bool = False
    cpu_time_ns: int = 0
    reason: str = ""
    teardown_verified: bool = False
    kill_on_close_assumed: bool = False
    report: Report | None = None

    def __post_init__(self) -> None:
        if (type(self.policy) is not AdmissionPolicy or type(self.action) is not Action
                or not _integer(self.started_ns) or not _integer(self.last_ns)
                or self.last_ns < self.started_ns
                or (self.cleanup_started_ns is not None and (
                    not _integer(self.cleanup_started_ns)
                    or not self.started_ns <= self.cleanup_started_ns <= self.last_ns))
                or not _integer(self.output_bytes) or not _integer(self.cpu_time_ns)
                or type(self.reason) is not str or len(self.reason) > MAX_TEXT
                or (self.report is not None and type(self.report) is not Report)
                or any(type(getattr(self, name)) is not bool for name in (
                    "process_may_exist", "job_assigned", "containment_confirmed",
                    "resumed", "client_seen", "output_overflow",
                    "teardown_verified", "kill_on_close_assumed"))):
            raise ValueError("invalid immutable supervisor state")
        if self.policy.refusal() and self.action not in {Action.EMIT, Action.DONE}:
            raise ValueError("unadmitted policy cannot create")


def start(policy: AdmissionPolicy, now_ns: int) -> Supervisor:
    """The wall clock begins here, before any setup action is returned."""
    if type(policy) is not AdmissionPolicy or not _integer(now_ns):
        raise ValueError("exact policy and bounded monotonic timestamp required")
    reason = policy.refusal()
    return Supervisor(policy, Action.EMIT if reason else Action.SELECT_PRIVATE,
                      now_ns, now_ns, reason=reason)


def _refuse(state: Supervisor, reason: str) -> Supervisor:
    state = replace(state, reason=state.reason or reason)
    if state.cleanup_started_ns is not None:
        return replace(state, action=Action.EMIT)
    if state.process_may_exist:
        return replace(state, action=Action.TEARDOWN,
                       cleanup_started_ns=state.last_ns)
    return replace(state, action=Action.EMIT)


def _emit(state: Supervisor) -> Supervisor:
    report = Report(state.policy.run_id, state.policy.identity(),
                    "refused" if state.reason else "synthetic_complete", state.reason,
                    state.started_ns, state.last_ns, state.output_bytes,
                    state.output_overflow, state.resumed, state.teardown_verified,
                    state.kill_on_close_assumed)
    return replace(state, action=Action.DONE, report=report)


def advance(state: Supervisor, event: Event) -> Supervisor:
    """Consume one synthetic result. No loops, output buffers or native calls.

    Setup uncertainty never resumes. Once creation is attempted, refusal requests
    job teardown even if the failed adapter result cannot establish creation.
    Loss of the owner can only record the externally required kill-on-close
    assumption; it cannot manufacture a verified zero-membership observation.
    """
    if type(state) is not Supervisor or type(event) is not Event:
        raise ValueError("exact immutable state/event required")
    if state.action is Action.DONE:
        raise ValueError("terminal state")
    if state.action is Action.EMIT:
        if event.action is not Action.EMIT or event.now_ns < state.last_ns:
            raise ValueError("emit requires ordered evidence event")
        # Emission is an internal fixed-size operation, not another output producer.
        if (event.output_bytes or not event.success or event.interactive_fallback
                or not event.supervisor_alive or not event.supervisor_owns_job_handle):
            return _emit(replace(state, reason=state.reason or "evidence_emission_failed"))
        return _emit(state)
    state = replace(state, process_may_exist=state.process_may_exist
                    or state.action is Action.CREATE_SUSPENDED)
    # Count even rejected/out-of-order observations; retain no output payload.
    cap = state.policy.cap(Resource.OUTPUT)
    total = state.output_bytes + event.output_bytes
    state = replace(state, output_bytes=min(total, cap),
                    output_overflow=state.output_overflow or total > cap)
    if event.now_ns < state.last_ns:
        return _refuse(state, "nonmonotonic_time")
    state = replace(state, last_ns=event.now_ns)
    if event.action is not state.action:
        return _refuse(state, "transition_order")
    if not event.supervisor_alive or not event.supervisor_owns_job_handle:
        return replace(state, action=Action.EMIT,
                       reason=state.reason or "supervisor_or_owner_lost",
                       kill_on_close_assumed=state.containment_confirmed)
    cleanup = state.action in {Action.TEARDOWN, Action.VERIFY_ZERO}
    if cleanup:
        if state.cleanup_started_ns is None:
            raise ValueError("cleanup requires its own start time")
        if event.now_ns - state.cleanup_started_ns > state.policy.cap(Resource.CLEANUP):
            return replace(state, action=Action.EMIT,
                           reason=state.reason or "cleanup_deadline")
        if (not event.success or event.interactive_fallback or state.output_overflow
                or event.usage.active_members is None):
            return replace(state, action=Action.EMIT,
                           reason=state.reason or "unverified_teardown")
        cleanup_readings = ((Resource.MEMORY, event.usage.memory_bytes),
                            (Resource.PROCESSES, event.usage.active_members),
                            (Resource.CPU_RATE, event.usage.cpu_rate_units),
                            (Resource.CPU_TIME, event.usage.cpu_time_ns),
                            (Resource.STORAGE, event.usage.storage_bytes))
        if any(value > state.policy.cap(resource) for resource, value in cleanup_readings):
            state = replace(state, reason=state.reason or "resource_cap_overrun")
        if event.usage.cpu_time_ns < state.cpu_time_ns:
            state = replace(state, reason=state.reason or "cpu_time_regression")
        state = replace(state, cpu_time_ns=max(state.cpu_time_ns, event.usage.cpu_time_ns))
        if state.action is Action.TEARDOWN:
            if (not event.termination_confirmed or (not state.job_assigned
                    and not event.owned_launcher_tree_reaped_confirmed)):
                return replace(state, action=Action.EMIT,
                               reason=state.reason or "termination_failed")
            return replace(state, action=Action.VERIFY_ZERO)
        if not event.membership_confirmed or event.usage.active_members != 0:
            return replace(state, action=Action.EMIT,
                           reason=state.reason or "nonzero_or_uncertain_membership")
        return replace(state, action=Action.EMIT, teardown_verified=True)
    if event.interactive_fallback:
        return _refuse(state, "interactive_fallback")
    if state.output_overflow:
        return _refuse(state, "output_overflow")
    if event.now_ns - state.started_ns > state.policy.cap(Resource.WALL):
        return _refuse(state, "wall_deadline")
    if event.usage.active_members is None:
        return _refuse(state, "uncertain_membership")
    readings = ((Resource.MEMORY, event.usage.memory_bytes),
                (Resource.PROCESSES, event.usage.active_members),
                (Resource.CPU_RATE, event.usage.cpu_rate_units),
                (Resource.CPU_TIME, event.usage.cpu_time_ns),
                (Resource.STORAGE, event.usage.storage_bytes))
    if any(value > state.policy.cap(resource) for resource, value in readings):
        return _refuse(state, "resource_cap_overrun")
    if event.usage.cpu_time_ns < state.cpu_time_ns:
        return _refuse(state, "cpu_time_regression")
    state = replace(state, cpu_time_ns=event.usage.cpu_time_ns)
    if not event.success:
        return _refuse(state, "setup_or_observation_failed")
    if state.action is Action.SELECT_PRIVATE:
        if (event.boundary_id != state.policy.private_boundary_id
                or not event.private_boundary_confirmed):
            return _refuse(state, "private_boundary_unconfirmed")
        return replace(state, action=Action.CREATE_SUSPENDED)
    if state.action is Action.CREATE_SUSPENDED:
        if (event.boundary_id != state.policy.private_boundary_id
                or not event.private_boundary_confirmed or not event.suspended_confirmed
                or not event.launcher_handle_owned):
            return _refuse(state, "suspended_creation_unconfirmed")
        # Job membership may be zero until the following assignment succeeds.
        return replace(state, action=Action.ASSIGN_JOB)
    if state.action is Action.ASSIGN_JOB:
        if not event.membership_confirmed or event.usage.active_members == 0:
            return _refuse(state, "job_assignment_unconfirmed")
        return replace(state, action=Action.CONFIRM_JOB, job_assigned=True)
    if state.action is Action.CONFIRM_JOB:
        if not all((event.job_limits_confirmed, event.membership_confirmed,
                    event.no_breakaway_confirmed, event.kill_on_close_confirmed,
                    event.usage.active_members > 0)):
            return _refuse(state, "containment_unconfirmed")
        return replace(state, action=Action.RESUME, containment_confirmed=True)
    if state.action is Action.RESUME:
        if not event.membership_confirmed or event.usage.active_members == 0:
            return _refuse(state, "resume_membership_unconfirmed")
        return replace(state, action=Action.OBSERVE, resumed=True)
    if state.action is Action.OBSERVE:
        if not event.membership_confirmed:
            return _refuse(state, "observation_membership_unconfirmed")
        state = replace(state, client_seen=state.client_seen or event.client_seen)
        if event.sentinel_complete and state.client_seen:
            return replace(state, action=Action.TEARDOWN, cleanup_started_ns=state.last_ns)
        if event.usage.active_members == 0:
            return _refuse(state, "members_exited_without_completion")
        # Launcher exit with surviving descendants deliberately stays OBSERVE.
        return state
    raise ValueError("unreachable action")
