"""Synthetic evaluation only: uncertain association is not physical identity.

No production caller, image ingestion, matcher, export or legacy certificate.
Factor-count optimality depends on proposal fragmentation; even a unique optimum
may be a coherent spurious association. Independent image verification and a
source-consistent full-chain synthetic control remain prerequisites for use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph

SCHEMA = "mc-photogrammetry-synthetic-association-counterfactual-v1"
LIMITS = {"branches": 16384, "quadrant_plans": 16384, "relation_visits": 262144,
          "endpoints": 128, "alternatives": 16, "factors": 4096,
          "rows": 16384, "proof_bytes": 4 * 1024 * 1024}


@dataclass(frozen=True)
class Budget:
    branches: int = LIMITS["branches"]
    quadrant_plans: int = LIMITS["quadrant_plans"]
    relation_visits: int = LIMITS["relation_visits"]
    endpoints: int = LIMITS["endpoints"]
    alternatives: int = LIMITS["alternatives"]
    factors: int = LIMITS["factors"]
    rows: int = LIMITS["rows"]
    proof_bytes: int = LIMITS["proof_bytes"]

    def __post_init__(self):
        for name, maximum in LIMITS.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("counterfactual budgets may only decrease hard maxima")


class _Exhausted(Exception):
    pass


class _Ledger:
    def __init__(self, budget):
        self.budget, self.source_digest, self.endpoint_keys = budget, "", ()
        self.counts = dict(branches=0, quadrant_plans=0, relation_visits=0,
            proof_bytes=0, registry_rows=0, registry_factors=0, alternative_states=0,
            association_states=0, coverage_rejections=0, diameter_rejections=0,
            cycle_rejections=0, physical_relation_visits=0,
            physical_cycle_relaxations=0, rollback_visits=0, replay_modes=0)
        self.denial, self.transcript = None, hashlib.sha256()

    def bound(self, name, value):
        if value > getattr(self.budget, name):
            self.denial = {"resource": name, "attempted": value, "limit": getattr(self.budget, name)}
            raise _Exhausted

    def charge(self, name, amount=1):
        self.bound(name, self.counts[name] + amount)
        self.counts[name] += amount

    def encode(self, value, category=None):
        remaining = self.budget.proof_bytes - self.counts["proof_bytes"]
        try:
            raw = selection._encode(value, limit=remaining, name="proof_bytes")
        except selection._RecordExhausted:
            self.bound("proof_bytes", self.budget.proof_bytes + 1)
            raise AssertionError("unreachable")
        self.charge("proof_bytes", len(raw))
        self.transcript.update(raw)
        return raw

    def rollback(self):
        self.charge("relation_visits")
        self.counts["rollback_visits"] += 1


def _finish(work, status, reason, bindings, factors, modes, optimum, exhaustive):
    membership = {"mandatory": [], "possible": [], "never_selected": []}
    if exhaustive and modes:
        all_ids = set(range(len(factors)))
        possible, mandatory = set(), set(all_ids)
        for mode in modes:
            work.charge("relation_visits", len(mode["associated_factors"]))
            chosen = set(mode["associated_factors"])
            possible.update(chosen); mandatory.intersection_update(chosen)
        membership = {"mandatory": sorted(mandatory), "possible": sorted(possible),
                      "never_selected": sorted(all_ids - possible)}
    body = {"schema": SCHEMA, "status": status, "reason": reason,
        "synthetic_only": True, "production_certificate": False,
        "source_bindings": bindings, "source_digest": work.source_digest,
        "proof_sha256": work.transcript.hexdigest(), "exhaustive": exhaustive,
        "optimal_unknown_factor_count": optimum, "membership": membership,
        "factors": factors, "modes": modes, "competing_mode_count": len(modes),
        "hypothesis": modes[0] if status == "robust_hypothesis_only" else None,
        "unknown_semantics": "unverified_association_not_outlier",
        "limitations": ["factor_count_depends_on_proposal_fragmentation",
            "coherent_spurious_unique_subset_possible", "physical_identity_not_established",
            "image_producer_realizability_not_established", "independent_image_verification_required"],
        "hard_maxima": LIMITS, "counters": dict(work.counts), "denial": work.denial}
    # Final proof allocation is charged once, including its counters and seal.
    base, size = work.counts["proof_bytes"], 0
    for _ in range(8):
        body["counters"]["proof_bytes"] = base + size
        try:
            raw = selection._encode(dict(body, record_sha256="0" * 64),
                                    limit=work.budget.proof_bytes - base, name="proof_bytes")
        except selection._RecordExhausted:
            work.bound("proof_bytes", work.budget.proof_bytes + 1)
        if len(raw) == size: break
        size = len(raw)
    else: raise ValueError("proof charge did not converge")
    work.charge("proof_bytes", size)
    body["record_sha256"] = hashlib.sha256(selection._encode(body, limit=size)).hexdigest()
    assert len(selection._encode(body, limit=size)) == size
    return body


def run_counterfactual(*, readings, signatures, universe, frame_ids,
                       session_sha256, lattice_sha256, signatures_sha256,
                       quadrant_constraints=(), expected_source_digest=None,
                       budget=Budget()):
    """Search a complete immutable synthetic source packet with one global ledger.

    Associated endpoint pairs require authentic b/Q-compatible admitted support.
    Unknown candidate factors retain their source IDs; every endpoint must still
    be covered. The result is never an EndpointSelectionCertificate.
    """
    if type(budget) is not Budget: raise ValueError("exact counterfactual Budget required")
    Budget(**asdict(budget))  # revalidate even a forged frozen instance
    work = _Ledger(budget)
    bindings, factor_records, modes = {}, [], []
    best = None
    try:
        if type(readings) is not tuple or type(signatures) is not tuple or type(frame_ids) is not tuple:
            raise ValueError("complete immutable source tuples required")
        if type(universe) is not plane_graph.AdmittedCandidateUniverse:
            raise ValueError("complete admitted candidate universe required")
        work.bound("endpoints", len(readings)); work.bound("rows", len(universe.candidates))
        if len(signatures) > len(readings) * budget.alternatives:
            raise ValueError("signature universe not bounded by retained alternatives")
        for reading in readings:
            if type(reading) is not lattice_planes.PlaneReading or type(reading.alternatives) is not tuple:
                raise ValueError("foreign reading")
            work.bound("alternatives", len(reading.alternatives))
            if not reading.alternatives: raise ValueError("empty retained alternative roster")
            if reading.side not in (-1.0, 1.0): raise ValueError("unverified side")
            work.charge("relation_visits", 1 + len(reading.alternatives))
            for alternative in reading.alternatives:
                values = (alternative.blocks_per_period, alternative.distance_blocks, alternative.support)
                if any(not math.isfinite(float(v)) for v in values) or min(values[:2]) <= 0:
                    raise ValueError("unverified retained alternative")
        for evidence in signatures:
            work.charge("relation_visits")
            if type(evidence) is not signatures_mod.SignatureEvidence:
                raise ValueError("foreign signature")
            if any(len(array.shape) != 2 or any(n > lattice_planes.MAX_SIGNATURE_CELLS for n in array.shape)
                   for array in (evidence.signature.cells, evidence.signature.interior, evidence.signature.coverage)):
                raise ValueError("signature shape exceeds bounded source contract")
        selection._frames(frame_ids)
        reading_bindings = selection.reading_bindings(readings, signatures)
        indexed, _, frames = plane_graph._canonical_reading_index(readings)
        if frames != tuple(sorted(frame_ids)) or len(frames) < 2:
            raise ValueError("complete session/reading frame universe required")
        plane_graph.AdmittedCandidateUniverse(universe.candidates, universe.dispositions, universe.source_rows)
        for digest in (session_sha256, lattice_sha256, signatures_sha256):
            if not selection._hex(digest): raise ValueError("source digest malformed")
        source = hashlib.sha256(work.encode([frame_ids, quadrant_constraints,
            session_sha256, lattice_sha256, signatures_sha256, reading_bindings]))
        candidate_hash = hashlib.sha256(b"[")
        registry = {}
        for i, row in enumerate(universe.candidates):
            work.charge("relation_visits"); work.counts["registry_rows"] += 1
            left, right = plane_graph._validate_indexed_match(row, indexed)
            if row.left == row.right or type(row.quadrant_of_right) is not int or row.quadrant_of_right not in range(4):
                raise ValueError("invalid source relative Q")
            if plane_graph.axis_transform(left.axis, 0)[0] != plane_graph.axis_transform(right.axis, row.quadrant_of_right)[0]:
                raise ValueError("source physical axis mismatch")
            work.charge("relation_visits", len(left.alternatives) + len(right.alternatives))
            left_matches = left.matching_alternatives(row.left_b)
            right_matches = right.matching_alternatives(row.right_b)
            work.charge("relation_visits", len(left_matches) + len(right_matches)
                        + len(left.alternatives) + len(right.alternatives))
            left_ids, right_ids = {id(v) for v in left_matches}, {id(v) for v in right_matches}
            la = sum(1 << a for a, alt in enumerate(left.alternatives) if id(alt) in left_ids)
            ra = sum(1 << a for a, alt in enumerate(right.alternatives) if id(alt) in right_ids)
            if not la or not ra: raise ValueError("source b lacks retained alternative")
            u, v = (row.left, row.left_plane), (row.right, row.right_plane)
            if v < u: u, v, la, ra = v, u, ra, la
            key = (u, v)
            if key not in registry:
                work.bound("factors", len(registry) + 1)
                registry[key] = []
            raw = work.encode(selection.legacy._match_payload(row))
            if i: candidate_hash.update(b",")
            candidate_hash.update(raw)
            source.update(work.encode([i, universe.dispositions[i], hashlib.sha256(raw).hexdigest()]))
            registry[key].append((i, la, ra))
        candidate_hash.update(b"]")
        work.source_digest = source.hexdigest()
        if expected_source_digest is not None and expected_source_digest != work.source_digest:
            raise ValueError("complete source digest mismatch")
        work.endpoint_keys = tuple(sorted(indexed))
        bindings = {"session_sha256": session_sha256, "lattice_sha256": lattice_sha256,
            "signatures_sha256": signatures_sha256, "reading_bindings_sha256": selection._sha(reading_bindings),
            "candidate_matches_sha256": candidate_hash.hexdigest(), "candidate_count": len(universe.candidates),
            "endpoint_count": len(readings), "retained_alternative_count": len(signatures)}
        work.charge("relation_visits", len(registry))
        pairs = tuple(sorted(registry))
        work.counts["registry_factors"] = len(pairs)
        factor_records = [{"factor_id": fid, "endpoints": [list(u), list(v)],
            "source_rows": [item[0] for item in registry[(u, v)]]} for fid, (u, v) in enumerate(pairs)]
        work.encode(factor_records)
        constraints = {}
        if type(quadrant_constraints) is not tuple: raise ValueError("immutable Q constraints required")
        for left, right, relative in quadrant_constraints:
            work.charge("relation_visits")
            if left == right or left not in frames or right not in frames or type(relative) is not int or relative not in range(4):
                raise ValueError("invalid Q constraint")
            if right < left: left, right, relative = right, left, (-relative) % 4
            if (left, right) in constraints and constraints[(left, right)] != relative:
                raise ValueError("contradictory Q constraints")
            constraints[(left, right)] = relative
        reason, planned, plans = plane_graph._quadrant_assignment_plan(frames, constraints)
        if reason:
            work.bound("quadrant_plans", budget.quadrant_plans + 1)
        work.bound("quadrant_plans", planned)

        def support_for(q, selected):
            available = []
            for fid, (u, v) in enumerate(pairs):
                work.charge("relation_visits")  # reserve this factor projection
                supported, group = [], None
                for row_id, la, ra in registry[(u, v)]:
                    work.charge("relation_visits")
                    row = universe.candidates[row_id]
                    if ((q[row.right] - q[row.left]) % 4 != row.quadrant_of_right
                            or not la & (1 << selected[u]) or not ra & (1 << selected[v])): continue
                    work.charge("relation_visits")
                    key, delta, *_ = plane_graph.signed_physical_relation(indexed, row, q, selected, row_id, compact=True)
                    supported.append(row_id)
                    if group is None: group = (key, delta, delta)
                    else: group = (key, min(group[1], delta), max(group[2], delta))
                available.append((tuple(supported), group))
            return available

        def check_full(q, selected, associated, available):
            work.charge("relation_visits", len(associated))
            factors = [(pairs[fid][0], pairs[fid][1], {}) for fid in associated]
            if csp._eligibility_reason(work, factors, indexed, frames, q):
                work.counts["coverage_rejections"] += 1
                return False
            work.charge("relation_visits", len(associated))
            supports = [csp.FactorSupport(pairs[fid], (selected[pairs[fid][0]], selected[pairs[fid][1]]),
                available[fid][0], ()) for fid in associated]
            reason, *_ = csp._physical_feasibility(work, indexed, universe.candidates, q, selected,
                                                   supports, frames, 0)
            if reason:
                work.counts["cycle_rejections" if reason == "physical_frame_cycle" else "diameter_rejections"] += 1
                return False
            return True

        def associations(q, selected, available):
            nonlocal best, modes
            active, extrema, stack = [], {}, [(0, 0, None)]
            while stack:
                index, phase, previous = stack[-1]
                if phase == 0:
                    work.charge("branches")
                    if best is not None and len(active) + len(pairs) - index < len(pairs) - best:
                        stack.pop(); continue
                    if index == len(pairs):
                        work.counts["association_states"] += 1
                        if check_full(q, selected, active, available):
                            unknown = len(pairs) - len(active)
                            mode = {"quadrants": [[f, q[f]] for f in frames],
                                "alternatives": [[f, i, selected[(f, i)]] for f, i in work.endpoint_keys],
                                "associated_factors": list(active),
                                "unknown_factors": [fid for fid in range(len(pairs)) if fid not in active],
                                "support_rows": [[fid, list(available[fid][0])] for fid in active],
                                "coverage": {"supported_endpoints": [list(k) for k in work.endpoint_keys],
                                             "unknown_endpoints": [], "axes_connected": [0, 1, 2]}}
                            work.encode(mode)
                            if best is None or unknown < best: best, modes = unknown, [mode]
                            elif unknown == best: modes.append(mode)
                        stack.pop(); continue
                    rows, group = available[index]
                    stack[-1] = (index, 1, None)
                    if rows:
                        key, low, high = group
                        work.charge("relation_visits")
                        old = extrema.get(key)
                        merged = ((low, high) if old is None else (min(old[0], low), max(old[1], high)))
                        if merged[1] - merged[0] <= plane_graph.SAME_AXIS_TOLERANCE:
                            extrema[key] = merged; active.append(index)
                            stack[-1] = (index, 1, (key, old))
                            stack.append((index + 1, 0, None)); continue
                        work.counts["diameter_rejections"] += 1
                elif phase == 1:
                    if previous is not None:
                        key, old = previous
                        work.rollback(); active.pop()
                        if old is None: del extrema[key]
                        else: extrema[key] = old
                    stack[-1] = (index, 2, None)
                    stack.append((index + 1, 0, None)); continue
                else:
                    stack.pop()

        for q in plans:
            work.charge("quadrant_plans"); work.charge("branches")
            selected, stack = {}, [(0, 0)]
            while stack:
                index, next_alt = stack[-1]
                if index == len(work.endpoint_keys):
                    work.counts["alternative_states"] += 1
                    associations(q, selected, support_for(q, selected))
                    stack.pop(); continue
                endpoint = work.endpoint_keys[index]
                if next_alt == len(indexed[endpoint].alternatives):
                    stack.pop()
                    if endpoint in selected: work.rollback(); del selected[endpoint]
                    continue
                work.charge("branches")
                selected[endpoint] = next_alt
                stack[-1] = (index, next_alt + 1)
                stack.append((index + 1, 0))
        # Replay only retained optima using this same ledger, never a new CSP.
        for mode in modes:
            work.counts["replay_modes"] += 1
            q = dict(mode["quadrants"])
            selected = {(f, i): a for f, i, a in mode["alternatives"]}
            if not check_full(q, selected, mode["associated_factors"], support_for(q, selected)):
                raise ValueError("retained optimum replay differs")
        status = ("robust_hypothesis_only" if len(modes) == 1 else
                  "multiple_hypotheses" if modes else "coverage_or_physical_refused")
        return _finish(work, status, "" if modes else "no_full_coverage_feasible_hypothesis",
                       bindings, factor_records, modes, best, True)
    except _Exhausted:
        # Refusal receipt is small and outside the exhausted proof; never retain
        # a best-so-far mode or an optimum/membership claim after exhaustion.
        return {"schema": SCHEMA, "status": "budget_refused", "reason": "global_budget_exhausted",
                "exhaustive": False, "source_bindings": bindings, "source_digest": work.source_digest,
                "optimal_unknown_factor_count": None, "membership": None, "modes": [],
                "hypothesis": None, "production_certificate": False,
                "counters": dict(work.counts), "denial": work.denial}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        return {"schema": SCHEMA, "status": "input_refused", "reason": str(error)[:256],
                "exhaustive": False, "optimal_unknown_factor_count": None,
                "membership": None, "modes": [], "hypothesis": None,
                "production_certificate": False, "counters": dict(work.counts)}
