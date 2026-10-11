"""Evaluation-only Route B replay runner for the A1 image-only frontier.

This runner calls the existing public production entry point
:func:`mc_photogrammetry.pipeline.image_derived_production.derive_image_output_from_paths`
with frozen image paths and a temporary project target, captures the returned full
in-memory ``ImageDerivedProductionResult``, normalizes it through the pure A1
``route_b_frontier`` adapter, and then proves that the frontier's terminal
stage/status/reason agrees with the schema-3 envelope persisted by that *same*
invocation.

It is evaluation-only and read/write-bounded: it writes the project target (and,
when asked, one JSON report) and nothing else; it never reads a camera manifest,
F3/directions, scene transform, bounds, JAR, mask or reference; and it produces no
visible-truth number.  Run it with the repository's interpreter, for example::

    .\\.venv\\Scripts\\python.exe scripts\\evaluation\\run_image_derived_frontier.py ^
        --sources "templates 2\\2026-09-02_13.22.14.png" ... --project .tmp-a1-route-b.mcscan

``--project`` must not already exist: the production entry refuses to replace a
prior terminal.  With no ``--project`` the runner publishes into a fresh private
temporary directory that it removes afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mc_photogrammetry.evaluation_image_only_frontier import (  # noqa: E402
    FRONTIER_PRODUCER,
    FrontierRefusal,
    canonical_json,
    first_observable_failure,
    frontier_unavailable,
    route_b_frontier,
)
from mc_photogrammetry.pipeline.image_derived_production import (  # noqa: E402
    ImageDerivedProductionResult,
    derive_image_output_from_paths,
)
from mc_photogrammetry.pipeline.image_derived_production_record import (  # noqa: E402
    ProductionEnvelopeError,
    production_envelope_from_record,
)
from mc_photogrammetry.project.store import ProjectStoreError, load_project  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402


def _production_defaults() -> dict:
    """The normal CLI's default budget flags for the image-derived commands."""
    from mc_photogrammetry.pipeline import image_derived_entry as entry_mod

    return {
        "recursive": False,
        "max_frames": entry_mod.DEFAULT_MAX_FRAMES,
        "max_pairs": entry_mod.DEFAULT_MAX_PAIRS,
        "max_total_support": entry_mod.DEFAULT_MAX_TOTAL_SUPPORT,
        "max_edges": entry_mod.DEFAULT_MAX_EDGES,
        "max_cycle_checks": entry_mod.DEFAULT_MAX_CYCLE_CHECKS,
        "rotation_tolerance_degrees": entry_mod.DEFAULT_ROTATION_TOLERANCE_DEGREES,
        "direction_tolerance": entry_mod.DEFAULT_DIRECTION_TOLERANCE,
    }


def _persisted_envelope(target: Path):
    """The schema-3 terminal envelope the invocation persisted, replayed."""
    project = load_project(target)
    record = project.image_derived_production
    if record is None:
        raise FrontierRefusal(
            "the published project carries no schema-3 production envelope")
    return production_envelope_from_record(record)


def _bound_refusal_envelope(outcome, target: Path):
    """The persisted envelope only when it binds *this* returned refusal.

    A ``PlacementRefusal`` is a policy outcome, not a stage, so it may only be
    attributed to a persisted schema-3 envelope when that envelope is the
    refusal the invocation actually returned: ``status == "refused"``, the same
    ``refusal_sha256`` and the same ``reason``.  An absent envelope returns
    ``None`` (typed unavailable attribution); an envelope that exists but does
    not bind this refusal fails closed rather than mislabelling a stale, resealed
    or unrelated terminal as this refusal's stage.
    """
    if not target.exists():
        return None
    try:
        envelope = _persisted_envelope(target)
    except FrontierRefusal:
        # The path exists but carries no schema-3 terminal envelope; a
        # malformed persisted envelope still fails loud in _persisted_envelope.
        return None
    reason = str(getattr(outcome, "reason", ""))
    refusal_sha256 = str(getattr(outcome, "refusal_sha256", ""))
    if (envelope.status == "refused"
            and envelope.refusal_sha256 == refusal_sha256
            and envelope.reason == reason):
        return envelope
    raise FrontierRefusal(
        "the persisted schema-3 envelope does not bind the returned placement "
        "refusal")


def _refusal_summary(outcome, target: Path) -> dict:
    """A typed refusal summary that never invents a stage.

    A direct ``PlacementRefusal`` may still have persisted a schema-3 refusal
    envelope for the stage that refused.  When it did *and* that envelope binds
    this exact refusal (refused status, matching ``refusal_sha256`` and reason),
    the validated envelope's ``stage``/``status``/``reason`` is reported;
    otherwise the attribution is a typed unavailable with no stage (an absent
    ``entry`` is never fabricated, and an unbound envelope fails closed).
    """
    reason = str(getattr(outcome, "reason", ""))
    envelope = _bound_refusal_envelope(outcome, target)
    if envelope is None:
        terminal_stage = None
        terminal_status = None
        terminal_reason = None
        terminal_detail = ""
        envelope_sha256 = None
    else:
        terminal_stage = envelope.stage
        terminal_status = envelope.status
        terminal_reason = envelope.reason
        terminal_detail = envelope.detail
        envelope_sha256 = envelope.record_sha256
    return {
        "producer": FRONTIER_PRODUCER,
        "route": "route_b",
        "frontier_available": False,
        "terminal_stage": terminal_stage,
        "terminal_status": terminal_status,
        "terminal_reason": terminal_reason,
        "terminal_detail": terminal_detail,
        "refusal_reason": reason,
        "refusal_sha256": str(getattr(outcome, "refusal_sha256", "")),
        "envelope_sha256": envelope_sha256,
        "frontier": None,
        "first_observable_failure": frontier_unavailable(
            "placement_refusal", stage=terminal_stage or "", detail=reason),
        "terminal_policy_agrees": False,
        "notes": ("an image-derived placement refusal is a policy outcome, "
                  "never an accuracy measurement"),
    }


def run(
    sources,
    *,
    project_path: str | Path | None = None,
    report_path: str | Path | None = None,
    **budgets,
) -> dict:
    """Run Route B once and return its frontier + schema-3 parity summary."""
    source_paths = [Path(item) for item in sources]
    if not source_paths:
        raise FrontierRefusal("route B needs at least one frozen source image")
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FrontierRefusal(f"frozen source image is unavailable: {missing[0]}")
    resolved = dict(_production_defaults())
    resolved.update(budgets)

    if project_path is None:
        with tempfile.TemporaryDirectory(prefix="imgloss-a1-route-b-") as directory:
            target = Path(directory) / "route-b.mcscan"
            summary = _run_once(source_paths, target, resolved)
    else:
        target = Path(project_path)
        if target.exists():
            raise FrontierRefusal(
                f"the route B project target already exists: {target}")
        summary = _run_once(source_paths, target, resolved)

    if report_path is not None:
        Path(report_path).write_text(
            canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def _run_once(source_paths, target: Path, budgets: dict) -> dict:
    outcome = derive_image_output_from_paths(
        source_paths, project_path=target, **budgets)
    if isinstance(outcome, plane_placement.PlacementRefusal):
        return _refusal_summary(outcome, target)
    if not isinstance(outcome, ImageDerivedProductionResult):
        raise FrontierRefusal("the production entry returned an unknown result type")

    frontier = route_b_frontier(
        entry=outcome.entry, lattice=outcome.lattice, geometry=outcome.geometry,
        output=outcome.output, result=outcome)
    envelope = _persisted_envelope(target)
    # The frontier's terminal must be the envelope's terminal, and the envelope
    # must bind the exact session and stage digests the returned result carried.
    if (envelope.stage != outcome.stage or envelope.status != outcome.status
            or envelope.reason != outcome.reason
            or envelope.entry_sha256 != outcome.entry_sha256
            or envelope.lattice_sha256 != outcome.lattice_sha256
            or envelope.geometry_sha256 != outcome.geometry_sha256
            or envelope.output_sha256 != outcome.output_sha256
            or envelope.session_sha256 != outcome.entry.session.session_sha256):
        raise FrontierRefusal(
            "the persisted schema-3 envelope does not replay the returned result")
    if not frontier.terminal_policy_agrees \
            or frontier.terminal_policy.get("stage") != envelope.stage \
            or frontier.terminal_status != envelope.status \
            or frontier.terminal_reason != envelope.reason:
        raise FrontierRefusal(
            "the frontier terminal does not agree with the schema-3 envelope")

    return {
        "producer": FRONTIER_PRODUCER,
        "route": "route_b",
        "frontier_available": True,
        "terminal_stage": envelope.stage,
        "terminal_status": envelope.status,
        "terminal_reason": envelope.reason,
        "terminal_detail": envelope.detail,
        "entry_sha256": envelope.entry_sha256,
        "lattice_sha256": envelope.lattice_sha256,
        "geometry_sha256": envelope.geometry_sha256,
        "output_sha256": envelope.output_sha256,
        "session_sha256": envelope.session_sha256,
        "envelope_sha256": envelope.record_sha256,
        "result_sha256": outcome.result_sha256,
        "project_path": str(target),
        "frontier": frontier.to_record(),
        "first_observable_failure": first_observable_failure(frontier),
        "terminal_policy_agrees": frontier.terminal_policy_agrees,
        "notes": ("typed frontier/status only; no visible-truth number, "
                  "no causal attribution"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A1 evaluation-only Route B image-only frontier runner")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="frozen image paths for the multiview route")
    parser.add_argument("--project", default=None,
                        help="new schema-3 project target (must not exist)")
    parser.add_argument("--report", default=None,
                        help="optional JSON report output path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.sources, project_path=args.project,
                      report_path=args.report)
    except (FrontierRefusal, ProductionEnvelopeError, ProjectStoreError) as error:
        print(canonical_json({
            "producer": FRONTIER_PRODUCER, "route": "route_b",
            "frontier_available": False,
            "reason": type(error).__name__,
            "detail": str(error),
            "frontier": None,
            "notes": "typed refusal; no frontier was built",
        }))
        return 2
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
