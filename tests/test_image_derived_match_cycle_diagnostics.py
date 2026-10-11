"""Focused controls for the output-neutral selected-match cycle audit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_entry import _valid_result  # noqa: E402
from test_plane_graph import make_match, make_reading  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production as production  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "diagnose_image_derived_match_cycles.py"
SPEC = importlib.util.spec_from_file_location("match_cycle_diagnostic", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def _reading(frame, distance, *, plane=0):
    return make_reading(frame, 0, -1.0, [(1.0, distance, 1.0)])


def _clean_cycle():
    readings = (_reading("v0", 1.0), _reading("v1", 2.0),
                _reading("v2", 3.0))
    matches = (make_match("v0", "v1", 0, 0, 0, 0, 0),
               make_match("v1", "v2", 0, 0, 0, 0, 0),
               make_match("v0", "v2", 0, 0, 0, 0, 0))
    constraints = (("v0", "v1", 0), ("v1", "v2", 0))
    return readings, matches, constraints


def test_clean_cycle_is_consistent_and_permutation_invariant():
    readings, matches, constraints = _clean_cycle()
    first = subject.audit_match_cycles(readings, matches, constraints)
    second = subject.audit_match_cycles(
        readings, tuple(reversed(matches)), tuple(reversed(constraints)))
    assert first == second
    assert first["status"] == "consistent"
    assert first["path_count"] == 1
    assert first["cycles"][0]["closure_residual"] == pytest.approx(0.0)
    assert subject.canonical_json(first) == subject.canonical_json(second)


def test_uniquely_inconsistent_injected_edge_is_identified():
    readings = (_reading("v0", 1.0), _reading("v1", 2.0),
                _reading("v2", 2.5), _reading("v3", 4.0),
                _reading("v3", 4.3))
    matches = (
        make_match("v0", "v1", 0, 0, 0, 0, 0),
        make_match("v1", "v3", 0, 0, 0, 0, 0),
        make_match("v0", "v2", 0, 0, 0, 0, 0),
        make_match("v2", "v3", 0, 0, 0, 0, 0),
        make_match("v0", "v3", 0, 0, 0, 0, 1),
    )
    constraints = (("v0", "v1", 0), ("v1", "v3", 0),
                   ("v0", "v2", 0))
    result = subject.audit_match_cycles(readings, matches, constraints)
    assert result["status"] == "inconsistent"
    assert len(result["uniquely_inconsistent_edges"]) == 1
    edge = result["uniquely_inconsistent_edges"][0]
    assert any(edge in cycle["edge_ids"] and not cycle["consistent"]
               for cycle in result["cycles"])


def test_equal_conflicts_abstain_without_edge_adjudication():
    readings, matches, constraints = _clean_cycle()
    readings = readings + (_reading("v2", 3.3),)
    conflicting = list(matches[:2])
    # A second independently measured direct edge has a conflicting closure;
    # one cycle alone cannot identify which of its three edges is wrong.
    conflicting.append(make_match("v0", "v2", 0, 0, 0, 0, 1))
    result = subject.audit_match_cycles(
        readings, tuple(conflicting), constraints)
    assert result["status"] == "abstained"
    assert result["abstention"] == "equally_plausible_conflict"
    assert result["uniquely_inconsistent_edges"] == []


def test_correlated_duplicate_paths_do_not_manufacture_confidence():
    readings, matches, constraints = _clean_cycle()
    duplicated = matches + (matches[0],)
    result = subject.audit_match_cycles(readings, duplicated, constraints)
    assert result["status"] == "abstained"
    assert result["abstention"] == "correlated_duplicate_paths"
    assert result["correlated_edge_groups"]


def test_normal_snapshot_and_digest_are_unchanged(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    monkeypatch.setattr(production.entry_mod, "build_image_derived_entry",
                        lambda *_args, **_kwargs: entry)
    result = production.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "normal.mcscan")
    before = (result.stage, result.status, result.reason,
              result.result_sha256, result.entry.graph.graph_sha256)
    first = subject.diagnose_production_result(result)
    second = subject.diagnose_production_result(result)
    after = (result.stage, result.status, result.reason,
             result.result_sha256, result.entry.graph.graph_sha256)
    assert after == before
    assert first == second
    assert first["normal_mode"]["unchanged"] is True


def test_surface_has_no_threshold_or_metadata_inputs():
    import inspect
    assert tuple(inspect.signature(
        subject.diagnose_image_derived_match_cycles).parameters) == (
            "image_inputs", "project_path", "recursive")
    assert subject.INTEGER_TOLERANCE == 0.15
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "bounds", "synthetic",
                      "manual_metadata"):
        assert forbidden not in source
