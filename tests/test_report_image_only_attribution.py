"""Focused tests for the A2 evaluation report's Route A-only partition option.

These tests drive ``scripts/evaluation/report_image_only_attribution.py`` with
synthetic scorer/report objects only: no reconstruction, fixture, reference,
image, mask, camera, scene, asset, network or product-CLI call is made, and no
file is opened.  The report module is loaded by file path because ``scripts/`` is
not an installed package.  Coverage:

* the ``--include-palette-loss-partition`` flag is default off;
* the default-off report is byte-for-byte unchanged and carries no partition row;
* opt-in appends exactly the reviewed five rows at the Route A ``palette_label``
  checkpoint and leaves every pre-existing row in place;
* the report is strict canonical JSON before and after opt-in;
* the flag routes to Route A scoring, and Route B is rejected with the documented
  typed reason before any production call.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mc_photogrammetry import evaluation_image_only_attribution as attribution
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation_image_only_frontier import (
    canonical_json,
    route_a_frontier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "evaluation" / "report_image_only_attribution.py"

STONE = BlockState("minecraft:stone")
DIRT = BlockState("minecraft:dirt")
GOLD = BlockState("minecraft:gold_block")

#: The exact five metric tokens the reviewed partition helper emits.
PARTITION_METRICS = frozenset({
    "in_palette_geometry_missed", "in_palette_wrong_state",
    "in_palette_exact_state_hit", "candidate_opportunity_in_palette",
    "exact_state_in_candidate_opportunity"})

REFERENCE = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
#: (0,0,0) exact hit, (1,0,0) covered-wrong, (2,0,0) geometry-missed.
FACADE = {(0, 0, 0): STONE, (1, 0, 0): GOLD}


def _runner(name="a2_report_partition"):
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _visible_reference(reference, pixels=4):
    return attribution.visible_reference_from_counts(
        reference, {coordinate: pixels for coordinate in reference},
        source="synthetic")


def _frame(state=STONE):
    return SimpleNamespace(state=state, rgb=(125, 125, 125))


class _Volume:
    """A minimal synthetic facade/decoded volume for the A1 frontier builder."""

    def __init__(self, blocks):
        self.blocks = dict(blocks)

    def to_dict(self):
        return json.dumps(
            {"blocks": [[list(key), str(value.name)]
                        for key, value in sorted(self.blocks.items())]},
            sort_keys=True, separators=(",", ":"))


def _frontier():
    image = SimpleNamespace(shape=(4, 5, 3))
    camera = SimpleNamespace(
        position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0,
        roll_degrees=0.0, fov_degrees=70.0)
    return route_a_frontier(
        admission=image, camera=camera, facade=_Volume(FACADE), palette=(_frame(),),
        decoded_export=_Volume(REFERENCE), camera_solved=True).frontier


def _outcome():
    facade = SimpleNamespace(
        blocks={cell: SimpleNamespace(state=state)
                for cell, state in FACADE.items()})
    decoded = SimpleNamespace(
        blocks={cell: SimpleNamespace(state=state)
                for cell, state in REFERENCE.items()})
    return SimpleNamespace(
        settings=SimpleNamespace(arm="camera-projected"),
        facade=facade, decoded_export=decoded,
        palette=tuple(_frame(state) for state in (STONE, DIRT, GOLD)),
        route_a=SimpleNamespace(frontier=_frontier()),
        camera=SimpleNamespace(), camera_error=None, candidate_seal="")


def _truth(reference):
    return {
        "reference": reference,
        "visible_reference": _visible_reference(reference),
        "reference_sha256": "a" * 64,
        "mask_origin": "synthetic",
        "mask_convention": "white keeps a pixel",
        "frames": [],
        "caveats": [],
    }


def _score(module, **extra):
    return module._score_route_a(
        image=Path("frame.png"), outcome=_outcome(), directory=Path("."),
        truth=_truth(REFERENCE), cli_parity=False, **extra)


def _report_rows(summary):
    return [row for section in summary["attribution"]["sections"]
            for row in section["rows"]]


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------

def test_parser_exposes_the_partition_option_default_off():
    module = _runner("a2_report_parser_partition")
    parser = module.build_parser()
    base = ["--route", "route-a", "--reference", "r", "--scene", "s",
            "--cameras", "c"]
    assert parser.parse_args(base).include_palette_loss_partition is False
    opt_in = parser.parse_args(base + ["--include-palette-loss-partition"])
    assert opt_in.include_palette_loss_partition is True
    action = next(item for item in parser._actions
                  if item.dest == "include_palette_loss_partition")
    assert action.default is False
    assert "--include-palette-loss-partition" in action.option_strings


# ---------------------------------------------------------------------------
# Default-off equality, opt-in rows and strict JSON
# ---------------------------------------------------------------------------

def test_route_a_default_off_report_is_byte_for_byte_unchanged():
    module = _runner("a2_report_default_off")
    default = _score(module)
    explicit_off = _score(module, include_palette_loss_partition=False)
    assert canonical_json(default) == canonical_json(explicit_off)
    assert default["attribution_sha256"] == explicit_off["attribution_sha256"]
    assert {row["metric"] for row in _report_rows(default)} & PARTITION_METRICS \
        == set()
    encoded = canonical_json(default)
    assert canonical_json(json.loads(encoded)) == encoded


def test_route_a_opt_in_report_adds_the_five_partition_rows():
    module = _runner("a2_report_opt_in")
    default = _score(module)
    opt_in = _score(module, include_palette_loss_partition=True)
    default_rows = _report_rows(default)
    opt_rows = _report_rows(opt_in)
    added = [row for row in opt_rows if row["metric"] in PARTITION_METRICS]
    assert len(added) == 5
    assert all(row["checkpoint"] == "palette_label" for row in added)
    assert all(row["space"] == "E" for row in added)
    assert all(row["universe"] == "frozen_reference_cells" for row in added)
    # Purely additive: every pre-existing row survives in the same order.
    assert [row for row in opt_rows if row["metric"] not in PARTITION_METRICS] == \
        default_rows
    assert opt_in["attribution_sha256"] != default["attribution_sha256"]
    by_metric = {row["metric"]: row for row in added}
    assert by_metric["in_palette_geometry_missed"]["metric_value"]["numerator"] == 1
    assert by_metric["in_palette_wrong_state"]["metric_value"]["numerator"] == 1
    assert by_metric["in_palette_exact_state_hit"]["metric_value"]["numerator"] == 1
    assert by_metric["candidate_opportunity_in_palette"]["metric_value"]["value"] == \
        2 / 3
    assert by_metric["exact_state_in_candidate_opportunity"][
        "metric_value"]["value"] == 0.5
    encoded = canonical_json(opt_in)
    assert canonical_json(json.loads(encoded)) == encoded


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------

def _cli_args(tmp_path, route, flag):
    arguments = ["--route", route,
                 "--reference", str(tmp_path / "r.litematic"),
                 "--scene", str(tmp_path / "scene.json"),
                 "--cameras", str(tmp_path / "cameras.json")]
    if route == "route-a":
        arguments += ["--image", "frame.png"]
    else:
        arguments += ["--sources", "a.png", "b.png"]
    if flag:
        arguments.append("--include-palette-loss-partition")
    return arguments


def test_main_routes_the_partition_flag_to_route_a_scoring(monkeypatch, capsys,
                                                           tmp_path):
    module = _runner("a2_report_main_route_a")
    captured = {}
    monkeypatch.setattr(module, "run_route_a", lambda **kwargs: (object(), None))
    monkeypatch.setattr(module, "load_frozen_truth", lambda **kwargs: {})
    monkeypatch.setattr(module, "_score_route_a",
                        lambda **kwargs: captured.update(kwargs) or {"ok": True})
    code = module.main(_cli_args(tmp_path, "route-a", flag=True))
    assert code == 0
    assert captured["include_palette_loss_partition"] is True
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_main_default_off_does_not_request_the_partition(monkeypatch, capsys,
                                                         tmp_path):
    module = _runner("a2_report_main_default_off")
    captured = {}
    monkeypatch.setattr(module, "run_route_a", lambda **kwargs: (object(), None))
    monkeypatch.setattr(module, "load_frozen_truth", lambda **kwargs: {})
    monkeypatch.setattr(module, "_score_route_a",
                        lambda **kwargs: captured.update(kwargs) or {"ok": True})
    code = module.main(_cli_args(tmp_path, "route-a", flag=False))
    assert code == 0
    assert captured["include_palette_loss_partition"] is False
    capsys.readouterr()


def test_main_rejects_the_partition_option_for_route_b(monkeypatch, capsys,
                                                       tmp_path):
    module = _runner("a2_report_main_route_b_reject")
    started = []
    monkeypatch.setattr(module, "run_route_b",
                        lambda **kwargs: started.append(kwargs) or (object(), None))
    monkeypatch.setattr(module, "load_frozen_truth", lambda **kwargs: {})
    code = module.main(_cli_args(tmp_path, "route-b", flag=True))
    assert code == 2
    # Rejected before production: no reconstruction is started.
    assert started == []
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["attribution_available"] is False
    assert refusal["attribution"] is None
    assert refusal["reason"] == "AttributionRefusal"
    assert attribution.PALETTE_LOSS_PARTITION_ROUTE_B_REASON in refusal["detail"]


def test_main_route_b_default_off_is_unchanged_and_still_scores(monkeypatch,
                                                                capsys, tmp_path):
    module = _runner("a2_report_main_route_b_default")
    captured = {}
    monkeypatch.setattr(module, "run_route_b", lambda **kwargs: (object(), None))
    monkeypatch.setattr(module, "load_frozen_truth", lambda **kwargs: {})
    monkeypatch.setattr(module, "_score_route_b",
                        lambda **kwargs: captured.update(kwargs) or {"ok": True})
    code = module.main(_cli_args(tmp_path, "route-b", flag=False))
    assert code == 0
    assert "include_palette_loss_partition" not in captured
    assert json.loads(capsys.readouterr().out) == {"ok": True}
