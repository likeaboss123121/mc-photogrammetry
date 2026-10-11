"""Focused CLI controls for normal image-derived output wiring."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry import cli  # noqa: E402


def test_text_cli_reports_original_admitted_excluded_and_every_reason(capsys):
    summary = {"original": 3, "admitted": 2, "excluded": 1,
               "exclusions": [{"frame_id": "image-0001", "stage": "camera_recovery", "reason": "camera_estimate_unavailable"}]}
    cli._print_camera_admission(SimpleNamespace(admission_summary=summary))
    lines = capsys.readouterr().out
    assert "Original frames: 3; admitted: 2; excluded: 1" in lines
    assert "Excluded image-0001: camera_recovery: camera_estimate_unavailable" in lines


def test_parser_exposes_only_paths_and_m7_caps():
    args = cli.build_parser().parse_args([
        "image-derived-output", "a.png", "b.png", "--recursive",
        "--max-frames", "3"])
    assert args.command == "image-derived-output"
    assert args.sources == [Path("a.png"), Path("b.png")]
    assert args.recursive is True and args.max_frames == 3
    for forbidden in ("destination", "palette", "bounds", "cameras", "scene",
                      "camera_manifest", "name", "description", "policy"):
        assert not hasattr(args, forbidden)


def test_cli_forwards_exact_caps_and_prints_stable_terminal(
        monkeypatch, tmp_path, capsys):
    captured = {}

    class Result:
        stage = "geometry"
        status = "geometry_unavailable"
        reason = "g_unavailable"
        entry_sha256 = "1" * 64
        lattice_sha256 = "2" * 64
        geometry_sha256 = "3" * 64
        output_sha256 = ""
        result_sha256 = "4" * 64

    def derive(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(cli, "ImageDerivedProductionResult", Result)
    monkeypatch.setattr(cli, "derive_image_output_from_paths", derive)
    project = tmp_path / "audit.mcscan"
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "image-derived-output", "a.png", "b.png",
        "--project", str(project), "--max-frames", "4", "--max-pairs", "5",
        "--max-total-support", "6", "--max-edges", "7",
        "--max-cycle-checks", "8", "--rotation-tolerance-degrees", "1.5",
        "--direction-tolerance", "0.25"])
    assert cli.main() == 0
    assert captured["paths"] == [Path("a.png"), Path("b.png")]
    assert captured["project_path"] == project
    assert [captured[name] for name in (
        "max_frames", "max_pairs", "max_total_support", "max_edges",
        "max_cycle_checks", "rotation_tolerance_degrees",
        "direction_tolerance")] == [4, 5, 6, 7, 8, 1.5, 0.25]
    assert capsys.readouterr().out.splitlines() == [
        "Stage: geometry", "Status: geometry_unavailable",
        "Reason: g_unavailable", "Entry: " + "1" * 64,
        "Lattice: " + "2" * 64, "Geometry: " + "3" * 64,
        "Production: " + "4" * 64]


def test_existing_session_command_parser_is_unchanged():
    args = cli.build_parser().parse_args(["image-derived-session", "images"])
    assert args.command == "image-derived-session"
    assert args.sources == [Path("images")]


def test_output_cli_reports_a_typed_refusal_for_an_existing_project(
        monkeypatch, tmp_path, capsys):
    target = tmp_path / "existing.mcscan"
    target.write_text("prior bytes", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "image-derived-output", str(tmp_path),
        "--project", str(target)])

    assert cli.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert output[0] == "Stage: refusal"
    assert output[1] == "Status: invalid_input"
    assert output[2].startswith("Refusal: ")
    assert target.read_text(encoding="utf-8") == "prior bytes"


def test_deliver_cli_reports_a_typed_refusal_for_an_existing_project(
        monkeypatch, tmp_path, capsys):
    target = tmp_path / "existing.mcscan"
    target.write_text("prior bytes", encoding="utf-8")
    artifact = tmp_path / "out.litematic"
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "image-derived-deliver", str(tmp_path),
        "--destination", str(artifact), "--name", "Build",
        "--description", "Derived", "--project", str(target)])

    assert cli.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "refusal"
    assert payload["status"] == "invalid_input"
    assert target.read_text(encoding="utf-8") == "prior bytes"
    assert not artifact.exists()
