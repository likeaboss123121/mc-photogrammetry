"""Focused CLI controls for normal image-derived delivery."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry import cli  # noqa: E402


def test_parser_requires_delivery_intent_and_has_no_manual_geometry():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["image-derived-deliver", "a.png"])
    args = parser.parse_args([
        "image-derived-deliver", "a.png", "b.png",
        "--destination", "out.litematic", "--name", "Build",
        "--description", "Derived"])
    assert args.sources == [Path("a.png"), Path("b.png")]
    assert args.destination == Path("out.litematic")
    for forbidden in ("palette", "bounds", "cameras", "scene",
                      "camera_manifest", "policy", "coordinates", "f3"):
        assert not hasattr(args, forbidden)


def test_cli_forwards_intent_and_prints_stable_json(monkeypatch, tmp_path,
                                                    capsys):
    captured = {}

    class Result:
        status = "geometry_unavailable"
        reason = "g_unavailable"
        production_sha256 = "1" * 64
        delivery_sha256 = ""
        result_sha256 = "2" * 64

    def derive(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(cli, "ImageDerivedProductionDeliveryResult", Result)
    monkeypatch.setattr(cli, "derive_and_deliver_image_output", derive)
    destination = tmp_path / "out.litematic"
    project = tmp_path / "audit.mcscan"
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "image-derived-deliver", "a.png",
        "--destination", str(destination), "--name", "Build",
        "--description", "Derived", "--overwrite", "--project", str(project),
        "--max-frames", "3"])
    assert cli.main() == 0
    assert captured["paths"] == [Path("a.png")]
    assert captured["destination"] == destination
    assert captured["name"] == "Build"
    assert captured["description"] == "Derived"
    assert captured["overwrite"] is True
    assert captured["project_path"] == project
    assert captured["max_frames"] == 3
    assert json.loads(capsys.readouterr().out) == {
        "stage": "delivery", "status": "geometry_unavailable",
        "reason": "g_unavailable", "production_sha256": "1" * 64,
        "delivery_sha256": "", "result_sha256": "2" * 64}


def test_image_derived_output_command_remains_available():
    args = cli.build_parser().parse_args(["image-derived-output", "a.png"])
    assert args.command == "image-derived-output"


def test_delivery_cli_discloses_every_partial_camera_exclusion(monkeypatch, capsys):
    from types import SimpleNamespace
    summary = {"original": 3, "admitted": 2, "excluded": 1,
               "exclusions": [{"frame_id": "image-0001", "status": "excluded",
                               "stage": "camera_recovery", "reason": "camera_estimate_unavailable"}]}
    class Result:
        status = "geometry_unavailable"
        reason = "synthetic review"
        production_sha256 = "1" * 64
        delivery_sha256 = ""
        result_sha256 = "2" * 64
        production = SimpleNamespace(entry=SimpleNamespace(session=SimpleNamespace(admission_summary=summary)))
    monkeypatch.setattr(cli, "ImageDerivedProductionDeliveryResult", Result)
    monkeypatch.setattr(cli, "derive_and_deliver_image_output", lambda *args, **kwargs: Result())
    monkeypatch.setattr(sys, "argv", ["mc-photogrammetry", "image-derived-deliver", "synthetic.png",
                                     "--destination", "synthetic.litematic", "--name", "synthetic", "--description", "synthetic"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["camera_admission"] == summary
