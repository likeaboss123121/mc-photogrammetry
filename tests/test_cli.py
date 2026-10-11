

import sys

import pytest


def test_the_parser_builds_and_every_command_is_reachable():
    """A smoke test for the argument parser.

    `ADJACENCY_STRENGTH` was made the CLI default without importing it, which left
    every invocation raising `NameError` from `build_parser`. Nothing caught it,
    because the tests exercised the command bodies and never the parser itself.
    """
    from mc_photogrammetry.cli import build_parser

    parser = build_parser()
    names: set[str] = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            names.update(choices)
    assert {"compare", "validate", "export"} <= names


def test_compare_accepts_the_attribution_flags():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args(
        ["compare", "a.litematic", "b.litematic", "--attribute",
         "--cameras", "c.json", "--scene", "s.json", "--minecraft-jar", "v.jar"]
    )
    assert args.attribute is True
    assert args.cameras.name == "c.json"


def test_the_export_cli_keeps_the_schema2_litematic_path(monkeypatch, tmp_path):
    """Schema-1/2 projects retain the direct litematic export behavior."""
    from mc_photogrammetry import cli
    from mc_photogrammetry.domain.models import (
        BlockObservation,
        BlockState,
        BlockVolume,
        ScanProject,
    )
    from mc_photogrammetry.project.store import save_project

    volume = BlockVolume()
    volume.set((0, 0, 0), BlockObservation(BlockState("minecraft:stone")))
    project_path = tmp_path / "legacy.mcscan"
    output_path = tmp_path / "legacy.litematic"
    save_project(ScanProject(name="legacy", volume=volume), project_path)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(project_path), str(output_path)])

    assert cli.main() == 0
    assert output_path.exists()


def test_the_reconstruction_command_defaults_to_the_measured_adjacency():
    # 46: the library, the CLI and the benchmark script each defaulted to a
    # different value, and the CLI's was the worst of the three.
    from mc_photogrammetry.cli import build_parser
    from mc_photogrammetry.reconstruction.raycast import ADJACENCY_STRENGTH

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images",
        "--cameras", "c.json", "--scene", "s.json",
    ])
    assert args.adjacency_strength == ADJACENCY_STRENGTH


def test_visible_multiview_exposes_opt_in_partial_shape_fitting():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images",
        "--cameras", "c.json", "--scene", "s.json", "--fit-partial-shapes",
    ])
    assert args.fit_partial_shapes is True


def test_visible_multiview_exposes_partial_render_work_cap():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images", "--partial-fit-max-renders", "17",
    ])

    assert args.partial_fit_max_renders == 17


def test_visible_multiview_has_a_finite_partial_render_budget_by_default():
    from mc_photogrammetry.cli import build_parser
    from mc_photogrammetry.reconstruction.raycast import DEFAULT_PARTIAL_FIT_MAX_RENDERS

    args = build_parser().parse_args(["reconstruct-visible-multiview", "images"])

    assert args.partial_fit_max_renders == DEFAULT_PARTIAL_FIT_MAX_RENDERS


def test_calibrated_multiview_defaults_to_normal_mode_without_sidecars():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args(["reconstruct-multiview", "images"])

    assert args.mode == "normal"
    assert args.cameras is None
    assert args.scene is None


def test_calibrated_multiview_sidecars_require_explicit_developer_mode():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images", "--mode", "developer",
        "--cameras", "c.json", "--scene", "s.json",
    ])

    assert args.mode == "developer"
    assert args.cameras.name == "c.json"


def test_normal_multiview_rejects_before_reading_fixture_inputs(monkeypatch):
    from mc_photogrammetry import cli

    monkeypatch.setattr(
        "sys.argv", ["mc-photogrammetry", "reconstruct-visible-multiview", "images"]
    )
    with pytest.raises(SystemExit, match="requires --mode developer"):
        cli.main()


def test_normal_single_view_rejects_supplied_camera_metadata(monkeypatch):
    from mc_photogrammetry import cli

    monkeypatch.setattr(
        "sys.argv", [
            "mc-photogrammetry", "reconstruct", "image.png",
            "--camera-manifest", "c.json",
        ]
    )
    with pytest.raises(SystemExit, match="developer diagnostics"):
        cli.main()


def _write_png(path, width=60, height=40):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(path)
    return width, height


def _stub_calibration(monkeypatch, width, height):
    """Replace the real solver: these tests own the CLI's delegation, not the fit."""
    import math

    from mc_photogrammetry.reconstruction import calibration
    from mc_photogrammetry.reconstruction.calibration import CalibrationResult

    focal = (height / 2.0) / math.tan(math.radians(70.0) / 2.0)

    def fake_estimate_session(frames, **kwargs):
        return {
            frame_id: CalibrationResult(
                focal_pixels=focal, yaw_degrees=14.0, pitch_degrees=17.0,
                width=width, height=height, inliers=30, segments=40,
                median_residual=0.4, roll_degrees=3.5, principal_point=(41.0, 12.0),
            )
            for frame_id in frames
        }

    monkeypatch.setattr(calibration, "estimate_session", fake_estimate_session)


def _single_block_volume():
    from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume

    volume = BlockVolume()
    volume.set((0, 0, 5), BlockObservation(BlockState("minecraft:stone")))
    return volume


@pytest.mark.parametrize("flag", [
    ["--horizon", "0.4"],
    ["--camera-manifest", "c.json"],
    ["--scene", "s.json"],
])
def test_normal_single_view_rejects_horizon_and_manifests(monkeypatch, flag):
    from mc_photogrammetry import cli

    monkeypatch.setattr(
        "sys.argv", ["mc-photogrammetry", "reconstruct", "image.png", *flag]
    )
    with pytest.raises(SystemExit, match="developer diagnostics"):
        cli.main()


def test_normal_reconstruct_solves_a_camera_and_writes_the_project_and_export(
    monkeypatch, tmp_path
):
    from mc_photogrammetry import cli
    from mc_photogrammetry.pipeline.image_only import LOCAL_FRAME_ORIGIN

    width, height = _write_png(tmp_path / "shot.png")
    _stub_calibration(monkeypatch, width, height)
    output_path = tmp_path / "shot.litematic"
    project_path = tmp_path / "shot.mcscan"
    seen = {}

    def fake_facade(image, mask, **kwargs):
        seen.update(kwargs)
        return _single_block_volume()

    monkeypatch.setattr(cli, "reconstruct_facade", fake_facade)
    monkeypatch.setattr(
        "sys.argv", [
            "mc-photogrammetry", "reconstruct",
            str(tmp_path / "shot.png"), str(output_path), "--project", str(project_path),
        ]
    )

    assert cli.main() == 0

    camera = seen["camera"]
    assert camera is not None, "normal mode silently used no camera"
    assert camera.position == LOCAL_FRAME_ORIGIN
    assert camera.intrinsics.image_size == (width, height)
    assert camera.intrinsics.principal_point_pixels == (41.0, 12.0)
    assert camera.yaw_degrees == pytest.approx(14.0)
    assert camera.pitch_degrees == pytest.approx(17.0)
    assert camera.roll_degrees == pytest.approx(3.5)
    assert camera.fov_degrees == pytest.approx(70.0, rel=1e-3)
    assert seen["horizon"] is None
    assert seen["scene_transform"] is None
    assert project_path.exists()
    assert output_path.exists()


def test_normal_reconstruct_solve_failure_writes_no_project_or_export(
    monkeypatch, tmp_path
):
    from mc_photogrammetry import cli
    from mc_photogrammetry.reconstruction import calibration

    _write_png(tmp_path / "shot.png")
    monkeypatch.setattr(calibration, "estimate_session", lambda frames, **kwargs: {})
    output_path = tmp_path / "shot.litematic"
    project_path = tmp_path / "shot.mcscan"
    reached = {"facade": False}

    def fake_facade(image, mask, **kwargs):
        reached["facade"] = True
        return _single_block_volume()

    monkeypatch.setattr(cli, "reconstruct_facade", fake_facade)
    monkeypatch.setattr(
        "sys.argv", [
            "mc-photogrammetry", "reconstruct",
            str(tmp_path / "shot.png"), str(output_path), "--project", str(project_path),
        ]
    )

    with pytest.raises(SystemExit, match="camera recovery failed"):
        cli.main()

    assert reached["facade"] is False
    assert not output_path.exists()
    assert not project_path.exists()


def test_developer_single_view_keeps_the_horizon_path_without_a_camera(
    monkeypatch, tmp_path
):
    from mc_photogrammetry import cli

    _write_png(tmp_path / "shot.png")
    output_path = tmp_path / "shot.litematic"
    project_path = tmp_path / "shot.mcscan"
    seen = {}

    def fake_facade(image, mask, **kwargs):
        seen.update(kwargs)
        return _single_block_volume()

    def fail_if_the_seam_runs(frames):
        raise AssertionError("developer mode must not run the image-only seam")

    monkeypatch.setattr(cli, "reconstruct_facade", fake_facade)
    monkeypatch.setattr(cli, "solve_image_cameras", fail_if_the_seam_runs)
    monkeypatch.setattr(
        "sys.argv", [
            "mc-photogrammetry", "reconstruct",
            str(tmp_path / "shot.png"), str(output_path), "--project", str(project_path),
            "--mode", "developer", "--horizon", "0.4",
        ]
    )

    assert cli.main() == 0
    assert seen["horizon"] == pytest.approx(0.4)
    assert seen["camera"] is None
    assert output_path.exists()


def test_developer_single_view_still_honours_a_supplied_camera_manifest(
    monkeypatch, tmp_path
):
    import json

    from mc_photogrammetry import cli
    from mc_photogrammetry.reconstruction.camera import CameraModel

    _write_png(tmp_path / "shot.png")
    supplied = CameraModel(
        position=(1.0, 2.0, 3.0), yaw_degrees=10.0, pitch_degrees=5.0, fov_degrees=70.0
    )
    manifest = tmp_path / "cameras.json"
    manifest.write_text(json.dumps({"shot.png": supplied.to_dict()}), encoding="utf-8")
    output_path = tmp_path / "shot.litematic"
    project_path = tmp_path / "shot.mcscan"
    seen = {}

    def fake_facade(image, mask, **kwargs):
        seen.update(kwargs)
        return _single_block_volume()

    def fail_if_the_seam_runs(frames):
        raise AssertionError("developer mode must not run the image-only seam")

    monkeypatch.setattr(cli, "reconstruct_facade", fake_facade)
    monkeypatch.setattr(cli, "solve_image_cameras", fail_if_the_seam_runs)
    monkeypatch.setattr(
        "sys.argv", [
            "mc-photogrammetry", "reconstruct",
            str(tmp_path / "shot.png"), str(output_path), "--project", str(project_path),
            "--mode", "developer", "--camera-manifest", str(manifest),
        ]
    )

    assert cli.main() == 0
    assert seen["camera"] is not None
    assert seen["camera"].position == (1.0, 2.0, 3.0)
    assert seen["camera"].yaw_degrees == pytest.approx(10.0)
    assert seen["horizon"] is None
    assert output_path.exists()
