from __future__ import annotations

import sys
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.ui import main_window  # noqa: E402


def test_desktop_discloses_partial_input_counts_and_exclusion_before_success():
    summary = {"original": 3, "admitted": 2, "excluded": 1,
               "exclusions": [{"frame_id": "image-0001", "stage": "camera_recovery", "reason": "camera_estimate_unavailable"}]}
    result = main_window._DesktopResult("delivered", Path("synthetic.litematic"), Path("synthetic.mcscan"),
                                        camera_admission=summary)
    message = main_window._format_desktop_result(result)
    assert message.index("3 original, 2 admitted, 1 excluded") < message.index("Created")
    assert "Excluded image-0001: camera_recovery: camera_estimate_unavailable" in message


def _frames(*names: str):
    return SimpleNamespace(image_frames=[
        SimpleNamespace(source_path=str(Path("images") / name))
        for name in names
    ])


def test_multi_photo_route_passes_every_image_and_reports_delivery(
        monkeypatch, tmp_path):
    destination = tmp_path / "chosen.litematic"
    project = tmp_path / "chosen.mcscan"
    production = SimpleNamespace(entry=SimpleNamespace(project_path=str(project)))
    calls = []

    monkeypatch.setattr(main_window, "discover_media",
                        lambda *_a, **_k: _frames("a.png", "b.png", "c.png"))
    monkeypatch.setattr(main_window, "reconstruct_facade",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError("multi-photo route must not use facade")))

    def deliver(paths, **kwargs):
        calls.append((paths, kwargs))
        return SimpleNamespace(status="delivered", reason="", production=production)

    monkeypatch.setattr(
        "mc_photogrammetry.pipeline.image_derived_production_delivery.derive_and_deliver_image_output",
        deliver,
    )
    result = main_window._reconstruct_source(
        tmp_path / "images", tmp_path / "exports", Queue(), Event(),
        destination=destination,
    )

    assert calls == [(
        [Path("images/a.png"), Path("images/b.png"), Path("images/c.png")],
        {
            "destination": destination,
            "name": "chosen",
            "description": "Image-derived visible scene; hidden geometry omitted",
            "project_path": project,
        },
    )]
    assert result.status == "delivered"
    assert result.output == destination
    assert result.project == project


def test_multi_photo_refusal_reports_diagnostic_without_export(
        monkeypatch, tmp_path):
    destination = tmp_path / "chosen.litematic"
    project = tmp_path / "chosen.mcscan"
    production = SimpleNamespace(entry=SimpleNamespace(project_path=str(project)))

    monkeypatch.setattr(main_window, "discover_media",
                        lambda *_a, **_k: _frames("a.png", "b.png"))
    monkeypatch.setattr(main_window, "reconstruct_facade",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError("refused multi-photo route must not fall back")))
    monkeypatch.setattr(
        "mc_photogrammetry.pipeline.image_derived_production_delivery.derive_and_deliver_image_output",
        lambda *_a, **_k: SimpleNamespace(
            status="geometry_unavailable", reason="g_unavailable",
            production=production,
        ),
    )

    result = main_window._reconstruct_source(
        tmp_path / "images", tmp_path / "exports", Queue(), Event(),
        destination=destination,
    )

    assert result.status == "geometry_unavailable"
    assert result.reason == "g_unavailable"
    assert result.output is None
    assert result.project == project
    assert not destination.exists()


def test_multi_photo_route_requires_destination_before_running_pipeline(
        monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "discover_media",
                        lambda *_a, **_k: _frames("a.png", "b.png"))
    monkeypatch.setattr(
        "mc_photogrammetry.pipeline.image_derived_production_delivery.derive_and_deliver_image_output",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("pipeline must not run without a user destination")),
    )

    try:
        main_window._reconstruct_source(
            tmp_path / "images", tmp_path / "exports", Queue(), Event())
    except ValueError as error:
        assert "Choose a .litematic destination" in str(error)
    else:
        raise AssertionError("missing destination should stop reconstruction")


def test_bare_refusal_does_not_claim_preexisting_mcscan_as_current_diagnostic(
        monkeypatch, tmp_path):
    destination = tmp_path / "chosen.litematic"
    project = destination.with_suffix(".mcscan")
    sentinel = b"existing unrelated project"
    project.write_bytes(sentinel)
    monkeypatch.setattr(main_window, "discover_media",
                        lambda *_a, **_k: _frames("a.png", "b.png"))
    monkeypatch.setattr(main_window, "reconstruct_facade",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError("must not fall back to facade")))
    monkeypatch.setattr(
        "mc_photogrammetry.pipeline.image_derived_production_delivery.derive_and_deliver_image_output",
        lambda *_a, **_k: SimpleNamespace(
            reason="placement_unavailable", refusal_sha256="a" * 64),
    )

    result = main_window._reconstruct_source(
        tmp_path / "images", tmp_path / "exports", Queue(), Event(),
        destination=destination,
    )
    message = main_window._format_desktop_result(result)

    assert project.read_bytes() == sentinel
    assert result.output is None
    assert result.project == project
    assert not result.diagnostic_confirmed
    assert "target (not confirmed for this run)" in message
    assert "Schema-3 diagnostic:" not in message


def test_selected_multiview_intent_refuses_if_folder_shrinks_before_worker(
        monkeypatch, tmp_path):
    destination = tmp_path / "chosen.litematic"
    monkeypatch.setattr(main_window, "discover_media",
                        lambda *_a, **_k: _frames("remaining.png"))
    monkeypatch.setattr(main_window, "reconstruct_facade",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError("must not silently switch to facade")))
    monkeypatch.setattr(
        "mc_photogrammetry.pipeline.image_derived_production_delivery.derive_and_deliver_image_output",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("must refuse before invoking single-view route")),
    )

    result = main_window._reconstruct_source(
        tmp_path / "images", tmp_path / "exports", Queue(), Event(),
        destination=destination,
    )

    assert result.status == "refused"
    assert result.reason == "insufficient_images_for_multiview"
    assert result.output is None
    assert result.project == destination.with_suffix(".mcscan")
