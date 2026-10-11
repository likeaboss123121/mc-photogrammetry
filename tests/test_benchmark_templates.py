import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.benchmark_templates as benchmark
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.vision.gui_mask import create_gui_mask
from scripts.benchmark_templates import _load_usable_masks

STONE = BlockState("minecraft:stone")


def test_frozen_mask_spec_supports_compact_polygon_overlays(tmp_path):
    specification = tmp_path / "mask.json"
    specification.write_text(json.dumps({
        "image_size": [10, 8],
        "excluded_rectangles": [[0, 0, 2, 1]],
        "excluded_polygons": [[[4, 2], [7, 2], [4, 5]]],
    }), encoding="utf-8")

    masks = _load_usable_masks({"frame.png": specification}, {
        "frame.png": np.zeros((8, 10, 3), dtype=np.uint8),
    })
    assert masks is not None
    mask = masks["frame.png"]

    assert not mask[0, 0]
    assert not mask[3, 5]
    assert mask[6, 8]


def test_no_mask_paths_are_not_an_authoritative_empty_mapping():
    """No ``--usable-mask`` means ``None``, not ``{}``.

    ``{}`` is authoritative under the reconstruction contract: every frame is
    missing, so nothing is usable.  A run without masks must keep deriving its
    own domain instead of silently reconstructing from fallbacks only.
    """
    frames = {"frame.png": np.zeros((8, 10, 3), dtype=np.uint8)}
    assert _load_usable_masks({}, frames) is None


class _FakeVisible:
    """Minimal ``visible_metrics`` result the report/print path reads."""

    exact_state_recovery = 0.0
    geometry_f1 = 0.0
    geometry_precision = 0.0
    geometry_recall = 0.0
    hidden_reference_blocks = 0
    hidden_inferred_blocks = 0
    hidden_exact_state_blocks = 0
    hidden_inference_recovery = 0.0
    hidden_exact_state_recovery = 0.0

    def to_dict(self):
        return {}


def _capture_report_run(
    monkeypatch, tmp_path, *, usable_mask_paths, fixture_role=None, template_name=None,
    pin_manifest_path=None, palette_support=None,
    stage_trace_path=None, stage_trace_candidate_limit=None,
    stage_sidecar_path=None, sidecar_document=None,
):
    """Drive ``run`` through both mask consumers and return its written report.

    Reconstruction and visibility are replaced by captures; every later stage is
    stubbed so the run reaches the report without a real reconstruction.
    """
    reference = tmp_path / "reference.litematic"
    reference.write_bytes(b"reference")
    camera_manifest = tmp_path / "cameras.json"
    camera_manifest.write_text("{}", encoding="utf-8")
    scene_definition = tmp_path / "scene.json"
    scene_definition.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "report.json"
    frame_info = SimpleNamespace(source_path=str(tmp_path / "frame.png"))
    template_dir = tmp_path / template_name if template_name is not None else tmp_path

    monkeypatch.setattr(benchmark, "discover_media",
                        lambda template_dir: SimpleNamespace(image_frames=[frame_info]))
    monkeypatch.setattr(benchmark, "load_camera_manifest", lambda path: {"frame.png": object()})
    monkeypatch.setattr(benchmark, "load_scene_transform", lambda path: object())
    monkeypatch.setattr(benchmark, "load_image",
                        lambda path: np.zeros((8, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(benchmark, "load_litematic",
                        lambda path: SimpleNamespace(blocks={}))
    monkeypatch.setattr(benchmark, "_source_provenance", lambda: {})
    monkeypatch.setattr(benchmark, "_file_provenance",
                        lambda path: {"path": str(path)})
    if pin_manifest_path is not None:
        monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", pin_manifest_path)

    captured = {}

    def fake_reconstruct(*args, **kwargs):
        captured["reconstruct"] = kwargs.get("usable_masks")
        captured["reconstruct_kwargs"] = dict(kwargs)
        # Emulate production filling the opt-in sidecar buffer in place.
        sidecar = kwargs.get("stage_sidecar")
        if sidecar is not None and sidecar_document is not None:
            sidecar.update(sidecar_document)
        return SimpleNamespace()

    def fake_visible(*args, **kwargs):
        captured["visible"] = kwargs.get("usable_masks")
        return {}

    class _Exporter:
        def export(self, volume, output, *, name):
            output.write_bytes(b"candidate")

    monkeypatch.setattr(benchmark, "reconstruct_scene_surfaces", fake_reconstruct)
    monkeypatch.setattr(benchmark, "visible_cells", fake_visible)
    monkeypatch.setattr(benchmark, "LitematicExporter", _Exporter)
    monkeypatch.setattr(benchmark, "compare_litematics", lambda reference, output: SimpleNamespace(
        candidate=SimpleNamespace(blocks={}, non_air_count=0),
        reference=SimpleNamespace(non_air_count=0),
        exact_coordinate_matches=0, coordinate_recall=0.0, coordinate_precision=0.0,
        coordinate_f1=0.0, reference_exposed=0, surface_matches=0,
        surface_precision=0.0, surface_recall=0.0, surface_f1=0.0,
        exact_state_matches=0, state_accuracy=0.0, whole_reference_state_accuracy=0.0,
        best_translation=(0, 0, 0), direct_coordinate_matches=0,
        direct_coordinate_recall=0.0, direct_state_matches=0, confusion_pairs=(),
        per_state_accuracy={}, per_state_counts={},
    ))
    monkeypatch.setattr(benchmark, "best_visible_alignment", lambda *a, **k: (0, (0, 0, 0)))
    monkeypatch.setattr(benchmark, "visible_metrics", lambda *a, **k: _FakeVisible())
    monkeypatch.setattr(benchmark, "visible_breakdowns", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "observation_kind_counts", lambda volume: {})

    benchmark.run(
        template_dir, tmp_path / "out.litematic", mode="supplied-camera",
        reference=reference, camera_manifest=camera_manifest,
        scene_definition=scene_definition, jar=None, adjacency_strength=0.0,
        report_path=report_path, allow_default_palette=True,
        usable_mask_paths=usable_mask_paths,
        fixture_role=fixture_role,
        palette_support=palette_support,
        stage_trace_path=stage_trace_path,
        stage_sidecar_path=stage_sidecar_path,
        **(
            {} if stage_trace_candidate_limit is None
            else {"stage_trace_candidate_limit": stage_trace_candidate_limit}
        ),
    )
    captured["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return captured


@pytest.mark.parametrize("usable_mask_paths", [None, {}])
def test_run_without_mask_paths_shares_one_derived_gui_mapping(
        monkeypatch, tmp_path, usable_mask_paths):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=usable_mask_paths)

    assert captured["reconstruct"] is captured["visible"]
    assert set(captured["reconstruct"]) == {"frame.png"}
    expected = ~create_gui_mask(8, 10)
    assert np.array_equal(captured["reconstruct"]["frame.png"], expected)
    assert not np.array_equal(expected, np.ones((8, 10), dtype=bool))


def test_derived_run_reports_provisional_gui_policy_and_counts(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)
    report = captured["report"]
    policy = report["usable_masks"]["policy"]

    assert report["usable_masks"]["origin"] == "derived"
    assert "derived production GUI mask" in policy
    assert "provisional" in policy
    assert "frozen" not in policy and "all decoded" not in policy
    assert report["usable_masks"]["usable_pixels"] == {
        "frame.png": int((~create_gui_mask(8, 10)).sum())}
    assert report["usable_masks"]["usable_pixels"]["frame.png"] > 0
    assert report["effective_settings"]["usable_mask_policy"] == policy
    assert report["visibility"]["mask_policy"] == policy


def test_run_with_mask_paths_forwards_one_frozen_mapping(monkeypatch, tmp_path):
    specification = tmp_path / "mask.json"
    specification.write_text(json.dumps({
        "image_size": [10, 8], "excluded_rectangles": [[0, 0, 2, 1]]}), encoding="utf-8")

    captured = _capture_report_run(
        monkeypatch, tmp_path, usable_mask_paths={"frame.png": specification})

    reconstruct = captured["reconstruct"]
    assert reconstruct is captured["visible"]
    assert set(reconstruct) == {"frame.png"}
    assert reconstruct["frame.png"].shape == (8, 10)
    assert not reconstruct["frame.png"][0, 0]
    report = captured["report"]
    assert report["usable_masks"]["origin"] == "frozen"
    assert report["usable_masks"]["policy"] == "frozen supplied masks"
    assert report["usable_masks"]["usable_pixels"] == {
        "frame.png": int(reconstruct["frame.png"].sum())}
    assert report["effective_settings"]["usable_mask_policy"] == "frozen supplied masks"
    assert report["visibility"]["mask_policy"] == "frozen supplied masks"


def _frame_provenance(names):
    return {name: {"sha256": f"{index:064x}"} for index, name in enumerate(names)}


def _empty_pin_manifest(tmp_path):
    """A tracked-manifest-shaped file whose entries are present but unpinned."""
    path = tmp_path / "evaluator-fixture-pins.json"
    path.write_text(json.dumps({
        "manifest_schema": 1,
        "roles": {
            role: {
                "fixture": specification["fixture"],
                "subset": specification.get("subset"),
                "reviewed": False,
                "reference_sha256": None,
                "frames": [],
                "usable_masks": [],
            }
            for role, specification in benchmark.PINNED_FIXTURE_ROLES.items()
        },
    }), encoding="utf-8")
    return path


def _install_pin_manifest(monkeypatch, tmp_path, role, *, reference_sha256, frames,
                          mask_domain_sha256, reviewed=True, fixture=None):
    """Write a synthetic reviewed pin manifest and point the module at it."""
    entry = {
        "fixture": (
            fixture if fixture is not None
            else benchmark.PINNED_FIXTURE_ROLES[role]["fixture"]
        ),
        "subset": benchmark.PINNED_FIXTURE_ROLES[role].get("subset"),
        "reviewed": reviewed,
        "reference_sha256": reference_sha256,
        "frames": [{"name": name, "sha256": digest} for name, digest in sorted(frames.items())],
        "usable_masks": [
            {"frame": name, "domain_sha256": digest}
            for name, digest in sorted(mask_domain_sha256.items())
        ],
    }
    path = tmp_path / "evaluator-fixture-pins.json"
    path.write_text(json.dumps({"manifest_schema": 1, "roles": {role: entry}}), encoding="utf-8")
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", path)
    return entry


def _pinned_identity(monkeypatch, tmp_path, role, *, reference_sha256="a" * 64, frames=None,
                     mask_domain_sha256=None, reviewed=True, fixture=None, template_dir=None):
    """Return an identity whose synthetic reviewed expected pins match its observations."""
    frames = {"frame.png": "1" * 64} if frames is None else dict(frames)
    if mask_domain_sha256 is None:
        mask_domain_sha256 = {name: "2" * 64 for name in frames}
    _install_pin_manifest(
        monkeypatch, tmp_path, role, reference_sha256=reference_sha256, frames=frames,
        mask_domain_sha256=mask_domain_sha256, reviewed=reviewed, fixture=fixture)
    return benchmark._fixture_identity(
        role, Path(template_dir or benchmark.PINNED_FIXTURE_ROLES[role]["fixture"]),
        {name: {"sha256": digest} for name, digest in frames.items()}, reference_sha256,
        mask_domain_sha256=mask_domain_sha256)


def test_tracked_pin_manifest_declares_every_required_role():
    specification = json.loads(
        benchmark.EVALUATOR_PIN_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert specification["manifest_schema"] == benchmark.EVALUATOR_PIN_MANIFEST_SCHEMA
    roles = specification["roles"]
    assert set(roles) == set(benchmark.PINNED_FIXTURE_ROLES)
    for role, entry in roles.items():
        assert entry["fixture"] == benchmark.PINNED_FIXTURE_ROLES[role]["fixture"]
        assert entry["subset"] == benchmark.PINNED_FIXTURE_ROLES[role].get("subset")
        assert set(entry) >= set(benchmark.REQUIRED_PIN_FIELDS)


def test_tracked_manifest_leaves_every_role_unpinned_for_unmatched_observations():
    """The shipped manifest cannot pin a role from its folder name alone."""
    observations = {"frame.png": {"sha256": "0" * 64}}
    for role, specification in benchmark.PINNED_FIXTURE_ROLES.items():
        identity = benchmark._fixture_identity(
            role, Path(specification["fixture"]), observations, "0" * 64,
            mask_domain_sha256={"frame.png": "0" * 64})
        gate = benchmark._acceptance_gate(
            mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
            visible_exact_state_recovery=1.0, visible_geometry_f1=1.0)

        assert identity["pinned"] is False
        assert gate["eligible"] is False and gate["passed"] is False


def test_frozen_small_template_3_role_verifies_its_recorded_subset_pin(monkeypatch, tmp_path):
    specification = json.loads(benchmark.FROZEN_SUBSETS_PATH.read_text(encoding="utf-8"))
    names = sorted(specification["subsets"]["t3-sparse-8"])
    frames = {name: f"{index + 1:064x}" for index, name in enumerate(names)}
    mask_domains = {name: f"{index + 101:064x}" for index, name in enumerate(names)}

    identity = _pinned_identity(
        monkeypatch, tmp_path, "frozen-small-template-3",
        frames=frames, mask_domain_sha256=mask_domains)

    assert identity["pinned"] is True
    assert identity["fixture"] == "templates 3"
    assert identity["fixture_matches"] is True
    assert identity["frames_match"] is True
    assert identity["subset_pin"]["digest_matches"] is True
    assert identity["subset_pin"]["digest"] == specification["digests"]["t3-sparse-8"]
    assert identity["fixture_sha256"]


def test_frozen_small_template_3_role_detects_frame_drift(monkeypatch, tmp_path):
    specification = json.loads(benchmark.FROZEN_SUBSETS_PATH.read_text(encoding="utf-8"))
    names = sorted(specification["subsets"]["t3-sparse-8"])
    frames = {name: f"{index + 1:064x}" for index, name in enumerate(names)}
    mask_domains = {name: f"{index + 101:064x}" for index, name in enumerate(names)}
    _install_pin_manifest(
        monkeypatch, tmp_path, "frozen-small-template-3", reference_sha256="a" * 64,
        frames=frames, mask_domain_sha256=mask_domains)

    drifted = dict(list(frames.items())[:-1])
    identity = benchmark._fixture_identity(
        "frozen-small-template-3", Path("templates 3"),
        {name: {"sha256": digest} for name, digest in drifted.items()}, "a" * 64,
        mask_domain_sha256={name: mask_domains[name] for name in drifted})

    assert identity["pinned"] is False
    assert identity["frames_match"] is False
    assert "frames" in identity["mismatched_pin_fields"]


def test_frozen_subset_pin_rejects_a_tampered_recorded_digest(monkeypatch, tmp_path):
    pin = tmp_path / "frozen-subsets.json"
    pin.write_text(json.dumps({
        "subsets": {"t3-sparse-8": ["a.png", "b.png"]},
        "digests": {"t3-sparse-8": "0000000000000000"},
    }), encoding="utf-8")
    monkeypatch.setattr(benchmark, "FROZEN_SUBSETS_PATH", pin)

    result = benchmark._frozen_subset_pin("t3-sparse-8")

    assert result["digest_matches"] is False
    assert result["pinned_digest"] == "0000000000000000"


def test_unpinned_fixture_role_cannot_pass_even_above_thresholds():
    identity = benchmark._fixture_identity(
        None, Path("templates 2"), _frame_provenance(["frame.png"]), "a" * 64)

    gate = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.95, visible_geometry_f1=0.95)

    assert identity["pinned"] is False
    assert gate["fixture_pinned"] is False
    assert gate["metrics_met"] is True
    assert gate["eligible"] is False
    assert gate["passed"] is False
    assert gate["missing_pin_fields"] == list(benchmark.REQUIRED_PIN_FIELDS)
    assert any("expected pin fields are missing" in reason
               for reason in gate["ineligible_reasons"])


def test_declared_role_without_reviewed_pins_is_unpinned_and_ineligible(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", _empty_pin_manifest(tmp_path))

    identity = benchmark._fixture_identity(
        "template-2", Path("templates 2"), _frame_provenance(["frame.png"]), "a" * 64,
        mask_domain_sha256={"frame.png": "b" * 64})
    gate = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.99, visible_geometry_f1=0.99)

    assert identity["fixture_matches"] is True
    assert identity["pinned"] is False
    assert set(identity["missing_pin_fields"]) == set(benchmark.REQUIRED_PIN_FIELDS) - {"fixture"}
    assert gate["eligible"] is False and gate["passed"] is False
    assert any("expected pin fields are missing" in reason
               for reason in gate["ineligible_reasons"])


def test_folder_name_and_observed_hashes_alone_never_pin_a_role(monkeypatch, tmp_path):
    manifest = tmp_path / "evaluator-fixture-pins.json"
    manifest.write_text(json.dumps({"manifest_schema": 1, "roles": {}}), encoding="utf-8")
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    identity = benchmark._fixture_identity(
        "template-2", Path("templates 2"), _frame_provenance(["frame.png"]), "a" * 64,
        mask_domain_sha256={"frame.png": "b" * 64})

    assert identity["fixture_matches"] is True
    assert identity["reference_sha256"] == "a" * 64
    assert identity["fixture_sha256"]
    assert identity["pinned"] is False


def _reviewed_pin_entry(*, fixture, subset, frames, mask_domain_sha256, reference_sha256="a" * 64):
    return {
        "fixture": fixture,
        "subset": subset,
        "reviewed": True,
        "reference_sha256": reference_sha256,
        "frames": [{"name": name, "sha256": digest} for name, digest in sorted(frames.items())],
        "usable_masks": [
            {"frame": name, "domain_sha256": digest}
            for name, digest in sorted(mask_domain_sha256.items())
        ],
    }


@pytest.mark.parametrize("manifest_schema", [None, 2, "1", True, 1.0])
def test_pin_manifest_with_missing_or_unsupported_schema_is_untrusted(
    monkeypatch, tmp_path, manifest_schema
):
    """A reviewed entry cannot pin anything under a missing/foreign schema.

    JSON ``true`` and ``1.0`` compare equal to the integer schema in Python, so
    the guard must reject them by type rather than by equality alone.
    """
    frames = {"frame.png": "1" * 64}
    mask_domains = {"frame.png": "2" * 64}
    entry = _reviewed_pin_entry(
        fixture="templates 2", subset=None, frames=frames, mask_domain_sha256=mask_domains)
    specification = {"roles": {"template-2": entry}}
    if manifest_schema is not None:
        specification["manifest_schema"] = manifest_schema
    manifest = tmp_path / "evaluator-fixture-pins.json"
    manifest.write_text(json.dumps(specification), encoding="utf-8")
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    # The schema gate short-circuits before roles are considered trusted.
    assert benchmark._expected_pin_manifest() == {}

    identity = benchmark._fixture_identity(
        "template-2", Path("templates 2"),
        {name: {"sha256": digest} for name, digest in frames.items()}, "a" * 64,
        mask_domain_sha256=mask_domains)

    assert identity["pinned"] is False
    assert identity["missing_pin_fields"] == list(benchmark.REQUIRED_PIN_FIELDS)
    assert identity["mismatched_pin_fields"] == []


def test_pin_manifest_schema_must_be_exactly_an_integer(monkeypatch, tmp_path):
    """Python equality would let JSON true/1.0 pass as schema 1; require an int."""
    frames = {"frame.png": "1" * 64}
    mask_domains = {"frame.png": "2" * 64}
    entry = _reviewed_pin_entry(
        fixture="templates 2", subset=None, frames=frames, mask_domain_sha256=mask_domains)
    manifest = tmp_path / "evaluator-fixture-pins.json"
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    assert type(benchmark.EVALUATOR_PIN_MANIFEST_SCHEMA) is int
    for smuggled in (True, 1.0):
        manifest.write_text(
            json.dumps({"manifest_schema": smuggled, "roles": {"template-2": entry}}),
            encoding="utf-8")
        assert benchmark._expected_pin_manifest() == {}

    manifest.write_text(
        json.dumps({"manifest_schema": 1, "roles": {"template-2": entry}}), encoding="utf-8")
    assert set(benchmark._expected_pin_manifest()) == {"template-2"}


def test_pin_manifest_subset_must_agree_with_the_declared_role(monkeypatch, tmp_path):
    frames = {"frame.png": "1" * 64}
    mask_domains = {"frame.png": "2" * 64}
    entry = _reviewed_pin_entry(
        fixture="templates 3", subset="t3-other", frames=frames, mask_domain_sha256=mask_domains)
    manifest = tmp_path / "evaluator-fixture-pins.json"
    manifest.write_text(
        json.dumps({"manifest_schema": 1, "roles": {"frozen-small-template-3": entry}}),
        encoding="utf-8")
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    identity = benchmark._fixture_identity(
        "frozen-small-template-3", Path("templates 3"),
        {name: {"sha256": digest} for name, digest in frames.items()}, "a" * 64,
        mask_domain_sha256=mask_domains)

    assert identity["pinned"] is False
    assert "subset" in identity["mismatched_pin_fields"]
    assert "subset" not in identity["missing_pin_fields"]


def test_pin_manifest_missing_a_declared_subset_is_reported(monkeypatch, tmp_path):
    frames = {"frame.png": "1" * 64}
    mask_domains = {"frame.png": "2" * 64}
    entry = _reviewed_pin_entry(
        fixture="templates 3", subset=None, frames=frames, mask_domain_sha256=mask_domains)
    manifest = tmp_path / "evaluator-fixture-pins.json"
    manifest.write_text(
        json.dumps({"manifest_schema": 1, "roles": {"frozen-small-template-3": entry}}),
        encoding="utf-8")
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    identity = benchmark._fixture_identity(
        "frozen-small-template-3", Path("templates 3"),
        {name: {"sha256": digest} for name, digest in frames.items()}, "a" * 64,
        mask_domain_sha256=mask_domains)

    assert identity["pinned"] is False
    assert "subset" in identity["missing_pin_fields"]
    assert "subset" not in identity["mismatched_pin_fields"]


def test_pin_manifest_matching_schema_and_subset_still_pins(monkeypatch, tmp_path):
    identity = _pinned_identity(monkeypatch, tmp_path, "frozen-small-template-3")

    assert set(benchmark._expected_pin_manifest()) == {"frozen-small-template-3"}
    assert identity["pinned"] is True
    assert identity["missing_pin_fields"] == []
    assert identity["mismatched_pin_fields"] == []


def test_reviewed_matching_pins_make_the_gate_eligible_at_thresholds(monkeypatch, tmp_path):
    identity = _pinned_identity(monkeypatch, tmp_path, "template-2")

    gate = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.80, visible_geometry_f1=0.90)

    assert identity["pinned"] is True
    assert identity["missing_pin_fields"] == [] and identity["mismatched_pin_fields"] == []
    assert gate["eligible"] is True and gate["passed"] is True
    assert gate["fixture_pinned"] is True


def test_acceptance_gate_blocks_derived_masks_and_reference_assisted_runs(monkeypatch, tmp_path):
    identity = _pinned_identity(monkeypatch, tmp_path, "template-2")
    frozen = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.80, visible_geometry_f1=0.90)
    derived = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="derived", fixture_identity=identity,
        visible_exact_state_recovery=0.95, visible_geometry_f1=0.95)
    assisted = benchmark._acceptance_gate(
        mode="reference-assisted", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.95, visible_geometry_f1=0.95)

    assert frozen["passed"] is True and frozen["eligible"] is True
    assert derived["passed"] is False
    assert any("derived" in reason for reason in derived["ineligible_reasons"])
    assert assisted["passed"] is False
    assert any("reference-assisted" in reason for reason in assisted["ineligible_reasons"])


def test_acceptance_gate_blocks_a_folder_mismatch_for_a_pinned_role(monkeypatch, tmp_path):
    identity = _pinned_identity(
        monkeypatch, tmp_path, "template-2", template_dir="templates 9")

    gate = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.95, visible_geometry_f1=0.95)

    assert identity["pinned"] is True
    assert gate["passed"] is False
    assert any("folder does not match" in reason for reason in gate["ineligible_reasons"])


def test_pin_drift_is_reported_per_field(monkeypatch, tmp_path):
    matched = _pinned_identity(monkeypatch, tmp_path, "template-2")
    _install_pin_manifest(
        monkeypatch, tmp_path, "template-2", reference_sha256="b" * 64,
        frames={"frame.png": "9" * 64}, mask_domain_sha256={"frame.png": "8" * 64})

    drifted = benchmark._fixture_identity(
        "template-2", Path("templates 2"), _frame_provenance(["frame.png"]), "a" * 64,
        mask_domain_sha256={"frame.png": "b" * 64})

    assert matched["pinned"] is True
    assert drifted["pinned"] is False
    assert "reviewed" not in drifted["missing_pin_fields"]
    assert set(drifted["mismatched_pin_fields"]) == {
        "reference_sha256", "frames", "usable_masks"}


def test_acceptance_gate_keeps_thresholds_and_reports_a_partial_fixture_set(monkeypatch, tmp_path):
    identity = _pinned_identity(monkeypatch, tmp_path, "template-2")
    below = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.7999, visible_geometry_f1=0.90)

    assert below["metrics_met"] is False
    assert below["passed"] is False
    assert below["eligible"] is True
    assert below["minimum_pixels"] == 1
    assert below["minimum_visible_exact_state_recovery"] == 0.80
    assert below["minimum_visible_geometry_f1"] == 0.90
    assert below["fixture_set_complete"] is False
    assert below["required_fixtures"] == list(benchmark.REQUIRED_PROMOTION_FIXTURES)


def test_visibility_threshold_rows_report_primary_and_larger_domains():
    reference = {(0, 0, 0): STONE, (1, 0, 0): STONE, (2, 0, 0): STONE}
    candidate = {(0, 0, 0): STONE, (1, 0, 0): STONE}
    counts = {(0, 0, 0): 50, (1, 0, 0): 3, (2, 0, 0): 1}

    rows = benchmark._visibility_threshold_rows(
        reference, candidate, counts, translation=(0, 0, 0), rotation_quarter_turns=0)

    assert [row["minimum_pixels"] for row in rows] == [1, 2, 5, 10]
    assert [row["primary"] for row in rows] == [True, False, False, False]
    assert rows[0]["visible_reference_blocks"] == 3
    assert rows[0]["hidden_reference_blocks"] == 0
    assert rows[1]["visible_reference_blocks"] == 2
    assert rows[1]["geometry_recall"] == 1.0
    assert rows[2]["visible_reference_blocks"] == 1
    # Only the 50-pixel cell survives the 5- and 10-pixel domains.
    assert rows[3]["visible_reference_blocks"] == 1
    assert rows[3]["geometry_recall"] == 1.0


def _synthetic_gate(role, *, passed=True, fixture_sha256="f" * 64, frames_sha256="a" * 64,
                    reference_sha256="b" * 64, **overrides):
    gate = {
        "fixture_role": role,
        "passed": passed,
        "eligible": passed,
        "metrics_met": passed,
        "fixture_pinned": True,
        "reference_sha256": reference_sha256,
        "input_frames_sha256": frames_sha256,
        "fixture_sha256": fixture_sha256,
    }
    gate.update(overrides)
    return {"report_schema": benchmark.REPORT_SCHEMA_VERSION, "acceptance_gate": gate}


def test_aggregate_promotion_gate_requires_every_required_fixture():
    partial = benchmark.aggregate_promotion_gate([_synthetic_gate("template-2")])

    assert partial["passed"] is False
    assert partial["missing_fixtures"] == ["template-6", "frozen-small-template-3"]


def test_aggregate_promotion_gate_passes_only_a_complete_stable_set():
    complete = benchmark.aggregate_promotion_gate([
        _synthetic_gate("template-2"),
        _synthetic_gate("template-6"),
        _synthetic_gate("frozen-small-template-3"),
    ])

    assert complete["passed"] is True
    assert complete["missing_fixtures"] == [] and complete["failed_fixtures"] == []
    assert complete["invalid_fixtures"] == {}
    assert complete["required_identity_fields"] == list(benchmark.GATE_REQUIRED_IDENTITY_FIELDS)


def test_aggregate_promotion_gate_reports_failures_and_reference_drift():
    failed = benchmark.aggregate_promotion_gate([
        _synthetic_gate("template-2"),
        _synthetic_gate("template-6", passed=False),
        _synthetic_gate("frozen-small-template-3"),
    ])
    drifted = benchmark.aggregate_promotion_gate([
        _synthetic_gate("template-2", fixture_sha256="a" * 64),
        _synthetic_gate("template-2", fixture_sha256="b" * 64),
        _synthetic_gate("template-6"),
        _synthetic_gate("frozen-small-template-3"),
    ])

    assert failed["passed"] is False
    assert failed["failed_fixtures"] == ["template-6"]
    assert drifted["passed"] is False
    assert "template-2" in drifted["drifted_fixtures"]


def test_aggregate_promotion_gate_rejects_missing_identity_fields():
    incomplete = benchmark.aggregate_promotion_gate([
        {"report_schema": benchmark.REPORT_SCHEMA_VERSION,
         "acceptance_gate": {"fixture_role": "template-2", "passed": True}},
        _synthetic_gate("template-6"),
        _synthetic_gate("frozen-small-template-3"),
    ])

    assert incomplete["passed"] is False
    problems = incomplete["invalid_fixtures"]["template-2"]
    for field in ("eligible", "metrics_met", "fixture_pinned", "reference_sha256",
                  "input_frames_sha256", "fixture_sha256"):
        assert field in problems


def test_aggregate_promotion_gate_rejects_a_superseded_report_schema():
    stale = benchmark.aggregate_promotion_gate([
        {**_synthetic_gate("template-2"), "report_schema": 2},
        _synthetic_gate("template-6"),
        _synthetic_gate("frozen-small-template-3"),
    ])

    assert stale["passed"] is False
    assert stale["invalid_fixtures"]["template-2"] == ["report_schema"]


def test_aggregate_promotion_gate_rejects_unpinned_entries():
    unpinned = benchmark.aggregate_promotion_gate([
        _synthetic_gate("template-2", fixture_pinned=False),
        _synthetic_gate("template-6"),
        _synthetic_gate("frozen-small-template-3"),
    ])

    assert unpinned["passed"] is False
    assert "fixture_pinned" in unpinned["invalid_fixtures"]["template-2"]


def test_report_carries_primary_gate_thresholds_and_mask_domain(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)
    report = captured["report"]

    assert report["report_schema"] == benchmark.REPORT_SCHEMA_VERSION == 3
    assert report["acceptance_gate"]["minimum_pixels"] == 1
    assert report["acceptance_gate"]["fixture_set_complete"] is False
    assert report["fixture_identity"]["pinned"] is False
    assert report["visibility"]["minimum_pixels"] == 1
    assert [row["minimum_pixels"] for row in report["visibility"]["thresholds"]] == [1, 2, 5, 10]
    assert report["usable_masks"]["domain_sha256"]["frame.png"]
    assert report["usable_masks"]["reproducible"] is True


def test_report_qualification_separates_metric_status_from_production(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)
    report = captured["report"]
    qualification = report["qualification"]

    assert report["report_schema"] == 3
    assert qualification["is_image_only_production_qualification"] is False
    assert qualification["image_only_production"] is False
    assert qualification["per_fixture_metric_status"] == {
        "fixture_role": None,
        "eligible": False,
        "metrics_met": False,
        "per_fixture_passed": False,
        "fixture_pinned": False,
    }
    assert report["qualifies_image_only_production"] is False


def test_a_declared_role_without_reviewed_pins_cannot_be_gate_eligible(monkeypatch, tmp_path):
    specification = tmp_path / "mask.json"
    specification.write_text(json.dumps({
        "image_size": [10, 8], "excluded_rectangles": [[0, 0, 2, 1]]}), encoding="utf-8")
    manifest = _empty_pin_manifest(tmp_path)
    monkeypatch.setattr(benchmark, "EVALUATOR_PIN_MANIFEST_PATH", manifest)

    captured = _capture_report_run(
        monkeypatch, tmp_path, usable_mask_paths={"frame.png": specification},
        fixture_role="template-2", template_name="templates 2")
    report = captured["report"]

    assert report["report_schema"] == 3
    assert report["fixture_identity"]["pinned"] is False
    assert report["fixture_identity"]["fixture_matches"] is True
    assert report["acceptance_gate"]["fixture_pinned"] is False
    assert report["acceptance_gate"]["eligible"] is False
    # The stubbed metrics are zero, so eligibility must not by itself pass.
    assert report["acceptance_gate"]["metrics_met"] is False
    assert report["acceptance_gate"]["passed"] is False
    assert any("expected pin fields are missing" in reason
               for reason in report["acceptance_gate"]["ineligible_reasons"])


# ---------------------------------------------------------------------------
# DS-R33: benchmark-only support-pruning ablation control. ``None`` must forward
# the unchanged ``raycast.PALETTE_SUPPORT`` default, an explicit value must reach
# the reconstruction kwargs and be recorded in the effective settings, and no
# explicit override may make a per-run acceptance gate eligible even when the
# fixture identity is fully pinned and the metrics clear their thresholds.


def test_palette_support_override_defaults_to_the_module_value_unchanged(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)

    assert captured["reconstruct_kwargs"]["palette_support"] == raycast.PALETTE_SUPPORT
    settings = captured["report"]["effective_settings"]
    assert settings["palette_support"] == raycast.PALETTE_SUPPORT
    assert settings["palette_support_override"] is None
    assert captured["report"]["acceptance_gate"]["palette_support_override"] is None


def test_explicit_zero_palette_support_reaches_reconstruction_kwargs(monkeypatch, tmp_path):
    captured = _capture_report_run(
        monkeypatch, tmp_path, usable_mask_paths=None, palette_support=0.0)

    assert captured["reconstruct_kwargs"]["palette_support"] == 0.0
    report = captured["report"]
    assert report["report_schema"] == benchmark.REPORT_SCHEMA_VERSION == 3
    assert report["effective_settings"]["palette_support"] == 0.0
    assert report["effective_settings"]["palette_support_override"] == 0.0
    assert report["acceptance_gate"]["palette_support_override"] == 0.0
    assert report["acceptance_gate"]["eligible"] is False
    assert report["acceptance_gate"]["passed"] is False
    assert any("diagnostic-only" in reason
               for reason in report["acceptance_gate"]["ineligible_reasons"])


@pytest.mark.parametrize("override", [0.0, 0.02, 0.5])
def test_any_support_override_blocks_a_pinned_gate(monkeypatch, tmp_path, override):
    identity = _pinned_identity(monkeypatch, tmp_path, "template-2")

    gate = benchmark._acceptance_gate(
        mode="supplied-camera", mask_origin="frozen", fixture_identity=identity,
        visible_exact_state_recovery=0.95, visible_geometry_f1=0.95,
        palette_support_override=override)

    assert identity["pinned"] is True
    assert gate["metrics_met"] is True
    assert gate["fixture_pinned"] is True
    assert gate["palette_support_override"] == override
    assert gate["eligible"] is False
    assert gate["passed"] is False
    assert any("diagnostic-only" in reason for reason in gate["ineligible_reasons"])
    # Thresholds and the report contract stay unchanged under the control.
    assert gate["minimum_pixels"] == 1
    assert gate["minimum_visible_exact_state_recovery"] == 0.80
    assert gate["minimum_visible_geometry_f1"] == 0.90


def test_cli_parses_the_palette_support_override_option():
    parser = benchmark._build_parser()

    default = parser.parse_args(["template", "output"])
    explicit = parser.parse_args(["template", "output", "--palette-support-override", "0.0"])

    assert default.palette_support_override is None
    assert explicit.palette_support_override == 0.0
    with pytest.raises(SystemExit):
        parser.parse_args(["template", "output", "--palette-support-override", "not-a-number"])


# ---------------------------------------------------------------------------
# DS-R34: the support-pruning override must be finite and nonnegative. A NaN,
# infinite or negative value silently disables pruning (`palette_support > 0` is
# false) and can leak non-standard JSON, so both direct ``run()`` callers and the
# CLI must reject it before any fixture input is loaded, while ``None`` and every
# valid nonnegative float (including exactly ``0.0``) keep their R33 behavior.


@pytest.mark.parametrize(
    "override",
    [-0.01, -1.0, -123.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_support_override_fails_before_fixture_input_is_read(
        monkeypatch, tmp_path, override):
    touched = []
    for name in (
        "discover_media", "load_camera_manifest", "load_scene_transform",
        "load_image", "load_litematic", "_file_provenance",
    ):
        monkeypatch.setattr(
            benchmark, name, lambda *a, _name=name, **k: touched.append(_name))

    with pytest.raises(ValueError):
        benchmark.run(
            tmp_path / "missing-template", tmp_path / "out.litematic",
            mode="supplied-camera", reference=tmp_path / "missing-reference.litematic",
            camera_manifest=tmp_path / "missing-cameras.json",
            scene_definition=tmp_path / "missing-scene.json", jar=None,
            adjacency_strength=0.0, palette_support=override)

    # The reference path does not exist; a fixture read would surface a
    # FileNotFoundError or one of the stubbed loaders instead of ValueError.
    assert touched == []


@pytest.mark.parametrize("override", [0.0, 0, 0.02, 0.5, 1.0])
def test_valid_support_override_is_finite_and_nonnegative(override):
    assert benchmark._validated_palette_support(override) == float(override)


def test_support_override_helper_rejects_nonnumeric_and_none():
    for invalid in ("not-a-number", None, object()):
        with pytest.raises(ValueError):
            benchmark._validated_palette_support(invalid)


@pytest.mark.parametrize(
    "raw", ["-0.01", "-1", "nan", "inf", "+inf", "-inf", "not-a-number"])
def test_cli_rejects_invalid_palette_support_overrides(raw):
    parser = benchmark._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["template", "output", "--palette-support-override", raw])


@pytest.mark.parametrize(
    "raw,expected", [("0", 0.0), ("0.0", 0.0), ("0.02", 0.02), ("2", 2.0)])
def test_cli_accepts_finite_nonnegative_palette_support_overrides(raw, expected):
    parser = benchmark._build_parser()

    parsed = parser.parse_args(["template", "output", "--palette-support-override", raw])

    assert parsed.palette_support_override == expected


def test_cli_states_the_finite_nonnegative_palette_support_bound(capsys):
    parser = benchmark._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["template", "output", "--palette-support-override", "-1"])

    assert "finite and >= 0" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# DS-R29/DS-R30: stagewise visible-state loss join. The trace carries no
# reference label; these tests pin the post-hoc join, the corrected
# availability-vs-label checkpoint separation, the allowed alignment transform
# (coordinates *and* state properties), and fail-closed reporting when the
# trace is absent or malformed.


def _trace_state(name, **properties):
    return {"name": name, "properties": [[key, value] for key, value in properties.items()]}


def _trace_entry(coordinate, *, top=None, raw_winner=None, label=None, final=None):
    """One synthetic per-coordinate trace record.

    ``top`` is the cell's retained ranked candidate list, best first; it
    defaults to the raw winner alone. Keeping it independent of ``raw_winner``
    lets a test model a reference state that survives pruning and sits in the
    retained top-K without ever being the ranking's head.
    """
    entry = {"coordinate": list(coordinate)}
    if raw_winner is not None:
        entry["raw_winner"] = raw_winner
    if top is None and raw_winner is not None:
        top = [(raw_winner, 0.0)]
    if top is not None:
        entry["top_candidates"] = [
            {"state": state, "cost": float(cost)} for state, cost in top
        ]
    if label is not None:
        entry["post_smoothing_label"] = label
    if final is not None:
        entry["final_state"] = final
    return entry


def _synthetic_stage_trace():
    """Eight visible cells, each isolating one checkpoint's behaviour.

    The global pruning membership is keyed by state (one round drops
    ``dark_oak_log``), so the join can ask whether the *reference* state
    survived rather than whether some current winner did.
    """
    stone = _trace_state("minecraft:stone")
    dirt = _trace_state("minecraft:dirt")
    oak_log = _trace_state("minecraft:oak_log", axis="y")
    dark_oak_log = _trace_state("minecraft:dark_oak_log", axis="x")
    return {
        "schema": 1,
        "candidate_limit": 8,
        "palette": {
            "size": 4,
            "states": [stone, dirt, oak_log, dark_oak_log],
        },
        "support_rounds": [
            {
                "round": 0,
                "candidates_before": 4,
                "candidates_after": 3,
                "states_before": [stone, dirt, oak_log, dark_oak_log],
                "states_kept": [stone, dirt, oak_log],
                "states_dropped": [dark_oak_log],
                "deciding_excluded_proposals": False,
                "support_prior_applied": False,
            },
        ],
        "final_palette": {"size": 3, "states": [stone, dirt, oak_log]},
        "coordinates": [
            # every availability and label checkpoint reaches the right state
            _trace_entry(
                (0, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=stone, final=stone),
            # the canonical palette never held the reference state (cobblestone)
            _trace_entry(
                (1, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=stone, final=stone),
            # the reference state was in the palette but global pruning dropped it
            _trace_entry(
                (2, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=stone, final=stone),
            # survives support and sits in the retained top-K, but ranks second
            _trace_entry(
                (3, 0, 0), top=[(stone, 0.0), (dirt, 1.0)], raw_winner=stone,
                label=stone, final=stone),
            # survives support, but the cell's retained top-K excludes it
            _trace_entry(
                (4, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=stone, final=stone),
            # raw ranking was right; adjacency smoothing flipped it to the wrong state
            _trace_entry(
                (5, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=dirt, final=dirt),
            # right through smoothing; never placed in the volume
            _trace_entry(
                (6, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                label=stone, final=None),
            # raw ranking lost it, but smoothing *gained* it back: a later
            # checkpoint matching more cells than an earlier one
            _trace_entry(
                (7, 0, 0), top=[(stone, 0.0), (dirt, 1.0)], raw_winner=stone,
                label=dirt, final=dirt),
        ],
    }


def _synthetic_reference():
    return {
        (0, 0, 0): STONE,
        (1, 0, 0): BlockState("minecraft:cobblestone"),
        (2, 0, 0): BlockState("minecraft:dark_oak_log", (("axis", "x"),)),
        (3, 0, 0): BlockState("minecraft:dirt"),
        (4, 0, 0): BlockState("minecraft:oak_log", (("axis", "y"),)),
        (5, 0, 0): STONE,
        (6, 0, 0): STONE,
        (7, 0, 0): BlockState("minecraft:dirt"),
    }


def test_stagewise_loss_separates_availability_checkpoints_from_labels():
    result = benchmark._stagewise_visible_loss(
        _synthetic_reference(), {}, _synthetic_reference(),
        stage_trace=_synthetic_stage_trace(),
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    assert result["available"] is True and result["unavailable_reason"] is None
    assert result["visible_reference_blocks"] == 8
    matches = {stage["stage"]: stage["exact_state_matches"] for stage in result["stages"]}
    assert matches == {
        "available_palette": 7,   # (1,0,0) cobblestone is not in the canonical palette
        "support": 6,             # (2,0,0) dark_oak_log is dropped by global pruning
        "candidate": 5,           # (4,0,0) oak_log is not in the cell's retained top-K
        "raw_winner": 3,          # (3,0,0) and (7,0,0) rank stone over their dirt
        "post_smoothing": 3,      # (5,0,0) smoothing flipped stone to dirt
        "placed": 2,              # (6,0,0) was never placed; (5,0,0) was placed wrong
    }
    kinds = {stage["stage"]: stage["kind"] for stage in result["stages"]}
    assert kinds == {
        "available_palette": benchmark.STAGE_TRACE_AVAILABILITY,
        "support": benchmark.STAGE_TRACE_AVAILABILITY,
        "candidate": benchmark.STAGE_TRACE_AVAILABILITY,
        "raw_winner": benchmark.STAGE_TRACE_LABEL,
        "post_smoothing": benchmark.STAGE_TRACE_LABEL,
        "placed": benchmark.STAGE_TRACE_LABEL,
    }
    assert [stage["visible_reference_blocks"] for stage in result["stages"]] == [8] * 6

    # Availability and actual labels are reported separately.
    summary = result["summary"]
    assert summary["availability_matches"] == {
        "available_palette": 7, "support": 6, "candidate": 5}
    assert summary["label_matches"] == {
        "raw_winner": 3, "post_smoothing": 3, "placed": 2}

    transitions = {transition["from_stage"]: transition for transition in result["stage_transitions"]}
    # Support pruning drops (2,0,0); its entry still exists and now carries the
    # wrong state, so the delta is a state loss rather than missing geometry.
    assert transitions["available_palette"]["state_lost"] == 1
    assert transitions["available_palette"]["geometry_lost"] == 0
    # (4,0,0) survives pruning but the cell's retained top-K no longer offers it.
    assert transitions["support"]["state_lost"] == 1
    # Ranking, not availability, costs (3,0,0) and (7,0,0) the reference state -
    # exactly the loss the old winner-equality candidate stage hid.
    assert transitions["candidate"]["state_lost"] == 2
    assert transitions["candidate"]["geometry_lost"] == 0
    # (5,0,0) is flipped by smoothing while (7,0,0) is *gained* by it, so this
    # delta is not a monotone ceiling and nets to zero.
    assert transitions["raw_winner"]["state_lost"] == 1
    assert transitions["raw_winner"]["gained"] == 1
    assert transitions["raw_winner"]["net_change"] == 0
    # (6,0,0) survives every label checkpoint and then never lands in the volume.
    assert transitions["post_smoothing"]["geometry_lost"] == 1
    assert transitions["post_smoothing"]["state_lost"] == 0

    assert summary["largest_stage_loss"] == "candidate"
    assert summary["largest_stage_loss_cells"] == 2
    assert summary["largest_stage_loss_geometry_cells"] == 0
    assert summary["largest_stage_loss_state_cells"] == 2
    assert summary["geometry_missing_cells"] == 1
    assert summary["matched_wrong_state_cells"] == 5
    assert summary["final_exact_state_matches"] == 2
    assert summary["trace_malformed_records"] == 0
    assert summary["trace_records_outside_reference"] == 0
    assert summary["final_state_confusion"] == [
        {"reference_state": "minecraft:cobblestone", "candidate_state": "minecraft:stone", "count": 1},
        {"reference_state": "minecraft:dark_oak_log[axis=x]", "candidate_state": "minecraft:stone", "count": 1},
        {"reference_state": "minecraft:dirt", "candidate_state": "minecraft:stone", "count": 1},
        {"reference_state": "minecraft:oak_log[axis=y]", "candidate_state": "minecraft:stone", "count": 1},
        {"reference_state": "minecraft:stone", "candidate_state": "minecraft:dirt", "count": 1},
    ]

    # Geometry and state losses together account for the full visible deficit.
    total_loss = 8 - summary["final_exact_state_matches"]
    assert summary["geometry_missing_cells"] + summary["matched_wrong_state_cells"] == total_loss


@pytest.mark.parametrize(
    "name, reference, palette_states, support_kept, entry, expected, transition",
    [
        (
            "support_keeps_reference_but_ranking_loses_it",
            BlockState("minecraft:dirt"),
            [_trace_state("minecraft:stone"), _trace_state("minecraft:dirt")],
            [_trace_state("minecraft:stone"), _trace_state("minecraft:dirt")],
            {"top": [(_trace_state("minecraft:stone"), 0.0),
                     (_trace_state("minecraft:dirt"), 1.0)],
             "raw_winner": _trace_state("minecraft:stone"),
             "label": _trace_state("minecraft:stone"),
             "final": _trace_state("minecraft:stone")},
            {"available_palette": 1, "support": 1, "candidate": 1,
             "raw_winner": 0, "post_smoothing": 0, "placed": 0},
            ("candidate", "raw_winner", {"state_lost": 1, "geometry_lost": 0}),
        ),
        (
            "support_drops_reference_state",
            BlockState("minecraft:dark_oak_log", (("axis", "x"),)),
            [_trace_state("minecraft:stone"), _trace_state("minecraft:dark_oak_log", axis="x")],
            [_trace_state("minecraft:stone")],
            {"top": [(_trace_state("minecraft:stone"), 0.0)],
             "raw_winner": _trace_state("minecraft:stone"),
             "label": _trace_state("minecraft:stone"),
             "final": _trace_state("minecraft:stone")},
            {"available_palette": 1, "support": 0, "candidate": 0,
             "raw_winner": 0, "post_smoothing": 0, "placed": 0},
            ("available_palette", "support", {"state_lost": 1, "geometry_lost": 0}),
        ),
        (
            "retained_top_k_excludes_reference",
            BlockState("minecraft:dirt"),
            [_trace_state("minecraft:stone"), _trace_state("minecraft:dirt")],
            [_trace_state("minecraft:stone"), _trace_state("minecraft:dirt")],
            {"top": [(_trace_state("minecraft:stone"), 0.0)],
             "raw_winner": _trace_state("minecraft:stone"),
             "label": _trace_state("minecraft:stone"),
             "final": _trace_state("minecraft:stone")},
            {"available_palette": 1, "support": 1, "candidate": 0,
             "raw_winner": 0, "post_smoothing": 0, "placed": 0},
            ("support", "candidate", {"state_lost": 1, "geometry_lost": 0}),
        ),
        (
            "raw_winner_right_but_smoothing_flips_it",
            STONE,
            [_trace_state("minecraft:stone")],
            [_trace_state("minecraft:stone")],
            {"top": [(_trace_state("minecraft:stone"), 0.0)],
             "raw_winner": _trace_state("minecraft:stone"),
             "label": _trace_state("minecraft:dirt"),
             "final": _trace_state("minecraft:dirt")},
            {"available_palette": 1, "support": 1, "candidate": 1,
             "raw_winner": 1, "post_smoothing": 0, "placed": 0},
            ("raw_winner", "post_smoothing", {"state_lost": 1, "geometry_lost": 0}),
        ),
        (
            "final_placement_omits_reference",
            STONE,
            [_trace_state("minecraft:stone")],
            [_trace_state("minecraft:stone")],
            {"top": [(_trace_state("minecraft:stone"), 0.0)],
             "raw_winner": _trace_state("minecraft:stone"),
             "label": _trace_state("minecraft:stone"),
             "final": None},
            {"available_palette": 1, "support": 1, "candidate": 1,
             "raw_winner": 1, "post_smoothing": 1, "placed": 0},
            ("post_smoothing", "placed", {"state_lost": 0, "geometry_lost": 1}),
        ),
    ],
    ids=(
        "support_keeps_reference_ranking_loses_it",
        "support_drops_reference",
        "top_k_excludes_reference",
        "smoothing_flips_raw_winner",
        "placement_omits_reference",
    ),
)
def test_stagewise_loss_synthetic_checkpoint_cases(
    name, reference, palette_states, support_kept, entry, expected, transition
):
    """One cell per R30 case: availability and labels move independently."""
    trace = {
        "schema": 1,
        "palette": {"size": len(palette_states), "states": palette_states},
        "support_rounds": [
            {
                "round": 0,
                "states_before": palette_states,
                "states_kept": support_kept,
                "states_dropped": [
                    state for state in palette_states if state not in support_kept
                ],
            },
        ],
        "final_palette": {"states": support_kept},
        "coordinates": [
            _trace_entry(
                (0, 0, 0),
                top=entry["top"],
                raw_winner=entry["raw_winner"],
                label=entry["label"],
                final=entry["final"],
            ),
        ],
    }
    result = benchmark._stagewise_visible_loss(
        {(0, 0, 0): reference}, {}, {(0, 0, 0): 1},
        stage_trace=trace, translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    assert result["available"] is True, name
    matches = {stage["stage"]: stage["exact_state_matches"] for stage in result["stages"]}
    assert matches == expected, name
    transitions = {item["from_stage"]: item for item in result["stage_transitions"]}
    from_stage, to_stage, losses = transition
    assert transitions[from_stage]["to_stage"] == to_stage, name
    assert transitions[from_stage]["geometry_lost"] == losses["geometry_lost"], name
    assert transitions[from_stage]["state_lost"] == losses["state_lost"], name


def test_stagewise_loss_uses_the_report_alignment_transform_on_coordinates_and_states():
    """The join rotates directional state properties exactly as the report's transform does.

    ``oak_stairs[facing=west]`` under a single quarter turn must be compared as
    ``oak_stairs[facing=north]``, and the local (1,0,2) must be looked up at
    (-2,0,1); a join that transformed only coordinates, or that compared the
    unrotated palette/support membership, would fail this. (A ``log[axis]``
    state is deliberately *not* used: a Y-axis rotation cannot move a log's
    axis, so the evaluator's transform leaves it alone.)
    """
    reference = {(-2, 0, 1): BlockState("minecraft:oak_stairs", (("facing", "north"),))}
    visible = {(-2, 0, 1): 10}
    candidate_frame_stairs = _trace_state("minecraft:oak_stairs", facing="west")
    trace = {
        "schema": 1,
        "palette": {"size": 1, "states": [candidate_frame_stairs]},
        "support_rounds": [
            {
                "round": 0,
                "states_before": [candidate_frame_stairs],
                "states_kept": [candidate_frame_stairs],
                "states_dropped": [],
            },
        ],
        "final_palette": {"states": [candidate_frame_stairs]},
        "coordinates": [
            _trace_entry(
                (1, 0, 2),
                raw_winner=candidate_frame_stairs,
                label=candidate_frame_stairs,
                final=candidate_frame_stairs),
        ],
    }

    rotated = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=trace,
        translation=(0, 0, 0), rotation_quarter_turns=1, requested=True)
    unrotated = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=trace,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)

    assert [stage["exact_state_matches"] for stage in rotated["stages"]] == [1, 1, 1, 1, 1, 1]
    # Without the transform the local cell is not visible and the candidate-frame
    # palette, support membership and label all stay unrotated, so every
    # checkpoint reports nothing.
    assert [stage["exact_state_matches"] for stage in unrotated["stages"]] == [0, 0, 0, 0, 0, 0]
    assert unrotated["summary"]["geometry_missing_cells"] == 1

    # And an integer translation composes after the rotation: the local cell at
    # (4,0,2) rotates to (-2,0,4) and is then mapped by (0,0,-3) onto (-2,0,1).
    shifted_entry = _trace_entry(
        (4, 0, 2),
        raw_winner=candidate_frame_stairs,
        label=candidate_frame_stairs,
        final=candidate_frame_stairs)
    translated = benchmark._stagewise_visible_loss(
        reference, {}, visible,
        stage_trace={**trace, "coordinates": [shifted_entry]},
        translation=(0, 0, -3), rotation_quarter_turns=1, requested=True)
    assert translated["summary"]["final_exact_state_matches"] == 1
    assert translated["alignment"]["translation"] == [0, 0, -3]
    # The same trace without the translation does not land on the visible cell.
    unshifted = benchmark._stagewise_visible_loss(
        reference, {}, visible,
        stage_trace={**trace, "coordinates": [shifted_entry]},
        translation=(0, 0, 0), rotation_quarter_turns=1, requested=True)
    assert unshifted["summary"]["final_exact_state_matches"] == 0


def _trace_shape(trace, *, coordinates=None, **overrides):
    """Copy a synthetic trace with overrides, so fail-closed cases stay readable."""
    shaped = {**trace, **overrides}
    if coordinates is not None:
        shaped["coordinates"] = coordinates
    return shaped


def test_stagewise_loss_fails_closed_when_the_trace_is_absent_or_malformed():
    reference, visible = _synthetic_reference(), _synthetic_reference()
    cases = {
        "not_requested": (None, False),
        "empty_trace": ({}, True),
        "no_coordinates": ({"schema": 1, "palette": {"states": []}}, True),
        "records_not_a_list": ({"coordinates": "nope"}, True),
    }

    for name, (trace, requested) in cases.items():
        result = benchmark._stagewise_visible_loss(
            reference, {}, visible, stage_trace=trace,
            translation=(0, 0, 0), rotation_quarter_turns=0, requested=requested)
        assert result["available"] is False, name
        assert result["unavailable_reason"], name
        assert result["stages"] == [] and result["stage_transitions"] == [], name
        assert result["summary"]["geometry_missing_cells"] == 0, name
        assert result["summary"]["matched_wrong_state_cells"] == 0, name
        assert result["summary"]["largest_stage_loss"] is None, name
        assert result["stage_order"] == [
            stage for stage, _kind, _description in benchmark.STAGE_TRACE_STAGES]

    # Malformed *coordinate records* are still tolerated and counted: the trace
    # structure is intact, only two per-coordinate records cannot be read.
    malformed = benchmark._stagewise_visible_loss(
        reference, {}, visible,
        stage_trace=_trace_shape(
            _synthetic_stage_trace(),
            coordinates=[{"coordinate": [1, 2]}, "nope"]),
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert malformed["available"] is True
    assert malformed["summary"]["trace_malformed_records"] == 2
    assert malformed["summary"]["trace_records"] == 0

    # A run with no visible reference cells cannot produce a denominator.
    empty = benchmark._stagewise_visible_loss(
        {}, {}, {}, stage_trace=_synthetic_stage_trace(),
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert empty["available"] is False
    assert "no visible reference cells" in empty["unavailable_reason"]


def test_stagewise_loss_fails_closed_when_the_trace_structure_cannot_evidence_support():
    """DS-R31: a partial trace may not claim support availability it cannot evidence.

    The previous join read a missing ``support_rounds``/``final_palette`` as
    "nothing was pruned", so a trace that simply omitted the membership lists
    counted every visible cell as support-available. The section must instead be
    unavailable whenever the canonical palette, the support-round list or the
    final palette is missing or malformed, while a legitimately empty round list
    still computes against a valid final palette.
    """
    trace = _synthetic_stage_trace()
    reference, visible = _synthetic_reference(), _synthetic_reference()
    stone_record, dirt_record = _trace_state("minecraft:stone"), _trace_state("minecraft:dirt")
    round_record = {
        "round": 0,
        "states_before": [stone_record, dirt_record],
        "states_kept": [stone_record, dirt_record],
        "states_dropped": [],
    }
    cell = _trace_entry(
        (3, 0, 0), raw_winner=dirt_record, label=dirt_record, final=dirt_record)

    unavailability_cases = {
        "missing_support_rounds": _trace_shape(
            trace, coordinates=[cell], support_rounds=None),
        "support_rounds_not_a_list": _trace_shape(
            trace, coordinates=[cell], support_rounds="nope"),
        "missing_final_palette": _trace_shape(
            trace, coordinates=[cell], final_palette=None),
        "final_palette_without_states": _trace_shape(
            trace, coordinates=[cell], final_palette={"size": 0}),
        "malformed_kept_states_entry": _trace_shape(
            trace, coordinates=[cell],
            support_rounds=[{**round_record, "states_kept": [stone_record, {"properties": []}]}]),
        "malformed_final_palette_states": _trace_shape(
            trace, coordinates=[cell],
            final_palette={"states": [stone_record, "not-a-state-record"]}),
    }
    for name, malformed_trace in unavailability_cases.items():
        result = benchmark._stagewise_visible_loss(
            reference, {}, visible, stage_trace=malformed_trace,
            translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
        assert result["available"] is False, name
        assert result["unavailable_reason"], name
        assert result["stages"] == [] and result["stage_transitions"] == [], name
        assert result["summary"]["availability_matches"] == {}, name
        assert result["summary"]["largest_stage_loss"] is None, name

    # A genuinely empty prune-round list is not malformed: zero rounds compute
    # normally, and support availability then follows the declared final palette.
    empty_rounds = _trace_shape(trace, coordinates=[cell], support_rounds=[])
    result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=empty_rounds,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert result["available"] is True and result["unavailable_reason"] is None
    matches = {stage["stage"]: stage["exact_state_matches"] for stage in result["stages"]}
    # The support checkpoint is global: every visible cell whose reference state
    # is still in the declared final palette is support-available even when the
    # trace formed no per-coordinate record for it, so six cells survive (only
    # the dark-oak and cobblestone references left that palette). Labels, by
    # contrast, need the one record that exists here.
    assert matches == {"available_palette": 7, "support": 6,
                       "candidate": 1, "raw_winner": 1,
                       "post_smoothing": 1, "placed": 1}
    assert result["summary"]["availability_matches"] == {
        "available_palette": 7, "support": 6, "candidate": 1}

    # Zero rounds plus a final palette that drops the reference state is still a
    # real support loss, not a missing-membership pass.
    narrowed = _trace_shape(
        trace, coordinates=[cell], support_rounds=[],
        final_palette={"states": [stone_record]})
    narrowed_result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=narrowed,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert narrowed_result["available"] is True
    narrowed_matches = {
        stage["stage"]: stage["exact_state_matches"] for stage in narrowed_result["stages"]}
    assert narrowed_matches["support"] == 4
    assert narrowed_matches["available_palette"] == 7

    # A valid empty round list with a final palette that keeps every state leaves
    # the reference state reachable, so the checkpoint is not falsely a loss.
    kept_all = _trace_shape(
        trace, coordinates=[cell], support_rounds=[],
        final_palette={
            "states": [stone_record, dirt_record, _trace_state("minecraft:oak_log", axis="y")]})
    kept_result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=kept_all,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert kept_result["available"] is True
    kept_matches = {stage["stage"]: stage["exact_state_matches"] for stage in kept_result["stages"]}
    assert kept_matches["support"] == 6
    assert kept_matches["available_palette"] == 7


def test_stagewise_loss_rejects_records_the_decoder_cannot_rehydrate():
    """DS-R32: structural validation must agree with the decoder.

    R31 accepted any non-empty string ``name``, so ``{"name": "stone"}`` passed
    the membership gate even though :class:`BlockState` needs a namespace, and a
    malformed ``properties`` value was silently dropped or ``str()``-coerced by
    ``_state_from_record``. Either shape in the canonical palette, a support
    round's kept states or the final palette must make the whole section
    unavailable rather than quietly widening or narrowing support membership.
    """
    trace = _synthetic_stage_trace()
    reference, visible = _synthetic_reference(), _synthetic_reference()
    stone_record, dirt_record = _trace_state("minecraft:stone"), _trace_state("minecraft:dirt")
    round_record = {
        "round": 0,
        "states_before": [stone_record, dirt_record],
        "states_kept": [stone_record, dirt_record],
        "states_dropped": [],
    }
    cell = _trace_entry(
        (3, 0, 0), raw_winner=dirt_record, label=dirt_record, final=dirt_record)

    # The predicate itself is the contract: a name the decoder cannot rehydrate
    # and every property shape the writer never emits are malformed records.
    assert benchmark._is_trace_state_record(stone_record) is True
    assert benchmark._is_trace_state_record({"name": "minecraft:stone"}) is True
    assert benchmark._is_trace_state_record({"name": "stone", "properties": []}) is False
    assert benchmark._is_trace_state_record({"name": "minecraft:stone", "properties": None}) is False

    malformed_members = {
        "unnamespaced_name": {"name": "stone", "properties": []},
        "rehydration_invalid_name": {"name": "", "properties": []},
        "properties_not_a_list": {"name": "minecraft:stone", "properties": {"axis": "y"}},
        "properties_string": {"name": "minecraft:stone", "properties": "axis=y"},
        "properties_null": {"name": "minecraft:stone", "properties": None},
        "pair_too_short": {"name": "minecraft:stone", "properties": [["axis"]]},
        "pair_too_long": {"name": "minecraft:stone", "properties": [["axis", "y", "z"]]},
        "pair_not_a_sequence": {"name": "minecraft:stone", "properties": ["axis=y"]},
        "empty_property_value": {"name": "minecraft:stone", "properties": [["axis", ""]]},
        "empty_property_key": {"name": "minecraft:stone", "properties": [["", "y"]]},
        "non_string_property_value": {"name": "minecraft:stone", "properties": [["axis", 3]]},
    }
    malformed_traces = {}
    for label, record in malformed_members.items():
        malformed_traces[f"palette/{label}"] = _trace_shape(
            trace, coordinates=[cell],
            palette={**trace["palette"], "states": [stone_record, record]})
        malformed_traces[f"support_rounds/{label}"] = _trace_shape(
            trace, coordinates=[cell],
            support_rounds=[{**round_record, "states_kept": [stone_record, record]}])
        malformed_traces[f"final_palette/{label}"] = _trace_shape(
            trace, coordinates=[cell],
            final_palette={"states": [stone_record, record]})

    for name, malformed_trace in malformed_traces.items():
        result = benchmark._stagewise_visible_loss(
            reference, {}, visible, stage_trace=malformed_trace,
            translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
        assert result["available"] is False, name
        assert result["unavailable_reason"], name
        assert result["stages"] == [] and result["stage_transitions"] == [], name
        assert result["summary"]["availability_matches"] == {}, name
        assert result["summary"]["largest_stage_loss"] is None, name

    # The decoder's intentional compatibility shapes still compute everywhere:
    # the emitted pair list, an empty list and an absent properties key.
    oak_log_record = _trace_state("minecraft:oak_log", axis="y")
    dark_oak_record = _trace_state("minecraft:dark_oak_log", axis="x")
    compatible = _trace_shape(
        trace, coordinates=[cell],
        palette={**trace["palette"], "states": [
            {"name": "minecraft:stone"}, dirt_record, oak_log_record, dark_oak_record]},
        support_rounds=[{**round_record, "states_kept": [
            {"name": "minecraft:stone", "properties": []}, dirt_record, oak_log_record]}],
        final_palette={"states": [stone_record, dirt_record, oak_log_record]})
    result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=compatible,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True)
    assert result["available"] is True and result["unavailable_reason"] is None
    matches = {stage["stage"]: stage["exact_state_matches"] for stage in result["stages"]}
    assert matches["support"] == 6 and matches["available_palette"] == 7


def test_report_omits_stagewise_instrumentation_by_default(monkeypatch, tmp_path):
    """CLI default behaviour is unchanged: no trace, no stagewise data."""
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)
    report = captured["report"]

    section = report["stagewise_visible_loss"]
    assert section["available"] is False
    assert section["unavailable_reason"] == "stage trace was not requested for this run"
    assert report["report_schema"] == 3
    assert report["effective_settings"]["stage_trace"] is False


# ---------------------------------------------------------------------------
# DS-R36: candidate-availability ceiling at K = 8/16/32, read post hoc from the
# stage trace's own ranked candidate list. The sweep must use the pinned visible
# denominator, separate the truth-in-palette ceiling from geometry presence,
# report a K above the trace's captured depth as uncaptured rather than as a
# zero, and never touch candidate selection, scoring, thresholds or a gate.


def _availability_trace(captured_limit):
    """Three visible cells whose reference state sits at different K depths.

    ``(3,0,0)`` has its truth first, ``(4,0,0)`` second, and ``(5,0,0)`` never
    within the captured list; ``(0,0,0)`` carries a record with no ranked
    candidates, so it is a geometry-present shape rather than a ranking miss.
    """
    stone = _trace_state("minecraft:stone")
    dirt = _trace_state("minecraft:dirt")
    oak_log = _trace_state("minecraft:oak_log", axis="y")
    return {
        "schema": 1,
        "candidate_limit": captured_limit,
        "palette": {"size": 3, "states": [stone, dirt, oak_log]},
        "support_rounds": [],
        "final_palette": {"size": 3, "states": [stone, dirt, oak_log]},
        "coordinates": [
            _trace_entry((0, 0, 0), top=[], final=None),
            _trace_entry(
                (3, 0, 0), top=[(stone, 0.0), (dirt, 1.0), (oak_log, 2.0)],
                raw_winner=stone, label=stone, final=stone),
            _trace_entry(
                (4, 0, 0), top=[(stone, 0.0), (dirt, 1.0)],
                raw_winner=stone, label=stone, final=stone),
            _trace_entry(
                (5, 0, 0), top=[(stone, 0.0), (oak_log, 1.0)],
                raw_winner=stone, label=stone, final=stone),
        ],
    }


def _availability_reference():
    return {
        (0, 0, 0): STONE,
        (3, 0, 0): STONE,
        (4, 0, 0): BlockState("minecraft:dirt"),
        (5, 0, 0): BlockState("minecraft:dirt"),
    }


def _availability_visible():
    return {(0, 0, 0): 5, (3, 0, 0): 5, (4, 0, 0): 5, (5, 0, 0): 5}


def test_candidate_availability_sweep_reports_depth_and_uncaptured_ks():
    result = benchmark._stagewise_visible_loss(
        _availability_reference(), {}, _availability_visible(),
        stage_trace=_availability_trace(captured_limit=3),
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    assert result["available"] is True
    availability = result["candidate_availability"]
    assert isinstance(availability, dict)
    assert availability["captured_limit"] == 3
    assert availability["ks"] == list(benchmark.STAGE_TRACE_CANDIDATE_AVAILABILITY_K)
    assert availability["visible_denominator"] == 4
    assert availability["truth_in_palette_denominator"] == 4
    assert availability["geometry_present_denominator"] == 3
    assert availability["cells_without_a_trace_record"] == 0
    assert availability["cells_with_no_ranked_candidates"] == 1
    assert availability["reference_states_beyond_palette"] == 0

    checkpoints = {checkpoint["k"]: checkpoint for checkpoint in availability["checkpoints"]}
    # The captured list reaches 3, so K=8/16/32 are honestly uncaptured.
    assert availability["uncaptured_ks"] == [8, 16, 32]
    assert all(checkpoint["captured"] is False for checkpoint in checkpoints.values())
    assert all(checkpoint["captured_limit_reason"] for checkpoint in checkpoints.values())
    assert all(
        checkpoint["truth_in_palette_denominator"] == 4
        and checkpoint["geometry_present_denominator"] == 3
        for checkpoint in checkpoints.values()
    )
    # K=1 finds only (3,0,0); K=2 also finds (4,0,0) - a depth ceiling, not a gate.
    assert checkpoints[8]["exact_state_available"] == 2
    assert checkpoints[8]["exact_state_available_in_palette"] == 2
    assert checkpoints[8]["exact_state_recall_in_palette"] == pytest.approx(2 / 4)
    assert checkpoints[8]["captured_candidates_beyond_palette"] == 0
    assert checkpoints[8]["mean_captured_candidates_per_visible_cell"] == pytest.approx(
        (3 + 2 + 2 + 0) / 4)


def test_candidate_availability_labels_a_captured_depth_as_measured():
    result = benchmark._stagewise_visible_loss(
        _availability_reference(), {}, _availability_visible(),
        stage_trace=_availability_trace(captured_limit=32),
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    availability = result["candidate_availability"]
    assert availability["captured_limit_sufficient_for_ks"] == [8, 16, 32]
    assert availability["uncaptured_ks"] == []
    checkpoints = {checkpoint["k"]: checkpoint for checkpoint in availability["checkpoints"]}
    # Only (3,0,0) and (4,0,0) ever rank the truth inside the captured list.
    assert checkpoints[8]["exact_state_available"] == 2
    assert checkpoints[8]["exact_state_available_in_palette"] == 2
    assert checkpoints[8]["exact_state_recall_in_palette"] == pytest.approx(0.5)
    # The report is additive: the six checkpoints are untouched by the sweep.
    assert len(result["stages"]) == 6
    assert result["summary"]["geometry_missing_cells"] == 1


def test_candidate_availability_is_absent_when_the_trace_is_unusable():
    result = benchmark._stagewise_visible_loss(
        _availability_reference(), {}, _availability_visible(),
        stage_trace=None, translation=(0, 0, 0), rotation_quarter_turns=0, requested=False,
    )

    assert result["available"] is False
    assert result["candidate_availability"] is None


def test_candidate_availability_counts_states_beyond_the_canonical_palette():
    stone = _trace_state("minecraft:stone")
    dirt = _trace_state("minecraft:dirt")
    reference = dict(_availability_reference())
    reference[(9, 0, 0)] = BlockState("minecraft:cobblestone")
    visible = dict(_availability_visible())
    visible[(9, 0, 0)] = 5
    trace = _availability_trace(captured_limit=32)
    trace["coordinates"].append(
        _trace_entry((9, 0, 0), top=[(stone, 0.0), (dirt, 1.0)], raw_winner=stone,
                     label=stone, final=stone))

    result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=trace,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    availability = result["candidate_availability"]
    assert availability["visible_denominator"] == 5
    assert availability["truth_in_palette_denominator"] == 4
    assert availability["reference_states_beyond_palette"] == 1
    checkpoints = {checkpoint["k"]: checkpoint for checkpoint in availability["checkpoints"]}
    assert checkpoints[8]["exact_state_available"] == 2
    assert checkpoints[8]["exact_state_available_in_palette"] == 2
    assert checkpoints[8]["exact_state_recall_in_palette"] == pytest.approx(0.5)


def test_candidate_availability_does_not_count_a_duplicate_name_twice():
    """The evaluator palette can carry several entries under one name.

    The candidate checkpoint is set membership, so a name repeated in the ranked
    list must occupy one rank, not two; otherwise deeper K would read as recall
    the checkpoint itself would not grant.
    """
    stone = _trace_state("minecraft:stone")
    dirt = _trace_state("minecraft:dirt")
    reference = {(0, 0, 0): BlockState("minecraft:dirt")}
    visible = {(0, 0, 0): 5}
    trace = {
        "schema": 1,
        "candidate_limit": 32,
        "palette": {"size": 2, "states": [stone, dirt]},
        "support_rounds": [],
        "final_palette": {"size": 2, "states": [stone, dirt]},
        "coordinates": [
            _trace_entry(
                (0, 0, 0),
                top=[(stone, 0.0), (dirt, 1.0), (dirt, 1.5)],
                raw_winner=stone, label=stone, final=stone),
        ],
    }

    result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=trace,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    availability = result["candidate_availability"]
    checkpoint = availability["checkpoints"][0]
    # The truth is second in the distinct order, after one stone occurrence.
    assert checkpoint["exact_state_available"] == 1
    assert checkpoint["mean_captured_candidates_per_visible_cell"] == pytest.approx(2.0)
    assert checkpoint["captured_candidates_beyond_palette"] == 0


def test_candidate_availability_reads_the_ordered_list_not_the_top_k_set():
    """Production's top-K is an unordered argpartition set, not a cost slice.

    A state can rank inside the cheapest eight by cost yet sit outside
    ``top_candidates``, so the sweep must read the ordered field whenever the
    trace carries one. The K=8 row is therefore a cheapest-eight ceiling and is
    allowed to differ from the ``candidate`` set-membership checkpoint.
    """
    stone = _trace_state("minecraft:stone")
    dirt = _trace_state("minecraft:dirt")
    reference = {(0, 0, 0): BlockState("minecraft:dirt")}
    visible = {(0, 0, 0): 5}
    entry = _trace_entry((0, 0, 0), top=[(stone, 0.0)], raw_winner=stone,
                         label=stone, final=stone)
    entry["ranked_candidates"] = [
        {"state": stone, "cost": 0.0},
        {"state": dirt, "cost": 1.0},
    ]
    trace = {
        "schema": 1,
        "candidate_limit": 8,
        "palette": {"size": 2, "states": [stone, dirt]},
        "support_rounds": [],
        "final_palette": {"size": 2, "states": [stone, dirt]},
        "coordinates": [entry],
    }

    result = benchmark._stagewise_visible_loss(
        reference, {}, visible, stage_trace=trace,
        translation=(0, 0, 0), rotation_quarter_turns=0, requested=True,
    )

    # The set-based candidate checkpoint excludes dirt; the ordered K=8 list has it.
    matches = {stage["stage"]: stage["exact_state_matches"] for stage in result["stages"]}
    assert matches["candidate"] == 0
    checkpoint = result["candidate_availability"]["checkpoints"][0]
    assert checkpoint["k"] == 8
    assert checkpoint["exact_state_available"] == 1
    assert checkpoint["mean_captured_candidates_per_visible_cell"] == pytest.approx(2.0)


def test_stage_trace_candidate_limit_reaches_reconstruction_kwargs(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.json"
    captured = _capture_report_run(
        monkeypatch, tmp_path, usable_mask_paths=None,
        stage_trace_path=trace_path, stage_trace_candidate_limit=32)

    assert captured["reconstruct_kwargs"]["stage_trace_candidate_limit"] == 32
    settings = captured["report"]["effective_settings"]
    assert settings["stage_trace_candidate_limit"] == 32


def test_stage_trace_candidate_limit_is_omitted_without_a_trace(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)

    assert captured["reconstruct_kwargs"]["stage_trace_candidate_limit"] == (
        raycast.STAGE_TRACE_CANDIDATE_LIMIT)
    assert captured["report"]["effective_settings"]["stage_trace_candidate_limit"] is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "8", None])
def test_invalid_stage_trace_candidate_limit_is_rejected(monkeypatch, tmp_path, limit):
    reference = tmp_path / "reference.litematic"
    reference.write_bytes(b"reference")
    camera_manifest = tmp_path / "cameras.json"
    camera_manifest.write_text("{}", encoding="utf-8")
    scene_definition = tmp_path / "scene.json"
    scene_definition.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic", mode="supplied-camera",
            reference=reference, camera_manifest=camera_manifest,
            scene_definition=scene_definition, jar=None, adjacency_strength=0.0,
            allow_default_palette=True, stage_trace_path=tmp_path / "trace.json",
            stage_trace_candidate_limit=limit)


def test_stage_trace_candidate_limit_requires_a_trace_path(monkeypatch, tmp_path):
    reference = tmp_path / "reference.litematic"
    reference.write_bytes(b"reference")
    camera_manifest = tmp_path / "cameras.json"
    camera_manifest.write_text("{}", encoding="utf-8")
    scene_definition = tmp_path / "scene.json"
    scene_definition.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="requires stage_trace_path"):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic", mode="supplied-camera",
            reference=reference, camera_manifest=camera_manifest,
            scene_definition=scene_definition, jar=None, adjacency_strength=0.0,
            allow_default_palette=True, stage_trace_candidate_limit=32)


def test_cli_parses_the_stage_trace_candidate_limit():
    parser = benchmark._build_parser()

    default = parser.parse_args(["template", "output"])
    explicit = parser.parse_args(
        ["template", "output", "--stage-trace", "trace.json",
         "--stage-trace-candidate-limit", "32"])

    assert default.stage_trace_candidate_limit == raycast.STAGE_TRACE_CANDIDATE_LIMIT
    assert explicit.stage_trace_candidate_limit == 32
    for raw in ("0", "-1", "1.5", "not-a-number"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["template", "output", "--stage-trace-candidate-limit", raw])


# ---------------------------------------------------------------------------
# E2c: the opt-in resolved-face payload sidecar at the benchmark boundary. It
# requires the E2A trace, refuses an existing or aliasing destination, binds the
# exact trace file-content digest, and refuses an over-bound payload before any
# destination or temporary artifact exists.


def _sidecar_run_kwargs(tmp_path):
    """Required ``run`` arguments for the refusal paths, without fixture input."""
    reference = tmp_path / "reference.litematic"
    reference.write_bytes(b"reference")
    camera_manifest = tmp_path / "cameras.json"
    camera_manifest.write_text("{}", encoding="utf-8")
    scene_definition = tmp_path / "scene.json"
    scene_definition.write_text("{}", encoding="utf-8")
    return dict(
        mode="supplied-camera", reference=reference, camera_manifest=camera_manifest,
        scene_definition=scene_definition, jar=None, adjacency_strength=0.0,
        allow_default_palette=True,
    )


def _sidecar_document():
    return {
        "schema": raycast.STAGE_SIDECAR_SCHEMA,
        "trace_schema": 2,
        "observation_count": 1,
        "resolved_face_records": 1,
        "excluded_fallback_observations": 0,
        "records": [{
            "ordinal": 0,
            "frame_id": "frame.png",
            "coordinate": [0, 0, 0],
            "slot": "north",
            "source": "resolved_face",
            "raw_patch": {"dtype": "<f4", "shape": [16, 16, 3], "base64": "AAAAAA=="},
            "observed_patch": {"dtype": "<f4", "shape": [16, 16, 3], "base64": "AAAAAA=="},
            "valid": {"dtype": "|b1", "shape": [16, 16], "base64": "AA=="},
            "coverage": {"dtype": "<f4", "shape": [16, 16], "base64": "AAAAAA=="},
            "digests": {
                "raw_patch": "0" * 64, "observed_patch": "1" * 64,
                "valid": "2" * 64, "coverage": "3" * 64,
            },
        }],
    }


def test_cli_parses_the_stage_sidecar_option():
    parser = benchmark._build_parser()

    default = parser.parse_args(["template", "output"])
    explicit = parser.parse_args(
        ["template", "output", "--stage-trace", "trace.json",
         "--stage-sidecar", "sidecar.json"])

    assert default.stage_sidecar is None
    assert explicit.stage_sidecar == Path("sidecar.json")


def test_stage_sidecar_requires_a_trace_path_and_writes_nothing(tmp_path):
    destination = tmp_path / "sidecar.json"

    with pytest.raises(ValueError, match="requires stage_trace_path"):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic",
            stage_sidecar_path=destination, **_sidecar_run_kwargs(tmp_path))

    assert not destination.exists()


def test_stage_sidecar_refuses_a_path_that_names_the_trace(tmp_path):
    same = tmp_path / "trace.json"

    with pytest.raises(ValueError, match="must be different files"):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic",
            stage_trace_path=same, stage_sidecar_path=same,
            **_sidecar_run_kwargs(tmp_path))

    assert not same.exists()


def test_paths_alias_is_case_insensitive_on_windows():
    first = Path("Template") / "Trace.JSON"
    second = Path("template") / "trace.json"

    assert benchmark._paths_alias(first, second) is (os.name == "nt")


def test_paths_alias_resolves_a_linked_parent_canonically(monkeypatch, tmp_path):
    """Distinct lexical paths through a linked parent are the same destination.

    The canonicalizer is exercised deterministically, without creating a real
    symlink/reparse point: ``realpath`` maps the linked parent onto its real
    target exactly as it does for a directory junction, while the two spellings
    differ under the old ``abspath`` + ``normcase`` check.
    """
    linked = tmp_path / "linked" / "trace.json"
    real = tmp_path / "real" / "trace.json"

    def fake_realpath(value):
        return str(value).replace(str(tmp_path / "linked"), str(tmp_path / "real"))

    monkeypatch.setattr(benchmark.os.path, "realpath", fake_realpath)

    assert benchmark._canonical_path(linked) == benchmark._canonical_path(real)
    assert benchmark._paths_alias(linked, real) is True


def test_stage_sidecar_refuses_a_canonically_aliased_trace(monkeypatch, tmp_path):
    """``run`` refuses two spellings that resolve to the same linked file."""
    trace_path = tmp_path / "real" / "trace.json"
    sidecar_path = tmp_path / "linked" / "trace.json"

    def fake_realpath(value):
        return str(value).replace(str(tmp_path / "linked"), str(tmp_path / "real"))

    monkeypatch.setattr(benchmark.os.path, "realpath", fake_realpath)

    with pytest.raises(ValueError, match="must be different files"):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic",
            stage_trace_path=trace_path, stage_sidecar_path=sidecar_path,
            **_sidecar_run_kwargs(tmp_path))

    assert not trace_path.exists()
    assert not sidecar_path.exists()
    assert not (tmp_path / "out.litematic").exists()


def test_stage_sidecar_refuses_an_existing_destination_without_touching_it(tmp_path):
    destination = tmp_path / "sidecar.json"
    destination.write_text("keep me", encoding="utf-8")
    trace_path = tmp_path / "trace.json"

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        benchmark.run(
            tmp_path, tmp_path / "out.litematic",
            stage_trace_path=trace_path, stage_sidecar_path=destination,
            **_sidecar_run_kwargs(tmp_path))

    assert destination.read_text(encoding="utf-8") == "keep me"
    assert not trace_path.exists()
    assert not (tmp_path / "out.litematic").exists()


def test_stage_sidecar_effective_setting_is_disabled_by_default(monkeypatch, tmp_path):
    captured = _capture_report_run(monkeypatch, tmp_path, usable_mask_paths=None)

    assert captured["reconstruct_kwargs"]["stage_sidecar"] is None
    settings = captured["report"]["effective_settings"]
    assert settings["stage_sidecar"] is False
    assert settings["stage_sidecar_records"] is None


def test_stage_sidecar_writes_atomically_and_binds_the_trace_content_digest(
        monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.json"
    sidecar_path = tmp_path / "sidecar.json"
    document = _sidecar_document()

    captured = _capture_report_run(
        monkeypatch, tmp_path, usable_mask_paths=None,
        stage_trace_path=trace_path, stage_sidecar_path=sidecar_path,
        sidecar_document=document)

    written = json.loads(sidecar_path.read_text(encoding="utf-8"))
    trace_bytes = trace_path.read_bytes()
    # The sidecar binds the exact bytes of the trace file it is keyed to.
    assert written["trace_sha256"] == hashlib.sha256(trace_bytes).hexdigest()
    assert written["trace_bytes"] == len(trace_bytes)
    assert written["schema"] == raycast.STAGE_SIDECAR_SCHEMA
    assert written["records"] == document["records"]
    # The opt-in buffer reaches reconstruction, and the report records the setting.
    assert captured["reconstruct_kwargs"]["stage_sidecar"] is not None
    settings = captured["report"]["effective_settings"]
    assert settings["stage_sidecar"] is True
    assert settings["stage_sidecar_records"] == 1


def test_trace_output_bytes_match_the_previous_text_writer(tmp_path):
    """Preflighting must not change the trace destination's exact bytes.

    ``_trace_output_bytes`` is what the trace path hashes, binds into the sidecar
    and writes, so it must equal byte-for-byte what the previous text-mode writer
    produced under the platform's newline translation - otherwise the digest the
    sidecar binds would stop describing the trace file, and existing traces would
    silently change bytes.
    """
    trace_text = json.dumps({"schema": 2, "observations": [{"a": 1}]}, indent=2,
                            sort_keys=True) + "\n"
    previous = tmp_path / "previous.json"
    previous.write_text(trace_text, encoding="utf-8")
    preflighted_bytes = benchmark._trace_output_bytes(trace_text)
    preflighted = tmp_path / "preflighted.json"
    preflighted.write_bytes(preflighted_bytes)

    assert preflighted_bytes == previous.read_bytes()
    assert hashlib.sha256(preflighted_bytes).hexdigest() == hashlib.sha256(
        previous.read_bytes()).hexdigest()


def test_stage_sidecar_byte_bound_refusal_creates_neither_trace_nor_sidecar(
        monkeypatch, tmp_path):
    """An over-bound sidecar must refuse before the trace is written.

    The byte bound can only be measured once the exact trace bytes and sidecar
    serialization exist, so this is the ordering the preflight fixes: bind and
    validate first, write the trace only after the sidecar has passed. A byte-bound
    refusal must therefore leave no trace, no sidecar, no temporary artifact and no
    report. The candidate output is written earlier by the export stage, before the
    preflight, and is not part of this contract.
    """
    trace_path = tmp_path / "trace.json"
    sidecar_path = tmp_path / "sidecar.json"
    # Any real serialized sidecar is non-empty, so a zero bound is a guaranteed
    # byte-bound refusal that does not depend on the fixture's trace size.
    monkeypatch.setattr(benchmark.raycast, "STAGE_SIDECAR_MAX_BYTES", 0)

    with pytest.raises(ValueError, match="byte bound"):
        _capture_report_run(
            monkeypatch, tmp_path, usable_mask_paths=None,
            stage_trace_path=trace_path, stage_sidecar_path=sidecar_path,
            sidecar_document=_sidecar_document())

    assert not trace_path.exists()
    assert not sidecar_path.exists()
    assert not (tmp_path / "report.json").exists()
    # No task-owned temporary artifact survives the refusal.
    assert [path.name for path in tmp_path.iterdir() if path.suffix == ".tmp"] == []


def test_stage_sidecar_refuses_over_the_record_bound():
    document = {"records": [{} for _ in range(raycast.STAGE_SIDECAR_MAX_RECORDS + 1)]}

    with pytest.raises(ValueError, match="record bound"):
        benchmark._stage_sidecar_payload(document)


def test_stage_sidecar_refuses_over_the_byte_bound(monkeypatch):
    monkeypatch.setattr(benchmark.raycast, "STAGE_SIDECAR_MAX_BYTES", 64)
    document = {"records": [{"pad": "x" * 256}]}

    with pytest.raises(ValueError, match="byte bound"):
        benchmark._stage_sidecar_payload(document)


def test_stage_sidecar_payload_rejects_a_malformed_document():
    with pytest.raises(ValueError, match="records list"):
        benchmark._stage_sidecar_payload({})
    with pytest.raises(ValueError, match="must be an object"):
        benchmark._stage_sidecar_payload([])


def test_stage_sidecar_failure_leaves_no_destination_or_temp_artifact(
        monkeypatch, tmp_path):
    destination = tmp_path / "sidecar.json"
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n", encoding="utf-8")

    def fail_link(source, target):
        raise OSError("simulated link failure")

    monkeypatch.setattr(benchmark.os, "link", fail_link)

    with pytest.raises(OSError):
        benchmark._write_stage_sidecar(destination, b"payload", trace_path=trace)

    assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["trace.json"]
