"""Evaluation-only A2 runner: visible-denominator attribution for both routes.

This is the only A2 artifact permitted to read frozen evaluator paths and to call
production/evaluator loaders.  It replays the accepted production functions on
frozen inputs (Route A's normal-mode single-view sequence and, optionally, Route
B's ``derive_image_output_from_paths``), computes the frozen evaluator-only truth
- the pinned reference litematic, the evaluator cameras, the tracked frozen mask
specification and the raycast visible-pixel counts - and then scores the reachable
checkpoints through the pure A2 module
:mod:`mc_photogrammetry.evaluation_image_only_attribution`.

Discipline
----------

* **Evaluator-only truth.**  The reference litematic, the evaluator cameras, the
  scene transform and the frozen masks are read *here*, after reconstruction, and
  are never passed to production.  No production module imports this script.
* **Tracked mask specifications, fail-closed pins.**  The scored mask comes from
  the reviewed tracked JSON specification ``docs/evaluation_masks/<fixture>/``
  (the same rectangle/polygon semantics as
  ``benchmark_templates._load_usable_masks``), not from the ignored ``.masks/``
  PNG derivatives.  Whenever reviewed pins are supplied, the observed scored
  boolean domain must equal the pinned ``domain_sha256`` for every frame or the
  run refuses and builds no attribution; ``--allow-pin-mismatch`` explicitly
  downgrades that to a typed caveat for diagnostic-only runs.
* **No scale fitting.**  The alignment comes from the accepted evaluator
  ``best_visible_alignment`` (Route A) or its coordinate-only specialisation
  ``best_visible_alignment_for_coordinates`` (Route B geometry-only placement/
  volume records) - integer translation plus lattice Y-quarter-turn, or a supplied
  alignment record; no scale, reflection or per-cell transform is ever derived.
* **No manual inspection.**  Frame images are read only to obtain their pixel
  dimensions and content digests for the pinned identity check; no screenshot is
  displayed or interpreted.
* **No accuracy claim from policy.**  Status counts are policy rates; the first
  stop is a frontier indicator, never a causal attribution of loss.

Typical invocations::

    .\\.venv\\Scripts\\python.exe scripts\\evaluation\\report_image_only_attribution.py ^
        --route route-a --image "templates 2\\2026-09-02_13.22.14.png" ^
        --reference "templates 2\\Template 2.litematic" --scene "templates 2\\scene.json" ^
        --cameras "templates 2\\cameras.json" ^
        --masks docs\\evaluation_masks\\templates-2 ^
        --pins docs\\reports\\evaluator-fixture-pins.json --fixture-role template-2

    .\\.venv\\Scripts\\python.exe scripts\\evaluation\\report_image_only_attribution.py ^
        --route route-b --fixture "templates 2" --fixture-role template-2 ^
        --reference "templates 2\\Template 2.litematic" --scene "templates 2\\scene.json" ^
        --cameras "templates 2\\cameras.json" ^
        --masks docs\\evaluation_masks\\templates-2 ^
        --pins docs\\reports\\evaluator-fixture-pins.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mc_photogrammetry.evaluation import best_visible_alignment  # noqa: E402
from mc_photogrammetry.evaluation_image_only_attribution import (  # noqa: E402
    ATTRIBUTION_PRODUCER,
    ENTRY_MATCH_UNBOUND_REASON,
    PALETTE_LOSS_PARTITION_ROUTE_B_REASON,
    AttributionRefusal,
    GeometryCellEvidence,
    alignment_record,
    attribution_digest,
    best_visible_alignment_for_coordinates,
    route_a_attribution,
    route_b_attribution,
    visible_reference_from_counts,
)
from mc_photogrammetry.evaluation_image_only_frontier import (  # noqa: E402
    FrontierRefusal,
    canonical_json,
    route_a_frontier,
    route_b_frontier,
)
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.export.visibility import visible_cells  # noqa: E402
from mc_photogrammetry.pipeline.image_derived_production import (  # noqa: E402
    ImageDerivedProductionResult,
    derive_image_output_from_paths,
)
from mc_photogrammetry.pipeline.image_derived_production_record import (  # noqa: E402
    ProductionEnvelopeError,
    production_envelope_from_record,
)
from mc_photogrammetry.project.store import load_project  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction.camera import load_camera_manifest  # noqa: E402
from mc_photogrammetry.reconstruction.scene import load_scene_transform  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_SCHEMA = 1
MASK_KEEP_TOKEN = "white keeps a pixel"


# ---------------------------------------------------------------------------
# Frozen-input identity
# ---------------------------------------------------------------------------

def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def mask_domain_sha256(mask) -> str:
    """The accepted scored boolean-domain digest (shape + bytes), not the file bytes."""
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_pins(path: str | Path, role: str) -> dict:
    """Load one reviewed fixture pin, failing closed on an unpinned role."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_schema") != PIN_SCHEMA:
        raise AttributionRefusal(
            "the evaluator fixture pin manifest schema is unsupported")
    entry = (manifest.get("roles") or {}).get(role)
    if not isinstance(entry, dict):
        raise AttributionRefusal(f"the fixture role {role} is not declared")
    if entry.get("reviewed") is not True:
        raise AttributionRefusal(f"the fixture role {role} is not reviewed/pinned")
    missing = [key for key in ("fixture", "reference_sha256", "frames", "usable_masks")
               if not entry.get(key)]
    if missing:
        raise AttributionRefusal(
            f"the fixture role {role} is missing pinned {missing[0]}")
    return entry


def resolve_mask_path(directory: Path, frame_name: str) -> Path | None:
    """Resolve one frame's frozen mask, preferring the tracked JSON specification.

    ``docs/evaluation_masks/<fixture>/*.json`` is the reviewed, tracked source of
    truth.  The ignored ``.masks/`` PNG derivatives are a mutable local artifact
    and are only a fallback when no specification exists for the frame.
    """
    directory = Path(directory)
    specification = directory / f"{Path(frame_name).stem}.json"
    if specification.is_file():
        return specification
    candidate = directory / frame_name
    if candidate.is_file():
        return candidate
    return None


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> object:
    """Load the scored boolean mask through the accepted evaluator semantics.

    A tracked JSON specification freezes integer excluded rectangles and polygons
    for a fixed image size, exactly as ``benchmark_templates._load_usable_masks``
    builds them: the mask starts all-``True`` (white keeps a pixel), rectangles are
    cleared, then polygons are filled ``False``.  A PNG mask is read with white as
    usable.  The resulting boolean domain is what the pinned ``domain_sha256``
    describes.
    """
    import numpy as np

    if not path.is_file():
        raise AttributionRefusal(f"the frozen mask is unavailable: {path}")
    if path.suffix.lower() == ".json":
        specification = json.loads(path.read_text(encoding="utf-8"))
        try:
            width, height = specification["image_size"]
        except (KeyError, TypeError, ValueError) as error:
            raise AttributionRefusal(
                f"the frozen mask specification {path.name} is malformed: {error}")
        if (int(height), int(width)) != expected_shape:
            raise AttributionRefusal(
                f"frozen mask {path.name} fixes {(int(width), int(height))}, "
                f"expected {(expected_shape[1], expected_shape[0])}")
        mask = np.ones(expected_shape, dtype=bool)
        for rectangle in specification.get("excluded_rectangles", ()):
            try:
                x0, y0, x1, y1 = (int(value) for value in rectangle)
            except (TypeError, ValueError) as error:
                raise AttributionRefusal(
                    f"frozen mask {path.name} has a malformed rectangle: {error}")
            if not (0 <= x0 <= x1 <= expected_shape[1]
                    and 0 <= y0 <= y1 <= expected_shape[0]):
                raise AttributionRefusal(
                    f"frozen mask {path.name} has out-of-bounds rectangle {rectangle}")
            mask[y0:y1, x0:x1] = False
        for polygon in specification.get("excluded_polygons", ()):
            points = np.asarray(polygon, dtype=np.int32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                raise AttributionRefusal(
                    f"frozen mask {path.name} has invalid polygon {polygon}")
            if (points[:, 0] < 0).any() or (points[:, 0] >= expected_shape[1]).any() \
                    or (points[:, 1] < 0).any() \
                    or (points[:, 1] >= expected_shape[0]).any():
                raise AttributionRefusal(
                    f"frozen mask {path.name} has out-of-bounds polygon {polygon}")
            import cv2

            cv2.fillPoly(mask, [points], False)
    else:
        from PIL import Image

        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != expected_shape:
        raise AttributionRefusal(
            f"frozen mask {path.name} has shape {mask.shape}, expected {expected_shape}")
    return mask


def _frame_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def load_frozen_truth(
    *, fixture: Path, reference: Path, scene_path: Path, cameras_path: Path,
    mask_dir: Path | None, filter_frames: set[str] | None, pins: dict | None,
    frame_paths: dict[str, Path] | None = None, require_pins: bool | None = None,
) -> dict:
    """Load the evaluator-only truth and verify it against the pinned identity.

    The reference litematic and the frame content digests fail closed against the
    reviewed pins.  The frozen mask is read from the tracked reviewed JSON
    specification under ``docs/evaluation_masks/<fixture>/`` when one exists, and
    its scored boolean *domain* is checked against the reviewed pin whenever pins
    are supplied: ``require_pins`` defaults to "pins were supplied", so a pinned
    run can never silently score a divergent mask domain.
    """
    if require_pins is None:
        require_pins = pins is not None
    reference_bytes = reference.read_bytes()
    reference_sha = _sha256_bytes(reference_bytes)
    snapshot = load_litematic(reference)
    if not snapshot.blocks:
        raise AttributionRefusal("the reference litematic carries no blocks")
    if pins is not None and pins["reference_sha256"] != reference_sha:
        raise AttributionRefusal(
            "the reference litematic does not match the reviewed fixture pin")

    scenes = load_scene_transform(scene_path)
    cameras = load_camera_manifest(cameras_path)
    pinned_frames = {item["name"]: item for item in (pins or {}).get("frames", [])}
    pinned_masks = {item["frame"]: item for item in (pins or {}).get("usable_masks", [])}

    resolved_frames: dict[str, Path] = dict(frame_paths or {})
    if not resolved_frames:
        for name in sorted(cameras):
            candidate = fixture / name
            if candidate.is_file():
                resolved_frames[name] = candidate
    if filter_frames is not None:
        resolved_frames = {name: path for name, path in resolved_frames.items()
                           if name in filter_frames}
    if not resolved_frames:
        raise AttributionRefusal("no frozen frame could be resolved for the fixture")

    frame_checks: list[dict] = []
    usable_masks: dict[str, object] = {}
    resolutions: dict[str, tuple[int, int]] = {}
    caveats: list[str] = []
    mask_pins_match = True
    pinned_masks_checked = 0
    for name in sorted(resolved_frames):
        path = resolved_frames[name]
        digest = _sha256_bytes(path.read_bytes())
        if pinned_frames and name in pinned_frames \
                and pinned_frames[name]["sha256"] != digest:
            raise AttributionRefusal(
                f"frozen frame {name} does not match the reviewed fixture pin")
        width, height = _frame_size(path)
        resolutions[name] = (width, height)
        mask_record = {"frame": name, "file": None, "file_sha256": None,
                       "spec": None, "spec_sha256": None,
                       "domain_sha256": None, "pin_matches": None,
                       "kind": None}
        if mask_dir is not None:
            mask_path = resolve_mask_path(Path(mask_dir), name)
            if mask_path is None:
                raise AttributionRefusal(
                    f"no frozen mask specification or mask exists for {name} "
                    f"under {mask_dir}")
            mask = _load_mask(mask_path, (height, width))
            domain = mask_domain_sha256(mask)
            pin_expected = (pinned_masks.get(name) or {}).get("domain_sha256")
            pin_matches = None
            if pin_expected is not None:
                pinned_masks_checked += 1
                pin_matches = pin_expected == domain
                mask_pins_match = mask_pins_match and pin_matches
            if pin_matches is False and require_pins:
                raise AttributionRefusal(
                    f"frozen mask for {name} does not match the reviewed pin: "
                    f"observed domain {domain} != pinned {pin_expected} "
                    f"({mask_path})")
            usable_masks[name] = mask
            mask_record = {"frame": name, "file": str(mask_path),
                           "file_sha256": _sha256_bytes(mask_path.read_bytes()),
                           "spec": (str(mask_path)
                                    if mask_path.suffix.lower() == ".json" else None),
                           "spec_sha256": (
                               _sha256_bytes(mask_path.read_bytes())
                               if mask_path.suffix.lower() == ".json" else None),
                           "domain_sha256": domain, "pin_matches": pin_matches,
                           "kind": ("json_specification"
                                    if mask_path.suffix.lower() == ".json" else "png")}
        frame_checks.append({"name": name, "sha256": digest,
                             "size": [width, height], "mask": mask_record})

    if pins is not None and pinned_masks_checked == 0:
        raise AttributionRefusal(
            "reviewed pins were supplied but no frozen mask domain was checked "
            "against them; supply --masks so the pinned scored domain is verified")
    if pins is not None and mask_dir is not None and not mask_pins_match:
        caveats.append(
            "the frozen mask domain does not match the reviewed fixture pin; "
            "the tracked reference and frame digests do match, so this run's visible "
            "denominator is a local-mask diagnostic and not the pinned identity")
    missing_cameras = sorted(set(resolved_frames) - set(cameras))
    if missing_cameras:
        raise AttributionRefusal(
            f"the evaluator camera manifest lacks {missing_cameras[0]}")
    selected_cameras = {name: cameras[name] for name in sorted(resolved_frames)}
    counts = visible_cells(
        snapshot.blocks, scenes, selected_cameras, resolutions,
        usable_masks=usable_masks or None)
    visible = visible_reference_from_counts(
        snapshot.blocks, counts, reference_sha256=reference_sha,
        source=("evaluator_raycast_pinned_reference_cameras_frozen_masks"
                if usable_masks else "evaluator_raycast_no_frozen_mask"))
    mask_kinds = {record["mask"]["kind"] for record in frame_checks
                  if record["mask"]["kind"] is not None}
    if not usable_masks:
        mask_origin = "none"
    elif mask_kinds == {"json_specification"}:
        mask_origin = "tracked_json_specification"
    elif mask_kinds == {"png"}:
        mask_origin = "local_png"
    else:
        mask_origin = "mixed"
    return {
        "reference": snapshot.blocks,
        "reference_sha256": reference_sha,
        "visible_reference": visible,
        "frames": frame_checks,
        "camera_names": sorted(selected_cameras),
        "cameras": selected_cameras,
        "scene_path": scene_path,
        "scene_transform": scenes,
        "frame_paths": dict(resolved_frames),
        "mask_origin": mask_origin,
        "mask_convention": MASK_KEEP_TOKEN,
        "mask_pinned_domains_checked": pinned_masks_checked,
        "mask_pins_match": None if not pinned_masks else mask_pins_match,
        "reference_pin_matches": (None if pins is None
                                  else pins["reference_sha256"] == reference_sha),
        "caveats": caveats,
    }


def compute_alignment(reference, candidate, visible_reference) -> dict:
    """The one permitted alignment, computed by the accepted evaluator."""
    turns, translation = best_visible_alignment(
        reference, candidate, visible_reference.visible)
    return alignment_record(rotation_quarter_turns=int(turns),
                            translation=tuple(int(value) for value in translation),
                            source="evaluation.best_visible_alignment")


# ---------------------------------------------------------------------------
# Route A
# ---------------------------------------------------------------------------

def _states_of(volume) -> dict:
    if volume is None:
        return {}
    return {coordinate: observation.state
            for coordinate, observation in dict(volume.blocks).items()}


def _palette_states(palette) -> tuple:
    return tuple(item.state if hasattr(item, "state") else item for item in palette)


def _route_a_replay():
    """Load the accepted A1 Route A replay runner by file path (no package)."""
    import importlib.util

    module_name = "_imgloss_a2_route_a_replay"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / "scripts" / "evaluation" / "run_image_only_route_a.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AttributionRefusal("the Route A replay runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_route_a(
    *, image: Path, truth: dict | None = None, workdir: Path | None = None,
    arm: str = "camera-projected",
) -> dict:
    """Replay the normal-mode Route A sequence and return its production outcome.

    Production-only: this never receives, reads or configures anything from the
    evaluator truth, so the reconstruction stage cannot be influenced by it.  The
    scoring step is :func:`_score_route_a`, which the runner calls only after the
    evaluator references, scene, cameras and masks have been opened.
    """
    replay_route_a = _route_a_replay().replay

    directory = Path(workdir) if workdir is not None \
        else Path(tempfile.mkdtemp(prefix="imgloss-a2-route-a-"))
    directory.mkdir(parents=True, exist_ok=True)
    outcome = replay_route_a(image, workdir=directory, arm=arm)
    (directory / "candidate-frozen.json").write_text(
        outcome.candidate_seal + "\n", encoding="utf-8")
    return outcome, directory


def _score_route_a(
    *, image: Path, outcome, directory: Path, truth: dict, cli_parity: bool,
    include_palette_loss_partition: bool = False,
) -> dict:
    """Score the already-completed Route A production outcome against the truth.

    ``include_palette_loss_partition`` is the default-off Route A opt-in that
    appends the reviewed five in-palette loss-partition rows at the
    ``palette_label`` checkpoint; with it absent the report is unchanged.
    """
    arm = getattr(getattr(outcome, "settings", None), "arm", "camera-projected")
    if cli_parity and arm != "camera-projected":
        raise AttributionRefusal("pixel-grid ablation is not normal CLI parity")
    facade_states = _states_of(outcome.facade)
    terminal_states = _states_of(outcome.decoded_export)
    palette_states = _palette_states(outcome.palette)
    frontier = outcome.route_a.frontier
    base = facade_states or terminal_states
    alignment = compute_alignment(
        truth["reference"], base, truth["visible_reference"]) if base else \
        alignment_record(rotation_quarter_turns=0, translation=(0, 0, 0),
                         source="no_route_a_candidate_cells_identity_alignment")
    attribution = route_a_attribution(
        frontier=frontier, visible_reference=truth["visible_reference"],
        reference=truth["reference"], alignment=alignment,
        facade=facade_states or None, palette=palette_states or None,
        palette_terminal=terminal_states or None,
        include_palette_loss_partition=include_palette_loss_partition)
    record = attribution.to_record()
    geometry_rows = {row["metric"]: row["metric_value"]
                     for section in record["sections"] if section["space"] == "G"
                     for row in section["rows"] if row["checkpoint"] == "facade_grid"}
    recall = geometry_rows["geometry_recall"]
    precision = geometry_rows["geometry_precision"]
    correct = recall["numerator"]
    precision_denominator = precision["denominator"]
    summary = {
        "producer": ATTRIBUTION_PRODUCER,
        "route": "route_a",
        "arm": arm,
        "normal_cli_parity_eligible": arm == "camera-projected",
        "candidate_seal": json.loads(outcome.candidate_seal)
        if getattr(outcome, "candidate_seal", "") else None,
        "attribution_available": True,
        "geometry_summary": {
            "correct_cells": correct,
            "visible_cells": recall["denominator"],
            "candidate_cells": len(facade_states),
            "missing_cells": None if correct is None else recall["denominator"] - correct,
            "phantom_cells": None if correct is None else precision_denominator - correct,
            "hidden_reference_hits": None if correct is None else len(facade_states) - precision_denominator,
            "recall": recall["value"], "precision": precision["value"],
            "f1": None if recall["value"] is None or precision["value"] is None else (
                2 * recall["value"] * precision["value"] / (recall["value"] + precision["value"])
                if recall["value"] + precision["value"] else 0.0),
        },
        "attribution": record,
        "attribution_sha256": attribution_digest(attribution),
        "alignment": alignment,
        "production_records": {
            "camera_solved": outcome.camera is not None,
            "camera_error": outcome.camera_error,
            "palette_size": len(palette_states),
            "facade_block_count": len(facade_states),
            "decoded_block_count": len(terminal_states),
            "facade_digest": _volume_digest(facade_states),
            "decoded_digest": _volume_digest(terminal_states),
            "export_round_trip_exact": (
                None if not facade_states or not terminal_states
                else facade_states == terminal_states),
        },
        "truth": _truth_summary(truth),
        "caveats": list(attribution.caveats) + list(truth.get("caveats", ())),
        "cli_parity": _route_a_cli_parity(image, outcome, directory)
        if cli_parity else None,
        "notes": ("visible-denominator attribution for Route A; unknown metric "
                  "scale is a score-reducing caveat, never an unavailable geometry"),
    }
    return summary


def _volume_digest(states: dict) -> str:
    payload = [[list(coordinate), state.to_dict()]
               for coordinate, state in sorted(states.items())]
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _route_a_cli_parity(image: Path, outcome, directory: Path) -> dict:
    """Compare the runner's decoded export with the normal CLI's decoded export.

    The CLI is invoked as a subprocess with inherited stdio (the only approved
    launcher-free form here); the comparison itself is over the decoded
    ``.litematic`` coordinate/state sets under the one permitted global integer
    translation, exactly as A1's frozen parity check does.
    """
    if getattr(getattr(outcome, "settings", None), "arm", "camera-projected") != "camera-projected":
        raise AttributionRefusal("pixel-grid ablation is not normal CLI parity")
    import subprocess

    cli_dir = directory / "cli-parity"
    cli_dir.mkdir(parents=True, exist_ok=True)
    project = cli_dir / f"{image.stem}.mcscan"
    litematic = cli_dir / f"{image.stem}.litematic"
    command = [sys.executable, "-m", "mc_photogrammetry", "reconstruct",
               str(image), str(litematic), "--project", str(project)]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False,
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    runner_states = _states_of(outcome.decoded_export)
    if completed.returncode != 0 or not litematic.is_file():
        return {"available": False, "reason": "cli_reconstruct_failed",
                "exit_code": int(completed.returncode),
                "command": command}
    cli_states = {coordinate: state
                  for coordinate, state in load_litematic(litematic).blocks.items()}
    # Normalize only each exporter's region origin: the one permitted global
    # integer translation.  No scale or per-cell transform is applied.
    offset = _translation_offset(cli_states, runner_states)
    cli_norm = {_translate(coordinate, offset): state
                for coordinate, state in cli_states.items()}
    common = set(cli_norm) & set(runner_states)
    return {
        "available": True,
        "exit_code": int(completed.returncode),
        "command": command,
        "cli_blocks": len(cli_norm),
        "runner_blocks": len(runner_states),
        "region_origin_offset": list(offset),
        "normalized_sets_equal": cli_norm == runner_states,
        "only_cli": len(set(cli_norm) - set(runner_states)),
        "only_runner": len(set(runner_states) - set(cli_norm)),
        "state_mismatches": sum(1 for cell in common
                                if cli_norm[cell] != runner_states[cell]),
        "cli_volume_sha256": _volume_digest(cli_norm),
        "runner_volume_sha256": _volume_digest(runner_states),
    }


def _translate(coordinate, offset):
    return tuple(coordinate[index] + offset[index] for index in range(3))


def _translation_offset(left: dict, right: dict) -> tuple[int, int, int]:
    if not left or not right:
        return (0, 0, 0)
    left_min = tuple(min(coordinate[axis] for coordinate in left) for axis in range(3))
    right_min = tuple(min(coordinate[axis] for coordinate in right) for axis in range(3))
    return tuple(left_min[axis] - right_min[axis] for axis in range(3))


# ---------------------------------------------------------------------------
# Route B
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Route B entry-match evidence and the evaluator pair-correspondence CONTROL
# ---------------------------------------------------------------------------

def route_b_evaluator_pair_control(truth: dict) -> dict:
    """Aggregate the accepted evaluator's own truth-pair pool as a *control* only.

    This is **not** a Route B achieved P metric.  It runs a different matcher
    (``evaluation_pair_correspondence.evaluate_scene`` /
    ``select_mutual_matches``) from the entry's
    ``joint_matching.match_and_solve_pair``, so its numbers are reported under a
    separate ``evaluator_pair_control`` diagnostic that is explicitly excluded
    from the Route B attribution and loss funnel.  The Route B entry row is scored
    only from a digest-bound joint match set (see
    :func:`mc_photogrammetry.evaluation_image_only_attribution.pair_metric_rows`);
    since ``RelativeCameraEvidence`` retains no pixel/keypoint-pair identities,
    this runner reports B0 P as typed unavailable with
    ``entry_match_identities_not_retained_or_digest_unbound``.

    A missing or unusable frozen coverage yields a typed reason and ``totals:
    None``, never a fabricated zero or a connectivity substitute.
    """
    cameras = truth.get("cameras") or {}
    resolved = truth.get("frame_paths") or {}
    reference = truth.get("reference") or {}
    scene_transform = truth.get("scene_transform")
    if scene_transform is None:
        raise AttributionRefusal(
            "the frozen truth carries no evaluator scene transform for the P universe")
    names = sorted(resolved)
    if len(names) < 2:
        return {"totals": None, "reason": "entry_pair_frames_below_two",
                "coverage": {"frames": len(names)}}
    missing = [name for name in names if name not in cameras]
    if missing:
        return {"totals": None, "reason": "entry_pair_camera_missing",
                "coverage": {"frames": len(names), "missing_camera": missing[0]}}
    if not reference:
        return {"totals": None, "reason": "entry_pair_reference_empty",
                "coverage": {"frames": len(names)}}

    from mc_photogrammetry.evaluation_pair_correspondence import (
        DatasetSceneInput,
        FinalizedScene,
        FrameInput,
        evaluate_scene,
    )

    frame_inputs: list[FrameInput] = []
    for name in names:
        try:
            size = _frame_size(Path(resolved[name]))
        except (OSError, ValueError) as error:
            return {"totals": None, "reason": "entry_pair_frame_unreadable",
                    "coverage": {"frames": len(names), "detail": str(error)}}
        frame_inputs.append(FrameInput(
            frame_id=name, filename=name, camera=cameras[name],
            width=int(size[0]), height=int(size[1]),
            screenshot_path=str(resolved[name]), screenshot_sha256=""))
    # The evaluator cameras live in the scene's world frame, and
    # ``evaluate_scene`` raycasts a world-frame occupancy.  The reference
    # litematic is local-frame, so it is mapped through the same evaluator scene
    # transform ``visible_cells`` uses - otherwise every ray would miss and report
    # a false "no truth pairs" while the visible denominator (which does apply the
    # transform) is non-empty.
    world_reference = {
        tuple(scene_transform.to_world(coordinate)): state
        for coordinate, state in reference.items()}
    coordinates = tuple(world_reference)
    occupancy = {coordinate: world_reference[coordinate] for coordinate in coordinates}
    region_min = tuple(min(cell[axis] for cell in coordinates) for axis in range(3))
    region_max = tuple(max(cell[axis] for cell in coordinates) for axis in range(3))
    scene_input = DatasetSceneInput(
        dataset_id="a2-frozen-route-b-entry",
        index_path=str(truth.get("scene_path", "")),
        index_sha256="", scene_id="pinned-reference-pair-coverage",
        scene_group="frozen-evaluator", split="evaluation",
        directory="", scene_root="", hashes=(), screenshots=())
    scene = FinalizedScene(
        input=scene_input, frames=tuple(frame_inputs), occupancy=occupancy,
        region_min=region_min, region_max=region_max)
    try:
        evaluation = evaluate_scene(scene)
    except ValueError as error:
        return {"totals": None, "reason": "entry_pair_evaluator_error",
                "coverage": {"frames": len(names), "detail": str(error)}}
    returned = truth_pairs = truth_pairs_returned = positive = 0
    for pair in evaluation.pairs:
        geometry = pair.geometry
        returned += int(geometry.returned)
        positive += int(geometry.positive)
        truth_pairs += int(geometry.truth_pairs)
        truth_pairs_returned += int(geometry.truth_pairs_returned)
    totals = {
        "pairs": len(evaluation.pairs), "returned": returned, "positive": positive,
        "truth_pairs": truth_pairs, "truth_pairs_returned": truth_pairs_returned,
    }
    reason = "" if truth_pairs else "no_truth_keypoint_pairs"
    return {
        "totals": totals, "reason": reason,
        "coverage": {
            "frames": len(evaluation.observations), "pairs": len(evaluation.pairs),
            "keypoints": sum(len(observation.points)
                             for observation in evaluation.observations),
            "keypoints_with_truth": sum(
                1 for observation in evaluation.observations
                for hit in observation.hits if hit is not None),
        },
    }


def entry_match_context(entry) -> dict:
    """The retained Route B entry match evidence, as context only.

    ``RelativeCameraEvidence`` retains ``match_count``, ``mutual_survivors``,
    ``matches_sha256`` and the evidence digest, but no pixel/keypoint-pair
    identities.  These counts and digests therefore cannot be scored as an
    achieved P row on their own: :func:`route_b_attribution` receives no
    digest-bound joint set and reports the entry row typed unavailable.
    """
    graph = getattr(entry, "graph", None)
    evidence = tuple(getattr(graph, "evidence", ()) or ())
    pairs = []
    for item in evidence:
        pairs.append({
            "left_id": str(getattr(item, "left_id", "")),
            "right_id": str(getattr(item, "right_id", "")),
            "status": str(getattr(item, "status", "")),
            "match_count": int(getattr(item, "match_count", 0) or 0),
            "mutual_survivors": int(getattr(item, "mutual_survivors", 0) or 0),
            "matches_sha256": str(getattr(item, "matches_sha256", "")),
            "evidence_sha256": str(getattr(item, "evidence_sha256", "") or ""),
        })
    return {
        "graph_status": str(getattr(graph, "status", "") or ""),
        "pair_count": len(pairs),
        "pairs": pairs,
        "match_count_total": sum(item["match_count"] for item in pairs),
        "matches_sha256": [item["matches_sha256"] for item in pairs],
        "identities_retained": False,
    }



def _persisted_envelope(target: Path):
    record = load_project(target).image_derived_production
    if record is None:
        raise AttributionRefusal(
            "the published project carries no schema-3 production envelope")
    return production_envelope_from_record(record)


def run_route_b(
    *, sources: list[Path], truth: dict | None = None, project_path: Path | None,
):
    """Run Route B production only and return ``(outcome, a2_project_dir)``.

    Production-only: like :func:`run_route_a` this never receives, reads or
    configures anything from the evaluator truth, so no reference, mask, camera or
    scene can influence reconstruction.  The scoring step is
    :func:`_score_route_b`, which runs only after the evaluator truth is loaded.
    When no ``project_path`` was supplied a private directory is created and
    returned for the caller to remove after scoring.
    """
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise AttributionRefusal(f"the frozen source image is unavailable: {missing[0]}")
    if project_path is None:
        directory = Path(tempfile.mkdtemp(prefix="imgloss-a2-route-b-"))
        target = directory / "route-b.mcscan"
    else:
        directory = None
        target = Path(project_path)
        if target.exists():
            raise AttributionRefusal(f"the route B project target exists: {target}")
    outcome = _route_b_once(sources, target)
    return outcome, directory


def _route_b_once(sources, target: Path):
    """Execute the production entry once, before any evaluator data is opened."""
    outcome = derive_image_output_from_paths(sources, project_path=target)
    return outcome


def _score_route_b(*, outcome, truth: dict) -> dict:
    """Score the already-completed Route B production outcome against the truth."""
    if isinstance(outcome, plane_placement.PlacementRefusal):
        return {
            "producer": ATTRIBUTION_PRODUCER, "route": "route_b",
            "attribution_available": False,
            "reason": "placement_refusal",
            "refusal_reason": str(getattr(outcome, "reason", "")),
            "refusal_sha256": str(getattr(outcome, "refusal_sha256", "")),
            "attribution": None,
            "truth": _truth_summary(truth),
            "notes": "a placement refusal is a policy outcome, never accuracy",
        }
    if not isinstance(outcome, ImageDerivedProductionResult):
        raise AttributionRefusal("the production entry returned an unknown result type")
    target = Path(str(getattr(outcome.entry, "project_path", "")))
    frontier = route_b_frontier(
        entry=outcome.entry, lattice=outcome.lattice, geometry=outcome.geometry,
        output=outcome.output, result=outcome)
    envelope = _persisted_envelope(target)
    if (envelope.stage != outcome.stage or envelope.status != outcome.status
            or envelope.reason != outcome.reason
            or envelope.entry_sha256 != outcome.entry_sha256
            or envelope.lattice_sha256 != outcome.lattice_sha256
            or envelope.geometry_sha256 != outcome.geometry_sha256
            or envelope.output_sha256 != outcome.output_sha256):
        raise AttributionRefusal(
            "the persisted schema-3 envelope does not replay the returned result")

    palette_states = None
    placement_evidence = None
    volume_evidence = None
    promoted_states = {}
    promotion_record = None
    proposals = {}
    if outcome.output is not None:
        # Real production records: ``output.placement`` is a
        # ``ResolvedFacePlacement``, ``output.volume`` a ``PlaneVolumeResult`` and
        # ``output.promotion`` a ``PromotionResult``.  Placement and volume are
        # geometry-only coordinate evidence (no block state), so they are carried as
        # ``GeometryCellEvidence`` coordinate records - never as ``{coordinate:
        # None}`` state maps, which the state-rotating aligner cannot consume.
        # Promotion is the only label-bearing tail record and is read through the
        # real ``PromotedBlock.state()`` API.  Nothing is read from the fabricated
        # ``geometry.placement_records["axis_support"]`` field the old adapter
        # guessed at.
        palette_states = _palette_states(outcome.output.policy.palette)
        placement_evidence = _placement_coordinates(outcome.output.placement)
        volume_evidence = _volume_coordinates(outcome.output.volume)
        promotion_record = outcome.output.promotion
        promoted_states = _promoted_states(promotion_record)
        proposals = _proposal_records(outcome.output.proposals)
    alignment = _measured_route_b_alignment(
        reference=truth["reference"], visible_reference=truth["visible_reference"],
        placement_evidence=placement_evidence, volume_evidence=volume_evidence,
        promoted_states=promoted_states, proposals=proposals)
    # Candidate/proposal records stay in the run's own frame: the pure builder
    # applies the one permitted alignment exactly once (to every cell map and to
    # each proposal's coordinate and ranked states).  Pre-aligning them here would
    # transform them twice.  The measured alignment record above is retained for
    # scoring/reporting only.
    # The reachable entry/B0 checkpoint may only be scored from a digest-bound
    # joint match set.  RelativeCameraEvidence retains match counts and
    # matches_sha256 but no pixel/keypoint-pair identities, and this runner is not
    # permitted another Route B run, so no bit-identical joint match replay can be
    # proven here: the entry row is typed unavailable.  The evaluator's own
    # mutual-match aggregate is a separate control, never an achieved Route B P
    # metric.
    # Promotion-record presence is explicit and never inferred from mapping
    # truthiness: a carried promotion with zero cells proves the state-promotion
    # substage ran and must reach the builder as an empty mapping
    # (``checkpoint_reached: true`` beside an unavailable metric), while only
    # ``promotion is None`` is an absent record (``checkpoint_reached: null``).
    # ``promoted_states or None`` collapsed both cases at the adapter boundary and
    # erased the reach evidence.
    match_context = entry_match_context(outcome.entry)
    evaluator_control = route_b_evaluator_pair_control(truth)
    entry_binding = {"entry_reachable": any(
        item.stage == "entry" and item.reachable for item in frontier.checkpoints)}
    attribution = route_b_attribution(
        frontier=frontier, visible_reference=truth["visible_reference"],
        reference=truth["reference"], alignment=alignment,
        placement=placement_evidence,
        volume_cells=volume_evidence,
        promoted_cells=(promoted_states if promotion_record is not None else None),
        palette=palette_states, proposals=proposals or None,
        entry_pair_evidence=None,
        entry_pair_unavailable_reason=ENTRY_MATCH_UNBOUND_REASON)
    summary = {
        "producer": ATTRIBUTION_PRODUCER, "route": "route_b",
        "attribution_available": True,
        "attribution": attribution.to_record(),
        "attribution_sha256": attribution_digest(attribution),
        "alignment": alignment,
        "entry_pair_attribution": {
            "space": "P", "universe": "frozen_pixel_pair_identities",
            "checkpoint": "entry",
            "available": False,
            "metric": None,
            "reason": ENTRY_MATCH_UNBOUND_REASON,
            "entry_binding": entry_binding,
            "retained_match_evidence": match_context,
            "truth_pair_coverage": evaluator_control["coverage"],
            "notes": ("the retained RelativeCameraEvidence carries match counts and "
                      "matches_sha256 but no pixel/keypoint-pair identities, so no "
                      "bit-identical rederived joint match set can be bound to the "
                      "accepted digest; the entry P row is typed unavailable and "
                      "contributes no zero.  Entry connectivity and |V| are never "
                      "substituted, and P is never added to or subtracted from G/O/C/E"),
        },
        "evaluator_pair_control": {
            "space": "control", "universe": "frozen_pixel_pair_identities",
            "source": ("evaluation_pair_correspondence.construct_truth_pairs / "
                       "evaluate_scene / select_mutual_matches over the pinned "
                       "reference occupancy, frozen frames and evaluator cameras"),
            "matcher": "evaluate_scene/select_mutual_matches",
            "role": "control",
            "excluded_from_route_b_attribution": True,
            "status": ("measured" if evaluator_control["totals"] is not None
                       else "unavailable"),
            "reason": evaluator_control["reason"],
            "totals": evaluator_control["totals"],
            "coverage": evaluator_control["coverage"],
            "notes": ("this is a different matcher from the Route B entry's "
                      "joint_matching.match_and_solve_pair, so it is never attached to "
                      "the entry row and never enters the Route B attribution or loss "
                      "funnel; it is retained only as a labelled correspondence control. "
                      "Its truth-pair denominator uses the evaluator's own usable-pixel "
                      "mask (usable_mask_for), not the tracked visible-cell mask, and is "
                      "never divided by |V| or compared with |V|"),
        },
        "terminal": {"stage": envelope.stage, "status": envelope.status,
                     "reason": envelope.reason, "detail": envelope.detail},
        "envelope_sha256": envelope.record_sha256,
        "result_sha256": outcome.result_sha256,
        "entry_sha256": envelope.entry_sha256,
        "lattice_sha256": envelope.lattice_sha256,
        "geometry_sha256": envelope.geometry_sha256,
        "output_sha256": envelope.output_sha256,
        "session_sha256": envelope.session_sha256,
        "envelope_replays_result": True,
        "production_records": {
            "placement_cells": 0 if placement_evidence is None
                               else len(placement_evidence.cells),
            "volume_cells": 0 if volume_evidence is None
                            else len(volume_evidence.cells),
            "promoted_cells": len(promoted_states),
            "proposal_cells": len(proposals),
        },
        "truth": _truth_summary(truth),
        "caveats": list(attribution.caveats) + list(truth.get("caveats", ())) + [
            "the Route B placement and volume records are geometry-only: their "
            "coordinates are scored in G, but they carry no block state, so no "
            "exact-state (E) metric is derived from them and no label is invented",
        ],
        "notes": ("reachable Route B checkpoints only; a stage the run never "
                  "reached is typed unavailable and contributes no zero"),
    }
    return summary


def _measured_route_b_alignment(*, reference, visible_reference,
                                placement_evidence, volume_evidence,
                                promoted_states=None, proposals=None) -> dict | None:
    """The one measured alignment for the run-native Route B coordinate records.

    The accepted alignment search is run over the run's actual placement/volume
    coordinates through the pure coordinate-only
    ``best_visible_alignment_for_coordinates`` (same integer translation + lattice
    Y-quarter-turn family) and is applied exactly once by the pure builder.  A
    non-identity production record therefore always carries **measured** alignment
    evidence.  There is deliberately no identity fallback: when no production
    coordinate record was carried there is nothing to align, and silently
    substituting an identity record (the old
    ``no_route_b_candidate_cells_identity_alignment`` path) could mis-align label
    evidence. If no coordinate-bearing record exists, alignment is not applicable:
    return ``None`` and let the builder preserve typed-unavailable stage rows. A
    carried candidate record without a measured transform remains a typed refusal.
    """
    cells = None
    if volume_evidence is not None and volume_evidence.cells:
        cells = volume_evidence.cells
    elif placement_evidence is not None and placement_evidence.cells:
        cells = placement_evidence.cells
    elif promoted_states:
        cells = promoted_states.keys()
    elif proposals:
        cells = proposals.keys()
    if not cells:
        # No coordinate-bearing record means there is no transform to estimate.
        # Preserve typed-unavailable stages; do not invent identity alignment.
        return None
    turns, translation = best_visible_alignment_for_coordinates(
        reference, cells, visible_reference.visible)
    return alignment_record(
        rotation_quarter_turns=int(turns),
        translation=[int(value) for value in translation],
        source="evaluation_image_only_attribution.best_visible_alignment_for_coordinates")


def _placed_cell_coordinate(face, cell) -> tuple[int, int, int] | None:
    """One ``PlacedCell``'s world cell coordinate from its own face evidence.

    The accepted placement record stores each supported cell as its canonical
    ``(row, column)`` plus the exact integer lower endpoints ``lateral_u`` and
    ``lateral_v`` on the face's two tangent world axes
    (``reconstruction/plane_placement.py``).  The normal-axis world cell is the
    face's own ``normal_block``, exactly as ``build_plane_volume`` derives the
    volume coordinate from the same face
    (``reconstruction/plane_volume.py``).  No label is invented here: a placement
    cell genuinely has no block state, so the adapter returns a bare coordinate.
    """
    normal_axis = getattr(face, "world_normal_axis", None)
    normal_block = getattr(face, "normal_block", None)
    mappings = tuple(getattr(face, "tangent_mappings", ()) or ())
    lateral_u = getattr(cell, "lateral_u", None)
    lateral_v = getattr(cell, "lateral_v", None)
    if normal_axis not in (0, 1, 2) or type(normal_block) is not int \
            or len(mappings) != 2 or type(lateral_u) is not int \
            or type(lateral_v) is not int:
        return None
    u_axis = getattr(mappings[0], "world_axis", None)
    v_axis = getattr(mappings[1], "world_axis", None)
    if u_axis not in (0, 1, 2) or v_axis not in (0, 1, 2) \
            or len({normal_axis, u_axis, v_axis}) != 3:
        return None
    coordinate = [0, 0, 0]
    coordinate[normal_axis] = normal_block
    coordinate[u_axis] = lateral_u
    coordinate[v_axis] = lateral_v
    return tuple(coordinate)


def _placement_coordinates(placement):
    """The real ``ResolvedFacePlacement`` cells as coordinate-only geometry evidence.

    The actual Route B placement record is ``output.placement``, a
    ``ResolvedFacePlacement`` whose ``faces`` carry ``ResolvedFace.cells``
    ``PlacedCell`` records (``reconstruction/plane_placement.py``).  Placement is
    geometry-only: it carries exact coordinates but no state, so the adapter returns
    a :class:`GeometryCellEvidence` coordinate record (never a ``{coordinate:
    None}`` state map, which the state-rotating aligner cannot consume).  An absent
    record returns ``None``; a carried record with no faces returns an empty
    coordinate record, which the builder reports as typed unavailable.  The old
    adapter read a nonexistent
    ``geometry.placement_records["axis_support"].placements`` field and therefore
    always returned no evidence, silently discarding the real G geometry.
    """
    if placement is None:
        return None
    cells = set()
    for face in getattr(placement, "faces", ()) or ():
        for cell in getattr(face, "cells", ()) or ():
            coordinate = _placed_cell_coordinate(face, cell)
            if coordinate is None:
                raise AttributionRefusal(
                    "a placement cell's coordinate cannot be resolved from its own "
                    "face evidence (placement_cell_coordinate_unavailable)")
            cells.add(coordinate)
    return GeometryCellEvidence(cells=frozenset(cells), source="output.placement")


def _volume_coordinates(volume):
    """The real ``PlaneVolumeResult`` cells as coordinate-only geometry evidence.

    ``output.volume.cells`` are ``SupportedCellObservation`` records; each carries
    its exact ``coordinate`` but no block state (``reconstruction/plane_volume.py``),
    so the adapter returns a :class:`GeometryCellEvidence` coordinate record.  The
    former unconditional ``return {}`` dropped every real volume coordinate, so the
    ``volume_cell`` checkpoint never scored.  An absent volume record returns
    ``None``.
    """
    if volume is None:
        return None
    return GeometryCellEvidence(
        cells=frozenset(cell.coordinate
                        for cell in getattr(volume, "cells", ()) or ()),
        source="output.volume")


def _promoted_states(promotion) -> dict:
    """The real ``PromotionResult.cells`` through the ``PromotedBlock.state()`` API.

    ``reconstruction/state_promotion.py`` exposes ``PromotedBlock.state()``, not
    the fabricated ``to_block_state()`` the old adapter called (which raised
    ``AttributeError`` on every genuine nonempty promotion and could lose the
    successful tail).  A carried promotion with zero cells returns ``{}``; the
    caller distinguishes that from an absent record by the record object itself.
    """
    if promotion is None:
        return {}
    return {cell.coordinate: cell.state()
            for cell in getattr(promotion, "cells", ()) or ()}


def _proposal_records(proposals) -> dict:
    if proposals is None:
        return {}
    cells = getattr(proposals, "proposals", None) or getattr(proposals, "cells", ())
    return {proposal.cell: proposal for proposal in cells}


def _truth_summary(truth: dict) -> dict:
    visible = truth["visible_reference"]
    return {
        "reference_sha256": truth["reference_sha256"],
        "reference_cells": len(truth["reference"]),
        "visible_cells": len(visible.visible),
        "minimum_pixels": visible.minimum_pixels,
        "mask_origin": truth["mask_origin"],
        "mask_convention": truth["mask_convention"],
        "mask_pinned_domains_checked": truth.get("mask_pinned_domains_checked"),
        "mask_pins_match": truth.get("mask_pins_match"),
        "reference_pin_matches": truth.get("reference_pin_matches"),
        "frame_count": len(truth["frames"]),
        "frames": truth["frames"],
        "caveats": list(truth.get("caveats", ())),
        "source": visible.source,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _mask_dir_for(fixture: Path) -> Path | None:
    """Prefer the tracked reviewed mask specifications, then the local PNGs."""
    tracked = REPO_ROOT / "docs" / "evaluation_masks" / fixture.name.replace(" ", "-")
    if tracked.is_dir():
        return tracked
    candidate = REPO_ROOT / ".masks" / fixture.name.replace(" ", "-")
    return candidate if candidate.is_dir() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A2 evaluation-only visible-denominator attribution runner")
    parser.add_argument("--route", choices=("route-a", "route-b"), required=True)
    parser.add_argument("--route-a-arm", choices=("camera-projected", "pixel-grid"),
                        default="camera-projected",
                        help="evaluation-only Route A geometry prior")
    parser.add_argument("--expected-visible-cells", type=int, default=None)
    parser.add_argument("--expected-reference-cells", type=int, default=None)
    parser.add_argument("--image", default=None,
                        help="Route A frozen input image")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Route B frozen input images (>= 2 frames)")
    parser.add_argument("--fixture", default=None,
                        help="fixture directory holding the frozen frames")
    parser.add_argument("--reference", required=True,
                        help="evaluator-only reference litematic")
    parser.add_argument("--scene", required=True, help="evaluator-only scene.json")
    parser.add_argument("--cameras", required=True, help="evaluator-only cameras.json")
    parser.add_argument("--masks", default=None, help="evaluator-only frozen mask dir")
    parser.add_argument("--pins", default=None,
                        help="reviewed evaluator fixture pin manifest")
    parser.add_argument("--fixture-role", default=None,
                        help="pinned fixture role (for example template-2)")
    parser.add_argument("--workdir", default=None,
                        help="output directory for the replay project/export")
    parser.add_argument("--project", default=None,
                        help="Route B target project (must not exist)")
    parser.add_argument("--report", default=None, help="JSON report output path")
    parser.add_argument("--cli-parity", action="store_true",
                        help="also run the normal CLI on the same frozen input")
    parser.add_argument("--require-pins", action="store_true", default=None,
                        help="fail closed on a frozen-mask domain/pin mismatch; "
                             "defaults to on whenever --pins is supplied")
    parser.add_argument("--allow-pin-mismatch", dest="require_pins",
                        action="store_false",
                        help="explicitly downgrade a mask-domain/pin mismatch to a "
                             "typed caveat (diagnostic runs only)")
    parser.add_argument("--include-palette-loss-partition", action="store_true",
                        help="append the reviewed five in-palette loss-partition "
                             "rows at the Route A palette_label checkpoint; this is "
                             "Route A only, and Route B is rejected with a typed "
                             "refusal because it has no palette_label checkpoint")
    return parser


def _frozen_fixture_inputs(args) -> tuple:
    """Resolve the frozen frame paths and the evaluator mask directory.

    This reads only CLI arguments and directory names; it opens no evaluator
    reference, scene, camera manifest or mask.
    """
    fixture = Path(args.fixture) if args.fixture else None
    if args.route == "route-a":
        if not args.image:
            raise AttributionRefusal("route-a needs --image")
        image = Path(args.image)
        fixture = fixture or image.parent
        frame_paths = {image.name: image}
    else:
        if not args.sources:
            raise AttributionRefusal("route-b needs --sources (>= 2 frames)")
        resolved = [Path(item) for item in args.sources]
        fixture = fixture or resolved[0].parent
        frame_paths = {path.name: path for path in resolved}
    masks = Path(args.masks) if args.masks else _mask_dir_for(fixture)
    return fixture, frame_paths, masks


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.include_palette_loss_partition and args.route != "route-a":
            # The reviewed partition is defined only at Route A's palette_label
            # exact-state checkpoint.  Route B owns no such checkpoint, so the
            # option is rejected with the documented typed reason instead of being
            # silently dropped or mistaken for supported Route B coverage.  This
            # check runs before production, so no reconstruction is started.
            raise AttributionRefusal(
                "--include-palette-loss-partition is Route A palette_label only: "
                + PALETTE_LOSS_PARTITION_ROUTE_B_REASON)
        if args.route_a_arm != "camera-projected" and (args.route != "route-a" or args.cli_parity):
            raise AttributionRefusal("pixel-grid requires route-a and cannot be normal CLI parity")
        fixture, frame_paths, masks = _frozen_fixture_inputs(args)
        # Production runs *before* any evaluator file is opened - the pin manifest,
        # reference, scene, cameras and masks are all deferred until after inference:
        # the reconstruction stage must never be configured by (or even preceded by)
        # evaluator truth.  The production entry is called without the truth argument;
        # the evaluator pins and truth are loaded and the attribution scored only
        # afterwards.
        route_b_directory = None
        if args.route == "route-a":
            outcome, route_directory = run_route_a(
                image=Path(args.image),
                workdir=Path(args.workdir) if args.workdir else None,
                arm=args.route_a_arm)
        else:
            outcome, route_b_directory = run_route_b(
                sources=[Path(item) for item in args.sources],
                project_path=Path(args.project) if args.project else None)
            route_directory = route_b_directory
        pins = None
        if args.pins is not None:
            if not args.fixture_role:
                raise AttributionRefusal("--pins requires --fixture-role")
            pins = load_pins(args.pins, args.fixture_role)
        truth = load_frozen_truth(
            fixture=fixture, reference=Path(args.reference),
            scene_path=Path(args.scene), cameras_path=Path(args.cameras),
            mask_dir=masks, filter_frames=set(frame_paths),
            pins=pins, frame_paths=frame_paths, require_pins=args.require_pins)
        if args.expected_visible_cells is not None and len(truth["visible_reference"].visible) != args.expected_visible_cells:
            raise AttributionRefusal("fixed visible-cell denominator mismatch")
        if args.expected_reference_cells is not None and len(truth["reference"]) != args.expected_reference_cells:
            raise AttributionRefusal("fixed reference-cell universe mismatch")
        if args.route == "route-a":
            summary = _score_route_a(
                image=Path(args.image), outcome=outcome, directory=route_directory,
                truth=truth, cli_parity=args.cli_parity,
                include_palette_loss_partition=args.include_palette_loss_partition)
        else:
            summary = _score_route_b(outcome=outcome, truth=truth)
    except (AttributionRefusal, FrontierRefusal, ProductionEnvelopeError,
            ValueError, OSError) as error:
        print(canonical_json({
            "producer": ATTRIBUTION_PRODUCER,
            "attribution_available": False,
            "reason": type(error).__name__,
            "detail": str(error),
            "attribution": None,
            "notes": "typed refusal; no attribution was built",
        }))
        return 2
    if args.report is not None:
        Path(args.report).write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
