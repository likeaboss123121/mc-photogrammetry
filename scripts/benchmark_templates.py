"""Run a reproducible, explicitly-scoped reconstruction diagnostic.

Fixture references, camera manifests, and scene bounds are never inferred to be
production inputs. This script labels supplied-camera and reference-assisted
runs as diagnostics; only a future image-only pipeline can qualify for the
production gate.

The report still computes the reviewed >=0.80 visible exact-state and >=0.90
visible geometry-F1 thresholds at the primary one-pixel domain, but a single run
never claims promotion: the gate is fail-closed unless the run declares a
fixture role whose expected identity is explicitly populated and reviewed in the
tracked :data:`EVALUATOR_PIN_MANIFEST_PATH`, uses frozen evaluator-owned masks and
is not reference-assisted. A folder name and a run's own recorded hashes are
observations, never pins, so an unreviewed role stays ineligible even when its
numbers look stable. ``aggregate_promotion_gate`` must see one passing, eligible,
reviewed-pin, identity-stable report for Template 2, Template 6 and the frozen
small-view Template 3 subset before any promotion claim, and it reports
per-fixture metric status only, never image-only production qualification.
Larger visibility thresholds (2/5/10 pixels) are reported beside the primary row,
and hidden cells carry separate inference credit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation import (
    best_visible_alignment,
    observation_kind_counts,
    visible_breakdowns,
    visible_metrics,
)
# The stagewise join must apply exactly the report's allowed alignment transform,
# including to state properties, so it reuses the evaluator's own helpers rather
# than re-deriving a second, potentially divergent, rotation.
from mc_photogrammetry.evaluation import _rotate_coordinate, _rotate_state, _translate
from mc_photogrammetry.export.compare import compare_litematics, load_litematic
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.export.visibility import full_cube_keys, shape_table, visible_cells
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.facade import DEFAULT_PALETTE
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.reconstruction.texture_calibration import calibrate_appearance
from mc_photogrammetry.vision.gui_mask import create_gui_mask


MODES = ("image-only", "supplied-camera", "reference-assisted")

ACCEPTANCE_MINIMUM_VISIBLE_EXACT_STATE_RECOVERY = 0.80
ACCEPTANCE_MINIMUM_VISIBLE_GEOMETRY_F1 = 0.90
PRIMARY_VISIBILITY_MINIMUM_PIXELS = 1
"""The reviewed primary domain: a cell reached by at least one usable pixel."""
REPORTED_VISIBILITY_MINIMUMS = (1, 2, 5, 10)
"""Larger thresholds are reported beside the primary row, never instead of it."""
REQUIRED_PROMOTION_FIXTURES = ("template-2", "template-6", "frozen-small-template-3")
"""Per AGENT_HANDOFF section 5: every role must pass before any promotion claim."""
FROZEN_SUBSETS_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "reports" / "M3-frozen-subsets.json"
)
"""Tracked frozen-subset definition with a digest pin per named subset."""
PINNED_FIXTURE_ROLES: dict[str, dict[str, str]] = {
    "template-2": {"fixture": "templates 2"},
    "template-6": {"fixture": "templates 6"},
    "frozen-small-template-3": {"fixture": "templates 3", "subset": "t3-sparse-8"},
}
"""Hash-pinned acceptance fixtures; any other fixture cannot be a gate candidate."""

EVALUATOR_PIN_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "reports" / "evaluator-fixture-pins.json"
)
"""Tracked expected-pin manifest.

Each declared role may carry the reviewed reference digest, the ordered
frame-name+content digests and the frozen usable-mask domain identities. Until a
reviewer explicitly populates and marks an entry, that role is unpinned.
"""

EVALUATOR_PIN_MANIFEST_SCHEMA = 1
REPORT_SCHEMA_VERSION = 3

STAGE_TRACE_SCHEMA = 1
"""Schema of the reconstruction's pipeline trace and of this module's join."""

STAGE_TRACE_AVAILABILITY = "availability"
STAGE_TRACE_LABEL = "label"

STAGE_TRACE_STAGES = (
    ("available_palette", STAGE_TRACE_AVAILABILITY,
     "the transformed reference state is present in the canonical palette"),
    ("support", STAGE_TRACE_AVAILABILITY,
     "the transformed reference state survived every global support-pruning round and the final palette"),
    ("candidate", STAGE_TRACE_AVAILABILITY,
     "the transformed reference state occurs in the cell's retained top-K ranked candidates"),
    ("raw_winner", STAGE_TRACE_LABEL,
     "the transformed reference state is the cell's raw pre-smoothing winner"),
    ("post_smoothing", STAGE_TRACE_LABEL,
     "the transformed reference state is the label adjacency smoothing settled on"),
    ("placed", STAGE_TRACE_LABEL,
     "the transformed reference state is what the exported volume placed"),
)
"""Ordered pipeline checkpoints, evaluated against the pinned visible denominator.

The three availability checkpoints and the three label checkpoints answer
different questions and are reported separately: availability asks whether the
pipeline could still have reached the reference state, a label asks whether it
actually chose it. They are *checkpoints*, not strictly monotone ceilings. A
cell's candidate set is fixed before adjacency smoothing, but smoothing may
adopt a neighbour's majority state outside it, so a later checkpoint can match
more cells than an earlier one; deltas are therefore reported in both
directions. The per-checkpoint counts are reference-*aware* only here, in the
evaluator: the trace itself carries no reference label.
"""

STAGE_TRACE_CANDIDATE_AVAILABILITY_K = (8, 16, 32)
"""Candidate-depth checkpoints for the evaluator-only recall sweep (DS-R36).

Read post hoc from each cell's own ranked candidate list in the stage trace. K=8
is the production candidate cap, so it is always captured; K=16 and K=32 need a
trace whose ``stage_trace_candidate_limit`` reaches them, and the join reports
which checkpoints its captured list can actually evidence rather than guessing.
This is a candidate-availability ceiling, not a reconstruction setting: nothing
in the sweep can change the retained candidates, the winner or a gate.
"""

REQUIRED_PIN_FIELDS = ("reviewed", "fixture", "reference_sha256", "frames", "usable_masks")
"""Fields an expected-pin manifest entry must populate before a role can pin."""

GATE_REQUIRED_IDENTITY_FIELDS = (
    "fixture_role",
    "reference_sha256",
    "input_frames_sha256",
    "fixture_sha256",
    "fixture_pinned",
)
"""Identity fields an aggregate gate entry must carry; a missing one fails closed."""


def run(
    template_dir: Path,
    output: Path,
    *,
    mode: str,
    reference: Path | None,
    camera_manifest: Path | None,
    scene_definition: Path | None,
    jar: Path | None,
    adjacency_strength: float,
    report_path: Path | None = None,
    allow_default_palette: bool = False,
    bounded_adjacency: bool = False,
    palette_support: float | None = None,
    usable_mask_paths: dict[str, Path] | None = None,
    correct_light_cast: bool = raycast.CORRECT_LIGHT_CAST,
    fixture_role: str | None = None,
    stage_trace_path: Path | None = None,
    stage_trace_candidate_limit: int = raycast.STAGE_TRACE_CANDIDATE_LIMIT,
    stage_sidecar_path: Path | None = None,
) -> None:
    """Produce a diagnostic report without silently choosing fixture inputs.

    ``fixture_role`` names one of :data:`PINNED_FIXTURE_ROLES`. It is required
    for a run to be eligible for the per-run acceptance gate: without it, or
    when the reviewed expected pins in :data:`EVALUATOR_PIN_MANIFEST_PATH`, the
    folder, frame set or frozen-subset digest do not match this run, the run is
    reported but cannot pass.

    ``stage_trace_path`` enables the optional stagewise visible-state loss
    instrumentation for benchmark diagnostics only. It changes no inference
    input: the reconstruction records its own reference-free intermediate
    decisions, and this harness joins them to the reference, the visibility
    domain and the aligned candidate transform *after* reconstruction returns.

    ``palette_support`` is an explicit ablation control for the global
    support-pruning threshold. ``None`` (the default) passes the unchanged
    :data:`mc_photogrammetry.reconstruction.raycast.PALETTE_SUPPORT` value; any
    explicit numeric override must be finite and nonnegative (``>= 0``), is
    recorded in the effective settings and makes the run's acceptance gate
    ineligible as diagnostic-only, regardless of metrics. ``NaN``/``inf`` and
    negative overrides are rejected before any fixture input is read. It
    changes no inference default.

    ``stage_trace_candidate_limit`` widens only how many entries of each cell's
    already-computed cheapest-first candidate order the trace records, so an
    evaluator can read candidate availability at K = 16 or 32 as well as the
    production K = 8. It must be a positive integer, requires
    ``stage_trace_path`` (otherwise it is rejected rather than silently ignored)
    and changes no reconstruction decision, metric or gate.

    ``stage_sidecar_path`` is the separate, opt-in evaluation sidecar. It carries
    the production sampler's actual resolved-face payloads - both the raw sampler
    patch (the array E1 retains) and the scorer-facing ``observed_patch`` (the
    array the E2A trace digests) plus the valid/coverage masks - keyed to the
    schema-2 trace's observation ordinals and bound to that trace's exact
    file-content digest. It requires ``stage_trace_path``, must name a different
    and not-yet-existing file, and is refused (before either the trace or the
    sidecar is written) when it would exceed the fixed record or byte bound. Off by
    default; it changes no inference input, metric, decision or gate.
    ``stage_trace`` keeps its exact schema-2 content whether or not the sidecar is
    requested.
    """
    if mode not in MODES:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    if fixture_role is not None and fixture_role not in PINNED_FIXTURE_ROLES:
        raise ValueError(f"unsupported fixture role: {fixture_role}")
    # Validate the diagnostic trace depth before touching any fixture input, and
    # reject it outright when no trace will be written so it can never look like
    # it changed a run's diagnostics while doing nothing.
    _validated_stage_trace_candidate_limit(stage_trace_candidate_limit)
    if stage_trace_path is None and stage_trace_candidate_limit != raycast.STAGE_TRACE_CANDIDATE_LIMIT:
        raise ValueError(
            "stage_trace_candidate_limit requires stage_trace_path; "
            "a trace depth without a trace would be silently ignored"
        )
    # The sidecar is keyed to the trace's observation ordinals, so it is refused
    # without one. Its destination must be distinct from the trace and must not
    # already exist; both are checked here, before any fixture input is read, so
    # a refusal can never leave a destination or temporary artifact behind.
    if stage_sidecar_path is not None:
        if stage_trace_path is None:
            raise ValueError(
                "stage_sidecar_path requires stage_trace_path; the sidecar is keyed "
                "to the schema-2 trace's observation ordinals"
            )
        if _paths_alias(stage_sidecar_path, stage_trace_path):
            raise ValueError(
                "stage_sidecar_path and stage_trace_path must be different files"
            )
        if stage_sidecar_path.exists():
            raise FileExistsError(
                "stage_sidecar_path already exists and will not be overwritten: "
                f"{stage_sidecar_path}"
            )
    # Validate the diagnostic override before touching any fixture input. A NaN
    # or negative threshold silently disables support pruning in the
    # reconstruction layer (``palette_support > 0`` is false) and can leak
    # non-standard ``NaN``/``Infinity`` JSON, so fail closed here for direct
    # ``run()`` callers as well as the CLI.
    if palette_support is not None:
        _validated_palette_support(palette_support)
    if mode == "image-only":
        raise SystemExit(
            "image-only benchmarking is unavailable: the current scene-surface reconstructor "
            "requires supplied camera and scene inputs. Do not relabel this diagnostic."
        )
    if reference is None:
        raise SystemExit("--reference is required for a scored diagnostic; no reference is selected implicitly.")
    if camera_manifest is None or scene_definition is None:
        raise SystemExit("--camera-manifest and --scene are required for a supplied-camera diagnostic.")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if mode == "reference-assisted" and jar is None:
        raise SystemExit("reference-assisted mode needs --minecraft-jar to resolve the reference palette.")

    tracemalloc.start()
    started = time.perf_counter()
    input_started = time.perf_counter()
    frames_info = list(discover_media([template_dir]).image_frames)
    cameras = load_camera_manifest(camera_manifest)
    scene = load_scene_transform(scene_definition)
    frames = {Path(frame.source_path).name: load_image(frame.source_path) for frame in frames_info}
    missing_cameras = sorted(set(frames) - set(cameras))
    if missing_cameras:
        raise ValueError(f"camera manifest lacks image frames: {', '.join(missing_cameras)}")
    usable_masks = _load_usable_masks(usable_mask_paths, frames)
    mask_origin = "frozen" if usable_masks is not None else "derived"
    if usable_masks is None:
        # Policy C: with no frozen masks the evaluator must score the same
        # production GUI domain the reconstructor already samples, so derive it
        # once here and share that exact mapping with both consumers.
        usable_masks = {
            name: ~create_gui_mask(image.shape[0], image.shape[1])
            for name, image in frames.items()
        }
    usable_mask_policy = (
        "frozen supplied masks" if mask_origin == "frozen"
        else "derived production GUI mask (provisional)"
    )
    frame_provenance = {
        name: _file_provenance(Path(info.source_path))
        for name, info in zip(frames, frames_info)
    }
    reference_provenance = _file_provenance(reference)
    mask_domain_sha256 = {
        name: _mask_domain_sha256(mask) for name, mask in sorted(usable_masks.items())
    }
    mask_file_provenance = {
        name: _file_provenance(path)
        for name, path in sorted((usable_mask_paths or {}).items())
    }
    fixture_identity = _fixture_identity(
        fixture_role, template_dir, frame_provenance,
        str(reference_provenance.get("sha256", "")),
        mask_domain_sha256=mask_domain_sha256,
        mask_file_sha256={
            name: str(entry.get("sha256", ""))
            for name, entry in mask_file_provenance.items()
        },
    )
    input_seconds = time.perf_counter() - input_started

    palette = DEFAULT_PALETTE
    palette_source = "default"
    texture_source = None
    provider: VanillaJarTextureProvider | None = None
    palette_started = time.perf_counter()
    if jar is not None:
        provider = VanillaJarTextureProvider()
        if provider.load_jar(jar):
            palette = provider.canonical_palette()
            palette_source = "jar"
            texture_source = provider.source
    if not allow_default_palette and palette_source == "default":
        raise SystemExit(f"Palette resolved to DEFAULT_PALETTE for {template_dir}; pass --allow-default-palette to continue.")
    palette_seconds = time.perf_counter() - palette_started

    reference_started = time.perf_counter()
    reference_snapshot = load_litematic(reference)
    calibrations = None
    if mode == "reference-assisted":
        if provider is None:
            raise SystemExit("reference-assisted mode needs a loaded Minecraft jar palette.")
        # Keep the exact reference states and all baked renderer metadata. The
        # old path rebuilt two-field PaletteColor values keyed by name, which
        # discarded partial shape boxes, face polygons, and alpha masks and also
        # collapsed distinct state properties under one block name.
        palette = provider.palette_for_states(
            tuple(reference_snapshot.palette), include_partial=False
        )
        palette_source = "reference"
        texture_colors = {
            item.state: tuple(item.rgb)
            for item in palette
        }
        calibrations = {
            name: calibrate_appearance(frames[name], reference_snapshot, cameras[name], scene, texture_colors)
            for name in frames
        }
    reference_seconds = time.perf_counter() - reference_started

    diagnostics: dict[str, object] = {}
    # ``None`` must forward the existing module default byte-for-byte; only an
    # explicit override replaces it, and that override is diagnostic-only.
    effective_palette_support = (
        raycast.PALETTE_SUPPORT if palette_support is None
        else _validated_palette_support(palette_support)
    )
    stage_trace: dict[str, object] | None = {} if stage_trace_path is not None else None
    # The sidecar's capture buffer is allocated only when the caller opted in;
    # ``None`` keeps the ordinary traced and untraced paths free of any payload
    # allocation or base64 encoding.
    stage_sidecar: dict[str, object] | None = {} if stage_sidecar_path is not None else None
    reconstruction_started = time.perf_counter()
    volume = reconstruct_scene_surfaces(
        frames, cameras, scene, palette=palette, appearance_calibrations=calibrations,
        adjacency_strength=adjacency_strength, bounded_adjacency=bounded_adjacency, diagnostics=diagnostics,
        correct_light_cast=correct_light_cast,
        usable_masks=usable_masks,
        palette_support=effective_palette_support,
        stage_trace=stage_trace,
        stage_trace_candidate_limit=stage_trace_candidate_limit,
        stage_sidecar=stage_sidecar,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    export_started = time.perf_counter()
    LitematicExporter().export(volume, output, name=template_dir.name)
    export_seconds = time.perf_counter() - export_started
    compare_started = time.perf_counter()
    comparison = compare_litematics(reference, output)
    compare_seconds = time.perf_counter() - compare_started
    resolutions = {name: (image.shape[1], image.shape[0]) for name, image in frames.items()}
    visibility_started = time.perf_counter()
    visibility = visible_cells(
        reference_snapshot.blocks, scene, cameras, resolutions,
        full_cubes=full_cube_keys(provider.samples) if provider else None,
        shapes=shape_table(provider.samples) if provider else None,
        palette=provider.samples if provider else None,
        usable_masks=usable_masks,
    )
    visibility_seconds = time.perf_counter() - visibility_started
    rotation_quarter_turns, visible_translation = best_visible_alignment(
        reference_snapshot.blocks, comparison.candidate.blocks, visibility,
    )
    visible = visible_metrics(
        reference_snapshot.blocks, comparison.candidate.blocks, visibility,
        translation=visible_translation, rotation_quarter_turns=rotation_quarter_turns,
    )
    visibility_rows = _visibility_threshold_rows(
        reference_snapshot.blocks, comparison.candidate.blocks, visibility,
        translation=visible_translation, rotation_quarter_turns=rotation_quarter_turns,
    )
    stagewise = _stagewise_visible_loss(
        reference_snapshot.blocks, comparison.candidate.blocks, visibility,
        stage_trace=stage_trace,
        translation=visible_translation, rotation_quarter_turns=rotation_quarter_turns,
        requested=stage_trace_path is not None,
    )
    acceptance_gate = _acceptance_gate(
        mode=mode, mask_origin=mask_origin, fixture_identity=fixture_identity,
        visible_exact_state_recovery=visible.exact_state_recovery,
        visible_geometry_f1=visible.geometry_f1,
        palette_support_override=palette_support,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Mode: {mode} (diagnostic; does not qualify for image-only production)")
    print(f"Reference: {reference}")
    print(f"Candidate: {output}")
    print(f"Blocks: {comparison.candidate.non_air_count}")
    print(f"Visible exact-state recovery: {visible.exact_state_recovery:.2%}")
    print(f"Visible geometry F1: {visible.geometry_f1:.2%} (precision {visible.geometry_precision:.2%}, recall {visible.geometry_recall:.2%})")
    print(f"Hidden-cell inference credit: {visible.hidden_inferred_blocks}/{visible.hidden_reference_blocks} "
          f"({visible.hidden_inference_recovery:.2%})")
    print(f"Acceptance gate (minimum_pixels={PRIMARY_VISIBILITY_MINIMUM_PIXELS}): "
          f"{'PASS' if acceptance_gate['passed'] else 'NOT PASSED'} "
          f"[eligible={acceptance_gate['eligible']}]")
    print("Gate scope: per-fixture metric status only; image-only production qualification is separate and not claimed.")
    print(f"Conditional common-coordinate state accuracy: {comparison.state_accuracy:.2%}")
    print(f"Whole-reference exact rate: {comparison.whole_reference_state_accuracy:.2%}")
    if stagewise.get("available"):
        summary = stagewise.get("summary") or {}
        print(f"Stagewise visible exact-state checkpoints (pinned denominator "
              f"{stagewise['visible_reference_blocks']}, minimum_pixels={PRIMARY_VISIBILITY_MINIMUM_PIXELS}):")
        for stage in stagewise["stages"]:
            print(f"  {stage['stage']} [{stage['kind']}]: "
                  f"{stage['exact_state_matches']}/{stage['visible_reference_blocks']} "
                  f"({stage['match_rate']:.2%})")
        print(f"  Availability matches: {summary.get('availability_matches')}; "
              f"label matches: {summary.get('label_matches')}")
        print(f"  Largest checkpoint loss: {summary.get('largest_stage_loss')} "
              f"({summary.get('largest_stage_loss_cells')} cells); "
              f"final geometry-missing={summary.get('geometry_missing_cells')} "
              f"matched-wrong-state={summary.get('matched_wrong_state_cells')}")
        availability = stagewise.get("candidate_availability")
        if isinstance(availability, dict):
            print(f"  Candidate availability by depth (trace captured "
                  f"{availability['captured_limit']} per cell; "
                  f"truth-in-palette denominator {availability['truth_in_palette_denominator']}, "
                  f"geometry-present denominator {availability['geometry_present_denominator']}):")
            for checkpoint in availability["checkpoints"]:
                if checkpoint["captured"]:
                    print(f"    K={checkpoint['k']}: "
                          f"{checkpoint['exact_state_available_in_palette']}"
                          f"/{checkpoint['truth_in_palette_denominator']} "
                          f"({checkpoint['exact_state_recall_in_palette']:.2%}) in-palette, "
                          f"{checkpoint['exact_state_available']}"
                          f"/{checkpoint['visible_denominator']} of visible")
                else:
                    print(f"    K={checkpoint['k']}: uncaptured "
                          f"({checkpoint['captured_limit_reason']})")
    else:
        print(f"Stagewise visible-state loss: unavailable ({stagewise.get('unavailable_reason')})")
    print(f"Reconstruct seconds: {reconstruction_seconds:.1f}")

    effective_settings = {
        f"mc_photogrammetry.reconstruction.raycast.{name}": _json_safe(getattr(raycast, name))
        for name in sorted(dir(raycast)) if name.isupper() and not name.startswith("_")
    }
    effective_settings.update(
        adjacency_strength=adjacency_strength,
        bounded_adjacency=bounded_adjacency,
        mode=mode,
        benchmark_script="scripts/benchmark_templates.py",
        visibility_renderer="shared alpha-tested model renderer" if provider else "analytic fallback",
        usable_mask_policy=usable_mask_policy,
        correct_light_cast=correct_light_cast,
        stage_trace=stage_trace_path is not None,
        stage_trace_candidate_limit=(
            stage_trace_candidate_limit if stage_trace_path is not None else None
        ),
        stage_sidecar=stage_sidecar_path is not None,
        stage_sidecar_records=(
            len(stage_sidecar["records"])  # type: ignore[arg-type]
            if stage_sidecar_path is not None and stage_sidecar
            and isinstance(stage_sidecar.get("records"), list)
            else None
        ),
        palette_support=effective_palette_support,
        palette_support_override=palette_support,
    )
    report = {
        "report_schema": REPORT_SCHEMA_VERSION,
        "source": _source_provenance(),
        "mode": mode,
        "qualifies_image_only_production": False,
        "qualification_reason": "This run consumes a supplied fixture camera manifest and scene bounds.",
        "qualification": {
            "image_only_production": False,
            "reason": "This run consumes a supplied fixture camera manifest and scene bounds.",
            "is_image_only_production_qualification": False,
            "per_fixture_metric_status": {
                "fixture_role": fixture_identity.get("role"),
                "eligible": acceptance_gate["eligible"],
                "metrics_met": acceptance_gate["metrics_met"],
                "per_fixture_passed": acceptance_gate["passed"],
                "fixture_pinned": acceptance_gate["fixture_pinned"],
            },
            "production_requirement": (
                "per-fixture metric status is not production qualification; image-only production "
                "additionally requires an image-only, mask-frozen, non-reference-assisted run plus one "
                "passing, reviewed-pin, identity-stable run for every required fixture in one aggregate"
            ),
        },
        "reference": reference_provenance,
        "camera_manifest": _file_provenance(camera_manifest),
        "scene_definition": _file_provenance(scene_definition),
        "fixture_identity": fixture_identity,
        "input_frames": frame_provenance,
        "usable_masks": {
            "policy": usable_mask_policy,
            "origin": mask_origin,
            "frames": mask_file_provenance,
            "usable_pixels": {
                name: int(mask.sum()) for name, mask in sorted(usable_masks.items())
            },
            "domain_sha256": mask_domain_sha256,
            "reproducible": True,
            "gate_requirement": "frozen evaluator-owned masks; a derived GUI mask is provisional and cannot pass",
        },
        "palette_source": palette_source,
        "texture_source": texture_source,
        "palette_size": len(palette),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "minecraft_jar": _file_provenance(jar) if jar else None,
        "visibility": {
            **visible.to_dict(), "pixel_counts_by_reference_cell": sum(visibility.values()),
            "mask_policy": usable_mask_policy,
            "minimum_pixels": PRIMARY_VISIBILITY_MINIMUM_PIXELS,
            "thresholds": visibility_rows,
            "breakdowns": visible_breakdowns(
                reference_snapshot.blocks, comparison.candidate.blocks, visibility,
                translation=visible_translation, rotation_quarter_turns=rotation_quarter_turns,
            ),
        },
        "sample_count": comparison.reference.non_air_count,
        "reference_blocks": comparison.reference.non_air_count,
        "candidate_blocks": comparison.candidate.non_air_count,
        "common_coordinates": comparison.exact_coordinate_matches,
        "coordinate_recall": comparison.coordinate_recall,
        "coordinate_precision": comparison.coordinate_precision,
        "coordinate_f1": comparison.coordinate_f1,
        "reference_exposed": comparison.reference_exposed,
        "surface_matches": comparison.surface_matches,
        "surface_precision": comparison.surface_precision,
        "surface_recall": comparison.surface_recall,
        "surface_f1": comparison.surface_f1,
        "exact_state_matches": comparison.exact_state_matches,
        "state_accuracy": comparison.state_accuracy,
        "whole_reference_state_accuracy": comparison.whole_reference_state_accuracy,
        "best_translation": list(comparison.best_translation),
        "evaluation_alignment": {
            "rotation_quarter_turns": rotation_quarter_turns,
            "translation": list(visible_translation),
            "coordinate_transform": "candidate (x,y,z) rotated about +Y, then translated; no reflection or scale fit",
            "state_transform": "cardinal facing/axis/rotation properties rotate with coordinates",
        },
        "direct_coordinate_matches": comparison.direct_coordinate_matches,
        "direct_coordinate_recall": comparison.direct_coordinate_recall,
        "direct_state_matches": comparison.direct_state_matches,
        "observation_kind_counts": observation_kind_counts(volume),
        "timings_seconds": {
            "input_loading": input_seconds,
            "palette_loading": palette_seconds,
            "reference_and_calibration": reference_seconds,
            "reconstruct": reconstruction_seconds,
            "export": export_seconds,
            "comparison": compare_seconds,
            "visibility": visibility_seconds,
            "total": time.perf_counter() - started,
        },
        "peak_python_heap_bytes": peak_bytes,
        "candidate": _file_provenance(output),
        "environment": {
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "dependencies": _dependency_versions(),
        },
        "effective_settings": effective_settings,
        "diagnostics": _json_safe(diagnostics),
        "top_confusion_pairs": [
            {"reference_state": ref_name, "candidate_state": cand_name, "count": count}
            for ref_name, cand_name, count in comparison.confusion_pairs
        ],
        "per_state_accuracy": comparison.per_state_accuracy,
        "per_state_counts": comparison.per_state_counts,
        "acceptance_gate": acceptance_gate,
        "stagewise_visible_loss": stagewise,
    }
    if stage_trace_path is not None and stage_trace is not None:
        # Reference-free trace, written beside the report. The report's join is
        # already computed, so this is an observational artifact only. Its exact
        # output bytes are preflighted before anything is written, so a sidecar
        # that exceeds its serialized bound refuses without leaving a trace.
        trace_text = json.dumps(_json_safe(stage_trace), indent=2, sort_keys=True) + "\n"
        trace_bytes = _trace_output_bytes(trace_text)
        if stage_sidecar_path is not None and stage_sidecar is not None:
            # Bind the sidecar to the exact bytes the trace is about to receive
            # and refuse an over-bound sidecar here, before either destination
            # exists, so a byte-bound refusal creates neither trace nor sidecar.
            stage_sidecar["trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
            stage_sidecar["trace_bytes"] = len(trace_bytes)
            sidecar_payload = _stage_sidecar_payload(stage_sidecar)
            stage_trace_path.parent.mkdir(parents=True, exist_ok=True)
            stage_trace_path.write_bytes(trace_bytes)
            _write_stage_sidecar(
                stage_sidecar_path, sidecar_payload, trace_path=stage_trace_path)
        else:
            # No sidecar: write the trace exactly as before. ``write_bytes``
            # receives the same bytes the old ``write_text`` produced (see
            # ``_trace_output_bytes``), so trace output bytes stay unchanged.
            stage_trace_path.parent.mkdir(parents=True, exist_ok=True)
            stage_trace_path.write_bytes(trace_bytes)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_identity(
    role: str | None,
    template_dir: Path,
    frame_provenance: dict[str, dict[str, object]],
    reference_sha256: str,
    *,
    mask_domain_sha256: dict[str, str] | None = None,
    mask_file_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    """Bind a run to a reviewed expected pin, or report it as unpinned.

    A role only counts as ``pinned`` when the tracked expected-pin manifest
    carries a reviewed reference digest, the ordered frame name+content digests
    and the frozen usable-mask domain identities, and every one of them matches
    this run. The fixture folder name and the run's own recorded hashes are
    observations, never pins, so an unpopulated or drifting manifest leaves the
    role unpinned and its per-run gate ineligible. The frozen-subset digest check
    stays an additional constraint rather than a substitute for content pins.
    """
    frame_names = sorted(frame_provenance)
    observed_frames = [
        {"name": name, "sha256": str(frame_provenance[name].get("sha256", ""))}
        for name in frame_names
    ]
    observed_mask_domains = {
        name: str(value) for name, value in sorted((mask_domain_sha256 or {}).items())
    }
    observed_mask_files = {
        name: str(value) for name, value in sorted((mask_file_sha256 or {}).items())
    }
    identity: dict[str, object] = {
        "role": role,
        "frames": frame_names,
        "input_frames_sha256": _ordered_frames_sha256(frame_provenance),
        "reference_sha256": reference_sha256,
        "fixture_sha256": _fixture_sha256(role, reference_sha256, frame_provenance),
        "observed_frames": observed_frames,
        "observed_usable_mask_domain_sha256": observed_mask_domains,
        "observed_usable_mask_file_sha256": observed_mask_files,
        "pin_manifest": str(EVALUATOR_PIN_MANIFEST_PATH),
        "pin_manifest_schema": EVALUATOR_PIN_MANIFEST_SCHEMA,
        "pinned": False,
        "missing_pin_fields": list(REQUIRED_PIN_FIELDS),
        "mismatched_pin_fields": [],
        "pin_reason": "no pinned acceptance fixture role was declared",
    }
    specification = PINNED_FIXTURE_ROLES.get(role) if role else None
    if specification is None:
        identity["fixture_present"] = role is not None
        return identity
    expected = _expected_pin_manifest().get(str(role))
    missing, mismatched = _expected_pin_problems(
        specification, expected, reference_sha256, observed_frames,
        observed_mask_domains, observed_mask_files,
    )
    subset = specification.get("subset")
    subset_pin = _frozen_subset_pin(subset) if subset else None
    frames_match = True if subset_pin is None else frame_names == sorted(subset_pin["frames"])
    identity.update({
        "fixture": specification["fixture"],
        "fixture_folder": template_dir.name,
        "fixture_matches": template_dir.name == specification["fixture"],
        "subset": subset,
        "subset_pin": subset_pin,
        "frames_match": frames_match,
        "missing_pin_fields": missing,
        "mismatched_pin_fields": mismatched,
        "pinned": not missing and not mismatched,
        "pin_reason": (
            "reviewed expected pins match this run"
            if not missing and not mismatched
            else "expected pins are missing or do not match this run"
        ),
    })
    return identity


def _expected_pin_manifest() -> dict[str, dict[str, object]]:
    """Load the tracked expected-pin manifest without touching fixture bytes.

    A missing, unreadable or malformed manifest pins nothing rather than
    silently trusting whatever the current run observed. ``manifest_schema``
    must be exactly an ``int`` equal to :data:`EVALUATOR_PIN_MANIFEST_SCHEMA`;
    JSON ``true`` (Python ``bool``) and ``1.0`` (float) compare equal to the
    integer but are rejected as untrusted.
    """
    path = EVALUATOR_PIN_MANIFEST_PATH
    if not path.is_file():
        return {}
    try:
        specification = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(specification, dict):
        return {}
    manifest_schema = specification.get("manifest_schema")
    if type(manifest_schema) is not int or manifest_schema != EVALUATOR_PIN_MANIFEST_SCHEMA:
        return {}
    roles = specification.get("roles")
    if not isinstance(roles, dict):
        return {}
    return {
        str(role): dict(entry)
        for role, entry in roles.items()
        if isinstance(entry, dict)
    }


def _expected_pin_problems(
    specification: dict[str, str],
    expected: dict[str, object] | None,
    reference_sha256: str,
    observed_frames: list[dict[str, str]],
    observed_mask_domains: dict[str, str],
    observed_mask_files: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return the missing and mismatched expected-pin fields for one role."""
    if expected is None:
        return list(REQUIRED_PIN_FIELDS), []
    missing: list[str] = []
    mismatched: list[str] = []
    if expected.get("reviewed") is not True:
        missing.append("reviewed")
    if str(expected.get("fixture") or "") != str(specification.get("fixture") or ""):
        missing.append("fixture")
    declared_subset = specification.get("subset")
    expected_subset = expected.get("subset")
    if declared_subset is None:
        if expected_subset not in (None, ""):
            mismatched.append("subset")
    elif expected_subset in (None, ""):
        missing.append("subset")
    elif str(expected_subset) != str(declared_subset):
        mismatched.append("subset")
    expected_reference = str(expected.get("reference_sha256") or "")
    if not _is_sha256(expected_reference):
        missing.append("reference_sha256")
    elif expected_reference != reference_sha256:
        mismatched.append("reference_sha256")
    expected_frames = _expected_frame_entries(expected.get("frames"))
    if expected_frames is None:
        missing.append("frames")
    elif expected_frames != observed_frames:
        mismatched.append("frames")
    expected_masks = _expected_mask_entries(expected.get("usable_masks"))
    if expected_masks is None:
        missing.append("usable_masks")
    elif not _expected_masks_match(expected_masks, observed_mask_domains, observed_mask_files):
        mismatched.append("usable_masks")
    return missing, mismatched


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _expected_frame_entries(value: object) -> list[dict[str, str]] | None:
    """Normalise the ordered expected ``[{name, sha256}]`` pin list, or ``None``."""
    if not isinstance(value, list) or not value:
        return None
    entries: list[dict[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name or name in names or not _is_sha256(digest):
            return None
        names.add(name)
        entries.append({"name": name, "sha256": str(digest)})
    return entries


def _expected_mask_entries(value: object) -> list[dict[str, str]] | None:
    """Normalise the expected frozen usable-mask identity list, or ``None``."""
    if not isinstance(value, list) or not value:
        return None
    entries: list[dict[str, str]] = []
    frames: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            return None
        frame = item.get("frame")
        domain = item.get("domain_sha256")
        if not isinstance(frame, str) or not frame or frame in frames or not _is_sha256(domain):
            return None
        entry = {"frame": frame, "domain_sha256": str(domain)}
        file_digest = item.get("file_sha256")
        if file_digest is not None:
            if not _is_sha256(file_digest):
                return None
            entry["file_sha256"] = str(file_digest)
        frames.add(frame)
        entries.append(entry)
    return entries


def _expected_masks_match(
    expected: list[dict[str, str]],
    observed_mask_domains: dict[str, str],
    observed_mask_files: dict[str, str],
) -> bool:
    if sorted(entry["frame"] for entry in expected) != sorted(observed_mask_domains):
        return False
    for entry in expected:
        frame = entry["frame"]
        if observed_mask_domains.get(frame) != entry["domain_sha256"]:
            return False
        file_digest = entry.get("file_sha256")
        if file_digest is not None and observed_mask_files.get(frame) != file_digest:
            return False
    return True


def _frozen_subset_pin(subset: str) -> dict[str, object]:
    """Recompute a named subset's digest and compare it with its recorded pin."""
    if not FROZEN_SUBSETS_PATH.is_file():
        raise FileNotFoundError(FROZEN_SUBSETS_PATH)
    specification = json.loads(FROZEN_SUBSETS_PATH.read_text(encoding="utf-8"))
    names = [str(name) for name in specification["subsets"][subset]]
    pinned = str(specification["digests"][subset])
    actual = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return {
        "subset": subset,
        "frames": names,
        "pinned_digest": pinned,
        "digest": actual,
        "digest_matches": actual == pinned,
    }


def _ordered_frames_sha256(frame_provenance: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(frame_provenance):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(frame_provenance[name].get("sha256", "")).encode("ascii") + b"\0")
    return digest.hexdigest()


def _fixture_sha256(
    role: str | None,
    reference_sha256: str,
    frame_provenance: dict[str, dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(role).encode("utf-8") + b"\0")
    digest.update(str(reference_sha256).encode("ascii") + b"\0")
    digest.update(_ordered_frames_sha256(frame_provenance).encode("ascii"))
    return digest.hexdigest()


def _mask_domain_sha256(mask: object) -> str:
    """Digest the exact scored boolean domain, not just the mask file bytes.

    That makes the usable-pixel denominator reproducible from the report even
    when the mask arrived as a PNG rather than a rectangle specification.
    """
    digest = hashlib.sha256()
    digest.update(str(getattr(mask, "shape", ())).encode("ascii") + b"\0")
    digest.update(mask.tobytes())  # type: ignore[attr-defined]
    return digest.hexdigest()


def _visibility_threshold_rows(
    reference,
    candidate,
    visible_pixels,
    *,
    translation,
    rotation_quarter_turns,
) -> list[dict[str, object]]:
    """Score the same aligned candidate at the primary and larger pixel domains."""
    rows: list[dict[str, object]] = []
    for minimum in REPORTED_VISIBILITY_MINIMUMS:
        counts = {coordinate: pixels for coordinate, pixels in visible_pixels.items()
                  if pixels >= minimum}
        metrics = visible_metrics(
            reference, candidate, counts,
            translation=translation, rotation_quarter_turns=rotation_quarter_turns,
        )
        rows.append({
            "minimum_pixels": minimum,
            "primary": minimum == PRIMARY_VISIBILITY_MINIMUM_PIXELS,
            **metrics.to_dict(),
        })
    return rows


def _state_from_record(record: object) -> BlockState | None:
    """Rehydrate one trace state record, or ``None`` when it is absent/malformed."""
    if not isinstance(record, dict):
        return None
    name = record.get("name")
    if not isinstance(name, str) or not name:
        return None
    properties = record.get("properties") or []
    pairs: list[tuple[str, str]] = []
    if isinstance(properties, list):
        for item in properties:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
    try:
        return BlockState(name, tuple(sorted(pairs)))
    except ValueError:
        return None


def _aligned_state(record: object, turns: int) -> BlockState | None:
    """Apply the report's permitted state transform to a trace state record."""
    state = _state_from_record(record)
    return None if state is None else _rotate_state(state, turns)


def _aligned_trace_coordinate(value: object, turns: int, translation) -> tuple | None:
    """Apply the report's permitted coordinate transform to a trace coordinate."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        coordinate = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None
    return _translate(_rotate_coordinate(coordinate, turns), translation)


def _rotated_state_records(records: object, turns: int) -> set[BlockState]:
    """Rehydrate a trace state-record list and rotate it into the reference frame."""
    if not isinstance(records, list):
        return set()
    states: set[BlockState] = set()
    for record in records:
        state = _aligned_state(record, turns)
        if state is not None:
            states.add(state)
    return states


def _support_states(stage_trace: dict, turns: int) -> set[BlockState] | None:
    """The states the global support pruning left, in the reference frame.

    The transform's direction is candidate -> reference, so every recorded state
    is rotated by ``turns`` before a reference state is looked up in it. The
    result is the intersection of every executed round's survivors with the
    final palette, because a state dropped by any round (or absent from the
    palette the pipeline settled on) is no longer reachable by any later stage.

    ``None`` is a defensive guard only: the join validates the trace structure
    before calling this helper, so a malformed round list never reaches it. An
    empty set means pruning was recorded and left nothing.
    """
    survivors: set[BlockState] | None = None
    for record in stage_trace.get("support_rounds", []):
        states = _rotated_state_records(record.get("states_kept"), turns)
        survivors = states if survivors is None else survivors & states
    final = stage_trace.get("final_palette")
    if isinstance(final, dict) and isinstance(final.get("states"), list):
        states = _rotated_state_records(final["states"], turns)
        survivors = states if survivors is None else survivors & states
    return survivors


def _is_trace_state_properties(value: object) -> bool:
    """Whether ``value`` is the property shape ``_state_from_record`` can rehydrate.

    The writer emits ``[[key, value], ...]`` (see ``_state_record``), so a
    non-list, a pair that is not exactly two items, an empty string member or a
    non-string member is malformed: ``_state_from_record`` would either drop it
    silently or coerce it with ``str()``, changing the rehydrated state.
    """
    if not isinstance(value, list):
        return False
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False
        key, property_value = pair
        if not isinstance(key, str) or not isinstance(property_value, str):
            return False
        if not key or not property_value:
            return False
    return True


def _is_trace_state_record(record: object) -> bool:
    """Whether ``record`` is exactly what ``_state_from_record`` can rehydrate.

    A name that :class:`BlockState` rejects cannot name a state, so it is a
    malformed membership entry rather than a state that merely lost its
    properties. Properties are validated against the emitted pair-list shape
    whenever the key is present, because ``_state_from_record`` would otherwise
    ignore or coerce malformed properties silently. An absent ``properties`` key
    stays accepted: the decoder intentionally rehydrates that record as a state
    without properties.
    """
    if not isinstance(record, dict):
        return False
    name = record.get("name")
    if not isinstance(name, str) or not name:
        return False
    if "properties" in record and not _is_trace_state_properties(record["properties"]):
        return False
    return _state_from_record(record) is not None


def _is_trace_state_record_list(value: object) -> bool:
    """Whether ``value`` is a list of well-formed trace state records."""
    return isinstance(value, list) and all(_is_trace_state_record(record) for record in value)


def _trace_structure_error(stage_trace: dict) -> str | None:
    """Fail-closed reason when the trace shape cannot support the stagewise join.

    The six checkpoints cannot be read without a canonical palette, the executed
    support-prune rounds and the final palette: the support checkpoint would have
    to guess whether a state survived pruning. Any missing or malformed piece
    makes the whole section unavailable rather than letting a partial trace claim
    availability it cannot evidence. An entity with no executed rounds is still
    structurally valid - zero rounds can be the truth - but it must still declare
    its final palette, so ``support_rounds == []`` is accepted only alongside a
    well-formed ``final_palette.states`` list.
    """
    palette = stage_trace.get("palette")
    if not isinstance(palette, dict) or not _is_trace_state_record_list(palette.get("states")):
        return "the stage trace carries no well-formed canonical palette states"
    rounds = stage_trace.get("support_rounds")
    if not isinstance(rounds, list):
        return "the stage trace carries no per-round support-prune membership"
    for index, record in enumerate(rounds):
        if not isinstance(record, dict):
            return f"support round {index} is not a state-membership record"
        if not _is_trace_state_record_list(record.get("states_kept")):
            return f"support round {index} carries malformed kept-state entries"
    final = stage_trace.get("final_palette")
    if not isinstance(final, dict):
        return "the stage trace carries no final palette"
    if not _is_trace_state_record_list(final.get("states")):
        return "the stage trace carries no well-formed final palette states"
    return None


def _candidate_states(entry: dict, turns: int) -> set[BlockState]:
    """The ranked candidate states the trace retained for one coordinate."""
    candidates = entry.get("top_candidates")
    if not isinstance(candidates, list):
        return set()
    return _rotated_state_records(
        [candidate.get("state") for candidate in candidates if isinstance(candidate, dict)],
        turns,
    )


def _observation_or_geometry(state: BlockState | None) -> tuple[BlockState | None, bool]:
    """A checkpoint that carried no state at all is a geometry loss, not a wrong label."""
    return (state, False) if state is not None else (None, True)


def _ranked_candidate_states(entry: dict, turns: int) -> list[BlockState]:
    """The trace's distinct per-cell ranked candidate states, best first, in the reference frame.

    Prefers the trace-only ``ranked_candidates`` list and falls back to
    ``top_candidates`` only when the trace predates that field, so a legacy or
    default-depth trace still reads at K = 8. The two lists are not slices of
    each other: production's top-K comes from an unordered ``argpartition`` and
    may exclude a state the ordered deep list ranks inside K, so the sweep must
    read the ordered list whenever the trace carries one. Rehydrates and rotates
    the ordered records rather than collapsing them to a set, because the K sweep
    asks how deep in the ranking the reference state sits. Deduplicates on the
    exact rotated state, keeping the best-ranked occurrence, so a state repeated
    in the ranking occupies one rank. A malformed member is dropped, matching the
    membership decoder's tolerance.
    """
    if "ranked_candidates" in entry:
        records = entry.get("ranked_candidates")
    else:
        records = entry.get("top_candidates")
    if not isinstance(records, list):
        return []
    ranked: list[BlockState] = []
    seen: set[BlockState] = set()
    for candidate in records:
        if not isinstance(candidate, dict):
            continue
        state = _aligned_state(candidate.get("state"), turns)
        if state is None or state in seen:
            continue
        seen.add(state)
        ranked.append(state)
    return ranked


def _candidate_availability(
    trace_entries: dict[tuple, dict],
    reference: dict,
    visible_coordinates,
    palette_states: set[BlockState],
    captured_limit: int,
    turns: int,
) -> dict[str, object]:
    """Candidate-availability ceiling at K = 8/16/32 on the pinned visible denominator.

    Evaluator-only and post-hoc: for every visible reference cell the transformed
    reference state is looked up in that cell's own ranked candidate list. Two
    denominators are reported and they answer different questions. The
    truth-in-canonical-palette denominator is the availability ceiling - a state the
    palette lacks can never be ranked at any K - and the geometry-present
    denominator is the visible cells that actually formed a candidate record, so a
    cell that never reached candidate generation is charged to geometry, not to
    ranking depth. ``captured_limit`` is what the trace's own header says it kept:
    a K above it is reported as unavailable rather than as a missing candidate.
    """
    visible = [coordinate for coordinate in visible_coordinates if coordinate in reference]
    rankable = [
        coordinate for coordinate in visible if reference[coordinate] in palette_states
    ]
    ranked_by_coordinate: dict[tuple, list[BlockState]] = {}
    for coordinate in visible:
        entry = trace_entries.get(coordinate)
        if entry is not None:
            ranked_by_coordinate[coordinate] = _ranked_candidate_states(entry, turns)
    geometry_present = [
        coordinate for coordinate in visible if len(ranked_by_coordinate.get(coordinate, ())) > 0
    ]
    checkpoints: list[dict[str, object]] = []
    for k in STAGE_TRACE_CANDIDATE_AVAILABILITY_K:
        captured = k <= captured_limit
        exact = sum(
            1 for coordinate in visible
            if reference[coordinate] in ranked_by_coordinate.get(coordinate, ())[:k]
        )
        exact_rankable = sum(
            1 for coordinate in rankable
            if reference[coordinate] in ranked_by_coordinate.get(coordinate, ())[:k]
        )
        candidates_per_cell = [
            min(k, len(ranked_by_coordinate.get(coordinate, ()))) for coordinate in visible
        ]
        cells = len(candidates_per_cell)
        checkpoints.append({
            "k": k,
            "captured": captured,
            "captured_limit_reason": (
                None if captured
                else (
                    f"the stage trace kept at most {captured_limit} candidates per cell, "
                    f"so K={k} availability cannot be evidenced by this run"
                )
            ),
            "visible_denominator": len(visible),
            "truth_in_palette_denominator": len(rankable),
            "geometry_present_denominator": len(geometry_present),
            "exact_state_available": exact,
            "exact_state_available_in_palette": exact_rankable,
            "exact_state_recall_in_palette": (
                (exact_rankable / len(rankable)) if rankable else 0.0
            ),
            "captured_candidates_beyond_palette": sum(
                candidate not in palette_states
                for coordinate in visible
                for candidate in ranked_by_coordinate.get(coordinate, ())[:k]
            ),
            "mean_captured_candidates_per_visible_cell": (
                (sum(candidates_per_cell) / cells) if cells else 0.0
            ),
        })
    return {
        "schema": STAGE_TRACE_SCHEMA,
        "source": "stage trace top_candidates ranked list (reference-agnostic; joined post hoc)",
        "ks": [checkpoint["k"] for checkpoint in checkpoints],
        "captured_limit": int(captured_limit),
        "captured_limit_sufficient_for_ks": [
            checkpoint["k"] for checkpoint in checkpoints if checkpoint["captured"]
        ],
        "uncaptured_ks": [
            checkpoint["k"] for checkpoint in checkpoints if not checkpoint["captured"]
        ],
        "visible_denominator": len(visible),
        "truth_in_palette_denominator": len(rankable),
        "geometry_present_denominator": len(geometry_present),
        "cells_without_a_trace_record": sum(
            1 for coordinate in visible if coordinate not in trace_entries
        ),
        "cells_with_no_ranked_candidates": sum(
            1 for coordinate in visible
            if coordinate in trace_entries
            and not ranked_by_coordinate.get(coordinate)
        ),
        "reference_states_beyond_palette": len(visible) - len(rankable),
        "checkpoints": checkpoints,
        "accounting": (
            "candidate availability is the per-cell ceiling, not a label: it asks whether "
            "the reference state sits within the cell's ranked K candidates, whatever won. "
            "The sweep reads the trace's ordered ``ranked_candidates`` list, so its K=8 row "
            "is the cheapest-eight availability ceiling and can differ slightly from the "
            "``candidate`` checkpoint, which tests set membership in production's "
            "``top_candidates``; production selects those with an unordered argpartition, "
            "so a state can rank inside the cheapest eight yet sit outside that set. "
            "A state the canonical palette lacks is unreachable at every K and is charged "
            "to the palette, not to ranking depth; a visible cell that formed no candidate "
            "record is charged to geometry. K above ``captured_limit`` is reported as "
            "uncaptured, never as a zero-recall measurement."
        ),
    }


def _stagewise_visible_loss(
    reference,
    candidate,
    visible_pixels,
    *,
    stage_trace,
    translation,
    rotation_quarter_turns: int,
    requested: bool,
) -> dict[str, object]:
    """Join the reference-free stage trace to the pinned visible denominator.

    The join is deliberately post-hoc and evaluator-only. Reference labels,
    visible coordinates, the alignment transform and the mask-derived denominator
    are read here, after reconstruction has finished, and for each visible
    reference cell the *transformed* reference state is checked against what the
    trace says the canonical palette, the global support pruning, the cell's own
    retained candidates, the raw candidate ranking, adjacency smoothing and the
    final placement carried.

    Returned counts distinguish a cell that is missing from the reconstruction
    (geometry) from a cell that was matched but carried the wrong state, and the
    checkpoint match rates use the same pinned visible denominator the report's
    primary row uses. Availability and label checkpoints are tallied separately
    and every transition reports gains as well as losses, because a later
    checkpoint is not bounded by an earlier one. Absent, malformed or empty trace
    data fails closed: the section is reported as unavailable rather than
    silently as zero loss. A trace whose *structure* cannot establish the
    canonical palette, the executed support-prune rounds and the final palette
    also fails the whole section closed, because no checkpoint can be read
    without that membership evidence.
    """
    visible_reference = {coordinate for coordinate in visible_pixels if coordinate in reference}
    alignment = {
        "rotation_quarter_turns": int(rotation_quarter_turns),
        "translation": [int(value) for value in translation],
        "coordinate_transform": "candidate (x,y,z) rotated about +Y, then translated; no reflection or scale fit",
        "state_transform": "cardinal facing/axis/rotation properties rotate with coordinates",
    }
    if not requested:
        return _stagewise_unavailable("stage trace was not requested for this run", alignment)
    if not isinstance(stage_trace, dict) or not stage_trace:
        return _stagewise_unavailable("the reconstruction produced no stage trace", alignment)
    records = stage_trace.get("coordinates")
    if not isinstance(records, list) or not records:
        return _stagewise_unavailable("the stage trace carries no per-coordinate records", alignment)

    # Structural fail-closed gate: a trace that cannot establish the canonical
    # palette, the support-round membership and the final palette must never be
    # read as a checkpoint result. In particular a partial trace may not borrow
    # support availability from a missing membership list.
    structure_error = _trace_structure_error(stage_trace)
    if structure_error is not None:
        return _stagewise_unavailable(structure_error, alignment)

    palette_states = _rotated_state_records(
        stage_trace["palette"]["states"], rotation_quarter_turns)
    support_states = _support_states(stage_trace, rotation_quarter_turns)
    # The trace header declares how deep its own ranked copy goes. A malformed or
    # absent header falls back to the production candidate cap, so the K sweep
    # reports "uncaptured" rather than borrowing availability from a wider list.
    raw_captured_limit = stage_trace.get("candidate_limit")
    captured_limit = (
        raw_captured_limit
        if isinstance(raw_captured_limit, int) and not isinstance(raw_captured_limit, bool)
        and raw_captured_limit >= 1
        else raycast.STAGE_TRACE_CANDIDATE_LIMIT
    )

    trace_by_coordinate: dict[tuple, dict] = {}
    malformed = 0
    for entry in records:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        coordinate = _aligned_trace_coordinate(
            entry.get("coordinate"), rotation_quarter_turns, translation)
        if coordinate is None:
            malformed += 1
            continue
        trace_by_coordinate[coordinate] = entry

    def observe(stage: str, coordinate: tuple) -> tuple[BlockState | None, bool]:
        return _stage_observation(
            stage, reference[coordinate], trace_by_coordinate.get(coordinate),
            palette_states=palette_states,
            support_states=support_states,
            turns=rotation_quarter_turns,
        )

    stages: list[dict[str, object]] = []
    for name, kind, description in STAGE_TRACE_STAGES:
        exact = sum(
            1 for coordinate in visible_reference
            if observe(name, coordinate)[0] == reference[coordinate]
        )
        stages.append({
            "stage": name,
            "kind": kind,
            "description": description,
            "visible_reference_blocks": len(visible_reference),
            "exact_state_matches": exact,
            "match_rate": (exact / len(visible_reference)) if visible_reference else 0.0,
        })

    observations = {
        stage["stage"]: {
            coordinate: observe(stage["stage"], coordinate)
            for coordinate in visible_reference
        }
        for stage in stages
    }
    if not visible_reference:
        return _stagewise_unavailable("the run has no visible reference cells to score", alignment)
    if set(observations) != {name for name, _kind, _description in STAGE_TRACE_STAGES}:
        return _stagewise_unavailable(
            "the stage trace cannot be read across all six checkpoints", alignment)
    candidate_availability = _candidate_availability(
        trace_by_coordinate, reference, visible_reference, palette_states,
        captured_limit, rotation_quarter_turns)
    transitions: list[dict[str, object]] = []
    for earlier, later in zip(stages, stages[1:]):
        geometry_lost = 0
        state_lost = 0
        gained = 0
        for coordinate in visible_reference:
            expected = reference[coordinate]
            before = observations[earlier["stage"]][coordinate][0]
            after, unmatched = observations[later["stage"]][coordinate]
            if before == expected and after != expected:
                if unmatched:
                    geometry_lost += 1
                else:
                    state_lost += 1
            elif before != expected and after == expected:
                gained += 1
        transitions.append({
            "from_stage": earlier["stage"],
            "to_stage": later["stage"],
            "geometry_lost": geometry_lost,
            "state_lost": state_lost,
            "total_lost": geometry_lost + state_lost,
            "gained": gained,
            "net_change": gained - (geometry_lost + state_lost),
        })

    final = {
        coordinate: observation[0]
        for coordinate, observation in observations["placed"].items()
    }
    geometry_missing = [coordinate for coordinate in visible_reference if final[coordinate] is None]
    matched_wrong = [
        coordinate for coordinate in visible_reference
        if final[coordinate] is not None and reference[coordinate] != final[coordinate]
    ]

    candidates_present = sum(1 for entry in trace_by_coordinate.values() if entry.get("raw_winner"))
    stage_losses = {
        transition["from_stage"]: transition["total_lost"] for transition in transitions
    }
    largest_stage = max(stage_losses, key=lambda name: (stage_losses[name], name), default=None)
    return {
        "schema": STAGE_TRACE_SCHEMA,
        "available": True,
        "unavailable_reason": None,
        "visible_reference_blocks": len(visible_reference),
        "minimum_pixels": PRIMARY_VISIBILITY_MINIMUM_PIXELS,
        "denominator": "pinned visible reference cells (reference intersect usable-mask pixel domain)",
        "alignment": alignment,
        "stage_order": [name for name, _kind, _description in STAGE_TRACE_STAGES],
        "stages": stages,
        "stage_transitions": transitions,
        "candidate_availability": candidate_availability,
        "summary": {
            "largest_stage_loss": largest_stage,
            "largest_stage_loss_cells": stage_losses.get(largest_stage, 0),
            "largest_stage_loss_geometry_cells": next(
                (transition["geometry_lost"] for transition in transitions
                 if transition["from_stage"] == largest_stage), 0),
            "largest_stage_loss_state_cells": next(
                (transition["state_lost"] for transition in transitions
                 if transition["from_stage"] == largest_stage), 0),
            "stage_losses": stage_losses,
            "availability_matches": {
                stage["stage"]: stage["exact_state_matches"]
                for stage in stages if stage["kind"] == STAGE_TRACE_AVAILABILITY
            },
            "label_matches": {
                stage["stage"]: stage["exact_state_matches"]
                for stage in stages if stage["kind"] == STAGE_TRACE_LABEL
            },
            "geometry_missing_cells": len(geometry_missing),
            "matched_wrong_state_cells": len(matched_wrong),
            "final_exact_state_matches": sum(
                1 for coordinate in visible_reference if final[coordinate] == reference[coordinate]),
            "trace_candidate_records": candidates_present,
            "trace_records": len(trace_by_coordinate),
            "trace_malformed_records": malformed,
            "trace_records_outside_reference": len(
                set(trace_by_coordinate) - set(reference)),
            "final_state_confusion": _final_state_confusion(
                matched_wrong, reference, final),
        },
        "accounting": {
            "geometry_missing": (
                "geometric loss: the reference cell has no aligned candidate block at all"
            ),
            "matched_wrong_state": (
                "state loss: the reference cell has an aligned candidate block whose state differs"
            ),
            "stage_losses": (
                "checkpoints, not strictly monotone ceilings: availability says whether the "
                "reference state was still reachable, a label says whether the pipeline chose "
                "it, a delta counts cells that matched at the earlier checkpoint and stopped "
                "matching at the later one (gains are counted separately, so a later checkpoint "
                "may match more cells than an earlier one), and a geometry loss means the later "
                "checkpoint carried no state at all"
            ),
        },
    }


def _stagewise_unavailable(reason: str, alignment: dict[str, object]) -> dict[str, object]:
    """Fail-closed stagewise section when the trace is absent or unusable."""
    return {
        "schema": STAGE_TRACE_SCHEMA,
        "available": False,
        "unavailable_reason": reason,
        "visible_reference_blocks": 0,
        "minimum_pixels": PRIMARY_VISIBILITY_MINIMUM_PIXELS,
        "denominator": "pinned visible reference cells (reference intersect usable-mask pixel domain)",
        "alignment": alignment,
        "stage_order": [name for name, _kind, _description in STAGE_TRACE_STAGES],
        "stages": [],
        "stage_transitions": [],
        # No trace metric is measured here, so the K sweep is omitted rather than
        # reported as a measured zero ceiling. Its presence is the contract.
        "candidate_availability": None,
        "summary": {
            "largest_stage_loss": None,
            "largest_stage_loss_cells": 0,
            "largest_stage_loss_geometry_cells": 0,
            "largest_stage_loss_state_cells": 0,
            "stage_losses": {},
            "availability_matches": {},
            "label_matches": {},
            "geometry_missing_cells": 0,
            "matched_wrong_state_cells": 0,
            "final_exact_state_matches": 0,
            "trace_candidate_records": 0,
            "trace_records": 0,
            "trace_malformed_records": 0,
            "trace_records_outside_reference": 0,
            "final_state_confusion": [],
        },
        "accounting": {
            "geometry_missing": (
                "geometric loss: the reference cell has no aligned candidate block at all"
            ),
            "matched_wrong_state": (
                "state loss: the reference cell has an aligned candidate block whose state differs"
            ),
            "stage_losses": (
                "checkpoints, not strictly monotone ceilings: availability says whether the "
                "reference state was still reachable, a label says whether the pipeline chose "
                "it, a delta counts cells that matched at the earlier checkpoint and stopped "
                "matching at the later one (gains are counted separately, so a later checkpoint "
                "may match more cells than an earlier one), and a geometry loss means the later "
                "checkpoint carried no state at all"
            ),
        },
    }


def _stage_observation(
    stage: str,
    reference_state: BlockState,
    entry: dict | None,
    *,
    palette_states: set[BlockState],
    support_states: set[BlockState] | None,
    turns: int,
) -> tuple[BlockState | None, bool]:
    """Return ``(carried_state, unmatched_is_geometry)`` for one cell at one checkpoint.

    ``carried_state`` is the state the checkpoint reached for this coordinate -
    the reference state if the checkpoint got there, otherwise the wrong state it
    held. It is ``None`` when the checkpoint had no state at all, which no later
    checkpoint can recover from; ``unmatched_is_geometry`` reports whether that
    absence, rather than a matched block carrying the wrong state, is why the
    cell is wrong.

    The availability checkpoints (palette, support, candidate) ask whether the
    reference state was still *reachable*; they never compare it with the state
    the pipeline happened to prefer. That separation is the point: a state that
    survives pruning but loses the raw ranking is still available, and a cell
    whose retained candidates exclude it is not, however the winner compares.
    The label checkpoints (raw winner, smoothing, placement) ask which state was
    actually chosen. Trace states are rotated by ``turns`` because the report's
    transform is candidate -> reference.
    """
    if stage == "available_palette":
        # The palette exists independently of any cell, so this checkpoint counts
        # a visible cell even when the reconstruction never formed a record for
        # it; the cell is unreachable here exactly when the canonical palette
        # lacks its state.
        if reference_state in palette_states:
            return reference_state, False
        return None, True
    if stage == "support":
        # Global support pruning and the final palette, not this cell's winner:
        # a state dropped by any round is unreachable no matter what still ranks
        # first for the cell. A trace with no usable membership at all can have
        # cost nothing, so it counts as surviving rather than inventing a loss.
        if support_states is None or reference_state in support_states:
            return reference_state, False
        return _observation_or_geometry(
            _aligned_state(entry.get("raw_winner") if entry else None, turns))
    if entry is None:
        return None, True
    if stage == "candidate":
        # The cell's own retained top-K ranked candidates: the reference state is
        # available here even when it is not the winner, which is what keeps this
        # checkpoint from hiding a recoverable alternative behind the ranking.
        if reference_state in _candidate_states(entry, turns):
            return reference_state, False
        return _observation_or_geometry(_aligned_state(entry.get("raw_winner"), turns))
    if stage == "raw_winner":
        return _observation_or_geometry(_aligned_state(entry.get("raw_winner"), turns))
    if stage == "post_smoothing":
        return _observation_or_geometry(
            _aligned_state(entry.get("post_smoothing_label"), turns))
    return _observation_or_geometry(_aligned_state(entry.get("final_state"), turns))


def _final_state_confusion(matched_wrong, reference, final) -> list[dict[str, object]]:
    """Reference-state to placed-state confusion among visible matched-wrong cells."""
    counts: dict[tuple[str, str], int] = {}
    for coordinate in matched_wrong:
        placed = final.get(coordinate)
        if placed is None:
            continue
        key = (_state_label(reference[coordinate]), _state_label(placed))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"reference_state": key[0], "candidate_state": key[1], "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _state_label(state: BlockState) -> str:
    if not state.properties:
        return state.name
    values = ",".join(f"{key}={value}" for key, value in state.properties)
    return f"{state.name}[{values}]"


def _acceptance_gate(
    *,
    mode: str,
    mask_origin: str,
    fixture_identity: dict[str, object],
    visible_exact_state_recovery: float,
    visible_geometry_f1: float,
    palette_support_override: float | None = None,
) -> dict[str, object]:
    """Report the reviewed thresholds without ever promoting an ineligible run.

    Eligibility is deliberately fail-closed: a derived GUI mask, a
    reference-assisted run, an unreviewed or unpopulated expected pin, a folder
    mismatch, a drifted frozen subset or a frame-set mismatch all block a pass
    even when the numbers clear the thresholds. An explicit
    ``palette_support_override`` is a diagnostic-only ablation control, so it
    likewise blocks a pass for every fixture and every metric value.
    """
    reasons: list[str] = []
    if mode == "reference-assisted":
        reasons.append("reference-assisted runs are evaluation-separated and never promote")
    if mask_origin != "frozen":
        reasons.append("derived production GUI mask is provisional; freeze an evaluator-owned mask")
    if palette_support_override is not None:
        reasons.append(
            f"explicit --palette-support-override={palette_support_override!r} is a "
            "diagnostic-only ablation control; this run cannot pass the acceptance gate"
        )
    if not fixture_identity.get("pinned"):
        missing = [str(field) for field in (fixture_identity.get("missing_pin_fields") or [])]
        mismatched = [str(field) for field in (fixture_identity.get("mismatched_pin_fields") or [])]
        if missing:
            reasons.append(
                "fixture identity is not pinned; expected pin fields are missing: "
                + ", ".join(missing)
            )
        if mismatched:
            reasons.append(
                "fixture identity is not pinned; observed content does not match expected pins: "
                + ", ".join(mismatched)
            )
        if not missing and not mismatched:
            reasons.append("fixture role is not a pinned acceptance fixture")
    if fixture_identity.get("role") is not None and not fixture_identity.get("fixture_matches"):
        reasons.append("fixture folder does not match the pinned role")
    if fixture_identity.get("subset") is not None:
        pin = fixture_identity.get("subset_pin") or {}
        if not pin.get("digest_matches"):
            reasons.append("frozen subset digest does not match its recorded pin")
        if not fixture_identity.get("frames_match"):
            reasons.append("input frames do not match the frozen subset")
    metrics_met = (
        visible_exact_state_recovery >= ACCEPTANCE_MINIMUM_VISIBLE_EXACT_STATE_RECOVERY
        and visible_geometry_f1 >= ACCEPTANCE_MINIMUM_VISIBLE_GEOMETRY_F1
    )
    return {
        "minimum_pixels": PRIMARY_VISIBILITY_MINIMUM_PIXELS,
        "minimum_visible_exact_state_recovery": ACCEPTANCE_MINIMUM_VISIBLE_EXACT_STATE_RECOVERY,
        "minimum_visible_geometry_f1": ACCEPTANCE_MINIMUM_VISIBLE_GEOMETRY_F1,
        "observed_visible_exact_state_recovery": visible_exact_state_recovery,
        "observed_visible_geometry_f1": visible_geometry_f1,
        "metrics_met": metrics_met,
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "passed": bool(metrics_met and not reasons),
        "fixture_role": fixture_identity.get("role"),
        "fixture_pinned": bool(fixture_identity.get("pinned")),
        "palette_support_override": palette_support_override,
        "reference_sha256": fixture_identity.get("reference_sha256"),
        "input_frames_sha256": fixture_identity.get("input_frames_sha256"),
        "fixture_sha256": fixture_identity.get("fixture_sha256"),
        "missing_pin_fields": list(fixture_identity.get("missing_pin_fields") or []),
        "mismatched_pin_fields": list(fixture_identity.get("mismatched_pin_fields") or []),
        "required_fixtures": list(REQUIRED_PROMOTION_FIXTURES),
        "fixture_set_complete": False,
        "required_identity_fields": list(GATE_REQUIRED_IDENTITY_FIELDS),
        "scope": (
            "single-run per-fixture metric status only; not image-only production qualification, "
            "and promotion requires one passing, reviewed-pin, identity-stable run for every "
            "required fixture in one aggregate"
        ),
    }


def _gate_identity_problems(gate: dict[str, object]) -> list[str]:
    """Return the eligibility/metric/identity fields an aggregate entry is missing."""
    problems: list[str] = []
    for field in ("eligible", "metrics_met", "passed", "fixture_pinned"):
        if gate.get(field) is not True:
            problems.append(field)
    for field in GATE_REQUIRED_IDENTITY_FIELDS:
        if not gate.get(field):
            problems.append(field)
    return problems


def aggregate_promotion_gate(reports) -> dict[str, object]:
    """Require one passing, reviewed-pin, identity-stable run per fixture role.

    Per-fixture metric success is not production qualification: an aggregate
    passes only when every required role is present exactly once and its entry is
    eligible, meets the metrics, is bound to a reviewed expected pin and carries
    every required identity field. A report written under a superseded schema is
    rejected too, because the identity contract changed with the report contract.
    Missing fields fail closed instead of being treated as agreement.
    """
    entries: dict[str, list[dict[str, object]]] = {}
    schema_problems: dict[str, set[str]] = {}
    for report in reports:
        gate = report.get("acceptance_gate") or {}
        role = gate.get("fixture_role")
        if role is None:
            continue
        key = str(role)
        entries.setdefault(key, []).append(gate)
        if report.get("report_schema") != REPORT_SCHEMA_VERSION:
            schema_problems.setdefault(key, set()).add("report_schema")
    missing = [role for role in REQUIRED_PROMOTION_FIXTURES if role not in entries]
    invalid: dict[str, list[str]] = {}
    for role, gates in entries.items():
        problems = {problem for gate in gates for problem in _gate_identity_problems(gate)}
        problems |= schema_problems.get(role, set())
        if problems:
            invalid[role] = sorted(problems)
    failed = [
        role for role in REQUIRED_PROMOTION_FIXTURES
        if role in entries and not all(gate.get("passed") is True for gate in entries[role])
    ]
    duplicated = sorted(role for role, gates in entries.items() if len(gates) > 1)
    unexpected = sorted(role for role in entries if role not in REQUIRED_PROMOTION_FIXTURES)
    drifted = sorted(
        role for role, gates in entries.items()
        if len({gate.get("fixture_sha256") for gate in gates}) > 1
        or len({gate.get("input_frames_sha256") for gate in gates}) > 1
    )
    passed = not (missing or failed or duplicated or unexpected or drifted or invalid)
    return {
        "passed": passed,
        "required_fixtures": list(REQUIRED_PROMOTION_FIXTURES),
        "present_fixtures": sorted(entries),
        "missing_fixtures": missing,
        "failed_fixtures": failed,
        "duplicate_fixtures": duplicated,
        "unexpected_fixtures": unexpected,
        "drifted_fixtures": drifted,
        "invalid_fixtures": invalid,
        "required_identity_fields": list(GATE_REQUIRED_IDENTITY_FIELDS),
        "scope": (
            "aggregate per-fixture metric status; image-only production qualification is a "
            "separate contract and is not claimed here"
        ),
    }


def _file_provenance(path: Path) -> dict[str, object]:
    source = path.resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _load_usable_masks(
    paths: dict[str, Path] | None, frames: dict[str, object]
) -> dict[str, object] | None:
    """Load evaluator-owned masks and verify coverage.

    PNG masks use white for usable pixels. A tracked JSON specification freezes
    integer excluded rectangles and polygons for a fixed image size, avoiding a
    dependency on the mutable production GUI-mask heuristic.

    ``None`` means no mask was supplied, which is *not* an authoritative map: the
    reconstructor derives its own usable domain. Returning ``{}`` here instead
    would be an authoritative map with every frame missing, so the run would
    silently score no rectified-face evidence and reconstruct from centre-patch
    fallbacks alone.
    """
    if not paths:
        return None
    missing = sorted(set(frames) - set(paths))
    unknown = sorted(set(paths) - set(frames))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing masks for {', '.join(missing)}")
        if unknown:
            details.append(f"masks without input frames: {', '.join(unknown)}")
        raise ValueError("; ".join(details))
    from PIL import Image
    import numpy as np

    masks: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        image = frames[name]
        expected = image.shape[:2]  # type: ignore[attr-defined]
        if path.suffix.lower() == ".json":
            specification = json.loads(path.read_text(encoding="utf-8"))
            width, height = specification["image_size"]
            if (int(height), int(width)) != expected:
                raise ValueError(
                    f"usable mask {path} fixes {(width, height)}, expected {(expected[1], expected[0])} for {name}"
                )
            mask = np.ones(expected, dtype=bool)
            for rectangle in specification["excluded_rectangles"]:
                x0, y0, x1, y1 = (int(value) for value in rectangle)
                if not (0 <= x0 <= x1 <= expected[1] and 0 <= y0 <= y1 <= expected[0]):
                    raise ValueError(f"usable mask {path} has out-of-bounds rectangle {rectangle}")
                mask[y0:y1, x0:x1] = False
            for polygon in specification.get("excluded_polygons", []):
                points = np.asarray(polygon, dtype=np.int32)
                if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                    raise ValueError(f"usable mask {path} has invalid polygon {polygon}")
                if (points[:, 0] < 0).any() or (points[:, 0] >= expected[1]).any() or (
                    points[:, 1] < 0
                ).any() or (points[:, 1] >= expected[0]).any():
                    raise ValueError(f"usable mask {path} has out-of-bounds polygon {polygon}")
                import cv2

                cv2.fillPoly(mask, [points], False)
        else:
            mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
        if mask.shape != expected:
            raise ValueError(f"usable mask {path} has shape {mask.shape}, expected {expected} for {name}")
        masks[name] = mask
    return masks


def _source_provenance() -> dict[str, object]:
    """Capture revision and content hashes from the actual worktree used."""
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[1],
            check=False, capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    root = Path(__file__).resolve().parents[1]
    revision = git("rev-parse", "HEAD")
    dirty = (git("status", "--porcelain=v1") or "").splitlines()
    sources = (git("ls-files", "src", "scripts", "pyproject.toml") or "").splitlines()
    digest = hashlib.sha256()
    for name in sources:
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "git_revision": revision,
        "git_dirty": bool(dirty),
        "git_status_porcelain": dirty,
        "tracked_source_sha256": digest.hexdigest(),
        "tracked_source_file_count": len(sources),
    }


def _dependency_versions() -> dict[str, str | None]:
    return {name: _distribution_version(name) for name in ("numpy", "opencv-python", "Pillow", "nbtlib")}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _trace_output_bytes(trace_text: str) -> bytes:
    """The exact bytes the trace destination receives for ``trace_text``.

    The trace is now written with ``write_bytes`` so it can be preflighted and
    digested before either destination exists. ``Path.write_text`` opens in text
    mode with the platform's newline translation, so replicating that translation
    here keeps the trace output byte-for-byte what the previous writer produced
    (on Windows every ``\\n`` becomes ``\\r\\n``; on POSIX it is unchanged). The
    returned bytes are the ones hashed into the sidecar and written to disk, so
    the bound digest always describes the real file content.
    """
    return trace_text.encode("utf-8").replace(b"\n", os.linesep.encode("utf-8"))


def _canonical_path(path: Path) -> str:
    """Canonical, case-folded identity of a path that need not already exist.

    ``os.path.realpath`` resolves symlinked/reparse-point parents (and ``.``/``..``)
    even when the leaf itself is absent, so two lexically distinct arguments that
    reach the same destination through a linked parent compare equal. Applying
    ``os.path.normcase`` afterwards keeps the comparison case-insensitive on
    Windows and case-sensitive on POSIX, exactly as the previous lexical check
    did.
    """
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _paths_alias(first: Path, second: Path) -> bool:
    """Whether two paths name the same destination after canonical resolution.

    Both arguments are reduced to their real, symlink-resolved path before the
    platform case fold, so a trace/sidecar pair that differs only by case,
    separator style, ``..`` or a linked parent is refused as one file. Neither
    path has to exist.
    """
    return _canonical_path(first) == _canonical_path(second)


def _stage_sidecar_payload(document: object) -> bytes:
    """Serialize the sidecar deterministically, refusing an over-bound payload.

    The record count and the exact serialized byte length are checked here,
    before any file is opened, so a payload above either fixed bound leaves no
    destination and no temporary artifact. The measured bytes are the same UTF-8
    bytes handed to the atomic writer.
    """
    if not isinstance(document, dict):
        raise ValueError("stage sidecar document must be an object")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("stage sidecar document must carry a records list")
    if len(records) > raycast.STAGE_SIDECAR_MAX_RECORDS:
        raise ValueError(
            f"stage sidecar carries {len(records)} resolved-face records, above the "
            f"{raycast.STAGE_SIDECAR_MAX_RECORDS} record bound"
        )
    payload = (
        json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(payload) > raycast.STAGE_SIDECAR_MAX_BYTES:
        raise ValueError(
            f"stage sidecar serializes to {len(payload)} bytes, above the "
            f"{raycast.STAGE_SIDECAR_MAX_BYTES} byte bound"
        )
    return payload


def _write_stage_sidecar(path: Path, payload: bytes, *, trace_path: Path) -> None:
    """Write the sidecar atomically, refusing to overwrite or alias the trace.

    The bytes go to a uniquely named, task-owned temporary file in the target
    directory and are linked to the destination only once complete. A hard link
    is used instead of a rename precisely because it fails atomically when the
    destination exists, so no overwrite is possible even under a race. On any
    failure the temporary file is removed, and the destination is never created
    unless the full payload was written.
    """
    if _paths_alias(path, trace_path):
        raise ValueError("stage_sidecar_path and stage_trace_path must be different files")
    if path.exists():
        raise FileExistsError(
            f"stage_sidecar_path already exists and will not be overwritten: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.link(temporary, path)
    finally:
        # The temporary file is task-owned in both the success and failure paths;
        # the destination is a separate link to the same bytes.
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _parse_usable_masks(values: list[str]) -> dict[str, Path]:
    masks: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--usable-mask must be FRAME=PATH")
        if name in masks:
            raise ValueError(f"duplicate usable mask for {name}")
        masks[name] = Path(raw_path)
    return masks


def _validated_palette_support(value: object) -> float:
    """Return ``value`` as a finite, nonnegative support-pruning override.

    ``None`` is not accepted here: callers keep ``None`` meaning "forward the
    unchanged module default". A NaN, infinite or negative override is rejected
    because ``palette_support > 0`` would silently disable support pruning and a
    non-finite value would serialize as non-standard ``NaN``/``Infinity`` JSON.
    """
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"palette_support override must be a finite nonnegative number, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"palette_support override must be finite and >= 0, got {value!r}"
        )
    return numeric


def _palette_support_override_argument(value: str) -> float:
    """argparse ``type`` for ``--palette-support-override``.

    Rejects nonnumeric, non-finite and negative values at parse time, before any
    ``run()`` call can load fixture input.
    """
    try:
        return _validated_palette_support(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validated_stage_trace_candidate_limit(value: object) -> int:
    """Return ``value`` as a positive integer stage-trace candidate depth.

    The depth only widens the evaluator trace's copy of each cell's already
    computed cheapest-first candidate order, so it can never be zero, negative or
    fractional. ``bool`` is rejected explicitly: ``True`` would otherwise pass an
    ``isinstance(value, int)`` check and silently become a depth of one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"stage_trace_candidate_limit must be a positive integer, got {value!r}"
        )
    return value


def _stage_trace_candidate_limit_argument(value: str) -> int:
    """argparse ``type`` for ``--stage-trace-candidate-limit``."""
    try:
        return _validated_stage_trace_candidate_limit(int(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "stage-trace candidate limit must be a positive integer"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser.

    Kept separate from the ``__main__`` block so focused tests can exercise
    option parsing without running a benchmark.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=MODES, default="supplied-camera")
    parser.add_argument("--reference", type=Path, help="explicit evaluation-only litematic")
    parser.add_argument("--camera-manifest", type=Path, help="explicit supplied-camera diagnostic input")
    parser.add_argument("--scene", type=Path, help="explicit supplied scene bounds diagnostic input")
    parser.add_argument("--minecraft-jar", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-default-palette", action="store_true")
    parser.add_argument("--adjacency-strength", type=float, default=0.55)
    parser.add_argument("--bounded-adjacency", action="store_true")
    parser.add_argument(
        "--palette-support-override", type=_palette_support_override_argument,
        default=None, metavar="FLOAT",
        help=(
            "Diagnostic-only ablation control: override the reconstruction's global "
            "palette support-pruning threshold. Omitted (the default) forwards the "
            "unchanged raycast.PALETTE_SUPPORT value. Any explicit finite nonnegative "
            "value is recorded in the effective settings and makes this run's "
            "acceptance gate ineligible; negative, NaN and infinite values are rejected."
        ),
    )
    parser.add_argument(
        "--correct-light-cast", action=argparse.BooleanOptionalAction,
        default=raycast.CORRECT_LIGHT_CAST,
        help="Use the image-only chroma-spread light-cast correction (enabled by default).",
    )
    parser.add_argument(
        "--usable-mask", action="append", default=[], metavar="FRAME=PATH",
        help="Frozen evaluator-owned white=usable image mask or tracked rectangle-spec JSON; provide one for every frame.",
    )
    parser.add_argument(
        "--fixture-role", choices=sorted(PINNED_FIXTURE_ROLES), default=None,
        help="Pinned acceptance fixture role; required before this run can pass the gate.",
    )
    parser.add_argument(
        "--stage-trace", type=Path, default=None, metavar="PATH",
        help=(
            "Diagnostic-only stagewise visible-state loss instrumentation: write the "
            "reconstruction's reference-free stage trace to PATH and join it to the "
            "pinned visible denominator in the report after reconstruction completes. "
            "Off by default; changes no inference input and no acceptance decision."
        ),
    )
    parser.add_argument(
        "--stage-trace-candidate-limit", type=_stage_trace_candidate_limit_argument,
        default=raycast.STAGE_TRACE_CANDIDATE_LIMIT, metavar="N",
        help=(
            "Diagnostic-only trace depth: how many entries of each cell's "
            "cheapest-first candidate order the stage trace records (default %(default)s). "
            "Raise it (for example to 32) to read candidate availability at K = 16 or 32. "
            "Requires --stage-trace; changes no reconstruction decision, metric or gate."
        ),
    )
    parser.add_argument(
        "--stage-sidecar", type=Path, default=None, metavar="PATH",
        help=(
            "Opt-in evaluation-only sidecar: write the actual resolved-face sampled "
            "patch, valid-mask and coverage payloads to PATH, keyed to the --stage-trace "
            "schema-2 observation ordinals and bound to that trace's exact content "
            "digest. Requires --stage-trace; off by default; changes no inference "
            "input, metric, decision or gate. Refuses a path that already exists, a "
            "path that names the trace, or a payload above 10000 records / 64 MiB."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        args.template, args.output, mode=args.mode, reference=args.reference,
        camera_manifest=args.camera_manifest, scene_definition=args.scene, jar=args.minecraft_jar,
        adjacency_strength=args.adjacency_strength, report_path=args.report,
        allow_default_palette=args.allow_default_palette, bounded_adjacency=args.bounded_adjacency,
        palette_support=args.palette_support_override,
        usable_mask_paths=_parse_usable_masks(args.usable_mask),
        correct_light_cast=args.correct_light_cast,
        fixture_role=args.fixture_role,
        stage_trace_path=args.stage_trace,
        stage_trace_candidate_limit=args.stage_trace_candidate_limit,
        stage_sidecar_path=args.stage_sidecar,
    )
