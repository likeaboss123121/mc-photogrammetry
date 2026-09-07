from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
from mc_photogrammetry.export.compare import compare_litematics, load_litematic
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction.camera import load_camera_manifest, project_points
from mc_photogrammetry.reconstruction.facade import DEFAULT_PALETTE
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.texture_calibration import calibrate_appearance


def run(template_dir: Path, output: Path, jar: Path | None, reference_palette: bool, adjacency_strength: float, report_path: Path | None = None, allow_default_palette: bool = False, bounded_adjacency: bool = False) -> None:
    reference = next(template_dir.glob("*.litematic"))
    frames_info = list(discover_media([template_dir]).image_frames)
    cameras = load_camera_manifest(template_dir / "cameras.json")
    scene = load_scene_transform(template_dir / "scene.json")
    frames = {Path(frame.source_path).name: load_image(frame.source_path) for frame in frames_info}
    palette = DEFAULT_PALETTE
    palette_source = "default"
    texture_source = None
    available: dict[str, object] = {}
    if jar:
        provider = VanillaJarTextureProvider()
        if provider.load_jar(jar):
            palette = provider.canonical_palette()
            palette_source = "jar"
            texture_source = provider.source
            available = {sample.state.name: sample for sample in provider.samples}
            if reference_palette:
                palette = tuple(
                    PaletteColor(available[state.name].state, available[state.name].rgb)
                    for state in load_litematic(reference).palette
                    if state.name in available
                )
                palette_source = "reference"
    if not allow_default_palette and palette_source == "default":
        raise SystemExit(f"Palette resolved to DEFAULT_PALETTE for {template_dir}; pass --allow-default-palette to continue.")
    calibrations = None
    if reference_palette:
        texture_colors = {state: tuple(available[state.name].rgb) for state in load_litematic(reference).palette if state.name in available}
        calibrations = {
            name: calibrate_appearance(frames[name], load_litematic(reference), cameras[name], scene, texture_colors)
            for name in frames
        }
    started = time.perf_counter()
    diagnostics: dict[str, object] = {}
    volume = reconstruct_scene_surfaces(
        frames,
        cameras,
        scene,
        palette=palette,
        appearance_calibrations=calibrations,
        adjacency_strength=adjacency_strength,
        bounded_adjacency=bounded_adjacency,
        diagnostics=diagnostics,
    )
    reconstruct_seconds = time.perf_counter() - started
    LitematicExporter().export(volume, output, name=template_dir.name)
    comparison = compare_litematics(reference, output)
    print(f"Reference: {reference}")
    print(f"Candidate: {output}")
    print(f"Dimensions: {comparison.candidate.size}")
    print(f"Blocks: {comparison.candidate.non_air_count}")
    print(f"Coordinate recall: {comparison.coordinate_recall:.2%}")
    print(f"Coordinate precision: {comparison.coordinate_precision:.2%}")
    print(f"Coordinate F1: {comparison.coordinate_f1:.2%}")
    print(f"Surface F1: {comparison.surface_f1:.2%} (precision {comparison.surface_precision:.2%}, recall {comparison.surface_recall:.2%})")
    print(f"Exact state accuracy: {comparison.state_accuracy:.2%}")
    print(f"Whole-reference exact rate: {comparison.whole_reference_state_accuracy:.2%}")
    print(f"Reconstruct seconds: {reconstruct_seconds:.1f}")

    # Which classifier constants were actually in force. A report that records
    # only the score cannot be reproduced once a default moves, and several of
    # these defaults have moved more than once.
    effective_settings = {
        name: getattr(raycast, name)
        for name in sorted(dir(raycast))
        if name.isupper() and isinstance(getattr(raycast, name), (int, float, bool, str))
    }
    effective_settings["adjacency_strength"] = adjacency_strength
    effective_settings["bounded_adjacency"] = bounded_adjacency

    report = {
        "template": template_dir.name,
        "palette_source": palette_source,
        "texture_source": texture_source,
        "palette_size": len(palette),
        "cache_schema_version": getattr(VanillaJarTextureProvider, "CACHE_SCHEMA_VERSION", None),
        "frames": len(frames),
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
        "direct_coordinate_matches": comparison.direct_coordinate_matches,
        "direct_coordinate_recall": comparison.direct_coordinate_recall,
        "direct_state_matches": comparison.direct_state_matches,
        "timings_seconds": {
            "reconstruct": reconstruct_seconds,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "effective_settings": effective_settings,
        "diagnostics": diagnostics,
        "top_confusion_pairs": [
            {"reference_state": ref_name, "candidate_state": cand_name, "count": count}
            for ref_name, cand_name, count in comparison.confusion_pairs
        ],
        "per_state_accuracy": {name: value for name, value in comparison.per_state_accuracy.items()},
        "per_state_counts": comparison.per_state_counts,
        "acceptance_gate": {
            "passed": (
                comparison.state_accuracy >= 0.80
                and comparison.coordinate_f1 >= 0.86
                and not any(
                    reference_state == "minecraft:dark_oak_planks"
                    and candidate_state == "minecraft:dirt"
                    for reference_state, candidate_state, _ in comparison.confusion_pairs
                )
            ),
            "minimum_state_accuracy": 0.80,
            "minimum_coordinate_f1": 0.86,
            "forbidden_confusion": {
                "reference_state": "minecraft:dark_oak_planks",
                "candidate_state": "minecraft:dirt",
            },
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minecraft-jar", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-default-palette", action="store_true")
    parser.add_argument("--reference-palette", action="store_true")
    parser.add_argument("--adjacency-strength", type=float, default=0.55)
    parser.add_argument("--bounded-adjacency", action="store_true")
    args = parser.parse_args()
    run(
        args.template,
        args.output,
        args.minecraft_jar,
        args.reference_palette,
        args.adjacency_strength,
        args.report,
        args.allow_default_palette,
        args.bounded_adjacency,
    )
