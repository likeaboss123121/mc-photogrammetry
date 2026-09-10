"""Profile the first frozen M5 Template 2 cell with the v2 settings.

This is intentionally a one-cell, bounded control.  It measures the fitter
without starting another full 40-cell diagnostic run and keeps setup time
separate from the fit and its instrumented stages.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider
from mc_photogrammetry.export.compare import load_litematic
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.raycast import TEXTURE_WEIGHT
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.reconstruction.shape_fit import (
    EDGE_WEIGHT,
    PARTIAL_FAMILIES,
    RENDER_SCALE,
    candidate_states,
    fit_states,
)

try:
    from scripts.benchmark_templates import _load_usable_masks
except ModuleNotFoundError:
    from benchmark_templates import _load_usable_masks

from m5_matched_t2_diagnostic import _git_provenance, _load_proposals, _palette_hash, _sha256, _state_json


def run(
    template: Path,
    baseline: Path,
    proposals: Path,
    jar: Path,
    mask_dir: Path,
    report: Path,
    *,
    max_candidate_renders: int = 12,
) -> dict[str, object]:
    if max_candidate_renders < 1:
        raise ValueError("max_candidate_renders must be positive")
    root = Path(__file__).resolve().parents[1]
    setup_started = time.perf_counter()
    frames_info = list(discover_media([template]).image_frames)
    frames = {Path(item.source_path).name: load_image(item.source_path) for item in frames_info}
    cameras = load_camera_manifest(template / "cameras.json")
    scene = load_scene_transform(template / "scene.json")
    mask_paths = {name: mask_dir / f"{Path(name).stem}.json" for name in frames}
    masks = _load_usable_masks(mask_paths, frames)
    coordinates = _load_proposals(proposals)
    if not coordinates:
        raise RuntimeError("proposal inventory is empty")
    coordinate = coordinates[0]
    provider = VanillaJarTextureProvider()
    if not provider.load_jar(jar):
        raise RuntimeError(f"could not load Minecraft jar: {jar}")
    palette = provider.complete_palette()
    shapes, materials = candidate_states(palette, PARTIAL_FAMILIES, include_full_cubes=True)
    baseline_snapshot = load_litematic(baseline)
    settled = dict(baseline_snapshot.blocks)
    setup_seconds = time.perf_counter() - setup_started

    diagnostics: dict[str, object] = {}
    tracemalloc.start()
    fit_started = time.perf_counter()
    fitted = fit_states(
        [coordinate], settled, scene, cameras, frames, palette,
        scale=RENDER_SCALE,
        edge_weight=EDGE_WEIGHT,
        texture_weight=TEXTURE_WEIGHT,
        usable_masks=masks,
        diagnostics=diagnostics,
        max_candidate_renders=max_candidate_renders,
    )
    fit_seconds = time.perf_counter() - fit_started
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages = diagnostics.get("shape_fit_stage_seconds", {})
    if not isinstance(stages, dict):
        stages = {}

    files = {
        "baseline": baseline,
        "proposals": proposals,
        "jar": jar,
        "camera_manifest": template / "cameras.json",
        "scene": template / "scene.json",
        **{f"frame:{name}": Path(item.source_path) for name, item in zip(frames, frames_info)},
        **{f"mask:{name}": path for name, path in mask_paths.items()},
    }
    result = {
        "schema": 1,
        "experiment": "M5-Template-2-first-cell-profile",
        "mode": "supplied-camera one-cell control",
        "production_promotion": False,
        "provenance": _git_provenance(root),
        "executed_code": {
            "shape_fit.py": _sha256(root / "src/mc_photogrammetry/reconstruction/shape_fit.py"),
            "profile_runner.py": _sha256(Path(__file__).resolve()),
        },
        "inputs": {key: {"path": str(path.resolve()), "sha256": _sha256(path)} for key, path in files.items()},
        "cell": list(coordinate),
        "effective_settings": {
            "palette": "VanillaJarTextureProvider.complete_palette",
            "palette_sha256": _palette_hash(palette),
            "palette_entries": len(palette),
            "geometry_groups": len(shapes),
            "material_groups": len(materials),
            "shape_fit_scale": RENDER_SCALE,
            "shape_fit_edge_weight": EDGE_WEIGHT,
            "shape_fit_texture_weight": TEXTURE_WEIGHT,
            "candidate_render_cap": max_candidate_renders,
            "proposal_order": "first lexicographic entry from frozen v2 inventory",
            "usable_mask_policy": "frozen supplied JSON masks",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "texture_source": provider.source,
        },
        "selected_state": _state_json(fitted.get(coordinate)),
        "timing_seconds": {
            "setup": setup_seconds,
            "fit_only": fit_seconds,
            "total_process": setup_seconds + fit_seconds,
        },
        "stage_seconds": {name: float(value) for name, value in stages.items()},
        "peak_python_heap_bytes": peak_bytes,
        "render_diagnostics": diagnostics,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-candidate-renders", type=int, default=12)
    args = parser.parse_args()
    data = run(**vars(args))
    print(json.dumps({
        "experiment": data["experiment"],
        "cell": data["cell"],
        "fit_only_seconds": data["timing_seconds"]["fit_only"],
        "candidate_renders": data["render_diagnostics"].get("shape_fit_candidate_renders"),
        "stages": data["stage_seconds"],
    }, indent=2))
