"""Run the maintained, supplied-camera M5 partial-fitting diagnostic.

This runner deliberately consumes an explicitly frozen settled baseline and
proposal inventory. It never uses the reference to choose proposals. The
reference is loaded only for post-hoc evaluation, and the report is marked
diagnostic rather than a production promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider
from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.compare import load_litematic
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.raycast import TEXTURE_WEIGHT
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.reconstruction.shape_fit import (
    DEFAULT_MAX_CANDIDATE_RENDERS,
    EDGE_WEIGHT,
    PARTIAL_FAMILIES,
    RENDER_SCALE,
    candidate_states,
    fit_states,
)
try:
    from scripts.benchmark_templates import _load_usable_masks
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from benchmark_templates import _load_usable_masks


FAMILY_NAMES = ("slab", "carpet", "stairs", "fence", "wall", "pane", "bars", "torch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_json(state: BlockState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {"name": state.name, "properties": dict(state.properties)}


def _palette_hash(palette: tuple) -> str:
    payload = [
        {
            "state": _state_json(item.state),
            "full_cube": bool(item.full_cube),
            "shape_boxes": item.shape_boxes,
        }
        for item in palette
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _family(name: str | None) -> str | None:
    if not name:
        return None
    for family in FAMILY_NAMES:
        if family in name:
            return family
    return "other"


def _git_provenance(root: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=root, check=False, capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "revision": git("rev-parse", "HEAD"),
        "status": (git("status", "--porcelain=v1") or "").splitlines(),
    }


def _load_proposals(path: Path) -> list[tuple[int, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"unsupported proposal schema in {path}")
    coordinates = [tuple(int(v) for v in row) for row in data.get("coordinates", [])]
    if any(len(row) != 3 for row in coordinates):
        raise ValueError("proposal coordinates must have three axes")
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("proposal inventory contains duplicate coordinates")
    if coordinates != sorted(coordinates):
        raise ValueError("v2 proposal inventory must be lexicographically ordered")
    return coordinates


def run(
    template: Path,
    baseline: Path,
    proposals_path: Path,
    reference: Path,
    jar: Path,
    mask_dir: Path,
    output: Path,
    report: Path,
    *,
    max_candidate_renders: int = DEFAULT_MAX_CANDIDATE_RENDERS,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    frames_info = list(discover_media([template]).image_frames)
    frames = {Path(item.source_path).name: load_image(item.source_path) for item in frames_info}
    cameras = load_camera_manifest(template / "cameras.json")
    scene = load_scene_transform(template / "scene.json")
    mask_paths = {
        name: mask_dir / f"{Path(name).stem}.json" for name in frames
    }
    masks = _load_usable_masks(mask_paths, frames)
    proposals = _load_proposals(proposals_path)

    provider = VanillaJarTextureProvider()
    if not provider.load_jar(jar):
        raise RuntimeError(f"could not load Minecraft jar: {jar}")
    palette = provider.complete_palette()
    shapes, materials = candidate_states(palette, PARTIAL_FAMILIES, include_full_cubes=True)
    non_full = [item for item in palette if not item.full_cube]
    family_eligibility = {
        family: sum(
            family in item.state.name and not item.full_cube
            for item in palette
        )
        for family in FAMILY_NAMES
    }
    if not non_full or not shapes:
        raise RuntimeError("complete palette preflight has no partial vocabulary")

    baseline_path = baseline
    baseline_snapshot = load_litematic(baseline_path)
    reference_snapshot = load_litematic(reference)
    settled = {
        coordinate: state for coordinate, state in baseline_snapshot.blocks.items()
    }
    baseline_volume = BlockVolume({
        coordinate: BlockObservation(state=state, confidence=0.0, frame_ids=[])
        for coordinate, state in settled.items()
    })
    diagnostics: dict[str, object] = {}
    started = time.perf_counter()
    tracemalloc.start()
    fitted = fit_states(
        proposals, settled, scene, cameras, frames, palette,
        scale=RENDER_SCALE,
        edge_weight=EDGE_WEIGHT,
        texture_weight=TEXTURE_WEIGHT,
        usable_masks=masks,
        diagnostics=diagnostics,
        max_candidate_renders=max_candidate_renders,
    )
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    after = BlockVolume(blocks=dict(baseline_volume.blocks))
    supporting = getattr(fitted, "supporting_frames", {})
    for coordinate, state in fitted.items():
        if state is None:
            after.blocks.pop(coordinate, None)
        else:
            after.set(coordinate, BlockObservation(
                state=state, confidence=0.0,
                frame_ids=list(supporting.get(coordinate, ())),
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = (
        baseline_snapshot.position,
        tuple(
            baseline_snapshot.position[axis] + baseline_snapshot.size[axis] - 1
            for axis in range(3)
        ),
    )
    LitematicExporter().export(after, output, name="M5 Template 2 matched v2", bounds=bounds)

    before_map = settled
    after_map = {coordinate: item.state for coordinate, item in after.blocks.items()}
    reference_map = reference_snapshot.blocks
    exact_before = sum(before_map.get(c) == state for c, state in reference_map.items())
    exact_after = sum(after_map.get(c) == state for c, state in reference_map.items())
    partial_reference = {
        c: state for c, state in reference_map.items()
        if _family(state.name) in FAMILY_NAMES
    }
    exact_partial_before = sum(before_map.get(c) == state for c, state in partial_reference.items())
    exact_partial_after = sum(after_map.get(c) == state for c, state in partial_reference.items())
    added = sorted(set(after_map) - set(before_map))
    deleted = sorted(set(before_map) - set(after_map))
    changed = sorted(
        c for c in set(before_map) & set(after_map) if before_map[c] != after_map[c]
    )
    phantoms_before = sorted(set(before_map) - set(reference_map))
    phantoms_after = sorted(set(after_map) - set(reference_map))

    cell_results = []
    for item in diagnostics.get("shape_fit_cell_results", []):
        coordinate = tuple(item["coordinate"])
        reference_state = reference_map.get(coordinate)
        selected_name = item.get("selected_state")
        cell_results.append({
            **item,
            "reference_state": _state_json(reference_state),
            "reference_family": _family(reference_state.name) if reference_state else None,
            "selected_family": _family(str(selected_name)) if selected_name else None,
            "supporting_frames": list(supporting.get(coordinate, ())),
        })
    reached_coordinates = {tuple(item["coordinate"]) for item in cell_results}
    for coordinate in proposals:
        if coordinate in reached_coordinates:
            continue
        reference_state = reference_map.get(coordinate)
        cell_results.append({
            "coordinate": list(coordinate),
            "status": "not_reached",
            "usable_views": 0,
            "candidate_renders": 0,
            "current_state": settled.get(coordinate).name if settled.get(coordinate) else None,
            "reference_state": _state_json(reference_state),
            "reference_family": _family(reference_state.name) if reference_state else None,
            "selected_family": None,
            "supporting_frames": [],
        })
    cell_results.sort(key=lambda item: tuple(item["coordinate"]))
    family_coverage = Counter(
        item["reference_family"]
        for item in cell_results
        if item.get("reference_family") is not None
    )

    files = {
        "baseline": baseline_path,
        "reference": reference,
        "proposals": proposals_path,
        "jar": jar,
        "camera_manifest": template / "cameras.json",
        "scene": template / "scene.json",
        **{f"frame:{name}": Path(item.source_path) for name, item in zip(frames, frames_info)},
        **{f"mask:{name}": path for name, path in mask_paths.items()},
    }
    report_data: dict[str, object] = {
        "schema": 2,
        "experiment": "M5-Template-2-matched-v2",
        "mode": "supplied-camera diagnostic",
        "production_promotion": False,
        "provenance": _git_provenance(root),
        "inputs": {key: {"path": str(path.resolve()), "sha256": _sha256(path)} for key, path in files.items()},
        "outputs": {
            "before": {"path": str(baseline_path.resolve()), "sha256": _sha256(baseline_path)},
            "after": {"path": str(output.resolve()), "sha256": _sha256(output)},
        },
        "effective_settings": {
            "palette": "VanillaJarTextureProvider.complete_palette",
            "palette_sha256": _palette_hash(palette),
            "palette_entries": len(palette),
            "non_full_palette_entries": len(non_full),
            "geometry_groups": len(shapes),
            "material_groups": len(materials),
            "family_eligibility": family_eligibility,
            "shape_fit_scale": RENDER_SCALE,
            "shape_fit_edge_weight": EDGE_WEIGHT,
            "shape_fit_texture_weight": TEXTURE_WEIGHT,
            "candidate_render_cap": max_candidate_renders,
            "proposal_order": "lexicographic unique v2 inventory",
            "usable_mask_policy": "frozen supplied JSON masks",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "texture_source": provider.source,
        },
        "processing_order": [list(coordinate) for coordinate in proposals],
        "baseline_blocks": len(before_map),
        "candidate_blocks": len(after_map),
        "exact_reference_cells": {"before": exact_before, "after": exact_after},
        "partial_family": {
            "reference_total": len(partial_reference),
            "exact_before": exact_partial_before,
            "exact_after": exact_partial_after,
            "recall_before": exact_partial_before / len(partial_reference) if partial_reference else 0.0,
            "recall_after": exact_partial_after / len(partial_reference) if partial_reference else 0.0,
        },
        "changes": {
            "additions": [list(c) for c in added],
            "deletions": [list(c) for c in deleted],
            "state_changes": [list(c) for c in changed],
            "lost_exact_cells": sum(
                before_map.get(c) == state and after_map.get(c) != state
                for c, state in reference_map.items()
            ),
            "phantoms_before": [list(c) for c in phantoms_before],
            "phantoms_after": [list(c) for c in phantoms_after],
        },
        "timing_seconds": {
            "shape_fit": elapsed,
            "total": elapsed,
        },
        "peak_python_heap_bytes": peak_bytes,
        "render_diagnostics": diagnostics,
        "cell_results": cell_results,
        "cell_family_coverage": dict(sorted(family_coverage.items())),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposals", dest="proposals_path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-candidate-renders", type=int, default=DEFAULT_MAX_CANDIDATE_RENDERS)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({
        "experiment": result["experiment"],
        "palette_entries": result["effective_settings"]["palette_entries"],
        "non_full_palette_entries": result["effective_settings"]["non_full_palette_entries"],
        "candidate_blocks": result["candidate_blocks"],
        "partial_family": result["partial_family"],
        "render_work": result["render_diagnostics"].get("shape_fit_render_work"),
        "seconds": result["timing_seconds"]["shape_fit"],
    }, indent=2))
