"""Run a reproducible, explicitly-scoped reconstruction diagnostic.

Fixture references, camera manifests, and scene bounds are never inferred to be
production inputs. This script labels supplied-camera and reference-assisted
runs as diagnostics; only a future image-only pipeline can qualify for the
production gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider
from mc_photogrammetry.evaluation import (
    best_visible_alignment,
    observation_kind_counts,
    visible_breakdowns,
    visible_metrics,
)
from mc_photogrammetry.export.compare import compare_litematics, load_litematic
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.export.visibility import full_cube_keys, shape_table, visible_cells
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.facade import DEFAULT_PALETTE, PaletteColor
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.reconstruction.texture_calibration import calibrate_appearance


MODES = ("image-only", "supplied-camera", "reference-assisted")


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
    usable_mask_paths: dict[str, Path] | None = None,
    correct_light_cast: bool = raycast.CORRECT_LIGHT_CAST,
) -> None:
    """Produce a diagnostic report without silently choosing fixture inputs."""
    if mode not in MODES:
        raise ValueError(f"unsupported benchmark mode: {mode}")
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
    usable_masks = _load_usable_masks(usable_mask_paths or {}, frames)
    input_seconds = time.perf_counter() - input_started

    palette = DEFAULT_PALETTE
    palette_source = "default"
    texture_source = None
    provider: VanillaJarTextureProvider | None = None
    available: dict[str, object] = {}
    palette_started = time.perf_counter()
    if jar is not None:
        provider = VanillaJarTextureProvider()
        if provider.load_jar(jar):
            palette = provider.canonical_palette()
            palette_source = "jar"
            texture_source = provider.source
            available = {sample.state.name: sample for sample in provider.samples}
    if not allow_default_palette and palette_source == "default":
        raise SystemExit(f"Palette resolved to DEFAULT_PALETTE for {template_dir}; pass --allow-default-palette to continue.")
    palette_seconds = time.perf_counter() - palette_started

    reference_started = time.perf_counter()
    reference_snapshot = load_litematic(reference)
    calibrations = None
    if mode == "reference-assisted":
        palette = tuple(
            PaletteColor(available[state.name].state, available[state.name].rgb)
            for state in reference_snapshot.palette if state.name in available
        )
        palette_source = "reference"
        texture_colors = {
            state: tuple(available[state.name].rgb)
            for state in reference_snapshot.palette if state.name in available
        }
        calibrations = {
            name: calibrate_appearance(frames[name], reference_snapshot, cameras[name], scene, texture_colors)
            for name in frames
        }
    reference_seconds = time.perf_counter() - reference_started

    diagnostics: dict[str, object] = {}
    reconstruction_started = time.perf_counter()
    volume = reconstruct_scene_surfaces(
        frames, cameras, scene, palette=palette, appearance_calibrations=calibrations,
        adjacency_strength=adjacency_strength, bounded_adjacency=bounded_adjacency, diagnostics=diagnostics,
        correct_light_cast=correct_light_cast,
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
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Mode: {mode} (diagnostic; does not qualify for image-only production)")
    print(f"Reference: {reference}")
    print(f"Candidate: {output}")
    print(f"Blocks: {comparison.candidate.non_air_count}")
    print(f"Visible exact-state recovery: {visible.exact_state_recovery:.2%}")
    print(f"Visible geometry F1: {visible.geometry_f1:.2%} (precision {visible.geometry_precision:.2%}, recall {visible.geometry_recall:.2%})")
    print(f"Conditional common-coordinate state accuracy: {comparison.state_accuracy:.2%}")
    print(f"Whole-reference exact rate: {comparison.whole_reference_state_accuracy:.2%}")
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
        usable_mask_policy="frozen supplied masks" if usable_masks else "all decoded pixels (provisional)",
        correct_light_cast=correct_light_cast,
    )
    report = {
        "report_schema": 2,
        "source": _source_provenance(),
        "mode": mode,
        "qualifies_image_only_production": False,
        "qualification_reason": "This run consumes a supplied fixture camera manifest and scene bounds.",
        "reference": _file_provenance(reference),
        "camera_manifest": _file_provenance(camera_manifest),
        "scene_definition": _file_provenance(scene_definition),
        "input_frames": {name: _file_provenance(Path(info.source_path)) for name, info in zip(frames, frames_info)},
        "usable_masks": {
            "policy": "frozen supplied masks" if usable_masks else "all decoded image pixels; no frozen mask was supplied",
            "frames": {
                name: _file_provenance(path)
                for name, path in sorted((usable_mask_paths or {}).items())
            },
            "usable_pixels": {name: int(mask.sum()) for name, mask in sorted(usable_masks.items())},
        },
        "palette_source": palette_source,
        "texture_source": texture_source,
        "palette_size": len(palette),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "minecraft_jar": _file_provenance(jar) if jar else None,
        "visibility": {
            **visible.to_dict(), "pixel_counts_by_reference_cell": sum(visibility.values()),
            "mask_policy": "frozen supplied masks" if usable_masks else "all decoded image pixels; no frozen fixture mask was supplied",
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
        "acceptance_gate": {
            "passed": False,
            "minimum_visible_exact_state_recovery": 0.80,
            "minimum_visible_geometry_f1": 0.90,
            "reason": "Only image-derived production runs may pass the promotion gate.",
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_provenance(path: Path) -> dict[str, object]:
    source = path.resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _load_usable_masks(paths: dict[str, Path], frames: dict[str, object]) -> dict[str, object]:
    """Load evaluator-owned masks and verify coverage.

    PNG masks use white for usable pixels. A tracked JSON specification freezes
    integer excluded rectangles for a fixed image size, avoiding a dependency on
    the mutable production GUI-mask heuristic.
    """
    if not paths:
        return {}
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


if __name__ == "__main__":
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
        "--correct-light-cast", action=argparse.BooleanOptionalAction,
        default=raycast.CORRECT_LIGHT_CAST,
        help="Use the image-only chroma-spread light-cast correction (enabled by default).",
    )
    parser.add_argument(
        "--usable-mask", action="append", default=[], metavar="FRAME=PATH",
        help="Frozen evaluator-owned white=usable image mask or tracked rectangle-spec JSON; provide one for every frame.",
    )
    args = parser.parse_args()
    run(
        args.template, args.output, mode=args.mode, reference=args.reference,
        camera_manifest=args.camera_manifest, scene_definition=args.scene, jar=args.minecraft_jar,
        adjacency_strength=args.adjacency_strength, report_path=args.report,
        allow_default_palette=args.allow_default_palette, bounded_adjacency=args.bounded_adjacency,
        usable_mask_paths=_parse_usable_masks(args.usable_mask),
        correct_light_cast=args.correct_light_cast,
    )
