from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_photogrammetry.domain.models import (
    SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
    ScanProject,
)
from mc_photogrammetry.export.anvil import AnvilExporter
from mc_photogrammetry.export.litematic import LitematicExportError, LitematicExporter
from mc_photogrammetry.export.compare import (
    _add,
    compare_litematics,
    load_litematic,
)
from mc_photogrammetry.export.validation import validate_litematic
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.project.store import (
    ProjectStoreError,
    load_project,
    save_project,
)
from mc_photogrammetry.reconstruction.facade import DEFAULT_PALETTE, reconstruct_facade
from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
from mc_photogrammetry.reconstruction.sfm import SfmSettings, reconstruct_from_calibrated_views
from mc_photogrammetry.reconstruction.multiview import reconstruct_visible_surfaces
from mc_photogrammetry.reconstruction.raycast import (
    ADJACENCY_STRENGTH,
    DEFAULT_PARTIAL_FIT_MAX_RENDERS,
    reconstruct_scene_surfaces,
)
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.pipeline.image_only import (
    LOCAL_FRAME_ORIGIN,
    ImageOnlyCameraError,
    camera_model_from_estimate,
    solve_image_cameras,
)
from mc_photogrammetry.pipeline.image_derived_entry import (
    DEFAULT_DIRECTION_TOLERANCE,
    DEFAULT_MAX_CYCLE_CHECKS,
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PAIRS,
    DEFAULT_MAX_TOTAL_SUPPORT,
    DEFAULT_ROTATION_TOLERANCE_DEGREES,
    ImageDerivedEntryError,
    build_image_derived_entry,
)
from mc_photogrammetry.pipeline.image_derived_production import (
    ImageDerivedProductionResult,
    derive_image_output_from_paths,
)
from mc_photogrammetry.pipeline.image_derived_production_delivery import (
    ImageDerivedProductionDeliveryResult,
    derive_and_deliver_image_output,
)
from mc_photogrammetry.pipeline.image_derived_delivery import (
    DeliveryRefusal,
    deliver_image_derived_litematic,
)
from mc_photogrammetry.pipeline.image_derived_production_record import (
    ProductionEnvelopeError,
    production_envelope_from_record,
)
from mc_photogrammetry.vision.gui_mask import create_gui_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mc-photogrammetry")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create an empty scan project")
    init.add_argument("path", type=Path)

    inspect = commands.add_parser("inspect", help="show scan project metadata")
    inspect.add_argument("path", type=Path)

    scan = commands.add_parser("scan", help="create a project from source media")
    scan.add_argument("output", type=Path)
    scan.add_argument("sources", type=Path, nargs="+")
    scan.add_argument("--recursive", action="store_true")

    reconstruct = commands.add_parser(
        "reconstruct", help="build a baseline façade from one still image"
    )
    reconstruct.add_argument("image", type=Path)
    reconstruct.add_argument("output", type=Path, nargs="?")
    reconstruct.add_argument("--project", type=Path)
    reconstruct.add_argument("--columns", type=int, default=32)
    reconstruct.add_argument("--rows", type=int, default=32)
    reconstruct.add_argument(
        "--depth-layers",
        type=int,
        default=8,
        help="maximum estimated camera-ray depth for visible surface blocks",
    )
    reconstruct.add_argument("--horizon", type=float, help="override normalized perspective horizon")
    reconstruct.add_argument(
        "--mode", choices=("normal", "developer"), default="normal",
        help="normal image-only mode, or developer mode for supplied camera/scene diagnostics",
    )
    reconstruct.add_argument("--camera-manifest", type=Path, help="camera JSON keyed by image filename")
    reconstruct.add_argument("--scene", type=Path, help="scene JSON for local/world coordinate conversion")
    reconstruct.add_argument("--minecraft-jar", type=Path, help="local Minecraft client JAR for texture palette")

    validate = commands.add_parser("validate", help="validate a litematic file")
    validate.add_argument("path", type=Path)

    multiview = commands.add_parser(
        "reconstruct-multiview", help="reconstruct a sparse 3D shell from calibrated screenshots"
    )
    multiview.add_argument("sources", type=Path)
    multiview.add_argument("output", type=Path, nargs="?")
    multiview.add_argument(
        "--mode", choices=("normal", "developer"), default="normal",
        help="normal image-only mode, or developer mode for supplied camera/scene diagnostics",
    )
    multiview.add_argument("--cameras", type=Path, help="developer-mode camera manifest")
    multiview.add_argument("--scene", type=Path, help="developer-mode scene bounds")
    multiview.add_argument("--max-reprojection-error", type=float, default=8.0)
    multiview.add_argument("--max-features", type=int, default=1600)

    surfaces = commands.add_parser(
        "reconstruct-visible-multiview",
        help="merge visible calibrated surface blocks from multiple screenshots",
    )
    surfaces.add_argument("sources", type=Path)
    surfaces.add_argument("output", type=Path, nargs="?")
    surfaces.add_argument(
        "--mode", choices=("normal", "developer"), default="normal",
        help="normal image-only mode, or developer mode for supplied camera/scene diagnostics",
    )
    surfaces.add_argument("--cameras", type=Path, help="developer-mode camera manifest")
    surfaces.add_argument("--scene", type=Path, help="developer-mode scene bounds")
    surfaces.add_argument("--columns", type=int, default=36)
    surfaces.add_argument("--rows", type=int, default=48)
    surfaces.add_argument("--depth-layers", type=int, default=8)
    surfaces.add_argument("--min-views", type=int, default=1)
    surfaces.add_argument("--max-color-distance", type=float, default=140.0)
    surfaces.add_argument("--min-palette-margin", type=float, default=0.0)
    surfaces.add_argument("--min-state-agreement", type=float, default=0.0)
    surfaces.add_argument("--min-surface-contrast", type=float, default=0.0)
    surfaces.add_argument("--include-enclosed", action="store_true")
    surfaces.add_argument("--adjacency-strength", type=float,
                          default=ADJACENCY_STRENGTH)
    surfaces.add_argument("--adjacency-min-neighbors", type=int, default=3)
    surfaces.add_argument(
        "--fit-partial-shapes", action="store_true",
        help="opt in to renderer-backed fitting of slabs, stairs, carpets and other partial models",
    )
    surfaces.add_argument(
        "--partial-fit-max-renders", type=int,
        default=DEFAULT_PARTIAL_FIT_MAX_RENDERS,
        help=("work cap for partial-shape candidate renders "
              f"(default: {DEFAULT_PARTIAL_FIT_MAX_RENDERS})"),
    )
    surfaces.add_argument("--minecraft-jar", type=Path, help="local Minecraft client JAR for texture palette")

    session = commands.add_parser(
        "image-derived-session",
        help="build a schema-2 image-derived session from ordinary images "
             "(geometry unavailable in normal mode)",
    )
    session.add_argument("sources", type=Path, nargs="+")
    session.add_argument("--project", type=Path, help="output .mcscan path")
    session.add_argument("--recursive", action="store_true")
    session.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help=f"maximum ordinary images to load (default: {DEFAULT_MAX_FRAMES})",
    )
    session.add_argument(
        "--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
        help=f"frame-pair budget for the connection graph (default: {DEFAULT_MAX_PAIRS})",
    )
    session.add_argument(
        "--max-total-support", type=int, default=DEFAULT_MAX_TOTAL_SUPPORT,
        help=("total support budget for the connection graph "
              f"(default: {DEFAULT_MAX_TOTAL_SUPPORT})"),
    )
    session.add_argument(
        "--max-edges", type=int, default=DEFAULT_MAX_EDGES,
        help=f"solved-edge budget for the pose graph (default: {DEFAULT_MAX_EDGES})",
    )
    session.add_argument(
        "--max-cycle-checks", type=int, default=DEFAULT_MAX_CYCLE_CHECKS,
        help=("fundamental-cycle check budget for the pose graph "
              f"(default: {DEFAULT_MAX_CYCLE_CHECKS})"),
    )
    session.add_argument(
        "--rotation-tolerance-degrees", type=float,
        default=DEFAULT_ROTATION_TOLERANCE_DEGREES,
        help=("largest accepted cycle rotation error in degrees "
              f"(default: {DEFAULT_ROTATION_TOLERANCE_DEGREES})"),
    )
    session.add_argument(
        "--direction-tolerance", type=float, default=DEFAULT_DIRECTION_TOLERANCE,
        help=("largest accepted cycle direction-hull distance "
              f"(default: {DEFAULT_DIRECTION_TOLERANCE})"),
    )

    output = commands.add_parser(
        "image-derived-output",
        help="derive typed geometry/state output from ordinary images",
    )
    output.add_argument("sources", type=Path, nargs="+")
    output.add_argument("--project", type=Path, help="output .mcscan audit path")
    output.add_argument("--recursive", action="store_true")
    output.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    output.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    output.add_argument("--max-total-support", type=int,
                        default=DEFAULT_MAX_TOTAL_SUPPORT)
    output.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    output.add_argument("--max-cycle-checks", type=int,
                        default=DEFAULT_MAX_CYCLE_CHECKS)
    output.add_argument("--rotation-tolerance-degrees", type=float,
                        default=DEFAULT_ROTATION_TOLERANCE_DEGREES)
    output.add_argument("--direction-tolerance", type=float,
                        default=DEFAULT_DIRECTION_TOLERANCE)

    deliver = commands.add_parser(
        "image-derived-deliver",
        help="derive from ordinary images and deliver exact promoted output",
    )
    deliver.add_argument("sources", type=Path, nargs="+")
    deliver.add_argument("--destination", type=Path, required=True)
    deliver.add_argument("--name", required=True)
    deliver.add_argument("--description", required=True)
    deliver.add_argument("--overwrite", action="store_true")
    deliver.add_argument("--project", type=Path, help="output .mcscan audit path")
    deliver.add_argument("--recursive", action="store_true")
    deliver.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    deliver.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    deliver.add_argument("--max-total-support", type=int,
                         default=DEFAULT_MAX_TOTAL_SUPPORT)
    deliver.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    deliver.add_argument("--max-cycle-checks", type=int,
                         default=DEFAULT_MAX_CYCLE_CHECKS)
    deliver.add_argument("--rotation-tolerance-degrees", type=float,
                         default=DEFAULT_ROTATION_TOLERANCE_DEGREES)
    deliver.add_argument("--direction-tolerance", type=float,
                         default=DEFAULT_DIRECTION_TOLERANCE)

    compare = commands.add_parser("compare", help="compare a candidate to a reference litematic")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--cameras", type=Path,
        help="camera manifest, so surface scores count only air a camera reached",
    )
    compare.add_argument("--scene", type=Path, help="scene manifest, with --cameras")
    compare.add_argument(
        "--attribute", action="store_true",
        help="say where the visible error is and why, by rendering both volumes "
             "(needs --cameras, --scene and --minecraft-jar)",
    )
    compare.add_argument("--minecraft-jar", type=Path, help="palette source for --attribute")

    export = commands.add_parser("export", help="export a saved project as litematic or world")
    export.add_argument("project", type=Path)
    export.add_argument("output", type=Path, nargs="?")
    export.add_argument(
        "--format",
        choices=("litematic", "world"),
        default="litematic",
        help="litematic needs a mod to view; world writes a save you can open directly",
    )
    export.add_argument(
        "--origin",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0, 0, 0),
        help="world position to place the reconstruction at",
    )

    visibility = commands.add_parser(
        "visibility",
        help="write a theoretical-maximum litematic: the reference restricted to "
             "the blocks some camera could actually see",
    )
    visibility.add_argument("reference", type=Path)
    visibility.add_argument("output", type=Path)
    visibility.add_argument("--cameras", type=Path, required=True)
    visibility.add_argument("--scene", type=Path, required=True)
    visibility.add_argument("--minecraft-jar", type=Path, required=True,
                            help="palette source, for block shapes")
    visibility.add_argument(
        "--min-pixels", type=int, default=1,
        help="pixels a block must win before it counts as reachable (default 1)",
    )

    desktop = commands.add_parser("desktop", help="launch the optional desktop shell")
    desktop.add_argument(
        "--mode", choices=("normal", "developer"), default="normal",
        help="normal image-only mode, or developer mode for supplied sidecar calibration",
    )
    return parser


def _export_schema3_project(args, project) -> int:
    """Export a schema-3 terminal only through the verified delivery contract.

    A schema-3 project is already an exact-state, digest-bound production
    terminal, so the generic ``export`` command must not re-materialize its
    volume directly.  It replays the persisted envelope, then routes the
    promoted result through the same dense-cell, no-clobber, typed-refusal and
    readback-verified delivery seam that the image-derived pipeline uses.
    """
    if args.format == "world":
        raise SystemExit(
            "export refused: a schema-3 image-derived project has no exact-state "
            "world export; export its verified litematic instead")
    try:
        envelope = production_envelope_from_record(
            project.image_derived_production)
    except ProductionEnvelopeError as error:
        raise SystemExit(f"export refused: {error}") from error
    if envelope.status != "promoted" or envelope.promotion is None:
        raise SystemExit(
            "export refused: the stored production terminal is "
            f"{envelope.status} ({envelope.reason})")
    output_path = args.output or (
        Path.cwd() / "exports" / f"{args.project.stem}.litematic")
    delivered = deliver_image_derived_litematic(
        envelope.promotion, output_path, name=project.name,
        description="Reconstructed Minecraft build")
    if isinstance(delivered, DeliveryRefusal):
        raise SystemExit(f"export refused: {delivered.reason}")
    print(f"Created {output_path}")
    return 0


def _print_camera_admission(session) -> None:
    summary = getattr(session, "admission_summary", None)
    if summary is None:
        return
    print(f"Original frames: {summary['original']}; admitted: {summary['admitted']}; excluded: {summary['excluded']}")
    for exclusion in summary["exclusions"]:
        print(f"Excluded {exclusion['frame_id']}: {exclusion['stage']}: {exclusion['reason']}")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init":
        save_project(ScanProject(), args.path)
        print(f"Created {args.path}")
        return 0

    if args.command == "visibility":
        from PIL import Image

        from mc_photogrammetry.domain.models import BlockObservation, BlockVolume
        from mc_photogrammetry.export.compare import load_litematic
        from mc_photogrammetry.export.visibility import (
            full_cube_keys,
            shape_table,
            theoretical_maximum,
            visible_cells,
        )
        from mc_photogrammetry.input.media import discover_media
        from mc_photogrammetry.reconstruction.camera import load_camera_manifest
        from mc_photogrammetry.reconstruction.scene import load_scene_transform

        provider = VanillaJarTextureProvider()
        if not provider.load_jar(args.minecraft_jar):
            print("Cannot compute visibility without a readable Minecraft jar")
            return 2
        palette = provider.complete_palette()
        reference = load_litematic(args.reference)
        scene = load_scene_transform(args.scene)
        cameras = load_camera_manifest(args.cameras)
        resolutions = {
            Path(frame.source_path).name: Image.open(frame.source_path).size
            for frame in discover_media([args.cameras.parent]).image_frames
        }
        counts = visible_cells(reference.blocks, scene, cameras, resolutions,
                               full_cubes=full_cube_keys(palette),
                               shapes=shape_table(palette), palette=palette)
        kept = theoretical_maximum(reference.blocks, counts,
                                   minimum_pixels=args.min_pixels)
        if not kept:
            print("No block was reached by any camera; nothing to write")
            return 1
        volume = BlockVolume()
        for coordinate, state in kept.items():
            volume.set(coordinate, BlockObservation(state=state, confidence=1.0))
        LitematicExporter().export(volume, args.output, name=args.output.stem,
                                   description="camera-reachable subset of the reference")
        total = len(reference.blocks)
        print(f"Reference: {total} blocks")
        print(f"Reachable: {len(kept)} ({len(kept) / total:.2%}) "
              f"at >= {args.min_pixels} pixel(s)")
        print(f"Unreachable: {total - len(kept)} blocks no camera can see")
        print(f"Wrote {args.output}")
        return 0

    if args.command == "desktop":
        from mc_photogrammetry.ui.main_window import run_desktop

        return run_desktop(developer_mode=args.mode == "developer")

    if args.command == "scan":
        manifest = discover_media(args.sources, recursive=args.recursive)
        if not manifest.frames:
            raise SystemExit("No supported image or video files were found.")
        project = ScanProject(source_paths=[frame.source_path for frame in manifest.frames])
        save_project(project, args.output)
        print(f"Created {args.output} with {len(manifest.frames)} source(s)")
        return 0

    if args.command == "image-derived-session":
        project_path = args.project or (
            Path.cwd() / "exports" / f"{args.sources[0].stem}.mcscan")
        try:
            result = build_image_derived_entry(
                args.sources,
                project_path=project_path,
                recursive=args.recursive,
                max_frames=args.max_frames,
                max_pairs=args.max_pairs,
                max_total_support=args.max_total_support,
                max_edges=args.max_edges,
                max_cycle_checks=args.max_cycle_checks,
                rotation_tolerance_degrees=args.rotation_tolerance_degrees,
                direction_tolerance=args.direction_tolerance,
            )
        except ImageDerivedEntryError as error:
            raise SystemExit(str(error)) from error
        print(f"Frames: {len(result.session.frames)}")
        _print_camera_admission(result.session)
        print(f"Session: {result.session.session_sha256}")
        print(f"Connections: {result.graph.status if result.graph is not None else 'not_run'}")
        print(f"Pose graph: {result.pose_graph.status if result.pose_graph is not None else 'not_run'}")
        print(f"Terminal: {result.terminal}")
        print("Geometry: unavailable from images alone in normal mode; no V1/V2/V3 "
              "geometry was fabricated.")
        print(f"Saved project {result.project_path}")
        return 0

    if args.command == "image-derived-output":
        project_path = args.project or (
            Path.cwd() / "exports" / f"{args.sources[0].stem}.mcscan")
        try:
            result = derive_image_output_from_paths(
                args.sources, project_path=project_path,
                recursive=args.recursive, max_frames=args.max_frames,
                max_pairs=args.max_pairs,
                max_total_support=args.max_total_support,
                max_edges=args.max_edges,
                max_cycle_checks=args.max_cycle_checks,
                rotation_tolerance_degrees=args.rotation_tolerance_degrees,
                direction_tolerance=args.direction_tolerance)
        except ImageDerivedEntryError as error:
            raise SystemExit(str(error)) from error
        if not isinstance(result, ImageDerivedProductionResult):
            print("Stage: refusal")
            print(f"Status: {result.reason}")
            print(f"Refusal: {result.refusal_sha256}")
            return 0
        _print_camera_admission(getattr(getattr(result, "entry", None), "session", None))
        print(f"Stage: {result.stage}")
        print(f"Status: {result.status}")
        print(f"Reason: {result.reason}")
        print(f"Entry: {result.entry_sha256}")
        if result.lattice_sha256:
            print(f"Lattice: {result.lattice_sha256}")
        if result.geometry_sha256:
            print(f"Geometry: {result.geometry_sha256}")
        if result.output_sha256:
            print(f"Output: {result.output_sha256}")
        print(f"Production: {result.result_sha256}")
        return 0

    if args.command == "image-derived-deliver":
        project_path = args.project or (
            Path.cwd() / "exports" / f"{args.sources[0].stem}.mcscan")
        try:
            result = derive_and_deliver_image_output(
                args.sources, destination=args.destination, name=args.name,
                description=args.description, overwrite=args.overwrite,
                project_path=project_path, recursive=args.recursive,
                max_frames=args.max_frames, max_pairs=args.max_pairs,
                max_total_support=args.max_total_support,
                max_edges=args.max_edges,
                max_cycle_checks=args.max_cycle_checks,
                rotation_tolerance_degrees=args.rotation_tolerance_degrees,
                direction_tolerance=args.direction_tolerance)
        except ImageDerivedEntryError as error:
            raise SystemExit(str(error)) from error
        if not isinstance(result, ImageDerivedProductionDeliveryResult):
            print(json.dumps({"stage": "refusal", "status": result.reason,
                              "refusal_sha256": result.refusal_sha256},
                             sort_keys=True, separators=(",", ":")))
            return 0
        admission = getattr(getattr(getattr(getattr(result, "production", None), "entry", None), "session", None), "admission_summary", None)
        print(json.dumps({
            "stage": "delivery", "status": result.status,
            "reason": result.reason,
            "production_sha256": result.production_sha256,
            "delivery_sha256": result.delivery_sha256,
            "result_sha256": result.result_sha256,
            **({"camera_admission": admission} if admission is not None else {})},
            sort_keys=True, separators=(",", ":")))
        return 0

    if args.command == "reconstruct":
        if args.mode != "developer":
            manual_inputs = [
                name
                for name, supplied in (
                    ("--horizon", args.horizon is not None),
                    ("--camera-manifest", args.camera_manifest is not None),
                    ("--scene", args.scene is not None),
                )
                if supplied
            ]
            if manual_inputs:
                raise SystemExit(
                    "horizon, camera and scene metadata are developer diagnostics; "
                    "rerun with --mode developer "
                    f"({', '.join(manual_inputs)})"
                )
        export_directory = Path.cwd() / "exports"
        output_path = args.output or export_directory / f"{args.image.stem}.litematic"
        image = load_image(args.image)
        mask = create_gui_mask(image.shape[0], image.shape[1])
        camera = None
        scene_transform = None
        if args.mode == "developer":
            # The visibility/compare branches import these names locally, which
            # makes them local to `main` for the whole function; importing them
            # here as well keeps this branch off the (shadowed) module binding.
            if args.camera_manifest:
                from mc_photogrammetry.reconstruction.camera import load_camera_manifest

                cameras = load_camera_manifest(args.camera_manifest)
                camera = cameras.get(args.image.name)
                if camera is None:
                    raise SystemExit(f"No camera pose found for {args.image.name}")
            if args.scene:
                from mc_photogrammetry.reconstruction.scene import load_scene_transform

                scene_transform = load_scene_transform(args.scene)
        else:
            frame_id = args.image.name
            try:
                solution = solve_image_cameras({frame_id: image})
                estimate = solution.cameras.get(frame_id)
                if estimate is None:
                    raise SystemExit(
                        f"Image-only camera recovery returned no camera for {frame_id}"
                    )
                camera = camera_model_from_estimate(
                    estimate, position=LOCAL_FRAME_ORIGIN
                )
            except ImageOnlyCameraError as error:
                raise SystemExit(
                    f"Image-only camera recovery failed for {frame_id}: {error}"
                ) from error
        palette = DEFAULT_PALETTE
        if args.minecraft_jar:
            provider = VanillaJarTextureProvider()
            count = provider.load_jar(args.minecraft_jar)
            if count:
                palette = provider.canonical_palette()
            else:
                print("Warning: local Minecraft jar palette is unavailable; falling back to the default palette.")
            print(f"Loaded {count} local vanilla texture samples")
        volume = reconstruct_facade(
            image,
            mask,
            columns=args.columns,
            rows=args.rows,
            depth_layers=args.depth_layers,
            horizon=args.horizon,
            camera=camera,
            scene_transform=scene_transform,
            palette=palette,
        )
        if not volume.blocks:
            raise SystemExit("No visible blocks were inferred from the image.")
        project = ScanProject(name=args.image.stem, source_paths=[str(args.image.resolve())], volume=volume)
        project_path = args.project or output_path.with_suffix(".mcscan")
        save_project(project, project_path)
        description = (
            "Visible surface blocks with estimated depth; hidden geometry omitted"
            if args.mode == "developer"
            else "Image-only visible surfaces in a local frame; metric scale and "
            "translation unresolved"
        )
        LitematicExporter().export(
            volume,
            output_path,
            name=args.image.stem,
            description=description,
        )
        if args.mode == "developer":
            print("Mode: developer diagnostic (supplied camera/scene metadata)")
        else:
            print(
                "Camera: image-only solve in a local frame; metric scale and "
                "translation remain unresolved."
            )
        print(f"Created {output_path} with {len(volume.blocks)} block(s)")
        print("Visible-surface mode: depth estimated per visible block; hidden geometry omitted.")
        print(f"Saved project {project_path}")
        return 0

    if args.command == "reconstruct-multiview":
        if args.mode != "developer":
            raise SystemExit(
                "automatic camera recovery is unresolved; reconstruct-multiview requires --mode developer"
            )
        if args.cameras is None or args.scene is None:
            raise SystemExit("developer mode requires explicit --cameras and --scene paths")
        manifest = discover_media([args.sources])
        image_frames = [frame for frame in manifest.image_frames]
        if len(image_frames) < 2:
            raise SystemExit("At least two still images are required for multiview reconstruction.")
        frames = {Path(frame.source_path).name: load_image(frame.source_path) for frame in image_frames}
        cameras = load_camera_manifest(args.cameras)
        scene = load_scene_transform(args.scene)
        volume = reconstruct_from_calibrated_views(
            frames,
            cameras,
            scene=scene,
            settings=SfmSettings(
                max_features=args.max_features,
                max_reprojection_error=args.max_reprojection_error,
            ),
            palette=tuple((item.state, item.rgb) for item in DEFAULT_PALETTE),
        )
        if not volume.blocks:
            raise SystemExit("No stable 3D block observations were reconstructed.")
        output_path = args.output or Path.cwd() / "exports" / f"{args.sources.name}.litematic"
        project_path = output_path.with_suffix(".mcscan")
        project = ScanProject(name=args.sources.name, source_paths=[frame.source_path for frame in image_frames], volume=volume)
        save_project(project, project_path)
        LitematicExporter().export(
            volume,
            output_path,
            name=args.sources.name,
            description="Calibrated multi-view sparse 3D reconstruction; low-confidence shell",
        )
        print("Mode: developer diagnostic (supplied cameras and scene)")
        print(f"Created {output_path} with {len(volume.blocks)} reconstructed block(s)")
        print(f"Saved project {project_path}")
        return 0

    if args.command == "reconstruct-visible-multiview":
        if args.mode != "developer":
            raise SystemExit(
                "automatic camera recovery is unresolved; reconstruct-visible-multiview requires --mode developer"
            )
        if args.cameras is None or args.scene is None:
            raise SystemExit("developer mode requires explicit --cameras and --scene paths")
        manifest = discover_media([args.sources])
        image_frames = list(manifest.image_frames)
        cameras = load_camera_manifest(args.cameras)
        missing = [Path(frame.source_path).name for frame in image_frames if Path(frame.source_path).name not in cameras]
        if len(image_frames) < 2:
            raise SystemExit("At least two still images are required for visible multiview reconstruction.")
        if missing:
            raise SystemExit(f"No camera pose found for: {', '.join(missing)}")
        frames = {Path(frame.source_path).name: load_image(frame.source_path) for frame in image_frames}
        scene = load_scene_transform(args.scene)
        palette = DEFAULT_PALETTE
        shape_palette = None
        if args.minecraft_jar:
            provider = VanillaJarTextureProvider()
            count = provider.load_jar(args.minecraft_jar)
            if count:
                # Classify ordinary surfaces against full-cube appearances.
                # Partial models enter only through the explicit renderer-backed
                # fitting pass, where projected geometry supplies occupancy evidence.
                palette = provider.canonical_palette()
                if args.fit_partial_shapes:
                    shape_palette = provider.complete_palette()
            else:
                print("Warning: local Minecraft jar palette is unavailable; falling back to the default palette.")
            print(f"Loaded {count} local vanilla texture samples")
        volume = reconstruct_scene_surfaces(
            frames,
            cameras,
            scene=scene,
            palette=palette,
            min_views=args.min_views,
            columns=args.columns,
            rows=args.rows,
            depth_layers=args.depth_layers,
            surface_only=not args.include_enclosed,
            adjacency_strength=args.adjacency_strength,
            adjacency_min_neighbors=args.adjacency_min_neighbors,
            fit_partial_shapes=args.fit_partial_shapes,
            shape_palette=shape_palette,
            partial_fit_max_renders=args.partial_fit_max_renders,
        )
        if not volume.blocks:
            raise SystemExit("No visible surface blocks landed inside the calibrated scene.")
        output_path = args.output or Path.cwd() / "exports" / f"{args.sources.name}-visible.litematic"
        project_path = output_path.with_suffix(".mcscan")
        project = ScanProject(name=args.sources.name, source_paths=[frame.source_path for frame in image_frames], volume=volume)
        save_project(project, project_path)
        LitematicExporter().export(
            volume,
            output_path,
            name=args.sources.name,
            description="Calibrated visible surfaces only; hidden geometry omitted",
        )
        print("Mode: developer diagnostic (supplied cameras and scene)")
        print(f"Created {output_path} with {len(volume.blocks)} visible block(s)")
        print(f"Saved project {project_path}")
        return 0

    if args.command == "validate":
        result = validate_litematic(args.path)
        print(f"Valid litematic: {args.path}")
        print(f"Dimensions: {result['dimensions']}")
        print(f"Palette entries: {result['palette_size']}")
        return 0

    if args.command == "compare":
        camera_cells = None
        if args.cameras and args.scene:
            import numpy as np

            from mc_photogrammetry.reconstruction.camera import load_camera_manifest
            from mc_photogrammetry.reconstruction.scene import load_scene_transform

            scene = load_scene_transform(args.scene)
            camera_cells = []
            for camera in load_camera_manifest(args.cameras).values():
                eye = np.asarray(camera.position, dtype=float) + np.asarray(
                    [0.0, camera.eye_height, 0.0]
                )
                camera_cells.append(
                    tuple(int(v) for v in np.floor(scene.to_local(tuple(eye))))
                )
        result = compare_litematics(
            args.reference, args.candidate, camera_cells=camera_cells
        )
        print(f"Reference: {result.reference.size}, {result.reference.non_air_count} non-air blocks")
        print(f"Candidate: {result.candidate.size}, {result.candidate.non_air_count} non-air blocks")
        print(f"Coordinate overlap: {result.exact_coordinate_matches} ({result.coordinate_recall:.2%})")
        print(f"Coordinate precision: {result.coordinate_precision:.2%}, F1: {result.coordinate_f1:.2%}")
        print(f"Direct coordinate overlap: {result.direct_coordinate_matches} ({result.direct_coordinate_recall:.2%})")
        print(f"Surface F1: {result.surface_f1:.2%} "
              f"(precision {result.surface_precision:.2%}, recall {result.surface_recall:.2%}, "
              f"{result.reference_exposed} seeable)")
        print(f"State matches: {result.exact_state_matches} ({result.state_accuracy:.2%})")
        print(f"Best translation: {result.best_translation}")
        print(f"Palette name overlap: {result.candidate_palette_matches}/{len(result.candidate.palette)}")
        if args.attribute:
            if not (args.cameras and args.scene and args.minecraft_jar):
                print("--attribute needs --cameras, --scene and --minecraft-jar")
                return 2
            from mc_photogrammetry.export.attribution import attribute_error
            from mc_photogrammetry.input.media import discover_media

            provider = VanillaJarTextureProvider()
            if not provider.load_jar(args.minecraft_jar):
                print("Cannot attribute error without a readable Minecraft jar")
                return 2
            folder = args.cameras.parent
            frames = {
                Path(frame.source_path).name: load_image(frame.source_path)
                for frame in discover_media([folder]).image_frames
            }
            reference = load_litematic(args.reference)
            candidate = load_litematic(args.candidate)
            shifted = {
                _add(coordinate, result.best_translation): state
                for coordinate, state in candidate.blocks.items()
            }
            report = attribute_error(
                reference.blocks, shifted, load_scene_transform(args.scene),
                load_camera_manifest(args.cameras), frames,
                provider.complete_palette(),
            )
            print()
            print("Visible error by cause:")
            for cause, share in report.share().items():
                print(f"  {cause:11} {share:6.1%}  ({report.pixels.get(cause, 0)} px)")
            print("Worst blocks:")
            for name, share in report.worst_blocks():
                print(f"  {share:6.1%}  {name}")
            print("Worst confusions:")
            for wanted, obtained, share in report.worst_confusions():
                print(f"  {share:6.1%}  {wanted} -> {obtained}")
        return 0

    if args.command == "export":
        try:
            project = load_project(args.project)
        except ProjectStoreError as error:
            raise SystemExit(f"export refused: {error}") from error
        if project.schema_version == SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION:
            return _export_schema3_project(args, project)
        if args.format == "world":
            output_path = args.output or Path.cwd() / "exports" / f"{args.project.stem}.zip"
            AnvilExporter().export(
                project.volume, output_path, name=project.name, origin=tuple(args.origin)
            )
            print(f"Created {output_path}")
            print("Unzip it into your saves folder and open it from the world list.")
            return 0
        output_path = args.output or Path.cwd() / "exports" / f"{args.project.stem}.litematic"
        try:
            LitematicExporter().export(project.volume, output_path, name=project.name)
        except LitematicExportError as error:
            raise SystemExit(f"export refused: {error}") from error
        print(f"Created {output_path}")
        return 0

    project = load_project(args.path)
    print(f"Name: {project.name}")
    print(f"Minecraft: {project.minecraft_version}")
    print(f"Sources: {len(project.source_paths)}")
    print(f"Blocks: {len(project.volume.blocks)}")
    return 0
