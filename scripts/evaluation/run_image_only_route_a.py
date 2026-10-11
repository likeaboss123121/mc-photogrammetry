"""Evaluation-only Route A replay runner for the A1 image-only frontier.

The default arm replays the *normal-mode* single-view ``reconstruct`` CLI sequence on
frozen inputs, calling the same production functions in the same order:

    load_image -> create_gui_mask -> solve_image_cameras
    -> camera_model_from_estimate(position=LOCAL_FRAME_ORIGIN)
    -> reconstruct_facade(palette=run palette)
    -> save_project (schema-2 .mcscan) -> LitematicExporter -> load_litematic / validate_litematic

It then normalizes the solved camera, the facade ``BlockVolume``, the palette the
run used and the decoded export ``BlockVolume`` through the pure A1
``route_a_frontier`` adapters, so the camera/facade/palette/decoded-export records
exist without changing ``cli.py``.

Evaluation-only and explicitly image-only: it never accepts a camera manifest,
scene transform, F3/directions, bounds, coordinates or other manual reconstruction
metadata, and it reads no mask, reference or JAR.  The optional ``--palette-jar``
flag reproduces the *existing normal CLI* palette option only by passing the JAR
to the accepted vanilla texture provider; leave it unset and no JAR is opened.
The explicit pixel-grid arm keeps ordinary camera admission, then discards the
solved camera only at the facade call. It is an evaluation-only geometry prior
with unresolved scale, not normal CLI parity. The runner produces no visible-truth number.

Typical frozen-input invocation::

    .\\.venv\\Scripts\\python.exe scripts\\evaluation\\run_image_only_route_a.py ^
        --image "templates 2\\2026-09-02_13.22.14.png" --workdir .tmp-a1-route-a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE  # noqa: E402
from mc_photogrammetry.domain.models import (  # noqa: E402
    BlockObservation,
    BlockVolume,
    ObservationKind,
    ScanProject,
)
from mc_photogrammetry.evaluation_image_only_frontier import (  # noqa: E402
    FRONTIER_PRODUCER,
    FrontierRefusal,
    canonical_json,
    first_observable_failure,
    route_a_frontier,
)
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.export.litematic import LitematicExporter  # noqa: E402
from mc_photogrammetry.export.validation import validate_litematic  # noqa: E402
from mc_photogrammetry.input.image import load_image  # noqa: E402
from mc_photogrammetry.pipeline.image_only import (  # noqa: E402
    LOCAL_FRAME_ORIGIN,
    ImageOnlyCameraError,
    camera_model_from_estimate,
    solve_image_cameras,
)
from mc_photogrammetry.project.store import save_project  # noqa: E402
from mc_photogrammetry.reconstruction.facade import reconstruct_facade  # noqa: E402
from mc_photogrammetry.vision.gui_mask import create_gui_mask  # noqa: E402


ARMS = ("camera-projected", "pixel-grid")


@dataclass(frozen=True, slots=True)
class RouteASettings:
    arm: str = "camera-projected"
    columns: int = 32
    rows: int = 32
    depth_layers: int = 8
    horizon: None = None
    scene_transform: None = None
    scale_known: bool = False


@dataclass(frozen=True, slots=True)
class RouteARun:
    """Every production record one Route A replay produced, plus its frontier."""

    route_a: object
    image: object
    mask: object
    camera: object
    facade: object
    palette: tuple
    decoded_export: object
    project_path: Path
    litematic_path: Path
    camera_error: str = ""
    settings: RouteASettings = RouteASettings()
    candidate_seal: str = ""


def _volume_digest(states: dict) -> str:
    """Canonical digest of one ``{coordinate: BlockState}`` comparison map."""
    payload = [[list(coordinate), state.to_dict()]
               for coordinate, state in sorted(states.items())]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")).hexdigest()


def _translation_offset(left: dict, right: dict) -> tuple[int, int, int]:
    """The one permitted global integer translation between two volumes."""
    if not left or not right:
        return (0, 0, 0)
    left_min = tuple(min(coordinate[axis] for coordinate in left)
                     for axis in range(3))
    right_min = tuple(min(coordinate[axis] for coordinate in right)
                      for axis in range(3))
    return tuple(left_min[axis] - right_min[axis] for axis in range(3))


def _decoded_volume(path: Path):
    """Decode one exported ``.litematic`` into the scored ``BlockVolume`` input.

    ``load_litematic`` returns the region's own coordinates together with the
    region origin, so the caller can compare in one frame under the single
    permitted global integer translation.
    """
    validate_litematic(path)
    snapshot = load_litematic(path)
    volume = BlockVolume()
    for coordinate, state in sorted(snapshot.blocks.items()):
        volume.set(coordinate, BlockObservation(
            state=state, kind=ObservationKind.INFERRED))
    return volume, snapshot.position


def replay(
    image_path: str | Path,
    *,
    workdir: str | Path | None = None,
    columns: int = 32,
    rows: int = 32,
    depth_layers: int = 8,
    palette_jar: str | Path | None = None,
    arm: str = "camera-projected",
) -> RouteARun:
    """Replay the normal CLI Route A sequence on one frozen image."""
    if arm not in ARMS:
        raise FrontierRefusal("unknown Route A geometry arm")
    if arm == "pixel-grid" and ((columns, rows, depth_layers) != (32, 32, 8)
                                or palette_jar is not None):
        raise FrontierRefusal("pixel-grid ablation requires fixed 32x32x8/default palette")
    settings = RouteASettings(arm, columns, rows, depth_layers)
    source = Path(image_path)
    if not source.is_file():
        raise FrontierRefusal(f"the frozen image is unavailable: {source}")
    directory = (Path(workdir) if workdir is not None
                 else Path(tempfile.mkdtemp(prefix="imgloss-a1-route-a-")))
    directory.mkdir(parents=True, exist_ok=True)
    project_path = directory / f"{source.stem}.mcscan"
    litematic_path = directory / f"{source.stem}.litematic"

    image = load_image(source)
    mask = create_gui_mask(image.shape[0], image.shape[1])
    camera = None
    camera_error = ""
    frame_id = source.name
    try:
        solution = solve_image_cameras({frame_id: image})
        estimate = solution.cameras.get(frame_id)
        if estimate is None:
            raise ImageOnlyCameraError(
                f"image-only camera recovery returned no camera for {frame_id}")
        camera = camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)
    except ImageOnlyCameraError as error:
        # The normal CLI raises SystemExit here (cli.py:548-551).  The runner
        # records the same typed failure and leaves the camera checkpoint
        # unavailable instead of exiting, so the frontier has a typed stop.
        camera_error = f"ImageOnlyCameraError: {error}"
        camera = None

    palette = DEFAULT_PALETTE
    if palette_jar is not None:
        from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider

        provider = VanillaJarTextureProvider()
        count = provider.load_jar(palette_jar)
        if count:
            palette = tuple(provider.canonical_palette())

    facade = None
    decoded_export = None
    blocked = "camera_unavailable" if camera is None else ""
    if camera is not None:
        facade = reconstruct_facade(
            image, mask, columns=columns, rows=rows, depth_layers=depth_layers,
            horizon=None, camera=camera if arm == "camera-projected" else None,
            scene_transform=None, palette=palette)
        if not facade.blocks:
            facade = None
            blocked = "no_visible_blocks"
        else:
            save_project(
                ScanProject(name=source.stem,
                            source_paths=[str(source.resolve())], volume=facade),
                project_path)
            LitematicExporter().export(
                facade, litematic_path, name=source.stem,
                description=("Image-only visible surfaces in a local frame; metric "
                             "scale and translation unresolved"))
            decoded_export, _decoded_origin = _decoded_volume(litematic_path)

    route_a = route_a_frontier(
        admission=image, camera=camera, facade=facade, palette=palette,
        decoded_export=decoded_export, camera_solved=camera is not None,
        blocked_reason=blocked)
    # Freeze native identities/settings before returning to the evaluator. This
    # immutable JSON string cannot be changed by later mutation of image/volume.
    states = {} if facade is None else {
        coordinate: observation.state for coordinate, observation in facade.blocks.items()}
    repo = Path(__file__).resolve().parents[2]
    code_paths = (
        "scripts/evaluation/run_image_only_route_a.py",
        "scripts/evaluation/report_image_only_attribution.py",
        "src/mc_photogrammetry/reconstruction/facade.py",
        "src/mc_photogrammetry/pipeline/image_only.py",
        "src/mc_photogrammetry/vision/gui_mask.py",
        "src/mc_photogrammetry/blocks/appearance.py",
    )
    assignments = tuple((coordinate[0], rows - coordinate[1] - 1, coordinate)
                        for coordinate in sorted(states)) if arm == "pixel-grid" else ()
    seal = canonical_json({
        "settings": asdict(settings),
        "camera_admission": "solved" if camera is not None else "refused",
        "camera_error": camera_error,
        "camera": None if camera is None else camera.to_dict(),
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gui_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "palette": [item.state.to_dict() for item in palette],
        "facade_volume_sha256": _volume_digest(states),
        "facade_cell_count": len(states),
        # The no-camera branch emits exactly one distinct (column,row) key per
        # admitted sample; its retained cell count therefore counts samples.
        "admitted_pixel_samples": len(states) if arm == "pixel-grid" and facade is not None else None,
        "pixel_sample_collisions": 0 if assignments else None,
        "collision_evidence": "retained_injective_grid_sample_assignments" if assignments
        else "control_sample_assignments_not_retained",
        "pixel_sample_assignments": [[column, row, list(cell)]
                                     for column, row, cell in assignments],
        "native_artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (project_path, litematic_path) if path.is_file()},
        "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                        for name in code_paths},
        "evaluator_access": "not_started",
        "normal_cli_parity_eligible": arm == "camera-projected",
    })
    return RouteARun(
        route_a=route_a, image=image, mask=mask, camera=camera, facade=facade,
        palette=tuple(palette), decoded_export=decoded_export,
        project_path=project_path, litematic_path=litematic_path,
        camera_error=camera_error, settings=settings, candidate_seal=seal)


def run(
    image_path: str | Path,
    *,
    workdir: str | Path | None = None,
    report_path: str | Path | None = None,
    columns: int = 32,
    rows: int = 32,
    depth_layers: int = 8,
    palette_jar: str | Path | None = None,
    arm: str = "camera-projected",
) -> dict:
    """Run Route A once and return its frontier + production-record summary."""
    outcome = replay(
        image_path, workdir=workdir, columns=columns, rows=rows,
        depth_layers=depth_layers, palette_jar=palette_jar, arm=arm)
    routed = outcome.route_a
    frontier = routed.frontier
    facade_blocks = {} if outcome.facade is None else dict(outcome.facade.blocks)
    decoded_blocks = ({} if outcome.decoded_export is None
                      else dict(outcome.decoded_export.blocks))
    facade_states = {coordinate: observation.state
                     for coordinate, observation in facade_blocks.items()}
    # The litematic writer stores its region with a local origin, so the decoded
    # volume is compared under the one permitted global integer translation: the
    # offset between the two volumes' own minimum corners.  Block identity and
    # state are then compared exactly.
    decoded_states = {coordinate: observation.state
                      for coordinate, observation in decoded_blocks.items()}
    origin = list(_translation_offset(facade_states, decoded_states))
    decoded_states = {tuple(coordinate[index] + origin[index]
                            for index in range(3)): state
                      for coordinate, state in decoded_states.items()}

    summary = {
        "producer": FRONTIER_PRODUCER,
        "route": "route_a",
        "candidate_seal": json.loads(outcome.candidate_seal),
        "frontier_available": True,
        "terminal_stage": frontier.terminal_policy["stage"],
        "terminal_status": frontier.terminal_status,
        "terminal_reason": frontier.terminal_reason,
        "image": str(Path(image_path)),
        "camera_solved": outcome.camera is not None,
        "camera_error": outcome.camera_error,
        "camera_position": (None if outcome.camera is None
                            else list(outcome.camera.position)),
        "palette_size": len(outcome.palette),
        "facade_block_count": len(facade_blocks),
        "decoded_block_count": len(decoded_blocks),
        # The facade -> export round trip is only known when the run actually
        # exported and decoded; otherwise the check is typed unavailable (null).
        "export_round_trip_exact": (
            None if outcome.facade is None or outcome.decoded_export is None
            else facade_states == decoded_states),
        "facade_volume_sha256": _volume_digest(facade_states),
        "decoded_volume_sha256": _volume_digest(decoded_states),
        "decoded_region_origin": list(origin),
        "project_path": str(outcome.project_path),
        "litematic_path": str(outcome.litematic_path),
        "frontier": frontier.to_record(),
        "first_observable_failure": first_observable_failure(frontier),
        "notes": ("typed frontier/status only; unknown metric scale is reported "
                  "as a score-reducing caveat, never as an accuracy number"),
    }
    if report_path is not None:
        Path(report_path).write_text(canonical_json(summary) + "\n",
                                     encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A1 evaluation-only Route A image-only frontier runner")
    parser.add_argument("--arm", choices=ARMS, default="camera-projected",
                        help="evaluation-only geometry prior; default matches normal CLI")
    parser.add_argument("--image", required=True,
                        help="one frozen image path for the single-view route")
    parser.add_argument("--workdir", default=None,
                        help="directory for the .mcscan/.litematic outputs")
    parser.add_argument("--report", default=None,
                        help="optional JSON report output path")
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--rows", type=int, default=32)
    # The normal-mode ``reconstruct`` CLI defaults are 32x32x8 (cli.py:94-101);
    # the runner mirrors them so its default invocation is the same one a normal
    # user gets and the frozen CLI-vs-runner parity check needs no overrides.
    parser.add_argument("--depth-layers", type=int, default=8)
    parser.add_argument("--palette-jar", default=None,
                        help="existing normal-CLI palette option; unset opens no JAR")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            args.image, workdir=args.workdir, report_path=args.report,
            columns=args.columns, rows=args.rows,
            depth_layers=args.depth_layers, palette_jar=args.palette_jar, arm=args.arm)
    except (FrontierRefusal, ValueError) as error:
        print(canonical_json({
            "producer": FRONTIER_PRODUCER, "route": "route_a",
            "frontier_available": False,
            "reason": type(error).__name__,
            "detail": str(error),
            "frontier": None,
            "notes": "typed refusal; no frontier was built",
        }))
        return 2
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
