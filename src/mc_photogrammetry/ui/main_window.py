from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
from mc_photogrammetry.domain.models import ScanProject
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.input.image import load_image
from mc_photogrammetry.input.media import discover_media
from mc_photogrammetry.reconstruction.camera import load_camera_manifest
from mc_photogrammetry.reconstruction.facade import DEFAULT_PALETTE, reconstruct_facade
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import load_scene_transform
from mc_photogrammetry.vision.gui_mask import create_gui_mask


@dataclass(frozen=True, slots=True)
class _DesktopResult:
    status: str
    output: Path | None
    project: Path | None
    reason: str = ""
    block_count: int | None = None
    diagnostic_confirmed: bool = False
    camera_admission: dict | None = None


def _camera_admission_text(summary: dict | None) -> str:
    if summary is None:
        return ""
    lines = [f"Screenshots: {summary['original']} original, {summary['admitted']} admitted, {summary['excluded']} excluded"]
    lines.extend(f"Excluded {item['frame_id']}: {item['stage']}: {item['reason']}"
                 for item in summary["exclusions"])
    return "\n".join(lines) + "\n"


def _format_desktop_result(result: _DesktopResult) -> str:
    admission = _camera_admission_text(result.camera_admission)
    if result.status == "delivered" and result.output is not None:
        count = (f"{result.block_count} visible scene blocks\n"
                 if result.block_count is not None else "")
        diagnostic = (
            f"Diagnostic: {result.project}\n"
            if result.project is not None and result.diagnostic_confirmed
            else ""
        )
        return (
            f"{admission}Created {result.output.name}\n{count}{diagnostic}"
            "Review low-confidence/background candidates before building."
        )
    if result.project is None:
        diagnostic = "No schema-3 diagnostic path is available."
    elif result.diagnostic_confirmed:
        diagnostic = f"Schema-3 diagnostic: {result.project}"
    else:
        diagnostic = (
            f"Schema-3 diagnostic target (not confirmed for this run): "
            f"{result.project}"
        )
    return (
        f"{admission}Reconstruction {result.status}: {result.reason or 'no exact export'}\n"
        f"{diagnostic}"
    )


def _reconstruct_source(
    source: Path,
    output_root: Path,
    progress: Queue[tuple[str, int]],
    cancel_event: Event,
    *,
    developer_mode: bool = False,
    destination: Path | None = None,
) -> _DesktopResult:
    image_frames = list(discover_media([source]).image_frames)
    if destination is not None and len(image_frames) < 2:
        return _DesktopResult(
            "refused", None, destination.with_suffix(".mcscan"),
            "insufficient_images_for_multiview",
        )
    if not image_frames:
        raise ValueError("No supported screenshots found.")
    cameras_path = source / "cameras.json"
    scene_path = source / "scene.json"
    # Sidecar calibration is an advanced diagnostic input. Normal desktop mode
    # must not discover or consume neighboring manifests implicitly.
    if developer_mode and len(image_frames) >= 2 and cameras_path.exists() and scene_path.exists():
        progress.put(("Loading calibrated screenshots", 5))
        cameras = load_camera_manifest(cameras_path)
        filenames = [Path(frame.source_path).name for frame in image_frames]
        missing = [filename for filename in filenames if filename not in cameras]
        if missing:
            raise ValueError(f"Missing camera poses for: {', '.join(missing)}")
        frames = {filename: load_image(frame.source_path) for filename, frame in zip(filenames, image_frames)}
        scene = load_scene_transform(scene_path)
        palette = DEFAULT_PALETTE
        for version in ("1.21.5", "1.21.1"):
            jar = Path.home() / "AppData" / "Roaming" / ".minecraft" / "versions" / version / f"{version}.jar"
            if jar.exists():
                progress.put(("Loading Minecraft textures", 10))
                provider = VanillaJarTextureProvider()
                if provider.load_jar(jar):
                    # Keep the main classifier on canonical full-cube
                    # appearances. Partial models are renderer candidates only
                    # when an explicit shape-fitting stage requests them.
                    palette = provider.canonical_palette()
                else:
                    progress.put(("Warning: default palette fallback; local Minecraft jar textures unavailable", 15))
                break
        total = len(frames)
        volume = reconstruct_scene_surfaces(
            frames,
            cameras,
            scene,
            palette=palette,
            min_views=1,
            progress_callback=lambda current, total_work: progress.put(
                (f"Reconstructing scene ({current:,} / {total_work:,})", 10 + int(80 * current / total_work))
            ),
            cancel_event=cancel_event,
        )
        name = source.name
        source_paths = [frame.source_path for frame in image_frames]
    elif len(image_frames) >= 2:
        if destination is None:
            raise ValueError("Choose a .litematic destination for this screenshot set.")
        from mc_photogrammetry.pipeline.image_derived_production_delivery import (
            derive_and_deliver_image_output,
        )
        from mc_photogrammetry.pipeline.image_derived_production import (
            ImageDerivedProductionResult,
        )

        image_paths = [Path(frame.source_path) for frame in image_frames]
        project_file = destination.with_suffix(".mcscan")
        progress.put((f"Reconstructing {len(image_paths)} screenshots", 5))
        result = derive_and_deliver_image_output(
            image_paths,
            destination=destination,
            name=destination.stem,
            description="Image-derived visible scene; hidden geometry omitted",
            project_path=project_file,
        )
        production = getattr(result, "production", result)
        entry = getattr(production, "entry", None)
        admission = getattr(getattr(entry, "session", None), "admission_summary", None)
        if admission is not None:
            progress.put((_camera_admission_text(admission).rstrip(), 95))
        diagnostic = Path(entry.project_path) if entry is not None else project_file
        diagnostic_confirmed = isinstance(production, ImageDerivedProductionResult)
        status = getattr(result, "status", "refused")
        reason = getattr(result, "reason", "")
        if status == "delivered":
            progress.put(("Complete", 100))
            return _DesktopResult(
                status, destination, diagnostic, reason,
                diagnostic_confirmed=diagnostic_confirmed,
                camera_admission=admission,
            )
        progress.put((f"Reconstruction {status}: {reason or 'no exact export'}", 100))
        return _DesktopResult(
            status, None, diagnostic, reason,
            diagnostic_confirmed=diagnostic_confirmed,
            camera_admission=admission,
        )
    else:
        selected = Path(image_frames[0].source_path)
        progress.put(("Loading screenshot", 10))
        image = load_image(selected)
        volume = reconstruct_facade(image, create_gui_mask(image.shape[0], image.shape[1]), depth_layers=8)
        name = selected.stem
        source_paths = [str(selected.resolve())]
    if not volume.blocks:
        raise ValueError("No visible blocks were reconstructed.")
    progress.put(("Writing project and litematic", 95))
    output = output_root / f"{name}.litematic"
    project_file = output_root / f"{name}.mcscan"
    from mc_photogrammetry.project.store import save_project
    save_project(ScanProject(name=name, source_paths=source_paths, volume=volume), project_file)
    LitematicExporter().export(volume, output, name=name, description="Calibrated visible scene surfaces; hidden geometry omitted")
    progress.put(("Complete", 100))
    return _DesktopResult("delivered", output, project_file,
                          block_count=len(volume.blocks))


def run_desktop(project_path: Path | None = None, *, developer_mode: bool = False) -> int:
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget
    except ImportError as error:
        raise RuntimeError("The desktop UI requires the optional dependencies. Install them with: uv sync --extra desktop") from error

    application = QApplication.instance() or QApplication([])
    project = ScanProject()
    window = QMainWindow()
    window.setWindowTitle("Minecraft Photogrammetry")
    window.resize(960, 640)
    content = QWidget()
    layout = QVBoxLayout(content)
    title = QLabel("Minecraft Photogrammetry")
    title.setStyleSheet("font-size: 24px; font-weight: 600;")
    mode_label = "developer diagnostic (sidecar calibration allowed)" if developer_mode else "normal image-only mode"
    status = QLabel(
        f"Project: {project_path or 'New scan'}\n"
        f"Target: Minecraft Java {project.minecraft_version}\nMode: {mode_label}"
    )
    status.setWordWrap(True)
    selected_source: Path | None = None
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[_DesktopResult] | None = None
    cancel_event = Event()
    messages: Queue[tuple[str, int]] = Queue()
    choose_button = QPushButton("Choose screenshot folder")
    reconstruct_button = QPushButton("Reconstruct and export")
    cancel_button = QPushButton("Cancel")
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    cancel_button.setEnabled(False)

    def choose_source() -> None:
        nonlocal selected_source
        directory = QFileDialog.getExistingDirectory(window, "Choose screenshot folder")
        if directory:
            selected_source = Path(directory)
            count = len(discover_media([selected_source]).image_frames)
            status.setText(f"Selected: {selected_source}\n{count} screenshot(s) found.")

    def start_reconstruction() -> None:
        nonlocal future
        if selected_source is None:
            status.setText("Choose a screenshot folder first.")
            return
        if future is not None and not future.done():
            return
        destination: Path | None = None
        frames = discover_media([selected_source]).image_frames
        has_developer_sidecars = (
            developer_mode and len(frames) >= 2
            and (selected_source / "cameras.json").exists()
            and (selected_source / "scene.json").exists()
        )
        if len(frames) >= 2 and not has_developer_sidecars:
            default_output = selected_source.parent / f"{selected_source.name}.litematic"
            filename, _ = QFileDialog.getSaveFileName(
                window, "Save reconstructed litematic", str(default_output),
                "Litematic files (*.litematic)",
            )
            if not filename:
                return
            destination = Path(filename)
            if destination.suffix.casefold() != ".litematic":
                destination = destination.with_suffix(".litematic")
        cancel_event.clear()
        progress_bar.setValue(0)
        choose_button.setEnabled(False)
        reconstruct_button.setEnabled(False)
        # The image-derived production API has no cancellation hook. Keep the
        # existing cancel control for legacy/developer reconstruction only.
        cancel_button.setEnabled(destination is None)
        future = executor.submit(
            _reconstruct_source, selected_source, Path.cwd() / "exports", messages,
            cancel_event, developer_mode=developer_mode, destination=destination,
        )

    def cancel_reconstruction() -> None:
        cancel_event.set()
        status.setText("Cancellation requested...")

    def poll_progress() -> None:
        nonlocal future
        try:
            while True:
                message, value = messages.get_nowait()
                status.setText(message)
                progress_bar.setValue(value)
        except Empty:
            pass
        if future is not None and future.done():
            try:
                status.setText(_format_desktop_result(future.result()))
            except Exception as error:
                status.setText(f"Reconstruction failed: {error}")
            finally:
                future = None
                choose_button.setEnabled(True)
                reconstruct_button.setEnabled(True)
                cancel_button.setEnabled(False)

    choose_button.clicked.connect(choose_source)
    reconstruct_button.clicked.connect(start_reconstruction)
    cancel_button.clicked.connect(cancel_reconstruction)
    timer = QTimer(window)
    timer.timeout.connect(poll_progress)
    timer.start(100)
    layout.addWidget(title)
    layout.addWidget(status)
    layout.addWidget(choose_button)
    layout.addWidget(reconstruct_button)
    layout.addWidget(cancel_button)
    layout.addWidget(progress_bar)
    layout.addStretch()
    window.setCentralWidget(content)
    window.show()
    exit_code = application.exec()
    executor.shutdown(wait=False, cancel_futures=True)
    return exit_code
