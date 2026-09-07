from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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


def _reconstruct_source(
    source: Path,
    output_root: Path,
    progress: Queue[tuple[str, int]],
    cancel_event: Event,
) -> tuple[str, int, Path, Path]:
    image_frames = list(discover_media([source]).image_frames)
    if not image_frames:
        raise ValueError("No supported screenshots found.")
    cameras_path = source / "cameras.json"
    scene_path = source / "scene.json"
    if len(image_frames) >= 2 and cameras_path.exists() and scene_path.exists():
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
                    palette = provider.palette()
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
    return output.name, len(volume.blocks), output, project_file


def run_desktop(project_path: Path | None = None) -> int:
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
    status = QLabel(f"Project: {project_path or 'New scan'}\nTarget: Minecraft Java {project.minecraft_version}")
    status.setWordWrap(True)
    selected_source: Path | None = None
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[tuple[str, int, Path, Path]] | None = None
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
        cancel_event.clear()
        progress_bar.setValue(0)
        choose_button.setEnabled(False)
        reconstruct_button.setEnabled(False)
        cancel_button.setEnabled(True)
        future = executor.submit(_reconstruct_source, selected_source, Path.cwd() / "exports", messages, cancel_event)

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
                filename, block_count, _, project_file = future.result()
                status.setText(f"Created {filename}\n{block_count} visible scene blocks\nReview low-confidence/background candidates before building.\nSaved {project_file.name}")
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
