# Minecraft Photogrammetry

A Windows-first desktop application for turning screenshots, image sets, and video of Minecraft Java builds into an editable, confidence-scored block model.

The first implementation targets Minecraft 1.21.x and `.litematic` export. Reconstruction is intentionally explicit about uncertainty: visible geometry is observed, while inferred or hidden geometry is marked separately.

## Development

Requires Python 3.10 or newer. With `uv` installed:

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run pyright
```

Install the optional desktop shell with `uv sync --extra desktop`, then launch it with:

```powershell
uv run mc-photogrammetry desktop
```

The desktop app accepts a screenshot folder. When that folder contains `cameras.json` and `scene.json`, it automatically uses the calibrated visible-scene reconstruction path and writes the result to `exports`.

Reconstruction runs off the UI thread. The desktop window shows stage progress and provides a cancel button while screenshots, textures, and scene views are processed.
The local Minecraft texture/state index is cached beside the client JAR and invalidated when the JAR changes, so later launches avoid rebuilding the full index.
The desktop path keeps one-view visible candidates so it favors complete visible coverage; low-confidence/background candidates still require review before building.

You can also double-click `launch.bat` or run `./launch.ps1` from PowerShell in the project folder. The launcher installs/synchronizes the optional desktop dependencies when `uv` is available.
When `uv` is unavailable, the launchers prefer the project `.venv` before falling back to global Python.
The checked-in launcher environment can be prepared manually with `py -m venv .venv`, followed by installing the dependencies from `pyproject.toml`; the current local `.venv` has been validated with Python 3.10, PySide6, OpenCV, Pillow, NumPy, and nbtlib.

The initial CLI can create and inspect a project file:

```powershell
uv run mc-photogrammetry init scan.mcscan
uv run mc-photogrammetry inspect scan.mcscan
uv run mc-photogrammetry validate build.litematic
```

Create a project from source media:

```powershell
uv run mc-photogrammetry scan scan.mcscan .\captures --recursive
```

### Exporting as a world instead of a litematic

A `.litematic` needs Litematica or a similar mod to view. A reconstruction can
also be written as an ordinary Minecraft save, which needs nothing:

```powershell
py -m mc_photogrammetry export build.mcscan --format world
py -m mc_photogrammetry export build.mcscan --format world --origin 100586 167 11680
```

This writes a `.zip` containing `level.dat` and the region files. Unzip it into
your `saves` folder and the world appears in the world list, spawning you at the
build. `--origin` places the reconstruction at the world coordinates it was
scanned from, so it lands where it was found; without it the volume is written
around `(0, 0, 0)`. The world generates as a void superflat, so nothing surrounds
the reconstruction and what you see is exactly what was reconstructed.

When no output path is supplied to `reconstruct`, the `.litematic` and `.mcscan` files are written to the project-root `exports` folder. The desktop app uses the same folder. You can still provide an explicit output path to the CLI.

The current core also discovers supported images and video sources into a stable manifest. Video decoding and frame sampling are the next pipeline layer.

Decoded still frames can now be quality-scored and filtered with perceptual hashes before reconstruction. Video decoding will feed the same selector API.

Video decoding is now available through a lazy OpenCV adapter with frame stride, timestamps, and bounded memory usage. Install the core dependencies before using it.

Material classification can use a local Minecraft client JAR without copying assets into the project:

```powershell
py -m mc_photogrammetry reconstruct-visible-multiview .\templates .\exports\build.litematic --cameras .\templates\cameras.json --scene .\templates\scene.json --minecraft-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar"
```

A deterministic single-image grid-depth baseline is available: it samples visible cells, excludes GUI-mask pixels, maps colors to a small vanilla palette, estimates a perspective horizon, maps screen distance to discrete depth, and fills contiguous camera-ray columns without shearing neighboring image columns. Front faces are observed and behind-face blocks are marked inferred. It is a best-effort single-view estimate, not guaranteed hidden geometry.

Visible-surface mode is the default. It emits one block per accepted image cell at an estimated camera-ray depth. It never fills blocks behind that visible hit; `--depth-layers` controls the maximum depth estimate, not the number of fabricated blocks.

Calibrated screenshot folders can use the sparse multi-view path:

```powershell
py -m mc_photogrammetry reconstruct-multiview .\templates .\exports\template-sfm.litematic --cameras .\templates\cameras.json --scene .\templates\scene.json
```

This path requires matching camera entries for the image filenames and reports only stable triangulated observations; it does not fill unseen interiors automatically.

The calibrated multiview prototype is intentionally conservative. Its current output is a sparse shell while camera-pose refinement is pending; it should be treated as a diagnostic reconstruction path, not a completed replacement for the single-image workflow.

For calibrated screenshots where you want visible surfaces only, use `reconstruct-visible-multiview`. It merges one camera-aware surface estimate per screenshot and never fills hidden blocks. Its default 36x48 sampling is intentionally finer than the single-image 32x32 fallback.

A compact three-view regression fixture is also available in `templates 2`. Its camera poses are in `templates 2/cameras.json`, its world transform is in `templates 2/scene.json`, and its reference is `Template 2.litematic`.

Generated litematic files can be checked with the independent structural validator used by the test suite before importing them into Minecraft.

The supplied reference can be compared with a generated file using:

```powershell
py -m mc_photogrammetry compare ".\templates\Photogrammetry Template.litematic" ".\exports\build.litematic"
```

The supplied world bounds are recorded in `templates/scene.json`. The reference litematic's local origin `(0, 0, 0)` maps to world `(-236, 45, 248)` and its exclusive size is `(161, 47, 180)`.

The scene-grid path is currently high-recall but not 80%-accurate end to end: on `templates 2`, the measured coordinate recall is 92.49%, precision 44.54%, F1 60.13%, and exact block-state accuracy 1.14%. These metrics are generated automatically so later occupancy and texture-classification changes can be judged against the same reference.

Surface-only pruning is now enabled by default. On the compact fixture it produces 86.14% coordinate F1 (81.69% recall, 91.10% precision). Fixture-only appearance calibration plus the known reference palette reaches 56.32% exact state accuracy, so the geometry target is met but the end-to-end 80% target is not.

The benchmark script also accepts `--reference-palette` for fixture-only diagnosis. This restricts classification to states known to exist in the ground-truth litematic and is not used by normal scans.
The benchmark's calibrated diagnostic uses a conservative adjacency strength of `0.25`, selected from the compact fixture sweep; it preserves material gradients better than stronger smoothing.

To re-export an existing project after exporter updates, use:

```powershell
py -m mc_photogrammetry export .\exports\build.mcscan
```

The first exporter targets single-region `.litematic` files for Minecraft 1.21.x. It mirrors the supplied native Version 7 reference shape: top-level `Version`, `SubVersion`, and `MinecraftDataVersion` tags directly in the root compound, plus native metadata and region fields. World ZIP export will come later.

Model weights, Minecraft assets, source media, scan projects, and generated exports are local user data and are not committed to this repository.
