# Minecraft Photogrammetry

This project attempted to reconstruct Minecraft builds from screenshots or video and export `.litematic` or world ZIP files. It is a **failed research prototype**, not a working or supported product. The project did not meet its image-only reconstruction goals; ordinary screenshots do not reliably produce a correct build. There is no official release or user support.

The original proof of concept targeted Minecraft Java **1.21.5**. That is a development target, not a verified compatibility guarantee. The codebase contains experiments, partial reconstruction paths, evaluation tools, exporters, and historical artifacts. An exporter or passing unit test does not establish that image reconstruction works end to end.

## Project status and evidence

The failure handoff summarizes the original scope, what was implemented, the final evidence, limitations, and lessons learned:

- [Project failure handoff](AGENT_HANDOFF.md)
- [Curated postmortem screenshots](docs/failure-postmortem/screenshots/)
- [Archived previous handoff](docs/archive/AGENT_HANDOFF-pre-failure-2026-10-11.md)

The last camera/matcher experiment had zero correct accepted camera edges across six views and did not reach block reconstruction. Other image-only geometry proposals failed their evaluation criteria. No qualifying image-only `.litematic` or visible accuracy score was demonstrated at project close. Some older benchmark results used supplied camera poses, fixture references, or other assistance; they are not autonomous product performance.

## Development environment

The Python package targets Python 3.10 or newer and uses `uv` in the current development instructions. To explore the prototype:

```powershell
uv sync --extra dev
uv run pytest
```

The optional desktop dependencies are installed separately with `uv sync --extra desktop`. See `pyproject.toml` for dependency groups and entry points. Test results cover code behavior and contracts; they do not measure reconstruction accuracy.

The source is under `src/mc_photogrammetry/`, tests under `tests/`, and diagnostic/evaluation scripts under `scripts/`. The dated reports in `docs/reports/` are research records; check each report's fixture, assistance, and status before relying on its metrics. Some captures contain F3/debug overlays and world coordinates for evaluation. Those are not normal-mode inputs and may disclose details of the original test worlds.

## License

There is currently no `LICENSE` file. Public visibility does not grant permission to reuse, modify, or redistribute this code. Add a license only after the project owner chooses one.
