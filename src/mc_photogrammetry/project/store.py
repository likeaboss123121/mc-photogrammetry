from __future__ import annotations

import json
from pathlib import Path

from mc_photogrammetry.domain.models import ScanProject


class ProjectStoreError(ValueError):
    """Raised when a scan project cannot be read or written."""


def save_project(project: ScanProject, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(project.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError as error:
        raise ProjectStoreError(f"could not save project to {destination}: {error}") from error


def load_project(path: str | Path) -> ScanProject:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectStoreError(f"could not load project from {source}: {error}") from error
    if not isinstance(data, dict):
        raise ProjectStoreError("project root must be a JSON object")
    try:
        return ScanProject.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectStoreError(f"invalid project data in {source}: {error}") from error
