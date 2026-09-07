from __future__ import annotations

from pathlib import Path

from mc_photogrammetry.ui.main_window import run_desktop


def main() -> int:
    return run_desktop(Path("scan.mcscan") if Path("scan.mcscan").exists() else None)


if __name__ == "__main__":
    raise SystemExit(main())
