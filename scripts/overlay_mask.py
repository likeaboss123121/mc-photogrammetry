"""Mask the pixels that never change between views of a scene.

PyCOLMAP classified every Template 2 pair as `WATERMARK` - its name for a static
overlay - and refused to recover a pose from any of them, despite finding over
two hundred geometric inliers in each. The cause is the F3 debug overlay: it is
drawn identically in every screenshot of a session, so hundreds of features match
at *exactly* the same coordinates, and a large set of zero-displacement
correspondences is indistinguishable from a watermark because it is one.

The overlay is not part of the scene. A pixel whose value is the same from every
viewpoint has no parallax and cannot contribute to pose, so excluding it removes
no evidence. This finds those pixels from the frames themselves rather than
assuming where the overlay sits, which also catches whatever else happens to be
painted on top - a crosshair, a hotbar, a HUD element this project has not met.

That it is needed at all is an artefact of the fixtures: production screenshots
have no debug overlay, so masking brings the experiment *closer* to the real
case rather than further from it.

Writes COLMAP-style masks, where black excludes a pixel::

    .venv/Scripts/python.exe scripts/overlay_mask.py "templates 2" --out masks/templates-2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TOLERANCE = 2
"""Levels of difference still counted as unchanged.

Screenshots of the same overlay are bit-identical, so this only absorbs the
rounding of a colour conversion. Raising it starts masking dark scene pixels
that merely look similar.
"""

DILATION = 5
"""Pixels of growth around each masked region.

A feature is described by the neighbourhood around it, so a keypoint sitting
just outside the overlay still has the overlay inside its descriptor.
"""


def static_mask(images, tolerance: int = TOLERANCE) -> np.ndarray:
    """Pixels whose value is the same in every frame."""
    stack = np.stack([image.astype(np.int16) for image in images])
    return (stack.max(axis=0) - stack.min(axis=0)) <= tolerance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=TOLERANCE)
    parser.add_argument("--dilation", type=int, default=DILATION)
    args = parser.parse_args(argv)

    import cv2

    folder = ROOT / args.fixture
    paths = sorted(folder.glob("*.png"))
    if len(paths) < 2:
        raise SystemExit("need at least two frames to tell static from moving")

    greys, colour = [], {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        colour[path.name] = image
        greys.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if len(greys) < 2:
        raise SystemExit("could not read enough frames")

    static = static_mask(greys, args.tolerance)
    if args.dilation > 0:
        kernel = np.ones((args.dilation, args.dilation), np.uint8)
        static = cv2.dilate(static.astype(np.uint8), kernel).astype(bool)

    keep = (~static).astype(np.uint8) * 255
    args.out.mkdir(parents=True, exist_ok=True)
    for name in colour:
        # COLMAP looks for a mask named after the image with .png appended;
        # the plain name is written too so either convention finds it.
        cv2.imwrite(str(args.out / f"{name}.png"), keep)
        cv2.imwrite(str(args.out / name), keep)

    summary = {
        "fixture": args.fixture,
        "frames": len(greys),
        "tolerance": args.tolerance,
        "dilation": args.dilation,
        "masked_fraction": float(static.mean()),
        "kept_fraction": float((~static).mean()),
    }
    (args.out / "mask-summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"{args.fixture}: masked {static.mean():.1%} of each frame "
          f"({len(colour)} masks written to {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
