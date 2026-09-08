"""Render reviewed evaluator-mask coverage overlays for visual audit.

The JSON files are the frozen source of truth. This helper only makes compact
red-on-photo thumbnails so a reviewer can check each exclusion before a scored
run. It deliberately has no image-content or static-pixel inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def mask_from_spec(path: Path) -> np.ndarray:
    specification = json.loads(path.read_text(encoding="utf-8"))
    width, height = (int(value) for value in specification["image_size"])
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in specification["excluded_rectangles"]:
        mask[int(y0):int(y1), int(x0):int(x1)] = True
    for polygon in specification.get("excluded_polygons", []):
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], True)
    return mask


def render(source: Path, specification: Path, output: Path, width: int = 960) -> float:
    image = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    excluded = mask_from_spec(specification)
    if excluded.shape != image.shape[:2]:
        raise ValueError(f"{specification} does not match {source}")
    image[excluded] = image[excluded] * 0.35 + np.asarray((255, 0, 0), dtype=np.float32) * 0.65
    height = round(image.shape[0] * width / image.shape[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS).save(output)
    return float(excluded.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("specifications", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for specification in sorted(args.specifications.glob("*.json")):
        source = args.images / f"{specification.stem}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        fraction = render(source, specification, args.output / f"{specification.stem}.png")
        print(f"{source.name}: excluded {fraction:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
