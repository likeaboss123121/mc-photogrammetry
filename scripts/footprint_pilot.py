"""Save a reviewable bundle for the footprint adapter's synthetic control.

Two subcommands, deliberately run as two separate processes so "reload"
cannot lean on any in-memory state "build" happened to still have:

    python scripts/footprint_pilot.py build --out-dir <dir>
    python scripts/footprint_pilot.py reload --out-dir <dir>

``build`` projects the known 4x3 control (10 visible / 1 opening / 1
foreground-occluded cell) through a declared synthetic camera to get
image-space region polygons - standing in for a person's drawn polygons on a
photograph - runs them through the adapter and the accepted `footprint.py`
placement, and writes:

* ``transform.json`` - the whole `LatticeTransform`, plus the image size and
  a SHA-256 of its own serialized content, so a later invocation can prove it
  read exactly this transform rather than a silently-edited one.
* ``regions.json`` - the region polygons, similarly hashed.
* ``evidence.json`` - every candidate cell's coverage by label, its declared
  label (or none), and the resolved geometry (`supported`/`opening`/
  `unknown`) - the sidecar `footprint.py` itself does not write to disk.
* ``control.litematic`` - the real exported geometry, solids only (see
  `footprint_to_volume`'s `include_confirmed_openings`).
* ``overlay.png`` - the projected cell grid coloured by resolved geometry,
  with the region polygons drawn under it.

``reload`` reads only ``transform.json`` and ``regions.json`` back, recomputes
everything from them in this fresh process, and checks that the recovered
non-air block-coordinate set and the opening/unknown cell labels are
identical to what ``build`` recorded in ``evidence.json`` - the actual test of
whether the saved transform is sufficient on its own, not a rerun that
happens to reuse the same Python process's state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS, COLUMNS = 3, 4
OPENING_CELL = (1, 2)
OCCLUDED_CELL = (2, 0)


def _control_declared(VISIBLE, OPENING, FOREGROUND_OCCLUSION):
    declared = {(r, c): VISIBLE for r in range(ROWS) for c in range(COLUMNS)}
    declared[OPENING_CELL] = OPENING
    declared[OCCLUDED_CELL] = FOREGROUND_OCCLUSION
    return declared


def _content_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def build(out_dir: Path) -> None:
    from mc_photogrammetry.reconstruction.calibration import axis_directions
    from mc_photogrammetry.reconstruction.footprint import (
        FOREGROUND_OCCLUSION, GEOMETRY_OPENING, OPENING, SUPPORTED, UNKNOWN,
        VISIBLE, build_footprint, cell_to_block, footprint_to_volume)
    from mc_photogrammetry.reconstruction.footprint_adapter import (
        LatticeTransform, RegionEvidence, cell_corners_pixels,
        declared_labels_from_evidence, rasterize_cells)
    from mc_photogrammetry.export.litematic import LitematicExporter

    out_dir.mkdir(parents=True, exist_ok=True)

    axes = axis_directions(5.0, -3.0, 2.0)
    probe = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                             focal=800.0, principal_point=(0.0, 0.0),
                             lateral_origin=(0.37, -0.62), period=1.0)
    xs, ys = [], []
    for row in range(ROWS):
        for column in range(COLUMNS):
            corners = cell_corners_pixels(probe, (row, column))
            xs += [p[0] for p in corners]
            ys += [p[1] for p in corners]
    margin = 60.0
    principal_point = (-min(xs) + margin, -min(ys) + margin)
    image_size = (int(max(xs) - min(xs) + 2 * margin), int(max(ys) - min(ys) + 2 * margin))
    transform = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                                 focal=800.0, principal_point=principal_point,
                                 lateral_origin=(0.37, -0.62), period=1.0)

    declared = _control_declared(VISIBLE, OPENING, FOREGROUND_OCCLUSION)
    regions = []
    for cell, label in declared.items():
        corners = cell_corners_pixels(transform, cell)
        regions.append(RegionEvidence(label, tuple(corners)))

    transform_record = {
        "axis": transform.axis, "side": transform.side, "distance": transform.distance,
        "axis_matrix": transform.axis_matrix.tolist(), "focal": transform.focal,
        "principal_point": list(transform.principal_point),
        "lateral_origin": list(transform.lateral_origin), "period": transform.period,
        "image_size": list(image_size),
        "face_coordinate": 1,
    }
    transform_record["content_sha256"] = _content_hash(transform_record)
    (out_dir / "transform.json").write_text(json.dumps(transform_record, indent=2), encoding="utf-8")

    regions_record = {"regions": [{"label": r.label, "polygon": [list(p) for p in r.polygon]}
                                  for r in regions]}
    regions_record["content_sha256"] = _content_hash(regions_record)
    (out_dir / "regions.json").write_text(json.dumps(regions_record, indent=2), encoding="utf-8")

    evidence = rasterize_cells(transform, regions, image_size=image_size,
                              rows=range(ROWS), columns=range(COLUMNS))
    recovered_labels = declared_labels_from_evidence(evidence)
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(0, 0), rows=ROWS, columns=COLUMNS,
                               declared=recovered_labels)

    evidence_record = {
        "cells": [{"cell": list(e.cell), "coverage_by_label": e.coverage_by_label,
                   "coverage_of_in_frame": e.coverage_of_in_frame,
                   "declared": e.declared, "outside_image_fraction": e.outside_image_fraction,
                   "conflict_fraction": e.conflict_fraction}
                  for e in evidence],
        "resolved": [{"cell": list(c.cell), "declared": c.declared, "geometry": c.geometry}
                    for c in footprint.cells],
        "supported_blocks": sorted(list(cell_to_block(footprint, c.cell))
                                   for c in footprint.of(SUPPORTED)),
        "opening_cells": [list(c.cell) for c in footprint.of(GEOMETRY_OPENING)],
        "unknown_cells": [list(c.cell) for c in footprint.of(UNKNOWN)],
    }
    (out_dir / "evidence.json").write_text(json.dumps(evidence_record, indent=2), encoding="utf-8")

    volume = footprint_to_volume(footprint)
    LitematicExporter().export(volume, out_dir / "control.litematic",
                               name="footprint pilot control",
                               description="M4 B synthetic image-adapter control")

    reprojected, minimum = _readback_reprojection(out_dir / "control.litematic", footprint, transform)
    export_transform_record = {
        "minimum_shift": list(minimum),
        "axis": transform.axis, "side": transform.side,
        "face_coordinate": 1,
        "lateral_origin": [0, 0], "period": 1.0,
        "note": "the export-local-to-reconstruction transform: how a block "
                "coordinate actually read out of the litematic file maps back "
                "to a cell on transform.json's camera-space lattice. "
                "minimum_shift is added to a read-back local block coordinate "
                "before block_to_cell (see export/litematic.py - a "
                "single-region file always stores Position (0,0,0) and its "
                "shape local to its own minimum corner). face_coordinate/"
                "lateral_origin/period are the arbitrary local export "
                "convention cell_to_block/block_to_cell use, distinct from "
                "the continuous camera transform.",
    }
    export_transform_record["content_sha256"] = _content_hash(export_transform_record)
    (out_dir / "export_transform.json").write_text(
        json.dumps(export_transform_record, indent=2), encoding="utf-8")
    _write_overlay(out_dir / "overlay.png", transform, regions, footprint, image_size,
                   reprojected=reprojected)

    print(f"supported={len(footprint.of(SUPPORTED))} opening={len(footprint.of(GEOMETRY_OPENING))} "
          f"unknown={len(footprint.of(UNKNOWN))}")
    print(f"wrote transform.json, regions.json, evidence.json, export_transform.json, "
         f"control.litematic, overlay.png to {out_dir}")


def _readback_reprojection(litematic_path: Path, footprint, transform):
    """Read the exported litematic back and independently reproject the
    blocks it actually contains through the saved transform - see
    `footprint_adapter.reproject_exported_volume`'s own docstring for why
    this has to trace back through the file's own block values rather than
    the pre-export footprint's `cell_to_block`/`block_to_cell` round trip
    (a gap a lead follow-up review named directly in an earlier version of
    this check). Returns ``(reprojected, minimum)``.
    """
    from mc_photogrammetry.reconstruction.footprint_adapter import reproject_exported_volume

    reprojected, minimum = reproject_exported_volume(transform, footprint, litematic_path)
    print(f"read-back verification OK: {len(reprojected)} solid blocks in "
         f"{litematic_path.name} independently reproject to their pre-export cells")
    return reprojected, minimum


def _write_overlay(path, transform, regions, footprint, image_size, *, reprojected=None):
    """The candidate grid, coloured by resolved geometry - but every
    `supported` (green) cell's drawn polygon comes from `reprojected`
    (`_readback_reprojection`'s return value: corners computed by taking a
    block straight out of the exported litematic file and reprojecting it)
    when given, not from calling `cell_corners_pixels` on the in-memory
    footprint's own cell list. Opening/unknown cells have nothing in the
    file to read back (opening is omitted by default, unknown always is),
    so those still come from the in-memory footprint - labelled as such
    rather than implied to be file-verified too.
    """
    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.footprint import SUPPORTED
    from mc_photogrammetry.reconstruction.footprint_adapter import cell_corners_pixels

    width, height = image_size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    region_colours = {"visible": (210, 235, 210), "opening": (210, 210, 250),
                      "foreground_occlusion": (200, 225, 250)}
    for region in regions:
        points = np.array([[round(x), round(y)] for x, y in region.polygon], dtype=np.int32)
        cv2.fillPoly(canvas, [points], region_colours.get(region.label, (230, 230, 230)))

    reprojected = reprojected or {}
    verdict_colours = {SUPPORTED: (60, 180, 60), "opening": (60, 60, 220), "unknown": (140, 140, 140)}
    by_cell = {c.cell: c for c in footprint.cells}
    for row in range(ROWS):
        for column in range(COLUMNS):
            cell = (row, column)
            geometry = by_cell[cell].geometry
            file_derived = cell in reprojected
            corners = reprojected[cell] if file_derived else cell_corners_pixels(transform, cell)
            points = np.array([[round(x), round(y)] for x, y in corners], dtype=np.int32)
            colour = verdict_colours.get(geometry, (0, 0, 0))
            thickness = 4 if file_derived else 2
            cv2.polylines(canvas, [points], True, colour, thickness)
            centre = (int(sum(p[0] for p in points) / 4), int(sum(p[1] for p in points) / 4))
            label = f"{row},{column}" + (" (file)" if file_derived else "")
            cv2.putText(canvas, label, centre, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if reprojected:
        cv2.putText(canvas,
                   "thick outline = corners reprojected from the exported litematic file's own blocks",
                   (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def reload_and_verify(out_dir: Path) -> None:
    import numpy as np

    from mc_photogrammetry.reconstruction.footprint import (
        GEOMETRY_OPENING, SUPPORTED, UNKNOWN, build_footprint, cell_to_block)
    from mc_photogrammetry.reconstruction.footprint_adapter import (
        LatticeTransform, RegionEvidence, declared_labels_from_evidence, rasterize_cells)

    transform_record = json.loads((out_dir / "transform.json").read_text(encoding="utf-8"))
    saved_hash = transform_record.pop("content_sha256")
    assert _content_hash(transform_record) == saved_hash, "transform.json content changed since it was saved"
    regions_record = json.loads((out_dir / "regions.json").read_text(encoding="utf-8"))
    saved_regions_hash = regions_record.pop("content_sha256")
    assert _content_hash(regions_record) == saved_regions_hash, "regions.json content changed since it was saved"

    transform = LatticeTransform(
        axis=transform_record["axis"], side=transform_record["side"],
        distance=transform_record["distance"],
        axis_matrix=np.array(transform_record["axis_matrix"]),
        focal=transform_record["focal"],
        principal_point=tuple(transform_record["principal_point"]),
        lateral_origin=tuple(transform_record["lateral_origin"]),
        period=transform_record["period"])
    image_size = tuple(transform_record["image_size"])
    regions = [RegionEvidence(r["label"], tuple(tuple(p) for p in r["polygon"]))
              for r in regions_record["regions"]]

    evidence = rasterize_cells(transform, regions, image_size=image_size,
                              rows=range(ROWS), columns=range(COLUMNS))
    recovered_labels = declared_labels_from_evidence(evidence)
    footprint = build_footprint(axis=2, face_coordinate=transform_record["face_coordinate"],
                               side=1.0, lateral_origin=(0, 0), rows=ROWS, columns=COLUMNS,
                               declared=recovered_labels)

    saved_evidence = json.loads((out_dir / "evidence.json").read_text(encoding="utf-8"))
    recovered_blocks = sorted(list(cell_to_block(footprint, c.cell)) for c in footprint.of(SUPPORTED))
    recovered_opening = [list(c.cell) for c in footprint.of(GEOMETRY_OPENING)]
    recovered_unknown = [list(c.cell) for c in footprint.of(UNKNOWN)]

    assert recovered_blocks == saved_evidence["supported_blocks"], (
        recovered_blocks, saved_evidence["supported_blocks"])
    assert recovered_opening == saved_evidence["opening_cells"]
    assert recovered_unknown == saved_evidence["unknown_cells"]
    print(f"RELOAD OK: {len(recovered_blocks)} supported blocks, "
          f"{len(recovered_opening)} opening, {len(recovered_unknown)} unknown - "
          "matches the saved evidence.json exactly, from transform.json/regions.json alone")

    # The export-to-reconstruction transform must be sufficient for this
    # fresh process to re-verify the litematic file on its own - reading
    # minimum_shift back from export_transform.json rather than
    # recomputing it from footprint's expected cells, which `build`
    # already did once in a different process.
    from mc_photogrammetry.reconstruction.footprint_adapter import reproject_exported_volume

    export_transform_record = json.loads((out_dir / "export_transform.json").read_text(encoding="utf-8"))
    saved_export_hash = export_transform_record.pop("content_sha256")
    assert _content_hash(export_transform_record) == saved_export_hash, (
        "export_transform.json content changed since it was saved")
    reprojected, _ = reproject_exported_volume(
        transform, footprint, out_dir / "control.litematic",
        minimum_shift=tuple(export_transform_record["minimum_shift"]))
    print(f"RELOAD OK: {len(reprojected)} solid blocks in control.litematic independently "
         f"reproject, using only the persisted export_transform.json minimum_shift "
         f"{tuple(export_transform_record['minimum_shift'])} - nothing recomputed from footprint")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("build", "reload"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.out_dir)
    else:
        reload_and_verify(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
