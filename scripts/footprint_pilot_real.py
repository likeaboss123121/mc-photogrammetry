"""M4 B, step 6: one corrected real-image Template 2 pilot.

Authorized only after the synthetic checks in `footprint_pilot.py` passed.
Every geometric input comes from
`docs/reports/M4-partB-pilot-annotation-2026-09-09.json` - picked by looking
directly at the screenshot, frozen with a SHA-256 before this script ever
scores anything against the reference. Camera optics come from image-derived
calibration (`axis_directions`, `focal_pixels`), never the manifest. The
manifest is read only at the very end, to report - not to build anything
above it.

**This is an assumed-grid developer diagnostic, not a resolved reconstruction.**
A lead review made this precise: marking the *length* low-confidence does not
resolve *phase*. Distance places the plane; it says nothing about where, along
that plane, block boundaries actually sit. `lateral_origin=(0, 0)` - where
this script's lattice grid starts - is an arbitrary, unverified choice, not a
measurement, and it is exactly the kind of thing that decides which specific
cells end up called "supported": shift it and a different set of grid squares
covers the same real wall. So the primary output here is the *continuous*
supported region (a bounding extent in real-world u/v coordinates, independent
of any particular grid alignment); the discretized cell/block layout is saved
alongside it, explicitly labelled hypothetical, not asserted as the recovered
geometry.

Shares its bundle schema with `footprint_pilot.py`'s synthetic control
(transform.json/regions.json, each self-hashed; evidence.json with full
per-cell coverage; a litematic; an overlay) plus source/calibration
provenance hashes the synthetic bundle does not need, since it has no real
photograph or calibration artifact behind it.

    python scripts/footprint_pilot_real.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ANNOTATION_PATH = ROOT / "docs/reports/M4-partB-pilot-annotation-2026-09-09.json"
FROZEN_SHA256 = "9621bc5a07b1c69f649a8e3792128708157c16c50ef9c09135a5d87078248978"
CALIBRATION_PATH = ROOT / "docs/reports/M3-calibration-templates-2.json"
FIXTURE = ROOT / "templates 2"


def _content_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    raw = ANNOTATION_PATH.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_SHA256:
        raise SystemExit(
            f"annotation file changed since it was frozen: expected {FROZEN_SHA256}, "
            f"got {actual_hash}. Not scoring a moved target.")
    annotation = json.loads(raw)
    print(f"annotation frozen and verified: {FROZEN_SHA256}")

    import cv2
    import numpy as np

    from mc_photogrammetry.reconstruction.calibration import axis_directions
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length
    from mc_photogrammetry.reconstruction.footprint import (
        FOREGROUND_OCCLUSION, GEOMETRY_OPENING, OPENING, SUPPORTED, UNKNOWN,
        VISIBLE, block_to_cell, build_footprint, cell_to_block, footprint_to_volume)
    from mc_photogrammetry.reconstruction.footprint_adapter import (
        LatticeTransform, RegionEvidence, cell_corners_pixels, pixel_to_lateral,
        declared_labels_from_evidence, rasterize_cells)
    from mc_photogrammetry.export.litematic import LitematicExporter
    from mc_photogrammetry.export.compare import load_litematic

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_id = annotation["frame"]
    image_path = FIXTURE / frame_id
    calibration_hash = _file_hash(CALIBRATION_PATH)
    source_hash = _file_hash(image_path)

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    entry = next(e for e in calibration["estimates"] if e["frame"] == frame_id)
    size = tuple(calibration["inputs"][frame_id]["size"])
    axes = axis_directions(entry["yaw_degrees_mod90"], entry["pitch_degrees"],
                           entry.get("roll_degrees", 0.0))
    focal = entry["focal_pixels"]
    centre_pp = entry.get("principal_point")
    principal_point = tuple(centre_pp) if centre_pp else (size[0] / 2.0, size[1] / 2.0)

    length = annotation["length_estimate"]
    points = [tuple(p) for p in length["points"]]
    blocks_apart = float(length["blocks_apart"])

    # All axis/side candidates that resolve, not just the first - each is a
    # distinct hypothesis about which plane is being measured, and which
    # ones resolve at all is itself provenance worth keeping.
    axis_alternatives = []
    for axis in annotation["axis_candidates"]:
        for side in (1.0, -1.0):
            constraint = scale_from_known_length(
                points[0], points[1], blocks_apart, axis, axes, focal, size,
                principal_point=principal_point)
            axis_alternatives.append({
                "axis": axis, "side": side, "status": constraint.status,
                "distance_blocks": constraint.distance,
            })
    resolved_alternatives = [a for a in axis_alternatives if a["distance_blocks"] is not None]

    def write_status(status: str, **extra) -> None:
        payload = {
            "status": status, "annotation_sha256": FROZEN_SHA256,
            "calibration_artifact_sha256": calibration_hash,
            "source_image_sha256": source_hash,
            "frame": frame_id, "axis_alternatives": axis_alternatives,
            **extra,
        }
        (args.out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not resolved_alternatives:
        print("STATED LENGTH DID NOT RESOLVE on any axis/side candidate.")
        write_status("length_unresolved")
        return 0

    # The first resolved alternative is used below, same as before - but
    # every alternative, resolved or not, is now saved, and the choice among
    # them is not asserted as the only geometrically possible one.
    axis, side, distance = (resolved_alternatives[0]["axis"], resolved_alternatives[0]["side"],
                            resolved_alternatives[0]["distance_blocks"])
    print(f"length resolved: axis={axis} side={side:+.0f} distance={distance:.3f} blocks "
         f"[confidence: {length['confidence']}] "
         f"({len(resolved_alternatives)} of {len(axis_alternatives)} axis/side candidates resolved)")

    # lateral_origin is an ARBITRARY, UNRESOLVED phase choice - see the module
    # docstring. Nothing here establishes it from evidence; it exists only to
    # give the hypothetical discretization below a concrete grid to sit on.
    transform = LatticeTransform(axis=axis, side=side, distance=distance, axis_matrix=axes,
                                 focal=focal, principal_point=principal_point,
                                 lateral_origin=(0.0, 0.0), period=1.0)

    transform_record = {
        "axis": transform.axis, "side": transform.side, "distance": transform.distance,
        "axis_matrix": transform.axis_matrix.tolist(), "focal": transform.focal,
        "principal_point": list(transform.principal_point),
        "lateral_origin": list(transform.lateral_origin), "period": transform.period,
        "image_size": list(size),
        "lateral_origin_is_assumed_not_measured": True,
        "note": "this is the continuous camera-space transform only; the "
                "separate, arbitrary block-export convention "
                "(face_coordinate/export lateral_origin) is recorded in "
                "result.json, not here - they answer different questions.",
    }
    transform_record["content_sha256"] = _content_hash(transform_record)
    (args.out_dir / "transform.json").write_text(json.dumps(transform_record, indent=2), encoding="utf-8")

    regions = [RegionEvidence(VISIBLE, tuple(tuple(p) for p in annotation["visible_extent_polygon"]))]
    for region in annotation.get("occlusion_regions_image_px", []):
        regions.append(RegionEvidence(FOREGROUND_OCCLUSION, tuple(tuple(p) for p in region["polygon"])))
    regions_record = {"regions": [{"label": r.label, "polygon": [list(p) for p in r.polygon]}
                                  for r in regions]}
    regions_record["content_sha256"] = _content_hash(regions_record)
    (args.out_dir / "regions.json").write_text(json.dumps(regions_record, indent=2), encoding="utf-8")

    lateral_corners = [pixel_to_lateral(transform, tuple(p))
                       for p in annotation["visible_extent_polygon"]]
    lateral_corners = [c for c in lateral_corners if c is not None]
    if not lateral_corners:
        print("STATED PLANE DOES NOT INTERSECT THE DECLARED EXTENT in front of the camera.")
        write_status("extent_behind_camera")
        return 0

    # The continuous supported region - in real-world u/v, independent of any
    # grid alignment - computed from the extent polygon's own bounds, before
    # any discretization choice enters.
    us = [c[0] for c in lateral_corners]
    vs = [c[1] for c in lateral_corners]
    continuous_region = {"u_range": [min(us), max(us)], "v_range": [min(vs), max(vs)],
                         "note": "bounds of the declared visible extent in real-world "
                                 "(u, v) blocks on the plane - independent of the "
                                 "hypothetical grid below"}

    columns = range(int(np.floor(min(us))) - 1, int(np.ceil(max(us))) + 1)
    rows = range(int(np.floor(min(vs))) - 1, int(np.ceil(max(vs))) + 1)
    print(f"candidate grid: {len(rows)} rows x {len(columns)} columns "
         f"(hypothetical - see lateral_origin above)")

    evidence = rasterize_cells(transform, regions, image_size=size, rows=rows, columns=columns)
    recovered_labels = declared_labels_from_evidence(evidence)
    # face_coordinate/lateral_origin are the arbitrary local-export choice
    # cell_to_block's own docstring says is fine ("Astra's example... is the
    # side=1 case") - not the continuous camera transform above, a separate
    # choice recorded in result.json's export_convention.
    export_face_coordinate = 1 if side > 0 else 0
    footprint = build_footprint(axis=axis, face_coordinate=export_face_coordinate, side=side,
                               lateral_origin=(0, 0),
                               rows=len(rows), columns=len(columns),
                               declared={(r - min(rows), c - min(columns)): label
                                        for (r, c), label in recovered_labels.items()})

    from collections import Counter
    totals = Counter(c.geometry for c in footprint.cells)
    print(f"hypothetical grid resolved (at the assumed phase): {dict(totals)}")

    evidence_record = {
        "cells": [{"cell": list(e.cell), "coverage_by_label": e.coverage_by_label,
                   "coverage_of_in_frame": e.coverage_of_in_frame,
                   "declared": e.declared, "outside_image_fraction": e.outside_image_fraction,
                   "conflict_fraction": e.conflict_fraction}
                  for e in evidence],
        "supported_blocks": sorted(list(cell_to_block(footprint, c.cell))
                                   for c in footprint.of(SUPPORTED)),
        "opening_cells": [list(c.cell) for c in footprint.of(GEOMETRY_OPENING)],
        "unknown_cells": len(footprint.of(UNKNOWN)),
    }
    (args.out_dir / "evidence.json").write_text(json.dumps(evidence_record, indent=2), encoding="utf-8")

    litematic_path = None
    if footprint.of(SUPPORTED):
        volume = footprint_to_volume(footprint)
        litematic_path = args.out_dir / "pilot.litematic"
        LitematicExporter().export(volume, litematic_path,
                                   name="footprint pilot, real image (HYPOTHETICAL GRID)",
                                   description=("M4 B real-image pilot, Template 2 - assumed-grid "
                                                "developer diagnostic, not a resolved reconstruction"))
        snapshot = load_litematic(litematic_path)
        # A single-region litematic always writes Position (0, 0, 0) - a
        # tested invariant (export.validation), not a bug: single-region
        # files store a shape local to its own minimum corner, since
        # litematics are meant to be relocatable. Read-back coordinates are
        # shifted by the exported volume's own minimum corner relative to
        # what cell_to_block produced; add that back before inverting.
        expected_cells = {c.cell for c in footprint.of(SUPPORTED)}
        exported_blocks = {cell_to_block(footprint, cell) for cell in expected_cells}
        minimum = tuple(min(block[axis] for block in exported_blocks) for axis in range(3))
        readback_cells = {block_to_cell(footprint, tuple(local[i] + minimum[i] for i in range(3)))
                          for local, state in snapshot.blocks.items()
                          if state.name != "minecraft:air"}
        assert readback_cells == expected_cells, (readback_cells, expected_cells)
        print(f"read-back verification OK: {len(expected_cells)} solid blocks in "
             f"{litematic_path.name} match their pre-export cells")

    write_status(
        "assumed_grid_diagnostic",
        length_confidence=length["confidence"],
        continuous_supported_region=continuous_region,
        hypothetical_grid=dict(totals),
        export_convention={
            "face_coordinate": export_face_coordinate, "lateral_origin": [0, 0],
            "note": "arbitrary local export choice, distinct from the continuous "
                    "camera transform in transform.json. A single-region litematic "
                    "always stores its shape local to its own minimum corner "
                    "(Position is always (0,0,0) - a tested convention, litematics "
                    "are meant to be relocatable), so the exported file's own "
                    "coordinates are not directly the ones in evidence.json's "
                    "supported_blocks; read-back verification accounts for that shift.",
        },
        note="lateral_origin is an unresolved, arbitrary phase choice (see module "
             "docstring and transform.json's lateral_origin_is_assumed_not_measured); "
             "the supported_blocks/opening_cells in evidence.json are a hypothetical "
             "discretization at that assumed phase, not recovered exact cells - the "
             "continuous_supported_region above is the only claim this pilot makes "
             "without assuming a grid alignment.",
    )

    # Overlay: extent/occlusion regions and every candidate cell's resolved
    # geometry (from the in-memory footprint, since opening/unknown cells have
    # nothing in the litematic to read back), on the actual photograph.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    overlay = image.copy()
    cv2.fillPoly(overlay, [np.array(annotation["visible_extent_polygon"], dtype=np.int32)],
                (0, 200, 0))
    for region in annotation.get("occlusion_regions_image_px", []):
        cv2.fillPoly(overlay, [np.array(region["polygon"], dtype=np.int32)], (0, 0, 200))
    image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)
    verdict_colours = {SUPPORTED: (60, 200, 60), GEOMETRY_OPENING: (60, 60, 220), UNKNOWN: (140, 140, 140)}
    for cell in footprint.cells:
        world_cell = (cell.cell[0] + min(rows), cell.cell[1] + min(columns))
        corners = cell_corners_pixels(transform, world_cell)
        if corners is None:
            continue
        pts = np.array([[round(x), round(y)] for x, y in corners], dtype=np.int32)
        cv2.polylines(image, [pts], True, verdict_colours.get(cell.geometry, (0, 0, 0)), 2)
    p0, p1 = [round(v) for v in points[0]], [round(v) for v in points[1]]
    cv2.line(image, p0, p1, (0, 255, 255), 3)
    cv2.putText(image, f"{blocks_apart:.0f} blocks ({length['confidence']} confidence)",
               (p0[0] + 15, p0[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(image, "HYPOTHETICAL GRID - phase not resolved, see result.json",
               (15, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)
    cv2.imwrite(str(args.out_dir / "overlay.png"), image)
    print(f"wrote transform.json, regions.json, evidence.json, result.json, overlay.png"
         f"{', pilot.litematic' if litematic_path else ''} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
