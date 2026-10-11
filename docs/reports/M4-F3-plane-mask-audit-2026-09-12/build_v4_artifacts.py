"""Build the frozen, image-only M4 F3 v4 annotation and review overlays.

This hand-traced annotation is evaluation-only. Plane identity is based on
visible architectural corners in the screenshots; no camera/fixture/reference
metadata is loaded by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V3 = ROOT / "docs/reports/M4-C2-boundary-targets-v3-2026-09-09.json"
V4 = ROOT / "docs/reports/M4-C2-boundary-targets-v4-2026-09-12.json"
FRAMES = ROOT / "templates 2"


def _inside_polygon(points: list[list[int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


def build_annotation() -> dict:
    data = json.loads(V3.read_text(encoding="utf-8"))
    data["annotation_id"] = "M4-F3-plane-specific-targets-v4-2026-09-12"
    data["note"] = (
        "Versioned visual audit of the three Template 2 source screenshots. Each target names one "
        "physical wall plane before tracing. v1-v3 are preserved. Frame 1 replaces the dining-area "
        "bounding rectangle with separate traced furniture silhouettes and an unknown transition band; "
        "(1150,740) remains clear wall. Frame 3 narrows the left-wall claim to the visible corner, "
        "with the corner band unknown; (600,300) is left wall while (1400,300) is another wall plane. "
        "The existing frame 2 plane boundary was reviewed and retained. The target file is evaluation-only; "
        "no pose, fixture metadata, reference geometry, or reconstruction output was used to identify planes."
    )
    data["visual_audit"] = {
        "method": (
            "Native 2560x1369 source frames reviewed side-by-side with composed v4 overlays; "
            "boundaries traced on source-coordinate grid crops. Plane identity follows visible "
            "room-corner lines, not repeated wood texture."
        ),
        "source_frames": [],
        "unknown_policy": (
            "The listed uncertainty polygons cover only ambiguous boundary strips or occlusion "
            "transitions; reasons are listed per target. Unknown area is reported after composition "
            "and before HUD/domain exclusions."
        ),
    }
    for frame in FRAME_NAMES:
        raw = (FRAMES / frame).read_bytes()
        data["visual_audit"]["source_frames"].append({
            "frame": frame,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "image_size": [2560, 1369],
        })

    for i, target in enumerate(data["targets"], 1):
        target["target_id"] = f"t2_f{i}_left_wall"
        target["physical_plane"] = (
            "One visible, continuous left-hand wall plane, bounded at its first visible inside "
            "corner; repeated wood texture beyond that corner does not continue this target."
        )
        target.pop("crop_provenance", None)
        if i == 1:
            _frame1(target)
        elif i == 2:
            _frame2(target)
        else:
            _frame3(target)
    return data


FRAME_NAMES = [
    "2026-09-02_13.22.14.png",
    "2026-09-02_13.24.24.png",
    "2026-09-02_13.24.33.png",
]


def _set_anchor_lists(target: dict) -> None:
    anchors = target["review_anchors"]
    positives = [a for a in anchors if a["label"] == "positive"]
    negatives = [a for a in anchors if a["label"] == "negative"]
    target["wall_interior_anchors"] = [a["point"] for a in positives]
    target["negative_anchors"] = [a["point"] for a in negatives]
    target["wall_interior_anchor_reasons"] = [a["reason"] for a in positives]
    target["negative_anchor_reasons"] = [a["reason"] for a in negatives]


def _frame1(target: dict) -> None:
    target["description"] = (
        "Left-hand dark-oak panelled wall in the dining view. The broad wall silhouette is retained "
        "to the reviewed corner uncertainty band; the dining furniture is traced as individual "
        "foreground silhouettes, leaving visible wall around it unmasked."
    )
    old = target["negative_polygons"]["foreground_occlusion"]
    # Preserve the foreground bed/table and three narrow wall sconces. Replace
    # only the oversized dining rectangle with the traced table, legs, flower,
    # and two end benches/chairs.
    target["negative_polygons"]["foreground_occlusion"] = [
        old[0],
        [[1065, 755], [1090, 752], [1125, 748], [1160, 750], [1195, 750],
         [1230, 754], [1270, 760], [1320, 766], [1375, 771], [1430, 777],
         [1480, 785], [1515, 794], [1510, 805], [1455, 806], [1390, 799],
         [1325, 792], [1260, 788], [1200, 786], [1145, 783], [1105, 778], [1080, 772]],
        [[1080, 772], [1105, 778], [1145, 783], [1200, 786], [1260, 788],
         [1325, 792], [1390, 799], [1455, 806], [1500, 806], [1485, 825],
         [1425, 824], [1360, 817], [1300, 812], [1240, 808], [1185, 805],
         [1135, 800], [1090, 791]],
        [[1115, 798], [1135, 800], [1145, 850], [1123, 848]],
        [[1195, 804], [1217, 807], [1227, 858], [1204, 855]],
        [[1290, 806], [1312, 807], [1307, 859], [1286, 858]],
        [[1370, 810], [1392, 812], [1383, 860], [1363, 858]],
        # Flower/pot: keep the thin stem localized instead of absorbing the
        # whole dining area into a rectangular occluder.
        [[1442, 693], [1452, 693], [1456, 715], [1468, 716], [1472, 758],
         [1450, 761], [1448, 735], [1438, 730]],
        [[985, 757], [1012, 752], [1040, 760], [1064, 780], [1055, 811],
         [1030, 824], [1002, 815], [986, 793]],
        [[1690, 772], [1720, 764], [1754, 770], [1784, 789], [1798, 827],
         [1780, 849], [1746, 845], [1724, 824], [1704, 819]],
        old[2], old[3], old[4],
    ]
    # The narrow strip follows the table's upper silhouette; uncertain edge
    # pixels are not silently promoted to either wall or furniture.
    target["uncertain_polygons"].append(
        [[1065, 748], [1090, 745], [1125, 741], [1160, 743], [1195, 743],
         [1230, 747], [1270, 753], [1320, 759], [1375, 764], [1430, 770],
         [1480, 778], [1515, 787], [1515, 794], [1480, 785], [1430, 777],
         [1375, 771], [1320, 766], [1270, 760], [1230, 754], [1195, 750],
         [1160, 750], [1125, 748], [1090, 752], [1065, 755]]
    )
    target["uncertain_polygon_reasons"] = [
        "Roof-gap/ceiling transition is dark and low-contrast; the wall-vs-roof edge is uncertain.",
        "Original inside-corner/support-beam transition; plane ownership is ambiguous in this strip.",
        "Dining tabletop edge: the dark-oak-to-wood-object transition is antialiased and shadowed; "
        "the tabletop interior is traced separately as negative.",
    ]
    target["review_anchors"] = [
        {"point": [1150, 740], "label": "positive", "reason": "Clear left-wall paneling above the dining tabletop; v3's dining rectangle wrongly removed it."},
        {"point": [1000, 700], "label": "positive", "reason": "Visible wall margin left of the table group."},
        {"point": [1200, 748], "label": "unknown", "reason": "On the traced tabletop transition ribbon."},
        {"point": [1200, 775], "label": "negative", "reason": "Inside the tabletop surface."},
        {"point": [1400, 780], "label": "unknown", "reason": "Tabletop overlaps the existing far-corner uncertainty band here; it remains excluded."},
        {"point": [1460, 720], "label": "negative", "reason": "Inside the narrow flower/stem silhouette on the table."},
        {"point": [1750, 805], "label": "negative", "reason": "Inside the separate right-hand bench/chair silhouette."},
    ]
    _set_anchor_lists(target)


def _frame2(target: dict) -> None:
    target["description"] = (
        "Left-hand wall panelling looking down the room. The far inside-corner boundary is kept as "
        "the existing narrow unknown band; the far gable-end wall remains outside this target."
    )
    target["uncertain_polygon_reasons"] = [
        "Visible corner and beam transition between the left wall and far gable-end; do not assign "
        "either plane through the occluding beam."
    ]
    target["review_anchors"] = [
        {"point": [500, 300], "label": "positive", "reason": "Interior of the left-hand wall, well before the visible corner."},
        {"point": [700, 700], "label": "positive", "reason": "Left-wall side of the traced lower wall/bench boundary."},
        {"point": [1000, 300], "label": "unknown", "reason": "Inside the narrow visible corner transition band."},
        {"point": [1200, 250], "label": "negative", "reason": "Far gable-end opening; it is not the left-wall plane."},
        {"point": [1200, 850], "label": "negative", "reason": "Far gable-end dining area, beyond the left-wall corner."},
    ]
    _set_anchor_lists(target)


def _frame3(target: dict) -> None:
    target["description"] = (
        "Left-hand wall plane in the steep downward view. Its traced right edge stops at the visible "
        "room corner; the far wall at (1400,300) is a different plane despite matching wood appearance. "
        "The source-visible lower left wall is retained."
    )
    # The existing gamma-boosted source crop supports wall to the bottom edge
    # for x<1080. The corner band keeps the plane claim conservative at x=960..1060.
    target["positive_polygon"] = [[0, 0], [960, 0], [960, 1369], [0, 1369]]
    target["uncertain_polygons"] = [
        [[650, 0], [1900, 0], [1900, 100], [650, 100]],
        [[960, 100], [1060, 100], [1060, 780], [960, 780]],
    ]
    target["uncertain_polygon_reasons"] = [
        "Low-contrast ceiling/wall transition; ownership cannot be determined reliably from this view.",
        "Visible corner/beam transition between the left-hand wall and farther wall planes.",
    ]
    # Remove v3's coarse furniture rectangle. The other walls/floor are
    # negative by default once the single left-wall positive footprint ends.
    target["negative_polygons"] = {"foreground_occlusion": [], "opening": []}
    target["review_anchors"] = [
        {"point": [600, 300], "label": "positive", "reason": "Interior of the left-hand wall paneling."},
        {"point": [600, 1200], "label": "positive", "reason": "The gamma-boosted crop confirms the left-hand paneling continues to the bottom edge."},
        {"point": [940, 300], "label": "positive", "reason": "Clear left-wall side immediately before the corner uncertainty band."},
        {"point": [975, 300], "label": "unknown", "reason": "Inside the visible corner transition band."},
        {"point": [1000, 300], "label": "unknown", "reason": "Inside the visible corner transition band."},
        {"point": [1100, 300], "label": "negative", "reason": "Beyond the corner on a distinct farther wall plane."},
        {"point": [1400, 300], "label": "negative", "reason": "The lead-cited far-wall anchor; same wood appearance does not make it the left-wall plane."},
    ]
    _set_anchor_lists(target)


def write_annotation() -> str:
    data = build_annotation()
    V4.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return hashlib.sha256(V4.read_bytes()).hexdigest()


def render_review_artifacts() -> None:
    data = json.loads(V4.read_text(encoding="utf-8"))
    panel_rows = []
    for target in data["targets"]:
        frame = target["frame"]
        image = cv2.imread(str(FRAMES / frame), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        from mc_photogrammetry.reconstruction.boundary_scoring import target_masks, usable_domain

        domain = usable_domain(width, height, top_exclude=0, bottom_exclude=0)
        positive, unknown = target_masks(target, width, height, domain)
        negative = domain & ~positive & ~unknown
        color = np.zeros_like(image)
        color[positive] = (40, 180, 30)
        color[negative] = (30, 30, 190)
        color[unknown] = (0, 210, 255)
        overlay = cv2.addWeighted(image, 0.67, color, 0.33, 0)
        # Explicit source polygons get their own outline over the composed
        # target regions so a broad negative region cannot hide the trace.
        for polygons in target.get("negative_polygons", {}).values():
            for polygon in polygons:
                cv2.polylines(overlay, [_inside_polygon(polygon)], True, (0, 120, 255), 2, cv2.LINE_AA)
        for polygon in target.get("uncertain_polygons", []):
            cv2.polylines(overlay, [_inside_polygon(polygon)], True, (0, 230, 255), 2, cv2.LINE_AA)
        contour, _ = cv2.findContours(positive.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contour, -1, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, target["target_id"], (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        for anchor in target["review_anchors"]:
            x, y = anchor["point"]
            color = {"positive": (0, 255, 0), "negative": (0, 0, 255), "unknown": (0, 210, 255)}[anchor["label"]]
            cv2.circle(overlay, (x, y), 8, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)
        side_by_side = np.concatenate([image, overlay], axis=1)
        cv2.imwrite(str(OUT / f"side_by_side_{Path(frame).stem}.png"), side_by_side)
        panel_rows.append(cv2.resize(side_by_side, (2048, 548), interpolation=cv2.INTER_AREA))

    cv2.imwrite(str(OUT / "all_frames_side_by_side.png"), np.concatenate(panel_rows, axis=0))
    # The two required transition views retain native coordinates and labels.
    crop_specs = [
        (FRAME_NAMES[0], (980, 680, 1580, 875), "frame1_dining_transition.png", 2),
        (FRAME_NAMES[2], (800, 150, 1500, 800), "frame3_corner_transition.png", 1),
    ]
    for frame, (x0, y0, x1, y1), name, scale in crop_specs:
        image = cv2.imread(str(FRAMES / frame), cv2.IMREAD_COLOR)
        crop = image[y0:y1, x0:x1].copy()
        target = next(t for t in data["targets"] if t["frame"] == frame)
        for point, color in [([1150, 740], (0, 255, 0)), ([1200, 775], (0, 0, 255)), ([1200, 748], (0, 210, 255))] if frame == FRAME_NAMES[0] else [([600, 300], (0, 255, 0)), ([975, 300], (0, 210, 255)), ([1400, 300], (0, 0, 255))]:
            x, y = point
            if x0 <= x < x1 and y0 <= y < y1:
                cv2.circle(crop, (x - x0, y - y0), 8, color, -1, cv2.LINE_AA)
        for x in range((x0 // 25) * 25, x1, 25):
            cv2.line(crop, (x - x0, 0), (x - x0, crop.shape[0] - 1), (0, 255, 255), 1)
            cv2.putText(crop, str(x), (x - x0 + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        for y in range((y0 // 25) * 25, y1, 25):
            cv2.line(crop, (0, y - y0), (crop.shape[1] - 1, y - y0), (0, 255, 255), 1)
            cv2.putText(crop, str(y), (2, y - y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        for polygon in target.get("positive_polygon", []),:
            cv2.polylines(crop, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2) - np.array([[[x0, y0]]])], True, (255, 255, 0), 2, cv2.LINE_AA)
        for polygons in target.get("negative_polygons", {}).values():
            for polygon in polygons:
                pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2) - np.array([[[x0, y0]]])
                cv2.polylines(crop, [pts], True, (0, 120, 255), 2, cv2.LINE_AA)
        for polygon in target.get("uncertain_polygons", []):
            pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2) - np.array([[[x0, y0]]])
            cv2.polylines(crop, [pts], True, (0, 230, 255), 2, cv2.LINE_AA)
        if scale != 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / name), crop)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    digest = write_annotation()
    render_review_artifacts()
    print(f"wrote {V4.relative_to(ROOT)} sha256={digest}")
