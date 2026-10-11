"""Fixed-input evaluation-only SIFT supply probe. No matching or inference.

No ordinary input is accessed on import. A new independent lead authorization
must bind this probe/test before the runtime command can read encoded pixels.
The 512-feature roster and production detector remain unmodified.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROBE = "scripts/evaluation/probe_native_feature_density.py"
TEST = "tests/test_native_feature_density.py"
AUTH = "agent swarm instantiation template/runs/RESULT-LEAD-NATIVE-FEATURE-DENSITY-AUTH-01-2026-10-09.json"
OUTPUT = "exports/diagnostics/native-feature-density-2026-10-09"
INPUT = "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-05.png"
INPUT_SHA = "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"
DIAGNOSTIC = "exports/diagnostics/endpoint-conflict-mask-bundle-2026-10-09/diagnostic.json"
DIAGNOSTIC_SHA = "a4bb71e275b277a8b52a8b14884cb72a8be71df698a5f9e4c0d44894daf3aa98"
BASELINE = "exports/diagnostics/endpoint-conflict-native-roster-2026-10-09-repair-01/roster.json"
BASELINE_SHA = "8fccb47e79c3079d218d9c59cf5ff70307079cb83490a0fcd9d66d662f70ca7b"
SOURCE_PINS = {
    "scripts/evaluation/acquire_endpoint_conflict_native_roster.py": "6cb854e1800779514efe26e6ed3ea83a5c7a0975743380e8969d9bc46c6cc14c",
    "scripts/evaluation/image_point_candidate_roster.py": "0e9a027e1209f3a96c9e2bd7443311a14e654f7bd3440f30cab0404f32f28ca2",
    "scripts/evaluation/diagnose_image_geometry_failure.py": "5c3bdb9ed5dc34b021f3e60c92d0b7cd5ffa2aba8aea35225354213107d0d2af",
    "src/mc_photogrammetry/input/image.py": "43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732",
    "src/mc_photogrammetry/pipeline/correspondence.py": "192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1",
    "src/mc_photogrammetry/pipeline/image_derived_session.py": "e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e",
    "src/mc_photogrammetry/vision/gui_mask.py": "1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py": "c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b",
    "src/mc_photogrammetry/pipeline/image_derived_signatures.py": "7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py": "fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e",
    "src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py": "f8065946eaedb95cdbb8d2bfceace0246bf1369d2f8683f2fd555aa1904a7c0a",
    "src/mc_photogrammetry/pipeline/image_derived_joint_graph.py": "60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab",
}
MASK_PINS = (
    (("image-0000", 2), "338a47a2cb081a1991014fd59b9413f04e2769195f39c373335e31dec9d25098", "3ecfb4b4709962ecbc5a8e122858c54a5c3334e7a0451ea55ce02bcb0381019a", "bd86f23c0191e91a9c93220c50e798fabd5e3e43c576a5fd7cd5b8bd6e94d708"),
    (("image-0000", 3), "61b803612ab055368b8bdab037441f780b9014eace8c117b41da6b28cd77d55d", "85b28d1243dd715b2bddb02b129bd24df4e868129541b1b71d29f68e0354367c", "cc7f4bbdca723ea630ddd7a2d3abfb3b97632ee8cc1569ac6465b7b8aaeb989e"),
)
SCHEMA = "mc-photogrammetry-native-feature-density-v1"
INVENTORY_SCHEMA = "native-density-feature-content-2048-v1"
LIMITS = {"frames": 1, "raw_returned_features": 2048, "membership_checks": 4096,
          "cross_view_comparisons": 0, "wall_seconds": 180, "job_committed_bytes": 2 * 1024**3,
          "evidence_bytes": 4 * 1024**2, "artifact_bytes": 8 * 1024**2}


class Refusal(RuntimeError):
    """Terminal: no retry, partial inventory or substitute mask."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("duplicate_json_key")
        result[key] = value
    return result


def read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise Refusal("source_byte_limit")
    return data


def pinned(path, expected, limit):
    data = read(path, limit)
    if sha(data) != expected.lower():
        raise Refusal("source_pin_mismatch: " + str(path))
    return data


def parse(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(Refusal("nonfinite_json")))


def sealed_source(data):
    record = parse(data)
    if canonical(record) + b"\n" != data:
        raise Refusal("noncanonical_source")
    body = dict(record)
    if body.pop("record_sha256", None) != sha(canonical(body)):
        raise Refusal("source_record_digest")
    return record


def verify_masks(record, pins=MASK_PINS):
    entries = record["source_region_masks"]["entries"]
    if len(entries) != 3 or [tuple(e["endpoint"]) for e in entries] != [p[0] for p in pins] + [("image-0001", 1)]:
        raise Refusal("exact_source_mask_set")
    for entry, (endpoint, region_id, packed_sha, raw_sha) in zip(entries[:2], pins):
        if (tuple(entry["endpoint"]) != endpoint
                or entry["mask"]["source_region_id"] != region_id
                or entry["mask"]["mask_sha256"] != packed_sha
                or entry["full_native_bool_c_sha256"] != raw_sha):
            raise Refusal("exact_source_mask_binding")


def authorize(root=ROOT):
    # Opening the absent auth is the very first I/O. No project/image imports.
    auth_bytes = read(root / AUTH, 65536)
    auth = parse(auth_bytes)
    if (auth.get("disposition") != "ACCEPT_PROBE; AUTHORIZE_ONE_DENSITY_RUN"
            or auth.get("single_probe") is not True or auth.get("output") != OUTPUT
            or auth.get("input_pins") != {INPUT: INPUT_SHA}
            or auth.get("diagnostic_sha256") != DIAGNOSTIC_SHA
            or auth.get("baseline_roster_sha256") != BASELINE_SHA):
        raise Refusal("independent_density_authorization_required")
    code_pins = {**SOURCE_PINS, PROBE: auth.get("code_pins", {}).get(PROBE),
                 TEST: auth.get("code_pins", {}).get(TEST)}
    if any(type(pin) is not str or len(pin) != 64 for pin in code_pins.values()):
        raise Refusal("authorization_code_pins_required")
    for name, pin in code_pins.items():
        pinned(root / name, pin, LIMITS["evidence_bytes"])
    record = sealed_source(pinned(root / DIAGNOSTIC, DIAGNOSTIC_SHA, 1024**2))
    verify_masks(record)
    ingress = record["ingress"][0]
    if (ingress["frame_id"] != "image-0000" or ingress["byte_sha256"] != INPUT_SHA
            or ingress["decoded_shape"] != [900, 1600, 3] or ingress["decoded_dtype"] != "|u1"):
        raise Refusal("fixed_native_ingress")
    baseline = sealed_source(pinned(root / BASELINE, BASELINE_SHA, LIMITS["evidence_bytes"]))
    if baseline["native_ingress"] != record["ingress"] or baseline["mask_bundle_sha256"] != record["source_region_masks"]["record_sha256"]:
        raise Refusal("baseline_source_join")
    return record, baseline, {"authorization_sha256": sha(auth_bytes), "code_pins": code_pins,
                              "input_pins": {INPUT: INPUT_SHA}, "diagnostic_sha256": DIAGNOSTIC_SHA,
                              "baseline_roster_sha256": BASELINE_SHA}


def _services():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from mc_photogrammetry.input.image import load_image_bytes
    from mc_photogrammetry.pipeline import correspondence
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    from scripts.evaluation import image_point_candidate_roster as roster
    from scripts.evaluation import diagnose_image_geometry_failure as diagnostic
    return np, load_image_bytes, correspondence, image_sha256, roster, diagnostic


def settings(roster):
    return replace(roster.SETTINGS, max_features=2048)


def bind_frame(data, ingress, *, services=None):
    np, decode, correspondence, image_sha256, roster, _ = services or _services()
    if (type(data) is not bytes or sha(data) != ingress["byte_sha256"]
            or len(data) != ingress["byte_length"] or ingress["frame_id"] != "image-0000"
            or ingress["decoded_dtype"] != "|u1"):
        raise Refusal("encoded_frame_binding")
    shape = ingress["decoded_shape"]
    if len(shape) != 3 or shape[2] != 3 or any(type(n) is not int or n <= 0 for n in shape) or shape[0] * shape[1] > 1_440_000:
        raise Refusal("native_frame_shape")
    transform = roster.identity(shape[1], shape[0], ingress["transform"])
    pixels = decode(data, suffix=".png")
    if pixels.dtype != np.uint8 or list(pixels.shape) != shape or image_sha256(pixels) != ingress["decoded_pixel_sha256"]:
        raise Refusal("decoded_frame_binding")
    pixels = np.array(pixels, copy=True, order="C")
    usable = np.array(correspondence.usable_mask_for("image-0000", pixels.shape, None), dtype=bool, copy=True, order="C")
    pixels.setflags(write=False)
    usable.setflags(write=False)
    source = {"frame_id": "image-0000", "encoded_sha256": sha(data), "encoded_length": len(data),
              "decoded_sha256": roster.array_digest(pixels), "shape": list(pixels.shape), "dtype": pixels.dtype.str,
              "mask_sha256": roster.array_digest(usable), "usable_true": True, "transform": transform,
              "domain": roster.DOMAIN, "raster_kind": "native", "decoder": ingress["decoder"]}
    frame = roster.NativeFrame("image-0000", data, pixels, usable, source, ingress["transform"])
    roster._frame_source(frame)
    return frame


def bind_masks(record, frame, *, services=None):
    _, _, _, _, roster, diagnostic = services or _services()
    diagnostic._validate_mask_diagnostic(record)
    bundle = record["source_region_masks"]
    result = []
    for entry in bundle["entries"][:2]:
        pixels = diagnostic.load_native_source_region_mask(bundle, record, entry["endpoint"])
        mask = roster.RegionMask("image-0000", frame.source["decoded_sha256"], pixels,
                                 entry["full_native_bool_c_sha256"], "exact sealed native source;" + entry["mask"]["source_region_id"])
        roster._region_record(mask, frame)
        result.append((entry, mask))
    return tuple(result)


def general_position(points):
    """Exact, linear existence test; never enumerate n-choose-4 hypotheses.

    Find a triangle. A fourth point off its three edge lines gives a quadrangle.
    Otherwise, non-vertex points on two different edge lines give a quadrangle
    among five witnesses. Points confined to one edge plus its opposite vertex
    cannot contain four points with no three collinear.
    """
    points = sorted(set(tuple(p) for p in points))
    if len(points) < 4:
        return False
    points = [(Fraction(x), Fraction(y)) for x, y in points]
    def area(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    a, b = points[:2]
    c = next((p for p in points[2:] if area(a, b, p)), None)
    if c is None:
        return False
    triangle = (a, b, c)
    edge_points = {}
    for p in points:
        if p in triangle:
            continue
        edges = [i for i, (u, v) in enumerate(((a, b), (b, c), (c, a))) if not area(u, v, p)]
        if not edges:
            return True
        edge_points.setdefault(edges[0], p)
        if len(edge_points) >= 2:
            witnesses = triangle + tuple(edge_points.values())
            return any(all(area(*triple) for triple in itertools.combinations(quad, 3))
                       for quad in itertools.combinations(witnesses, 4))
    return False


def finalize(body):
    result = dict(body)
    result["record_sha256"] = sha(canonical(body))
    if len(canonical(result)) + 1 > LIMITS["evidence_bytes"]:
        raise Refusal("evidence_byte_limit")
    return result


def inventory(frame, masks, features, bindings, *, services=None, config=None):
    np, _, correspondence, _, roster, _ = services or _services()
    config = config or settings(roster)
    if type(config) is not correspondence.CorrespondenceSettings or config != settings(roster):
        raise Refusal("detector_config_changed")
    source = roster._frame_source(frame)
    if len(masks) != 2 or [tuple(e["endpoint"]) for e, _ in masks] != [("image-0000", 2), ("image-0000", 3)]:
        raise Refusal("exact_two_native_masks")
    for _, mask in masks:
        roster._region_record(mask, frame)
    if (type(features) is not correspondence.FrameFeatures or features.frame_id != "image-0000"
            or (features.height, features.width) != frame.pixels.shape[:2]
            or features.usable_pixels != int(np.count_nonzero(frame.usable_mask))):
        raise Refusal("feature_source_binding")
    points, sizes, descriptors = features.points, features.sizes, features.descriptors
    if (not isinstance(points, np.ndarray) or points.dtype != np.float64 or points.ndim != 2 or points.shape[1:] != (2,)
            or not isinstance(sizes, np.ndarray) or sizes.dtype != np.float64 or sizes.shape != (len(points),)
            or not isinstance(descriptors, np.ndarray) or descriptors.ndim != 2
            or descriptors.dtype.kind != "f" or descriptors.dtype.itemsize not in (4, 8)
            or len(descriptors) != len(points) or (len(points) and descriptors.shape[1] != 128)
            or type(features.boundary_dropped) is not int or features.boundary_dropped < 0):
        raise Refusal("feature_table_contract")
    count, dropped = len(points), features.boundary_dropped
    if count + dropped > LIMITS["raw_returned_features"]:
        raise Refusal("raw_returned_feature_limit")
    if (not np.isfinite(points).all() or not np.isfinite(sizes).all() or not np.isfinite(descriptors).all()
            or (sizes <= 0).any() or (points < 0).any()
            or (points[:, 0] >= frame.pixels.shape[1]).any() or (points[:, 1] >= frame.pixels.shape[0]).any()):
        raise Refusal("feature_numeric_contract")
    # Runtime metadata reuse only. No capture_inventory/acquire_roster call.
    runtime = roster._runtime(roster.SETTINGS)
    runtime["settings"] = asdict(config)
    runtime["detector"] = "cv2.SIFT_create(nfeatures=2048)"
    anchor = {"schema": INVENTORY_SCHEMA, "frame_source": source, "runtime": runtime}
    anchor_sha = sha(canonical(anchor))
    descriptors = np.asarray(descriptors, dtype=descriptors.dtype.newbyteorder("<"))
    classes, row_ids = {}, []
    for row in range(count):
        content = {"xy_f64_le": np.asarray(points[row], dtype="<f8").tobytes().hex(),
                   "size_f64_le": np.asarray(sizes[row], dtype="<f8").tobytes().hex(),
                   "descriptor_dtype": descriptors.dtype.str, "descriptor_shape": [descriptors.shape[1]],
                   "descriptor_le": descriptors[row].tobytes().hex()}
        identifier = sha(canonical({"schema": INVENTORY_SCHEMA, "anchor_sha256": anchor_sha, "content": content}))
        row_ids.append(identifier)
        if identifier not in classes:
            classes[identifier] = {"feature_id": identifier, "content": content, "xy": points[row].tolist(),
                                   "size": float(sizes[row]), "members": [], "multiplicity": 0}
        classes[identifier]["members"].append(row)
        classes[identifier]["multiplicity"] += 1
    ordered = [classes[key] for key in sorted(classes)]
    requested = 2 * len(ordered)
    if requested > LIMITS["membership_checks"]:
        raise Refusal("membership_limit")
    regions, charged = [], 0
    height, width = frame.pixels.shape[:2]
    for entry, mask in masks:
        bins = [[] for _ in range(16)]
        eligible = []
        for feature in ordered:
            x, y = feature["xy"]
            charged += 1
            if mask.pixels[int(y), int(x)]:
                eligible.append(feature)
                bins[int(y * 4 / height) * 4 + int(x * 4 / width)].append(feature["feature_id"])
        coordinates = {tuple(f["xy"]) for f in eligible}
        regions.append({"source_row": entry["row_refs"][0]["source_row"], "endpoint": entry["endpoint"],
                        "region_id": entry["mask"]["source_region_id"], "mask_sha256": mask.mask_sha256,
                        "header_packed_sha256": entry["mask"]["mask_sha256"], "mask_pixel_count": entry["mask"]["pixel_count"],
                        "reading_binding_sha256": entry["reading_binding_sha256"], "ingress_sha256": entry["ingress_sha256"],
                        "eligible_distinct_classes": len(eligible), "eligible_distinct_coordinates": len(coordinates),
                        "outside_classes": len(ordered) - len(eligible), "bins": bins,
                        "occupied_bins": sum(bool(b) for b in bins),
                        "four_distinct_points_in_general_position": general_position(coordinates)})
    density = {"anchor": anchor, "anchor_sha256": anchor_sha, "classes": ordered,
               "row_ids": row_ids, "descriptor_shape": list(descriptors.shape), "descriptor_dtype": descriptors.dtype.str,
               "descriptor_table_sha256": sha(descriptors.tobytes(order="C")),
               "ordered_table_sha256": sha(canonical(row_ids) + descriptors.tobytes(order="C"))}
    counts = {"raw_returned": count + dropped, "retained_rows": count, "boundary_dropped": dropped,
              "distinct_classes": len(ordered), "exact_duplicate_rows": count - len(ordered),
              "distinct_coordinates": len({tuple(f["xy"]) for f in ordered}),
              "membership_requested": requested, "membership_charged": charged,
              "membership_unprocessed": requested - charged, "cross_view_comparisons": 0}
    body = {"schema": SCHEMA, "status": "sufficient_feature_supply" if regions[1]["four_distinct_points_in_general_position"] else "insufficient_feature_supply",
            "sources": bindings, "inventory": density, "counts": counts, "regions": regions, "limits": LIMITS,
            "interpretation": "Feature supply only. No correspondence, association, camera, metric or accuracy conclusion."}
    return finalize(body)


def collect(frame, masks, bindings, *, services=None):
    services = services or _services()
    _, _, correspondence, _, roster, _ = services
    before = canonical(roster._frame_source(frame))
    for _, mask in masks:
        roster._region_record(mask, frame)
    config = settings(roster)
    features = correspondence.detect_features("image-0000", frame.pixels,
                                              usable_mask=frame.usable_mask, settings=config)
    if canonical(roster._frame_source(frame)) != before:
        raise Refusal("frame_mutated_during_detection")
    return inventory(frame, masks, features, bindings, services=services, config=config)


def validate(record):
    body = dict(record)
    if (body.pop("record_sha256", None) != sha(canonical(body)) or record.get("schema") != SCHEMA
            or len(canonical(record)) + 1 > LIMITS["evidence_bytes"] or record["limits"] != LIMITS):
        raise Refusal("density_record_seal")
    counts = record["counts"]
    inv = record["inventory"]
    anchor = inv["anchor"]
    expected_config = {"max_features": 2048, "mask_boundary_pad_px": 0.0,
                       "min_keypoints_per_frame": 24, "min_matches": 12,
                       "min_mutual_fraction": 0.02, "min_spatial_coverage": 0.2,
                       "ratio_test": 0.75, "spatial_grid": [4, 4]}
    if (anchor["schema"] != INVENTORY_SCHEMA or inv["anchor_sha256"] != sha(canonical(anchor))
            or canonical(anchor["runtime"]["settings"]) != canonical(expected_config)
            or anchor["runtime"]["detector"] != "cv2.SIFT_create(nfeatures=2048)"):
        raise Refusal("inventory_provenance")
    if (counts["raw_returned"] != counts["retained_rows"] + counts["boundary_dropped"]
            or counts["raw_returned"] > 2048 or counts["retained_rows"] != len(inv["row_ids"])
            or counts["distinct_classes"] != len(inv["classes"])
            or counts["exact_duplicate_rows"] != counts["retained_rows"] - counts["distinct_classes"]
            or counts["membership_requested"] != 2 * counts["distinct_classes"]
            or counts["membership_charged"] != counts["membership_requested"]
            or counts["membership_charged"] > 4096 or counts["membership_unprocessed"] != 0
            or counts["cross_view_comparisons"] != 0):
        raise Refusal("density_counter_closure")
    rows = [None] * counts["retained_rows"]
    seen_ids = set()
    for feature in inv["classes"]:
        content = feature["content"]
        identifier = sha(canonical({"schema": INVENTORY_SCHEMA, "anchor_sha256": inv["anchor_sha256"], "content": content}))
        if (feature["feature_id"] != identifier or identifier in seen_ids
                or list(struct.unpack("<dd", bytes.fromhex(content["xy_f64_le"]))) != feature["xy"]
                or struct.unpack("<d", bytes.fromhex(content["size_f64_le"]))[0] != feature["size"]
                or feature["multiplicity"] != len(feature["members"])):
            raise Refusal("feature_content_seal")
        seen_ids.add(identifier)
        descriptor = bytes.fromhex(content["descriptor_le"])
        if (content["descriptor_dtype"] != inv["descriptor_dtype"]
                or content["descriptor_shape"] != [128]
                or len(descriptor) != 128 * int(inv["descriptor_dtype"][-1])):
            raise Refusal("descriptor_content_seal")
        for row in feature["members"]:
            if type(row) is not int or not 0 <= row < len(rows) or rows[row] is not None or inv["row_ids"][row] != identifier:
                raise Refusal("feature_row_seal")
            rows[row] = descriptor
    if any(row is None for row in rows):
        raise Refusal("feature_row_seal")
    table = b"".join(rows)
    if inv["descriptor_table_sha256"] != sha(table) or inv["ordered_table_sha256"] != sha(canonical(inv["row_ids"]) + table):
        raise Refusal("feature_table_seal")
    if len(record["regions"]) != 2 or [r["endpoint"] for r in record["regions"]] != [["image-0000", 2], ["image-0000", 3]]:
        raise Refusal("density_region_set")
    features = {f["feature_id"]: f for f in inv["classes"]}
    for region in record["regions"]:
        ids = [identifier for b in region["bins"] for identifier in b]
        points = {tuple(features[key]["xy"]) for key in ids}
        if (len(region["bins"]) != 16 or len(set(ids)) != len(ids)
                or len(ids) != region["eligible_distinct_classes"]
                or region["eligible_distinct_classes"] + region["outside_classes"] != len(features)
                or region["occupied_bins"] != sum(bool(b) for b in region["bins"])
                or region["eligible_distinct_coordinates"] != len(points)
                or region["four_distinct_points_in_general_position"] != general_position(points)):
            raise Refusal("density_region_closure")
    expected_status = "sufficient_feature_supply" if record["regions"][1]["four_distinct_points_in_general_position"] else "insufficient_feature_supply"
    if record["status"] != expected_status:
        raise Refusal("density_status")


def _infrastructure():
    sys.path.insert(0, str(ROOT))
    from scripts.evaluation import acquire_endpoint_conflict_native_roster as infrastructure
    return infrastructure


def _worker():
    record, baseline, bindings = authorize()
    services = _services()
    services[5]._validate_mask_diagnostic(record)  # Verify sealed masks before decode.
    data = pinned(ROOT / INPUT, INPUT_SHA, 8 * 1024**2)
    frame = bind_frame(data, record["ingress"][0], services=services)
    if frame.source["decoded_sha256"] != baseline["roster"]["frames"][0]["source"]["decoded_sha256"]:
        raise Refusal("baseline_raw_pixel_join")
    masks = bind_masks(record, frame, services=services)
    bindings = {**bindings, "ingress": record["ingress"][0], "mask_bundle_sha256": record["source_region_masks"]["record_sha256"]}
    result = collect(frame, masks, bindings, services=services)
    after, after_baseline, after_bindings = authorize()
    if canonical(after) != canonical(record) or canonical(after_baseline) != canonical(baseline):
        raise Refusal("postflight_artifact_changed")
    if any(bindings[key] != value for key, value in after_bindings.items()):
        raise Refusal("postflight_authorization_changed")
    pinned(ROOT / INPUT, INPUT_SHA, 8 * 1024**2)
    _infrastructure().publish(ROOT / OUTPUT / "density.json", result, validate)


def supervise():
    started = time.monotonic()
    _, _, bindings = authorize()
    directory = ROOT / OUTPUT
    if directory.exists():
        raise Refusal("output_already_exists")
    infrastructure = _infrastructure()
    job = infrastructure._WindowsJob()
    process = None
    peak_rss = peak_commit = peak_output = 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "worker.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-B", "-c",
                "from scripts.evaluation.probe_native_feature_density import _worker; _worker()"],
                cwd=ROOT, stdout=log, stderr=log, creationflags=0x4,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            job.attach_and_resume(process)
            while True:
                rss, commit = job.snapshot()
                size = infrastructure.artifact_size(directory)
                peak_rss, peak_commit, peak_output = max(peak_rss, rss), max(peak_commit, commit), max(peak_output, size)
                infrastructure.budget_check(time.monotonic() - started, rss, size)
                if process.poll() is not None:
                    break
                time.sleep(0.01)
        if process.returncode != 0:
            raise Refusal("worker_failure: " + str(process.returncode))
        data = read(directory / "density.json", LIMITS["evidence_bytes"])
        record = sealed_source(data)
        validate(record)
        _, _, after_bindings = authorize()
        if bindings != after_bindings or any(record["sources"][key] != value for key, value in bindings.items()):
            raise Refusal("completion_source_changed")
        pinned(ROOT / INPUT, INPUT_SHA, 8 * 1024**2)
        completion = finalize({"schema": "native-feature-density-completion-v1", "status": record["status"],
                    "density_sha256": sha(data), "density_bytes": len(data), "sources": bindings,
                    "wall_seconds_before_completion": time.monotonic() - started,
                    "peak_sampled_highwater_rss_bytes": peak_rss, "peak_job_committed_memory_bytes": peak_commit,
                    "peak_artifact_name_bytes": peak_output, "limits": LIMITS,
                    "limits_note": "10ms samples and kernel child-tree committed limit; no unsampled RSS or power-loss guarantee"})
        def validate_completion(value):
            if value != completion:
                raise Refusal("completion_readback")
            infrastructure.budget_check(time.monotonic() - started, job.snapshot()[0], infrastructure.artifact_size(directory))
        infrastructure.publish(directory / "completion.json", completion, validate_completion, remove_failed_completion=True)
        print(record["status"] + ": " + str(directory / "completion.json"))
        return 0 if record["status"] == "sufficient_feature_supply" else 2
    finally:
        job.close()
        if process is not None and process.poll() is None:
            process.wait(timeout=5)


def main():
    if len(sys.argv) != 1:
        raise Refusal("fixed_input_no_overrides")
    return supervise()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError, TypeError, KeyError) as error:
        print("TERMINAL: " + str(error), file=sys.stderr)
        raise SystemExit(1)
