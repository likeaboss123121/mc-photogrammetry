"""Evaluation-only complete descriptor rosters in a declared native raster domain.

No physical identity is inferred. Content classes are permutation-stable only
within a frozen extraction. Caller source seals authenticate supplied evidence;
they do not attest acquisition completeness. Region membership uses floor(x,y)
in source-bound native masks; detector coordinates themselves are never altered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings, FrameFeatures, _k_nearest, detect_features,
)

DOMAIN = "native_exif_oriented_rgb__opencv_keypoint_pt_v1"
SCHEMA = "native-point-content-roster-v1"
LIMITS = {"frames": 2, "groups": 3, "queries_per_group": 4, "features": 512,
          "comparisons": 6144, "pairs_per_group": 32, "pairs": 96, "bytes": 4194304}
SETTINGS = CorrespondenceSettings(max_features=512, mask_boundary_pad_px=0.0)
QUERY_RULE = "row-major 4x4 spatial bin, then content SHA256; first 4 classes; floor membership"
_SOURCE_PIN = "192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_digest(array: np.ndarray) -> str:
    """Hash exact C-order supplied bytes; shape/dtype are separately bound."""
    return digest(array.tobytes(order="C"))


class Refusal(ValueError):
    def __init__(self, reason: str, detail: object = None):
        self.reason, self.detail = reason, detail
        super().__init__(reason)


@dataclass(frozen=True)
class NativeFrame:
    frame_id: str
    encoded_bytes: bytes
    pixels: np.ndarray
    usable_mask: np.ndarray
    source: dict
    transform: dict | None = None
    domain: str = DOMAIN
    raster_kind: str = "native"


@dataclass(frozen=True)
class RegionMask:
    frame_id: str
    source_pixel_sha256: str
    pixels: np.ndarray
    mask_sha256: str
    derivation: str


@dataclass(frozen=True)
class RegionGroup:
    group_id: str
    left: RegionMask | None
    right: RegionMask | None


@dataclass(frozen=True)
class Inventory:
    frame: NativeFrame
    payload: dict
    descriptors: np.ndarray
    seal: str
    pixels_owner: np.ndarray
    mask_owner: np.ndarray
    table_owner: np.ndarray


@dataclass
class ComparisonBudget:
    charged: int = 0
    requested: int = 0

    def reserve(self, units: int) -> None:
        if type(units) is not int or units < 0:
            raise Refusal("invalid_work_units")
        self.requested += units
        if self.charged + units > LIMITS["comparisons"]:
            raise Refusal("comparison_limit", {"requested": self.requested,
                          "charged": self.charged, "uncharged": units})
        self.charged += units


def _owned(array: object, dtype: object, shape: tuple | None = None) -> None:
    if (not isinstance(array, np.ndarray) or array.dtype != dtype
            or not array.flags.owndata or array.base is not None
            or array.flags.writeable or not array.flags.c_contiguous
            or (shape is not None and array.shape != shape)):
        raise Refusal("mutable_or_unowned_source")


def identity(width: int, height: int, transform: dict | None) -> dict:
    sizes = [width, height]
    normalized = {"source_size": sizes, "crop_origin": [0, 0],
                  "crop_size": sizes, "output_size": sizes}
    if transform is None:
        return normalized
    if (type(transform) is not dict
            or set(transform) != set(normalized)):
        raise Refusal("nonidentity_transform")
    for key, target in normalized.items():
        value = transform[key]
        if key in ("crop_size", "output_size") and value is None:
            value = sizes
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or any(type(v) is not int for v in value) or list(value) != target):
            raise Refusal("nonidentity_transform")
    return normalized


def _frame_source(frame: NativeFrame) -> dict:
    if (not isinstance(frame, NativeFrame) or frame.domain != DOMAIN
            or frame.raster_kind != "native"):
        raise Refusal("unsupported_raster_domain")
    pixels = frame.pixels
    _owned(pixels, np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape) < 1:
        raise Refusal("invalid_rgb_shape")
    height, width = pixels.shape[:2]
    transform = identity(width, height, frame.transform)
    _owned(frame.usable_mask, np.bool_, (height, width))
    if (type(frame.frame_id) is not str or not frame.frame_id
            or type(frame.encoded_bytes) is not bytes or not frame.encoded_bytes):
        raise Refusal("invalid_frame_identity")
    expected = {"frame_id": frame.frame_id, "encoded_sha256": digest(frame.encoded_bytes),
                "encoded_length": len(frame.encoded_bytes), "decoded_sha256": array_digest(pixels),
                "shape": list(pixels.shape), "dtype": pixels.dtype.str,
                "mask_sha256": array_digest(frame.usable_mask), "usable_true": True,
                "transform": transform, "domain": DOMAIN, "raster_kind": "native",
                "decoder": "input.image.load_image:exif_transpose+convert('RGB')"}
    try:
        matches = type(frame.source) is dict and canonical(frame.source) == canonical(expected)
    except (ValueError, TypeError):
        matches = False
    if not matches:
        raise Refusal("source_binding_mismatch")
    return expected


@lru_cache(maxsize=8)
def _runtime_bytes(settings: CorrespondenceSettings) -> bytes:
    if type(settings) is not CorrespondenceSettings or settings.max_features != 512:
        raise Refusal("detector_config_mismatch")
    if not np.isfinite(settings.mask_boundary_pad_px):
        raise Refusal("detector_config_mismatch")
    import mc_photogrammetry.pipeline.correspondence as source
    source_digest = digest(Path(source.__file__).read_bytes())
    if source_digest != _SOURCE_PIN:
        raise Refusal("detector_source_changed")
    cv_binary = next(iter(sorted(Path(cv2.__file__).parent.glob("*.pyd"))), None)
    decoder_file = Path(source.__file__).parents[1] / "input" / "image.py"
    decoder_digest = digest(decoder_file.read_bytes())
    if decoder_digest != "43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732":
        raise Refusal("decoder_source_changed")
    numpy_binary = sorted((Path(np.__file__).parent / "_core").glob("*multiarray_umath*.pyd"))
    return canonical({"python": platform.python_version(), "python_executable": sys.executable,
            "numpy": np.__version__, "opencv": cv2.__version__,
            "numpy_distance_binary_sha256": digest(numpy_binary[0].read_bytes()) if numpy_binary else None,
            "decoder_source_sha256": decoder_digest,
            "opencv_binary_sha256": digest(cv_binary.read_bytes()) if cv_binary else None,
            "opencv_build_sha256": digest(cv2.getBuildInformation().encode()),
            "opencv_threads": cv2.getNumThreads(), "correspondence_sha256": source_digest,
            "settings": asdict(settings), "detector": "cv2.SIFT_create(nfeatures=512)",
            "preprocess": "RGB2GRAY.copy; grey[~usable]=0; mask=(usable*255).uint8",
            "intermediate_raster_digest": None,
            "intermediate_digest_disposition": "exact detector grayscale buffer not retained"})


def _runtime(settings: CorrespondenceSettings) -> dict:
    runtime = json.loads(_runtime_bytes(settings))
    runtime["opencv_threads"] = cv2.getNumThreads()
    return runtime


def _frozen(array: np.ndarray) -> np.ndarray:
    # Bytes-backed tables cannot regain write access through a NumPy alias.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _inventory_seal(payload: dict, descriptors: np.ndarray) -> str:
    return digest(canonical(payload) + descriptors.tobytes(order="C"))


def capture_inventory(frame: NativeFrame, *, settings: CorrespondenceSettings = SETTINGS) -> Inventory:
    anchor = _frame_source(frame)
    runtime = _runtime(settings)
    features = detect_features(frame.frame_id, frame.pixels,
                               usable_mask=frame.usable_mask, settings=settings)
    if canonical(_frame_source(frame)) != canonical(anchor):
        raise Refusal("source_mutated_during_detection")
    if (type(features) is not FrameFeatures or features.frame_id != frame.frame_id
            or (features.width, features.height) != (frame.pixels.shape[1], frame.pixels.shape[0])
            or features.usable_pixels != int(np.count_nonzero(frame.usable_mask))):
        raise Refusal("feature_source_mismatch")
    points, descriptors, sizes = features.points, features.descriptors, features.sizes
    if (not isinstance(points, np.ndarray) or points.dtype != np.float64
            or points.ndim != 2 or points.shape[1:] != (2,)
            or not isinstance(descriptors, np.ndarray) or descriptors.ndim != 2
            or descriptors.dtype.kind != "f" or descriptors.dtype.itemsize not in (4, 8)
            or not isinstance(sizes, np.ndarray) or sizes.dtype != np.float64
            or sizes.shape != (len(points),) or len(descriptors) != len(points)
            or (len(points) > 0 and descriptors.shape[1] == 0)):
        raise Refusal("feature_alignment")
    dropped = features.boundary_dropped
    if type(dropped) is not int or dropped < 0:
        raise Refusal("invalid_boundary_exclusion")
    count = len(points)
    if count + dropped > LIMITS["features"]:
        raise Refusal("feature_limit", {"retained": count, "boundary_dropped": dropped,
                                       "raw_detected": count + dropped, "unprocessed": count})
    if (not np.isfinite(points).all() or not np.isfinite(descriptors).all()
            or not np.isfinite(sizes).all() or (sizes <= 0).any()
            or (points < 0).any() or (points[:, 0] >= features.width).any()
            or (points[:, 1] >= features.height).any()):
        raise Refusal("invalid_feature_numeric")
    anchor = {**anchor, "runtime": runtime}
    anchor_hash = digest(canonical(anchor))
    dtype = descriptors.dtype.newbyteorder("<")
    descriptors = _frozen(np.asarray(descriptors, dtype=dtype))
    classes: dict[str, dict] = {}
    row_ids = []
    for row in range(count):
        content = {"schema": "exact-retained-feature-le-v1",
                   "xy_f64_le": np.asarray(points[row], dtype="<f8").tobytes().hex(),
                   "size_f64_le": np.asarray(sizes[row], dtype="<f8").tobytes().hex(),
                   "descriptor_dtype": dtype.str, "descriptor_shape": [descriptors.shape[1]],
                   "descriptor_le": descriptors[row].tobytes().hex()}
        feature_id = digest(canonical({"schema": SCHEMA, "anchor": anchor_hash, "content": content}))
        row_ids.append(feature_id)
        if feature_id not in classes:
            classes[feature_id] = {"feature_id": feature_id, "content": content,
                                   "xy": points[row].tolist(), "size": float(sizes[row]),
                                   "members": [], "multiplicity": 0}
        classes[feature_id]["members"].append(row)
        classes[feature_id]["multiplicity"] += 1
    canonical_classes = [classes[key] for key in sorted(classes)]
    inventory = [{"feature_id": c["feature_id"], "multiplicity": c["multiplicity"]}
                 for c in canonical_classes]
    payload = {"source": anchor, "source_anchor_sha256": anchor_hash,
               "canonical_inventory_sha256": digest(canonical(inventory)),
               "classes": canonical_classes, "retained_row_index_to_content_id": row_ids,
               "ordered_table_sha256": digest(canonical(row_ids) + descriptors.tobytes()),
               "descriptor_table_sha256": array_digest(descriptors),
               "descriptor_dtype": dtype.str, "descriptor_shape": list(descriptors.shape),
               "retained_count": count, "raw_detected_count": count + dropped,
               "exclusions": {"boundary_dropped": dropped,
                              "original_detector_indices": "not_retained_by_FrameFeatures",
                              "angle_octave_response": "not_retained_by_FrameFeatures",
                              "detector_nfeatures_requested": 512,
                              "full_scene_completeness": "unknown"}}
    return Inventory(frame, payload, descriptors, _inventory_seal(payload, descriptors),
                     frame.pixels, frame.usable_mask, descriptors)


def _verify_inventory(inventory: Inventory) -> None:
    if (type(inventory) is not Inventory or inventory.descriptors.flags.writeable
            or inventory.descriptors is not inventory.table_owner
            or _inventory_seal(inventory.payload, inventory.descriptors) != inventory.seal):
        raise Refusal("inventory_substitution")
    if (inventory.frame.pixels is not inventory.pixels_owner
            or inventory.frame.usable_mask is not inventory.mask_owner):
        raise Refusal("source_buffer_replaced")
    expected = _frame_source(inventory.frame)
    captured = {k: v for k, v in inventory.payload["source"].items() if k != "runtime"}
    if canonical(expected) != canonical(captured):
        raise Refusal("source_binding_mismatch")


def _region_record(mask: RegionMask | None, frame: NativeFrame) -> dict:
    if mask is None:
        return {"membership": "unknown", "reason": "no source-bound region mask"}
    if (type(mask) is not RegionMask or mask.frame_id != frame.frame_id
            or mask.source_pixel_sha256 != frame.source["decoded_sha256"]
            or type(mask.derivation) is not str or not mask.derivation):
        raise Refusal("region_source_mismatch")
    _owned(mask.pixels, np.bool_, frame.pixels.shape[:2])
    if mask.mask_sha256 != array_digest(mask.pixels):
        raise Refusal("region_mask_mismatch")
    return {"frame_id": mask.frame_id, "source_pixel_sha256": mask.source_pixel_sha256,
            "mask_sha256": mask.mask_sha256, "shape": list(mask.pixels.shape),
            "derivation": mask.derivation, "membership": "floor(x,y); True included"}


def _membership(mask: RegionMask | None, feature: dict) -> bool | None:
    if mask is None:
        return None
    x, y = feature["xy"]
    return bool(mask.pixels[int(np.floor(y)), int(np.floor(x))])


def _packet_refusal(reason: str, counters: dict, detail: object = None) -> dict:
    packet = {"schema": SCHEMA, "status": "refused", "reason": reason, "detail": detail,
              "frames": [], "groups": [], "counters": counters, "limits": LIMITS.copy()}
    if len(canonical(packet)) > LIMITS["bytes"]:
        raw = canonical(detail)
        packet["detail"] = {"disposition": "oversize diagnostic retained by hash only",
                            "bytes": len(raw), "sha256": digest(raw)}
    return packet


def candidate_roster(inventories: tuple[Inventory, ...], groups: tuple[RegionGroup, ...]) -> dict:
    counters = {"comparisons_charged": 0, "comparisons_requested": 0,
                "comparisons_uncharged": 0, "comparisons_unprocessed": 0,
                "alternatives_counted": 0, "alternatives_exposed": 0,
                "queries_selected": 0, "queries_unprocessed": 0}
    budget = ComparisonBudget()
    try:
        if len(inventories) != 2 or not 1 <= len(groups) <= 3:
            raise Refusal("request_cardinality")
        for inv in inventories:
            _verify_inventory(inv)
        left, right = inventories
        if left.frame.frame_id == right.frame.frame_id:
            raise Refusal("frame_identity_collision")
        if (right.payload["retained_count"] and left.payload["retained_count"]
                and right.descriptors.shape[1] != left.descriptors.shape[1]):
            raise Refusal("descriptor_dimension_mismatch")
        if (len({g.group_id for g in groups}) != len(groups)
                or any(type(g.group_id) is not str or not g.group_id for g in groups)):
            raise Refusal("group_identity")
        plans = []
        # Complete region validation and query selection precede ALL descriptor work.
        for group in sorted(groups, key=lambda g: g.group_id):
            source = {"left": _region_record(group.left, left.frame),
                      "right": _region_record(group.right, right.frame)}
            eligible, outside = [], []
            for feature in left.payload["classes"]:
                if _membership(group.left, feature) is False:
                    outside.append(feature["feature_id"])
                else:
                    eligible.append(feature)
            width, height = left.frame.pixels.shape[1], left.frame.pixels.shape[0]
            def key(feature: dict) -> tuple:
                x, y = feature["xy"]
                return (int(y * 4 / height) * 4 + int(x * 4 / width), feature["feature_id"])
            eligible.sort(key=key)
            selected = eligible[:4]
            plans.append((group, source, selected, {
                "query_limit": [f["feature_id"] for f in eligible[4:]],
                "outside_region": outside}))
        counters["queries_selected"] = sum(len(plan[2]) for plan in plans)
        work = counters["queries_selected"] * right.payload["retained_count"]
        counters["comparisons_requested"] = work
        counters["comparisons_unprocessed"] = work
        counters["queries_unprocessed"] = counters["queries_selected"]
        # Reserve the entire plan atomically, so no partial request is charged/exposed.
        if work > LIMITS["comparisons"]:
            raise Refusal("comparison_limit")
        output_groups = []
        for group, source, selected, omissions in plans:
            alternatives = []
            per_query = []
            for query in selected:
                train_count = len(right.descriptors)
                budget.reserve(train_count)
                counters["comparisons_charged"] = budget.charged
                counters["comparisons_unprocessed"] -= train_count
                counters["queries_unprocessed"] -= 1
                if train_count:
                    indices, distances = _k_nearest(
                        left.descriptors[query["members"][0]:query["members"][0] + 1],
                        right.descriptors, neighbours=train_count)
                    by_row = dict(zip(indices[0].tolist(), distances[0].tolist()))
                    if len(by_row) != train_count or not np.isfinite(distances).all():
                        raise Refusal("incomplete_distance_inventory")
                    candidates = [(by_row[f["members"][0]], f) for f in right.payload["classes"]]
                    candidates.sort(key=lambda c: (c[0], c[1]["feature_id"]))
                    cutoff = candidates[min(5, len(candidates)) - 1][0]
                    retained = [(d, f) for d, f in candidates if d <= cutoff]
                else:
                    cutoff, retained = None, []
                pair_ids = []
                for distance, target in retained:
                    pair_id = digest(canonical({"schema": SCHEMA, "left": query["feature_id"],
                                                "right": target["feature_id"]}))
                    pair_ids.append(pair_id)
                    memberships = {g.group_id: {"left": _membership(g.left, query),
                                               "right": _membership(g.right, target)} for g in groups}
                    alternatives.append({"pair_id": pair_id, "left_id": query["feature_id"],
                                         "right_id": target["feature_id"], "query_id": query["feature_id"],
                                         "left_xy": query["xy"], "right_xy": target["xy"],
                                         "descriptor_distance": distance, "memberships": memberships,
                                         "left_multiplicity": query["multiplicity"],
                                         "right_multiplicity": target["multiplicity"],
                                         "support_classes": 1})
                counters["alternatives_counted"] += len(retained)
                per_query.append({"query_id": query["feature_id"], "cutoff_distance": cutoff,
                                  "all_cutoff_ties": True, "candidate_domain_classes": len(right.payload["classes"]),
                                  "retained_pair_ids": pair_ids,
                                  "omitted_above_cutoff": len(right.payload["classes"]) - len(retained),
                                  "mutually_exclusive_alternatives": True})
                if len(alternatives) > 32:
                    raise Refusal("group_pair_limit", {"group_id": group.group_id,
                                                       "group_counted": len(alternatives)})
                if counters["alternatives_counted"] > 96:
                    raise Refusal("total_pair_limit")
            output_groups.append({"group_id": group.group_id, "source": source,
                                  "query_ids": [f["feature_id"] for f in selected],
                                  "query_omissions": omissions, "queries": per_query,
                                  "alternatives": alternatives})
        for inv in inventories:
            _verify_inventory(inv)
        for group, source, _, _ in plans:
            if canonical(source) != canonical({"left": _region_record(group.left, left.frame),
                                               "right": _region_record(group.right, right.frame)}):
                raise Refusal("region_mutated_during_comparison")
        counters["alternatives_exposed"] = counters["alternatives_counted"]
        packet = {"schema": SCHEMA, "status": "ready", "domain": DOMAIN,
                  "frames": [inv.payload for inv in inventories], "groups": output_groups,
                  "limits": LIMITS.copy(), "query_rule": QUERY_RULE, "k": 5,
                  "right_candidate_domain": "complete usable retained inventory; region labels do not filter",
                  "completeness": "bounded extraction and spatial query domain only",
                  "physical_correctness": "unknown", "counters": counters}
        if len(canonical(packet)) > LIMITS["bytes"]:
            raise Refusal("output_limit")
        # Detach all mutable dict/list references from inventories and source envelopes.
        return json.loads(canonical(packet))
    except Refusal as refusal:
        counters["alternatives_exposed"] = 0
        counters["comparisons_uncharged"] = max(
            0, counters["comparisons_requested"] - counters["comparisons_charged"])
        return _packet_refusal(refusal.reason, counters, refusal.detail)


def acquire_roster(frames: tuple[NativeFrame, ...], groups: tuple[RegionGroup, ...],
                   *, settings: CorrespondenceSettings = SETTINGS) -> dict:
    try:
        if len(frames) != 2 or not 1 <= len(groups) <= 3:
            raise Refusal("request_cardinality")
        # All source envelopes are validated before the first detector invocation.
        for frame in frames:
            _frame_source(frame)
        for group in groups:
            _region_record(group.left, frames[0])
            _region_record(group.right, frames[1])
        inventories = tuple(capture_inventory(frame, settings=settings) for frame in frames)
        return candidate_roster(inventories, groups)
    except Refusal as refusal:
        return _packet_refusal(refusal.reason,
                               {"comparisons_charged": 0, "comparisons_requested": 0,
                                "comparisons_uncharged": 0, "comparisons_unprocessed": 0,
                                "alternatives_counted": 0, "alternatives_exposed": 0,
                                "queries_selected": 0, "queries_unprocessed": 0}, refusal.detail)


def write_packet(path: str | Path, packet: dict) -> None:
    """No-clobber publication of a complete, bounded ready packet only."""
    if packet.get("status") != "ready" or len(canonical(packet)) > LIMITS["bytes"]:
        raise Refusal("not_publishable")
    with Path(path).open("xb") as handle:
        handle.write(canonical(packet))
