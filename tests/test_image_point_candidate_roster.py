"""Single-use, preregistered native synthetic controls; no ordinary images.

The final control publishes a new no-clobber evidence path. Do not rerun this
consumed module without a fresh task/archive disposition.
"""
from dataclasses import replace
import ast
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from mc_photogrammetry.input.image import load_image_bytes
from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from scripts.evaluation import image_point_candidate_roster as subject

REAL_DETECT = subject.detect_features
REAL_DISTANCE = subject._k_nearest
RECEIPT = {}
ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "agent swarm instantiation template/runs/SOL-IMAGE-POINT-ROSTER-02.controls.json"


def owned(array):
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def frame(name="left", pixels=None, encoded=None, transform=None):
    if pixels is None:
        pixels = np.full((64, 64, 3), 127, dtype=np.uint8)
    pixels = owned(pixels)
    if encoded is None:
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        encoded = output.getvalue()
    mask = owned(np.ones(pixels.shape[:2], dtype=bool))
    h, w = pixels.shape[:2]
    source = {"frame_id": name, "encoded_sha256": subject.digest(encoded),
              "encoded_length": len(encoded), "decoded_sha256": subject.array_digest(pixels),
              "shape": list(pixels.shape), "dtype": pixels.dtype.str,
              "mask_sha256": subject.array_digest(mask), "usable_true": True,
              "transform": subject.identity(w, h, None), "domain": subject.DOMAIN,
              "raster_kind": "native", "decoder": "input.image.load_image:exif_transpose+convert('RGB')"}
    return subject.NativeFrame(name, encoded, pixels, mask, source, transform)


def features(source, count=1, descriptors=None, points=None, sizes=None, dropped=0):
    if points is None:
        points = np.array([(i % 32 + 0.125, i // 32 + 0.375) for i in range(count)],
                          dtype=np.float64).reshape(count, 2)
    if descriptors is None:
        descriptors = np.arange(count, dtype=np.float32).reshape(count, 1)
    if sizes is None:
        sizes = np.full(count, 2.125, dtype=np.float64)
    return FrameFeatures(source.frame_id, points, descriptors, source.pixels.shape[1],
                         source.pixels.shape[0], int(source.usable_mask.sum()), sizes, dropped)


def inventories(monkeypatch, left_count=1, right_count=8, left_desc=None, right_desc=None,
                left_features=None, right_features=None):
    left, right = frame(), frame("right")
    tables = {"left": left_features or features(left, left_count, left_desc),
              "right": right_features or features(right, right_count, right_desc)}
    monkeypatch.setattr(subject, "detect_features", lambda name, *args, **kwargs: tables[name])
    return (subject.capture_inventory(left), subject.capture_inventory(right))


def region(source, value=True):
    mask = owned(np.full(source.pixels.shape[:2], value, dtype=bool))
    return subject.RegionMask(source.frame_id, source.source["decoded_sha256"], mask,
                              subject.array_digest(mask), "frozen synthetic native intensity region v1")


def groups(inv, count=1, unknown=False):
    return tuple(subject.RegionGroup(f"g{i}", None if unknown else region(inv[0].frame),
                                    None if unknown else region(inv[1].frame)) for i in range(count))


def refuse(packet, reason):
    assert packet["status"] == "refused" and packet["reason"] == reason
    assert packet["groups"] == [] and packet["frames"] == []
    assert packet["counters"]["alternatives_exposed"] == 0


def logical(packet):
    return [(g["query_ids"], [(p["pair_id"], p["descriptor_distance"])
                              for p in g["alternatives"]]) for g in packet["groups"]]


def test_identity_none_and_explicit_are_canonical(monkeypatch):
    source = frame()
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: features(source))
    first = subject.capture_inventory(source)
    for transform in [source.source["transform"], {"source_size": (64, 64),
                      "crop_origin": (0, 0), "crop_size": None, "output_size": None}]:
        second = subject.capture_inventory(replace(source, transform=transform))
        assert first.payload == second.payload and first.seal == second.seal
    assert first.payload["classes"][0]["xy"] == [0.125, 0.375]
    assert first.payload["source"]["runtime"]["intermediate_raster_digest"] is None


@pytest.mark.parametrize("transform", [
    {}, "unknown", {"source_size": [64, 64], "crop_origin": [1, 0], "crop_size": None, "output_size": None},
    {"source_size": [128, 128], "crop_origin": [0, 0], "crop_size": [64, 64], "output_size": [64, 64]},
    {"source_size": [64, 64], "crop_origin": [0, 0], "crop_size": [32, 64], "output_size": [64, 64]},
    {"source_size": [64, 64], "crop_origin": [0, 0], "crop_size": None, "output_size": [32, 32]},
    {"source_size": [64.0, 64], "crop_origin": [0, 0], "crop_size": None, "output_size": None},
    {"source_size": [64, 64], "crop_origin": [False, 0], "crop_size": None, "output_size": None},
    {"source_size": [64, 64], "crop_origin": [0, 0], "crop_size": None, "output_size": None, "extra": 0},
])
def test_nonidentity_refused_before_detector(monkeypatch, transform):
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    packet = subject.acquire_roster((replace(frame(), transform=transform), frame("right")),
                                   (subject.RegionGroup("g", None, None),))
    refuse(packet, "nonidentity_transform")


@pytest.mark.parametrize("kind,value", [("domain", "pre_exif"), ("domain", "unknown"),
                                       ("raster_kind", "pyramid"), ("raster_kind", "resampled")])
def test_non_native_domains_early(monkeypatch, kind, value):
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    refuse(subject.acquire_roster((replace(frame(), **{kind: value}), frame("right")),
                                 (subject.RegionGroup("g", None, None),)), "unsupported_raster_domain")


@pytest.mark.parametrize("field", ["frame_id", "encoded_bytes", "pixels", "usable_mask"])
def test_input_substitutions_early(monkeypatch, field):
    source = frame()
    replacements = {"frame_id": "swapped", "encoded_bytes": source.encoded_bytes + b"x",
                    "pixels": owned(np.zeros_like(source.pixels)),
                    "usable_mask": owned(np.zeros_like(source.usable_mask))}
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    refuse(subject.acquire_roster((replace(source, **{field: replacements[field]}), frame("right")),
                                 (subject.RegionGroup("g", None, None),)), "source_binding_mismatch")


@pytest.mark.parametrize("field", ["encoded_sha256", "encoded_length", "decoded_sha256", "mask_sha256",
                                   "frame_id", "shape", "dtype", "usable_true", "decoder"])
def test_independent_source_seal_substitution(monkeypatch, field):
    source = frame()
    changed = dict(source.source)
    changed[field] = "substitution"
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    refuse(subject.acquire_roster((replace(source, source=changed), frame("right")),
                                 (subject.RegionGroup("g", None, None),)), "source_binding_mismatch")


@pytest.mark.parametrize("field,alias", [("pixels", False), ("pixels", True),
                                       ("usable_mask", False), ("usable_mask", True)])
def test_owned_immutable_inputs(monkeypatch, field, alias):
    source = frame()
    mutable = np.array(getattr(source, field), copy=True)
    replacement = mutable.view() if alias else mutable
    if alias:
        replacement.setflags(write=False)
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    refuse(subject.acquire_roster((replace(source, **{field: replacement}), frame("right")),
                                 (subject.RegionGroup("g", None, None),)), "mutable_or_unowned_source")


def test_owned_buffer_mutation_is_checked_after_detection(monkeypatch):
    source = frame()
    def detector(*a, **k):
        source.pixels.setflags(write=True)
        source.pixels[0, 0, 0] = 0
        source.pixels.setflags(write=False)
        return features(source)
    monkeypatch.setattr(subject, "detect_features", detector)
    refuse(subject.acquire_roster((source, frame("right")), (subject.RegionGroup("g", None, None),)),
           "source_binding_mismatch")


@pytest.mark.parametrize("change", ["source", "digest", "shape", "mutable"])
def test_region_source_early_refusal(monkeypatch, change):
    left, right = frame(), frame("right")
    mask = region(left)
    if change == "source":
        mask = replace(mask, source_pixel_sha256="0" * 64)
    elif change == "digest":
        mask = replace(mask, mask_sha256="0" * 64)
    elif change == "shape":
        mask = replace(mask, pixels=owned(np.ones((32, 32), dtype=bool)))
    else:
        mask = replace(mask, pixels=np.ones((64, 64), dtype=bool))
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: pytest.fail("early refusal required"))
    reason = {"source": "region_source_mismatch", "digest": "region_mask_mismatch",
              "shape": "mutable_or_unowned_source", "mutable": "mutable_or_unowned_source"}[change]
    refuse(subject.acquire_roster((left, right), (subject.RegionGroup("g", mask, None),)), reason)


def test_content_permutation_keeps_ids_candidates_and_inventory(monkeypatch):
    source = frame()
    original = features(source, 8)
    inv = inventories(monkeypatch, left_features=original, right_count=8)
    packet = subject.candidate_roster(inv, groups(inv))
    permutation = np.array([7, 3, 1, 6, 0, 5, 4, 2])
    table = replace(original, points=original.points[permutation],
                    descriptors=original.descriptors[permutation], sizes=original.sizes[permutation])
    changed = inventories(monkeypatch, left_features=table, right_count=8)
    packet2 = subject.candidate_roster(changed, groups(changed))
    assert logical(packet) == logical(packet2)
    assert inv[0].payload["canonical_inventory_sha256"] == changed[0].payload["canonical_inventory_sha256"]
    assert inv[0].payload["ordered_table_sha256"] != changed[0].payload["ordered_table_sha256"]
    RECEIPT["permutation"] = {"logical_same": True, "inventory_same": True, "ordered_audit_different": True}


def test_right_permutation_cutoff_ties(monkeypatch):
    right = frame("right")
    table = features(right, 8, np.zeros((8, 1), dtype=np.float32))
    inv = inventories(monkeypatch, right_features=table)
    first = subject.candidate_roster(inv, groups(inv))
    order = np.arange(7, -1, -1)
    changed = replace(table, points=table.points[order], descriptors=table.descriptors[order],
                      sizes=table.sizes[order])
    inv2 = inventories(monkeypatch, right_features=changed)
    second = subject.candidate_roster(inv2, groups(inv2))
    assert logical(first) == logical(second)
    assert inv[1].payload["canonical_inventory_sha256"] == inv2[1].payload["canonical_inventory_sha256"]


def test_exact_duplicate_class_multiplicity_is_not_independent_support(monkeypatch):
    right = frame("right")
    table = features(right, 2, np.zeros((2, 1), np.float32),
                     np.array([[0.125, 0.375], [0.125, 0.375]], dtype=np.float64))
    inv = inventories(monkeypatch, right_features=table)
    packet = subject.candidate_roster(inv, groups(inv))
    assert len(inv[1].payload["classes"]) == 1
    assert inv[1].payload["classes"][0]["members"] == [0, 1]
    pairs = packet["groups"][0]["alternatives"]
    assert len(pairs) == 1 and pairs[0]["right_multiplicity"] == 2 and pairs[0]["support_classes"] == 1
    assert packet["counters"]["comparisons_charged"] == 2
    RECEIPT["duplicate"] = {"raw_rows": 2, "classes": 1, "alternatives": 1, "multiplicity": 2}


@pytest.mark.parametrize("change", ["xy", "descriptor", "size"])
def test_tiny_content_change_changes_id(monkeypatch, change):
    source = frame()
    table = features(source)
    inv = inventories(monkeypatch, left_features=table)
    changed = replace(table, points=table.points.copy(), descriptors=table.descriptors.copy(),
                      sizes=table.sizes.copy())
    if change == "xy":
        changed.points[0, 0] = np.nextafter(changed.points[0, 0], np.inf)
    elif change == "size":
        changed.sizes[0] = np.nextafter(changed.sizes[0], np.inf)
    else:
        changed.descriptors[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    other = inventories(monkeypatch, left_features=changed)
    assert inv[0].payload["classes"][0]["feature_id"] != other[0].payload["classes"][0]["feature_id"]


@pytest.mark.parametrize("change", ["descriptor", "size"])
def test_coincident_distinct_classes(monkeypatch, change):
    source = frame()
    table = features(source, 2, np.zeros((2, 1), np.float32),
                     np.array([[0.125, 0.375], [0.125, 0.375]], np.float64))
    if change == "descriptor":
        table.descriptors[1, 0] = 1
    else:
        table.sizes[1] = 3
    inv = inventories(monkeypatch, left_features=table)
    assert len(inv[0].payload["classes"]) == 2


@pytest.mark.parametrize("field", ["frame_id", "width", "height", "usable_pixels"])
def test_feature_frame_substitution(monkeypatch, field):
    source = frame()
    table = replace(features(source), **{field: "wrong" if field == "frame_id" else 1})
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: table)
    with pytest.raises(subject.Refusal, match="feature_source_mismatch"):
        subject.capture_inventory(source)


@pytest.mark.parametrize("bad", ["missing_size", "size_count", "desc_count", "points_shape", "integer_desc",
                                 "nan_point", "inf_desc", "nan_size", "negative_xy", "high_x", "high_y",
                                 "zero_size", "negative_size", "empty_descriptor_width", "bad_boundary"])
def test_feature_validation(monkeypatch, bad):
    source = frame()
    table = features(source)
    reason = "feature_alignment"
    if bad == "missing_size":
        table = replace(table, sizes=None)
    elif bad == "size_count":
        table = replace(table, sizes=np.empty(0, np.float64))
    elif bad == "desc_count":
        table = replace(table, descriptors=np.empty((0, 1), np.float32))
    elif bad == "points_shape":
        table = replace(table, points=np.zeros((1, 3), np.float64))
    elif bad == "integer_desc":
        table = replace(table, descriptors=np.zeros((1, 1), np.int32))
    elif bad == "empty_descriptor_width":
        table = replace(table, descriptors=np.zeros((1, 0), np.float32))
    elif bad == "bad_boundary":
        table = replace(table, boundary_dropped=-1)
        reason = "invalid_boundary_exclusion"
    else:
        reason = "invalid_feature_numeric"
        if bad == "nan_point":
            table.points[0, 0] = np.nan
        elif bad == "inf_desc":
            table.descriptors[0, 0] = np.inf
        elif bad == "nan_size":
            table.sizes[0] = np.nan
        elif bad == "negative_xy":
            table.points[0, 0] = -0.01
        elif bad == "high_x":
            table.points[0, 0] = 64
        elif bad == "high_y":
            table.points[0, 1] = 64
        elif bad == "zero_size":
            table.sizes[0] = 0
        else:
            table.sizes[0] = -1
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: table)
    with pytest.raises(subject.Refusal, match=reason):
        subject.capture_inventory(source)


def test_feature_input_aliases_are_detached_and_frozen(monkeypatch):
    source = frame()
    table = features(source)
    inv = inventories(monkeypatch, left_features=table)
    original = inv[0].payload["classes"][0]["xy"][:]
    table.points[0] = 3
    table.sizes[0] = 4
    table.descriptors[0] = 5
    assert inv[0].payload["classes"][0]["xy"] == original
    assert inv[0].descriptors[0, 0] == 0
    with pytest.raises(ValueError):
        inv[0].descriptors.setflags(write=True)
    assert subject.candidate_roster(inv, groups(inv))["status"] == "ready"


@pytest.mark.parametrize("field", ["descriptor", "content", "index", "inventory", "frame"])
def test_bound_inventory_substitution_refuses(monkeypatch, field):
    inv = inventories(monkeypatch)
    changed = inv[0]
    if field == "descriptor":
        changed = replace(changed, descriptors=subject._frozen(np.ones_like(changed.descriptors)))
    elif field == "frame":
        changed = replace(changed, frame=replace(changed.frame, pixels=owned(np.zeros_like(changed.frame.pixels))))
    else:
        payload = json.loads(subject.canonical(changed.payload))
        if field == "content":
            payload["classes"][0]["xy"][0] = 1
        elif field == "index":
            payload["retained_row_index_to_content_id"][0] = "0" * 64
        else:
            payload["canonical_inventory_sha256"] = "0" * 64
        changed = replace(changed, payload=payload)
    refuse(subject.candidate_roster((changed, inv[1]), groups(inv)),
           "source_buffer_replaced" if field == "frame" else "inventory_substitution")


@pytest.mark.parametrize("field", ["pixels", "usable_mask", "descriptors"])
def test_identical_replacement_buffer_is_refused_after_binding(monkeypatch, field):
    inv = inventories(monkeypatch)
    changed = inv[0]
    if field == "descriptors":
        changed = replace(changed, descriptors=subject._frozen(changed.descriptors))
    else:
        changed = replace(changed, frame=replace(changed.frame,
                          **{field: owned(getattr(changed.frame, field))}))
    refuse(subject.candidate_roster((changed, inv[1]), groups(inv)),
           "inventory_substitution" if field == "descriptors" else "source_buffer_replaced")


@pytest.mark.parametrize("count", [512, 513])
def test_raw_feature_boundary(monkeypatch, count):
    source = frame()
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: features(source, count))
    if count == 512:
        inv = subject.capture_inventory(source)
        assert inv.payload["raw_detected_count"] == 512
    else:
        with pytest.raises(subject.Refusal, match="feature_limit") as error:
            subject.capture_inventory(source)
        assert error.value.detail["raw_detected"] == 513


def test_boundary_exclusions_are_explicit_and_count_toward_raw_limit(monkeypatch):
    source = frame()
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: features(source, 1, dropped=511))
    inv = subject.capture_inventory(source)
    assert inv.payload["raw_detected_count"] == 512
    assert inv.payload["exclusions"]["boundary_dropped"] == 511
    assert inv.payload["exclusions"]["original_detector_indices"] == "not_retained_by_FrameFeatures"
    monkeypatch.setattr(subject, "detect_features", lambda *a, **k: features(source, 1, dropped=512))
    with pytest.raises(subject.Refusal, match="feature_limit"):
        subject.capture_inventory(source)


@pytest.mark.parametrize("left_count,right_count", [(0, 8), (1, 0), (0, 0)])
def test_empty_feature_inventories(monkeypatch, left_count, right_count):
    inv = inventories(monkeypatch, left_count, right_count)
    packet = subject.candidate_roster(inv, groups(inv))
    assert packet["status"] == "ready" and packet["counters"]["comparisons_charged"] == 0
    assert packet["groups"][0]["alternatives"] == []


def test_query_selection_complete_before_first_kernel_call(monkeypatch):
    inv = inventories(monkeypatch, 8, 8)
    original = subject.candidate_roster(inv, groups(inv, 3))
    expected_queries = [g["query_ids"] for g in original["groups"]]
    calls = []
    def kernel(query, train, neighbours):
        calls.append(query.copy())
        return REAL_DISTANCE(query, train, neighbours)
    monkeypatch.setattr(subject, "_k_nearest", kernel)
    packet = subject.candidate_roster(inv, groups(inv, 3))
    assert [g["query_ids"] for g in packet["groups"]] == expected_queries
    assert len(calls) == 12
    assert all(len(g["query_omissions"]["query_limit"]) == 4 for g in packet["groups"])
    changed_right = inventories(monkeypatch, 8, 8, right_desc=np.arange(8, dtype=np.float32)[::-1, None])
    changed = subject.candidate_roster(changed_right, groups(changed_right, 3))
    assert [g["query_ids"] for g in changed["groups"]] == expected_queries


def test_later_invalid_region_refuses_before_any_comparison(monkeypatch):
    inv = inventories(monkeypatch)
    bad = replace(region(inv[0].frame), mask_sha256="0" * 64)
    declared = groups(inv) + (subject.RegionGroup("late", bad, None),)
    monkeypatch.setattr(subject, "_k_nearest", lambda *a, **k: pytest.fail("query plan must close first"))
    refuse(subject.candidate_roster(inv, declared), "region_mask_mismatch")


def test_finite_descriptor_arithmetic_overflow_is_atomic(monkeypatch):
    inv = inventories(monkeypatch, left_desc=np.array([[1e308]], np.float64),
                      right_desc=np.array([[1e308]] * 8, np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        packet = subject.candidate_roster(inv, groups(inv))
    refuse(packet, "incomplete_distance_inventory")
    assert packet["counters"]["comparisons_charged"] == 8


@pytest.mark.parametrize("problem", ["one_frame", "same_frame_id", "zero_groups", "four_groups", "duplicate_groups"])
def test_request_cardinality_and_identity(monkeypatch, problem):
    inv = inventories(monkeypatch)
    declared = groups(inv)
    reason = "request_cardinality"
    if problem == "one_frame":
        inv = inv[:1]
    elif problem == "same_frame_id":
        inv = (inv[0], inv[0])
        reason = "frame_identity_collision"
    elif problem == "zero_groups":
        declared = ()
    elif problem == "four_groups":
        declared = groups(inv, 4)
    else:
        declared = (declared[0], declared[0])
        reason = "group_identity"
    refuse(subject.candidate_roster(inv, declared), reason)


def test_k5_all_cutoff_ties_use_genuine_full_right_distance(monkeypatch):
    values = np.array([0, 1, 2, 3, 4, 4, 4, 9], np.float32).reshape(8, 1)
    inv = inventories(monkeypatch, 1, 8, right_desc=values)
    packet = subject.candidate_roster(inv, groups(inv))
    assert packet["status"] == "ready"
    alternatives = packet["groups"][0]["alternatives"]
    assert [p["descriptor_distance"] for p in alternatives] == [0, 1, 2, 3, 4, 4, 4]
    query = packet["groups"][0]["queries"][0]
    assert query["cutoff_distance"] == 4 and query["omitted_above_cutoff"] == 1
    assert packet["counters"]["comparisons_charged"] == 8
    RECEIPT["cutoff_ties"] = {"k": 5, "retained": 7, "cutoff": 4, "comparisons": 8}


def test_overlap_and_unknown_memberships_not_deduplicated(monkeypatch):
    inv = inventories(monkeypatch, 1, 8)
    declared = groups(inv, 2) + (subject.RegionGroup("unknown", None, None),)
    packet = subject.candidate_roster(inv, declared)
    assert packet["status"] == "ready"
    assert len(packet["groups"]) == 3
    pairs = [p for g in packet["groups"] for p in g["alternatives"]]
    assert len(pairs) == 15 and len({p["pair_id"] for p in pairs}) == 5
    assert pairs[0]["memberships"] == {"g0": {"left": True, "right": True},
                                        "g1": {"left": True, "right": True},
                                        "unknown": {"left": None, "right": None}}


def test_region_outside_omissions_and_right_annotation(monkeypatch):
    inv = inventories(monkeypatch, 3, 8)
    declared = (subject.RegionGroup("outside", region(inv[0].frame, False), region(inv[1].frame, False)),
                subject.RegionGroup("right_outside", region(inv[0].frame), region(inv[1].frame, False)))
    packet = subject.candidate_roster(inv, declared)
    assert packet["status"] == "ready"
    assert len(packet["groups"][0]["query_omissions"]["outside_region"]) == 3
    assert packet["groups"][0]["alternatives"] == []
    assert packet["groups"][1]["alternatives"][0]["memberships"]["right_outside"]["right"] is False


def test_6144_exact_work_closure(monkeypatch):
    inv = inventories(monkeypatch, 4, 512)
    packet = subject.candidate_roster(inv, groups(inv, 3))
    assert packet["status"] == "ready"
    c = packet["counters"]
    assert c["comparisons_charged"] == c["comparisons_requested"] == 6144
    assert c["comparisons_uncharged"] == c["comparisons_unprocessed"] == c["queries_unprocessed"] == 0
    assert c["queries_selected"] == 12
    RECEIPT["exact_work"] = c


@pytest.mark.parametrize("units", [6144, 6145])
def test_comparison_reservation_boundary(units):
    budget = subject.ComparisonBudget()
    if units == 6144:
        budget.reserve(units)
        assert budget.charged == budget.requested == units
    else:
        with pytest.raises(subject.Refusal, match="comparison_limit") as error:
            budget.reserve(units)
        assert budget.charged == 0 and budget.requested == 6145
        assert error.value.detail == {"requested": 6145, "charged": 0, "uncharged": 6145}


@pytest.mark.parametrize("right_count", [32, 33])
def test_group_tie_boundary_atomic(monkeypatch, right_count):
    inv = inventories(monkeypatch, 1, right_count, right_desc=np.zeros((right_count, 1), np.float32))
    packet = subject.candidate_roster(inv, groups(inv))
    if right_count == 32:
        assert packet["status"] == "ready" and len(packet["groups"][0]["alternatives"]) == 32
    else:
        refuse(packet, "group_pair_limit")
        assert packet["detail"]["group_counted"] == 33
        assert packet["counters"]["comparisons_charged"] == 33
        RECEIPT["tie_overflow"] = packet


def test_96_and_97_total_cardinality_boundaries(monkeypatch):
    inv = inventories(monkeypatch, 1, 32, right_desc=np.zeros((32, 1), np.float32))
    packet = subject.candidate_roster(inv, groups(inv, 3))
    assert packet["status"] == "ready" and packet["counters"]["alternatives_exposed"] == 96
    RECEIPT["exact_alternatives"] = packet["counters"]
    # First two groups choose zero (32 ties); third chooses one (33 ties).
    # Exact 97 is refused at the stricter per-group bound without cap changes.
    larger = inventories(monkeypatch, 2, 33, left_desc=np.array([[0], [1]], np.float32),
                         right_desc=np.array([0] * 32 + [1], np.float32).reshape(33, 1))
    declared = []
    for i in range(3):
        native_mask = np.zeros((64, 64), dtype=bool)
        native_mask[0, 0 if i < 2 else 1] = True
        left_region = replace(region(larger[0].frame), pixels=owned(native_mask),
                              mask_sha256=subject.array_digest(native_mask))
        declared.append(subject.RegionGroup(f"g{i}", left_region, None))
    refused = subject.candidate_roster(larger, tuple(declared))
    refuse(refused, "group_pair_limit")
    assert refused["counters"]["comparisons_requested"] == 99
    assert refused["counters"]["comparisons_charged"] == 99
    assert refused["counters"]["alternatives_counted"] == 97
    assert refused["counters"]["comparisons_unprocessed"] == refused["counters"]["comparisons_uncharged"] == 0
    assert refused["counters"]["queries_unprocessed"] == 0
    RECEIPT["total_overflow_prerequisite"] = refused


def test_output_size_refuses_atomically_without_changing_cap(monkeypatch):
    inv = inventories(monkeypatch)
    group = subject.RegionGroup("x" * subject.LIMITS["bytes"], None, None)
    packet = subject.candidate_roster(inv, (group,))
    refuse(packet, "output_limit")
    assert packet["counters"]["comparisons_charged"] == 8
    assert subject.LIMITS["bytes"] == 4194304
    RECEIPT["output_overflow"] = packet


def test_refusal_not_written_and_ready_no_clobber(monkeypatch, tmp_path):
    inv = inventories(monkeypatch)
    ready = subject.candidate_roster(inv, groups(inv))
    path = tmp_path / "packet.json"
    refused = subject._packet_refusal("control", {})
    with pytest.raises(subject.Refusal, match="not_publishable"):
        subject.write_packet(path, refused)
    assert not path.exists()
    subject.write_packet(path, ready)
    data = path.read_bytes()
    assert json.loads(data) == ready and data == subject.canonical(ready)
    with pytest.raises(FileExistsError):
        subject.write_packet(path, ready)
    assert path.read_bytes() == data


def test_api_imports_and_packet_have_no_geometry_mode_fields(monkeypatch):
    inv = inventories(monkeypatch)
    packet = subject.candidate_roster(inv, groups(inv))
    text = subject.canonical(packet).decode()
    for key in ['"mode"', '"camera"', '"period"', '"translation"', '"rotation"', '"best.matches"']:
        assert key not in text
    tree = ast.parse(Path(subject.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(not any(s in (m or "") for s in ("camera", "geometry", "plane", "mode", "joint_matching"))
               for m in imports)
    with pytest.raises(TypeError):
        subject.candidate_roster(inv, groups(inv), mode="forbidden")


def test_actual_native_detector_coordinates_exact_once_per_input(monkeypatch):
    frames = tuple(frame(name, np.random.default_rng(seed).integers(
        0, 256, (160, 192, 3), dtype=np.uint8)) for name, seed in [("left", 173), ("right", 179)])
    captured = {}
    def detector(name, pixels, **kwargs):
        assert name not in captured
        captured[name] = REAL_DETECT(name, pixels, **kwargs)
        return captured[name]
    monkeypatch.setattr(subject, "detect_features", detector)
    declared = (subject.RegionGroup("native", region(frames[0]), region(frames[1])),)
    packet = subject.acquire_roster(frames, declared)
    assert packet["status"] == "ready" and len(captured) == 2
    assert all(len(f.points) > 4 for f in captured.values())
    for bound in packet["frames"]:
        original = captured[bound["source"]["frame_id"]]
        by_id = {c["feature_id"]: c for c in bound["classes"]}
        restored = np.array([by_id[i]["xy"] for i in bound["retained_row_index_to_content_id"]])
        assert np.array_equal(restored, original.points)
        assert any(np.any(p != np.floor(p)) for p in original.points)
    assert packet["counters"]["comparisons_charged"] <= 6144
    RECEIPT["actual_native_detector"] = {"calls": 2, "coordinates_exact": True,
                                         "packet": packet}


def test_synthetic_exif_oriented_domain_is_native_not_pre_exif(monkeypatch):
    output = BytesIO()
    raw = np.random.default_rng(193).integers(0, 256, (48, 72, 3), dtype=np.uint8)
    image = Image.fromarray(raw)
    exif = Image.Exif()
    exif[274] = 6
    image.save(output, format="JPEG", quality=95, exif=exif)
    encoded = output.getvalue()
    decoded = load_image_bytes(encoded, suffix=".jpg")
    assert decoded.shape == (72, 48, 3) and decoded.flags.owndata
    frames = (frame("oriented-left", decoded, encoded), frame("oriented-right", decoded, encoded))
    captured = {}
    def detector(name, pixels, **kwargs):
        assert name not in captured
        captured[name] = REAL_DETECT(name, pixels, **kwargs)
        return captured[name]
    monkeypatch.setattr(subject, "detect_features", detector)
    packet = subject.acquire_roster(frames, (subject.RegionGroup("exif", None, None),))
    assert packet["status"] == "ready" and len(captured) == 2
    assert all(p["source"]["shape"] == [72, 48, 3] and p["source"]["domain"] == subject.DOMAIN
               for p in packet["frames"])
    RECEIPT["synthetic_exif"] = {"encoded_shape": [48, 72, 3], "decoded_shape": [72, 48, 3],
                                 "detector_calls": 2, "packet": packet}


def test_final_controls_canonical_bounded_receipt():
    assert set(RECEIPT) == {"permutation", "duplicate", "cutoff_ties", "exact_work", "tie_overflow",
                            "exact_alternatives", "total_overflow_prerequisite", "output_overflow",
                            "actual_native_detector", "synthetic_exif"}
    data = subject.canonical(RECEIPT)
    assert len(data) <= 4194304 and json.loads(data) == RECEIPT
    with CONTROLS.open("xb") as handle:
        handle.write(data)
