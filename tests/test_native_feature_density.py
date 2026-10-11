"""Synthetic-only frozen controls for the separately authorized density probe."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
from io import BytesIO
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from mc_photogrammetry.input.image import load_image_bytes
from mc_photogrammetry.pipeline import correspondence
from mc_photogrammetry.pipeline.image_derived_session import image_sha256
from mc_photogrammetry.reconstruction import lattice_planes
from scripts.evaluation import acquire_endpoint_conflict_native_roster as infrastructure
from scripts.evaluation import diagnose_image_geometry_failure as diagnostic
from scripts.evaluation import image_point_candidate_roster as roster
from scripts.evaluation import probe_native_feature_density as subject

SERVICES = (np, load_image_bytes, correspondence, image_sha256, roster, diagnostic)
QUAD = [(8.125, 12.375), (24.125, 12.375), (8.125, 28.375), (24.125, 28.375)]


def fixture(height=64, width=64):
    pixels = np.zeros((height, width, 3), np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    data = stream.getvalue()
    ingress = [{"frame_id": name, "source_label": "synthetic-density-only",
                "byte_sha256": subject.sha(data), "byte_length": len(data),
                "decoded_shape": [height, width, 3], "decoded_dtype": "|u1",
                "decoded_pixel_sha256": image_sha256(pixels),
                "decoder": "input.image.load_image:exif_transpose+convert('RGB')",
                "schema": "mc-photogrammetry-frame-ingress-provenance-v1",
                "transform": {"source_size": [width, height], "crop_origin": [0, 0], "crop_size": None, "output_size": None}}
               for name in ("image-0000", "image-0001")]
    readings, summaries, bindings = [], [], []
    polygon = ((4., 8.), (30., 8.), (30., 32.), (4., 32.))
    for index, endpoint in enumerate(diagnostic.MASK_ENDPOINTS):
        mask = np.zeros((height, width), bool)
        mask[8:32, 4:30] = True
        mask[18:21, 15:18] = False
        mask[30, 29] = False
        region = lattice_planes._make_region_evidence(mask, 1, polygon, (f"synthetic-density-{index}",))
        readings.append(SimpleNamespace(region_evidence=region, polygon=polygon, axis=1, frame=endpoint[0]))
        binding_sha = str(index + 1) * 64
        bindings.append((endpoint, index, 1, binding_sha, (), ()))
        summary = region.to_record(); summary.pop("packed_mask_base64"); summary["type"] = "available"
        summaries.append({"endpoint": list(endpoint), "reading_index": index,
                          "reading_binding_sha256": binding_sha, "polygon": [list(p) for p in polygon], "source_region": summary})
    stage = {key: "b" * 64 for key in ("entry", "session", "lattice", "signatures", "connections", "graph", "attribution")}
    rows = [{"source_row": row, "row_sha256": str(i + 6) * 64,
             "original_direction": [list(diagnostic.MASK_ENDPOINTS[i]), list(diagnostic.MASK_ENDPOINTS[2])]}
            for i, row in enumerate(diagnostic.MASK_ROWS)]
    joint = {"schema": diagnostic.JOINT_ENDPOINT_SCHEMA, "producer": diagnostic.JOINT_ENDPOINT_PRODUCER,
             "first_witness_provenance": {"type": "available", "partial_exclusion_index": 0,
                "proof_sha256": "c" * 64, "source_universe_sha256": "d" * 64, "rows": rows, "readings": summaries},
             "proof_sha256": "c" * 64, "candidate_matches_sha256": "d" * 64,
             "selection_sha256": "e" * 64, "reading_bindings_sha256": "f" * 64,
             "source_bindings": {key + "_sha256": stage[key] for key in ("session", "lattice", "signatures", "graph")}}
    joint["record_sha256"] = diagnostic._digest(joint)
    record = {"schema": diagnostic.MASK_DIAGNOSTIC_SCHEMA, "ingress": ingress,
              "stage_digests": stage, "joint_endpoint_selection": joint,
              "status": "incomplete", "issues": ["synthetic-unrelated:missing"]}
    capture = SimpleNamespace(graph=SimpleNamespace(readings=tuple(readings), selection=SimpleNamespace(reading_bindings=tuple(bindings))))
    record["source_region_masks"] = diagnostic._source_region_mask_bundle(capture, record)
    record["record_sha256"] = diagnostic._digest(record)
    frame = subject.bind_frame(data, ingress[0], services=SERVICES)
    masks = subject.bind_masks(record, frame, services=SERVICES)
    return data, record, frame, masks


def features(frame, points=QUAD, dropped=0, descriptors=None):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.zeros((len(points), 128), np.float32)
        descriptors[:, 0] = np.arange(len(points), dtype=np.float32)
    return correspondence.FrameFeatures("image-0000", points, descriptors,
        frame.pixels.shape[1], frame.pixels.shape[0], int(frame.usable_mask.sum()),
        np.full(len(points), 2.125, np.float64), dropped)


def packet(points=QUAD, dropped=0, descriptors=None):
    _, _, frame, masks = fixture()
    return subject.inventory(frame, masks, features(frame, points, dropped, descriptors), {"synthetic": True}, services=SERVICES)


def test_absent_authorization_is_first_io_before_any_image_or_import(tmp_path, monkeypatch):
    seen = []
    original = subject.read
    def read(path, limit):
        seen.append(path)
        return original(path, limit)
    monkeypatch.setattr(subject, "read", read)
    monkeypatch.setattr(subject, "_services", lambda: pytest.fail("project services before auth"))
    with pytest.raises(FileNotFoundError): subject.authorize(tmp_path)
    assert seen == [tmp_path / subject.AUTH]
    assert not (tmp_path / subject.OUTPUT).exists()


def test_invalid_auth_and_cli_overrides_do_not_read_pixels(tmp_path, monkeypatch):
    path = tmp_path / subject.AUTH
    path.parent.mkdir(parents=True)
    path.write_bytes(subject.canonical({"disposition": "not authorized"}))
    with pytest.raises(subject.Refusal, match="authorization_required"): subject.authorize(tmp_path)
    monkeypatch.setattr(subject.sys, "argv", [subject.PROBE, "--image", "other.png"])
    with pytest.raises(subject.Refusal, match="fixed_input_no_overrides"): subject.main()


def test_valid_synthetic_authorization_binds_code_masks_baseline_without_opening_image(tmp_path, monkeypatch):
    data, record, _, _ = fixture(900,1600)
    source_data = subject.canonical(record) + b"\n"
    baseline_body = {"native_ingress":record["ingress"], "mask_bundle_sha256":record["source_region_masks"]["record_sha256"]}
    baseline_body["record_sha256"] = subject.sha(subject.canonical(baseline_body))
    baseline_data = subject.canonical(baseline_body) + b"\n"
    synthetic_pins = tuple((tuple(e["endpoint"]),e["mask"]["source_region_id"],e["mask"]["mask_sha256"],e["full_native_bool_c_sha256"])
                          for e in record["source_region_masks"]["entries"][:2])
    original_verify = subject.verify_masks
    monkeypatch.setattr(subject,"verify_masks",lambda value: original_verify(value,synthetic_pins))
    monkeypatch.setattr(subject,"SOURCE_PINS",{})
    monkeypatch.setattr(subject,"INPUT_SHA",subject.sha(data))
    monkeypatch.setattr(subject,"DIAGNOSTIC_SHA",subject.sha(source_data))
    monkeypatch.setattr(subject,"BASELINE_SHA",subject.sha(baseline_data))
    code_pins = {}
    for name in (subject.PROBE,subject.TEST):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic-frozen-code")
        code_pins[name]=subject.sha(path.read_bytes())
    for name,value in ((subject.DIAGNOSTIC,source_data),(subject.BASELINE,baseline_data)):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    auth={"disposition":"ACCEPT_PROBE; AUTHORIZE_ONE_DENSITY_RUN","single_probe":True,
          "output":subject.OUTPUT,"input_pins":{subject.INPUT:subject.INPUT_SHA},
          "diagnostic_sha256":subject.DIAGNOSTIC_SHA,"baseline_roster_sha256":subject.BASELINE_SHA,"code_pins":code_pins}
    path=tmp_path/subject.AUTH;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(subject.canonical(auth))
    assert subject.authorize(tmp_path)[0]==record
    assert not (tmp_path/subject.INPUT).exists() and not (tmp_path/subject.OUTPUT).exists()
    (tmp_path/subject.PROBE).write_bytes(b"changed-code")
    with pytest.raises(subject.Refusal,match="source_pin_mismatch"): subject.authorize(tmp_path)


def test_only_max_features_changes_from_accepted_baseline():
    before = asdict(roster.SETTINGS)
    config = subject.settings(roster)
    assert asdict(config) == {**before, "max_features": 2048}
    assert roster.SETTINGS.max_features == 512 and config.mask_boundary_pad_px == 0.0


@pytest.mark.parametrize("field,value", [("max_features", 512), ("mask_boundary_pad_px", 1.), ("ratio_test", .8), ("spatial_grid", (8, 8))])
def test_other_setting_changes_refuse(field, value):
    _, _, frame, masks = fixture()
    changed = replace(subject.settings(roster), **{field: value})
    with pytest.raises(subject.Refusal, match="detector_config_changed"):
        subject.inventory(frame, masks, features(frame), {}, services=SERVICES, config=changed)


def test_one_genuine_sift_detection_on_synthetic_native_zero_png_no_cross_view(monkeypatch):
    _, _, frame, masks = fixture()
    original = correspondence.detect_features
    calls = []
    def detect(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs): pytest.fail("cross-view/helper acquisition forbidden")
    monkeypatch.setattr(correspondence, "detect_features", detect)
    monkeypatch.setattr(correspondence, "extract_pair_matches", forbidden)
    monkeypatch.setattr(roster, "acquire_roster", forbidden)
    monkeypatch.setattr(roster, "_k_nearest", forbidden)
    result = subject.collect(frame, masks, {"synthetic": True}, services=SERVICES)
    subject.validate(result)
    assert len(calls) == 1
    assert calls[0][1]["usable_mask"] is frame.usable_mask
    assert calls[0][1]["settings"] == subject.settings(roster)
    assert result["counts"]["raw_returned"] == 0
    assert result["counts"]["membership_charged"] == 0
    assert result["counts"]["cross_view_comparisons"] == 0
    assert result["status"] == "insufficient_feature_supply"


def test_exact_mask_source_and_raw_vs_framed_digest_semantics():
    _, record, frame, masks = fixture()
    pins = tuple((tuple(e["endpoint"]), e["mask"]["source_region_id"], e["mask"]["mask_sha256"], e["full_native_bool_c_sha256"])
                 for e in record["source_region_masks"]["entries"][:2])
    subject.verify_masks(record, pins)
    for entry, mask in masks:
        assert mask.mask_sha256 == entry["full_native_bool_c_sha256"]
        assert mask.mask_sha256 != entry["mask"]["mask_sha256"]
        assert mask.pixels.flags.owndata and not mask.pixels.flags.writeable
    assert frame.source["decoded_sha256"] == roster.array_digest(frame.pixels)
    assert frame.source["decoded_sha256"] != record["ingress"][0]["decoded_pixel_sha256"]


@pytest.mark.parametrize("damage", ["missing", "extra", "order", "id", "packed", "raw", "row", "payload"])
def test_exact_mask_inventory_corruption_refuses(damage):
    _, record, frame, _ = fixture()
    pins = tuple((tuple(e["endpoint"]), e["mask"]["source_region_id"], e["mask"]["mask_sha256"], e["full_native_bool_c_sha256"])
                 for e in record["source_region_masks"]["entries"][:2])
    changed = deepcopy(record);entries = changed["source_region_masks"]["entries"]
    if damage == "missing": entries.pop()
    elif damage == "extra": entries.append(deepcopy(entries[0]))
    elif damage == "order": entries.reverse()
    elif damage == "id": entries[0]["mask"]["source_region_id"] = "0" * 64
    elif damage == "packed": entries[0]["mask"]["mask_sha256"] = "0" * 64
    elif damage == "raw": entries[0]["full_native_bool_c_sha256"] = "0" * 64
    elif damage == "row": entries[0]["row_refs"][0]["source_row"] += 1
    else: entries[0]["mask"]["packed_mask_base64"] = "!"
    if damage in ("row", "payload"):
        with pytest.raises(diagnostic.DiagnosticError): subject.bind_masks(changed, frame, services=SERVICES)
    else:
        with pytest.raises(subject.Refusal): subject.verify_masks(changed, pins)


@pytest.mark.parametrize("damage", ["encoded", "length", "identity", "dtype", "shape", "pixels"])
def test_native_frame_source_refuses(damage):
    data, record, _, _ = fixture();pin = deepcopy(record["ingress"][0])
    services = SERVICES
    if damage == "encoded": pin["byte_sha256"] = "0" * 64
    elif damage == "length": pin["byte_length"] += 1
    elif damage == "identity": pin["transform"]["crop_origin"] = [1, 0]
    else:
        def substitute(*args, **kwargs):
            if damage == "dtype": return np.zeros((64, 64, 3), np.float32)
            if damage == "shape": return np.zeros((63, 64, 3), np.uint8)
            result = np.zeros((64, 64, 3), np.uint8);result[0, 0, 0] = 1;return result
        services = (np, substitute, correspondence, image_sha256, roster, diagnostic)
    with pytest.raises((subject.Refusal, roster.Refusal)): subject.bind_frame(data, pin, services=services)


def test_fresh_inventory_identity_bins_provenance_and_counter_closure():
    result = packet()
    subject.validate(result)
    assert result["inventory"]["anchor"]["schema"] == subject.INVENTORY_SCHEMA
    assert subject.INVENTORY_SCHEMA != roster.SCHEMA
    assert result["counts"] == {"raw_returned":4,"retained_rows":4,"boundary_dropped":0,"distinct_classes":4,
        "exact_duplicate_rows":0,"distinct_coordinates":4,"membership_requested":8,"membership_charged":8,
        "membership_unprocessed":0,"cross_view_comparisons":0}
    for region in result["regions"]:
        assert region["eligible_distinct_classes"] == 4
        assert region["occupied_bins"] == 4
        assert [len(b) for b in region["bins"]] == [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert region["four_distinct_points_in_general_position"] is True
    assert result["status"] == "sufficient_feature_supply"
    assert "pair_id" not in subject.canonical(result).decode()


def test_exact_duplicates_and_coincident_classes_are_not_extra_point_support():
    descriptors = np.zeros((5, 128), np.float32)
    exact = packet([QUAD[0]] * 5, descriptors=descriptors)
    subject.validate(exact)
    assert exact["counts"]["distinct_classes"] == 1 and exact["counts"]["exact_duplicate_rows"] == 4
    assert exact["counts"]["membership_charged"] == 2
    coincident = packet([QUAD[0]] * 5)
    subject.validate(coincident)
    assert coincident["counts"]["distinct_classes"] == 5 and coincident["counts"]["distinct_coordinates"] == 1
    assert coincident["regions"][1]["four_distinct_points_in_general_position"] is False


def test_mask_hole_and_outside_coordinates_do_not_become_eligible():
    result = packet(QUAD + [(16.125, 19.125), (40.125, 12.125)])
    subject.validate(result)
    for region in result["regions"]:
        assert region["eligible_distinct_classes"] == 4 and region["outside_classes"] == 2
    assert result["counts"]["membership_charged"] == 12


def test_exact_2048_features_4096_memberships_and_no_truncation():
    points = [(8.125 + (i % 32) * .25, 12.375 + (i // 32) * .25) for i in range(2048)]
    result = packet(points)
    subject.validate(result)
    assert result["counts"]["raw_returned"] == result["counts"]["distinct_classes"] == 2048
    assert result["counts"]["membership_requested"] == result["counts"]["membership_charged"] == 4096
    assert len(subject.canonical(result)) < subject.LIMITS["evidence_bytes"]


@pytest.mark.parametrize("retained,dropped", [(2049,0),(2048,1),(1,2048)])
def test_raw_returned_overflow_refuses_before_support_work(monkeypatch, retained, dropped):
    _, _, frame, masks = fixture()
    monkeypatch.setattr(subject, "general_position", lambda _: pytest.fail("overflow entered support work"))
    with pytest.raises(subject.Refusal, match="raw_returned_feature_limit"):
        subject.inventory(frame, masks, features(frame, [QUAD[0]] * retained, dropped), {}, services=SERVICES)


def test_boundary_count_is_reported_and_not_replaced():
    result = packet(dropped=3)
    subject.validate(result)
    assert result["counts"]["raw_returned"] == 7
    assert result["counts"]["boundary_dropped"] == 3 and result["counts"]["retained_rows"] == 4


@pytest.mark.parametrize("damage", ["nan", "size", "outside", "descriptor_width", "misaligned", "foreign_frame"])
def test_invalid_feature_table_refuses(damage):
    _, _, frame, masks = fixture();table = features(frame)
    if damage == "nan": table.points[0,0] = np.nan
    elif damage == "size": table.sizes[0] = 0
    elif damage == "outside": table.points[0,0] = 64.
    elif damage == "descriptor_width": table = replace(table, descriptors=np.zeros((4,64), np.float32))
    elif damage == "misaligned": table = replace(table, sizes=table.sizes[:3])
    else: table = replace(table, frame_id="image-0001")
    with pytest.raises(subject.Refusal): subject.inventory(frame, masks, table, {}, services=SERVICES)


def brute_general_position(points):
    points = sorted(set(points))
    def area(a,b,c):
        ax,ay,bx,by,cx,cy = map(Fraction, (*a,*b,*c))
        return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    return any(all(area(*triple) for triple in itertools.combinations(quad,3))
               for quad in itertools.combinations(points,4))


def test_bounded_exact_general_position_matches_exhaustive_small_grid_oracle():
    grid = tuple(itertools.product(range(3),repeat=2))
    for count in (4,5,6):
        for points in itertools.combinations(grid,count):
            assert subject.general_position(points) == brute_general_position(points)
    assert subject.general_position([(0.,0.),(1.,0.),(0.,1.),(.5,2.**-50)])
    assert not subject.general_position([(float(i),0.) for i in range(2048)])
    assert not subject.general_position([(float(i),0.) for i in range(2047)] + [(0.,1.)])


@pytest.mark.parametrize("damage", ["record", "anchor", "config", "classes", "members", "descriptor", "counter", "bins", "status"])
def test_output_seal_provenance_and_counter_corruption_refuse(damage):
    result = packet()
    if damage == "record": result["record_sha256"] = "0" * 64
    elif damage == "anchor": result["inventory"]["anchor_sha256"] = "0" * 64
    elif damage == "config": result["inventory"]["anchor"]["runtime"]["settings"]["max_features"] = 512
    elif damage == "classes": result["inventory"]["classes"][0]["feature_id"] = "0" * 64
    elif damage == "members": result["inventory"]["classes"][0]["members"] = [0,0]
    elif damage == "descriptor": result["inventory"]["descriptor_table_sha256"] = "0" * 64
    elif damage == "counter": result["counts"]["cross_view_comparisons"] = 1
    elif damage == "bins": result["regions"][0]["occupied_bins"] += 1
    else: result["status"] = "insufficient_feature_supply"
    if damage != "record":
        result.pop("record_sha256");result=subject.finalize(result)
    with pytest.raises(subject.Refusal): subject.validate(result)


def test_evidence_bound_before_publish_and_no_clobber_roundtrip(tmp_path):
    overhead=len(subject.canonical(subject.finalize({"pad":""})))+1
    exact=subject.finalize({"pad":"x"*(subject.LIMITS["evidence_bytes"]-overhead)})
    assert len(subject.canonical(exact))+1==subject.LIMITS["evidence_bytes"]
    with pytest.raises(subject.Refusal,match="evidence_byte_limit"):
        subject.finalize({"pad":"x"*(subject.LIMITS["evidence_bytes"]-overhead+1)})
    with pytest.raises(subject.Refusal, match="evidence_byte_limit"):
        subject.finalize({"large":"x"*subject.LIMITS["evidence_bytes"]})
    assert list(tmp_path.iterdir()) == []
    result = subject.parse(subject.canonical(packet()));path=tmp_path/"density.json"
    infrastructure.publish(path,result,subject.validate)
    assert subject.sealed_source(path.read_bytes()) == result
    with pytest.raises(FileExistsError): infrastructure.publish(path,result,subject.validate)
    assert subject.sealed_source(path.read_bytes()) == result


def test_preserved_budget_infrastructure_and_no_density_runtime_invocation():
    assert infrastructure.WALL_SECONDS == subject.LIMITS["wall_seconds"] == 180
    assert infrastructure.RSS_BYTES == subject.LIMITS["job_committed_bytes"] == 2147483648
    assert infrastructure.ARTIFACT_BYTES == subject.LIMITS["artifact_bytes"] == 8388608
    # No Windows supervisor or runtime worker is invoked by this test module.
