"""Generated pixels only; no capture, reference, or real screenshot access."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/evaluation/diagnose_multiview_plane_tracks.py"
spec = importlib.util.spec_from_file_location("diagnose_multiview_plane_tracks", SOURCE)
assert spec and spec.loader
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

PLANAR_TRANSFORMS = (
    np.eye(3),
    np.array([[1.02, .035, 11], [-.025, 1.01, 9], [.00010, -.00008, 1]]),
    np.array([[.99, -.025, -8], [.035, 1.02, 13], [-.00007, .00012, 1]]),
)


def texture(seed: int = 17, size: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 30, np.uint8)
    for _ in range(320):
        x, y = rng.integers(12, size - 12, size=2)
        color = tuple(int(c) for c in rng.integers(70, 250, size=3))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 6)), color, -1)
    return image


def planar_frames(*, noise: bool = True, occlusion: bool = False) -> list[tuple[str, np.ndarray]]:
    base = texture()
    rng = np.random.default_rng(1234)
    result = []
    for i, h in enumerate(PLANAR_TRANSFORMS):
        image = cv2.warpPerspective(base, h, (384, 384), borderValue=(30, 30, 30))
        if noise:
            image = np.clip(image.astype(np.int16) + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
            for _ in range(12):
                x, y = rng.integers(0, 384, size=2)
                cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
        if occlusion and i == 1:
            image[145:215, 140:210] = 20
        result.append((f"view-{i}", image))
    return result


def test_planar_noise_outliers_recovers_projective_support_and_determinism():
    frames = planar_frames()
    first = module.diagnose(frames)
    second = module.diagnose([(name, image.copy()) for name, image in frames])
    assert first == second
    assert first["status"] == "supported"
    assert first["tracks"]
    track = first["tracks"][0]
    assert track["triangle_count"] >= module.MIN_CYCLE_TRIPLES
    assert track["held_out_triangle_count"] >= 2
    assert track["cycle_all"]["p95_px"] <= module.TRANSFER_PX
    assert track["cycle_held_out"]["p95_px"] <= module.TRANSFER_PX
    assert track["cycle_held_out"]["count"] == track["held_out_triangle_count"]
    assert len(track["triangles"]) == track["triangle_count"]
    assert sum(track["triangle_is_held_out"]) == track["held_out_triangle_count"]
    assert all(row["count"] == len(row["held_out_rows"]) and
               row["joint_accepted_count"] + row["joint_rejected_count"] == row["count"]
               for row in track["pair_held_out"].values())
    assert first["unresolved"] == list(module.UNRESOLVED)
    payload = dict(first)
    digest = payload.pop("canonical_sha256")
    assert hashlib.sha256(module._canonical(payload)).hexdigest() == digest
    assert module._canonical(json.loads(module.canonical_report_bytes(first))) == module.canonical_report_bytes(first)
    assert first["frame_sha256"]["view-0"] == hashlib.sha256(
        (384).to_bytes(4, "big") * 2 + frames[0][1].tobytes()).hexdigest()
    module.verify_frame_binding(first, frames)
    frames[0][1][:] = 0
    assert module.canonical_report_bytes(first)
    with pytest.raises(ValueError, match="drift"):
        module.verify_frame_binding(first, frames)


def test_partial_occlusion_support_only_on_observed_features():
    report = module.diagnose(planar_frames(occlusion=True))
    assert report["status"] == "supported"
    for track in report["tracks"]:
        assert all(not (140 <= x < 210 and 145 <= y < 215)
                   for x, y in track["support"]["view-1"])


def test_planar_precision_recall_are_synthetic_feature_metrics():
    frames = planar_frames()
    report = module.diagnose(frames)
    extracted = [module._features(image) for _, image in frames]
    xy = [x[0] for x in extracted]
    pair = {(a, b): dict(module._matches(extracted[a][1], extracted[b][1]))
            for a, b in ((0, 1), (0, 2), (1, 2))}
    eligible = []
    for ia, ib in pair[0, 1].items():
        ic = pair[0, 2].get(ia)
        if ic is None or pair[1, 2].get(ib) != ic:
            continue
        predicted_b = module._transfer(PLANAR_TRANSFORMS[1], xy[0][ia:ia + 1])[0]
        predicted_c = module._transfer(PLANAR_TRANSFORMS[2], xy[0][ia:ia + 1])[0]
        if np.linalg.norm(predicted_b - xy[1][ib]) <= 3 and np.linalg.norm(predicted_c - xy[2][ic]) <= 3:
            eligible.append((ia, ib, ic))
    emitted = [tuple(track["support"][f"view-{i}"][k] for i in range(3))
               for track in report["tracks"] for k in range(track["triangle_count"])]
    true_emitted = 0
    for a, b, c in emitted:
        predicted_b = module._transfer(PLANAR_TRANSFORMS[1], np.array([a]))[0]
        predicted_c = module._transfer(PLANAR_TRANSFORMS[2], np.array([a]))[0]
        true_emitted += np.linalg.norm(predicted_b - b) <= 3 and np.linalg.norm(predicted_c - c) <= 3
    precision = true_emitted / len(emitted)
    recall = true_emitted / len(eligible)
    print(f"synthetic_feature_precision={precision:.6f} recall={recall:.6f} "
          f"true={true_emitted} emitted={len(emitted)} eligible={len(eligible)}")
    assert precision >= .95 and recall >= .5


def test_invalid_inputs_and_order_fail_closed():
    frames = planar_frames(noise=False)
    with pytest.raises(ValueError):
        module.diagnose(frames[:2])
    with pytest.raises(ValueError):
        module.diagnose(frames[::-1])
    with pytest.raises(ValueError):
        module.diagnose([frames[0], frames[0], frames[2]])
    for bad in (np.zeros((10, 10, 3), np.uint8),
                np.full((384, 384, 3), np.nan),
                np.zeros((1100, 1100, 3), np.uint8),
                np.zeros((384, 384, 4), np.uint8)):
        with pytest.raises(ValueError):
            module.diagnose([frames[0], frames[1], ("view-2", bad)])
    with pytest.raises(ValueError):
        module._normalized(np.zeros((3, 3)))


def test_no_descriptors_abstains():
    blank = np.zeros((384, 384, 3), np.uint8)
    report = module.diagnose([(f"view-{i}", blank) for i in range(3)])
    assert report["status"] == "abstained"
    assert report["abstention"] == "no_coherent_three_view_plane"


def test_disconnected_coplanar_looking_islands_abstain():
    base = np.full((384, 384, 3), 30, np.uint8)
    patterned = texture()
    base[105:270, 15:125] = patterned[105:270, 15:125]
    base[105:270, 260:370] = patterned[105:270, 260:370]
    frames = [(f"view-{i}", cv2.warpPerspective(base,
               np.array([[1, 0, i * 5], [0, 1, i * 3], [0, 0, 1]], float),
               (384, 384), borderValue=(30, 30, 30))) for i in range(3)]
    report = module.diagnose(frames)
    assert report["status"] == "abstained"


@pytest.mark.parametrize("repeat", [False, True])
def test_perpendicular_corner_keeps_face_support_separate_or_abstains(repeat):
    base = texture()
    source_left = base.copy()
    source_left[:, 180:] = 30
    source_right = np.full_like(base, 30)
    source_right[:, 204:] = (base if repeat else texture(seed=89))[:, :180]
    frames = []
    for i in range(3):
        left_h = np.array([[1, .015 * i, 2 * i], [0, 1, 2 * i], [0, 0, 1]], float)
        right_h = np.array([[1, -.03 * i, -3 * i], [.025 * i, 1, i], [0, 0, 1]], float)
        left = cv2.warpPerspective(source_left, left_h, (384, 384), borderValue=(30, 30, 30))
        right = cv2.warpPerspective(source_right, right_h, (384, 384), borderValue=(30, 30, 30))
        left_mask = cv2.warpPerspective((np.any(source_left != 30, axis=2).astype(np.uint8)),
                                        left_h, (384, 384))
        right_mask = cv2.warpPerspective((np.any(source_right != 30, axis=2).astype(np.uint8)),
                                         right_h, (384, 384))
        image = np.full_like(base, 30)
        image[left_mask > 0] = left[left_mask > 0]
        image[right_mask > 0] = right[right_mask > 0]
        frames.append((f"view-{i}", image))
    report = module.diagnose(frames)
    print(f"corner_repeat={repeat} status={report['status']} tracks={len(report['tracks'])}")
    for track in report["tracks"]:
        xs = [x for x, _ in track["support"]["view-0"]]
        assert max(xs) < 192 or min(xs) >= 192


def test_contiguous_same_texture_perpendicular_corner_has_no_merged_track():
    base = texture(seed=41)
    left_source = np.full_like(base, 30)
    right_source = np.full_like(base, 30)
    left_source[:, :192] = base[:, :192]
    right_source[:, 192:] = base[:, :192]  # identical repeated texture at the seam
    left_mask = np.zeros(base.shape[:2], np.uint8)
    right_mask = np.zeros_like(left_mask)
    left_mask[:, :192] = 1
    right_mask[:, 192:] = 1
    assert np.all(left_mask[:, 191] == 1) and np.all(right_mask[:, 192] == 1)
    frames = []
    for i in range(3):
        lh = np.array([[1, .012 * i, 2 * i], [0, 1, i], [0, 0, 1]], float)
        rh = np.array([[1, -.023 * i, -2 * i], [.018 * i, 1, i], [0, 0, 1]], float)
        left = cv2.warpPerspective(left_source, lh, (384, 384), borderValue=(30, 30, 30))
        right = cv2.warpPerspective(right_source, rh, (384, 384), borderValue=(30, 30, 30))
        lm = cv2.warpPerspective(left_mask, lh, (384, 384), flags=cv2.INTER_NEAREST)
        rm = cv2.warpPerspective(right_mask, rh, (384, 384), flags=cv2.INTER_NEAREST)
        image = np.full_like(base, 30)
        image[lm > 0] = left[lm > 0]
        image[rm > 0] = right[rm > 0]
        frames.append((f"view-{i}", image))
    report = module.diagnose(frames)
    for track in report["tracks"]:
        xs = [x for x, _ in track["support"]["view-0"]]
        assert max(xs) < 192 or min(xs) >= 192


def test_inconsistent_repeated_alias_cycle_rejected():
    ids = ("a", "b", "c")
    grid = np.array([(x, y) for x in (40, 90, 140, 190, 240, 290, 340)
                     for y in (40, 90, 140, 190, 240, 290, 340)], float)
    keys = tuple((i, i) for i in range(len(grid)))
    identity = tuple(tuple(float(x) for x in row) for row in np.eye(3))
    shifted = tuple(tuple(float(x) for x in row) for row in
                    np.array([[1, 0, 18], [0, 1, 0], [0, 0, 1]]))
    def model(pair, transform):
        return module.PairModel(pair, transform, tuple(range(35)), tuple(range(35, 49)),
                                keys[:35], keys,
                                tuple((i, i, *grid[i], *grid[i], 0., 0.)
                                      for i in range(35, 49)))
    track = module._track(*ids, model(("a", "b"), identity),
                          model(("b", "c"), identity), model(("a", "c"), shifted),
                          dict.fromkeys(ids, grid), dict.fromkeys(ids, (384, 384)),
                          dict.fromkeys(ids, "x" * 64))
    assert track is None


def test_collinear_and_low_coverage_are_rejected():
    line = np.array([(x, 150.) for x in range(25, 360, 10)])
    tiny = np.array([(100 + x, 100 + y) for x in range(8) for y in range(8)], float)
    assert not module._coverage(line, (384, 384))
    assert not module._coverage(tiny, (384, 384))
    assert module._model(("a", "b"), tuple((i, i) for i in range(len(line))),
                         line, line, (384, 384), (384, 384),
                         tuple(range(len(line))), 1) is None


def test_holdout_is_fixed_before_fit_and_reverse_transfer_matters(monkeypatch):
    pair = ("a", "b")
    grid = np.array([(x, y) for x in (35, 75, 115, 155, 195, 235, 275, 315, 355)
                     for y in (35, 75, 115, 155, 195, 235, 275, 315, 355)], float)
    keys = tuple((i, i) for i in range(len(grid)))
    fit = [i for i in range(len(keys)) if module._seed("a", i) % 5 != 0]
    seen = []
    original = module.cv2.findHomography
    def spy(src, dst, *args, **kwargs):
        seen.append((src.copy(), dst.copy()))
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(module.cv2, "findHomography", spy)
    result = module._model(pair, keys, grid, grid.copy(), (384, 384), (384, 384),
                           tuple(range(len(keys))), 1)
    assert result is not None
    assert len(seen) == 1 and np.array_equal(seen[0][0], grid[fit])
    corrupted = grid.copy()
    hold = [i for i in range(len(keys)) if i not in fit]
    corrupted[hold] += np.array([28, -17])
    refused = module._model(pair, keys, grid, corrupted, (384, 384), (384, 384),
                            tuple(range(len(keys))), 1)
    assert refused is None
    assert np.array_equal(seen[0][0], seen[1][0])


def test_reverse_holdout_rejects_forward_only_pass():
    grid = np.array([(x, y) for x in (35, 75, 115, 155, 195, 235, 275, 315, 355)
                     for y in (35, 75, 115, 155, 195, 235, 275, 315, 355)], float)
    target = grid * .5 + 80
    keys = tuple((i, i) for i in range(len(grid)))
    hold = [i for i in range(len(keys)) if module._seed("a", i) % 5 == 0]
    target[hold, 0] += 2  # forward 2 px; inverse transfer 4 px
    result = module._model(("a", "b"), keys, grid, target, (384, 384), (384, 384),
                           tuple(range(len(keys))), 1)
    assert result is None


def test_holdout_distribution_includes_rejected_residuals():
    grid = np.array([(x, y) for x in (35, 75, 115, 155, 195, 235, 275, 315, 355)
                     for y in (35, 75, 115, 155, 195, 235, 275, 315, 355)], float)
    target = grid.copy()
    hold = [i for i in range(len(grid)) if module._seed("a", i) % 5 == 0]
    target[hold[:2]] += np.array([20., 0.])
    model = module._model(("a", "b"), tuple((i, i) for i in range(len(grid))),
                          grid, target, (384, 384), (384, 384), tuple(range(len(grid))), 1)
    assert model is not None
    row = module._pair_holdout_report(model)
    assert row["count"] == len(hold)
    assert row["joint_accepted_count"] == len(hold) - 2
    assert row["forward_rejected_count"] == row["reverse_rejected_count"] == 2
    assert row["joint_rejected_count"] == 2
    assert row["forward"]["max_px"] > module.TRANSFER_PX
    assert row["reverse"]["max_px"] > module.TRANSFER_PX
    assert any(x[6] > module.TRANSFER_PX for x in row["held_out_rows"])


def test_nonfinite_rejected_holdout_is_explicitly_reported():
    values = np.array([0.] * 8 + [np.inf])
    summary = module._residual_summary(values)
    assert summary == {"count": 9, "p50_px": 0., "p95_px": "inf", "max_px": "inf"}


def test_persistence_collision_and_digest_refusal(tmp_path):
    report = module.diagnose([(f"view-{i}", np.zeros((64, 64, 3), np.uint8)) for i in range(3)])
    target = tmp_path / "sealed.json"
    module.persist_report(target, report)
    original = target.read_bytes()
    assert original == module.canonical_report_bytes(report)
    with pytest.raises(FileExistsError):
        module.persist_report(target, report)
    assert target.read_bytes() == original
    changed = dict(report, status="supported")
    with pytest.raises(ValueError, match="digest mismatch"):
        module.persist_report(tmp_path / "bad.json", changed)
    assert not (tmp_path / "bad.json").exists()
    assert not list(tmp_path.glob(".plane-tracks-*"))
    original_link = module.os.link
    def fail_link(*args):
        raise OSError("injected publication failure")
    module.os.link = fail_link
    try:
        with pytest.raises(OSError, match="injected"):
            module.persist_report(tmp_path / "new.json", report)
    finally:
        module.os.link = original_link
    assert not (tmp_path / "new.json").exists()
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".plane-tracks-*"))


def test_resealed_malformed_transform_refuses_before_write(tmp_path):
    report = json.loads(module.canonical_report_bytes(module.diagnose(planar_frames())))
    key = next(iter(report["tracks"][0]["transforms"]))
    report["tracks"][0]["transforms"][key] = [[0, 0, 0]] * 3
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = hashlib.sha256(module._canonical(payload)).hexdigest()
    with pytest.raises(ValueError, match="transform"):
        module.persist_report(tmp_path / "invalid.json", report)
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def supported_report():
    return module.diagnose(planar_frames())


@pytest.mark.parametrize("mutation", [
    "omit_triangle", "duplicate_triangle", "reorder_triangles", "triangle_count",
    "held_partition", "held_count", "cycle_summary", "held_cycle_summary",
    "omit_pair_holdout", "duplicate_pair_holdout", "reorder_pair_holdout",
    "pair_count", "pair_summary", "pair_rejection_count", "descriptor_work",
    "pair_residual_replay",
])
def test_resealed_identity_and_summary_mutations_refuse_before_write(
        supported_report, mutation, tmp_path):
    report = json.loads(module.canonical_report_bytes(supported_report))
    track = report["tracks"][0]
    pair = next(iter(track["pair_held_out"].values()))
    if mutation == "omit_triangle":
        track["triangles"].pop()
    elif mutation == "duplicate_triangle":
        track["triangles"][1] = track["triangles"][0]
    elif mutation == "reorder_triangles":
        track["triangles"][0], track["triangles"][1] = track["triangles"][1], track["triangles"][0]
    elif mutation == "triangle_count":
        track["triangle_count"] += 1
    elif mutation == "held_partition":
        track["triangle_is_held_out"][0] = not track["triangle_is_held_out"][0]
    elif mutation == "held_count":
        track["held_out_triangle_count"] += 1
    elif mutation == "cycle_summary":
        track["cycle_all"]["p95_px"] += 0.1
    elif mutation == "held_cycle_summary":
        track["cycle_held_out"]["max_px"] += 0.1
    elif mutation == "omit_pair_holdout":
        pair["held_out_rows"].pop()
    elif mutation == "duplicate_pair_holdout":
        pair["held_out_rows"][1] = pair["held_out_rows"][0]
    elif mutation == "reorder_pair_holdout":
        pair["held_out_rows"][0], pair["held_out_rows"][1] = (
            pair["held_out_rows"][1], pair["held_out_rows"][0])
    elif mutation == "pair_count":
        pair["joint_rejected_count"] += 1
    elif mutation == "pair_summary":
        pair["forward"]["max_px"] += 0.1
    elif mutation == "pair_rejection_count":
        pair["forward_rejected_count"] += 1
    elif mutation == "pair_residual_replay":
        pair["held_out_rows"][0][6] += 0.1
    elif mutation == "descriptor_work":
        report["work"]["descriptor_comparisons"] -= 1
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = hashlib.sha256(module._canonical(payload)).hexdigest()
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "invalid.json", report)
    assert not list(tmp_path.iterdir())


def reseal(report):
    payload = dict(report)
    payload.pop("canonical_sha256", None)
    report["canonical_sha256"] = hashlib.sha256(module._canonical(payload)).hexdigest()
    return report


def refresh_pair_counts(row):
    ef = np.array([float(item[6]) for item in row["held_out_rows"]])
    er = np.array([float(item[7]) for item in row["held_out_rows"]])
    both = (ef <= module.TRANSFER_PX) & (er <= module.TRANSFER_PX)
    row.update(count=len(ef), joint_accepted_count=int(sum(both)),
               forward_rejected_count=int(sum(ef > module.TRANSFER_PX)),
               reverse_rejected_count=int(sum(er > module.TRANSFER_PX)),
               joint_rejected_count=int(sum(~both)),
               forward=module._residual_summary(ef), reverse=module._residual_summary(er))


def refresh_triangle_partition(track):
    a, b, c = track["frame_ids"]
    keys = [module._pair_key(*pair) for pair in ((a, b), (b, c), (a, c))]
    fit = [set(map(tuple, track["pair_held_out"][key]["fit_keys"])) for key in keys]
    track["triangle_is_held_out"] = [
        (i, j) not in fit[0] or (j, k) not in fit[1] or (i, k) not in fit[2]
        for i, j, k in track["triangles"]]
    flags = track["triangle_is_held_out"]
    track["held_out_triangle_count"] = sum(flags)
    xyz = np.asarray(track["support"][a])
    hab, hbc, hac = [np.asarray(track["transforms"][key]) for key in keys]
    closure = np.linalg.norm(module._transfer(hac, xyz) -
                             module._transfer(hbc, module._transfer(hab, xyz)), axis=1)
    track["cycle_all"] = module._residual_summary(closure)
    track["cycle_held_out"] = module._residual_summary(closure[flags])


@pytest.mark.parametrize("mutation", [
    "held_to_fit", "fit_to_held", "omit_valid_triangle", "support_coordinate",
    "held_coordinate", "support_outside_dimensions", "duplicate_feature_coordinate",
])
def test_consistently_resealed_replay_mutations_refuse(supported_report, mutation, tmp_path):
    report = json.loads(module.canonical_report_bytes(supported_report))
    track = report["tracks"][0]
    a, b, c = track["frame_ids"]
    ab = module._pair_key(a, b)
    row = track["pair_held_out"][ab]
    if mutation == "held_to_fit":
        held = next(item for item in row["held_out_rows"]
                    if item[6] <= module.TRANSFER_PX and item[7] <= module.TRANSFER_PX)
        row["held_out_rows"].remove(held)
        row["fit_keys"] = sorted(row["fit_keys"] + [held[:2]])
        refresh_pair_counts(row)
        refresh_triangle_partition(track)
    elif mutation == "fit_to_held":
        i, j = row["fit_keys"].pop()
        src = dict((point[0], point[1:]) for point in track["feature_points"][a])[i]
        dst = dict((point[0], point[1:]) for point in track["feature_points"][b])[j]
        h = np.asarray(track["transforms"][ab])
        ef = float(np.linalg.norm(module._transfer(h, np.asarray([src]))[0] - dst))
        er = float(np.linalg.norm(module._transfer(np.linalg.inv(h), np.asarray([dst]))[0] - src))
        row["held_out_rows"].append([i, j, *src, *dst, ef, er])
        row["held_out_rows"].sort(key=lambda item: item[:2])
        refresh_pair_counts(row)
        refresh_triangle_partition(track)
    elif mutation == "omit_valid_triangle":
        for field in ("triangles", "triangle_is_held_out"):
            track[field].pop()
        for frame in track["frame_ids"]:
            track["support"][frame].pop()
        track["triangle_count"] -= 1
        refresh_triangle_partition(track)
    elif mutation == "support_coordinate":
        track["support"][b][0][0] += 0.25
    elif mutation == "held_coordinate":
        supported_edges = {(i, j) for i, j, _ in track["triangles"]}
        item = next(item for item in row["held_out_rows"]
                    if tuple(item[:2]) in supported_edges)
        item[2] += 0.25
        h = np.asarray(track["transforms"][ab])
        src = np.asarray([item[2:4]])
        dst = np.asarray([item[4:6]])
        item[6] = float(np.linalg.norm(module._transfer(h, src)[0] - dst[0]))
        item[7] = float(np.linalg.norm(module._transfer(np.linalg.inv(h), dst)[0] - src[0]))
        refresh_pair_counts(row)
    elif mutation == "support_outside_dimensions":
        track["support"][b][0][0] = report["frame_dimensions"][b][1] + 1.
    elif mutation == "duplicate_feature_coordinate":
        # The same feature is used in two triangle rows; one row diverges.
        indices = [triple[1] for triple in track["triangles"]]
        repeated = next((index for index in indices if indices.count(index) > 1), None)
        if repeated is None:
            # The registry is still an independent declaration of that identity.
            index = indices[0]
            next(point for point in track["feature_points"][b]
                 if point[0] == index)[1] += 0.25
        else:
            n = indices.index(repeated)
            track["support"][b][n][0] += 0.25
    reseal(report)
    with pytest.raises(ValueError):
        module.canonical_report_bytes(report)
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "invalid.json", report)
    assert not list(tmp_path.iterdir())


def test_opaque_arrow_frame_ids_roundtrip_and_dimensions(tmp_path):
    left = module._pair_key("a->b", "c")
    right = module._pair_key("a", "b->c")
    assert left != right
    assert json.loads(left) == ["a->b", "c"]
    assert json.loads(right) == ["a", "b->c"]
    ids = ("a->:[]Î»", "b->:[é›ª]", "c->:{}Ã©")
    frames = [(ids[i], image) for i, (_, image) in enumerate(planar_frames())]
    report = module.diagnose(frames)
    assert report["status"] == "supported"
    assert report["schema"] == "projective-plane-tracks-v4"
    assert report["frame_dimensions"] == {name: [384, 384] for name in ids}
    for track in report["tracks"]:
        assert {tuple(json.loads(key)) for key in track["pair_held_out"]} == {
            (ids[0], ids[1]), (ids[1], ids[2]), (ids[0], ids[2])}
    module.verify_frame_binding(report, frames)
    target = tmp_path / "arrows.json"
    module.persist_report(target, report)
    assert target.read_bytes() == module.canonical_report_bytes(report)
    changed = json.loads(target.read_bytes())
    changed["frame_dimensions"][ids[0]][0] += 1
    changed["work"]["input_rgb_bytes"] += 384 * 3
    reseal(changed)
    assert module.canonical_report_bytes(changed)
    with pytest.raises(ValueError, match="drift"):
        module.verify_frame_binding(changed, frames)


def test_partial_write_failure_leaves_no_target_or_temp(monkeypatch, tmp_path):
    report = module.diagnose([(f"view-{i}", np.zeros((64, 64, 3), np.uint8)) for i in range(3)])
    real_fdopen = module.os.fdopen
    class ShortWriter:
        def __init__(self, fd, mode):
            self.stream = real_fdopen(fd, mode)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, data):
            return self.stream.write(data[:-1])
        def flush(self):
            self.stream.flush()
        def fileno(self):
            return self.stream.fileno()
    monkeypatch.setattr(module.os, "fdopen", ShortWriter)
    with pytest.raises(OSError, match="partial"):
        module.persist_report(tmp_path / "short.json", report)
    assert not list(tmp_path.iterdir())


def test_output_budget_refuses_before_publication(monkeypatch, tmp_path):
    frames = [(f"view-{i}", np.zeros((64, 64, 3), np.uint8)) for i in range(3)]
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 80)
    with pytest.raises(ValueError, match="output budget"):
        module.diagnose(frames)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1_000_000)
    monkeypatch.setattr(module, "MAX_PAIRS", 2)
    with pytest.raises(ValueError, match="pair budget"):
        module.diagnose(frames)


def test_track_identity_rows_are_charged_to_output_budget(monkeypatch, supported_report):
    full_size = len(module.canonical_report_bytes(supported_report))
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", full_size - 1)
    with pytest.raises(ValueError, match="output budget"):
        module.diagnose(planar_frames())


def test_descriptor_and_cycle_work_caps_refuse(monkeypatch):
    frames = planar_frames()
    monkeypatch.setattr(module, "MAX_DESCRIPTOR_COMPARISONS", 1)
    with pytest.raises(ValueError, match="descriptor work budget"):
        module.diagnose(frames)
    monkeypatch.setattr(module, "MAX_DESCRIPTOR_COMPARISONS", 12_000_000)
    monkeypatch.setattr(module, "MAX_CYCLE_COMBINATIONS", 0)
    with pytest.raises(ValueError, match="cycle work budget"):
        module.diagnose(frames)


def test_bidirectional_descriptor_work_is_charged_before_matching(monkeypatch):
    frames = [(f"view-{i}", np.zeros((64, 64, 3), np.uint8)) for i in range(3)]
    monkeypatch.setattr(module, "_features", lambda image: (
        np.zeros((100, 2), np.float64), np.zeros((100, 128), np.float32)))
    def forbidden_match(*args):
        raise AssertionError("matcher ran over budget")
    monkeypatch.setattr(module, "_matches", forbidden_match)
    monkeypatch.setattr(module, "MAX_DESCRIPTOR_COMPARISONS", 59_999)
    with pytest.raises(ValueError, match="descriptor work budget"):
        module.diagnose(frames)
    monkeypatch.setattr(module, "_matches", lambda *args: ())
    monkeypatch.setattr(module, "MAX_DESCRIPTOR_COMPARISONS", 60_000)
    assert module.diagnose(frames)["work"]["descriptor_comparisons"] == 60_000


def attempt_fixture(fit_count=24, held_count=10, left="a"):
    fit_ids = [i for i in range(700) if module._seed(left, i) % 5 != 0][:fit_count]
    held_ids = [i for i in range(700) if module._seed(left, i) % 5 == 0][:held_count]
    assert len(fit_ids) == fit_count and len(held_ids) == held_count
    ids = sorted(fit_ids + held_ids)
    points = np.zeros((max(ids) + 1, 2), np.float64)
    for rank, index in enumerate(ids):
        points[index] = (35 + rank % 10 * 30, 35 + rank // 10 * 30)
    matches = tuple((i, i) for i in ids)
    hold_indices = [n for n, (i, _) in enumerate(matches) if module._seed(left, i) % 5 == 0]
    return matches, points, hold_indices


def test_pair_ratio_and_reciprocal_telemetry_respects_matcher_calls(monkeypatch):
    trace = {}
    def forbidden(*args):
        raise AssertionError("matcher reached")
    monkeypatch.setattr(module.cv2, "BFMatcher", forbidden)
    for n in (0, 1):
        trace = {}
        assert module._matches(np.zeros((n, 128), np.float32),
                               np.zeros((2, 128), np.float32), trace) == ()
        assert trace["matching_status"] == "insufficient_descriptors"
    assert trace["forward_ratio_pass_count"] == trace["reverse_ratio_pass_count"] == 0
    assert trace["reciprocal_pairs"] == []

    def pair(q, t, near, far):
        return [cv2.DMatch(q, t, near), cv2.DMatch(q, 1 - t, far)]
    class Matcher:
        def __init__(self, forward, reverse):
            self.rows = [forward, reverse]
        def knnMatch(self, *_args, **_kwargs):
            return self.rows.pop(0)
    monkeypatch.setattr(module.cv2, "BFMatcher", lambda *_: Matcher(
        [pair(0, 0, 0.8, 1.), pair(1, 1, 0.8, 1.)],
        [pair(0, 0, 0.8, 1.), pair(1, 1, 0.8, 1.)]))
    trace = {}
    assert module._matches(np.zeros((2, 128), np.float32),
                           np.zeros((2, 128), np.float32), trace) == ()
    assert trace["forward_ratio_pass_count"] == trace["reverse_ratio_pass_count"] == 0
    monkeypatch.setattr(module.cv2, "BFMatcher", lambda *_: Matcher(
        [pair(0, 0, 0.1, 1.), pair(1, 1, 0.1, 1.)],
        [pair(0, 1, 0.1, 1.), pair(1, 0, 0.1, 1.)]))
    trace = {}
    assert module._matches(np.zeros((2, 128), np.float32),
                           np.zeros((2, 128), np.float32), trace) == ()
    assert trace["forward_ratio_pass_count"] == trace["reverse_ratio_pass_count"] == 2
    assert trace["reciprocal_count"] == 0


@pytest.mark.parametrize("fit_count,held_count,reason", [
    (15, 7, "fit_starved"), (16, 7, "held_starved"),
    (16, 8, "accepted"),
])
def test_partition_first_exit_and_boundary(monkeypatch, fit_count, held_count, reason):
    matches, points, _ = attempt_fixture(fit_count, held_count)
    calls = []
    def homography(src, dst, method, threshold, *, maxIters, confidence):
        calls.append((src.copy(), dst.copy(), method, threshold, maxIters, confidence))
        return np.eye(3), np.ones((len(src), 1), np.uint8)
    monkeypatch.setattr(module.cv2, "findHomography", homography)
    monkeypatch.setattr(module, "_coverage", lambda _p, _s, _t=None: True)
    trace = {}
    result = module._model(("a", "b"), matches, points, points.copy(), (384, 384), (384, 384),
                           tuple(range(len(matches))), 19, trace)
    assert trace["first_rejection"] == reason
    if reason == "accepted":
        assert result is not None and len(calls) == 1
        assert calls[0][2:] == (cv2.RANSAC, module.TRANSFER_PX, 500, .995)
        assert np.array_equal(calls[0][0], points[[matches[i][0] for i in trace["fit_indices"]]])
    else:
        assert result is None and not calls
        assert trace["homography_available"] is None


@pytest.mark.parametrize("mode,reason", [
    ("missing_h", "homography_unavailable"), ("missing_mask", "mask_unavailable"),
    ("bad_forward", "forward_normalization"),
    ("bad_inversion", "inversion_unavailable"), ("bad_inverse", "inverse_normalization"),
    ("few_inliers", "fit_inlier_starved"),
])
def test_homography_normalization_and_inlier_first_exits(monkeypatch, mode, reason):
    matches, points, _ = attempt_fixture()
    original_normalized = module._normalized
    normalized_calls = []
    def normalized(h, telemetry=None):
        normalized_calls.append(h.copy())
        if mode == "bad_inverse" and len(normalized_calls) == 2:
            raise ValueError("injected inverse failure")
        return original_normalized(h, telemetry)
    monkeypatch.setattr(module, "_normalized", normalized)
    if mode == "bad_inversion":
        monkeypatch.setattr(module.np.linalg, "inv", lambda *_: (_ for _ in ()).throw(
            np.linalg.LinAlgError("injected inversion failure")))
    def homography(src, _dst, *_args, **_kwargs):
        mask = np.ones((len(src), 1), np.uint8)
        if mode == "few_inliers":
            mask[15:] = 0
        return (None if mode == "missing_h" else
                np.zeros((3, 3)) if mode == "bad_forward" else np.eye(3)), (
                None if mode == "missing_mask" else mask)
    monkeypatch.setattr(module.cv2, "findHomography", homography)
    monkeypatch.setattr(module, "_coverage", lambda *_: (_ for _ in ()).throw(
        AssertionError("coverage reached after prior rejection")))
    trace = {}
    assert module._model(("a", "b"), matches, points, points.copy(), (384, 384), (384, 384),
                         tuple(range(len(matches))), 7, trace) is None
    assert trace["first_rejection"] == reason
    if mode == "missing_h":
        assert trace["mask_available"] is None
    if mode == "bad_inversion":
        assert trace["inversion_available"] is False and trace["inverse_normalized"] is None
    assert trace["held_rows"] is None and trace["source_coverage"] is None
    assert len(normalized_calls) == (0 if mode.startswith("missing") else
                                    1 if mode in ("bad_forward", "bad_inversion") else 2)


@pytest.mark.parametrize("held_count,good,reason", [
    (10, 7, "joint_holdout_starved"),
    (21, 8, "accepted"), (22, 8, "holdout_fraction"),
])
def test_joint_holdout_fraction_boundary_and_unfiltered_rows(monkeypatch, held_count, good, reason):
    matches, points, held = attempt_fixture(24, held_count)
    target = points.copy()
    for index in held[good:]:
        target[matches[index][1], 0] += 30
    monkeypatch.setattr(module.cv2, "findHomography", lambda src, *_args, **_kwargs:
                        (np.eye(3), np.ones((len(src), 1), np.uint8)))
    monkeypatch.setattr(module, "_coverage", lambda _p, _s, _t=None: True)
    trace = {}
    result = module._model(("a", "b"), matches, points, target, (384, 384), (384, 384),
                           tuple(range(len(matches))), 2, trace)
    assert trace["first_rejection"] == reason
    assert trace["held_joint_pass_count"] == good
    assert len(trace["held_rows"]) == held_count
    assert trace["held_forward"]["max_px"] == trace["held_reverse"]["max_px"] == 30
    assert (result is not None) == (reason == "accepted")


def test_pair_telemetry_serializes_infinite_held_residual(monkeypatch):
    matches, points, _ = attempt_fixture(24, 10)
    monkeypatch.setattr(module.cv2, "findHomography", lambda src, *_args, **_kwargs:
                        (np.eye(3), np.ones((len(src), 1), np.uint8)))
    original_transfer = module._transfer
    calls = []
    def transfer(h, xy):
        out = original_transfer(h, xy)
        calls.append(len(calls))
        if len(calls) == 1:
            out[0] = np.inf
        return out
    monkeypatch.setattr(module, "_transfer", transfer)
    monkeypatch.setattr(module, "_coverage", lambda _p, _s, _t=None: True)
    trace = {}
    assert module._model(("a", "b"), matches, points, points.copy(),
                         (384, 384), (384, 384), tuple(range(len(matches))), 3, trace)
    assert trace["held_rows"][0][1] == "inf"
    assert trace["held_forward"]["max_px"] == "inf"
    assert trace["held_joint_pass_count"] == 9


@pytest.mark.parametrize("points,gate", [
    (np.array([(20. + i, 20.) for i in range(11)]), "too_few_points"),
    (np.array([(100. + i % 5, 100. + i // 5) for i in range(20)]), "extent"),
    (np.array([(5. + i % 5, 5. + i // 5) for i in range(10)] +
              [(100. + i % 5, 100. + i // 5) for i in range(10)]), "grid"),
    (np.array([(x + i % 3, y + i // 3) for x, y in
               ((20., 20.), (130., 20.), (20., 130.), (130., 130.))
               for i in range(5)]), "component"),
])
def test_coverage_measurements_stop_at_first_failed_guard(points, gate):
    trace = {}
    assert not module._coverage(points, (384, 384), trace)
    assert trace["first_failed_gate"] == gate
    assert trace["passed"] is False
    if gate == "too_few_points":
        assert trace["extent_xy"] is None
    if gate in ("too_few_points", "extent"):
        assert trace["grid_cell_count"] is None
    if gate != "component":
        assert trace["connected_count"] is None


@pytest.mark.parametrize("first_failure", ["source_coverage", "destination_coverage"])
def test_pair_coverage_precedence_and_call_count(monkeypatch, first_failure):
    matches, points, _ = attempt_fixture()
    monkeypatch.setattr(module.cv2, "findHomography", lambda src, *_args, **_kwargs:
                        (np.eye(3), np.ones((len(src), 1), np.uint8)))
    calls = []
    def coverage(_points, _shape, telemetry=None):
        calls.append(len(calls))
        if telemetry is not None:
            telemetry.update(count=len(_points), extent_xy=[200., 200.], extent_area=40_000.,
                             grid_cell_count=8, connected_count=len(_points),
                             connected_required=int(np.ceil(.9 * len(_points))),
                             first_failed_gate="component" if len(calls) == (1 if first_failure == "source_coverage" else 2) else None,
                             passed=len(calls) != (1 if first_failure == "source_coverage" else 2))
        return len(calls) != (1 if first_failure == "source_coverage" else 2)
    monkeypatch.setattr(module, "_coverage", coverage)
    trace = {}
    assert module._model(("a", "b"), matches, points, points.copy(), (384, 384), (384, 384),
                         tuple(range(len(matches))), 2, trace) is None
    assert trace["first_rejection"] == first_failure
    assert len(calls) == (1 if first_failure == "source_coverage" else 2)
    assert trace["source_coverage"] is not None
    assert (trace["destination_coverage"] is None) == (first_failure == "source_coverage")


def test_accepted_then_rejected_second_attempt_and_three_model_limit(monkeypatch):
    matches, points, held = attempt_fixture(100, 30)
    monkeypatch.setattr(module, "_coverage", lambda _p, _s, _t=None: True)
    find_calls = []
    def identity_fit(src, *_args, **_kwargs):
        find_calls.append(len(src))
        return np.eye(3), np.ones((len(src), 1), np.uint8)
    monkeypatch.setattr(module.cv2, "findHomography", identity_fit)
    first_trace = {"attempts": []}
    first = module._pair_models(("a", "b"), matches, points, points.copy(),
                                ((384, 384), (384, 384)), first_trace)
    assert len(first) == 1 and len(first_trace["attempts"]) == 2
    assert [row["first_rejection"] for row in first_trace["attempts"]] == ["accepted", "fit_starved"]
    assert first_trace["termination"] == "later_attempt_rejected"
    assert find_calls == [100]
    used = set(first[0].fit_indices + first[0].holdout_indices)
    assert first_trace["attempts"][1]["remaining_indices"] == [
        i for i in range(len(matches)) if i not in used]

    # Keep 16 fit inliers and enough held passes per transform, leaving enough
    # reciprocal rows for three sequential accepted extractions.
    target = points.copy()
    for start, stop, shift in ((0, 12, 0), (12, 20, 10), (20, 28, 20), (28, 30, 30)):
        for index in held[start:stop]:
            target[matches[index][1], 1] += shift
    calls = []
    def three_fits(src, *_args, **_kwargs):
        number = len(calls)
        calls.append(len(src))
        h = np.array([[1., 0., 0.], [0., 1., number * 10.], [0., 0., 1.]])
        mask = np.zeros((len(src), 1), np.uint8)
        mask[:16] = 1
        return h, mask
    monkeypatch.setattr(module.cv2, "findHomography", three_fits)
    trace = {"attempts": []}
    accepted = module._pair_models(("a", "b"), matches, points, target,
                                   ((384, 384), (384, 384)), trace)
    assert len(accepted) == 3 and len(calls) == 3
    assert trace["termination"] == "model_limit_reached"
    assert [row["first_rejection"] for row in trace["attempts"]] == ["accepted"] * 3
    assert [row["fit_inlier_count"] for row in trace["attempts"]] == [16] * 3
    assert [row["held_joint_pass_count"] for row in trace["attempts"]] == [12, 8, 8]


def test_three_edge_model_product_replays_twenty_seven_track_attempts(monkeypatch):
    def coverage(points, _shape, trace=None):
        if trace is not None:
            trace.update(count=len(points), extent_xy=[200., 200.], extent_area=40_000.,
                         grid_cell_count=8, connected_count=len(points),
                         connected_required=int(np.ceil(.9 * len(points))),
                         first_failed_gate=None, passed=True)
        return True
    monkeypatch.setattr(module, "_coverage", coverage)
    records = []
    for pair in (("a", "b"), ("a", "c"), ("b", "c")):
        matches, points, held = attempt_fixture(100, 30, pair[0])
        for rank, (index, _) in enumerate(matches):
            points[index] = (30 + rank % 13 * 22, 30 + rank // 13 * 22)
        target = points.copy()
        for start, stop, shift in ((0, 12, 0), (12, 20, 10), (20, 28, 20), (28, 30, 30)):
            for index in held[start:stop]:
                target[matches[index][1], 1] += shift
        calls = []
        def homography(src, *_args, **_kwargs):
            number = len(calls)
            calls.append(number)
            h = np.array([[1., 0., 0.], [0., 1., number * 10.], [0., 0., 1.]])
            mask = np.zeros((len(src), 1), np.uint8)
            mask[:16] = 1
            return h, mask
        monkeypatch.setattr(module.cv2, "findHomography", homography)
        record = {"frame_ids": list(pair), "pair_key": module._pair_key(*pair),
                  "left_feature_count": 700, "right_feature_count": 700,
                  "matching_status": "evaluated",
                  "forward_ratio_pairs": [list(key) for key in matches],
                  "reverse_ratio_pairs": [list(key) for key in matches],
                  "reciprocal_pairs": [list(key) for key in matches],
                  "forward_ratio_pass_count": len(matches),
                  "reverse_ratio_pass_count": len(matches),
                  "reciprocal_count": len(matches), "attempts": []}
        models = module._pair_models(pair, matches, points, target,
                                     ((384, 384), (384, 384)), record)
        assert len(models) == 3 and len(calls) == 3
        records.append(record)
    names = ["a", "b", "c"]
    report = {
        "schema": "projective-plane-tracks-v4", "status": "abstained",
        "abstention": "no_coherent_three_view_plane", "frame_ids": names,
        "frame_sha256": {name: "0" * 64 for name in names},
        "frame_dimensions": {name: [384, 384] for name in names},
        "frame_hash_scheme": "sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)",
        "unresolved": list(module.UNRESOLVED),
        "work": {"input_rgb_bytes": 3 * 3 * 384 * 384, "pair_count": 3,
                 "feature_counts": dict.fromkeys(names, 700),
                 "descriptor_comparisons": 2 * 3 * 700 * 700,
                 "cycle_model_combinations": 27, "track_attempt_count": 27},
        "pair_telemetry": records, "tracks": [],
    }
    assert all(row["accepted_model_count"] == 3 and
               row["termination"] == "model_limit_reached" for row in records)
    assert module.canonical_report_bytes(reseal(report))
    report["work"]["cycle_model_combinations"] = 26
    report["work"]["track_attempt_count"] = 26
    with pytest.raises(ValueError, match="graph"):
        module.canonical_report_bytes(reseal(report))


@pytest.mark.parametrize("mode,reason", [
    ("no_h", "homography_unavailable"), ("no_mask", "mask_unavailable"),
    ("bad_forward", "forward_normalization"),
    ("few_inliers", "fit_inlier_starved"),
    ("source_coverage", "source_coverage"),
    ("destination_coverage", "destination_coverage"),
])
def test_abstained_report_replays_first_exit_with_null_later_stages(monkeypatch, mode, reason):
    rng = np.random.default_rng(918)
    descriptors = rng.normal(size=(80, 128)).astype(np.float32)
    points = np.array([(6 + x * 7, 5 + y * 5) for y in range(10) for x in range(8)], np.float64)
    monkeypatch.setattr(module, "_features", lambda _image: (points.copy(), descriptors.copy()))
    def homography(src, *_args, **_kwargs):
        mask = np.ones((len(src), 1), np.uint8)
        if mode == "few_inliers":
            mask[15:] = 0
        return (None if mode == "no_h" else
                np.zeros((3, 3)) if mode == "bad_forward" else np.eye(3)), (
                None if mode == "no_mask" else mask)
    monkeypatch.setattr(module.cv2, "findHomography", homography)
    if mode.endswith("coverage"):
        calls = []
        def coverage(xy, _shape, trace=None):
            calls.append(len(calls))
            passed = mode == "destination_coverage" and len(calls) % 2 == 1
            if trace is not None:
                trace.update(count=len(xy), extent_xy=[40., 40.], extent_area=1600.,
                             grid_cell_count=8, connected_count=len(xy) if passed else 1,
                             connected_required=int(np.ceil(.9 * len(xy))),
                             first_failed_gate=None if passed else "component", passed=passed)
            return passed
        monkeypatch.setattr(module, "_coverage", coverage)
    frames = [(name, np.full((64, 64, 3), i, np.uint8))
              for i, name in enumerate(("a", "b", "c"))]
    report = module.diagnose(frames)
    assert report["status"] == "abstained" and not report["tracks"]
    assert all(row["attempts"][0]["first_rejection"] == reason
               for row in report["pair_telemetry"])
    assert report["work"]["track_attempt_count"] == 0
    assert module.canonical_report_bytes(report)


def synthetic_six_view_features(monkeypatch, weak=True):
    rng = np.random.default_rng(404)
    descriptors = rng.normal(size=(80, 128)).astype(np.float32)
    xy = np.array([(6 + x * 7, 5 + y * 5) for y in range(10) for x in range(8)], np.float64)
    counts = [80, 80, 15 if weak else 80, 80, 80, 80]
    frames = [(f"view-{i:02d}", np.full((64, 64, 3), i, np.uint8)) for i in range(6)]
    def features(image):
        i = int(image[0, 0, 0])
        return xy[:counts[i]].copy(), descriptors[:counts[i]].copy()
    monkeypatch.setattr(module, "_features", features)
    return frames


def test_six_view_weak_frame_keeps_other_successful_triples(monkeypatch):
    frames = synthetic_six_view_features(monkeypatch)
    original_track = module._track
    calls = []
    def track_spy(*args):
        calls.append(tuple(args[:3]))
        return original_track(*args)
    monkeypatch.setattr(module, "_track", track_spy)
    report = module.diagnose(frames)
    assert report["status"] == "supported"
    assert len(report["pair_telemetry"]) == 15
    weak = [row for row in report["pair_telemetry"] if "view-02" in row["frame_ids"]]
    assert len(weak) == 5
    assert all(row["accepted_model_count"] == 0 and
               row["attempts"][0]["first_rejection"] == "fit_starved" for row in weak)
    strong = [row for row in report["pair_telemetry"] if "view-02" not in row["frame_ids"]]
    assert len(strong) == 10 and all(row["accepted_model_count"] == 1 for row in strong)
    assert report["work"]["cycle_model_combinations"] == report["work"]["track_attempt_count"] == 10
    assert len(calls) == 10 and all("view-02" not in triple for triple in calls)
    assert any(set(track["frame_ids"]) == {"view-00", "view-01", "view-03"}
               for track in report["tracks"])
    shared_key = module._pair_key("view-00", "view-01")
    shared = [track for track in report["tracks"] if shared_key in track["pair_held_out"]]
    assert len(shared) >= 2
    assert next(row for row in strong if row["pair_key"] == shared_key)["accepted_model_count"] == 1
    assert all(track["pair_held_out"][shared_key] == shared[0]["pair_held_out"][shared_key]
               for track in shared)
    assert module.canonical_report_bytes(report)


@pytest.mark.parametrize("edges", [
    {(0, 1), (1, 2)},
    {(0, 1), (0, 2), (0, 3)},
])
def test_accepted_chain_or_star_retains_pairs_without_track_calls(monkeypatch, edges):
    frames = synthetic_six_view_features(monkeypatch, weak=False)[:max(max(pair) for pair in edges) + 1]
    original_matches = module._matches
    pair_number = 0
    pair_order = list(itertools.combinations(range(len(frames)), 2))
    def matches(left, right, telemetry=None):
        nonlocal pair_number
        pair = pair_order[pair_number]
        pair_number += 1
        if pair in edges:
            return original_matches(left, right, telemetry)
        telemetry.update(matching_status="evaluated", forward_ratio_pairs=[],
                         reverse_ratio_pairs=[], reciprocal_pairs=[],
                         forward_ratio_pass_count=0, reverse_ratio_pass_count=0,
                         reciprocal_count=0)
        return ()
    monkeypatch.setattr(module, "_matches", matches)
    monkeypatch.setattr(module, "_track", lambda *_: (_ for _ in ()).throw(
        AssertionError("track called without accepted triangle")))
    report = module.diagnose(frames)
    assert report["status"] == "abstained"
    assert report["work"]["cycle_model_combinations"] == report["work"]["track_attempt_count"] == 0
    assert {tuple(row["frame_ids"]) for row in report["pair_telemetry"]
            if row["accepted_model_count"]} == {
                (frames[i][0], frames[j][0]) for i, j in edges}
    assert module.canonical_report_bytes(report)


def test_planar_candidate_golden_and_estimator_call_inputs_unchanged(monkeypatch):
    frames = planar_frames()
    original_seed = module.cv2.setRNGSeed
    original_fit = module.cv2.findHomography
    original_track = module._track
    seeds, fits, track_calls = [], [], []
    def seed_spy(value):
        seeds.append(value)
        return original_seed(value)
    def fit_spy(src, dst, method, threshold, *, maxIters, confidence):
        raw, mask = original_fit(src, dst, method, threshold,
                                 maxIters=maxIters, confidence=confidence)
        fits.append((src.copy(), dst.copy(), method, threshold, maxIters,
                     confidence, mask.copy() if mask is not None else None))
        return raw, mask
    def track_spy(*args):
        track_calls.append(tuple(args[:3]))
        return original_track(*args)
    monkeypatch.setattr(module.cv2, "setRNGSeed", seed_spy)
    monkeypatch.setattr(module.cv2, "findHomography", fit_spy)
    monkeypatch.setattr(module, "_track", track_spy)
    report = module.diagnose(frames)
    assert hashlib.sha256(module._canonical(report["tracks"])).hexdigest() == (
        "879fdb1fbe5bf496d091abf427151ad9a0ce5336a91d7a79163ac48add79be2c")
    assert report["tracks"][0]["triangle_count"] == 79
    assert report["tracks"][0]["held_out_triangle_count"] == 35
    pairs = list(itertools.combinations(report["frame_ids"], 2))
    assert seeds == [0] * 3 + [module._seed(pair, 0) for pair in pairs]
    assert len(fits) == len(pairs) == 3
    assert track_calls == [tuple(report["frame_ids"])]
    monkeypatch.setattr(module.cv2, "setRNGSeed", original_seed)
    extracted = {name: module._features(image)[0] for name, image in frames}
    for pair, call, record in zip(pairs, fits, report["pair_telemetry"]):
        left, right = pair
        match_rows = record["reciprocal_pairs"]
        fit_indices = record["attempts"][0]["fit_indices"]
        assert np.array_equal(call[0], extracted[left][[match_rows[i][0] for i in fit_indices]])
        assert np.array_equal(call[1], extracted[right][[match_rows[i][1] for i in fit_indices]])
        assert call[2:6] == (cv2.RANSAC, module.TRANSFER_PX, 500, .995)
        assert call[6] is not None and len(call[6]) == len(fit_indices)


@pytest.mark.parametrize("mutation", [
    "omit_pair", "duplicate_pair", "reorder_pair", "pair_key", "matcher_reach", "ratio_count",
    "reciprocal_pair", "omit_attempt", "remaining", "fit_count", "held_index",
    "inlier_subset", "held_row", "held_count", "held_summary", "coverage_metric",
    "reason", "nullable_stage", "model_count", "termination", "cycle_graph",
    "track_attempt_count", "extra_attempt",
])
def test_resealed_pair_telemetry_mutations_refuse_before_write(supported_report, mutation, tmp_path):
    report = json.loads(module.canonical_report_bytes(supported_report))
    pairs = report["pair_telemetry"]
    pair = pairs[0]
    attempt = pair["attempts"][0]
    later = pair["attempts"][1]
    if mutation == "omit_pair":
        pairs.pop()
    elif mutation == "duplicate_pair":
        pairs[1] = pairs[0]
    elif mutation == "reorder_pair":
        pairs[0], pairs[1] = pairs[1], pairs[0]
    elif mutation == "pair_key":
        pair["pair_key"] = module._pair_key(*pair["frame_ids"][::-1])
    elif mutation == "matcher_reach":
        pair["matching_status"] = "insufficient_descriptors"
    elif mutation == "ratio_count":
        pair["forward_ratio_pass_count"] += 1
    elif mutation == "reciprocal_pair":
        pair["reciprocal_pairs"].pop()
        pair["reciprocal_count"] -= 1
    elif mutation == "omit_attempt":
        pair["attempts"].pop()
    elif mutation == "remaining":
        later["remaining_indices"].append(0)
        later["remaining_count"] += 1
    elif mutation == "fit_count":
        attempt["fit_count"] -= 1
    elif mutation == "held_index":
        attempt["held_indices"][0] = attempt["fit_indices"][0]
    elif mutation == "inlier_subset":
        attempt["fit_inlier_indices"][0] = attempt["held_indices"][0]
        attempt["fit_inlier_indices"].sort()
    elif mutation == "held_row":
        attempt["held_rows"][0][1] += 0.5
    elif mutation == "held_count":
        attempt["held_joint_pass_count"] -= 1
    elif mutation == "held_summary":
        attempt["held_forward"]["max_px"] += 1
    elif mutation == "coverage_metric":
        attempt["source_coverage"]["extent_area"] += 1
    elif mutation == "reason":
        attempt["first_rejection"] = "source_coverage"
    elif mutation == "nullable_stage":
        later["homography_available"] = False
    elif mutation == "model_count":
        pair["accepted_model_count"] += 1
    elif mutation == "termination":
        pair["termination"] = "model_limit_reached"
    elif mutation == "cycle_graph":
        report["work"]["cycle_model_combinations"] = 0
        report["work"]["track_attempt_count"] = 0
    elif mutation == "track_attempt_count":
        report["work"]["track_attempt_count"] += 1
    elif mutation == "extra_attempt":
        pair["attempts"].append(json.loads(json.dumps(later)))
    reseal(report)
    with pytest.raises(ValueError):
        module.canonical_report_bytes(report)
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "mutated.json", report)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["forward", "reverse", "infinity"])
def test_consistently_resealed_attempt_residual_must_match_track(
        supported_report, mutation, tmp_path):
    report = json.loads(module.canonical_report_bytes(supported_report))
    pair = report["pair_telemetry"][0]
    attempt = pair["attempts"][0]
    rows = attempt["held_rows"]
    if mutation == "infinity":
        row = next(row for row in rows if row[1] > module.TRANSFER_PX or
                   row[2] > module.TRANSFER_PX)
        direction = 1 if row[1] > module.TRANSFER_PX else 2
        row[direction] = "inf"
    else:
        row = rows[0]
        assert pair["reciprocal_pairs"][row[0]] == [29, 111]
        direction = 1 if mutation == "forward" else 2
        row[direction] += 0.5
        assert row[direction] <= module.TRANSFER_PX
    values = np.asarray([np.inf if item[direction] == "inf" else item[direction]
                         for item in rows])
    attempt["held_forward" if direction == 1 else "held_reverse"] = module._residual_summary(values)
    reseal(report)
    with pytest.raises(ValueError, match="contradicts accepted attempt"):
        module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="contradicts accepted attempt"):
        module.persist_report(tmp_path / "contradiction.json", report)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("role", ["fit", "held"])
def test_resealed_accepted_identity_substitution_preserves_counts_but_refuses(
        supported_report, role, tmp_path):
    report = json.loads(module.canonical_report_bytes(supported_report))
    pair = report["pair_telemetry"][0]
    attempt = pair["attempts"][0]
    indices = attempt["fit_inlier_indices" if role == "fit" else "held_indices"][:2]
    first, second = (pair["reciprocal_pairs"][index][:] for index in indices)
    assert first[0] != second[0] and first[1] != second[1]
    pair["reciprocal_pairs"][indices[0]][1] = second[1]
    pair["reciprocal_pairs"][indices[1]][1] = first[1]
    forward = {row[0]: row for row in pair["forward_ratio_pairs"]}
    reverse = {row[0]: row for row in pair["reverse_ratio_pairs"]}
    forward[first[0]][1] = second[1]
    forward[second[0]][1] = first[1]
    reverse[first[1]][1] = second[0]
    reverse[second[1]][1] = first[0]
    assert pair["reciprocal_count"] == len(pair["reciprocal_pairs"])
    assert attempt["fit_inlier_count"] == len(attempt["fit_inlier_indices"])
    assert attempt["held_count"] == len(attempt["held_indices"])
    reseal(report)
    with pytest.raises(ValueError, match="contradicts accepted attempt"):
        module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="contradicts accepted attempt"):
        module.persist_report(tmp_path / "substitution.json", report)
    assert not list(tmp_path.iterdir())


def test_multiple_accepted_attempts_and_one_pair_shared_by_two_tracks(monkeypatch, tmp_path):
    matches_a, _, held_a = attempt_fixture(100, 30, "a")
    matches_b, _, _ = attempt_fixture(100, 30, "b")
    points_a = np.array([(30 + i % 20 * 15, 30 + (i // 20) % 20 * 15)
                         for i in range(700)], np.float64)
    points_b = points_a.copy()
    for start, stop, shift in ((0, 12, 0), (12, 20, 10), (20, 28, 20), (28, 30, 30)):
        for index in held_a[start:stop]:
            points_b[matches_a[index][0], 1] += shift
    points_c = points_b.copy()
    points = {"a": points_a, "b": points_b, "c": points_c}
    def coverage(xy, _shape, trace=None):
        if trace is not None:
            trace.update(count=len(xy), extent_xy=[200., 200.], extent_area=40_000.,
                         grid_cell_count=8, connected_count=len(xy),
                         connected_required=int(np.ceil(.9 * len(xy))),
                         first_failed_gate=None, passed=True)
        return True
    monkeypatch.setattr(module, "_coverage", coverage)
    models, records = {}, []
    for pair, matches, model_limit in (
            (("a", "b"), matches_a, 3),
            (("a", "c"), matches_a, 3),
            (("b", "c"), matches_b, 1)):
        calls = []
        def homography(src, *_args, **_kwargs):
            number = len(calls)
            calls.append(number)
            h = np.array([[1., 0., 0.], [0., 1., number * 10. if model_limit == 3 else 0.],
                          [0., 0., 1.]])
            mask = np.zeros((len(src), 1), np.uint8)
            mask[:16 if model_limit == 3 else len(src)] = 1
            return h, mask
        monkeypatch.setattr(module.cv2, "findHomography", homography)
        record = {"frame_ids": list(pair), "pair_key": module._pair_key(*pair),
                  "left_feature_count": 700, "right_feature_count": 700,
                  "matching_status": "evaluated",
                  "forward_ratio_pairs": [list(key) for key in matches],
                  "reverse_ratio_pairs": [list(key) for key in matches],
                  "reciprocal_pairs": [list(key) for key in matches],
                  "forward_ratio_pass_count": len(matches),
                  "reverse_ratio_pass_count": len(matches),
                  "reciprocal_count": len(matches), "attempts": []}
        models[pair] = module._pair_models(pair, matches, points[pair[0]], points[pair[1]],
                                           ((384, 384), (384, 384)), record)
        assert len(models[pair]) == model_limit
        records.append(record)
    hashes = dict.fromkeys(("a", "b", "c"), "0" * 64)
    shapes = dict.fromkeys(("a", "b", "c"), (384, 384))
    tracks = [module._track("a", "b", "c", models["a", "b"][number],
                            models["b", "c"][0], models["a", "c"][number],
                            points, shapes, hashes) for number in (0, 1)]
    assert all(track is not None for track in tracks)
    assert [(track["triangle_count"], track["held_out_triangle_count"])
            for track in tracks] == [(28, 17), (24, 14)]
    ab = module._pair_key("a", "b")
    bc = module._pair_key("b", "c")
    assert tracks[0]["pair_held_out"][ab] != tracks[1]["pair_held_out"][ab]
    assert tracks[0]["pair_held_out"][bc] == tracks[1]["pair_held_out"][bc]
    report = {
        "schema": "projective-plane-tracks-v4", "status": "supported", "abstention": None,
        "frame_ids": ["a", "b", "c"], "frame_sha256": hashes,
        "frame_dimensions": {name: [384, 384] for name in hashes},
        "frame_hash_scheme": "sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)",
        "unresolved": list(module.UNRESOLVED),
        "work": {"input_rgb_bytes": 3 * 3 * 384 * 384, "pair_count": 3,
                 "feature_counts": dict.fromkeys(hashes, 700),
                 "descriptor_comparisons": 2 * 3 * 700 * 700,
                 "cycle_model_combinations": 9, "track_attempt_count": 9},
        "pair_telemetry": records, "tracks": tracks,
    }
    raw = module.canonical_report_bytes(reseal(report))
    target = tmp_path / "two-model-tracks.json"
    module.persist_report(target, report)
    assert target.read_bytes() == raw


def test_resealed_frame_source_binding_and_telemetry_cap(supported_report, tmp_path, monkeypatch):
    frames = planar_frames()
    report = json.loads(module.canonical_report_bytes(supported_report))
    name = report["frame_ids"][0]
    report["frame_sha256"][name] = "0" * 64
    for track in report["tracks"]:
        track["frame_sha256"][name] = "0" * 64
    reseal(report)
    assert module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="drift"):
        module.verify_frame_binding(report, frames)
    report = json.loads(module.canonical_report_bytes(supported_report))
    report["frame_dimensions"][name][0] += 1
    report["work"]["input_rgb_bytes"] += report["frame_dimensions"][name][1] * 3
    reseal(report)
    assert module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="drift"):
        module.verify_frame_binding(report, frames)
    size = len(module.canonical_report_bytes(supported_report))
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", size - 1)
    with pytest.raises(ValueError, match="output budget"):
        module.persist_report(tmp_path / "too-large.json", supported_report)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("count", range(9, 33))
def test_extended_schedule_covers_every_frame_with_bounded_triangles(count):
    names = [f"frame-{i:02}" for i in range(count)]
    pairs = module._pair_schedule(names)
    triangles = module._bounded_cycle_triangles(names)
    assert pairs == sorted(set(pairs))
    assert len(pairs) == count * (count - 1) // 2 <= module.MAX_PAIRS
    assert len(triangles) == count
    assert set(itertools.chain.from_iterable(triangles)) == set(names)
    assert len(triangles) * module.MAX_MODELS_PER_PAIR ** 3 <= module.MAX_CYCLE_COMBINATIONS
    assert 2 * len(pairs) * module.MAX_FEATURES ** 2 <= module.EXTENDED_MAX_DESCRIPTOR_COMPARISONS


@pytest.mark.parametrize("count", [9, 16, 32])
def test_extended_blank_reports_bind_schedule_and_persist(count, tmp_path):
    frames = [(f"view-{i:02}", np.zeros((64, 64, 3), np.uint8)) for i in range(count)]
    report = module.diagnose(frames)
    assert report["schema"] == "projective-plane-tracks-v6"
    assert report["status"] == "abstained"
    assert report["work"]["pair_count"] == count * (count - 1) // 2
    assert report["work"]["pair_schedule"] == module.PAIR_SCHEDULE
    assert report["work"]["omitted_pair_count"] == 0
    assert report["work"]["descriptor_comparisons"] == 0
    assert [row["frame_ids"] for row in report["pair_telemetry"]] == [
        list(pair) for pair in module._pair_schedule([name for name, _ in frames])]
    module.verify_frame_binding(report, frames)
    module.persist_report(tmp_path / "v6.json", report)
    assert (tmp_path / "v6.json").read_bytes() == module.canonical_report_bytes(report)


def test_32_textured_frames_produce_deterministic_replayable_support():
    base = planar_frames(noise=False)
    frames = [(f"view-{i:02}", base[i % 3][1]) for i in range(32)]
    first = module.diagnose(frames)
    assert first == module.diagnose(frames)
    assert first["status"] == "supported"
    assert set(itertools.chain.from_iterable(track["frame_ids"] for track in first["tracks"])) == {
        name for name, _ in frames}
    assert first["work"]["descriptor_comparisons"] > module.MAX_DESCRIPTOR_COMPARISONS
    assert first["work"]["cycle_model_combinations"] <= module.MAX_CYCLE_COMBINATIONS
    module.canonical_report_bytes(first)
    module.verify_frame_binding(first, frames)


def test_extended_rgb_budget_allows_32_full_resolution_frames(monkeypatch):
    image = np.zeros((1000, 1000, 3), np.uint8)
    frames = [(f"view-{i:02}", image) for i in range(32)]
    calls = []
    def no_features(image):
        calls.append(image.shape)
        return np.empty((0, 2), np.float64), np.empty((0, 128), np.float32)
    monkeypatch.setattr(module, "_features", no_features)
    report = module.diagnose(frames)
    assert len(calls) == 32
    assert report["work"]["input_rgb_bytes"] == module.EXTENDED_MAX_TOTAL_RGB_BYTES
    module.canonical_report_bytes(report)
    with pytest.raises(ValueError, match="frame count"):
        module.diagnose(frames + [("view-32", image)])


@pytest.mark.parametrize("budget", ["EXTENDED_MAX_TOTAL_RGB_BYTES", "EXTENDED_MAX_DESCRIPTOR_COMPARISONS",
                                     "EXTENDED_MAX_OUTPUT_BYTES", "MAX_PAIRS"])
def test_extended_budget_refusals_remain_bounded(monkeypatch, budget):
    frames = [(f"view-{i:02}", np.zeros((64, 64, 3), np.uint8)) for i in range(9)]
    if budget == "EXTENDED_MAX_DESCRIPTOR_COMPARISONS":
        monkeypatch.setattr(module, "_features", lambda image: (
            np.zeros((100, 2), np.float64), np.zeros((100, 128), np.float32)))
        def forbidden_match(*args):
            raise AssertionError("matching started after work budget was exceeded")
        monkeypatch.setattr(module, "_matches", forbidden_match)
    monkeypatch.setattr(module, budget, 1)
    with pytest.raises(ValueError, match="budget"):
        module.diagnose(frames)


@pytest.mark.parametrize("mutation", ["policy", "omitted", "pair_count", "pair_identity", "descriptor_work",
                                       "v4_disguise", "ledger_missing", "ledger_reordered"])
def test_resealed_extended_schedule_mutations_refuse_before_publication(mutation, tmp_path):
    report = module.diagnose([(f"view-{i:02}", np.zeros((64, 64, 3), np.uint8)) for i in range(9)])
    if mutation == "policy":
        report["work"]["pair_schedule"] = "id-order-cyclic-distance-2-v1"
    elif mutation == "omitted":
        report["work"]["omitted_pair_count"] += 1
    elif mutation == "pair_count":
        report["work"]["pair_count"] += 1
    elif mutation == "pair_identity":
        record = report["pair_telemetry"][0]
        record["frame_ids"] = ["view-00", "view-04"]  # unscheduled
        record["pair_key"] = module._pair_key(*record["frame_ids"])
    elif mutation == "descriptor_work":
        report["work"]["descriptor_comparisons"] += 1
    elif mutation == "v4_disguise":
        report["schema"] = "projective-plane-tracks-v4"
        report["work"].pop("pair_schedule")
        report["work"].pop("omitted_pair_count")
    elif mutation == "ledger_missing":
        report["pair_telemetry"].pop()
    elif mutation == "ledger_reordered":
        report["pair_telemetry"][0], report["pair_telemetry"][1] = report["pair_telemetry"][1], report["pair_telemetry"][0]
    payload = dict(report)
    payload.pop("canonical_sha256")
    report["canonical_sha256"] = module._digest_bytes(module._canonical(payload))
    with pytest.raises(ValueError):
        module.persist_report(tmp_path / "invalid.json", report)
    assert not list(tmp_path.iterdir())


def test_legacy_eight_frame_schedule_and_schema_remain_all_pairs():
    frames = [(f"view-{i:02}", np.zeros((64, 64, 3), np.uint8)) for i in range(8)]
    report = module.diagnose(frames)
    assert report["schema"] == "projective-plane-tracks-v4"
    assert report["work"]["pair_count"] == 28
    assert "pair_schedule" not in report["work"]
    assert "omitted_pair_count" not in report["work"]
    assert module._pair_schedule(report["frame_ids"]) == list(itertools.combinations(report["frame_ids"], 2))
    module.canonical_report_bytes(report)
