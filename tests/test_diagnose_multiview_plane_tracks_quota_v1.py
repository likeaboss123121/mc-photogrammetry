"""Synthetic-only fit/held quota controls; no project images or truth files."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from scripts.evaluation import diagnose_multiview_plane_tracks_quota_v1 as quota
from scripts.evaluation import diagnose_multiview_plane_tracks as legacy
from scripts.evaluation import diagnose_multiview_mnn_quota_v1 as paired


def _matches(pair, n, legacy_role=None):
    left = [i for i in range(1000) if legacy_role is None or
            (quota._seed(pair[0], i) % 5 == 0) == legacy_role][:n]
    assert len(left) == n
    return tuple((i, j) for j, i in enumerate(left))


def _points_for(matches):
    left = np.zeros((max(i for i, _ in matches) + 1, 2), np.float64)
    right = np.zeros((len(matches), 2), np.float64)
    for j, (i, _) in enumerate(matches):
        point = [65. + 35. * (j % 6), 70. + 35. * (j // 6)]
        left[i] = point
        right[j] = point
    return left, right


def test_quota_boundary_starvation_repair_and_legacy_identity_preservation():
    pair = ("left", "right")
    for n in (23, 24):
        matches = _matches(pair, n, False)
        fit, held, mode = quota._quota_partition(pair, matches)
        if n == 23:
            assert mode == "legacy-preserved" and len(held) == 0
            a, b = _points_for(matches)
            row = {}
            assert quota._model(pair, matches, a, b, (384, 384), (384, 384),
                                tuple(range(n)), (fit, held, mode), 17, row) is None
            assert row["first_rejection"] == "held_starved"
            assert row["homography_available"] is None
        else:
            assert mode == "quota-sha256" and (len(fit), len(held)) == (16, 8)
            assert set(fit).isdisjoint(held) and set(fit + held) == set(range(n))
            a, b = _points_for(matches)
            row = {}
            model = quota._model(pair, matches, a, b, (384, 384), (384, 384),
                                 tuple(range(n)), (fit, held, mode), 17, row)
            assert model is not None and row["first_rejection"] == "accepted"
            assert row["source_coverage"]["grid_cell_count"] >= 4
            assert row["destination_coverage"]["passed"] is True
            b[matches[held[0]][1], 0] += 4.0
            refused = {}
            assert quota._model(pair, matches, a, b, (384, 384), (384, 384),
                                tuple(range(n)), (fit, held, mode), 17, refused) is None
            assert refused["first_rejection"] == "joint_holdout_starved"
    valid_left = sorted([i for i, _ in _matches(pair, 24, False)] +
                        [i for i, _ in _matches(pair, 8, True)])
    valid = tuple((left, j) for j, left in enumerate(valid_left))
    f, h, mode = quota._quota_partition(pair, valid)
    assert len(f) >= 16 and len(h) >= 8 and mode == "legacy-preserved"
    assert f == tuple(i for i, (left, _) in enumerate(valid) if quota._seed(pair[0], left) % 5)
    assert h == tuple(i for i, (left, _) in enumerate(valid) if not quota._seed(pair[0], left) % 5)
    a, b = _points_for(valid)
    old = legacy._pair_models(pair, valid, a, b, ((384, 384), (384, 384)))
    new = quota._pair_models(pair, valid, a, b, ((384, 384), (384, 384)))
    assert [dataclasses.asdict(row) for row in new] == [dataclasses.asdict(row) for row in old]


def test_quota_permutation_pair_identity_and_frozen_sequential_roles():
    pair = ("left", "right")
    matches = _matches(pair, 48, False)
    f, h, mode = quota._quota_partition(pair, matches)
    assert mode == "quota-sha256"
    permuted = tuple(reversed(matches))
    f2, h2, mode2 = quota._quota_partition(pair, permuted)
    assert mode2 == mode
    assert {matches[i] for i in h} == {permuted[i] for i in h2}
    assert quota._quota_partition(pair, matches) == (f, h, mode)
    assert {matches[i] for i in h} != {matches[i] for i in quota._quota_partition(("other", "right"), matches)[1]}
    remaining = tuple(i for i in range(48) if i not in set(f[:16] + h[:8]))
    assert not set(h).intersection(i for i in f if i in remaining)
    assert set(i for i in h if i in remaining).isdisjoint(i for i in f if i in remaining)


def test_sequential_attempts_keep_original_roles(monkeypatch):
    from types import SimpleNamespace

    pair = ("left", "right")
    matches = _matches(pair, 48, False)
    fit, held, mode = quota._quota_partition(pair, matches)
    seen = []

    def fake_model(_pair, _matches, a, b, sa, sb, remaining, roles, seed, telemetry):
        seen.append((remaining, roles))
        if telemetry is not None:
            telemetry["first_rejection"] = "accepted" if len(seen) == 1 else "fit_starved"
        if len(seen) == 1:
            return SimpleNamespace(fit_indices=fit[:16], holdout_indices=held[:8])
        return None

    monkeypatch.setattr(quota, "_model", fake_model)
    dummy = np.zeros((1000, 2))
    quota._pair_models(pair, matches, dummy, dummy, ((384, 384), (384, 384)))
    assert len(seen) == 2 and seen[0][1] == seen[1][1] == (fit, held, mode)
    assert not set(fit[:16] + held[:8]).intersection(seen[1][0])
    assert set(held).isdisjoint(fit)


def test_quota_report_reseal_refuses_false_partition_and_policy_binding():
    blank = np.zeros((64, 64, 3), np.uint8)
    frames = [(name, blank) for name in ("a", "b", "c")]
    report = quota.diagnose(frames)
    assert quota.canonical_report_bytes(report)
    assert report["partition_policy"]["config_sha256"] == quota.QUOTA_CONFIG_SHA256
    mutated = copy.deepcopy(report)
    mutated["partition_policy"]["policy_id"] = "false-partition"
    payload = dict(mutated)
    payload.pop("canonical_sha256")
    mutated["canonical_sha256"] = quota._digest_bytes(quota._canonical(payload))
    with pytest.raises(ValueError, match="quota policy"):
        quota.canonical_report_bytes(mutated)
    assert quota.MAX_FEATURES == legacy.MAX_FEATURES == 700
    assert quota.TRANSFER_PX == legacy.TRANSFER_PX == 3
    assert quota.MIN_HOLDOUT == legacy.MIN_HOLDOUT == 8


def _accepted_quota_track():
    names = ("a", "b", "c")
    chosen = [i for i in range(1000) if all(quota._seed(name, i) % 5 != 0 for name in names[:2])][:24]
    assert len(chosen) == 24
    points = np.zeros((max(chosen) + 1, 2), np.float64)
    for ordinal, index in enumerate(chosen):
        points[index] = [65. + 35. * (ordinal % 6), 70. + 35. * (ordinal // 6)]
    feature_points = {name: points.copy() for name in names}
    frames = [(name, np.zeros((384, 384, 3), np.uint8)) for name in names]
    data, _ = quota._frame_input(frames)
    pairs = quota._pair_schedule(names)
    reciprocal = [[index, index] for index in chosen]
    proposals = {pair: reciprocal for pair in pairs}
    ledgers = {pair: dict(matching_status="evaluated", forward_ratio_pairs=reciprocal,
                          reverse_ratio_pairs=reciprocal, reciprocal_pairs=reciprocal,
                          forward_ratio_pass_count=24, reverse_ratio_pass_count=24,
                          reciprocal_count=24) for pair in pairs}
    comparisons = 2 * len(pairs) * len(points) ** 2
    return paired._geometry_from_proposals(data, feature_points, proposals, ledgers,
                                           comparisons, quota)


def test_accepted_quota_track_replays_and_resealed_partition_tamper_refuses():
    report = _accepted_quota_track()
    assert report["status"] == "supported" and report["tracks"]
    assert all(row["attempts"][0]["partition_mode"] == "quota-sha256"
               for row in report["pair_telemetry"])
    assert quota.canonical_report_bytes(report)
    altered = copy.deepcopy(report)
    attempt = altered["pair_telemetry"][0]["attempts"][0]
    attempt["fit_indices"][0], attempt["held_indices"][0] = (
        attempt["held_indices"][0], attempt["fit_indices"][0])
    payload = dict(altered)
    payload.pop("canonical_sha256")
    altered["canonical_sha256"] = quota._digest_bytes(quota._canonical(payload))
    with pytest.raises(ValueError, match="deterministic attempt partition"):
        quota.canonical_report_bytes(altered)
