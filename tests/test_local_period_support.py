"""Fixed local-window diagnostic controls; generated arrays only."""
import json
from collections import Counter

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice, lattice_planes as planes, region_growth as growth
from scripts.evaluation import diagnose_image_geometry_failure as runner


def _fixture():
    y, x = np.indices((513, 513), dtype=float)
    image = np.where(x < 257, (x % 16 < 2).astype(float),
                     10000. * np.exp((x + y) / 256.)).astype(np.float32)
    return image, np.ones(image.shape, np.float32)


def _kwargs(image, mask, *, centre=(0., 0.), extent=1., cap=512):
    return dict(image=image, axis=2, axis_matrix=np.eye(3), focal=256., mask=mask,
                centre=centre, principal_point=(256., 256.), extent=extent,
                side=1., max_rectified_side=cap)


def _parent(kwargs, observer, identity=("synthetic", 0)):
    with growth._scalar_period_context(identity=identity, axis=2):
        growth._scalar_period_expect("period_measurement")
        return planes._bounded_measure(**kwargs)


def _diagnose(kwargs, *, observer=None, enabled=True, identity=("synthetic", 0)):
    observer = observer or growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        with planes._local_period_diagnostic_context(observer, enabled=enabled):
            parent = _parent(kwargs, observer, identity)
            if parent[0] is None:
                planes._local_period_proposals(**kwargs, parent_identity=identity)
    record = observer.record()
    runner._validate_attrition_record(record)
    return parent, record


def _details(record, stage="local_period_child"):
    return [row["details"] for row in record["stages"][stage]["counts"]]


def test_integer_bounded_measure_has_float_parity_and_complete_neutral_observation(monkeypatch):
    coordinate = np.arange(513)
    image = np.tile(np.where(coordinate % 16 < 2, 255, 0).astype(np.uint8), (513, 1))
    mask = np.ones(image.shape, np.float32)
    before = image.tobytes(), mask.tobytes()
    kwargs = _kwargs(image, mask, cap=513)
    calls = Counter()
    for owner, name in ((lattice, "rectify_plane"), (lattice, "_profile"),
                        (lattice.np, "diff"), (lattice.np, "correlate"),
                        (lattice.np, "exp"), (lattice.np, "angle")):
        original = getattr(owner, name)
        def track(*args, _original=original, _name=name, **kw):
            calls[_name] += 1
            return _original(*args, **kw)
        monkeypatch.setattr(owner, name, track)
    ordinary = planes._bounded_measure(**kwargs)
    ordinary_calls = calls.copy()
    assert ordinary[0] is not None and ordinary[2].dtype == np.uint8
    floating = planes._bounded_measure(**{**kwargs, "image": image.astype(np.float64)})
    assert ordinary[:2] == floating[:2] and ordinary[4] == floating[4]
    np.testing.assert_array_equal(ordinary[2], floating[2])
    np.testing.assert_array_equal(ordinary[3], floating[3])
    records = []
    for _ in range(2):
        calls.clear()
        observed, record = _diagnose(kwargs, enabled=False)
        assert observed[:2] == ordinary[:2] and observed[4] == ordinary[4]
        np.testing.assert_array_equal(observed[2], ordinary[2])
        np.testing.assert_array_equal(observed[3], ordinary[3])
        assert calls == ordinary_calls
        assert record["status"] == "complete"
        assert not any(stage.startswith("local_") for stage in record["stages"])
        records.append(record)
    assert records[0] == records[1]
    assert records[0]["stages"]["period_measurement"]["classified"] == 1
    assert records[0]["stages"]["period_round"]["classified"] == 1
    assert (image.tobytes(), mask.tobytes()) == before


def test_accepted_fixed_clutter_falsifier_recovers_two_predetermined_children():
    image, mask = _fixture()
    parent, record = _diagnose(_kwargs(image, mask))
    assert parent[0] is None and record["status"] == "complete"
    rows = sorted(_details(record), key=lambda row: row["child_index"])
    assert len(rows) == 5
    for index, row in enumerate(rows):
        assert row["identity"] == ("synthetic", 0, "local", index)
        assert row["parent_identity"] == ("synthetic", 0)
        assert row["requested_samples"] == row["measured_samples"] == 512
        assert row["extent"] == .5 and row["pixel_step"] == 1 / 511
        if index < 2:
            assert row["plane_unit_period"] == pytest.approx(.0625, rel=.02)
            assert row["period_pixels"] == pytest.approx(31.954404830932617)
            block, texel = row["hypotheses"]
            assert block["plane_distance"] == pytest.approx(15.991535523933152)
            assert texel["plane_distance"] == pytest.approx(.999470970245822)
            assert row["result_state"] == "measured"
        else:
            assert row["period_pixels"] is None and row["hypotheses"] == ()
            assert row["result_state"] == "abstained"
    assert record["stages"]["local_period_parent"]["classified"] == 1
    assert record["stages"]["local_period_measurement"]["classified"] == 5
    assert record["checks"]["local_period_child_accounting"]


@pytest.mark.parametrize("kind", ("constant", "monotonic"))
def test_constant_and_nonperiodic_monotonic_children_abstain(kind):
    y, x = np.indices((513, 513), dtype=float)
    image = (np.ones(x.shape) if kind == "constant" else np.exp((x + y) / 256.)).astype(np.float32)
    parent, record = _diagnose(_kwargs(image, np.ones(image.shape, np.float32)))
    assert parent[0] is None and record["status"] == "complete"
    assert all(row["period_pixels"] is None and row["hypotheses"] == () for row in _details(record))


def test_distinct_child_spacings_remain_independent_canonical_alternatives():
    y, x = np.indices((513, 513), dtype=float)
    # The upper/lower clean quadrants contain different known grid spacings.
    # Shared boundary energy has the same unit amplitude, not bright clutter.
    grid = np.where(y <= 256, x % 16 < 2, x % 24 < 2).astype(float)
    image = np.where(x < 257, grid, 10000. * np.exp((x + y) / 256.)).astype(np.float32)
    parent, record = _diagnose(_kwargs(image, np.ones(image.shape, np.float32)))
    assert parent[0] is None and record["status"] == "complete"
    rows = sorted(_details(record), key=lambda row: row["child_index"])
    for row, spacing in zip(rows[:2], (24. / 256., 16. / 256.)):
        assert row["plane_unit_period"] == pytest.approx(spacing, rel=.02)
        assert row["measured_samples"] == 512
        converted = lattice.period_hypotheses(
            lattice.PeriodEstimate(row["period_pixels"], 0., row["strength"]),
            extent=row["extent"], samples=row["measured_samples"])
        expected = sorted(h.plane_distance for h in converted.hypotheses)
        assert sorted(h["plane_distance"] for h in row["hypotheses"]) == expected
    assert rows[0]["plane_unit_period"] != rows[1]["plane_unit_period"]
    assert rows[0]["hypotheses"] != rows[1]["hypotheses"]
    assert not any("consensus" in row or "reading" in row for row in rows)


@pytest.mark.parametrize("centre,extent,cap", (((.2, -.3), .8, 256), ((-.7, .4), 1.6, 128)))
def test_translated_unequal_extents_and_original_boundary_mask_mapping(monkeypatch, centre, extent, cap):
    y, x = np.indices((513, 513))
    image = np.zeros(x.shape, np.float32)
    mask = ((x > 96) & (x < 405) & (y > 112) & (y < 433)).astype(np.float32)
    calls = []
    original = lattice.rectify_plane
    def rectify(source, *args, **kwargs):
        result = original(source, *args, **kwargs)
        state = growth._period_scalar_state.get()
        if state and state["stage_prefix"] == "local_":
            calls.append((source is image, source is mask, dict(kwargs), result.copy()))
        return result
    monkeypatch.setattr(lattice, "rectify_plane", rectify)
    parent, record = _diagnose(_kwargs(image, mask, centre=centre, extent=extent, cap=cap))
    assert parent[0] is None and record["status"] == "complete"
    rows = sorted(_details(record), key=lambda row: row["child_index"])
    expected_centres = [(centre[0] + du * extent / 2, centre[1] + dv * extent / 2)
                        for du, dv in planes._LOCAL_PERIOD_WINDOWS]
    assert [row["centre"] for row in rows] == expected_centres
    assert len(calls) == 10
    for i, row in enumerate(rows):
        image_call, mask_call = calls[2*i:2*i+2]
        assert image_call[:2] == (True, False) and mask_call[:2] == (False, True)
        assert mask_call[2]["centre"] == expected_centres[i]
        assert mask_call[2]["extent"] == extent / 2 and mask_call[2]["samples"] == cap
        assert row["parent_centre"] == centre and row["parent_extent"] == extent
        assert row["extent"] == extent / 2 and row["measured_samples"] == cap
        assert row["pixel_step"] == extent / (cap - 1)
        np.testing.assert_array_equal(mask_call[3], original(mask, 2, np.eye(3), 256., **mask_call[2]))
    assert any(np.any((call[3] > 0) & (call[3] < 1)) for call in calls if call[1])


def test_default_off_and_opt_in_preserve_parent_results_digests_and_separate_calls(monkeypatch):
    image, mask = _fixture()
    calls = Counter()
    targets = ((lattice, "rectify_plane"), (lattice.np, "correlate"),
               (lattice.np, "diff"), (lattice.np, "divide"),
               (lattice.np, "flatnonzero"), (lattice.np, "exp"), (lattice.np, "angle"))
    for owner, name in targets:
        original = getattr(owner, name)
        def track(*args, _original=original, _name=name, **kwargs):
            state = growth._period_scalar_state.get()
            local = state is not None and state["stage_prefix"] == "local_"
            calls[("child" if local else "parent", _name)] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, track)
    kwargs = _kwargs(image, mask)
    with growth._scalar_attrition_context(None):
        off = planes._bounded_measure(**kwargs)
    baseline = calls.copy(); calls.clear()
    disabled, disabled_record = _diagnose(kwargs, enabled=False)
    assert calls == baseline and not any(s.startswith("local_") for s in disabled_record["stages"])
    calls.clear()
    enabled, enabled_record = _diagnose(kwargs)
    assert {key: n for key, n in calls.items() if key[0] == "parent"} == dict(baseline)
    assert calls[("child", "rectify_plane")] == 10 and calls[("child", "correlate")] == 8
    assert off[0:2] == disabled[0:2] == enabled[0:2] and off[4] == disabled[4] == enabled[4]
    for result in (disabled, enabled):
        np.testing.assert_array_equal(off[2], result[2]); np.testing.assert_array_equal(off[3], result[3])
    for stage, record in disabled_record["stages"].items():
        assert record == enabled_record["stages"][stage]


@pytest.mark.parametrize("cause", ("target_period_reached", "image_rectification_failed", "mask_rectification_failed", "zero_mask_support"))
def test_noneligible_parent_reasons_do_not_allocate_children(monkeypatch, cause):
    image, mask = _fixture(); observer = growth._ScalarAttritionObserver()
    observer.expect("period_measurement", ("synthetic", 0))
    observer.finish("period_measurement", ("synthetic", 0), cause, axis=2)
    monkeypatch.setattr(lattice, "rectify_plane", lambda *a, **k: pytest.fail("ineligible child allocation"))
    with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
        planes._local_period_proposals(**_kwargs(image, mask), parent_identity=("synthetic", 0))
    assert not any(s.startswith("local_") for s in observer.record()["stages"])


def test_five_expectations_are_reserved_before_first_allocation_and_fortieth_limit(monkeypatch):
    image = np.ones((65, 65), np.float32); mask = np.ones_like(image)
    observer = growth._ScalarAttritionObserver(); calls = []
    original = lattice.rectify_plane
    def rectify(*args, **kwargs):
        assert len(observer.pending("local_period_child")) == 5 - len(calls) // 2 % 5
        calls.append(1)
        return np.ones((64, 64), np.float32)
    monkeypatch.setattr(lattice, "rectify_plane", rectify)
    with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
        for i in range(9):
            identity = ("synthetic", i)
            observer.expect("period_measurement", identity)
            observer.finish("period_measurement", identity, "no_directional_period", axis=2)
            planes._local_period_proposals(**_kwargs(image, mask, cap=64), parent_identity=identity)
    record = observer.record()
    assert len(calls) == 80 and record["status"] == "incomplete"
    assert record["stages"]["local_period_child"]["expected"] == 40
    assert record["stages"]["local_period_parent"]["classified"] == 9
    assert any(row["disposition"] == "child_budget_exhausted" for row in _details(record, "local_period_parent"))


@pytest.mark.parametrize("fault", ("omit_expect", "omit_finish", "omit_both", "duplicate", "budget"))
def test_child_omission_duplication_or_budget_is_incomplete_before_atomic_report(monkeypatch, tmp_path, fault):
    image, mask = _fixture()
    observer = growth._ScalarAttritionObserver(max_bytes=7000 if fault == "budget" else 16 * 1024**2)
    original_expect, original_finish = observer.expect, observer.finish
    target = ("synthetic", 0, "local", 2)
    def expect(stage, identity):
        if stage == "local_period_child" and identity == target and fault in ("omit_expect", "omit_both"):
            return
        original_expect(stage, identity)
    def finish(stage, identity, disposition, **fields):
        if stage == "local_period_child" and identity == target:
            if fault in ("omit_finish", "omit_both"): return
            if fault == "duplicate": original_finish(stage, identity, disposition, **fields)
        original_finish(stage, identity, disposition, **fields)
    monkeypatch.setattr(observer, "expect", expect); monkeypatch.setattr(observer, "finish", finish)
    parent, record = _diagnose(_kwargs(image, mask), observer=observer)
    assert parent[0] is None and record["status"] == "incomplete"
    with pytest.raises(runner.DiagnosticError, match="vertical_attrition_closure"):
        runner._validate_attrition_record(dict(record, status="complete"))
    path = tmp_path / "report.json"
    runner._publish_new(path, runner._bounded_bytes(record))
    assert json.loads(path.read_bytes())["status"] == "incomplete"


def test_duplicate_parent_refuses_before_extra_child_calls(monkeypatch):
    image, mask = _fixture(); observer = growth._ScalarAttritionObserver()
    calls = []
    original = lattice.rectify_plane
    monkeypatch.setattr(lattice, "rectify_plane", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
        _parent(_kwargs(image, mask), observer)
        planes._local_period_proposals(**_kwargs(image, mask), parent_identity=("synthetic", 0))
        before = len(calls)
        planes._local_period_proposals(**_kwargs(image, mask), parent_identity=("synthetic", 0))
    assert len(calls) == before == 12 and observer.record()["status"] == "incomplete"


def test_runner_flag_is_explicit_and_default_off():
    args = runner._parse_args(["--image", "a.png", "--output", "new"])
    assert args.local_period_support is False
    assert runner._parse_args(["--image", "a.png", "--output", "new", "--local-period-support"]).local_period_support


def _reading_seam(image, mask):
    diagnostics = {"abstentions": [], "low_structural_share": 0}
    readings = planes._measure_plane_candidates(image, mask.astype(bool), np.zeros_like(image),
        [(2, mask, (), 1., None)], np.eye(3), 256., (256., 256.), frame="synthetic",
        max_readings=6, min_structural_share=.15, max_rectified_side=512, diagnostics=diagnostics)
    return readings, diagnostics


def test_actual_reading_seam_output_and_parent_digests_are_identical(monkeypatch):
    image, mask = _fixture(); calls = Counter()
    monkeypatch.setattr(lattice, "plane_window", lambda *a, **k: (0., 0., 1., 1.))
    original = planes._bounded_measure
    def measure(*args, **kwargs):
        state = growth._period_scalar_state.get()
        calls["child" if state and state["stage_prefix"] == "local_" else "parent"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(planes, "_bounded_measure", measure)
    baseline = _reading_seam(image, mask)
    assert calls == {"parent": 1}; calls.clear()
    records = []
    for enabled in (False, True):
        observer = growth._ScalarAttritionObserver()
        with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=enabled):
            result = _reading_seam(image, mask)
        assert result == baseline and result[0] == ()
        assert calls["parent"] == 1 and calls["child"] == (5 if enabled else 0)
        calls.clear(); records.append(observer.record())
    assert records[0]["status"] == records[1]["status"] == "complete"
    for stage, old in records[0]["stages"].items():
        assert old == records[1]["stages"][stage]
    assert records[1]["stages"]["local_period_parent"]["expected"] == 1
    assert {row["parent_identity"] for row in _details(records[1])} == {("synthetic", 0)}


def test_wholly_missing_local_helper_at_reading_seam_remains_incomplete(monkeypatch):
    image, mask = _fixture(); observer = growth._ScalarAttritionObserver()
    monkeypatch.setattr(lattice, "plane_window", lambda *a, **k: (0., 0., 1., 1.))
    monkeypatch.setattr(planes, "_local_period_proposals", lambda *a, **k: None)
    with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
        result = _reading_seam(image, mask)
    assert result[0] == ()
    record = observer.record()
    assert record["status"] == "incomplete"
    assert record["stages"]["local_period_parent"]["unclassified"] == 1


@pytest.mark.parametrize("operand", ("parent", "child"))
def test_reading_seam_exceptions_preserve_parent_behavior(monkeypatch, operand):
    image, mask = _fixture(); observer = growth._ScalarAttritionObserver()
    monkeypatch.setattr(lattice, "plane_window", lambda *a, **k: (0., 0., 1., 1.))
    original = lattice.rectify_plane
    calls = Counter()
    def rectify(*args, **kwargs):
        state = growth._period_scalar_state.get()
        scope = "child" if state and state["stage_prefix"] == "local_" else "parent"
        calls[scope] += 1
        if scope == operand: raise RuntimeError("synthetic kernel failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(lattice, "rectify_plane", rectify)
    if operand == "parent":
        for enabled in (False, True):
            with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=enabled):
                with pytest.raises(RuntimeError, match="synthetic kernel failure"):
                    _reading_seam(image, mask)
        assert calls == {"parent": 2}
        assert not any(stage.startswith("local_") for stage in observer.record()["stages"])
    else:
        baseline = _reading_seam(image, mask)
        calls.clear()
        with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
            result = _reading_seam(image, mask)
        assert result == baseline and calls == {"parent": 2, "child": 1}
        record = observer.record()
        assert record["status"] == "incomplete"
        assert record["stages"]["local_period_child"]["unclassified"] == 5
        assert _details(record, "local_period_parent")[0]["disposition"] == "exception"


@pytest.mark.parametrize("cap", (63, 2049))
def test_invalid_child_side_cap_refuses_before_rectification(monkeypatch, cap):
    image, mask = _fixture(); observer = growth._ScalarAttritionObserver()
    observer.expect("period_measurement", ("synthetic", 0))
    observer.finish("period_measurement", ("synthetic", 0), "no_directional_period", axis=2)
    monkeypatch.setattr(lattice, "rectify_plane", lambda *a, **k: pytest.fail("invalid cap allocated"))
    with growth._scalar_attrition_context(observer), planes._local_period_diagnostic_context(observer, enabled=True):
        planes._local_period_proposals(**_kwargs(image, mask, cap=cap), parent_identity=("synthetic", 0))
    assert observer.record()["status"] == "incomplete"
    assert _details(observer.record(), "local_period_parent")[0]["disposition"] == "invalid_geometry_or_side_cap"


@pytest.mark.parametrize("flag", (1, "false", None, [], {}))
def test_runner_rejects_non_boolean_opt_in_before_inputs_or_allocation(flag, monkeypatch):
    monkeypatch.setattr(runner, "_input_records", lambda *a, **k: pytest.fail("invalid flag reached inputs"))
    with pytest.raises(runner.DiagnosticError, match="local_period_support"):
        runner.run(images=(), output="never-created", local_period_support=flag)
