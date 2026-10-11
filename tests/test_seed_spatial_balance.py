"""Focused synthetic-only falsifier tests; production source is parsed, never imported."""
import ast
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/evaluation/seed_spatial_balance.py"
SOURCE = ROOT / "src/mc_photogrammetry/reconstruction/region_growth.py"
spec = importlib.util.spec_from_file_location("spatial_under_test", HELPER)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def source_oracle(supports):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_normal_scores")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["_normal_scores"](supports)


def split(**kwargs):
    return m._synthetic([(0, 24, 0), (0, 0, 24)], **kwargs)


def row(report, node=0, axis=0, frame=0):
    return next(r for r in report["frames"][frame]["rows"] if r["node"] == node and r["axis"] == axis)


def change_edge(frame, field, value):
    edges = [list(r) for r in frame.adjacency]
    for i, rows in enumerate(edges):
        edges[i] = [tuple(value if k == field else v for k, v in enumerate(r)) for r in rows]
    return replace(frame, adjacency=tuple(tuple(r) for r in edges))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_source_exact_bytes_and_empty_dtype(dtype):
    rng = np.random.default_rng(77)
    support = rng.uniform(0, 50, (80, 3)).astype(dtype)
    np.testing.assert_array_equal(m.source_scores(support).view(np.uint8), source_oracle(support).view(np.uint8))
    zero = np.zeros((2, 3), dtype=dtype)
    np.testing.assert_array_equal(m.source_scores(zero), source_oracle(zero))
    frame = split()
    frame = replace(frame, supports=zero, plane_scores=source_oracle(zero))
    report = m.replay((frame,))
    assert not any(r["seed"] for r in report["frames"][0]["rows"])
    empty_support = np.zeros((0, 3), dtype=dtype)
    assert m.source_scores(empty_support).tobytes() == source_oracle(empty_support).tobytes()


def test_paired_falsifier_truth_only_after_seal_and_frozen_matrix():
    frames, truth = m.synthetic_matrix()
    assert len(frames) == len(truth) == len(m.FIXTURE_NAMES) == 12
    a, b = frames[2], frames[7]
    for field in a.__dataclass_fields__:
        left, right = getattr(a, field), getattr(b, field)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        else:
            assert left == right
    report = m.replay(frames)
    sealed = m.seal(report)
    assert len(sealed.payload) == report["resources"]["output_bytes"]
    result = m.evaluate(sealed, truth)
    assert result["hypothesis_falsified"] is True
    assert result["false_new_seeds"] == 2
    assert result["new_seeds"] == 8
    assert result["correct_new_seeds"] == 6
    assert result["cases"][2]["recall_gain"] == 1.0
    assert result["cases"][7]["false_new_seeds"] == 2
    assert result["gates"] == dict(positive_recall=True, new_seed_precision=False,
        zero_false_plane_seeds=False, baseline_retained=True)
    assert result["disposition"] == "stop_before_image_capture"
    for node in (0, 1):
        positive, negative = row(report, node, frame=2), row(report, node, frame=7)
        assert positive == negative
        assert positive["baseline_score"] == 0 and positive["score"] == 0.5
        assert positive["pooled_balance"] == 1 and positive["reliability"] == 0.5
        assert positive["suppression"] == 1 and positive["tangent_support"] == 24
        assert positive["seed"] and not positive["baseline_seed"]
    with pytest.raises(m.Refusal):
        m.evaluate(report, truth)
    with pytest.raises(TypeError):
        m.replay(frames, truth=truth)
    assert sealed == m.seal(m.replay(frames))
    print("SYNTHETIC_RECEIPT=" + json.dumps(dict(helper_sha256=sealed.sha256,
        output_bytes=len(sealed.payload), resources=report["resources"], evaluation=result), sort_keys=True))


def test_target_only_change_support_mass_and_input_neutrality():
    frames, _ = m.synthetic_matrix()
    before = [[getattr(f, name).tobytes() for name in f.__dataclass_fields__ if isinstance(getattr(f, name), np.ndarray)] for f in frames]
    report = m.replay(frames)
    for f, output in zip(frames, report["frames"]):
        assert output["support_sha256"] == hashlib.sha256(f.supports.tobytes()).hexdigest()
        assert output["support_mass"] == f.supports.astype(np.float64).sum(axis=0).tolist()
        for r in output["rows"]:
            n, a = r["node"], r["axis"]
            balance, suppression, reliability, support = m._factors(f.supports, a)
            assert r["suppression"] == float(suppression[n])
            assert r["reliability"] == float(reliability[n])
            assert r["tangent_support"] == float(support[n])
            assert r["normal_support"] == float(f.supports[n, a])
            assert r["side"] == int(f.sides[n, a])
            assert r["opponent"] == float(max(f.plane_scores[n, k] for k in range(3) if k != a))
            expected_margin = np.array([r["score"]], np.float32) - np.array([r["opponent"]], np.float32)
            assert r["margin"] == float(expected_margin[0])
            assert r["balance"] >= float(balance[n])
            assert r["score"] >= r["baseline_score"]
    after = [[getattr(f, name).tobytes() for name in f.__dataclass_fields__ if isinstance(getattr(f, name), np.ndarray)] for f in frames]
    assert after == before


def test_unequal_area_equal_density_mean_and_canonical_reduction():
    frame = m._synthetic([(0, 24, 0), (0, 0, 48)], areas=[256, 512])
    r = row(m.replay((frame,)))
    assert r["pooled_balance"] == 1 and r["score"] == 0.5
    naive_mass_balance = 2 * np.sqrt(24 * 48) / (24 + 48)
    assert naive_mass_balance != r["pooled_balance"]
    three = m._synthetic([(0, 24, 0), (0, 0, 24), (0, 0, 48)], edges=((0, 1), (0, 2)))
    three = replace(three, node_xy=np.array([[0, 0], [8, 0], [16, 0]], np.float32))
    reverse = replace(three, adjacency=tuple(tuple(reversed(edges)) for edges in three.adjacency))
    assert m.seal(m.replay((three,))) == m.seal(m.replay((reverse,)))
    sb = np.float64(0)
    sc = np.float64(0)
    for donor in range(3):
        sb += np.float64(three.supports[donor, 1]) / np.float64(three.node_area[donor])
        sc += np.float64(three.supports[donor, 2]) / np.float64(three.node_area[donor])
    sb, sc = sb / np.float64(3), sc / np.float64(3)
    assert row(m.replay((three,)))["pooled_balance"] == float(2 * np.sqrt(sb * sc) / max(sb + sc, 1e-6))


def test_no_recursive_pooling():
    frame = m._synthetic([(0, 24, 0), (0, 24, 0), (0, 0, 24)], edges=((0, 1), (1, 2)))
    frame = replace(frame, node_xy=np.array([[0, 0], [8, 0], [16, 0]], np.float32))
    report = m.replay((frame,))
    assert row(report, 0)["donors"] == [0, 1]
    assert row(report, 0)["score"] == 0
    assert row(report, 1)["donors"] == [0, 1, 2]
    assert row(report, 1)["seed"]


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_axis_sign_permutation(permutation, sign):
    frame = split()
    permuted = replace(frame, supports=frame.supports[:, permutation].copy(),
        plane_scores=frame.plane_scores[:, permutation].copy(), sides=(frame.sides[:, permutation] * sign).copy())
    old, new = m.replay((frame,)), m.replay((permuted,))
    for node in range(2):
        for axis in range(3):
            left, right = row(old, node, permutation[axis]), row(new, node, axis)
            for key in left:
                if key not in ("axis", "side"):
                    assert left[key] == right[key]
            assert right["side"] == left["side"] * sign


@pytest.mark.parametrize("field,threshold,reason", [(1, 48.0, "gradient"), (2, 40.0, "lab"), (3, 28.0, "texture")])
def test_each_strict_boundary_and_next_representable(field, threshold, reason):
    frame = split()
    below = float(np.nextafter(np.float64(threshold), -np.inf))
    assert row(m.replay((change_edge(frame, field, below),)))["seed"]
    rejected = row(m.replay((change_edge(frame, field, threshold),)))
    assert rejected["rejected"] == [[1, reason]] and not rejected["seed"]
    above = float(np.nextafter(np.float64(threshold), np.inf))
    assert row(m.replay((change_edge(frame, field, above),)))["rejected"] == [[1, reason]]


@pytest.mark.parametrize("condition,reason", [("distance", "distance"), ("side", "side"), ("zero", "zero_side")])
def test_other_individual_barriers(condition, reason):
    frame = split()
    if condition == "distance":
        xy = np.array([[0, 0], [32, 0]], np.float64)
        assert row(m.replay((replace(frame, node_xy=xy),)))["seed"]
        xy[1, 0] = np.nextafter(32.0, np.inf)
        frame = replace(frame, node_xy=xy)
    else:
        sides = frame.sides.copy()
        sides[1, 0] = -1 if condition == "side" else 0
        frame = replace(frame, sides=sides)
    assert row(m.replay((frame,)))["rejected"] == [[1, reason]]
    assert not row(m.replay((frame,)))["seed"]


@pytest.mark.parametrize("endpoint,reason", [(0, "target_competing"), (1, "donor_competing")])
def test_original_endpoint_competing_barrier(endpoint, reason):
    frame = split()
    support = frame.supports.copy()
    support[endpoint] = (24, 24, 0)
    frame = replace(frame, supports=support, plane_scores=m.source_scores(support))
    assert row(m.replay((frame,)))["rejected"] == [[1, reason]]
    # An earlier target boost cannot alter eligibility of an edge for a later node.
    assert frame.plane_scores.tobytes() == m.source_scores(support).tobytes()


def test_python_float_competing_distinct_from_float32_seed_margin():
    scores = np.zeros((1, 3), np.float32)
    scores[0, 0] = np.float32(0.26)
    endpoint = np.float32(float(scores[0, 0]) + 0.035)
    for value in [np.nextafter(endpoint, -np.inf, dtype=np.float32), endpoint,
                  np.nextafter(endpoint, np.inf, dtype=np.float32)]:
        scores[0, 1] = value
        expected = float(value) >= 0.26 and float(value) >= float(scores[0, 0]) + 0.035
        assert m.competing(scores, 0, 0) == expected
    scores[0] = (0.2249, np.float32(0.26), 0)
    assert not m.competing(scores, 0, 0)
    assert scores[0, 1] >= 0.26 and np.float32(scores[0, 1] - scores[0, 0]) >= np.float32(0.035)
    scores[0] = (0, np.nextafter(np.float32(0.26), -np.inf), 0)
    assert not m.competing(scores, 0, 0)
    scores[0, 1] = np.float32(0.26)
    assert m.competing(scores, 0, 0) == (float(np.float32(0.26)) >= 0.26)


@pytest.mark.parametrize("threshold", [0.26, 0.035])
def test_seed_float32_next_representable_threshold(threshold):
    middle = np.float32(threshold)
    values = np.array([np.nextafter(middle, -np.inf), middle, np.nextafter(middle, np.inf)], np.float32)
    opponent = np.zeros(3, np.float32) if threshold == 0.26 else np.ones(3, np.float32) * np.float32(0.3)
    score = values if threshold == 0.26 else opponent + values
    margin, actual = m._seed(score, opponent, np.ones(3, np.float32) * 12, np.ones(3, np.int8))
    expected = (score >= 0.26) & ((score - opponent) >= 0.035)
    assert actual.tobytes() == expected.tobytes()
    assert margin.tobytes() == (score - opponent).tobytes()


@pytest.mark.parametrize("field", ["supports", "plane_scores", "node_lab", "node_texture", "node_xy"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_each_nonfinite_array(field, invalid):
    frame = split()
    array = getattr(frame, field).copy()
    array.flat[0] = invalid
    with pytest.raises(m.Refusal, match="nonfinite"):
        m.replay((replace(frame, **{field: array}),))


@pytest.mark.parametrize("area", [0, -1])
def test_bad_area(area):
    frame = split(areas=[area, 256])
    with pytest.raises(m.Refusal, match="area"):
        m.replay((frame,))


@pytest.mark.parametrize("ids", [[0, 0], [0, 2], [-1, 1], [1, 0]])
def test_invalid_node_ids(ids):
    frame = replace(split(), node_ids=np.array(ids, np.int32))
    with pytest.raises(m.Refusal, match="node IDs"):
        m.replay((frame,))


@pytest.mark.parametrize("bad", ["asymmetric", "duplicate", "range", "self", "row_length", "row_type", "nan", "negative", "bad_id"])
def test_malformed_adjacency(bad):
    frame = split()
    first, second = frame.adjacency
    if bad == "asymmetric":
        adjacency = (first, ())
    elif bad == "duplicate":
        adjacency = (first + first, second)
    elif bad == "range":
        adjacency = (((2, 0.0, 0.0, 0.0, 1.0, 0.0),), second)
    elif bad == "self":
        adjacency = (((0, 0.0, 0.0, 0.0, 1.0, 0.0),), second)
    elif bad == "row_length":
        adjacency = ((first[0][:-1],), second)
    elif bad == "row_type":
        adjacency = ([first[0]], second)
    elif bad == "bad_id":
        adjacency = (((True, 0.0, 0.0, 0.0, 1.0, 0.0),), second)
    else:
        adjacency = change_edge(frame, 1, float("nan") if bad == "nan" else -1.0).adjacency
    with pytest.raises(m.Refusal):
        m.replay((replace(frame, adjacency=adjacency),))


@pytest.mark.parametrize("bad", ["float16", "integer_support", "score_dtype", "shape", "byte_swapped", "negative_support", "extreme_side"])
def test_bad_shapes_types_ranges(bad):
    frame = split()
    if bad in ("float16", "integer_support", "byte_swapped"):
        dtype = {"float16": np.float16, "integer_support": np.int32, "byte_swapped": ">f4"}[bad]
        frame = replace(frame, supports=frame.supports.astype(dtype))
    elif bad == "score_dtype":
        frame = replace(frame, plane_scores=frame.plane_scores.astype(np.float64))
    elif bad == "shape":
        frame = replace(frame, node_lab=np.zeros((2, 2), np.float32))
    elif bad == "extreme_side":
        frame = replace(frame, sides=np.full((2, 3), -(2**63), np.int64))
    else:
        support = frame.supports.copy()
        support[0, 0] = -1
        frame = replace(frame, supports=support)
    with pytest.raises(m.Refusal):
        m.replay((frame,))


def test_mismatch_baseline_byte_and_arithmetic_overflow():
    frame = split()
    scores = frame.plane_scores.copy()
    scores[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(m.Refusal, match="byte mismatch"):
        m.replay((replace(frame, plane_scores=scores),))
    support = np.full((2, 3), np.finfo(np.float32).max, np.float32)
    with pytest.raises(m.Refusal, match="arithmetic"):
        m.replay((replace(frame, supports=support),))
    xy = np.array([[1e308, 0], [-1e308, 0]], np.float64)
    with pytest.raises(m.Refusal, match="arithmetic"):
        m.replay((replace(frame, node_xy=xy),))


@pytest.mark.parametrize("key", ["frames", "atoms_per_frame", "atoms", "edges", "visits", "retained_bytes", "output_bytes"])
def test_exact_and_one_under_independent_caps(key):
    frame = split()
    reference = m.replay((frame,))
    count = 2 if key == "atoms_per_frame" else reference["resources"][key]
    # Output length changes when lowered limits are represented, determine its new fixed point.
    if key == "output_bytes":
        high = count
        for _ in range(5):
            report = m.replay((frame,), limits={key: high})
            if report["resources"][key] == high:
                break
            high = report["resources"][key]
        count = high
    result = m.replay((frame,), limits={key: count})
    assert len(result["frames"][0]["rows"]) == 6
    with pytest.raises(m.Refusal, match="cap"):
        m.replay((frame,), limits={key: count - 1})


def test_charge_before_retention_and_work_sticky_integer_bounds(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source allocation reached after exhausted cap")
    monkeypatch.setattr(m, "source_scores", forbidden)
    with pytest.raises(m.Refusal, match="retained_bytes"):
        m.replay((split_frame,), limits={"retained_bytes": 0})
    budget = m.Budget({"visits": 1})
    budget.charge("visits", 1)
    with pytest.raises(m.Refusal):
        budget.charge("visits", 1)
    assert budget.used["visits"] == 1
    with pytest.raises(m.Refusal, match="sticky"):
        budget.charge("frames", 0)
    for value in (True, -1, 2**63, 1.0):
        with pytest.raises(m.Refusal):
            m.Budget({"atoms": value})
    with pytest.raises(m.Refusal):
        m.Budget({"atoms": m.CAPS["atoms"] + 1})
    bounded = m.Budget({"counter_max": 1})
    with pytest.raises(m.Refusal, match="counter|cap"):
        bounded.charge("visits", 2)


split_frame = split()


@pytest.mark.parametrize("key,over", [("frames", 33), ("atoms_per_frame", 20001), ("atoms", 32001)])
def test_hard_caps_without_oversized_retention(key, over):
    if key == "frames":
        frames = (split_frame,) * over
    elif key == "atoms_per_frame":
        frames = (replace(split_frame, supports=np.zeros((over, 3), np.float32)),)
    else:
        # No retention occurs in the second frame because global count checks precede arrays.
        frames = (replace(split_frame, supports=np.zeros((16001, 3), np.float32)),) * 2
    with pytest.raises(m.Refusal):
        m.replay(frames)


def test_canonical_bounded_serialization_and_truth_tamper():
    sealed = m.seal(m.replay((split_frame,)))
    assert m.seal(json.loads(sealed.payload)) == sealed
    with pytest.raises(m.Refusal):
        m.seal(json.loads(sealed.payload), output_limit=len(sealed.payload) - 1)
    for value in (float("nan"), np.float32(1), 2**63, "x" * 257, (1,)):
        with pytest.raises(m.Refusal):
            m.seal({"bad": value})
    truth = (m.Truth((0, 0), ((1, 0, 0), (1, 0, 0)), True),)
    for invalid in (replace(sealed, sha256="0" * 64), replace(sealed, payload=sealed.payload + b" ")):
        with pytest.raises(m.Refusal):
            m.evaluate(invalid, truth)
    with pytest.raises(m.Refusal):
        m.evaluate(sealed, ())
    with pytest.raises(m.Refusal, match="canonical work"):
        m.seal([[0] * 1024] * 256)


def test_publication_create_new_readback_and_collision(tmp_path):
    sealed = m.seal(m.replay((split_frame,)))
    target = tmp_path / "synthetic.json"
    m.publish(sealed, target)
    assert target.read_bytes() == sealed.payload
    assert list(tmp_path.iterdir()) == [target]
    with pytest.raises(m.Refusal):
        m.publish(sealed, target)
    assert target.read_bytes() == sealed.payload


@pytest.mark.parametrize("failure", ["fsync", "link", "readback", "short_write", "race"])
def test_atomic_publication_failure_cleanup(tmp_path, monkeypatch, failure):
    sealed = m.seal(m.replay((split_frame,)))
    target = tmp_path / "synthetic.json"
    if failure in ("fsync", "link"):
        def fail(*args):
            raise OSError("injected")
        monkeypatch.setattr(m.os, failure, fail)
    elif failure == "readback":
        real = m._readback
        def bad_readback(path, payload):
            if Path(path) == target:
                raise m.Refusal("injected")
            real(path, payload)
        monkeypatch.setattr(m, "_readback", bad_readback)
    elif failure == "race":
        real = m.os.link
        def racing_link(source, destination):
            Path(destination).write_bytes(b"racing output")
            real(source, destination)
        monkeypatch.setattr(m.os, "link", racing_link)
    else:
        real = m.os.fdopen
        class ShortWriter:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            def write(self, value):
                return self.handle.write(value[:-1])
        monkeypatch.setattr(m.os, "fdopen", lambda *args: ShortWriter(real(*args)))
    with pytest.raises(m.Refusal):
        m.publish(sealed, target)
    if failure == "race":
        assert target.read_bytes() == b"racing output"
        assert list(tmp_path.iterdir()) == [target]
    else:
        assert list(tmp_path.iterdir()) == []


def test_oversize_publication_refuses_before_tempfile(tmp_path, monkeypatch):
    monkeypatch.setattr(m.tempfile, "mkstemp", lambda **kwargs: pytest.fail("retention before size guard"))
    payload = b"x" * (m.CAPS["output_bytes"] + 1)
    with pytest.raises(m.Refusal):
        m.publish(m.Sealed(payload, hashlib.sha256(payload).hexdigest()), tmp_path / "bad.json")


def test_published_readback_race_preserves_other_output(tmp_path, monkeypatch):
    sealed = m.seal(m.replay((split_frame,)))
    target = tmp_path / "synthetic.json"
    real = m._readback
    def raced_readback(path, payload):
        if Path(path) == target:
            target.unlink()
            target.write_bytes(b"replacement")
        real(path, payload)
    monkeypatch.setattr(m, "_readback", raced_readback)
    with pytest.raises(m.Refusal):
        m.publish(sealed, target)
    assert target.read_bytes() == b"replacement"
    assert list(tmp_path.iterdir()) == [target]


def test_no_production_execution_image_loading_or_metadata_inputs():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == m.SOURCE_SHA256
    text = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("mc_photogrammetry" in ast.unparse(n) for n in imports)
    assert not any(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and any(word in ast.unparse(n.func) for word in ("_build_graph", "imread", "Image.open", "inference")))
    assert set(inspect.signature(m.replay).parameters) == {"frames", "atom_spacing", "limits"}
    assert "truth" not in m.Frame.__dataclass_fields__
    assert not any(name.startswith("mc_photogrammetry.reconstruction.region_growth") for name in sys.modules)
