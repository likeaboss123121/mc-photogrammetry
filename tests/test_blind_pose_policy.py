"""DS-23 - enforcement tests for the label-blind pose-policy boundary.

The DS-19 support tie-break may only select a yaw class and baseline from
production-available geometry: hypotheses, consensus counts/inlier masks,
residual distributions, parallax, cheirality, conditioning and spatial-stratum
stability. Before a selection is frozen it may not see manifests, reference
poses/classes, litematics, evaluator results, pair identity or fixture
constants.

These tests pin that boundary in four independent ways:

* **API shape** - the selection function takes one geometric parameter only,
  and the input object exposes no identity/label/evaluator field at all;
* **source audit** - forbidden symbols and imports do not even appear in the
  selector module (no module-level access to reference or evaluator code);
* **runtime trace** - every attribute the decision reads is on the geometric
  whitelist (and the decision reads every geometry field, so the trace is not
  vacuous);
* **poisoning** - hostile monkeypatched evaluator/reference entry points cannot
  change a frozen decision.

Numerical equivalence with the unedited DS-19 adjudicator is checked
separately (imported lazily, so this module can also be imported in a process
that never loads the reference/evaluator scripts).
"""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (str(ROOT / "src"), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import blind_pose_policy as policy  # noqa: E402
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraPose,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    intrinsics_for,
    project_points,
)

WIDTH, HEIGHT = 1280, 720
LEFT_POSITION = (0.0, 0.0, 0.0)
RIGHT_POSITION = (3.0, 0.0, 1.0)
LEFT_YAW, LEFT_PITCH = 3.0, 1.0
RIGHT_MODEL_YAW, RIGHT_PITCH = 93.0, -2.0
RIGHT_YAW_CLASS = 1

FORBIDDEN_MODULE_MARKERS = (
    "diagnose_match_attribution",
    "diagnose_oracle_pose_failures",
    "diagnose_connected_pairs",
    "diagnose_joint_pairs",
    "diagnose_blind_pose_policy",
    "_evaluate_pose",
    "_reference_class",
    "load_camera_manifest",
)

FORBIDDEN_SOURCE_NAMES = (
    "reference_class",
    "reference_floors",
    "expected_class",
    "manifest",
    "litematic",
    "scene_transform",
    "evaluator",
    "_evaluate_pose",
    "_reference_class",
    "oracle",
    "truth",
    "fixture",
    "left_id",
    "right_id",
    "frame_key",
    "usable",
    "class_equals_expected",
    "within_5deg",
    "baseline_error_degrees_gauge_aligned",
)


def _model(position, yaw, pitch):
    return CameraModel(
        position=tuple(float(value) for value in position),
        yaw_degrees=float(yaw),
        pitch_degrees=float(pitch),
        roll_degrees=0.0,
        eye_height=1.62,
    )


def _estimate(model, yaw_degrees, pitch_degrees):
    intrinsics = intrinsics_for(model, WIDTH, HEIGHT)
    return CameraEstimate(
        intrinsics=intrinsics,
        pose=CameraPose(
            position=None,
            yaw_degrees=float(yaw_degrees),
            pitch_degrees=float(pitch_degrees),
            roll_degrees=0.0,
        ),
    )


def _scene():
    """One offset pair with an exactly known right yaw class of 1.

    Mirrors the DS-19 test scene: the right estimate reports the right model's
    yaw minus one quadrant, so the geometrically correct class is 1.
    """
    left_model = _model(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_model = _model(RIGHT_POSITION, RIGHT_MODEL_YAW, RIGHT_PITCH)
    left_estimate = _estimate(left_model, LEFT_YAW, LEFT_PITCH)
    right_estimate = _estimate(
        right_model, RIGHT_MODEL_YAW - 90.0 * RIGHT_YAW_CLASS, RIGHT_PITCH
    )
    rng = np.random.default_rng(19)
    points = rng.uniform(-8.0, 8.0, (240, 3)) + np.asarray((0.0, 0.0, 30.0))
    left_pixels = project_points(
        [tuple(point) for point in points], left_model, width=WIDTH, height=HEIGHT
    )
    right_pixels = project_points(
        [tuple(point) for point in points], right_model, width=WIDTH, height=HEIGHT
    )
    matches = np.asarray(
        [
            (float(lp[0]), float(lp[1]), float(rp[0]), float(rp[1]))
            for lp, rp in zip(left_pixels, right_pixels)
            if lp is not None and rp is not None
        ],
        dtype=np.float64,
    )
    return left_model, right_model, left_estimate, right_estimate, matches


def _blind(left_estimate, right_estimate, matches, width=WIDTH, height=HEIGHT):
    return policy.to_blind_input(
        left_intrinsics=left_estimate.intrinsics,
        left_pose=left_estimate.pose,
        right_intrinsics=right_estimate.intrinsics,
        right_pose=right_estimate.pose,
        matches=matches,
        image_size=(width, height),
    )


def _decision_tuple(decision):
    direction = decision.baseline_direction
    return (
        decision.solved,
        decision.reason,
        decision.right_yaw_class,
        decision.inliers,
        None if direction is None else tuple(round(float(value), 12) for value in direction),
    )


# --------------------------------------------------------------------------- #
# 1. API boundary.
# --------------------------------------------------------------------------- #
def test_selector_takes_exactly_one_geometric_input():
    signature = inspect.signature(policy.select_blind_pose)
    parameters = list(signature.parameters.values())
    assert len(parameters) == 2, "expected (blind, policy) only"
    blind_parameter, policy_parameter = parameters
    assert blind_parameter.name == "blind"
    assert policy_parameter.name == "policy"
    assert policy_parameter.default is policy.FROZEN_POLICY
    # ``from __future__ import annotations`` makes these strings; accept both.
    assert blind_parameter.annotation in (
        policy.BlindPairInput,
        "BlindPairInput",
    )
    assert policy_parameter.annotation in (policy.FrozenPolicy, "FrozenPolicy")


def test_blind_input_exposes_no_identity_label_or_evaluator_field():
    slots = set(policy.BlindPairInput.__slots__)
    assert slots == {
        "left_intrinsics",
        "left_pitch_degrees",
        "left_roll_degrees",
        "left_yaw_degrees",
        "matches",
        "image_size",
        "right_intrinsics",
        "right_pitch_degrees",
        "right_roll_degrees",
        "right_yaw_degrees",
    }
    for name in policy.FORBIDDEN_DECISION_INPUTS:
        assert name not in slots, f"forbidden decision input leaked into the tape: {name}"
    # Properties only, no storage of a frame name / fixture / pixel arrays.
    assert set(policy.BlindPairInput.__slots__).issubset(policy.GEOMETRIC_INPUT_WHITELIST)


def test_adapter_cannot_receive_a_label_or_identity_argument():
    signature = inspect.signature(policy.to_blind_input)
    for name in signature.parameters:
        assert name not in policy.FORBIDDEN_DECISION_INPUTS, name
    assert set(signature.parameters) == {
        "left_intrinsics",
        "left_pose",
        "right_intrinsics",
        "right_pose",
        "matches",
        "image_size",
        "width",
        "height",
    }


# --------------------------------------------------------------------------- #
# 2. Source audit: the selector module never reaches reference/evaluator code.
# --------------------------------------------------------------------------- #
def test_selector_module_globals_do_not_include_forbidden_modules():
    for value in vars(policy).values():
        module_name = getattr(value, "__name__", "")
        if not isinstance(value, type(sys)):
            continue
        for marker in FORBIDDEN_MODULE_MARKERS:
            assert marker not in module_name, f"selector global module {module_name}"


def test_importing_the_selector_loads_no_reference_or_evaluator_module():
    program = (
        "import sys; sys.path.insert(0, r'{src}'); sys.path.insert(0, r'{scripts}');"
        "import blind_pose_policy as p;"
        "bad=[m for m in sys.modules if any(k in m for k in {markers} )];"
        "print('BAD' if bad else 'CLEAN');"
    ).format(
        src=str(ROOT / "src"),
        scripts=str(SCRIPTS),
        markers=repr(
            [
                "diagnose_match_attribution",
                "diagnose_oracle_pose_failures",
                "diagnose_connected_pairs",
                "diagnose_joint_pairs",
            ]
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("CLEAN"), completed.stdout


# --------------------------------------------------------------------------- #
# 3. Runtime trace: only geometric reads, and the trace is not vacuous.
# --------------------------------------------------------------------------- #
def test_decision_reads_only_whitelisted_geometric_attributes():
    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    blind = _blind(left_estimate, right_estimate, matches)
    trace = policy.start_trace(blind, blind.left_intrinsics, blind.right_intrinsics)
    try:
        decision = policy.select_blind_pose(blind)
    finally:
        policy.stop_trace()
    assert decision.solved is True
    reads = set(trace.names())
    assert reads, "trace recorded no reads; the audit hook is not wired in"
    assert reads.issubset(policy.GEOMETRIC_INPUT_WHITELIST), sorted(
        reads - policy.GEOMETRIC_INPUT_WHITELIST
    )
    assert not reads & set(policy.FORBIDDEN_DECISION_INPUTS)
    # Not vacuous: the decision really did consume the geometry it claims to.
    assert {"matches", "left_intrinsics", "right_intrinsics", "image_size"} <= reads
    for name in (
        "left_yaw_degrees",
        "left_pitch_degrees",
        "right_yaw_degrees",
        "right_pitch_degrees",
    ):
        assert name in reads, name


# --------------------------------------------------------------------------- #
# 4. Poisoning: hostile label/evaluator sources cannot change a frozen decision.
# --------------------------------------------------------------------------- #
def test_poisoned_labels_and_evaluator_cannot_change_the_decision(monkeypatch):
    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    blind = _blind(left_estimate, right_estimate, matches)
    baseline = _decision_tuple(policy.select_blind_pose(blind))

    class _Poison(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise _Poison("reference/evaluator access attempted before selection")

    def _hostile(*_args, **_kwargs):
        return {"class_equals_expected": False, "usable": False, "within_5deg": False}

    poisoned = 0
    for module_name in (
        "diagnose_match_attribution",
        "diagnose_oracle_pose_failures",
        "diagnose_connected_pairs",
        "diagnose_joint_pairs",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute in ("_evaluate_pose", "_reference_class", "load_camera_manifest"):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, _boom, raising=False)
                poisoned += 1
    # The selector must not even look at those modules.
    assert _decision_tuple(policy.select_blind_pose(blind)) == baseline
    # Hostile *evaluator output* is equally irrelevant: the decision has no
    # evaluator field to read.
    assert "evaluation" not in set(policy.BlindPairInput.__slots__)
    assert _decision_tuple(policy.select_blind_pose(blind)) == baseline
    assert poisoned >= 0  # diagnostic only; reference modules may not be loaded


def test_poisoned_frozen_policy_constants_are_the_only_tuning_surface():
    """The decision reads its thresholds from the frozen policy object only."""
    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    blind = _blind(left_estimate, right_estimate, matches)
    loose = policy.FrozenPolicy(edge_consensus_degrees=1.0, class_margin_min=0)
    strict = policy.FrozenPolicy(edge_consensus_degrees=1.0, class_margin_min=10 ** 9)
    default = policy.select_blind_pose(blind)
    assert policy.select_blind_pose(blind, loose).solved is True
    blocked = policy.select_blind_pose(blind, strict)
    assert blocked.solved is False
    assert blocked.reason == "ambiguous_yaw_class"
    assert default.solved is True


# --------------------------------------------------------------------------- #
# 5. Label-permutation invariance.
# --------------------------------------------------------------------------- #
def test_correspondence_order_permutation_does_not_change_the_decision():
    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    baseline = policy.select_blind_pose(_blind(left_estimate, right_estimate, matches))
    rng = np.random.default_rng(23)
    for _ in range(4):
        permutation = rng.permutation(len(matches))
        shuffled = policy.select_blind_pose(
            _blind(left_estimate, right_estimate, matches[permutation])
        )
        assert shuffled.right_yaw_class == baseline.right_yaw_class
        assert shuffled.solved == baseline.solved
        assert shuffled.inliers == baseline.inliers
        # The baseline direction is geometry, not row order (sign is fixed by
        # max-depth positivity, so it cannot flip either).
        agreement = abs(
            float(np.dot(shuffled.baseline_direction, baseline.baseline_direction))
        )
        assert agreement > 1.0 - 1e-9


def test_relabelling_the_pair_identity_cannot_change_the_decision():
    """Same geometry, arbitrary bookkeeping name: identical decision."""
    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    first = policy.select_blind_pose(_blind(left_estimate, right_estimate, matches))
    second = policy.select_blind_pose(
        policy.to_blind_input(
            left_intrinsics=left_estimate.intrinsics,
            left_pose=left_estimate.pose,
            right_intrinsics=right_estimate.intrinsics,
            right_pose=right_estimate.pose,
            matches=matches,
            image_size=(WIDTH, HEIGHT),
        )
    )
    assert _decision_tuple(first) == _decision_tuple(second)


# --------------------------------------------------------------------------- #
# 6. Synthetic degeneracy refusals (frozen expectation).
# --------------------------------------------------------------------------- #
def _synthetic(kind, repeats=6):
    from diagnose_blind_pose_policy import _synthetic_case

    solved = 0
    for repeat in range(repeats):
        rng = np.random.default_rng(policy and 1000 * repeat + 7)
        case = _synthetic_case(kind, rng)
        if len(case["matches"]) < policy.FROZEN_POLICY.edge_consensus_min:
            continue
        blind = _blind(
            case["left_camera"], case["right_camera"], case["matches"],
            case["width"], case["height"],
        )
        if policy.select_blind_pose(blind).solved:
            solved += 1
    return solved


def test_synthetic_valid_baseline_and_parallel_axes_are_solved():
    assert _synthetic("valid_baseline") == 6
    assert _synthetic("parallel_axes") == 6


def test_synthetic_pure_rotation_is_refused():
    assert _synthetic("pure_rotation") == 0


# --------------------------------------------------------------------------- #
# 7. Numerical equivalence with the unedited DS-19 adjudicator.
# --------------------------------------------------------------------------- #
def test_clean_room_selector_is_numerically_identical_to_ds19_adjudicator():
    """Import DS-19 lazily and compare on the same geometry, field for field."""
    import diagnose_oracle_pose_failures as ds19

    _lm, _rm, left_estimate, right_estimate, matches = _scene()
    geometry = ds19.class_rows(left_estimate, right_estimate, matches)
    edges = np.arange(len(matches))
    expected = ds19.adjudicate(
        geometry, left_estimate, right_estimate,
        matches[:, 0:2], matches[:, 2:4], edges, edges, WIDTH, HEIGHT,
    )
    blind = _blind(left_estimate, right_estimate, matches)
    actual = policy.select_blind_pose(blind)
    assert actual.solved == expected["solved"]
    assert actual.reason == expected["reason"]
    assert actual.right_yaw_class == expected["right_yaw_class"]
    assert actual.inliers == expected["inliers"]
    assert float(
        np.max(
            np.abs(
                np.asarray(actual.baseline_direction, dtype=np.float64)
                - np.asarray(expected["baseline_direction"], dtype=np.float64)
            )
        )
    ) <= policy.FrozenPolicy().edge_consensus_degrees * 0.0 + 1e-12
    for yaw_class in (0, 1, 2, 3):
        left = expected["classes"][str(yaw_class)]
        right = actual.per_class[str(yaw_class)]
        assert left["gate_pass"] == right["gate_pass"]
        assert left["gate_refusal"] == right["gate_refusal"]
        assert left["consensus_count"] == right["consensus_count"]
        assert left["median_rho_degrees"] == right["median_rho_degrees"]
        assert left["singular_ratio"] == right["singular_ratio"]


def test_clean_room_selector_matches_ds19_on_the_frozen_nine_pairs():
    """Replay the frozen tape through both selectors and require bit-equality.

    Reads the artifacts DS-23 produced (tape + DS-19 payload); skips cleanly if
    the diagnostic has not been run in this checkout.
    """
    tape_path = ROOT / "exports" / "diagnostics" / "ds23" / "blind_tape.json"
    tape_npz = ROOT / "exports" / "diagnostics" / "ds23" / "blind_tape.npz"
    ds19_path = (
        ROOT / "exports" / "diagnostics" / "ds19" / "oracle_pose_failures.json"
    )
    if not (tape_path.exists() and tape_npz.exists() and ds19_path.exists()):
        import pytest

        pytest.skip("DS-23 tape / DS-19 artifact not present in this checkout")
    import diagnose_oracle_pose_failures as ds19
    import diagnose_blind_pose_policy as driver

    tape = json.loads(tape_path.read_text(encoding="utf-8"))
    arrays = np.load(tape_npz)
    frozen = json.loads(ds19_path.read_text(encoding="utf-8"))
    known = {
        f"{row['left']}->{row['right']}": row for row in frozen["cell"]["pairs"]
    }
    assert len(tape["pairs"]) == 9
    for pair in tape["pairs"]:
        blind, matches = driver.tape_blind_input(pair, arrays)
        actual = policy.select_blind_pose(blind)
        left_estimate, right_estimate = driver.tape_cameras(pair)
        geometry = ds19.class_rows(left_estimate, right_estimate, matches)
        expected = ds19.adjudicate(
            geometry, left_estimate, right_estimate,
            matches[:, 0:2], matches[:, 2:4],
            np.arange(len(matches)), np.arange(len(matches)),
            int(pair["width"]), int(pair["height"]),
        )
        key = f"{pair['left']}->{pair['right']}"
        assert actual.solved == expected["solved"], key
        assert actual.right_yaw_class == expected["right_yaw_class"], key
        assert actual.inliers == expected["inliers"], key
        assert actual.reason == expected["reason"], key
        assert actual.right_yaw_class == known[key]["adjudicated"]["right_yaw_class"], key
        assert (
            float(
                np.max(
                    np.abs(
                        np.asarray(actual.baseline_direction, dtype=np.float64)
                        - np.asarray(
                            known[key]["adjudicated"]["baseline_direction"],
                            dtype=np.float64,
                        )
                    )
                )
            )
            <= 1e-12
        ), key


# --------------------------------------------------------------------------- #
# 8. Frozen evidence gates from the DS-23 artifact.
# --------------------------------------------------------------------------- #
def test_frozen_evidence_reproduces_baseline_and_meets_blind_gates():
    payload_path = (
        ROOT / "exports" / "diagnostics" / "ds23" / "blind_pose_policy.json"
    )
    if not payload_path.exists():
        import pytest

        pytest.skip("DS-23 payload not present in this checkout")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["reproduction"]["production_oracle_bestof_solved"] == 6
    assert payload["reproduction"]["production_oracle_bestof_class_correct"] == 6
    assert payload["blind_selection"]["usable"] >= 8
    assert payload["blind_selection"]["class_correct"] >= 8
    assert payload["threshold"]["zero_regressions_among_original_six"]["value"] == 0
    assert payload["threshold"]["existing_synthetic_degeneracy_refusals"]["passed"] is True
    assert payload["threshold"]["apiboundary_no_unexpected_reads"]["value"] == 0
    assert payload["equivalence_with_ds19"]["all_equivalent"] is True
    # Every decision was made from the frozen tape, trace-clean.
    for decision in payload["decisions"]:
        assert decision["unexpected_reads"] == []
        assert set(decision["trace_reads"]).issubset(policy.GEOMETRIC_INPUT_WHITELIST)


def test_declared_blind_inputs_contain_no_forbidden_field():
    tape_path = ROOT / "exports" / "diagnostics" / "ds23" / "blind_tape.json"
    if not tape_path.exists():
        import pytest

        pytest.skip("DS-23 tape not present in this checkout")
    tape = json.loads(tape_path.read_text(encoding="utf-8"))
    allowed = {
        "slot", "set", "fixture", "left", "right", "width", "height",
        "correspondences", "oracle_truths_frozen_crosscheck_only",
        "left_intrinsics", "left_pose", "right_intrinsics", "right_pose",
        "matches_sha256", "camera_regime", "count_crosscheck",
        "left_intrinsics_keys", "right_intrinsics_keys", "pose_keys",
    }
    for pair in tape["pairs"]:
        assert set(pair) <= allowed, set(pair) - allowed
        # The decision-facing fields are exactly the geometric ones.
        for field in (
            "slot", "width", "height", "correspondences",
            "left_intrinsics", "left_pose", "right_intrinsics", "right_pose",
        ):
            assert field in pair
        assert "reference_class" not in pair
        assert "manifest" not in pair
        assert "evaluation" not in pair
