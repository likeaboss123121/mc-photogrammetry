"""Public-output controls for the DS-19 oracle-pose diagnostic.

The 2026-09-12 13:16 CDT correction freezes DS-16B's reference yaw-class and
gauge-aligned baseline-error evaluator: the diagnostic may not redefine a
"correct" pose or add an accepted alternate label. These controls pin that
boundary on synthetic geometry, without rendering, manifests or the frozen
artifacts:

* ``evaluate_offline`` must hand the adjudicated pose to the *frozen*
  ``diagnose_match_attribution._evaluate_pose`` and reproduce it verbatim ---
  the regression that made every 9/9 pose read ``within_5deg=False`` while the
  same pose read small under the production conventions;
* a correct yaw class with the exact reference rotation and baseline is usable,
  a wrong class label is not silently rescored, and a class-error pose stays
  unusable;
* the frozen 90-degree about-``y`` gauge search is discovery, not an accepted
  alternate baseline, so ``baseline_error_degrees_signless`` keeps reporting the
  raw disagreement.
"""
import math
import sys
from pathlib import Path

import numpy as np

from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_rotation_matrix,
    intrinsics_for,
    project_points,
)

# The DS-16/DS-19 evaluator lives in the diagnostic scripts tree, which is not an
# installed package; the DS-19 scoped test is the only place it is imported.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import diagnose_match_attribution as attribution  # noqa: E402
import diagnose_oracle_pose_failures as diagnostic  # noqa: E402

WIDTH, HEIGHT = 1280, 720
LEFT_POSITION = (0.0, 0.0, 0.0)
RIGHT_POSITION = (3.0, 0.0, 1.0)
LEFT_YAW, LEFT_PITCH = 3.0, 1.0
RIGHT_YAW, RIGHT_PITCH = 93.0, -2.0
RIGHT_YAW_CLASS = 1


def _model(position, yaw, pitch):
    return CameraModel(
        position=tuple(float(value) for value in position),
        yaw_degrees=float(yaw),
        pitch_degrees=float(pitch),
        roll_degrees=0.0,
        eye_height=1.62,
    )


def _estimate(model, yaw_degrees):
    intrinsics = intrinsics_for(model, WIDTH, HEIGHT)
    return CameraEstimate(
        intrinsics=intrinsics,
        pose=CameraPose(
            position=tuple(float(value + 1.62) if index == 1 else float(value)
                           for index, value in enumerate(model.position)),
            yaw_degrees=float(yaw_degrees),
            pitch_degrees=float(model.pitch_degrees),
            roll_degrees=0.0,
        ),
    )


def _scene():
    """One offset pair with an exactly known right yaw class of 1.

    The left estimate reports the left model's yaw; the right estimate is built
    so that ``right_est_yaw mod 90 == right_model_yaw mod 90 + 1`` in quadrant
    terms, i.e. the right view is the left rotated by one quadrant.
    """
    left_model = _model(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_model = _model(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_estimate = _estimate(left_model, LEFT_YAW)
    right_estimate = _estimate(right_model, RIGHT_YAW - 90.0 * RIGHT_YAW_CLASS)

    rng = np.random.default_rng(19)
    points = rng.uniform(-8.0, 8.0, (240, 3)) + np.asarray((0.0, 0.0, 30.0))
    left_pixels = project_points(
        [tuple(point) for point in points], left_model, width=WIDTH, height=HEIGHT
    )
    right_pixels = project_points(
        [tuple(point) for point in points], right_model, width=WIDTH, height=HEIGHT
    )
    rows = [
        (float(lp[0]), float(lp[1]), float(rp[0]), float(rp[1]))
        for lp, rp in zip(left_pixels, right_pixels)
        if lp is not None and rp is not None
    ]
    matches = np.asarray(rows, dtype=np.float64)
    return left_model, right_model, left_estimate, right_estimate, matches


def _ry(step_degrees):
    angle = math.radians(float(step_degrees))
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


class _Pose:
    """Minimal stand-in for :class:`PairPose` with the frozen evaluator's fields."""

    solved = True
    reason = ""
    inliers = 0
    inlier_fraction = 0.0
    median_rho_degrees = 0.0
    parallax_degrees = 0.0
    cheirality_fraction = 1.0

    def __init__(self, yaw_class, relative_rotation, baseline_direction):
        self.right_yaw_class = int(yaw_class)
        self.relative_rotation = np.asarray(relative_rotation, dtype=np.float64)
        self.baseline_direction = tuple(float(value) for value in baseline_direction)


def _true_pair():
    left_model, right_model, left_estimate, right_estimate, matches = _scene()
    true_rotation = camera_rotation_matrix(right_model) @ camera_rotation_matrix(left_model).T
    baseline_true = np.asarray(right_model.position, dtype=np.float64) - np.asarray(
        left_model.position, dtype=np.float64
    )
    baseline_true = baseline_true / np.linalg.norm(baseline_true)
    return (
        left_model, right_model, left_estimate, right_estimate, matches,
        true_rotation, baseline_true,
    )


def test_offline_evaluation_is_the_frozen_evaluator_verbatim(monkeypatch):
    """``evaluate_offline`` must surface the frozen evaluator's fields untouched."""
    canned = {
        "solved": True,
        "reason": "",
        "right_yaw_class": RIGHT_YAW_CLASS,
        "class_equals_expected": True,
        "inliers": 7,
        "inlier_fraction": 0.5,
        "median_rho_degrees": 0.25,
        "parallax_degrees": 4.0,
        "cheirality_fraction": 0.9,
        "rotation_error_degrees": 123.5,
        "within_5deg": True,
        "baseline_error_degrees_gauge_aligned": 4.75,
        "baseline_error_degrees_signless": 6.5,
        "gauge_index": 2,
    }
    calls = {}

    def spy(pose, left_model, right_model, expected_class):
        calls["pose"] = pose
        calls["expected"] = expected_class
        return dict(canned)

    monkeypatch.setattr(diagnostic._frozen_evaluator, "_evaluate_pose", spy)
    left_model, right_model, left_estimate, right_estimate, _, _, baseline_true = _true_pair()
    record = {
        "solved": True,
        "reason": "",
        "right_yaw_class": RIGHT_YAW_CLASS,
        "baseline_direction": tuple(baseline_true),
        "inliers": 1,
        "inlier_fraction": 0.5,
        "median_rho_degrees": 0.25,
        "parallax_degrees": 4.0,
        "cheirality_fraction": 0.9,
        "singular_ratio": 0.5,
    }
    result = diagnostic.evaluate_offline(
        record, left_estimate, right_estimate, left_model, right_model, RIGHT_YAW_CLASS
    )
    assert calls["expected"] == RIGHT_YAW_CLASS
    # the pose the offline adjudicator handed over is in production conventions
    assert calls["pose"].right_yaw_class == RIGHT_YAW_CLASS
    assert calls["pose"].baseline_direction == tuple(float(v) for v in baseline_true)
    # Field-for-field passthrough: the frozen numbers are never recomputed.
    for key in (
        "rotation_error_degrees",
        "baseline_error_degrees_gauge_aligned",
        "baseline_error_degrees_signless",
        "gauge_index",
        "within_5deg",
        "median_rho_degrees",
        "parallax_degrees",
        "cheirality_fraction",
    ):
        assert result[key] == canned[key]
    # ``usable`` is the only DS-19 addition, and it is the preregistered
    # conjunction (never a re-derived pose score).
    assert result["usable"] is True
    assert result["class_equals_expected"] is True


def test_offline_evaluation_is_not_reimplemented_in_the_diagnostic():
    """No second copy of the rotation/baseline metric may live in the script.

    The frozen evaluator is imported; re-deriving the angle here is exactly the
    regression the correction forbids, so the module source must not contain the
    old model-frame rotation construction.
    """
    source = (diagnostic.__file__ and open(diagnostic.__file__, encoding="utf-8").read()) or ""
    assert "_frozen_evaluator._evaluate_pose" in source
    assert "camera_rotation_matrix(right_model)\n        @ _ry90(" not in source
    # the frozen module is the declared source of the metric
    assert diagnostic._frozen_evaluator._evaluate_pose is attribution._evaluate_pose


def test_correct_class_and_pose_is_usable_under_the_frozen_evaluator():
    """A correct class with the exact reference relative rotation and baseline."""
    (left_model, right_model, left_estimate, right_estimate, _matches,
     true_rotation, baseline_true) = _true_pair()
    pose = _Pose(RIGHT_YAW_CLASS, true_rotation, baseline_true)
    frozen = attribution._evaluate_pose(pose, left_model, right_model, RIGHT_YAW_CLASS)
    assert frozen["class_equals_expected"] is True
    assert frozen["rotation_error_degrees"] <= 5.0
    assert frozen["baseline_error_degrees_gauge_aligned"] <= 5.0

    record = {
        "solved": True,
        "reason": "",
        "right_yaw_class": RIGHT_YAW_CLASS,
        "baseline_direction": tuple(baseline_true),
        "inliers": 1,
        "inlier_fraction": 0.5,
        "median_rho_degrees": 0.25,
        "parallax_degrees": 4.0,
        "cheirality_fraction": 0.9,
        "singular_ratio": 0.5,
    }
    offline = diagnostic.evaluate_offline(
        record, left_estimate, right_estimate, left_model, right_model, RIGHT_YAW_CLASS
    )
    # The offline hand-over must reproduce the frozen read-out, not exceed it.
    assert offline["usable"] is True
    assert offline["within_5deg"] == frozen["within_5deg"]
    assert (
        offline["baseline_error_degrees_gauge_aligned"]
        == frozen["baseline_error_degrees_gauge_aligned"]
    )


def test_reference_gauge_rotation_is_discovery_not_an_alternate_answer():
    """A nominal gauge rotation must not be reported as a corrected baseline."""
    (left_model, right_model, _left_estimate, _right_estimate, _matches,
     true_rotation, baseline_true) = _true_pair()
    good = _Pose(RIGHT_YAW_CLASS, true_rotation, baseline_true)
    shifted = _Pose(RIGHT_YAW_CLASS, true_rotation, _ry(90) @ baseline_true)
    good_record = attribution._evaluate_pose(good, left_model, right_model, RIGHT_YAW_CLASS)
    shifted_record = attribution._evaluate_pose(shifted, left_model, right_model, RIGHT_YAW_CLASS)
    # The frozen evaluator's 4-way gauge search bounds the gauge-aligned error...
    assert shifted_record["baseline_error_degrees_gauge_aligned"] <= 5.0
    # ...but the raw disagreement is still reported, and it is large.
    assert shifted_record["baseline_error_degrees_signless"] > 45.0
    assert shifted_record["gauge_index"] != good_record["gauge_index"]


def test_wrong_baseline_is_not_repaired_by_the_gauge_search():
    """A genuinely wrong direction stays wrong through all four gauge rotations.

    Every gauge candidate is a rotation about world ``y``, so world up is a full
    90 degrees from all of them on this scene and cannot be brought into
    agreement by the search.
    """
    (left_model, right_model, _left_estimate, _right_estimate, _matches,
     true_rotation, baseline_true) = _true_pair()
    assert abs(float(np.dot(baseline_true, (0.0, 1.0, 0.0)))) < 1e-9
    record = attribution._evaluate_pose(
        _Pose(RIGHT_YAW_CLASS, true_rotation, (0.0, 1.0, 0.0)),
        left_model, right_model, RIGHT_YAW_CLASS,
    )
    assert record["baseline_error_degrees_gauge_aligned"] > 45.0
    assert record["baseline_error_degrees_signless"] > 45.0


def test_wrong_yaw_class_is_never_rescored():
    """A wrong class label stays wrong; the evaluator does not infer the class."""
    (left_model, right_model, left_estimate, right_estimate, _matches,
     true_rotation, baseline_true) = _true_pair()
    wrong = _Pose((RIGHT_YAW_CLASS + 2) % 4, true_rotation, baseline_true)
    frozen = attribution._evaluate_pose(wrong, left_model, right_model, RIGHT_YAW_CLASS)
    assert frozen["class_equals_expected"] is False
    record = {
        "solved": True,
        "reason": "",
        "right_yaw_class": (RIGHT_YAW_CLASS + 2) % 4,
        "baseline_direction": tuple(baseline_true),
        "inliers": 1,
        "inlier_fraction": 0.5,
        "median_rho_degrees": 0.25,
        "parallax_degrees": 4.0,
        "cheirality_fraction": 0.9,
        "singular_ratio": 0.5,
    }
    offline = diagnostic.evaluate_offline(
        record, left_estimate, right_estimate, left_model, right_model, RIGHT_YAW_CLASS
    )
    assert offline["class_equals_expected"] is False
    assert offline["usable"] is False


def test_unsolved_adjudication_is_unusable_without_a_pose():
    result = diagnostic.evaluate_offline(
        {
            "solved": False,
            "reason": "no_yaw_class_passes_gates",
            "right_yaw_class": None,
            "baseline_direction": None,
            "inliers": 0,
            "inlier_fraction": 0.0,
            "median_rho_degrees": None,
            "parallax_degrees": None,
            "cheirality_fraction": None,
            "singular_ratio": None,
        },
        None, None, None, None, RIGHT_YAW_CLASS,
    )
    assert result["solved"] is False
    assert result["usable"] is False
    assert result["rotation_error_degrees"] is None


def test_adjudicator_recovers_the_known_class_and_a_gauge_comparable_pose():
    """End-to-end on synthetic geometry: the adjudicator's handle is production-shaped.

    The offline adjudicator builds its own Gram/SVD fit; whatever it returns must
    still be read by the frozen evaluator in the same gauge as the production
    ``solve_pair`` pose on the identical correspondences.
    """
    left_model, right_model, left_estimate, right_estimate, matches = _scene()
    geometry = diagnostic.class_rows(left_estimate, right_estimate, matches)
    decision = diagnostic.adjudicate(
        geometry, left_estimate, right_estimate,
        matches[:, 0:2], matches[:, 2:4],
        np.arange(len(matches)), np.arange(len(matches)), WIDTH, HEIGHT,
    )
    assert decision["solved"] is True
    assert decision["right_yaw_class"] == RIGHT_YAW_CLASS

    evaluation = diagnostic.evaluate_offline(
        decision, left_estimate, right_estimate, left_model, right_model, RIGHT_YAW_CLASS
    )
    assert evaluation["usable"] is True
    assert evaluation["rotation_error_degrees"] <= 5.0

    production = solve_pair(
        left_estimate, right_estimate, matches,
        left_id="left", right_id="right", gates=PairGates(),
    )
    assert production.solved is True
    production_record = attribution._evaluate_pose(
        production, left_model, right_model, RIGHT_YAW_CLASS
    )
    # Same class, same gauge: the two handles agree in the frozen metric.
    assert production_record["rotation_error_degrees"] <= 5.0
    assert production_record["class_equals_expected"] is True
    assert abs(
        production_record["baseline_error_degrees_gauge_aligned"]
        - evaluation["baseline_error_degrees_gauge_aligned"]
    ) <= 5.0
