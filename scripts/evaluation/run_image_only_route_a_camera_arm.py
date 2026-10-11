"""DS-STAGEWISE-K1: evaluation-only paired Route A camera-object arm.

This sibling runner measures the **oracle-camera sensitivity / upper-bound** of the
Route A facade generator on one frozen frame.  It is described by
``docs/reports/DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md`` section 4.2 and was
authorized by the 2026-10-04 lead entries in ``AGENT_COMMS.md``.

What it does
------------

1. **A0 (image-only, production path).**  Calls the ordinary Route A replay runner
   (:mod:`run_image_only_route_a`) on the pinned frozen frame.  A0 uses only
   screenshot-derived inputs, including the ordinary GUI mask produced from the
   image shape.  No evaluator camera, reference, mask or truth is reachable.
2. **Seal A0.**  Serializes and hashes A0 (camera record, facade/decoded volume
   digests, image/GUI-mask/palette digests, settings, runner/facade code hashes,
   export parity) and writes it to ``a0-seal.json`` *before* any evaluator file is
   opened.  The :class:`ArmOrderingGuard` refuses evaluator access until the seal
   exists.
3. **A1 (reference-assisted, evaluator-only).**  Only after the seal, reads the
   evaluator camera for that frame and rebuilds the facade through the **same**
   ``reconstruct_facade`` function with the identical already-decoded screenshot,
   GUI mask, palette, rows/columns/depth and ``scene_transform=None``.  Only the
   ``CameraModel`` object differs.  The A1 record is labelled
   ``reference-assisted`` and is out of production and promotion.
4. **Freeze both arms, then load truth.**  Both arms are hashed and frozen before
   the visible-reference litematic, frozen mask specification, scene transform and
   evaluator raycast are opened.  The truth load is refused until both arms are
   frozen.
5. **Score both arms.**  Uses the accepted evaluator's
   ``evaluation.best_visible_alignment`` family (integer translation plus lattice
   Y-quarter-turn; no scale fit, no reflection) and the accepted
   ``score_aligned_cells`` G scorer on the same frozen ``V``.  Reports each arm's
   recall/precision/F1, the exact correct/missing/false cell sets (counts plus
   digests) and the exact A0 -> A1 delta.

Non-claims
----------

This is **not** a causal K-versus-geometry attribution and **not** production or
promotion evidence.  The arm swaps the whole camera object (position, scale,
intrinsics, orientation) at once, so any delta is oracle-camera headroom only.
The project's existing 90% visible-geometry-F1 gate is cited descriptively; this
runner creates no threshold and changes no production signature, gate or CLI.

Typical frozen-input invocation::

    .\\.venv\\Scripts\\python.exe scripts\\evaluation\\run_image_only_route_a_camera_arm.py ^
        --image "templates 2\\2026-09-02_13.22.14.png" ^
        --fixture "templates 2" ^
        --reference "templates 2\\Template 2.litematic" ^
        --scene "templates 2\\scene.json" --cameras "templates 2\\cameras.json" ^
        --masks docs\\evaluation_masks\\templates-2 ^
        --pins docs\\reports\\evaluator-fixture-pins.json --fixture-role template-2 ^
        --workdir .tmp-k1-route-a-camera-arm --report k1-result.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mc_photogrammetry.evaluation import best_visible_alignment  # noqa: E402
from mc_photogrammetry.evaluation_image_only_attribution import (  # noqa: E402
    MetricSpace,
    align_candidate,
    alignment_record,
    score_aligned_cells,
)
from mc_photogrammetry.evaluation_image_only_frontier import (  # noqa: E402
    canonical_json,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    CameraModel,
    load_camera_manifest,
)
from mc_photogrammetry.reconstruction.facade import reconstruct_facade  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = Path(__file__).resolve()
ROUTE_A_RUNNER_PATH = REPO_ROOT / "scripts" / "evaluation" / "run_image_only_route_a.py"
ATTRIBUTION_RUNNER_PATH = (
    REPO_ROOT / "scripts" / "evaluation" / "report_image_only_attribution.py")

ARM_SCHEMA = "mc-photogrammetry-ds-stagewise-k1-camera-arm-1"
ARM_PRODUCER = "ds-stagewise-k1-route-a-camera-arm-1"
TASK_ID = "DS-STAGEWISE-K1"
IMAGE_ONLY = "image-only"
REFERENCE_ASSISTED = "reference-assisted"
OUT_OF_PROMOTION = True
ORDER_SENTINEL = "a0_serialized_and_sealed_before_evaluator_camera_or_truth_access"
ALIGNMENT_SOURCE = "evaluation.best_visible_alignment"
ALIGNMENT_FAMILY = (
    "integer translation + lattice Y-quarter-turn; no scale fit, no reflection, "
    "no per-cell alignment")
DESCRIPTIVE_F1_GATE = 0.90
# The frozen denominator the lead entry fixes for this frame (A2 route-a truth).
EXPECTED_VISIBLE_CELLS = 398

PREDECLARED_INTERPRETATION = (
    "A large positive A0->A1 delta establishes substantial oracle-camera headroom "
    "on this frame only; it does not establish that image-derived camera estimation "
    "is the sole or dominant cause, because the arm swaps the entire camera bundle "
    "while holding only the facade generator fixed.",
    "A near-zero delta is consistent with the facade engine limiting this frame but "
    "does not prove it dominates: the bundle swap could be absorbed by the alignment "
    "search, and only one frame and one alignment family are measured.",
    "If A1 reaches the frozen visible ceiling under the permitted alignment family, "
    "report the ceiling and stop; do not call camera recovery the sole limiter.",
    "No K-versus-G causal decomposition is claimed in any case; the single-factor "
    "statement is limited to 'the camera object was the only object changed'.",
)

LIMITATIONS = (
    "one frozen frame, one alignment family, one run per arm, no retry",
    "the whole camera object (position/scale/intrinsics/orientation) changes at once, "
    "so orientation/focal/position/scale effects are not separately identified",
    "A1 is reference-assisted and out of production/promotion; it is never image-only "
    "evidence and never a supplied-camera funnel row",
    "Route A's metric scale is unresolved, so each facade score carries that caveat",
    "the 90% visible geometry-F1 gate is quoted descriptively only; no threshold is "
    "created, changed or passed",
)


class ArmRefusal(ValueError):
    """A typed refusal for the paired camera arm."""


class ArmOrderingRefusal(ArmRefusal):
    """An evaluator read was attempted before A0 was sealed / both arms frozen."""


class ArmCameraRefusal(ArmRefusal):
    """The evaluator camera was absent, malformed or not a ``CameraModel``."""


@dataclass(frozen=True, slots=True)
class ArmSettings:
    """The reconstruction settings shared by both arms (no camera fields)."""

    columns: int = 32
    rows: int = 32
    depth_layers: int = 8
    palette_jar: str | None = None

    def __post_init__(self) -> None:
        for name in ("columns", "rows", "depth_layers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ArmRefusal(f"{name} must be a positive integer")

    def to_record(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "depth_layers": self.depth_layers,
            "palette_jar": self.palette_jar,
            "horizon": None,
            "scene_transform": None,
        }


class ArmOrderingGuard:
    """Enforce ``A0 seal < evaluator camera < both arms frozen < truth``.

    The guard is a plain state machine so the ordering contract is testable
    without touching production modules or running a reconstruction.
    """

    __slots__ = ("_sealed", "_frozen")

    def __init__(self) -> None:
        self._sealed = False
        self._frozen = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def frozen(self) -> bool:
        return self._frozen

    def mark_sealed(self) -> None:
        self._sealed = True

    def mark_frozen(self) -> None:
        if not self._sealed:
            raise ArmOrderingRefusal("both arms cannot be frozen before the A0 seal")
        self._frozen = True

    def require_sealed(self, action: str) -> None:
        if not self._sealed:
            raise ArmOrderingRefusal(
                f"{action} was attempted before A0 was serialized and sealed")

    def require_frozen(self, action: str) -> None:
        if not self._sealed:
            raise ArmOrderingRefusal(
                f"{action} was attempted before A0 was serialized and sealed")
        if not self._frozen:
            raise ArmOrderingRefusal(
                f"{action} was attempted before both reconstructions were frozen")


@dataclass(frozen=True, slots=True)
class ReferenceAssistedArm:
    """The A1 record: evaluator-only, labelled and excluded from promotion."""

    arm: str
    provenance: str
    out_of_promotion: bool
    evaluator_camera: Mapping[str, Any]
    evaluator_camera_sha256: str
    facade_block_count: int
    facade_volume_sha256: str

    def __post_init__(self) -> None:
        if self.arm != "A1":
            raise ArmRefusal("a reference-assisted record belongs to arm A1 only")
        if self.provenance != REFERENCE_ASSISTED:
            raise ArmRefusal(
                "an A1 camera arm record must be labelled reference-assisted")
        if self.out_of_promotion is not True:
            raise ArmRefusal("an A1 camera arm record must be out of promotion")
        if not isinstance(self.evaluator_camera, Mapping):
            raise ArmCameraRefusal("the A1 record needs the evaluator camera record")
        if type(self.facade_block_count) is not int or self.facade_block_count < 0:
            raise ArmRefusal("the A1 facade block count must be a non-negative integer")

    def to_record(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "provenance": self.provenance,
            "out_of_promotion": self.out_of_promotion,
            "evaluator_camera": dict(self.evaluator_camera),
            "evaluator_camera_sha256": self.evaluator_camera_sha256,
            "facade_block_count": self.facade_block_count,
            "facade_volume_sha256": self.facade_volume_sha256,
            "role": "evaluator-only oracle-camera sensitivity arm",
        }


@dataclass(frozen=True, slots=True)
class ArmHooks:
    """Injection seam for the focused synthetic tests (no fixture data needed)."""

    run_a0: Callable[..., Any]
    load_evaluator_camera: Callable[..., Any]
    reconstruct: Callable[..., Any]
    load_truth: Callable[..., Mapping[str, Any]]
    record: Callable[[str, Mapping[str, Any]], None]


def _noop_record(event: str, detail: Mapping[str, Any]) -> None:
    return None


# ---------------------------------------------------------------------------
# Digests and small converters
# ---------------------------------------------------------------------------

def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _digest_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _mask_domain_sha256(mask: Any) -> str:
    """The accepted scored boolean-domain digest (shape + bytes), not file bytes."""
    array = np.asarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _states_of(volume: Any) -> dict:
    if volume is None:
        return {}
    return {coordinate: observation.state
            for coordinate, observation in dict(volume.blocks).items()}


def _palette_states(palette: Any) -> tuple:
    return tuple(item.state if hasattr(item, "state") else item for item in palette)


def _palette_record(palette: Any) -> list:
    return [state.to_dict() for state in _palette_states(palette)]


def _camera_record(camera: Any) -> dict | None:
    if camera is None:
        return None
    if not isinstance(camera, CameraModel):
        raise ArmCameraRefusal("a camera record needs a CameraModel or None")
    return camera.to_dict()


def _volume_digest(states: Mapping[Any, Any]) -> str:
    payload = [[list(coordinate), state.to_dict()]
               for coordinate, state in sorted(states.items())]
    return _digest_json(payload)


def _cell_set_digest(cells: Any) -> str:
    return _digest_json(sorted([list(coordinate) for coordinate in cells]))


def _translation_offset(left: Mapping, right: Mapping) -> tuple[int, int, int]:
    if not left or not right:
        return (0, 0, 0)
    left_min = tuple(min(coordinate[axis] for coordinate in left)
                     for axis in range(3))
    right_min = tuple(min(coordinate[axis] for coordinate in right)
                      for axis in range(3))
    return tuple(left_min[axis] - right_min[axis] for axis in range(3))


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArmRefusal(f"the evaluation module is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Default hooks (the real single-run path)
# ---------------------------------------------------------------------------

def _default_run_a0(*, image_path: Path, workdir: Path, settings: ArmSettings) -> Any:
    """A0: the ordinary image-only Route A replay, production inputs only."""
    module = _load_module("_k1_route_a_replay", ROUTE_A_RUNNER_PATH)
    return module.replay(
        image_path, workdir=workdir, columns=settings.columns, rows=settings.rows,
        depth_layers=settings.depth_layers, palette_jar=settings.palette_jar)


def _default_load_evaluator_camera(*, cameras_path: Path, frame_id: str) -> CameraModel:
    """The evaluator camera for one frame, typed refusal on anything unusable."""
    try:
        manifest = load_camera_manifest(cameras_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArmCameraRefusal(
            f"the evaluator camera manifest is unusable: {error}") from error
    if frame_id not in manifest:
        raise ArmCameraRefusal(
            f"the evaluator camera manifest has no entry for {frame_id}")
    camera = manifest[frame_id]
    if not isinstance(camera, CameraModel):
        raise ArmCameraRefusal(
            f"the evaluator camera entry for {frame_id} is not a CameraModel")
    return camera


def _default_load_truth(*, fixture: Path, reference: Path, scene: Path,
                        cameras: Path, masks: Path, pins: Path,
                        fixture_role: str, frame_paths: Mapping[str, Path]) -> Mapping:
    """Pins + evaluator truth, read only after both arms are frozen."""
    if pins is None or fixture_role is None:
        raise ArmRefusal(
            "the paired camera arm requires reviewed evaluator fixture pins")
    runner = _load_module("_k1_attribution_runner", ATTRIBUTION_RUNNER_PATH)
    pin_entry = runner.load_pins(pins, fixture_role)
    return runner.load_frozen_truth(
        fixture=fixture, reference=reference, scene_path=scene,
        cameras_path=cameras, mask_dir=masks,
        filter_frames=set(frame_paths), pins=pin_entry,
        frame_paths=dict(frame_paths), require_pins=True)


DEFAULT_HOOKS = ArmHooks(
    run_a0=_default_run_a0,
    load_evaluator_camera=_default_load_evaluator_camera,
    reconstruct=reconstruct_facade,
    load_truth=_default_load_truth,
    record=_noop_record,
)


def resolve_hooks(hooks: ArmHooks | None) -> ArmHooks:
    return DEFAULT_HOOKS if hooks is None else hooks


def _emit(hooks: ArmHooks, event: str, **detail: Any) -> None:
    hooks.record(event, detail)


# ---------------------------------------------------------------------------
# Code / factor evidence
# ---------------------------------------------------------------------------

def code_hashes() -> dict[str, str]:
    """SHA-256 of every implementation the arm depends on, repo-relative."""
    paths = {
        "scripts/evaluation/run_image_only_route_a_camera_arm.py": HARNESS_PATH,
        "scripts/evaluation/run_image_only_route_a.py": ROUTE_A_RUNNER_PATH,
        "scripts/evaluation/report_image_only_attribution.py": ATTRIBUTION_RUNNER_PATH,
        "src/mc_photogrammetry/reconstruction/facade.py":
            REPO_ROOT / "src" / "mc_photogrammetry" / "reconstruction" / "facade.py",
        "src/mc_photogrammetry/reconstruction/camera.py":
            REPO_ROOT / "src" / "mc_photogrammetry" / "reconstruction" / "camera.py",
        "src/mc_photogrammetry/pipeline/image_only.py":
            REPO_ROOT / "src" / "mc_photogrammetry" / "pipeline" / "image_only.py",
        "src/mc_photogrammetry/evaluation.py":
            REPO_ROOT / "src" / "mc_photogrammetry" / "evaluation.py",
        "src/mc_photogrammetry/evaluation_image_only_attribution.py":
            REPO_ROOT / "src" / "mc_photogrammetry" / "evaluation_image_only_attribution.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def factor_record(outcome: Any, settings: ArmSettings) -> dict[str, Any]:
    """The non-camera objects/settings both arms must share, by digest.

    ``outcome`` is the ordinary Route A replay outcome, so its ``image``,
    ``mask`` and ``palette`` are exactly the objects that runner passed to
    ``reconstruct_facade`` (``run_image_only_route_a.py:141-175``); the A1 arm
    reuses those same objects.
    """
    palette = _palette_record(outcome.palette)
    return {
        "image_sha256": _sha256_bytes(
            np.ascontiguousarray(outcome.image).tobytes()),
        "image_shape": list(outcome.image.shape),
        "image_dtype": str(outcome.image.dtype),
        "gui_mask_domain_sha256": _mask_domain_sha256(outcome.mask),
        "palette_states": palette,
        "palette_states_sha256": _digest_json(palette),
        "settings": settings.to_record(),
        "camera_is_the_only_changed_object": True,
    }


# ---------------------------------------------------------------------------
# A0 seal and A1 arm
# ---------------------------------------------------------------------------

def build_a0_seal(outcome: Any, settings: ArmSettings,
                  hashes: Mapping[str, str]) -> dict[str, Any]:
    """Serialize and hash A0 before any evaluator file is opened."""
    facade_states = _states_of(outcome.facade)
    decoded_states = _states_of(outcome.decoded_export)
    offset = list(_translation_offset(facade_states, decoded_states))
    normalized_decoded = {
        tuple(coordinate[index] + offset[index] for index in range(3)): state
        for coordinate, state in decoded_states.items()}
    facade_record = factor_record(outcome, settings)
    camera = _camera_record(outcome.camera)
    frontier = getattr(getattr(outcome, "route_a", None), "frontier", None)
    record = {
        "schema": ARM_SCHEMA,
        "task": TASK_ID,
        "arm": "A0",
        "provenance": IMAGE_ONLY,
        "out_of_promotion": False,
        "sealed": True,
        "order_sentinel": ORDER_SENTINEL,
        "evaluator_reads_before_seal": 0,
        "camera": camera,
        "camera_sha256": _digest_json(camera),
        "camera_solved": outcome.camera is not None,
        "camera_error": outcome.camera_error,
        "facade_block_count": len(facade_states),
        "facade_volume_sha256": _volume_digest(facade_states),
        "decoded_block_count": len(decoded_states),
        "decoded_volume_sha256": _volume_digest(decoded_states),
        "decoded_region_origin": offset,
        "export_round_trip_exact": (
            None if not facade_states or not decoded_states
            else facade_states == normalized_decoded),
        "factor_record": facade_record,
        "factor_record_sha256": _digest_json(facade_record),
        "settings": settings.to_record(),
        "code_hashes": dict(hashes),
        "frontier_sha256": (None if frontier is None
                            else _digest_json(frontier.to_record())),
        "frontier_terminal_stage": (
            None if frontier is None else frontier.terminal_policy["stage"]),
        "frontier_terminal_status": (
            None if frontier is None else frontier.terminal_status),
    }
    record["a0_seal_sha256"] = _digest_json(record)
    return record


def build_a1_arm(*, outcome: Any, camera: CameraModel, settings: ArmSettings,
                 reconstruct: Callable[..., Any],
                 factors: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the facade through the same call with only the camera changed."""
    if not isinstance(camera, CameraModel):
        raise ArmCameraRefusal("A1 needs an evaluator CameraModel")
    volume = reconstruct(
        outcome.image, outcome.mask, columns=settings.columns, rows=settings.rows,
        depth_layers=settings.depth_layers, horizon=None, camera=camera,
        scene_transform=None, palette=outcome.palette)
    states = _states_of(volume)
    camera_record = _camera_record(camera)
    assisted = ReferenceAssistedArm(
        arm="A1", provenance=REFERENCE_ASSISTED, out_of_promotion=OUT_OF_PROMOTION,
        evaluator_camera=camera_record,
        evaluator_camera_sha256=_digest_json(camera_record),
        facade_block_count=len(states), facade_volume_sha256=_volume_digest(states))
    return {
        "arm": "A1",
        "provenance": REFERENCE_ASSISTED,
        "out_of_promotion": OUT_OF_PROMOTION,
        "states": states,
        "camera": camera,
        "camera_record": camera_record,
        "facade_block_count": len(states),
        "facade_volume_sha256": _volume_digest(states),
        "factor_record_sha256": _digest_json(factors),
        "reference_assisted": assisted,
        "order_sentinel": ORDER_SENTINEL,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def score_arm(*, label: str, provenance: str, states: Mapping, truth: Mapping,
              out_of_promotion: bool) -> dict[str, Any]:
    """Score one arm's frozen facade through the accepted evaluator G scorer."""
    reference = truth["reference"]
    visible_reference = truth["visible_reference"]
    turns, translation = best_visible_alignment(
        reference, states, visible_reference.visible)
    alignment = alignment_record(
        rotation_quarter_turns=int(turns),
        translation=[int(value) for value in translation],
        source=ALIGNMENT_SOURCE)
    score = score_aligned_cells(
        reference=reference, candidate=states, visible_reference=visible_reference,
        alignment=alignment, checkpoint="facade_grid",
        space=MetricSpace.GEOMETRY)
    aligned = align_candidate(states, alignment)
    visible_cells = set(visible_reference.visible)
    correct = set(score.matched)
    missing = visible_cells - correct
    false = set(aligned) - set(reference)
    denominator = score.visible_cells
    recall = _rate(len(correct), denominator)
    precision = _rate(len(correct), len(correct) + len(false))
    return {
        "arm": label,
        "provenance": provenance,
        "out_of_promotion": out_of_promotion,
        "space": "G",
        "universe": "frozen_reference_cells",
        "checkpoint": "facade_grid",
        "alignment": alignment,
        "visible_cells": denominator,
        "aligned_candidate_cells": len(aligned),
        "correct_cells": len(correct),
        "missing_cells": len(missing),
        "false_cells": len(false),
        "exact_state_cells": len(score.exact),
        "hidden_inferred_cells": score.hidden_inferred_cells,
        "recall": recall,
        "precision": precision,
        "f1": _f1(precision, recall),
        "correct_cell_sha256": _cell_set_digest(correct),
        "missing_cell_sha256": _cell_set_digest(missing),
        "false_cell_sha256": _cell_set_digest(false),
        "correct_cells_sorted": sorted(list(coordinate) for coordinate in correct),
        "missing_cells_sorted": sorted(list(coordinate) for coordinate in missing),
        "false_cells_sorted": sorted(list(coordinate) for coordinate in false),
        "volume_sha256": _volume_digest(states),
    }


def _delta(left: Mapping[str, Any], right: Mapping[str, Any],
           keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        a0 = left.get(key)
        a1 = right.get(key)
        result[key] = None if a0 is None or a1 is None else a1 - a0
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_camera_arm(
    *,
    image: Path,
    fixture: Path,
    reference: Path,
    scene: Path,
    cameras: Path,
    masks: Path | None,
    pins: Path | None,
    fixture_role: str | None,
    workdir: Path,
    settings: ArmSettings | None = None,
    expected_visible_cells: int = EXPECTED_VISIBLE_CELLS,
    hooks: ArmHooks | None = None,
) -> dict[str, Any]:
    """Run the one paired A0/A1 arm in the mandated evaluator order."""
    hooks = resolve_hooks(hooks)
    settings = settings or ArmSettings()
    image = Path(image)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frame_paths = {image.name: image}
    masks = Path(masks) if masks is not None else _mask_dir_for(fixture)
    guard = ArmOrderingGuard()

    _emit(hooks, "start", image=str(image), frame_id=image.name)

    # 1. A0 on the unmodified production path, evaluator data unreachable.
    outcome = hooks.run_a0(image_path=image, workdir=workdir, settings=settings)
    _emit(hooks, "a0_reconstructed",
          facade_block_count=len(_states_of(getattr(outcome, "facade", None))))

    # 2. Serialize and seal A0 before any evaluator file is opened.
    hashes = code_hashes()
    seal = build_a0_seal(outcome, settings, hashes)
    seal_path = workdir / "a0-seal.json"
    seal_path.write_text(canonical_json(seal) + "\n", encoding="utf-8")
    guard.mark_sealed()
    _emit(hooks, "a0_sealed", a0_seal_sha256=seal["a0_seal_sha256"],
          seal_path=str(seal_path))

    # 3. Evaluator camera -> A1 through the same reconstruct_facade call.
    guard.require_sealed("the evaluator camera load")
    camera = hooks.load_evaluator_camera(cameras_path=cameras, frame_id=image.name)
    if not isinstance(camera, CameraModel):
        raise ArmCameraRefusal("the evaluator camera is not a CameraModel")
    _emit(hooks, "evaluator_camera_loaded",
          camera_sha256=_digest_json(_camera_record(camera)))
    factors = factor_record(outcome, settings)
    a1 = build_a1_arm(outcome=outcome, camera=camera, settings=settings,
                      reconstruct=hooks.reconstruct, factors=factors)
    _emit(hooks, "a1_reconstructed", facade_block_count=a1["facade_block_count"])

    # 4. Freeze and hash both arms before opening any visible-reference truth.
    a0_frozen = {
        "arm": "A0", "provenance": IMAGE_ONLY, "out_of_promotion": False,
        "facade_block_count": seal["facade_block_count"],
        "facade_volume_sha256": seal["facade_volume_sha256"],
        "camera_sha256": seal["camera_sha256"],
        "factor_record_sha256": seal["factor_record_sha256"],
    }
    a1_frozen = {
        "arm": "A1", "provenance": REFERENCE_ASSISTED,
        "out_of_promotion": OUT_OF_PROMOTION,
        "facade_block_count": a1["facade_block_count"],
        "facade_volume_sha256": a1["facade_volume_sha256"],
        "camera_sha256": a1["reference_assisted"].evaluator_camera_sha256,
        "factor_record_sha256": a1["factor_record_sha256"],
    }
    frozen_record = {
        "schema": ARM_SCHEMA, "task": TASK_ID,
        "order_sentinel": ORDER_SENTINEL,
        "camera_is_the_only_changed_object": (
            a0_frozen["factor_record_sha256"] == a1_frozen["factor_record_sha256"]),
        "arms": {"A0": a0_frozen, "A1": a1_frozen},
        "truth_opened_after_freeze": True,
    }
    (workdir / "arms-frozen.json").write_text(
        canonical_json(frozen_record) + "\n", encoding="utf-8")
    guard.mark_frozen()
    _emit(hooks, "arms_frozen",
          camera_is_the_only_changed_object=(
              a0_frozen["factor_record_sha256"] == a1_frozen["factor_record_sha256"]))

    # 5. Truth load and scoring, only now.
    guard.require_frozen("the visible-reference truth load")
    truth = hooks.load_truth(
        fixture=Path(fixture), reference=Path(reference), scene=Path(scene),
        cameras=Path(cameras), masks=masks, pins=pins, fixture_role=fixture_role,
        frame_paths=frame_paths)
    _emit(hooks, "truth_loaded")

    a0_states = _states_of(getattr(outcome, "facade", None))
    a0_score = score_arm(label="A0", provenance=IMAGE_ONLY, states=a0_states,
                         truth=truth, out_of_promotion=False)
    a1_score = score_arm(label="A1", provenance=REFERENCE_ASSISTED,
                         states=a1["states"], truth=truth,
                         out_of_promotion=OUT_OF_PROMOTION)
    observed_visible = a0_score["visible_cells"]
    if observed_visible != expected_visible_cells:
        raise ArmRefusal(
            "the frozen visible denominator does not match the fixed V="
            f"{expected_visible_cells}: observed {observed_visible}; the paired "
            "comparison would not be on the fixed denominator - stop and report "
            "this seam")
    deltas = _delta(
        a0_score, a1_score,
        ("visible_cells", "correct_cells", "missing_cells", "false_cells",
         "exact_state_cells", "aligned_candidate_cells", "recall", "precision",
         "f1"))

    truth_record = {
        "reference_sha256": truth.get("reference_sha256"),
        "reference_cells": len(truth.get("reference", {})),
        "visible_cells": observed_visible,
        "minimum_pixels": truth["visible_reference"].minimum_pixels,
        "mask_origin": truth.get("mask_origin"),
        "mask_pins_match": truth.get("mask_pins_match"),
        "reference_pin_matches": truth.get("reference_pin_matches"),
        "frames": truth.get("frames"),
        "caveats": list(truth.get("caveats", ())),
    }
    summary = {
        "schema": ARM_SCHEMA,
        "producer": ARM_PRODUCER,
        "task": TASK_ID,
        "kind": ("evaluation-only oracle-camera sensitivity / upper-bound arm; "
                 "not causal attribution, not production/promotion evidence"),
        "single_factor": "the CameraModel object is the only object changed",
        "order_sentinel": ORDER_SENTINEL,
        "order_checked": [
            "a0_reconstructed", "a0_sealed", "evaluator_camera_loaded",
            "a1_reconstructed", "arms_frozen", "truth_loaded", "scored"],
        "fixed_visible_cells": {
            "expected": expected_visible_cells, "observed": observed_visible},
        "alignment_family": ALIGNMENT_FAMILY,
        "alignment_source": ALIGNMENT_SOURCE,
        "arms": {"A0": a0_score, "A1": a1_score},
        "deltas": {"A1_minus_A0": deltas},
        "camera_is_the_only_changed_object":
            frozen_record["camera_is_the_only_changed_object"],
        "factor_record_sha256": a1["factor_record_sha256"],
        "a0_seal_sha256": seal["a0_seal_sha256"],
        "a0_export_round_trip_exact": seal["export_round_trip_exact"],
        "a0_camera_solved": seal["camera_solved"],
        "a0_camera_error": seal["camera_error"],
        "a1_provenance": REFERENCE_ASSISTED,
        "a1_out_of_promotion": OUT_OF_PROMOTION,
        "descriptive_reference": {
            "visible_geometry_f1_gate": DESCRIPTIVE_F1_GATE,
            "role": "descriptive only; K1 creates, changes and passes no threshold",
        },
        "code_hashes": hashes,
        "truth": truth_record,
        "predeclared_interpretation": list(PREDECLARED_INTERPRETATION),
        "limitations": list(LIMITATIONS),
    }
    summary["result_sha256"] = _digest_json(summary)
    _emit(hooks, "scored", result_sha256=summary["result_sha256"],
          a0_correct=a0_score["correct_cells"],
          a1_correct=a1_score["correct_cells"])
    return summary


def _mask_dir_for(fixture: Path) -> Path | None:
    tracked = REPO_ROOT / "docs" / "evaluation_masks" / fixture.name.replace(" ", "-")
    if tracked.is_dir():
        return tracked
    candidate = REPO_ROOT / ".masks" / fixture.name.replace(" ", "-")
    return candidate if candidate.is_dir() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("DS-STAGEWISE-K1 evaluation-only paired Route A image-camera "
                     "(A0) / evaluator-camera (A1) sensitivity arm"))
    parser.add_argument("--image", required=True, help="one frozen frame")
    parser.add_argument("--fixture", required=True, help="frozen fixture directory")
    parser.add_argument("--reference", required=True,
                        help="evaluator-only reference litematic")
    parser.add_argument("--scene", required=True, help="evaluator-only scene.json")
    parser.add_argument("--cameras", required=True,
                        help="evaluator-only cameras.json")
    parser.add_argument("--masks", default=None,
                        help="evaluator-only frozen mask directory")
    parser.add_argument("--pins", required=True,
                        help="reviewed evaluator fixture pin manifest")
    parser.add_argument("--fixture-role", required=True,
                        help="pinned fixture role (for example template-2)")
    parser.add_argument("--workdir", required=True,
                        help="output directory for the seal/arm/result records")
    parser.add_argument("--report", default=None, help="result JSON output path")
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--depth-layers", type=int, default=8)
    parser.add_argument("--expected-visible", type=int,
                        default=EXPECTED_VISIBLE_CELLS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_camera_arm(
            image=Path(args.image), fixture=Path(args.fixture),
            reference=Path(args.reference), scene=Path(args.scene),
            cameras=Path(args.cameras),
            masks=Path(args.masks) if args.masks else None,
            pins=Path(args.pins), fixture_role=args.fixture_role,
            workdir=Path(args.workdir),
            settings=ArmSettings(columns=args.columns, rows=args.rows,
                                 depth_layers=args.depth_layers),
            expected_visible_cells=args.expected_visible)
    except (ArmRefusal, ValueError, OSError) as error:
        print(canonical_json({
            "schema": ARM_SCHEMA, "producer": ARM_PRODUCER, "task": TASK_ID,
            "arm_available": False, "reason": type(error).__name__,
            "detail": str(error), "result": None,
            "notes": "typed refusal; no paired camera arm was built",
        }))
        return 2
    target = Path(args.report) if args.report else Path(args.workdir) / "k1-result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
