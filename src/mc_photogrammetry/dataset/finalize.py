"""Finalize staged captures into a dataset scene, and index the dataset.

Finalization is the gate between "a client took some screenshots" and "this is
dataset material": every planned frame must have a result, every result must
point at a real screenshot whose hash is recorded, the recorded camera must match
the planned feet/eye pose within a documented tolerance, and the export's blocks
must convert into a region-pinned ``reference.litematic``. Only after that does the
scene enter ``index.json`` with its deterministic split.

Nothing here touches the source world. Screenshots are read from the staging
game directory and copied into the dataset, never the other way round.
"""
from __future__ import annotations

import json
import hashlib
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable

from mc_photogrammetry.dataset.blocks import parse_ground_truth, write_reference_litematic
from mc_photogrammetry.dataset.hashing import sha256_file
from mc_photogrammetry.dataset.inventory import safe_join
from mc_photogrammetry.dataset.plan import (
    EYE_POSITION_TOLERANCE,
    HEADING_TOLERANCE_DEGREES,
)
from mc_photogrammetry.dataset.runtime_render_provenance import (
    MAX_SIDECAR_BYTES,
    ProvenanceError,
    read_bounded_file_snapshot,
    validate_sidecar,
)
from mc_photogrammetry.dataset.schemas import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    WEATHERS,
    CaptureFrameResult,
    CapturePlan,
    DatasetSchemaError,
    SceneRecipe,
)
from mc_photogrammetry.dataset.splits import (
    scene_group,
    split_for_scene,
    split_report,
    volume_topology,
)
from mc_photogrammetry.dataset.blocks import volume_from_ground_truth
from mc_photogrammetry.reconstruction.camera import camera_eye

MOD_OUTPUT_DIR = "mc-photo-capture"
SCENES_DIR = "scenes"
CAPTURES_DIR = "captures"
GROUND_TRUTH_LABEL = (
    "developer/evaluation ground truth only; never normal reconstruction input"
)

#: Requested-vs-actual tolerance for the client brightness option. The value is a
#: plain double that survives the JSON round trip, so the bound only absorbs
#: formatting; the schema already refuses a request outside the client's [0, 1]
#: option range, which is the only way a client-side clamp could occur.
GAMMA_TOLERANCE = 0.01

#: PNG signature. The IHDR chunk always starts at byte 8 and holds the width and
#: height as two big-endian 32-bit integers at bytes 16 and 20.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(path: Path) -> tuple[int, int] | None:
    """The pixel size a PNG declares in its IHDR, or None when it is not a PNG.

    The header is parsed directly rather than decoded: the finalizer only needs the
    declared dimensions to prove the screenshot matches the planned resolution, and
    a 24-byte read cannot depend on an image library or decompress a large frame.
    """
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != _PNG_SIGNATURE or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def write_json(path: Path, payload: Any, *, force: bool = False) -> Path:
    """Write canonical, deterministic JSON; refuse to clobber unless told to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise DatasetSchemaError(f"refusing to overwrite {path}")
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    if not path.exists():
        raise DatasetSchemaError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scene_dir(dataset_dir: Path | str, scene_id: str) -> Path:
    return safe_join(dataset_dir, SCENES_DIR, scene_id)


def plan_path(dataset_dir: Path | str, scene_id: str) -> Path:
    return scene_dir(dataset_dir, scene_id) / "plan.json"


def write_scene_plan(dataset_dir: Path | str, plan: CapturePlan, *,
                     force: bool = False) -> Path:
    return write_json(plan_path(dataset_dir, plan.scene_id), plan.to_dict(), force=force)


def write_scene_recipe(dataset_dir: Path | str, scene_id: str, recipe: SceneRecipe, *,
                       force: bool = False) -> Path:
    return write_json(scene_dir(dataset_dir, scene_id) / "recipe.json",
                      recipe.to_dict(), force=force)


def mod_output_dir(game_dir: Path | str, plan_id: str) -> Path:
    return safe_join(game_dir, MOD_OUTPUT_DIR, "out", plan_id)


def load_results(game_dir: Path | str, plan_id: str) -> tuple[CaptureFrameResult, ...]:
    payload = read_json(mod_output_dir(game_dir, plan_id) / "results.json")
    if not isinstance(payload, (list, tuple)):
        raise DatasetSchemaError("results.json must be a list of frame results")
    return tuple(CaptureFrameResult.from_dict(entry) for entry in payload)


def load_ground_truth(game_dir: Path | str, plan_id: str):
    return parse_ground_truth(read_json(mod_output_dir(game_dir, plan_id) / "ground_truth.json"))


def _triple(value: Any) -> tuple[float, float, float] | None:
    """A finite 3-vector from recorded JSON, or None when it is malformed."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values  # type: ignore[return-value]


def angle_delta_degrees(first: float, second: float) -> float:
    """Smallest absolute angle between two headings, in degrees.

    Yaw wraps: a client that reports 359.9 where the plan asked for -0.1 is
    exactly on target, and a plain subtraction would call it a 360-degree miss.
    The difference is rounded before the modulo so a full turn that floating point
    represents as 360.00000000000006 lands on 0 rather than on a whole turn.
    """
    difference = abs(round((float(first) - float(second)) % 360.0, 9)) % 360.0
    if difference > 180.0:
        difference = 360.0 - difference
    return difference


def validate_pose(
    plan: CapturePlan,
    results: Iterable[CaptureFrameResult],
    *,
    tolerance: float = EYE_POSITION_TOLERANCE,
    heading_tolerance: float = HEADING_TOLERANCE_DEGREES,
) -> list[str]:
    """Compare every recorded feet/eye pose with what the plan requested.

    ``CameraModel.position`` is the player's feet and the camera eye is
    ``position + eye_height``; the client records both, so a capture that put the
    feet where the eye should be is caught here rather than silently entering the
    dataset as ground truth. Yaw and pitch are checked too, with yaw compared as a
    circular delta.
    """
    by_frame = {result.frame_id: result for result in results}
    problems: list[str] = []
    for job in plan.jobs:
        result = by_frame.get(job.frame_id)
        if result is None or not result.captured:
            continue
        actual = result.actual if isinstance(result.actual, dict) else {}
        feet = _triple(actual.get("feet_position"))
        eye = _triple(actual.get("eye_position"))
        if feet is None or eye is None:
            problems.append(
                f"{job.frame_id}: result must record feet_position and eye_position")
            continue
        requested_feet = tuple(float(value) for value in job.camera.position)
        requested_eye = tuple(float(value) for value in camera_eye(job.camera))
        feet_delta = math.dist(feet, requested_feet)
        eye_delta = math.dist(eye, requested_eye)
        if feet_delta > tolerance:
            problems.append(
                f"{job.frame_id}: recorded feet {tuple(round(v, 4) for v in feet)} are "
                f"{feet_delta:.4f} blocks from the requested {requested_feet} "
                f"(tolerance {tolerance})")
        if eye_delta > tolerance:
            problems.append(
                f"{job.frame_id}: recorded eye {tuple(round(v, 4) for v in eye)} is "
                f"{eye_delta:.4f} blocks from the requested {requested_eye} "
                f"(tolerance {tolerance})")
        yaw = actual.get("yaw_degrees")
        pitch = actual.get("pitch_degrees")
        if not isinstance(yaw, (int, float)) or not isinstance(pitch, (int, float)):
            problems.append(
                f"{job.frame_id}: result must record yaw_degrees and pitch_degrees")
            continue
        yaw_delta = angle_delta_degrees(float(yaw), job.camera.yaw_degrees)
        pitch_delta = abs(float(pitch) - float(job.camera.pitch_degrees))
        if yaw_delta > heading_tolerance:
            problems.append(
                f"{job.frame_id}: recorded yaw {float(yaw):.4f} is {yaw_delta:.4f} degrees "
                f"from the requested {job.camera.yaw_degrees:.4f} "
                f"(tolerance {heading_tolerance})")
        if pitch_delta > heading_tolerance:
            problems.append(
                f"{job.frame_id}: recorded pitch {float(pitch):.4f} is "
                f"{pitch_delta:.4f} degrees from the requested "
                f"{job.camera.pitch_degrees:.4f} (tolerance {heading_tolerance})")
    return problems


def _number(value: Any) -> float | None:
    """A finite JSON number, or None; booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int_pair(data: Any, width_key: str, height_key: str) -> tuple[int, int] | None:
    """A pair of JSON integers under two keys, or None when either is malformed."""
    if not isinstance(data, dict):
        return None
    width, height = data.get(width_key), data.get(height_key)
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    return int(width), int(height)


def _recorded_weather(value: Any) -> str | None:
    """The weather a mapping describes, preferring the explicit label."""
    if not isinstance(value, dict):
        return None
    label = value.get("weather")
    if isinstance(label, str) and label in WEATHERS:
        return label
    raining, thundering = value.get("raining"), value.get("thundering")
    if isinstance(raining, bool) and isinstance(thundering, bool):
        if thundering:
            return "thunder"
        return "rain" if raining else "clear"
    return None


def validate_settings(plan: CapturePlan,
                      results: Iterable[CaptureFrameResult]) -> list[str]:
    """Compare every recorded request and actual client/world setting with the plan.

    A capture is ground truth only when the client really ran with the settings the
    plan declared. The result therefore carries both halves: ``actual.requested``
    (what the mod asked the client and world to do) and the flat ``actual`` values
    (what the client and server reported afterwards). Both are checked, because a
    mod that requested the wrong value and then honoured it is just as wrong as one
    that requested correctly and failed to apply it.

    The checks are:

    * FOV uses the integer-round contract: the plan's value is rounded once, and the
      requested and actual values must round to the same integer.
    * gamma, HUD, and the requested/actual framebuffer dimensions.
    * the time and weather contract. The mod freezes the staged copy's daylight
      cycle before applying the capture time, so the recorded time must equal the
      requested time exactly and the recorded weather must name the requested one.
    * the screenshot's own pixel size, checked by the caller against the planned
      resolution through :func:`png_dimensions`.
    """
    by_frame = {result.frame_id: result for result in results}
    problems: list[str] = []
    for job in plan.jobs:
        result = by_frame.get(job.frame_id)
        if result is None or not result.captured:
            continue
        actual = result.actual if isinstance(result.actual, dict) else {}
        requested = actual.get("requested")
        if not isinstance(requested, dict):
            problems.append(
                f"{job.frame_id}: result must record actual.requested settings")
            continue

        # FOV: one integer round, used for the plan, the request, and the actual.
        planned_fov = int(round(float(job.camera.fov_degrees)))
        recorded_fov = _number(requested.get("fov_degrees"))
        if recorded_fov is None or int(round(recorded_fov)) != planned_fov:
            problems.append(
                f"{job.frame_id}: recorded request fov {requested.get('fov_degrees')!r} "
                f"does not round to the planned {planned_fov}")
        actual_fov = _number(actual.get("fov_degrees"))
        if actual_fov is None or int(round(actual_fov)) != planned_fov:
            problems.append(
                f"{job.frame_id}: actual fov {actual.get('fov_degrees')!r} does not "
                f"round to the requested integer {planned_fov}")

        planned_gamma = float(job.settings.gamma)
        recorded_gamma = _number(requested.get("gamma"))
        if recorded_gamma is None or abs(recorded_gamma - planned_gamma) > GAMMA_TOLERANCE:
            problems.append(
                f"{job.frame_id}: recorded request gamma {requested.get('gamma')!r} is "
                f"not the planned {planned_gamma}")
        actual_gamma = _number(actual.get("gamma"))
        if actual_gamma is None or abs(actual_gamma - planned_gamma) > GAMMA_TOLERANCE:
            problems.append(
                f"{job.frame_id}: actual gamma {actual.get('gamma')!r} is not within "
                f"{GAMMA_TOLERANCE} of the requested {planned_gamma}")

        planned_hud = bool(job.settings.hud_hidden)
        if requested.get("hud_hidden") is not planned_hud:
            problems.append(
                f"{job.frame_id}: recorded request hud_hidden "
                f"{requested.get('hud_hidden')!r} is not the planned {planned_hud}")
        if actual.get("hud_hidden") is not planned_hud:
            problems.append(
                f"{job.frame_id}: actual hud_hidden {actual.get('hud_hidden')!r} does not "
                f"match the requested {planned_hud}")

        planned_size = (int(job.resolution[0]), int(job.resolution[1]))
        if _int_pair(requested, "framebuffer_width", "framebuffer_height") != planned_size:
            problems.append(
                f"{job.frame_id}: recorded request framebuffer "
                f"{_int_pair(requested, 'framebuffer_width', 'framebuffer_height')} is not "
                f"the planned resolution {planned_size}")
        actual_size = _int_pair(actual, "framebuffer_width", "framebuffer_height")
        if actual_size != planned_size:
            problems.append(
                f"{job.frame_id}: actual framebuffer {actual_size} does not match the "
                f"planned resolution {planned_size}")

        planned_time = int(job.settings.time_of_day)
        if requested.get("time_of_day") != planned_time:
            problems.append(
                f"{job.frame_id}: recorded request time_of_day "
                f"{requested.get('time_of_day')!r} is not the planned {planned_time}")
        if actual.get("time_of_day") != planned_time:
            problems.append(
                f"{job.frame_id}: actual time_of_day {actual.get('time_of_day')!r} does "
                f"not match the requested {planned_time}")

        planned_weather = str(job.settings.weather)
        if requested.get("weather") != planned_weather:
            problems.append(
                f"{job.frame_id}: recorded request weather "
                f"{requested.get('weather')!r} is not the planned {planned_weather!r}")
        actual_weather = _recorded_weather(actual)
        if actual_weather != planned_weather:
            problems.append(
                f"{job.frame_id}: actual weather {actual_weather!r} does not match the "
                f"requested {planned_weather!r}")
    return problems


def validate_results(plan: CapturePlan, results: Iterable[CaptureFrameResult]) -> list[str]:
    """Return every problem with a result set; empty means the set is complete."""
    by_frame = {result.frame_id: result for result in results}
    problems: list[str] = []
    for job in plan.jobs:
        result = by_frame.pop(job.frame_id, None)
        if result is None:
            problems.append(f"{job.frame_id}: no result recorded")
            continue
        if not result.captured:
            problems.append(f"{job.frame_id}: capture failed: {'; '.join(result.errors)}")
        if result.filename != job.filename:
            problems.append(
                f"{job.frame_id}: result filename {result.filename!r} != plan {job.filename!r}")
    for frame_id in sorted(by_frame):
        problems.append(f"{frame_id}: result has no matching capture job")
    return problems


def _resolve_screenshot(game_dir: Path, result: CaptureFrameResult) -> Path:
    relative = result.screenshot or f"screenshots/{result.filename}"
    parts = [part for part in relative.replace("\\", "/").split("/") if part]
    if not parts:
        raise DatasetSchemaError(f"{result.frame_id}: empty screenshot path")
    resolved = safe_join(game_dir, *parts)
    if not resolved.is_file():
        raise DatasetSchemaError(
            f"{result.frame_id}: screenshot not found at {resolved}")
    return resolved


def _recorded_screenshot_digest(result: CaptureFrameResult) -> str | None:
    """The SHA-256 the mod says it wrote, or None when it is absent/malformed."""
    actual = result.actual if isinstance(result.actual, dict) else {}
    value = actual.get("screenshot_sha256")
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def _validate_runtime_provenance(game_root: Path, plan: CapturePlan,
                                 results: list[CaptureFrameResult], *,
                                 scene: Path, mode: str,
                                 expectation_path: Path | None,
                                 expectation_sha256: str | None) -> dict[str, Any] | None:
    """Validate required v1 provenance before any finalized output is written.

    Older captures with no requirement marker stay readable, but are not
    backfilled or upgraded. New executor attempts mark the run status as
    requiring this sidecar; missing or incomplete evidence refuses finalization.
    """
    output = game_root / MOD_OUTPUT_DIR / "out" / plan.plan_id
    status_path = output / "run_status.json"
    fixed_expectation = scene / "runtime-render-provenance.expect.json"
    if mode == "legacy":
        if fixed_expectation.exists() or expectation_path is not None:
            raise DatasetSchemaError(
                "legacy provenance mode cannot bypass an executor-bound new attempt")
        if not status_path.is_file():
            return None
        try:
            legacy_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetSchemaError("cannot parse capture run status") from exc
        if isinstance(legacy_status, dict) and legacy_status.get(
                "runtime_render_provenance_required") is True:
            raise DatasetSchemaError("new capture requires provenance; legacy bypass refused")
        return None
    if mode not in ("report_only", "strict"):
        raise DatasetSchemaError("unknown runtime/render provenance mode")
    if expectation_path is None or expectation_sha256 is None:
        raise DatasetSchemaError("new capture finalization requires its launch expectation pin")
    try:
        if expectation_path.resolve() != fixed_expectation.resolve():
            raise DatasetSchemaError("launch expectation path is not the scene-owned expectation")
        if expectation_path.is_symlink() or expectation_path.stat().st_size > 16_384:
            raise DatasetSchemaError("executor-owned launch expectation is linked or over cap")
        with expectation_path.open("rb") as stream:
            expectation_raw = stream.read(16_385)
        if len(expectation_raw) > 16_384:
            raise DatasetSchemaError("executor-owned launch expectation exceeds byte cap")
    except OSError as exc:
        raise DatasetSchemaError("executor-owned launch expectation is missing") from exc
    if hashlib.sha256(expectation_raw).hexdigest() != expectation_sha256:
        raise DatasetSchemaError("executor-owned launch expectation digest mismatch")
    try:
        trusted = json.loads(expectation_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetSchemaError("executor-owned launch expectation is malformed") from exc
    if not isinstance(trusted, dict) or trusted.get("schema_version") != 1:
        raise DatasetSchemaError("executor-owned launch expectation schema is invalid")
    if trusted.get("mode") != mode or trusted.get("scene_id") != plan.scene_id or \
            trusted.get("plan_id") != plan.plan_id or \
            trusted.get("staging_token") != plan.staging_token or \
            trusted.get("world_identity") != plan.world_identity:
        raise DatasetSchemaError("executor-owned expectation does not match this scene/plan")
    dataset_plan_bytes = (scene / "plan.json").read_bytes()
    if hashlib.sha256(dataset_plan_bytes).hexdigest() != trusted.get("dataset_plan_sha256"):
        raise DatasetSchemaError("dataset plan changed after the runner armed this attempt")
    if trusted.get("staging_game_dir") != str(game_root.resolve()):
        raise DatasetSchemaError("launch expectation staging root differs from finalizer root")
    staged_plan_path = game_root / MOD_OUTPUT_DIR / "plans" / f"{plan.plan_id}.json"
    if not staged_plan_path.is_file() or sha256_file(staged_plan_path) != trusted.get("plan_sha256"):
        raise DatasetSchemaError("staged plan bytes differ from executor expectation")
    staged_mod_path = Path(str(trusted.get("capture_mod_staged_path", "")))
    expected_mod_root = (game_root / "mods").resolve()
    if staged_mod_path.resolve().parent != expected_mod_root or \
            not staged_mod_path.is_file() or \
            sha256_file(staged_mod_path) != trusted.get("capture_mod_staged_sha256"):
        raise DatasetSchemaError("staged capture mod bytes differ from executor expectation")
    if mode == "strict" and (not trusted.get("capture_mod_source_sha256") or
                               trusted.get("capture_mod_source_sha256") !=
                               trusted.get("capture_mod_expected_sha256")):
        raise DatasetSchemaError("strict provenance requires a verified capture-mod source pin")
    if not status_path.is_file():
        raise DatasetSchemaError("required runtime/render run status is missing")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetSchemaError("cannot parse capture run status for provenance") from exc
    if not isinstance(status, dict) or status.get("runtime_render_provenance_required") is not True:
        raise DatasetSchemaError("run status omitted the required provenance marker")
    if status.get("plan_id") != trusted.get("plan_id") or \
            status.get("staging_token") != trusted.get("staging_token"):
        raise DatasetSchemaError("run status plan/staging identity differs from expectation")
    reference = status.get("runtime_render_provenance")
    if not isinstance(reference, dict):
        raise DatasetSchemaError("required runtime/render provenance receipt is missing")
    if reference.get("attempt_nonce") != trusted.get("attempt_nonce") or \
            reference.get("expectation_sha256") != expectation_sha256:
        raise DatasetSchemaError("run status does not bind the executor launch expectation")
    if reference.get("plan_sha256") != trusted.get("plan_sha256") or \
            reference.get("capture_mod_staged_sha256") != trusted.get("capture_mod_staged_sha256") or \
            reference.get("capture_mod_source_sha256") != trusted.get("capture_mod_source_sha256") or \
            reference.get("mode") != mode:
        raise DatasetSchemaError("run status plan or staged-mod identity differs from expectation")
    filename = reference.get("filename")
    if filename != "runtime-render-provenance.json":
        raise DatasetSchemaError("runtime/render sidecar filename is invalid")
    sidecar_path = output / filename
    try:
        raw = read_bounded_file_snapshot(sidecar_path, max_bytes=MAX_SIDECAR_BYTES)
    except (OSError, ProvenanceError) as exc:
        if isinstance(exc, ProvenanceError) and "byte cap" in str(exc):
            raise DatasetSchemaError("runtime/render sidecar exceeds byte cap") from exc
        raise DatasetSchemaError("required runtime/render sidecar cannot be read") from exc
    if len(raw) > MAX_SIDECAR_BYTES:
        raise DatasetSchemaError("runtime/render sidecar exceeds byte cap")
    digest = hashlib.sha256(raw).hexdigest()
    if reference.get("bytes") != len(raw) or reference.get("sha256") != digest:
        raise DatasetSchemaError("runtime/render sidecar byte/hash binding is invalid")
    try:
        document = json.loads(raw.decode("utf-8"))
        expected = {
            "attempt_nonce": trusted.get("attempt_nonce"),
            "plan_id": plan.plan_id,
            "plan_sha256": trusted.get("plan_sha256"),
            "dataset_plan_sha256": trusted.get("dataset_plan_sha256"),
            "staging_token": plan.staging_token,
            "scene_id": plan.scene_id,
            "world_identity": plan.world_identity,
            "mode": mode,
            "expectation_sha256": expectation_sha256,
            "capture_mod_sha256": trusted.get("capture_mod_staged_sha256"),
            "capture_mod_source_sha256": trusted.get("capture_mod_source_sha256"),
            "capture_mod_staged_sha256": trusted.get("capture_mod_staged_sha256"),
        }
        expected_frames = []
        for result in results:
            actual = dict(result.actual)
            expected_frames.append({
                "frame_id": result.frame_id,
                "filename": result.filename,
                "screenshot_sha256": actual.get("screenshot_sha256"),
                "screenshot_bytes": actual.get("screenshot_bytes"),
            })
        sidecar_seal = validate_sidecar(
            document, expected=expected, expected_frames=expected_frames,
            require_certifiable=(mode == "strict"),
        )
        if reference.get("seal_sha256") != sidecar_seal:
            raise ProvenanceError("run status sidecar seal reference mismatch")
        frame_rows = {row["frame_id"]: row for row in document["frames"]}
        for result in results:
            actual = dict(result.actual)
            binding = actual.get("runtime_render_provenance")
            row = frame_rows[result.frame_id]
            if not isinstance(binding, dict) or \
                    binding.get("filename") != filename or \
                    binding.get("attempt_nonce") != trusted.get("attempt_nonce") or \
                    binding.get("snapshot_sha256") != row.get("snapshot_sha256"):
                raise ProvenanceError(
                    f"frame {result.frame_id} does not bind to its sidecar snapshot")
    except (UnicodeDecodeError, json.JSONDecodeError, ProvenanceError, KeyError) as exc:
        raise DatasetSchemaError(f"runtime/render provenance validation failed: {exc}") from exc
    certification = document.get("certification")
    unavailable_reasons = certification.get("unavailable_reasons", []) \
        if isinstance(certification, dict) else ["certification evidence unavailable"]
    return {"filename": filename, "sha256": digest, "bytes": len(raw),
            "seal_sha256": sidecar_seal,
            "expectation_sha256": expectation_sha256,
            "status": "certified" if mode == "strict" else "report_only_incomplete",
            "certification_status": (certification.get("status")
                                     if isinstance(certification, dict) else "unavailable"),
            "unavailable_reasons": unavailable_reasons}


def _ensure_outside_dataset(dataset_dir: Path, game_dir: Path) -> None:
    dataset = dataset_dir.resolve()
    game = game_dir.resolve()
    if dataset == game or game in dataset.parents:
        raise DatasetSchemaError(
            "dataset root must not live inside the staging game directory")


def finalize_scene(
    dataset_dir: Path | str,
    scene_id: str,
    game_dir: Path | str,
    *,
    force: bool = False,
    provenance_mode: str = "legacy",
    provenance_expectation_path: Path | str | None = None,
    provenance_expectation_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one staged capture and write its dataset scene files.

    Returns the dataset index entry for the scene. Raises with every problem
    listed rather than writing a partially valid scene.
    """
    dataset_root = Path(dataset_dir)
    game_root = Path(game_dir)
    _ensure_outside_dataset(dataset_root, game_root)
    scene = scene_dir(dataset_root, scene_id)
    plan = CapturePlan.from_dict(read_json(scene / "plan.json"))
    if plan.scene_id != scene_id:
        raise DatasetSchemaError(
            f"plan scene_id {plan.scene_id!r} does not match directory {scene_id!r}")
    if plan.context_export_only:
        raise DatasetSchemaError(
            "context-export-only ground truth cannot be finalized as a capture dataset scene")
    for name in ("capture.json", "ground_truth.json", "reference.litematic"):
        if (scene / name).exists() and not force:
            raise DatasetSchemaError(f"refusing to overwrite {scene / name}")

    results = load_results(game_root, plan.plan_id)
    problems = validate_results(plan, results)
    problems.extend(validate_pose(plan, results))
    problems.extend(validate_settings(plan, results))
    if problems:
        raise DatasetSchemaError("capture validation failed: " + "; ".join(problems))

    runtime_provenance = _validate_runtime_provenance(
        game_root, plan, results, scene=scene, mode=provenance_mode,
        expectation_path=(Path(provenance_expectation_path)
                         if provenance_expectation_path is not None else None),
        expectation_sha256=provenance_expectation_sha256)

    job_by_frame = {job.frame_id: job for job in plan.jobs}
    source_by_frame: dict[str, tuple[Path, str, int]] = {}
    # Verify every original image join before creating/copying any finalized
    # output, so a late frame mismatch cannot leave a partial dataset scene.
    for result in results:
        source = _resolve_screenshot(game_root, result)
        recorded_digest = _recorded_screenshot_digest(result)
        if recorded_digest is None:
            raise DatasetSchemaError(
                f"{result.frame_id}: result must record actual.screenshot_sha256")
        source_digest = sha256_file(source)
        source_bytes = source.stat().st_size
        actual_bytes = dict(result.actual).get("screenshot_bytes")
        if source_digest != recorded_digest or (
                runtime_provenance is not None and actual_bytes != source_bytes):
            raise DatasetSchemaError(
                f"{result.frame_id}: refusing to copy a changed screenshot; hash/byte binding changed")
        job = job_by_frame.get(result.frame_id)
        planned_size = (int(job.resolution[0]), int(job.resolution[1])) if job else None
        if planned_size is not None and png_dimensions(source) != planned_size:
            raise DatasetSchemaError(
                f"{result.frame_id}: screenshot pixels do not match the plan")
        source_by_frame[result.frame_id] = (source, source_digest, source_bytes)

    screenshots_dir = scene / CAPTURES_DIR
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    screenshot_rows: list[dict[str, Any]] = []
    for result in results:
        source, source_digest, source_bytes = source_by_frame[result.frame_id]
        destination = safe_join(screenshots_dir, result.filename)
        if destination.exists():
            if not force:
                raise DatasetSchemaError(f"refusing to overwrite {destination}")
            destination.unlink()
        shutil.copy2(source, destination)
        copied_digest = sha256_file(destination)
        if copied_digest != source_digest:
            raise DatasetSchemaError(
                f"{result.frame_id}: copied screenshot {destination} hashes to "
                f"{copied_digest} but the source hashed to {source_digest}")
        screenshot_rows.append({
            "filename": result.filename,
            "sha256": copied_digest,
            "source_sha256": source_digest,
            "bytes": destination.stat().st_size,
            "frame_id": result.frame_id,
        })

    ground_truth = load_ground_truth(game_root, plan.plan_id)
    if ground_truth.scene_id != scene_id:
        raise DatasetSchemaError(
            f"ground truth scene_id {ground_truth.scene_id!r} does not match {scene_id!r}")
    if ground_truth.world_identity != plan.world_identity:
        raise DatasetSchemaError(
            "ground truth world identity does not match the plan")

    recipe_path = scene / "recipe.json"
    recipe = SceneRecipe.from_dict(read_json(recipe_path)) if recipe_path.exists() else None
    volume = volume_from_ground_truth(ground_truth)
    topology_digest = volume_topology(volume)
    # The split is keyed by the topology alone: recapturing this structure with a
    # different capture seed must not be able to move it between splits.
    group = scene_group(topology_digest)
    split = split_for_scene(volume=volume)

    ground_truth_path = write_json(scene / "ground_truth.json",
                                   ground_truth.to_dict(), force=force)
    capture_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": plan.dataset_id,
        "scene_id": scene_id,
        "plan_id": plan.plan_id,
        "world_identity": plan.world_identity,
        "region": plan.region.to_dict(),
        "advanced_only": True,
        "usage": GROUND_TRUTH_LABEL,
        "position_tolerance_blocks": EYE_POSITION_TOLERANCE,
        "frames": [result.to_dict() for result in results],
        "screenshots": screenshot_rows,
        "cameras": [job.camera.to_dict() for job in plan.jobs],
        "resolutions": [list(job.resolution) for job in plan.jobs],
    }
    if runtime_provenance is not None:
        source_sidecar = game_root / MOD_OUTPUT_DIR / "out" / plan.plan_id / runtime_provenance["filename"]
        sidecar_destination = scene / runtime_provenance["filename"]
        if sidecar_destination.exists():
            if not force:
                raise DatasetSchemaError(f"refusing to overwrite {sidecar_destination}")
            sidecar_destination.unlink()
        shutil.copy2(source_sidecar, sidecar_destination)
        if sha256_file(sidecar_destination) != runtime_provenance["sha256"]:
            raise DatasetSchemaError("copied runtime/render sidecar hash changed")
        capture_payload["runtime_render_provenance"] = dict(runtime_provenance)
        capture_payload["runtime_render_provenance"]["mode"] = provenance_mode
    capture_path = write_json(scene / "capture.json", capture_payload, force=force)

    reference_path = scene / "reference.litematic"
    if reference_path.exists():
        if not force:
            raise DatasetSchemaError(f"refusing to overwrite {reference_path}")
        reference_path.unlink()
    write_reference_litematic(ground_truth, reference_path, name=scene_id)

    return _index_entry(
        dataset_root=dataset_root,
        scene=scene,
        scene_id=scene_id,
        split=split,
        recipe=recipe,
        plan=plan,
        screenshots=screenshot_rows,
        topology_digest=topology_digest,
        scene_group_digest=group,
    )


def _index_entry(
    *,
    dataset_root: Path,
    scene: Path,
    scene_id: str,
    split: str,
    recipe: SceneRecipe | None,
    plan: CapturePlan,
    screenshots: list[dict[str, Any]],
    topology_digest: str = "",
    scene_group_digest: str = "",
) -> dict[str, Any]:
    relative = scene.relative_to(dataset_root).as_posix()
    files = ("recipe.json", "plan.json", "capture.json", "ground_truth.json",
             "reference.litematic", "runtime-render-provenance.json")
    files = (*files, "runtime-render-provenance.expect.json")
    hashes = {}
    for name in files:
        path = scene / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return {
        "scene_id": scene_id,
        "split": split,
        "family": recipe.family if recipe else "custom",
        "seed": recipe.seed if recipe else 0,
        "mod_contract_version": int(plan.mod_contract_version),
        "captures": len(plan.jobs),
        "directory": relative,
        "topology_digest": topology_digest,
        "scene_group": scene_group_digest,
        "hashes": hashes,
        "screenshots": sorted(screenshots, key=lambda row: row["filename"]),
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "source": "real-minecraft-capture",
            "world_identity": plan.world_identity,
            "plan_id": plan.plan_id,
            "staging_token": plan.staging_token,
            "split_identity": "scene-topology",
            "advanced_only": True,
            "usage": GROUND_TRUTH_LABEL,
        },
    }


def write_dataset_index(dataset_dir: Path | str, *, force: bool = False) -> dict[str, Any]:
    """Rebuild ``index.json`` from the scenes on disk, newest content wins."""
    root = Path(dataset_dir)
    scenes_root = root / SCENES_DIR
    if not scenes_root.is_dir():
        raise DatasetSchemaError(f"no scenes directory under {root}")
    entries: list[dict[str, Any]] = []
    for scene in sorted(scenes_root.iterdir()):
        if not scene.is_dir():
            continue
        capture = read_json(scene / "capture.json")
        plan = CapturePlan.from_dict(read_json(scene / "plan.json"))
        recipe_path = scene / "recipe.json"
        recipe = (SceneRecipe.from_dict(read_json(recipe_path))
                  if recipe_path.exists() else None)
        volume = volume_from_ground_truth(
            parse_ground_truth(read_json(scene / "ground_truth.json")))
        topology_digest = volume_topology(volume)
        entries.append(_index_entry(
            dataset_root=root,
            scene=scene,
            scene_id=str(capture["scene_id"]),
            split=split_for_scene(volume=volume),
            recipe=recipe,
            plan=plan,
            screenshots=list(capture.get("screenshots", [])),
            topology_digest=topology_digest,
            scene_group_digest=scene_group(topology_digest),
        ))
    report = split_report(entries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "mod_contract_version": int(entries[0]["mod_contract_version"]) if entries else 0,
        "advanced_only": True,
        "usage": GROUND_TRUTH_LABEL,
        "summary": report,
        "scenes": entries,
    }
    write_json(root / "index.json", payload, force=force)
    return payload
