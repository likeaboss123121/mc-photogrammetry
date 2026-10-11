"""Evaluation-only splitting of line detections at automatic GUI exclusions.

This module has no calibration, image, camera, truth, or OpenCV dependency. It
operates on an already detected set of native-pixel line segments and the
automatic boolean exclusion mask from the same image. It is deliberately not
wired into production inference; CAM-SUPPORT-02 evaluates this transformation
as a paired treatment before any integration decision.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
import base64
import gzip
import io
import re
from pathlib import Path
import sys
import threading
import time
import numpy as np

from .calibration import MINIMUM_LENGTH
from .evaluation_valid_stencil_segments import (
    AcquisitionTerminalError, AcquisitionCapError, AcquisitionIntegrityError,
    AllocationBudget, FROZEN_CONFIG, PackedBuffer, check_input, extract_segments,
    pointwise_gray, sample_occupancy, pack_categories,
)

MAX_INPUT_SEGMENTS = 10_000
MAX_SAMPLES_PER_SEGMENT = 20_001
MAX_TOTAL_SAMPLES = 50_000_000
SAMPLE_SPACING_PIXELS = 0.5
MAX_STUDY_JSON_BYTES = 512_000_000
MAX_STUDY_TOTAL_BYTES = 512_000_000
STUDY_RESERVE_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_MEMBERS = 24
MAX_COMPRESSED_MEMBER_BYTES = 16 * 1024 * 1024
MAX_DECODED_MEMBER_BYTES = 96 * 1024 * 1024
MAX_BOUNDARY_MASK_PIXELS = 50_000_000
BOUNDARY_DILATION_RADIUS = 3
_STUDY_LOCK = threading.RLock()

_GUI_BORDER_REFERENCE_PINS = {
    "scripts/evaluation/run_gui_border.py": "44c6d3342729297ea1301f1885caccb24bd0896e009cc3bb0f7a5ba938418eae",
    "docs/reports/GUI-BORDER-01-2026-10-05.md": "2e1a7ba127da6949df4a20cfa1ea5f36e035bc273cbda6f6b5bca4df097215af",
    "mask_boundary_source": "6359c074ba6f29edc037884bb376256c67b578aba258b7cbb1bdc9f4e8d21564",
}

_EXPECTED_RUNTIME = {
    "python": "3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]",
    "numpy": "2.2.6",
    "opencv": "5.0.0",
    "pillow": "12.3.0",
}
_EXPECTED_DEPENDENCY_SHA256 = {
    "scripts/evaluation/run_unit_face_input.py": "52ea25ead4c0862230b9a72d807e0f5baa170dfbfd8c418f78d353504ec08832",
    "src/mc_photogrammetry/blocks/unit_face_catalogue.py": "ab828d782c7a7d5b83f66968b5900e392263c7839ea6569f8b3e8b9c81eda33d",
    "src/mc_photogrammetry/domain/models.py": "6a87d543661cc62ada9a154bcdaa089ca844f41691ff3b53207ad89e3f05a478",
    "src/mc_photogrammetry/pipeline/contracts.py": "db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc",
    "src/mc_photogrammetry/reconstruction/calibration.py": "2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc",
    "src/mc_photogrammetry/reconstruction/camera.py": "0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680",
    "src/mc_photogrammetry/vision/gui_mask.py": "1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9",
    "scripts/evaluation/run_camera_mixed_depth.py": "29f2704b7f6a329e8058abd1c5fbfda5bb886a3a99aead99520f2b39b5d0a2e2",
}
_EXPECTED_CV2_PYD_SHA256 = "f1e76ab01cf4d33af780adf8ada1a26c5eaeecc55d2cbe8a5e288195e8f9b0dd"
_EXPECTED_CV2_BUILD_SHA256 = "f71df2f163323276b66a0c26f605490476e207bf50042bae19501cff4f1d865a"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GzipMemberStore:
    """Bounded append-only storage for canonical per-arm payloads."""

    def __init__(self, sidecar_path: Path, *,
                 max_members: int = MAX_PAYLOAD_MEMBERS,
                 max_compressed_bytes: int = MAX_COMPRESSED_MEMBER_BYTES,
                 max_decoded_bytes: int = MAX_DECODED_MEMBER_BYTES,
                 total_bytes: int = MAX_STUDY_TOTAL_BYTES,
                 reserve_bytes: int = STUDY_RESERVE_BYTES,
                 writer=None):
        self.sidecar_path = Path(sidecar_path)
        self.max_members = max_members
        self.max_compressed_bytes = max_compressed_bytes
        self.max_decoded_bytes = max_decoded_bytes
        self.total_bytes = total_bytes
        self.reserve_bytes = reserve_bytes
        self.writer = writer

    def _write_append(self, blob: bytes) -> int:
        if self.writer is not None:
            return self.writer(self.sidecar_path, blob)
        with self.sidecar_path.open("ab") as stream:
            written = stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
            return written

    def append(self, payload: bytes, plan: dict[str, object],
               identity: str, arm: str) -> dict[str, object]:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be canonical bytes")
        if len(payload) > self.max_decoded_bytes:
            raise RuntimeError("camera_support_decoded_member_byte_cap")
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("camera_support_payload_not_json") from exc
        if _canonical(parsed).encode("utf-8") != payload:
            raise RuntimeError("camera_support_payload_not_canonical")
        descriptors = plan.setdefault("sealed_predictions", [])
        if len(descriptors) >= self.max_members:
            raise RuntimeError("camera_support_member_count_cap")
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        if len(compressed) > self.max_compressed_bytes:
            raise RuntimeError("camera_support_compressed_member_byte_cap")
        offset = self.sidecar_path.stat().st_size if self.sidecar_path.exists() else 0
        expected_offset = sum(int(item["compressed_bytes"]) for item in descriptors)
        if offset != expected_offset:
            raise RuntimeError("camera_support_sidecar_tail_or_offset_mismatch")
        descriptor = {
            "identity": identity, "arm": arm,
            "member_index": len(descriptors), "offset": offset,
            "compressed_bytes": len(compressed), "decoded_bytes": len(payload),
            "payload_sha256": _sha256(payload),
            "compressed_sha256": _sha256(compressed),
        }
        candidate = dict(plan)
        candidate["sealed_predictions"] = [*descriptors, descriptor]
        candidate_json_bytes = len((_canonical(candidate) + "\n").encode("utf-8"))
        projected_total = (candidate_json_bytes + offset + len(compressed)
                           + self.reserve_bytes)
        if projected_total > self.total_bytes:
            raise RuntimeError("camera_support_result_byte_cap")
        written = self._write_append(compressed)
        if written != len(compressed):
            raise RuntimeError("camera_support_short_sidecar_write")
        if not self.sidecar_path.is_file() or self.sidecar_path.stat().st_size != offset + len(compressed):
            raise RuntimeError("camera_support_sidecar_append_size_mismatch")
        decoded = self.read_member(descriptor)
        if decoded != payload or _sha256(decoded) != descriptor["payload_sha256"]:
            raise RuntimeError("camera_support_payload_readback_hash_mismatch")
        return descriptor

    def read_member(self, descriptor: dict[str, object]) -> bytes:
        compressed_length = int(descriptor["compressed_bytes"])
        decoded_length = int(descriptor["decoded_bytes"])
        offset = int(descriptor["offset"])
        if (compressed_length < 1 or compressed_length > self.max_compressed_bytes
                or decoded_length < 0 or decoded_length > self.max_decoded_bytes
                or offset < 0):
            raise RuntimeError("camera_support_member_descriptor_cap")
        with self.sidecar_path.open("rb") as stream:
            stream.seek(offset)
            compressed = stream.read(compressed_length)
        if len(compressed) != compressed_length:
            raise RuntimeError("camera_support_truncated_sidecar_member")
        if _sha256(compressed) != descriptor["compressed_sha256"]:
            raise RuntimeError("camera_support_compressed_member_hash_mismatch")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
                decoded = stream.read(self.max_decoded_bytes + 1)
                if len(decoded) > self.max_decoded_bytes or stream.read(1):
                    raise RuntimeError("camera_support_decoded_member_byte_cap")
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise RuntimeError("camera_support_gzip_member_corrupt") from exc
        if len(decoded) != decoded_length:
            raise RuntimeError("camera_support_decoded_member_length_mismatch")
        try:
            parsed = json.loads(decoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("camera_support_payload_not_json") from exc
        if _canonical(parsed).encode("utf-8") != decoded:
            raise RuntimeError("camera_support_payload_not_canonical")
        if _sha256(decoded) != descriptor["payload_sha256"]:
            raise RuntimeError("camera_support_payload_hash_mismatch")
        return decoded

    def validate_all(self, plan: dict[str, object]) -> dict[str, object]:
        descriptors = plan.get("sealed_predictions", [])
        if len(descriptors) > self.max_members:
            raise RuntimeError("camera_support_member_count_cap")
        offset = 0
        hashes = []
        for index, descriptor in enumerate(descriptors):
            if descriptor["member_index"] != index or descriptor["offset"] != offset:
                raise RuntimeError("camera_support_member_order_mismatch")
            self.read_member(descriptor)
            offset += int(descriptor["compressed_bytes"])
            hashes.append(descriptor["payload_sha256"])
        actual_size = self.sidecar_path.stat().st_size if self.sidecar_path.exists() else 0
        if actual_size != offset:
            raise RuntimeError("camera_support_sidecar_unregistered_tail")
        return {"member_count": len(descriptors), "sidecar_bytes": actual_size,
                "payload_sha256": _sha256(_canonical(hashes))}


class ParityRefusal(RuntimeError):
    """Detector parity failure with independently recorded diagnostics."""

    def __init__(self, diagnostics: dict[str, object]):
        self.diagnostics = diagnostics
        failed = ",".join(key for key, value in diagnostics.items()
                           if isinstance(value, dict) and value.get("passed") is False)
        super().__init__("camera_support_pair_validation_failure:" + failed)


def canonical_dtype(value: object) -> str:
    """Normalize NumPy dtype descriptors such as ``<f8`` and ``float64``."""
    return np.dtype(value).str


def detector_parity_diagnostics(
    baseline: np.ndarray,
    candidate_dtype: object,
    candidate_shape: object,
    candidate_raw_sha256: str,
) -> dict[str, object]:
    """Evaluate and report all parity dimensions without short-circuiting."""
    baseline_array = np.asarray(baseline)
    baseline_digest = _sha256(np.ascontiguousarray(baseline_array).tobytes())
    try:
        canonical_candidate_dtype = canonical_dtype(candidate_dtype)
        dtype_error = None
    except (TypeError, ValueError) as exc:
        canonical_candidate_dtype = None
        dtype_error = str(exc)
    try:
        normalized_candidate_shape = [int(value) for value in candidate_shape]
        shape_error = None
    except (TypeError, ValueError, OverflowError) as exc:
        normalized_candidate_shape = None
        shape_error = str(exc)
    dtype_passed = (canonical_candidate_dtype == canonical_dtype(baseline_array.dtype))
    shape_passed = (normalized_candidate_shape == list(baseline_array.shape))
    digest_passed = (isinstance(candidate_raw_sha256, str)
                     and candidate_raw_sha256 == baseline_digest)
    return {
        "dtype": {"passed": dtype_passed,
                  "expected": canonical_dtype(baseline_array.dtype),
                  "observed": canonical_candidate_dtype, "error": dtype_error},
        "shape": {"passed": shape_passed,
                  "expected": list(baseline_array.shape),
                  "observed": normalized_candidate_shape, "error": shape_error},
        "raw_data_sha256": {"passed": digest_passed,
                            "expected": baseline_digest,
                            "observed": candidate_raw_sha256},
    }


def require_detector_parity(
    baseline: np.ndarray,
    candidate_dtype: object,
    candidate_shape: object,
    candidate_raw_sha256: str,
) -> dict[str, object]:
    diagnostics = detector_parity_diagnostics(
        baseline, candidate_dtype, candidate_shape, candidate_raw_sha256)
    if not all(item["passed"] for item in diagnostics.values()):
        raise ParityRefusal(diagnostics)
    return diagnostics


def execute_one_shot_attempt(
    plan: dict[str, object],
    case_ids: tuple[str, ...],
    *,
    trace_arm,
    validate_pair,
    seal_payload,
    store_payload,
    validate_payloads,
    persist,
    score_seals,
    encode_payload=None,
) -> dict[str, object]:
    """The single study runner: seal each arm, validate all seals, then score."""
    if (plan.get("execution_state") != "NOT_STARTED"
            or plan.get("sealed_predictions")
            or plan.get("evaluations") is not None):
        raise RuntimeError("camera_support_one_shot_already_claimed")
    plan["execution_state"] = "RUNNING"
    plan.setdefault("sealed_predictions", [])
    plan.setdefault("parity_diagnostics", [])
    persist(plan)
    active_case = None
    active_arm = None
    try:
        for case_id in case_ids:
            active_case = case_id
            arms = {}
            for arm in ("baseline", "candidate"):
                active_arm = arm
                try:
                    traced = trace_arm(case_id, arm)
                except AcquisitionTerminalError:
                    raise
                except Exception as exc:
                    traced = {"trace_result": _bounded_exception_receipt(exc),
                              "support": None}
                arms[arm] = traced
                sealed = seal_payload(case_id, arm, traced)
                payload_bytes = (_canonical(sealed).encode("utf-8") if encode_payload is None
                                 else encode_payload(sealed))
                descriptor = store_payload(case_id, arm, payload_bytes, plan)
                plan["sealed_predictions"].append(descriptor)
                persist(plan)
            active_arm = "pair_validation"
            diagnostics = validate_pair(case_id, arms["baseline"], arms["candidate"])
            plan["parity_diagnostics"].append({"case": case_id, **diagnostics})
            persist(plan)
            failed_checks = [key for key, value in diagnostics.items()
                             if isinstance(value, dict) and value.get("passed") is False]
            if failed_checks:
                raise ParityRefusal(diagnostics)
        plan["execution_state"] = "PREDICTIONS_SEALED"
        persist(plan)
        expected = {(identity, arm) for identity in case_ids
                    for arm in ("baseline", "candidate")}
        actual = {(item["identity"], item["arm"])
                  for item in plan["sealed_predictions"]}
        if len(plan["sealed_predictions"]) != len(expected) or actual != expected:
            raise RuntimeError("camera_support_seal_validation_failure")
        validation = validate_payloads(plan)
        plan["seal_validation"] = {"validated": True,
                                    "prediction_count": len(plan["sealed_predictions"]),
                                    **validation}
        plan["execution_state"] = "PREDICTIONS_VALIDATED"
        persist(plan)
        active_arm = "truth_scoring"
        scored = score_seals(plan["sealed_predictions"])
        if isinstance(scored, dict) and {"evaluations", "summary"} <= set(scored):
            plan["evaluations"] = scored["evaluations"]
            plan["summary"] = scored["summary"]
        else:
            plan["evaluations"] = scored
        plan["execution_state"] = "COMPLETE"
        persist(plan)
        return plan
    except Exception as exc:
        plan["execution_state"] = "FAILED"
        plan["failure"] = {
            "reason": (exc.args[0] if isinstance(exc, ParityRefusal)
                       else type(exc).__name__ + ":" + str(exc)),
            "sealed_prediction_count": len(plan["sealed_predictions"]),
            "post_hoc_truth_accessed": active_arm == "truth_scoring",
            "retry_performed": False,
            "case": active_case,
            "stage": active_arm,
        }
        if isinstance(exc, AcquisitionTerminalError):
            plan["failure"]["terminal_kind"] = type(exc).__name__
            plan["failure"]["terminal_evidence"] = exc.evidence
        if isinstance(exc, ParityRefusal):
            plan["failure"]["parity_diagnostics"] = exc.diagnostics
        persist(plan)
        raise


def _bounded_exception_receipt(exc: Exception) -> dict[str, object]:
    """Convert adapter exceptions into small, hashable partial-stage receipts."""
    receipt = getattr(exc, "partial_receipt", None)
    stages = receipt.get("stages", []) if isinstance(receipt, dict) else []
    bounded_stages = []
    for item in stages[:64] if isinstance(stages, list) else []:
        if isinstance(item, dict):
            row = {}
            for key in ("stage", "input_sha256", "output", "delegate_seconds"):
                if key not in item:
                    continue
                value = item[key]
                if key == "output":
                    try:
                        encoded = _canonical(value)
                    except (TypeError, ValueError):
                        encoded = repr(value)
                    if len(encoded.encode("utf-8")) > 4096:
                        value = {"truncated": True, "sha256": _sha256(encoded),
                                 "preview": encoded[:256]}
                elif isinstance(value, str):
                    value = value[:256]
                row[key] = value
            bounded_stages.append(row)
    return {
        "status": "trace_refusal",
        "reason": (type(exc).__name__ + ":" + str(exc))[:240],
        "prediction": None,
        "trace": {"trace_complete": False, "stages": bounded_stages,
                  "partial_stage_count": len(bounded_stages),
                  "truncated_stage_count": max(0, len(stages) - len(bounded_stages))
                    if isinstance(stages, list) else 0},
        "rgb_sha256": getattr(exc, "rgb_sha256", None),
        "rgb_shape": getattr(exc, "rgb_shape", None),
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _study_paths() -> tuple[Path, Path]:
    root = _project_root()
    return (root / "docs/reports/CAM-SUPPORT-02-2026-10-06.md",
            root / "docs/reports/CAM-SUPPORT-02-2026-10-06.json")


def _r1_paths() -> tuple[Path, Path]:
    root = _project_root()
    return (root / "docs/reports/CAM-SUPPORT-02R1-2026-10-06.md",
            root / "docs/reports/CAM-SUPPORT-02R1-2026-10-06.json")


def _r1a_paths() -> tuple[Path, Path]:
    root = _project_root()
    return (root / "docs/reports/CAM-SUPPORT-02R1A-2026-10-06.md",
            root / "docs/reports/CAM-SUPPORT-02R1A-2026-10-06.json")


def _r2_paths() -> tuple[Path, Path, Path]:
    root = _project_root()
    base = root / "docs/reports/CAM-SUPPORT-02R2-2026-10-06"
    return (Path(str(base) + ".md"), Path(str(base) + ".json"),
            Path(str(base) + ".payloads.gz"))


def _r3_paths() -> tuple[Path, Path, Path]:
    root = _project_root()
    base = root / "docs/reports/CAM-BOUNDARY-SPLIT-03-2026-10-06"
    return (Path(str(base) + ".md"), Path(str(base) + ".json"),
            Path(str(base) + ".payloads.gz"))


def _gui_border_reference_pins() -> dict[str, str]:
    root = _project_root()
    implementation = root / "scripts/evaluation/run_gui_border.py"
    report = root / "docs/reports/GUI-BORDER-01-2026-10-05.md"
    source = implementation.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "mask_boundary"), None)
    if function is None:
        raise RuntimeError("camera_boundary_reference_function_missing")
    function_source = "\n".join(source.splitlines()[function.lineno - 1:function.end_lineno]) + "\n"
    actual = {
        "scripts/evaluation/run_gui_border.py": _file_sha256(implementation),
        "docs/reports/GUI-BORDER-01-2026-10-05.md": _file_sha256(report),
        "mask_boundary_source": _sha256(function_source),
    }
    if actual != _GUI_BORDER_REFERENCE_PINS:
        raise RuntimeError("camera_boundary_reference_pin_mismatch")
    return actual


def _active_study_paths() -> tuple[Path, Path]:
    r3_report, r3_json, _ = _r3_paths()
    if r3_report.exists() or r3_json.exists():
        if not r3_report.is_file() or not r3_json.is_file():
            raise RuntimeError("camera_boundary_incomplete_r3_registry")
        return r3_report, r3_json
    r2_report, r2_json, _ = _r2_paths()
    if r2_report.exists() or r2_json.exists():
        if not r2_report.is_file() or not r2_json.is_file():
            raise RuntimeError("camera_support_incomplete_r2_registry")
        return r2_report, r2_json
    r1a_report, r1a_json = _r1a_paths()
    if r1a_report.exists() or r1a_json.exists():
        if not r1a_report.is_file() or not r1a_json.is_file():
            raise RuntimeError("camera_support_incomplete_r1a_registry")
        return r1a_report, r1a_json
    r1_report, r1_json = _r1_paths()
    if r1_report.exists() or r1_json.exists():
        if not r1_report.is_file() or not r1_json.is_file():
            raise RuntimeError("camera_support_incomplete_r1_registry")
        return r1_report, r1_json
    return _study_paths()


def _study_harness():
    # Runtime import is intentionally delayed so unit tests for the pure split
    # contract do not load cv2 or render any study cases.
    from scripts.evaluation import run_camera_mixed_depth as harness
    return harness


def _runtime_identity() -> dict[str, object]:
    import cv2
    import numpy
    import PIL

    pyd_candidates = sorted(Path(cv2.__file__).parent.glob("cv2*.pyd"))
    if len(pyd_candidates) != 1:
        raise RuntimeError("opencv_binary_identity_unavailable")
    pyd_sha = _sha256(pyd_candidates[0].read_bytes())
    build_sha = _sha256(cv2.getBuildInformation())
    runtime = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "cv2_binary_path": str(pyd_candidates[0].resolve()),
        "cv2_binary_sha256": pyd_sha,
        "opencv_build_information_sha256": build_sha,
    }
    expected = dict(_EXPECTED_RUNTIME)
    expected.update({"cv2_binary_sha256": _EXPECTED_CV2_PYD_SHA256,
                     "opencv_build_information_sha256": _EXPECTED_CV2_BUILD_SHA256})
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise RuntimeError("opencv_or_python_runtime_pin_mismatch")
    return runtime


def _fresh_scene_recipes(harness) -> tuple[dict[str, object], ...]:
    """Twelve new inputs: six ordinary and six preregistered stress variants."""
    common = {
        "version": "CAM-SUPPORT-02R1-ray-world-v2",
        "shape": [360, 480],
        "roll": 0.0,
        "principal": [240.0, 180.0],
        "eye": [1.25, 3.15, -0.5],
        "back_plane": {"axis": 2, "value": 9.0, "side": -1,
                       "bounds": [[-11.0, 8.0], [0.0, 10.0]]},
        "boxes": [],
        "floor": False,
        "crop": None,
        "low_texture": False,
        "gui_contamination": False,
        "native_side": 16,
        "seam_width_blocks": 0.018,
    }
    camera_sets = (
        (13.0, -19.0, 416.0), (41.0, -11.0, 372.0),
        (59.0, -25.0, 438.0), (78.0, -18.0, 398.0),
        (27.0, -14.0, 447.0), (51.0, -21.0, 407.0),
    )
    structures = (
        ("planar-a", False, []),
        ("planar-b", False, []),
        ("stepped-a", True, [
            {"lo": [-3.0, 0.0, 3.0], "hi": [-1.0, 2.0, 5.0]},
            {"lo": [-6.0, 0.0, 6.0], "hi": [-3.0, 4.0, 8.0]},
        ]),
        ("stepped-b", True, [
            {"lo": [-2.0, 0.0, 3.0], "hi": [0.0, 3.0, 5.0]},
            {"lo": [-5.0, 0.0, 6.0], "hi": [-2.0, 4.0, 8.0]},
        ]),
        ("occluding-a", True, [
            {"lo": [-1.0, 0.0, 1.5], "hi": [0.5, 3.5, 2.5]},
            {"lo": [3.0, 0.0, 4.0], "hi": [5.0, 2.0, 6.0]},
        ]),
        ("occluding-b", True, [
            {"lo": [-2.0, 0.0, 1.0], "hi": [-0.5, 3.0, 2.0]},
            {"lo": [2.5, 0.0, 4.0], "hi": [4.5, 2.5, 6.0]},
        ]),
    )
    bases: list[dict[str, object]] = []
    for index, (name, camera, structure) in enumerate(zip(
            (item[0] for item in structures), camera_sets, structures)):
        recipe = dict(common)
        recipe["identity"] = "support-ordinary-" + name
        recipe["yaw"], recipe["pitch"], recipe["focal"] = camera
        recipe["floor"] = structure[1]
        recipe["boxes"] = structure[2]
        # Distinct world windows avoid repeating the same image under another ID.
        if name == "planar-b":
            recipe["back_plane"] = {"axis": 2, "value": 10.0, "side": -1,
                                    "bounds": [[-9.0, 9.0], [-1.0, 11.0]]}
        if structure[1]:
            recipe["floor"] = True
        recipe["seed"] = 1_820_271 + index * 127
        bases.append(recipe)

    cases: list[dict[str, object]] = []
    for recipe in bases:
        cases.append({"identity": recipe["identity"], "stratum": "ordinary",
                      "base_identity": None, "recipe": recipe})

    # Two GUI-only contamination cases reuse ordinary scene geometry/camera.
    for base_index in (0, 4):
        recipe = dict(bases[base_index])
        recipe["identity"] = f"support-stress-gui-{base_index + 1}"
        recipe["gui_contamination"] = True
        cases.append({"identity": recipe["identity"], "stratum": "gui_contamination",
                      "base_identity": bases[base_index]["identity"], "recipe": recipe})

    # Two native crops exercise mask and principal-point handling at frame edges.
    for base_index, crop in ((1, [31, 21, 424, 318]), (3, [43, 17, 406, 326])):
        recipe = dict(bases[base_index])
        recipe["identity"] = f"support-stress-crop-{base_index + 1}"
        recipe["crop"] = crop
        cases.append({"identity": recipe["identity"], "stratum": "crop_boundary",
                      "base_identity": bases[base_index]["identity"], "recipe": recipe})

    # Two appearance/geometry stress cases: low texture and deeper occlusion.
    low = dict(bases[2])
    low["identity"] = "support-stress-low-texture"
    low["low_texture"] = True
    cases.append({"identity": low["identity"], "stratum": "texture_geometry",
                  "base_identity": bases[2]["identity"], "recipe": low})
    layered = dict(bases[5])
    layered["identity"] = "support-stress-multidepth"
    layered["boxes"] = list(layered["boxes"]) + [
        {"lo": [-4.0, 0.0, 2.0], "hi": [-2.0, 4.0, 3.5]},
        {"lo": [0.5, 0.0, 5.0], "hi": [2.5, 3.0, 7.0]},
    ]
    cases.append({"identity": layered["identity"], "stratum": "texture_geometry",
                  "base_identity": bases[5]["identity"], "recipe": layered})

    result = []
    for case in cases:
        result.append({
            "identity": case["identity"],
            "stratum": case["stratum"],
            "base_identity": case["base_identity"],
            "recipe_json": _canonical(case["recipe"]),
        })
    if len(result) != 12 or sum(item["stratum"] != "ordinary" for item in result) != 6:
        raise RuntimeError("frozen_case_registry_count_mismatch")
    if [item["stratum"] for item in result].count("gui_contamination") < 2:
        raise RuntimeError("gui_stress_cohort_incomplete")
    if [item["stratum"] for item in result].count("crop_boundary") < 2:
        raise RuntimeError("crop_stress_cohort_incomplete")
    if [item["stratum"] for item in result].count("texture_geometry") < 2:
        raise RuntimeError("texture_geometry_stress_cohort_incomplete")
    return tuple(result)


def _prior_registered_rgb_hashes() -> dict[str, tuple[str, ...]]:
    root = _project_root()
    pattern = re.compile(rb'"rgb_sha256":"([0-9a-f]{64})"')
    result = {}
    for study, relative in (
            ("CAM-SUPPORT-02", "docs/reports/CAM-SUPPORT-02-2026-10-06.json"),
            ("CAM-SUPPORT-02R1", "docs/reports/CAM-SUPPORT-02R1-2026-10-06.json"),
            ("CAM-SUPPORT-02R2", "docs/reports/CAM-SUPPORT-02R2-2026-10-06.json")):
        data = (root / relative).read_bytes()
        hashes = tuple(match.decode("ascii") for match in pattern.findall(data))
        if len(set(hashes)) != 12:
            raise RuntimeError("camera_support_prior_registry_hash_count_mismatch:" + study)
        result[study] = tuple(sorted(set(hashes)))
    r1a_path = root / "docs/reports/CAM-SUPPORT-02R1A-2026-10-06.json"
    with r1a_path.open("rb") as stream:
        prefix = stream.read(16 * 1024 * 1024)
    hashes = tuple(match.decode("ascii") for match in pattern.findall(prefix)[:12])
    if len(set(hashes)) != 12:
        raise RuntimeError("camera_support_prior_registry_hash_count_mismatch:CAM-SUPPORT-02R1A")
    result["CAM-SUPPORT-02R1A"] = tuple(sorted(set(hashes)))
    return result


def _fresh_r2_scene_recipes(harness) -> tuple[dict[str, object], ...]:
    source = _fresh_scene_recipes(harness)
    ordinary_ids = [row["identity"] for row in source if row["stratum"] == "ordinary"]
    root_index = {identity: index for index, identity in enumerate(ordinary_ids)}
    updated = []
    for row in source:
        recipe = json.loads(row["recipe_json"])
        old_identity = row["identity"]
        parent = row["base_identity"] or old_identity
        index = root_index[parent]
        identity = "support-r2-" + old_identity.removeprefix("support-")
        recipe["identity"] = identity
        recipe["version"] = "CAM-SUPPORT-02R2-ray-world-v3"
        recipe["seed"] = int(recipe["seed"]) + 3_000_019 + index * 137
        recipe["yaw"] = round(float(recipe["yaw"]) + 4.25 + index * 0.17, 4)
        recipe["pitch"] = round(float(recipe["pitch"]) + 1.75 - index * 0.11, 4)
        recipe["focal"] = round(float(recipe["focal"]) + 9.5 + index * 1.25, 4)
        base_identity = (None if row["base_identity"] is None else
                         "support-r2-" + row["base_identity"].removeprefix("support-"))
        updated.append({
            "identity": identity, "stratum": row["stratum"],
            "base_identity": base_identity, "recipe_json": _canonical(recipe),
        })
    if len(updated) != 12 or [row["stratum"] for row in updated] != [row["stratum"] for row in source]:
        raise RuntimeError("camera_support_r2_recipe_registry_mismatch")
    return tuple(updated)


def _fresh_r3_scene_recipes(harness) -> tuple[dict[str, object], ...]:
    source = _fresh_r2_scene_recipes(harness)
    parent_seed = 2026100603
    rng = np.random.Generator(np.random.PCG64(parent_seed))
    ordinary = [json.loads(row["recipe_json"]) for row in source
                if row["stratum"] == "ordinary"]
    ordinary_ids = [row["identity"] for row in source if row["stratum"] == "ordinary"]
    yaw_values = [float(recipe["yaw"]) for recipe in ordinary]
    pitch_values = [float(recipe["pitch"]) for recipe in ordinary]
    focal_values = [float(recipe["focal"]) for recipe in ordinary]
    camera_by_parent = {}
    for index, identity in enumerate(ordinary_ids):
        recipe = ordinary[index]
        camera_by_parent[identity] = {
            "eye": list(recipe["eye"]),
            "yaw": round(float(rng.uniform(min(yaw_values), max(yaw_values))), 4),
            "pitch": round(float(rng.uniform(min(pitch_values), max(pitch_values))), 4),
            "focal": round(float(rng.uniform(min(focal_values), max(focal_values))), 4),
        }
    updated = []
    for row in source:
        recipe = json.loads(row["recipe_json"])
        old_identity = row["identity"]
        parent = row["base_identity"] or old_identity
        identity = "boundary-r3-" + old_identity.removeprefix("support-r2-")
        recipe["identity"] = identity
        recipe["version"] = "CAM-BOUNDARY-SPLIT-03-ray-world-v6"
        recipe["r3_camera_parent_seed"] = parent_seed
        recipe.update(camera_by_parent[parent])
        base_identity = (None if row["base_identity"] is None else
                         "boundary-r3-" + row["base_identity"].removeprefix("support-r2-"))
        updated.append({
            "identity": identity, "stratum": row["stratum"],
            "base_identity": base_identity, "family": parent,
            "recipe_json": _canonical(recipe),
        })
    if (len(updated) != 12
            or [row["stratum"] for row in updated] != [row["stratum"] for row in source]
            or len({row["identity"] for row in updated}) != 12):
        raise RuntimeError("camera_boundary_r3_recipe_registry_mismatch")
    return tuple(updated)


@dataclass(frozen=True, slots=True)
class SegmentInterval:
    """One retained clean interval and its exact parent-line provenance."""

    parent_index: int
    interval_index: int
    t_start: float
    t_end: float
    x0: float
    y0: float
    x1: float
    y1: float
    length: float

    def to_record(self) -> dict[str, int | float]:
        return {
            "parent_index": self.parent_index,
            "interval_index": self.interval_index,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "length": self.length,
        }


@dataclass(frozen=True, slots=True)
class SegmentSupportResult:
    """Candidate segments, provenance, and bounded-work accounting."""

    segments: np.ndarray
    intervals: tuple[SegmentInterval, ...]
    input_count: int
    sample_count: int
    retained_count: int
    minimum_length: float


def mask_transition_band(
    automatic_mask: np.ndarray,
    *,
    max_mask_pixels: int = MAX_BOUNDARY_MASK_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact in-image transition seeds and their fixed 3-pixel band.

    Both pixels in every horizontal/vertical neighboring pair whose automatic
    mask values differ are seeds. The outside of the image is constant false,
    so no imaginary perimeter transition is introduced. A genuine transition
    on a frame edge still marks its in-image edge pixel. Dilation is a fixed
    radius-three Chebyshev square (7x7 all-ones kernel).
    """
    mask = np.asarray(automatic_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size:
        raise ValueError("automatic_mask must be a nonempty 2D boolean array")
    if max_mask_pixels < 1 or mask.size > max_mask_pixels:
        raise ValueError("boundary mask pixel cap exceeded")

    seeds = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    seeds[:, 1:] |= horizontal
    seeds[:, :-1] |= horizontal
    seeds[1:, :] |= vertical
    seeds[:-1, :] |= vertical

    radius = BOUNDARY_DILATION_RADIUS
    horizontal_dilation = seeds.copy()
    for offset in range(1, radius + 1):
        horizontal_dilation[:, offset:] |= seeds[:, :-offset]
        horizontal_dilation[:, :-offset] |= seeds[:, offset:]
    band = horizontal_dilation.copy()
    for offset in range(1, radius + 1):
        band[offset:, :] |= horizontal_dilation[:-offset, :]
        band[:-offset, :] |= horizontal_dilation[offset:, :]
    seeds.setflags(write=False)
    band.setflags(write=False)
    return seeds, band


def split_with_boundary_band(
    segments: np.ndarray,
    automatic_mask: np.ndarray,
    *,
    apply_boundary_band: bool,
    minimum_length: float = MINIMUM_LENGTH,
    sample_spacing: float = SAMPLE_SPACING_PIXELS,
    max_input_segments: int = MAX_INPUT_SEGMENTS,
    max_samples_per_segment: int = MAX_SAMPLES_PER_SEGMENT,
    max_total_samples: int = MAX_TOTAL_SAMPLES,
    max_mask_pixels: int = MAX_BOUNDARY_MASK_PIXELS,
) -> tuple[SegmentSupportResult, dict[str, object]]:
    """Apply A (original mask) or B (original mask plus fixed boundary band)."""
    mask = np.asarray(automatic_mask)
    seeds, band = mask_transition_band(mask, max_mask_pixels=max_mask_pixels)
    expanded = np.logical_or(mask, band)
    exclusion = expanded if apply_boundary_band else mask
    result = split_at_excluded_mask(
        segments, exclusion, minimum_length=minimum_length,
        sample_spacing=sample_spacing, max_input_segments=max_input_segments,
        max_samples_per_segment=max_samples_per_segment,
        max_total_samples=max_total_samples)
    for array in (expanded,):
        array.setflags(write=False)
    return result, {
        "boundary_seeds": seeds,
        "boundary_band": band,
        "expanded_mask": expanded,
        "split_exclusion_mask": exclusion,
        "applied": bool(apply_boundary_band),
    }


def split_at_excluded_mask(
    segments: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    minimum_length: float = MINIMUM_LENGTH,
    sample_spacing: float = SAMPLE_SPACING_PIXELS,
    max_input_segments: int = MAX_INPUT_SEGMENTS,
    max_samples_per_segment: int = MAX_SAMPLES_PER_SEGMENT,
    max_total_samples: int = MAX_TOTAL_SAMPLES,
) -> SegmentSupportResult:
    """Split each line into maximal in-frame, unexcluded sampled intervals.

    Segment coordinates remain in the input image's native pixel frame. The
    source arrays are read-only from this function's perspective. Samples use
    nearest-pixel lookup ``floor(v + .5)`` at spacing no greater than half a
    pixel; out-of-frame samples are excluded. Runs shorter than the existing
    calibration minimum are dropped. Output order is parent index, then the
    interval's source parameter, so detector order and provenance are stable.

    The returned interval endpoints are sampled support endpoints and may be
    inset by at most one sample spacing at a mask transition. The provenance
    records parameter bounds against the unmodified detector segment.
    """
    source = np.asarray(segments)
    mask = np.asarray(excluded_mask)
    if source.ndim != 2 or source.shape[1:] != (4,) or source.dtype.kind not in "fiu":
        raise ValueError("segments must be a numeric (n, 4) array")
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size:
        raise ValueError("excluded_mask must be a nonempty 2D boolean array")
    if len(source) > max_input_segments:
        raise ValueError("segment input cap exceeded")
    if not np.isfinite(source).all():
        raise ValueError("segments must be finite")
    if (not math.isfinite(minimum_length) or minimum_length < 0.0
            or not math.isfinite(sample_spacing) or sample_spacing <= 0.0
            or max_input_segments < 0 or max_samples_per_segment < 2
            or max_total_samples < 0):
        raise ValueError("invalid bounded segment policy")

    height, width = mask.shape
    retained: list[tuple[float, float, float, float]] = []
    intervals: list[SegmentInterval] = []
    used_samples = 0

    for parent_index, row in enumerate(source):
        x0, y0, x1, y1 = (float(value) for value in row)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        sample_count = max(2, int(math.ceil(length / sample_spacing)) + 1)
        if sample_count > max_samples_per_segment:
            raise ValueError("per-segment sample cap exceeded")
        if used_samples + sample_count > max_total_samples:
            raise ValueError("aggregate sample cap exceeded")
        used_samples += sample_count

        parameters = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        xs = x0 + parameters * dx
        ys = y0 + parameters * dy
        columns = np.floor(xs + 0.5).astype(np.int64)
        rows = np.floor(ys + 0.5).astype(np.int64)
        in_frame = ((columns >= 0) & (columns < width)
                    & (rows >= 0) & (rows < height))
        supported = np.zeros(sample_count, dtype=bool)
        valid_indices = np.flatnonzero(in_frame)
        if valid_indices.size:
            supported[valid_indices] = ~mask[rows[valid_indices], columns[valid_indices]]

        starts = np.flatnonzero(supported & np.r_[True, ~supported[:-1]])
        ends = np.flatnonzero(supported & np.r_[~supported[1:], True])
        interval_index = 0
        for start, end in zip(starts, ends):
            if end <= start:
                continue
            t_start = float(parameters[start])
            t_end = float(parameters[end])
            sx0, sy0 = x0 + t_start * dx, y0 + t_start * dy
            sx1, sy1 = x0 + t_end * dx, y0 + t_end * dy
            run_length = math.hypot(sx1 - sx0, sy1 - sy0)
            if run_length < minimum_length:
                continue
            retained.append((sx0, sy0, sx1, sy1))
            intervals.append(SegmentInterval(
                parent_index=parent_index,
                interval_index=interval_index,
                t_start=t_start,
                t_end=t_end,
                x0=sx0,
                y0=sy0,
                x1=sx1,
                y1=sy1,
                length=run_length,
            ))
            interval_index += 1

    output = np.asarray(retained, dtype=source.dtype).reshape((-1, 4))
    output.setflags(write=False)
    return SegmentSupportResult(
        segments=output,
        intervals=tuple(intervals),
        input_count=len(source),
        sample_count=used_samples,
        retained_count=len(intervals),
        minimum_length=float(minimum_length),
    )


def paired_segment_inputs(
    segments: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    minimum_length: float = MINIMUM_LENGTH,
) -> tuple[np.ndarray, SegmentSupportResult]:
    """Return untouched baseline input and masked-split candidate input.

    This helper makes the one-factor contract explicit for the experiment:
    the baseline receives the exact detector array object, and only the
    candidate receives the derived segment array. The mask is never modified.
    """
    baseline = np.asarray(segments)
    candidate = split_at_excluded_mask(
        baseline, excluded_mask, minimum_length=minimum_length)
    return baseline, candidate


def trace_candidate_with_support(
    image: np.ndarray,
    frame_id: str,
    *,
    apply_boundary_band: bool = False,
    record_boundary_masks: bool = False,
    trace_rgb,
    calibration_module,
    gui_mask_module,
) -> tuple[dict[str, object], dict[str, object]]:
    """Call the accepted RGB trace with only its detector output transformed.

    The wrapper delegates mask and LSD calls exactly once, captures the
    automatic mask from that same RGB inference, and substitutes only the
    returned segment array. All module attributes are restored in ``finally``.
    ``trace_rgb`` is the existing bounded call-through harness; dependency
    injection keeps the pure contract testable without importing cv2.
    """
    if not _STUDY_LOCK.acquire(blocking=False):
        raise RuntimeError("camera_support_reentrancy_or_concurrency")
    detector = getattr(calibration_module, "detect_segments", None)
    mask_function = getattr(gui_mask_module, "create_gui_mask", None)
    if not callable(detector) or not callable(mask_function):
        _STUDY_LOCK.release()
        raise TypeError("calibration and GUI mask callables are required")
    captured: dict[str, object] = {"mask_calls": 0, "detector_calls": 0}

    def mask_delegate(height, width, *args, **kwargs):
        mask = mask_function(height, width, *args, **kwargs)
        captured["mask_calls"] = int(captured["mask_calls"]) + 1
        if captured["mask_calls"] != 1:
            raise RuntimeError("automatic_gui_mask_call_count_mismatch")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (height, width):
            raise RuntimeError("automatic_gui_mask_contract_mismatch")
        captured["mask"] = mask.copy()
        return mask

    def detector_delegate(grey, **kwargs):
        raw = detector(grey, **kwargs)
        captured["detector_calls"] = int(captured["detector_calls"]) + 1
        if captured["detector_calls"] != 1:
            raise RuntimeError("LSD_call_count_mismatch")
        mask = captured.get("mask")
        if not isinstance(mask, np.ndarray):
            raise RuntimeError("automatic_gui_mask_missing_before_LSD")
        if record_boundary_masks:
            result, boundary = split_with_boundary_band(
                raw, mask, apply_boundary_band=apply_boundary_band)
        else:
            result = split_at_excluded_mask(raw, mask)
            boundary = None
        raw_snapshot = np.ascontiguousarray(raw).copy()
        raw_snapshot.setflags(write=False)
        captured.update({
            "raw_segments": raw_snapshot,
            "raw_segment_sha256": _sha256(raw_snapshot.tobytes()),
            "raw_segment_dtype": str(raw.dtype),
            "raw_segment_shape": list(raw.shape),
            "mask_sha256": _sha256(np.ascontiguousarray(mask).tobytes()),
            "split": result,
        })
        if boundary is not None:
            captured.update({
                "boundary_seeds": boundary["boundary_seeds"],
                "boundary_band": boundary["boundary_band"],
                "expanded_mask": boundary["expanded_mask"],
                "split_exclusion_mask": boundary["split_exclusion_mask"],
                "boundary_band_sha256": _sha256(boundary["boundary_band"].tobytes()),
                "expanded_mask_sha256": _sha256(boundary["expanded_mask"].tobytes()),
                "split_exclusion_mask_sha256": _sha256(
                    boundary["split_exclusion_mask"].tobytes()),
                "boundary_band_applied": bool(boundary["applied"]),
            })
        return result.segments

    try:
        calibration_module.detect_segments = detector_delegate
        gui_mask_module.create_gui_mask = mask_delegate
        result = trace_rgb(image, frame_id)
        if result.get("status") == "trace_refusal":
            captured["partial_stage_receipt"] = {
                "status": "trace_refusal",
                "reason": str(result.get("reason", "trace_refusal"))[:240],
                "stages": list(result.get("trace", {}).get("stages", []))[:64],
            }
            return result, captured
        if int(captured["mask_calls"]) != 1 or int(captured["detector_calls"]) != 1:
            raise RuntimeError("candidate_support_stage_call_count_mismatch")
        return result, captured
    finally:
        calibration_module.detect_segments = detector
        gui_mask_module.create_gui_mask = mask_function
        _STUDY_LOCK.release()


def _case_objects(harness, registry):
    return tuple(harness.CameraRecipe(
        identity=row["identity"], physical=True, recipe_json=row["recipe_json"])
        for row in registry)


def _expected_dependencies(harness) -> dict[str, object]:
    pins = harness.dependency_pins()
    source_hashes = dict(pins["source_sha256"])
    source_hashes["scripts/evaluation/run_camera_mixed_depth.py"] = _sha256(
        (_project_root() / "scripts/evaluation/run_camera_mixed_depth.py").read_bytes())
    if source_hashes != _EXPECTED_DEPENDENCY_SHA256:
        raise RuntimeError("camera_support_source_pin_mismatch")
    if pins["runtime"] != _EXPECTED_RUNTIME:
        raise RuntimeError("camera_support_runtime_pin_mismatch")
    runtime = _runtime_identity()
    own_sources = {
        "src/mc_photogrammetry/reconstruction/evaluation_gui_masked_segments.py": _sha256(
            Path(__file__).read_bytes()),
        "tests/test_evaluation_gui_masked_segments.py": _sha256(
            (_project_root() / "tests/test_evaluation_gui_masked_segments.py").read_bytes()),
        "split_at_excluded_mask": _sha256(inspect.getsource(split_at_excluded_mask)),
        "mask_transition_band": _sha256(inspect.getsource(mask_transition_band)),
        "split_with_boundary_band": _sha256(inspect.getsource(split_with_boundary_band)),
        "trace_candidate_with_support": _sha256(inspect.getsource(trace_candidate_with_support)),
        "_fresh_scene_recipes": _sha256(inspect.getsource(_fresh_scene_recipes)),
        "_fresh_r3_scene_recipes": _sha256(inspect.getsource(_fresh_r3_scene_recipes)),
        "run_study": _sha256(inspect.getsource(run_study)),
    }
    return {"source_sha256": source_hashes, "owned_source_sha256": own_sources,
            "runtime": runtime}


def preregister() -> dict[str, object]:
    """Freeze the new twelve-case stimulus registry before any inference."""
    from mc_photogrammetry.vision.gui_mask import create_gui_mask
    harness = _study_harness()
    dependencies = _expected_dependencies(harness)
    registry = _fresh_scene_recipes(harness)
    scenes = _case_objects(harness, registry)
    rgb_hashes = {}
    rendered_rgb = {}
    recipe_hashes = {}
    for scene, row in zip(scenes, registry):
        rgb = harness.render_rgb(scene)
        rgb_hashes[scene.identity] = _sha256(rgb.tobytes())
        rendered_rgb[scene.identity] = rgb
        recipe_hashes[scene.identity] = _sha256(scene.recipe_json)
    if len(set(rgb_hashes.values())) != len(registry):
        raise RuntimeError("camera_support_duplicate_rgb_in_registered_cases")
    for row in registry:
        if row["stratum"] == "gui_contamination":
            base = rendered_rgb[row["base_identity"]]
            contaminated = rendered_rgb[row["identity"]]
            mask = create_gui_mask(*base.shape[:2])
            if (base.shape != contaminated.shape
                    or not np.array_equal(base[~mask], contaminated[~mask])
                    or np.array_equal(base[mask], contaminated[mask])):
                raise RuntimeError("gui_contamination_stimulus_contract_mismatch")
        elif row["stratum"] == "crop_boundary":
            base = rendered_rgb[row["base_identity"]]
            crop = json.loads(row["recipe_json"])["crop"]
            x, y, width, height = crop
            expected_crop = base[y:y + height, x:x + width]
            if not np.array_equal(expected_crop, rendered_rgb[row["identity"]]):
                raise RuntimeError("native_crop_stimulus_contract_mismatch")
    cases = []
    for row in registry:
        cases.append({
            "identity": row["identity"],
            "stratum": row["stratum"],
            "base_identity": row["base_identity"],
            "recipe_json": row["recipe_json"],
            "recipe_sha256": recipe_hashes[row["identity"]],
            "rgb_sha256": rgb_hashes[row["identity"]],
        })
    plan = {
        "schema": "mc-photogrammetry-cam-support-02r1-v1",
        "execution_state": "NOT_STARTED",
        "study": "CAM-SUPPORT-02R1",
        "dependencies": dependencies,
        "case_registry": cases,
        "policy": {
            "intervention": "split LSD segments at automatic GUI-excluded mask intervals",
            "baseline": "unchanged RGB -> automatic GUI mask -> zero-fill -> enhancement -> LSD -> calibration",
            "candidate": "same path; replace only detected segment array with clean long in-frame intervals",
            "minimum_segment_length_pixels": MINIMUM_LENGTH,
            "sample_spacing_pixels": SAMPLE_SPACING_PIXELS,
            "max_input_segments": MAX_INPUT_SEGMENTS,
            "max_samples_per_segment": MAX_SAMPLES_PER_SEGMENT,
            "max_total_samples": MAX_TOTAL_SAMPLES,
            "camera_gates": {"focal_relative_error": 0.05,
                             "orientation_degrees_mod90": 1.0,
                             "principal_error_over_diagonal": 0.01},
            "success": {"candidate_passes_min": 9, "net_added_passes_min": 3,
                        "baseline_passes_lost_max": 0,
                        "newly_wrong_stress_cases_max": 0,
                        "candidate_stress_wrong_count_le_baseline": True},
            "no_sweep_or_retry": True,
            "truth_order": "seal all 24 full trace payloads before evaluator scoring",
            "maximum_result_json_bytes": MAX_STUDY_JSON_BYTES,
        },
        "preregistration_sha256": None,
        "sealed_predictions": [],
        "evaluations": None,
        "summary": None,
    }
    plan["preregistration_sha256"] = _sha256(_canonical({
        key: value for key, value in plan.items() if key != "preregistration_sha256"}))
    return plan


def write_r1_preregistration() -> dict[str, object]:
    """Freeze the authorized R1 registry without opening renderer truth."""
    report_path, json_path = _r1_paths()
    if report_path.exists() != json_path.exists():
        raise RuntimeError("camera_support_incomplete_r1_registry")
    if report_path.exists():
        old_r1 = json.loads(json_path.read_text(encoding="utf-8"))
        if (old_r1.get("execution_state") != "NOT_STARTED"
                or old_r1.get("sealed_predictions")
                or old_r1.get("evaluations") is not None):
            raise RuntimeError("camera_support_r1_registry_already_consumed")
    old_plan = json.loads(_study_paths()[1].read_text(encoding="utf-8"))
    old_rgb_hashes = {case["rgb_sha256"] for case in old_plan["case_registry"]}
    plan = preregister()
    new_rgb_hashes = {case["rgb_sha256"] for case in plan["case_registry"]}
    if len(new_rgb_hashes) != 12 or new_rgb_hashes & old_rgb_hashes:
        raise RuntimeError("camera_support_r1_rgb_not_fresh")
    counts = {name: sum(case["stratum"] == name for case in plan["case_registry"])
              for name in ("ordinary", "gui_contamination", "crop_boundary", "texture_geometry")}
    if counts != {"ordinary": 6, "gui_contamination": 2,
                  "crop_boundary": 2, "texture_geometry": 2}:
        raise RuntimeError("camera_support_r1_strata_mismatch")
    plan["freshness_gate"] = {
        "failed_registry_preregistration_sha256": old_plan["preregistration_sha256"],
        "failed_registry_case_count": len(old_rgb_hashes),
        "r1_unique_rgb_count": len(new_rgb_hashes),
        "intersection_count": 0,
    }
    plan["preregistration_sha256"] = _sha256(_canonical({
        key: value for key, value in plan.items() if key != "preregistration_sha256"}))
    report = (
        "# CAM-SUPPORT-02R1 evaluation-only masked segment support\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "R1 is a fresh successor cohort after CAM-SUPPORT-02 was permanently marked FAILED. "
        "The failed report and JSON remain unchanged. All 12 R1 RGB digests are distinct from "
        "the failed registry. This record freezes recipes, rendering, strata, case order, "
        "source/runtime/test pins, and the existing intervention and five acceptance criteria.\n\n"
        "No camera truth values are included in this preregistration. No inference or scoring "
        "has run. Execution remains blocked pending Astra review and explicit GO.\n\n"
        "The one-factor candidate still splits detected LSD segments at the default automatic "
        "GUI-exclusion mask. The baseline path, camera gates, five acceptance criteria, and "
        "production behavior are unchanged.\n"
    )
    encoded = (_canonical(plan) + "\n").encode("utf-8")
    with json_path.open("wb" if json_path.exists() else "xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("wb" if report_path.exists() else "xb") as stream:
        stream.write(report.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def write_r1a_preregistration() -> dict[str, object]:
    """Freeze exact R1 images/recipes under the repaired runner contract."""
    report_path, json_path = _r1a_paths()
    if report_path.exists() != json_path.exists():
        raise RuntimeError("camera_support_incomplete_r1a_registry")
    if report_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if (existing.get("execution_state") != "NOT_STARTED"
                or existing.get("sealed_predictions")
                or existing.get("evaluations") is not None):
            raise RuntimeError("camera_support_r1a_registry_already_consumed")
    prior = json.loads(_r1_paths()[1].read_text(encoding="utf-8"))
    if prior.get("execution_state") != "NOT_STARTED" or prior.get("sealed_predictions"):
        raise RuntimeError("camera_support_r1a_source_r1_not_unstarted")
    harness = _study_harness()
    dependencies = _expected_dependencies(harness)
    renderer_registry = []
    for row in prior["case_registry"]:
        scene = harness.CameraRecipe(
            identity=row["identity"], physical=True, recipe_json=row["recipe_json"])
        rgb = np.ascontiguousarray(harness.render_rgb(scene), dtype=np.uint8)
        rgb_sha256 = _sha256(rgb.tobytes())
        if rgb_sha256 != row["rgb_sha256"]:
            raise RuntimeError("camera_support_r1a_rgb_changed")
        renderer_registry.append({
            "identity": row["identity"], "stratum": row["stratum"],
            "base_identity": row["base_identity"],
            "recipe_json": row["recipe_json"],
            "recipe_sha256": row["recipe_sha256"],
            "rgb_sha256": rgb_sha256,
            "rgb_shape": list(rgb.shape),
            "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
        })
    if [row["rgb_sha256"] for row in renderer_registry] != [
            row["rgb_sha256"] for row in prior["case_registry"]]:
        raise RuntimeError("camera_support_r1a_rgb_registry_order_mismatch")
    truth_registry = [{
        "identity": row["identity"], "stratum": row["stratum"],
        "base_identity": row["base_identity"],
        "renderer_recipe_sha256": row["recipe_sha256"],
        "truth_loader": "renderer CameraRecipe/truth_camera; instantiate after validated 24-seal marker",
    } for row in renderer_registry]
    plan = {
        "schema": "mc-photogrammetry-cam-support-02r1a-v1",
        "study": "CAM-SUPPORT-02R1A",
        "execution_state": "NOT_STARTED",
        "supersedes": {
            "study": "CAM-SUPPORT-02R1",
            "status": "superseded before inference",
            "reason": "Astra preflight identified runner/order and adapter-boundary defects.",
        },
        "dependencies": dependencies,
        "renderer_registry": renderer_registry,
        "evaluator_truth_registry": truth_registry,
        "policy": prior["policy"],
        "freshness_gate": {
            "source_r1_preregistration_sha256": prior["preregistration_sha256"],
            "exact_rgb_hashes_preserved": True,
            "exact_recipes_and_strata_preserved": True,
        },
        "sealed_predictions": [], "parity_diagnostics": [],
        "seal_validation": None, "evaluations": None, "summary": None,
        "preregistration_sha256": None,
    }
    plan["preregistration_sha256"] = _sha256(_canonical({
        key: value for key, value in plan.items() if key != "preregistration_sha256"}))
    report = (
        "# CAM-SUPPORT-02R1A evaluation-only masked segment support\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "R1A supersedes R1 before inference and freezes the same 12 R1 RGB pixels, recipes, "
        "strata, case order, runtime, detector treatment, thresholds and five acceptance gates. "
        "RGB bytes are embedded and hash-checked so the attempt never rebuilds renderer cameras "
        "before scoring.\n\n"
        "Recipe values include camera/world parameters used only as renderer/evaluator metadata; "
        "they are present in `renderer_registry`, not absent. The image-only arm adapter accepts "
        "only a read-only RGB array and frame ID. The separate evaluator truth registry contains "
        "identity-to-recipe references only; evaluator camera objects are reconstructed after all "
        "24 arm outputs are persisted and their seals validated. No truth is available to an arm.\n\n"
        "No inference or scoring has run. Execution awaits Astra review and explicit GO.\n"
    )
    with json_path.open("wb" if json_path.exists() else "xb") as stream:
        stream.write((_canonical(plan) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("wb" if report_path.exists() else "xb") as stream:
        stream.write(report.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def write_r2_preregistration() -> dict[str, object]:
    """Freeze a fresh disjoint cohort and create its empty append-only sidecar."""
    report_path, json_path, sidecar_path = _r2_paths()
    if report_path.exists() or json_path.exists() or sidecar_path.exists():
        raise RuntimeError("camera_support_r2_preregistration_target_exists")
    prior_sets = _prior_registered_rgb_hashes()
    prior_hashes = {digest for hashes in prior_sets.values() for digest in hashes}
    harness = _study_harness()
    dependencies = _expected_dependencies(harness)
    registry = _fresh_r2_scene_recipes(harness)
    renderer_registry = []
    for row in registry:
        scene = harness.CameraRecipe(identity=row["identity"], physical=True,
                                     recipe_json=row["recipe_json"])
        rgb = np.ascontiguousarray(harness.render_rgb(scene), dtype=np.uint8)
        digest = _sha256(rgb.tobytes())
        if digest in prior_hashes:
            raise RuntimeError("camera_support_r2_rgb_reuses_prior_pixel_hash")
        renderer_registry.append({
            **row, "recipe_sha256": _sha256(row["recipe_json"]),
            "rgb_sha256": digest, "rgb_shape": list(rgb.shape),
            "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
        })
    current_hashes = [row["rgb_sha256"] for row in renderer_registry]
    if len(set(current_hashes)) != 12 or set(current_hashes) & prior_hashes:
        raise RuntimeError("camera_support_r2_rgb_not_disjoint")
    expected_strata = {"ordinary": 6, "gui_contamination": 2,
                       "crop_boundary": 2, "texture_geometry": 2}
    strata = {name: sum(row["stratum"] == name for row in renderer_registry)
              for name in expected_strata}
    if strata != expected_strata:
        raise RuntimeError("camera_support_r2_strata_mismatch")
    truth_registry = [{
        "identity": row["identity"], "stratum": row["stratum"],
        "base_identity": row["base_identity"],
        "renderer_recipe_sha256": row["recipe_sha256"],
        "truth_loader": "instantiate renderer CameraRecipe/truth_camera after validated 24-member barrier",
    } for row in renderer_registry]
    r1a_json_sha256 = _file_sha256(_r1a_paths()[1])
    plan = {
        "schema": "mc-photogrammetry-cam-support-02r2-v1",
        "study": "CAM-SUPPORT-02R2",
        "execution_state": "NOT_STARTED",
        "supersedes": {"study": "CAM-SUPPORT-02R1A", "status": "FAILED; preserve permanently",
                       "reason": "R1A exceeded inline JSON result cap after 14 sealed arms."},
        "dependencies": dependencies,
        "renderer_registry": renderer_registry,
        "evaluator_truth_registry": truth_registry,
        "prior_rgb_exclusion": {
            "counts": {name: len(values) for name, values in prior_sets.items()},
            "hashes_sha256": {name: _sha256(_canonical(sorted(values)))
                               for name, values in prior_sets.items()},
            "total_prior_rgb_count": len(prior_hashes),
            "intersection_count": 0,
            "r1a_failed_json_sha256": r1a_json_sha256,
        },
        "policy": json.loads((_r1_paths()[1]).read_text(encoding="utf-8"))["policy"],
        "payload_storage": {
            "format": "concatenated-independent-gzip-members",
            "sidecar": sidecar_path.name,
            "max_members": MAX_PAYLOAD_MEMBERS,
            "max_compressed_bytes_per_member": MAX_COMPRESSED_MEMBER_BYTES,
            "max_decoded_bytes_per_member": MAX_DECODED_MEMBER_BYTES,
            "aggregate_bytes_including_json_sidecar_rgb_metadata": MAX_STUDY_TOTAL_BYTES,
            "reserved_bytes": STUDY_RESERVE_BYTES,
            "lossless_canonical_payload_digest_semantics": True,
            "append_only_no_truncate_resume_or_retry": True,
        },
        "sealed_predictions": [], "parity_diagnostics": [],
        "seal_validation": None, "evaluations": None, "summary": None,
        "preregistration_sha256": None,
    }
    plan["preregistration_sha256"] = _sha256(_canonical({
        key: value for key, value in plan.items() if key != "preregistration_sha256"}))
    encoded = (_canonical(plan) + "\n").encode("utf-8")
    if len(encoded) + STUDY_RESERVE_BYTES > MAX_STUDY_TOTAL_BYTES:
        raise RuntimeError("camera_support_r2_preregistration_aggregate_cap")
    report = (
        "# CAM-SUPPORT-02R2 evaluation-only masked segment support\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "R2 is a fresh successor to the permanently FAILED R1A attempt. Its 12 newly rendered "
        "RGB pixel hashes are disjoint from all 36 registered RGB hashes across R0, R1 and R1A. "
        "The same treatment, runtime, thresholds, acceptance gates, and six ordinary/six stress "
        "strata are retained. R2 RGB bytes, recipes, order, source pins and seeds are frozen in JSON.\n\n"
        "Prediction payloads will use independent lossless gzip members in the adjacent append-only "
        "sidecar. Each descriptor is published only after append, fsync, bounded readback, gzip CRC, "
        "canonical payload hash and length checks. Aggregate storage includes JSON, sidecar, embedded "
        "RGB, descriptors and metadata, leaving the preregistered 16 MiB reserve. No resume, truncate, "
        "reset, retry or scoring after failure is permitted.\n\n"
        "Camera/world recipe values are renderer/evaluator metadata and are not adapter inputs. The "
        "image-only arms accept immutable RGB plus frame ID. Evaluator camera reconstruction is gated "
        "behind all 24 validated member descriptors. No inference or scoring has run; Astra review "
        "and explicit GO are required before any attempt.\n"
    )
    with sidecar_path.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with json_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("xb") as stream:
        stream.write(report.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def write_r3_preregistration() -> dict[str, object]:
    """Freeze a fresh R3 cohort for a single fixed boundary-band ablation."""
    report_path, json_path, sidecar_path = _r3_paths()
    if report_path.exists() or json_path.exists() or sidecar_path.exists():
        raise RuntimeError("camera_boundary_r3_preregistration_target_exists")
    references = _gui_border_reference_pins()
    prior_sets = _prior_registered_rgb_hashes()
    prior_hashes = {digest for values in prior_sets.values() for digest in values}
    harness = _study_harness()
    dependencies = _expected_dependencies(harness)
    registry = _fresh_r3_scene_recipes(harness)
    renderer_registry = []
    for row in registry:
        scene = harness.CameraRecipe(identity=row["identity"], physical=True,
                                     recipe_json=row["recipe_json"])
        rgb = np.ascontiguousarray(harness.render_rgb(scene), dtype=np.uint8)
        digest = _sha256(rgb.tobytes())
        renderer_registry.append({
            **row, "recipe_sha256": _sha256(row["recipe_json"]),
            "rgb_sha256": digest, "rgb_shape": list(rgb.shape),
            "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
        })
    rgb_hashes = [row["rgb_sha256"] for row in renderer_registry]
    if len(set(rgb_hashes)) != 12 or set(rgb_hashes) & prior_hashes:
        raise RuntimeError("camera_boundary_r3_rgb_not_disjoint")
    expected_strata = {"ordinary": 6, "gui_contamination": 2,
                       "crop_boundary": 2, "texture_geometry": 2}
    strata = {name: sum(row["stratum"] == name for row in renderer_registry)
              for name in expected_strata}
    if strata != expected_strata:
        raise RuntimeError("camera_boundary_r3_strata_mismatch")
    prior = json.loads(_r2_paths()[1].read_text(encoding="utf-8"))
    policy = dict(prior["policy"])
    policy.update({
        "intervention": "change only segment-split exclusion mask by adding a fixed transition band",
        "baseline": "A: original automatic mask for zero-fill/enhancement/LSD; R2 splitter excludes original mask",
        "candidate": "B: identical upstream RGB/mask/LSD; R2 splitter excludes original mask OR fixed band",
        "band_definition": {
            "pairs": "horizontal and vertical adjacent in-image pixels with differing automatic-mask values",
            "seeds": "mark both pixels in every differing pair; no comparison against outside-image pixels",
            "dilation": "7x7 all-ones kernel; fixed radius-3 Chebyshev dilation",
            "exterior": "constant zero",
            "edge_rule": "genuine transition touching a frame edge/corner remains a seed",
            "combine": "expanded_mask = automatic_mask OR dilated_transition_band",
            "upstream_isolation": "expanded mask is used only by the segment splitter, never by zero-fill, enhancement, or LSD",
        },
        "arm_assignment": {
            "baseline": "A: unchanged R2 segment splitter",
            "candidate": "B: unchanged R2 splitter with expanded exclusion mask",
            "comparison": "B versus A on the same frozen RGB, family and automatic mask",
        },
        "cohort_generation": {
            "rule": "one deterministic parent-seed draw of family eye/focal/yaw/pitch within R2 camera ranges; stress variants inherit their paired ordinary camera",
            "parent_seed": 2026100603,
            "eye_distribution_note": "R2 has one eye value; eye is preserved",
            "camera_ranges_from_r2": {
                "yaw": [min(float(json.loads(row["recipe_json"])["yaw"])
                            for row in _fresh_r2_scene_recipes(harness)
                            if row["stratum"] == "ordinary"),
                         max(float(json.loads(row["recipe_json"])["yaw"])
                            for row in _fresh_r2_scene_recipes(harness)
                            if row["stratum"] == "ordinary")],
                "pitch": [min(float(json.loads(row["recipe_json"])["pitch"])
                              for row in _fresh_r2_scene_recipes(harness)
                              if row["stratum"] == "ordinary"),
                           max(float(json.loads(row["recipe_json"])["pitch"])
                              for row in _fresh_r2_scene_recipes(harness)
                              if row["stratum"] == "ordinary")],
                "focal": [min(float(json.loads(row["recipe_json"])["focal"])
                              for row in _fresh_r2_scene_recipes(harness)
                              if row["stratum"] == "ordinary"),
                           max(float(json.loads(row["recipe_json"])["focal"])
                              for row in _fresh_r2_scene_recipes(harness)
                              if row["stratum"] == "ordinary")],
            },
        },
        "treatment_reference": references,
        "refusal_is_accuracy_gain": False,
        "family_cluster_unit": "ordinary scene family; each stress row retains its ordinary base_identity",
    })
    truth_registry = [{
        "identity": row["identity"], "stratum": row["stratum"],
        "base_identity": row["base_identity"], "family": row["family"],
        "renderer_recipe_sha256": row["recipe_sha256"],
        "truth_loader": "renderer CameraRecipe/truth_camera after validated 24-member barrier",
    } for row in renderer_registry]
    plan = {
        "schema": "mc-photogrammetry-cam-boundary-split-03-v2",
        "study": "CAM-BOUNDARY-SPLIT-03",
        "execution_state": "NOT_STARTED",
        "supersedes": {"study": "CAM-SUPPORT-02R2", "status": "COMPLETE; frozen cohort consumed",
                       "reason": "New confirmatory cohort isolates a fixed boundary-band extension to the R2 splitter."},
        "dependencies": dependencies,
        "treatment_reference": references,
        "renderer_registry": renderer_registry,
        "evaluator_truth_registry": truth_registry,
        "prior_rgb_exclusion": {
            "registry_entry_count": sum(len(values) for values in prior_sets.values()),
            "distinct_prior_rgb_count": len(prior_hashes),
            "counts_by_study": {name: len(values) for name, values in prior_sets.items()},
            "hashes_sha256_by_study": {name: _sha256(_canonical(sorted(values)))
                                        for name, values in prior_sets.items()},
            "r2_terminal_json_sha256": _file_sha256(_r2_paths()[1]),
            "intersection_count": 0,
        },
        "policy": policy,
        "payload_storage": {
            "format": "concatenated-independent-gzip-members",
            "sidecar": sidecar_path.name,
            "max_members": MAX_PAYLOAD_MEMBERS,
            "max_compressed_bytes_per_member": MAX_COMPRESSED_MEMBER_BYTES,
            "max_decoded_bytes_per_member": MAX_DECODED_MEMBER_BYTES,
            "aggregate_bytes_including_json_sidecar_rgb_metadata": MAX_STUDY_TOTAL_BYTES,
            "reserved_bytes": STUDY_RESERVE_BYTES,
            "lossless_canonical_payload_digest_semantics": True,
            "append_only_no_truncate_resume_or_retry": True,
        },
        "sealed_predictions": [], "parity_diagnostics": [],
        "seal_validation": None, "evaluations": None, "summary": None,
        "preregistration_sha256": None,
    }
    plan["preregistration_sha256"] = _sha256(_canonical({
        key: value for key, value in plan.items() if key != "preregistration_sha256"}))
    encoded = (_canonical(plan) + "\n").encode("utf-8")
    if len(encoded) + STUDY_RESERVE_BYTES > MAX_STUDY_TOTAL_BYTES:
        raise RuntimeError("camera_boundary_r3_preregistration_aggregate_cap")
    report = (
        "# CAM-BOUNDARY-SPLIT-03 evaluation-only A/B study\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "A is the unchanged R2 segment splitter. B changes only its exclusion mask: for each in-image "
        "horizontal or vertical adjacent pair with differing automatic-mask values, mark both pixels; "
        "dilate seeds with a fixed 7x7 square (radius-three Chebyshev) and constant-zero exterior; union "
        "the band with the original automatic mask. Genuine edge transitions remain seeds; no outside-image "
        "transition is invented. The expanded mask is passed only to segment splitting and never to the "
        "zero-fill, enhancement, or LSD path. The band implementation and report are pinned in JSON.\n\n"
        "The cohort contains six ordinary and six stress cases (two GUI-boundary, two crop-boundary, two "
        "texture/geometry), with 12 unique RGB hashes disjoint from all R0/R1/R1A/R2 registry entries. "
        "Base-family labels, order, seeds, recipes, source/runtime pins, RGB bytes and truth references "
        "are frozen in JSON. The whole cohort uses one deterministic parent-seed draw for family camera "
        "values within R2 ranges; paired stress cases inherit their base camera, while R2's singleton eye "
        "value is preserved. The original automatic mask remains the only mask supplied upstream.\n\n"
        "The accepted bounded gzip-member store and truth-isolated one-shot runner are reused. The unchanged "
        "five gates compare B to A; refusals are not accuracy recoveries. No inference or scoring has run. "
        "Astra review and separate lead GO are required before execution. No production integration or product "
        "accuracy claim is authorized.\n"
    )
    with sidecar_path.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with json_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("xb") as stream:
        stream.write(report.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def write_preregistration() -> dict[str, object]:
    """Create the two assigned preregistration artifacts exclusively."""
    report_path, json_path = _active_study_paths()
    if report_path.exists() or json_path.exists():
        raise RuntimeError("camera_support_preregistration_target_already_exists")
    plan = preregister()
    report = (
        "# CAM-SUPPORT-02 evaluation-only masked segment support\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "This study changes only the segment array presented to the existing camera estimator. "
        "Inference receives each RGB image and its default automatic GUI mask; renderer camera "
        "and geometry are kept for scoring after all baseline and candidate traces are sealed. "
        "No production calibration, threshold, camera gate, or default behavior changes.\n\n"
        "The registry contains six ordinary structural cases and six stress cases: two GUI-only "
        "contamination, two crop-boundary, and two texture/geometry cases. Inputs are frozen in "
        "the adjacent JSON artifact, together with runtime/source pins, RGB hashes, the full "
        "paired evaluation protocol, and the unchanged camera gates.\n\n"
        "No cases have been scored at preregistration. Existing CAM-MD-01, GUI-BORDER-01, and "
        "ANGULAR-OBJECTIVE-01 cases are excluded.\n"
    )
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(plan) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def refresh_unstarted_preregistration() -> dict[str, object]:
    """Replace a preregistration only while still NOT_STARTED and unscored."""
    report_path, json_path = _active_study_paths()
    if not report_path.is_file() or not json_path.is_file():
        raise RuntimeError("camera_support_preregistration_missing")
    old = json.loads(json_path.read_text(encoding="utf-8"))
    if (old.get("execution_state") != "NOT_STARTED"
            or old.get("sealed_predictions")
            or old.get("evaluations") is not None):
        raise RuntimeError("camera_support_preregistration_already_consumed")
    plan = preregister()
    report = (
        "# CAM-SUPPORT-02 evaluation-only masked segment support\n\n"
        "CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED\n\n"
        f"Preregistration SHA-256: `{plan['preregistration_sha256']}`.\n\n"
        "Pre-inference revision: bind the splitter, trace wrapper, runner, focused tests, "
        "automatic-mask-only control and native-crop stimulus parity into the frozen record. "
        "No inference or scoring has occurred.\n\n"
        "The treatment changes only the detected line-segment array. Inference receives RGB "
        "and the default automatic GUI mask; renderer camera and geometry are accessed only "
        "after all baseline/candidate prediction traces have been sealed. No production "
        "calibration, threshold, camera gate or default behavior changes.\n\n"
        "The 12 unique RGBs comprise six ordinary structural cases and six stress cases: two "
        "GUI-only contamination, two native crops and two texture/geometry cases. Existing "
        "CAM-MD-01, GUI-BORDER-01 and ANGULAR-OBJECTIVE-01 cases are excluded.\n"
    )
    encoded = (_canonical(plan) + "\n").encode("utf-8")
    with json_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(encoded)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    with report_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(report.encode("utf-8"))
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return plan


def _save_plan(path: Path, plan: dict[str, object]) -> None:
    encoded = (_canonical(plan) + "\n").encode("utf-8")
    if len(encoded) > MAX_STUDY_JSON_BYTES:
        raise RuntimeError("camera_support_result_byte_cap")
    r2_report, r2_json, r2_sidecar = _r2_paths()
    r3_report, r3_json, r3_sidecar = _r3_paths()
    if Path(path) == r2_json or Path(path) == r3_json:
        sidecar_path = r3_sidecar if Path(path) == r3_json else r2_sidecar
        sidecar_bytes = sidecar_path.stat().st_size if sidecar_path.exists() else 0
        if len(encoded) + sidecar_bytes + STUDY_RESERVE_BYTES > MAX_STUDY_TOTAL_BYTES:
            raise RuntimeError("camera_support_result_byte_cap")
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(encoded)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def _prediction_payload(identity: str, arm: str, rgb_sha256: str,
                        rgb_shape: tuple[int, ...], traced: dict[str, object],
                        support: dict[str, object] | None,
                        preregistration_sha256: str, recipe_sha256: str) -> dict[str, object]:
    if support is None:
        support_payload = None
    elif "split" not in support:
        partial = support.get("partial_stage_receipt", {})
        support_payload = {"partial_stage_receipt": partial}
    else:
        support_payload = {
            "mask_sha256": support["mask_sha256"],
            "raw_segment_sha256": support["raw_segment_sha256"],
            "raw_segment_dtype": canonical_dtype(support["raw_segment_dtype"]),
            "raw_segment_shape": support["raw_segment_shape"],
            "split": {
                "input_count": support["split"].input_count,
                "sample_count": support["split"].sample_count,
                "retained_count": support["split"].retained_count,
                "minimum_length": support["split"].minimum_length,
                "intervals": [item.to_record() for item in support["split"].intervals],
                "output_sha256": _sha256(support["split"].segments.tobytes()),
            },
        }
        for key in ("boundary_band_sha256", "expanded_mask_sha256",
                    "split_exclusion_mask_sha256", "boundary_band_applied"):
            if key in support:
                support_payload[key] = support[key]
    return {
        "identity": identity,
        "frame_id": identity,
        "recipe_sha256": recipe_sha256,
        "arm": arm,
        "preregistration_sha256": preregistration_sha256,
        "rgb_sha256": rgb_sha256,
        "rgb_shape": list(rgb_shape),
        "trace_result": traced,
        "support": support_payload,
    }


def _extract_prediction(traced: dict[str, object]) -> dict[str, object] | None:
    value = traced.get("prediction")
    return value if isinstance(value, dict) else None


def _require_image_only_input(rgb: np.ndarray, frame_id: str) -> None:
    if rgb.flags.writeable:
        raise ValueError("camera_support_adapter_requires_readonly_rgb")


def _run_baseline_arm(rgb: np.ndarray, frame_id: str) -> dict[str, object]:
    """Baseline arm API is exactly immutable RGB plus a frame ID."""
    _require_image_only_input(rgb, frame_id)
    harness = _study_harness()
    rgb_sha256 = _sha256(np.ascontiguousarray(rgb).tobytes())
    try:
        traced = harness.trace_rgb(rgb, frame_id)
    except Exception as exc:
        raise TraceReceiptError(str(exc), getattr(exc, "partial_receipt", {}),
                                rgb_sha256, list(rgb.shape)) from exc
    return {"trace_result": _bounded_trace_result(traced), "support": None,
            "adapter_rgb_sha256": rgb_sha256}


def _run_candidate_arm(rgb: np.ndarray, frame_id: str) -> dict[str, object]:
    """Candidate arm API is exactly immutable RGB plus a frame ID."""
    return _run_split_arm(rgb, frame_id, apply_boundary_band=False,
                          record_boundary_masks=False)


def _run_split_arm(rgb: np.ndarray, frame_id: str, *, apply_boundary_band: bool,
                   record_boundary_masks: bool) -> dict[str, object]:
    """Image-only adapter shared by the R2 candidate and R3 A/B arms."""
    _require_image_only_input(rgb, frame_id)
    harness = _study_harness()
    rgb_sha256 = _sha256(np.ascontiguousarray(rgb).tobytes())
    from mc_photogrammetry.reconstruction import calibration
    from mc_photogrammetry.vision import gui_mask
    try:
        traced, support = trace_candidate_with_support(
            rgb, frame_id, apply_boundary_band=apply_boundary_band,
            record_boundary_masks=record_boundary_masks,
            trace_rgb=harness.trace_rgb,
            calibration_module=calibration, gui_mask_module=gui_mask)
    except Exception as exc:
        raise TraceReceiptError(str(exc), getattr(exc, "partial_receipt", {}),
                                rgb_sha256, list(rgb.shape)) from exc
    return {"trace_result": _bounded_trace_result(traced), "support": support,
            "adapter_rgb_sha256": rgb_sha256}


def _run_r3_a_arm(rgb: np.ndarray, frame_id: str) -> dict[str, object]:
    """R3 arm A: unchanged R2 splitter, with boundary diagnostics recorded."""
    return _run_split_arm(rgb, frame_id, apply_boundary_band=False,
                          record_boundary_masks=True)


def _run_r3_b_arm(rgb: np.ndarray, frame_id: str) -> dict[str, object]:
    """R3 arm B: R2 splitter with the frozen fixed boundary band excluded."""
    return _run_split_arm(rgb, frame_id, apply_boundary_band=True,
                          record_boundary_masks=True)


def _bounded_trace_result(traced: dict[str, object]) -> dict[str, object]:
    if traced.get("status") != "trace_refusal":
        return traced
    trace = traced.get("trace") if isinstance(traced.get("trace"), dict) else {}
    stages = trace.get("stages", [])
    bounded = _bounded_exception_receipt(TraceReceiptError(
        str(traced.get("reason", "trace_refusal")),
        {"stages": stages if isinstance(stages, list) else []},
        traced.get("rgb_sha256"), traced.get("rgb_shape")))
    bounded["rgb_sha256"] = traced.get("rgb_sha256")
    bounded["rgb_shape"] = traced.get("rgb_shape")
    return bounded


class TraceReceiptError(RuntimeError):
    def __init__(self, reason, partial_receipt, rgb_sha256=None, rgb_shape=None):
        self.partial_receipt = partial_receipt
        self.rgb_sha256 = rgb_sha256
        self.rgb_shape = rgb_shape
        super().__init__(reason)


def _frozen_rgb(row: dict[str, object]) -> np.ndarray:
    raw = base64.b64decode(row['rgb_base64'], validate=True)
    shape = tuple(int(value) for value in row['rgb_shape'])
    image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if _sha256(image.tobytes()) != row['rgb_sha256']:
        raise RuntimeError('camera_support_frozen_rgb_digest_mismatch')
    image.setflags(write=False)
    return image


def _assert_prediction_seals(plan: dict[str, object], seals: list[dict[str, object]],
                             load_payload) -> None:
    expected_ids = [row['identity'] for row in plan['renderer_registry']]
    expected = {(identity, arm) for identity in expected_ids
                for arm in ('baseline', 'candidate')}
    actual = {(item['identity'], item['arm']) for item in seals}
    if len(seals) != 24 or actual != expected:
        raise RuntimeError('camera_support_not_all_24_prediction_seals')
    for item in seals:
        payload = json.loads(load_payload(item))
        if (payload.get('identity') != item['identity']
                or payload.get('arm') != item['arm']
                or payload.get('preregistration_sha256') != plan['preregistration_sha256']):
            raise RuntimeError('camera_support_prediction_seal_binding_mismatch')


def _score_sealed_predictions(harness, plan: dict[str, object],
                              seals: list[dict[str, object]], load_payload) -> dict[str, object]:
    if (plan.get('execution_state') != 'PREDICTIONS_VALIDATED'
            or not plan.get('seal_validation', {}).get('validated')):
        raise RuntimeError('camera_support_truth_before_validated_seals')
    _assert_prediction_seals(plan, seals, load_payload)
    renderer = {row['identity']: row for row in plan['renderer_registry']}
    truth_rows = plan['evaluator_truth_registry']
    scenes = {}
    for row in truth_rows:
        source = renderer.get(row['identity'])
        if (source is None or source['recipe_sha256'] != row['renderer_recipe_sha256']):
            raise RuntimeError('camera_support_evaluator_recipe_binding_mismatch')
        # This is the first point at which an evaluator camera is reconstructed.
        scenes[row['identity']] = harness.CameraRecipe(
            identity=row['identity'], physical=True, recipe_json=source['recipe_json'])

    evaluations = []
    for row in truth_rows:
        by_arm = {}
        for seal in (item for item in seals if item['identity'] == row['identity']):
            payload = json.loads(load_payload(seal))
            traced = payload['trace_result']
            prediction = _extract_prediction(traced)
            if prediction is None:
                by_arm[seal['arm']] = {
                    'outcome': 'refusal', 'reason': traced.get('reason', 'ordinary_refusal'),
                    'metrics': None,
                }
                continue
            metrics = harness.camera_metrics(
                prediction, harness.truth_camera(scenes[row['identity']]))
            outcome = 'pass' if metrics['diagnostic_pass'] else 'wrong'
            by_arm[seal['arm']] = {
                'outcome': outcome, 'reason': '', 'metrics': metrics,
                'inliers': prediction['inliers'], 'segments': prediction['segments'],
                'per_axis_inliers': prediction['per_axis_inliers'],
            }
        candidate_descriptor = next(item for item in seals
                                    if item['identity'] == row['identity']
                                    and item['arm'] == 'candidate')
        candidate_payload = json.loads(load_payload(candidate_descriptor))
        support = candidate_payload.get('support')
        evaluations.append({
            'identity': row['identity'], 'stratum': row['stratum'],
            'base_identity': row['base_identity'], 'baseline': by_arm['baseline'],
            'candidate': by_arm['candidate'],
            'support': None if not isinstance(support, dict) else support.get('split'),
        })

    baseline_pass = sum(item['baseline']['outcome'] == 'pass' for item in evaluations)
    candidate_pass = sum(item['candidate']['outcome'] == 'pass' for item in evaluations)
    lost_passes = sum(item['baseline']['outcome'] == 'pass'
                       and item['candidate']['outcome'] != 'pass' for item in evaluations)
    stress = [item for item in evaluations if item['stratum'] != 'ordinary']
    baseline_stress_wrong = sum(item['baseline']['outcome'] == 'wrong' for item in stress)
    candidate_stress_wrong = sum(item['candidate']['outcome'] == 'wrong' for item in stress)
    new_wrong_stress = sum(item['baseline']['outcome'] != 'wrong'
                           and item['candidate']['outcome'] == 'wrong' for item in stress)
    gui_stability = []
    for item in evaluations:
        if item['stratum'] == 'gui_contamination':
            base = next(value for value in evaluations
                        if value['identity'] == item['base_identity'])
            gui_stability.append({
                'identity': item['identity'],
                'baseline_equal': item['baseline'].get('metrics') == base['baseline'].get('metrics'),
                'candidate_equal': item['candidate'].get('metrics') == base['candidate'].get('metrics'),
            })
    gates = {
        'candidate_passes_at_least_9_of_12': candidate_pass >= 9,
        'net_added_passes_at_least_3': candidate_pass - baseline_pass >= 3,
        'zero_lost_baseline_passes': lost_passes == 0,
        'zero_newly_wrong_stress': new_wrong_stress == 0,
        'candidate_stress_wrong_not_above_baseline': candidate_stress_wrong <= baseline_stress_wrong,
    }
    summary = {
        'baseline_passes': baseline_pass, 'candidate_passes': candidate_pass,
        'net_added_passes': candidate_pass - baseline_pass,
        'baseline_passes_lost': lost_passes,
        'baseline_stress_wrong': baseline_stress_wrong,
        'candidate_stress_wrong': candidate_stress_wrong,
        'newly_wrong_stress': new_wrong_stress, 'gui_only_stability': gui_stability,
        'gates': gates, 'promotion_gate_passed': all(gates.values()),
        'production_gate_changed': False, 'product_accuracy_claim': False,
    }
    return {'evaluations': evaluations, 'summary': summary}


def _write_report_state(report_stream, state: str) -> str:
    report_stream.seek(0)
    report_text = report_stream.read()
    import re
    updated = re.sub(r'CAM_SUPPORT_EXECUTION_STATE: [A-Z_]+',
                     'CAM_SUPPORT_EXECUTION_STATE: ' + state, report_text, count=1)
    report_stream.seek(0)
    report_stream.write(updated)
    report_stream.truncate()
    report_stream.flush()
    os.fsync(report_stream.fileno())
    return updated


def _r3_pair_diagnostics(harness, baseline: dict[str, object],
                         candidate: dict[str, object]) -> dict[str, object]:
    """Validate one paired R3 trace, mask isolation, and exact A/B replay."""
    base_trace = baseline["trace_result"]
    cand_trace = candidate["trace_result"]
    status_ok = (base_trace.get("status") != "trace_refusal"
                 and cand_trace.get("status") != "trace_refusal")
    base_rgb, cand_rgb = base_trace.get("rgb_sha256"), cand_trace.get("rgb_sha256")
    rgb_ok = (base_rgb is not None and base_rgb == cand_rgb
              and baseline.get("adapter_rgb_sha256") == base_rgb
              and candidate.get("adapter_rgb_sha256") == cand_rgb)
    diagnostics = {
        "arm_status": {"passed": status_ok,
                       "A": base_trace.get("status"), "B": cand_trace.get("status")},
        "rgb_pair": {"passed": rgb_ok, "A": base_rgb, "B": cand_rgb},
        "raw_segment_dtype": {"passed": False},
        "raw_segment_shape": {"passed": False},
        "raw_segment_sha256": {"passed": False},
        "automatic_mask": {"passed": False},
        "boundary_band": {"passed": False},
        "expanded_mask": {"passed": False},
        "split_exclusion": {"passed": False},
        "upstream_path": {"passed": False},
        "splitter_replay": {"passed": False},
    }
    if not status_ok:
        return diagnostics
    base_support, cand_support = baseline.get("support"), candidate.get("support")
    if not isinstance(base_support, dict) or not isinstance(cand_support, dict):
        return diagnostics
    try:
        base_raw = base_support["raw_segments"]
        cand_raw = cand_support["raw_segments"]
        dtype_ok = (canonical_dtype(base_support["raw_segment_dtype"])
                    == canonical_dtype(cand_support["raw_segment_dtype"])
                    == canonical_dtype(base_raw.dtype)
                    == canonical_dtype(cand_raw.dtype))
        shape_ok = (base_support["raw_segment_shape"] == list(base_raw.shape)
                    == cand_support["raw_segment_shape"] == list(cand_raw.shape))
        raw_digest_a = _sha256(np.ascontiguousarray(base_raw).tobytes())
        raw_digest_b = _sha256(np.ascontiguousarray(cand_raw).tobytes())
        raw_digest_ok = (raw_digest_a == raw_digest_b
                         == base_support["raw_segment_sha256"]
                         == cand_support["raw_segment_sha256"])
        diagnostics["raw_segment_dtype"] = {
            "passed": dtype_ok, "A": base_support["raw_segment_dtype"],
            "B": cand_support["raw_segment_dtype"]}
        diagnostics["raw_segment_shape"] = {
            "passed": shape_ok, "A": base_support["raw_segment_shape"],
            "B": cand_support["raw_segment_shape"]}
        diagnostics["raw_segment_sha256"] = {
            "passed": raw_digest_ok, "A": raw_digest_a, "B": raw_digest_b}

        mask_a, mask_b = base_support["mask"], cand_support["mask"]
        band_a, band_b = base_support["boundary_band"], cand_support["boundary_band"]
        expanded_a, expanded_b = base_support["expanded_mask"], cand_support["expanded_mask"]
        mask_hash_a, mask_hash_b = _sha256(mask_a.tobytes()), _sha256(mask_b.tobytes())
        band_hash_a, band_hash_b = _sha256(band_a.tobytes()), _sha256(band_b.tobytes())
        expanded_hash_a = _sha256(expanded_a.tobytes())
        expanded_hash_b = _sha256(expanded_b.tobytes())
        mask_ok = (np.array_equal(mask_a, mask_b)
                   and mask_hash_a == mask_hash_b
                   == base_support["mask_sha256"] == cand_support["mask_sha256"])
        band_ok = (np.array_equal(band_a, band_b)
                   and band_hash_a == band_hash_b
                   == base_support["boundary_band_sha256"]
                   == cand_support["boundary_band_sha256"])
        expanded_ok = (np.array_equal(expanded_a, expanded_b)
                       and expanded_hash_a == expanded_hash_b
                       == base_support["expanded_mask_sha256"]
                       == cand_support["expanded_mask_sha256"]
                       and np.array_equal(expanded_a, np.logical_or(mask_a, band_a)))
        split_masks_ok = (
            base_support["boundary_band_applied"] is False
            and cand_support["boundary_band_applied"] is True
            and np.array_equal(base_support["split_exclusion_mask"], mask_a)
            and np.array_equal(cand_support["split_exclusion_mask"], expanded_a)
            and base_support["split_exclusion_mask_sha256"] == mask_hash_a
            and cand_support["split_exclusion_mask_sha256"] == expanded_hash_a)
        diagnostics["automatic_mask"] = {"passed": mask_ok,
                                          "sha256": mask_hash_a}
        diagnostics["boundary_band"] = {"passed": band_ok,
                                        "sha256": band_hash_a}
        diagnostics["expanded_mask"] = {"passed": expanded_ok,
                                         "sha256": expanded_hash_a}
        diagnostics["split_exclusion"] = {"passed": split_masks_ok}

        mask_stage_a = next(item for item in base_trace["trace"]["stages"]
                            if item["stage"] == "create_gui_mask")
        mask_stage_b = next(item for item in cand_trace["trace"]["stages"]
                            if item["stage"] == "create_gui_mask")
        enhance_a = next(item for item in base_trace["trace"]["stages"]
                         if item["stage"] == "enhance")
        enhance_b = next(item for item in cand_trace["trace"]["stages"]
                         if item["stage"] == "enhance")
        mask_trace_a = harness.decode_array(base_trace["trace"], mask_stage_a["output"])
        mask_trace_b = harness.decode_array(cand_trace["trace"], mask_stage_b["output"])
        enhance_trace_a = harness.decode_array(base_trace["trace"], enhance_a["output"])
        enhance_trace_b = harness.decode_array(cand_trace["trace"], enhance_b["output"])
        upstream_ok = (np.array_equal(mask_trace_a, mask_a)
                       and np.array_equal(mask_trace_b, mask_b)
                       and np.array_equal(enhance_trace_a, enhance_trace_b))
        diagnostics["upstream_path"] = {"passed": upstream_ok}

        expected_a, _ = split_with_boundary_band(
            base_raw, mask_a, apply_boundary_band=False)
        expected_b, _ = split_with_boundary_band(
            base_raw, mask_a, apply_boundary_band=True)
        detected_a = next(item for item in base_trace["trace"]["stages"]
                          if item["stage"] == "detect_segments")
        detected_b = next(item for item in cand_trace["trace"]["stages"]
                          if item["stage"] == "detect_segments")
        output_a = harness.decode_array(base_trace["trace"], detected_a["output"])
        output_b = harness.decode_array(cand_trace["trace"], detected_b["output"])
        replay_ok = (
            np.array_equal(expected_a.segments, base_support["split"].segments)
            and np.array_equal(expected_b.segments, cand_support["split"].segments)
            and np.array_equal(output_a, expected_a.segments)
            and np.array_equal(output_b, expected_b.segments)
            and expected_a.intervals == base_support["split"].intervals
            and expected_b.intervals == cand_support["split"].intervals)
        diagnostics["splitter_replay"] = {"passed": replay_ok}
    except (KeyError, TypeError, ValueError, RuntimeError, StopIteration):
        pass
    return diagnostics


def run_study() -> dict[str, object]:
    """Execute only the tested one-shot path; scoring follows validated 24 seals."""
    report_path, json_path = _active_study_paths()
    if not report_path.is_file() or not json_path.is_file():
        raise RuntimeError('camera_support_preregistration_missing')
    if not _STUDY_LOCK.acquire(blocking=False):
        raise RuntimeError('camera_support_reentrancy_or_concurrency')
    try:
        with report_path.open('r+', encoding='utf-8', newline='') as report_stream:
            if os.name == 'nt':
                import msvcrt
                report_stream.seek(0)
                msvcrt.locking(report_stream.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = lambda: (report_stream.seek(0), msvcrt.locking(report_stream.fileno(), msvcrt.LK_UNLCK, 1))
            else:
                import fcntl
                fcntl.flock(report_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                unlock = lambda: fcntl.flock(report_stream.fileno(), fcntl.LOCK_UN)
            try:
                plan = json.loads(json_path.read_text(encoding='utf-8'))
                report_text = report_stream.read()
                study_id = plan.get('study')
                if (study_id not in ('CAM-SUPPORT-02R2', 'CAM-BOUNDARY-SPLIT-03')
                        or plan.get('execution_state') != 'NOT_STARTED'
                        or 'CAM_SUPPORT_EXECUTION_STATE: NOT_STARTED' not in report_text):
                    raise RuntimeError('camera_support_study_already_claimed')
                prereg_material = {key: value for key, value in plan.items()
                                   if key != 'preregistration_sha256'}
                if _sha256(_canonical(prereg_material)) != plan.get('preregistration_sha256'):
                    raise RuntimeError('camera_support_preregistration_digest_mismatch')
                harness = _study_harness()
                if _expected_dependencies(harness) != plan['dependencies']:
                    raise RuntimeError('camera_support_runtime_or_source_drift')
                if study_id == 'CAM-BOUNDARY-SPLIT-03':
                    if _gui_border_reference_pins() != plan.get('treatment_reference'):
                        raise RuntimeError('camera_boundary_reference_drift')
                    _, _, sidecar_path = _r3_paths()
                else:
                    _, _, sidecar_path = _r2_paths()
                if not sidecar_path.is_file() or sidecar_path.stat().st_size != 0:
                    raise RuntimeError('camera_support_r2_sidecar_not_empty_at_start')
                store = GzipMemberStore(sidecar_path)
                rows = plan['renderer_registry']
                identities = tuple(row['identity'] for row in rows)
                if len(identities) != 12 or len(set(identities)) != 12:
                    raise RuntimeError('camera_support_renderer_registry_count_mismatch')
                renderer = {row['identity']: row for row in rows}
                frozen = {identity: _frozen_rgb(renderer[identity]) for identity in identities}

                def persist(state):
                    _save_plan(json_path, state)
                    _write_report_state(report_stream, state['execution_state'])

                def store_payload(identity, arm, payload_bytes, state):
                    return store.append(payload_bytes, state, identity, arm)

                def validate_payloads(state):
                    result = store.validate_all(state)
                    if result['member_count'] != 24:
                        raise RuntimeError('camera_support_not_all_24_gzip_members')
                    for descriptor in state['sealed_predictions']:
                        payload = json.loads(store.read_member(descriptor))
                        row = renderer.get(descriptor['identity'])
                        if (row is None or payload.get('identity') != descriptor['identity']
                                or payload.get('arm') != descriptor['arm']
                                or payload.get('rgb_sha256') != row['rgb_sha256']
                                or payload.get('preregistration_sha256') != plan['preregistration_sha256']):
                            raise RuntimeError('camera_support_member_binding_mismatch')
                    return result

                def trace_arm(identity, arm):
                    image = frozen[identity]
                    if study_id == 'CAM-BOUNDARY-SPLIT-03':
                        adapter = _run_r3_a_arm if arm == 'baseline' else _run_r3_b_arm
                    else:
                        adapter = _run_baseline_arm if arm == 'baseline' else _run_candidate_arm
                    return adapter(image, identity)

                def validate_pair(identity, baseline, candidate):
                    if study_id == 'CAM-BOUNDARY-SPLIT-03':
                        return _r3_pair_diagnostics(harness, baseline, candidate)
                    base_trace = baseline['trace_result']
                    cand_trace = candidate['trace_result']
                    status_ok = (base_trace.get('status') != 'trace_refusal'
                                 and cand_trace.get('status') != 'trace_refusal')
                    expected_hash = renderer[identity]['rgb_sha256']
                    base_hash_ok = base_trace.get('rgb_sha256') == expected_hash
                    cand_hash_ok = cand_trace.get('rgb_sha256') == expected_hash
                    rgb_pair_ok = base_trace.get('rgb_sha256') == cand_trace.get('rgb_sha256')
                    if not (status_ok and base_hash_ok and cand_hash_ok and rgb_pair_ok):
                        passed = status_ok and base_hash_ok and cand_hash_ok and rgb_pair_ok
                        return {
                            'arm_status': {'passed': status_ok,
                                           'baseline': base_trace.get('status'),
                                           'candidate': cand_trace.get('status')},
                            'rgb_pair': {'passed': base_hash_ok and cand_hash_ok and rgb_pair_ok,
                                         'expected': expected_hash,
                                         'baseline': base_trace.get('rgb_sha256'),
                                         'candidate': cand_trace.get('rgb_sha256')},
                            'dtype': {'passed': passed}, 'shape': {'passed': passed},
                            'raw_data_sha256': {'passed': passed},
                        }
                    stages = base_trace['trace']['stages']
                    detect = next(item for item in stages if item['stage'] == 'detect_segments')
                    mask_stage = next(item for item in stages if item['stage'] == 'create_gui_mask')
                    raw_baseline = harness.decode_array(base_trace['trace'], detect['output'])
                    mask_baseline = harness.decode_array(base_trace['trace'], mask_stage['output'])
                    support = candidate.get('support')
                    if not isinstance(support, dict) or 'split' not in support:
                        raise RuntimeError('camera_support_candidate_support_missing')
                    parity = detector_parity_diagnostics(
                        raw_baseline, support['raw_segment_dtype'], support['raw_segment_shape'],
                        support['raw_segment_sha256'])
                    mask_ok = (_sha256(mask_baseline.tobytes()) == support['mask_sha256']
                               and mask_baseline.shape == support['mask'].shape
                               and np.array_equal(mask_baseline, support['mask']))
                    return {**parity, 'arm_status': {'passed': True},
                            'rgb_pair': {'passed': True}, 'automatic_mask': {'passed': mask_ok}}

                def seal_payload(identity, arm, output):
                    row = renderer[identity]
                    return _prediction_payload(
                        identity, arm, row['rgb_sha256'], tuple(row['rgb_shape']),
                        output['trace_result'], output.get('support'),
                        plan['preregistration_sha256'], row['recipe_sha256'])

                def score(seals):
                    return _score_sealed_predictions(harness, plan, seals, store.read_member)

                try:
                    result = execute_one_shot_attempt(
                        plan, identities, trace_arm=trace_arm, validate_pair=validate_pair,
                        seal_payload=seal_payload, store_payload=store_payload,
                        validate_payloads=validate_payloads,
                        persist=persist, score_seals=score)
                except Exception:
                    raise
                plan['result_sha256'] = _sha256(_canonical({
                    'evaluations': plan['evaluations'], 'summary': plan['summary'],
                    'sealed_predictions': [item['payload_sha256']
                                           for item in plan['sealed_predictions']]}))
                _save_plan(json_path, plan)
                report_text = _write_report_state(report_stream, 'COMPLETE')
                evaluations = plan['evaluations']
                summary = plan['summary']
                baseline_label, candidate_label = (
                    ('A', 'B') if study_id == 'CAM-BOUNDARY-SPLIT-03'
                    else ('Baseline', 'Candidate'))
                rows_text = [
                    '\n\n## Paired outcomes\n\n'
                    f'| Case | Stratum | {baseline_label} | {candidate_label} | Raw segments | Retained | Samples |\n'
                    '|---|---|---:|---:|---:|---:|---:|'
                ]
                for item in evaluations:
                    support = item['support'] or {}
                    rows_text.append(
                        f"| {item['identity']} | {item['stratum']} | "
                        f"{item['baseline']['outcome']} | {item['candidate']['outcome']} | "
                        f"{support.get('input_count', 0)} | {support.get('retained_count', 0)} | "
                        f"{support.get('sample_count', 0)} |")
                rows_text.extend([
                    '\n\n## Frozen gates\n',
                    f"{baseline_label} passes: {summary['baseline_passes']}/12; "
                    f"{candidate_label} passes: {summary['candidate_passes']}/12; "
                    f"net gain: {summary['net_added_passes']}; "
                    f"lost {baseline_label} passes: {summary['baseline_passes_lost']}.",
                    f"Stress wrong cases: {baseline_label} {summary['baseline_stress_wrong']}/6, "
                    f"{candidate_label} {summary['candidate_stress_wrong']}/6, "
                    f"newly wrong {summary['newly_wrong_stress']}.",
                    'Promotion gate: ' + ('PASS' if summary['promotion_gate_passed'] else 'FAIL') + '.',
                    'Synthetic camera diagnostics only; no product geometry/state accuracy or '
                    'production integration claim is made.',
                    'Full traces, seals, runtime/source pins and per-gate values are in the adjacent JSON.',
                ])
                report_stream.seek(0, os.SEEK_END)
                report_stream.write('\n'.join(rows_text) + '\n')
                report_stream.flush()
                os.fsync(report_stream.fileno())
                return result
            except Exception as exc:
                try:
                    current = json.loads(json_path.read_text(encoding='utf-8'))
                    if current.get('execution_state') not in ('FAILED', 'COMPLETE'):
                        current['execution_state'] = 'FAILED'
                        current['failure'] = {
                            'reason': type(exc).__name__ + ':' + str(exc),
                            'sealed_prediction_count': len(current.get('sealed_predictions', [])),
                            'post_hoc_truth_accessed': current.get('execution_state')
                                in ('PREDICTIONS_SEALED', 'PREDICTIONS_VALIDATED'),
                            'retry_performed': False,
                        }
                        _save_plan(json_path, current)
                        _write_report_state(report_stream, 'FAILED')
                except (OSError, ValueError, KeyError):
                    pass
                raise
            finally:
                unlock()
    finally:
        _STUDY_LOCK.release()

# CAM-VALID-SUPPORT-05 additive seams. No existing study runner selects these.
_VALID_RUN = np.dtype([("parent", "<u4"), ("start", "<u4"), ("end", "<u4"),
                       ("reason", "u1"), ("length", "<f8"), ("output", "<i4")])


def _managed_tree_size(value, *, ceiling):
    """Bounded object-tree measurement without building JSON or raw copies."""
    remaining = int(ceiling)
    def visit(item, depth):
        nonlocal remaining
        if depth > 32:
            raise AcquisitionIntegrityError("payload_nesting")
        size = sys.getsizeof(item)
        remaining -= size
        if remaining < 0:
            raise AcquisitionCapError("object_tree_byte_cap")
        if isinstance(item, dict):
            for key, child in item.items():
                visit(key, depth+1)
                visit(child, depth+1)
        elif isinstance(item, (tuple,list)):
            for child in item:
                visit(child, depth+1)
        elif not isinstance(item, (str,int,float,bool,type(None))):
            raise AcquisitionIntegrityError("payload_object_type")
        return size
    visit(value,0)
    return int(ceiling)-remaining


class LosslessScoreStream:
    """Source-compatible append/last-replace storage with lossless JSON lines.

    Only the pending last score stays as an object. Earlier complete scores are
    canonical, gzip-compressed records. No score field is omitted or truncated.
    """
    def __init__(self,budget,max_records=120_000):
        if type(max_records) is not int or not 0<=max_records<=120_000:
            raise AcquisitionCapError("score_stream_event_cap")
        import zlib
        self.budget,self.maximum=budget,max_records
        self._workspace=budget.acquire(512*1024)
        self._compressor=zlib.compressobj(6,zlib.DEFLATED,31)
        self._compressed=bytearray()
        self._storage=budget.acquire(4096)
        self._pending=None; self._pending_lease=0
        self._count=self._raw_bytes=0
        self._raw_sha=hashlib.sha256()
        self._packet=None

    def __len__(self):
        return self._count

    def __getitem__(self,index):
        if index!=-1 or self._pending is None:
            raise AcquisitionIntegrityError("score_stream_last_access_only")
        return self._pending

    def __setitem__(self,index,event):
        if index!=-1 or self._pending is None or self._packet is not None:
            raise AcquisitionIntegrityError("score_stream_last_replace_only")
        size=_managed_tree_size(event,ceiling=32768)+256
        lease=self.budget.acquire(size)
        old_lease=self._pending_lease
        self._pending=event
        self._pending_lease=lease
        self.budget.release(old_lease)

    def _extend(self,chunk):
        required=len(self._compressed)+len(chunk)
        if required>16*1024*1024:
            raise AcquisitionCapError("score_stream_compressed_cap")
        replacement=self.budget.acquire(required+required//8+4096)
        old=self._storage
        self._compressed.extend(chunk)
        self._storage=replacement
        self.budget.release(old)

    def _commit(self):
        if self._pending is None:
            return
        size=_managed_tree_size(self._pending,ceiling=32768)
        scratch=self.budget.acquire(12*size+128*1024)
        raw=chunk=None
        try:
            raw=(_canonical(self._pending)+"\n").encode("utf-8")
            if len(raw)>128*1024 or self._raw_bytes+len(raw)>64_120_000:
                raise AcquisitionCapError("score_stream_decoded_cap")
            self._raw_sha.update(raw)
            self._raw_bytes+=len(raw)
            chunk=self._compressor.compress(raw)
            self._extend(chunk)
            self._pending=None
            self.budget.release(self._pending_lease)
            self._pending_lease=0
        finally:
            raw=chunk=None
            self.budget.release(scratch)

    def append(self,event):
        if self._packet is not None or self._count>=self.maximum:
            raise AcquisitionCapError("score_stream_event_cap")
        self._commit()
        size=_managed_tree_size(event,ceiling=32768)+256
        self._pending_lease=self.budget.acquire(size)
        self._pending=event
        self._count+=1

    def finish(self):
        if self._packet is not None:
            return self._packet
        self._commit()
        scratch=self.budget.acquire(1024*1024)
        chunk=None
        try:
            chunk=self._compressor.flush()
            self._extend(chunk)
            self._compressor=None
            self.budget.release(self._workspace)
            self._workspace=0
        finally:
            chunk=None
            self.budget.release(scratch)
        size=len(self._compressed)
        scratch=self.budget.acquire(3*4*((size+2)//3)+4096)
        encoded=None
        try:
            encoded=base64.b64encode(self._compressed).decode("ascii")
            self.budget.acquire(len(encoded)+4096)
            self._packet={"encoding":"gzip-json-lines-v1","record_count":self._count,
                "raw_bytes":self._raw_bytes,"raw_sha256":self._raw_sha.hexdigest(),
                "compressed_bytes":size,"compressed_sha256":_sha256(self._compressed),
                "base64":encoded}
            self._compressed=None
            self.budget.release(self._storage)
            self._storage=0
            return self._packet
        finally:
            encoded=None
            self.budget.release(scratch)

def _valid_recorder_class(base, budget):
    """Observe inherited instrumentation without changing its hex snapshots.

    Charge stored events/objects plus raw, hex, JSON and UTF-8 scratch before
    inherited array/add allocation. Opaque calibration/native buffers remain
    outside this managed accounting, as disclosed in the accepted design.
    """
    class GuardedRecorder(base):
        def __init__(self, policy):
            self._array_ids = set()
            # At most max_score_events+max_stage_events arrays are first seen.
            # Reserve scalar bookkeeping and one fixed-schema event being built.
            self._bookkeeping = budget.acquire(65536)
            super().__init__(policy)
            self.scores=LosslessScoreStream(budget,policy.max_score_events)

        def payload(self):
            result=super().payload()
            result["scores"]=self.scores.finish()
            return result

        def add(self, collection, event, cap):
            if len(collection) >= cap:
                raise AcquisitionCapError("trace_event_cap")
            size = _managed_tree_size(event,ceiling=65536)
            if size > 32768:
                raise AcquisitionCapError("trace_event_object_cap")
            # Bound serialization by six escaped bytes per managed input byte.
            lease=budget.acquire(12*size+4096)
            try:
                super().add(collection,event,cap)
                if collection is not self.scores:
                    budget.acquire(size+128)
            finally:
                budget.release(lease)

        def array(self, value, *, segments=False):
            if not isinstance(value,np.ndarray) or value.dtype.hasobject:
                raise AcquisitionIntegrityError("trace_array_contract")
            if segments and (value.ndim != 2 or value.shape[1:] != (4,) or len(value)>10_000):
                raise AcquisitionCapError("trace_segment_cap")
            # Existing recorder keeps the exact array object even for repeated
            # digests; count unique object identities separately from raw bytes.
            first = id(value) not in self._array_ids
            if value.nbytes > self.policy.max_array_bytes:
                raise AcquisitionCapError("trace_array_cap")
            lease=budget.acquire(12*value.nbytes+16384)
            try:
                if first:
                    budget.acquire(value.nbytes+2*value.nbytes+2048)
                    self._array_ids.add(id(value))
                return super().array(value,segments=segments)
            finally:
                budget.release(lease)
    return GuardedRecorder


def audit_valid_support(rows, original, expanded, split, *, budget,
                        config=FROZEN_CONFIG):
    """Persist exhaustive occupancy/run attribution; verify the shared splitter."""
    if len(rows)>MAX_INPUT_SEGMENTS:
        raise AcquisitionCapError("detector_input_cap")
    runs=PackedBuffer(_VALID_RUN,config.max_runs,budget,"supported_run_cap")
    packed=PackedBuffer(np.uint8,(MAX_TOTAL_SAMPLES+3)//4,budget,"support_sample_cap")
    totals={"frame":0,"original":0,"band":0,"supported":0,"samples":0,
            "singleton_samples":0,"short_samples":0,"emitted_samples":0,
            "singleton_runs":0,"short_runs":0,"emitted_runs":0}
    emitted=0
    for parent,row in enumerate(rows):
        remaining=min(MAX_SAMPLES_PER_SEGMENT,MAX_TOTAL_SAMPLES-totals["samples"])
        categories,_,_=sample_occupancy(row,original,expanded,budget,remaining)
        n=len(categories)
        parameters_lease=budget.acquire(n*8+256)
        try:
            parameters=np.linspace(0.,1.,n,dtype=np.float64)
            for category in categories:
                totals[("frame","original","band","supported")[int(category)]]+=1
            totals["samples"]=pack_categories(categories,packed,totals["samples"])
            position=0
            while position<n:
                if categories[position] != 3:
                    position+=1
                    continue
                first=position
                while position+1<n and categories[position+1] == 3:
                    position+=1
                last=position
                x0,y0,x1,y1=(float(v) for v in row)
                ta,tb=float(parameters[first]),float(parameters[last])
                sx0,sy0=x0+ta*(x1-x0),y0+ta*(y1-y0)
                sx1,sy1=x0+tb*(x1-x0),y0+tb*(y1-y0)
                length=math.hypot(sx1-sx0,sy1-sy0)
                reason=0 if last==first else (1 if length<40. else 2)
                name=("singleton","short","emitted")[reason]
                totals[name+"_runs"]+=1
                totals[name+"_samples"]+=last-first+1
                if reason==2:
                    if emitted>=len(split.segments):
                        raise AcquisitionIntegrityError("split_output_count")
                    lease=budget.acquire(1280)
                    try:
                        expected=np.asarray((sx0,sy0,sx1,sy1),dtype=rows.dtype)
                        if expected.tobytes()!=split.segments[emitted].tobytes():
                            raise AcquisitionIntegrityError("split_output_byte_parity")
                    finally:
                        expected=None
                        budget.release(lease)
                runs.append((parent,first,last,reason,length,emitted if reason==2 else -1))
                if reason==2:
                    emitted+=1
                position+=1
        finally:
            category_bytes=categories.nbytes+256
            del parameters,categories
            budget.release(parameters_lease)
            budget.release(category_bytes)
    if (totals["samples"]!=sum(totals[k] for k in ("frame","original","band","supported"))
            or totals["supported"]!=sum(totals[k] for k in ("singleton_samples","short_samples","emitted_samples"))
            or emitted!=len(split.segments) or totals["samples"]!=split.sample_count):
        raise AcquisitionIntegrityError("split_support_accounting",totals)
    return {"runs":runs.view(),"occupancy":packed.view(),"counts":totals}


def trace_valid_acquisition(image,frame_id,*,candidate,trace_rgb,
                            calibration_module,gui_mask_module,trace_module=None,
                            config=FROZEN_CONFIG):
    """Injected RGB-only adapter. Detector pre-40 control telemetry is unavailable.

    Dependency injection permits pure tests. Eventual real harness use must pass
    its module so inherited recorder encoding/objects are budgeted before growth.
    """
    from .evaluation_valid_stencil_segments import validate_config
    validate_config(config)
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or not image.flags.c_contiguous or image.flags.writeable):
        raise AcquisitionIntegrityError("readonly_contiguous_rgb_required")
    h,w=image.shape[:2]
    if not h or not w or max(h,w)>config.max_dimension or h*w>config.max_pixels:
        raise AcquisitionCapError("input_pixel_or_dimension_cap")
    if type(candidate) is not bool or not isinstance(frame_id,str):
        raise AcquisitionIntegrityError("image_only_adapter_contract")
    budget=AllocationBudget(config.max_live_bytes)
    budget.packed_limit=config.max_ledger_bytes
    budget.acquire(3*image.nbytes+65536)
    detector=getattr(calibration_module,"detect_segments",None)
    mask_function=getattr(gui_mask_module,"create_gui_mask",None)
    if not callable(detector) or not callable(mask_function):
        raise AcquisitionIntegrityError("adapter_dependencies")
    if not _STUDY_LOCK.acquire(blocking=False):
        raise AcquisitionIntegrityError("adapter_concurrency")
    original_recorder=None
    captured={"mask_calls":0,"detector_calls":0,"candidate":candidate,"budget":budget,
              "pre_length_observability":"candidate" if candidate else "unobservable",
              "config_sha256":config.sha256,"frame_id":frame_id,
              "rgb_shape":list(image.shape),"rgb_sha256":_sha256(image.tobytes())}

    def mask_delegate(height,width,*args,**kwargs):
        if captured["mask_calls"] or (height,width)!=(h,w):
            raise AcquisitionIntegrityError("automatic_mask_call_contract")
        # Reserve delegated mask plus copies and every boundary routine scratch
        # before delegate invocation. Native/control opaque allocations excluded.
        lease=budget.acquire(32*h*w+16384)
        try:
            mask=mask_function(height,width,*args,**kwargs)
            if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=(h,w) or not mask.flags.c_contiguous:
                raise AcquisitionIntegrityError("automatic_mask_contract")
            _,band=mask_transition_band(mask,max_mask_pixels=config.max_pixels)
            original=mask.copy()
            expanded=np.logical_or(original,band)
            original.setflags(write=False); expanded.setflags(write=False)
            captured.update({"mask":original,"boundary_band":band,"expanded_mask":expanded,
                             "mask_calls":1})
            budget.acquire(4*h*w+2048)
            return mask
        finally:
            budget.release(lease)

    def detector_delegate(grey,**kwargs):
        if captured["detector_calls"] or captured["mask_calls"]!=1:
            raise AcquisitionIntegrityError("detector_call_contract")
        captured["detector_calls"]=1
        original,expanded=captured["mask"],captured["expanded_mask"]
        if candidate:
            # Enhanced/zero-filled trace input deliberately not used.
            gray=pointwise_gray(image,budget=budget,config=config)
            extraction=extract_segments(gray,expanded,original_mask=original,
                                         frozen_config=config,budget=budget)
            raw=extraction.segments
            captured["extraction"]=extraction
        else:
            raw=detector(grey,**kwargs)
        if (not isinstance(raw,np.ndarray) or raw.ndim!=2 or raw.shape[1:]!=(4,)
                or raw.dtype.kind not in "fiu" or not raw.flags.c_contiguous):
            raise AcquisitionIntegrityError("detector_output_contract")
        if len(raw)>MAX_INPUT_SEGMENTS:
            raise AcquisitionCapError("detector_input_cap")
        # Reserve all snapshot, splitter intermediate and result records before
        # its original implementation constructs untyped retained/interval lists.
        # An emitted run contains >=81 samples, bounding emitted objects.
        samples=0
        for row in raw:
            x0,y0,x1,y1=(float(v) for v in row)
            if not all(math.isfinite(v) for v in (x0,y0,x1,y1)):
                raise AcquisitionIntegrityError("nonfinite_detector_output")
            n=max(2,math.ceil(math.hypot(x1-x0,y1-y0)/0.5)+1)
            if n>MAX_SAMPLES_PER_SEGMENT or samples+n>MAX_TOTAL_SAMPLES:
                raise AcquisitionCapError("downstream_sample_cap")
            samples+=n
        largest=min(samples,MAX_SAMPLES_PER_SEGMENT)
        lease=budget.acquire(128*largest+raw.nbytes+4096+2048*(samples//80+len(raw)))
        try:
            snapshot=raw.copy()
            split=split_at_excluded_mask(raw,expanded)
            budget.acquire(snapshot.nbytes+split.segments.nbytes+2048*len(split.intervals)+2048)
        except (ValueError,TypeError) as exc:
            raise AcquisitionIntegrityError("shared_split_contract") from exc
        finally:
            budget.release(lease)
        snapshot.setflags(write=False)
        # Observation captures the exact returned detector matrix; never change
        # its dtype/rounding or substitute a recomputed control matrix.
        parity_lease=budget.acquire(3*raw.nbytes+4096)
        try:
            digest=_sha256(snapshot.tobytes())
            require_detector_parity(raw,snapshot.dtype,snapshot.shape,digest)
            captured["raw_segment_sha256"]=digest
            captured["raw_segment_count"]=len(snapshot)
        finally:
            budget.release(parity_lease)
        audit=audit_valid_support(snapshot,original,expanded,split,budget=budget,config=config)
        captured.update({"raw_segments":snapshot,"split":split,"audit":audit})
        return split.segments

    try:
        if trace_module is not None:
            original_recorder=getattr(trace_module,"TraceRecorder",None)
            if not isinstance(original_recorder,type):
                raise AcquisitionIntegrityError("trace_recorder_observer_required")
            trace_module.TraceRecorder=_valid_recorder_class(original_recorder,budget)
        elif getattr(trace_rgb,"__module__","").endswith("run_camera_mixed_depth"):
            raise AcquisitionIntegrityError("trace_recorder_observer_required")
        calibration_module.detect_segments=detector_delegate
        gui_mask_module.create_gui_mask=mask_delegate
        traced=trace_rgb(image,frame_id)
        if not isinstance(traced,dict):
            raise AcquisitionIntegrityError("trace_result_contract")
        # The inherited harness's resource/integrity trace_refusal is NOT an
        # ordinary camera refusal. Preserve exact bounded stage evidence.
        if traced.get("status")=="trace_refusal":
            raise AcquisitionIntegrityError("inherited_trace_refusal",{
                "reason":str(traced.get("reason",""))[:240],
                "mask_calls":captured["mask_calls"],"detector_calls":captured["detector_calls"]})
        if traced.get("status") not in ("returned","ordinary_refusal"):
            raise AcquisitionIntegrityError("unknown_trace_status")
        if captured["mask_calls"]!=1 or captured["detector_calls"]!=1:
            raise AcquisitionIntegrityError("missing_acquisition_stage")
        # Pure injected recorders may not implement the pre-allocation observer;
        # their returned JSON object remains capped/charged before persistence.
        measured=_managed_tree_size(traced,ceiling=config.max_member_bytes)
        budget.acquire(measured+4096)
        captured["trace_managed_bytes"]=measured
        return traced,captured
    except AcquisitionTerminalError as exc:
        exc.evidence.update({"mask_calls":captured["mask_calls"],
                             "detector_calls":captured["detector_calls"],
                             "managed_peak_bytes":budget.peak,"rgb_sha256":captured["rgb_sha256"],
                             "raw_segment_sha256":captured.get("raw_segment_sha256"),
                             "raw_segment_count":captured.get("raw_segment_count")})
        raise
    except Exception as exc:
        raise AcquisitionIntegrityError("unexpected_adapter_exception",{
            "exception_type":type(exc).__name__,"mask_calls":captured["mask_calls"],
            "detector_calls":captured["detector_calls"],"managed_peak_bytes":budget.peak}) from exc
    finally:
        calibration_module.detect_segments=detector
        gui_mask_module.create_gui_mask=mask_function
        if original_recorder is not None:
            trace_module.TraceRecorder=original_recorder
        _STUDY_LOCK.release()


def _run_valid_native_arm(rgb,frame_id,*,candidate):
    # Unrouted evaluation-only seam; never called by any existing run_study.
    _require_image_only_input(rgb,frame_id)
    harness=_study_harness()
    from mc_photogrammetry.reconstruction import calibration
    from mc_photogrammetry.vision import gui_mask
    traced,support=trace_valid_acquisition(rgb,frame_id,candidate=candidate,
        trace_rgb=harness.trace_rgb,trace_module=harness,
        calibration_module=calibration,gui_mask_module=gui_mask)
    return {"trace_result":traced,"support":support}


def _run_valid_native_control_arm(rgb,frame_id):
    return _run_valid_native_arm(rgb,frame_id,candidate=False)


def _run_valid_native_candidate_arm(rgb,frame_id):
    return _run_valid_native_arm(rgb,frame_id,candidate=True)


def valid_acquisition_payload(bindings,traced,support,*,config=FROZEN_CONFIG):
    """Exact bounded replay bytes; preserve existing trace hex snapshots."""
    required={"identity","arm","family","base_identity","stratum","rgb_sha256",
              "rgb_shape","preregistration_sha256","source_sha256","runtime_sha256"}
    if set(bindings)!=required or bindings["arm"] not in ("baseline","candidate"):
        raise AcquisitionIntegrityError("payload_binding_contract")
    from .evaluation_valid_stencil_segments import validate_config
    validate_config(config)
    if (bindings["identity"]!=support["frame_id"] or bindings["rgb_sha256"]!=support["rgb_sha256"]
            or bindings["rgb_shape"]!=support["rgb_shape"]
            or (bindings["arm"]=="candidate")!=support["candidate"]):
        raise AcquisitionIntegrityError("payload_case_arm_rgb_binding")
    for name in required-{"rgb_shape","base_identity"}:
        if not isinstance(bindings[name],str) or len(bindings[name])>256:
            raise AcquisitionIntegrityError("payload_scalar_binding")
    if bindings["base_identity"] is not None and (
            not isinstance(bindings["base_identity"],str) or len(bindings["base_identity"])>256):
        raise AcquisitionIntegrityError("payload_scalar_binding")
    budget=support["budget"]
    arrays={"original_mask":support["mask"],"boundary_band":support["boundary_band"],
            "expanded_mask":support["expanded_mask"],"detector_segments":support["raw_segments"],
            "emitted_segments":support["split"].segments,"run_ledger":support["audit"]["runs"],
            "sample_occupancy":support["audit"]["occupancy"]}
    extraction=support.get("extraction")
    if extraction is not None:
        arrays.update({"candidate_pre_length":extraction.pre_length,**extraction.arrays})
    raw_total=sum(value.nbytes for value in arrays.values())
    inherited_arrays=traced.get("trace",{}).get("arrays",{})
    if not isinstance(inherited_arrays,dict):
        raise AcquisitionIntegrityError("trace_array_metadata")
    # Count actual hex strings without decoding or changing existing encoding.
    hex_chars=0
    for meta in inherited_arrays.values():
        if not isinstance(meta,dict) or not isinstance(meta.get("bytes_hex"),str):
            raise AcquisitionIntegrityError("trace_hex_metadata")
        hex_chars+=len(meta["bytes_hex"])
    if hex_chars%2 or raw_total+hex_chars//2>config.max_raw_bytes:
        raise AcquisitionCapError("payload_raw_array_cap")
    # Bound escaped metadata without constructing a competing entire JSON tree.
    meta_size=_managed_tree_size(bindings,ceiling=config.max_metadata_bytes)
    trace_size=_managed_tree_size(traced,ceiling=config.max_member_bytes)
    # Six bytes per non-array input character plus known base64/hex lengths.
    scores=traced.get("trace",{}).get("scores")
    score_chars=0; score_bytes=0
    if isinstance(scores,dict) and scores.get("encoding")=="gzip-json-lines-v1":
        score_chars=len(scores["base64"]); score_bytes=int(scores["compressed_bytes"])
    if raw_total+hex_chars//2+score_bytes>config.max_raw_bytes:
        raise AcquisitionCapError("payload_raw_array_cap")
    trace_nonhex=max(0,trace_size-hex_chars-score_chars)
    metadata_bound=6*(meta_size+trace_nonhex+32768)
    if metadata_bound>config.max_metadata_bytes:
        raise AcquisitionCapError("payload_metadata_cap")
    encoded_arrays=sum(4*((value.nbytes+2)//3) for value in arrays.values())
    member_bound=metadata_bound+encoded_arrays+hex_chars+score_chars
    if member_bound>config.max_member_bytes:
        raise AcquisitionCapError("payload_member_cap")
    # Raw tobytes, base64 bytes/string, JSON text and UTF-8 temporary coexist.
    lease=budget.acquire(3*member_bound+raw_total+65536)
    try:
        records={}
        for name,value in arrays.items():
            if value.dtype.hasobject or not value.flags.c_contiguous:
                raise AcquisitionIntegrityError("payload_array_contract")
            raw=value.tobytes()
            records[name]={"dtype":value.dtype.str,
                "dtype_descr":value.dtype.descr if value.dtype.fields else None,
                "shape":list(value.shape),"byte_count":len(raw),"sha256":_sha256(raw),
                "base64":base64.b64encode(raw).decode("ascii")}
        payload={**bindings,"schema":"VALID-NATIVE-CHAIN-01-member-v1",
                 "config_sha256":support["config_sha256"],"trace_result":traced,
                 "coordinate_transform":{"scale":1.,"offset":[0.,0.],"units":"native_center_pixels"},
                 "control_pre_length":"unobservable","arrays":records,
                 "split_counts":support["audit"]["counts"],
                 "acquisition_counts":None if extraction is None else extraction.counts}
        encoded=_canonical(payload).encode("utf-8")
        if len(encoded)>config.max_member_bytes:
            raise AcquisitionCapError("payload_member_cap")
        budget.acquire(len(encoded)+256)
        return encoded
    finally:
        records=payload=raw=None
        budget.release(lease)



def valid_sealed_bytes(value):
    if not isinstance(value,bytes) or len(value)>FROZEN_CONFIG.max_member_bytes:
        raise AcquisitionCapError("valid_seal_bytes_cap")
    return value


def _json_allocation_bound(payload):
    """Conservative parser/encoding object budget without parsing JSON first."""
    inside=False; escaped=False; structural=strings=0
    for byte in payload:
        if inside:
            if escaped:
                escaped=False
            elif byte==92:
                escaped=True
            elif byte==34:
                inside=False
        elif byte==34:
            inside=True; strings+=1
        elif byte in (123,91,44):
            structural+=1
    return 6*len(payload)+512*structural+128*strings+65536


class ValidAcquisitionMemberStore(GzipMemberStore):
    """Typed terminal facade with precharged compression/parser scratch.

    Coordinator callbacks pass the active support budget to append. Standalone
    use has its own budget; it still charges the caller's encoded payload.
    """
    def __init__(self,*args,**kwargs):
        kwargs["max_decoded_bytes"]=min(kwargs.get("max_decoded_bytes",MAX_DECODED_MEMBER_BYTES),
                                         FROZEN_CONFIG.max_member_bytes)
        super().__init__(*args,**kwargs)
        self._active_budget=None

    def append(self,payload,plan,identity,arm,*,budget=None):
        if not isinstance(payload,bytes) or len(payload)>self.max_decoded_bytes:
            raise AcquisitionCapError("valid_member_decoded_cap")
        supplied=budget is not None
        budget=budget or AllocationBudget(FROZEN_CONFIG.max_live_bytes)
        if not supplied:
            budget.acquire(len(payload)+256)
        plan_size=_managed_tree_size(plan,ceiling=8*1024*1024)
        plan_json_bytes=_exact_json_size(plan)+4096
        # The inherited implementation builds candidate plan JSON, compressed
        # bytes and canonical decoded text. Reserve their entire coexistence.
        lease=budget.acquire(_json_allocation_bound(payload)+plan_size+3*plan_json_bytes+65536)
        self._active_budget=budget
        try:
            return super().append(payload,plan,identity,arm)
        except AcquisitionTerminalError:
            raise
        except Exception as exc:
            category=AcquisitionCapError if "cap" in str(exc) else AcquisitionIntegrityError
            raise category("valid_member_store_failure",{
                "reason":str(exc)[:240],"identity":identity,"arm":arm,
                "sealed_prediction_count":len(plan.get("sealed_predictions",[])),
                "sidecar_bytes":self.sidecar_path.stat().st_size if self.sidecar_path.exists() else 0}) from exc
        finally:
            self.last_append_peak=budget.peak
            self._active_budget=None
            budget.release(lease)

    def read_member(self,descriptor):
        # During append, exact payload grammar has already been budgeted before
        # writing. Inherited CRC/hash checks bind readback to that same payload.
        if self._active_budget is not None:
            decoded=super().read_member(descriptor)
            _validate_valid_member_blob(decoded,self._active_budget)
            return decoded
        compressed_length=int(descriptor["compressed_bytes"])
        decoded_length=int(descriptor["decoded_bytes"])
        offset=int(descriptor["offset"])
        if (not 0<compressed_length<=self.max_compressed_bytes
                or not 0<=decoded_length<=self.max_decoded_bytes or offset<0):
            raise AcquisitionCapError("valid_member_descriptor_cap")
        budget=AllocationBudget(FROZEN_CONFIG.max_live_bytes)
        lease=budget.acquire(4*decoded_length+2*compressed_length+65536)
        try:
            with self.sidecar_path.open("rb") as stream:
                stream.seek(offset); compressed=stream.read(compressed_length)
            if len(compressed)!=compressed_length or _sha256(compressed)!=descriptor["compressed_sha256"]:
                raise AcquisitionIntegrityError("valid_member_compressed_integrity")
            with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode="rb") as stream:
                decoded=stream.read(decoded_length+1)
                if len(decoded)!=decoded_length or stream.read(1):
                    raise AcquisitionIntegrityError("valid_member_decoded_length")
            parser_lease=budget.acquire(_json_allocation_bound(decoded))
            try:
                # The inherited path independently verifies CRC, both digests,
                # canonical JSON and exact length; no unbounded parse precedes
                # this reservation.
                checked=super().read_member(descriptor)
                _validate_valid_member_blob(checked,budget)
                decoded=compressed=None
                return checked
            finally:
                budget.release(parser_lease)
        except AcquisitionTerminalError:
            raise
        except Exception as exc:
            raise AcquisitionIntegrityError("valid_member_read_failure",{
                "reason":str(exc)[:240],"offset":offset}) from exc
        finally:
            decoded=compressed=None
            self.last_read_peak=budget.peak
            budget.release(lease)



def validate_score_stream(packet,budget,*,expected_records=None,on_record=None):
    """Lossless streaming CRC/hash/count/schema validation; never expand a list."""
    if (not isinstance(packet,dict) or packet.get("encoding")!="gzip-json-lines-v1"
            or not isinstance(packet.get("base64"),str)):
        raise AcquisitionIntegrityError("score_stream_structure")
    count=int(packet["record_count"]); compressed_bytes=int(packet["compressed_bytes"])
    raw_bytes=int(packet["raw_bytes"])
    if not 0<=count<=120_000 or not 0<=raw_bytes<=64_120_000 or not 0<compressed_bytes<=16*1024*1024:
        raise AcquisitionCapError("score_stream_descriptor_cap")
    if expected_records is not None and count!=expected_records:
        raise AcquisitionIntegrityError("score_stream_count_binding")
    if len(packet["base64"])!=4*((compressed_bytes+2)//3):
        raise AcquisitionIntegrityError("score_stream_base64_length")
    lease=budget.acquire(3*compressed_bytes+1024*1024)
    compressed=line=record=None
    total=seen=0; digest=hashlib.sha256()
    try:
        compressed=base64.b64decode(packet["base64"],validate=True)
        if len(compressed)!=compressed_bytes or _sha256(compressed)!=packet["compressed_sha256"]:
            raise AcquisitionIntegrityError("score_stream_compressed_hash")
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode="rb") as stream:
            while True:
                line=stream.readline(128*1024+1)
                if not line:
                    break
                if len(line)>128*1024 or not line.endswith(b"\n") or seen>=count:
                    raise AcquisitionIntegrityError("score_stream_record_boundary")
                total+=len(line)
                if total>raw_bytes:
                    raise AcquisitionIntegrityError("score_stream_raw_length")
                digest.update(line)
                parsing=budget.acquire(_json_allocation_bound(line))
                try:
                    record=json.loads(line)
                    if not isinstance(record,dict) or (_canonical(record)+"\n").encode()!=line:
                        raise AcquisitionIntegrityError("score_stream_record_canonical")
                    if on_record is not None:
                        on_record(seen,record)
                finally:
                    record=None
                    budget.release(parsing)
                seen+=1
        if total!=raw_bytes or seen!=count or digest.hexdigest()!=packet["raw_sha256"]:
            raise AcquisitionIntegrityError("score_stream_raw_integrity")
        return {"records":seen,"raw_bytes":total,"compressed_bytes":compressed_bytes}
    except AcquisitionTerminalError:
        raise
    except Exception as exc:
        raise AcquisitionIntegrityError("score_stream_decode_failure",{
            "exception_type":type(exc).__name__,"validated_records":seen}) from exc
    finally:
        compressed=line=record=None
        budget.release(lease)


def _validate_valid_member_blob(encoded,budget):
    """Validate CAM-VALID replay structure in addition to outer member integrity.

    Generic canonical injected failure receipts retain the inherited store seam.
    Actual v1 members receive exact arrays, camera/trace and record checks.
    """
    parsing=budget.acquire(_json_allocation_bound(encoded))
    payload=raw=array=None
    try:
        payload=json.loads(encoded)
        if payload.get("schema")!="VALID-NATIVE-CHAIN-01-member-v1":
            return
        if (payload.get("control_pre_length")!="unobservable"
                or payload.get("arm") not in ("baseline","candidate")
                or not isinstance(payload.get("arrays"),dict)):
            raise AcquisitionIntegrityError("valid_member_structure")
        required={"original_mask","boundary_band","expanded_mask","detector_segments",
                  "emitted_segments","run_ledger","sample_occupancy"}
        if not required<=payload["arrays"].keys():
            raise AcquisitionIntegrityError("valid_member_missing_replay_arrays")
        total=0
        for name,meta in payload["arrays"].items():
            if not isinstance(meta,dict) or not isinstance(meta.get("base64"),str):
                raise AcquisitionIntegrityError("valid_member_array_structure")
            size=int(meta["byte_count"])
            total+=size
            if size<0 or total>FROZEN_CONFIG.max_raw_bytes or len(meta["base64"])!=4*((size+2)//3):
                raise AcquisitionCapError("valid_member_raw_array_cap")
            scratch=budget.acquire(3*size+65536)
            try:
                raw=base64.b64decode(meta["base64"],validate=True)
                if len(raw)!=size or _sha256(raw)!=meta["sha256"]:
                    raise AcquisitionIntegrityError("valid_member_array_hash")
                descriptor=meta.get("dtype_descr")
                dtype=np.dtype([tuple(field) for field in descriptor]) if descriptor else np.dtype(meta["dtype"])
                shape=meta.get("shape")
                if dtype.hasobject or not isinstance(shape,list) or len(shape)>2 or any(type(v) is not int or v<0 for v in shape):
                    raise AcquisitionIntegrityError("valid_member_array_dtype_shape")
                if math.prod(shape)*dtype.itemsize!=size or dtype.str!=meta["dtype"]:
                    raise AcquisitionIntegrityError("valid_member_array_byte_count")
                array=np.frombuffer(raw,dtype=dtype).reshape(shape)
                if name in ("detector_segments","emitted_segments","candidate_pre_length"):
                    if len(shape)!=2 or shape[1]!=4 or len(array)>10_000 or array.dtype.kind!="f" or not np.isfinite(array).all():
                        raise AcquisitionIntegrityError("valid_member_coordinate_contract")
                if name in ("original_mask","boundary_band","expanded_mask"):
                    if dtype!=np.bool_ or shape!=payload["rgb_shape"][:2]:
                        raise AcquisitionIntegrityError("valid_member_mask_contract")
            finally:
                raw=array=None
                budget.release(scratch)
        trace=payload["trace_result"].get("trace",{})
        inherited=trace.get("arrays",{})
        if not isinstance(inherited,dict):
            raise AcquisitionIntegrityError("valid_member_trace_arrays")
        inherited_bytes=0
        for digest,meta in inherited.items():
            if not isinstance(meta,dict) or not isinstance(meta.get("bytes_hex"),str):
                raise AcquisitionIntegrityError("valid_member_trace_hex")
            dtype=np.dtype(meta["dtype"]); shape=meta["shape"]
            if dtype.hasobject or not isinstance(shape,list) or len(shape)>3 or any(type(v) is not int or v<0 for v in shape):
                raise AcquisitionIntegrityError("valid_member_trace_dtype_shape")
            size=math.prod(shape)*dtype.itemsize
            inherited_bytes+=size
            if inherited_bytes>2_000_000 or len(meta["bytes_hex"])!=2*size:
                raise AcquisitionCapError("valid_member_inherited_array_cap")
            scratch=budget.acquire(8*size+65536)
            try:
                if _sha256(_canonical(meta))!=digest:
                    raise AcquisitionIntegrityError("valid_member_trace_snapshot_hash")
                raw=bytes.fromhex(meta["bytes_hex"])
                array=np.frombuffer(raw,dtype=dtype).reshape(shape)
                if not np.isfinite(array).all():
                    raise AcquisitionIntegrityError("valid_member_trace_snapshot_nonfinite")
            finally:
                raw=array=None
                budget.release(scratch)
        if inherited and trace.get("array_bytes")!=inherited_bytes:
            raise AcquisitionIntegrityError("valid_member_trace_array_total")
        total+=inherited_bytes
        scores=trace.get("scores")
        if isinstance(scores,dict):
            result=validate_score_stream(scores,budget,expected_records=trace.get("score_events"))
            if total+result["compressed_bytes"]>FROZEN_CONFIG.max_raw_bytes:
                raise AcquisitionCapError("valid_member_trace_raw_cap")
        elif scores is not None:
            raise AcquisitionIntegrityError("valid_member_score_representation")
    except AcquisitionTerminalError:
        raise
    except Exception as exc:
        raise AcquisitionIntegrityError("valid_member_structure_decode",{
            "exception_type":type(exc).__name__}) from exc
    finally:
        payload=raw=array=meta=trace=scores=result=None
        budget.release(parsing)


def _exact_json_size(value,depth=0):
    """Exact canonical ASCII JSON length without materializing a plan copy."""
    if depth>32:
        raise AcquisitionIntegrityError("plan_nesting")
    if isinstance(value,str):
        size=2
        for char in value:
            code=ord(char)
            if code in (34,92,8,12,10,13,9):
                size+=2
            elif code<32:
                size+=6
            elif code<128:
                size+=1
            else:
                size+=6 if code<=65535 else 12
        return size
    if isinstance(value,dict):
        if not all(isinstance(key,str) for key in value):
            raise AcquisitionIntegrityError("plan_key_type")
        return 2+max(0,len(value)-1)+sum(_exact_json_size(key,depth+1)+1+
                _exact_json_size(child,depth+1) for key,child in value.items())
    if isinstance(value,(tuple,list)):
        return 2+max(0,len(value)-1)+sum(_exact_json_size(v,depth+1) for v in value)
    if value is None:
        return 4
    if isinstance(value,bool):
        return 4 if value else 5
    if type(value) in (int,float):
        if isinstance(value,float) and not math.isfinite(value):
            raise AcquisitionIntegrityError("plan_nonfinite")
        return len(_canonical(value))
    raise AcquisitionIntegrityError("plan_scalar_type")



