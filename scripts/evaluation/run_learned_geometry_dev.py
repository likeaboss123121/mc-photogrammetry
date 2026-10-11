"""One-shot fresh train-scene CPU smoke; runtime requires a separate lead GO."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".venv-vggt"
VENDOR = ENV / "vendor"
DATA = ROOT / "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures"
NAMES = tuple(f"view-canal-town-architectural-component-bbox-{i:02d}.png" for i in range(4))
INPUT_SHA256 = (
    "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c",
    "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696",
    "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95",
    "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9",
)
OUTPUT = ROOT / "exports/diagnostics/vggt-c1-fresh-canal-town-framed-2026-10-07"
CACHE = Path("C:/Users/Likea/.cache/huggingface/hub/models--facebook--VGGT-1B")
COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
WEIGHT_SHA = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
WEIGHT_BYTES = 5026367224
GIB = 1024 ** 3


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class ReceiptPublisher:
    """Single serialized receipt owner; a resource refusal dominates all updates."""

    def __init__(self, path, writer=write_json):
        self.path, self.writer = path, writer
        self.lock = threading.RLock()
        self.snapshot = {}
        self.terminal_priority = 0

    def publish(self, value, resource_failure=False):
        with self.lock:
            priority = 3 if resource_failure else (2 if value.get("status") == "BLOCKED" else
                                                  1 if value.get("status") == "COMPLETE" else 0)
            if self.terminal_priority > priority or (self.terminal_priority >= 2 and priority == self.terminal_priority):
                return copy.deepcopy(self.snapshot)
            candidate = copy.deepcopy(value)
            # Validate before changing ownership; terminal state survives writer failure.
            json.dumps(candidate, allow_nan=False)
            self.snapshot, self.terminal_priority = candidate, priority
            self.writer(self.path, candidate)
            return copy.deepcopy(candidate)

    def resource_failure(self, gate, metrics):
        with self.lock:
            value = copy.deepcopy(self.snapshot)
            value.update(metrics)
            value.update(status="BLOCKED", failure=gate)
            return self.publish(value, resource_failure=True)

    def finalize_telemetry(self, metrics):
        """Merge only final numeric counters without replacing terminal evidence."""
        with self.lock:
            telemetry = {key: copy.deepcopy(metrics[key]) for key in (
                "wall_seconds", "cpu_seconds", "peak_rss_bytes", "new_disk_bytes") if key in metrics}
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                   not math.isfinite(value) or value < 0 for value in telemetry.values()):
                raise ValueError("final telemetry requires finite nonnegative numeric counters")
            candidate = copy.deepcopy(self.snapshot)
            candidate.update(telemetry)
            json.dumps(candidate, allow_nan=False)
            self.snapshot = candidate  # terminal priority and all failure fields stay unchanged
            self.writer(self.path, candidate)
            return copy.deepcopy(candidate)


def stop_for_resource_failure(publisher, gate, metrics, terminate=os._exit):
    """Receipt I/O failure must never disable the resource stop."""
    try:
        publisher.resource_failure(gate, metrics)
    finally:
        terminate(3)


def directory_bytes(roots):
    return sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob("*") if p.is_file())


def write_arrays(path, **arrays):
    """Publish one small frame artifact atomically before validation gates."""
    import numpy as np
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def residual_summary(errors, pixels):
    import numpy as np
    values = np.asarray(errors)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("finite nonempty residuals required")
    worst = int(np.argmax(values))
    return {"maximum_pixels": float(values[worst]), "argmax_index": worst,
            "argmax_model_pixel": np.asarray(pixels)[worst].tolist()}


def process_geometry_predictions(output, state, extrinsics, intrinsics, depths, confidences,
                                 transforms, array_writer=write_arrays, receipt_writer=write_json,
                                 repository_projector=None, publisher=None):
    """Pure numeric artifact boundary; injectable writers support synthetic tests.

    All raw frames are saved first, including nonfinite/refused predictions.
    Canonical matrices and diagnostic arrays precede every parity refusal.
    Does not load a model, images, metadata or checkpoint.
    """
    import numpy as np
    from mc_photogrammetry.reconstruction.camera import project_points
    from mc_photogrammetry.reconstruction.learned_geometry_adapter import (
        FLOAT64_UNIT_ROUNDOFF, ScaleUncertainty, canonical_roundtrip_roundoff_estimate,
        canonicalize_pose, projection_delta_bound, project_world, repository_camera,
        rotation_diagnostics, source_intrinsic, unproject_axial, validate_depth_confidence,
    )
    projector = project_points if repository_projector is None else repository_projector
    receipt = output / "run.json"
    publisher = publisher or ReceiptPublisher(receipt, receipt_writer)
    state["stage"] = "raw_prediction_preservation"
    try:
        for index, frame in enumerate(state["frames"]):
            raw_path = output / f"frame-{index:02d}-raw.npz"
            frame.update(raw_artifact=raw_path.name,
                         raw_dtypes={name: str(np.asarray(array[index]).dtype) for name, array in (
                             ("extrinsic", extrinsics), ("intrinsic", intrinsics),
                             ("depth", depths), ("confidence", confidences))},
                         preprocessing=transforms[index].to_dict())
            array_writer(raw_path, frame_id=np.asarray(frame["frame_id"]),
                         extrinsic=extrinsics[index], intrinsic=intrinsics[index], depth=depths[index],
                         confidence=confidences[index], source_to_model=transforms[index].affine,
                         valid_source_mask=transforms[index].valid_source_mask())
        # Optional linear algebra diagnostics run only after ALL raw writes.
        for index, frame in enumerate(state["frames"]):
            frame["raw_rotation"] = rotation_diagnostics(extrinsics[index])
        publisher.publish(state)
        if not all(np.isfinite(v).all() for v in (extrinsics, intrinsics, depths, confidences)):
            state["stage"] = "raw_finite_validation"
            raise ValueError("nonfinite raw model output; raw evidence preserved")
        state["finite_outputs"] = True
        for index, (frame, transform) in enumerate(zip(state["frames"], transforms)):
            state["failure_frame_id"] = frame["frame_id"]
            state["stage"] = "rotation_canonicalization"
            pose = canonicalize_pose(extrinsics[index], intrinsics[index])
            e, k = pose.extrinsic, pose.intrinsic
            frame["rotation_policy"] = pose.diagnostics
            frame["canonical_extrinsic"] = e.tolist()
            frame["model_intrinsic"] = k.tolist()
            ks = source_intrinsic(k, transform)
            if not np.isfinite(ks).all():
                raise ValueError("nonfinite source intrinsic")
            frame["source_intrinsic"] = ks.tolist()
            canonical_path = output / f"frame-{index:02d}-canonical.npz"
            frame["canonical_artifact"] = canonical_path.name
            array_writer(canonical_path, extrinsic=e, intrinsic=k, source_intrinsic=ks)
            publisher.publish(state)
            state["stage"] = "depth_confidence_validation"
            mask = validate_depth_confidence(depths[index], confidences[index], transform.valid_source_mask())
            if not mask.any():
                raise ValueError("empty valid-source evidence")
            yy, xx = np.nonzero(mask)
            pixels = np.column_stack((xx, yy))
            axial = depths[index][mask].astype(np.float64)
            expected = transform.to_source(pixels)
            state["stage"] = "canonical_algebra"
            world = unproject_axial(pixels, axial, e, k)
            source, _ = project_world(world, e, ks)
            canonical_errors = np.linalg.norm(source - expected, axis=1)
            frame["canonical_roundtrip"] = residual_summary(canonical_errors, pixels)
            state["stage"] = "roundoff_diagnostic_estimation"
            # Scale a partial diagnostic estimate; affine construction is omitted.
            source_error_scale = float(np.linalg.norm(np.linalg.inv(transform.affine)[:2, :2], 2))
            estimate_reasons = []
            try:
                canonical_bound = canonical_roundtrip_roundoff_estimate(pixels, axial, pose) * source_error_scale
                frame["canonical_roundoff_estimate"] = residual_summary(canonical_bound, pixels)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                canonical_bound = np.full(len(pixels), np.nan)
                frame["canonical_roundoff_estimate"] = None
                estimate_reasons.append(str(exc))
            frame["roundoff_estimate_policy"] = {
                "certified": False, "used_for_acceptance": False,
                "omissions": "rounded inverse affine/source K/homogeneous cancellation/repository centre and Euler arithmetic",
                "unavailable_reasons": estimate_reasons}
            translation_norm = float(np.linalg.norm(e[:, 3]))
            frame["translation_norm"] = translation_norm if np.isfinite(translation_norm) else None
            if not np.isfinite(translation_norm):
                estimate_reasons.append("translation norm unavailable: overflow")
            frame["minimum_axial_depth"] = float(axial.min())

            # Raw matrices use a true inverse explicitly, never transpose.
            state["stage"] = "raw_matrix_roundtrip"
            view = np.column_stack(((pixels - k[:2, 2]) / np.diag(k)[:2] * axial[:, None], axial))
            raw_e = pose.raw_extrinsic.astype(np.float64)
            raw_world = np.linalg.solve(raw_e[:, :3], (view - raw_e[:, 3]).T).T
            raw_round_camera = raw_world @ raw_e[:, :3].T + raw_e[:, 3]
            raw_round_hom = raw_round_camera @ ks.T
            if not np.isfinite(raw_round_hom).all() or (raw_round_camera[:, 2] <= 0).any():
                raise ValueError("nonfinite/nonpositive raw inverse roundtrip")
            raw_round_source = raw_round_hom[:, :2] / raw_round_hom[:, 2:3]
            raw_round_errors = np.linalg.norm(raw_round_source - expected, axis=1)
            frame["raw_matrix_inverse_roundtrip"] = residual_summary(raw_round_errors, pixels)

            state["stage"] = "raw_to_canonical_projection"
            can_view = world @ e[:, :3].T + e[:, 3]
            raw_view = world @ raw_e[:, :3].T + raw_e[:, 3]
            if not np.isfinite(raw_view).all() or (raw_view[:, 2] <= 0).any():
                raise ValueError("nonfinite/nonpositive raw-to-canonical projection")
            raw_hom = raw_view @ ks.T
            raw_projection = raw_hom[:, :2] / raw_hom[:, 2:3]
            raw_delta = np.linalg.norm(raw_projection - source, axis=1)
            frame["raw_to_canonical_projection_delta"] = residual_summary(raw_delta, pixels)
            u = FLOAT64_UNIT_ROUNDOFF
            gamma4 = 4 * u / (1 - 4 * u)
            rounding = gamma4 * ((np.abs(raw_e[:, :3]) + np.abs(e[:, :3])) @ np.abs(world).T).T
            rounding += gamma4 * 2 * np.abs(e[:, 3])
            displacement_bound = pose.diagnostics["correction_frobenius"] * np.linalg.norm(world, axis=1)
            displacement_bound += np.linalg.norm(rounding, axis=1)
            try:
                raw_delta_bound = projection_delta_bound(can_view, displacement_bound, ks)
                raw_delta_bound += gamma4 * (np.linalg.norm(raw_projection, axis=1) + np.linalg.norm(source, axis=1))
                frame["raw_to_canonical_delta_estimate"] = residual_summary(raw_delta_bound, pixels)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                raw_delta_bound = np.full(len(pixels), np.nan)
                frame["raw_to_canonical_delta_estimate"] = None
                estimate_reasons.append(str(exc))
            frame["raw_to_canonical_projection_delta"] = residual_summary(raw_delta, pixels)

            state["stage"] = "repository_projection"
            camera = repository_camera(e, ks, transform.source_size)
            frame["repository_camera"] = camera.to_dict()
            samples = np.linspace(0, len(world) - 1, min(200, len(world)), dtype=int)
            repository_pixels = np.asarray(projector(world[samples], camera, width=transform.source_size[0],
                                                     height=transform.source_size[1]), dtype=float)
            if repository_pixels.shape != (len(samples), 3) or not np.isfinite(repository_pixels).all() or (repository_pixels[:, 2] <= 0).any():
                raise ValueError("nonfinite/nonpositive repository projection")
            parity_errors = np.linalg.norm(repository_pixels[:, :2] - source[samples], axis=1)
            frame["canonical_repository_parity"] = residual_summary(parity_errors, pixels[samples])
            diagnostics_path = output / f"frame-{index:02d}-diagnostics.npz"
            frame["diagnostics_artifact"] = diagnostics_path.name
            frame["valid_source_pixels"] = int(mask.sum())
            frame["uncertainty"] = ScaleUncertainty().to_dict()
            array_writer(diagnostics_path, model_pixels=pixels, source_expected=expected,
                         canonical_source_reprojection=source, canonical_roundtrip_errors=canonical_errors,
                         canonical_roundtrip_roundoff_estimate=canonical_bound,
                         raw_matrix_source_roundtrip=raw_round_source, raw_matrix_roundtrip_errors=raw_round_errors,
                         raw_to_canonical_projection=raw_projection, raw_to_canonical_delta=raw_delta,
                         raw_to_canonical_delta_estimate=raw_delta_bound, repository_sample_indices=samples,
                         repository_projection=repository_pixels, repository_parity_errors=parity_errors)
            publisher.publish(state)
            # Measured algebra gates stay 1e-5; partial estimates are diagnostic only.
            gates = [("raw_matrix_roundtrip", raw_round_errors.max() > 1e-5),
                     ("canonical_algebra_parity", canonical_errors.max() > 1e-5),
                     ("canonical_repository_parity", parity_errors.max() > 1e-5)]
            failed = next((name for name, refused in gates if refused), None)
            if failed:
                state["stage"] = failed
                raise ValueError(f"{failed} refused; raw and computed diagnostics preserved")
            frame["status"] = "VALIDATED"
            publisher.publish(state)
        state.pop("failure_frame_id", None)
        state.update(status="GEOMETRY_VALIDATED", stage="geometry_validated")
        publisher.publish(state)
    except Exception as exc:
        if getattr(exc, "diagnostics", None) is not None:
            state["frames"][index]["rotation_policy"] = exc.diagnostics
        state.update(status="BLOCKED", failure=type(exc).__name__, detail=str(exc))
        publisher.publish(state)
        raise


def complete_required_artifacts(output, state, publisher, build_gallery, validate_artifact):
    """Outer completion gate, after gallery construction and artifact validation."""
    try:
        if state.get("status") != "GEOMETRY_VALIDATED":
            raise ValueError("geometry validation required before completion")
        state["stage"] = "source_reprojection_gallery"
        publisher.publish(state)
        for index, frame in enumerate(state["frames"]):
            state["failure_frame_id"] = frame["frame_id"]
            build_gallery(index, frame)
            frame["gallery_artifact"] = f"frame-{index:02d}-reprojection.png"
        state["stage"] = "required_artifact_validation"
        for frame in state["frames"]:
            state["failure_frame_id"] = frame["frame_id"]
            for kind in ("raw", "canonical", "diagnostics", "gallery"):
                validate_artifact(output / frame[f"{kind}_artifact"], kind)
        state.pop("failure_frame_id", None)
        state.update(status="COMPLETE", stage="complete", finite_outputs=True,
                     reprojection_semantics="same-view algebra/preprocessing check; no scene accuracy or cross-view consistency claim")
        published = publisher.publish(state)
        state.update(published)
    except Exception as exc:
        state.update(status="BLOCKED", failure=type(exc).__name__, detail=str(exc))
        publisher.publish(state)
        raise


def validate_output_artifact(path, kind):
    """Validate required output payloads; this function never reads input images."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty {kind} artifact: {path.name}")
    if kind == "gallery":
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    else:
        import numpy as np
        required = {"raw": ("extrinsic", "intrinsic", "depth", "confidence", "source_to_model", "valid_source_mask"),
                    "canonical": ("extrinsic", "intrinsic", "source_intrinsic"),
                    "diagnostics": ("canonical_roundtrip_errors", "repository_parity_errors")}[kind]
        with np.load(path, allow_pickle=False) as arrays:
            for key in required:
                if not arrays[key].size or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"invalid {kind} artifact field: {key}")


def bind_inputs(raw_images, raw_output):
    images = tuple(Path(os.path.abspath(p)) for p in raw_images)
    expected = tuple(DATA.absolute() / name for name in NAMES)
    output = Path(os.path.abspath(raw_output))
    if images != expected or output != OUTPUT.absolute():
        raise ValueError("fresh validation permits only the four exact ordered PNGs and fresh output")
    # Compare against fixed lexical paths, not resolved expectations that could
    # silently follow an input or output ancestor into the consumed C1 packet.
    if any(p.resolve() != p for p in images) or output.resolve() != output:
        raise ValueError("fresh validation refuses redirected input/output paths")
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=4, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    images = bind_inputs(args.images, args.output_dir)
    if Path(sys.prefix).resolve() != ENV.resolve() or sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError("C1 requires the exact isolated Python 3.12.10 environment")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    # Persistent exclusive receipt consumes this one CPU fallback even on failure.
    receipt = OUTPUT / "run.json"
    with receipt.open("x", encoding="utf-8") as f:
        pass  # exclusive one-shot reservation; all receipt content has one publisher
    start = time.monotonic()
    state = {"status": "STARTED", "stage": "setup", "cpu_fallback_processes": 1,
             "gpu_forwards": 0, "gpu_memory_bytes": None, "device": "cpu",
             "model_revision": REVISION, "checkpoint_sha256": WEIGHT_SHA,
             "checkpoint_bytes": WEIGHT_BYTES, "upstream_code_commit": COMMIT,
             "source_sha256": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "adapter": hashlib.sha256((ROOT / "src/mc_photogrammetry/reconstruction/learned_geometry_adapter.py").read_bytes()).hexdigest()},
             "expected_input_sha256": dict(zip(NAMES, INPUT_SHA256)),
             "model_license": "CC-BY-NC-4.0; private local research; no redistribution",
             "input_role": {"run": "developer_runtime_validation", "split": "train",
                            "advanced_only": True, "normal_mode_validation": False,
                            "accuracy_claim": False},
             "scale": {"metric": "unknown", "blocks": "unknown", "orientation": "model gauge"},
             "accuracy": "unknown", "renderer_version": "not read; no renderer launched",
             "network": "offline; cached checkpoint only; no credentials/contact/upload/download",
             "caps": {"disk_bytes": 9 * GIB, "rss_bytes": 24 * GIB, "wall_seconds": 1200,
                      "cpu_seconds": 3600, "gpu_vram_bytes": 14 * GIB},
             "peak_rss_bytes": 0, "cpu_seconds": 0, "wall_seconds": 0,
             "frames": [], "dataset_open_events": [], "cpu_forwards": 0}
    publisher = ReceiptPublisher(receipt)
    publisher.publish(state)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      TOKENIZERS_PARALLELISM="false", TEMP=str(OUTPUT), TMP=str(OUTPUT))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "src"), str(VENDOR)]
    import psutil
    process = psutil.Process()
    done = threading.Event()
    lock = threading.RLock()
    peak_rss_bytes = 0

    def sample():
        nonlocal peak_rss_bytes
        memory = process.memory_info()
        with lock:
            peak_rss_bytes = max(peak_rss_bytes, memory.rss, getattr(memory, "peak_wset", 0))
            return {"peak_rss_bytes": peak_rss_bytes,
                    "cpu_seconds": sum(process.cpu_times()[:2]),
                    "wall_seconds": time.monotonic() - start,
                    "new_disk_bytes": directory_bytes([OUTPUT])}

    def watchdog():
        while not done.wait(.5):
            try:
                metrics = sample()
                gate = next((key for key, value, cap in (
                    ("RSS_CAP", metrics["peak_rss_bytes"], 24 * GIB),
                    ("CPU_COMPUTE_CAP", metrics["cpu_seconds"], 3600),
                    ("WALL_TIME_CAP", metrics["wall_seconds"], 1200),
                    ("NEW_DISK_CAP", metrics["new_disk_bytes"], 9 * GIB)) if value >= cap), None)
                if gate:
                    stop_for_resource_failure(publisher, gate, metrics)
            except Exception as exc:
                stop_for_resource_failure(publisher, "RESOURCE_MONITOR_FAILURE", {"detail": str(exc)})

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    permitted = set(images)

    def audit(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            dataset_root = (ROOT / "exports/datasets").resolve()
            if path.is_relative_to(dataset_root):
                if path not in permitted:
                    raise PermissionError(f"C1 evaluator/data isolation: {path.name}")
                with lock:
                    state["dataset_open_events"].append(path.name)
            mode, flags = arguments[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing and not path.is_relative_to(OUTPUT.absolute()):
                raise PermissionError("fresh validation writes only inside its exclusive output")
        if event == "socket.connect":
            raise PermissionError("fresh validation is offline; network access forbidden")

    sys.addaudithook(audit)
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from safetensors.torch import load_model
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from mc_photogrammetry.reconstruction.learned_geometry_adapter import (
            ScaleUncertainty, project_world, repository_camera, source_intrinsic,
            unproject_axial, validate_camera, validate_depth_confidence, vggt_transforms,
        )
        if torch.__version__ != "2.9.1+cpu":
            raise RuntimeError("exact official CPU torch version required")
        if importlib.metadata.version("torchvision") != "0.24.1+cpu":
            raise RuntimeError("exact existing CPU torchvision version required")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        state["python"] = sys.version
        state["interpreter"] = sys.executable
        state["dependencies"] = {d: importlib.metadata.version(d) for d in (
            "torch", "torchvision", "numpy", "pillow", "huggingface-hub", "einops", "safetensors", "psutil")}
        state["cpu_threads"] = 8
        state["dtype"] = "float32; no autocast"
        state["code_license"] = {"name": "VGGT License, v1 July 29 2025",
                                 "sha256": hashlib.sha256((VENDOR / "LICENSE.txt").read_bytes()).hexdigest()}
        source_pin = json.loads((ENV / "upstream-source.json").read_text(encoding="utf-8"))
        if source_pin["commit"] != COMMIT:
            raise RuntimeError("upstream source pin mismatch")
        state["upstream_source"] = source_pin
        checkpoint = CACHE / "snapshots" / REVISION / "model.safetensors"
        if not checkpoint.is_file() or checkpoint.stat().st_size != WEIGHT_BYTES:
            raise RuntimeError("exact existing cached checkpoint is absent or wrong size; no download")
        state["stage"] = "cached_checkpoint_verification"
        digest, count = hashlib.sha256(), 0
        with checkpoint.open("rb") as source:
            while chunk := source.read(8 * 1024 ** 2):
                count += len(chunk)
                if count > WEIGHT_BYTES:
                    raise RuntimeError("checkpoint byte-size exceeded")
                digest.update(chunk)
        if count != WEIGHT_BYTES or digest.hexdigest() != WEIGHT_SHA:
            raise RuntimeError("checkpoint file checksum/size mismatch")
        state["checkpoint_path"] = str(checkpoint)
        state["checkpoint_verified"] = True
        state["stage"] = "input_pixels"
        print("Checkpoint verified; loading exact four screenshot pixels", flush=True)
        sizes = []
        for path, expected_hash in zip(images, INPUT_SHA256):
            data = path.read_bytes()
            image_hash = hashlib.sha256(data).hexdigest()
            if image_hash != expected_hash:
                raise RuntimeError(f"fixed input hash mismatch: {path.name}")
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("input is not PNG")
            with Image.open(path) as image:
                sizes.append(image.size)
            state["frames"].append({"frame_id": path.name, "sha256": image_hash,
                                    "source_size": sizes[-1]})
        transforms = vggt_transforms(sizes, mode="crop")
        batch = load_and_preprocess_images([str(p) for p in images], mode="crop")
        for frame, transform in zip(state["frames"], transforms):
            if tuple(reversed(batch.shape[-2:])) != transform.model_size:
                raise RuntimeError("upstream preprocessing size mismatch")
            frame["preprocessing"] = transform.to_dict()
        state["stage"] = "model_load"
        print("Loading model weights on CPU", flush=True)
        model = VGGT()
        load_model(model, str(checkpoint), strict=True, device="cpu")
        model.point_head = None
        model.track_head = None
        model.eval()
        gc.collect()
        state["model_loaded"] = True
        state["enabled_heads"] = ["camera", "depth"]
        state["stage"] = "forward"
        publisher.publish(state)
        forward_start = time.monotonic()
        state["cpu_forwards"] = 1
        print(f"Single 4-view CPU forward: {tuple(batch.shape)}", flush=True)
        with torch.inference_mode():
            prediction = model(batch)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], batch.shape[-2:])
        state["forward_wall_seconds"] = time.monotonic() - forward_start
        e = extrinsic[0].cpu().numpy()
        k = intrinsic[0].cpu().numpy()
        depths = prediction["depth"][0, ..., 0].cpu().numpy()
        confidences = prediction["depth_conf"][0].cpu().numpy()
        process_geometry_predictions(OUTPUT, state, e, k, depths, confidences, transforms, publisher=publisher)

        def build_gallery(index, frame):
            path = images[index]
            with np.load(OUTPUT / frame["diagnostics_artifact"], allow_pickle=False) as diagnostic:
                source_pixels = diagnostic["canonical_source_reprojection"]
                sample_indices = diagnostic["repository_sample_indices"]
            with Image.open(path) as image:
                overlay = image.convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for x, y in source_pixels[sample_indices]:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 0), width=2)
            overlay.save(OUTPUT / f"frame-{index:02d}-reprojection.png")
            frame.update(depth_range=[float(depths[index].min()), float(depths[index].max())],
                         confidence_range=[float(confidences[index].min()), float(confidences[index].max())])
        complete_required_artifacts(OUTPUT, state, publisher, build_gallery, validate_output_artifact)
    except Exception as exc:
        state.update(status="BLOCKED", failure=type(exc).__name__, detail=str(exc))
        traceback.print_exc()
    finally:
        done.set()
        thread.join(timeout=2)
        publisher.publish(state)
        state.update(publisher.finalize_telemetry(sample()))
        print(json.dumps({k: state.get(k) for k in ("status", "stage", "failure", "detail", "wall_seconds", "cpu_seconds", "peak_rss_bytes", "new_disk_bytes", "forward_wall_seconds")}), flush=True)
    return 0 if state["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
