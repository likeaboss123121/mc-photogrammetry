"""Evaluation-only worker for one frozen signature replay campaign batch."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

from PIL import Image

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = (_REPOSITORY_ROOT / "src").resolve()
for _index, _path in enumerate((_SOURCE_ROOT, _REPOSITORY_ROOT.resolve())):
    if str(_path) not in sys.path:
        sys.path.insert(_index, str(_path))

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling
from mc_photogrammetry.dataset.schemas import GroundTruthExport
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import camera as camera_mod, lattice, lattice_planes
from scripts import signature_replay_batch_campaign as campaign


def _rss() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t)] + [
                        (name, ctypes.c_size_t) for name in
                        ("qpp", "qp", "qnp", "qn", "page", "peak_page")]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    value = Counters(); value.cb = ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if value.working <= 0:
        raise RuntimeError("invalid RSS sample")
    return int(value.working)


def execute_batch(*, manifest, batch_index, regenerated, evidence_by_frame,
                  replay=sampling.replay_signature_sampling_witness,
                  rss=_rss, clock=time.perf_counter):
    selected = campaign.select_batch(manifest, batch_index, regenerated)
    baseline = rss(); peak = [baseline]; running = [True]
    def monitor():
        while running[0]:
            peak[0] = max(peak[0], rss()); time.sleep(.02)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    started = clock(); pending = []
    budget = lattice.WitnessBudgetV1(*manifest.caps)
    try:
        for item in selected:
            result = replay(signature_evidence=item,
                            truth_evidence=evidence_by_frame[item.frame_id],
                            witness_budget=budget)
            pending.append(campaign.BatchRow(
                (item.frame_id, item.canonical_index, item.alternative_index,
                 item.signature_sha256), result))
            gc.collect()
        elapsed = clock() - started
        gc.collect(); post = rss(); peak[0] = max(peak[0], post)
        return campaign.BatchArtifact(manifest.manifest_sha256, batch_index,
                                      "complete", "", tuple(pending), baseline,
                                      peak[0], post, elapsed)
    finally:
        running[0] = False; thread.join(1)


def _regenerate(args):
    truth_path, jar_path, palette_path = map(Path, (args.truth, args.jar, args.palette))
    truth = GroundTruthExport.from_dict(json.loads(truth_path.read_text(encoding="utf-8")))
    capture = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    base = [camera_mod.CameraModel.from_dict(row) for row in capture["cameras"]]
    positions = ((173.758, 88.966, 379.706), (172.408, 101.381, 369.362),
                 (172.695, 94.533, 358.268), (164.372, 88.024, 386.652))
    from dataclasses import replace
    cameras = tuple(replace(base[index], position=position)
                    for index, position in zip((1, 2, 3, 0), positions))
    frames = tuple(f"image-{index:04d}" for index in range(4))
    evidences = []
    for camera in cameras:
        width, height = camera.intrinsics.image_size
        evidences.append(truth_mod.render_truth_evidence(
            truth, camera, width=width, height=height,
            truth_artifact_path=truth_path, vanilla_jar_path=jar_path,
            palette_cache_path=palette_path))
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mc-photo-batch-worker-") as directory:
        paths = []
        for frame, evidence in zip(frames, evidences):
            path = Path(directory) / f"{frame}.png"
            Image.fromarray(evidence.rgb, "RGB").save(path); paths.append(path)
        entry = entry_mod.build_image_derived_entry(paths, project_path=args.project)
        measured = lattice_mod.measure_image_derived_lattice(
            session=entry.session, pose_graph=entry.pose_graph)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=entry.session, lattice_result=measured)
    return signatures.signatures, dict(zip(frames, evidences))


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("be", "truth", "jar", "palette", "capture", "project", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--memory-limit", required=True, type=int)
    parser.add_argument("--deadline", required=True, type=int)
    args = parser.parse_args(argv)
    manifest = campaign.CampaignManifest()
    if (args.manifest_sha256 != manifest.manifest_sha256
            or args.memory_limit != manifest.memory_limit_bytes
            or args.deadline != manifest.deadline_seconds):
        raise ValueError("worker invocation differs from frozen manifest")
    campaign.verify_sources((args.be, args.truth, args.jar, args.palette,
                             Path(__file__), campaign.Path(campaign.__file__).parent.parent /
                             "src/mc_photogrammetry/synthetic_targeted_capture_preflight.py"))
    regenerated, evidence_by_frame = _regenerate(args)
    artifact = execute_batch(manifest=manifest, batch_index=args.batch_index,
                             regenerated=regenerated,
                             evidence_by_frame=evidence_by_frame)
    campaign.atomic_write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
