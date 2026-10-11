"""One-shot development acquisition; fixed sources, no geometry/camera scoring.

The CLI is disabled until the independent lead driver review explicitly binds
these code bytes and the fresh output path. Imports do not read image inputs.
An orphan roster without completion.json is not successful evidence.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "exports/diagnostics/endpoint-conflict-native-roster-2026-10-09-repair-01"
DIAGNOSTIC = "exports/diagnostics/endpoint-conflict-mask-bundle-2026-10-09/diagnostic.json"
DIAGNOSTIC_SHA = "a4bb71e275b277a8b52a8b14884cb72a8be71df698a5f9e4c0d44894daf3aa98"
REVIEW = "agent swarm instantiation template/runs/RESULT-LEAD-ENDPOINT-CONFLICT-NATIVE-ROSTER-AUTH-REPAIR-01-2026-10-09.json"
DRIVER = "scripts/evaluation/acquire_endpoint_conflict_native_roster.py"
TEST = "tests/test_endpoint_conflict_native_roster.py"
CAPTURE = "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/"
INPUTS = (
    (CAPTURE + "view-canal-town-architectural-component-bbox-05.png", "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"),
    (CAPTURE + "view-canal-town-architectural-component-bbox-06.png", "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923"),
)
SOURCE_PINS = {
    "scripts/evaluation/image_point_candidate_roster.py": "0e9a027e1209f3a96c9e2bd7443311a14e654f7bd3440f30cab0404f32f28ca2",
    "scripts/evaluation/diagnose_image_geometry_failure.py": "5c3bdb9ed5dc34b021f3e60c92d0b7cd5ffa2aba8aea35225354213107d0d2af",
    "src/mc_photogrammetry/input/image.py": "43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732",
    "src/mc_photogrammetry/pipeline/correspondence.py": "192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1",
    "src/mc_photogrammetry/pipeline/image_derived_session.py": "e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e",
    "src/mc_photogrammetry/vision/gui_mask.py": "1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py": "c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b",
    "src/mc_photogrammetry/pipeline/image_derived_signatures.py": "7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py": "fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e",
    "src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py": "f8065946eaedb95cdbb8d2bfceace0246bf1369d2f8683f2fd555aa1904a7c0a",
    "src/mc_photogrammetry/pipeline/image_derived_joint_graph.py": "60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab",
}
# endpoint, native region ID, header+packed crop SHA, full native bool C SHA.
MASK_PINS = (
    (("image-0000", 2), "338a47a2cb081a1991014fd59b9413f04e2769195f39c373335e31dec9d25098", "3ecfb4b4709962ecbc5a8e122858c54a5c3334e7a0451ea55ce02bcb0381019a", "bd86f23c0191e91a9c93220c50e798fabd5e3e43c576a5fd7cd5b8bd6e94d708"),
    (("image-0000", 3), "61b803612ab055368b8bdab037441f780b9014eace8c117b41da6b28cd77d55d", "85b28d1243dd715b2bddb02b129bd24df4e868129541b1b71d29f68e0354367c", "cc7f4bbdca723ea630ddd7a2d3abfb3b97632ee8cc1569ac6465b7b8aaeb989e"),
    (("image-0001", 1), "d28eb153b89b3850b00baa218a90584ff92fa4b036048551362fdff50c832d33", "1e275ace2937394b743e1af64b16c5e767a579f00997123d2d034a63f5211977", "e557cba4a4a0401e6a2ebd5f7146ecee61d1727348c5e68310db7ac49e32aaf9"),
)
GROUP_IDS = ("row-176-plane-2-to-1", "row-197-plane-3-to-1")
WALL_SECONDS = 180
RSS_BYTES = 2 * 1024**3
ARTIFACT_BYTES = 8 * 1024**2
SCHEMA = "endpoint-conflict-native-roster-acquisition-v1"


class Stop(RuntimeError):
    """Terminal failure; no retry or substitute source is permitted."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Stop("duplicate_json_key")
        result[key] = value
    return result


def strict_json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(Stop("nonfinite_json")))
    except (ValueError, TypeError, RecursionError) as error:
        raise Stop("invalid_json") from error


def read_bounded(path, limit):
    with Path(path).open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise Stop("source_byte_limit")
    return data


def pinned_bytes(path, expected, limit):
    data = read_bounded(path, limit)
    if sha(data) != expected.lower():
        raise Stop("source_pin_mismatch: " + str(path))
    return data


def authorize(root=ROOT):
    """Review and all code/mask pins precede any encoded image access."""
    review_bytes = read_bounded(root / REVIEW, 65536)
    review = strict_json(review_bytes)
    if (review.get("disposition") != "ACCEPT_DRIVER; AUTHORIZE_ONE_ACQUISITION"
            or review.get("output") != OUTPUT or review.get("single_acquisition") is not True):
        raise Stop("independent_driver_review_required")
    expected_code = {**SOURCE_PINS, DRIVER: review.get("code_pins", {}).get(DRIVER),
                     TEST: review.get("code_pins", {}).get(TEST)}
    if any(type(pin) is not str or len(pin) != 64 for pin in expected_code.values()):
        raise Stop("review_code_pins_required")
    for name, expected in expected_code.items():
        pinned_bytes(root / name, expected, 4 * 1024**2)
    if review.get("input_pins") != {name: pin for name, pin in INPUTS}:
        raise Stop("review_input_pins_required")
    if review.get("diagnostic_sha256") != DIAGNOSTIC_SHA:
        raise Stop("review_mask_pin_required")
    data = pinned_bytes(root / DIAGNOSTIC, DIAGNOSTIC_SHA, 1024**2)
    record = strict_json(data)
    if canonical(record) + b"\n" != data:
        raise Stop("noncanonical_diagnostic")
    verify_mask_pins(record)
    return record, {"review_sha256": sha(review_bytes), "code_pins": expected_code,
                    "diagnostic_sha256": DIAGNOSTIC_SHA, "input_pins": dict(INPUTS)}


def verify_mask_pins(record, pins=MASK_PINS):
    entries = record["source_region_masks"]["entries"]
    if len(entries) != 3:
        raise Stop("exact_three_masks_required")
    for entry, (endpoint, region, packed, raw) in zip(entries, pins):
        if (tuple(entry["endpoint"]) != endpoint
                or entry["mask"]["source_region_id"] != region
                or entry["mask"]["mask_sha256"] != packed
                or entry["full_native_bool_c_sha256"] != raw):
            raise Stop("historical_native_mask_mismatch")


def _services():
    # No heavy imports, image inputs, detector, solver or geometry at import time.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from mc_photogrammetry.input.image import load_image_bytes
    from mc_photogrammetry.pipeline.correspondence import usable_mask_for
    from scripts.evaluation import image_point_candidate_roster as roster
    from scripts.evaluation import diagnose_image_geometry_failure as diagnostic
    return np, load_image_bytes, usable_mask_for, roster, diagnostic


def bind_frames(buffers, ingress, *, services=None):
    np, decode, usable_mask_for, roster, _ = services or _services()
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    if len(buffers) != 2 or len(ingress) != 2:
        raise Stop("exact_two_frames_required")
    # Validate BOTH encoded buffers before the first decode.
    for data, pin, frame_id in zip(buffers, ingress, ("image-0000", "image-0001")):
        if (type(data) is not bytes or pin["frame_id"] != frame_id
                or len(data) != pin["byte_length"] or sha(data) != pin["byte_sha256"]):
            raise Stop("encoded_ingress_mismatch")
        shape = pin["decoded_shape"]
        if (len(shape) != 3 or shape[2] != 3 or pin["decoded_dtype"] != "|u1"
                or any(type(v) is not int or v <= 0 for v in shape)
                or shape[0] * shape[1] > 4_000_000):
            raise Stop("native_ingress_shape")
        roster.identity(shape[1], shape[0], pin["transform"])
    frames = []
    for data, pin in zip(buffers, ingress):
        pixels = decode(data, suffix=".png")
        if (pixels.dtype != np.uint8 or list(pixels.shape) != pin["decoded_shape"]
                or image_sha256(pixels) != pin["decoded_pixel_sha256"]):
            raise Stop("decoded_ingress_mismatch")
        pixels = np.array(pixels, dtype=np.uint8, copy=True, order="C")
        usable = np.array(usable_mask_for(pin["frame_id"], pixels.shape, None),
                          dtype=bool, copy=True, order="C")
        pixels.setflags(write=False)
        usable.setflags(write=False)
        h, w = pixels.shape[:2]
        source = {"frame_id": pin["frame_id"], "encoded_sha256": sha(data),
                  "encoded_length": len(data), "decoded_sha256": roster.array_digest(pixels),
                  "shape": list(pixels.shape), "dtype": pixels.dtype.str,
                  "mask_sha256": roster.array_digest(usable), "usable_true": True,
                  "transform": roster.identity(w, h, pin["transform"]), "domain": roster.DOMAIN,
                  "raster_kind": "native", "decoder": pin["decoder"]}
        frame = roster.NativeFrame(pin["frame_id"], data, pixels, usable, source,
                                   transform=pin["transform"])
        roster._frame_source(frame)
        frames.append(frame)
    return tuple(frames)


def bind_groups(record, frames, *, services=None):
    _, _, _, roster, diagnostic = services or _services()
    bundle = record["source_region_masks"]
    diagnostic._validate_mask_diagnostic(record)
    masks = []
    for entry in bundle["entries"]:
        pixels = diagnostic.load_native_source_region_mask(bundle, record, entry["endpoint"])
        frame = next(f for f in frames if f.frame_id == entry["endpoint"][0])
        if (pixels.shape != frame.pixels.shape[:2]
                or roster.array_digest(pixels) != entry["full_native_bool_c_sha256"]):
            raise Stop("native_region_digest_mismatch")
        provenance = {"bundle_sha256": bundle["record_sha256"], "producer": bundle["producer"],
                      "endpoint": entry["endpoint"], "region_id": entry["mask"]["source_region_id"],
                      "header_packed_sha256": entry["mask"]["mask_sha256"],
                      "full_native_bool_c_sha256": entry["full_native_bool_c_sha256"],
                      "reading_binding_sha256": entry["reading_binding_sha256"],
                      "ingress_sha256": entry["ingress_sha256"], "row_refs": entry["row_refs"]}
        mask = roster.RegionMask(frame.frame_id, frame.source["decoded_sha256"], pixels,
                                 entry["full_native_bool_c_sha256"], canonical(provenance).decode())
        roster._region_record(mask, frame)
        masks.append(mask)
    if len(masks) != 3:
        raise Stop("exact_three_masks_required")
    return (roster.RegionGroup(GROUP_IDS[0], masks[0], masks[2]),
            roster.RegionGroup(GROUP_IDS[1], masks[1], masks[2]))


def _general_position(points):
    if len(set(points)) != 4:
        return False
    # Exact binary-coordinate arithmetic; no fitted transform, tolerance or score.
    for a, b, c in itertools.combinations(points, 3):
        ax, ay, bx, by, cx, cy = map(Fraction, (*a, *b, *c))
        if (bx - ax) * (cy - ay) == (by - ay) * (cx - ax):
            return False
    return True


def support_condition(group):
    pool = [p for p in group["alternatives"]
            if p["memberships"][group["group_id"]] == {"left": True, "right": True}]
    if len(pool) > 32:
        raise Stop("support_inventory_limit")
    for pairs in itertools.combinations(pool, 4):
        if (len({p["left_id"] for p in pairs}) == 4
                and len({p["right_id"] for p in pairs}) == 4
                and _general_position(tuple(tuple(p["left_xy"]) for p in pairs))
                and _general_position(tuple(tuple(p["right_xy"]) for p in pairs))):
            return {"eligible_alternatives": len(pool), "four_distinct_general_position_pairs_exist": True,
                    "meaning": "existence only; mutually exclusive alternatives remain unselected"}
    return {"eligible_alternatives": len(pool), "four_distinct_general_position_pairs_exist": False,
            "meaning": "inconclusive; no association or geometry conclusion"}


def validate_packet(packet):
    if packet.get("status") != "ready":
        raise Stop("helper_refused: " + str(packet.get("reason")))
    if (packet.get("schema") != "native-point-content-roster-v1"
            or len(packet.get("frames", [])) != 2
            or [g["group_id"] for g in packet.get("groups", [])] != list(GROUP_IDS)
            or packet.get("k") != 5 or packet.get("physical_correctness") != "unknown"
            or len(canonical(packet)) > 4 * 1024**2):
        raise Stop("incomplete_roster_inventory")
    counters = packet["counters"]
    groups = packet["groups"]
    count = sum(len(g["alternatives"]) for g in groups)
    queries = sum(len(g["queries"]) for g in groups)
    right_count = packet["frames"][1]["retained_count"]
    if (counters["comparisons_charged"] != queries * right_count
            or counters["comparisons_requested"] != counters["comparisons_charged"]
            or counters["comparisons_charged"] > 6144
            or any(counters[k] != 0 for k in ("comparisons_uncharged", "comparisons_unprocessed", "queries_unprocessed"))
            or counters["queries_selected"] != queries
            or counters["alternatives_counted"] != count or counters["alternatives_exposed"] != count
            or count > 96):
        raise Stop("roster_counter_closure")
    for group in groups:
        if len(group["queries"]) > 4 or len(group["alternatives"]) > 32:
            raise Stop("roster_group_limit")
        for query in group["queries"]:
            pairs = [p for p in group["alternatives"] if p["query_id"] == query["query_id"]]
            if (query["all_cutoff_ties"] is not True
                    or [p["pair_id"] for p in pairs] != query["retained_pair_ids"]
                    or query["candidate_domain_classes"] != len(packet["frames"][1]["classes"])
                    or query["omitted_above_cutoff"] + len(pairs) != query["candidate_domain_classes"]):
                raise Stop("roster_query_closure")


def seal_acquisition(record, bindings, packet):
    validate_packet(packet)
    conditions = {g["group_id"]: support_condition(g) for g in packet["groups"]}
    complete = all(c["four_distinct_general_position_pairs_exist"] for c in conditions.values())
    body = {"schema": SCHEMA, "status": "ready" if complete else "inconclusive",
            "sources": bindings, "mask_bundle_sha256": record["source_region_masks"]["record_sha256"],
            "native_ingress": record["ingress"], "usable_mask_producer": "correspondence.usable_mask_for(frame, shape, None):~create_gui_mask",
            "roster": packet, "structural_support": conditions,
            "limitations": "No physical correspondence selection, projective fit, camera, checker, scale or accuracy conclusion."}
    body["record_sha256"] = sha(canonical(body))
    return body


def validate_acquisition(record):
    if record.get("schema") != SCHEMA or record.get("status") not in ("ready", "inconclusive"):
        raise Stop("invalid_acquisition")
    body = dict(record)
    checksum = body.pop("record_sha256", None)
    if checksum != sha(canonical(body)):
        raise Stop("acquisition_record_seal")
    validate_packet(record["roster"])
    expected = {g["group_id"]: support_condition(g) for g in record["roster"]["groups"]}
    if record["structural_support"] != expected:
        raise Stop("support_seal")
    ready = all(c["four_distinct_general_position_pairs_exist"] for c in expected.values())
    if record["status"] != ("ready" if ready else "inconclusive"):
        raise Stop("support_status")


def _readback(path, encoded, validator):
    actual = read_bounded(path, ARTIFACT_BYTES)
    if actual != encoded:
        raise Stop("publication_readback")
    record = strict_json(actual)
    if canonical(record) + b"\n" != actual:
        raise Stop("publication_canonical")
    validator(record)


def publish(path, record, validator, *, remove_failed_completion=False):
    """Exclusive staged write/fsync/readback, no-clobber link and final readback."""
    encoded = canonical(record) + b"\n"
    # Conservatively count both names during linking plus 64 KiB receipt/log.
    if 2 * len(encoded) + 65536 > ARTIFACT_BYTES:
        raise Stop("artifact_byte_limit")
    temporary = path.with_name("." + path.name + ".partial")
    created = False
    linked = False
    try:
        with temporary.open("xb") as stream:
            created = True
            if stream.write(encoded) != len(encoded):
                raise Stop("short_write")
            stream.flush()
            os.fsync(stream.fileno())
        _readback(temporary, encoded, validator)
        os.link(temporary, path)
        linked = True
        _readback(path, encoded, validator)
    except BaseException:
        if remove_failed_completion and linked:
            # Only a link created by this invocation; never remove a collision.
            path.unlink()
        raise
    finally:
        if created:
            temporary.unlink()
    return {"path": path.name, "sha256": sha(encoded), "bytes": len(encoded)}


def _worker():
    record, bindings = authorize()
    services = _services()
    services[4]._validate_mask_diagnostic(record)  # Before encoded screenshot reads.
    # Both external input pins are checked before the first and only decode/frame.
    buffers = tuple(pinned_bytes(ROOT / name, pin, 8 * 1024**2) for name, pin in INPUTS)
    frames = bind_frames(buffers, record["ingress"], services=services)
    groups = bind_groups(record, frames, services=services)
    packet = services[3].acquire_roster(frames, groups)  # Exactly once; no tuning.
    acquisition = seal_acquisition(record, bindings, packet)
    # Source substitutions during work invalidate the entire invocation.
    after, after_bindings = authorize()
    if canonical(record) != canonical(after) or bindings != after_bindings:
        raise Stop("postflight_source_mismatch")
    for name, pin in INPUTS:
        pinned_bytes(ROOT / name, pin, 8 * 1024**2)
    publish(ROOT / OUTPUT / "roster.json", acquisition, validate_acquisition)


class _WindowsJob:
    """Kernel committed-memory cap and kill-on-close for the whole child tree."""
    def __init__(self):
        if os.name != "nt":
            raise Stop("windows_budget_supervisor_required")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll")
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                        ("peak_job_memory", ctypes.c_size_t)]
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("peak_ws", "ws", "peak_pool_paged", "pool_paged", "peak_pool_nonpaged", "pool_nonpaged", "pagefile", "peak_pagefile")]
        self.Extended, self.Memory = Extended, Memory
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000 | 0x200  # kill on close + aggregate committed memory.
        limits.job_memory = RSS_BYTES
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach_and_resume(self, process):
        handle = wintypes.HANDLE(int(process._handle))
        if not self.kernel.AssignProcessToJobObject(self.handle, handle):
            process.kill()
            raise ctypes.WinError(ctypes.get_last_error())
        if self.ntdll.NtResumeProcess(handle) != 0:
            process.kill()
            raise Stop("resume_failed")

    def snapshot(self):
        # Fixed array is ample for this two-process venv redirector/worker tree;
        # overflow refuses rather than omitting memory from an active child.
        class Processes(ctypes.Structure):
            _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                        ("ids", ctypes.c_size_t * 32)]
        ids = Processes()
        if not self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if ids.assigned > 32 or ids.count > 32:
            raise Stop("process_tree_limit")
        rss = 0
        for pid in {os.getpid(), *list(ids.ids)[:ids.count]}:
            handle = self.kernel.OpenProcess(0x410, False, pid)
            if not handle:
                # A child may exit between snapshot and OpenProcess.
                if pid == os.getpid():
                    raise Stop("supervisor_memory_query_failed")
                continue
            try:
                memory = self.Memory()
                memory.cb = ctypes.sizeof(memory)
                if not self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    raise ctypes.WinError(ctypes.get_last_error())
                rss += memory.peak_ws
            finally:
                self.kernel.CloseHandle(handle)
        limits = self.Extended()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return rss, limits.peak_job_memory

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def artifact_size(directory):
    # Count BOTH hardlink names conservatively while publishing.
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def budget_check(elapsed, rss, size):
    if elapsed > WALL_SECONDS:
        raise Stop("wall_limit")
    if rss > RSS_BYTES:
        raise Stop("rss_limit")
    if size > ARTIFACT_BYTES:
        raise Stop("artifact_byte_limit")


def supervise():
    started = time.monotonic()
    _, bindings = authorize()  # Missing review/code pin fails before output or pixels.
    directory = ROOT / OUTPUT
    if directory.exists():
        raise Stop("output_already_exists")
    job = _WindowsJob()
    process = None
    peak_rss = peak_commit = peak_output = 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "worker.log").open("xb") as log:
            command = [sys.executable, "-B", "-c",
                       "from scripts.evaluation.acquire_endpoint_conflict_native_roster import _worker; _worker()"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                       creationflags=0x4, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            job.attach_and_resume(process)
            while True:
                rss, commit = job.snapshot()
                size = artifact_size(directory)
                peak_rss, peak_commit, peak_output = max(peak_rss, rss), max(peak_commit, commit), max(peak_output, size)
                budget_check(time.monotonic() - started, rss, size)
                if process.poll() is not None:
                    break
                time.sleep(0.01)
        if process.returncode != 0:
            raise Stop("worker_failure: " + str(process.returncode))
        data = read_bounded(directory / "roster.json", ARTIFACT_BYTES)
        record = strict_json(data)
        _readback(directory / "roster.json", data, validate_acquisition)
        _, after = authorize()
        if bindings != after or record["sources"] != bindings:
            raise Stop("supervisor_source_mismatch")
        for name, pin in INPUTS:
            pinned_bytes(ROOT / name, pin, 8 * 1024**2)
        # Completion covers worker exit, final seal, postflight and supervisor
        # bounds. It is never produced on a worker/refusal/budget failure.
        receipt = {"schema": "endpoint-conflict-native-roster-completion-v1",
                   "status": record["status"], "roster_sha256": sha(data), "roster_bytes": len(data),
                   "sources": bindings, "wall_seconds_before_completion": time.monotonic() - started,
                   "peak_sampled_aggregate_process_highwater_rss_bytes": peak_rss,
                   "peak_job_committed_memory_bytes": peak_commit, "peak_artifact_bytes": peak_output,
                   "budget": {"wall_seconds": WALL_SECONDS, "rss_bytes": RSS_BYTES, "artifact_bytes": ARTIFACT_BYTES},
                   "limits_note": "10 ms samples plus kernel child-tree committed-memory limit; no unsampled RSS guarantee or power-loss guarantee"}
        receipt["record_sha256"] = sha(canonical(receipt))
        budget_check(time.monotonic() - started, job.snapshot()[0],
                     artifact_size(directory) + 2 * len(canonical(receipt)) + 2)
        def validate_completion(value):
            if value != receipt:
                raise Stop("completion_readback")
            budget_check(time.monotonic() - started, job.snapshot()[0], artifact_size(directory))
        publish(directory / "completion.json", receipt, validate_completion,
                remove_failed_completion=True)
        print(record["status"] + ": " + str(directory / "completion.json"))
        return 0 if record["status"] == "ready" else 2
    finally:
        job.close()
        if process is not None and process.poll() is None:
            process.wait(timeout=5)


def main():
    if len(sys.argv) != 1:
        raise Stop("no_acquisition_overrides_permitted")
    return supervise()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Stop, OSError, ValueError, KeyError, TypeError) as error:
        print("TERMINAL: " + str(error), file=sys.stderr)
        raise SystemExit(1)
