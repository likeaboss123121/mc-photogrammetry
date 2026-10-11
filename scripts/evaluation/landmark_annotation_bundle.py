"""Create blinded, headless annotation bundles and merge independent observations.

This evaluator-only utility does not inspect candidate cameras, manifests, mod
actuals, projections or residuals. It copies one externally bound PNG unchanged
into two annotator workspaces and provides only that raster and the frozen 3D
landmark identities/cues. The hashes and blinding fields are audit declarations;
they cannot prove that an annotator remained blind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_qualification as qa  # noqa: E402

INPUT_SCHEMA = "camera-landmark-annotation-input-v1"
MERGE_SCHEMA = "camera-landmark-double-annotation-v1"
MAX_PNG_BYTES = 128 * 1024 * 1024
MAX_BINDING_BYTES = 16 * 1024
MAX_PLAN_BYTES = 64 * 1024
MAX_OBSERVATION_BYTES = 64 * 1024
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}


class BundleError(ValueError):
    """Invalid, stale, mismatched, or unsafe annotation evidence."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_png(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Read bounded bytes and PNG IHDR only; never decode/open the raster."""
    try:
        if path.is_symlink() or not path.is_file():
            raise BundleError("png_not_regular_file")
        size = path.stat().st_size
        if size < 33 or size > MAX_PNG_BYTES:
            raise BundleError("png_byte_cap")
        raw = path.read_bytes()
    except OSError as error:
        raise BundleError("png_unreadable") from error
    if len(raw) != size or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise BundleError("invalid_png")
    length = struct.unpack(">I", raw[8:12])[0]
    if length != 13 or raw[12:16] != b"IHDR":
        raise BundleError("invalid_png_ihdr")
    if zlib.crc32(raw[12:29]) & 0xffffffff != struct.unpack(">I", raw[29:33])[0]:
        raise BundleError("invalid_png_ihdr_crc")
    width, height = struct.unpack(">II", raw[16:24])
    if (width <= 0 or height <= 0 or width > qa.MAX_DIMENSION
            or height > qa.MAX_DIMENSION or width * height > qa.MAX_IMAGE_PIXELS):
        raise BundleError("image_dimension_cap")
    return raw, (width, height)


def _read_bounded(path: Path, cap: int, label: str) -> bytes:
    """Read a regular non-link file with a hard cap enforced before allocation."""
    try:
        if path.is_symlink():
            raise BundleError(label + "_symlink")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(label + "_not_regular_file")
        if info.st_size > cap:
            raise BundleError(label + "_byte_cap")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != info.st_size:
                raise BundleError(label + "_changed_during_read")
            raw = stream.read(cap + 1)
        if len(raw) != info.st_size or len(raw) > cap:
            raise BundleError(label + "_changed_during_read")
        return raw
    except BundleError:
        raise
    except OSError as error:
        raise BundleError(label + "_unreadable") from error


def _load_binding_json(path: Path) -> qa.FrameBinding:
    raw = _read_bounded(path, MAX_BINDING_BYTES, "binding")
    try:
        doc = json.loads(raw.decode("utf-8"))
        binding = qa.FrameBinding(doc["frame_id"], doc["png_sha256"],
                                  doc["dimensions"][0], doc["dimensions"][1],
                                  doc["capture_id"], doc["world_sha256"],
                                  doc["model_sha256"], doc["context_sha256"])
        qa._binding(binding)
        return binding
    except (UnicodeError, ValueError, TypeError, KeyError, IndexError) as error:
        if isinstance(error, qa.QualificationError):
            raise BundleError("binding_" + error.code) from error
        raise BundleError("invalid_binding_json") from error


def _load_plan(plan_raw: bytes, plan_sha256: str) -> qa.LandmarkPlan:
    try:
        return qa.load_landmark_plan(plan_raw, expected_sha256=plan_sha256)
    except qa.QualificationError as error:
        raise BundleError("landmark_plan_" + error.code) from error


def _binding_record(binding: qa.FrameBinding) -> dict:
    return {
        "frame_id": binding.frame_id, "png_sha256": binding.png_sha256,
        "dimensions": [binding.width, binding.height], "capture_id": binding.capture_id,
        "world_sha256": binding.world_sha256, "model_sha256": binding.model_sha256,
        "context_sha256": binding.context_sha256,
    }


def create_annotation_inputs(*, png_path: Path | str, binding: qa.FrameBinding,
                             plan_raw: bytes, plan_sha256: str,
                             annotators: tuple[tuple[str, str], tuple[str, str]],
                             output_root: Path | str) -> dict:
    """Create two identical-raster, camera-free input directories, refusing overwrite."""
    plan = _load_plan(plan_raw, plan_sha256)
    try:
        qa._binding(binding)
    except qa.QualificationError as error:
        raise BundleError("invalid_binding_" + error.code) from error
    if (plan.world_sha256, plan.model_sha256, plan.context_sha256) != (
            binding.world_sha256, binding.model_sha256, binding.context_sha256):
        raise BundleError("plan_context_mismatch")
    if len(annotators) != 2:
        raise BundleError("annotator_count")
    checked = []
    normalized_ids = set()
    literal_ids = set()
    for annotator_id, tool_version in annotators:
        if not isinstance(annotator_id, str) or not _ID.fullmatch(annotator_id):
            raise BundleError("invalid_annotator_id")
        if annotator_id in literal_ids:
            raise BundleError("duplicate_annotator_identity")
        literal_ids.add(annotator_id)
        normalized = annotator_id.rstrip(" .").casefold()
        device_stem = normalized.split(".", 1)[0]
        if (not normalized or device_stem in _WINDOWS_RESERVED
                or normalized in normalized_ids):
            raise BundleError("colliding_or_reserved_annotator_id")
        normalized_ids.add(normalized)
        if (not isinstance(tool_version, str) or not tool_version
                or len(tool_version) > 128 or any(ord(char) < 32 for char in tool_version)):
            raise BundleError("invalid_tool_version")
        checked.append((annotator_id, tool_version))
    if checked[0][0] == checked[1][0]:
        raise BundleError("duplicate_annotator_identity")
    png, dimensions = _read_png(Path(png_path))
    if (_sha(png) != binding.png_sha256 or dimensions != (binding.width, binding.height)):
        raise BundleError("png_binding_mismatch")
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise BundleError("output_exists")
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".landmark-bundle-", dir=root.parent))
    except OSError as error:
        raise BundleError("bundle_filesystem_refusal") from error
    try:
        landmark_rows = [{
            "landmark_id": item.landmark_id,
            "world_xyz": list(item.world_xyz),
            "surface_id": item.surface_id,
            "structural_identity": item.structural_identity,
            "identity_cue": item.identity_cue,
            "support_class": item.support_class,
        } for item in plan.landmarks]
        for annotator_id, tool_version in checked:
            directory = temp / annotator_id
            directory.mkdir()
            (directory / "frame.png").write_bytes(png)
            packet_template = {
                "schema": qa.PACKET_SCHEMA, "evaluation_only": True,
                "method": None, "observer_id": annotator_id,
                "tool_version": tool_version, "protocol": None,
                "pixel_convention": None,
                "projected_coordinates_exposed": None, "frozen_before_projection": None,
                "landmark_plan_sha256": plan_sha256, "policy_sha256": qa.POLICY_SHA256,
                "frame_id": binding.frame_id, "png_sha256": binding.png_sha256,
                "dimensions": [binding.width, binding.height], "capture_id": binding.capture_id,
                "world_sha256": binding.world_sha256, "model_sha256": binding.model_sha256,
                "context_sha256": binding.context_sha256,
                "observations": [{"landmark_id": row["landmark_id"], "disposition": None}
                                 for row in landmark_rows],
            }
            manifest = {
                "schema": INPUT_SCHEMA, "evaluation_only": True,
                "annotator_id": annotator_id, "tool_version": tool_version,
                "image_file": "frame.png", "frame_binding": _binding_record(binding),
                "landmark_plan_sha256": plan_sha256,
                "policy_sha256": qa.POLICY_SHA256,
                "protocol": qa.PROTOCOL, "pixel_convention": qa.PIXEL_CONVENTION,
                "projected_coordinates_exposed": False,
                "candidate_camera_exposed": False,
                "mod_actual_exposed": False,
                "attestation_instructions": {
                    "method": "Annotator must explicitly confirm manual PNG observation.",
                    "protocol": "Annotator must explicitly confirm the declared protocol.",
                    "pixel_convention": "Annotator must explicitly confirm the pixel convention.",
                    "projected_coordinates_exposed": "Annotator must author this value as false.",
                    "frozen_before_projection": "Annotator must author this value as true.",
                    "limits": "These are caller-authored claims, not machine-proven facts.",
                },
                "landmarks": landmark_rows,
                "packet_template": packet_template,
            }
            (directory / "annotation-input.json").write_bytes(_canonical(manifest))
        os.replace(temp, root)
    except OSError as error:
        shutil.rmtree(temp, ignore_errors=True)
        raise BundleError("bundle_filesystem_refusal") from error
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {
        "schema": INPUT_SCHEMA, "output_root": str(root),
        "frame_binding": _binding_record(binding),
        "landmark_plan_sha256": plan_sha256,
        "annotators": [{"annotator_id": who, "tool_version": tool}
                       for who, tool in checked],
        "input_sha256": {
            who: _sha((root / who / "annotation-input.json").read_bytes())
            for who, _tool in checked
        },
    }


def merge_observations(*, plan_raw: bytes, plan_sha256: str,
                       first_raw: bytes, first_sha256: str,
                       second_raw: bytes, second_sha256: str,
                       binding: qa.FrameBinding,
                       expected_annotators: tuple[str, str]) -> tuple[bytes, bytes]:
    """Validate two independent packet files; produce deterministic merged packet and audit."""
    if len(expected_annotators) != 2 or expected_annotators[0] == expected_annotators[1]:
        raise BundleError("duplicate_annotator_identity")
    plan = _load_plan(plan_raw, plan_sha256)
    try:
        first = qa.seal_observations(first_raw, plan, expected_packet_sha256=first_sha256,
                                     expected_binding=binding)
        second = qa.seal_observations(second_raw, plan, expected_packet_sha256=second_sha256,
                                      expected_binding=binding)
    except qa.QualificationError as error:
        raise BundleError("observation_packet_" + error.code) from error
    if (first.observer_id, second.observer_id) != expected_annotators:
        raise BundleError("annotator_assignment_mismatch")
    if first.binding != second.binding or first.plan_sha256 != second.plan_sha256:
        raise BundleError("annotation_binding_mismatch")
    merged_rows = []
    for a, b in zip(first.observations, second.observations):
        if a.landmark_id != b.landmark_id:
            raise BundleError("observation_membership")
        if a.disposition == b.disposition == "observed":
            merged_rows.append({
                "landmark_id": a.landmark_id, "disposition": "observed", "uv": list(a.uv),
                "uncertainty_pixels": a.uncertainty_pixels,
                "uncertainty_method": "manual_localization_bound",
                "support_segments": [[list(point) for point in segment]
                                     for segment in a.support_segments],
                "second_annotation": {"observer_id": second.observer_id,
                                      "uv": list(b.uv),
                                      "uncertainty_pixels": b.uncertainty_pixels},
            })
        else:
            disposition = (a.disposition if a.disposition == b.disposition
                           and a.disposition != "observed" else "ambiguous")
            reason = (f"independent annotations: {a.disposition}; {b.disposition}")
            merged_rows.append({"landmark_id": a.landmark_id,
                                "disposition": disposition, "reason": reason})
    first_doc = json.loads(first_raw)
    merged_doc = dict(first_doc)
    merged_doc["observations"] = merged_rows
    merged_raw = _canonical(merged_doc)
    if len(merged_raw) > qa.MAX_PACKET_BYTES:
        raise BundleError("merged_packet_byte_cap")
    try:
        qa.seal_observations(merged_raw, plan, expected_packet_sha256=_sha(merged_raw),
                             expected_binding=binding)
    except qa.QualificationError as error:
        raise BundleError("merged_packet_" + error.code) from error
    audit = {
        "schema": MERGE_SCHEMA, "evaluation_only": True,
        "frame_binding": _binding_record(binding), "landmark_plan_sha256": plan_sha256,
        "policy_sha256": qa.POLICY_SHA256,
        "annotations": [
            {"annotator_id": first.observer_id, "tool_version": first.tool_version,
             "packet_sha256": first_sha256, "method": "manual_png_observation",
             "projected_coordinates_exposed": False, "frozen_before_projection": True},
            {"annotator_id": second.observer_id, "tool_version": second.tool_version,
             "packet_sha256": second_sha256, "method": "manual_png_observation",
             "projected_coordinates_exposed": False, "frozen_before_projection": True},
        ],
        "merged_packet_sha256": _sha(merged_raw),
        "projected_coordinates_exposed": False,
        "candidate_camera_exposed": False, "mod_actual_exposed": False,
    }
    return merged_raw, _canonical(audit)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-inputs")
    create.add_argument("--png", type=Path, required=True)
    create.add_argument("--binding-json", type=Path, required=True)
    create.add_argument("--plan", type=Path, required=True)
    create.add_argument("--plan-sha256", required=True)
    create.add_argument("--annotator", nargs=2, action="append", metavar=("ID", "TOOL"), required=True)
    create.add_argument("--out", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--binding-json", type=Path, required=True)
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--plan-sha256", required=True)
    merge.add_argument("--first", type=Path, required=True)
    merge.add_argument("--first-sha256", required=True)
    merge.add_argument("--second", type=Path, required=True)
    merge.add_argument("--second-sha256", required=True)
    merge.add_argument("--annotator", nargs=2, required=True, metavar=("FIRST_ID", "SECOND_ID"))
    merge.add_argument("--out", type=Path, required=True,
                       help="new directory for merged packet and provenance receipt")
    args = parser.parse_args(argv)
    try:
        binding = _load_binding_json(args.binding_json)
    except BundleError as error:
        raise SystemExit("refused: " + str(error)) from error
    try:
        if args.command == "create-inputs":
            result = create_annotation_inputs(
                png_path=args.png, binding=binding,
                plan_raw=_read_bounded(args.plan, MAX_PLAN_BYTES, "plan"),
                plan_sha256=args.plan_sha256,
                annotators=tuple(tuple(item) for item in args.annotator), output_root=args.out)
            print(json.dumps(result, sort_keys=True, indent=2))
        elif args.command == "merge":
            merged, audit = merge_observations(
                plan_raw=_read_bounded(args.plan, MAX_PLAN_BYTES, "plan"),
                plan_sha256=args.plan_sha256,
                first_raw=_read_bounded(args.first, MAX_OBSERVATION_BYTES, "first_observation"),
                first_sha256=args.first_sha256,
                second_raw=_read_bounded(args.second, MAX_OBSERVATION_BYTES, "second_observation"),
                second_sha256=args.second_sha256,
                binding=binding, expected_annotators=tuple(args.annotator))
            root = args.out
            if root.exists() or root.is_symlink():
                raise SystemExit("output directory exists; refusing overwrite")
            try:
                root.parent.mkdir(parents=True, exist_ok=True)
                temp = Path(tempfile.mkdtemp(prefix=".landmark-merge-", dir=root.parent))
            except OSError as error:
                raise SystemExit("refused: merge_filesystem_refusal") from error
            try:
                (temp / "observations.json").write_bytes(merged)
                (temp / "merge-receipt.json").write_bytes(audit)
                os.replace(temp, root)
            except OSError as error:
                shutil.rmtree(temp, ignore_errors=True)
                raise SystemExit("refused: merge_filesystem_refusal") from error
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                raise
            print(json.dumps({"output_root": str(root),
                              "merged_sha256": _sha(merged),
                              "receipt_sha256": _sha(audit)}, sort_keys=True))
    except BundleError as error:
        raise SystemExit("refused: " + str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
