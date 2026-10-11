"""Bounded v1 capture runtime/render provenance sidecar contract.

This is factory/evaluation metadata.  It is deliberately separate from normal
image-derived reconstruction inputs.  Unknown runtime observations stay typed
unknown; requested settings are never promoted to observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any, Mapping
import zipfile

SIDECAR_SCHEMA_VERSION = 1
MAX_SIDECAR_BYTES = 1_048_576
MAX_CANONICAL_DEPTH = 64
MAX_CANONICAL_OBJECT_KEYS = 8192
MAX_PACK_ENTRIES = 8192
MAX_PACK_FILE_BYTES = 64 * 1024 * 1024
MAX_PACK_TOTAL_BYTES = 512 * 1024 * 1024
MAX_PACK_MANIFEST_BYTES = 8 *  1024 * 1024
MAX_PACK_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 1024
MAX_PACK_DEPTH = 64
MAX_PACK_SCAN_SECONDS = 15.0
# Release gate: the capture worker still cannot bind shader/program state and
# framebuffer readback to a stable resource epoch, so strict certification must
# remain unavailable even for a self-consistent synthetic receipt.
STRICT_RENDER_CERTIFICATION_ENABLED = False
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(ValueError):
    """Malformed, incomplete, or over-limit provenance evidence."""


def _canonical_bytes(value: Any, *, max_bytes: int, label: str) -> bytes:
    try:
        _measure_canonical_bytes(value, max_bytes=max_bytes)
        _validate_canonical_domain(value)
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
        text = _escape_gson_unicode(text)
        encoded = text.encode("utf-8")
    except ProvenanceError as exc:
        if str(exc) == "canonical JSON exceeds byte cap":
            raise ProvenanceError(f"{label} exceeds byte cap") from exc
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProvenanceError("provenance is not canonical JSON data") from exc
    if len(encoded) > max_bytes:
        raise ProvenanceError(f"{label} exceeds byte cap")
    return encoded


def _measure_canonical_bytes(value: Any, *, max_bytes: int) -> int:
    """Bound canonical output before json.dumps allocates the serialized text."""
    total = 0

    def add(amount: int) -> None:
        nonlocal total
        total += amount
        if total > max_bytes:
            raise ProvenanceError("canonical JSON exceeds byte cap")

    def string_size(text: str) -> int:
        size = 2
        index = 0
        while index < len(text):
            point = ord(text[index])
            if 0xD800 <= point <= 0xDFFF:
                raise ProvenanceError("canonical JSON contains an unpaired surrogate")
            if text[index] in ('"', "\\", "\b", "\f", "\n", "\r", "\t"):
                size += 2
            elif point < 0x20 or point in (0x2028, 0x2029):
                size += 6
            elif point > 0xFFFF:
                size += 12
            elif point < 0x80:
                size += 1
            elif point < 0x800:
                size += 2
            else:
                size += 3
            if size > max_bytes:
                raise ProvenanceError("canonical JSON exceeds byte cap")
            index += 1
        return size

    def visit(item: Any, depth: int) -> None:
        if depth > MAX_CANONICAL_DEPTH:
            raise ProvenanceError("canonical JSON nesting exceeds depth cap")
        if item is None:
            add(4)
        elif isinstance(item, bool):
            add(4 if item else 5)
        elif isinstance(item, int):
            if not -(2**63) <= item <= 2**63 - 1:
                raise ProvenanceError("canonical JSON integer exceeds signed 64-bit range")
            add(len(str(item)))
        elif isinstance(item, float):
            raise ProvenanceError("floating-point values are outside the canonical JSON contract")
        elif isinstance(item, str):
            add(string_size(item))
        elif isinstance(item, list):
            if len(item) > max_bytes:
                raise ProvenanceError("canonical JSON exceeds byte cap")
            add(2 + max(0, len(item) - 1))
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict):
            if len(item) > MAX_CANONICAL_OBJECT_KEYS or \
                    not all(isinstance(key, str) for key in item):
                raise ProvenanceError("canonical JSON object key count/type is invalid")
            add(2 + max(0, len(item) - 1))
            for key, child in item.items():
                add(string_size(key) + 1)
                visit(child, depth + 1)
        else:
            raise ProvenanceError("value is outside the canonical JSON contract")

    visit(value, 0)
    return total


def _validate_canonical_domain(value: Any) -> None:
    """Restrict wire numbers to integers; Python/Gson float spellings differ."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            raise ProvenanceError("canonical JSON integer exceeds signed 64-bit range")
        return
    if isinstance(value, float):
        raise ProvenanceError("floating-point values are outside the canonical JSON contract")
    if isinstance(value, list):
        for item in value:
            _validate_canonical_domain(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ProvenanceError("canonical JSON object keys must be strings")
        for item in value.values():
            _validate_canonical_domain(item)
        return
    raise ProvenanceError("value is outside the canonical JSON contract")


def _escape_gson_unicode(text: str) -> str:
    """Match Gson JsonWriter: retain BMP text except U+2028/2029, escape astral pairs."""
    output: list[str] = []
    for char in text:
        code = ord(char)
        if code in (0x2028, 0x2029):
            output.append(f"\\u{code:04x}")
        elif code > 0xFFFF:
            adjusted = code - 0x10000
            high = 0xD800 + (adjusted >> 10)
            low = 0xDC00 + (adjusted & 0x3FF)
            output.append(f"\\u{high:04x}\\u{low:04x}")
        else:
            output.append(char)
    return "".join(output)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value, max_bytes=MAX_SIDECAR_BYTES,
                            label="provenance sidecar")


def canonical_manifest_json(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value, max_bytes=MAX_PACK_MANIFEST_BYTES,
                            label="resource-pack manifest")


def seal_sidecar(document: Mapping[str, Any]) -> dict[str, Any]:
    if len(document) > MAX_CANONICAL_OBJECT_KEYS:
        raise ProvenanceError("provenance root exceeds canonical key cap")
    result = dict(document)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    # Bound the serialized form including the seal.
    canonical_json(result)
    return result


def verify_seal(document: Mapping[str, Any]) -> str:
    if not isinstance(document, Mapping):
        raise ProvenanceError("sidecar must be a JSON object")
    if len(document) > MAX_CANONICAL_OBJECT_KEYS:
        raise ProvenanceError("provenance root exceeds canonical key cap")
    value = document.get("seal_sha256")
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProvenanceError("sidecar seal is missing or malformed")
    unsigned = dict(document)
    unsigned.pop("seal_sha256", None)
    expected = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if value != expected:
        raise ProvenanceError("sidecar seal does not match canonical contents")
    return expected


def sha256_file_bounded(path: Path, *, max_bytes: int = MAX_PACK_FILE_BYTES,
                        deadline: float | None = None) -> tuple[str, int]:
    path = Path(path)
    if _linklike(path):
        raise ProvenanceError(f"symlink is not allowed in provenance input: {path}")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > max_bytes:
        raise ProvenanceError(f"provenance file is not regular or exceeds byte cap: {path}")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not _same_file_identity(info, opened):
            raise ProvenanceError(f"provenance file changed before hashing: {path}")
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise ProvenanceError("provenance file hashing timed out")
            chunk = stream.read(min(1024 * 1024, max_bytes + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > max_bytes:
                raise ProvenanceError(f"provenance file grew beyond byte cap: {path}")
            digest.update(chunk)
        closed = os.fstat(stream.fileno())
    after = path.lstat()
    if _linklike(path) or count != info.st_size or \
            not _same_file_identity(opened, closed) or \
            not _same_file_identity(info, after):
        raise ProvenanceError(f"provenance file changed while hashing: {path}")
    return digest.hexdigest(), count


def _relative_posix(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ProvenanceError("pack entry has an invalid relative path")
    raw_parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in raw_parts) or \
            ":" in raw_parts[0]:
        raise ProvenanceError("pack entry path is aliased, absolute, or escapes its root")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ProvenanceError("pack entry escapes its root")
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_RELATIVE_PATH_BYTES:
        raise ProvenanceError("pack entry path exceeds byte cap")
    return normalized


def _linklike(path: Path) -> bool:
    """Reject symlinks and Windows reparse points, including junctions."""
    try:
        info = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    attrs = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & reparse)


def directory_manifest(root: Path) -> dict[str, Any]:
    """Hash a bounded directory pack; reject links, aliases and path escapes."""
    root = Path(root)
    if _linklike(root) or not root.is_dir():
        raise ProvenanceError("resource-pack directory must be a real directory")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    visited = 0
    # Explicit stack avoids recursion depth and never follows symlinks.
    started = time.monotonic()
    root_info = root.lstat()
    pending = [(root, 0, root_info)]
    while pending:
        parent, depth, expected_parent = pending.pop()
        if _linklike(parent):
            raise ProvenanceError("resource-pack directory became a symlink or reparse point")
        parent_info = parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or \
                not _same_file_identity(expected_parent, parent_info):
            raise ProvenanceError("resource-pack directory changed during traversal")
        children = []
        try:
            with os.scandir(parent) as scan:
                for child in scan:
                    visited += 1
                    if visited > MAX_PACK_ENTRIES:
                        raise ProvenanceError("resource-pack total path count exceeds cap")
                    if time.monotonic() - started > MAX_PACK_SCAN_SECONDS:
                        raise ProvenanceError("resource-pack directory scan timed out")
                    children.append(child)
            after_scan = parent.lstat()
            if _linklike(parent) or not _same_file_identity(parent_info, after_scan):
                raise ProvenanceError("resource-pack directory changed during enumeration")
        except OSError as exc:
            raise ProvenanceError("cannot enumerate resource-pack directory") from exc
        children.sort(key=lambda item: item.name)
        for child in children:
            path = Path(child.path)
            rel = _relative_posix(path.relative_to(root).as_posix())
            folded = rel.casefold()
            if folded in seen:
                raise ProvenanceError("duplicate resource-pack path")
            seen.add(folded)
            if child.is_symlink() or _linklike(path):
                raise ProvenanceError("symlink is not allowed in resource-pack directory")
            if child.is_dir(follow_symlinks=False):
                if depth + 1 > MAX_PACK_DEPTH:
                    raise ProvenanceError("resource-pack directory depth exceeds cap")
                child_info = path.lstat()
                if _linklike(path) or not stat.S_ISDIR(child_info.st_mode):
                    raise ProvenanceError("resource-pack child is not a stable real directory")
                pending.append((path, depth + 1, child_info))
                continue
            if not child.is_file(follow_symlinks=False):
                raise ProvenanceError("non-regular resource-pack entry")
            if len(entries) >= MAX_PACK_ENTRIES:
                raise ProvenanceError("resource-pack entry count exceeds cap")
            digest, size = sha256_file_bounded(
                path, deadline=started + MAX_PACK_SCAN_SECONDS)
            total_bytes += size
            if total_bytes > MAX_PACK_TOTAL_BYTES:
                raise ProvenanceError("resource-pack total bytes exceed cap")
            entries.append({"path": rel, "bytes": size, "sha256": digest})
    entries.sort(key=lambda item: item["path"])
    manifest = {"kind": "directory", "entries": entries,
                "entry_count": len(entries), "total_bytes": total_bytes}
    encoded = canonical_manifest_json(manifest)
    manifest["sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns) == \
        (right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns)


def _read_bounded_snapshot(path: Path, *, max_bytes: int) -> bytes:
    path = Path(path)
    if _linklike(path):
        raise ProvenanceError("symlink is not allowed in provenance input")
    before_path = path.lstat()
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_size > max_bytes:
        raise ProvenanceError("provenance input exceeds byte cap or is not a regular file")
    deadline = time.monotonic() + MAX_PACK_SCAN_SECONDS
    try:
        with path.open("rb") as stream:
            before_fd = os.fstat(stream.fileno())
            if not _same_file_identity(before_path, before_fd):
                raise ProvenanceError("provenance input changed before read")
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                if time.monotonic() > deadline:
                    raise ProvenanceError("provenance input read timed out")
                chunk = stream.read(min(1024 * 1024, max_bytes + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            raw = bytes(chunks)
            after_fd = os.fstat(stream.fileno())
        after_path = path.lstat()
    except OSError as exc:
        raise ProvenanceError("cannot read provenance input") from exc
    if len(raw) > max_bytes:
        raise ProvenanceError("provenance input grew beyond byte cap")
    if len(raw) != before_path.st_size or not _same_file_identity(before_fd, after_fd) or \
            not _same_file_identity(before_path, after_path):
        raise ProvenanceError("provenance input changed while reading")
    return raw


def read_bounded_file_snapshot(path: Path, *, max_bytes: int = MAX_SIDECAR_BYTES) -> bytes:
    """Read one bounded stable regular-file snapshot, rejecting links/reparse points."""
    return _read_bounded_snapshot(Path(path), max_bytes=max_bytes)


def _zip_central_records(raw: bytes) -> int:
    search_from = max(0, len(raw) - 65_557)
    eocd = raw.rfind(b"PK\x05\x06", search_from)
    if eocd < 0 or eocd + 22 > len(raw):
        raise ProvenanceError("resource-pack ZIP directory is malformed")
    disk = int.from_bytes(raw[eocd + 4:eocd + 6], "little")
    central_disk = int.from_bytes(raw[eocd + 6:eocd + 8], "little")
    entries_on_disk = int.from_bytes(raw[eocd + 8:eocd + 10], "little")
    entries_total = int.from_bytes(raw[eocd + 10:eocd + 12], "little")
    central_size = int.from_bytes(raw[eocd + 12:eocd + 16], "little")
    central_offset = int.from_bytes(raw[eocd + 16:eocd + 20], "little")
    comment_size = int.from_bytes(raw[eocd + 20:eocd + 22], "little")
    if eocd + 22 + comment_size != len(raw) or disk != 0 or central_disk != 0:
        raise ProvenanceError("resource-pack ZIP EOCD/comment/disk fields are inconsistent")
    if entries_on_disk == 0xFFFF or entries_total == 0xFFFF or \
            central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise ProvenanceError("ZIP64 resource packs are not supported by this bounded contract")
    if entries_on_disk != entries_total or entries_total > MAX_PACK_ENTRIES:
        raise ProvenanceError("resource-pack central-record count exceeds cap or is inconsistent")
    central_end = central_offset + central_size
    if central_end != eocd or central_offset > len(raw):
        raise ProvenanceError("resource-pack ZIP central directory bounds are inconsistent")
    cursor = central_offset
    for _ in range(entries_total):
        if cursor + 46 > central_end or raw[cursor:cursor + 4] != b"PK\x01\x02":
            raise ProvenanceError("resource-pack ZIP central record is malformed")
        name_len = int.from_bytes(raw[cursor + 28:cursor + 30], "little")
        extra_len = int.from_bytes(raw[cursor + 30:cursor + 32], "little")
        comment_len = int.from_bytes(raw[cursor + 32:cursor + 34], "little")
        cursor += 46 + name_len + extra_len + comment_len
        if cursor > central_end:
            raise ProvenanceError("resource-pack ZIP central record exceeds directory bounds")
    if cursor != central_end:
        raise ProvenanceError("resource-pack ZIP central record count is inconsistent")
    return entries_total


def zip_manifest(path: Path) -> dict[str, Any]:
    """Hash bounded ZIP contents in canonical path order, rejecting aliases."""
    raw = _read_bounded_snapshot(path, max_bytes=MAX_PACK_ARCHIVE_BYTES)
    archive_sha = hashlib.sha256(raw).hexdigest()
    archive_size = len(raw)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    deadline = time.monotonic() + MAX_PACK_SCAN_SECONDS
    try:
        entry_count = _zip_central_records(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) != entry_count or len(infos) > MAX_PACK_ENTRIES:
                raise ProvenanceError("resource-pack parsed central-record count mismatch")
            infos.sort(key=lambda info: info.filename)
            for info in infos:
                if time.monotonic() > deadline:
                    raise ProvenanceError("resource-pack ZIP scan timed out")
                raw_name = info.filename[:-1] if info.is_dir() and info.filename.endswith("/") else info.filename
                rel = _relative_posix(raw_name)
                folded = rel.casefold()
                if folded in seen:
                    raise ProvenanceError("duplicate resource-pack path")
                seen.add(folded)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ProvenanceError("symlink entry is not allowed in resource-pack ZIP")
                if info.is_dir():
                    continue
                if info.file_size < 0 or info.file_size > MAX_PACK_FILE_BYTES:
                    raise ProvenanceError("resource-pack entry exceeds byte cap")
                total += info.file_size
                if total > MAX_PACK_TOTAL_BYTES:
                    raise ProvenanceError("resource-pack total bytes exceed cap")
                digest = hashlib.sha256()
                read = 0
                with archive.open(info, "r") as source:
                    while chunk := source.read(1024 * 1024):
                        if time.monotonic() > deadline:
                            raise ProvenanceError("resource-pack ZIP expansion timed out")
                        read += len(chunk)
                        if read > info.file_size or read > MAX_PACK_FILE_BYTES:
                            raise ProvenanceError("resource-pack entry expanded beyond cap")
                        digest.update(chunk)
                if read != info.file_size:
                    raise ProvenanceError("resource-pack entry size changed while reading")
                rows.append({"path": rel, "bytes": read, "sha256": digest.hexdigest()})
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ProvenanceError("cannot read resource-pack archive safely") from exc
    manifest = {"kind": "zip", "archive_sha256": archive_sha,
                "archive_bytes": archive_size, "entries": rows,
                "entry_count": len(rows), "total_bytes": total}
    encoded = canonical_manifest_json(manifest)
    manifest["sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def validate_sidecar(document: Mapping[str, Any], *, expected: Mapping[str, Any],
                     expected_frames: list[Mapping[str, Any]],
                     require_certifiable: bool = True) -> str:
    """Validate seal, attempt identity, unique frame joins and capture bindings.

    Unknown optional adapter values are preserved, but a strict new factory run
    cannot be finalized unless runtime/resource/render identities are certified
    and every frame has a capture-bound epoch.
    """
    verify_seal(document)
    if document.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ProvenanceError("unsupported runtime/render sidecar schema")
    for key in ("attempt_nonce", "plan_id", "plan_sha256", "dataset_plan_sha256",
                "staging_token", "scene_id", "world_identity", "mode",
                "expectation_sha256"):
        if key not in expected:
            continue
        if document.get(key) != expected.get(key):
            raise ProvenanceError(f"sidecar {key} does not match armed attempt")
    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ProvenanceError("sidecar runtime record is missing")
    for key in ("minecraft_version", "fabric_loader_version", "java_version",
                "capture_mod_sha256", "capture_mod_source_sha256",
                "capture_mod_staged_sha256", "capture_mod_loaded_sha256",
                "client_artifacts"):
        if key not in runtime:
            raise ProvenanceError(f"sidecar runtime lacks {key}")
    if runtime.get("capture_mod_source_sha256") != expected.get("capture_mod_source_sha256") or \
            runtime.get("capture_mod_staged_sha256") != expected.get("capture_mod_staged_sha256") or \
            runtime.get("capture_mod_sha256") != expected.get("capture_mod_staged_sha256"):
        raise ProvenanceError("capture mod source/staged digest differs from trusted runner expectation")
    if runtime.get("capture_mod_loaded_sha256") != expected.get("capture_mod_staged_sha256") and require_certifiable:
        raise ProvenanceError("loaded capture mod digest is missing or mismatched")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != len(expected_frames):
        raise ProvenanceError("sidecar frame count differs from plan")
    expected_by_id = {str(row["frame_id"]): row for row in expected_frames}
    if len(expected_by_id) != len(expected_frames):
        raise ProvenanceError("expected frame list contains duplicate ids")
    seen: set[str] = set()
    for row in frames:
        if not isinstance(row, Mapping):
            raise ProvenanceError("sidecar frame entry must be an object")
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or frame_id in seen or frame_id not in expected_by_id:
            raise ProvenanceError("sidecar frame ids are duplicated or unexpected")
        seen.add(frame_id)
        wanted = expected_by_id[frame_id]
        for key in ("filename", "screenshot_sha256", "screenshot_bytes"):
            if row.get(key) != wanted.get(key):
                raise ProvenanceError(f"sidecar frame {frame_id} {key} mismatch")
        if not isinstance(row.get("snapshot_sha256"), str) or not _SHA256.fullmatch(
                row["snapshot_sha256"]):
            raise ProvenanceError(f"sidecar frame {frame_id} snapshot digest is invalid")
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, Mapping) or snapshot.get("frame_id") != frame_id:
            raise ProvenanceError(f"sidecar frame {frame_id} snapshot identity is invalid")
        snapshot_digest = hashlib.sha256(canonical_json(snapshot)).hexdigest()
        if snapshot_digest != row["snapshot_sha256"]:
            raise ProvenanceError(f"sidecar frame {frame_id} snapshot digest mismatch")
        if row.get("capture_epoch_status") != snapshot.get("capture_epoch_status"):
            raise ProvenanceError(f"sidecar frame {frame_id} epoch status contradicts snapshot")
        epoch_status = row.get("capture_epoch_status")
        if require_certifiable and epoch_status != "bound":
            raise ProvenanceError(f"sidecar frame {frame_id} capture epoch is unbound")
        if not require_certifiable and epoch_status not in (
                "bound", "capture_epoch_unbound"):
            raise ProvenanceError(f"sidecar frame {frame_id} has an invalid report-only epoch status")
    if require_certifiable:
        certification = document.get("certification")
        if not isinstance(certification, Mapping) or certification.get("status") != "verified":
            reasons = certification.get("unavailable_reasons") if isinstance(certification, Mapping) else None
            detail = ", ".join(item for item in reasons if isinstance(item, str)) \
                if isinstance(reasons, list) else "strict adapter evidence is absent"
            raise ProvenanceError(f"strict render certification unavailable: {detail}")
        if runtime.get("identity_status") != "complete":
            raise ProvenanceError("runtime identity is not certifiable")
        artifacts = runtime.get("client_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ProvenanceError("client artifact inventory is empty")
        artifact_by_id: dict[str, Mapping[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("id"), str):
                raise ProvenanceError("client artifact inventory entry is malformed")
            artifact_id = artifact["id"]
            if artifact_id in artifact_by_id:
                raise ProvenanceError("client artifact inventory has duplicate ids")
            artifact_by_id[artifact_id] = artifact
            if not isinstance(artifact.get("origins"), list) or not artifact["origins"] or \
                    not all(isinstance(origin, str) and origin for origin in artifact["origins"]):
                raise ProvenanceError(f"client artifact {artifact_id} has no origin")
            if not isinstance(artifact.get("sha256"), str) or not _SHA256.fullmatch(artifact["sha256"]):
                raise ProvenanceError(f"client artifact {artifact_id} has no content digest")
            origins = artifact["origins"]
            if len(origins) != 1:
                raise ProvenanceError(f"client artifact {artifact_id} has an unbounded origin set")
            try:
                origin_digest, _origin_bytes = sha256_file_bounded(Path(origins[0]))
            except (OSError, ProvenanceError) as exc:
                raise ProvenanceError(f"client artifact {artifact_id} origin cannot be verified") from exc
            if origin_digest != artifact["sha256"]:
                raise ProvenanceError(f"client artifact {artifact_id} origin digest mismatch")
        if not {"minecraft", "fabricloader", "mc_photo_capture"}.issubset(artifact_by_id):
            raise ProvenanceError("client artifact inventory lacks Minecraft/loader/capture mod")
        capture_artifact = artifact_by_id["mc_photo_capture"]
        if capture_artifact.get("sha256") != runtime.get("capture_mod_loaded_sha256") or \
                capture_artifact.get("sha256") != expected.get("capture_mod_staged_sha256"):
            raise ProvenanceError("capture-mod artifact digest differs from loaded/trusted identity")
        if document.get("resource_status") != "complete":
            raise ProvenanceError("resource-pack provenance is not certifiable")
        packs = document.get("resource_packs")
        if not isinstance(packs, list) or not packs:
            raise ProvenanceError("resource-pack inventory is empty")
        for pack in packs:
            if not isinstance(pack, Mapping) or not isinstance(pack.get("id"), str) or \
                    not isinstance(pack.get("manifest"), Mapping):
                raise ProvenanceError("resource-pack entry is incomplete")
            manifest = pack["manifest"]
            entries = manifest.get("entries")
            if not isinstance(entries, list) or not entries or \
                    manifest.get("entry_count") != len(entries):
                raise ProvenanceError("resource-pack content manifest is empty or has a bad count")
            if not all(isinstance(entry, Mapping) and isinstance(entry.get("path"), str) and
                       isinstance(entry.get("bytes"), int) and entry["bytes"] >= 0 and
                       isinstance(entry.get("sha256"), str) and _SHA256.fullmatch(entry["sha256"])
                       for entry in entries):
                raise ProvenanceError("resource-pack manifest entry is malformed")
            unsigned_manifest = {key: value for key, value in manifest.items() if key != "sha256"}
            computed_manifest = hashlib.sha256(
                canonical_manifest_json(unsigned_manifest)).hexdigest()
            if manifest.get("sha256") != computed_manifest:
                raise ProvenanceError("resource-pack content manifest digest mismatch")
            origin = pack.get("origin")
            if not isinstance(origin, str) or not origin:
                raise ProvenanceError("resource-pack content origin is unavailable")
            try:
                if manifest.get("kind") == "directory":
                    actual_manifest = directory_manifest(Path(origin))
                elif manifest.get("kind") == "zip":
                    actual_manifest = zip_manifest(Path(origin))
                else:
                    raise ProvenanceError("resource-pack manifest kind is unsupported")
            except (OSError, ProvenanceError) as exc:
                raise ProvenanceError("resource-pack origin cannot be verified") from exc
            if actual_manifest != manifest:
                raise ProvenanceError("resource-pack origin differs from its content manifest")
        shader_status = document.get("shader_status")
        if shader_status not in ("vanilla_builtin", "external_active"):
            raise ProvenanceError("shader state is unknown")
        shader_identity = document.get("shader_identity")
        if not isinstance(shader_identity, Mapping) or shader_identity.get("status") != "verified":
            raise ProvenanceError("shader/program identity is unavailable")
        artifact_digests = {item["sha256"] for item in artifact_by_id.values()}
        if shader_status == "vanilla_builtin":
            shader_artifact = artifact_by_id.get(shader_identity.get("artifact_id"))
            if shader_artifact is None or shader_identity.get("sha256") != shader_artifact.get("sha256"):
                raise ProvenanceError("built-in shader identity is not bound to a client artifact")
        else:
            shader_origin = shader_identity.get("origin")
            shader_sha = shader_identity.get("sha256")
            if not isinstance(shader_origin, str) or not isinstance(shader_sha, str) or \
                    not _SHA256.fullmatch(shader_sha):
                raise ProvenanceError("external shader lacks a content identity")
            try:
                observed_shader_sha, _ = sha256_file_bounded(Path(shader_origin))
            except (OSError, ProvenanceError) as exc:
                raise ProvenanceError("external shader origin cannot be verified") from exc
            if observed_shader_sha != shader_sha:
                raise ProvenanceError("external shader origin digest mismatch")
        render_policy = document.get("render_policy")
        if not isinstance(render_policy, Mapping) or \
                render_policy.get("status") != "verified" or \
                not isinstance(render_policy.get("pipelines"), list) or \
                not render_policy["pipelines"]:
            raise ProvenanceError("effective render-policy evidence is unavailable")
        policy_unsigned = {key: value for key, value in render_policy.items() if key != "sha256"}
        policy_sha = hashlib.sha256(canonical_json(policy_unsigned)).hexdigest()
        if render_policy.get("sha256") != policy_sha:
            raise ProvenanceError("render-policy digest does not match pipeline evidence")
        manifest_digests = {pack["manifest"]["sha256"] for pack in packs}
        programs = document.get("program_epochs")
        if not isinstance(programs, list) or not programs:
            raise ProvenanceError("render-program epoch inventory is unavailable")
        program_by_id: dict[str, Mapping[str, Any]] = {}
        for program in programs:
            if not isinstance(program, Mapping) or not isinstance(program.get("id"), str) or \
                    not isinstance(program.get("sha256"), str) or \
                    not _SHA256.fullmatch(program["sha256"]) or \
                    program.get("shader_identity_sha256") != shader_identity.get("sha256") or \
                    program.get("render_policy_sha256") != policy_sha:
                raise ProvenanceError("render-program epoch entry is malformed")
            if program["id"] in program_by_id:
                raise ProvenanceError("render-program epoch id is duplicated")
            program_by_id[program["id"]] = program
        for pipeline in render_policy["pipelines"]:
            if not isinstance(pipeline, Mapping) or not isinstance(pipeline.get("id"), str) or \
                    pipeline.get("render_layer") not in ("opaque", "cutout", "translucent") or \
                    pipeline.get("artifact_sha256") not in artifact_digests or \
                    pipeline.get("resource_manifest_sha256") not in manifest_digests or \
                    pipeline.get("program_sha256") not in {
                        value["sha256"] for value in program_by_id.values()}:
                raise ProvenanceError("render pipeline is untyped or not bound to runtime evidence")
        pack_ids = [pack["id"] for pack in packs]
        pack_order_sha = hashlib.sha256(canonical_json(pack_ids)).hexdigest()
        combined_manifest_sha = hashlib.sha256(canonical_json(
            [pack["manifest"]["sha256"] for pack in packs])).hexdigest()
        if document.get("resource_pack_order_sha256") != pack_order_sha:
            raise ProvenanceError("resource-pack precedence digest is invalid")
        epochs = document.get("resource_epochs")
        if not isinstance(epochs, list) or not epochs:
            raise ProvenanceError("effective resource epoch inventory is unavailable")
        epoch_by_id: dict[str, Mapping[str, Any]] = {}
        for epoch in epochs:
            if not isinstance(epoch, Mapping) or not isinstance(epoch.get("id"), str) or \
                    epoch.get("pack_order_sha256") != pack_order_sha or \
                    epoch.get("manifest_sha256") != combined_manifest_sha:
                raise ProvenanceError("resource epoch does not bind ordered pack manifests")
            if epoch["id"] in epoch_by_id:
                raise ProvenanceError("resource epoch id is duplicated")
            epoch_by_id[epoch["id"]] = epoch
        if document.get("gl_status") != "complete":
            raise ProvenanceError("GL/render adapter is unavailable")
        for row in frames:
            snapshot = row["snapshot"]
            if snapshot.get("capture_epoch_status") != "bound" or \
                    snapshot.get("resource_epoch_status") != "bound" or \
                    snapshot.get("resource_order_stable") is not True or \
                    snapshot.get("ordered_resource_packs") != pack_ids or \
                    snapshot.get("resource_pack_order_sha256") != pack_order_sha or \
                    not isinstance(snapshot.get("gl"), Mapping) or \
                    not all(isinstance(snapshot["gl"].get(key), str) and snapshot["gl"].get(key)
                            for key in ("vendor", "renderer", "version", "glsl_version")):
                raise ProvenanceError("frame resource/render epoch is not bound")
            if snapshot.get("resource_epoch_id") not in epoch_by_id:
                raise ProvenanceError("frame resource epoch id is missing or unknown")
            program = program_by_id.get(snapshot.get("program_epoch_id"))
            if program is None or snapshot.get("program_sha256") != program.get("sha256"):
                raise ProvenanceError("frame render-program epoch is missing or unknown")
            acquisition = snapshot.get("acquisition_epoch")
            if not isinstance(acquisition, Mapping) or \
                    acquisition.get("status") != "readback_bound" or \
                    acquisition.get("resource_epoch_id") != snapshot.get("resource_epoch_id") or \
                    acquisition.get("program_epoch_id") != snapshot.get("program_epoch_id") or \
                    acquisition.get("screenshot_sha256") != row.get("screenshot_sha256") or \
                    acquisition.get("screenshot_bytes") != row.get("screenshot_bytes"):
                raise ProvenanceError("frame acquisition epoch is not bound to readback bytes")
        if not STRICT_RENDER_CERTIFICATION_ENABLED:
            raise ProvenanceError(
                "strict render certification unavailable: trusted shader/resource/readback adapters are not implemented")
    return str(document["seal_sha256"])

