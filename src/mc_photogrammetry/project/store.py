from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mc_photogrammetry.domain.models import (
    IMAGE_DERIVED_PRODUCTION_FIELD,
    IMAGE_DERIVED_SESSION_FIELD,
    SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
    ScanProject,
)
from mc_photogrammetry.pipeline.image_derived_production_record import (
    ProductionEnvelopeError,
    canonicalise_production_record,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    ImageDerivedSessionError,
    image_derived_session_from_record,
    image_derived_session_to_record,
)


class ProjectStoreError(ValueError):
    """Raised when a scan project cannot be read or written."""


def _reject_production_schema(project: ScanProject, destination: Path) -> None:
    """Refuse a schema-3 payload through the replace-based writer.

    Schema-3 is the durable production terminal.  Its integrity depends on the
    no-clobber publication path, so routing it through :func:`save_project` would
    silently replace a predecessor terminal.  The refusal happens before the
    destination directory or any temporary file is touched.
    """
    if project.schema_version == SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION:
        raise ProjectStoreError(
            "schema-3 production projects must be written with "
            f"publish_project_no_clobber, not save_project: {destination}")


def save_project(project: ScanProject, path: str | Path, *,
                 max_bytes: int | None = None, retain_lease: bool = False):
    """Write a schema-1 or schema-2 project through an atomic replace.

    This is the schema-2 checkpoint writer and the general-purpose writer for
    schema-1/schema-2 payloads; its replace behavior for those schemas is
    unchanged.  A schema-3 production project is refused here: this writer would
    atomically clobber an existing durable terminal, so schema-3 payloads must be
    written through :func:`publish_project_no_clobber`, which never replaces an
    existing file.
    """
    destination = Path(path)
    _reject_production_schema(project, destination)
    if type(retain_lease) is not bool or (retain_lease and (max_bytes is None or os.name != "nt")):
        raise ProjectStoreError("retained project lease needs a capped Windows write")
    if max_bytes is not None:
        return _save_project_bounded(project, destination, max_bytes, retain_lease=retain_lease)
    document = _project_document(project, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(destination)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise ProjectStoreError(f"could not save project to {destination}: {error}") from error


def _indented_json_size(value, limit: int, *, depth: int = 0,
                        newline_bytes: int = 1) -> int:
    """Measure json.dumps(indent=2, ensure_ascii=True) before serialization.

    The bounded capture writer accepts only the store's plain JSON data. Depth,
    total encoded bytes and each string scan are bounded by the byte limit.
    """
    if depth > 64:
        raise ProjectStoreError("project JSON nesting exceeds 64")
    if isinstance(value, str):
        size = 2
        for char in value:
            code = ord(char)
            size += (2 if char in '\"\\\\\b\f\n\r\t' else
                     6 if code < 32 or 126 < code <= 0xffff else
                     12 if code > 0xffff else 1)
            if size > limit:
                raise ProjectStoreError("project_over_cap")
        return size
    if value is None:
        return 4
    if type(value) is bool:
        return 4 if value else 5
    if type(value) in (int, float):
        if type(value) is int:
            if value.bit_length() > max(1, limit) * 4:
                raise ProjectStoreError("project_over_cap")
            absolute = abs(value)
            digits = max(1, (absolute.bit_length() * 1233) >> 12)
            threshold = 10 ** digits
            while absolute >= threshold:
                digits += 1
                threshold *= 10
            while digits > 1 and absolute < threshold // 10:
                digits -= 1
                threshold //= 10
            size = digits + (value < 0)
            if size > limit:
                raise ProjectStoreError("project_over_cap")
            return size
        return len(json.dumps(value))  # a float's representation has bounded size
    if type(value) not in (dict, list, tuple):
        raise ProjectStoreError("project contains non-JSON data")
    if not value:
        return 2
    size = 2 + newline_bytes + 2 * depth
    items = value.items() if type(value) is dict else enumerate(value)
    for position, (key, item) in enumerate(items):
        size += 2 * (depth + 1) + newline_bytes
        if position:
            size += 1  # previous separator comma
        if type(value) is dict:
            if type(key) is not str:
                raise ProjectStoreError("project JSON key is not a string")
            size += _indented_json_size(key, limit - size, depth=depth + 1,
                                        newline_bytes=newline_bytes) + 2
        size += _indented_json_size(item, limit - size, depth=depth + 1,
                                    newline_bytes=newline_bytes)
        if size > limit:
            raise ProjectStoreError("project_over_cap")
    return size


def _write_bounded(stream, data: bytes, max_bytes: int) -> None:
    """Bound every write request; cope with short writes without duplicating data."""
    if len(data) > max_bytes:
        raise ProjectStoreError("project_over_cap")
    offset = 0
    view = memoryview(data)
    while offset < len(data):
        chunk = view[offset:min(len(data), offset + 65536)]
        count = stream.write(chunk)
        if type(count) is not int or not 0 < count <= len(chunk):
            raise ProjectStoreError("invalid/zero project write result")
        offset += count



def _windows_file_api():
    """Typed Windows file APIs; never silently fall back to pathname mutation."""
    if os.name != "nt":
        raise OSError("retained file handles require Windows")
    import ctypes
    from ctypes import wintypes as w

    class FileInfo(ctypes.Structure):
        _fields_ = [("attributes", w.DWORD), ("created", w.FILETIME),
                    ("accessed", w.FILETIME), ("written", w.FILETIME),
                    ("volume", w.DWORD), ("size_high", w.DWORD),
                    ("size_low", w.DWORD), ("links", w.DWORD),
                    ("index_high", w.DWORD), ("index_low", w.DWORD)]

    class FileId(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong),
                    ("identifier", ctypes.c_ubyte * 16)]

    class RenameInfo(ctypes.Structure):
        _fields_ = [("replace", w.BOOL), ("root", w.HANDLE),
                    ("length", w.DWORD), ("name", w.WCHAR * 1)]

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateFileW": ([w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p,
                         w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
        "GetFileInformationByHandle": ([w.HANDLE, ctypes.POINTER(FileInfo)], w.BOOL),
        "GetFileInformationByHandleEx": ([w.HANDLE, ctypes.c_int,
                                         ctypes.c_void_p, w.DWORD], w.BOOL),
        "SetFileInformationByHandle": ([w.HANDLE, ctypes.c_int,
                                         ctypes.c_void_p, w.DWORD], w.BOOL),
        "SetFilePointerEx": ([w.HANDLE, ctypes.c_longlong,
                              ctypes.c_void_p, w.DWORD], w.BOOL),
        "ReadFile": ([w.HANDLE, ctypes.c_void_p, w.DWORD,
                      ctypes.POINTER(w.DWORD), ctypes.c_void_p], w.BOOL),
        "WriteFile": ([w.HANDLE, ctypes.c_void_p, w.DWORD,
                       ctypes.POINTER(w.DWORD), ctypes.c_void_p], w.BOOL),
        "FlushFileBuffers": ([w.HANDLE], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
        "GetCurrentProcess": ([], w.HANDLE),
        "DuplicateHandle": ([w.HANDLE, w.HANDLE, w.HANDLE, ctypes.POINTER(w.HANDLE),
                             w.DWORD, w.BOOL, w.DWORD], w.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api, FileInfo, FileId, RenameInfo


class _WindowsRetainedFile:
    """One handle-bound file with write/delete exclusion and native file ID.

    Created files retain READ/WRITE/DELETE access from CREATE_NEW. Read snapshots
    retain READ access and deny competing WRITE/DELETE. Mutation always addresses
    this handle; no identity-check/pathname-unlink fallback is permitted.
    """

    def __init__(self, path, *, create=False, directory=False, shared_reader=False):
        import ctypes
        self._ctypes = ctypes
        self._api, self._info_type, self._id_type, self._rename_type = _windows_file_api()
        self.path = Path(os.path.abspath(path))
        self.handle = None
        self.directory = directory
        self.mutable = create
        self.writable = create
        access = 0x81 if directory else (0xC0010000 if create else 0x80000000)
        # Directory anchors deny DELETE, but do not pretend to exclude child creation.
        # LIST_DIRECTORY participates in sharing so DELETE/rename is excluded.
        # WRITE sharing is required by the current full-path rename publisher;
        # directory WRITE/reparse/child namespace exclusion is NOT proven.
        share = 3 if directory else (7 if shared_reader else 1)
        flags = 0x00200000 | (0x02000000 if directory else 0)
        handle = self._api.CreateFileW(str(self.path), access, share, None,
                                      1 if create else 3, flags, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        try:
            info = self._info()
            if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
                raise OSError("unsupported reparse/type in retained file handle")
            self.identity = self._identity()
        except BaseException:
            self.close()
            raise

    def _info(self):
        if self.handle is None:
            raise OSError("retained file handle is closed")
        info = self._info_type()
        if not self._api.GetFileInformationByHandle(self.handle, self._ctypes.byref(info)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return info

    def _identity(self):
        identifier = self._id_type()
        if self.handle is None or not self._api.GetFileInformationByHandleEx(
                self.handle, 18, self._ctypes.byref(identifier), self._ctypes.sizeof(identifier)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return (int(identifier.volume), bytes(identifier.identifier))

    @property
    def size(self):
        info = self._info()
        return (int(info.size_high) << 32) | int(info.size_low)

    def _verify(self):
        info = self._info()
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != self.directory:
            raise OSError("retained file reparse/type changed")
        if self._identity() != self.identity:
            raise OSError("retained file identity mismatch")

    def write(self, data):
        from ctypes import wintypes as w
        if not self.writable or self.directory or len(data) > 65536:
            raise OSError("invalid retained write request")
        self._verify()
        chunk = bytes(data)
        buffer = self._ctypes.create_string_buffer(chunk)
        written = w.DWORD()
        if not self._api.WriteFile(self.handle, buffer, len(chunk),
                                   self._ctypes.byref(written), None):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(written.value)

    def flush(self):
        if not self._api.FlushFileBuffers(self.handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def seal_read_only(self):
        """Drop WRITE rights without dropping the retained file-object exclusion."""
        from ctypes import wintypes as w
        if not self.mutable or self.directory:
            raise OSError("cannot seal unowned/directory file")
        if not self.writable:
            return
        self._verify()
        current, duplicate = self._api.GetCurrentProcess(), w.HANDLE()
        if not self._api.DuplicateHandle(current, self.handle, current,
                self._ctypes.byref(duplicate), 0x80010000, False, 0):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        previous, self.handle = self.handle, duplicate.value
        if not self._api.CloseHandle(previous):
            self.close()
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        self.writable = False
        self._verify()

    def read_bounded(self, cap):
        from ctypes import wintypes as w
        if type(cap) is not int or cap < 0 or self.size > cap:
            raise ProjectStoreError("retained artifact_over_cap")
        self._verify()
        if not self._api.SetFilePointerEx(self.handle, 0, None, 0):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        chunks, total = [], 0
        while True:
            count = min(65536, cap - total + 1)
            buffer, received = self._ctypes.create_string_buffer(count), w.DWORD()
            if not self._api.ReadFile(self.handle, buffer, count,
                                      self._ctypes.byref(received), None):
                raise self._ctypes.WinError(self._ctypes.get_last_error())
            if not received.value:
                break
            total += received.value
            if total > cap:
                raise ProjectStoreError("retained artifact_over_cap")
            chunks.append(buffer.raw[:received.value])
        self._verify()
        if self.size != total:
            raise OSError("retained artifact size changed")
        return b"".join(chunks)

    def rename_no_replace(self, destination):
        if not self.mutable:
            raise OSError("retained snapshot cannot rename")
        self._verify()
        destination = Path(os.path.abspath(destination))
        encoded = str(destination).encode("utf-16-le")
        length = self._rename_type.name.offset + len(encoded) + 2
        buffer = self._ctypes.create_string_buffer(length)
        info = self._rename_type.from_buffer(buffer)
        info.replace, info.root, info.length = False, None, len(encoded)
        self._ctypes.memmove(self._ctypes.addressof(buffer) + self._rename_type.name.offset,
                             encoded, len(encoded))
        if not self._api.SetFileInformationByHandle(self.handle, 3, buffer, length):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        self.path = destination
        self._verify()

    def delete_owned(self):
        if not self.mutable:
            raise OSError("retained snapshot cannot delete")
        self._verify()
        disposition = self._ctypes.c_ubyte(1)
        if not self._api.SetFileInformationByHandle(self.handle, 4,
                self._ctypes.byref(disposition), self._ctypes.sizeof(disposition)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self._api.CloseHandle(handle):
                raise self._ctypes.WinError(self._ctypes.get_last_error())

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _WindowsRootAnchor:
    """Retain ancestor LIST_DIRECTORY access against DELETE opens and rename.

    Namespace exclusivity is deliberately FALSE: WRITE access is shared for the
    current publisher; creation/removal/reparse mutation is not excluded. Exact inventory is observed
    at declared boundaries, never claimed atomic against hostile namespace writers.
    """
    namespace_exclusive = False

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        self.files = []
        try:
            for component in (*reversed(self.root.parents), self.root):
                self.files.append(_WindowsRetainedFile(component, directory=True))
            self.identity = self.files[-1].identity
        except BaseException:
            self.close()
            raise

    def verify(self):
        for file in self.files:
            file._verify()

    def close(self):
        files, self.files = self.files, []
        for file in reversed(files):
            file.close()

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ProjectWriteLease:
    """Runtime-only capped writer birth identity and retained immutable read lease.

    It is excluded from serialized project/entry payloads. Transfer consumes the
    lease once; release never reopens the pathname or adopts a replacement.
    """
    def __init__(self, held, anchor, cap, data):
        import hashlib
        if (not isinstance(held, _WindowsRetainedFile) or not held.mutable
                or type(cap) is not int or cap < 1 or len(data) > cap):
            raise ProjectStoreError("invalid project write lease")
        held.seal_read_only()
        self._held, self._anchor = held, anchor
        self.max_bytes, self.byte_length = cap, len(data)
        self.identity = held.identity
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.path = held.path

    @property
    def active(self):
        return self._held is not None and self._held.handle is not None

    def validate_live(self, expected_path):
        if not self.active or self.path != Path(os.path.abspath(expected_path)):
            raise ProjectStoreError("project_lease_missing/mismatched/released")
        self._held._verify()
        if self._held.identity != self.identity:
            raise ProjectStoreError("project_lease_identity_mismatch")
        self._anchor.verify()

    def read_bytes(self):
        import hashlib
        self.validate_live(self.path)
        data = self._held.read_bounded(self.max_bytes)
        if len(data) != self.byte_length or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ProjectStoreError("project_lease_bytes_mismatch")
        return data

    def take_file(self, destination):
        self.validate_live(self.path)
        self.read_bytes()
        destination = Path(os.path.abspath(destination))
        if destination != self.path:
            self._held.rename_no_replace(destination)
        held, self._held = self._held, None
        self._anchor.close()
        self._anchor = None
        return held

    def discard(self):
        if self.active:
            self._held.delete_owned()
        self.close()

    def close(self):
        if self._held is not None:
            self._held.close()
            self._held = None
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass


def _save_project_bounded(project, destination: Path, max_bytes: int, *, retain_lease=False):
    """Capture-only bounded, exclusive publication; ordinary replacement is unchanged."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ProjectStoreError("project byte limit must be a positive integer")
    # The session copy/canonicalizer itself allocates, so bound its plain input
    # first. The capture producer has an empty volume; ordinary callers remain
    # on the unchanged uncapped branch.
    _indented_json_size(project.image_derived_session, max_bytes)
    _indented_json_size(project.source_paths, max_bytes)
    _indented_json_size(project.name, max_bytes)
    _indented_json_size(project.minecraft_version, max_bytes)
    if len(project.volume.blocks) > max_bytes:
        raise ProjectStoreError("project_over_cap")
    for observation in project.volume.blocks.values():
        _indented_json_size(observation.state.name, max_bytes)
        _indented_json_size(observation.state.properties, max_bytes)
    data = _project_data(project, destination)
    newline_bytes = len(os.linesep)
    size = _indented_json_size(data, max_bytes - newline_bytes,
                              newline_bytes=newline_bytes) + newline_bytes
    if size > max_bytes:
        raise ProjectStoreError("project_over_cap")
    document = (json.dumps(data, indent=2) + "\n").replace(
        "\n", os.linesep).encode("utf-8")
    if len(document) != size:
        raise ProjectStoreError("project JSON size mismatch")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if os.path.lexists(destination) or os.path.lexists(temporary):
        raise ProjectStoreError("project destination/temporary already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        anchor = stream = None
        try:
            anchor = _WindowsRootAnchor(destination.parent)
            stream = _WindowsRetainedFile(temporary, create=True)
            _write_bounded(stream, document, max_bytes)
            stream.flush()
            stream.rename_no_replace(destination)
            if retain_lease:
                lease = ProjectWriteLease(stream, anchor, max_bytes, document)
                stream = anchor = None  # SAME birth handles transfer, never reopen/adopt
                return lease
        except BaseException as error:
            if stream is not None:
                stream.delete_owned()
            if isinstance(error, (OSError, TypeError, ValueError)):
                raise ProjectStoreError(f"could not save bounded project: {error}") from error
            raise
        finally:
            if stream is not None:
                stream.close()
            if anchor is not None:
                anchor.close()
        return
    owned = False
    identity = None
    try:
        with temporary.open("xb") as stream:
            owned = True
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino)
            _write_bounded(stream, document, max_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        raise ProjectStoreError(f"could not save bounded project: {error}") from error
    finally:
        if owned:
            try:
                info = os.lstat(temporary)
                if (info.st_dev, info.st_ino) == identity:
                    temporary.unlink()
            except FileNotFoundError:
                pass


def publish_project_no_clobber(project: ScanProject, path: str | Path) -> None:
    """Publish a project without ever replacing an existing destination.

    The payload is fully validated and staged in a fresh same-directory temporary
    file, then linked into place.  ``os.link`` is atomic and fails with
    ``FileExistsError`` when the destination appears between the caller's
    pre-check and publication, so a destination race, a failed write, a failed
    link or an interruption leaves any pre-existing bytes exactly as they were.
    The pre-check uses an occupied-entry test (``os.path.lexists``) so a dangling
    symbolic link at the destination is refused rather than replaced.
    """
    destination = Path(path)
    if os.path.lexists(destination):
        raise ProjectStoreError(f"project destination already exists: {destination}")
    document = _project_document(project, destination)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProjectStoreError(
            f"could not create the project directory for {destination}: {error}") from error
    try:
        handle, raw = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".staging",
            dir=str(destination.parent))
    except OSError as error:
        raise ProjectStoreError(
            f"could not stage project for {destination}: {error}") from error
    staging = Path(raw)
    try:
        os.close(handle)
    except OSError as error:
        try:
            staging.unlink()
        except OSError:
            pass
        raise ProjectStoreError(
            f"could not stage project for {destination}: {error}") from error
    try:
        staging.write_text(document, encoding="utf-8")
        try:
            os.link(staging, destination)
        except FileExistsError as error:
            raise ProjectStoreError(
                f"project destination already exists: {destination}") from error
        except OSError as error:
            raise ProjectStoreError(
                f"could not publish project to {destination}: {error}") from error
    except ProjectStoreError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ProjectStoreError(
            f"could not publish project to {destination}: {error}") from error
    finally:
        try:
            staging.unlink()
        except OSError:
            pass


def load_project(path: str | Path, *, snapshot_bytes: bytes | None = None) -> ScanProject:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8") if snapshot_bytes is None else snapshot_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ProjectStoreError(f"could not load project from {source}: {error}") from error
    if not isinstance(data, dict):
        raise ProjectStoreError("project root must be a JSON object")
    try:
        project = ScanProject.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectStoreError(f"invalid project data in {source}: {error}") from error
    session_record = project.image_derived_session
    canonical_session: dict[str, Any] | None = None
    if session_record is not None:
        try:
            canonical_session = _canonical_session_record(session_record)
        except ImageDerivedSessionError as error:
            raise ProjectStoreError(
                f"invalid image-derived session in {source}: {error}") from error
        project.image_derived_session = canonical_session
        _bind_v2_source_paths(canonical_session, project.source_paths)
    production_record = project.image_derived_production
    if production_record is not None:
        try:
            project.image_derived_production = _canonical_production_record(
                production_record,
                session_record=canonical_session,
                volume=project.volume,
            )
        except ProductionEnvelopeError as error:
            raise ProjectStoreError(
                f"invalid image-derived production in {source}: {error}") from error
    return project


def _canonical_session_record(record: Any) -> dict[str, Any]:
    """Verify a session record and return its canonical form.

    ``image_derived_session_from_record`` is the sole validator: it enforces the
    exact record key set, schema/version, digests, forbidden scale/translation
    claims, and retained unresolved facts. Re-serializing the rebuilt session
    with ``image_derived_session_to_record`` yields a canonical, deterministic
    record that never contains pixels.
    """
    return image_derived_session_to_record(
        image_derived_session_from_record(record))


def _bind_v2_source_paths(record, source_paths):
    if record.get("version") == 2:
        labels = [frame["source_label"] for frame in record["frames"]]
        if any(not isinstance(label, str) or not label for label in labels) or list(source_paths) != labels:
            raise ProjectStoreError("v2 source_paths must bind the full ordered input inventory")


def _canonical_production_record(
    record: Any,
    *,
    session_record: dict[str, Any] | None,
    volume: Any,
) -> dict[str, Any]:
    """Verify a schema-3 terminal envelope against the session and volume."""
    if session_record is None:
        raise ProductionEnvelopeError(
            "a production record requires the schema-2 session record")
    session_sha256 = session_record.get("session_sha256")
    return canonicalise_production_record(
        record, session_sha256=session_sha256, volume=volume)


def _project_data(project: ScanProject, destination: Path) -> dict:
    """Validate and canonicalize before choosing the bounded or legacy serializer."""
    try:
        data = project.to_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectStoreError(
            f"invalid project data for {destination}: {error}") from error
    record = data.get(IMAGE_DERIVED_SESSION_FIELD)
    canonical_session: dict[str, Any] | None = None
    if record is not None:
        try:
            canonical_session = _canonical_session_record(record)
        except ImageDerivedSessionError as error:
            raise ProjectStoreError(
                f"invalid image-derived session for {destination}: {error}") from error
        data[IMAGE_DERIVED_SESSION_FIELD] = canonical_session
        _bind_v2_source_paths(canonical_session, data["source_paths"])
    production = data.get(IMAGE_DERIVED_PRODUCTION_FIELD)
    if production is not None:
        try:
            data[IMAGE_DERIVED_PRODUCTION_FIELD] = _canonical_production_record(
                production, session_record=canonical_session,
                volume=project.volume)
        except ProductionEnvelopeError as error:
            raise ProjectStoreError(
                f"invalid image-derived production for {destination}: {error}") from error
    return data


def _project_document(project: ScanProject, destination: Path) -> str:
    data = _project_data(project, destination)
    try:
        return json.dumps(data, indent=2) + "\n"
    except (TypeError, ValueError) as error:
        raise ProjectStoreError(
            f"invalid project data for {destination}: {error}") from error
