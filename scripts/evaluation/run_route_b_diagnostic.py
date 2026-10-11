"""DS-R4-V2-ROUTEB-RUNNER: one-shot internal Route B complete-support runner.

This is the separate, reviewed runner the Lead required before any real Route B
diagnostic: it captures the **exact single production pass** over ordinary
screenshot inputs and records the complete V2 support universe through the
already-accepted, default-off seams.

Call chain (each stage exactly once, no re-sign / re-match / re-filter / re-solve):

1. ``build_image_derived_entry(..., capture_ingress=True)`` - the ordinary
   image-only entry. ``capture_ingress=True`` attaches the entry's own typed
   per-frame ingress pin to the session; it is identity-bearing metadata about
   the ordinary single decoded read, never an externally supplied manifest,
   annotation record, direction, mask or truth file. The entry terminal is
   accepted by the production readiness contract, not by a ``status`` field:
   ``ImageDerivedEntryResult`` deliberately has no ``status`` and its normal
   successful terminal is the informational ``GEOMETRY_UNAVAILABLE`` with an
   empty ``reason``. The runner therefore proceeds whenever
   ``entry.graph.status == "ok"`` and ``entry.pose_graph.status == "ok"``; only
   a typed ``PlacementRefusal`` or a graph/pose-not-ok entry refuses, carrying
   the entry's own typed reason and detail.
2. ``measure_image_derived_lattice`` - the ordinary lattice measure.
3. ``derive_image_derived_signatures(..., signature_witness_sink=[...])`` - the
   ordinary signature pass also emits its own ``SourceTapWitnessBindingV1``
   values (bound to the accepted session ingress pins) into the sink. This is
   the *same* pass; the runner never re-signs to obtain witnesses.
4. ``derive_image_derived_graph_attribution(...)`` **once**, forwarding
   ``signature_witnesses`` and ``complete_support_sink`` through the adapter's
   existing ``**kwargs`` into ``_derive_image_derived_graph``. The graph seam
   attaches the V2 collector to the same pre-filter witness rows, same raw
   observation, same post-filter match tuple and same accepted session ingress
   pins it already produced, and calls it only when the component solver
   abstains.

The runner then enforces the accepted identity gate on the frozen 1,928-row
profile - exact row count, complete admitted order, the sealed
``whole_universe_identity_sha256`` recomputed from the persisted index rows, the
kept-subset ``input_match_identity_sha256``, and the module's strict
``validate_complete_support_capture`` replay - and serializes the canonical V2
record **only after** that replay validation succeeds, atomically. Any count,
order, content, identity or cap failure refuses the whole run atomically.

Content identity (Lead requirement). No independently pre-registered, non-truth
digest of the exact frozen 1,928 admitted rows exists anywhere in this
repository, and deriving one now from a different run's artifact would bind a
foreign universe rather than this pass. The accepted content identity for this
diagnostic is therefore exactly the two things the capture itself seals and this
runner independently replays: the collector's exact ``expected_row_count =
1,928`` and the **sealed** ``whole_universe_identity_sha256`` recomputed by this
runner from the persisted index rows (and re-checked by the module validator).
There is **no** caller-supplied identity pin and no independently preregistered
universe digest; the interface exposes no option, argument or keyword that can
weaken, replace or bypass that gate.

Input contract. The normal runner interface is ordinary screenshot paths plus
one output root: ``execute_route_b_diagnostic(image_paths=..., root=...)``, or
the CLI's ``--image`` (repeatable, in order) and ``--root``. Exactly the frozen
three-image input is accepted; a fourth image is refused because the sealed
row count and digest prove run integrity, not that a fourth view belongs to the
same frozen inventory. The runner never pre-reads or pre-hashes an input path:
each recorded source identity is taken verbatim from the entry's own
``FrameIngressProvenanceV1.byte_sha256``, computed from the exact encoded byte
buffer the entry decoded, and the ordered pins are bound back to the supplied
paths positionally, so a permuted pin order refuses and the recorded inputs stay
in contract order. Before anything is written, the entry's ordered pins, the
complete capture's own ``root.ingress``, the serialized top-level ``ingress``
records and the positional ``inputs`` rows are cross-bound; a foreign or permuted
capture-root ingress refuses. ``validate_record`` repeats the same cross-binding
against the persisted record, so tampering with an outer ``ingress``/``inputs``
row refuses even after the outer artifact digest is recomputed. The runner
accepts no caller-supplied image hash, identity pin, settings
mapping or record-mode switch. The graph thresholds are frozen to the production
defaults in ``DEFAULT_SETTINGS``. ``api`` exists only as an injectable test
seam.

Failure atomicity. The entry's schema-2 project is written into a fresh staging
child of the validated output root and linked into place only after the full
chain and its replay gate succeed. Any refusal cleans up the exact runner-owned
staged ``route_b.mcscan``/``.partial`` names and the exact empty staging
directory with nonrecursive ``unlink``/``rmdir``; no tree walk or recursive
deletion is ever performed. The record and project are published with
``os.link`` no-clobber semantics; ``.partial`` cleanup removes only a file this
invocation created with a successful exclusive open, so a name collision
preserves pre-existing bytes. A symlink/reparse output root, or any existing
reparse ancestor component of the absolute output root up to its filesystem
anchor, is refused from read-only metadata before anything is created, and an
``lstat`` error refuses closed.

No real capture is performed by this task and none is authorized by writing this
file. The CLI refuses any non-PNG/absent input, and no annotation, truth or
externally supplied alignment source is referenced. It changes no production
source, default path or output digest.
"""
from __future__ import annotations

import argparse
import dataclasses
from contextlib import nullcontext
import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA = "mc-photogrammetry-route-b-diagnostic-runner-v1"
# New captures write the V2 record with the independently versioned, capped
# ``sampling_persistence`` section. The V1 schema stays accepted on read as
# legacy with the sampling payload typed unavailable; the frozen R4 record is
# never rewritten.
RECORD_SCHEMA = "mc-photogrammetry-route-b-diagnostic-record-v2"
LEGACY_RECORD_SCHEMA = "mc-photogrammetry-route-b-diagnostic-record-v1"
RECORD_SCHEMAS = (RECORD_SCHEMA, LEGACY_RECORD_SCHEMA)
REFUSAL_SCHEMA = "mc-photogrammetry-route-b-diagnostic-runner-refusal-v1"

RECORD_NAME = "record.json"
REFUSAL_NAME = "refusal.json"
PROJECT_NAME = "route_b.mcscan"
STAGING_DIR_NAME = "route_b_staging"
# Every name any Phase A/route-B invocation may own in one output root. A root
# that already carries any of them is a collision: the runner refuses with zero
# writes (no staging, no refusal diagnostics).
OWNED_ARTIFACT_NAMES = (
    RECORD_NAME, PROJECT_NAME, REFUSAL_NAME, STAGING_DIR_NAME,
    "diagnostics.json", "scorecard.json", "freeze.json")
# One typed refusal record cap; the declared canonical size is checked before
# any bytes are written.
MAX_REFUSAL_BYTES = 64 * 1024

# The complete serialized V2 record cap. Base64 inflates the capped 8 MiB sampling
# payload to ~10.7 MiB on top of the unchanged support capture, so the accepted
# profile stays inside this bound; the sampling section degrades to a typed
# refusal rather than pushing the record over it.
MAX_RECORD_BYTES = 32 * 1024 * 1024

# The frozen Route B fixture is the documented three-view T2 input. Exactly three
# ordinary image inputs are accepted: the 1,928-row universe plus its sealed
# digest proves run integrity, not that a fourth view belongs to the same
# candidate inventory, so a fourth input is refused rather than admitted.
ALLOWED_FRAME_COUNTS = (3,)
DEFAULT_SETTINGS = {
    "max_shift_cells": 32,
    "integer_tolerance": 0.15,
    "min_resultant": 0.90,
    "ambiguous_margin": 0.02,
}
# The frozen single-run expectation is owned by the V2 module; the runner reads
# it live so the two can never drift apart.
EXPECTED_ROW_COUNT = 1_928

# Windows reparse tags used by the read-only output-root probe. ``_reparse_tag``
# only inspects metadata; it never creates or manipulates a reparse point.
_REPARSE_TAG_MOUNT_POINT = 0xA0000003
_REPARSE_TAG_SYMLINK = 0xA000000C


# ---------------------------------------------------------------------------
# Canonical JSON / digest helpers (mirror the evaluation harnesses)
# ---------------------------------------------------------------------------

def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path, *, ownership=None) -> str:
    """Post-write digest of one artifact this runner already published.

    This is never used on an input screenshot path: source identity comes only
    from the entry's own ingress pins.
    """
    cap = MAX_RECORD_BYTES
    if ownership is not None and ownership.native and ownership.owns(path):
        return hashlib.sha256(ownership.read_bytes(path, cap)).hexdigest()
    if os.name == "nt":
        from mc_photogrammetry.project.store import _WindowsRetainedFile, _WindowsRootAnchor
        with _WindowsRootAnchor(Path(path).parent):
            with _WindowsRetainedFile(path) as held:
                return hashlib.sha256(held.read_bounded(cap)).hexdigest()
    with Path(path).open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise DiagnosticRefusal("artifact_over_cap", Path(path).name)
    return hashlib.sha256(data).hexdigest()


# --- Explicit, nonrecursive runner-owned cleanup -----------------------------

def _unlink_if_present(path: Path) -> None:
    """Remove exactly one named runner-owned file; never walk or follow links."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _rmdir_if_empty(path: Path) -> None:
    """Remove exactly one staging directory only when it is already empty.

    ``Path.rmdir`` refuses a non-empty directory, so this can never recurse over
    or delete an unexpected entry.
    """
    try:
        Path(path).rmdir()
    except (FileNotFoundError, OSError):
        pass


class InvocationOwnership:
    """Invocation ownership: birth handles for native captures, legacy path IDs otherwise.

    Native files are registered only from a retained CREATE_NEW handle/typed lease;
    disposition addresses that same object. Legacy ordinary callers preserve their
    prior sequential identity checks, which are not hostile-writer race protection.
    Directory child namespaces are not sealed by either mode.
    """

    def __init__(self, *, native=False) -> None:
        self._owned: dict = {}
        self.native = bool(native and os.name == "nt")
        self._handles = {}
        self._anchor = None
        self._staging_anchors = {}

    def anchor(self, root):
        if self.native and self._anchor is None:
            from mc_photogrammetry.project.store import _WindowsRootAnchor
            self._anchor = _WindowsRootAnchor(root)

    def attach_project_lease(self, lease, path, *, max_bytes):
        from mc_photogrammetry.project.store import ProjectWriteLease, _WindowsRootAnchor
        if not self.native or not isinstance(lease, ProjectWriteLease):
            raise DiagnosticRefusal("entry_project_lease_unavailable", "entry")
        try:
            lease.validate_live(path)
            if lease.max_bytes != max_bytes or lease.byte_length > max_bytes:
                raise ValueError("project lease cap mismatch")
            staging = Path(os.path.abspath(path)).parent
            self._staging_anchors[staging] = _WindowsRootAnchor(staging)
            self.register_handle(path, lease.take_file(path))
        except (OSError, ValueError) as error:
            lease.close()  # preserve bytes; never adopt/delete an invalid lease by path
            raise DiagnosticRefusal("entry_project_lease_invalid", "entry", str(error)) from error

    def release_staging_anchor(self, staging):
        anchor = self._staging_anchors.pop(Path(os.path.abspath(staging)), None)
        if anchor is not None:
            anchor.close()

    def register_handle(self, path, held):
        if not self.native:
            raise DiagnosticRefusal("ownership_unavailable", "root")
        resolved = Path(os.path.abspath(path))
        held._verify()
        if held.path != resolved or not held.mutable:
            raise DiagnosticRefusal("ownership_identity_mismatch", resolved.name)
        self._owned[resolved] = held.identity
        self._handles[resolved] = held
        return resolved

    def take_handle(self, path):
        resolved = Path(os.path.abspath(path))
        held = self._handles.pop(resolved, None)
        self._owned.pop(resolved, None)
        return held

    def read_bytes(self, path, cap):
        held = self._handles.get(Path(os.path.abspath(path)))
        if held is None:
            raise DiagnosticRefusal("ownership_unavailable", Path(path).name)
        return held.read_bounded(cap)

    def close(self):
        handles, self._handles = self._handles, {}
        for held in reversed(list(handles.values())):
            held.close()
        for anchor in self._staging_anchors.values():
            anchor.close()
        self._staging_anchors.clear()
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def register(self, path) -> Path:
        resolved = Path(os.path.abspath(Path(path)))
        if self.native:
            # Native ownership must come from CREATE_NEW; a later name lookup
            # cannot establish that the object was born to this invocation.
            raise DiagnosticRefusal("ownership_birth_unproven", resolved.name)
        self._owned[resolved] = self._identity(resolved)
        return resolved

    @staticmethod
    def _identity(path):
        info = os.lstat(path)
        return (info.st_dev, info.st_ino, info.st_ctime_ns, info.st_mtime_ns,
                info.st_size, info.st_mode, getattr(info, "st_reparse_tag", 0))

    def owns(self, path) -> bool:
        return Path(os.path.abspath(Path(path))) in self._owned

    def unlink(self, path) -> bool:
        resolved = Path(os.path.abspath(Path(path)))
        if resolved not in self._owned:
            return False
        if self.native:
            held = self._handles[resolved]
            held.delete_owned()
            held.close()
            self._handles.pop(resolved, None)
            self._owned.pop(resolved, None)
            return True
        try:
            if self._identity(resolved) != self._owned[resolved]:
                self._owned.pop(resolved, None)
                return False
        except OSError:
            self._owned.pop(resolved, None)
            return False
        _unlink_if_present(resolved)
        self._owned.pop(resolved, None)
        return True

    def rollback(self) -> None:
        for resolved in list(reversed(list(self._owned))):
            self.unlink(resolved)

    def paths(self) -> tuple:
        return tuple(self._owned)


def _reparse_tag(path: Path) -> int:
    """Read-only metadata probe for one symlink/reparse path.

    Returns ``0`` for an absent path or an ordinary file/directory and a nonzero
    Windows reparse tag (or a synthetic symlink tag) for a symlink, junction or
    mount point. It only inspects ``os.lstat`` metadata; it never creates,
    follows or edits a reparse point.

    Fail-closed: an absent component (``FileNotFoundError``) is treated as
    ordinary, but every other ``lstat`` error refuses instead of silently
    reporting an ordinary path. In particular ``NotADirectoryError`` (a
    non-directory parent component such as ``.../file/child``) is an inspection
    failure, not evidence that the component is absent, so it refuses closed.
    """
    try:
        info = os.lstat(Path(path))
    except FileNotFoundError:
        # An absent component is ordinary: there is nothing to redirect through.
        return 0
    except NotADirectoryError as error:
        # A non-directory parent component cannot be inspected; refusing closed
        # is required because its metadata is unreadable, not absent.
        raise DiagnosticRefusal(
            "root_uninspectable", "root",
            "output-root metadata is not inspectable") from error
    except OSError as error:
        raise DiagnosticRefusal(
            "root_uninspectable", "root",
            "output-root metadata is not inspectable") from error
    if stat.S_ISLNK(info.st_mode):
        return int(getattr(info, "st_reparse_tag", 0) or _REPARSE_TAG_SYMLINK)
    return int(getattr(info, "st_reparse_tag", 0) or 0)


def _refuse_reparse_components(root: Path) -> None:
    """Probe every existing output-root component from the path to its anchor.

    The absolute root and each of its ancestors (up to and including the
    filesystem anchor) are inspected with read-only ``os.lstat`` metadata. An
    absent component is fine; any reparse/symlink component, or any component
    whose metadata cannot be read, refuses before the runner creates anything.
    """
    root = Path(root)
    for component in (root, *root.parents):
        if _reparse_tag(component):
            raise DiagnosticRefusal(
                "root_reparse", "root",
                f"output-root component {component} is a symlink/reparse point")


# --- Race-safe no-clobber publication ----------------------------------------

def _write_partial(temporary: Path, data: bytes):
    """Exclusive partial creation. Windows retains birth identity through publication."""
    if os.name == "nt":
        from mc_photogrammetry.project.store import _WindowsRetainedFile, _write_bounded
        held = _WindowsRetainedFile(temporary, create=True)
        try:
            _write_bounded(held, data, len(data))
            held.flush()
            held.seal_read_only()
            return held
        except BaseException:
            try:
                held.delete_owned()
            finally:
                held.close()
            raise
    path = Path(temporary)
    stream = path.open("xb")
    try:
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _unlink_if_present(path)
        raise
    return None


def _publish_owned_partial(temporary, destination, held):
    if os.name == "nt":
        if held is None:
            raise DiagnosticRefusal("ownership_birth_unproven", Path(destination).name)
        held.rename_no_replace(destination)
    else:
        os.link(temporary, destination)


def _atomic_publish(path: Path, payload: object, *, field: str = "record",
                    ownership: "InvocationOwnership | None" = None,
                    max_bytes: int | None = None, output_budget=None) -> None:
    """Premeasure/write and no-replace publish; Windows cleanup uses the birth handle."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    cap = MAX_RECORD_BYTES if max_bytes is None else max_bytes
    declared = persistence_canonical_size(payload) + 1
    if type(cap) is not int or declared > cap:
        raise DiagnosticRefusal(f"{field}_over_cap", field, f"{declared} > {cap} bytes")
    if output_budget is not None:
        output_budget.replace(declared, field=path.name)
    data = _canonical_bytes(payload) + b"\n"
    if len(data) != declared:
        raise DiagnosticRefusal("serialization_size_mismatch", field)
    from mc_photogrammetry.project.store import _WindowsRootAnchor
    anchor = _WindowsRootAnchor(path.parent) if os.name == "nt" else nullcontext()
    with anchor:
        held = None
        owned = published = False
        try:
            try:
                held = _write_partial(temporary, data)
                owned = True
            except FileExistsError as error:
                raise DiagnosticRefusal("output_exists", field, str(error)) from error
            try:
                _publish_owned_partial(temporary, path, held)
                published = True
            except OSError as error:
                reason = "output_exists" if isinstance(error, FileExistsError) or getattr(error, "winerror", None) in (80, 183) else "output_failed"
                raise DiagnosticRefusal(reason, field, str(error)) from error
            if ownership is not None:
                if ownership.native:
                    ownership.register_handle(path, held)
                    held = None  # ledger retains the SAME birth handle
                else:
                    ownership.register(path)
        finally:
            if held is not None:
                try:
                    if not published:
                        held.delete_owned()
                finally:
                    held.close()
            elif owned and os.name != "nt":
                _unlink_if_present(temporary)


class DiagnosticRefusal(RuntimeError):
    """One typed, atomic runner refusal (never a partial record)."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.field = field
        self.detail = detail


# ---------------------------------------------------------------------------
# Injectable ordinary-image API (the single-pass seam the tests count)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteBApi:
    entry: object
    lattice: object
    signatures: object
    graph: object


def _default_api() -> RouteBApi:
    from mc_photogrammetry.pipeline.image_derived_entry import (
        build_image_derived_entry)
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import (
        derive_image_derived_graph_attribution)
    from mc_photogrammetry.pipeline.image_derived_lattice_measure import (
        measure_image_derived_lattice)
    from mc_photogrammetry.pipeline.image_derived_signatures import (
        derive_image_derived_signatures)
    return RouteBApi(build_image_derived_entry, measure_image_derived_lattice,
                     derive_image_derived_signatures,
                     derive_image_derived_graph_attribution)


def _resolve_api(api) -> RouteBApi:
    if api is None:
        return _default_api()
    if isinstance(api, RouteBApi):
        return api
    if isinstance(api, tuple) and len(api) == 4:
        return RouteBApi(*api)
    raise DiagnosticRefusal("invalid_api", "api")


# ---------------------------------------------------------------------------
# Input contract: ordinary screenshot paths only
# ---------------------------------------------------------------------------

def resolve_images(images):
    """Validate ordinary screenshot inputs; never read or hash their bytes.

    The runner accepts exactly three ordinary ``.png`` screenshot paths. It never
    takes a caller-supplied hash or identity pin and it never pre-reads the
    files: the recorded source identity is taken only from the entry's own
    ``FrameIngressProvenanceV1.byte_sha256`` pins, which are computed from the
    exact byte buffer the entry decoded. This function therefore performs
    existence/extension/cardinality checks only and returns the ordered paths.
    """
    paths = tuple(Path(value) for value in images)
    if len(paths) != ALLOWED_FRAME_COUNTS[0]:
        raise DiagnosticRefusal(
            "invalid_input", "png_count",
            "exactly three PNG paths are required")
    if len(set(paths)) != len(paths):
        raise DiagnosticRefusal("invalid_input", "png_paths",
                                "PNG inputs must be distinct")
    for position, path in enumerate(paths):
        if path.suffix.lower() != ".png":
            raise DiagnosticRefusal("invalid_input", f"png.{position}.suffix",
                                    "only ordinary .png screenshot inputs are accepted")
        if not path.is_file():
            raise DiagnosticRefusal("invalid_input", f"png.{position}.path",
                                    "the screenshot path does not exist")
    return paths


def resolve_settings():
    """The frozen production graph settings for this diagnostic.

    This is a constant lookup, not an input: the runner exposes no caller-facing
    settings surface to weaken or replace the production thresholds.
    """
    return dict(DEFAULT_SETTINGS)


# ---------------------------------------------------------------------------
# The single production pass
# ---------------------------------------------------------------------------

@dataclass
class ChainRun:
    """The exact in-memory material one single pass produced."""

    entry: object
    lattice: object
    signatures: object
    graph: object
    trace: object
    signature_witnesses: tuple
    capture: object
    elapsed_seconds: float
    calls: tuple
    pin_hashes: tuple = ()
    solver_snapshots: tuple = ()


def _progress_module():
    """The evaluation-only Phase A scorecard/policy module (lazy import)."""
    from mc_photogrammetry import evaluation_template_match_progress as progress
    return progress


def _solver_snapshot_summary(snapshot, *, max_axes: int = 8) -> dict:
    """One bounded, serializable summary of a same-pass solver snapshot.

    The raw snapshot may carry large arrays; a diagnostic checkpoint keeps only
    the typed scalars Phase A needs to attribute a solver abstention. No value is
    invented: an absent field stays ``None`` and an axis pair list is only its
    integer keys. Numeric axis residual/resultant values are included as bounded
    ``[[axis, rounded_value], ...]`` diagnostics - they are the same pass's own
    numbers, never recomputed and never a second solver pass.
    """

    def text(value):
        return value if type(value) is str else None

    def axis_pairs(value):
        pairs = []
        try:
            for pair in value:
                if (type(pair) is tuple and len(pair) == 2
                        and type(pair[0]) is int):
                    number = pair[1]
                    if type(number) in (int, float):
                        pairs.append([pair[0], round(float(number), 9)])
                    else:
                        pairs.append([pair[0], None])
                    if len(pairs) >= max_axes:
                        break
        except TypeError:
            return None
        return sorted(pairs, key=lambda item: item[0])

    def axis_keys(value):
        pairs = axis_pairs(value)
        if pairs is None:
            return None
        return [axis for axis, _ in pairs]

    return {
        "reason": text(getattr(snapshot, "reason", None)),
        "invalid_field": text(getattr(snapshot, "invalid_field", None)),
        "frames": len(getattr(snapshot, "frames", ()) or ()),
        "quadrants": len(getattr(snapshot, "quadrants", ()) or ()),
        "axis_resultant_keys": axis_keys(
            getattr(snapshot, "axis_resultants", ()) or ()),
        "axis_residual_keys": axis_keys(
            getattr(snapshot, "axis_residuals", ()) or ()),
        "axis_resultants": axis_pairs(
            getattr(snapshot, "axis_resultants", ()) or ()),
        "axis_residuals": axis_pairs(
            getattr(snapshot, "axis_residuals", ()) or ()),
        "has_solve_context": getattr(snapshot, "solve_context", None) is not None,
        "has_active_edge_aliases": getattr(
            snapshot, "active_edge_aliases", None) is not None,
        "has_boundary_saturation": getattr(
            snapshot, "boundary_saturation", None) is not None,
    }


def _emit_checkpoint(sink, name: str, **fields) -> None:
    """Emit one typed stage checkpoint to the optional Phase A sink.

    The sink is the caller's bounded diagnostics journal; it may itself raise a
    typed refusal or a resource abort, which propagates exactly like any other
    single-pass failure (the runner's own rollback still runs).
    """
    if sink is None:
        return
    record = {"stage": name}
    record.update(fields)
    sink(record)



def _status_refusal(stage, value):
    from mc_photogrammetry.reconstruction import plane_placement as placement
    if isinstance(value, placement.PlacementRefusal):
        return DiagnosticRefusal(value.reason, stage, value.detail)
    status = getattr(value, "status", None)
    if status != "assembled":
        return DiagnosticRefusal(getattr(value, "reason", "") or "unavailable",
                                 stage, getattr(value, "detail", ""))
    return None


def _entry_refusal(entry):
    """Adapter from the entry terminal contract to one typed runner refusal.

    ``ImageDerivedEntryResult`` intentionally has no ``status`` field and its
    normal successful terminal is always ``GEOMETRY_UNAVAILABLE`` with an empty
    ``reason`` and an informational ``detail``. Reading that terminal through a
    status check would refuse every successful entry, so this mirrors the
    production readiness contract exactly: the entry is accepted when both
    ``entry.graph.status == "ok"`` and ``entry.pose_graph.status == "ok"``, and
    the informational geometry-unavailable terminal then proceeds to the
    ordinary image-only lattice/signature/graph chain.

    Typed handling is preserved first: a ``PlacementRefusal`` is carried through
    as a typed entry refusal exactly like the other stages. Otherwise only a
    graph- or pose-graph-not-ok entry refuses, carrying that terminal's own typed
    ``reason`` and ``detail`` (falling back to a typed reason and an explanatory
    detail only when the entry carries none). Returns ``None`` for a normal
    successful entry terminal.
    """
    from mc_photogrammetry.reconstruction import plane_placement as placement

    if isinstance(entry, placement.PlacementRefusal):
        return DiagnosticRefusal(entry.reason, "entry", entry.detail)
    graph_status = getattr(getattr(entry, "graph", None), "status", None)
    pose_status = getattr(getattr(entry, "pose_graph", None), "status", None)
    if graph_status == "ok" and pose_status == "ok":
        return None
    detail = getattr(entry, "detail", "") or ""
    if not detail:
        unavailable = []
        if graph_status != "ok":
            unavailable.append(f"connection graph ({graph_status or 'missing'})")
        if pose_status != "ok":
            unavailable.append(f"pose graph ({pose_status or 'missing'})")
        detail = "entry " + " and ".join(unavailable) + " is unavailable"
    return DiagnosticRefusal(
        getattr(entry, "reason", "") or "entry_unavailable", "entry", detail)


def _run_single_pass(*, resolved, project_path, settings, api,
                     expected_pins=None, observer=None,
                     checkpoint=None, project_max_bytes=None, project_ownership=None) -> ChainRun:
    """Run the ordinary image-only chain exactly once with the V2 sinks attached.

    ``expected_pins`` (Phase A only) is the ordered accepted T2 pin profile: when
    supplied it is compared against the entry's own same-pass ingress pins
    **before** lattice or signatures run, so a foreign or arbitrary PNG refuses
    ``input_pin_mismatch`` before any later stage. ``observer`` collects the
    same-pass solver-failure snapshots (no second graph/matcher/solver pass);
    ``checkpoint`` receives one bounded typed record per stage boundary.
    """
    from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2

    started = time.monotonic()
    calls: list[str] = []
    solver_snapshots: list = [] if observer is None else observer

    paths = [str(path) for path in resolved]
    # One pass means each ordinary stage is entered exactly once, in order. The
    # stage name is appended immediately before each direct invocation, so the
    # recorded call sequence is the actual pass the runner performed.
    calls.append("entry")
    entry_kwargs = {"project_path": Path(project_path), "capture_ingress": True}
    if project_max_bytes is not None:
        entry_kwargs["project_max_bytes"] = project_max_bytes
    if project_ownership is not None and project_ownership.native:
        entry_kwargs["retain_project_lease"] = True
    try:
        entry = api.entry(paths, **entry_kwargs)
    except Exception as error:
        # Only the additive bounded-capture branch translates its new producer
        # persistence exception; ordinary legacy calls retain their behavior.
        if project_max_bytes is None:
            raise
        from mc_photogrammetry.pipeline.image_derived_entry import ImageDerivedEntryError
        if not isinstance(error, ImageDerivedEntryError):
            raise
        _emit_checkpoint(checkpoint, "entry", status="refused",
                         reason=error.reason, detail=error.detail,
                         elapsed_seconds=time.monotonic() - started)
        raise DiagnosticRefusal(error.reason, "entry", error.detail) from error
    if project_ownership is not None and project_ownership.native:
        try:
            project_ownership.attach_project_lease(getattr(entry, "project_lease", None), project_path,
                                                   max_bytes=project_max_bytes)
        except DiagnosticRefusal as error:
            _emit_checkpoint(checkpoint, "entry", status="refused", reason=error.reason,
                             detail=error.detail, elapsed_seconds=time.monotonic() - started)
            raise
    # The entry checkpoint is emitted before the entry terminal is judged, so a
    # typed entry refusal still leaves its stage evidence in the Phase A journal.
    _emit_checkpoint(checkpoint, "entry",
                     entry_sha256=getattr(entry, "entry_sha256", ""),
                     session_sha256=getattr(getattr(entry, "session", None),
                                            "session_sha256", ""),
                     project_file_sha256=_file_sha256(project_path, ownership=project_ownership)
                         if Path(project_path).is_file() else None,
                     elapsed_seconds=time.monotonic() - started)
    failure = _entry_refusal(entry)
    if failure is not None:
        raise failure

    pin_hashes: tuple = ()
    if expected_pins is not None:
        progress = _progress_module()
        pins = tuple(pin for pin in
                     (getattr(frame, "ingress", None)
                      for frame in getattr(entry.session, "frames", ()))
                     if pin is not None)
        try:
            pin_hashes = progress.check_pin_profile(pins,
                                                    expected=expected_pins)
        except progress.TemplateMatchProgressError as error:
            raise DiagnosticRefusal(error.reason, error.field,
                                    error.detail) from error
        _emit_checkpoint(checkpoint, "pins", pin_count=len(pin_hashes))

    calls.append("lattice")
    lattice = api.lattice(session=entry.session, pose_graph=entry.pose_graph)
    # Emit the stage checkpoint *before* judging the status so a typed lattice
    # refusal still leaves its own bounded evidence in the journal.
    _emit_checkpoint(checkpoint, "lattice",
                     status=getattr(lattice, "status", ""),
                     reason=getattr(lattice, "reason", "") or "",
                     detail=getattr(lattice, "detail", "") or "",
                     result_sha256=getattr(lattice, "result_sha256", ""),
                     reading_count=len(getattr(lattice, "readings", ()) or ()),
                     elapsed_seconds=time.monotonic() - started)
    failure = _status_refusal("lattice", lattice)
    if failure is not None:
        raise failure

    signature_witnesses: list = []
    calls.append("signatures")
    signatures = api.signatures(
        session=entry.session, lattice_result=lattice,
        signature_witness_sink=signature_witnesses)
    # Same rule: persist the signature stage evidence before the refusal check.
    _emit_checkpoint(checkpoint, "signatures",
                     status=getattr(signatures, "status", ""),
                     reason=getattr(signatures, "reason", "") or "",
                     detail=getattr(signatures, "detail", "") or "",
                     result_sha256=getattr(signatures, "result_sha256", ""),
                     witness_count=len(signature_witnesses),
                     signature_count=getattr(signatures, "signature_count",
                                             len(signature_witnesses)),
                     elapsed_seconds=time.monotonic() - started)
    failure = _status_refusal("signatures", signatures)
    if failure is not None:
        raise failure

    captured: list = []
    calls.append("graph")
    graph_kwargs = dict(settings)
    if observer is not None:
        graph_kwargs["solver_failure_observer"] = solver_snapshots.append
    graph, trace = api.graph(
        session=entry.session, lattice_result=lattice,
        signatures_result=signatures, connection_graph=entry.graph,
        signature_witnesses=signature_witnesses,
        complete_support_sink=captured.append, **graph_kwargs)
    from mc_photogrammetry.reconstruction import plane_placement as placement
    if isinstance(graph, placement.PlacementRefusal):
        _emit_checkpoint(checkpoint, "graph",
                         status="refused",
                         reason=graph.reason,
                         detail=graph.detail,
                         solver=[_solver_snapshot_summary(snapshot)
                                 for snapshot in solver_snapshots],
                         elapsed_seconds=time.monotonic() - started)
        raise DiagnosticRefusal(graph.reason, "graph", graph.detail)
    if trace is None:
        raise DiagnosticRefusal("trace_unavailable", "graph")
    _emit_checkpoint(
        checkpoint, "graph",
        status=getattr(graph, "status", ""),
        reason=getattr(graph, "reason", "") or "",
        detail=getattr(graph, "detail", "") or "",
        result_sha256=getattr(graph, "result_sha256", ""),
        trace_stage=getattr(trace, "stage", ""),
        post_admission_candidates=getattr(
            trace, "post_admission_candidates", None),
        post_same_axis_candidates=getattr(
            trace, "post_same_axis_candidates", None),
        proposal_count=getattr(trace, "proposal_count", None),
        captured_count=len(captured),
        solver=[_solver_snapshot_summary(snapshot)
                for snapshot in solver_snapshots],
        elapsed_seconds=time.monotonic() - started)
    if len(captured) != 1:
        # Legacy route-runner refusal reason/fields: the runner does not invent a
        # Phase A terminal. The wrapper alone reads the *same-pass* graph
        # checkpoint emitted just above and classifies
        # ``assembled_no_support_capture`` when the component solver assembled.
        # A duplicate/missing capture stays this same legacy refusal.
        raise DiagnosticRefusal(
            "capture_unavailable", "capture",
            "the complete-support sink fired "
            f"{len(captured)} times; the solver must abstain exactly once")
    capture = captured[0]
    if not isinstance(capture, v2.CompleteSupportCaptureV2):
        raise DiagnosticRefusal("capture_type_mismatch", "capture")
    _emit_checkpoint(
        checkpoint, "capture",
        status=getattr(capture, "status", ""),
        row_count=getattr(capture, "row_count", None),
        reason=getattr(getattr(capture, "root", None), "reason", "") or "",
        whole_universe_identity_sha256=getattr(
            getattr(capture, "root", None),
            "whole_universe_identity_sha256", ""),
        elapsed_seconds=time.monotonic() - started)
    return ChainRun(entry=entry, lattice=lattice, signatures=signatures,
                    graph=graph, trace=trace,
                    signature_witnesses=tuple(signature_witnesses),
                    capture=capture, elapsed_seconds=time.monotonic() - started,
                    calls=tuple(calls), pin_hashes=tuple(pin_hashes),
                    solver_snapshots=tuple(solver_snapshots))


# ---------------------------------------------------------------------------
# Replay validation and the identity gate
# ---------------------------------------------------------------------------

def index_rows(capture):
    """The complete admitted universe in persisted page order."""
    return tuple(row for page in capture.index_pages for row in page.rows)


def replay_capture(capture, *, expected_row_count=None):
    """Strict fail-closed replay of one capture; returns the canonical record.

    This is the module's own validator (which recomputes every page digest, the
    admitted order, the identity digest section and the whole-universe digest)
    plus the runner's exact-row-count gate. It raises ``DiagnosticRefusal`` so a
    caller can only write after this returns.
    """
    from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2

    expected = (EXPECTED_ROW_COUNT if expected_row_count is None
                else expected_row_count)
    if not isinstance(capture, v2.CompleteSupportCaptureV2):
        raise DiagnosticRefusal("capture_type_mismatch", "capture")
    canonical = v2._canonical_bytes(capture.to_record())
    document = json.loads(canonical.decode("utf-8"))
    if document.get("schema") != v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA:
        raise DiagnosticRefusal("replay_mismatch", "schema")
    if capture.status != "complete":
        raise DiagnosticRefusal(capture.root.reason or "capture_refused",
                                capture.root.field or "record")
    if type(expected) is int and capture.row_count != expected:
        raise DiagnosticRefusal("row_count_mismatch", "row_count",
                                f"expected {expected} admitted rows, "
                                f"captured {capture.row_count}")
    try:
        v2.validate_complete_support_capture(capture)
    except (ValueError, TypeError, AttributeError, KeyError,
            IndexError) as error:
        raise DiagnosticRefusal("replay_mismatch", "capture",
                                type(error).__name__) from error
    rows = index_rows(capture)
    recomputed = v2.complete_support_universe_digest(rows)
    if capture.root.whole_universe_identity_sha256 != recomputed:
        raise DiagnosticRefusal("universe_identity_mismatch",
                                "whole_universe_identity_sha256")
    if v2._canonical_bytes(document) != canonical:
        raise DiagnosticRefusal("replay_mismatch", "canonical_bytes")
    return document


# ---------------------------------------------------------------------------
# Bounded V2 record prescan and sampling-section replay
# ---------------------------------------------------------------------------

# The support sidecar arrays keep their exact V1 contract dtype. These caps
# mirror the accepted support capture profile; they are checked on the plain
# parsed document before any NumPy allocation.
SUPPORT_CONTRACT_DTYPE = "<i8"
SUPPORT_CONTRACT_ITEM_BYTES = 8
MAX_SUPPORT_ELEMENTS = 6_000_000
MAX_SUPPORT_ARRAY_BYTES = 48_000_000


def _shape_elements(shape):
    if (type(shape) is not list or not shape
            or any(type(item) is not int or item < 0 for item in shape)):
        raise DiagnosticRefusal("record_malformed", "payload_array.shape")
    product = 1
    for item in shape:
        product *= item
    return product


def _validate_nested_values(values, shape, *, allow_border_sentinel):
    """Validate nested ``values`` against ``shape`` before NumPy allocation.

    Every level must be a JSON list whose length equals the declared dimension,
    every leaf must be an exact JSON integer (never ``bool``/float/str) and the
    documented ``-1`` border sentinel is allowed only where the field contract
    allows it. Counting aborts the instant the element budget is exceeded, so a
    malformed or over-cap nesting is refused before ``np.asarray``.
    """
    v2 = _v2_module()
    if len(shape) > 64:
        raise DiagnosticRefusal("record_malformed", "payload_array.shape")
    last = len(shape) - 1
    counted = 0
    stack = [(values, 0)]
    while stack:
        node, depth = stack.pop()
        if type(node) is not list or len(node) != shape[depth]:
            raise DiagnosticRefusal("replay_mismatch", "payload_array.values")
        if depth == last:
            for item in node:
                if type(item) is not int:
                    raise DiagnosticRefusal("record_malformed",
                                            "payload_array.values")
                if item < (-1 if allow_border_sentinel else 0):
                    raise DiagnosticRefusal("record_malformed",
                                            "payload_array.values")
                if item > v2.PAYLOAD_MAX_ABS_ELEMENT:
                    raise DiagnosticRefusal("record_malformed",
                                            "payload_array.values")
                counted += 1
                if counted > MAX_SUPPORT_ELEMENTS:
                    raise DiagnosticRefusal("payload_over_cap",
                                            "payload_array.values")
        else:
            for item in node:
                stack.append((item, depth + 1))
    return counted


def _v2_module():
    from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2
    return v2


def _prescan_support_arrays(capture):
    """Bound every persisted support array on the plain parsed document."""
    sidecar_pages = capture.get("sidecar_pages")
    if type(sidecar_pages) is not list:
        raise DiagnosticRefusal("record_malformed", "capture.sidecar_pages")
    seen: set[int] = set()
    total_elements = 0
    entry_count = 0
    field_sentinels = (("logical_pairs", False), ("warp_source_rows", True),
                       ("warp_source_columns", True))
    for page in sidecar_pages:
        if type(page) is not dict:
            raise DiagnosticRefusal("record_malformed", "capture.sidecar_pages")
        entries = page.get("entries")
        if type(entries) is not list:
            raise DiagnosticRefusal("record_malformed", "sidecar.entries")
        for entry in entries:
            if type(entry) is not dict:
                raise DiagnosticRefusal("record_malformed", "sidecar.entry")
            ordinal = entry.get("admitted_row_index")
            if type(ordinal) is not int or ordinal < 0 or ordinal in seen:
                raise DiagnosticRefusal("record_malformed",
                                        "sidecar.admitted_row_index")
            seen.add(ordinal)
            payload = entry.get("payload")
            if type(payload) is not dict:
                raise DiagnosticRefusal("record_malformed", "sidecar.payload")
            for name, sentinel_allowed in field_sentinels:
                array = payload.get(name)
                if type(array) is not dict or array.get("dtype") != SUPPORT_CONTRACT_DTYPE:
                    raise DiagnosticRefusal("payload_dtype_invalid",
                                            f"payload.{name}.dtype")
                elements = _shape_elements(array.get("shape"))
                total_elements += elements
                if total_elements > MAX_SUPPORT_ELEMENTS:
                    raise DiagnosticRefusal("payload_over_cap", "payload_elements")
                if elements * SUPPORT_CONTRACT_ITEM_BYTES > MAX_SUPPORT_ARRAY_BYTES:
                    raise DiagnosticRefusal("payload_over_cap", f"payload.{name}")
                counted = _validate_nested_values(
                    array.get("values"), array["shape"],
                    allow_border_sentinel=sentinel_allowed)
                if counted != elements:
                    raise DiagnosticRefusal("replay_mismatch",
                                            f"payload.{name}.values")
            entry_count += 1
    return entry_count


def _prescan_record_document(document):
    """Prescan the plain record before any typed rebuild or NumPy allocation."""
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    if type(document) is not dict:
        raise DiagnosticRefusal("record_malformed", "record")
    capture = document.get("capture")
    if type(capture) is not dict:
        raise DiagnosticRefusal("record_malformed", "capture")
    _prescan_support_arrays(capture)
    section = document.get("sampling_persistence")
    if section is not None:
        try:
            persistence.validate_section_plain(section)
        except persistence.SamplingPersistenceError as error:
            raise DiagnosticRefusal(error.reason, error.field,
                                    error.detail) from error


def _replay_sampling_section(document, capture):
    """Re-validate the persisted sampling section from a re-parsed record."""
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    schema = document.get("schema")
    section = document.get("sampling_persistence")
    if schema == LEGACY_RECORD_SCHEMA:
        if section is not None:
            raise DiagnosticRefusal("record_malformed", "sampling_persistence")
        return None
    if schema != RECORD_SCHEMA or section is None:
        raise DiagnosticRefusal("record_malformed", "sampling_persistence")
    try:
        persistence.cross_bind_witness_summaries(
            section, document.get("signature_witnesses"))
        ingress = (None if capture is None
                   else {pin.frame_id: pin for pin in capture.root.ingress})
        # With a typed capture the persisted record's own root pins are supplied,
        # so every bound witness must be pinned to exactly one of them
        # (``require_ingress=True``): a bound frame with no capture-root pin is a
        # typed refusal, not a silently unbound outcome. Only the post-write
        # structural replay has no typed pins to offer, and it never computes
        # sampling availability; the preflight owns that decision and always
        # demands root-pinned ingress.
        return persistence.decode_section(
            section, ingress_by_frame=ingress,
            require_ingress=capture is not None)
    except persistence.SamplingPersistenceError as error:
        raise DiagnosticRefusal(error.reason, error.field, error.detail) from error


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _ingress_records(run):
    records = []
    for frame in getattr(run.entry.session, "frames", ()):
        pin = getattr(frame, "ingress", None)
        if pin is not None:
            records.append(pin)
    return tuple(records)


def _ingress_bindings(run, resolved):
    """Bind the ordered input paths to the entry's own ingress pins positionally.

    The only source identity this runner records is each pin's
    ``byte_sha256``, computed by the entry from the exact encoded byte buffer it
    decoded. The binding is positional: the pin produced from the first supplied
    path must carry that path's source label, the second must carry the second
    path's label, and so on. The two ordered tuples are compared element by
    element, so a permuted pin order refuses rather than being accepted as the
    same set. A changed, reordered, duplicated or mismatched source binding
    therefore refuses instead of recording a foreign or reordered identity.
    """
    pins = _ingress_records(run)
    if len(pins) != len(resolved):
        raise DiagnosticRefusal(
            "ingress_binding_mismatch", "ingress",
            f"the entry produced {len(pins)} ingress pins for "
            f"{len(resolved)} inputs")
    frame_ids = [pin.frame_id for pin in pins]
    if len(set(frame_ids)) != len(frame_ids):
        raise DiagnosticRefusal("ingress_binding_mismatch", "frame_id")
    labels = tuple(str(Path(pin.source_label).resolve()) for pin in pins)
    expected = tuple(str(Path(path).resolve()) for path in resolved)
    if labels != expected:
        raise DiagnosticRefusal(
            "ingress_binding_mismatch", "source_label",
            "the entry ingress frames do not match the supplied input paths "
            "in contract order")
    return pins


def _ingress_payload_rows(pins):
    """Canonical serialized ``ingress`` records and positional ``inputs`` rows.

    ``inputs[i]`` is derived from ``ingress[i]`` alone, so the two serialized
    fields can never drift from the ordered pins they name.
    """
    ingress = tuple(pin.to_record() for pin in pins)
    inputs = tuple({"frame_id": pin.frame_id, "path": pin.source_label,
                    "sha256": pin.byte_sha256} for pin in pins)
    return ingress, inputs


def _input_identities(ingress):
    """The recorded input identity, in frozen contract order.

    Positional binding in ``_ingress_bindings`` guarantees the pin order is
    exactly the supplied contract order, so recording the pins verbatim records
    the inputs in that order rather than whichever order the entry returns.
    """
    return _ingress_payload_rows(ingress)[1]


def _cross_bind_ingress(*, entry_pins, root_pins, ingress, inputs, field):
    """Require all four ingress views to agree positionally.

    The entry's own session pins, the complete capture's ``root.ingress``, the
    serialized top-level ``ingress`` records and the positional ``inputs`` rows
    must be the exact same ordered pins. A foreign or permuted capture-root
    ingress therefore cannot be paired with the entry's correct outer rows, and
    tampering with either serialized field is refused.
    """
    entry_records = _ingress_payload_rows(entry_pins)[0]
    root_records, root_inputs = _ingress_payload_rows(root_pins)
    if entry_records != root_records:
        raise DiagnosticRefusal(
            "ingress_binding_mismatch", f"{field}.capture_root",
            "the capture-root ingress is not the entry's ordered pins")
    if not isinstance(ingress, (list, tuple)) or tuple(ingress) != root_records:
        raise DiagnosticRefusal(
            "ingress_binding_mismatch", f"{field}.ingress",
            "the serialized ingress is not the ordered capture-root pins")
    if not isinstance(inputs, (list, tuple)) or tuple(inputs) != root_inputs:
        raise DiagnosticRefusal(
            "ingress_binding_mismatch", f"{field}.inputs",
            "the serialized inputs are not the positional capture-root pins")


def _stage_digests(run):
    return {
        "entry_sha256": getattr(run.entry, "entry_sha256", ""),
        "session_sha256": getattr(getattr(run.entry, "session", None),
                                 "session_sha256", ""),
        "lattice_sha256": getattr(run.lattice, "result_sha256", ""),
        "signatures_sha256": getattr(run.signatures, "result_sha256", ""),
        "graph_sha256": getattr(run.graph, "result_sha256", ""),
        "trace_sha256": getattr(run.trace, "attribution_sha256", ""),
    }


def _record_payload(run, *, ingress, settings, canonical_capture):
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        SourceTapWitnessBindingV1)
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    witnesses = []
    for item in run.signature_witnesses:
        if isinstance(item, SourceTapWitnessBindingV1):
            witness = dict(item.to_record())
            witness["outcome"] = "bound"
        else:
            witness = dataclasses.asdict(item)
            witness["outcome"] = "refused"
        witnesses.append(witness)
    section = persistence.persist_signature_witnesses(run.signature_witnesses)
    return {
        "schema": RECORD_SCHEMA,
        "status": "complete",
        "settings": dict(settings),
        "inputs": [dict(row) for row in _input_identities(ingress)],
        "stage_digests": _stage_digests(run),
        "signature_witness_count": len(run.signature_witnesses),
        "signature_witnesses": witnesses,
        "sampling_persistence": section,
        "ingress": [pin.to_record() for pin in ingress],
        "expected_row_count": EXPECTED_ROW_COUNT,
        "whole_universe_identity_sha256":
            run.capture.root.whole_universe_identity_sha256,
        "input_match_identity_sha256": run.capture.root.input_match_identity_sha256,
        "failed_assignment_sha256": run.capture.root.failed_assignment_sha256,
        "record_sha256": run.capture.root.record_sha256,
        "capture": canonical_capture,
        "note": ("One-shot internal Route B diagnostic over ordinary image-only "
                 "inputs; no externally supplied annotation or alignment source, "
                 "and no second signature, matcher, filter or solver pass."),
    }


def _record_bytes(payload) -> bytes:
    return _canonical_bytes(payload) + b"\n"


def _record_declared_size(payload) -> int:
    """Exact serialized record size (bytes plus the trailing newline).

    Computed from the plain structure before any serialization, so the record cap
    is applied without first materializing an over-cap JSON string.
    """
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    return persistence.canonical_json_size(payload) + 1


def _fit_sampling_section(payload):
    """Keep the complete record inside its byte cap by degrading the section.

    The support capture and every disposition are unchanged; only the additive
    sampling section becomes an explicit typed refusal, never a truncation. The
    cap is applied to the *declared* size before any serialization.
    """
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    if _record_declared_size(payload) <= MAX_RECORD_BYTES:
        return payload
    capped = dict(payload)
    capped["sampling_persistence"] = persistence.refused_section_record(
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP,
        field="record", detail=str(_record_declared_size(payload)))
    if _record_declared_size(capped) > MAX_RECORD_BYTES:
        raise DiagnosticRefusal("record_over_cap", "record")
    return capped


def _write_record(root, payload, canonical_capture, *,
                  ownership: "InvocationOwnership | None" = None,
                  output_budget=None):
    root = Path(root)
    artifact = root / RECORD_NAME
    # Fit the complete record to its byte cap *before* the artifact digest is
    # serialized. The declared-size probe below carries a placeholder 64-hex
    # digest, so the exact final serialized size is known (the real digest is
    # always 64 lowercase hex characters) without ever building the JSON string;
    # a record that still cannot fit is rejected here.
    payload = _fit_sampling_section(payload)
    body = dict(payload)
    body["artifact_sha256"] = "0" * 64
    if _record_declared_size(body) > MAX_RECORD_BYTES:
        raise DiagnosticRefusal("record_over_cap", "record")
    body["artifact_sha256"] = _digest({key: value for key, value in body.items()
                                       if key != "artifact_sha256"})
    # No-clobber publication: ``_atomic_publish`` fails rather than replacing a
    # target that appeared after the root pre-check, and registers the published
    # record as invocation-owned.
    _atomic_publish(artifact, body, field="record", ownership=ownership,
                    output_budget=output_budget)
    # Post-write replay: the file must re-parse to the exact canonical capture and
    # (for a V2 record) re-decode its sampling section from the bounded snapshot.
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    try:
        snapshot = persistence.read_bounded_record_bytes(
            artifact, reader=(lambda _: ownership.read_bytes(artifact, MAX_RECORD_BYTES))
            if ownership is not None and ownership.native else None)
        replayed = json.loads(snapshot.data.decode("utf-8"))
        if (replayed.get("capture") != canonical_capture
                or _digest({key: value for key, value in replayed.items()
                            if key != "artifact_sha256"})
                != replayed.get("artifact_sha256")):
            raise DiagnosticRefusal("replay_mismatch", "artifact")
        # Ownership-aware rollback: this call created ``artifact`` and registered
        # it, so *every* post-publication verification failure - including the
        # second sampling replay - must remove exactly this artifact before the
        # error propagates.
        _replay_sampling_section(replayed, capture=None)
    except BaseException as error:
        if ownership is not None:
            ownership.unlink(artifact)
        else:
            _unlink_if_present(artifact)
        if isinstance(error, DiagnosticRefusal):
            raise
        if isinstance(error, Exception):
            raise DiagnosticRefusal("replay_mismatch", "artifact",
                                    type(error).__name__) from error
        raise
    return artifact


def _refusal_record(reason, field, detail, *, resolved, settings):
    return {
        "schema": REFUSAL_SCHEMA,
        "status": "refused",
        "reason": reason,
        "field": field,
        "detail": detail,
        "settings": dict(settings or {}),
        "inputs": [{"frame_id": "", "path": str(path), "sha256": ""}
                   for path in (resolved or ())],
    }


def _prepare_root(root: Path) -> tuple[Path, Path]:
    """Validate a fresh output root and create its fresh staging child.

    Every existing component from the absolute root through its filesystem
    anchor is probed for symlink/reparse metadata first (fail-closed), then
    **every** artifact name any Phase A/route-B invocation can own is refused
    when already present (no-clobber), then a non-empty existing root is refused,
    and only then is the root and its exact empty staging child created.

    Zero writes happen on a collision or reparse refusal: the collision checks
    precede any ``mkdir``, and a root directory this call created but could not
    finish validating is removed again when still empty. Returns the normalized
    ``(root, staging)`` pair.
    """
    root = Path(os.path.abspath(root))
    _refuse_reparse_components(root)
    for name in OWNED_ARTIFACT_NAMES:
        if os.path.lexists(root / name):
            raise DiagnosticRefusal("output_exists", "root")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise DiagnosticRefusal("root_not_fresh", "root")
    created_root = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    # Re-probe after creation: a racing replacement must not redirect publication.
    try:
        _refuse_reparse_components(root)
    except BaseException:
        if created_root:
            _rmdir_if_empty(root)
        raise
    staging = root / STAGING_DIR_NAME
    if os.path.lexists(staging) or _reparse_tag(staging):
        if created_root:
            _rmdir_if_empty(root)
        raise DiagnosticRefusal("root_not_fresh", "staging")
    try:
        staging.mkdir()
    except BaseException:
        if created_root:
            _rmdir_if_empty(root)
        raise
    return root, staging


def _publish_staged_file(staged: Path, target: Path, *, field: str,
                         max_bytes=None,
                         ownership: "InvocationOwnership | None" = None) -> None:
    """Link one validated staged file into place without clobbering a target.

    ``max_bytes`` refuses an over-cap staged artifact before it is linked into
    place; the byte size is measured on the staged file, so an over-cap project
    never becomes a published output. On success the target is registered as
    invocation-owned.
    """
    if ownership is not None and ownership.native:
        staged, target = Path(os.path.abspath(staged)), Path(os.path.abspath(target))
        held = ownership._handles.get(staged)
        if held is None:
            raise DiagnosticRefusal("ownership_birth_unproven", field)
        if type(max_bytes) is int and held.size > max_bytes:
            raise DiagnosticRefusal(f"{field}_over_cap", field)
        try:
            held.rename_no_replace(target)
        except OSError as error:
            reason = "output_exists" if isinstance(error, FileExistsError) or getattr(error, "winerror", None) in (80, 183) else "output_failed"
            raise DiagnosticRefusal(reason, field, str(error)) from error
        ownership.take_handle(staged)
        ownership.register_handle(target, held)
        return
    staged = Path(staged)
    target = Path(target)
    if not staged.is_file():
        raise DiagnosticRefusal("entry_output_missing", field)
    if type(max_bytes) is int and staged.stat().st_size > max_bytes:
        raise DiagnosticRefusal(f"{field}_over_cap", field,
                                f"{staged.stat().st_size} > {max_bytes} bytes")
    if os.path.lexists(target):
        raise DiagnosticRefusal("output_exists", field)
    try:
        os.link(staged, target)
    except FileExistsError as error:
        raise DiagnosticRefusal("output_exists", field) from error
    except OSError as error:
        raise DiagnosticRefusal("output_failed", field, str(error)) from error
    if ownership is not None:
        ownership.register(target)
    _unlink_if_present(staged)


def _discard_staging(staging: Path, staged_project: Path, *, ownership=None) -> None:
    """Remove only the exact runner-owned staged files, then the empty staging dir.

    Cleanup never walks a tree and never uses recursive deletion: it unlinks the
    specifically named staged project (plus the store's own exact ``.tmp``
    sibling) and then removes the exact staging directory only when ``rmdir``
    finds it already empty.
    """
    if ownership is not None and ownership.native:
        ownership.unlink(staged_project)
        # Unknown store temporaries are never adopted/deleted by pathname.
        ownership.release_staging_anchor(staging)
        _rmdir_if_empty(staging)
        return
    _unlink_if_present(staged_project)
    _unlink_if_present(Path(str(staged_project) + ".tmp"))
    _rmdir_if_empty(staging)


def _write_refusal(root, refusal, resolved, settings, *,
                   ownership: "InvocationOwnership | None" = None) -> None:
    """Best-effort typed refusal record; it can never mask the real refusal.

    The declared canonical size is bounded by ``MAX_REFUSAL_BYTES`` before any
    bytes are written, and a root that was itself a collision/reparse never
    reaches this function (``_prepare_root`` refuses with zero writes), so a
    failure diagnostic is never written into a foreign or redirected root.
    """
    try:
        payload = _refusal_record(refusal.reason, refusal.field, refusal.detail,
                                  resolved=resolved, settings=settings)
        if persistence_canonical_size(payload) + 1 > MAX_REFUSAL_BYTES:
            return
        _atomic_publish(Path(root) / REFUSAL_NAME, payload, field="refusal",
                        ownership=ownership)
    except (DiagnosticRefusal, OSError):
        pass


def persistence_canonical_size(payload) -> int:
    """Exact canonical JSON size of one plain refusal document (bytes)."""
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence
    return persistence.canonical_json_size(payload)


def execute_route_b_diagnostic(
    *,
    image_paths,
    root,
    api=None,
    expected_pins=None,
    checkpoint_sink=None,
    observer=None,
    project_max_bytes=None,
    output_budget=None,
    invocation_ownership=None,
) -> dict:
    """Capture, replay-validate and atomically record one Route B diagnostic.

    The interface is ordinary screenshot paths plus one output root. Source
    identity is bound solely to the entry's own
    ``FrameIngressProvenanceV1.byte_sha256`` pins (from the exact byte buffer the
    entry decoded); the runner never pre-reads or pre-hashes an input path. The
    graph settings are the frozen production defaults, and there is no
    caller-supplied identity pin: the capture's sealed/replayed
    ``whole_universe_identity_sha256`` plus the exact 1,928 count are the content
    identity.

    The entry's schema-2 project is staged in a fresh child of the validated
    output root and published only after replay validation. A precondition or
    capture refusal raises ``DiagnosticRefusal``, writes a typed refusal record
    when the root is usable, and leaves no complete capture, staged
    ``route_b.mcscan``, staging directory or ``.partial`` artifact behind.

    The four optional Phase A seams are inert for the legacy caller:
    ``expected_pins`` enables the ordered same-pass T2 pin gate before lattice
    and signatures; ``checkpoint_sink`` receives one bounded record per stage
    boundary (the journal that preserves evidence on an early refusal);
    ``observer`` is a list that collects the same-pass solver-failure snapshots
    (it never triggers a second matcher/solver pass); and ``project_max_bytes``
    refuses an over-cap staged project before it is published. The returned
    ``terminal`` is always one of the accepted Phase A terminal states.
    """
    root = Path(root)
    resolved = resolve_images(image_paths)
    effective_settings = resolve_settings()
    root, staging = _prepare_root(root)
    staged_project = staging / PROJECT_NAME
    published_project = root / PROJECT_NAME
    ownership = invocation_ownership or InvocationOwnership()
    ownership.anchor(root)
    refusal: DiagnosticRefusal | None = None
    failed = False
    try:
        run = _run_single_pass(resolved=resolved, project_path=staged_project,
                               settings=effective_settings,
                               api=_resolve_api(api),
                               expected_pins=expected_pins,
                               observer=observer,
                                checkpoint=checkpoint_sink,
                                project_max_bytes=project_max_bytes, project_ownership=ownership)
        ingress = _ingress_bindings(run, resolved)
        canonical_capture = replay_capture(run.capture)
        # Cross-bind the entry's ordered pins, the capture's own root ingress,
        # the serialized top-level ingress and the positional input rows before
        # anything is published.
        payload = _record_payload(run, ingress=ingress,
                                  settings=effective_settings,
                                  canonical_capture=canonical_capture)
        payload = _fit_sampling_section(payload)
        _cross_bind_ingress(entry_pins=ingress,
                            root_pins=tuple(run.capture.root.ingress),
                            ingress=payload["ingress"], inputs=payload["inputs"],
                            field="record")
        # Validate the persisted sampling section and its ordered cross-bind
        # against the typed ingress pins before anything is published.
        _replay_sampling_section(payload, run.capture)
        # Publish only after the whole chain and its replay gate succeeded.
        _publish_staged_file(staged_project, published_project, field="project",
                             max_bytes=project_max_bytes, ownership=ownership)
        artifact = _write_record(root, payload, canonical_capture,
                                 ownership=ownership, output_budget=output_budget)
        _emit_checkpoint(checkpoint_sink, "record",
                         record=str(artifact),
                         record_bytes=artifact.stat().st_size,
                         record_file_sha256=_file_sha256(artifact, ownership=ownership))
    except DiagnosticRefusal as error:
        failed = True
        refusal = error
        raise
    except BaseException:
        failed = True
        raise
    finally:
        # On failure, roll back exactly the artifacts this call created and
        # registered; never touch a target the call did not create.
        if failed:
            ownership.rollback()
        _discard_staging(staging, staged_project, ownership=ownership)
        if refusal is not None:
            _write_refusal(root, refusal, resolved, effective_settings,
                           ownership=ownership)
    progress = _progress_module()
    return {
        "schema": SCHEMA,
        "status": "complete",
        "terminal": progress.TERMINAL_CAPTURED_SOLVER_ABSTENTION,
        "root": str(root),
        "record": str(artifact),
        "record_file_sha256": _file_sha256(artifact, ownership=ownership),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "row_count": run.capture.row_count,
        "whole_universe_identity_sha256":
            run.capture.root.whole_universe_identity_sha256,
        "input_match_identity_sha256": run.capture.root.input_match_identity_sha256,
        "failed_assignment_sha256": run.capture.root.failed_assignment_sha256,
        "stage_digests": _stage_digests(run),
        "signature_witness_count": len(run.signature_witnesses),
        "sampling_persistence": _sampling_section_summary(payload),
        "record_schema": RECORD_SCHEMA,
        "calls": list(run.calls),
        "pin_profile": list(run.pin_hashes),
        "solver_diagnostics": [
            _solver_snapshot_summary(snapshot)
            for snapshot in run.solver_snapshots],
        "elapsed_seconds": run.elapsed_seconds,
    }


def _sampling_section_summary(payload) -> dict:
    section = payload.get("sampling_persistence") or {}
    entries = section.get("entries") or []
    return {
        "status": section.get("status", ""),
        "reason": section.get("reason", ""),
        "entry_count": section.get("entry_count", 0),
        "bound_count": sum(1 for entry in entries
                           if entry.get("outcome") == "bound"),
        "persistence_refused_count": sum(
            1 for entry in entries
            if entry.get("outcome") == "persistence_refused"),
        "payload_bytes": section.get("payload_bytes", 0),
        "metadata_bytes": section.get("metadata_bytes", 0),
    }


def validate_record(record_path, *, snapshot_bytes=None) -> dict:
    """Re-parse and fully replay one written record; refuse on any mismatch.

    The canonical capture is rebuilt into live V2 objects from its own record,
    then passed through the same strict ``replay_capture`` used before writing,
    so a tampered on-disk record cannot pass on a stale digest.
    """
    from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    path = Path(record_path)
    if not path.is_file():
        raise DiagnosticRefusal("record_missing", "record")
    # One bounded, stable, non-link snapshot; the depth/token scan runs on the
    # raw bytes before ``json.loads`` and the array prescan runs before any
    # typed rebuild or NumPy allocation.
    try:
        snapshot = persistence.read_bounded_record_bytes(path, reader=(lambda _: snapshot_bytes) if snapshot_bytes is not None else None)
    except persistence.SamplingPersistenceError as error:
        raise DiagnosticRefusal(error.reason, error.field,
                                error.detail) from error
    try:
        document = json.loads(snapshot.data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DiagnosticRefusal("record_malformed", "record",
                                type(error).__name__) from error
    if (type(document) is not dict
            or document.get("schema") not in RECORD_SCHEMAS
            or document.get("status") != "complete"):
        raise DiagnosticRefusal("record_malformed", "record")
    # Enforce the canonical serialized record cap and run every structural
    # prescan *before* the artifact digest serializes the complete document. A
    # bounded raw read can still expand canonically (``ensure_ascii=True`` escapes
    # each non-ASCII code point as ``\uXXXX``), so only the exact declared-size
    # walk can refuse an over-cap record without ever allocating the over-cap JSON
    # string; the prescans then run on a document already known to fit the cap.
    try:
        declared = persistence.canonical_json_size(document)
    except persistence.SamplingPersistenceError as error:
        raise DiagnosticRefusal(error.reason, error.field, error.detail) from error
    if declared + 1 > MAX_RECORD_BYTES:
        raise DiagnosticRefusal(
            persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, "record")
    _prescan_record_document(document)
    claimed = document.get("artifact_sha256")
    recomputed = _digest({key: value for key, value in document.items()
                          if key != "artifact_sha256"})
    if claimed != recomputed:
        raise DiagnosticRefusal("replay_mismatch", "artifact_sha256")
    capture_record = document.get("capture")
    if not isinstance(capture_record, dict):
        raise DiagnosticRefusal("record_malformed", "capture")
    try:
        capture = _capture_from_record(v2, capture_record)
    except (TypeError, ValueError, KeyError, IndexError,
            AttributeError) as error:
        raise DiagnosticRefusal("record_malformed", "capture",
                                type(error).__name__) from error
    replay_capture(capture)
    # Cross-bind the persisted capture root's ordered ingress to the serialized
    # top-level ingress and positional input rows; a foreign or permuted root
    # ingress, or a tampered outer row (even with a recomputed artifact digest),
    # refuses.
    _cross_bind_ingress(entry_pins=tuple(capture.root.ingress),
                        root_pins=tuple(capture.root.ingress),
                        ingress=document.get("ingress"),
                        inputs=document.get("inputs"), field="record")
    # Re-decode the sampling section against the typed ingress pins and
    # cross-bind it to the ordered witness summary; a V1 record stays legacy
    # with no section at all.
    _replay_sampling_section(document, capture)
    if (capture.root.whole_universe_identity_sha256
            != document.get("whole_universe_identity_sha256")
            or capture.root.input_match_identity_sha256
            != document.get("input_match_identity_sha256")
            or capture.root.failed_assignment_sha256
            != document.get("failed_assignment_sha256")
            or capture.root.record_sha256 != document.get("record_sha256")
            or capture.root.row_count != document.get("expected_row_count")):
        raise DiagnosticRefusal("replay_mismatch", "record_binding")
    return document


# --- Record -> live capture rebuilding (fail-closed) -------------------------

_SCALAR_SEQUENCES = {
    "decoded_shape": tuple, "source_shape": tuple, "target_shape": tuple,
}


def _optional_tuple(value):
    return None if value is None else tuple(value)


def _init_fields(cls) -> frozenset:
    return frozenset(field.name for field in dataclasses.fields(cls) if field.init)


def _init_kwargs(cls, record, *, transform=(), rebuild=()):
    """Build constructor kwargs from a record, keeping only init fields.

    Canonical records carry derived, ``init=False`` fields (page digests, counts,
    aggregate properties), so every reconstruction filters to the class's own
    initializable fields and only supplies the nested objects explicitly.
    """
    if not isinstance(record, dict):
        raise DiagnosticRefusal("record_malformed", "capture")
    allowed = _init_fields(cls)
    kwargs = {}
    for name, value in record.items():
        if name not in allowed:
            continue
        if name in rebuild:
            kwargs[name] = rebuild[name]
        elif name in transform:
            kwargs[name] = transform[name](value)
        elif name in _SCALAR_SEQUENCES:
            kwargs[name] = _SCALAR_SEQUENCES[name](value)
        else:
            kwargs[name] = value
    return kwargs


def _capture_from_record(v2, capture_record):
    """Rebuild a live ``CompleteSupportCaptureV2`` from its canonical record."""
    if capture_record.get("schema") != v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA:
        raise DiagnosticRefusal("record_malformed", "capture.schema")
    index_pages = tuple(
        v2.CandidateSupportIndexPageV2(**_init_kwargs(
            v2.CandidateSupportIndexPageV2, row, rebuild={"rows": tuple(
                v2.CandidateSupportIndexRowV2(**_init_kwargs(
                    v2.CandidateSupportIndexRowV2, item,
                    transform={"cell_shift_rc": _optional_tuple}))
                for item in row["rows"])}))
        for row in capture_record.get("index_pages", ()))
    digest_pages = tuple(
        v2.CandidateSupportDigestPageV2(**_init_kwargs(
            v2.CandidateSupportDigestPageV2, row,
            transform={"digests": tuple}))
        for row in capture_record.get("digest_pages", ()))
    sidecar_pages = tuple(_sidecar_page_from_record(v2, row)
                          for row in capture_record.get("sidecar_pages", ()))
    root = _capture_root_from_record(v2, capture_record["root"])
    return v2.CompleteSupportCaptureV2(
        root=root, index_pages=index_pages, digest_pages=digest_pages,
        sidecar_pages=sidecar_pages, schema=capture_record["schema"])


def _capture_root_from_record(v2, record):
    from mc_photogrammetry.pipeline.frame_ingress import FrameIngressProvenanceV1

    plan_record = record.get("cap_plan")
    plan = (None if plan_record is None
            else v2.CompleteSupportCapPlanV1(**_init_kwargs(
                v2.CompleteSupportCapPlanV1, plan_record)))
    sections = tuple(v2.CandidateSupportDigestSectionV2(**_init_kwargs(
        v2.CandidateSupportDigestSectionV2, item, rebuild={"pages": tuple(
            v2.CandidateSupportDigestPageBindingV2(**_init_kwargs(
                v2.CandidateSupportDigestPageBindingV2, binding))
            for binding in item["pages"])}))
        for item in record.get("digest_sections", ()))
    ingress = tuple(FrameIngressProvenanceV1.from_record(item)
                    for item in record.get("ingress", ()))
    # A complete root's canonical payload omits the absent ``reason``/``field``
    # (and a refused root omits ``basis``); both are explicit ``None`` here.
    root_record = {name: record.get(name)
                   for name in ("status", "basis", "reason", "field")}
    root_record.update({name: value for name, value in record.items()
                        if name not in ("status", "basis", "reason", "field")})
    return v2.CandidateSupportCaptureRootV2(**_init_kwargs(
        v2.CandidateSupportCaptureRootV2, root_record, rebuild={
            "index_pages": tuple(v2.CandidateSupportIndexPageBindingV2(
                **_init_kwargs(v2.CandidateSupportIndexPageBindingV2, item))
                for item in record["index_pages"]),
            "sidecar_pages": tuple(v2.CandidateSupportSidecarPageBindingV2(
                **_init_kwargs(v2.CandidateSupportSidecarPageBindingV2, item))
                for item in record["sidecar_pages"]),
            "digest_sections": sections,
            "ingress": ingress,
            "cap_plan": plan,
        }))


def _sidecar_page_from_record(v2, record):
    entries = tuple(
        v2.MatcherSupportSidecarEntryV2(**_init_kwargs(
            v2.MatcherSupportSidecarEntryV2, item,
            rebuild={"payload": _payload_from_record(v2, item["payload"])}))
        for item in record.get("entries", ()))
    return v2.MatcherSupportSidecarPageV2(**_init_kwargs(
        v2.MatcherSupportSidecarPageV2, record, rebuild={"entries": entries}))


def _payload_from_record(v2, record):
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        MatcherSupportPayloadV1)
    return MatcherSupportPayloadV1(**_init_kwargs(
        MatcherSupportPayloadV1, record, rebuild={
            "logical_pairs": _array_from_record(
                record["logical_pairs"], allow_border_sentinel=False),
            "warp_source_rows": _array_from_record(
                record["warp_source_rows"], allow_border_sentinel=True),
            "warp_source_columns": _array_from_record(
                record["warp_source_columns"], allow_border_sentinel=True),
        }))


def _array_from_record(record, *, allow_border_sentinel):
    import numpy as np

    if not isinstance(record, dict):
        raise DiagnosticRefusal("record_malformed", "payload_array")
    values = record.get("values")
    dtype = record.get("dtype")
    shape = record.get("shape")
    # Exact contract dtype first: an unvalidated dtype string must never reach
    # ``np.dtype``/``np.asarray``.
    if dtype != SUPPORT_CONTRACT_DTYPE:
        raise DiagnosticRefusal("payload_dtype_invalid", "payload_array.dtype")
    if not isinstance(values, list) or not isinstance(shape, list):
        raise DiagnosticRefusal("record_malformed", "payload_array")
    elements = _shape_elements(shape)
    if elements > MAX_SUPPORT_ELEMENTS \
            or elements * SUPPORT_CONTRACT_ITEM_BYTES > MAX_SUPPORT_ARRAY_BYTES:
        raise DiagnosticRefusal("payload_over_cap", "payload_array")
    # Exact nested row/element counts and exact JSON-integer leaves are verified
    # before NumPy is called, so a small declared shape can never smuggle a much
    # larger ``values`` list past the cap into an allocation.
    counted = _validate_nested_values(
        values, shape, allow_border_sentinel=allow_border_sentinel)
    if counted != elements:
        raise DiagnosticRefusal("replay_mismatch", "payload_array.values")
    array = np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape))
    if hashlib.sha256(array.tobytes(order="C")).hexdigest() != record.get("sha256"):
        raise DiagnosticRefusal("replay_mismatch", "payload_array_sha256")
    array.flags.writeable = False
    return array


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="DS-R4-V2-ROUTEB-RUNNER one-shot internal Route B diagnostic")
    parser.add_argument("--image", action="append", default=None,
                        help="ordinary screenshot path (repeatable, in order)")
    parser.add_argument("--root", default=None,
                        help="fresh output root for the record")
    # The parser deliberately exposes no --sha256, --whole-universe-digest,
    # --settings or --record mode: image digests are derived internally, the
    # graph settings are frozen to production, and the sealed/replayed universe
    # digest plus exact 1,928 count are the only content identity. The record
    # replay validator remains a Python function for tests.
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.image is None or args.root is None:
        print("--image and --root are required to run a capture",
              file=sys.stderr)
        return 2
    try:
        report = execute_route_b_diagnostic(
            image_paths=args.image, root=Path(args.root))
    except DiagnosticRefusal as refusal:
        print(json.dumps({"status": "refused", "reason": refusal.reason,
                          "field": refusal.field, "detail": refusal.detail},
                         sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "record": report["record"],
                      "row_count": report["row_count"],
                      "record_file_sha256": report["record_file_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
