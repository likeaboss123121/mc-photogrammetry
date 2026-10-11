"""Evaluator-only symbolic support for one audited OpenCV CPU contract.

No native code, image sampling, filesystem access, or runtime attestation occurs
here. Native buffer/layout and FP-policy records must come from separately sealed
and verified ingress. This conditional symbolic witness is neither byte-linear
uint8 resampling nor a minimal set of pixels influencing a quantized output.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import struct
from threading import Lock
from traceback import clear_frames

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINTPTR_MAX = (1 << 64) - 1
MAX_ROWS = 262_144
HEADER_BYTES = 192
HEADER_LIMIT = 4096
TERM_BYTES = 40
SCRATCH_BYTES = 16 << 20
PACKED_VERSION = 4
MAGIC = b"OCVCPU4\0"
HEADER = struct.Struct("<8s9I6Q32s32s36s")
TERM = struct.Struct("<32sBBBBI")
BASE = struct.Struct("<iiII")
U32 = struct.Struct("<I")
UINT32_MAX = (1 << 32) - 1
MAX_BUFFER_BYTES = 16 << 20
MAX_EXTRA_BYTES = 256 << 20
FP_POLICY = "binary32-rne; reject-subnormal-coordinates-and-fractions"
SOURCE_PINS = (
    ("modules/imgproc/src/imgwarp.cpp", "b7a95e4bb5c2c5adc5378c43151fa4bc6e461ec3bb267d9d72eda91a004595ae"),
    ("modules/imgproc/src/warp_kernels.simd.hpp", "3e19fdf116c99babd38775f543301276db59ae8bbfa50edaaba7afe89c1a7c73"),
    ("modules/imgproc/src/warp_common.scalar.hpp", "3da7138a38c611aedd4f4805b35f35cde9e5e038107bedce04c4295fe1f35af6"),
    ("modules/imgproc/src/warp_common.vector.hpp", "ca5314ca884189cf790afc4d14ef121f3fd30f7b25bf3054f977cef81160df21"),
)


@dataclass(frozen=True, slots=True)
class KernelIdentity:
    source_commit: str
    artifact_manifest_sha256: str
    extension_sha256: str
    core_sha256: str
    imgproc_sha256: str
    compile_commands_sha256: str
    build_information_sha256: str
    precompile_gates_sha256: str
    source_files: tuple[tuple[str, str], ...]


REFERENCE_KERNEL = KernelIdentity(
    "40738fb16ceddb5fb3fea747585f7ce6abb0605b",
    "46d3b51cd4b3ea2efa2592e47b3204ae079652ef9ac7a6c3cb31d5f38ffe3421",
    "15e8c9f063076e4e6c07bbba053b0ce02fcfb201eff35145ce2395aae686e4e0",
    "2149d6ca8914330ac9d2ebfbbade70af4e9d58f6a427fb8010664fd45fb55159",
    "6a7be1e21436a9f909eed1dfdf12f9277fe472010de51b82108537cbae8d1276",
    "f7b7540b3afff8edea808dee333c0d4d43640aeedfc95a2a2f7e92e047d24280",
    "20e38ff4f7bb09560492f2b4bca416f6d626feb7762b62dfe89abda66f8e7eab",
    "4db1525235cab2261a30a29e09e4fdd19ed9fc4415bbeb0cff19aa33482cb077",
    SOURCE_PINS,
)


@dataclass(frozen=True, slots=True)
class BufferView:
    """Detached bytes spanning native rows, including padding; address is ingress metadata."""
    data: bytes
    shape: tuple[int, int]
    row_stride: int
    address: int
    kind: str
    channels: int = 1


@dataclass(frozen=True, slots=True)
class OutputLayout:
    shape: tuple[int, int]
    row_stride: int
    address: int
    kind: str = "u8"
    channels: int = 1


@dataclass(frozen=True, slots=True)
class SupportRequest:
    kernel: KernelIdentity | None
    source: BufferView
    map_x: BufferView
    map_y: BufferView
    destination: OutputLayout
    fp_policy: str | None = None
    active_dispatch: str | None = None
    resolved_hint: str | None = None
    interpolation: str = "INTER_LINEAR"
    border_type: str = "BORDER_CONSTANT"
    border_value: int = 0
    relative_maps: bool = False
    source_kind: str = "Mat"


@dataclass(frozen=True, slots=True)
class SupportRefusal:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class Dyadic:
    """Reduced exact nonnegative numerator / 2**denominator_shift."""
    numerator: int
    denominator_shift: int

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, 1 << self.denominator_shift)


@dataclass(frozen=True, slots=True)
class BorderTerm:
    row: int
    corner: int
    weight: Dyadic
    value_u8: int


@dataclass(frozen=True, slots=True)
class SymbolicSupport:
    """Immutable schema-4 packed spatial CSR; same-channel dependencies only; no retained per-pixel Python objects."""
    payload: bytes
    request_sha256: str
    planned_extra_bytes: int
    seal_sha256: str


class _AggregateLedger:
    """One immutable authoritative snapshot: (permanent entries, transaction).

    Each entry is (exact witness, charge). Current transaction is
    (token, phase, charge, exact witness or None). One reference replacement
    publishes admission/commit/finalization; no caller handoff flags exist.
    """
    def __init__(self):
        self._operation = Lock()
        self._state = ((), None)

    @property
    def used(self):
        entries, current = self._state
        return sum(amount for _, amount in entries) + (current[2] if current else 0)

    def transaction(self, amount, budget, witness=None):
        return _LedgerTransaction(self, amount, budget, witness)

    def begin(self, token):
        with self._operation:
            entries, current = self._state
            _require(current is None, "ownership_busy", "exclusive ledger transaction required")
            owned = any(w is token.witness and amount == token.amount for w, amount in entries)
            charge = 0 if token.witness is not None and owned else token.amount
            _require(self.used + charge <= token.budget, "resource_cap", "aggregate reservation unavailable")
            phase = "verify" if token.witness is not None else "provisional"
            self._state = (entries, (token, phase, charge, None))

    def retain(self, token, witness):
        with self._operation:
            entries, current = self._state
            _require(current is not None and current[0] is token and current[1] == "provisional"
                     and type(witness) is SymbolicSupport and current[2] == witness.planned_extra_bytes,
                     "invalid_ownership", "commit must match the admitted build transaction")
            # This single replacement both retains bytes and commits their charge.
            self._state = (entries, (token, "committed", current[2], witness))

    def finish(self, token):
        with self._operation:
            entries, current = self._state
            if current is None or current[0] is not token:
                return  # Failed/nested admission cannot cancel another owner.
            if current[1] == "committed":
                entries = entries + ((current[3], current[2]),)
            # Provisional/verify entries disappear only after caller cleanup.
            self._state = (entries, None)


class _LedgerTransaction:
    """Precreated identity survives an exception before __enter__ returns.

    One cleanup attempt; interruption during cleanup leaves authoritative
    ownership/charge fail-closed. This is not arbitrary signal immunity.
    """
    __slots__ = ("ledger", "amount", "budget", "witness")

    def __init__(self, ledger, amount, budget, witness):
        self.ledger, self.amount, self.budget, self.witness = ledger, amount, budget, witness

    def __enter__(self):
        try:
            self.ledger.begin(self)
            return self
        except BaseException as error:
            clear_frames(error.__traceback__)
            self.ledger.finish(self)
            raise

    def __exit__(self, exception_type, error, tb):
        if tb is not None:
            # Worker frames are inactive here; owning API wrapper has no output
            # buffer locals. Cleanup precedes authoritative refund/finalization.
            clear_frames(tb)
        self.ledger.finish(self)
        return False


_LEDGER = _AggregateLedger()


class _Refuse(Exception):
    def __init__(self, code: str, detail: str):
        self.refusal = SupportRefusal(code, detail)


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise _Refuse(code, detail)


def _integer(value: object) -> bool:
    return type(value) is int


def _shape(value: object) -> tuple[int, int]:
    _require(type(value) is tuple and len(value) == 2, "unsupported_layout", "two-dimensional tuple required")
    _require(all(_integer(x) and 0 < x < 32_767 for x in value), "unsupported_layout", "native remap SHRT_MAX shape contract")
    return value


def _layout(shape: object, stride: object, address: object, itemsize: int, alignment: int | None = None) -> int:
    h, w = _shape(shape)
    alignment = itemsize if alignment is None else alignment
    _require(_integer(stride) and w * itemsize <= stride <= INT32_MAX and stride % alignment == 0,
             "unsupported_layout", "positive integral aligned row stride required")
    _require(_integer(address) and 0 < address <= UINTPTR_MAX and address % alignment == 0,
             "unsupported_layout", "aligned nonzero native address required")
    span = (h - 1) * stride + w * itemsize
    _require(span <= MAX_BUFFER_BYTES and address + span <= UINTPTR_MAX,
             "unsafe_address", "span cap or native pointer arithmetic exceeded")
    return span


def _known_kernel(value: object) -> bool:
    if type(value) is not KernelIdentity:
        return False
    for field in fields(value):
        actual = getattr(value, field.name)
        expected = getattr(REFERENCE_KERNEL, field.name)
        if field.name == "source_files":
            if (type(actual) is not tuple or len(actual) != len(SOURCE_PINS)
                    or any(type(pair) is not tuple or len(pair) != 2 or any(type(x) is not str for x in pair) for pair in actual)):
                return False
        elif type(actual) is not str:
            return False
        if actual != expected:
            return False
    return True

def _source_contract(view):
    """(sealed dtype code, channels, pixel bytes, scalar/offset unit bytes).

    Code 2 shares spatial CSR across three independent same-index channels.
    Code 3 has float32 +0 border bits and bounded positive-zero/normal sources.
    No arbitrary pixel stride or cross-channel interpretation is admitted.
    """
    _require(type(view.kind) is str and _integer(view.channels),
             "unsupported_layout", "exact dtype and channel metadata required")
    if view.kind == "u8" and view.channels == 1:
        return 1, 1, 1, 1
    if view.kind == "u8" and view.channels == 3:
        return 2, 3, 3, 1
    if view.kind == "f32le" and view.channels == 1:
        return 3, 1, 4, 4
    raise _Refuse("unsupported_layout", "only uint8 C1/C3 and bounded float32 C1 sources")


def _validate_source_values(request):
    """Constant-space scan after reservation, including zero-coefficient pixels.

    Source +0 or normal [2**-126,1] only. Every fetched subtraction is finite
    and bounded; zero weight cannot turn an admitted operand into NaN/Inf.
    Intermediate underflow/rounding and byte outputs are NOT certified here.
    """
    if request.source.kind != "f32le":
        return
    source = request.source
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            bits = U32.unpack_from(source.data, y * source.row_stride + 4 * x)[0]
            _require(bits == 0 or 0x00800000 <= bits <= 0x3F800000,
                     "unsupported_source_domain", "float source must be positive zero or normal [0,1]")


def _validate(request: SupportRequest, budget: int) -> tuple[int, int]:
    _require(type(request) is SupportRequest, "unsupported_input", "typed request required")
    _require(_known_kernel(request.kernel),
             "unsupported_kernel", "missing or unrecognized immutable source/build identity")
    _require(type(request.fp_policy) is str and request.fp_policy == FP_POLICY,
             "unsupported_fp_policy", "only the declared binary32 RNE model is supported")
    _require(type(request.active_dispatch) is str and request.active_dispatch in ("BASELINE", "SSE4_1", "AVX2"),
             "unsupported_kernel", "unknown remap dispatcher")
    _require(type(request.relative_maps) is bool and not request.relative_maps,
             "unsupported_maps", "relative maps are outside this contract")
    _require(all(type(x) is str for x in (request.source_kind, request.resolved_hint, request.interpolation, request.border_type))
             and request.source_kind == "Mat" and request.resolved_hint == "ALGO_HINT_ACCURATE"
             and request.interpolation == "INTER_LINEAR" and request.border_type == "BORDER_CONSTANT",
             "unsupported_contract", "only accurate Mat planar linear constant-border contract")
    _require(_integer(request.border_value) and INT32_MIN < request.border_value <= INT32_MAX,
             "unsupported_border", "only int32 integer border values with unambiguous uint8 saturation")
    _require(type(request.source) is BufferView, "unsupported_layout", "typed source required")
    dtype, channels, pixel_bytes, element_bytes = _source_contract(request.source)
    if dtype != 1:
        _require(request.border_value == 0, "unsupported_border", "new producer types require integer positive-zero border")
    spans = []
    total = 0
    for view, kind, count, size, alignment in (
            (request.source, request.source.kind, channels, pixel_bytes, element_bytes),
            (request.map_x, "f32le", 1, 4, 4), (request.map_y, "f32le", 1, 4, 4)):
        _require(type(view) is BufferView and type(view.data) is bytes and type(view.kind) is str and view.kind == kind
                 and _integer(view.channels) and view.channels == count,
                 "unsupported_layout", "immutable bytes and exact native dtype/channels required")
        span = _layout(view.shape, view.row_stride, view.address, size, alignment)
        _require(len(view.data) == span, "unsupported_layout", "snapshot span differs from native layout")
        spans.append((view.address, view.address + span))
        total += span
    _require(type(request.destination) is OutputLayout, "unsupported_layout", "typed destination layout required")
    dst = request.destination
    _require(type(dst.kind) is str and dst.kind == request.source.kind
             and _integer(dst.channels) and dst.channels == channels,
             "unsupported_layout", "destination dtype/channels must equal source")
    dspan = _layout(dst.shape, dst.row_stride, dst.address, pixel_bytes, element_bytes)
    spans.append((dst.address, dst.address + dspan))
    _require(request.map_x.shape == request.map_y.shape == dst.shape,
             "unsupported_layout", "map/destination shapes differ")
    for i, (a, b) in enumerate(spans):
        for c, d in spans[i + 1:]:
            _require(b <= c or d <= a, "unsupported_alias", "native buffers overlap, including identical planar starts")
    rows = dst.shape[0] * dst.shape[1]
    _require(rows <= MAX_ROWS, "resource_cap", "destination row cap exceeded before support allocation")
    planned = 2 * total + 2 * witness_bound(rows) + SCRATCH_BYTES
    _require(_integer(budget) and 0 < budget <= MAX_EXTRA_BYTES and planned <= budget,
             "resource_cap", "planned aggregate extra-memory reservation unavailable")
    return rows, planned


def _value(bits: int, *, reject_subnormal: bool = True) -> Fraction:
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    _require(exponent != 255, "nonfinite_coordinate", "NaN/infinity is not an integer conversion domain")
    _require(not (reject_subnormal and exponent == 0 and mantissa), "unsupported_fp_domain", "subnormal coordinate/fraction")
    if exponent == 0:
        value = Fraction(mantissa, 1 << 149)
    else:
        power = exponent - 150
        significand = (1 << 23) | mantissa
        value = Fraction(significand << power) if power >= 0 else Fraction(significand, 1 << -power)
    return -value if bits >> 31 else value


def _round_ratio(n: int, d: int) -> int:
    q, r = divmod(n, d)
    return q + int(2 * r > d or (2 * r == d and q % 2 != 0))


def _round_binary32(value: Fraction, *, negative_zero: bool = False) -> int:
    """Software IEEE round-to-nearest/ties-to-even; no host float arithmetic."""
    sign = int(value < 0 or (value == 0 and negative_zero)) << 31
    value = abs(value)
    if not value:
        return sign
    n, d = value.numerator, value.denominator
    e = n.bit_length() - d.bit_length()
    below_power = (n < d << e) if e >= 0 else (n << -e < d)
    if below_power:
        e -= 1
    _require(e <= 127, "unsafe_coordinate", "binary32 overflow")
    if e < -126:
        m = _round_ratio(n << 149, d)
        return sign | m
    shift = 23 - e
    m = _round_ratio(n << shift, d) if shift >= 0 else _round_ratio(n, d << -shift)
    if m == 1 << 24:
        m >>= 1
        e += 1
    _require(e <= 127, "unsafe_coordinate", "binary32 rounding overflow")
    return sign | ((e + 127) << 23) | (m - (1 << 23))


def _coordinate(bits: int) -> tuple[int, int, Fraction]:
    x = _value(bits)
    base = x.numerator // x.denominator
    _require(INT32_MIN <= base < INT32_MAX, "unsafe_coordinate", "floor or neighbor exceeds int32")
    native_integer = _value(_round_binary32(Fraction(base)))
    fraction_bits = _round_binary32(x - native_integer, negative_zero=(bits == 0x80000000 and base == 0))
    fraction = _value(fraction_bits)
    _require(0 <= fraction <= 1, "unsupported_fp_domain", "rounded native fraction outside [0,1]")
    return base, fraction_bits, fraction


def _dyadic(value: Fraction) -> Dyadic:
    d = value.denominator
    _require(d & (d - 1) == 0 and 0 <= value <= 1 and d.bit_length() <= 253,
             "unsupported_fp_domain", "symbolic coefficient outside bounded dyadic domain")
    return Dyadic(value.numerator, d.bit_length() - 1)


def _feed(h, value: object) -> None:
    """Canonical, length-tagged, streaming encoding; no serialized trace copy."""
    if type(value) in (KernelIdentity, BufferView, OutputLayout, SupportRequest, Dyadic, BorderTerm):
        h.update(b"D")
        _feed(h, type(value).__name__)
        for field in fields(value):
            if field.name != "seal_sha256":
                _feed(h, field.name)
                _feed(h, getattr(value, field.name))
    elif type(value) is tuple:
        h.update(b"T" + str(len(value)).encode() + b":")
        for element in value:
            _feed(h, element)
    elif type(value) is bytes:
        h.update(b"B" + str(len(value)).encode() + b":" + hashlib.sha256(value).digest())
    elif type(value) in (str, int, bool):
        b = str(value).encode("utf8")
        h.update(type(value).__name__.encode() + b":" + str(len(b)).encode() + b":" + b)
    else:
        raise _Refuse("unsupported_input", "noncanonical field type")


def _seal(value: object, domain: bytes) -> str:
    h = hashlib.sha256(domain)
    _feed(h, value)
    return h.hexdigest()


def witness_bound(rows: int) -> int:
    return HEADER_LIMIT + 184 * rows + 8


def _row(request: SupportRequest, row: int):
    """The accepted numerical oracle, retaining at most four terms for one row."""
    dy, dx = divmod(row, request.destination.shape[1])
    xb = U32.unpack_from(request.map_x.data, dy * request.map_x.row_stride + 4 * dx)[0]
    yb = U32.unpack_from(request.map_y.data, dy * request.map_y.row_stride + 4 * dx)[0]
    ix, ab, a = _coordinate(xb)
    iy, bb, b = _coordinate(yb)
    weights = ((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b)
    height, width = request.source.shape
    dtype, channels, pixel_bytes, element_bytes = _source_contract(request.source)
    source, border = [], []
    constant = min(255, max(0, request.border_value))
    for corner, (oy, ox) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        cy, cx = iy + oy, ix + ox
        # C3 SIMD offsets are bytes, including v_mul(ix,3); float C1 offsets
        # are float elements after dividing the native byte stride by four.
        product = (request.source.row_stride // element_bytes) * cy
        column = (pixel_bytes // element_bytes) * cx
        _require(INT32_MIN <= product <= INT32_MAX and INT32_MIN <= column <= INT32_MAX
                 and INT32_MIN <= product + column <= INT32_MAX
                 and INT32_MIN <= product + column + channels - 1 <= INT32_MAX,
                 "unsafe_address", "native signed32 element/channel offset arithmetic")
        if 0 <= cx < width and 0 <= cy < height:
            byte_offset = element_bytes * (product + column)
            _require(0 <= byte_offset and byte_offset + pixel_bytes <= len(request.source.data),
                     "unsafe_address", "native source pixel byte extent")
        weight = weights[corner]
        if not weight:
            continue
        coefficient = _dyadic(weight)
        if 0 <= cx < width and 0 <= cy < height:
            source.append((coefficient, corner, 0, 0, cy * width + cx))
        else:
            border.append((coefficient, corner, 1, constant, UINT32_MAX))
    return (ix, iy, ab, bb), source, border


def _offsets(rows: int, sources: int, borders: int):
    bases = HEADER_BYTES
    source_ptr = bases + 16 * rows
    border_ptr = source_ptr + 4 * (rows + 1)
    source_terms = border_ptr + 4 * (rows + 1)
    border_terms = source_terms + TERM_BYTES * sources
    total = border_terms + TERM_BYTES * borders
    return bases, source_ptr, border_ptr, source_terms, border_terms, total


def _contract_bytes(request):
    dtype, channels, pixel_bytes, element_bytes = _source_contract(request.source)
    source_span = len(request.source.data)
    dst = request.destination
    destination_span = (dst.shape[0] - 1) * dst.row_stride + dst.shape[1] * pixel_bytes
    # Nine uint32 words: dtype, channels, pixel stride bytes, offset unit bytes,
    # channel mapping (1=same channel), border scalar bits/value, source span,
    # destination span, float domain (1=positive zero or normal [0,1]).
    border = min(255, max(0, request.border_value)) if dtype == 1 else 0
    return struct.pack("<9I", dtype, channels, pixel_bytes, element_bytes, 1,
                       border, source_span, destination_span, int(dtype == 3))


def _header(request, rows, sources, borders):
    sh, sw = request.source.shape
    dh, dw = request.destination.shape
    return (MAGIC, PACKED_VERSION, HEADER_BYTES, rows, sources, borders, sh, sw, dh, dw,
            *_offsets(rows, sources, borders), bytes.fromhex(_seal(request, b"OPENCV-CPU-V2-INPUT\0")),
            bytes.fromhex(_seal(request.kernel, b"OPENCV-CPU-V3-KERNEL\0")), _contract_bytes(request))


def _allocate(size):
    return bytearray(size)


def _freeze(buffer):
    return bytes(buffer)


def _pack_term(buffer, offset, term):
    weight, corner, kind, value, index = term
    TERM.pack_into(buffer, offset, weight.numerator.to_bytes(32, "little"),
                   weight.denominator_shift, corner, kind, value, index)


def _support_seal(support):
    h = hashlib.sha256(b"OPENCV-CPU-V4-PACKED-SYMBOLIC-SUPPORT\0")
    _feed(h, PACKED_VERSION)
    _feed(h, support.request_sha256)
    _feed(h, support.planned_extra_bytes)
    # All bytes, including reserved fields and offsets, are sealed directly.
    h.update(memoryview(support.payload))
    return h.hexdigest()


def _build_packed(request, rows, planned):
    """All temporary row/output buffers live in this cleanup-bounded frame."""
    _validate_source_values(request)
    ns = nb = 0
    for row in range(rows):
        bases, source, border = _row(request, row)
        ns += len(source)
        nb += len(border)
    del bases, source, border
    header = _header(request, rows, ns, nb)
    offsets = _offsets(rows, ns, nb)
    buffer = _allocate(offsets[-1])
    HEADER.pack_into(buffer, 0, *header)
    source_count = border_count = 0
    for row in range(rows):
        bases, source, border = _row(request, row)
        BASE.pack_into(buffer, offsets[0] + 16 * row, *bases)
        U32.pack_into(buffer, offsets[1] + 4 * row, source_count)
        U32.pack_into(buffer, offsets[2] + 4 * row, border_count)
        for term in source:
            _pack_term(buffer, offsets[3] + TERM_BYTES * source_count, term)
            source_count += 1
        for term in border:
            _pack_term(buffer, offsets[4] + TERM_BYTES * border_count, term)
            border_count += 1
    _require(source_count == ns and border_count == nb, "inconsistent_input", "symbolic passes disagree")
    U32.pack_into(buffer, offsets[1] + 4 * rows, ns)
    U32.pack_into(buffer, offsets[2] + 4 * rows, nb)
    payload = _freeze(buffer)
    del buffer, bases, source, border, term
    request_hash = header[16].hex()
    support = SymbolicSupport(payload, request_hash, planned, "")
    return SymbolicSupport(payload, request_hash, planned, _support_seal(support))


def _build_and_commit(transaction, request, rows, planned):
    support = _build_packed(request, rows, planned)
    transaction.ledger.retain(transaction, support)
    return support


def build_symbolic_support(request: SupportRequest, *, max_extra_bytes: int = MAX_EXTRA_BYTES) -> SymbolicSupport | SupportRefusal:
    """Authoritative transaction admission/commit, or cleanup then rollback."""
    try:
        rows, planned = _validate(request, max_extra_bytes)
        with _LEDGER.transaction(planned, max_extra_bytes) as transaction:
            return _build_and_commit(transaction, request, rows, planned)
    except _Refuse as error:
        return error.refusal


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_header(request, support, rows, planned):
    _require(type(support.payload) is bytes and HEADER_BYTES <= len(support.payload) <= witness_bound(rows),
             "invalid_witness", "immutable packed bytes length out of bounds")
    _require(_integer(support.planned_extra_bytes) and support.planned_extra_bytes == planned
             and _digest(support.request_sha256) and _digest(support.seal_sha256),
             "invalid_witness", "typed metadata required")
    head = HEADER.unpack_from(support.payload)
    ns, nb = head[4:6]
    _require(ns + nb <= 4 * rows and ns + nb >= rows,
             "invalid_witness", "nonzero term count outside per-pixel support bound")
    expected = _header(request, rows, ns, nb)
    _require(head == expected and head[15] == len(support.payload)
             and support.request_sha256 == head[16].hex(),
             "invalid_witness", "unknown schema, header, offsets, reserved bytes or length")
    return ns, nb, head[10:16]


def _compare_term(view, offset, expected):
    numerator_bytes, shift, corner, kind, value, index = TERM.unpack_from(view, offset)
    numerator = int.from_bytes(numerator_bytes, "little")
    _require(0 <= shift <= 252 and 0 < numerator <= 1 << shift
             and (shift == 0 or numerator % 2 == 1), "invalid_witness", "noncanonical exact dyadic")
    coefficient, ec, ek, ev, ei = expected
    _require((numerator, shift, corner, kind, value, index)
             == (coefficient.numerator, coefficient.denominator_shift, ec, ek, ev, ei),
             "invalid_witness", "term differs from the approved row oracle")


def _verify_packed(request, support, rows, planned):
    ns, nb, offsets = _read_header(request, support, rows, planned)
    _validate_source_values(request)
    _require(_support_seal(support) == support.seal_sha256, "invalid_witness", "packed digest mismatch")
    view = memoryview(support.payload)
    source_count = border_count = 0
    for row in range(rows):
        bases, source, border = _row(request, row)
        _require(BASE.unpack_from(view, offsets[0] + 16 * row) == bases,
                 "invalid_witness", "base or raw native fractions differ")
        _require(U32.unpack_from(view, offsets[1] + 4 * row)[0] == source_count
                 and U32.unpack_from(view, offsets[2] + 4 * row)[0] == border_count,
                 "invalid_witness", "noncanonical CSR row boundary")
        for term in source:
            _require(source_count < ns, "invalid_witness", "source count exhausted")
            _compare_term(view, offsets[3] + TERM_BYTES * source_count, term)
            source_count += 1
        for term in border:
            _require(border_count < nb, "invalid_witness", "border count exhausted")
            _compare_term(view, offsets[4] + TERM_BYTES * border_count, term)
            border_count += 1
    _require(source_count == ns and border_count == nb
             and U32.unpack_from(view, offsets[1] + 4 * rows)[0] == ns
             and U32.unpack_from(view, offsets[2] + 4 * rows)[0] == nb,
             "invalid_witness", "CSR endpoints or table counts differ")
    return True


def verify_symbolic_support(request: SupportRequest, support: SymbolicSupport, *, max_extra_bytes: int = MAX_EXTRA_BYTES) -> bool:
    """Bounded worker replay under an identity-bound ledger transaction."""
    if type(support) is not SymbolicSupport:
        return False
    try:
        rows, planned = _validate(request, max_extra_bytes)
        _require(type(support.payload) is bytes and HEADER_BYTES <= len(support.payload) <= witness_bound(rows),
                 "invalid_witness", "packed payload bound before admission")
        with _LEDGER.transaction(planned, max_extra_bytes, support):
            return _verify_packed(request, support, rows, planned)
    except (_Refuse, struct.error):
        return False
