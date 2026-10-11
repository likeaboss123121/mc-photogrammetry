"""Pure synthetic tests; direct file loading avoids production package imports."""
import ast
import builtins
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import tracemalloc
import weakref
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

PATH = Path(__file__).parents[1] / "src/mc_photogrammetry/reconstruction/evaluation_remap_support_v2.py"
spec = importlib.util.spec_from_file_location("_evaluation_remap_support_v2_test", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
original_import = builtins.__import__


def no_native_import(name, *args, **kwargs):
    if name.split(".")[0] in ("cv2", "numpy", "mc_photogrammetry"):
        raise AssertionError("No native/production imports permitted: " + name)
    return original_import(name, *args, **kwargs)


builtins.__import__ = no_native_import
try:
    spec.loader.exec_module(m)
finally:
    builtins.__import__ = original_import


@pytest.fixture(autouse=True)
def isolated_test_lifetime(monkeypatch):
    # Test-only isolation. The production module has no ledger reset/release API.
    monkeypatch.setattr(m, "_LEDGER", m._AggregateLedger())


def decode(support):
    """Small-fixture test oracle only; production never unpacks full arrays."""
    h = m.HEADER.unpack_from(support.payload)
    n, ns, nb = h[3:6]
    if n > 1024:
        raise AssertionError("Test decoder must not materialize target-size witnesses")
    bases_offset, sp, bp, st, bt, _ = h[10:16]
    bases, fractions = [], []
    for i in range(n):
        x, y, a, b = m.BASE.unpack_from(support.payload, bases_offset + 16*i)
        bases.append((x,y)); fractions.append((a,b))
    ptr = tuple(m.U32.unpack_from(support.payload, sp+4*i)[0] for i in range(n+1))
    border_ptr = tuple(m.U32.unpack_from(support.payload, bp+4*i)[0] for i in range(n+1))
    def term(offset):
        raw, shift, corner, kind, value, index = m.TERM.unpack_from(support.payload, offset)
        return m.Dyadic(int.from_bytes(raw,"little"),shift), corner, kind, value, index
    sources = [term(st+40*i) for i in range(ns)]
    borders = []
    for row in range(n):
        for i in range(border_ptr[row],border_ptr[row+1]):
            weight,corner,kind,value,index=term(bt+40*i)
            borders.append(m.BorderTerm(row,corner,weight,value))
    return SimpleNamespace(bases=tuple(bases), fraction_bits=tuple(fractions), row_ptr=ptr,
                           border_ptr=border_ptr, source_weights=tuple(x[0] for x in sources),
                           source_corners=tuple(x[1] for x in sources), source_indices=tuple(x[4] for x in sources),
                           border_terms=tuple(borders))


def mutate_payload(support, offset, raw):
    data = bytearray(support.payload)
    data[offset:offset+len(raw)] = raw
    return replace(support, payload=bytes(data))


def req(xs=(0x3F000000,), ys=(0x3F000000,), *, source_shape=(2, 2), output_shape=(1, 1), source_stride=None,
        map_padding=0, source_value=0, border=0):
    h, w = source_shape
    stride = source_stride or w
    source = m.BufferView(bytes([source_value]) * ((h - 1) * stride + w), source_shape, stride, 0x10000000, "u8")
    oh, ow = output_shape
    step = 4 * ow + map_padding
    def view(values, address):
        if len(values) != oh * ow:
            raise AssertionError("Fixture point count mismatch")
        rows = [b"".join(struct.pack("<I", x) for x in values[y * ow:(y + 1) * ow]) for y in range(oh)]
        data = bytes(map_padding).join(rows)
        return m.BufferView(data, output_shape, step, address, "f32le")
    return m.SupportRequest(m.REFERENCE_KERNEL, source, view(xs, 0x20000000), view(ys, 0x30000000),
                            m.OutputLayout(output_shape, ow, 0x40000000), fp_policy=m.FP_POLICY, active_dispatch="BASELINE",
                            resolved_hint="ALGO_HINT_ACCURATE", border_value=border)


def build(request=None, **kwargs):
    result = m.build_symbolic_support(request or req(), **kwargs)
    assert type(result) is m.SymbolicSupport, result
    return result


def refuse(request, code=None, **kwargs):
    result = m.build_symbolic_support(request, **kwargs)
    assert type(result) is m.SupportRefusal, result
    if code:
        assert result.code == code
    return result


def weights(support):
    return tuple(x.as_fraction() for x in decode(support).source_weights)


def reseal(support):
    return replace(support, seal_sha256=m._support_seal(support))


def test_pure_import_no_production_or_native_runtime():
    assert "cv2" not in sys.modules
    tree = ast.parse(PATH.read_text())
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(x.startswith(("cv2", "numpy", "mc_photogrammetry")) for x in imports)


def test_half_pixel_exact_canonical_weights_and_zero_pixels_keep_membership():
    s = build()
    assert decode(s).bases == ((0, 0),)
    assert decode(s).fraction_bits == ((0x3F000000, 0x3F000000),)
    assert decode(s).row_ptr == (0, 4)
    assert decode(s).source_indices == (0, 1, 2, 3)
    assert decode(s).source_corners == (0, 1, 2, 3)
    assert weights(s) == (Fraction(1, 4),) * 4
    assert not decode(s).border_terms
    nonzero = build(req(source_value=255))
    assert decode(nonzero).source_indices == decode(s).source_indices
    assert decode(nonzero).source_weights == decode(s).source_weights
    assert nonzero.request_sha256 != s.request_sha256


@pytest.mark.parametrize("xb,base,alpha,indices,corners", [
    (0, 0, 0, (0,), (0,)),
    (0x80000000, 0, 0x80000000, (0,), (0,)),
    (0x3F800000, 1, 0, (1,), (0,)),
    (0x3F7FFFFF, 0, 0x3F7FFFFF, (0, 1), (0, 1)),
    (0x3F800001, 1, 0x34000000, (1, 2), (0, 1)),
    (0xB2800000, -1, 0x3F800000, (0,), (1,)),  # -2**-26
    (0xB3000000, -1, 0x3F800000, (0,), (1,)),  # -2**-25 tie -> even1
    (0xB3800000, -1, 0x3F7FFFFF, (0,), (1,)),  # -2**-24
])
def test_native_fraction_rounding_endpoints_and_nextafter(xb, base, alpha, indices, corners):
    s = build(req((xb,), (0,), source_shape=(2, 3)))
    assert decode(s).bases == ((base, 0),)
    assert decode(s).fraction_bits == ((alpha, 0),)
    assert decode(s).source_indices == indices
    assert decode(s).source_corners == corners
    assert sum(weights(s), Fraction()) + sum((x.weight.as_fraction() for x in decode(s).border_terms), Fraction()) == 1


@pytest.mark.parametrize("value,bits", [
    (Fraction(1, 2), 0x3F000000),
    (Fraction(1) - Fraction(1, 1 << 25), 0x3F800000),
    (Fraction(1) - Fraction(3, 1 << 25), 0x3F7FFFFE),
    (Fraction(1) + Fraction(1, 1 << 24), 0x3F800000),
    (Fraction(1) + Fraction(3, 1 << 24), 0x3F800002),
    (Fraction(1, 1 << 126), 0x00800000),
    (Fraction(-1), 0xBF800000),
])
def test_software_binary32_rounding_known_ties(value, bits):
    assert m._round_binary32(value) == bits


def test_normal_fractions_keep_exact_product_even_if_binary32_product_underflows():
    s = build(req((0x00800000,), (0x00800000,)))
    assert decode(s).fraction_bits == ((0x00800000, 0x00800000),)
    assert decode(s).source_indices == (0, 1, 2, 3)
    assert decode(s).source_weights[-1] == m.Dyadic(1, 252)
    assert m._round_binary32(Fraction(1, 1 << 252)) == 0
    assert sum(weights(s)) == 1


@pytest.mark.parametrize("xb,yb,expected_source,expected_corners,expected_border", [
    (0xBF000000, 0x3F000000, (0, 2), (1, 3), (0, 2)),
    (0x3FC00000, 0x3F000000, (1, 3), (0, 2), (1, 3)),
    (0x3F000000, 0xBF000000, (0, 1), (2, 3), (0, 1)),
    (0x3F000000, 0x3FC00000, (2, 3), (0, 1), (2, 3)),
    (0xBF000000, 0xBF000000, (0,), (3,), (0, 1, 2)),
    (0x3FC00000, 0xBF000000, (1,), (2,), (0, 1, 3)),
    (0xBF000000, 0x3FC00000, (2,), (1,), (0, 2, 3)),
    (0x3FC00000, 0x3FC00000, (3,), (0,), (1, 2, 3)),
    (0xBF800000, 0xBF800000, (), (), (0,)),
    (0xC0000000, 0xC0000000, (), (), (0,)),
    (0x40000000, 0x40000000, (), (), (0,)),
])
def test_all_borders_corners_and_oob_are_constants_without_clipping(xb, yb, expected_source, expected_corners, expected_border):
    s = build(req((xb,), (yb,), border=17))
    assert decode(s).source_indices == expected_source
    assert decode(s).source_corners == expected_corners
    assert tuple(x.corner for x in decode(s).border_terms) == expected_border
    assert all(x.row == 0 and x.value_u8 == 17 for x in decode(s).border_terms)
    assert sum(weights(s), Fraction()) + sum((x.weight.as_fraction() for x in decode(s).border_terms), Fraction()) == 1


@pytest.mark.parametrize("border,expected", [(-42, 0), (0, 0), (255, 255), (300, 255)])
def test_integer_constant_saturation_is_sealed(border, expected):
    s = build(req((0xBF800000,), (0xBF800000,), border=border))
    assert decode(s).row_ptr == (0, 0) and decode(s).source_indices == ()
    assert decode(s).border_terms == (m.BorderTerm(0, 0, m.Dyadic(1, 0), expected),)


@pytest.mark.parametrize("shape,indices,corners", [((1, 1), (0,), (0,)), ((1, 2), (0, 1), (0, 1)), ((2, 1), (0, 1), (0, 2))])
def test_degenerate_dimensions(shape, indices, corners):
    s = build(req(source_shape=shape))
    assert decode(s).source_indices == indices and decode(s).source_corners == corners
    assert sum(weights(s)) + sum(x.weight.as_fraction() for x in decode(s).border_terms) == 1


@pytest.mark.parametrize("width", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("dispatch", ["BASELINE", "SSE4_1", "AVX2"])
def test_scalar_vector_relevant_widths_have_canonical_complete_rows(width, dispatch):
    request = replace(req((0x3F000000,) * (2 * width), (0x3F000000,) * (2 * width), output_shape=(2, width), map_padding=12), active_dispatch=dispatch)
    s = build(request)
    assert decode(s).row_ptr == tuple(4 * x for x in range(2 * width + 1))
    assert decode(s).source_indices == (0, 1, 2, 3) * (2 * width)
    assert decode(s).source_corners == (0, 1, 2, 3) * (2 * width)
    assert m.verify_symbolic_support(request, s)


def test_padded_source_maps_bind_layout_and_ignore_padding_for_indexing():
    request = req(source_stride=5, map_padding=8, xs=(0x3F000000, 0), ys=(0x3F000000, 0), output_shape=(2, 1))
    s = build(request)
    assert decode(s).row_ptr == (0, 4, 5) and decode(s).source_indices == (0, 1, 2, 3, 0)
    changed = bytearray(request.map_x.data); changed[4] = 255
    changed_request = replace(request, map_x=replace(request.map_x, data=bytes(changed)))
    other = build(changed_request)
    assert decode(other).source_indices == decode(s).source_indices
    assert other.request_sha256 != s.request_sha256
    assert not m.verify_symbolic_support(changed_request, s)


@pytest.mark.parametrize("bits,code", [(0x7F800000, "nonfinite_coordinate"), (0xFF800000, "nonfinite_coordinate"),
    (0x7FC00001, "nonfinite_coordinate"), (0x7F800001, "nonfinite_coordinate"), (1, "unsupported_fp_domain"),
    (0x80000001, "unsupported_fp_domain"), (0x007FFFFF, "unsupported_fp_domain"),
    (0x4F000000, "unsafe_coordinate"), (0xCF000001, "unsafe_coordinate")])
def test_nonfinite_subnormal_and_int_overflow_refuse(bits, code):
    refuse(req((bits,), (0,)), code)


def test_signed32_vector_address_arithmetic_guard_even_for_oob_or_zero_weight_neighbor():
    refuse(req((0,), (0x4B000000,), source_shape=(1, 1), source_stride=256), "unsafe_address")
    safe = build(req((0,), (0x4AFFFFFC,), source_shape=(1, 1), source_stride=256))
    assert decode(safe).source_indices == () and len(decode(safe).border_terms) == 1


@pytest.mark.parametrize("field", ["map_x", "map_y", "source"])
def test_overlapping_native_spans_refuse(field):
    request = req()
    view = getattr(request, field)
    request = replace(request, **{field: replace(view, address=request.destination.address)})
    refuse(request, "unsupported_alias")


def test_partial_map_overlap_and_same_planar_start_refuse_without_any_sampling():
    request = req(output_shape=(1, 2), xs=(0, 0), ys=(0, 0))
    for delta in [0, 4]:
        refuse(replace(request, map_y=replace(request.map_y, address=request.map_x.address + delta)), "unsupported_alias")


@pytest.mark.parametrize("mutator", [
    lambda r: replace(r, kernel=None),
    lambda r: replace(r, kernel=replace(r.kernel, source_commit="0" * 40)),
    lambda r: replace(r, kernel=replace(r.kernel, extension_sha256="f" * 64)),
    lambda r: replace(r, kernel=replace(r.kernel, source_files=tuple(reversed(r.kernel.source_files)))),
    lambda r: replace(r, active_dispatch=None),
    lambda r: replace(r, active_dispatch="IPP"),
])
def test_unknown_or_mismatched_kernel_refuses(mutator):
    refuse(mutator(req()), "unsupported_kernel")


@pytest.mark.parametrize("changes,code", [
    ({"fp_policy": None}, "unsupported_fp_policy"),
    ({"fp_policy": "nearest-away"}, "unsupported_fp_policy"),
    ({"fp_policy": "FTZ"}, "unsupported_fp_policy"),
    ({"relative_maps": True}, "unsupported_maps"),
    ({"resolved_hint": None}, "unsupported_contract"),
    ({"resolved_hint": "ALGO_HINT_DEFAULT"}, "unsupported_contract"),
    ({"resolved_hint": "ALGO_HINT_APPROX"}, "unsupported_contract"),
    ({"interpolation": "INTER_NEAREST"}, "unsupported_contract"),
    ({"source_kind": "UMat"}, "unsupported_contract"),
    ({"border_type": "BORDER_REPLICATE"}, "unsupported_contract"),
    ({"border_value": .5}, "unsupported_border"),
    ({"border_value": True}, "unsupported_border"),
])
def test_unverified_contract_and_fp_policy_refuse(changes, code):
    refuse(replace(req(), **changes), code)


@pytest.mark.parametrize("mutator,code", [
    (lambda r: replace(r, map_x=replace(r.map_x, kind="f64le")), "unsupported_layout"),
    (lambda r: replace(r, source=replace(r.source, kind="rgb8")), "unsupported_layout"),
    (lambda r: replace(r, map_x=replace(r.map_x, data=bytearray(r.map_x.data))), "unsupported_layout"),
    (lambda r: replace(r, map_x=replace(r.map_x, row_stride=-4)), "unsupported_layout"),
    (lambda r: replace(r, map_x=replace(r.map_x, address=r.map_x.address+1)), "unsupported_layout"),
    (lambda r: replace(r, map_x=replace(r.map_x, data=r.map_x.data+b"x")), "unsupported_layout"),
    (lambda r: replace(r, source=replace(r.source, shape=(0, 2))), "unsupported_layout"),
    (lambda r: replace(r, source=replace(r.source, shape=(32767, 1))), "unsupported_layout"),
    (lambda r: replace(r, source=replace(r.source, row_stride=1<<31)), "unsupported_layout"),
    (lambda r: replace(r, destination=replace(r.destination, address=m.UINTPTR_MAX)), "unsafe_address"),
    (lambda r: replace(r, map_x=replace(r.map_x, shape=(1, True))), "unsupported_layout"),
])
def test_dtype_layout_stride_and_native_span_refusal(mutator, code):
    refuse(mutator(req()), code)


def test_memory_reservation_boundary_and_preallocation_sentinel(monkeypatch):
    request = req()
    planned = m._validate(request, m.MAX_EXTRA_BYTES)[1]
    assert build(request, max_extra_bytes=planned).planned_extra_bytes == planned
    def forbidden(*args):
        raise AssertionError("coordinate allocations ran before reservation")
    monkeypatch.setattr(m, "_coordinate", forbidden)
    refuse(request, "resource_cap", max_extra_bytes=planned-1)
    refuse(request, "resource_cap", max_extra_bytes=m.MAX_EXTRA_BYTES+1)


def test_row_cap_before_coordinate_work(monkeypatch):
    request = req((0,) * (262145), (0,) * (262145), output_shape=(13, 20165))
    monkeypatch.setattr(m, "_coordinate", lambda *args: pytest.fail("preallocation cap bypass"))
    refuse(request, "resource_cap")


def test_immutable_inputs_identity_and_output_and_deterministic_seals():
    request = req()
    s = build(request)
    assert s == build(request) and m.verify_symbolic_support(request, s)
    for value, field, replacement in [(request.source, "data", b"x"), (s, "payload", b"x"), (m.REFERENCE_KERNEL, "source_commit", "bad")]:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)
    with pytest.raises(TypeError):
        s.payload[0] = 7


@pytest.mark.parametrize("offset,raw", [
    (m.HEADER_BYTES, struct.pack("<i",1)),                    # base
    (m.HEADER_BYTES+8, struct.pack("<I",0)),                  # fraction
    (m.HEADER_BYTES+16+4, struct.pack("<I",3)),               # source end pointer
    (m.HEADER_BYTES+16+8+4, struct.pack("<I",1)),             # border end pointer
    (m.HEADER_BYTES+16+16, (3).to_bytes(32,"little")),         # coefficient
    (m.HEADER_BYTES+16+16+33, b"\x03"),                    # corner
    (m.HEADER_BYTES+16+16+34, b"\x01"),                    # kind
    (m.HEADER_BYTES+16+16+35, b"\x01"),                    # reserved/value
    (m.HEADER_BYTES+16+16+36, struct.pack("<I",3)),            # logical index
    (8, struct.pack("<I",99)),                              # schema version
    (48, struct.pack("<Q",0)),                              # section offset
    (188, b"\x01"),                                      # dtype domain header
])
def test_resealed_semantic_tamper_rejected(offset,raw):
    request = req()
    changed = reseal(mutate_payload(build(request),offset,raw))
    assert changed.seal_sha256 == m._support_seal(changed)
    assert not m.verify_symbolic_support(request,changed)


def test_resealed_border_terms_and_foreign_source_substitution_refuse():
    request = req((0xBF800000,), (0xBF800000,), border=17)
    s = build(request)
    bt = m.HEADER.unpack_from(s.payload)[14]
    altered = mutate_payload(s,bt+33,b"\x03")
    assert not m.verify_symbolic_support(request, reseal(altered))
    assert not m.verify_symbolic_support(replace(request, source=replace(request.source, data=b"\x01"*4)), s)
    assert not m.verify_symbolic_support(request, replace(s, seal_sha256="f"*64))


def test_bool_and_foreign_objects_cannot_equal_valid_typed_witness():
    request = req((0,), (0,))
    s = build(request)
    assert not m.verify_symbolic_support(request, replace(s, planned_extra_bytes=True))
    class Equal:
        def __eq__(self,other):
            return True
    assert not m.verify_symbolic_support(request, replace(s, payload=Equal()))


def test_source_pins_and_reference_scope_are_explicit():
    assert len(m.SOURCE_PINS) == 4
    assert m.REFERENCE_KERNEL.source_commit == "40738fb16ceddb5fb3fea747585f7ce6abb0605b"
    assert all(len(h) == 64 for _, h in m.SOURCE_PINS)
    assert m.REFERENCE_KERNEL.extension_sha256 != "f1e76ab01cf4d33af780adf8ada1a26c5eaeecc55d2cbe8a5e288195e8f9b0dd"

def test_mixed_vector_relevant_groups_and_all_outside_rows_are_complete():
    width = 16
    xs = (0x3F000000,) * width + tuple(0x3F000000 if i % 2 == 0 else 0xC0000000 for i in range(width)) + (0xC0000000,) * width
    ys = (0x3F000000,) * (2 * width) + (0xC0000000,) * width
    s = build(req(xs, ys, output_shape=(3, width), border=23))
    for row in range(3 * width):
        n = decode(s).row_ptr[row + 1] - decode(s).row_ptr[row]
        expected = 4 if row < width or (row < 2 * width and row % 2 == 0) else 0
        assert n == expected
        terms = [t for t in decode(s).border_terms if t.row == row]
        assert all(t.value_u8 == 23 for t in terms)
        assert sum((t.weight.as_fraction() for t in terms), Fraction()) == (0 if n else 1)
    assert tuple((t.row, t.corner) for t in decode(s).border_terms) == tuple(sorted((t.row, t.corner) for t in decode(s).border_terms))


def large_request(x=0x3F000000,y=0x3F000000, *, shape=(512,512), worst=False):
    n = shape[0]*shape[1]
    base = req()
    step=4*shape[1]
    source=base.source
    if worst:
        step=((m.MAX_BUFFER_BYTES-4*shape[1])//(shape[0]-1)//4)*4
        source=replace(source,data=bytes(m.MAX_BUFFER_BYTES),row_stride=m.MAX_BUFFER_BYTES-2)
    span=(shape[0]-1)*step+4*shape[1]
    return replace(base,source=source,
        map_x=replace(base.map_x,data=struct.pack("<I",x)*(span//4),shape=shape,row_stride=step),
        map_y=replace(base.map_y,data=struct.pack("<I",y)*(span//4),shape=shape,row_stride=step),
        destination=replace(base.destination,shape=shape,row_stride=shape[1]))


@pytest.mark.parametrize("x,y,per_source,per_border", [
    (0x3F000000,0x3F000000,4,0),
    (0xBF000000,0x3F000000,2,2),
    (0xBF800000,0xBF800000,0,1),
])
def test_row_cap_equality_is_supported_and_independently_verified(x,y,per_source,per_border):
    request = large_request(x,y,worst=True)
    track = per_source == 4
    if track:
        tracemalloc.start()
    try:
        s = build(request)
        if track:
            current, peak = tracemalloc.get_traced_memory()
            assert 2*len(s.payload) <= peak < s.planned_extra_bytes
            assert current < len(s.payload)+m.SCRATCH_BYTES
    finally:
        if track:
            tracemalloc.stop()
    header = m.HEADER.unpack_from(s.payload)
    assert header[3:6] == (262144,per_source*262144,per_border*262144)
    assert len(s.payload) == 192+24*262144+8+40*(per_source+per_border)*262144
    assert len(s.payload) <= m.witness_bound(m.MAX_ROWS)
    assert m.verify_symbolic_support(request,s,max_extra_bytes=s.planned_extra_bytes)
    assert m._LEDGER.used==s.planned_extra_bytes
    assert 2*s.planned_extra_bytes>m.MAX_EXTRA_BYTES
    refuse(request,"resource_cap")
    assert not m.verify_symbolic_support(request,replace(s))


def test_native_buffer_cap_prevents_allocation_or_coordinate_work(monkeypatch):
    request = req()
    request = replace(request, source=replace(request.source, shape=(2, 1), row_stride=m.MAX_BUFFER_BYTES, data=b""))
    monkeypatch.setattr(m, "_coordinate", lambda *args: pytest.fail("buffer cap bypass"))
    refuse(request, "unsafe_address")


def test_source_span_padding_does_not_authorize_disguised_overlap():
    request = req(source_stride=16)
    request = replace(request, map_x=replace(request.map_x, address=request.source.address+8))
    refuse(request, "unsupported_alias")


def test_request_scalar_and_output_field_tamper_remain_typed():
    assert not m.verify_symbolic_support(req(),object())
    s=build()
    assert not m.verify_symbolic_support(req(),replace(s,request_sha256="0"*64))
    assert not m.verify_symbolic_support(req(),replace(s,planned_extra_bytes=s.planned_extra_bytes+1))
    assert not m.verify_symbolic_support(req(),replace(s,payload=bytearray(s.payload)))
    st=m.HEADER.unpack_from(s.payload)[13]
    # Zero, non-reduced and out-of-domain symbolic coefficient encodings.
    for numerator,shift in [(0,2),(2,3),(1,253),(2,0)]:
        changed=mutate_payload(s,st,numerator.to_bytes(32,"little")+bytes([shift]))
        assert not m.verify_symbolic_support(req(),reseal(changed))


def test_asymmetric_fractions_bind_x_y_and_corner_order_independently():
    s = build(req((0x3E800000, 0x3F400000), (0x3F400000, 0x3E800000), output_shape=(1, 2)))
    assert decode(s).fraction_bits == ((0x3E800000, 0x3F400000), (0x3F400000, 0x3E800000))
    assert weights(s) == tuple(Fraction(n, 16) for n in (3, 1, 9, 3, 3, 9, 1, 3))
    assert decode(s).source_indices == (0, 1, 2, 3) * 2
    assert decode(s).row_ptr == (0, 4, 8)


@pytest.mark.parametrize("bits,base", [(0x4B800001, (1<<24)+2), (0x4EFFFFFF, 2147483520), (0xCF000000, -(1<<31))])
def test_representable_large_integer_coordinates_stay_exact_with_safe_offsets(bits, base):
    s = build(req((bits,), (0,)))
    assert decode(s).bases == ((base, 0),)
    assert decode(s).fraction_bits == ((0, 0),)
    assert decode(s).source_indices == ()
    assert decode(s).border_terms == (m.BorderTerm(0, 0, m.Dyadic(1, 0), 0),)


def test_maximum_reservation_formula_and_native_python_headers():
    assert m.HEADER.size == m.HEADER_BYTES == 192
    assert m.TERM.size == 40 and m.BASE.size == 16 and m.U32.size == 4
    assert m.witness_bound(262144) == 48238600
    assert 2*(48<<20)+2*m.witness_bound(262144)+m.SCRATCH_BYTES == 213917712
    assert 213917712 < m.MAX_EXTRA_BYTES
    # Actual pinned CPython object overhead, not RSS or native allocator claims.
    assert sys.version_info[:3] == (3,10,11)
    assert sys.getsizeof(b"") == 33
    assert sys.getsizeof(bytearray()) == 56
    assert sys.getsizeof(0) == 24
    assert sys.getsizeof(1<<252) == 60
    assert sys.getsizeof(Fraction(1,1<<252)) == 48
    assert sys.getsizeof([]) == 56
    assert sys.getsizeof(()) == 40
    assert sys.getsizeof(memoryview(b"")) == 184
    assert sys.getsizeof(m.Dyadic(1,252)) == 48
    # 10,000 worst 1024-bit integers/Fractions/containers still <16MiB.
    bound = 10000*(sys.getsizeof(1<<1024)+sys.getsizeof(Fraction(1,1<<252))
                   +sys.getsizeof([None]*16)+sys.getsizeof((None,)*16))
    assert bound < m.SCRATCH_BYTES


def test_bulk_allocation_follows_complete_count_pass_and_freezes_once(monkeypatch):
    request=req((0x3F000000,0xBF000000),(0x3F000000,0x3F000000),output_shape=(1,2))
    calls=[]
    oracle=m._row
    allocate=m._allocate
    freeze=m._freeze
    def row(r,i):
        calls.append(("row",i,m._LEDGER.used))
        return oracle(r,i)
    def alloc(n):
        assert [c[:2] for c in calls] == [("row",0),("row",1)]
        calls.append(("allocate",n))
        return allocate(n)
    def frozen(b):
        calls.append(("freeze",len(b)))
        return freeze(b)
    monkeypatch.setattr(m,"_row",row)
    monkeypatch.setattr(m,"_allocate",alloc)
    monkeypatch.setattr(m,"_freeze",frozen)
    s=build(request)
    assert sum(c[0]=="allocate" for c in calls)==1
    assert sum(c[0]=="freeze" for c in calls)==1
    assert [c[1] for c in calls if c[0]=="row"] == [0,1,0,1]
    assert all(c[2]==s.planned_extra_bytes for c in calls if c[0]=="row")
    # Verifier never allocates a witness or invokes the builder.
    monkeypatch.setattr(m,"_allocate",lambda *a:pytest.fail("verifier allocated packed payload"))
    monkeypatch.setattr(m,"build_symbolic_support",lambda *a,**k:pytest.fail("verifier called builder"))
    assert m.verify_symbolic_support(request,s)


def test_cap_plus_one_and_budget_minus_one_before_work_or_allocation(monkeypatch):
    request=large_request(shape=(13,20165))
    monkeypatch.setattr(m,"_coordinate",lambda *a:pytest.fail("cap before coordinates"))
    monkeypatch.setattr(m,"_allocate",lambda *a:pytest.fail("cap before bulk allocation"))
    refuse(request,"resource_cap")
    valid=large_request()
    planned=m._validate(valid,m.MAX_EXTRA_BYTES)[1]
    refuse(valid,"resource_cap",max_extra_bytes=planned-1)
    assert m._LEDGER.used == 0


def test_interrupted_provisional_maximum_build_rolls_back_and_unlocks(monkeypatch):
    request=large_request(worst=True)
    original=m._row
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m,"_row",interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.build_symbolic_support(request)
    assert m._LEDGER.used == 0
    monkeypatch.setattr(m,"_row",original)
    # No new native/sample work: a small valid witness now fits the same ledger.
    assert m.verify_symbolic_support(req(),build())
    assert not hasattr(m._LEDGER,"release") and not hasattr(m._LEDGER,"reset")


def test_real_maximum_witness_second_build_and_foreign_equal_wrapper_refuse(monkeypatch):
    # Reserve the formal maximum input envelope, then build a tiny semantic
    # fixture under that same envelope. This isolates ownership from traversal.
    request=req()
    actual_validate=m._validate
    maximum=213917712
    monkeypatch.setattr(m,"_validate",lambda r,b:(actual_validate(r,b)[0],maximum))
    s=build(request)
    assert m.verify_symbolic_support(request,s)
    assert not m.verify_symbolic_support(request,replace(s))
    assert m._LEDGER.used == maximum
    monkeypatch.setattr(m,"_coordinate",lambda *a:pytest.fail("reservation bypass"))
    refuse(request,"resource_cap")
    payload=s.payload
    del s
    assert type(payload) is bytes and m._LEDGER.used==maximum


def test_exclusive_borrow_and_nested_or_concurrent_build_refuse(monkeypatch):
    request=req()
    s=build(request)
    original=m._row
    events=[]
    def row(r,i):
        # Both nested and another-thread operations see the same busy envelope.
        events.append(m.build_symbolic_support(r).code)
        events.append(m.verify_symbolic_support(r,s))
        with ThreadPoolExecutor(max_workers=1) as executor:
            events.append(executor.submit(m.build_symbolic_support,r).result().code)
        return original(r,i)
    monkeypatch.setattr(m,"_row",row)
    assert m.verify_symbolic_support(request,s)
    assert events == ["ownership_busy",False,"ownership_busy"]
    assert m._LEDGER.used == s.planned_extra_bytes


def test_foreign_verifier_reserves_then_refunds_its_temporary_envelope(monkeypatch):
    request=req()
    s=build(request)
    foreign=replace(s,payload=memoryview(s.payload).tobytes())
    seen=[]
    original=m._row
    def row(r,i):
        seen.append(m._LEDGER.used)
        return original(r,i)
    monkeypatch.setattr(m,"_row",row)
    assert m.verify_symbolic_support(request,foreign)
    assert seen == [2*s.planned_extra_bytes]
    assert m._LEDGER.used == s.planned_extra_bytes
    changed=reseal(mutate_payload(s,m.HEADER_BYTES,b"\x01"))
    assert not m.verify_symbolic_support(request,changed)
    assert m._LEDGER.used == s.planned_extra_bytes


@pytest.mark.parametrize("change", [lambda b:b[:-1],lambda b:b+b"\0",lambda b:b[:100]])
def test_truncated_or_trailing_resealed_payload_refuses_before_coordinate_work(change,monkeypatch):
    request=req()
    s=build(request)
    changed=reseal(replace(s,payload=change(s.payload)))
    monkeypatch.setattr(m,"_row",lambda *a:pytest.fail("header refusal bypass"))
    assert not m.verify_symbolic_support(request,changed)


def test_allocation_failure_rolls_back_charge_and_unlocks(monkeypatch):
    request=req()
    planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    allocate=m._allocate
    monkeypatch.setattr(m,"_allocate",lambda *a:(_ for _ in ()).throw(MemoryError()))
    with pytest.raises(MemoryError):
        m.build_symbolic_support(request)
    assert m._LEDGER.used==0
    monkeypatch.setattr(m,"_allocate",allocate)
    assert build(request).planned_extra_bytes==planned
    assert m._LEDGER.used==planned


def legacy_small_oracle(request):
    # Independently aggregate the previously accepted tuple witness recipe.
    # Only the unchanged binary32 coordinate primitive is shared.
    h,w=request.source.shape
    bases=[];fractions=[];ptr=[0];indices=[];corners=[];coefficients=[];constants=[]
    for row in range(request.destination.shape[0]*request.destination.shape[1]):
        y,x=divmod(row,request.destination.shape[1])
        xb=struct.unpack_from("<I",request.map_x.data,y*request.map_x.row_stride+4*x)[0]
        yb=struct.unpack_from("<I",request.map_y.data,y*request.map_y.row_stride+4*x)[0]
        ix,ab,a=m._coordinate(xb);iy,bb,b=m._coordinate(yb)
        bases.append((ix,iy));fractions.append((ab,bb))
        weights=((1-a)*(1-b),a*(1-b),(1-a)*b,a*b)
        for corner,(dy,dx) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            weight=weights[corner]
            if not weight:
                continue
            numerator,denominator=weight.numerator,weight.denominator
            coefficient=m.Dyadic(numerator,denominator.bit_length()-1)
            cy,cx=iy+dy,ix+dx
            if 0<=cy<h and 0<=cx<w:
                indices.append(cy*w+cx);corners.append(corner);coefficients.append(coefficient)
            else:
                constants.append(m.BorderTerm(row,corner,coefficient,min(255,max(0,request.border_value))))
        ptr.append(len(indices))
    return dict(bases=tuple(bases),fraction_bits=tuple(fractions),row_ptr=tuple(ptr),
                source_indices=tuple(indices),source_corners=tuple(corners),
                source_weights=tuple(coefficients),border_terms=tuple(constants))


@pytest.mark.parametrize("dispatch",["BASELINE","SSE4_1","AVX2"])
def test_packed_matches_accepted_small_oracle_for_mixed_rows(dispatch):
    bits=(0,0x80000000,0x3F000000,0xBF000000,0xBF800000,0xB2800000,0xB3800000,
          0x3F7FFFFF,0x3F800001,0x00800000,0x3E800000,0x3FC00000)
    request=replace(req(bits,tuple(reversed(bits)),output_shape=(3,4),source_stride=5,map_padding=12,border=300),active_dispatch=dispatch)
    expected=legacy_small_oracle(request)
    actual=decode(build(request))
    for field,value in expected.items():
        assert getattr(actual,field)==value


def test_invalid_then_valid_build_restores_provisional_capacity():
    invalid=req((0x7FC00001,),(0,))
    valid=req()
    planned=m._validate(valid,m.MAX_EXTRA_BYTES)[1]
    for _ in range(20):
        refuse(invalid,"nonfinite_coordinate",max_extra_bytes=planned)
        assert m._LEDGER.used==0
    support=build(valid,max_extra_bytes=planned)
    assert m._LEDGER.used==planned
    assert m.verify_symbolic_support(valid,support,max_extra_bytes=planned)
    # Returned witnesses stay charged; successful return is never refunded.
    refuse(valid,"resource_cap",max_extra_bytes=planned)
    assert m._LEDGER.used==planned


@pytest.mark.parametrize("stage",["pack","freeze","seal"])
@pytest.mark.parametrize("error_type",[MemoryError,KeyboardInterrupt])
def test_preoutput_exception_drops_buffers_before_locked_rollback(stage,error_type,monkeypatch):
    class TrackedBuffer(bytearray):
        pass
    refs=[]
    def allocation(size):
        buffer=TrackedBuffer(size)
        refs.append(weakref.ref(buffer))
        return buffer
    monkeypatch.setattr(m,"_allocate",allocation)
    def failure(*args):
        raise error_type("synthetic failure")
    monkeypatch.setattr(m,{"pack":"_pack_term","freeze":"_freeze","seal":"_support_seal"}[stage],failure)
    finish=m._LEDGER.finish
    seen=[]
    def rollback(transaction):
        if m._LEDGER._state[1][0] is not transaction:
            return finish(transaction)
        # No temporary buffer remains in worker/callee traceback frames.
        assert len(refs)==1 and refs[0]() is None
        current=m._LEDGER._state[1]
        assert current[0] is transaction and current[1]=="provisional"
        assert current[2]>0 and m._LEDGER.used==current[2]
        # Logical exclusive ownership remains held through cleanup/refund.
        assert m.build_symbolic_support(req()).code=="ownership_busy"
        seen.append(current[2])
        finish(transaction)
    monkeypatch.setattr(m._LEDGER,"finish",rollback)
    with pytest.raises(error_type) as caught:
        m.build_symbolic_support(req())
    assert str(caught.value)=="synthetic failure"
    assert len(seen)==1 and m._LEDGER.used==0
    assert not m._LEDGER._operation.locked()


@pytest.mark.parametrize("error_type",[MemoryError,KeyboardInterrupt])
def test_after_real_admission_exception_rolls_back_without_caller_flags(error_type,monkeypatch):
    request=req()
    begin=m._LEDGER.begin
    coordinate=m._coordinate
    seen=[]
    def admitted(transaction):
        begin(transaction)
        seen.append(m._LEDGER.used)
        raise error_type("after admission")
    monkeypatch.setattr(m._LEDGER,"begin",admitted)
    monkeypatch.setattr(m,"_coordinate",lambda *a:pytest.fail("post-admission failure reached worker"))
    with pytest.raises(error_type):
        m.build_symbolic_support(request)
    assert seen==[m._validate(request,m.MAX_EXTRA_BYTES)[1]]
    assert m._LEDGER.used==0 and m._LEDGER._state==((),None)
    assert not m._LEDGER._operation.locked()
    monkeypatch.setattr(m._LEDGER,"begin",begin)
    monkeypatch.setattr(m,"_coordinate",coordinate)
    support=build(request,max_extra_bytes=seen[0])
    assert m.verify_symbolic_support(request,support,max_extra_bytes=seen[0])


@pytest.mark.parametrize("owned",[False,True])
@pytest.mark.parametrize("error_type",[MemoryError,KeyboardInterrupt])
def test_verifier_post_admission_exception_restores_only_its_transaction(owned,error_type,monkeypatch):
    request=req();support=build(request)
    if not owned:
        support=replace(support,payload=memoryview(support.payload).tobytes())
    original_used=m._LEDGER.used
    begin=m._LEDGER.begin
    seen=[]
    def admitted(transaction):
        begin(transaction)
        seen.append(m._LEDGER.used)
        raise error_type("verifier after admission")
    monkeypatch.setattr(m._LEDGER,"begin",admitted)
    with pytest.raises(error_type):
        m.verify_symbolic_support(request,support)
    assert seen==[original_used if owned else original_used+support.planned_extra_bytes]
    assert m._LEDGER.used==original_used and m._LEDGER._state[1] is None
    assert not m._LEDGER._operation.locked()
    monkeypatch.setattr(m._LEDGER,"begin",begin)
    assert m.verify_symbolic_support(request,support)


@pytest.mark.parametrize("error_type",[MemoryError,KeyboardInterrupt])
def test_real_commit_then_exception_keeps_registered_bytes_charged(error_type,monkeypatch):
    request=req();planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    retain=m._LEDGER.retain
    recorded=[]
    def committed(transaction,support):
        retain(transaction,support)
        current=m._LEDGER._state[1]
        assert current[1]=="committed" and current[3] is support
        recorded.append(weakref.ref(transaction) if hasattr(transaction,"__weakref__") else id(transaction))
        raise error_type("after commit")
    monkeypatch.setattr(m._LEDGER,"retain",committed)
    with pytest.raises(error_type):
        m.build_symbolic_support(request,max_extra_bytes=planned)
    assert len(recorded)==1 and m._LEDGER.used==planned
    entries,current=m._LEDGER._state
    assert current is None and len(entries)==1
    witness,charge=entries[0]
    assert type(witness) is m.SymbolicSupport and charge==planned
    assert not m._LEDGER._operation.locked()
    assert m.verify_symbolic_support(request,witness,max_extra_bytes=planned)
    refuse(request,"resource_cap",max_extra_bytes=planned)
    monkeypatch.setattr(m._LEDGER,"retain",retain)
    other=build(request,max_extra_bytes=2*planned)
    assert m._LEDGER.used==2*planned and m.verify_symbolic_support(request,other,max_extra_bytes=2*planned)


def test_precommit_failure_rolls_back_without_retaining_witness(monkeypatch):
    request=req();planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    retain=m._LEDGER.retain
    def interrupted(transaction,support):
        raise KeyboardInterrupt("before authoritative commit")
    monkeypatch.setattr(m._LEDGER,"retain",interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.build_symbolic_support(request,max_extra_bytes=planned)
    assert m._LEDGER.used==0 and m._LEDGER._state==((),None)
    monkeypatch.setattr(m._LEDGER,"retain",retain)
    support=build(request,max_extra_bytes=planned)
    assert m.verify_symbolic_support(request,support,max_extra_bytes=planned)


def test_interrupted_cleanup_is_fail_closed_and_cannot_erase_charge(monkeypatch):
    request=req();planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    finish=m._LEDGER.finish
    def stop_cleanup(transaction):
        raise KeyboardInterrupt("cleanup interrupted")
    monkeypatch.setattr(m._LEDGER,"finish",stop_cleanup)
    with pytest.raises(KeyboardInterrupt):
        m.build_symbolic_support(request)
    entries,current=m._LEDGER._state
    assert current[1]=="committed" and current[3].planned_extra_bytes==planned
    assert m._LEDGER.used==planned
    # A failed cleanup never resets another owner's transaction or admits work.
    monkeypatch.setattr(m._LEDGER,"finish",finish)
    refuse(request,"ownership_busy")
    assert m._LEDGER.used==planned and m._LEDGER._state[1] is current


def test_ledger_state_transitions_are_one_authoritative_snapshot():
    request=req();planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    with m._LEDGER.transaction(planned,planned) as transaction:
        assert m._LEDGER._state[1][:3]==(transaction,"provisional",planned)
        support=m._build_packed(request,1,planned)
        m._LEDGER.retain(transaction,support)
        assert m._LEDGER._state[1]==(transaction,"committed",planned,support)
    assert m._LEDGER._state==(((support,planned),),None)
    assert m.verify_symbolic_support(request,support,max_extra_bytes=planned)


# Producer extension: these are symbolic Fraction evaluations, never native output.
def producer_request(kind, channels, values=None, *, source_stride=None, **kwargs):
    base = req(**kwargs)
    h, w = base.source.shape
    scalar_bytes = 4 if kind == "f32le" else 1
    pixel_bytes = scalar_bytes * channels
    stride = source_stride if source_stride is not None else w * pixel_bytes
    if values is None:
        values = (0,) * (h * w * channels)
    if len(values) != h * w * channels:
        raise AssertionError("producer fixture count")
    packed = bytearray((h - 1) * stride + w * pixel_bytes)
    for y in range(h):
        for x in range(w):
            for c in range(channels):
                value = values[(y * w + x) * channels + c]
                offset = y * stride + pixel_bytes * x + scalar_bytes * c
                if kind == "f32le":
                    struct.pack_into("<I", packed, offset, value)
                else:
                    packed[offset] = value
    return replace(base, source=replace(base.source, data=bytes(packed), kind=kind,
                   channels=channels, row_stride=stride), destination=replace(base.destination,
                   kind=kind, channels=channels, row_stride=base.destination.shape[1] * pixel_bytes))


def symbolic_channels(request, support):
    decoded = decode(support)
    result = [Fraction(0)] * request.source.channels
    for index, weight in zip(decoded.source_indices, decoded.source_weights):
        y, x = divmod(index, request.source.shape[1])
        for channel in range(request.source.channels):
            size = 4 if request.source.kind == "f32le" else 1
            offset = y * request.source.row_stride + size * (x * request.source.channels + channel)
            value = (m._value(struct.unpack_from("<I", request.source.data, offset)[0])
                     if size == 4 else request.source.data[offset])
            result[channel] += weight.as_fraction() * value
    return tuple(result)


@pytest.mark.parametrize("channel", range(3))
@pytest.mark.parametrize("pixel", range(4))
def test_rgb_channel_impulses_order_and_zero_channel_membership(channel, pixel):
    values = [0] * 12
    values[pixel * 3 + channel] = 255
    request = producer_request("u8", 3, values)
    support = build(request)
    assert decode(support).source_indices == (0, 1, 2, 3)
    assert weights(support) == (Fraction(1, 4),) * 4
    assert symbolic_channels(request, support) == tuple(Fraction(255, 4) if c == channel else 0 for c in range(3))
    assert m.verify_symbolic_support(request, support)
    contract = struct.unpack("<9I", m.HEADER.unpack_from(support.payload)[18])
    assert contract == (2, 3, 3, 1, 1, 0, 12, 3, 0)
    assert m.HEADER.unpack_from(support.payload)[4:6] == (4, 0)  # shared, not tripled


@pytest.mark.parametrize("stride,address", [(7,0x10000001),(8,0x10000002),(9,0x10000000)])
def test_rgb_padded_rows_need_byte_alignment_only(stride, address):
    request = producer_request("u8", 3, tuple(range(12)), source_stride=stride)
    request = replace(request, source=replace(request.source, address=address),
                      destination=replace(request.destination, address=0x40000001))
    support = build(request)
    assert symbolic_channels(request, support) == (Fraction(9,2), Fraction(11,2), Fraction(13,2))
    assert m.verify_symbolic_support(request, support)


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
@pytest.mark.parametrize("field", ["map_x","map_y"])
def test_producer_maps_still_require_planar_float_alignment(kind, channels, field):
    request = producer_request(kind, channels)
    request = replace(request, **{field: replace(getattr(request, field), address=getattr(request, field).address+1)})
    refuse(request, "unsupported_layout")


@pytest.mark.parametrize("bits", [0,0x00800000,0x3E800000,0x3F000000,0x3F800000])
def test_float_mask_domain_symbolic_values_and_zero_border(bits):
    request = producer_request("f32le",1,(bits,)*4)
    support = build(request)
    assert symbolic_channels(request, support) == (m._value(bits),)
    assert m.verify_symbolic_support(request, support)
    contract = struct.unpack("<9I", m.HEADER.unpack_from(support.payload)[18])
    assert contract == (3,1,4,4,1,0,16,4,1)
    outside = producer_request("f32le",1,(bits,)*4,xs=(0xBF800000,),ys=(0xBF800000,))
    witness = build(outside)
    assert decode(witness).source_indices == ()
    assert decode(witness).border_terms[0].value_u8 == 0  # header defines float32 +0 bits
    assert m.verify_symbolic_support(outside,witness)


@pytest.mark.parametrize("bits", [0x80000000,0xBF000000,0x3F800001,0x7F7FFFFF,1,0x007FFFFF,
                                  0x80000001,0x7F800000,0xFF800000,0x7FC00000])
@pytest.mark.parametrize("pixel", range(4))
def test_bad_float_sources_refuse_even_zero_coefficients_before_output(bits,pixel,monkeypatch):
    values = [0]*4
    values[pixel] = bits
    request = producer_request("f32le",1,values,xs=(0,),ys=(0,))
    monkeypatch.setattr(m,"_allocate",lambda *args: pytest.fail("invalid source allocated output"))
    refuse(request,"unsupported_source_domain")
    assert m._LEDGER.used == 0


def test_float_invalid_then_valid_rolls_back_and_finite_cancellation_is_allowed():
    bad = producer_request("f32le",1,(0,0x7F800000,0,0),xs=(0,),ys=(0,))
    refuse(bad,"unsupported_source_domain")
    good = producer_request("f32le",1,(0x3F800000,0,0,0),xs=(0,),ys=(0,))
    support = build(good)
    assert decode(support).source_indices == (0,)
    assert symbolic_channels(good,support) == (1,)
    assert m.verify_symbolic_support(good,support)


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
@pytest.mark.parametrize("border", [-1,1,255,0.0,-0.0,False])
def test_new_producers_require_integer_positive_zero_border(kind,channels,border):
    refuse(producer_request(kind,channels,border=border),"unsupported_border")


def test_float_offsets_are_element_units_not_bytes():
    # y=2**29: byte stride*iy overflows int32, float element stride*iy does not.
    request = producer_request("f32le",1,source_shape=(1,1),xs=(0,),ys=(0x4E000000,))
    support=build(request)
    assert not decode(support).source_indices
    assert m.verify_symbolic_support(request,support)
    byte_request = req(source_shape=(1,1),source_stride=4,xs=(0,),ys=(0x4E000000,))
    refuse(byte_request,"unsafe_address")


@pytest.mark.parametrize("kind,channels,x,y", [
    ("u8",3,0x4E800000,0),  # 3*ix overflow
    ("u8",3,0,0x4E800000),  # byte row-stride overflow
    ("f32le",1,0,0x4E800000),  # stride=8 -> 2 element stride overflow
])
def test_actual_unit_offset_overflow_refuses(kind,channels,x,y):
    request=producer_request(kind,channels,source_shape=(1,2),xs=(x,),ys=(y,))
    refuse(request,"unsafe_address")


@pytest.mark.parametrize("field", ["source","destination"])
@pytest.mark.parametrize("change", [{"kind":"u16"},{"channels":2},{"channels":True}])
def test_producer_dtype_and_channel_contract_strict(field,change):
    request=producer_request("u8",3)
    refuse(replace(request,**{field:replace(getattr(request,field),**change)}),"unsupported_layout")


@pytest.mark.parametrize("stride,address", [(9,0x10000000),(8,0x10000001)])
def test_float_source_alignment(stride,address):
    request=producer_request("f32le",1,source_stride=stride)
    refuse(replace(request,source=replace(request.source,address=address)),"unsupported_layout")


@pytest.mark.parametrize("word", range(9))
def test_resealed_dtype_channel_unit_span_and_semantics_substitutions_rejected(word):
    request=producer_request("u8",3)
    support=build(request)
    offset=156+4*word
    old=struct.unpack_from("<I",support.payload,offset)[0]
    changed=reseal(mutate_payload(support,offset,struct.pack("<I",old+1)))
    assert not m.verify_symbolic_support(request,changed)


def test_resealed_request_dtype_substitution_cannot_reuse_witness():
    request=producer_request("u8",3)
    support=build(request)
    other=producer_request("f32le",1)
    changed=replace(support,request_sha256=m._seal(other,b"OPENCV-CPU-V2-INPUT\0"),
                    planned_extra_bytes=m._validate(other,m.MAX_EXTRA_BYTES)[1])
    assert not m.verify_symbolic_support(other,reseal(changed))


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
def test_new_producer_source_and_destination_byte_extents(kind,channels):
    request=producer_request(kind,channels)
    size=3 if channels==3 else 4
    refuse(replace(request,destination=replace(request.destination,row_stride=size-1)),"unsupported_layout")
    refuse(replace(request,source=replace(request.source,data=request.source.data[:-1])),"unsupported_layout")
    refuse(replace(request,destination=replace(request.destination,address=m.UINTPTR_MAX-size+1)),"unsafe_address")


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
def test_new_producer_maximum_packed_reservation_and_exclusive_lifetime(kind,channels):
    request=large_request(worst=True)
    pixel_bytes=3 if channels==3 else 4
    source=replace(request.source,kind=kind,channels=channels,row_stride=m.MAX_BUFFER_BYTES-2*pixel_bytes)
    request=replace(request,source=source,destination=replace(request.destination,kind=kind,channels=channels,
                    row_stride=request.destination.shape[1]*pixel_bytes))
    rows,planned=m._validate(request,m.MAX_EXTRA_BYTES)
    assert rows==262144
    assert planned==2*sum(len(v.data) for v in (request.source,request.map_x,request.map_y))+2*m.witness_bound(rows)+m.SCRATCH_BYTES
    assert planned==213917472 < m.MAX_EXTRA_BYTES
    support=build(request)
    assert len(support.payload)==48234696
    assert m.HEADER.unpack_from(support.payload)[3:6]==(262144,1048576,0)
    assert m.verify_symbolic_support(request,support,max_extra_bytes=planned)
    assert m._LEDGER.used==planned
    refuse(request,"resource_cap")
    assert not m.verify_symbolic_support(request,replace(support))


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
def test_new_producer_budget_gate_precedes_source_scan_and_coordinate_work(kind,channels,monkeypatch):
    request=producer_request(kind,channels)
    planned=m._validate(request,m.MAX_EXTRA_BYTES)[1]
    monkeypatch.setattr(m,"_validate_source_values",lambda *args: pytest.fail("scan before budget"))
    monkeypatch.setattr(m,"_coordinate",lambda *args: pytest.fail("coordinate before budget"))
    refuse(request,"resource_cap",max_extra_bytes=planned-1)
    assert m._LEDGER.used==0


def test_float_padding_is_not_a_pixel_and_source_bits_remain_sealed():
    request=producer_request("f32le",1,(0,0x3F800000,0x3F000000,0x3E800000),source_stride=12)
    data=bytearray(request.source.data)
    struct.pack_into("<I",data,8,0x7FC00000)  # native row padding, never a source pixel
    request=replace(request,source=replace(request.source,data=bytes(data)))
    support=build(request)
    assert symbolic_channels(request,support)==(Fraction(7,16),)
    changed=replace(request,source=replace(request.source,data=request.source.data[:8]+bytes(4)+request.source.data[12:]))
    assert not m.verify_symbolic_support(changed,support)
    assert m.verify_symbolic_support(request,support)


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
@pytest.mark.parametrize("dispatch", ["BASELINE","SSE4_1","AVX2"])
def test_new_producer_conditional_dispatch_labels_share_exact_symbolic_oracle(kind,channels,dispatch):
    # Declared labels are conditional ingress metadata, not observed reachability.
    request=replace(producer_request(kind,channels),active_dispatch=dispatch)
    support=build(request)
    assert weights(support)==(Fraction(1,4),)*4
    assert m.verify_symbolic_support(request,support)


@pytest.mark.parametrize("kind,channels", [("u8",3),("f32le",1)])
def test_new_producer_tiny_normal_coordinate_products_and_degenerate_border(kind,channels):
    request=producer_request(kind,channels,source_shape=(1,1),xs=(0x00800000,),ys=(0x00800000,))
    support=build(request)
    decoded=decode(support)
    assert decoded.source_indices==(0,)
    assert decoded.border_terms[-1].weight.as_fraction()==Fraction(1,1<<252)
    assert sum(x.weight.as_fraction() for x in decoded.border_terms)+sum(weights(support))==1
    assert m.verify_symbolic_support(request,support)


def test_resealed_float_bad_source_is_rejected_before_row_oracle(monkeypatch):
    request=producer_request("f32le",1)
    support=build(request)
    bad=replace(request,source=replace(request.source,data=struct.pack("<I",0x7FC00000)+request.source.data[4:]))
    header=m._header(bad,1,4,0)
    changed=reseal(replace(mutate_payload(support,0,m.HEADER.pack(*header)),request_sha256=header[16].hex()))
    monkeypatch.setattr(m,"_row",lambda *args: pytest.fail("bad source reached row oracle"))
    assert not m.verify_symbolic_support(bad,changed)
