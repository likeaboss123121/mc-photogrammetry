"""Independent descriptor contract checks; never execute the consumed packet."""
import ast
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/evaluation/rgb_context_pure_falsifier.py'
FROZEN = (
    (SOURCE, '9a267c3ae052396272180edb1bb0bce19215181fbebe0840ee36eac7236a6e7d'),
    (ROOT/'tests/test_rgb_context_pure_falsifier.py',
     '414fca105c9de175fe30ae52c0eafa6325d5e0cbb84642ce29ee1fa53fecacb6'),
)
spec = importlib.util.spec_from_file_location('rgb_context_supplemental_contract',SOURCE)
pure = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pure
spec.loader.exec_module(pure)


@pytest.fixture(autouse=True)
def consumed_paths_unchanged():
    for path,digest in FROZEN:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    yield
    for path,digest in FROZEN:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def affine_bytes(width=48,height=48):
    # Byte-valued affine planes, with no saturation, wraps or interpolation table.
    return bytes(channel for y in range(height) for x in range(width)
                 for channel in (3*x+2*y+5,2*x+3*y+9,x+4*y+13))


def original_d(n):
    # Deliberately different from the registered packet's constant32 SIFT rows.
    row = struct.pack('<128f',*(0.125 if i%2 else 0.25 for i in range(128)))
    return row*n


def rational_sample_domain(point,radius,width):
    # Independent exact coordinate reference. No descriptor implementation helper.
    px,py = map(Fraction,point)
    pixels = set()
    for by in range(20):
        yy = py+radius*(Fraction(2*by+1,20)-1)
        iy = yy.numerator//yy.denominator
        for ax in range(20):
            xx = px+radius*(Fraction(2*ax+1,20)-1)
            ix = xx.numerator//xx.denominator
            for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                pixels.add((iy+dy)*width+ix+dx)
    return pixels


def test_nonconstant_fractional_bilinear_pooled_reference():
    point = (23.25,24.75)
    frame = pure.Frame(48,48,affine_bytes(),bytes([1])*(48*48),
                       (point,),(8.0,),original_d(1),(0,))
    before = (frame.rgb,frame.usable,frame.D)
    d = pure.describe(frame)
    # A bilinear interpolant reproduces an affine plane. Averaging a5x5 cell
    # evaluates it at that cell's sample-coordinate mean, independently rational.
    expected = []
    for cy in range(4):
        y = Fraction(point[1])+16*(Fraction(2*cy+1,4)-1)
        for cx in range(4):
            x = Fraction(point[0])+16*(Fraction(2*cx+1,4)-1)
            red,green,blue = 3*x+2*y+5,2*x+3*y+9,x+4*y+13
            expected.extend(((red+2*green+blue)/1020,
                             (red-green)/255,(blue-green)/255))
    assert d.raw[0] == pytest.approx(tuple(map(float,expected)),rel=0,abs=2e-15)
    norm = math.sqrt(sum(float(v)**2 for v in expected))
    assert d.unit[0] == pytest.approx(tuple(float(v)/norm for v in expected),rel=0,abs=2e-15)
    assert d.footprints == ((7,8,40,41),)
    assert d.mask_checks == (34*34,)
    assert d.first_bad == ((-1,-1),) and d.availability == ('positive',)
    observed = {i for i,value in enumerate(d.read_domain) if value}
    assert observed == rational_sample_domain(point,16,48)
    assert all(7 <= i%48 <= 40 and 8 <= i//48 <= 41 for i in observed)
    assert d.rgb_taps == 400*4
    assert (frame.rgb,frame.usable,frame.D) == before


def test_duplicate_coordinates_keep_distinct_rows_and_exact_ties():
    point = (20.5,20.5)
    frame = pure.Frame(48,48,affine_bytes(),bytes([1])*(48*48),
                       (point,point),(8.0,8.0),original_d(2),(0,1))
    d = pure.describe(frame)
    assert frame.row_ids == (0,1) and len(d.A) == len(d.P) == len(d.B) == 2
    assert d.A[0] == d.A[1] and d.P[0] == d.P[1] and d.B[0] == d.B[1]
    assert d.mask_checks == (34*34,34*34)
    assert d.rgb_taps == 2*400*4
    q = tuple(tuple(pure.exact_squared(a,b) for b in d.B) for a in d.B)
    assert q == ((0,0),(0,0))
    assert pure.strict_matches(q) == ()


def test_dense_mask_retains_excluded_row_without_rgb_reads():
    width,height = 96,64
    points = ((20.25,20.75),(72.25,40.75))
    # All but one pixel of the first footprint excluded; second footprint valid.
    mask = bytearray(width*height)
    mask[4*width+4] = 1
    for y in range(24,58):
        for x in range(56,90):
            mask[y*width+x] = 1
    rgb = bytes(v for y in range(height) for x in range(width) for v in (x,y,x+y))
    frame = pure.Frame(width,height,rgb,bytes(mask),points,(8.0,8.0),original_d(2),(0,1))
    d = pure.describe(frame)
    assert d.footprints == ((4,4,37,37),(56,24,89,57))
    # Full inspection despite an early first-bad pixel; no early count truncation.
    assert d.mask_checks == (34*34,34*34)
    assert d.first_bad == ((5,4),(-1,-1))
    assert d.availability == ('excluded','positive')
    assert len(d.A) == len(d.P) == len(d.B) == 2 and frame.row_ids == (0,1)
    assert d.P[0][128:] == d.B[0][128:] == (0.0,)*48
    assert d.raw[0] == d.unit[0] == (0.0,)*48
    assert d.rgb_taps == 400*4
    assert all(not d.read_domain[y*width+x] for y in range(4,38) for x in range(4,38))
    observed = {i for i,v in enumerate(d.read_domain) if v}
    assert observed == rational_sample_domain(points[1],16,width)
    assert all(mask[i] for i in observed)
    assert any(v != 0 for v in d.unit[1])


def test_size_below_eight_uses_fixed_radius_floor():
    frame = pure.Frame(48,48,affine_bytes(),bytes([1])*(48*48),
                       ((24.0,24.0),),(7.25,),original_d(1),(0,))
    d = pure.describe(frame)
    assert d.radii == (16.0,) and d.footprints == ((8,8,40,40),)
    assert d.mask_checks == (33*33,) and d.availability == ('positive',)
    assert d.rgb_taps == 400*4
    assert {i for i,v in enumerate(d.read_domain) if v} == rational_sample_domain((24,24),16,48)


def test_supplement_calls_only_authorized_pure_contracts():
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute)
             and isinstance(node.func.value,ast.Name) and node.func.value.id == 'pure'}
    assert calls == {'Frame','describe','exact_squared','strict_matches'}
    assert 'cv2' not in sys.modules
    assert not any(name.endswith('run_camera_mixed_depth') for name in sys.modules)
