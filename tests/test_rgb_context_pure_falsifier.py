"""Only the registered synthetic descriptor package; no native execution."""
import ast
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'scripts/evaluation/rgb_context_pure_falsifier.py'
SPEC = importlib.util.spec_from_file_location('rgb_context_pure_falsifier', PATH)
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


@pytest.fixture(scope='module')
def result():
    value = f.run_fixed()
    print(f'PURE_REPORT status={value.status} components={value.components} '
          f'evidence_bytes={value.evidence_bytes} elapsed_seconds={value.elapsed_seconds:.6f} '
          f'failures={value.failures}')
    return value


def outcome(result, name):
    return next(o for o in result.outcomes if o.name == name)


@pytest.mark.parametrize('x', [0.0,-0.0,0.5,-1.25,math.ulp(0.0),
                             float.fromhex('0x1.fffffffffffffp+1023')])
def test_bit_decoder_exact(x):
    assert Fraction(f.dyadic(x), 1 << 1074) == Fraction.from_float(x)


def test_hand_rational_distance_and_squared_ratio():
    q = f.exact_squared((0.5,-1.0),(0.25,0.5))
    assert Fraction(q,1 << f.DENOMINATOR_EXPONENT) == Fraction(37,16)
    assert not f.ratio_pass((9,16))  # strict equality: .75 squared =9/16
    assert f.ratio_pass((8,16))
    assert not f.ratio_pass((0,0))
    assert f.rank_bounds((0,0,1),0) == (1,2)


def test_maximum_finite_integer_bound():
    x = float.fromhex('0x1.fffffffffffffp+1023')
    q = f.exact_squared((x,)*176,(-x,)*176)
    assert q.bit_length() <= f.MAX_INTEGER_BITS
    assert len(q.to_bytes(f.NUMERATOR_BYTES,'little')) == 528


def test_registered_package_passes(result):
    assert result.status == 'PURE_PASS', result.failures
    assert not result.failures
    assert tuple(o.name for o in result.outcomes) == f.CONTROL_NAMES
    assert 'REFERENCE_ONLY' in result.interpretation
    assert result.components <= 1_000_000
    assert result.evidence_bytes <= 16*1024*1024
    assert result.elapsed_seconds > 0


def test_aliases_resolved_reciprocally_without_reference_tie_break(result):
    o = outcome(result,'aliases')
    assert o.matches[:2] == ((),())
    assert o.matches[2] == o.intended == o.attributed
    for arm in (0,1):
        assert all(value == 0 for row in o.distances[arm] for value in row)
        assert f.rank_bounds(o.distances[arm][0],10) == (1,11)
    for i,j in o.intended:
        assert f.rank_bounds(o.distances[2][i],j) == (1,1)
        assert f.ratio_pass(o.distances[2][i])
        assert f.rank_bounds(tuple(row[j] for row in o.distances[2]),i) == (1,1)


@pytest.mark.parametrize('name', ['identical_outer','zero_energy','constant_y',
                                'mixed_availability','one_ulp'])
def test_negative_controls_no_information(result,name):
    assert not outcome(result,name).attributed


def test_constant_y_and_placebo(result):
    d = outcome(result,'constant_y').descriptors[0]
    assert d.availability == ('positive',)*11
    for p,b,c in zip(d.P,d.B,d.unit):
        assert p[:128] == b[:128]
        assert p[128:] == (0.5,)+(0.0,)*47
        assert all(c[i] == 0 for i in range(48) if i%3)
        assert all(abs(c[i]-0.25) <= 8*math.ulp(0.25) for i in range(0,48,3))


def test_zero_energy_policy_preserves_rows(result):
    o = outcome(result,'zero_energy')
    for d in o.descriptors:
        assert d.availability == ('zero_energy',)*11
        assert len(d.A) == len(d.P) == len(d.B) == 11
        assert all(row[128:] == (0.0,)*48 for row in d.P+d.B)
        assert d.rgb_taps == 11*400*4


def test_mixed_availability_not_context_information(result):
    o = outcome(result,'mixed_availability')
    assert o.descriptors[0].availability[:3] == ('excluded',)*3
    assert o.descriptors[1].availability[-3:] == ('excluded',)*3
    assert any(value > 0 for row in o.distances[1] for value in row)
    assert not o.attributed


def test_excluded_rgb_invariance_preserves_other_valid_gains(result):
    base,perturbed = (outcome(result,name) for name in ('excluded_base','excluded_perturbed'))
    assert base.descriptors == perturbed.descriptors
    assert base.distances == perturbed.distances
    assert base.matches == perturbed.matches
    assert base.attributed == perturbed.attributed
    assert len(base.attributed) == 10
    for o in (base,perturbed):
        for side,d in zip(('L','R'),o.descriptors):
            blobs = {b.name:b for b in o.blobs}
            mask = blobs[side+'/usable'].data
            assert all(not used for used,allowed in zip(d.read_domain,mask) if not allowed)
    assert next(b.data for b in base.blobs if b.name=='L/rgb') != \
           next(b.data for b in perturbed.blobs if b.name=='L/rgb')


def test_misleading_permutation_keeps_information_but_not_identity_credit(result):
    o = outcome(result,'misleading_permutation')
    assert len(o.matches[2]) == len(o.attributed) == 11
    assert not set(o.matches[2]) & set(o.intended)
    assert o.intended_recovered == 0
    assert o.matches[:2] == ((),())


def test_one_ulp_control_is_actual_float32_ulp(result):
    o = outcome(result,'one_ulp')
    d = next(b.data for b in o.blobs if b.name=='L/D')
    assert struct.unpack('<I',d[:4])[0] == 0x42000000
    assert struct.unpack('<I',d[512:516])[0] == 0x42000001
    assert not o.attributed


def test_mask_rectangle_and_zero_weight_domain(result):
    d = outcome(result,'zero_weight_tap').descriptors[0]
    assert d.radii == (20.0,)
    assert d.availability == ('excluded',)
    assert d.first_bad == ((2,1),)
    assert d.mask_checks == (41*41,)
    assert d.rgb_taps == 0 and not any(d.read_domain)


def test_border_fractional_radius_and_opponent_controls(result):
    for name in ('fractional_border','integer_border','opponent_color','radius_bounds'):
        assert outcome(result,name).descriptors[0].availability == ('positive',)
    for d in outcome(result,'radius_bounds').descriptors:
        assert d.radii == (64.0,)
        assert d.mask_checks == (129*129,)
    d = outcome(result,'out_of_frame').descriptors[0]
    assert d.availability == ('out_of_frame',)
    assert not d.rgb_taps and not any(d.read_domain)
    c = outcome(result,'opponent_color').descriptors[0].raw[0]
    assert math.isclose(c[0],0.25,abs_tol=1e-15)
    assert math.isclose(c[1],1.0,abs_tol=1e-15)
    assert c[2] == 0


def test_capture_is_typed_hash_bound_replayable_and_immutable(result):
    total = 0
    for o in result.outcomes:
        names = set()
        for b in o.blobs:
            assert b.name not in names; names.add(b.name)
            assert type(b.data) is bytes
            assert hashlib.sha256(b.data).hexdigest() == b.sha256
            total += len(b.data)
        qblob = next(b for b in o.blobs if b.name=='Qnum')
        raw_values = tuple(int.from_bytes(qblob.data[i:i+528],'little')
                           for i in range(0,len(qblob.data),528))
        assert raw_values == tuple(v for q in o.distances for row in q for v in row)
        with pytest.raises(FrozenInstanceError):
            o.name = 'mutated'
    assert total == result.evidence_bytes
    # Independently decode captured D/P/B and verify one exact distance per arm.
    o = outcome(result,'aliases')
    blobs = {b.name:b for b in o.blobs}
    for arm,name,d in ((0,'A',128),(1,'P',176),(2,'B',176)):
        left = struct.unpack('<'+'d'*d,blobs['L/'+name].data[:d*8])
        right = struct.unpack('<'+'d'*d,blobs['R/'+name].data[:d*8])
        expected = sum((Fraction.from_float(x)-Fraction.from_float(y))**2
                       for x,y in zip(left,right))
        assert expected == Fraction(o.distances[arm][0][0],1 << 2148)


def test_input_immutability_and_row_denominator(result):
    packets = f.fixed_packets()
    for (_,l,r,_),o in zip(packets,result.outcomes):
        for side,frame,d in zip(('L','R'),(l,r),o.descriptors):
            blobs = {b.name:b for b in o.blobs}
            assert frame.rgb == blobs[side+'/rgb'].data
            assert frame.D == blobs[side+'/D'].data
            assert frame.row_ids == tuple(range(len(d.A)))
            with pytest.raises(FrozenInstanceError):
                frame.rgb = b''
            assert _common_bytes(d.P) == _common_bytes(d.B)


def _common_bytes(rows):
    return b''.join(struct.pack('<128d',*row[:128]) for row in rows)


@pytest.mark.parametrize('budget', [f.Budget(endpoints=12),f.Budget(components=1_000_001),
                                    f.Budget(evidence=16*1024*1024+1),f.Budget(integer_bits=4225)])
def test_caps_cannot_be_raised(budget):
    with pytest.raises(f.Refusal):
        budget.validate()


def test_preflight_limits_before_distance_work(monkeypatch):
    def forbidden(*a,**k):
        pytest.fail('distance reached despite package preflight rejection')
    monkeypatch.setattr(f,'exact_squared',forbidden)
    for budget in (f.Budget(components=1),f.Budget(evidence=1),f.Budget(integer_bits=100)):
        with pytest.raises(f.Refusal):
            f.run_fixed(budget)


@pytest.mark.parametrize('change', [dict(rgb=bytearray()),dict(row_ids=(1,)),
                                   dict(D=b''),dict(sizes=(float('nan'),)),
                                   dict(width=1025)])
def test_bad_packet_refuses(change):
    frame = f.fixed_packets()[-1][1]
    with pytest.raises(f.Refusal):
        replace(frame,**change).validate()


@pytest.mark.parametrize('value', [float('nan'),float('inf'),float('-inf')])
def test_nonfinite_reference_refuses(value):
    with pytest.raises(f.Refusal):
        f.exact_squared((value,),(0.0,))


def test_source_pins_without_imports():
    for name,digest in f.SOURCE_PINS:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    allowed = {'__future__','dataclasses','hashlib','math','struct','time'}
    imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports |= {alias.name.split('.')[0] for node in ast.walk(tree)
                if isinstance(node,ast.Import) for alias in node.names}
    assert imports <= allowed
    assert 'cv2' not in sys.modules
    assert not any(name.endswith('run_camera_mixed_depth') for name in sys.modules)
