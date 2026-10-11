"""Fixed synthetic RGB descriptor falsifier; no native graph or camera claim.

All dependencies are Python standard library. Exact ranks refer to stored IEEE
values. Fixture identities assess outcomes after matching; they never select
neighbors. This module has no execution or file-write side effect on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
import time

MAX_ENDPOINTS = 11
MAX_COMPONENTS = 1_000_000
MAX_EVIDENCE = 16 * 1024 * 1024
MAX_INTEGER_BITS = 4224
NUMERATOR_BYTES = 528
DENOMINATOR_EXPONENT = 2148
MAX_PIXELS = 1024 * 256
SOURCE_PINS = (
    ('src/mc_photogrammetry/pipeline/correspondence.py', '192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1'),
    ('src/mc_photogrammetry/pipeline/joint_matching.py', '9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3'),
    ('src/mc_photogrammetry/pipeline/connected_camera.py', '3b5df735084daddccb7f25511af1b8549c0300d02e4841ade39e39461b8275a6'),
    ('src/mc_photogrammetry/pipeline/image_derived_connections.py', 'a2d166bf711d6cd77c169f044da5ca28d00073e66e60b9bff992572fb56ba211'),
    ('src/mc_photogrammetry/pipeline/contracts.py', 'db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc'),
    ('src/mc_photogrammetry/vision/gui_mask.py', '1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9'),
)
CONTROL_NAMES = (
    'aliases', 'identical_outer', 'zero_energy', 'constant_y',
    'mixed_availability', 'excluded_base', 'excluded_perturbed', 'one_ulp',
    'misleading_permutation', 'opponent_color', 'fractional_border',
    'zero_weight_tap', 'radius_bounds', 'integer_border', 'out_of_frame',
)


class Refusal(ValueError):
    """Fail-closed input, arithmetic or resource violation."""


@dataclass(frozen=True)
class Budget:
    endpoints: int = MAX_ENDPOINTS
    components: int = MAX_COMPONENTS
    evidence: int = MAX_EVIDENCE
    integer_bits: int = MAX_INTEGER_BITS

    def validate(self):
        for value, ceiling in ((self.endpoints, MAX_ENDPOINTS),
                               (self.components, MAX_COMPONENTS),
                               (self.evidence, MAX_EVIDENCE),
                               (self.integer_bits, MAX_INTEGER_BITS)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise Refusal('budget cannot exceed fixed ceiling')


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgb: bytes
    usable: bytes
    points: tuple[tuple[float, float], ...]
    sizes: tuple[float, ...]
    D: bytes
    row_ids: tuple[int, ...]

    def validate(self, budget=Budget()):
        budget.validate()
        if type(self.width) is not int or type(self.height) is not int:
            raise Refusal('dimensions')
        if not (0 < self.width <= 1024 and 0 < self.height <= 256
                and self.width * self.height <= MAX_PIXELS):
            raise Refusal('pixel cap')
        n = len(self.points)
        if not 0 < n <= budget.endpoints:
            raise Refusal('endpoint cap')
        if any(type(x) is not bytes for x in (self.rgb, self.usable, self.D)):
            raise Refusal('immutable byte owners required')
        if not all(type(x) is tuple for x in (self.points, self.sizes, self.row_ids)):
            raise Refusal('immutable row owners required')
        if len(self.rgb) != self.width*self.height*3 or len(self.usable) != self.width*self.height:
            raise Refusal('image shape')
        if any(x not in (0, 1) for x in self.usable):
            raise Refusal('mask bool')
        if len(self.D) != n*128*4 or len(self.sizes) != n or self.row_ids != tuple(range(n)):
            raise Refusal('native row universe')
        if any(type(p) is not tuple or len(p) != 2 or
               any(not math.isfinite(x) for x in p) for p in self.points):
            raise Refusal('point coordinates')
        if any(not math.isfinite(s) or s <= 0 for s in self.sizes):
            raise Refusal('size')
        if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', self.D)):
            raise Refusal('nonfinite original D')


@dataclass(frozen=True)
class Descriptor:
    A: tuple[tuple[float, ...], ...]
    P: tuple[tuple[float, ...], ...]
    B: tuple[tuple[float, ...], ...]
    raw: tuple[tuple[float, ...], ...]
    unit: tuple[tuple[float, ...], ...]
    availability: tuple[str, ...]
    radii: tuple[float, ...]
    footprints: tuple[tuple[int, int, int, int], ...]
    mask_checks: tuple[int, ...]
    first_bad: tuple[tuple[int, int], ...]
    read_domain: bytes
    rgb_taps: int


def describe(frame: Frame, budget=Budget()) -> Descriptor:
    frame.validate(budget)
    raw_rows, units, flags, radii, footprints, counts, bads = [], [], [], [], [], [], []
    read_domain = bytearray(frame.width*frame.height)
    taps = 0
    for (x, y), size in zip(frame.points, frame.sizes):
        r = min(64.0, max(16.0, 2.0*size))
        box = (math.floor(x-r), math.floor(y-r), math.ceil(x+r), math.ceil(y+r))
        radii.append(r)
        footprints.append(box)
        lo_x, lo_y, hi_x, hi_y = box
        raw = [0.0]*48
        bad = (-1, -1)
        count = 0
        if lo_x < 0 or lo_y < 0 or hi_x >= frame.width or hi_y >= frame.height:
            reason = 'out_of_frame'
        else:
            # Finish the entire closed mask rectangle before ANY RGB access.
            for yy in range(lo_y, hi_y+1):
                for xx in range(lo_x, hi_x+1):
                    count += 1
                    if not frame.usable[yy*frame.width+xx] and bad == (-1, -1):
                        bad = (xx, yy)
            reason = 'excluded' if bad != (-1, -1) else 'positive'
            if reason == 'positive':
                # Frozen cell-center grid; row/column, TL/TR/BL/BR, Y/U/V.
                for gy in range(20):
                    py = y+r*((2*gy+1)/20.0-1.0)
                    iy, fy = math.floor(py), py-math.floor(py)
                    for gx in range(20):
                        px = x+r*((2*gx+1)/20.0-1.0)
                        ix, fx = math.floor(px), px-math.floor(px)
                        rr = gg = bb = 0.0
                        for dx, dy, weight in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)),
                                               (0,1,(1-fx)*fy), (1,1,fx*fy)):
                            xx, yy = ix+dx, iy+dy
                            # Even zero-weight taps belong to the footprint.
                            if not (lo_x <= xx <= hi_x and lo_y <= yy <= hi_y):
                                raise Refusal('bilinear tap outside closed footprint')
                            pos = yy*frame.width+xx
                            read_domain[pos] = 1
                            taps += 1
                            rr += weight*frame.rgb[3*pos]
                            gg += weight*frame.rgb[3*pos+1]
                            bb += weight*frame.rgb[3*pos+2]
                        at = ((gy//5)*4+gx//5)*3
                        raw[at] += (rr+2.0*gg+bb)/(4.0*255.0)
                        raw[at+1] += (rr-gg)/255.0
                        raw[at+2] += (bb-gg)/255.0
                raw = [v/25.0 for v in raw]
        energy = 0.0
        for value in raw:
            energy += value*value
        if reason == 'positive' and energy == 0.0:
            reason = 'zero_energy'
        unit = tuple(v/math.sqrt(energy) for v in raw) if energy > 0.0 else (0.0,)*48
        raw_rows.append(tuple(raw)); units.append(unit); flags.append(reason)
        counts.append(count); bads.append(bad)
    d = tuple(tuple(v[0] for v in struct.iter_unpack('<f', frame.D[i*512:(i+1)*512]))
              for i in range(len(frame.points)))
    scale = math.sqrt(3.0)/2.0
    common = tuple(tuple(scale*(v/512.0) for v in row) for row in d)
    p = tuple(row + ((0.5,)+(0.0,)*47 if flag == 'positive' else (0.0,)*48)
              for row, flag in zip(common, flags))
    b = tuple(row + tuple(v/2.0 for v in unit) for row, unit in zip(common, units))
    return Descriptor(d, p, b, tuple(raw_rows), tuple(units), tuple(flags), tuple(radii),
                      tuple(footprints), tuple(counts), tuple(bads), bytes(read_domain), taps)


def dyadic(value: float) -> int:
    """Exact stored binary64 value times2**1074, decoded without FP arithmetic."""
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    sign, exponent, mantissa = bits >> 63, (bits >> 52) & 2047, bits & ((1 << 52)-1)
    if exponent == 2047:
        raise Refusal('nonfinite exact-distance input')
    k = mantissa if exponent == 0 else ((1 << 52) | mantissa) << (exponent-1)
    return -k if sign else k


def exact_squared(a, b, budget=Budget()) -> int:
    budget.validate()
    if len(a) != len(b) or not 0 < len(a) <= 176:
        raise Refusal('descriptor dimension')
    if len(a) > budget.components:
        raise Refusal('component preflight')
    if budget.integer_bits < MAX_INTEGER_BITS:
        # Universal bound checked BEFORE computation, even for small values.
        raise Refusal('integer-width preflight')
    total = 0
    for x, y in zip(a, b):
        delta = dyadic(x)-dyadic(y)
        total += delta*delta
    if total.bit_length() > budget.integer_bits:
        raise Refusal('integer-width overflow')
    return total


def rank_bounds(row, target):
    value = row[target]
    return (1+sum(v < value for i,v in enumerate(row) if i != target),
            1+sum(v <= value for i,v in enumerate(row) if i != target))


def ratio_pass(row):
    values = sorted(row)
    return len(values) >= 2 and 16*values[0] < 9*values[1]


def strict_matches(q):
    edges = []
    for i, row in enumerate(q):
        j = min(range(len(row)), key=row.__getitem__)
        column = tuple(r[j] for r in q)
        if rank_bounds(row,j) == (1,1) and rank_bounds(column,i) == (1,1) \
                and ratio_pass(row) and ratio_pass(column):
            edges.append((i,j))
    return tuple(edges)


@dataclass(frozen=True)
class Blob:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    sha256: str

    @classmethod
    def make(cls, name, dtype, shape, data):
        widths = {'|u1':1, '<f4':4, '<f8':8, '<i4':4, '<u4':4, '<u2':2}
        if dtype not in widths or math.prod(shape)*widths[dtype] != len(data):
            raise Refusal('evidence schema')
        return cls(name,dtype,shape,data,hashlib.sha256(data).hexdigest())


def _packed(rows):
    return b''.join(struct.pack('<'+'d'*len(row), *row) for row in rows)


@dataclass(frozen=True)
class Outcome:
    name: str
    descriptors: tuple[Descriptor, Descriptor]
    distances: tuple[tuple[tuple[int, ...], ...], ...]
    matches: tuple[tuple[tuple[int, int], ...], ...]
    attributed: tuple[tuple[int, int], ...]
    reasons: tuple[tuple[int, int, str], ...]
    intended: tuple[tuple[int, int], ...]
    intended_recovered: int
    blobs: tuple[Blob, ...]
    components: int


def _information(qs, dl, dr):
    a,p,b = qs
    credited, reasons = [], []
    one = 1 << DENOMINATOR_EXPONENT
    def delta_norm(row):
        return abs(sum(dyadic(v)**2 for v in row[128:]) - one//4)
    nl = tuple(delta_norm(row) for row in dl.B)
    nr = tuple(delta_norm(row) for row in dr.B)
    for i,j in strict_matches(b):
        reason = 'observed_context'
        if (i,j) in strict_matches(a) or (i,j) in strict_matches(p):
            reason = 'not_new_reference'
        for matrix_p,matrix_b,query,target,fd,td,norms in (
            (p,b,i,j,dl,dr,nr),
            (tuple(zip(*p)),tuple(zip(*b)),j,i,dr,dl,nl)):
            if fd.availability[query] != 'positive' or td.availability[target] != 'positive':
                reason = 'availability'
                continue
            if rank_bounds(matrix_p[query],target)[1] <= 1:
                reason = 'P_already_unique'
                continue
            rivals = tuple(k for k,v in enumerate(matrix_p[query])
                           if k != target and v <= matrix_p[query][target])
            if not rivals:
                reason = 'no_ambiguous_rivals'
            for rival in rivals:
                if td.availability[rival] != 'positive':
                    reason = 'availability'
                gap_p = matrix_p[query][rival]-matrix_p[query][target]
                gap_b = matrix_b[query][rival]-matrix_b[query][target]
                # Norm-only differences cannot earn spatial information credit.
                if gap_b <= 0 or gap_b-gap_p <= norms[rival]+norms[target]:
                    reason = 'norm_or_unresolved'
        if reason == 'observed_context':
            credited.append((i,j))
        reasons.append((i,j,reason))
    return tuple(credited), tuple(reasons)


def analyze(name, left, right, intended, budget=Budget()):
    left.validate(budget); right.validate(budget)
    n,m = len(left.points),len(right.points)
    components = n*m*480 + (n+m)*48
    if components > budget.components or budget.integer_bits < MAX_INTEGER_BITS:
        raise Refusal('exact-work preflight')
    # Typed capture worst case, checked before descriptor/distance allocation.
    estimate = sum(len(f.rgb)+len(f.usable)+len(f.D)+f.width*f.height+
                   len(f.points)*(8* (128+176+176+48+48+3)+4*9+2)
                   for f in (left,right)) + 3*n*m*(NUMERATOR_BYTES+4) + 16384
    if estimate > budget.evidence:
        raise Refusal('evidence preflight')
    dl,dr = describe(left,budget),describe(right,budget)
    qs = tuple(tuple(tuple(exact_squared(x,y,budget) for y in rr) for x in ll)
               for ll,rr in zip((dl.A,dl.P,dl.B),(dr.A,dr.P,dr.B)))
    matches = tuple(strict_matches(q) for q in qs)
    credited,reasons = _information(qs,dl,dr)
    blobs = []
    for side,f,d in (('L',left,dl),('R',right,dr)):
        count = len(f.points)
        for key,dtype,shape,data in (
            ('rgb','|u1',(f.height,f.width,3),f.rgb),
            ('usable','|u1',(f.height,f.width),f.usable),
            ('D','<f4',(count,128),f.D),
            ('points','<f8',(count,2),_packed(f.points)),
            ('sizes','<f8',(count,),struct.pack('<'+'d'*count,*f.sizes)),
            ('row_ids','<u4',(count,),struct.pack('<'+'I'*count,*f.row_ids)),
            ('A','<f8',(count,128),_packed(d.A)),
            ('P','<f8',(count,176),_packed(d.P)),
            ('B','<f8',(count,176),_packed(d.B)),
            ('raw','<f8',(count,48),_packed(d.raw)),
            ('unit','<f8',(count,48),_packed(d.unit)),
            ('read_domain','|u1',(f.height,f.width),d.read_domain),
            ('radii','<f8',(count,),struct.pack('<'+'d'*count,*d.radii)),
            ('footprints','<i4',(count,4),b''.join(struct.pack('<4i',*x) for x in d.footprints)),
            ('first_bad','<i4',(count,2),b''.join(struct.pack('<2i',*x) for x in d.first_bad)),
            ('mask_checks','<u4',(count,),struct.pack('<'+'I'*count,*d.mask_checks)),
            ('availability','|u1',(count,),bytes(('positive','out_of_frame','excluded','zero_energy').index(x) for x in d.availability)),
        ):
            blobs.append(Blob.make(side+'/'+key,dtype,shape,data))
    blobs.append(Blob.make('Qnum','|u1',(3,n,m,NUMERATOR_BYTES),b''.join(
        value.to_bytes(NUMERATOR_BYTES,'little') for q in qs for row in q for value in row)))
    ranks = b''.join(struct.pack('<2H',*rank_bounds(row,j))
                     for q in qs for matrix in (q,tuple(zip(*q)))
                     for row in matrix for j in range(len(row)))
    blobs.append(Blob.make('ranks','<u2',(3,2,n*m,2),ranks))
    blobs.append(Blob.make('intended','<u4',(len(intended),2),b''.join(struct.pack('<2I',*e) for e in intended)))
    for label,edges in zip(('A','P','B','attributed'),matches+(credited,)):
        blobs.append(Blob.make('edges/'+label,'<u4',(len(edges),2),
                               b''.join(struct.pack('<2I',*e) for e in edges)))
    reason_codes = ('observed_context','not_new_reference','availability',
                    'P_already_unique','no_ambiguous_rivals','norm_or_unresolved')
    blobs.append(Blob.make('reason_ids','<u4',(len(reasons),2),
                           b''.join(struct.pack('<2I',i,j) for i,j,_ in reasons)))
    blobs.append(Blob.make('reason_codes','|u1',(len(reasons),),
                           bytes(reason_codes.index(reason) for _,_,reason in reasons)))
    if sum(len(x.data) for x in blobs) > budget.evidence:
        raise Refusal('evidence cap')
    return Outcome(name,(dl,dr),qs,matches,credited,reasons,intended,
                   len(set(matches[2]) & set(intended)),tuple(blobs),components)


def _frame(patterns, *, constant=None, excluded=(), ulp=False):
    width,height = 48+36*len(patterns),64
    rgb = bytearray(width*height*3)
    centers = tuple((float(24+36*i),32.0) for i in range(len(patterns)))
    outer_cells = (0,1,2,3,4,7,8,11,12,13,14)
    for i,pattern in enumerate(patterns):
        cx,cy = int(centers[i][0]),32
        for dy in range(-16,17):
            for dx in range(-16,17):
                color = (32,32,32)
                cell = min(3,(dy+16)//8)*4+min(3,(dx+16)//8)
                if pattern >= 0 and cell == outer_cells[pattern]:
                    color = (230,55,90)
                if abs(dx)<=4 and abs(dy)<=4:
                    color = (128,128,128)
                if constant is not None:
                    color = constant
                at = ((cy+dy)*width+cx+dx)*3
                rgb[at:at+3] = bytes(color)
    usable = bytearray(b'\1'*(width*height))
    for i in excluded:
        usable[16*width+int(centers[i][0])-16] = 0
    rows = []
    for i in range(len(patterns)):
        row = [32.0]*128
        if ulp and i%2:
            row[0] = struct.unpack('<f',struct.pack('<I',0x42000001))[0]
        rows.append(struct.pack('<128f',*row))
    return Frame(width,height,bytes(rgb),bytes(usable),centers,(8.0,)*len(patterns),
                 b''.join(rows),tuple(range(len(patterns))))


def _small(point, size=8.0, *, excluded_pixel=None, color=(255,0,0)):
    w,h = 145,145
    usable = bytearray(b'\1'*(w*h))
    if excluded_pixel is not None:
        x,y = excluded_pixel; usable[y*w+x] = 0
    return Frame(w,h,bytes(color)*(w*h),bytes(usable),(point,),(size,),
                 struct.pack('<128f',*([32.0]*128)),(0,))


def fixed_packets():
    """Registered immutable packet/control family; no random or adaptive input."""
    ids,reverse = tuple(range(11)),tuple(reversed(range(11)))
    intended = tuple((i,10-i) for i in ids)
    base_l,base_r = _frame(ids),_frame(reverse)
    excluded_l,excluded_r = _frame(ids,excluded=(0,)),_frame(reverse,excluded=(10,))
    def perturb(frame):
        rgb = bytearray(frame.rgb)
        for at,allowed in enumerate(frame.usable):
            if not allowed:
                rgb[3*at:3*at+3] = bytes((255,0,249))
        return Frame(frame.width,frame.height,bytes(rgb),frame.usable,frame.points,
                     frame.sizes,frame.D,frame.row_ids)
    packets = [
        ('aliases',base_l,base_r,intended),
        ('identical_outer',_frame((0,)*11),_frame((0,)*11),intended),
        ('zero_energy',_frame(ids,constant=(0,0,0)),_frame(reverse,constant=(0,0,0)),intended),
        ('constant_y',_frame(ids,constant=(255,255,255)),_frame(reverse,constant=(255,255,255)),intended),
        ('mixed_availability',_frame(ids,constant=(255,255,255),excluded=(0,1,2)),
         _frame(reverse,constant=(255,255,255),excluded=(8,9,10)),intended),
        ('excluded_base',excluded_l,excluded_r,intended),
        ('excluded_perturbed',perturb(excluded_l),perturb(excluded_r),intended),
        ('one_ulp',_frame(ids,constant=(255,255,255),ulp=True),
         _frame(reverse,constant=(255,255,255),ulp=True),intended),
        ('misleading_permutation',base_l,_frame(tuple((x+1)%11 for x in reverse)),intended),
    ]
    small = (
        ('opponent_color',_small((72.0,72.0)),_small((72.0,72.0))),
        ('fractional_border',_small((16.25,16.25)),_small((16.25,16.25))),
        # r=20 centers include exact-integer taps; (1,1) has zero weight at (1,1).
        ('zero_weight_tap',_small((20.0,20.0),size=10.0,excluded_pixel=(2,1)),
         _small((20.0,20.0),size=10.0,excluded_pixel=(2,1))),
        ('radius_bounds',_small((72.0,72.0),size=32.0),_small((72.0,72.0),size=33.0)),
        ('integer_border',_small((16.0,16.0)),_small((16.0,16.0))),
        ('out_of_frame',_small((15.999,16.0)),_small((15.999,16.0))),
    )
    packets.extend((name,l,r,((0,0),)) for name,l,r in small)
    return tuple(packets)


@dataclass(frozen=True)
class Result:
    status: str
    interpretation: str
    outcomes: tuple[Outcome, ...]
    failures: tuple[str, ...]
    components: int
    evidence_bytes: int
    elapsed_seconds: float
    source_pins: tuple[tuple[str,str], ...]


def run_fixed(budget=Budget()):
    budget.validate()
    started = time.perf_counter()
    packets = fixed_packets()
    if tuple(p[0] for p in packets) != CONTROL_NAMES or len(packets)>16:
        raise Refusal('fixed control registry')
    components = sum(len(l.points)*len(r.points)*480+(len(l.points)+len(r.points))*48
                     for _,l,r,_ in packets)
    if components > budget.components or budget.integer_bits < MAX_INTEGER_BITS:
        raise Refusal('package work preflight')
    # Conservative typed payload bound for the entire retained package.
    estimate = sum(sum(len(f.rgb)+len(f.usable)+len(f.D)+f.width*f.height+
                       len(f.points)*(8* (128+176+176+48+48+3)+4*9+2)
                       for f in (l,r)) + 3*len(l.points)*len(r.points)*(NUMERATOR_BYTES+4)+16384
                   for _,l,r,_ in packets)
    if estimate > budget.evidence:
        raise Refusal('package evidence preflight')
    outcomes = tuple(analyze(name,l,r,intended,budget) for name,l,r,intended in packets)
    by_name = {o.name:o for o in outcomes}
    failures = []
    aliases = by_name['aliases']
    if aliases.matches[0] or aliases.matches[1] or aliases.matches[2] != aliases.intended \
            or aliases.attributed != aliases.intended or aliases.intended_recovered != 11:
        failures.append('alias_information')
    for name in ('identical_outer','zero_energy','constant_y','mixed_availability','one_ulp'):
        if by_name[name].attributed:
            failures.append(name+'_false_information')
    eb,ep = by_name['excluded_base'],by_name['excluded_perturbed']
    if eb.descriptors != ep.descriptors or eb.distances != ep.distances \
            or eb.matches != ep.matches or eb.attributed != ep.attributed:
        failures.append('excluded_domain_changed_information')
    misleading = by_name['misleading_permutation']
    if len(misleading.matches[2]) != 11 or len(misleading.attributed) != 11 \
            or misleading.intended_recovered != 0:
        failures.append('permutation_semantics')
    if by_name['zero_weight_tap'].descriptors[0].availability != ('excluded',):
        failures.append('zero_weight_permission')
    if by_name['out_of_frame'].descriptors[0].availability != ('out_of_frame',):
        failures.append('border_permission')
    if by_name['radius_bounds'].descriptors[0].radii != (64.0,) \
            or by_name['radius_bounds'].descriptors[1].radii != (64.0,):
        failures.append('radius_clamp')
    evidence_bytes = sum(len(blob.data) for o in outcomes for blob in o.blobs)
    if evidence_bytes > budget.evidence:
        raise Refusal('retained evidence cap')
    return Result('NO_GO' if failures else 'PURE_PASS',
                  'REFERENCE_ONLY; no native graph absence, pose, accuracy or production readiness',
                  outcomes,tuple(failures),components,evidence_bytes,
                  time.perf_counter()-started,SOURCE_PINS)
