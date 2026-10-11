"""Pure, evaluation-only VALID-NATIVE-CHAIN-01 acquisition.

No OpenCV, renderer, calibration, camera or truth dependencies. All caps are
checked before managed buffer growth. Resource/integrity errors are terminal;
an empty finite extraction is an ordinary outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np


class AcquisitionTerminalError(RuntimeError):
    def __init__(self, reason, evidence=None):
        self.evidence = dict(evidence or {})
        super().__init__(reason)


class AcquisitionCapError(AcquisitionTerminalError):
    pass


class AcquisitionIntegrityError(AcquisitionTerminalError):
    pass


@dataclass(frozen=True)
class FrozenConfig:
    identifier: str = "VALID-NATIVE-CHAIN-01"
    max_dimension: int = 2048
    max_pixels: int = 1_048_576
    max_edgels: int = 65_536
    max_edges: int = 262_144
    max_neighbor_probes: int = 524_288
    max_groups: int = 65_536
    max_path_vertices: int = 8192
    max_fit_nodes: int = 131_072
    max_depth: int = 32
    max_fit_points: int = 16_777_216
    max_candidates: int = 10_000
    max_projection_samples: int = 8_000_000
    max_runs: int = 131_072
    max_ledger_bytes: int = 16 * 1024 * 1024
    max_raw_bytes: int = 32 * 1024 * 1024
    max_metadata_bytes: int = 8 * 1024 * 1024
    max_member_bytes: int = 64 * 1024 * 1024
    max_live_bytes: int = 128 * 1024 * 1024

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


FROZEN_CONFIG = FrozenConfig()


def validate_config(config):
    if type(config) is not FrozenConfig or config.identifier != FROZEN_CONFIG.identifier:
        raise AcquisitionIntegrityError("invalid_frozen_config")
    # Lower caps enable deterministic pure failure injection; never relax caps.
    for name, default in asdict(FROZEN_CONFIG).items():
        if name == "identifier":
            continue
        value = getattr(config, name)
        if type(value) is not int or not 0 <= value <= default:
            raise AcquisitionIntegrityError("invalid_cap:" + name)


class AllocationBudget:
    """Managed byte leases, including scratch/encoding reservations.

    This is deliberately not a process-RSS limit. Opaque library internals are
    outside the guarantee; callers must reserve instrumentation explicitly.
    """
    def __init__(self, limit):
        self.limit, self.live, self.peak = int(limit), 0, 0
        self.packed_live, self.packed_limit = 0, FROZEN_CONFIG.max_ledger_bytes

    def acquire(self, size, reason="live_allocation_cap"):
        size = int(size)
        if size < 0:
            raise AcquisitionIntegrityError("negative_allocation")
        if self.live + size > self.limit:
            raise AcquisitionCapError(reason, {"live": self.live, "requested": size})
        self.live += size
        self.peak = max(self.peak, self.live)
        return size

    def release(self, size):
        self.live -= int(size)
        if self.live < 0:
            raise AcquisitionIntegrityError("allocation_lease_underflow")

    def array(self, shape, dtype, *, fill=None):
        dtype = np.dtype(dtype)
        count = math.prod(shape) if isinstance(shape, tuple) else int(shape)
        self.acquire(count * dtype.itemsize + 256)
        result = np.empty(shape, dtype=dtype)
        if fill is not None:
            result.fill(fill)
        return result


class PackedBuffer:
    """Fixed-width geometric-growth buffer; no Python-object record lists."""
    def __init__(self, dtype, maximum, budget, label):
        self.dtype, self.maximum = np.dtype(dtype), int(maximum)
        self.budget, self.label = budget, label
        self.data = None
        self.count = 0

    def append(self, value):
        if self.count >= self.maximum:
            raise AcquisitionCapError(self.label, {"count": self.count})
        if self.data is None or self.count == len(self.data):
            capacity = min(self.maximum, max(16, 2 * self.count))
            previous = self.data
            old_bytes = 0 if previous is None else previous.nbytes
            new_bytes = capacity*self.dtype.itemsize
            if self.budget.packed_live-old_bytes+new_bytes > self.budget.packed_limit:
                raise AcquisitionCapError("ledger_byte_cap")
            self.budget.packed_live += new_bytes-old_bytes
            grown = self.budget.array(capacity, self.dtype)
            if previous is not None:
                grown[:self.count] = previous[:self.count]
            self.data = grown
            if previous is not None:
                old_allocation=previous.nbytes+256
                previous=None
                self.budget.release(old_allocation)
        self.data[self.count] = value
        self.count += 1

    def view(self):
        if self.data is None:
            result = self.budget.array(0, self.dtype)
        else:
            result = self.data[:self.count].view()
        result.setflags(write=False)
        return result


EDGEL = np.dtype([("x", "<u2"), ("y", "<u2"), ("gx", "<i2"),
                  ("gy", "<i2"), ("m", "<u4")])
EDGE = np.dtype([("a", "<u4"), ("b", "<u4")])
GROUP = np.dtype([("path", "<u4"), ("start", "<u4"), ("stop", "<u4"),
                  ("parent", "<i4"), ("depth", "<u4"), ("reason", "u1")])
PATH = np.dtype([("offset", "<u4"), ("count", "<u4")])
LINE = np.dtype([("node", "<u4"), ("offset", "<u4"), ("count", "<u4"),
                 ("score", "<u8"), ("residual", "<f8"), ("reason", "u1"),
                 ("sample_offset", "<u4"), ("samples", "<u4")])
# Footprints: frame/original/band/valid. Occupancy: frame/original/band/valid.
# Group reason: 0 split, 1 terminal, 2 singleton, 3 isotropic/zero extent.
# Line reason: 0 admitted, 1 duplicate, 2 frame, 3 original, 4 band, 5 short.


@dataclass(frozen=True)
class ExtractionResult:
    segments: np.ndarray
    pre_length: np.ndarray
    arrays: dict
    counts: dict
    config_sha256: str
    live_bytes: int
    peak_bytes: int


def check_input(image, original, expanded, config=FROZEN_CONFIG, *, rgb=False):
    validate_config(config)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or not image.flags.c_contiguous:
        raise AcquisitionIntegrityError("uint8_contiguous_image_required")
    wanted = 3 if rgb else 2
    if image.ndim != wanted or (rgb and image.shape[2] != 3):
        raise AcquisitionIntegrityError("image_shape")
    h, w = image.shape[:2]
    if h < 1 or w < 1 or max(h, w) > config.max_dimension or h*w > config.max_pixels:
        raise AcquisitionCapError("input_pixel_or_dimension_cap", {"height": h, "width": w})
    for mask in (original, expanded):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (h, w) or not mask.flags.c_contiguous:
            raise AcquisitionIntegrityError("bool_contiguous_mask_required")
    # Scalar scan: no H*W implicit boolean comparison scratch.
    for y in range(h):
        for x in range(w):
            if original[y, x] and not expanded[y, x]:
                raise AcquisitionIntegrityError("expanded_mask_not_superset")
    return h, w


def pointwise_gray(rgb, *, budget=None, config=FROZEN_CONFIG):
    validate_config(config)
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or not rgb.flags.c_contiguous):
        raise AcquisitionIntegrityError("uint8_contiguous_rgb_required")
    h, w = rgb.shape[:2]
    if not h or not w or max(h, w) > config.max_dimension or h*w > config.max_pixels:
        raise AcquisitionCapError("input_pixel_or_dimension_cap")
    budget = budget or AllocationBudget(config.max_live_bytes)
    result = budget.array((h, w), np.uint8)
    # Scalar widening prevents uint8 overflow and avoids uncharged image copies.
    for y in range(h):
        for x in range(w):
            r, g, b = (int(v) for v in rgb[y, x])
            result[y, x] = (77*r + 150*g + 29*b + 128) // 256
    result.setflags(write=False)
    return result


def _sobel(gray, y, x):
    gx = (int(gray[y-1,x+1]) + 2*int(gray[y,x+1]) + int(gray[y+1,x+1])
          - int(gray[y-1,x-1]) - 2*int(gray[y,x-1]) - int(gray[y+1,x-1]))
    gy = (int(gray[y+1,x-1]) + 2*int(gray[y+1,x]) + int(gray[y+1,x+1])
          - int(gray[y-1,x-1]) - 2*int(gray[y-1,x]) - int(gray[y-1,x+1]))
    return gx, gy


def _sector(gx, gy):
    a, b = abs(gx), abs(gy)
    if 1_000_000*b <= 414_214*a:
        return 1, 0
    if 1_000_000*a <= 414_214*b:
        return 0, 1
    return 1, 1 if gx*gy >= 0 else -1


def _compatible(first, second, dx, dy):
    ax, ay, bx, by = (int(first["gx"]), int(first["gy"]),
                       int(second["gx"]), int(second["gy"]))
    am, bm = int(first["m"]), int(second["m"])
    cross = ax*by-ay*bx
    if 1_000_000*cross*cross > 146_447*am*bm:
        return False
    distance = dx*dx+dy*dy
    return (1_000_000*(dx*ax+dy*ay)**2 <= 853_554*distance*am
            and 1_000_000*(dx*bx+dy*by)**2 <= 853_554*distance*bm)


def _freeze(array):
    array.setflags(write=False)
    return array


def sample_occupancy(row, original, expanded, budget, maximum, *, endpoint_domain=False):
    """Exact native R3 half-pixel sampling. Category array is uint8 scratch."""
    h, w = original.shape
    x0,y0,x1,y1 = (float(v) for v in row)
    if not all(math.isfinite(v) for v in (x0,y0,x1,y1)):
        raise AcquisitionIntegrityError("nonfinite_segment")
    length = math.hypot(x1-x0, y1-y0)
    n = max(2, math.ceil(length/0.5)+1)
    if n > maximum:
        raise AcquisitionCapError("sample_cap", {"required": n, "maximum": maximum})
    # Original linspace/coordinate rounding; reserve every live expression temp.
    lease = budget.acquire(n*80 + 4096)
    parameters=xs=ys=columns=rows=None
    try:
        parameters = np.linspace(0., 1., n, dtype=np.float64)
        xs, ys = x0 + parameters*(x1-x0), y0 + parameters*(y1-y0)
        columns = np.floor(xs+0.5).astype(np.int64)
        rows = np.floor(ys+0.5).astype(np.int64)
        categories = budget.array(n, np.uint8, fill=0)
        for i in range(n):
            y, x = int(rows[i]), int(columns[i])
            if 0 <= x < w and 0 <= y < h:
                categories[i] = 1 if original[y,x] else (2 if expanded[y,x] else 3)
        outside_end = endpoint_domain and not (0 <= x0 <= w-1 and 0 <= x1 <= w-1
                                               and 0 <= y0 <= h-1 and 0 <= y1 <= h-1)
        return categories, bool(outside_end), length
    finally:
        parameters=xs=ys=columns=rows=None
        budget.release(lease)


def pack_categories(values, packed, count):
    for value in values:
        if count % 4 == 0:
            packed.append(0)
        packed.data[count//4] |= int(value) << (2*(count % 4))
        count += 1
    return count


class _ProjectionState:
    __slots__=("low","high","residual","split","last","direction","reversal")
    def __init__(self):
        self.low,self.high,self.residual=math.inf,-math.inf,-1.
        self.split=self.last=self.reversal=None
        self.direction=0

    def add(self,t,r,k):
        if not math.isfinite(t) or not math.isfinite(r):
            raise AcquisitionIntegrityError("nonfinite_fit")
        self.low,self.high=min(self.low,t),max(self.high,t)
        if r>self.residual:
            self.residual,self.split=r,k
        if self.last is not None:
            delta=t-self.last
            sign=(delta>0)-(delta<0)
            if sign and not self.direction:
                self.direction=sign
            elif sign and sign!=self.direction and self.reversal is None:
                self.reversal=k-1
        self.last=t


def _progress_cut(start,stop,reversal,split):
    cut=reversal if reversal is not None else split
    if cut<=start or cut>=stop-1:
        cut=start+(stop-start)//2
    if not start<cut<stop-1:
        raise AcquisitionIntegrityError("fit_nonshrinking_split")
    return cut

def _trace_support_graph(neighbors,degree,budget,config,counts):
    visited = budget.array((len(degree),8), np.bool_, fill=False)
    paths = PackedBuffer(PATH, config.max_groups, budget, "group_cap")
    ids = PackedBuffer("<u4", 2*config.max_edges+config.max_edgels, budget, "path_membership_cap")
    closures = PackedBuffer(EDGE, config.max_groups, budget, "cycle_closure_cap")

    def mark(a,b):
        ia = next(k for k in range(int(degree[a])) if int(neighbors[a,k]) == b)
        ib = next(k for k in range(int(degree[b])) if int(neighbors[b,k]) == a)
        visited[a,ia] = visited[b,ib] = True

    def trace_path(start, nxt):
        if paths.count >= config.max_groups:
            raise AcquisitionCapError("group_cap", counts)
        offset, length = ids.count, 0
        def put(vertex):
            nonlocal length
            if length >= config.max_path_vertices:
                raise AcquisitionCapError("path_vertex_cap", counts)
            ids.append(vertex)
            length += 1
        put(start)
        previous, current = start, nxt
        mark(start,nxt)
        while True:
            put(current)
            if int(degree[current]) != 2:
                break
            following = int(neighbors[current,0])
            if following == previous:
                following = int(neighbors[current,1])
            slot = next(k for k in range(2) if int(neighbors[current,k]) == following)
            if visited[current,slot]:
                break
            mark(current,following)
            previous,current = current,following
        paths.append((offset,length))

    for i in range(len(degree)):
        if degree[i] == 0:
            if paths.count >= config.max_groups:
                raise AcquisitionCapError("group_cap", counts)
            offset = ids.count
            ids.append(i)
            paths.append((offset,1))
            counts["singletons"] += 1
        elif degree[i] != 2:
            for k in range(int(degree[i])):
                if not visited[i,k]:
                    trace_path(i,int(neighbors[i,k]))
    # Every remaining component is a pure cycle. Its first unvisited edge is
    # canonical smallest because vertices and neighbors are sorted.
    for i in range(len(degree)):
        for k in range(int(degree[i])):
            if not visited[i,k]:
                j = int(neighbors[i,k])
                closures.append((i,j))
                mark(i,j)
                other = next(int(neighbors[i,z]) for z in range(int(degree[i])) if not visited[i,z])
                trace_path(i,other)
    counts["paths"], counts["cycle_closures"] = paths.count, closures.count
    # Graph edge universe: edges traversed by paths + deliberately removed edges.
    traversed = sum(int(row["count"])-1 for row in paths.view())
    if traversed + closures.count != sum(int(v) for v in degree)//2:
        raise AcquisitionIntegrityError("graph_edge_accounting", counts)
    del mark,trace_path
    visited_bytes=visited.nbytes+256
    del visited
    budget.release(visited_bytes)
    return paths,ids,closures

def _rank_and_deduplicate(cv,lv,support,budget):
    # Python sort operates only on candidate IDs, with a conservative object/
    # TimSort reservation; graph/path/ledger records remain typed buffers.
    sorting_lease=budget.acquire(len(cv)*1024+4096)
    try:
        order=sorted(range(len(cv)),key=lambda i:(-int(lv[i]["count"]),-int(lv[i]["score"]),
                    float(lv[i]["residual"]),tuple(float(v) for v in cv[i]),int(lv[i]["node"])))
    except Exception:
        budget.release(sorting_lease)
        raise
    # Keep order/rank container charge until all candidate processing ends.
    signature_bytes=sum(int(row["count"])*40+128 for row in lv)
    signature_lease=budget.acquire(signature_bytes + len(cv)*1024+4096)
    try:
        signatures=sorted(order,key=lambda i:tuple(int(v) for v in support[
            int(lv[i]["offset"]):int(lv[i]["offset"])+int(lv[i]["count"])]))
        duplicate=budget.array(len(cv),np.bool_,fill=False)
        last_id=None
        for i in signatures:
            if last_id is not None:
                a,b=lv[last_id],lv[i]
                if int(a["count"]) == int(b["count"]):
                    ao,bo,n=int(a["offset"]),int(b["offset"]),int(a["count"])
                    # Scalar equality avoids an uncharged signature scratch.
                    if all(support[ao+k] == support[bo+k] for k in range(n)):
                        duplicate[i]=True
                        continue
            last_id=i
        del signatures
    finally:
        budget.release(signature_lease)
    return order,duplicate,sorting_lease


def extract_segments(gray_u8, excluded_bool, *, original_mask=None,
                     frozen_config=FROZEN_CONFIG, budget=None):
    """Native acquisition; only caps can vary downward for pure unit tests."""
    config = frozen_config
    original = excluded_bool if original_mask is None else original_mask
    h, w = check_input(gray_u8, original, excluded_bool, config)
    owned_budget = budget is None
    budget = budget or AllocationBudget(config.max_live_bytes)
    budget.packed_limit = min(budget.packed_limit, config.max_ledger_bytes)
    start_bytes = budget.live
    # Account caller-owned numeric inputs as part of candidate live footprint.
    budget.acquire(gray_u8.nbytes + original.nbytes + excluded_bool.nbytes + 16384)
    counts = {"footprint_frame": 0, "footprint_original": 0, "footprint_band": 0,
              "footprint_valid": 0, "below_threshold": 0, "nms_suppressed": 0,
              "edgels": 0, "neighbor_probes": 0, "graph_edges": 0,
              "paths": 0, "singletons": 0, "cycle_closures": 0,
              "fit_nodes": 0, "fit_points": 0, "degenerate_groups": 0,
              "pre_length": 0, "duplicate": 0, "frame": 0, "original": 0,
              "band": 0, "short": 0, "admitted": 0, "projection_samples": 0}
    edgels = PackedBuffer(EDGEL, config.max_edgels, budget, "edgel_cap")
    # Index map determines graph support identities; no invalid RGB enters it.
    index = budget.array((h,w), "<i4", fill=-1)
    for y in range(h):
        for x in range(w):
            if x < 2 or y < 2 or x+2 >= w or y+2 >= h:
                counts["footprint_frame"] += 1
                continue
            original_hit, band_hit = False, False
            for yy in range(y-2,y+3):
                for xx in range(x-2,x+3):
                    original_hit |= bool(original[yy,xx])
                    band_hit |= bool(excluded_bool[yy,xx])
            if original_hit or band_hit:
                counts["footprint_original" if original_hit else "footprint_band"] += 1
                continue
            counts["footprint_valid"] += 1
            gx, gy = _sobel(gray_u8,y,x)
            magnitude = gx*gx+gy*gy
            if magnitude < 4096:
                counts["below_threshold"] += 1
                continue
            dx,dy = _sector(gx,gy)
            mx,my = _sobel(gray_u8,y-dy,x-dx)
            px,py = _sobel(gray_u8,y+dy,x+dx)
            if magnitude < mx*mx+my*my or magnitude <= px*px+py*py:
                counts["nms_suppressed"] += 1
                continue
            index[y,x] = edgels.count
            edgels.append((x,y,gx,gy,magnitude))
    ev = edgels.view()
    counts["edgels"] = len(ev)
    edges = PackedBuffer(EDGE, config.max_edges, budget, "graph_edge_cap")
    # Degree <=8 and neighbors sorted by row-major identity.
    neighbors = budget.array((len(ev),8), "<i4", fill=-1)
    degree = budget.array(len(ev), np.uint8, fill=0)
    for i, first in enumerate(ev):
        x,y = int(first["x"]), int(first["y"])
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx == dy == 0:
                    continue
                if counts["neighbor_probes"] >= config.max_neighbor_probes:
                    raise AcquisitionCapError("neighbor_probe_cap", counts)
                counts["neighbor_probes"] += 1
                nx,ny = x+dx,y+dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                j = int(index[ny,nx])
                if j <= i:
                    continue
                if dx and dy and (index[y,nx] >= 0 or index[ny,x] >= 0):
                    continue
                if not _compatible(first,ev[j],dx,dy):
                    continue
                edges.append((i,j))
                neighbors[i,int(degree[i])] = j
                neighbors[j,int(degree[j])] = i
                degree[i] += 1
                degree[j] += 1
    gv = edges.view()
    counts["graph_edges"] = len(gv)
    for i in range(len(ev)):
        # In-place sort of at most eight IDs; no materialized graph objects.
        neighbors[i,:int(degree[i])].sort()
    paths,ids,closures=_trace_support_graph(neighbors,degree,budget,config,counts)
    groups = PackedBuffer(GROUP, config.max_fit_nodes, budget, "fit_node_cap")
    coordinates = PackedBuffer(np.dtype(("<f8",(4,))), config.max_candidates, budget, "candidate_cap")
    lines = PackedBuffer(LINE, config.max_candidates, budget, "candidate_cap")
    support = PackedBuffer("<u4", config.max_fit_points, budget, "support_membership_cap")
    stack = budget.array((config.max_depth+2,5), "<i8")
    # Rows in explicit bounded stack: path, range start/stop, parent node, depth.
    for path_index, path in enumerate(paths.view()):
        offset,size = int(path["offset"]),int(path["count"])
        stack[0] = (path_index,offset,offset+size,-1,0)
        top = 1
        while top:
            top -= 1
            path_id,start,stop,parent,depth = (int(v) for v in stack[top])
            n = stop-start
            if depth > config.max_depth:
                raise AcquisitionCapError("fit_depth_cap", counts)
            if groups.count >= config.max_fit_nodes:
                raise AcquisitionCapError("fit_node_cap", counts)
            if counts["fit_points"] + 2*n > config.max_fit_points:
                raise AcquisitionCapError("fit_point_cap", counts)
            counts["fit_points"] += 2*n
            counts["fit_nodes"] += 1
            node = groups.count
            if n < 2:
                groups.append((path_id,start,stop,parent,depth,2))
                continue
            sx=sy=sxx=sxy=syy=0
            score=0
            for k in range(start,stop):
                e = ev[int(ids.data[k])]
                x,y = int(e["x"]),int(e["y"])
                sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; syy+=y*y
                score+=int(e["m"])
            cx,cy = sx/n,sy/n
            xx,xy,yy = float(sxx)-sx*cx,float(sxy)-sx*cy,float(syy)-sy*cy
            if xx == yy and xy == 0:
                groups.append((path_id,start,stop,parent,depth,3))
                counts["degenerate_groups"] += 1
                continue
            angle = 0.5*math.atan2(2*xy,xx-yy)
            ux,uy = math.cos(angle),math.sin(angle)
            if ux < 0 or (ux == 0 and uy < 0):
                ux,uy = -ux,-uy
            state=_ProjectionState()
            for k in range(start,stop):
                e = ev[int(ids.data[k])]
                x,y = float(e["x"])-cx,float(e["y"])-cy
                state.add(x*ux+y*uy,abs(-x*uy+y*ux),k)
            low,high,residual=state.low,state.high,state.residual
            reversal,split=state.reversal,state.split
            if high <= low:
                groups.append((path_id,start,stop,parent,depth,3))
                counts["degenerate_groups"] += 1
                continue
            if n > 2 and (reversal is not None or residual > 0.75):
                cut=_progress_cut(start,stop,reversal,split)
                if depth+1 > config.max_depth or top+2 > len(stack):
                    raise AcquisitionCapError("fit_depth_cap", counts)
                groups.append((path_id,start,stop,parent,depth,0))
                stack[top]=(path_id,cut,stop,node,depth+1)
                stack[top+1]=(path_id,start,cut+1,node,depth+1)
                top+=2
                continue
            row=(cx+low*ux,cy+low*uy,cx+high*ux,cy+high*uy)
            if not all(math.isfinite(v) for v in row):
                raise AcquisitionIntegrityError("nonfinite_fit", counts)
            if coordinates.count >= config.max_candidates:
                raise AcquisitionCapError("candidate_cap", counts)
            # Reserve before materializing this sorted exact-support signature.
            signature_lease=budget.acquire(n*4+256)
            signature=np.empty(n,dtype="<u4")
            try:
                signature[:]=ids.data[start:stop]
                signature.sort()
                support_offset=support.count
                for item in signature:
                    support.append(item)
            finally:
                del signature
                budget.release(signature_lease)
            coordinates.append(row)
            lines.append((node,support_offset,n,score,residual,0,0,0))
            groups.append((path_id,start,stop,parent,depth,1))
    cv=coordinates.view()
    lv=lines.view()
    order,duplicate,sorting_lease=_rank_and_deduplicate(cv,lv,support.data,budget)
    emitted=PackedBuffer(np.dtype(("<f8",(4,))),config.max_candidates,budget,"detector_input_cap")
    admitted_ids=PackedBuffer("<u4",config.max_candidates,budget,"detector_input_cap")
    packed=PackedBuffer(np.uint8,(config.max_projection_samples+3)//4,budget,"projection_sample_cap")
    sample_count=0
    # Writable line data is private until freeze below; record original row IDs.
    for i in order:
        row=cv[i]
        categories,outside,length=sample_occupancy(row,original,excluded_bool,budget,
                          config.max_projection_samples-sample_count,endpoint_domain=True)
        try:
            lines.data[i]["sample_offset"]=sample_count
            lines.data[i]["samples"]=len(categories)
            sample_count=pack_categories(categories,packed,sample_count)
            reason=1 if duplicate[i] else (2 if outside or any(v == 0 for v in categories)
                    else (3 if any(v == 1 for v in categories) else
                          (4 if any(v == 2 for v in categories) else (5 if length < 40. else 0))))
            lines.data[i]["reason"]=reason
            counts[("admitted","duplicate","frame","original","band","short")[reason]]+=1
            if reason == 0:
                emitted.append(row)
                admitted_ids.append(i)
        finally:
            category_bytes=categories.nbytes+256
            del categories
            budget.release(category_bytes)
    counts["projection_samples"],counts["pre_length"]=sample_count,len(cv)
    if sum(counts[k] for k in ("admitted","duplicate","frame","original","band","short")) != len(cv):
        raise AcquisitionIntegrityError("line_accounting",counts)
    if sum(counts[k] for k in ("footprint_frame","footprint_original","footprint_band","footprint_valid")) != h*w:
        raise AcquisitionIntegrityError("footprint_accounting",counts)
    arrays={"edgels":ev,"graph_edges":gv,"paths":paths.view(),"path_membership":ids.view(),
            "cycle_closures":closures.view(),"fit_groups":groups.view(),
            "support_membership":support.view(),"line_ledger":lines.view(),
            "projection_occupancy":packed.view(),"admitted_candidate_ids":admitted_ids.view()}
    if sum(arr.nbytes for arr in arrays.values()) > config.max_ledger_bytes:
        raise AcquisitionCapError("ledger_byte_cap",counts)

    # Scratch no longer retained by result; lease accounting returns to live
    # buffer ownership. Buffer capacities remain charged (views retain bases).
    scratch_bytes=sum(v.nbytes+256 for v in (index,neighbors,degree,stack,duplicate))
    del index,neighbors,degree,stack,duplicate
    budget.release(scratch_bytes)
    if not owned_budget:
        budget.release(gray_u8.nbytes+original.nbytes+excluded_bool.nbytes+16384)
    del order
    budget.release(sorting_lease)
    return ExtractionResult(emitted.view().reshape((-1,4)),cv.reshape((-1,4)),arrays,counts,
                            config.sha256,budget.live-start_bytes,budget.peak)






