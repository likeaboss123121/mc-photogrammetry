"""Bounded evaluator harness. Native entry is CLOSED; only injected fake tests.

No cv2/NumPy import, native backend, attestor or execution authorization exists.
A synthetic receipt is orchestration evidence, never native numerical evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import hashlib
import importlib.util
import json
import os
import secrets
from pathlib import Path
import struct
import subprocess
import sys
from threading import Lock, Thread

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path(r"C:/Users/Likea/AppData/Local/Temp/mc-photo-opencv5-cpu")
SYMBOLIC_PATH = ROOT / "src/mc_photogrammetry/reconstruction/evaluation_remap_support_v2.py"
SYMBOLIC_SHA = "7f332788762ec308ad396aa3c8c82a4364b81ccd09d6311200824ce0c844769d"
PRODUCER_REPORT_SHA = "f9b854cb06affc350c383b2f6ef1a48d7885ccc73817ea388710b91d78a36056"
MAX_WIRE_BYTES = 16 << 20
MAX_PARENT_BYTES = 64 << 20
PARENT_SCRATCH_BYTES = 8 << 20
MAX_METADATA_BYTES = 64 << 10
MAX_CASES = 768
TIMEOUT_SECONDS = 30
NATIVE_ATTESTOR_PINS = ()  # No actual FP/selected-entry provider has been admitted.
PINNED_FILES = (
    ("provenance/final-artifacts-sha256.json", "46d3b51cd4b3ea2efa2592e47b3204ae079652ef9ac7a6c3cb31d5f38ffe3421"),
    ("provenance/precompile-gates.json", "4db1525235cab2261a30a29e09e4fdd19ed9fc4415bbeb0cff19aa33482cb077"),
    ("provenance/generated-static-gates.json", "d2c33f2d5807baa20d56295cd8f58db58f19bc5bb5738edc19fe0995dd0aed57"),
    ("provenance/built-binary-gates.json", "d1b950a3f1ee3fee85089d350c98fba36e591bbda7043dff3a81f5c84305d2bc"),
    ("provenance/import-probe.json", "e2f86db59653cd4c607f4467f5cd320b04bfb5003b2e814956204683fbc8199b"),
    ("provenance/reference-build-information.txt", "d7449aeff4c79db372adbe3bc778a0dde2c5d218aafbeca57ac2c57860eccc68"),
    ("build-clean/CMakeCache.txt", "7932fade553a41cf4f11acdb9fb7cfc157be53d7c880e7662766d62bef241bbb"),
    ("build-clean/cvconfig.h", "36b3f6b4f3a56625322b53ec4648fbb0f6ff4a7e17117bf9c8e451701f38ee68"),
    ("build-clean/cv_cpu_config.h", "4e9cda34d3df32950977d63b30b3a921847ce0aac7bbff83f781614a3c61d3ea"),
    ("build-clean/opencv_python_config.cmake", "d0cd0958eb88909562d202f158b87887120b816afd823ef29850158baabd27a0"),
    ("build-clean/compile_commands.json", "f7b7540b3afff8edea808dee333c0d4d43640aeedfc95a2a2f7e92e047d24280"),
    ("build-clean/python_loader/cv2/config-3.10.py", "ed08d9584d5d3f26d9d06f66c961acdac5eed47e373ef2e8c6de1c6be74ed5de"),
    ("build-clean/lib/python3/cv2.cp310-win_amd64.pyd", "15e8c9f063076e4e6c07bbba053b0ce02fcfb201eff35145ce2395aae686e4e0"),
    ("build-clean/bin/opencv_core500.dll", "2149d6ca8914330ac9d2ebfbbade70af4e9d58f6a427fb8010664fd45fb55159"),
    ("build-clean/bin/opencv_imgproc500.dll", "6a7be1e21436a9f909eed1dfdf12f9277fe472010de51b82108537cbae8d1276"),
    ("build-clean/bin/opencv_geometry500.dll", "df7263c34d850e58390d80618de97bfa7b1a62b90c47c617bfb34ffa2bc415be"),
    ("build-clean/bin/opencv_flann500.dll", "9b8c6315a0ab8fab1007ebb927928402d0007d9ebb44ffc5d9eea27ad336c314"),
    ("source/modules/imgproc/src/imgwarp.cpp", "b7a95e4bb5c2c5adc5378c43151fa4bc6e461ec3bb267d9d72eda91a004595ae"),
    ("source/modules/imgproc/src/warp_kernels.simd.hpp", "3e19fdf116c99babd38775f543301276db59ae8bbfa50edaaba7afe89c1a7c73"),
    ("source/modules/imgproc/src/warp_common.scalar.hpp", "3da7138a38c611aedd4f4805b35f35cde9e5e038107bedce04c4295fe1f35af6"),
    ("source/modules/imgproc/src/warp_common.vector.hpp", "ca5314ca884189cf790afc4d14ef121f3fd30f7b25bf3054f977cef81160df21"),
)
REFERENCE_PYTHON = Path(r"C:/Users/Likea/AppData/Local/Programs/Python/Python310/python.exe")
REFERENCE_PYTHON_SHA = "3cce33d75d6fdae4e004d0bdf149320b3147482a9caf370079dcb9c191a1b260"
REFERENCE_PYTHON_DLL_SHA = "14b06796f288bc6599e458fb23a944ab0c843e9868058f02a91d4606533505ed"
REFERENCE_PYTHON_VERSION = "3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]"
STDLIB_IMPORTS = frozenset(("__future__","ast","dataclasses","fractions","hashlib","importlib","json",
                          "os","pathlib","secrets","struct","subprocess","sys","threading","traceback"))
SYMBOLIC = None


@dataclass(frozen=True)
class Refusal:
    code: str
    detail: str


class Closed(Exception):
    def __init__(self, code, detail):
        self.refusal = Refusal(code, detail)
        super().__init__(detail)


def require(condition, code, detail):
    if not condition:
        raise Closed(code, detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def exact_metadata(value, expected):
    """Constant-schema equality rejects subclasses/extra nested fields pre-JSON."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (len(value)==len(expected) and all(type(k) is str for k in value)
                and set(value)==set(expected) and all(exact_metadata(value[k],v) for k,v in expected.items()))
    if type(expected) is list:
        return len(value)==len(expected) and all(exact_metadata(a,b) for a,b in zip(value,expected))
    return value==expected


def attempt_nonce():
    return secrets.token_hex(32)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def seal(value):
    return digest(b"CPU-REFERENCE-HARNESS-01\0" + json_bytes(value))


def symbolic():
    global SYMBOLIC
    if SYMBOLIC is None:
        source = SYMBOLIC_PATH.read_bytes()
        require(digest(source) == SYMBOLIC_SHA, "source_drift", "symbolic module differs from accepted pin")
        spec = importlib.util.spec_from_file_location("_cpu_reference_symbolic", SYMBOLIC_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(source, str(SYMBOLIC_PATH), "exec"), module.__dict__)
        SYMBOLIC = module
    return SYMBOLIC


def reference_hashes(root=REFERENCE_ROOT):
    """Read-only chunked provenance check; no extension import or loading."""
    found = []
    base = root.resolve()
    for relative, expected in PINNED_FILES:
        path = (base / relative).resolve()
        require(base in path.parents, "scope_escape", "reference path escaped pinned root")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(64 << 10), b""):
                h.update(block)
        require(h.hexdigest() == expected, "reference_drift", relative)
        found.append((relative, expected))
    return tuple(found)


def request_hash(request):
    return symbolic()._seal(request, b"OPENCV-CPU-V2-INPUT\0")


def parent_reservation(request):
    m = symbolic()
    try:
        rows, worker = m._validate(request, m.MAX_EXTRA_BYTES)
    except m._Refuse as error:
        raise Closed(error.refusal.code, error.refusal.detail) from None
    total = sum(len(v.data) for v in (request.source, request.map_x, request.map_y))
    output = (request.destination.shape[0] - 1) * request.destination.row_stride
    output += request.destination.shape[1] * m._source_contract(request.source)[2]
    # Hex strings/JSON/wire and decoding plus one detached output. No witness IPC.
    parent = 8 * total + 8 * output + 3 * (MAX_WIRE_BYTES + 8) + PARENT_SCRATCH_BYTES
    require(parent <= MAX_PARENT_BYTES, "parent_cap", "bounded parent serialization envelope unavailable")
    return rows, worker, parent, output


def expected_events(request):
    """Source-derived planned coverage; ONLY an attestor can prove execution."""
    lanes = {"BASELINE": 4, "SSE4_1": 4, "AVX2": 8}[request.active_dispatch]
    h, w = request.destination.shape
    block = 2 * lanes  # audited accurate invokers use uf=2*vlanes_32
    return {"entry": {("u8",1): "remapLinearInvoker_8UC1",
                      ("u8",3): "remapLinearInvoker_8UC3",
                      ("f32le",1): "remapLinearInvoker_32FC1"}[(request.source.kind,request.source.channels)],
            "dispatcher": request.active_dispatch, "float_lanes": lanes, "block_pixels": block,
            "vector_blocks": h * (w // block), "scalar_pixels": h * (w % block)}


def identity_record():
    return {"import_path": str(REFERENCE_ROOT / "build-clean/lib/python3/cv2.cp310-win_amd64.pyd"),
            "loaded_opencv": [[str(REFERENCE_ROOT / name), sha] for name, sha in PINNED_FILES
                              if name.endswith(".dll")],
            "files": [list(p) for p in PINNED_FILES], "symbolic_sha": SYMBOLIC_SHA}


def check_capability(request, capability, attempt=None, pid=None):
    require(type(capability) is dict and set(capability)=={"mode","identity","atomic","fp_policy","entry","dispatcher","attempt","pid"},
            "missing_attestation", "call-boundary attestor CAPABILITY required, not an old receipt")
    require(type(capability["mode"]) is str and capability["mode"]=="SYNTHETIC_ONLY", "unsupported_attestor", "real allowlist empty")
    require(exact_metadata(capability["identity"],identity_record()), "identity_mismatch", "provider identities differ")
    require(capability["atomic"] is True and type(capability["fp_policy"]) is str
            and capability["fp_policy"]==symbolic().FP_POLICY,
            "missing_attestation", "atomic pre/post FP and selected-entry capability required")
    require(canonical_digest(capability["attempt"]) and type(capability["pid"]) is int
            and 0<capability["pid"]<=0xFFFFFFFF
            and (attempt is None or capability["attempt"]==attempt)
            and (pid is None or capability["pid"]==pid),
            "attempt_mismatch","capability attempt/PID differs")
    expected=expected_events(request)
    require(type(capability["entry"]) is str and type(capability["dispatcher"]) is str
            and (capability["entry"],capability["dispatcher"])==(expected["entry"],expected["dispatcher"]),
            "branch_mismatch", "attestor cannot cover requested source entry")


def check_evidence(request, evidence, phase, attempt=None, pid=None):
    require(type(evidence) is dict and set(evidence) == {"mode","phase","identity","fp","events","request_sha","attempt","pid"},
            "missing_attestation", "complete fixed-shape call-boundary evidence required")
    require(type(evidence["mode"]) is str and evidence["mode"] == "SYNTHETIC_ONLY", "unsupported_attestor", "no admitted native attestor")
    require(canonical_digest(evidence["attempt"]) and canonical_digest(evidence["request_sha"])
            and type(evidence["pid"]) is int and 0<evidence["pid"]<=0xFFFFFFFF
            and (attempt is None or evidence["attempt"]==attempt)
            and (pid is None or evidence["pid"]==pid),
            "attempt_mismatch","phase attempt/PID differs")
    require(type(evidence["phase"]) is str and evidence["phase"] == phase, "event_order", "call-boundary event order differs")
    require(exact_metadata(evidence["identity"],identity_record()), "identity_mismatch", "import/DLL/source/config pins differ")
    fp = evidence["fp"]
    require(type(fp) is dict and set(fp)=={"mxcsr","x87","policy","scalar_arithmetic"}
            and type(fp["mxcsr"]) is int and 0 <= fp["mxcsr"] < 65536
            and fp["mxcsr"] & 0xFFC0 == 0x1F80
            and type(fp["x87"]) is int and 0 <= fp["x87"] < 65536 and fp["x87"] & 0xC00 == 0
            and type(fp["policy"]) is str and fp["policy"] == symbolic().FP_POLICY
            and type(fp["scalar_arithmetic"]) is str and fp["scalar_arithmetic"] == "SSE_BINARY32",
            "fp_mismatch", "RNE/gradual-underflow controls and binary32 arithmetic not attested")
    events=evidence["events"]
    expected=expected_events(request)
    require(type(events) is dict and set(events)==set(expected)
            and all(type(events[k]) is type(v) and events[k]==v for k,v in expected.items()),
            "branch_mismatch", "typed entry/lane/block/tail events differ")
    require(evidence["request_sha"] == request_hash(request), "metadata_mismatch", "exact call buffer request differs")
    require(len(json_bytes(evidence)) <= MAX_METADATA_BYTES, "metadata_cap", "attestation too large")


class FakeProvider:
    """Explicit fake provider. Bytes are arbitrary deterministic zeros, not remap."""
    mode = "SYNTHETIC_ONLY"

    def __init__(self):
        self.calls = 0
        self.attempt, self.pid = attempt_nonce(), os.getpid()

    def bind_attempt(self, attempt, pid):
        self.attempt, self.pid = attempt, pid

    def capability(self, request):
        expected=expected_events(request)
        return {"mode":self.mode,"identity":identity_record(),"atomic":True,
                "fp_policy":symbolic().FP_POLICY,"entry":expected["entry"],"dispatcher":expected["dispatcher"],
                "attempt":self.attempt,"pid":self.pid}

    def evidence(self, request, phase):
        return {"mode": self.mode, "phase": phase, "identity": identity_record(),
                "fp": {"mxcsr":0x1F80,"x87":0x37F,"policy":symbolic().FP_POLICY,
                       "scalar_arithmetic":"SSE_BINARY32"},
                "events":expected_events(request),"request_sha":request_hash(request),
                "attempt":self.attempt,"pid":self.pid}

    def call(self, request):
        self.calls += 1
        before=self.evidence(request,"before")
        output=bytes(parent_reservation(request)[3])
        return output,before,self.evidence(request,"after")


class Worker:
    """One request per owned worker lifetime; no ledger reset/release."""
    def __init__(self, provider):
        self.provider, self.consumed = provider, False
        self.lock = Lock()

    def run(self, request, *, attempt=None, expected_pid=None):
        require(self.lock.acquire(blocking=False), "ownership_busy", "concurrent worker use")
        try:
            require(not self.consumed, "one_shot", "worker already consumed")
            self.consumed = True  # before work: refusal/interruption cannot permit rerun
            attempt = attempt_nonce() if attempt is None else attempt
            expected_pid = os.getpid() if expected_pid is None else expected_pid
            require(canonical_digest(attempt) and type(expected_pid) is int and expected_pid==os.getpid(),
                    "attempt_mismatch","worker nonce and actual PID must match parent envelope")
            rows, reservation, parent, output_span = parent_reservation(request)
            require(getattr(self.provider,"mode",None)=="SYNTHETIC_ONLY", "unsupported_attestor", "native path closed")
            self.provider.bind_attempt(attempt,expected_pid)
            check_capability(request,self.provider.capability(request),attempt,expected_pid)
            m=symbolic()
            witness=m.build_symbolic_support(request,max_extra_bytes=reservation)
            require(type(witness) is m.SymbolicSupport, "input_refusal", str(witness))
            require(m.verify_symbolic_support(request,witness,max_extra_bytes=reservation),
                    "support_refusal", "worker symbolic replay failed")
            output,before,after=self.provider.call(request)
            check_evidence(request,before,"before",attempt,expected_pid)
            check_evidence(request,after,"after",attempt,expected_pid)
            require(type(output) is bytes and len(output)==output_span, "output_mismatch", "detached output span differs")
            record={"status":"SYNTHETIC_ONLY","pid":os.getpid(),"attempt":attempt,"request_sha":request_hash(request),
                    "before":before,"after":after,"support_seal":witness.seal_sha256,
                    "support_bytes":len(witness.payload),"worker_reservation":reservation,"parent_reservation":parent,
                    "kind":request.destination.kind,"channels":request.destination.channels,
                    "shape":list(request.destination.shape),"stride":request.destination.row_stride,
                    "output_sha":digest(output),"output_hex":output.hex()}
            encoded_record=json_bytes(record)
            require(len(encoded_record)<=MAX_WIRE_BYTES,"wire_cap","receipt too large")
            h=hashlib.sha256(b"CPU-REFERENCE-HARNESS-01\0")
            h.update(encoded_record)
            return {"record":record,"seal":h.hexdigest()}
        finally:
            self.lock.release()


class OneShotGuard:
    """Durable exclusive create before worker launch. Caller must authorize path.

    Used only under pytest tmp_path here. This assignment creates no run marker.
    IN_PROGRESS survives a crash; no rollback/retry/truncation exists.
    """
    def __init__(self,path):
        self.path=Path(path)

    def claim(self,plan_sha):
        require(type(plan_sha) is str and len(plan_sha)==64
                and all(c in "0123456789abcdef" for c in plan_sha),"invalid_plan","canonical plan pin required")
        try:
            with self.path.open("xb") as f:
                f.write(json_bytes({"state":"IN_PROGRESS","plan_sha":plan_sha})+b"\n")
                f.flush(); os.fsync(f.fileno())
        except FileExistsError:
            raise Closed("one_shot","run marker already exists") from None

    def complete(self,plan_sha,result_sha):
        require(canonical_digest(plan_sha) and canonical_digest(result_sha),"invalid_plan","canonical completion digests")
        with self.path.open("rb") as f:
            existing=f.read(MAX_METADATA_BYTES+1)
        require(existing==json_bytes({"state":"IN_PROGRESS","plan_sha":plan_sha})+b"\n",
                "marker_mismatch","run ownership/status differs")
        with self.path.open("ab") as f:
            f.write(json_bytes({"state":"COMPLETE","plan_sha":plan_sha,"result_sha":result_sha})+b"\n")
            f.flush(); os.fsync(f.fileno())


def encode_request(request):
    parent_reservation(request)  # before allocating hex/JSON copies
    value=asdict(request)
    for field in ("source","map_x","map_y"):
        value[field]["data"]=getattr(request,field).data.hex()
    encoded=json_bytes(value)
    require(len(encoded)<=MAX_WIRE_BYTES,"wire_cap","request too large")
    return encoded


def bounded_json(data):
    """Bound structure before JSON allocation; hex data may be long, metadata not.

    Fixed protocol permits <=2048 structural punctuation, depth<=12 and <=64
    characters per unquoted primitive. This blocks large object/number attacks.
    """
    require(type(data) in (bytes,memoryview) and len(data)<=MAX_WIRE_BYTES,"wire_cap","JSON byte cap")
    quoted=escaped=False
    depth=count=primitive=0
    for value in data:
        require(value<128,"wire_mismatch","ASCII protocol required")
        if quoted:
            if escaped:
                escaped=False
            elif value==92:
                escaped=True
            elif value==34:
                quoted=False
            continue
        if value==34:
            quoted=True; primitive=0
        elif value in (123,91):
            depth+=1; count+=1; primitive=0
        elif value in (125,93):
            depth-=1; count+=1; primitive=0
        elif value in (44,58):
            count+=1; primitive=0
        elif value in (9,10,13,32):
            primitive=0
        else:
            primitive+=1
        require(0<=depth<=12 and count<=2048 and primitive<=64,"metadata_cap","JSON structure/primitive cap")
    require(not quoted and depth==0,"wire_mismatch","incomplete JSON structure")
    return json.loads(bytes(data).decode("ascii"))


def decode_request(data):
    require(type(data) is bytes and len(data)<=MAX_WIRE_BYTES,"wire_cap","bounded immutable packet required")
    return decode_request_value(bounded_json(data))


def decode_request_value(value):
    m=symbolic()
    require(type(value) is dict and set(value)==set(m.SupportRequest.__dataclass_fields__),
            "wire_mismatch","exact request schema before conversion")
    kernel=value.pop("kernel")
    require(type(kernel) is dict and set(kernel)==set(m.KernelIdentity.__dataclass_fields__),
            "wire_mismatch","exact kernel schema before conversion")
    pins=kernel["source_files"]
    require(type(pins) is list and len(pins)==4
            and all(type(pair) is list and len(pair)==2 and all(type(x) is str and len(x)<=128 for x in pair) for pair in pins),
            "wire_mismatch","bounded source pin tuples before conversion")
    kernel["source_files"]=tuple(tuple(p) for p in kernel["source_files"])
    value["kernel"]=m.KernelIdentity(**kernel)
    for field in ("source","map_x","map_y"):
        view=value[field]
        require(type(view) is dict and set(view)==set(m.BufferView.__dataclass_fields__)
                and type(view["shape"]) is list and len(view["shape"])==2
                and all(type(v) is int for v in view["shape"])
                and type(view["data"]) is str and len(view["data"])<=2*m.MAX_BUFFER_BYTES,
                "wire_mismatch","bounded typed view before tuple/hex conversion")
        view["shape"]=tuple(view["shape"])
        view["data"]=bytes.fromhex(view["data"])
        value[field]=m.BufferView(**view)
    destination=value["destination"]
    require(type(destination) is dict and set(destination)==set(m.OutputLayout.__dataclass_fields__)
            and type(destination["shape"]) is list and len(destination["shape"])==2
            and all(type(v) is int for v in destination["shape"]),
            "wire_mismatch","bounded output shape before tuple conversion")
    value["destination"]["shape"]=tuple(value["destination"]["shape"])
    value["destination"]=m.OutputLayout(**value["destination"])
    request=m.SupportRequest(**value)
    parent_reservation(request)
    return request


def validate_receipt(request,receipt,*,expected_attempt,expected_pid,require_child=True):
    require(type(receipt) is dict and set(receipt)=={"record","seal"},"receipt_mismatch","sealed receipt required")
    require(canonical_digest(expected_attempt) and type(expected_pid) is int and 0<expected_pid<=0xFFFFFFFF,
            "attempt_mismatch","parent-issued attempt and spawned PID required")
    require(canonical_digest(receipt["seal"]),"receipt_mismatch","canonical outer seal required before serialization")
    record=receipt["record"]
    rows,worker,parent,output_span=parent_reservation(request)
    require(type(record) is dict and set(record)=={"status","pid","request_sha","before","after","support_seal",
            "support_bytes","worker_reservation","parent_reservation","kind","channels","shape","stride",
            "output_sha","output_hex","attempt"},"receipt_mismatch","fixed receipt schema required")
    require(all(type(record[k]) is int for k in ("worker_reservation","parent_reservation","channels","stride"))
            and type(record["shape"]) is list and len(record["shape"])==2
            and all(type(v) is int for v in record["shape"])
            and all(type(record[k]) is str for k in ("status","request_sha","kind","output_sha")),
            "receipt_mismatch","typed output metadata required")
    require(all(canonical_digest(record[k]) for k in ("request_sha","output_sha","support_seal","attempt")),
            "receipt_mismatch","canonical bounded digests required before serialization")
    require(record["attempt"]==expected_attempt,"attempt_mismatch","receipt belongs to another attempt")
    require(record["status"]=="SYNTHETIC_ONLY" and type(record["pid"]) is int and record["pid"]>0
            and record["pid"]==expected_pid and (not require_child or record["pid"]!=os.getpid()),"isolation_mismatch","synthetic isolated worker required")
    check_evidence(request,record["before"],"before",expected_attempt,expected_pid)
    check_evidence(request,record["after"],"after",expected_attempt,expected_pid)
    require((record["request_sha"],record["worker_reservation"],record["parent_reservation"],record["kind"],
             record["channels"],record["shape"],record["stride"]) ==
            (request_hash(request),worker,parent,request.destination.kind,request.destination.channels,
             list(request.destination.shape),request.destination.row_stride),"receipt_mismatch","request/output metadata differs")
    require(type(record["support_seal"]) is str and len(record["support_seal"])==64
            and type(record["support_bytes"]) is int and 192<=record["support_bytes"]<=symbolic().witness_bound(rows),
            "receipt_mismatch","bounded worker support summary required")
    require(type(record["output_hex"]) is str and len(record["output_hex"])==2*output_span,
            "output_mismatch","output serialized length differs")
    require(receipt["seal"]==seal(record),"receipt_mismatch","receipt seal differs")
    output=bytes.fromhex(record["output_hex"])
    require(digest(output)==record["output_sha"],"output_mismatch","output bytes differ from sealed worker output")
    return {"status":"SYNTHETIC_ONLY","request_sha":record["request_sha"],"output_sha":record["output_sha"],
            "support_seal":record["support_seal"],"receipt_sha":receipt["seal"],
            "attempt":expected_attempt,"pid":expected_pid}


def framed(data):
    require(type(data) is bytes and len(data)<=MAX_WIRE_BYTES,"wire_cap","frame too large")
    return struct.pack("<I",len(data))+data


def unframe(data):
    require(type(data) is bytes and 4<=len(data)<=MAX_WIRE_BYTES+4,"wire_cap","bounded frame required")
    count=struct.unpack_from("<I",data)[0]
    require(count<=MAX_WIRE_BYTES and count+4==len(data),"wire_mismatch","truncated/extra/oversize frame")
    return memoryview(data)[4:]



def worker_python():
    """Avoid Windows venv redirector: own the executing interpreter's actual PID."""
    require(sys.version==REFERENCE_PYTHON_VERSION and sys.implementation.name=="cpython"
            and sys.platform=="win32" and struct.calcsize("P")==8,
            "python_mismatch","same pinned CPython build/version required")
    actual=Path(getattr(sys,"_base_executable",sys.executable)).resolve()
    require(actual==REFERENCE_PYTHON.resolve(),"python_mismatch","base interpreter path differs")
    for path,expected in ((actual,REFERENCE_PYTHON_SHA),(actual.parent/"python310.dll",REFERENCE_PYTHON_DLL_SHA)):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(64<<10),b""):h.update(block)
        require(h.hexdigest()==expected,"python_mismatch","base interpreter binary identity differs")
    for path in (Path(__file__),SYMBOLIC_PATH):
        tree=ast.parse(path.read_text(encoding="utf-8"))
        imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        require(all(name.split(".")[0] in STDLIB_IMPORTS for name in imports),
                "nonstdlib_dependency","worker imports must remain stdlib-only")
    return str(actual)


class ChildOwner:
    """Cleanup context exists before spawn; owns every post-spawn acquisition.

    Cleanup independently attempts every operation, including after BaseException.
    Incomplete teardown latches refusal; no receipt can return successfully.
    """
    def __init__(self, attempt):
        self.attempt=attempt
        self.process=None
        self.threads=()
        self.phase="new"

    def __enter__(self):
        return self

    def spawn(self):
        self.process=subprocess.Popen([worker_python(),"-I","-S","-B",str(Path(__file__).resolve()),"--synthetic-worker"],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=str(ROOT))
        self.phase="spawned"
        return self.process

    def thread(self,target):
        thread=Thread(target=target,daemon=True)
        self.threads=self.threads+(thread,)
        return thread

    def accept(self,request,receipt):
        require(self.phase=="spawned","one_shot","one receipt per owned attempt")
        self.phase="accepting"  # no second promotion even when verification fails
        result=validate_receipt(request,receipt,expected_attempt=self.attempt,expected_pid=self.process.pid)
        self.phase="accepted"
        return result

    def __exit__(self,exception_type,error,tb):
        failures=[]
        def attempt(tag,call):
            try:
                return call()
            except BaseException as cleanup_error:
                failures.append((tag,type(cleanup_error).__name__))
                return None
        process=self.process
        if process is not None:
            status=attempt("poll",process.poll)
            if status is None:
                attempt("kill",process.kill)
            attempt("reap",lambda:process.wait(timeout=5))
            for name in ("stdin","stdout","stderr"):
                stream=attempt("get_"+name,lambda name=name:getattr(process,name,None))
                if stream is not None:
                    attempt("close_"+name,stream.close)
            for index,thread in enumerate(self.threads):
                identity=attempt("thread_identity",lambda thread=thread:thread.ident)
                if identity is not None:
                    attempt("join_"+str(index),lambda thread=thread:thread.join(timeout=1))
                alive=attempt("thread_alive",thread.is_alive)
                if alive is not False:
                    failures.append(("thread_alive","incomplete"))
            final_status=attempt("final_poll",process.poll)
            if final_status is None:
                failures.append(("child_alive","incomplete"))
        self.phase="incomplete" if failures else "closed"
        require(not failures,"cleanup_incomplete","child teardown incomplete: "+repr(failures))
        return False


def encode_attempt(request_data,attempt,pid):
    require(canonical_digest(attempt) and type(pid) is int and 0<pid<=0xFFFFFFFF,
            "attempt_mismatch","bounded parent attempt/PID required")
    prefix=b'{"attempt":"'+attempt.encode("ascii")+b'","pid":'+str(pid).encode("ascii")+b',"request":'
    return framed(prefix+request_data+b'}')


def subprocess_fake(request,*,timeout=TIMEOUT_SECONDS):
    """Actual isolated stdlib-only child; no native provider or import exists."""
    request_data=encode_request(request)
    nonce=attempt_nonce()
    outputs=[]; errors=[]
    with ChildOwner(nonce) as owner:
        process=owner.spawn()
        encoded=encode_attempt(request_data,nonce,process.pid)
        def write():
            try:
                process.stdin.write(encoded); process.stdin.close()
            except Exception as error:
                errors.append(type(error).__name__)
        def read():
            outputs.append(process.stdout.read(MAX_WIRE_BYTES+5))
        writer=owner.thread(write); reader=owner.thread(read)
        writer.start(); reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Closed("worker_timeout","worker did not terminate") from None
        writer.join(timeout=1); reader.join(timeout=1)
        require(not writer.is_alive() and not reader.is_alive() and not errors,"worker_failure","worker pipe failed")
        require(process.returncode==0,"worker_failure","worker returned nonzero")
        require(len(outputs)==1,"worker_failure","worker output missing")
        return owner.accept(request,bounded_json(unframe(outputs[0])))


def synthetic_worker():
    header=sys.stdin.buffer.read(4)
    require(len(header)==4,"wire_mismatch","missing header")
    size=struct.unpack("<I",header)[0]
    require(size<=MAX_WIRE_BYTES,"wire_cap","input frame cap")
    body=sys.stdin.buffer.read(size+1)
    require(len(body)==size,"wire_mismatch","truncated/extra input")
    envelope=bounded_json(body)
    require(type(envelope) is dict and set(envelope)=={"attempt","pid","request"}
            and canonical_digest(envelope["attempt"]) and type(envelope["pid"]) is int
            and envelope["pid"]==os.getpid(),"attempt_mismatch","parent envelope PID/nonce differs")
    nonce,pid=envelope["attempt"],envelope["pid"]
    request=decode_request_value(envelope["request"])
    del envelope
    del body  # parser/packet temporaries gone before witness allocation
    receipt=Worker(FakeProvider()).run(request,attempt=nonce,expected_pid=pid)
    sys.stdout.buffer.write(framed(json_bytes(receipt)))
    sys.stdout.buffer.flush()


@dataclass(frozen=True)
class Recipe:
    name: str
    kind: str
    channels: int
    dispatch: str
    width: int
    padded: bool
    pattern: str
    border: int = 0


def matrix_recipes():
    patterns=("integer","half","nextbelow1","negative_near0","mixed","outside","degenerate","tiny")
    for kind,channels in (("u8",1),("u8",3),("f32le",1)):
        for dispatch,block in (("BASELINE",8),("SSE4_1",8),("AVX2",16)):
            for width in (block-1,block,block+1):
                for padded in (False,True):
                    for pattern in patterns + (("impulse0","impulse1","impulse2") if channels==3 else ()):
                        name=f"{kind}-{channels}-{dispatch}-{width}-{int(padded)}-{pattern}"
                        yield Recipe(name,kind,channels,dispatch,width,padded,pattern)
    for dispatch in ("BASELINE","SSE4_1","AVX2"):
        for border in (-17,300):
            yield Recipe(f"u8C1-{dispatch}-border{border}","u8",1,dispatch,9,False,"outside",border)


def matrix_pin():
    records=[asdict(r) for r in matrix_recipes()]
    require(len(records)<=MAX_CASES,"case_cap","finite matrix exceeds cap")
    return seal({"recipes":records,"harness_sha":digest(Path(__file__).read_bytes()),
                 "symbolic":SYMBOLIC_SHA,"reference":PINNED_FILES,
                 "parent_bytes":MAX_PARENT_BYTES,"wire_bytes":MAX_WIRE_BYTES,
                 "status":"SYNTHETIC_PLAN_NATIVE_NOT_STARTED"})


def make_request(recipe):
    m=symbolic()
    size=4 if recipe.kind=="f32le" else 1
    shape=(1,1) if recipe.pattern=="degenerate" else (2,2)
    h,w=shape
    stride=w*size*recipe.channels+(4 if recipe.padded else 0)
    data=bytearray((h-1)*stride+w*size*recipe.channels)
    for y in range(h):
        for x in range(w):
            for c in range(recipe.channels):
                offset=y*stride+size*(x*recipe.channels+c)
                if size==4:
                    struct.pack_into("<I",data,offset,(0,0x3E800000,0x3F000000,0x3F800000)[(y*w+x)%4])
                else:
                    data[offset]=(11,37,211)[c]
    if recipe.pattern.startswith("impulse"):
        data=bytearray(len(data)); data[int(recipe.pattern[-1])]=255
    bits={"integer":(0,0),"half":(0x3F000000,0x3F000000),
          "nextbelow1":(0x3F7FFFFF,0x3F7FFFFF),"negative_near0":(0x80800000,0),
          "mixed":(0xBF000000,0x3F000000),"outside":(0xBF800000,0xBF800000),
          "degenerate":(0x3F000000,0x3F000000),"tiny":(0x00800000,0x00800000)}
    xbits,ybits=bits.get(recipe.pattern,(0x3F000000,0x3F000000))
    outshape=(2,recipe.width)
    map_stride=4*recipe.width+(4 if recipe.padded else 0)
    def mapping(value,address):
        row=struct.pack("<I",value)*recipe.width
        raw=row+bytes(map_stride-len(row))+row
        return m.BufferView(raw,outshape,map_stride,address,"f32le")
    source=m.BufferView(bytes(data),shape,stride,0x10000000,recipe.kind,recipe.channels)
    destination=m.OutputLayout(outshape,recipe.width*size*recipe.channels+(4 if recipe.padded else 0),
                               0x40000000,recipe.kind,recipe.channels)
    return m.SupportRequest(m.REFERENCE_KERNEL,source,mapping(xbits,0x20000000),mapping(ybits,0x30000000),
                            destination,fp_policy=m.FP_POLICY,active_dispatch=recipe.dispatch,
                            resolved_hint="ALGO_HINT_ACCURATE",border_value=recipe.border)


def run_fake_matrix(guard,transport=subprocess_fake):
    """Single consumed fake plan; only digest summaries retained, no trace list.

    No native mode or evidence path is authorized/implemented. Default subprocess
    transport validates the full receipt and ends each owned child before return.
    """
    pin=matrix_pin()
    guard.claim(pin)  # durable IN_PROGRESS before first transport; never retried
    chain=hashlib.sha256(b"CPU-REFERENCE-FAKE-MATRIX-01\0")
    count=0
    seen_attempts=set()
    for recipe in matrix_recipes():
        require(count<MAX_CASES,"case_cap","matrix count exceeded")
        request=make_request(recipe)
        result=transport(request)
        require(type(result) is dict and set(result)=={"status","request_sha","output_sha","support_seal",
                "receipt_sha","attempt","pid"} and type(result["status"]) is str and result["status"]=="SYNTHETIC_ONLY"
                and all(canonical_digest(result[k]) for k in ("request_sha","output_sha","support_seal","receipt_sha","attempt"))
                and type(result["pid"]) is int and 0<result["pid"]<=0xFFFFFFFF
                and result["request_sha"]==request_hash(request),
                "receipt_mismatch","exact bounded synthetic summary required before serialization")
        require(result["attempt"] not in seen_attempts,"attempt_mismatch","duplicate attempt summary")
        seen_attempts.add(result["attempt"])
        chain.update(json_bytes({"recipe":asdict(recipe),"result":result}))
        count+=1
    summary={"status":"SYNTHETIC_ONLY","cases":count,"plan_sha":pin,"chain_sha":chain.hexdigest()}
    guard.complete(pin,seal(summary))
    return summary


def native_preflight():
    # Labels, old smoke output and available ISA flags cannot fill this gap.
    raise Closed("unsupported_attestor","no approved actual-entry/FP provider; native NOT_STARTED")


if __name__=="__main__":
    try:
        if sys.argv[1:]==["--synthetic-worker"]:
            synthetic_worker()
        else:
            native_preflight()
    except Exception as error:
        refusal=error.refusal if isinstance(error,Closed) else Refusal("worker_failure",type(error).__name__)
        sys.stdout.write(json_bytes(asdict(refusal)).decode("ascii")+"\n")
        raise SystemExit(2)
