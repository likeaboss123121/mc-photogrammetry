"""Authored synthetic metadata/PNG/masks only; all production services are fake."""
import copy
import hashlib
import json
import struct
from types import SimpleNamespace
import zlib

import pytest
from scripts.evaluation import spatial_context_native_io as native


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


def authored_png():
    # Decoder is fake; this is authored PNG wire structure, never a capture.
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",1600,900,8,2,0,0,0))
            + chunk(b"IDAT", zlib.compress(b"authored-synthetic-buffer")) + chunk(b"IEND",b""))


def seal(record, field):
    record[field] = native.sha(native.canonical(record, 1048576))
    return record


class FakeArray:
    def __init__(self, data, *, mask=False):
        self.data = data
        self.shape = (900,1600) if mask else (900,1600,3)
        self.dtype = SimpleNamespace(str="|b1" if mask else "|u1")
        self.flags = SimpleNamespace(c_contiguous=True, writeable=True)
    def tobytes(self, order="C"):
        assert order == "C"
        return self.data
    def setflags(self, *, write):
        self.flags.writeable = write


def authored_manifest(base=r"C:\authored-synthetic"):
    png = authored_png()
    rgb = b"authored-rgb"
    usable = b"authored-usable-holes"
    sources = []
    for index, endpoint in enumerate(native.ENDPOINTS):
        sources.append(dict(frame_id=endpoint[0],endpoint=list(endpoint),region_id=native.sha(str(index).encode()),
            encoded_sha256=native.sha(png),encoded_length=len(png),decoded_shape=[900,1600,3],decoded_dtype="|u1",
            decoded_rgb_sha256=native.sha(rgb),ingress_sha256="b"*64,packed_mask_sha256="c"*64,
            full_native_bool_sha256=native.sha(b"authored-mask-"+str(index).encode()),
            usable_mask_sha256=native.sha(usable),usable_helper_sha256="d"*64))
    polygon = [[0,0],[1,0],[1,1],[0,1]]
    ingress = [dict(frame_id=name,source_label="authored-synthetic",byte_sha256=native.sha(png),byte_length=len(png),
        decoded_shape=[900,1600,3],decoded_dtype="|u1",decoded_pixel_sha256="b"*64,
        transform=dict(source_size=[1600,900],crop_origin=[0,0],crop_size=None,output_size=None),
        schema="mc-photogrammetry-frame-ingress-provenance-v1",decoder="input.image.load_image:exif_transpose+convert('RGB')")
        for name in ("image-0000","image-0001")]
    ingress_hashes={row["frame_id"]:native.sha(native.canonical(row)) for row in ingress}
    entries = []
    inventory = []
    for index,(label,src) in enumerate(zip(native.LABELS,sources)):
        entry=dict(endpoint=src["endpoint"],reading_index=index,reading_binding_sha256="e"*64,
            canonical_polygon=polygon,ingress_sha256=ingress_hashes[src["frame_id"]],row_refs=[],
            mask=dict(source_region_id=src["region_id"],axis=0,sources=["authored"],image_shape=[900,1600],
                bbox_xyxy=[0,0,1,1],crop_shape=[1,1],encoding="packbits-base64",bit_order="little",
                packed_mask_base64="AQ==",mask_sha256="a"*64,pixel_count=1),
            full_native_bool_c_sha256=src["full_native_bool_sha256"])
        entries.append(entry)
        inventory.append(dict(label=label,endpoint=src["endpoint"],region_id=src["region_id"],
            reading_binding_sha256="e"*64,ingress_record_sha256=ingress_hashes[src["frame_id"]],header_packed_sha256="a"*64,
            full_bool_sha256=src["full_native_bool_sha256"],polygon_vertices=4))
    context=dict(ingress=ingress,
                 readings=[dict(endpoint=src["endpoint"]) for src in sources],rows=[{},{}])
    code_pins={path:"a"*64 for path in native.HISTORICAL_PATHS}
    bundle=seal(dict(schema="mc-photogrammetry-source-region-mask-bundle-v1",producer="native-plane-region-mask-export-1",
        status="complete",source_context=context,code_pins=code_pins,limits=dict(native.MASK_LIMITS),entries=entries),"record_sha256")
    diagnostic=seal(dict(schema="mc-photogrammetry-image-geometry-failure-diagnostic-v2",
        ingress=context["ingress"],joint_endpoint_selection=dict(first_witness_provenance=dict(
            readings=[dict(polygon=polygon) for _ in range(3)],rows=[{},{}])),source_region_masks=bundle),"record_sha256")
    diagnostic_bytes=native.canonical(diagnostic,1048576)+b"\n"
    runtime=dict(implementation="authored-fake-only")
    binding=dict(sources=sources[1:],source_pins=[list(row) for row in native.SOURCE_PINS],
        diagnostic_outer_sha256=diagnostic["record_sha256"],diagnostic_nested_sha256=bundle["record_sha256"],
        contract_sha256="c14d8ce72fc7131ea6b17b7c26df735a9475d2bec5eacd9b6e04bcf0f9391bcb",
        source_sha256=native.SOURCE_PINS[0][1],runtime_sha256=native.sha(native.canonical(runtime)))
    associations=[dict(label=label,frame_id=src["frame_id"],endpoint=src["endpoint"],region_id=src["region_id"])
                  for label,src in zip(native.LABELS,sources)]
    reservation=dict(plans=3,grid_positions=16800,boolean_operands=77414400,extractions=2,
        extraction_windows=2607,extraction_pixels=6006528,gradient_sites=4598748,derivation_elements=834240,
        comparisons_per_arm=95090,comparisons_total=190180,distance_matrix_bytes=760720,component_contributions=30428800,
        tuples_per_arm=4096,determinants_per_arm=32768,tuples_total=8192,determinants_total=65536,
        model_bytes=10092544,publication_bytes=20381696,wall_seconds=180,committed_bytes=2147483648,log_bytes=65536)
    dependency=dict(descriptor=native.SOURCE_PINS[0][1],evidence=native.SOURCE_PINS[1][1])
    metadata=dict(expected_supply=[0,37,2570],feasibility_identities=list(native.LABELS),reservation=reservation,
        dependency_pins=dependency,source_pins=binding["source_pins"],contract_sha256=binding["contract_sha256"],
        source_sha256=binding["source_sha256"],source_inventory=sources,label_associations=associations,
        inventory_sha256=native.sha(native.canonical(dict(sources=sources,label_associations=associations))),
        diagnostic_payload_sha256=native.sha(diagnostic_bytes),diagnostic_outer_sha256=binding["diagnostic_outer_sha256"],
        diagnostic_nested_sha256=binding["diagnostic_nested_sha256"],runtime_sha256=binding["runtime_sha256"])
    request=dict(bindings=binding,runtime=runtime,metadata=metadata,row176=sources[0],diagnostic_sha256=native.sha(diagnostic_bytes),
        diagnostic_length=len(diagnostic_bytes),label_associations=associations,
        feasibility_sha256="29e4fb299158377fd791b0f24185dce8c6b6f55df9650c691942a863d58f46fb",dependency_pins=dependency,limits=dict(native.LIMITS))
    output=base+r"\attempt"
    files=dict(diagnostic=dict(path=base+r"\authored-diagnostic.json",bytes=len(diagnostic_bytes),sha256=native.sha(diagnostic_bytes)),
        frames=[dict(frame_id=name,path=base+f"\\authored-{i}.png",bytes=len(png),sha256=native.sha(png),png_width=1600,png_height=900,exif_orientation=1)
                for i,name in enumerate(("image-0000","image-0001"))])
    receipt=dict(path=base+r"\authored-inventory.json",bytes=10,sha256="a"*64)
    record=dict(schema="spatial-context-native-manifest-v1",attempt_id="authored_attempt",request_template=request,files=files,
        provenance=dict(inventory_receipt=receipt,diagnostic_outer_sha256=diagnostic["record_sha256"],diagnostic_nested_sha256=bundle["record_sha256"],
            mask_context_sha256=native.sha(native.canonical(context)),mask_code_pins=code_pins,mask_limits=dict(native.MASK_LIMITS),mask_inventory=inventory),
        runtime=dict(attestation=dict(path=base+r"\authored-attestation.json",bytes=10,sha256="a"*64),
            python=dict(path=base+r"\python.exe",sha256="a"*64,version="authored-fake-python"),
            modules=[dict(module=name,path=base+"\\modules\\"+name.replace(".","\\")+".py",sha256=pin,kind="repository")
                     for name,pin in sorted({**native.REPOSITORY_PINS, **native.SEED_PINS, "scripts.evaluation.spatial_context_native_io":"e"*64, "scripts.evaluation.run_spatial_context_native":"f"*64}.items())]),
        launch=dict(cwd=base,argv=[base+r"\python.exe","authored-worker"],environment=dict(SystemRoot=r"C:\Windows",TEMP=output+r"\temp",
            TMP=output+r"\temp",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")),
        artifacts=dict(output_directory=output,authorization_path=base+r"\authored-auth.json",consumed_path=base+r"\authored-consumed.json",
            evidence=output+r"\evidence.json",worker_candidate=output+r"\worker-staged-candidate.json",completion=output+r"\completion.json",
            failure=output+r"\failure.json",log=output+r"\worker.log"),limits=dict(native.LIMITS))
    return record,diagnostic_bytes,png


def manifest(record):
    record=copy.deepcopy(record)
    record.pop("manifest_sha256",None)
    seal(record,"manifest_sha256")
    wire=native.canonical(record)+b"\n"
    return native.Manifest(wire,native.sha(wire))


class FakeDiagnostic:
    def __init__(self):
        self.calls=[]
    def _bounded_bytes(self,value,limit=None):
        self.calls.append("bound")
        return native.canonical(value,limit or 1048576)
    def _digest(self,value,limit=None):
        return native.sha(native.canonical(value,limit or 1048576))
    def _preflight_mask_entries(self,entries):
        self.calls.append("preflight")
        assert len(entries)==3
    def _mask_source_context(self,record):
        self.calls.append("context")
        return record["source_region_masks"]["source_context"]
    def _mask_limits(self):
        return dict(native.MASK_LIMITS)
    def _validate_mask_entry_joins(self,entries,context):
        self.calls.append("joins")
    def _decode_source_mask(self,entry):
        self.calls.append(("unpack",entry["reading_index"]))
        return FakeArray(b"authored-mask-"+str(entry["reading_index"]).encode(),mask=True)
    def _validate_mask_diagnostic(self,*args):
        raise AssertionError("15-unpack high-level helper forbidden")
    load_native_source_region_mask=_validate_mask_diagnostic


def services():
    events=[]
    def decode(data,*,suffix):
        events.append(("decode",data,suffix))
        return FakeArray(b"authored-rgb")
    def usable(frame_id,shape,mapping):
        assert mapping is None, "empty mapping changes GUI mask semantics"
        events.append(("usable",frame_id))
        return FakeArray(b"authored-usable-holes",mask=True)
    inspection=lambda *args: SimpleNamespace(payload_sha256=args[0],outer_sha256=args[1],nested_sha256=args[2],sources=args[3])
    return SimpleNamespace(events=events,diagnostic=FakeDiagnostic(),driver=SimpleNamespace(DiagnosticInspection=inspection),
        load_image_bytes=decode,image_sha256=lambda pixels:"b"*64,usable_mask_for=usable)


def test_manifest_non_circular_template_and_exact_closed_seal():
    record,_,_=authored_manifest()
    accepted=manifest(record)
    assert "authorization_sha256" not in accepted.record["request_template"]["bindings"]
    assert "binding_sha256" not in accepted.record["request_template"]["bindings"]
    accepted.check()


@pytest.mark.parametrize("kind",["unknown","duplicate_path","source_pin","seed_pin","endpoint","bool","polygon","modules","unc","deep"])
def test_manifest_refusals_before_open_or_decoder(monkeypatch,kind):
    record,_,_=authored_manifest()
    if kind=="unknown":record["extra"]="bad"
    elif kind=="duplicate_path":record["files"]["frames"][1]["path"]=record["files"]["frames"][0]["path"]
    elif kind=="source_pin":record["request_template"]["bindings"]["source_pins"][0][1]="f"*64
    elif kind=="seed_pin":record["runtime"]["modules"][0]["sha256"]="f"*64
    elif kind=="endpoint":record["provenance"]["mask_inventory"][0]["endpoint"]=["image-0000",176]
    elif kind=="bool":record["files"]["frames"][0]["exif_orientation"]=True
    elif kind=="polygon":record["provenance"]["mask_inventory"][2]["polygon_vertices"]=65
    elif kind=="modules":record["runtime"]["modules"]*=257
    elif kind=="unc":record["files"]["diagnostic"]["path"]=r"\\server\image.png"
    else:
        for _ in range(25):record={"nested":record}
    def forbidden(*args,**kwargs):raise AssertionError("open/import before metadata refusal")
    monkeypatch.setattr("builtins.open",forbidden)
    monkeypatch.setattr(native.importlib,"import_module",forbidden)
    with pytest.raises(native.Refusal):manifest(record)


@pytest.mark.parametrize("data",[b"{\"a\":1,\"a\":2}",b"{\"x\":NaN}",b"["*25+b"0"+b"]"*25,b"1"*65,b"x"*262145],
                         ids=["duplicate-keys","nonfinite","depth-cap-plus-one","integer-width","bytes-cap-plus-one"])
def test_lexical_caps_duplicates_nonfinite_before_delegate(data):
    with pytest.raises(native.Refusal):native.parse(data,cap=262144)


def test_zero_unpack_validation_then_three_exact_region_requests():
    record,data,_=authored_manifest()
    accepted=manifest(record);fake=services();adapter=native.DiagnosticAdapter(accepted,fake)
    inspection=adapter.inspect(data)
    assert [c for c in fake.diagnostic.calls if c in ("preflight","context","joins")]==["preflight","context","joins"]
    assert not any(type(c) is tuple for c in fake.diagnostic.calls)
    for src in inspection.sources:adapter.decode_mask(data,src["region_id"])
    assert [c for c in fake.diagnostic.calls if type(c) is tuple]==[("unpack",0),("unpack",1),("unpack",2)]
    with pytest.raises(native.Refusal,match="duplicate_mask"):adapter.decode_mask(data,inspection.sources[0]["region_id"])


@pytest.mark.parametrize("target",["third_entry","witness_polygon","bundle_polygon","external_reseal"])
def test_diagnostic_rejection_precedes_context_or_unpack(target):
    record,data,_=authored_manifest()
    parsed=json.loads(data)
    if target=="third_entry":parsed["source_region_masks"]["entries"][2]["mask"]["crop_shape"]=[900,1601]
    elif target=="witness_polygon":parsed["joint_endpoint_selection"]["first_witness_provenance"]["readings"][2]["polygon"]=[[0,0]]*65
    elif target=="bundle_polygon":parsed["source_region_masks"]["entries"][2]["canonical_polygon"]=[[0,0]]*65
    else:parsed["source_region_masks"]["entries"][2]["mask"]["source_region_id"]="f"*64
    altered=native.canonical(parsed,1048576)+b"\n"
    if target!="external_reseal":
        record["files"]["diagnostic"].update(bytes=len(altered),sha256=native.sha(altered))
        record["request_template"].update(diagnostic_length=len(altered),diagnostic_sha256=native.sha(altered))
        record["request_template"]["metadata"]["diagnostic_payload_sha256"]=native.sha(altered)
    fake=services();adapter=native.DiagnosticAdapter(manifest(record),fake)
    with pytest.raises(native.Refusal):adapter.inspect(altered)
    assert "context" not in fake.diagnostic.calls and not any(type(c) is tuple for c in fake.diagnostic.calls)


@pytest.mark.parametrize("kind",["signature","short","crc","dimensions","apng","compressed_ancillary","chunk_length","trailing"])
def test_png_refusal_before_fake_decoder(kind):
    record,_,data=authored_manifest();pin=record["files"]["frames"][0]
    if kind=="signature":data=b"bad"+data[3:]
    elif kind=="short":data=data[:20]
    elif kind=="crc":data=data[:29]+b"xxxx"+data[33:]
    elif kind=="dimensions":data=data[:8]+chunk(b"IHDR",struct.pack(">IIBBBBB",1601,900,8,2,0,0,0))+data[33:]
    elif kind=="apng":data=data[:33]+chunk(b"acTL",b"12345678")+data[33:]
    elif kind=="compressed_ancillary":data=data[:33]+chunk(b"iCCP",b"zip")+data[33:]
    elif kind=="chunk_length":data=data[:8]+struct.pack(">I",0xffffffff)+data[12:]
    else:data+=b"x"
    with pytest.raises(native.Refusal):native.png_preflight(data,pin)


def test_decode_buffer_identity_once_and_none_usable_delegation():
    record,_,png=authored_manifest();fake=services();adapter=native.RGBAdapter(manifest(record),fake)
    pixels=adapter.decode_rgb(png,"image-0000")
    mask=adapter.usable_mask(pixels,"d"*64)
    assert fake.events[0][1] is png and fake.events[0][2]==".png"
    assert not pixels.flags.writeable and not mask.flags.writeable
    with pytest.raises(native.Refusal):adapter.decode_rgb(png,"image-0000")
    with pytest.raises(native.Refusal):adapter.usable_mask(pixels,"d"*64)


def test_fixed_authored_temp_reader_single_read_and_mutation(tmp_path):
    path=tmp_path/"authored-synthetic-input.bin";data=b"authored-synthetic"
    path.write_bytes(data)
    pin=dict(path=str(path),bytes=len(data),sha256=native.sha(data))
    assert native.read_fixed(pin,1024)==data
    path.write_bytes(data+b"x")
    with pytest.raises(native.Refusal):native.read_fixed(pin,1024)


def test_fixed_read_mapping_unknown_duplicate_cap_before_accessor():
    record,data,png=authored_manifest();accepted=manifest(record);calls=[]
    def read(pin,cap):calls.append(pin["path"]);return data if cap==1048576 else png
    reader=native.FixedReaders(accepted,SimpleNamespace(active=True),read)
    reader.read_payload("diagnostic","sealed_source",len(data)+1)
    with pytest.raises(native.Refusal):reader.read_payload("diagnostic","sealed_source",len(data)+1)
    with pytest.raises(native.Refusal):reader.read_payload("encoded","third",1)
    with pytest.raises(native.Refusal):reader.read_payload("encoded","image-0000",len(png)+2)
    assert len(calls)==1


def test_raw_framed_and_record_hashes_have_distinct_known_vector():
    raw=b"\x01\x02\x03"
    framed=(b"mc-photogrammetry-image-derived-session-frame-v1"+b"\x03"
            +b"\x01\x00\x00\x00\x00\x00\x00\x00"*3+b"|u1"+raw)
    assert hashlib.sha256(raw).hexdigest()=="039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
    assert hashlib.sha256(framed).hexdigest()=="76ec00c93ffcbbf6243f1bc7d356974f1cc0eaac5888e7806622065531bd2e1a"
    assert native.sha(native.canonical(dict(decoded_pixel_sha256=native.sha(framed)))) not in (native.sha(raw),native.sha(framed))


@pytest.mark.parametrize("orientation", [1,2,3,4,5,6,7,8])
def test_png_orientation_pin_is_verified_on_authored_bytes(orientation):
    record,_,png=authored_manifest();pin=record["files"]["frames"][0]
    tiff=b"II"+struct.pack("<HIH",42,8,1)+struct.pack("<HHIHHI",274,3,1,orientation,0,0)
    wire=png[:33]+chunk(b"eXIf",tiff)+png[33:]
    pin["exif_orientation"]=orientation
    native.png_preflight(wire,pin)
    pin["exif_orientation"]=orientation%8+1
    with pytest.raises(native.Refusal):native.png_preflight(wire,pin)


@pytest.mark.parametrize("payload",[b"",b"II"+struct.pack("<HIH",42,8,65),
    b"II"+struct.pack("<HIH",42,8,0)+struct.pack("<I",8)])
def test_exif_bad_bounds_refuse_before_decoder(payload):
    record,_,png=authored_manifest()
    with pytest.raises(native.Refusal):
        native.png_preflight(png[:33]+chunk(b"eXIf",payload)+png[33:],record["files"]["frames"][0])


def test_missing_grant_precedes_real_loader_and_input(monkeypatch):
    record,_,_=authored_manifest();accepted=manifest(record)
    with pytest.raises(native.Refusal):native.lazy_services(accepted,SimpleNamespace(active=False,job_enforced=True))
    with pytest.raises(native.Refusal):native.FixedReaders(accepted,SimpleNamespace(active=False),lambda *a:pytest.fail("reader activated"))


def reseal_diagnostic(record, diagnostic):
    bundle=diagnostic["source_region_masks"]
    ingress_hashes={row["frame_id"]:native.sha(native.canonical(row)) for row in bundle["source_context"]["ingress"]}
    for entry, inventory in zip(bundle["entries"],record["provenance"]["mask_inventory"]):
        entry["ingress_sha256"]=ingress_hashes.get(entry["endpoint"][0],entry["ingress_sha256"])
        inventory["ingress_record_sha256"]=entry["ingress_sha256"]
    bundle.pop("record_sha256");seal(bundle,"record_sha256")
    diagnostic.pop("record_sha256");seal(diagnostic,"record_sha256")
    wire=native.canonical(diagnostic,1048576)+b"\n"
    record["files"]["diagnostic"].update(bytes=len(wire),sha256=native.sha(wire))
    request=record["request_template"]
    request.update(diagnostic_length=len(wire),diagnostic_sha256=native.sha(wire))
    for obj in (request["bindings"],request["metadata"],record["provenance"]):
        obj.update(diagnostic_outer_sha256=diagnostic["record_sha256"],diagnostic_nested_sha256=bundle["record_sha256"])
    request["metadata"]["diagnostic_payload_sha256"]=native.sha(wire)
    record["provenance"]["mask_context_sha256"]=native.sha(native.canonical(bundle["source_context"]))
    return wire


@pytest.mark.parametrize("field",["byte_sha256","byte_length","decoded_shape","decoded_dtype","decoded_pixel_sha256",
    "transform_crop","transform_bool","frame_id","entry_ingress"])
def test_coherent_diagnostic_reseal_cannot_cross_independent_rgb_sources(field):
    record,data,_=authored_manifest();independent=copy.deepcopy(record["request_template"]["metadata"]["source_inventory"])
    diagnostic=json.loads(data);context=diagnostic["source_region_masks"]["source_context"]
    ingress=context["ingress"][0]
    if field in ("byte_sha256","decoded_pixel_sha256"):ingress[field]="f"*64
    elif field=="byte_length":ingress[field]+=1
    elif field=="decoded_shape":ingress[field]=[900,1599,3]
    elif field=="decoded_dtype":ingress[field]="<u2"
    elif field=="transform_crop":ingress["transform"]["crop_origin"]=[1,0]
    elif field=="transform_bool":ingress["transform"]["crop_origin"]=[False,0]
    elif field=="frame_id":ingress[field]="image-0001"
    diagnostic["ingress"]=copy.deepcopy(context["ingress"])
    wire=reseal_diagnostic(record,diagnostic)
    if field=="entry_ingress":
        diagnostic["source_region_masks"]["entries"][0]["ingress_sha256"]="f"*64
        bundle=diagnostic["source_region_masks"];bundle.pop("record_sha256");seal(bundle,"record_sha256")
        diagnostic.pop("record_sha256");seal(diagnostic,"record_sha256")
        wire=native.canonical(diagnostic,1048576)+b"\n"
        record["files"]["diagnostic"].update(bytes=len(wire),sha256=native.sha(wire))
        request=record["request_template"];request.update(diagnostic_length=len(wire),diagnostic_sha256=native.sha(wire))
        for obj in (request["bindings"],request["metadata"],record["provenance"]):
            obj.update(diagnostic_outer_sha256=diagnostic["record_sha256"],diagnostic_nested_sha256=bundle["record_sha256"])
        request["metadata"]["diagnostic_payload_sha256"]=native.sha(wire)
        record["provenance"]["mask_inventory"][0]["ingress_record_sha256"]="f"*64
    assert record["request_template"]["metadata"]["source_inventory"]==independent
    fake=services();adapter=native.DiagnosticAdapter(manifest(record),fake)
    with pytest.raises(native.Refusal):adapter.inspect(wire)
    assert not any(type(call) is tuple for call in fake.diagnostic.calls)


def test_verified_authored_source_executes_exact_retained_bytes_no_reopen_or_pyc(tmp_path,monkeypatch):
    from types import ModuleType
    path=tmp_path/"authored_module.py";data=b"AUTHORED_VALUE = 7\n"
    path.write_bytes(data);path.with_suffix(".pyc").write_bytes(b"AUTHORED_VALUE = 999")
    retained=path.read_bytes();loader=native.VerifiedSourceLoader("authored_module",str(path),retained,native.sha(retained))
    path.write_bytes(b"AUTHORED_VALUE = 99\n")
    seen=[];original_compile=compile
    def capture(source,*args,**kwargs):seen.append(source);return original_compile(source,*args,**kwargs)
    monkeypatch.setattr("builtins.compile",capture)
    monkeypatch.setattr("builtins.open",lambda *a,**k:pytest.fail("source/pyc reopen"))
    module=ModuleType("authored_module");loader.exec_module(module)
    assert module.AUTHORED_VALUE==7 and seen==[retained]
    assert not hasattr(loader,"get_code") and module.__file__==str(path)


@pytest.mark.parametrize("kind",["preloaded_same_origin","preloaded_mutated","wrong_pin","pyc","undeclared"])
def test_authored_source_registry_refuses_substitution_without_native_loader(tmp_path,kind):
    from types import ModuleType
    data=b"AUTHORED_VALUE = 7\n";name="authored_module";path=str(tmp_path/"authored_module.py")
    row=dict(module=name,path=path,sha256=native.sha(data),kind="repository")
    cache={}
    if kind.startswith("preloaded"):
        module=ModuleType(name);module.__file__=path;module.AUTHORED_VALUE=999;cache[name]=module
    elif kind=="wrong_pin":row["sha256"]="f"*64
    elif kind=="pyc":row["path"]=path+"c"
    if kind=="undeclared":
        finder=native.VerifiedSourceFinder({name:row},{name:data},cache)
        with pytest.raises(native.Refusal):finder.find_spec("authored_missing_service")
    else:
        with pytest.raises(native.Refusal):native.VerifiedSourceFinder({name:row},{name:data},cache)


def authored_transitive_source_session(tmp_path):
    from types import ModuleType
    sources={"authored_diagnostic":b'ENTRY = IMPORT("authored_entry")\n',
             "authored_entry":b'IMAGE = IMPORT("authored_image")\nSESSION = IMPORT("authored_session")\n',
             "authored_image":b'AUTHORED_VALUE = "image"\n',"authored_session":b'AUTHORED_VALUE = "session"\n'}
    records={name:dict(module=name,path=str(tmp_path/(name+".py")),sha256=native.sha(data),kind="repository")
             for name,data in sources.items()}
    cache={};executed=[];finder=native.VerifiedSourceFinder(records,sources,cache)
    def importer(name):
        spec=finder.find_spec(name);module=ModuleType(name);module.__spec__=spec;module.__loader__=spec.loader
        module.__dict__["IMPORT"]=lambda child:finder.load(child,importer)
        cache[name]=module;executed.append(name)
        spec.loader.exec_module(module)
        return module
    return finder,cache,executed,importer


def test_authored_transitive_source_reuse_requires_same_session_exact_objects(tmp_path):
    finder,cache,executed,importer=authored_transitive_source_session(tmp_path)
    diagnostic=finder.load("authored_diagnostic",importer)
    image=finder.load("authored_image",importer);session=finder.load("authored_session",importer)
    assert diagnostic.ENTRY.IMAGE is image and diagnostic.ENTRY.SESSION is session
    assert executed==["authored_diagnostic","authored_entry","authored_image","authored_session"]
    assert finder.load("authored_diagnostic",lambda *args:pytest.fail("verified source executed twice")) is diagnostic
    assert finder.session["authored_image"][0] is image and finder.session["authored_session"][0] is session
    assert finder.find_spec("authored_image") is image.__spec__


@pytest.mark.parametrize("kind",["module","spec","loader","module_loader","origin","unattested"])
def test_same_session_substitution_or_unattested_cache_entry_refuses(tmp_path,kind):
    from types import ModuleType
    finder,cache,_,importer=authored_transitive_source_session(tmp_path)
    module=finder.load("authored_image",importer)
    if kind=="module":
        replacement=ModuleType(module.__name__);replacement.__file__=module.__file__;replacement.__spec__=module.__spec__
        cache["authored_image"]=replacement
    elif kind=="spec":module.__spec__=SimpleNamespace(loader=module.__spec__.loader,origin=module.__spec__.origin)
    elif kind=="loader":module.__spec__.loader=object()
    elif kind=="module_loader":module.__loader__=object()
    elif kind=="origin":module.__file__=str(tmp_path/"authored_other.py")
    else:
        del finder.session["authored_image"]
    with pytest.raises(native.Refusal):finder.load("authored_image",lambda *args:pytest.fail("untrusted reimport"))


def test_failed_authored_execution_earns_no_session_attestation(tmp_path):
    from types import ModuleType
    name="authored_failing";data=b'raise RuntimeError("authored failure")\n'
    row=dict(module=name,path=str(tmp_path/(name+".py")),sha256=native.sha(data),kind="repository")
    cache={};finder=native.VerifiedSourceFinder({name:row},{name:data},cache)
    spec=finder.find_spec(name);module=ModuleType(name);module.__spec__=spec;cache[name]=module
    with pytest.raises(RuntimeError,match="authored failure"):spec.loader.exec_module(module)
    assert name not in finder.session
    with pytest.raises(native.Refusal,match="unattested"):finder.load(name,lambda *args:pytest.fail("failed source reused"))


def test_bootstrap_exemption_binds_exact_baseline_object(tmp_path):
    from types import ModuleType
    name="scripts.evaluation.spatial_context_native_io";data=b'AUTHORED_BOOTSTRAP = True\n'
    row=dict(module=name,path=str(tmp_path/"authored_bootstrap.py"),sha256=native.sha(data),kind="repository")
    module=ModuleType(name);module.__file__=row["path"];cache={name:module}
    finder=native.VerifiedSourceFinder({name:row},{name:data},cache,(name,))
    assert finder.load(name,lambda *args:pytest.fail("bootstrap re-execution")) is module
    replacement=ModuleType(name);replacement.__file__=row["path"];cache[name]=replacement
    with pytest.raises(native.Refusal,match="bootstrap_identity"):finder.load(name,lambda *args:pytest.fail("replacement bootstrap"))
