"""Evaluation-only fixed campaign contract for BE signature replay diagnostics."""
from dataclasses import dataclass
import ctypes, hashlib, json, os, subprocess, tempfile
from pathlib import Path
from ctypes import wintypes
from mc_photogrammetry import evaluation_signature_sampling as sampling

SCHEMA="mc-photogrammetry-signature-replay-campaign-v1"
BE_FILE_SHA256="A33F269242232D8530F1B8E5B405B5FE437857D657B760B37EE0F6B155B5C597".lower()
SOURCE_HASHES=("36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a","522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11","d119381cf3717d6e300c7da73701c3d0f8358a5b7b9d6091697bee2c1e2c1934")
PRODUCER_HASHES=("a3e71af2fbd043b58e21b7f152a6674b223217b6984aaa6093ad8b5fbacb9ba1","471e065b02cfb9a6bb27200d6ef68743a77cb555ba0ad195073545fe89343177")
CAPS=(12582912,50331648,67108864,1256194072)
MEMORY_LIMIT_BYTES=8*1024**3
DEADLINE_SECONDS=600
IDENTITIES=(('image-0000', 0, 0, '72a772bee4f3ae36ed90ade77a5a8087e66d1effe656aad0c7ada76ec023fd9e'),

 ('image-0000', 0, 1, 'e1b409974f11f04966d61819dbbcb8cc75f75f9bf61eefb244968c4aa7d9223d'),

 ('image-0000', 0, 2, 'f397421e8f1ecdde961fb65869fa65470d41279bb19d191782e6ea77baf584f2'),

 ('image-0000', 0, 3, 'cf4e35b53885f9a03beb1fe8cdb23988d815496c4d8f4961075dbbb9fb8e344d'),

 ('image-0000', 0, 4, 'a5a86b11c4badf76274185919d3ea064f2a6c6587dce42b3477b6890e494c130'),

 ('image-0000', 0, 5, '02e64b05ece06f44599eaf4e6119e26cc9664cef3f2b5bb4a7bf649349c83aa5'),

 ('image-0000', 0, 6, '0b96e2f8214b917466cee441750af0bee0a6a5150ffa1d38bd2492c4f05308ca'),

 ('image-0000', 0, 7, 'd6d57a7a381e68d0247b88a35446e6b96dd8b00102f29d65b8ee4b43d6f0454e'),

 ('image-0000', 1, 0, 'ef4aa3738b533b792cc3678001fed689087587f87fe25285b47ac7e40cd510c6'),

 ('image-0000', 1, 1, 'c288cf8534354b8790f27f1190f99ef59752a183f47223bd9654ab5d83c96673'),

 ('image-0000', 1, 2, '03bf635ac48e8d92f4970db5b11855157261cd0a43065601fb1d50d813d3b10e'),

 ('image-0000', 1, 3, '3ecc9040f826c9c7968adb4a3d2a4bd28a7f8f161ca2ce256041ab89c4d6349a'),

 ('image-0000', 1, 4, '26d3657288482e49a0944d73c8101b4541077408d00b41f0ceb980232267d6ac'),

 ('image-0000', 2, 0, '57de496c0385dc65c2a7d08be3a8d4b294292d1f650384a9e74ba21a504d3d12'),

 ('image-0001', 3, 0, 'dd5eb3e7f03782bebe93fd4f6d037d60e49c96cbfc37b5f79376c02b78c99fb3'),

 ('image-0001', 3, 1, 'd736a63d063c8c97c81a1ecbcf36918f0cc531afc5ce1156186c6a29ed95a524'),

 ('image-0001', 3, 2, '8fb35e9785651b2d853b777a504fd5728c750e88695b67f5916e3f93158b8aaa'),

 ('image-0001', 3, 3, 'e2247f7208ede1dc00f2c892a4f74d389c8bb6d8f74c26c6302f45ec76022feb'),

 ('image-0001', 3, 4, 'a5416975eb5958dbb8cbf82fb33ed6d169f6ea0281b043becf48bf186ff8de90'),

 ('image-0001', 3, 5, '5f57916372fef47b5a1e6dcaef994e48c58fb7a01f07ee09ea6b393d612ec04a'),

 ('image-0001', 3, 6, '9f55ca92332679552e3dbca1eede51870f43214d70d6aecaf96dd0d99d5e1895'),

 ('image-0001', 4, 0, 'cd8bc303b5630c5e8cc9ff4e16ca0d3476863c85956c7093156c12e83374487d'),

 ('image-0001', 4, 1, '1a02cec4c3deaee9ecedf0177df11deb916311327db0130ef715a2b420b83ed2'),

 ('image-0001', 4, 2, '7bec1dbd4911a9f6c6892a1e07bf0cac7a24ef04cee40be1e42d281763be461d'),

 ('image-0001', 4, 3, '06078db6851bcb71b9c45cb744851d03290f05356f2f3226710d6eceba92aa3c'),

 ('image-0001', 4, 4, '7063e78724b4d9e470110a2a6bde6d323bd2e257d67c40af3fad9947ec3a7051'),

 ('image-0001', 4, 5, 'a174ff95f0eb03e45116a7d1231c11707886008c7ef71eece4aaf6b91897cc85'),

 ('image-0002', 5, 0, '6a13bf1fbf10b3905533e61f0e3a45e471335d543fe8465e7f177e81f56fe1f9'),

 ('image-0002', 5, 1, 'ba7a45f6cf3bf02df068ebe3ecd97b0b9dccfe1e63b8e7a2567ce00e83ade7bf'),

 ('image-0002', 6, 0, 'b16709f68e09915b2e873abc53bafe3adcc251a730cf38eb58e98e797bfc547f'),

 ('image-0003', 7, 0, '8a6466c2e81c1837f606cf81e32e12ce55c4f00d680e245a75b537757b61bb06'),

 ('image-0003', 7, 1, '5f1fcc5cacc13a7aff6cf0cac183ac065e63176e5c4f5db6e69f87d5e9bd765b'),

 ('image-0003', 7, 2, 'd844cd6b17abec59f4018b4cdb9378af0ab874b834da848f46281a9b8af9de14'),

 ('image-0003', 7, 3, 'ecc615b0dd7cdcb59e097a4823b8fb0dc8897192405c2c6bd65aa5a5fcb93e5e'),

 ('image-0003', 7, 4, '2cfe6582e8cbd8bcc12a9f8ee302b20876e7be2ee3a80e880046d60b46ab687c'),

 ('image-0003', 7, 5, '206a9c69588bd4e8a319cd96d889edfa41ae007d11af7d963eef63e84f0883a1'),

 ('image-0003', 8, 0, '4b10a850f7cccf6c4164c19655e37ec5e23458b88451385d4c3a8032d94f78b8'))



def _sha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True,slots=True)
class CampaignManifest:
 schema:str=SCHEMA; be_file_sha256:str=BE_FILE_SHA256; source_hashes:tuple=SOURCE_HASHES; producer_hashes:tuple=PRODUCER_HASHES; caps:tuple=CAPS; memory_limit_bytes:int=MEMORY_LIMIT_BYTES; deadline_seconds:int=DEADLINE_SECONDS; identities:tuple=IDENTITIES; batches:tuple=tuple(tuple(range(i,min(i+4,37))) for i in range(0,37,4)); manifest_sha256:str=""
 def __post_init__(self):
  if self.schema!=SCHEMA or self.be_file_sha256!=BE_FILE_SHA256 or self.source_hashes!=SOURCE_HASHES or self.producer_hashes!=PRODUCER_HASHES or self.identities!=IDENTITIES or self.batches!=tuple(tuple(range(i,min(i+4,37))) for i in range(0,37,4)) or len(self.batches)!=10 or tuple(x for b in self.batches for x in b)!=tuple(range(37)) or self.caps!=CAPS or self.memory_limit_bytes!=MEMORY_LIMIT_BYTES or self.deadline_seconds!=DEADLINE_SECONDS: raise ValueError("campaign manifest mismatch")
  d=_sha(self.to_record(False));
  if self.manifest_sha256 not in ("",d): raise ValueError("campaign manifest digest mismatch")
  object.__setattr__(self,"manifest_sha256",d)
 def to_record(self,digest=True):
  x={"schema":self.schema,"be_file_sha256":self.be_file_sha256,"source_hashes":list(self.source_hashes),"producer_hashes":list(self.producer_hashes),"caps":list(self.caps),"memory_limit_bytes":self.memory_limit_bytes,"deadline_seconds":self.deadline_seconds,"identities":[list(v) for v in self.identities],"batches":[list(v) for v in self.batches]}
  if digest:x["manifest_sha256"]=self.manifest_sha256
  return x

@dataclass(frozen=True,slots=True)
class BatchRow:
 identity:tuple; result:object
 def __post_init__(self):
  if self.identity not in IDENTITIES or not isinstance(self.result,(sampling.SignatureSamplingBindingV1,sampling.SignatureSamplingReplayRefusalV1)) or self.result.signature_sha256!=self.identity[3]: raise ValueError("batch row mismatch")
  if isinstance(self.result,sampling.SignatureSamplingBindingV1) and (self.result.frame_id,self.result.canonical_reading_index,self.result.alternative_index)!=self.identity[:3]: raise ValueError("batch row identity mismatch")
  object.__setattr__(self,"result",type(self.result)(**{k:getattr(self.result,k) for k in self.result.__dataclass_fields__}))
 def to_record(self): return {"identity":list(self.identity),"result_type":type(self.result).__name__,"result":self.result.to_record()}

@dataclass(frozen=True,slots=True)
class BatchArtifact:
 manifest_sha256:str; batch_index:int; status:str; reason:str; rows:tuple; baseline_rss_bytes:int|None; peak_rss_bytes:int|None; post_gc_rss_bytes:int|None; elapsed_seconds:float; artifact_sha256:str=""
 def __post_init__(self):
  expected=CampaignManifest(); ids=expected.batches[self.batch_index] if type(self.batch_index) is int and 0<=self.batch_index<10 else ()
  measured=(self.baseline_rss_bytes,self.peak_rss_bytes,self.post_gc_rss_bytes)
  valid_rss=all(type(v) is int and v>0 for v in measured) and self.peak_rss_bytes>=max(self.baseline_rss_bytes,self.post_gc_rss_bytes)
  elapsed_ok=type(self.elapsed_seconds) in (int,float) and self.elapsed_seconds>=0
  success_ok=self.status=="complete" and elapsed_ok and not self.reason and len(self.rows)==len(ids) and valid_rss and self.elapsed_seconds<=DEADLINE_SECONDS
  refusal_ok=self.status=="refused" and elapsed_ok and bool(self.reason) and not self.rows and (all(v is None for v in measured) or valid_rss)
  if self.manifest_sha256!=expected.manifest_sha256 or not ids or not (success_ok or refusal_ok): raise ValueError("batch artifact malformed")
  object.__setattr__(self,"rows",tuple(BatchRow(r.identity,r.result) for r in self.rows))
  if self.status=="complete" and tuple(r.identity for r in self.rows)!=tuple(expected.identities[i] for i in ids): raise ValueError("batch identities mismatch")
  d=_sha(self.to_record(False));
  if self.artifact_sha256 not in ("",d): raise ValueError("batch artifact digest mismatch")
  object.__setattr__(self,"artifact_sha256",d)
 def to_record(self,digest=True):
  x={"manifest_sha256":self.manifest_sha256,"batch_index":self.batch_index,"status":self.status,"reason":self.reason,"rows":[r.to_record() for r in self.rows],"baseline_rss_bytes":self.baseline_rss_bytes,"peak_rss_bytes":self.peak_rss_bytes,"post_gc_rss_bytes":self.post_gc_rss_bytes,"elapsed_seconds":self.elapsed_seconds}
  if digest:x["artifact_sha256"]=self.artifact_sha256
  return x

@dataclass(frozen=True,slots=True)
class CampaignMerge:
 execution_complete:bool; all_signatures_accepted:bool; rows:tuple; reason:str; merge_sha256:str=""
 def __post_init__(self):
  d=_sha({"execution_complete":self.execution_complete,"all_signatures_accepted":self.all_signatures_accepted,"rows":[r.to_record() for r in self.rows],"reason":self.reason})
  if self.merge_sha256 not in ("",d): raise ValueError("merge digest mismatch")
  object.__setattr__(self,"merge_sha256",d)

def select_batch(manifest, batch_index, regenerated):
 manifest=CampaignManifest(**{k:getattr(manifest,k) for k in manifest.__dataclass_fields__})
 actual=tuple((row.frame_id,row.canonical_index,row.alternative_index,row.signature_sha256)
              for row in regenerated)
 if actual!=manifest.identities: raise ValueError("regenerated signature universe mismatch")
 if type(batch_index) is not int or not 0<=batch_index<10: raise ValueError("batch index is invalid")
 return tuple(regenerated[i] for i in manifest.batches[batch_index])

def merge_campaign(manifest,artifacts):
 manifest=CampaignManifest(**{k:getattr(manifest,k) for k in manifest.__dataclass_fields__})
 if len(artifacts)!=10: return CampaignMerge(False,False,(),"missing_batch")
 try: rows=[]
 except Exception: raise
 seen=set()
 for raw in artifacts:
  artifact=BatchArtifact(**{k:getattr(raw,k) for k in raw.__dataclass_fields__})
  if artifact.batch_index in seen:return CampaignMerge(False,False,(),"duplicate_batch")
  seen.add(artifact.batch_index)
  if artifact.status!="complete":return CampaignMerge(False,False,(),f"batch_{artifact.batch_index}_{artifact.reason}")
  rows.extend(artifact.rows)
 if seen!=set(range(10)) or tuple(r.identity for r in rows)!=manifest.identities:return CampaignMerge(False,False,(),"identity_mismatch")
 accepted=all(isinstance(r.result,sampling.SignatureSamplingBindingV1) for r in rows)
 return CampaignMerge(True,accepted,tuple(rows),"" if accepted else "signature_refusal")

def verify_sources(paths):
 expected=(BE_FILE_SHA256,)+SOURCE_HASHES+PRODUCER_HASHES
 actual=tuple(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths)
 if actual!=expected: raise ValueError("campaign source hash mismatch")

def atomic_write_artifact(path,artifact):
 artifact=BatchArtifact(**{k:getattr(artifact,k) for k in artifact.__dataclass_fields__})
 target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix=target.name+".",suffix=".tmp",dir=target.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(artifact.to_record(),stream,sort_keys=True,indent=2); stream.write("\n")
  os.replace(name,target)
 except BaseException:
  try: os.unlink(name)
  except OSError: pass
  raise

def reviewed_worker_command(python,worker,batch_index,manifest,*,be,truth,jar,
                            palette,capture,project,output):
 manifest=CampaignManifest(**{k:getattr(manifest,k) for k in manifest.__dataclass_fields__})
 if type(batch_index) is not int or not 0<=batch_index<10: raise ValueError("batch index is invalid")
 return (str(python),str(worker),"--manifest-sha256",manifest.manifest_sha256,
         "--batch-index",str(batch_index),"--memory-limit",str(MEMORY_LIMIT_BYTES),
         "--deadline",str(DEADLINE_SECONDS),"--be",str(be),"--truth",str(truth),
         "--jar",str(jar),"--palette",str(palette),"--capture",str(capture),
         "--project",str(project),"--output",str(output))

def launch_reviewed_worker(command, *, verified_paths, artifact_path, batch_index):
 """Launch once under the frozen Windows Job cap and deadline; never retries."""
 manifest=CampaignManifest(); verify_sources(verified_paths)
 if os.name!="nt": raise RuntimeError("reviewed campaign runner requires Windows")
 class Basic(ctypes.Structure):
  _fields_=[("a",ctypes.c_longlong),("b",ctypes.c_longlong),("flags",wintypes.DWORD),("c",ctypes.c_size_t),("d",ctypes.c_size_t),("e",wintypes.DWORD),("f",ctypes.c_size_t),("g",wintypes.DWORD),("h",wintypes.DWORD)]
 class Io(ctypes.Structure): _fields_=[(name,ctypes.c_ulonglong) for name in ("a","b","c","d","e","f")]
 class Extended(ctypes.Structure): _fields_=[("basic",Basic),("io",Io),("process_limit",ctypes.c_size_t),("job_limit",ctypes.c_size_t),("peak_process",ctypes.c_size_t),("peak_job",ctypes.c_size_t)]
 kernel=ctypes.WinDLL("kernel32",use_last_error=True)
 kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]; kernel.CreateJobObjectW.restype=wintypes.HANDLE
 kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]; kernel.SetInformationJobObject.restype=wintypes.BOOL
 kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]; kernel.AssignProcessToJobObject.restype=wintypes.BOOL
 job=kernel.CreateJobObjectW(None,None); info=Extended(); info.basic.flags=0x100; info.process_limit=MEMORY_LIMIT_BYTES
 if not kernel.SetInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info)): raise ctypes.WinError(ctypes.get_last_error())
 started=__import__("time").perf_counter()
 process=subprocess.Popen(tuple(command),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=0x4)
 if not kernel.AssignProcessToJobObject(job,wintypes.HANDLE(process._handle)):
  process.kill(); raise ctypes.WinError(ctypes.get_last_error())
 ntdll=ctypes.WinDLL("ntdll",use_last_error=True)
 if ntdll.NtResumeProcess(wintypes.HANDLE(process._handle))!=0:
  process.kill(); raise RuntimeError("contained worker could not be resumed")
 try:
  stdout,stderr=process.communicate(timeout=DEADLINE_SECONDS)
  elapsed=__import__("time").perf_counter()-started
  if process.returncode:
   refusal=BatchArtifact(manifest.manifest_sha256,batch_index,"refused",f"worker_exit_{process.returncode}",(),None,None,None,elapsed)
   atomic_write_artifact(artifact_path,refusal)
  return stdout,stderr,process.returncode
 except subprocess.TimeoutExpired:
  process.kill(); stdout,stderr=process.communicate()
  refusal=BatchArtifact(manifest.manifest_sha256,batch_index,"refused","deadline_exhausted",(),None,None,None,float(DEADLINE_SECONDS))
  atomic_write_artifact(artifact_path,refusal)
  return stdout,stderr,"deadline_exhausted"

