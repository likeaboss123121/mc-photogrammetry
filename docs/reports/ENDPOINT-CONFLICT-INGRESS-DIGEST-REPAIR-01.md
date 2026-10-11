# Endpoint conflict ingress digest repair

Completed the bounded driver/test digest repair. `bind_frames` now validates `decoded_pixel_sha256` with the actual production `image_derived_session.image_sha256`, which hashes the domain/rank/dimensions/dtype and C-order pixels. The upstream module is now a frozen source pin. The accepted roster helper still uses its separate raw RGB-array digest; native RegionMask source/raw-bool and header+packed digest contracts remain unchanged. No production/helper source, cap or acceptance-gate change.

Exactly the task-card output and authorization paths changed to `exports/diagnostics/endpoint-conflict-native-roster-2026-10-09-repair-01` and `agent swarm instantiation template/runs/RESULT-LEAD-ENDPOINT-CONFLICT-NATIVE-ROSTER-AUTH-REPAIR-01-2026-10-09.json`. Both remain absent. The old consumed acquisition directory still contains only worker.log with SHA256 `211AF371BB4BD091EEB516ED5C2235E02D6B376167B086F064096D487B53E855`; all prior failure/source/review/receipt pins remain unchanged.

67 starting pins verified (driver/test plus65 read-only); all final frozen/read-only pins match. In-memory reversal of the allowed driver edits reproduces its starting SHA256 exactly. The synthetic fixture now uses the production digest. Four added regression cases verify valid native decode through genuine production `build_frame_ingress_provenance`, preserved raw helper identity, dtype and shape refusals, and rejection when raw bytes SHA is substituted for a production ingress digest. Existing pixel mismatch refusal remains.

Frozen before one prescribed command:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -o addopts= -q -x --tb=short tests/test_endpoint_conflict_native_roster.py
```

**51 passed,0 failed/errors/unreached in1.79s**, operator wall2.063000s,exit0. No retry. Only the exact synthetic module executed; its existing synthetic decoder/helper/Windows child controls remain included. No ordinary PNG hash/read/decode, standalone acquisition/detector/helper/supervisor or production command, geometry/fitting/scoring/truth/checker/camera or `.litematic`. AST/compile-to-memory, whitespace and strict receipt checks passed.

SHA256:

- Starting driver `1B7DB2DEDF159D9007E745D569E08D14109979116F38012AEB626671EA1D12CA`
- Final driver `6CB854E1800779514EFE26E6ED3EA83A5C7A0975743380E8969D9BC46C6CC14C`
- Starting test `28F2994FB25AB72F94BDA8D473B7F7A9AEE60356182E2367DF5AA740DB65C73B`
- Final test `6D747FBD580853D435E4BB31BB27ADEC73C30A703F5FF9350E224BFF19BFB8FF`
- Upstream image_derived_session.py `E48414BF9EEA6A7DA7A7CC7D8C19865E0494953AC8796D9F3CD2E57E81B6248E`
- Preflight `20C296464DA26DF1B61A19104CB0A9128113223276CD858813E7CF56201EEDE5`
- Freeze `37C45A18E6E5E0A7492D42227C7AEBD8A3469FEF86DD7B1F7FEE9AA4715B7CB6`
- Stdout `E2F63BFC9670F49BE31CDF2D21832E8B3AC65B23DB83076D5DCC1C4A99543DA1`
- Test-run `DBE60335ACEACC0440420F1EA823FDAE7BD53A2BF561A872493B46E37220572B`
- Evidence `06C21702011D56D6C083C95E8F62CEABDA762F9364C0188B96A3F8DC14344603`

The source contract defect is demonstrated; the consumed worker log still does not establish which actual frame/field failed. No ordinary image validation of this repair occurred. Source-bound native masks and physical correctness are not reclassified by synthetic success. STOP for independent Astra/lead review before any new image access or authorization file.

