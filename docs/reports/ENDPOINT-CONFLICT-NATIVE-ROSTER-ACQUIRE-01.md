# Endpoint conflict native roster acquisition ? terminal failure

The exact authorized command ran once and failed before feature acquisition: **exit1, `TERMINAL: worker_failure:1`**, worker log `Stop:decoded_ingress_mismatch`. Neither `ready` nor `inconclusive` was produced. No retry, code/test edit, tuning, alternate input, subsequent decode or physical-association inference.

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluation/acquire_endpoint_conflict_native_roster.py
```

Preflight verified50 code/source/artifact pins, the exact lead authorization hash and both PNG hashes before any decode; output directory was absent. After failure the same50 pins and both input hashes remain unchanged.

The captured stack ends at `bind_frames` driverline173, called by `_worker`line357. The guard checks decoded uint8 dtype, exact pinned native shape and exact C-order decoded pixel SHA256. The log does not retain which frame/field differed or actual decoded scalar values. At least one and at most two native decodes occurred; no `bind_groups`/accepted-helper acquisition/feature detection was reached. This result alone does not identify a decoder, ingress, or driver defect. No further source investigation or image decode was performed.

Observed resources: wall0.766000s, descendant-tree CPU0.671875s; sampled aggregate process-highwater RSS114573312B; sampled sum process-pagefile-highwater591073280B; output610B. The accepted driver's180s/2GiB committed-memory Job Object/8MiB artifact enforcement was active and did not report a budget exceedance; operator emergency timeout false. Operator measurements use read-only10ms descendant snapshots and conservative per-process peaks; they are not exact job commit statistics and may miss short-lived children.

Output directory `exports/diagnostics/endpoint-conflict-native-roster-2026-10-09` contains only `worker.log` (610B), SHA256 `211AF371BB4BD091EEB516ED5C2235E02D6B376167B086F064096D487B53E855`. `roster.json` and `completion.json` are absent. The retained log is failure evidence, not a completed roster. No geometry/camera/projective fit, association selection, checker, truth, scoring, accuracy or `.litematic` output.

Receipt SHA256:

- Preflight `1F4EECCE65537BA11E5740A51FFF1F6885DEA8DA0EFC47E9DAAED5BA74676248`
- Operator run `B97C815B2BC4223E51437623196769BCDB44A104C887589EFE4D427626AE6811`
- Command stdout `24985E004DB42D80E799592AE2F1C6DE4BD9B03D39EFAAFFB211275459FD5A31`
- Evidence `D25F6B74F55DD1621E9AC23E5D1EF5E3DB8D17FDFDAFB3C11DEA844ED6ABE137`

Driver remains `1B7DB2DEDF159D9007E745D569E08D14109979116F38012AEB626671EA1D12CA`; test remains `28F2994FB25AB72F94BDA8D473B7F7A9AEE60356182E2367DF5AA740DB65C73B`. Previous accepted materialization and synthetic records remain sealed. No tests run in this acquisition task.

STOP for independent lead review. The sole authorized acquisition is consumed. A new source investigation/amended task is required to resolve the unobserved dtype/shape/pixel-digest discrepancy; no retry is implied.

