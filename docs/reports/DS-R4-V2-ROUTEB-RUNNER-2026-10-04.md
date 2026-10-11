# Route B V2 diagnostic runner - 2026-10-04

Task: `DS-R4-V2-ROUTEB-RUNNER`, interface repair, and finalization, then the lead's
`DS-R4-V2-ROUTEB-RUNNER-REMEDIATION` repair of the Luna HOLD findings. This is an
internal diagnostic utility; it does not change normal reconstruction or claim accuracy.

## Runner contract

`scripts/evaluation/run_route_b_diagnostic.py` accepts ordinary `.png` paths (`--image`,
repeatable and ordered) and one output root (`--root`). It fixes graph settings to
production defaults and has no caller-supplied image hash, universe pin, settings override,
or record-mode CLI; the injectable `api` argument exists only for tests.

The runner executes entry, lattice, signature, and graph-attribution once each. Entry capture
attaches its own ingress provenance; the single signature pass emits source-tap bindings; the
graph attribution call forwards those bindings and the V2 support sink. The sink must fire
exactly once on solver abstention. Before writing `record.json`, the runner validates the
complete V2 record, enforces exactly 1,928 ordered admitted rows, recomputes the sealed
whole-universe identity, and verifies the serialized record by replay. Refusal produces no
complete record.

## Remediation (DS-R4-V2-ROUTEB-RUNNER-REMEDIATION)

The four Luna HOLD findings were repaired without touching any production source:

1. **Single-read ingress identity.** `resolve_images` no longer reads or hashes input bytes;
   it performs only cardinality/extension/existence checks. Each recorded source identity is
   the entry's own `FrameIngressProvenanceV1.byte_sha256`, computed by the entry from the
   exact encoded buffer it decoded. `_ingress_bindings` binds the ordered pins back to the
   supplied paths positionally (pin count, unique frame ids, exact ordered source labels) and
   refuses a changed, duplicated, reordered or mismatched binding before any record is written.
2. **Exactly three inputs.** `ALLOWED_FRAME_COUNTS = (3,)`; the frozen three-view T2 fixture
   is the only accepted inventory. A fourth input refuses on `png_count` before entry.
3. **Failure atomicity.** The entry's schema-2 project is written into a fresh staging child
   of the validated output root and linked into place only after the full chain and replay
   gate succeed. Any refusal cleans up the exact runner-owned staged `route_b.mcscan`/`.tmp`
   names and the exact empty staging directory with nonrecursive `unlink`/`rmdir`; no tree
   walk or recursive deletion exists. `.partial` files are removed in a `finally` block on a
   failed temporary write.
4. **Output-root safety / no-clobber.** Every existing component from the absolute output
   root through its filesystem anchor is inspected with read-only `os.lstat`; symlink/reparse
   components refuse, `FileNotFoundError` means absent, and all other inspection errors
   (including `NotADirectoryError`) refuse closed. `record.json` and the staged project
   publish with `os.link`, which fails if a target appears after the pre-check.
5. **Capture-root and serialized ingress integrity.** Before publication, entry pins must
   equal `capture.root.ingress` positionally; outer serialized `ingress` and `inputs` rows are
   derived from and checked against the same ordered sequence. `validate_record` rebuilds and
   replays the capture, then repeats these cross-bindings. Synthetic controls refuse foreign/
   permuted root ingress and tampered outer `ingress`/`inputs`, even when the outer artifact
   digest is recomputed.
6. **Partial-file ownership.** Cleanup occurs only after this invocation's exclusive `xb`
   open succeeds. A pre-existing `.partial` collision refuses without changing its bytes;
   failures after successful creation remove only the owned partial.

## Verification

- `.venv\\Scripts\\python.exe -m py_compile scripts/evaluation/run_route_b_diagnostic.py tests/test_route_b_diagnostic_runner.py`: exit 0.
- `.venv\\Scripts\\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -q`: exit 0; 34 passed after integrity repair.
- Synthetic controls cover single-read ordered ingress; mismatch, duplicate and permutation refusal; foreign/permuted capture-root refusal; recomputed-digest tampering of outer ingress/inputs; fourth-image refusal; side-effect rollback; write-failure cleanup; pre-existing partial preservation; no-clobber target races; ancestor reparse, permission failure and file-parent `NotADirectoryError`; and nonrecursive cleanup.
- Runner SHA-256: `331D0557EA2AEBA436BAA25D364386F63F90C9C5383066F5CD55CCF2F4CAE16A` (48,505 bytes).
- Test SHA-256: `7D7AFD62AF0BC6C16A7DDC310C2E85D786595AB55D08F2835D79001D28BDB618` (45,302 bytes).
- Production modules remain unchanged by this task. The test pins the accepted entry/signature/graph sources and checks real graph-sink output neutrality using synthetic inputs.

No real screenshot set was supplied to the runner, no image payload was inspected, no runner
invocation, capture/R1, or accuracy result was produced, and no Minecraft capture occurred.
Reparse detection is read-only metadata inspection; the symlink control uses a mocked tag and
creates no native symlink/reparse point.

## Content identity

No independent preregistered digest exists for this exact 1,928-row universe. Its content
identity is the V2 record's sealed and replay-verified whole-universe digest plus the exact
expected row count. This supports integrity and later comparison, but does not prove the
candidate inventory matches an external expected universe.

## Review disposition

Luna accepted the synthetic integrity contract on 2026-10-04 at the exact runner, test and
report hashes recorded in `AGENT_COMMS.md`: positional cross-binding, tamper refusal, partial
ownership, ancestor reparse/error handling and failure cleanup. This accepts only the internal
runner contract. It does not authorize a real-image run; that requires a separate lead
assignment. No image or capture result is claimed here.
